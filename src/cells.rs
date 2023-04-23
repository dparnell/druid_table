use std::marker::PhantomData;

use druid::widget::prelude::*;
use druid::{
    BoxConstraints, Data, Env, Event, EventCtx, InternalLifeCycle, KbKey, LayoutCtx, LifeCycle,
    LifeCycleCtx, PaintCtx, Point, Rect, Size, UpdateCtx, Widget, WidgetPod,
};

use crate::axis_measure::{AxisPair, LogIdx, TableAxis, VisOffset};
use crate::cells::Editing::Inactive;
use crate::columns::{CellCtx, DisplayFactory};
use crate::config::ResolvedTableConfig;
use crate::data::{IndexedData, Remapper};
use crate::ensured_pool::EnsuredPool;
use crate::render_ext::RenderContextExt;
use crate::selection::{CellDemap, CellRect, SingleCell, TableSelection};
use crate::table::{PixelRange, TableState};
use crate::{bindable_self_body, Remap, RemapSpec};

use std::ops::Deref;
use std::sync::Arc;
use crate::bindings::BindableAccess;

pub trait CellsDelegate<TableData: IndexedData>:
    DisplayFactory<TableData::Item> + Remapper<TableData>
{
    fn data_fields(&self, data: &TableData) -> usize;
}

impl<TableData: IndexedData> CellsDelegate<TableData> for Arc<dyn CellsDelegate<TableData>> {
    fn data_fields(&self, data: &TableData) -> usize {
        self.as_ref().data_fields(data)
    }
}

impl<TableData: IndexedData> DisplayFactory<TableData::Item> for Arc<dyn CellsDelegate<TableData>> {
    fn make_display(
        &self,
        cell: &CellCtx,
    ) -> Option<Box<dyn Widget<<TableData as IndexedData>::Item>>> {
        self.deref().make_display(cell)
    }

    fn make_editor(
        &self,
        ctx: &CellCtx,
    ) -> Option<Box<dyn Widget<<TableData as IndexedData>::Item>>> {
        self.deref().make_editor(ctx)
    }
}

impl<TableData: IndexedData> Remapper<TableData> for Arc<dyn CellsDelegate<TableData>> {
    fn sort_fixed(&self, idx: usize) -> bool {
        self.deref().sort_fixed(idx)
    }

    fn initial_spec(&self) -> RemapSpec {
        self.deref().initial_spec()
    }

    fn remap_from_records(&self, table_data: &TableData, remap_spec: &RemapSpec) -> Remap {
        self.deref().remap_from_records(table_data, remap_spec)
    }
}

enum Editing<RowData> {
    Inactive,
    Cell {
        single_cell: SingleCell,
        child: WidgetPod<RowData, Box<dyn Widget<RowData>>>,
    },
}

impl<RowData: Data> Editing<RowData> {
    fn is_active(&self) -> bool {
        match self {
            Inactive => false,
            _ => true,
        }
    }

    fn is_editing(&self, cell: &SingleCell) -> bool {
        match self {
            Editing::Cell { single_cell, .. } => single_cell.vis.eq(&cell.vis),
            _ => false,
        }
    }

    fn handle_event<TableData: IndexedData<Item = RowData>>(
        &mut self,
        ctx: &mut EventCtx,
        event: &Event,
        data: &mut TableData,
        env: &Env,
    ) {
        if let Editing::Cell { single_cell, child } = self {
            data.with_mut(single_cell.log.row, |row| child.event(ctx, event, row, env));
        }
    }

    fn start_editing<TableData: IndexedData<Item = RowData>>(
        &mut self,
        ctx: &mut EventCtx,
        data: &mut TableData,
        cell: &SingleCell,
        make_editor: impl FnMut(&CellCtx) -> Option<Box<dyn Widget<RowData>>>,
    ) {
        self.stop_editing(data);
        let mut me = make_editor;
        let cell_ctx = CellCtx::Cell(&cell);
        if let Some(editor) = me(&cell_ctx) {
            let pod = WidgetPod::new(editor);

            *self = Editing::Cell {
                single_cell: cell.clone(),
                child: pod,
            };

            ctx.children_changed();
            ctx.request_layout();
            ctx.set_handled();
        }
    }

    fn stop_editing<TableData: IndexedData<Item = RowData>>(&mut self, _data: &mut TableData) {
        match self {
            Editing::Cell { .. } => {
                // Work out what to do with the previous pod if there is one.
                // We could have lazy editors (that don't write back to data immediately)
                // and w
                // Would need to give them data for their row
            }
            Editing::Inactive => {}
        }
        *self = Editing::Inactive
    }
}

pub struct Cells<TableData: IndexedData> {
    cell_pool: EnsuredPool<
        AxisPair<LogIdx>,
        Option<WidgetPod<TableData::Item, Box<dyn Widget<TableData::Item>>>>,
    >,
    editing: Editing<TableData::Item>,
    dragging_selection: bool,
    phantom_td: PhantomData<TableData>,
}

fn override_rect(mut pix_rect: Rect, pix_ranges: AxisPair<Option<&PixelRange>>) -> Rect {
    if let Some(ov) = pix_ranges[TableAxis::Rows] {
        pix_rect.y0 = ov.p_0;
        pix_rect.y1 = ov.p_1;
    }

    if let Some(ov) = pix_ranges[TableAxis::Columns] {
        pix_rect.x0 = ov.p_0;
        pix_rect.x1 = ov.p_1;
    }
    pix_rect
}

impl<TableData: IndexedData> Cells<TableData> {
    pub fn new() -> Cells<TableData> {
        Cells {
            cell_pool: Default::default(),
            editing: Inactive,
            dragging_selection: false,
            phantom_td: PhantomData::default(),
        }
    }

    fn ensure_cell_pods(&mut self, data: &TableState<TableData>) -> bool {
        let draw_rect = data.visible_rect();
        let cell_rect = data.measures.cell_rect_from_pixels(draw_rect);
        let cell_delegate = &data.cells_del;

        let single_cells = cell_rect.cells().flat_map(|vis| {
            data.remaps
                .get_log_cell(&vis)
                .map(|log| SingleCell::new(vis, log))
        });

        self.cell_pool.ensure(
            single_cells,
            |sc| &sc.log,
            |sc| {
                let cell = CellCtx::Cell(&sc);
                cell_delegate.make_display(&cell).map(WidgetPod::new)
            },
        )
    }

    fn paint_cells(
        &mut self,
        ctx: &mut PaintCtx,
        data: &TableState<TableData>,
        env: &Env,
        rect: &CellRect,
    ) {
        let rtc = &data.resolved_config;
        for vis in rect.cells() {
            if let Some(log) = data.remaps.get_log_cell(&vis) {
                data.table_data.with(log.row, |row| {
                    let overridden = data.overrides.measure.zip_with(&log, |m, k| m.get(k));
                    if let Some(pix_rect) = data.measures.pixel_rect_for_cell(vis) {
                        let pix_rect = override_rect(pix_rect, overridden);
                        let padded_rect = pix_rect.inset(-rtc.cell_padding);

                        ctx.with_save(|ctx| {
                            ctx.clip(padded_rect);
                            if let Some(Some(pod)) = self.cell_pool.get_mut(&log) {
                                if pod.is_initialized() {
                                    pod.paint(ctx, row, env)
                                } else {
                                    log::warn!("Cell pod not init out at {:?}", (log, vis))
                                }
                            }
                        });

                        ctx.stroke_bottom_right_border(
                            &pix_rect,
                            &rtc.cells_border,
                            rtc.cell_border_thickness,
                        );
                    }
                });
            }
        }
    }

    fn paint_selections(
        &mut self,
        ctx: &mut PaintCtx,
        data: &TableState<TableData>,
        rtc: &ResolvedTableConfig,
        cell_rect: &CellRect,
    ) {
        let selected = data.selection.get_drawable_selections(cell_rect);

        let sel_color = &rtc.selection_color;
        let sel_fill = &sel_color.clone().with_alpha(0.2);

        for range_rect in &selected.ranges {
            if let Some(range_draw_rect) = range_rect.to_pixel_rect(&data.measures) {
                ctx.fill(range_draw_rect, sel_fill);
                ctx.stroke(range_draw_rect, sel_color, rtc.cell_border_thickness)
            }
        }

        if let Some(focus) = selected.focus {
            if let Some(pixel_rect) = CellRect::point(focus).to_pixel_rect(&data.measures) {
                ctx.stroke(
                    pixel_rect,
                    &rtc.focus_color,
                    (rtc.cell_border_thickness * 1.5).min(2.),
                );
            }
        }
    }

    fn paint_editing(&mut self, ctx: &mut PaintCtx, data: &TableState<TableData>, env: &Env) {
        match &mut self.editing {
            Editing::Cell { single_cell, child } => {
                if let Some(rect) = CellRect::point(single_cell.vis).to_pixel_rect(&data.measures) {
                    ctx.with_save(|ctx| {
                        ctx.render_ctx.clip(rect);
                        data.table_data
                            .with(single_cell.log.row, |row| child.paint(ctx, row, env));
                    });
                }
            }
            _ => (),
        }
    }
}

impl<TableData: IndexedData> Widget<TableState<TableData>> for Cells<TableData> {
    fn event(
        &mut self,
        ctx: &mut EventCtx,
        event: &Event,
        data: &mut TableState<TableData>,
        env: &Env,
    ) {
        let mut new_selection: Option<TableSelection> = None;

        match event {
            Event::MouseDown(me) => {
                if let Some(cell) = data.find_cell(me.pos) {
                    if self.editing.is_editing(&cell) {
                        self.editing
                            .handle_event(ctx, event, &mut data.table_data, env);
                    } else {
                        if me.count == 1 {
                            let selected_cell = cell.clone();
                            if me.mods.meta() || me.mods.ctrl() {
                                new_selection = data.selection.add_selection(selected_cell.into());
                            } else if me.mods.shift() {
                                new_selection = data.selection.move_extent(selected_cell.into());
                            } else {
                                new_selection = Some(selected_cell.into());
                            }

                            //ctx.set_handled();
                            self.editing.stop_editing(&mut data.table_data);
                            self.dragging_selection = true;
                            ctx.set_active(true);
                        } else if me.count == 2 {
                            let cd = &data.cells_del;
                            self.editing.start_editing(
                                ctx,
                                &mut data.table_data,
                                &cell,
                                |cell_ctx| cd.make_editor(cell_ctx),
                            );
                        }
                    }
                }
            }
            Event::MouseMove(me) if !self.editing.is_active() && self.dragging_selection => {
                if let Some(cell) = data.find_cell(me.pos) {
                    new_selection = data.selection.move_extent(cell.into());
                }
            }
            Event::MouseUp(_) if self.dragging_selection => {
                self.dragging_selection = false;
                ctx.set_active(false);
            }
            Event::KeyDown(ke) if !self.editing.is_active() => {
                match &ke.key {
                    KbKey::ArrowDown => {
                        new_selection =
                            data.selection
                                .move_focus(TableAxis::Rows, VisOffset(1), &data.remaps);
                        ctx.set_handled();
                    }
                    KbKey::ArrowUp => {
                        new_selection =
                            data.selection
                                .move_focus(TableAxis::Rows, VisOffset(-1), &data.remaps);
                        ctx.set_handled();
                    }
                    KbKey::ArrowRight => {
                        new_selection = data.selection.move_focus(
                            TableAxis::Columns,
                            VisOffset(1),
                            &data.remaps,
                        );
                        ctx.set_handled();
                    }
                    KbKey::ArrowLeft => {
                        new_selection = data.selection.move_focus(
                            TableAxis::Columns,
                            VisOffset(-1),
                            &data.remaps,
                        );
                        ctx.set_handled();
                    }
                    KbKey::Character(s) if s == " " => {
                        // This is to match Excel
                        if ke.mods.meta() || ke.mods.ctrl() {
                            new_selection = data
                                .selection
                                .extend_from_focus_in_axis(&TableAxis::Columns, &data.remaps);
                            ctx.set_handled();
                        } else if ke.mods.shift() {
                            new_selection = data
                                .selection
                                .extend_from_focus_in_axis(&TableAxis::Rows, &data.remaps);
                            ctx.set_handled();
                        }

                        // TODO - when Ctrl + Shift, select full grid
                    }
                    KbKey::Copy => log::info!("Copy key"),
                    k => log::info!("Key {:?}", k),
                }
            }
            Event::Command(ref cmd) if ctx.is_focused() && cmd.is(druid::commands::COPY) => {
                log::info!("Copy command");
                ctx.set_handled();
            }
            _ => (),
        }

        if let Some(sel) = new_selection {
            data.selection = sel;
            if data.selection.has_focus() && !self.editing.is_active() {
                ctx.request_focus();
            }
        }

        if let Editing::Cell { single_cell, child } = &mut self.editing {
            if child.is_initialized() {
                data.table_data
                    .with_mut(single_cell.log.row, |row| child.event(ctx, event, row, env));
            }
        }

        if let Some(foc) = data.selection.focus() {
            let mut delivered = false;

            data.table_data.with_mut(foc.log.row, |item| {
                if let Some(Some(pod)) = self.cell_pool.get_mut(&foc.log) {
                    if pod.is_initialized() {
                        pod.event(ctx, event, item, env);
                        delivered = true;
                    }
                }
            });
            //log::info!("Wanted to forward event to focused cell {:?} {:?}", event, delivered);
        }
    }

    fn lifecycle(
        &mut self,
        ctx: &mut LifeCycleCtx,
        event: &LifeCycle,
        data: &TableState<TableData>,
        env: &Env,
    ) {
        if let Editing::Cell { single_cell, child } = &mut self.editing {
            data.table_data.with(single_cell.log.row, |row| {
                child.lifecycle(ctx, event, row, env)
            });
        }
        // TODO: visibility?
        for (log_cell, pod) in &mut self.cell_pool.entries_mut() {
            if let Some(pod) = pod {
                data.table_data.with(log_cell.row, |row| {
                    if matches!(
                        event,
                        LifeCycle::WidgetAdded
                            | LifeCycle::Internal(InternalLifeCycle::RouteWidgetAdded)
                    ) || pod.is_initialized()
                    {
                        pod.lifecycle(ctx, event, row, env);
                    }
                });
            }
        }
    }

    fn update(
        &mut self,
        ctx: &mut UpdateCtx,
        old_data: &TableState<TableData>,
        data: &TableState<TableData>,
        env: &Env,
    ) {
        if !old_data.table_data.same(&data.table_data) || !old_data.remaps.same(&data.remaps) {
            log::info!("table data or remaps changed, request cells layout");
            ctx.request_layout();
        }

        if !old_data.selection.same(&data.selection) {
            ctx.request_paint();
        }

        let editor_valid = match &mut self.editing {
            Editing::Cell { single_cell, child } => {
                let valid = data
                    .remaps
                    .get_log_cell(&single_cell.vis)
                    .map(|log| log.same(&single_cell.log))
                    .unwrap_or(false);
                if valid {
                    data.table_data
                        .with(single_cell.log.row, |row| child.update(ctx, row, env));
                }
                valid
            }
            _ => true,
        };

        if !editor_valid {
            self.editing = Editing::Inactive;
            ctx.children_changed();
        }

        if !old_data.scroll_rect.same(&data.scroll_rect) {
            if self.ensure_cell_pods(data) {
                ctx.children_changed();
                ctx.request_anim_frame();
            }
        }

        // TODO: Stateless cell widgets?
        // TODO: Extract changed cells from data.table_data (extend IndexedData interface)
        for (log_cell, pod) in &mut self.cell_pool.entries_mut() {
            if let Some(pod) = pod {
                if pod.is_initialized() {
                    data.table_data.with(log_cell.row, |row| {
                        pod.update(ctx, row, env);
                    });
                }
            }
        }
    }

    fn layout(
        &mut self,
        ctx: &mut LayoutCtx,
        bc: &BoxConstraints,
        data: &TableState<TableData>,
        env: &Env,
    ) -> Size {
        bc.debug_check("TableCells");
        //log::info!("Layout {:?}", (Instant::now() - self.start).as_secs_f32());

        if let Editing::Cell { single_cell, child } = &mut self.editing {
            let vis = &single_cell.vis;

            let pixels_len = data
                .measures
                .zip_with(&vis, |m, v| m.pixels_length_for_vis(*v))
                .opt();
            let first_pix = data
                .measures
                .zip_with(&vis, |m, v| m.first_pixel_from_vis(*v))
                .opt();

            if let (Some(size), Some(origin)) = (
                pixels_len.as_ref().map(AxisPair::size),
                first_pix.as_ref().map(AxisPair::point),
            ) {
                let bc = BoxConstraints::tight(size).loosen();
                data.table_data.with(single_cell.log.row, |row| {
                    child.layout(ctx, &bc, row, env);
                    child.set_origin(ctx, origin)
                });
            }
        }
        let measured = data.measures.measured_size();
        let size = bc.constrain(measured);

        let draw_rect = data
            .scroll_rect
            .intersect(Rect::from_origin_size(Point::ZERO, measured));

        let cell_rect = data.measures.cell_rect_from_pixels(draw_rect);

        for vis in cell_rect.cells() {
            if let Some(log) = data.remaps.get_log_cell(&vis) {
                data.table_data.with(log.row, |row| {
                    if let Some(Some(cell_pod)) = self.cell_pool.get_mut(&log) {
                        let overridden = data.overrides.measure.zip_with(&log, |m, k| m.get(k));

                        if let Some(mut vis_rect) =
                            CellRect::point(vis).to_pixel_rect(&data.measures)
                        {
                            if let Some(ov) = overridden[TableAxis::Rows] {
                                vis_rect.y0 = ov.p_0;
                                vis_rect.y1 = ov.p_1;
                            }

                            if let Some(ov) = overridden[TableAxis::Columns] {
                                vis_rect.x0 = ov.p_0;
                                vis_rect.x1 = ov.p_1;
                            }

                            if cell_pod.is_initialized() {
                                cell_pod.layout(
                                    ctx,
                                    &BoxConstraints::tight(vis_rect.size()).loosen(),
                                    row,
                                    env,
                                );
                                // TODO: could align the given size to different edge/corner
                                cell_pod.set_origin(ctx, vis_rect.origin());
                            }
                        }
                    }
                });
            }
        }

        size
    }

    fn paint(&mut self, ctx: &mut PaintCtx, data: &TableState<TableData>, env: &Env) {
        let rtc = &data.resolved_config;
        let rect = ctx.region().bounding_box();

        let draw_rect = rect.intersect(Rect::from_origin_size(
            Point::ZERO,
            data.measures.measured_size(),
        ));

        let cell_rect = data.measures.cell_rect_from_pixels(draw_rect);

        ctx.fill(draw_rect, &rtc.cells_background);
        self.paint_cells(ctx, data, env, &cell_rect);
        self.paint_selections(ctx, data, &rtc, &cell_rect);

        self.paint_editing(ctx, data, env);
    }
}

impl<TableData> BindableAccess for Cells<TableData>
where
    TableData: IndexedData,
    TableData::Item: Data,
{
    bindable_self_body!();
}
