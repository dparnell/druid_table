use crate::axis_measure::{AxisMeasure, AxisPair, TableAxis, VisOffset};
use crate::cells::CellsDelegate;
use crate::config::ResolvedTableConfig;
use crate::data::{IndexedDataDiff, IndexedDataDiffer, RemapDetails};
use crate::headings::HeadersFromData;
use crate::selection::{CellDemap, SingleCell};
use crate::{Cells, DisplayFactory, Headings, IndexedData, LogIdx, Remap, RemapSpec, Remapper, TableConfig, TableSelection, VisIdx, bindable_self_body};
use druid::widget::{ClipBox, Scope, ScopePolicy, ScopeTransfer, Scroll};
use druid::{
    BoxConstraints, Data, Env, Event, EventCtx, LayoutCtx, Lens, LensExt, LifeCycle, LifeCycleCtx,
    PaintCtx, Point, Rect, Size, UpdateCtx, Widget, WidgetExt, WidgetPod,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc};
use std::time::{SystemTime};
use crate::bindings::{AxisPositionProperty, BindableAccess, Property, ReadScrollRect, WidgetBindingExt};

pub struct HeaderBuild<HeadersSource: HeadersFromData + 'static> {
    source: HeadersSource,
    render: Box<dyn DisplayFactory<HeadersSource::Header>>,
}

impl<HeadersSource: HeadersFromData + 'static> HeaderBuild<HeadersSource> {
    pub fn new(
        source: HeadersSource,
        render: Box<dyn DisplayFactory<HeadersSource::Header>>,
    ) -> Self {
        HeaderBuild { source, render }
    }
}

// This trait exists to move type parameters to associated types
pub trait HeaderBuildT {
    type TableData: Data;
    type Header: Data;
    type Headers: IndexedData<Item = Self::Header> + 'static;
    type HeadersSource: HeadersFromData<Headers = Self::Headers, Header = Self::Header, TableData = Self::TableData>
        + 'static;

    fn content(self) -> (Self::HeadersSource, Box<dyn DisplayFactory<Self::Header>>);
}

impl<HeadersSource: HeadersFromData + 'static> HeaderBuildT for HeaderBuild<HeadersSource> {
    type TableData = HeadersSource::TableData;
    type Header = HeadersSource::Header;
    type Headers = HeadersSource::Headers;
    type HeadersSource = HeadersSource;

    fn content(
        self,
    ) -> (
        Self::HeadersSource,
        Box<dyn DisplayFactory<HeadersSource::Header>>,
    ) {
        (self.source, self.render)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PixelRange {
    pub(crate) p_0: f64,
    pub(crate) p_1: f64,
}

impl PixelRange {
    pub fn new(p_0: f64, p_1: f64) -> Self {
        PixelRange {
            p_0: p_0.min(p_1),
            p_1: p_0.max(p_1),
        }
    }

    pub fn move_to(&mut self, p_0: f64) {
        let diff = self.p_1 - self.p_0;
        self.p_0 = p_0;
        self.p_1 = p_0 + diff;
        log::info!("Move px range {:?}", (diff, self.p_0, self.p_1))
    }

    pub fn extent(&self) -> f64 {
        self.p_1 - self.p_0
    }
}


struct ExitingRow {
    pub(crate) y_0: f64,
    pub(crate) y_1: f64,
    //pub(crate) row: ImageBuffer
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TableOverrides {
    // rows that will be visible at the end of animation
    pub(crate) measure: AxisPair<HashMap<LogIdx, PixelRange>>,
    // exiting rows
    //pub(crate) exiting_rows: Vec<ExitingRow>
}

#[derive(Data, Clone, Lens)]
pub(crate) struct TableState<TableData> {
    pub(crate) table_data: TableData,
    pub(crate) scroll_rect: Rect,
    pub(crate) config: TableConfig,
    pub(crate) resolved_config: ResolvedTableConfig,
    pub(crate) remap_specs: AxisPair<RemapSpec>,
    pub(crate) remaps: AxisPair<Remap>,
    pub(crate) measures: AxisPair<AxisMeasure>,
    pub(crate) selection: TableSelection,

    pub(crate) cells_del: Arc<dyn CellsDelegate<TableData>>,
    pub(crate) last_diff: Option<IndexedDataDiff>,
    #[data(ignore)]
    pub(crate) overrides: TableOverrides,
}

impl<TableData: IndexedData> TableState<TableData> {
    pub fn new(
        config: TableConfig,
        data: TableData,
        measures: AxisPair<AxisMeasure>,
        cells_del: Arc<dyn CellsDelegate<TableData>>,
    ) -> Self {
        let remaps = AxisPair::new(
            Remap::Pristine(data.data_len()),
            Remap::Pristine(cells_del.data_fields(&data)),
        );

        let mut state = TableState {
            scroll_rect: Rect::ZERO,
            config,
            resolved_config: ResolvedTableConfig::default(),
            table_data: data,
            remap_specs: AxisPair::new(cells_del.initial_spec(), RemapSpec::default()),
            remaps,
            selection: TableSelection::default(),
            measures,
            cells_del,
            last_diff: None,
            overrides: TableOverrides::default(),
        };
        state.remap_rows();
        state.refresh_measure(TableAxis::Rows);
        state.refresh_measure(TableAxis::Columns);
        state
    }

    fn axis_log_len(&self, axis: TableAxis) -> usize {
        match axis {
            TableAxis::Rows => self.table_data.data_len(),
            TableAxis::Columns => self.cells_del.data_fields(&self.table_data),
        }
    }

    fn refresh_measure(&mut self, axis: TableAxis) {
        let log_len = self.axis_log_len(axis);
        self.measures[axis].set_axis_properties(
            self.resolved_config.cell_border_thickness,
            log_len,
            &self.remaps[axis],
        );
        // TODO: Maintain logical selection
        self.selection = TableSelection::NoSelection;
    }

    fn remap_rows(&mut self) {
        self.remaps[TableAxis::Rows] = self
            .cells_del
            .remap_from_records(&self.table_data, &self.remap_specs[TableAxis::Rows]);
    }

    pub(crate) fn visible_rect(&self) -> Rect {
        self.scroll_rect.intersect(Rect::from_origin_size(
            Point::ZERO,
            self.measures.measured_size(),
        ))
    }

    pub(crate) fn find_cell(&self, pos: Point) -> Option<SingleCell> {
        let vis = self
            .measures
            .zip_with(&AxisPair::new(pos.y, pos.x), |m, p| {
                m.vis_idx_from_pixel(*p)
            })
            .opt()?;
        let log = self.remaps.get_log_cell(&vis)?;
        Some(SingleCell::new(vis, log))
    }

    pub(crate) fn vis_idx_visible_for_axis(&self, axis: TableAxis) -> impl Iterator<Item = VisIdx> {
        let vis_rect = self.visible_rect();
        let cells = self.measures.cell_rect_from_pixels(vis_rect);
        let (from, to) = cells.range(axis);
        VisIdx::range_inc_iter(from, to)
    }

    pub(crate) fn log_idx_in_visible_order_for_axis(
        &self,
        axis: TableAxis,
    ) -> impl Iterator<Item = LogIdx> + '_ {
        let remap = &self.remaps[axis];
        self.vis_idx_visible_for_axis(axis)
            .flat_map(move |vis| remap.get_log_idx(vis))
    }

    pub(crate) fn log_and_vis_idx_for_axis(
        &self,
        axis: TableAxis,
    ) -> impl Iterator<Item = (LogIdx, VisIdx)> + '_ {
        let remap = &self.remaps[axis];
        self.vis_idx_visible_for_axis(axis)
            .flat_map(move |vis| remap.get_log_idx(vis).map(|log| (log, vis)))
    }

    pub fn explicit_header_move(
        &mut self,
        axis: TableAxis,
        moved_from_idx: VisIdx,
        moved_to_idx: VisIdx,
    ) {
        log::info!(
            "Move selection {:?}\n\t on {:?} from {:?} to {:?}",
            self.selection,
            axis,
            moved_from_idx,
            moved_to_idx
        );

        let size = match axis {
            TableAxis::Columns => self.cells_del.data_fields(&self.table_data),
            TableAxis::Rows => self.table_data.data_len(),
        };

        if size > 0 {
            let last_vis = VisIdx(size - 1);

            let move_by = moved_to_idx - moved_from_idx;

            if move_by != VisOffset(0) {
                if let Some(mut headers_moved) = self.selection.fully_selected_on_axis(axis) {
                    let mut past_end: Vec<LogIdx> = Default::default();

                    if move_by.0 > 0 {
                        headers_moved.reverse()
                    }

                    let mut current: Vec<_> =
                        self.remaps[axis].log_idx_in_vis_order(last_vis).collect();
                    for vis_idx in headers_moved {
                        let new_vis = vis_idx + move_by;
                        if vis_idx.0 >= current.len() {
                            log::warn!(
                                "Trying to move {:?}->{:?} to {:?} but len is {}",
                                vis_idx,
                                current.get(vis_idx.0),
                                new_vis,
                                current.len()
                            )
                        } else {
                            let log_idx = current.remove(vis_idx.0);

                            if new_vis.0 >= current.len() {
                                past_end.push(log_idx)
                            } else {
                                current.insert(new_vis.0, log_idx)
                            }
                        }
                    }

                    if move_by.0 > 0 {
                        past_end.reverse()
                    }
                    current.append(&mut past_end);

                    //self.selection.move_by(move_by, axis);
                    self.remaps[axis] = Remap::Selected(RemapDetails::make_full(current));
                    self.selection = TableSelection::NoSelection;
                }
            }
        }
    }
}

impl CellDemap for AxisPair<Remap> {
    fn get_log_idx(&self, axis: TableAxis, vis: VisIdx) -> Option<LogIdx> {
        self[axis].get_log_idx(vis)
    }

    fn get_vis_idx(&self, axis: TableAxis, log: LogIdx) -> Option<VisIdx> {
        self[axis].get_vis_idx(log)
    }
}

type TableChild<TableData> = WidgetPod<
    TableData,
    Scope<TableScopePolicy<TableData>, Box<dyn Widget<TableState<TableData>>>>,
>;

struct TableScopePolicy<TableData> {
    config: TableConfig,
    resolved_config: Option<ResolvedTableConfig>,
    measures: AxisPair<AxisMeasure>,
    cells_delegate: Arc<dyn CellsDelegate<TableData>>,
    differ: Box<dyn IndexedDataDiffer<TableData>>,
    phantom_td: PhantomData<TableData>,
}

impl<TableData> TableScopePolicy<TableData> {
    pub fn new(
        config: TableConfig,
        measures: AxisPair<AxisMeasure>,
        cells_delegate: Arc<dyn CellsDelegate<TableData>>,
        differ: Box<dyn IndexedDataDiffer<TableData>>,
    ) -> Self {
        TableScopePolicy {
            config,
            resolved_config: None,
            measures,
            cells_delegate,
            differ,
            phantom_td: Default::default(),
        }
    }
}

impl<TableData: IndexedData> ScopePolicy for TableScopePolicy<TableData> {
    type In = TableData;
    type State = TableState<TableData>;
    type Transfer = TableScopeTransfer<TableData>;

    fn create(self, inner: &Self::In) -> (Self::State, Self::Transfer) {
        (
            TableState::new(
                self.config,
                inner.clone(),
                self.measures,
                self.cells_delegate,
            ),
            TableScopeTransfer::new(self.differ),
        )
    }
}

struct TableScopeTransfer<TableData> {
    phantom_td: PhantomData<TableData>,
    differ: Box<dyn IndexedDataDiffer<TableData>>,
}

impl<TableData: IndexedData> TableScopeTransfer<TableData> {
    pub fn new(differ: Box<dyn IndexedDataDiffer<TableData>>) -> Self {
        TableScopeTransfer {
            phantom_td: Default::default(),
            differ,
        }
    }
}

impl<TableData: IndexedData> ScopeTransfer for TableScopeTransfer<TableData> {
    type In = TableData;
    type State = TableState<TableData>;

    fn read_input(&self, state: &mut Self::State, input: &Self::In) {
        log::info!("Read input table data to TableState");
        if !input.same(&state.table_data) {
            log::info!("Actually wrote table data to TableState");
            state.table_data = input.clone();
        }
    }

    fn write_back_input(&self, state: &Self::State, input: &mut Self::In) {
        if !input.same(&state.table_data) {
            *input = state.table_data.clone();
        }
    }

}

type LayoutChild<T> = WidgetPod<T, Box<dyn Widget<T>>>;

struct TableLayout<T> {
    cells: LayoutChild<T>,
    headers: AxisPair<Option<LayoutChild<T>>>,
}

impl<T> TableLayout<T> {
    pub fn new(cells: LayoutChild<T>, headers: AxisPair<Option<LayoutChild<T>>>) -> Self {
        TableLayout { cells, headers }
    }

    fn for_each(&mut self, mut f: impl FnMut(&mut LayoutChild<T>)) {
        f(&mut self.cells);
        self.headers.for_each_mut(|_, opt| {
            if let Some(w) = opt {
                f(w)
            }
        })
    }
}

impl<T: Data> Widget<T> for TableLayout<T> {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, data: &mut T, env: &Env) {
        self.for_each(|c| c.event(ctx, event, data, env));
    }

    fn lifecycle(&mut self, ctx: &mut LifeCycleCtx, event: &LifeCycle, data: &T, env: &Env) {
        self.for_each(|c| c.lifecycle(ctx, event, data, env))
    }

    fn update(&mut self, ctx: &mut UpdateCtx, _old_data: &T, data: &T, env: &Env) {
        self.for_each(|c| c.update(ctx, data, env))
    }

    fn layout(&mut self, ctx: &mut LayoutCtx, bc: &BoxConstraints, data: &T, env: &Env) -> Size {
        // Assume that the current row header width is what it will be given.
        // This is to avoid shrinking one of the header clip boxes which will screw up its pan position.
        // This is slightly dodgy afaict but in practice converges
        let cur_row_h_width = self
            .headers
            .row
            .as_ref()
            .map_or(0., |w| w.layout_rect().width());

        let bc = &bc.loosen();
        dbg!(&bc);
        let bc = bc.shrink(Size::new(cur_row_h_width, 0.));

        let col_size = if let Some(w) = &mut self.headers.col {
            w.layout(ctx, &bc, data, env)
        } else {
            Size::ZERO
        };
        dbg!(&col_size);

        let bc = bc.shrink(Size::new(0., col_size.height));
        dbg!(&bc);
        let row_size = if let Some(w) = &mut self.headers.row {
            w.layout(ctx, &bc, data, env)
        } else {
            Size::ZERO
        };
        dbg!(&row_size);

        let corner = Size::new(row_size.width, col_size.height);
        let corner_point = Point::new(corner.width, corner.height);

        self.headers.for_each_mut(|t_axis, header_w| {
            if let Some(w) = header_w {
                let (main, _) = t_axis.pixels_from_point(&corner_point);
                w.set_origin(ctx, t_axis.coords(main, 0.).into())
            }
        });

        let cells_size = self.cells.layout(ctx, &bc, data, env);
        self.cells.set_origin(ctx, corner_point);
        dbg!(&cells_size);
        cells_size + corner
    }

    fn paint(&mut self, ctx: &mut PaintCtx, data: &T, env: &Env) {
        self.for_each(|c| c.paint(ctx, data, env))
    }
}

pub struct Table<TableData: IndexedData> {
    child: TableChild<TableData>,
}

impl<TableData: IndexedData> Table<TableData> {
    pub fn new<RowH, ColH, CellsDel>(
        cells_delegate: CellsDel,
        row_h: Option<RowH>,
        col_h: Option<ColH>,
        table_config: TableConfig,
        measures: AxisPair<AxisMeasure>,
        differ: Box<dyn IndexedDataDiffer<TableData>>,
    ) -> Self
    where
        RowH: HeaderBuildT<TableData = TableData>,
        ColH: HeaderBuildT<TableData = TableData>,
        CellsDel: CellsDelegate<TableData> + 'static,
    {
        Table {
            child: Table::build_child(cells_delegate, row_h, col_h, table_config, measures, differ),
        }
    }

    fn build_child<RowH, ColH, CellsDel>(
        cells_delegate: CellsDel,
        row_h: Option<RowH>,
        col_h: Option<ColH>,
        table_config: TableConfig,
        measures: AxisPair<AxisMeasure>,
        differ: Box<dyn IndexedDataDiffer<TableData>>,
    ) -> TableChild<TableData>
    where
        RowH: HeaderBuildT<TableData = TableData>,
        ColH: HeaderBuildT<TableData = TableData>,
        CellsDel: CellsDelegate<TableData> + 'static,
    {
        let cells = WidgetPod::new(
            Scroll::new(Cells::new())
                .binding(ReadScrollRect::PROP.with(TableState::scroll_rect))
                .boxed(),
        );

        let row = row_h.map(|hb| {
            let (source, render) = hb.content();
            let row_headings = Headings::new(TableAxis::Rows, source, Box::new(render), false);

            let row_scroll = ClipBox::managed(row_headings).binding(
                AxisPositionProperty::VERTICAL
                    .with(TableState::<TableData>::scroll_rect.then(lens!(Rect, y0))),
            );
            WidgetPod::new(row_scroll.boxed())
        });

        let col = col_h.map(|cb| {
            let (source, render) = cb.content();

            let col_headings = Headings::new(TableAxis::Columns, source, Box::new(render), true);
            let ch_scroll = ClipBox::managed(col_headings).binding(
                AxisPositionProperty::HORIZONTAL
                    .with(TableState::<TableData>::scroll_rect.then(lens!(Rect, x0))),
            );
            WidgetPod::new(ch_scroll.boxed())
        });

        let headers = AxisPair::new(row, col);

        let tl = TableLayout::new(cells, headers);
        let policy = TableScopePolicy::new(
            table_config.clone(),
            measures,
            Arc::new(cells_delegate),
            differ,
        );
        Self::wrap_in_scope(policy, tl)
    }

    fn wrap_in_scope<W: Widget<TableState<TableData>> + 'static>(
        policy: TableScopePolicy<TableData>,
        widget: W,
    ) -> TableChild<TableData> {
        WidgetPod::new(Scope::new(policy, Box::new(widget)))
    }

    fn state(&self) -> Option<&TableState<TableData>> {
        self.child.widget().state()
    }

    fn state_mut(&mut self) -> Option<&mut TableState<TableData>> {
        self.child.widget_mut().state_mut()
    }
}

impl<TableData: IndexedData> Widget<TableData> for Table<TableData> {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, data: &mut TableData, env: &Env) {
        self.child.event(ctx, event, data, env);
    }

    fn lifecycle(
        &mut self,
        ctx: &mut LifeCycleCtx,
        event: &LifeCycle,
        data: &TableData,
        env: &Env,
    ) {
        if let LifeCycle::WidgetAdded = event {
            if let Some(state) = self.child.widget_mut().state_mut() {
                state.resolved_config = state.config.resolve(env);
            }
        }
        self.child.lifecycle(ctx, event, data, env);
    }

    fn update(&mut self, ctx: &mut UpdateCtx, _old_data: &TableData, data: &TableData, env: &Env) {
        log::info!(
            "Table update {:?} data:{}, env:{}, req_up:{}",
            SystemTime::now(),
            !_old_data.same(data),
            ctx.env_changed(),
            ctx.has_requested_update()
        );
        if ctx.env_changed() {
            if let Some(state) = self.child.widget_mut().state_mut() {
                state.resolved_config = state.config.resolve(env);
            }
        }
        self.child.update(ctx, data, env);
    }

    fn layout(
        &mut self,
        ctx: &mut LayoutCtx,
        bc: &BoxConstraints,
        data: &TableData,
        env: &Env,
    ) -> Size {
        let size = self.child.layout(ctx, bc, data, env);
        /*
        self.child
            .set_layout_rect(ctx, data, env, Rect::from_origin_size(Point::ORIGIN, size));
         */
        size
    }

    fn paint(&mut self, ctx: &mut PaintCtx, data: &TableData, env: &Env) {
        self.child.paint_raw(ctx, data, env);
    }
}

impl<TableData: IndexedData> BindableAccess for Table<TableData> {
    bindable_self_body!();
}

pub struct TableSelectionProp<TableData> {
    phantom_td: PhantomData<TableData>,
}

impl<TableData> Default for TableSelectionProp<TableData> {
    fn default() -> Self {
        Self {
            phantom_td: Default::default(),
        }
    }
}

impl<TableData: IndexedData> Property for TableSelectionProp<TableData> {
    type Controlled = Table<TableData>;
    type Value = TableSelection;
    type Change = ();
    type Requests = ();

    fn write_prop(
        &self,
        controlled: &mut Self::Controlled,
        _ctx: &mut UpdateCtx,
        field_val: &Self::Value,
        _env: &Env,
    ) {
        if let Some(s) = controlled.state_mut() {
            s.selection = field_val.clone()
        }
    }

    fn append_changes(
        &self,
        controlled: &Self::Controlled,
        field_val: &Self::Value,
        change: &mut Option<Self::Change>,
        _env: &Env,
    ) {
        if let Some(s) = controlled.state() {
            if !s.selection.same(field_val) {
                *change = Some(())
            }
        }
    }

    fn update_data_from_change(
        &self,
        controlled: &Self::Controlled,
        _ctx: &mut EventCtx,
        field: &mut Self::Value,
        _change: Self::Change,
        _env: &Env,
    ) {
        if let Some(s) = controlled.state() {
            *field = s.selection.clone()
        }
    }

    fn initialise_data(
        &self,
        controlled: &Self::Controlled,
        _ctx: &mut EventCtx,
        field: &mut Self::Value,
        _env: &Env,
    ) {
        if let Some(s) = controlled.state() {
            *field = s.selection.clone()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::VisIdx;

    #[test]
    fn test_range() {
        let v: Vec<_> = VisIdx::range_inc_iter(VisIdx(1), VisIdx(0)).collect();
        assert!(v.is_empty())
    }
}
