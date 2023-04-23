use std::fmt::{Debug, Display, Formatter};

use druid_table::{
    column, AxisMeasurementType, CellDelegate, ShowHeadings, SlowVectorDiffer, SortDirection,
    Table, TableAxis, TableBuilder, TableSelection, TableSelectionProp, WidgetCell,
};

use druid::im::{vector, Vector};
use druid::kurbo::CircleSegment;
use druid::theme::PLACEHOLDER_COLOR;
use druid::widget::{Button, Checkbox, Container, CrossAxisAlignment, Flex, Label, LineBreaking, MainAxisAlignment, Padding, Painter, RadioGroup, RawLabel, SizedBox, Stepper, TextBox, ViewSwitcher};
use druid::{AppLauncher, Data, Env, FontDescriptor, FontFamily, Lens, LensExt, LocalizedString, Menu, PaintCtx, RenderContext, Widget, WidgetExt, WindowDesc, WindowId};
use druid::{Color};

use std::cell::RefCell;
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::fmt;
use druid_table::bindings::{Property, WidgetBindingExt};

const WINDOW_TITLE: LocalizedString<HelloState> = LocalizedString::new("Hello Table!");

#[derive(Clone, Data, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum WordOrder {
    SubjectObjectVerb,
    SubjectVerbObject,
}

impl Display for WordOrder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Clone, Data, Lens, Debug)]
struct HelloRow {
    lang: String,
    greeting: String,
    westernised: String,
    who_knows: f64,
    complete: bool,
    word_order: WordOrder,
}

impl HelloRow {
    fn new(
        lang: impl Into<String>,
        greeting: impl Into<String>,
        westernised: impl Into<String>,
        percent: f64,
        complete: bool,
        word_order: WordOrder,
    ) -> HelloRow {
        HelloRow {
            lang: lang.into(),
            greeting: greeting.into(),
            westernised: westernised.into(),
            who_knows: percent / 100.,
            complete,
            word_order,
        }
    }
}

#[derive(Clone, Lens, Data, Debug)]
struct Settings {
    show_headings: ShowHeadings,
    border_thickness: f64,
    row_fixed: bool,
    col_fixed: bool,
}

impl PartialEq for Settings {
    fn eq(&self, other: &Self) -> bool {
        self.same(other)
    }
}

#[derive(Clone, Data, Lens)]
struct HelloState {
    items: Vector<HelloRow>,
    settings: Settings,
    table_selection: TableSelection,
}

fn pie_cell<Row: Data, MakeLens: Fn() -> L, L: Lens<Row, f64> + 'static>(
    make_lens: MakeLens,
) -> impl CellDelegate<Row> {
    WidgetCell::new_unsorted(
        |_| {
            Painter::new(|ctx: &mut PaintCtx, data: &f64, _env: &Env| {
                let rect = ctx.size().to_rect().inset(-5.);
                let circle = CircleSegment::new(
                    rect.center(),
                    f64::min(rect.height(), rect.width()) / 2.,
                    0.,
                    0.,
                    2. * PI * *data,
                );
                ctx.fill(&circle, &Color::rgb8(0x0, 0xFF, 0x0));
                ctx.stroke(&circle, &Color::BLACK, 1.5);
            })
        },
        make_lens,
    )
    .edit_with(|_| Stepper::new().with_range(0.0, 1.0).with_step(0.02))
    .compare_with(|a, b| f64::partial_cmp(a, b).unwrap_or(Ordering::Equal))
}

fn build_main_widget() -> impl Widget<HelloState> {
    // Need a wrapper widget to get selection/scroll events out of it

    let count = RefCell::new(0);
    let row = move || {
        let mut cur = count.borrow_mut();
        *cur = *cur + 1;
        HelloRow::new(
            format!("Japanese_{}", *cur),
            "こんにちは",
            "Kon'nichiwa",
            63.,
            true,
            WordOrder::SubjectObjectVerb,
        )
    };

    let buttons = Flex::column()
        .cross_axis_alignment(CrossAxisAlignment::Start)
        .with_child(decor(Label::new("Modify table")))
        .with_child(
            Flex::column()
                .with_child(
                    Button::new("Add row")
                        .on_click(move |_, data: &mut Vector<HelloRow>, _| {
                            data.insert(data.len().saturating_sub(2), row());
                        })
                        .expand_width(),
                )
                .with_child(
                    Button::new("Remove row")
                        .on_click(|_, data: &mut Vector<HelloRow>, _| {
                            data.remove(data.len().saturating_sub(4));
                        })
                        .expand_width(),
                )
                .padding(5.0),
        )
        .fix_width(200.0)
        .lens(HelloState::items);
    let headings_control = Flex::column()
        .cross_axis_alignment(CrossAxisAlignment::Start)
        .with_child(decor(Label::new("Headings to show")))
        .with_child(RadioGroup::column(vec![
            ("Just cells", ShowHeadings::JustCells),
            ("Column headings", ShowHeadings::One(TableAxis::Columns)),
            ("Row headings", ShowHeadings::One(TableAxis::Rows)),
            ("Both", ShowHeadings::Both),
        ]))
        .lens(HelloState::settings.then(Settings::show_headings));
    let style = Flex::column()
        .cross_axis_alignment(CrossAxisAlignment::Start)
        .with_child(decor(Label::new("Style")))
        .with_child(
            Flex::row()
                .with_child(Label::new("Border thickness"))
                .with_flex_spacer(1.0)
                .with_child(Label::new(|p: &f64, _: &Env| p.to_string()))
                .with_child(Stepper::new().with_range(0., 20.0).with_step(0.5))
                .lens(HelloState::settings.then(Settings::border_thickness)),
        );

    let measurements = Flex::column()
        .cross_axis_alignment(CrossAxisAlignment::Start)
        .with_child(decor(Label::new("Uniform axes")))
        .with_child(Flex::row().with_child(Checkbox::new("Rows").lens(Settings::row_fixed)))
        .with_child(Flex::row().with_child(Checkbox::new("Columns").lens(Settings::col_fixed)))
        .lens(HelloState::settings);

    let selection = Flex::column()
        .cross_axis_alignment(CrossAxisAlignment::Start)
        .with_child(decor(Label::new("Selection")))
        .with_child(
            Label::new(|t: &TableSelection, _e: &Env| format!("{:#?}", t))
                .with_line_break_mode(LineBreaking::WordWrap),
        )
        .lens(HelloState::table_selection);

    let sidebar = Flex::column()
        .main_axis_alignment(MainAxisAlignment::Start)
        .cross_axis_alignment(CrossAxisAlignment::Start)
        .with_child(group(buttons))
        .with_child(group(headings_control))
        .with_child(group(style))
        .with_child(group(measurements))
        .with_child(group(selection))
        .with_flex_spacer(1.)
        .fix_width(200.0);

    let vs = ViewSwitcher::new(
        |ts: &HelloState, _| ts.settings.clone(),
        |sh, _, env| {
            let table = build_table(sh.clone(), env).lens(HelloState::items);
            table
                .binding(
                    TableSelectionProp::default()
                        .read()
                        .with(HelloState::table_selection),
                )
                .boxed()
        },
    )
    .padding(10.);

    Flex::row()
        .cross_axis_alignment(CrossAxisAlignment::Start)
        .with_child(sidebar)
        .with_flex_child(vs, 1.)
}

fn decor<T: Data>(label: Label<T>) -> SizedBox<T> {
    label
        .padding(5.)
        .background(PLACEHOLDER_COLOR)
        .expand_width()
}

fn group<T: Data, W: Widget<T> + 'static>(w: W) -> Padding<T, Container<T>> {
    w.border(Color::WHITE, 0.5).padding(5.)
}

fn build_table(settings: Settings, env: &Env) -> Table<Vector<HelloRow>> {
    let measurement_type = if settings.col_fixed {
        AxisMeasurementType::Uniform
    } else {
        AxisMeasurementType::Individual
    };
    TableBuilder::new()
        .measuring_axis(TableAxis::Rows, measurement_type)
        .measuring_axis(TableAxis::Columns, measurement_type)
        .headings(settings.show_headings)
        .border(settings.border_thickness)
        .with_column(
            "Language",
            WidgetCell::new(
                |_| {
                    RawLabel::new()
                        .with_font(FontDescriptor::new(FontFamily::SERIF))
                        .with_text_size(15.)
                        .with_text_color(Color::BLUE)
                },
                || HelloRow::lang,
            )
            .compare_with(|a, b| a.len().cmp(&b.len()))
            .edit_with(|_| TextBox::new()),
        )
        .with_column(
            "Complete",
            WidgetCell::new(|_| Checkbox::new(""), || HelloRow::complete),
        )
        .with_column("Greeting", WidgetCell::text(|| HelloRow::greeting))
        .with_column(
            "Westernised",
            WidgetCell::text_configured(|rl| rl.with_text_size(17.), || HelloRow::westernised),
        )
        .with(column("Who knows?", pie_cell(|| HelloRow::who_knows)).sort(SortDirection::Ascending))

        .with_column(
            "Greeting 2 with very long column name",
            WidgetCell::text_configured(
                |rl| {
                    rl.with_font(FontDescriptor::new(FontFamily::new_unchecked(
                        "Courier New",
                    )))
                },
                || HelloRow::greeting,
            ),
        )
        .with_column(
            "Greeting 3",
            WidgetCell::text_configured(
                |rl| rl.with_text_color(Color::rgb8(0xD0, 0, 0)),
                || HelloRow::greeting,
            ),
        )
        .with_column("Greeting 4", WidgetCell::text(|| HelloRow::greeting))
        .with_column("Greeting 5", WidgetCell::text(|| HelloRow::greeting))
        .diff_with(SlowVectorDiffer::new(|row: &HelloRow| row.lang.clone()))
        .build()
}

fn make_menu(_: Option<WindowId>, _state: &HelloState, _: &Env) -> Menu<HelloState> {
    let mut base = Menu::empty();
    #[cfg(target_os = "macos")]
    {
        base = base.append(druid::platform_menus::mac::application::default())
    }
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        base = base.entry(druid::platform_menus::win::file::default());
    }
    base.entry(
        Menu::new(LocalizedString::new("common-menu-edit-menu"))
            .entry(druid::platform_menus::common::undo())
            .entry(druid::platform_menus::common::redo())
            .separator()
            .entry(druid::platform_menus::common::cut().enabled(false))
            .entry(druid::platform_menus::common::copy())
            .entry(druid::platform_menus::common::paste()),
    )
}

pub fn main() {
    use WordOrder::*;

    // describe the main window
    let main_window = WindowDesc::new(build_main_widget())
        .menu(make_menu)
        .title(WINDOW_TITLE)
        .window_size((1100.0, 500.0));

    // create the initial app state
    let initial_state = HelloState {
        items: vector![
            HelloRow::new("English", "Hello", "Hello", 99.1, true, SubjectVerbObject),
            HelloRow::new(
                "Français",
                "Bonjour",
                "Bonjour",
                95.0,
                false,
                SubjectVerbObject
            ),
            HelloRow::new("Espanol", "Hola", "Hola", 95.0, true, SubjectVerbObject),
            HelloRow::new("Mandarin", "你好", "nǐ hǎo", 85., false, SubjectVerbObject),
            HelloRow::new("Hindi", "नमस्ते", "namaste", 74., true, SubjectObjectVerb),
            HelloRow::new("Arabic", "مرحبا", "marhabaan", 24., true, SubjectObjectVerb),
            HelloRow::new("Portuguese", "olá", "olá", 30., false, SubjectVerbObject),
            HelloRow::new("Russian", "Привет", "Privet", 42., false, SubjectVerbObject),
            HelloRow::new(
                "Japanese",
                "こんにちは",
                "Kon'nichiwa",
                63.,
                false,
                SubjectObjectVerb
            ),
        ],
        settings: Settings {
            show_headings: ShowHeadings::Both,
            border_thickness: 1.,
            row_fixed: false,
            col_fixed: false,
        },
        table_selection: TableSelection::NoSelection,
    };

    // start the application
    AppLauncher::with_window(main_window)
        .launch(initial_state)
        .expect("Failed to launch application");
}
