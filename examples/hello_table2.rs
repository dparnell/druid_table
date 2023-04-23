use std::fmt::{Debug, Display, Formatter};

use druid_table::{
    AxisMeasurementType, CellCtx, LogIdx, ShowHeadings, Table, TableAxis, TableBuilder,
    TableSelection, TableSelectionProp, WidgetCell,
};

use druid::im::{vector, Vector};
use druid::lens::{Identity, Index};
use druid::theme::PLACEHOLDER_COLOR;
use druid::widget::{Button, Checkbox, Container, CrossAxisAlignment, Flex, Label, LineBreaking, MainAxisAlignment, Padding, RadioGroup, RawLabel, SizedBox, Stepper, ViewSwitcher};
use druid::{theme, Color};
use druid::{
    AppLauncher, Data, Env, Lens, LensExt, LocalizedString, Widget, WidgetExt, WindowDesc,
};
use std::cell::RefCell;
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
        |sh, _, _| {
            let table = build_table(sh.clone()).lens(HelloState::items);
            table
                /*
                .binding(
                    HelloState::table_selection
                        .bind(TableSelectionProp::default())
                        .back(),
                ) */
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

fn build_table(settings: Settings) -> Table<Vector<HelloRow>> {
    let measurement_type = if settings.col_fixed {
        AxisMeasurementType::Uniform
    } else {
        AxisMeasurementType::Individual
    };

    let col_del = WidgetCell::<_, [String; 2], _>::new(
        |ctx| {
            let level = if let CellCtx::Header(h) = ctx {
                h.level
            } else {
                LogIdx(0)
            };
            RawLabel::new()
                .with_text_color(theme::LABEL_COLOR)
                .lens(Index::new(level.0))
        },
        || Identity,
    );

    TableBuilder::new_custom_col(col_del)
        .measuring_axis(TableAxis::Rows, measurement_type)
        .measuring_axis(TableAxis::Columns, measurement_type)
        .headings(settings.show_headings)
        .border(settings.border_thickness)
        .with_column(
            ["Language".to_string(), "Group".to_string()],
            WidgetCell::text(|| HelloRow::lang),
        )
        .with_column(
            ["Greeting".to_string(), "Group".to_string()],
            WidgetCell::text(|| HelloRow::greeting),
        )
        .with_column(
            ["Greeting2".to_string(), "Group".to_string()],
            WidgetCell::text(|| HelloRow::greeting),
        )
        .with_column(
            ["Greeting3".to_string(), "Group2".to_string()],
            WidgetCell::text(|| HelloRow::greeting),
        )
        .with_column(
            ["Greeting4".to_string(), "Group2".to_string()],
            WidgetCell::text(|| HelloRow::greeting),
        )
        .with_column(
            ["Greeting5".to_string(), "Group".to_string()],
            WidgetCell::text(|| HelloRow::greeting),
        )
        .build()
}

pub fn main() {
    use WordOrder::*;

    // describe the main window
    let main_window = WindowDesc::new(build_main_widget())
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
        .use_simple_logger()
        .launch(initial_state)
        .expect("Failed to launch application");
}
