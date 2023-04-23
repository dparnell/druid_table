#[macro_use]
mod macros;

mod axis_measure;
mod builder;
mod cells;
mod columns;
mod config;
mod data;
pub mod ensured_pool;
mod headings;
mod lens;
pub mod numbers_table;
mod render_ext;
mod selection;
mod table;
pub mod bindings;

pub use axis_measure::{
    AxisMeasure, AxisPair, FixedAxisMeasure, LogIdx, StoredAxisMeasure, TableAxis, VisIdx,
};
pub use builder::{AxisMeasurementType, ShowHeadings, TableBuilder};
pub use cells::{Cells, CellsDelegate};
pub use columns::{column, CellCtx, CellDelegate, DataCompare, DisplayFactory, WidgetCell};
pub use config::TableConfig;
pub use data::{
    IndexedData, RefreshDiffer, Remap, RemapSpec, Remapper, SlowVectorDiffer, SortDirection,
};
pub use headings::{Headers, HeadersFromIndices, Headings, SuppliedHeaders};
pub use selection::{IndicesSelection, TableSelection};
pub use table::{HeaderBuild, Table, TableSelectionProp};


pub use lens::ReadOnly;

#[macro_use]
extern crate druid;

