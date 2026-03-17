//! Visualization data structures and renderers for NLP metamorphic fault localization.
//!
//! This crate generates data structures and output formats for visualizing
//! fault localization results, including:
//!
//! - [`heatmap`]: Stage×transformation differential heatmaps
//! - [`pipeline_diagram`]: Annotated NLP pipeline diagrams
//! - [`differential_plot`]: Per-stage differential distribution plots
//! - [`causal_graph`]: Causal influence graphs from DCE/IE analysis
//! - [`color_scale`]: Color mapping utilities for numeric-to-color conversions
//! - [`svg_renderer`]: SVG output generation
//! - [`ascii_renderer`]: Terminal-friendly ASCII art output
//! - [`chart_data`]: Export formats for Vega-Lite, Plotly, and D3.js

pub mod ascii_renderer;
pub mod causal_graph;
pub mod chart_data;
pub mod color_scale;
pub mod differential_plot;
pub mod heatmap;
pub mod pipeline_diagram;
pub mod svg_renderer;

pub use ascii_renderer::{AsciiCanvas, AsciiRenderer, AsciiStyle, BorderStyle};
pub use causal_graph::{
    CausalEdge, CausalGraphData, CausalNode, CausalPathway, GraphLayout, GraphLayoutEngine,
};
pub use chart_data::{
    ChartExporter, D3DataBinding, PlotlyTrace, VegaLiteSpec,
};
pub use color_scale::{
    CategoricalPalette, Color, ColorScale, ColorStop, DivergingScale, LinearScale,
};
pub use differential_plot::{
    AxisConfig, DataPoint, DifferentialPlotData, PlotConfig, PlotSeries, PlotType,
};
pub use heatmap::{
    ColorMapping, HeatmapCell, HeatmapConfig, HeatmapData, HeatmapRow,
};
pub use pipeline_diagram::{
    DiagramEdge, DiagramLayout, DiagramNode, FaultMarker, FaultSeverity, PipelineDiagram,
};
pub use svg_renderer::{SvgDocument, SvgElement, SvgRenderer, SvgStyle};
