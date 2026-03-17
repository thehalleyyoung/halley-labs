//! Differential plot data structures for charting per-stage differentials.
//!
//! Produces [`DifferentialPlotData`] that can drive line plots, bar charts, or
//! violin plots showing how individual pipeline stages respond across
//! metamorphic transformations, including statistical annotations (mean,
//! confidence intervals, outliers).

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use shared_types::{ConfidenceInterval, StageId, TransformationId};
use std::fmt;

// ---------------------------------------------------------------------------
// Plot type & axis config
// ---------------------------------------------------------------------------

/// The type of plot to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlotType {
    /// Line plot connecting per-transformation differentials.
    Line,
    /// Grouped bar chart.
    Bar,
    /// Box-and-whisker plot of distributions.
    Box,
    /// Violin plot (kernel density + box).
    Violin,
    /// Scatter plot of individual data points.
    Scatter,
}

impl Default for PlotType {
    fn default() -> Self {
        Self::Bar
    }
}

/// Configuration for a single axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    /// Axis label.
    pub label: String,
    /// Minimum bound (`None` = auto).
    pub min: Option<f64>,
    /// Maximum bound (`None` = auto).
    pub max: Option<f64>,
    /// Tick step (`None` = auto).
    pub tick_step: Option<f64>,
    /// Whether to show grid lines.
    pub grid: bool,
}

impl AxisConfig {
    /// Create an axis with just a label.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.into(),
            min: None,
            max: None,
            tick_step: None,
            grid: true,
        }
    }

    /// Set explicit bounds.
    pub fn with_bounds(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Effective minimum, falling back to a default.
    pub fn effective_min(&self, data_min: f64) -> f64 {
        self.min.unwrap_or(data_min)
    }

    /// Effective maximum, falling back to a default.
    pub fn effective_max(&self, data_max: f64) -> f64 {
        self.max.unwrap_or(data_max)
    }
}

// ---------------------------------------------------------------------------
// Plot configuration
// ---------------------------------------------------------------------------

/// Overall plot configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotConfig {
    /// Plot title.
    pub title: String,
    /// X-axis configuration.
    pub x_axis: AxisConfig,
    /// Y-axis configuration.
    pub y_axis: AxisConfig,
    /// Plot type.
    pub plot_type: PlotType,
    /// Whether to show a legend.
    pub show_legend: bool,
    /// Whether to annotate statistical measures.
    pub show_statistics: bool,
    /// Width in abstract units.
    pub width: f64,
    /// Height in abstract units.
    pub height: f64,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: "Per-Stage Differential Distribution".into(),
            x_axis: AxisConfig::new("Transformation"),
            y_axis: AxisConfig::new("Differential (Δ)"),
            plot_type: PlotType::default(),
            show_legend: true,
            show_statistics: true,
            width: 800.0,
            height: 500.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Data point & series
// ---------------------------------------------------------------------------

/// A single data point in a plot series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// X-axis value or category index.
    pub x: f64,
    /// Y-axis value (the differential).
    pub y: f64,
    /// Optional label (e.g. transformation name).
    pub label: Option<String>,
    /// Optional error bar (± half-width).
    pub error: Option<f64>,
}

impl DataPoint {
    /// Create a labeled data point.
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            label: None,
            error: None,
        }
    }

    /// Attach a label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Attach an error margin.
    pub fn with_error(mut self, e: f64) -> Self {
        self.error = Some(e);
        self
    }
}

/// Statistical summary for a data series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStatistics {
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Median value.
    pub median: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// 95% confidence interval.
    pub ci_95: Option<ConfidenceInterval>,
}

impl SeriesStatistics {
    /// Compute statistics from a slice of values.
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                median: 0.0,
                min: 0.0,
                max: 0.0,
                ci_95: None,
            };
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = var.sqrt();

        let mut sorted: Vec<OrderedFloat<f64>> =
            values.iter().copied().map(OrderedFloat).collect();
        sorted.sort();
        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1].0 + sorted[mid].0) / 2.0
        } else {
            sorted[sorted.len() / 2].0
        };

        let min = sorted.first().unwrap().0;
        let max = sorted.last().unwrap().0;

        let ci_95 = if values.len() >= 2 {
            let se = std_dev / (n.sqrt());
            Some(ConfidenceInterval {
                lower: mean - 1.96 * se,
                upper: mean + 1.96 * se,
                confidence_level: 0.95,
            })
        } else {
            None
        };

        Self {
            mean,
            std_dev,
            median,
            min,
            max,
            ci_95,
        }
    }
}

/// A named data series (one per pipeline stage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotSeries {
    /// Series identifier (stage id).
    pub stage_id: StageId,
    /// Human-readable series name.
    pub name: String,
    /// Data points.
    pub points: Vec<DataPoint>,
    /// Computed statistics.
    pub statistics: Option<SeriesStatistics>,
    /// Series color hex.
    pub color: String,
}

impl PlotSeries {
    /// Create a new plot series.
    pub fn new(stage_id: StageId, name: &str, color: &str) -> Self {
        Self {
            stage_id,
            name: name.into(),
            points: Vec::new(),
            statistics: None,
            color: color.into(),
        }
    }

    /// Add a data point.
    pub fn add_point(&mut self, point: DataPoint) {
        self.points.push(point);
    }

    /// Compute and store descriptive statistics.
    pub fn compute_statistics(&mut self) {
        let values: Vec<f64> = self.points.iter().map(|p| p.y).collect();
        self.statistics = Some(SeriesStatistics::from_values(&values));
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the series has no data points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ---------------------------------------------------------------------------
// DifferentialPlotData
// ---------------------------------------------------------------------------

/// Complete data for a differential plot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPlotData {
    /// Plot configuration.
    pub config: PlotConfig,
    /// Data series (one per pipeline stage).
    pub series: Vec<PlotSeries>,
    /// Transformation labels for the x-axis categories.
    pub categories: Vec<String>,
}

impl DifferentialPlotData {
    /// Create a new plot with the given configuration.
    pub fn new(config: PlotConfig) -> Self {
        Self {
            config,
            series: Vec::new(),
            categories: Vec::new(),
        }
    }

    /// Build a differential plot from a per-stage differential map.
    ///
    /// `data` maps `(stage_id, transformation_id)` → differential value.
    pub fn from_differentials(
        data: &indexmap::IndexMap<(StageId, TransformationId), f64>,
        stage_labels: &indexmap::IndexMap<StageId, String>,
        transformation_labels: &indexmap::IndexMap<TransformationId, String>,
        config: PlotConfig,
    ) -> Self {
        let palette = crate::color_scale::CategoricalPalette::pipeline_stages();
        let categories: Vec<String> = transformation_labels.values().cloned().collect();
        let trans_ids: Vec<TransformationId> = transformation_labels.keys().cloned().collect();

        let mut series_list = Vec::new();
        for (idx, (sid, slabel)) in stage_labels.iter().enumerate() {
            let color = palette.by_index(idx).to_hex();
            let mut series = PlotSeries::new(sid.clone(), slabel, &color);

            for (cat_idx, tid) in trans_ids.iter().enumerate() {
                let val = data
                    .get(&(sid.clone(), tid.clone()))
                    .copied()
                    .unwrap_or(0.0);
                series.add_point(
                    DataPoint::new(cat_idx as f64, val)
                        .with_label(&categories[cat_idx]),
                );
            }
            series.compute_statistics();
            series_list.push(series);
        }

        Self {
            config,
            series: series_list,
            categories,
        }
    }

    /// Add a pre-built series.
    pub fn add_series(&mut self, series: PlotSeries) {
        self.series.push(series);
    }

    /// Global y-range across all series.
    pub fn y_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for s in &self.series {
            for p in &s.points {
                if p.y < min {
                    min = p.y;
                }
                if p.y > max {
                    max = p.y;
                }
            }
        }
        if min == f64::INFINITY {
            (0.0, 1.0)
        } else {
            (min, max)
        }
    }

    /// Return the series with the highest mean differential.
    pub fn most_divergent_series(&self) -> Option<&PlotSeries> {
        self.series.iter().max_by_key(|s| {
            OrderedFloat(
                s.statistics
                    .as_ref()
                    .map(|st| st.mean)
                    .unwrap_or(0.0),
            )
        })
    }

    /// Export as JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "title": self.config.title,
            "plot_type": format!("{:?}", self.config.plot_type),
            "categories": self.categories,
            "series": self.series.iter().map(|s| serde_json::json!({
                "name": s.name,
                "color": s.color,
                "points": s.points.iter().map(|p| serde_json::json!({
                    "x": p.x,
                    "y": p.y,
                    "label": p.label,
                    "error": p.error,
                })).collect::<Vec<_>>(),
                "statistics": s.statistics.as_ref().map(|st| serde_json::json!({
                    "mean": st.mean,
                    "std_dev": st.std_dev,
                    "median": st.median,
                    "min": st.min,
                    "max": st.max,
                })),
            })).collect::<Vec<_>>(),
        })
    }
}

impl fmt::Display for DifferentialPlotData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.config.title)?;
        writeln!(f, "{}", "─".repeat(self.config.title.len()))?;
        for s in &self.series {
            write!(f, "  {:<20}", s.name)?;
            if let Some(ref st) = s.statistics {
                write!(
                    f,
                    " mean={:.4}  sd={:.4}  median={:.4}",
                    st.mean, st.std_dev, st.median
                )?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    fn sample_data() -> (
        IndexMap<(StageId, TransformationId), f64>,
        IndexMap<StageId, String>,
        IndexMap<TransformationId, String>,
    ) {
        let s1 = StageId::new("tok");
        let s2 = StageId::new("par");
        let t1 = TransformationId::new("passive");
        let t2 = TransformationId::new("negation");

        let mut data = IndexMap::new();
        data.insert((s1.clone(), t1.clone()), 0.15);
        data.insert((s1.clone(), t2.clone()), 0.80);
        data.insert((s2.clone(), t1.clone()), 0.45);
        data.insert((s2.clone(), t2.clone()), 0.10);

        let mut sl = IndexMap::new();
        sl.insert(s1, "Tokenizer".into());
        sl.insert(s2, "Parser".into());

        let mut tl = IndexMap::new();
        tl.insert(t1, "Passive".into());
        tl.insert(t2, "Negation".into());

        (data, sl, tl)
    }

    #[test]
    fn from_differentials_creates_series() {
        let (data, sl, tl) = sample_data();
        let plot = DifferentialPlotData::from_differentials(&data, &sl, &tl, PlotConfig::default());
        assert_eq!(plot.series.len(), 2);
        assert_eq!(plot.categories.len(), 2);
    }

    #[test]
    fn series_statistics_computed() {
        let (data, sl, tl) = sample_data();
        let plot = DifferentialPlotData::from_differentials(&data, &sl, &tl, PlotConfig::default());
        for s in &plot.series {
            assert!(s.statistics.is_some());
        }
    }

    #[test]
    fn y_range_correct() {
        let (data, sl, tl) = sample_data();
        let plot = DifferentialPlotData::from_differentials(&data, &sl, &tl, PlotConfig::default());
        let (min, max) = plot.y_range();
        assert!((min - 0.10).abs() < 1e-9);
        assert!((max - 0.80).abs() < 1e-9);
    }

    #[test]
    fn most_divergent_series() {
        let (data, sl, tl) = sample_data();
        let plot = DifferentialPlotData::from_differentials(&data, &sl, &tl, PlotConfig::default());
        let best = plot.most_divergent_series().unwrap();
        // Tokenizer: mean of 0.15 and 0.80 = 0.475 vs Parser 0.275
        assert_eq!(best.name, "Tokenizer");
    }

    #[test]
    fn statistics_from_values() {
        let stats = SeriesStatistics::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((stats.mean - 3.0).abs() < 1e-9);
        assert!((stats.median - 3.0).abs() < 1e-9);
        assert!((stats.min - 1.0).abs() < 1e-9);
        assert!((stats.max - 5.0).abs() < 1e-9);
        assert!(stats.ci_95.is_some());
    }

    #[test]
    fn statistics_empty() {
        let stats = SeriesStatistics::from_values(&[]);
        assert!((stats.mean).abs() < 1e-9);
    }

    #[test]
    fn to_json_structure() {
        let (data, sl, tl) = sample_data();
        let plot = DifferentialPlotData::from_differentials(&data, &sl, &tl, PlotConfig::default());
        let j = plot.to_json();
        assert!(j["series"].is_array());
        assert_eq!(j["categories"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn display_format() {
        let (data, sl, tl) = sample_data();
        let plot = DifferentialPlotData::from_differentials(&data, &sl, &tl, PlotConfig::default());
        let s = format!("{}", plot);
        assert!(s.contains("Tokenizer"));
        assert!(s.contains("mean="));
    }

    #[test]
    fn data_point_with_error() {
        let p = DataPoint::new(1.0, 2.0).with_error(0.5).with_label("test");
        assert_eq!(p.error, Some(0.5));
        assert_eq!(p.label.as_deref(), Some("test"));
    }
}
