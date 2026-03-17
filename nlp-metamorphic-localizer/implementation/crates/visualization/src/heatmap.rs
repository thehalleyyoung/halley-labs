//! Heatmap data structures for visualizing stage×transformation differential matrices.
//!
//! Generates [`HeatmapData`] that can be rendered as colored grids showing how
//! each pipeline stage responds to each metamorphic transformation. Cells encode
//! differential magnitudes and include annotations for values exceeding
//! configurable thresholds.

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use shared_types::{StageId, TransformationId};
use std::fmt;

use crate::color_scale::{Color, ColorScale, LinearScale};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration controlling heatmap appearance and thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapConfig {
    /// Title displayed above the heatmap.
    pub title: String,
    /// Label for the row axis (typically "Stages").
    pub row_axis_label: String,
    /// Label for the column axis (typically "Transformations").
    pub col_axis_label: String,
    /// Threshold above which a cell is flagged as suspicious.
    pub suspicious_threshold: f64,
    /// Threshold above which a cell is flagged as faulty.
    pub faulty_threshold: f64,
    /// Whether to display numeric values inside cells.
    pub show_values: bool,
    /// Number of decimal places for displayed values.
    pub decimal_places: usize,
    /// Whether to normalize values to `[0, 1]`.
    pub normalize: bool,
    /// Cell width in abstract units (used by renderers).
    pub cell_width: f64,
    /// Cell height in abstract units.
    pub cell_height: f64,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            title: "Stage × Transformation Differential Heatmap".into(),
            row_axis_label: "Pipeline Stage".into(),
            col_axis_label: "Transformation".into(),
            suspicious_threshold: 0.3,
            faulty_threshold: 0.7,
            show_values: true,
            decimal_places: 3,
            normalize: false,
            cell_width: 60.0,
            cell_height: 40.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Color mapping
// ---------------------------------------------------------------------------

/// Determines how numeric cell values are mapped to colors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorMapping {
    /// Use a suspiciousness palette (green → yellow → red).
    Suspiciousness,
    /// Use a differential palette (blue → white → red).
    Differential,
    /// Use a custom min/max pair.
    Custom { low: Color, high: Color },
}

impl ColorMapping {
    /// Build a concrete [`LinearScale`] for the given value range.
    pub fn to_scale(&self, min: f64, max: f64) -> LinearScale {
        match self {
            ColorMapping::Suspiciousness => LinearScale::suspiciousness(min, max),
            ColorMapping::Differential => LinearScale::differential(min, max),
            ColorMapping::Custom { low, high } => LinearScale::two_stop(*low, *high, min, max),
        }
    }
}

impl Default for ColorMapping {
    fn default() -> Self {
        Self::Suspiciousness
    }
}

// ---------------------------------------------------------------------------
// Cell / row types
// ---------------------------------------------------------------------------

/// Annotation attached to a heatmap cell that exceeds a threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAnnotation {
    /// Human-readable label (e.g. "suspicious", "faulty").
    pub label: String,
    /// Symbol rendered on the cell (e.g. "!", "✗").
    pub symbol: String,
}

/// A single cell in the heatmap grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapCell {
    /// Raw differential value.
    pub value: f64,
    /// Normalized value in `[0, 1]` (populated after normalization).
    pub normalized_value: f64,
    /// Computed background color.
    pub color: Color,
    /// Contrasting text color for readability.
    pub text_color: Color,
    /// Annotation if the cell exceeds a threshold.
    pub annotation: Option<CellAnnotation>,
    /// Column identifier (transformation id display name).
    pub column_label: String,
}

/// A row of cells corresponding to one pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapRow {
    /// Stage identifier.
    pub stage_id: StageId,
    /// Human-readable label for this row.
    pub label: String,
    /// Cells in column order.
    pub cells: Vec<HeatmapCell>,
    /// Mean value across cells.
    pub row_mean: f64,
    /// Maximum value across cells.
    pub row_max: f64,
}

// ---------------------------------------------------------------------------
// HeatmapData
// ---------------------------------------------------------------------------

/// Complete heatmap data ready for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    /// Configuration used to build this heatmap.
    pub config: HeatmapConfig,
    /// Column headers (transformation labels).
    pub column_labels: Vec<String>,
    /// Rows (one per pipeline stage).
    pub rows: Vec<HeatmapRow>,
    /// Global minimum value.
    pub global_min: f64,
    /// Global maximum value.
    pub global_max: f64,
    /// Color mapping applied.
    pub color_mapping: ColorMapping,
}

impl HeatmapData {
    // ----- construction ----------------------------------------------------

    /// Build a heatmap from a raw matrix of differentials.
    ///
    /// `matrix` maps `(stage_id, transformation_id)` → differential value.
    /// `stage_labels` and `transformation_labels` provide display names.
    pub fn from_matrix(
        matrix: &IndexMap<(StageId, TransformationId), f64>,
        stage_labels: &IndexMap<StageId, String>,
        transformation_labels: &IndexMap<TransformationId, String>,
        config: HeatmapConfig,
        color_mapping: ColorMapping,
    ) -> Self {
        let col_ids: Vec<TransformationId> = transformation_labels.keys().cloned().collect();
        let col_labels: Vec<String> = transformation_labels.values().cloned().collect();

        let (mut g_min, mut g_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in matrix.values() {
            if *v < g_min {
                g_min = *v;
            }
            if *v > g_max {
                g_max = *v;
            }
        }
        if g_min == f64::INFINITY {
            g_min = 0.0;
            g_max = 1.0;
        }

        let scale = color_mapping.to_scale(g_min, g_max);

        let rows = stage_labels
            .iter()
            .map(|(sid, label)| {
                let cells: Vec<HeatmapCell> = col_ids
                    .iter()
                    .zip(col_labels.iter())
                    .map(|(tid, tname)| {
                        let value = matrix
                            .get(&(sid.clone(), tid.clone()))
                            .copied()
                            .unwrap_or(0.0);
                        let norm = scale.normalize(value);
                        let color = scale.map_value(value);
                        let text_color = color.contrast_text();
                        let annotation = Self::annotate(value, &config);
                        HeatmapCell {
                            value,
                            normalized_value: norm,
                            color,
                            text_color,
                            annotation,
                            column_label: tname.clone(),
                        }
                    })
                    .collect();

                let vals: Vec<f64> = cells.iter().map(|c| c.value).collect();
                let row_mean = if vals.is_empty() {
                    0.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                };
                let row_max = vals
                    .iter()
                    .cloned()
                    .max_by_key(|v| OrderedFloat(*v))
                    .unwrap_or(0.0);

                HeatmapRow {
                    stage_id: sid.clone(),
                    label: label.clone(),
                    cells,
                    row_mean,
                    row_max,
                }
            })
            .collect();

        Self {
            config,
            column_labels: col_labels,
            rows,
            global_min: g_min,
            global_max: g_max,
            color_mapping,
        }
    }

    // ----- normalization ---------------------------------------------------

    /// Normalize all cell values to `[0, 1]` range.
    pub fn normalize(&mut self) {
        let range = self.global_max - self.global_min;
        if range.abs() < f64::EPSILON {
            return;
        }
        for row in &mut self.rows {
            for cell in &mut row.cells {
                cell.normalized_value = (cell.value - self.global_min) / range;
            }
        }
    }

    // ----- queries ---------------------------------------------------------

    /// Return cells exceeding the suspicious threshold.
    pub fn suspicious_cells(&self) -> Vec<(&HeatmapRow, &HeatmapCell)> {
        self.rows
            .iter()
            .flat_map(|r| r.cells.iter().map(move |c| (r, c)))
            .filter(|(_, c)| c.value >= self.config.suspicious_threshold)
            .collect()
    }

    /// Return cells exceeding the faulty threshold.
    pub fn faulty_cells(&self) -> Vec<(&HeatmapRow, &HeatmapCell)> {
        self.rows
            .iter()
            .flat_map(|r| r.cells.iter().map(move |c| (r, c)))
            .filter(|(_, c)| c.value >= self.config.faulty_threshold)
            .collect()
    }

    /// Find the row with the highest mean differential.
    pub fn most_suspicious_stage(&self) -> Option<&HeatmapRow> {
        self.rows
            .iter()
            .max_by_key(|r| OrderedFloat(r.row_mean))
    }

    // ----- export ----------------------------------------------------------

    /// Export the heatmap as CSV text.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        // Header
        out.push_str("stage");
        for col in &self.column_labels {
            out.push(',');
            out.push_str(col);
        }
        out.push('\n');
        // Rows
        for row in &self.rows {
            out.push_str(&row.label);
            for cell in &row.cells {
                out.push(',');
                out.push_str(&format!("{:.prec$}", cell.value, prec = self.config.decimal_places));
            }
            out.push('\n');
        }
        out
    }

    /// Export the heatmap as a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "title": self.config.title,
            "columns": self.column_labels,
            "rows": self.rows.iter().map(|r| {
                serde_json::json!({
                    "stage": r.label,
                    "values": r.cells.iter().map(|c| c.value).collect::<Vec<_>>(),
                    "colors": r.cells.iter().map(|c| c.color.to_hex()).collect::<Vec<_>>(),
                    "mean": r.row_mean,
                    "max": r.row_max,
                })
            }).collect::<Vec<_>>(),
            "global_min": self.global_min,
            "global_max": self.global_max,
        })
    }

    /// Export as a JSON string.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.to_json()).unwrap_or_default()
    }

    // ----- helpers ---------------------------------------------------------

    fn annotate(value: f64, config: &HeatmapConfig) -> Option<CellAnnotation> {
        if value >= config.faulty_threshold {
            Some(CellAnnotation {
                label: "faulty".into(),
                symbol: "✗".into(),
            })
        } else if value >= config.suspicious_threshold {
            Some(CellAnnotation {
                label: "suspicious".into(),
                symbol: "!".into(),
            })
        } else {
            None
        }
    }
}

impl fmt::Display for HeatmapData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.config.title)?;
        writeln!(f, "{}", "=".repeat(self.config.title.len()))?;
        write!(f, "{:>16}", "")?;
        for col in &self.column_labels {
            write!(f, " {:>10}", &col[..col.len().min(10)])?;
        }
        writeln!(f)?;
        for row in &self.rows {
            write!(f, "{:>16}", &row.label[..row.label.len().min(16)])?;
            for cell in &row.cells {
                let sym = cell
                    .annotation
                    .as_ref()
                    .map(|a| a.symbol.as_str())
                    .unwrap_or(" ");
                write!(f, " {:>8.3}{}", cell.value, sym)?;
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

    fn sample_heatmap() -> HeatmapData {
        let mut matrix = IndexMap::new();
        let s1 = StageId::new("tokenizer");
        let s2 = StageId::new("parser");
        let t1 = TransformationId::new("passive");
        let t2 = TransformationId::new("negation");

        matrix.insert((s1.clone(), t1.clone()), 0.1);
        matrix.insert((s1.clone(), t2.clone()), 0.8);
        matrix.insert((s2.clone(), t1.clone()), 0.5);
        matrix.insert((s2.clone(), t2.clone()), 0.2);

        let mut stages = IndexMap::new();
        stages.insert(s1, "Tokenizer".into());
        stages.insert(s2, "Parser".into());

        let mut trans = IndexMap::new();
        trans.insert(t1, "Passive".into());
        trans.insert(t2, "Negation".into());

        HeatmapData::from_matrix(&matrix, &stages, &trans, HeatmapConfig::default(), ColorMapping::Suspiciousness)
    }

    #[test]
    fn heatmap_dimensions() {
        let hm = sample_heatmap();
        assert_eq!(hm.rows.len(), 2);
        assert_eq!(hm.column_labels.len(), 2);
        assert_eq!(hm.rows[0].cells.len(), 2);
    }

    #[test]
    fn heatmap_global_bounds() {
        let hm = sample_heatmap();
        assert!((hm.global_min - 0.1).abs() < 1e-9);
        assert!((hm.global_max - 0.8).abs() < 1e-9);
    }

    #[test]
    fn heatmap_annotation_faulty() {
        let hm = sample_heatmap();
        let cell = &hm.rows[0].cells[1]; // 0.8 => faulty
        assert!(cell.annotation.is_some());
        assert_eq!(cell.annotation.as_ref().unwrap().label, "faulty");
    }

    #[test]
    fn heatmap_annotation_suspicious() {
        let hm = sample_heatmap();
        let cell = &hm.rows[1].cells[0]; // 0.5 => suspicious
        assert!(cell.annotation.is_some());
        assert_eq!(cell.annotation.as_ref().unwrap().label, "suspicious");
    }

    #[test]
    fn heatmap_csv_export() {
        let hm = sample_heatmap();
        let csv = hm.to_csv();
        assert!(csv.contains("stage,Passive,Negation"));
        assert!(csv.contains("Tokenizer"));
    }

    #[test]
    fn heatmap_json_export() {
        let hm = sample_heatmap();
        let json = hm.to_json();
        assert!(json["rows"].is_array());
        assert_eq!(json["rows"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn heatmap_suspicious_cells() {
        let hm = sample_heatmap();
        let sus = hm.suspicious_cells();
        assert!(sus.len() >= 2); // 0.5 and 0.8
    }

    #[test]
    fn heatmap_most_suspicious_stage() {
        let hm = sample_heatmap();
        let best = hm.most_suspicious_stage().unwrap();
        // Tokenizer has cells (0.1, 0.8) mean=0.45, Parser (0.5, 0.2) mean=0.35
        assert_eq!(best.label, "Tokenizer");
    }

    #[test]
    fn heatmap_normalize() {
        let mut hm = sample_heatmap();
        hm.normalize();
        for row in &hm.rows {
            for cell in &row.cells {
                assert!(cell.normalized_value >= 0.0 && cell.normalized_value <= 1.0);
            }
        }
    }

    #[test]
    fn heatmap_display() {
        let hm = sample_heatmap();
        let s = format!("{}", hm);
        assert!(s.contains("Tokenizer"));
    }
}
