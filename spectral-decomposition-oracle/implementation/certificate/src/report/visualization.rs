//! Data preparation for visualization.
//!
//! Produces serializable structs for scatter plots, histograms, heatmaps,
//! box plots, and ROC curves. Does not perform rendering—only data preparation.

use crate::report::comparison::InstanceSummary;
use crate::report::DecomposabilityTier;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A single data point for scatter plots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterPoint {
    pub x: f64,
    pub y: f64,
    pub label: String,
    pub color_group: String,
}

/// Scatter plot data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterPlotData {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub points: Vec<ScatterPoint>,
}

/// Histogram bin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBin {
    pub bin_start: f64,
    pub bin_end: f64,
    pub count: usize,
    pub frequency: f64,
}

/// Histogram data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub title: String,
    pub x_label: String,
    pub bins: Vec<HistogramBin>,
    pub total_count: usize,
}

impl HistogramData {
    pub fn from_values(title: &str, x_label: &str, values: &[f64], num_bins: usize) -> Self {
        if values.is_empty() || num_bins == 0 {
            return Self {
                title: title.to_string(),
                x_label: x_label.to_string(),
                bins: Vec::new(),
                total_count: 0,
            };
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let bin_width = if range > 1e-15 {
            range / num_bins as f64
        } else {
            1.0
        };

        let mut counts = vec![0usize; num_bins];
        for &v in values {
            let idx = ((v - min) / bin_width) as usize;
            let idx = idx.min(num_bins - 1);
            counts[idx] += 1;
        }

        let total = values.len();
        let bins: Vec<HistogramBin> = (0..num_bins)
            .map(|i| {
                let bin_start = min + i as f64 * bin_width;
                let bin_end = bin_start + bin_width;
                HistogramBin {
                    bin_start,
                    bin_end,
                    count: counts[i],
                    frequency: counts[i] as f64 / total as f64,
                }
            })
            .collect();

        Self {
            title: title.to_string(),
            x_label: x_label.to_string(),
            bins,
            total_count: total,
        }
    }
}

/// Heatmap cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapCell {
    pub row: String,
    pub col: String,
    pub value: f64,
}

/// Heatmap data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    pub title: String,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    pub cells: Vec<HeatmapCell>,
}

/// Box plot data for one group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxPlotGroup {
    pub group_name: String,
    pub min: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
    pub n: usize,
}

impl BoxPlotGroup {
    pub fn from_values(name: &str, values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                group_name: name.to_string(),
                min: 0.0,
                q25: 0.0,
                median: 0.0,
                q75: 0.0,
                max: 0.0,
                outliers: Vec::new(),
                n: 0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();

        let q25 = sorted[n / 4];
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let q75 = sorted[(3 * n) / 4];
        let iqr = q75 - q25;
        let lower_fence = q25 - 1.5 * iqr;
        let upper_fence = q75 + 1.5 * iqr;

        let outliers: Vec<f64> = sorted
            .iter()
            .filter(|&&v| v < lower_fence || v > upper_fence)
            .cloned()
            .collect();

        let min_non_outlier = sorted
            .iter()
            .find(|&&v| v >= lower_fence)
            .cloned()
            .unwrap_or(sorted[0]);
        let max_non_outlier = sorted
            .iter()
            .rev()
            .find(|&&v| v <= upper_fence)
            .cloned()
            .unwrap_or(sorted[n - 1]);

        Self {
            group_name: name.to_string(),
            min: min_non_outlier,
            q25,
            median,
            q75,
            max: max_non_outlier,
            outliers,
            n,
        }
    }
}

/// Box plot data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxPlotData {
    pub title: String,
    pub y_label: String,
    pub groups: Vec<BoxPlotGroup>,
}

/// ROC curve point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocPoint {
    pub fpr: f64,
    pub tpr: f64,
    pub threshold: f64,
}

/// ROC curve data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocCurveData {
    pub title: String,
    pub points: Vec<RocPoint>,
    pub auc: f64,
}

impl RocCurveData {
    /// Compute ROC curve from predicted scores and true labels.
    pub fn compute(scores: &[f64], labels: &[bool]) -> Self {
        if scores.len() != labels.len() || scores.is_empty() {
            return Self {
                title: "ROC Curve".to_string(),
                points: vec![RocPoint {
                    fpr: 0.0,
                    tpr: 0.0,
                    threshold: 1.0,
                }],
                auc: 0.5,
            };
        }

        let mut indexed: Vec<(f64, bool)> =
            scores.iter().zip(labels.iter()).map(|(&s, &l)| (s, l)).collect();
        indexed.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_pos = labels.iter().filter(|&&l| l).count() as f64;
        let total_neg = labels.iter().filter(|&&l| !l).count() as f64;

        if total_pos < 1.0 || total_neg < 1.0 {
            return Self {
                title: "ROC Curve".to_string(),
                points: vec![
                    RocPoint {
                        fpr: 0.0,
                        tpr: 0.0,
                        threshold: 1.0,
                    },
                    RocPoint {
                        fpr: 1.0,
                        tpr: 1.0,
                        threshold: 0.0,
                    },
                ],
                auc: 0.5,
            };
        }

        let mut points = Vec::new();
        points.push(RocPoint {
            fpr: 0.0,
            tpr: 0.0,
            threshold: indexed[0].0 + 0.01,
        });

        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut auc = 0.0;
        let mut prev_fpr = 0.0;
        let mut prev_tpr = 0.0;

        for (score, label) in &indexed {
            if *label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let tpr = tp / total_pos;
            let fpr = fp / total_neg;
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            points.push(RocPoint {
                fpr,
                tpr,
                threshold: *score,
            });
            prev_fpr = fpr;
            prev_tpr = tpr;
        }

        Self {
            title: "ROC Curve".to_string(),
            points,
            auc,
        }
    }
}

/// All visualization data for an analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub spectral_ratio_vs_gap: Option<ScatterPlotData>,
    pub l3_bound_histogram: Option<HistogramData>,
    pub structure_vs_method_heatmap: Option<HeatmapData>,
    pub feature_distributions: Vec<BoxPlotData>,
    pub futility_roc: Option<RocCurveData>,
}

impl VisualizationData {
    /// Build all visualization data from instance summaries.
    pub fn from_instances(instances: &[InstanceSummary]) -> Self {
        let scatter = Self::build_scatter(instances);
        let histogram = Self::build_histogram(instances);
        let heatmap = Self::build_heatmap(instances);
        let box_plots = Self::build_box_plots(instances);
        let roc = Self::build_roc(instances);

        Self {
            spectral_ratio_vs_gap: scatter,
            l3_bound_histogram: histogram,
            structure_vs_method_heatmap: heatmap,
            feature_distributions: box_plots,
            futility_roc: roc,
        }
    }

    fn build_scatter(instances: &[InstanceSummary]) -> Option<ScatterPlotData> {
        let points: Vec<ScatterPoint> = instances
            .iter()
            .filter_map(|inst| {
                let x = inst.spectral_ratio?;
                let y = inst.actual_gap.or(inst.l3_bound)?;
                Some(ScatterPoint {
                    x,
                    y,
                    label: inst.name.clone(),
                    color_group: inst.tier.to_string(),
                })
            })
            .collect();

        if points.is_empty() {
            return None;
        }

        Some(ScatterPlotData {
            title: "Spectral Ratio vs Decomposition Gap".to_string(),
            x_label: "δ²/γ² (spectral ratio)".to_string(),
            y_label: "Gap".to_string(),
            points,
        })
    }

    fn build_histogram(instances: &[InstanceSummary]) -> Option<HistogramData> {
        let l3_bounds: Vec<f64> = instances.iter().filter_map(|i| i.l3_bound).collect();
        if l3_bounds.is_empty() {
            return None;
        }
        Some(HistogramData::from_values(
            "L3 Bound Distribution",
            "L3 Bound",
            &l3_bounds,
            20,
        ))
    }

    fn build_heatmap(instances: &[InstanceSummary]) -> Option<HeatmapData> {
        let mut structure_tier: IndexMap<(String, String), usize> = IndexMap::new();
        let mut structures = std::collections::BTreeSet::new();
        let mut tiers = std::collections::BTreeSet::new();

        for inst in instances {
            let key = (inst.structure_type.clone(), inst.tier.to_string());
            *structure_tier.entry(key).or_insert(0) += 1;
            structures.insert(inst.structure_type.clone());
            tiers.insert(inst.tier.to_string());
        }

        if structures.is_empty() {
            return None;
        }

        let row_labels: Vec<String> = structures.into_iter().collect();
        let col_labels: Vec<String> = tiers.into_iter().collect();

        let cells: Vec<HeatmapCell> = structure_tier
            .iter()
            .map(|((s, t), &count)| HeatmapCell {
                row: s.clone(),
                col: t.clone(),
                value: count as f64,
            })
            .collect();

        Some(HeatmapData {
            title: "Structure Type vs Decomposability Tier".to_string(),
            row_labels,
            col_labels,
            cells,
        })
    }

    fn build_box_plots(instances: &[InstanceSummary]) -> Vec<BoxPlotData> {
        let mut plots = Vec::new();

        // L3 bounds by tier
        let mut by_tier: IndexMap<String, Vec<f64>> = IndexMap::new();
        for inst in instances {
            if let Some(l3) = inst.l3_bound {
                by_tier.entry(inst.tier.to_string()).or_default().push(l3);
            }
        }

        if !by_tier.is_empty() {
            let groups: Vec<BoxPlotGroup> = by_tier
                .iter()
                .map(|(tier, values)| BoxPlotGroup::from_values(tier, values))
                .collect();
            plots.push(BoxPlotData {
                title: "L3 Bound by Tier".to_string(),
                y_label: "L3 Bound".to_string(),
                groups,
            });
        }

        // Spectral ratio by tier
        let mut sr_by_tier: IndexMap<String, Vec<f64>> = IndexMap::new();
        for inst in instances {
            if let Some(sr) = inst.spectral_ratio {
                sr_by_tier
                    .entry(inst.tier.to_string())
                    .or_default()
                    .push(sr);
            }
        }

        if !sr_by_tier.is_empty() {
            let groups: Vec<BoxPlotGroup> = sr_by_tier
                .iter()
                .map(|(tier, values)| BoxPlotGroup::from_values(tier, values))
                .collect();
            plots.push(BoxPlotData {
                title: "Spectral Ratio by Tier".to_string(),
                y_label: "Spectral Ratio".to_string(),
                groups,
            });
        }

        plots
    }

    fn build_roc(instances: &[InstanceSummary]) -> Option<RocCurveData> {
        let data: Vec<(f64, bool)> = instances
            .iter()
            .filter_map(|inst| {
                let score = inst.futility_score?;
                let is_hard = inst.tier == DecomposabilityTier::Hard
                    || inst.tier == DecomposabilityTier::Intractable;
                Some((score, is_hard))
            })
            .collect();

        if data.len() < 5 {
            return None;
        }

        let scores: Vec<f64> = data.iter().map(|(s, _)| *s).collect();
        let labels: Vec<bool> = data.iter().map(|(_, l)| *l).collect();

        Some(RocCurveData::compute(&scores, &labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instances() -> Vec<InstanceSummary> {
        vec![
            InstanceSummary {
                name: "easy1".to_string(),
                structure_type: "BlockAngular".to_string(),
                tier: DecomposabilityTier::Easy,
                l3_bound: Some(1.0),
                t2_bound: Some(5.0),
                t2_vacuous: false,
                spectral_ratio: Some(0.1),
                futility_score: Some(0.2),
                actual_gap: Some(0.5),
                num_blocks: Some(3),
                crossing_edges: Some(5),
                verified: Some(true),
            },
            InstanceSummary {
                name: "hard1".to_string(),
                structure_type: "Staircase".to_string(),
                tier: DecomposabilityTier::Hard,
                l3_bound: Some(50.0),
                t2_bound: Some(1000.0),
                t2_vacuous: true,
                spectral_ratio: Some(0.8),
                futility_score: Some(0.9),
                actual_gap: Some(30.0),
                num_blocks: Some(2),
                crossing_edges: Some(50),
                verified: Some(true),
            },
        ]
    }

    #[test]
    fn test_histogram_from_values() {
        let hist = HistogramData::from_values("test", "x", &[1.0, 2.0, 3.0, 4.0, 5.0], 5);
        assert_eq!(hist.total_count, 5);
        assert_eq!(hist.bins.len(), 5);
        let total_freq: f64 = hist.bins.iter().map(|b| b.frequency).sum();
        assert!((total_freq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_empty() {
        let hist = HistogramData::from_values("test", "x", &[], 5);
        assert_eq!(hist.total_count, 0);
    }

    #[test]
    fn test_box_plot_from_values() {
        let bp = BoxPlotGroup::from_values("test", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(bp.n, 5);
        assert!((bp.median - 3.0).abs() < 1e-10);
        assert!(bp.min <= bp.q25);
        assert!(bp.q25 <= bp.median);
        assert!(bp.median <= bp.q75);
        assert!(bp.q75 <= bp.max);
    }

    #[test]
    fn test_box_plot_empty() {
        let bp = BoxPlotGroup::from_values("test", &[]);
        assert_eq!(bp.n, 0);
    }

    #[test]
    fn test_roc_curve_compute() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        let labels = vec![true, true, true, true, true, false, false, false, false, false];
        let roc = RocCurveData::compute(&scores, &labels);
        assert!(roc.auc > 0.9);
        assert!(!roc.points.is_empty());
    }

    #[test]
    fn test_roc_curve_random() {
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![true, false, true, false];
        let roc = RocCurveData::compute(&scores, &labels);
        assert!(roc.auc >= 0.0 && roc.auc <= 1.0);
    }

    #[test]
    fn test_roc_curve_empty() {
        let roc = RocCurveData::compute(&[], &[]);
        assert!((roc.auc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_visualization_data_from_instances() {
        let instances = make_instances();
        let viz = VisualizationData::from_instances(&instances);
        assert!(viz.spectral_ratio_vs_gap.is_some());
        assert!(viz.l3_bound_histogram.is_some());
        assert!(viz.structure_vs_method_heatmap.is_some());
    }

    #[test]
    fn test_scatter_plot() {
        let instances = make_instances();
        let scatter = VisualizationData::build_scatter(&instances).unwrap();
        assert_eq!(scatter.points.len(), 2);
        assert!(scatter.title.contains("Spectral Ratio"));
    }

    #[test]
    fn test_heatmap() {
        let instances = make_instances();
        let heatmap = VisualizationData::build_heatmap(&instances).unwrap();
        assert!(!heatmap.row_labels.is_empty());
        assert!(!heatmap.col_labels.is_empty());
        assert!(!heatmap.cells.is_empty());
    }

    #[test]
    fn test_box_plots() {
        let instances = make_instances();
        let plots = VisualizationData::build_box_plots(&instances);
        assert!(!plots.is_empty());
    }

    #[test]
    fn test_visualization_no_instances() {
        let viz = VisualizationData::from_instances(&[]);
        assert!(viz.spectral_ratio_vs_gap.is_none());
        assert!(viz.l3_bound_histogram.is_none());
    }

    #[test]
    fn test_histogram_single_value() {
        let hist = HistogramData::from_values("test", "x", &[5.0, 5.0, 5.0], 10);
        assert_eq!(hist.total_count, 3);
    }
}
