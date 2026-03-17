//! Evaluation metrics for leakage analysis precision, overhead, and scalability.
//!
//! Provides standard information-retrieval metrics (precision, recall, F1) adapted
//! for cache side-channel analysis, plus overhead and scalability measurements.

use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;

/// Precision: fraction of reported leaking sets that truly leak.
///
/// precision = true_positives / (true_positives + false_positives)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Precision {
    /// True positive count.
    pub true_positives: u32,
    /// False positive count.
    pub false_positives: u32,
}

impl Precision {
    /// Compute precision from counts.
    pub fn new(true_positives: u32, false_positives: u32) -> Self {
        Self {
            true_positives,
            false_positives,
        }
    }

    /// The precision value in [0, 1]. Returns 1.0 if both counts are zero.
    pub fn value(&self) -> f64 {
        let total = self.true_positives + self.false_positives;
        if total == 0 {
            1.0
        } else {
            self.true_positives as f64 / total as f64
        }
    }
}

/// Recall: fraction of truly leaking sets that are reported.
///
/// recall = true_positives / (true_positives + false_negatives)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Recall {
    /// True positive count.
    pub true_positives: u32,
    /// False negative count.
    pub false_negatives: u32,
}

impl Recall {
    /// Compute recall from counts.
    pub fn new(true_positives: u32, false_negatives: u32) -> Self {
        Self {
            true_positives,
            false_negatives,
        }
    }

    /// The recall value in [0, 1]. Returns 1.0 if both counts are zero.
    pub fn value(&self) -> f64 {
        let total = self.true_positives + self.false_negatives;
        if total == 0 {
            1.0
        } else {
            self.true_positives as f64 / total as f64
        }
    }
}

/// False positive rate: fraction of non-leaking sets incorrectly flagged.
///
/// fpr = false_positives / (false_positives + true_negatives)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FalsePositiveRate {
    /// False positive count.
    pub false_positives: u32,
    /// True negative count.
    pub true_negatives: u32,
}

impl FalsePositiveRate {
    /// Create from counts.
    pub fn new(false_positives: u32, true_negatives: u32) -> Self {
        Self {
            false_positives,
            true_negatives,
        }
    }

    /// The false positive rate in [0, 1].
    pub fn value(&self) -> f64 {
        let total = self.false_positives + self.true_negatives;
        if total == 0 {
            0.0
        } else {
            self.false_positives as f64 / total as f64
        }
    }
}

/// Tightness ratio: how close the reported bound is to the true leakage.
///
/// tightness = true_leakage / reported_bound (1.0 means perfectly tight).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TightnessRatio {
    /// The true leakage in bits (ground truth).
    pub true_leakage_bits: f64,
    /// The reported upper bound in bits.
    pub reported_bound_bits: f64,
}

impl TightnessRatio {
    /// Create a new tightness ratio.
    pub fn new(true_leakage_bits: f64, reported_bound_bits: f64) -> Self {
        Self {
            true_leakage_bits,
            reported_bound_bits,
        }
    }

    /// The tightness ratio in (0, 1]. Returns 1.0 if reported bound equals true leakage.
    /// Returns 0.0 if the reported bound is zero (degenerate case).
    pub fn value(&self) -> f64 {
        if self.reported_bound_bits <= 0.0 {
            if self.true_leakage_bits <= 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            (self.true_leakage_bits / self.reported_bound_bits).min(1.0)
        }
    }

    /// Whether the bound is sound (reported ≥ true).
    pub fn is_sound(&self) -> bool {
        self.reported_bound_bits >= self.true_leakage_bits - f64::EPSILON
    }
}

/// Performance overhead metrics for the analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadMetrics {
    /// Wall-clock time of the analysis in seconds.
    pub analysis_time_secs: f64,
    /// Peak memory usage in megabytes.
    pub peak_memory_mb: f64,
    /// Number of instructions analyzed.
    pub instructions_analyzed: u64,
    /// Instructions analyzed per second.
    pub throughput: f64,
}

impl OverheadMetrics {
    /// Compute overhead from a benchmark result.
    pub fn from_result(result: &BenchmarkResult, instruction_count: u64) -> Self {
        let secs = result.elapsed.as_secs_f64();
        let throughput = if secs > 0.0 {
            instruction_count as f64 / secs
        } else {
            0.0
        };
        let peak_memory_mb = result
            .peak_memory_bytes
            .map(|b| b as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);

        Self {
            analysis_time_secs: secs,
            peak_memory_mb,
            instructions_analyzed: instruction_count,
            throughput,
        }
    }

    /// Create from raw values.
    pub fn new(time_secs: f64, memory_mb: f64, instructions: u64) -> Self {
        let throughput = if time_secs > 0.0 {
            instructions as f64 / time_secs
        } else {
            0.0
        };
        Self {
            analysis_time_secs: time_secs,
            peak_memory_mb: memory_mb,
            instructions_analyzed: instructions,
            throughput,
        }
    }
}

/// Scalability profile measuring how analysis cost grows with input size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityProfile {
    /// Data points: (input_size, analysis_time_secs).
    pub data_points: Vec<(u64, f64)>,
    /// Estimated complexity class (e.g., "O(n)", "O(n²)", "O(n log n)").
    pub estimated_complexity: Option<String>,
    /// Coefficient of the growth model.
    pub growth_coefficient: Option<f64>,
}

impl ScalabilityProfile {
    /// Create a new empty scalability profile.
    pub fn new() -> Self {
        Self {
            data_points: Vec::new(),
            estimated_complexity: None,
            growth_coefficient: None,
        }
    }

    /// Add a data point.
    pub fn add_point(&mut self, input_size: u64, time_secs: f64) {
        self.data_points.push((input_size, time_secs));
    }

    /// Number of data points collected.
    pub fn point_count(&self) -> usize {
        self.data_points.len()
    }

    /// Estimate complexity class from collected data points.
    ///
    /// Uses a simple ratio-based heuristic; for production use, fit regression models.
    pub fn estimate_complexity(&mut self) {
        if self.data_points.len() < 3 {
            return;
        }

        let mut sorted = self.data_points.clone();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        // Compute growth ratios between consecutive points.
        let mut ratios = Vec::new();
        for w in sorted.windows(2) {
            let size_ratio = w[1].0 as f64 / w[0].0 as f64;
            let time_ratio = w[1].1 / w[0].1;
            if size_ratio > 1.0 && time_ratio > 0.0 {
                let exponent = time_ratio.ln() / size_ratio.ln();
                ratios.push(exponent);
            }
        }

        if ratios.is_empty() {
            return;
        }

        let avg_exponent = ratios.iter().sum::<f64>() / ratios.len() as f64;
        self.growth_coefficient = Some(avg_exponent);

        self.estimated_complexity = Some(if avg_exponent < 0.5 {
            "O(1)".into()
        } else if avg_exponent < 1.2 {
            "O(n)".into()
        } else if avg_exponent < 1.6 {
            "O(n log n)".into()
        } else if avg_exponent < 2.5 {
            "O(n²)".into()
        } else {
            format!("O(n^{:.1})", avg_exponent)
        });
    }
}

impl Default for ScalabilityProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregator for computing metrics across a set of benchmark results.
#[derive(Debug)]
pub struct MetricAggregator {
    /// Precision values collected.
    precisions: Vec<Precision>,
    /// Recall values collected.
    recalls: Vec<Recall>,
    /// Tightness ratios collected.
    tightness_ratios: Vec<TightnessRatio>,
    /// Overhead measurements collected.
    overheads: Vec<OverheadMetrics>,
}

impl MetricAggregator {
    /// Create a new empty aggregator.
    pub fn new() -> Self {
        Self {
            precisions: Vec::new(),
            recalls: Vec::new(),
            tightness_ratios: Vec::new(),
            overheads: Vec::new(),
        }
    }

    /// Record a precision measurement.
    pub fn add_precision(&mut self, p: Precision) {
        self.precisions.push(p);
    }

    /// Record a recall measurement.
    pub fn add_recall(&mut self, r: Recall) {
        self.recalls.push(r);
    }

    /// Record a tightness ratio.
    pub fn add_tightness(&mut self, t: TightnessRatio) {
        self.tightness_ratios.push(t);
    }

    /// Record an overhead measurement.
    pub fn add_overhead(&mut self, o: OverheadMetrics) {
        self.overheads.push(o);
    }

    /// Mean precision across all recorded values.
    pub fn mean_precision(&self) -> f64 {
        if self.precisions.is_empty() {
            return 1.0;
        }
        self.precisions.iter().map(|p| p.value()).sum::<f64>() / self.precisions.len() as f64
    }

    /// Mean recall across all recorded values.
    pub fn mean_recall(&self) -> f64 {
        if self.recalls.is_empty() {
            return 1.0;
        }
        self.recalls.iter().map(|r| r.value()).sum::<f64>() / self.recalls.len() as f64
    }

    /// Mean tightness ratio.
    pub fn mean_tightness(&self) -> f64 {
        if self.tightness_ratios.is_empty() {
            return 1.0;
        }
        self.tightness_ratios.iter().map(|t| t.value()).sum::<f64>() / self.tightness_ratios.len() as f64
    }

    /// Mean analysis throughput (instructions/sec).
    pub fn mean_throughput(&self) -> f64 {
        if self.overheads.is_empty() {
            return 0.0;
        }
        self.overheads.iter().map(|o| o.throughput).sum::<f64>() / self.overheads.len() as f64
    }

    /// F1 score: harmonic mean of mean precision and mean recall.
    pub fn f1_score(&self) -> f64 {
        let p = self.mean_precision();
        let r = self.mean_recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

impl Default for MetricAggregator {
    fn default() -> Self {
        Self::new()
    }
}
