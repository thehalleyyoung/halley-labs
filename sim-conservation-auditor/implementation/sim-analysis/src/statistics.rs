//! Descriptive statistics and statistical tests.
use serde::{Serialize, Deserialize};

/// Descriptive statistics of a sample.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DescriptiveStats { pub mean: f64, pub variance: f64, pub std_dev: f64, pub min: f64, pub max: f64, pub median: f64, pub n: usize }
impl DescriptiveStats {
    /// Compute descriptive statistics for a data vector.
    pub fn from_data(data: &[f64]) -> Self {
        if data.is_empty() { return Self::default(); }
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let mut sorted = data.to_vec();
        sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let median = if n % 2 == 0 { (sorted[n/2-1] + sorted[n/2]) / 2.0 } else { sorted[n/2] };
        Self { mean, variance: var, std_dev: var.sqrt(), min: sorted[0], max: sorted[n-1], median, n }
    }
}

/// Histogram bin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBin { pub lower: f64, pub upper: f64, pub count: usize }

/// Histogram of values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram { pub bins: Vec<HistogramBin>, pub total: usize }

/// Result of a bootstrap analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult { pub estimate: f64, pub std_error: f64, pub ci_lower: f64, pub ci_upper: f64 }

/// Kolmogorov-Smirnov test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KsTestResult { pub statistic: f64, pub p_value: f64 }

/// Anderson-Darling test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndersonDarlingResult { pub statistic: f64, pub critical_values: Vec<f64> }

/// Statistical analysis engine.
#[derive(Debug, Clone, Default)]
pub struct StatisticalAnalyzer;
impl StatisticalAnalyzer {
    /// Compute descriptive statistics.
    pub fn describe(&self, data: &[f64]) -> DescriptiveStats { DescriptiveStats::from_data(data) }
}
