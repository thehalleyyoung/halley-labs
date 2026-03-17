//! Correlation analysis.
use serde::{Serialize, Deserialize};

/// Result of a correlation computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult { pub coefficient: f64, pub p_value: f64 }

/// Autocorrelation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationResult { pub lags: Vec<usize>, pub values: Vec<f64> }

/// Cross-correlation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCorrelationResult { pub lags: Vec<i64>, pub values: Vec<f64> }

/// Rank correlation (Spearman).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankCorrelation { pub rho: f64, pub p_value: f64 }

/// Correlation analysis engine.
#[derive(Debug, Clone, Default)]
pub struct CorrelationAnalyzer;
impl CorrelationAnalyzer {
    /// Compute Pearson correlation between two series.
    pub fn pearson(&self, x: &[f64], y: &[f64]) -> CorrelationResult {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 { return CorrelationResult { coefficient: 0.0, p_value: 1.0 }; }
        let mx: f64 = x.iter().sum::<f64>() / n;
        let my: f64 = y.iter().sum::<f64>() / n;
        let cov: f64 = x.iter().zip(y).map(|(a,b)| (a-mx)*(b-my)).sum::<f64>() / n;
        let sx = (x.iter().map(|a| (a-mx).powi(2)).sum::<f64>() / n).sqrt();
        let sy = (y.iter().map(|b| (b-my).powi(2)).sum::<f64>() / n).sqrt();
        let r = if sx * sy > 1e-30 { cov / (sx * sy) } else { 0.0 };
        CorrelationResult { coefficient: r, p_value: 0.0 }
    }
}
