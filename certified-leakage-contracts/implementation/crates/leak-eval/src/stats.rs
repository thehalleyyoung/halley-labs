//! Statistical analysis utilities for benchmark evaluation.
//!
//! Provides descriptive statistics, confidence intervals, effect sizes,
//! and correlation analysis for rigorous evaluation of analysis tools.

use serde::{Deserialize, Serialize};

/// Descriptive statistics for a sample of numeric observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    /// Number of observations.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Median value.
    pub median: f64,
    /// 25th percentile.
    pub p25: f64,
    /// 75th percentile.
    pub p75: f64,
}

impl DescriptiveStats {
    /// Compute descriptive statistics from a slice of values.
    ///
    /// Returns `None` if the slice is empty.
    pub fn from_values(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;

        let variance = if count > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[count - 1];
        let median = percentile(&sorted, 50.0);
        let p25 = percentile(&sorted, 25.0);
        let p75 = percentile(&sorted, 75.0);

        Some(Self {
            count,
            mean,
            std_dev,
            min,
            max,
            median,
            p25,
            p75,
        })
    }

    /// Interquartile range (IQR = P75 − P25).
    pub fn iqr(&self) -> f64 {
        self.p75 - self.p25
    }

    /// Coefficient of variation (CV = std_dev / mean).
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < f64::EPSILON {
            0.0
        } else {
            self.std_dev / self.mean.abs()
        }
    }
}

/// Confidence interval for a population parameter.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound of the interval.
    pub lower: f64,
    /// Upper bound of the interval.
    pub upper: f64,
    /// Point estimate (typically the sample mean).
    pub point_estimate: f64,
    /// Confidence level (e.g., 0.95 for 95%).
    pub confidence_level: f64,
}

impl ConfidenceInterval {
    /// Compute a confidence interval for the mean using the normal approximation.
    ///
    /// Uses z-scores for common confidence levels (90%, 95%, 99%).
    pub fn for_mean(stats: &DescriptiveStats, confidence_level: f64) -> Self {
        let z = z_score(confidence_level);
        let margin = z * stats.std_dev / (stats.count as f64).sqrt();

        Self {
            lower: stats.mean - margin,
            upper: stats.mean + margin,
            point_estimate: stats.mean,
            confidence_level,
        }
    }

    /// Width of the confidence interval.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Whether a value falls within this interval.
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// Effect size measurement for comparing two groups.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EffectSize {
    /// Cohen's d value.
    pub cohens_d: f64,
    /// Qualitative interpretation.
    pub interpretation: EffectInterpretation,
}

/// Qualitative interpretation of an effect size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectInterpretation {
    /// |d| < 0.2
    Negligible,
    /// 0.2 ≤ |d| < 0.5
    Small,
    /// 0.5 ≤ |d| < 0.8
    Medium,
    /// |d| ≥ 0.8
    Large,
}

impl EffectSize {
    /// Compute Cohen's d between two groups.
    ///
    /// Uses pooled standard deviation: sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2)).
    pub fn cohens_d(group1: &DescriptiveStats, group2: &DescriptiveStats) -> Self {
        let n1 = group1.count as f64;
        let n2 = group2.count as f64;

        let pooled_var = if n1 + n2 > 2.0 {
            ((n1 - 1.0) * group1.std_dev.powi(2) + (n2 - 1.0) * group2.std_dev.powi(2))
                / (n1 + n2 - 2.0)
        } else {
            0.0
        };
        let pooled_sd = pooled_var.sqrt();

        let d = if pooled_sd > f64::EPSILON {
            (group1.mean - group2.mean) / pooled_sd
        } else {
            0.0
        };

        let interpretation = match d.abs() {
            x if x < 0.2 => EffectInterpretation::Negligible,
            x if x < 0.5 => EffectInterpretation::Small,
            x if x < 0.8 => EffectInterpretation::Medium,
            _ => EffectInterpretation::Large,
        };

        Self {
            cohens_d: d,
            interpretation,
        }
    }
}

/// Correlation analysis between two variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    /// Pearson correlation coefficient r ∈ [-1, 1].
    pub pearson_r: f64,
    /// Coefficient of determination R².
    pub r_squared: f64,
    /// Number of data points.
    pub n: usize,
    /// Label for the x variable.
    pub x_label: String,
    /// Label for the y variable.
    pub y_label: String,
}

impl CorrelationAnalysis {
    /// Compute Pearson correlation between paired observations.
    ///
    /// Returns `None` if fewer than 2 pairs or zero variance.
    pub fn compute(
        x: &[f64],
        y: &[f64],
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Option<Self> {
        let n = x.len().min(y.len());
        if n < 2 {
            return None;
        }

        let x_mean = x[..n].iter().sum::<f64>() / n as f64;
        let y_mean = y[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for i in 0..n {
            let dx = x[i] - x_mean;
            let dy = y[i] - y_mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < f64::EPSILON {
            return None;
        }

        let r = cov / denom;

        Some(Self {
            pearson_r: r,
            r_squared: r * r,
            n,
            x_label: x_label.into(),
            y_label: y_label.into(),
        })
    }

    /// Qualitative strength of the correlation.
    pub fn strength(&self) -> &'static str {
        match self.pearson_r.abs() {
            x if x < 0.1 => "negligible",
            x if x < 0.3 => "weak",
            x if x < 0.5 => "moderate",
            x if x < 0.7 => "strong",
            _ => "very strong",
        }
    }
}

/// Compute the p-th percentile of a sorted slice using linear interpolation.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Return the z-score for a given confidence level.
fn z_score(confidence: f64) -> f64 {
    // Common z-scores; fall back to 1.96 (95%) for uncommon levels.
    if (confidence - 0.90).abs() < 0.005 {
        1.645
    } else if (confidence - 0.95).abs() < 0.005 {
        1.960
    } else if (confidence - 0.99).abs() < 0.005 {
        2.576
    } else {
        1.960
    }
}
