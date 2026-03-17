//! Error bounds: absolute, relative, and ULP-based error metrics.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Types of error measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorMetric {
    /// Absolute error: |computed - exact|
    Absolute,
    /// Relative error: |computed - exact| / |exact|
    Relative,
    /// ULP distance: number of representable values between computed and exact
    Ulp,
    /// Bits of precision lost: -log2(relative_error)
    BitsLost,
    /// Significant digits: -log10(relative_error)
    SignificantDigits,
}

impl fmt::Display for ErrorMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absolute => write!(f, "absolute"),
            Self::Relative => write!(f, "relative"),
            Self::Ulp => write!(f, "ULP"),
            Self::BitsLost => write!(f, "bits_lost"),
            Self::SignificantDigits => write!(f, "sig_digits"),
        }
    }
}

/// An error bound with associated metadata.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ErrorBound {
    /// The error value.
    pub value: f64,
    /// The metric used.
    pub metric: ErrorMetric,
    /// Confidence level (1.0 = guaranteed, <1.0 = statistical).
    pub confidence: f64,
    /// Whether this is a tight bound or an over-approximation.
    pub is_tight: bool,
}

impl ErrorBound {
    /// Create a guaranteed absolute error bound.
    pub fn absolute(value: f64) -> Self {
        Self {
            value,
            metric: ErrorMetric::Absolute,
            confidence: 1.0,
            is_tight: false,
        }
    }

    /// Create a guaranteed relative error bound.
    pub fn relative(value: f64) -> Self {
        Self {
            value,
            metric: ErrorMetric::Relative,
            confidence: 1.0,
            is_tight: false,
        }
    }

    /// Create a ULP error bound.
    pub fn ulp(value: f64) -> Self {
        Self {
            value,
            metric: ErrorMetric::Ulp,
            confidence: 1.0,
            is_tight: false,
        }
    }

    /// Create a bits-lost metric.
    pub fn bits_lost(value: f64) -> Self {
        Self {
            value,
            metric: ErrorMetric::BitsLost,
            confidence: 1.0,
            is_tight: true,
        }
    }

    /// Mark as tight bound.
    pub fn tight(mut self) -> Self {
        self.is_tight = true;
        self
    }

    /// Mark with a confidence level.
    pub fn with_confidence(mut self, conf: f64) -> Self {
        self.confidence = conf;
        self
    }

    /// Whether this error is acceptable (within machine epsilon range).
    pub fn is_acceptable(&self, threshold: f64) -> bool {
        self.value <= threshold
    }

    /// Convert between error metrics given the magnitude and ULP size.
    pub fn convert_to(&self, target: ErrorMetric, magnitude: f64, ulp_size: f64) -> Option<Self> {
        let value = match (self.metric, target) {
            (ErrorMetric::Absolute, ErrorMetric::Relative) => {
                if magnitude == 0.0 {
                    return None;
                }
                self.value / magnitude.abs()
            }
            (ErrorMetric::Absolute, ErrorMetric::Ulp) => {
                if ulp_size == 0.0 {
                    return None;
                }
                self.value / ulp_size
            }
            (ErrorMetric::Relative, ErrorMetric::Absolute) => self.value * magnitude.abs(),
            (ErrorMetric::Relative, ErrorMetric::Ulp) => {
                if ulp_size == 0.0 {
                    return None;
                }
                self.value * magnitude.abs() / ulp_size
            }
            (ErrorMetric::Ulp, ErrorMetric::Absolute) => self.value * ulp_size,
            (ErrorMetric::Ulp, ErrorMetric::Relative) => {
                if magnitude == 0.0 {
                    return None;
                }
                self.value * ulp_size / magnitude.abs()
            }
            (ErrorMetric::Relative, ErrorMetric::BitsLost) => {
                if self.value <= 0.0 {
                    return None;
                }
                -self.value.log2()
            }
            (ErrorMetric::BitsLost, ErrorMetric::Relative) => 2.0_f64.powf(-self.value),
            (a, b) if a == b => self.value,
            _ => return None,
        };

        Some(Self {
            value,
            metric: target,
            confidence: self.confidence,
            is_tight: false,
        })
    }
}

impl fmt::Display for ErrorBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tight_str = if self.is_tight { "=" } else { "≤" };
        write!(f, "{} {} {}", self.metric, tight_str, self.value)?;
        if self.confidence < 1.0 {
            write!(f, " (conf={:.1}%)", self.confidence * 100.0)?;
        }
        Ok(())
    }
}

/// Measurement of actual error between computed and reference values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ErrorMeasurement {
    /// Computed (floating-point) value.
    pub computed: f64,
    /// Reference (high-precision or exact) value.
    pub reference: f64,
    /// Absolute error.
    pub absolute_error: f64,
    /// Relative error (None if reference is zero).
    pub relative_error: Option<f64>,
    /// ULP distance (None if not computable).
    pub ulp_distance: Option<u64>,
    /// Bits of precision lost.
    pub bits_lost: Option<f64>,
}

impl ErrorMeasurement {
    /// Compute error measurement from computed and reference values.
    pub fn compute(computed: f64, reference: f64) -> Self {
        let absolute_error = (computed - reference).abs();

        let relative_error = if reference.abs() > f64::MIN_POSITIVE {
            Some(absolute_error / reference.abs())
        } else if computed.abs() > f64::MIN_POSITIVE {
            Some(absolute_error / computed.abs())
        } else {
            None
        };

        let ulp_distance = if computed.is_finite() && reference.is_finite() {
            let a = crate::ieee754::Ieee754Bits::from_f64(computed);
            let b = crate::ieee754::Ieee754Bits::from_f64(reference);
            a.ulp_distance(b).map(|d| d as u64)
        } else {
            None
        };

        let bits_lost = relative_error.map(|re| {
            if re <= 0.0 {
                0.0
            } else {
                (-re.log2()).max(0.0)
            }
        });

        Self {
            computed,
            reference,
            absolute_error,
            relative_error,
            ulp_distance,
            bits_lost,
        }
    }

    /// Number of correct significant decimal digits.
    pub fn significant_digits(&self) -> Option<f64> {
        self.relative_error.map(|re| {
            if re <= 0.0 {
                15.9 // f64 max
            } else {
                (-re.log10()).max(0.0)
            }
        })
    }

    /// Whether this measurement indicates significant precision loss.
    pub fn is_significant_loss(&self, threshold_bits: f64) -> bool {
        self.bits_lost.map_or(false, |bl| bl < threshold_bits)
    }

    /// Severity level of the error (0-4).
    pub fn severity(&self) -> ErrorSeverity {
        let bits = self.bits_lost.unwrap_or(53.0);
        if bits >= 50.0 {
            ErrorSeverity::Negligible
        } else if bits >= 40.0 {
            ErrorSeverity::Minor
        } else if bits >= 20.0 {
            ErrorSeverity::Moderate
        } else if bits >= 5.0 {
            ErrorSeverity::Severe
        } else {
            ErrorSeverity::Catastrophic
        }
    }
}

/// Severity classification for floating-point errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Error is within expected machine precision bounds.
    Negligible,
    /// Some precision loss but results are still usable.
    Minor,
    /// Significant precision loss; results may be unreliable.
    Moderate,
    /// Severe precision loss; results are likely wrong.
    Severe,
    /// Nearly all precision lost; result is essentially random.
    Catastrophic,
}

impl ErrorSeverity {
    /// Numeric severity level (0 = negligible, 4 = catastrophic).
    pub fn level(self) -> u32 {
        match self {
            Self::Negligible => 0,
            Self::Minor => 1,
            Self::Moderate => 2,
            Self::Severe => 3,
            Self::Catastrophic => 4,
        }
    }

    /// Whether this severity indicates the result should be investigated.
    pub fn needs_attention(self) -> bool {
        matches!(self, Self::Moderate | Self::Severe | Self::Catastrophic)
    }

    /// Whether this severity indicates the result is essentially unusable.
    pub fn is_critical(self) -> bool {
        matches!(self, Self::Severe | Self::Catastrophic)
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Negligible => write!(f, "negligible"),
            Self::Minor => write!(f, "minor"),
            Self::Moderate => write!(f, "moderate"),
            Self::Severe => write!(f, "severe"),
            Self::Catastrophic => write!(f, "CATASTROPHIC"),
        }
    }
}

/// Aggregated error statistics across multiple measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorStatistics {
    pub count: usize,
    pub max_absolute: f64,
    pub max_relative: f64,
    pub mean_absolute: f64,
    pub mean_relative: f64,
    pub max_ulp: u64,
    pub mean_ulp: f64,
    pub worst_bits_lost: f64,
    pub mean_bits_lost: f64,
    pub worst_severity: ErrorSeverity,
    measurements: Vec<ErrorMeasurement>,
}

impl ErrorStatistics {
    /// Create from a collection of error measurements.
    pub fn from_measurements(measurements: Vec<ErrorMeasurement>) -> Self {
        if measurements.is_empty() {
            return Self {
                count: 0,
                max_absolute: 0.0,
                max_relative: 0.0,
                mean_absolute: 0.0,
                mean_relative: 0.0,
                max_ulp: 0,
                mean_ulp: 0.0,
                worst_bits_lost: 53.0,
                mean_bits_lost: 53.0,
                worst_severity: ErrorSeverity::Negligible,
                measurements: Vec::new(),
            };
        }

        let count = measurements.len();
        let mut max_absolute = 0.0_f64;
        let mut sum_absolute = 0.0;
        let mut max_relative = 0.0_f64;
        let mut sum_relative = 0.0;
        let mut rel_count = 0usize;
        let mut max_ulp = 0u64;
        let mut sum_ulp = 0.0;
        let mut ulp_count = 0usize;
        let mut worst_bits = 53.0_f64;
        let mut sum_bits = 0.0;
        let mut bits_count = 0usize;
        let mut worst_severity = ErrorSeverity::Negligible;

        for m in &measurements {
            max_absolute = max_absolute.max(m.absolute_error);
            sum_absolute += m.absolute_error;

            if let Some(re) = m.relative_error {
                max_relative = max_relative.max(re);
                sum_relative += re;
                rel_count += 1;
            }

            if let Some(ulp) = m.ulp_distance {
                max_ulp = max_ulp.max(ulp);
                sum_ulp += ulp as f64;
                ulp_count += 1;
            }

            if let Some(bl) = m.bits_lost {
                worst_bits = worst_bits.min(bl);
                sum_bits += bl;
                bits_count += 1;
            }

            let sev = m.severity();
            if sev > worst_severity {
                worst_severity = sev;
            }
        }

        Self {
            count,
            max_absolute,
            max_relative,
            mean_absolute: sum_absolute / count as f64,
            mean_relative: if rel_count > 0 {
                sum_relative / rel_count as f64
            } else {
                0.0
            },
            max_ulp,
            mean_ulp: if ulp_count > 0 {
                sum_ulp / ulp_count as f64
            } else {
                0.0
            },
            worst_bits_lost: worst_bits,
            mean_bits_lost: if bits_count > 0 {
                sum_bits / bits_count as f64
            } else {
                53.0
            },
            worst_severity,
            measurements,
        }
    }

    /// Get the top-k worst measurements by severity.
    pub fn worst_k(&self, k: usize) -> Vec<&ErrorMeasurement> {
        let mut sorted: Vec<&ErrorMeasurement> = self.measurements.iter().collect();
        sorted.sort_by(|a, b| {
            a.bits_lost
                .unwrap_or(53.0)
                .partial_cmp(&b.bits_lost.unwrap_or(53.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k);
        sorted
    }

    /// Percentile of absolute error.
    pub fn percentile_absolute(&self, p: f64) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        let mut errors: Vec<f64> = self.measurements.iter().map(|m| m.absolute_error).collect();
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (errors.len() - 1) as f64).round() as usize;
        errors[idx.min(errors.len() - 1)]
    }

    /// Check if error exceeds a threshold.
    pub fn exceeds_threshold(&self, metric: ErrorMetric, threshold: f64) -> bool {
        match metric {
            ErrorMetric::Absolute => self.max_absolute > threshold,
            ErrorMetric::Relative => self.max_relative > threshold,
            ErrorMetric::Ulp => self.max_ulp as f64 > threshold,
            ErrorMetric::BitsLost => self.worst_bits_lost < threshold,
            ErrorMetric::SignificantDigits => {
                self.worst_bits_lost * std::f64::consts::LOG10_2 < threshold
            }
        }
    }
}

/// Error comparison between two versions (e.g., original vs repaired).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorComparison {
    pub original: ErrorStatistics,
    pub repaired: ErrorStatistics,
    /// Improvement ratio (>1 means repaired is better).
    pub improvement_ratio: f64,
    /// Number of points where repair improved accuracy.
    pub improved_count: usize,
    /// Number of points where repair made accuracy worse.
    pub degraded_count: usize,
}

impl ErrorComparison {
    /// Compare two sets of error statistics.
    pub fn compare(original: ErrorStatistics, repaired: ErrorStatistics) -> Self {
        let improvement_ratio = if repaired.max_relative > 0.0 {
            original.max_relative / repaired.max_relative
        } else if original.max_relative > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };

        let mut improved = 0;
        let mut degraded = 0;
        let n = original.count.min(repaired.count);
        for i in 0..n {
            let orig_err = original.measurements[i].absolute_error;
            let rep_err = repaired.measurements[i].absolute_error;
            if rep_err < orig_err * 0.99 {
                improved += 1;
            } else if rep_err > orig_err * 1.01 {
                degraded += 1;
            }
        }

        Self {
            original,
            repaired,
            improvement_ratio,
            improved_count: improved,
            degraded_count: degraded,
        }
    }

    /// Whether the repair was overall beneficial.
    pub fn is_beneficial(&self) -> bool {
        self.improvement_ratio > 1.0 && self.degraded_count == 0
    }

    /// Whether the repair was strictly better everywhere.
    pub fn is_strictly_better(&self) -> bool {
        self.improved_count > 0 && self.degraded_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_measurement() {
        let m = ErrorMeasurement::compute(1.0000001, 1.0);
        assert!(m.absolute_error < 2e-7);
        assert!(m.relative_error.unwrap() < 2e-7);
        assert!(m.bits_lost.unwrap() > 20.0);
        assert_eq!(m.severity(), ErrorSeverity::Negligible);
    }

    #[test]
    fn test_catastrophic_error() {
        let m = ErrorMeasurement::compute(1.0, 2.0);
        assert_eq!(m.absolute_error, 1.0);
        assert_eq!(m.relative_error.unwrap(), 0.5);
        assert!(m.severity() >= ErrorSeverity::Severe);
    }

    #[test]
    fn test_error_bound_conversion() {
        let abs = ErrorBound::absolute(1e-10);
        let rel = abs.convert_to(ErrorMetric::Relative, 1.0, 0.0).unwrap();
        assert!((rel.value - 1e-10).abs() < 1e-25);

        let ulp_bound = abs.convert_to(ErrorMetric::Ulp, 1.0, f64::EPSILON).unwrap();
        assert!(ulp_bound.value > 0.0);
    }

    #[test]
    fn test_error_statistics() {
        let measurements = vec![
            ErrorMeasurement::compute(1.0, 1.0 + 1e-10),
            ErrorMeasurement::compute(2.0, 2.0 + 1e-8),
            ErrorMeasurement::compute(3.0, 3.0 + 1e-6),
        ];
        let stats = ErrorStatistics::from_measurements(measurements);
        assert_eq!(stats.count, 3);
        assert!(stats.max_absolute > 9e-7);
        assert!(stats.mean_absolute > 0.0);
    }

    #[test]
    fn test_error_severity() {
        assert!(!ErrorSeverity::Negligible.needs_attention());
        assert!(ErrorSeverity::Moderate.needs_attention());
        assert!(ErrorSeverity::Catastrophic.is_critical());
        assert!(!ErrorSeverity::Minor.is_critical());
    }

    #[test]
    fn test_error_comparison() {
        let orig = vec![
            ErrorMeasurement::compute(1.0, 1.1),
            ErrorMeasurement::compute(2.0, 2.2),
        ];
        let repaired = vec![
            ErrorMeasurement::compute(1.09, 1.1),
            ErrorMeasurement::compute(2.19, 2.2),
        ];
        let cmp = ErrorComparison::compare(
            ErrorStatistics::from_measurements(orig),
            ErrorStatistics::from_measurements(repaired),
        );
        assert!(cmp.improvement_ratio > 1.0);
        assert!(cmp.is_beneficial());
    }

    #[test]
    fn test_error_bound_display() {
        let b = ErrorBound::relative(1e-15).with_confidence(0.95);
        let s = format!("{}", b);
        assert!(s.contains("relative"));
        assert!(s.contains("conf=95.0%"));
    }
}
