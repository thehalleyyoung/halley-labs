//! IEEE 754 rounding modes and rounding operations.

use serde::{Deserialize, Serialize};
use std::fmt;

/// IEEE 754 rounding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default IEEE 754).
    NearestEven,
    /// Round to nearest, ties away from zero.
    NearestAway,
    /// Round toward positive infinity (ceiling).
    TowardPositive,
    /// Round toward negative infinity (floor).
    TowardNegative,
    /// Round toward zero (truncation).
    TowardZero,
    /// Stochastic rounding (for Monte Carlo error analysis).
    Stochastic,
    /// Faithful rounding: result is one of the two nearest representable values.
    Faithful,
}

impl RoundingMode {
    /// Whether this is a directed rounding mode.
    pub fn is_directed(self) -> bool {
        matches!(
            self,
            Self::TowardPositive | Self::TowardNegative | Self::TowardZero
        )
    }

    /// Whether this is a round-to-nearest mode.
    pub fn is_nearest(self) -> bool {
        matches!(self, Self::NearestEven | Self::NearestAway)
    }

    /// Return the opposite directed rounding mode.
    pub fn opposite(self) -> Self {
        match self {
            Self::TowardPositive => Self::TowardNegative,
            Self::TowardNegative => Self::TowardPositive,
            Self::TowardZero => Self::NearestAway,
            other => other,
        }
    }

    /// All standard rounding modes (useful for testing all modes).
    pub fn all_standard() -> &'static [RoundingMode] {
        &[
            Self::NearestEven,
            Self::TowardPositive,
            Self::TowardNegative,
            Self::TowardZero,
        ]
    }
}

impl fmt::Display for RoundingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NearestEven => write!(f, "RNE"),
            Self::NearestAway => write!(f, "RNA"),
            Self::TowardPositive => write!(f, "RU"),
            Self::TowardNegative => write!(f, "RD"),
            Self::TowardZero => write!(f, "RZ"),
            Self::Stochastic => write!(f, "RS"),
            Self::Faithful => write!(f, "RF"),
        }
    }
}

impl Default for RoundingMode {
    fn default() -> Self {
        Self::NearestEven
    }
}

/// Simulated rounding: apply a rounding mode to a real-valued result.
///
/// This simulates the effect of rounding on a value, given the precision
/// (unit in the last place at the result magnitude).
#[derive(Debug, Clone, Copy)]
pub struct RoundingSimulator {
    pub mode: RoundingMode,
}

impl RoundingSimulator {
    pub fn new(mode: RoundingMode) -> Self {
        Self { mode }
    }

    /// Round a value to the nearest representable number at the given ULP size.
    pub fn round(&self, exact: f64, ulp: f64) -> f64 {
        if exact.is_nan() || exact.is_infinite() || ulp <= 0.0 {
            return exact;
        }

        let lo = (exact / ulp).floor() * ulp;
        let hi = lo + ulp;

        match self.mode {
            RoundingMode::NearestEven => {
                let diff_lo = (exact - lo).abs();
                let diff_hi = (hi - exact).abs();
                if diff_lo < diff_hi {
                    lo
                } else if diff_hi < diff_lo {
                    hi
                } else {
                    // Tie: round to even (the one whose last bit is 0)
                    let lo_int = (lo / ulp).round() as i64;
                    if lo_int % 2 == 0 {
                        lo
                    } else {
                        hi
                    }
                }
            }
            RoundingMode::NearestAway => {
                let diff_lo = (exact - lo).abs();
                let diff_hi = (hi - exact).abs();
                if diff_lo < diff_hi {
                    lo
                } else if diff_hi < diff_lo {
                    hi
                } else {
                    // Tie: round away from zero
                    if exact >= 0.0 {
                        hi
                    } else {
                        lo
                    }
                }
            }
            RoundingMode::TowardPositive => hi,
            RoundingMode::TowardNegative => lo,
            RoundingMode::TowardZero => {
                if exact >= 0.0 {
                    lo
                } else {
                    hi
                }
            }
            RoundingMode::Stochastic => {
                // Deterministic simulation with midpoint threshold
                let fraction = (exact - lo) / ulp;
                if fraction >= 0.5 {
                    hi
                } else {
                    lo
                }
            }
            RoundingMode::Faithful => {
                // Return the closer of lo and hi
                if (exact - lo).abs() <= (hi - exact).abs() {
                    lo
                } else {
                    hi
                }
            }
        }
    }

    /// Compute the rounding error for a given exact value and ULP.
    pub fn rounding_error(&self, exact: f64, ulp: f64) -> f64 {
        let rounded = self.round(exact, ulp);
        rounded - exact
    }

    /// Maximum possible rounding error for this mode.
    pub fn max_error(&self, ulp: f64) -> f64 {
        match self.mode {
            RoundingMode::NearestEven | RoundingMode::NearestAway => ulp / 2.0,
            RoundingMode::TowardPositive
            | RoundingMode::TowardNegative
            | RoundingMode::TowardZero => ulp,
            RoundingMode::Faithful => ulp,
            RoundingMode::Stochastic => ulp / 2.0,
        }
    }
}

/// Rounding error model for a computation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoundingErrorModel {
    /// The rounding mode in effect.
    pub mode: RoundingMode,
    /// Bound on the relative rounding error (|δ| ≤ bound).
    pub relative_bound: f64,
    /// Whether this is a first-order error model.
    pub first_order: bool,
}

impl RoundingErrorModel {
    /// Standard model: fl(x op y) = (x op y)(1 + δ) where |δ| ≤ u.
    pub fn standard(mode: RoundingMode, unit_roundoff: f64) -> Self {
        Self {
            mode,
            relative_bound: unit_roundoff,
            first_order: true,
        }
    }

    /// Compose two rounding error models (sequential operations).
    pub fn compose(self, other: Self) -> Self {
        // (1+δ₁)(1+δ₂) ≈ 1 + δ₁ + δ₂ for first order
        let bound = if self.first_order && other.first_order {
            self.relative_bound + other.relative_bound
        } else {
            self.relative_bound + other.relative_bound
                + self.relative_bound * other.relative_bound
        };
        Self {
            mode: self.mode,
            relative_bound: bound,
            first_order: false,
        }
    }

    /// Gamma function: γ_n = nu / (1 - nu) for n rounding errors.
    pub fn gamma(n: usize, unit_roundoff: f64) -> f64 {
        let nu = n as f64 * unit_roundoff;
        if nu >= 1.0 {
            f64::INFINITY
        } else {
            nu / (1.0 - nu)
        }
    }

    /// Tilde-gamma function: ~γ_n ≈ (1 + u)^n - 1 for tighter bounds.
    pub fn tilde_gamma(n: usize, unit_roundoff: f64) -> f64 {
        (1.0 + unit_roundoff).powi(n as i32) - 1.0
    }
}

/// Stochastic rounding perturbation for CADNA/Verificarlo-style analysis.
#[derive(Debug, Clone)]
pub struct StochasticRounding {
    /// Number of stochastic samples to use.
    pub num_samples: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl StochasticRounding {
    pub fn new(num_samples: usize, seed: u64) -> Self {
        Self { num_samples, seed }
    }

    /// Perturb a rounded value stochastically.
    /// Returns the value rounded up or down randomly.
    pub fn perturb(&self, value: f64, ulp: f64, sample_index: usize) -> f64 {
        // Simple deterministic hash-based perturbation for reproducibility
        let hash = self.hash_sample(value, sample_index);
        let lo = (value / ulp).floor() * ulp;
        let hi = lo + ulp;
        let fraction = (value - lo) / ulp;

        // Round up with probability proportional to the fractional part
        if (hash as f64 / u64::MAX as f64) < fraction {
            hi
        } else {
            lo
        }
    }

    fn hash_sample(&self, value: f64, index: usize) -> u64 {
        let bits = value.to_bits();
        let mut h = self.seed;
        h = h.wrapping_mul(6364136223846793005).wrapping_add(bits);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(index as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }

    /// Run a stochastic evaluation: compute f(x) multiple times with different
    /// rounding perturbations and return (mean, std_dev, num_significant_digits).
    pub fn evaluate_stochastic<F: Fn(usize) -> f64>(
        &self,
        compute: F,
    ) -> StochasticResult {
        let mut values = Vec::with_capacity(self.num_samples);
        for i in 0..self.num_samples {
            values.push(compute(i));
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let significant_digits = if mean.abs() < f64::MIN_POSITIVE || std_dev == 0.0 {
            15.0
        } else {
            -(std_dev / mean.abs()).log10()
        };

        StochasticResult {
            mean,
            std_dev,
            significant_digits: significant_digits.max(0.0),
            num_samples: self.num_samples,
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Result of stochastic rounding analysis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StochasticResult {
    pub mean: f64,
    pub std_dev: f64,
    pub significant_digits: f64,
    pub num_samples: usize,
    pub min: f64,
    pub max: f64,
}

impl StochasticResult {
    /// Confidence interval at 95%.
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        let margin = 1.96 * self.std_dev / (self.num_samples as f64).sqrt();
        (self.mean - margin, self.mean + margin)
    }

    /// Number of bits that are significant.
    pub fn significant_bits(&self) -> f64 {
        self.significant_digits * std::f64::consts::LOG2_10
    }

    /// Whether the result has lost significant digits compared to expected precision.
    pub fn has_precision_loss(&self, expected_digits: f64) -> bool {
        self.significant_digits < expected_digits * 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounding_modes() {
        let rne = RoundingMode::NearestEven;
        assert!(rne.is_nearest());
        assert!(!rne.is_directed());

        let ru = RoundingMode::TowardPositive;
        assert!(ru.is_directed());
        assert!(!ru.is_nearest());
    }

    #[test]
    fn test_rounding_simulator_nearest_even() {
        let sim = RoundingSimulator::new(RoundingMode::NearestEven);
        let ulp = 1.0;

        // Value closer to lower
        assert_eq!(sim.round(1.3, ulp), 1.0);
        // Value closer to upper
        assert_eq!(sim.round(1.7, ulp), 2.0);
        // Tie: round to even
        assert_eq!(sim.round(1.5, ulp), 2.0); // 2 is even
        assert_eq!(sim.round(2.5, ulp), 2.0); // 2 is even
    }

    #[test]
    fn test_rounding_simulator_directed() {
        let ulp = 0.5;

        let ru = RoundingSimulator::new(RoundingMode::TowardPositive);
        assert_eq!(ru.round(1.1, ulp), 1.5);

        let rd = RoundingSimulator::new(RoundingMode::TowardNegative);
        assert_eq!(rd.round(1.1, ulp), 1.0);

        let rz = RoundingSimulator::new(RoundingMode::TowardZero);
        assert_eq!(rz.round(1.1, ulp), 1.0);
        assert_eq!(rz.round(-1.1, ulp), -1.0);
    }

    #[test]
    fn test_rounding_error_model() {
        let u = f64::EPSILON / 2.0;
        let model = RoundingErrorModel::standard(RoundingMode::NearestEven, u);
        assert_eq!(model.relative_bound, u);

        let composed = model.compose(model);
        assert!((composed.relative_bound - 2.0 * u).abs() < 1e-30);
    }

    #[test]
    fn test_gamma_function() {
        let u = f64::EPSILON / 2.0;
        let g1 = RoundingErrorModel::gamma(1, u);
        assert!((g1 - u / (1.0 - u)).abs() < 1e-30);

        let g10 = RoundingErrorModel::gamma(10, u);
        assert!(g10 > 10.0 * u);
    }

    #[test]
    fn test_stochastic_result() {
        let result = StochasticResult {
            mean: 1.0,
            std_dev: 1e-10,
            significant_digits: 10.0,
            num_samples: 100,
            min: 0.9999999999,
            max: 1.0000000001,
        };

        let (lo, hi) = result.confidence_interval_95();
        assert!(lo < 1.0);
        assert!(hi > 1.0);
        assert!(!result.has_precision_loss(10.0));
        assert!(result.has_precision_loss(15.0));
    }

    #[test]
    fn test_stochastic_rounding() {
        let sr = StochasticRounding::new(100, 42);
        let result = sr.evaluate_stochastic(|_| 1.0);
        assert!((result.mean - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opposite_rounding() {
        assert_eq!(
            RoundingMode::TowardPositive.opposite(),
            RoundingMode::TowardNegative
        );
        assert_eq!(
            RoundingMode::TowardNegative.opposite(),
            RoundingMode::TowardPositive
        );
    }

    #[test]
    fn test_max_error() {
        let ulp = 1e-10;
        let rne = RoundingSimulator::new(RoundingMode::NearestEven);
        assert_eq!(rne.max_error(ulp), ulp / 2.0);

        let ru = RoundingSimulator::new(RoundingMode::TowardPositive);
        assert_eq!(ru.max_error(ulp), ulp);
    }
}
