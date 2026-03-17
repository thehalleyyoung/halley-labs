//! Leakage metrics for reporting and comparison.
//!
//! Provides multiple ways to express and compare leakage quantities: bits
//! leaked, multiplicative leakage, vulnerability scores, and guessing advantage.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::bounds::LeakageBound;
use crate::entropy::{MinEntropy, ShannonEntropy};
use crate::{QuantifyError, QuantifyResult};

// ---------------------------------------------------------------------------
// Leakage Metric Trait
// ---------------------------------------------------------------------------

/// A metric that can express information leakage in a specific unit or scale.
pub trait LeakageMetric: fmt::Debug + fmt::Display + Send + Sync {
    /// A human-readable name for this metric.
    fn name(&self) -> &str;

    /// The metric value as a dimensionless f64.
    fn value(&self) -> f64;

    /// Whether this metric indicates zero leakage.
    fn is_zero(&self) -> bool {
        self.value().abs() < 1e-15
    }

    /// Convert this metric to bits leaked.
    fn to_bits_leaked(&self) -> BitsLeaked;

    /// A short summary string for reports.
    fn summary(&self) -> String {
        format!("{}: {:.4}", self.name(), self.value())
    }
}

// ---------------------------------------------------------------------------
// Bits Leaked
// ---------------------------------------------------------------------------

/// Leakage measured in bits of information.
///
/// The most fundamental metric: directly represents the number of bits of
/// secret information that can be extracted by the attacker.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BitsLeaked {
    /// Number of bits leaked.
    pub bits: f64,
}

impl BitsLeaked {
    /// Create from a bit count.
    pub fn new(bits: f64) -> Self {
        Self { bits: bits.max(0.0) }
    }

    /// Create from a leakage bound.
    pub fn from_bound(bound: &LeakageBound) -> Self {
        Self::new(bound.bits)
    }

    /// Zero leakage.
    pub fn zero() -> Self {
        Self { bits: 0.0 }
    }

    /// The number of bits leaked.
    pub fn value(&self) -> f64 {
        self.bits
    }

    /// Convert to multiplicative leakage: 2^bits.
    pub fn to_multiplicative(&self) -> MultiplicativeLeakage {
        MultiplicativeLeakage {
            factor: 2.0_f64.powf(self.bits),
        }
    }

    /// Whether this represents zero leakage.
    pub fn is_zero(&self) -> bool {
        self.bits.abs() < 1e-15
    }

    /// Add two leakage measurements.
    pub fn add(&self, other: &BitsLeaked) -> BitsLeaked {
        BitsLeaked::new(self.bits + other.bits)
    }

    /// Maximum of two leakage measurements.
    pub fn max(&self, other: &BitsLeaked) -> BitsLeaked {
        BitsLeaked::new(self.bits.max(other.bits))
    }
}

impl LeakageMetric for BitsLeaked {
    fn name(&self) -> &str {
        "BitsLeaked"
    }

    fn value(&self) -> f64 {
        self.bits
    }

    fn to_bits_leaked(&self) -> BitsLeaked {
        *self
    }
}

impl fmt::Display for BitsLeaked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} bits", self.bits)
    }
}

// ---------------------------------------------------------------------------
// Multiplicative Leakage
// ---------------------------------------------------------------------------

/// Multiplicative leakage: the factor by which the attacker's advantage
/// increases due to the leakage.
///
/// Equal to 2^(bits leaked). A factor of 1 means no leakage; a factor of 2
/// means the attacker's advantage doubled (1 bit leaked).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MultiplicativeLeakage {
    /// The multiplicative factor (≥ 1).
    pub factor: f64,
}

impl MultiplicativeLeakage {
    /// Create from a multiplicative factor.
    pub fn new(factor: f64) -> Self {
        Self {
            factor: factor.max(1.0),
        }
    }

    /// Create from bits leaked.
    pub fn from_bits(bits: f64) -> Self {
        Self {
            factor: 2.0_f64.powf(bits.max(0.0)),
        }
    }

    /// No leakage (factor = 1).
    pub fn one() -> Self {
        Self { factor: 1.0 }
    }

    /// Convert to bits leaked: log₂(factor).
    pub fn to_bits(&self) -> f64 {
        if self.factor <= 1.0 {
            0.0
        } else {
            self.factor.log2()
        }
    }

    /// The multiplicative factor.
    pub fn value(&self) -> f64 {
        self.factor
    }
}

impl LeakageMetric for MultiplicativeLeakage {
    fn name(&self) -> &str {
        "MultiplicativeLeakage"
    }

    fn value(&self) -> f64 {
        self.factor
    }

    fn is_zero(&self) -> bool {
        (self.factor - 1.0).abs() < 1e-15
    }

    fn to_bits_leaked(&self) -> BitsLeaked {
        BitsLeaked::new(self.to_bits())
    }
}

impl fmt::Display for MultiplicativeLeakage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "×{:.4}", self.factor)
    }
}

// ---------------------------------------------------------------------------
// Vulnerability Score
// ---------------------------------------------------------------------------

/// Vulnerability score V(X) after observing Y: the probability of correctly
/// guessing the secret in one try.
///
/// V = 2^{−H∞(X|Y)}, where H∞(X|Y) is the conditional min-entropy.
/// Range: [1/|X|, 1]. A score of 1 means the secret is fully determined.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VulnerabilityScore {
    /// The posterior vulnerability (probability of correct guess).
    pub posterior: f64,
    /// The prior vulnerability (before observation).
    pub prior: f64,
}

impl VulnerabilityScore {
    /// Create from posterior and prior vulnerability.
    pub fn new(posterior: f64, prior: f64) -> Self {
        Self {
            posterior: posterior.clamp(0.0, 1.0),
            prior: prior.clamp(0.0, 1.0),
        }
    }

    /// Create from a uniform prior over `n` secrets and a posterior vulnerability.
    pub fn from_uniform_prior(n: usize, posterior: f64) -> Self {
        Self {
            posterior: posterior.clamp(0.0, 1.0),
            prior: if n > 0 { 1.0 / n as f64 } else { 0.0 },
        }
    }

    /// Multiplicative increase in vulnerability.
    pub fn multiplicative_leakage(&self) -> f64 {
        if self.prior <= 0.0 {
            return f64::INFINITY;
        }
        self.posterior / self.prior
    }

    /// Leakage in bits: log₂(posterior / prior).
    pub fn leakage_bits(&self) -> f64 {
        let ratio = self.multiplicative_leakage();
        if ratio <= 1.0 {
            0.0
        } else {
            ratio.log2()
        }
    }

    /// Whether the secret is fully determined.
    pub fn is_fully_leaked(&self) -> bool {
        (self.posterior - 1.0).abs() < 1e-9
    }

    /// The posterior vulnerability value.
    pub fn value(&self) -> f64 {
        self.posterior
    }
}

impl LeakageMetric for VulnerabilityScore {
    fn name(&self) -> &str {
        "VulnerabilityScore"
    }

    fn value(&self) -> f64 {
        self.posterior
    }

    fn is_zero(&self) -> bool {
        (self.posterior - self.prior).abs() < 1e-15
    }

    fn to_bits_leaked(&self) -> BitsLeaked {
        BitsLeaked::new(self.leakage_bits())
    }
}

impl fmt::Display for VulnerabilityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "V = {:.4} (prior={:.4}, ×{:.2})",
            self.posterior,
            self.prior,
            self.multiplicative_leakage()
        )
    }
}

// ---------------------------------------------------------------------------
// Guessing Advantage
// ---------------------------------------------------------------------------

/// Guessing advantage: the reduction in the expected number of guesses needed
/// to determine the secret after observing side-channel information.
///
/// Advantage = G(X) − G(X|Y), where G is the guessing entropy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GuessingAdvantage {
    /// Expected guesses before observation.
    pub guesses_before: f64,
    /// Expected guesses after observation.
    pub guesses_after: f64,
    /// The advantage (reduction in guesses).
    pub advantage: f64,
}

impl GuessingAdvantage {
    /// Create from before and after guess counts.
    pub fn new(guesses_before: f64, guesses_after: f64) -> Self {
        let advantage = (guesses_before - guesses_after).max(0.0);
        Self {
            guesses_before,
            guesses_after,
            advantage,
        }
    }

    /// Create from a uniform secret space.
    pub fn from_uniform(secret_size: usize, guesses_after: f64) -> Self {
        let guesses_before = (secret_size as f64 + 1.0) / 2.0;
        Self::new(guesses_before, guesses_after)
    }

    /// The advantage as a fraction of the original guessing entropy.
    pub fn relative_advantage(&self) -> f64 {
        if self.guesses_before <= 0.0 {
            return 0.0;
        }
        self.advantage / self.guesses_before
    }

    /// Advantage in bits: log₂(guesses_before / guesses_after).
    pub fn advantage_bits(&self) -> f64 {
        if self.guesses_after <= 0.0 || self.guesses_before <= 0.0 {
            return 0.0;
        }
        (self.guesses_before / self.guesses_after).log2().max(0.0)
    }

    /// The raw advantage value (reduction in guesses).
    pub fn value(&self) -> f64 {
        self.advantage
    }
}

impl LeakageMetric for GuessingAdvantage {
    fn name(&self) -> &str {
        "GuessingAdvantage"
    }

    fn value(&self) -> f64 {
        self.advantage
    }

    fn is_zero(&self) -> bool {
        self.advantage < 1e-15
    }

    fn to_bits_leaked(&self) -> BitsLeaked {
        BitsLeaked::new(self.advantage_bits())
    }
}

impl fmt::Display for GuessingAdvantage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GA = {:.2} guesses saved ({:.1}%)",
            self.advantage,
            self.relative_advantage() * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_leaked_zero() {
        let bl = BitsLeaked::zero();
        assert!(bl.is_zero());
        assert!((bl.value() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_bits_leaked_to_multiplicative() {
        let bl = BitsLeaked::new(3.0);
        let ml = bl.to_multiplicative();
        assert!((ml.value() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiplicative_leakage_from_bits() {
        let ml = MultiplicativeLeakage::from_bits(1.0);
        assert!((ml.value() - 2.0).abs() < 1e-10);
        assert!((ml.to_bits() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiplicative_leakage_no_leak() {
        let ml = MultiplicativeLeakage::one();
        assert!(ml.is_zero());
    }

    #[test]
    fn test_vulnerability_score() {
        let vs = VulnerabilityScore::from_uniform_prior(256, 1.0 / 16.0);
        assert!((vs.prior - 1.0 / 256.0).abs() < 1e-10);
        assert!((vs.multiplicative_leakage() - 16.0).abs() < 1e-8);
        assert!((vs.leakage_bits() - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_guessing_advantage() {
        let ga = GuessingAdvantage::from_uniform(256, 16.0);
        assert!(ga.value() > 100.0);
        assert!(ga.relative_advantage() > 0.8);
    }

    #[test]
    fn test_bits_leaked_add() {
        let a = BitsLeaked::new(1.5);
        let b = BitsLeaked::new(2.5);
        assert!((a.add(&b).value() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trait_object_compatibility() {
        let bl = BitsLeaked::new(2.0);
        let metric: &dyn LeakageMetric = &bl;
        assert!((metric.value() - 2.0).abs() < 1e-10);
        assert_eq!(metric.name(), "BitsLeaked");
    }
}
