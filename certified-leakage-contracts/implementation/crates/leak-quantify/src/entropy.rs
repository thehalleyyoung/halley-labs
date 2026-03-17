//! Entropy measures for quantitative information flow analysis.
//!
//! Provides Shannon entropy, min-entropy, guessing entropy, max-leakage,
//! conditional entropy, mutual information, and entropy bounds over discrete
//! probability distributions.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::distribution::Distribution;
use crate::{QuantifyError, QuantifyResult};

// ---------------------------------------------------------------------------
// Shannon Entropy
// ---------------------------------------------------------------------------

/// Shannon entropy H(X) = −Σ p(x) log₂ p(x).
///
/// Measures the average information content of a distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShannonEntropy {
    /// Computed entropy value in bits.
    pub bits: f64,
}

impl ShannonEntropy {
    /// Compute Shannon entropy from a distribution.
    pub fn compute<T: Ord + Clone + fmt::Debug>(dist: &Distribution<T>) -> QuantifyResult<Self> {
        let bits = Self::entropy_of_probs(&dist.probabilities());
        Ok(Self { bits })
    }

    /// Compute Shannon entropy from a raw probability vector.
    pub fn from_probabilities(probs: &[f64]) -> QuantifyResult<Self> {
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(QuantifyError::InvalidDistribution(format!(
                "probabilities sum to {sum}, expected 1.0"
            )));
        }
        Ok(Self {
            bits: Self::entropy_of_probs(probs),
        })
    }

    /// The raw entropy value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }

    /// Shannon entropy of a uniform distribution over `n` outcomes.
    pub fn uniform(n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        (n as f64).log2()
    }

    /// Shannon entropy of a binary (Bernoulli) distribution with parameter `p`.
    pub fn binary(p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return 0.0;
        }
        -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
    }

    fn entropy_of_probs(probs: &[f64]) -> f64 {
        let mut h = 0.0;
        for &p in probs {
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }
}

impl fmt::Display for ShannonEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H = {:.4} bits", self.bits)
    }
}

// ---------------------------------------------------------------------------
// Min-Entropy
// ---------------------------------------------------------------------------

/// Min-entropy H∞(X) = −log₂ max_x p(x).
///
/// Measures the worst-case predictability of a distribution. A conservative
/// (pessimistic) entropy measure often preferred in security contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinEntropy {
    /// Computed min-entropy in bits.
    pub bits: f64,
    /// The maximum probability in the distribution.
    pub max_prob: f64,
}

impl MinEntropy {
    /// Compute min-entropy from a distribution.
    pub fn compute<T: Ord + Clone + fmt::Debug>(dist: &Distribution<T>) -> QuantifyResult<Self> {
        let max_prob = dist.max_probability();
        if max_prob <= 0.0 {
            return Err(QuantifyError::EmptySupport);
        }
        Ok(Self {
            bits: -max_prob.log2(),
            max_prob,
        })
    }

    /// Compute min-entropy from a raw probability vector.
    pub fn from_probabilities(probs: &[f64]) -> QuantifyResult<Self> {
        let max_prob = probs
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        if max_prob <= 0.0 {
            return Err(QuantifyError::EmptySupport);
        }
        Ok(Self {
            bits: -max_prob.log2(),
            max_prob,
        })
    }

    /// The raw min-entropy value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }

    /// Bayes vulnerability V(X) = 2^{-H∞(X)} = max_x p(x).
    pub fn vulnerability(&self) -> f64 {
        self.max_prob
    }
}

impl fmt::Display for MinEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H∞ = {:.4} bits (V = {:.4})", self.bits, self.max_prob)
    }
}

// ---------------------------------------------------------------------------
// Guessing Entropy
// ---------------------------------------------------------------------------

/// Guessing entropy G(X): the expected number of guesses to determine X when
/// guessing in decreasing probability order.
///
/// G(X) = Σᵢ i · p_{(i)} where p_{(1)} ≥ p_{(2)} ≥ … ≥ p_{(n)}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuessingEntropy {
    /// Expected number of guesses.
    pub expected_guesses: f64,
    /// Log₂ of expected guesses (an entropy-like measure).
    pub bits: f64,
}

impl GuessingEntropy {
    /// Compute guessing entropy from a distribution.
    pub fn compute<T: Ord + Clone + fmt::Debug>(dist: &Distribution<T>) -> QuantifyResult<Self> {
        Self::from_probabilities(&dist.probabilities())
    }

    /// Compute guessing entropy from a raw probability vector.
    pub fn from_probabilities(probs: &[f64]) -> QuantifyResult<Self> {
        let mut sorted: Vec<f64> = probs.iter().copied().filter(|&p| p > 0.0).collect();
        if sorted.is_empty() {
            return Err(QuantifyError::EmptySupport);
        }
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let expected_guesses: f64 = sorted
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as f64 + 1.0) * p)
            .sum();

        let bits = if expected_guesses > 0.0 {
            expected_guesses.log2()
        } else {
            0.0
        };

        Ok(Self {
            expected_guesses,
            bits,
        })
    }

    /// The raw guessing entropy value (expected number of guesses).
    pub fn value(&self) -> f64 {
        self.expected_guesses
    }
}

impl fmt::Display for GuessingEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "G = {:.2} guesses ({:.4} bits)",
            self.expected_guesses, self.bits
        )
    }
}

// ---------------------------------------------------------------------------
// Max-Leakage
// ---------------------------------------------------------------------------

/// Max-leakage ℒ(X→Y) = log₂ Σ_y max_x P(y|x).
///
/// An operational measure of the maximum information an adversary can extract
/// about *any* function of the secret, given the observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxLeakage {
    /// Max-leakage in bits.
    pub bits: f64,
    /// The linear sum Σ_y max_x P(y|x) before taking log.
    pub sum_max_column: f64,
}

impl MaxLeakage {
    /// Compute max-leakage from a channel matrix (rows = inputs, columns = outputs).
    ///
    /// Each row must be a valid conditional distribution P(Y|X=x).
    pub fn compute(channel_matrix: &[Vec<f64>]) -> QuantifyResult<Self> {
        if channel_matrix.is_empty() {
            return Err(QuantifyError::EmptySupport);
        }
        let num_outputs = channel_matrix[0].len();
        if num_outputs == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        for row in channel_matrix {
            if row.len() != num_outputs {
                return Err(QuantifyError::DimensionMismatch {
                    expected: num_outputs,
                    got: row.len(),
                });
            }
        }

        let mut sum = 0.0;
        for j in 0..num_outputs {
            let col_max = channel_matrix
                .iter()
                .map(|row| row[j])
                .fold(0.0_f64, f64::max);
            sum += col_max;
        }

        let bits = if sum > 0.0 { sum.log2() } else { 0.0 };

        Ok(Self {
            bits,
            sum_max_column: sum,
        })
    }

    /// The raw max-leakage value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }
}

impl fmt::Display for MaxLeakage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ML = {:.4} bits", self.bits)
    }
}

// ---------------------------------------------------------------------------
// Conditional Entropy
// ---------------------------------------------------------------------------

/// Conditional entropy H(X|Y) = Σ_y P(y) H(X|Y=y).
///
/// Measures the remaining uncertainty about X after observing Y.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalEntropy {
    /// Conditional entropy in bits.
    pub bits: f64,
    /// Individual conditional entropies H(X|Y=y) for each observation y.
    pub per_observation: Vec<f64>,
}

impl ConditionalEntropy {
    /// Compute H(X|Y) from a joint distribution P(X,Y).
    ///
    /// `joint_matrix[x][y]` is P(X=x, Y=y).
    pub fn compute(joint_matrix: &[Vec<f64>]) -> QuantifyResult<Self> {
        if joint_matrix.is_empty() {
            return Err(QuantifyError::EmptySupport);
        }
        let num_y = joint_matrix[0].len();
        if num_y == 0 {
            return Err(QuantifyError::EmptySupport);
        }

        // Compute marginal P(Y=y) = Σ_x P(X=x, Y=y)
        let mut py = vec![0.0; num_y];
        for row in joint_matrix {
            for (j, &p) in row.iter().enumerate() {
                py[j] += p;
            }
        }

        let mut per_observation = Vec::with_capacity(num_y);
        let mut h_x_given_y = 0.0;

        for j in 0..num_y {
            if py[j] <= 0.0 {
                per_observation.push(0.0);
                continue;
            }
            // H(X|Y=y) = −Σ_x P(x|y) log₂ P(x|y)
            let mut h_y = 0.0;
            for row in joint_matrix {
                let p_x_given_y = row[j] / py[j];
                if p_x_given_y > 0.0 {
                    h_y -= p_x_given_y * p_x_given_y.log2();
                }
            }
            per_observation.push(h_y);
            h_x_given_y += py[j] * h_y;
        }

        Ok(Self {
            bits: h_x_given_y,
            per_observation,
        })
    }

    /// Compute from a prior distribution P(X) and a channel matrix P(Y|X).
    pub fn from_channel(prior: &[f64], channel: &[Vec<f64>]) -> QuantifyResult<Self> {
        if prior.len() != channel.len() {
            return Err(QuantifyError::DimensionMismatch {
                expected: prior.len(),
                got: channel.len(),
            });
        }
        // Build joint matrix P(x,y) = P(x) * P(y|x)
        let joint: Vec<Vec<f64>> = prior
            .iter()
            .zip(channel.iter())
            .map(|(&px, row)| row.iter().map(|&pyx| px * pyx).collect())
            .collect();
        Self::compute(&joint)
    }

    /// The raw conditional entropy value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }
}

impl fmt::Display for ConditionalEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H(X|Y) = {:.4} bits", self.bits)
    }
}

// ---------------------------------------------------------------------------
// Mutual Information
// ---------------------------------------------------------------------------

/// Mutual information I(X;Y) = H(X) − H(X|Y).
///
/// Measures the information about X revealed by observing Y.  In the leakage
/// context this is the *average* information leaked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualInformation {
    /// Mutual information in bits.
    pub bits: f64,
    /// H(X) used in the computation.
    pub prior_entropy: f64,
    /// H(X|Y) used in the computation.
    pub conditional_entropy: f64,
}

impl MutualInformation {
    /// Compute I(X;Y) = H(X) − H(X|Y) from a joint distribution matrix.
    pub fn compute(joint_matrix: &[Vec<f64>]) -> QuantifyResult<Self> {
        if joint_matrix.is_empty() {
            return Err(QuantifyError::EmptySupport);
        }

        // Marginal P(X)
        let px: Vec<f64> = joint_matrix.iter().map(|row| row.iter().sum()).collect();

        let prior_entropy = ShannonEntropy::entropy_of_probs(&px);
        let cond = ConditionalEntropy::compute(joint_matrix)?;
        let bits = (prior_entropy - cond.bits).max(0.0);

        Ok(Self {
            bits,
            prior_entropy,
            conditional_entropy: cond.bits,
        })
    }

    /// Compute from a prior distribution and a channel matrix.
    pub fn from_channel(prior: &[f64], channel: &[Vec<f64>]) -> QuantifyResult<Self> {
        let prior_entropy = ShannonEntropy::entropy_of_probs(prior);
        let cond = ConditionalEntropy::from_channel(prior, channel)?;
        let bits = (prior_entropy - cond.bits).max(0.0);

        Ok(Self {
            bits,
            prior_entropy,
            conditional_entropy: cond.bits,
        })
    }

    /// The raw mutual information value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }
}

impl fmt::Display for MutualInformation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I(X;Y) = {:.4} bits", self.bits)
    }
}

// ---------------------------------------------------------------------------
// Entropy Bound
// ---------------------------------------------------------------------------

/// A provable upper or lower bound on an entropy quantity.
///
/// Used for sound over-approximation when exact computation is intractable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyBound {
    /// The bound value in bits.
    pub bits: f64,
    /// Whether this is an upper or lower bound.
    pub kind: BoundKind,
    /// Human-readable description of how the bound was derived.
    pub justification: String,
    /// Confidence in the bound (1.0 = provably sound).
    pub confidence: f64,
}

/// Whether an entropy bound is an upper or lower bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundKind {
    Upper,
    Lower,
}

impl EntropyBound {
    /// Create a new entropy bound.
    pub fn new(bits: f64, kind: BoundKind, justification: impl Into<String>) -> Self {
        Self {
            bits,
            kind,
            justification: justification.into(),
            confidence: 1.0,
        }
    }

    /// Create an upper bound.
    pub fn upper(bits: f64, justification: impl Into<String>) -> Self {
        Self::new(bits, BoundKind::Upper, justification)
    }

    /// Create a lower bound.
    pub fn lower(bits: f64, justification: impl Into<String>) -> Self {
        Self::new(bits, BoundKind::Lower, justification)
    }

    /// Trivial upper bound: H(X) ≤ log₂|X|.
    pub fn log_support(support_size: usize) -> Self {
        let bits = (support_size as f64).log2();
        Self::upper(bits, format!("log₂({support_size}) = {bits:.4}"))
    }

    /// Set a confidence level for this bound.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// The bound value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }

    /// Check whether a measured value satisfies this bound.
    pub fn is_satisfied_by(&self, measured: f64) -> bool {
        match self.kind {
            BoundKind::Upper => measured <= self.bits + 1e-9,
            BoundKind::Lower => measured >= self.bits - 1e-9,
        }
    }

    /// Tighten two bounds of the same kind by taking the tighter one.
    pub fn tighten(&self, other: &Self) -> QuantifyResult<Self> {
        if self.kind != other.kind {
            return Err(QuantifyError::BoundComputationFailed(
                "cannot tighten bounds of different kinds".into(),
            ));
        }
        match self.kind {
            BoundKind::Upper => {
                if self.bits <= other.bits {
                    Ok(self.clone())
                } else {
                    Ok(other.clone())
                }
            }
            BoundKind::Lower => {
                if self.bits >= other.bits {
                    Ok(self.clone())
                } else {
                    Ok(other.clone())
                }
            }
        }
    }
}

impl fmt::Display for EntropyBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sym = match self.kind {
            BoundKind::Upper => "≤",
            BoundKind::Lower => "≥",
        };
        write!(f, "H {sym} {:.4} bits", self.bits)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution;

    #[test]
    fn test_shannon_entropy_uniform() {
        let d = distribution::distribution_from_vec(&[0.25, 0.25, 0.25, 0.25]).unwrap();
        let h = ShannonEntropy::compute(&d).unwrap();
        assert!((h.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_binary() {
        assert!((ShannonEntropy::binary(0.5) - 1.0).abs() < 1e-10);
        assert!((ShannonEntropy::binary(0.0)).abs() < 1e-10);
        assert!((ShannonEntropy::binary(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_min_entropy() {
        let d = distribution::distribution_from_vec(&[0.5, 0.25, 0.25]).unwrap();
        let h = MinEntropy::compute(&d).unwrap();
        assert!((h.value() - 1.0).abs() < 1e-10);
        assert!((h.vulnerability() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_guessing_entropy_point() {
        let d = distribution::distribution_from_vec(&[1.0]).unwrap();
        let g = GuessingEntropy::compute(&d).unwrap();
        assert!((g.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_leakage_deterministic() {
        // Deterministic channel: each input maps to a unique output
        let ch = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ml = MaxLeakage::compute(&ch).unwrap();
        assert!((ml.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_bound_tighten() {
        let b1 = EntropyBound::upper(3.0, "first");
        let b2 = EntropyBound::upper(2.0, "second");
        let tight = b1.tighten(&b2).unwrap();
        assert!((tight.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent: P(x,y) = P(x)*P(y) => I(X;Y) = 0
        let joint = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        let mi = MutualInformation::compute(&joint).unwrap();
        assert!(mi.value() < 1e-10);
    }
}
