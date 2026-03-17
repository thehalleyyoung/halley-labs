//! Discrete probability distributions for quantitative information flow analysis.
//!
//! Provides generic discrete distributions with operations needed for channel
//! capacity computation and leakage quantification.

use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::{QuantifyError, QuantifyResult};

// ---------------------------------------------------------------------------
// Core Distribution
// ---------------------------------------------------------------------------

/// A discrete probability distribution over a finite set of outcomes.
///
/// Internally stored as a mapping from outcome to probability mass.
/// Invariant: all probabilities are non-negative and sum to 1 (within tolerance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution<T: Ord + Clone> {
    masses: BTreeMap<T, f64>,
    total: f64,
}

/// Tolerance for floating-point probability comparisons.
const PROB_TOLERANCE: f64 = 1e-9;

impl<T: Ord + Clone + fmt::Debug> Distribution<T> {
    /// Create a distribution from an iterator of (outcome, probability) pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (T, f64)>) -> QuantifyResult<Self> {
        let mut masses = BTreeMap::new();
        let mut total = 0.0;

        for (outcome, prob) in pairs {
            if prob < -PROB_TOLERANCE {
                return Err(QuantifyError::InvalidDistribution(format!(
                    "negative probability {prob} for outcome {outcome:?}"
                )));
            }
            let prob = prob.max(0.0);
            if prob > 0.0 {
                *masses.entry(outcome).or_insert(0.0) += prob;
                total += prob;
            }
        }

        if masses.is_empty() {
            return Err(QuantifyError::EmptySupport);
        }

        Ok(Self { masses, total })
    }

    /// Create a distribution and normalize so probabilities sum to 1.
    pub fn from_weights(pairs: impl IntoIterator<Item = (T, f64)>) -> QuantifyResult<Self> {
        let mut dist = Self::from_pairs(pairs)?;
        dist.normalize()?;
        Ok(dist)
    }

    /// Normalize the distribution in-place so probabilities sum to 1.
    pub fn normalize(&mut self) -> QuantifyResult<()> {
        if self.total <= 0.0 {
            return Err(QuantifyError::InvalidDistribution(
                "cannot normalize zero-weight distribution".into(),
            ));
        }
        let scale = 1.0 / self.total;
        for prob in self.masses.values_mut() {
            *prob *= scale;
        }
        self.total = 1.0;
        Ok(())
    }

    /// Check whether this distribution is valid (probabilities sum to ~1).
    pub fn is_valid(&self) -> bool {
        let sum: f64 = self.masses.values().sum();
        (sum - 1.0).abs() < 1e-6 && self.masses.values().all(|&p| p >= -PROB_TOLERANCE)
    }

    /// Return the probability of a specific outcome.
    pub fn probability(&self, outcome: &T) -> f64 {
        self.masses.get(outcome).copied().unwrap_or(0.0)
    }

    /// Return the support set (outcomes with positive probability).
    pub fn support(&self) -> Vec<&T> {
        self.masses.keys().collect()
    }

    /// Return the number of outcomes with positive probability.
    pub fn support_size(&self) -> usize {
        self.masses.len()
    }

    /// Iterate over (outcome, probability) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&T, f64)> {
        self.masses.iter().map(|(k, &v)| (k, v))
    }

    /// Return just the probability values.
    pub fn probabilities(&self) -> Vec<f64> {
        self.masses.values().copied().collect()
    }

    /// Return the maximum probability (mode probability).
    pub fn max_probability(&self) -> f64 {
        self.masses.values().copied().fold(0.0_f64, f64::max)
    }

    /// Return the minimum nonzero probability.
    pub fn min_probability(&self) -> f64 {
        self.masses
            .values()
            .copied()
            .filter(|&p| p > 0.0)
            .fold(f64::INFINITY, f64::min)
    }

    /// Total mass (should be 1.0 if normalized).
    pub fn total_mass(&self) -> f64 {
        self.total
    }

    /// Map outcomes through a function, summing probabilities for collisions.
    pub fn map_outcomes<U: Ord + Clone + fmt::Debug>(
        &self,
        f: impl Fn(&T) -> U,
    ) -> QuantifyResult<Distribution<U>> {
        let pairs: Vec<(U, f64)> = self.masses.iter().map(|(k, &v)| (f(k), v)).collect();
        Distribution::from_pairs(pairs)
    }

    /// Filter to outcomes satisfying a predicate, then renormalize.
    pub fn condition(
        &self,
        predicate: impl Fn(&T) -> bool,
    ) -> QuantifyResult<Self> {
        let pairs: Vec<(T, f64)> = self
            .masses
            .iter()
            .filter(|(k, _)| predicate(k))
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        if pairs.is_empty() {
            return Err(QuantifyError::InvalidDistribution(
                "conditioning on zero-probability event".into(),
            ));
        }

        let mut dist = Self::from_pairs(pairs)?;
        dist.normalize()?;
        Ok(dist)
    }

    /// Compute the statistical distance (total variation) to another distribution.
    pub fn total_variation_distance(&self, other: &Self) -> f64
    where
        T: Hash,
    {
        let mut all_outcomes: BTreeMap<&T, (f64, f64)> = BTreeMap::new();
        for (k, &v) in &self.masses {
            all_outcomes.entry(k).or_insert((0.0, 0.0)).0 = v;
        }
        for (k, &v) in &other.masses {
            all_outcomes.entry(k).or_insert((0.0, 0.0)).1 = v;
        }
        let sum: f64 = all_outcomes.values().map(|(a, b)| (a - b).abs()).sum();
        sum / 2.0
    }

    /// Compute the KL divergence D(self || other).
    /// Returns +inf if other assigns zero probability to any outcome in self's support.
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        let mut kl = 0.0;
        for (outcome, &p) in &self.masses {
            if p <= 0.0 {
                continue;
            }
            let q = other.probability(outcome);
            if q <= 0.0 {
                return f64::INFINITY;
            }
            kl += p * (p / q).ln();
        }
        kl / std::f64::consts::LN_2
    }

    /// Compute the inner product <self, other> = sum_x self(x) * other(x).
    pub fn inner_product(&self, other: &Self) -> f64 {
        let mut result = 0.0;
        for (outcome, &p) in &self.masses {
            let q = other.probability(outcome);
            result += p * q;
        }
        result
    }
}

impl<T: Ord + Clone + fmt::Debug + fmt::Display> fmt::Display for Distribution<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dist{{")?;
        let mut first = true;
        for (outcome, &prob) in &self.masses {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{outcome}: {prob:.4}")?;
            first = false;
        }
        write!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// Uniform Distribution
// ---------------------------------------------------------------------------

/// A uniform distribution over a finite set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformDistribution {
    pub size: usize,
}

impl UniformDistribution {
    /// Create a uniform distribution over `n` outcomes.
    pub fn new(n: usize) -> QuantifyResult<Self> {
        if n == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        Ok(Self { size: n })
    }

    /// The probability of each outcome.
    pub fn probability(&self) -> f64 {
        1.0 / self.size as f64
    }

    /// Shannon entropy of a uniform distribution = log2(n).
    pub fn entropy(&self) -> f64 {
        (self.size as f64).log2()
    }

    /// Min-entropy of a uniform distribution = log2(n).
    pub fn min_entropy(&self) -> f64 {
        (self.size as f64).log2()
    }

    /// Convert to a generic Distribution<usize>.
    pub fn to_distribution(&self) -> QuantifyResult<Distribution<usize>> {
        let p = self.probability();
        Distribution::from_pairs((0..self.size).map(|i| (i, p)))
    }

    /// Number of outcomes.
    pub fn support_size(&self) -> usize {
        self.size
    }
}

impl fmt::Display for UniformDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Uniform({})", self.size)
    }
}

// ---------------------------------------------------------------------------
// Point (Dirac) Distribution
// ---------------------------------------------------------------------------

/// A point distribution concentrated on a single outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointDistribution<T: Ord + Clone> {
    pub point: T,
}

impl<T: Ord + Clone + fmt::Debug> PointDistribution<T> {
    pub fn new(point: T) -> Self {
        Self { point }
    }

    /// The probability of the point is 1.
    pub fn probability(&self, outcome: &T) -> f64 {
        if outcome == &self.point {
            1.0
        } else {
            0.0
        }
    }

    /// Shannon entropy of a point distribution is 0.
    pub fn entropy(&self) -> f64 {
        0.0
    }

    /// Convert to a generic Distribution.
    pub fn to_distribution(&self) -> QuantifyResult<Distribution<T>> {
        Distribution::from_pairs(std::iter::once((self.point.clone(), 1.0)))
    }
}

impl<T: Ord + Clone + fmt::Display> fmt::Display for PointDistribution<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point({})", self.point)
    }
}

// ---------------------------------------------------------------------------
// Conditional Distribution
// ---------------------------------------------------------------------------

/// A conditional distribution P(Y|X) represented as a family of distributions
/// indexed by the conditioning variable X.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalDistribution<X: Ord + Clone, Y: Ord + Clone> {
    /// For each value of X, a distribution over Y.
    conditionals: BTreeMap<X, Distribution<Y>>,
}

impl<X: Ord + Clone + fmt::Debug, Y: Ord + Clone + fmt::Debug> ConditionalDistribution<X, Y> {
    /// Create from a mapping of X values to distributions over Y.
    pub fn new(conditionals: BTreeMap<X, Distribution<Y>>) -> QuantifyResult<Self> {
        if conditionals.is_empty() {
            return Err(QuantifyError::EmptySupport);
        }
        for (_x, dist) in &conditionals {
            if !dist.is_valid() {
                return Err(QuantifyError::InvalidDistribution(
                    "conditional distribution not valid".into(),
                ));
            }
        }
        Ok(Self { conditionals })
    }

    /// Get P(Y|X=x).
    pub fn given(&self, x: &X) -> Option<&Distribution<Y>> {
        self.conditionals.get(x)
    }

    /// Return the set of conditioning values.
    pub fn conditioning_values(&self) -> Vec<&X> {
        self.conditionals.keys().collect()
    }

    /// Number of conditioning values.
    pub fn num_conditions(&self) -> usize {
        self.conditionals.len()
    }

    /// Compute the joint distribution P(X,Y) given a prior P(X).
    pub fn joint(
        &self,
        prior: &Distribution<X>,
    ) -> QuantifyResult<JointDistribution<X, Y>> {
        let mut pairs = Vec::new();
        for (x, &px) in &prior.masses {
            if let Some(py_given_x) = self.conditionals.get(x) {
                for (y, py) in py_given_x.iter() {
                    pairs.push(((x.clone(), y.clone()), px * py));
                }
            }
        }
        JointDistribution::from_pairs(pairs)
    }

    /// Iterate over (x, P(Y|X=x)) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&X, &Distribution<Y>)> {
        self.conditionals.iter()
    }
}

// ---------------------------------------------------------------------------
// Joint Distribution
// ---------------------------------------------------------------------------

/// A joint distribution P(X,Y) over pairs of outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointDistribution<X: Ord + Clone, Y: Ord + Clone> {
    joint: Distribution<(X, Y)>,
}

impl<X: Ord + Clone + fmt::Debug, Y: Ord + Clone + fmt::Debug> JointDistribution<X, Y> {
    /// Create from (outcome_pair, probability) pairs.
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = ((X, Y), f64)>,
    ) -> QuantifyResult<Self> {
        let mut dist = Distribution::from_pairs(pairs)?;
        dist.normalize()?;
        Ok(Self { joint: dist })
    }

    /// Create a joint distribution that is the product of two independent distributions.
    pub fn independent(
        dx: &Distribution<X>,
        dy: &Distribution<Y>,
    ) -> QuantifyResult<Self> {
        let mut pairs = Vec::new();
        for (x, px) in dx.iter() {
            for (y, py) in dy.iter() {
                pairs.push(((x.clone(), y.clone()), px * py));
            }
        }
        Self::from_pairs(pairs)
    }

    /// Marginalize out Y to get P(X).
    pub fn marginalize_y(&self) -> QuantifyResult<Distribution<X>> {
        let mut masses: BTreeMap<X, f64> = BTreeMap::new();
        for ((x, _y), p) in self.joint.iter() {
            *masses.entry(x.clone()).or_insert(0.0) += p;
        }
        Distribution::from_pairs(masses.into_iter())
    }

    /// Marginalize out X to get P(Y).
    pub fn marginalize_x(&self) -> QuantifyResult<Distribution<Y>> {
        let mut masses: BTreeMap<Y, f64> = BTreeMap::new();
        for ((_x, y), p) in self.joint.iter() {
            *masses.entry(y.clone()).or_insert(0.0) += p;
        }
        Distribution::from_pairs(masses.into_iter())
    }

    /// Compute the conditional distribution P(Y|X).
    pub fn conditional_y_given_x(&self) -> QuantifyResult<ConditionalDistribution<X, Y>> {
        let marginal_x = self.marginalize_y()?;
        let mut conditionals = BTreeMap::new();

        for (x, px) in marginal_x.iter() {
            if px <= 0.0 {
                continue;
            }
            let pairs: Vec<(Y, f64)> = self
                .joint
                .iter()
                .filter(|((xi, _), _)| xi == x)
                .map(|((_xi, y), p)| (y.clone(), p / px))
                .collect();
            if !pairs.is_empty() {
                conditionals.insert(x.clone(), Distribution::from_weights(pairs)?);
            }
        }

        ConditionalDistribution::new(conditionals)
    }

    /// Compute the conditional distribution P(X|Y).
    pub fn conditional_x_given_y(&self) -> QuantifyResult<ConditionalDistribution<Y, X>> {
        let marginal_y = self.marginalize_x()?;
        let mut conditionals = BTreeMap::new();

        for (y, py) in marginal_y.iter() {
            if py <= 0.0 {
                continue;
            }
            let pairs: Vec<(X, f64)> = self
                .joint
                .iter()
                .filter(|((_x, yi), _)| yi == y)
                .map(|((x, _yi), p)| (x.clone(), p / py))
                .collect();
            if !pairs.is_empty() {
                conditionals.insert(y.clone(), Distribution::from_weights(pairs)?);
            }
        }

        ConditionalDistribution::new(conditionals)
    }

    /// Probability of a specific pair.
    pub fn probability(&self, x: &X, y: &Y) -> f64 {
        // We need to look up (x, y) in the joint
        // Since BTreeMap keys are (X, Y) tuples, we construct the key
        self.joint.probability(&(x.clone(), y.clone()))
    }

    /// The underlying joint distribution.
    pub fn inner(&self) -> &Distribution<(X, Y)> {
        &self.joint
    }

    /// Iterate over ((x, y), probability) triples.
    pub fn iter(&self) -> impl Iterator<Item = (&(X, Y), f64)> {
        self.joint.iter()
    }

    /// Number of joint outcomes with positive probability.
    pub fn support_size(&self) -> usize {
        self.joint.support_size()
    }

    /// Check if X and Y are independent (within tolerance).
    pub fn is_independent(&self, tolerance: f64) -> QuantifyResult<bool> {
        let mx = self.marginalize_y()?;
        let my = self.marginalize_x()?;

        for ((x, y), pxy) in self.joint.iter() {
            let px = mx.probability(x);
            let py = my.probability(y);
            if (pxy - px * py).abs() > tolerance {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Helper to create a distribution from a vector of probabilities
/// with outcomes 0, 1, 2, ...
pub fn distribution_from_vec(probs: &[f64]) -> QuantifyResult<Distribution<usize>> {
    Distribution::from_pairs(probs.iter().copied().enumerate())
}

/// Create a binary distribution with P(0) = p, P(1) = 1-p.
pub fn binary_distribution(p: f64) -> QuantifyResult<Distribution<usize>> {
    if !(0.0..=1.0).contains(&p) {
        return Err(QuantifyError::InvalidDistribution(format!(
            "binary probability {p} not in [0,1]"
        )));
    }
    let mut pairs = Vec::new();
    if p > 0.0 {
        pairs.push((0, p));
    }
    if 1.0 - p > 0.0 {
        pairs.push((1, 1.0 - p));
    }
    Distribution::from_pairs(pairs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let u = UniformDistribution::new(8).unwrap();
        assert_eq!(u.support_size(), 8);
        assert!((u.probability() - 0.125).abs() < 1e-10);
        assert!((u.entropy() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_to_distribution() {
        let u = UniformDistribution::new(4).unwrap();
        let d = u.to_distribution().unwrap();
        assert!(d.is_valid());
        assert_eq!(d.support_size(), 4);
        for i in 0..4 {
            assert!((d.probability(&i) - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_point_distribution() {
        let p = PointDistribution::new(42);
        assert!((p.probability(&42) - 1.0).abs() < 1e-10);
        assert!((p.probability(&0) - 0.0).abs() < 1e-10);
        assert!((p.entropy() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_to_distribution() {
        let p = PointDistribution::new(7);
        let d = p.to_distribution().unwrap();
        assert!(d.is_valid());
        assert_eq!(d.support_size(), 1);
        assert!((d.probability(&7) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_from_pairs() {
        let d = Distribution::from_pairs(vec![(0, 0.5), (1, 0.3), (2, 0.2)]).unwrap();
        assert!(d.is_valid());
        assert!((d.probability(&0) - 0.5).abs() < 1e-10);
        assert!((d.probability(&1) - 0.3).abs() < 1e-10);
        assert!((d.probability(&2) - 0.2).abs() < 1e-10);
        assert!((d.probability(&3) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_from_weights() {
        let d = Distribution::from_weights(vec![(0, 2.0), (1, 3.0), (2, 5.0)]).unwrap();
        assert!(d.is_valid());
        assert!((d.probability(&0) - 0.2).abs() < 1e-10);
        assert!((d.probability(&1) - 0.3).abs() < 1e-10);
        assert!((d.probability(&2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_condition() {
        let d = Distribution::from_pairs(vec![(0, 0.3), (1, 0.3), (2, 0.4)]).unwrap();
        let cond = d.condition(|&x| x >= 1).unwrap();
        assert!(cond.is_valid());
        assert!((cond.probability(&0) - 0.0).abs() < 1e-10);
        // P(1 | X>=1) = 0.3/0.7
        assert!((cond.probability(&1) - 3.0 / 7.0).abs() < 1e-8);
        assert!((cond.probability(&2) - 4.0 / 7.0).abs() < 1e-8);
    }

    #[test]
    fn test_distribution_map_outcomes() {
        let d = Distribution::from_pairs(vec![(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)])
            .unwrap();
        // Map to even/odd: 0,2 -> 0; 1,3 -> 1
        let mapped = d.map_outcomes(|&x| x % 2).unwrap();
        assert!((mapped.probability(&0) - 0.5).abs() < 1e-10);
        assert!((mapped.probability(&1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_same() {
        let d = Distribution::from_pairs(vec![(0, 0.5), (1, 0.5)]).unwrap();
        assert!((d.kl_divergence(&d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_variation() {
        let d1 = Distribution::from_pairs(vec![(0, 1.0)]).unwrap();
        let d2 = Distribution::from_pairs(vec![(1, 1.0)]).unwrap();
        assert!((d1.total_variation_distance(&d2) - 1.0).abs() < 1e-10);

        let d3 = Distribution::from_pairs(vec![(0, 0.5), (1, 0.5)]).unwrap();
        assert!((d1.total_variation_distance(&d3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_binary_distribution() {
        let d = binary_distribution(0.5).unwrap();
        assert!(d.is_valid());
        assert!((d.probability(&0) - 0.5).abs() < 1e-10);
        assert!((d.probability(&1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_binary_edge_cases() {
        let d0 = binary_distribution(1.0).unwrap();
        assert_eq!(d0.support_size(), 1);
        assert!((d0.probability(&0) - 1.0).abs() < 1e-10);

        let d1 = binary_distribution(0.0).unwrap();
        assert_eq!(d1.support_size(), 1);
        assert!((d1.probability(&1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_joint_distribution_marginals() {
        let joint = JointDistribution::from_pairs(vec![
            ((0, 0), 0.3),
            ((0, 1), 0.1),
            ((1, 0), 0.2),
            ((1, 1), 0.4),
        ])
        .unwrap();

        let mx = joint.marginalize_y().unwrap();
        assert!((mx.probability(&0) - 0.4).abs() < 1e-8);
        assert!((mx.probability(&1) - 0.6).abs() < 1e-8);

        let my = joint.marginalize_x().unwrap();
        assert!((my.probability(&0) - 0.5).abs() < 1e-8);
        assert!((my.probability(&1) - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_joint_independent() {
        let dx = Distribution::from_pairs(vec![(0, 0.5), (1, 0.5)]).unwrap();
        let dy = Distribution::from_pairs(vec![(0, 0.3), (1, 0.7)]).unwrap();
        let joint = JointDistribution::independent(&dx, &dy).unwrap();

        assert!((joint.probability(&0, &0) - 0.15).abs() < 1e-8);
        assert!((joint.probability(&0, &1) - 0.35).abs() < 1e-8);
        assert!((joint.probability(&1, &0) - 0.15).abs() < 1e-8);
        assert!((joint.probability(&1, &1) - 0.35).abs() < 1e-8);

        assert!(joint.is_independent(1e-6).unwrap());
    }

    #[test]
    fn test_joint_conditional() {
        let joint = JointDistribution::from_pairs(vec![
            ((0, 0), 0.4),
            ((0, 1), 0.1),
            ((1, 0), 0.1),
            ((1, 1), 0.4),
        ])
        .unwrap();

        let cond = joint.conditional_y_given_x().unwrap();
        let py_given_0 = cond.given(&0).unwrap();
        assert!((py_given_0.probability(&0) - 0.8).abs() < 1e-8);
        assert!((py_given_0.probability(&1) - 0.2).abs() < 1e-8);

        let py_given_1 = cond.given(&1).unwrap();
        assert!((py_given_1.probability(&0) - 0.2).abs() < 1e-8);
        assert!((py_given_1.probability(&1) - 0.8).abs() < 1e-8);
    }

    #[test]
    fn test_distribution_from_vec() {
        let d = distribution_from_vec(&[0.25, 0.25, 0.25, 0.25]).unwrap();
        assert!(d.is_valid());
        assert_eq!(d.support_size(), 4);
    }

    #[test]
    fn test_max_min_probability() {
        let d = Distribution::from_pairs(vec![(0, 0.1), (1, 0.6), (2, 0.3)]).unwrap();
        assert!((d.max_probability() - 0.6).abs() < 1e-10);
        assert!((d.min_probability() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let d = Distribution::from_pairs(vec![(0, 0.5), (1, 0.5)]).unwrap();
        // <d, d> = 0.25 + 0.25 = 0.5
        assert!((d.inner_product(&d) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_conditional_distribution_joint() {
        let mut conds = BTreeMap::new();
        conds.insert(0, Distribution::from_pairs(vec![(0, 0.8), (1, 0.2)]).unwrap());
        conds.insert(1, Distribution::from_pairs(vec![(0, 0.3), (1, 0.7)]).unwrap());
        let cd = ConditionalDistribution::new(conds).unwrap();

        let prior = Distribution::from_pairs(vec![(0, 0.5), (1, 0.5)]).unwrap();
        let joint = cd.joint(&prior).unwrap();

        // P(0,0) = 0.5 * 0.8 = 0.4
        assert!((joint.probability(&0, &0) - 0.4).abs() < 1e-8);
        // P(1,1) = 0.5 * 0.7 = 0.35
        assert!((joint.probability(&1, &1) - 0.35).abs() < 1e-8);
    }

    #[test]
    fn test_empty_support_error() {
        let result = Distribution::<usize>::from_pairs(Vec::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_probability_error() {
        let result = Distribution::from_pairs(vec![(0, -0.5), (1, 1.5)]);
        assert!(result.is_err());
    }
}
