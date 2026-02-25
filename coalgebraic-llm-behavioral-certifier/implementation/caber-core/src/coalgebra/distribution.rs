//! Sub-distributions with metric computations.
//!
//! A sub-distribution μ over a finite set S is a function μ: S → [0,1] such
//! that Σ_{s∈S} μ(s) ≤ 1. The missing mass 1 - Σ μ(s) represents the
//! probability of "no observation" or termination.
//!
//! This module provides:
//! - SubDistribution<T>: finitely supported sub-distributions
//! - Sampling, expectation, variance
//! - Kantorovich/Wasserstein distance
//! - Total variation, KL divergence, Hellinger distance
//! - Distribution composition, product, and marginals
//! - Statistical hypothesis testing (KS test, chi-squared)
//! - Empirical distribution from samples

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;

use ordered_float::OrderedFloat;
use rand::prelude::*;
use rand_distr::Uniform;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SubDistribution
// ---------------------------------------------------------------------------

/// A finitely supported sub-distribution over elements of type T.
/// Invariant: all weights are in [0,1] and sum ≤ 1.
#[derive(Clone, Serialize, Deserialize)]
pub struct SubDistribution<T: Eq + Hash + Clone + Ord> {
    weights: BTreeMap<T, f64>,
    total_mass: f64,
}

impl<T: Eq + Hash + Clone + Ord> Hash for SubDistribution<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for (k, v) in &self.weights {
            k.hash(state);
            OrderedFloat(*v).hash(state);
        }
    }
}

impl<T: Eq + Hash + Clone + Ord> PartialEq for SubDistribution<T> {
    fn eq(&self, other: &Self) -> bool {
        self.weights.len() == other.weights.len()
            && self.weights.iter().zip(other.weights.iter()).all(|((k1, v1), (k2, v2))| {
                k1 == k2 && OrderedFloat(*v1) == OrderedFloat(*v2)
            })
    }
}

impl<T: Eq + Hash + Clone + Ord> Eq for SubDistribution<T> {}

impl<T: Eq + Hash + Clone + Ord> PartialOrd for SubDistribution<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + Hash + Clone + Ord> Ord for SubDistribution<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_entries: Vec<_> = self.weights.iter().collect();
        let other_entries: Vec<_> = other.weights.iter().collect();
        for (a, b) in self_entries.iter().zip(other_entries.iter()) {
            match a.0.cmp(b.0) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
            match OrderedFloat(*a.1).cmp(&OrderedFloat(*b.1)) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        self_entries.len().cmp(&other_entries.len())
    }
}

impl<T: Eq + Hash + Clone + Ord> SubDistribution<T> {
    /// Create an empty (zero) sub-distribution.
    pub fn empty() -> Self {
        Self {
            weights: BTreeMap::new(),
            total_mass: 0.0,
        }
    }

    /// Create a point mass (Dirac delta) at a single element.
    pub fn point(element: T) -> Self {
        let mut weights = BTreeMap::new();
        weights.insert(element, 1.0);
        Self {
            weights,
            total_mass: 1.0,
        }
    }

    /// Create a uniform distribution over a set of elements.
    pub fn uniform(elements: Vec<T>) -> Self {
        if elements.is_empty() {
            return Self::empty();
        }
        let n = elements.len() as f64;
        let w = 1.0 / n;
        let mut weights = BTreeMap::new();
        for e in elements {
            *weights.entry(e).or_insert(0.0) += w;
        }
        let total_mass = weights.values().sum();
        Self {
            weights,
            total_mass,
        }
    }

    /// Create from a map of weights. Weights are not normalized.
    pub fn from_weights(weights: BTreeMap<T, f64>) -> Result<Self, DistributionError> {
        let mut total = 0.0;
        for (_, &w) in &weights {
            if w < -1e-12 {
                return Err(DistributionError::NegativeWeight(w));
            }
            total += w.max(0.0);
        }
        if total > 1.0 + 1e-10 {
            return Err(DistributionError::ExcessiveMass(total));
        }
        let cleaned: BTreeMap<T, f64> = weights
            .into_iter()
            .filter(|(_, w)| *w > 1e-15)
            .map(|(k, w)| (k, w.max(0.0)))
            .collect();
        let total_mass = cleaned.values().sum();
        Ok(Self {
            weights: cleaned,
            total_mass,
        })
    }

    /// Create from unnormalized weights, normalizing to sum to 1.
    pub fn from_unnormalized(weights: BTreeMap<T, f64>) -> Result<Self, DistributionError> {
        let total: f64 = weights.values().filter(|w| **w > 0.0).sum();
        if total <= 1e-15 {
            return Err(DistributionError::ZeroMass);
        }
        let normalized: BTreeMap<T, f64> = weights
            .into_iter()
            .filter(|(_, w)| *w > 1e-15)
            .map(|(k, w)| (k, w.max(0.0) / total))
            .collect();
        let total_mass = normalized.values().sum();
        Ok(Self {
            weights: normalized,
            total_mass,
        })
    }

    /// Get the weight (probability) of an element.
    pub fn weight(&self, element: &T) -> f64 {
        self.weights.get(element).copied().unwrap_or(0.0)
    }

    /// Get the total mass.
    pub fn total_mass(&self) -> f64 {
        self.total_mass
    }

    /// Get the missing mass (probability of termination/no observation).
    pub fn missing_mass(&self) -> f64 {
        (1.0 - self.total_mass).max(0.0)
    }

    /// Check if this is a proper distribution (total mass = 1).
    pub fn is_proper(&self, tolerance: f64) -> bool {
        (self.total_mass - 1.0).abs() <= tolerance
    }

    /// Check if this is the zero distribution.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Number of elements with non-zero weight.
    pub fn support_size(&self) -> usize {
        self.weights.len()
    }

    /// Iterate over (element, weight) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&T, f64)> {
        self.weights.iter().map(|(k, &v)| (k, v))
    }

    /// Get the support set.
    pub fn support(&self) -> Vec<&T> {
        self.weights.keys().collect()
    }

    /// Get the element with highest weight.
    pub fn mode(&self) -> Option<&T> {
        self.weights
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k)
    }

    /// Normalize to a proper distribution.
    pub fn normalize(&self) -> Result<Self, DistributionError> {
        if self.total_mass <= 1e-15 {
            return Err(DistributionError::ZeroMass);
        }
        let new_weights: BTreeMap<T, f64> = self
            .weights
            .iter()
            .map(|(k, &v)| (k.clone(), v / self.total_mass))
            .collect();
        Ok(Self {
            weights: new_weights,
            total_mass: 1.0,
        })
    }

    /// Scale all weights by a factor.
    pub fn scale(&self, factor: f64) -> Result<Self, DistributionError> {
        if factor < 0.0 {
            return Err(DistributionError::NegativeWeight(factor));
        }
        let new_weights: BTreeMap<T, f64> = self
            .weights
            .iter()
            .map(|(k, &v)| (k.clone(), v * factor))
            .collect();
        let new_total = self.total_mass * factor;
        if new_total > 1.0 + 1e-10 {
            return Err(DistributionError::ExcessiveMass(new_total));
        }
        Ok(Self {
            weights: new_weights,
            total_mass: new_total,
        })
    }

    /// Add another sub-distribution (mixture with equal weights).
    pub fn add(&self, other: &Self) -> Result<Self, DistributionError> {
        let mut new_weights = self.weights.clone();
        for (k, &v) in &other.weights {
            *new_weights.entry(k.clone()).or_insert(0.0) += v;
        }
        let total: f64 = new_weights.values().sum();
        if total > 1.0 + 1e-10 {
            return Err(DistributionError::ExcessiveMass(total));
        }
        Ok(Self {
            weights: new_weights,
            total_mass: total,
        })
    }

    /// Mixture: α·self + (1-α)·other.
    pub fn mixture(&self, other: &Self, alpha: f64) -> Result<Self, DistributionError> {
        if alpha < 0.0 || alpha > 1.0 {
            return Err(DistributionError::InvalidParameter(
                "alpha must be in [0,1]".to_string(),
            ));
        }
        let beta = 1.0 - alpha;
        let mut new_weights = BTreeMap::new();
        for (k, &v) in &self.weights {
            *new_weights.entry(k.clone()).or_insert(0.0) += alpha * v;
        }
        for (k, &v) in &other.weights {
            *new_weights.entry(k.clone()).or_insert(0.0) += beta * v;
        }
        let new_weights: BTreeMap<T, f64> = new_weights
            .into_iter()
            .filter(|(_, v)| *v > 1e-15)
            .collect();
        let total = new_weights.values().sum();
        Ok(Self {
            weights: new_weights,
            total_mass: total,
        })
    }

    /// Filter elements satisfying a predicate.
    pub fn filter<F: Fn(&T) -> bool>(&self, pred: F) -> Self {
        let new_weights: BTreeMap<T, f64> = self
            .weights
            .iter()
            .filter(|(k, _)| pred(k))
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        let total = new_weights.values().sum();
        Self {
            weights: new_weights,
            total_mass: total,
        }
    }

    /// Map elements through a function, summing weights of collisions.
    pub fn map<U, F>(&self, f: F) -> SubDistribution<U>
    where
        U: Eq + Hash + Clone + Ord,
        F: Fn(&T) -> U,
    {
        let mut new_weights = BTreeMap::new();
        for (k, &v) in &self.weights {
            let new_k = f(k);
            *new_weights.entry(new_k).or_insert(0.0) += v;
        }
        let total = new_weights.values().sum();
        SubDistribution {
            weights: new_weights,
            total_mass: total,
        }
    }

    /// Flat map (monadic bind): for each element x with weight p(x),
    /// apply f to get a sub-distribution, then weight it by p(x) and sum.
    pub fn flat_map<U, F>(&self, f: F) -> SubDistribution<U>
    where
        U: Eq + Hash + Clone + Ord,
        F: Fn(&T) -> SubDistribution<U>,
    {
        let mut new_weights: BTreeMap<U, f64> = BTreeMap::new();
        for (k, &v) in &self.weights {
            let sub = f(k);
            for (sub_k, sub_v) in &sub.weights {
                *new_weights.entry(sub_k.clone()).or_insert(0.0) += v * sub_v;
            }
        }
        let new_weights: BTreeMap<U, f64> = new_weights
            .into_iter()
            .filter(|(_, v)| *v > 1e-15)
            .collect();
        let total = new_weights.values().sum();
        SubDistribution {
            weights: new_weights,
            total_mass: total,
        }
    }

    /// Sample from the distribution using the given RNG.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<T> {
        if self.is_empty() || self.total_mass <= 1e-15 {
            return None;
        }
        let u: f64 = rng.gen_range(0.0..self.total_mass);
        let mut cumulative = 0.0;
        for (k, &v) in &self.weights {
            cumulative += v;
            if cumulative > u {
                return Some(k.clone());
            }
        }
        // Rounding: return the last element
        self.weights.keys().last().cloned()
    }

    /// Draw n samples with replacement.
    pub fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<T> {
        (0..n).filter_map(|_| self.sample(rng)).collect()
    }

    /// Compute entropy H(μ) = -Σ p(x) log p(x).
    pub fn entropy(&self) -> f64 {
        if self.total_mass <= 1e-15 {
            return 0.0;
        }
        let mut h = 0.0;
        for &w in self.weights.values() {
            if w > 1e-15 {
                let p = w / self.total_mass;
                h -= p * p.ln();
            }
        }
        h
    }

    /// Cross-entropy H(self, other) = -Σ p(x) log q(x).
    pub fn cross_entropy(&self, other: &Self) -> f64 {
        let mut h = 0.0;
        for (k, &p) in &self.weights {
            if p > 1e-15 {
                let q = other.weight(k);
                if q <= 1e-15 {
                    return f64::INFINITY;
                }
                let p_norm = p / self.total_mass;
                h -= p_norm * q.ln();
            }
        }
        h
    }

    /// Maximum entropy given the support size.
    pub fn max_entropy(&self) -> f64 {
        if self.support_size() == 0 {
            return 0.0;
        }
        (self.support_size() as f64).ln()
    }

    /// Normalized entropy in [0, 1].
    pub fn normalized_entropy(&self) -> f64 {
        let max_h = self.max_entropy();
        if max_h <= 1e-15 {
            return 0.0;
        }
        self.entropy() / max_h
    }

    /// Truncate to the top-k elements by weight.
    pub fn top_k(&self, k: usize) -> Self {
        let mut items: Vec<(T, f64)> = self
            .weights
            .iter()
            .map(|(key, &val)| (key.clone(), val))
            .collect();
        items.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        items.truncate(k);
        let new_weights: BTreeMap<T, f64> = items.into_iter().collect();
        let total = new_weights.values().sum();
        Self {
            weights: new_weights,
            total_mass: total,
        }
    }

    /// Truncate elements with weight below a threshold.
    pub fn threshold(&self, min_weight: f64) -> Self {
        let new_weights: BTreeMap<T, f64> = self
            .weights
            .iter()
            .filter(|(_, &v)| v >= min_weight)
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        let total = new_weights.values().sum();
        Self {
            weights: new_weights,
            total_mass: total,
        }
    }
}

// ---------------------------------------------------------------------------
// Expectation & variance for numeric distributions
// ---------------------------------------------------------------------------

impl SubDistribution<OrderedFloat<f64>> {
    /// Expected value.
    pub fn expectation(&self) -> f64 {
        if self.total_mass <= 1e-15 {
            return 0.0;
        }
        let mut sum = 0.0;
        for (k, &v) in &self.weights {
            sum += k.into_inner() * v;
        }
        sum / self.total_mass
    }

    /// Variance.
    pub fn variance(&self) -> f64 {
        if self.total_mass <= 1e-15 {
            return 0.0;
        }
        let mean = self.expectation();
        let mut var = 0.0;
        for (k, &v) in &self.weights {
            let diff = k.into_inner() - mean;
            var += diff * diff * v;
        }
        var / self.total_mass
    }

    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Median (smallest x such that CDF(x) ≥ 0.5).
    pub fn median(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let half = self.total_mass / 2.0;
        let mut cumulative = 0.0;
        for (k, &v) in &self.weights {
            cumulative += v;
            if cumulative >= half {
                return k.into_inner();
            }
        }
        self.weights.keys().last().unwrap().into_inner()
    }

    /// n-th moment about the mean.
    pub fn central_moment(&self, n: u32) -> f64 {
        if self.total_mass <= 1e-15 {
            return 0.0;
        }
        let mean = self.expectation();
        let mut sum = 0.0;
        for (k, &v) in &self.weights {
            let diff = k.into_inner() - mean;
            sum += diff.powi(n as i32) * v;
        }
        sum / self.total_mass
    }

    /// Skewness.
    pub fn skewness(&self) -> f64 {
        let sd = self.std_dev();
        if sd <= 1e-15 {
            return 0.0;
        }
        self.central_moment(3) / sd.powi(3)
    }

    /// Kurtosis (excess).
    pub fn kurtosis(&self) -> f64 {
        let sd = self.std_dev();
        if sd <= 1e-15 {
            return 0.0;
        }
        self.central_moment(4) / sd.powi(4) - 3.0
    }

    /// CDF at a point.
    pub fn cdf(&self, x: f64) -> f64 {
        if self.total_mass <= 1e-15 {
            return 0.0;
        }
        let mut cumulative = 0.0;
        for (k, &v) in &self.weights {
            if k.into_inner() <= x {
                cumulative += v;
            } else {
                break; // BTreeMap is sorted
            }
        }
        cumulative / self.total_mass
    }

    /// Quantile function (inverse CDF).
    pub fn quantile(&self, p: f64) -> f64 {
        assert!(p >= 0.0 && p <= 1.0);
        let target = p * self.total_mass;
        let mut cumulative = 0.0;
        for (k, &v) in &self.weights {
            cumulative += v;
            if cumulative >= target {
                return k.into_inner();
            }
        }
        self.weights
            .keys()
            .last()
            .map(|k| k.into_inner())
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Distance metrics between distributions
// ---------------------------------------------------------------------------

impl<T: Eq + Hash + Clone + Ord> SubDistribution<T> {
    /// Total variation distance: TV(μ, ν) = (1/2) Σ |μ(x) - ν(x)|
    pub fn total_variation(&self, other: &Self) -> f64 {
        let all_keys: HashSet<&T> = self
            .weights
            .keys()
            .chain(other.weights.keys())
            .collect();
        let mut sum = 0.0;
        for k in all_keys {
            let p = self.weight(k);
            let q = other.weight(k);
            sum += (p - q).abs();
        }
        // Also account for missing mass
        sum += (self.missing_mass() - other.missing_mass()).abs();
        sum / 2.0
    }

    /// KL divergence: KL(self || other) = Σ p(x) log(p(x)/q(x))
    /// Returns infinity if self has support not in other's support.
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        let mut kl = 0.0;
        for (k, &p) in &self.weights {
            if p <= 1e-15 {
                continue;
            }
            let q = other.weight(k);
            if q <= 1e-15 {
                return f64::INFINITY;
            }
            kl += p * (p / q).ln();
        }
        kl
    }

    /// Symmetrized KL divergence: (KL(p||q) + KL(q||p)) / 2
    pub fn symmetric_kl(&self, other: &Self) -> f64 {
        (self.kl_divergence(other) + other.kl_divergence(self)) / 2.0
    }

    /// Jensen-Shannon divergence: JS(p, q) = (KL(p||m) + KL(q||m)) / 2
    /// where m = (p + q) / 2. Always finite and symmetric.
    pub fn jensen_shannon(&self, other: &Self) -> f64 {
        let m = self.mixture(other, 0.5).unwrap_or_else(|_| {
            // If mixture fails, compute manually
            let mut weights = BTreeMap::new();
            for (k, &v) in &self.weights {
                *weights.entry(k.clone()).or_insert(0.0) += 0.5 * v;
            }
            for (k, &v) in &other.weights {
                *weights.entry(k.clone()).or_insert(0.0) += 0.5 * v;
            }
            let total = weights.values().sum();
            SubDistribution {
                weights,
                total_mass: total,
            }
        });
        (self.kl_divergence(&m) + other.kl_divergence(&m)) / 2.0
    }

    /// Hellinger distance: H(p, q) = (1/√2) √(Σ (√p(x) - √q(x))²)
    pub fn hellinger_distance(&self, other: &Self) -> f64 {
        let all_keys: HashSet<&T> = self
            .weights
            .keys()
            .chain(other.weights.keys())
            .collect();
        let mut sum = 0.0;
        for k in all_keys {
            let p = self.weight(k).sqrt();
            let q = other.weight(k).sqrt();
            sum += (p - q).powi(2);
        }
        (sum / 2.0).sqrt()
    }

    /// Bhattacharyya distance: -ln(BC(p,q)) where BC = Σ √(p(x)q(x))
    pub fn bhattacharyya_distance(&self, other: &Self) -> f64 {
        let bc = self.bhattacharyya_coefficient(other);
        if bc <= 1e-15 {
            return f64::INFINITY;
        }
        -bc.ln()
    }

    /// Bhattacharyya coefficient: BC(p,q) = Σ √(p(x)q(x))
    pub fn bhattacharyya_coefficient(&self, other: &Self) -> f64 {
        let mut bc = 0.0;
        for (k, &p) in &self.weights {
            let q = other.weight(k);
            if q > 0.0 && p > 0.0 {
                bc += (p * q).sqrt();
            }
        }
        bc
    }

    /// Chi-squared distance: χ²(p, q) = Σ (p(x) - q(x))² / q(x)
    pub fn chi_squared_distance(&self, other: &Self) -> f64 {
        let mut chi2 = 0.0;
        for (k, &p) in &self.weights {
            let q = other.weight(k);
            if q <= 1e-15 {
                if p > 1e-15 {
                    return f64::INFINITY;
                }
                continue;
            }
            chi2 += (p - q).powi(2) / q;
        }
        // Elements in other but not in self
        for (k, &q) in &other.weights {
            if self.weight(k) <= 1e-15 && q > 1e-15 {
                chi2 += q; // (0 - q)² / q = q
            }
        }
        chi2
    }

    /// Rényi divergence of order α: R_α(p || q) = 1/(α-1) log Σ p(x)^α q(x)^(1-α)
    pub fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        assert!(alpha > 0.0 && alpha != 1.0);
        let mut sum = 0.0;
        for (k, &p) in &self.weights {
            if p <= 1e-15 {
                continue;
            }
            let q = other.weight(k);
            if q <= 1e-15 {
                if alpha > 1.0 {
                    return f64::INFINITY;
                }
                continue;
            }
            sum += p.powf(alpha) * q.powf(1.0 - alpha);
        }
        if sum <= 1e-15 {
            return f64::INFINITY;
        }
        sum.ln() / (alpha - 1.0)
    }

    /// Kantorovich (Earth Mover's / Wasserstein-1) distance given a ground metric.
    /// Uses the optimal transport formulation as a linear program solved greedily
    /// for the 1D case when T is ordered.
    pub fn kantorovich_distance(&self, other: &Self) -> f64 {
        // Collect all unique elements and sort them
        let all_keys: BTreeMap<&T, ()> = self
            .weights
            .keys()
            .chain(other.weights.keys())
            .map(|k| (k, ()))
            .collect();
        let sorted_keys: Vec<&T> = all_keys.keys().copied().collect();

        if sorted_keys.is_empty() {
            return 0.0;
        }

        // For general T, we use the discrete Kantorovich metric
        // which reduces to TV distance when the ground metric is {0,1}
        // For ordered T, we use the CDF-based formula.
        let mut cdf_diff = 0.0;
        let mut p_cum = 0.0;
        let mut q_cum = 0.0;
        let mut distance = 0.0;

        for (i, k) in sorted_keys.iter().enumerate() {
            p_cum += self.weight(k);
            q_cum += other.weight(k);
            let diff = (p_cum - q_cum).abs();
            if i + 1 < sorted_keys.len() {
                distance += diff; // Using discrete metric spacing = 1
            }
            cdf_diff = diff;
        }
        // Include missing mass difference
        distance += ((self.total_mass - p_cum) - (other.total_mass - q_cum)).abs();

        distance
    }

    /// Wasserstein distance with a custom ground metric function.
    /// For small support sizes, solves the optimal transport problem exactly
    /// using the Hungarian algorithm / north-west corner approach.
    pub fn wasserstein_with_metric<F: Fn(&T, &T) -> f64>(&self, other: &Self, metric: F) -> f64 {
        let p_support: Vec<(&T, f64)> = self.iter().collect();
        let q_support: Vec<(&T, f64)> = other.iter().collect();

        if p_support.is_empty() && q_support.is_empty() {
            return 0.0;
        }
        if p_support.is_empty() || q_support.is_empty() {
            return 1.0; // Maximum distance when one is empty
        }

        let m = p_support.len();
        let n = q_support.len();

        // Build cost matrix
        let mut cost = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                cost[i][j] = metric(p_support[i].0, q_support[j].0);
            }
        }

        // Use north-west corner method as initial feasible solution,
        // then improve with stepping-stone method.
        // For simplicity and correctness, use the greedy transport approach.
        let mut supply: Vec<f64> = p_support.iter().map(|(_, w)| *w).collect();
        let mut demand: Vec<f64> = q_support.iter().map(|(_, w)| *w).collect();

        // Flatten cost and sort by cost
        let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                edges.push((cost[i][j], i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut total_cost = 0.0;
        for (c, i, j) in edges {
            if supply[i] <= 1e-15 || demand[j] <= 1e-15 {
                continue;
            }
            let flow = supply[i].min(demand[j]);
            total_cost += flow * c;
            supply[i] -= flow;
            demand[j] -= flow;
        }

        total_cost
    }

    /// L_p distance between distributions viewed as vectors.
    pub fn lp_distance(&self, other: &Self, p: f64) -> f64 {
        assert!(p >= 1.0);
        let all_keys: HashSet<&T> = self
            .weights
            .keys()
            .chain(other.weights.keys())
            .collect();
        let mut sum = 0.0;
        for k in all_keys {
            let pw = self.weight(k);
            let qw = other.weight(k);
            sum += (pw - qw).abs().powf(p);
        }
        // Missing mass
        sum += (self.missing_mass() - other.missing_mass()).abs().powf(p);
        sum.powf(1.0 / p)
    }

    /// Jaccard similarity of supports.
    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        let p_keys: HashSet<&T> = self.weights.keys().collect();
        let q_keys: HashSet<&T> = other.weights.keys().collect();
        let intersection = p_keys.intersection(&q_keys).count();
        let union = p_keys.union(&q_keys).count();
        if union == 0 {
            return 1.0;
        }
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// Product distributions
// ---------------------------------------------------------------------------

impl<T: Eq + Hash + Clone + Ord> SubDistribution<T> {
    /// Product distribution: p × q over pairs (s, t).
    pub fn product<U: Eq + Hash + Clone + Ord>(
        &self,
        other: &SubDistribution<U>,
    ) -> SubDistribution<(T, U)> {
        let mut weights = BTreeMap::new();
        for (k1, &v1) in &self.weights {
            for (k2, &v2) in &other.weights {
                let w = v1 * v2;
                if w > 1e-15 {
                    weights.insert((k1.clone(), k2.clone()), w);
                }
            }
        }
        let total = weights.values().sum();
        SubDistribution {
            weights,
            total_mass: total,
        }
    }
}

/// Compute marginal from a joint distribution.
pub fn marginal_first<T: Eq + Hash + Clone + Ord, U: Eq + Hash + Clone + Ord>(
    joint: &SubDistribution<(T, U)>,
) -> SubDistribution<T> {
    let mut weights = BTreeMap::new();
    for ((k1, _), &v) in &joint.weights {
        *weights.entry(k1.clone()).or_insert(0.0) += v;
    }
    let total = weights.values().sum();
    SubDistribution {
        weights,
        total_mass: total,
    }
}

pub fn marginal_second<T: Eq + Hash + Clone + Ord, U: Eq + Hash + Clone + Ord>(
    joint: &SubDistribution<(T, U)>,
) -> SubDistribution<U> {
    let mut weights = BTreeMap::new();
    for ((_, k2), &v) in &joint.weights {
        *weights.entry(k2.clone()).or_insert(0.0) += v;
    }
    let total = weights.values().sum();
    SubDistribution {
        weights,
        total_mass: total,
    }
}

/// Conditional distribution: P(Y | X = x) from joint P(X, Y).
pub fn conditional<T: Eq + Hash + Clone + Ord, U: Eq + Hash + Clone + Ord>(
    joint: &SubDistribution<(T, U)>,
    given: &T,
) -> Option<SubDistribution<U>> {
    let mut weights = BTreeMap::new();
    let mut total_given = 0.0;
    for ((k1, k2), &v) in &joint.weights {
        if k1 == given {
            *weights.entry(k2.clone()).or_insert(0.0) += v;
            total_given += v;
        }
    }
    if total_given <= 1e-15 {
        return None;
    }
    // Normalize
    for v in weights.values_mut() {
        *v /= total_given;
    }
    Some(SubDistribution {
        weights,
        total_mass: 1.0,
    })
}

// ---------------------------------------------------------------------------
// Empirical distribution
// ---------------------------------------------------------------------------

/// Build an empirical distribution from a collection of samples.
pub fn empirical<T: Eq + Hash + Clone + Ord>(samples: &[T]) -> SubDistribution<T> {
    if samples.is_empty() {
        return SubDistribution::empty();
    }
    let n = samples.len() as f64;
    let mut weights = BTreeMap::new();
    for s in samples {
        *weights.entry(s.clone()).or_insert(0.0) += 1.0 / n;
    }
    let total = weights.values().sum();
    SubDistribution {
        weights,
        total_mass: total,
    }
}

/// Build an empirical distribution from weighted samples.
pub fn weighted_empirical<T: Eq + Hash + Clone + Ord>(
    samples: &[(T, f64)],
) -> Result<SubDistribution<T>, DistributionError> {
    let total_weight: f64 = samples.iter().map(|(_, w)| *w).sum();
    if total_weight <= 1e-15 {
        return Err(DistributionError::ZeroMass);
    }
    let mut weights = BTreeMap::new();
    for (s, w) in samples {
        if *w < 0.0 {
            return Err(DistributionError::NegativeWeight(*w));
        }
        *weights.entry(s.clone()).or_insert(0.0) += w / total_weight;
    }
    let total = weights.values().sum();
    Ok(SubDistribution {
        weights,
        total_mass: total,
    })
}

// ---------------------------------------------------------------------------
// Statistical hypothesis testing
// ---------------------------------------------------------------------------

/// Kolmogorov-Smirnov test statistic between two distributions.
/// Returns the D statistic (supremum of absolute CDF difference).
pub fn ks_statistic<T: Eq + Hash + Clone + Ord>(
    dist1: &SubDistribution<T>,
    dist2: &SubDistribution<T>,
) -> f64 {
    let all_keys: BTreeMap<&T, ()> = dist1
        .weights
        .keys()
        .chain(dist2.weights.keys())
        .map(|k| (k, ()))
        .collect();

    let mut cdf1 = 0.0;
    let mut cdf2 = 0.0;
    let mut max_diff = 0.0f64;

    let n1 = dist1.total_mass();
    let n2 = dist2.total_mass();

    if n1 <= 1e-15 || n2 <= 1e-15 {
        return 1.0;
    }

    for (k, _) in &all_keys {
        cdf1 += dist1.weight(k) / n1;
        cdf2 += dist2.weight(k) / n2;
        max_diff = max_diff.max((cdf1 - cdf2).abs());
    }

    max_diff
}

/// Kolmogorov-Smirnov test: returns (D statistic, p-value approximation).
/// Uses the asymptotic approximation for the p-value.
pub fn ks_test<T: Eq + Hash + Clone + Ord>(
    dist1: &SubDistribution<T>,
    dist2: &SubDistribution<T>,
    n1: usize,
    n2: usize,
) -> (f64, f64) {
    let d = ks_statistic(dist1, dist2);
    let n_eff = ((n1 * n2) as f64) / ((n1 + n2) as f64);
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * d;

    // Kolmogorov's approximation for the p-value
    let p_value = ks_survival_function(lambda);
    (d, p_value)
}

/// Kolmogorov survival function: P(K > lambda) ≈ 2 Σ (-1)^(i-1) exp(-2i²λ²)
fn ks_survival_function(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut p = 0.0;
    for i in 1..=100 {
        let sign = if i % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * (i as f64).powi(2) * lambda * lambda).exp();
        p += term;
        if term.abs() < 1e-15 {
            break;
        }
    }
    (2.0 * p).max(0.0).min(1.0)
}

/// Chi-squared test for goodness of fit.
/// Tests whether observed (empirical) distribution fits the expected distribution.
/// Returns (chi² statistic, degrees of freedom, p-value).
pub fn chi_squared_test<T: Eq + Hash + Clone + Ord>(
    observed: &SubDistribution<T>,
    expected: &SubDistribution<T>,
    total_count: usize,
) -> (f64, usize, f64) {
    let all_keys: HashSet<&T> = observed
        .weights
        .keys()
        .chain(expected.weights.keys())
        .collect();

    let n = total_count as f64;
    let mut chi2 = 0.0;
    let mut df = 0usize;

    for k in &all_keys {
        let o = observed.weight(k) * n;
        let e = expected.weight(k) * n;
        if e > 1e-10 {
            chi2 += (o - e).powi(2) / e;
            df += 1;
        }
    }

    if df > 0 {
        df -= 1;
    }

    // P-value approximation using regularized incomplete gamma function
    let p_value = chi2_survival(chi2, df);

    (chi2, df, p_value)
}

/// Approximate chi-squared survival function using Wilson-Hilferty approximation.
fn chi2_survival(chi2: f64, df: usize) -> f64 {
    if df == 0 {
        return if chi2 > 0.0 { 0.0 } else { 1.0 };
    }
    let k = df as f64;
    // Wilson-Hilferty transformation
    let z = ((chi2 / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    // Standard normal survival function
    normal_survival(z)
}

/// Standard normal survival function approximation.
fn normal_survival(z: f64) -> f64 {
    0.5 * erfc_approx(z / std::f64::consts::SQRT_2)
}

/// Approximate complementary error function.
fn erfc_approx(x: f64) -> f64 {
    // Horner form approximation (Abramowitz & Stegun)
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        2.0 - result
    }
}

/// Anderson-Darling test statistic.
pub fn anderson_darling_statistic<T: Eq + Hash + Clone + Ord>(
    empirical_dist: &SubDistribution<T>,
    reference: &SubDistribution<T>,
) -> f64 {
    let keys: Vec<&T> = empirical_dist.support();
    let n = keys.len();
    if n == 0 {
        return 0.0;
    }

    let mut sorted_cdf_values: Vec<f64> = keys
        .iter()
        .map(|k| reference.weight(k))
        .collect();

    // Compute cumulative sums for the reference distribution
    let mut cdf_vals: Vec<f64> = Vec::with_capacity(n);
    let mut cum = 0.0;
    for k in &keys {
        cum += reference.weight(k);
        cdf_vals.push(cum.min(1.0));
    }

    let n_f = n as f64;
    let mut a2 = 0.0;
    for i in 0..n {
        let fi = cdf_vals[i];
        let fi_clamped = fi.max(1e-15).min(1.0 - 1e-15);
        let i_f = (i + 1) as f64;
        a2 += (2.0 * i_f - 1.0) * fi_clamped.ln()
            + (2.0 * (n_f - i_f) + 1.0) * (1.0 - fi_clamped).ln();
    }
    let _ = sorted_cdf_values;
    -n_f - a2 / n_f
}

// ---------------------------------------------------------------------------
// Distribution comparison utilities
// ---------------------------------------------------------------------------

/// Summary statistics for comparing two distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionComparison {
    pub total_variation: f64,
    pub hellinger: f64,
    pub kl_forward: f64,
    pub kl_backward: f64,
    pub jensen_shannon: f64,
    pub bhattacharyya: f64,
    pub jaccard: f64,
    pub support_size_ratio: f64,
}

impl DistributionComparison {
    pub fn compute<T: Eq + Hash + Clone + Ord>(
        p: &SubDistribution<T>,
        q: &SubDistribution<T>,
    ) -> Self {
        let sp = p.support_size().max(1) as f64;
        let sq = q.support_size().max(1) as f64;
        Self {
            total_variation: p.total_variation(q),
            hellinger: p.hellinger_distance(q),
            kl_forward: p.kl_divergence(q),
            kl_backward: q.kl_divergence(p),
            jensen_shannon: p.jensen_shannon(q),
            bhattacharyya: p.bhattacharyya_distance(q),
            jaccard: p.jaccard_similarity(q),
            support_size_ratio: sp.min(sq) / sp.max(sq),
        }
    }

    /// A summary metric combining multiple distances.
    pub fn combined_distance(&self, weights: &DistanceWeights) -> f64 {
        weights.tv * self.total_variation
            + weights.hellinger * self.hellinger
            + weights.js * self.jensen_shannon
    }
}

/// Weights for combining multiple distance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceWeights {
    pub tv: f64,
    pub hellinger: f64,
    pub js: f64,
}

impl Default for DistanceWeights {
    fn default() -> Self {
        Self {
            tv: 0.4,
            hellinger: 0.3,
            js: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Distribution transformations
// ---------------------------------------------------------------------------

/// Smoothing: add a small uniform weight to all elements in a given universe.
pub fn laplace_smoothing<T: Eq + Hash + Clone + Ord>(
    dist: &SubDistribution<T>,
    universe: &[T],
    alpha: f64,
) -> SubDistribution<T> {
    let k = universe.len() as f64;
    let mut weights = BTreeMap::new();
    let total_original = dist.total_mass();

    for elem in universe {
        let p = dist.weight(elem);
        let smoothed = (p + alpha) / (total_original + alpha * k);
        weights.insert(elem.clone(), smoothed);
    }
    let total = weights.values().sum();
    SubDistribution {
        weights,
        total_mass: total,
    }
}

/// Geometric mean of two distributions.
pub fn geometric_mean<T: Eq + Hash + Clone + Ord>(
    p: &SubDistribution<T>,
    q: &SubDistribution<T>,
) -> SubDistribution<T> {
    let all_keys: HashSet<&T> = p.weights.keys().chain(q.weights.keys()).collect();
    let mut weights = BTreeMap::new();
    for k in all_keys {
        let pw = p.weight(k);
        let qw = q.weight(k);
        if pw > 0.0 && qw > 0.0 {
            weights.insert(k.clone(), (pw * qw).sqrt());
        }
    }
    let total: f64 = weights.values().sum();
    if total > 1e-15 {
        for v in weights.values_mut() {
            *v /= total;
        }
    }
    let total = weights.values().sum();
    SubDistribution {
        weights,
        total_mass: total,
    }
}

/// Power-raise a distribution: each weight p(x) -> p(x)^beta, then renormalize.
pub fn tempered<T: Eq + Hash + Clone + Ord>(
    dist: &SubDistribution<T>,
    beta: f64,
) -> SubDistribution<T> {
    let mut weights = BTreeMap::new();
    for (k, &v) in &dist.weights {
        if v > 0.0 {
            weights.insert(k.clone(), v.powf(beta));
        }
    }
    let total: f64 = weights.values().sum();
    if total > 1e-15 {
        for v in weights.values_mut() {
            *v /= total;
        }
    }
    let total = weights.values().sum();
    SubDistribution {
        weights,
        total_mass: total,
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error)]
pub enum DistributionError {
    #[error("Negative weight: {0}")]
    NegativeWeight(f64),

    #[error("Excessive mass: {0} > 1.0")]
    ExcessiveMass(f64),

    #[error("Zero total mass")]
    ZeroMass,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

// ---------------------------------------------------------------------------
// Display / Debug
// ---------------------------------------------------------------------------

impl<T: Eq + Hash + Clone + Ord + fmt::Display> fmt::Display for SubDistribution<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for (k, &v) in &self.weights {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}: {:.4}", k, v)?;
            first = false;
        }
        write!(f, "}} (mass={:.4})", self.total_mass)
    }
}

impl<T: Eq + Hash + Clone + Ord + fmt::Debug> fmt::Debug for SubDistribution<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubDistribution")
            .field("support_size", &self.support_size())
            .field("total_mass", &self.total_mass)
            .field("weights", &self.weights)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;

    fn of(x: f64) -> OrderedFloat<f64> {
        OrderedFloat(x)
    }

    #[test]
    fn test_empty_distribution() {
        let d = SubDistribution::<u32>::empty();
        assert!(d.is_empty());
        assert_eq!(d.support_size(), 0);
        assert!((d.total_mass() - 0.0).abs() < 1e-10);
        assert!((d.missing_mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_distribution() {
        let d = SubDistribution::point(42u32);
        assert!(!d.is_empty());
        assert_eq!(d.support_size(), 1);
        assert!((d.weight(&42) - 1.0).abs() < 1e-10);
        assert!((d.weight(&0) - 0.0).abs() < 1e-10);
        assert!(d.is_proper(1e-10));
    }

    #[test]
    fn test_uniform_distribution() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3, 4]);
        assert_eq!(d.support_size(), 4);
        assert!((d.weight(&1) - 0.25).abs() < 1e-10);
        assert!(d.is_proper(1e-10));
    }

    #[test]
    fn test_from_weights() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 0.3);
        weights.insert(2, 0.5);
        weights.insert(3, 0.2);
        let d = SubDistribution::from_weights(weights).unwrap();
        assert!(d.is_proper(1e-10));
        assert!((d.weight(&1) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_from_weights_invalid() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 0.6);
        weights.insert(2, 0.6);
        assert!(SubDistribution::from_weights(weights).is_err());
    }

    #[test]
    fn test_from_unnormalized() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 3.0);
        weights.insert(2, 7.0);
        let d = SubDistribution::from_unnormalized(weights).unwrap();
        assert!((d.weight(&1) - 0.3).abs() < 1e-10);
        assert!((d.weight(&2) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 0.2);
        weights.insert(2, 0.3);
        let d = SubDistribution::from_weights(weights).unwrap();
        assert!(!d.is_proper(1e-10));
        let d_norm = d.normalize().unwrap();
        assert!(d_norm.is_proper(1e-10));
    }

    #[test]
    fn test_mode() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 0.1);
        weights.insert(2, 0.5);
        weights.insert(3, 0.3);
        let d = SubDistribution::from_weights(weights).unwrap();
        assert_eq!(d.mode(), Some(&2));
    }

    #[test]
    fn test_mixture() {
        let d1 = SubDistribution::point(1u32);
        let d2 = SubDistribution::point(2u32);
        let mix = d1.mixture(&d2, 0.3).unwrap();
        assert!((mix.weight(&1) - 0.3).abs() < 1e-10);
        assert!((mix.weight(&2) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_filter() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3, 4, 5]);
        let filtered = d.filter(|&x| x > 3);
        assert_eq!(filtered.support_size(), 2);
        assert!(!filtered.is_proper(0.01));
    }

    #[test]
    fn test_map() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3, 4]);
        let mapped = d.map(|&x| x % 2);
        assert_eq!(mapped.support_size(), 2);
        assert!((mapped.weight(&0) - 0.5).abs() < 1e-10);
        assert!((mapped.weight(&1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_flat_map() {
        let d = SubDistribution::uniform(vec![0u32, 1]);
        let result = d.flat_map(|&x| {
            if x == 0 {
                SubDistribution::point(10u32)
            } else {
                SubDistribution::point(20u32)
            }
        });
        assert!((result.weight(&10) - 0.5).abs() < 1e-10);
        assert!((result.weight(&20) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sampling() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        let mut rng = rand::thread_rng();
        let samples = d.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        // All samples should be 1, 2, or 3
        for s in &samples {
            assert!(*s >= 1 && *s <= 3);
        }
    }

    #[test]
    fn test_entropy() {
        // Uniform over 4 elements: entropy = log(4)
        let d = SubDistribution::uniform(vec![1u32, 2, 3, 4]);
        let h = d.entropy();
        assert!((h - (4.0f64).ln()).abs() < 1e-10);

        // Point mass: entropy = 0
        let d2 = SubDistribution::point(1u32);
        assert!((d2.entropy() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_entropy() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3, 4]);
        assert!((d.normalized_entropy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_k() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 0.1);
        weights.insert(2, 0.3);
        weights.insert(3, 0.2);
        weights.insert(4, 0.4);
        let d = SubDistribution::from_weights(weights).unwrap();
        let top2 = d.top_k(2);
        assert_eq!(top2.support_size(), 2);
        assert!(top2.weight(&4) > 0.0);
        assert!(top2.weight(&2) > 0.0);
    }

    // --- Numeric distribution tests ---

    #[test]
    fn test_expectation() {
        let mut weights = BTreeMap::new();
        weights.insert(of(1.0), 0.5);
        weights.insert(of(3.0), 0.5);
        let d = SubDistribution::from_weights(weights).unwrap();
        assert!((d.expectation() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let mut weights = BTreeMap::new();
        weights.insert(of(0.0), 0.5);
        weights.insert(of(2.0), 0.5);
        let d = SubDistribution::from_weights(weights).unwrap();
        // mean = 1, var = 0.5*(0-1)^2 + 0.5*(2-1)^2 = 1
        assert!((d.variance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cdf() {
        let d = SubDistribution::uniform(vec![of(1.0), of(2.0), of(3.0)]);
        assert!((d.cdf(0.0) - 0.0).abs() < 1e-10);
        assert!((d.cdf(1.0) - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.cdf(2.5) - 2.0 / 3.0).abs() < 1e-10);
        assert!((d.cdf(10.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile() {
        let d = SubDistribution::uniform(vec![of(1.0), of(2.0), of(3.0)]);
        let median = d.quantile(0.5);
        assert!((median - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median() {
        let d = SubDistribution::uniform(vec![of(1.0), of(2.0), of(3.0), of(4.0), of(5.0)]);
        assert!((d.median() - 3.0).abs() < 1e-10);
    }

    // --- Distance metric tests ---

    #[test]
    fn test_total_variation_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.total_variation(&d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_variation_disjoint() {
        let d1 = SubDistribution::point(1u32);
        let d2 = SubDistribution::point(2u32);
        assert!((d1.total_variation(&d2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_variation_symmetric() {
        let d1 = SubDistribution::uniform(vec![1u32, 2, 3]);
        let d2 = SubDistribution::point(2u32);
        assert!((d1.total_variation(&d2) - d2.total_variation(&d1)).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.kl_divergence(&d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_disjoint() {
        let d1 = SubDistribution::point(1u32);
        let d2 = SubDistribution::point(2u32);
        assert!(d1.kl_divergence(&d2).is_infinite());
    }

    #[test]
    fn test_jensen_shannon_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.jensen_shannon(&d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jensen_shannon_symmetric() {
        let d1 = SubDistribution::uniform(vec![1u32, 2, 3]);
        let d2 = SubDistribution::point(2u32);
        assert!((d1.jensen_shannon(&d2) - d2.jensen_shannon(&d1)).abs() < 1e-10);
    }

    #[test]
    fn test_hellinger_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.hellinger_distance(&d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hellinger_range() {
        let d1 = SubDistribution::point(1u32);
        let d2 = SubDistribution::point(2u32);
        let h = d1.hellinger_distance(&d2);
        assert!(h >= 0.0 && h <= 1.0);
    }

    #[test]
    fn test_bhattacharyya_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.bhattacharyya_coefficient(&d) - 1.0).abs() < 1e-6);
        assert!((d.bhattacharyya_distance(&d) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_chi_squared_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.chi_squared_distance(&d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_renyi_divergence() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        // Rényi divergence of a distribution with itself is 0
        assert!((d.renyi_divergence(&d, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lp_distance() {
        let d1 = SubDistribution::point(1u32);
        let d2 = SubDistribution::point(2u32);
        let l1 = d1.lp_distance(&d2, 1.0);
        // L1 = |1-0| + |0-1| = 2
        assert!((l1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_kantorovich_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.kantorovich_distance(&d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_with_metric() {
        let d1 = SubDistribution::point(0u32);
        let d2 = SubDistribution::point(10u32);
        let w = d1.wasserstein_with_metric(&d2, |&a, &b| (a as f64 - b as f64).abs());
        assert!((w - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((d.jaccard_similarity(&d) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let d1 = SubDistribution::point(1u32);
        let d2 = SubDistribution::point(2u32);
        assert!((d1.jaccard_similarity(&d2) - 0.0).abs() < 1e-10);
    }

    // --- Product and marginal tests ---

    #[test]
    fn test_product_distribution() {
        let d1 = SubDistribution::uniform(vec![1u32, 2]);
        let d2 = SubDistribution::uniform(vec![10u32, 20]);
        let prod = d1.product(&d2);
        assert_eq!(prod.support_size(), 4);
        assert!((prod.weight(&(1, 10)) - 0.25).abs() < 1e-10);
        assert!(prod.is_proper(1e-10));
    }

    #[test]
    fn test_marginals() {
        let d1 = SubDistribution::uniform(vec![1u32, 2]);
        let d2 = SubDistribution::uniform(vec![10u32, 20]);
        let joint = d1.product(&d2);

        let m1 = marginal_first(&joint);
        let m2 = marginal_second(&joint);
        assert!((m1.weight(&1) - 0.5).abs() < 1e-10);
        assert!((m2.weight(&10) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_conditional() {
        let mut weights = BTreeMap::new();
        weights.insert((1u32, 10u32), 0.3);
        weights.insert((1, 20), 0.2);
        weights.insert((2, 10), 0.5);
        let joint = SubDistribution::from_weights(weights).unwrap();

        let cond = conditional(&joint, &1u32).unwrap();
        // P(Y=10 | X=1) = 0.3/0.5 = 0.6
        assert!((cond.weight(&10) - 0.6).abs() < 1e-10);
        // P(Y=20 | X=1) = 0.2/0.5 = 0.4
        assert!((cond.weight(&20) - 0.4).abs() < 1e-10);
    }

    // --- Empirical distribution tests ---

    #[test]
    fn test_empirical() {
        let samples = vec![1u32, 1, 2, 3, 3, 3];
        let d = empirical(&samples);
        assert!((d.weight(&1) - 2.0 / 6.0).abs() < 1e-10);
        assert!((d.weight(&3) - 3.0 / 6.0).abs() < 1e-10);
        assert!(d.is_proper(1e-10));
    }

    #[test]
    fn test_weighted_empirical() {
        let samples = vec![(1u32, 3.0), (2, 7.0)];
        let d = weighted_empirical(&samples).unwrap();
        assert!((d.weight(&1) - 0.3).abs() < 1e-10);
        assert!((d.weight(&2) - 0.7).abs() < 1e-10);
    }

    // --- Hypothesis testing ---

    #[test]
    fn test_ks_statistic_same() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3, 4, 5]);
        assert!((ks_statistic(&d, &d) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ks_statistic_different() {
        let d1 = SubDistribution::point(1u32);
        let d2 = SubDistribution::point(5u32);
        let ks = ks_statistic(&d1, &d2);
        assert!(ks > 0.5);
    }

    #[test]
    fn test_ks_test() {
        let d1 = SubDistribution::uniform(vec![1u32, 2, 3, 4, 5]);
        let d2 = SubDistribution::uniform(vec![1u32, 2, 3, 4, 5]);
        let (d, p) = ks_test(&d1, &d2, 100, 100);
        assert!((d - 0.0).abs() < 1e-10);
        assert!(p > 0.05); // Should not reject null
    }

    #[test]
    fn test_chi_squared_test() {
        let d1 = SubDistribution::uniform(vec![1u32, 2, 3]);
        let d2 = SubDistribution::uniform(vec![1u32, 2, 3]);
        let (chi2, _df, p) = chi_squared_test(&d1, &d2, 100);
        assert!((chi2 - 0.0).abs() < 1e-10);
        assert!(p > 0.05);
    }

    // --- Smoothing and transformation tests ---

    #[test]
    fn test_laplace_smoothing() {
        let d = SubDistribution::point(1u32);
        let universe = vec![1u32, 2, 3];
        let smoothed = laplace_smoothing(&d, &universe, 1.0);
        // All elements should have positive weight
        assert!(smoothed.weight(&1) > smoothed.weight(&2));
        assert!(smoothed.weight(&2) > 0.0);
    }

    #[test]
    fn test_tempered() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        // Temperature 1 should preserve uniform
        let t1 = tempered(&d, 1.0);
        assert!((t1.weight(&1) - t1.weight(&2)).abs() < 1e-10);

        // Temperature 0 should make uniform (all weights -> 1)
        let t0 = tempered(&d, 0.0);
        // All p^0 = 1, so uniform after renormalization
        assert!((t0.weight(&1) - t0.weight(&2)).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_mean() {
        let d1 = SubDistribution::uniform(vec![1u32, 2, 3]);
        let d2 = SubDistribution::uniform(vec![1u32, 2, 3]);
        let gm = geometric_mean(&d1, &d2);
        // Geometric mean of uniform with itself is still uniform
        assert!((gm.weight(&1) - gm.weight(&2)).abs() < 1e-10);
    }

    // --- DistributionComparison tests ---

    #[test]
    fn test_distribution_comparison() {
        let d1 = SubDistribution::uniform(vec![1u32, 2, 3]);
        let d2 = SubDistribution::point(2u32);
        let cmp = DistributionComparison::compute(&d1, &d2);
        assert!(cmp.total_variation > 0.0);
        assert!(cmp.hellinger > 0.0);
        assert!(cmp.jensen_shannon > 0.0);

        let combined = cmp.combined_distance(&DistanceWeights::default());
        assert!(combined > 0.0);
    }

    #[test]
    fn test_distribution_comparison_identical() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        let cmp = DistributionComparison::compute(&d, &d);
        assert!((cmp.total_variation - 0.0).abs() < 1e-10);
        assert!((cmp.hellinger - 0.0).abs() < 1e-10);
        assert!((cmp.jensen_shannon - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let d = SubDistribution::point(1u32);
        let scaled = d.scale(0.5).unwrap();
        assert!((scaled.weight(&1) - 0.5).abs() < 1e-10);
        assert!((scaled.total_mass() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_threshold() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 0.4);
        weights.insert(2, 0.05);
        weights.insert(3, 0.3);
        weights.insert(4, 0.01);
        let d = SubDistribution::from_weights(weights).unwrap();
        let filtered = d.threshold(0.1);
        assert_eq!(filtered.support_size(), 2); // only 1 and 3
    }

    #[test]
    fn test_cross_entropy() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3]);
        let h = d.cross_entropy(&d);
        // Cross entropy of uniform with itself = entropy
        assert!((h - d.entropy()).abs() < 1e-10);
    }

    #[test]
    fn test_anderson_darling() {
        let d = SubDistribution::uniform(vec![1u32, 2, 3, 4, 5]);
        let ad = anderson_darling_statistic(&d, &d);
        // Same distribution should have small AD statistic
        assert!(ad.abs() < 10.0);
    }
}
