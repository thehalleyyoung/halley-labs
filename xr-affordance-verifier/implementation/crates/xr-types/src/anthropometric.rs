//! Anthropometric database types based on ANSUR-II data.
//!
//! Provides percentile tables for the five body parameters
//! (stature, arm_length, shoulder_breadth, forearm_length, hand_length),
//! correlation matrices, and sampling functions (uniform, stratified,
//! Latin Hypercube).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::kinematic::BodyParameters;
use crate::{ParamVec, NUM_BODY_PARAMS};

// ---------------------------------------------------------------------------
// Constants — ANSUR-II reference data (combined male + female)
// ---------------------------------------------------------------------------

/// Number of tabulated percentiles.
const NUM_PERCENTILES: usize = 19;

/// Percentile keys: 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99
const PERCENTILE_KEYS: [f64; NUM_PERCENTILES] = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
    // sentinel entries for interpolation at the tails
    0.001, 0.999,
];

/// Stature (m) at percentiles.
const STATURE_TABLE: [f64; NUM_PERCENTILES] = [
    1.511, 1.537, 1.562, 1.580, 1.596, 1.610, 1.624, 1.650, 1.690,
    1.720, 1.750, 1.765, 1.780, 1.800, 1.830, 1.860, 1.905,
    1.480, 1.930,
];

/// Arm (upper-arm) length (m) at percentiles.
const ARM_LENGTH_TABLE: [f64; NUM_PERCENTILES] = [
    0.299, 0.310, 0.318, 0.323, 0.327, 0.330, 0.334, 0.340, 0.348,
    0.356, 0.364, 0.368, 0.372, 0.378, 0.385, 0.395, 0.410,
    0.290, 0.420,
];

/// Shoulder breadth (bi-deltoid, m) at percentiles.
const SHOULDER_BREADTH_TABLE: [f64; NUM_PERCENTILES] = [
    0.375, 0.390, 0.402, 0.410, 0.416, 0.422, 0.428, 0.438, 0.450,
    0.462, 0.474, 0.480, 0.486, 0.494, 0.505, 0.520, 0.545,
    0.365, 0.560,
];

/// Forearm length (m) at percentiles.
const FOREARM_LENGTH_TABLE: [f64; NUM_PERCENTILES] = [
    0.218, 0.226, 0.232, 0.236, 0.239, 0.242, 0.245, 0.250, 0.256,
    0.262, 0.268, 0.271, 0.275, 0.280, 0.286, 0.295, 0.308,
    0.212, 0.315,
];

/// Hand length (m) at percentiles.
const HAND_LENGTH_TABLE: [f64; NUM_PERCENTILES] = [
    0.163, 0.168, 0.172, 0.175, 0.177, 0.179, 0.181, 0.184, 0.188,
    0.192, 0.196, 0.198, 0.200, 0.203, 0.207, 0.213, 0.222,
    0.158, 0.228,
];

/// Approximate Pearson correlation matrix for the 5 body params.
/// Row/col order: stature, arm_length, shoulder_breadth, forearm_length, hand_length.
const CORRELATION_MATRIX: [[f64; 5]; 5] = [
    [1.000, 0.870, 0.710, 0.850, 0.780],
    [0.870, 1.000, 0.620, 0.820, 0.750],
    [0.710, 0.620, 1.000, 0.600, 0.550],
    [0.850, 0.820, 0.600, 1.000, 0.720],
    [0.780, 0.750, 0.550, 0.720, 1.000],
];

// ---------------------------------------------------------------------------
// Percentile table utilities
// ---------------------------------------------------------------------------

fn all_tables() -> [&'static [f64; NUM_PERCENTILES]; 5] {
    [
        &STATURE_TABLE,
        &ARM_LENGTH_TABLE,
        &SHOULDER_BREADTH_TABLE,
        &FOREARM_LENGTH_TABLE,
        &HAND_LENGTH_TABLE,
    ]
}

/// Linearly interpolate a value at a given percentile from a table.
fn interpolate_percentile(table: &[f64; NUM_PERCENTILES], percentile: f64) -> f64 {
    let p = percentile.clamp(PERCENTILE_KEYS[NUM_PERCENTILES - 1].min(PERCENTILE_KEYS[0]),
                              PERCENTILE_KEYS.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    // Sort the key-value pairs by percentile.
    let mut pairs: Vec<(f64, f64)> = PERCENTILE_KEYS
        .iter()
        .copied()
        .zip(table.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Find bracketing interval.
    if p <= pairs[0].0 {
        return pairs[0].1;
    }
    if p >= pairs[pairs.len() - 1].0 {
        return pairs[pairs.len() - 1].1;
    }
    for i in 0..pairs.len() - 1 {
        if p >= pairs[i].0 && p <= pairs[i + 1].0 {
            let t = (p - pairs[i].0) / (pairs[i + 1].0 - pairs[i].0);
            return pairs[i].1 + t * (pairs[i + 1].1 - pairs[i].1);
        }
    }
    pairs[pairs.len() / 2].1
}

/// Inverse percentile lookup: given a measurement value, return the approximate percentile.
fn inverse_percentile(table: &[f64; NUM_PERCENTILES], value: f64) -> f64 {
    let mut pairs: Vec<(f64, f64)> = PERCENTILE_KEYS
        .iter()
        .copied()
        .zip(table.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if value <= pairs[0].1 {
        return pairs[0].0;
    }
    if value >= pairs[pairs.len() - 1].1 {
        return pairs[pairs.len() - 1].0;
    }
    for i in 0..pairs.len() - 1 {
        if value >= pairs[i].1 && value <= pairs[i + 1].1 {
            let t = (value - pairs[i].1) / (pairs[i + 1].1 - pairs[i].1);
            return pairs[i].0 + t * (pairs[i + 1].0 - pairs[i].0);
        }
    }
    0.5
}

// ---------------------------------------------------------------------------
// AnthropometricSample
// ---------------------------------------------------------------------------

/// A single sample from the anthropometric distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropometricSample {
    /// The body parameters.
    pub params: BodyParameters,
    /// Percentile for each parameter (0-1).
    pub percentiles: [f64; NUM_BODY_PARAMS],
    /// Stratum index if from stratified sampling.
    pub stratum: Option<usize>,
    /// Weight for importance sampling.
    pub weight: f64,
}

impl AnthropometricSample {
    /// Create from body parameters, computing percentiles from the database.
    pub fn from_params(params: BodyParameters) -> Self {
        let arr = params.to_array();
        let tables = all_tables();
        let mut percentiles = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            percentiles[i] = inverse_percentile(tables[i], arr[i]);
        }
        Self {
            params,
            percentiles,
            stratum: None,
            weight: 1.0,
        }
    }

    /// Create from percentiles, looking up body parameters.
    pub fn from_percentiles(percentiles: [f64; NUM_BODY_PARAMS]) -> Self {
        let tables = all_tables();
        let mut vals = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            vals[i] = interpolate_percentile(tables[i], percentiles[i]);
        }
        let params = BodyParameters::from_array(&vals);
        Self {
            params,
            percentiles,
            stratum: None,
            weight: 1.0,
        }
    }

    /// Convert body parameters to a ParamVec.
    pub fn to_param_vec(&self) -> ParamVec {
        let arr = self.params.to_array();
        ParamVec::from_row_slice(&arr)
    }
}

// ---------------------------------------------------------------------------
// PopulationDistribution
// ---------------------------------------------------------------------------

/// Describes a sub-population within the ANSUR-II parameter space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationDistribution {
    /// Lower percentile bound per parameter.
    pub lower_percentiles: [f64; NUM_BODY_PARAMS],
    /// Upper percentile bound per parameter.
    pub upper_percentiles: [f64; NUM_BODY_PARAMS],
    /// Mean body parameters (50th percentile of this sub-population).
    pub mean_params: BodyParameters,
    /// Standard deviations per parameter (approximate).
    pub std_devs: [f64; NUM_BODY_PARAMS],
}

impl PopulationDistribution {
    /// Default population covering the 5th–95th percentile range.
    pub fn default_range() -> Self {
        let lower = [0.05; NUM_BODY_PARAMS];
        let upper = [0.95; NUM_BODY_PARAMS];
        let tables = all_tables();
        let mut mean = [0.0; NUM_BODY_PARAMS];
        let mut std_devs = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            mean[i] = interpolate_percentile(tables[i], 0.50);
            let lo = interpolate_percentile(tables[i], 0.05);
            let hi = interpolate_percentile(tables[i], 0.95);
            // Approximate std from 5th–95th range ≈ 3.29 σ
            std_devs[i] = (hi - lo) / 3.29;
        }
        Self {
            lower_percentiles: lower,
            upper_percentiles: upper,
            mean_params: BodyParameters::from_array(&mean),
            std_devs,
        }
    }

    /// Parameter-space bounding box for this distribution.
    pub fn parameter_bounds(&self) -> (ParamVec, ParamVec) {
        let tables = all_tables();
        let mut lower = [0.0; NUM_BODY_PARAMS];
        let mut upper = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            lower[i] = interpolate_percentile(tables[i], self.lower_percentiles[i]);
            upper[i] = interpolate_percentile(tables[i], self.upper_percentiles[i]);
        }
        (ParamVec::from_row_slice(&lower), ParamVec::from_row_slice(&upper))
    }

    /// Volume of the parameter-space bounding box.
    pub fn parameter_volume(&self) -> f64 {
        let (lo, hi) = self.parameter_bounds();
        let mut vol = 1.0;
        for i in 0..NUM_BODY_PARAMS {
            vol *= (hi[i] - lo[i]).max(0.0);
        }
        vol
    }
}

// ---------------------------------------------------------------------------
// StratifiedSampler
// ---------------------------------------------------------------------------

/// Stratified sampler with optional adaptive refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedSampler {
    /// Number of strata per dimension.
    pub strata_per_dim: usize,
    /// Lower percentile bound per parameter.
    pub lower: [f64; NUM_BODY_PARAMS],
    /// Upper percentile bound per parameter.
    pub upper: [f64; NUM_BODY_PARAMS],
    /// Per-stratum failure counts (for adaptive refinement).
    pub failure_counts: HashMap<usize, usize>,
    /// Per-stratum total counts.
    pub total_counts: HashMap<usize, usize>,
}

impl StratifiedSampler {
    /// Create a new stratified sampler for the 5th–95th percentile range.
    pub fn new(strata_per_dim: usize) -> Self {
        Self {
            strata_per_dim,
            lower: [0.05; NUM_BODY_PARAMS],
            upper: [0.95; NUM_BODY_PARAMS],
            failure_counts: HashMap::new(),
            total_counts: HashMap::new(),
        }
    }

    /// Create with custom percentile bounds.
    pub fn with_bounds(
        strata_per_dim: usize,
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
    ) -> Self {
        Self {
            strata_per_dim,
            lower,
            upper,
            failure_counts: HashMap::new(),
            total_counts: HashMap::new(),
        }
    }

    /// Total number of strata.
    pub fn total_strata(&self) -> usize {
        self.strata_per_dim.pow(NUM_BODY_PARAMS as u32)
    }

    /// Convert a linear stratum index to a multi-dimensional index.
    pub fn stratum_to_indices(&self, stratum: usize) -> [usize; NUM_BODY_PARAMS] {
        let mut indices = [0usize; NUM_BODY_PARAMS];
        let mut remaining = stratum;
        for i in (0..NUM_BODY_PARAMS).rev() {
            indices[i] = remaining % self.strata_per_dim;
            remaining /= self.strata_per_dim;
        }
        indices
    }

    /// Convert a multi-dimensional index to a linear stratum index.
    pub fn indices_to_stratum(&self, indices: &[usize; NUM_BODY_PARAMS]) -> usize {
        let mut stratum = 0;
        for i in 0..NUM_BODY_PARAMS {
            stratum = stratum * self.strata_per_dim + indices[i];
        }
        stratum
    }

    /// Get the percentile bounds for a given stratum.
    pub fn stratum_bounds(&self, stratum: usize) -> ([f64; NUM_BODY_PARAMS], [f64; NUM_BODY_PARAMS]) {
        let indices = self.stratum_to_indices(stratum);
        let mut lo = [0.0; NUM_BODY_PARAMS];
        let mut hi = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            let range = self.upper[i] - self.lower[i];
            let step = range / self.strata_per_dim as f64;
            lo[i] = self.lower[i] + indices[i] as f64 * step;
            hi[i] = lo[i] + step;
        }
        (lo, hi)
    }

    /// Generate one sample from a specific stratum using the stratum midpoint.
    pub fn sample_stratum_center(&self, stratum: usize) -> AnthropometricSample {
        let (lo, hi) = self.stratum_bounds(stratum);
        let mut percentiles = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            percentiles[i] = (lo[i] + hi[i]) * 0.5;
        }
        let mut sample = AnthropometricSample::from_percentiles(percentiles);
        sample.stratum = Some(stratum);
        sample
    }

    /// Generate a random sample within a stratum (deterministic given seed).
    pub fn sample_stratum_random(&self, stratum: usize, seed: u64) -> AnthropometricSample {
        let (lo, hi) = self.stratum_bounds(stratum);
        // Simple deterministic pseudo-random using a linear congruential generator.
        let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut percentiles = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = (rng_state >> 33) as f64 / (u32::MAX as f64);
            percentiles[i] = lo[i] + t * (hi[i] - lo[i]);
        }
        let mut sample = AnthropometricSample::from_percentiles(percentiles);
        sample.stratum = Some(stratum);
        sample
    }

    /// Record a result for adaptive refinement.
    pub fn record_result(&mut self, stratum: usize, failed: bool) {
        *self.total_counts.entry(stratum).or_insert(0) += 1;
        if failed {
            *self.failure_counts.entry(stratum).or_insert(0) += 1;
        }
    }

    /// Get strata sorted by failure rate (highest first) for adaptive refinement.
    pub fn strata_by_failure_rate(&self) -> Vec<(usize, f64)> {
        let mut rates: Vec<(usize, f64)> = self
            .total_counts
            .iter()
            .map(|(&stratum, &total)| {
                let failures = self.failure_counts.get(&stratum).copied().unwrap_or(0);
                let rate = if total > 0 {
                    failures as f64 / total as f64
                } else {
                    0.0
                };
                (stratum, rate)
            })
            .collect();
        rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rates
    }

    /// Generate all stratum center samples.
    pub fn all_center_samples(&self) -> Vec<AnthropometricSample> {
        (0..self.total_strata())
            .map(|s| self.sample_stratum_center(s))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AnthropometricDatabase
// ---------------------------------------------------------------------------

/// ANSUR-II anthropometric database with percentile tables and sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropometricDatabase {
    /// Population distribution.
    pub distribution: PopulationDistribution,
    /// Correlation matrix (5×5).
    pub correlation_matrix: [[f64; NUM_BODY_PARAMS]; NUM_BODY_PARAMS],
    /// Parameter names.
    pub parameter_names: [String; NUM_BODY_PARAMS],
}

impl Default for AnthropometricDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropometricDatabase {
    /// Create the database with ANSUR-II reference data.
    pub fn new() -> Self {
        Self {
            distribution: PopulationDistribution::default_range(),
            correlation_matrix: CORRELATION_MATRIX,
            parameter_names: [
                "stature".into(),
                "arm_length".into(),
                "shoulder_breadth".into(),
                "forearm_length".into(),
                "hand_length".into(),
            ],
        }
    }

    /// Look up body parameters at a given percentile (same percentile for all params).
    pub fn at_percentile(&self, percentile: f64) -> BodyParameters {
        let tables = all_tables();
        let mut vals = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            vals[i] = interpolate_percentile(tables[i], percentile);
        }
        BodyParameters::from_array(&vals)
    }

    /// Look up body parameters with per-parameter percentiles.
    pub fn at_percentiles(&self, percentiles: &[f64; NUM_BODY_PARAMS]) -> BodyParameters {
        let tables = all_tables();
        let mut vals = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            vals[i] = interpolate_percentile(tables[i], percentiles[i]);
        }
        BodyParameters::from_array(&vals)
    }

    /// Inverse lookup: find the percentile for each parameter.
    pub fn percentiles_of(&self, params: &BodyParameters) -> [f64; NUM_BODY_PARAMS] {
        let arr = params.to_array();
        let tables = all_tables();
        let mut pcts = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            pcts[i] = inverse_percentile(tables[i], arr[i]);
        }
        pcts
    }

    /// Check if body params fall within the given percentile range.
    pub fn is_in_range(&self, params: &BodyParameters, low: f64, high: f64) -> bool {
        let pcts = self.percentiles_of(params);
        pcts.iter().all(|&p| p >= low && p <= high)
    }

    /// Generate uniform samples within the configured percentile range.
    pub fn sample_uniform(&self, n: usize, seed: u64) -> Vec<AnthropometricSample> {
        let mut samples = Vec::with_capacity(n);
        let mut rng_state = seed;
        for _ in 0..n {
            let mut percentiles = [0.0; NUM_BODY_PARAMS];
            for i in 0..NUM_BODY_PARAMS {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let t = (rng_state >> 33) as f64 / (u32::MAX as f64);
                percentiles[i] = self.distribution.lower_percentiles[i]
                    + t * (self.distribution.upper_percentiles[i]
                        - self.distribution.lower_percentiles[i]);
            }
            samples.push(AnthropometricSample::from_percentiles(percentiles));
        }
        samples
    }

    /// Generate stratified samples using the given sampler.
    pub fn sample_stratified(
        &self,
        strata_per_dim: usize,
        seed: u64,
    ) -> Vec<AnthropometricSample> {
        let sampler = StratifiedSampler::with_bounds(
            strata_per_dim,
            self.distribution.lower_percentiles,
            self.distribution.upper_percentiles,
        );
        let total = sampler.total_strata();
        (0..total)
            .map(|s| sampler.sample_stratum_random(s, seed.wrapping_add(s as u64)))
            .collect()
    }

    /// Generate Latin Hypercube samples.
    pub fn sample_latin_hypercube(&self, n: usize, seed: u64) -> Vec<AnthropometricSample> {
        let mut samples = Vec::with_capacity(n);
        // For each dimension, create n equally-spaced intervals and shuffle.
        let mut permutations = vec![vec![0usize; n]; NUM_BODY_PARAMS];
        for dim in 0..NUM_BODY_PARAMS {
            for i in 0..n {
                permutations[dim][i] = i;
            }
            // Fisher-Yates shuffle with deterministic seed.
            let mut rng_state = seed.wrapping_add(dim as u64 * 999983);
            for i in (1..n).rev() {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                permutations[dim].swap(i, j);
            }
        }

        for i in 0..n {
            let mut percentiles = [0.0; NUM_BODY_PARAMS];
            for dim in 0..NUM_BODY_PARAMS {
                let lo = self.distribution.lower_percentiles[dim];
                let hi = self.distribution.upper_percentiles[dim];
                let idx = permutations[dim][i];
                // Center of the interval.
                let t = (idx as f64 + 0.5) / n as f64;
                percentiles[dim] = lo + t * (hi - lo);
            }
            samples.push(AnthropometricSample::from_percentiles(percentiles));
        }
        samples
    }

    /// Get the mean body parameters.
    pub fn mean_params(&self) -> &BodyParameters {
        &self.distribution.mean_params
    }

    /// Get the correlation between two parameters by index.
    pub fn correlation(&self, i: usize, j: usize) -> f64 {
        if i < NUM_BODY_PARAMS && j < NUM_BODY_PARAMS {
            self.correlation_matrix[i][j]
        } else {
            0.0
        }
    }

    /// Compute basic statistics over a set of samples.
    pub fn compute_statistics(samples: &[AnthropometricSample]) -> PopulationStatistics {
        if samples.is_empty() {
            return PopulationStatistics::empty();
        }
        let n = samples.len() as f64;
        let mut mean = [0.0; NUM_BODY_PARAMS];
        let mut min = [f64::INFINITY; NUM_BODY_PARAMS];
        let mut max = [f64::NEG_INFINITY; NUM_BODY_PARAMS];

        for s in samples {
            let arr = s.params.to_array();
            for i in 0..NUM_BODY_PARAMS {
                mean[i] += arr[i];
                min[i] = min[i].min(arr[i]);
                max[i] = max[i].max(arr[i]);
            }
        }
        for i in 0..NUM_BODY_PARAMS {
            mean[i] /= n;
        }

        let mut variance = [0.0; NUM_BODY_PARAMS];
        for s in samples {
            let arr = s.params.to_array();
            for i in 0..NUM_BODY_PARAMS {
                variance[i] += (arr[i] - mean[i]).powi(2);
            }
        }
        for i in 0..NUM_BODY_PARAMS {
            variance[i] /= n;
        }

        PopulationStatistics {
            count: samples.len(),
            mean,
            variance,
            min,
            max,
        }
    }

    /// Parameter-space bounds for the configured population.
    pub fn parameter_bounds(&self) -> (ParamVec, ParamVec) {
        self.distribution.parameter_bounds()
    }

    /// Total volume of the parameter space.
    pub fn parameter_volume(&self) -> f64 {
        self.distribution.parameter_volume()
    }
}

// ---------------------------------------------------------------------------
// PopulationStatistics
// ---------------------------------------------------------------------------

/// Summary statistics over a population sample set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStatistics {
    /// Number of samples.
    pub count: usize,
    /// Mean per parameter.
    pub mean: [f64; NUM_BODY_PARAMS],
    /// Variance per parameter.
    pub variance: [f64; NUM_BODY_PARAMS],
    /// Minimum per parameter.
    pub min: [f64; NUM_BODY_PARAMS],
    /// Maximum per parameter.
    pub max: [f64; NUM_BODY_PARAMS],
}

impl PopulationStatistics {
    /// Create empty statistics.
    pub fn empty() -> Self {
        Self {
            count: 0,
            mean: [0.0; NUM_BODY_PARAMS],
            variance: [0.0; NUM_BODY_PARAMS],
            min: [0.0; NUM_BODY_PARAMS],
            max: [0.0; NUM_BODY_PARAMS],
        }
    }

    /// Standard deviation per parameter.
    pub fn std_dev(&self) -> [f64; NUM_BODY_PARAMS] {
        let mut sd = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            sd[i] = self.variance[i].sqrt();
        }
        sd
    }
}

impl std::fmt::Display for PopulationStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = ["stature", "arm_length", "shoulder_breadth", "forearm_length", "hand_length"];
        writeln!(f, "Population statistics (n={}):", self.count)?;
        for i in 0..NUM_BODY_PARAMS {
            writeln!(
                f,
                "  {}: mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
                names[i],
                self.mean[i],
                self.variance[i].sqrt(),
                self.min[i],
                self.max[i],
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = AnthropometricDatabase::new();
        assert_eq!(db.parameter_names[0], "stature");
        assert_eq!(db.correlation_matrix[0][0], 1.0);
    }

    #[test]
    fn test_at_percentile_50th() {
        let db = AnthropometricDatabase::new();
        let params = db.at_percentile(0.50);
        // 50th percentile stature should be around 1.69 m.
        assert!((params.stature - 1.69).abs() < 0.01);
    }

    #[test]
    fn test_at_percentile_5th() {
        let db = AnthropometricDatabase::new();
        let params = db.at_percentile(0.05);
        assert!(params.stature < 1.6);
    }

    #[test]
    fn test_at_percentile_95th() {
        let db = AnthropometricDatabase::new();
        let params = db.at_percentile(0.95);
        assert!(params.stature > 1.8);
    }

    #[test]
    fn test_at_percentiles_independent() {
        let db = AnthropometricDatabase::new();
        let params = db.at_percentiles(&[0.05, 0.50, 0.50, 0.50, 0.95]);
        assert!(params.stature < 1.6);
        assert!(params.hand_length > 0.20);
    }

    #[test]
    fn test_percentile_roundtrip() {
        let db = AnthropometricDatabase::new();
        let params = db.at_percentile(0.50);
        let pcts = db.percentiles_of(&params);
        for &p in &pcts {
            assert!((p - 0.50).abs() < 0.05);
        }
    }

    #[test]
    fn test_is_in_range() {
        let db = AnthropometricDatabase::new();
        let params_50 = db.at_percentile(0.50);
        assert!(db.is_in_range(&params_50, 0.05, 0.95));

        let params_01 = db.at_percentile(0.01);
        assert!(!db.is_in_range(&params_01, 0.05, 0.95));
    }

    #[test]
    fn test_sample_uniform() {
        let db = AnthropometricDatabase::new();
        let samples = db.sample_uniform(100, 42);
        assert_eq!(samples.len(), 100);
        for s in &samples {
            for &p in &s.percentiles {
                assert!(p >= 0.04 && p <= 0.96);
            }
        }
    }

    #[test]
    fn test_sample_stratified() {
        let db = AnthropometricDatabase::new();
        let samples = db.sample_stratified(2, 42);
        assert_eq!(samples.len(), 32); // 2^5 = 32
    }

    #[test]
    fn test_sample_latin_hypercube() {
        let db = AnthropometricDatabase::new();
        let samples = db.sample_latin_hypercube(10, 42);
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_stratified_sampler_indices() {
        let sampler = StratifiedSampler::new(3);
        let idx = sampler.stratum_to_indices(0);
        assert_eq!(idx, [0, 0, 0, 0, 0]);

        let back = sampler.indices_to_stratum(&idx);
        assert_eq!(back, 0);

        let idx2 = sampler.stratum_to_indices(1);
        assert_eq!(idx2, [0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_stratified_sampler_total() {
        let sampler = StratifiedSampler::new(3);
        assert_eq!(sampler.total_strata(), 243); // 3^5
    }

    #[test]
    fn test_stratified_sampler_bounds() {
        let sampler = StratifiedSampler::new(2);
        let (lo, hi) = sampler.stratum_bounds(0);
        // First stratum of 2: 0.05 to 0.50 per dimension.
        for i in 0..NUM_BODY_PARAMS {
            assert!((lo[i] - 0.05).abs() < 1e-10);
            assert!((hi[i] - 0.50).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stratified_adaptive() {
        let mut sampler = StratifiedSampler::new(2);
        sampler.record_result(0, true);
        sampler.record_result(0, true);
        sampler.record_result(0, false);
        sampler.record_result(1, false);
        sampler.record_result(1, false);

        let rates = sampler.strata_by_failure_rate();
        assert_eq!(rates[0].0, 0);
        assert!((rates[0].1 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_from_params() {
        let params = BodyParameters::average_male();
        let sample = AnthropometricSample::from_params(params);
        // Average male should be near the 50th percentile.
        for &p in &sample.percentiles {
            assert!(p > 0.3 && p < 0.8);
        }
    }

    #[test]
    fn test_compute_statistics() {
        let db = AnthropometricDatabase::new();
        let samples = db.sample_uniform(1000, 42);
        let stats = AnthropometricDatabase::compute_statistics(&samples);
        assert_eq!(stats.count, 1000);
        // Mean stature should be roughly in the middle.
        assert!(stats.mean[0] > 1.55 && stats.mean[0] < 1.85);
        // Variance should be small but positive.
        for &v in &stats.variance {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_statistics_display() {
        let db = AnthropometricDatabase::new();
        let samples = db.sample_uniform(50, 42);
        let stats = AnthropometricDatabase::compute_statistics(&samples);
        let s = format!("{stats}");
        assert!(s.contains("stature"));
        assert!(s.contains("n=50"));
    }

    #[test]
    fn test_population_distribution_volume() {
        let dist = PopulationDistribution::default_range();
        let vol = dist.parameter_volume();
        assert!(vol > 0.0);
    }

    #[test]
    fn test_correlation() {
        let db = AnthropometricDatabase::new();
        assert!((db.correlation(0, 0) - 1.0).abs() < 1e-10);
        assert!(db.correlation(0, 1) > 0.5);
        assert_eq!(db.correlation(10, 0), 0.0);
    }

    #[test]
    fn test_mean_params() {
        let db = AnthropometricDatabase::new();
        let mean = db.mean_params();
        assert!((mean.stature - 1.69).abs() < 0.01);
    }

    #[test]
    fn test_parameter_bounds() {
        let db = AnthropometricDatabase::new();
        let (lo, hi) = db.parameter_bounds();
        for i in 0..NUM_BODY_PARAMS {
            assert!(lo[i] < hi[i]);
        }
    }

    #[test]
    fn test_database_serde_roundtrip() {
        let db = AnthropometricDatabase::new();
        let json = serde_json::to_string(&db).unwrap();
        let back: AnthropometricDatabase = serde_json::from_str(&json).unwrap();
        assert_eq!(db.parameter_names, back.parameter_names);
    }

    #[test]
    fn test_lhs_covers_space() {
        let db = AnthropometricDatabase::new();
        let samples = db.sample_latin_hypercube(20, 1);
        let stats = AnthropometricDatabase::compute_statistics(&samples);
        // Should cover a reasonable range.
        assert!(stats.max[0] - stats.min[0] > 0.1);
    }

    #[test]
    fn test_all_center_samples() {
        let sampler = StratifiedSampler::new(2);
        let centers = sampler.all_center_samples();
        assert_eq!(centers.len(), 32);
        for c in &centers {
            assert!(c.stratum.is_some());
        }
    }

    #[test]
    fn test_to_param_vec() {
        let sample = AnthropometricSample::from_percentiles([0.5; 5]);
        let pv = sample.to_param_vec();
        assert_eq!(pv.len(), 5);
    }
}
