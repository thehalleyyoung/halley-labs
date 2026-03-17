//! Stratified sampling engine for the body-parameter space Θ\_target.
//!
//! Implements adaptive stratified sampling, Latin Hypercube Sampling (LHS),
//! and quasi-random Halton sequences over the 5-dimensional body parameter
//! space defined by ANSUR-II anthropometric data.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use xr_types::kinematic::BodyParameters;
use xr_types::NUM_BODY_PARAMS;

// ───────────────────────────── Stratum ──────────────────────────────────────

/// A single stratum (hyper-rectangular cell) in the body-parameter space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stratum {
    /// Unique index of this stratum.
    pub index: usize,
    /// Lower bounds for each of the 5 body parameters.
    pub lower: [f64; NUM_BODY_PARAMS],
    /// Upper bounds for each of the 5 body parameters.
    pub upper: [f64; NUM_BODY_PARAMS],
    /// Number of samples drawn from this stratum.
    pub sample_count: usize,
    /// Number of passing verdicts in this stratum.
    pub pass_count: usize,
    /// Number of failing verdicts in this stratum.
    pub fail_count: usize,
    /// Number of unknown verdicts in this stratum.
    pub unknown_count: usize,
    /// Multi-dimensional index [i0, i1, ..., i4] within the grid.
    pub grid_indices: [usize; NUM_BODY_PARAMS],
}

impl Stratum {
    /// Create a new stratum with given bounds.
    pub fn new(
        index: usize,
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
        grid_indices: [usize; NUM_BODY_PARAMS],
    ) -> Self {
        Self {
            index,
            lower,
            upper,
            sample_count: 0,
            pass_count: 0,
            fail_count: 0,
            unknown_count: 0,
            grid_indices,
        }
    }

    /// Hyper-volume of this stratum.
    pub fn volume(&self) -> f64 {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product()
    }

    /// Center point of this stratum.
    pub fn center(&self) -> [f64; NUM_BODY_PARAMS] {
        let mut c = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            c[i] = (self.lower[i] + self.upper[i]) * 0.5;
        }
        c
    }

    /// Failure rate observed so far.
    pub fn failure_rate(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.fail_count as f64 / self.sample_count as f64
        }
    }

    /// Pass rate observed so far.
    pub fn pass_rate(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.pass_count as f64 / self.sample_count as f64
        }
    }

    /// Variance estimator for the stratum proportion (p * (1-p) / n).
    pub fn variance_estimate(&self) -> f64 {
        if self.sample_count < 2 {
            return 0.25; // maximum variance for unknown proportion
        }
        let p = self.pass_rate();
        p * (1.0 - p) / self.sample_count as f64
    }

    /// Whether this stratum has mixed verdicts (both pass and fail).
    pub fn is_frontier(&self) -> bool {
        self.pass_count > 0 && self.fail_count > 0
    }

    /// Convert a center point to `BodyParameters`.
    pub fn center_body_params(&self) -> BodyParameters {
        let c = self.center();
        BodyParameters::new(c[0], c[1], c[2], c[3], c[4])
    }

    /// Record a verdict for this stratum.
    pub fn record_verdict(&mut self, passed: bool) {
        self.sample_count += 1;
        if passed {
            self.pass_count += 1;
        } else {
            self.fail_count += 1;
        }
    }

    /// Record an unknown verdict.
    pub fn record_unknown(&mut self) {
        self.sample_count += 1;
        self.unknown_count += 1;
    }

    /// Check if a point falls within this stratum.
    pub fn contains(&self, point: &[f64; NUM_BODY_PARAMS]) -> bool {
        point
            .iter()
            .zip(self.lower.iter())
            .zip(self.upper.iter())
            .all(|((&p, &lo), &hi)| p >= lo && p <= hi)
    }

    /// Sample a uniform random point within this stratum.
    pub fn sample_uniform(&self, rng: &mut impl Rng) -> [f64; NUM_BODY_PARAMS] {
        let mut point = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            point[i] = rng.gen_range(self.lower[i]..=self.upper[i]);
        }
        point
    }
}

// ───────────────────────── Halton Sequence ──────────────────────────────────

/// Generator for a Halton quasi-random sequence.
#[derive(Debug, Clone)]
pub struct HaltonSequence {
    /// Prime bases for each dimension.
    bases: Vec<u64>,
    /// Current index in the sequence.
    index: u64,
}

impl HaltonSequence {
    /// First several primes used as bases for the Halton sequence.
    const PRIMES: [u64; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

    /// Create a new Halton sequence for the given dimensionality.
    pub fn new(dimension: usize) -> Self {
        assert!(
            dimension <= Self::PRIMES.len(),
            "Halton sequence supports at most {} dimensions",
            Self::PRIMES.len()
        );
        Self {
            bases: Self::PRIMES[..dimension].to_vec(),
            index: 0,
        }
    }

    /// Compute the Halton value for a given index and base.
    fn halton_value(index: u64, base: u64) -> f64 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f64;
        let mut i = index;
        while i > 0 {
            result += (i % base) as f64 * f;
            i /= base;
            f /= base as f64;
        }
        result
    }

    /// Generate the next point in the sequence as a unit-cube point [0, 1)^d.
    pub fn next_unit(&mut self) -> Vec<f64> {
        self.index += 1;
        self.bases
            .iter()
            .map(|&b| Self::halton_value(self.index, b))
            .collect()
    }

    /// Generate the next point scaled to the given bounds.
    pub fn next_scaled(
        &mut self,
        lower: &[f64],
        upper: &[f64],
    ) -> Vec<f64> {
        let unit = self.next_unit();
        unit.iter()
            .zip(lower.iter())
            .zip(upper.iter())
            .map(|((&u, &lo), &hi)| lo + u * (hi - lo))
            .collect()
    }

    /// Generate n points in the unit cube.
    pub fn generate_unit(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_unit()).collect()
    }

    /// Generate n points scaled to bounds.
    pub fn generate_scaled(
        &mut self,
        n: usize,
        lower: &[f64],
        upper: &[f64],
    ) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_scaled(lower, upper)).collect()
    }

    /// Reset the sequence to the beginning.
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Skip ahead by `count` positions.
    pub fn skip(&mut self, count: u64) {
        self.index += count;
    }

    /// Current index in the sequence.
    pub fn current_index(&self) -> u64 {
        self.index
    }
}

// ─────────────────── Latin Hypercube Sampling ──────────────────────────────

/// Latin Hypercube Sampler for space-filling designs.
#[derive(Debug, Clone)]
pub struct LatinHypercubeSampler {
    lower: [f64; NUM_BODY_PARAMS],
    upper: [f64; NUM_BODY_PARAMS],
    seed: u64,
}

impl LatinHypercubeSampler {
    /// Create a new LHS sampler with the given parameter bounds.
    pub fn new(
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
        seed: u64,
    ) -> Self {
        Self { lower, upper, seed }
    }

    /// Generate `n` Latin Hypercube samples.
    ///
    /// The algorithm partitions each dimension into `n` equal intervals,
    /// then randomly assigns one sample to each interval in each dimension,
    /// ensuring uniform marginal coverage.
    pub fn generate(&self, n: usize) -> Vec<[f64; NUM_BODY_PARAMS]> {
        if n == 0 {
            return Vec::new();
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut result = vec![[0.0; NUM_BODY_PARAMS]; n];

        for dim in 0..NUM_BODY_PARAMS {
            let lo = self.lower[dim];
            let hi = self.upper[dim];
            let step = (hi - lo) / n as f64;

            // Create a permutation for this dimension
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                perm.swap(i, j);
            }

            // Assign each sample to its permuted interval with jitter
            for (sample_idx, &interval_idx) in perm.iter().enumerate() {
                let base = lo + interval_idx as f64 * step;
                let jitter: f64 = rng.gen_range(0.0..1.0);
                result[sample_idx][dim] = base + jitter * step;
            }
        }

        result
    }

    /// Generate `n` LHS samples as `BodyParameters`.
    pub fn generate_body_params(&self, n: usize) -> Vec<BodyParameters> {
        self.generate(n)
            .into_iter()
            .map(|p| BodyParameters::new(p[0], p[1], p[2], p[3], p[4]))
            .collect()
    }

    /// Generate `n` LHS samples with correlation adjustment.
    ///
    /// Uses the Iman-Conover method: generate LHS samples, then reorder
    /// each dimension to induce the desired rank correlation.
    pub fn generate_correlated(
        &self,
        n: usize,
        target_correlation: &[[f64; NUM_BODY_PARAMS]; NUM_BODY_PARAMS],
    ) -> Vec<[f64; NUM_BODY_PARAMS]> {
        if n < 3 {
            return self.generate(n);
        }

        let mut samples = self.generate(n);
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(12345));

        // Compute Cholesky decomposition of target correlation matrix
        let chol = match cholesky_5x5(target_correlation) {
            Some(c) => c,
            None => return samples, // fall back to uncorrelated if not PD
        };

        // Generate independent standard normal scores
        let mut normal_scores = vec![[0.0; NUM_BODY_PARAMS]; n];
        for row in &mut normal_scores {
            for val in row.iter_mut() {
                *val = standard_normal(&mut rng);
            }
        }

        // Apply Cholesky factor to induce correlation
        let mut correlated_scores = vec![[0.0; NUM_BODY_PARAMS]; n];
        for i in 0..n {
            for j in 0..NUM_BODY_PARAMS {
                let mut sum = 0.0;
                for k in 0..=j {
                    sum += chol[j][k] * normal_scores[i][k];
                }
                correlated_scores[i][j] = sum;
            }
        }

        // Rank the correlated scores and reorder the LHS samples accordingly
        for dim in 0..NUM_BODY_PARAMS {
            let mut score_indices: Vec<(usize, f64)> = correlated_scores
                .iter()
                .enumerate()
                .map(|(i, s)| (i, s[dim]))
                .collect();
            score_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut dim_values: Vec<f64> = samples.iter().map(|s| s[dim]).collect();
            dim_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (rank, &(original_idx, _)) in score_indices.iter().enumerate() {
                samples[original_idx][dim] = dim_values[rank];
            }
        }

        samples
    }
}

/// Box-Muller transform for standard normal sampling.
fn standard_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen_range(1e-10..1.0);
    let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

/// Cholesky decomposition of a 5×5 symmetric positive-definite matrix.
/// Returns `None` if the matrix is not positive definite.
fn cholesky_5x5(
    a: &[[f64; NUM_BODY_PARAMS]; NUM_BODY_PARAMS],
) -> Option<[[f64; NUM_BODY_PARAMS]; NUM_BODY_PARAMS]> {
    let n = NUM_BODY_PARAMS;
    let mut l = [[0.0; NUM_BODY_PARAMS]; NUM_BODY_PARAMS];
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j][k] * l[j][k];
        }
        let diag = a[j][j] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[j][j] = diag.sqrt();
        for i in (j + 1)..n {
            let mut sum2 = 0.0;
            for k in 0..j {
                sum2 += l[i][k] * l[j][k];
            }
            l[i][j] = (a[i][j] - sum2) / l[j][j];
        }
    }
    Some(l)
}

// ─────────────────── Stratified Sampler ────────────────────────────────────

/// Adaptive stratified sampler over the body-parameter space.
///
/// Divides the parameter space into a regular grid of strata, samples
/// within each stratum, and adaptively allocates more samples to strata
/// near the accessibility frontier.
#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    /// All strata in the grid.
    pub strata: Vec<Stratum>,
    /// Number of strata per dimension.
    pub strata_per_dim: usize,
    /// Total number of strata.
    pub total_strata: usize,
    /// Lower bounds for the parameter space.
    pub lower: [f64; NUM_BODY_PARAMS],
    /// Upper bounds for the parameter space.
    pub upper: [f64; NUM_BODY_PARAMS],
    /// Random seed.
    pub seed: u64,
    /// Budget tracking: samples allocated so far.
    pub total_samples_drawn: usize,
}

impl StratifiedSampler {
    /// Create a new stratified sampler.
    pub fn new(
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
        strata_per_dim: usize,
        seed: u64,
    ) -> Self {
        let strata = Self::create_strata_grid(&lower, &upper, strata_per_dim);
        let total_strata = strata.len();
        Self {
            strata,
            strata_per_dim,
            total_strata,
            lower,
            upper,
            seed,
            total_samples_drawn: 0,
        }
    }

    /// Create the grid of strata by dividing each dimension into `k` equal parts.
    pub fn create_strata(
        lower: &[f64; NUM_BODY_PARAMS],
        upper: &[f64; NUM_BODY_PARAMS],
        num_strata_per_dim: usize,
    ) -> Vec<Stratum> {
        Self::create_strata_grid(lower, upper, num_strata_per_dim)
    }

    fn create_strata_grid(
        lower: &[f64; NUM_BODY_PARAMS],
        upper: &[f64; NUM_BODY_PARAMS],
        k: usize,
    ) -> Vec<Stratum> {
        let total = k.pow(NUM_BODY_PARAMS as u32);
        let mut strata = Vec::with_capacity(total);

        let steps: Vec<f64> = (0..NUM_BODY_PARAMS)
            .map(|d| (upper[d] - lower[d]) / k as f64)
            .collect();

        for flat_idx in 0..total {
            let mut grid_indices = [0usize; NUM_BODY_PARAMS];
            let mut remainder = flat_idx;
            for d in (0..NUM_BODY_PARAMS).rev() {
                grid_indices[d] = remainder % k;
                remainder /= k;
            }

            let mut lo = [0.0; NUM_BODY_PARAMS];
            let mut hi = [0.0; NUM_BODY_PARAMS];
            for d in 0..NUM_BODY_PARAMS {
                lo[d] = lower[d] + grid_indices[d] as f64 * steps[d];
                hi[d] = lower[d] + (grid_indices[d] + 1) as f64 * steps[d];
            }

            strata.push(Stratum::new(flat_idx, lo, hi, grid_indices));
        }

        strata
    }

    /// Convert a flat stratum index to grid indices.
    pub fn flat_to_grid(&self, flat_idx: usize) -> [usize; NUM_BODY_PARAMS] {
        let k = self.strata_per_dim;
        let mut indices = [0usize; NUM_BODY_PARAMS];
        let mut remainder = flat_idx;
        for d in (0..NUM_BODY_PARAMS).rev() {
            indices[d] = remainder % k;
            remainder /= k;
        }
        indices
    }

    /// Convert grid indices to a flat stratum index.
    pub fn grid_to_flat(&self, indices: &[usize; NUM_BODY_PARAMS]) -> usize {
        let k = self.strata_per_dim;
        let mut flat = 0;
        for d in 0..NUM_BODY_PARAMS {
            flat = flat * k + indices[d];
        }
        flat
    }

    /// Sample `n` points uniformly from a specific stratum.
    pub fn sample_stratum(
        &mut self,
        stratum_idx: usize,
        n: usize,
    ) -> Vec<BodyParameters> {
        if stratum_idx >= self.strata.len() {
            return Vec::new();
        }
        let seed = self.seed.wrapping_add(stratum_idx as u64 * 1000 + self.total_samples_drawn as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let stratum = &self.strata[stratum_idx];
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let point = stratum.sample_uniform(&mut rng);
            result.push(BodyParameters::new(
                point[0], point[1], point[2], point[3], point[4],
            ));
        }
        self.total_samples_drawn += n;
        result
    }

    /// Sample `n` points total, distributed uniformly across all strata.
    pub fn sample_uniform(&mut self, n: usize) -> Vec<(usize, BodyParameters)> {
        let k = self.strata.len();
        if k == 0 {
            return Vec::new();
        }
        let base_per_stratum = n / k;
        let remainder = n % k;
        let mut result = Vec::with_capacity(n);

        let seed_base = self.seed.wrapping_add(self.total_samples_drawn as u64);
        let mut rng = StdRng::seed_from_u64(seed_base);

        for (idx, stratum) in self.strata.iter().enumerate() {
            let count = base_per_stratum + if idx < remainder { 1 } else { 0 };
            for _ in 0..count {
                let point = stratum.sample_uniform(&mut rng);
                result.push((
                    idx,
                    BodyParameters::new(point[0], point[1], point[2], point[3], point[4]),
                ));
            }
        }
        self.total_samples_drawn += result.len();
        result
    }

    /// Adaptive resampling: allocate more samples to strata near the frontier.
    ///
    /// The allocation strategy assigns budget proportional to:
    /// - Estimated variance (frontier strata have p ≈ 0.5 → max variance)
    /// - Stratum volume (larger strata get more samples)
    ///
    /// This implements Neyman allocation: n_s ∝ w_s · σ_s,
    /// where w_s = V_s / V_total is the volume weight and σ_s is the
    /// estimated standard deviation within the stratum.
    pub fn adaptive_resampling(
        &mut self,
        budget: usize,
    ) -> Vec<(usize, Vec<BodyParameters>)> {
        let total_volume: f64 = self.strata.iter().map(|s| s.volume()).sum();
        if total_volume <= 0.0 || self.strata.is_empty() {
            return Vec::new();
        }

        // Compute allocation weights using Neyman allocation
        let weights: Vec<f64> = self
            .strata
            .iter()
            .map(|s| {
                let w_volume = s.volume() / total_volume;
                let sigma = if s.sample_count == 0 {
                    0.5 // maximum uncertainty
                } else {
                    let p = s.pass_rate();
                    (p * (1.0 - p)).sqrt().max(0.01)
                };
                w_volume * sigma
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Vec::new();
        }

        // Allocate samples proportionally, ensuring at least 1 per frontier stratum
        let mut allocations: Vec<usize> = weights
            .iter()
            .map(|w| ((w / total_weight) * budget as f64).floor() as usize)
            .collect();

        // Distribute remaining budget to frontier strata
        let allocated: usize = allocations.iter().sum();
        let mut remaining = budget.saturating_sub(allocated);

        // Priority: frontier strata first, then uncertain strata
        let mut priority: Vec<(usize, f64)> = self
            .strata
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let score = if s.is_frontier() {
                    10.0 + weights[i]
                } else if s.sample_count == 0 {
                    5.0 + weights[i]
                } else {
                    weights[i]
                };
                (i, score)
            })
            .collect();
        priority.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(idx, _) in &priority {
            if remaining == 0 {
                break;
            }
            allocations[idx] += 1;
            remaining -= 1;
        }

        // Draw samples according to allocations
        let mut result = Vec::new();
        for (idx, &count) in allocations.iter().enumerate() {
            if count > 0 {
                let samples = self.sample_stratum(idx, count);
                result.push((idx, samples));
            }
        }

        result
    }

    /// Sample using Latin Hypercube Sampling within each stratum.
    pub fn sample_lhs_per_stratum(
        &self,
        samples_per_stratum: usize,
    ) -> Vec<(usize, Vec<BodyParameters>)> {
        let mut result = Vec::with_capacity(self.strata.len());
        for (idx, stratum) in self.strata.iter().enumerate() {
            let lhs = LatinHypercubeSampler::new(
                stratum.lower,
                stratum.upper,
                self.seed.wrapping_add(idx as u64 * 7919),
            );
            let samples = lhs.generate_body_params(samples_per_stratum);
            result.push((idx, samples));
        }
        result
    }

    /// Sample using Halton quasi-random sequence within each stratum.
    pub fn sample_halton_per_stratum(
        &self,
        samples_per_stratum: usize,
    ) -> Vec<(usize, Vec<BodyParameters>)> {
        let mut result = Vec::with_capacity(self.strata.len());
        for (idx, stratum) in self.strata.iter().enumerate() {
            let mut halton = HaltonSequence::new(NUM_BODY_PARAMS);
            halton.skip(idx as u64 * samples_per_stratum as u64);
            let points = halton.generate_scaled(
                samples_per_stratum,
                &stratum.lower,
                &stratum.upper,
            );
            let samples: Vec<BodyParameters> = points
                .into_iter()
                .map(|p| BodyParameters::new(p[0], p[1], p[2], p[3], p[4]))
                .collect();
            result.push((idx, samples));
        }
        result
    }

    /// Record a verdict for a stratum.
    pub fn record_verdict(&mut self, stratum_idx: usize, passed: bool) {
        if stratum_idx < self.strata.len() {
            self.strata[stratum_idx].record_verdict(passed);
        }
    }

    /// Find the stratum index containing a given point.
    pub fn find_stratum(&self, point: &[f64; NUM_BODY_PARAMS]) -> Option<usize> {
        let k = self.strata_per_dim;
        let mut grid_indices = [0usize; NUM_BODY_PARAMS];
        for d in 0..NUM_BODY_PARAMS {
            let range = self.upper[d] - self.lower[d];
            if range <= 0.0 {
                return None;
            }
            let normalized = (point[d] - self.lower[d]) / range;
            let idx = (normalized * k as f64).floor() as usize;
            grid_indices[d] = idx.min(k - 1);
        }
        let flat = self.grid_to_flat(&grid_indices);
        if flat < self.strata.len() {
            Some(flat)
        } else {
            None
        }
    }

    /// Get strata sorted by failure rate (descending).
    pub fn strata_by_failure_rate(&self) -> Vec<(usize, f64)> {
        let mut rates: Vec<(usize, f64)> = self
            .strata
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.failure_rate()))
            .collect();
        rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rates
    }

    /// Get frontier strata (those with both pass and fail verdicts).
    pub fn frontier_strata(&self) -> Vec<usize> {
        self.strata
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_frontier())
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute overall pass rate across all strata (volume-weighted).
    pub fn overall_pass_rate(&self) -> f64 {
        let total_volume: f64 = self.strata.iter().map(|s| s.volume()).sum();
        if total_volume <= 0.0 {
            return 0.0;
        }
        let weighted_pass: f64 = self
            .strata
            .iter()
            .map(|s| s.pass_rate() * s.volume())
            .sum();
        weighted_pass / total_volume
    }

    /// Total parameter-space volume.
    pub fn total_volume(&self) -> f64 {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product()
    }

    /// Number of strata with at least one sample.
    pub fn sampled_strata_count(&self) -> usize {
        self.strata.iter().filter(|s| s.sample_count > 0).count()
    }

    /// Get neighboring strata (differ by ±1 in exactly one dimension).
    pub fn neighbors(&self, stratum_idx: usize) -> Vec<usize> {
        if stratum_idx >= self.strata.len() {
            return Vec::new();
        }
        let k = self.strata_per_dim;
        let grid = self.flat_to_grid(stratum_idx);
        let mut neighbors = Vec::new();

        for d in 0..NUM_BODY_PARAMS {
            if grid[d] > 0 {
                let mut nb = grid;
                nb[d] -= 1;
                neighbors.push(self.grid_to_flat(&nb));
            }
            if grid[d] + 1 < k {
                let mut nb = grid;
                nb[d] += 1;
                neighbors.push(self.grid_to_flat(&nb));
            }
        }
        neighbors
    }

    /// Identify frontier strata: strata adjacent to both passing and failing strata.
    pub fn identify_classification_frontier(&self) -> Vec<usize> {
        let mut frontier = Vec::new();
        for (idx, stratum) in self.strata.iter().enumerate() {
            if stratum.sample_count == 0 {
                continue;
            }
            let nb_indices = self.neighbors(idx);
            let has_pass_neighbor = nb_indices.iter().any(|&ni| {
                ni < self.strata.len() && self.strata[ni].pass_count > 0
            });
            let has_fail_neighbor = nb_indices.iter().any(|&ni| {
                ni < self.strata.len() && self.strata[ni].fail_count > 0
            });
            if (stratum.is_frontier()) || (has_pass_neighbor && has_fail_neighbor) {
                frontier.push(idx);
            }
        }
        frontier
    }

    /// Summary statistics for the sampling state.
    pub fn summary(&self) -> SamplingSummary {
        let total_samples: usize = self.strata.iter().map(|s| s.sample_count).sum();
        let total_pass: usize = self.strata.iter().map(|s| s.pass_count).sum();
        let total_fail: usize = self.strata.iter().map(|s| s.fail_count).sum();
        let frontier_count = self.frontier_strata().len();
        let sampled_count = self.sampled_strata_count();

        SamplingSummary {
            total_strata: self.strata.len(),
            sampled_strata: sampled_count,
            frontier_strata: frontier_count,
            total_samples,
            total_pass,
            total_fail,
            overall_pass_rate: self.overall_pass_rate(),
        }
    }
}

/// Summary of sampling state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingSummary {
    pub total_strata: usize,
    pub sampled_strata: usize,
    pub frontier_strata: usize,
    pub total_samples: usize,
    pub total_pass: usize,
    pub total_fail: usize,
    pub overall_pass_rate: f64,
}

impl std::fmt::Display for SamplingSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Strata: {}/{} sampled ({} frontier) | Samples: {} ({} pass, {} fail) | Rate: {:.4}",
            self.sampled_strata,
            self.total_strata,
            self.frontier_strata,
            self.total_samples,
            self.total_pass,
            self.total_fail,
            self.overall_pass_rate,
        )
    }
}

// ───────────────────────── Sampling Config ─────────────────────────────────

/// Configuration for the sampling engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingStrategy {
    /// Method to use for initial sampling.
    pub method: SamplingMethod,
    /// Number of strata per dimension.
    pub strata_per_dim: usize,
    /// Initial samples per stratum.
    pub initial_samples_per_stratum: usize,
    /// Total adaptive resampling budget.
    pub adaptive_budget: usize,
    /// Random seed.
    pub seed: u64,
}

/// Sampling method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Uniform random within each stratum.
    Uniform,
    /// Latin Hypercube Sampling within each stratum.
    LatinHypercube,
    /// Halton quasi-random sequence within each stratum.
    Halton,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self {
            method: SamplingMethod::LatinHypercube,
            strata_per_dim: 3,
            initial_samples_per_stratum: 5,
            adaptive_budget: 500,
            seed: 42,
        }
    }
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bounds() -> ([f64; 5], [f64; 5]) {
        let lower = [1.50, 0.25, 0.35, 0.22, 0.16];
        let upper = [1.90, 0.40, 0.50, 0.33, 0.22];
        (lower, upper)
    }

    #[test]
    fn test_stratum_creation() {
        let s = Stratum::new(0, [0.0; 5], [1.0; 5], [0; 5]);
        assert_eq!(s.volume(), 1.0);
        assert_eq!(s.center(), [0.5; 5]);
        assert_eq!(s.failure_rate(), 0.0);
    }

    #[test]
    fn test_stratum_verdicts() {
        let mut s = Stratum::new(0, [0.0; 5], [1.0; 5], [0; 5]);
        s.record_verdict(true);
        s.record_verdict(true);
        s.record_verdict(false);
        assert_eq!(s.sample_count, 3);
        assert_eq!(s.pass_count, 2);
        assert_eq!(s.fail_count, 1);
        assert!((s.pass_rate() - 2.0 / 3.0).abs() < 1e-10);
        assert!(s.is_frontier());
    }

    #[test]
    fn test_stratum_contains() {
        let s = Stratum::new(0, [0.0; 5], [1.0; 5], [0; 5]);
        assert!(s.contains(&[0.5; 5]));
        assert!(!s.contains(&[1.5, 0.5, 0.5, 0.5, 0.5]));
    }

    #[test]
    fn test_create_strata_grid() {
        let (lower, upper) = default_bounds();
        let strata = StratifiedSampler::create_strata(&lower, &upper, 2);
        assert_eq!(strata.len(), 32); // 2^5 = 32
        let total_vol: f64 = strata.iter().map(|s| s.volume()).sum();
        let expected: f64 = (0..5).map(|i| upper[i] - lower[i]).product();
        assert!((total_vol - expected).abs() < 1e-10);
    }

    #[test]
    fn test_stratified_sampler_basic() {
        let (lower, upper) = default_bounds();
        let mut sampler = StratifiedSampler::new(lower, upper, 2, 42);
        assert_eq!(sampler.total_strata, 32);

        let samples = sampler.sample_stratum(0, 10);
        assert_eq!(samples.len(), 10);
        for s in &samples {
            assert!(s.stature >= lower[0] && s.stature <= upper[0]);
        }
    }

    #[test]
    fn test_find_stratum() {
        let (lower, upper) = default_bounds();
        let sampler = StratifiedSampler::new(lower, upper, 2, 42);
        let center = [1.6, 0.30, 0.40, 0.26, 0.18];
        let idx = sampler.find_stratum(&center);
        assert!(idx.is_some());
        let stratum = &sampler.strata[idx.unwrap()];
        assert!(stratum.contains(&center));
    }

    #[test]
    fn test_neighbors() {
        let (lower, upper) = default_bounds();
        let sampler = StratifiedSampler::new(lower, upper, 3, 42);
        let center_idx = sampler.grid_to_flat(&[1, 1, 1, 1, 1]);
        let nbs = sampler.neighbors(center_idx);
        assert_eq!(nbs.len(), 10); // 2 neighbors per dimension × 5 dims
    }

    #[test]
    fn test_adaptive_resampling() {
        let (lower, upper) = default_bounds();
        let mut sampler = StratifiedSampler::new(lower, upper, 2, 42);
        // Simulate some verdicts
        sampler.strata[0].record_verdict(true);
        sampler.strata[0].record_verdict(false);
        sampler.strata[1].record_verdict(true);
        sampler.strata[1].record_verdict(true);

        let resamples = sampler.adaptive_resampling(20);
        let total: usize = resamples.iter().map(|(_, v)| v.len()).sum();
        assert!(total <= 20);
    }

    #[test]
    fn test_uniform_sampling() {
        let (lower, upper) = default_bounds();
        let mut sampler = StratifiedSampler::new(lower, upper, 2, 42);
        let samples = sampler.sample_uniform(64);
        assert_eq!(samples.len(), 64);
    }

    #[test]
    fn test_halton_sequence() {
        let mut halton = HaltonSequence::new(5);
        let points = halton.generate_unit(100);
        assert_eq!(points.len(), 100);
        for p in &points {
            assert_eq!(p.len(), 5);
            for &v in p {
                assert!(v >= 0.0 && v < 1.0);
            }
        }
        // Halton points should be relatively well-distributed
        let mean: f64 = points.iter().map(|p| p[0]).sum::<f64>() / 100.0;
        assert!((mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_halton_scaled() {
        let lower = vec![1.5, 0.25, 0.35, 0.22, 0.16];
        let upper = vec![1.9, 0.40, 0.50, 0.33, 0.22];
        let mut halton = HaltonSequence::new(5);
        let points = halton.generate_scaled(50, &lower, &upper);
        for p in &points {
            for (i, &v) in p.iter().enumerate() {
                assert!(v >= lower[i] && v <= upper[i]);
            }
        }
    }

    #[test]
    fn test_latin_hypercube() {
        let (lower, upper) = default_bounds();
        let lhs = LatinHypercubeSampler::new(lower, upper, 42);
        let samples = lhs.generate(50);
        assert_eq!(samples.len(), 50);
        for s in &samples {
            for d in 0..5 {
                assert!(s[d] >= lower[d] && s[d] <= upper[d]);
            }
        }
    }

    #[test]
    fn test_lhs_body_params() {
        let (lower, upper) = default_bounds();
        let lhs = LatinHypercubeSampler::new(lower, upper, 42);
        let params = lhs.generate_body_params(20);
        assert_eq!(params.len(), 20);
        for p in &params {
            assert!(p.stature >= lower[0] && p.stature <= upper[0]);
        }
    }

    #[test]
    fn test_lhs_correlated() {
        let (lower, upper) = default_bounds();
        let lhs = LatinHypercubeSampler::new(lower, upper, 42);
        let corr = [
            [1.0, 0.8, 0.5, 0.7, 0.4],
            [0.8, 1.0, 0.3, 0.9, 0.5],
            [0.5, 0.3, 1.0, 0.4, 0.3],
            [0.7, 0.9, 0.4, 1.0, 0.6],
            [0.4, 0.5, 0.3, 0.6, 1.0],
        ];
        let samples = lhs.generate_correlated(50, &corr);
        assert_eq!(samples.len(), 50);
    }

    #[test]
    fn test_cholesky() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let l = cholesky_5x5(&identity).unwrap();
        for i in 0..5 {
            assert!((l[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sampling_summary() {
        let (lower, upper) = default_bounds();
        let mut sampler = StratifiedSampler::new(lower, upper, 2, 42);
        sampler.strata[0].record_verdict(true);
        sampler.strata[0].record_verdict(false);
        let summary = sampler.summary();
        assert_eq!(summary.total_samples, 2);
        assert_eq!(summary.total_pass, 1);
        assert_eq!(summary.total_fail, 1);
    }

    #[test]
    fn test_halton_per_stratum() {
        let (lower, upper) = default_bounds();
        let sampler = StratifiedSampler::new(lower, upper, 2, 42);
        let results = sampler.sample_halton_per_stratum(3);
        assert_eq!(results.len(), 32);
        for (_, samples) in &results {
            assert_eq!(samples.len(), 3);
        }
    }

    #[test]
    fn test_lhs_per_stratum() {
        let (lower, upper) = default_bounds();
        let sampler = StratifiedSampler::new(lower, upper, 2, 42);
        let results = sampler.sample_lhs_per_stratum(3);
        assert_eq!(results.len(), 32);
        for (_, samples) in &results {
            assert_eq!(samples.len(), 3);
        }
    }

    #[test]
    fn test_frontier_identification() {
        let (lower, upper) = default_bounds();
        let mut sampler = StratifiedSampler::new(lower, upper, 2, 42);
        sampler.strata[0].record_verdict(true);
        sampler.strata[0].record_verdict(false);
        let frontier = sampler.frontier_strata();
        assert!(frontier.contains(&0));
    }

    #[test]
    fn test_overall_pass_rate() {
        let (lower, upper) = default_bounds();
        let mut sampler = StratifiedSampler::new(lower, upper, 2, 42);
        for s in &mut sampler.strata {
            s.record_verdict(true);
        }
        assert!((sampler.overall_pass_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_index_roundtrip() {
        let (lower, upper) = default_bounds();
        let sampler = StratifiedSampler::new(lower, upper, 3, 42);
        for flat in 0..sampler.total_strata {
            let grid = sampler.flat_to_grid(flat);
            let back = sampler.grid_to_flat(&grid);
            assert_eq!(flat, back);
        }
    }
}
