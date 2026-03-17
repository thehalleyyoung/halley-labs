//! Stratified and adaptive population sampling.
//!
//! Provides various sampling strategies over the 5-dimensional body-parameter
//! space (stature, arm length, shoulder breadth, forearm length, hand length):
//! - Stratified random sampling with configurable strata
//! - Latin Hypercube Sampling (LHS)
//! - Simplified Sobol quasi-random sequences
//! - Adaptive sampling that concentrates effort near the accessibility frontier

use xr_types::kinematic::BodyParameterRange;
use xr_types::{BodyParameters, VerifierResult};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stratum definitions & statistics
// ---------------------------------------------------------------------------

/// A single stratum in the parameter space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumDefinition {
    /// Lower bounds in [0,1]^5 normalized parameter space.
    pub lo: [f64; 5],
    /// Upper bounds in [0,1]^5 normalized parameter space.
    pub hi: [f64; 5],
    /// Expected number of samples to draw from this stratum.
    pub expected_count: usize,
    /// Human-readable label (e.g. "small-stature / long-arm").
    pub label: String,
}

impl StratumDefinition {
    /// Normalized volume of this stratum in [0,1]^5.
    pub fn volume(&self) -> f64 {
        (0..5).map(|i| (self.hi[i] - self.lo[i]).max(0.0)).product()
    }

    /// Draw a uniform random sample inside this stratum.
    pub fn sample(&self, rng: &mut impl Rng) -> [f64; 5] {
        let mut s = [0.0; 5];
        for i in 0..5 {
            s[i] = rng.gen_range(self.lo[i]..=self.hi[i]);
        }
        s
    }
}

/// Verdict for a single sample's reachability test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleVerdict {
    Reachable,
    Unreachable,
    Error,
}

/// Statistics accumulated for one stratum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumStatistics {
    /// Stratum index.
    pub stratum_index: usize,
    /// Number of samples drawn.
    pub sample_count: usize,
    /// Number of samples that were reachable.
    pub reachable_count: usize,
    /// Number of samples that were unreachable.
    pub unreachable_count: usize,
    /// Number of samples that produced errors.
    pub error_count: usize,
    /// Estimated reachable fraction within the stratum.
    pub reachable_fraction: f64,
    /// Variance estimate of the reachable fraction.
    pub variance: f64,
}

impl StratumStatistics {
    pub fn new(stratum_index: usize) -> Self {
        Self {
            stratum_index,
            sample_count: 0,
            reachable_count: 0,
            unreachable_count: 0,
            error_count: 0,
            reachable_fraction: 0.0,
            variance: 0.0,
        }
    }

    /// Record one verdict and update running statistics.
    pub fn record(&mut self, verdict: SampleVerdict) {
        self.sample_count += 1;
        match verdict {
            SampleVerdict::Reachable => self.reachable_count += 1,
            SampleVerdict::Unreachable => self.unreachable_count += 1,
            SampleVerdict::Error => self.error_count += 1,
        }
        let n = self.sample_count as f64;
        self.reachable_fraction = self.reachable_count as f64 / n;
        let p = self.reachable_fraction;
        self.variance = if n > 1.0 { p * (1.0 - p) / (n - 1.0) } else { 0.25 };
    }
}

// ---------------------------------------------------------------------------
// PopulationSampler
// ---------------------------------------------------------------------------

/// Population sampler using stratified and Latin Hypercube sampling.
pub struct PopulationSampler {
    num_samples: usize,
    strata_per_dim: usize,
    seed: u64,
    range: BodyParameterRange,
}

impl PopulationSampler {
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            strata_per_dim: 3,
            seed: 42,
            range: BodyParameterRange::default(),
        }
    }

    pub fn with_strata(mut self, strata: usize) -> Self {
        self.strata_per_dim = strata.max(1);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_range(mut self, range: BodyParameterRange) -> Self {
        self.range = range;
        self
    }

    // -- Public sampling methods --

    /// Generate stratified samples spanning the population.
    pub fn sample(&self) -> Vec<BodyParameters> {
        self.sample_stratified(None)
    }

    /// Stratified random sampling: divide [0,1]^5 into a regular grid of
    /// strata, draw `expected_count` samples per stratum, then truncate /
    /// extend to `self.num_samples`.
    pub fn sample_stratified(&self, strata: Option<&[StratumDefinition]>) -> Vec<BodyParameters> {
        let auto_strata;
        let strata = match strata {
            Some(s) => s,
            None => {
                auto_strata = self.generate_strata();
                &auto_strata
            }
        };

        let mut rng = make_rng(self.seed);
        let mut samples: Vec<BodyParameters> = Vec::with_capacity(self.num_samples);

        for stratum in strata {
            for _ in 0..stratum.expected_count {
                let u = stratum.sample(&mut rng);
                samples.push(self.unit_to_params(&u));
                if samples.len() >= self.num_samples {
                    return samples;
                }
            }
        }

        // If strata didn't generate enough, pad with uniform random
        while samples.len() < self.num_samples {
            let u: [f64; 5] = std::array::from_fn(|_| rng.gen::<f64>());
            samples.push(self.unit_to_params(&u));
        }

        samples
    }

    /// Latin Hypercube Sampling: each dimension is divided into N equal
    /// strata and exactly one sample is placed in each stratum (randomly
    /// permuted across dimensions).
    pub fn sample_lhs(&self) -> Vec<BodyParameters> {
        let n = self.num_samples;
        let mut rng = make_rng(self.seed);

        // For each dimension generate a permutation of 0..n
        let mut perms: Vec<Vec<usize>> = (0..5)
            .map(|_| {
                let mut v: Vec<usize> = (0..n).collect();
                fisher_yates_shuffle(&mut v, &mut rng);
                v
            })
            .collect();

        (0..n)
            .map(|i| {
                let u: [f64; 5] = std::array::from_fn(|d| {
                    let stratum = perms[d][i];
                    (stratum as f64 + rng.gen::<f64>()) / n as f64
                });
                self.unit_to_params(&u)
            })
            .collect()
    }

    /// Simplified Sobol-like quasi-random sequence using the R2 sequence
    /// (generalised golden ratio low-discrepancy sequence in 5D).
    pub fn sample_sobol(&self) -> Vec<BodyParameters> {
        let n = self.num_samples;
        // Generalised golden ratio constants for d=5
        // α_d = the smallest positive root of x^(d+1) = x + 1
        let phi = compute_golden_ratio_d(5);
        let alphas: [f64; 5] = std::array::from_fn(|i| {
            let mut v = 1.0 / phi.powi(i as i32 + 1);
            v %= 1.0;
            v
        });

        let mut rng = make_rng(self.seed);
        let seed_offset: f64 = rng.gen();

        (0..n)
            .map(|i| {
                let u: [f64; 5] = std::array::from_fn(|d| {
                    (seed_offset + (i as f64 + 1.0) * alphas[d]) % 1.0
                });
                self.unit_to_params(&u)
            })
            .collect()
    }

    /// Adaptive sampling: start with a coarse stratified pass, identify
    /// strata near the accessibility frontier (mixed reachable / unreachable),
    /// then concentrate additional samples there.
    ///
    /// `test_fn` takes a `&BodyParameters` and returns a `SampleVerdict`.
    pub fn sample_adaptive<F>(&self, mut test_fn: F) -> (Vec<(BodyParameters, SampleVerdict)>, Vec<StratumStatistics>)
    where
        F: FnMut(&BodyParameters) -> SampleVerdict,
    {
        let strata = self.generate_strata();
        let mut stats: Vec<StratumStatistics> = (0..strata.len())
            .map(StratumStatistics::new)
            .collect();
        let mut results: Vec<(BodyParameters, SampleVerdict)> = Vec::new();
        let mut rng = make_rng(self.seed);

        // Phase 1: uniform allocation (half the budget)
        let phase1_budget = self.num_samples / 2;
        let per_stratum = (phase1_budget / strata.len()).max(1);
        for (si, stratum) in strata.iter().enumerate() {
            for _ in 0..per_stratum {
                let u = stratum.sample(&mut rng);
                let bp = self.unit_to_params(&u);
                let v = test_fn(&bp);
                stats[si].record(v);
                results.push((bp, v));
            }
        }

        // Phase 2: allocate remaining budget proportional to variance
        let phase2_budget = self.num_samples - results.len();
        let total_var: f64 = stats.iter().map(|s| s.variance).sum();
        if total_var < 1e-15 {
            // All strata are homogeneous; allocate uniformly
            let per = (phase2_budget / strata.len()).max(1);
            for (si, stratum) in strata.iter().enumerate() {
                for _ in 0..per {
                    if results.len() >= self.num_samples {
                        break;
                    }
                    let u = stratum.sample(&mut rng);
                    let bp = self.unit_to_params(&u);
                    let v = test_fn(&bp);
                    stats[si].record(v);
                    results.push((bp, v));
                }
            }
        } else {
            for (si, stratum) in strata.iter().enumerate() {
                let share = (stats[si].variance / total_var * phase2_budget as f64).round() as usize;
                for _ in 0..share {
                    if results.len() >= self.num_samples {
                        break;
                    }
                    let u = stratum.sample(&mut rng);
                    let bp = self.unit_to_params(&u);
                    let v = test_fn(&bp);
                    stats[si].record(v);
                    results.push((bp, v));
                }
            }
        }

        // Fill any remaining
        while results.len() < self.num_samples {
            let si = rng.gen_range(0..strata.len());
            let u = strata[si].sample(&mut rng);
            let bp = self.unit_to_params(&u);
            let v = test_fn(&bp);
            stats[si].record(v);
            results.push((bp, v));
        }

        (results, stats)
    }

    /// Generate percentile-based samples (simple linear interpolation).
    pub fn percentile_samples(&self, lo: f64, hi: f64) -> Vec<BodyParameters> {
        let small = self.range.min;
        let large = self.range.max;
        (0..self.num_samples)
            .map(|i| {
                let t = lo + (hi - lo) * (i as f64 / (self.num_samples as f64 - 1.0).max(1.0));
                small.lerp(&large, t)
            })
            .collect()
    }

    // -- Strata generation --

    /// Generate a regular grid of strata in [0,1]^5.
    pub fn generate_strata(&self) -> Vec<StratumDefinition> {
        let k = self.strata_per_dim;
        let total_cells = k.pow(5);
        let per_cell = (self.num_samples / total_cells).max(1);

        let mut strata = Vec::with_capacity(total_cells);
        let step = 1.0 / k as f64;

        // Iterate over a 5D grid using a single counter
        for idx in 0..total_cells {
            let mut lo = [0.0; 5];
            let mut hi = [0.0; 5];
            let mut rem = idx;
            for d in (0..5).rev() {
                let cell = rem % k;
                rem /= k;
                lo[d] = cell as f64 * step;
                hi[d] = lo[d] + step;
            }

            let label = format!(
                "stratum-{}-{}-{}-{}-{}",
                (lo[0] / step) as usize,
                (lo[1] / step) as usize,
                (lo[2] / step) as usize,
                (lo[3] / step) as usize,
                (lo[4] / step) as usize,
            );

            strata.push(StratumDefinition {
                lo,
                hi,
                expected_count: per_cell,
                label,
            });
        }

        strata
    }

    // -- Helpers --

    /// Map a point in [0,1]^5 to `BodyParameters` using the configured range.
    pub fn unit_to_params(&self, u: &[f64; 5]) -> BodyParameters {
        let lo = self.range.min.to_array();
        let hi = self.range.max.to_array();
        let arr: [f64; 5] = std::array::from_fn(|i| {
            lo[i] + u[i].clamp(0.0, 1.0) * (hi[i] - lo[i])
        });
        BodyParameters::from_array(&arr)
    }

    /// Map `BodyParameters` back to [0,1]^5 using the configured range.
    pub fn params_to_unit(&self, bp: &BodyParameters) -> [f64; 5] {
        bp.normalized(&self.range.min, &self.range.max)
    }
}

impl Default for PopulationSampler {
    fn default() -> Self {
        Self::new(10)
    }
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

/// Simple seeded RNG (we use rand::rngs::StdRng for reproducibility).
fn make_rng(seed: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed)
}

/// Fisher-Yates shuffle.
fn fisher_yates_shuffle<T>(v: &mut [T], rng: &mut impl Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}

/// Compute the generalised golden ratio for dimension d.
/// This is the unique positive root of x^(d+1) = x + 1, found via Newton's method.
fn compute_golden_ratio_d(d: usize) -> f64 {
    let n = d as f64 + 1.0;
    let mut x = 1.5; // initial guess
    for _ in 0..64 {
        let xn = x.powf(n);
        let f = xn - x - 1.0;
        let fp = n * x.powf(n - 1.0) - 1.0;
        if fp.abs() < 1e-15 {
            break;
        }
        x -= f / fp;
    }
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler() {
        let s = PopulationSampler::new(5);
        let samples = s.sample();
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_lhs_sample_count() {
        let s = PopulationSampler::new(20).with_seed(123);
        let samples = s.sample_lhs();
        assert_eq!(samples.len(), 20);
    }

    #[test]
    fn test_lhs_within_range() {
        let range = BodyParameterRange::default();
        let s = PopulationSampler::new(50).with_seed(7).with_range(range.clone());
        for bp in s.sample_lhs() {
            assert!(bp.stature >= range.min.stature && bp.stature <= range.max.stature);
            assert!(bp.arm_length >= range.min.arm_length && bp.arm_length <= range.max.arm_length);
        }
    }

    #[test]
    fn test_sobol_unique_and_in_range() {
        let range = BodyParameterRange::default();
        let s = PopulationSampler::new(30).with_range(range.clone());
        let samples = s.sample_sobol();
        assert_eq!(samples.len(), 30);
        for bp in &samples {
            assert!(bp.stature >= range.min.stature - 1e-9);
            assert!(bp.stature <= range.max.stature + 1e-9);
        }
        // Quasi-random: check that the first and last differ meaningfully
        assert!((samples[0].stature - samples[29].stature).abs() > 0.01);
    }

    #[test]
    fn test_generate_strata() {
        let s = PopulationSampler::new(100).with_strata(2);
        let strata = s.generate_strata();
        assert_eq!(strata.len(), 32); // 2^5
        // Each stratum's volume should be 1/32
        for st in &strata {
            assert!((st.volume() - 1.0 / 32.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_stratified_sample_count() {
        let s = PopulationSampler::new(50).with_strata(2);
        let samples = s.sample_stratified(None);
        assert_eq!(samples.len(), 50);
    }

    #[test]
    fn test_adaptive_converges() {
        let s = PopulationSampler::new(60).with_strata(2).with_seed(99);
        // Simple test function: reachable if stature > 1.7
        let (results, stats) = s.sample_adaptive(|bp: &BodyParameters| {
            if bp.stature > 1.7 { SampleVerdict::Reachable } else { SampleVerdict::Unreachable }
        });
        assert_eq!(results.len(), 60);
        let total_samples: usize = stats.iter().map(|s| s.sample_count).sum();
        assert_eq!(total_samples, 60);
        // There should be both reachable and unreachable
        let reachable = results.iter().filter(|(_, v)| *v == SampleVerdict::Reachable).count();
        assert!(reachable > 0 && reachable < 60);
    }

    #[test]
    fn test_unit_to_params_round_trip() {
        let s = PopulationSampler::new(10);
        let u = [0.5, 0.5, 0.5, 0.5, 0.5];
        let bp = s.unit_to_params(&u);
        let u2 = s.params_to_unit(&bp);
        for i in 0..5 {
            assert!((u[i] - u2[i]).abs() < 1e-6, "dim {} round-trip failed", i);
        }
    }

    #[test]
    fn test_unit_to_params_extremes() {
        let range = BodyParameterRange::default();
        let s = PopulationSampler::new(10).with_range(range.clone());
        let lo = s.unit_to_params(&[0.0; 5]);
        let hi = s.unit_to_params(&[1.0; 5]);
        assert!((lo.stature - range.min.stature).abs() < 1e-9);
        assert!((hi.stature - range.max.stature).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_samples() {
        let s = PopulationSampler::new(5);
        let samples = s.percentile_samples(0.25, 0.75);
        assert_eq!(samples.len(), 5);
        // First should be closer to small, last closer to large
        assert!(samples[0].stature < samples[4].stature);
    }

    #[test]
    fn test_golden_ratio_d() {
        let phi5 = compute_golden_ratio_d(5);
        // Should satisfy x^6 - x - 1 ≈ 0
        let residual = phi5.powi(6) - phi5 - 1.0;
        assert!(residual.abs() < 1e-10, "golden ratio residual = {}", residual);
    }

    #[test]
    fn test_stratum_statistics() {
        let mut st = StratumStatistics::new(0);
        st.record(SampleVerdict::Reachable);
        st.record(SampleVerdict::Reachable);
        st.record(SampleVerdict::Unreachable);
        assert_eq!(st.sample_count, 3);
        assert_eq!(st.reachable_count, 2);
        assert!((st.reachable_fraction - 2.0 / 3.0).abs() < 1e-9);
        assert!(st.variance > 0.0);
    }

    #[test]
    fn test_stratum_definition_volume() {
        let s = StratumDefinition {
            lo: [0.0; 5],
            hi: [0.5; 5],
            expected_count: 10,
            label: "test".to_string(),
        };
        assert!((s.volume() - 0.5f64.powi(5)).abs() < 1e-12);
    }
}
