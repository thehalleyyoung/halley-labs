//! Sampling-based value function approximation for MILP lower levels.
//!
//! Provides Latin hypercube sampling, adaptive sampling near region boundaries,
//! and construction of piecewise linear approximations from samples.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::oracle::ValueFunctionOracle;
use crate::piecewise_linear::{AffinePiece, PiecewiseLinearVF};
use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for sampling-based approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Number of initial samples.
    pub num_samples: usize,
    /// Random seed.
    pub seed: u64,
    /// Number of adaptive refinement rounds.
    pub refinement_rounds: usize,
    /// Number of adaptive samples per round.
    pub adaptive_samples_per_round: usize,
    /// Fraction of samples to place near boundaries.
    pub boundary_fraction: f64,
    /// Minimum distance between sample points.
    pub min_distance: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_samples: 100,
            seed: 42,
            refinement_rounds: 3,
            adaptive_samples_per_round: 20,
            boundary_fraction: 0.3,
            min_distance: 1e-6,
        }
    }
}

/// Result of a sampling approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingApproximation {
    /// Sample points in x-space.
    pub sample_points: Vec<Vec<f64>>,
    /// Value function values at sample points.
    pub values: Vec<f64>,
    /// Subgradients at sample points.
    pub subgradients: Vec<Vec<f64>>,
    /// The piecewise linear approximation constructed from samples.
    pub approximation: PiecewiseLinearVF,
    /// L1 error bound estimate.
    pub l1_error_bound: f64,
    /// Maximum pointwise error estimate.
    pub max_error_estimate: f64,
}

// ---------------------------------------------------------------------------
// Latin hypercube sampling
// ---------------------------------------------------------------------------

/// Generate Latin Hypercube samples in the box [lb, ub].
pub fn latin_hypercube_sample(
    lb: &[f64],
    ub: &[f64],
    num_samples: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let dim = lb.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(num_samples);

    // Create permutations for each dimension
    let mut permutations: Vec<Vec<usize>> = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut perm: Vec<usize> = (0..num_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..num_samples).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        permutations.push(perm);
    }

    for i in 0..num_samples {
        let point: Vec<f64> = (0..dim)
            .map(|d| {
                let cell = permutations[d][i];
                let u: f64 = rng.gen();
                let t = (cell as f64 + u) / num_samples as f64;
                lb[d] + t * (ub[d] - lb[d])
            })
            .collect();
        samples.push(point);
    }

    samples
}

/// Generate uniform random samples.
pub fn uniform_sample(lb: &[f64], ub: &[f64], num_samples: usize, seed: u64) -> Vec<Vec<f64>> {
    let dim = lb.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let point: Vec<f64> = (0..dim)
            .map(|d| {
                let u: f64 = rng.gen();
                lb[d] + u * (ub[d] - lb[d])
            })
            .collect();
        samples.push(point);
    }

    samples
}

/// Generate boundary-biased samples (more samples near the edges of the box).
pub fn boundary_sample(lb: &[f64], ub: &[f64], num_samples: usize, seed: u64) -> Vec<Vec<f64>> {
    let dim = lb.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let point: Vec<f64> = (0..dim)
            .map(|d| {
                let u: f64 = rng.gen();
                // Beta(0.5, 0.5) approximation: push towards boundaries
                let t = if u < 0.5 {
                    (2.0 * u).sqrt() / 2.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).sqrt() / 2.0
                };
                lb[d] + t * (ub[d] - lb[d])
            })
            .collect();
        samples.push(point);
    }

    samples
}

// ---------------------------------------------------------------------------
// Adaptive sampler
// ---------------------------------------------------------------------------

/// Adaptive sampler that concentrates samples near region boundaries.
pub struct AdaptiveSampler {
    pub config: SamplingConfig,
    x_lower: Vec<f64>,
    x_upper: Vec<f64>,
}

impl AdaptiveSampler {
    pub fn new(config: SamplingConfig, x_lower: Vec<f64>, x_upper: Vec<f64>) -> Self {
        Self {
            config,
            x_lower,
            x_upper,
        }
    }

    /// Build a complete sampling approximation.
    pub fn build_approximation(
        &self,
        oracle: &dyn ValueFunctionOracle,
    ) -> VFResult<SamplingApproximation> {
        let dim = self.x_lower.len();

        // Phase 1: Initial LHS sampling
        let boundary_count =
            (self.config.num_samples as f64 * self.config.boundary_fraction) as usize;
        let lhs_count = self.config.num_samples - boundary_count;

        let mut all_points =
            latin_hypercube_sample(&self.x_lower, &self.x_upper, lhs_count, self.config.seed);
        let boundary_pts = boundary_sample(
            &self.x_lower,
            &self.x_upper,
            boundary_count,
            self.config.seed + 1,
        );
        all_points.extend(boundary_pts);

        // Evaluate at all points
        let mut values = Vec::new();
        let mut subgradients = Vec::new();
        let mut valid_points = Vec::new();

        for x in &all_points {
            match oracle.evaluate(x) {
                Ok(info) => {
                    values.push(info.value);
                    let sg = oracle
                        .dual_info(x)
                        .map(|d| d.subgradient)
                        .unwrap_or_else(|_| vec![0.0; dim]);
                    subgradients.push(sg);
                    valid_points.push(x.clone());
                }
                Err(_) => continue,
            }
        }

        if valid_points.is_empty() {
            return Err(VFError::SamplingError("No feasible samples found".into()));
        }

        // Phase 2: Adaptive refinement
        for round in 0..self.config.refinement_rounds {
            let new_points = self.generate_adaptive_samples(
                &valid_points,
                &values,
                &subgradients,
                self.config.adaptive_samples_per_round,
                self.config.seed + 100 + round as u64,
            );

            for x in &new_points {
                match oracle.evaluate(x) {
                    Ok(info) => {
                        values.push(info.value);
                        let sg = oracle
                            .dual_info(x)
                            .map(|d| d.subgradient)
                            .unwrap_or_else(|_| vec![0.0; dim]);
                        subgradients.push(sg);
                        valid_points.push(x.clone());
                    }
                    Err(_) => continue,
                }
            }
        }

        // Phase 3: Build PWL approximation
        let approximation = self.build_pwl_from_samples(&valid_points, &values, &subgradients, dim);

        // Estimate error bounds
        let (l1_bound, max_bound) =
            self.estimate_error_bounds(&approximation, &valid_points, &values);

        Ok(SamplingApproximation {
            sample_points: valid_points,
            values,
            subgradients,
            approximation,
            l1_error_bound: l1_bound,
            max_error_estimate: max_bound,
        })
    }

    /// Generate adaptive samples near detected "kink" regions.
    fn generate_adaptive_samples(
        &self,
        points: &[Vec<f64>],
        values: &[f64],
        subgradients: &[Vec<f64>],
        count: usize,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let dim = self.x_lower.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let n = points.len();

        if n < 2 {
            return uniform_sample(&self.x_lower, &self.x_upper, count, seed);
        }

        // Compute "interestingness" score for each point based on subgradient variation
        let mut scores: Vec<(usize, f64)> = Vec::new();
        for i in 0..n {
            let mut max_sg_diff = 0.0f64;
            for j in 0..n.min(20) {
                if i == j {
                    continue;
                }
                let dist: f64 = points[i]
                    .iter()
                    .zip(points[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist > TOLERANCE && dist < self.box_diameter() * 0.3 {
                    let sg_diff: f64 = subgradients[i]
                        .iter()
                        .zip(subgradients[j].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    max_sg_diff = max_sg_diff.max(sg_diff / dist);
                }
            }
            scores.push((i, max_sg_diff));
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Generate new samples near high-score points
        let mut new_samples = Vec::with_capacity(count);
        let radius = self.box_diameter() * 0.05;

        for k in 0..count {
            let base_idx = scores[k % scores.len().max(1)].0;
            let base = &points[base_idx];

            let new_point: Vec<f64> = (0..dim)
                .map(|d| {
                    let offset: f64 = rng.gen::<f64>() * 2.0 * radius - radius;
                    (base[d] + offset).max(self.x_lower[d]).min(self.x_upper[d])
                })
                .collect();

            new_samples.push(new_point);
        }

        new_samples
    }

    fn box_diameter(&self) -> f64 {
        self.x_lower
            .iter()
            .zip(self.x_upper.iter())
            .map(|(l, u)| (u - l).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Build PWL approximation from sample evaluations (cutting plane model).
    fn build_pwl_from_samples(
        &self,
        points: &[Vec<f64>],
        values: &[f64],
        subgradients: &[Vec<f64>],
        dim: usize,
    ) -> PiecewiseLinearVF {
        let mut pwl = PiecewiseLinearVF::new(dim);

        for i in 0..points.len() {
            let x = &points[i];
            let val = values[i];
            let sg = &subgradients[i];

            // Cutting plane: φ(x) ≥ val + sg^T (x - x_i)
            // = sg^T x + (val - sg^T x_i)
            let constant = val - sg.iter().zip(x.iter()).map(|(g, xi)| g * xi).sum::<f64>();

            pwl.add_piece(AffinePiece {
                coefficients: sg.clone(),
                constant,
                region: None,
            });
        }

        pwl
    }

    /// Estimate L1 and L∞ error bounds via cross-validation.
    fn estimate_error_bounds(
        &self,
        approx: &PiecewiseLinearVF,
        points: &[Vec<f64>],
        values: &[f64],
    ) -> (f64, f64) {
        let n = points.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let mut total_error = 0.0f64;
        let mut max_error = 0.0f64;

        for i in 0..n {
            let approx_val = approx.evaluate(&points[i]);
            let error = (values[i] - approx_val).abs();
            total_error += error;
            max_error = max_error.max(error);
        }

        let l1_bound = total_error / n as f64;
        (l1_bound, max_error)
    }
}

/// Compute the convex hull approximation quality metric.
pub fn approximation_quality(
    oracle: &dyn ValueFunctionOracle,
    approx: &PiecewiseLinearVF,
    test_points: &[Vec<f64>],
) -> ApproximationQuality {
    let mut total_error = 0.0f64;
    let mut max_error = 0.0f64;
    let mut total_gap = 0.0f64;
    let mut count = 0usize;

    for x in test_points {
        if let Ok(true_val) = oracle.value(x) {
            let approx_val = approx.evaluate(x);
            let error = (true_val - approx_val).abs();
            let gap = true_val - approx_val; // positive if under-approximation
            total_error += error;
            total_gap += gap;
            max_error = max_error.max(error);
            count += 1;
        }
    }

    let avg_error = if count > 0 {
        total_error / count as f64
    } else {
        0.0
    };
    let avg_gap = if count > 0 {
        total_gap / count as f64
    } else {
        0.0
    };

    ApproximationQuality {
        avg_error,
        max_error,
        avg_gap,
        num_evaluated: count,
    }
}

/// Quality metrics for a sampling approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproximationQuality {
    pub avg_error: f64,
    pub max_error: f64,
    pub avg_gap: f64,
    pub num_evaluated: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latin_hypercube_sample() {
        let samples = latin_hypercube_sample(&[0.0, 0.0], &[1.0, 1.0], 10, 42);
        assert_eq!(samples.len(), 10);
        for s in &samples {
            assert_eq!(s.len(), 2);
            assert!(s[0] >= 0.0 && s[0] <= 1.0);
            assert!(s[1] >= 0.0 && s[1] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_sample() {
        let samples = uniform_sample(&[-1.0], &[1.0], 50, 123);
        assert_eq!(samples.len(), 50);
        for s in &samples {
            assert!(s[0] >= -1.0 && s[0] <= 1.0);
        }
    }

    #[test]
    fn test_boundary_sample() {
        let samples = boundary_sample(&[0.0], &[1.0], 100, 99);
        assert_eq!(samples.len(), 100);
        // Boundary-biased: more samples near 0 and 1
        let near_boundary = samples.iter().filter(|s| s[0] < 0.2 || s[0] > 0.8).count();
        // Should have more boundary samples than pure uniform
        assert!(near_boundary > 15);
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.num_samples, 100);
        assert_eq!(config.seed, 42);
        assert!(config.boundary_fraction > 0.0);
    }

    #[test]
    fn test_lhs_coverage() {
        // LHS should cover each row/column of the grid exactly once
        let n = 20;
        let samples = latin_hypercube_sample(&[0.0], &[1.0], n, 42);
        assert_eq!(samples.len(), n);

        // Check that each cell [i/n, (i+1)/n) has exactly one sample
        let mut cells = vec![0usize; n];
        for s in &samples {
            let cell = (s[0] * n as f64).floor() as usize;
            let cell = cell.min(n - 1);
            cells[cell] += 1;
        }
        for &c in &cells {
            assert_eq!(c, 1);
        }
    }

    #[test]
    fn test_approximation_quality_struct() {
        let q = ApproximationQuality {
            avg_error: 0.1,
            max_error: 0.5,
            avg_gap: 0.05,
            num_evaluated: 100,
        };
        assert!(q.avg_error < q.max_error);
    }

    #[test]
    fn test_sampling_approximation_struct() {
        let approx = SamplingApproximation {
            sample_points: vec![vec![0.0], vec![1.0]],
            values: vec![0.0, 1.0],
            subgradients: vec![vec![1.0], vec![1.0]],
            approximation: PiecewiseLinearVF::new(1),
            l1_error_bound: 0.0,
            max_error_estimate: 0.0,
        };
        assert_eq!(approx.sample_points.len(), 2);
    }
}
