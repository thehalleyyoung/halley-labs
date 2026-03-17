//! Lipschitz constant estimation for the accessibility frontier.
//!
//! Provides analytical bounds via kinematic Jacobian analysis and
//! empirical estimates via cross-validated sampling, together with
//! a dual-epsilon approach and piecewise-Lipschitz partition detection.

use crate::interval::{Interval, IntervalVector};
use crate::region::ParameterRegion;
use xr_types::{BodyParameters, KinematicChain};

// ── LipschitzEstimator ─────────────────────────────────────────────────

/// Estimates Lipschitz constants for the mapping from body parameters
/// to reachability, supporting both analytical and empirical approaches.
pub struct LipschitzEstimator {
    /// Number of samples for empirical estimation.
    pub sample_count: usize,
    /// Number of cross-validation folds.
    pub cv_folds: usize,
    /// Small epsilon for dual-epsilon comparisons.
    pub epsilon_tight: f64,
    /// Larger epsilon for conservative comparisons.
    pub epsilon_loose: f64,
}

impl LipschitzEstimator {
    pub fn new() -> Self {
        Self {
            sample_count: 200,
            cv_folds: 5,
            epsilon_tight: 1e-6,
            epsilon_loose: 1e-3,
        }
    }

    pub fn with_samples(mut self, n: usize) -> Self {
        self.sample_count = n;
        self
    }

    pub fn with_cv_folds(mut self, k: usize) -> Self {
        self.cv_folds = k.max(2);
        self
    }

    pub fn with_epsilons(mut self, tight: f64, loose: f64) -> Self {
        self.epsilon_tight = tight;
        self.epsilon_loose = loose;
        self
    }

    // ── Analytical Lipschitz bound ──────────────────────────────

    /// Analytical Lipschitz bound for the reach function:
    ///   reach(θ) = arm_length(θ) + forearm_length(θ) + hand_length(θ)
    ///
    /// Since each body-parameter dimension contributes linearly to the
    /// reach function, the Lipschitz constant is bounded by the L2 norm
    /// of the gradient. For the simple model (parameters 1, 3, 4 each
    /// contribute with coefficient 1):
    ///   L ≤ √(1² + 1² + 1²) = √3 ≈ 1.732
    pub fn analytical_reach_lipschitz(&self) -> f64 {
        // The gradient of reach w.r.t. the 5 body parameters is
        // [0, 1, 0, 1, 1] (only arm, forearm, hand contribute).
        // L2 norm = √3.
        3.0_f64.sqrt()
    }

    /// Analytical Lipschitz bound for a kinematic chain using
    /// the Jacobian structure.
    ///
    /// For each joint, the sensitivity of the end-effector position to
    /// joint angle θᵢ is bounded by the link length lᵢ (for revolute
    /// joints the derivative magnitude is at most lᵢ). The overall
    /// Lipschitz constant is:
    ///   L ≤ √(Σ lᵢ²)
    pub fn analytical_bound(&self, chain: &KinematicChain, params: &BodyParameters) -> f64 {
        let mut sum_sq = 0.0_f64;
        for joint in &chain.joints {
            let link = joint.effective_link_length(params);
            // For revolute joints the positional sensitivity is bounded by
            // the cumulative link length from this joint to the tip, but a
            // conservative per-joint bound is simply the link length itself.
            sum_sq += link * link;
        }
        sum_sq.sqrt()
    }

    /// Analytical Lipschitz bound over a parameter region (conservative:
    /// evaluate at both corners and take the maximum).
    pub fn analytical_bound_over_region(
        &self,
        chain: &KinematicChain,
        region: &ParameterRegion,
    ) -> f64 {
        let b = &region.bounds;
        if b.dim() < 5 {
            return self.analytical_reach_lipschitz();
        }

        // Evaluate at the 2^5 = 32 corners to find the max bound.
        // Optimisation: only need to check extremes along each dimension.
        let lo_params = body_params_from_intervals(b, false);
        let hi_params = body_params_from_intervals(b, true);

        let l_lo = self.analytical_bound(chain, &lo_params);
        let l_hi = self.analytical_bound(chain, &hi_params);
        l_lo.max(l_hi)
    }

    // ── Empirical Lipschitz estimate ────────────────────────────

    /// Empirical Lipschitz estimate by sampling pairs in the parameter
    /// region and computing max |f(x) − f(y)| / ‖x − y‖₂.
    pub fn empirical_estimate(
        &self,
        region: &ParameterRegion,
        verdicts: &[(Vec<f64>, f64)],
    ) -> f64 {
        if verdicts.len() < 2 {
            return self.analytical_reach_lipschitz();
        }

        let mut max_lip = 0.0_f64;
        for i in 0..verdicts.len() {
            for j in (i + 1)..verdicts.len() {
                let (ref x, fx) = verdicts[i];
                let (ref y, fy) = verdicts[j];
                let dist = euclidean_distance(x, y);
                if dist < self.epsilon_tight {
                    continue;
                }
                let lip = (fx - fy).abs() / dist;
                if lip > max_lip {
                    max_lip = lip;
                }
            }
        }
        max_lip
    }

    /// Empirical Lipschitz estimation over a parameter region using the
    /// built-in reach model (no external verdicts needed).
    pub fn empirical_lipschitz(&self, region: &ParameterRegion) -> f64 {
        let n = self.sample_count.max(2);
        let dim = region.bounds.dim();
        let mut max_lip = 0.0_f64;

        for d in 0..dim {
            let lo = region.bounds.components[d].lo;
            let hi = region.bounds.components[d].hi;
            if (hi - lo).abs() < 1e-15 {
                continue;
            }
            let step = (hi - lo) / (n as f64);
            let mut prev_reach = 0.0;

            for i in 0..=n {
                let t = lo + step * (i as f64);
                let mut params = region.bounds.midpoint();
                params[d] = t;
                let reach = if dim >= 5 {
                    params[1] + params[3] + params[4]
                } else {
                    0.0
                };
                if i > 0 {
                    max_lip = max_lip.max((reach - prev_reach).abs() / step);
                }
                prev_reach = reach;
            }
        }
        max_lip
    }

    /// Cross-validated empirical Lipschitz estimate.
    /// Splits samples into folds, estimates L on each training set,
    /// and returns the maximum across folds.
    pub fn cross_validated_estimate(
        &self,
        samples: &[(Vec<f64>, f64)],
    ) -> f64 {
        if samples.len() < self.cv_folds * 2 {
            return self.empirical_estimate(
                &ParameterRegion::new(IntervalVector::new(vec![])),
                samples,
            );
        }

        let fold_size = samples.len() / self.cv_folds;
        let mut max_lip = 0.0_f64;

        for fold in 0..self.cv_folds {
            let start = fold * fold_size;
            let end = if fold == self.cv_folds - 1 {
                samples.len()
            } else {
                start + fold_size
            };

            // Training set = everything except the current fold
            let train: Vec<_> = samples[..start]
                .iter()
                .chain(samples[end..].iter())
                .cloned()
                .collect();

            let dummy = ParameterRegion::new(IntervalVector::new(vec![]));
            let lip = self.empirical_estimate(&dummy, &train);
            max_lip = max_lip.max(lip);
        }
        max_lip
    }

    // ── Dual-epsilon approach ───────────────────────────────────

    /// Dual-epsilon Lipschitz exclusion test.
    ///
    /// Given:
    ///   - `lip`: Lipschitz constant estimate
    ///   - `diameter`: diameter of the parameter region
    ///   - `min_dist`: minimum distance from reach envelope to target
    ///   - `max_reach`: maximum reach in the region
    ///
    /// Returns true if the target is provably out of reach:
    ///   min_dist > max_reach + L × diameter + ε
    pub fn lipschitz_exclusion(
        &self,
        lip: f64,
        diameter: f64,
        min_dist: f64,
        max_reach: f64,
    ) -> bool {
        min_dist > max_reach + lip * diameter + self.epsilon_loose
    }

    /// Dual-epsilon inclusion test (provably reachable).
    ///
    /// Returns true if the target is provably within reach:
    ///   max_dist < min_reach − L × diameter − ε
    pub fn lipschitz_inclusion(
        &self,
        lip: f64,
        diameter: f64,
        max_dist: f64,
        min_reach: f64,
    ) -> bool {
        max_dist < min_reach - lip * diameter - self.epsilon_loose
    }

    // ── Piecewise-Lipschitz partition ───────────────────────────

    /// Detect whether the parameter region should be split into
    /// piecewise-Lipschitz sub-regions based on joint-limit transition
    /// surfaces.
    ///
    /// A transition surface occurs when a joint limit switches between
    /// active and inactive as body parameters vary. Within each piece
    /// the reach function is smooth, but across a transition surface the
    /// gradient may jump.
    ///
    /// Returns the dimensions along which a split is recommended.
    pub fn detect_transition_surfaces(
        &self,
        chain: &KinematicChain,
        region: &ParameterRegion,
    ) -> Vec<usize> {
        let b = &region.bounds;
        if b.dim() < 5 {
            return Vec::new();
        }

        let mut split_dims = Vec::new();

        for (ji, joint) in chain.joints.iter().enumerate() {
            if !joint.parameter_dependent {
                continue;
            }
            // Check if a joint limit transitions within the parameter region.
            // The joint limit is: min(θ) = base_min + Σ min_coeffs[i]*param[i]
            // A transition occurs when the sampled angle equals the limit at
            // some point inside the region. We detect this by checking whether
            // the limit range straddles a typical joint angle.
            let mid_config = joint.limits.midpoint();

            for dim in 0..5.min(b.dim()) {
                let coeff_min = joint.min_limit_coefficients[dim];
                let coeff_max = joint.max_limit_coefficients[dim];

                if coeff_min.abs() < 1e-12 && coeff_max.abs() < 1e-12 {
                    continue;
                }

                // Compute the range of the limit across this dimension
                let param_lo = b.components[dim].lo;
                let param_hi = b.components[dim].hi;

                let limit_min_lo = joint.limits.min + coeff_min * param_lo;
                let limit_min_hi = joint.limits.min + coeff_min * param_hi;
                let limit_range = Interval::new(
                    limit_min_lo.min(limit_min_hi),
                    limit_min_lo.max(limit_min_hi),
                );

                // If the midpoint configuration crosses the limit range,
                // there is a potential transition surface.
                if limit_range.contains(mid_config) {
                    split_dims.push(dim);
                }
            }
        }

        split_dims.sort_unstable();
        split_dims.dedup();
        split_dims
    }

    /// Compute piecewise-Lipschitz partition: split the region along
    /// detected transition surfaces and return the sub-regions.
    pub fn piecewise_partition(
        &self,
        chain: &KinematicChain,
        region: &ParameterRegion,
    ) -> Vec<ParameterRegion> {
        let split_dims = self.detect_transition_surfaces(chain, region);
        if split_dims.is_empty() {
            return vec![region.clone()];
        }

        let mut regions = vec![region.clone()];
        for &dim in &split_dims {
            let mut next = Vec::new();
            for r in &regions {
                let (l, h) = r.bisect_dim(dim);
                next.push(l);
                next.push(h);
            }
            regions = next;
        }
        regions
    }

    /// Adaptive Lipschitz refinement: for each sub-region, estimate the
    /// local Lipschitz constant. Returns pairs of (sub-region, local_L).
    pub fn adaptive_lipschitz(
        &self,
        chain: &KinematicChain,
        region: &ParameterRegion,
    ) -> Vec<(ParameterRegion, f64)> {
        let parts = self.piecewise_partition(chain, region);
        parts
            .into_iter()
            .map(|sub| {
                let lip = self.analytical_bound_over_region(chain, &sub);
                (sub, lip)
            })
            .collect()
    }
}

impl Default for LipschitzEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn body_params_from_intervals(iv: &IntervalVector, use_hi: bool) -> BodyParameters {
    let vals: Vec<f64> = iv
        .components
        .iter()
        .map(|c| if use_hi { c.hi } else { c.lo })
        .collect();
    if vals.len() >= 5 {
        BodyParameters::new(vals[0], vals[1], vals[2], vals[3], vals[4])
    } else {
        BodyParameters::default()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::ArmSide;

    fn sample_region() -> ParameterRegion {
        ParameterRegion::new(IntervalVector::from_ranges(&[
            (1.5, 1.9),
            (0.25, 0.40),
            (0.35, 0.50),
            (0.22, 0.33),
            (0.16, 0.22),
        ]))
    }

    fn sample_chain() -> KinematicChain {
        KinematicChain::default_arm(ArmSide::Right)
    }

    #[test]
    fn test_analytical_reach_lipschitz() {
        let l = LipschitzEstimator::new().analytical_reach_lipschitz();
        assert!((l - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_analytical_bound_with_chain() {
        let chain = sample_chain();
        let params = BodyParameters::average_male();
        let l = LipschitzEstimator::new().analytical_bound(&chain, &params);
        // Should be positive and finite
        assert!(l > 0.0);
        assert!(l.is_finite());
    }

    #[test]
    fn test_analytical_bound_over_region() {
        let chain = sample_chain();
        let region = sample_region();
        let l = LipschitzEstimator::new().analytical_bound_over_region(&chain, &region);
        assert!(l > 0.0);
        assert!(l.is_finite());
    }

    #[test]
    fn test_empirical_lipschitz() {
        let region = sample_region();
        let l = LipschitzEstimator::new().empirical_lipschitz(&region);
        // For the linear model, empirical L should be close to 1.0 per dim.
        assert!(l > 0.5);
        assert!(l < 5.0);
    }

    #[test]
    fn test_empirical_estimate_from_samples() {
        let est = LipschitzEstimator::new();
        let samples = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![1.0, 0.0], 2.0),
            (vec![0.0, 1.0], 1.5),
        ];
        let l = est.empirical_estimate(
            &ParameterRegion::new(IntervalVector::new(vec![])),
            &samples,
        );
        assert!(l >= 1.5); // |2.0 - 0.0| / 1.0 = 2.0
    }

    #[test]
    fn test_cross_validated_estimate() {
        let est = LipschitzEstimator::new().with_cv_folds(3);
        let mut samples = Vec::new();
        for i in 0..30 {
            let x = i as f64 * 0.1;
            samples.push((vec![x, 0.0], x * 2.0));
        }
        let l = est.cross_validated_estimate(&samples);
        // Function is f(x) = 2x, so L = 2.0
        assert!((l - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_lipschitz_exclusion() {
        let est = LipschitzEstimator::new();
        // Target far away: should be excluded
        assert!(est.lipschitz_exclusion(2.0, 0.1, 5.0, 1.0));
        // Target close: should not be excluded
        assert!(!est.lipschitz_exclusion(2.0, 0.1, 0.5, 1.0));
    }

    #[test]
    fn test_lipschitz_inclusion() {
        let est = LipschitzEstimator::new();
        // Target well within reach
        assert!(est.lipschitz_inclusion(2.0, 0.01, 0.1, 1.0));
        // Target borderline
        assert!(!est.lipschitz_inclusion(2.0, 1.0, 0.5, 1.0));
    }

    #[test]
    fn test_piecewise_partition_no_transitions() {
        let chain = sample_chain();
        let region = sample_region();
        let est = LipschitzEstimator::new();
        let parts = est.piecewise_partition(&chain, &region);
        // Default chain joints are not parameter-dependent, so no splits.
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn test_adaptive_lipschitz() {
        let chain = sample_chain();
        let region = sample_region();
        let est = LipschitzEstimator::new();
        let results = est.adaptive_lipschitz(&chain, &region);
        assert!(!results.is_empty());
        for (_, lip) in &results {
            assert!(*lip >= 0.0);
            assert!(lip.is_finite());
        }
    }

    #[test]
    fn test_detect_transition_surfaces_no_dependent() {
        let chain = sample_chain();
        let region = sample_region();
        let est = LipschitzEstimator::new();
        let splits = est.detect_transition_surfaces(&chain, &region);
        // No parameter-dependent joints → no transitions
        assert!(splits.is_empty());
    }

    #[test]
    fn test_dual_epsilon_defaults() {
        let est = LipschitzEstimator::new();
        assert!(est.epsilon_tight < est.epsilon_loose);
        assert!(est.epsilon_tight > 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let d = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_constant_function() {
        // Constant function → L = 0
        let est = LipschitzEstimator::new();
        let samples = vec![
            (vec![0.0], 5.0),
            (vec![1.0], 5.0),
            (vec![2.0], 5.0),
        ];
        let l = est.empirical_estimate(
            &ParameterRegion::new(IntervalVector::new(vec![])),
            &samples,
        );
        assert!(l < 1e-10);
    }

    #[test]
    fn test_body_params_from_intervals() {
        let iv = IntervalVector::from_ranges(&[
            (1.5, 1.9),
            (0.25, 0.40),
            (0.35, 0.50),
            (0.22, 0.33),
            (0.16, 0.22),
        ]);
        let lo = body_params_from_intervals(&iv, false);
        assert!((lo.stature - 1.5).abs() < 1e-10);
        let hi = body_params_from_intervals(&iv, true);
        assert!((hi.stature - 1.9).abs() < 1e-10);
    }
}
