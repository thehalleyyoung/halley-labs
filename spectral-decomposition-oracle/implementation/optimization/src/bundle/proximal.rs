//! Proximal bundle method for minimising a convex nonsmooth function.
//!
//! Given an oracle that, for any point *x*, returns a function value f(x) and a
//! subgradient g ∈ ∂f(x), this solver builds a piecewise-linear cutting-plane
//! model of f and uses a quadratic proximity term to stabilise the iterates.

use log::{debug, info, warn};
use std::time::Instant;

use crate::error::{OptError, OptResult};
use crate::bundle::{BundleConfig, BundleResult, IterationInfo, StepType, SubgradientInfo};

// ---------------------------------------------------------------------------
// Cutting plane
// ---------------------------------------------------------------------------

/// A single cutting plane l(x) = g^T x + c, derived from a linearisation of f.
#[derive(Debug, Clone)]
pub struct CuttingPlane {
    /// Subgradient (slope) of this plane.
    pub subgradient: Vec<f64>,
    /// Constant term (= f(y) - g^T y where y is the evaluation point).
    pub constant: f64,
    /// Number of iterations since this plane was last active.
    pub age: usize,
    /// Whether this plane was binding at the last master solution.
    pub active: bool,
}

impl CuttingPlane {
    /// Evaluate the plane at `x`: g^T x + c.
    #[inline]
    fn evaluate(&self, x: &[f64]) -> f64 {
        let mut val = self.constant;
        for (gi, xi) in self.subgradient.iter().zip(x.iter()) {
            val += gi * xi;
        }
        val
    }
}

// ---------------------------------------------------------------------------
// BundleMethod
// ---------------------------------------------------------------------------

/// Proximal bundle solver.
pub struct BundleMethod {
    config: BundleConfig,
    bundle: Vec<CuttingPlane>,
    stability_center: Vec<f64>,
    best_value: f64,
    mu: f64,
    dimension: usize,
}

impl BundleMethod {
    /// Create a new bundle solver for an `n`-dimensional problem.
    pub fn new(config: BundleConfig, dimension: usize) -> Self {
        let mu = config.initial_mu;
        Self {
            config,
            bundle: Vec::new(),
            stability_center: vec![0.0; dimension],
            best_value: f64::INFINITY,
            mu,
            dimension,
        }
    }

    /// Set an initial stability centre and value (warm-start).
    pub fn warm_start(&mut self, point: Vec<f64>, value: f64) {
        assert_eq!(point.len(), self.dimension);
        self.stability_center = point;
        self.best_value = value;
    }

    /// Run the proximal bundle method.
    ///
    /// `oracle` is called at each trial point and must return a [`SubgradientInfo`].
    pub fn solve<F>(&mut self, mut oracle: F) -> OptResult<BundleResult>
    where
        F: FnMut(&[f64]) -> OptResult<SubgradientInfo>,
    {
        let start = Instant::now();
        let mut history = Vec::new();

        // Evaluate at the initial stability centre.
        if self.bundle.is_empty() {
            let info = oracle(&self.stability_center)?;
            self.best_value = info.value;
            self.add_cutting_plane(&info);
        }

        let mut best_bound = f64::NEG_INFINITY;

        for iter in 0..self.config.max_iterations {
            // Check time limit.
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > self.config.time_limit {
                return Err(OptError::TimeLimitExceeded {
                    elapsed,
                    limit: self.config.time_limit,
                });
            }

            // 1) Solve QP master problem.
            let (trial_point, model_value) = self.solve_qp_master()?;

            // Update best lower bound from the model value at the trial point.
            if model_value > best_bound {
                best_bound = model_value;
            }

            // 2) Evaluate oracle at trial point.
            let info = oracle(&trial_point)?;
            let trial_value = info.value;

            // 3) Serious or null step decision.
            let step_type = self.serious_or_null_step(&trial_point, trial_value, model_value);

            // 4) Update proximity parameter.
            self.update_proximity(step_type);

            // 5) Add a cutting plane from the new evaluation.
            self.add_cutting_plane(&info);

            // 6) Manage bundle size.
            if self.bundle.len() > self.config.bundle_capacity {
                self.aggregate_bundle();
            }

            // Compute gap.
            let gap = if self.best_value.abs() > 1e-10 {
                (self.best_value - best_bound).abs() / self.best_value.abs()
            } else {
                (self.best_value - best_bound).abs()
            };

            let iter_info = IterationInfo {
                iteration: iter,
                objective: self.best_value,
                best_bound,
                gap,
                step_type,
                mu: self.mu,
            };
            history.push(iter_info);

            if self.config.verbose {
                info!(
                    "Bundle iter {}: obj={:.6}, bound={:.6}, gap={:.6e}, step={}, mu={:.4e}",
                    iter, self.best_value, best_bound, gap, step_type, self.mu
                );
            }

            // Convergence check.
            if gap < self.config.gap_tolerance {
                debug!("Bundle method converged at iteration {} with gap {:.2e}", iter, gap);
                return Ok(BundleResult {
                    optimal_point: self.stability_center.clone(),
                    optimal_value: self.best_value,
                    iterations: iter + 1,
                    gap,
                    converged: true,
                    history,
                });
            }
        }

        let gap = if self.best_value.abs() > 1e-10 {
            (self.best_value - best_bound).abs() / self.best_value.abs()
        } else {
            (self.best_value - best_bound).abs()
        };

        warn!(
            "Bundle method did not converge after {} iterations (gap={:.2e})",
            self.config.max_iterations, gap
        );
        Ok(BundleResult {
            optimal_point: self.stability_center.clone(),
            optimal_value: self.best_value,
            iterations: self.config.max_iterations,
            gap,
            converged: false,
            history,
        })
    }

    // -----------------------------------------------------------------------
    // QP master problem
    // -----------------------------------------------------------------------

    /// Solve the QP master:
    ///
    ///   min_x  { max_i [ g_i^T x + c_i ] + (μ/2) ||x − x̂||² }
    ///
    /// We reformulate as:
    ///
    ///   min_{x,t}  t + (μ/2) ||x − x̂||²
    ///   s.t.       t ≥ g_i^T x + c_i   ∀i
    ///
    /// Returns (trial point, model value at trial point).
    fn solve_qp_master(&self) -> OptResult<(Vec<f64>, f64)> {
        let n = self.dimension;
        let num_planes = self.bundle.len();

        if num_planes == 0 {
            return Ok((self.stability_center.clone(), f64::NEG_INFINITY));
        }

        // We solve this via projected gradient descent on the dual of the QP.
        //
        // Dual: max_{α≥0, Σα_i=1}  Σ α_i c_i  −  (1/(2μ)) || Σ α_i g_i ||²
        //                            + (Σ α_i g_i)^T x̂
        //
        // where the primal recovery is: x* = x̂ − (1/μ) Σ α_i g_i
        //
        // This is a simplex-constrained concave QP in α.

        let mut alpha = vec![0.0; num_planes];
        // Initialise uniformly.
        let init_val = 1.0 / num_planes as f64;
        for a in alpha.iter_mut() {
            *a = init_val;
        }

        let max_inner_iter = 500.max(num_planes * 10);
        let tol = 1e-10;

        for _inner in 0..max_inner_iter {
            // Compute aggregate subgradient: g_agg = Σ α_i g_i
            let mut g_agg = vec![0.0; n];
            for (i, plane) in self.bundle.iter().enumerate() {
                if alpha[i] > 0.0 {
                    for (j, &gj) in plane.subgradient.iter().enumerate() {
                        g_agg[j] += alpha[i] * gj;
                    }
                }
            }

            // Compute gradient w.r.t. α_i:
            //   dL/dα_i = c_i + g_i^T x̂  −  (1/μ) g_i^T g_agg
            let mut grad = vec![0.0; num_planes];
            for (i, plane) in self.bundle.iter().enumerate() {
                let mut gi_dot_xhat = 0.0;
                let mut gi_dot_gagg = 0.0;
                for j in 0..n {
                    gi_dot_xhat += plane.subgradient[j] * self.stability_center[j];
                    gi_dot_gagg += plane.subgradient[j] * g_agg[j];
                }
                grad[i] = plane.constant + gi_dot_xhat - gi_dot_gagg / self.mu;
            }

            // Projected gradient step on the simplex.
            // Step size: use 1/L where L ≈ max(||g_i||²)/μ
            let mut l_est = 1.0;
            for plane in &self.bundle {
                let norm_sq: f64 = plane.subgradient.iter().map(|v| v * v).sum();
                if norm_sq > l_est {
                    l_est = norm_sq;
                }
            }
            l_est /= self.mu;
            if l_est < 1e-12 {
                l_est = 1.0;
            }
            let step = 1.0 / l_est;

            let mut alpha_new = vec![0.0; num_planes];
            for i in 0..num_planes {
                alpha_new[i] = alpha[i] + step * grad[i];
            }

            // Project onto simplex.
            project_onto_simplex(&mut alpha_new);

            // Check convergence.
            let mut diff_sq = 0.0;
            for i in 0..num_planes {
                let d = alpha_new[i] - alpha[i];
                diff_sq += d * d;
            }
            alpha = alpha_new;

            if diff_sq < tol * tol {
                break;
            }
        }

        // Recover primal: x* = x̂ − (1/μ) g_agg
        let mut g_agg = vec![0.0; n];
        for (i, plane) in self.bundle.iter().enumerate() {
            if alpha[i] > 0.0 {
                for (j, &gj) in plane.subgradient.iter().enumerate() {
                    g_agg[j] += alpha[i] * gj;
                }
            }
        }

        let mut trial = vec![0.0; n];
        for j in 0..n {
            trial[j] = self.stability_center[j] - g_agg[j] / self.mu;
        }

        // Model value at trial: max_i { g_i^T trial + c_i }
        let model_value = self
            .bundle
            .iter()
            .map(|p| p.evaluate(&trial))
            .fold(f64::NEG_INFINITY, f64::max);

        Ok((trial, model_value))
    }

    // -----------------------------------------------------------------------
    // Cutting-plane management
    // -----------------------------------------------------------------------

    /// Add a cutting plane derived from the oracle evaluation `info`.
    ///
    /// l(x) = f(y) + g^T(x − y) = g^T x + (f(y) − g^T y)
    fn add_cutting_plane(&mut self, info: &SubgradientInfo) {
        let constant = info.value - dot(&info.subgradient, &info.point);
        // Age existing planes.
        for plane in &mut self.bundle {
            plane.age += 1;
        }
        self.bundle.push(CuttingPlane {
            subgradient: info.subgradient.clone(),
            constant,
            age: 0,
            active: true,
        });
    }

    /// When the bundle exceeds capacity, aggregate the oldest cuts into a single
    /// aggregate cut (the one with maximum value at the current centre).
    fn aggregate_bundle(&mut self) {
        if self.bundle.len() <= 2 {
            return;
        }

        // Sort by age descending; we aggregate the oldest half.
        self.bundle.sort_by(|a, b| b.age.cmp(&a.age));

        let keep = self.config.bundle_capacity / 2;
        if keep >= self.bundle.len() {
            return;
        }

        let n = self.dimension;
        let _num_to_agg = self.bundle.len() - keep;

        // Compute aggregate cut from the tail planes (oldest).
        let values: Vec<f64> = self.bundle[keep..]
            .iter()
            .map(|p| p.evaluate(&self.stability_center))
            .collect();

        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut weights: Vec<f64> = values.iter().map(|v| (v - max_val).exp()).collect();
        let wsum: f64 = weights.iter().sum();
        if wsum > 1e-15 {
            for w in &mut weights {
                *w /= wsum;
            }
        } else {
            let u = 1.0 / weights.len() as f64;
            for w in &mut weights {
                *w = u;
            }
        }

        let mut agg_sub = vec![0.0; n];
        let mut agg_const = 0.0;
        for (idx, plane) in self.bundle[keep..].iter().enumerate() {
            agg_const += weights[idx] * plane.constant;
            for j in 0..n {
                agg_sub[j] += weights[idx] * plane.subgradient[j];
            }
        }

        let old_len = self.bundle.len();
        self.bundle.truncate(keep);
        self.bundle.push(CuttingPlane {
            subgradient: agg_sub,
            constant: agg_const,
            age: 0,
            active: true,
        });

        debug!(
            "Aggregated bundle: {} → {} planes",
            old_len,
            self.bundle.len()
        );
    }

    // -----------------------------------------------------------------------
    // Step decision
    // -----------------------------------------------------------------------

    /// Decide whether the new iterate warrants a *serious step* (move the centre)
    /// or a *null step* (keep centre, only enrich the model).
    fn serious_or_null_step(
        &mut self,
        trial: &[f64],
        trial_value: f64,
        model_value: f64,
    ) -> StepType {
        // Predicted decrease: v_k = f(x̂) − m_k(y_k)  (model predicts this much decrease)
        let predicted = self.best_value - model_value;

        // Actual decrease: δ_k = f(x̂) − f(y_k)
        let actual = self.best_value - trial_value;

        // Serious step if actual decrease ≥ threshold × predicted decrease.
        if predicted > 1e-15 && actual >= self.config.serious_step_threshold * predicted {
            self.stability_center = trial.to_vec();
            self.best_value = trial_value;
            StepType::Serious
        } else {
            StepType::Null
        }
    }

    /// Update the proximity parameter μ after a step.
    fn update_proximity(&mut self, step_type: StepType) {
        match step_type {
            StepType::Serious => {
                // Decrease μ → allow larger steps next time.
                self.mu *= self.config.mu_decrease_factor;
                if self.mu < self.config.min_mu {
                    self.mu = self.config.min_mu;
                }
            }
            StepType::Null => {
                // Increase μ → tighten around current centre.
                self.mu *= self.config.mu_increase_factor;
                if self.mu > self.config.max_mu {
                    self.mu = self.config.max_mu;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simple dense QP helper (projected gradient on simplex)
// ---------------------------------------------------------------------------

/// Solve a dense QP of the form:
///
///   min  (1/2) x^T H x + g^T x    s.t.  x ≥ 0
///
/// using projected gradient descent. Used internally for the master QP when
/// a direct simplex-dual formulation is not available.
pub fn solve_simple_qp(h: &[Vec<f64>], g: &[f64], dim: usize) -> Vec<f64> {
    let max_iter = 2000;
    let tol = 1e-12;

    let mut x = vec![0.0; dim];

    // Lipschitz constant estimate (max diagonal of H, or Frobenius-based).
    let mut lip = 1.0;
    for i in 0..dim {
        if i < h.len() && i < h[i].len() {
            if h[i][i] > lip {
                lip = h[i][i];
            }
        }
    }
    let step = 1.0 / lip;

    for _ in 0..max_iter {
        // Gradient: Hx + g
        let mut grad = vec![0.0; dim];
        for i in 0..dim {
            grad[i] = g[i];
            for j in 0..dim {
                if i < h.len() && j < h[i].len() {
                    grad[i] += h[i][j] * x[j];
                }
            }
        }

        let mut max_change = 0.0_f64;
        for i in 0..dim {
            let x_new = (x[i] - step * grad[i]).max(0.0);
            let change = (x_new - x[i]).abs();
            if change > max_change {
                max_change = change;
            }
            x[i] = x_new;
        }

        if max_change < tol {
            break;
        }
    }

    x
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Project a vector onto the probability simplex Δ = { x ≥ 0 : Σx_i = 1 }.
///
/// Algorithm by Duchi et al. (2008): sort descending, find the threshold, clip.
fn project_onto_simplex(v: &mut [f64]) {
    let n = v.len();
    if n == 0 {
        return;
    }

    let mut sorted: Vec<f64> = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut rho = 0;
    for j in 0..n {
        cumsum += sorted[j];
        if sorted[j] + (1.0 - cumsum) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }

    let mut cumsum_rho = 0.0;
    for j in 0..=rho {
        cumsum_rho += sorted[j];
    }
    let theta = (cumsum_rho - 1.0) / (rho as f64 + 1.0);

    for vi in v.iter_mut() {
        *vi = (*vi - theta).max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple quadratic: f(x) = 0.5*(x-3)^2, subgradient g(x) = x-3.
    fn quadratic_oracle(x: &[f64]) -> OptResult<SubgradientInfo> {
        let val = 0.5 * (x[0] - 3.0).powi(2);
        let grad = vec![x[0] - 3.0];
        Ok(SubgradientInfo {
            point: x.to_vec(),
            value: val,
            subgradient: grad,
        })
    }

    /// Nonsmooth: f(x) = max(x, -x) = |x|, subgradient = sign(x).
    fn abs_oracle(x: &[f64]) -> OptResult<SubgradientInfo> {
        let val = x[0].abs();
        let grad = if x[0] >= 0.0 {
            vec![1.0]
        } else {
            vec![-1.0]
        };
        Ok(SubgradientInfo {
            point: x.to_vec(),
            value: val,
            subgradient: grad,
        })
    }

    #[test]
    fn test_cutting_plane_evaluate() {
        let cp = CuttingPlane {
            subgradient: vec![2.0, -1.0],
            constant: 3.0,
            age: 0,
            active: true,
        };
        // 2*1 + (-1)*2 + 3 = 3
        assert!((cp.evaluate(&[1.0, 2.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_dot_product() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_simplex_already_on() {
        let mut v = vec![0.5, 0.3, 0.2];
        project_onto_simplex(&mut v);
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(v.iter().all(|&x| x >= -1e-12));
    }

    #[test]
    fn test_project_onto_simplex_uniform() {
        let mut v = vec![1.0, 1.0, 1.0];
        project_onto_simplex(&mut v);
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &vi in &v {
            assert!((vi - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_project_onto_simplex_negative() {
        let mut v = vec![-1.0, 5.0, -2.0];
        project_onto_simplex(&mut v);
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(v.iter().all(|&x| x >= -1e-12));
    }

    #[test]
    fn test_bundle_new() {
        let cfg = BundleConfig::default();
        let bm = BundleMethod::new(cfg, 3);
        assert_eq!(bm.dimension, 3);
        assert!(bm.bundle.is_empty());
        assert_eq!(bm.stability_center.len(), 3);
    }

    #[test]
    fn test_warm_start() {
        let cfg = BundleConfig::default();
        let mut bm = BundleMethod::new(cfg, 2);
        bm.warm_start(vec![1.0, 2.0], 5.0);
        assert!((bm.stability_center[0] - 1.0).abs() < 1e-12);
        assert!((bm.best_value - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_add_cutting_plane() {
        let cfg = BundleConfig::default();
        let mut bm = BundleMethod::new(cfg, 2);
        let info = SubgradientInfo {
            point: vec![1.0, 0.0],
            value: 3.0,
            subgradient: vec![2.0, 1.0],
        };
        bm.add_cutting_plane(&info);
        assert_eq!(bm.bundle.len(), 1);
        // constant = 3 - (2*1 + 1*0) = 1
        assert!((bm.bundle[0].constant - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_quadratic() {
        let cfg = BundleConfig {
            max_iterations: 200,
            gap_tolerance: 1e-4,
            verbose: false,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 1);
        let result = bm.solve(quadratic_oracle).unwrap();
        // Minimum at x=3, value=0.
        assert!(result.optimal_value < 0.1, "value should be near 0, got {}", result.optimal_value);
        assert!(
            (result.optimal_point[0] - 3.0).abs() < 0.5,
            "point should be near 3, got {}",
            result.optimal_point[0]
        );
    }

    #[test]
    fn test_solve_abs() {
        let cfg = BundleConfig {
            max_iterations: 300,
            gap_tolerance: 1e-3,
            verbose: false,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 1);
        let result = bm.solve(abs_oracle).unwrap();
        assert!(result.optimal_value < 0.1, "value should be near 0, got {}", result.optimal_value);
    }

    #[test]
    fn test_solve_2d_quadratic() {
        // f(x,y) = (x-1)^2 + (y-2)^2
        let oracle = |x: &[f64]| -> OptResult<SubgradientInfo> {
            let val = (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
            let grad = vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
            Ok(SubgradientInfo {
                point: x.to_vec(),
                value: val,
                subgradient: grad,
            })
        };

        let cfg = BundleConfig {
            max_iterations: 500,
            gap_tolerance: 1e-3,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 2);
        let result = bm.solve(oracle).unwrap();
        assert!(result.optimal_value < 1.0, "value={}", result.optimal_value);
    }

    #[test]
    fn test_aggregate_bundle() {
        let cfg = BundleConfig {
            bundle_capacity: 4,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 2);
        for i in 0..6 {
            let info = SubgradientInfo {
                point: vec![i as f64, 0.0],
                value: (i as f64).powi(2),
                subgradient: vec![2.0 * i as f64, 0.0],
            };
            bm.add_cutting_plane(&info);
        }
        assert_eq!(bm.bundle.len(), 6);
        bm.aggregate_bundle();
        // Should be reduced.
        assert!(bm.bundle.len() <= 4, "got {} planes", bm.bundle.len());
    }

    #[test]
    fn test_step_decision_serious() {
        let cfg = BundleConfig {
            serious_step_threshold: 0.1,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 1);
        bm.best_value = 10.0;
        // model predicts decrease of 10 → 2, actual is 10 → 3.
        // predicted = 10 - 2 = 8, actual = 10 - 3 = 7, ratio = 7/8 > 0.1 → serious.
        let step = bm.serious_or_null_step(&[1.0], 3.0, 2.0);
        assert_eq!(step, StepType::Serious);
        assert!((bm.best_value - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_step_decision_null() {
        let cfg = BundleConfig {
            serious_step_threshold: 0.5,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 1);
        bm.best_value = 10.0;
        // predicted = 10 - 5 = 5, actual = 10 - 9 = 1, ratio = 1/5 = 0.2 < 0.5 → null.
        let step = bm.serious_or_null_step(&[1.0], 9.0, 5.0);
        assert_eq!(step, StepType::Null);
        assert!((bm.best_value - 10.0).abs() < 1e-12); // centre unchanged
    }

    #[test]
    fn test_proximity_update() {
        let cfg = BundleConfig {
            initial_mu: 1.0,
            mu_increase_factor: 2.0,
            mu_decrease_factor: 0.5,
            min_mu: 0.01,
            max_mu: 100.0,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 1);
        bm.update_proximity(StepType::Null);
        assert!((bm.mu - 2.0).abs() < 1e-12);
        bm.update_proximity(StepType::Serious);
        assert!((bm.mu - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_simple_qp() {
        // min 0.5 * x^2 - x  subject to x >= 0  →  x* = 1
        let h = vec![vec![1.0]];
        let g = vec![-1.0];
        let x = solve_simple_qp(&h, &g, 1);
        assert!((x[0] - 1.0).abs() < 1e-4, "got {}", x[0]);
    }

    #[test]
    fn test_solve_simple_qp_2d() {
        // min 0.5*(x1^2 + x2^2) - x1 - x2, x>=0  →  x* = (1, 1)
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let g = vec![-1.0, -1.0];
        let x = solve_simple_qp(&h, &g, 2);
        assert!((x[0] - 1.0).abs() < 1e-4, "x1={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x2={}", x[1]);
    }

    #[test]
    fn test_history_recorded() {
        let cfg = BundleConfig {
            max_iterations: 10,
            gap_tolerance: 1e-12,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(cfg, 1);
        let result = bm.solve(quadratic_oracle).unwrap();
        assert!(!result.history.is_empty());
        assert_eq!(result.history.len(), result.iterations);
    }
}
