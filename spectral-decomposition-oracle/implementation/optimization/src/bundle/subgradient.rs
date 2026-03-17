//! Subgradient methods for nonsmooth convex minimisation.
//!
//! Provides a simpler (and often faster per-iteration) alternative to the
//! proximal bundle method. Several step-size rules are implemented, including
//! Polyak, diminishing, constant, geometric and adaptive (Camerini–Fratta–
//! Maffioli) rules.  An optional Volume Algorithm variant is also provided.

use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::{OptError, OptResult};
use crate::bundle::{BundleResult, IterationInfo, StepType, SubgradientInfo};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Step-size rule used by the subgradient solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubgradientStepRule {
    /// Fixed step size t_k = c.
    Constant(f64),
    /// Diminishing: t_k = initial / sqrt(k+1).
    Diminishing,
    /// Polyak: t_k = (f(x_k) − f*_est) / ||g_k||².
    Polyak,
    /// Camerini–Fratta–Maffioli adaptive rule.
    Adaptive,
    /// Geometric: t_k = initial * ratio^k.
    Geometric(f64),
}

/// Configuration for the subgradient solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgradientConfig {
    pub max_iterations: usize,
    pub step_rule: SubgradientStepRule,
    pub initial_step_size: f64,
    pub step_decay: f64,
    pub best_bound_estimate: f64,
    pub min_step_size: f64,
    pub use_averaging: bool,
    pub oscillation_window: usize,
    pub verbose: bool,
}

impl Default for SubgradientConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            step_rule: SubgradientStepRule::Polyak,
            initial_step_size: 2.0,
            step_decay: 0.95,
            best_bound_estimate: f64::INFINITY,
            min_step_size: 1e-10,
            use_averaging: true,
            oscillation_window: 20,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Subgradient optimisation solver.
pub struct SubgradientSolver {
    config: SubgradientConfig,
    dimension: usize,
    current_point: Vec<f64>,
    best_point: Vec<f64>,
    best_value: f64,
    iterate_sum: Vec<f64>,
    weight_sum: f64,
    prev_subgradient: Option<Vec<f64>>,
    // Adaptive step state.
    adaptive_lambda: f64,
    value_history: Vec<f64>,
}

impl SubgradientSolver {
    /// Create a new subgradient solver for an `n`-dimensional problem.
    pub fn new(config: SubgradientConfig, dimension: usize) -> Self {
        Self {
            dimension,
            current_point: vec![0.0; dimension],
            best_point: vec![0.0; dimension],
            best_value: f64::INFINITY,
            iterate_sum: vec![0.0; dimension],
            weight_sum: 0.0,
            prev_subgradient: None,
            adaptive_lambda: config.initial_step_size,
            value_history: Vec::new(),
            config,
        }
    }

    /// Set an initial starting point.
    pub fn set_initial_point(&mut self, point: Vec<f64>) {
        assert_eq!(point.len(), self.dimension);
        self.current_point = point.clone();
        self.best_point = point;
    }

    // -----------------------------------------------------------------------
    // Main solve loop
    // -----------------------------------------------------------------------

    /// Run the subgradient method.
    pub fn solve<F>(&mut self, mut oracle: F) -> OptResult<BundleResult>
    where
        F: FnMut(&[f64]) -> OptResult<SubgradientInfo>,
    {
        let start = Instant::now();
        let mut history = Vec::new();
        let mut best_bound = f64::NEG_INFINITY;

        for iter in 0..self.config.max_iterations {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 3600.0 {
                return Err(OptError::TimeLimitExceeded {
                    elapsed,
                    limit: 3600.0,
                });
            }

            // Evaluate oracle.
            let info = oracle(&self.current_point)?;
            let value = info.value;
            let subgrad = &info.subgradient;

            self.value_history.push(value);

            // Track best.
            if value < self.best_value {
                self.best_value = value;
                self.best_point = self.current_point.clone();
            }

            // Compute step size.
            let step = self.compute_step_size(iter, subgrad, value);

            // Update: x_{k+1} = x_k - t_k * g_k  (minimisation).
            for j in 0..self.dimension {
                self.current_point[j] -= step * subgrad[j];
            }

            // Ergodic averaging.
            if self.config.use_averaging {
                let weight = step;
                for j in 0..self.dimension {
                    self.iterate_sum[j] += weight * self.current_point[j];
                }
                self.weight_sum += weight;
            }

            // Simple lower bound tracking: the model value at the best point
            // provides a bound when using the Lagrangian interpretation.
            // Here we use the best value seen.
            if value > best_bound {
                best_bound = value;
            }

            let gap = if self.best_value.abs() > 1e-10 {
                (self.best_value - best_bound).abs() / self.best_value.abs()
            } else {
                (self.best_value - best_bound).abs()
            };

            let step_type = if value < self.best_value + 1e-12 {
                StepType::Serious
            } else {
                StepType::Null
            };

            history.push(IterationInfo {
                iteration: iter,
                objective: self.best_value,
                best_bound,
                gap,
                step_type,
                mu: step,
            });

            if self.config.verbose {
                info!(
                    "Subgrad iter {}: val={:.6}, best={:.6}, step={:.4e}",
                    iter, value, self.best_value, step
                );
            }

            // Detect stagnation / oscillation and adapt.
            if iter > 0 && self.detect_oscillation() {
                self.adaptive_lambda *= self.config.step_decay;
                debug!("Oscillation detected at iter {}, reducing lambda to {:.4e}", iter, self.adaptive_lambda);
            }

            self.prev_subgradient = Some(subgrad.to_vec());

            if step < self.config.min_step_size {
                debug!("Step size below minimum at iteration {}", iter);
                break;
            }
        }

        let optimal_point = if self.config.use_averaging && self.weight_sum > 1e-15 {
            self.ergodic_average()
        } else {
            self.best_point.clone()
        };

        let gap = if self.best_value.abs() > 1e-10 {
            (self.best_value - best_bound).abs() / self.best_value.abs()
        } else {
            (self.best_value - best_bound).abs()
        };

        Ok(BundleResult {
            optimal_point,
            optimal_value: self.best_value,
            iterations: history.len(),
            gap,
            converged: gap < 1e-4,
            history,
        })
    }

    // -----------------------------------------------------------------------
    // Step-size computation
    // -----------------------------------------------------------------------

    /// Compute the step size for the current iteration.
    fn compute_step_size(&self, iteration: usize, subgradient: &[f64], current_value: f64) -> f64 {
        let norm_sq: f64 = subgradient.iter().map(|g| g * g).sum();
        if norm_sq < 1e-30 {
            return self.config.min_step_size;
        }

        let raw = match &self.config.step_rule {
            SubgradientStepRule::Constant(c) => *c,
            SubgradientStepRule::Diminishing => self.diminishing_step(iteration),
            SubgradientStepRule::Polyak => {
                Self::polyak_step(current_value, self.config.best_bound_estimate, norm_sq)
            }
            SubgradientStepRule::Adaptive => self.adaptive_step(subgradient, norm_sq, current_value),
            SubgradientStepRule::Geometric(ratio) => {
                self.config.initial_step_size * ratio.powi(iteration as i32)
            }
        };

        raw.max(self.config.min_step_size)
    }

    /// Polyak step: t = λ(f(x) − f*) / ||g||².
    fn polyak_step(current_value: f64, best_bound: f64, subgradient_norm_sq: f64) -> f64 {
        if subgradient_norm_sq < 1e-30 {
            return 1e-10;
        }
        let gap = current_value - best_bound;
        if gap <= 0.0 {
            // Already at or past the target.
            return 1e-10;
        }
        gap / subgradient_norm_sq
    }

    /// Diminishing step: t = c / sqrt(k+1).
    fn diminishing_step(&self, iteration: usize) -> f64 {
        self.config.initial_step_size / ((iteration as f64) + 1.0).sqrt()
    }

    /// Camerini–Fratta–Maffioli adaptive step.
    ///
    /// Track the angle between consecutive subgradients. If they point in a
    /// similar direction (cosine > 0), increase λ; otherwise decrease.
    fn adaptive_step(&self, subgradient: &[f64], norm_sq: f64, current_value: f64) -> f64 {
        let mut lambda = self.adaptive_lambda;

        if let Some(ref prev) = self.prev_subgradient {
            let dot: f64 = subgradient.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            let prev_norm_sq: f64 = prev.iter().map(|v| v * v).sum();
            let denom = (norm_sq * prev_norm_sq).sqrt();
            if denom > 1e-15 {
                let cosine = dot / denom;
                if cosine > 0.1 {
                    // Same direction → increase step.
                    lambda *= 1.1;
                } else if cosine < -0.1 {
                    // Opposite direction → decrease step.
                    lambda *= 0.5;
                }
                // Otherwise keep lambda.
            }
        }

        // Polyak-like with adaptive lambda.
        let gap = current_value - self.config.best_bound_estimate;
        if gap > 0.0 && norm_sq > 1e-30 {
            lambda * gap / norm_sq
        } else {
            lambda / (norm_sq.sqrt().max(1.0))
        }
    }

    // -----------------------------------------------------------------------
    // Averaging and oscillation detection
    // -----------------------------------------------------------------------

    /// Ergodic (weighted) average of all iterates.
    pub fn ergodic_average(&self) -> Vec<f64> {
        if self.weight_sum < 1e-15 {
            return self.best_point.clone();
        }
        self.iterate_sum
            .iter()
            .map(|s| s / self.weight_sum)
            .collect()
    }

    /// Detect oscillation by counting sign changes in the objective improvement
    /// over the last `oscillation_window` iterations.
    fn detect_oscillation(&self) -> bool {
        let w = self.config.oscillation_window;
        let n = self.value_history.len();
        if n < w + 1 {
            return false;
        }

        let window = &self.value_history[n - w - 1..];
        let mut sign_changes = 0usize;
        for i in 2..window.len() {
            let diff_prev = window[i - 1] - window[i - 2];
            let diff_curr = window[i] - window[i - 1];
            if diff_prev * diff_curr < 0.0 {
                sign_changes += 1;
            }
        }
        // Oscillating if more than half of the window shows sign changes.
        sign_changes > w / 2
    }

    // -----------------------------------------------------------------------
    // Volume algorithm
    // -----------------------------------------------------------------------

    /// Volume algorithm variant.
    ///
    /// Maintains primal weights and builds a convex combination of subproblem
    /// solutions to approximate a primal feasible solution while driving the
    /// subgradient iteration.
    pub fn volume_algorithm<F>(&mut self, mut oracle: F) -> OptResult<BundleResult>
    where
        F: FnMut(&[f64]) -> OptResult<SubgradientInfo>,
    {
        let start = Instant::now();
        let mut history = Vec::new();
        let mut best_bound = f64::NEG_INFINITY;

        // Primal estimate as a weighted average of oracle points.
        let mut primal_avg = vec![0.0; self.dimension];
        let mut total_weight = 0.0;

        for iter in 0..self.config.max_iterations {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 3600.0 {
                return Err(OptError::TimeLimitExceeded {
                    elapsed,
                    limit: 3600.0,
                });
            }

            let info = oracle(&self.current_point)?;
            let value = info.value;

            self.value_history.push(value);

            if value < self.best_value {
                self.best_value = value;
                self.best_point = self.current_point.clone();
            }

            if value > best_bound {
                best_bound = value;
            }

            // Volume algorithm step-size: use adaptive/Polyak.
            let norm_sq: f64 = info.subgradient.iter().map(|g| g * g).sum();
            let step = if norm_sq > 1e-30 {
                let gap = value - self.config.best_bound_estimate;
                if gap > 0.0 {
                    self.adaptive_lambda * gap / norm_sq
                } else {
                    self.adaptive_lambda / norm_sq.sqrt().max(1.0)
                }
            } else {
                self.config.min_step_size
            };

            // Subgradient update.
            for j in 0..self.dimension {
                self.current_point[j] -= step * info.subgradient[j];
            }

            // Primal weight update: volume algorithm uses
            // w_{k+1} = (1-α_k) w_k + α_k x_k where α_k = 1/(k+1).
            let alpha = 1.0 / (iter as f64 + 2.0);
            total_weight += 1.0;
            for j in 0..self.dimension {
                primal_avg[j] = (1.0 - alpha) * primal_avg[j] + alpha * info.point[j];
            }

            // Adaptive lambda update based on consecutive subgradient direction.
            if let Some(ref prev) = self.prev_subgradient {
                let dot_val: f64 = info
                    .subgradient
                    .iter()
                    .zip(prev.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let prev_norm_sq: f64 = prev.iter().map(|v| v * v).sum();
                let denom = (norm_sq * prev_norm_sq).sqrt();
                if denom > 1e-15 {
                    let cosine = dot_val / denom;
                    if cosine > 0.1 {
                        self.adaptive_lambda = (self.adaptive_lambda * 1.05).min(10.0);
                    } else if cosine < -0.1 {
                        self.adaptive_lambda = (self.adaptive_lambda * 0.7).max(0.001);
                    }
                }
            }

            let gap = if self.best_value.abs() > 1e-10 {
                (self.best_value - best_bound).abs() / self.best_value.abs()
            } else {
                (self.best_value - best_bound).abs()
            };

            history.push(IterationInfo {
                iteration: iter,
                objective: self.best_value,
                best_bound,
                gap,
                step_type: if value <= self.best_value + 1e-12 {
                    StepType::Serious
                } else {
                    StepType::Null
                },
                mu: step,
            });

            if self.config.verbose {
                info!(
                    "Volume iter {}: val={:.6}, best={:.6}, step={:.4e}, lambda={:.4e}",
                    iter, value, self.best_value, step, self.adaptive_lambda
                );
            }

            self.prev_subgradient = Some(info.subgradient.clone());

            if step < self.config.min_step_size {
                break;
            }
        }

        let gap = if self.best_value.abs() > 1e-10 {
            (self.best_value - best_bound).abs() / self.best_value.abs()
        } else {
            (self.best_value - best_bound).abs()
        };

        // Use the primal average as the optimal point if averaging is on.
        let optimal_point = if total_weight > 0.0 {
            primal_avg
        } else {
            self.best_point.clone()
        };

        Ok(BundleResult {
            optimal_point,
            optimal_value: self.best_value,
            iterations: history.len(),
            gap,
            converged: gap < 1e-4,
            history,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x) = |x - 2|
    fn abs_shifted_oracle(x: &[f64]) -> OptResult<SubgradientInfo> {
        let d = x[0] - 2.0;
        let val = d.abs();
        let grad = if d >= 0.0 { vec![1.0] } else { vec![-1.0] };
        Ok(SubgradientInfo {
            point: x.to_vec(),
            value: val,
            subgradient: grad,
        })
    }

    /// f(x) = 0.5 * ||x - target||^2
    fn quadratic_oracle_2d(x: &[f64]) -> OptResult<SubgradientInfo> {
        let target = [1.0, 2.0];
        let mut val = 0.0;
        let mut grad = vec![0.0; 2];
        for i in 0..2 {
            val += 0.5 * (x[i] - target[i]).powi(2);
            grad[i] = x[i] - target[i];
        }
        Ok(SubgradientInfo {
            point: x.to_vec(),
            value: val,
            subgradient: grad,
        })
    }

    /// f(x) = max(x, 2*(-x) + 3) – piecewise linear
    fn piecewise_oracle(x: &[f64]) -> OptResult<SubgradientInfo> {
        let f1 = x[0];
        let f2 = -2.0 * x[0] + 3.0;
        if f1 >= f2 {
            Ok(SubgradientInfo {
                point: x.to_vec(),
                value: f1,
                subgradient: vec![1.0],
            })
        } else {
            Ok(SubgradientInfo {
                point: x.to_vec(),
                value: f2,
                subgradient: vec![-2.0],
            })
        }
    }

    #[test]
    fn test_subgradient_config_default() {
        let cfg = SubgradientConfig::default();
        assert_eq!(cfg.max_iterations, 1000);
        assert!(cfg.use_averaging);
        assert_eq!(cfg.oscillation_window, 20);
    }

    #[test]
    fn test_solver_new() {
        let cfg = SubgradientConfig::default();
        let solver = SubgradientSolver::new(cfg, 3);
        assert_eq!(solver.dimension, 3);
        assert_eq!(solver.current_point.len(), 3);
        assert!(solver.best_value.is_infinite());
    }

    #[test]
    fn test_set_initial_point() {
        let cfg = SubgradientConfig::default();
        let mut solver = SubgradientSolver::new(cfg, 2);
        solver.set_initial_point(vec![3.0, 4.0]);
        assert!((solver.current_point[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_diminishing_step() {
        let cfg = SubgradientConfig {
            initial_step_size: 2.0,
            ..SubgradientConfig::default()
        };
        let solver = SubgradientSolver::new(cfg, 1);
        let s0 = solver.diminishing_step(0);
        let s99 = solver.diminishing_step(99);
        assert!((s0 - 2.0).abs() < 1e-12);
        assert!(s99 < s0);
        assert!((s99 - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_polyak_step() {
        // gap=4, norm_sq=2 → step = 4/2 = 2
        let step = SubgradientSolver::polyak_step(6.0, 2.0, 2.0);
        assert!((step - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_polyak_step_zero_gap() {
        let step = SubgradientSolver::polyak_step(2.0, 2.0, 4.0);
        assert!(step <= 1e-9);
    }

    #[test]
    fn test_solve_constant_step() {
        let cfg = SubgradientConfig {
            max_iterations: 500,
            step_rule: SubgradientStepRule::Constant(0.01),
            use_averaging: true,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 1);
        let result = solver.solve(abs_shifted_oracle).unwrap();
        // Should get close to x=2, value=0.
        assert!(result.optimal_value < 0.5, "val={}", result.optimal_value);
    }

    #[test]
    fn test_solve_diminishing_step() {
        let cfg = SubgradientConfig {
            max_iterations: 1000,
            step_rule: SubgradientStepRule::Diminishing,
            initial_step_size: 1.0,
            use_averaging: true,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 1);
        let result = solver.solve(abs_shifted_oracle).unwrap();
        assert!(
            result.optimal_value < 1.0,
            "val={}",
            result.optimal_value
        );
    }

    #[test]
    fn test_solve_geometric_step() {
        let cfg = SubgradientConfig {
            max_iterations: 500,
            step_rule: SubgradientStepRule::Geometric(0.99),
            initial_step_size: 0.5,
            use_averaging: true,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 2);
        let result = solver.solve(quadratic_oracle_2d).unwrap();
        assert!(result.optimal_value < 1.0, "val={}", result.optimal_value);
    }

    #[test]
    fn test_solve_piecewise() {
        // min max(x, -2x+3) → optimum at x=1, value=1
        let cfg = SubgradientConfig {
            max_iterations: 2000,
            step_rule: SubgradientStepRule::Diminishing,
            initial_step_size: 2.0,
            use_averaging: true,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 1);
        let result = solver.solve(piecewise_oracle).unwrap();
        assert!(
            result.optimal_value < 2.0,
            "val={}",
            result.optimal_value
        );
    }

    #[test]
    fn test_detect_oscillation_no_history() {
        let cfg = SubgradientConfig::default();
        let solver = SubgradientSolver::new(cfg, 1);
        assert!(!solver.detect_oscillation());
    }

    #[test]
    fn test_detect_oscillation_yes() {
        let cfg = SubgradientConfig {
            oscillation_window: 5,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 1);
        // Alternating up and down.
        solver.value_history = vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0];
        assert!(solver.detect_oscillation());
    }

    #[test]
    fn test_detect_oscillation_no() {
        let cfg = SubgradientConfig {
            oscillation_window: 5,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 1);
        // Monotonically decreasing.
        solver.value_history = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0];
        assert!(!solver.detect_oscillation());
    }

    #[test]
    fn test_ergodic_average_empty() {
        let cfg = SubgradientConfig::default();
        let solver = SubgradientSolver::new(cfg, 2);
        let avg = solver.ergodic_average();
        assert_eq!(avg.len(), 2);
    }

    #[test]
    fn test_volume_algorithm() {
        let cfg = SubgradientConfig {
            max_iterations: 300,
            step_rule: SubgradientStepRule::Adaptive,
            initial_step_size: 1.0,
            best_bound_estimate: 0.0,
            use_averaging: true,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 2);
        let result = solver.volume_algorithm(quadratic_oracle_2d).unwrap();
        assert!(
            result.optimal_value < 2.0,
            "val={}",
            result.optimal_value
        );
        assert!(!result.history.is_empty());
    }

    #[test]
    fn test_adaptive_step_rule() {
        let cfg = SubgradientConfig {
            max_iterations: 500,
            step_rule: SubgradientStepRule::Adaptive,
            initial_step_size: 1.0,
            best_bound_estimate: 0.0,
            use_averaging: true,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 1);
        let result = solver.solve(abs_shifted_oracle).unwrap();
        assert!(result.optimal_value < 1.0, "val={}", result.optimal_value);
    }

    #[test]
    fn test_history_length_matches_iterations() {
        let cfg = SubgradientConfig {
            max_iterations: 50,
            step_rule: SubgradientStepRule::Constant(0.1),
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(cfg, 1);
        let result = solver.solve(abs_shifted_oracle).unwrap();
        assert_eq!(result.history.len(), result.iterations);
    }
}
