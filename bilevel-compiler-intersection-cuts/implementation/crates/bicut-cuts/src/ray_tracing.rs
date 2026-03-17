//! Ray-boundary intersection computation for bilevel intersection cuts.
//!
//! Traces rays from an LP vertex through the bilevel-infeasible set B_bar,
//! detects critical region transitions along each ray, and computes exact
//! intersection with the bilevel-feasible boundary {(x,y): c^T y = phi(x)}.

use crate::balas::RayLength;
use crate::{BilevelCut, CutError, CutResult, BIG_M, TOLERANCE};
use bicut_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the ray tracer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayTracerConfig {
    /// Maximum step length along a ray before declaring it infinite.
    pub max_step: f64,
    /// Step size for initial bracketing when binary search is needed.
    pub bracket_step: f64,
    /// Tolerance for declaring intersection found.
    pub tolerance: f64,
    /// Maximum number of bisection iterations.
    pub max_bisection_iters: usize,
    /// Maximum number of region transitions to follow along a single ray.
    pub max_region_transitions: usize,
    /// Whether to use adaptive step sizes based on gradient information.
    pub adaptive_stepping: bool,
    /// Minimum step size for adaptive stepping.
    pub min_step: f64,
}

impl Default for RayTracerConfig {
    fn default() -> Self {
        Self {
            max_step: 1e8,
            bracket_step: 0.1,
            tolerance: TOLERANCE,
            max_bisection_iters: 60,
            max_region_transitions: 100,
            adaptive_stepping: true,
            min_step: 1e-12,
        }
    }
}

/// A transition between critical regions detected along a ray.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionTransition {
    /// Step length t at which the transition occurs.
    pub t: f64,
    /// ID of the region we are leaving.
    pub from_region: Option<usize>,
    /// ID of the region we are entering.
    pub to_region: Option<usize>,
    /// The point (x,y) at the transition.
    pub point: Vec<f64>,
    /// Value of c^T y - phi(x) at the transition point.
    pub gap: f64,
}

/// Result of tracing a single ray.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayResult {
    /// The nonbasic variable index defining this ray.
    pub variable_index: usize,
    /// Whether the ray intersects the bilevel-feasible boundary.
    pub intersects: bool,
    /// Step length to the boundary (alpha_j).
    pub alpha: f64,
    /// The boundary point, if intersection was found.
    pub boundary_point: Option<Vec<f64>>,
    /// All region transitions detected along this ray.
    pub transitions: Vec<RegionTransition>,
    /// The bilevel gap c^T y - phi(x) at the boundary point.
    pub boundary_gap: f64,
    /// Number of bisection iterations used.
    pub bisection_iters: usize,
    /// Whether the ray exits the LP feasible region before reaching the boundary.
    pub exits_lp_feasible: bool,
    /// Step length at which the ray exits LP feasibility.
    pub lp_exit_t: f64,
}

impl RayResult {
    pub fn to_ray_length(&self) -> RayLength {
        if self.intersects {
            let mut rl = RayLength::finite(self.variable_index, self.alpha, true);
            rl.intersection_point = self.boundary_point.clone();
            rl
        } else {
            RayLength::infinite(self.variable_index, true)
        }
    }
}

/// Simplex ray direction: how basic variables change when we increase a nonbasic variable.
#[derive(Debug, Clone)]
pub struct SimplexRayDirection {
    pub nonbasic_var: usize,
    pub basic_var_changes: Vec<(usize, f64)>,
    pub at_lower_bound: bool,
}

/// The ray tracer computes intersections with the bilevel-feasible boundary.
#[derive(Debug, Clone)]
pub struct RayTracer {
    pub config: RayTracerConfig,
    /// Number of leader variables (to split (x,y) vectors).
    n_leader: usize,
    /// Number of follower variables.
    n_follower: usize,
    /// Follower objective coefficients c.
    follower_obj: Vec<f64>,
    /// Cached critical region lookup.
    region_cache: HashMap<usize, Vec<f64>>,
    /// Statistics.
    total_rays_traced: usize,
    total_bisections: usize,
}

impl RayTracer {
    pub fn new(
        config: RayTracerConfig,
        n_leader: usize,
        n_follower: usize,
        follower_obj: Vec<f64>,
    ) -> Self {
        Self {
            config,
            n_leader,
            n_follower,
            follower_obj,
            region_cache: HashMap::new(),
            total_rays_traced: 0,
            total_bisections: 0,
        }
    }

    /// Compute the bilevel gap: c^T y - phi(x).
    /// Positive means bilevel-infeasible, zero/negative means feasible.
    fn bilevel_gap(&self, point: &[f64], phi_x: f64) -> f64 {
        let y_start = self.n_leader;
        let cy: f64 = self
            .follower_obj
            .iter()
            .enumerate()
            .map(|(i, &c)| c * point.get(y_start + i).copied().unwrap_or(0.0))
            .sum();
        cy - phi_x
    }

    /// Evaluate phi(x) using a simple affine region lookup.
    fn evaluate_phi(
        &self,
        x: &[f64],
        regions: &[(Vec<f64>, f64, Vec<(Vec<f64>, f64)>)],
    ) -> Option<f64> {
        // Each region is (gradient, offset, [(normal, rhs)] defining the region polytope)
        for (gradient, offset, halfspaces) in regions {
            let mut inside = true;
            for (normal, rhs) in halfspaces {
                let val: f64 = normal.iter().zip(x.iter()).map(|(n, v)| n * v).sum();
                if val > rhs + self.config.tolerance {
                    inside = false;
                    break;
                }
            }
            if inside {
                let phi: f64 = *offset
                    + gradient
                        .iter()
                        .zip(x.iter())
                        .map(|(g, v)| g * v)
                        .sum::<f64>();
                return Some(phi);
            }
        }
        None
    }

    /// Compute the point along a ray: point(t) = vertex + t * direction.
    fn ray_point(&self, vertex: &[f64], direction: &[f64], t: f64) -> Vec<f64> {
        vertex
            .iter()
            .zip(direction.iter())
            .map(|(&v, &d)| v + t * d)
            .collect()
    }

    /// Check LP feasibility of a point given bounds.
    fn check_lp_feasibility(
        &self,
        point: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> bool {
        for (i, &val) in point.iter().enumerate() {
            let lb = lower_bounds.get(i).copied().unwrap_or(f64::NEG_INFINITY);
            let ub = upper_bounds.get(i).copied().unwrap_or(f64::INFINITY);
            if val < lb - self.config.tolerance || val > ub + self.config.tolerance {
                return false;
            }
        }
        true
    }

    /// Find the maximum t such that vertex + t*direction remains LP-feasible.
    fn max_feasible_step(
        &self,
        vertex: &[f64],
        direction: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> f64 {
        let mut max_t = self.config.max_step;
        for (i, (&v, &d)) in vertex.iter().zip(direction.iter()).enumerate() {
            if d.abs() < self.config.tolerance {
                continue;
            }
            let lb = lower_bounds.get(i).copied().unwrap_or(f64::NEG_INFINITY);
            let ub = upper_bounds.get(i).copied().unwrap_or(f64::INFINITY);
            if d > 0.0 && ub < f64::INFINITY {
                let t = (ub - v) / d;
                if t > 0.0 {
                    max_t = max_t.min(t);
                }
            } else if d < 0.0 && lb > f64::NEG_INFINITY {
                let t = (lb - v) / d;
                if t > 0.0 {
                    max_t = max_t.min(t);
                }
            }
        }
        max_t
    }

    /// Trace a single ray from vertex along a simplex direction.
    /// Returns the step length to the bilevel-feasible boundary.
    pub fn trace_ray(
        &mut self,
        vertex: &[f64],
        direction: &[f64],
        variable_index: usize,
        phi_evaluator: &dyn Fn(&[f64]) -> Option<f64>,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> CutResult<RayResult> {
        self.total_rays_traced += 1;
        let n = vertex.len();

        // Check that the vertex is bilevel-infeasible.
        let x_vertex = &vertex[..self.n_leader.min(n)];
        let phi_at_vertex = phi_evaluator(x_vertex);

        let gap_at_vertex = match phi_at_vertex {
            Some(phi) => self.bilevel_gap(vertex, phi),
            None => {
                return Ok(RayResult {
                    variable_index,
                    intersects: false,
                    alpha: f64::INFINITY,
                    boundary_point: None,
                    transitions: Vec::new(),
                    boundary_gap: 0.0,
                    bisection_iters: 0,
                    exits_lp_feasible: false,
                    lp_exit_t: f64::INFINITY,
                })
            }
        };

        if gap_at_vertex <= self.config.tolerance {
            return Err(CutError::AlreadyFeasible);
        }

        // Find maximum feasible step.
        let max_t = self.max_feasible_step(vertex, direction, lower_bounds, upper_bounds);
        let exits_lp = max_t < self.config.max_step;

        // Binary search for the boundary: find t where gap changes sign.
        let mut lo = 0.0;
        let mut hi = max_t;
        let mut transitions = Vec::new();
        let mut bisection_iters = 0;

        // First, check if the gap is still positive at the LP boundary.
        let point_at_max = self.ray_point(vertex, direction, hi);
        let x_at_max = &point_at_max[..self.n_leader.min(point_at_max.len())];
        let phi_at_max = phi_evaluator(x_at_max);
        let gap_at_max = match phi_at_max {
            Some(phi) => self.bilevel_gap(&point_at_max, phi),
            None => gap_at_vertex,
        };

        if gap_at_max > self.config.tolerance {
            // Ray never reaches feasible boundary within LP region.
            return Ok(RayResult {
                variable_index,
                intersects: false,
                alpha: f64::INFINITY,
                boundary_point: None,
                transitions,
                boundary_gap: gap_at_max,
                bisection_iters: 0,
                exits_lp_feasible: exits_lp,
                lp_exit_t: max_t,
            });
        }

        // Bisection to find the exact crossing point.
        while hi - lo > self.config.tolerance && bisection_iters < self.config.max_bisection_iters {
            let mid = (lo + hi) / 2.0;
            let point_mid = self.ray_point(vertex, direction, mid);
            let x_mid = &point_mid[..self.n_leader.min(point_mid.len())];
            let phi_mid = phi_evaluator(x_mid);
            let gap_mid = match phi_mid {
                Some(phi) => self.bilevel_gap(&point_mid, phi),
                None => gap_at_vertex * 0.5,
            };

            if gap_mid > self.config.tolerance {
                lo = mid;
            } else {
                hi = mid;
            }
            bisection_iters += 1;
        }
        self.total_bisections += bisection_iters;

        let alpha = (lo + hi) / 2.0;
        let boundary_point = self.ray_point(vertex, direction, alpha);
        let x_boundary = &boundary_point[..self.n_leader.min(boundary_point.len())];
        let phi_boundary = phi_evaluator(x_boundary).unwrap_or(0.0);
        let boundary_gap = self.bilevel_gap(&boundary_point, phi_boundary);

        Ok(RayResult {
            variable_index,
            intersects: true,
            alpha,
            boundary_point: Some(boundary_point),
            transitions,
            boundary_gap,
            bisection_iters,
            exits_lp_feasible: exits_lp,
            lp_exit_t: max_t,
        })
    }

    /// Trace all simplex rays from the vertex.
    pub fn trace_all_rays(
        &mut self,
        vertex: &[f64],
        ray_directions: &[SimplexRayDirection],
        phi_evaluator: &dyn Fn(&[f64]) -> Option<f64>,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> CutResult<Vec<RayResult>> {
        let n = vertex.len();
        let mut results = Vec::new();

        for ray_dir in ray_directions {
            let mut direction = vec![0.0; n];
            // The nonbasic variable changes by +1 (or -1 for upper bound).
            let sign = if ray_dir.at_lower_bound { 1.0 } else { -1.0 };
            if ray_dir.nonbasic_var < n {
                direction[ray_dir.nonbasic_var] = sign;
            }
            // Basic variables change according to the tableau coefficients.
            for &(basic_var, coeff) in &ray_dir.basic_var_changes {
                if basic_var < n {
                    direction[basic_var] = -sign * coeff;
                }
            }

            match self.trace_ray(
                vertex,
                &direction,
                ray_dir.nonbasic_var,
                phi_evaluator,
                lower_bounds,
                upper_bounds,
            ) {
                Ok(result) => results.push(result),
                Err(CutError::AlreadyFeasible) => {
                    return Err(CutError::AlreadyFeasible);
                }
                Err(_) => {
                    results.push(RayResult {
                        variable_index: ray_dir.nonbasic_var,
                        intersects: false,
                        alpha: f64::INFINITY,
                        boundary_point: None,
                        transitions: Vec::new(),
                        boundary_gap: 0.0,
                        bisection_iters: 0,
                        exits_lp_feasible: false,
                        lp_exit_t: f64::INFINITY,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Convert ray results to RayLength values for the Balas formula.
    pub fn to_ray_lengths(&self, results: &[RayResult]) -> Vec<RayLength> {
        results.iter().map(|r| r.to_ray_length()).collect()
    }

    /// Compute adaptive step size based on the gradient of the bilevel gap.
    fn adaptive_step_size(&self, gap: f64, gap_derivative: f64) -> f64 {
        if gap_derivative.abs() < self.config.tolerance {
            return self.config.bracket_step;
        }
        let newton_step = -gap / gap_derivative;
        newton_step
            .abs()
            .max(self.config.min_step)
            .min(self.config.max_step)
    }

    /// Detect region transitions along a ray by sampling at regular intervals.
    pub fn detect_transitions(
        &self,
        vertex: &[f64],
        direction: &[f64],
        max_t: f64,
        num_samples: usize,
        region_identifier: &dyn Fn(&[f64]) -> Option<usize>,
    ) -> Vec<RegionTransition> {
        let mut transitions = Vec::new();
        let dt = max_t / (num_samples as f64);
        let mut prev_region: Option<usize> = None;

        for i in 0..=num_samples {
            let t = dt * i as f64;
            let point = self.ray_point(vertex, direction, t);
            let x = &point[..self.n_leader.min(point.len())];
            let current_region = region_identifier(x);

            if i > 0 && current_region != prev_region {
                transitions.push(RegionTransition {
                    t,
                    from_region: prev_region,
                    to_region: current_region,
                    point: point.clone(),
                    gap: 0.0,
                });
            }
            prev_region = current_region;

            if transitions.len() >= self.config.max_region_transitions {
                break;
            }
        }

        transitions
    }

    /// Refine a region transition point using bisection.
    pub fn refine_transition(
        &self,
        vertex: &[f64],
        direction: &[f64],
        t_lo: f64,
        t_hi: f64,
        region_identifier: &dyn Fn(&[f64]) -> Option<usize>,
    ) -> f64 {
        let mut lo = t_lo;
        let mut hi = t_hi;
        let target_region = {
            let p = self.ray_point(vertex, direction, hi);
            let x = &p[..self.n_leader.min(p.len())];
            region_identifier(x)
        };

        for _ in 0..self.config.max_bisection_iters {
            if hi - lo < self.config.tolerance {
                break;
            }
            let mid = (lo + hi) / 2.0;
            let p = self.ray_point(vertex, direction, mid);
            let x = &p[..self.n_leader.min(p.len())];
            let mid_region = region_identifier(x);
            if mid_region == target_region {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (lo + hi) / 2.0
    }

    pub fn total_rays_traced(&self) -> usize {
        self.total_rays_traced
    }
    pub fn total_bisections(&self) -> usize {
        self.total_bisections
    }
    pub fn reset_stats(&mut self) {
        self.total_rays_traced = 0;
        self.total_bisections = 0;
    }
}

/// Build simplex ray directions from LP basis information.
pub fn build_simplex_directions(
    basis_status: &[BasisStatus],
    tableau_rows: &[(usize, Vec<f64>)],
) -> Vec<SimplexRayDirection> {
    let mut directions = Vec::new();

    for (i, &status) in basis_status.iter().enumerate() {
        let at_lower = match status {
            BasisStatus::NonBasicLower => true,
            BasisStatus::NonBasicUpper => false,
            _ => continue,
        };

        let mut basic_changes = Vec::new();
        for (basic_var, row_coeffs) in tableau_rows {
            let coeff = row_coeffs.get(i).copied().unwrap_or(0.0);
            if coeff.abs() > TOLERANCE {
                basic_changes.push((*basic_var, coeff));
            }
        }

        directions.push(SimplexRayDirection {
            nonbasic_var: i,
            basic_var_changes: basic_changes,
            at_lower_bound: at_lower,
        });
    }

    directions
}

/// Compute the LP exit step for a single ray direction.
pub fn lp_exit_step(
    vertex: &[f64],
    direction: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    tolerance: f64,
) -> f64 {
    let mut max_t = f64::INFINITY;
    for (i, (&v, &d)) in vertex.iter().zip(direction.iter()).enumerate() {
        if d.abs() < tolerance {
            continue;
        }
        let lb = lower_bounds.get(i).copied().unwrap_or(f64::NEG_INFINITY);
        let ub = upper_bounds.get(i).copied().unwrap_or(f64::INFINITY);
        if d > 0.0 && ub < f64::INFINITY {
            let t = (ub - v) / d;
            if t > tolerance {
                max_t = max_t.min(t);
            }
        } else if d < 0.0 && lb > f64::NEG_INFINITY {
            let t = (lb - v) / d;
            if t > tolerance {
                max_t = max_t.min(t);
            }
        }
    }
    max_t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_tracer_config_default() {
        let cfg = RayTracerConfig::default();
        assert!(cfg.max_step > 0.0);
        assert!(cfg.max_bisection_iters > 0);
    }

    #[test]
    fn test_ray_point() {
        let tracer = RayTracer::new(RayTracerConfig::default(), 2, 2, vec![1.0, 1.0]);
        let p = tracer.ray_point(&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5, 1.0, -1.0], 2.0);
        assert!((p[0] - 2.0).abs() < 1e-10);
        assert!((p[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_feasible_step() {
        let tracer = RayTracer::new(RayTracerConfig::default(), 1, 1, vec![1.0]);
        let t = tracer.max_feasible_step(&[0.5, 0.5], &[1.0, 0.0], &[0.0, 0.0], &[1.0, 1.0]);
        assert!((t - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_trace_ray_feasible_point() {
        let mut tracer = RayTracer::new(RayTracerConfig::default(), 1, 1, vec![1.0]);
        // phi(x) = x, point (0.5, 0.3): c^T y = 0.3, phi(x) = 0.5, gap = -0.2 (feasible)
        let result = tracer.trace_ray(
            &[0.5, 0.3],
            &[0.0, 1.0],
            0,
            &|x: &[f64]| Some(x.get(0).copied().unwrap_or(0.0)),
            &[0.0, 0.0],
            &[1.0, 1.0],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_trace_ray_infeasible_finds_boundary() {
        let mut tracer = RayTracer::new(RayTracerConfig::default(), 1, 1, vec![1.0]);
        // phi(x) = 0.3 constant, point (0.5, 0.8): c^T y = 0.8, phi = 0.3, gap = 0.5
        // direction: y decreases. At t, y = 0.8 - t. Gap = (0.8 - t) - 0.3 = 0.5 - t.
        // Boundary at t = 0.5.
        let result = tracer.trace_ray(
            &[0.5, 0.8],
            &[0.0, -1.0],
            0,
            &|_x: &[f64]| Some(0.3),
            &[0.0, 0.0],
            &[1.0, 1.0],
        );
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.intersects);
        assert!((r.alpha - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_build_simplex_directions() {
        let basis = vec![
            BasisStatus::Basic,
            BasisStatus::NonBasicLower,
            BasisStatus::NonBasicUpper,
        ];
        let tableau = vec![(0, vec![0.0, 0.5, -0.3])];
        let dirs = build_simplex_directions(&basis, &tableau);
        assert_eq!(dirs.len(), 2);
        assert!(dirs[0].at_lower_bound);
        assert!(!dirs[1].at_lower_bound);
    }

    #[test]
    fn test_lp_exit_step() {
        let t = lp_exit_step(&[0.5, 0.5], &[1.0, -1.0], &[0.0, 0.0], &[1.0, 1.0], 1e-10);
        assert!((t - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ray_result_to_ray_length() {
        let r = RayResult {
            variable_index: 3,
            intersects: true,
            alpha: 2.5,
            boundary_point: Some(vec![1.0, 2.0]),
            transitions: vec![],
            boundary_gap: 0.0,
            bisection_iters: 10,
            exits_lp_feasible: false,
            lp_exit_t: f64::INFINITY,
        };
        let rl = r.to_ray_length();
        assert!(rl.intersects);
        assert!((rl.alpha - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_detect_transitions() {
        let tracer = RayTracer::new(RayTracerConfig::default(), 2, 0, vec![]);
        let transitions =
            tracer.detect_transitions(&[0.0, 0.0], &[1.0, 0.0], 1.0, 10, &|_x: &[f64]| Some(0));
        assert!(transitions.is_empty()); // No transitions when region is constant
    }

    #[test]
    fn test_check_lp_feasibility() {
        let tracer = RayTracer::new(RayTracerConfig::default(), 1, 1, vec![1.0]);
        assert!(tracer.check_lp_feasibility(&[0.5, 0.5], &[0.0, 0.0], &[1.0, 1.0]));
        assert!(!tracer.check_lp_feasibility(&[1.5, 0.5], &[0.0, 0.0], &[1.0, 1.0]));
    }

    #[test]
    fn test_refine_transition() {
        let tracer = RayTracer::new(RayTracerConfig::default(), 1, 0, vec![]);
        let t = tracer.refine_transition(&[0.0], &[1.0], 0.0, 1.0, &|x: &[f64]| {
            if x[0] < 0.5 {
                Some(0)
            } else {
                Some(1)
            }
        });
        assert!((t - 0.5).abs() < 0.01);
    }
}
