//! Balas intersection cut formula implementation.
//!
//! Given ray lengths from an LP vertex to the boundary of the bilevel-infeasible
//! set along each simplex direction, compute cut coefficients via Balas's 1971
//! formula:  sum_j (1/alpha_j) * (x_j - l_j) >= 1  for nonbasic variables at
//! their lower bounds, and similarly for upper-bound directions.

use crate::{BilevelCut, CutError, CutResult, TOLERANCE};
use bicut_types::ConstraintSense;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the Balas formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalasConfig {
    pub min_ray_length: f64,
    pub max_ray_length: f64,
    pub tolerance: f64,
    pub normalize: bool,
    pub round_coefficients: bool,
    pub rounding_digits: u32,
    pub handle_infinite_rays: bool,
    pub min_finite_rays: usize,
}

impl Default for BalasConfig {
    fn default() -> Self {
        Self {
            min_ray_length: 1e-10,
            max_ray_length: 1e12,
            tolerance: TOLERANCE,
            normalize: true,
            round_coefficients: false,
            rounding_digits: 10,
            handle_infinite_rays: true,
            min_finite_rays: 1,
        }
    }
}

/// Ray length from LP vertex to bilevel-feasible boundary along a simplex direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayLength {
    pub variable_index: usize,
    pub alpha: f64,
    pub direction: f64,
    pub at_lower_bound: bool,
    pub intersects: bool,
    pub intersection_point: Option<Vec<f64>>,
}

impl RayLength {
    pub fn finite(variable_index: usize, alpha: f64, at_lower_bound: bool) -> Self {
        Self {
            variable_index,
            alpha,
            direction: if at_lower_bound { 1.0 } else { -1.0 },
            at_lower_bound,
            intersects: true,
            intersection_point: None,
        }
    }

    pub fn infinite(variable_index: usize, at_lower_bound: bool) -> Self {
        Self {
            variable_index,
            alpha: f64::INFINITY,
            direction: if at_lower_bound { 1.0 } else { -1.0 },
            at_lower_bound,
            intersects: false,
            intersection_point: None,
        }
    }

    pub fn is_finite(&self, max_length: f64) -> bool {
        self.intersects && self.alpha.is_finite() && self.alpha < max_length && self.alpha > 0.0
    }

    pub fn reciprocal(&self) -> f64 {
        if self.intersects && self.alpha.is_finite() && self.alpha > 1e-15 {
            1.0 / self.alpha
        } else {
            0.0
        }
    }
}

/// Computed Balas cut coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalasCoefficients {
    pub coefficients: Vec<(usize, f64)>,
    pub rhs: f64,
    pub num_finite_rays: usize,
    pub num_infinite_rays: usize,
    pub cut: Option<BilevelCut>,
    pub max_coeff: f64,
    pub min_nonzero_coeff: f64,
    pub condition_estimate: f64,
}

/// Balas intersection cut formula engine.
#[derive(Debug, Clone)]
pub struct BalasFormula {
    pub config: BalasConfig,
    cut_counter: u64,
    coeff_cache: HashMap<Vec<OrderedFloat<f64>>, BalasCoefficients>,
}

impl BalasFormula {
    pub fn new(config: BalasConfig) -> Self {
        Self {
            config,
            cut_counter: 0,
            coeff_cache: HashMap::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(BalasConfig::default())
    }

    /// Core Balas formula: given ray lengths {alpha_j} for each nonbasic direction j,
    /// intersection cut is: sum_{j at lower} (1/alpha_j)(x_j - l_j)
    ///                     + sum_{j at upper} (1/alpha_j)(u_j - x_j) >= 1
    pub fn compute(
        &mut self,
        ray_lengths: &[RayLength],
        _vertex: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> CutResult<BalasCoefficients> {
        if ray_lengths.is_empty() {
            return Err(CutError::NumericalIssue("No ray lengths provided".into()));
        }

        let mut var_coeffs: Vec<(usize, f64)> = Vec::new();
        let mut rhs = 1.0;
        let mut num_finite = 0usize;
        let mut num_infinite = 0usize;
        let mut max_coeff = 0.0f64;
        let mut min_nonzero_coeff = f64::INFINITY;

        for ray in ray_lengths {
            if !ray.is_finite(self.config.max_ray_length) {
                num_infinite += 1;
                if !self.config.handle_infinite_rays {
                    return Err(CutError::RayTracingFailed(format!(
                        "Infinite ray for variable {} and infinite ray handling disabled",
                        ray.variable_index
                    )));
                }
                continue;
            }

            let alpha = ray.alpha.max(self.config.min_ray_length);
            let inv_alpha = 1.0 / alpha;

            if inv_alpha.abs() < self.config.tolerance {
                num_infinite += 1;
                continue;
            }

            if ray.at_lower_bound {
                let lb = lower_bounds.get(ray.variable_index).copied().unwrap_or(0.0);
                var_coeffs.push((ray.variable_index, inv_alpha));
                rhs += inv_alpha * lb;
            } else {
                let ub = upper_bounds.get(ray.variable_index).copied().unwrap_or(1.0);
                var_coeffs.push((ray.variable_index, -inv_alpha));
                rhs -= inv_alpha * ub;
            }

            max_coeff = max_coeff.max(inv_alpha.abs());
            if inv_alpha.abs() > self.config.tolerance {
                min_nonzero_coeff = min_nonzero_coeff.min(inv_alpha.abs());
            }
            num_finite += 1;
        }

        if num_finite < self.config.min_finite_rays {
            return Err(CutError::NumericalIssue(format!(
                "Only {} finite rays, need at least {}",
                num_finite, self.config.min_finite_rays
            )));
        }

        if self.config.round_coefficients {
            let factor = 10.0f64.powi(self.config.rounding_digits as i32);
            for (_, c) in &mut var_coeffs {
                *c = (*c * factor).round() / factor;
            }
            rhs = (rhs * factor).round() / factor;
        }

        if min_nonzero_coeff == f64::INFINITY {
            min_nonzero_coeff = 0.0;
        }

        let condition_estimate = if min_nonzero_coeff > self.config.tolerance {
            max_coeff / min_nonzero_coeff
        } else {
            f64::INFINITY
        };

        let mut cut = BilevelCut::new(var_coeffs.clone(), rhs, ConstraintSense::Ge)
            .with_name(format!("balas_{}", self.cut_counter));
        cut.rank = 1;
        if self.config.normalize {
            cut.normalize();
        }
        self.cut_counter += 1;

        Ok(BalasCoefficients {
            coefficients: var_coeffs,
            rhs,
            num_finite_rays: num_finite,
            num_infinite_rays: num_infinite,
            cut: Some(cut),
            max_coeff,
            min_nonzero_coeff,
            condition_estimate,
        })
    }

    /// Strengthened Balas cut using integrality.
    pub fn compute_strengthened(
        &mut self,
        ray_lengths: &[RayLength],
        vertex: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        integer_vars: &[usize],
    ) -> CutResult<BalasCoefficients> {
        let mut result = self.compute(ray_lengths, vertex, lower_bounds, upper_bounds)?;
        let int_set: std::collections::HashSet<usize> = integer_vars.iter().copied().collect();

        for (var_idx, coeff) in &mut result.coefficients {
            if int_set.contains(var_idx) && coeff.abs() > self.config.tolerance {
                let sign = coeff.signum();
                let abs_val = coeff.abs();
                let inv = 1.0 / abs_val;
                let strengthened_inv = inv.ceil();
                if strengthened_inv > 0.0 {
                    *coeff = sign / strengthened_inv;
                }
            }
        }

        if let Some(ref mut cut) = result.cut {
            cut.coeffs = result.coefficients.clone();
            cut.name = format!("balas_str_{}", self.cut_counter - 1);
            if self.config.normalize {
                cut.normalize();
            }
        }
        Ok(result)
    }

    /// Monoidal strengthening for integer variables.
    pub fn monoidal_strengthening(
        &self,
        base: &BalasCoefficients,
        integer_vars: &[usize],
        fractional_parts: &HashMap<usize, f64>,
    ) -> CutResult<BalasCoefficients> {
        let mut result = base.clone();
        let int_set: std::collections::HashSet<usize> = integer_vars.iter().copied().collect();
        let f_0 = 0.5;
        for (var_idx, coeff) in &mut result.coefficients {
            if !int_set.contains(var_idx) {
                continue;
            }
            let frac = fractional_parts.get(var_idx).copied().unwrap_or(0.0);
            if frac < self.config.tolerance || frac > 1.0 - self.config.tolerance {
                continue;
            }
            let strengthened = coeff.abs() * (f_0 / frac).min(1.0);
            *coeff = coeff.signum() * strengthened;
        }
        if let Some(ref mut cut) = result.cut {
            cut.coeffs = result.coefficients.clone();
        }
        Ok(result)
    }

    /// Validate that the cut separates the vertex.
    pub fn validate_cut(&self, coefficients: &BalasCoefficients, vertex: &[f64]) -> bool {
        if let Some(ref cut) = coefficients.cut {
            cut.is_violated(vertex, self.config.tolerance)
        } else {
            let lhs: f64 = coefficients
                .coefficients
                .iter()
                .map(|&(j, c)| c * vertex.get(j).copied().unwrap_or(0.0))
                .sum();
            lhs < coefficients.rhs - self.config.tolerance
        }
    }

    /// Condition number of the cut.
    pub fn conditioning(&self, c: &BalasCoefficients) -> f64 {
        c.condition_estimate
    }

    /// Rescale so maximum coefficient is 1.
    pub fn rescale_cut(&self, c: &mut BalasCoefficients) {
        if c.max_coeff > self.config.tolerance {
            let scale = 1.0 / c.max_coeff;
            for (_, v) in &mut c.coefficients {
                *v *= scale;
            }
            c.rhs *= scale;
            c.min_nonzero_coeff *= scale;
            c.max_coeff = 1.0;
            if let Some(ref mut cut) = c.cut {
                for (_, v) in &mut cut.coeffs {
                    *v *= scale;
                }
                cut.rhs *= scale;
            }
        }
    }

    /// Merge two Balas cuts via max-back procedure.
    pub fn merge_cuts(&self, a: &BalasCoefficients, b: &BalasCoefficients) -> BalasCoefficients {
        let mut merged: HashMap<usize, f64> = HashMap::new();
        for &(idx, coeff) in &a.coefficients {
            merged.insert(idx, coeff);
        }
        for &(idx, coeff) in &b.coefficients {
            let entry = merged.entry(idx).or_insert(0.0);
            if coeff.abs() > entry.abs() {
                *entry = coeff;
            }
        }
        let coefficients: Vec<(usize, f64)> = {
            let mut v: Vec<_> = merged.into_iter().collect();
            v.sort_by_key(|&(i, _)| i);
            v
        };
        let max_coeff = coefficients
            .iter()
            .map(|&(_, c)| c.abs())
            .fold(0.0f64, f64::max);
        let min_nz = coefficients
            .iter()
            .filter(|&&(_, c)| c.abs() > self.config.tolerance)
            .map(|&(_, c)| c.abs())
            .fold(f64::INFINITY, f64::min);
        let min_nonzero_coeff = if min_nz == f64::INFINITY { 0.0 } else { min_nz };
        let rhs = a.rhs.max(b.rhs);
        let cut = BilevelCut::new(coefficients.clone(), rhs, ConstraintSense::Ge)
            .with_name("balas_merged".to_string());
        BalasCoefficients {
            coefficients,
            rhs,
            num_finite_rays: a.num_finite_rays.max(b.num_finite_rays),
            num_infinite_rays: a.num_infinite_rays.min(b.num_infinite_rays),
            cut: Some(cut),
            max_coeff,
            min_nonzero_coeff,
            condition_estimate: if min_nonzero_coeff > TOLERANCE {
                max_coeff / min_nonzero_coeff
            } else {
                f64::INFINITY
            },
        }
    }

    pub fn cuts_generated(&self) -> u64 {
        self.cut_counter
    }
    pub fn clear_cache(&mut self) {
        self.coeff_cache.clear();
    }

    pub fn estimated_strength(&self, c: &BalasCoefficients) -> f64 {
        let total = c.num_finite_rays + c.num_infinite_rays;
        if total == 0 {
            0.0
        } else {
            c.num_finite_rays as f64 / total as f64
        }
    }
}

pub fn fractional_part(x: f64) -> f64 {
    x - x.floor()
}

pub fn is_integer(x: f64, tol: f64) -> bool {
    fractional_part(x) < tol || fractional_part(x) > 1.0 - tol
}

pub fn displacement_form_cut(ray_lengths: &[RayLength], tolerance: f64) -> Vec<(usize, f64)> {
    ray_lengths
        .iter()
        .filter(|r| r.intersects && r.alpha.is_finite() && r.alpha > tolerance)
        .map(|r| (r.variable_index, 1.0 / r.alpha))
        .collect()
}

pub fn displacement_to_variable_space(
    disp: &[(usize, f64)],
    ray_lengths: &[RayLength],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
) -> (Vec<(usize, f64)>, f64) {
    let ray_map: HashMap<usize, &RayLength> =
        ray_lengths.iter().map(|r| (r.variable_index, r)).collect();
    let mut var_coeffs = Vec::new();
    let mut rhs_adj = 0.0;
    for &(var_idx, inv_alpha) in disp {
        if let Some(ray) = ray_map.get(&var_idx) {
            if ray.at_lower_bound {
                let lb = lower_bounds.get(var_idx).copied().unwrap_or(0.0);
                var_coeffs.push((var_idx, inv_alpha));
                rhs_adj += inv_alpha * lb;
            } else {
                let ub = upper_bounds.get(var_idx).copied().unwrap_or(1.0);
                var_coeffs.push((var_idx, -inv_alpha));
                rhs_adj -= inv_alpha * ub;
            }
        } else {
            var_coeffs.push((var_idx, inv_alpha));
        }
    }
    (var_coeffs, 1.0 + rhs_adj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_finite_rays(n: usize, base_alpha: f64) -> Vec<RayLength> {
        (0..n)
            .map(|i| RayLength::finite(i, base_alpha * (i as f64 + 1.0), true))
            .collect()
    }

    #[test]
    fn test_balas_config_default() {
        let cfg = BalasConfig::default();
        assert!(cfg.min_ray_length > 0.0);
        assert!(cfg.normalize);
    }

    #[test]
    fn test_ray_length_finite() {
        let r = RayLength::finite(0, 2.5, true);
        assert!(r.intersects);
        assert!(r.is_finite(1e12));
        assert!((r.reciprocal() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_ray_length_infinite() {
        let r = RayLength::infinite(1, false);
        assert!(!r.intersects);
        assert!(!r.is_finite(1e12));
    }

    #[test]
    fn test_balas_formula_basic() {
        let mut f = BalasFormula::with_default_config();
        let rays = vec![
            RayLength::finite(0, 2.0, true),
            RayLength::finite(1, 4.0, true),
        ];
        let r = f
            .compute(&rays, &[0.5, 0.5], &[0.0, 0.0], &[1.0, 1.0])
            .unwrap();
        assert_eq!(r.num_finite_rays, 2);
    }

    #[test]
    fn test_balas_with_infinite_ray() {
        let mut f = BalasFormula::with_default_config();
        let rays = vec![
            RayLength::finite(0, 3.0, true),
            RayLength::infinite(1, true),
        ];
        let r = f
            .compute(&rays, &[0.5, 0.5], &[0.0, 0.0], &[1.0, 1.0])
            .unwrap();
        assert_eq!(r.num_infinite_rays, 1);
    }

    #[test]
    fn test_balas_empty_rays_error() {
        let mut f = BalasFormula::with_default_config();
        assert!(f.compute(&[], &[], &[], &[]).is_err());
    }

    #[test]
    fn test_balas_strengthened() {
        let mut f = BalasFormula::with_default_config();
        let rays = make_finite_rays(3, 1.0);
        assert!(f
            .compute_strengthened(&rays, &[0.5; 3], &[0.0; 3], &[1.0; 3], &[0, 2])
            .is_ok());
    }

    #[test]
    fn test_displacement_form() {
        let rays = vec![
            RayLength::finite(0, 2.0, true),
            RayLength::finite(1, 3.0, true),
        ];
        let c = displacement_form_cut(&rays, 1e-10);
        assert_eq!(c.len(), 2);
        assert!((c[0].1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_merge_cuts() {
        let f = BalasFormula::with_default_config();
        let a = BalasCoefficients {
            coefficients: vec![(0, 0.5), (1, 0.3)],
            rhs: 1.0,
            num_finite_rays: 2,
            num_infinite_rays: 0,
            cut: None,
            max_coeff: 0.5,
            min_nonzero_coeff: 0.3,
            condition_estimate: 0.5 / 0.3,
        };
        let b = BalasCoefficients {
            coefficients: vec![(0, 0.4), (2, 0.7)],
            rhs: 1.0,
            num_finite_rays: 2,
            num_infinite_rays: 0,
            cut: None,
            max_coeff: 0.7,
            min_nonzero_coeff: 0.4,
            condition_estimate: 0.7 / 0.4,
        };
        assert!(f.merge_cuts(&a, &b).coefficients.len() >= 2);
    }

    #[test]
    fn test_fractional_part_fn() {
        assert!((fractional_part(2.7) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_is_integer_fn() {
        assert!(is_integer(3.0, 1e-8));
        assert!(!is_integer(3.5, 1e-8));
    }
}
