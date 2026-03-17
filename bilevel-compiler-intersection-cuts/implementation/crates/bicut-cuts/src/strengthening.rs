//! Cut strengthening procedures for bilevel intersection cuts.
//!
//! Implements coefficient tightening via integrality, MIR strengthening for
//! mixed-integer bilevel problems, value-function lifting (Gomory-Johnson style),
//! and shift-and-project strengthening.

use crate::balas::{fractional_part, is_integer, BalasCoefficients};
use crate::{BilevelCut, CutError, CutResult, MIN_EFFICACY, TOLERANCE};
use bicut_types::{BasisStatus, ConstraintSense, VariableType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strengthening method to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrengtheningMethod {
    /// Tighten coefficients using integrality of variables.
    CoefficientTightening,
    /// Mixed-integer rounding strengthening.
    MIR,
    /// Value-function based lifting (Gomory-Johnson).
    ValueFunctionLifting,
    /// Shift-and-project strengthening.
    ShiftAndProject,
    /// Apply all methods in sequence.
    All,
}

/// Configuration for cut strengthening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrengtheningConfig {
    pub methods: Vec<StrengtheningMethod>,
    pub tolerance: f64,
    pub min_improvement: f64,
    pub max_iterations: usize,
    pub mir_complement: bool,
    pub lifting_max_terms: usize,
    pub shift_project_depth: usize,
}

impl Default for StrengtheningConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                StrengtheningMethod::CoefficientTightening,
                StrengtheningMethod::MIR,
            ],
            tolerance: TOLERANCE,
            min_improvement: 1e-6,
            max_iterations: 10,
            mir_complement: true,
            lifting_max_terms: 50,
            shift_project_depth: 1,
        }
    }
}

/// Result of strengthening a cut.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrengtheningResult {
    pub original: BilevelCut,
    pub strengthened: BilevelCut,
    pub improvement: f64,
    pub method_used: Vec<StrengtheningMethod>,
    pub coefficients_changed: usize,
    pub rhs_changed: bool,
}

/// The cut strengthener.
#[derive(Debug, Clone)]
pub struct CutStrengthener {
    pub config: StrengtheningConfig,
    integer_vars: Vec<usize>,
    var_types: Vec<VariableType>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    stats: StrengtheningStats,
}

/// Statistics for strengthening.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrengtheningStats {
    pub total_attempts: usize,
    pub total_improved: usize,
    pub total_unchanged: usize,
    pub avg_improvement: f64,
    pub max_improvement: f64,
    pub method_counts: HashMap<String, usize>,
}

impl CutStrengthener {
    pub fn new(
        config: StrengtheningConfig,
        var_types: Vec<VariableType>,
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
    ) -> Self {
        let integer_vars: Vec<usize> = var_types
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, VariableType::Integer | VariableType::Binary))
            .map(|(i, _)| i)
            .collect();
        Self {
            config,
            integer_vars,
            var_types,
            lower_bounds,
            upper_bounds,
            stats: StrengtheningStats::default(),
        }
    }

    /// Strengthen a cut using the configured methods.
    pub fn strengthen(
        &mut self,
        cut: &BilevelCut,
        point: &[f64],
    ) -> CutResult<StrengtheningResult> {
        self.stats.total_attempts += 1;
        let original = cut.clone();
        let mut current = cut.clone();
        let mut methods_used = Vec::new();
        let mut total_coeffs_changed = 0;
        let mut rhs_changed = false;

        for method in &self.config.methods.clone() {
            let before_violation = current.violation(point);
            match method {
                StrengtheningMethod::CoefficientTightening => {
                    let (new_cut, changed) = self.coefficient_tightening(&current)?;
                    current = new_cut;
                    total_coeffs_changed += changed;
                    if changed > 0 {
                        methods_used.push(*method);
                    }
                }
                StrengtheningMethod::MIR => {
                    let result = self.mir_strengthening(&current, point)?;
                    if result.rhs != current.rhs {
                        rhs_changed = true;
                    }
                    current = result;
                    methods_used.push(*method);
                }
                StrengtheningMethod::ValueFunctionLifting => {
                    let result = self.value_function_lifting(&current, point)?;
                    total_coeffs_changed += result.1;
                    current = result.0;
                    if result.1 > 0 {
                        methods_used.push(*method);
                    }
                }
                StrengtheningMethod::ShiftAndProject => {
                    let result = self.shift_and_project(&current, point)?;
                    current = result;
                    methods_used.push(*method);
                }
                StrengtheningMethod::All => {
                    let (c1, ch1) = self.coefficient_tightening(&current)?;
                    current = c1;
                    total_coeffs_changed += ch1;
                    let c2 = self.mir_strengthening(&current, point)?;
                    if c2.rhs != current.rhs {
                        rhs_changed = true;
                    }
                    current = c2;
                    let (c3, ch3) = self.value_function_lifting(&current, point)?;
                    current = c3;
                    total_coeffs_changed += ch3;
                    let c4 = self.shift_and_project(&current, point)?;
                    current = c4;
                    methods_used.push(*method);
                }
            }
        }

        let improvement = current.violation(point) - original.violation(point);
        if improvement > self.config.min_improvement {
            self.stats.total_improved += 1;
            self.update_improvement_stats(improvement);
        } else {
            self.stats.total_unchanged += 1;
        }

        Ok(StrengtheningResult {
            original,
            strengthened: current,
            improvement,
            method_used: methods_used,
            coefficients_changed: total_coeffs_changed,
            rhs_changed,
        })
    }

    /// Coefficient tightening: for integer variables, round coefficients
    /// towards zero to make the cut tighter.
    fn coefficient_tightening(&self, cut: &BilevelCut) -> CutResult<(BilevelCut, usize)> {
        let mut new_coeffs = cut.coeffs.clone();
        let mut new_rhs = cut.rhs;
        let mut changed = 0;

        for (j, coeff) in &mut new_coeffs {
            if !self.integer_vars.contains(j) {
                continue;
            }
            let lb = self.lower_bounds.get(*j).copied().unwrap_or(0.0);
            let ub = self.upper_bounds.get(*j).copied().unwrap_or(f64::INFINITY);

            match cut.sense {
                ConstraintSense::Ge => {
                    // For >= constraint: can increase positive coefficients or
                    // decrease negative coefficients to tighten.
                    if *coeff > self.config.tolerance {
                        let max_increase = if ub < f64::INFINITY {
                            (new_rhs - cut.rhs) / (ub - lb).max(1.0)
                        } else {
                            0.0
                        };
                        let rounded = coeff.ceil();
                        if rounded > *coeff
                            && (rounded - *coeff) <= max_increase.abs() + self.config.tolerance
                        {
                            *coeff = rounded;
                            changed += 1;
                        }
                    } else if *coeff < -self.config.tolerance {
                        let floored = coeff.floor();
                        if floored < *coeff {
                            new_rhs += (floored - *coeff) * lb;
                            *coeff = floored;
                            changed += 1;
                        }
                    }
                }
                ConstraintSense::Le => {
                    if *coeff > self.config.tolerance {
                        let floored = coeff.floor();
                        if floored < *coeff {
                            *coeff = floored;
                            changed += 1;
                        }
                    }
                }
                ConstraintSense::Eq => {}
            }
        }

        let mut result = BilevelCut::new(new_coeffs, new_rhs, cut.sense);
        result.name = format!("{}_tight", cut.name);
        result.rank = cut.rank;
        Ok((result, changed))
    }

    /// MIR strengthening: apply the mixed-integer rounding procedure.
    fn mir_strengthening(&self, cut: &BilevelCut, _point: &[f64]) -> CutResult<BilevelCut> {
        let rhs_frac = fractional_part(cut.rhs);
        if rhs_frac < self.config.tolerance || rhs_frac > 1.0 - self.config.tolerance {
            return Ok(cut.clone());
        }

        let mut new_coeffs = Vec::new();
        let mut new_rhs = cut.rhs.floor();

        for &(j, coeff) in &cut.coeffs {
            if self.integer_vars.contains(&j) {
                // Integer variable: apply MIR formula.
                let f_j = fractional_part(coeff);
                let mir_coeff = if f_j <= rhs_frac {
                    coeff.floor()
                } else {
                    coeff.floor() + (f_j - rhs_frac) / (1.0 - rhs_frac)
                };
                if mir_coeff.abs() > self.config.tolerance {
                    new_coeffs.push((j, mir_coeff));
                }
            } else {
                // Continuous variable: apply MIR continuous rule.
                let mir_coeff = if coeff >= 0.0 {
                    coeff / rhs_frac
                } else if self.config.mir_complement {
                    coeff / (1.0 - rhs_frac)
                } else {
                    coeff
                };
                if mir_coeff.abs() > self.config.tolerance {
                    new_coeffs.push((j, mir_coeff));
                }
            }
        }

        new_rhs = new_rhs + 1.0;

        let mut result = BilevelCut::new(new_coeffs, new_rhs, cut.sense);
        result.name = format!("{}_mir", cut.name);
        result.rank = cut.rank;
        Ok(result)
    }

    /// Value-function based lifting (Gomory-Johnson style).
    fn value_function_lifting(
        &self,
        cut: &BilevelCut,
        _point: &[f64],
    ) -> CutResult<(BilevelCut, usize)> {
        let mut new_coeffs = cut.coeffs.clone();
        let mut changed = 0;

        // For each integer variable, attempt to lift the coefficient
        // using the subadditive structure of the value function.
        for (j, coeff) in &mut new_coeffs {
            if !self.integer_vars.contains(j) {
                continue;
            }

            let frac = fractional_part(*coeff);
            if frac < self.config.tolerance || frac > 1.0 - self.config.tolerance {
                continue;
            }

            // Gomory-Johnson lifting: use the two-slope function.
            // For f_0 (RHS fractional), f_j (coeff fractional):
            // If f_j <= f_0: lifted coeff = f_j / f_0
            // If f_j > f_0: lifted coeff = (1 - f_j) / (1 - f_0)
            let f_0 = fractional_part(cut.rhs);
            if f_0 < self.config.tolerance || f_0 > 1.0 - self.config.tolerance {
                continue;
            }

            let lifted = if frac <= f_0 {
                frac / f_0
            } else {
                (1.0 - frac) / (1.0 - f_0)
            };

            if (lifted - *coeff).abs() > self.config.tolerance {
                *coeff = coeff.floor() + lifted;
                changed += 1;
            }

            if changed >= self.config.lifting_max_terms {
                break;
            }
        }

        let mut result = BilevelCut::new(new_coeffs, cut.rhs, cut.sense);
        result.name = format!("{}_lifted", cut.name);
        result.rank = cut.rank;
        Ok((result, changed))
    }

    /// Shift-and-project strengthening.
    fn shift_and_project(&self, cut: &BilevelCut, _point: &[f64]) -> CutResult<BilevelCut> {
        let mut new_coeffs = cut.coeffs.clone();
        let mut new_rhs = cut.rhs;

        // Shift: translate the cut to improve integer properties.
        for &(j, coeff) in &cut.coeffs {
            if !self.integer_vars.contains(&j) {
                continue;
            }
            let lb = self.lower_bounds.get(j).copied().unwrap_or(0.0);
            let ub = self.upper_bounds.get(j).copied().unwrap_or(f64::INFINITY);

            if ub - lb < 1.0 + self.config.tolerance {
                continue;
            }

            // Compute the shift that makes the coefficient integer.
            let frac = fractional_part(coeff);
            if frac > self.config.tolerance && frac < 1.0 - self.config.tolerance {
                let shift = frac.min(1.0 - frac);
                if coeff > 0.0 {
                    new_rhs += shift * lb;
                } else {
                    new_rhs -= shift * ub;
                }
            }
        }

        // Project: round coefficients for integer variables.
        for (j, coeff) in &mut new_coeffs {
            if self.integer_vars.contains(j) {
                let rounded = coeff.round();
                if (rounded - *coeff).abs() < 0.5 {
                    *coeff = rounded;
                }
            }
        }

        let mut result = BilevelCut::new(new_coeffs, new_rhs, cut.sense);
        result.name = format!("{}_sp", cut.name);
        result.rank = cut.rank;
        Ok(result)
    }

    fn update_improvement_stats(&mut self, improvement: f64) {
        let n = self.stats.total_improved as f64;
        self.stats.avg_improvement = (self.stats.avg_improvement * (n - 1.0) + improvement) / n;
        self.stats.max_improvement = self.stats.max_improvement.max(improvement);
    }

    pub fn stats(&self) -> &StrengtheningStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = StrengtheningStats::default();
    }
}

/// Strengthen a single cut coefficient using bound information.
pub fn tighten_coefficient(
    coeff: f64,
    var_lb: f64,
    var_ub: f64,
    is_integer: bool,
    sense: ConstraintSense,
) -> f64 {
    if !is_integer {
        return coeff;
    }
    match sense {
        ConstraintSense::Ge => {
            if coeff > 0.0 {
                coeff.ceil()
            } else {
                coeff.floor()
            }
        }
        ConstraintSense::Le => {
            if coeff > 0.0 {
                coeff.floor()
            } else {
                coeff.ceil()
            }
        }
        ConstraintSense::Eq => coeff.round(),
    }
}

/// Apply MIR to a single tableau row.
pub fn mir_single_row(
    coeffs: &[(usize, f64)],
    rhs: f64,
    integer_vars: &[usize],
    tolerance: f64,
) -> Option<(Vec<(usize, f64)>, f64)> {
    let f_0 = fractional_part(rhs);
    if f_0 < tolerance || f_0 > 1.0 - tolerance {
        return None;
    }
    let int_set: std::collections::HashSet<usize> = integer_vars.iter().copied().collect();
    let mut new_coeffs = Vec::new();
    for &(j, a) in coeffs {
        let new_a = if int_set.contains(&j) {
            let f_j = fractional_part(a);
            if f_j <= f_0 {
                a.floor()
            } else {
                a.floor() + (f_j - f_0) / (1.0 - f_0)
            }
        } else {
            if a >= 0.0 {
                a / f_0
            } else {
                a / (1.0 - f_0)
            }
        };
        if new_a.abs() > tolerance {
            new_coeffs.push((j, new_a));
        }
    }
    Some((new_coeffs, rhs.floor() + 1.0))
}

/// Compute the two-slope lifting function value.
pub fn two_slope_lifting(f_j: f64, f_0: f64) -> f64 {
    if f_0 < 1e-12 || f_0 > 1.0 - 1e-12 {
        return f_j;
    }
    if f_j <= f_0 {
        f_j / f_0
    } else {
        (1.0 - f_j) / (1.0 - f_0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_strengthener() -> CutStrengthener {
        CutStrengthener::new(
            StrengtheningConfig::default(),
            vec![
                VariableType::Integer,
                VariableType::Continuous,
                VariableType::Integer,
            ],
            vec![0.0, 0.0, 0.0],
            vec![10.0, 10.0, 1.0],
        )
    }

    #[test]
    fn test_config_default() {
        let cfg = StrengtheningConfig::default();
        assert!(!cfg.methods.is_empty());
        assert!(cfg.tolerance > 0.0);
    }

    #[test]
    fn test_strengthener_creation() {
        let s = make_strengthener();
        assert_eq!(s.integer_vars, vec![0, 2]);
    }

    #[test]
    fn test_coefficient_tightening() {
        let s = make_strengthener();
        let cut = BilevelCut::new(
            vec![(0, 1.3), (1, 2.5), (2, -0.7)],
            3.0,
            ConstraintSense::Ge,
        );
        let (result, changed) = s.coefficient_tightening(&cut).unwrap();
        assert!(result.coeffs.len() > 0);
        // Integer var 0 coeff 1.3 should be rounded; var 2 coeff -0.7 should be floored to -1.
        let var2_coeff = result
            .coeffs
            .iter()
            .find(|&&(j, _)| j == 2)
            .map(|&(_, c)| c);
        assert!(var2_coeff.is_some());
    }

    #[test]
    fn test_mir_strengthening() {
        let s = make_strengthener();
        let cut = BilevelCut::new(vec![(0, 1.5), (1, 2.3), (2, 0.8)], 3.7, ConstraintSense::Ge);
        let result = s.mir_strengthening(&cut, &[0.5, 0.5, 0.5]).unwrap();
        assert!(result.coeffs.len() > 0);
    }

    #[test]
    fn test_value_function_lifting() {
        let s = make_strengthener();
        let cut = BilevelCut::new(vec![(0, 1.3), (2, 0.6)], 2.4, ConstraintSense::Ge);
        let (result, changed) = s.value_function_lifting(&cut, &[0.5, 0.5, 0.5]).unwrap();
        assert!(result.coeffs.len() > 0);
    }

    #[test]
    fn test_shift_and_project() {
        let s = make_strengthener();
        let cut = BilevelCut::new(vec![(0, 1.3), (1, 2.0), (2, 0.7)], 3.0, ConstraintSense::Ge);
        let result = s.shift_and_project(&cut, &[0.5, 0.5, 0.5]).unwrap();
        let c0 = result
            .coeffs
            .iter()
            .find(|&&(j, _)| j == 0)
            .map(|&(_, c)| c)
            .unwrap_or(0.0);
        assert!((c0 - c0.round()).abs() < 0.01); // Should be rounded
    }

    #[test]
    fn test_strengthen_all() {
        let mut s = make_strengthener();
        let cut = BilevelCut::new(vec![(0, 1.3), (1, 2.5), (2, 0.7)], 3.7, ConstraintSense::Ge);
        let result = s.strengthen(&cut, &[0.5, 0.5, 0.5]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tighten_coefficient_fn() {
        assert_eq!(
            tighten_coefficient(1.3, 0.0, 10.0, true, ConstraintSense::Ge),
            2.0
        );
        assert_eq!(
            tighten_coefficient(1.3, 0.0, 10.0, false, ConstraintSense::Ge),
            1.3
        );
    }

    #[test]
    fn test_mir_single_row() {
        let result = mir_single_row(&[(0, 1.5), (1, 2.3)], 3.7, &[0], 1e-8);
        assert!(result.is_some());
    }

    #[test]
    fn test_two_slope_lifting_fn() {
        let v = two_slope_lifting(0.3, 0.5);
        assert!((v - 0.6).abs() < 1e-10);
        let v2 = two_slope_lifting(0.7, 0.5);
        assert!((v2 - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_stats_default() {
        let stats = StrengtheningStats::default();
        assert_eq!(stats.total_attempts, 0);
    }
}
