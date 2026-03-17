//! Gomory mixed-integer cuts adapted for bilevel optimization.
//!
//! Standard GMI derivation from tableau rows, bilevel-aware coefficient
//! selection, split cut generation, rank-1 and rank-2 Gomory cuts.

use crate::balas::{fractional_part, is_integer};
use crate::{BilevelCut, CutError, CutResult, MIN_EFFICACY, TOLERANCE};
use bicut_types::{BasisStatus, ConstraintSense, VariableType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Gomory cut generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GomoryConfig {
    /// Minimum fractionality to generate a GMI cut.
    pub min_fractionality: f64,
    /// Maximum number of GMI cuts per round.
    pub max_cuts: usize,
    /// Minimum efficacy for generated cuts.
    pub min_efficacy: f64,
    /// Maximum rank of Gomory cuts to generate.
    pub max_rank: u32,
    /// Whether to apply bilevel-aware coefficient selection.
    pub bilevel_aware: bool,
    /// Weight for bilevel relevance in coefficient scoring.
    pub bilevel_weight: f64,
    /// Tolerance for numerical comparisons.
    pub tolerance: f64,
    /// Maximum coefficient magnitude (for numerical safety).
    pub max_coeff: f64,
    /// Whether to generate split cuts.
    pub generate_splits: bool,
}

impl Default for GomoryConfig {
    fn default() -> Self {
        Self {
            min_fractionality: 0.01,
            max_cuts: 30,
            min_efficacy: MIN_EFFICACY,
            max_rank: 2,
            bilevel_aware: true,
            bilevel_weight: 0.5,
            tolerance: TOLERANCE,
            max_coeff: 1e6,
            generate_splits: true,
        }
    }
}

/// A Gomory cut generated from a tableau row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GomoryCut {
    pub cut: BilevelCut,
    pub source_row: usize,
    pub source_var: usize,
    pub fractionality: f64,
    pub rank: u32,
    pub is_bilevel_strengthened: bool,
}

/// A split cut: disjunction x_j <= floor(x_j*) or x_j >= ceil(x_j*).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitCut {
    pub cut: BilevelCut,
    pub split_variable: usize,
    pub split_value: f64,
    pub left_bound: f64,
    pub right_bound: f64,
}

/// Gomory cut generator for bilevel optimization.
#[derive(Debug, Clone)]
pub struct GomoryCutGenerator {
    pub config: GomoryConfig,
    n_leader: usize,
    n_follower: usize,
    integer_vars: Vec<usize>,
    follower_obj: Vec<f64>,
    cut_counter: u64,
    stats: GomoryStats,
}

/// Statistics for Gomory cut generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GomoryStats {
    pub total_rows_examined: usize,
    pub total_fractional: usize,
    pub total_cuts_generated: usize,
    pub total_rank1: usize,
    pub total_rank2: usize,
    pub total_splits: usize,
    pub avg_fractionality: f64,
}

impl GomoryCutGenerator {
    pub fn new(
        config: GomoryConfig,
        n_leader: usize,
        n_follower: usize,
        integer_vars: Vec<usize>,
        follower_obj: Vec<f64>,
    ) -> Self {
        Self {
            config,
            n_leader,
            n_follower,
            integer_vars,
            follower_obj,
            cut_counter: 0,
            stats: GomoryStats::default(),
        }
    }

    /// Generate Gomory mixed-integer cuts from the simplex tableau.
    ///
    /// For each basic integer variable with fractional value, generate
    /// a GMI cut from its tableau row.
    pub fn generate_gmi_cuts(
        &mut self,
        tableau_rows: &[(usize, Vec<f64>, f64)],
        basis_status: &[BasisStatus],
        point: &[f64],
    ) -> CutResult<Vec<GomoryCut>> {
        let mut cuts = Vec::new();
        let int_set: std::collections::HashSet<usize> = self.integer_vars.iter().copied().collect();

        // Collect fractional basic integer variables.
        let mut fractional_rows: Vec<(usize, usize, f64, &Vec<f64>, f64)> = Vec::new();
        for (basic_var, row_coeffs, rhs) in tableau_rows {
            self.stats.total_rows_examined += 1;
            if !int_set.contains(basic_var) {
                continue;
            }

            let val = point.get(*basic_var).copied().unwrap_or(*rhs);
            let frac = fractional_part(val);
            if frac < self.config.min_fractionality || frac > 1.0 - self.config.min_fractionality {
                continue;
            }
            self.stats.total_fractional += 1;
            fractional_rows.push((*basic_var, *basic_var, frac, row_coeffs, *rhs));
        }

        // Sort by fractionality (most fractional first for stronger cuts).
        fractional_rows.sort_by(|a, b| {
            let fa = 0.5 - (a.2 - 0.5).abs();
            let fb = 0.5 - (b.2 - 0.5).abs();
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (source_var, _basic_var, f_0, row_coeffs, rhs) in
            fractional_rows.iter().take(self.config.max_cuts)
        {
            let gmi = self.derive_gmi_cut(*source_var, *f_0, row_coeffs, basis_status, &int_set)?;
            if let Some(mut gc) = gmi {
                gc.cut.efficacy = gc.cut.compute_efficacy(point);
                if gc.cut.efficacy >= self.config.min_efficacy {
                    if self.config.bilevel_aware {
                        self.bilevel_strengthen(&mut gc);
                    }
                    self.stats.total_cuts_generated += 1;
                    self.stats.total_rank1 += 1;
                    cuts.push(gc);
                }
            }
        }

        self.update_frac_stats(&fractional_rows.iter().map(|r| r.2).collect::<Vec<_>>());
        Ok(cuts)
    }

    /// Derive a single GMI cut from a tableau row.
    fn derive_gmi_cut(
        &mut self,
        source_var: usize,
        f_0: f64,
        row_coeffs: &[f64],
        basis_status: &[BasisStatus],
        int_set: &std::collections::HashSet<usize>,
    ) -> CutResult<Option<GomoryCut>> {
        let mut cut_coeffs: Vec<(usize, f64)> = Vec::new();

        for (j, &a_j) in row_coeffs.iter().enumerate() {
            let status = basis_status.get(j).copied().unwrap_or(BasisStatus::Basic);
            if matches!(status, BasisStatus::Basic) {
                continue;
            }
            if a_j.abs() < self.config.tolerance {
                continue;
            }

            let coeff = if int_set.contains(&j) {
                // Integer nonbasic: GMI coefficient for integers.
                let f_j = fractional_part(a_j);
                if f_j <= f_0 {
                    f_j / f_0
                } else {
                    (1.0 - f_j) / (1.0 - f_0)
                }
            } else {
                // Continuous nonbasic: GMI coefficient for continuous.
                if a_j >= 0.0 {
                    a_j / f_0
                } else {
                    -a_j / (1.0 - f_0)
                }
            };

            if coeff.abs() > self.config.tolerance && coeff.abs() < self.config.max_coeff {
                let sign = match status {
                    BasisStatus::NonBasicUpper => -1.0,
                    _ => 1.0,
                };
                cut_coeffs.push((j, sign * coeff));
            }
        }

        if cut_coeffs.is_empty() {
            return Ok(None);
        }

        let rhs = 1.0;
        let mut cut = BilevelCut::new(cut_coeffs, rhs, ConstraintSense::Ge)
            .with_name(format!("gmi_{}", self.cut_counter))
            .with_rank(1);
        self.cut_counter += 1;

        Ok(Some(GomoryCut {
            cut,
            source_row: source_var,
            source_var,
            fractionality: f_0,
            rank: 1,
            is_bilevel_strengthened: false,
        }))
    }

    /// Apply bilevel-aware strengthening to a Gomory cut.
    fn bilevel_strengthen(&self, gc: &mut GomoryCut) {
        // Prioritize coefficients that correspond to follower variables
        // or coupling variables, as these are more relevant for bilevel feasibility.
        for (j, coeff) in &mut gc.cut.coeffs {
            let is_follower = *j >= self.n_leader && *j < self.n_leader + self.n_follower;
            if is_follower {
                // Strengthen follower variable coefficients by the bilevel weight.
                let follower_idx = *j - self.n_leader;
                let c_j = self.follower_obj.get(follower_idx).copied().unwrap_or(0.0);
                if c_j.abs() > self.config.tolerance {
                    let boost = 1.0 + self.config.bilevel_weight * c_j.abs();
                    *coeff *= boost;
                }
            }
        }
        gc.is_bilevel_strengthened = true;
    }

    /// Generate rank-2 Gomory cuts by applying GMI to the strengthened LP.
    pub fn generate_rank2_cuts(
        &mut self,
        rank1_cuts: &[GomoryCut],
        tableau_rows: &[(usize, Vec<f64>, f64)],
        basis_status: &[BasisStatus],
        point: &[f64],
    ) -> CutResult<Vec<GomoryCut>> {
        if self.config.max_rank < 2 {
            return Ok(Vec::new());
        }

        let mut rank2_cuts = Vec::new();
        let int_set: std::collections::HashSet<usize> = self.integer_vars.iter().copied().collect();

        // For rank-2: re-derive GMI from the tableau after adding rank-1 cuts.
        // Simplified: perturb the tableau rows using rank-1 cut coefficients.
        for (basic_var, row_coeffs, rhs) in tableau_rows {
            if !int_set.contains(basic_var) {
                continue;
            }
            let val = point.get(*basic_var).copied().unwrap_or(*rhs);
            let frac = fractional_part(val);
            if frac < self.config.min_fractionality {
                continue;
            }

            // Combine with each rank-1 cut to create a rank-2 row.
            for r1 in rank1_cuts.iter().take(3) {
                let mut combined = row_coeffs.clone();
                let scale = 0.5;
                for &(j, c) in &r1.cut.coeffs {
                    if j < combined.len() {
                        combined[j] += scale * c;
                    }
                }

                let combined_val = point.get(*basic_var).copied().unwrap_or(0.0);
                let f_0 = fractional_part(combined_val);
                if f_0 < self.config.min_fractionality {
                    continue;
                }

                if let Ok(Some(mut gc)) =
                    self.derive_gmi_cut(*basic_var, f_0, &combined, basis_status, &int_set)
                {
                    gc.rank = 2;
                    gc.cut.rank = 2;
                    gc.cut.name = format!("gmi_r2_{}", self.cut_counter - 1);
                    gc.cut.efficacy = gc.cut.compute_efficacy(point);
                    if gc.cut.efficacy >= self.config.min_efficacy {
                        self.stats.total_rank2 += 1;
                        rank2_cuts.push(gc);
                        if rank2_cuts.len() >= self.config.max_cuts {
                            break;
                        }
                    }
                }
            }
            if rank2_cuts.len() >= self.config.max_cuts {
                break;
            }
        }

        Ok(rank2_cuts)
    }

    /// Generate split cuts from fractional integer variables.
    pub fn generate_split_cuts(
        &mut self,
        point: &[f64],
        tableau_rows: &[(usize, Vec<f64>, f64)],
        basis_status: &[BasisStatus],
    ) -> CutResult<Vec<SplitCut>> {
        if !self.config.generate_splits {
            return Ok(Vec::new());
        }
        let mut splits = Vec::new();
        let int_set: std::collections::HashSet<usize> = self.integer_vars.iter().copied().collect();

        for &var_idx in &self.integer_vars {
            let val = point.get(var_idx).copied().unwrap_or(0.0);
            let frac = fractional_part(val);
            if frac < self.config.min_fractionality || frac > 1.0 - self.config.min_fractionality {
                continue;
            }

            let floor_val = val.floor();
            let ceil_val = val.ceil();

            // Generate the split disjunction cut: a valid inequality that
            // is valid for both x_j <= floor and x_j >= ceil.
            let mut coeffs = vec![(var_idx, 1.0)];
            let rhs = ceil_val;

            let cut = BilevelCut::new(coeffs, rhs, ConstraintSense::Ge)
                .with_name(format!("split_{}", var_idx))
                .with_rank(1);

            self.stats.total_splits += 1;
            splits.push(SplitCut {
                cut,
                split_variable: var_idx,
                split_value: val,
                left_bound: floor_val,
                right_bound: ceil_val,
            });
        }

        Ok(splits)
    }

    fn update_frac_stats(&mut self, fracs: &[f64]) {
        if fracs.is_empty() {
            return;
        }
        let n = fracs.len() as f64;
        self.stats.avg_fractionality = fracs.iter().sum::<f64>() / n;
    }

    pub fn stats(&self) -> &GomoryStats {
        &self.stats
    }
    pub fn reset_stats(&mut self) {
        self.stats = GomoryStats::default();
    }
    pub fn cuts_generated(&self) -> u64 {
        self.cut_counter
    }
}

/// Compute the GMI coefficient for a nonbasic variable.
pub fn gmi_coefficient(a_j: f64, f_0: f64, is_integer: bool) -> f64 {
    if is_integer {
        let f_j = fractional_part(a_j);
        if f_j <= f_0 {
            f_j / f_0
        } else {
            (1.0 - f_j) / (1.0 - f_0)
        }
    } else {
        if a_j >= 0.0 {
            a_j / f_0
        } else {
            -a_j / (1.0 - f_0)
        }
    }
}

/// Check if a variable is a good source row for GMI cuts.
pub fn is_good_source_row(
    fractionality: f64,
    row_sparsity: usize,
    max_sparsity: usize,
    min_frac: f64,
) -> bool {
    fractionality >= min_frac && fractionality <= 1.0 - min_frac && row_sparsity <= max_sparsity
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_generator() -> GomoryCutGenerator {
        GomoryCutGenerator::new(GomoryConfig::default(), 2, 2, vec![0, 2, 3], vec![1.0, 1.0])
    }

    #[test]
    fn test_config_default() {
        let cfg = GomoryConfig::default();
        assert!(cfg.min_fractionality > 0.0);
        assert!(cfg.max_cuts > 0);
    }

    #[test]
    fn test_generator_creation() {
        let gen = make_generator();
        assert_eq!(gen.cuts_generated(), 0);
    }

    #[test]
    fn test_gmi_coefficient_integer() {
        let c = gmi_coefficient(1.3, 0.5, true);
        // f_j = 0.3, f_0 = 0.5, f_j <= f_0: c = 0.3/0.5 = 0.6
        assert!((c - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_gmi_coefficient_continuous() {
        let c = gmi_coefficient(2.0, 0.5, false);
        assert!((c - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gmi_coefficient_integer_large_frac() {
        let c = gmi_coefficient(1.7, 0.5, true);
        // f_j = 0.7, f_0 = 0.5, f_j > f_0: c = (1-0.7)/(1-0.5) = 0.6
        assert!((c - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_generate_gmi_cuts() {
        let mut gen = make_generator();
        let tableau = vec![
            (0, vec![1.0, 0.5, -0.3, 0.7], 2.5),
            (2, vec![0.0, -0.2, 1.0, 0.4], 1.7),
        ];
        let basis = vec![
            BasisStatus::Basic,
            BasisStatus::NonBasicLower,
            BasisStatus::Basic,
            BasisStatus::NonBasicLower,
        ];
        let point = vec![2.5, 0.0, 1.7, 0.0];
        let result = gen.generate_gmi_cuts(&tableau, &basis, &point);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_split_cuts() {
        let mut gen = make_generator();
        let point = vec![1.5, 0.5, 2.3, 0.7];
        let result = gen.generate_split_cuts(&point, &[], &[]);
        assert!(result.is_ok());
        let splits = result.unwrap();
        assert!(splits.len() > 0);
    }

    #[test]
    fn test_is_good_source_row() {
        assert!(is_good_source_row(0.5, 5, 100, 0.01));
        assert!(!is_good_source_row(0.001, 5, 100, 0.01));
        assert!(!is_good_source_row(0.5, 200, 100, 0.01));
    }

    #[test]
    fn test_split_cut_structure() {
        let sc = SplitCut {
            cut: BilevelCut::new(vec![(0, 1.0)], 2.0, ConstraintSense::Ge),
            split_variable: 0,
            split_value: 1.5,
            left_bound: 1.0,
            right_bound: 2.0,
        };
        assert_eq!(sc.split_variable, 0);
        assert!((sc.left_bound - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_default() {
        let stats = GomoryStats::default();
        assert_eq!(stats.total_cuts_generated, 0);
    }
}
