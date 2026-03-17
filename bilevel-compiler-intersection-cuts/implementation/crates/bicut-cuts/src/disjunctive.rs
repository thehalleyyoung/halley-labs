//! Disjunctive cuts for bilevel optimization.
//!
//! Lift-and-project (Balas-Ceria-Cornuejols) cuts, bilevel disjunctions from
//! follower optimality conditions, CGLP (cut generating LP) formulation.

use crate::{BilevelCut, CutError, CutResult, BIG_M, MIN_EFFICACY, TOLERANCE};
use bicut_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for disjunctive cut generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisjunctiveConfig {
    /// Maximum number of disjunctive cuts per round.
    pub max_cuts: usize,
    /// Minimum efficacy for a cut to be accepted.
    pub min_efficacy: f64,
    /// Tolerance for numerical comparisons.
    pub tolerance: f64,
    /// Normalization method for the CGLP.
    pub normalization: NormalizationMethod,
    /// Maximum size of the CGLP (number of variables).
    pub max_cglp_size: usize,
    /// Whether to use bilevel-specific disjunctions.
    pub bilevel_disjunctions: bool,
    /// Time limit for solving each CGLP (in seconds).
    pub cglp_time_limit: f64,
    /// Whether to generate lift-and-project cuts.
    pub lift_and_project: bool,
    /// Depth of lift-and-project (number of disjunctions to combine).
    pub lap_depth: usize,
}

impl Default for DisjunctiveConfig {
    fn default() -> Self {
        Self {
            max_cuts: 10,
            min_efficacy: MIN_EFFICACY,
            tolerance: TOLERANCE,
            normalization: NormalizationMethod::Standard,
            max_cglp_size: 1000,
            bilevel_disjunctions: true,
            cglp_time_limit: 10.0,
            lift_and_project: true,
            lap_depth: 1,
        }
    }
}

/// Normalization method for the CGLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Standard: sum of multipliers = 1.
    Standard,
    /// Balas-Perregaard: single variable normalization.
    BalasPerregaard,
    /// Euclidean normalization: ||alpha||_2 <= 1.
    Euclidean,
}

/// A disjunction: either x in S1 or x in S2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disjunction {
    /// Coefficients for the "left" side: ax <= b.
    pub left_coeffs: Vec<(usize, f64)>,
    pub left_rhs: f64,
    /// Coefficients for the "right" side: ax >= b (or -ax <= -b).
    pub right_coeffs: Vec<(usize, f64)>,
    pub right_rhs: f64,
    /// Source of this disjunction.
    pub source: DisjunctionSource,
}

/// Source of a disjunction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisjunctionSource {
    /// From integrality of a variable.
    Integrality(usize),
    /// From follower optimality conditions.
    FollowerOptimality,
    /// From complementarity slackness.
    ComplementarySlackness(usize),
    /// Custom disjunction.
    Custom(String),
}

/// A disjunctive cut.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisjunctiveCut {
    pub cut: BilevelCut,
    pub disjunction: Disjunction,
    pub cglp_objective: f64,
    pub rank: u32,
}

/// CGLP formulation for generating the deepest disjunctive cut.
#[derive(Debug, Clone)]
pub struct CglpFormulation {
    /// Number of original variables (x,y).
    pub n_vars: usize,
    /// The disjunction.
    pub disjunction: Disjunction,
    /// Original constraint matrix rows: Ax sense b.
    pub constraint_coeffs: Vec<Vec<f64>>,
    pub constraint_rhs: Vec<f64>,
    pub constraint_senses: Vec<ConstraintSense>,
    /// Point to separate.
    pub point_to_separate: Vec<f64>,
    /// Normalization method.
    pub normalization: NormalizationMethod,
}

impl CglpFormulation {
    pub fn new(
        n_vars: usize,
        disjunction: Disjunction,
        constraint_coeffs: Vec<Vec<f64>>,
        constraint_rhs: Vec<f64>,
        constraint_senses: Vec<ConstraintSense>,
        point: Vec<f64>,
        normalization: NormalizationMethod,
    ) -> Self {
        Self {
            n_vars,
            disjunction,
            constraint_coeffs,
            constraint_rhs,
            constraint_senses,
            point_to_separate: point,
            normalization,
        }
    }

    /// Build the CGLP as an LP problem.
    /// The CGLP finds the deepest valid cut (alpha, beta) that is valid for
    /// the disjunction and most violated at the given point.
    pub fn build_lp(&self) -> CutResult<LpProblem> {
        let n = self.n_vars;
        let m = self.constraint_coeffs.len();

        // CGLP variables: alpha (n), beta_0 (1), u1 (m), u2 (m), v1 (1), v2 (1)
        let total_vars = n + 1 + 2 * m + 2;

        let mut lp = LpProblem::new(total_vars, 0);
        lp.direction = OptDirection::Maximize;

        // Objective: maximize alpha^T x* - beta_0
        let mut obj_c = vec![0.0; total_vars];
        for i in 0..n {
            obj_c[i] = self.point_to_separate.get(i).copied().unwrap_or(0.0);
        }
        obj_c[n] = -1.0; // beta_0
        lp.c = obj_c;

        // Variable bounds.
        lp.var_bounds = vec![
            VarBound {
                lower: f64::NEG_INFINITY,
                upper: f64::INFINITY
            };
            n + 1
        ];
        // Multipliers u1, u2 are non-negative.
        for _ in 0..2 * m + 2 {
            lp.var_bounds.push(VarBound {
                lower: 0.0,
                upper: f64::INFINITY,
            });
        }

        // Constraints:
        // alpha = A^T u1 + left_disj * v1 (for left side)
        // alpha = A^T u2 + right_disj * v2 (for right side)
        // beta_0 <= b^T u1 + left_rhs * v1
        // beta_0 <= b^T u2 + right_rhs * v2
        // normalization constraint

        // For simplicity, we build a condensed version.
        let mut con_rows: Vec<Vec<f64>> = Vec::new();
        let mut con_rhs: Vec<f64> = Vec::new();
        let mut con_senses: Vec<ConstraintSense> = Vec::new();

        // Normalization: sum of u1 + u2 + v1 + v2 = 1
        let mut norm_row = vec![0.0; total_vars];
        for i in (n + 1)..total_vars {
            norm_row[i] = 1.0;
        }
        con_rows.push(norm_row);
        con_rhs.push(1.0);
        con_senses.push(ConstraintSense::Eq);

        // Linking: for each original variable j, alpha_j = ...
        for j in 0..n {
            let mut row = vec![0.0; total_vars];
            row[j] = 1.0;
            for (i, con) in self.constraint_coeffs.iter().enumerate() {
                let a_ij = con.get(j).copied().unwrap_or(0.0);
                row[n + 1 + i] = -a_ij;
                row[n + 1 + m + i] = -a_ij;
            }
            // Disjunction terms.
            let left_coeff = self
                .disjunction
                .left_coeffs
                .iter()
                .find(|&&(idx, _)| idx == j)
                .map(|&(_, c)| c)
                .unwrap_or(0.0);
            let right_coeff = self
                .disjunction
                .right_coeffs
                .iter()
                .find(|&&(idx, _)| idx == j)
                .map(|&(_, c)| c)
                .unwrap_or(0.0);
            row[n + 1 + 2 * m] = -left_coeff;
            row[n + 1 + 2 * m + 1] = -right_coeff;

            con_rows.push(row);
            con_rhs.push(0.0);
            con_senses.push(ConstraintSense::Eq);
        }

        // RHS linking: beta_0 <= b^T u1 + ...
        let mut rhs_row = vec![0.0; total_vars];
        rhs_row[n] = 1.0; // beta_0
        for (i, &rhs_val) in self.constraint_rhs.iter().enumerate() {
            rhs_row[n + 1 + i] = -rhs_val;
        }
        rhs_row[n + 1 + 2 * m] = -self.disjunction.left_rhs;
        con_rows.push(rhs_row.clone());
        con_rhs.push(0.0);
        con_senses.push(ConstraintSense::Le);

        for (i, &rhs_val) in self.constraint_rhs.iter().enumerate() {
            rhs_row[n + 1 + m + i] = -rhs_val;
        }
        rhs_row[n + 1 + 2 * m + 1] = -self.disjunction.right_rhs;
        con_rows.push(rhs_row);
        con_rhs.push(0.0);
        con_senses.push(ConstraintSense::Le);

        lp.num_constraints = con_rows.len();
        lp.senses = con_senses;
        lp.b_rhs = con_rhs;

        // Build sparse constraint matrix.
        let mut entries = Vec::new();
        for (row_idx, row) in con_rows.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                if val.abs() > self.normalization_tolerance() {
                    entries.push(SparseEntry {
                        row: row_idx,
                        col: col_idx,
                        value: val,
                    });
                }
            }
        }
        lp.a_matrix = SparseMatrix {
            rows: lp.num_constraints,
            cols: total_vars,
            entries,
        };

        Ok(lp)
    }

    fn normalization_tolerance(&self) -> f64 {
        TOLERANCE
    }

    /// Extract a cut from the CGLP solution.
    pub fn extract_cut(&self, solution: &LpSolution) -> CutResult<BilevelCut> {
        let n = self.n_vars;
        let mut coeffs = Vec::new();
        for j in 0..n {
            let alpha_j = solution.primal.get(j).copied().unwrap_or(0.0);
            if alpha_j.abs() > TOLERANCE {
                coeffs.push((j, alpha_j));
            }
        }
        let beta_0 = solution.primal.get(n).copied().unwrap_or(0.0);

        if coeffs.is_empty() {
            return Err(CutError::NumericalIssue("CGLP produced zero cut".into()));
        }

        let cut =
            BilevelCut::new(coeffs, beta_0, ConstraintSense::Le).with_name("disj".to_string());
        Ok(cut)
    }
}

/// Disjunctive cut generator.
#[derive(Debug)]
pub struct DisjunctiveCutGenerator {
    pub config: DisjunctiveConfig,
    n_leader: usize,
    n_follower: usize,
    follower_obj: Vec<f64>,
    cut_counter: u64,
    stats: DisjunctiveStats,
}

/// Statistics for disjunctive cut generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DisjunctiveStats {
    pub total_disjunctions: usize,
    pub total_cglps_solved: usize,
    pub total_cuts_generated: usize,
    pub total_bilevel_disj: usize,
    pub avg_cglp_obj: f64,
}

impl DisjunctiveCutGenerator {
    pub fn new(
        config: DisjunctiveConfig,
        n_leader: usize,
        n_follower: usize,
        follower_obj: Vec<f64>,
    ) -> Self {
        Self {
            config,
            n_leader,
            n_follower,
            follower_obj,
            cut_counter: 0,
            stats: DisjunctiveStats::default(),
        }
    }

    /// Generate integrality-based disjunctions for fractional integer variables.
    pub fn create_integrality_disjunctions(
        &self,
        point: &[f64],
        integer_vars: &[usize],
    ) -> Vec<Disjunction> {
        let mut disjs = Vec::new();
        for &j in integer_vars {
            let val = point.get(j).copied().unwrap_or(0.0);
            let frac = val - val.floor();
            if frac < self.config.tolerance || frac > 1.0 - self.config.tolerance {
                continue;
            }
            disjs.push(Disjunction {
                left_coeffs: vec![(j, 1.0)],
                left_rhs: val.floor(),
                right_coeffs: vec![(j, 1.0)],
                right_rhs: val.ceil(),
                source: DisjunctionSource::Integrality(j),
            });
        }
        disjs
    }

    /// Create bilevel disjunctions from follower optimality conditions.
    pub fn create_bilevel_disjunctions(
        &self,
        point: &[f64],
        follower_dual: &[f64],
    ) -> Vec<Disjunction> {
        if !self.config.bilevel_disjunctions {
            return Vec::new();
        }
        let mut disjs = Vec::new();

        // For each follower constraint with near-zero dual:
        // either the constraint is tight or the dual is zero.
        for (i, &dual) in follower_dual.iter().enumerate() {
            if dual.abs() < 10.0 * self.config.tolerance {
                // Complementary slackness disjunction:
                // either dual_i = 0 (already the case) or slack_i = 0.
                // We create a disjunction on the slack variable.
                let n_total = self.n_leader + self.n_follower;
                let slack_var = n_total + i;
                disjs.push(Disjunction {
                    left_coeffs: vec![(slack_var, 1.0)],
                    left_rhs: 0.0,
                    right_coeffs: vec![(slack_var, 1.0)],
                    right_rhs: self.config.tolerance,
                    source: DisjunctionSource::ComplementarySlackness(i),
                });
            }
        }

        self.stats_update_bilevel(disjs.len());
        disjs
    }

    fn stats_update_bilevel(&self, _count: usize) {
        // Stats are updated through mutable reference in generate methods.
    }

    /// Generate disjunctive cuts using the CGLP approach.
    pub fn generate(
        &mut self,
        point: &[f64],
        disjunctions: &[Disjunction],
        constraint_coeffs: &[Vec<f64>],
        constraint_rhs: &[f64],
        constraint_senses: &[ConstraintSense],
    ) -> CutResult<Vec<DisjunctiveCut>> {
        let mut cuts = Vec::new();
        let n_vars = point.len();

        for disj in disjunctions.iter().take(self.config.max_cuts) {
            self.stats.total_disjunctions += 1;

            let cglp = CglpFormulation::new(
                n_vars,
                disj.clone(),
                constraint_coeffs.to_vec(),
                constraint_rhs.to_vec(),
                constraint_senses.to_vec(),
                point.to_vec(),
                self.config.normalization,
            );

            match cglp.build_lp() {
                Ok(lp) => {
                    self.stats.total_cglps_solved += 1;
                    match bicut_lp::solve_lp(&lp) {
                        Ok(sol) if sol.status == LpStatus::Optimal => {
                            if let Ok(mut cut) = cglp.extract_cut(&sol) {
                                cut.name = format!("disj_{}", self.cut_counter);
                                cut.rank = 1;
                                let efficacy = cut.compute_efficacy(point);
                                if efficacy >= self.config.min_efficacy {
                                    self.cut_counter += 1;
                                    self.stats.total_cuts_generated += 1;
                                    cuts.push(DisjunctiveCut {
                                        cut,
                                        disjunction: disj.clone(),
                                        cglp_objective: sol.objective,
                                        rank: 1,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Err(_) => {}
            }
        }

        Ok(cuts)
    }

    /// Convenience: generate cuts from integrality disjunctions.
    pub fn generate_from_integrality(
        &mut self,
        point: &[f64],
        integer_vars: &[usize],
        constraint_coeffs: &[Vec<f64>],
        constraint_rhs: &[f64],
        constraint_senses: &[ConstraintSense],
    ) -> CutResult<Vec<DisjunctiveCut>> {
        let disjs = self.create_integrality_disjunctions(point, integer_vars);
        self.generate(
            point,
            &disjs,
            constraint_coeffs,
            constraint_rhs,
            constraint_senses,
        )
    }

    pub fn stats(&self) -> &DisjunctiveStats {
        &self.stats
    }
    pub fn reset_stats(&mut self) {
        self.stats = DisjunctiveStats::default();
    }
    pub fn cuts_generated(&self) -> u64 {
        self.cut_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_generator() -> DisjunctiveCutGenerator {
        DisjunctiveCutGenerator::new(DisjunctiveConfig::default(), 2, 2, vec![1.0, 1.0])
    }

    #[test]
    fn test_config_default() {
        let cfg = DisjunctiveConfig::default();
        assert!(cfg.max_cuts > 0);
        assert!(cfg.bilevel_disjunctions);
    }

    #[test]
    fn test_generator_creation() {
        let gen = make_generator();
        assert_eq!(gen.cuts_generated(), 0);
    }

    #[test]
    fn test_create_integrality_disjunctions() {
        let gen = make_generator();
        let disjs = gen.create_integrality_disjunctions(&[1.5, 2.0, 0.7, 1.0], &[0, 2]);
        assert_eq!(disjs.len(), 2);
        assert_eq!(disjs[0].left_rhs, 1.0);
        assert_eq!(disjs[0].right_rhs, 2.0);
    }

    #[test]
    fn test_create_bilevel_disjunctions() {
        let gen = make_generator();
        let disjs = gen.create_bilevel_disjunctions(&[0.5, 0.5, 0.5, 0.5], &[0.0, 0.1]);
        assert!(disjs.len() >= 1);
    }

    #[test]
    fn test_normalization_methods() {
        assert_eq!(NormalizationMethod::Standard, NormalizationMethod::Standard);
        assert_ne!(
            NormalizationMethod::Standard,
            NormalizationMethod::Euclidean
        );
    }

    #[test]
    fn test_disjunction_source_variants() {
        let _s1 = DisjunctionSource::Integrality(0);
        let _s2 = DisjunctionSource::FollowerOptimality;
        let _s3 = DisjunctionSource::ComplementarySlackness(1);
        let _s4 = DisjunctionSource::Custom("test".into());
    }

    #[test]
    fn test_cglp_build() {
        let disj = Disjunction {
            left_coeffs: vec![(0, 1.0)],
            left_rhs: 1.0,
            right_coeffs: vec![(0, 1.0)],
            right_rhs: 2.0,
            source: DisjunctionSource::Integrality(0),
        };
        let cglp = CglpFormulation::new(
            2,
            disj,
            vec![vec![1.0, 0.0]],
            vec![5.0],
            vec![ConstraintSense::Le],
            vec![1.5, 0.5],
            NormalizationMethod::Standard,
        );
        let lp = cglp.build_lp();
        assert!(lp.is_ok());
    }

    #[test]
    fn test_stats_default() {
        let stats = DisjunctiveStats::default();
        assert_eq!(stats.total_cuts_generated, 0);
    }

    #[test]
    fn test_disjunctive_cut_struct() {
        let dc = DisjunctiveCut {
            cut: BilevelCut::new(vec![(0, 1.0)], 1.0, ConstraintSense::Ge),
            disjunction: Disjunction {
                left_coeffs: vec![],
                left_rhs: 0.0,
                right_coeffs: vec![],
                right_rhs: 1.0,
                source: DisjunctionSource::FollowerOptimality,
            },
            cglp_objective: 0.5,
            rank: 1,
        };
        assert_eq!(dc.rank, 1);
    }

    #[test]
    fn test_generate_empty_disjunctions() {
        let mut gen = make_generator();
        let result = gen.generate(&[0.5, 0.5], &[], &[], &[], &[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
