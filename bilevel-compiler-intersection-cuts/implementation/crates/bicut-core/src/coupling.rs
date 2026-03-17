//! Coupling analysis for bilevel optimization problems.
//!
//! Identifies how leader and follower variables interact, detects coupling
//! types (objective-only, constraint-only, both), and computes coupling
//! strength metrics.

use bicut_types::{BilevelProblem, CouplingType, SparseMatrix, SparseMatrixCsr, DEFAULT_TOLERANCE};
use indexmap::IndexSet;
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Coupling strength classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CouplingStrength {
    None,
    Weak,
    Moderate,
    Strong,
    Full,
}

/// Detailed coupling analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingReport {
    pub coupling_type: CouplingType,
    pub strength: CouplingStrength,
    pub objective_coupling: ObjectiveCouplingInfo,
    pub constraint_coupling: ConstraintCouplingInfo,
    pub coupling_variables: CouplingVariables,
    pub metrics: CouplingMetrics,
    pub notes: Vec<String>,
}

/// Information about how the leader participates in the follower's objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveCouplingInfo {
    pub leader_in_upper_obj: bool,
    pub follower_in_upper_obj: bool,
    pub leader_in_lower_obj: bool,
    pub cross_term_count: usize,
}

/// Information about constraint coupling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintCouplingInfo {
    pub num_linking_constraints: usize,
    pub num_leader_vars_in_lower: usize,
    pub num_follower_vars_in_upper: usize,
    pub linking_density: f64,
    pub max_leader_per_constraint: usize,
    pub constraint_coupling_map: HashMap<usize, Vec<usize>>,
}

/// The sets of coupling variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingVariables {
    pub leader_indices_in_lower: Vec<usize>,
    pub follower_indices_in_upper: Vec<usize>,
    pub shared_indices: Vec<usize>,
}

/// Quantitative coupling metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMetrics {
    pub density: f64,
    pub strength_score: f64,
    pub symmetry: f64,
    pub max_coefficient_ratio: f64,
    pub effective_coupling_dimension: usize,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Configuration for coupling analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingConfig {
    pub tolerance: f64,
}

impl Default for CouplingConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

/// Coupling analysis engine.
pub struct CouplingAnalyzer {
    config: CouplingConfig,
}

impl CouplingAnalyzer {
    pub fn new(config: CouplingConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(CouplingConfig::default())
    }

    /// Perform complete coupling analysis.
    pub fn analyze(&self, problem: &BilevelProblem) -> CouplingReport {
        let obj_info = self.analyze_objective_coupling(problem);
        let constr_info = self.analyze_constraint_coupling(problem);
        let coupling_vars = self.identify_coupling_variables(problem);
        let metrics = self.compute_metrics(problem, &obj_info, &constr_info);

        let coupling_type = determine_coupling_type(&obj_info, &constr_info);
        let strength = classify_strength(metrics.strength_score);

        let mut notes = Vec::new();
        if coupling_type == CouplingType::None {
            notes
                .push("Leader and follower are decoupled; can be solved independently".to_string());
        }
        if matches!(strength, CouplingStrength::Strong | CouplingStrength::Full) {
            notes.push("Strong coupling detected; decomposition may be less effective".to_string());
        }
        if metrics.symmetry > 0.8 {
            notes.push("Nearly symmetric coupling structure detected".to_string());
        }
        if constr_info.max_leader_per_constraint > problem.num_upper_vars / 2 {
            notes.push(
                "Dense linking: many leader variables appear in each lower constraint".to_string(),
            );
        }

        debug!(
            "Coupling analysis: type={:?}, strength={:?}, score={:.3}",
            coupling_type, strength, metrics.strength_score
        );

        CouplingReport {
            coupling_type,
            strength,
            objective_coupling: obj_info,
            constraint_coupling: constr_info,
            coupling_variables: coupling_vars,
            metrics,
            notes,
        }
    }

    /// Analyze objective coupling.
    pub fn analyze_objective_coupling(&self, problem: &BilevelProblem) -> ObjectiveCouplingInfo {
        let tol = self.config.tolerance;

        let leader_in_upper = problem.upper_obj_c_x.iter().any(|&c| c.abs() > tol);
        let follower_in_upper = problem.upper_obj_c_y.iter().any(|&c| c.abs() > tol);

        // The lower-level objective is c^T y; check if there's any x dependence
        // In the current model, lower_obj_c is purely in y space, so no leader in lower obj
        let leader_in_lower = false;

        // Cross terms: count leader-follower product terms in upper objective
        // In the linear model, cross terms = positions where both c_x[i] and c_y[j] are nonzero
        let cx_nnz = problem
            .upper_obj_c_x
            .iter()
            .filter(|&&c| c.abs() > tol)
            .count();
        let cy_nnz = problem
            .upper_obj_c_y
            .iter()
            .filter(|&&c| c.abs() > tol)
            .count();
        let cross_term_count = if leader_in_upper && follower_in_upper {
            cx_nnz.min(cy_nnz)
        } else {
            0
        };

        ObjectiveCouplingInfo {
            leader_in_upper_obj: leader_in_upper,
            follower_in_upper_obj: follower_in_upper,
            leader_in_lower_obj: leader_in_lower,
            cross_term_count,
        }
    }

    /// Analyze constraint coupling.
    pub fn analyze_constraint_coupling(&self, problem: &BilevelProblem) -> ConstraintCouplingInfo {
        let tol = self.config.tolerance;
        let linking_csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_linking_b);

        // Leader variables in lower-level constraints
        let mut leader_vars_set = IndexSet::new();
        let mut constraint_map: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut max_leader = 0usize;

        for r in 0..linking_csr.nrows {
            let entries = linking_csr.row_entries(r);
            let leader_cols: Vec<usize> = entries
                .iter()
                .filter(|(_, v)| v.abs() > tol)
                .map(|(c, _)| *c)
                .collect();

            if leader_cols.len() > max_leader {
                max_leader = leader_cols.len();
            }

            for &c in &leader_cols {
                leader_vars_set.insert(c);
            }

            if !leader_cols.is_empty() {
                constraint_map.insert(r, leader_cols);
            }
        }

        // Follower variables in upper-level constraints
        let nx = problem.num_upper_vars;
        let mut follower_vars_set = IndexSet::new();
        for entry in &problem.upper_constraints_a.entries {
            if entry.col >= nx && entry.value.abs() > tol {
                follower_vars_set.insert(entry.col - nx);
            }
        }

        let linking_total = (linking_csr.nrows * linking_csr.ncols).max(1);
        let linking_nnz = linking_csr.nnz();

        ConstraintCouplingInfo {
            num_linking_constraints: constraint_map.len(),
            num_leader_vars_in_lower: leader_vars_set.len(),
            num_follower_vars_in_upper: follower_vars_set.len(),
            linking_density: linking_nnz as f64 / linking_total as f64,
            max_leader_per_constraint: max_leader,
            constraint_coupling_map: constraint_map,
        }
    }

    /// Identify coupling variable indices.
    pub fn identify_coupling_variables(&self, problem: &BilevelProblem) -> CouplingVariables {
        let tol = self.config.tolerance;

        let mut leader_in_lower = IndexSet::new();
        for entry in &problem.lower_linking_b.entries {
            if entry.value.abs() > tol {
                leader_in_lower.insert(entry.col);
            }
        }

        let nx = problem.num_upper_vars;
        let mut follower_in_upper = IndexSet::new();
        for entry in &problem.upper_constraints_a.entries {
            if entry.col >= nx && entry.value.abs() > tol {
                follower_in_upper.insert(entry.col - nx);
            }
        }

        // Shared: leader vars in lower AND follower vars in upper (both directions coupled)
        let shared: Vec<usize> = if leader_in_lower.is_empty() || follower_in_upper.is_empty() {
            Vec::new()
        } else {
            // In this model, shared refers to index pairs; approximate with min set
            let leader_count = leader_in_lower.len();
            let follower_count = follower_in_upper.len();
            (0..leader_count.min(follower_count)).collect()
        };

        CouplingVariables {
            leader_indices_in_lower: leader_in_lower.into_iter().collect(),
            follower_indices_in_upper: follower_in_upper.into_iter().collect(),
            shared_indices: shared,
        }
    }

    /// Compute quantitative coupling metrics.
    pub fn compute_metrics(
        &self,
        problem: &BilevelProblem,
        obj: &ObjectiveCouplingInfo,
        constr: &ConstraintCouplingInfo,
    ) -> CouplingMetrics {
        let tol = self.config.tolerance;
        let nx = problem.num_upper_vars;
        let ny = problem.num_lower_vars;
        let ml = problem.num_lower_constraints;
        let mu = problem.num_upper_constraints;

        // Density: fraction of possible couplings that exist
        let possible_couplings = (nx * ml + ny * mu).max(1) as f64;
        let actual_couplings = (constr.num_leader_vars_in_lower * constr.num_linking_constraints
            + constr.num_follower_vars_in_upper * mu) as f64;
        let density = (actual_couplings / possible_couplings).min(1.0);

        // Strength score: composite of density, objective coupling, constraint coupling
        let obj_score = if obj.leader_in_upper_obj && obj.follower_in_upper_obj {
            0.5
        } else {
            0.0
        } + if obj.leader_in_lower_obj { 0.3 } else { 0.0 };
        let constr_score = constr.linking_density;
        let strength_score = (0.4 * obj_score + 0.6 * constr_score).min(1.0);

        // Symmetry: how balanced is the coupling in both directions?
        let leader_ratio = if nx > 0 {
            constr.num_leader_vars_in_lower as f64 / nx as f64
        } else {
            0.0
        };
        let follower_ratio = if ny > 0 {
            constr.num_follower_vars_in_upper as f64 / ny as f64
        } else {
            0.0
        };
        let symmetry = if (leader_ratio + follower_ratio) > 0.0 {
            1.0 - (leader_ratio - follower_ratio).abs() / (leader_ratio + follower_ratio)
        } else {
            1.0
        };

        // Max coefficient ratio in linking matrix
        let linking_vals: Vec<f64> = problem
            .lower_linking_b
            .entries
            .iter()
            .map(|e| e.value.abs())
            .filter(|&v| v > tol)
            .collect();
        let max_coeff_ratio = if linking_vals.len() >= 2 {
            let min_v = linking_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_v = linking_vals.iter().cloned().fold(0.0f64, f64::max);
            if min_v > tol {
                max_v / min_v
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Effective coupling dimension: number of independent coupling directions
        let eff_dim = constr
            .num_leader_vars_in_lower
            .min(constr.num_linking_constraints)
            .max(if obj.follower_in_upper_obj { 1 } else { 0 });

        CouplingMetrics {
            density,
            strength_score,
            symmetry,
            max_coefficient_ratio: max_coeff_ratio,
            effective_coupling_dimension: eff_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn determine_coupling_type(
    obj: &ObjectiveCouplingInfo,
    constr: &ConstraintCouplingInfo,
) -> CouplingType {
    let has_obj = obj.follower_in_upper_obj || obj.leader_in_lower_obj;
    let has_constr = constr.num_linking_constraints > 0 || constr.num_follower_vars_in_upper > 0;

    match (has_obj, has_constr) {
        (false, false) => CouplingType::None,
        (true, false) => CouplingType::ObjectiveOnly,
        (false, true) => CouplingType::ConstraintOnly,
        (true, true) => CouplingType::Both,
    }
}

fn classify_strength(score: f64) -> CouplingStrength {
    if score < 0.01 {
        CouplingStrength::None
    } else if score < 0.2 {
        CouplingStrength::Weak
    } else if score < 0.5 {
        CouplingStrength::Moderate
    } else if score < 0.8 {
        CouplingStrength::Strong
    } else {
        CouplingStrength::Full
    }
}

/// Quick coupling type detection without full analysis.
pub fn quick_coupling_type(problem: &BilevelProblem) -> CouplingType {
    let tol = DEFAULT_TOLERANCE;
    let has_y_in_upper_obj = problem.upper_obj_c_y.iter().any(|&c| c.abs() > tol);
    let has_linking = problem
        .lower_linking_b
        .entries
        .iter()
        .any(|e| e.value.abs() > tol);

    let nx = problem.num_upper_vars;
    let has_y_in_upper_constr = problem
        .upper_constraints_a
        .entries
        .iter()
        .any(|e| e.col >= nx && e.value.abs() > tol);

    let has_obj = has_y_in_upper_obj;
    let has_constr = has_linking || has_y_in_upper_constr;

    match (has_obj, has_constr) {
        (false, false) => CouplingType::None,
        (true, false) => CouplingType::ObjectiveOnly,
        (false, true) => CouplingType::ConstraintOnly,
        (true, true) => CouplingType::Both,
    }
}

/// Compute a simple coupling density metric.
pub fn coupling_density(problem: &BilevelProblem) -> f64 {
    let total = (problem.num_upper_vars * problem.num_lower_constraints).max(1);
    let nnz = problem
        .lower_linking_b
        .entries
        .iter()
        .filter(|e| e.value.abs() > DEFAULT_TOLERANCE)
        .count();
    nnz as f64 / total as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_coupled_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);

        let mut linking = SparseMatrix::new(2, 2);
        linking.add_entry(0, 0, 1.0);
        linking.add_entry(1, 1, 2.0);

        let mut upper_a = SparseMatrix::new(1, 4);
        upper_a.add_entry(0, 2, 1.0); // follower var in upper constraint

        BilevelProblem {
            upper_obj_c_x: vec![1.0, 0.0],
            upper_obj_c_y: vec![1.0, 0.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: linking,
            upper_constraints_a: upper_a,
            upper_constraints_b: vec![10.0],
            num_upper_vars: 2,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 1,
        }
    }

    fn make_decoupled_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![0.0, 0.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    #[test]
    fn test_coupled_type_both() {
        let p = make_coupled_problem();
        let analyzer = CouplingAnalyzer::with_defaults();
        let report = analyzer.analyze(&p);
        assert_eq!(report.coupling_type, CouplingType::Both);
    }

    #[test]
    fn test_decoupled() {
        let p = make_decoupled_problem();
        let analyzer = CouplingAnalyzer::with_defaults();
        let report = analyzer.analyze(&p);
        assert_eq!(report.coupling_type, CouplingType::None);
    }

    #[test]
    fn test_coupling_strength() {
        let p = make_coupled_problem();
        let analyzer = CouplingAnalyzer::with_defaults();
        let report = analyzer.analyze(&p);
        assert!(!matches!(report.strength, CouplingStrength::None));
    }

    #[test]
    fn test_coupling_variables() {
        let p = make_coupled_problem();
        let analyzer = CouplingAnalyzer::with_defaults();
        let report = analyzer.analyze(&p);
        assert!(!report.coupling_variables.leader_indices_in_lower.is_empty());
    }

    #[test]
    fn test_quick_coupling() {
        let p = make_coupled_problem();
        let ct = quick_coupling_type(&p);
        assert_eq!(ct, CouplingType::Both);
    }

    #[test]
    fn test_coupling_density() {
        let p = make_coupled_problem();
        let d = coupling_density(&p);
        assert!(d > 0.0);
    }

    #[test]
    fn test_metrics_symmetry() {
        let p = make_coupled_problem();
        let analyzer = CouplingAnalyzer::with_defaults();
        let report = analyzer.analyze(&p);
        assert!(report.metrics.symmetry >= 0.0 && report.metrics.symmetry <= 1.0);
    }

    #[test]
    fn test_constraint_coupling_map() {
        let p = make_coupled_problem();
        let analyzer = CouplingAnalyzer::with_defaults();
        let constr = analyzer.analyze_constraint_coupling(&p);
        assert!(constr.num_linking_constraints > 0);
    }
}
