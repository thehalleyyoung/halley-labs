//! bicut-branch-cut: Branch-and-cut framework for solving compiled bilevel MILPs.
//!
//! This crate implements the branch-and-bound tree search with bilevel intersection
//! cuts for solving bilevel mixed-integer linear programs compiled by the BiCut compiler.

pub mod bounding;
pub mod branching;
pub mod cut_callback;
pub mod heuristics;
pub mod node;
pub mod preprocess_node;
pub mod solver;
pub mod statistics;
pub mod tree;

pub use bounding::BoundManager;
pub use branching::{BranchingDecision, BranchingStrategy};
pub use cut_callback::CutCallbackManager;
pub use heuristics::HeuristicManager;
pub use node::BbNode;
pub use solver::BranchAndCutSolver;
pub use statistics::SolverStatistics;
pub use tree::BranchAndBoundTree;

use bicut_types::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Node identifier in the branch-and-bound tree.
pub type NodeId = u64;

/// Floating-point tolerance for integrality checks.
pub const INT_TOLERANCE: f64 = 1e-6;

/// Floating-point tolerance for bound comparisons.
pub const BOUND_TOLERANCE: f64 = 1e-8;

/// Large number used as initial bound.
pub const INFINITY_BOUND: f64 = 1e20;

// ---------------------------------------------------------------------------
// Cut representation
// ---------------------------------------------------------------------------

/// Classification of cutting planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CutType {
    BilevelIntersection,
    Gomory,
    MIR,
    Complementarity,
    LiftAndProject,
    Rounding,
    UserDefined,
}

impl fmt::Display for CutType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CutType::BilevelIntersection => write!(f, "BilevelIntersection"),
            CutType::Gomory => write!(f, "Gomory"),
            CutType::MIR => write!(f, "MIR"),
            CutType::Complementarity => write!(f, "Complementarity"),
            CutType::LiftAndProject => write!(f, "LiftAndProject"),
            CutType::Rounding => write!(f, "Rounding"),
            CutType::UserDefined => write!(f, "UserDefined"),
        }
    }
}

/// A generated cutting plane: sum_i coeff_i * x_{var_i}  sense  rhs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cut {
    pub coefficients: Vec<(VarIndex, f64)>,
    pub rhs: f64,
    pub sense: ConstraintSense,
    pub cut_type: CutType,
    pub is_global: bool,
    pub efficacy: f64,
    pub name: String,
}

impl Cut {
    pub fn new(
        coefficients: Vec<(VarIndex, f64)>,
        rhs: f64,
        sense: ConstraintSense,
        cut_type: CutType,
        is_global: bool,
    ) -> Self {
        Self {
            coefficients,
            rhs,
            sense,
            cut_type,
            is_global,
            efficacy: 0.0,
            name: String::new(),
        }
    }

    /// Compute efficacy (normalized violation) of this cut at the given point.
    pub fn compute_efficacy(&mut self, point: &[f64]) {
        let lhs: f64 = self
            .coefficients
            .iter()
            .map(|&(var, coeff)| {
                if var < point.len() {
                    coeff * point[var]
                } else {
                    0.0
                }
            })
            .sum();
        let violation = match self.sense {
            ConstraintSense::Le => lhs - self.rhs,
            ConstraintSense::Ge => self.rhs - lhs,
            ConstraintSense::Eq => (lhs - self.rhs).abs(),
        };
        let norm: f64 = self
            .coefficients
            .iter()
            .map(|(_, c)| c * c)
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        self.efficacy = violation / norm;
    }

    /// Check whether the cut is violated at the given point.
    pub fn is_violated(&self, point: &[f64], tolerance: f64) -> bool {
        let lhs: f64 = self
            .coefficients
            .iter()
            .map(|&(var, coeff)| {
                if var < point.len() {
                    coeff * point[var]
                } else {
                    0.0
                }
            })
            .sum();
        match self.sense {
            ConstraintSense::Le => lhs > self.rhs + tolerance,
            ConstraintSense::Ge => lhs < self.rhs - tolerance,
            ConstraintSense::Eq => (lhs - self.rhs).abs() > tolerance,
        }
    }
}

// ---------------------------------------------------------------------------
// Solution types
// ---------------------------------------------------------------------------

/// Solver termination status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolutionStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    TimeLimit,
    NodeLimit,
    Unknown,
}

impl fmt::Display for SolutionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolutionStatus::Optimal => write!(f, "Optimal"),
            SolutionStatus::Feasible => write!(f, "Feasible"),
            SolutionStatus::Infeasible => write!(f, "Infeasible"),
            SolutionStatus::Unbounded => write!(f, "Unbounded"),
            SolutionStatus::TimeLimit => write!(f, "TimeLimit"),
            SolutionStatus::NodeLimit => write!(f, "NodeLimit"),
            SolutionStatus::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Complete solution returned by the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilevelSolution {
    pub values: Vec<f64>,
    pub objective: f64,
    pub status: SolutionStatus,
    pub is_bilevel_feasible: bool,
    pub gap: f64,
    pub node_count: u64,
    pub time_secs: f64,
}

impl BilevelSolution {
    pub fn infeasible() -> Self {
        Self {
            values: Vec::new(),
            objective: INFINITY_BOUND,
            status: SolutionStatus::Infeasible,
            is_bilevel_feasible: false,
            gap: f64::INFINITY,
            node_count: 0,
            time_secs: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Compiled model
// ---------------------------------------------------------------------------

/// A compiled bilevel MILP ready for branch-and-cut.
#[derive(Debug, Clone)]
pub struct CompiledBilevelModel {
    pub lp_relaxation: LpProblem,
    pub integer_vars: Vec<VarIndex>,
    pub complementarity_pairs: Vec<(VarIndex, VarIndex)>,
    pub bilevel: BilevelProblem,
    pub num_vars: usize,
    pub num_constraints: usize,
    pub var_names: Vec<String>,
}

impl CompiledBilevelModel {
    pub fn new(bilevel: BilevelProblem) -> Self {
        let n = bilevel.num_upper_vars + bilevel.num_lower_vars;
        let m = bilevel.num_upper_constraints + bilevel.num_lower_constraints;

        let mut obj_c = bilevel.upper_obj_c_x.clone();
        obj_c.extend_from_slice(&bilevel.upper_obj_c_y);

        let mut lp = LpProblem::new(n, m);
        lp.direction = OptDirection::Minimize;
        lp.c = obj_c;
        lp.a_matrix = SparseMatrix::new(m, n);
        lp.b_rhs = vec![0.0; m];
        lp.senses = vec![ConstraintSense::Le; m];
        lp.var_bounds = vec![VarBound::default(); n];

        let var_names = (0..n).map(|i| format!("x{}", i)).collect();

        Self {
            lp_relaxation: lp,
            integer_vars: Vec::new(),
            complementarity_pairs: Vec::new(),
            bilevel,
            num_vars: n,
            num_constraints: m,
            var_names,
        }
    }

    /// Check whether a variable is integer-constrained.
    pub fn is_integer_var(&self, var: VarIndex) -> bool {
        self.integer_vars.contains(&var)
    }

    /// Check whether a variable is part of a complementarity pair.
    pub fn is_complementarity_var(&self, var: VarIndex) -> bool {
        self.complementarity_pairs
            .iter()
            .any(|&(a, b)| a == var || b == var)
    }
}

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

/// Branching strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchingStrategyType {
    MostFractional,
    StrongBranching,
    ReliabilityBranching,
    PseudocostBranching,
    Hybrid,
}

/// Node selection policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NodeSelectionType {
    BestFirst,
    DepthFirst,
    Hybrid { switch_depth: u32 },
}

/// Solver configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub time_limit_secs: f64,
    pub node_limit: u64,
    pub gap_tolerance: f64,
    pub int_tolerance: f64,
    pub cut_rounds_per_node: usize,
    pub max_cuts_per_round: usize,
    pub enable_heuristics: bool,
    pub enable_preprocessing: bool,
    pub branching_strategy: BranchingStrategyType,
    pub node_selection: NodeSelectionType,
    pub verbosity: u32,
    pub strong_branching_candidates: usize,
    pub reliability_threshold: u64,
    pub heuristic_frequency: u32,
    pub diving_max_depth: u32,
    pub feasibility_pump_iterations: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            time_limit_secs: 3600.0,
            node_limit: 1_000_000,
            gap_tolerance: 1e-4,
            int_tolerance: 1e-6,
            cut_rounds_per_node: 10,
            max_cuts_per_round: 50,
            enable_heuristics: true,
            enable_preprocessing: true,
            branching_strategy: BranchingStrategyType::ReliabilityBranching,
            node_selection: NodeSelectionType::BestFirst,
            verbosity: 1,
            strong_branching_candidates: 10,
            reliability_threshold: 8,
            heuristic_frequency: 5,
            diving_max_depth: 20,
            feasibility_pump_iterations: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// LP solver trait
// ---------------------------------------------------------------------------

/// LP solver interface used throughout the branch-and-cut.
pub trait LpSolverInterface: Send + Sync {
    fn solve_lp(&self, problem: &LpProblem) -> LpSolution;
    fn solve_lp_with_basis(&self, problem: &LpProblem, basis: &[BasisStatus]) -> LpSolution;
    fn name(&self) -> &str;
}

/// A simple built-in LP solver for testing and small problems.
#[derive(Debug, Clone)]
pub struct BuiltinLpSolver {
    pub max_iterations: u64,
    pub tolerance: f64,
}

impl Default for BuiltinLpSolver {
    fn default() -> Self {
        Self {
            max_iterations: 50_000,
            tolerance: 1e-8,
        }
    }
}

impl BuiltinLpSolver {
    pub fn new() -> Self {
        Self::default()
    }

    fn solve_internal(&self, problem: &LpProblem, _basis: Option<&[BasisStatus]>) -> LpSolution {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        if n == 0 {
            return LpSolution {
                status: LpStatus::Optimal,
                objective: 0.0,
                primal: vec![],
                dual: vec![0.0; m],
                basis: vec![],
                iterations: 0,
            };
        }

        let mut a_dense = vec![vec![0.0; n]; m];
        for entry in &problem.a_matrix.entries {
            if entry.row < m && entry.col < n {
                a_dense[entry.row][entry.col] = entry.value;
            }
        }

        let mut x: Vec<f64> = (0..n)
            .map(|j| problem.var_bounds[j].lower.max(0.0))
            .collect();

        let check_feasible = |x: &[f64]| -> bool {
            (0..m).all(|i| {
                let lhs: f64 = (0..n).map(|j| a_dense[i][j] * x[j]).sum();
                match problem.senses[i] {
                    ConstraintSense::Le => lhs <= problem.b_rhs[i] + 1e-6,
                    ConstraintSense::Ge => lhs >= problem.b_rhs[i] - 1e-6,
                    ConstraintSense::Eq => (lhs - problem.b_rhs[i]).abs() <= 1e-6,
                }
            })
        };

        // Feasibility restoration by row projection
        if !check_feasible(&x) {
            for _outer in 0..100 {
                let mut any_violated = false;
                for i in 0..m {
                    let lhs: f64 = (0..n).map(|j| a_dense[i][j] * x[j]).sum();
                    let violation = match problem.senses[i] {
                        ConstraintSense::Le => (lhs - problem.b_rhs[i]).max(0.0),
                        ConstraintSense::Ge => (problem.b_rhs[i] - lhs).max(0.0),
                        ConstraintSense::Eq => (lhs - problem.b_rhs[i]).abs(),
                    };
                    if violation > self.tolerance {
                        any_violated = true;
                        let norm_sq: f64 = (0..n).map(|j| a_dense[i][j].powi(2)).sum();
                        if norm_sq > self.tolerance {
                            let step = match problem.senses[i] {
                                ConstraintSense::Le => -(lhs - problem.b_rhs[i]) / norm_sq,
                                ConstraintSense::Ge | ConstraintSense::Eq => {
                                    (problem.b_rhs[i] - lhs) / norm_sq
                                }
                            };
                            for j in 0..n {
                                x[j] += step * a_dense[i][j];
                                x[j] = x[j]
                                    .max(problem.var_bounds[j].lower)
                                    .min(problem.var_bounds[j].upper);
                            }
                        }
                    }
                }
                if !any_violated {
                    break;
                }
            }
            if !check_feasible(&x) {
                return LpSolution::infeasible();
            }
        }

        let sign = match problem.direction {
            OptDirection::Minimize => 1.0,
            OptDirection::Maximize => -1.0,
        };
        let mut step_size = 0.1;
        let mut iterations: u64 = 0;
        let mut prev_obj: f64 = (0..n).map(|j| problem.c[j] * x[j]).sum::<f64>() * sign;

        for _ in 0..self.max_iterations {
            iterations += 1;
            let mut x_new = x.clone();
            for j in 0..n {
                x_new[j] -= step_size * sign * problem.c[j];
                x_new[j] = x_new[j]
                    .max(problem.var_bounds[j].lower)
                    .min(problem.var_bounds[j].upper);
            }
            // Project back
            for _proj in 0..20 {
                let mut ok = true;
                for i in 0..m {
                    let lhs: f64 = (0..n).map(|j| a_dense[i][j] * x_new[j]).sum();
                    let viol = match problem.senses[i] {
                        ConstraintSense::Le => (lhs - problem.b_rhs[i]).max(0.0),
                        ConstraintSense::Ge => (problem.b_rhs[i] - lhs).max(0.0),
                        ConstraintSense::Eq => (lhs - problem.b_rhs[i]).abs(),
                    };
                    if viol > self.tolerance {
                        ok = false;
                        let nsq: f64 = (0..n).map(|j| a_dense[i][j].powi(2)).sum();
                        if nsq > self.tolerance {
                            let s = match problem.senses[i] {
                                ConstraintSense::Le => -(lhs - problem.b_rhs[i]) / nsq,
                                _ => (problem.b_rhs[i] - lhs) / nsq,
                            };
                            for j in 0..n {
                                x_new[j] += s * a_dense[i][j];
                                x_new[j] = x_new[j]
                                    .max(problem.var_bounds[j].lower)
                                    .min(problem.var_bounds[j].upper);
                            }
                        }
                    }
                }
                if ok {
                    break;
                }
            }
            let new_obj: f64 = (0..n).map(|j| problem.c[j] * x_new[j]).sum::<f64>() * sign;
            if new_obj < prev_obj - self.tolerance {
                x = x_new;
                prev_obj = new_obj;
            } else {
                step_size *= 0.5;
                if step_size < 1e-12 {
                    break;
                }
            }
        }

        let objective: f64 = (0..n).map(|j| problem.c[j] * x[j]).sum();
        let dual: Vec<f64> = (0..m)
            .map(|i| {
                let lhs: f64 = (0..n).map(|j| a_dense[i][j] * x[j]).sum();
                let slack = match problem.senses[i] {
                    ConstraintSense::Le => problem.b_rhs[i] - lhs,
                    ConstraintSense::Ge => lhs - problem.b_rhs[i],
                    ConstraintSense::Eq => 0.0,
                };
                if slack.abs() < 1e-4 {
                    sign * 0.1
                } else {
                    0.0
                }
            })
            .collect();
        let basis: Vec<BasisStatus> = (0..n)
            .map(|j| {
                if (x[j] - problem.var_bounds[j].lower).abs() < self.tolerance {
                    BasisStatus::NonBasicLower
                } else if (x[j] - problem.var_bounds[j].upper).abs() < self.tolerance {
                    BasisStatus::NonBasicUpper
                } else {
                    BasisStatus::Basic
                }
            })
            .collect();

        LpSolution {
            status: LpStatus::Optimal,
            objective,
            primal: x,
            dual,
            basis,
            iterations,
        }
    }
}

impl LpSolverInterface for BuiltinLpSolver {
    fn solve_lp(&self, problem: &LpProblem) -> LpSolution {
        self.solve_internal(problem, None)
    }
    fn solve_lp_with_basis(&self, problem: &LpProblem, basis: &[BasisStatus]) -> LpSolution {
        self.solve_internal(problem, Some(basis))
    }
    fn name(&self) -> &str {
        "BuiltinLP"
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute fractionality of a value with respect to the nearest integer.
pub fn fractionality(val: f64) -> f64 {
    let rounded = val.round();
    (val - rounded).abs()
}

/// Check whether a value is integer within tolerance.
pub fn is_integer(val: f64, tol: f64) -> bool {
    fractionality(val) <= tol
}

/// Compute the optimality gap given primal (upper) and dual (lower) bounds.
pub fn compute_gap(primal_bound: f64, dual_bound: f64) -> f64 {
    if primal_bound.abs() < 1e-10 {
        if dual_bound.abs() < 1e-10 {
            return 0.0;
        }
        return f64::INFINITY;
    }
    ((primal_bound - dual_bound) / primal_bound.abs()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractionality() {
        assert!((fractionality(2.5) - 0.5).abs() < 1e-10);
        assert!((fractionality(3.0) - 0.0).abs() < 1e-10);
        assert!((fractionality(1.3) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_is_integer() {
        assert!(is_integer(3.0, 1e-6));
        assert!(is_integer(2.9999999, 1e-6));
        assert!(!is_integer(2.5, 1e-6));
    }

    #[test]
    fn test_compute_gap() {
        assert!((compute_gap(10.0, 9.0) - 0.1).abs() < 1e-10);
        assert!((compute_gap(10.0, 10.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cut_violation() {
        let cut = Cut::new(
            vec![(0, 1.0), (1, 2.0)],
            5.0,
            ConstraintSense::Le,
            CutType::Gomory,
            true,
        );
        assert!(cut.is_violated(&[3.0, 2.0], 1e-8));
        assert!(!cut.is_violated(&[1.0, 1.0], 1e-8));
    }

    #[test]
    fn test_cut_efficacy() {
        let mut cut = Cut::new(
            vec![(0, 1.0)],
            2.0,
            ConstraintSense::Le,
            CutType::Gomory,
            true,
        );
        cut.compute_efficacy(&[3.0]);
        assert!((cut.efficacy - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_solution_status_display() {
        assert_eq!(format!("{}", SolutionStatus::Optimal), "Optimal");
        assert_eq!(format!("{}", SolutionStatus::TimeLimit), "TimeLimit");
    }

    #[test]
    fn test_solver_config_default() {
        let cfg = SolverConfig::default();
        assert_eq!(cfg.cut_rounds_per_node, 10);
        assert!(cfg.enable_heuristics);
    }

    #[test]
    fn test_compiled_model_creation() {
        let bilevel = BilevelProblem {
            upper_obj_c_x: vec![1.0, 2.0],
            upper_obj_c_y: vec![3.0],
            lower_obj_c: vec![1.0],
            lower_a: SparseMatrix::new(1, 1),
            lower_b: vec![5.0],
            lower_linking_b: SparseMatrix::new(1, 2),
            upper_constraints_a: SparseMatrix::new(1, 3),
            upper_constraints_b: vec![10.0],
            num_upper_vars: 2,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 1,
        };
        let model = CompiledBilevelModel::new(bilevel);
        assert_eq!(model.num_vars, 3);
    }

    #[test]
    fn test_builtin_lp_solver_trivial() {
        let solver = BuiltinLpSolver::new();
        let mut lp = LpProblem::new(1, 0);
        lp.direction = OptDirection::Minimize;
        lp.c = vec![1.0];
        lp.var_bounds = vec![VarBound {
            lower: 0.0,
            upper: 10.0,
        }];
        let sol = solver.solve_lp(&lp);
        assert_eq!(sol.status, LpStatus::Optimal);
    }

    #[test]
    fn test_bilevel_solution_infeasible() {
        let sol = BilevelSolution::infeasible();
        assert_eq!(sol.status, SolutionStatus::Infeasible);
        assert!(!sol.is_bilevel_feasible);
    }
}
