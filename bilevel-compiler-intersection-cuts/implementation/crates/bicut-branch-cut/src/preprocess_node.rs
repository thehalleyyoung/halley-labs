//! Node preprocessing: domain propagation, constraint propagation, implied bounds,
//! probing, conflict analysis.

use crate::node::{BbNode, BranchDirection, BranchRecord, NodeStatus};
use crate::{
    CompiledBilevelModel, LpSolverInterface, SolverConfig, BOUND_TOLERANCE, INFINITY_BOUND,
};
use bicut_types::*;
use serde::{Deserialize, Serialize};

/// Results of preprocessing a node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreprocessResult {
    pub bounds_tightened: usize,
    pub vars_fixed: usize,
    pub infeasible_detected: bool,
    pub implications_found: usize,
    pub probing_fixings: usize,
}

/// Node preprocessor configuration and logic.
#[derive(Debug, Clone)]
pub struct NodePreprocessor {
    pub enable_domain_propagation: bool,
    pub enable_constraint_propagation: bool,
    pub enable_probing: bool,
    pub enable_conflict_analysis: bool,
    pub max_propagation_rounds: usize,
    pub probing_max_vars: usize,
}

impl NodePreprocessor {
    pub fn new(config: &SolverConfig) -> Self {
        Self {
            enable_domain_propagation: config.enable_preprocessing,
            enable_constraint_propagation: config.enable_preprocessing,
            enable_probing: false,
            enable_conflict_analysis: config.enable_preprocessing,
            max_propagation_rounds: 5,
            probing_max_vars: 10,
        }
    }

    /// Run all enabled preprocessing steps.
    pub fn preprocess(
        &self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
    ) -> PreprocessResult {
        let mut result = PreprocessResult::default();

        if detect_infeasibility(node) {
            result.infeasible_detected = true;
            node.status = NodeStatus::Infeasible;
            return result;
        }

        if self.enable_domain_propagation {
            let (tightened, infeasible) = self.domain_propagation(node, model);
            result.bounds_tightened += tightened;
            if infeasible {
                result.infeasible_detected = true;
                node.status = NodeStatus::Infeasible;
                return result;
            }
        }

        if self.enable_constraint_propagation {
            let (tightened, infeasible) = self.constraint_propagation(node, model);
            result.bounds_tightened += tightened;
            if infeasible {
                result.infeasible_detected = true;
                node.status = NodeStatus::Infeasible;
                return result;
            }
        }

        // Count fixed variables
        let n = node.var_lower_bounds.len().min(node.var_upper_bounds.len());
        result.vars_fixed = (0..n)
            .filter(|&j| {
                (node.var_upper_bounds[j] - node.var_lower_bounds[j]).abs() < BOUND_TOLERANCE
            })
            .count();

        if self.enable_probing {
            let (probed, infeasible) = self.probing(node, model, lp_solver);
            result.probing_fixings = probed;
            if infeasible {
                result.infeasible_detected = true;
                node.status = NodeStatus::Infeasible;
                return result;
            }
        }

        result.implications_found = self.implied_bounds_from_branching(node);

        if detect_infeasibility(node) {
            result.infeasible_detected = true;
            node.status = NodeStatus::Infeasible;
        }

        result
    }

    /// Tighten bounds based on branching history and complementarity.
    pub fn domain_propagation(
        &self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
    ) -> (usize, bool) {
        let mut tightened = 0usize;

        // Apply branching history bounds
        for record in node.branching_history.clone() {
            let var = record.variable;
            if var >= node.var_lower_bounds.len() {
                continue;
            }
            match record.direction {
                BranchDirection::Down => {
                    if record.bound_value < node.var_upper_bounds[var] - BOUND_TOLERANCE {
                        node.var_upper_bounds[var] = record.bound_value;
                        tightened += 1;
                    }
                }
                BranchDirection::Up => {
                    if record.bound_value > node.var_lower_bounds[var] + BOUND_TOLERANCE {
                        node.var_lower_bounds[var] = record.bound_value;
                        tightened += 1;
                    }
                }
            }
        }

        // Propagate complementarity
        tightened += propagate_complementarity(node, model);

        let infeasible = detect_infeasibility(node);
        (tightened, infeasible)
    }

    /// Iterate over constraints to tighten variable bounds.
    pub fn constraint_propagation(
        &self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
    ) -> (usize, bool) {
        let n = model.num_vars;
        let m = model.lp_relaxation.num_constraints;
        let mut total_tightened = 0usize;

        // Build row-wise representation
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for entry in &model.lp_relaxation.a_matrix.entries {
            if entry.row < m && entry.col < n {
                rows[entry.row].push((entry.col, entry.value));
            }
        }

        for _round in 0..self.max_propagation_rounds {
            let mut round_tightened = 0;

            for i in 0..m {
                if rows[i].is_empty() {
                    continue;
                }
                let rhs = model.lp_relaxation.b_rhs[i];
                let sense = model.lp_relaxation.senses[i];

                let t = tighten_bounds_from_constraint(
                    &rows[i],
                    rhs,
                    sense,
                    &mut node.var_lower_bounds,
                    &mut node.var_upper_bounds,
                );
                round_tightened += t;
            }

            total_tightened += round_tightened;
            if round_tightened == 0 {
                break;
            }

            if detect_infeasibility(node) {
                return (total_tightened, true);
            }
        }

        (total_tightened, false)
    }

    /// Derive implied bounds from the branching history.
    pub fn implied_bounds_from_branching(&self, node: &mut BbNode) -> usize {
        let mut implications = 0;

        for record in node.branching_history.clone() {
            let var = record.variable;
            if var >= node.var_lower_bounds.len() {
                continue;
            }

            // Apply the branch bound itself
            match record.direction {
                BranchDirection::Down => {
                    if record.bound_value < node.var_upper_bounds[var] - BOUND_TOLERANCE {
                        node.var_upper_bounds[var] = record.bound_value;
                        implications += 1;
                    }
                }
                BranchDirection::Up => {
                    if record.bound_value > node.var_lower_bounds[var] + BOUND_TOLERANCE {
                        node.var_lower_bounds[var] = record.bound_value;
                        implications += 1;
                    }
                }
            }

            let lb = node.var_lower_bounds[var];
            let ub = node.var_upper_bounds[var];

            // Tighten to nearest integer bounds
            let int_lb = lb.ceil();
            let int_ub = ub.floor();

            if int_lb > lb + BOUND_TOLERANCE {
                node.var_lower_bounds[var] = int_lb;
                implications += 1;
            }
            if int_ub < ub - BOUND_TOLERANCE {
                node.var_upper_bounds[var] = int_ub;
                implications += 1;
            }
        }
        implications
    }

    /// Probing: temporarily fix variables and check for infeasibility.
    pub fn probing(
        &self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
    ) -> (usize, bool) {
        let mut fixings = 0;
        let n = node.var_lower_bounds.len().min(node.var_upper_bounds.len());

        let probe_vars: Vec<VarIndex> = model
            .integer_vars
            .iter()
            .filter(|&&v| {
                v < n
                    && (node.var_upper_bounds[v] - node.var_lower_bounds[v]).abs() > BOUND_TOLERANCE
            })
            .copied()
            .take(self.probing_max_vars)
            .collect();

        for &var in &probe_vars {
            let orig_lb = node.var_lower_bounds[var];
            let orig_ub = node.var_upper_bounds[var];

            // Probe down: fix var to lower bound
            node.var_upper_bounds[var] = orig_lb;
            let down_lp = node.build_node_lp(model);
            let down_sol = lp_solver.solve_lp(&down_lp);
            let down_feasible = down_sol.status == LpStatus::Optimal;
            node.var_upper_bounds[var] = orig_ub;

            // Probe up: fix var to upper bound
            node.var_lower_bounds[var] = orig_ub;
            let up_lp = node.build_node_lp(model);
            let up_sol = lp_solver.solve_lp(&up_lp);
            let up_feasible = up_sol.status == LpStatus::Optimal;
            node.var_lower_bounds[var] = orig_lb;

            if !down_feasible && !up_feasible {
                // Both directions infeasible
                return (fixings, true);
            } else if !down_feasible {
                // Must go up
                let ceil_val = orig_lb.ceil().max(orig_lb);
                if ceil_val > orig_lb + BOUND_TOLERANCE {
                    node.var_lower_bounds[var] = ceil_val;
                    fixings += 1;
                }
            } else if !up_feasible {
                // Must go down
                let floor_val = orig_ub.floor().min(orig_ub);
                if floor_val < orig_ub - BOUND_TOLERANCE {
                    node.var_upper_bounds[var] = floor_val;
                    fixings += 1;
                }
            }
        }

        (fixings, false)
    }

    /// Analyze conflicts when infeasibility is detected.
    pub fn conflict_analysis(
        &self,
        node: &BbNode,
        _infeasible_var: VarIndex,
    ) -> Vec<(VarIndex, f64, f64)> {
        // Return the branching decisions that led to this infeasibility
        let mut conflicts = Vec::new();
        for record in &node.branching_history {
            let var = record.variable;
            if var < node.var_lower_bounds.len() && var < node.var_upper_bounds.len() {
                conflicts.push((var, node.var_lower_bounds[var], node.var_upper_bounds[var]));
            }
        }
        conflicts
    }
}

/// Tighten variable bounds from a single constraint row.
pub fn tighten_bounds_from_constraint(
    row_coeffs: &[(usize, f64)],
    rhs: f64,
    sense: ConstraintSense,
    lb: &mut [f64],
    ub: &mut [f64],
) -> usize {
    let n = lb.len().min(ub.len());
    let mut tightened = 0;

    for &(target_var, target_coeff) in row_coeffs {
        if target_var >= n || target_coeff.abs() < 1e-12 {
            continue;
        }

        let (min_act, max_act) = compute_activity_bounds_excluding(row_coeffs, target_var, lb, ub);

        if min_act.is_infinite() || max_act.is_infinite() {
            continue;
        }

        match sense {
            ConstraintSense::Le => {
                // sum <= rhs => target_coeff * x_target <= rhs - min_others
                if target_coeff > 0.0 {
                    let new_ub = (rhs - min_act) / target_coeff;
                    if new_ub < ub[target_var] - BOUND_TOLERANCE {
                        ub[target_var] = new_ub;
                        tightened += 1;
                    }
                } else {
                    let new_lb = (rhs - min_act) / target_coeff;
                    if new_lb > lb[target_var] + BOUND_TOLERANCE {
                        lb[target_var] = new_lb;
                        tightened += 1;
                    }
                }
            }
            ConstraintSense::Ge => {
                if target_coeff > 0.0 {
                    let new_lb = (rhs - max_act) / target_coeff;
                    if new_lb > lb[target_var] + BOUND_TOLERANCE {
                        lb[target_var] = new_lb;
                        tightened += 1;
                    }
                } else {
                    let new_ub = (rhs - max_act) / target_coeff;
                    if new_ub < ub[target_var] - BOUND_TOLERANCE {
                        ub[target_var] = new_ub;
                        tightened += 1;
                    }
                }
            }
            ConstraintSense::Eq => {
                if target_coeff > 0.0 {
                    let ub_bound = (rhs - min_act) / target_coeff;
                    let lb_bound = (rhs - max_act) / target_coeff;
                    if ub_bound < ub[target_var] - BOUND_TOLERANCE {
                        ub[target_var] = ub_bound;
                        tightened += 1;
                    }
                    if lb_bound > lb[target_var] + BOUND_TOLERANCE {
                        lb[target_var] = lb_bound;
                        tightened += 1;
                    }
                } else {
                    let lb_bound = (rhs - min_act) / target_coeff;
                    let ub_bound = (rhs - max_act) / target_coeff;
                    if ub_bound < ub[target_var] - BOUND_TOLERANCE {
                        ub[target_var] = ub_bound;
                        tightened += 1;
                    }
                    if lb_bound > lb[target_var] + BOUND_TOLERANCE {
                        lb[target_var] = lb_bound;
                        tightened += 1;
                    }
                }
            }
        }
    }
    tightened
}

/// Compute min and max activity bounds excluding a target variable.
fn compute_activity_bounds_excluding(
    row_coeffs: &[(usize, f64)],
    exclude_var: usize,
    lb: &[f64],
    ub: &[f64],
) -> (f64, f64) {
    let n = lb.len().min(ub.len());
    let mut min_act = 0.0f64;
    let mut max_act = 0.0f64;

    for &(var, coeff) in row_coeffs {
        if var == exclude_var || var >= n {
            continue;
        }
        if coeff > 0.0 {
            min_act += coeff * lb[var];
            max_act += coeff * ub[var];
        } else {
            min_act += coeff * ub[var];
            max_act += coeff * lb[var];
        }
    }
    (min_act, max_act)
}

/// If one variable in a complementarity pair is fixed > 0, fix the other to 0.
pub fn propagate_complementarity(node: &mut BbNode, model: &CompiledBilevelModel) -> usize {
    let mut fixed = 0;
    for &(a, b) in &model.complementarity_pairs {
        let a_in = a < node.var_lower_bounds.len() && a < node.var_upper_bounds.len();
        let b_in = b < node.var_lower_bounds.len() && b < node.var_upper_bounds.len();
        if !a_in || !b_in {
            continue;
        }

        if node.var_lower_bounds[a] > BOUND_TOLERANCE && node.var_upper_bounds[b] > BOUND_TOLERANCE
        {
            node.var_upper_bounds[b] = 0.0;
            node.var_lower_bounds[b] = node.var_lower_bounds[b].min(0.0);
            fixed += 1;
        }
        if node.var_lower_bounds[b] > BOUND_TOLERANCE && node.var_upper_bounds[a] > BOUND_TOLERANCE
        {
            node.var_upper_bounds[a] = 0.0;
            node.var_lower_bounds[a] = node.var_lower_bounds[a].min(0.0);
            fixed += 1;
        }
    }
    fixed
}

/// Check if any variable has lb > ub (infeasible).
pub fn detect_infeasibility(node: &BbNode) -> bool {
    let n = node.var_lower_bounds.len().min(node.var_upper_bounds.len());
    (0..n).any(|j| node.var_lower_bounds[j] > node.var_upper_bounds[j] + BOUND_TOLERANCE)
}

/// Compute min and max activity of a constraint row.
pub fn compute_activity_bounds(row_coeffs: &[(usize, f64)], lb: &[f64], ub: &[f64]) -> (f64, f64) {
    let n = lb.len().min(ub.len());
    let mut min_act = 0.0f64;
    let mut max_act = 0.0f64;

    for &(var, coeff) in row_coeffs {
        if var >= n {
            continue;
        }
        if coeff > 0.0 {
            min_act += coeff * lb[var];
            max_act += coeff * ub[var];
        } else {
            min_act += coeff * ub[var];
            max_act += coeff * lb[var];
        }
    }
    (min_act, max_act)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinLpSolver;

    fn make_model() -> CompiledBilevelModel {
        let bilevel = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a: SparseMatrix::new(1, 1),
            lower_b: vec![5.0],
            lower_linking_b: SparseMatrix::new(1, 1),
            upper_constraints_a: SparseMatrix::new(1, 2),
            upper_constraints_b: vec![10.0],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 1,
        };
        let mut m = CompiledBilevelModel::new(bilevel);
        m.integer_vars = vec![0];
        m
    }

    #[test]
    fn test_preprocessor_new() {
        let cfg = SolverConfig::default();
        let pp = NodePreprocessor::new(&cfg);
        assert!(pp.enable_domain_propagation);
        assert!(pp.enable_constraint_propagation);
    }

    #[test]
    fn test_detect_infeasibility_ok() {
        let model = make_model();
        let node = BbNode::root(&model);
        assert!(!detect_infeasibility(&node));
    }

    #[test]
    fn test_detect_infeasibility_bad() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        node.var_lower_bounds[0] = 10.0;
        node.var_upper_bounds[0] = 5.0;
        assert!(detect_infeasibility(&node));
    }

    #[test]
    fn test_complementarity_propagation() {
        let mut model = make_model();
        model.complementarity_pairs = vec![(0, 1)];
        let mut node = BbNode::root(&model);
        node.var_lower_bounds = vec![1.0, 0.0];
        node.var_upper_bounds = vec![5.0, 5.0];
        let fixed = propagate_complementarity(&mut node, &model);
        assert!(fixed > 0);
        assert!(node.var_upper_bounds[1] <= BOUND_TOLERANCE);
    }

    #[test]
    fn test_tighten_bounds_le() {
        // 2*x0 + 3*x1 <= 12
        let row = vec![(0, 2.0), (1, 3.0)];
        let mut lb = vec![0.0, 0.0];
        let mut ub = vec![10.0, 10.0];
        let t = tighten_bounds_from_constraint(&row, 12.0, ConstraintSense::Le, &mut lb, &mut ub);
        assert!(t > 0);
        // x0 <= (12 - 0) / 2 = 6, x1 <= (12 - 0) / 3 = 4
        assert!(ub[0] <= 6.0 + 1e-6);
        assert!(ub[1] <= 4.0 + 1e-6);
    }

    #[test]
    fn test_activity_bounds() {
        let row = vec![(0, 2.0), (1, -1.0)];
        let lb = vec![0.0, 0.0];
        let ub = vec![5.0, 3.0];
        let (min_a, max_a) = compute_activity_bounds(&row, &lb, &ub);
        // min: 2*0 + (-1)*3 = -3
        // max: 2*5 + (-1)*0 = 10
        assert!((min_a - (-3.0)).abs() < 1e-10);
        assert!((max_a - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_preprocess_basic() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        let cfg = SolverConfig::default();
        let pp = NodePreprocessor::new(&cfg);
        let solver = BuiltinLpSolver::new();
        let result = pp.preprocess(&mut node, &model, &solver);
        assert!(!result.infeasible_detected);
    }

    #[test]
    fn test_implied_bounds() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        node.branching_history.push(BranchRecord {
            variable: 0,
            direction: BranchDirection::Down,
            bound_value: 3.5,
            parent_lp_obj: 0.0,
        });
        let cfg = SolverConfig::default();
        let pp = NodePreprocessor::new(&cfg);
        let imps = pp.implied_bounds_from_branching(&mut node);
        // Should tighten ub to floor(3.5) = 3
        assert!(imps > 0 || node.var_upper_bounds[0] <= 3.5 + 1e-6);
    }
}
