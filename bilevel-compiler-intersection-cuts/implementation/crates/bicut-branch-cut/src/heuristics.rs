//! Primal heuristics: rounding, feasibility pump, diving, RINS, local branching,
//! bilevel feasibility repair.

use crate::node::{BbNode, BranchDirection, NodeStatus};
use crate::{
    fractionality, is_integer, CompiledBilevelModel, LpSolverInterface, SolverConfig,
    BOUND_TOLERANCE, INFINITY_BOUND,
};
use bicut_types::*;
use serde::{Deserialize, Serialize};

/// Result of running a primal heuristic.
#[derive(Debug, Clone)]
pub struct HeuristicResult {
    pub solution: Option<Vec<f64>>,
    pub objective: f64,
    pub is_bilevel_feasible: bool,
    pub heuristic_name: String,
}

impl HeuristicResult {
    pub fn failure(name: &str) -> Self {
        Self {
            solution: None,
            objective: INFINITY_BOUND,
            is_bilevel_feasible: false,
            heuristic_name: name.to_string(),
        }
    }

    pub fn success(sol: Vec<f64>, obj: f64, bilevel_feas: bool, name: &str) -> Self {
        Self {
            solution: Some(sol),
            objective: obj,
            is_bilevel_feasible: bilevel_feas,
            heuristic_name: name.to_string(),
        }
    }

    pub fn is_success(&self) -> bool {
        self.solution.is_some()
    }
}

/// Types of available heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeuristicType {
    SimpleRounding,
    FeasibilityPump,
    FractionalDiving,
    CoefficientDiving,
    RINS,
    LocalBranching,
    BilevelFeasibilityRepair,
}

/// Manages running primal heuristics.
#[derive(Debug, Clone)]
pub struct HeuristicManager {
    pub enabled_heuristics: Vec<HeuristicType>,
    pub best_solution: Option<Vec<f64>>,
    pub best_objective: f64,
    pub call_count: u64,
    pub success_count: u64,
    pub frequency: u32,
}

impl HeuristicManager {
    pub fn new(config: &SolverConfig) -> Self {
        let mut enabled = vec![
            HeuristicType::SimpleRounding,
            HeuristicType::FeasibilityPump,
            HeuristicType::FractionalDiving,
        ];
        if config.enable_heuristics {
            enabled.push(HeuristicType::RINS);
            enabled.push(HeuristicType::BilevelFeasibilityRepair);
        }
        Self {
            enabled_heuristics: enabled,
            best_solution: None,
            best_objective: INFINITY_BOUND,
            call_count: 0,
            success_count: 0,
            frequency: config.heuristic_frequency,
        }
    }

    /// Check if heuristics should run at this node count.
    pub fn should_run(&self, node_count: u64) -> bool {
        if self.frequency == 0 {
            return false;
        }
        node_count % self.frequency as u64 == 0
    }

    /// Run all enabled heuristics and return results.
    pub fn run_heuristics(
        &mut self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        incumbent: f64,
        lp_solver: &dyn LpSolverInterface,
    ) -> Vec<HeuristicResult> {
        self.call_count += 1;
        let mut results = Vec::new();

        for &htype in &self.enabled_heuristics.clone() {
            let result = match htype {
                HeuristicType::SimpleRounding => simple_rounding(node, model),
                HeuristicType::FeasibilityPump => feasibility_pump(node, model, lp_solver, 30),
                HeuristicType::FractionalDiving => fractional_diving(node, model, lp_solver, 20),
                HeuristicType::CoefficientDiving => coefficient_diving(node, model, lp_solver, 20),
                HeuristicType::RINS => {
                    if let Some(ref inc) = self.best_solution {
                        rins_heuristic(node, inc, model, lp_solver)
                    } else {
                        HeuristicResult::failure("RINS")
                    }
                }
                HeuristicType::LocalBranching => {
                    if let Some(ref inc) = self.best_solution {
                        local_branching(node, inc, model, lp_solver, 5)
                    } else {
                        HeuristicResult::failure("LocalBranching")
                    }
                }
                HeuristicType::BilevelFeasibilityRepair => {
                    bilevel_feasibility_repair(node, model, lp_solver)
                }
            };
            if result.is_success() && result.objective < incumbent - BOUND_TOLERANCE {
                self.success_count += 1;
            }
            results.push(result);
        }
        results
    }

    /// Update the manager's best known solution.
    pub fn update_best(&mut self, result: &HeuristicResult) -> bool {
        if let Some(ref sol) = result.solution {
            if result.objective < self.best_objective - BOUND_TOLERANCE {
                self.best_objective = result.objective;
                self.best_solution = Some(sol.clone());
                return true;
            }
        }
        false
    }

    pub fn get_stats(&self) -> (u64, u64) {
        (self.call_count, self.success_count)
    }
}

/// Check LP feasibility of a solution.
pub fn check_lp_feasibility(solution: &[f64], model: &CompiledBilevelModel) -> bool {
    let lp = &model.lp_relaxation;
    let n = lp.num_vars;
    let m = lp.num_constraints;

    // Check bounds
    for j in 0..n.min(solution.len()) {
        if solution[j] < lp.var_bounds[j].lower - 1e-4
            || solution[j] > lp.var_bounds[j].upper + 1e-4
        {
            return false;
        }
    }

    // Build dense and check constraints
    let mut row_lhs = vec![0.0; m];
    for entry in &lp.a_matrix.entries {
        if entry.row < m && entry.col < n && entry.col < solution.len() {
            row_lhs[entry.row] += entry.value * solution[entry.col];
        }
    }

    for i in 0..m {
        let ok = match lp.senses[i] {
            ConstraintSense::Le => row_lhs[i] <= lp.b_rhs[i] + 1e-4,
            ConstraintSense::Ge => row_lhs[i] >= lp.b_rhs[i] - 1e-4,
            ConstraintSense::Eq => (row_lhs[i] - lp.b_rhs[i]).abs() <= 1e-4,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Round integer variables and optionally check feasibility.
pub fn round_to_feasible(solution: &[f64], model: &CompiledBilevelModel) -> Option<Vec<f64>> {
    let mut rounded = solution.to_vec();
    for &var in &model.integer_vars {
        if var < rounded.len() {
            rounded[var] = rounded[var].round();
            // Clamp to bounds
            if var < model.lp_relaxation.var_bounds.len() {
                let lb = model.lp_relaxation.var_bounds[var].lower;
                let ub = model.lp_relaxation.var_bounds[var].upper;
                rounded[var] = rounded[var].max(lb).min(ub);
            }
        }
    }
    if check_lp_feasibility(&rounded, model) {
        Some(rounded)
    } else {
        None
    }
}

/// Compute the objective value for a solution.
fn compute_objective(solution: &[f64], model: &CompiledBilevelModel) -> f64 {
    let c = &model.lp_relaxation.c;
    solution.iter().zip(c.iter()).map(|(x, ci)| x * ci).sum()
}

/// Check bilevel complementarity feasibility.
fn check_bilevel_complementarity(solution: &[f64], model: &CompiledBilevelModel) -> bool {
    for &(a, b) in &model.complementarity_pairs {
        if a < solution.len() && b < solution.len() {
            if solution[a] * solution[b] > 1e-4 {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Heuristic implementations
// ---------------------------------------------------------------------------

/// Simple rounding heuristic: round fractional integers to nearest integer.
pub fn simple_rounding(node: &BbNode, model: &CompiledBilevelModel) -> HeuristicResult {
    match round_to_feasible(&node.lp_solution, model) {
        Some(sol) => {
            let obj = compute_objective(&sol, model);
            let bilevel = check_bilevel_complementarity(&sol, model);
            HeuristicResult::success(sol, obj, bilevel, "SimpleRounding")
        }
        None => HeuristicResult::failure("SimpleRounding"),
    }
}

/// Feasibility pump: iterate between rounding and LP projection.
pub fn feasibility_pump(
    node: &BbNode,
    model: &CompiledBilevelModel,
    lp_solver: &dyn LpSolverInterface,
    max_iter: u32,
) -> HeuristicResult {
    let n = model.num_vars;
    let mut current = node.lp_solution.clone();

    for _iter in 0..max_iter {
        // Step 1: Round integer variables
        let mut rounded = current.clone();
        for &var in &model.integer_vars {
            if var < rounded.len() {
                rounded[var] = rounded[var].round();
                if var < model.lp_relaxation.var_bounds.len() {
                    let lb = model.lp_relaxation.var_bounds[var].lower;
                    let ub = model.lp_relaxation.var_bounds[var].upper;
                    rounded[var] = rounded[var].max(lb).min(ub);
                }
            }
        }

        // Check if rounded solution is feasible
        if check_lp_feasibility(&rounded, model) {
            let obj = compute_objective(&rounded, model);
            let bilevel = check_bilevel_complementarity(&rounded, model);
            return HeuristicResult::success(rounded, obj, bilevel, "FeasibilityPump");
        }

        // Step 2: Solve LP minimizing distance to rounded point
        // Build modified objective: min sum |x_i - rounded_i| for integer vars
        let mut mod_c = vec![0.0; n];
        for &var in &model.integer_vars {
            if var < n {
                // Linearise |x - r|: if current > rounded, use +1, else -1
                mod_c[var] = if current.get(var).copied().unwrap_or(0.0) > rounded[var] {
                    1.0
                } else {
                    -1.0
                };
            }
        }

        let mut pump_lp = LpProblem::new(n, model.lp_relaxation.num_constraints);
        pump_lp.direction = OptDirection::Minimize;
        pump_lp.c = mod_c;
        pump_lp.a_matrix = model.lp_relaxation.a_matrix.clone();
        pump_lp.b_rhs = model.lp_relaxation.b_rhs.clone();
        pump_lp.senses = model.lp_relaxation.senses.clone();
        pump_lp.var_bounds = model.lp_relaxation.var_bounds.clone();

        let sol = lp_solver.solve_lp(&pump_lp);
        if sol.status != LpStatus::Optimal {
            break;
        }

        // Check for cycling
        let dist: f64 = model
            .integer_vars
            .iter()
            .filter(|&&v| v < sol.primal.len() && v < current.len())
            .map(|&v| (sol.primal[v] - current[v]).abs())
            .sum();

        current = sol.primal;
        if dist < 1e-6 {
            break;
        }
    }

    HeuristicResult::failure("FeasibilityPump")
}

/// Fractional diving: iteratively fix the most fractional variable.
pub fn fractional_diving(
    node: &BbNode,
    model: &CompiledBilevelModel,
    lp_solver: &dyn LpSolverInterface,
    max_depth: u32,
) -> HeuristicResult {
    let mut current_node = node.clone();

    for _depth in 0..max_depth {
        // Find most fractional integer variable
        let mut best_var = None;
        let mut best_frac = 0.0;
        for &var in &model.integer_vars {
            if var < current_node.lp_solution.len() {
                let frac = fractionality(current_node.lp_solution[var]);
                if frac > 1e-6 && frac > best_frac {
                    best_frac = frac;
                    best_var = Some(var);
                }
            }
        }

        let var = match best_var {
            Some(v) => v,
            None => {
                // All integer vars are integral - we have a solution!
                let obj = compute_objective(&current_node.lp_solution, model);
                let bilevel = check_bilevel_complementarity(&current_node.lp_solution, model);
                return HeuristicResult::success(
                    current_node.lp_solution,
                    obj,
                    bilevel,
                    "FractionalDiving",
                );
            }
        };

        // Round to nearest and fix
        let val = current_node.lp_solution[var];
        let rounded = val.round();
        let direction = if rounded <= val {
            BranchDirection::Down
        } else {
            BranchDirection::Up
        };

        current_node = current_node.create_child(u64::MAX, var, direction, rounded);
        let status = current_node.solve_lp(model, lp_solver);
        if status != LpStatus::Optimal {
            break;
        }
    }

    HeuristicResult::failure("FractionalDiving")
}

/// Coefficient diving: fix variable with smallest objective coefficient impact.
pub fn coefficient_diving(
    node: &BbNode,
    model: &CompiledBilevelModel,
    lp_solver: &dyn LpSolverInterface,
    max_depth: u32,
) -> HeuristicResult {
    let mut current_node = node.clone();
    let c = &model.lp_relaxation.c;

    for _depth in 0..max_depth {
        let mut best_var = None;
        let mut best_score = f64::INFINITY;

        for &var in &model.integer_vars {
            if var < current_node.lp_solution.len() {
                let frac = fractionality(current_node.lp_solution[var]);
                if frac > 1e-6 {
                    let coeff = if var < c.len() { c[var].abs() } else { 0.0 };
                    let score = coeff * frac;
                    if score < best_score {
                        best_score = score;
                        best_var = Some(var);
                    }
                }
            }
        }

        let var = match best_var {
            Some(v) => v,
            None => {
                let obj = compute_objective(&current_node.lp_solution, model);
                let bilevel = check_bilevel_complementarity(&current_node.lp_solution, model);
                return HeuristicResult::success(
                    current_node.lp_solution,
                    obj,
                    bilevel,
                    "CoefficientDiving",
                );
            }
        };

        let val = current_node.lp_solution[var];
        let rounded = val.round();
        let direction = if rounded <= val {
            BranchDirection::Down
        } else {
            BranchDirection::Up
        };
        current_node = current_node.create_child(u64::MAX, var, direction, rounded);
        let status = current_node.solve_lp(model, lp_solver);
        if status != LpStatus::Optimal {
            break;
        }
    }

    HeuristicResult::failure("CoefficientDiving")
}

/// RINS: fix variables where LP relaxation and incumbent agree.
pub fn rins_heuristic(
    node: &BbNode,
    incumbent: &[f64],
    model: &CompiledBilevelModel,
    lp_solver: &dyn LpSolverInterface,
) -> HeuristicResult {
    let n = model.num_vars;
    let mut fixed_node = node.clone();
    let mut num_fixed = 0;

    for &var in &model.integer_vars {
        if var >= node.lp_solution.len() || var >= incumbent.len() {
            continue;
        }
        let lp_val = node.lp_solution[var].round();
        let inc_val = incumbent[var].round();
        if (lp_val - inc_val).abs() < 0.5 {
            // They agree: fix the variable
            if var < fixed_node.var_lower_bounds.len() {
                fixed_node.var_lower_bounds[var] = lp_val;
                fixed_node.var_upper_bounds[var] = lp_val;
                num_fixed += 1;
            }
        }
    }

    if num_fixed == 0 {
        return HeuristicResult::failure("RINS");
    }

    let status = fixed_node.solve_lp(model, lp_solver);
    if status == LpStatus::Optimal {
        if let Some(rounded) = round_to_feasible(&fixed_node.lp_solution, model) {
            let obj = compute_objective(&rounded, model);
            let bilevel = check_bilevel_complementarity(&rounded, model);
            return HeuristicResult::success(rounded, obj, bilevel, "RINS");
        }
    }

    HeuristicResult::failure("RINS")
}

/// Local branching: search k-neighborhood of incumbent.
pub fn local_branching(
    node: &BbNode,
    incumbent: &[f64],
    model: &CompiledBilevelModel,
    lp_solver: &dyn LpSolverInterface,
    k: usize,
) -> HeuristicResult {
    let n = model.num_vars;

    // Fix all but k integer variables to their incumbent values
    let mut free_vars: Vec<VarIndex> = model.integer_vars.clone();
    // Sort by fractionality in the LP solution (most fractional = most likely to change)
    free_vars.sort_by(|&a, &b| {
        let fa = if a < node.lp_solution.len() {
            fractionality(node.lp_solution[a])
        } else {
            0.0
        };
        let fb = if b < node.lp_solution.len() {
            fractionality(node.lp_solution[b])
        } else {
            0.0
        };
        fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut restricted = node.clone();
    for (i, &var) in free_vars.iter().enumerate() {
        if i >= k && var < incumbent.len() {
            let val = incumbent[var].round();
            if var < restricted.var_lower_bounds.len() {
                restricted.var_lower_bounds[var] = val;
                restricted.var_upper_bounds[var] = val;
            }
        }
    }

    let status = restricted.solve_lp(model, lp_solver);
    if status == LpStatus::Optimal {
        if let Some(rounded) = round_to_feasible(&restricted.lp_solution, model) {
            let obj = compute_objective(&rounded, model);
            let bilevel = check_bilevel_complementarity(&rounded, model);
            return HeuristicResult::success(rounded, obj, bilevel, "LocalBranching");
        }
    }

    HeuristicResult::failure("LocalBranching")
}

/// Try to repair bilevel feasibility by adjusting complementarity-violating variables.
pub fn bilevel_feasibility_repair(
    node: &BbNode,
    model: &CompiledBilevelModel,
    lp_solver: &dyn LpSolverInterface,
) -> HeuristicResult {
    let mut repaired = node.lp_solution.clone();

    // For each violated complementarity pair, set the smaller variable to 0
    for &(a, b) in &model.complementarity_pairs {
        if a >= repaired.len() || b >= repaired.len() {
            continue;
        }
        if repaired[a].abs() > 1e-6 && repaired[b].abs() > 1e-6 {
            if repaired[a] < repaired[b] {
                repaired[a] = 0.0;
            } else {
                repaired[b] = 0.0;
            }
        }
    }

    // Round integer variables
    for &var in &model.integer_vars {
        if var < repaired.len() {
            repaired[var] = repaired[var].round();
            if var < model.lp_relaxation.var_bounds.len() {
                let lb = model.lp_relaxation.var_bounds[var].lower;
                let ub = model.lp_relaxation.var_bounds[var].upper;
                repaired[var] = repaired[var].max(lb).min(ub);
            }
        }
    }

    // Check if repair created a feasible solution
    if check_lp_feasibility(&repaired, model) {
        let obj = compute_objective(&repaired, model);
        let bilevel = check_bilevel_complementarity(&repaired, model);
        return HeuristicResult::success(repaired, obj, bilevel, "BilevelRepair");
    }

    // Try solving a restricted LP around the repaired point
    let mut restricted = node.clone();
    for &var in &model.integer_vars {
        if var < repaired.len() && var < restricted.var_lower_bounds.len() {
            restricted.var_lower_bounds[var] = repaired[var];
            restricted.var_upper_bounds[var] = repaired[var];
        }
    }
    let status = restricted.solve_lp(model, lp_solver);
    if status == LpStatus::Optimal {
        let sol = &restricted.lp_solution;
        if check_lp_feasibility(sol, model) {
            let obj = compute_objective(sol, model);
            let bilevel = check_bilevel_complementarity(sol, model);
            return HeuristicResult::success(sol.clone(), obj, bilevel, "BilevelRepair");
        }
    }

    HeuristicResult::failure("BilevelRepair")
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
    fn test_heuristic_manager_new() {
        let cfg = SolverConfig::default();
        let mgr = HeuristicManager::new(&cfg);
        assert!(!mgr.enabled_heuristics.is_empty());
    }

    #[test]
    fn test_should_run() {
        let cfg = SolverConfig::default();
        let mgr = HeuristicManager::new(&cfg);
        assert!(mgr.should_run(0));
        assert!(!mgr.should_run(1));
        assert!(mgr.should_run(mgr.frequency as u64));
    }

    #[test]
    fn test_simple_rounding_integral() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        node.lp_solution = vec![3.0, 1.0];
        let result = simple_rounding(&node, &model);
        // Already integral; depends on feasibility
        assert_eq!(result.heuristic_name, "SimpleRounding");
    }

    #[test]
    fn test_simple_rounding_fractional() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        node.lp_solution = vec![2.5, 1.0];
        let result = simple_rounding(&node, &model);
        assert_eq!(result.heuristic_name, "SimpleRounding");
    }

    #[test]
    fn test_check_lp_feasibility_trivial() {
        let model = make_model();
        // Trivial model with no real constraints
        let sol = vec![0.0, 0.0];
        assert!(check_lp_feasibility(&sol, &model));
    }

    #[test]
    fn test_heuristic_result_failure() {
        let r = HeuristicResult::failure("test");
        assert!(!r.is_success());
        assert!(r.objective > 1e10);
    }

    #[test]
    fn test_heuristic_result_success() {
        let r = HeuristicResult::success(vec![1.0], 5.0, true, "test");
        assert!(r.is_success());
        assert!((r.objective - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_round_to_feasible() {
        let model = make_model();
        let sol = vec![0.0, 0.0];
        let rounded = round_to_feasible(&sol, &model);
        assert!(rounded.is_some());
    }
}
