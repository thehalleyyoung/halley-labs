//! Branching strategies for the bilevel branch-and-cut solver.
//!
//! This module provides several branching variable-selection heuristics:
//! most-fractional, strong branching, pseudocost branching, reliability
//! branching (a hybrid of strong and pseudocost), a depth-adaptive hybrid
//! strategy, and a complementarity-aware strategy that prioritises variables
//! involved in complementarity constraints of the bilevel reformulation.

use crate::node::{BbNode, BranchDirection, BranchRecord, NodeStatus};
use crate::statistics::BranchingStats;
use crate::{
    fractionality, is_integer, BranchingStrategyType, CompiledBilevelModel, LpSolverInterface,
    SolverConfig, INFINITY_BOUND, INT_TOLERANCE,
};
use bicut_types::{LpStatus, VarBound, VarIndex};

// ---------------------------------------------------------------------------
// BranchingDecision
// ---------------------------------------------------------------------------

/// The result of a branching variable-selection strategy.
#[derive(Debug, Clone)]
pub struct BranchingDecision {
    /// Index of the variable to branch on.
    pub variable: VarIndex,
    /// Current LP-relaxation value of that variable.
    pub value: f64,
    /// Score assigned by the strategy (higher is better).
    pub score: f64,
    /// Preferred branching direction, if any.
    pub direction: Option<BranchDirection>,
}

// ---------------------------------------------------------------------------
// BranchingStrategy trait
// ---------------------------------------------------------------------------

/// Trait implemented by all branching-variable selectors.
pub trait BranchingStrategy: Send + Sync {
    /// Select a variable for branching at the given B&B node.
    ///
    /// Returns `None` when no fractional integer variable exists (the LP
    /// solution is already integer-feasible for all integer-constrained
    /// variables).
    fn select_variable(
        &self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
        stats: &mut BranchingStats,
    ) -> Option<BranchingDecision>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the LP primal solution from the node's warm-start data, or return
/// an empty slice if no warm-start is available.
fn node_primal(node: &BbNode) -> &[f64] {
    match &node.warm_start {
        Some(ws) => &ws.primal_solution,
        None => &[],
    }
}

/// Collect fractional integer-constrained variables from the node's warm-start
/// solution, sorted by proximity to 0.5 (most fractional first).
///
/// Returns at most `max` candidates as `(VarIndex, lp_value)` pairs.
pub fn compute_branching_candidates(
    node: &BbNode,
    model: &CompiledBilevelModel,
    max: usize,
) -> Vec<(VarIndex, f64)> {
    let primal = node_primal(node);
    if primal.is_empty() {
        return Vec::new();
    }

    let mut candidates: Vec<(VarIndex, f64, f64)> = model
        .integer_vars
        .iter()
        .filter_map(|&var| {
            if var >= primal.len() {
                return None;
            }
            let val = primal[var];
            if is_integer(val, INT_TOLERANCE) {
                return None;
            }
            let frac = fractionality(val);
            // Distance from the ideal 0.5 — smaller means more fractional.
            let dist = (frac - 0.5).abs();
            Some((var, val, dist))
        })
        .collect();

    // Sort: most fractional (smallest distance to 0.5) first.
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max);
    candidates.into_iter().map(|(v, val, _)| (v, val)).collect()
}

/// Evaluate a candidate variable via strong branching: solve the down-branch
/// and up-branch child LPs and return `(delta_down, delta_up)` —
/// the objective-value improvements relative to the parent node.
///
/// If a child LP is infeasible the corresponding entry is [`INFINITY_BOUND`]
/// so that the product score correctly prefers variables that prune a subtree.
pub fn evaluate_strong_branch(
    node: &BbNode,
    var: VarIndex,
    val: f64,
    model: &CompiledBilevelModel,
    solver: &dyn LpSolverInterface,
) -> (f64, f64) {
    let parent_obj = node.lower_bound;

    let obj_down = solve_child_lp(node, var, val, BranchDirection::Down, model, solver);
    let obj_up = solve_child_lp(node, var, val, BranchDirection::Up, model, solver);

    let delta_down = if obj_down >= INFINITY_BOUND {
        INFINITY_BOUND
    } else {
        (obj_down - parent_obj).max(0.0)
    };
    let delta_up = if obj_up >= INFINITY_BOUND {
        INFINITY_BOUND
    } else {
        (obj_up - parent_obj).max(0.0)
    };

    (delta_down, delta_up)
}

/// Build a child LP by cloning the relaxation, applying all accumulated bound
/// changes from the node, then imposing the new branching bound, and solving.
fn solve_child_lp(
    node: &BbNode,
    var: VarIndex,
    val: f64,
    direction: BranchDirection,
    model: &CompiledBilevelModel,
    solver: &dyn LpSolverInterface,
) -> f64 {
    let mut child_lp = model.lp_relaxation.clone();

    // Replay bound changes along the path to this node.
    for j in 0..child_lp.var_bounds.len().min(node.var_lower_bounds.len()) {
        child_lp.var_bounds[j].lower = child_lp.var_bounds[j].lower.max(node.var_lower_bounds[j]);
    }
    for j in 0..child_lp.var_bounds.len().min(node.var_upper_bounds.len()) {
        child_lp.var_bounds[j].upper = child_lp.var_bounds[j].upper.min(node.var_upper_bounds[j]);
    }

    // Apply the new branching bound.
    if var < child_lp.var_bounds.len() {
        match direction {
            BranchDirection::Down => {
                let new_ub = val.floor();
                child_lp.var_bounds[var].upper = child_lp.var_bounds[var].upper.min(new_ub);
            }
            BranchDirection::Up => {
                let new_lb = val.ceil();
                child_lp.var_bounds[var].lower = child_lp.var_bounds[var].lower.max(new_lb);
            }
        }
    }

    // Bounds inconsistency → infeasible.
    if var < child_lp.var_bounds.len()
        && child_lp.var_bounds[var].lower > child_lp.var_bounds[var].upper + 1e-8
    {
        return INFINITY_BOUND;
    }

    // Warm-start from parent basis when available.
    let sol = match &node.warm_start {
        Some(ws) if !ws.basis.is_empty() => solver.solve_lp_with_basis(&child_lp, &ws.basis),
        _ => solver.solve_lp(&child_lp),
    };

    match sol.status {
        LpStatus::Optimal => sol.objective,
        _ => INFINITY_BOUND,
    }
}

/// SCIP-style product score: heavily weight the smaller improvement, with a
/// small contribution from the larger one to break ties.
fn product_score(delta_down: f64, delta_up: f64) -> f64 {
    let eps = 1e-6;
    let d = delta_down.max(eps);
    let u = delta_up.max(eps);
    (1.0 - eps) * d.min(u) + eps * d.max(u)
}

/// Choose a branching direction using pseudocost estimates.
fn direction_from_pseudocosts(
    stats: &BranchingStats,
    var: VarIndex,
    frac: f64,
) -> Option<BranchDirection> {
    let pc_up = stats.pseudocost_up.get(&var).copied().unwrap_or(1.0);
    let pc_down = stats.pseudocost_down.get(&var).copied().unwrap_or(1.0);
    let est_up = pc_up * (1.0 - frac);
    let est_down = pc_down * frac;
    if est_up >= est_down {
        Some(BranchDirection::Up)
    } else {
        Some(BranchDirection::Down)
    }
}

/// Choose a branching direction by proximity to the nearest integer.
fn direction_from_value(val: f64) -> Option<BranchDirection> {
    if val - val.floor() <= 0.5 {
        Some(BranchDirection::Down)
    } else {
        Some(BranchDirection::Up)
    }
}

// ---------------------------------------------------------------------------
// MostFractionalBranching
// ---------------------------------------------------------------------------

/// Select the integer variable whose LP relaxation value is closest to 0.5
/// (i.e. the "most fractional" variable).
#[derive(Debug, Clone, Default)]
pub struct MostFractionalBranching;

impl MostFractionalBranching {
    pub fn new() -> Self {
        Self
    }
}

impl BranchingStrategy for MostFractionalBranching {
    fn select_variable(
        &self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        _lp_solver: &dyn LpSolverInterface,
        _stats: &mut BranchingStats,
    ) -> Option<BranchingDecision> {
        let candidates = compute_branching_candidates(node, model, usize::MAX);
        // candidates[0] is the most fractional.
        candidates.first().map(|&(var, val)| {
            let frac = fractionality(val);
            BranchingDecision {
                variable: var,
                value: val,
                score: 0.5 - (frac - 0.5).abs(),
                direction: direction_from_value(val),
            }
        })
    }

    fn name(&self) -> &str {
        "MostFractional"
    }
}

// ---------------------------------------------------------------------------
// StrongBranching
// ---------------------------------------------------------------------------

/// Full strong branching: for the top `max_candidates` fractional variables,
/// solve both child LPs and pick the variable with the best product score.
#[derive(Debug, Clone)]
pub struct StrongBranching {
    pub max_candidates: usize,
}

impl StrongBranching {
    pub fn new(max_candidates: usize) -> Self {
        Self { max_candidates }
    }
}

impl Default for StrongBranching {
    fn default() -> Self {
        Self { max_candidates: 10 }
    }
}

impl BranchingStrategy for StrongBranching {
    fn select_variable(
        &self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
        stats: &mut BranchingStats,
    ) -> Option<BranchingDecision> {
        let candidates = compute_branching_candidates(node, model, self.max_candidates);
        if candidates.is_empty() {
            return None;
        }

        let mut best: Option<BranchingDecision> = None;

        for &(var, val) in &candidates {
            let (delta_down, delta_up) = evaluate_strong_branch(node, var, val, model, lp_solver);
            stats.strong_branching_calls += 1;

            let score = product_score(delta_down, delta_up);

            // Record pseudocost observations from the strong-branching LPs.
            let frac = fractionality(val);
            if delta_down < INFINITY_BOUND {
                stats.record_down(var, delta_down, frac);
            }
            if delta_up < INFINITY_BOUND {
                stats.record_up(var, delta_up, 1.0 - frac);
            }

            let direction = if delta_down >= delta_up {
                Some(BranchDirection::Up)
            } else {
                Some(BranchDirection::Down)
            };

            let dominated = match &best {
                Some(b) => score <= b.score,
                None => false,
            };
            if !dominated {
                best = Some(BranchingDecision {
                    variable: var,
                    value: val,
                    score,
                    direction,
                });
            }
        }

        best
    }

    fn name(&self) -> &str {
        "StrongBranching"
    }
}

// ---------------------------------------------------------------------------
// PseudocostBranching
// ---------------------------------------------------------------------------

/// Pseudocost branching: use historically accumulated pseudocosts to estimate
/// the objective improvement for each candidate and pick the best.
#[derive(Debug, Clone, Default)]
pub struct PseudocostBranching;

impl PseudocostBranching {
    pub fn new() -> Self {
        Self
    }
}

impl BranchingStrategy for PseudocostBranching {
    fn select_variable(
        &self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        _lp_solver: &dyn LpSolverInterface,
        stats: &mut BranchingStats,
    ) -> Option<BranchingDecision> {
        let candidates = compute_branching_candidates(node, model, usize::MAX);
        if candidates.is_empty() {
            return None;
        }

        let mut best: Option<BranchingDecision> = None;

        for &(var, val) in &candidates {
            let frac = fractionality(val);
            let score = stats.pseudocost_score(var, frac);
            let direction = direction_from_pseudocosts(stats, var, frac);

            let dominated = match &best {
                Some(b) => score <= b.score,
                None => false,
            };
            if !dominated {
                best = Some(BranchingDecision {
                    variable: var,
                    value: val,
                    score,
                    direction,
                });
            }
        }

        best
    }

    fn name(&self) -> &str {
        "Pseudocost"
    }
}

// ---------------------------------------------------------------------------
// ReliabilityBranching
// ---------------------------------------------------------------------------

/// Reliability branching: use pseudocosts when they are considered reliable
/// (enough historical observations), and fall back to strong branching for
/// variables with insufficient data.
#[derive(Debug, Clone)]
pub struct ReliabilityBranching {
    pub reliability_threshold: u64,
    pub max_strong_candidates: usize,
}

impl ReliabilityBranching {
    pub fn new(reliability_threshold: u64, max_strong_candidates: usize) -> Self {
        Self {
            reliability_threshold,
            max_strong_candidates,
        }
    }
}

impl Default for ReliabilityBranching {
    fn default() -> Self {
        Self {
            reliability_threshold: 8,
            max_strong_candidates: 10,
        }
    }
}

impl BranchingStrategy for ReliabilityBranching {
    fn select_variable(
        &self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
        stats: &mut BranchingStats,
    ) -> Option<BranchingDecision> {
        let all_candidates = compute_branching_candidates(node, model, usize::MAX);
        if all_candidates.is_empty() {
            return None;
        }

        let mut best: Option<BranchingDecision> = None;
        let mut strong_count: usize = 0;

        for &(var, val) in &all_candidates {
            let frac = fractionality(val);

            let score = if stats.is_reliable(var, self.reliability_threshold) {
                // Pseudocost data is reliable — use it directly.
                stats.pseudocost_score(var, frac)
            } else if strong_count < self.max_strong_candidates {
                // Not enough history — evaluate with strong branching.
                strong_count += 1;
                let (delta_down, delta_up) =
                    evaluate_strong_branch(node, var, val, model, lp_solver);
                stats.strong_branching_calls += 1;

                if delta_down < INFINITY_BOUND {
                    stats.record_down(var, delta_down, frac);
                }
                if delta_up < INFINITY_BOUND {
                    stats.record_up(var, delta_up, 1.0 - frac);
                }

                product_score(delta_down, delta_up)
            } else {
                // Budget exhausted — use whatever pseudocost data we have.
                stats.pseudocost_score(var, frac)
            };

            let direction = direction_from_pseudocosts(stats, var, frac);

            let dominated = match &best {
                Some(b) => score <= b.score,
                None => false,
            };
            if !dominated {
                best = Some(BranchingDecision {
                    variable: var,
                    value: val,
                    score,
                    direction,
                });
            }
        }

        best
    }

    fn name(&self) -> &str {
        "Reliability"
    }
}

// ---------------------------------------------------------------------------
// HybridBranching
// ---------------------------------------------------------------------------

/// Depth-adaptive hybrid: uses strong branching near the root of the tree
/// (where its cost is amortised over many descendant nodes) and switches
/// to pseudocost branching deeper in the tree.
#[derive(Debug, Clone)]
pub struct HybridBranching {
    /// Maximum depth at which strong branching is used.
    pub strong_depth_limit: u32,
    /// Maximum candidates to evaluate with strong branching.
    pub max_strong_candidates: usize,
}

impl HybridBranching {
    pub fn new(strong_depth_limit: u32, max_strong_candidates: usize) -> Self {
        Self {
            strong_depth_limit,
            max_strong_candidates,
        }
    }
}

impl Default for HybridBranching {
    fn default() -> Self {
        Self {
            strong_depth_limit: 5,
            max_strong_candidates: 10,
        }
    }
}

impl BranchingStrategy for HybridBranching {
    fn select_variable(
        &self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
        stats: &mut BranchingStats,
    ) -> Option<BranchingDecision> {
        if node.depth <= self.strong_depth_limit {
            let strong = StrongBranching::new(self.max_strong_candidates);
            strong.select_variable(node, model, lp_solver, stats)
        } else {
            let pseudo = PseudocostBranching::new();
            pseudo.select_variable(node, model, lp_solver, stats)
        }
    }

    fn name(&self) -> &str {
        "Hybrid"
    }
}

// ---------------------------------------------------------------------------
// ComplementarityBranching
// ---------------------------------------------------------------------------

/// Branching strategy that prioritises variables appearing in complementarity
/// pairs of the bilevel reformulation.
///
/// Among those variables, the most-fractional rule is used as a tie-breaker.
/// If no complementarity variable is fractional, the strategy falls back to
/// the regular most-fractional rule over all integer variables.
#[derive(Debug, Clone, Default)]
pub struct ComplementarityBranching;

impl ComplementarityBranching {
    pub fn new() -> Self {
        Self
    }
}

impl BranchingStrategy for ComplementarityBranching {
    fn select_variable(
        &self,
        node: &BbNode,
        model: &CompiledBilevelModel,
        _lp_solver: &dyn LpSolverInterface,
        _stats: &mut BranchingStats,
    ) -> Option<BranchingDecision> {
        let all_candidates = compute_branching_candidates(node, model, usize::MAX);
        if all_candidates.is_empty() {
            return None;
        }

        // Partition into complementarity and non-complementarity candidates.
        let mut comp_candidates: Vec<(VarIndex, f64)> = Vec::new();
        let mut other_candidates: Vec<(VarIndex, f64)> = Vec::new();

        for &(var, val) in &all_candidates {
            if model.is_complementarity_var(var) {
                comp_candidates.push((var, val));
            } else {
                other_candidates.push((var, val));
            }
        }

        // Prefer complementarity variables; fall back to the rest.
        let source = if !comp_candidates.is_empty() {
            &comp_candidates
        } else {
            &other_candidates
        };

        let mut best: Option<BranchingDecision> = None;

        for (rank, &(var, val)) in source.iter().enumerate() {
            let frac = fractionality(val);
            let base_score = 0.5 - (frac - 0.5).abs();
            let comp_bonus = if model.is_complementarity_var(var) {
                1.0
            } else {
                0.0
            };
            // Rank penalty so earlier (more fractional) entries are preferred.
            let score = comp_bonus + base_score - rank as f64 * 1e-8;

            let direction = complementarity_direction(var, val, model, node);

            let dominated = match &best {
                Some(b) => score <= b.score,
                None => false,
            };
            if !dominated {
                best = Some(BranchingDecision {
                    variable: var,
                    value: val,
                    score,
                    direction,
                });
            }
        }

        best
    }

    fn name(&self) -> &str {
        "Complementarity"
    }
}

/// For a complementarity variable, decide the preferred branching direction
/// by examining its partner.  If the partner is near zero we prefer driving
/// *this* variable up (away from zero), and vice versa.
fn complementarity_direction(
    var: VarIndex,
    val: f64,
    model: &CompiledBilevelModel,
    node: &BbNode,
) -> Option<BranchDirection> {
    let primal = node_primal(node);
    if !primal.is_empty() {
        for &(a, b) in &model.complementarity_pairs {
            let partner = if a == var {
                Some(b)
            } else if b == var {
                Some(a)
            } else {
                None
            };
            if let Some(p) = partner {
                if p < primal.len() {
                    let partner_val = primal[p];
                    if partner_val.abs() < INT_TOLERANCE {
                        return Some(BranchDirection::Up);
                    }
                    if partner_val > 1.0 {
                        return Some(BranchDirection::Down);
                    }
                }
            }
        }
    }
    direction_from_value(val)
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a boxed branching strategy from a configuration enum.
pub fn create_branching_strategy(
    strategy_type: BranchingStrategyType,
    config: &SolverConfig,
) -> Box<dyn BranchingStrategy> {
    match strategy_type {
        BranchingStrategyType::MostFractional => Box::new(MostFractionalBranching::new()),
        BranchingStrategyType::StrongBranching => {
            Box::new(StrongBranching::new(config.strong_branching_candidates))
        }
        BranchingStrategyType::PseudocostBranching => Box::new(PseudocostBranching::new()),
        BranchingStrategyType::ReliabilityBranching => Box::new(ReliabilityBranching::new(
            config.reliability_threshold,
            config.strong_branching_candidates,
        )),
        BranchingStrategyType::Hybrid => {
            Box::new(HybridBranching::new(5, config.strong_branching_candidates))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::WarmStartInfo;
    use crate::BuiltinLpSolver;
    use bicut_types::*;

    /// Build a tiny compiled bilevel model with `n` variables, the first
    /// `n_int` of which are integer-constrained, and optional complementarity
    /// pairs.
    fn make_test_model(
        n: usize,
        n_int: usize,
        comp_pairs: Vec<(VarIndex, VarIndex)>,
    ) -> CompiledBilevelModel {
        let bilevel = BilevelProblem {
            upper_obj_c_x: vec![1.0; n],
            upper_obj_c_y: vec![],
            lower_obj_c: vec![],
            lower_a: SparseMatrix::new(0, 0),
            lower_b: vec![],
            lower_linking_b: SparseMatrix::new(0, n),
            upper_constraints_a: SparseMatrix::new(0, n),
            upper_constraints_b: vec![],
            num_upper_vars: n,
            num_lower_vars: 0,
            num_lower_constraints: 0,
            num_upper_constraints: 0,
        };

        let mut lp = LpProblem::new(n, 0);
        lp.direction = OptDirection::Minimize;
        lp.c = vec![1.0; n];
        for i in 0..n {
            lp.var_bounds[i] = VarBound {
                lower: 0.0,
                upper: 10.0,
            };
        }

        CompiledBilevelModel {
            lp_relaxation: lp,
            integer_vars: (0..n_int).collect(),
            complementarity_pairs: comp_pairs,
            bilevel,
            num_vars: n,
            num_constraints: 0,
            var_names: (0..n).map(|i| format!("x{}", i)).collect(),
        }
    }

    /// Build a root node whose warm-start carries the given primal values.
    fn make_node_with_solution(values: Vec<f64>) -> BbNode {
        let n = values.len();
        let model = make_test_model(n, n, vec![]);
        let mut node = BbNode::root(&model);
        node.lp_solution = values.clone();
        node.lower_bound = values.iter().sum();
        node.warm_start = Some(WarmStartInfo {
            basis: vec![BasisStatus::Basic; n],
            primal_solution: values,
            dual_solution: vec![],
        });
        node
    }

    // --- Test 1: compute_branching_candidates returns fractional vars ---

    #[test]
    fn test_compute_candidates_returns_fractional_vars() {
        let model = make_test_model(4, 3, vec![]);
        // x0=1.5 (frac), x1=2.0 (int), x2=0.3 (frac), x3=4.7 (not int-constrained)
        let node = make_node_with_solution(vec![1.5, 2.0, 0.3, 4.7]);
        let cands = compute_branching_candidates(&node, &model, 10);
        assert_eq!(cands.len(), 2);
        // Most fractional first (0.5 is closest to 0.5).
        assert_eq!(cands[0].0, 0);
    }

    // --- Test 2: empty when integer-feasible ---

    #[test]
    fn test_compute_candidates_empty_when_integer_feasible() {
        let model = make_test_model(3, 3, vec![]);
        let node = make_node_with_solution(vec![1.0, 2.0, 3.0]);
        let cands = compute_branching_candidates(&node, &model, 10);
        assert!(cands.is_empty());
    }

    // --- Test 3: MostFractionalBranching picks closest-to-half ---

    #[test]
    fn test_most_fractional_selects_closest_to_half() {
        let model = make_test_model(3, 3, vec![]);
        let node = make_node_with_solution(vec![1.5, 2.3, 3.8]);
        let solver = BuiltinLpSolver::new();
        let mut stats = BranchingStats::new();
        let strategy = MostFractionalBranching::new();
        let dec = strategy
            .select_variable(&node, &model, &solver, &mut stats)
            .expect("should find a variable");
        assert_eq!(dec.variable, 0);
        assert!((dec.value - 1.5).abs() < 1e-10);
    }

    // --- Test 4: MostFractionalBranching returns None on all-integer ---

    #[test]
    fn test_most_fractional_returns_none_all_integer() {
        let model = make_test_model(2, 2, vec![]);
        let node = make_node_with_solution(vec![3.0, 7.0]);
        let solver = BuiltinLpSolver::new();
        let mut stats = BranchingStats::new();
        let strategy = MostFractionalBranching::new();
        assert!(strategy
            .select_variable(&node, &model, &solver, &mut stats)
            .is_none());
    }

    // --- Test 5: StrongBranching evaluates child LPs ---

    #[test]
    fn test_strong_branching_selects_variable() {
        let model = make_test_model(3, 3, vec![]);
        let node = make_node_with_solution(vec![1.5, 2.7, 0.4]);
        let solver = BuiltinLpSolver::new();
        let mut stats = BranchingStats::new();
        let strategy = StrongBranching::new(5);
        let dec = strategy
            .select_variable(&node, &model, &solver, &mut stats)
            .expect("should select a variable");
        assert!(stats.strong_branching_calls > 0);
        assert!(model.integer_vars.contains(&dec.variable));
    }

    // --- Test 6: PseudocostBranching with seeded stats ---

    #[test]
    fn test_pseudocost_branching_uses_stats() {
        let model = make_test_model(3, 3, vec![]);
        let node = make_node_with_solution(vec![1.5, 2.5, 0.5]);
        let solver = BuiltinLpSolver::new();
        let mut stats = BranchingStats::new();
        // Make variable 2 look very attractive.
        for _ in 0..10 {
            stats.record_up(2, 100.0, 0.5);
            stats.record_down(2, 100.0, 0.5);
        }
        let strategy = PseudocostBranching::new();
        let dec = strategy
            .select_variable(&node, &model, &solver, &mut stats)
            .expect("should select a variable");
        assert_eq!(dec.variable, 2);
    }

    // --- Test 7: ReliabilityBranching falls back to strong branching ---

    #[test]
    fn test_reliability_branching_uses_strong_for_unreliable() {
        let model = make_test_model(3, 3, vec![]);
        let node = make_node_with_solution(vec![1.5, 2.3, 0.7]);
        let solver = BuiltinLpSolver::new();
        let mut stats = BranchingStats::new();
        let strategy = ReliabilityBranching::new(8, 5);
        let dec = strategy
            .select_variable(&node, &model, &solver, &mut stats)
            .expect("should select a variable");
        assert!(stats.strong_branching_calls > 0);
        assert!(model.integer_vars.contains(&dec.variable));
    }

    // --- Test 8: HybridBranching switches by depth ---

    #[test]
    fn test_hybrid_branching_depth_switch() {
        let model = make_test_model(3, 3, vec![]);
        let solver = BuiltinLpSolver::new();

        // Root (depth 0) → strong branching path.
        let root = make_node_with_solution(vec![1.5, 2.5, 0.5]);
        let mut stats = BranchingStats::new();
        let strategy = HybridBranching::new(2, 5);
        let _ = strategy
            .select_variable(&root, &model, &solver, &mut stats)
            .expect("should select at root");
        assert!(
            stats.strong_branching_calls > 0,
            "root should use strong branching"
        );

        // Deep node (depth 10) → pseudocost path.
        let mut deep = make_node_with_solution(vec![1.5, 2.5, 0.5]);
        deep.depth = 10;
        let mut stats2 = BranchingStats::new();
        let _ = strategy
            .select_variable(&deep, &model, &solver, &mut stats2)
            .expect("should select at depth");
        assert_eq!(
            stats2.strong_branching_calls, 0,
            "deep node should use pseudocost, not strong"
        );
    }

    // --- Test 9: ComplementarityBranching prefers comp vars ---

    #[test]
    fn test_complementarity_branching_prioritises_comp_vars() {
        let model = make_test_model(4, 4, vec![(1, 2)]);
        // x0=0.5 (most fractional, not comp), x1=0.7, x2=0.3 (both comp), x3=5.0 (int)
        let node = make_node_with_solution(vec![0.5, 0.7, 0.3, 5.0]);
        let solver = BuiltinLpSolver::new();
        let mut stats = BranchingStats::new();
        let strategy = ComplementarityBranching::new();
        let dec = strategy
            .select_variable(&node, &model, &solver, &mut stats)
            .expect("should select a complementarity variable");
        assert!(
            dec.variable == 1 || dec.variable == 2,
            "expected comp var, got {}",
            dec.variable
        );
    }

    // --- Test 10: create_branching_strategy factory ---

    #[test]
    fn test_factory_creates_all_strategy_types() {
        let config = SolverConfig::default();
        let strategies = vec![
            BranchingStrategyType::MostFractional,
            BranchingStrategyType::StrongBranching,
            BranchingStrategyType::PseudocostBranching,
            BranchingStrategyType::ReliabilityBranching,
            BranchingStrategyType::Hybrid,
        ];
        let model = make_test_model(3, 3, vec![]);
        let node = make_node_with_solution(vec![1.5, 2.5, 0.5]);
        let solver = BuiltinLpSolver::new();
        for st in strategies {
            let strat = create_branching_strategy(st, &config);
            let mut stats = BranchingStats::new();
            let result = strat.select_variable(&node, &model, &solver, &mut stats);
            assert!(
                result.is_some(),
                "strategy {:?} should select a variable",
                st
            );
        }
    }
}
