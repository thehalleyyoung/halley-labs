//! Cut callback integration: bilevel intersection cut separation, cut round management,
//! cut propagation (global vs local), lazy constraint handling, callback statistics.

use crate::node::{BbNode, NodeStatus};
use crate::{
    fractionality, CompiledBilevelModel, Cut, CutType, LpSolverInterface, NodeId, SolverConfig,
    BOUND_TOLERANCE, INFINITY_BOUND,
};
use bicut_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics collected by the cut callback.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CutCallbackStats {
    pub rounds_total: u32,
    pub cuts_generated: u64,
    pub cuts_applied: u64,
    pub cuts_rejected_parallel: u64,
    pub cuts_rejected_efficacy: u64,
    pub cuts_rejected_duplicate: u64,
    pub bilevel_intersection_generated: u64,
    pub gomory_generated: u64,
    pub complementarity_generated: u64,
    pub mir_generated: u64,
}

/// Result of a single cut round.
#[derive(Debug, Clone)]
pub struct CutRoundResult {
    pub cuts_added: Vec<Cut>,
    pub lp_bound_before: f64,
    pub lp_bound_after: f64,
    pub round_number: u32,
}

/// Configuration for cut callback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutCallbackConfig {
    pub max_rounds: usize,
    pub max_cuts_per_round: usize,
    pub min_efficacy: f64,
    pub max_parallel_ratio: f64,
    pub enable_bilevel_intersection: bool,
    pub enable_gomory: bool,
    pub enable_complementarity: bool,
    pub enable_mir: bool,
}

impl Default for CutCallbackConfig {
    fn default() -> Self {
        Self {
            max_rounds: 10,
            max_cuts_per_round: 50,
            min_efficacy: 1e-4,
            max_parallel_ratio: 0.9,
            enable_bilevel_intersection: true,
            enable_gomory: true,
            enable_complementarity: true,
            enable_mir: false,
        }
    }
}

impl CutCallbackConfig {
    pub fn from_solver_config(cfg: &SolverConfig) -> Self {
        Self {
            max_rounds: cfg.cut_rounds_per_node,
            max_cuts_per_round: cfg.max_cuts_per_round,
            ..Self::default()
        }
    }
}

/// Pool of generated cuts (global and local).
#[derive(Debug, Clone)]
pub struct CutPool {
    global_cuts: Vec<Cut>,
    local_cut_stacks: HashMap<NodeId, Vec<Cut>>,
    max_pool_size: usize,
    min_efficacy: f64,
}

impl CutPool {
    pub fn new(max_size: usize, min_efficacy: f64) -> Self {
        Self {
            global_cuts: Vec::new(),
            local_cut_stacks: HashMap::new(),
            max_pool_size: max_size,
            min_efficacy,
        }
    }

    /// Add a cut to the pool. Returns true if added.
    pub fn add_cut(&mut self, cut: Cut, node_id: Option<NodeId>) -> bool {
        if cut.efficacy < self.min_efficacy {
            return false;
        }
        if cut.is_global {
            if self.global_cuts.len() < self.max_pool_size {
                self.global_cuts.push(cut);
                return true;
            }
            // Replace weakest cut if new one is stronger
            if let Some(weakest_idx) = self
                .global_cuts
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.efficacy
                        .partial_cmp(&b.1.efficacy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if cut.efficacy > self.global_cuts[weakest_idx].efficacy {
                    self.global_cuts[weakest_idx] = cut;
                    return true;
                }
            }
        } else if let Some(nid) = node_id {
            let stack = self.local_cut_stacks.entry(nid).or_default();
            stack.push(cut);
            return true;
        }
        false
    }

    pub fn get_global_cuts(&self) -> &[Cut] {
        &self.global_cuts
    }

    pub fn get_node_cuts(&self, node_id: NodeId) -> Vec<&Cut> {
        self.local_cut_stacks
            .get(&node_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Remove cuts with efficacy below threshold.
    pub fn remove_ineffective(&mut self, threshold: f64) -> usize {
        let before = self.global_cuts.len();
        self.global_cuts.retain(|c| c.efficacy >= threshold);
        before - self.global_cuts.len()
    }

    pub fn len(&self) -> usize {
        self.global_cuts.len()
            + self
                .local_cut_stacks
                .values()
                .map(|v| v.len())
                .sum::<usize>()
    }

    pub fn purge_old_cuts(&mut self, keep_node_ids: &[NodeId]) {
        let keep_set: std::collections::HashSet<NodeId> = keep_node_ids.iter().copied().collect();
        self.local_cut_stacks.retain(|k, _| keep_set.contains(k));
    }
}

/// Check parallelism (cosine similarity) between two cuts.
pub fn check_cut_parallelism(cut1: &Cut, cut2: &Cut) -> f64 {
    let mut dot = 0.0f64;
    let mut norm1_sq = 0.0f64;
    let mut norm2_sq = 0.0f64;

    // Build a map for cut2 coefficients
    let map2: HashMap<VarIndex, f64> = cut2.coefficients.iter().copied().collect();

    for &(var, c1) in &cut1.coefficients {
        norm1_sq += c1 * c1;
        if let Some(&c2) = map2.get(&var) {
            dot += c1 * c2;
        }
    }
    for &(_, c2) in &cut2.coefficients {
        norm2_sq += c2 * c2;
    }

    let denom = (norm1_sq.sqrt() * norm2_sq.sqrt()).max(1e-12);
    (dot / denom).abs()
}

/// Filter parallel cuts from a set, keeping the most efficacious.
pub fn filter_parallel_cuts(cuts: &mut Vec<Cut>, max_parallel: f64) {
    let mut keep = Vec::new();
    for cut in cuts.drain(..) {
        let is_parallel = keep
            .iter()
            .any(|existing: &Cut| check_cut_parallelism(&cut, existing) > max_parallel);
        if !is_parallel {
            keep.push(cut);
        }
    }
    *cuts = keep;
}

// ---------------------------------------------------------------------------
// Separation routines
// ---------------------------------------------------------------------------

/// Generate bilevel intersection cuts from the LP basis and complementarity structure.
pub fn separate_bilevel_intersection(node: &BbNode, model: &CompiledBilevelModel) -> Vec<Cut> {
    let mut cuts = Vec::new();
    if model.complementarity_pairs.is_empty() {
        return cuts;
    }

    let sol = &node.lp_solution;
    let n = sol.len();

    for &(s_var, y_var) in &model.complementarity_pairs {
        if s_var >= n || y_var >= n {
            continue;
        }
        let s_val = sol[s_var];
        let y_val = sol[y_var];

        // If both are positive, the complementarity is violated -> generate a cut
        if s_val > 1e-6 && y_val > 1e-6 {
            // Intersection cut: use the disjunction s=0 OR y=0
            // The resulting cut is: s/s_val + y/y_val <= 1
            let mut cut = Cut::new(
                vec![
                    (s_var, 1.0 / s_val.max(1e-8)),
                    (y_var, 1.0 / y_val.max(1e-8)),
                ],
                1.0,
                ConstraintSense::Le,
                CutType::BilevelIntersection,
                true,
            );
            cut.compute_efficacy(sol);
            if cut.efficacy > 1e-6 {
                cuts.push(cut);
            }
        }
    }
    cuts
}

/// Generate Gomory fractional cuts from the LP basis.
pub fn separate_gomory(node: &BbNode, model: &CompiledBilevelModel) -> Vec<Cut> {
    let mut cuts = Vec::new();
    let sol = &node.lp_solution;
    let n = model.num_vars;

    for &var in &model.integer_vars {
        if var >= sol.len() {
            continue;
        }
        let val = sol[var];
        let frac = fractionality(val);
        if frac < 1e-6 {
            continue;
        }

        let f0 = val - val.floor();

        // Simple Gomory cut: x_var <= floor(val) OR x_var >= ceil(val)
        // Derive cut from the tableau row (simplified: single-variable cut)
        // For a proper Gomory cut we'd need the simplex tableau.
        // Here we generate a simple rounding cut: x_var + slack = val
        // => (1/f0) * (fractional contributions) >= 1

        // Simplified version: x_var >= ceil(val) * indicator - M * (1-indicator)
        // We produce: x_var <= floor(val) cut (the down side)
        let mut cut_down = Cut::new(
            vec![(var, 1.0)],
            val.floor(),
            ConstraintSense::Le,
            CutType::Gomory,
            false,
        );
        cut_down.compute_efficacy(sol);

        let mut cut_up = Cut::new(
            vec![(var, -1.0)],
            -val.ceil(),
            ConstraintSense::Le,
            CutType::Gomory,
            false,
        );
        cut_up.compute_efficacy(sol);

        // Keep the more violated one
        if cut_down.efficacy > cut_up.efficacy && cut_down.efficacy > 1e-6 {
            cuts.push(cut_down);
        } else if cut_up.efficacy > 1e-6 {
            cuts.push(cut_up);
        }
    }
    cuts
}

/// Generate cuts from violated complementarity constraints.
pub fn separate_complementarity(node: &BbNode, model: &CompiledBilevelModel) -> Vec<Cut> {
    let mut cuts = Vec::new();
    let sol = &node.lp_solution;

    for &(s_var, y_var) in &model.complementarity_pairs {
        if s_var >= sol.len() || y_var >= sol.len() {
            continue;
        }
        let s_val = sol[s_var];
        let y_val = sol[y_var];

        if s_val > 1e-6 && y_val > 1e-6 {
            // Linearization of s * y <= 0: use McCormick
            // If s <= s_max and y <= y_max:
            //   s * y <= s_max * y + y_max * s - s_max * y_max
            // For the disjunction, we add: s <= M*(1-z), y <= M*z
            // Simpler: just force one to zero
            let mut cut = Cut::new(
                vec![(s_var, y_val), (y_var, s_val)],
                s_val * y_val,
                ConstraintSense::Le,
                CutType::Complementarity,
                false,
            );
            cut.compute_efficacy(sol);
            if cut.efficacy > 1e-8 {
                cuts.push(cut);
            }
        }
    }
    cuts
}

/// Generate mixed-integer rounding cuts.
pub fn separate_mir(node: &BbNode, model: &CompiledBilevelModel) -> Vec<Cut> {
    let mut cuts = Vec::new();
    let sol = &node.lp_solution;

    for &var in &model.integer_vars {
        if var >= sol.len() {
            continue;
        }
        let val = sol[var];
        let frac = fractionality(val);
        if frac < 1e-6 || frac > 1.0 - 1e-6 {
            continue;
        }

        let f = val - val.floor();
        // MIR cut: x_var >= ceil(val) if f > 0.5
        //          x_var <= floor(val) if f <= 0.5
        if f > 0.5 {
            let mut cut = Cut::new(
                vec![(var, -1.0)],
                -val.ceil(),
                ConstraintSense::Le,
                CutType::MIR,
                false,
            );
            cut.compute_efficacy(sol);
            if cut.efficacy > 1e-6 {
                cuts.push(cut);
            }
        } else {
            let mut cut = Cut::new(
                vec![(var, 1.0)],
                val.floor(),
                ConstraintSense::Le,
                CutType::MIR,
                false,
            );
            cut.compute_efficacy(sol);
            if cut.efficacy > 1e-6 {
                cuts.push(cut);
            }
        }
    }
    cuts
}

/// The main cut callback manager.
#[derive(Debug, Clone)]
pub struct CutCallbackManager {
    pub cut_pool: CutPool,
    pub config: CutCallbackConfig,
    pub stats: CutCallbackStats,
}

impl CutCallbackManager {
    pub fn new(config: CutCallbackConfig) -> Self {
        let pool_size = config.max_cuts_per_round * config.max_rounds;
        Self {
            cut_pool: CutPool::new(pool_size, config.min_efficacy),
            config,
            stats: CutCallbackStats::default(),
        }
    }

    pub fn from_solver_config(cfg: &SolverConfig) -> Self {
        Self::new(CutCallbackConfig::from_solver_config(cfg))
    }

    /// Run multiple rounds of cut separation at a node.
    pub fn run_cut_rounds(
        &mut self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
        max_rounds: usize,
    ) -> Vec<CutRoundResult> {
        let mut results = Vec::new();
        let rounds = max_rounds.min(self.config.max_rounds);

        for round in 0..rounds {
            let lb_before = node.lp_objective;
            let mut cuts = self.separate_single_round(node, model);

            if cuts.is_empty() {
                break;
            }

            // Filter parallel cuts
            filter_parallel_cuts(&mut cuts, self.config.max_parallel_ratio);

            // Truncate to max per round
            cuts.truncate(self.config.max_cuts_per_round);

            let added = self.filter_and_add_cuts(cuts.clone(), node);

            if added == 0 {
                break;
            }

            // Re-solve the LP with the new cuts
            node.solve_lp(model, lp_solver);

            let lb_after = node.lp_objective;

            results.push(CutRoundResult {
                cuts_added: cuts,
                lp_bound_before: lb_before,
                lp_bound_after: lb_after,
                round_number: round as u32,
            });

            node.num_cut_rounds += 1;
            self.stats.rounds_total += 1;

            // Stop if bound improvement is tiny
            if (lb_after - lb_before).abs() < 1e-6 {
                break;
            }
        }
        results
    }

    /// Run one round of all enabled separators.
    pub fn separate_single_round(
        &mut self,
        node: &BbNode,
        model: &CompiledBilevelModel,
    ) -> Vec<Cut> {
        let mut all_cuts = Vec::new();

        if self.config.enable_bilevel_intersection {
            let cuts = separate_bilevel_intersection(node, model);
            self.stats.bilevel_intersection_generated += cuts.len() as u64;
            all_cuts.extend(cuts);
        }

        if self.config.enable_gomory {
            let cuts = separate_gomory(node, model);
            self.stats.gomory_generated += cuts.len() as u64;
            all_cuts.extend(cuts);
        }

        if self.config.enable_complementarity {
            let cuts = separate_complementarity(node, model);
            self.stats.complementarity_generated += cuts.len() as u64;
            all_cuts.extend(cuts);
        }

        if self.config.enable_mir {
            let cuts = separate_mir(node, model);
            self.stats.mir_generated += cuts.len() as u64;
            all_cuts.extend(cuts);
        }

        self.stats.cuts_generated += all_cuts.len() as u64;
        all_cuts
    }

    /// Filter and add cuts to the node.
    pub fn filter_and_add_cuts(&mut self, cuts: Vec<Cut>, node: &mut BbNode) -> usize {
        let mut added = 0;
        for cut in cuts {
            if cut.efficacy < self.config.min_efficacy {
                self.stats.cuts_rejected_efficacy += 1;
                continue;
            }
            // Check parallelism with existing cuts
            let is_parallel = node.local_cuts.iter().any(|existing| {
                check_cut_parallelism(&cut, existing) > self.config.max_parallel_ratio
            });
            if is_parallel {
                self.stats.cuts_rejected_parallel += 1;
                continue;
            }

            let is_global = cut.is_global;
            node.add_local_cut(cut.clone());
            if is_global {
                self.cut_pool.add_cut(cut, None);
            } else {
                self.cut_pool.add_cut(cut, Some(node.id));
            }
            added += 1;
            self.stats.cuts_applied += 1;
        }
        added
    }

    pub fn get_stats(&self) -> &CutCallbackStats {
        &self.stats
    }

    /// Propagate global cuts from parent to children.
    pub fn propagate_cuts_to_children(&self, _parent: &BbNode, children: &mut [BbNode]) {
        for child in children.iter_mut() {
            for gcut in &self.cut_pool.global_cuts {
                let already_has = child
                    .local_cuts
                    .iter()
                    .any(|c| check_cut_parallelism(c, gcut) > 0.99);
                if !already_has {
                    child.add_local_cut(gcut.clone());
                }
            }
        }
    }
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
        m.complementarity_pairs = vec![(0, 1)];
        m
    }

    #[test]
    fn test_cut_pool_new() {
        let pool = CutPool::new(100, 1e-4);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_cut_pool_add() {
        let mut pool = CutPool::new(100, 1e-4);
        let mut cut = Cut::new(
            vec![(0, 1.0)],
            5.0,
            ConstraintSense::Le,
            CutType::Gomory,
            true,
        );
        cut.efficacy = 0.5;
        assert!(pool.add_cut(cut, None));
        assert_eq!(pool.get_global_cuts().len(), 1);
    }

    #[test]
    fn test_parallelism_check() {
        let c1 = Cut::new(
            vec![(0, 1.0), (1, 2.0)],
            3.0,
            ConstraintSense::Le,
            CutType::Gomory,
            true,
        );
        let c2 = Cut::new(
            vec![(0, 2.0), (1, 4.0)],
            6.0,
            ConstraintSense::Le,
            CutType::Gomory,
            true,
        );
        let par = check_cut_parallelism(&c1, &c2);
        assert!(par > 0.99); // Parallel cuts
    }

    #[test]
    fn test_bilevel_intersection_separation() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        node.lp_solution = vec![0.5, 0.5]; // Both positive => violation
        let cuts = separate_bilevel_intersection(&node, &model);
        assert!(!cuts.is_empty());
    }

    #[test]
    fn test_gomory_separation() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        node.lp_solution = vec![2.5, 1.0];
        let cuts = separate_gomory(&node, &model);
        assert!(!cuts.is_empty());
    }

    #[test]
    fn test_complementarity_separation() {
        let model = make_model();
        let mut node = BbNode::root(&model);
        node.lp_solution = vec![1.0, 1.0];
        let cuts = separate_complementarity(&node, &model);
        assert!(!cuts.is_empty());
    }

    #[test]
    fn test_callback_manager_creation() {
        let cfg = CutCallbackConfig::default();
        let mgr = CutCallbackManager::new(cfg);
        assert_eq!(mgr.stats.rounds_total, 0);
    }

    #[test]
    fn test_filter_parallel() {
        let mut cuts = vec![
            Cut::new(
                vec![(0, 1.0), (1, 2.0)],
                3.0,
                ConstraintSense::Le,
                CutType::Gomory,
                true,
            ),
            Cut::new(
                vec![(0, 2.0), (1, 4.0)],
                6.0,
                ConstraintSense::Le,
                CutType::Gomory,
                true,
            ),
            Cut::new(
                vec![(0, 1.0), (1, -1.0)],
                1.0,
                ConstraintSense::Le,
                CutType::Gomory,
                true,
            ),
        ];
        filter_parallel_cuts(&mut cuts, 0.9);
        // First two are parallel, so one should be removed
        assert!(cuts.len() <= 2);
    }
}
