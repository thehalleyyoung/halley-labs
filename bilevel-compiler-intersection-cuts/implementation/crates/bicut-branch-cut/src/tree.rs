//! Branch-and-bound tree management.
//!
//! This module owns the open-node pool, the incumbent solution, global cuts,
//! and the tree-level statistics.  It provides different node-selection
//! strategies (best-first, depth-first, hybrid) and exposes helpers used by
//! the main [`BranchAndCutSolver`](crate::solver::BranchAndCutSolver) loop.

use crate::node::*;
use crate::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Node-selection strategy
// ---------------------------------------------------------------------------

/// Node-selection strategy mirroring the user-facing [`NodeSelectionType`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeSelectionStrategy {
    /// Always pick the node with the smallest lower bound.
    BestFirst,
    /// Always pick the deepest node (LIFO-like).
    DepthFirst,
    /// Use depth-first until `switch_depth`, then best-first.
    Hybrid { switch_depth: u32 },
}

impl NodeSelectionStrategy {
    /// Convert from the config-level enum.
    pub fn from_config(nst: NodeSelectionType) -> Self {
        match nst {
            NodeSelectionType::BestFirst => Self::BestFirst,
            NodeSelectionType::DepthFirst => Self::DepthFirst,
            NodeSelectionType::Hybrid { switch_depth } => Self::Hybrid { switch_depth },
        }
    }
}

impl Default for NodeSelectionStrategy {
    fn default() -> Self {
        Self::BestFirst
    }
}

// ---------------------------------------------------------------------------
// Tree-level statistics
// ---------------------------------------------------------------------------

/// Lightweight counters maintained by the tree itself (as opposed to
/// the heavier [`SolverStatistics`](crate::SolverStatistics)).
#[derive(Debug, Clone, Default)]
pub struct TreeStatistics {
    /// Total nodes ever created (including the root).
    pub total_created: u64,
    /// Nodes whose LP relaxation has been solved.
    pub total_processed: u64,
    /// Nodes fathomed (integer-feasible or dominated).
    pub total_fathomed: u64,
    /// Nodes pruned by bound.
    pub total_pruned: u64,
    /// Deepest depth seen so far.
    pub max_depth_reached: u32,
    /// High-water mark of open nodes at any point.
    pub max_open_nodes: u64,
    /// Current number of open (active) nodes.
    pub current_open: u64,
}

impl TreeStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a new node was created.
    pub fn record_created(&mut self, depth: u32) {
        self.total_created += 1;
        if depth > self.max_depth_reached {
            self.max_depth_reached = depth;
        }
        self.current_open += 1;
        if self.current_open > self.max_open_nodes {
            self.max_open_nodes = self.current_open;
        }
    }

    /// Record that a node was selected for processing.
    pub fn record_processed(&mut self) {
        self.total_processed += 1;
        // The node leaves the open set when it is selected.
        if self.current_open > 0 {
            self.current_open -= 1;
        }
    }

    /// Record a fathom event.
    pub fn record_fathomed(&mut self) {
        self.total_fathomed += 1;
    }

    /// Record a prune event (bound-based removal from the pool).
    pub fn record_pruned(&mut self, count: u64) {
        self.total_pruned += count;
        self.current_open = self.current_open.saturating_sub(count);
    }
}

impl std::fmt::Display for TreeStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tree stats: created={}, processed={}, fathomed={}, pruned={}, \
             max_depth={}, max_open={}, current_open={}",
            self.total_created,
            self.total_processed,
            self.total_fathomed,
            self.total_pruned,
            self.max_depth_reached,
            self.max_open_nodes,
            self.current_open,
        )
    }
}

// ---------------------------------------------------------------------------
// Node pool
// ---------------------------------------------------------------------------

/// A pool of active (open) B&B nodes waiting to be processed.
///
/// Internally backed by a `Vec`.  Selection methods perform linear scans
/// which is acceptable for typical tree sizes; for very large trees a
/// priority-queue back-end could be swapped in.
#[derive(Debug, Clone)]
pub struct NodePool {
    nodes: Vec<BbNode>,
}

impl NodePool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Insert a node into the pool.
    pub fn push(&mut self, node: BbNode) {
        self.nodes.push(node);
    }

    /// Number of nodes currently in the pool.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return the smallest `lower_bound` across all open nodes, or `None`
    /// if the pool is empty.
    pub fn peek_best_bound(&self) -> Option<f64> {
        self.nodes
            .iter()
            .map(|n| n.lower_bound)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    // -- Selection helpers ---------------------------------------------------

    /// Select and remove the node with the smallest lower bound.
    pub fn pop_best_first(&mut self) -> Option<BbNode> {
        if self.nodes.is_empty() {
            return None;
        }
        let best_idx = self
            .nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.lower_bound
                    .partial_cmp(&b.lower_bound)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        Some(self.nodes.swap_remove(best_idx))
    }

    /// Select and remove the deepest node (largest `depth`).  Ties are
    /// broken by smallest lower bound.
    pub fn pop_depth_first(&mut self) -> Option<BbNode> {
        if self.nodes.is_empty() {
            return None;
        }
        let best_idx = self
            .nodes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.depth.cmp(&b.depth).then_with(|| {
                    // For equal depth, prefer the node with *smaller* lower bound.
                    b.lower_bound
                        .partial_cmp(&a.lower_bound)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        Some(self.nodes.swap_remove(best_idx))
    }

    /// Hybrid strategy: depth-first below `switch_depth`, best-first above.
    pub fn pop_hybrid(&mut self, switch_depth: u32) -> Option<BbNode> {
        if self.nodes.is_empty() {
            return None;
        }
        // If any node is deeper than the switch depth, pick depth-first
        // among those deep nodes.
        let has_deep = self.nodes.iter().any(|n| n.depth >= switch_depth);
        if has_deep {
            // Among nodes at or below switch_depth, pick deepest (then best bound).
            let best_idx = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.depth >= switch_depth)
                .max_by(|(_, a), (_, b)| {
                    a.depth.cmp(&b.depth).then_with(|| {
                        b.lower_bound
                            .partial_cmp(&a.lower_bound)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                })
                .map(|(i, _)| i);
            if let Some(idx) = best_idx {
                return Some(self.nodes.swap_remove(idx));
            }
        }
        // Fall back to best-first.
        self.pop_best_first()
    }

    /// Remove all nodes whose lower bound is ≥ `cutoff`.
    /// Returns the number of pruned nodes.
    pub fn prune_by_bound(&mut self, cutoff: f64) -> u64 {
        let before = self.nodes.len();
        self.nodes
            .retain(|n| n.lower_bound < cutoff - BOUND_TOLERANCE);
        let removed = before - self.nodes.len();
        removed as u64
    }

    /// Drain every remaining node out of the pool, returning them.
    pub fn drain_all(&mut self) -> Vec<BbNode> {
        self.nodes.drain(..).collect()
    }
}

impl Default for NodePool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Branch-and-bound tree
// ---------------------------------------------------------------------------

/// Central data structure for the branch-and-bound search.
///
/// The tree owns the open-node pool, the best known (incumbent) solution,
/// a collection of globally-valid cutting planes, and a log of all nodes
/// that have been processed.
#[derive(Debug, Clone)]
pub struct BranchAndBoundTree {
    /// Pool of open (active) nodes.
    node_pool: NodePool,
    /// Node-selection strategy in use.
    strategy: NodeSelectionStrategy,
    /// Monotonically increasing counter for node ids.
    next_id: NodeId,
    /// Objective value of the best known feasible solution.
    incumbent_value: f64,
    /// Variable values of the best known feasible solution.
    incumbent_solution: Option<Vec<f64>>,
    /// Globally-valid cutting planes (valid at every node).
    global_cuts: Vec<Cut>,
    /// Aggregate tree-level statistics.
    tree_stats: TreeStatistics,
    /// Map from `NodeId` to processed (no-longer-open) nodes, for logging
    /// and post-mortem analysis.
    node_log: HashMap<NodeId, BbNode>,
}

impl BranchAndBoundTree {
    // -- Construction --------------------------------------------------------

    /// Create a new, empty tree with the given node-selection strategy.
    pub fn new(strategy: NodeSelectionStrategy) -> Self {
        Self {
            node_pool: NodePool::new(),
            strategy,
            next_id: 0,
            incumbent_value: INFINITY_BOUND,
            incumbent_solution: None,
            global_cuts: Vec::new(),
            tree_stats: TreeStatistics::new(),
            node_log: HashMap::new(),
        }
    }

    // -- Strategy ------------------------------------------------------------

    /// Change the node-selection strategy (e.g. after a phase switch).
    pub fn set_strategy(&mut self, strategy: NodeSelectionStrategy) {
        self.strategy = strategy;
    }

    /// Return the current strategy.
    pub fn strategy(&self) -> NodeSelectionStrategy {
        self.strategy
    }

    // -- Node id allocation --------------------------------------------------

    /// Allocate and return the next unique node id.
    pub fn next_node_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // -- Node management -----------------------------------------------------

    /// Insert a node into the open pool.
    ///
    /// The tree statistics are updated to record the creation.
    pub fn add_node(&mut self, node: BbNode) {
        let depth = node.depth;
        self.tree_stats.record_created(depth);
        log::trace!(
            "Tree: added node {} (depth={}, lb={:.6})",
            node.id,
            depth,
            node.lower_bound,
        );
        self.node_pool.push(node);
    }

    /// Select the next node to process according to the active strategy.
    ///
    /// Returns `None` when the pool is exhausted.
    pub fn select_next_node(&mut self) -> Option<BbNode> {
        let node = match self.strategy {
            NodeSelectionStrategy::BestFirst => self.node_pool.pop_best_first(),
            NodeSelectionStrategy::DepthFirst => self.node_pool.pop_depth_first(),
            NodeSelectionStrategy::Hybrid { switch_depth } => {
                self.node_pool.pop_hybrid(switch_depth)
            }
        };
        if node.is_some() {
            self.tree_stats.record_processed();
        }
        node
    }

    /// Whether the open-node pool still has work to do.
    pub fn has_nodes(&self) -> bool {
        !self.node_pool.is_empty()
    }

    /// Number of open nodes.
    pub fn num_open_nodes(&self) -> usize {
        self.node_pool.len()
    }

    // -- Incumbent management ------------------------------------------------

    /// Try to update the incumbent (primal bound).
    ///
    /// If `objective` is strictly better than the current incumbent value
    /// the incumbent is replaced and the method returns `true`.  A bound
    /// prune is triggered automatically.
    pub fn update_incumbent(&mut self, objective: f64, solution: Vec<f64>) -> bool {
        if objective < self.incumbent_value - BOUND_TOLERANCE {
            log::info!(
                "Tree: new incumbent {:.8} (previous {:.8})",
                objective,
                self.incumbent_value,
            );
            self.incumbent_value = objective;
            self.incumbent_solution = Some(solution);
            // Eagerly prune nodes that can no longer improve upon this bound.
            let pruned = self.node_pool.prune_by_bound(self.incumbent_value);
            if pruned > 0 {
                self.tree_stats.record_pruned(pruned);
                log::debug!("Tree: pruned {} nodes after incumbent update", pruned);
            }
            true
        } else {
            false
        }
    }

    /// Return the current incumbent value and solution (if any).
    pub fn get_incumbent(&self) -> (f64, Option<&Vec<f64>>) {
        (self.incumbent_value, self.incumbent_solution.as_ref())
    }

    /// Return the incumbent objective value (upper bound).
    pub fn incumbent_value(&self) -> f64 {
        self.incumbent_value
    }

    // -- Bounds --------------------------------------------------------------

    /// Global lower bound = best (smallest) lower bound across all open nodes.
    ///
    /// If no open nodes remain, returns the incumbent value (the tree is
    /// closed and the incumbent is proven optimal).
    pub fn get_global_lower_bound(&self) -> f64 {
        self.node_pool
            .peek_best_bound()
            .unwrap_or(self.incumbent_value)
    }

    /// Relative optimality gap between the current incumbent and the global
    /// lower bound.
    pub fn get_gap(&self) -> f64 {
        let lb = self.get_global_lower_bound();
        compute_gap(self.incumbent_value, lb)
    }

    // -- Global cuts ---------------------------------------------------------

    /// Register a globally-valid cut.
    pub fn add_global_cut(&mut self, cut: Cut) {
        log::trace!(
            "Tree: added global {} cut (rhs={:.6})",
            cut.cut_type,
            cut.rhs,
        );
        self.global_cuts.push(cut);
    }

    /// Borrow the list of global cuts.
    pub fn get_global_cuts(&self) -> &[Cut] {
        &self.global_cuts
    }

    /// Number of global cuts.
    pub fn num_global_cuts(&self) -> usize {
        self.global_cuts.len()
    }

    // -- Pruning / fathoming -------------------------------------------------

    /// Explicitly prune the open pool against the current incumbent.
    ///
    /// Returns the number of nodes removed.
    pub fn prune_by_bound(&mut self) -> u64 {
        let cutoff = self.incumbent_value;
        let pruned = self.node_pool.prune_by_bound(cutoff);
        if pruned > 0 {
            self.tree_stats.record_pruned(pruned);
            log::debug!("Tree: prune_by_bound removed {} nodes", pruned);
        }
        pruned
    }

    /// Record that a node has been fathomed (infeasible, integer-feasible,
    /// or dominated).  The node is moved into the log.
    pub fn fathom_node(&mut self, mut node: BbNode, reason: NodeStatus) {
        node.set_status(reason);
        self.tree_stats.record_fathomed();
        log::trace!("Tree: fathomed node {} (reason={:?})", node.id, reason,);
        self.node_log.insert(node.id, node);
    }

    /// Record a processed node in the log (after branching children have
    /// been created).
    pub fn record_processed_node(&mut self, mut node: BbNode) {
        node.set_status(NodeStatus::Branched);
        self.node_log.insert(node.id, node);
    }

    // -- Statistics & queries ------------------------------------------------

    /// Borrow the tree-level statistics.
    pub fn get_tree_stats(&self) -> &TreeStatistics {
        &self.tree_stats
    }

    /// Borrow the processed-node log.
    pub fn get_node_log(&self) -> &HashMap<NodeId, BbNode> {
        &self.node_log
    }

    /// Look up a processed node by id.
    pub fn get_logged_node(&self, id: NodeId) -> Option<&BbNode> {
        self.node_log.get(&id)
    }

    /// Total number of nodes created so far.
    pub fn total_nodes_created(&self) -> u64 {
        self.tree_stats.total_created
    }

    /// The highest node id that has been allocated so far.
    pub fn highest_node_id(&self) -> NodeId {
        if self.next_id == 0 {
            0
        } else {
            self.next_id - 1
        }
    }

    // -- Reset ---------------------------------------------------------------

    /// Reset the tree to a clean initial state, keeping the strategy.
    pub fn reset(&mut self) {
        let strategy = self.strategy;
        *self = Self::new(strategy);
    }
}

impl Default for BranchAndBoundTree {
    fn default() -> Self {
        Self::new(NodeSelectionStrategy::default())
    }
}

impl std::fmt::Display for BranchAndBoundTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BranchAndBoundTree(open={}, incumbent={:.6}, lb={:.6}, gap={:.4}%, \
             global_cuts={}, {})",
            self.num_open_nodes(),
            self.incumbent_value,
            self.get_global_lower_bound(),
            self.get_gap() * 100.0,
            self.global_cuts.len(),
            self.tree_stats,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{BbNode, BranchDirection, BranchRecord, NodeStatus};

    // -- helpers ------------------------------------------------------------

    fn make_node(id: NodeId, depth: u32, lb: f64) -> BbNode {
        BbNode {
            id,
            parent_id: if id == 0 { None } else { Some(0) },
            depth,
            lower_bound: lb,
            status: NodeStatus::Pending,
            lp_objective: INFINITY_BOUND,
            lp_solution: Vec::new(),
            lp_dual: Vec::new(),
            lp_basis: Vec::new(),
            lp_iterations: 0,
            var_lower_bounds: Vec::new(),
            var_upper_bounds: Vec::new(),
            branching_history: if id == 0 {
                Vec::new()
            } else {
                vec![BranchRecord {
                    variable: 0,
                    direction: BranchDirection::Down,
                    bound_value: 0.0,
                    parent_lp_obj: 0.0,
                }]
            },
            local_cuts: Vec::new(),
            warm_start: None,
            fractional_vars: Vec::new(),
            age: 0,
            num_cut_rounds: 0,
            is_root: id == 0,
        }
    }

    // -- NodePool tests -----------------------------------------------------

    #[test]
    fn test_node_pool_push_and_len() {
        let mut pool = NodePool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);

        pool.push(make_node(0, 0, 1.0));
        pool.push(make_node(1, 1, 2.0));
        assert_eq!(pool.len(), 2);
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_node_pool_best_first() {
        let mut pool = NodePool::new();
        pool.push(make_node(1, 1, 5.0));
        pool.push(make_node(2, 2, 3.0));
        pool.push(make_node(3, 1, 7.0));

        let best = pool.pop_best_first().unwrap();
        assert_eq!(best.id, 2);
        assert!((best.lower_bound - 3.0).abs() < 1e-12);
        assert_eq!(pool.len(), 2);

        let next = pool.pop_best_first().unwrap();
        assert_eq!(next.id, 1);
    }

    #[test]
    fn test_node_pool_depth_first() {
        let mut pool = NodePool::new();
        pool.push(make_node(1, 1, 5.0));
        pool.push(make_node(2, 3, 8.0));
        pool.push(make_node(3, 3, 4.0));
        pool.push(make_node(4, 2, 1.0));

        // Deepest is depth 3; among those, the one with better (smaller) lb
        // should be preferred.
        let first = pool.pop_depth_first().unwrap();
        assert_eq!(first.depth, 3);
        assert_eq!(first.id, 3); // lb=4.0 < lb=8.0
    }

    #[test]
    fn test_node_pool_hybrid() {
        let mut pool = NodePool::new();
        // switch_depth = 2: nodes at depth >= 2 → depth-first, else best-first.
        pool.push(make_node(1, 0, 1.0));
        pool.push(make_node(2, 1, 2.0));
        pool.push(make_node(3, 2, 10.0));
        pool.push(make_node(4, 3, 9.0));

        // There are deep nodes (depth >= 2), so we pick depth-first among them.
        let first = pool.pop_hybrid(2).unwrap();
        assert_eq!(first.id, 4); // deepest (depth 3)

        let second = pool.pop_hybrid(2).unwrap();
        assert_eq!(second.id, 3); // depth 2

        // Now only shallow nodes remain (depth 0, 1) → best-first.
        let third = pool.pop_hybrid(2).unwrap();
        assert_eq!(third.id, 1); // lb=1.0 < lb=2.0
    }

    #[test]
    fn test_node_pool_prune_by_bound() {
        let mut pool = NodePool::new();
        pool.push(make_node(1, 1, 2.0));
        pool.push(make_node(2, 1, 5.0));
        pool.push(make_node(3, 1, 8.0));
        pool.push(make_node(4, 1, 3.0));

        let removed = pool.prune_by_bound(5.0);
        assert_eq!(removed, 2); // nodes with lb >= 5.0 (ids 2 and 3)
        assert_eq!(pool.len(), 2);

        let bounds: Vec<f64> = pool.drain_all().iter().map(|n| n.lower_bound).collect();
        assert!(bounds.iter().all(|&b| b < 5.0));
    }

    #[test]
    fn test_node_pool_peek_best_bound() {
        let mut pool = NodePool::new();
        assert!(pool.peek_best_bound().is_none());

        pool.push(make_node(1, 0, 10.0));
        pool.push(make_node(2, 0, 3.0));
        pool.push(make_node(3, 0, 7.0));

        let best = pool.peek_best_bound().unwrap();
        assert!((best - 3.0).abs() < 1e-12);
        // peek should not remove
        assert_eq!(pool.len(), 3);
    }

    // -- BranchAndBoundTree tests -------------------------------------------

    #[test]
    fn test_tree_add_and_select_best_first() {
        let mut tree = BranchAndBoundTree::new(NodeSelectionStrategy::BestFirst);

        let id0 = tree.next_node_id();
        tree.add_node(make_node(id0, 0, 0.0));

        let id1 = tree.next_node_id();
        tree.add_node(make_node(id1, 1, 5.0));

        let id2 = tree.next_node_id();
        tree.add_node(make_node(id2, 1, 2.0));

        assert_eq!(tree.num_open_nodes(), 3);
        assert!(tree.has_nodes());

        let selected = tree.select_next_node().unwrap();
        assert_eq!(selected.id, id0); // lb = 0.0
        assert_eq!(tree.num_open_nodes(), 2);
    }

    #[test]
    fn test_tree_incumbent_and_pruning() {
        let mut tree = BranchAndBoundTree::new(NodeSelectionStrategy::BestFirst);

        tree.add_node(make_node(1, 1, 2.0));
        tree.add_node(make_node(2, 1, 6.0));
        tree.add_node(make_node(3, 1, 10.0));
        tree.add_node(make_node(4, 1, 4.0));

        // No incumbent yet; value should be INFINITY_BOUND.
        let (val, sol) = tree.get_incumbent();
        assert!(val > 1e19);
        assert!(sol.is_none());

        // Update incumbent → nodes with lb >= 5.0 should be pruned.
        let improved = tree.update_incumbent(5.0, vec![1.0, 2.0]);
        assert!(improved);
        assert_eq!(tree.num_open_nodes(), 2); // only lb 2.0 and 4.0 survive

        // Gap should be computable.
        let gap = tree.get_gap();
        // lb = 2.0, ub = 5.0 → gap = |5-2|/|5| = 0.6
        assert!((gap - 0.6).abs() < 1e-8);
    }

    #[test]
    fn test_tree_global_cuts() {
        let mut tree = BranchAndBoundTree::default();

        assert_eq!(tree.num_global_cuts(), 0);
        assert!(tree.get_global_cuts().is_empty());

        let cut = Cut::new(
            vec![(0, 1.0), (1, -1.0)],
            3.0,
            ConstraintSense::Le,
            CutType::Gomory,
            true,
        );
        tree.add_global_cut(cut);

        assert_eq!(tree.num_global_cuts(), 1);
        let cuts = tree.get_global_cuts();
        assert_eq!(cuts[0].cut_type, CutType::Gomory);
        assert!((cuts[0].rhs - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_tree_fathom_and_reset() {
        let mut tree = BranchAndBoundTree::new(NodeSelectionStrategy::DepthFirst);

        tree.add_node(make_node(0, 0, 0.0));
        tree.add_node(make_node(1, 1, 1.0));

        let node = tree.select_next_node().unwrap();
        tree.fathom_node(node, NodeStatus::Infeasible);

        assert_eq!(tree.get_tree_stats().total_fathomed, 1);
        assert_eq!(tree.get_node_log().len(), 1);

        // Process the remaining node and record it.
        let node2 = tree.select_next_node().unwrap();
        tree.record_processed_node(node2);
        assert_eq!(tree.get_node_log().len(), 2);
        assert!(!tree.has_nodes());

        // Reset should clear everything.
        tree.reset();
        assert!(!tree.has_nodes());
        assert_eq!(tree.get_tree_stats().total_created, 0);
        assert!(tree.get_node_log().is_empty());
        assert!(tree.get_global_cuts().is_empty());
        assert_eq!(tree.strategy(), NodeSelectionStrategy::DepthFirst);
    }
}
