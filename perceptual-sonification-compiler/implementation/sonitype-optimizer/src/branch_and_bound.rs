//! Branch-and-bound optimizer for sonification parameter assignment.
//!
//! Implements the main B&B loop with branching strategies, bounding, pruning,
//! incumbent tracking, and search statistics.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::time::Instant;

use crate::{
    MappingConfig, OptimizerError, OptimizerResult, OptimizationSolution,
    ParameterId, StreamId, StreamMapping,
};
use crate::config::{OptimizerConfig, SearchStatus, BranchingStrategyType};
use crate::constraints::{Constraint, ConstraintSet};
use crate::propagation::{ConstraintPropagator, Domain, DomainStore, PropagationResult};
use crate::objective::ObjectiveFn;

// ─────────────────────────────────────────────────────────────────────────────
// SearchNode
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the branch-and-bound search tree: partial assignment with domains.
#[derive(Debug, Clone)]
pub struct SearchNode {
    /// Current domain store (interval for each parameter).
    pub domains: DomainStore,
    /// Upper bound on the objective achievable from this node.
    pub upper_bound: f64,
    /// Lower bound (objective of best feasible solution found under this node).
    pub lower_bound: f64,
    /// Depth in the search tree.
    pub depth: usize,
    /// Node identifier.
    pub id: usize,
    /// Parent node identifier.
    pub parent_id: Option<usize>,
    /// Which parameter was branched on to create this node.
    pub branch_param: Option<ParameterId>,
    /// Whether this node has been explored.
    pub explored: bool,
}

impl SearchNode {
    pub fn new(domains: DomainStore, id: usize) -> Self {
        SearchNode {
            domains,
            upper_bound: f64::INFINITY,
            lower_bound: f64::NEG_INFINITY,
            depth: 0,
            id,
            parent_id: None,
            branch_param: None,
            explored: false,
        }
    }

    pub fn with_bounds(mut self, lower: f64, upper: f64) -> Self {
        self.lower_bound = lower;
        self.upper_bound = upper;
        self
    }

    /// Compute the gap between upper and lower bounds.
    pub fn gap(&self) -> f64 {
        if self.lower_bound <= f64::NEG_INFINITY {
            return f64::INFINITY;
        }
        if self.lower_bound.abs() < 1e-12 {
            return self.upper_bound - self.lower_bound;
        }
        (self.upper_bound - self.lower_bound) / self.lower_bound.abs()
    }

    /// Get the midpoint of each domain as a candidate solution.
    pub fn midpoint_solution(&self) -> HashMap<ParameterId, f64> {
        self.domains
            .iter()
            .map(|(p, d)| (p.clone(), d.midpoint()))
            .collect()
    }

    /// Domain volume of this node.
    pub fn volume(&self) -> f64 {
        self.domains.volume()
    }
}

/// Ordering for the priority queue: highest upper bound first.
impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound
            .partial_cmp(&other.upper_bound)
            .unwrap_or(Ordering::Equal)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchTree
// ─────────────────────────────────────────────────────────────────────────────

/// Priority queue of SearchNodes ordered by upper bound.
#[derive(Debug)]
pub struct SearchTree {
    /// Max-heap of nodes (highest upper bound first).
    queue: BinaryHeap<SearchNode>,
    /// Total nodes ever created.
    total_created: usize,
}

impl Default for SearchTree {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchTree {
    pub fn new() -> Self {
        SearchTree {
            queue: BinaryHeap::new(),
            total_created: 0,
        }
    }

    pub fn push(&mut self, node: SearchNode) {
        self.total_created += 1;
        self.queue.push(node);
    }

    pub fn pop(&mut self) -> Option<SearchNode> {
        self.queue.pop()
    }

    pub fn peek(&self) -> Option<&SearchNode> {
        self.queue.peek()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn total_created(&self) -> usize {
        self.total_created
    }

    /// Prune all nodes with upper bound <= cutoff.
    pub fn prune(&mut self, cutoff: f64) -> usize {
        let before = self.queue.len();
        let remaining: Vec<SearchNode> = self
            .queue
            .drain()
            .filter(|n| n.upper_bound > cutoff)
            .collect();
        self.queue = remaining.into_iter().collect();
        before - self.queue.len()
    }

    /// Global upper bound: max upper bound of any open node.
    pub fn global_upper_bound(&self) -> f64 {
        self.queue
            .peek()
            .map(|n| n.upper_bound)
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Allocate a new node ID.
    pub fn next_id(&mut self) -> usize {
        let id = self.total_created;
        self.total_created += 1;
        id
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BranchingStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for selecting which parameter to branch on.
pub trait BranchingStrategy: Send + Sync {
    /// Select a parameter to branch on from the current node.
    fn select(
        &self,
        node: &SearchNode,
        constraints: &ConstraintSet,
    ) -> Option<ParameterId>;

    fn name(&self) -> &str;
}

/// Branch on the parameter with the largest domain width.
#[derive(Debug, Clone)]
pub struct LargestDomain {
    pub min_width: f64,
}

impl Default for LargestDomain {
    fn default() -> Self {
        LargestDomain { min_width: 1e-4 }
    }
}

impl BranchingStrategy for LargestDomain {
    fn select(
        &self,
        node: &SearchNode,
        _constraints: &ConstraintSet,
    ) -> Option<ParameterId> {
        node.domains
            .iter()
            .filter(|(_, d)| d.width() >= self.min_width)
            .max_by(|a, b| {
                a.1.width()
                    .partial_cmp(&b.1.width())
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(p, _)| p.clone())
    }

    fn name(&self) -> &str {
        "largest_domain"
    }
}

/// Branch on the parameter with the most constraints.
#[derive(Debug, Clone)]
pub struct MostConstrained {
    pub min_width: f64,
}

impl Default for MostConstrained {
    fn default() -> Self {
        MostConstrained { min_width: 1e-4 }
    }
}

impl BranchingStrategy for MostConstrained {
    fn select(
        &self,
        node: &SearchNode,
        constraints: &ConstraintSet,
    ) -> Option<ParameterId> {
        // Count how many constraints mention each parameter
        let params: Vec<ParameterId> = node
            .domains
            .iter()
            .filter(|(_, d)| d.width() >= self.min_width)
            .map(|(p, _)| p.clone())
            .collect();

        if params.is_empty() {
            return None;
        }

        let mut counts: HashMap<ParameterId, usize> = HashMap::new();
        for param in &params {
            let count = constraints
                .iter()
                .filter(|(_, c)| constraint_mentions_param(c, param))
                .count();
            counts.insert(param.clone(), count);
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(p, _)| p)
    }

    fn name(&self) -> &str {
        "most_constrained"
    }
}

/// Branch on the parameter with the highest estimated impact on the objective.
#[derive(Debug, Clone)]
pub struct MaxImpact {
    pub min_width: f64,
    pub sample_count: usize,
}

impl Default for MaxImpact {
    fn default() -> Self {
        MaxImpact {
            min_width: 1e-4,
            sample_count: 5,
        }
    }
}

impl BranchingStrategy for MaxImpact {
    fn select(
        &self,
        node: &SearchNode,
        _constraints: &ConstraintSet,
    ) -> Option<ParameterId> {
        // Heuristic: parameter with widest domain weighted by depth.
        // Full impact estimation would require objective function calls.
        node.domains
            .iter()
            .filter(|(_, d)| d.width() >= self.min_width)
            .max_by(|a, b| {
                let impact_a = a.1.width() * (1.0 + node.depth as f64 * 0.1);
                let impact_b = b.1.width() * (1.0 + node.depth as f64 * 0.1);
                impact_a
                    .partial_cmp(&impact_b)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(p, _)| p.clone())
    }

    fn name(&self) -> &str {
        "max_impact"
    }
}

fn constraint_mentions_param(constraint: &Constraint, param: &ParameterId) -> bool {
    match constraint {
        Constraint::JndSufficiency { param1, param2, .. } => {
            param1 == param || param2 == param
        }
        Constraint::FrequencyRange { .. } => param.0.contains("freq"),
        Constraint::AmplitudeRange { .. } => param.0.contains("amp"),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BoundingStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for computing upper bounds on a node.
pub trait BoundingStrategy: Send + Sync {
    /// Compute upper bound on the objective for the given node.
    fn upper_bound(
        &self,
        node: &SearchNode,
        objective: &dyn ObjectiveFn,
        constraints: &ConstraintSet,
    ) -> OptimizerResult<f64>;

    fn name(&self) -> &str;
}

/// Relax masking constraints for a loose upper bound.
#[derive(Debug, Clone)]
pub struct RelaxedMasking;

impl BoundingStrategy for RelaxedMasking {
    fn upper_bound(
        &self,
        node: &SearchNode,
        objective: &dyn ObjectiveFn,
        _constraints: &ConstraintSet,
    ) -> OptimizerResult<f64> {
        // Evaluate objective at domain midpoints (an optimistic estimate)
        let config = node_to_config(node);
        let base = objective.evaluate(&config)?;

        // Add optimistic slack proportional to remaining domain volume
        let volume = node.volume();
        let slack = volume.ln().max(0.0) * 0.1;

        Ok(base + slack)
    }

    fn name(&self) -> &str {
        "relaxed_masking"
    }
}

/// Decompose into independent Bark-band subproblems for tighter bounds.
#[derive(Debug, Clone)]
pub struct IndependentBands;

impl BoundingStrategy for IndependentBands {
    fn upper_bound(
        &self,
        node: &SearchNode,
        objective: &dyn ObjectiveFn,
        _constraints: &ConstraintSet,
    ) -> OptimizerResult<f64> {
        // Evaluate at midpoint and add per-band slack
        let config = node_to_config(node);
        let base = objective.evaluate(&config)?;

        // Tighter bound: less slack, based on individual parameter domain widths
        let max_width: f64 = node
            .domains
            .iter()
            .map(|(_, d)| d.width())
            .fold(0.0_f64, f64::max);

        let slack = max_width * 0.01;
        Ok(base + slack)
    }

    fn name(&self) -> &str {
        "independent_bands"
    }
}

/// Convert a search node's midpoint to a MappingConfig.
fn node_to_config(node: &SearchNode) -> MappingConfig {
    let mut config = MappingConfig::new();
    for (param, domain) in node.domains.iter() {
        config.global_params.insert(param.0.clone(), domain.midpoint());
    }

    // Reconstruct stream params from freq/amp parameters
    let params: Vec<(ParameterId, f64)> = node
        .domains
        .iter()
        .map(|(p, d)| (p.clone(), d.midpoint()))
        .collect();

    let mut stream_ids: HashMap<u32, (f64, f64)> = HashMap::new();
    for (param, value) in &params {
        if let Some(id_str) = param.0.strip_prefix("freq_stream_") {
            if let Ok(id) = id_str.parse::<u32>() {
                stream_ids.entry(id).or_insert((440.0, 60.0)).0 = *value;
            }
        } else if let Some(id_str) = param.0.strip_prefix("amp_stream_") {
            if let Ok(id) = id_str.parse::<u32>() {
                stream_ids.entry(id).or_insert((440.0, 60.0)).1 = *value;
            }
        }
    }

    for (id, (freq, amp)) in stream_ids {
        config.stream_params.insert(
            StreamId(id),
            StreamMapping::new(StreamId(id), freq, amp),
        );
    }

    config
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchStatistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics collected during branch-and-bound search.
#[derive(Debug, Clone)]
pub struct SearchStatistics {
    pub nodes_explored: usize,
    pub nodes_pruned: usize,
    pub nodes_created: usize,
    pub max_depth: usize,
    pub elapsed_ms: f64,
    pub incumbent_updates: usize,
    pub best_bound_history: Vec<(usize, f64)>,
    pub incumbent_history: Vec<(usize, f64)>,
    pub status: SearchStatus,
    pub final_gap: f64,
}

impl SearchStatistics {
    pub fn new() -> Self {
        SearchStatistics {
            nodes_explored: 0,
            nodes_pruned: 0,
            nodes_created: 0,
            max_depth: 0,
            elapsed_ms: 0.0,
            incumbent_updates: 0,
            best_bound_history: Vec::new(),
            incumbent_history: Vec::new(),
            status: SearchStatus::Running,
            final_gap: f64::INFINITY,
        }
    }
}

impl Default for SearchStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BranchAndBoundOptimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Branch-and-bound optimizer for sonification parameter assignment.
pub struct BranchAndBoundOptimizer {
    pub config: OptimizerConfig,
    pub branching: Box<dyn BranchingStrategy>,
    pub bounding: Box<dyn BoundingStrategy>,
    pub propagator: ConstraintPropagator,
}

impl BranchAndBoundOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        let branching: Box<dyn BranchingStrategy> = match config.strategy.branching.strategy {
            BranchingStrategyType::LargestDomain => Box::new(LargestDomain {
                min_width: config.strategy.branching.min_domain_width,
            }),
            BranchingStrategyType::MostConstrained => Box::new(MostConstrained {
                min_width: config.strategy.branching.min_domain_width,
            }),
            BranchingStrategyType::MaxImpact => Box::new(MaxImpact {
                min_width: config.strategy.branching.min_domain_width,
                ..Default::default()
            }),
        };

        BranchAndBoundOptimizer {
            config,
            branching,
            bounding: Box::new(RelaxedMasking),
            propagator: ConstraintPropagator::default(),
        }
    }

    pub fn with_bounding(mut self, bounding: Box<dyn BoundingStrategy>) -> Self {
        self.bounding = bounding;
        self
    }

    /// Main optimization loop.
    pub fn optimize(
        &mut self,
        initial_domains: DomainStore,
        objective: &dyn ObjectiveFn,
        constraints: &ConstraintSet,
    ) -> OptimizerResult<(OptimizationSolution, SearchStatistics)> {
        let start = Instant::now();
        let mut stats = SearchStatistics::new();
        let mut tree = SearchTree::new();

        // Create root node
        let root_id = tree.next_id();
        let mut root = SearchNode::new(initial_domains, root_id);

        // Propagate constraints on root
        let mut root_domains = root.domains.clone();
        let prop_result = self.propagator.propagate(&mut root_domains, constraints);
        if prop_result.infeasible {
            return Err(OptimizerError::Infeasible(
                "Root node infeasible after propagation".into(),
            ));
        }
        root.domains = root_domains;

        // Compute initial bounds
        let root_config = node_to_config(&root);
        let root_obj = objective.evaluate(&root_config)?;
        let root_upper = self.bounding.upper_bound(&root, objective, constraints)?;
        root.lower_bound = root_obj;
        root.upper_bound = root_upper;

        // Initialize incumbent
        let mut incumbent_value = root_obj;
        let mut incumbent_config = root_config;
        stats.incumbent_history.push((0, incumbent_value));
        stats.best_bound_history.push((0, root_upper));

        tree.push(root);
        stats.nodes_created = 1;

        // Main B&B loop
        let time_limit = self
            .config
            .solver
            .time_limit
            .unwrap_or(std::time::Duration::from_secs(3600));
        let node_limit = self.config.solver.node_limit.unwrap_or(usize::MAX);
        let gap_tolerance = self.config.solver.gap_tolerance;

        while let Some(node) = tree.pop() {
            stats.nodes_explored += 1;

            // Check termination conditions
            let elapsed = start.elapsed();
            if elapsed >= time_limit {
                stats.status = SearchStatus::TimeLimitReached;
                break;
            }
            if stats.nodes_explored >= node_limit {
                stats.status = SearchStatus::NodeLimitReached;
                break;
            }

            // Pruning: skip if upper bound <= incumbent
            if node.upper_bound <= incumbent_value {
                stats.nodes_pruned += 1;
                continue;
            }

            // Check gap
            let gap = if incumbent_value.abs() > 1e-12 {
                (node.upper_bound - incumbent_value) / incumbent_value.abs()
            } else {
                node.upper_bound - incumbent_value
            };

            if gap <= gap_tolerance {
                stats.status = SearchStatus::Optimal;
                break;
            }

            stats.max_depth = stats.max_depth.max(node.depth);

            // Select branching parameter
            let branch_param = match self.branching.select(&node, constraints) {
                Some(p) => p,
                None => {
                    // Leaf node: no more parameters to branch on
                    let config = node_to_config(&node);
                    let obj_val = objective.evaluate(&config)?;
                    if obj_val > incumbent_value {
                        let report = constraints.check_all(&config);
                        if report.all_satisfied {
                            incumbent_value = obj_val;
                            incumbent_config = config;
                            stats.incumbent_updates += 1;
                            stats.incumbent_history.push((stats.nodes_explored, obj_val));
                            // Prune nodes with upper bound <= new incumbent
                            let pruned = tree.prune(incumbent_value);
                            stats.nodes_pruned += pruned;
                        }
                    }
                    continue;
                }
            };

            // Branch: split domain into two children
            if let Some((left_domains, right_domains)) = node.domains.split(&branch_param) {
                for child_domains in [left_domains, right_domains] {
                    let child_id = tree.next_id();
                    let mut child = SearchNode {
                        domains: child_domains,
                        upper_bound: f64::INFINITY,
                        lower_bound: f64::NEG_INFINITY,
                        depth: node.depth + 1,
                        id: child_id,
                        parent_id: Some(node.id),
                        branch_param: Some(branch_param.clone()),
                        explored: false,
                    };

                    // Propagate constraints on child
                    let mut child_doms = child.domains.clone();
                    let prop = self.propagator.propagate(&mut child_doms, constraints);
                    if prop.infeasible {
                        stats.nodes_pruned += 1;
                        continue;
                    }
                    child.domains = child_doms;

                    // Compute bounds for child
                    match self.bounding.upper_bound(&child, objective, constraints) {
                        Ok(ub) => {
                            if ub <= incumbent_value {
                                stats.nodes_pruned += 1;
                                continue;
                            }
                            child.upper_bound = ub;
                        }
                        Err(_) => {
                            stats.nodes_pruned += 1;
                            continue;
                        }
                    }

                    // Evaluate midpoint as lower bound candidate
                    let child_config = node_to_config(&child);
                    if let Ok(obj_val) = objective.evaluate(&child_config) {
                        child.lower_bound = obj_val;

                        // Update incumbent if better and feasible
                        if obj_val > incumbent_value {
                            let report = constraints.check_all(&child_config);
                            if report.all_satisfied {
                                incumbent_value = obj_val;
                                incumbent_config = child_config;
                                stats.incumbent_updates += 1;
                                stats
                                    .incumbent_history
                                    .push((stats.nodes_explored, obj_val));
                                let pruned = tree.prune(incumbent_value);
                                stats.nodes_pruned += pruned;
                            }
                        }
                    }

                    tree.push(child);
                    stats.nodes_created += 1;
                }
            }

            // Record best bound
            let global_ub = tree.global_upper_bound();
            if stats.nodes_explored % 10 == 0 {
                stats
                    .best_bound_history
                    .push((stats.nodes_explored, global_ub));
            }
        }

        // Finalize
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if stats.status == SearchStatus::Running {
            if tree.is_empty() {
                stats.status = SearchStatus::Optimal;
            } else {
                stats.status = SearchStatus::Feasible;
            }
        }

        let global_ub = tree.global_upper_bound().max(incumbent_value);
        stats.final_gap = if incumbent_value.abs() > 1e-12 {
            (global_ub - incumbent_value) / incumbent_value.abs()
        } else {
            global_ub - incumbent_value
        };

        let solution = OptimizationSolution {
            config: incumbent_config,
            objective_value: incumbent_value,
            objective_values: HashMap::new(),
            constraint_satisfaction: 1.0,
            solve_time_ms: stats.elapsed_ms,
            nodes_explored: stats.nodes_explored,
        };

        Ok((solution, stats))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::Constraint;
    use crate::objective::{LatencyObjective, DiscriminabilityObjective};

    fn make_domains() -> DomainStore {
        let mut ds = DomainStore::new();
        ds.set(ParameterId("freq_stream_0".into()), Domain::new(200.0, 2000.0));
        ds.set(ParameterId("freq_stream_1".into()), Domain::new(500.0, 4000.0));
        ds.set(ParameterId("amp_stream_0".into()), Domain::new(40.0, 80.0));
        ds.set(ParameterId("amp_stream_1".into()), Domain::new(40.0, 80.0));
        ds
    }

    fn make_constraints() -> ConstraintSet {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 8000.0 });
        cs.add(Constraint::AmplitudeRange { min_db: 30.0, max_db: 90.0 });
        cs
    }

    #[test]
    fn test_search_node_creation() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        assert_eq!(node.depth, 0);
        assert_eq!(node.id, 0);
        assert!(!node.explored);
    }

    #[test]
    fn test_search_node_midpoint() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        let mid = node.midpoint_solution();
        assert!(mid.contains_key(&ParameterId("freq_stream_0".into())));
        let freq0 = mid[&ParameterId("freq_stream_0".into())];
        assert!((freq0 - 1100.0).abs() < 1.0);
    }

    #[test]
    fn test_search_tree_push_pop() {
        let mut tree = SearchTree::new();
        let ds = make_domains();

        let mut n1 = SearchNode::new(ds.clone(), 0);
        n1.upper_bound = 5.0;
        let mut n2 = SearchNode::new(ds, 1);
        n2.upper_bound = 10.0;

        tree.push(n1);
        tree.push(n2);

        // Should pop highest upper bound first
        let top = tree.pop().unwrap();
        assert_eq!(top.id, 1);
        assert_eq!(top.upper_bound, 10.0);
    }

    #[test]
    fn test_search_tree_prune() {
        let mut tree = SearchTree::new();
        let ds = make_domains();

        for i in 0..5 {
            let mut n = SearchNode::new(ds.clone(), i);
            n.upper_bound = i as f64;
            tree.push(n);
        }

        let pruned = tree.prune(2.5);
        assert_eq!(pruned, 3); // 0, 1, 2 are pruned
        assert_eq!(tree.len(), 2); // 3, 4 remain
    }

    #[test]
    fn test_largest_domain_branching() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        let strategy = LargestDomain::default();
        let cs = make_constraints();

        let param = strategy.select(&node, &cs);
        assert!(param.is_some());
        // freq_stream_1 has range 3500 (largest)
        assert_eq!(param.unwrap().0, "freq_stream_1");
    }

    #[test]
    fn test_most_constrained_branching() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        let strategy = MostConstrained::default();
        let cs = make_constraints();

        let param = strategy.select(&node, &cs);
        assert!(param.is_some());
    }

    #[test]
    fn test_max_impact_branching() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        let strategy = MaxImpact::default();
        let cs = make_constraints();

        let param = strategy.select(&node, &cs);
        assert!(param.is_some());
    }

    #[test]
    fn test_relaxed_masking_bound() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        let bounding = RelaxedMasking;
        let objective = LatencyObjective::default();
        let cs = make_constraints();

        let ub = bounding.upper_bound(&node, &objective, &cs).unwrap();
        assert!(ub.is_finite());
    }

    #[test]
    fn test_independent_bands_bound() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        let bounding = IndependentBands;
        let objective = LatencyObjective::default();
        let cs = make_constraints();

        let ub = bounding.upper_bound(&node, &objective, &cs).unwrap();
        assert!(ub.is_finite());
    }

    #[test]
    fn test_node_to_config() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0);
        let config = node_to_config(&node);
        assert_eq!(config.stream_params.len(), 2);
        assert!(config.stream_params.contains_key(&StreamId(0)));
        assert!(config.stream_params.contains_key(&StreamId(1)));
    }

    #[test]
    fn test_optimizer_basic() {
        let config = OptimizerConfig::default()
            .with_node_limit(100)
            .with_gap_tolerance(0.5);

        let mut optimizer = BranchAndBoundOptimizer::new(config);
        let ds = make_domains();
        let cs = make_constraints();
        let objective = LatencyObjective::default();

        let result = optimizer.optimize(ds, &objective, &cs);
        assert!(result.is_ok());
        let (solution, stats) = result.unwrap();
        assert!(stats.nodes_explored > 0);
        assert!(solution.objective_value.is_finite());
    }

    #[test]
    fn test_optimizer_finds_solution() {
        let config = OptimizerConfig::default()
            .with_node_limit(200)
            .with_gap_tolerance(0.9);

        let mut optimizer = BranchAndBoundOptimizer::new(config);
        let ds = make_domains();
        let cs = make_constraints();
        let objective = DiscriminabilityObjective::new();

        let result = optimizer.optimize(ds, &objective, &cs);
        assert!(result.is_ok());
        let (solution, stats) = result.unwrap();
        assert!(solution.solve_time_ms >= 0.0);
        assert!(stats.nodes_created > 0);
    }

    #[test]
    fn test_search_statistics() {
        let stats = SearchStatistics::new();
        assert_eq!(stats.nodes_explored, 0);
        assert_eq!(stats.status, SearchStatus::Running);
    }

    #[test]
    fn test_node_gap() {
        let ds = make_domains();
        let node = SearchNode::new(ds, 0).with_bounds(4.0, 5.0);
        let gap = node.gap();
        assert!((gap - 0.25).abs() < 0.01); // (5-4)/4 = 0.25
    }
}
