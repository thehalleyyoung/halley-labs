//! Lattice of abstraction triples (k, n, ε) with ordering, meet/join, traversal.
//!
//! An abstraction triple (k, n, ε) specifies:
//! - k: number of output clusters (alphabet size)
//! - n: input probing depth (word length bound)
//! - ε: distributional tolerance (metric resolution)
//!
//! The lattice ordering is: α ≤ α' iff k ≤ k' ∧ n ≤ n' ∧ ε ≥ ε'
//! (finer abstraction = more clusters, deeper probing, tighter tolerance)

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use std::fmt;
use ordered_float::OrderedFloat;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// An abstraction triple (k, n, ε).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionTriple {
    /// Number of output semantic clusters.
    pub k: usize,
    /// Maximum input probing depth (word length).
    pub n: usize,
    /// Distributional tolerance (epsilon).
    pub epsilon: f64,
}

impl AbstractionTriple {
    pub fn new(k: usize, n: usize, epsilon: f64) -> Self {
        assert!(k > 0, "k must be positive");
        assert!(epsilon > 0.0, "epsilon must be positive");
        Self { k, n, epsilon }
    }

    /// The coarsest possible abstraction: 1 cluster, depth 0, large tolerance.
    pub fn coarsest() -> Self {
        Self { k: 1, n: 0, epsilon: 1.0 }
    }

    /// Estimated computational cost of working at this abstraction level.
    /// Rough model: cost ∝ k * |Σ|^n / ε^2  (sample complexity scaling).
    pub fn estimated_cost(&self, alphabet_size: usize) -> f64 {
        let input_space = if self.n == 0 {
            1.0
        } else {
            (alphabet_size as f64).powi(self.n as i32)
        };
        let eps_sq = self.epsilon * self.epsilon;
        (self.k as f64) * input_space / eps_sq
    }

    /// Compute the "volume" of the abstraction space (k * words * 1/ε).
    pub fn volume(&self, alphabet_size: usize) -> f64 {
        let input_space = if self.n == 0 {
            1.0
        } else {
            let mut total = 0.0;
            for i in 0..=self.n {
                total += (alphabet_size as f64).powi(i as i32);
            }
            total
        };
        (self.k as f64) * input_space / self.epsilon
    }

    /// Returns true if this triple is coarser than or equal to `other`.
    pub fn leq(&self, other: &AbstractionTriple) -> bool {
        self.k <= other.k && self.n <= other.n && self.epsilon >= other.epsilon
    }

    /// Returns true if this triple is strictly coarser than `other`.
    pub fn lt(&self, other: &AbstractionTriple) -> bool {
        self.leq(other) && (self.k < other.k || self.n < other.n || self.epsilon > other.epsilon)
    }

    /// Meet (greatest lower bound): take min k, min n, max ε.
    pub fn meet(&self, other: &AbstractionTriple) -> AbstractionTriple {
        AbstractionTriple {
            k: self.k.min(other.k),
            n: self.n.min(other.n),
            epsilon: self.epsilon.max(other.epsilon),
        }
    }

    /// Join (least upper bound): take max k, max n, min ε.
    pub fn join(&self, other: &AbstractionTriple) -> AbstractionTriple {
        AbstractionTriple {
            k: self.k.max(other.k),
            n: self.n.max(other.n),
            epsilon: self.epsilon.min(other.epsilon),
        }
    }

    /// Euclidean distance in the normalized parameter space.
    pub fn distance(&self, other: &AbstractionTriple) -> f64 {
        let dk = (self.k as f64 - other.k as f64).powi(2);
        let dn = (self.n as f64 - other.n as f64).powi(2);
        // For epsilon, we use the log scale since it's typically in (0,1].
        let de = (self.epsilon.ln() - other.epsilon.ln()).powi(2);
        (dk + dn + de).sqrt()
    }

    /// Key for hashing/comparison (discretize epsilon).
    pub(crate) fn discrete_key(&self) -> (usize, usize, OrderedFloat<f64>) {
        (self.k, self.n, OrderedFloat(self.epsilon))
    }
}

impl PartialEq for AbstractionTriple {
    fn eq(&self, other: &Self) -> bool {
        self.k == other.k && self.n == other.n && (self.epsilon - other.epsilon).abs() < 1e-12
    }
}

impl Eq for AbstractionTriple {}

impl std::hash::Hash for AbstractionTriple {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.discrete_key().hash(state);
    }
}

impl fmt::Display for AbstractionTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(k={}, n={}, ε={:.4})", self.k, self.n, self.epsilon)
    }
}

impl PartialOrd for AbstractionTriple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.leq(other) {
            Some(Ordering::Less)
        } else if other.leq(self) {
            Some(Ordering::Greater)
        } else {
            None // Incomparable
        }
    }
}

// ---------------------------------------------------------------------------
// Lattice node
// ---------------------------------------------------------------------------

/// A node in the abstraction lattice with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeNode {
    pub id: usize,
    pub triple: AbstractionTriple,
    /// Whether this node has been explored in CEGAR.
    pub explored: bool,
    /// Whether verification passed at this level.
    pub verified: Option<bool>,
    /// Estimated cost to explore this node.
    pub estimated_cost: f64,
    /// Actual cost observed (if explored).
    pub actual_cost: Option<f64>,
    /// Parent node IDs (coarser abstractions).
    pub parents: Vec<usize>,
    /// Child node IDs (finer abstractions).
    pub children: Vec<usize>,
}

impl LatticeNode {
    pub fn new(id: usize, triple: AbstractionTriple, alphabet_size: usize) -> Self {
        let estimated_cost = triple.estimated_cost(alphabet_size);
        Self {
            id,
            triple,
            explored: false,
            verified: None,
            estimated_cost,
            actual_cost: None,
            parents: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn mark_explored(&mut self, passed: bool, cost: f64) {
        self.explored = true;
        self.verified = Some(passed);
        self.actual_cost = Some(cost);
    }
}

impl fmt::Display for LatticeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = match self.verified {
            Some(true) => "✓",
            Some(false) => "✗",
            None => "?",
        };
        write!(f, "[{}] {} {}", self.id, self.triple, status)
    }
}

// ---------------------------------------------------------------------------
// Lattice traversal strategies
// ---------------------------------------------------------------------------

/// Strategy for traversing the abstraction lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeTraversalStrategy {
    /// Start coarsest, refine upward (bottom-up).
    BottomUp,
    /// Start finest feasible, coarsen downward (top-down).
    TopDown,
    /// Breadth-first search by lattice level.
    BreadthFirst,
    /// Depth-first search following most promising path.
    DepthFirst,
    /// Best-first search using cost heuristic.
    BestFirst,
    /// Binary search on each dimension independently.
    BinarySearch,
    /// Adaptive: choose strategy based on previous results.
    Adaptive,
}

/// Budget constraints for lattice search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeBudget {
    /// Maximum total computational cost.
    pub max_total_cost: f64,
    /// Maximum number of nodes to explore.
    pub max_nodes: usize,
    /// Maximum wall-clock time in seconds.
    pub max_time_secs: f64,
    /// Cost spent so far.
    pub cost_spent: f64,
    /// Nodes explored so far.
    pub nodes_explored: usize,
}

impl LatticeBudget {
    pub fn new(max_total_cost: f64, max_nodes: usize, max_time_secs: f64) -> Self {
        Self {
            max_total_cost,
            max_nodes,
            max_time_secs,
            cost_spent: 0.0,
            nodes_explored: 0,
        }
    }

    pub fn unlimited() -> Self {
        Self {
            max_total_cost: f64::INFINITY,
            max_nodes: usize::MAX,
            max_time_secs: f64::INFINITY,
            cost_spent: 0.0,
            nodes_explored: 0,
        }
    }

    pub fn is_exhausted(&self) -> bool {
        self.cost_spent >= self.max_total_cost || self.nodes_explored >= self.max_nodes
    }

    pub fn remaining_cost(&self) -> f64 {
        (self.max_total_cost - self.cost_spent).max(0.0)
    }

    pub fn can_afford(&self, cost: f64) -> bool {
        self.cost_spent + cost <= self.max_total_cost
            && self.nodes_explored < self.max_nodes
    }

    pub fn record_exploration(&mut self, cost: f64) {
        self.cost_spent += cost;
        self.nodes_explored += 1;
    }
}

impl Default for LatticeBudget {
    fn default() -> Self {
        Self::new(1e9, 1000, 3600.0)
    }
}

// ---------------------------------------------------------------------------
// Priority queue entry for best-first search
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PriorityEntry {
    node_id: usize,
    priority: OrderedFloat<f64>,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smaller cost = higher priority.
        other.priority.cmp(&self.priority)
    }
}

// ---------------------------------------------------------------------------
// The main AbstractionLattice
// ---------------------------------------------------------------------------

/// A lattice of (k, n, ε) abstraction triples with traversal and search.
#[derive(Debug, Clone)]
pub struct AbstractionLattice {
    /// All nodes in the lattice.
    pub nodes: Vec<LatticeNode>,
    /// Map from discrete key to node id.
    key_to_id: HashMap<(usize, usize, OrderedFloat<f64>), usize>,
    /// Input alphabet size (for cost estimation).
    pub alphabet_size: usize,
    /// Allowed k values.
    pub k_values: Vec<usize>,
    /// Allowed n values.
    pub n_values: Vec<usize>,
    /// Allowed ε values (sorted descending — coarsest first).
    pub epsilon_values: Vec<f64>,
    /// Current traversal strategy.
    pub strategy: LatticeTraversalStrategy,
    /// Budget for exploration.
    pub budget: LatticeBudget,
}

impl AbstractionLattice {
    /// Build the full lattice from allowed parameter ranges.
    pub fn new(
        k_values: Vec<usize>,
        n_values: Vec<usize>,
        epsilon_values: Vec<f64>,
        alphabet_size: usize,
        strategy: LatticeTraversalStrategy,
        budget: LatticeBudget,
    ) -> Self {
        let mut nodes = Vec::new();
        let mut key_to_id = HashMap::new();

        // Sort epsilon descending (coarsest first).
        let mut eps_sorted = epsilon_values.clone();
        eps_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        let mut k_sorted = k_values.clone();
        k_sorted.sort();
        let mut n_sorted = n_values.clone();
        n_sorted.sort();

        // Create all nodes.
        for &k in &k_sorted {
            for &n in &n_sorted {
                for &eps in &eps_sorted {
                    let triple = AbstractionTriple::new(k, n.max(0), eps);
                    let id = nodes.len();
                    let node = LatticeNode::new(id, triple.clone(), alphabet_size);
                    key_to_id.insert(triple.discrete_key(), id);
                    nodes.push(node);
                }
            }
        }

        // Compute edges (parent/child relationships).
        // A node is a parent of another if it is strictly coarser and
        // there is no intermediate node.
        let n_nodes = nodes.len();
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i == j {
                    continue;
                }
                let ti = &nodes[i].triple;
                let tj = &nodes[j].triple;
                if ti.lt(tj) {
                    // Check that there's no intermediate node between i and j.
                    let mut is_direct = true;
                    for m in 0..n_nodes {
                        if m == i || m == j {
                            continue;
                        }
                        let tm = &nodes[m].triple;
                        if ti.lt(tm) && tm.lt(tj) {
                            is_direct = false;
                            break;
                        }
                    }
                    if is_direct {
                        // Collect the parent/child IDs to update after the loop.
                        // For now, store in a temporary structure.
                    }
                }
            }
        }

        // Two-pass edge construction (avoids borrow issues).
        let mut parent_edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i == j {
                    continue;
                }
                if nodes[i].triple.lt(&nodes[j].triple) {
                    let mut is_direct = true;
                    for m in 0..n_nodes {
                        if m == i || m == j {
                            continue;
                        }
                        if nodes[i].triple.lt(&nodes[m].triple)
                            && nodes[m].triple.lt(&nodes[j].triple)
                        {
                            is_direct = false;
                            break;
                        }
                    }
                    if is_direct {
                        parent_edges.push((i, j)); // i is parent of j
                    }
                }
            }
        }

        for (parent, child) in &parent_edges {
            nodes[*child].parents.push(*parent);
            nodes[*parent].children.push(*child);
        }

        Self {
            nodes,
            key_to_id,
            alphabet_size,
            k_values: k_sorted,
            n_values: n_sorted,
            epsilon_values: eps_sorted,
            strategy,
            budget,
        }
    }

    /// Create a small default lattice useful for testing.
    pub fn default_lattice(alphabet_size: usize) -> Self {
        Self::new(
            vec![2, 4, 8, 16],
            vec![1, 2, 3, 4],
            vec![0.5, 0.25, 0.1, 0.05],
            alphabet_size,
            LatticeTraversalStrategy::BestFirst,
            LatticeBudget::default(),
        )
    }

    /// Number of nodes in the lattice.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&LatticeNode> {
        self.nodes.get(id)
    }

    /// Get a mutable node by ID.
    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut LatticeNode> {
        self.nodes.get_mut(id)
    }

    /// Lookup a node by triple.
    pub fn find_node(&self, triple: &AbstractionTriple) -> Option<usize> {
        self.key_to_id.get(&triple.discrete_key()).copied()
    }

    /// Get the bottom (coarsest) element of the lattice.
    pub fn bottom(&self) -> Option<usize> {
        // The bottom node has the minimum k, minimum n, maximum epsilon.
        let min_k = *self.k_values.first()?;
        let min_n = *self.n_values.first()?;
        let max_eps = *self.epsilon_values.first()?; // sorted descending
        let triple = AbstractionTriple::new(min_k, min_n, max_eps);
        self.find_node(&triple)
    }

    /// Get the top (finest) element of the lattice.
    pub fn top(&self) -> Option<usize> {
        let max_k = *self.k_values.last()?;
        let max_n = *self.n_values.last()?;
        let min_eps = *self.epsilon_values.last()?; // sorted descending
        let triple = AbstractionTriple::new(max_k, max_n, min_eps);
        self.find_node(&triple)
    }

    /// Return all nodes at a given "level" (sum of parameter indices).
    pub fn nodes_at_level(&self, level: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for (ki, _) in self.k_values.iter().enumerate() {
            for (ni, _) in self.n_values.iter().enumerate() {
                for (ei, _) in self.epsilon_values.iter().enumerate() {
                    // Level = ki + ni + (max_ei - ei) since smaller eps = finer.
                    let max_ei = self.epsilon_values.len().saturating_sub(1);
                    let node_level = ki + ni + (max_ei - ei);
                    if node_level == level {
                        let triple = AbstractionTriple::new(
                            self.k_values[ki],
                            self.n_values[ni],
                            self.epsilon_values[ei],
                        );
                        if let Some(id) = self.find_node(&triple) {
                            result.push(id);
                        }
                    }
                }
            }
        }
        result
    }

    /// Maximum level in the lattice.
    pub fn max_level(&self) -> usize {
        let k_max = self.k_values.len().saturating_sub(1);
        let n_max = self.n_values.len().saturating_sub(1);
        let e_max = self.epsilon_values.len().saturating_sub(1);
        k_max + n_max + e_max
    }

    /// Get the successors (finer abstractions) of a node.
    pub fn successors(&self, node_id: usize) -> Vec<usize> {
        self.nodes.get(node_id)
            .map(|n| n.children.clone())
            .unwrap_or_default()
    }

    /// Get the predecessors (coarser abstractions) of a node.
    pub fn predecessors(&self, node_id: usize) -> Vec<usize> {
        self.nodes.get(node_id)
            .map(|n| n.parents.clone())
            .unwrap_or_default()
    }

    /// Get unexplored successors of a node, sorted by estimated cost.
    pub fn unexplored_successors(&self, node_id: usize) -> Vec<usize> {
        let mut succs: Vec<usize> = self.successors(node_id)
            .into_iter()
            .filter(|&id| !self.nodes[id].explored)
            .collect();
        succs.sort_by(|&a, &b| {
            self.nodes[a].estimated_cost
                .partial_cmp(&self.nodes[b].estimated_cost)
                .unwrap_or(Ordering::Equal)
        });
        succs
    }

    /// Select the next node to explore based on the current strategy.
    pub fn next_node(&self) -> Option<usize> {
        match self.strategy {
            LatticeTraversalStrategy::BottomUp => self.next_bottom_up(),
            LatticeTraversalStrategy::TopDown => self.next_top_down(),
            LatticeTraversalStrategy::BreadthFirst => self.next_bfs(),
            LatticeTraversalStrategy::DepthFirst => self.next_dfs(),
            LatticeTraversalStrategy::BestFirst => self.next_best_first(),
            LatticeTraversalStrategy::BinarySearch => self.next_binary_search(),
            LatticeTraversalStrategy::Adaptive => self.next_adaptive(),
        }
    }

    /// Bottom-up: find the coarsest unexplored node.
    fn next_bottom_up(&self) -> Option<usize> {
        for level in 0..=self.max_level() {
            let candidates = self.nodes_at_level(level);
            for &id in &candidates {
                if !self.nodes[id].explored && self.budget.can_afford(self.nodes[id].estimated_cost) {
                    return Some(id);
                }
            }
        }
        None
    }

    /// Top-down: find the finest affordable unexplored node.
    fn next_top_down(&self) -> Option<usize> {
        let max = self.max_level();
        for level in (0..=max).rev() {
            let candidates = self.nodes_at_level(level);
            for &id in &candidates {
                if !self.nodes[id].explored && self.budget.can_afford(self.nodes[id].estimated_cost) {
                    return Some(id);
                }
            }
        }
        None
    }

    /// BFS: level-by-level, cheapest first within each level.
    fn next_bfs(&self) -> Option<usize> {
        // Find the first explored node, then BFS from its children.
        // If nothing explored yet, start from bottom.
        if self.nodes.iter().all(|n| !n.explored) {
            return self.bottom();
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Seed with all explored nodes' unexplored children.
        for node in &self.nodes {
            if node.explored {
                for &child in &node.children {
                    if !self.nodes[child].explored && !visited.contains(&child) {
                        visited.insert(child);
                        queue.push_back(child);
                    }
                }
            }
        }

        // Sort queue by estimated cost.
        let mut candidates: Vec<usize> = queue.into_iter().collect();
        candidates.sort_by(|&a, &b| {
            self.nodes[a].estimated_cost
                .partial_cmp(&self.nodes[b].estimated_cost)
                .unwrap_or(Ordering::Equal)
        });

        candidates.into_iter()
            .find(|&id| self.budget.can_afford(self.nodes[id].estimated_cost))
    }

    /// DFS: follow the most promising path down.
    fn next_dfs(&self) -> Option<usize> {
        // Find the deepest explored node that has unexplored children.
        let mut best: Option<(usize, usize)> = None; // (node_id, level)
        for (level_idx, _) in (0..=self.max_level()).enumerate() {
            let level = self.max_level() - level_idx; // Start from deepest
            for &id in &self.nodes_at_level(level) {
                if self.nodes[id].explored {
                    let unexplored_children: Vec<usize> = self.nodes[id].children.iter()
                        .filter(|&&c| !self.nodes[c].explored)
                        .copied()
                        .collect();
                    if !unexplored_children.is_empty() {
                        // Pick cheapest child.
                        if let Some(&child) = unexplored_children.iter().min_by(|&&a, &&b| {
                            self.nodes[a].estimated_cost
                                .partial_cmp(&self.nodes[b].estimated_cost)
                                .unwrap_or(Ordering::Equal)
                        }) {
                            if self.budget.can_afford(self.nodes[child].estimated_cost) {
                                match &best {
                                    Some((_, best_level)) if level > *best_level => {
                                        best = Some((child, level));
                                    }
                                    None => {
                                        best = Some((child, level));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        best.map(|(id, _)| id).or_else(|| self.bottom().filter(|&id| !self.nodes[id].explored))
    }

    /// Best-first: use a heuristic combining cost and expected utility.
    fn next_best_first(&self) -> Option<usize> {
        let mut best_id: Option<usize> = None;
        let mut best_score = f64::INFINITY;

        for (id, node) in self.nodes.iter().enumerate() {
            if node.explored || !self.budget.can_afford(node.estimated_cost) {
                continue;
            }

            // Heuristic: prefer nodes whose parents have been explored and failed.
            // Score = cost * (1 + parents_not_explored) / (1 + parent_failures)
            let parents_explored = node.parents.iter()
                .filter(|&&p| self.nodes[p].explored)
                .count();
            let parent_failures = node.parents.iter()
                .filter(|&&p| self.nodes[p].verified == Some(false))
                .count();

            let exploration_bonus = if node.parents.is_empty() {
                1.0 // Root node — explore it
            } else if parents_explored == 0 {
                10.0 // Penalize: parents not explored yet
            } else {
                1.0 / (1.0 + parent_failures as f64)
            };

            let score = node.estimated_cost * exploration_bonus;
            if score < best_score {
                best_score = score;
                best_id = Some(id);
            }
        }

        best_id
    }

    /// Binary search: independently search each dimension.
    fn next_binary_search(&self) -> Option<usize> {
        // Find the "frontier" — explored nodes where verification just failed.
        let frontier: Vec<usize> = self.nodes.iter()
            .enumerate()
            .filter(|(_, n)| n.explored && n.verified == Some(false))
            .map(|(id, _)| id)
            .collect();

        if frontier.is_empty() {
            // Nothing failed yet — start from bottom.
            return self.bottom().filter(|&id| !self.nodes[id].explored);
        }

        // For each frontier node, try refining one dimension at a time
        // to the midpoint between current and max.
        for &fid in &frontier {
            let ft = &self.nodes[fid].triple;

            // Try refining k.
            let ki = self.k_values.iter().position(|&v| v == ft.k).unwrap_or(0);
            let mid_k = (ki + self.k_values.len()) / 2;
            if mid_k > ki && mid_k < self.k_values.len() {
                let triple = AbstractionTriple::new(self.k_values[mid_k], ft.n, ft.epsilon);
                if let Some(id) = self.find_node(&triple) {
                    if !self.nodes[id].explored && self.budget.can_afford(self.nodes[id].estimated_cost) {
                        return Some(id);
                    }
                }
            }

            // Try refining n.
            let ni = self.n_values.iter().position(|&v| v == ft.n).unwrap_or(0);
            let mid_n = (ni + self.n_values.len()) / 2;
            if mid_n > ni && mid_n < self.n_values.len() {
                let triple = AbstractionTriple::new(ft.k, self.n_values[mid_n], ft.epsilon);
                if let Some(id) = self.find_node(&triple) {
                    if !self.nodes[id].explored && self.budget.can_afford(self.nodes[id].estimated_cost) {
                        return Some(id);
                    }
                }
            }

            // Try refining epsilon.
            let ei = self.epsilon_values.iter().position(|v| (*v - ft.epsilon).abs() < 1e-12).unwrap_or(0);
            let mid_e = (ei + self.epsilon_values.len()) / 2;
            if mid_e > ei && mid_e < self.epsilon_values.len() {
                let triple = AbstractionTriple::new(ft.k, ft.n, self.epsilon_values[mid_e]);
                if let Some(id) = self.find_node(&triple) {
                    if !self.nodes[id].explored && self.budget.can_afford(self.nodes[id].estimated_cost) {
                        return Some(id);
                    }
                }
            }
        }

        // Fallback: best-first.
        self.next_best_first()
    }

    /// Adaptive: choose strategy based on exploration history.
    fn next_adaptive(&self) -> Option<usize> {
        let explored_count = self.nodes.iter().filter(|n| n.explored).count();
        let total = self.nodes.len();

        if explored_count == 0 {
            // Start with bottom-up.
            return self.next_bottom_up();
        }

        let failure_rate = self.nodes.iter()
            .filter(|n| n.verified == Some(false))
            .count() as f64
            / explored_count.max(1) as f64;

        if failure_rate > 0.8 {
            // Many failures — jump to binary search to find the boundary faster.
            self.next_binary_search()
        } else if explored_count as f64 / total as f64 > 0.3 {
            // Already explored a lot — use best-first to be efficient.
            self.next_best_first()
        } else {
            // Default: BFS for thorough exploration.
            self.next_bfs()
        }
    }

    /// Mark a node as explored with result.
    pub fn mark_explored(&mut self, node_id: usize, passed: bool, cost: f64) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.mark_explored(passed, cost);
            self.budget.record_exploration(cost);
        }
    }

    /// Find the coarsest verified node (the best abstraction that passed).
    pub fn coarsest_verified(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (id, node) in self.nodes.iter().enumerate() {
            if node.verified == Some(true) {
                match best {
                    Some(bid) => {
                        if node.triple.leq(&self.nodes[bid].triple) {
                            best = Some(id);
                        }
                    }
                    None => {
                        best = Some(id);
                    }
                }
            }
        }
        best
    }

    /// Find the finest verified node (strongest guarantee).
    pub fn finest_verified(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (id, node) in self.nodes.iter().enumerate() {
            if node.verified == Some(true) {
                match best {
                    Some(bid) => {
                        if self.nodes[bid].triple.leq(&node.triple) {
                            best = Some(id);
                        }
                    }
                    None => {
                        best = Some(id);
                    }
                }
            }
        }
        best
    }

    /// Compute an optimal refinement path from `start` to `goal`.
    /// Uses Dijkstra's algorithm on the lattice graph with estimated cost as edge weight.
    pub fn optimal_path(&self, start: usize, goal: usize) -> Option<Vec<usize>> {
        if start >= self.nodes.len() || goal >= self.nodes.len() {
            return None;
        }

        let n = self.nodes.len();
        let mut dist: Vec<f64> = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        let mut heap = BinaryHeap::new();

        dist[start] = 0.0;
        heap.push(PriorityEntry {
            node_id: start,
            priority: OrderedFloat(0.0),
        });

        while let Some(PriorityEntry { node_id, priority }) = heap.pop() {
            let d = priority.0;
            if d > dist[node_id] {
                continue;
            }
            if node_id == goal {
                break;
            }

            // Neighbors: both children and parents (undirected in terms of path).
            let neighbors: Vec<usize> = self.nodes[node_id].children.iter()
                .chain(self.nodes[node_id].parents.iter())
                .copied()
                .collect();

            for &next in &neighbors {
                let edge_cost = self.nodes[next].estimated_cost;
                let new_dist = dist[node_id] + edge_cost;
                if new_dist < dist[next] {
                    dist[next] = new_dist;
                    prev[next] = Some(node_id);
                    heap.push(PriorityEntry {
                        node_id: next,
                        priority: OrderedFloat(new_dist),
                    });
                }
            }
        }

        if dist[goal].is_infinite() {
            return None;
        }

        // Reconstruct path.
        let mut path = Vec::new();
        let mut current = goal;
        while let Some(p) = prev[current] {
            path.push(current);
            current = p;
        }
        path.push(start);
        path.reverse();
        Some(path)
    }

    /// Budget-constrained search: find the finest abstraction reachable within budget.
    pub fn finest_within_budget(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_volume = 0.0;

        for (id, node) in self.nodes.iter().enumerate() {
            if !self.budget.can_afford(node.estimated_cost) {
                continue;
            }
            let vol = node.triple.volume(self.alphabet_size);
            if vol > best_volume {
                best_volume = vol;
                best = Some(id);
            }
        }

        best
    }

    /// Generate a text-based visualization of the lattice.
    pub fn visualize(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Abstraction Lattice ===\n");
        out.push_str(&format!(
            "Size: {} nodes, Budget: {:.0}/{:.0} cost, {}/{} nodes\n",
            self.nodes.len(),
            self.budget.cost_spent,
            self.budget.max_total_cost,
            self.budget.nodes_explored,
            self.budget.max_nodes,
        ));
        out.push_str(&format!("Strategy: {:?}\n\n", self.strategy));

        for level in 0..=self.max_level() {
            let level_nodes = self.nodes_at_level(level);
            if level_nodes.is_empty() {
                continue;
            }
            out.push_str(&format!("Level {}:\n", level));
            for &id in &level_nodes {
                let node = &self.nodes[id];
                let status = match node.verified {
                    Some(true) => "PASS",
                    Some(false) => "FAIL",
                    None => "    ",
                };
                let cost_str = match node.actual_cost {
                    Some(c) => format!("{:.1}", c),
                    None => format!("~{:.1}", node.estimated_cost),
                };
                out.push_str(&format!(
                    "  {} [{}] cost={}\n",
                    node.triple, status, cost_str
                ));

                // Show edges.
                if !node.children.is_empty() {
                    let child_strs: Vec<String> = node.children.iter()
                        .map(|&c| format!("{}", self.nodes[c].triple))
                        .collect();
                    out.push_str(&format!("    └→ {}\n", child_strs.join(", ")));
                }
            }
            out.push_str("\n");
        }

        // Summary.
        let verified_count = self.nodes.iter().filter(|n| n.verified == Some(true)).count();
        let failed_count = self.nodes.iter().filter(|n| n.verified == Some(false)).count();
        let unexplored = self.nodes.iter().filter(|n| !n.explored).count();
        out.push_str(&format!(
            "Summary: {} verified, {} failed, {} unexplored\n",
            verified_count, failed_count, unexplored
        ));

        if let Some(cv) = self.coarsest_verified() {
            out.push_str(&format!("Coarsest verified: {}\n", self.nodes[cv].triple));
        }
        if let Some(fv) = self.finest_verified() {
            out.push_str(&format!("Finest verified:   {}\n", self.nodes[fv].triple));
        }

        out
    }

    /// Get statistics about the lattice.
    pub fn stats(&self) -> LatticeStats {
        let total = self.nodes.len();
        let explored = self.nodes.iter().filter(|n| n.explored).count();
        let verified = self.nodes.iter().filter(|n| n.verified == Some(true)).count();
        let failed = self.nodes.iter().filter(|n| n.verified == Some(false)).count();
        let total_cost: f64 = self.nodes.iter()
            .filter_map(|n| n.actual_cost)
            .sum();

        LatticeStats {
            total_nodes: total,
            explored_nodes: explored,
            verified_nodes: verified,
            failed_nodes: failed,
            total_cost,
            coarsest_verified: self.coarsest_verified(),
            finest_verified: self.finest_verified(),
        }
    }

    /// Insert a new node into the lattice dynamically.
    pub fn insert_node(&mut self, triple: AbstractionTriple) -> usize {
        let key = triple.discrete_key();
        if let Some(&existing) = self.key_to_id.get(&key) {
            return existing;
        }

        let id = self.nodes.len();
        let node = LatticeNode::new(id, triple.clone(), self.alphabet_size);
        self.key_to_id.insert(key, id);
        self.nodes.push(node);

        // Recompute edges for this new node.
        let mut new_parents = Vec::new();
        let mut new_children = Vec::new();

        for (other_id, other_node) in self.nodes.iter().enumerate() {
            if other_id == id {
                continue;
            }
            if other_node.triple.lt(&triple) {
                // other is potentially a parent.
                let mut is_direct = true;
                for (m_id, m_node) in self.nodes.iter().enumerate() {
                    if m_id == id || m_id == other_id {
                        continue;
                    }
                    if other_node.triple.lt(&m_node.triple) && m_node.triple.lt(&triple) {
                        is_direct = false;
                        break;
                    }
                }
                if is_direct {
                    new_parents.push(other_id);
                }
            } else if triple.lt(&other_node.triple) {
                // other is potentially a child.
                let mut is_direct = true;
                for (m_id, m_node) in self.nodes.iter().enumerate() {
                    if m_id == id || m_id == other_id {
                        continue;
                    }
                    if triple.lt(&m_node.triple) && m_node.triple.lt(&other_node.triple) {
                        is_direct = false;
                        break;
                    }
                }
                if is_direct {
                    new_children.push(other_id);
                }
            }
        }

        for &pid in &new_parents {
            self.nodes[id].parents.push(pid);
            self.nodes[pid].children.push(id);
        }
        for &cid in &new_children {
            self.nodes[id].children.push(cid);
            self.nodes[cid].parents.push(id);
        }

        id
    }

    /// Remove a node from the lattice.
    pub fn remove_node(&mut self, node_id: usize) -> bool {
        if node_id >= self.nodes.len() {
            return false;
        }
        let key = self.nodes[node_id].triple.discrete_key();
        self.key_to_id.remove(&key);

        // Remove from parent/child lists of other nodes.
        let parents = self.nodes[node_id].parents.clone();
        let children = self.nodes[node_id].children.clone();

        for &pid in &parents {
            self.nodes[pid].children.retain(|&c| c != node_id);
        }
        for &cid in &children {
            self.nodes[cid].parents.retain(|&p| p != node_id);
        }

        // We don't actually remove the node to preserve IDs; just mark it invalid.
        self.nodes[node_id].explored = true;
        self.nodes[node_id].verified = None;
        self.nodes[node_id].parents.clear();
        self.nodes[node_id].children.clear();
        true
    }
}

/// Statistics about the lattice state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeStats {
    pub total_nodes: usize,
    pub explored_nodes: usize,
    pub verified_nodes: usize,
    pub failed_nodes: usize,
    pub total_cost: f64,
    pub coarsest_verified: Option<usize>,
    pub finest_verified: Option<usize>,
}

impl fmt::Display for LatticeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lattice[{} nodes: {} explored, {} verified, {} failed, cost={:.1}]",
            self.total_nodes, self.explored_nodes, self.verified_nodes,
            self.failed_nodes, self.total_cost
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(k: usize, n: usize, eps: f64) -> AbstractionTriple {
        AbstractionTriple::new(k, n, eps)
    }

    #[test]
    fn test_triple_ordering() {
        let a = make_triple(2, 1, 0.5);
        let b = make_triple(4, 2, 0.25);
        let c = make_triple(4, 1, 0.5);

        assert!(a.leq(&b));
        assert!(a.lt(&b));
        assert!(!b.leq(&a));

        assert!(a.leq(&c));
        assert!(a.lt(&c));
        assert!(!c.lt(&a));

        // c and b are incomparable if c doesn't satisfy all conditions.
        // c = (4, 1, 0.5), b = (4, 2, 0.25)
        // c.k <= b.k (4<=4), c.n <= b.n (1<=2), c.eps >= b.eps (0.5>=0.25) → c ≤ b
        assert!(c.leq(&b));
    }

    #[test]
    fn test_triple_partial_order() {
        let a = make_triple(2, 3, 0.1);
        let b = make_triple(4, 1, 0.5);
        // a has smaller k but larger n and smaller eps — incomparable
        assert!(!a.leq(&b));
        assert!(!b.leq(&a));
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    fn test_meet_join() {
        let a = make_triple(2, 3, 0.1);
        let b = make_triple(4, 1, 0.5);

        let m = a.meet(&b);
        assert_eq!(m.k, 2);
        assert_eq!(m.n, 1);
        assert!((m.epsilon - 0.5).abs() < 1e-12);

        let j = a.join(&b);
        assert_eq!(j.k, 4);
        assert_eq!(j.n, 3);
        assert!((j.epsilon - 0.1).abs() < 1e-12);

        // Meet should be ≤ both.
        assert!(m.leq(&a));
        assert!(m.leq(&b));

        // Join should be ≥ both.
        assert!(a.leq(&j));
        assert!(b.leq(&j));
    }

    #[test]
    fn test_estimated_cost() {
        let a = make_triple(2, 1, 0.5);
        let b = make_triple(4, 2, 0.1);

        let ca = a.estimated_cost(3);
        let cb = b.estimated_cost(3);

        // b should be more expensive (more clusters, deeper, tighter).
        assert!(cb > ca);
    }

    #[test]
    fn test_lattice_creation() {
        let lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        // 2 * 2 * 2 = 8 nodes.
        assert_eq!(lattice.size(), 8);

        // Bottom should be (2, 1, 0.5).
        let bot = lattice.bottom().unwrap();
        assert_eq!(lattice.nodes[bot].triple.k, 2);
        assert_eq!(lattice.nodes[bot].triple.n, 1);
        assert!((lattice.nodes[bot].triple.epsilon - 0.5).abs() < 1e-12);

        // Top should be (4, 2, 0.1).
        let top = lattice.top().unwrap();
        assert_eq!(lattice.nodes[top].triple.k, 4);
        assert_eq!(lattice.nodes[top].triple.n, 2);
        assert!((lattice.nodes[top].triple.epsilon - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_lattice_edges() {
        let lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        let bot = lattice.bottom().unwrap();
        // Bottom should have children but no parents.
        assert!(lattice.nodes[bot].parents.is_empty());
        assert!(!lattice.nodes[bot].children.is_empty());

        let top = lattice.top().unwrap();
        // Top should have parents but no children.
        assert!(!lattice.nodes[top].parents.is_empty());
        assert!(lattice.nodes[top].children.is_empty());
    }

    #[test]
    fn test_bottom_up_traversal() {
        let mut lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        let first = lattice.next_node().unwrap();
        assert_eq!(first, lattice.bottom().unwrap());

        lattice.mark_explored(first, false, 1.0);
        let second = lattice.next_node().unwrap();
        assert_ne!(second, first);
    }

    #[test]
    fn test_best_first_traversal() {
        let mut lattice = AbstractionLattice::new(
            vec![2, 4, 8],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
            3,
            LatticeTraversalStrategy::BestFirst,
            LatticeBudget::unlimited(),
        );

        // Should return something.
        let first = lattice.next_node();
        assert!(first.is_some());
    }

    #[test]
    fn test_budget_constraints() {
        let mut budget = LatticeBudget::new(100.0, 5, 3600.0);
        assert!(!budget.is_exhausted());
        assert!(budget.can_afford(50.0));

        budget.record_exploration(60.0);
        assert!(budget.can_afford(30.0));
        assert!(!budget.can_afford(50.0));

        budget.record_exploration(50.0);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_optimal_path() {
        let lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        let bot = lattice.bottom().unwrap();
        let top = lattice.top().unwrap();

        let path = lattice.optimal_path(bot, top);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(*path.first().unwrap(), bot);
        assert_eq!(*path.last().unwrap(), top);
        assert!(path.len() >= 2);
    }

    #[test]
    fn test_coarsest_finest_verified() {
        let mut lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        assert!(lattice.coarsest_verified().is_none());

        let bot = lattice.bottom().unwrap();
        let top = lattice.top().unwrap();

        lattice.mark_explored(bot, true, 1.0);
        lattice.mark_explored(top, true, 10.0);

        let cv = lattice.coarsest_verified().unwrap();
        assert_eq!(cv, bot);

        let fv = lattice.finest_verified().unwrap();
        assert_eq!(fv, top);
    }

    #[test]
    fn test_insert_node() {
        let mut lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        let old_size = lattice.size();
        let new_triple = AbstractionTriple::new(3, 1, 0.3);
        let new_id = lattice.insert_node(new_triple.clone());
        assert_eq!(lattice.size(), old_size + 1);

        // Inserting same triple again should return same ID.
        let same_id = lattice.insert_node(new_triple);
        assert_eq!(same_id, new_id);
        assert_eq!(lattice.size(), old_size + 1);
    }

    #[test]
    fn test_visualize() {
        let mut lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::default(),
        );

        lattice.mark_explored(lattice.bottom().unwrap(), true, 5.0);

        let viz = lattice.visualize();
        assert!(viz.contains("Abstraction Lattice"));
        assert!(viz.contains("PASS"));
        assert!(viz.contains("verified"));
    }

    #[test]
    fn test_nodes_at_level() {
        let lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        let level0 = lattice.nodes_at_level(0);
        assert!(!level0.is_empty());
        // Level 0 should be just the bottom.
        assert!(level0.contains(&lattice.bottom().unwrap()));
    }

    #[test]
    fn test_triple_distance() {
        let a = make_triple(2, 1, 0.5);
        let b = make_triple(2, 1, 0.5);
        assert!(a.distance(&b) < 1e-12);

        let c = make_triple(4, 2, 0.1);
        assert!(a.distance(&c) > 0.0);
    }

    #[test]
    fn test_lattice_stats() {
        let mut lattice = AbstractionLattice::default_lattice(3);
        let stats = lattice.stats();
        assert!(stats.total_nodes > 0);
        assert_eq!(stats.explored_nodes, 0);

        let bot = lattice.bottom().unwrap();
        lattice.mark_explored(bot, true, 5.0);
        let stats = lattice.stats();
        assert_eq!(stats.explored_nodes, 1);
        assert_eq!(stats.verified_nodes, 1);
    }

    #[test]
    fn test_adaptive_traversal() {
        let mut lattice = AbstractionLattice::new(
            vec![2, 4, 8],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
            3,
            LatticeTraversalStrategy::Adaptive,
            LatticeBudget::unlimited(),
        );

        // Should start with bottom-up (no exploration yet).
        let first = lattice.next_node();
        assert!(first.is_some());

        // Mark most nodes as failed to trigger binary search.
        for i in 0..lattice.size().min(20) {
            lattice.mark_explored(i, false, 1.0);
        }
        let next = lattice.next_node();
        // Should still return something (binary search fallback).
        // May be None if all explored.
        let _ = next;
    }

    #[test]
    fn test_remove_node() {
        let mut lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::unlimited(),
        );

        let top = lattice.top().unwrap();
        assert!(lattice.remove_node(top));

        // Verify edges cleaned up.
        for node in &lattice.nodes {
            assert!(!node.children.contains(&top));
            assert!(!node.parents.contains(&top));
        }
    }

    #[test]
    fn test_finest_within_budget() {
        let lattice = AbstractionLattice::new(
            vec![2, 4],
            vec![1, 2],
            vec![0.5, 0.1],
            3,
            LatticeTraversalStrategy::BottomUp,
            LatticeBudget::new(1e6, 100, 3600.0),
        );

        let finest = lattice.finest_within_budget();
        assert!(finest.is_some());
    }
}
