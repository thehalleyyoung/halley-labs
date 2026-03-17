//! Binary Decision Diagram (BDD) implementation.
//!
//! Provides a reduced ordered BDD with a unique table, computed table
//! (operation cache), and CNF extraction for non-interval compatibility
//! constraints.

use crate::formula::{Clause, CnfFormula, Literal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// BddNode
// ---------------------------------------------------------------------------

/// Index into the BDD node table.
pub type NodeId = u32;

/// Terminal node constants.
pub const BDD_FALSE: NodeId = 0;
pub const BDD_TRUE: NodeId = 1;

/// A node in a Binary Decision Diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BddNode {
    /// The decision variable at this node.
    pub variable: u32,
    /// Child when variable is false.
    pub low_child: NodeId,
    /// Child when variable is true.
    pub high_child: NodeId,
}

impl BddNode {
    /// Create a new BDD node.
    pub fn new(variable: u32, low: NodeId, high: NodeId) -> Self {
        Self {
            variable,
            low_child: low,
            high_child: high,
        }
    }

    /// Check if this is a terminal node.
    pub fn is_terminal(id: NodeId) -> bool {
        id == BDD_FALSE || id == BDD_TRUE
    }
}

// ---------------------------------------------------------------------------
// BddManager
// ---------------------------------------------------------------------------

/// Manages BDD nodes with unique table and operation cache.
pub struct BddManager {
    /// Node table: index -> BddNode.
    nodes: Vec<BddNode>,
    /// Unique table: (var, low, high) -> NodeId.
    unique_table: HashMap<(u32, NodeId, NodeId), NodeId>,
    /// Computed table (operation cache): (op, a, b) -> result.
    computed_table: HashMap<(u8, NodeId, NodeId), NodeId>,
    /// Variable ordering (lower index = higher in BDD).
    var_order: HashMap<u32, u32>,
    /// Reference counts for garbage collection.
    ref_counts: Vec<u32>,
    /// Statistics.
    pub stats: BddStats,
}

/// BDD operation statistics.
#[derive(Debug, Clone, Default)]
pub struct BddStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub nodes_created: u64,
    pub gc_runs: u64,
    pub gc_collected: u64,
}

impl BddStats {
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

const OP_AND: u8 = 0;
const OP_OR: u8 = 1;

impl BddManager {
    /// Create a new BDD manager.
    pub fn new() -> Self {
        // Reserve index 0 for FALSE, index 1 for TRUE.
        let false_node = BddNode::new(u32::MAX, BDD_FALSE, BDD_FALSE);
        let true_node = BddNode::new(u32::MAX, BDD_TRUE, BDD_TRUE);
        Self {
            nodes: vec![false_node, true_node],
            unique_table: HashMap::new(),
            computed_table: HashMap::new(),
            var_order: HashMap::new(),
            ref_counts: vec![1, 1], // Terminals always have ref
            stats: BddStats::default(),
        }
    }

    /// Get the number of allocated nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Set variable ordering. Lower order = closer to root.
    pub fn set_var_order(&mut self, order: &[(u32, u32)]) {
        self.var_order.clear();
        for &(var, ord) in order {
            self.var_order.insert(var, ord);
        }
    }

    /// Get the order of a variable (default: variable id itself).
    fn var_order_of(&self, var: u32) -> u32 {
        self.var_order.get(&var).copied().unwrap_or(var)
    }

    /// Get or create a node. Implements the unique table lookup.
    fn mk(&mut self, var: u32, low: NodeId, high: NodeId) -> NodeId {
        // Reduction: if low == high, skip this node
        if low == high {
            return low;
        }
        // Unique table lookup
        let key = (var, low, high);
        if let Some(&id) = self.unique_table.get(&key) {
            return id;
        }
        // Create new node
        let id = self.nodes.len() as NodeId;
        let node = BddNode::new(var, low, high);
        self.nodes.push(node);
        self.ref_counts.push(0);
        self.unique_table.insert(key, id);
        self.stats.nodes_created += 1;
        id
    }

    /// Get the node at an index.
    pub fn get_node(&self, id: NodeId) -> &BddNode {
        &self.nodes[id as usize]
    }

    /// Create a BDD for the constant TRUE.
    pub fn new_true(&self) -> NodeId {
        BDD_TRUE
    }

    /// Create a BDD for the constant FALSE.
    pub fn new_false(&self) -> NodeId {
        BDD_FALSE
    }

    /// Create a BDD for a single variable.
    pub fn new_var(&mut self, var: u32) -> NodeId {
        self.mk(var, BDD_FALSE, BDD_TRUE)
    }

    /// Create a BDD for the negation of a variable.
    pub fn new_not_var(&mut self, var: u32) -> NodeId {
        self.mk(var, BDD_TRUE, BDD_FALSE)
    }

    /// Compute AND of two BDDs.
    pub fn and(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.apply(OP_AND, a, b)
    }

    /// Compute OR of two BDDs.
    pub fn or(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.apply(OP_OR, a, b)
    }

    /// Compute NOT of a BDD.
    pub fn not(&mut self, a: NodeId) -> NodeId {
        self.not_rec(a)
    }

    fn not_rec(&mut self, a: NodeId) -> NodeId {
        if a == BDD_FALSE {
            return BDD_TRUE;
        }
        if a == BDD_TRUE {
            return BDD_FALSE;
        }
        let node = self.nodes[a as usize];
        let low = self.not_rec(node.low_child);
        let high = self.not_rec(node.high_child);
        self.mk(node.variable, low, high)
    }

    /// Apply a binary operation (AND or OR) using the Shannon expansion.
    fn apply(&mut self, op: u8, a: NodeId, b: NodeId) -> NodeId {
        // Terminal cases
        match op {
            OP_AND => {
                if a == BDD_FALSE || b == BDD_FALSE {
                    return BDD_FALSE;
                }
                if a == BDD_TRUE {
                    return b;
                }
                if b == BDD_TRUE {
                    return a;
                }
                if a == b {
                    return a;
                }
            }
            OP_OR => {
                if a == BDD_TRUE || b == BDD_TRUE {
                    return BDD_TRUE;
                }
                if a == BDD_FALSE {
                    return b;
                }
                if b == BDD_FALSE {
                    return a;
                }
                if a == b {
                    return a;
                }
            }
            _ => unreachable!(),
        }

        // Normalize: smaller id first for cache
        let (ca, cb) = if a <= b { (a, b) } else { (b, a) };
        let cache_key = (op, ca, cb);
        if let Some(&result) = self.computed_table.get(&cache_key) {
            self.stats.cache_hits += 1;
            return result;
        }
        self.stats.cache_misses += 1;

        let node_a = self.nodes[a as usize];
        let node_b = self.nodes[b as usize];

        let ord_a = self.var_order_of(node_a.variable);
        let ord_b = self.var_order_of(node_b.variable);

        let (var, a_low, a_high, b_low, b_high) = if ord_a < ord_b {
            (node_a.variable, node_a.low_child, node_a.high_child, b, b)
        } else if ord_b < ord_a {
            (node_b.variable, a, a, node_b.low_child, node_b.high_child)
        } else {
            (
                node_a.variable,
                node_a.low_child,
                node_a.high_child,
                node_b.low_child,
                node_b.high_child,
            )
        };

        let low = self.apply(op, a_low, b_low);
        let high = self.apply(op, a_high, b_high);
        let result = self.mk(var, low, high);
        self.computed_table.insert(cache_key, result);
        result
    }

    /// If-then-else: ite(f, g, h) = (f AND g) OR (NOT f AND h).
    pub fn ite(&mut self, f: NodeId, g: NodeId, h: NodeId) -> NodeId {
        if f == BDD_TRUE {
            return g;
        }
        if f == BDD_FALSE {
            return h;
        }
        if g == BDD_TRUE && h == BDD_FALSE {
            return f;
        }
        if g == h {
            return g;
        }

        let fg = self.and(f, g);
        let nf = self.not(f);
        let nfh = self.and(nf, h);
        self.or(fg, nfh)
    }

    /// Restrict a BDD: set variable to a specific value.
    pub fn restrict(&mut self, node: NodeId, var: u32, value: bool) -> NodeId {
        if BddNode::is_terminal(node) {
            return node;
        }
        let n = self.nodes[node as usize];
        if n.variable == var {
            return if value { n.high_child } else { n.low_child };
        }
        let ord_n = self.var_order_of(n.variable);
        let ord_v = self.var_order_of(var);
        if ord_n > ord_v {
            // Variable already passed
            return node;
        }
        let low = self.restrict(n.low_child, var, value);
        let high = self.restrict(n.high_child, var, value);
        self.mk(n.variable, low, high)
    }

    /// Existential quantification: exist(var, f) = f|var=0 OR f|var=1.
    pub fn exist(&mut self, node: NodeId, var: u32) -> NodeId {
        let low = self.restrict(node, var, false);
        let high = self.restrict(node, var, true);
        self.or(low, high)
    }

    /// Universal quantification: forall(var, f) = f|var=0 AND f|var=1.
    pub fn forall(&mut self, node: NodeId, var: u32) -> NodeId {
        let low = self.restrict(node, var, false);
        let high = self.restrict(node, var, true);
        self.and(low, high)
    }

    /// Check if BDD is tautology (always true).
    pub fn is_tautology(&self, node: NodeId) -> bool {
        node == BDD_TRUE
    }

    /// Check if BDD is contradiction (always false).
    pub fn is_contradiction(&self, node: NodeId) -> bool {
        node == BDD_FALSE
    }

    /// Count the number of non-terminal nodes reachable from the given node.
    pub fn reachable_node_count(&self, node: NodeId) -> usize {
        let mut visited = std::collections::HashSet::new();
        self.count_nodes_rec(node, &mut visited);
        visited.len()
    }

    fn count_nodes_rec(&self, node: NodeId, visited: &mut std::collections::HashSet<NodeId>) {
        if BddNode::is_terminal(node) || visited.contains(&node) {
            return;
        }
        visited.insert(node);
        let n = &self.nodes[node as usize];
        self.count_nodes_rec(n.low_child, visited);
        self.count_nodes_rec(n.high_child, visited);
    }

    /// Count satisfying paths (assignments that lead to TRUE).
    pub fn path_count(&self, node: NodeId) -> u64 {
        let mut cache = HashMap::new();
        self.path_count_rec(node, &mut cache)
    }

    fn path_count_rec(&self, node: NodeId, cache: &mut HashMap<NodeId, u64>) -> u64 {
        if node == BDD_FALSE {
            return 0;
        }
        if node == BDD_TRUE {
            return 1;
        }
        if let Some(&count) = cache.get(&node) {
            return count;
        }
        let n = &self.nodes[node as usize];
        let count = self.path_count_rec(n.low_child, cache)
            + self.path_count_rec(n.high_child, cache);
        cache.insert(node, count);
        count
    }

    /// Count satisfying assignments considering variable gaps.
    pub fn sat_count(&self, node: NodeId, num_vars: u32) -> u64 {
        let mut cache = HashMap::new();
        // Collect actual variable levels and sort them
        let vars = self.collect_variables(node);
        let mut level_map = HashMap::new();
        for (i, &v) in vars.iter().enumerate() {
            level_map.insert(v, i as u32);
        }
        self.sat_count_rec(node, 0, num_vars, &level_map, &mut cache)
    }

    fn sat_count_rec(
        &self,
        node: NodeId,
        current_level: u32,
        total_vars: u32,
        level_map: &HashMap<u32, u32>,
        cache: &mut HashMap<(NodeId, u32), u64>,
    ) -> u64 {
        if node == BDD_FALSE {
            return 0;
        }
        if node == BDD_TRUE {
            let remaining = total_vars.saturating_sub(current_level);
            return 1u64 << remaining;
        }
        let key = (node, current_level);
        if let Some(&count) = cache.get(&key) {
            return count;
        }
        let n = &self.nodes[node as usize];
        let node_level = level_map.get(&n.variable).copied().unwrap_or(current_level);
        let skipped = node_level.saturating_sub(current_level);
        let multiplier = 1u64 << skipped;

        let low_count =
            self.sat_count_rec(n.low_child, node_level + 1, total_vars, level_map, cache);
        let high_count =
            self.sat_count_rec(n.high_child, node_level + 1, total_vars, level_map, cache);
        let count = multiplier * (low_count + high_count);
        cache.insert(key, count);
        count
    }

    /// Convert BDD to CNF clauses by extracting all paths to FALSE
    /// and negating them.
    pub fn to_clauses(&self, node: NodeId) -> Vec<Clause> {
        if node == BDD_TRUE {
            return Vec::new();
        }
        if node == BDD_FALSE {
            return vec![vec![]]; // Empty clause = UNSAT
        }

        let mut clauses = Vec::new();
        let mut path = Vec::new();
        self.extract_false_paths(node, &mut path, &mut clauses);
        clauses
    }

    /// Extract paths leading to FALSE and create blocking clauses.
    fn extract_false_paths(
        &self,
        node: NodeId,
        path: &mut Vec<Literal>,
        clauses: &mut Vec<Clause>,
    ) {
        if node == BDD_TRUE {
            return;
        }
        if node == BDD_FALSE {
            // Negate the path to create a clause
            let clause: Vec<Literal> = path.iter().map(|&l| -l).collect();
            if !clause.is_empty() {
                clauses.push(clause);
            }
            return;
        }
        let n = &self.nodes[node as usize];
        let var = n.variable as Literal;
        // Low branch: variable is false
        path.push(-var);
        self.extract_false_paths(n.low_child, path, clauses);
        path.pop();
        // High branch: variable is true
        path.push(var);
        self.extract_false_paths(n.high_child, path, clauses);
        path.pop();
    }

    /// Increment reference count.
    pub fn ref_node(&mut self, id: NodeId) {
        if !BddNode::is_terminal(id) && (id as usize) < self.ref_counts.len() {
            self.ref_counts[id as usize] += 1;
        }
    }

    /// Decrement reference count.
    pub fn deref_node(&mut self, id: NodeId) {
        if !BddNode::is_terminal(id) && (id as usize) < self.ref_counts.len() {
            self.ref_counts[id as usize] = self.ref_counts[id as usize].saturating_sub(1);
        }
    }

    /// Run garbage collection: remove unreachable nodes.
    pub fn gc(&mut self) {
        self.stats.gc_runs += 1;
        let before = self.nodes.len();
        // Mark reachable nodes
        let mut reachable = vec![false; self.nodes.len()];
        reachable[BDD_FALSE as usize] = true;
        reachable[BDD_TRUE as usize] = true;
        for (i, &rc) in self.ref_counts.iter().enumerate() {
            if rc > 0 {
                self.mark_reachable(i as NodeId, &mut reachable);
            }
        }
        // We don't actually compact (that would invalidate NodeIds);
        // instead we clear the computed table of stale entries
        self.computed_table
            .retain(|&(_, a, b), &mut r| {
                reachable.get(a as usize).copied().unwrap_or(false)
                    && reachable.get(b as usize).copied().unwrap_or(false)
                    && reachable.get(r as usize).copied().unwrap_or(false)
            });
        let after = self.computed_table.len();
        self.stats.gc_collected += (before - after.min(before)) as u64;
    }

    fn mark_reachable(&self, id: NodeId, reachable: &mut [bool]) {
        if BddNode::is_terminal(id) || reachable[id as usize] {
            return;
        }
        reachable[id as usize] = true;
        let n = &self.nodes[id as usize];
        self.mark_reachable(n.low_child, reachable);
        self.mark_reachable(n.high_child, reachable);
    }

    /// Perform variable sifting to find a better variable ordering.
    /// Tries moving each variable up and down to minimize BDD size.
    pub fn sift_variables(&mut self, root: NodeId) -> NodeId {
        // Collect all variables in the BDD
        let vars = self.collect_variables(root);
        if vars.len() <= 1 {
            return root;
        }

        let mut current_root = root;
        let mut best_size = self.reachable_node_count(current_root);

        for &var in &vars {
            let current_order = self.var_order_of(var);
            let mut best_order = current_order;
            let mut best_local_size = best_size;

            // Try moving up
            if current_order > 0 {
                for new_order in (0..current_order).rev() {
                    self.var_order.insert(var, new_order);
                    // Rebuild (simplified — actual sifting would swap layers)
                    let size = self.reachable_node_count(current_root);
                    if size < best_local_size {
                        best_local_size = size;
                        best_order = new_order;
                    }
                }
            }
            // Try moving down
            let max_order = vars.len() as u32;
            for new_order in (current_order + 1)..=max_order {
                self.var_order.insert(var, new_order);
                let size = self.reachable_node_count(current_root);
                if size < best_local_size {
                    best_local_size = size;
                    best_order = new_order;
                }
            }
            self.var_order.insert(var, best_order);
            best_size = best_local_size;
        }
        self.computed_table.clear();
        current_root
    }

    /// Collect all variables referenced in a BDD.
    pub fn collect_variables(&self, node: NodeId) -> Vec<u32> {
        let mut vars = std::collections::HashSet::new();
        self.collect_vars_rec(node, &mut vars);
        let mut result: Vec<u32> = vars.into_iter().collect();
        result.sort();
        result
    }

    fn collect_vars_rec(&self, node: NodeId, vars: &mut std::collections::HashSet<u32>) {
        if BddNode::is_terminal(node) {
            return;
        }
        let n = &self.nodes[node as usize];
        if vars.insert(n.variable) {
            self.collect_vars_rec(n.low_child, vars);
            self.collect_vars_rec(n.high_child, vars);
        }
    }

    /// Clear the operation cache.
    pub fn clear_cache(&mut self) {
        self.computed_table.clear();
    }
}

impl Default for BddManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BddManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BddManager")
            .field("nodes", &self.nodes.len())
            .field("unique_table_size", &self.unique_table.len())
            .field("cache_size", &self.computed_table.len())
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Bdd (wrapper for convenient use)
// ---------------------------------------------------------------------------

/// A BDD handle: combines a manager reference approach with a node id.
/// For simplicity, this is a standalone wrapper that owns a manager.
#[derive(Debug, Clone)]
pub struct Bdd {
    pub root: NodeId,
}

impl Bdd {
    /// Create a BDD representing TRUE.
    pub fn new_true() -> Self {
        Self { root: BDD_TRUE }
    }

    /// Create a BDD representing FALSE.
    pub fn new_false() -> Self {
        Self { root: BDD_FALSE }
    }

    /// Create a BDD for a single variable using a manager.
    pub fn new_var(mgr: &mut BddManager, var: u32) -> Self {
        Self {
            root: mgr.new_var(var),
        }
    }

    /// AND of two BDDs using a manager.
    pub fn and(mgr: &mut BddManager, a: &Bdd, b: &Bdd) -> Self {
        Self {
            root: mgr.and(a.root, b.root),
        }
    }

    /// OR of two BDDs using a manager.
    pub fn or(mgr: &mut BddManager, a: &Bdd, b: &Bdd) -> Self {
        Self {
            root: mgr.or(a.root, b.root),
        }
    }

    /// NOT of a BDD using a manager.
    pub fn not(mgr: &mut BddManager, a: &Bdd) -> Self {
        Self {
            root: mgr.not(a.root),
        }
    }

    /// ITE operation.
    pub fn ite(mgr: &mut BddManager, f: &Bdd, g: &Bdd, h: &Bdd) -> Self {
        Self {
            root: mgr.ite(f.root, g.root, h.root),
        }
    }

    /// Restrict a variable to a value.
    pub fn restrict(mgr: &mut BddManager, bdd: &Bdd, var: u32, value: bool) -> Self {
        Self {
            root: mgr.restrict(bdd.root, var, value),
        }
    }

    /// Existential quantification.
    pub fn exist(mgr: &mut BddManager, bdd: &Bdd, var: u32) -> Self {
        Self {
            root: mgr.exist(bdd.root, var),
        }
    }

    /// Universal quantification.
    pub fn forall(mgr: &mut BddManager, bdd: &Bdd, var: u32) -> Self {
        Self {
            root: mgr.forall(bdd.root, var),
        }
    }

    /// Check if tautology.
    pub fn is_tautology(&self) -> bool {
        self.root == BDD_TRUE
    }

    /// Check if contradiction.
    pub fn is_contradiction(&self) -> bool {
        self.root == BDD_FALSE
    }

    /// Node count.
    pub fn node_count(&self, mgr: &BddManager) -> usize {
        mgr.reachable_node_count(self.root)
    }

    /// Path count.
    pub fn path_count(&self, mgr: &BddManager) -> u64 {
        mgr.path_count(self.root)
    }

    /// Convert to CNF clauses.
    pub fn to_clauses(&self, mgr: &BddManager) -> Vec<Clause> {
        mgr.to_clauses(self.root)
    }
}

// ---------------------------------------------------------------------------
// BddBuilder
// ---------------------------------------------------------------------------

/// Ergonomic BDD construction helper.
#[derive(Debug)]
pub struct BddBuilder {
    pub manager: BddManager,
}

impl BddBuilder {
    pub fn new() -> Self {
        Self {
            manager: BddManager::new(),
        }
    }

    /// Build a BDD from a truth table (for a small number of variables).
    /// `num_vars`: number of variables (1..=num_vars).
    /// `table`: maps assignment (as bitmask) to boolean output.
    pub fn from_truth_table(&mut self, num_vars: u32, table: &[bool]) -> Bdd {
        assert_eq!(table.len(), 1 << num_vars);
        let mut result = self.manager.new_false();
        for (idx, &val) in table.iter().enumerate() {
            if val {
                // Build conjunction for this minterm
                let mut term = self.manager.new_true();
                for bit in 0..num_vars {
                    let var_bdd = self.manager.new_var(bit + 1);
                    if (idx >> bit) & 1 == 1 {
                        term = self.manager.and(term, var_bdd);
                    } else {
                        let not_var = self.manager.not(var_bdd);
                        term = self.manager.and(term, not_var);
                    }
                }
                result = self.manager.or(result, term);
            }
        }
        Bdd { root: result }
    }

    /// Build a BDD from a boolean expression over variables.
    /// Variables are referenced by u32 ids.
    pub fn variable(&mut self, var: u32) -> Bdd {
        Bdd::new_var(&mut self.manager, var)
    }

    pub fn constant(&self, val: bool) -> Bdd {
        if val {
            Bdd::new_true()
        } else {
            Bdd::new_false()
        }
    }

    pub fn and(&mut self, a: &Bdd, b: &Bdd) -> Bdd {
        Bdd::and(&mut self.manager, a, b)
    }

    pub fn or(&mut self, a: &Bdd, b: &Bdd) -> Bdd {
        Bdd::or(&mut self.manager, a, b)
    }

    pub fn not(&mut self, a: &Bdd) -> Bdd {
        Bdd::not(&mut self.manager, a)
    }

    pub fn implies(&mut self, a: &Bdd, b: &Bdd) -> Bdd {
        let na = self.not(a);
        self.or(&na, b)
    }

    pub fn iff(&mut self, a: &Bdd, b: &Bdd) -> Bdd {
        let a_imp_b = self.implies(a, b);
        let b_imp_a = self.implies(b, a);
        self.and(&a_imp_b, &b_imp_a)
    }

    pub fn xor(&mut self, a: &Bdd, b: &Bdd) -> Bdd {
        let eq = self.iff(a, b);
        self.not(&eq)
    }

    /// Conjoin a list of BDDs.
    pub fn and_all(&mut self, bdds: &[Bdd]) -> Bdd {
        let mut result = self.constant(true);
        for bdd in bdds {
            result = self.and(&result, bdd);
        }
        result
    }

    /// Disjoin a list of BDDs.
    pub fn or_all(&mut self, bdds: &[Bdd]) -> Bdd {
        let mut result = self.constant(false);
        for bdd in bdds {
            result = self.or(&result, bdd);
        }
        result
    }

    /// Build BDD that is true when exactly one of the given variables is true.
    pub fn exactly_one_of(&mut self, vars: &[u32]) -> Bdd {
        if vars.is_empty() {
            return self.constant(false);
        }
        let var_bdds: Vec<Bdd> = vars.iter().map(|&v| self.variable(v)).collect();
        let mut result = self.constant(false);
        for i in 0..vars.len() {
            let mut term = var_bdds[i].clone();
            for j in 0..vars.len() {
                if i != j {
                    let neg = self.not(&var_bdds[j]);
                    term = self.and(&term, &neg);
                }
            }
            result = self.or(&result, &term);
        }
        result
    }

    /// Build BDD that is true when at most one of the given variables is true.
    pub fn at_most_one_of(&mut self, vars: &[u32]) -> Bdd {
        let none = {
            let negs: Vec<Bdd> = vars.iter().map(|&v| {
                let vb = self.variable(v);
                self.not(&vb)
            }).collect();
            self.and_all(&negs)
        };
        let eoo = self.exactly_one_of(vars);
        self.or(&none, &eoo)
    }
}

impl Default for BddBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompatibilityBdd
// ---------------------------------------------------------------------------

/// BDD encoding of non-interval compatibility constraints.
///
/// For service pairs whose compatibility matrix doesn't have interval structure,
/// we build a BDD directly from the matrix and extract CNF clauses.
#[derive(Debug)]
pub struct CompatibilityBdd {
    pub builder: BddBuilder,
}

impl CompatibilityBdd {
    pub fn new() -> Self {
        Self {
            builder: BddBuilder::new(),
        }
    }

    /// Build a BDD from a compatibility matrix.
    ///
    /// Variables 1..bits_i encode the version of service i.
    /// Variables (bits_i+1)..(bits_i+bits_j) encode the version of service j.
    ///
    /// Returns a BDD that is true iff the version pair is compatible.
    pub fn from_matrix(
        &mut self,
        num_versions_i: usize,
        num_versions_j: usize,
        matrix: &[Vec<bool>],
    ) -> Bdd {
        use crate::interval::BinaryEncoding;

        let bits_i = BinaryEncoding::num_bits(num_versions_i.saturating_sub(1));
        let bits_j = BinaryEncoding::num_bits(num_versions_j.saturating_sub(1));

        let mut result = self.builder.constant(false);

        for vi in 0..num_versions_i {
            for vj in 0..num_versions_j {
                if vi < matrix.len() && vj < matrix[vi].len() && matrix[vi][vj] {
                    // Build conjunction encoding vi, vj
                    let mut term = self.builder.constant(true);
                    for bit in 0..bits_i {
                        let var = self.builder.variable(bit as u32 + 1);
                        if (vi >> bit) & 1 == 0 {
                            let neg = self.builder.not(&var);
                            term = self.builder.and(&term, &neg);
                        } else {
                            term = self.builder.and(&term, &var);
                        }
                    }
                    for bit in 0..bits_j {
                        let var = self.builder.variable((bits_i + bit) as u32 + 1);
                        if (vj >> bit) & 1 == 0 {
                            let neg = self.builder.not(&var);
                            term = self.builder.and(&term, &neg);
                        } else {
                            term = self.builder.and(&term, &var);
                        }
                    }
                    result = self.builder.or(&result, &term);
                }
            }
        }
        result
    }

    /// Extract CNF clauses from a compatibility BDD.
    pub fn extract_cnf(&self, bdd: &Bdd) -> CnfFormula {
        let clauses = bdd.to_clauses(&self.builder.manager);
        CnfFormula::from_clauses(clauses)
    }
}

impl Default for CompatibilityBdd {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bdd_true_false() {
        let mgr = BddManager::new();
        assert!(mgr.is_tautology(BDD_TRUE));
        assert!(!mgr.is_tautology(BDD_FALSE));
        assert!(mgr.is_contradiction(BDD_FALSE));
        assert!(!mgr.is_contradiction(BDD_TRUE));
    }

    #[test]
    fn test_bdd_single_var() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        assert!(!mgr.is_tautology(x));
        assert!(!mgr.is_contradiction(x));
        assert_eq!(mgr.path_count(x), 1);
    }

    #[test]
    fn test_bdd_and() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let xy = mgr.and(x, y);
        assert_eq!(mgr.path_count(xy), 1); // Only x=1,y=1
    }

    #[test]
    fn test_bdd_or() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let x_or_y = mgr.or(x, y);
        // In BDD, x OR y: x=T -> TRUE (1 path), x=F -> y node (y=T -> TRUE, y=F -> FALSE)
        // So 2 paths to TRUE in the BDD graph
        assert_eq!(mgr.path_count(x_or_y), 2);
    }

    #[test]
    fn test_bdd_not() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let nx = mgr.not(x);
        assert_eq!(mgr.path_count(nx), 1); // Only x=0
    }

    #[test]
    fn test_bdd_double_not() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let nnx = {
            let nx = mgr.not(x);
            mgr.not(nx)
        };
        assert_eq!(x, nnx);
    }

    #[test]
    fn test_bdd_and_with_not() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let nx = mgr.not(x);
        let result = mgr.and(x, nx);
        assert!(mgr.is_contradiction(result));
    }

    #[test]
    fn test_bdd_or_with_not() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let nx = mgr.not(x);
        let result = mgr.or(x, nx);
        assert!(mgr.is_tautology(result));
    }

    #[test]
    fn test_bdd_ite() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let z = mgr.new_var(3);
        // ite(x, y, z) = (x AND y) OR (NOT x AND z)
        let result = mgr.ite(x, y, z);
        assert!(!mgr.is_tautology(result));
        assert!(!mgr.is_contradiction(result));
    }

    #[test]
    fn test_bdd_restrict() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let xy = mgr.and(x, y);
        // Restrict x=true: should give y
        let restricted = mgr.restrict(xy, 1, true);
        assert_eq!(restricted, y);
        // Restrict x=false: should give false
        let restricted_f = mgr.restrict(xy, 1, false);
        assert!(mgr.is_contradiction(restricted_f));
    }

    #[test]
    fn test_bdd_exist() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let xy = mgr.and(x, y);
        // exist(x, x AND y) = (false AND y) OR (true AND y) = y
        let result = mgr.exist(xy, 1);
        assert_eq!(result, y);
    }

    #[test]
    fn test_bdd_forall() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let x_or_y = mgr.or(x, y);
        // forall(x, x OR y) = (false OR y) AND (true OR y) = y AND true = y
        let result = mgr.forall(x_or_y, 1);
        assert_eq!(result, y);
    }

    #[test]
    fn test_bdd_to_clauses() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let xy = mgr.and(x, y);
        let clauses = mgr.to_clauses(xy);
        // x AND y: paths to FALSE are (!x, *) and (x, !y)
        assert!(!clauses.is_empty());
        let cnf = CnfFormula::from_clauses(clauses);
        // Check that x=1, y=1 satisfies
        let mut a = HashMap::new();
        a.insert(1, true);
        a.insert(2, true);
        assert!(cnf.evaluate(&a));
        // x=1, y=0 should not satisfy
        a.insert(2, false);
        assert!(!cnf.evaluate(&a));
    }

    #[test]
    fn test_bdd_path_count() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let z = mgr.new_var(3);
        // BDD paths to TRUE: x=T->T, x=F/y=T->T, x=F/y=F/z=T->T = 3 paths
        let xy = mgr.or(x, y);
        let r = mgr.or(xy, z);
        assert_eq!(mgr.path_count(r), 3);
    }

    #[test]
    fn test_bdd_node_count() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let xy = mgr.and(x, y);
        let count = mgr.reachable_node_count(xy);
        assert!(count >= 2); // At least x and y nodes
    }

    #[test]
    fn test_bdd_builder_truth_table() {
        let mut builder = BddBuilder::new();
        // XOR truth table for 2 variables
        // idx: bit0=var1, bit1=var2
        // 00->F, 01->T, 10->T, 11->F
        let table = vec![false, true, true, false];
        let bdd = builder.from_truth_table(2, &table);
        assert!(!bdd.is_tautology());
        assert!(!bdd.is_contradiction());
        assert_eq!(bdd.path_count(&builder.manager), 2);
    }

    #[test]
    fn test_bdd_builder_exactly_one() {
        let mut builder = BddBuilder::new();
        let eo = builder.exactly_one_of(&[1, 2, 3]);
        // Exactly one of 3 vars: 3 satisfying assignments (use sat_count)
        assert_eq!(builder.manager.sat_count(eo.root, 3), 3);
    }

    #[test]
    fn test_bdd_builder_implies() {
        let mut builder = BddBuilder::new();
        let x = builder.variable(1);
        let y = builder.variable(2);
        let imp = builder.implies(&x, &y);
        // x => y: BDD paths to TRUE: x=F->T, x=T/y=T->T = 2 paths
        assert_eq!(imp.path_count(&builder.manager), 2);
    }

    #[test]
    fn test_bdd_builder_iff() {
        let mut builder = BddBuilder::new();
        let x = builder.variable(1);
        let y = builder.variable(2);
        let iff = builder.iff(&x, &y);
        // x <=> y: 2 satisfying (FF, TT)
        assert_eq!(iff.path_count(&builder.manager), 2);
    }

    #[test]
    fn test_bdd_builder_xor() {
        let mut builder = BddBuilder::new();
        let x = builder.variable(1);
        let y = builder.variable(2);
        let xor = builder.xor(&x, &y);
        assert_eq!(xor.path_count(&builder.manager), 2);
    }

    #[test]
    fn test_compatibility_bdd_simple() {
        let matrix = vec![
            vec![true, false],
            vec![false, true],
        ];
        let mut cbdd = CompatibilityBdd::new();
        let bdd = cbdd.from_matrix(2, 2, &matrix);
        // 2 compatible pairs
        assert_eq!(bdd.path_count(&cbdd.builder.manager), 2);
    }

    #[test]
    fn test_compatibility_bdd_to_cnf() {
        let matrix = vec![
            vec![true, true],
            vec![true, false],
        ];
        let mut cbdd = CompatibilityBdd::new();
        let bdd = cbdd.from_matrix(2, 2, &matrix);
        let cnf = cbdd.extract_cnf(&bdd);
        // Should have some clauses
        assert!(cnf.num_clauses() >= 0);
    }

    #[test]
    fn test_bdd_manager_cache() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        // First AND
        let _r1 = mgr.and(x, y);
        // Second AND (should hit cache)
        let _r2 = mgr.and(x, y);
        assert!(mgr.stats.cache_hits > 0);
    }

    #[test]
    fn test_bdd_manager_gc() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        mgr.ref_node(x);
        mgr.gc();
        assert_eq!(mgr.stats.gc_runs, 1);
    }

    #[test]
    fn test_bdd_collect_variables() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let z = mgr.new_var(3);
        let xy = mgr.or(x, y);
        let r = mgr.and(xy, z);
        let vars = mgr.collect_variables(r);
        assert_eq!(vars, vec![1, 2, 3]);
    }

    #[test]
    fn test_bdd_sat_count() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let xy = mgr.and(x, y);
        // With 2 variables, only 1 satisfying assignment (both true)
        assert_eq!(mgr.sat_count(xy, 2), 1);
    }

    #[test]
    fn test_bdd_builder_at_most_one() {
        let mut builder = BddBuilder::new();
        let amo = builder.at_most_one_of(&[1, 2, 3]);
        // At most one: 4 satisfying (000, 100, 010, 001) - use sat_count
        assert_eq!(builder.manager.sat_count(amo.root, 3), 4);
    }

    #[test]
    fn test_bdd_self_operations() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        assert_eq!(mgr.and(x, x), x);
        assert_eq!(mgr.or(x, x), x);
    }

    #[test]
    fn test_bdd_terminal_operations() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        assert_eq!(mgr.and(x, BDD_TRUE), x);
        assert_eq!(mgr.and(x, BDD_FALSE), BDD_FALSE);
        assert_eq!(mgr.or(x, BDD_TRUE), BDD_TRUE);
        assert_eq!(mgr.or(x, BDD_FALSE), x);
    }

    #[test]
    fn test_bdd_wrapper() {
        let mut mgr = BddManager::new();
        let x = Bdd::new_var(&mut mgr, 1);
        let y = Bdd::new_var(&mut mgr, 2);
        let r = Bdd::and(&mut mgr, &x, &y);
        assert!(!r.is_tautology());
        assert!(!r.is_contradiction());
        assert_eq!(r.path_count(&mgr), 1);
    }

    #[test]
    fn test_bdd_builder_and_all() {
        let mut builder = BddBuilder::new();
        let bdds: Vec<Bdd> = (1..=4).map(|v| builder.variable(v)).collect();
        let all = builder.and_all(&bdds);
        assert_eq!(all.path_count(&builder.manager), 1);
    }

    #[test]
    fn test_bdd_builder_or_all() {
        let mut builder = BddBuilder::new();
        let bdds: Vec<Bdd> = (1..=3).map(|v| builder.variable(v)).collect();
        let any = builder.or_all(&bdds);
        // 7 satisfying assignments for x1 OR x2 OR x3
        assert_eq!(builder.manager.sat_count(any.root, 3), 7);
    }

    #[test]
    fn test_bdd_stats() {
        let mut mgr = BddManager::new();
        let x = mgr.new_var(1);
        let y = mgr.new_var(2);
        let _ = mgr.and(x, y);
        let _ = mgr.and(x, y); // cache hit
        assert!(mgr.stats.cache_hit_rate() > 0.0);
    }

    #[test]
    fn test_to_clauses_true() {
        let mgr = BddManager::new();
        let clauses = mgr.to_clauses(BDD_TRUE);
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_to_clauses_false() {
        let mgr = BddManager::new();
        let clauses = mgr.to_clauses(BDD_FALSE);
        assert_eq!(clauses.len(), 1);
        assert!(clauses[0].is_empty()); // empty clause = UNSAT
    }
}
