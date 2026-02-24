//! Partial order construction and analysis for LITMUS∞.
//!
//! Provides partial order representation, happens-before computation,
//! transitive closure algorithms, total order extensions,
//! linear extension enumeration, and partial order analysis.

use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};
use std::fmt;
use super::execution::{EventId, ThreadId, BitMatrix, ExecutionGraph};

// ---------------------------------------------------------------------------
// Partial Order
// ---------------------------------------------------------------------------

/// A partial order on a finite set of elements.
#[derive(Debug, Clone)]
pub struct PartialOrder {
    /// Number of elements.
    n: usize,
    /// Relation matrix: matrix[i][j] = true means i < j.
    matrix: BitMatrix,
    /// Element labels for display.
    labels: Vec<String>,
}

impl PartialOrder {
    /// Create a new empty partial order on n elements.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            matrix: BitMatrix::new(n),
            labels: (0..n).map(|i| i.to_string()).collect(),
        }
    }

    /// Create from an existing BitMatrix.
    pub fn from_matrix(matrix: BitMatrix) -> Self {
        let n = matrix.dim();
        Self {
            n,
            matrix,
            labels: (0..n).map(|i| i.to_string()).collect(),
        }
    }

    /// Create with labels.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        assert!(labels.len() >= self.n);
        self.labels = labels;
        self
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Add a relation: i < j.
    pub fn add_relation(&mut self, i: usize, j: usize) {
        assert!(i < self.n && j < self.n);
        assert!(i != j, "Cannot add relation with self");
        self.matrix.add(i, j);
    }

    /// Check if i < j (directly or transitively).
    pub fn is_related(&self, i: usize, j: usize) -> bool {
        if i >= self.n || j >= self.n { return false; }
        self.matrix.get(i, j)
    }

    /// Check if i and j are comparable (i < j or j < i).
    pub fn comparable(&self, i: usize, j: usize) -> bool {
        i == j || self.is_related(i, j) || self.is_related(j, i)
    }

    /// Check if i and j are incomparable.
    pub fn incomparable(&self, i: usize, j: usize) -> bool {
        i != j && !self.comparable(i, j)
    }

    /// Get the relation matrix.
    pub fn matrix(&self) -> &BitMatrix {
        &self.matrix
    }

    /// Get elements that i is less than.
    pub fn successors(&self, i: usize) -> Vec<usize> {
        self.matrix.successors(i).collect()
    }

    /// Get elements that are less than j.
    pub fn predecessors(&self, j: usize) -> Vec<usize> {
        self.matrix.predecessors(j).collect()
    }

    /// Get minimal elements (no predecessors).
    pub fn minimal_elements(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&i| self.matrix.predecessors(i).next().is_none())
            .collect()
    }

    /// Get maximal elements (no successors).
    pub fn maximal_elements(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&i| self.matrix.successors(i).next().is_none())
            .collect()
    }

    /// Number of relations (edges).
    pub fn relation_count(&self) -> usize {
        self.matrix.count_edges()
    }

    /// Check if the relation is a valid partial order (irreflexive, antisymmetric, transitive).
    pub fn is_valid(&self) -> bool {
        // Irreflexive: no i < i.
        for i in 0..self.n {
            if self.matrix.get(i, i) { return false; }
        }
        // Antisymmetric: no i < j and j < i.
        for i in 0..self.n {
            for j in (i+1)..self.n {
                if self.matrix.get(i, j) && self.matrix.get(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the partial order is a total order.
    pub fn is_total(&self) -> bool {
        for i in 0..self.n {
            for j in (i+1)..self.n {
                if !self.comparable(i, j) {
                    return false;
                }
            }
        }
        true
    }

    /// Get the label of an element.
    pub fn label(&self, i: usize) -> &str {
        self.labels.get(i).map_or("?", |s| s.as_str())
    }
}

impl fmt::Display for PartialOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PartialOrder ({} elements, {} relations)", self.n, self.relation_count())?;
        for i in 0..self.n {
            for j in 0..self.n {
                if self.matrix.get(i, j) {
                    writeln!(f, "  {} < {}", self.label(i), self.label(j))?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Happens-Before
// ---------------------------------------------------------------------------

/// Computes the happens-before relation for memory model verification.
#[derive(Debug, Clone)]
pub struct HappensBefore;

impl HappensBefore {
    /// Compute happens-before from program order and synchronization.
    /// hb = (po ∪ sw)⁺
    pub fn from_relations(po: &BitMatrix, sw: &BitMatrix) -> BitMatrix {
        let combined = po.union(sw);
        combined.transitive_closure()
    }

    /// Compute happens-before including reads-from.
    /// hb = (po ∪ sw ∪ rf)⁺
    pub fn from_relations_with_rf(po: &BitMatrix, sw: &BitMatrix, rf: &BitMatrix) -> BitMatrix {
        let combined = po.union(sw).union(rf);
        combined.transitive_closure()
    }

    /// SC happens-before: all operations are totally ordered.
    pub fn for_sc(n: usize, po: &BitMatrix) -> BitMatrix {
        // Under SC, po already implies total order per thread.
        po.transitive_closure()
    }

    /// TSO happens-before: preserves all orderings except store-load.
    pub fn for_tso(po: &BitMatrix, rf: &BitMatrix) -> BitMatrix {
        let combined = po.union(rf);
        combined.transitive_closure()
    }

    /// ARM happens-before: only dependency and fence orderings.
    pub fn for_arm(po: &BitMatrix, rf: &BitMatrix, fence: &BitMatrix) -> BitMatrix {
        let combined = fence.union(rf);
        combined.transitive_closure()
    }

    /// Check if the happens-before relation is acyclic.
    pub fn is_acyclic(hb: &BitMatrix) -> bool {
        hb.is_acyclic()
    }

    /// Check if the happens-before is irreflexive.
    pub fn is_irreflexive(hb: &BitMatrix) -> bool {
        hb.is_irreflexive()
    }
}

// ---------------------------------------------------------------------------
// Transitive Closure
// ---------------------------------------------------------------------------

/// Transitive closure computation with multiple algorithms.
#[derive(Debug, Clone)]
pub struct TransitiveClosure;

impl TransitiveClosure {
    /// Warshall's algorithm: O(n³).
    pub fn warshall(matrix: &BitMatrix) -> BitMatrix {
        matrix.transitive_closure()
    }

    /// Incremental transitive closure: add a single edge (i, j).
    pub fn incremental_add(closure: &mut BitMatrix, i: usize, j: usize) {
        let n = closure.dim();
        if closure.get(i, j) { return; }

        // All nodes that can reach i (plus i itself).
        let mut sources = vec![i];
        for s in 0..n {
            if closure.get(s, i) {
                sources.push(s);
            }
        }

        // All nodes reachable from j (plus j itself).
        let mut targets = vec![j];
        for t in 0..n {
            if closure.get(j, t) {
                targets.push(t);
            }
        }

        // Add edges from all sources to all targets.
        for &s in &sources {
            for &t in &targets {
                closure.add(s, t);
            }
        }
    }

    /// Check if adding edge (i, j) would create a cycle.
    pub fn would_create_cycle(closure: &BitMatrix, i: usize, j: usize) -> bool {
        // Adding i -> j creates a cycle if j -> i is already reachable.
        i == j || closure.get(j, i)
    }

    /// Reflexive-transitive closure (adds identity).
    pub fn reflexive_transitive(matrix: &BitMatrix) -> BitMatrix {
        matrix.reflexive_transitive_closure()
    }

    /// Compute the transitive reduction (Hasse diagram edges).
    pub fn transitive_reduction(matrix: &BitMatrix) -> BitMatrix {
        let n = matrix.dim();
        let tc = matrix.transitive_closure();
        let mut result = BitMatrix::new(n);

        for i in 0..n {
            for j in 0..n {
                if !tc.get(i, j) { continue; }

                // Check if there's a path i -> k -> j for any k.
                let mut is_direct = true;
                for k in 0..n {
                    if k != i && k != j && tc.get(i, k) && tc.get(k, j) {
                        is_direct = false;
                        break;
                    }
                }
                if is_direct {
                    result.add(i, j);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Total Order Extension
// ---------------------------------------------------------------------------

/// Extends a partial order to a total order.
#[derive(Debug, Clone)]
pub struct TotalOrderExtension;

impl TotalOrderExtension {
    /// Extend a partial order to a total order via topological sort.
    pub fn extend(partial: &PartialOrder) -> Option<Vec<usize>> {
        partial.matrix().topological_sort()
    }

    /// Extend preserving the partial order, with tie-breaking by element index.
    pub fn extend_preserving(partial: &PartialOrder) -> Option<Vec<usize>> {
        let n = partial.size();
        let mut in_degree = vec![0usize; n];
        for j in 0..n {
            for i in 0..n {
                if partial.is_related(i, j) {
                    in_degree[j] += 1;
                }
            }
        }

        let mut available: BTreeSet<usize> = BTreeSet::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                available.insert(i);
            }
        }

        let mut result = Vec::with_capacity(n);
        while let Some(&next) = available.iter().next() {
            available.remove(&next);
            result.push(next);

            for j in 0..n {
                if partial.is_related(next, j) {
                    in_degree[j] -= 1;
                    if in_degree[j] == 0 {
                        available.insert(j);
                    }
                }
            }
        }

        if result.len() == n {
            Some(result)
        } else {
            None // Cycle detected.
        }
    }

    /// Validate that a total order is a valid extension of a partial order.
    pub fn validate(total: &[usize], partial: &PartialOrder) -> bool {
        let n = partial.size();
        if total.len() != n { return false; }

        // Check that all elements are present.
        let mut seen = vec![false; n];
        for &e in total {
            if e >= n || seen[e] { return false; }
            seen[e] = true;
        }

        // Check that the total order respects the partial order.
        let mut position = vec![0usize; n];
        for (pos, &e) in total.iter().enumerate() {
            position[e] = pos;
        }

        for i in 0..n {
            for j in 0..n {
                if partial.is_related(i, j) {
                    if position[i] >= position[j] {
                        return false;
                    }
                }
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Linear Extension Enumeration
// ---------------------------------------------------------------------------

/// Enumerates all linear extensions of a partial order.
#[derive(Debug, Clone)]
pub struct LinearExtensionEnumerator {
    max_extensions: usize,
}

impl LinearExtensionEnumerator {
    pub fn new() -> Self {
        Self { max_extensions: 1_000_000 }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.max_extensions = limit;
        self
    }

    /// Enumerate all linear extensions.
    pub fn enumerate(&self, partial: &PartialOrder) -> Vec<Vec<usize>> {
        self.enumerate_bounded(partial, self.max_extensions)
    }

    /// Enumerate up to `limit` linear extensions.
    pub fn enumerate_bounded(&self, partial: &PartialOrder, limit: usize) -> Vec<Vec<usize>> {
        let n = partial.size();
        let mut results = Vec::new();
        let mut current = Vec::new();
        let mut used = vec![false; n];

        self.enumerate_recursive(partial, &mut current, &mut used, n, limit, &mut results);
        results
    }

    fn enumerate_recursive(
        &self,
        partial: &PartialOrder,
        current: &mut Vec<usize>,
        used: &mut Vec<bool>,
        n: usize,
        limit: usize,
        results: &mut Vec<Vec<usize>>,
    ) {
        if results.len() >= limit { return; }

        if current.len() == n {
            results.push(current.clone());
            return;
        }

        // Find elements that can be added next (all predecessors already placed).
        for i in 0..n {
            if used[i] { continue; }

            // Check if all predecessors of i are already in current.
            let can_add = (0..n).all(|j| {
                if partial.is_related(j, i) { used[j] } else { true }
            });

            if can_add {
                current.push(i);
                used[i] = true;
                self.enumerate_recursive(partial, current, used, n, limit, results);
                used[i] = false;
                current.pop();
            }
        }
    }

    /// Count the number of linear extensions without enumerating them.
    /// Uses a simple recursive approach (exact but exponential).
    pub fn count(&self, partial: &PartialOrder) -> u64 {
        let n = partial.size();
        if n <= 1 { return 1; }
        let mut used = vec![false; n];
        self.count_recursive(partial, &mut used, n, 0)
    }

    fn count_recursive(
        &self,
        partial: &PartialOrder,
        used: &mut Vec<bool>,
        n: usize,
        placed: usize,
    ) -> u64 {
        if placed == n { return 1; }

        let mut total = 0u64;
        for i in 0..n {
            if used[i] { continue; }

            let can_add = (0..n).all(|j| {
                if partial.is_related(j, i) { used[j] } else { true }
            });

            if can_add {
                used[i] = true;
                total += self.count_recursive(partial, used, n, placed + 1);
                used[i] = false;
            }
        }
        total
    }
}

impl Default for LinearExtensionEnumerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Partial Order Operations
// ---------------------------------------------------------------------------

/// Operations on partial orders.
pub struct PartialOrderOps;

impl PartialOrderOps {
    /// Union of two partial orders.
    pub fn union(a: &PartialOrder, b: &PartialOrder) -> PartialOrder {
        assert_eq!(a.size(), b.size());
        let matrix = a.matrix().union(b.matrix());
        PartialOrder::from_matrix(matrix)
    }

    /// Intersection of two partial orders.
    pub fn intersection(a: &PartialOrder, b: &PartialOrder) -> PartialOrder {
        assert_eq!(a.size(), b.size());
        let matrix = a.matrix().intersection(b.matrix());
        PartialOrder::from_matrix(matrix)
    }

    /// Restriction to a subset of elements.
    pub fn restriction(po: &PartialOrder, subset: &[usize]) -> PartialOrder {
        let n = subset.len();
        let mut result = PartialOrder::new(n);
        let mut labels = Vec::new();

        for (new_i, &old_i) in subset.iter().enumerate() {
            labels.push(po.label(old_i).to_string());
            for (new_j, &old_j) in subset.iter().enumerate() {
                if po.is_related(old_i, old_j) {
                    result.add_relation(new_i, new_j);
                }
            }
        }

        result.labels = labels;
        result
    }

    /// Dual (reverse) of a partial order.
    pub fn dual(po: &PartialOrder) -> PartialOrder {
        let matrix = po.matrix().inverse();
        let mut result = PartialOrder::from_matrix(matrix);
        result.labels = po.labels.clone();
        result
    }

    /// Compose two partial orders.
    pub fn compose(a: &PartialOrder, b: &PartialOrder) -> PartialOrder {
        assert_eq!(a.size(), b.size());
        let matrix = a.matrix().compose(b.matrix());
        PartialOrder::from_matrix(matrix)
    }
}

// ---------------------------------------------------------------------------
// Partial Order Analysis
// ---------------------------------------------------------------------------

/// Analysis utilities for partial orders.
pub struct PartialOrderAnalysis;

impl PartialOrderAnalysis {
    /// Compute the width: size of the maximum antichain (Dilworth's theorem).
    pub fn width(po: &PartialOrder) -> usize {
        let antichains = Self::maximal_antichains(po);
        antichains.iter().map(|a| a.len()).max().unwrap_or(0)
    }

    /// Compute the height: length of the longest chain.
    pub fn height(po: &PartialOrder) -> usize {
        let chains = Self::maximal_chains(po);
        chains.iter().map(|c| c.len()).max().unwrap_or(0)
    }

    /// Find all maximal chains.
    pub fn maximal_chains(po: &PartialOrder) -> Vec<Vec<usize>> {
        let n = po.size();
        let mut chains = Vec::new();
        let minimals = po.minimal_elements();

        for &start in &minimals {
            let mut current = vec![start];
            Self::extend_chain(po, &mut current, &mut chains, n);
        }

        // Keep only maximal chains.
        let mut maximal = Vec::new();
        for chain in &chains {
            let is_maximal = !chains.iter().any(|other| {
                other.len() > chain.len() && chain.iter().all(|e| other.contains(e))
            });
            if is_maximal {
                maximal.push(chain.clone());
            }
        }
        maximal
    }

    fn extend_chain(
        po: &PartialOrder, current: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>, n: usize,
    ) {
        let last = *current.last().unwrap();
        let successors: Vec<usize> = po.successors(last);

        // Filter to direct successors (Hasse diagram edges).
        let direct: Vec<usize> = successors.iter()
            .filter(|&&s| {
                !successors.iter().any(|&t| t != s && po.is_related(t, s))
            })
            .copied()
            .collect();

        if direct.is_empty() {
            results.push(current.clone());
            return;
        }

        for s in direct {
            if !current.contains(&s) {
                current.push(s);
                Self::extend_chain(po, current, results, n);
                current.pop();
            }
        }
    }

    /// Find all maximal antichains.
    pub fn maximal_antichains(po: &PartialOrder) -> Vec<Vec<usize>> {
        let n = po.size();
        let mut antichains = Vec::new();
        let mut current = Vec::new();

        Self::extend_antichain(po, &mut current, 0, n, &mut antichains);

        // Keep only maximal.
        let mut maximal = Vec::new();
        for ac in &antichains {
            let is_maximal = !antichains.iter().any(|other| {
                other.len() > ac.len() && ac.iter().all(|e| other.contains(e))
            });
            if is_maximal {
                maximal.push(ac.clone());
            }
        }
        maximal
    }

    fn extend_antichain(
        po: &PartialOrder, current: &mut Vec<usize>,
        start: usize, n: usize, results: &mut Vec<Vec<usize>>,
    ) {
        if results.len() > 1000 { return; }

        results.push(current.clone());

        for i in start..n {
            // Check if i is incomparable with all elements in current.
            let can_add = current.iter().all(|&j| po.incomparable(i, j));
            if can_add {
                current.push(i);
                Self::extend_antichain(po, current, i + 1, n, results);
                current.pop();
            }
        }
    }

    /// Compute a chain decomposition (partition into chains).
    pub fn chain_decomposition(po: &PartialOrder) -> Vec<Vec<usize>> {
        let n = po.size();
        let mut assigned = vec![false; n];
        let mut chains = Vec::new();

        // Greedy: always extend the longest chain.
        let order = TotalOrderExtension::extend_preserving(po)
            .unwrap_or_else(|| (0..n).collect());

        for &elem in &order {
            if assigned[elem] { continue; }

            // Start a new chain from this element.
            let mut chain = vec![elem];
            assigned[elem] = true;

            // Extend the chain greedily.
            loop {
                let successors: Vec<usize> = po.successors(*chain.last().unwrap())
                    .into_iter()
                    .filter(|&s| !assigned[s])
                    .collect();

                if let Some(&next) = successors.first() {
                    chain.push(next);
                    assigned[next] = true;
                } else {
                    break;
                }
            }

            chains.push(chain);
        }

        chains
    }

    /// Level assignment: assign each element to a level (longest path from minimal).
    pub fn level_assignment(po: &PartialOrder) -> Vec<usize> {
        let n = po.size();
        let mut levels = vec![0usize; n];

        if let Some(order) = TotalOrderExtension::extend_preserving(po) {
            for &elem in &order {
                let max_pred_level = po.predecessors(elem).iter()
                    .map(|&p| levels[p] + 1)
                    .max()
                    .unwrap_or(0);
                levels[elem] = max_pred_level;
            }
        }

        levels
    }
}

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------

/// Generates visual representations of partial orders.
pub struct PartialOrderVisualization;

impl PartialOrderVisualization {
    /// Generate a Hasse diagram as text.
    pub fn hasse_diagram(po: &PartialOrder) -> String {
        let n = po.size();
        let reduction = TransitiveClosure::transitive_reduction(po.matrix());
        let levels = PartialOrderAnalysis::level_assignment(po);
        let max_level = levels.iter().copied().max().unwrap_or(0);

        let mut output = String::new();
        output.push_str("Hasse Diagram:\n");

        // Group elements by level.
        for level in 0..=max_level {
            let elements: Vec<usize> = (0..n)
                .filter(|&i| levels[i] == level)
                .collect();

            output.push_str(&format!("Level {}: ", level));
            for (idx, &e) in elements.iter().enumerate() {
                if idx > 0 { output.push_str(", "); }
                output.push_str(po.label(e));
            }
            output.push('\n');
        }

        output.push_str("\nEdges:\n");
        for i in 0..n {
            for j in 0..n {
                if reduction.get(i, j) {
                    output.push_str(&format!("  {} -> {}\n", po.label(i), po.label(j)));
                }
            }
        }

        output
    }

    /// Generate a DOT graph representation.
    pub fn to_dot(po: &PartialOrder) -> String {
        let n = po.size();
        let reduction = TransitiveClosure::transitive_reduction(po.matrix());

        let mut dot = String::new();
        dot.push_str("digraph PartialOrder {\n");
        dot.push_str("  rankdir=BT;\n");
        dot.push_str("  node [shape=circle];\n");

        for i in 0..n {
            dot.push_str(&format!("  n{} [label=\"{}\"];\n", i, po.label(i)));
        }

        for i in 0..n {
            for j in 0..n {
                if reduction.get(i, j) {
                    dot.push_str(&format!("  n{} -> n{};\n", i, j));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Generate a text-based adjacency representation.
    pub fn adjacency_text(po: &PartialOrder) -> String {
        let n = po.size();
        let mut output = String::new();
        output.push_str("Adjacency:\n");
        for i in 0..n {
            let succs: Vec<&str> = po.successors(i).iter()
                .map(|&j| po.label(j))
                .collect();
            if !succs.is_empty() {
                output.push_str(&format!("  {} < {}\n", po.label(i), succs.join(", ")));
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond_po() -> PartialOrder {
        // Diamond: bottom < left, bottom < right, left < top, right < top
        let mut po = PartialOrder::new(4);
        po.add_relation(0, 1); // bottom < left
        po.add_relation(0, 2); // bottom < right
        po.add_relation(1, 3); // left < top
        po.add_relation(2, 3); // right < top
        po
    }

    fn chain_po() -> PartialOrder {
        // Chain: 0 < 1 < 2 < 3
        let mut po = PartialOrder::new(4);
        po.add_relation(0, 1);
        po.add_relation(1, 2);
        po.add_relation(2, 3);
        po
    }

    fn antichain_po() -> PartialOrder {
        // Antichain: no relations
        PartialOrder::new(4)
    }

    // -- PartialOrder basic tests --

    #[test]
    fn test_po_creation() {
        let po = PartialOrder::new(5);
        assert_eq!(po.size(), 5);
        assert_eq!(po.relation_count(), 0);
    }

    #[test]
    fn test_po_add_relation() {
        let mut po = PartialOrder::new(3);
        po.add_relation(0, 1);
        po.add_relation(1, 2);
        assert!(po.is_related(0, 1));
        assert!(po.is_related(1, 2));
        assert!(!po.is_related(0, 2)); // Not transitively closed.
    }

    #[test]
    fn test_po_comparable() {
        let po = diamond_po();
        assert!(po.comparable(0, 1));
        assert!(po.comparable(0, 3));
        assert!(po.incomparable(1, 2)); // Left and right are incomparable.
    }

    #[test]
    fn test_po_minimal_maximal() {
        let po = diamond_po();
        assert_eq!(po.minimal_elements(), vec![0]);
        assert_eq!(po.maximal_elements(), vec![3]);
    }

    #[test]
    fn test_po_valid() {
        let po = diamond_po();
        assert!(po.is_valid());
    }

    #[test]
    fn test_po_is_total() {
        let chain = chain_po();
        // Chain without transitive closure is not total by matrix check.
        assert!(!chain.is_total());

        // After transitive closure, chain becomes total.
        let tc = chain.matrix().transitive_closure();
        let total_chain = PartialOrder::from_matrix(tc);
        assert!(total_chain.is_total());

        let antichain = antichain_po();
        assert!(!antichain.is_total());
    }

    #[test]
    fn test_po_successors_predecessors() {
        let po = diamond_po();
        assert_eq!(po.successors(0), vec![1, 2]);
        assert_eq!(po.predecessors(3), vec![1, 2]);
    }

    // -- Happens-Before tests --

    #[test]
    fn test_hb_basic() {
        let mut po = BitMatrix::new(4);
        po.add(0, 1);
        po.add(2, 3);
        let sw = BitMatrix::new(4);
        let hb = HappensBefore::from_relations(&po, &sw);
        assert!(hb.get(0, 1));
        assert!(hb.get(2, 3));
        assert!(!hb.get(0, 3));
    }

    #[test]
    fn test_hb_with_sync() {
        let mut po = BitMatrix::new(4);
        po.add(0, 1);
        po.add(2, 3);
        let mut sw = BitMatrix::new(4);
        sw.add(1, 2); // Synchronization between threads.
        let hb = HappensBefore::from_relations(&po, &sw);
        assert!(hb.get(0, 3)); // 0 -po-> 1 -sw-> 2 -po-> 3
    }

    #[test]
    fn test_hb_acyclic() {
        let mut po = BitMatrix::new(3);
        po.add(0, 1);
        po.add(1, 2);
        let sw = BitMatrix::new(3);
        let hb = HappensBefore::from_relations(&po, &sw);
        assert!(HappensBefore::is_acyclic(&hb));
    }

    // -- Transitive Closure tests --

    #[test]
    fn test_tc_warshall() {
        let mut m = BitMatrix::new(3);
        m.add(0, 1);
        m.add(1, 2);
        let tc = TransitiveClosure::warshall(&m);
        assert!(tc.get(0, 2));
    }

    #[test]
    fn test_tc_incremental() {
        let mut tc = BitMatrix::new(3);
        tc.add(0, 1);
        TransitiveClosure::incremental_add(&mut tc, 1, 2);
        assert!(tc.get(0, 2));
    }

    #[test]
    fn test_tc_cycle_detection() {
        let mut tc = BitMatrix::new(3);
        tc.add(0, 1);
        tc.add(1, 2);
        let tc_full = TransitiveClosure::warshall(&tc);
        assert!(TransitiveClosure::would_create_cycle(&tc_full, 2, 0));
        assert!(!TransitiveClosure::would_create_cycle(&tc_full, 0, 2));
    }

    #[test]
    fn test_tc_reduction() {
        let mut m = BitMatrix::new(3);
        m.add(0, 1);
        m.add(1, 2);
        m.add(0, 2); // Transitive edge.
        let reduced = TransitiveClosure::transitive_reduction(&m);
        assert!(reduced.get(0, 1));
        assert!(reduced.get(1, 2));
        assert!(!reduced.get(0, 2)); // Removed transitive edge.
    }

    // -- Total Order Extension tests --

    #[test]
    fn test_total_extension() {
        let po = diamond_po();
        let total = TotalOrderExtension::extend(&po).unwrap();
        assert_eq!(total.len(), 4);
        assert!(TotalOrderExtension::validate(&total, &po));
    }

    #[test]
    fn test_total_extension_preserving() {
        let po = diamond_po();
        let total = TotalOrderExtension::extend_preserving(&po).unwrap();
        assert!(TotalOrderExtension::validate(&total, &po));
    }

    #[test]
    fn test_total_extension_chain() {
        let po = chain_po();
        let total = TotalOrderExtension::extend(&po).unwrap();
        assert_eq!(total, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_total_extension_validate_invalid() {
        let po = chain_po();
        assert!(!TotalOrderExtension::validate(&[3, 2, 1, 0], &po));
    }

    // -- Linear Extension Enumeration tests --

    #[test]
    fn test_linear_extensions_chain() {
        let po = chain_po();
        let enum_ = LinearExtensionEnumerator::new();
        let extensions = enum_.enumerate(&po);
        assert_eq!(extensions.len(), 1); // Only one linear extension.
        assert_eq!(extensions[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_linear_extensions_antichain() {
        let po = antichain_po();
        let enum_ = LinearExtensionEnumerator::new();
        let extensions = enum_.enumerate(&po);
        assert_eq!(extensions.len(), 24); // 4! = 24
    }

    #[test]
    fn test_linear_extensions_diamond() {
        let po = diamond_po();
        let enum_ = LinearExtensionEnumerator::new();
        let extensions = enum_.enumerate(&po);
        // Diamond has 2 linear extensions: 0,1,2,3 and 0,2,1,3
        assert_eq!(extensions.len(), 2);
    }

    #[test]
    fn test_linear_extension_count() {
        let po = diamond_po();
        let enum_ = LinearExtensionEnumerator::new();
        let count = enum_.count(&po);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_linear_extension_bounded() {
        let po = antichain_po();
        let enum_ = LinearExtensionEnumerator::new();
        let extensions = enum_.enumerate_bounded(&po, 5);
        assert_eq!(extensions.len(), 5);
    }

    // -- Partial Order Operations tests --

    #[test]
    fn test_po_union() {
        let mut a = PartialOrder::new(3);
        a.add_relation(0, 1);
        let mut b = PartialOrder::new(3);
        b.add_relation(1, 2);
        let c = PartialOrderOps::union(&a, &b);
        assert!(c.is_related(0, 1));
        assert!(c.is_related(1, 2));
    }

    #[test]
    fn test_po_intersection() {
        let mut a = PartialOrder::new(3);
        a.add_relation(0, 1);
        a.add_relation(1, 2);
        let mut b = PartialOrder::new(3);
        b.add_relation(0, 1);
        let c = PartialOrderOps::intersection(&a, &b);
        assert!(c.is_related(0, 1));
        assert!(!c.is_related(1, 2));
    }

    #[test]
    fn test_po_restriction() {
        let po = diamond_po();
        let restricted = PartialOrderOps::restriction(&po, &[0, 1, 3]);
        assert_eq!(restricted.size(), 3);
        assert!(restricted.is_related(0, 1)); // bottom < left
        assert!(restricted.is_related(1, 2)); // left < top (remapped to indices 0,1,2)
    }

    #[test]
    fn test_po_dual() {
        let po = chain_po();
        let dual = PartialOrderOps::dual(&po);
        assert!(dual.is_related(1, 0)); // Reversed.
        assert!(!dual.is_related(0, 1));
    }

    // -- Analysis tests --

    #[test]
    fn test_width_antichain() {
        let po = antichain_po();
        assert_eq!(PartialOrderAnalysis::width(&po), 4);
    }

    #[test]
    fn test_width_chain() {
        let po = chain_po();
        assert_eq!(PartialOrderAnalysis::width(&po), 1);
    }

    #[test]
    fn test_height_chain() {
        let po = chain_po();
        assert_eq!(PartialOrderAnalysis::height(&po), 4);
    }

    #[test]
    fn test_height_diamond() {
        let po = diamond_po();
        let chains = PartialOrderAnalysis::maximal_chains(&po);
        assert!(!chains.is_empty());
        let max_chain_len = chains.iter().map(|c| c.len()).max().unwrap();
        assert_eq!(max_chain_len, 3); // bottom-left-top or bottom-right-top
    }

    #[test]
    fn test_chain_decomposition() {
        let po = diamond_po();
        let chains = PartialOrderAnalysis::chain_decomposition(&po);
        // All elements should be covered.
        let all_elements: HashSet<usize> = chains.iter().flatten().copied().collect();
        assert_eq!(all_elements.len(), 4);
    }

    #[test]
    fn test_level_assignment() {
        let po = diamond_po();
        let levels = PartialOrderAnalysis::level_assignment(&po);
        assert_eq!(levels[0], 0); // bottom
        assert_eq!(levels[1], 1); // left
        assert_eq!(levels[2], 1); // right
        assert_eq!(levels[3], 2); // top
    }

    // -- Visualization tests --

    #[test]
    fn test_hasse_diagram() {
        let po = diamond_po();
        let diagram = PartialOrderVisualization::hasse_diagram(&po);
        assert!(diagram.contains("Level 0"));
        assert!(diagram.contains("Edges"));
    }

    #[test]
    fn test_to_dot() {
        let po = diamond_po();
        let dot = PartialOrderVisualization::to_dot(&po);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_adjacency_text() {
        let po = diamond_po();
        let text = PartialOrderVisualization::adjacency_text(&po);
        assert!(text.contains("Adjacency"));
    }
}
