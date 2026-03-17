//! Mutant subsumption analysis.
//!
//! **Subsumption relation**: mutant *m₁* subsumes mutant *m₂* (written m₁ ≻ m₂)
//! iff every input that triggers *m₂*'s error also triggers *m₁*'s error:
//!
//!   E(m₂) ⊆ E(m₁)
//!
//! Equivalently, every test that kills *m₂* also kills *m₁*.
//!
//! This module provides three detection strategies:
//! 1. **Dynamic** – infer subsumption from a kill matrix.
//! 2. **Static** – confirm subsumption via SMT implication checking.
//! 3. **Hybrid** – use dynamic detection first, optionally confirm with static.
//!
//! The subsumption graph is a directed graph where edges represent subsumption
//! relations.  We also compute the *transitive reduction* to remove redundant
//! edges.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use indexmap::IndexMap;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use crate::{
    CoverageError, Formula, KillMatrix, MutantDescriptor, MutantId, MutationOperator, Result,
    SmtSolver, SolverResult,
};

// ────────────────────────────────────────────────────────────────────────────
// Subsumption edge
// ────────────────────────────────────────────────────────────────────────────

/// An edge in the subsumption graph from *subsumer* to *subsumed*.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubsumptionEdge {
    /// The mutant that subsumes (harder to kill).
    pub subsumer: MutantId,
    /// The mutant that is subsumed (easier to kill).
    pub subsumed: MutantId,
    /// Confidence in this relation.
    pub confidence: SubsumptionConfidence,
}

/// How confidently we know the subsumption relation holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SubsumptionConfidence {
    /// Observed from dynamic analysis only (may be coincidental).
    Dynamic,
    /// Confirmed by SMT solver.
    Static,
    /// Dynamic + static confirmation.
    Hybrid,
}

impl fmt::Display for SubsumptionConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dynamic => write!(f, "dynamic"),
            Self::Static => write!(f, "static"),
            Self::Hybrid => write!(f, "hybrid"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for subsumption analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsumptionConfig {
    /// Whether to use dynamic detection.
    pub use_dynamic: bool,
    /// Whether to use static (SMT) detection.
    pub use_static: bool,
    /// Whether to compute the transitive reduction.
    pub compute_transitive_reduction: bool,
    /// Whether to detect equivalence classes.
    pub detect_equivalence_classes: bool,
    /// Maximum number of SMT queries before giving up.
    pub max_smt_queries: usize,
    /// Timeout per SMT query in milliseconds.
    pub smt_timeout_ms: u64,
}

impl Default for SubsumptionConfig {
    fn default() -> Self {
        Self {
            use_dynamic: true,
            use_static: false,
            compute_transitive_reduction: true,
            detect_equivalence_classes: true,
            max_smt_queries: 10_000,
            smt_timeout_ms: 5_000,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Subsumption graph
// ────────────────────────────────────────────────────────────────────────────

/// Directed graph of subsumption relations among mutants.
///
/// Nodes are mutant IDs, edges point from subsumer to subsumed:
/// an edge (A → B) means A subsumes B.
#[derive(Debug, Clone)]
pub struct SubsumptionGraph {
    /// petgraph directed graph.
    graph: DiGraph<MutantId, SubsumptionConfidence>,
    /// Map from mutant ID to node index.
    node_map: IndexMap<MutantId, NodeIndex>,
    /// Whether the transitive reduction has been computed.
    is_reduced: bool,
}

impl SubsumptionGraph {
    /// Create an empty subsumption graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: IndexMap::new(),
            is_reduced: false,
        }
    }

    /// Create a graph pre-populated with nodes for the given mutant IDs.
    pub fn with_mutants(mutant_ids: &[MutantId]) -> Self {
        let mut g = Self::new();
        for id in mutant_ids {
            g.ensure_node(id.clone());
        }
        g
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns true if the graph has been transitively reduced.
    pub fn is_reduced(&self) -> bool {
        self.is_reduced
    }

    /// Ensure a node exists for the given mutant, returning its index.
    pub fn ensure_node(&mut self, id: MutantId) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&id) {
            idx
        } else {
            let idx = self.graph.add_node(id.clone());
            self.node_map.insert(id, idx);
            idx
        }
    }

    /// Get node index for a mutant ID.
    pub fn node_index(&self, id: &MutantId) -> Option<NodeIndex> {
        self.node_map.get(id).copied()
    }

    /// Add a subsumption edge: `subsumer` subsumes `subsumed`.
    pub fn add_edge(
        &mut self,
        subsumer: MutantId,
        subsumed: MutantId,
        confidence: SubsumptionConfidence,
    ) {
        let a = self.ensure_node(subsumer);
        let b = self.ensure_node(subsumed);
        if a != b && self.graph.find_edge(a, b).is_none() {
            self.graph.add_edge(a, b, confidence);
            self.is_reduced = false;
        }
    }

    /// Check if `subsumer` directly subsumes `subsumed`.
    pub fn has_edge(&self, subsumer: &MutantId, subsumed: &MutantId) -> bool {
        if let (Some(&a), Some(&b)) = (self.node_map.get(subsumer), self.node_map.get(subsumed)) {
            self.graph.find_edge(a, b).is_some()
        } else {
            false
        }
    }

    /// Get all edges as SubsumptionEdge structs.
    pub fn edges(&self) -> Vec<SubsumptionEdge> {
        self.graph
            .edge_references()
            .map(|e| SubsumptionEdge {
                subsumer: self.graph[e.source()].clone(),
                subsumed: self.graph[e.target()].clone(),
                confidence: *e.weight(),
            })
            .collect()
    }

    /// Get the set of mutants that `id` directly subsumes.
    pub fn subsumed_by(&self, id: &MutantId) -> BTreeSet<MutantId> {
        if let Some(&idx) = self.node_map.get(id) {
            self.graph
                .edges_directed(idx, Direction::Outgoing)
                .map(|e| self.graph[e.target()].clone())
                .collect()
        } else {
            BTreeSet::new()
        }
    }

    /// Get the set of mutants that directly subsume `id`.
    pub fn subsumers_of(&self, id: &MutantId) -> BTreeSet<MutantId> {
        if let Some(&idx) = self.node_map.get(id) {
            self.graph
                .edges_directed(idx, Direction::Incoming)
                .map(|e| self.graph[e.source()].clone())
                .collect()
        } else {
            BTreeSet::new()
        }
    }

    /// Get all mutant IDs that are "roots" (not subsumed by anything).
    pub fn roots(&self) -> Vec<MutantId> {
        self.node_map
            .iter()
            .filter(|(_, &idx)| {
                self.graph
                    .edges_directed(idx, Direction::Incoming)
                    .next()
                    .is_none()
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get all mutant IDs that are "leaves" (do not subsume anything).
    pub fn leaves(&self) -> Vec<MutantId> {
        self.node_map
            .iter()
            .filter(|(_, &idx)| {
                self.graph
                    .edges_directed(idx, Direction::Outgoing)
                    .next()
                    .is_none()
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get all mutant IDs in the graph.
    pub fn mutant_ids(&self) -> Vec<MutantId> {
        self.node_map.keys().cloned().collect()
    }

    /// Get the in-degree of a node (number of mutants that subsume it).
    pub fn in_degree(&self, id: &MutantId) -> usize {
        self.node_map
            .get(id)
            .map(|&idx| self.graph.edges_directed(idx, Direction::Incoming).count())
            .unwrap_or(0)
    }

    /// Get the out-degree of a node (number of mutants it subsumes).
    pub fn out_degree(&self, id: &MutantId) -> usize {
        self.node_map
            .get(id)
            .map(|&idx| self.graph.edges_directed(idx, Direction::Outgoing).count())
            .unwrap_or(0)
    }

    /// Compute the transitive reduction of the graph.
    ///
    /// After reduction, if A→B and B→C exist, the edge A→C is removed because
    /// it is implied by transitivity.
    pub fn transitive_reduction(&mut self) {
        if self.is_reduced {
            return;
        }

        // Compute the transitive closure via DFS from each node, then remove
        // edges that are implied by paths of length ≥ 2.
        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();

        // For each node, find all transitively reachable nodes.
        let mut reachable: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();

        // Sort topologically if possible; fall back to arbitrary order.
        let order = match toposort(&self.graph, None) {
            Ok(sorted) => sorted,
            Err(_) => nodes.clone(),
        };

        // Compute transitive closure bottom-up (reverse topological order).
        for &node in order.iter().rev() {
            let mut reach = HashSet::new();
            for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                reach.insert(neighbor);
                if let Some(neighbor_reach) = reachable.get(&neighbor) {
                    reach.extend(neighbor_reach.iter().copied());
                }
            }
            reachable.insert(node, reach);
        }

        // Identify edges to remove: edge (u, v) is redundant if v is reachable
        // from u through some other path (i.e., through a neighbor w of u where
        // v is transitively reachable from w).
        let mut edges_to_remove = Vec::new();
        for &u in &nodes {
            let neighbors: Vec<NodeIndex> = self
                .graph
                .neighbors_directed(u, Direction::Outgoing)
                .collect();
            for &v in &neighbors {
                // Check if v is reachable from u through some other neighbor.
                let redundant = neighbors
                    .iter()
                    .any(|&w| w != v && reachable.get(&w).map_or(false, |r| r.contains(&v)));
                if redundant {
                    edges_to_remove.push((u, v));
                }
            }
        }

        // Remove redundant edges.
        for (u, v) in edges_to_remove {
            if let Some(edge) = self.graph.find_edge(u, v) {
                self.graph.remove_edge(edge);
            }
        }

        self.is_reduced = true;
    }

    /// Compute the transitive closure (add all implied edges).
    pub fn transitive_closure(&self) -> SubsumptionGraph {
        let mut result = self.clone();
        let nodes: Vec<NodeIndex> = result.graph.node_indices().collect();

        // Warshall's algorithm.
        for &k in &nodes {
            for &i in &nodes {
                for &j in &nodes {
                    if i != j
                        && result.graph.find_edge(i, k).is_some()
                        && result.graph.find_edge(k, j).is_some()
                        && result.graph.find_edge(i, j).is_none()
                    {
                        result.graph.add_edge(i, j, SubsumptionConfidence::Dynamic);
                    }
                }
            }
        }
        result.is_reduced = false;
        result
    }

    /// Depth of the deepest subsumption chain starting from any root.
    pub fn max_depth(&self) -> usize {
        let mut max_d = 0;
        for root_id in self.roots() {
            let depth = self.depth_from(&root_id);
            if depth > max_d {
                max_d = depth;
            }
        }
        max_d
    }

    fn depth_from(&self, id: &MutantId) -> usize {
        if let Some(&idx) = self.node_map.get(id) {
            let children: Vec<MutantId> = self
                .graph
                .neighbors_directed(idx, Direction::Outgoing)
                .map(|n| self.graph[n].clone())
                .collect();
            if children.is_empty() {
                0
            } else {
                1 + children
                    .iter()
                    .map(|c| self.depth_from(c))
                    .max()
                    .unwrap_or(0)
            }
        } else {
            0
        }
    }
}

impl Default for SubsumptionGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Equivalence classes
// ────────────────────────────────────────────────────────────────────────────

/// A group of mutants with identical kill sets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquivalenceClass {
    /// Representative mutant (first in sorted order).
    pub representative: MutantId,
    /// All members of the class (including the representative).
    pub members: BTreeSet<MutantId>,
    /// The common kill set (test indices).
    pub kill_set: BTreeSet<usize>,
}

impl EquivalenceClass {
    pub fn size(&self) -> usize {
        self.members.len()
    }

    pub fn is_singleton(&self) -> bool {
        self.members.len() == 1
    }

    /// Non-representative members.
    pub fn redundant_members(&self) -> BTreeSet<MutantId> {
        self.members
            .iter()
            .filter(|m| *m != &self.representative)
            .cloned()
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Statistics
// ────────────────────────────────────────────────────────────────────────────

/// Statistics about subsumption analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsumptionStats {
    /// Total mutants analyzed.
    pub total_mutants: usize,
    /// Number of subsumption edges found.
    pub total_edges: usize,
    /// Edges after transitive reduction.
    pub reduced_edges: usize,
    /// Number of equivalence classes found.
    pub equivalence_classes: usize,
    /// Size of the largest equivalence class.
    pub max_class_size: usize,
    /// Number of root mutants (not subsumed by anything).
    pub root_count: usize,
    /// Number of leaf mutants (do not subsume anything).
    pub leaf_count: usize,
    /// Maximum subsumption chain depth.
    pub max_depth: usize,
    /// Number of dynamic detections.
    pub dynamic_detections: usize,
    /// Number of static confirmations.
    pub static_confirmations: usize,
    /// Number of static refutations.
    pub static_refutations: usize,
    /// Number of SMT unknowns.
    pub smt_unknowns: usize,
    /// Per-operator subsumption counts.
    pub per_operator: BTreeMap<String, OperatorSubsumptionStats>,
}

/// Per-operator subsumption statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperatorSubsumptionStats {
    pub mutant_count: usize,
    pub subsumption_edges: usize,
    pub equivalence_classes: usize,
    pub root_count: usize,
}

impl SubsumptionStats {
    /// Reduction ratio: how much the transitive reduction removed.
    pub fn reduction_ratio(&self) -> f64 {
        if self.total_edges == 0 {
            0.0
        } else {
            1.0 - (self.reduced_edges as f64 / self.total_edges as f64)
        }
    }

    /// Average equivalence class size.
    pub fn avg_class_size(&self) -> f64 {
        if self.equivalence_classes == 0 {
            0.0
        } else {
            self.total_mutants as f64 / self.equivalence_classes as f64
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Analyzer
// ────────────────────────────────────────────────────────────────────────────

/// Performs mutant subsumption analysis.
pub struct SubsumptionAnalyzer {
    config: SubsumptionConfig,
    /// Optional SMT solver for static analysis.
    solver: Option<Box<dyn SmtSolver>>,
    /// Mutant error formulas for static analysis: mutant_id → formula
    /// representing the error condition E(m).
    error_formulas: HashMap<MutantId, Formula>,
    /// Descriptors for known mutants.
    descriptors: HashMap<MutantId, MutantDescriptor>,
}

impl SubsumptionAnalyzer {
    /// Create a new analyzer with default configuration.
    pub fn new() -> Self {
        Self {
            config: SubsumptionConfig::default(),
            solver: None,
            error_formulas: HashMap::new(),
            descriptors: HashMap::new(),
        }
    }

    /// Create a new analyzer with the given configuration.
    pub fn with_config(config: SubsumptionConfig) -> Self {
        Self {
            config,
            solver: None,
            error_formulas: HashMap::new(),
            descriptors: HashMap::new(),
        }
    }

    /// Set the SMT solver.
    pub fn set_solver(&mut self, solver: Box<dyn SmtSolver>) {
        self.solver = Some(solver);
    }

    /// Register an error formula for a mutant.
    pub fn register_error_formula(&mut self, id: MutantId, formula: Formula) {
        self.error_formulas.insert(id, formula);
    }

    /// Register a mutant descriptor.
    pub fn register_descriptor(&mut self, descriptor: MutantDescriptor) {
        self.descriptors.insert(descriptor.id.clone(), descriptor);
    }

    /// Register multiple descriptors.
    pub fn register_descriptors(&mut self, descriptors: Vec<MutantDescriptor>) {
        for d in descriptors {
            self.register_descriptor(d);
        }
    }

    // ── Dynamic detection ───────────────────────────────────────────────

    /// Detect subsumption dynamically from a kill matrix.
    ///
    /// m₁ dynamically subsumes m₂ if kill_set(m₂) ⊆ kill_set(m₁), i.e.,
    /// every test that kills m₂ also kills m₁.
    pub fn detect_dynamic(&self, kill_matrix: &KillMatrix) -> Result<SubsumptionGraph> {
        if kill_matrix.num_mutants() == 0 {
            return Ok(SubsumptionGraph::new());
        }

        let kill_sets = kill_matrix.kill_sets();
        let n = kill_matrix.num_mutants();
        let mut graph = SubsumptionGraph::with_mutants(&kill_matrix.mutants);

        for i in 0..n {
            if kill_sets[i].is_empty() {
                continue;
            }
            for j in 0..n {
                if i == j || kill_sets[j].is_empty() {
                    continue;
                }
                // m_i subsumes m_j if kill_set(m_j) ⊆ kill_set(m_i)
                if kill_sets[j].is_subset(&kill_sets[i]) && kill_sets[i] != kill_sets[j] {
                    graph.add_edge(
                        kill_matrix.mutants[i].clone(),
                        kill_matrix.mutants[j].clone(),
                        SubsumptionConfidence::Dynamic,
                    );
                }
            }
        }

        Ok(graph)
    }

    /// Optimized dynamic detection that groups mutants by kill set first.
    pub fn detect_dynamic_optimized(&self, kill_matrix: &KillMatrix) -> Result<SubsumptionGraph> {
        if kill_matrix.num_mutants() == 0 {
            return Ok(SubsumptionGraph::new());
        }

        let kill_sets = kill_matrix.kill_sets();
        let n = kill_matrix.num_mutants();
        let mut graph = SubsumptionGraph::with_mutants(&kill_matrix.mutants);

        // Group mutants by kill set size for efficient subset checking.
        let mut by_size: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, ks) in kill_sets.iter().enumerate() {
            by_size.entry(ks.len()).or_default().push(i);
        }

        let sizes: Vec<usize> = by_size.keys().copied().collect();

        // For each mutant, check only mutants with strictly larger kill sets.
        for (si, &size_i) in sizes.iter().enumerate() {
            if size_i == 0 {
                continue;
            }
            for &size_j in &sizes[si + 1..] {
                // Mutants with kill set size_j (> size_i) might subsume
                // mutants with kill set size_i.
                for &j in &by_size[&size_j] {
                    for &i in &by_size[&size_i] {
                        if kill_sets[i].is_subset(&kill_sets[j]) {
                            graph.add_edge(
                                kill_matrix.mutants[j].clone(),
                                kill_matrix.mutants[i].clone(),
                                SubsumptionConfidence::Dynamic,
                            );
                        }
                    }
                }
            }
        }

        Ok(graph)
    }

    // ── Static detection ────────────────────────────────────────────────

    /// Detect subsumption statically via SMT implication checking.
    ///
    /// m₁ subsumes m₂ iff E(m₂) |= E(m₁), which holds when
    /// E(m₂) ∧ ¬E(m₁) is unsatisfiable.
    pub fn detect_static(&self, mutant_ids: &[MutantId]) -> Result<SubsumptionGraph> {
        let solver = self
            .solver
            .as_ref()
            .ok_or_else(|| CoverageError::SolverError("no solver configured".to_string()))?;

        let mut graph = SubsumptionGraph::with_mutants(mutant_ids);
        let mut query_count = 0;

        for i in 0..mutant_ids.len() {
            let id_i = &mutant_ids[i];
            let formula_i = match self.error_formulas.get(id_i) {
                Some(f) => f,
                None => continue,
            };

            for j in 0..mutant_ids.len() {
                if i == j || query_count >= self.config.max_smt_queries {
                    continue;
                }
                let id_j = &mutant_ids[j];
                let formula_j = match self.error_formulas.get(id_j) {
                    Some(f) => f,
                    None => continue,
                };

                // Check E(m_j) |= E(m_i), i.e., m_i subsumes m_j.
                // This holds iff E(m_j) ∧ ¬E(m_i) is UNSAT.
                let result = solver.check_implies(formula_j, formula_i);
                query_count += 1;

                if result.is_unsat() {
                    graph.add_edge(id_i.clone(), id_j.clone(), SubsumptionConfidence::Static);
                }
            }
        }

        Ok(graph)
    }

    /// Check a single subsumption relation statically.
    pub fn check_static_subsumption(
        &self,
        subsumer: &MutantId,
        subsumed: &MutantId,
    ) -> Result<Option<bool>> {
        let solver = match &self.solver {
            Some(s) => s,
            None => return Ok(None),
        };

        let f_subsumer = match self.error_formulas.get(subsumer) {
            Some(f) => f,
            None => return Ok(None),
        };
        let f_subsumed = match self.error_formulas.get(subsumed) {
            Some(f) => f,
            None => return Ok(None),
        };

        // subsumer subsumes subsumed iff E(subsumed) |= E(subsumer)
        let result = solver.check_implies(f_subsumed, f_subsumer);
        match result {
            SolverResult::Unsat => Ok(Some(true)),
            SolverResult::Sat(_) => Ok(Some(false)),
            SolverResult::Unknown(_) => Ok(None),
        }
    }

    // ── Hybrid detection ────────────────────────────────────────────────

    /// Hybrid subsumption detection: dynamic first, confirm with static.
    pub fn detect_hybrid(
        &self,
        kill_matrix: &KillMatrix,
    ) -> Result<(SubsumptionGraph, SubsumptionStats)> {
        let mut stats = SubsumptionStats {
            total_mutants: kill_matrix.num_mutants(),
            total_edges: 0,
            reduced_edges: 0,
            equivalence_classes: 0,
            max_class_size: 0,
            root_count: 0,
            leaf_count: 0,
            max_depth: 0,
            dynamic_detections: 0,
            static_confirmations: 0,
            static_refutations: 0,
            smt_unknowns: 0,
            per_operator: BTreeMap::new(),
        };

        // Step 1: Dynamic detection.
        let dynamic_graph = if self.config.use_dynamic {
            self.detect_dynamic_optimized(kill_matrix)?
        } else {
            SubsumptionGraph::with_mutants(&kill_matrix.mutants)
        };
        stats.dynamic_detections = dynamic_graph.edge_count();

        // Step 2: If static is enabled, confirm dynamic edges.
        let mut final_graph = if self.config.use_static && self.solver.is_some() {
            let mut confirmed = SubsumptionGraph::with_mutants(&kill_matrix.mutants);
            let mut query_count = 0;

            for edge in dynamic_graph.edges() {
                if query_count >= self.config.max_smt_queries {
                    // Keep unconfirmed edges with dynamic confidence.
                    confirmed.add_edge(
                        edge.subsumer.clone(),
                        edge.subsumed.clone(),
                        SubsumptionConfidence::Dynamic,
                    );
                    continue;
                }

                match self.check_static_subsumption(&edge.subsumer, &edge.subsumed)? {
                    Some(true) => {
                        confirmed.add_edge(
                            edge.subsumer,
                            edge.subsumed,
                            SubsumptionConfidence::Hybrid,
                        );
                        stats.static_confirmations += 1;
                    }
                    Some(false) => {
                        stats.static_refutations += 1;
                    }
                    None => {
                        // Unknown: keep with dynamic confidence.
                        confirmed.add_edge(
                            edge.subsumer,
                            edge.subsumed,
                            SubsumptionConfidence::Dynamic,
                        );
                        stats.smt_unknowns += 1;
                    }
                }
                query_count += 1;
            }

            // Also try to find new static-only subsumptions.
            if self.config.use_static {
                let static_graph = self.detect_static(&kill_matrix.mutants)?;
                for edge in static_graph.edges() {
                    if !confirmed.has_edge(&edge.subsumer, &edge.subsumed) {
                        confirmed.add_edge(
                            edge.subsumer,
                            edge.subsumed,
                            SubsumptionConfidence::Static,
                        );
                    }
                }
            }

            confirmed
        } else {
            dynamic_graph
        };

        stats.total_edges = final_graph.edge_count();

        // Step 3: Transitive reduction.
        if self.config.compute_transitive_reduction {
            final_graph.transitive_reduction();
        }
        stats.reduced_edges = final_graph.edge_count();

        // Step 4: Compute remaining statistics.
        stats.root_count = final_graph.roots().len();
        stats.leaf_count = final_graph.leaves().len();
        stats.max_depth = final_graph.max_depth();

        // Equivalence classes.
        if self.config.detect_equivalence_classes {
            let classes = self.detect_equivalence_classes(kill_matrix);
            stats.equivalence_classes = classes.len();
            stats.max_class_size = classes.iter().map(|c| c.size()).max().unwrap_or(0);
        }

        // Per-operator stats.
        self.compute_operator_stats(&final_graph, kill_matrix, &mut stats);

        Ok((final_graph, stats))
    }

    // ── Full analysis ───────────────────────────────────────────────────

    /// Run the full subsumption analysis pipeline.
    pub fn analyze(&self, kill_matrix: &KillMatrix) -> Result<SubsumptionAnalysisResult> {
        let (graph, stats) = self.detect_hybrid(kill_matrix)?;
        let equivalence_classes = if self.config.detect_equivalence_classes {
            self.detect_equivalence_classes(kill_matrix)
        } else {
            Vec::new()
        };

        Ok(SubsumptionAnalysisResult {
            graph,
            stats,
            equivalence_classes,
        })
    }

    // ── Equivalence class detection ─────────────────────────────────────

    /// Detect equivalence classes: groups of mutants with identical kill sets.
    pub fn detect_equivalence_classes(&self, kill_matrix: &KillMatrix) -> Vec<EquivalenceClass> {
        let kill_sets = kill_matrix.kill_sets();
        let mut class_map: IndexMap<Vec<usize>, Vec<usize>> = IndexMap::new();

        for (i, ks) in kill_sets.iter().enumerate() {
            let key: Vec<usize> = ks.iter().copied().collect();
            class_map.entry(key).or_default().push(i);
        }

        class_map
            .into_iter()
            .map(|(key, indices)| {
                let members: BTreeSet<MutantId> = indices
                    .iter()
                    .map(|&i| kill_matrix.mutants[i].clone())
                    .collect();
                let representative = members.iter().next().unwrap().clone();
                let kill_set: BTreeSet<usize> = key.into_iter().collect();
                EquivalenceClass {
                    representative,
                    members,
                    kill_set,
                }
            })
            .collect()
    }

    /// Detect equivalence classes and return only non-singleton classes.
    pub fn detect_nontrivial_equivalence_classes(
        &self,
        kill_matrix: &KillMatrix,
    ) -> Vec<EquivalenceClass> {
        self.detect_equivalence_classes(kill_matrix)
            .into_iter()
            .filter(|c| !c.is_singleton())
            .collect()
    }

    // ── Operator statistics ─────────────────────────────────────────────

    fn compute_operator_stats(
        &self,
        graph: &SubsumptionGraph,
        kill_matrix: &KillMatrix,
        stats: &mut SubsumptionStats,
    ) {
        // Count mutants per operator.
        let mut op_mutants: BTreeMap<String, Vec<MutantId>> = BTreeMap::new();
        for id in &kill_matrix.mutants {
            if let Some(desc) = self.descriptors.get(id) {
                op_mutants
                    .entry(desc.operator.short_name().to_string())
                    .or_default()
                    .push(id.clone());
            }
        }

        for (op_name, mutant_ids) in &op_mutants {
            let mut op_stats = OperatorSubsumptionStats {
                mutant_count: mutant_ids.len(),
                ..Default::default()
            };

            for edge in graph.edges() {
                let subsumer_op = self
                    .descriptors
                    .get(&edge.subsumer)
                    .map(|d| d.operator.short_name().to_string());
                if subsumer_op.as_deref() == Some(op_name.as_str()) {
                    op_stats.subsumption_edges += 1;
                }
            }

            for id in mutant_ids {
                if graph.in_degree(id) == 0 {
                    op_stats.root_count += 1;
                }
            }

            stats.per_operator.insert(op_name.clone(), op_stats);
        }
    }

    // ── Utilities ───────────────────────────────────────────────────────

    /// Compute the subsumption ratio: edges / (n*(n-1)).
    pub fn subsumption_density(graph: &SubsumptionGraph) -> f64 {
        let n = graph.node_count();
        if n <= 1 {
            return 0.0;
        }
        let max_edges = n * (n - 1);
        graph.edge_count() as f64 / max_edges as f64
    }

    /// Check if the graph is a DAG (no cycles).
    pub fn is_dag(graph: &SubsumptionGraph) -> bool {
        toposort(&graph.graph, None).is_ok()
    }

    /// Get a topological ordering of the mutants.
    pub fn topological_order(graph: &SubsumptionGraph) -> Option<Vec<MutantId>> {
        toposort(&graph.graph, None).ok().map(|order| {
            order
                .into_iter()
                .map(|idx| graph.graph[idx].clone())
                .collect()
        })
    }

    /// Find strongly connected components (should be singletons in a proper
    /// subsumption graph).
    pub fn strongly_connected_components(graph: &SubsumptionGraph) -> Vec<Vec<MutantId>> {
        let sccs = petgraph::algo::kosaraju_scc(&graph.graph);
        sccs.into_iter()
            .map(|scc| {
                scc.into_iter()
                    .map(|idx| graph.graph[idx].clone())
                    .collect()
            })
            .collect()
    }
}

impl Default for SubsumptionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Analysis result
// ────────────────────────────────────────────────────────────────────────────

/// Complete result of subsumption analysis.
#[derive(Debug, Clone)]
pub struct SubsumptionAnalysisResult {
    pub graph: SubsumptionGraph,
    pub stats: SubsumptionStats,
    pub equivalence_classes: Vec<EquivalenceClass>,
}

impl SubsumptionAnalysisResult {
    /// Get the representative mutant IDs (one per equivalence class).
    pub fn representatives(&self) -> Vec<MutantId> {
        self.equivalence_classes
            .iter()
            .map(|c| c.representative.clone())
            .collect()
    }

    /// Get the number of redundant mutants (total - representatives).
    pub fn redundant_count(&self) -> usize {
        let total: usize = self.equivalence_classes.iter().map(|c| c.size()).sum();
        total - self.equivalence_classes.len()
    }

    /// Check if a given mutant is a representative of its equivalence class.
    pub fn is_representative(&self, id: &MutantId) -> bool {
        self.equivalence_classes
            .iter()
            .any(|c| &c.representative == id)
    }

    /// Find the equivalence class a mutant belongs to.
    pub fn class_of(&self, id: &MutantId) -> Option<&EquivalenceClass> {
        self.equivalence_classes
            .iter()
            .find(|c| c.members.contains(id))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        make_test_kill_matrix, make_test_mutant, MutantId, MutationOperator, TestId, TestOutcome,
        TrivialSolver, TrivialSolverMode,
    };

    fn km_3x4() -> KillMatrix {
        // Tests: t0, t1, t2
        // Mutants: m0, m1, m2, m3
        //       m0  m1  m2  m3
        // t0: [  K   K   .   . ]
        // t1: [  K   K   K   . ]
        // t2: [  K   .   K   . ]
        //
        // kill(m0) = {t0,t1,t2}, kill(m1) = {t0,t1}, kill(m2) = {t1,t2}, kill(m3) = {}
        // m0 subsumes m1 (kill(m1) ⊂ kill(m0))
        // m0 subsumes m2 (kill(m2) ⊂ kill(m0))
        // m1 does NOT subsume m2 or vice versa
        make_test_kill_matrix(
            3,
            4,
            &[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)],
        )
    }

    #[test]
    fn test_dynamic_subsumption_basic() {
        let km = km_3x4();
        let analyzer = SubsumptionAnalyzer::new();
        let graph = analyzer.detect_dynamic(&km).unwrap();

        // m0 should subsume m1 and m2
        assert!(graph.has_edge(&MutantId::new("m0"), &MutantId::new("m1")));
        assert!(graph.has_edge(&MutantId::new("m0"), &MutantId::new("m2")));
        // m1 and m2 should not subsume each other
        assert!(!graph.has_edge(&MutantId::new("m1"), &MutantId::new("m2")));
        assert!(!graph.has_edge(&MutantId::new("m2"), &MutantId::new("m1")));
    }

    #[test]
    fn test_dynamic_optimized_matches_naive() {
        let km = km_3x4();
        let analyzer = SubsumptionAnalyzer::new();
        let naive = analyzer.detect_dynamic(&km).unwrap();
        let optimized = analyzer.detect_dynamic_optimized(&km).unwrap();

        // Both should produce the same edges.
        let mut naive_edges: Vec<_> = naive.edges();
        naive_edges.sort_by(|a, b| (&a.subsumer, &a.subsumed).cmp(&(&b.subsumer, &b.subsumed)));
        let mut opt_edges: Vec<_> = optimized.edges();
        opt_edges.sort_by(|a, b| (&a.subsumer, &a.subsumed).cmp(&(&b.subsumer, &b.subsumed)));

        assert_eq!(naive_edges.len(), opt_edges.len());
        for (n, o) in naive_edges.iter().zip(opt_edges.iter()) {
            assert_eq!(n.subsumer, o.subsumer);
            assert_eq!(n.subsumed, o.subsumed);
        }
    }

    #[test]
    fn test_equivalence_classes() {
        // m0 and m1 have identical kill sets
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]);
        let analyzer = SubsumptionAnalyzer::new();
        let classes = analyzer.detect_equivalence_classes(&km);

        // Should have 2 classes: {m0, m1} and {m2}
        assert_eq!(classes.len(), 2);

        let nontrivial = analyzer.detect_nontrivial_equivalence_classes(&km);
        assert_eq!(nontrivial.len(), 1);
        assert_eq!(nontrivial[0].size(), 2);
    }

    #[test]
    fn test_transitive_reduction() {
        // A → B → C, A → C (A→C is redundant)
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("B"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );

        assert_eq!(graph.edge_count(), 3);
        graph.transitive_reduction();
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.has_edge(&MutantId::new("A"), &MutantId::new("B")));
        assert!(graph.has_edge(&MutantId::new("B"), &MutantId::new("C")));
        assert!(!graph.has_edge(&MutantId::new("A"), &MutantId::new("C")));
    }

    #[test]
    fn test_transitive_reduction_diamond() {
        // A → B, A → C, B → D, C → D, A → D
        // A→D is redundant (via A→B→D or A→C→D)
        let mut graph = SubsumptionGraph::new();
        let edges = vec![("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("A", "D")];
        for (from, to) in edges {
            graph.add_edge(
                MutantId::new(from),
                MutantId::new(to),
                SubsumptionConfidence::Dynamic,
            );
        }
        assert_eq!(graph.edge_count(), 5);
        graph.transitive_reduction();
        assert_eq!(graph.edge_count(), 4);
        assert!(!graph.has_edge(&MutantId::new("A"), &MutantId::new("D")));
    }

    #[test]
    fn test_transitive_closure() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("B"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );

        let closure = graph.transitive_closure();
        assert_eq!(closure.edge_count(), 3);
        assert!(closure.has_edge(&MutantId::new("A"), &MutantId::new("C")));
    }

    #[test]
    fn test_graph_roots_and_leaves() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );

        let roots = graph.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], MutantId::new("A"));

        let leaves = graph.leaves();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn test_graph_max_depth() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("B"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("C"),
            MutantId::new("D"),
            SubsumptionConfidence::Dynamic,
        );

        assert_eq!(graph.max_depth(), 3);
    }

    #[test]
    fn test_subsumption_density() {
        let mut graph = SubsumptionGraph::new();
        graph.ensure_node(MutantId::new("A"));
        graph.ensure_node(MutantId::new("B"));
        graph.ensure_node(MutantId::new("C"));

        // 3 nodes, max 6 directed edges.
        assert_eq!(SubsumptionAnalyzer::subsumption_density(&graph), 0.0);

        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        let density = SubsumptionAnalyzer::subsumption_density(&graph);
        assert!((density - 1.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_dag() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        assert!(SubsumptionAnalyzer::is_dag(&graph));

        // Add a cycle.
        graph.add_edge(
            MutantId::new("B"),
            MutantId::new("A"),
            SubsumptionConfidence::Dynamic,
        );
        assert!(!SubsumptionAnalyzer::is_dag(&graph));
    }

    #[test]
    fn test_topological_order() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("B"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );

        let order = SubsumptionAnalyzer::topological_order(&graph).unwrap();
        let pos_a = order
            .iter()
            .position(|id| id == &MutantId::new("A"))
            .unwrap();
        let pos_b = order
            .iter()
            .position(|id| id == &MutantId::new("B"))
            .unwrap();
        let pos_c = order
            .iter()
            .position(|id| id == &MutantId::new("C"))
            .unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_sccs() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("B"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );
        let sccs = SubsumptionAnalyzer::strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 3); // All singletons in a DAG.
    }

    #[test]
    fn test_hybrid_detection_dynamic_only() {
        let km = km_3x4();
        let analyzer = SubsumptionAnalyzer::new();
        let (graph, stats) = analyzer.detect_hybrid(&km).unwrap();

        assert!(stats.dynamic_detections > 0);
        assert_eq!(stats.static_confirmations, 0);
        assert_eq!(stats.total_mutants, 4);
    }

    #[test]
    fn test_hybrid_detection_with_solver() {
        let km = km_3x4();
        let config = SubsumptionConfig {
            use_dynamic: true,
            use_static: true,
            ..Default::default()
        };
        let mut analyzer = SubsumptionAnalyzer::with_config(config);
        analyzer.set_solver(Box::new(TrivialSolver {
            default_result: TrivialSolverMode::AlwaysUnsat,
        }));

        // Register dummy error formulas.
        for id in &km.mutants {
            analyzer.register_error_formula(id.clone(), Formula::True);
        }

        let (graph, stats) = analyzer.detect_hybrid(&km).unwrap();
        assert!(stats.static_confirmations > 0 || stats.dynamic_detections > 0);
    }

    #[test]
    fn test_full_analysis() {
        let km = km_3x4();
        let analyzer = SubsumptionAnalyzer::new();
        let result = analyzer.analyze(&km).unwrap();

        assert!(result.stats.total_mutants == 4);
        assert!(!result.equivalence_classes.is_empty());
    }

    #[test]
    fn test_equivalence_class_representative() {
        let km = make_test_kill_matrix(2, 3, &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        let analyzer = SubsumptionAnalyzer::new();
        let result = analyzer.analyze(&km).unwrap();

        // m0 and m1 are equivalent, m2 survives all
        assert!(result.is_representative(&MutantId::new("m0")));
        let class = result.class_of(&MutantId::new("m1")).unwrap();
        assert!(class.members.contains(&MutantId::new("m0")));
        assert!(class.members.contains(&MutantId::new("m1")));
    }

    #[test]
    fn test_empty_kill_matrix() {
        let km = make_test_kill_matrix(0, 0, &[]);
        let analyzer = SubsumptionAnalyzer::new();
        let graph = analyzer.detect_dynamic(&km).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_single_mutant() {
        let km = make_test_kill_matrix(2, 1, &[(0, 0), (1, 0)]);
        let analyzer = SubsumptionAnalyzer::new();
        let graph = analyzer.detect_dynamic(&km).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_no_kills() {
        let km = make_test_kill_matrix(3, 3, &[]);
        let analyzer = SubsumptionAnalyzer::new();
        let graph = analyzer.detect_dynamic(&km).unwrap();
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_all_equivalent() {
        // All mutants killed by exactly the same tests.
        let km = make_test_kill_matrix(2, 3, &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
        let analyzer = SubsumptionAnalyzer::new();
        let classes = analyzer.detect_nontrivial_equivalence_classes(&km);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].size(), 3);
    }

    #[test]
    fn test_redundant_count() {
        let km = make_test_kill_matrix(2, 3, &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        let analyzer = SubsumptionAnalyzer::new();
        let result = analyzer.analyze(&km).unwrap();
        // m0 and m1 are equivalent (same kill set), m2 is separate
        assert!(result.redundant_count() >= 1);
    }

    #[test]
    fn test_graph_subsumed_by_and_subsumers_of() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );

        let subsumed = graph.subsumed_by(&MutantId::new("A"));
        assert_eq!(subsumed.len(), 2);
        assert!(subsumed.contains(&MutantId::new("B")));
        assert!(subsumed.contains(&MutantId::new("C")));

        let subsumers = graph.subsumers_of(&MutantId::new("B"));
        assert_eq!(subsumers.len(), 1);
        assert!(subsumers.contains(&MutantId::new("A")));
    }

    #[test]
    fn test_in_degree_out_degree() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("C"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );

        assert_eq!(graph.out_degree(&MutantId::new("A")), 1);
        assert_eq!(graph.in_degree(&MutantId::new("B")), 2);
        assert_eq!(graph.in_degree(&MutantId::new("A")), 0);
    }

    #[test]
    fn test_chain_subsumption() {
        // Linear chain: t0 kills m0,m1,m2; t1 kills m0,m1; t2 kills m0
        // kill(m0) = {t0,t1,t2}, kill(m1) = {t0,t1}, kill(m2) = {t0}
        // m0 subsumes m1 subsumes m2
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]);
        let analyzer = SubsumptionAnalyzer::new();
        let graph = analyzer.detect_dynamic(&km).unwrap();

        assert!(graph.has_edge(&MutantId::new("m0"), &MutantId::new("m1")));
        assert!(graph.has_edge(&MutantId::new("m1"), &MutantId::new("m2")));
        // Also has transitive edge m0 → m2 before reduction.
        assert!(graph.has_edge(&MutantId::new("m0"), &MutantId::new("m2")));
    }

    #[test]
    fn test_chain_after_reduction() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]);
        let config = SubsumptionConfig {
            compute_transitive_reduction: true,
            ..Default::default()
        };
        let analyzer = SubsumptionAnalyzer::with_config(config);
        let (graph, _) = analyzer.detect_hybrid(&km).unwrap();

        assert!(graph.has_edge(&MutantId::new("m0"), &MutantId::new("m1")));
        assert!(graph.has_edge(&MutantId::new("m1"), &MutantId::new("m2")));
        // Transitive edge should be removed.
        assert!(!graph.has_edge(&MutantId::new("m0"), &MutantId::new("m2")));
    }

    #[test]
    fn test_stats_reduction_ratio() {
        let stats = SubsumptionStats {
            total_mutants: 10,
            total_edges: 20,
            reduced_edges: 10,
            equivalence_classes: 5,
            max_class_size: 3,
            root_count: 3,
            leaf_count: 4,
            max_depth: 2,
            dynamic_detections: 20,
            static_confirmations: 0,
            static_refutations: 0,
            smt_unknowns: 0,
            per_operator: BTreeMap::new(),
        };
        assert!((stats.reduction_ratio() - 0.5).abs() < 1e-9);
        assert!((stats.avg_class_size() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_graph_with_mutants() {
        let ids = vec![
            MutantId::new("m0"),
            MutantId::new("m1"),
            MutantId::new("m2"),
        ];
        let graph = SubsumptionGraph::with_mutants(&ids);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_idempotent_reduction() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("B"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("C"),
            SubsumptionConfidence::Dynamic,
        );

        graph.transitive_reduction();
        assert_eq!(graph.edge_count(), 2);
        graph.transitive_reduction(); // Should be idempotent.
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_duplicate_edge_ignored() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Dynamic,
        );
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("B"),
            SubsumptionConfidence::Static,
        );
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_self_edge_ignored() {
        let mut graph = SubsumptionGraph::new();
        graph.add_edge(
            MutantId::new("A"),
            MutantId::new("A"),
            SubsumptionConfidence::Dynamic,
        );
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_operator_stats_computed() {
        let km = km_3x4();
        let mut analyzer = SubsumptionAnalyzer::new();
        for (i, id) in km.mutants.iter().enumerate() {
            let op = if i % 2 == 0 {
                MutationOperator::AOR
            } else {
                MutationOperator::ROR
            };
            analyzer.register_descriptor(crate::make_test_mutant(id.as_str(), op));
        }
        let (_, stats) = analyzer.detect_hybrid(&km).unwrap();
        // Should have per-operator stats for AOR and ROR.
        assert!(stats.per_operator.contains_key("AOR"));
        assert!(stats.per_operator.contains_key("ROR"));
    }
}
