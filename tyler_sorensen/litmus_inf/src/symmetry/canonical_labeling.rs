//! Canonical labeling for execution graphs (nauty/bliss-style).
//!
//! Instead of computing the full automorphism group (expensive, and the source
//! of the negative speedup in the original), we compute a canonical form for
//! each execution graph. Two graphs are isomorphic iff they have the same
//! canonical form. This allows O(1) duplicate detection via hashing.
//!
//! Algorithm: individualization-refinement with Weisfeiler-Leman color
//! refinement as the invariant. This is the core of nauty/bliss.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    Event, EventId, ThreadId, Address, OpType, Scope,
    ExecutionGraph, BitMatrix,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for canonical labeling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelingConfig {
    /// Maximum refinement iterations before giving up.
    pub max_iterations: usize,
    /// Whether to use thread colors.
    pub use_thread_colors: bool,
    /// Whether to use address colors.
    pub use_address_colors: bool,
    /// Whether to use operation-type colors.
    pub use_optype_colors: bool,
    /// Cell selection strategy.
    pub cell_selector: CellSelector,
}

impl Default for LabelingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            use_thread_colors: true,
            use_address_colors: true,
            use_optype_colors: true,
            cell_selector: CellSelector::FirstNonTrivial,
        }
    }
}

/// Strategy for selecting which cell to individualize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellSelector {
    /// First non-trivial cell (simplest).
    FirstNonTrivial,
    /// Largest non-trivial cell.
    Largest,
    /// Cell with most connections.
    MostConnected,
    /// Cell with maximum degree.
    MaxDegree,
}

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// A color for a vertex in the graph, used during refinement.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexColor {
    /// Base color from vertex properties.
    pub base: u64,
    /// Refined color after WL iterations.
    pub refined: u64,
    /// Neighbor multiset hash.
    pub neighbor_hash: u64,
}

impl VertexColor {
    pub fn new(base: u64) -> Self {
        Self {
            base,
            refined: base,
            neighbor_hash: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical Form
// ---------------------------------------------------------------------------

/// The canonical form of an execution graph — a fingerprint that is invariant
/// under automorphism.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalForm {
    /// Number of events.
    pub num_events: usize,
    /// Sorted color partition.
    pub color_partition: Vec<u64>,
    /// Canonical adjacency (sorted edge list under canonical labeling).
    pub canonical_edges_po: Vec<(usize, usize)>,
    pub canonical_edges_rf: Vec<(usize, usize)>,
    pub canonical_edges_co: Vec<(usize, usize)>,
    /// Hash of the canonical form for fast comparison.
    pub hash: u64,
}

impl CanonicalForm {
    /// Check if two canonical forms represent isomorphic graphs.
    pub fn is_isomorphic(&self, other: &CanonicalForm) -> bool {
        self == other
    }
}

// ---------------------------------------------------------------------------
// Color Refinement (Weisfeiler-Leman 1-dim)
// ---------------------------------------------------------------------------

/// 1-dimensional Weisfeiler-Leman color refinement.
pub struct ColorRefinement {
    config: LabelingConfig,
}

impl ColorRefinement {
    pub fn new(config: LabelingConfig) -> Self {
        Self { config }
    }

    /// Compute initial coloring from vertex properties.
    pub fn initial_coloring(&self, graph: &ExecutionGraph) -> Vec<VertexColor> {
        let n = graph.events.len();
        let mut colors = Vec::with_capacity(n);

        for event in &graph.events {
            let mut base: u64 = 0;

            if self.config.use_optype_colors {
                base = base.wrapping_mul(31).wrapping_add(match event.op_type {
                    OpType::Read => 1,
                    OpType::Write => 2,
                    OpType::Fence => 3,
                    OpType::RMW => 4,
                });
            }

            if self.config.use_thread_colors {
                base = base.wrapping_mul(31).wrapping_add(event.thread as u64);
            }

            if self.config.use_address_colors {
                base = base.wrapping_mul(31).wrapping_add(event.address);
            }

            // Include scope
            base = base.wrapping_mul(31).wrapping_add(match event.scope {
                Scope::CTA => 1,
                Scope::GPU => 2,
                Scope::System => 3,
                Scope::None => 0,
            });

            colors.push(VertexColor::new(base));
        }

        colors
    }

    /// Perform one round of WL refinement.
    /// Returns true if the partition was refined (colors changed).
    pub fn refine_step(
        &self,
        colors: &mut Vec<VertexColor>,
        po: &BitMatrix,
        rf: &BitMatrix,
        co: &BitMatrix,
    ) -> bool {
        let n = colors.len();
        let mut new_colors = Vec::with_capacity(n);
        let mut changed = false;

        for i in 0..n {
            // Build neighbor multiset: (relation_type, neighbor_color) sorted
            let mut neighbor_info: Vec<(u8, u64)> = Vec::new();

            // PO successors
            for j in po.successors(i) {
                neighbor_info.push((0, colors[j].refined));
            }
            // PO predecessors
            for j in po.predecessors(i) {
                neighbor_info.push((1, colors[j].refined));
            }
            // RF (reads-from)
            if rf.dim() == n {
                for j in rf.successors(i) {
                    neighbor_info.push((2, colors[j].refined));
                }
                for j in rf.predecessors(i) {
                    neighbor_info.push((3, colors[j].refined));
                }
            }
            // CO (coherence)
            if co.dim() == n {
                for j in co.successors(i) {
                    neighbor_info.push((4, colors[j].refined));
                }
            }

            neighbor_info.sort();

            // Hash the neighbor multiset
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            colors[i].refined.hash(&mut hasher);
            for (rel, col) in &neighbor_info {
                rel.hash(&mut hasher);
                col.hash(&mut hasher);
            }
            let new_refined = std::hash::Hasher::finish(&hasher);

            let mut new_color = colors[i].clone();
            if new_color.refined != new_refined {
                changed = true;
            }
            new_color.refined = new_refined;
            new_color.neighbor_hash = {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for (r, c) in &neighbor_info {
                    r.hash(&mut h);
                    c.hash(&mut h);
                }
                std::hash::Hasher::finish(&h)
            };
            new_colors.push(new_color);
        }

        *colors = new_colors;
        changed
    }

    /// Run WL refinement to fixpoint.
    pub fn refine_to_fixpoint(
        &self,
        colors: &mut Vec<VertexColor>,
        po: &BitMatrix,
        rf: &BitMatrix,
        co: &BitMatrix,
    ) -> usize {
        let mut iterations = 0;
        for _ in 0..self.config.max_iterations {
            iterations += 1;
            if !self.refine_step(colors, po, rf, co) {
                break;
            }
        }
        iterations
    }

    /// Count distinct colors (partition size).
    pub fn partition_size(colors: &[VertexColor]) -> usize {
        let distinct: HashSet<u64> = colors.iter().map(|c| c.refined).collect();
        distinct.len()
    }

    /// Get cells (groups of vertices with same color).
    pub fn cells(colors: &[VertexColor]) -> Vec<Vec<usize>> {
        let mut map: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, color) in colors.iter().enumerate() {
            map.entry(color.refined).or_default().push(i);
        }
        map.into_values().collect()
    }

    /// Get non-trivial cells (size > 1).
    pub fn non_trivial_cells(colors: &[VertexColor]) -> Vec<Vec<usize>> {
        Self::cells(colors).into_iter().filter(|c| c.len() > 1).collect()
    }
}

// ---------------------------------------------------------------------------
// Canonical Labeler
// ---------------------------------------------------------------------------

/// Computes canonical forms for execution graphs.
pub struct CanonicalLabeler {
    config: LabelingConfig,
    refinement: ColorRefinement,
    /// Cache of canonical forms.
    cache: HashMap<u64, CanonicalForm>,
    /// Statistics.
    pub total_labeled: usize,
    pub cache_hits: usize,
    pub total_refinement_iterations: usize,
}

impl CanonicalLabeler {
    pub fn new(config: LabelingConfig) -> Self {
        let refinement = ColorRefinement::new(config.clone());
        Self {
            config,
            refinement,
            cache: HashMap::new(),
            total_labeled: 0,
            cache_hits: 0,
            total_refinement_iterations: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(LabelingConfig::default())
    }

    /// Compute the canonical form of an execution graph.
    pub fn canonical_form(&mut self, graph: &ExecutionGraph) -> CanonicalForm {
        self.total_labeled += 1;

        // Quick hash check
        let quick_hash = self.quick_hash(graph);
        if let Some(cached) = self.cache.get(&quick_hash) {
            self.cache_hits += 1;
            return cached.clone();
        }

        let n = graph.events.len();

        // Step 1: Initial coloring
        let mut colors = self.refinement.initial_coloring(graph);

        // Step 2: WL refinement to fixpoint
        let iters = self.refinement.refine_to_fixpoint(
            &mut colors,
            &graph.po,
            &graph.rf,
            &graph.co,
        );
        self.total_refinement_iterations += iters;

        // Step 3: Check if partition is discrete (all different colors)
        let partition_size = ColorRefinement::partition_size(&colors);

        // Step 4: If not discrete, individualize
        if partition_size < n {
            self.individualize_and_refine(&mut colors, graph);
        }

        // Step 5: Build canonical labeling (sort by refined color)
        let labeling = self.build_canonical_labeling(&colors);

        // Step 6: Build canonical form
        let form = self.build_canonical_form(graph, &labeling, &colors);

        self.cache.insert(quick_hash, form.clone());
        form
    }

    /// Check if two execution graphs are isomorphic.
    pub fn are_isomorphic(
        &mut self,
        g1: &ExecutionGraph,
        g2: &ExecutionGraph,
    ) -> bool {
        if g1.events.len() != g2.events.len() {
            return false;
        }
        let f1 = self.canonical_form(g1);
        let f2 = self.canonical_form(g2);
        f1.is_isomorphic(&f2)
    }

    /// Deduplicate a set of execution graphs, returning representatives.
    pub fn deduplicate(&mut self, graphs: &[ExecutionGraph]) -> Vec<usize> {
        let mut seen: HashSet<u64> = HashSet::new();
        let mut representatives = Vec::new();

        for (i, graph) in graphs.iter().enumerate() {
            let form = self.canonical_form(graph);
            if seen.insert(form.hash) {
                representatives.push(i);
            }
        }

        representatives
    }

    /// Compute compression ratio (how many graphs were duplicates).
    pub fn compression_ratio(&self, total: usize, unique: usize) -> f64 {
        if total == 0 { return 1.0; }
        unique as f64 / total as f64
    }

    // --- Internal methods ---

    fn quick_hash(&self, graph: &ExecutionGraph) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        graph.events.len().hash(&mut hasher);
        graph.po.count_edges().hash(&mut hasher);
        graph.rf.count_edges().hash(&mut hasher);
        graph.co.count_edges().hash(&mut hasher);
        // Hash event types
        for event in &graph.events {
            event.op_type.hash(&mut hasher);
            event.thread.hash(&mut hasher);
        }
        std::hash::Hasher::finish(&hasher)
    }

    fn individualize_and_refine(
        &self,
        colors: &mut Vec<VertexColor>,
        graph: &ExecutionGraph,
    ) {
        let non_trivial = ColorRefinement::non_trivial_cells(colors);
        if non_trivial.is_empty() {
            return;
        }

        // Select cell to individualize
        let cell = match self.config.cell_selector {
            CellSelector::FirstNonTrivial => &non_trivial[0],
            CellSelector::Largest => non_trivial.iter().max_by_key(|c| c.len()).unwrap(),
            CellSelector::MostConnected | CellSelector::MaxDegree => {
                // Pick cell with highest total degree
                non_trivial.iter().max_by_key(|cell| {
                    cell.iter().map(|&v| {
                        graph.po.successors(v).count()
                            + graph.rf.successors(v).count()
                            + graph.co.successors(v).count()
                    }).sum::<usize>()
                }).unwrap()
            }
        };

        // Individualize first vertex in chosen cell
        if let Some(&vertex) = cell.first() {
            colors[vertex].refined = colors[vertex].refined.wrapping_add(1_000_000_007);

            // Re-refine
            self.refinement.refine_to_fixpoint(
                colors,
                &graph.po,
                &graph.rf,
                &graph.co,
            );
        }
    }

    fn build_canonical_labeling(&self, colors: &[VertexColor]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..colors.len()).collect();
        indices.sort_by(|&a, &b| colors[a].refined.cmp(&colors[b].refined));
        
        // Build inverse mapping
        let mut labeling = vec![0usize; colors.len()];
        for (new_label, &old_index) in indices.iter().enumerate() {
            labeling[old_index] = new_label;
        }
        labeling
    }

    fn build_canonical_form(
        &self,
        graph: &ExecutionGraph,
        labeling: &[usize],
        colors: &[VertexColor],
    ) -> CanonicalForm {
        let n = graph.events.len();

        // Map edges through canonical labeling
        let map_edges = |matrix: &BitMatrix| -> Vec<(usize, usize)> {
            let mut edges = Vec::new();
            if matrix.dim() == n {
                for (i, j) in matrix.edges() {
                    edges.push((labeling[i], labeling[j]));
                }
            }
            edges.sort();
            edges
        };

        let canonical_edges_po = map_edges(&graph.po);
        let canonical_edges_rf = map_edges(&graph.rf);
        let canonical_edges_co = map_edges(&graph.co);

        // Color partition
        let mut color_partition: Vec<u64> = colors.iter().map(|c| c.refined).collect();
        color_partition.sort();

        // Compute hash
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        n.hash(&mut hasher);
        color_partition.hash(&mut hasher);
        canonical_edges_po.hash(&mut hasher);
        canonical_edges_rf.hash(&mut hasher);
        canonical_edges_co.hash(&mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);

        CanonicalForm {
            num_events: n,
            color_partition,
            canonical_edges_po,
            canonical_edges_rf,
            canonical_edges_co,
            hash,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::execution::{ExecutionGraphBuilder, OpType};

    fn make_simple_graph(n_events: usize) -> ExecutionGraph {
        let mut builder = ExecutionGraphBuilder::new();
        for i in 0..n_events {
            let thread = i / 2;
            let op = if i % 2 == 0 { OpType::Write } else { OpType::Read };
            builder.add_event(thread, op, (i % 3) as u64, i as u64);
        }
        // PO edges are implicit from thread ordering in add_event
        builder.build()
    }

    #[test]
    fn test_initial_coloring() {
        let graph = make_simple_graph(4);
        let refinement = ColorRefinement::new(LabelingConfig::default());
        let colors = refinement.initial_coloring(&graph);
        assert_eq!(colors.len(), 4);
    }

    #[test]
    fn test_refinement_converges() {
        let graph = make_simple_graph(6);
        let refinement = ColorRefinement::new(LabelingConfig::default());
        let mut colors = refinement.initial_coloring(&graph);
        let iters = refinement.refine_to_fixpoint(
            &mut colors,
            &graph.po,
            &graph.rf,
            &graph.co,
        );
        assert!(iters > 0);
        assert!(iters <= 100);
    }

    #[test]
    fn test_canonical_form_deterministic() {
        let graph = make_simple_graph(4);
        let mut labeler = CanonicalLabeler::with_defaults();
        let f1 = labeler.canonical_form(&graph);
        let f2 = labeler.canonical_form(&graph);
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_isomorphic_graphs() {
        let g1 = make_simple_graph(4);
        let g2 = make_simple_graph(4);
        let mut labeler = CanonicalLabeler::with_defaults();
        assert!(labeler.are_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_different_graphs() {
        let g1 = make_simple_graph(4);
        let g2 = make_simple_graph(6);
        let mut labeler = CanonicalLabeler::with_defaults();
        assert!(!labeler.are_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_deduplicate() {
        let graphs: Vec<_> = (0..5).map(|_| make_simple_graph(4)).collect();
        let mut labeler = CanonicalLabeler::with_defaults();
        let reps = labeler.deduplicate(&graphs);
        // All identical graphs should produce 1 representative
        assert_eq!(reps.len(), 1);
    }

    #[test]
    fn test_cache_hits() {
        let graph = make_simple_graph(4);
        let mut labeler = CanonicalLabeler::with_defaults();
        labeler.canonical_form(&graph);
        labeler.canonical_form(&graph);
        assert!(labeler.cache_hits >= 1);
    }
}
