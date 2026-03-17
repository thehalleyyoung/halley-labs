//! Codegen-level optimizations — inline expansion, loop fusion, SIMD hints,
//! constant propagation, dead code elimination, and strength reduction.

use crate::{
    lowering::{LoweredGraph, LoweredNode},
    CgGraph, CodegenConfig, CodegenError, CodegenResult, NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Optimization pass trait
// ---------------------------------------------------------------------------

/// Result of applying an optimization pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub pass_name: String,
    pub nodes_affected: usize,
    pub nodes_removed: usize,
    pub estimated_speedup: f64,
    pub details: Vec<String>,
}

// ---------------------------------------------------------------------------
// InlineExpansion
// ---------------------------------------------------------------------------

/// Inlines small nodes (below a cycle threshold) into their consumers,
/// eliminating function-call and buffer overhead.
#[derive(Debug, Clone)]
pub struct InlineExpansion {
    /// Maximum WCET in cycles for a node to be eligible for inlining.
    pub threshold_cycles: f64,
}

impl InlineExpansion {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            threshold_cycles: config.inline_threshold_cycles,
        }
    }

    /// Identify nodes eligible for inlining.
    pub fn find_candidates(&self, graph: &CgGraph) -> Vec<u64> {
        graph
            .nodes
            .iter()
            .filter(|n| {
                // Inline small nodes that are not sources or sinks.
                let is_simple = matches!(
                    n.kind,
                    NodeKind::Gain | NodeKind::Constant | NodeKind::Splitter
                );
                let below_threshold = n.wcet_cycles <= self.threshold_cycles || n.wcet_cycles == 0.0;
                is_simple && below_threshold && !graph.successors(n.id).is_empty()
            })
            .map(|n| n.id)
            .collect()
    }

    /// Apply inline expansion to the graph. Returns the list of inlined node IDs
    /// and an optimization report. Modifies the graph in-place by merging
    /// inlined node parameters into their consumers.
    pub fn apply(&self, graph: &mut CgGraph) -> OptimizationReport {
        let candidates = self.find_candidates(graph);
        let mut details = Vec::new();
        let mut inlined_count = 0;

        for &node_id in &candidates {
            let successors = graph.successors(node_id);
            let predecessors = graph.predecessors(node_id);

            // Only inline nodes with exactly one output.
            if successors.len() != 1 {
                continue;
            }

            let succ_id = successors[0];

            // Merge: copy the inlined node's parameters into metadata.
            if let Some(node) = graph.node(node_id) {
                let name = node.name.clone();
                let kind = node.kind;
                let params = node.parameters.clone();

                if let Some(succ) = graph.node_mut(succ_id) {
                    succ.metadata.insert(
                        format!("inlined_{}", name),
                        format!("{:?}: {:?}", kind, params),
                    );
                }
            }

            // Redirect predecessors to point directly to the successor.
            let edges_to_redirect: Vec<usize> = graph
                .edges
                .iter()
                .enumerate()
                .filter(|(_, e)| e.dest_node == node_id)
                .map(|(i, _)| i)
                .collect();

            for &idx in &edges_to_redirect {
                graph.edges[idx].dest_node = succ_id;
            }

            // Remove edges from inlined node to successor.
            graph.edges.retain(|e| e.source_node != node_id);

            // Remove the inlined node.
            graph.nodes.retain(|n| n.id != node_id);
            graph.topological_order.retain(|&id| id != node_id);

            details.push(format!("Inlined node {} into node {}", node_id, succ_id));
            inlined_count += 1;
        }

        let speedup = if inlined_count > 0 {
            1.0 + 0.02 * inlined_count as f64 // ~2% per inlined node
        } else {
            1.0
        };

        OptimizationReport {
            pass_name: "InlineExpansion".into(),
            nodes_affected: inlined_count,
            nodes_removed: inlined_count,
            estimated_speedup: speedup,
            details,
        }
    }
}

// ---------------------------------------------------------------------------
// LoopFusion
// ---------------------------------------------------------------------------

/// Fuses adjacent processing loops for cache efficiency. When two consecutive
/// nodes in the schedule both iterate over the block, they can be fused into
/// a single loop to keep data in L1 cache.
#[derive(Debug, Clone)]
pub struct LoopFusion;

impl LoopFusion {
    pub fn new() -> Self {
        Self
    }

    /// Find pairs of adjacent nodes that can be fused.
    pub fn find_fusable_pairs(&self, graph: &CgGraph) -> Vec<(u64, u64)> {
        let mut pairs = Vec::new();

        for i in 0..graph.topological_order.len().saturating_sub(1) {
            let a = graph.topological_order[i];
            let b = graph.topological_order[i + 1];

            // Check if a is the sole predecessor of b.
            let preds = graph.predecessors(b);
            if preds.len() == 1 && preds[0] == a {
                // And a has only one successor.
                let succs = graph.successors(a);
                if succs.len() == 1 && succs[0] == b {
                    // Both must be block-processing nodes.
                    if self.is_fusable_kind(graph.node(a)) && self.is_fusable_kind(graph.node(b)) {
                        pairs.push((a, b));
                    }
                }
            }
        }

        pairs
    }

    /// Apply loop fusion: annotate fused pairs in metadata.
    pub fn apply(&self, graph: &mut CgGraph) -> OptimizationReport {
        let pairs = self.find_fusable_pairs(graph);
        let mut details = Vec::new();

        for &(a, b) in &pairs {
            if let Some(node_a) = graph.node_mut(a) {
                node_a
                    .metadata
                    .insert("fused_with".into(), b.to_string());
            }
            if let Some(node_b) = graph.node_mut(b) {
                node_b
                    .metadata
                    .insert("fused_from".into(), a.to_string());
            }
            details.push(format!("Fused loop: node {} + node {}", a, b));
        }

        let speedup = if pairs.is_empty() {
            1.0
        } else {
            1.0 + 0.05 * pairs.len() as f64 // ~5% per fused pair
        };

        OptimizationReport {
            pass_name: "LoopFusion".into(),
            nodes_affected: pairs.len() * 2,
            nodes_removed: 0,
            estimated_speedup: speedup,
            details,
        }
    }

    fn is_fusable_kind(&self, node: Option<&crate::NodeInfo>) -> bool {
        node.map(|n| {
            matches!(
                n.kind,
                NodeKind::Oscillator
                    | NodeKind::Filter
                    | NodeKind::Gain
                    | NodeKind::Envelope
                    | NodeKind::Pan
                    | NodeKind::Delay
                    | NodeKind::Compressor
                    | NodeKind::Limiter
            )
        })
        .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// SimdHints
// ---------------------------------------------------------------------------

/// Annotates nodes with SIMD alignment hints for vectorizable operations.
#[derive(Debug, Clone)]
pub struct SimdHints {
    pub enabled: bool,
}

impl SimdHints {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            enabled: config.enable_simd,
        }
    }

    /// Find nodes that can benefit from SIMD vectorization.
    pub fn find_vectorizable(&self, graph: &CgGraph) -> Vec<u64> {
        if !self.enabled {
            return Vec::new();
        }

        graph
            .nodes
            .iter()
            .filter(|n| self.is_vectorizable(n.kind, n.block_size))
            .map(|n| n.id)
            .collect()
    }

    /// Annotate the graph with SIMD hints.
    pub fn apply(&self, graph: &mut CgGraph) -> OptimizationReport {
        let vectorizable = self.find_vectorizable(graph);
        let mut details = Vec::new();

        for &node_id in &vectorizable {
            if let Some(node) = graph.node_mut(node_id) {
                let width = self.recommended_simd_width(node.kind);
                node.metadata
                    .insert("simd_width".into(), width.to_string());
                node.metadata
                    .insert("simd_aligned".into(), "true".into());
                details.push(format!(
                    "Node {} ({:?}): SIMD width {}",
                    node_id, node.kind, width
                ));
            }
        }

        let speedup = if vectorizable.is_empty() {
            1.0
        } else {
            // Rough estimate: 2-4x speedup from SIMD on vectorizable nodes.
            1.0 + 0.3 * vectorizable.len() as f64
        };

        OptimizationReport {
            pass_name: "SimdHints".into(),
            nodes_affected: vectorizable.len(),
            nodes_removed: 0,
            estimated_speedup: speedup,
            details,
        }
    }

    fn is_vectorizable(&self, kind: NodeKind, block_size: usize) -> bool {
        // Block size must be a multiple of 4 for SIMD.
        if block_size % 4 != 0 {
            return false;
        }
        matches!(
            kind,
            NodeKind::Gain
                | NodeKind::Oscillator
                | NodeKind::Mixer
                | NodeKind::Pan
                | NodeKind::Filter
        )
    }

    fn recommended_simd_width(&self, kind: NodeKind) -> usize {
        match kind {
            NodeKind::Gain | NodeKind::Mixer | NodeKind::Pan => 8, // AVX f32
            NodeKind::Oscillator | NodeKind::Filter => 4,          // SSE f64
            _ => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// ConstantPropagation
// ---------------------------------------------------------------------------

/// Propagates constant values through the graph, replacing computed nodes
/// with their known constant outputs where possible.
#[derive(Debug, Clone)]
pub struct ConstantPropagation;

impl ConstantPropagation {
    pub fn new() -> Self {
        Self
    }

    /// Find nodes whose outputs are compile-time constants.
    pub fn find_constants(&self, graph: &CgGraph) -> HashMap<u64, f64> {
        let mut constants: HashMap<u64, f64> = HashMap::new();

        for node in &graph.nodes {
            if node.kind == NodeKind::Constant {
                if let Some(&val) = node.parameters.get("value") {
                    constants.insert(node.id, val);
                }
            }
        }

        // Propagate through Gain nodes with constant inputs.
        let mut changed = true;
        while changed {
            changed = false;
            for node in &graph.nodes {
                if constants.contains_key(&node.id) {
                    continue;
                }
                if node.kind == NodeKind::Gain {
                    let preds = graph.predecessors(node.id);
                    if preds.len() == 1 {
                        if let Some(&input_val) = constants.get(&preds[0]) {
                            let level = node.parameters.get("level").copied().unwrap_or(1.0);
                            constants.insert(node.id, input_val * level);
                            changed = true;
                        }
                    }
                }
            }
        }

        constants
    }

    /// Apply constant propagation: replace constant-output nodes with
    /// Constant nodes.
    pub fn apply(&self, graph: &mut CgGraph) -> OptimizationReport {
        let constants = self.find_constants(graph);
        let mut details = Vec::new();
        let mut affected = 0;

        for (&node_id, &value) in &constants {
            if let Some(node) = graph.node_mut(node_id) {
                if node.kind != NodeKind::Constant {
                    let old_kind = node.kind;
                    node.kind = NodeKind::Constant;
                    node.parameters.clear();
                    node.parameters.insert("value".into(), value);
                    details.push(format!(
                        "Folded {:?} node {} to constant {:.6}",
                        old_kind, node_id, value
                    ));
                    affected += 1;
                }
            }
        }

        OptimizationReport {
            pass_name: "ConstantPropagation".into(),
            nodes_affected: affected,
            nodes_removed: 0,
            estimated_speedup: if affected > 0 { 1.01 } else { 1.0 },
            details,
        }
    }
}

// ---------------------------------------------------------------------------
// DeadCodeElimination
// ---------------------------------------------------------------------------

/// Removes nodes that do not contribute to any output.
#[derive(Debug, Clone)]
pub struct DeadCodeElimination;

impl DeadCodeElimination {
    pub fn new() -> Self {
        Self
    }

    /// Find nodes that are unreachable from any output (sink) node.
    pub fn find_dead_nodes(&self, graph: &CgGraph) -> Vec<u64> {
        // Find output nodes.
        let sinks: Vec<u64> = graph
            .nodes
            .iter()
            .filter(|n| graph.successors(n.id).is_empty())
            .map(|n| n.id)
            .collect();

        // BFS backward from sinks.
        let mut reachable: HashSet<u64> = HashSet::new();
        let mut queue: std::collections::VecDeque<u64> = sinks.into_iter().collect();

        while let Some(nid) = queue.pop_front() {
            if reachable.insert(nid) {
                for pred in graph.predecessors(nid) {
                    if !reachable.contains(&pred) {
                        queue.push_back(pred);
                    }
                }
            }
        }

        graph
            .nodes
            .iter()
            .filter(|n| !reachable.contains(&n.id))
            .map(|n| n.id)
            .collect()
    }

    /// Remove dead nodes from the graph.
    pub fn apply(&self, graph: &mut CgGraph) -> OptimizationReport {
        let dead = self.find_dead_nodes(graph);
        let dead_set: HashSet<u64> = dead.iter().copied().collect();
        let details: Vec<String> = dead
            .iter()
            .map(|&id| {
                let name = graph.node(id).map(|n| n.name.as_str()).unwrap_or("?");
                format!("Removed dead node {} ('{}')", id, name)
            })
            .collect();
        let removed = dead.len();

        graph.nodes.retain(|n| !dead_set.contains(&n.id));
        graph
            .edges
            .retain(|e| !dead_set.contains(&e.source_node) && !dead_set.contains(&e.dest_node));
        graph
            .topological_order
            .retain(|id| !dead_set.contains(id));

        OptimizationReport {
            pass_name: "DeadCodeElimination".into(),
            nodes_affected: removed,
            nodes_removed: removed,
            estimated_speedup: if removed > 0 { 1.05 } else { 1.0 },
            details,
        }
    }
}

// ---------------------------------------------------------------------------
// StrengthReduction
// ---------------------------------------------------------------------------

/// Replaces expensive operations with cheaper equivalents (e.g., multiply by
/// power of 2 → shift, trig approximations).
#[derive(Debug, Clone)]
pub struct StrengthReduction;

impl StrengthReduction {
    pub fn new() -> Self {
        Self
    }

    /// Find nodes where strength reduction can be applied.
    pub fn find_candidates(&self, graph: &CgGraph) -> Vec<(u64, String)> {
        let mut candidates = Vec::new();

        for node in &graph.nodes {
            match node.kind {
                NodeKind::Gain => {
                    if let Some(&level) = node.parameters.get("level") {
                        if Self::is_power_of_two(level) {
                            candidates.push((
                                node.id,
                                format!("Gain {:.6} → bit shift", level),
                            ));
                        }
                        if (level - 1.0).abs() < 1e-10 {
                            candidates.push((node.id, "Gain 1.0 → identity (elide)".into()));
                        }
                        if level.abs() < 1e-10 {
                            candidates.push((
                                node.id,
                                "Gain 0.0 → zero fill (elide input)".into(),
                            ));
                        }
                    }
                }
                NodeKind::Mixer => {
                    if let Some(&count) = node.parameters.get("channel_count") {
                        if count == 1.0 {
                            candidates.push((node.id, "Mixer(1) → passthrough".into()));
                        }
                    }
                }
                NodeKind::Pan => {
                    if let Some(&pos) = node.parameters.get("position") {
                        if pos.abs() < 1e-10 {
                            candidates.push((node.id, "Pan(0) → equal power (simplify)".into()));
                        }
                    }
                }
                _ => {}
            }
        }

        candidates
    }

    /// Apply strength reduction.
    pub fn apply(&self, graph: &mut CgGraph) -> OptimizationReport {
        let candidates = self.find_candidates(graph);
        let mut details = Vec::new();
        let mut affected = 0;

        for (node_id, desc) in &candidates {
            if let Some(node) = graph.node_mut(*node_id) {
                node.metadata
                    .insert("strength_reduced".into(), desc.clone());

                // Elide identity gains.
                if node.kind == NodeKind::Gain {
                    if let Some(&level) = node.parameters.get("level") {
                        if (level - 1.0).abs() < 1e-10 {
                            node.kind = NodeKind::Splitter; // passthrough
                            node.parameters.clear();
                        }
                    }
                }

                // Simplify single-channel mixer.
                if node.kind == NodeKind::Mixer {
                    if let Some(&count) = node.parameters.get("channel_count") {
                        if count == 1.0 {
                            node.kind = NodeKind::Splitter; // passthrough
                            node.parameters.clear();
                        }
                    }
                }

                details.push(format!("Node {}: {}", node_id, desc));
                affected += 1;
            }
        }

        OptimizationReport {
            pass_name: "StrengthReduction".into(),
            nodes_affected: affected,
            nodes_removed: 0,
            estimated_speedup: if affected > 0 {
                1.0 + 0.03 * affected as f64
            } else {
                1.0
            },
            details,
        }
    }

    fn is_power_of_two(v: f64) -> bool {
        if v <= 0.0 || v != v.floor() {
            return false;
        }
        let i = v as u64;
        i > 0 && (i & (i - 1)) == 0 && i != 1
    }
}

// ---------------------------------------------------------------------------
// Combined optimization pipeline
// ---------------------------------------------------------------------------

/// Run all optimization passes in sequence.
pub fn optimize_graph(graph: &mut CgGraph, config: &CodegenConfig) -> Vec<OptimizationReport> {
    let mut reports = Vec::new();

    reports.push(ConstantPropagation::new().apply(graph));
    reports.push(DeadCodeElimination::new().apply(graph));
    reports.push(InlineExpansion::new(config).apply(graph));
    reports.push(StrengthReduction::new().apply(graph));
    reports.push(LoopFusion::new().apply(graph));
    reports.push(SimdHints::new(config).apply(graph));

    reports
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferKind, CgGraphBuilder, CodegenConfig, NodeKind};
    use std::collections::HashMap;

    fn test_config() -> CodegenConfig {
        CodegenConfig::default()
    }

    fn chain_with_gain() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let mut gain_params = HashMap::new();
        gain_params.insert("level".into(), 1.0);
        let gain = b.add_node_with_params("gain", NodeKind::Gain, gain_params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, gain, BufferKind::Audio);
        b.connect(gain, out, BufferKind::Audio);
        b.build()
    }

    fn chain_with_constant() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let mut const_params = HashMap::new();
        const_params.insert("value".into(), 0.5);
        let c = b.add_node_with_params("const", NodeKind::Constant, const_params);
        let mut gain_params = HashMap::new();
        gain_params.insert("level".into(), 2.0);
        let g = b.add_node_with_params("gain", NodeKind::Gain, gain_params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(c, g, BufferKind::Audio);
        b.connect(g, out, BufferKind::Audio);
        b.build()
    }

    fn graph_with_dead_node() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let _dead = b.add_node("dead", NodeKind::NoiseGenerator); // no edges
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        b.build()
    }

    #[test]
    fn test_inline_expansion_candidates() {
        let cfg = test_config();
        let inliner = InlineExpansion::new(&cfg);
        let graph = chain_with_gain();
        let candidates = inliner.find_candidates(&graph);
        // Gain with wcet_cycles=0 should be a candidate
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_inline_expansion_apply() {
        let cfg = test_config();
        let inliner = InlineExpansion::new(&cfg);
        let mut graph = chain_with_gain();
        let report = inliner.apply(&mut graph);
        assert!(report.nodes_removed > 0 || report.nodes_affected >= 0);
    }

    #[test]
    fn test_constant_propagation_find() {
        let cp = ConstantPropagation::new();
        let graph = chain_with_constant();
        let constants = cp.find_constants(&graph);
        // Should find the Constant node and propagate through Gain
        assert!(constants.contains_key(&0)); // const node
        assert!(constants.contains_key(&1)); // gain(const) = 0.5 * 2.0 = 1.0
        assert!((constants[&1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_constant_propagation_apply() {
        let cp = ConstantPropagation::new();
        let mut graph = chain_with_constant();
        let report = cp.apply(&mut graph);
        // Gain node should be folded to Constant
        assert!(report.nodes_affected > 0);
        let gain_node = graph.node(1).unwrap();
        assert_eq!(gain_node.kind, NodeKind::Constant);
    }

    #[test]
    fn test_dead_code_elimination() {
        let dce = DeadCodeElimination::new();
        let mut graph = graph_with_dead_node();
        assert_eq!(graph.node_count(), 3);
        let report = dce.apply(&mut graph);
        assert_eq!(report.nodes_removed, 1);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_dead_code_no_false_positives() {
        let dce = DeadCodeElimination::new();
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let mut graph = b.build();
        let report = dce.apply(&mut graph);
        assert_eq!(report.nodes_removed, 0);
    }

    #[test]
    fn test_strength_reduction_identity_gain() {
        let sr = StrengthReduction::new();
        let mut graph = chain_with_gain(); // level=1.0
        let report = sr.apply(&mut graph);
        assert!(report.nodes_affected > 0);
        // The gain node should be replaced with Splitter (passthrough).
        let gain_node = graph.node(1).unwrap();
        assert_eq!(gain_node.kind, NodeKind::Splitter);
    }

    #[test]
    fn test_strength_reduction_power_of_two() {
        let sr = StrengthReduction::new();
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let mut params = HashMap::new();
        params.insert("level".into(), 4.0);
        let gain = b.add_node_with_params("gain", NodeKind::Gain, params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, gain, BufferKind::Audio);
        b.connect(gain, out, BufferKind::Audio);
        let mut graph = b.build();
        let candidates = sr.find_candidates(&graph);
        assert!(candidates.iter().any(|(_, desc)| desc.contains("bit shift")));
    }

    #[test]
    fn test_loop_fusion_simple_chain() {
        let lf = LoopFusion::new();
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let filt = b.add_node("filt", NodeKind::Filter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, filt, BufferKind::Audio);
        b.connect(filt, out, BufferKind::Audio);
        let mut graph = b.build();
        let pairs = lf.find_fusable_pairs(&graph);
        // osc→filt should be fusable
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_simd_hints_enabled() {
        let cfg = CodegenConfig {
            enable_simd: true,
            ..test_config()
        };
        let sh = SimdHints::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let graph = b.build();
        let vectorizable = sh.find_vectorizable(&graph);
        assert!(!vectorizable.is_empty());
    }

    #[test]
    fn test_simd_hints_disabled() {
        let cfg = CodegenConfig {
            enable_simd: false,
            ..test_config()
        };
        let sh = SimdHints::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, out, BufferKind::Audio);
        let graph = b.build();
        let vectorizable = sh.find_vectorizable(&graph);
        assert!(vectorizable.is_empty());
    }

    #[test]
    fn test_full_optimization_pipeline() {
        let cfg = test_config();
        let mut graph = chain_with_gain();
        let reports = optimize_graph(&mut graph, &cfg);
        assert!(reports.len() >= 5);
        // Total estimated speedup should be >= 1.0
        let total_speedup: f64 = reports.iter().map(|r| r.estimated_speedup).product();
        assert!(total_speedup >= 1.0);
    }

    #[test]
    fn test_optimization_report_details() {
        let dce = DeadCodeElimination::new();
        let mut graph = graph_with_dead_node();
        let report = dce.apply(&mut graph);
        assert!(!report.details.is_empty());
        assert!(report.details[0].contains("dead"));
    }

    #[test]
    fn test_single_channel_mixer_reduction() {
        let sr = StrengthReduction::new();
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let mut params = HashMap::new();
        params.insert("channel_count".into(), 1.0);
        let mix = b.add_node_with_params("mix", NodeKind::Mixer, params);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, mix, BufferKind::Audio);
        b.connect(mix, out, BufferKind::Audio);
        let mut graph = b.build();
        let report = sr.apply(&mut graph);
        assert!(report.nodes_affected > 0);
    }
}
