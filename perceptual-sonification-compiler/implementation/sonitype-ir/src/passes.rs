//! Optimization pass framework and built-in passes.
//!
//! A [`Pass`] is a self-contained graph transformation that may remove dead
//! nodes, fold constants, fuse adjacent operations, reuse buffers, or
//! eliminate common sub-expressions.  The [`PassManager`] runs an ordered
//! sequence of passes and collects statistics.

use std::collections::{HashMap, HashSet};
use crate::graph::{AudioGraph, NodeId, NodeType};
use crate::IrResult;

// ---------------------------------------------------------------------------
// Pass trait & result
// ---------------------------------------------------------------------------

/// Statistics collected after one pass execution.
#[derive(Debug, Clone, Default)]
pub struct PassResult {
    pub modified: bool,
    pub nodes_removed: usize,
    pub nodes_added: usize,
    pub edges_removed: usize,
    pub edges_added: usize,
    pub messages: Vec<String>,
}

impl PassResult {
    pub fn unchanged() -> Self { Self::default() }
    pub fn modified_with(nodes_removed: usize, edges_removed: usize) -> Self {
        Self { modified: true, nodes_removed, edges_removed, ..Default::default() }
    }
}

/// A single optimisation pass over the audio graph.
pub trait Pass {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Execute the pass, mutating the graph in place.
    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult>;

    /// Optional: declares pass names that must run before this one.
    fn dependencies(&self) -> Vec<&str> { vec![] }

    /// Optional: declares pass names that must run after this one.
    fn invalidates(&self) -> Vec<&str> { vec![] }
}

// ---------------------------------------------------------------------------
// PassManager
// ---------------------------------------------------------------------------

/// Ordered collection of passes.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    max_iterations: usize,
}

impl Default for PassManager {
    fn default() -> Self {
        Self { passes: Vec::new(), max_iterations: 10 }
    }
}

/// Aggregated statistics across all passes.
#[derive(Debug, Clone, Default)]
pub struct PassManagerStats {
    pub total_nodes_removed: usize,
    pub total_nodes_added: usize,
    pub total_edges_removed: usize,
    pub total_edges_added: usize,
    pub iterations: usize,
    pub per_pass: Vec<(String, PassResult)>,
}

impl PassManager {
    pub fn new() -> Self { Self::default() }

    /// Set the maximum number of fixed-point iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Append a pass.
    pub fn add_pass<P: Pass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Add pass (builder pattern).
    pub fn with_pass<P: Pass + 'static>(mut self, pass: P) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Run all passes once in order.
    pub fn run_once(&self, graph: &mut AudioGraph) -> IrResult<PassManagerStats> {
        let mut stats = PassManagerStats::default();
        stats.iterations = 1;
        for pass in &self.passes {
            let result = pass.run(graph)?;
            stats.total_nodes_removed += result.nodes_removed;
            stats.total_nodes_added += result.nodes_added;
            stats.total_edges_removed += result.edges_removed;
            stats.total_edges_added += result.edges_added;
            stats.per_pass.push((pass.name().to_string(), result));
        }
        Ok(stats)
    }

    /// Run passes iteratively until a fixed point (no pass modifies the graph)
    /// or `max_iterations` is reached.
    pub fn run_to_fixed_point(&self, graph: &mut AudioGraph) -> IrResult<PassManagerStats> {
        let mut stats = PassManagerStats::default();
        for iteration in 0..self.max_iterations {
            let mut changed = false;
            for pass in &self.passes {
                let result = pass.run(graph)?;
                if result.modified { changed = true; }
                stats.total_nodes_removed += result.nodes_removed;
                stats.total_nodes_added += result.nodes_added;
                stats.total_edges_removed += result.edges_removed;
                stats.total_edges_added += result.edges_added;
                stats.per_pass.push((pass.name().to_string(), result));
            }
            stats.iterations = iteration + 1;
            if !changed { break; }
        }
        Ok(stats)
    }

    /// Order passes respecting declared dependencies (topological sort).
    pub fn order_passes(&mut self) {
        // Simple insertion-sort-like approach: move passes with unmet
        // dependencies rightward.
        let n = self.passes.len();
        let names: Vec<String> = self.passes.iter().map(|p| p.name().to_string()).collect();
        let mut ordered: Vec<usize> = (0..n).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                let deps = self.passes[ordered[i]].dependencies();
                for dep in deps {
                    if let Some(dep_pos) = names.iter().position(|nm| nm == dep) {
                        let dep_ordered_pos = ordered.iter().position(|&x| x == dep_pos).unwrap();
                        let cur_ordered_pos = ordered.iter().position(|&x| x == ordered[i]).unwrap();
                        if dep_ordered_pos > cur_ordered_pos {
                            ordered.swap(dep_ordered_pos, cur_ordered_pos);
                            changed = true;
                        }
                    }
                }
            }
        }
        let mut new_passes: Vec<Box<dyn Pass>> = Vec::with_capacity(n);
        // Safety: we need to move out of self.passes by index.
        let mut old = std::mem::take(&mut self.passes);
        // Use a Vec of Options to allow taking by index.
        let mut opts: Vec<Option<Box<dyn Pass>>> = old.drain(..).map(Some).collect();
        for &idx in &ordered {
            if let Some(p) = opts[idx].take() {
                new_passes.push(p);
            }
        }
        self.passes = new_passes;
    }

    /// Create a pass manager with all built-in passes.
    pub fn with_builtin_passes() -> Self {
        Self::new()
            .with_pass(DeadStreamElimination)
            .with_pass(ConstantFolding)
            .with_pass(NodeFusion)
            .with_pass(BufferReuse)
            .with_pass(CommonSubexpressionElimination)
    }

    /// Number of registered passes.
    pub fn pass_count(&self) -> usize { self.passes.len() }
}

// ---------------------------------------------------------------------------
// DeadStreamElimination
// ---------------------------------------------------------------------------

/// Remove nodes that have no path to any Output node.
pub struct DeadStreamElimination;

impl Pass for DeadStreamElimination {
    fn name(&self) -> &str { "DeadStreamElimination" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let outputs = graph.output_nodes();
        if outputs.is_empty() {
            return Ok(PassResult::unchanged());
        }
        // Backward reachability from all output nodes.
        let mut live: HashSet<NodeId> = HashSet::new();
        for &out in &outputs {
            let reach = graph.reachable_backward(out);
            live.extend(reach);
        }
        let all: HashSet<NodeId> = graph.node_ids().into_iter().collect();
        let dead: Vec<NodeId> = all.difference(&live).copied().collect();
        if dead.is_empty() {
            return Ok(PassResult::unchanged());
        }
        let edges_before = graph.edge_count();
        for id in &dead {
            graph.remove_node(*id)?;
        }
        let edges_after = graph.edge_count();
        Ok(PassResult {
            modified: true,
            nodes_removed: dead.len(),
            edges_removed: edges_before - edges_after,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// ConstantFolding
// ---------------------------------------------------------------------------

/// Fold chains of Constant → Gain by pre-multiplying the constant value.
pub struct ConstantFolding;

impl Pass for ConstantFolding {
    fn name(&self) -> &str { "ConstantFolding" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let mut folds = 0usize;
        // Identify Constant nodes whose *only* consumer is another Constant or
        // a Gain.  In those cases we can merge the value.
        let const_ids: Vec<(NodeId, f64)> = graph.nodes.iter()
            .filter_map(|n| {
                if let NodeType::Constant(v) = &n.node_type { Some((n.id, *v)) } else { None }
            })
            .collect();

        for (cid, cval) in const_ids {
            let consumers: Vec<NodeId> = graph.outgoing_edges(cid).iter().map(|e| e.dest_node).collect();
            if consumers.len() != 1 { continue; }
            let consumer_id = consumers[0];
            let consumer = match graph.node(consumer_id) { Some(c) => c, None => continue };
            match &consumer.node_type {
                NodeType::Gain { level } => {
                    let new_val = cval * level;
                    // Remove gain, replace constant value.
                    let gain_successors: Vec<(NodeId, crate::graph::PortId, crate::graph::PortId)> = graph
                        .outgoing_edges(consumer_id)
                        .iter()
                        .map(|e| (e.dest_node, e.source_port, e.dest_port))
                        .collect();

                    // Remove gain node.
                    graph.remove_node(consumer_id)?;
                    // Update the constant.
                    if let Some(cnode) = graph.node_mut(cid) {
                        cnode.node_type = NodeType::Constant(new_val);
                    }
                    // Re-wire constant to gain's successors.
                    if let Some(cnode) = graph.node(cid) {
                        if let Some(out_port) = cnode.first_output() {
                            let op = out_port.id;
                            for (dest, _old_sp, dp) in &gain_successors {
                                let _ = graph.add_edge(cid, op, *dest, *dp);
                            }
                        }
                    }
                    folds += 1;
                }
                NodeType::Constant(v2) => {
                    // Two constants chained: keep the downstream one with
                    // the sum (treating the edge as additive).
                    let new_val = cval + v2;
                    graph.remove_node(cid)?;
                    if let Some(cnode) = graph.node_mut(consumer_id) {
                        cnode.node_type = NodeType::Constant(new_val);
                    }
                    folds += 1;
                }
                _ => {}
            }
        }

        if folds == 0 {
            Ok(PassResult::unchanged())
        } else {
            Ok(PassResult { modified: true, nodes_removed: folds, ..Default::default() })
        }
    }
}

// ---------------------------------------------------------------------------
// NodeFusion
// ---------------------------------------------------------------------------

/// Merge compatible adjacent nodes (e.g. two consecutive Gain nodes become one
/// with their levels multiplied).
pub struct NodeFusion;

impl Pass for NodeFusion {
    fn name(&self) -> &str { "NodeFusion" }

    fn dependencies(&self) -> Vec<&str> { vec!["DeadStreamElimination"] }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let mut fused = 0usize;
        let mut edges_removed = 0usize;

        loop {
            let mut found = None;
            // Find a pair of consecutive Gain nodes.
            for edge in &graph.edges {
                let src = match graph.node(edge.source_node) { Some(n) => n, None => continue };
                let dst = match graph.node(edge.dest_node) { Some(n) => n, None => continue };
                if let (NodeType::Gain { level: l1 }, NodeType::Gain { level: l2 }) =
                    (&src.node_type, &dst.node_type)
                {
                    // Only fuse if the source has exactly one outgoing edge.
                    let out_count = graph.outgoing_edges(edge.source_node).len();
                    if out_count == 1 {
                        found = Some((edge.source_node, edge.dest_node, l1 * l2));
                        break;
                    }
                }
            }

            if let Some((src_id, dst_id, combined)) = found {
                // Rewire: src's predecessors → dst, with combined level.
                let incoming: Vec<crate::graph::AudioEdge> = graph.incoming_edges(src_id).iter().map(|e| (*e).clone()).collect();
                let edges_before = graph.edge_count();
                graph.remove_node(src_id)?;
                if let Some(dst_node) = graph.node_mut(dst_id) {
                    dst_node.node_type = NodeType::Gain { level: combined };
                }
                // Re-add the predecessor edges targeting dst.
                if let Some(dst_node) = graph.node(dst_id) {
                    if let Some(in_port) = dst_node.first_input() {
                        let dp = in_port.id;
                        for old_edge in &incoming {
                            let _ = graph.add_edge(old_edge.source_node, old_edge.source_port, dst_id, dp);
                        }
                    }
                }
                fused += 1;
                edges_removed += edges_before - graph.edge_count();
            } else {
                break;
            }
        }

        // Fuse consecutive Delay nodes.
        loop {
            let mut found = None;
            for edge in &graph.edges {
                let src = match graph.node(edge.source_node) { Some(n) => n, None => continue };
                let dst = match graph.node(edge.dest_node) { Some(n) => n, None => continue };
                if let (NodeType::Delay { samples: s1 }, NodeType::Delay { samples: s2 }) =
                    (&src.node_type, &dst.node_type)
                {
                    let out_count = graph.outgoing_edges(edge.source_node).len();
                    if out_count == 1 {
                        found = Some((edge.source_node, edge.dest_node, s1 + s2));
                        break;
                    }
                }
            }
            if let Some((src_id, dst_id, combined)) = found {
                let incoming: Vec<crate::graph::AudioEdge> = graph.incoming_edges(src_id).iter().map(|e| (*e).clone()).collect();
                graph.remove_node(src_id)?;
                if let Some(dst_node) = graph.node_mut(dst_id) {
                    dst_node.node_type = NodeType::Delay { samples: combined };
                }
                if let Some(dst_node) = graph.node(dst_id) {
                    if let Some(ip) = dst_node.first_input() {
                        let dp = ip.id;
                        for old_edge in &incoming {
                            let _ = graph.add_edge(old_edge.source_node, old_edge.source_port, dst_id, dp);
                        }
                    }
                }
                fused += 1;
            } else {
                break;
            }
        }

        if fused == 0 {
            Ok(PassResult::unchanged())
        } else {
            Ok(PassResult { modified: true, nodes_removed: fused, edges_removed, ..Default::default() })
        }
    }
}

// ---------------------------------------------------------------------------
// BufferReuse
// ---------------------------------------------------------------------------

/// Minimize buffer allocations by reusing buffers once their consumer is done.
/// This is an analysis pass that annotates edges rather than removing nodes,
/// but we model it as an optimization pass that emits reuse suggestions.
pub struct BufferReuse;

impl Pass for BufferReuse {
    fn name(&self) -> &str { "BufferReuse" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        graph.ensure_sorted()?;
        let order = graph.topological_order.clone();

        // For each node in topological order, determine when each of its output
        // buffers is last consumed.
        let mut last_use: HashMap<(NodeId, crate::graph::PortId), usize> = HashMap::new();
        for (step, &nid) in order.iter().enumerate() {
            for edge in &graph.edges {
                if edge.dest_node == nid {
                    let key = (edge.source_node, edge.source_port);
                    last_use.insert(key, step);
                }
            }
        }

        // Greedy reuse: assign buffer slots.
        let mut free_slots: Vec<usize> = Vec::new();
        let mut _assignments: HashMap<(NodeId, crate::graph::PortId), usize> = HashMap::new();
        let mut reuse_count = 0usize;
        let mut slot_counter = 0usize;

        for (step, &nid) in order.iter().enumerate() {
            // Free slots whose last use was a previous step.
            let to_free: Vec<(NodeId, crate::graph::PortId)> = last_use
                .iter()
                .filter(|(_, &lu)| lu < step)
                .map(|(k, _)| *k)
                .collect();
            for key in to_free {
                if let Some(slot) = _assignments.remove(&key) {
                    free_slots.push(slot);
                }
                last_use.remove(&key);
            }

            // Assign slots for this node's outputs.
            let node = match graph.node(nid) { Some(n) => n, None => continue };
            for port in &node.outputs {
                let key = (nid, port.id);
                if let Some(slot) = free_slots.pop() {
                    _assignments.insert(key, slot);
                    reuse_count += 1;
                } else {
                    _assignments.insert(key, slot_counter);
                    slot_counter += 1;
                }
            }
        }

        let msgs = vec![
            format!("total buffer slots: {}", slot_counter),
            format!("reuse events: {}", reuse_count),
        ];
        Ok(PassResult {
            modified: reuse_count > 0,
            messages: msgs,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// CommonSubexpressionElimination
// ---------------------------------------------------------------------------

/// Identify identical subgraph structures and merge them.
pub struct CommonSubexpressionElimination;

impl CommonSubexpressionElimination {
    /// Compute a structural hash for a node and its predecessors (depth 1).
    fn node_signature(graph: &AudioGraph, nid: NodeId) -> Option<String> {
        let node = graph.node(nid)?;
        let mut sig = format!("{:?}", node.node_type);
        let mut preds: Vec<String> = graph.predecessors(nid)
            .iter()
            .filter_map(|&pid| graph.node(pid).map(|n| format!("{:?}", n.node_type)))
            .collect();
        preds.sort();
        for p in preds {
            sig.push_str("|");
            sig.push_str(&p);
        }
        Some(sig)
    }
}

impl Pass for CommonSubexpressionElimination {
    fn name(&self) -> &str { "CommonSubexpressionElimination" }

    fn run(&self, graph: &mut AudioGraph) -> IrResult<PassResult> {
        let ids = graph.node_ids();
        let mut sig_map: HashMap<String, Vec<NodeId>> = HashMap::new();
        for &nid in &ids {
            if let Some(sig) = Self::node_signature(graph, nid) {
                sig_map.entry(sig).or_default().push(nid);
            }
        }

        let mut removed = 0usize;
        let mut edges_changed = 0usize;

        for (_sig, group) in &sig_map {
            if group.len() < 2 { continue; }
            // Skip output and source nodes.
            let non_trivial: Vec<NodeId> = group.iter().copied().filter(|&id| {
                graph.node(id).map(|n| !n.node_type.is_sink() && !n.node_type.is_source()).unwrap_or(false)
            }).collect();
            if non_trivial.len() < 2 { continue; }

            let keep = non_trivial[0];
            for &dup in &non_trivial[1..] {
                // Rewire all consumers of `dup` to `keep`.
                let dup_consumers: Vec<crate::graph::AudioEdge> = graph.outgoing_edges(dup)
                    .iter().map(|e| (*e).clone()).collect();

                let keep_out = match graph.node(keep).and_then(|n| n.first_output()) {
                    Some(p) => p.id,
                    None => continue,
                };

                for consumer_edge in &dup_consumers {
                    let _ = graph.add_edge(keep, keep_out, consumer_edge.dest_node, consumer_edge.dest_port);
                    edges_changed += 1;
                }
                graph.remove_node(dup)?;
                removed += 1;
            }
        }

        if removed == 0 {
            Ok(PassResult::unchanged())
        } else {
            Ok(PassResult {
                modified: true,
                nodes_removed: removed,
                edges_removed: edges_changed,
                ..Default::default()
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{AudioGraph, NodeType, GraphBuilder};
    use crate::node::{Waveform, FilterType};

    fn build_osc_gain_out() -> AudioGraph {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, g) = b.add_gain("g", 0.5);
        let (b, out) = b.add_output("out", "stereo");
        b.connect(osc, "out", g, "in")
         .connect(g, "out", out, "in")
         .build().unwrap()
    }

    #[test]
    fn test_dead_stream_elimination_no_change() {
        let mut g = build_osc_gain_out();
        let r = DeadStreamElimination.run(&mut g).unwrap();
        assert!(!r.modified);
    }

    #[test]
    fn test_dead_stream_elimination_removes_dead() {
        let mut g = build_osc_gain_out();
        // Add an isolated node.
        g.add_node("dead", NodeType::Constant(42.0));
        assert_eq!(g.node_count(), 4);
        let r = DeadStreamElimination.run(&mut g).unwrap();
        assert!(r.modified);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_constant_folding() {
        let mut g = AudioGraph::default();
        let c = g.add_node("c", NodeType::Constant(2.0));
        let gain = g.add_node("gain", NodeType::Gain { level: 3.0 });
        let out = g.add_node("out", NodeType::Output { format: "mono".into() });
        // Constant → Gain (control → audio port mismatch for realistic, but
        // for this test we just check the pass logic with a direct edge).
        // Use port IDs directly.
        let c_out = g.node(c).unwrap().first_output().unwrap().id;
        let gain_control = g.node(gain).unwrap().inputs.iter()
            .find(|p| p.name == "gain_mod").unwrap().id;
        let _ = g.add_edge(c, c_out, gain, gain_control);
        g.add_edge_by_name(gain, "out", out, "in").unwrap();

        let r = ConstantFolding.run(&mut g).unwrap();
        assert!(r.modified);
        // The constant should now hold 2.0 * 3.0 = 6.0 and gain is removed.
        let cnodes: Vec<_> = g.nodes.iter().filter(|n| matches!(n.node_type, NodeType::Constant(_))).collect();
        assert_eq!(cnodes.len(), 1);
        if let NodeType::Constant(v) = &cnodes[0].node_type {
            assert!((v - 6.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_node_fusion_gain() {
        let mut g = AudioGraph::default();
        let osc = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let g1 = g.add_node("g1", NodeType::Gain { level: 0.5 });
        let g2 = g.add_node("g2", NodeType::Gain { level: 0.4 });
        let out = g.add_node("out", NodeType::Output { format: "mono".into() });
        g.add_edge_by_name(osc, "out", g1, "in").unwrap();
        g.add_edge_by_name(g1, "out", g2, "in").unwrap();
        g.add_edge_by_name(g2, "out", out, "in").unwrap();
        assert_eq!(g.node_count(), 4);

        let r = NodeFusion.run(&mut g).unwrap();
        assert!(r.modified);
        // The two gains should become one with level 0.2.
        let gains: Vec<_> = g.nodes.iter().filter(|n| matches!(n.node_type, NodeType::Gain { .. })).collect();
        assert_eq!(gains.len(), 1);
        if let NodeType::Gain { level } = &gains[0].node_type {
            assert!((level - 0.2).abs() < 0.001);
        }
    }

    #[test]
    fn test_node_fusion_delay() {
        let mut g = AudioGraph::default();
        let osc = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let d1 = g.add_node("d1", NodeType::Delay { samples: 100 });
        let d2 = g.add_node("d2", NodeType::Delay { samples: 200 });
        let out = g.add_node("out", NodeType::Output { format: "mono".into() });
        g.add_edge_by_name(osc, "out", d1, "in").unwrap();
        g.add_edge_by_name(d1, "out", d2, "in").unwrap();
        g.add_edge_by_name(d2, "out", out, "in").unwrap();

        let r = NodeFusion.run(&mut g).unwrap();
        assert!(r.modified);
        let delays: Vec<_> = g.nodes.iter().filter(|n| matches!(n.node_type, NodeType::Delay { .. })).collect();
        assert_eq!(delays.len(), 1);
        if let NodeType::Delay { samples } = &delays[0].node_type {
            assert_eq!(*samples, 300);
        }
    }

    #[test]
    fn test_buffer_reuse() {
        let mut g = build_osc_gain_out();
        let r = BufferReuse.run(&mut g).unwrap();
        // Should produce buffer slot info.
        assert!(!r.messages.is_empty());
    }

    #[test]
    fn test_cse_no_duplicates() {
        let mut g = build_osc_gain_out();
        let r = CommonSubexpressionElimination.run(&mut g).unwrap();
        assert!(!r.modified);
    }

    #[test]
    fn test_cse_removes_duplicate() {
        let mut g = AudioGraph::default();
        let osc1 = g.add_node("osc1", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let osc2 = g.add_node("osc2", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let g1 = g.add_node("g1", NodeType::Gain { level: 0.5 });
        let g2 = g.add_node("g2", NodeType::Gain { level: 0.5 });
        let mix = g.add_node("mix", NodeType::Mixer { channel_count: 2 });
        let out = g.add_node("out", NodeType::Output { format: "mono".into() });
        g.add_edge_by_name(osc1, "out", g1, "in").unwrap();
        g.add_edge_by_name(osc2, "out", g2, "in").unwrap();
        // Mix the two gains.
        let mix_in0 = g.node(mix).unwrap().inputs[0].id;
        let mix_in1 = g.node(mix).unwrap().inputs[1].id;
        let g1_out = g.node(g1).unwrap().first_output().unwrap().id;
        let g2_out = g.node(g2).unwrap().first_output().unwrap().id;
        let _ = g.add_edge(g1, g1_out, mix, mix_in0);
        let _ = g.add_edge(g2, g2_out, mix, mix_in1);
        g.add_edge_by_name(mix, "out", out, "in").unwrap();

        let before = g.node_count();
        let r = CommonSubexpressionElimination.run(&mut g).unwrap();
        assert!(r.modified);
        assert!(g.node_count() < before);
    }

    #[test]
    fn test_pass_manager_builtin() {
        let pm = PassManager::with_builtin_passes();
        assert_eq!(pm.pass_count(), 5);
    }

    #[test]
    fn test_pass_manager_run_once() {
        let mut g = build_osc_gain_out();
        g.add_node("dead", NodeType::Constant(1.0));
        let pm = PassManager::new().with_pass(DeadStreamElimination);
        let stats = pm.run_once(&mut g).unwrap();
        assert_eq!(stats.total_nodes_removed, 1);
    }

    #[test]
    fn test_pass_manager_fixed_point() {
        let mut g = build_osc_gain_out();
        g.add_node("dead1", NodeType::Constant(1.0));
        let pm = PassManager::new()
            .with_pass(DeadStreamElimination)
            .with_max_iterations(5);
        let stats = pm.run_to_fixed_point(&mut g).unwrap();
        // First iteration removes the dead node, second finds nothing.
        assert!(stats.iterations <= 2);
    }

    #[test]
    fn test_pass_order() {
        let mut pm = PassManager::new()
            .with_pass(NodeFusion)        // depends on DSE
            .with_pass(DeadStreamElimination);
        pm.order_passes();
        // After ordering, DSE should come first.
        assert_eq!(pm.passes[0].name(), "DeadStreamElimination");
    }

    #[test]
    fn test_pass_result_helpers() {
        let pr = PassResult::unchanged();
        assert!(!pr.modified);
        let pr2 = PassResult::modified_with(3, 2);
        assert!(pr2.modified);
        assert_eq!(pr2.nodes_removed, 3);
    }
}
