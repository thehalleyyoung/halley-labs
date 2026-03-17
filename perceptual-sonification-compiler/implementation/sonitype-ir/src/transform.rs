//! Graph transformations.
//!
//! - [`GraphTransformer`] – split, insert, replace, flatten, normalise.
//! - [`StreamSplitter`] – split multi-channel nodes into per-channel chains.
//! - [`GraphCloner`] – deep copy with new IDs.
//! - [`SubgraphExtractor`] – extract connected components.

use std::collections::{HashMap, HashSet};
use crate::graph::{
    AudioGraph, AudioEdge, NodeId,
    NodeType, BufferType,
};
use crate::{IrError, IrResult};

// ---------------------------------------------------------------------------
// GraphTransformer
// ---------------------------------------------------------------------------

/// Collection of graph-level transformations.
pub struct GraphTransformer;

impl GraphTransformer {
    /// Split a node into `n` parallel sub-nodes of the same type, each
    /// receiving a copy of the original inputs and feeding into a new Mixer.
    pub fn split_into_parallel(
        graph: &mut AudioGraph,
        node_id: NodeId,
        n: usize,
    ) -> IrResult<Vec<NodeId>> {
        if n < 2 {
            return Err(IrError::ValidationError("split count must be >= 2".into()));
        }
        let orig = graph.node(node_id).ok_or(IrError::NodeNotFound(node_id))?.clone();

        // Collect incoming and outgoing edges.
        let incoming: Vec<AudioEdge> = graph.incoming_edges(node_id).into_iter().cloned().collect();
        let outgoing: Vec<AudioEdge> = graph.outgoing_edges(node_id).into_iter().cloned().collect();

        // Remove the original node.
        graph.remove_node(node_id)?;

        // Create n copies.
        let mut copy_ids = Vec::new();
        for i in 0..n {
            let name = format!("{}_{}", orig.name, i);
            let id = graph.add_node_with_params(&name, orig.node_type.clone(), orig.parameters.clone());
            copy_ids.push(id);
        }

        // Re-wire incoming edges to each copy.
        for copy_id in &copy_ids {
            let copy_node = graph.node(*copy_id).unwrap();
            if let Some(first_input) = copy_node.first_input() {
                let dp = first_input.id;
                for edge in &incoming {
                    let _ = graph.add_edge(edge.source_node, edge.source_port, *copy_id, dp);
                }
            }
        }

        // Create a Mixer to combine the copies.
        let mixer_id = graph.add_node(
            &format!("{}_mixer", orig.name),
            NodeType::Mixer { channel_count: n },
        );

        for (i, copy_id) in copy_ids.iter().enumerate() {
            let copy_out = graph.node(*copy_id).and_then(|n| n.first_output()).map(|p| p.id);
            let mixer_in = graph.node(mixer_id).map(|n| n.inputs.get(i).map(|p| p.id)).flatten();
            if let (Some(sp), Some(dp)) = (copy_out, mixer_in) {
                let _ = graph.add_edge(*copy_id, sp, mixer_id, dp);
            }
        }

        // Wire mixer output to original outgoing edges.
        let mixer_out = graph.node(mixer_id).and_then(|n| n.first_output()).map(|p| p.id);
        if let Some(sp) = mixer_out {
            for edge in &outgoing {
                let _ = graph.add_edge(mixer_id, sp, edge.dest_node, edge.dest_port);
            }
        }

        copy_ids.push(mixer_id);
        Ok(copy_ids)
    }

    /// Insert a monitoring/metering node on every edge of a given buffer type.
    pub fn insert_metering_nodes(
        graph: &mut AudioGraph,
        meter_name_prefix: &str,
        buffer_type_filter: BufferType,
    ) -> IrResult<Vec<NodeId>> {
        let edges_to_split: Vec<AudioEdge> = graph.edges.iter()
            .filter(|e| e.buffer_type == buffer_type_filter)
            .cloned()
            .collect();

        let mut meter_ids = Vec::new();
        for edge in &edges_to_split {
            let meter_id = graph.add_node(
                &format!("{}_{}", meter_name_prefix, meter_ids.len()),
                NodeType::Gain { level: 1.0 }, // unity gain = transparent meter
            );
            let meter_in = graph.node(meter_id).and_then(|n| n.first_input()).map(|p| p.id);
            let meter_out = graph.node(meter_id).and_then(|n| n.first_output()).map(|p| p.id);

            if let (Some(mi), Some(mo)) = (meter_in, meter_out) {
                // Remove the original edge.
                graph.remove_edge(edge.id)?;
                // source → meter.
                let _ = graph.add_edge(edge.source_node, edge.source_port, meter_id, mi);
                // meter → dest.
                let _ = graph.add_edge(meter_id, mo, edge.dest_node, edge.dest_port);
            }
            meter_ids.push(meter_id);
        }
        Ok(meter_ids)
    }

    /// Replace a node with an equivalent subgraph. The subgraph must have
    /// exactly one "entry" node (matching the original's inputs) and one "exit"
    /// node (matching the original's outputs).
    pub fn replace_with_subgraph(
        graph: &mut AudioGraph,
        target: NodeId,
        subgraph: &AudioGraph,
        entry: NodeId,
        exit: NodeId,
    ) -> IrResult<HashMap<NodeId, NodeId>> {
        let _orig = graph.node(target).ok_or(IrError::NodeNotFound(target))?.clone();
        let incoming: Vec<AudioEdge> = graph.incoming_edges(target).into_iter().cloned().collect();
        let outgoing: Vec<AudioEdge> = graph.outgoing_edges(target).into_iter().cloned().collect();

        graph.remove_node(target)?;

        // Merge subgraph into main graph.
        let id_map = graph.merge(subgraph);

        let new_entry = *id_map.get(&entry).ok_or(IrError::NodeNotFound(entry))?;
        let new_exit = *id_map.get(&exit).ok_or(IrError::NodeNotFound(exit))?;

        // Wire incoming to entry.
        if let Some(entry_node) = graph.node(new_entry) {
            if let Some(ip) = entry_node.first_input() {
                let dp = ip.id;
                for e in &incoming {
                    let _ = graph.add_edge(e.source_node, e.source_port, new_entry, dp);
                }
            }
        }

        // Wire exit to outgoing.
        if let Some(exit_node) = graph.node(new_exit) {
            if let Some(op) = exit_node.first_output() {
                let sp = op.id;
                for e in &outgoing {
                    let _ = graph.add_edge(new_exit, sp, e.dest_node, e.dest_port);
                }
            }
        }

        Ok(id_map)
    }

    /// Flatten all Splitter/Merger patterns by removing the intermediate nodes
    /// and connecting sources directly to destinations.
    pub fn flatten_split_merge(graph: &mut AudioGraph) -> IrResult<usize> {
        let mut removed = 0usize;
        loop {
            let splitter = graph.nodes.iter()
                .find(|n| matches!(n.node_type, NodeType::Splitter))
                .map(|n| n.id);
            let id = match splitter {
                Some(id) => id,
                None => break,
            };
            let incoming: Vec<AudioEdge> = graph.incoming_edges(id).into_iter().cloned().collect();
            let outgoing: Vec<AudioEdge> = graph.outgoing_edges(id).into_iter().cloned().collect();
            graph.remove_node(id)?;
            // Connect each source to each destination directly.
            for ie in &incoming {
                for oe in &outgoing {
                    let _ = graph.add_edge(ie.source_node, ie.source_port, oe.dest_node, oe.dest_port);
                }
            }
            removed += 1;
        }
        // Similarly for Mergers.
        loop {
            let merger = graph.nodes.iter()
                .find(|n| matches!(n.node_type, NodeType::Merger))
                .map(|n| n.id);
            let id = match merger {
                Some(id) => id,
                None => break,
            };
            let incoming: Vec<AudioEdge> = graph.incoming_edges(id).into_iter().cloned().collect();
            let outgoing: Vec<AudioEdge> = graph.outgoing_edges(id).into_iter().cloned().collect();
            graph.remove_node(id)?;
            for ie in &incoming {
                for oe in &outgoing {
                    let _ = graph.add_edge(ie.source_node, ie.source_port, oe.dest_node, oe.dest_port);
                }
            }
            removed += 1;
        }
        Ok(removed)
    }

    /// Normalize the graph into a canonical form:
    /// 1. Remove all isolated nodes (no edges).
    /// 2. Sort nodes by topological order.
    /// 3. Reassign sequential IDs.
    pub fn normalize(graph: &mut AudioGraph) -> IrResult<HashMap<NodeId, NodeId>> {
        // Remove isolated nodes.
        let connected: HashSet<NodeId> = graph.edges.iter()
            .flat_map(|e| vec![e.source_node, e.dest_node])
            .collect();
        let isolated: Vec<NodeId> = graph.nodes.iter()
            .filter(|n| !connected.contains(&n.id))
            .map(|n| n.id)
            .collect();
        for id in &isolated {
            graph.remove_node(*id)?;
        }

        // Topological sort.
        graph.topological_sort()?;

        // Build ID remapping.
        let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
        for (i, &old_id) in graph.topological_order.iter().enumerate() {
            remap.insert(old_id, NodeId((i + 1) as u64));
        }

        // Apply remapping.
        for node in &mut graph.nodes {
            if let Some(&new_id) = remap.get(&node.id) {
                node.id = new_id;
            }
        }
        for edge in &mut graph.edges {
            if let Some(&nid) = remap.get(&edge.source_node) { edge.source_node = nid; }
            if let Some(&nid) = remap.get(&edge.dest_node) { edge.dest_node = nid; }
        }
        graph.topological_order = graph.topological_order.iter()
            .filter_map(|id| remap.get(id).copied())
            .collect();
        graph.rebuild_node_index();

        Ok(remap)
    }
}

// ---------------------------------------------------------------------------
// StreamSplitter
// ---------------------------------------------------------------------------

/// Split multi-channel Mixer nodes into per-channel processing chains.
pub struct StreamSplitter;

impl StreamSplitter {
    /// Split a Mixer node into individual gain nodes.
    pub fn split_mixer(graph: &mut AudioGraph, mixer_id: NodeId) -> IrResult<Vec<NodeId>> {
        let mixer = graph.node(mixer_id).ok_or(IrError::NodeNotFound(mixer_id))?.clone();
        let ch_count = match &mixer.node_type {
            NodeType::Mixer { channel_count } => *channel_count,
            _ => return Err(IrError::ValidationError("node is not a Mixer".into())),
        };

        let incoming: Vec<AudioEdge> = graph.incoming_edges(mixer_id).into_iter().cloned().collect();
        let outgoing: Vec<AudioEdge> = graph.outgoing_edges(mixer_id).into_iter().cloned().collect();
        graph.remove_node(mixer_id)?;

        let mut gain_ids = Vec::new();
        for i in 0..ch_count {
            let gid = graph.add_node(
                &format!("{}_ch{}", mixer.name, i),
                NodeType::Gain { level: 1.0 },
            );
            gain_ids.push(gid);
        }

        // Wire original inputs to corresponding gains.
        for (i, edge) in incoming.iter().enumerate() {
            if i < gain_ids.len() {
                let gin = graph.node(gain_ids[i]).and_then(|n| n.first_input()).map(|p| p.id);
                if let Some(dp) = gin {
                    let _ = graph.add_edge(edge.source_node, edge.source_port, gain_ids[i], dp);
                }
            }
        }

        // Create a new mixer to recombine.
        let new_mix = graph.add_node(
            &format!("{}_recombine", mixer.name),
            NodeType::Mixer { channel_count: ch_count },
        );
        for (i, &gid) in gain_ids.iter().enumerate() {
            let gout = graph.node(gid).and_then(|n| n.first_output()).map(|p| p.id);
            let min = graph.node(new_mix).map(|n| n.inputs.get(i).map(|p| p.id)).flatten();
            if let (Some(sp), Some(dp)) = (gout, min) {
                let _ = graph.add_edge(gid, sp, new_mix, dp);
            }
        }

        // Wire new mixer to original outgoing.
        let mout = graph.node(new_mix).and_then(|n| n.first_output()).map(|p| p.id);
        if let Some(sp) = mout {
            for edge in &outgoing {
                let _ = graph.add_edge(new_mix, sp, edge.dest_node, edge.dest_port);
            }
        }

        gain_ids.push(new_mix);
        Ok(gain_ids)
    }
}

// ---------------------------------------------------------------------------
// GraphCloner
// ---------------------------------------------------------------------------

/// Deep copy of a graph with fresh IDs.
pub struct GraphCloner;

impl GraphCloner {
    /// Clone the entire graph, returning the new graph and an ID mapping.
    pub fn deep_clone(source: &AudioGraph) -> (AudioGraph, HashMap<NodeId, NodeId>) {
        let mut dest = AudioGraph::new(source.sample_rate, source.block_size);
        let id_map = dest.merge(source);
        (dest, id_map)
    }

    /// Clone a subset of nodes (and the edges between them).
    pub fn clone_subset(
        source: &AudioGraph,
        node_ids: &HashSet<NodeId>,
    ) -> IrResult<(AudioGraph, HashMap<NodeId, NodeId>)> {
        let sub = source.subgraph_from_set(node_ids)?;
        Ok(Self::deep_clone(&sub))
    }
}

// ---------------------------------------------------------------------------
// SubgraphExtractor
// ---------------------------------------------------------------------------

/// Extract connected components from the graph.
pub struct SubgraphExtractor;

impl SubgraphExtractor {
    /// Extract all connected components as separate graphs.
    pub fn extract_all_components(graph: &AudioGraph) -> IrResult<Vec<AudioGraph>> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut components = Vec::new();
        for node in &graph.nodes {
            if visited.contains(&node.id) { continue; }
            let comp = graph.connected_component(node.id)?;
            visited.extend(comp.iter());
            let sub = graph.subgraph_from_set(&comp)?;
            components.push(sub);
        }
        Ok(components)
    }

    /// Extract the component containing a specific node.
    pub fn extract_component(graph: &AudioGraph, node_id: NodeId) -> IrResult<AudioGraph> {
        let comp = graph.connected_component(node_id)?;
        graph.subgraph_from_set(&comp)
    }

    /// Extract the subgraph between two nodes (all paths).
    pub fn extract_between(
        graph: &AudioGraph,
        from: NodeId,
        to: NodeId,
    ) -> IrResult<AudioGraph> {
        let fwd = graph.reachable_forward(from);
        let bwd = graph.reachable_backward(to);
        let between: HashSet<NodeId> = fwd.intersection(&bwd).copied().collect();
        if between.is_empty() {
            return Err(IrError::InvalidSubgraph(
                format!("no path from {} to {}", from.0, to.0),
            ));
        }
        graph.subgraph_from_set(&between)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{AudioGraph, NodeType, GraphBuilder, BufferType};
    use crate::node::Waveform;

    fn basic_graph() -> AudioGraph {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, g) = b.add_gain("gain", 0.5);
        let (b, out) = b.add_output("out", "stereo");
        b.connect(osc, "out", g, "in")
         .connect(g, "out", out, "in")
         .build().unwrap()
    }

    #[test]
    fn test_split_into_parallel() {
        let mut g = basic_graph();
        let gain_id = g.nodes.iter().find(|n| n.name == "gain").unwrap().id;
        let result = GraphTransformer::split_into_parallel(&mut g, gain_id, 3);
        assert!(result.is_ok());
        let ids = result.unwrap();
        assert_eq!(ids.len(), 4); // 3 copies + 1 mixer
    }

    #[test]
    fn test_split_invalid_count() {
        let mut g = basic_graph();
        let gain_id = g.nodes.iter().find(|n| n.name == "gain").unwrap().id;
        let result = GraphTransformer::split_into_parallel(&mut g, gain_id, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_metering_nodes() {
        let mut g = basic_graph();
        let meters = GraphTransformer::insert_metering_nodes(&mut g, "meter", BufferType::AudioBuffer);
        assert!(meters.is_ok());
        let ids = meters.unwrap();
        assert_eq!(ids.len(), 2); // two audio edges
    }

    #[test]
    fn test_replace_with_subgraph() {
        let mut g = basic_graph();
        let gain_id = g.nodes.iter().find(|n| n.name == "gain").unwrap().id;

        // Build a subgraph: gain→delay
        let mut sub = AudioGraph::new(48000.0, 256);
        let sg = sub.add_node("sub_gain", NodeType::Gain { level: 0.3 });
        let sd = sub.add_node("sub_delay", NodeType::Delay { samples: 100 });
        sub.add_edge_by_name(sg, "out", sd, "in").unwrap();

        let result = GraphTransformer::replace_with_subgraph(&mut g, gain_id, &sub, sg, sd);
        assert!(result.is_ok());
        // Original gain removed, two new nodes added.
        assert!(g.nodes.iter().any(|n| n.name == "sub_gain"));
        assert!(g.nodes.iter().any(|n| n.name == "sub_delay"));
    }

    #[test]
    fn test_flatten_split_merge() {
        let mut g = AudioGraph::default();
        let osc = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let spl = g.add_node("spl", NodeType::Splitter);
        let out = g.add_node("out", NodeType::Output { format: "stereo".into() });
        g.add_edge_by_name(osc, "out", spl, "in").unwrap();
        g.add_edge_by_name(spl, "out_0", out, "in").unwrap();

        let removed = GraphTransformer::flatten_split_merge(&mut g).unwrap();
        assert!(removed >= 1);
    }

    #[test]
    fn test_normalize() {
        let mut g = basic_graph();
        // Add an isolated node.
        g.add_node("isolated", NodeType::Constant(99.0));
        let remap = GraphTransformer::normalize(&mut g).unwrap();
        assert!(!remap.is_empty());
        // Isolated node should be gone.
        assert!(!g.nodes.iter().any(|n| n.name == "isolated"));
    }

    #[test]
    fn test_stream_splitter() {
        let mut g = AudioGraph::default();
        let osc1 = g.add_node("osc1", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let osc2 = g.add_node("osc2", NodeType::Oscillator { waveform: Waveform::Saw, frequency: 880.0 });
        let mix = g.add_node("mix", NodeType::Mixer { channel_count: 2 });
        let out = g.add_node("out", NodeType::Output { format: "stereo".into() });
        let mix_in0 = g.node(mix).unwrap().inputs[0].id;
        let mix_in1 = g.node(mix).unwrap().inputs[1].id;
        let o1out = g.node(osc1).unwrap().first_output().unwrap().id;
        let o2out = g.node(osc2).unwrap().first_output().unwrap().id;
        g.add_edge(osc1, o1out, mix, mix_in0).unwrap();
        g.add_edge(osc2, o2out, mix, mix_in1).unwrap();
        g.add_edge_by_name(mix, "out", out, "in").unwrap();

        let result = StreamSplitter::split_mixer(&mut g, mix);
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_cloner() {
        let g = basic_graph();
        let (clone, id_map) = GraphCloner::deep_clone(&g);
        assert_eq!(clone.node_count(), g.node_count());
        assert_eq!(clone.edge_count(), g.edge_count());
        assert_eq!(id_map.len(), g.node_count());
    }

    #[test]
    fn test_graph_cloner_subset() {
        let g = basic_graph();
        let ids: HashSet<NodeId> = g.nodes.iter().take(2).map(|n| n.id).collect();
        let (sub, _) = GraphCloner::clone_subset(&g, &ids).unwrap();
        assert_eq!(sub.node_count(), 2);
    }

    #[test]
    fn test_subgraph_extractor_all_components() {
        let mut g = AudioGraph::default();
        let a = g.add_node("a", NodeType::Constant(1.0));
        let b = g.add_node("b", NodeType::Constant(2.0));
        // Two isolated nodes = two components.
        let comps = SubgraphExtractor::extract_all_components(&g).unwrap();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_subgraph_extractor_between() {
        let g = basic_graph();
        let ids = g.node_ids();
        let sub = SubgraphExtractor::extract_between(&g, ids[0], ids[2]).unwrap();
        assert_eq!(sub.node_count(), 3);
    }

    #[test]
    fn test_subgraph_extractor_no_path() {
        let mut g = AudioGraph::default();
        let a = g.add_node("a", NodeType::Constant(1.0));
        let b = g.add_node("b", NodeType::Constant(2.0));
        let result = SubgraphExtractor::extract_between(&g, a, b);
        assert!(result.is_err());
    }
}
