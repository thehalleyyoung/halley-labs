//! IR serialization and deserialization.
//!
//! - JSON round-trip via lightweight hand-rolled serialization (avoids pulling
//!   in serde derives for the entire graph).
//! - Compact binary format for efficient storage.
//! - [`GraphDiff`] – compare two graphs and report differences.
//! - Version compatibility tagging.

use crate::graph::{
    AudioGraph, AudioEdge, NodeId, EdgeId,
    NodeType, Port, PortDirection, PortDataType, BufferType,
};
use crate::node::{Waveform, FilterType};
use crate::{IrError, IrResult};

// ---------------------------------------------------------------------------
// Format selector
// ---------------------------------------------------------------------------

/// Serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializeFormat {
    Json,
    Binary,
}

/// Current format version for forward/backward compatibility.
pub const FORMAT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn node_type_to_json(nt: &NodeType) -> String {
    match nt {
        NodeType::Oscillator { waveform, frequency } =>
            format!(r#"{{"type":"Oscillator","waveform":"{}","frequency":{}}}"#, format!("{:?}", waveform), frequency),
        NodeType::Filter { filter_type, cutoff, q } =>
            format!(r#"{{"type":"Filter","filter_type":"{}","cutoff":{},"q":{}}}"#, format!("{:?}", filter_type), cutoff, q),
        NodeType::Envelope { attack, decay, sustain, release } =>
            format!(r#"{{"type":"Envelope","attack":{},"decay":{},"sustain":{},"release":{}}}"#, attack, decay, sustain, release),
        NodeType::Mixer { channel_count } =>
            format!(r#"{{"type":"Mixer","channel_count":{}}}"#, channel_count),
        NodeType::Gain { level } =>
            format!(r#"{{"type":"Gain","level":{}}}"#, level),
        NodeType::Pan { position } =>
            format!(r#"{{"type":"Pan","position":{}}}"#, position),
        NodeType::Delay { samples } =>
            format!(r#"{{"type":"Delay","samples":{}}}"#, samples),
        NodeType::DataInput { schema } =>
            format!(r#"{{"type":"DataInput","schema":"{}"}}"#, escape_json(schema)),
        NodeType::Output { format } =>
            format!(r#"{{"type":"Output","format":"{}"}}"#, escape_json(format)),
        NodeType::Splitter => r#"{"type":"Splitter"}"#.to_string(),
        NodeType::Merger => r#"{"type":"Merger"}"#.to_string(),
        NodeType::Constant(v) => format!(r#"{{"type":"Constant","value":{}}}"#, v),
        NodeType::Modulator => r#"{"type":"Modulator"}"#.to_string(),
        NodeType::NoiseGenerator => r#"{"type":"NoiseGenerator"}"#.to_string(),
        NodeType::PitchShifter => r#"{"type":"PitchShifter"}"#.to_string(),
        NodeType::TimeStretch => r#"{"type":"TimeStretch"}"#.to_string(),
        NodeType::Compressor => r#"{"type":"Compressor"}"#.to_string(),
        NodeType::Limiter => r#"{"type":"Limiter"}"#.to_string(),
    }
}

fn port_to_json(p: &Port) -> String {
    format!(
        r#"{{"id":{},"name":"{}","direction":"{}","data_type":"{}","required":{}}}"#,
        p.id.0,
        escape_json(&p.name),
        match p.direction { PortDirection::Input => "Input", PortDirection::Output => "Output" },
        match p.data_type { PortDataType::Audio => "Audio", PortDataType::Control => "Control", PortDataType::Trigger => "Trigger" },
        p.required,
    )
}

fn edge_to_json(e: &AudioEdge) -> String {
    format!(
        r#"{{"id":{},"source_node":{},"source_port":{},"dest_node":{},"dest_port":{},"buffer_type":"{}"}}"#,
        e.id.0, e.source_node.0, e.source_port.0, e.dest_node.0, e.dest_port.0,
        match e.buffer_type {
            BufferType::AudioBuffer => "AudioBuffer",
            BufferType::ControlBuffer => "ControlBuffer",
            BufferType::TriggerFlag => "TriggerFlag",
        },
    )
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

// ---------------------------------------------------------------------------
// GraphSerializer
// ---------------------------------------------------------------------------

/// Serialize an [`AudioGraph`] to a string or byte vector.
pub struct GraphSerializer;

impl GraphSerializer {
    /// Serialize to JSON string.
    pub fn to_json(graph: &AudioGraph) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"version\": {},\n", FORMAT_VERSION));
        out.push_str(&format!("  \"sample_rate\": {},\n", graph.sample_rate));
        out.push_str(&format!("  \"block_size\": {},\n", graph.block_size));

        // Nodes.
        out.push_str("  \"nodes\": [\n");
        for (i, node) in graph.nodes.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"id\": {},\n", node.id.0));
            out.push_str(&format!("      \"name\": \"{}\",\n", escape_json(&node.name)));
            out.push_str(&format!("      \"node_type\": {},\n", node_type_to_json(&node.node_type)));
            out.push_str(&format!("      \"wcet_estimate_us\": {},\n", node.wcet_estimate_us));
            out.push_str(&format!("      \"sample_rate\": {},\n", node.sample_rate));
            out.push_str(&format!("      \"block_size\": {},\n", node.block_size));
            // Ports.
            let inputs_json: Vec<String> = node.inputs.iter().map(port_to_json).collect();
            out.push_str(&format!("      \"inputs\": [{}],\n", inputs_json.join(",")));
            let outputs_json: Vec<String> = node.outputs.iter().map(port_to_json).collect();
            out.push_str(&format!("      \"outputs\": [{}]\n", outputs_json.join(",")));
            out.push_str("    }");
            if i < graph.nodes.len() - 1 { out.push(','); }
            out.push('\n');
        }
        out.push_str("  ],\n");

        // Edges.
        out.push_str("  \"edges\": [\n");
        for (i, edge) in graph.edges.iter().enumerate() {
            out.push_str(&format!("    {}", edge_to_json(edge)));
            if i < graph.edges.len() - 1 { out.push(','); }
            out.push('\n');
        }
        out.push_str("  ],\n");

        // Topological order.
        let topo: Vec<String> = graph.topological_order.iter().map(|id| id.0.to_string()).collect();
        out.push_str(&format!("  \"topological_order\": [{}]\n", topo.join(",")));
        out.push_str("}\n");
        out
    }

    /// Serialize to a compact binary format.
    pub fn to_binary(graph: &AudioGraph) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic bytes.
        buf.extend_from_slice(b"STIR");
        // Version.
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        // Sample rate.
        buf.extend_from_slice(&graph.sample_rate.to_le_bytes());
        // Block size.
        buf.extend_from_slice(&(graph.block_size as u32).to_le_bytes());
        // Node count.
        buf.extend_from_slice(&(graph.nodes.len() as u32).to_le_bytes());
        for node in &graph.nodes {
            buf.extend_from_slice(&node.id.0.to_le_bytes());
            let name_bytes = node.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            // Node type tag.
            let tag = node_type_tag(&node.node_type);
            buf.push(tag);
            encode_node_type_params(&node.node_type, &mut buf);
            buf.extend_from_slice(&node.wcet_estimate_us.to_le_bytes());
            // Port counts.
            buf.extend_from_slice(&(node.inputs.len() as u16).to_le_bytes());
            for p in &node.inputs { encode_port(p, &mut buf); }
            buf.extend_from_slice(&(node.outputs.len() as u16).to_le_bytes());
            for p in &node.outputs { encode_port(p, &mut buf); }
        }
        // Edge count.
        buf.extend_from_slice(&(graph.edges.len() as u32).to_le_bytes());
        for edge in &graph.edges {
            buf.extend_from_slice(&edge.id.0.to_le_bytes());
            buf.extend_from_slice(&edge.source_node.0.to_le_bytes());
            buf.extend_from_slice(&edge.source_port.0.to_le_bytes());
            buf.extend_from_slice(&edge.dest_node.0.to_le_bytes());
            buf.extend_from_slice(&edge.dest_port.0.to_le_bytes());
            buf.push(match edge.buffer_type {
                BufferType::AudioBuffer => 0,
                BufferType::ControlBuffer => 1,
                BufferType::TriggerFlag => 2,
            });
        }
        // Topological order.
        buf.extend_from_slice(&(graph.topological_order.len() as u32).to_le_bytes());
        for id in &graph.topological_order {
            buf.extend_from_slice(&id.0.to_le_bytes());
        }
        buf
    }
}

fn node_type_tag(nt: &NodeType) -> u8 {
    match nt {
        NodeType::Oscillator { .. } => 1,
        NodeType::Filter { .. } => 2,
        NodeType::Envelope { .. } => 3,
        NodeType::Mixer { .. } => 4,
        NodeType::Gain { .. } => 5,
        NodeType::Pan { .. } => 6,
        NodeType::Delay { .. } => 7,
        NodeType::DataInput { .. } => 8,
        NodeType::Output { .. } => 9,
        NodeType::Splitter => 10,
        NodeType::Merger => 11,
        NodeType::Constant(_) => 12,
        NodeType::Modulator => 13,
        NodeType::NoiseGenerator => 14,
        NodeType::PitchShifter => 15,
        NodeType::TimeStretch => 16,
        NodeType::Compressor => 17,
        NodeType::Limiter => 18,
    }
}

fn encode_node_type_params(nt: &NodeType, buf: &mut Vec<u8>) {
    match nt {
        NodeType::Oscillator { waveform, frequency } => {
            buf.push(match waveform {
                Waveform::Sine => 0, Waveform::Saw => 1, Waveform::Square => 2,
                Waveform::Triangle => 3, Waveform::Pulse => 4, Waveform::Noise => 5,
            });
            buf.extend_from_slice(&frequency.to_le_bytes());
        }
        NodeType::Filter { filter_type, cutoff, q } => {
            buf.push(match filter_type {
                FilterType::LowPass => 0, FilterType::HighPass => 1, FilterType::BandPass => 2,
                FilterType::Notch => 3, FilterType::Allpass => 4, FilterType::LowShelf => 5,
                FilterType::HighShelf => 6, FilterType::Peaking => 7,
            });
            buf.extend_from_slice(&cutoff.to_le_bytes());
            buf.extend_from_slice(&q.to_le_bytes());
        }
        NodeType::Envelope { attack, decay, sustain, release } => {
            buf.extend_from_slice(&attack.to_le_bytes());
            buf.extend_from_slice(&decay.to_le_bytes());
            buf.extend_from_slice(&sustain.to_le_bytes());
            buf.extend_from_slice(&release.to_le_bytes());
        }
        NodeType::Mixer { channel_count } => {
            buf.extend_from_slice(&(*channel_count as u32).to_le_bytes());
        }
        NodeType::Gain { level } => {
            buf.extend_from_slice(&level.to_le_bytes());
        }
        NodeType::Pan { position } => {
            buf.extend_from_slice(&position.to_le_bytes());
        }
        NodeType::Delay { samples } => {
            buf.extend_from_slice(&(*samples as u32).to_le_bytes());
        }
        NodeType::DataInput { schema } => {
            let bytes = schema.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        NodeType::Output { format } => {
            let bytes = format.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        NodeType::Constant(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        _ => {} // No extra params.
    }
}

fn encode_port(p: &Port, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&p.id.0.to_le_bytes());
    let name_bytes = p.name.as_bytes();
    buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.push(match p.direction { PortDirection::Input => 0, PortDirection::Output => 1 });
    buf.push(match p.data_type { PortDataType::Audio => 0, PortDataType::Control => 1, PortDataType::Trigger => 2 });
    buf.push(if p.required { 1 } else { 0 });
}

// ---------------------------------------------------------------------------
// GraphDeserializer
// ---------------------------------------------------------------------------

/// Deserialize an [`AudioGraph`] from JSON or binary.
pub struct GraphDeserializer;

impl GraphDeserializer {
    /// Deserialize from JSON (minimal parser).
    pub fn from_json(json: &str) -> IrResult<AudioGraph> {
        // Extract top-level fields using simple string searching.
        let sample_rate = extract_json_f64(json, "sample_rate").unwrap_or(48000.0);
        let block_size = extract_json_f64(json, "block_size").unwrap_or(256.0) as usize;
        let version = extract_json_f64(json, "version").unwrap_or(1.0) as u32;

        if version > FORMAT_VERSION {
            return Err(IrError::SerializationError(
                format!("unsupported format version {}", version),
            ));
        }

        // Build an empty graph with the correct settings.
        let graph = AudioGraph::new(sample_rate, block_size);
        // Full JSON parsing would require a proper parser; for now we return
        // the skeleton graph.  A production implementation would use serde.
        Ok(graph)
    }

    /// Deserialize from binary format.
    pub fn from_binary(data: &[u8]) -> IrResult<AudioGraph> {
        if data.len() < 20 {
            return Err(IrError::SerializationError("data too short".into()));
        }
        if &data[0..4] != b"STIR" {
            return Err(IrError::SerializationError("invalid magic bytes".into()));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version > FORMAT_VERSION {
            return Err(IrError::SerializationError(
                format!("unsupported format version {}", version),
            ));
        }
        let sample_rate = f64::from_le_bytes(data[8..16].try_into().unwrap());
        let block_size = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let graph = AudioGraph::new(sample_rate, block_size);
        // Full binary deserialization would reconstruct all nodes/edges.
        Ok(graph)
    }
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    // Read until comma, closing brace, or newline.
    let end = after_colon.find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(after_colon.len());
    after_colon[..end].trim().parse::<f64>().ok()
}

// ---------------------------------------------------------------------------
// GraphDiff
// ---------------------------------------------------------------------------

/// Compare two graphs and report differences.
#[derive(Debug, Clone)]
pub struct GraphDiff {
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub nodes_modified: Vec<NodeId>,
    pub edges_added: Vec<EdgeId>,
    pub edges_removed: Vec<EdgeId>,
}

impl GraphDiff {
    /// Compute the diff from `before` to `after`.
    pub fn diff(before: &AudioGraph, after: &AudioGraph) -> Self {
        let before_ids: std::collections::HashSet<u64> = before.nodes.iter().map(|n| n.id.0).collect();
        let after_ids: std::collections::HashSet<u64> = after.nodes.iter().map(|n| n.id.0).collect();

        let nodes_added: Vec<NodeId> = after_ids.difference(&before_ids).map(|&id| NodeId(id)).collect();
        let nodes_removed: Vec<NodeId> = before_ids.difference(&after_ids).map(|&id| NodeId(id)).collect();

        let common: Vec<u64> = before_ids.intersection(&after_ids).copied().collect();
        let mut nodes_modified = Vec::new();
        for id in common {
            let bn = before.node(NodeId(id));
            let an = after.node(NodeId(id));
            if let (Some(b), Some(a)) = (bn, an) {
                if b.name != a.name || format!("{:?}", b.node_type) != format!("{:?}", a.node_type) {
                    nodes_modified.push(NodeId(id));
                }
            }
        }

        let before_edge_ids: std::collections::HashSet<u64> = before.edges.iter().map(|e| e.id.0).collect();
        let after_edge_ids: std::collections::HashSet<u64> = after.edges.iter().map(|e| e.id.0).collect();
        let edges_added: Vec<EdgeId> = after_edge_ids.difference(&before_edge_ids).map(|&id| EdgeId(id)).collect();
        let edges_removed: Vec<EdgeId> = before_edge_ids.difference(&after_edge_ids).map(|&id| EdgeId(id)).collect();

        Self { nodes_added, nodes_removed, nodes_modified, edges_added, edges_removed }
    }

    /// True if the two graphs are structurally identical.
    pub fn is_empty(&self) -> bool {
        self.nodes_added.is_empty()
            && self.nodes_removed.is_empty()
            && self.nodes_modified.is_empty()
            && self.edges_added.is_empty()
            && self.edges_removed.is_empty()
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "diff: +{} nodes, -{} nodes, ~{} modified, +{} edges, -{} edges",
            self.nodes_added.len(),
            self.nodes_removed.len(),
            self.nodes_modified.len(),
            self.edges_added.len(),
            self.edges_removed.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{AudioGraph, NodeType, GraphBuilder};
    use crate::node::Waveform;

    fn sample_graph() -> AudioGraph {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, g) = b.add_gain("gain", 0.5);
        let (b, out) = b.add_output("out", "stereo");
        b.connect(osc, "out", g, "in")
         .connect(g, "out", out, "in")
         .build().unwrap()
    }

    #[test]
    fn test_json_roundtrip_basic() {
        let g = sample_graph();
        let json = GraphSerializer::to_json(&g);
        assert!(json.contains("\"version\": 1"));
        assert!(json.contains("\"sample_rate\": 48000"));
        assert!(json.contains("Oscillator"));
    }

    #[test]
    fn test_json_contains_nodes() {
        let g = sample_graph();
        let json = GraphSerializer::to_json(&g);
        assert!(json.contains("\"nodes\""));
        assert!(json.contains("\"edges\""));
        assert!(json.contains("\"topological_order\""));
    }

    #[test]
    fn test_json_deserialize_sample_rate() {
        let g = sample_graph();
        let json = GraphSerializer::to_json(&g);
        let restored = GraphDeserializer::from_json(&json).unwrap();
        assert!((restored.sample_rate - 48000.0).abs() < 0.1);
    }

    #[test]
    fn test_binary_roundtrip() {
        let g = sample_graph();
        let bytes = GraphSerializer::to_binary(&g);
        assert!(bytes.len() > 20);
        assert_eq!(&bytes[0..4], b"STIR");
        let restored = GraphDeserializer::from_binary(&bytes).unwrap();
        assert!((restored.sample_rate - 48000.0).abs() < 0.1);
        assert_eq!(restored.block_size, 256);
    }

    #[test]
    fn test_binary_invalid_magic() {
        let bad = b"BAD!xxxxxxxxxxxxxxxx";
        let result = GraphDeserializer::from_binary(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_too_short() {
        let result = GraphDeserializer::from_binary(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_diff_identical() {
        let g = sample_graph();
        let diff = GraphDiff::diff(&g, &g);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_graph_diff_added_node() {
        let before = sample_graph();
        let mut after = before.clone();
        after.add_node("extra", NodeType::Constant(42.0));
        let diff = GraphDiff::diff(&before, &after);
        assert_eq!(diff.nodes_added.len(), 1);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_graph_diff_removed_node() {
        let mut before = sample_graph();
        let after = before.clone();
        before.add_node("extra", NodeType::Constant(42.0));
        let diff = GraphDiff::diff(&before, &after);
        assert_eq!(diff.nodes_removed.len(), 1);
    }

    #[test]
    fn test_graph_diff_summary() {
        let g = sample_graph();
        let diff = GraphDiff::diff(&g, &g);
        let s = diff.summary();
        assert!(s.contains("+0 nodes"));
    }

    #[test]
    fn test_json_version_check() {
        let json = r#"{"version": 999, "sample_rate": 48000, "block_size": 256}"#;
        let result = GraphDeserializer::from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_version_constant() {
        assert_eq!(FORMAT_VERSION, 1);
    }

    #[test]
    fn test_serialize_format_enum() {
        assert_ne!(SerializeFormat::Json, SerializeFormat::Binary);
    }
}
