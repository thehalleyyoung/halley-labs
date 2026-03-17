//! Audio graph intermediate representation.
//!
//! The core data structure is [`AudioGraph`], a directed acyclic graph of typed
//! audio-processing nodes connected by edges that carry audio, control, or
//! trigger data.  The graph supports topological sorting (Kahn's algorithm),
//! cycle detection, subgraph extraction, graph merging, and a fluent
//! [`GraphBuilder`] API.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::node::NodeParameters;
use crate::{IrError, IrResult};

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a node inside a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

/// Unique identifier for a port on a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PortId(pub u64);

/// Unique identifier for an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub u64);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "n{}", self.0) }
}
impl std::fmt::Display for PortId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "p{}", self.0) }
}
impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "e{}", self.0) }
}

// ---------------------------------------------------------------------------
// Port / Edge / Buffer types
// ---------------------------------------------------------------------------

/// Direction of a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortDirection {
    Input,
    Output,
}

/// Data type carried by a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortDataType {
    /// Multi-sample audio buffer.
    Audio,
    /// Single-value control signal (e.g. LFO).
    Control,
    /// Boolean trigger / gate.
    Trigger,
}

/// A port on a node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Port {
    pub id: PortId,
    pub name: String,
    pub direction: PortDirection,
    pub data_type: PortDataType,
    pub required: bool,
}

/// Buffer strategy on an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferType {
    /// Full-rate audio buffer (block_size samples).
    AudioBuffer,
    /// Single-value control buffer.
    ControlBuffer,
    /// Boolean trigger.
    TriggerFlag,
}

impl BufferType {
    pub fn from_data_type(dt: PortDataType) -> Self {
        match dt {
            PortDataType::Audio => Self::AudioBuffer,
            PortDataType::Control => Self::ControlBuffer,
            PortDataType::Trigger => Self::TriggerFlag,
        }
    }
}

/// An edge connecting two ports in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AudioEdge {
    pub id: EdgeId,
    pub source_node: NodeId,
    pub source_port: PortId,
    pub dest_node: NodeId,
    pub dest_port: PortId,
    pub buffer_type: BufferType,
}

// ---------------------------------------------------------------------------
// NodeType
// ---------------------------------------------------------------------------

/// The kind of processing a node performs.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Oscillator { waveform: crate::node::Waveform, frequency: f64 },
    Filter { filter_type: crate::node::FilterType, cutoff: f64, q: f64 },
    Envelope { attack: f64, decay: f64, sustain: f64, release: f64 },
    Mixer { channel_count: usize },
    Gain { level: f64 },
    Pan { position: f64 },
    Delay { samples: usize },
    DataInput { schema: String },
    Output { format: String },
    Splitter,
    Merger,
    Constant(f64),
    Modulator,
    NoiseGenerator,
    PitchShifter,
    TimeStretch,
    Compressor,
    Limiter,
}

impl NodeType {
    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            Self::Oscillator { .. } => "Oscillator",
            Self::Filter { .. } => "Filter",
            Self::Envelope { .. } => "Envelope",
            Self::Mixer { .. } => "Mixer",
            Self::Gain { .. } => "Gain",
            Self::Pan { .. } => "Pan",
            Self::Delay { .. } => "Delay",
            Self::DataInput { .. } => "DataInput",
            Self::Output { .. } => "Output",
            Self::Splitter => "Splitter",
            Self::Merger => "Merger",
            Self::Constant(_) => "Constant",
            Self::Modulator => "Modulator",
            Self::NoiseGenerator => "NoiseGenerator",
            Self::PitchShifter => "PitchShifter",
            Self::TimeStretch => "TimeStretch",
            Self::Compressor => "Compressor",
            Self::Limiter => "Limiter",
        }
    }

    /// Default WCET estimate in microseconds.
    pub fn default_wcet_us(&self) -> f64 {
        match self {
            Self::Oscillator { .. } => 10.0,
            Self::Filter { .. } => 15.0,
            Self::Envelope { .. } => 5.0,
            Self::Mixer { channel_count, .. } => 3.0 + *channel_count as f64 * 1.0,
            Self::Gain { .. } => 2.0,
            Self::Pan { .. } => 3.0,
            Self::Delay { .. } => 8.0,
            Self::DataInput { .. } => 12.0,
            Self::Output { .. } => 8.0,
            Self::Splitter => 1.0,
            Self::Merger => 1.0,
            Self::Constant(_) => 0.5,
            Self::Modulator => 12.0,
            Self::NoiseGenerator => 6.0,
            Self::PitchShifter => 50.0,
            Self::TimeStretch => 80.0,
            Self::Compressor => 20.0,
            Self::Limiter => 15.0,
        }
    }

    /// Whether this node type produces output without any audio input.
    pub fn is_source(&self) -> bool {
        matches!(self,
            Self::Oscillator { .. }
            | Self::DataInput { .. }
            | Self::Constant(_)
            | Self::NoiseGenerator
        )
    }

    /// Whether this node type is a terminal sink.
    pub fn is_sink(&self) -> bool {
        matches!(self, Self::Output { .. })
    }

    /// Default ports for this node type.
    pub fn default_ports(&self, port_id_gen: &mut u64) -> (Vec<Port>, Vec<Port>) {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut next_port = || { let id = PortId(*port_id_gen); *port_id_gen += 1; id };

        match self {
            Self::Oscillator { .. } => {
                inputs.push(Port { id: next_port(), name: "freq_mod".into(), direction: PortDirection::Input, data_type: PortDataType::Control, required: false });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Filter { .. } => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                inputs.push(Port { id: next_port(), name: "cutoff_mod".into(), direction: PortDirection::Input, data_type: PortDataType::Control, required: false });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Envelope { .. } => {
                inputs.push(Port { id: next_port(), name: "gate".into(), direction: PortDirection::Input, data_type: PortDataType::Trigger, required: true });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Control, required: false });
            }
            Self::Mixer { channel_count } => {
                for i in 0..*channel_count {
                    inputs.push(Port { id: next_port(), name: format!("in_{}", i), direction: PortDirection::Input, data_type: PortDataType::Audio, required: false });
                }
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Gain { .. } => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                inputs.push(Port { id: next_port(), name: "gain_mod".into(), direction: PortDirection::Input, data_type: PortDataType::Control, required: false });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Pan { .. } => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                outputs.push(Port { id: next_port(), name: "left".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
                outputs.push(Port { id: next_port(), name: "right".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Delay { .. } => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::DataInput { .. } => {
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Control, required: false });
            }
            Self::Output { .. } => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
            }
            Self::Splitter => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                outputs.push(Port { id: next_port(), name: "out_0".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
                outputs.push(Port { id: next_port(), name: "out_1".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Merger => {
                inputs.push(Port { id: next_port(), name: "in_0".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: false });
                inputs.push(Port { id: next_port(), name: "in_1".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: false });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Constant(_) => {
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Control, required: false });
            }
            Self::Modulator => {
                inputs.push(Port { id: next_port(), name: "carrier".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                inputs.push(Port { id: next_port(), name: "modulator".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: false });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::NoiseGenerator => {
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::PitchShifter => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::TimeStretch => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Compressor => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                inputs.push(Port { id: next_port(), name: "sidechain".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: false });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
            Self::Limiter => {
                inputs.push(Port { id: next_port(), name: "in".into(), direction: PortDirection::Input, data_type: PortDataType::Audio, required: true });
                outputs.push(Port { id: next_port(), name: "out".into(), direction: PortDirection::Output, data_type: PortDataType::Audio, required: false });
            }
        }
        (inputs, outputs)
    }
}

// ---------------------------------------------------------------------------
// AudioGraphNode
// ---------------------------------------------------------------------------

/// A single processing node in the audio graph.
#[derive(Debug, Clone)]
pub struct AudioGraphNode {
    pub id: NodeId,
    pub name: String,
    pub node_type: NodeType,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub parameters: NodeParameters,
    pub wcet_estimate_us: f64,
    pub sample_rate: f64,
    pub block_size: usize,
    pub metadata: HashMap<String, String>,
}

impl AudioGraphNode {
    /// Find an input port by name.
    pub fn input_port(&self, name: &str) -> Option<&Port> {
        self.inputs.iter().find(|p| p.name == name)
    }

    /// Find an output port by name.
    pub fn output_port(&self, name: &str) -> Option<&Port> {
        self.outputs.iter().find(|p| p.name == name)
    }

    /// Find any port by id.
    pub fn port_by_id(&self, id: PortId) -> Option<&Port> {
        self.inputs.iter().chain(self.outputs.iter()).find(|p| p.id == id)
    }

    /// All port IDs belonging to this node.
    pub fn all_port_ids(&self) -> Vec<PortId> {
        self.inputs.iter().chain(self.outputs.iter()).map(|p| p.id).collect()
    }

    /// First output port (convenience).
    pub fn first_output(&self) -> Option<&Port> {
        self.outputs.first()
    }

    /// First input port (convenience).
    pub fn first_input(&self) -> Option<&Port> {
        self.inputs.first()
    }
}

// ---------------------------------------------------------------------------
// AudioGraph
// ---------------------------------------------------------------------------

/// Directed acyclic graph of audio processing nodes.
#[derive(Debug, Clone)]
pub struct AudioGraph {
    pub nodes: Vec<AudioGraphNode>,
    pub edges: Vec<AudioEdge>,
    pub topological_order: Vec<NodeId>,
    pub sample_rate: f64,
    pub block_size: usize,
    next_node_id: u64,
    next_edge_id: u64,
    next_port_id: u64,
    node_index: HashMap<NodeId, usize>,
}

impl Default for AudioGraph {
    fn default() -> Self {
        Self::new(48000.0, 256)
    }
}

impl AudioGraph {
    // -- construction -------------------------------------------------------

    pub fn new(sample_rate: f64, block_size: usize) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            topological_order: Vec::new(),
            sample_rate,
            block_size,
            next_node_id: 1,
            next_edge_id: 1,
            next_port_id: 1,
            node_index: HashMap::new(),
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Number of edges.
    pub fn edge_count(&self) -> usize { self.edges.len() }

    /// Check whether the graph is empty.
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    // -- node operations ----------------------------------------------------

    /// Add a node and return its ID.  Ports are generated from the node type.
    pub fn add_node(&mut self, name: &str, node_type: NodeType) -> NodeId {
        self.add_node_with_params(name, node_type, NodeParameters::None)
    }

    /// Add a node with explicit parameters.
    pub fn add_node_with_params(
        &mut self,
        name: &str,
        node_type: NodeType,
        params: NodeParameters,
    ) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let wcet = node_type.default_wcet_us();
        let (inputs, outputs) = node_type.default_ports(&mut self.next_port_id);
        let node = AudioGraphNode {
            id,
            name: name.to_string(),
            node_type,
            inputs,
            outputs,
            parameters: params,
            wcet_estimate_us: wcet,
            sample_rate: self.sample_rate,
            block_size: self.block_size,
            metadata: HashMap::new(),
        };
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.node_index.insert(id, idx);
        id
    }

    /// Remove a node and all edges attached to it.
    pub fn remove_node(&mut self, id: NodeId) -> IrResult<()> {
        if !self.node_index.contains_key(&id) {
            return Err(IrError::NodeNotFound(id));
        }
        // Remove edges touching this node.
        self.edges.retain(|e| e.source_node != id && e.dest_node != id);
        // Remove the node.
        self.nodes.retain(|n| n.id != id);
        self.rebuild_node_index();
        // Invalidate topological order.
        self.topological_order.clear();
        Ok(())
    }

    /// Look up a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&AudioGraphNode> {
        self.node_index.get(&id).map(|&i| &self.nodes[i])
    }

    /// Mutable reference to a node.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut AudioGraphNode> {
        self.node_index.get(&id).copied().map(move |i| &mut self.nodes[i])
    }

    /// All node IDs.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().map(|n| n.id).collect()
    }

    /// Return IDs of all Output nodes.
    pub fn output_nodes(&self) -> Vec<NodeId> {
        self.nodes.iter().filter(|n| n.node_type.is_sink()).map(|n| n.id).collect()
    }

    /// Return IDs of all source nodes (no required audio inputs).
    pub fn source_nodes(&self) -> Vec<NodeId> {
        self.nodes.iter().filter(|n| n.node_type.is_source()).map(|n| n.id).collect()
    }

    // -- edge operations ----------------------------------------------------

    /// Add an edge between two ports.
    pub fn add_edge(
        &mut self,
        source_node: NodeId,
        source_port: PortId,
        dest_node: NodeId,
        dest_port: PortId,
    ) -> IrResult<EdgeId> {
        // Validate existence.
        let src = self.node(source_node).ok_or(IrError::NodeNotFound(source_node))?;
        let src_port_obj = src.port_by_id(source_port)
            .ok_or(IrError::ValidationError(format!("source port {} not found on node {}", source_port, source_node)))?;
        let src_dt = src_port_obj.data_type;

        let dst = self.node(dest_node).ok_or(IrError::NodeNotFound(dest_node))?;
        let dst_port_obj = dst.port_by_id(dest_port)
            .ok_or(IrError::ValidationError(format!("dest port {} not found on node {}", dest_port, dest_node)))?;
        let dst_dt = dst_port_obj.data_type;

        if src_dt != dst_dt {
            let eid = EdgeId(self.next_edge_id);
            return Err(IrError::PortTypeMismatch { edge: eid, expected: dst_dt, found: src_dt });
        }

        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.edges.push(AudioEdge {
            id,
            source_node,
            source_port,
            dest_node,
            dest_port,
            buffer_type: BufferType::from_data_type(src_dt),
        });
        // Invalidate topological order.
        self.topological_order.clear();
        Ok(id)
    }

    /// Add edge by port names (convenience).
    pub fn add_edge_by_name(
        &mut self,
        source_node: NodeId,
        source_port_name: &str,
        dest_node: NodeId,
        dest_port_name: &str,
    ) -> IrResult<EdgeId> {
        let sp = self.node(source_node)
            .ok_or(IrError::NodeNotFound(source_node))?
            .output_port(source_port_name)
            .ok_or_else(|| IrError::ValidationError(format!("output port '{}' not found", source_port_name)))?
            .id;
        let dp = self.node(dest_node)
            .ok_or(IrError::NodeNotFound(dest_node))?
            .input_port(dest_port_name)
            .ok_or_else(|| IrError::ValidationError(format!("input port '{}' not found", dest_port_name)))?
            .id;
        self.add_edge(source_node, sp, dest_node, dp)
    }

    /// Remove an edge by ID.
    pub fn remove_edge(&mut self, id: EdgeId) -> IrResult<()> {
        let before = self.edges.len();
        self.edges.retain(|e| e.id != id);
        if self.edges.len() == before {
            return Err(IrError::EdgeNotFound(id));
        }
        self.topological_order.clear();
        Ok(())
    }

    /// Edges originating from a node.
    pub fn outgoing_edges(&self, node: NodeId) -> Vec<&AudioEdge> {
        self.edges.iter().filter(|e| e.source_node == node).collect()
    }

    /// Edges arriving at a node.
    pub fn incoming_edges(&self, node: NodeId) -> Vec<&AudioEdge> {
        self.edges.iter().filter(|e| e.dest_node == node).collect()
    }

    /// Immediate successors of a node.
    pub fn successors(&self, node: NodeId) -> Vec<NodeId> {
        self.outgoing_edges(node).iter().map(|e| e.dest_node).collect()
    }

    /// Immediate predecessors of a node.
    pub fn predecessors(&self, node: NodeId) -> Vec<NodeId> {
        self.incoming_edges(node).iter().map(|e| e.source_node).collect()
    }

    // -- topological sort ---------------------------------------------------

    /// Compute a topological ordering using Kahn's algorithm.
    /// Returns `Err(CycleDetected)` if the graph contains a cycle.
    pub fn topological_sort(&mut self) -> IrResult<()> {
        let ids: Vec<NodeId> = self.node_ids();
        let mut in_degree: HashMap<NodeId, usize> = ids.iter().map(|&id| (id, 0)).collect();
        for edge in &self.edges {
            *in_degree.entry(edge.dest_node).or_insert(0) += 1;
        }

        let mut queue: VecDeque<NodeId> = in_degree.iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        // Sort the initial queue so the ordering is deterministic.
        let mut init: Vec<NodeId> = queue.drain(..).collect();
        init.sort();
        queue.extend(init);

        let mut order = Vec::with_capacity(ids.len());
        while let Some(n) = queue.pop_front() {
            order.push(n);
            let succs: Vec<NodeId> = self.successors(n);
            let mut new_zero: Vec<NodeId> = Vec::new();
            for s in succs {
                if let Some(d) = in_degree.get_mut(&s) {
                    *d -= 1;
                    if *d == 0 {
                        new_zero.push(s);
                    }
                }
            }
            new_zero.sort();
            queue.extend(new_zero);
        }

        if order.len() != ids.len() {
            return Err(IrError::CycleDetected);
        }
        self.topological_order = order;
        Ok(())
    }

    /// Ensure topological order is computed.
    pub fn ensure_sorted(&mut self) -> IrResult<()> {
        if self.topological_order.len() != self.nodes.len() {
            self.topological_sort()?;
        }
        Ok(())
    }

    // -- validation ---------------------------------------------------------

    /// Quick validation: no cycles and all required inputs connected.
    pub fn validate(&mut self) -> IrResult<()> {
        self.topological_sort()?;
        for node in &self.nodes {
            for port in &node.inputs {
                if port.required {
                    let connected = self.edges.iter().any(|e| e.dest_node == node.id && e.dest_port == port.id);
                    if !connected {
                        return Err(IrError::UnconnectedInput { node: node.id, port: port.id });
                    }
                }
            }
        }
        Ok(())
    }

    /// Check for cycles without mutating topological_order.
    pub fn has_cycle(&self) -> bool {
        let mut in_degree: HashMap<NodeId, usize> = self.nodes.iter().map(|n| (n.id, 0)).collect();
        for e in &self.edges { *in_degree.entry(e.dest_node).or_insert(0) += 1; }
        let mut queue: VecDeque<NodeId> = in_degree.iter().filter(|(_, &d)| d == 0).map(|(&id, _)| id).collect();
        let mut visited = 0usize;
        while let Some(n) = queue.pop_front() {
            visited += 1;
            for e in self.edges.iter().filter(|e| e.source_node == n) {
                if let Some(d) = in_degree.get_mut(&e.dest_node) {
                    *d -= 1;
                    if *d == 0 { queue.push_back(e.dest_node); }
                }
            }
        }
        visited != self.nodes.len()
    }

    // -- subgraph extraction ------------------------------------------------

    /// Extract a subgraph rooted at `root_ids` (all ancestors included).
    pub fn extract_subgraph(&self, root_ids: &[NodeId]) -> IrResult<AudioGraph> {
        let mut keep: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = root_ids.to_vec();
        while let Some(id) = stack.pop() {
            if keep.insert(id) {
                for e in &self.edges {
                    if e.dest_node == id && !keep.contains(&e.source_node) {
                        stack.push(e.source_node);
                    }
                }
            }
        }
        self.subgraph_from_set(&keep)
    }

    /// Extract a subgraph that includes only the specified node IDs and the
    /// edges between them.
    pub fn subgraph_from_set(&self, ids: &HashSet<NodeId>) -> IrResult<AudioGraph> {
        let mut sub = AudioGraph::new(self.sample_rate, self.block_size);
        sub.next_node_id = self.next_node_id;
        sub.next_edge_id = self.next_edge_id;
        sub.next_port_id = self.next_port_id;
        for node in &self.nodes {
            if ids.contains(&node.id) {
                let idx = sub.nodes.len();
                sub.nodes.push(node.clone());
                sub.node_index.insert(node.id, idx);
            }
        }
        for edge in &self.edges {
            if ids.contains(&edge.source_node) && ids.contains(&edge.dest_node) {
                sub.edges.push(edge.clone());
            }
        }
        Ok(sub)
    }

    /// Extract the connected component containing `start`.
    pub fn connected_component(&self, start: NodeId) -> IrResult<HashSet<NodeId>> {
        if self.node(start).is_none() {
            return Err(IrError::NodeNotFound(start));
        }
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if visited.insert(id) {
                for e in &self.edges {
                    if e.source_node == id && !visited.contains(&e.dest_node) {
                        stack.push(e.dest_node);
                    }
                    if e.dest_node == id && !visited.contains(&e.source_node) {
                        stack.push(e.source_node);
                    }
                }
            }
        }
        Ok(visited)
    }

    // -- graph merging ------------------------------------------------------

    /// Merge another graph into this one. Node and edge IDs are remapped to
    /// avoid collisions.  Returns a map from old NodeId → new NodeId.
    pub fn merge(&mut self, other: &AudioGraph) -> HashMap<NodeId, NodeId> {
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut port_map: HashMap<PortId, PortId> = HashMap::new();

        for node in &other.nodes {
            let new_id = NodeId(self.next_node_id);
            self.next_node_id += 1;
            id_map.insert(node.id, new_id);
            let mut new_node = node.clone();
            new_node.id = new_id;
            // Remap port IDs.
            for p in new_node.inputs.iter_mut().chain(new_node.outputs.iter_mut()) {
                let new_pid = PortId(self.next_port_id);
                self.next_port_id += 1;
                port_map.insert(p.id, new_pid);
                p.id = new_pid;
            }
            let idx = self.nodes.len();
            self.nodes.push(new_node);
            self.node_index.insert(new_id, idx);
        }
        for edge in &other.edges {
            let new_eid = EdgeId(self.next_edge_id);
            self.next_edge_id += 1;
            self.edges.push(AudioEdge {
                id: new_eid,
                source_node: id_map[&edge.source_node],
                source_port: port_map[&edge.source_port],
                dest_node: id_map[&edge.dest_node],
                dest_port: port_map[&edge.dest_port],
                buffer_type: edge.buffer_type,
            });
        }
        self.topological_order.clear();
        id_map
    }

    // -- helpers ------------------------------------------------------------

    pub fn rebuild_node_index(&mut self) {
        self.node_index.clear();
        for (i, n) in self.nodes.iter().enumerate() {
            self.node_index.insert(n.id, i);
        }
    }

    /// All nodes reachable from `start` following forward edges.
    pub fn reachable_forward(&self, start: NodeId) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if visited.insert(id) {
                for e in &self.edges {
                    if e.source_node == id { stack.push(e.dest_node); }
                }
            }
        }
        visited
    }

    /// All nodes reachable from `start` following backward edges.
    pub fn reachable_backward(&self, start: NodeId) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if visited.insert(id) {
                for e in &self.edges {
                    if e.dest_node == id { stack.push(e.source_node); }
                }
            }
        }
        visited
    }

    /// Compute the total WCET of the critical path (longest path by WCET).
    pub fn critical_path_wcet(&self) -> f64 {
        if self.topological_order.is_empty() { return 0.0; }
        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        for &nid in &self.topological_order {
            let node_wcet = self.node(nid).map(|n| n.wcet_estimate_us).unwrap_or(0.0);
            let max_pred: f64 = self.predecessors(nid)
                .iter()
                .filter_map(|p| dist.get(p))
                .cloned()
                .fold(0.0f64, f64::max);
            dist.insert(nid, max_pred + node_wcet);
        }
        dist.values().cloned().fold(0.0f64, f64::max)
    }

    /// Total WCET summed over all nodes (serial execution bound).
    pub fn total_wcet(&self) -> f64 {
        self.nodes.iter().map(|n| n.wcet_estimate_us).sum()
    }

    /// Allocate new port id (public for graph builder).
    pub fn alloc_port_id(&mut self) -> PortId {
        let id = PortId(self.next_port_id);
        self.next_port_id += 1;
        id
    }

    /// Allocate new node id without inserting.
    pub fn alloc_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

/// Fluent API for building audio graphs.
pub struct GraphBuilder {
    graph: AudioGraph,
}

impl GraphBuilder {
    pub fn new(sample_rate: f64, block_size: usize) -> Self {
        Self { graph: AudioGraph::new(sample_rate, block_size) }
    }

    pub fn with_defaults() -> Self {
        Self::new(48000.0, 256)
    }

    pub fn add_oscillator(mut self, name: &str, waveform: crate::node::Waveform, freq: f64) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Oscillator { waveform, frequency: freq });
        (self, id)
    }

    pub fn add_filter(mut self, name: &str, ft: crate::node::FilterType, cutoff: f64, q: f64) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Filter { filter_type: ft, cutoff, q });
        (self, id)
    }

    pub fn add_gain(mut self, name: &str, level: f64) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Gain { level });
        (self, id)
    }

    pub fn add_mixer(mut self, name: &str, channels: usize) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Mixer { channel_count: channels });
        (self, id)
    }

    pub fn add_output(mut self, name: &str, format: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Output { format: format.into() });
        (self, id)
    }

    pub fn add_delay(mut self, name: &str, samples: usize) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Delay { samples });
        (self, id)
    }

    pub fn add_pan(mut self, name: &str, position: f64) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Pan { position });
        (self, id)
    }

    pub fn add_constant(mut self, name: &str, value: f64) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Constant(value));
        (self, id)
    }

    pub fn add_envelope(mut self, name: &str, a: f64, d: f64, s: f64, r: f64) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Envelope { attack: a, decay: d, sustain: s, release: r });
        (self, id)
    }

    pub fn add_compressor(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Compressor);
        (self, id)
    }

    pub fn add_limiter(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Limiter);
        (self, id)
    }

    pub fn add_splitter(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Splitter);
        (self, id)
    }

    pub fn add_merger(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Merger);
        (self, id)
    }

    pub fn add_noise(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::NoiseGenerator);
        (self, id)
    }

    pub fn add_pitch_shifter(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::PitchShifter);
        (self, id)
    }

    pub fn add_time_stretch(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::TimeStretch);
        (self, id)
    }

    pub fn add_modulator(mut self, name: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::Modulator);
        (self, id)
    }

    pub fn add_data_input(mut self, name: &str, schema: &str) -> (Self, NodeId) {
        let id = self.graph.add_node(name, NodeType::DataInput { schema: schema.into() });
        (self, id)
    }

    /// Connect two nodes by port name.
    pub fn connect(mut self, src: NodeId, src_port: &str, dst: NodeId, dst_port: &str) -> Self {
        self.graph.add_edge_by_name(src, src_port, dst, dst_port)
            .expect("GraphBuilder::connect failed");
        self
    }

    /// Finalize and return the graph (topological sort included).
    pub fn build(mut self) -> IrResult<AudioGraph> {
        self.graph.topological_sort()?;
        Ok(self.graph)
    }

    /// Finalize without sorting.
    pub fn build_unsorted(self) -> AudioGraph {
        self.graph
    }

    /// Borrow the graph being built (useful for port ID lookups).
    pub fn graph(&self) -> &AudioGraph { &self.graph }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Waveform, FilterType};

    fn simple_graph() -> AudioGraph {
        let mut g = AudioGraph::new(48000.0, 256);
        let osc = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let gain = g.add_node("gain", NodeType::Gain { level: 0.5 });
        let out = g.add_node("out", NodeType::Output { format: "stereo".into() });
        g.add_edge_by_name(osc, "out", gain, "in").unwrap();
        g.add_edge_by_name(gain, "out", out, "in").unwrap();
        g
    }

    #[test]
    fn test_add_node() {
        let mut g = AudioGraph::default();
        let id = g.add_node("osc1", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        assert_eq!(g.node_count(), 1);
        assert!(g.node(id).is_some());
        assert_eq!(g.node(id).unwrap().name, "osc1");
    }

    #[test]
    fn test_remove_node() {
        let mut g = simple_graph();
        let ids = g.node_ids();
        assert_eq!(g.node_count(), 3);
        g.remove_node(ids[1]).unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 0); // both edges touched the gain node
    }

    #[test]
    fn test_add_edge() {
        let mut g = AudioGraph::default();
        let osc = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Saw, frequency: 220.0 });
        let gain = g.add_node("gain", NodeType::Gain { level: 1.0 });
        let eid = g.add_edge_by_name(osc, "out", gain, "in").unwrap();
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.edges[0].id, eid);
    }

    #[test]
    fn test_remove_edge() {
        let mut g = simple_graph();
        let eid = g.edges[0].id;
        g.remove_edge(eid).unwrap();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut g = simple_graph();
        g.topological_sort().unwrap();
        assert_eq!(g.topological_order.len(), 3);
        // osc must come before gain, gain before out.
        let pos = |id: NodeId| g.topological_order.iter().position(|&x| x == id).unwrap();
        let ids = g.node_ids();
        assert!(pos(ids[0]) < pos(ids[1]));
        assert!(pos(ids[1]) < pos(ids[2]));
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = AudioGraph::default();
        let a = g.add_node("a", NodeType::Gain { level: 1.0 });
        let b = g.add_node("b", NodeType::Gain { level: 1.0 });
        // a→b via audio ports
        g.add_edge_by_name(a, "out", b, "in").unwrap();
        // b→a  creates a cycle
        g.add_edge_by_name(b, "out", a, "in").unwrap();
        assert!(g.has_cycle());
        assert!(g.topological_sort().is_err());
    }

    #[test]
    fn test_validate_unconnected_input() {
        let mut g = AudioGraph::default();
        // Filter has required "in" port.
        let _f = g.add_node("filt", NodeType::Filter { filter_type: FilterType::LowPass, cutoff: 1000.0, q: 1.0 });
        let result = g.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_subgraph_extraction() {
        let mut g = simple_graph();
        g.topological_sort().unwrap();
        let out_id = g.output_nodes()[0];
        let sub = g.extract_subgraph(&[out_id]).unwrap();
        assert_eq!(sub.node_count(), 3); // entire chain feeds into output
    }

    #[test]
    fn test_graph_merge() {
        let mut g1 = AudioGraph::default();
        g1.add_node("a", NodeType::Constant(1.0));

        let mut g2 = AudioGraph::default();
        g2.add_node("b", NodeType::Constant(2.0));

        let map = g1.merge(&g2);
        assert_eq!(g1.node_count(), 2);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_graph_builder() {
        let (b, osc) = GraphBuilder::with_defaults()
            .add_oscillator("osc", Waveform::Sine, 440.0);
        let (b, gain) = b.add_gain("g", 0.5);
        let (b, out) = b.add_output("out", "stereo");
        let g = b
            .connect(osc, "out", gain, "in")
            .connect(gain, "out", out, "in")
            .build()
            .unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.topological_order.len(), 3);
    }

    #[test]
    fn test_successors_predecessors() {
        let mut g = simple_graph();
        let ids = g.node_ids();
        assert_eq!(g.successors(ids[0]).len(), 1);
        assert_eq!(g.predecessors(ids[1]).len(), 1);
        assert!(g.predecessors(ids[0]).is_empty());
        assert!(g.successors(ids[2]).is_empty());
    }

    #[test]
    fn test_reachable_forward() {
        let g = simple_graph();
        let ids = g.node_ids();
        let reach = g.reachable_forward(ids[0]);
        assert_eq!(reach.len(), 3);
    }

    #[test]
    fn test_critical_path_wcet() {
        let mut g = simple_graph();
        g.topological_sort().unwrap();
        let wcet = g.critical_path_wcet();
        assert!(wcet > 0.0);
    }

    #[test]
    fn test_connected_component() {
        let mut g = AudioGraph::default();
        let a = g.add_node("a", NodeType::Constant(1.0));
        let b = g.add_node("b", NodeType::Constant(2.0));
        let _ = g.add_node("c", NodeType::Constant(3.0)); // isolated
        // a → b (using control ports)
        let sp = g.node(a).unwrap().first_output().unwrap().id;
        // b is Constant so it has an output but no input - connect a.out to nowhere useful.
        // Just test that a is its own component since no edge.
        let comp = g.connected_component(a).unwrap();
        assert!(comp.contains(&a));
        // c is not in a's component since there are no edges.
        let _ = b; // suppress warning
    }

    #[test]
    fn test_port_type_mismatch() {
        let mut g = AudioGraph::default();
        let osc = g.add_node("osc", NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 });
        let env = g.add_node("env", NodeType::Envelope { attack: 0.01, decay: 0.1, sustain: 0.7, release: 0.3 });
        // osc.out is Audio, env.gate is Trigger → mismatch
        let result = g.add_edge_by_name(osc, "out", env, "gate");
        assert!(result.is_err());
    }

    #[test]
    fn test_node_type_labels() {
        assert_eq!(NodeType::Oscillator { waveform: Waveform::Sine, frequency: 440.0 }.label(), "Oscillator");
        assert_eq!(NodeType::Compressor.label(), "Compressor");
        assert_eq!(NodeType::Constant(1.0).label(), "Constant");
    }

    #[test]
    fn test_output_and_source_nodes() {
        let g = simple_graph();
        assert_eq!(g.output_nodes().len(), 1);
        assert_eq!(g.source_nodes().len(), 1);
    }
}
