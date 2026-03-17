//! # sonitype-codegen
//!
//! Code generator and static WCET (Worst-Case Execution Time) analyzer for
//! bounded-latency audio rendering in the SoniType perceptual sonification compiler.
//!
//! This crate transforms an optimized `AudioGraph` from `sonitype-ir` into an
//! executable audio renderer while statically guaranteeing that the renderer
//! meets real-time deadlines (Theorem 7: Schedulability).
//!
//! ## Architecture
//!
//! ```text
//! AudioGraph (sonitype-ir)
//!     │
//!     ▼
//!  ┌──────────┐
//!  │ Lowering  │  Resolve parameters, expand compound nodes, insert smoothing
//!  └────┬─────┘
//!       ▼
//!  ┌──────────────┐
//!  │ Optimization  │  Inline expansion, loop fusion, SIMD hints, constant prop
//!  └────┬─────────┘
//!       ▼
//!  ┌──────────────┐
//!  │  Scheduling   │  Topological order, parallel grouping, buffer reuse
//!  └────┬─────────┘
//!       ▼
//!  ┌──────────────┐
//!  │   Codegen     │  Per-node code generation, buffer allocation
//!  └────┬─────────┘
//!       ▼
//!  ┌──────────────┐
//!  │   Emitter     │  Rust source emission, WAV writer, inline renderer
//!  └────┬─────────┘
//!       ▼
//!  ┌──────────────────┐
//!  │   Verification    │  WCET bounds, soundness (Theorem 4), benchmarks
//!  └──────────────────┘
//! ```

pub mod codegen;
pub mod emitter;
pub mod lowering;
pub mod optimization;
pub mod scheduler;
pub mod verification;
pub mod wcet;

// Re-export primary interfaces
pub use codegen::{BufferAllocationPlan, CodeGenerator, GeneratedRenderer, NodeCodegen};
pub use emitter::{EmittedCode, InlineRenderer, RustEmitter, WavEmitter};
pub use lowering::{IrLowerer, ParameterSmoothing, SampleRateConversion};
pub use optimization::{
    ConstantPropagation, DeadCodeElimination, InlineExpansion, LoopFusion, SimdHints,
    StrengthReduction,
};
pub use scheduler::{ExecutionScheduler, Schedule, ScheduleOptimizer};
pub use verification::{BenchmarkHarness, RendererVerifier, SoundnessChecker};
pub use wcet::{CostModel, SchedulabilityChecker, WcetAnalyzer, WcetBudget};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Common types used across modules (local equivalents for types not yet
// available in dependency crates)
// ---------------------------------------------------------------------------

/// Re-export sonitype_ir graph types under a convenient alias.
pub mod ir {
    pub use sonitype_ir::graph::*;
}

/// Unique identifier for a processing node within the codegen graph.
/// Wraps sonitype_ir::graph::NodeId for convenience.
pub type CgNodeId = sonitype_ir::graph::NodeId;

/// Unique identifier for an edge in the codegen graph.
pub type CgEdgeId = sonitype_ir::graph::EdgeId;

/// Hardware architecture target for cost modelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    X86_64,
    Aarch64,
    AppleSilicon,
    GenericArm,
    Wasm32,
}

impl Default for Architecture {
    fn default() -> Self {
        Architecture::X86_64
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Architecture::X86_64 => write!(f, "x86-64"),
            Architecture::Aarch64 => write!(f, "aarch64"),
            Architecture::AppleSilicon => write!(f, "Apple Silicon"),
            Architecture::GenericArm => write!(f, "ARM (generic)"),
            Architecture::Wasm32 => write!(f, "wasm32"),
        }
    }
}

/// Codegen-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodegenConfig {
    pub sample_rate: f64,
    pub block_size: usize,
    pub channel_count: usize,
    pub cpu_frequency_hz: f64,
    pub architecture: Architecture,
    /// Target safety margin (WCET headroom multiplier, e.g. 50-100x).
    pub target_safety_margin: f64,
    /// Maximum allowed utilization (0.0–1.0). Typically 0.5–0.8.
    pub max_utilization: f64,
    /// Whether to enable SIMD optimizations.
    pub enable_simd: bool,
    /// Whether to enable loop unrolling.
    pub enable_loop_unroll: bool,
    /// Inline threshold in estimated cycles.
    pub inline_threshold_cycles: f64,
    /// Number of processing cores available.
    pub num_cores: usize,
}

impl Default for CodegenConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000.0,
            block_size: 256,
            channel_count: 2,
            cpu_frequency_hz: 3_000_000_000.0, // 3 GHz
            architecture: Architecture::X86_64,
            target_safety_margin: 50.0,
            max_utilization: 0.7,
            enable_simd: true,
            enable_loop_unroll: true,
            inline_threshold_cycles: 30.0,
            num_cores: 1,
        }
    }
}

impl CodegenConfig {
    /// Budget in seconds for one audio buffer period.
    pub fn buffer_period_seconds(&self) -> f64 {
        self.block_size as f64 / self.sample_rate
    }

    /// Budget in CPU cycles for one audio buffer period.
    pub fn buffer_period_cycles(&self) -> f64 {
        self.buffer_period_seconds() * self.cpu_frequency_hz
    }

    /// Budget in microseconds for one audio buffer period.
    pub fn buffer_period_us(&self) -> f64 {
        self.buffer_period_seconds() * 1_000_000.0
    }
}

/// Errors from the codegen pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodegenError {
    WcetExceeded {
        total_wcet_cycles: f64,
        budget_cycles: f64,
    },
    SchedulabilityViolation {
        utilization: f64,
        max_utilization: f64,
    },
    BufferAllocationFailed {
        reason: String,
    },
    NodeNotFound {
        node_id: u64,
    },
    InvalidGraph {
        reason: String,
    },
    EmissionError {
        reason: String,
    },
    VerificationFailed {
        reason: String,
    },
    LoweringError {
        reason: String,
    },
    OptimizationError {
        reason: String,
    },
    InternalError {
        reason: String,
    },
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::WcetExceeded {
                total_wcet_cycles,
                budget_cycles,
            } => {
                write!(
                    f,
                    "WCET exceeded: {:.0} cycles > {:.0} budget",
                    total_wcet_cycles, budget_cycles
                )
            }
            CodegenError::SchedulabilityViolation {
                utilization,
                max_utilization,
            } => {
                write!(
                    f,
                    "Schedulability violation: {:.1}% > {:.1}% max",
                    utilization * 100.0,
                    max_utilization * 100.0
                )
            }
            CodegenError::BufferAllocationFailed { reason } => {
                write!(f, "Buffer allocation failed: {}", reason)
            }
            CodegenError::NodeNotFound { node_id } => {
                write!(f, "Node not found: {}", node_id)
            }
            CodegenError::InvalidGraph { reason } => {
                write!(f, "Invalid graph: {}", reason)
            }
            CodegenError::EmissionError { reason } => {
                write!(f, "Emission error: {}", reason)
            }
            CodegenError::VerificationFailed { reason } => {
                write!(f, "Verification failed: {}", reason)
            }
            CodegenError::LoweringError { reason } => {
                write!(f, "Lowering error: {}", reason)
            }
            CodegenError::OptimizationError { reason } => {
                write!(f, "Optimization error: {}", reason)
            }
            CodegenError::InternalError { reason } => {
                write!(f, "Internal error: {}", reason)
            }
        }
    }
}

impl std::error::Error for CodegenError {}

pub type CodegenResult<T> = Result<T, CodegenError>;

/// A simplified node-type tag used within codegen for cost modelling and
/// code generation when the full `sonitype_ir::graph::NodeType` with its
/// associated parameters is not needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    Oscillator,
    Filter,
    Envelope,
    Mixer,
    Gain,
    Pan,
    Delay,
    Modulator,
    Compressor,
    Limiter,
    DataInput,
    Output,
    Splitter,
    Merger,
    Constant,
    NoiseGenerator,
    PitchShifter,
    TimeStretch,
}

impl NodeKind {
    /// Classify a full `sonitype_ir::graph::NodeType` into a `NodeKind`.
    pub fn from_ir_node_type(nt: &sonitype_ir::graph::NodeType) -> Self {
        use sonitype_ir::graph::NodeType;
        match nt {
            NodeType::Oscillator { .. } => NodeKind::Oscillator,
            NodeType::Filter { .. } => NodeKind::Filter,
            NodeType::Envelope { .. } => NodeKind::Envelope,
            NodeType::Mixer { .. } => NodeKind::Mixer,
            NodeType::Gain { .. } => NodeKind::Gain,
            NodeType::Pan { .. } => NodeKind::Pan,
            NodeType::Delay { .. } => NodeKind::Delay,
            NodeType::Modulator => NodeKind::Modulator,
            NodeType::Compressor => NodeKind::Compressor,
            NodeType::Limiter => NodeKind::Limiter,
            NodeType::DataInput { .. } => NodeKind::DataInput,
            NodeType::Output { .. } => NodeKind::Output,
            NodeType::Splitter => NodeKind::Splitter,
            NodeType::Merger => NodeKind::Merger,
            NodeType::Constant(_) => NodeKind::Constant,
            NodeType::NoiseGenerator => NodeKind::NoiseGenerator,
            NodeType::PitchShifter => NodeKind::PitchShifter,
            NodeType::TimeStretch => NodeKind::TimeStretch,
        }
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Per-node metadata carried through the codegen pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: u64,
    pub name: String,
    pub kind: NodeKind,
    pub sample_rate: f64,
    pub block_size: usize,
    pub wcet_cycles: f64,
    pub parameters: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

/// Represents a directed edge between two processing nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInfo {
    pub id: u64,
    pub source_node: u64,
    pub source_port: u64,
    pub dest_node: u64,
    pub dest_port: u64,
    pub buffer_type: BufferKind,
}

/// Buffer kind in the codegen pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BufferKind {
    Audio,
    Control,
    Trigger,
}

/// A simplified audio graph representation used internally by codegen.
/// Derived from `sonitype_ir::graph::AudioGraph`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CgGraph {
    pub nodes: Vec<NodeInfo>,
    pub edges: Vec<EdgeInfo>,
    pub topological_order: Vec<u64>,
    pub sample_rate: f64,
    pub block_size: usize,
}

impl CgGraph {
    /// Build a `CgGraph` from an `sonitype_ir::graph::AudioGraph`.
    pub fn from_ir(graph: &sonitype_ir::graph::AudioGraph) -> Self {
        let nodes: Vec<NodeInfo> = graph
            .nodes
            .iter()
            .map(|n| {
                let kind = NodeKind::from_ir_node_type(&n.node_type);
                let mut parameters = HashMap::new();
                match &n.node_type {
                    sonitype_ir::graph::NodeType::Oscillator { frequency, .. } => {
                        parameters.insert("frequency".into(), *frequency);
                    }
                    sonitype_ir::graph::NodeType::Filter { cutoff, q, .. } => {
                        parameters.insert("cutoff".into(), *cutoff);
                        parameters.insert("q".into(), *q);
                    }
                    sonitype_ir::graph::NodeType::Envelope {
                        attack,
                        decay,
                        sustain,
                        release,
                    } => {
                        parameters.insert("attack".into(), *attack);
                        parameters.insert("decay".into(), *decay);
                        parameters.insert("sustain".into(), *sustain);
                        parameters.insert("release".into(), *release);
                    }
                    sonitype_ir::graph::NodeType::Mixer { channel_count } => {
                        parameters.insert("channel_count".into(), *channel_count as f64);
                    }
                    sonitype_ir::graph::NodeType::Gain { level } => {
                        parameters.insert("level".into(), *level);
                    }
                    sonitype_ir::graph::NodeType::Pan { position } => {
                        parameters.insert("position".into(), *position);
                    }
                    sonitype_ir::graph::NodeType::Delay { samples } => {
                        parameters.insert("samples".into(), *samples as f64);
                    }
                    sonitype_ir::graph::NodeType::Constant(v) => {
                        parameters.insert("value".into(), *v);
                    }
                    _ => {}
                }
                NodeInfo {
                    id: n.id.0,
                    name: n.name.clone(),
                    kind,
                    sample_rate: n.sample_rate,
                    block_size: n.block_size,
                    wcet_cycles: 0.0, // computed later by WcetAnalyzer
                    parameters,
                    metadata: n.metadata.clone(),
                }
            })
            .collect();

        let edges: Vec<EdgeInfo> = graph
            .edges
            .iter()
            .map(|e| {
                let buffer_type = match e.buffer_type {
                    sonitype_ir::graph::BufferType::AudioBuffer => BufferKind::Audio,
                    sonitype_ir::graph::BufferType::ControlBuffer => BufferKind::Control,
                    sonitype_ir::graph::BufferType::TriggerFlag => BufferKind::Trigger,
                };
                EdgeInfo {
                    id: e.id.0,
                    source_node: e.source_node.0,
                    source_port: e.source_port.0,
                    dest_node: e.dest_node.0,
                    dest_port: e.dest_port.0,
                    buffer_type,
                }
            })
            .collect();

        let topological_order = graph.topological_order.iter().map(|n| n.0).collect();

        CgGraph {
            nodes,
            edges,
            topological_order,
            sample_rate: graph.sample_rate,
            block_size: graph.block_size,
        }
    }

    /// Get a node by id.
    pub fn node(&self, id: u64) -> Option<&NodeInfo> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get a mutable reference to a node by id.
    pub fn node_mut(&mut self, id: u64) -> Option<&mut NodeInfo> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Return the IDs of immediate predecessors (nodes that feed into `node_id`).
    pub fn predecessors(&self, node_id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|e| e.dest_node == node_id)
            .map(|e| e.source_node)
            .collect()
    }

    /// Return the IDs of immediate successors (nodes that `node_id` feeds into).
    pub fn successors(&self, node_id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|e| e.source_node == node_id)
            .map(|e| e.dest_node)
            .collect()
    }

    /// Return edges originating from `node_id`.
    pub fn outgoing_edges(&self, node_id: u64) -> Vec<&EdgeInfo> {
        self.edges
            .iter()
            .filter(|e| e.source_node == node_id)
            .collect()
    }

    /// Return edges arriving at `node_id`.
    pub fn incoming_edges(&self, node_id: u64) -> Vec<&EdgeInfo> {
        self.edges
            .iter()
            .filter(|e| e.dest_node == node_id)
            .collect()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return all paths from source nodes (no predecessors) to sink nodes.
    pub fn all_paths(&self) -> Vec<Vec<u64>> {
        let sources: Vec<u64> = self
            .nodes
            .iter()
            .filter(|n| self.predecessors(n.id).is_empty())
            .map(|n| n.id)
            .collect();

        let mut result = Vec::new();
        for src in &sources {
            let mut stack: Vec<(u64, Vec<u64>)> = vec![(*src, vec![*src])];
            while let Some((current, path)) = stack.pop() {
                let succs = self.successors(current);
                if succs.is_empty() {
                    result.push(path);
                } else {
                    for s in succs {
                        let mut new_path = path.clone();
                        new_path.push(s);
                        stack.push((s, new_path));
                    }
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Convenience builder for tests
// ---------------------------------------------------------------------------

/// Helper for constructing `CgGraph` instances in tests.
pub struct CgGraphBuilder {
    graph: CgGraph,
    next_id: u64,
    next_edge_id: u64,
}

impl CgGraphBuilder {
    pub fn new(sample_rate: f64, block_size: usize) -> Self {
        Self {
            graph: CgGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
                topological_order: Vec::new(),
                sample_rate,
                block_size,
            },
            next_id: 0,
            next_edge_id: 0,
        }
    }

    pub fn add_node(&mut self, name: &str, kind: NodeKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.graph.nodes.push(NodeInfo {
            id,
            name: name.to_string(),
            kind,
            sample_rate: self.graph.sample_rate,
            block_size: self.graph.block_size,
            wcet_cycles: 0.0,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        });
        id
    }

    pub fn add_node_with_params(
        &mut self,
        name: &str,
        kind: NodeKind,
        params: HashMap<String, f64>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.graph.nodes.push(NodeInfo {
            id,
            name: name.to_string(),
            kind,
            sample_rate: self.graph.sample_rate,
            block_size: self.graph.block_size,
            wcet_cycles: 0.0,
            parameters: params,
            metadata: HashMap::new(),
        });
        id
    }

    pub fn connect(&mut self, src: u64, dst: u64, buffer_kind: BufferKind) {
        let id = self.next_edge_id;
        self.next_edge_id += 1;
        self.graph.edges.push(EdgeInfo {
            id,
            source_node: src,
            source_port: 0,
            dest_node: dst,
            dest_port: 0,
            buffer_type: buffer_kind,
        });
    }

    pub fn build(mut self) -> CgGraph {
        // Compute topological order via Kahn's algorithm
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        for n in &self.graph.nodes {
            in_degree.insert(n.id, 0);
        }
        for e in &self.graph.edges {
            *in_degree.entry(e.dest_node).or_insert(0) += 1;
        }
        let mut queue: Vec<u64> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort();
        let mut order = Vec::new();
        while let Some(n) = queue.pop() {
            order.push(n);
            for e in &self.graph.edges {
                if e.source_node == n {
                    let deg = in_degree.get_mut(&e.dest_node).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(e.dest_node);
                        queue.sort();
                    }
                }
            }
        }
        // Reverse because we pop from end
        order.reverse();
        // Actually re-do: Kahn's with a proper FIFO
        let mut in_degree2: HashMap<u64, usize> = HashMap::new();
        for n in &self.graph.nodes {
            in_degree2.insert(n.id, 0);
        }
        for e in &self.graph.edges {
            *in_degree2.entry(e.dest_node).or_insert(0) += 1;
        }
        let mut queue2: std::collections::VecDeque<u64> = in_degree2
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect::<Vec<_>>()
            .into_iter()
            .collect();
        let mut topo = Vec::new();
        while let Some(n) = queue2.pop_front() {
            topo.push(n);
            let succs: Vec<u64> = self
                .graph
                .edges
                .iter()
                .filter(|e| e.source_node == n)
                .map(|e| e.dest_node)
                .collect();
            for s in succs {
                let deg = in_degree2.get_mut(&s).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue2.push_back(s);
                }
            }
        }
        self.graph.topological_order = topo;
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let filt = b.add_node("filt", NodeKind::Filter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, filt, BufferKind::Audio);
        b.connect(filt, out, BufferKind::Audio);
        b.build()
    }

    #[test]
    fn test_cg_graph_builder_basic() {
        let g = simple_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_cg_graph_topological_order() {
        let g = simple_graph();
        assert_eq!(g.topological_order.len(), 3);
        let osc_pos = g.topological_order.iter().position(|&x| x == 0).unwrap();
        let filt_pos = g.topological_order.iter().position(|&x| x == 1).unwrap();
        let out_pos = g.topological_order.iter().position(|&x| x == 2).unwrap();
        assert!(osc_pos < filt_pos);
        assert!(filt_pos < out_pos);
    }

    #[test]
    fn test_cg_graph_predecessors() {
        let g = simple_graph();
        assert_eq!(g.predecessors(0), vec![]);
        assert_eq!(g.predecessors(1), vec![0]);
        assert_eq!(g.predecessors(2), vec![1]);
    }

    #[test]
    fn test_cg_graph_successors() {
        let g = simple_graph();
        assert_eq!(g.successors(0), vec![1]);
        assert_eq!(g.successors(1), vec![2]);
        assert_eq!(g.successors(2), vec![]);
    }

    #[test]
    fn test_cg_graph_node_lookup() {
        let g = simple_graph();
        let n = g.node(1).unwrap();
        assert_eq!(n.name, "filt");
        assert_eq!(n.kind, NodeKind::Filter);
        assert!(g.node(999).is_none());
    }

    #[test]
    fn test_cg_graph_all_paths() {
        let g = simple_graph();
        let paths = g.all_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_cg_graph_branching_paths() {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let src = b.add_node("src", NodeKind::Oscillator);
        let left = b.add_node("left", NodeKind::Gain);
        let right = b.add_node("right", NodeKind::Pan);
        let sink = b.add_node("sink", NodeKind::Output);
        b.connect(src, left, BufferKind::Audio);
        b.connect(src, right, BufferKind::Audio);
        b.connect(left, sink, BufferKind::Audio);
        b.connect(right, sink, BufferKind::Audio);
        let g = b.build();
        let paths = g.all_paths();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_codegen_config_defaults() {
        let cfg = CodegenConfig::default();
        assert_eq!(cfg.sample_rate, 48000.0);
        assert_eq!(cfg.block_size, 256);
        assert!(cfg.buffer_period_seconds() > 0.0);
        assert!(cfg.buffer_period_cycles() > 0.0);
    }

    #[test]
    fn test_codegen_config_budget() {
        let cfg = CodegenConfig {
            sample_rate: 48000.0,
            block_size: 256,
            cpu_frequency_hz: 3_000_000_000.0,
            ..Default::default()
        };
        let period = 256.0 / 48000.0; // ~5.33ms
        let cycles = period * 3e9; // ~16M cycles
        assert!((cfg.buffer_period_cycles() - cycles).abs() < 1.0);
    }

    #[test]
    fn test_node_kind_display() {
        assert_eq!(format!("{}", NodeKind::Oscillator), "Oscillator");
        assert_eq!(format!("{}", NodeKind::Compressor), "Compressor");
    }

    #[test]
    fn test_codegen_error_display() {
        let err = CodegenError::WcetExceeded {
            total_wcet_cycles: 20_000_000.0,
            budget_cycles: 16_000_000.0,
        };
        let s = format!("{}", err);
        assert!(s.contains("WCET exceeded"));
    }

    #[test]
    fn test_architecture_display() {
        assert_eq!(format!("{}", Architecture::X86_64), "x86-64");
        assert_eq!(format!("{}", Architecture::AppleSilicon), "Apple Silicon");
    }
}
