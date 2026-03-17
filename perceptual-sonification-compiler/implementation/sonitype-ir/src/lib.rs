//! # SoniType IR
//!
//! Audio graph intermediate representation and optimization passes for the
//! SoniType perceptual sonification compiler. This crate provides:
//!
//! - A directed acyclic graph (DAG) representation of audio processing pipelines
//! - Typed node definitions with per-node parameter structures
//! - Optimization passes including dead-stream elimination, constant folding,
//!   node fusion, buffer reuse, and common subexpression elimination
//! - Psychoacoustic-aware passes for masking optimization and spectral bin packing
//! - Temporal scheduling passes for cognitive-load-aware stream interleaving
//! - Graph analysis utilities (spectral occupancy, dependency, resource estimation)
//! - Graph transformations (splitting, cloning, flattening, normalization)
//! - Validation (type checking, WCET bounds, parameter ranges)
//! - Serialization to JSON and a compact binary format

pub mod graph;
pub mod node;
pub mod passes;
pub mod masking_pass;
pub mod temporal_pass;
pub mod analysis;
pub mod transform;
pub mod validation;
pub mod serialize;

// Re-exports for convenience
pub use graph::{
    AudioGraph, AudioGraphNode, AudioEdge, NodeId, PortId, EdgeId,
    NodeType, Port, PortDirection, PortDataType, BufferType, GraphBuilder,
};
pub use node::{
    OscillatorParams, FilterParams, EnvelopeParams, ModulatorParams,
    CompressorParams, LimiterParams, DelayParams, PitchShiftParams,
    TimeStretchParams, NoiseParams, Waveform, FilterType, CurveType,
    ModulationType, NoiseType, NodeParameters,
};
pub use passes::{Pass, PassManager, PassResult};
pub use masking_pass::{MaskingAwareStreamMerging, SpectralBinPacking, MaskingMarginOptimization};
pub use temporal_pass::{TemporalScheduler, TemporalBinPacking, LatencyAnalysis};
pub use analysis::{GraphAnalyzer, DependencyAnalysis, ResourceAnalysis, SpectralAnalysis};
pub use transform::{GraphTransformer, StreamSplitter, GraphCloner, SubgraphExtractor};
pub use validation::{IrValidator, ValidationReport, WcetValidator, ValidationSeverity, ValidationEntry};
pub use serialize::{GraphSerializer, GraphDeserializer, GraphDiff, SerializeFormat};

/// Crate-level error type.
#[derive(Debug, Clone)]
pub enum IrError {
    /// Graph contains a cycle.
    CycleDetected,
    /// Referenced node does not exist.
    NodeNotFound(NodeId),
    /// Referenced edge does not exist.
    EdgeNotFound(EdgeId),
    /// Port type mismatch on an edge.
    PortTypeMismatch { edge: EdgeId, expected: PortDataType, found: PortDataType },
    /// A required input port is unconnected.
    UnconnectedInput { node: NodeId, port: PortId },
    /// Parameter value out of valid range.
    ParameterOutOfRange { node: NodeId, param: String, value: f64, min: f64, max: f64 },
    /// WCET budget exceeded.
    WcetExceeded { total_wcet_us: f64, budget_us: f64 },
    /// Serialization / deserialization failure.
    SerializationError(String),
    /// Generic validation error.
    ValidationError(String),
    /// Graph is empty (no nodes).
    EmptyGraph,
    /// Duplicate node ID.
    DuplicateNodeId(NodeId),
    /// Duplicate edge.
    DuplicateEdge(EdgeId),
    /// Invalid subgraph specification.
    InvalidSubgraph(String),
}

impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "cycle detected in audio graph"),
            Self::NodeNotFound(id) => write!(f, "node not found: {}", id.0),
            Self::EdgeNotFound(id) => write!(f, "edge not found: {}", id.0),
            Self::PortTypeMismatch { edge, expected, found } =>
                write!(f, "port type mismatch on edge {}: expected {:?}, found {:?}", edge.0, expected, found),
            Self::UnconnectedInput { node, port } =>
                write!(f, "unconnected required input: node={}, port={}", node.0, port.0),
            Self::ParameterOutOfRange { node, param, value, min, max } =>
                write!(f, "parameter out of range: node={}, param={}, value={}, range=[{}, {}]", node.0, param, value, min, max),
            Self::WcetExceeded { total_wcet_us, budget_us } =>
                write!(f, "WCET exceeded: total={:.1}us, budget={:.1}us", total_wcet_us, budget_us),
            Self::SerializationError(msg) => write!(f, "serialization error: {}", msg),
            Self::ValidationError(msg) => write!(f, "validation error: {}", msg),
            Self::EmptyGraph => write!(f, "graph is empty"),
            Self::DuplicateNodeId(id) => write!(f, "duplicate node ID: {}", id.0),
            Self::DuplicateEdge(id) => write!(f, "duplicate edge: {}", id.0),
            Self::InvalidSubgraph(msg) => write!(f, "invalid subgraph: {}", msg),
        }
    }
}

impl std::error::Error for IrError {}

pub type IrResult<T> = Result<T, IrError>;
