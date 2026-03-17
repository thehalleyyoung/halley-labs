//! Trace storage for Penumbra.
//!
//! Provides storage and retrieval of floating-point execution traces,
//! including individual operation records and metadata.

use serde::{Deserialize, Serialize};
use penumbra_types::{FpOperation, OpId, SourceSpan};

/// A single record from a floating-point execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    /// Unique identifier for this operation.
    pub op_id: OpId,
    /// The floating-point operation performed.
    pub operation: FpOperation,
    /// Computed (float64) input values.
    pub inputs: Vec<f64>,
    /// Computed (float64) output value.
    pub output: f64,
    /// Higher-precision shadow input values.
    pub shadow_inputs: Vec<f64>,
    /// Higher-precision shadow output value.
    pub shadow_output: f64,
    /// Source location of the operation, if available.
    pub source_location: Option<SourceSpan>,
    /// IDs of the operations that produced the inputs.
    pub input_op_ids: Vec<OpId>,
    /// Sequence number in the trace.
    pub sequence: u64,
    /// Optional metadata tags.
    pub tags: Vec<String>,
}

/// A complete execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub records: Vec<TraceRecord>,
    pub metadata: TraceMetadata,
}

/// Metadata about a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub name: String,
    pub timestamp: String,
    pub record_count: usize,
}

impl TraceMetadata {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            timestamp: String::new(),
            record_count: 0,
        }
    }
}
