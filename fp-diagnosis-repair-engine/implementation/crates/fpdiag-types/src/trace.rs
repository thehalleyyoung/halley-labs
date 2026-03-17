//! Execution trace events and metadata.
//!
//! Defines the event types emitted by the shadow instrumentation layer
//! during a traced execution.  These events are consumed by the EAG
//! builder to construct the Error Amplification Graph.

use crate::expression::{FpOp, NodeId};
use crate::precision::Precision;
use crate::source::SourceSpan;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// ─── TraceEvent ─────────────────────────────────────────────────────────────

/// A single event in the execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceEvent {
    /// A primitive floating-point operation was executed.
    Operation {
        /// Unique sequential id for this event.
        seq: u64,
        /// The operation performed.
        op: FpOp,
        /// Input values (f64).
        inputs: Vec<f64>,
        /// Output value (f64).
        output: f64,
        /// Shadow (high-precision) output value.
        shadow_output: f64,
        /// Precision of the computation.
        precision: Precision,
        /// Source location, if known.
        source: Option<SourceSpan>,
        /// Expression-tree node id if available.
        expr_node: Option<NodeId>,
    },
    /// A black-box library call.
    LibraryCall {
        /// Unique sequential id.
        seq: u64,
        /// Fully qualified function name (e.g., "scipy.linalg.expm").
        function: String,
        /// Input error magnitude (max over elements).
        input_error: f64,
        /// Output error magnitude (max over elements).
        output_error: f64,
        /// Measured error amplification factor.
        amplification: f64,
        /// Source location.
        source: Option<SourceSpan>,
    },
    /// Start of a traced region / function.
    RegionEnter {
        seq: u64,
        name: String,
        source: Option<SourceSpan>,
    },
    /// End of a traced region / function.
    RegionExit { seq: u64, name: String },
    /// A metadata annotation (e.g., array shape, dtype).
    Annotation {
        seq: u64,
        key: String,
        value: String,
    },
}

impl TraceEvent {
    /// Sequence number of this event.
    pub fn seq(&self) -> u64 {
        match self {
            Self::Operation { seq, .. }
            | Self::LibraryCall { seq, .. }
            | Self::RegionEnter { seq, .. }
            | Self::RegionExit { seq, .. }
            | Self::Annotation { seq, .. } => *seq,
        }
    }

    /// Source location, if available.
    pub fn source(&self) -> Option<&SourceSpan> {
        match self {
            Self::Operation { source, .. }
            | Self::LibraryCall { source, .. }
            | Self::RegionEnter { source, .. } => source.as_ref(),
            _ => None,
        }
    }

    /// Whether this is an operation event (Tier 1).
    pub fn is_operation(&self) -> bool {
        matches!(self, Self::Operation { .. })
    }

    /// Whether this is a library call (Tier 2 / black-box).
    pub fn is_library_call(&self) -> bool {
        matches!(self, Self::LibraryCall { .. })
    }

    /// Local error at this event, if applicable.
    pub fn local_error(&self) -> Option<f64> {
        match self {
            Self::Operation {
                output,
                shadow_output,
                ..
            } => Some((output - shadow_output).abs()),
            Self::LibraryCall { output_error, .. } => Some(*output_error),
            _ => None,
        }
    }
}

impl fmt::Display for TraceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operation {
                seq,
                op,
                output,
                shadow_output,
                ..
            } => {
                write!(
                    f,
                    "[{}] {} → {:.16e} (shadow: {:.16e}, err: {:.4e})",
                    seq,
                    op,
                    output,
                    shadow_output,
                    (output - shadow_output).abs()
                )
            }
            Self::LibraryCall {
                seq,
                function,
                amplification,
                ..
            } => {
                write!(
                    f,
                    "[{}] call {} (amplification: {:.2}×)",
                    seq, function, amplification
                )
            }
            Self::RegionEnter { seq, name, .. } => {
                write!(f, "[{}] → enter {}", seq, name)
            }
            Self::RegionExit { seq, name } => {
                write!(f, "[{}] ← exit {}", seq, name)
            }
            Self::Annotation { seq, key, value } => {
                write!(f, "[{}] @{} = {}", seq, key, value)
            }
        }
    }
}

// ─── TraceMetadata ──────────────────────────────────────────────────────────

/// Metadata about a complete execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    /// Unique identifier for this trace.
    pub trace_id: Uuid,
    /// Timestamp when tracing started.
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Total number of events.
    pub event_count: u64,
    /// Number of Tier 1 (element-wise) operations.
    pub tier1_ops: u64,
    /// Number of Tier 2 (black-box library) calls.
    pub tier2_calls: u64,
    /// Instrumentation coverage: tier1_ops / (tier1_ops + estimated tier2 flops).
    pub coverage: f64,
    /// Shadow precision used (in bits).
    pub shadow_precision_bits: u32,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Wall-clock duration of the traced execution.
    pub wall_time_ms: u64,
}

impl TraceMetadata {
    /// Create metadata with defaults.
    pub fn new(trace_id: Uuid) -> Self {
        Self {
            trace_id,
            started_at: chrono::Utc::now(),
            event_count: 0,
            tier1_ops: 0,
            tier2_calls: 0,
            coverage: 0.0,
            shadow_precision_bits: 128,
            peak_memory_bytes: 0,
            wall_time_ms: 0,
        }
    }

    /// Update coverage fraction.
    pub fn compute_coverage(&mut self) {
        let total = self.tier1_ops + self.tier2_calls;
        self.coverage = if total > 0 {
            self.tier1_ops as f64 / total as f64
        } else {
            0.0
        };
    }
}

// ─── ExecutionTrace ─────────────────────────────────────────────────────────

/// A complete execution trace: ordered events plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Trace metadata.
    pub metadata: TraceMetadata,
    /// Ordered events.
    pub events: Vec<TraceEvent>,
}

impl ExecutionTrace {
    /// Create a new trace.
    pub fn new() -> Self {
        Self {
            metadata: TraceMetadata::new(Uuid::new_v4()),
            events: Vec::new(),
        }
    }

    /// Push an event and update counters.
    pub fn push(&mut self, event: TraceEvent) {
        match &event {
            TraceEvent::Operation { .. } => self.metadata.tier1_ops += 1,
            TraceEvent::LibraryCall { .. } => self.metadata.tier2_calls += 1,
            _ => {}
        }
        self.metadata.event_count += 1;
        self.events.push(event);
    }

    /// Finalize metadata (coverage, etc.).
    pub fn finalize(&mut self) {
        self.metadata.compute_coverage();
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}
