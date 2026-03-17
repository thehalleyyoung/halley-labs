//! # fpdiag-types
//!
//! Core types for the **Penumbra** floating-point diagnosis and repair engine.
//!
//! This crate is the foundational dependency for every other crate in the
//! workspace.  It defines the shared vocabulary—IEEE 754 representations,
//! expression trees, precision descriptors, rounding modes, error metrics,
//! the Error Amplification Graph (EAG), execution traces, diagnosis
//! categories, repair strategies, and configuration—so that all downstream
//! crates can interoperate without circular dependencies.
//!
//! # Module overview
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`ieee754`] | Bit-level IEEE 754 decomposition and classification |
//! | [`precision`] | Precision descriptors and cost modelling |
//! | [`rounding`] | Rounding modes including stochastic rounding |
//! | [`expression`] | Expression trees with arena storage |
//! | [`error_bounds`] | Error metrics, intervals, and summary statistics |
//! | [`eag`] | Error Amplification Graph types |
//! | [`trace`] | Execution trace events and metadata |
//! | [`diagnosis`] | Diagnosis categories and reports |
//! | [`repair`] | Repair strategies, candidates, and certifications |
//! | [`source`] | Source location / span types |
//! | [`config`] | Engine configuration structs |
//! | [`ulp`] | ULP computation utilities |
//! | [`double_double`] | Double-double (dd) extended-precision arithmetic |
//! | [`fpclass`] | Extended floating-point classification helpers |
//! | [`fpbench`] | FPBench / FPCore format parsing and emission |

pub mod config;
pub mod diagnosis;
pub mod double_double;
pub mod eag;
pub mod error_bounds;
pub mod expression;
pub mod fpbench;
pub mod fpclass;
pub mod ieee754;
pub mod precision;
pub mod repair;
pub mod rounding;
pub mod source;
pub mod trace;
pub mod ulp;

// ── Convenience re-exports ──────────────────────────────────────────────────

pub use config::PenumbraConfig;
pub use diagnosis::{Diagnosis, DiagnosisCategory, DiagnosisReport, DiagnosisSeverity};
pub use double_double::DoubleDouble;
pub use eag::{EagEdge, EagEdgeId, EagNode, EagNodeId, ErrorAmplificationGraph};
pub use error_bounds::{ErrorBound, ErrorInterval, ErrorMetric, ErrorSummary};
pub use expression::{ExprBuilder, ExprNode, ExprTree, FpOp, NodeId};
pub use fpclass::FpClassification;
pub use ieee754::{FpClass, Ieee754Format, Ieee754Value};
pub use precision::{Precision, PrecisionCost, PrecisionRequirement};
pub use repair::{RepairCandidate, RepairCertification, RepairResult, RepairStrategy};
pub use rounding::{RoundingError, RoundingMode};
pub use source::SourceSpan;
pub use trace::{ExecutionTrace, TraceEvent, TraceMetadata};
pub use ulp::{ulp_distance_f32, ulp_distance_f64, ulp_f32, ulp_f64};
