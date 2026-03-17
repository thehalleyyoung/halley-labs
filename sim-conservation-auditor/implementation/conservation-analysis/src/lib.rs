//! Core conservation-law checking and analysis for physics simulation auditing.
//!
//! This crate provides the central contribution of ConservationLint:
//!
//! - [`ConservationLawChecker`] — verify conservation laws hold across a simulation trace
//! - [`SymplecticIntegratorAuditor`] — check symplectic structure preservation
//! - [`DriftBoundCertifier`] — bound the growth rate of conservation violations
//! - [`AutoLawDetector`] — automatically detect conserved quantities from trace data

pub mod checker;
pub mod symplectic_auditor;
pub mod drift_bound;
pub mod auto_detect;

pub use checker::{ConservationLawChecker, LawCheckResult, TraceAuditReport};
pub use symplectic_auditor::{SymplecticIntegratorAuditor, SymplecticAuditResult};
pub use drift_bound::{DriftBoundCertifier, DriftBound, DriftPattern};
pub use auto_detect::{AutoLawDetector, DetectedInvariant};
