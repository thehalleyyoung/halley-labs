//! Core types for the ConservationLint simulation conservation auditor.
//!
//! This crate provides shared type definitions, traits, and error types
//! used throughout the conservation analysis pipeline.

pub mod error;
pub mod phase_space;
pub mod conservation_law;
pub mod symmetry;
pub mod numeric;
pub mod provenance;
pub mod diagnostic;
pub mod geometry;
pub mod expression;
pub mod config;
pub mod interval;
pub mod matrix;
pub mod polynomial;
pub mod tag;
pub mod source_loc;
pub mod units;

pub use error::{ConservationError, Result};
pub use phase_space::{PhaseSpace, PhaseSpaceKind, Coordinate, CoordinateKind};
pub use conservation_law::{ConservationLaw, ConservationKind, ConservedQuantity};
pub use symmetry::{SymmetryGroup, SymmetryGenerator, LieAlgebra};
pub use numeric::{Scalar, Tolerance, PrecisionLevel};
pub use provenance::{ProvenanceTag, TagAlgebra, TagId};
pub use diagnostic::{Diagnostic, DiagnosticLevel, DiagnosticReport};
pub use source_loc::SourceLocation;
