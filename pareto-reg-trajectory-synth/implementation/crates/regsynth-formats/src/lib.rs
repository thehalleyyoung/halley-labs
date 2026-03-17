//! # regsynth-formats
//!
//! Format support for RegSynth, providing import/export capabilities for
//! industry-standard compliance and linked-data formats.
//!
//! ## Supported Formats
//!
//! - **SPDX 2.3** — Software Package Data Exchange for license compliance
//! - **OpenChain ISO/IEC 5230:2020** — Open source compliance conformance
//! - **JSON-LD** — JSON for Linked Data with schema.org/Legislation vocabulary
//! - **OWL/RDF** — Web Ontology Language for regulatory knowledge graphs

pub mod spdx;
pub mod openchain;
pub mod jsonld;
pub mod owl_rdf;

pub use spdx::{SpdxDocument, SpdxPackage, SpdxLicenseExpression, SpdxRelationship};
pub use openchain::{OpenChainConformance, OpenChainReport, ConformanceArea};
pub use jsonld::{JsonLdContext, RegulatoryLinkedData, JsonLdNode};
pub use owl_rdf::{RdfTriple, RdfGraph, OwlOntology, RdfTerm};

use thiserror::Error;

/// Errors produced by format conversion operations.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("SPDX format error: {0}")]
    Spdx(String),

    #[error("OpenChain conformance error: {0}")]
    OpenChain(String),

    #[error("JSON-LD serialization error: {0}")]
    JsonLd(String),

    #[error("OWL/RDF error: {0}")]
    OwlRdf(String),

    #[error("unsupported conversion: {from} -> {to}")]
    UnsupportedConversion { from: String, to: String },

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type alias for format operations.
pub type FormatResult<T> = Result<T, FormatError>;
