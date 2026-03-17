//! Shared types, traits, and error handling for the Spectral Decomposition Oracle.
//!
//! This crate provides the foundational data structures used by every other crate
//! in the workspace. It defines sparse and dense matrix representations
//! (`CsrMatrix`, `CscMatrix`, `CooMatrix`, `DenseMatrix`, `DenseVector`),
//! hypergraph / graph types, partition descriptors, spectral feature containers,
//! MIP instance models, and a unified error hierarchy.
//!
//! # Re-exports
//!
//! The most commonly used types are re-exported at the crate root for convenience.

pub mod error;
pub mod scalar;
pub mod traits;
pub mod sparse;
pub mod dense;
pub mod graph;
pub mod partition;
pub mod features;
pub mod decomposition;
pub mod mip;
pub mod config;
pub mod stats;

// Re-export the most commonly used types at the crate root.
pub use error::{SpectralError, Result};
pub use scalar::Scalar;
pub use decomposition::DecompositionMethod;
pub use sparse::{CsrMatrix, CscMatrix, CooMatrix};
pub use dense::{DenseMatrix, DenseVector};
pub use graph::{Hypergraph, Graph};
pub use partition::Partition;
pub use features::{SpectralFeatures, SyntacticFeatures, GraphFeatures, CombinedFeatureVector};
pub use mip::MipInstance;
pub use config::GlobalConfig;
