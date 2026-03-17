//! spectral-core: Core spectral methods for MIP instance analysis.
//!
//! Provides hypergraph Laplacian construction, eigenvalue computation,
//! spectral feature extraction, and clustering.
//!
//! # Modules
//!
//! - [`error`] — Crate-specific error types wrapping [`spectral_types::error`].
//! - [`hypergraph`] — Constraint hypergraph construction and Laplacian matrices.
//! - [`eigensolve`] — Eigenvalue/eigenvector solvers for sparse Laplacians.
//! - [`features`] — Spectral feature extraction pipeline.
//! - [`clustering`] — Spectral clustering for constraint partitioning.
//! - [`k_selection`] — Automatic selection of the number of clusters.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use spectral_core::{
//!     build_constraint_hypergraph, build_laplacian,
//!     EigenSolver, FeaturePipeline, spectral_clustering, select_k,
//! };
//! ```

pub mod error;
pub mod hypergraph;
pub mod eigensolve;
pub mod features;
pub mod clustering;
pub mod k_selection;

// Re-exports for ergonomic top-level access.
pub use error::SpectralCoreError;
pub use hypergraph::construction::build_constraint_hypergraph;
pub use hypergraph::laplacian::{build_laplacian, build_normalized_laplacian};
pub use eigensolve::solver::EigenSolver;
pub use features::pipeline::FeaturePipeline;
pub use clustering::spectral::spectral_clustering;
pub use k_selection::select_k;
