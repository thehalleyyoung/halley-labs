//! Optimization solvers for the spectral decomposition oracle.
//!
//! This crate provides LP solvers (simplex, interior point), partition optimization
//! (refinement, greedy construction, evaluation), bundle methods for Lagrangian
//! relaxation, Benders decomposition, Dantzig-Wolfe decomposition, and unified
//! solver interfaces including mock SCIP and GCG adapters.

pub mod error;
pub mod lp;
pub mod partition;
pub mod bundle;
pub mod benders;
pub mod dw;
pub mod solver_interface;

pub use error::{OptError, OptResult};
pub use lp::{LpProblem, LpSolution, SolverStatus, ConstraintType, BasisStatus};
pub use partition::{Partition, AdjacencyInfo};
pub use bundle::{BundleConfig, BundleResult, SubgradientInfo};
pub use benders::{BendersConfig, BendersResult};
pub use dw::{DWConfig, DWResult};
pub use solver_interface::{SolverInterface, SolverType, SolverConfig, HighsAdapter};
