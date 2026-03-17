//! SafeStep Encoding: SAT/SMT encoding for verified deployment planning.
//!
//! This crate provides the core encoding infrastructure for translating
//! multi-service deployment planning into SAT/SMT formulas, exploiting:
//! - Interval structure in compatibility predicates
//! - BDD-based representation for non-interval constraints
//! - Bounded Model Checking (BMC) for plan finding
//! - Treewidth-based decomposition for FPT solving
//! - Replica symmetry breaking
//! - Resource constraint encoding

pub mod formula;
pub mod interval;
pub mod bdd;
pub mod bmc;
pub mod treewidth;
pub mod symmetry;
pub mod resource;
pub mod prefilter;

pub use formula::{Formula, CnfFormula, FormulaStats, Literal, Clause};
pub use interval::{
    IntervalPredicate, IntervalEncoder, BinaryEncoding, Comparator,
    IntervalCompressor,
};
pub use bdd::{BddNode, Bdd, BddManager, BddBuilder, CompatibilityBdd};
pub use bmc::{
    BmcEncoder, BmcUnrolling, MonotoneEncoder, StepEncoder,
    CompletenessChecker,
};
pub use treewidth::{
    TreeDecomposition, TreewidthComputer, TreeDpSolver, BagProcessor,
    MinDegreeElimination,
};
pub use symmetry::{
    ReplicaSymmetry, SymmetryDetector, SymmetryBreaker, ReplicaEncoder,
};
pub use resource::{
    ResourceEncoder, LinearConstraint, ResourceModel, ResourceSpec,
    CapacityChecker, FeasibilityResult,
};
pub use prefilter::{
    PairwisePrefilter, CompatibilityBitmap, FeasibilityFilter, ArcConsistency,
};
