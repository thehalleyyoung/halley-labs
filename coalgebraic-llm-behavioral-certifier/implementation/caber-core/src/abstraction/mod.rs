//! Abstraction module — CEGAR-based abstraction refinement for coalgebraic LLM models.
//!
//! This module implements the full counterexample-guided abstraction refinement
//! pipeline for coalgebraic behavioral certification of LLMs. It provides:
//!
//! - [`cegar`]: The main CEGAR loop (abstract → verify → refine or certify)
//! - [`alphabet`]: Finite alphabet construction from natural language responses
//! - [`lattice`]: Lattice of (k, n, ε) abstraction triples with ordering and traversal
//! - [`refinement`]: Refinement operators for each abstraction dimension
//! - [`galois`]: Galois connections between concrete and abstract coalgebras

pub mod cegar;
pub mod alphabet;
pub mod lattice;
pub mod refinement;
pub mod galois;

pub use cegar::{
    CegarConfig, CegarLoop, CegarState, CegarPhase, CegarResult,
    CegarTermination, CounterExample, AbstractionVerifier, HypothesisLearner,
};
pub use alphabet::{
    AlphabetAbstraction, AlphabetConfig, ClusterInfo, ClusterStats,
    AlphabetRefinementOp,
};
pub use lattice::{
    AbstractionLattice, AbstractionTriple, LatticeNode, LatticeTraversalStrategy,
    LatticeBudget,
};
pub use refinement::{
    RefinementOperator, RefinementKind, RefinementResult, RefinementHistory,
    RefinementStrategy, RefinementImpact,
};
pub use galois::{
    GaloisConnection, AbstractionMap, ConcretizationMap, PropertyPreservation,
    DegradationBound,
};
