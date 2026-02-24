//! LITMUS∞ Axiomatic Memory Model Checker
//!
//! Implements bit-packed execution graph verification for axiomatic memory models
//! including SC, TSO, PSO, ARM, RISC-V, PTX, and WebGPU.

pub mod execution;
pub mod memory_model;
pub mod litmus;
pub mod verifier;
pub mod decomposition;

pub use execution::{
    Event, EventId, ThreadId, Address, Value, OpType, Scope,
    ExecutionGraph, BitMatrix, Relation,
};
pub use memory_model::{
    MemoryModel, RelationDef, RelationType, DerivedRelation,
    Constraint, RelationExpr, BuiltinModel,
};
pub use litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, LitmusOutcome,
};
pub use verifier::{
    Verifier, VerificationResult, VerificationStats,
    CompositionalVerifier,
};
pub use decomposition::{
    TestDecomposer, DecompositionTree, DecompositionNode,
    GluingTheorem, OptimalDecomposition, DecompositionValidator,
};

pub mod webgpu;
pub mod operational;
pub mod axiom;
pub mod relations;
pub mod scope;
pub mod gpu_model;
pub mod x86_model;
pub mod sat_encoder;
pub mod bounded_model;
pub mod power_model;
pub mod compositional;
pub mod exhaustive;
pub mod coherence;
pub mod atomicity;
pub mod thread_model;
pub mod fence_analysis;
pub mod partial_order;
pub mod happens_before;
pub mod write_serialization;
pub mod portability;
