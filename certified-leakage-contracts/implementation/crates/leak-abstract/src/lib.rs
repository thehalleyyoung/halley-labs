//! # leak-abstract
//!
//! Abstract interpretation engine for certified leakage contract analysis.
//!
//! This crate provides the foundational abstract interpretation framework used to
//! analyze speculative side-channel leakage in x86-64 cryptographic binaries.
//!
//! ## Architecture
//!
//! The analysis is built on classic abstract interpretation theory:
//!
//! - **Lattice theory** ([`lattice`]): Algebraic structures (join-semilattices,
//!   complete lattices) that form the mathematical backbone of abstract domains.
//! - **Fixpoint computation** ([`fixpoint`]): Iterative solvers (worklist, chaotic
//!   iteration) that compute least fixpoints of monotone systems.
//! - **Widening / Narrowing** ([`widening`], [`narrowing`]): Acceleration operators
//!   that guarantee termination and recover precision.
//! - **Transfer functions** ([`transfer`]): Per-instruction semantic transformers
//!   that lift concrete semantics to abstract domains.
//! - **Dataflow analysis** ([`dataflow`]): The classical monotone-framework
//!   instantiation wiring CFGs, transfer functions, and fixpoint solvers together.
//! - **Reduced product** ([`reduced_product`]): Combines multiple abstract domains
//!   (speculation × cache × quantitative) with reduction for tighter bounds.
//! - **Abstract state** ([`abstract_state`]): Maps program locations to abstract
//!   values (registers, memory, flags).
//! - **Trace semantics** ([`trace`]): Over-approximates observable traces
//!   (cache accesses, branch directions) including speculative executions.

pub mod lattice;
pub mod fixpoint;
pub mod widening;
pub mod narrowing;
pub mod transfer;
pub mod dataflow;
pub mod reduced_product;
pub mod abstract_state;
pub mod trace;

// Re-export core traits
pub use lattice::{Lattice, BoundedLattice, CompleteLattice, FlatLattice};
pub use fixpoint::{FixpointEngine, FixpointConfig, FixpointResult, WorklistAlgorithm};
pub use widening::WideningOperator;
pub use narrowing::NarrowingOperator;
pub use transfer::{TransferFunction, ForwardTransfer, BackwardTransfer};
pub use dataflow::{DataflowAnalysis, DataflowResult};
pub use reduced_product::{ReducedProduct, ThreeWayProduct, ReductionOperator};
pub use abstract_state::{AbstractState, AbstractEnvironment};
pub use trace::{AbstractTrace, TraceElement, Observation};
