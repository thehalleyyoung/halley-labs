//! # choreo-ec
//!
//! Event Calculus intermediate representation engine with a spatial oracle
//! for the Choreo XR Interaction Compiler.
//!
//! This crate implements a full Discrete Event Calculus (DEC) engine that
//! reasons about fluents, events, and narratives in XR interaction scenarios.
//! It integrates spatial reasoning through an oracle that derives spatial
//! predicates from scene configurations.
//!
//! ## Architecture
//!
//! - **Fluents**: Time-varying properties (boolean, numeric, spatial, timer, state, custom)
//! - **Axioms**: Rules governing how events initiate/terminate fluents
//! - **Spatial Oracle**: Derives spatial fluents from geometric scene data
//! - **EC Engine**: Forward-chaining evaluation with abductive reasoning
//! - **Compiler**: Lowers EC programs to automaton transitions
//! - **Domain**: Pre-built XR interaction domains
//! - **Narrative**: Complete interaction traces with validation
//! - **Trace**: Differential testing and trace comparison

pub mod local_types;
pub mod fluent;
pub mod axioms;
pub mod spatial_oracle;
pub mod engine;
pub mod compiler;
pub mod domain;
pub mod narrative;
pub mod trace;

pub use local_types::*;
pub use fluent::{
    Fluent, FluentId, FluentStore, FluentSnapshot, FluentDelta,
    InitiatedBy, TerminatedBy, FluentHistory,
};
pub use axioms::{
    Axiom, AxiomSet, AxiomCondition, CircumscriptionEngine,
};
pub use spatial_oracle::{
    SpatialOracle, SpatialTransitionEvent, SpatialTrajectory,
};
pub use engine::{
    ECEngine, ECState, ECEngineConfig, Narrative as ECNarrative,
};
pub use compiler::{
    ECCompiler, TransitionGuard, StateInvariant, CompiledTransition,
    AutomatonBlueprint,
};
pub use domain::{XRDomain, DomainBuilder};
pub use narrative::{
    NarrativeDoc, NarrativeBuilder, NarrativeViolation, NarrativeQuery,
};
pub use trace::{
    TraceComparator, TraceComparisonResult, TraceDivergence,
    DifferentialTester,
};
