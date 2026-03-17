//! Shared types and data structures for the Choreo XR interaction compiler.
//!
//! This crate provides the foundational types used across all phases of the
//! Choreo compilation pipeline: parsing, spatial reasoning, temporal logic,
//! automaton construction, type checking, verification, and code generation.

pub mod geometry;
pub mod spatial;
pub mod temporal;
pub mod event;
pub mod automaton;
pub mod types;
pub mod error;
pub mod config;
pub mod identifiers;

pub use geometry::{
    Point3, Vector3, Quaternion, Transform3D, AABB, OBB, Sphere, Capsule,
    ConvexHull, Plane, Ray, LineSegment, ConvexPolytope, BoundingVolume,
    SpatialRelation,
};
pub use spatial::{
    SpatialPredicate, SpatialPredicateId, PredicateValuation,
    SpatialConstraint, SpatialRegion, EntityId, RegionId, ZoneId,
    SceneEntity, SceneConfiguration, SpatialPredicateEvaluator,
};
pub use temporal::{
    TimePoint, Duration, TimeInterval, AllenRelation, TemporalConstraint,
    MTLFormula, BoundedMTL, TimingBound, TemporalPredicateId,
};
pub use event::{
    EventKind, Event, EventId, EventTrace, EventPattern, EventStream,
    ActionType, GestureType, HandSide,
};
pub use automaton::{
    StateId, TransitionId, Guard, Action, Transition, State, AutomatonDef,
    ProductState, AutomatonKind, TimerId, VarId, Value,
};
pub use types::{
    SpatialType, RegionType, TemporalType, InteractionType,
    ChoreographyType, TypeEnv, TypeConstraint, TypeId, Subtyping,
};
pub use error::{ChoreoError, Diagnostic, Severity, Span, DiagnosticBag};
pub use config::{
    ChoreoConfig, VerificationConfig, CompilationConfig, SimulationConfig,
    SpatialConfig, OutputConfig,
};
pub use identifiers::{
    Identifier, SymbolTable, QualifiedName, Scope, ScopeStack,
};
