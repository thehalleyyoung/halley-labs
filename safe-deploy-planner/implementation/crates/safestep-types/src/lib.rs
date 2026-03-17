// SafeStep types crate: shared types, traits, and data structures for the
// verified deployment planner with rollback safety envelopes.

pub mod config;
pub mod constraint;
pub mod envelope;
pub mod error;
pub mod graph;
pub mod identifiers;
pub mod metrics;
pub mod plan;
pub mod service;
pub mod temporal;
pub mod traits;
pub mod version;

// Re-export key types for convenience.
pub use config::SafeStepConfig;
pub use constraint::{Constraint, ConstraintSet, ConstraintStrength, ConstraintStatus};
pub use envelope::{EnvelopeMembership, SafetyEnvelope};
pub use error::{ErrorSeverity, Result, SafeStepError};
pub use graph::{ClusterState, GraphEdge, SafetyPredicate, VersionProductGraph};
pub use identifiers::{
    ConstraintId, EnvelopeId, Id, PlanId, ServiceId, StepId, StateId,
};
pub use plan::{DeploymentPlan, PlanBuilder, PlanCost, PlanStep};
pub use service::{
    ReplicaConfig, ResourceQuantity, ResourceRequirements, ServiceDescriptor,
};
pub use traits::{CostMetric, Encodable, Hashable, Oracle, PlanOptimizer, Verifiable};
pub use version::{Version, VersionIndex, VersionRange, VersionReq, VersionSet};
