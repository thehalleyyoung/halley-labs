//! XR Affordance Verifier: kinematic body modeling, forward/inverse kinematics,
//! reach envelope computation, workspace analysis, and population sampling.
//!
//! This crate implements the core affordance verification pipeline:
//! - Forward and inverse kinematics for parameterized human arm models
//! - Reach envelope computation W(θ) = {FK(θ,q) | q ∈ J(θ)}
//! - Workspace analysis (dexterous workspace, volume, cross-sections)
//! - Population sampling with stratified and adaptive strategies
//! - Device constraint intersection with arm reach
//! - Self-collision and environment collision detection
//! - Comfort/ergonomic scoring for pose assessment

pub mod forward_kinematics;
pub mod inverse_kinematics;
pub mod reach_envelope;
pub mod workspace;
pub mod body_model;
pub mod population;
pub mod device_constraints;
pub mod collision;
pub mod comfort;

pub use forward_kinematics::ForwardKinematicsSolver;
pub use inverse_kinematics::InverseKinematicsSolver;
pub use reach_envelope::ReachEnvelope;
pub use workspace::WorkspaceAnalyzer;
pub use body_model::BodyModelFactory;
pub use population::PopulationSampler;
pub use device_constraints::DeviceConstraintModel;
pub use collision::CollisionChecker;
pub use comfort::ComfortModel;
