//! # regsynth-planner
//!
//! Remediation planner for RegSynth: transforms Pareto-optimal compliance strategies
//! into actionable, scheduled implementation roadmaps with resource constraints.
//!
//! ## Modules
//!
//! - [`roadmap`] — Phased implementation roadmaps with tasks, milestones, and resources
//! - [`scheduler`] — RCPSP-based schedule optimizer with priority-rule heuristics
//! - [`dependency_tracker`] — Dependency graph with topological sort, critical path, cycle detection
//! - [`resource_allocator`] — Multi-period resource allocation and load balancing
//! - [`cost_estimator`] — Three-point cost estimation with historical calibration
//! - [`milestone_tracker`] — Regulatory milestone tracking (EU AI Act deadlines)
//! - [`remediation`] — Remediation suggestion engine with minimum-weight hitting sets
//! - [`report_generator`] — Structured report generation (JSON, plain text)

pub mod roadmap;
pub mod scheduler;
pub mod dependency_tracker;
pub mod resource_allocator;
pub mod cost_estimator;
pub mod milestone_tracker;
pub mod remediation;
pub mod report_generator;

pub use roadmap::{ComplianceRoadmap, RoadmapPhase, RoadmapTask, TaskStatus, GanttEntry};
pub use scheduler::{Scheduler, Schedule, ScheduledTask, SchedulerConfig};
pub use dependency_tracker::{DependencyGraph, DependencyType, DependencyEdge};
pub use resource_allocator::{ResourceAllocator, ResourcePool, ResourceType, ResourceDemand, Allocation, AllocationEntry, ResourceUtilization};
pub use cost_estimator::{CostEstimator, CostEstimate, ThreePointEstimate, CostDatabase, CostSummary};
pub use milestone_tracker::{MilestoneTracker, Milestone, MilestoneStatus, StatusReport};
pub use remediation::{RemediationEngine, RemediationOption, RemediationKind, HittingSetResult};
pub use report_generator::{ReportGenerator, Report, ReportSection, ReportFormat};

use thiserror::Error;

/// Errors that can occur during planning operations.
#[derive(Debug, Error)]
pub enum PlannerError {
    #[error("Circular dependency detected: {0}")]
    CircularDependency(String),

    #[error("Resource infeasibility: {0}")]
    ResourceInfeasible(String),

    #[error("Schedule infeasibility: {0}")]
    ScheduleInfeasible(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Phase not found: {0}")]
    PhaseNotFound(String),

    #[error("Deadline violation: {0}")]
    DeadlineViolation(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type PlannerResult<T> = Result<T, PlannerError>;
