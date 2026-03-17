//! Diagnostic output for the SafeStep verified deployment planner.
//!
//! This crate provides witness generation, safety map visualization,
//! plan annotation, progress tracking, and multiple output formats.

pub mod witness;
pub mod safety_map;
pub mod plan_report;
pub mod format;
pub mod diff;
pub mod metrics_report;
pub mod explanation;
pub mod progress;

pub use witness::{
    WitnessGenerator, StuckWitness, ConstraintDescription, WitnessMinimizer,
    WitnessSeverity, InfeasibilityWitness, Suggestion,
};
pub use safety_map::{
    SafetyMap, StateEntry, EnvelopeMembership, SafetyMapRenderer,
    AsciiRenderer, DotRenderer, MermaidRenderer,
};
pub use plan_report::{
    PlanReport, PlanSummary, StepDetail, SafetyAnalysis, RiskAssessment,
    RiskFactor, Recommendation, RecommendationType, ReportGenerator,
    PnrDescription, TransitionDescription,
};
pub use format::{
    OutputFormatter, JsonFormatter, TextFormatter, MarkdownFormatter,
    HtmlFormatter, TableRenderer, ColumnAlignment,
};
pub use diff::{
    PlanDiff, PlanDiffResult, SafetyChange, DiffFormatter, DiffImpact, VersionDiff,
};
pub use metrics_report::{
    MetricsReport, PerformanceBreakdown, PhaseMetrics, MetricsFormatter,
    BenchmarkComparison,
};
pub use explanation::{
    ExplanationEngine, NaturalLanguageGenerator, ExplanationTemplate,
    CausalChain, VerbosityLevel, AudienceLevel,
};
pub use progress::{
    ProgressTracker, ProgressBar, ProgressCallback, ConsoleProgress,
    PhaseInfo,
};

use serde::{Deserialize, Serialize};
use safestep_types::{ServiceId, ConstraintId, StepId, StateId, EnvelopeId};

/// A semantic version used throughout the planner.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

/// A single deployment step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployStep {
    pub id: StepId,
    pub service: ServiceId,
    pub from_version: Version,
    pub to_version: Version,
    pub estimated_duration_secs: u64,
    pub prerequisites: Vec<StepId>,
    pub post_conditions: Vec<String>,
}

/// A deployment plan consisting of ordered steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPlan {
    pub id: String,
    pub name: String,
    pub steps: Vec<DeployStep>,
    pub created_at: String,
}

impl DeploymentPlan {
    pub fn total_estimated_duration(&self) -> u64 {
        self.steps.iter().map(|s| s.estimated_duration_secs).sum()
    }

    pub fn affected_services(&self) -> Vec<ServiceId> {
        let mut services: Vec<ServiceId> = self.steps.iter().map(|s| s.service.clone()).collect();
        services.sort();
        services.dedup();
        services
    }
}

/// A deployment state mapping services to versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentState {
    pub id: StateId,
    pub service_versions: Vec<(ServiceId, Version)>,
    pub labels: Vec<String>,
}

impl DeploymentState {
    pub fn new(id: StateId) -> Self {
        Self {
            id,
            service_versions: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn with_service(mut self, service: ServiceId, version: Version) -> Self {
        self.service_versions.push((service, version));
        self
    }

    pub fn version_of(&self, service: &ServiceId) -> Option<&Version> {
        self.service_versions
            .iter()
            .find(|(s, _)| s == service)
            .map(|(_, v)| v)
    }
}

/// A constraint on deployment ordering or compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployConstraint {
    pub id: ConstraintId,
    pub constraint_type: ConstraintType,
    pub description: String,
    pub involved_services: Vec<ServiceId>,
    pub severity: ConstraintSeverity,
}

/// The kind of deployment constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    VersionCompatibility,
    OrderingDependency,
    ResourceLimit,
    HealthCheck,
    RollbackSafety,
    Custom(String),
}

impl std::fmt::Display for ConstraintType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionCompatibility => write!(f, "version-compatibility"),
            Self::OrderingDependency => write!(f, "ordering-dependency"),
            Self::ResourceLimit => write!(f, "resource-limit"),
            Self::HealthCheck => write!(f, "health-check"),
            Self::RollbackSafety => write!(f, "rollback-safety"),
            Self::Custom(s) => write!(f, "custom({})", s),
        }
    }
}

/// How severe a constraint violation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    Advisory,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for ConstraintSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Advisory => write!(f, "advisory"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// The safety envelope for a deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    pub id: EnvelopeId,
    pub safe_states: Vec<StateId>,
    pub pnr_states: Vec<StateId>,
    pub boundary_states: Vec<StateId>,
    pub transitions: Vec<(StateId, StateId)>,
}

impl SafetyEnvelope {
    pub fn new(id: EnvelopeId) -> Self {
        Self {
            id,
            safe_states: Vec::new(),
            pnr_states: Vec::new(),
            boundary_states: Vec::new(),
            transitions: Vec::new(),
        }
    }

    pub fn is_safe(&self, state: &StateId) -> bool {
        self.safe_states.contains(state)
    }

    pub fn is_pnr(&self, state: &StateId) -> bool {
        self.pnr_states.contains(state)
    }

    pub fn is_boundary(&self, state: &StateId) -> bool {
        self.boundary_states.contains(state)
    }

    pub fn all_states(&self) -> Vec<&StateId> {
        self.safe_states
            .iter()
            .chain(self.pnr_states.iter())
            .chain(self.boundary_states.iter())
            .collect()
    }

    pub fn has_transition(&self, from: &StateId, to: &StateId) -> bool {
        self.transitions.iter().any(|(f, t)| f == from && t == to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_ordering() {
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert_eq!(v1, Version::new(1, 0, 0));
    }

    #[test]
    fn test_version_display() {
        assert_eq!(Version::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn test_deployment_state() {
        let svc = ServiceId::new("api");
        let state = DeploymentState::new(StateId::new("s1"))
            .with_service(svc.clone(), Version::new(1, 0, 0));
        assert_eq!(state.version_of(&svc), Some(&Version::new(1, 0, 0)));
        assert_eq!(state.version_of(&ServiceId::new("other")), None);
    }

    #[test]
    fn test_safety_envelope() {
        let mut env = SafetyEnvelope::new(EnvelopeId::new("e1"));
        env.safe_states.push(StateId::new("s1"));
        env.pnr_states.push(StateId::new("s2"));
        env.boundary_states.push(StateId::new("s3"));
        env.transitions.push((StateId::new("s1"), StateId::new("s2")));

        assert!(env.is_safe(&StateId::new("s1")));
        assert!(env.is_pnr(&StateId::new("s2")));
        assert!(env.is_boundary(&StateId::new("s3")));
        assert!(env.has_transition(&StateId::new("s1"), &StateId::new("s2")));
        assert!(!env.has_transition(&StateId::new("s2"), &StateId::new("s1")));
        assert_eq!(env.all_states().len(), 3);
    }

    #[test]
    fn test_deployment_plan() {
        let plan = DeploymentPlan {
            id: "p1".into(),
            name: "test plan".into(),
            steps: vec![
                DeployStep {
                    id: StepId::new("step1"),
                    service: ServiceId::new("api"),
                    from_version: Version::new(1, 0, 0),
                    to_version: Version::new(2, 0, 0),
                    estimated_duration_secs: 60,
                    prerequisites: vec![],
                    post_conditions: vec!["healthy".into()],
                },
                DeployStep {
                    id: StepId::new("step2"),
                    service: ServiceId::new("web"),
                    from_version: Version::new(1, 0, 0),
                    to_version: Version::new(1, 1, 0),
                    estimated_duration_secs: 30,
                    prerequisites: vec![StepId::new("step1")],
                    post_conditions: vec![],
                },
            ],
            created_at: "2024-01-01".into(),
        };
        assert_eq!(plan.total_estimated_duration(), 90);
        assert_eq!(plan.affected_services().len(), 2);
    }

    #[test]
    fn test_constraint_type_display() {
        assert_eq!(ConstraintType::VersionCompatibility.to_string(), "version-compatibility");
        assert_eq!(ConstraintType::Custom("foo".into()).to_string(), "custom(foo)");
    }
}
