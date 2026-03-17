//! Comprehensive deployment plan reporting.
//!
//! Generates `PlanReport` objects containing summary, step details, safety
//! analysis, risk assessment and actionable recommendations.

use serde::{Deserialize, Serialize};
use safestep_types::{ServiceId, StateId, StepId};

use crate::{
    DeployStep, DeploymentPlan, DeploymentState, SafetyEnvelope, DeployConstraint,
    Version, ConstraintType, ConstraintSeverity,
    safety_map::EnvelopeMembership,
};

// ---------------------------------------------------------------------------
// PlanSummary
// ---------------------------------------------------------------------------

/// High-level overview of a deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub plan_id: String,
    pub plan_name: String,
    pub total_steps: usize,
    pub estimated_duration_secs: u64,
    pub services_affected: Vec<ServiceId>,
    pub versions_changed: Vec<(ServiceId, Version, Version)>,
    pub pnr_count: usize,
    pub max_risk_score: f64,
    pub created_at: String,
}

impl PlanSummary {
    pub fn estimated_duration_human(&self) -> String {
        let secs = self.estimated_duration_secs;
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        }
    }
}

impl std::fmt::Display for PlanSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Plan '{}': {} steps, ~{}, {} services, {} PNR states, max risk {:.2}",
            self.plan_name,
            self.total_steps,
            self.estimated_duration_human(),
            self.services_affected.len(),
            self.pnr_count,
            self.max_risk_score,
        )
    }
}

// ---------------------------------------------------------------------------
// StepDetail
// ---------------------------------------------------------------------------

/// Detailed information about a single deployment step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDetail {
    pub step_number: usize,
    pub step_id: StepId,
    pub service: ServiceId,
    pub from_version: Version,
    pub to_version: Version,
    pub envelope_status: EnvelopeMembership,
    pub risk_score: f64,
    pub prerequisites: Vec<StepId>,
    pub post_conditions: Vec<String>,
    pub rollback_available: bool,
    pub estimated_duration_secs: u64,
    pub annotations: Vec<String>,
}

impl StepDetail {
    pub fn is_upgrade(&self) -> bool {
        self.to_version > self.from_version
    }

    pub fn is_downgrade(&self) -> bool {
        self.to_version < self.from_version
    }

    pub fn version_change_str(&self) -> String {
        format!("{} → {}", self.from_version, self.to_version)
    }
}

impl std::fmt::Display for StepDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Step {}: {} {} (risk: {:.2}, {})",
            self.step_number,
            self.service.as_str(),
            self.version_change_str(),
            self.risk_score,
            self.envelope_status,
        )
    }
}

// ---------------------------------------------------------------------------
// PnrDescription & TransitionDescription
// ---------------------------------------------------------------------------

/// Description of a PNR (point-of-no-return) state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnrDescription {
    pub state_id: StateId,
    pub step_index: usize,
    pub affected_services: Vec<ServiceId>,
    pub reason: String,
    pub risk_score: f64,
}

impl std::fmt::Display for PnrDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PNR at step {} (state {}): {} (risk: {:.2})",
            self.step_index,
            self.state_id.as_str(),
            self.reason,
            self.risk_score,
        )
    }
}

/// Description of a critical transition between states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionDescription {
    pub from_state: StateId,
    pub to_state: StateId,
    pub step_index: usize,
    pub description: String,
    pub risk_delta: f64,
}

// ---------------------------------------------------------------------------
// SafetyAnalysis
// ---------------------------------------------------------------------------

/// Analysis of plan safety properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAnalysis {
    pub overall_safety_score: f64,
    pub pnr_states: Vec<PnrDescription>,
    pub critical_transitions: Vec<TransitionDescription>,
    pub envelope_coverage: f64,
}

impl SafetyAnalysis {
    pub fn is_safe(&self) -> bool {
        self.overall_safety_score >= 0.8 && self.pnr_states.is_empty()
    }

    pub fn pnr_count(&self) -> usize {
        self.pnr_states.len()
    }
}

impl std::fmt::Display for SafetyAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Safety: {:.1}%, {} PNR states, {:.1}% envelope coverage",
            self.overall_safety_score * 100.0,
            self.pnr_states.len(),
            self.envelope_coverage * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// RiskFactor & RiskAssessment
// ---------------------------------------------------------------------------

/// An individual risk factor identified in the plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub description: String,
    pub severity: ConstraintSeverity,
    pub mitigation: String,
    pub affected_steps: Vec<usize>,
    pub score: f64,
}

impl std::fmt::Display for RiskFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} (score: {:.2}) – mitigation: {}",
            self.severity, self.description, self.score, self.mitigation,
        )
    }
}

/// Risk scoring for the entire plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub total_risk: f64,
    pub per_step_risk: Vec<f64>,
    pub risk_factors: Vec<RiskFactor>,
}

impl RiskAssessment {
    pub fn max_step_risk(&self) -> f64 {
        self.per_step_risk
            .iter()
            .cloned()
            .fold(0.0f64, f64::max)
    }

    pub fn mean_step_risk(&self) -> f64 {
        if self.per_step_risk.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.per_step_risk.iter().sum();
        sum / self.per_step_risk.len() as f64
    }

    pub fn risk_level(&self) -> &'static str {
        if self.total_risk < 0.2 {
            "LOW"
        } else if self.total_risk < 0.5 {
            "MODERATE"
        } else if self.total_risk < 0.8 {
            "HIGH"
        } else {
            "CRITICAL"
        }
    }
}

impl std::fmt::Display for RiskAssessment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Risk: {:.2} ({}) – {} factor(s), max step risk {:.2}",
            self.total_risk,
            self.risk_level(),
            self.risk_factors.len(),
            self.max_step_risk(),
        )
    }
}

// ---------------------------------------------------------------------------
// Recommendation
// ---------------------------------------------------------------------------

/// Type of recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    Warning,
    Suggestion,
    Required,
}

impl std::fmt::Display for RecommendationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Warning => write!(f, "WARNING"),
            Self::Suggestion => write!(f, "SUGGESTION"),
            Self::Required => write!(f, "REQUIRED"),
        }
    }
}

/// Actionable recommendation for the deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub type_: RecommendationType,
    pub message: String,
    pub affected_steps: Vec<usize>,
    pub suggested_action: String,
}

impl std::fmt::Display for Recommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} → {}",
            self.type_, self.message, self.suggested_action,
        )
    }
}

// ---------------------------------------------------------------------------
// PlanReport
// ---------------------------------------------------------------------------

/// Comprehensive deployment plan report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanReport {
    pub plan_summary: PlanSummary,
    pub step_details: Vec<StepDetail>,
    pub safety_analysis: SafetyAnalysis,
    pub risk_assessment: RiskAssessment,
    pub recommendations: Vec<Recommendation>,
}

impl PlanReport {
    pub fn has_warnings(&self) -> bool {
        self.recommendations
            .iter()
            .any(|r| r.type_ == RecommendationType::Warning)
    }

    pub fn has_required_actions(&self) -> bool {
        self.recommendations
            .iter()
            .any(|r| r.type_ == RecommendationType::Required)
    }

    pub fn warning_count(&self) -> usize {
        self.recommendations
            .iter()
            .filter(|r| r.type_ == RecommendationType::Warning)
            .count()
    }

    pub fn to_summary_string(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("{}\n", self.plan_summary));
        out.push_str(&format!("{}\n", self.safety_analysis));
        out.push_str(&format!("{}\n", self.risk_assessment));
        if !self.recommendations.is_empty() {
            out.push_str(&format!(
                "Recommendations: {}\n",
                self.recommendations.len()
            ));
        }
        out
    }
}

impl std::fmt::Display for PlanReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_summary_string())
    }
}

// ---------------------------------------------------------------------------
// ReportGenerator
// ---------------------------------------------------------------------------

/// Generates comprehensive plan reports from plan + envelope + constraints.
pub struct ReportGenerator {
    verbosity: Verbosity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Minimal,
    Normal,
    Verbose,
}

impl ReportGenerator {
    pub fn new() -> Self {
        Self {
            verbosity: Verbosity::Normal,
        }
    }

    pub fn with_verbosity(mut self, v: Verbosity) -> Self {
        self.verbosity = v;
        self
    }

    /// Generate a full plan report.
    pub fn generate(
        &self,
        plan: &DeploymentPlan,
        envelope: &SafetyEnvelope,
        constraints: &[DeployConstraint],
    ) -> PlanReport {
        let step_details = self.build_step_details(plan, envelope);
        let safety_analysis = self.build_safety_analysis(plan, envelope, &step_details);
        let risk_assessment = self.build_risk_assessment(plan, constraints, &step_details);
        let recommendations =
            self.build_recommendations(&safety_analysis, &risk_assessment, &step_details);
        let summary = self.build_summary(plan, &safety_analysis, &risk_assessment);

        PlanReport {
            plan_summary: summary,
            step_details,
            safety_analysis,
            risk_assessment,
            recommendations,
        }
    }

    fn build_summary(
        &self,
        plan: &DeploymentPlan,
        safety: &SafetyAnalysis,
        risk: &RiskAssessment,
    ) -> PlanSummary {
        let services = plan.affected_services();
        let versions_changed: Vec<(ServiceId, Version, Version)> = plan
            .steps
            .iter()
            .map(|s| (s.service.clone(), s.from_version.clone(), s.to_version.clone()))
            .collect();

        PlanSummary {
            plan_id: plan.id.clone(),
            plan_name: plan.name.clone(),
            total_steps: plan.steps.len(),
            estimated_duration_secs: plan.total_estimated_duration(),
            services_affected: services,
            versions_changed,
            pnr_count: safety.pnr_count(),
            max_risk_score: risk.max_step_risk(),
            created_at: plan.created_at.clone(),
        }
    }

    fn build_step_details(
        &self,
        plan: &DeploymentPlan,
        envelope: &SafetyEnvelope,
    ) -> Vec<StepDetail> {
        plan.steps
            .iter()
            .enumerate()
            .map(|(i, step)| {
                let state_id = StateId::new(format!("after-step-{}", i));
                let membership = if envelope.is_pnr(&state_id) {
                    EnvelopeMembership::PNR
                } else if envelope.is_boundary(&state_id) {
                    EnvelopeMembership::Boundary
                } else if envelope.is_safe(&state_id) {
                    EnvelopeMembership::Inside
                } else {
                    // Heuristic: assign based on step position
                    self.infer_membership(i, plan.steps.len())
                };

                let risk = self.compute_step_risk(&membership, step, i, plan.steps.len());
                let rollback = membership != EnvelopeMembership::PNR
                    && membership != EnvelopeMembership::Outside;

                let mut annotations = Vec::new();
                if membership == EnvelopeMembership::PNR {
                    annotations.push("⚠ Point of no return".to_string());
                }
                if !rollback {
                    annotations.push("No rollback available".to_string());
                }
                if step.to_version < step.from_version {
                    annotations.push("Downgrade".to_string());
                }

                StepDetail {
                    step_number: i + 1,
                    step_id: step.id.clone(),
                    service: step.service.clone(),
                    from_version: step.from_version.clone(),
                    to_version: step.to_version.clone(),
                    envelope_status: membership,
                    risk_score: risk,
                    prerequisites: step.prerequisites.clone(),
                    post_conditions: step.post_conditions.clone(),
                    rollback_available: rollback,
                    estimated_duration_secs: step.estimated_duration_secs,
                    annotations,
                }
            })
            .collect()
    }

    fn infer_membership(&self, step_idx: usize, total: usize) -> EnvelopeMembership {
        if total == 0 {
            return EnvelopeMembership::Inside;
        }
        let progress = step_idx as f64 / total as f64;
        if progress < 0.25 || progress > 0.85 {
            EnvelopeMembership::Inside
        } else if progress < 0.4 || progress > 0.7 {
            EnvelopeMembership::Boundary
        } else {
            EnvelopeMembership::Boundary
        }
    }

    fn compute_step_risk(
        &self,
        membership: &EnvelopeMembership,
        step: &DeployStep,
        step_idx: usize,
        total_steps: usize,
    ) -> f64 {
        let base = match membership {
            EnvelopeMembership::Inside => 0.1,
            EnvelopeMembership::Boundary => 0.4,
            EnvelopeMembership::PNR => 0.85,
            EnvelopeMembership::Outside => 0.95,
        };

        // Major version changes are riskier
        let version_delta = if step.to_version.major != step.from_version.major {
            0.1
        } else if step.to_version.minor != step.from_version.minor {
            0.05
        } else {
            0.0
        };

        // Middle steps are slightly riskier (harder to rollback partially)
        let position_factor = if total_steps <= 1 {
            0.0
        } else {
            let mid = total_steps as f64 / 2.0;
            let dist = (step_idx as f64 - mid).abs() / mid;
            (1.0 - dist) * 0.05
        };

        (base + version_delta + position_factor).min(1.0)
    }

    fn build_safety_analysis(
        &self,
        plan: &DeploymentPlan,
        envelope: &SafetyEnvelope,
        steps: &[StepDetail],
    ) -> SafetyAnalysis {
        let inside_count = steps
            .iter()
            .filter(|s| s.envelope_status == EnvelopeMembership::Inside)
            .count();
        let envelope_coverage = if steps.is_empty() {
            1.0
        } else {
            inside_count as f64 / steps.len() as f64
        };

        let pnr_states: Vec<PnrDescription> = steps
            .iter()
            .filter(|s| s.envelope_status == EnvelopeMembership::PNR)
            .map(|s| PnrDescription {
                state_id: StateId::new(format!("after-step-{}", s.step_number - 1)),
                step_index: s.step_number,
                affected_services: vec![s.service.clone()],
                reason: format!(
                    "Service {} transitions {} with no rollback path",
                    s.service.as_str(),
                    s.version_change_str()
                ),
                risk_score: s.risk_score,
            })
            .collect();

        let critical_transitions: Vec<TransitionDescription> = steps
            .iter()
            .zip(steps.iter().skip(1))
            .filter(|(a, b)| {
                (b.risk_score - a.risk_score) > 0.2
                    || b.envelope_status == EnvelopeMembership::PNR
            })
            .map(|(a, b)| TransitionDescription {
                from_state: StateId::new(format!("after-step-{}", a.step_number - 1)),
                to_state: StateId::new(format!("after-step-{}", b.step_number - 1)),
                step_index: b.step_number,
                description: format!(
                    "Risk increases from {:.2} to {:.2} when deploying {}",
                    a.risk_score,
                    b.risk_score,
                    b.service.as_str(),
                ),
                risk_delta: b.risk_score - a.risk_score,
            })
            .collect();

        let risk_scores: Vec<f64> = steps.iter().map(|s| s.risk_score).collect();
        let overall = if risk_scores.is_empty() {
            1.0
        } else {
            let avg_risk: f64 = risk_scores.iter().sum::<f64>() / risk_scores.len() as f64;
            (1.0 - avg_risk).max(0.0)
        };

        SafetyAnalysis {
            overall_safety_score: overall,
            pnr_states,
            critical_transitions,
            envelope_coverage,
        }
    }

    fn build_risk_assessment(
        &self,
        plan: &DeploymentPlan,
        constraints: &[DeployConstraint],
        steps: &[StepDetail],
    ) -> RiskAssessment {
        let per_step_risk: Vec<f64> = steps.iter().map(|s| s.risk_score).collect();
        let total_risk = if per_step_risk.is_empty() {
            0.0
        } else {
            per_step_risk.iter().cloned().fold(0.0f64, f64::max)
        };

        let mut risk_factors = Vec::new();

        // PNR risk factor
        let pnr_steps: Vec<usize> = steps
            .iter()
            .filter(|s| s.envelope_status == EnvelopeMembership::PNR)
            .map(|s| s.step_number)
            .collect();
        if !pnr_steps.is_empty() {
            risk_factors.push(RiskFactor {
                description: format!(
                    "{} step(s) reach point-of-no-return",
                    pnr_steps.len()
                ),
                severity: ConstraintSeverity::Critical,
                mitigation: "Add intermediate versions or relaxation constraints".to_string(),
                affected_steps: pnr_steps,
                score: 0.9,
            });
        }

        // Major version jump risk factor
        let major_jumps: Vec<usize> = steps
            .iter()
            .filter(|s| {
                (s.to_version.major as i64 - s.from_version.major as i64).unsigned_abs() > 1
            })
            .map(|s| s.step_number)
            .collect();
        if !major_jumps.is_empty() {
            risk_factors.push(RiskFactor {
                description: format!(
                    "{} step(s) have large version jumps (>1 major)",
                    major_jumps.len()
                ),
                severity: ConstraintSeverity::Warning,
                mitigation: "Consider adding intermediate versions".to_string(),
                affected_steps: major_jumps,
                score: 0.6,
            });
        }

        // No rollback risk factor
        let no_rollback: Vec<usize> = steps
            .iter()
            .filter(|s| !s.rollback_available)
            .map(|s| s.step_number)
            .collect();
        if !no_rollback.is_empty() {
            risk_factors.push(RiskFactor {
                description: format!(
                    "{} step(s) have no rollback available",
                    no_rollback.len()
                ),
                severity: ConstraintSeverity::Warning,
                mitigation: "Ensure manual rollback procedures are documented".to_string(),
                affected_steps: no_rollback,
                score: 0.7,
            });
        }

        // Critical constraints risk factor
        let critical_constraint_count = constraints
            .iter()
            .filter(|c| c.severity == ConstraintSeverity::Critical)
            .count();
        if critical_constraint_count > 0 {
            risk_factors.push(RiskFactor {
                description: format!(
                    "{} critical constraint(s) in the deployment",
                    critical_constraint_count
                ),
                severity: ConstraintSeverity::Critical,
                mitigation: "Review and validate all critical constraints before deployment"
                    .to_string(),
                affected_steps: (1..=steps.len()).collect(),
                score: 0.8,
            });
        }

        // Downgrade risk factor
        let downgrades: Vec<usize> = steps
            .iter()
            .filter(|s| s.is_downgrade())
            .map(|s| s.step_number)
            .collect();
        if !downgrades.is_empty() {
            risk_factors.push(RiskFactor {
                description: format!("{} step(s) involve version downgrades", downgrades.len()),
                severity: ConstraintSeverity::Warning,
                mitigation: "Verify data compatibility for downgrade paths".to_string(),
                affected_steps: downgrades,
                score: 0.5,
            });
        }

        RiskAssessment {
            total_risk,
            per_step_risk,
            risk_factors,
        }
    }

    fn build_recommendations(
        &self,
        safety: &SafetyAnalysis,
        risk: &RiskAssessment,
        steps: &[StepDetail],
    ) -> Vec<Recommendation> {
        let mut recs = Vec::new();

        // PNR warnings
        for pnr in &safety.pnr_states {
            recs.push(Recommendation {
                type_: RecommendationType::Warning,
                message: format!(
                    "Step {} enters a point-of-no-return state",
                    pnr.step_index
                ),
                affected_steps: vec![pnr.step_index],
                suggested_action:
                    "Consider adding a safe checkpoint before this step".to_string(),
            });
        }

        // High risk recommendations
        for (i, &risk_val) in risk.per_step_risk.iter().enumerate() {
            if risk_val > 0.7 {
                recs.push(Recommendation {
                    type_: RecommendationType::Warning,
                    message: format!(
                        "Step {} has high risk score ({:.2})",
                        i + 1,
                        risk_val
                    ),
                    affected_steps: vec![i + 1],
                    suggested_action: "Add health checks and monitoring before proceeding"
                        .to_string(),
                });
            }
        }

        // Envelope coverage recommendation
        if safety.envelope_coverage < 0.5 {
            recs.push(Recommendation {
                type_: RecommendationType::Required,
                message: format!(
                    "Only {:.0}% of the plan stays within the safety envelope",
                    safety.envelope_coverage * 100.0
                ),
                affected_steps: Vec::new(),
                suggested_action: "Restructure the plan to stay within the safety envelope"
                    .to_string(),
            });
        } else if safety.envelope_coverage < 0.8 {
            recs.push(Recommendation {
                type_: RecommendationType::Suggestion,
                message: format!(
                    "{:.0}% envelope coverage is below recommended 80%",
                    safety.envelope_coverage * 100.0
                ),
                affected_steps: Vec::new(),
                suggested_action: "Consider adding intermediate steps to improve coverage"
                    .to_string(),
            });
        }

        // No rollback recommendation
        let no_rb_steps: Vec<usize> = steps
            .iter()
            .filter(|s| !s.rollback_available)
            .map(|s| s.step_number)
            .collect();
        if !no_rb_steps.is_empty() {
            recs.push(Recommendation {
                type_: RecommendationType::Required,
                message: format!(
                    "{} step(s) have no rollback path",
                    no_rb_steps.len()
                ),
                affected_steps: no_rb_steps,
                suggested_action:
                    "Document manual rollback procedures for these steps".to_string(),
            });
        }

        // Overall safety recommendation
        if safety.overall_safety_score < 0.5 {
            recs.push(Recommendation {
                type_: RecommendationType::Required,
                message: format!(
                    "Overall safety score is low ({:.0}%)",
                    safety.overall_safety_score * 100.0
                ),
                affected_steps: Vec::new(),
                suggested_action: "Review and revise the deployment plan before proceeding"
                    .to_string(),
            });
        }

        recs
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safestep_types::{EnvelopeId, StepId};
    use crate::{SafetyEnvelope, DeployStep};

    fn sample_plan() -> DeploymentPlan {
        DeploymentPlan {
            id: "p1".into(),
            name: "Test Deploy".into(),
            steps: vec![
                DeployStep {
                    id: StepId::new("s1"),
                    service: ServiceId::new("api"),
                    from_version: Version::new(1, 0, 0),
                    to_version: Version::new(2, 0, 0),
                    estimated_duration_secs: 120,
                    prerequisites: vec![],
                    post_conditions: vec!["healthy".into()],
                },
                DeployStep {
                    id: StepId::new("s2"),
                    service: ServiceId::new("web"),
                    from_version: Version::new(1, 0, 0),
                    to_version: Version::new(1, 1, 0),
                    estimated_duration_secs: 60,
                    prerequisites: vec![StepId::new("s1")],
                    post_conditions: vec![],
                },
                DeployStep {
                    id: StepId::new("s3"),
                    service: ServiceId::new("db"),
                    from_version: Version::new(3, 0, 0),
                    to_version: Version::new(4, 0, 0),
                    estimated_duration_secs: 180,
                    prerequisites: vec![StepId::new("s2")],
                    post_conditions: vec!["migrated".into()],
                },
            ],
            created_at: "2024-01-01".into(),
        }
    }

    fn sample_envelope() -> SafetyEnvelope {
        let mut env = SafetyEnvelope::new(EnvelopeId::new("e1"));
        env.safe_states.push(StateId::new("after-step-0"));
        env.boundary_states.push(StateId::new("after-step-1"));
        env.safe_states.push(StateId::new("after-step-2"));
        env
    }

    fn sample_constraints() -> Vec<DeployConstraint> {
        vec![DeployConstraint {
            id: safestep_types::ConstraintId::new("c1"),
            constraint_type: ConstraintType::VersionCompatibility,
            description: "API and Web must be compatible".into(),
            involved_services: vec![
                ServiceId::new("api"),
                ServiceId::new("web"),
            ],
            severity: ConstraintSeverity::Warning,
        }]
    }

    #[test]
    fn test_plan_summary_duration_human() {
        let s = PlanSummary {
            plan_id: "p1".into(),
            plan_name: "test".into(),
            total_steps: 2,
            estimated_duration_secs: 90,
            services_affected: vec![],
            versions_changed: vec![],
            pnr_count: 0,
            max_risk_score: 0.5,
            created_at: "now".into(),
        };
        assert_eq!(s.estimated_duration_human(), "1m 30s");
    }

    #[test]
    fn test_plan_summary_duration_hours() {
        let s = PlanSummary {
            plan_id: "p1".into(),
            plan_name: "test".into(),
            total_steps: 2,
            estimated_duration_secs: 7200,
            services_affected: vec![],
            versions_changed: vec![],
            pnr_count: 0,
            max_risk_score: 0.5,
            created_at: "now".into(),
        };
        assert_eq!(s.estimated_duration_human(), "2h 0m");
    }

    #[test]
    fn test_plan_summary_duration_seconds() {
        let s = PlanSummary {
            plan_id: "p1".into(),
            plan_name: "test".into(),
            total_steps: 0,
            estimated_duration_secs: 45,
            services_affected: vec![],
            versions_changed: vec![],
            pnr_count: 0,
            max_risk_score: 0.0,
            created_at: "now".into(),
        };
        assert_eq!(s.estimated_duration_human(), "45s");
    }

    #[test]
    fn test_step_detail_upgrade() {
        let sd = StepDetail {
            step_number: 1,
            step_id: StepId::new("s1"),
            service: ServiceId::new("api"),
            from_version: Version::new(1, 0, 0),
            to_version: Version::new(2, 0, 0),
            envelope_status: EnvelopeMembership::Inside,
            risk_score: 0.3,
            prerequisites: vec![],
            post_conditions: vec![],
            rollback_available: true,
            estimated_duration_secs: 60,
            annotations: vec![],
        };
        assert!(sd.is_upgrade());
        assert!(!sd.is_downgrade());
        assert_eq!(sd.version_change_str(), "1.0.0 → 2.0.0");
    }

    #[test]
    fn test_step_detail_downgrade() {
        let sd = StepDetail {
            step_number: 1,
            step_id: StepId::new("s1"),
            service: ServiceId::new("api"),
            from_version: Version::new(2, 0, 0),
            to_version: Version::new(1, 0, 0),
            envelope_status: EnvelopeMembership::Inside,
            risk_score: 0.5,
            prerequisites: vec![],
            post_conditions: vec![],
            rollback_available: true,
            estimated_duration_secs: 60,
            annotations: vec![],
        };
        assert!(!sd.is_upgrade());
        assert!(sd.is_downgrade());
    }

    #[test]
    fn test_safety_analysis_is_safe() {
        let sa = SafetyAnalysis {
            overall_safety_score: 0.9,
            pnr_states: vec![],
            critical_transitions: vec![],
            envelope_coverage: 0.95,
        };
        assert!(sa.is_safe());
    }

    #[test]
    fn test_safety_analysis_unsafe_with_pnr() {
        let sa = SafetyAnalysis {
            overall_safety_score: 0.9,
            pnr_states: vec![PnrDescription {
                state_id: StateId::new("s1"),
                step_index: 1,
                affected_services: vec![],
                reason: "test".into(),
                risk_score: 0.9,
            }],
            critical_transitions: vec![],
            envelope_coverage: 0.95,
        };
        assert!(!sa.is_safe());
    }

    #[test]
    fn test_risk_assessment_levels() {
        let make = |total: f64| RiskAssessment {
            total_risk: total,
            per_step_risk: vec![total],
            risk_factors: vec![],
        };
        assert_eq!(make(0.1).risk_level(), "LOW");
        assert_eq!(make(0.3).risk_level(), "MODERATE");
        assert_eq!(make(0.6).risk_level(), "HIGH");
        assert_eq!(make(0.9).risk_level(), "CRITICAL");
    }

    #[test]
    fn test_risk_assessment_stats() {
        let ra = RiskAssessment {
            total_risk: 0.5,
            per_step_risk: vec![0.2, 0.4, 0.6],
            risk_factors: vec![],
        };
        assert_eq!(ra.max_step_risk(), 0.6);
        assert!((ra.mean_step_risk() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_risk_assessment_empty() {
        let ra = RiskAssessment {
            total_risk: 0.0,
            per_step_risk: vec![],
            risk_factors: vec![],
        };
        assert_eq!(ra.max_step_risk(), 0.0);
        assert_eq!(ra.mean_step_risk(), 0.0);
    }

    #[test]
    fn test_recommendation_display() {
        let r = Recommendation {
            type_: RecommendationType::Warning,
            message: "High risk".into(),
            affected_steps: vec![1],
            suggested_action: "Add monitoring".into(),
        };
        let s = format!("{}", r);
        assert!(s.contains("WARNING"));
        assert!(s.contains("High risk"));
    }

    #[test]
    fn test_report_generator_basic() {
        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);

        assert_eq!(report.plan_summary.total_steps, 3);
        assert_eq!(report.step_details.len(), 3);
        assert!(report.plan_summary.estimated_duration_secs > 0);
    }

    #[test]
    fn test_report_generator_step_details() {
        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);

        let s1 = &report.step_details[0];
        assert_eq!(s1.step_number, 1);
        assert_eq!(s1.service.as_str(), "api");
        assert!(s1.risk_score >= 0.0 && s1.risk_score <= 1.0);
    }

    #[test]
    fn test_report_generator_safety_analysis() {
        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);

        assert!(report.safety_analysis.overall_safety_score >= 0.0);
        assert!(report.safety_analysis.overall_safety_score <= 1.0);
        assert!(report.safety_analysis.envelope_coverage >= 0.0);
        assert!(report.safety_analysis.envelope_coverage <= 1.0);
    }

    #[test]
    fn test_report_generator_risk_assessment() {
        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);

        assert_eq!(report.risk_assessment.per_step_risk.len(), 3);
        assert!(report.risk_assessment.total_risk >= 0.0);
    }

    #[test]
    fn test_report_has_warnings() {
        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);
        // The report should have at least one recommendation for a 3-step plan
        assert!(!report.recommendations.is_empty() || report.safety_analysis.is_safe());
    }

    #[test]
    fn test_report_summary_string() {
        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);
        let summary = report.to_summary_string();
        assert!(summary.contains("Test Deploy"));
        assert!(summary.contains("3 steps"));
    }

    #[test]
    fn test_report_display() {
        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);
        let display = format!("{}", report);
        assert!(!display.is_empty());
    }

    #[test]
    fn test_report_generator_empty_plan() {
        let gen = ReportGenerator::new();
        let plan = DeploymentPlan {
            id: "empty".into(),
            name: "Empty".into(),
            steps: vec![],
            created_at: "now".into(),
        };
        let env = SafetyEnvelope::new(EnvelopeId::new("e1"));
        let constraints: Vec<DeployConstraint> = vec![];

        let report = gen.generate(&plan, &env, &constraints);
        assert_eq!(report.plan_summary.total_steps, 0);
        assert!(report.step_details.is_empty());
    }

    #[test]
    fn test_report_generator_with_pnr() {
        let mut env = SafetyEnvelope::new(EnvelopeId::new("e1"));
        env.pnr_states.push(StateId::new("after-step-1"));

        let gen = ReportGenerator::new();
        let plan = sample_plan();
        let constraints = sample_constraints();

        let report = gen.generate(&plan, &env, &constraints);
        let step2 = &report.step_details[1];
        assert_eq!(step2.envelope_status, EnvelopeMembership::PNR);
        assert!(!step2.rollback_available);
    }

    #[test]
    fn test_recommendation_type_display() {
        assert_eq!(RecommendationType::Warning.to_string(), "WARNING");
        assert_eq!(RecommendationType::Suggestion.to_string(), "SUGGESTION");
        assert_eq!(RecommendationType::Required.to_string(), "REQUIRED");
    }

    #[test]
    fn test_pnr_description_display() {
        let pnr = PnrDescription {
            state_id: StateId::new("s1"),
            step_index: 2,
            affected_services: vec![ServiceId::new("api")],
            reason: "no rollback path".into(),
            risk_score: 0.85,
        };
        let s = format!("{}", pnr);
        assert!(s.contains("PNR at step 2"));
        assert!(s.contains("no rollback path"));
    }

    #[test]
    fn test_step_detail_display() {
        let sd = StepDetail {
            step_number: 1,
            step_id: StepId::new("s1"),
            service: ServiceId::new("api"),
            from_version: Version::new(1, 0, 0),
            to_version: Version::new(2, 0, 0),
            envelope_status: EnvelopeMembership::Inside,
            risk_score: 0.3,
            prerequisites: vec![],
            post_conditions: vec![],
            rollback_available: true,
            estimated_duration_secs: 60,
            annotations: vec![],
        };
        let s = format!("{}", sd);
        assert!(s.contains("Step 1"));
        assert!(s.contains("api"));
        assert!(s.contains("1.0.0 → 2.0.0"));
    }

    #[test]
    fn test_report_generator_verbose() {
        let gen = ReportGenerator::new().with_verbosity(Verbosity::Verbose);
        let plan = sample_plan();
        let env = sample_envelope();
        let constraints = sample_constraints();
        let report = gen.generate(&plan, &env, &constraints);
        assert_eq!(report.plan_summary.total_steps, 3);
    }
}
