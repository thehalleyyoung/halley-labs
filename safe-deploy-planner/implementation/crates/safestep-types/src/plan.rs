// Deployment plan types for the SafeStep deployment planner.

use std::fmt;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::constraint::ConstraintEvaluation;
use crate::error::{Result, SafeStepError};
use crate::graph::{ClusterState, GraphEdge};
use crate::identifiers::{PlanId, StepId};
use crate::version::VersionIndex;

// ─── PlanStep ───────────────────────────────────────────────────────────

/// A single step in a deployment plan: upgrade one service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: StepId,
    pub index: usize,
    pub service_idx: usize,
    pub service_name: String,
    pub from_version: VersionIndex,
    pub to_version: VersionIndex,
    pub from_label: String,
    pub to_label: String,
    pub annotation: StepAnnotation,
}

impl PlanStep {
    pub fn new(
        index: usize,
        service_idx: usize,
        service_name: impl Into<String>,
        from_version: VersionIndex,
        to_version: VersionIndex,
    ) -> Self {
        Self {
            id: StepId::generate(),
            index,
            service_idx,
            service_name: service_name.into(),
            from_version,
            to_version,
            from_label: format!("v{}", from_version.0),
            to_label: format!("v{}", to_version.0),
            annotation: StepAnnotation::default(),
        }
    }

    pub fn with_labels(
        mut self,
        from_label: impl Into<String>,
        to_label: impl Into<String>,
    ) -> Self {
        self.from_label = from_label.into();
        self.to_label = to_label.into();
        self
    }

    pub fn with_annotation(mut self, annotation: StepAnnotation) -> Self {
        self.annotation = annotation;
        self
    }

    pub fn is_upgrade(&self) -> bool {
        self.to_version.0 > self.from_version.0
    }

    pub fn is_downgrade(&self) -> bool {
        self.to_version.0 < self.from_version.0
    }

    pub fn to_edge(&self) -> GraphEdge {
        GraphEdge::new(self.service_idx, self.from_version, self.to_version)
    }
}

impl fmt::Display for PlanStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Step {}: {} {} -> {}",
            self.index, self.service_name, self.from_label, self.to_label,
        )?;
        if self.annotation.is_pnr {
            write!(f, " [PNR]")?;
        }
        Ok(())
    }
}

// ─── StepAnnotation ─────────────────────────────────────────────────────

/// Annotations for a deployment step: envelope membership, PNR status, etc.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepAnnotation {
    pub in_envelope: bool,
    pub is_pnr: bool,
    pub confidence: OrderedFloat<f64>,
    pub risk_score: OrderedFloat<f64>,
    pub estimated_duration_secs: Option<u64>,
    pub rollback_info: Option<RollbackInfo>,
    pub notes: Vec<String>,
}

impl StepAnnotation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn safe() -> Self {
        Self {
            in_envelope: true,
            is_pnr: false,
            confidence: OrderedFloat(1.0),
            risk_score: OrderedFloat(0.0),
            ..Default::default()
        }
    }

    pub fn pnr() -> Self {
        Self {
            in_envelope: false,
            is_pnr: true,
            confidence: OrderedFloat(1.0),
            risk_score: OrderedFloat(1.0),
            ..Default::default()
        }
    }

    pub fn with_risk(mut self, risk: f64) -> Self {
        self.risk_score = OrderedFloat(risk);
        self
    }

    pub fn with_duration(mut self, secs: u64) -> Self {
        self.estimated_duration_secs = Some(secs);
        self
    }

    pub fn with_rollback(mut self, info: RollbackInfo) -> Self {
        self.rollback_info = Some(info);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

// ─── RollbackInfo ───────────────────────────────────────────────────────

/// Information about rollback safety for a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackInfo {
    pub can_rollback: bool,
    pub rollback_steps: Vec<RollbackStep>,
    pub data_loss_risk: bool,
    pub estimated_rollback_time_secs: u64,
    pub blocking_reason: Option<String>,
}

impl RollbackInfo {
    pub fn safe(rollback_time_secs: u64) -> Self {
        Self {
            can_rollback: true,
            rollback_steps: Vec::new(),
            data_loss_risk: false,
            estimated_rollback_time_secs: rollback_time_secs,
            blocking_reason: None,
        }
    }

    pub fn unsafe_rollback(reason: impl Into<String>) -> Self {
        Self {
            can_rollback: false,
            rollback_steps: Vec::new(),
            data_loss_risk: false,
            estimated_rollback_time_secs: 0,
            blocking_reason: Some(reason.into()),
        }
    }

    pub fn with_step(mut self, step: RollbackStep) -> Self {
        self.rollback_steps.push(step);
        self
    }

    pub fn with_data_loss_risk(mut self) -> Self {
        self.data_loss_risk = true;
        self
    }
}

/// A single step to execute during rollback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackStep {
    pub service_idx: usize,
    pub service_name: String,
    pub from_version: VersionIndex,
    pub to_version: VersionIndex,
    pub description: String,
}

impl RollbackStep {
    pub fn new(
        service_idx: usize,
        service_name: impl Into<String>,
        from_version: VersionIndex,
        to_version: VersionIndex,
    ) -> Self {
        Self {
            service_idx,
            service_name: service_name.into(),
            from_version,
            to_version,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

impl fmt::Display for RollbackStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rollback {}: v{} -> v{}",
            self.service_name, self.from_version.0, self.to_version.0
        )
    }
}

// ─── PlanCost ───────────────────────────────────────────────────────────

/// Multi-objective cost of a deployment plan.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanCost {
    pub total_steps: usize,
    pub total_risk: OrderedFloat<f64>,
    pub estimated_duration_secs: u64,
    pub resource_overhead: OrderedFloat<f64>,
    pub pnr_count: usize,
    pub soft_violations: usize,
    pub penalty_sum: OrderedFloat<f64>,
}

impl PlanCost {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scalar cost combining all objectives with weights.
    pub fn weighted_cost(&self, weights: &CostWeights) -> f64 {
        weights.step_weight * self.total_steps as f64
            + weights.risk_weight * self.total_risk.into_inner()
            + weights.time_weight * self.estimated_duration_secs as f64
            + weights.resource_weight * self.resource_overhead.into_inner()
            + weights.pnr_weight * self.pnr_count as f64
            + weights.penalty_weight * self.penalty_sum.into_inner()
    }

    /// Pareto-dominates another cost (strictly better or equal in all objectives, strictly better in at least one).
    pub fn dominates(&self, other: &PlanCost) -> bool {
        let self_vals = [
            self.total_steps as f64,
            self.total_risk.into_inner(),
            self.estimated_duration_secs as f64,
            self.pnr_count as f64,
        ];
        let other_vals = [
            other.total_steps as f64,
            other.total_risk.into_inner(),
            other.estimated_duration_secs as f64,
            other.pnr_count as f64,
        ];
        let all_leq = self_vals.iter().zip(other_vals.iter()).all(|(a, b)| a <= b);
        let some_lt = self_vals.iter().zip(other_vals.iter()).any(|(a, b)| a < b);
        all_leq && some_lt
    }
}

impl fmt::Display for PlanCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cost(steps={}, risk={:.2}, time={}s, PNRs={})",
            self.total_steps,
            self.total_risk,
            self.estimated_duration_secs,
            self.pnr_count,
        )
    }
}

/// Weights for combining cost objectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostWeights {
    pub step_weight: f64,
    pub risk_weight: f64,
    pub time_weight: f64,
    pub resource_weight: f64,
    pub pnr_weight: f64,
    pub penalty_weight: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            step_weight: 1.0,
            risk_weight: 10.0,
            time_weight: 0.01,
            resource_weight: 1.0,
            pnr_weight: 100.0,
            penalty_weight: 5.0,
        }
    }
}

// ─── PlanComparison ─────────────────────────────────────────────────────

/// Result of comparing two plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanComparison {
    FirstBetter,
    SecondBetter,
    Incomparable,
    Equal,
}

impl PlanComparison {
    pub fn compare(a: &PlanCost, b: &PlanCost) -> Self {
        if a.dominates(b) {
            Self::FirstBetter
        } else if b.dominates(a) {
            Self::SecondBetter
        } else if a.total_steps == b.total_steps
            && a.total_risk == b.total_risk
            && a.estimated_duration_secs == b.estimated_duration_secs
            && a.pnr_count == b.pnr_count
        {
            Self::Equal
        } else {
            Self::Incomparable
        }
    }
}

impl fmt::Display for PlanComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstBetter => write!(f, "first plan is better"),
            Self::SecondBetter => write!(f, "second plan is better"),
            Self::Incomparable => write!(f, "plans are Pareto-incomparable"),
            Self::Equal => write!(f, "plans are equal"),
        }
    }
}

// ─── PlanMetadata ───────────────────────────────────────────────────────

/// Metadata about a deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanMetadata {
    pub created_at: String,
    pub planner_version: String,
    pub solver_iterations: u64,
    pub encoding_type: String,
    pub bmc_depth: usize,
    pub is_optimal: bool,
    pub is_monotone: bool,
    pub notes: Vec<String>,
}

impl PlanMetadata {
    pub fn new() -> Self {
        Self {
            created_at: chrono::Utc::now().to_rfc3339(),
            planner_version: env!("CARGO_PKG_VERSION").to_string(),
            solver_iterations: 0,
            encoding_type: String::new(),
            bmc_depth: 0,
            is_optimal: false,
            is_monotone: true,
            notes: Vec::new(),
        }
    }

    pub fn with_solver_iterations(mut self, iterations: u64) -> Self {
        self.solver_iterations = iterations;
        self
    }

    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding_type = encoding.into();
        self
    }

    pub fn with_bmc_depth(mut self, depth: usize) -> Self {
        self.bmc_depth = depth;
        self
    }

    pub fn with_optimal(mut self, optimal: bool) -> Self {
        self.is_optimal = optimal;
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

impl Default for PlanMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// ─── PlanValidation ─────────────────────────────────────────────────────

/// Result of validating a plan against constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanValidation {
    pub is_valid: bool,
    pub step_evaluations: Vec<Vec<ConstraintEvaluation>>,
    pub violations: Vec<PlanViolation>,
    pub total_penalty: OrderedFloat<f64>,
}

impl PlanValidation {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            step_evaluations: Vec::new(),
            violations: Vec::new(),
            total_penalty: OrderedFloat(0.0),
        }
    }

    pub fn invalid(violations: Vec<PlanViolation>) -> Self {
        let penalty: f64 = violations.iter().map(|v| v.penalty.into_inner()).sum();
        Self {
            is_valid: false,
            step_evaluations: Vec::new(),
            violations,
            total_penalty: OrderedFloat(penalty),
        }
    }

    pub fn add_violation(&mut self, violation: PlanViolation) {
        self.total_penalty = OrderedFloat(
            self.total_penalty.into_inner() + violation.penalty.into_inner(),
        );
        self.violations.push(violation);
        self.is_valid = false;
    }
}

impl fmt::Display for PlanValidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid {
            write!(f, "Plan is VALID")
        } else {
            write!(
                f,
                "Plan is INVALID ({} violations, penalty={:.2})",
                self.violations.len(),
                self.total_penalty,
            )
        }
    }
}

/// A specific constraint violation in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanViolation {
    pub step_index: usize,
    pub constraint_id: String,
    pub message: String,
    pub penalty: OrderedFloat<f64>,
}

impl fmt::Display for PlanViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Step {}: [{}] {}",
            self.step_index, self.constraint_id, self.message,
        )
    }
}

// ─── DeploymentPlan ─────────────────────────────────────────────────────

/// A complete deployment plan: sequence of upgrade steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPlan {
    pub id: PlanId,
    pub start_state: ClusterState,
    pub target_state: ClusterState,
    pub steps: Vec<PlanStep>,
    pub cost: PlanCost,
    pub metadata: PlanMetadata,
}

impl DeploymentPlan {
    pub fn new(start: ClusterState, target: ClusterState) -> Self {
        Self {
            id: PlanId::generate(),
            start_state: start,
            target_state: target,
            steps: Vec::new(),
            cost: PlanCost::new(),
            metadata: PlanMetadata::new(),
        }
    }

    pub fn with_steps(mut self, steps: Vec<PlanStep>) -> Self {
        self.cost.total_steps = steps.len();
        self.cost.pnr_count = steps.iter().filter(|s| s.annotation.is_pnr).count();
        self.cost.total_risk = OrderedFloat(
            steps
                .iter()
                .map(|s| s.annotation.risk_score.into_inner())
                .sum(),
        );
        self.cost.estimated_duration_secs = steps
            .iter()
            .filter_map(|s| s.annotation.estimated_duration_secs)
            .sum();
        self.steps = steps;
        self
    }

    pub fn with_metadata(mut self, metadata: PlanMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// State after executing step at given index.
    pub fn state_after_step(&self, step_idx: usize) -> ClusterState {
        let mut state = self.start_state.clone();
        for step in &self.steps[..=step_idx] {
            state.set(step.service_idx, step.to_version);
        }
        state
    }

    /// All intermediate states (including start and target).
    pub fn all_states(&self) -> Vec<ClusterState> {
        let mut states = Vec::with_capacity(self.steps.len() + 1);
        let mut state = self.start_state.clone();
        states.push(state.clone());
        for step in &self.steps {
            state.set(step.service_idx, step.to_version);
            states.push(state.clone());
        }
        states
    }

    /// Check if this is a monotone plan (no service is ever downgraded).
    pub fn is_monotone(&self) -> bool {
        self.steps.iter().all(|s| s.is_upgrade() || s.from_version == s.to_version)
    }

    /// Services that are modified by this plan.
    pub fn affected_services(&self) -> Vec<usize> {
        let mut services: Vec<usize> = self.steps.iter().map(|s| s.service_idx).collect();
        services.sort_unstable();
        services.dedup();
        services
    }

    /// Steps that involve a specific service.
    pub fn steps_for_service(&self, service_idx: usize) -> Vec<&PlanStep> {
        self.steps
            .iter()
            .filter(|s| s.service_idx == service_idx)
            .collect()
    }

    /// Count of PNR states in this plan.
    pub fn pnr_count(&self) -> usize {
        self.steps.iter().filter(|s| s.annotation.is_pnr).count()
    }

    /// Maximum risk score among all steps.
    pub fn max_risk(&self) -> f64 {
        self.steps
            .iter()
            .map(|s| s.annotation.risk_score.into_inner())
            .fold(0.0f64, f64::max)
    }

    /// Validate that the plan is internally consistent.
    pub fn validate_consistency(&self) -> Result<()> {
        let mut state = self.start_state.clone();
        for (i, step) in self.steps.iter().enumerate() {
            if state.get(step.service_idx) != step.from_version {
                return Err(SafeStepError::plan_validation(format!(
                    "Step {}: expected service {} at version {}, found {}",
                    i,
                    step.service_idx,
                    step.from_version,
                    state.get(step.service_idx),
                )));
            }
            state.set(step.service_idx, step.to_version);
        }
        if state != self.target_state {
            return Err(SafeStepError::plan_validation(
                "Plan does not reach target state",
            ));
        }
        Ok(())
    }

    /// Compute cost from step annotations.
    pub fn recompute_cost(&mut self) {
        self.cost.total_steps = self.steps.len();
        self.cost.pnr_count = self.steps.iter().filter(|s| s.annotation.is_pnr).count();
        self.cost.total_risk = OrderedFloat(
            self.steps
                .iter()
                .map(|s| s.annotation.risk_score.into_inner())
                .sum(),
        );
        self.cost.estimated_duration_secs = self
            .steps
            .iter()
            .filter_map(|s| s.annotation.estimated_duration_secs)
            .sum();
    }
}

impl fmt::Display for DeploymentPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Plan {} ({} steps):", self.id, self.steps.len())?;
        writeln!(f, "  Start:  {}", self.start_state)?;
        writeln!(f, "  Target: {}", self.target_state)?;
        for step in &self.steps {
            writeln!(f, "  {}", step)?;
        }
        write!(f, "  {}", self.cost)
    }
}

// ─── PlanBuilder ─────────────────────────────────────────────────────────

/// Builder for constructing deployment plans step by step.
#[derive(Debug, Clone)]
pub struct PlanBuilder {
    start_state: ClusterState,
    target_state: ClusterState,
    current_state: ClusterState,
    steps: Vec<PlanStep>,
    metadata: PlanMetadata,
    service_names: Vec<String>,
}

impl PlanBuilder {
    pub fn new(
        start: ClusterState,
        target: ClusterState,
        service_names: Vec<String>,
    ) -> Self {
        let current = start.clone();
        Self {
            start_state: start,
            target_state: target,
            current_state: current,
            steps: Vec::new(),
            metadata: PlanMetadata::new(),
            service_names,
        }
    }

    pub fn current_state(&self) -> &ClusterState {
        &self.current_state
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Add a step to upgrade a service.
    pub fn add_step(
        &mut self,
        service_idx: usize,
        to_version: VersionIndex,
    ) -> Result<&PlanStep> {
        let from_version = self.current_state.get(service_idx);
        let name = self
            .service_names
            .get(service_idx)
            .cloned()
            .unwrap_or_else(|| format!("service-{}", service_idx));
        let step = PlanStep::new(
            self.steps.len(),
            service_idx,
            name,
            from_version,
            to_version,
        );
        self.current_state.set(service_idx, to_version);
        self.steps.push(step);
        Ok(self.steps.last().unwrap())
    }

    /// Add a step with a custom annotation.
    pub fn add_annotated_step(
        &mut self,
        service_idx: usize,
        to_version: VersionIndex,
        annotation: StepAnnotation,
    ) -> Result<&PlanStep> {
        let from_version = self.current_state.get(service_idx);
        let name = self
            .service_names
            .get(service_idx)
            .cloned()
            .unwrap_or_else(|| format!("service-{}", service_idx));
        let step = PlanStep::new(
            self.steps.len(),
            service_idx,
            name,
            from_version,
            to_version,
        )
        .with_annotation(annotation);
        self.current_state.set(service_idx, to_version);
        self.steps.push(step);
        Ok(self.steps.last().unwrap())
    }

    /// Undo the last step.
    pub fn undo_last(&mut self) -> Option<PlanStep> {
        if let Some(step) = self.steps.pop() {
            self.current_state
                .set(step.service_idx, step.from_version);
            Some(step)
        } else {
            None
        }
    }

    /// Check if the current state matches the target.
    pub fn is_complete(&self) -> bool {
        self.current_state == self.target_state
    }

    pub fn with_metadata(mut self, metadata: PlanMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Build the final plan.
    pub fn build(self) -> DeploymentPlan {
        DeploymentPlan::new(self.start_state, self.target_state)
            .with_steps(self.steps)
            .with_metadata(self.metadata)
    }

    /// Build and validate.
    pub fn build_validated(self) -> Result<DeploymentPlan> {
        let plan = self.build();
        plan.validate_consistency()?;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_states() -> (ClusterState, ClusterState) {
        let start = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        let target = ClusterState::new(&[VersionIndex(1), VersionIndex(2)]);
        (start, target)
    }

    #[test]
    fn test_plan_step() {
        let step = PlanStep::new(0, 0, "api", VersionIndex(0), VersionIndex(1));
        assert!(step.is_upgrade());
        assert!(!step.is_downgrade());
        let edge = step.to_edge();
        assert_eq!(edge.service_idx, 0);
    }

    #[test]
    fn test_plan_step_display() {
        let step = PlanStep::new(0, 0, "api", VersionIndex(0), VersionIndex(1))
            .with_labels("1.0.0", "2.0.0");
        let s = step.to_string();
        assert!(s.contains("api"));
        assert!(s.contains("1.0.0"));
        assert!(s.contains("2.0.0"));
    }

    #[test]
    fn test_step_annotation_safe() {
        let ann = StepAnnotation::safe();
        assert!(ann.in_envelope);
        assert!(!ann.is_pnr);
    }

    #[test]
    fn test_step_annotation_pnr() {
        let ann = StepAnnotation::pnr();
        assert!(ann.is_pnr);
        assert!(!ann.in_envelope);
    }

    #[test]
    fn test_rollback_info_safe() {
        let ri = RollbackInfo::safe(60);
        assert!(ri.can_rollback);
        assert!(ri.blocking_reason.is_none());
    }

    #[test]
    fn test_rollback_info_unsafe() {
        let ri = RollbackInfo::unsafe_rollback("schema migration");
        assert!(!ri.can_rollback);
        assert!(ri.blocking_reason.is_some());
    }

    #[test]
    fn test_plan_cost_dominates() {
        let c1 = PlanCost {
            total_steps: 3,
            total_risk: OrderedFloat(0.5),
            estimated_duration_secs: 100,
            pnr_count: 0,
            ..Default::default()
        };
        let c2 = PlanCost {
            total_steps: 4,
            total_risk: OrderedFloat(0.6),
            estimated_duration_secs: 120,
            pnr_count: 1,
            ..Default::default()
        };
        assert!(c1.dominates(&c2));
        assert!(!c2.dominates(&c1));
    }

    #[test]
    fn test_plan_cost_incomparable() {
        let c1 = PlanCost {
            total_steps: 3,
            total_risk: OrderedFloat(0.8),
            ..Default::default()
        };
        let c2 = PlanCost {
            total_steps: 5,
            total_risk: OrderedFloat(0.2),
            ..Default::default()
        };
        assert!(!c1.dominates(&c2));
        assert!(!c2.dominates(&c1));
    }

    #[test]
    fn test_plan_comparison() {
        let c1 = PlanCost {
            total_steps: 3,
            total_risk: OrderedFloat(0.5),
            ..Default::default()
        };
        let c2 = PlanCost {
            total_steps: 4,
            total_risk: OrderedFloat(0.6),
            ..Default::default()
        };
        assert_eq!(PlanComparison::compare(&c1, &c2), PlanComparison::FirstBetter);
    }

    #[test]
    fn test_plan_cost_weighted() {
        let cost = PlanCost {
            total_steps: 5,
            total_risk: OrderedFloat(0.5),
            ..Default::default()
        };
        let weights = CostWeights::default();
        let w = cost.weighted_cost(&weights);
        assert!(w > 0.0);
    }

    #[test]
    fn test_deployment_plan() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "svc-a", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(1, 1, "svc-b", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(2, 1, "svc-b", VersionIndex(1), VersionIndex(2)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        assert_eq!(plan.len(), 3);
        assert!(plan.is_monotone());
        assert_eq!(plan.affected_services(), vec![0, 1]);
    }

    #[test]
    fn test_plan_validate_consistency() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "svc-a", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(1, 1, "svc-b", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(2, 1, "svc-b", VersionIndex(1), VersionIndex(2)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        assert!(plan.validate_consistency().is_ok());
    }

    #[test]
    fn test_plan_validate_inconsistency() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "svc-a", VersionIndex(1), VersionIndex(2)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        assert!(plan.validate_consistency().is_err());
    }

    #[test]
    fn test_plan_all_states() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "svc-a", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(1, 1, "svc-b", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(2, 1, "svc-b", VersionIndex(1), VersionIndex(2)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        let states = plan.all_states();
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_plan_state_after_step() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "svc-a", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(1, 1, "svc-b", VersionIndex(0), VersionIndex(2)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        let s = plan.state_after_step(0);
        assert_eq!(s.get(0), VersionIndex(1));
        assert_eq!(s.get(1), VersionIndex(0));
    }

    #[test]
    fn test_plan_builder() {
        let (start, target) = make_states();
        let names = vec!["svc-a".into(), "svc-b".into()];
        let mut builder = PlanBuilder::new(start, target, names);

        builder.add_step(0, VersionIndex(1)).unwrap();
        builder.add_step(1, VersionIndex(1)).unwrap();
        builder.add_step(1, VersionIndex(2)).unwrap();

        assert!(builder.is_complete());
        let plan = builder.build_validated().unwrap();
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn test_plan_builder_undo() {
        let (start, target) = make_states();
        let names = vec!["svc-a".into(), "svc-b".into()];
        let mut builder = PlanBuilder::new(start, target, names);

        builder.add_step(0, VersionIndex(1)).unwrap();
        assert_eq!(builder.step_count(), 1);

        let undone = builder.undo_last();
        assert!(undone.is_some());
        assert_eq!(builder.step_count(), 0);
        assert_eq!(builder.current_state().get(0), VersionIndex(0));
    }

    #[test]
    fn test_plan_validation_valid() {
        let pv = PlanValidation::valid();
        assert!(pv.is_valid);
        assert!(pv.violations.is_empty());
    }

    #[test]
    fn test_plan_validation_invalid() {
        let violations = vec![PlanViolation {
            step_index: 0,
            constraint_id: "c1".into(),
            message: "bad".into(),
            penalty: OrderedFloat(5.0),
        }];
        let pv = PlanValidation::invalid(violations);
        assert!(!pv.is_valid);
        assert_eq!(pv.violations.len(), 1);
        assert!((pv.total_penalty.into_inner() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_metadata() {
        let meta = PlanMetadata::new()
            .with_bmc_depth(10)
            .with_encoding("interval")
            .with_solver_iterations(42)
            .with_optimal(true)
            .with_note("test note");
        assert_eq!(meta.bmc_depth, 10);
        assert_eq!(meta.solver_iterations, 42);
        assert!(meta.is_optimal);
    }

    #[test]
    fn test_plan_steps_for_service() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "a", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(1, 1, "b", VersionIndex(0), VersionIndex(1)),
            PlanStep::new(2, 1, "b", VersionIndex(1), VersionIndex(2)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        assert_eq!(plan.steps_for_service(0).len(), 1);
        assert_eq!(plan.steps_for_service(1).len(), 2);
    }

    #[test]
    fn test_plan_max_risk() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "a", VersionIndex(0), VersionIndex(1))
                .with_annotation(StepAnnotation::new().with_risk(0.3)),
            PlanStep::new(1, 1, "b", VersionIndex(0), VersionIndex(2))
                .with_annotation(StepAnnotation::new().with_risk(0.7)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        assert!((plan.max_risk() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_display() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "api", VersionIndex(0), VersionIndex(1)),
        ];
        let plan = DeploymentPlan::new(start, target).with_steps(steps);
        let s = plan.to_string();
        assert!(s.contains("Plan"));
        assert!(s.contains("api"));
    }

    #[test]
    fn test_plan_recompute_cost() {
        let (start, target) = make_states();
        let steps = vec![
            PlanStep::new(0, 0, "a", VersionIndex(0), VersionIndex(1))
                .with_annotation(StepAnnotation::pnr().with_duration(30)),
        ];
        let mut plan = DeploymentPlan::new(start, target).with_steps(steps);
        plan.recompute_cost();
        assert_eq!(plan.cost.pnr_count, 1);
        assert_eq!(plan.cost.estimated_duration_secs, 30);
    }
}
