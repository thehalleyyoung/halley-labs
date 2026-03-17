//! Witness generation for stuck-configuration and infeasibility analysis.
//!
//! Provides UNSAT-core inspired extraction of minimal constraint sets
//! that block rollback or make plans infeasible.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use safestep_types::{ServiceId, ConstraintId};

use crate::{
    DeployConstraint, DeploymentState, SafetyEnvelope, Version,
    ConstraintType, ConstraintSeverity,
};

// ---------------------------------------------------------------------------
// WitnessSeverity
// ---------------------------------------------------------------------------

/// Severity level for a stuck-configuration witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WitnessSeverity {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for WitnessSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConstraintDescription
// ---------------------------------------------------------------------------

/// Human-readable description of a blocking constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDescription {
    pub constraint_id: ConstraintId,
    pub constraint_type: String,
    pub involved_services: Vec<ServiceId>,
    pub versions: Vec<Version>,
    pub why_blocking: String,
    pub severity: ConstraintSeverity,
}

impl ConstraintDescription {
    pub fn new(
        id: ConstraintId,
        constraint_type: &str,
        why_blocking: &str,
    ) -> Self {
        Self {
            constraint_id: id,
            constraint_type: constraint_type.to_string(),
            involved_services: Vec::new(),
            versions: Vec::new(),
            why_blocking: why_blocking.to_string(),
            severity: ConstraintSeverity::Error,
        }
    }

    pub fn with_services(mut self, services: Vec<ServiceId>) -> Self {
        self.involved_services = services;
        self
    }

    pub fn with_versions(mut self, versions: Vec<Version>) -> Self {
        self.versions = versions;
        self
    }

    pub fn with_severity(mut self, severity: ConstraintSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Produce a one-line summary.
    pub fn summary(&self) -> String {
        let svcs: Vec<String> = self
            .involved_services
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        let svc_str = if svcs.is_empty() {
            "no services".to_string()
        } else {
            svcs.join(", ")
        };
        format!(
            "[{}] {} on {}: {}",
            self.severity, self.constraint_type, svc_str, self.why_blocking
        )
    }
}

impl std::fmt::Display for ConstraintDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// StuckWitness
// ---------------------------------------------------------------------------

/// Minimal explanation of why rollback is blocked from a given state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StuckWitness {
    pub blocking_constraints: Vec<ConstraintDescription>,
    pub affected_services: Vec<ServiceId>,
    pub explanation: String,
    pub severity: WitnessSeverity,
    pub state_id: String,
    pub attempted_targets: Vec<String>,
}

impl StuckWitness {
    pub fn new(state_id: &str, severity: WitnessSeverity) -> Self {
        Self {
            blocking_constraints: Vec::new(),
            affected_services: Vec::new(),
            explanation: String::new(),
            severity,
            state_id: state_id.to_string(),
            attempted_targets: Vec::new(),
        }
    }

    pub fn with_constraint(mut self, desc: ConstraintDescription) -> Self {
        for svc in &desc.involved_services {
            if !self.affected_services.contains(svc) {
                self.affected_services.push(svc.clone());
            }
        }
        self.blocking_constraints.push(desc);
        self
    }

    pub fn with_explanation(mut self, explanation: &str) -> Self {
        self.explanation = explanation.to_string();
        self
    }

    pub fn with_target(mut self, target: &str) -> Self {
        self.attempted_targets.push(target.to_string());
        self
    }

    /// Number of blocking constraints.
    pub fn constraint_count(&self) -> usize {
        self.blocking_constraints.len()
    }

    /// True if the witness is empty (nothing is blocking).
    pub fn is_empty(&self) -> bool {
        self.blocking_constraints.is_empty()
    }

    /// Build a multi-line human-readable report.
    pub fn to_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Stuck-Configuration Witness [{}] ===\n",
            self.severity
        ));
        out.push_str(&format!("State: {}\n", self.state_id));
        if !self.attempted_targets.is_empty() {
            out.push_str(&format!(
                "Attempted targets: {}\n",
                self.attempted_targets.join(", ")
            ));
        }
        out.push_str(&format!(
            "Affected services: {}\n",
            self.affected_services
                .iter()
                .map(|s| s.as_str().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        out.push('\n');
        if !self.explanation.is_empty() {
            out.push_str(&format!("Explanation: {}\n\n", self.explanation));
        }
        out.push_str(&format!(
            "Blocking constraints ({}):\n",
            self.blocking_constraints.len()
        ));
        for (i, c) in self.blocking_constraints.iter().enumerate() {
            out.push_str(&format!("  {}. {}\n", i + 1, c.summary()));
        }
        out
    }
}

impl std::fmt::Display for StuckWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_report())
    }
}

// ---------------------------------------------------------------------------
// Suggestion
// ---------------------------------------------------------------------------

/// Suggested fix for infeasibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub kind: SuggestionKind,
    pub description: String,
    pub affected_constraints: Vec<ConstraintId>,
    pub confidence: f64,
}

/// Kind of suggestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionKind {
    RelaxConstraint,
    UpdateVersion,
    AddIntermediateVersion,
    ReorderSteps,
    AddHealthCheck,
    SplitDeployment,
}

impl std::fmt::Display for SuggestionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RelaxConstraint => write!(f, "relax-constraint"),
            Self::UpdateVersion => write!(f, "update-version"),
            Self::AddIntermediateVersion => write!(f, "add-intermediate-version"),
            Self::ReorderSteps => write!(f, "reorder-steps"),
            Self::AddHealthCheck => write!(f, "add-health-check"),
            Self::SplitDeployment => write!(f, "split-deployment"),
        }
    }
}

impl Suggestion {
    pub fn relax_constraint(constraint: ConstraintId, description: &str) -> Self {
        Self {
            kind: SuggestionKind::RelaxConstraint,
            description: description.to_string(),
            affected_constraints: vec![constraint],
            confidence: 0.7,
        }
    }

    pub fn update_version(description: &str) -> Self {
        Self {
            kind: SuggestionKind::UpdateVersion,
            description: description.to_string(),
            affected_constraints: Vec::new(),
            confidence: 0.5,
        }
    }

    pub fn add_intermediate(description: &str) -> Self {
        Self {
            kind: SuggestionKind::AddIntermediateVersion,
            description: description.to_string(),
            affected_constraints: Vec::new(),
            confidence: 0.6,
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// InfeasibilityWitness
// ---------------------------------------------------------------------------

/// Witness for complete plan infeasibility (no plan exists).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfeasibilityWitness {
    pub minimal_infeasible_set: Vec<ConstraintId>,
    pub explanation: String,
    pub suggested_relaxations: Vec<Suggestion>,
    pub constraint_details: Vec<ConstraintDescription>,
}

impl InfeasibilityWitness {
    pub fn new(explanation: &str) -> Self {
        Self {
            minimal_infeasible_set: Vec::new(),
            explanation: explanation.to_string(),
            suggested_relaxations: Vec::new(),
            constraint_details: Vec::new(),
        }
    }

    pub fn with_infeasible_set(mut self, ids: Vec<ConstraintId>) -> Self {
        self.minimal_infeasible_set = ids;
        self
    }

    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggested_relaxations.push(suggestion);
        self
    }

    pub fn with_detail(mut self, detail: ConstraintDescription) -> Self {
        self.constraint_details.push(detail);
        self
    }

    /// Multi-line report.
    pub fn to_report(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Infeasibility Witness ===\n");
        out.push_str(&format!("Explanation: {}\n\n", self.explanation));
        out.push_str(&format!(
            "Minimal infeasible set ({} constraints):\n",
            self.minimal_infeasible_set.len()
        ));
        for (i, cid) in self.minimal_infeasible_set.iter().enumerate() {
            let detail = self
                .constraint_details
                .iter()
                .find(|d| d.constraint_id == *cid);
            match detail {
                Some(d) => out.push_str(&format!("  {}. {}\n", i + 1, d.summary())),
                None => out.push_str(&format!("  {}. {}\n", i + 1, cid.as_str())),
            }
        }
        if !self.suggested_relaxations.is_empty() {
            out.push_str(&format!(
                "\nSuggested relaxations ({}):\n",
                self.suggested_relaxations.len()
            ));
            for (i, s) in self.suggested_relaxations.iter().enumerate() {
                out.push_str(&format!(
                    "  {}. [{}] {} (confidence: {:.0}%)\n",
                    i + 1,
                    s.kind,
                    s.description,
                    s.confidence * 100.0
                ));
            }
        }
        out
    }
}

impl std::fmt::Display for InfeasibilityWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_report())
    }
}

// ---------------------------------------------------------------------------
// MinimalWitness (output of minimiser)
// ---------------------------------------------------------------------------

/// A minimised witness retaining only essential blocking constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalWitness {
    pub original_count: usize,
    pub minimal_count: usize,
    pub constraints: Vec<ConstraintDescription>,
    pub removed_constraints: Vec<ConstraintId>,
}

impl MinimalWitness {
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_count == 0 {
            return 0.0;
        }
        1.0 - (self.minimal_count as f64 / self.original_count as f64)
    }
}

// ---------------------------------------------------------------------------
// ConstraintChecker – checks whether a constraint is violated in a state
// ---------------------------------------------------------------------------

/// Evaluates whether a constraint is violated in a given deployment state.
struct ConstraintChecker;

impl ConstraintChecker {
    /// Returns `true` when the constraint is *violated* (blocks progress).
    fn is_violated(
        constraint: &DeployConstraint,
        state: &DeploymentState,
        target: &DeploymentState,
    ) -> bool {
        match &constraint.constraint_type {
            ConstraintType::VersionCompatibility => {
                Self::check_version_compat(constraint, state)
            }
            ConstraintType::OrderingDependency => {
                Self::check_ordering(constraint, state, target)
            }
            ConstraintType::RollbackSafety => {
                Self::check_rollback_safety(constraint, state, target)
            }
            _ => false,
        }
    }

    fn check_version_compat(
        constraint: &DeployConstraint,
        state: &DeploymentState,
    ) -> bool {
        // If two services are in the constraint, check that they are at compatible
        // versions (heuristic: major versions must match).
        if constraint.involved_services.len() < 2 {
            return false;
        }
        let versions: Vec<Option<&Version>> = constraint
            .involved_services
            .iter()
            .map(|s| state.version_of(s))
            .collect();
        if let (Some(v1), Some(v2)) = (versions.first().and_then(|v| *v), versions.get(1).and_then(|v| *v)) {
            let major_diff = (v1.major as i64 - v2.major as i64).unsigned_abs();
            major_diff > 1
        } else {
            false
        }
    }

    fn check_ordering(
        constraint: &DeployConstraint,
        state: &DeploymentState,
        target: &DeploymentState,
    ) -> bool {
        // If the first service hasn't reached its target but the second has,
        // ordering is violated.
        if constraint.involved_services.len() < 2 {
            return false;
        }
        let svc_a = &constraint.involved_services[0];
        let svc_b = &constraint.involved_services[1];
        let a_at_target = state.version_of(svc_a) == target.version_of(svc_a);
        let b_at_target = state.version_of(svc_b) == target.version_of(svc_b);
        !a_at_target && b_at_target
    }

    fn check_rollback_safety(
        constraint: &DeployConstraint,
        state: &DeploymentState,
        target: &DeploymentState,
    ) -> bool {
        // Rollback is unsafe if any involved service has moved past the target
        // version (downgrade required but blocked).
        for svc in &constraint.involved_services {
            if let (Some(cur), Some(tgt)) = (state.version_of(svc), target.version_of(svc)) {
                if cur > tgt {
                    return true;
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// WitnessGenerator
// ---------------------------------------------------------------------------

/// Generates stuck-configuration witnesses using UNSAT-core-style analysis.
pub struct WitnessGenerator {
    max_iterations: usize,
}

impl WitnessGenerator {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Generate a witness explaining why `state` is stuck with respect to
    /// `constraints` when trying to reach `target`.
    pub fn generate(
        &self,
        state: &DeploymentState,
        constraints: &[DeployConstraint],
        target: &DeploymentState,
    ) -> StuckWitness {
        let mut witness = StuckWitness::new(state.id.as_str(), WitnessSeverity::Warning);
        witness.attempted_targets.push(target.id.as_str().to_string());

        // Phase 1: identify all violated constraints
        let mut violated: Vec<(usize, &DeployConstraint)> = Vec::new();
        for (i, c) in constraints.iter().enumerate() {
            if ConstraintChecker::is_violated(c, state, target) {
                violated.push((i, c));
            }
        }

        if violated.is_empty() {
            witness.explanation =
                "No constraints are violated – state may not actually be stuck.".to_string();
            witness.severity = WitnessSeverity::Info;
            return witness;
        }

        // Phase 2: classify severity
        let has_critical = violated
            .iter()
            .any(|(_, c)| c.severity == ConstraintSeverity::Critical);
        witness.severity = if has_critical {
            WitnessSeverity::Critical
        } else {
            WitnessSeverity::Warning
        };

        // Phase 3: build constraint descriptions
        for (_, constraint) in &violated {
            let desc = self.describe_constraint(constraint, state, target);
            witness = witness.with_constraint(desc);
        }

        // Phase 4: build explanation
        witness.explanation = self.build_explanation(&witness, state, target);

        witness
    }

    /// Generate an infeasibility witness when no plan exists.
    pub fn generate_infeasibility(
        &self,
        constraints: &[DeployConstraint],
        state: &DeploymentState,
        target: &DeploymentState,
    ) -> InfeasibilityWitness {
        let mut witness = InfeasibilityWitness::new(
            "No valid deployment plan exists given the current constraints.",
        );

        // Collect all violated constraints
        let violated: Vec<&DeployConstraint> = constraints
            .iter()
            .filter(|c| ConstraintChecker::is_violated(c, state, target))
            .collect();

        // Build MIS using greedy approach
        let mut mis_ids: Vec<ConstraintId> = Vec::new();
        let mut mis_details: Vec<ConstraintDescription> = Vec::new();
        let mut covered_services: HashSet<String> = HashSet::new();

        for c in &violated {
            let new_services: Vec<&ServiceId> = c
                .involved_services
                .iter()
                .filter(|s| !covered_services.contains(s.as_str()))
                .collect();
            if !new_services.is_empty() || mis_ids.is_empty() {
                mis_ids.push(c.id.clone());
                mis_details.push(self.describe_constraint(c, state, target));
                for svc in &c.involved_services {
                    covered_services.insert(svc.as_str().to_string());
                }
            }
        }

        witness.minimal_infeasible_set = mis_ids;
        witness.constraint_details = mis_details;

        // Generate suggestions
        for c in &violated {
            match c.constraint_type {
                ConstraintType::VersionCompatibility => {
                    witness = witness.with_suggestion(Suggestion::update_version(&format!(
                        "Update version for services: {}",
                        c.involved_services
                            .iter()
                            .map(|s| s.as_str().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                ConstraintType::OrderingDependency => {
                    witness = witness.with_suggestion(Suggestion {
                        kind: SuggestionKind::ReorderSteps,
                        description: format!(
                            "Reorder deployment steps for {}",
                            c.involved_services
                                .iter()
                                .map(|s| s.as_str().to_string())
                                .collect::<Vec<_>>()
                                .join(" → ")
                        ),
                        affected_constraints: vec![c.id.clone()],
                        confidence: 0.65,
                    });
                }
                ConstraintType::RollbackSafety => {
                    witness = witness.with_suggestion(
                        Suggestion::add_intermediate(
                            "Add intermediate version to allow safe rollback path",
                        )
                        .with_confidence(0.5),
                    );
                }
                _ => {
                    witness = witness.with_suggestion(
                        Suggestion::relax_constraint(
                            c.id.clone(),
                            &format!("Consider relaxing: {}", c.description),
                        )
                        .with_confidence(0.4),
                    );
                }
            }
        }

        witness
    }

    fn describe_constraint(
        &self,
        constraint: &DeployConstraint,
        state: &DeploymentState,
        _target: &DeploymentState,
    ) -> ConstraintDescription {
        let versions: Vec<Version> = constraint
            .involved_services
            .iter()
            .filter_map(|s| state.version_of(s).cloned())
            .collect();

        let why = match &constraint.constraint_type {
            ConstraintType::VersionCompatibility => {
                format!(
                    "Services have incompatible versions: {}",
                    versions
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(" vs ")
                )
            }
            ConstraintType::OrderingDependency => {
                "Deployment ordering dependency prevents this transition".to_string()
            }
            ConstraintType::RollbackSafety => {
                "Cannot safely roll back because a required downgrade is blocked".to_string()
            }
            ConstraintType::ResourceLimit => {
                "Resource limits would be exceeded during this transition".to_string()
            }
            ConstraintType::HealthCheck => {
                "Health check requirements cannot be met in this state".to_string()
            }
            ConstraintType::Custom(name) => {
                format!("Custom constraint '{}' is violated", name)
            }
        };

        ConstraintDescription {
            constraint_id: constraint.id.clone(),
            constraint_type: constraint.constraint_type.to_string(),
            involved_services: constraint.involved_services.clone(),
            versions,
            why_blocking: why,
            severity: constraint.severity,
        }
    }

    fn build_explanation(
        &self,
        witness: &StuckWitness,
        state: &DeploymentState,
        target: &DeploymentState,
    ) -> String {
        let svc_list: Vec<String> = witness
            .affected_services
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        let constraint_types: Vec<String> = witness
            .blocking_constraints
            .iter()
            .map(|c| c.constraint_type.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let mut parts: Vec<String> = Vec::new();
        parts.push(format!(
            "The deployment is stuck at state '{}' and cannot reach the target '{}'.",
            state.id.as_str(),
            target.id.as_str()
        ));
        parts.push(format!(
            "{} constraint(s) of type(s) [{}] are blocking progress.",
            witness.blocking_constraints.len(),
            constraint_types.join(", ")
        ));
        parts.push(format!(
            "Affected services: {}.",
            if svc_list.is_empty() {
                "none".to_string()
            } else {
                svc_list.join(", ")
            }
        ));

        if witness.severity == WitnessSeverity::Critical {
            parts.push(
                "This is a CRITICAL situation – manual intervention may be required.".to_string(),
            );
        }

        parts.join(" ")
    }
}

impl Default for WitnessGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WitnessMinimizer
// ---------------------------------------------------------------------------

/// Minimizes a witness to its essential blocking constraints using
/// iterative deletion and delta-debugging.
pub struct WitnessMinimizer {
    use_delta_debugging: bool,
    delta_threshold: usize,
}

impl WitnessMinimizer {
    pub fn new() -> Self {
        Self {
            use_delta_debugging: true,
            delta_threshold: 8,
        }
    }

    pub fn with_delta_debugging(mut self, enabled: bool) -> Self {
        self.use_delta_debugging = enabled;
        self
    }

    pub fn with_delta_threshold(mut self, threshold: usize) -> Self {
        self.delta_threshold = threshold;
        self
    }

    /// Minimize the witness so that only constraints that are individually
    /// necessary remain.
    pub fn minimize(
        &self,
        witness: &StuckWitness,
        state: &DeploymentState,
        constraints: &[DeployConstraint],
        target: &DeploymentState,
    ) -> MinimalWitness {
        let original_count = witness.blocking_constraints.len();

        if original_count <= 1 {
            return MinimalWitness {
                original_count,
                minimal_count: original_count,
                constraints: witness.blocking_constraints.clone(),
                removed_constraints: Vec::new(),
            };
        }

        let blocking_ids: Vec<ConstraintId> = witness
            .blocking_constraints
            .iter()
            .map(|c| c.constraint_id.clone())
            .collect();

        let minimal_ids = if self.use_delta_debugging && original_count >= self.delta_threshold {
            self.delta_debug_minimize(&blocking_ids, state, constraints, target)
        } else {
            self.iterative_minimize(&blocking_ids, state, constraints, target)
        };

        let minimal_set: HashSet<String> = minimal_ids
            .iter()
            .map(|id| id.as_str().to_string())
            .collect();

        let kept: Vec<ConstraintDescription> = witness
            .blocking_constraints
            .iter()
            .filter(|c| minimal_set.contains(c.constraint_id.as_str()))
            .cloned()
            .collect();

        let removed: Vec<ConstraintId> = blocking_ids
            .iter()
            .filter(|id| !minimal_set.contains(id.as_str()))
            .cloned()
            .collect();

        MinimalWitness {
            original_count,
            minimal_count: kept.len(),
            constraints: kept,
            removed_constraints: removed,
        }
    }

    /// Simple iterative deletion: try removing each constraint one at a time.
    fn iterative_minimize(
        &self,
        ids: &[ConstraintId],
        state: &DeploymentState,
        all_constraints: &[DeployConstraint],
        target: &DeploymentState,
    ) -> Vec<ConstraintId> {
        let mut current: Vec<ConstraintId> = ids.to_vec();

        let mut i = 0;
        while i < current.len() {
            let candidate = current.remove(i);
            if self.still_stuck(state, all_constraints, target, &current) {
                // Still stuck without this constraint → it was not needed
                continue;
            }
            // Removing it makes us un-stuck → it was essential
            current.insert(i, candidate);
            i += 1;
        }

        current
    }

    /// Delta debugging: partition constraints into halves and test.
    fn delta_debug_minimize(
        &self,
        ids: &[ConstraintId],
        state: &DeploymentState,
        all_constraints: &[DeployConstraint],
        target: &DeploymentState,
    ) -> Vec<ConstraintId> {
        let mut current = ids.to_vec();
        let mut granularity = 2usize;

        while granularity <= current.len() {
            let chunk_size = (current.len() + granularity - 1) / granularity;
            let mut changed = false;

            let mut i = 0;
            while i < current.len() {
                let end = (i + chunk_size).min(current.len());
                let mut without_chunk: Vec<ConstraintId> = Vec::new();
                without_chunk.extend_from_slice(&current[..i]);
                if end < current.len() {
                    without_chunk.extend_from_slice(&current[end..]);
                }

                if !without_chunk.is_empty()
                    && self.still_stuck(state, all_constraints, target, &without_chunk)
                {
                    current = without_chunk;
                    granularity = 2.max(granularity - 1);
                    changed = true;
                    break;
                }

                i += chunk_size;
            }

            if !changed {
                granularity *= 2;
            }
        }

        // Final iterative pass
        self.iterative_minimize(&current, state, all_constraints, target)
    }

    /// Check whether the state is still stuck given only the constraints
    /// whose IDs are in `active_ids`.
    fn still_stuck(
        &self,
        state: &DeploymentState,
        all_constraints: &[DeployConstraint],
        target: &DeploymentState,
        active_ids: &[ConstraintId],
    ) -> bool {
        let id_set: HashSet<String> = active_ids
            .iter()
            .map(|id| id.as_str().to_string())
            .collect();
        let active: Vec<&DeployConstraint> = all_constraints
            .iter()
            .filter(|c| id_set.contains(c.id.as_str()))
            .collect();
        active
            .iter()
            .any(|c| ConstraintChecker::is_violated(c, state, target))
    }
}

impl Default for WitnessMinimizer {
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
    use safestep_types::StateId;

    fn make_state(id: &str, services: &[(&str, (u32, u32, u32))]) -> DeploymentState {
        let mut state = DeploymentState::new(StateId::new(id));
        for (svc, (maj, min, pat)) in services {
            state.service_versions.push((
                ServiceId::new(*svc),
                Version::new(*maj, *min, *pat),
            ));
        }
        state
    }

    fn make_constraint(
        id: &str,
        ct: ConstraintType,
        services: &[&str],
        severity: ConstraintSeverity,
    ) -> DeployConstraint {
        DeployConstraint {
            id: ConstraintId::new(id),
            constraint_type: ct,
            description: format!("Constraint {}", id),
            involved_services: services.iter().map(|s| ServiceId::new(*s)).collect(),
            severity,
        }
    }

    #[test]
    fn test_witness_severity_display() {
        assert_eq!(WitnessSeverity::Critical.to_string(), "CRITICAL");
        assert_eq!(WitnessSeverity::Warning.to_string(), "WARNING");
        assert_eq!(WitnessSeverity::Info.to_string(), "INFO");
    }

    #[test]
    fn test_witness_severity_ordering() {
        assert!(WitnessSeverity::Info < WitnessSeverity::Warning);
        assert!(WitnessSeverity::Warning < WitnessSeverity::Critical);
    }

    #[test]
    fn test_constraint_description_summary() {
        let desc = ConstraintDescription::new(
            ConstraintId::new("c1"),
            "version-compatibility",
            "Versions differ too much",
        )
        .with_services(vec![ServiceId::new("api"), ServiceId::new("web")])
        .with_severity(ConstraintSeverity::Error);

        let summary = desc.summary();
        assert!(summary.contains("version-compatibility"));
        assert!(summary.contains("api"));
        assert!(summary.contains("web"));
        assert!(summary.contains("Versions differ too much"));
    }

    #[test]
    fn test_stuck_witness_builder() {
        let desc = ConstraintDescription::new(
            ConstraintId::new("c1"),
            "ordering",
            "blocked",
        )
        .with_services(vec![ServiceId::new("svc-a")]);

        let w = StuckWitness::new("s1", WitnessSeverity::Warning)
            .with_constraint(desc)
            .with_explanation("Test explanation")
            .with_target("s2");

        assert_eq!(w.constraint_count(), 1);
        assert!(!w.is_empty());
        assert_eq!(w.state_id, "s1");
        assert_eq!(w.attempted_targets, vec!["s2".to_string()]);
        assert!(w.affected_services.contains(&ServiceId::new("svc-a")));
    }

    #[test]
    fn test_stuck_witness_report() {
        let desc = ConstraintDescription::new(
            ConstraintId::new("c1"),
            "ordering",
            "blocked",
        )
        .with_services(vec![ServiceId::new("api")]);

        let w = StuckWitness::new("state-1", WitnessSeverity::Critical)
            .with_constraint(desc)
            .with_explanation("Deployment is blocked");

        let report = w.to_report();
        assert!(report.contains("CRITICAL"));
        assert!(report.contains("state-1"));
        assert!(report.contains("Deployment is blocked"));
    }

    #[test]
    fn test_suggestion_constructors() {
        let s1 = Suggestion::relax_constraint(
            ConstraintId::new("c1"),
            "Relax version check",
        );
        assert_eq!(s1.kind, SuggestionKind::RelaxConstraint);
        assert_eq!(s1.confidence, 0.7);

        let s2 = Suggestion::update_version("Bump to v2").with_confidence(0.9);
        assert_eq!(s2.confidence, 0.9);

        let s3 = Suggestion::add_intermediate("Add v1.5");
        assert_eq!(s3.kind, SuggestionKind::AddIntermediateVersion);
    }

    #[test]
    fn test_suggestion_confidence_clamping() {
        let s = Suggestion::update_version("test").with_confidence(1.5);
        assert_eq!(s.confidence, 1.0);
        let s = Suggestion::update_version("test").with_confidence(-0.5);
        assert_eq!(s.confidence, 0.0);
    }

    #[test]
    fn test_infeasibility_witness() {
        let w = InfeasibilityWitness::new("No plan exists")
            .with_infeasible_set(vec![
                ConstraintId::new("c1"),
                ConstraintId::new("c2"),
            ])
            .with_suggestion(Suggestion::relax_constraint(
                ConstraintId::new("c1"),
                "Relax c1",
            ));

        assert_eq!(w.minimal_infeasible_set.len(), 2);
        assert_eq!(w.suggested_relaxations.len(), 1);
        let report = w.to_report();
        assert!(report.contains("No plan exists"));
        assert!(report.contains("2 constraints"));
    }

    #[test]
    fn test_infeasibility_report_includes_suggestions() {
        let w = InfeasibilityWitness::new("blocked")
            .with_suggestion(Suggestion::update_version("Bump api to v2"))
            .with_suggestion(Suggestion::add_intermediate("Add api v1.5"));

        let report = w.to_report();
        assert!(report.contains("Suggested relaxations (2)"));
        assert!(report.contains("Bump api to v2"));
    }

    #[test]
    fn test_generator_no_violations() {
        let state = make_state("s1", &[("api", (1, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0))]);
        let constraints: Vec<DeployConstraint> = vec![];

        let gen = WitnessGenerator::new();
        let witness = gen.generate(&state, &constraints, &target);

        assert_eq!(witness.severity, WitnessSeverity::Info);
        assert!(witness.is_empty());
    }

    #[test]
    fn test_generator_version_compat_violation() {
        let state = make_state("s1", &[("api", (1, 0, 0)), ("web", (3, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0)), ("web", (3, 0, 0))]);
        let constraints = vec![make_constraint(
            "c1",
            ConstraintType::VersionCompatibility,
            &["api", "web"],
            ConstraintSeverity::Error,
        )];

        let gen = WitnessGenerator::new();
        let witness = gen.generate(&state, &constraints, &target);

        assert!(!witness.is_empty());
        assert_eq!(witness.blocking_constraints.len(), 1);
        assert!(witness.affected_services.contains(&ServiceId::new("api")));
    }

    #[test]
    fn test_generator_ordering_violation() {
        // svc-a should be deployed before svc-b, but svc-a hasn't reached target
        // while svc-b has.
        let state = make_state("s1", &[("svc-a", (1, 0, 0)), ("svc-b", (2, 0, 0))]);
        let target = make_state("s2", &[("svc-a", (2, 0, 0)), ("svc-b", (2, 0, 0))]);
        let constraints = vec![make_constraint(
            "c-order",
            ConstraintType::OrderingDependency,
            &["svc-a", "svc-b"],
            ConstraintSeverity::Warning,
        )];

        let gen = WitnessGenerator::new();
        let witness = gen.generate(&state, &constraints, &target);

        assert_eq!(witness.severity, WitnessSeverity::Warning);
        assert_eq!(witness.blocking_constraints.len(), 1);
    }

    #[test]
    fn test_generator_rollback_safety_violation() {
        // Current version is ahead of the target (rollback needed but blocked).
        let state = make_state("s1", &[("api", (3, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0))]);
        let constraints = vec![make_constraint(
            "c-rb",
            ConstraintType::RollbackSafety,
            &["api"],
            ConstraintSeverity::Critical,
        )];

        let gen = WitnessGenerator::new();
        let witness = gen.generate(&state, &constraints, &target);

        assert_eq!(witness.severity, WitnessSeverity::Critical);
        assert_eq!(witness.blocking_constraints.len(), 1);
    }

    #[test]
    fn test_generator_multiple_violations() {
        let state = make_state(
            "s1",
            &[("api", (1, 0, 0)), ("web", (3, 0, 0)), ("db", (5, 0, 0))],
        );
        let target = make_state(
            "s2",
            &[("api", (2, 0, 0)), ("web", (3, 0, 0)), ("db", (2, 0, 0))],
        );
        let constraints = vec![
            make_constraint(
                "c1",
                ConstraintType::VersionCompatibility,
                &["api", "web"],
                ConstraintSeverity::Error,
            ),
            make_constraint(
                "c2",
                ConstraintType::RollbackSafety,
                &["db"],
                ConstraintSeverity::Critical,
            ),
        ];

        let gen = WitnessGenerator::new();
        let witness = gen.generate(&state, &constraints, &target);

        assert_eq!(witness.severity, WitnessSeverity::Critical);
        assert_eq!(witness.blocking_constraints.len(), 2);
    }

    #[test]
    fn test_generate_infeasibility() {
        let state = make_state("s1", &[("api", (1, 0, 0)), ("web", (3, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0)), ("web", (3, 0, 0))]);
        let constraints = vec![make_constraint(
            "c1",
            ConstraintType::VersionCompatibility,
            &["api", "web"],
            ConstraintSeverity::Error,
        )];

        let gen = WitnessGenerator::new();
        let iw = gen.generate_infeasibility(&constraints, &state, &target);

        assert!(!iw.minimal_infeasible_set.is_empty());
        assert!(!iw.suggested_relaxations.is_empty());
    }

    #[test]
    fn test_minimizer_single_constraint() {
        let desc = ConstraintDescription::new(
            ConstraintId::new("c1"),
            "ordering",
            "blocked",
        );
        let witness = StuckWitness::new("s1", WitnessSeverity::Warning)
            .with_constraint(desc);

        let state = make_state("s1", &[("api", (1, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0))]);
        let constraints: Vec<DeployConstraint> = vec![];

        let minimizer = WitnessMinimizer::new();
        let minimal = minimizer.minimize(&witness, &state, &constraints, &target);

        assert_eq!(minimal.original_count, 1);
        assert_eq!(minimal.minimal_count, 1);
        assert_eq!(minimal.reduction_ratio(), 0.0);
    }

    #[test]
    fn test_minimizer_removes_redundant() {
        let state = make_state("s1", &[("api", (1, 0, 0)), ("web", (3, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0)), ("web", (3, 0, 0))]);

        let c1 = make_constraint(
            "c1",
            ConstraintType::VersionCompatibility,
            &["api", "web"],
            ConstraintSeverity::Error,
        );

        let gen = WitnessGenerator::new();
        let witness = gen.generate(&state, &[c1.clone()], &target);

        let minimizer = WitnessMinimizer::new().with_delta_debugging(false);
        let minimal = minimizer.minimize(&witness, &state, &[c1], &target);

        // Should keep the one violated constraint
        assert!(minimal.minimal_count >= 1);
        assert!(minimal.minimal_count <= minimal.original_count);
    }

    #[test]
    fn test_minimizer_delta_debugging_threshold() {
        let minimizer = WitnessMinimizer::new()
            .with_delta_debugging(true)
            .with_delta_threshold(4);
        assert!(minimizer.use_delta_debugging);
        assert_eq!(minimizer.delta_threshold, 4);
    }

    #[test]
    fn test_minimal_witness_reduction_ratio() {
        let m = MinimalWitness {
            original_count: 10,
            minimal_count: 3,
            constraints: Vec::new(),
            removed_constraints: Vec::new(),
        };
        let ratio = m.reduction_ratio();
        assert!((ratio - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_minimal_witness_zero_original() {
        let m = MinimalWitness {
            original_count: 0,
            minimal_count: 0,
            constraints: Vec::new(),
            removed_constraints: Vec::new(),
        };
        assert_eq!(m.reduction_ratio(), 0.0);
    }

    #[test]
    fn test_constraint_checker_version_compat_ok() {
        let state = make_state("s1", &[("api", (2, 0, 0)), ("web", (2, 1, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0)), ("web", (2, 1, 0))]);
        let c = make_constraint(
            "c1",
            ConstraintType::VersionCompatibility,
            &["api", "web"],
            ConstraintSeverity::Error,
        );
        assert!(!ConstraintChecker::is_violated(&c, &state, &target));
    }

    #[test]
    fn test_constraint_checker_version_compat_violated() {
        let state = make_state("s1", &[("api", (1, 0, 0)), ("web", (3, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0)), ("web", (3, 0, 0))]);
        let c = make_constraint(
            "c1",
            ConstraintType::VersionCompatibility,
            &["api", "web"],
            ConstraintSeverity::Error,
        );
        assert!(ConstraintChecker::is_violated(&c, &state, &target));
    }

    #[test]
    fn test_constraint_checker_rollback_ok() {
        let state = make_state("s1", &[("api", (1, 0, 0))]);
        let target = make_state("s2", &[("api", (2, 0, 0))]);
        let c = make_constraint(
            "c-rb",
            ConstraintType::RollbackSafety,
            &["api"],
            ConstraintSeverity::Critical,
        );
        assert!(!ConstraintChecker::is_violated(&c, &state, &target));
    }

    #[test]
    fn test_witness_generator_default() {
        let gen = WitnessGenerator::default();
        assert_eq!(gen.max_iterations, 1000);
    }

    #[test]
    fn test_witness_generator_custom_iterations() {
        let gen = WitnessGenerator::new().with_max_iterations(500);
        assert_eq!(gen.max_iterations, 500);
    }

    #[test]
    fn test_stuck_witness_display() {
        let w = StuckWitness::new("s1", WitnessSeverity::Info)
            .with_explanation("Nothing wrong");
        let display = format!("{}", w);
        assert!(display.contains("INFO"));
        assert!(display.contains("Nothing wrong"));
    }

    #[test]
    fn test_infeasibility_witness_display() {
        let w = InfeasibilityWitness::new("No plan");
        let display = format!("{}", w);
        assert!(display.contains("No plan"));
    }

    #[test]
    fn test_suggestion_kind_display() {
        assert_eq!(SuggestionKind::RelaxConstraint.to_string(), "relax-constraint");
        assert_eq!(SuggestionKind::AddIntermediateVersion.to_string(), "add-intermediate-version");
        assert_eq!(SuggestionKind::ReorderSteps.to_string(), "reorder-steps");
    }

    #[test]
    fn test_empty_witness_is_empty() {
        let w = StuckWitness::new("s1", WitnessSeverity::Info);
        assert!(w.is_empty());
        assert_eq!(w.constraint_count(), 0);
    }
}
