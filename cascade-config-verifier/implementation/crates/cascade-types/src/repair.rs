use std::fmt;

use serde::{Deserialize, Serialize};

use crate::topology::EdgeId;

// ---------------------------------------------------------------------------
// RepairAction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RepairAction {
    ModifyRetryCount {
        edge_id: EdgeId,
        new_count: u32,
    },
    ModifyTimeout {
        edge_id: EdgeId,
        new_timeout_ms: u64,
    },
    ModifyCapacity {
        service_id: String,
        new_capacity: f64,
    },
    AddCircuitBreaker {
        edge_id: EdgeId,
        max_connections: u32,
        consecutive_errors: u32,
    },
    AddRateLimit {
        edge_id: EdgeId,
        requests_per_second: f64,
    },
    AddBulkhead {
        edge_id: EdgeId,
        max_concurrent: u32,
    },
    RemoveRetry {
        edge_id: EdgeId,
    },
    ModifyBackoff {
        edge_id: EdgeId,
        strategy: String,
    },
}

impl RepairAction {
    pub fn description(&self) -> String {
        match self {
            Self::ModifyRetryCount { edge_id, new_count } => {
                format!("Set retry count to {} on edge {}", new_count, edge_id)
            }
            Self::ModifyTimeout {
                edge_id,
                new_timeout_ms,
            } => {
                format!("Set timeout to {}ms on edge {}", new_timeout_ms, edge_id)
            }
            Self::ModifyCapacity {
                service_id,
                new_capacity,
            } => {
                format!(
                    "Set capacity to {:.2} on service {}",
                    new_capacity, service_id
                )
            }
            Self::AddCircuitBreaker {
                edge_id,
                max_connections,
                consecutive_errors,
            } => {
                format!(
                    "Add circuit breaker on edge {} (max_conn={}, consec_err={})",
                    edge_id, max_connections, consecutive_errors
                )
            }
            Self::AddRateLimit {
                edge_id,
                requests_per_second,
            } => {
                format!(
                    "Add rate limit of {:.1} req/s on edge {}",
                    requests_per_second, edge_id
                )
            }
            Self::AddBulkhead {
                edge_id,
                max_concurrent,
            } => {
                format!(
                    "Add bulkhead with {} max concurrent on edge {}",
                    max_concurrent, edge_id
                )
            }
            Self::RemoveRetry { edge_id } => {
                format!("Remove retry policy on edge {}", edge_id)
            }
            Self::ModifyBackoff { edge_id, strategy } => {
                format!(
                    "Set backoff strategy to '{}' on edge {}",
                    strategy, edge_id
                )
            }
        }
    }

    pub fn affected_edge(&self) -> Option<&EdgeId> {
        match self {
            Self::ModifyRetryCount { edge_id, .. }
            | Self::ModifyTimeout { edge_id, .. }
            | Self::AddCircuitBreaker { edge_id, .. }
            | Self::AddRateLimit { edge_id, .. }
            | Self::AddBulkhead { edge_id, .. }
            | Self::RemoveRetry { edge_id }
            | Self::ModifyBackoff { edge_id, .. } => Some(edge_id),
            Self::ModifyCapacity { .. } => None,
        }
    }

    pub fn affected_service(&self) -> Option<&str> {
        match self {
            Self::ModifyCapacity { service_id, .. } => Some(service_id.as_str()),
            _ => None,
        }
    }
}

impl fmt::Display for RepairAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// ParameterChange
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterChange {
    pub edge_id: EdgeId,
    pub parameter: String,
    pub old_value: f64,
    pub new_value: f64,
    pub weight: f64,
}

impl ParameterChange {
    pub fn deviation(&self) -> f64 {
        (self.new_value - self.old_value).abs()
    }

    pub fn relative_deviation(&self) -> f64 {
        if self.old_value == 0.0 {
            if self.new_value == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.deviation() / self.old_value.abs()
        }
    }
}

impl fmt::Display for ParameterChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "edge {}: {} changed from {:.4} to {:.4} (weight={:.2})",
            self.edge_id, self.parameter, self.old_value, self.new_value, self.weight
        )
    }
}

// ---------------------------------------------------------------------------
// RepairPlan
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairPlan {
    pub id: String,
    pub changes: Vec<ParameterChange>,
    pub actions: Vec<RepairAction>,
    pub cost: f64,
    pub effectiveness: f64,
    pub description: String,
}

impl RepairPlan {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            changes: Vec::new(),
            actions: Vec::new(),
            cost: 0.0,
            effectiveness: 0.0,
            description: description.into(),
        }
    }

    pub fn with_change(mut self, change: ParameterChange) -> Self {
        self.changes.push(change);
        self
    }

    pub fn with_action(mut self, action: RepairAction) -> Self {
        self.actions.push(action);
        self
    }

    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    pub fn total_deviation(&self) -> f64 {
        self.changes
            .iter()
            .map(|c| c.deviation() * c.weight)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty() && self.actions.is_empty()
    }

    pub fn cost_effectiveness_ratio(&self) -> f64 {
        if self.effectiveness == 0.0 {
            f64::INFINITY
        } else {
            self.cost / self.effectiveness
        }
    }

    pub fn validate(&self) -> RepairValidation {
        let mut validation = RepairValidation::new();

        if self.description.is_empty() {
            validation.add_violation("RepairPlan description must not be empty".into());
        }
        if self.cost < 0.0 {
            validation.add_violation(format!("Cost must be non-negative, got {}", self.cost));
        }
        if self.effectiveness < 0.0 {
            validation.add_violation(format!(
                "Effectiveness must be non-negative, got {}",
                self.effectiveness
            ));
        }
        for (i, change) in self.changes.iter().enumerate() {
            if change.weight < 0.0 {
                validation.add_violation(format!(
                    "Change {} has negative weight {}",
                    i, change.weight
                ));
            }
        }
        if self.is_empty() {
            validation.add_warning("RepairPlan has no changes and no actions".into());
        }

        validation
    }
}

impl PartialEq for RepairPlan {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

// ---------------------------------------------------------------------------
// RepairConstraint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairConstraint {
    pub parameter_type: String,
    pub min_value: f64,
    pub max_value: f64,
    pub description: Option<String>,
}

impl RepairConstraint {
    pub fn new(
        parameter_type: impl Into<String>,
        min_value: f64,
        max_value: f64,
    ) -> Self {
        Self {
            parameter_type: parameter_type.into(),
            min_value,
            max_value,
            description: None,
        }
    }

    pub fn is_satisfied(&self, value: f64) -> bool {
        value >= self.min_value && value <= self.max_value
    }

    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min_value, self.max_value)
    }
}

impl fmt::Display for RepairConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: [{:.4}, {:.4}]",
            self.parameter_type, self.min_value, self.max_value
        )?;
        if let Some(desc) = &self.description {
            write!(f, " ({})", desc)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RepairObjective
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RepairObjective {
    MinimizeChanges,
    MinimizeDeviation,
    MaximizeRobustness,
    MinimizeCost,
    MultiObjective(Vec<(RepairObjective, f64)>),
}

impl RepairObjective {
    pub fn description(&self) -> &str {
        match self {
            Self::MinimizeChanges => "Minimize the total number of parameter changes",
            Self::MinimizeDeviation => "Minimize the total deviation from original values",
            Self::MaximizeRobustness => "Maximize system robustness against cascading failures",
            Self::MinimizeCost => "Minimize the estimated cost of applying repairs",
            Self::MultiObjective(_) => "Weighted combination of multiple objectives",
        }
    }
}

impl fmt::Display for RepairObjective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MultiObjective(objectives) => {
                write!(f, "MultiObjective[")?;
                for (i, (obj, weight)) in objectives.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "({}, {:.2})", obj, weight)?;
                }
                write!(f, "]")
            }
            _ => write!(f, "{}", self.description()),
        }
    }
}

// ---------------------------------------------------------------------------
// RepairCandidate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairCandidate {
    pub plan: RepairPlan,
    pub cost: f64,
    pub effectiveness: f64,
    pub feasible: bool,
}

// ---------------------------------------------------------------------------
// ParetoFrontier
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier {
    candidates: Vec<RepairCandidate>,
}

impl ParetoFrontier {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Returns `true` if `a` Pareto-dominates `b`.
    ///
    /// `a` dominates `b` when a.cost <= b.cost AND a.effectiveness >= b.effectiveness
    /// with at least one strict inequality.
    pub fn dominates(a: &RepairCandidate, b: &RepairCandidate) -> bool {
        let cost_ok = a.cost <= b.cost;
        let eff_ok = a.effectiveness >= b.effectiveness;
        let strict = a.cost < b.cost || a.effectiveness > b.effectiveness;
        cost_ok && eff_ok && strict
    }

    /// Insert a candidate into the frontier.
    ///
    /// If the new candidate is dominated by any existing candidate it is
    /// discarded. Otherwise it is added and any existing candidates that the
    /// new one dominates are removed.
    pub fn insert(&mut self, candidate: RepairCandidate) {
        // If any existing candidate dominates the newcomer, skip.
        for existing in &self.candidates {
            if Self::dominates(existing, &candidate) {
                return;
            }
        }
        // Remove existing candidates dominated by the newcomer.
        self.candidates
            .retain(|existing| !Self::dominates(&candidate, existing));
        self.candidates.push(candidate);
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &RepairCandidate> {
        self.candidates.iter()
    }

    pub fn best_by_cost(&self) -> Option<&RepairCandidate> {
        self.candidates
            .iter()
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn best_by_effectiveness(&self) -> Option<&RepairCandidate> {
        self.candidates
            .iter()
            .max_by(|a, b| {
                a.effectiveness
                    .partial_cmp(&b.effectiveness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for ParetoFrontier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RepairValidation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairValidation {
    pub valid: bool,
    pub violations: Vec<String>,
    pub warnings: Vec<String>,
}

impl RepairValidation {
    pub fn new() -> Self {
        Self {
            valid: true,
            violations: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_violation(&mut self, violation: String) {
        self.valid = false;
        self.violations.push(violation);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn merge(&mut self, other: RepairValidation) {
        if !other.valid {
            self.valid = false;
        }
        self.violations.extend(other.violations);
        self.warnings.extend(other.warnings);
    }
}

impl Default for RepairValidation {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DiffHunk
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    pub line_number: usize,
    pub original_lines: Vec<String>,
    pub modified_lines: Vec<String>,
    pub context: Option<String>,
}

impl fmt::Display for DiffHunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ctx) = &self.context {
            write!(f, "@@ line {} @@ {}\n", self.line_number, ctx)?;
        } else {
            write!(f, "@@ line {} @@\n", self.line_number)?;
        }
        for line in &self.original_lines {
            write!(f, "-{}\n", line)?;
        }
        for line in &self.modified_lines {
            write!(f, "+{}\n", line)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConfigDiff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigDiff {
    pub file_path: String,
    pub original: String,
    pub modified: String,
    pub changes: Vec<DiffHunk>,
}

impl ConfigDiff {
    pub fn new(file_path: impl Into<String>, original: impl Into<String>, modified: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            original: original.into(),
            modified: modified.into(),
            changes: Vec::new(),
        }
    }

    pub fn add_hunk(&mut self, hunk: DiffHunk) {
        self.changes.push(hunk);
    }

    pub fn has_changes(&self) -> bool {
        !self.changes.is_empty()
    }

    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    pub fn to_unified_diff(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("--- {}\n", self.file_path));
        out.push_str(&format!("+++ {}\n", self.file_path));
        for hunk in &self.changes {
            let orig_count = hunk.original_lines.len();
            let mod_count = hunk.modified_lines.len();
            out.push_str(&format!(
                "@@ -{},{} +{},{} @@",
                hunk.line_number, orig_count, hunk.line_number, mod_count
            ));
            if let Some(ctx) = &hunk.context {
                out.push_str(&format!(" {}", ctx));
            }
            out.push('\n');
            for line in &hunk.original_lines {
                out.push_str(&format!("-{}\n", line));
            }
            for line in &hunk.modified_lines {
                out.push_str(&format!("+{}\n", line));
            }
        }
        out
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_edge() -> EdgeId {
        EdgeId("edge-a-b".into())
    }

    fn sample_change() -> ParameterChange {
        ParameterChange {
            edge_id: sample_edge(),
            parameter: "retry_count".into(),
            old_value: 3.0,
            new_value: 5.0,
            weight: 1.0,
        }
    }

    // -- RepairAction -------------------------------------------------------

    #[test]
    fn repair_action_description_and_display() {
        let action = RepairAction::ModifyRetryCount {
            edge_id: sample_edge(),
            new_count: 5,
        };
        let desc = action.description();
        assert!(desc.contains("retry count"));
        assert!(desc.contains("5"));
        assert_eq!(format!("{}", action), desc);
    }

    #[test]
    fn repair_action_affected_edge() {
        let action = RepairAction::AddRateLimit {
            edge_id: sample_edge(),
            requests_per_second: 100.0,
        };
        assert_eq!(action.affected_edge(), Some(&sample_edge()));
        assert_eq!(action.affected_service(), None);
    }

    #[test]
    fn repair_action_affected_service() {
        let action = RepairAction::ModifyCapacity {
            service_id: "svc-1".into(),
            new_capacity: 42.0,
        };
        assert_eq!(action.affected_service(), Some("svc-1"));
        assert_eq!(action.affected_edge(), None);
    }

    #[test]
    fn repair_action_all_variants_have_description() {
        let actions = vec![
            RepairAction::ModifyRetryCount { edge_id: sample_edge(), new_count: 3 },
            RepairAction::ModifyTimeout { edge_id: sample_edge(), new_timeout_ms: 500 },
            RepairAction::ModifyCapacity { service_id: "s".into(), new_capacity: 1.0 },
            RepairAction::AddCircuitBreaker { edge_id: sample_edge(), max_connections: 10, consecutive_errors: 5 },
            RepairAction::AddRateLimit { edge_id: sample_edge(), requests_per_second: 50.0 },
            RepairAction::AddBulkhead { edge_id: sample_edge(), max_concurrent: 8 },
            RepairAction::RemoveRetry { edge_id: sample_edge() },
            RepairAction::ModifyBackoff { edge_id: sample_edge(), strategy: "exponential".into() },
        ];
        for a in &actions {
            assert!(!a.description().is_empty());
        }
    }

    // -- ParameterChange ----------------------------------------------------

    #[test]
    fn parameter_change_deviation() {
        let c = sample_change();
        assert!((c.deviation() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parameter_change_relative_deviation() {
        let c = sample_change();
        let expected = 2.0 / 3.0;
        assert!((c.relative_deviation() - expected).abs() < 1e-9);
    }

    #[test]
    fn parameter_change_relative_deviation_zero_old() {
        let c = ParameterChange {
            edge_id: sample_edge(),
            parameter: "x".into(),
            old_value: 0.0,
            new_value: 5.0,
            weight: 1.0,
        };
        assert!(c.relative_deviation().is_infinite());

        let c2 = ParameterChange {
            old_value: 0.0,
            new_value: 0.0,
            ..c
        };
        assert!((c2.relative_deviation() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parameter_change_display() {
        let d = format!("{}", sample_change());
        assert!(d.contains("edge-a-b"));
        assert!(d.contains("retry_count"));
    }

    // -- RepairPlan ---------------------------------------------------------

    #[test]
    fn repair_plan_builder() {
        let plan = RepairPlan::new("fix cascading timeout")
            .with_change(sample_change())
            .with_action(RepairAction::ModifyRetryCount {
                edge_id: sample_edge(),
                new_count: 5,
            });

        assert_eq!(plan.change_count(), 1);
        assert!(!plan.is_empty());
        assert!(!plan.id.is_empty());
        assert!(plan.description.contains("timeout"));
    }

    #[test]
    fn repair_plan_total_deviation() {
        let c1 = ParameterChange {
            edge_id: sample_edge(),
            parameter: "a".into(),
            old_value: 1.0,
            new_value: 4.0,
            weight: 2.0,
        };
        let c2 = ParameterChange {
            edge_id: sample_edge(),
            parameter: "b".into(),
            old_value: 10.0,
            new_value: 7.0,
            weight: 1.0,
        };
        let plan = RepairPlan::new("test")
            .with_change(c1)
            .with_change(c2);
        // 3*2 + 3*1 = 9
        assert!((plan.total_deviation() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn repair_plan_cost_effectiveness_ratio() {
        let mut plan = RepairPlan::new("ratio test");
        plan.cost = 10.0;
        plan.effectiveness = 5.0;
        assert!((plan.cost_effectiveness_ratio() - 2.0).abs() < f64::EPSILON);

        plan.effectiveness = 0.0;
        assert!(plan.cost_effectiveness_ratio().is_infinite());
    }

    #[test]
    fn repair_plan_validate_ok() {
        let plan = RepairPlan::new("good plan")
            .with_change(sample_change());
        let v = plan.validate();
        assert!(v.is_valid());
        assert!(v.violations.is_empty());
    }

    #[test]
    fn repair_plan_validate_failures() {
        let mut plan = RepairPlan::new("");
        plan.cost = -1.0;
        let v = plan.validate();
        assert!(!v.is_valid());
        assert!(v.violations.iter().any(|v| v.contains("description")));
        assert!(v.violations.iter().any(|v| v.contains("Cost")));
        assert!(!v.warnings.is_empty()); // empty plan warning
    }

    // -- RepairConstraint ---------------------------------------------------

    #[test]
    fn constraint_satisfaction() {
        let c = RepairConstraint::new("timeout_ms", 100.0, 5000.0);
        assert!(c.is_satisfied(100.0));
        assert!(c.is_satisfied(5000.0));
        assert!(c.is_satisfied(2500.0));
        assert!(!c.is_satisfied(99.9));
        assert!(!c.is_satisfied(5001.0));
    }

    #[test]
    fn constraint_clamp() {
        let c = RepairConstraint::new("rate", 0.0, 1.0);
        assert!((c.clamp(-0.5) - 0.0).abs() < f64::EPSILON);
        assert!((c.clamp(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((c.clamp(1.5) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn constraint_display() {
        let mut c = RepairConstraint::new("retry", 1.0, 10.0);
        c.description = Some("max retries".into());
        let s = format!("{}", c);
        assert!(s.contains("retry"));
        assert!(s.contains("max retries"));
    }

    // -- RepairObjective ----------------------------------------------------

    #[test]
    fn objective_description_and_display() {
        assert!(!RepairObjective::MinimizeChanges.description().is_empty());
        assert!(!RepairObjective::MinimizeDeviation.description().is_empty());
        assert!(!RepairObjective::MaximizeRobustness.description().is_empty());
        assert!(!RepairObjective::MinimizeCost.description().is_empty());

        let multi = RepairObjective::MultiObjective(vec![
            (RepairObjective::MinimizeChanges, 0.5),
            (RepairObjective::MinimizeCost, 0.5),
        ]);
        let s = format!("{}", multi);
        assert!(s.contains("MultiObjective"));
    }

    // -- ParetoFrontier -----------------------------------------------------

    #[test]
    fn pareto_dominance() {
        let make = |cost, eff| RepairCandidate {
            plan: RepairPlan::new("p"),
            cost,
            effectiveness: eff,
            feasible: true,
        };
        // a strictly better on both axes
        assert!(ParetoFrontier::dominates(&make(1.0, 10.0), &make(2.0, 5.0)));
        // a equal cost, better effectiveness
        assert!(ParetoFrontier::dominates(&make(2.0, 10.0), &make(2.0, 5.0)));
        // a better cost, equal effectiveness
        assert!(ParetoFrontier::dominates(&make(1.0, 5.0), &make(2.0, 5.0)));
        // neither dominates (trade-off)
        assert!(!ParetoFrontier::dominates(&make(1.0, 5.0), &make(2.0, 10.0)));
        // equal on both — no strict inequality
        assert!(!ParetoFrontier::dominates(&make(2.0, 5.0), &make(2.0, 5.0)));
    }

    #[test]
    fn pareto_insert_removes_dominated() {
        let make = |cost, eff| RepairCandidate {
            plan: RepairPlan::new("p"),
            cost,
            effectiveness: eff,
            feasible: true,
        };
        let mut pf = ParetoFrontier::new();
        assert!(pf.is_empty());

        pf.insert(make(5.0, 5.0));
        pf.insert(make(3.0, 3.0));
        assert_eq!(pf.len(), 2); // trade-off, both stay

        // This dominates (3,3)
        pf.insert(make(2.0, 4.0));
        assert_eq!(pf.len(), 2); // (5,5) and (2,4) remain

        // This is dominated by (2,4)
        pf.insert(make(3.0, 3.0));
        assert_eq!(pf.len(), 2); // unchanged
    }

    #[test]
    fn pareto_best_by() {
        let make = |cost, eff| RepairCandidate {
            plan: RepairPlan::new("p"),
            cost,
            effectiveness: eff,
            feasible: true,
        };
        let mut pf = ParetoFrontier::new();
        pf.insert(make(1.0, 3.0));
        pf.insert(make(5.0, 10.0));

        assert!((pf.best_by_cost().unwrap().cost - 1.0).abs() < f64::EPSILON);
        assert!((pf.best_by_effectiveness().unwrap().effectiveness - 10.0).abs() < f64::EPSILON);
    }

    // -- RepairValidation ---------------------------------------------------

    #[test]
    fn validation_merge() {
        let mut v1 = RepairValidation::new();
        v1.add_warning("w1".into());

        let mut v2 = RepairValidation::new();
        v2.add_violation("bad".into());
        v2.add_warning("w2".into());

        v1.merge(v2);
        assert!(!v1.is_valid());
        assert_eq!(v1.violations.len(), 1);
        assert_eq!(v1.warnings.len(), 2);
    }

    // -- ConfigDiff / DiffHunk ----------------------------------------------

    #[test]
    fn diff_hunk_display() {
        let h = DiffHunk {
            line_number: 10,
            original_lines: vec!["old line".into()],
            modified_lines: vec!["new line".into()],
            context: Some("fn main".into()),
        };
        let s = format!("{}", h);
        assert!(s.contains("@@ line 10 @@"));
        assert!(s.contains("fn main"));
        assert!(s.contains("-old line"));
        assert!(s.contains("+new line"));
    }

    #[test]
    fn config_diff_unified() {
        let mut diff = ConfigDiff::new("envoy.yaml", "original", "modified");
        diff.add_hunk(DiffHunk {
            line_number: 5,
            original_lines: vec!["timeout: 1000".into()],
            modified_lines: vec!["timeout: 3000".into()],
            context: None,
        });
        diff.add_hunk(DiffHunk {
            line_number: 20,
            original_lines: vec!["retries: 3".into(), "backoff: fixed".into()],
            modified_lines: vec!["retries: 5".into(), "backoff: exponential".into()],
            context: Some("http_filters".into()),
        });

        assert!(diff.has_changes());
        assert_eq!(diff.change_count(), 2);

        let unified = diff.to_unified_diff();
        assert!(unified.contains("--- envoy.yaml"));
        assert!(unified.contains("+++ envoy.yaml"));
        assert!(unified.contains("@@ -5,1 +5,1 @@"));
        assert!(unified.contains("@@ -20,2 +20,2 @@ http_filters"));
        assert!(unified.contains("-timeout: 1000"));
        assert!(unified.contains("+timeout: 3000"));
    }

    #[test]
    fn config_diff_empty() {
        let diff = ConfigDiff::new("empty.yaml", "", "");
        assert!(!diff.has_changes());
        assert_eq!(diff.change_count(), 0);
        assert!(diff.to_unified_diff().contains("--- empty.yaml"));
    }

    // -- Serialization roundtrip --------------------------------------------

    #[test]
    fn repair_action_serde_roundtrip() {
        let action = RepairAction::AddCircuitBreaker {
            edge_id: sample_edge(),
            max_connections: 100,
            consecutive_errors: 5,
        };
        let json = serde_json::to_string(&action).unwrap();
        let back: RepairAction = serde_json::from_str(&json).unwrap();
        assert_eq!(action, back);
    }

    #[test]
    fn repair_plan_serde_roundtrip() {
        let plan = RepairPlan::new("roundtrip test")
            .with_change(sample_change())
            .with_action(RepairAction::RemoveRetry {
                edge_id: sample_edge(),
            });
        let json = serde_json::to_string_pretty(&plan).unwrap();
        let back: RepairPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, plan.id);
        assert_eq!(back.changes.len(), 1);
        assert_eq!(back.actions.len(), 1);
    }

    #[test]
    fn parameter_change_serde_roundtrip() {
        let c = sample_change();
        let json = serde_json::to_string(&c).unwrap();
        let back: ParameterChange = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn repair_candidate_serde_roundtrip() {
        let rc = RepairCandidate {
            plan: RepairPlan::new("candidate"),
            cost: 3.14,
            effectiveness: 0.95,
            feasible: true,
        };
        let json = serde_json::to_string(&rc).unwrap();
        let back: RepairCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, back);
    }
}
