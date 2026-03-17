//! Validation of repair plans before they are applied.
//!
//! The [`RepairValidator`] checks that a [`RepairPlan`] respects parameter
//! bounds, maintains timeout-chain consistency, and passes basic sensibility
//! checks.

use serde::{Deserialize, Serialize};

use super::synthesizer::ParameterBounds;
use super::{RepairAction, RepairActionType, RepairPlan};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An issue discovered during validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// `"error"` or `"warning"`.
    pub severity: String,
    pub description: String,
    /// Index into `RepairPlan.actions` that triggered the issue, if any.
    pub action_index: Option<usize>,
}

/// Aggregate result of validating a repair plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn ok() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.severity == "error")
    }
}

// ---------------------------------------------------------------------------
// RepairValidator
// ---------------------------------------------------------------------------

/// Validates repair plans for correctness and safety.
#[derive(Debug, Clone, Default)]
pub struct RepairValidator;

impl RepairValidator {
    pub fn new() -> Self {
        Self
    }

    /// Run all validation checks on a plan.
    pub fn validate(
        &self,
        plan: &RepairPlan,
        bounds: &ParameterBounds,
    ) -> ValidationResult {
        let mut issues = self.check_bounds(plan, bounds);
        let warnings = self.check_sensibility(plan);

        // Check for duplicate edge changes.
        issues.extend(self.check_duplicates(plan));

        let valid = !issues.iter().any(|i| i.severity == "error");

        ValidationResult {
            valid,
            issues,
            warnings,
        }
    }

    /// Check that every action stays within the allowed parameter bounds.
    pub fn check_bounds(
        &self,
        plan: &RepairPlan,
        bounds: &ParameterBounds,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        for (idx, action) in plan.actions.iter().enumerate() {
            match &action.action_type {
                RepairActionType::ReduceRetries { from, to } => {
                    if *to > bounds.max_retry {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: format!(
                                "Retry value {} exceeds maximum bound {}",
                                to, bounds.max_retry
                            ),
                            action_index: Some(idx),
                        });
                    }
                    if *to < bounds.min_retry {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: format!(
                                "Retry value {} is below minimum bound {}",
                                to, bounds.min_retry
                            ),
                            action_index: Some(idx),
                        });
                    }
                    if to > from {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: format!(
                                "ReduceRetries increases retries from {} to {}",
                                from, to
                            ),
                            action_index: Some(idx),
                        });
                    }
                }
                RepairActionType::AdjustTimeout { from_ms: _, to_ms } => {
                    if *to_ms > bounds.max_timeout_ms {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: format!(
                                "Timeout {}ms exceeds maximum bound {}ms",
                                to_ms, bounds.max_timeout_ms
                            ),
                            action_index: Some(idx),
                        });
                    }
                    if *to_ms < bounds.min_timeout_ms {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: format!(
                                "Timeout {}ms is below minimum bound {}ms",
                                to_ms, bounds.min_timeout_ms
                            ),
                            action_index: Some(idx),
                        });
                    }
                }
                RepairActionType::AddCircuitBreaker { threshold } => {
                    if *threshold == 0 {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: "Circuit breaker threshold must be > 0".into(),
                            action_index: Some(idx),
                        });
                    }
                }
                RepairActionType::AddRateLimit { rps } => {
                    if *rps <= 0.0 {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: "Rate limit must be positive".into(),
                            action_index: Some(idx),
                        });
                    }
                }
                RepairActionType::IncreaseCapacity { from, to } => {
                    if to < from {
                        issues.push(ValidationIssue {
                            severity: "error".into(),
                            description: format!(
                                "IncreaseCapacity decreases capacity from {:.0} to {:.0}",
                                from, to
                            ),
                            action_index: Some(idx),
                        });
                    }
                }
            }
        }

        issues
    }

    /// Check timeout-chain consistency: for each pair of consecutive edges
    /// `(A->B, B->C)` on a path, the upstream timeout should be ≥ the
    /// downstream timeout.
    ///
    /// `adj` provides `(source, target, timeout_ms)` tuples.
    pub fn check_consistency(
        &self,
        plan: &RepairPlan,
        adj: &[(String, String, u64)],
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Build an effective timeout map by overlaying plan changes on adj.
        let mut timeout_map: std::collections::HashMap<(String, String), u64> =
            std::collections::HashMap::new();
        for (s, t, to) in adj {
            timeout_map.insert((s.clone(), t.clone()), *to);
        }
        for action in &plan.actions {
            if let RepairActionType::AdjustTimeout { to_ms, .. } = &action.action_type {
                if let Some((ref src, ref tgt)) = action.edge {
                    timeout_map.insert((src.clone(), tgt.clone()), *to_ms);
                }
            }
        }

        // Check each adjacent pair of edges that share a middle node.
        let edges: Vec<(String, String)> = timeout_map.keys().cloned().collect();
        for (s1, t1) in &edges {
            for (s2, t2) in &edges {
                if t1 == s2 {
                    let up = timeout_map[&(s1.clone(), t1.clone())];
                    let down = timeout_map[&(s2.clone(), t2.clone())];
                    if up < down {
                        issues.push(ValidationIssue {
                            severity: "warning".into(),
                            description: format!(
                                "Timeout inconsistency: {}->{} ({}ms) < {}->{} ({}ms); \
                                 upstream should be >= downstream",
                                s1, t1, up, s2, t2, down
                            ),
                            action_index: None,
                        });
                    }
                }
            }
        }

        issues
    }

    /// Sensibility warnings (non-blocking).
    pub fn check_sensibility(&self, plan: &RepairPlan) -> Vec<String> {
        let mut warnings = Vec::new();

        if plan.actions.is_empty() {
            warnings.push("Repair plan has no actions.".into());
        }

        if !plan.feasible {
            warnings.push("Repair plan is marked as infeasible.".into());
        }

        if plan.total_deviation > 10.0 {
            warnings.push(format!(
                "Total deviation {:.2} is high; consider staged rollout.",
                plan.total_deviation
            ));
        }

        if plan.actions.len() > 10 {
            warnings.push(format!(
                "Plan has {} actions — review carefully before applying.",
                plan.actions.len()
            ));
        }

        // Check for zero-retry edges.
        for action in &plan.actions {
            if let RepairActionType::ReduceRetries { to, .. } = &action.action_type {
                if *to == 0 {
                    let edge_str = action
                        .edge
                        .as_ref()
                        .map(|(s, t)| format!("{}->{}", s, t))
                        .unwrap_or_else(|| action.service.clone());
                    warnings.push(format!(
                        "Edge {} reduced to 0 retries — failures will not be retried.",
                        edge_str
                    ));
                }
            }
        }

        // Check for very low timeouts.
        for action in &plan.actions {
            if let RepairActionType::AdjustTimeout { to_ms, .. } = &action.action_type {
                if *to_ms < 500 {
                    let edge_str = action
                        .edge
                        .as_ref()
                        .map(|(s, t)| format!("{}->{}", s, t))
                        .unwrap_or_else(|| action.service.clone());
                    warnings.push(format!(
                        "Edge {} timeout {}ms is very low — may cause premature failures.",
                        edge_str, to_ms
                    ));
                }
            }
        }

        warnings
    }

    /// Verify that applying the plan actually fixes an amplification
    /// violation for a specific path.
    pub fn validate_amplification_fix(
        &self,
        plan: &RepairPlan,
        adj: &[(String, String, u32, u64)],
        threshold: f64,
    ) -> bool {
        // Compute effective retries after applying the plan.
        let mut retry_map: std::collections::HashMap<(String, String), u32> =
            std::collections::HashMap::new();
        for (s, t, r, _) in adj {
            retry_map.insert((s.clone(), t.clone()), *r);
        }
        for action in &plan.actions {
            if let RepairActionType::ReduceRetries { to, .. } = &action.action_type {
                if let Some((ref src, ref tgt)) = action.edge {
                    retry_map.insert((src.clone(), tgt.clone()), *to);
                }
            }
        }

        // Enumerate all paths in the adjacency and check amplification.
        // For simplicity, check all simple paths of length ≥ 2.
        let paths = enumerate_simple_paths(adj);
        for path in &paths {
            let amp = path_amplification(path, &retry_map);
            if amp > threshold {
                return false;
            }
        }
        true
    }

    // -- private ----------------------------------------------------------

    fn check_duplicates(&self, plan: &RepairPlan) -> Vec<ValidationIssue> {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut issues = Vec::new();
        for (idx, action) in plan.actions.iter().enumerate() {
            let key = format!(
                "{:?}-{:?}-{}",
                action.edge,
                std::mem::discriminant(&action.action_type),
                action.service
            );
            if !seen.insert(key) {
                issues.push(ValidationIssue {
                    severity: "warning".into(),
                    description: format!(
                        "Duplicate action on edge {:?} for {}",
                        action.edge, action.service
                    ),
                    action_index: Some(idx),
                });
            }
        }
        issues
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Enumerate all simple paths of length ≥ 2 from the adjacency list.
fn enumerate_simple_paths(adj: &[(String, String, u32, u64)]) -> Vec<Vec<String>> {
    let mut successors: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    let mut nodes: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for (s, t, _, _) in adj {
        successors.entry(s.as_str()).or_default().push(t.as_str());
        nodes.insert(s.as_str());
        nodes.insert(t.as_str());
    }

    let mut all_paths = Vec::new();
    for &start in &nodes {
        let mut stack: Vec<(Vec<String>, std::collections::HashSet<String>)> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        visited.insert(start.to_string());
        stack.push((vec![start.to_string()], visited));

        while let Some((path, visited)) = stack.pop() {
            if path.len() >= 2 {
                all_paths.push(path.clone());
            }
            let last = path.last().unwrap().as_str();
            if let Some(succs) = successors.get(last) {
                for &next in succs {
                    if !visited.contains(next) {
                        let mut new_path = path.clone();
                        new_path.push(next.to_string());
                        let mut new_visited = visited.clone();
                        new_visited.insert(next.to_string());
                        stack.push((new_path, new_visited));
                    }
                }
            }
        }
    }
    all_paths
}

/// Compute amplification product along a path given a retry map.
fn path_amplification(
    path: &[String],
    retry_map: &std::collections::HashMap<(String, String), u32>,
) -> f64 {
    let mut product = 1.0_f64;
    for w in path.windows(2) {
        let key = (w[0].clone(), w[1].clone());
        let retries = retry_map.get(&key).copied().unwrap_or(0);
        product *= 1.0 + retries as f64;
    }
    product
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bounds() -> ParameterBounds {
        ParameterBounds {
            min_retry: 0,
            max_retry: 10,
            min_timeout_ms: 100,
            max_timeout_ms: 60_000,
        }
    }

    fn valid_plan() -> RepairPlan {
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::reduce_retries("A", "B", 5, 2));
        plan.add_action(RepairAction::adjust_timeout("B", "C", 5000, 3000));
        plan
    }

    #[test]
    fn test_valid_plan_passes() {
        let v = RepairValidator::new();
        let result = v.validate(&valid_plan(), &default_bounds());
        assert!(result.valid);
        assert!(!result.has_errors());
    }

    #[test]
    fn test_retry_exceeds_max() {
        let v = RepairValidator::new();
        let bounds = ParameterBounds {
            min_retry: 0,
            max_retry: 3,
            ..default_bounds()
        };
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::reduce_retries("A", "B", 8, 5));
        let result = v.validate(&plan, &bounds);
        assert!(!result.valid);
        assert!(result.issues.iter().any(|i| i.description.contains("exceeds maximum")));
    }

    #[test]
    fn test_retry_below_min() {
        let v = RepairValidator::new();
        let bounds = ParameterBounds {
            min_retry: 1,
            max_retry: 10,
            ..default_bounds()
        };
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::reduce_retries("A", "B", 3, 0));
        let result = v.validate(&plan, &bounds);
        assert!(!result.valid);
        assert!(result.issues.iter().any(|i| i.description.contains("below minimum")));
    }

    #[test]
    fn test_timeout_below_min() {
        let v = RepairValidator::new();
        let bounds = ParameterBounds {
            min_timeout_ms: 500,
            ..default_bounds()
        };
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::adjust_timeout("A", "B", 1000, 200));
        let result = v.validate(&plan, &bounds);
        assert!(!result.valid);
    }

    #[test]
    fn test_timeout_exceeds_max() {
        let v = RepairValidator::new();
        let bounds = ParameterBounds {
            max_timeout_ms: 5000,
            ..default_bounds()
        };
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::adjust_timeout("A", "B", 3000, 10000));
        let result = v.validate(&plan, &bounds);
        assert!(!result.valid);
    }

    #[test]
    fn test_reduce_retries_increases() {
        let v = RepairValidator::new();
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction {
            service: "A".into(),
            edge: Some(("A".into(), "B".into())),
            action_type: RepairActionType::ReduceRetries { from: 2, to: 5 },
            description: "bad".into(),
            deviation: 3.0,
        });
        let result = v.validate(&plan, &default_bounds());
        assert!(!result.valid);
        assert!(result.issues.iter().any(|i| i.description.contains("increases")));
    }

    #[test]
    fn test_consistency_check() {
        let v = RepairValidator::new();
        let plan = RepairPlan::default();
        let adj = vec![
            ("A".to_string(), "B".to_string(), 1000u64),
            ("B".to_string(), "C".to_string(), 5000u64),
        ];
        let issues = v.check_consistency(&plan, &adj);
        // A->B timeout (1000) < B->C timeout (5000) → warning
        assert!(!issues.is_empty());
        assert!(issues[0].description.contains("inconsistency"));
    }

    #[test]
    fn test_sensibility_empty_plan() {
        let v = RepairValidator::new();
        let plan = RepairPlan::default();
        let warnings = v.check_sensibility(&plan);
        assert!(warnings.iter().any(|w| w.contains("no actions")));
    }

    #[test]
    fn test_sensibility_infeasible() {
        let v = RepairValidator::new();
        let mut plan = RepairPlan::default();
        plan.feasible = false;
        let warnings = v.check_sensibility(&plan);
        assert!(warnings.iter().any(|w| w.contains("infeasible")));
    }

    #[test]
    fn test_sensibility_zero_retries_warning() {
        let v = RepairValidator::new();
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::reduce_retries("A", "B", 3, 0));
        let warnings = v.check_sensibility(&plan);
        assert!(warnings.iter().any(|w| w.contains("0 retries")));
    }

    #[test]
    fn test_validate_amplification_fix_passes() {
        let v = RepairValidator::new();
        let adj = vec![
            ("A".to_string(), "B".to_string(), 5u32, 1000u64),
            ("B".to_string(), "C".to_string(), 4u32, 1000u64),
        ];
        // Original amp = (1+5)*(1+4) = 30
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::reduce_retries("A", "B", 5, 1));
        plan.add_action(RepairAction::reduce_retries("B", "C", 4, 1));
        // New amp = 2 * 2 = 4
        assert!(v.validate_amplification_fix(&plan, &adj, 10.0));
    }

    #[test]
    fn test_validate_amplification_fix_fails() {
        let v = RepairValidator::new();
        let adj = vec![
            ("A".to_string(), "B".to_string(), 5u32, 1000u64),
            ("B".to_string(), "C".to_string(), 4u32, 1000u64),
        ];
        // No changes → amp still 30 > 10
        let plan = RepairPlan::default();
        assert!(!v.validate_amplification_fix(&plan, &adj, 10.0));
    }
}
