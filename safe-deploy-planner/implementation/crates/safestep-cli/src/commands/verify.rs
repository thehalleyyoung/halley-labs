//! Implementation of the `verify` subcommand.

use std::collections::HashSet;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cli::VerifyArgs;
use crate::commands::{Finding, FindingSeverity, render_findings};
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Serializable plan for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiablePlan {
    pub plan_id: String,
    pub start_state: Vec<u16>,
    pub target_state: Vec<u16>,
    pub steps: Vec<VerifiableStep>,
    pub services: Vec<String>,
}

/// A single step in a verifiable plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiableStep {
    pub step: usize,
    pub service: String,
    pub from_version: String,
    pub to_version: String,
    #[serde(default)]
    pub service_index: usize,
    #[serde(default)]
    pub from_index: u16,
    #[serde(default)]
    pub to_index: u16,
}

/// Constraint definition for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VerifyConstraint {
    #[serde(rename = "compatibility")]
    Compatibility {
        service_a: String,
        service_b: String,
        compatible_pairs: Vec<(u16, u16)>,
    },
    #[serde(rename = "ordering")]
    Ordering { before: String, after: String },
    #[serde(rename = "forbidden")]
    Forbidden { service: String, version: u16 },
}

/// Verification result summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub plan_id: String,
    pub passed: bool,
    pub checks_run: usize,
    pub violations: usize,
    pub warnings: usize,
}

// ---------------------------------------------------------------------------
// VerifyCommand
// ---------------------------------------------------------------------------

pub struct VerifyCommand {
    args: VerifyArgs,
}

impl VerifyCommand {
    pub fn new(args: VerifyArgs) -> Self {
        Self { args }
    }

    pub fn execute(&self, output: &mut OutputManager) -> Result<()> {
        info!(plan = %self.args.plan_file.display(), "verifying plan");

        let plan = self.load_plan()?;
        let constraints = self.load_constraints()?;

        output.section("Plan Verification");
        output.writeln(&format!("Plan: {}", plan.plan_id));
        output.writeln(&format!("Steps: {}", plan.steps.len()));
        output.writeln(&format!("Services: {}", plan.services.len()));
        output.writeln(&format!("Constraints: {}", constraints.len()));

        let mut findings = Vec::new();
        let mut checks_run = 0;

        // Check 1: Plan completeness (start -> target).
        if self.args.check_completeness {
            checks_run += 1;
            self.check_completeness(&plan, &mut findings);
        }

        // Check 2: Monotonicity (each service changes at most once).
        if self.args.check_monotonicity {
            checks_run += 1;
            self.check_monotonicity(&plan, &mut findings);
        }

        // Check 3: Step consistency (from_version matches current state).
        checks_run += 1;
        self.check_step_consistency(&plan, &mut findings);

        // Check 4: Constraint satisfaction at each intermediate state.
        checks_run += 1;
        self.check_constraints(&plan, &constraints, &mut findings);

        // Check 5: No duplicate steps.
        checks_run += 1;
        self.check_no_duplicates(&plan, &mut findings);

        // Check 6: Valid service references.
        checks_run += 1;
        self.check_service_references(&plan, &mut findings);

        output.blank_line();
        let error_count = findings.iter().filter(|f| f.severity == FindingSeverity::Error).count();
        let warning_count = findings.iter().filter(|f| f.severity == FindingSeverity::Warning).count();

        if findings.is_empty() {
            output.writeln(&output.colors().clone().safe("✓ All checks passed"));
        } else {
            if !self.args.show_all {
                let max_show = 10;
                let truncated = findings.len() > max_show;
                let display_findings: Vec<Finding> = findings.iter().take(max_show).cloned().collect();
                render_findings(output, &display_findings);
                if truncated {
                    output.writeln(&format!("... and {} more findings (use --show-all to see all)", findings.len() - max_show));
                }
            } else {
                render_findings(output, &findings);
            }
        }

        output.blank_line();
        let result = VerificationResult {
            plan_id: plan.plan_id.clone(),
            passed: error_count == 0,
            checks_run,
            violations: error_count,
            warnings: warning_count,
        };

        output.writeln(&format!("Checks: {}", result.checks_run));
        output.writeln(&format!("Violations: {}", result.violations));
        output.writeln(&format!("Warnings: {}", result.warnings));

        if error_count > 0 {
            output.writeln(&output.colors().clone().error("✗ Verification FAILED"));
            anyhow::bail!("plan verification failed with {} violation(s)", error_count);
        } else {
            output.writeln(&output.colors().clone().safe("✓ Verification PASSED"));
        }

        Ok(())
    }

    fn load_plan(&self) -> Result<VerifiablePlan> {
        let content = std::fs::read_to_string(&self.args.plan_file)
            .with_context(|| format!("failed to read plan: {}", self.args.plan_file.display()))?;
        let plan: VerifiablePlan = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse plan: {}", self.args.plan_file.display()))?;
        Ok(plan)
    }

    fn load_constraints(&self) -> Result<Vec<VerifyConstraint>> {
        if let Some(ref path) = self.args.constraints_file {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read constraints: {}", path.display()))?;
            let constraints: Vec<VerifyConstraint> = serde_json::from_str(&content)
                .with_context(|| format!("failed to parse constraints: {}", path.display()))?;
            Ok(constraints)
        } else {
            Ok(Vec::new())
        }
    }

    fn check_completeness(&self, plan: &VerifiablePlan, findings: &mut Vec<Finding>) {
        if plan.start_state.is_empty() || plan.target_state.is_empty() {
            findings.push(Finding::error("plan has empty start or target state"));
            return;
        }

        if plan.start_state.len() != plan.target_state.len() {
            findings.push(Finding::error(format!(
                "start state dimension ({}) != target state dimension ({})",
                plan.start_state.len(), plan.target_state.len()
            )));
            return;
        }

        // Simulate execution.
        let mut current = plan.start_state.clone();
        for step in &plan.steps {
            if step.service_index < current.len() {
                current[step.service_index] = step.to_index;
            }
        }

        if current != plan.target_state {
            findings.push(Finding::error(format!(
                "plan does not reach target state: final={:?} expected={:?}",
                current, plan.target_state
            )).with_suggestion("verify that all services are transitioned to target versions"));
        } else {
            findings.push(Finding::info("completeness check passed"));
        }
    }

    fn check_monotonicity(&self, plan: &VerifiablePlan, findings: &mut Vec<Finding>) {
        let mut seen_services: HashSet<String> = HashSet::new();
        for step in &plan.steps {
            if seen_services.contains(&step.service) {
                findings.push(Finding::warning(format!(
                    "service '{}' is modified more than once (step {})",
                    step.service, step.step
                )).with_suggestion("consider combining transitions for the same service"));
            }
            seen_services.insert(step.service.clone());
        }
        if findings.iter().all(|f| f.severity != FindingSeverity::Warning || !f.message.contains("modified more than once")) {
            findings.push(Finding::info("monotonicity check passed"));
        }
    }

    fn check_step_consistency(&self, plan: &VerifiablePlan, findings: &mut Vec<Finding>) {
        let mut current = plan.start_state.clone();
        for step in &plan.steps {
            let idx = step.service_index;
            if idx >= current.len() {
                findings.push(Finding::error(format!(
                    "step {} references service index {} but state has {} services",
                    step.step, idx, current.len()
                )));
                continue;
            }
            if current[idx] != step.from_index {
                findings.push(Finding::error(format!(
                    "step {}: service '{}' expected version {} but current is {}",
                    step.step, step.service, step.from_index, current[idx]
                )));
            }
            current[idx] = step.to_index;
        }
    }

    fn check_constraints(
        &self,
        plan: &VerifiablePlan,
        constraints: &[VerifyConstraint],
        findings: &mut Vec<Finding>,
    ) {
        if constraints.is_empty() {
            return;
        }

        let svc_to_idx: std::collections::HashMap<&str, usize> = plan
            .services
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        let mut current = plan.start_state.clone();

        // Check constraints at start state and each intermediate state.
        for (step_num, step) in plan.steps.iter().enumerate() {
            if step.service_index < current.len() {
                current[step.service_index] = step.to_index;
            }
            for c in constraints {
                match c {
                    VerifyConstraint::Compatibility { service_a, service_b, compatible_pairs } => {
                        if let (Some(&ia), Some(&ib)) = (svc_to_idx.get(service_a.as_str()), svc_to_idx.get(service_b.as_str())) {
                            if ia < current.len() && ib < current.len() {
                                let va = current[ia];
                                let vb = current[ib];
                                if !compatible_pairs.contains(&(va, vb)) {
                                    findings.push(Finding::error(format!(
                                        "after step {}: {} (v{}) and {} (v{}) are incompatible",
                                        step_num + 1, service_a, va, service_b, vb
                                    )));
                                }
                            }
                        }
                    }
                    VerifyConstraint::Forbidden { service, version } => {
                        if let Some(&idx) = svc_to_idx.get(service.as_str()) {
                            if idx < current.len() && current[idx] == *version {
                                findings.push(Finding::error(format!(
                                    "after step {}: service '{}' at forbidden version {}",
                                    step_num + 1, service, version
                                )));
                            }
                        }
                    }
                    VerifyConstraint::Ordering { before: _, after: _ } => {
                        // Check ordering at plan level (done once at end).
                    }
                }
            }
        }

        // Check ordering constraints.
        for c in constraints {
            if let VerifyConstraint::Ordering { before, after } = c {
                let before_pos = plan.steps.iter().position(|s| s.service == *before);
                let after_pos = plan.steps.iter().position(|s| s.service == *after);
                if let (Some(bp), Some(ap)) = (before_pos, after_pos) {
                    if bp >= ap {
                        findings.push(Finding::error(format!(
                            "ordering violation: '{}' (step {}) must come before '{}' (step {})",
                            before, bp + 1, after, ap + 1
                        )));
                    }
                }
            }
        }
    }

    fn check_no_duplicates(&self, plan: &VerifiablePlan, findings: &mut Vec<Finding>) {
        let mut seen = HashSet::new();
        for step in &plan.steps {
            let key = format!("{}:{}->{}", step.service, step.from_index, step.to_index);
            if !seen.insert(key.clone()) {
                findings.push(Finding::warning(format!(
                    "duplicate step detected: {}", key
                )));
            }
        }
    }

    fn check_service_references(&self, plan: &VerifiablePlan, findings: &mut Vec<Finding>) {
        let known: HashSet<&str> = plan.services.iter().map(|s| s.as_str()).collect();
        for step in &plan.steps {
            if !known.contains(step.service.as_str()) {
                findings.push(Finding::error(format!(
                    "step {} references unknown service '{}'",
                    step.step, step.service
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::OutputFormat;
    use crate::output::OutputManager;

    fn sample_plan() -> VerifiablePlan {
        VerifiablePlan {
            plan_id: "test-plan".into(),
            start_state: vec![0, 0],
            target_state: vec![1, 1],
            services: vec!["api".into(), "db".into()],
            steps: vec![
                VerifiableStep {
                    step: 1, service: "api".into(),
                    from_version: "v1".into(), to_version: "v2".into(),
                    service_index: 0, from_index: 0, to_index: 1,
                },
                VerifiableStep {
                    step: 2, service: "db".into(),
                    from_version: "v1".into(), to_version: "v2".into(),
                    service_index: 1, from_index: 0, to_index: 1,
                },
            ],
        }
    }

    #[test]
    fn test_check_completeness_passes() {
        let plan = sample_plan();
        let args = VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: true,
            check_completeness: true,
            show_all: false,
        };
        let cmd = VerifyCommand::new(args);
        let mut findings = Vec::new();
        cmd.check_completeness(&plan, &mut findings);
        assert!(findings.iter().any(|f| f.message.contains("passed")));
    }

    #[test]
    fn test_check_completeness_fails() {
        let mut plan = sample_plan();
        plan.target_state = vec![2, 1]; // unreachable
        let args = VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: false,
            check_completeness: true,
            show_all: false,
        };
        let cmd = VerifyCommand::new(args);
        let mut findings = Vec::new();
        cmd.check_completeness(&plan, &mut findings);
        assert!(findings.iter().any(|f| f.severity == FindingSeverity::Error));
    }

    #[test]
    fn test_check_monotonicity_passes() {
        let plan = sample_plan();
        let cmd = VerifyCommand::new(VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: true,
            check_completeness: false,
            show_all: false,
        });
        let mut findings = Vec::new();
        cmd.check_monotonicity(&plan, &mut findings);
        assert!(findings.iter().all(|f| f.severity != FindingSeverity::Error));
    }

    #[test]
    fn test_check_monotonicity_warns_on_revisit() {
        let mut plan = sample_plan();
        plan.steps.push(VerifiableStep {
            step: 3, service: "api".into(),
            from_version: "v2".into(), to_version: "v3".into(),
            service_index: 0, from_index: 1, to_index: 2,
        });
        let cmd = VerifyCommand::new(VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: true,
            check_completeness: false,
            show_all: false,
        });
        let mut findings = Vec::new();
        cmd.check_monotonicity(&plan, &mut findings);
        assert!(findings.iter().any(|f| f.message.contains("modified more than once")));
    }

    #[test]
    fn test_check_step_consistency() {
        let plan = sample_plan();
        let cmd = VerifyCommand::new(VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: false,
            check_completeness: false,
            show_all: false,
        });
        let mut findings = Vec::new();
        cmd.check_step_consistency(&plan, &mut findings);
        assert!(findings.iter().all(|f| f.severity != FindingSeverity::Error));
    }

    #[test]
    fn test_check_step_consistency_fails() {
        let mut plan = sample_plan();
        plan.steps[1].from_index = 5; // wrong
        let cmd = VerifyCommand::new(VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: false,
            check_completeness: false,
            show_all: false,
        });
        let mut findings = Vec::new();
        cmd.check_step_consistency(&plan, &mut findings);
        assert!(findings.iter().any(|f| f.severity == FindingSeverity::Error));
    }

    #[test]
    fn test_check_service_references() {
        let mut plan = sample_plan();
        plan.steps[0].service = "unknown_svc".into();
        let cmd = VerifyCommand::new(VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: false,
            check_completeness: false,
            show_all: false,
        });
        let mut findings = Vec::new();
        cmd.check_service_references(&plan, &mut findings);
        assert!(findings.iter().any(|f| f.message.contains("unknown")));
    }

    #[test]
    fn test_check_ordering_constraint() {
        let plan = sample_plan(); // api(step 1) then db(step 2)
        let constraints = vec![VerifyConstraint::Ordering {
            before: "db".into(), after: "api".into(), // requires db before api -> violation
        }];
        let cmd = VerifyCommand::new(VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: false,
            check_completeness: false,
            show_all: false,
        });
        let mut findings = Vec::new();
        cmd.check_constraints(&plan, &constraints, &mut findings);
        assert!(findings.iter().any(|f| f.message.contains("ordering violation")));
    }

    #[test]
    fn test_check_forbidden_constraint() {
        let plan = sample_plan();
        let constraints = vec![VerifyConstraint::Forbidden {
            service: "api".into(), version: 1,
        }];
        let cmd = VerifyCommand::new(VerifyArgs {
            plan_file: "/tmp/dummy".into(),
            constraints_file: None,
            check_monotonicity: false,
            check_completeness: false,
            show_all: false,
        });
        let mut findings = Vec::new();
        cmd.check_constraints(&plan, &constraints, &mut findings);
        assert!(findings.iter().any(|f| f.message.contains("forbidden")));
    }

    #[test]
    fn test_plan_serialization_roundtrip() {
        let plan = sample_plan();
        let json = serde_json::to_string(&plan).unwrap();
        let parsed: VerifiablePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.plan_id, plan.plan_id);
        assert_eq!(parsed.steps.len(), plan.steps.len());
    }
}
