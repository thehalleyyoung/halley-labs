//! Implementation of the `envelope` subcommand.
//!
//! Computes a safety envelope for a deployment plan by annotating each state
//! along the plan path with reachability information, point-of-no-return
//! detection, and robustness scoring under adversarial fault models.

use std::collections::HashSet;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cli::EnvelopeArgs;
use crate::config_loader::SafeStepConfig;
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Plan representation for envelope computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopePlan {
    pub plan_id: String,
    pub services: Vec<String>,
    pub start_state: Vec<u16>,
    pub target_state: Vec<u16>,
    pub steps: Vec<EnvelopeStep>,
}

/// A step in the plan for envelope analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeStep {
    pub step_index: usize,
    pub service: String,
    pub from_version: u16,
    pub to_version: u16,
    #[serde(default)]
    pub reversible: bool,
}

/// Safety status of a state in the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyStatus {
    /// State is safe: can reach target and retreat to start.
    Safe,
    /// State can reach target but retreat is costly.
    Warning,
    /// State cannot safely retreat; commitment point passed.
    PointOfNoReturn,
}

impl std::fmt::Display for SafetyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safe => write!(f, "SAFE"),
            Self::Warning => write!(f, "WARNING"),
            Self::PointOfNoReturn => write!(f, "PNR"),
        }
    }
}

/// Annotation for a single state along the plan path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateAnnotation {
    pub step_index: usize,
    pub state: Vec<u16>,
    pub status: SafetyStatus,
    pub can_reach_target: bool,
    pub can_retreat_to_start: bool,
    pub risk_score: f64,
}

/// Result of robustness analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessResult {
    pub score: f64,
    pub adversary_budget: usize,
    pub failure_scenarios: Vec<FailureScenario>,
    pub overall_safe: bool,
}

/// A failure scenario under adversarial conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureScenario {
    pub step: usize,
    pub description: String,
    pub severity: f64,
}

/// Full envelope result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeResult {
    pub plan_id: String,
    pub annotations: Vec<StateAnnotation>,
    pub pnr_steps: Vec<usize>,
    pub robustness: RobustnessResult,
    pub overall_status: String,
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

pub struct EnvelopeCommand {
    args: EnvelopeArgs,
    _config: SafeStepConfig,
}

impl EnvelopeCommand {
    pub fn new(args: EnvelopeArgs, config: SafeStepConfig) -> Self {
        Self { args, _config: config }
    }

    pub fn execute(&self, output: &mut OutputManager) -> Result<()> {
        info!("computing safety envelope from {:?}", self.args.plan_file);

        let plan = self.load_plan()?;
        self.validate_plan(&plan)?;

        // Build state sequence along the plan path.
        let states = self.build_state_sequence(&plan);

        // Annotate each state.
        let annotations = self.annotate_states(&plan, &states);

        // Detect PNR steps.
        let pnr_steps: Vec<usize> = if self.args.detect_pnr {
            annotations.iter()
                .filter(|a| a.status == SafetyStatus::PointOfNoReturn)
                .map(|a| a.step_index)
                .collect()
        } else {
            Vec::new()
        };

        // Compute robustness.
        let robustness = self.compute_robustness(&annotations);

        let meets_robustness_threshold =
            !self.args.robustness || robustness.score >= self.args.min_robustness;

        let overall_status = if meets_robustness_threshold {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        };

        let result = EnvelopeResult {
            plan_id: plan.plan_id.clone(),
            annotations: annotations.clone(),
            pnr_steps: pnr_steps.clone(),
            robustness: robustness.clone(),
            overall_status: overall_status.clone(),
        };

        self.render(output, &result);

        if self.args.robustness && overall_status == "FAIL" {
            anyhow::bail!(
                "safety envelope check failed: robustness {:.2} < minimum {:.2}",
                robustness.score, self.args.min_robustness
            );
        }

        Ok(())
    }

    fn load_plan(&self) -> Result<EnvelopePlan> {
        let content = std::fs::read_to_string(&self.args.plan_file)
            .with_context(|| format!("reading plan file {:?}", self.args.plan_file))?;

        if self.args.plan_file.extension().map_or(false, |e| e == "yaml" || e == "yml") {
            serde_yaml::from_str(&content)
                .with_context(|| "parsing YAML plan file")
        } else {
            serde_json::from_str(&content)
                .with_context(|| "parsing JSON plan file")
        }
    }

    fn validate_plan(&self, plan: &EnvelopePlan) -> Result<()> {
        if plan.services.is_empty() {
            anyhow::bail!("plan has no services");
        }
        if plan.start_state.len() != plan.services.len() {
            anyhow::bail!(
                "start_state length {} != services length {}",
                plan.start_state.len(), plan.services.len()
            );
        }
        if plan.target_state.len() != plan.services.len() {
            anyhow::bail!(
                "target_state length {} != services length {}",
                plan.target_state.len(), plan.services.len()
            );
        }
        if plan.steps.is_empty() {
            anyhow::bail!("plan has no steps");
        }
        // Check all step services exist.
        let svc_set: HashSet<&str> = plan.services.iter().map(|s| s.as_str()).collect();
        for step in &plan.steps {
            if !svc_set.contains(step.service.as_str()) {
                anyhow::bail!("step {} references unknown service '{}'", step.step_index, step.service);
            }
        }
        Ok(())
    }

    fn build_state_sequence(&self, plan: &EnvelopePlan) -> Vec<Vec<u16>> {
        let mut states = Vec::with_capacity(plan.steps.len() + 1);
        let mut current = plan.start_state.clone();
        states.push(current.clone());

        for step in &plan.steps {
            if let Some(idx) = plan.services.iter().position(|s| s == &step.service) {
                current[idx] = step.to_version;
            }
            states.push(current.clone());
        }
        states
    }

    fn annotate_states(
        &self,
        plan: &EnvelopePlan,
        states: &[Vec<u16>],
    ) -> Vec<StateAnnotation> {
        let target = &plan.target_state;
        let start = &plan.start_state;
        let total = states.len();

        states.iter().enumerate().map(|(i, state)| {
            let can_reach_target = self.forward_reachable(state, target, plan, i);
            let can_retreat = self.backward_reachable(state, start, plan, i);
            let risk = self.compute_risk(state, start, target, i, total);

            let status = if can_reach_target && can_retreat {
                if risk > 0.7 { SafetyStatus::Warning } else { SafetyStatus::Safe }
            } else if can_reach_target && !can_retreat {
                SafetyStatus::PointOfNoReturn
            } else {
                SafetyStatus::PointOfNoReturn
            };

            StateAnnotation {
                step_index: i,
                state: state.clone(),
                status,
                can_reach_target,
                can_retreat_to_start: can_retreat,
                risk_score: risk,
            }
        }).collect()
    }

    fn forward_reachable(&self, current: &[u16], target: &[u16], plan: &EnvelopePlan, from_step: usize) -> bool {
        // From this state, can we reach the target by applying remaining steps?
        let mut state = current.to_vec();
        for step in plan.steps.iter().skip(from_step) {
            if let Some(idx) = plan.services.iter().position(|s| s == &step.service) {
                state[idx] = step.to_version;
            }
        }
        state == target
    }

    fn backward_reachable(&self, _current: &[u16], _start: &[u16], plan: &EnvelopePlan, at_step: usize) -> bool {
        // Can we retreat to start? Only if all steps up to this point are reversible.
        plan.steps.iter().take(at_step).all(|s| s.reversible)
    }

    fn compute_risk(&self, state: &[u16], start: &[u16], target: &[u16], step_idx: usize, total_steps: usize) -> f64 {
        // Risk is based on: distance from start, distance to target, and step position.
        let dist_from_start: u32 = state.iter().zip(start.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
            .sum();
        let dist_to_target: u32 = state.iter().zip(target.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
            .sum();
        let total_dist = dist_from_start + dist_to_target;
        if total_dist == 0 {
            return 0.0;
        }

        let progress = if total_steps > 1 {
            step_idx as f64 / (total_steps - 1) as f64
        } else {
            1.0
        };

        // Risk peaks in the middle of the plan.
        let position_risk = 4.0 * progress * (1.0 - progress);
        let distance_risk = dist_from_start as f64 / total_dist as f64;

        (position_risk * 0.6 + distance_risk * 0.4).min(1.0)
    }

    fn compute_robustness(&self, annotations: &[StateAnnotation]) -> RobustnessResult {
        let budget = self.args.adversary_budget;
        let mut failure_scenarios = Vec::new();

        for ann in annotations {
            if ann.status == SafetyStatus::PointOfNoReturn {
                failure_scenarios.push(FailureScenario {
                    step: ann.step_index,
                    description: format!(
                        "at step {}, the plan cannot safely retreat (risk={:.2})",
                        ann.step_index, ann.risk_score
                    ),
                    severity: ann.risk_score,
                });
            }
        }

        // Check combinations of fault-affected steps.
        if budget > 0 && annotations.len() > budget {
            let warning_steps: Vec<&StateAnnotation> = annotations.iter()
                .filter(|a| a.status == SafetyStatus::Warning)
                .collect();

            if warning_steps.len() >= budget {
                let combined_risk: f64 = warning_steps.iter()
                    .take(budget)
                    .map(|a| a.risk_score)
                    .sum::<f64>() / budget as f64;

                if combined_risk > 0.5 {
                    failure_scenarios.push(FailureScenario {
                        step: warning_steps[0].step_index,
                        description: format!(
                            "combined risk of {} warning states exceeds threshold (avg={:.2})",
                            budget, combined_risk
                        ),
                        severity: combined_risk,
                    });
                }
            }
        }

        let pnr_count = annotations.iter()
            .filter(|a| a.status == SafetyStatus::PointOfNoReturn)
            .count();
        let total = annotations.len().max(1);
        let safe_fraction = 1.0 - (pnr_count as f64 / total as f64);

        // Penalize based on failure scenarios.
        let scenario_penalty = failure_scenarios.iter()
            .map(|s| s.severity * 0.1)
            .sum::<f64>()
            .min(0.5);

        let score = (safe_fraction - scenario_penalty).max(0.0).min(1.0);
        let overall_safe = score >= self.args.min_robustness;

        RobustnessResult {
            score,
            adversary_budget: budget,
            failure_scenarios,
            overall_safe,
        }
    }

    fn render(&self, output: &mut OutputManager, result: &EnvelopeResult) {
        let colors = output.colors().clone();

        output.writeln(&format!("Safety Envelope: plan '{}'", result.plan_id));
        output.writeln(&format!("Overall: {}", if result.overall_status == "PASS" {
            colors.safe(&result.overall_status)
        } else {
            colors.error(&result.overall_status)
        }));
        output.writeln("");

        output.writeln("State Annotations:");
        for ann in &result.annotations {
            let status_str = match ann.status {
                SafetyStatus::Safe => colors.safe(&ann.status.to_string()),
                SafetyStatus::Warning => colors.warning(&ann.status.to_string()),
                SafetyStatus::PointOfNoReturn => colors.error(&ann.status.to_string()),
            };
            output.writeln(&format!(
                "  Step {:>3}: [{}] risk={:.2} reach_target={} retreat={}",
                ann.step_index, status_str, ann.risk_score,
                ann.can_reach_target, ann.can_retreat_to_start
            ));
        }

        if !result.pnr_steps.is_empty() {
            output.writeln("");
            output.writeln(&format!(
                "Points of No Return: {:?}", result.pnr_steps
            ));
        }

        output.writeln("");
        output.writeln(&format!(
            "Robustness: {:.2} (budget={}, scenarios={})",
            result.robustness.score,
            result.robustness.adversary_budget,
            result.robustness.failure_scenarios.len()
        ));

        for scenario in &result.robustness.failure_scenarios {
            output.writeln(&format!(
                "  - step {}: {} (severity={:.2})",
                scenario.step, scenario.description, scenario.severity
            ));
        }

        let _ = output.render_value(&result);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::OutputFormat;

    fn sample_plan() -> EnvelopePlan {
        EnvelopePlan {
            plan_id: "test-001".to_string(),
            services: vec!["api".to_string(), "db".to_string()],
            start_state: vec![0, 0],
            target_state: vec![1, 1],
            steps: vec![
                EnvelopeStep {
                    step_index: 0,
                    service: "api".to_string(),
                    from_version: 0,
                    to_version: 1,
                    reversible: true,
                },
                EnvelopeStep {
                    step_index: 1,
                    service: "db".to_string(),
                    from_version: 0,
                    to_version: 1,
                    reversible: true,
                },
            ],
        }
    }

    fn sample_plan_with_pnr() -> EnvelopePlan {
        EnvelopePlan {
            plan_id: "pnr-001".to_string(),
            services: vec!["api".to_string(), "db".to_string()],
            start_state: vec![0, 0],
            target_state: vec![1, 1],
            steps: vec![
                EnvelopeStep {
                    step_index: 0,
                    service: "db".to_string(),
                    from_version: 0,
                    to_version: 1,
                    reversible: false,
                },
                EnvelopeStep {
                    step_index: 1,
                    service: "api".to_string(),
                    from_version: 0,
                    to_version: 1,
                    reversible: true,
                },
            ],
        }
    }

    fn make_command(plan: &EnvelopePlan, detect_pnr: bool) -> (EnvelopeCommand, std::path::PathBuf) {
        let dir = std::env::temp_dir().join("safestep-test-envelope");
        std::fs::create_dir_all(&dir).unwrap();
        let plan_path = dir.join("plan.json");
        std::fs::write(&plan_path, serde_json::to_string_pretty(plan).unwrap()).unwrap();

        let args = EnvelopeArgs {
            plan_file: plan_path.clone(),
            detailed: false,
            robustness: false,
            adversary_budget: 1,
            detect_pnr,
            min_robustness: 0.8,
        };
        let config = SafeStepConfig::default();
        (EnvelopeCommand::new(args, config), plan_path)
    }

    #[test]
    fn test_build_state_sequence() {
        let plan = sample_plan();
        let (cmd, _path) = make_command(&plan, false);
        let states = cmd.build_state_sequence(&plan);
        assert_eq!(states.len(), 3);
        assert_eq!(states[0], vec![0, 0]);
        assert_eq!(states[1], vec![1, 0]);
        assert_eq!(states[2], vec![1, 1]);
    }

    #[test]
    fn test_all_safe_annotations() {
        let plan = sample_plan();
        let (cmd, _path) = make_command(&plan, false);
        let states = cmd.build_state_sequence(&plan);
        let annotations = cmd.annotate_states(&plan, &states);
        assert_eq!(annotations.len(), 3);
        // Start state is always safe.
        assert_eq!(annotations[0].status, SafetyStatus::Safe);
    }

    #[test]
    fn test_pnr_detection() {
        let plan = sample_plan_with_pnr();
        let (cmd, _path) = make_command(&plan, true);
        let states = cmd.build_state_sequence(&plan);
        let annotations = cmd.annotate_states(&plan, &states);
        // After step 0 (irreversible db), state should be PNR.
        let pnr_found = annotations.iter().any(|a| a.status == SafetyStatus::PointOfNoReturn);
        assert!(pnr_found, "expected PNR annotation for irreversible step");
    }

    #[test]
    fn test_robustness_all_safe() {
        let plan = sample_plan();
        let (cmd, _path) = make_command(&plan, false);
        let states = cmd.build_state_sequence(&plan);
        let annotations = cmd.annotate_states(&plan, &states);
        let robustness = cmd.compute_robustness(&annotations);
        assert!(robustness.score > 0.5, "expected decent robustness for all-safe plan");
    }

    #[test]
    fn test_robustness_with_pnr() {
        let plan = sample_plan_with_pnr();
        let (cmd, _path) = make_command(&plan, true);
        let states = cmd.build_state_sequence(&plan);
        let annotations = cmd.annotate_states(&plan, &states);
        let robustness = cmd.compute_robustness(&annotations);
        assert!(!robustness.failure_scenarios.is_empty(), "expected failure scenarios for PNR plan");
    }

    #[test]
    fn test_validate_plan_empty_services() {
        let plan = EnvelopePlan {
            plan_id: "bad".to_string(),
            services: vec![],
            start_state: vec![],
            target_state: vec![],
            steps: vec![],
        };
        let args = EnvelopeArgs {
            plan_file: std::path::PathBuf::from("/tmp/fake.json"),
            detailed: false,
            robustness: false,
            adversary_budget: 1,
            detect_pnr: false,
            min_robustness: 0.8,
        };
        let cmd = EnvelopeCommand::new(args, SafeStepConfig::default());
        assert!(cmd.validate_plan(&plan).is_err());
    }

    #[test]
    fn test_compute_risk_start_state() {
        let plan = sample_plan();
        let (cmd, _path) = make_command(&plan, false);
        let risk = cmd.compute_risk(&[0, 0], &[0, 0], &[1, 1], 0, 3);
        assert!(risk < 0.5, "start state should have low risk, got {}", risk);
    }

    #[test]
    fn test_execute_pass() {
        let plan = sample_plan();
        let (cmd, _path) = make_command(&plan, false);
        let mut out = OutputManager::new(OutputFormat::Text, false);
        let result = cmd.execute(&mut out);
        assert!(result.is_ok(), "expected pass for all-reversible plan: {:?}", result);
    }

    #[test]
    fn test_safety_status_display() {
        assert_eq!(SafetyStatus::Safe.to_string(), "SAFE");
        assert_eq!(SafetyStatus::Warning.to_string(), "WARNING");
        assert_eq!(SafetyStatus::PointOfNoReturn.to_string(), "PNR");
    }
}
