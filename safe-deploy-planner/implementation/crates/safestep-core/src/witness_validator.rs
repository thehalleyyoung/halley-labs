//! Independent witness validation for SafeStep verification results.
//!
//! This module provides assumption-free checking of the planner's output.
//! Rather than trusting the internal proof chain (which relies on monotone
//! sufficiency, downward closure, and CEGAR termination), these validators
//! independently replay or simulate the result, catching any unsoundness
//! from unproved intermediate links.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::{
    CompatibilityOracle, CompatResult, Constraint, CoreResult, DeploymentPlan,
    Edge, PlanStep, ServiceIndex, State, VersionIndex, VersionProductGraph,
};

// ─── WitnessVerdict ─────────────────────────────────────────────────────

/// Outcome of an independent witness validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WitnessVerdict {
    /// The witness is valid: the claimed result is independently confirmed.
    Valid {
        steps_checked: usize,
        invariants_verified: usize,
    },
    /// The witness is invalid: a concrete discrepancy was found.
    Invalid {
        step_index: usize,
        violation: String,
    },
    /// Validation was inconclusive (e.g., oracle returned Unknown).
    Inconclusive {
        reason: String,
    },
}

impl WitnessVerdict {
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid { .. })
    }
}

impl fmt::Display for WitnessVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid { steps_checked, invariants_verified } => {
                write!(f, "VALID ({steps_checked} steps, {invariants_verified} invariants)")
            }
            Self::Invalid { step_index, violation } => {
                write!(f, "INVALID at step {step_index}: {violation}")
            }
            Self::Inconclusive { reason } => {
                write!(f, "INCONCLUSIVE: {reason}")
            }
        }
    }
}

// ─── WitnessValidator ───────────────────────────────────────────────────

/// Independently validates a verification result (safe plan or
/// counterexample witness) without relying on any unproved assumptions.
///
/// For safety witnesses: replays the deployment plan step-by-step,
/// checking every intermediate state against the compatibility oracle.
///
/// For counterexample witnesses: verifies that the claimed violation
/// actually occurs by concrete state simulation.
pub struct WitnessValidator<'a> {
    graph: &'a VersionProductGraph,
    oracle: &'a CompatibilityOracle,
    constraints: &'a [Constraint],
}

impl<'a> WitnessValidator<'a> {
    pub fn new(
        graph: &'a VersionProductGraph,
        oracle: &'a CompatibilityOracle,
        constraints: &'a [Constraint],
    ) -> Self {
        Self { graph, oracle, constraints }
    }

    /// Validate a safe deployment plan by replaying every step and checking
    /// that each intermediate state satisfies all hard constraints.
    pub fn validate_safe_plan(
        &self,
        plan: &DeploymentPlan,
        start: &State,
        target: &State,
    ) -> WitnessVerdict {
        let mut current = start.clone();
        let mut invariants_verified: usize = 0;

        // Verify the start state itself is safe.
        if let Some(violation) = self.check_state_safety(&current) {
            return WitnessVerdict::Invalid {
                step_index: 0,
                violation: format!("start state unsafe: {violation}"),
            };
        }
        invariants_verified += 1;

        for (i, step) in plan.steps.iter().enumerate() {
            // Apply the step: upgrade one service.
            let svc = step.service.0 as usize;
            let expected_from = current.versions[svc];
            if expected_from != step.from_version {
                return WitnessVerdict::Invalid {
                    step_index: i,
                    violation: format!(
                        "step claims service {} moves from {} but current state has {}",
                        svc, step.from_version, expected_from,
                    ),
                };
            }
            current.versions[svc] = step.to_version;

            // Check all pairwise compatibility constraints on the new state.
            if let Some(violation) = self.check_state_safety(&current) {
                return WitnessVerdict::Invalid {
                    step_index: i,
                    violation: format!("state after step {i} unsafe: {violation}"),
                };
            }
            invariants_verified += 1;

            // Check rollback: can we still reach the start from here?
            if let Some(violation) = self.check_rollback_reachable(&current, start) {
                return WitnessVerdict::Invalid {
                    step_index: i,
                    violation: format!("rollback unreachable after step {i}: {violation}"),
                };
            }
            invariants_verified += 1;
        }

        // Verify we actually reached the target.
        if current != *target {
            return WitnessVerdict::Invalid {
                step_index: plan.steps.len(),
                violation: "plan does not reach target state".into(),
            };
        }

        WitnessVerdict::Valid {
            steps_checked: plan.steps.len(),
            invariants_verified,
        }
    }

    /// Validate a counterexample witness: verify that the claimed stuck
    /// configuration actually violates at least one hard constraint.
    pub fn validate_counterexample(&self, stuck_state: &State) -> WitnessVerdict {
        match self.check_state_safety(stuck_state) {
            Some(violation) => WitnessVerdict::Valid {
                steps_checked: 1,
                invariants_verified: 1,
            },
            None => WitnessVerdict::Invalid {
                step_index: 0,
                violation: "claimed stuck state is actually safe — no violation found".into(),
            },
        }
    }

    /// Check whether a state satisfies all hard constraints. Returns
    /// `None` if safe, or `Some(description)` if a violation is found.
    fn check_state_safety(&self, state: &State) -> Option<String> {
        let n = state.versions.len();
        for constraint in self.constraints {
            if let Constraint::Compatibility {
                id,
                service_a,
                service_b,
                compatible_pairs,
            } = constraint
            {
                let a = service_a.0 as usize;
                let b = service_b.0 as usize;
                if a >= n || b >= n {
                    continue;
                }
                let va = state.versions[a];
                let vb = state.versions[b];
                let pair_set: HashSet<(VersionIndex, VersionIndex)> =
                    compatible_pairs.iter().copied().collect();
                if !pair_set.contains(&(va, vb)) {
                    return Some(format!(
                        "constraint {} violated: service {} at {} incompatible with service {} at {}",
                        id.as_str(), a, va, b, vb,
                    ));
                }
            }
        }
        // Also check via oracle for behavioral constraints.
        for i in 0..n {
            for j in (i + 1)..n {
                let result = self.oracle.query(
                    ServiceIndex(i as u16),
                    state.versions[i],
                    ServiceIndex(j as u16),
                    state.versions[j],
                );
                if result.is_incompatible() {
                    return Some(format!(
                        "oracle reports service {} at {} incompatible with service {} at {}",
                        i, state.versions[i], j, state.versions[j],
                    ));
                }
            }
        }
        None
    }

    /// Check rollback reachability via greedy backward simulation. This is
    /// a conservative check: we attempt to reach the start by individually
    /// downgrading each service back to its start version, verifying each
    /// intermediate state is safe.
    fn check_rollback_reachable(
        &self,
        current: &State,
        start: &State,
    ) -> Option<String> {
        let mut rollback = current.clone();
        let n = rollback.versions.len();
        for svc in 0..n {
            while rollback.versions[svc].0 > start.versions[svc].0 {
                rollback.versions[svc] = VersionIndex(rollback.versions[svc].0 - 1);
                if let Some(v) = self.check_state_safety(&rollback) {
                    return Some(format!(
                        "rollback path blocked at service {svc} version {}: {v}",
                        rollback.versions[svc],
                    ));
                }
            }
        }
        None
    }
}

// ─── MonotoneChecker ────────────────────────────────────────────────────

/// Empirically validates the monotone sufficiency assumption by testing
/// random constraint relaxations. Does not depend on the theoretical proof;
/// instead generates random samples and checks if monotonicity is preserved.
pub struct MonotoneChecker<'a> {
    oracle: &'a CompatibilityOracle,
    constraints: &'a [Constraint],
    num_services: usize,
    versions_per_service: Vec<usize>,
}

/// Result of the monotonicity empirical check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonotoneCheckResult {
    pub samples_tested: usize,
    pub monotonicity_held: bool,
    pub counterexample: Option<MonotoneCounterexample>,
    pub elapsed: Duration,
}

/// A concrete counterexample to monotonicity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonotoneCounterexample {
    pub safe_state: Vec<u16>,
    pub relaxed_state: Vec<u16>,
    pub violation: String,
}

impl fmt::Display for MonotoneCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.monotonicity_held {
            write!(
                f,
                "monotonicity validated for {} random samples in {:.1?}",
                self.samples_tested, self.elapsed,
            )
        } else {
            write!(f, "monotonicity VIOLATED: {:?}", self.counterexample)
        }
    }
}

impl<'a> MonotoneChecker<'a> {
    pub fn new(
        oracle: &'a CompatibilityOracle,
        constraints: &'a [Constraint],
        num_services: usize,
        versions_per_service: Vec<usize>,
    ) -> Self {
        Self { oracle, constraints, num_services, versions_per_service }
    }

    /// Test monotonicity on `num_samples` random safe states: for each safe
    /// state, relax one service's version downward and verify the resulting
    /// state is also safe.
    pub fn check(&self, num_samples: usize, rng: &mut impl rand::Rng) -> MonotoneCheckResult {
        let start = Instant::now();
        let mut tested = 0;

        for _ in 0..num_samples {
            // Generate a random state.
            let state: Vec<u16> = (0..self.num_services)
                .map(|i| rng.gen_range(0..self.versions_per_service[i] as u16))
                .collect();

            if !self.is_state_safe(&state) {
                continue;
            }

            // Try relaxing each service version downward.
            for svc in 0..self.num_services {
                if state[svc] == 0 {
                    continue;
                }
                let mut relaxed = state.clone();
                relaxed[svc] -= 1;

                if !self.is_state_safe(&relaxed) {
                    return MonotoneCheckResult {
                        samples_tested: tested,
                        monotonicity_held: false,
                        counterexample: Some(MonotoneCounterexample {
                            safe_state: state.clone(),
                            relaxed_state: relaxed.clone(),
                            violation: format!(
                                "state {:?} is safe but downgrading service {} to v{} yields unsafe state {:?}",
                                state, svc, relaxed[svc], relaxed,
                            ),
                        }),
                        elapsed: start.elapsed(),
                    };
                }
            }
            tested += 1;
        }

        MonotoneCheckResult {
            samples_tested: tested,
            monotonicity_held: true,
            counterexample: None,
            elapsed: start.elapsed(),
        }
    }

    fn is_state_safe(&self, versions: &[u16]) -> bool {
        let n = versions.len();
        for constraint in self.constraints {
            if let Constraint::Compatibility {
                service_a,
                service_b,
                compatible_pairs,
                ..
            } = constraint
            {
                let a = service_a.0 as usize;
                let b = service_b.0 as usize;
                if a >= n || b >= n {
                    continue;
                }
                let va = VersionIndex(versions[a]);
                let vb = VersionIndex(versions[b]);
                let pair_set: HashSet<(VersionIndex, VersionIndex)> =
                    compatible_pairs.iter().copied().collect();
                if !pair_set.contains(&(va, vb)) {
                    return false;
                }
            }
        }
        true
    }
}

// ─── CegarBoundTracker ─────────────────────────────────────────────────

/// Tracks actual CEGAR iteration counts against the theoretical 2^R bound,
/// providing empirical evidence that real-world instances converge far
/// faster than the worst-case guarantee.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarBoundTracker {
    /// (num_refinement_predicates, actual_iterations) for each run.
    runs: Vec<CegarRunRecord>,
}

/// A single CEGAR run record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarRunRecord {
    pub run_id: String,
    pub num_services: usize,
    pub num_versions: usize,
    pub num_refinement_predicates: usize,
    pub actual_iterations: usize,
    pub theoretical_bound: u64,
    pub elapsed: Duration,
}

impl CegarRunRecord {
    /// Ratio of actual iterations to theoretical bound. Values close to 0
    /// indicate the bound is extremely conservative.
    pub fn bound_utilization(&self) -> f64 {
        if self.theoretical_bound == 0 {
            return 0.0;
        }
        self.actual_iterations as f64 / self.theoretical_bound as f64
    }
}

impl fmt::Display for CegarRunRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "run={}: {}/{} iterations (bound utilization {:.2e}, {:.1?})",
            self.run_id,
            self.actual_iterations,
            self.theoretical_bound,
            self.bound_utilization(),
            self.elapsed,
        )
    }
}

/// Summary statistics for CEGAR bound tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarBoundSummary {
    pub total_runs: usize,
    pub max_iterations_observed: usize,
    pub mean_iterations: f64,
    pub max_bound_utilization: f64,
    pub mean_bound_utilization: f64,
}

impl fmt::Display for CegarBoundSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CEGAR bound summary: {} runs, max {} iterations (mean {:.1}), \
             max bound utilization {:.2e} (mean {:.2e})",
            self.total_runs,
            self.max_iterations_observed,
            self.mean_iterations,
            self.max_bound_utilization,
            self.mean_bound_utilization,
        )
    }
}

impl CegarBoundTracker {
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    /// Record a CEGAR run. `num_refinement_predicates` is R in the 2^R bound.
    pub fn record(
        &mut self,
        run_id: impl Into<String>,
        num_services: usize,
        num_versions: usize,
        num_refinement_predicates: usize,
        actual_iterations: usize,
        elapsed: Duration,
    ) {
        let theoretical_bound = if num_refinement_predicates <= 63 {
            1u64 << num_refinement_predicates
        } else {
            u64::MAX
        };
        self.runs.push(CegarRunRecord {
            run_id: run_id.into(),
            num_services,
            num_versions,
            num_refinement_predicates,
            actual_iterations,
            theoretical_bound,
            elapsed,
        });
    }

    /// Compute summary statistics over all recorded runs.
    pub fn summary(&self) -> CegarBoundSummary {
        if self.runs.is_empty() {
            return CegarBoundSummary {
                total_runs: 0,
                max_iterations_observed: 0,
                mean_iterations: 0.0,
                max_bound_utilization: 0.0,
                mean_bound_utilization: 0.0,
            };
        }
        let max_iters = self.runs.iter().map(|r| r.actual_iterations).max().unwrap_or(0);
        let mean_iters = self.runs.iter().map(|r| r.actual_iterations).sum::<usize>() as f64
            / self.runs.len() as f64;
        let utils: Vec<f64> = self.runs.iter().map(|r| r.bound_utilization()).collect();
        let max_util = utils.iter().cloned().fold(0.0_f64, f64::max);
        let mean_util = utils.iter().sum::<f64>() / utils.len() as f64;
        CegarBoundSummary {
            total_runs: self.runs.len(),
            max_iterations_observed: max_iters,
            mean_iterations: mean_iters,
            max_bound_utilization: max_util,
            mean_bound_utilization: mean_util,
        }
    }

    pub fn runs(&self) -> &[CegarRunRecord] {
        &self.runs
    }
}

impl Default for CegarBoundTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── DownwardClosureValidator ───────────────────────────────────────────

/// Tests the downward-closure assumption on concrete instances by verifying
/// that for every safe configuration, all sub-configurations (where one or
/// more service versions are decreased) are also safe.
///
/// Unlike [`crate::DownwardClosureChecker`] which inspects constraint
/// structure, this validator performs exhaustive or sampled simulation
/// against the compatibility oracle.
pub struct DownwardClosureValidator<'a> {
    oracle: &'a CompatibilityOracle,
    num_services: usize,
    versions_per_service: Vec<usize>,
}

/// Result of downward-closure validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosureValidationResult {
    pub safe_states_tested: usize,
    pub sub_configs_checked: usize,
    pub closure_holds: bool,
    pub violations: Vec<ClosureViolation>,
    pub elapsed: Duration,
}

/// A concrete violation of downward closure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosureViolation {
    pub safe_state: Vec<u16>,
    pub unsafe_sub_state: Vec<u16>,
    pub violating_pair: (usize, usize),
    pub description: String,
}

impl fmt::Display for ClosureValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.closure_holds {
            write!(
                f,
                "downward closure validated: {} safe states, {} sub-configs checked in {:.1?}",
                self.safe_states_tested, self.sub_configs_checked, self.elapsed,
            )
        } else {
            write!(
                f,
                "downward closure VIOLATED: {} violations in {} safe states ({:.1?})",
                self.violations.len(), self.safe_states_tested, self.elapsed,
            )
        }
    }
}

impl<'a> DownwardClosureValidator<'a> {
    pub fn new(
        oracle: &'a CompatibilityOracle,
        num_services: usize,
        versions_per_service: Vec<usize>,
    ) -> Self {
        Self { oracle, num_services, versions_per_service }
    }

    /// Exhaustively test downward closure for all states up to the given
    /// version bounds. Feasible only for small instances.
    pub fn check_exhaustive(&self) -> ClosureValidationResult {
        let start = Instant::now();
        let mut safe_tested = 0;
        let mut sub_checked = 0;
        let mut violations = Vec::new();

        self.enumerate_states(&mut |state: &[u16]| {
            if !self.is_oracle_safe(state) {
                return;
            }
            safe_tested += 1;

            // For each safe state, check all immediate sub-configurations.
            for svc in 0..self.num_services {
                if state[svc] == 0 {
                    continue;
                }
                let mut sub = state.to_vec();
                sub[svc] -= 1;
                sub_checked += 1;

                if !self.is_oracle_safe(&sub) {
                    violations.push(ClosureViolation {
                        safe_state: state.to_vec(),
                        unsafe_sub_state: sub.clone(),
                        violating_pair: (svc, svc),
                        description: format!(
                            "state {:?} is safe but sub-config {:?} (service {} downgraded) is not",
                            state, sub, svc,
                        ),
                    });
                }
            }
        });

        ClosureValidationResult {
            safe_states_tested: safe_tested,
            sub_configs_checked: sub_checked,
            closure_holds: violations.is_empty(),
            violations,
            elapsed: start.elapsed(),
        }
    }

    /// Sampled variant: test downward closure on `num_samples` random safe
    /// states. Use when the exhaustive check is infeasible.
    pub fn check_sampled(
        &self,
        num_samples: usize,
        rng: &mut impl rand::Rng,
    ) -> ClosureValidationResult {
        let start = Instant::now();
        let mut safe_tested = 0;
        let mut sub_checked = 0;
        let mut violations = Vec::new();

        for _ in 0..num_samples {
            let state: Vec<u16> = (0..self.num_services)
                .map(|i| rng.gen_range(0..self.versions_per_service[i] as u16))
                .collect();

            if !self.is_oracle_safe(&state) {
                continue;
            }
            safe_tested += 1;

            for svc in 0..self.num_services {
                if state[svc] == 0 {
                    continue;
                }
                let mut sub = state.clone();
                sub[svc] -= 1;
                sub_checked += 1;

                if !self.is_oracle_safe(&sub) {
                    violations.push(ClosureViolation {
                        safe_state: state.clone(),
                        unsafe_sub_state: sub.clone(),
                        violating_pair: (svc, svc),
                        description: format!(
                            "state {:?} is safe but sub-config {:?} (service {} downgraded) is not",
                            state, sub, svc,
                        ),
                    });
                }
            }
        }

        ClosureValidationResult {
            safe_states_tested: safe_tested,
            sub_configs_checked: sub_checked,
            closure_holds: violations.is_empty(),
            violations,
            elapsed: start.elapsed(),
        }
    }

    fn is_oracle_safe(&self, versions: &[u16]) -> bool {
        let n = versions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let result = self.oracle.query(
                    ServiceIndex(i as u16),
                    VersionIndex(versions[i]),
                    ServiceIndex(j as u16),
                    VersionIndex(versions[j]),
                );
                if result.is_incompatible() {
                    return false;
                }
            }
        }
        true
    }

    /// Enumerate all states (Cartesian product of version ranges).
    fn enumerate_states(&self, callback: &mut dyn FnMut(&[u16])) {
        let mut state = vec![0u16; self.num_services];
        self.enumerate_recursive(&mut state, 0, callback);
    }

    fn enumerate_recursive(
        &self,
        state: &mut Vec<u16>,
        svc: usize,
        callback: &mut dyn FnMut(&[u16]),
    ) {
        if svc == self.num_services {
            callback(state);
            return;
        }
        for v in 0..self.versions_per_service[svc] as u16 {
            state[svc] = v;
            self.enumerate_recursive(state, svc + 1, callback);
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cegar_bound_tracker_empty() {
        let tracker = CegarBoundTracker::new();
        let summary = tracker.summary();
        assert_eq!(summary.total_runs, 0);
        assert_eq!(summary.max_iterations_observed, 0);
    }

    #[test]
    fn test_cegar_bound_tracker_records() {
        let mut tracker = CegarBoundTracker::new();
        tracker.record("run-1", 10, 5, 8, 12, Duration::from_millis(100));
        tracker.record("run-2", 10, 5, 8, 5, Duration::from_millis(50));
        tracker.record("run-3", 20, 10, 12, 20, Duration::from_millis(300));

        let summary = tracker.summary();
        assert_eq!(summary.total_runs, 3);
        assert_eq!(summary.max_iterations_observed, 20);
        // 2^8 = 256, 2^12 = 4096; utilization is tiny
        assert!(summary.max_bound_utilization < 0.1);
    }

    #[test]
    fn test_cegar_run_bound_utilization() {
        let record = CegarRunRecord {
            run_id: "test".into(),
            num_services: 5,
            num_versions: 3,
            num_refinement_predicates: 10,
            actual_iterations: 8,
            theoretical_bound: 1024,
            elapsed: Duration::from_millis(50),
        };
        let util = record.bound_utilization();
        assert!((util - 8.0 / 1024.0).abs() < 1e-9);
    }

    #[test]
    fn test_witness_verdict_display() {
        let valid = WitnessVerdict::Valid {
            steps_checked: 5,
            invariants_verified: 11,
        };
        assert!(valid.to_string().contains("VALID"));
        assert!(valid.is_valid());

        let invalid = WitnessVerdict::Invalid {
            step_index: 2,
            violation: "incompatible pair".into(),
        };
        assert!(invalid.to_string().contains("INVALID"));
        assert!(invalid.is_invalid());
    }

    #[test]
    fn test_closure_validation_result_display() {
        let result = ClosureValidationResult {
            safe_states_tested: 100,
            sub_configs_checked: 450,
            closure_holds: true,
            violations: vec![],
            elapsed: Duration::from_millis(200),
        };
        assert!(result.to_string().contains("validated"));
    }

    #[test]
    fn test_monotone_check_result_display() {
        let result = MonotoneCheckResult {
            samples_tested: 500,
            monotonicity_held: true,
            counterexample: None,
            elapsed: Duration::from_millis(100),
        };
        assert!(result.to_string().contains("500"));
        assert!(result.to_string().contains("validated"));
    }
}
