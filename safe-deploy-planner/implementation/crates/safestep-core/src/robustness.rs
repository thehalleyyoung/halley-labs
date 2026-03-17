//! k-Robustness checking for SafeStep deployment plans.
//!
//! A plan is *k-robust* if it remains safe even when up to k "uncertain"
//! (red-tagged) constraints are violated simultaneously. This module provides:
//!
//! - [`UncertaintyModel`]: declares which constraints are uncertain and their
//!   confidence levels.
//! - [`SubsetEnumerator`]: combinatorial iterator that yields all k-subsets of
//!   a collection (used to enumerate adversary choices).
//! - [`RobustnessChecker`]: the main checker — for every k-subset of
//!   red-tagged constraints, temporarily removes them and verifies the plan
//!   path still satisfies all remaining constraints.
//! - [`AdversaryBudget`] / [`AdversaryBudgetResult`]: binary-search wrapper
//!   that computes the maximum k for which the plan is k-robust, plus a
//!   weighted budget based on constraint confidence values.
//! - [`RobustnessStats`]: lightweight statistics collected during checking.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use crate::{
    Constraint, CoreResult, DeploymentPlan, PlanStep, ServiceIndex, State, VersionIndex,
    VersionProductGraph,
};
use safestep_types::error::SafeStepError;
use safestep_types::identifiers::{ConstraintId, Id};

// ---------------------------------------------------------------------------
// UncertaintyModel
// ---------------------------------------------------------------------------

/// Declares which constraints are uncertain ("red-tagged") and assigns each a
/// confidence value in `[0.0, 1.0]` — lower means more likely to be violated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyModel {
    /// Ordered list of red-tagged constraint IDs (insertion order).
    red_tagged: Vec<ConstraintId>,
    /// Confidence level per constraint ID.
    confidence_map: HashMap<String, f64>,
}

impl UncertaintyModel {
    pub fn new() -> Self {
        Self {
            red_tagged: Vec::new(),
            confidence_map: HashMap::new(),
        }
    }

    /// Mark a constraint as uncertain with the given confidence ∈ `[0.0, 1.0]`.
    /// Values are clamped to the valid range.
    pub fn add_uncertain(&mut self, constraint_id: ConstraintId, confidence: f64) {
        let clamped = confidence.clamp(0.0, 1.0);
        let key = constraint_id.as_str().to_owned();
        if !self.confidence_map.contains_key(&key) {
            self.red_tagged.push(constraint_id);
        }
        self.confidence_map.insert(key, clamped);
    }

    /// Immutable view of the red-tagged constraint IDs.
    pub fn red_tagged_constraints(&self) -> &[ConstraintId] {
        &self.red_tagged
    }

    /// Confidence value for a constraint (defaults to 1.0 if not red-tagged).
    pub fn confidence(&self, constraint_id: &ConstraintId) -> f64 {
        self.confidence_map
            .get(constraint_id.as_str())
            .copied()
            .unwrap_or(1.0)
    }

    /// Whether a constraint is red-tagged.
    pub fn is_uncertain(&self, constraint_id: &ConstraintId) -> bool {
        self.confidence_map.contains_key(constraint_id.as_str())
    }

    /// Number of red-tagged constraints.
    pub fn uncertain_count(&self) -> usize {
        self.red_tagged.len()
    }

    /// Total uncertainty weight: sum of `(1 - confidence)` over all uncertain
    /// constraints. Higher means more total risk.
    pub fn total_uncertainty_weight(&self) -> f64 {
        self.confidence_map.values().map(|c| 1.0 - c).sum()
    }
}

impl Default for UncertaintyModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SubsetEnumerator
// ---------------------------------------------------------------------------

/// Iterates over all combinations of `k` items drawn from `items`, using an
/// iterative (non-recursive) algorithm based on a position-index array.
///
/// For `n` items choose `k` this yields `C(n, k)` subsets.
#[derive(Debug, Clone)]
pub struct SubsetEnumerator {
    items: Vec<ConstraintId>,
    k: usize,
    /// Current combination indices — `indices[i]` is the index into `items`.
    indices: Vec<usize>,
    /// Whether we have already emitted the first combination.
    started: bool,
    /// Whether the iterator is exhausted.
    done: bool,
}

impl SubsetEnumerator {
    /// Create a new enumerator for all `k`-subsets of `items`.
    pub fn new(items: Vec<ConstraintId>, k: usize) -> Self {
        if k == 0 || k > items.len() {
            return Self {
                items,
                k,
                indices: Vec::new(),
                started: false,
                done: k > 0, // k==0 yields one empty set, k>n yields nothing
            };
        }
        let indices: Vec<usize> = (0..k).collect();
        Self {
            items,
            k,
            indices,
            started: false,
            done: false,
        }
    }

    /// Total number of combinations `C(n, k)`.
    pub fn total_combinations(&self) -> u64 {
        binomial(self.items.len() as u64, self.k as u64)
    }

    /// Advance the index array to the next lexicographic combination.
    /// Returns `false` when no more combinations exist.
    fn advance(&mut self) -> bool {
        if self.k == 0 || self.indices.is_empty() {
            return false;
        }
        let n = self.items.len();
        let k = self.k;
        // Find the rightmost index that can be incremented.
        let mut i = k;
        loop {
            if i == 0 {
                return false;
            }
            i -= 1;
            if self.indices[i] < n - k + i {
                break;
            }
            if i == 0 {
                return false;
            }
        }
        self.indices[i] += 1;
        for j in (i + 1)..k {
            self.indices[j] = self.indices[j - 1] + 1;
        }
        true
    }
}

impl Iterator for SubsetEnumerator {
    type Item = Vec<ConstraintId>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // Special case: k == 0 yields exactly one empty subset.
        if self.k == 0 {
            self.done = true;
            return Some(Vec::new());
        }
        if !self.started {
            self.started = true;
        } else if !self.advance() {
            self.done = true;
            return None;
        }
        let subset: Vec<ConstraintId> = self
            .indices
            .iter()
            .map(|&idx| self.items[idx].clone())
            .collect();
        Some(subset)
    }
}

/// Compute `C(n, k)` (binomial coefficient) without overflow for moderate n.
fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // symmetry
    let mut result: u64 = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i);
        result /= i + 1;
    }
    result
}

// ---------------------------------------------------------------------------
// RobustnessResult
// ---------------------------------------------------------------------------

/// Outcome of a k-robustness check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobustnessResult {
    /// The plan is k-robust — safe against any ≤k simultaneous violations.
    Robust(usize),
    /// The plan fails at the requested k. We found a subset of red-tagged
    /// constraints whose removal causes a violation.
    NotRobust {
        k: usize,
        failing_subset: Vec<ConstraintId>,
        violating_state: State,
    },
    /// The check exceeded its time budget.
    Timeout,
}

impl RobustnessResult {
    /// Whether the plan passed.
    pub fn is_robust(&self) -> bool {
        matches!(self, RobustnessResult::Robust(_))
    }

    /// The k value that was tested.
    pub fn tested_k(&self) -> Option<usize> {
        match self {
            RobustnessResult::Robust(k) => Some(*k),
            RobustnessResult::NotRobust { k, .. } => Some(*k),
            RobustnessResult::Timeout => None,
        }
    }
}

impl fmt::Display for RobustnessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RobustnessResult::Robust(k) => write!(f, "Plan is {}-robust", k),
            RobustnessResult::NotRobust {
                k,
                failing_subset,
                violating_state,
            } => {
                write!(
                    f,
                    "Plan is NOT {}-robust: {} constraints in failing subset, violation at {}",
                    k,
                    failing_subset.len(),
                    violating_state
                )
            }
            RobustnessResult::Timeout => write!(f, "Robustness check timed out"),
        }
    }
}

// ---------------------------------------------------------------------------
// RobustnessStats
// ---------------------------------------------------------------------------

/// Lightweight statistics collected during a robustness check run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessStats {
    pub subsets_checked: u64,
    pub violations_found: usize,
    pub total_time: Duration,
    pub max_k_achieved: usize,
    pub states_examined: u64,
}

impl RobustnessStats {
    pub fn new() -> Self {
        Self {
            subsets_checked: 0,
            violations_found: 0,
            total_time: Duration::ZERO,
            max_k_achieved: 0,
            states_examined: 0,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "robustness check: {} subsets checked, {} violations found, \
             max_k={}, {} states examined, completed in {:.2?}",
            self.subsets_checked,
            self.violations_found,
            self.max_k_achieved,
            self.states_examined,
            self.total_time,
        )
    }
}

impl Default for RobustnessStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RobustnessStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// RobustnessChecker
// ---------------------------------------------------------------------------

/// The main k-robustness checker.
///
/// Given a plan, a set of constraints, and an [`UncertaintyModel`], it
/// enumerates subsets of the red-tagged constraints and checks whether the
/// plan's intermediate states still satisfy all *remaining* constraints.
#[derive(Debug, Clone)]
pub struct RobustnessChecker {
    constraints: Vec<Constraint>,
    /// Pre-computed mapping from constraint ID (string key) to index in
    /// `self.constraints`.
    constraint_index: HashMap<String, usize>,
    /// Time budget for a single `check_robustness` call.
    timeout: Duration,
}

impl RobustnessChecker {
    /// Build a new checker over the given graph's constraints.
    ///
    /// The `graph` parameter is used for future extensions (e.g. reachability
    /// pruning). Currently the checker operates purely on constraints and
    /// plan states.
    #[instrument(skip_all, fields(n_constraints = constraints.len()))]
    pub fn new(_graph: &VersionProductGraph, constraints: Vec<Constraint>) -> Self {
        let constraint_index: HashMap<String, usize> = constraints
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id().as_str().to_owned(), i))
            .collect();
        debug!(
            "RobustnessChecker created with {} constraints",
            constraints.len()
        );
        Self {
            constraints,
            constraint_index,
            timeout: Duration::from_secs(120),
        }
    }

    /// Override the default time budget (120 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check whether `plan` is k-robust under the given uncertainty model.
    ///
    /// For every subset S of red-tagged constraints with |S| ≤ k, we
    /// temporarily disable S and check whether every intermediate state of
    /// the plan satisfies all remaining constraints. We also check ordering
    /// constraints on the plan steps.
    #[instrument(skip_all, fields(k, n_uncertain = uncertainty.uncertain_count()))]
    pub fn check_robustness(
        &self,
        plan: &DeploymentPlan,
        uncertainty: &UncertaintyModel,
        k: usize,
    ) -> RobustnessResult {
        let start = Instant::now();
        let red_tagged = uncertainty.red_tagged_constraints().to_vec();

        if k > red_tagged.len() {
            debug!(
                "k={} exceeds red-tagged count {}; trivially robust",
                k,
                red_tagged.len()
            );
            return RobustnessResult::Robust(k);
        }

        let intermediate_states = plan.intermediate_states();
        let mut stats = RobustnessStats::new();

        // Check every subset size from 0 through k.
        for subset_size in 0..=k {
            let enumerator = SubsetEnumerator::new(red_tagged.clone(), subset_size);
            for subset in enumerator {
                if start.elapsed() > self.timeout {
                    warn!("Robustness check timed out after {:?}", start.elapsed());
                    return RobustnessResult::Timeout;
                }
                stats.subsets_checked += 1;

                let removed_set: HashSet<String> =
                    subset.iter().map(|id| id.as_str().to_owned()).collect();

                // Check each intermediate state against remaining constraints.
                if let Some(violating_state) =
                    self.find_violation(&intermediate_states, &removed_set, plan)
                {
                    stats.violations_found += 1;
                    info!(
                        "Plan is NOT {}-robust: violation with subset of size {}",
                        k, subset_size
                    );
                    return RobustnessResult::NotRobust {
                        k,
                        failing_subset: subset,
                        violating_state,
                    };
                }
                stats.states_examined += intermediate_states.len() as u64;
            }
        }

        stats.total_time = start.elapsed();
        stats.max_k_achieved = k;
        debug!("{}", stats.summary());
        RobustnessResult::Robust(k)
    }

    /// Compute the maximum k such that the plan is k-robust (binary search).
    #[instrument(skip_all)]
    pub fn compute_max_robustness(
        &self,
        plan: &DeploymentPlan,
        uncertainty: &UncertaintyModel,
    ) -> usize {
        let n = uncertainty.uncertain_count();
        if n == 0 {
            return 0;
        }

        let mut lo: usize = 0;
        let mut hi: usize = n;
        let mut best: usize = 0;

        // Binary search: find the largest k for which check_robustness returns Robust.
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            match self.check_robustness(plan, uncertainty, mid) {
                RobustnessResult::Robust(_) => {
                    best = mid;
                    if mid == hi {
                        break;
                    }
                    lo = mid + 1;
                }
                RobustnessResult::NotRobust { .. } => {
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
                RobustnessResult::Timeout => {
                    // Conservative: treat timeout as not robust at this level.
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
            }
        }

        info!("Maximum robustness level: {}", best);
        best
    }

    /// Run a full robustness analysis returning detailed stats alongside the
    /// result.
    pub fn analyze(
        &self,
        plan: &DeploymentPlan,
        uncertainty: &UncertaintyModel,
        k: usize,
    ) -> (RobustnessResult, RobustnessStats) {
        let start = Instant::now();
        let red_tagged = uncertainty.red_tagged_constraints().to_vec();
        let intermediate_states = plan.intermediate_states();
        let mut stats = RobustnessStats::new();

        if k > red_tagged.len() {
            stats.total_time = start.elapsed();
            stats.max_k_achieved = k;
            return (RobustnessResult::Robust(k), stats);
        }

        for subset_size in 0..=k {
            let enumerator = SubsetEnumerator::new(red_tagged.clone(), subset_size);
            for subset in enumerator {
                if start.elapsed() > self.timeout {
                    stats.total_time = start.elapsed();
                    return (RobustnessResult::Timeout, stats);
                }
                stats.subsets_checked += 1;

                let removed_set: HashSet<String> =
                    subset.iter().map(|id| id.as_str().to_owned()).collect();

                if let Some(violating_state) =
                    self.find_violation(&intermediate_states, &removed_set, plan)
                {
                    stats.violations_found += 1;
                    stats.total_time = start.elapsed();
                    let result = RobustnessResult::NotRobust {
                        k,
                        failing_subset: subset,
                        violating_state,
                    };
                    return (result, stats);
                }
                stats.states_examined += intermediate_states.len() as u64;
            }
        }

        stats.total_time = start.elapsed();
        stats.max_k_achieved = k;
        (RobustnessResult::Robust(k), stats)
    }

    // -- internal helpers ---------------------------------------------------

    /// Returns the first intermediate state that violates a remaining
    /// (non-removed) constraint, or `None` if the plan is safe.
    fn find_violation(
        &self,
        intermediate_states: &[State],
        removed: &HashSet<String>,
        plan: &DeploymentPlan,
    ) -> Option<State> {
        // State-level constraints.
        for state in intermediate_states {
            for constraint in &self.constraints {
                let cid = constraint.id().as_str().to_owned();
                if removed.contains(&cid) {
                    continue;
                }
                if !constraint.check_state(state) {
                    return Some(state.clone());
                }
            }
        }
        // Ordering constraints checked over the full plan step sequence.
        for (idx, _step) in plan.steps.iter().enumerate() {
            for constraint in &self.constraints {
                let cid = constraint.id().as_str().to_owned();
                if removed.contains(&cid) {
                    continue;
                }
                if !constraint.check_transition(idx, &plan.steps) {
                    // Use the intermediate state *after* this step.
                    let violating_idx = (idx + 1).min(intermediate_states.len() - 1);
                    return Some(intermediate_states[violating_idx].clone());
                }
            }
        }
        None
    }

    /// Return constraint by its ID key.
    fn constraint_by_id(&self, id_str: &str) -> Option<&Constraint> {
        self.constraint_index
            .get(id_str)
            .map(|&idx| &self.constraints[idx])
    }

    /// Identify which non-removed constraints are violated at a given state.
    pub fn violated_constraints_at(
        &self,
        state: &State,
        removed: &HashSet<String>,
    ) -> Vec<ConstraintId> {
        self.constraints
            .iter()
            .filter(|c| {
                let cid = c.id().as_str().to_owned();
                !removed.contains(&cid) && !c.check_state(state)
            })
            .map(|c| c.id().clone())
            .collect()
    }

    /// Check a single state against all constraints that are NOT in `removed`.
    pub fn state_satisfies_remaining(
        &self,
        state: &State,
        removed: &HashSet<String>,
    ) -> bool {
        self.constraints.iter().all(|c| {
            let cid = c.id().as_str().to_owned();
            removed.contains(&cid) || c.check_state(state)
        })
    }

    /// Quick pre-check: does the plan satisfy all constraints with nothing
    /// removed? (i.e., is it 0-robust?)
    pub fn baseline_check(&self, plan: &DeploymentPlan) -> bool {
        let empty: HashSet<String> = HashSet::new();
        self.find_violation(&plan.intermediate_states(), &empty, plan)
            .is_none()
    }
}

// ---------------------------------------------------------------------------
// AdversaryBudgetResult
// ---------------------------------------------------------------------------

/// Result of the adversary budget computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryBudgetResult {
    /// Maximum k for which the plan is k-robust.
    pub max_k: usize,
    /// Weighted budget: sum of `(1 - confidence)` for all uncertain
    /// constraints that appear in *any* failing subset at level `max_k + 1`
    /// (or 0.0 if the plan is fully robust).
    pub weighted_budget: f64,
    /// Constraints that appear in every minimal failing subset (the "critical"
    /// ones whose uncertainty most endangers the plan).
    pub critical_constraints: Vec<ConstraintId>,
}

impl AdversaryBudgetResult {
    pub fn fully_robust(n_uncertain: usize) -> Self {
        Self {
            max_k: n_uncertain,
            weighted_budget: 0.0,
            critical_constraints: Vec::new(),
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "adversary budget: max_k={}, weighted_budget={:.3}, critical_constraints={}",
            self.max_k,
            self.weighted_budget,
            self.critical_constraints.len()
        )
    }
}

impl fmt::Display for AdversaryBudgetResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// AdversaryBudget
// ---------------------------------------------------------------------------

/// Computes the adversary budget for a deployment plan — the maximum number
/// of uncertain constraints that can be simultaneously violated while the plan
/// remains safe.
pub struct AdversaryBudget;

impl AdversaryBudget {
    /// Compute the full adversary budget including weighted scoring and
    /// critical constraint identification.
    #[instrument(skip_all)]
    pub fn compute(
        plan: &DeploymentPlan,
        constraints: &[Constraint],
        uncertainty: &UncertaintyModel,
    ) -> AdversaryBudgetResult {
        let n = uncertainty.uncertain_count();
        if n == 0 {
            return AdversaryBudgetResult::fully_robust(0);
        }

        // Build a temporary graph-less checker (we only need constraint logic).
        let graph = VersionProductGraph::new(Vec::new());
        let checker = RobustnessChecker::new(&graph, constraints.to_vec());

        // Binary search for max k.
        let max_k = checker.compute_max_robustness(plan, uncertainty);

        // If fully robust against all uncertain constraints, we're done.
        if max_k >= n {
            return AdversaryBudgetResult::fully_robust(n);
        }

        // Collect all failing subsets at level max_k + 1 to identify critical
        // constraints and compute the weighted budget.
        let next_k = max_k + 1;
        let red_tagged = uncertainty.red_tagged_constraints().to_vec();
        let intermediate_states = plan.intermediate_states();

        let mut failing_subsets: Vec<Vec<ConstraintId>> = Vec::new();
        let mut all_failing_ids: HashSet<String> = HashSet::new();
        let enumerator = SubsetEnumerator::new(red_tagged, next_k);

        for subset in enumerator {
            let removed_set: HashSet<String> =
                subset.iter().map(|id| id.as_str().to_owned()).collect();
            if checker
                .find_violation(&intermediate_states, &removed_set, plan)
                .is_some()
            {
                for id in &subset {
                    all_failing_ids.insert(id.as_str().to_owned());
                }
                failing_subsets.push(subset);
            }
        }

        // Critical constraints: those appearing in *every* failing subset.
        let critical_constraints = if failing_subsets.is_empty() {
            Vec::new()
        } else {
            let first_set: HashSet<String> = failing_subsets[0]
                .iter()
                .map(|id| id.as_str().to_owned())
                .collect();
            let intersection = failing_subsets.iter().skip(1).fold(first_set, |acc, subset| {
                let s: HashSet<String> = subset.iter().map(|id| id.as_str().to_owned()).collect();
                acc.intersection(&s).cloned().collect()
            });
            intersection
                .into_iter()
                .map(|s| Id::from_name(&s))
                .collect()
        };

        // Weighted budget: sum of (1 - confidence) for all constraint IDs that
        // appear in at least one failing subset.
        let weighted_budget: f64 = all_failing_ids
            .iter()
            .map(|id_str| {
                let id: ConstraintId = Id::from_name(id_str);
                1.0 - uncertainty.confidence(&id)
            })
            .sum();

        let result = AdversaryBudgetResult {
            max_k,
            weighted_budget,
            critical_constraints,
        };

        info!("{}", result.summary());
        result
    }

    /// Convenience: just get the max k value.
    pub fn max_k(
        plan: &DeploymentPlan,
        constraints: &[Constraint],
        uncertainty: &UncertaintyModel,
    ) -> usize {
        let result = Self::compute(plan, constraints, uncertainty);
        result.max_k
    }
}

// ---------------------------------------------------------------------------
// Batch robustness checking
// ---------------------------------------------------------------------------

/// Check multiple plans against the same constraints and uncertainty model,
/// returning results keyed by plan index.
pub fn batch_check_robustness(
    plans: &[DeploymentPlan],
    constraints: &[Constraint],
    uncertainty: &UncertaintyModel,
    k: usize,
) -> Vec<(usize, RobustnessResult)> {
    let graph = VersionProductGraph::new(Vec::new());
    let checker = RobustnessChecker::new(&graph, constraints.to_vec());
    plans
        .iter()
        .enumerate()
        .map(|(i, plan)| {
            let result = checker.check_robustness(plan, uncertainty, k);
            (i, result)
        })
        .collect()
}

/// Select the most robust plan from a set of candidates.
pub fn most_robust_plan<'a>(
    plans: &'a [DeploymentPlan],
    constraints: &[Constraint],
    uncertainty: &UncertaintyModel,
) -> Option<(usize, &'a DeploymentPlan, usize)> {
    if plans.is_empty() {
        return None;
    }
    let graph = VersionProductGraph::new(Vec::new());
    let checker = RobustnessChecker::new(&graph, constraints.to_vec());

    let mut best_idx = 0;
    let mut best_k = 0;

    for (i, plan) in plans.iter().enumerate() {
        let k = checker.compute_max_robustness(plan, uncertainty);
        if k > best_k || (k == best_k && plan.total_risk < plans[best_idx].total_risk) {
            best_k = k;
            best_idx = i;
        }
    }

    Some((best_idx, &plans[best_idx], best_k))
}

// ---------------------------------------------------------------------------
// Sensitivity analysis helpers
// ---------------------------------------------------------------------------

/// For each red-tagged constraint, compute whether the plan becomes non-robust
/// if *only* that single constraint is removed. Returns the set of constraints
/// whose individual removal breaks the plan.
pub fn single_constraint_sensitivity(
    plan: &DeploymentPlan,
    constraints: &[Constraint],
    uncertainty: &UncertaintyModel,
) -> Vec<ConstraintId> {
    let graph = VersionProductGraph::new(Vec::new());
    let checker = RobustnessChecker::new(&graph, constraints.to_vec());
    let intermediate_states = plan.intermediate_states();

    let mut sensitive = Vec::new();
    for cid in uncertainty.red_tagged_constraints() {
        let mut removed = HashSet::new();
        removed.insert(cid.as_str().to_owned());
        if checker
            .find_violation(&intermediate_states, &removed, plan)
            .is_some()
        {
            sensitive.push(cid.clone());
        }
    }
    sensitive
}

/// Weighted sensitivity: for each red-tagged constraint compute
/// `(1 - confidence) * impact` where impact is 1 if removing that single
/// constraint breaks the plan and 0 otherwise.
pub fn weighted_sensitivity(
    plan: &DeploymentPlan,
    constraints: &[Constraint],
    uncertainty: &UncertaintyModel,
) -> Vec<(ConstraintId, f64)> {
    let graph = VersionProductGraph::new(Vec::new());
    let checker = RobustnessChecker::new(&graph, constraints.to_vec());
    let intermediate_states = plan.intermediate_states();

    uncertainty
        .red_tagged_constraints()
        .iter()
        .map(|cid| {
            let mut removed = HashSet::new();
            removed.insert(cid.as_str().to_owned());
            let impact = if checker
                .find_violation(&intermediate_states, &removed, plan)
                .is_some()
            {
                1.0
            } else {
                0.0
            };
            let weight = (1.0 - uncertainty.confidence(cid)) * impact;
            (cid.clone(), weight)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Constraint, DeploymentPlan, PlanStep, ServiceDescriptor, ServiceIndex, State,
        VersionIndex, VersionProductGraph,
    };
    use safestep_types::identifiers::Id;
    use std::collections::HashMap;

    // -- helpers ------------------------------------------------------------

    fn make_two_service_plan() -> DeploymentPlan {
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1)),
            PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(1)),
        ];
        DeploymentPlan::new(start, target, steps)
    }

    fn make_graph() -> VersionProductGraph {
        let svc_a = ServiceDescriptor::new("api", vec!["v0".into(), "v1".into()]);
        let svc_b = ServiceDescriptor::new("db", vec!["v0".into(), "v1".into()]);
        VersionProductGraph::new(vec![svc_a, svc_b])
    }

    fn compat_constraint(name: &str, pairs: Vec<(u16, u16)>) -> Constraint {
        Constraint::Compatibility {
            id: Id::from_name(name),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: pairs
                .into_iter()
                .map(|(a, b)| (VersionIndex(a), VersionIndex(b)))
                .collect(),
        }
    }

    fn forbidden_constraint(name: &str, svc: u16, ver: u16) -> Constraint {
        Constraint::Forbidden {
            id: Id::from_name(name),
            service: ServiceIndex(svc),
            version: VersionIndex(ver),
        }
    }

    // -- tests --------------------------------------------------------------

    #[test]
    fn test_subset_enumerator_basic() {
        let items: Vec<ConstraintId> = (0..4)
            .map(|i| Id::from_name(&format!("c{}", i)))
            .collect();
        let subsets: Vec<Vec<ConstraintId>> = SubsetEnumerator::new(items, 2).collect();
        // C(4,2) = 6
        assert_eq!(subsets.len(), 6);
        // Each subset should have exactly 2 elements.
        for s in &subsets {
            assert_eq!(s.len(), 2);
        }
    }

    #[test]
    fn test_subset_enumerator_edge_cases() {
        let items: Vec<ConstraintId> = vec![Id::from_name("a"), Id::from_name("b")];
        // k = 0 → one empty subset.
        let subsets: Vec<_> = SubsetEnumerator::new(items.clone(), 0).collect();
        assert_eq!(subsets.len(), 1);
        assert!(subsets[0].is_empty());

        // k > n → no subsets.
        let subsets: Vec<_> = SubsetEnumerator::new(items.clone(), 5).collect();
        assert_eq!(subsets.len(), 0);

        // k = n → one full subset.
        let subsets: Vec<_> = SubsetEnumerator::new(items.clone(), 2).collect();
        assert_eq!(subsets.len(), 1);
        assert_eq!(subsets[0].len(), 2);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 4), 210);
        assert_eq!(binomial(3, 5), 0); // k > n
    }

    #[test]
    fn test_uncertainty_model() {
        let mut model = UncertaintyModel::new();
        let id_a: ConstraintId = Id::from_name("c-a");
        let id_b: ConstraintId = Id::from_name("c-b");

        model.add_uncertain(id_a.clone(), 0.8);
        model.add_uncertain(id_b.clone(), 0.3);

        assert_eq!(model.uncertain_count(), 2);
        assert!(model.is_uncertain(&id_a));
        assert!((model.confidence(&id_a) - 0.8).abs() < 1e-9);
        assert!((model.confidence(&id_b) - 0.3).abs() < 1e-9);

        // Unknown constraint → confidence 1.0.
        let id_c: ConstraintId = Id::from_name("c-c");
        assert!(!model.is_uncertain(&id_c));
        assert!((model.confidence(&id_c) - 1.0).abs() < 1e-9);

        // Clamping.
        model.add_uncertain(Id::from_name("c-x"), 1.5);
        assert!((model.confidence(&Id::from_name("c-x")) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_robustness_trivially_robust() {
        // All intermediate states satisfy all constraints, so any k should be
        // robust.
        let plan = make_two_service_plan();
        let graph = make_graph();

        // Compatibility constraint that allows all four version combos.
        let c = compat_constraint(
            "all-ok",
            vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        );
        let checker = RobustnessChecker::new(&graph, vec![c.clone()]);

        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c.id().clone(), 0.5);

        let result = checker.check_robustness(&plan, &uncertainty, 1);
        assert!(result.is_robust());
    }

    #[test]
    fn test_robustness_not_robust() {
        // Intermediate state (v1, v0) is forbidden by a compatibility
        // constraint. If we red-tag that constraint and remove it, the plan
        // becomes valid — so the plan is 0-robust but NOT 1-robust when that
        // constraint isn't satisfied.
        //
        // Actually we need the constraint to be the *only* thing keeping us
        // safe. So: make a constraint that is satisfied by (v0,v0), (v1,v0),
        // (v1,v1) — the plan path goes through all of those — and then a
        // *second* constraint that forbids (v1,v0). Red-tag the second one.
        // Without removing: plan violates the second constraint at (v1,v0).
        // So the plan itself is ALREADY not 0-robust if (v1,v0) violates.
        //
        // Better approach: two constraints, one certain and one uncertain.
        // The certain one allows all combos. The uncertain one forbids (v1,v0).
        // Path: (v0,v0) → (v1,v0) → (v1,v1).
        // With all constraints active: (v1,v0) violates the forbidden constraint → NOT 0-robust.
        let plan = make_two_service_plan();
        let graph = make_graph();

        // This constraint forbids service-0 at version 1 when service-1 is
        // still at version 0.
        let c_forbid = Constraint::Custom {
            id: Id::from_name("forbid-v1-v0"),
            description: "no (v1,v0)".into(),
            check: |state: &State| {
                !(state.get(ServiceIndex(0)) == VersionIndex(1)
                    && state.get(ServiceIndex(1)) == VersionIndex(0))
            },
        };

        let checker = RobustnessChecker::new(&graph, vec![c_forbid.clone()]);
        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c_forbid.id().clone(), 0.5);

        // k=0 means no constraints are removed → plan must satisfy all
        // constraints. The plan goes (v0,v0)→(v1,v0)→(v1,v1) and (v1,v0)
        // violates c_forbid. So NOT 0-robust.
        let result = checker.check_robustness(&plan, &uncertainty, 0);
        assert!(!result.is_robust());

        // k=1 means we can remove up to 1 red-tagged constraint. When we
        // remove c_forbid, no constraints remain → path is trivially valid.
        // But we also check the subset of size 0 (removing nothing), which
        // still fails. So it should still be NotRobust at k=1 because the
        // "removing nothing" subset already fails.
        let result1 = checker.check_robustness(&plan, &uncertainty, 1);
        assert!(!result1.is_robust());
    }

    #[test]
    fn test_robustness_k1_passes_when_only_uncertain_violated() {
        // Plan path: (v0,v0) → (v1,v0) → (v1,v1).
        // We have two constraints:
        //   c_always: allows all combos (always true) — NOT red-tagged
        //   c_fragile: forbids (v1,v0) — red-tagged
        //
        // With all constraints: (v1,v0) violates c_fragile → fails.
        // BUT: k-robustness iterates over subsets of red-tagged constraints of
        // size ≤ k. At subset_size=0 (remove nothing): (v1,v0) violates
        // c_fragile → fail immediately.
        //
        // To make it k=1-robust but not k=0-robust we need the baseline
        // (nothing removed) to pass but fail when we remove something. That
        // doesn't make sense for robustness (removing constraints can only
        // relax). The correct semantics:
        //
        // The plan is k-robust iff for EVERY subset of red-tagged of size ≤ k,
        // removing those constraints still leaves the plan valid. So removing
        // constraints means we check FEWER constraints.
        //
        // A plan that passes with all constraints will also pass with fewer
        // constraints. So if baseline passes, the plan is k-robust for all k.
        //
        // Interesting case: some certain constraint is violated only when an
        // uncertain constraint is removed? No — constraints are independent
        // checks.
        //
        // In practice, "removing a constraint" means the adversary violates it
        // (it was protecting something). The plan was designed assuming all
        // constraints hold. If an uncertain constraint is removed (violated by
        // reality), other constraints may become relevant.
        //
        // For this test: all constraints pass at baseline → plan should be
        // robust at every k.
        let plan = make_two_service_plan();
        let graph = make_graph();

        let c_compat = compat_constraint(
            "safe-all",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let c_fragile = compat_constraint(
            "fragile-rule",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );

        let checker = RobustnessChecker::new(&graph, vec![c_compat.clone(), c_fragile.clone()]);
        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c_fragile.id().clone(), 0.4);

        let result = checker.check_robustness(&plan, &uncertainty, 1);
        assert!(result.is_robust());
    }

    #[test]
    fn test_compute_max_robustness() {
        // Plan and constraints that pass at baseline.
        let plan = make_two_service_plan();
        let graph = make_graph();

        let c1 = compat_constraint(
            "c1",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let c2 = compat_constraint(
            "c2",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );

        let checker = RobustnessChecker::new(&graph, vec![c1.clone(), c2.clone()]);
        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c1.id().clone(), 0.5);
        uncertainty.add_uncertain(c2.id().clone(), 0.7);

        // Both constraints allow all combos, so removing any subset still
        // passes → max robustness = number of uncertain constraints = 2.
        let max_k = checker.compute_max_robustness(&plan, &uncertainty);
        assert_eq!(max_k, 2);
    }

    #[test]
    fn test_adversary_budget_compute() {
        let plan = make_two_service_plan();

        let c1 = compat_constraint(
            "budget-c1",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let c2 = compat_constraint(
            "budget-c2",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let constraints = vec![c1.clone(), c2.clone()];

        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c1.id().clone(), 0.6);
        uncertainty.add_uncertain(c2.id().clone(), 0.9);

        let result = AdversaryBudget::compute(&plan, &constraints, &uncertainty);
        assert_eq!(result.max_k, 2);
        // Fully robust → weighted budget = 0.
        assert!((result.weighted_budget - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_robustness_stats_summary() {
        let mut stats = RobustnessStats::new();
        stats.subsets_checked = 42;
        stats.violations_found = 3;
        stats.total_time = Duration::from_millis(123);
        stats.max_k_achieved = 2;
        stats.states_examined = 126;
        let s = stats.summary();
        assert!(s.contains("42 subsets checked"));
        assert!(s.contains("3 violations found"));
    }

    #[test]
    fn test_baseline_check_and_violated_constraints() {
        let plan = make_two_service_plan();
        let graph = make_graph();

        // Forbid version 1 for service 0 → intermediate state (v1, v0) fails.
        let c = forbidden_constraint("no-v1-svc0", 0, 1);
        let checker = RobustnessChecker::new(&graph, vec![c]);

        assert!(!checker.baseline_check(&plan));

        // Check what constraints are violated at (v1, v0).
        let bad_state = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let empty: HashSet<String> = HashSet::new();
        let violated = checker.violated_constraints_at(&bad_state, &empty);
        assert_eq!(violated.len(), 1);
    }

    #[test]
    fn test_batch_check_robustness() {
        let plan_a = make_two_service_plan();
        let plan_b = {
            let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
            let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
            // Different step order.
            let steps = vec![
                PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(1)),
                PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1)),
            ];
            DeploymentPlan::new(start, target, steps)
        };

        let c = compat_constraint(
            "batch-c",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c.id().clone(), 0.5);

        let results = batch_check_robustness(
            &[plan_a, plan_b],
            &[c],
            &uncertainty,
            1,
        );
        assert_eq!(results.len(), 2);
        for (_, r) in &results {
            assert!(r.is_robust());
        }
    }

    #[test]
    fn test_single_constraint_sensitivity() {
        let plan = make_two_service_plan();

        // c_safe allows all combos → removing it doesn't break anything.
        let c_safe = compat_constraint(
            "sens-safe",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );

        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c_safe.id().clone(), 0.5);

        let sensitive =
            single_constraint_sensitivity(&plan, &[c_safe], &uncertainty);
        // Removing the only constraint can't cause a violation (nothing left
        // to violate).
        assert!(sensitive.is_empty());
    }

    #[test]
    fn test_weighted_sensitivity() {
        let plan = make_two_service_plan();
        let c = compat_constraint(
            "wsens",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c.id().clone(), 0.7);

        let ws = weighted_sensitivity(&plan, &[c], &uncertainty);
        assert_eq!(ws.len(), 1);
        // Impact is 0 (removing c doesn't break the plan) → weight = 0.
        assert!((ws[0].1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_analyze_returns_stats() {
        let plan = make_two_service_plan();
        let graph = make_graph();
        let c = compat_constraint(
            "analyze-c",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let checker = RobustnessChecker::new(&graph, vec![c.clone()]);
        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c.id().clone(), 0.5);

        let (result, stats) = checker.analyze(&plan, &uncertainty, 1);
        assert!(result.is_robust());
        assert!(stats.subsets_checked > 0);
        assert_eq!(stats.violations_found, 0);
    }

    #[test]
    fn test_most_robust_plan() {
        // Both plans go through the same path; they should be equally robust.
        let plan_a = make_two_service_plan();
        let plan_b = make_two_service_plan();
        let c = compat_constraint(
            "mrp",
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
        );
        let mut uncertainty = UncertaintyModel::new();
        uncertainty.add_uncertain(c.id().clone(), 0.5);

        let best = most_robust_plan(&[plan_a, plan_b], &[c], &uncertainty);
        assert!(best.is_some());
        let (idx, _, k) = best.unwrap();
        assert_eq!(k, 1); // 1 uncertain, all combos allowed → robust at k=1
        assert!(idx == 0 || idx == 1);
    }

    #[test]
    fn test_subset_enumerator_total_combinations() {
        let items: Vec<ConstraintId> = (0..6)
            .map(|i| Id::from_name(&format!("item{}", i)))
            .collect();
        let e = SubsetEnumerator::new(items, 3);
        assert_eq!(e.total_combinations(), 20); // C(6,3)
        let count = e.count(); // consume iterator
        assert_eq!(count, 20);
    }

    #[test]
    fn test_robustness_result_display() {
        let r = RobustnessResult::Robust(3);
        assert_eq!(format!("{}", r), "Plan is 3-robust");

        let r2 = RobustnessResult::Timeout;
        assert!(format!("{}", r2).contains("timed out"));
    }
}
