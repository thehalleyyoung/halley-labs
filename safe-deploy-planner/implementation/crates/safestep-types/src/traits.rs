// Trait definitions for the SafeStep deployment planner.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::constraint::ConstraintSet;
use crate::error::Result;
use crate::graph::ClusterState;
use crate::plan::{DeploymentPlan, PlanCost};

// ─── Verifiable ─────────────────────────────────────────────────────────

/// Types that can be verified against a specification.
pub trait Verifiable {
    type Specification;
    type Evidence;

    fn verify(&self, spec: &Self::Specification) -> Result<Self::Evidence>;
    fn is_valid(&self, spec: &Self::Specification) -> bool {
        self.verify(spec).is_ok()
    }
}

/// Verification evidence for a deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanEvidence {
    pub all_states_safe: bool,
    pub monotone: bool,
    pub reaches_target: bool,
    pub envelope_contained: bool,
    pub pnr_count: usize,
    pub violation_count: usize,
    pub details: Vec<String>,
}

impl PlanEvidence {
    pub fn valid() -> Self {
        Self {
            all_states_safe: true,
            monotone: true,
            reaches_target: true,
            envelope_contained: true,
            pnr_count: 0,
            violation_count: 0,
            details: Vec::new(),
        }
    }

    pub fn is_fully_verified(&self) -> bool {
        self.all_states_safe
            && self.reaches_target
            && self.envelope_contained
            && self.violation_count == 0
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.details.push(detail.into());
        self
    }
}

impl fmt::Display for PlanEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_fully_verified() {
            write!(f, "VERIFIED")
        } else {
            write!(
                f,
                "NOT VERIFIED ({} violations, {} PNRs)",
                self.violation_count, self.pnr_count
            )
        }
    }
}

impl Verifiable for DeploymentPlan {
    type Specification = ConstraintSet;
    type Evidence = PlanEvidence;

    fn verify(&self, spec: &ConstraintSet) -> Result<PlanEvidence> {
        let mut evidence = PlanEvidence::valid();
        evidence.monotone = self.is_monotone();
        evidence.pnr_count = self.pnr_count();

        // Verify all intermediate states satisfy constraints
        let states = self.all_states();
        for state in &states {
            if !spec.is_safe(state) {
                evidence.all_states_safe = false;
                let violations = spec.violations(state);
                evidence.violation_count += violations.len();
                for v in &violations {
                    evidence.details.push(format!(
                        "State {} violates {}",
                        state, v.constraint_id
                    ));
                }
            }
        }

        // Check that final state matches target
        if let Some(last) = states.last() {
            if *last != self.target_state {
                evidence.reaches_target = false;
                evidence.details.push("Plan does not reach target state".into());
            }
        }

        Ok(evidence)
    }
}

// ─── Encodable ──────────────────────────────────────────────────────────

/// Types that can be encoded as SAT/SMT formulas.
pub trait Encodable {
    type Formula;
    type Variables;

    fn encode(&self, vars: &Self::Variables) -> Result<Self::Formula>;
    fn variable_count(&self) -> usize;
    fn clause_count_estimate(&self) -> usize;
}

/// Placeholder formula type for encoding results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNFFormula {
    pub num_variables: usize,
    pub clauses: Vec<Vec<i32>>,
}

impl CNFFormula {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            clauses: Vec::new(),
        }
    }

    pub fn add_clause(&mut self, clause: Vec<i32>) {
        self.clauses.push(clause);
    }

    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    pub fn total_literals(&self) -> usize {
        self.clauses.iter().map(|c| c.len()).sum()
    }

    pub fn merge(&mut self, other: CNFFormula) {
        self.num_variables = self.num_variables.max(other.num_variables);
        self.clauses.extend(other.clauses);
    }
}

impl fmt::Display for CNFFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CNF({} vars, {} clauses, {} literals)",
            self.num_variables,
            self.num_clauses(),
            self.total_literals(),
        )
    }
}

/// Variable allocation for encoding.
#[derive(Debug, Clone)]
pub struct VariableAllocator {
    next_var: usize,
}

impl VariableAllocator {
    pub fn new() -> Self {
        Self { next_var: 1 }
    }

    pub fn allocate(&mut self) -> usize {
        let v = self.next_var;
        self.next_var += 1;
        v
    }

    pub fn allocate_block(&mut self, count: usize) -> Vec<usize> {
        let start = self.next_var;
        self.next_var += count;
        (start..start + count).collect()
    }

    pub fn total_allocated(&self) -> usize {
        self.next_var - 1
    }
}

impl Default for VariableAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Hashable ───────────────────────────────────────────────────────────

/// Content-addressed hashing for domain types.
pub trait Hashable {
    fn content_hash(&self) -> String;

    fn content_hash_bytes(&self) -> Vec<u8> {
        hex::decode(self.content_hash()).unwrap_or_default()
    }
}

impl Hashable for ClusterState {
    fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        for v in self.as_slice() {
            hasher.update(v.0.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }
}

impl Hashable for DeploymentPlan {
    fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.start_state.content_hash().as_bytes());
        hasher.update(self.target_state.content_hash().as_bytes());
        for step in &self.steps {
            hasher.update(step.service_idx.to_le_bytes());
            hasher.update(step.from_version.0.to_le_bytes());
            hasher.update(step.to_version.0.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }
}

/// Helper for hashing serializable types.
pub fn hash_json<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_string(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    hex::encode(hasher.finalize())
}

// ─── Mergeable ──────────────────────────────────────────────────────────

/// Types that can be merged (combined) from partial results.
pub trait Mergeable {
    fn merge(&mut self, other: Self);
    fn is_complete(&self) -> bool;
}

/// A partial result container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialResult<T> {
    pub value: T,
    pub completeness: f64,
    pub source: String,
}

impl<T> PartialResult<T> {
    pub fn new(value: T, completeness: f64, source: impl Into<String>) -> Self {
        Self {
            value,
            completeness,
            source: source.into(),
        }
    }

    pub fn complete(value: T, source: impl Into<String>) -> Self {
        Self::new(value, 1.0, source)
    }

    pub fn is_complete(&self) -> bool {
        self.completeness >= 1.0
    }
}

// ─── CostMetric ─────────────────────────────────────────────────────────

/// Trait for computing plan costs.
pub trait CostMetric: Send + Sync {
    fn name(&self) -> &str;
    fn compute(&self, plan: &DeploymentPlan) -> f64;
    fn combine(&self, a: f64, b: f64) -> f64 {
        a + b
    }
    fn is_lower_better(&self) -> bool {
        true
    }
}

/// Step count metric.
pub struct StepCountMetric;

impl CostMetric for StepCountMetric {
    fn name(&self) -> &str {
        "step_count"
    }

    fn compute(&self, plan: &DeploymentPlan) -> f64 {
        plan.len() as f64
    }
}

/// Risk metric: sum of per-step risk scores.
pub struct RiskMetric;

impl CostMetric for RiskMetric {
    fn name(&self) -> &str {
        "risk"
    }

    fn compute(&self, plan: &DeploymentPlan) -> f64 {
        plan.steps
            .iter()
            .map(|s| s.annotation.risk_score.into_inner())
            .sum()
    }
}

/// PNR count metric.
pub struct PNRCountMetric;

impl CostMetric for PNRCountMetric {
    fn name(&self) -> &str {
        "pnr_count"
    }

    fn compute(&self, plan: &DeploymentPlan) -> f64 {
        plan.pnr_count() as f64
    }
}

/// Composite cost metric combining multiple metrics with weights.
pub struct CompositeCostMetric {
    metrics: Vec<(Box<dyn CostMetric>, f64)>,
}

impl CompositeCostMetric {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn add(mut self, metric: Box<dyn CostMetric>, weight: f64) -> Self {
        self.metrics.push((metric, weight));
        self
    }
}

impl Default for CompositeCostMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl CostMetric for CompositeCostMetric {
    fn name(&self) -> &str {
        "composite"
    }

    fn compute(&self, plan: &DeploymentPlan) -> f64 {
        self.metrics
            .iter()
            .map(|(m, w)| m.compute(plan) * w)
            .sum()
    }
}

// ─── ProgressReporter ───────────────────────────────────────────────────

/// Trait for reporting progress during long operations.
pub trait ProgressReporter: Send + Sync {
    fn report(&self, progress: ProgressUpdate);
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// A progress update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    pub phase: String,
    pub current: u64,
    pub total: Option<u64>,
    pub message: Option<String>,
    pub elapsed_ms: u64,
}

impl ProgressUpdate {
    pub fn new(phase: impl Into<String>, current: u64) -> Self {
        Self {
            phase: phase.into(),
            current,
            total: None,
            message: None,
            elapsed_ms: 0,
        }
    }

    pub fn with_total(mut self, total: u64) -> Self {
        self.total = Some(total);
        self
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    pub fn with_elapsed(mut self, ms: u64) -> Self {
        self.elapsed_ms = ms;
        self
    }

    pub fn progress_ratio(&self) -> Option<f64> {
        self.total.map(|t| {
            if t == 0 {
                1.0
            } else {
                self.current as f64 / t as f64
            }
        })
    }
}

impl fmt::Display for ProgressUpdate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.phase, self.current)?;
        if let Some(total) = self.total {
            write!(f, "/{}", total)?;
        }
        if let Some(msg) = &self.message {
            write!(f, " - {}", msg)?;
        }
        Ok(())
    }
}

/// A no-op progress reporter.
pub struct NoopReporter;

impl ProgressReporter for NoopReporter {
    fn report(&self, _progress: ProgressUpdate) {}
}

/// A progress reporter that collects updates.
pub struct CollectingReporter {
    updates: std::sync::Mutex<Vec<ProgressUpdate>>,
}

impl CollectingReporter {
    pub fn new() -> Self {
        Self {
            updates: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn updates(&self) -> Vec<ProgressUpdate> {
        self.updates.lock().unwrap().clone()
    }

    pub fn last_update(&self) -> Option<ProgressUpdate> {
        self.updates.lock().unwrap().last().cloned()
    }
}

impl Default for CollectingReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for CollectingReporter {
    fn report(&self, progress: ProgressUpdate) {
        self.updates.lock().unwrap().push(progress);
    }
}

// ─── Oracle ─────────────────────────────────────────────────────────────

/// Trait for compatibility oracle queries.
pub trait Oracle: Send + Sync {
    /// Query whether two service versions are compatible.
    fn is_compatible(
        &self,
        service_a: usize,
        version_a: u32,
        service_b: usize,
        version_b: u32,
    ) -> Result<bool>;

    /// Batch query for efficiency.
    fn batch_query(
        &self,
        queries: &[(usize, u32, usize, u32)],
    ) -> Result<Vec<bool>> {
        queries
            .iter()
            .map(|&(sa, va, sb, vb)| self.is_compatible(sa, va, sb, vb))
            .collect()
    }
}

/// A simple oracle backed by a set of compatible pairs.
pub struct SetOracle {
    compatible: hashbrown::HashSet<(usize, u32, usize, u32)>,
}

impl SetOracle {
    pub fn new() -> Self {
        Self {
            compatible: hashbrown::HashSet::new(),
        }
    }

    pub fn add_compatible(&mut self, sa: usize, va: u32, sb: usize, vb: u32) {
        self.compatible.insert((sa, va, sb, vb));
    }
}

impl Default for SetOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl Oracle for SetOracle {
    fn is_compatible(
        &self,
        service_a: usize,
        version_a: u32,
        service_b: usize,
        version_b: u32,
    ) -> Result<bool> {
        Ok(self
            .compatible
            .contains(&(service_a, version_a, service_b, version_b)))
    }
}

/// An oracle that always returns true (for testing).
pub struct TrueOracle;

impl Oracle for TrueOracle {
    fn is_compatible(
        &self,
        _service_a: usize,
        _version_a: u32,
        _service_b: usize,
        _version_b: u32,
    ) -> Result<bool> {
        Ok(true)
    }
}

// ─── PlanOptimizer ──────────────────────────────────────────────────────

/// Trait for plan optimization strategies.
pub trait PlanOptimizer: Send + Sync {
    fn name(&self) -> &str;
    fn optimize(&self, plan: &DeploymentPlan) -> Result<DeploymentPlan>;
    fn can_optimize(&self, plan: &DeploymentPlan) -> bool;
}

/// Result of an optimization attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub original_cost: PlanCost,
    pub optimized_cost: PlanCost,
    pub improvement: f64,
    pub optimizer_name: String,
    pub iterations: usize,
}

impl OptimizationResult {
    pub fn new(
        original: PlanCost,
        optimized: PlanCost,
        name: impl Into<String>,
        iterations: usize,
    ) -> Self {
        let orig_steps = original.total_steps as f64;
        let opt_steps = optimized.total_steps as f64;
        let improvement = if orig_steps > 0.0 {
            (orig_steps - opt_steps) / orig_steps
        } else {
            0.0
        };
        Self {
            original_cost: original,
            optimized_cost: optimized,
            improvement,
            optimizer_name: name.into(),
            iterations,
        }
    }

    pub fn is_improved(&self) -> bool {
        self.improvement > 0.0
    }
}

impl fmt::Display for OptimizationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.1}% improvement ({} -> {} steps, {} iterations)",
            self.optimizer_name,
            self.improvement * 100.0,
            self.original_cost.total_steps,
            self.optimized_cost.total_steps,
            self.iterations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ClusterState;
    use crate::plan::{PlanCost, PlanStep, StepAnnotation};
    use crate::version::VersionIndex;
    use ordered_float::OrderedFloat;

    fn make_plan() -> DeploymentPlan {
        let start = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        let target = ClusterState::new(&[VersionIndex(1), VersionIndex(1)]);
        let steps = vec![
            PlanStep::new(0, 0, "svc-a", VersionIndex(0), VersionIndex(1))
                .with_annotation(StepAnnotation::safe().with_risk(0.2)),
            PlanStep::new(1, 1, "svc-b", VersionIndex(0), VersionIndex(1))
                .with_annotation(StepAnnotation::safe().with_risk(0.3)),
        ];
        DeploymentPlan::new(start, target).with_steps(steps)
    }

    #[test]
    fn test_verifiable_plan() {
        let plan = make_plan();
        let cs = ConstraintSet::new();
        let evidence = plan.verify(&cs).unwrap();
        assert!(evidence.all_states_safe);
        assert!(evidence.reaches_target);
    }

    #[test]
    fn test_plan_evidence_display() {
        let e = PlanEvidence::valid();
        assert_eq!(e.to_string(), "VERIFIED");
        assert!(e.is_fully_verified());
    }

    #[test]
    fn test_cnf_formula() {
        let mut f = CNFFormula::new(10);
        f.add_clause(vec![1, -2, 3]);
        f.add_clause(vec![-1, 2]);
        assert_eq!(f.num_clauses(), 2);
        assert_eq!(f.total_literals(), 5);
    }

    #[test]
    fn test_cnf_merge() {
        let mut f1 = CNFFormula::new(5);
        f1.add_clause(vec![1, 2]);
        let mut f2 = CNFFormula::new(8);
        f2.add_clause(vec![3, 4]);
        f1.merge(f2);
        assert_eq!(f1.num_variables, 8);
        assert_eq!(f1.num_clauses(), 2);
    }

    #[test]
    fn test_variable_allocator() {
        let mut alloc = VariableAllocator::new();
        assert_eq!(alloc.allocate(), 1);
        assert_eq!(alloc.allocate(), 2);
        let block = alloc.allocate_block(3);
        assert_eq!(block, vec![3, 4, 5]);
        assert_eq!(alloc.total_allocated(), 5);
    }

    #[test]
    fn test_hashable_cluster_state() {
        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        let s2 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        let s3 = ClusterState::new(&[VersionIndex(1), VersionIndex(0)]);
        assert_eq!(s1.content_hash(), s2.content_hash());
        assert_ne!(s1.content_hash(), s3.content_hash());
    }

    #[test]
    fn test_hashable_plan() {
        let plan = make_plan();
        let hash = plan.content_hash();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_hash_json() {
        let v1 = serde_json::json!({"a": 1});
        let v2 = serde_json::json!({"a": 1});
        assert_eq!(hash_json(&v1), hash_json(&v2));
    }

    #[test]
    fn test_step_count_metric() {
        let plan = make_plan();
        let m = StepCountMetric;
        assert_eq!(m.compute(&plan), 2.0);
        assert_eq!(m.name(), "step_count");
        assert!(m.is_lower_better());
    }

    #[test]
    fn test_risk_metric() {
        let plan = make_plan();
        let m = RiskMetric;
        assert!((m.compute(&plan) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pnr_count_metric() {
        let plan = make_plan();
        let m = PNRCountMetric;
        assert_eq!(m.compute(&plan), 0.0);
    }

    #[test]
    fn test_composite_metric() {
        let plan = make_plan();
        let m = CompositeCostMetric::new()
            .add(Box::new(StepCountMetric), 1.0)
            .add(Box::new(RiskMetric), 10.0);
        let cost = m.compute(&plan);
        // 2*1.0 + 0.5*10.0 = 7.0
        assert!((cost - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_update() {
        let p = ProgressUpdate::new("solving", 50)
            .with_total(100)
            .with_message("half done")
            .with_elapsed(5000);
        assert_eq!(p.progress_ratio(), Some(0.5));
        let s = p.to_string();
        assert!(s.contains("50/100"));
    }

    #[test]
    fn test_progress_ratio_none() {
        let p = ProgressUpdate::new("init", 0);
        assert!(p.progress_ratio().is_none());
    }

    #[test]
    fn test_noop_reporter() {
        let r = NoopReporter;
        r.report(ProgressUpdate::new("test", 0));
        assert!(!r.is_cancelled());
    }

    #[test]
    fn test_collecting_reporter() {
        let r = CollectingReporter::new();
        r.report(ProgressUpdate::new("phase1", 1));
        r.report(ProgressUpdate::new("phase2", 2));
        let updates = r.updates();
        assert_eq!(updates.len(), 2);
        assert_eq!(r.last_update().unwrap().current, 2);
    }

    #[test]
    fn test_set_oracle() {
        let mut oracle = SetOracle::new();
        oracle.add_compatible(0, 0, 1, 0);
        oracle.add_compatible(0, 0, 1, 1);
        assert!(oracle.is_compatible(0, 0, 1, 0).unwrap());
        assert!(oracle.is_compatible(0, 0, 1, 1).unwrap());
        assert!(!oracle.is_compatible(0, 1, 1, 0).unwrap());
    }

    #[test]
    fn test_true_oracle() {
        let oracle = TrueOracle;
        assert!(oracle.is_compatible(0, 0, 1, 0).unwrap());
        assert!(oracle.is_compatible(99, 99, 99, 99).unwrap());
    }

    #[test]
    fn test_oracle_batch() {
        let oracle = TrueOracle;
        let results = oracle
            .batch_query(&[(0, 0, 1, 0), (0, 1, 1, 1)])
            .unwrap();
        assert_eq!(results, vec![true, true]);
    }

    #[test]
    fn test_optimization_result() {
        let orig = PlanCost {
            total_steps: 10,
            ..Default::default()
        };
        let opt = PlanCost {
            total_steps: 7,
            ..Default::default()
        };
        let result = OptimizationResult::new(orig, opt, "test-opt", 5);
        assert!(result.is_improved());
        assert!((result.improvement - 0.3).abs() < f64::EPSILON);
        let s = result.to_string();
        assert!(s.contains("30.0%"));
    }

    #[test]
    fn test_partial_result() {
        let pr = PartialResult::new(42, 0.5, "solver");
        assert!(!pr.is_complete());
        let pr2 = PartialResult::complete(42, "solver");
        assert!(pr2.is_complete());
    }
}
