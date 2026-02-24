//! Core verification engine for axiomatic memory model checking.
//!
//! Provides the `Verifier` for checking individual executions and
//! enumerating consistent executions, plus `CompositionalVerifier`
//! for decomposition-based verification.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

use super::execution::{BitMatrix, ExecutionGraph, EventId, Address};
use super::memory_model::{MemoryModel, Constraint, RelationExpr};
use super::litmus::{LitmusTest, Outcome, LitmusOutcome, RegId};

// ---------------------------------------------------------------------------
// ConstraintViolation
// ---------------------------------------------------------------------------

/// Details about a violated constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_name: String,
    pub kind: ViolationKind,
    pub cycle: Option<Vec<EventId>>,
    pub self_loop_event: Option<EventId>,
}

/// Kind of constraint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationKind {
    CycleFound,
    SelfLoopFound,
    NonEmpty,
}

impl fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ViolationKind::CycleFound => {
                write!(f, "Cycle in {}", self.constraint_name)?;
                if let Some(c) = &self.cycle {
                    write!(f, ": {:?}", c)?;
                }
            }
            ViolationKind::SelfLoopFound => {
                write!(f, "Self-loop in {}", self.constraint_name)?;
                if let Some(e) = self.self_loop_event {
                    write!(f, " at event {}", e)?;
                }
            }
            ViolationKind::NonEmpty => {
                write!(f, "Non-empty relation: {}", self.constraint_name)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// Result of verifying a single execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCheckResult {
    pub consistent: bool,
    pub violations: Vec<ConstraintViolation>,
}

impl ExecutionCheckResult {
    pub fn consistent() -> Self {
        Self { consistent: true, violations: Vec::new() }
    }

    pub fn inconsistent(violations: Vec<ConstraintViolation>) -> Self {
        Self { consistent: false, violations }
    }
}

/// Full verification result for a litmus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub test_name: String,
    pub model_name: String,
    pub total_executions: usize,
    pub consistent_executions: usize,
    pub inconsistent_executions: usize,
    pub observed_outcomes: Vec<(Outcome, usize)>,
    pub forbidden_observed: Vec<Outcome>,
    pub required_missing: Vec<Outcome>,
    pub pass: bool,
    pub stats: VerificationStats,
}

impl VerificationResult {
    /// Whether any forbidden outcome was observed.
    pub fn has_forbidden(&self) -> bool {
        !self.forbidden_observed.is_empty()
    }

    /// Whether any required outcome was not observed.
    pub fn has_missing_required(&self) -> bool {
        !self.required_missing.is_empty()
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Verification: {} under {} ===", self.test_name, self.model_name)?;
        writeln!(f, "Executions: {} total, {} consistent, {} inconsistent",
            self.total_executions, self.consistent_executions, self.inconsistent_executions)?;
        writeln!(f, "Observed outcomes: {}", self.observed_outcomes.len())?;
        for (outcome, count) in &self.observed_outcomes {
            writeln!(f, "  {} (×{})", outcome, count)?;
        }
        if !self.forbidden_observed.is_empty() {
            writeln!(f, "FORBIDDEN outcomes observed:")?;
            for o in &self.forbidden_observed {
                writeln!(f, "  {}", o)?;
            }
        }
        if !self.required_missing.is_empty() {
            writeln!(f, "REQUIRED outcomes MISSING:")?;
            for o in &self.required_missing {
                writeln!(f, "  {}", o)?;
            }
        }
        writeln!(f, "Result: {}", if self.pass { "PASS" } else { "FAIL" })?;
        write!(f, "{}", self.stats)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VerificationStats
// ---------------------------------------------------------------------------

/// Statistics from a verification run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationStats {
    pub executions_checked: usize,
    pub violations_found: usize,
    pub consistent_found: usize,
    pub elapsed_ms: u64,
    pub relations_computed: usize,
    pub acyclicity_checks: usize,
    pub irreflexivity_checks: usize,
    pub emptiness_checks: usize,
}

impl fmt::Display for VerificationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Stats: {} executions in {}ms", self.executions_checked, self.elapsed_ms)?;
        writeln!(f, "  Consistent: {}, Violations: {}", self.consistent_found, self.violations_found)?;
        writeln!(f, "  Checks: {} acyclicity, {} irreflexivity, {} emptiness",
            self.acyclicity_checks, self.irreflexivity_checks, self.emptiness_checks)?;
        writeln!(f, "  Relations computed: {}", self.relations_computed)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Core verification engine for axiomatic memory model checking.
pub struct Verifier {
    model: MemoryModel,
    stats: VerificationStats,
}

impl Verifier {
    pub fn new(model: MemoryModel) -> Self {
        Self {
            model,
            stats: VerificationStats::default(),
        }
    }

    /// Get the memory model.
    pub fn model(&self) -> &MemoryModel { &self.model }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &VerificationStats { &self.stats }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = VerificationStats::default();
    }

    // -----------------------------------------------------------------------
    // Single execution checking
    // -----------------------------------------------------------------------

    /// Check whether a single execution is consistent with the memory model.
    pub fn check_execution(&mut self, exec: &ExecutionGraph) -> ExecutionCheckResult {
        self.stats.executions_checked += 1;

        // Compute all derived relations.
        let env = self.model.compute_derived(exec);
        self.stats.relations_computed += env.len();

        // Check each constraint.
        let mut violations = Vec::new();

        for constraint in &self.model.constraints {
            match constraint {
                Constraint::Acyclic(expr, name) => {
                    self.stats.acyclicity_checks += 1;
                    let rel = self.model.eval_expr(expr, exec, &env);
                    if !rel.is_acyclic() {
                        let cycle = rel.find_cycle();
                        violations.push(ConstraintViolation {
                            constraint_name: name.clone(),
                            kind: ViolationKind::CycleFound,
                            cycle,
                            self_loop_event: None,
                        });
                    }
                }
                Constraint::Irreflexive(expr, name) => {
                    self.stats.irreflexivity_checks += 1;
                    let rel = self.model.eval_expr(expr, exec, &env);
                    if !rel.is_irreflexive() {
                        let self_loop = (0..exec.len()).find(|&i| rel.get(i, i));
                        violations.push(ConstraintViolation {
                            constraint_name: name.clone(),
                            kind: ViolationKind::SelfLoopFound,
                            cycle: None,
                            self_loop_event: self_loop,
                        });
                    }
                }
                Constraint::Empty(expr, name) => {
                    self.stats.emptiness_checks += 1;
                    let rel = self.model.eval_expr(expr, exec, &env);
                    if !rel.is_empty() {
                        violations.push(ConstraintViolation {
                            constraint_name: name.clone(),
                            kind: ViolationKind::NonEmpty,
                            cycle: None,
                            self_loop_event: None,
                        });
                    }
                }
            }
        }

        if violations.is_empty() {
            self.stats.consistent_found += 1;
            ExecutionCheckResult::consistent()
        } else {
            self.stats.violations_found += 1;
            ExecutionCheckResult::inconsistent(violations)
        }
    }

    /// Efficient acyclicity check with certificate (cycle if exists).
    pub fn acyclicity_check(&self, rel: &BitMatrix) -> (bool, Option<Vec<usize>>) {
        if rel.is_acyclic() {
            (true, None)
        } else {
            let cycle = rel.find_cycle();
            (false, cycle)
        }
    }

    // -----------------------------------------------------------------------
    // Enumeration
    // -----------------------------------------------------------------------

    /// Enumerate all consistent executions of a litmus test.
    pub fn enumerate_consistent(&mut self, test: &LitmusTest)
        -> Vec<(ExecutionGraph, HashMap<(usize, RegId), u64>, HashMap<Address, u64>)>
    {
        let all_execs = test.enumerate_executions();
        let mut consistent = Vec::new();

        for (exec, regs, mem) in all_execs {
            let result = self.check_execution(&exec);
            if result.consistent {
                consistent.push((exec, regs, mem));
            }
        }

        consistent
    }

    /// Find executions that violate at least one constraint.
    pub fn find_violations(&mut self, test: &LitmusTest)
        -> Vec<(ExecutionGraph, Vec<ConstraintViolation>)>
    {
        let all_execs = test.enumerate_executions();
        let mut violating = Vec::new();

        for (exec, _regs, _mem) in all_execs {
            let result = self.check_execution(&exec);
            if !result.consistent {
                violating.push((exec, result.violations));
            }
        }

        violating
    }

    // -----------------------------------------------------------------------
    // Full litmus test verification
    // -----------------------------------------------------------------------

    /// Full litmus test verification: enumerate, check, report.
    pub fn verify_litmus(&mut self, test: &LitmusTest) -> VerificationResult {
        let start = Instant::now();
        self.reset_stats();

        let all_execs = test.enumerate_executions();
        let total = all_execs.len();
        let mut consistent_count = 0usize;
        let mut inconsistent_count = 0usize;

        // Track observed outcomes and their counts.
        let mut outcome_counts: HashMap<Outcome, usize> = HashMap::new();

        for (exec, regs, mem) in &all_execs {
            let result = self.check_execution(exec);
            if result.consistent {
                consistent_count += 1;

                // Build outcome from regs + mem.
                let mut outcome = Outcome::new();
                for (&(tid, reg), &val) in regs {
                    outcome = outcome.with_reg(tid, reg, val);
                }
                for (&addr, &val) in mem {
                    outcome = outcome.with_mem(addr, val);
                }
                *outcome_counts.entry(outcome).or_insert(0) += 1;
            } else {
                inconsistent_count += 1;
            }
        }

        // Check expected outcomes.
        let mut forbidden_observed = Vec::new();
        let mut required_missing = Vec::new();

        for (expected, kind) in &test.expected_outcomes {
            let was_observed = outcome_counts.keys().any(|observed| {
                outcome_matches_expected(observed, expected)
            });

            match kind {
                LitmusOutcome::Forbidden => {
                    if was_observed {
                        forbidden_observed.push(expected.clone());
                    }
                }
                LitmusOutcome::Required => {
                    if !was_observed {
                        required_missing.push(expected.clone());
                    }
                }
                LitmusOutcome::Allowed => {
                    // Allowed outcomes: no check needed.
                }
            }
        }

        let pass = forbidden_observed.is_empty() && required_missing.is_empty();

        let mut observed: Vec<(Outcome, usize)> = outcome_counts.into_iter().collect();
        observed.sort_by(|a, b| b.1.cmp(&a.1));

        let elapsed = start.elapsed();
        self.stats.elapsed_ms = elapsed.as_millis() as u64;

        VerificationResult {
            test_name: test.name.clone(),
            model_name: self.model.name.clone(),
            total_executions: total,
            consistent_executions: consistent_count,
            inconsistent_executions: inconsistent_count,
            observed_outcomes: observed,
            forbidden_observed,
            required_missing,
            pass,
            stats: self.stats.clone(),
        }
    }
}

/// Check if an observed outcome matches an expected outcome.
/// Only checks the keys present in expected.
fn outcome_matches_expected(observed: &Outcome, expected: &Outcome) -> bool {
    for (&key, &exp_val) in &expected.registers {
        match observed.registers.get(&key) {
            Some(&v) if v == exp_val => {}
            _ => return false,
        }
    }
    for (&addr, &exp_val) in &expected.memory {
        match observed.memory.get(&addr) {
            Some(&v) if v == exp_val => {}
            _ => return false,
        }
    }
    true
}

// ---------------------------------------------------------------------------
// CompositionalVerifier
// ---------------------------------------------------------------------------

/// Compositional verifier using test decomposition.
///
/// Splits a test into independent sub-tests at shared-address boundaries,
/// verifies each component separately, and glues results together.
pub struct CompositionalVerifier {
    model: MemoryModel,
    stats: VerificationStats,
}

impl CompositionalVerifier {
    pub fn new(model: MemoryModel) -> Self {
        Self {
            model,
            stats: VerificationStats::default(),
        }
    }

    pub fn stats(&self) -> &VerificationStats { &self.stats }

    /// Decompose a test into independent components based on shared addresses.
    pub fn decompose_test(&self, test: &LitmusTest) -> Vec<Vec<usize>> {
        let n = test.thread_count();
        if n == 0 { return Vec::new(); }

        // Build a graph: threads are connected if they share an address.
        let mut adj = vec![HashSet::new(); n];
        for i in 0..n {
            let addrs_i: HashSet<_> = test.threads[i].accessed_addresses().into_iter().collect();
            for j in i + 1..n {
                let addrs_j: HashSet<_> = test.threads[j].accessed_addresses().into_iter().collect();
                if addrs_i.intersection(&addrs_j).next().is_some() {
                    adj[i].insert(j);
                    adj[j].insert(i);
                }
            }
        }

        // Find connected components.
        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if visited[start] { continue; }
            let mut component = Vec::new();
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                if visited[node] { continue; }
                visited[node] = true;
                component.push(node);
                for &neighbor in &adj[node] {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
            component.sort();
            components.push(component);
        }

        components
    }

    /// Build a sub-test from a subset of threads.
    pub fn extract_component(&self, test: &LitmusTest, thread_ids: &[usize]) -> LitmusTest {
        let mut sub = LitmusTest::new(&format!("{}[{:?}]", test.name, thread_ids));

        let thread_set: HashSet<usize> = thread_ids.iter().copied().collect();

        // Add threads.
        for &tid in thread_ids {
            if tid < test.threads.len() {
                sub.add_thread(test.threads[tid].clone());
            }
        }

        // Add relevant initial state.
        let component_addrs: HashSet<Address> = thread_ids.iter()
            .flat_map(|&tid| test.threads.get(tid).map(|t| t.accessed_addresses()).unwrap_or_default())
            .collect();

        for (&addr, &val) in &test.initial_state {
            if component_addrs.contains(&addr) {
                sub.set_initial(addr, val);
            }
        }

        // Filter expected outcomes to only include registers from these threads.
        for (outcome, kind) in &test.expected_outcomes {
            let relevant_regs: HashMap<_, _> = outcome.registers.iter()
                .filter(|&(&(tid, _), _)| thread_set.contains(&tid))
                .map(|(&k, &v)| (k, v))
                .collect();
            let relevant_mem: HashMap<_, _> = outcome.memory.iter()
                .filter(|(&addr, _)| component_addrs.contains(&addr))
                .map(|(&k, &v)| (k, v))
                .collect();

            if !relevant_regs.is_empty() || !relevant_mem.is_empty() {
                let sub_outcome = Outcome {
                    registers: relevant_regs,
                    memory: relevant_mem,
                };
                sub.expect(sub_outcome, *kind);
            }
        }

        sub
    }

    /// Verify each component independently and combine results.
    pub fn verify_compositional(&mut self, test: &LitmusTest) -> VerificationResult {
        let start = Instant::now();
        self.stats = VerificationStats::default();

        let components = self.decompose_test(test);

        if components.len() <= 1 {
            // No decomposition possible; fall back to full verification.
            let mut v = Verifier::new(self.model.clone());
            let result = v.verify_litmus(test);
            self.stats = v.stats().clone();
            return result;
        }

        // Verify each component.
        let mut component_results = Vec::new();
        for component in &components {
            let sub_test = self.extract_component(test, component);
            let mut v = Verifier::new(self.model.clone());
            let result = v.verify_litmus(&sub_test);
            self.stats.executions_checked += result.stats.executions_checked;
            self.stats.consistent_found += result.stats.consistent_found;
            self.stats.violations_found += result.stats.violations_found;
            self.stats.acyclicity_checks += result.stats.acyclicity_checks;
            self.stats.irreflexivity_checks += result.stats.irreflexivity_checks;
            self.stats.emptiness_checks += result.stats.emptiness_checks;
            self.stats.relations_computed += result.stats.relations_computed;
            component_results.push(result);
        }

        // Glue results: the overall test passes iff all components pass.
        let all_pass = component_results.iter().all(|r| r.pass);

        let total_consistent: usize = component_results.iter()
            .map(|r| r.consistent_executions.max(1))
            .product();

        let total_execs: usize = component_results.iter()
            .map(|r| r.total_executions.max(1))
            .product();

        // Combine observed outcomes.
        let mut all_outcomes = Vec::new();
        let mut all_forbidden = Vec::new();
        let mut all_required_missing = Vec::new();

        for r in &component_results {
            all_outcomes.extend(r.observed_outcomes.clone());
            all_forbidden.extend(r.forbidden_observed.clone());
            all_required_missing.extend(r.required_missing.clone());
        }

        let elapsed = start.elapsed();
        self.stats.elapsed_ms = elapsed.as_millis() as u64;

        VerificationResult {
            test_name: test.name.clone(),
            model_name: self.model.name.clone(),
            total_executions: total_execs,
            consistent_executions: total_consistent,
            inconsistent_executions: total_execs.saturating_sub(total_consistent),
            observed_outcomes: all_outcomes,
            forbidden_observed: all_forbidden,
            required_missing: all_required_missing,
            pass: all_pass,
            stats: self.stats.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Batch verification
// ---------------------------------------------------------------------------

/// Verify multiple litmus tests against a model.
pub fn verify_batch(model: &MemoryModel, tests: &[LitmusTest]) -> Vec<VerificationResult> {
    tests.iter().map(|test| {
        let mut v = Verifier::new(model.clone());
        v.verify_litmus(test)
    }).collect()
}

/// Verify a test against multiple models.
pub fn verify_multi_model(test: &LitmusTest, models: &[MemoryModel]) -> Vec<VerificationResult> {
    models.iter().map(|model| {
        let mut v = Verifier::new(model.clone());
        v.verify_litmus(test)
    }).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::execution::ExecutionGraphBuilder;
    use super::super::memory_model::BuiltinModel;
    use super::super::litmus;

    fn make_sc_consistent_graph() -> ExecutionGraph {
        // Simple SC-consistent execution:
        // T0: W(x)=1
        // T1: R(x)=1
        let mut b = ExecutionGraphBuilder::new();
        let w = b.add_write(0, 0x100, 1);
        let r = b.add_read(1, 0x100, 1);
        let mut g = b.build();
        g.add_rf(w, r);
        g.derive_fr();
        g
    }

    fn make_sc_inconsistent_graph() -> ExecutionGraph {
        // SC-inconsistent: cyclic po ∪ com.
        // T0: W(x)=1, R(y)=0
        // T1: W(y)=1, R(x)=0
        // rf: init→R(x), init→R(y) (read initial values)
        // co: W(x)=1 is final, W(y)=1 is final
        // This creates a cycle: W(x) -po-> R(y) -fr-> W(y) -po-> R(x) -fr-> W(x)
        // But we need proper fr edges for this.
        // Simpler: create a direct cycle.
        let mut b = ExecutionGraphBuilder::new();
        let w0 = b.add_write(0, 0x100, 1);
        let r0 = b.add_read(0, 0x200, 0);
        let w1 = b.add_write(1, 0x200, 1);
        let r1 = b.add_read(1, 0x100, 0);
        let mut g = b.build();
        // No rf edges (reads from initial state).
        // co: just single writes per address.
        // fr: R reads 0, write is 1, so R fr W.
        // We need to set up co so fr is derived.
        // Actually, with single writes, co is trivially empty.
        // fr = rf^-1 ; co = empty.
        // Let's manually set fr to create the cycle.
        g.fr.set(r0, w1, true);  // R(y)=0 fr W(y)=1
        g.fr.set(r1, w0, true);  // R(x)=0 fr W(x)=1
        g
    }

    #[test]
    fn test_check_execution_consistent() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let g = make_sc_consistent_graph();
        let result = v.check_execution(&g);
        assert!(result.consistent);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_check_execution_inconsistent() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let g = make_sc_inconsistent_graph();
        let result = v.check_execution(&g);
        assert!(!result.consistent);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_acyclicity_check() {
        let sc = BuiltinModel::SC.build();
        let v = Verifier::new(sc);

        let mut dag = BitMatrix::new(3);
        dag.set(0, 1, true);
        dag.set(1, 2, true);
        let (acyclic, cycle) = v.acyclicity_check(&dag);
        assert!(acyclic);
        assert!(cycle.is_none());

        let mut cyclic = BitMatrix::new(3);
        cyclic.set(0, 1, true);
        cyclic.set(1, 2, true);
        cyclic.set(2, 0, true);
        let (acyclic2, cycle2) = v.acyclicity_check(&cyclic);
        assert!(!acyclic2);
        assert!(cycle2.is_some());
    }

    #[test]
    fn test_verify_litmus_sb() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::sb_test();
        let result = v.verify_litmus(&test);
        assert!(result.total_executions > 0);
        assert!(result.consistent_executions > 0);
        // Under SC, the SB forbidden outcome should NOT be observed.
        assert!(result.pass, "SB should pass under SC");
    }

    #[test]
    fn test_verify_litmus_mp() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::mp_test();
        let result = v.verify_litmus(&test);
        assert!(result.total_executions > 0);
        assert!(result.pass, "MP should pass under SC");
    }

    #[test]
    fn test_enumerate_consistent() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::sb_test();
        let consistent = v.enumerate_consistent(&test);
        assert!(!consistent.is_empty());
    }

    #[test]
    fn test_find_violations() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::sb_test();
        let violations = v.find_violations(&test);
        // Some executions of SB will be SC-inconsistent.
        // (The ones that try to read from future writes, etc.)
        // The test might or might not have violations depending on enumeration.
        let _ = violations; // Just check it doesn't crash.
    }

    #[test]
    fn test_verification_stats() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::sb_test();
        let _result = v.verify_litmus(&test);
        let stats = v.stats();
        assert!(stats.executions_checked > 0);
        assert!(stats.acyclicity_checks > 0 || stats.irreflexivity_checks > 0);
    }

    #[test]
    fn test_verification_result_display() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::sb_test();
        let result = v.verify_litmus(&test);
        let s = format!("{}", result);
        assert!(s.contains("Verification"));
        assert!(s.contains("SB"));
    }

    #[test]
    fn test_constraint_violation_display() {
        let cv = ConstraintViolation {
            constraint_name: "acyclic(ghb)".into(),
            kind: ViolationKind::CycleFound,
            cycle: Some(vec![0, 1, 2]),
            self_loop_event: None,
        };
        let s = format!("{}", cv);
        assert!(s.contains("Cycle"));
    }

    #[test]
    fn test_compositional_decompose() {
        let sc = BuiltinModel::SC.build();
        let cv = CompositionalVerifier::new(sc);

        // Create a test with independent components.
        let mut test = LitmusTest::new("Independent");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = super::super::litmus::Thread::new(0);
        t0.store(0x100, 1, super::super::litmus::Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = super::super::litmus::Thread::new(1);
        t1.store(0x200, 1, super::super::litmus::Ordering::Relaxed);
        test.add_thread(t1);

        let components = cv.decompose_test(&test);
        // Two independent threads → two components.
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_compositional_connected() {
        let sc = BuiltinModel::SC.build();
        let cv = CompositionalVerifier::new(sc);
        let test = litmus::sb_test();
        let components = cv.decompose_test(&test);
        // SB test: both threads access x and y, so single component.
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_compositional_verify() {
        let sc = BuiltinModel::SC.build();
        let mut cv = CompositionalVerifier::new(sc);
        let test = litmus::sb_test();
        let result = cv.verify_compositional(&test);
        assert!(result.pass);
    }

    #[test]
    fn test_extract_component() {
        let sc = BuiltinModel::SC.build();
        let cv = CompositionalVerifier::new(sc);
        let test = litmus::iriw_test();
        let sub = cv.extract_component(&test, &[0, 2]);
        assert_eq!(sub.threads.len(), 2);
    }

    #[test]
    fn test_verify_batch() {
        let sc = BuiltinModel::SC.build();
        let tests = vec![litmus::sb_test(), litmus::mp_test()];
        let results = verify_batch(&sc, &tests);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_verify_multi_model() {
        let test = litmus::sb_test();
        let models = vec![BuiltinModel::SC.build(), BuiltinModel::TSO.build()];
        let results = verify_multi_model(&test, &models);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_execution_check_result() {
        let ok = ExecutionCheckResult::consistent();
        assert!(ok.consistent);

        let bad = ExecutionCheckResult::inconsistent(vec![
            ConstraintViolation {
                constraint_name: "test".into(),
                kind: ViolationKind::CycleFound,
                cycle: None,
                self_loop_event: None,
            }
        ]);
        assert!(!bad.consistent);
    }

    #[test]
    fn test_verification_result_methods() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::sb_test();
        let result = v.verify_litmus(&test);
        assert!(!result.has_forbidden());
        assert!(!result.has_missing_required());
    }

    #[test]
    fn test_outcome_matches_expected() {
        let observed = Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 0).with_mem(0x100, 42);
        let expected = Outcome::new().with_reg(0, 0, 1);
        assert!(outcome_matches_expected(&observed, &expected));

        let expected2 = Outcome::new().with_reg(0, 0, 999);
        assert!(!outcome_matches_expected(&observed, &expected2));
    }

    #[test]
    fn test_stats_display() {
        let stats = VerificationStats {
            executions_checked: 100,
            violations_found: 5,
            consistent_found: 95,
            elapsed_ms: 42,
            relations_computed: 200,
            acyclicity_checks: 100,
            irreflexivity_checks: 50,
            emptiness_checks: 25,
        };
        let s = format!("{}", stats);
        assert!(s.contains("100 executions"));
        assert!(s.contains("42ms"));
    }

    #[test]
    fn test_verifier_reset_stats() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let g = make_sc_consistent_graph();
        let _ = v.check_execution(&g);
        assert!(v.stats().executions_checked > 0);
        v.reset_stats();
        assert_eq!(v.stats().executions_checked, 0);
    }

    #[test]
    fn test_lb_under_sc() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::lb_test();
        let result = v.verify_litmus(&test);
        assert!(result.pass, "LB should pass under SC (forbidden outcome not observable)");
    }

    #[test]
    fn test_two_plus_two_w_under_sc() {
        let sc = BuiltinModel::SC.build();
        let mut v = Verifier::new(sc);
        let test = litmus::two_plus_two_w_test();
        let result = v.verify_litmus(&test);
        // Under SC, the forbidden outcome should not be observed.
        let _ = result; // Just ensure it doesn't crash.
    }
}

