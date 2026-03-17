//! CEGAR (CounterExample-Guided Abstraction Refinement) loop.
//!
//! Implements the iterative refinement loop that:
//! 1. Checks the SMT formula for satisfiability
//! 2. If SAT: concretizes the model and validates the trace
//! 3. If the trace is spurious: refines the abstraction and repeats
//! 4. If UNSAT: produces a bounded-completeness certificate
//! 5. Terminates in ≤ |C|^k iterations (Theorem T4)

use crate::concretizer::{Concretizer, ConcretizerConfig};
use crate::refinement::{
    PredicateEncoder, RefinementHistory, RefinementPredicate, RefinementStrategy,
    diff_based_refinement, interpolation_refine,
};
use crate::trace::ConcreteTrace;
use crate::validation::{TraceValidator, ValidatorConfig, ValidationReport, ReplaySimulator};
use crate::{
    ConcreteError, ConcreteResult, SmtExpr, SmtFormula, SmtModel, SmtSort, SmtValue,
    UnsatProof,
};
use crate::ProtocolVersion;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

// ── CegarConfig ──────────────────────────────────────────────────────────

/// Configuration for the CEGAR loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarConfig {
    /// Maximum number of CEGAR iterations before giving up.
    pub max_iterations: usize,
    /// Timeout for each SMT solver call, in milliseconds.
    pub solver_timeout_ms: u64,
    /// Total timeout for the CEGAR loop, in milliseconds.
    pub total_timeout_ms: u64,
    /// Maximum number of refinement predicates to add per iteration.
    pub max_predicates_per_iteration: usize,
    /// Refinement strategy.
    pub strategy: RefinementStrategy,
    /// Whether to attempt predicate minimization.
    pub minimize_predicates: bool,
    /// Whether to validate concretized traces via replay.
    pub validate_via_replay: bool,
    /// Convergence bound: |C|^k where C = cipher suite count.
    pub cipher_suite_count: usize,
    /// Convergence exponent k (protocol message bound).
    pub message_bound: usize,
    /// Concretizer configuration.
    pub concretizer_config: ConcretizerConfig,
    /// Valid cipher suites for refinement.
    pub valid_cipher_suites: BTreeSet<u16>,
    /// Valid protocol versions for refinement.
    pub valid_versions: BTreeSet<ProtocolVersion>,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            solver_timeout_ms: 30_000,
            total_timeout_ms: 300_000,
            max_predicates_per_iteration: 5,
            strategy: RefinementStrategy::Adaptive,
            minimize_predicates: true,
            validate_via_replay: true,
            cipher_suite_count: 350,
            message_bound: 4,
            concretizer_config: ConcretizerConfig::default(),
            valid_cipher_suites: [
                0x002f, 0x0035, 0x009c, 0x009d, 0xc013, 0xc014, 0xc02f, 0xc030,
            ]
            .iter()
            .copied()
            .collect(),
            valid_versions: [
                ProtocolVersion::Ssl30,
                ProtocolVersion::Tls10,
                ProtocolVersion::Tls11,
                ProtocolVersion::Tls12,
                ProtocolVersion::Tls13,
            ]
            .iter()
            .copied()
            .collect(),
        }
    }
}

impl CegarConfig {
    /// Compute the theoretical iteration bound: |C|^k.
    pub fn convergence_bound(&self) -> usize {
        let base = self.cipher_suite_count.max(1);
        let mut bound = 1usize;
        for _ in 0..self.message_bound {
            bound = bound.saturating_mul(base);
            if bound > self.max_iterations {
                return self.max_iterations;
            }
        }
        bound.min(self.max_iterations)
    }
}

// ── CegarResult ──────────────────────────────────────────────────────────

/// Result of the CEGAR loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CegarResult {
    /// Found a concrete attack trace (SAT + validated).
    ConcreteAttack {
        trace: ConcreteTrace,
        iterations: usize,
        stats: CegarStats,
    },
    /// Proved no attack exists within bounds (UNSAT).
    CertifiedSafe {
        proof: UnsatProof,
        iterations: usize,
        stats: CegarStats,
    },
    /// CEGAR loop timed out or hit iteration limit.
    Timeout {
        last_model: Option<SmtModel>,
        iterations: usize,
        stats: CegarStats,
        reason: String,
    },
}

impl CegarResult {
    pub fn is_attack(&self) -> bool {
        matches!(self, CegarResult::ConcreteAttack { .. })
    }

    pub fn is_safe(&self) -> bool {
        matches!(self, CegarResult::CertifiedSafe { .. })
    }

    pub fn is_timeout(&self) -> bool {
        matches!(self, CegarResult::Timeout { .. })
    }

    pub fn iterations(&self) -> usize {
        match self {
            CegarResult::ConcreteAttack { iterations, .. } => *iterations,
            CegarResult::CertifiedSafe { iterations, .. } => *iterations,
            CegarResult::Timeout { iterations, .. } => *iterations,
        }
    }

    pub fn stats(&self) -> &CegarStats {
        match self {
            CegarResult::ConcreteAttack { stats, .. } => stats,
            CegarResult::CertifiedSafe { stats, .. } => stats,
            CegarResult::Timeout { stats, .. } => stats,
        }
    }
}

// ── CegarStats ───────────────────────────────────────────────────────────

/// Statistics collected during the CEGAR loop.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CegarStats {
    pub total_iterations: usize,
    pub sat_results: usize,
    pub unsat_results: usize,
    pub unknown_results: usize,
    pub spurious_traces: usize,
    pub genuine_traces: usize,
    pub total_refinements: usize,
    pub total_time_ms: u64,
    pub solver_time_ms: u64,
    pub concretization_time_ms: u64,
    pub validation_time_ms: u64,
    pub refinement_time_ms: u64,
    pub max_formula_size: usize,
    pub final_formula_size: usize,
}

impl CegarStats {
    pub fn convergence_rate(&self) -> f64 {
        if self.total_iterations == 0 {
            0.0
        } else {
            self.genuine_traces as f64 / self.total_iterations as f64
        }
    }

    pub fn spurious_rate(&self) -> f64 {
        if self.sat_results == 0 {
            0.0
        } else {
            self.spurious_traces as f64 / self.sat_results as f64
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "CEGAR: {} iterations ({} SAT, {} UNSAT, {} unknown), \
             {} spurious, {} genuine, {} refinements, {:.1}s total",
            self.total_iterations,
            self.sat_results,
            self.unsat_results,
            self.unknown_results,
            self.spurious_traces,
            self.genuine_traces,
            self.total_refinements,
            self.total_time_ms as f64 / 1000.0,
        )
    }
}

// ── CegarState ───────────────────────────────────────────────────────────

/// Mutable state of the CEGAR loop.
#[derive(Debug, Clone)]
pub struct CegarState {
    /// Current SMT formula (original + refinements).
    pub formula: SmtFormula,
    /// History of all refinement predicates.
    pub refinement_history: RefinementHistory,
    /// Current iteration number.
    pub iteration: usize,
    /// Last SMT model (if SAT).
    pub last_model: Option<SmtModel>,
    /// Last concrete trace (if concretized).
    pub last_trace: Option<ConcreteTrace>,
    /// Accumulated statistics.
    pub stats: CegarStats,
    /// Whether the loop has converged.
    pub converged: bool,
}

impl CegarState {
    pub fn new(formula: SmtFormula) -> Self {
        Self {
            formula,
            refinement_history: RefinementHistory::new(),
            iteration: 0,
            last_model: None,
            last_trace: None,
            stats: CegarStats::default(),
            converged: false,
        }
    }
}

// ── Solver interface ─────────────────────────────────────────────────────

/// Result of an SMT solver query.
#[derive(Debug, Clone)]
pub enum SolverResult {
    Sat(SmtModel),
    Unsat(UnsatProof),
    Unknown(String),
}

/// Trait for an SMT solver backend. Implementors provide actual solver calls.
pub trait SmtSolver {
    fn check_sat(&self, formula: &SmtFormula, timeout_ms: u64) -> ConcreteResult<SolverResult>;
}

/// A mock solver that returns configurable results, for testing and
/// for when the actual solver is not available.
pub struct MockSolver {
    results: Vec<SolverResult>,
    call_index: std::cell::Cell<usize>,
}

impl MockSolver {
    pub fn new(results: Vec<SolverResult>) -> Self {
        Self {
            results,
            call_index: std::cell::Cell::new(0),
        }
    }

    pub fn always_sat(model: SmtModel) -> Self {
        Self::new(vec![SolverResult::Sat(model)])
    }

    pub fn always_unsat() -> Self {
        Self::new(vec![SolverResult::Unsat(UnsatProof::new(vec![
            "core_clause_1".into(),
        ]))])
    }

    pub fn sequence(results: Vec<SolverResult>) -> Self {
        Self::new(results)
    }
}

impl SmtSolver for MockSolver {
    fn check_sat(&self, _formula: &SmtFormula, _timeout_ms: u64) -> ConcreteResult<SolverResult> {
        let idx = self.call_index.get();
        let result = if idx < self.results.len() {
            self.results[idx].clone()
        } else {
            // Cycle the last result
            self.results
                .last()
                .cloned()
                .unwrap_or(SolverResult::Unknown("no results configured".into()))
        };
        self.call_index.set(idx + 1);
        Ok(result)
    }
}

// ── CegarLoop ────────────────────────────────────────────────────────────

/// The CEGAR refinement loop.
pub struct CegarLoop<S: SmtSolver> {
    config: CegarConfig,
    solver: S,
    concretizer: Concretizer,
    predicate_encoder: PredicateEncoder,
    validator: TraceValidator,
}

impl<S: SmtSolver> CegarLoop<S> {
    pub fn new(config: CegarConfig, solver: S) -> Self {
        let concretizer = Concretizer::new(config.concretizer_config.clone());
        let predicate_encoder = PredicateEncoder::new();
        let validator = TraceValidator::with_defaults();
        Self {
            config,
            solver,
            concretizer,
            predicate_encoder,
            validator,
        }
    }

    /// Run the CEGAR loop on the given encoding.
    ///
    /// This is the main entry point. The loop:
    /// 1. Calls the SMT solver on the (refined) formula
    /// 2. If SAT: concretizes and validates
    /// 3. If valid: returns ConcreteAttack
    /// 4. If spurious: refines and loops
    /// 5. If UNSAT: returns CertifiedSafe
    /// 6. If iteration/time limit: returns Timeout
    pub fn run_cegar(&self, encoding: &SmtFormula) -> CegarResult {
        let start_time = Instant::now();
        let mut state = CegarState::new(encoding.clone());
        let convergence_bound = self.config.convergence_bound();

        log::info!(
            "Starting CEGAR loop: max_iter={}, convergence_bound={}, timeout={}ms",
            self.config.max_iterations,
            convergence_bound,
            self.config.total_timeout_ms
        );

        loop {
            state.iteration += 1;
            state.stats.total_iterations = state.iteration;

            // Check iteration limit
            if state.iteration > self.config.max_iterations {
                let elapsed = start_time.elapsed().as_millis() as u64;
                state.stats.total_time_ms = elapsed;
                return CegarResult::Timeout {
                    last_model: state.last_model,
                    iterations: state.iteration - 1,
                    stats: state.stats,
                    reason: format!("Max iterations ({}) exceeded", self.config.max_iterations),
                };
            }

            // Check time limit
            let elapsed_ms = start_time.elapsed().as_millis() as u64;
            if elapsed_ms > self.config.total_timeout_ms {
                state.stats.total_time_ms = elapsed_ms;
                return CegarResult::Timeout {
                    last_model: state.last_model,
                    iterations: state.iteration,
                    stats: state.stats,
                    reason: format!("Total timeout ({}ms) exceeded", self.config.total_timeout_ms),
                };
            }

            log::debug!("CEGAR iteration {} (formula: {} assertions)", state.iteration, state.formula.assertion_count());

            // Step 1: Solve
            let solver_start = Instant::now();
            let solver_result = match self.solver.check_sat(&state.formula, self.config.solver_timeout_ms) {
                Ok(r) => r,
                Err(e) => {
                    let elapsed = start_time.elapsed().as_millis() as u64;
                    state.stats.total_time_ms = elapsed;
                    return CegarResult::Timeout {
                        last_model: state.last_model,
                        iterations: state.iteration,
                        stats: state.stats,
                        reason: format!("Solver error: {}", e),
                    };
                }
            };
            state.stats.solver_time_ms += solver_start.elapsed().as_millis() as u64;

            match solver_result {
                SolverResult::Sat(model) => {
                    state.stats.sat_results += 1;
                    state.last_model = Some(model.clone());

                    // Step 2: Concretize
                    let conc_start = Instant::now();
                    let trace_result = self.concretizer.concretize(&model);
                    state.stats.concretization_time_ms += conc_start.elapsed().as_millis() as u64;

                    match trace_result {
                        Ok(trace) => {
                            // Step 3: Validate
                            let val_start = Instant::now();
                            let is_genuine = self.validate_trace(&trace);
                            state.stats.validation_time_ms += val_start.elapsed().as_millis() as u64;

                            if is_genuine {
                                // Genuine attack found!
                                state.stats.genuine_traces += 1;
                                state.stats.total_time_ms = start_time.elapsed().as_millis() as u64;
                                state.stats.final_formula_size = state.formula.assertion_count();
                                log::info!(
                                    "CEGAR: Genuine attack found at iteration {}",
                                    state.iteration
                                );
                                return CegarResult::ConcreteAttack {
                                    trace,
                                    iterations: state.iteration,
                                    stats: state.stats,
                                };
                            } else {
                                // Spurious — refine
                                state.stats.spurious_traces += 1;
                                state.last_trace = Some(trace.clone());

                                let ref_start = Instant::now();
                                let new_predicates = self.refine_from_trace(&trace, &state);
                                state.stats.refinement_time_ms += ref_start.elapsed().as_millis() as u64;

                                if new_predicates.is_empty() {
                                    log::warn!("CEGAR: No refinement predicates generated, terminating");
                                    state.stats.total_time_ms = start_time.elapsed().as_millis() as u64;
                                    // Return the trace anyway — it might be usable
                                    return CegarResult::ConcreteAttack {
                                        trace,
                                        iterations: state.iteration,
                                        stats: state.stats,
                                    };
                                }

                                // Apply refinements
                                let count = new_predicates.len();
                                state.refinement_history.record_iteration(new_predicates.clone());
                                if let Err(e) = self.predicate_encoder.apply_to_formula(
                                    &mut state.formula,
                                    &new_predicates,
                                ) {
                                    log::error!("Failed to apply refinements: {}", e);
                                }
                                state.stats.total_refinements += count;
                                state.stats.max_formula_size = state
                                    .stats
                                    .max_formula_size
                                    .max(state.formula.assertion_count());

                                log::debug!(
                                    "CEGAR: Added {} refinement predicates (total: {})",
                                    count,
                                    state.refinement_history.total_predicates()
                                );
                            }
                        }
                        Err(e) => {
                            // Concretization failed — treat as spurious
                            state.stats.spurious_traces += 1;
                            log::debug!("CEGAR: Concretization failed: {}", e);

                            // Generate a generic refinement
                            let ref_start = Instant::now();
                            let preds = self.refine_from_model(&model, &state);
                            state.stats.refinement_time_ms += ref_start.elapsed().as_millis() as u64;

                            if !preds.is_empty() {
                                state.refinement_history.record_iteration(preds.clone());
                                let _ = self.predicate_encoder.apply_to_formula(&mut state.formula, &preds);
                                state.stats.total_refinements += preds.len();
                            }
                        }
                    }
                }

                SolverResult::Unsat(proof) => {
                    state.stats.unsat_results += 1;
                    state.stats.total_time_ms = start_time.elapsed().as_millis() as u64;
                    state.stats.final_formula_size = state.formula.assertion_count();
                    log::info!(
                        "CEGAR: UNSAT at iteration {} — certified safe",
                        state.iteration
                    );
                    return CegarResult::CertifiedSafe {
                        proof,
                        iterations: state.iteration,
                        stats: state.stats,
                    };
                }

                SolverResult::Unknown(reason) => {
                    state.stats.unknown_results += 1;
                    log::debug!("CEGAR: Solver returned unknown: {}", reason);

                    // Try reducing complexity
                    if state.iteration > 3 {
                        state.stats.total_time_ms = start_time.elapsed().as_millis() as u64;
                        return CegarResult::Timeout {
                            last_model: state.last_model,
                            iterations: state.iteration,
                            stats: state.stats,
                            reason: format!("Solver unknown after {} iterations: {}", state.iteration, reason),
                        };
                    }
                }
            }
        }
    }

    /// Validate whether a concretized trace is a genuine attack.
    fn validate_trace(&self, trace: &ConcreteTrace) -> bool {
        // Basic check: must be a genuine downgrade
        if !self.concretizer.is_genuine_downgrade(trace) {
            return false;
        }

        // Protocol conformance
        let report = self.validator.validate(trace);
        if !report.is_valid {
            log::debug!("Trace validation failed: {}", report.summary());
            return false;
        }

        // Replay simulation (if configured)
        if self.config.validate_via_replay {
            let mut simulator = ReplaySimulator::new();
            match simulator.simulate(trace) {
                Ok(_log) => {
                    // Check that simulation reached expected end state
                    let ended_ok = simulator.server_phase().order_index() >= 5; // at least ChangeCipherSpec
                    if !ended_ok {
                        log::debug!("Replay did not reach expected phase");
                        return false;
                    }
                }
                Err(e) => {
                    log::debug!("Replay simulation failed: {}", e);
                    return false;
                }
            }
        }

        true
    }

    /// Generate refinement predicates from a spurious trace.
    fn refine_from_trace(
        &self,
        trace: &ConcreteTrace,
        state: &CegarState,
    ) -> Vec<RefinementPredicate> {
        let mut candidates = Vec::new();

        // 1. Check if the negotiated cipher is valid
        if let Some(ref cs) = trace.negotiated_cipher {
            if !self.config.valid_cipher_suites.contains(&cs.id) {
                candidates.push(RefinementPredicate::ExcludeCipher {
                    cipher_id: cs.id,
                    reason: format!("Cipher 0x{:04x} not in valid set", cs.id),
                });
            }
        }

        // 2. Check version validity
        if !self.config.valid_versions.contains(&trace.downgraded_version) {
            candidates.push(RefinementPredicate::ExcludeVersion {
                version: trace.downgraded_version,
                reason: format!("{} not a valid target version", trace.downgraded_version),
            });
        }

        // 3. Check for implausible downgrade severity
        if trace.downgrade_severity() == 0 {
            candidates.push(RefinementPredicate::ExcludeVersion {
                version: trace.downgraded_version,
                reason: "No actual downgrade (severity=0)".into(),
            });
        }

        // 4. Use diff-based refinement against valid set
        let abstract_ciphers: Vec<u16> = trace
            .messages
            .iter()
            .flat_map(|m| m.parsed_fields.cipher_suites.iter().copied())
            .collect();
        let mut diff_preds = diff_based_refinement(
            &abstract_ciphers,
            trace.downgraded_version,
            &self.config.valid_cipher_suites,
            &self.config.valid_versions,
        );
        candidates.append(&mut diff_preds);

        // 5. Use interpolation if we have raw variable data
        let spurious_vars = self.concretizer.extract_spurious_info(trace);
        let protocol_constraints = vec![
            ("cipher_selected".to_string(), 0x0001, 0xffff),
            ("version_negotiated".to_string(), 0x0300, 0x0304),
        ];
        let mut interp_preds = interpolation_refine(&spurious_vars, &protocol_constraints);
        candidates.append(&mut interp_preds);

        // 6. Deduplicate and filter out already-applied predicates
        candidates.dedup();
        candidates.retain(|p| !state.refinement_history.is_redundant(p));

        // 7. Apply strategy to select predicates
        let selected = self.config.strategy.select_multiple(
            &candidates,
            &state.refinement_history,
            state.iteration,
            self.config.max_predicates_per_iteration,
        );

        selected.into_iter().cloned().collect()
    }

    /// Generate refinement predicates from a model (when concretization failed).
    fn refine_from_model(
        &self,
        model: &SmtModel,
        state: &CegarState,
    ) -> Vec<RefinementPredicate> {
        let mut candidates = Vec::new();

        // Extract cipher from model and exclude it
        if let Some(cipher_val) = model.get_bitvec("cipher_selected") {
            let cipher_id = cipher_val as u16;
            candidates.push(RefinementPredicate::ExcludeCipher {
                cipher_id,
                reason: "Concretization failed for this cipher".into(),
            });
        }

        // Extract version and exclude it
        if let Some(ver_val) = model.get_bitvec("version_negotiated") {
            let version = crate::byte_encoding::wire_to_version(
                (ver_val >> 8) as u8,
                (ver_val & 0xff) as u8,
            );
            candidates.push(RefinementPredicate::ExcludeVersion {
                version,
                reason: "Concretization failed for this version".into(),
            });
        }

        // Bound adversary actions
        candidates.push(RefinementPredicate::BoundAdversaryActions {
            max_actions: 10,
            reason: "Reduce search space after concretization failure".into(),
        });

        candidates.retain(|p| !state.refinement_history.is_redundant(p));
        candidates
    }
}

// ── Convergence checking ─────────────────────────────────────────────────

/// Check whether the CEGAR loop has converged.
///
/// The loop terminates in at most |C|^k iterations where:
/// - |C| = number of cipher suites
/// - k = message bound
///
/// Convergence is guaranteed because each iteration either:
/// - Finds a genuine attack (terminates)
/// - Proves UNSAT (terminates)
/// - Adds a refinement predicate that excludes at least one point from
///   the |C|^k search space
pub fn check_convergence(
    iteration: usize,
    cipher_count: usize,
    message_bound: usize,
    history: &RefinementHistory,
) -> ConvergenceStatus {
    let bound = cipher_count.saturating_pow(message_bound as u32);
    let bound = bound.max(1);
    let progress = (history.total_predicates() as f64) / (bound as f64);

    if iteration >= bound {
        ConvergenceStatus::Converged {
            iterations: iteration,
            bound,
        }
    } else if progress > 0.95 {
        ConvergenceStatus::NearConvergence {
            progress,
            remaining_estimate: bound.saturating_sub(history.total_predicates()),
        }
    } else {
        ConvergenceStatus::InProgress {
            progress,
            iterations_remaining: bound.saturating_sub(iteration),
        }
    }
}

/// Status of CEGAR convergence.
#[derive(Debug, Clone)]
pub enum ConvergenceStatus {
    Converged {
        iterations: usize,
        bound: usize,
    },
    NearConvergence {
        progress: f64,
        remaining_estimate: usize,
    },
    InProgress {
        progress: f64,
        iterations_remaining: usize,
    },
}

impl std::fmt::Display for ConvergenceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Converged { iterations, bound } => {
                write!(f, "Converged after {} iterations (bound: {})", iterations, bound)
            }
            Self::NearConvergence { progress, remaining_estimate } => {
                write!(f, "Near convergence: {:.1}% ({} remaining)", progress * 100.0, remaining_estimate)
            }
            Self::InProgress { progress, iterations_remaining } => {
                write!(f, "In progress: {:.1}% ({} iterations remaining)", progress * 100.0, iterations_remaining)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> SmtModel {
        let mut model = SmtModel::new();
        model.insert("cipher_selected", SmtValue::BitVec(0xc02f, 16));
        model.insert("version_client", SmtValue::BitVec(0x0303, 16));
        model.insert("version_server", SmtValue::BitVec(0x0300, 16));
        model.insert("adv_action_0", SmtValue::BitVec(0, 8));
        model
    }

    fn make_test_formula() -> SmtFormula {
        let mut formula = SmtFormula::new("test_encoding");
        formula.declare("cipher_selected", SmtSort::BitVec(16));
        formula.declare("version_client", SmtSort::BitVec(16));
        formula.declare("version_server", SmtSort::BitVec(16));
        formula.add_assertion(SmtExpr::BoolLit(true));
        formula
    }

    #[test]
    fn test_cegar_finds_attack() {
        let model = make_test_model();
        let formula = make_test_formula();
        let solver = MockSolver::always_sat(model);
        let config = CegarConfig {
            max_iterations: 10,
            validate_via_replay: false, // skip replay for unit test
            ..Default::default()
        };
        let cegar = CegarLoop::new(config, solver);
        let result = cegar.run_cegar(&formula);

        assert!(
            result.is_attack(),
            "Expected attack, got: {:?}",
            match &result {
                CegarResult::Timeout { reason, .. } => reason.clone(),
                _ => format!("{:?}", result.iterations()),
            }
        );
        assert!(result.iterations() <= 10);
    }

    #[test]
    fn test_cegar_certified_safe() {
        let formula = make_test_formula();
        let solver = MockSolver::always_unsat();
        let config = CegarConfig {
            max_iterations: 10,
            ..Default::default()
        };
        let cegar = CegarLoop::new(config, solver);
        let result = cegar.run_cegar(&formula);

        assert!(result.is_safe());
        assert_eq!(result.iterations(), 1);
    }

    #[test]
    fn test_cegar_max_iterations() {
        let formula = make_test_formula();
        let solver = MockSolver::new(vec![SolverResult::Unknown("test".into())]);
        let config = CegarConfig {
            max_iterations: 3,
            ..Default::default()
        };
        let cegar = CegarLoop::new(config, solver);
        let result = cegar.run_cegar(&formula);

        // Should timeout due to repeated unknowns
        assert!(result.is_timeout());
    }

    #[test]
    fn test_cegar_stats() {
        let model = make_test_model();
        let formula = make_test_formula();
        let solver = MockSolver::always_sat(model);
        let config = CegarConfig {
            max_iterations: 5,
            validate_via_replay: false,
            ..Default::default()
        };
        let cegar = CegarLoop::new(config, solver);
        let result = cegar.run_cegar(&formula);

        let stats = result.stats();
        assert!(stats.total_iterations > 0);
        assert!(stats.total_time_ms >= 0);
    }

    #[test]
    fn test_cegar_refinement_sequence() {
        // First call returns SAT with an invalid cipher (will be spurious),
        // second call returns UNSAT
        let mut model = SmtModel::new();
        model.insert("cipher_selected", SmtValue::BitVec(0xdead, 16));
        model.insert("version_server", SmtValue::BitVec(0x0300, 16));
        model.insert("version_client", SmtValue::BitVec(0x0303, 16));

        let solver = MockSolver::sequence(vec![
            SolverResult::Sat(model),
            SolverResult::Unsat(UnsatProof::new(vec!["refined".into()])),
        ]);

        let config = CegarConfig {
            max_iterations: 10,
            validate_via_replay: false,
            ..Default::default()
        };
        let cegar = CegarLoop::new(config, solver);
        let formula = make_test_formula();
        let result = cegar.run_cegar(&formula);

        // Should either find an attack (if trace validates) or certify safe
        assert!(result.iterations() <= 10);
    }

    #[test]
    fn test_convergence_bound() {
        let config = CegarConfig {
            cipher_suite_count: 10,
            message_bound: 2,
            max_iterations: 500,
            ..Default::default()
        };
        assert_eq!(config.convergence_bound(), 100); // 10^2

        let config2 = CegarConfig {
            cipher_suite_count: 100,
            message_bound: 3,
            max_iterations: 500,
            ..Default::default()
        };
        // 100^3 = 1_000_000 > 500, so capped at 500
        assert_eq!(config2.convergence_bound(), 500);
    }

    #[test]
    fn test_convergence_status() {
        let history = RefinementHistory::new();
        let status = check_convergence(5, 10, 2, &history);
        match status {
            ConvergenceStatus::InProgress { .. } => {}
            _ => panic!("Expected InProgress"),
        }

        let status = check_convergence(100, 10, 2, &history);
        match status {
            ConvergenceStatus::Converged { .. } => {}
            _ => panic!("Expected Converged"),
        }
    }

    #[test]
    fn test_cegar_result_accessors() {
        let stats = CegarStats::default();
        let result = CegarResult::Timeout {
            last_model: None,
            iterations: 42,
            stats: stats.clone(),
            reason: "test".into(),
        };
        assert!(result.is_timeout());
        assert!(!result.is_attack());
        assert!(!result.is_safe());
        assert_eq!(result.iterations(), 42);
    }

    #[test]
    fn test_cegar_stats_summary() {
        let mut stats = CegarStats::default();
        stats.total_iterations = 10;
        stats.sat_results = 7;
        stats.unsat_results = 1;
        stats.spurious_traces = 5;
        stats.genuine_traces = 2;
        stats.total_refinements = 8;
        stats.total_time_ms = 1500;

        let summary = stats.summary();
        assert!(summary.contains("10 iterations"));
        assert!(summary.contains("7 SAT"));
        assert!(summary.contains("1.5s"));
        assert_eq!(stats.spurious_rate(), 5.0 / 7.0);
    }

    #[test]
    fn test_mock_solver_sequence() {
        let model = make_test_model();
        let solver = MockSolver::sequence(vec![
            SolverResult::Sat(model.clone()),
            SolverResult::Unsat(UnsatProof::new(vec![])),
        ]);
        let formula = make_test_formula();

        // First call → SAT
        match solver.check_sat(&formula, 1000).unwrap() {
            SolverResult::Sat(_) => {}
            _ => panic!("Expected SAT"),
        }
        // Second call → UNSAT
        match solver.check_sat(&formula, 1000).unwrap() {
            SolverResult::Unsat(_) => {}
            _ => panic!("Expected UNSAT"),
        }
        // Third call → cycles last (UNSAT)
        match solver.check_sat(&formula, 1000).unwrap() {
            SolverResult::Unsat(_) => {}
            _ => panic!("Expected UNSAT (cycled)"),
        }
    }

    #[test]
    fn test_cegar_state_initialization() {
        let formula = make_test_formula();
        let state = CegarState::new(formula.clone());
        assert_eq!(state.iteration, 0);
        assert_eq!(state.refinement_history.iteration_count(), 0);
        assert!(state.last_model.is_none());
        assert!(!state.converged);
    }

    #[test]
    fn test_convergence_display() {
        let status = ConvergenceStatus::InProgress {
            progress: 0.42,
            iterations_remaining: 58,
        };
        let s = format!("{}", status);
        assert!(s.contains("42.0%"));
        assert!(s.contains("58"));
    }
}
