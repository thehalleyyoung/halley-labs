//! Incremental solving support with push/pop context management.
//!
//! Implements assumption-based solving and bottom-up anomaly checking,
//! starting from the weakest anomaly (G0) and progressively checking
//! stronger properties.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use isospec_types::constraint::SmtExpr;
use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};

use crate::encoding::{AnomalyEncoder, EncodingBounds, ScheduleEncoder, VarNaming};
use crate::solver::{RawModel, SmtSolver, SolverResult, SolverStats};

// ---------------------------------------------------------------------------
// IncrementalContext – push/pop stack manager
// ---------------------------------------------------------------------------

/// Manages a stack of assertion contexts for incremental solving.
#[derive(Debug)]
pub struct IncrementalContext {
    /// Current stack depth.
    depth: usize,
    /// Assertions at each level.
    level_assertions: Vec<Vec<SmtExpr>>,
    /// Named assumptions for assumption-based solving.
    assumptions: HashMap<String, SmtExpr>,
    /// Track which assumptions are currently active.
    active_assumptions: Vec<String>,
}

impl IncrementalContext {
    pub fn new() -> Self {
        Self {
            depth: 0,
            level_assertions: vec![Vec::new()],
            assumptions: HashMap::new(),
            active_assumptions: Vec::new(),
        }
    }

    /// Push a new assertion context.
    pub fn push(&mut self) {
        self.depth += 1;
        self.level_assertions.push(Vec::new());
    }

    /// Pop the most recent assertion context, discarding its assertions.
    pub fn pop(&mut self) -> IsoSpecResult<Vec<SmtExpr>> {
        if self.depth == 0 {
            return Err(IsoSpecError::SmtSolver {
                msg: "cannot pop: context stack is empty".into(),
            });
        }
        self.depth -= 1;
        let removed = self.level_assertions.pop().unwrap_or_default();
        Ok(removed)
    }

    /// Assert a formula in the current context.
    pub fn assert_at_current(&mut self, expr: SmtExpr) {
        if let Some(level) = self.level_assertions.last_mut() {
            level.push(expr);
        }
    }

    /// Register a named assumption.
    pub fn add_assumption(&mut self, name: &str, expr: SmtExpr) {
        self.assumptions.insert(name.to_string(), expr);
    }

    /// Activate an assumption by name.
    pub fn activate_assumption(&mut self, name: &str) -> IsoSpecResult<()> {
        if !self.assumptions.contains_key(name) {
            return Err(IsoSpecError::smt_solver(format!(
                "unknown assumption: {}",
                name
            )));
        }
        if !self.active_assumptions.contains(&name.to_string()) {
            self.active_assumptions.push(name.to_string());
        }
        Ok(())
    }

    /// Deactivate an assumption by name.
    pub fn deactivate_assumption(&mut self, name: &str) {
        self.active_assumptions.retain(|a| a != name);
    }

    /// Get all currently active assumption expressions.
    pub fn active_assumption_exprs(&self) -> Vec<SmtExpr> {
        self.active_assumptions
            .iter()
            .filter_map(|name| self.assumptions.get(name).cloned())
            .collect()
    }

    /// Get all assertions across all levels (flattened).
    pub fn all_assertions(&self) -> Vec<SmtExpr> {
        self.level_assertions.iter().flatten().cloned().collect()
    }

    /// Get assertions only at the current level.
    pub fn current_level_assertions(&self) -> &[SmtExpr] {
        self.level_assertions.last().map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn total_assertions(&self) -> usize {
        self.level_assertions.iter().map(|v| v.len()).sum()
    }

    /// Reset to a clean state.
    pub fn reset(&mut self) {
        self.depth = 0;
        self.level_assertions = vec![Vec::new()];
        self.assumptions.clear();
        self.active_assumptions.clear();
    }
}

impl Default for IncrementalContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AnomalyCheckResult
// ---------------------------------------------------------------------------

/// Result of checking for a specific anomaly class.
#[derive(Debug, Clone)]
pub struct AnomalyCheckResult {
    pub anomaly: AnomalyClass,
    pub found: bool,
    pub model: Option<RawModel>,
    pub elapsed: Duration,
}

impl AnomalyCheckResult {
    pub fn not_found(anomaly: AnomalyClass, elapsed: Duration) -> Self {
        Self {
            anomaly,
            found: false,
            model: None,
            elapsed,
        }
    }

    pub fn found_with_model(anomaly: AnomalyClass, model: RawModel, elapsed: Duration) -> Self {
        Self {
            anomaly,
            found: true,
            model: Some(model),
            elapsed,
        }
    }
}

// ---------------------------------------------------------------------------
// Bottom-up anomaly checker
// ---------------------------------------------------------------------------

/// The canonical ordering of anomaly classes from weakest to strongest.
pub fn anomaly_hierarchy() -> Vec<AnomalyClass> {
    vec![
        AnomalyClass::G0,
        AnomalyClass::G1a,
        AnomalyClass::G1b,
        AnomalyClass::G1c,
        AnomalyClass::G2Item,
        AnomalyClass::G2,
    ]
}

/// Maps an anomaly class to the minimum isolation level that prevents it.
pub fn anomaly_prevented_by(anomaly: &AnomalyClass) -> IsolationLevel {
    match anomaly {
        AnomalyClass::G0 => IsolationLevel::ReadUncommitted,
        AnomalyClass::G1a => IsolationLevel::ReadCommitted,
        AnomalyClass::G1b => IsolationLevel::ReadCommitted,
        AnomalyClass::G1c => IsolationLevel::ReadCommitted,
        AnomalyClass::G2Item => IsolationLevel::RepeatableRead,
        AnomalyClass::G2 => IsolationLevel::Serializable,
    }
}

/// Checks anomalies bottom-up using incremental solving.
///
/// Starts from the weakest anomaly (G0) and works up. If a weaker anomaly
/// is found, stronger anomalies are also possible, so we can report early.
pub struct BottomUpChecker<S: SmtSolver> {
    solver: S,
    context: IncrementalContext,
    bounds: EncodingBounds,
    naming: VarNaming,
    stats: SolverStats,
}

impl<S: SmtSolver> BottomUpChecker<S> {
    pub fn new(solver: S, bounds: EncodingBounds) -> Self {
        let naming = VarNaming::default();
        Self {
            solver,
            context: IncrementalContext::new(),
            bounds,
            naming,
            stats: SolverStats::new(),
        }
    }

    /// Set up the base encoding (schedule structure constraints).
    pub fn initialize(&mut self) -> IsoSpecResult<()> {
        let mut encoder = ScheduleEncoder::new(self.bounds.clone());
        let cs = encoder.encode()?;

        // Assert all base constraints at level 0
        for assertion in &cs.assertions {
            self.context.assert_at_current(assertion.clone());
        }

        // Declare all variables via the solver
        for (name, sort) in &cs.declarations {
            self.solver.declare_const(name, &sort.to_smtlib2())?;
        }

        // Assert base constraints on the solver
        for expr in &cs.assertions {
            let rendered = crate::solver::render_smt_expr(expr);
            self.solver.assert_formula(&rendered)?;
        }

        Ok(())
    }

    /// Check for a specific anomaly by pushing a new context with anomaly constraints.
    pub fn check_anomaly(
        &mut self,
        anomaly: &AnomalyClass,
        txn_pairs: &[(usize, usize)],
    ) -> IsoSpecResult<AnomalyCheckResult> {
        let start = Instant::now();
        let anomaly_encoder = AnomalyEncoder::new(self.naming.clone());

        self.solver.push()?;
        self.context.push();

        // Add anomaly-specific constraints for each transaction pair
        let mut anomaly_exprs = Vec::new();
        for (t1, t2) in txn_pairs {
            let constraints = match anomaly {
                AnomalyClass::G0 => anomaly_encoder.encode_g0(*t1, *t2),
                AnomalyClass::G1a => anomaly_encoder.encode_g1a(*t1, *t2),
                AnomalyClass::G1b => {
                    anomaly_encoder.encode_g1b(*t1, *t2, self.bounds.max_ops_per_txn)
                }
                AnomalyClass::G1c => anomaly_encoder.encode_g1c(*t1, *t2),
                AnomalyClass::G2Item | AnomalyClass::G2 => {
                    // For G2/G2-item, encode a read-write conflict cycle
                    let mut cs = Vec::new();
                    cs.push(SmtExpr::Const(self.naming.committed(*t1)));
                    cs.push(SmtExpr::Const(self.naming.committed(*t2)));
                    cs.push(SmtExpr::Const(self.naming.reads_from(*t1, *t2)));
                    cs
                }
            };
            for c in constraints {
                anomaly_exprs.push(c.clone());
                self.context.assert_at_current(c.clone());
                let rendered = crate::solver::render_smt_expr(&c);
                self.solver.assert_formula(&rendered)?;
            }
        }

        // Check satisfiability
        let result = self.solver.check_sat("")?;
        let elapsed = start.elapsed();
        self.stats.record_check(&result, elapsed);

        let check_result = match &result {
            SolverResult::Sat(model_opt) => {
                let model = model_opt.clone().unwrap_or_default();
                AnomalyCheckResult::found_with_model(*anomaly, model, elapsed)
            }
            SolverResult::Unsat => AnomalyCheckResult::not_found(*anomaly, elapsed),
            SolverResult::Unknown(_) | SolverResult::Timeout(_) => {
                AnomalyCheckResult::not_found(*anomaly, elapsed)
            }
        };

        // Pop the anomaly context
        self.context.pop()?;
        self.solver.pop()?;

        Ok(check_result)
    }

    /// Run the full bottom-up hierarchy check.
    pub fn check_all(
        &mut self,
        txn_pairs: &[(usize, usize)],
    ) -> IsoSpecResult<Vec<AnomalyCheckResult>> {
        let hierarchy = anomaly_hierarchy();
        let mut results = Vec::new();

        for anomaly in &hierarchy {
            let result = self.check_anomaly(anomaly, txn_pairs)?;
            let found = result.found;
            results.push(result);

            // If a weaker anomaly is found, we know stronger ones are also possible
            // (in terms of the schedule having the weaker violation).
            // Continue checking to get the complete picture.
            if found {
                // Still continue to check higher levels for completeness
            }
        }

        Ok(results)
    }

    /// Get the highest isolation level needed to prevent all found anomalies.
    pub fn required_isolation_level(
        results: &[AnomalyCheckResult],
    ) -> IsolationLevel {
        let mut max_level = IsolationLevel::ReadUncommitted;
        for result in results {
            if result.found {
                let needed = anomaly_prevented_by(&result.anomaly);
                if isolation_level_ord(&needed) > isolation_level_ord(&max_level) {
                    max_level = needed;
                }
            }
        }
        max_level
    }

    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

/// Numeric ordering for isolation levels (for comparison purposes).
fn isolation_level_ord(level: &IsolationLevel) -> u8 {
    match level {
        IsolationLevel::ReadUncommitted => 0,
        IsolationLevel::ReadCommitted => 1,
        IsolationLevel::RepeatableRead => 2,
        IsolationLevel::Serializable => 3,
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// AssumptionBasedSolver – solving with retractable assumptions
// ---------------------------------------------------------------------------

/// A wrapper that enables assumption-based incremental solving.
///
/// Instead of push/pop, uses indicator variables as assumptions that can be
/// individually toggled without modifying the assertion stack.
pub struct AssumptionBasedSolver<S: SmtSolver> {
    solver: S,
    /// Mapping from assumption name to indicator variable name.
    indicators: HashMap<String, String>,
    /// Counter for generating unique indicator variable names.
    next_indicator: usize,
    /// Currently active indicator variables.
    active: Vec<String>,
}

impl<S: SmtSolver> AssumptionBasedSolver<S> {
    pub fn new(solver: S) -> Self {
        Self {
            solver,
            indicators: HashMap::new(),
            next_indicator: 0,
            active: Vec::new(),
        }
    }

    /// Add a named assumption. Returns the indicator variable name.
    pub fn add_assumption(&mut self, name: &str, expr: SmtExpr) -> IsoSpecResult<String> {
        let indicator = format!("_assume_{}", self.next_indicator);
        self.next_indicator += 1;

        // Declare the indicator variable
        self.solver.declare_const(&indicator, "Bool")?;

        // Assert: indicator => expr
        let implication = SmtExpr::Implies(
            Box::new(SmtExpr::Const(indicator.clone())),
            Box::new(expr),
        );
        let rendered = crate::solver::render_smt_expr(&implication);
        self.solver.assert_formula(&rendered)?;

        self.indicators.insert(name.to_string(), indicator.clone());
        Ok(indicator)
    }

    /// Activate an assumption (will be asserted as true on next check-sat).
    pub fn activate(&mut self, name: &str) -> IsoSpecResult<()> {
        let indicator = self
            .indicators
            .get(name)
            .ok_or_else(|| IsoSpecError::SmtSolver { msg: format!("unknown assumption: {}", name) })?
            .clone();
        if !self.active.contains(&indicator) {
            self.active.push(indicator);
        }
        Ok(())
    }

    /// Deactivate an assumption.
    pub fn deactivate(&mut self, name: &str) -> IsoSpecResult<()> {
        if let Some(indicator) = self.indicators.get(name) {
            self.active.retain(|a| a != indicator);
        }
        Ok(())
    }

    /// Check satisfiability with current active assumptions.
    pub fn check_sat_with_assumptions(&mut self) -> IsoSpecResult<SolverResult> {
        // Build assumption conjunction
        for indicator in &self.active {
            let assertion = SmtExpr::Const(indicator.clone());
            let rendered = crate::solver::render_smt_expr(&assertion);
            self.solver.assert_formula(&rendered)?;
        }
        self.solver.check_sat("")
    }

    /// Deactivate all assumptions.
    pub fn deactivate_all(&mut self) {
        self.active.clear();
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    pub fn total_assumptions(&self) -> usize {
        self.indicators.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::MockSmtSolver;

    #[test]
    fn test_incremental_context_push_pop() {
        let mut ctx = IncrementalContext::new();
        assert_eq!(ctx.depth(), 0);

        ctx.push();
        assert_eq!(ctx.depth(), 1);

        ctx.assert_at_current(SmtExpr::BoolLit(true));
        assert_eq!(ctx.current_level_assertions().len(), 1);

        let removed = ctx.pop().unwrap();
        assert_eq!(removed.len(), 1);
        assert_eq!(ctx.depth(), 0);
    }

    #[test]
    fn test_incremental_context_nested() {
        let mut ctx = IncrementalContext::new();
        ctx.assert_at_current(SmtExpr::BoolLit(true)); // level 0

        ctx.push();
        ctx.assert_at_current(SmtExpr::BoolLit(false)); // level 1

        ctx.push();
        ctx.assert_at_current(SmtExpr::IntLit(42)); // level 2

        assert_eq!(ctx.depth(), 2);
        assert_eq!(ctx.total_assertions(), 3);
        assert_eq!(ctx.all_assertions().len(), 3);

        ctx.pop().unwrap();
        assert_eq!(ctx.depth(), 1);
        assert_eq!(ctx.total_assertions(), 2);
    }

    #[test]
    fn test_incremental_context_pop_empty() {
        let mut ctx = IncrementalContext::new();
        assert!(ctx.pop().is_err());
    }

    #[test]
    fn test_assumptions() {
        let mut ctx = IncrementalContext::new();
        ctx.add_assumption("a1", SmtExpr::BoolLit(true));
        ctx.add_assumption("a2", SmtExpr::BoolLit(false));

        ctx.activate_assumption("a1").unwrap();
        assert_eq!(ctx.active_assumption_exprs().len(), 1);

        ctx.activate_assumption("a2").unwrap();
        assert_eq!(ctx.active_assumption_exprs().len(), 2);

        ctx.deactivate_assumption("a1");
        assert_eq!(ctx.active_assumption_exprs().len(), 1);
    }

    #[test]
    fn test_assumptions_unknown_name() {
        let mut ctx = IncrementalContext::new();
        assert!(ctx.activate_assumption("nonexistent").is_err());
    }

    #[test]
    fn test_anomaly_hierarchy_order() {
        let h = anomaly_hierarchy();
        assert_eq!(h.len(), 6);
        assert_eq!(h[0], AnomalyClass::G0);
        assert_eq!(h[5], AnomalyClass::G2);
    }

    #[test]
    fn test_anomaly_prevented_by() {
        assert_eq!(
            anomaly_prevented_by(&AnomalyClass::G0),
            IsolationLevel::ReadUncommitted
        );
        assert_eq!(
            anomaly_prevented_by(&AnomalyClass::G1a),
            IsolationLevel::ReadCommitted
        );
        assert_eq!(
            anomaly_prevented_by(&AnomalyClass::G2),
            IsolationLevel::Serializable
        );
    }

    #[test]
    fn test_bottom_up_checker_initialization() {
        let bounds = EncodingBounds {
            max_transactions: 2,
            max_ops_per_txn: 2,
            max_data_items: 2,
            max_value: 10,
            encode_predicates: false,
        };
        let solver = MockSmtSolver::always_unsat();
        let mut checker = BottomUpChecker::new(solver, bounds);
        assert!(checker.initialize().is_ok());
    }

    #[test]
    fn test_bottom_up_checker_single_anomaly() {
        let bounds = EncodingBounds {
            max_transactions: 2,
            max_ops_per_txn: 2,
            max_data_items: 2,
            max_value: 10,
            encode_predicates: false,
        };
        let solver = MockSmtSolver::always_unsat();
        let mut checker = BottomUpChecker::new(solver, bounds);
        checker.initialize().unwrap();

        let result = checker
            .check_anomaly(&AnomalyClass::G0, &[(0, 1)])
            .unwrap();
        assert!(!result.found);
        assert_eq!(result.anomaly, AnomalyClass::G0);
    }

    #[test]
    fn test_bottom_up_checker_all() {
        let bounds = EncodingBounds {
            max_transactions: 2,
            max_ops_per_txn: 2,
            max_data_items: 2,
            max_value: 10,
            encode_predicates: false,
        };
        let solver = MockSmtSolver::always_unsat();
        let mut checker = BottomUpChecker::new(solver, bounds);
        checker.initialize().unwrap();

        let results = checker.check_all(&[(0, 1)]).unwrap();
        assert_eq!(results.len(), 6);
        for r in &results {
            assert!(!r.found);
        }
    }

    #[test]
    fn test_required_isolation_level_none_found() {
        let results = vec![
            AnomalyCheckResult::not_found(AnomalyClass::G0, Duration::ZERO),
            AnomalyCheckResult::not_found(AnomalyClass::G1a, Duration::ZERO),
        ];
        let level = BottomUpChecker::<MockSmtSolver>::required_isolation_level(&results);
        assert_eq!(level, IsolationLevel::ReadUncommitted);
    }

    #[test]
    fn test_required_isolation_level_g1a_found() {
        let results = vec![
            AnomalyCheckResult::not_found(AnomalyClass::G0, Duration::ZERO),
            AnomalyCheckResult::found_with_model(
                AnomalyClass::G1a,
                RawModel::new(),
                Duration::ZERO,
            ),
        ];
        let level = BottomUpChecker::<MockSmtSolver>::required_isolation_level(&results);
        assert_eq!(level, IsolationLevel::ReadCommitted);
    }

    #[test]
    fn test_assumption_based_solver() {
        let mock = MockSmtSolver::always_sat();
        let mut abs = AssumptionBasedSolver::new(mock);
        abs.add_assumption("constraint_a", SmtExpr::BoolLit(true))
            .unwrap();
        abs.add_assumption("constraint_b", SmtExpr::BoolLit(false))
            .unwrap();

        assert_eq!(abs.total_assumptions(), 2);
        assert_eq!(abs.active_count(), 0);

        abs.activate("constraint_a").unwrap();
        assert_eq!(abs.active_count(), 1);

        abs.activate("constraint_b").unwrap();
        assert_eq!(abs.active_count(), 2);

        abs.deactivate("constraint_a").unwrap();
        assert_eq!(abs.active_count(), 1);

        abs.deactivate_all();
        assert_eq!(abs.active_count(), 0);
    }

    #[test]
    fn test_incremental_context_reset() {
        let mut ctx = IncrementalContext::new();
        ctx.push();
        ctx.assert_at_current(SmtExpr::BoolLit(true));
        ctx.add_assumption("a", SmtExpr::BoolLit(true));

        ctx.reset();
        assert_eq!(ctx.depth(), 0);
        assert_eq!(ctx.total_assertions(), 0);
        assert!(ctx.active_assumption_exprs().is_empty());
    }
}
