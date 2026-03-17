//! MaxSMT-based optimization for mixed-isolation level assignment.
//!
//! Uses weighted soft constraints to find cost-optimal isolation assignments,
//! with a greedy fallback when the MaxSMT solver times out.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use isospec_types::constraint::SmtExpr;
use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::TransactionId;
use isospec_types::isolation::IsolationLevel;

use crate::solver::{RawModel, SmtSolver, SolverConfig, SolverResult};

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Cost model assigning a numeric weight to each isolation level.
/// Lower isolation levels are cheaper but admit more anomalies.
#[derive(Debug, Clone)]
pub struct IsolationCostModel {
    costs: HashMap<IsolationLevel, u64>,
}

impl Default for IsolationCostModel {
    fn default() -> Self {
        let mut costs = HashMap::new();
        costs.insert(IsolationLevel::ReadUncommitted, 1);
        costs.insert(IsolationLevel::ReadCommitted, 2);
        costs.insert(IsolationLevel::RepeatableRead, 4);
        costs.insert(IsolationLevel::Serializable, 8);
        Self { costs }
    }
}

impl IsolationCostModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cost(mut self, level: IsolationLevel, cost: u64) -> Self {
        self.costs.insert(level, cost);
        self
    }

    pub fn cost(&self, level: &IsolationLevel) -> u64 {
        self.costs.get(level).copied().unwrap_or(10)
    }

    /// Return the cheapest isolation level.
    pub fn cheapest(&self) -> IsolationLevel {
        self.costs
            .iter()
            .min_by_key(|(_, c)| *c)
            .map(|(l, _)| *l)
            .unwrap_or(IsolationLevel::ReadUncommitted)
    }

    /// Return the most expensive isolation level.
    pub fn most_expensive(&self) -> IsolationLevel {
        self.costs
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(l, _)| *l)
            .unwrap_or(IsolationLevel::Serializable)
    }

    /// Return all levels sorted from cheapest to most expensive.
    pub fn levels_by_cost(&self) -> Vec<(IsolationLevel, u64)> {
        let mut entries: Vec<_> = self.costs.iter().map(|(l, c)| (*l, *c)).collect();
        entries.sort_by_key(|(_, c)| *c);
        entries
    }
}

// ---------------------------------------------------------------------------
// Soft constraints
// ---------------------------------------------------------------------------

/// A weighted soft constraint for MaxSMT optimization.
#[derive(Debug, Clone)]
pub struct SoftConstraint {
    /// The SMT expression to satisfy.
    pub expr: SmtExpr,
    /// Weight of this constraint (penalty for violation).
    pub weight: u64,
    /// Human-readable label.
    pub label: String,
}

impl SoftConstraint {
    pub fn new(expr: SmtExpr, weight: u64, label: &str) -> Self {
        Self {
            expr,
            weight,
            label: label.to_string(),
        }
    }
}

/// A MaxSMT problem combining hard and soft constraints.
#[derive(Debug, Clone)]
pub struct MaxSmtProblem {
    /// Hard constraints that MUST be satisfied.
    pub hard_constraints: Vec<SmtExpr>,
    /// Soft constraints with weights (to be maximized).
    pub soft_constraints: Vec<SoftConstraint>,
    /// Declarations.
    pub declarations: Vec<(String, String)>,
}

impl MaxSmtProblem {
    pub fn new() -> Self {
        Self {
            hard_constraints: Vec::new(),
            soft_constraints: Vec::new(),
            declarations: Vec::new(),
        }
    }

    pub fn add_hard(&mut self, expr: SmtExpr) {
        self.hard_constraints.push(expr);
    }

    pub fn add_soft(&mut self, expr: SmtExpr, weight: u64, label: &str) {
        self.soft_constraints
            .push(SoftConstraint::new(expr, weight, label));
    }

    pub fn declare(&mut self, name: &str, sort: &str) {
        self.declarations
            .push((name.to_string(), sort.to_string()));
    }

    pub fn total_soft_weight(&self) -> u64 {
        self.soft_constraints.iter().map(|sc| sc.weight).sum()
    }

    /// Render the MaxSMT problem as a SMTLIB2 script with (assert-soft ...).
    pub fn to_smtlib2(&self, config: &SolverConfig) -> String {
        let mut lines = Vec::new();
        lines.push(config.to_smtlib2_preamble());
        lines.push(String::new());

        for (name, sort) in &self.declarations {
            lines.push(format!("(declare-const {} {})", name, sort));
        }
        lines.push(String::new());

        for hc in &self.hard_constraints {
            lines.push(format!("(assert {})", crate::solver::render_smt_expr(hc)));
        }
        lines.push(String::new());

        for sc in &self.soft_constraints {
            lines.push(format!(
                "(assert-soft {} :weight {} :id {})",
                crate::solver::render_smt_expr(&sc.expr),
                sc.weight,
                sc.label,
            ));
        }
        lines.push(String::new());

        lines.push("(check-sat)".to_string());
        if config.produce_models {
            lines.push("(get-model)".to_string());
        }
        lines.push("(exit)".to_string());
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Optimization result
// ---------------------------------------------------------------------------

/// Result of an isolation level optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Assigned isolation levels per transaction.
    pub assignments: HashMap<TransactionId, IsolationLevel>,
    /// Total cost of the assignment.
    pub total_cost: u64,
    /// Whether this is an optimal result or a greedy approximation.
    pub is_optimal: bool,
    /// Time spent optimizing.
    pub elapsed: Duration,
    /// Soft constraints that were violated.
    pub violated_soft: Vec<String>,
}

impl OptimizationResult {
    pub fn num_transactions(&self) -> usize {
        self.assignments.len()
    }

    pub fn get_level(&self, txn: &TransactionId) -> Option<&IsolationLevel> {
        self.assignments.get(txn)
    }

    /// Build a summary string.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        let mut sorted_assignments: Vec<_> = self.assignments.iter().collect();
        sorted_assignments.sort_by_key(|(id, _)| format!("{}", id));

        for (txn, level) in &sorted_assignments {
            parts.push(format!("{}: {:?}", txn, level));
        }
        format!(
            "cost={} optimal={} ({}) [{}]",
            self.total_cost,
            self.is_optimal,
            self.elapsed.as_millis(),
            parts.join(", "),
        )
    }
}

// ---------------------------------------------------------------------------
// MaxSmtOptimizer
// ---------------------------------------------------------------------------

/// Optimizer that uses MaxSMT to find cost-optimal isolation level assignments.
pub struct MaxSmtOptimizer<S: SmtSolver> {
    solver: S,
    cost_model: IsolationCostModel,
    timeout: Duration,
}

impl<S: SmtSolver> MaxSmtOptimizer<S> {
    pub fn new(solver: S, cost_model: IsolationCostModel, timeout: Duration) -> Self {
        Self {
            solver,
            cost_model,
            timeout,
        }
    }

    /// Encode the isolation level choice for a transaction as an integer variable.
    fn level_var(txn_idx: usize) -> String {
        format!("iso_level_t{}", txn_idx)
    }

    /// Map isolation levels to integers for encoding.
    fn level_to_int(level: &IsolationLevel) -> i64 {
        match level {
            IsolationLevel::ReadUncommitted => 0,
            IsolationLevel::ReadCommitted => 1,
            IsolationLevel::RepeatableRead => 2,
            IsolationLevel::Serializable => 3,
            _ => 3, // Default to serializable for non-standard levels
        }
    }

    /// Map integers back to isolation levels.
    fn int_to_level(val: i64) -> IsolationLevel {
        match val {
            0 => IsolationLevel::ReadUncommitted,
            1 => IsolationLevel::ReadCommitted,
            2 => IsolationLevel::RepeatableRead,
            _ => IsolationLevel::Serializable,
        }
    }

    /// Build the MaxSMT problem for a set of transactions with anomaly constraints.
    pub fn build_problem(
        &self,
        num_transactions: usize,
        required_anomaly_freedom: &[(usize, usize, IsolationLevel)],
    ) -> MaxSmtProblem {
        let mut problem = MaxSmtProblem::new();

        // Declare isolation level variables for each transaction
        for t in 0..num_transactions {
            let var = Self::level_var(t);
            problem.declare(&var, "Int");
            // Bound: 0 <= iso_level <= 3
            problem.add_hard(SmtExpr::Ge(
                Box::new(SmtExpr::Const(var.clone())),
                Box::new(SmtExpr::IntLit(0)),
            ));
            problem.add_hard(SmtExpr::Le(
                Box::new(SmtExpr::Const(var.clone())),
                Box::new(SmtExpr::IntLit(3)),
            ));
        }

        // Hard constraints: required anomaly freedom
        // If two transactions must be free of anomalies up to level L,
        // both must be at least at level L.
        for (t1, t2, min_level) in required_anomaly_freedom {
            let min_val = Self::level_to_int(min_level);
            problem.add_hard(SmtExpr::Ge(
                Box::new(SmtExpr::Const(Self::level_var(*t1))),
                Box::new(SmtExpr::IntLit(min_val)),
            ));
            problem.add_hard(SmtExpr::Ge(
                Box::new(SmtExpr::Const(Self::level_var(*t2))),
                Box::new(SmtExpr::IntLit(min_val)),
            ));
        }

        // Soft constraints: prefer cheaper isolation levels
        for t in 0..num_transactions {
            let var = Self::level_var(t);
            for (level, cost) in self.cost_model.levels_by_cost() {
                let int_val = Self::level_to_int(&level);
                // Soft: prefer this level (penalize being above it)
                let at_or_below = SmtExpr::Le(
                    Box::new(SmtExpr::Const(var.clone())),
                    Box::new(SmtExpr::IntLit(int_val)),
                );
                let max_cost = self.cost_model.cost(&self.cost_model.most_expensive());
                let weight = max_cost.saturating_sub(cost) + 1;
                problem.add_soft(
                    at_or_below,
                    weight,
                    &format!("prefer_t{}_{:?}", t, level),
                );
            }
        }

        problem
    }

    /// Solve the optimization problem via MaxSMT.
    pub fn optimize(
        &mut self,
        num_transactions: usize,
        required_anomaly_freedom: &[(usize, usize, IsolationLevel)],
    ) -> IsoSpecResult<OptimizationResult> {
        let start = Instant::now();
        let problem = self.build_problem(num_transactions, required_anomaly_freedom);
        let script = problem.to_smtlib2(self.solver.config());

        let result = self.solver.check_sat(&script)?;
        let elapsed = start.elapsed();

        match result {
            SolverResult::Sat(Some(model)) => {
                let mut assignments = HashMap::new();
                let mut total_cost = 0u64;

                for t in 0..num_transactions {
                    let var = Self::level_var(t);
                    let val = model.get_int(&var).unwrap_or(3);
                    let level = Self::int_to_level(val);
                    let txn_id = TransactionId::new(t as u64);
                    total_cost += self.cost_model.cost(&level);
                    assignments.insert(txn_id, level);
                }

                Ok(OptimizationResult {
                    assignments,
                    total_cost,
                    is_optimal: true,
                    elapsed,
                    violated_soft: Vec::new(),
                })
            }
            SolverResult::Sat(None) => {
                // SAT but no model; fall back to greedy
                self.greedy_optimize(num_transactions, required_anomaly_freedom)
            }
            SolverResult::Timeout(_) => {
                self.greedy_optimize(num_transactions, required_anomaly_freedom)
            }
            SolverResult::Unsat => Err(IsoSpecError::smt_solver(
                "optimization problem is unsatisfiable",
            )),
            SolverResult::Unknown(reason) => {
                self.greedy_optimize(num_transactions, required_anomaly_freedom)
            }
        }
    }

    /// Greedy fallback: assign the cheapest level that satisfies all constraints.
    pub fn greedy_optimize(
        &self,
        num_transactions: usize,
        required_anomaly_freedom: &[(usize, usize, IsolationLevel)],
    ) -> IsoSpecResult<OptimizationResult> {
        let start = Instant::now();

        // Start everyone at the cheapest level
        let mut assignments: Vec<IsolationLevel> =
            vec![self.cost_model.cheapest(); num_transactions];

        // Raise levels to satisfy constraints
        for (t1, t2, min_level) in required_anomaly_freedom {
            let min_val = Self::level_to_int(min_level);
            if Self::level_to_int(&assignments[*t1]) < min_val {
                assignments[*t1] = *min_level;
            }
            if Self::level_to_int(&assignments[*t2]) < min_val {
                assignments[*t2] = *min_level;
            }
        }

        let mut result_assignments = HashMap::new();
        let mut total_cost = 0u64;
        for (t, level) in assignments.iter().enumerate() {
            let txn_id = TransactionId::new(t as u64);
            total_cost += self.cost_model.cost(level);
            result_assignments.insert(txn_id, *level);
        }

        Ok(OptimizationResult {
            assignments: result_assignments,
            total_cost,
            is_optimal: false,
            elapsed: start.elapsed(),
            violated_soft: Vec::new(),
        })
    }

    pub fn cost_model(&self) -> &IsolationCostModel {
        &self.cost_model
    }
}

// ---------------------------------------------------------------------------
// GreedyFallback – standalone greedy optimizer
// ---------------------------------------------------------------------------

/// Standalone greedy optimizer that does not require an SMT solver.
pub struct GreedyFallback {
    cost_model: IsolationCostModel,
}

impl GreedyFallback {
    pub fn new(cost_model: IsolationCostModel) -> Self {
        Self { cost_model }
    }

    /// Greedily assign the cheapest isolation levels satisfying all constraints.
    pub fn optimize(
        &self,
        num_transactions: usize,
        constraints: &[(usize, usize, IsolationLevel)],
    ) -> OptimizationResult {
        let start = Instant::now();
        let levels_by_cost = self.cost_model.levels_by_cost();
        let cheapest = levels_by_cost
            .first()
            .map(|(l, _)| *l)
            .unwrap_or(IsolationLevel::ReadUncommitted);

        let mut assignments = vec![cheapest; num_transactions];

        // Satisfy constraints by raising levels
        let mut changed = true;
        while changed {
            changed = false;
            for (t1, t2, min_level) in constraints {
                let min_val = MaxSmtOptimizer::<crate::solver::MockSmtSolver>::level_to_int(min_level);
                let cur1 = MaxSmtOptimizer::<crate::solver::MockSmtSolver>::level_to_int(&assignments[*t1]);
                let cur2 = MaxSmtOptimizer::<crate::solver::MockSmtSolver>::level_to_int(&assignments[*t2]);
                if cur1 < min_val {
                    assignments[*t1] = *min_level;
                    changed = true;
                }
                if cur2 < min_val {
                    assignments[*t2] = *min_level;
                    changed = true;
                }
            }
        }

        let mut result_assignments = HashMap::new();
        let mut total_cost = 0u64;
        for (t, level) in assignments.iter().enumerate() {
            let txn_id = TransactionId::new(t as u64);
            total_cost += self.cost_model.cost(level);
            result_assignments.insert(txn_id, *level);
        }

        OptimizationResult {
            assignments: result_assignments,
            total_cost,
            is_optimal: false,
            elapsed: start.elapsed(),
            violated_soft: Vec::new(),
        }
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
    fn test_cost_model_defaults() {
        let cm = IsolationCostModel::default();
        assert_eq!(cm.cost(&IsolationLevel::ReadUncommitted), 1);
        assert_eq!(cm.cost(&IsolationLevel::ReadCommitted), 2);
        assert_eq!(cm.cost(&IsolationLevel::RepeatableRead), 4);
        assert_eq!(cm.cost(&IsolationLevel::Serializable), 8);
    }

    #[test]
    fn test_cost_model_cheapest() {
        let cm = IsolationCostModel::default();
        assert_eq!(cm.cheapest(), IsolationLevel::ReadUncommitted);
        assert_eq!(cm.most_expensive(), IsolationLevel::Serializable);
    }

    #[test]
    fn test_cost_model_levels_by_cost() {
        let cm = IsolationCostModel::default();
        let sorted = cm.levels_by_cost();
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted[0].0, IsolationLevel::ReadUncommitted);
        assert_eq!(sorted[3].0, IsolationLevel::Serializable);
    }

    #[test]
    fn test_soft_constraint() {
        let sc = SoftConstraint::new(SmtExpr::BoolLit(true), 5, "test");
        assert_eq!(sc.weight, 5);
        assert_eq!(sc.label, "test");
    }

    #[test]
    fn test_max_smt_problem_build() {
        let solver = MockSmtSolver::always_sat();
        let cm = IsolationCostModel::default();
        let opt = MaxSmtOptimizer::new(solver, cm, Duration::from_secs(10));
        let problem = opt.build_problem(3, &[]);
        assert_eq!(problem.declarations.len(), 3);
        assert!(problem.hard_constraints.len() >= 6); // bounds
        assert!(problem.soft_constraints.len() > 0);
    }

    #[test]
    fn test_max_smt_problem_with_constraints() {
        let solver = MockSmtSolver::always_sat();
        let cm = IsolationCostModel::default();
        let opt = MaxSmtOptimizer::new(solver, cm, Duration::from_secs(10));
        let constraints = vec![
            (0, 1, IsolationLevel::RepeatableRead),
        ];
        let problem = opt.build_problem(2, &constraints);
        // Hard constraints include bounds + anomaly freedom
        assert!(problem.hard_constraints.len() >= 6);
    }

    #[test]
    fn test_greedy_fallback_no_constraints() {
        let cm = IsolationCostModel::default();
        let greedy = GreedyFallback::new(cm);
        let result = greedy.optimize(3, &[]);
        // All should be at cheapest level
        for (_, level) in &result.assignments {
            assert_eq!(*level, IsolationLevel::ReadUncommitted);
        }
        assert_eq!(result.total_cost, 3); // 3 * 1
        assert!(!result.is_optimal);
    }

    #[test]
    fn test_greedy_fallback_with_constraints() {
        let cm = IsolationCostModel::default();
        let greedy = GreedyFallback::new(cm);
        let constraints = vec![(0, 1, IsolationLevel::RepeatableRead)];
        let result = greedy.optimize(3, &constraints);
        let t0 = result.get_level(&TransactionId::new(0)).unwrap();
        let t1 = result.get_level(&TransactionId::new(1)).unwrap();
        let t2 = result.get_level(&TransactionId::new(2)).unwrap();
        assert_eq!(*t0, IsolationLevel::RepeatableRead);
        assert_eq!(*t1, IsolationLevel::RepeatableRead);
        assert_eq!(*t2, IsolationLevel::ReadUncommitted);
    }

    #[test]
    fn test_optimization_result_summary() {
        let mut assignments = HashMap::new();
        assignments.insert(TransactionId::new(0), IsolationLevel::ReadCommitted);
        assignments.insert(TransactionId::new(1), IsolationLevel::Serializable);
        let result = OptimizationResult {
            assignments,
            total_cost: 10,
            is_optimal: true,
            elapsed: Duration::from_millis(42),
            violated_soft: Vec::new(),
        };
        let summary = result.summary();
        assert!(summary.contains("cost=10"));
        assert!(summary.contains("optimal=true"));
    }

    #[test]
    fn test_max_smt_problem_render() {
        let config = SolverConfig::default();
        let mut problem = MaxSmtProblem::new();
        problem.declare("x", "Int");
        problem.add_hard(SmtExpr::Ge(
            Box::new(SmtExpr::Const("x".into())),
            Box::new(SmtExpr::IntLit(0)),
        ));
        problem.add_soft(
            SmtExpr::Le(
                Box::new(SmtExpr::Const("x".into())),
                Box::new(SmtExpr::IntLit(5)),
            ),
            3,
            "prefer_low",
        );
        let script = problem.to_smtlib2(&config);
        assert!(script.contains("declare-const x Int"));
        assert!(script.contains("assert (>= x 0)"));
        assert!(script.contains("assert-soft"));
        assert!(script.contains(":weight 3"));
    }
}
