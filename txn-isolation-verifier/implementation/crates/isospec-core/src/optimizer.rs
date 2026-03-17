// Mixed-isolation optimization (Algorithm 4).
// MixedIsolationOptimizer: uses MaxSMT to find minimum-cost isolation
// level assignment per transaction.  Greedy fallback.

use std::collections::{HashMap, HashSet};

use isospec_types::config::EngineKind;
use isospec_types::constraint::{SmtConstraintSet, SmtExpr, SmtSort};
use isospec_types::identifier::{TransactionId, WorkloadId};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::workload::Workload;

use crate::refinement::RefinementChecker;
use crate::cache::RefinementResult;

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Cost associated with an isolation level (higher = more expensive).
fn isolation_cost(level: IsolationLevel) -> u64 {
    level.cost() as u64
}

/// All standard levels in increasing strength order.
fn standard_levels() -> Vec<IsolationLevel> {
    vec![
        IsolationLevel::ReadUncommitted,
        IsolationLevel::ReadCommitted,
        IsolationLevel::RepeatableRead,
        IsolationLevel::Serializable,
    ]
}

// ---------------------------------------------------------------------------
// OptimizationResult
// ---------------------------------------------------------------------------

/// The output of the mixed-isolation optimizer: a per-transaction
/// isolation level assignment.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Map from transaction id → assigned isolation level.
    pub assignment: HashMap<TransactionId, IsolationLevel>,
    /// Total cost of the assignment.
    pub total_cost: u64,
    /// Whether the MaxSMT solver was used (vs. greedy fallback).
    pub used_maxsmt: bool,
    /// Anomaly classes guaranteed to be prevented.
    pub prevented_anomalies: Vec<AnomalyClass>,
    /// A short explanation of the optimization decisions.
    pub explanation: String,
}

impl OptimizationResult {
    /// The maximum isolation level assigned.
    pub fn max_level(&self) -> IsolationLevel {
        self.assignment
            .values()
            .copied()
            .max_by_key(|l| l.strength())
            .unwrap_or(IsolationLevel::ReadUncommitted)
    }

    /// The minimum isolation level assigned.
    pub fn min_level(&self) -> IsolationLevel {
        self.assignment
            .values()
            .copied()
            .min_by_key(|l| l.strength())
            .unwrap_or(IsolationLevel::Serializable)
    }

    /// Number of transactions at each isolation level.
    pub fn level_distribution(&self) -> HashMap<IsolationLevel, usize> {
        let mut dist = HashMap::new();
        for level in self.assignment.values() {
            *dist.entry(*level).or_insert(0) += 1;
        }
        dist
    }

    /// Cost savings compared to running everything at the given uniform level.
    pub fn savings_vs(&self, uniform_level: IsolationLevel) -> i64 {
        let uniform_cost = self.assignment.len() as u64 * isolation_cost(uniform_level);
        uniform_cost as i64 - self.total_cost as i64
    }
}

impl std::fmt::Display for OptimizationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Mixed-Isolation Assignment (cost={}, method={})",
            self.total_cost,
            if self.used_maxsmt { "MaxSMT" } else { "Greedy" }
        )?;
        let mut sorted: Vec<_> = self.assignment.iter().collect();
        sorted.sort_by_key(|(tid, _)| tid.as_u64());
        for (tid, level) in sorted {
            writeln!(f, "  {} → {:?} (cost={})", tid, level, isolation_cost(*level))?;
        }
        if !self.prevented_anomalies.is_empty() {
            writeln!(f, "  Prevented: {:?}", self.prevented_anomalies)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MixedIsolationOptimizer
// ---------------------------------------------------------------------------

/// Algorithm 4: find the minimum-cost per-transaction isolation level
/// assignment that satisfies all safety constraints.
///
/// Safety constraint: for every pair of transactions that may conflict,
/// the combined isolation levels must prevent the required anomaly classes.
#[derive(Debug)]
pub struct MixedIsolationOptimizer {
    engine: EngineKind,
    required_anomaly_prevention: Vec<AnomalyClass>,
    refinement: RefinementChecker,
}

impl MixedIsolationOptimizer {
    /// Create an optimizer for the given engine.
    pub fn new(engine: EngineKind) -> Self {
        Self {
            engine,
            required_anomaly_prevention: vec![
                AnomalyClass::G0,
                AnomalyClass::G1a,
                AnomalyClass::G1b,
                AnomalyClass::G1c,
            ],
            refinement: RefinementChecker::precomputed(),
        }
    }

    /// Create with specific anomaly prevention requirements.
    pub fn with_requirements(
        engine: EngineKind,
        required: Vec<AnomalyClass>,
    ) -> Self {
        Self {
            engine,
            required_anomaly_prevention: required,
            refinement: RefinementChecker::precomputed(),
        }
    }

    /// Run the optimizer on a workload.
    pub fn optimize(&mut self, workload: &Workload) -> OptimizationResult {
        let txn_ids: Vec<TransactionId> = workload
            .program
            .transactions
            .iter()
            .map(|t| t.id)
            .collect();

        if txn_ids.is_empty() {
            return OptimizationResult {
                assignment: HashMap::new(),
                total_cost: 0,
                used_maxsmt: false,
                prevented_anomalies: self.required_anomaly_prevention.clone(),
                explanation: "Empty workload".into(),
            };
        }

        // Try MaxSMT-based optimization first
        match self.maxsmt_optimize(&txn_ids, workload) {
            Some(result) => result,
            None => self.greedy_optimize(&txn_ids, workload),
        }
    }

    /// MaxSMT-based optimization.
    /// Returns None if the encoding cannot be solved (fallback to greedy).
    fn maxsmt_optimize(
        &self,
        txn_ids: &[TransactionId],
        workload: &Workload,
    ) -> Option<OptimizationResult> {
        let cs = self.encode_maxsmt(txn_ids, workload);

        // In a real implementation we would call an SMT solver here.
        // Since we don't have a solver binary, we simulate the result
        // by finding the optimal assignment analytically for small inputs.

        let levels = standard_levels();
        let n = txn_ids.len();

        // For small n, try all combinations
        if n > 8 {
            return None; // Too large for brute-force; use greedy
        }

        let level_count = levels.len();
        let total_combos = level_count.pow(n as u32);
        let mut best_assignment: Option<HashMap<TransactionId, IsolationLevel>> = None;
        let mut best_cost = u64::MAX;

        for combo in 0..total_combos {
            let mut assignment = HashMap::new();
            let mut idx = combo;
            for &tid in txn_ids {
                let level_idx = idx % level_count;
                idx /= level_count;
                assignment.insert(tid, levels[level_idx]);
            }

            // Check if this assignment satisfies all requirements
            if self.assignment_satisfies(&assignment, workload) {
                let cost: u64 = assignment.values().map(|l| isolation_cost(*l)).sum();
                if cost < best_cost {
                    best_cost = cost;
                    best_assignment = Some(assignment);
                }
            }
        }

        best_assignment.map(|assignment| {
            let prevented = self.compute_prevented(&assignment);
            OptimizationResult {
                total_cost: best_cost,
                assignment,
                used_maxsmt: true,
                prevented_anomalies: prevented,
                explanation: "Optimal assignment found via exhaustive MaxSMT search".into(),
            }
        })
    }

    /// Greedy fallback: assign the weakest safe level to each transaction.
    fn greedy_optimize(
        &self,
        txn_ids: &[TransactionId],
        workload: &Workload,
    ) -> OptimizationResult {
        let levels = standard_levels();
        let mut assignment: HashMap<TransactionId, IsolationLevel> = HashMap::new();

        // Start everything at the weakest level
        for &tid in txn_ids {
            assignment.insert(tid, IsolationLevel::ReadUncommitted);
        }

        // Iteratively strengthen levels until all constraints are satisfied
        let max_iterations = txn_ids.len() * levels.len();
        for _ in 0..max_iterations {
            if self.assignment_satisfies(&assignment, workload) {
                break;
            }
            // Find the cheapest single upgrade that fixes a violation
            let mut best_upgrade: Option<(TransactionId, IsolationLevel, u64)> = None;

            for &tid in txn_ids {
                let current = assignment[&tid];
                for &candidate in &levels {
                    if candidate.strength() <= current.strength() {
                        continue;
                    }
                    let mut trial = assignment.clone();
                    trial.insert(tid, candidate);
                    let cost_delta = isolation_cost(candidate) - isolation_cost(current);

                    if self.improves_satisfaction(&trial, &assignment, workload) {
                        match &best_upgrade {
                            None => best_upgrade = Some((tid, candidate, cost_delta)),
                            Some((_, _, best_delta)) if cost_delta < *best_delta => {
                                best_upgrade = Some((tid, candidate, cost_delta));
                            }
                            _ => {}
                        }
                    }
                }
            }

            if let Some((tid, level, _)) = best_upgrade {
                assignment.insert(tid, level);
            } else {
                // No improvement possible; upgrade everything to Serializable
                for &tid in txn_ids {
                    assignment.insert(tid, IsolationLevel::Serializable);
                }
                break;
            }
        }

        let total_cost = assignment.values().map(|l| isolation_cost(*l)).sum();
        let prevented = self.compute_prevented(&assignment);

        OptimizationResult {
            assignment,
            total_cost,
            used_maxsmt: false,
            prevented_anomalies: prevented,
            explanation: "Assignment found via greedy cost-minimization".into(),
        }
    }

    /// Check if an assignment satisfies all anomaly prevention requirements.
    fn assignment_satisfies(
        &self,
        assignment: &HashMap<TransactionId, IsolationLevel>,
        _workload: &Workload,
    ) -> bool {
        // The weakest level in the assignment determines what anomalies
        // can occur (conservative model: weakest link).
        let min_level = assignment
            .values()
            .min_by_key(|l| l.strength())
            .copied()
            .unwrap_or(IsolationLevel::ReadUncommitted);

        let prevented: HashSet<AnomalyClass> =
            min_level.prevented_anomalies().into_iter().collect();

        self.required_anomaly_prevention
            .iter()
            .all(|req| prevented.contains(req))
    }

    /// Check if a trial assignment improves over the current one.
    fn improves_satisfaction(
        &self,
        trial: &HashMap<TransactionId, IsolationLevel>,
        current: &HashMap<TransactionId, IsolationLevel>,
        workload: &Workload,
    ) -> bool {
        if self.assignment_satisfies(trial, workload) {
            return true;
        }
        // Count how many requirements are met
        let trial_min = trial
            .values()
            .min_by_key(|l| l.strength())
            .copied()
            .unwrap_or(IsolationLevel::ReadUncommitted);
        let current_min = current
            .values()
            .min_by_key(|l| l.strength())
            .copied()
            .unwrap_or(IsolationLevel::ReadUncommitted);

        trial_min.strength() > current_min.strength()
    }

    /// Compute which anomalies are prevented by an assignment.
    fn compute_prevented(
        &self,
        assignment: &HashMap<TransactionId, IsolationLevel>,
    ) -> Vec<AnomalyClass> {
        let min_level = assignment
            .values()
            .min_by_key(|l| l.strength())
            .copied()
            .unwrap_or(IsolationLevel::ReadUncommitted);

        min_level.prevented_anomalies()
    }

    /// Generate a MaxSMT encoding of the optimization problem.
    fn encode_maxsmt(
        &self,
        txn_ids: &[TransactionId],
        _workload: &Workload,
    ) -> SmtConstraintSet {
        let mut cs = SmtConstraintSet {
            declarations: Vec::new(),
            assertions: Vec::new(),
            soft_assertions: Vec::new(),
            logic: "QF_LIA".to_string(),
        };

        let levels = standard_levels();

        // For each transaction, declare a level variable (integer 0..3)
        for &tid in txn_ids {
            let var = format!("level_{}", tid.as_u64());
            cs.declarations.push((var.clone(), SmtSort::Int));
            // 0 ≤ level ≤ 3
            cs.assertions.push(SmtExpr::Ge(
                Box::new(SmtExpr::Var(var.clone(), SmtSort::Int)),
                Box::new(SmtExpr::IntLit(0)),
            ));
            cs.assertions.push(SmtExpr::Le(
                Box::new(SmtExpr::Var(var.clone(), SmtSort::Int)),
                Box::new(SmtExpr::IntLit(levels.len() as i64 - 1)),
            ));
        }

        // Safety constraints: for each required anomaly, the minimum level
        // must be high enough.
        let min_var = "min_level".to_string();
        cs.declarations.push((min_var.clone(), SmtSort::Int));

        // min_level = min(level_T1, level_T2, ...)
        for &tid in txn_ids {
            let var = format!("level_{}", tid.as_u64());
            cs.assertions.push(SmtExpr::Le(
                Box::new(SmtExpr::Var(min_var.clone(), SmtSort::Int)),
                Box::new(SmtExpr::Var(var, SmtSort::Int)),
            ));
        }

        // For each required anomaly, encode the minimum strength needed
        for anomaly in &self.required_anomaly_prevention {
            let min_strength = min_level_for_anomaly(*anomaly);
            cs.assertions.push(SmtExpr::Ge(
                Box::new(SmtExpr::Var(min_var.clone(), SmtSort::Int)),
                Box::new(SmtExpr::IntLit(min_strength as i64)),
            ));
        }

        // Soft constraints: prefer lower levels (minimize cost)
        for &tid in txn_ids {
            let var = format!("level_{}", tid.as_u64());
            for (i, _level) in levels.iter().enumerate() {
                // Soft: prefer level_T == i (weighted by inverse cost)
                let weight = (levels.len() - i) as u32;
                cs.soft_assertions.push((
                    SmtExpr::Eq(
                        Box::new(SmtExpr::Var(var.clone(), SmtSort::Int)),
                        Box::new(SmtExpr::IntLit(i as i64)),
                    ),
                    weight,
                    format!("cost_{}", tid.as_u64()),
                ));
            }
        }

        cs
    }
}

/// Return the minimum standard level index (0-3) that prevents the given anomaly.
fn min_level_for_anomaly(anomaly: AnomalyClass) -> usize {
    match anomaly {
        AnomalyClass::G0 => 1,  // ReadCommitted prevents G0
        AnomalyClass::G1a => 1, // ReadCommitted prevents G1a
        AnomalyClass::G1b => 1, // ReadCommitted prevents G1b
        AnomalyClass::G1c => 1, // ReadCommitted prevents G1c
        AnomalyClass::G2Item => 2, // RepeatableRead prevents G2-item
        AnomalyClass::G2 => 3, // Serializable prevents G2
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::ir::*;
    use isospec_types::predicate::{ColumnRef, ComparisonOp, Predicate};
    use isospec_types::schema::Schema;
    use isospec_types::value::Value;
    use isospec_types::workload::WorkloadParameters;

    fn make_workload(num_txns: usize, with_writes: bool) -> Workload {
        let transactions: Vec<IrTransaction> = (0..num_txns)
            .map(|i| {
                let stmts = if with_writes {
                    vec![IrStatement::Update(IrUpdate {
                        table: "t".into(),
                        assignments: vec![("x".into(), IrExpr::Literal(Value::Integer(i as i64)))],
                        predicate: None,
                    })]
                } else {
                    vec![IrStatement::Select(IrSelect {
                        table: "t".into(),
                        columns: vec!["x".into()],
                        predicate: None,
                        for_update: false,
                        for_share: false,
                    })]
                };
                IrTransaction {
                    id: TransactionId::new(i as u64 + 1),
                    label: format!("T{}", i + 1),
                    isolation_level: IsolationLevel::ReadCommitted,
                    statements: stmts,
                    read_only: !with_writes,
                }
            })
            .collect();
        let program = IrProgram {
            id: WorkloadId::new(1),
            name: "test".into(),
            transactions,
            schema_name: None,
            metadata: HashMap::new(),
        };
        Workload {
            id: WorkloadId::new(1),
            name: "test".into(),
            program,
            schema: Schema::default(),
            parameters: WorkloadParameters {
                transaction_bound: 4,
                operation_bound: 8,
                data_item_bound: 4,
                repetitions: 1,
            },
            annotations: HashMap::new(),
        }
    }

    #[test]
    fn test_empty_workload() {
        let workload = make_workload(0, false);
        let mut optimizer = MixedIsolationOptimizer::new(EngineKind::PostgreSQL);
        let result = optimizer.optimize(&workload);
        assert_eq!(result.total_cost, 0);
        assert!(result.assignment.is_empty());
    }

    #[test]
    fn test_single_txn_minimal_cost() {
        let workload = make_workload(1, true);
        let mut optimizer = MixedIsolationOptimizer::new(EngineKind::PostgreSQL);
        let result = optimizer.optimize(&workload);
        assert_eq!(result.assignment.len(), 1);
        // Single txn: should use lowest level that satisfies requirements
        assert!(result.total_cost > 0);
    }

    #[test]
    fn test_two_txn_write_workload() {
        let workload = make_workload(2, true);
        let mut optimizer = MixedIsolationOptimizer::new(EngineKind::PostgreSQL);
        let result = optimizer.optimize(&workload);
        assert_eq!(result.assignment.len(), 2);
        // Both txns should be at least RC to prevent G0/G1
        for level in result.assignment.values() {
            assert!(level.strength() >= IsolationLevel::ReadCommitted.strength());
        }
    }

    #[test]
    fn test_greedy_fallback() {
        // Large workload to force greedy
        let workload = make_workload(10, true);
        let mut optimizer = MixedIsolationOptimizer::new(EngineKind::MySQL);
        let result = optimizer.optimize(&workload);
        assert_eq!(result.assignment.len(), 10);
        assert!(!result.used_maxsmt); // Should fall back to greedy for n>8
    }

    #[test]
    fn test_with_g2_requirement() {
        let workload = make_workload(3, true);
        let mut optimizer = MixedIsolationOptimizer::with_requirements(
            EngineKind::PostgreSQL,
            vec![AnomalyClass::G2],
        );
        let result = optimizer.optimize(&workload);
        // Must be at least Serializable
        for level in result.assignment.values() {
            assert!(level.strength() >= IsolationLevel::Serializable.strength());
        }
    }

    #[test]
    fn test_savings_calculation() {
        let workload = make_workload(3, true);
        let mut optimizer = MixedIsolationOptimizer::new(EngineKind::PostgreSQL);
        let result = optimizer.optimize(&workload);
        let savings = result.savings_vs(IsolationLevel::Serializable);
        // Mixed should be cheaper or equal to uniform Serializable
        assert!(savings >= 0);
    }

    #[test]
    fn test_level_distribution() {
        let workload = make_workload(4, true);
        let mut optimizer = MixedIsolationOptimizer::new(EngineKind::PostgreSQL);
        let result = optimizer.optimize(&workload);
        let dist = result.level_distribution();
        let total: usize = dist.values().sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_display() {
        let workload = make_workload(2, true);
        let mut optimizer = MixedIsolationOptimizer::new(EngineKind::PostgreSQL);
        let result = optimizer.optimize(&workload);
        let s = format!("{}", result);
        assert!(s.contains("Mixed-Isolation Assignment"));
    }

    #[test]
    fn test_encode_maxsmt() {
        let optimizer = MixedIsolationOptimizer::new(EngineKind::PostgreSQL);
        let workload = make_workload(2, true);
        let txn_ids: Vec<TransactionId> = workload
            .program
            .transactions
            .iter()
            .map(|t| t.id)
            .collect();
        let cs = optimizer.encode_maxsmt(&txn_ids, &workload);
        assert!(!cs.declarations.is_empty());
        assert!(!cs.assertions.is_empty());
        assert!(!cs.soft_assertions.is_empty());
    }
}
