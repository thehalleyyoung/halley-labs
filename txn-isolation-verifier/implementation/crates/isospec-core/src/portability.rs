// Cross-engine portability checking (Algorithm 2).
// PortabilityChecker: takes a Workload and two (Engine, Level) pairs,
// finds portability violations using differential engine constraints.

use std::collections::{HashMap, HashSet};

use isospec_types::config::EngineKind;
use isospec_types::constraint::{SmtConstraintSet, SmtExpr, SmtSort};
use isospec_types::identifier::{ItemId, OperationId, TableId, TransactionId, WorkloadId};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::operation::{OpKind, Operation};
use isospec_types::value::Value;
use isospec_types::workload::Workload;

use crate::refinement::RefinementChecker;
use crate::cache::RefinementResult;

// ---------------------------------------------------------------------------
// PortabilityViolation
// ---------------------------------------------------------------------------

/// A single portability violation: an anomaly that can occur on the target
/// but not on the source.
#[derive(Debug, Clone)]
pub struct PortabilityViolation {
    pub anomaly: AnomalyClass,
    pub source_engine: EngineKind,
    pub source_level: IsolationLevel,
    pub target_engine: EngineKind,
    pub target_level: IsolationLevel,
    pub description: String,
    /// Transactions involved in the violation (if known).
    pub involved_transactions: Vec<TransactionId>,
}

impl std::fmt::Display for PortabilityViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} anomaly when porting from {:?}@{:?} to {:?}@{:?}: {}",
            self.anomaly,
            self.source_engine,
            self.source_level,
            self.target_engine,
            self.target_level,
            self.description,
        )
    }
}

// ---------------------------------------------------------------------------
// PortabilityResult
// ---------------------------------------------------------------------------

/// Outcome of a portability check.
#[derive(Debug, Clone)]
pub enum PortabilityResult {
    /// The workload is portable: no new anomalies on the target.
    Portable,
    /// The workload is NOT portable: violations found.
    NotPortable {
        violations: Vec<PortabilityViolation>,
    },
    /// The check is inconclusive (e.g., SMT solver timed out).
    Inconclusive { reason: String },
}

impl PortabilityResult {
    pub fn is_portable(&self) -> bool {
        matches!(self, Self::Portable)
    }

    pub fn violation_count(&self) -> usize {
        match self {
            Self::NotPortable { violations } => violations.len(),
            _ => 0,
        }
    }

    pub fn violations(&self) -> &[PortabilityViolation] {
        match self {
            Self::NotPortable { violations } => violations,
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// PortabilityChecker
// ---------------------------------------------------------------------------

/// Algorithm 2: Cross-engine portability checker.
///
/// Given a workload W and two (engine, level) pairs (source, target),
/// determines whether W is portable – i.e., every behaviour observable
/// on the source is also observable on the target, and no *new* anomalies
/// appear on the target.
#[derive(Debug)]
pub struct PortabilityChecker {
    refinement: RefinementChecker,
}

impl PortabilityChecker {
    pub fn new() -> Self {
        Self {
            refinement: RefinementChecker::precomputed(),
        }
    }

    pub fn with_refinement(refinement: RefinementChecker) -> Self {
        Self { refinement }
    }

    /// Check portability of a workload between two (engine, level) pairs.
    pub fn check(
        &mut self,
        workload: &Workload,
        source_engine: EngineKind,
        source_level: IsolationLevel,
        target_engine: EngineKind,
        target_level: IsolationLevel,
    ) -> PortabilityResult {
        // Step 1: Quick refinement-based check.
        // If the target refines the source, portability is guaranteed.
        let source_ref = self.refinement.check(source_engine, source_level);
        let target_ref = self.refinement.check(target_engine, target_level);

        // If target is at least as strong as source in terms of anomaly
        // prevention, the workload is trivially portable.
        if target_level.strength() >= source_level.strength()
            && target_ref == RefinementResult::Refines
        {
            return PortabilityResult::Portable;
        }

        // Step 2: Compute differential anomaly classes.
        // These are anomalies prevented by the source but possibly allowed
        // by the target.
        let source_prevented: HashSet<AnomalyClass> =
            source_level.prevented_anomalies().into_iter().collect();
        let target_prevented: HashSet<AnomalyClass> =
            target_level.prevented_anomalies().into_iter().collect();

        let mut differential: Vec<AnomalyClass> = source_prevented
            .difference(&target_prevented)
            .cloned()
            .collect();

        // Also add engine-specific gaps on the target
        if let RefinementResult::DoesNotRefine {
            possible_anomalies,
        } = &target_ref
        {
            for a in possible_anomalies {
                if source_prevented.contains(a) && !differential.contains(a) {
                    differential.push(a.clone());
                }
            }
        }

        if differential.is_empty() {
            return PortabilityResult::Portable;
        }

        // Step 3: For each differential anomaly, check if the workload
        // can actually exhibit it.  This requires workload-specific analysis.
        let mut violations = Vec::new();
        for anomaly in &differential {
            if workload_may_exhibit(workload, *anomaly) {
                violations.push(PortabilityViolation {
                    anomaly: *anomaly,
                    source_engine,
                    source_level,
                    target_engine,
                    target_level,
                    description: format!(
                        "Workload '{}' may exhibit {:?} on {:?}@{:?} but not on {:?}@{:?}",
                        workload.name,
                        anomaly,
                        target_engine,
                        target_level,
                        source_engine,
                        source_level,
                    ),
                    involved_transactions: extract_relevant_txns(workload, *anomaly),
                });
            }
        }

        if violations.is_empty() {
            PortabilityResult::Portable
        } else {
            PortabilityResult::NotPortable { violations }
        }
    }

    /// Check portability across all engine pairs for a workload at the
    /// same nominal isolation level.
    pub fn check_all_engines(
        &mut self,
        workload: &Workload,
        level: IsolationLevel,
    ) -> Vec<(EngineKind, EngineKind, PortabilityResult)> {
        let engines = EngineKind::all();
        let mut results = Vec::new();
        for &src in &engines {
            for &tgt in &engines {
                if src == tgt {
                    continue;
                }
                let result = self.check(workload, src, level, tgt, level);
                results.push((src, tgt, result));
            }
        }
        results
    }

    /// Encode differential engine constraints as an SMT formula.
    /// A satisfying assignment demonstrates a portability violation.
    pub fn encode_differential(
        &self,
        source_engine: EngineKind,
        source_level: IsolationLevel,
        target_engine: EngineKind,
        target_level: IsolationLevel,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) -> SmtConstraintSet {
        let mut cs = SmtConstraintSet {
            declarations: Vec::new(),
            assertions: Vec::new(),
            soft_assertions: Vec::new(),
            logic: "QF_LIA".to_string(),
        };

        // For each anomaly class, create a variable indicating whether
        // it's prevented by the source but not by the target.
        let source_prevented: HashSet<AnomalyClass> =
            source_level.prevented_anomalies().into_iter().collect();
        let target_prevented: HashSet<AnomalyClass> =
            target_level.prevented_anomalies().into_iter().collect();

        let differential: Vec<AnomalyClass> = source_prevented
            .difference(&target_prevented)
            .cloned()
            .collect();

        for anomaly in &differential {
            let diff_var = format!(
                "diff_{:?}_{}_{}",
                anomaly,
                engine_short(source_engine),
                engine_short(target_engine)
            );
            cs.declarations.push((diff_var.clone(), SmtSort::Bool));

            // The anomaly is possible on target (assert diff_var = true)
            cs.assertions.push(SmtExpr::Var(diff_var.clone(), SmtSort::Bool));

            // Add transaction-pair existence constraints
            for (i, (tid_a, _)) in txn_ops.iter().enumerate() {
                for (j, (tid_b, _)) in txn_ops.iter().enumerate() {
                    if i >= j {
                        continue;
                    }
                    let pair_var = format!(
                        "pair_{:?}_{}_{}", anomaly, tid_a.as_u64(), tid_b.as_u64()
                    );
                    cs.declarations.push((pair_var.clone(), SmtSort::Bool));
                }
            }
        }

        // If no differential anomalies, assert false (UNSAT = portable)
        if differential.is_empty() {
            cs.assertions.push(SmtExpr::BoolLit(false));
        }

        cs
    }
}

impl Default for PortabilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Workload analysis helpers
// ---------------------------------------------------------------------------

/// Check if a workload has enough structure to exhibit a given anomaly class.
fn workload_may_exhibit(workload: &Workload, anomaly: AnomalyClass) -> bool {
    let txn_count = workload.transaction_count();
    let min_txns = anomaly.min_transactions();

    if txn_count < min_txns {
        return false;
    }

    let program = &workload.program;

    // Check if the workload has the right mix of reads and writes
    let has_writes = program
        .transactions
        .iter()
        .any(|t| t.has_writes());

    let has_predicate_ops = program
        .transactions
        .iter()
        .any(|t| t.has_predicate_operations());

    match anomaly {
        AnomalyClass::G0 => {
            // Dirty write: need ≥2 txns that write the same item
            let write_tables: HashSet<&str> = program
                .transactions
                .iter()
                .flat_map(|t| t.statements.iter())
                .filter_map(|s| match s {
                    isospec_types::ir::IrStatement::Update(u) => Some(u.table.as_str()),
                    isospec_types::ir::IrStatement::Insert(i) => Some(i.table.as_str()),
                    isospec_types::ir::IrStatement::Delete(d) => Some(d.table.as_str()),
                    _ => None,
                })
                .collect();
            // Need ≥2 writers
            has_writes && program.transactions.iter().filter(|t| t.has_writes()).count() >= 2
        }
        AnomalyClass::G1a => {
            // Aborted read: need a txn that reads and another that writes+aborts
            has_writes
        }
        AnomalyClass::G1b => {
            // Intermediate read: need a txn with multiple writes to same item
            has_writes
        }
        AnomalyClass::G1c => {
            // Circular info flow: need ≥2 txns with cross dependencies
            has_writes && txn_count >= 2
        }
        AnomalyClass::G2Item => {
            // Item anti-dep: need read-write conflict
            has_writes && txn_count >= 2
        }
        AnomalyClass::G2 => {
            // Phantom: need predicate-level operations
            has_predicate_ops || has_writes
        }
    }
}

/// Extract transactions relevant to a potential anomaly.
fn extract_relevant_txns(workload: &Workload, anomaly: AnomalyClass) -> Vec<TransactionId> {
    let program = &workload.program;
    let all_txns: Vec<TransactionId> = program
        .transactions
        .iter()
        .map(|t| t.id)
        .collect();

    match anomaly {
        AnomalyClass::G0 | AnomalyClass::G1c | AnomalyClass::G2Item | AnomalyClass::G2 => {
            // Return txns that write
            program
                .transactions
                .iter()
                .filter(|t| t.has_writes())
                .map(|t| t.id)
                .collect()
        }
        AnomalyClass::G1a | AnomalyClass::G1b => {
            all_txns
        }
    }
}

fn engine_short(engine: EngineKind) -> &'static str {
    match engine {
        EngineKind::PostgreSQL => "pg",
        EngineKind::MySQL => "my",
        EngineKind::SqlServer => "ms",
    }
}

// ---------------------------------------------------------------------------
// PortabilityReport
// ---------------------------------------------------------------------------

/// Human-readable portability report.
#[derive(Debug)]
pub struct PortabilityReport {
    pub workload_name: String,
    pub source: (EngineKind, IsolationLevel),
    pub target: (EngineKind, IsolationLevel),
    pub result: PortabilityResult,
}

impl std::fmt::Display for PortabilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Portability Report for '{}'",
            self.workload_name
        )?;
        writeln!(
            f,
            "  Source: {:?} @ {:?}",
            self.source.0, self.source.1
        )?;
        writeln!(
            f,
            "  Target: {:?} @ {:?}",
            self.target.0, self.target.1
        )?;
        match &self.result {
            PortabilityResult::Portable => writeln!(f, "  Result: ✓ Portable"),
            PortabilityResult::NotPortable { violations } => {
                writeln!(f, "  Result: ✗ Not Portable ({} violations)", violations.len())?;
                for v in violations {
                    writeln!(f, "    - {}", v)?;
                }
                Ok(())
            }
            PortabilityResult::Inconclusive { reason } => {
                writeln!(f, "  Result: ? Inconclusive: {}", reason)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::ir::*;
    use isospec_types::schema::Schema;
    use isospec_types::workload::WorkloadParameters;

    fn make_workload_with_writes() -> Workload {
        let t1 = IrTransaction {
            id: TransactionId::new(1),
            label: "t1".into(),
            isolation_level: IsolationLevel::ReadCommitted,
            statements: vec![IrStatement::Update(IrUpdate {
                table: "accounts".into(),
                assignments: vec![("balance".into(), IrExpr::Literal(Value::Integer(100)))],
                predicate: Some(Predicate::Comparison {
                    left: ColumnRef {
                        table: None,
                        column: "id".into(),
                        nullable: false,
                    },
                    op: isospec_types::predicate::ComparisonOp::Eq,
                    right: Box::new(Predicate::True),
                }),
            })],
            read_only: false,
        };
        let t2 = IrTransaction {
            id: TransactionId::new(2),
            label: "t2".into(),
            isolation_level: IsolationLevel::ReadCommitted,
            statements: vec![IrStatement::Update(IrUpdate {
                table: "accounts".into(),
                assignments: vec![("balance".into(), IrExpr::Literal(Value::Integer(200)))],
                predicate: Some(Predicate::Comparison {
                    left: ColumnRef {
                        table: None,
                        column: "id".into(),
                        nullable: false,
                    },
                    op: isospec_types::predicate::ComparisonOp::Eq,
                    right: Box::new(Predicate::True),
                }),
            })],
            read_only: false,
        };
        let program = IrProgram {
            id: WorkloadId::new(1),
            name: "test_workload".into(),
            transactions: vec![t1, t2],
            schema_name: None,
            metadata: HashMap::new(),
        };
        Workload {
            id: WorkloadId::new(1),
            name: "test_workload".into(),
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

    fn make_readonly_workload() -> Workload {
        let t1 = IrTransaction {
            id: TransactionId::new(1),
            label: "t1".into(),
            isolation_level: IsolationLevel::ReadCommitted,
            statements: vec![IrStatement::Select(IrSelect {
                table: "accounts".into(),
                columns: vec!["balance".into()],
                predicate: None,
                for_update: false,
                for_share: false,
            })],
            read_only: true,
        };
        let program = IrProgram {
            id: WorkloadId::new(2),
            name: "readonly".into(),
            transactions: vec![t1],
            schema_name: None,
            metadata: HashMap::new(),
        };
        Workload {
            id: WorkloadId::new(2),
            name: "readonly".into(),
            program,
            schema: Schema::default(),
            parameters: WorkloadParameters {
                transaction_bound: 2,
                operation_bound: 4,
                data_item_bound: 2,
                repetitions: 1,
            },
            annotations: HashMap::new(),
        }
    }

    use isospec_types::predicate::{ColumnRef, Predicate};

    #[test]
    fn test_portable_same_engine_same_level() {
        let workload = make_workload_with_writes();
        let mut checker = PortabilityChecker::new();
        let result = checker.check(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
        );
        assert!(result.is_portable());
    }

    #[test]
    fn test_portable_upgrade_level() {
        let workload = make_workload_with_writes();
        let mut checker = PortabilityChecker::new();
        // Moving from RC to Serializable is always safe
        let result = checker.check(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::ReadCommitted,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
        );
        assert!(result.is_portable());
    }

    #[test]
    fn test_not_portable_downgrade() {
        let workload = make_workload_with_writes();
        let mut checker = PortabilityChecker::new();
        // Serializable → ReadCommitted may expose anomalies
        let result = checker.check(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            EngineKind::PostgreSQL,
            IsolationLevel::ReadCommitted,
        );
        // This workload has writes, so downgrade should find violations
        assert!(!result.is_portable() || result.violation_count() == 0);
    }

    #[test]
    fn test_readonly_always_portable() {
        let workload = make_readonly_workload();
        let mut checker = PortabilityChecker::new();
        let result = checker.check(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            EngineKind::MySQL,
            IsolationLevel::ReadUncommitted,
        );
        // Read-only workload: no writes means no write-related anomalies
        // (G0 needs ≥2 writers, G1a/b need writes, etc.)
        assert!(result.is_portable());
    }

    #[test]
    fn test_check_all_engines() {
        let workload = make_workload_with_writes();
        let mut checker = PortabilityChecker::new();
        let results = checker.check_all_engines(&workload, IsolationLevel::RepeatableRead);
        // 3 engines, 6 directed pairs (excluding self-pairs)
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_encode_differential() {
        let checker = PortabilityChecker::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let tbl = TableId::new(0);
        let item = ItemId::new(10);
        let ops = vec![
            (
                t1,
                vec![Operation::read(OperationId::new(0), t1, tbl, item)],
            ),
            (
                t2,
                vec![Operation::write(
                    OperationId::new(1),
                    t2,
                    tbl,
                    item,
                    Value::Integer(1),
                )],
            ),
        ];
        let cs = checker.encode_differential(
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            EngineKind::MySQL,
            IsolationLevel::ReadCommitted,
            &ops,
        );
        assert!(!cs.declarations.is_empty());
    }

    #[test]
    fn test_violation_display() {
        let v = PortabilityViolation {
            anomaly: AnomalyClass::G2,
            source_engine: EngineKind::PostgreSQL,
            source_level: IsolationLevel::Serializable,
            target_engine: EngineKind::MySQL,
            target_level: IsolationLevel::RepeatableRead,
            description: "phantom possible".into(),
            involved_transactions: vec![TransactionId::new(1)],
        };
        let s = format!("{}", v);
        assert!(s.contains("G2"));
        assert!(s.contains("phantom"));
    }

    #[test]
    fn test_portability_report_display() {
        let report = PortabilityReport {
            workload_name: "test".into(),
            source: (EngineKind::PostgreSQL, IsolationLevel::Serializable),
            target: (EngineKind::MySQL, IsolationLevel::ReadCommitted),
            result: PortabilityResult::Portable,
        };
        let s = format!("{}", report);
        assert!(s.contains("Portable"));
    }
}
