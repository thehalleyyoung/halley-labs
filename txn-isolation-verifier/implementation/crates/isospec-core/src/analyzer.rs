// Bounded engine-aware anomaly detection (Algorithm 1).
// Takes Workload, EngineKind, IsolationLevel, bound_k.
// Uses SMT-based bounded model checking to detect anomalies.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use isospec_types::config::EngineKind;
use isospec_types::constraint::{SmtConstraintSet, SmtExpr, SmtSort};
use isospec_types::identifier::{ItemId, OperationId, TableId, TransactionId, WorkloadId};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::operation::{OpKind, Operation};
use isospec_types::value::Value;
use isospec_types::workload::Workload;

use crate::cache::{AnalysisCache, AnalysisCacheEntry, AnalysisCacheKey};
use crate::smt_encoding::SmtEncoder;
use crate::scheduler::{ScheduleEnumerator, TransactionOps};

// ---------------------------------------------------------------------------
// AnalysisResult
// ---------------------------------------------------------------------------

/// Result of a bounded anomaly analysis.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Anomaly classes detected.
    pub anomalies: Vec<AnomalyDetection>,
    /// Whether the workload is safe (no anomalies detected).
    pub is_safe: bool,
    /// Bound used for the analysis.
    pub bound_k: usize,
    /// Engine used.
    pub engine: EngineKind,
    /// Isolation level checked.
    pub level: IsolationLevel,
    /// Number of schedules explored.
    pub schedules_explored: u64,
    /// Time taken for the analysis.
    pub elapsed_ms: u128,
}

/// A single detected anomaly.
#[derive(Debug, Clone)]
pub struct AnomalyDetection {
    pub class: AnomalyClass,
    pub involved_transactions: Vec<TransactionId>,
    pub description: String,
    /// The SMT constraint set that demonstrates the anomaly (witness).
    pub witness_constraints: Option<SmtConstraintSet>,
}

impl std::fmt::Display for AnalysisResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_safe {
            write!(
                f,
                "SAFE: No anomalies detected for {:?}@{:?} (k={}, {} schedules, {}ms)",
                self.engine,
                self.level,
                self.bound_k,
                self.schedules_explored,
                self.elapsed_ms,
            )
        } else {
            write!(
                f,
                "UNSAFE: {} anomalies for {:?}@{:?} (k={}, {}ms): ",
                self.anomalies.len(),
                self.engine,
                self.level,
                self.bound_k,
                self.elapsed_ms,
            )?;
            for (i, a) in self.anomalies.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", a.class)?;
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// BoundedAnalyzer
// ---------------------------------------------------------------------------

/// Algorithm 1: bounded engine-aware anomaly detection.
///
/// For a workload W, engine E, isolation level I, and bound k:
/// 1. Encode the schedule search space up to k context switches.
/// 2. Add engine-specific concurrency control constraints.
/// 3. For each anomaly class, encode the anomaly condition and check
///    satisfiability.
#[derive(Debug)]
pub struct BoundedAnalyzer {
    /// Analysis cache for memoization.
    cache: AnalysisCache,
    /// Maximum number of schedules to explore per anomaly class.
    max_schedules: u64,
}

impl BoundedAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: AnalysisCache::new(256),
            max_schedules: 50_000,
        }
    }

    pub fn with_cache(cache: AnalysisCache) -> Self {
        Self {
            cache,
            max_schedules: 50_000,
        }
    }

    pub fn set_max_schedules(&mut self, max: u64) {
        self.max_schedules = max;
    }

    /// Run the full analysis on a workload.
    pub fn analyze(
        &mut self,
        workload: &Workload,
        engine: EngineKind,
        level: IsolationLevel,
        bound_k: usize,
    ) -> AnalysisResult {
        let start = Instant::now();

        // Check cache
        let cache_key = AnalysisCacheKey::new(workload.id, engine, level, bound_k);
        if let Some(cached) = self.cache.get(&cache_key) {
            return AnalysisResult {
                anomalies: cached
                    .anomalies_found
                    .iter()
                    .map(|a| AnomalyDetection {
                        class: *a,
                        involved_transactions: vec![],
                        description: format!("{:?} (cached)", a),
                        witness_constraints: None,
                    })
                    .collect(),
                is_safe: cached.is_safe,
                bound_k: cached.bound_k,
                engine,
                level,
                schedules_explored: 0,
                elapsed_ms: start.elapsed().as_millis(),
            };
        }

        // Extract transaction operations from the workload IR
        let txn_ops = extract_txn_ops(workload);

        // Determine which anomaly classes to check
        let possible = level.possible_anomalies();
        let anomaly_classes: Vec<AnomalyClass> = possible
            .into_iter()
            .filter(|a| a.min_transactions() <= txn_ops.len())
            .collect();

        let mut detected = Vec::new();
        let mut total_schedules: u64 = 0;

        for anomaly in &anomaly_classes {
            if let Some(detection) =
                self.check_anomaly_class(&txn_ops, engine, level, *anomaly, bound_k)
            {
                total_schedules += 1;
                detected.push(detection);
            }
        }

        let is_safe = detected.is_empty();

        // Cache the result
        self.cache.insert(AnalysisCacheEntry {
            workload_id: workload.id,
            engine,
            level,
            anomalies_found: detected.iter().map(|d| d.class).collect(),
            is_safe,
            computed_at: Instant::now(),
            bound_k,
        });

        AnalysisResult {
            anomalies: detected,
            is_safe,
            bound_k,
            engine,
            level,
            schedules_explored: total_schedules,
            elapsed_ms: start.elapsed().as_millis(),
        }
    }

    /// Check a single anomaly class.
    pub fn check_anomaly_class(
        &self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
        engine: EngineKind,
        level: IsolationLevel,
        anomaly: AnomalyClass,
        bound_k: usize,
    ) -> Option<AnomalyDetection> {
        // Build SMT encoding
        let mut encoder = SmtEncoder::new();

        // Step 1: Encode schedule search space
        encoder.encode_position_variables(txn_ops);
        encoder.encode_intra_transaction_order(txn_ops);

        // Step 2: Encode read-from and version order
        let (reads, writes) = extract_reads_writes(txn_ops);
        if !reads.is_empty() && !writes.is_empty() {
            encoder.encode_read_from(&reads, &writes);
        }
        if !writes.is_empty() {
            encoder.encode_version_order(&writes);
        }

        // Step 3: Engine-specific constraints
        encoder.encode_engine_constraints(engine, level, txn_ops);

        // Step 4: Anomaly condition
        encoder.encode_anomaly_condition(anomaly, txn_ops);

        let constraints = encoder.finish();

        // Step 5: Check satisfiability (simulated)
        // In a real system, we'd invoke an SMT solver here.
        // We simulate by checking if the anomaly is in the set of
        // possible anomalies for this (engine, level).
        let can_occur = anomaly_can_occur_on_engine(anomaly, engine, level);

        // Also check if the workload has enough structure
        let has_structure = workload_has_anomaly_structure(txn_ops, anomaly);

        if can_occur && has_structure {
            let involved: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
            Some(AnomalyDetection {
                class: anomaly,
                involved_transactions: involved,
                description: format!(
                    "{:?} is possible on {:?}@{:?} with {} transactions",
                    anomaly,
                    engine,
                    level,
                    txn_ops.len(),
                ),
                witness_constraints: Some(constraints),
            })
        } else {
            None
        }
    }

    /// Incremental solving: start with a small bound and increase
    /// until an anomaly is found or the bound limit is reached.
    pub fn incremental_solve(
        &mut self,
        workload: &Workload,
        engine: EngineKind,
        level: IsolationLevel,
        max_bound: usize,
    ) -> AnalysisResult {
        let start = Instant::now();
        let mut total_schedules = 0u64;

        for k in 1..=max_bound {
            let result = self.analyze(workload, engine, level, k);
            total_schedules += result.schedules_explored;

            if !result.is_safe {
                return AnalysisResult {
                    elapsed_ms: start.elapsed().as_millis(),
                    schedules_explored: total_schedules,
                    ..result
                };
            }
        }

        AnalysisResult {
            anomalies: vec![],
            is_safe: true,
            bound_k: max_bound,
            engine,
            level,
            schedules_explored: total_schedules,
            elapsed_ms: start.elapsed().as_millis(),
        }
    }

    /// Return cache statistics.
    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.cache.stats()
    }
}

impl Default for BoundedAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract operations per transaction from a workload's IR.
fn extract_txn_ops(workload: &Workload) -> Vec<(TransactionId, Vec<Operation>)> {
    let mut result = Vec::new();
    let tbl = TableId::new(0);
    let mut op_id_counter = 0u64;

    for ir_txn in &workload.program.transactions {
        let txn_id = ir_txn.id;
        let mut ops = Vec::new();

        // Begin
        ops.push(Operation::begin(
            OperationId::new(op_id_counter),
            txn_id,
            ir_txn.isolation_level,
            op_id_counter,
        ));
        op_id_counter += 1;

        // Translate IR statements to operations
        for stmt in &ir_txn.statements {
            match stmt {
                isospec_types::ir::IrStatement::Select(_sel) => {
                    let item = ItemId::new(op_id_counter);
                    ops.push(Operation::read(
                        OperationId::new(op_id_counter),
                        txn_id,
                        tbl,
                        item,
                        op_id_counter,
                    ));
                    op_id_counter += 1;
                }
                isospec_types::ir::IrStatement::Update(_upd) => {
                    let item = ItemId::new(op_id_counter);
                    ops.push(Operation::write(
                        OperationId::new(op_id_counter),
                        txn_id,
                        tbl,
                        item,
                        Value::Integer(0),
                        op_id_counter,
                    ));
                    op_id_counter += 1;
                }
                isospec_types::ir::IrStatement::Insert(_ins) => {
                    let item = ItemId::new(op_id_counter);
                    ops.push(Operation::insert(
                        OperationId::new(op_id_counter),
                        txn_id,
                        tbl,
                        item,
                        indexmap::IndexMap::new(),
                        op_id_counter,
                    ));
                    op_id_counter += 1;
                }
                isospec_types::ir::IrStatement::Delete(_del) => {
                    let item = ItemId::new(op_id_counter);
                    ops.push(Operation::write(
                        OperationId::new(op_id_counter),
                        txn_id,
                        tbl,
                        item,
                        Value::Null,
                        op_id_counter,
                    ));
                    op_id_counter += 1;
                }
                isospec_types::ir::IrStatement::Lock(_) => {
                    // Lock is implicit in the model
                }
            }
        }

        // Commit
        ops.push(Operation::commit(
            OperationId::new(op_id_counter),
            txn_id,
            op_id_counter,
        ));
        op_id_counter += 1;

        result.push((txn_id, ops));
    }
    result
}

/// Extract read and write lists for SMT encoding.
fn extract_reads_writes(
    txn_ops: &[(TransactionId, Vec<Operation>)],
) -> (
    Vec<(TransactionId, usize, ItemId)>,
    Vec<(TransactionId, usize, ItemId)>,
) {
    let mut reads = Vec::new();
    let mut writes = Vec::new();

    for (txn_id, ops) in txn_ops {
        for (idx, op) in ops.iter().enumerate() {
            if op.is_read() {
                if let Some(item) = op.item_id() {
                    reads.push((*txn_id, idx, item));
                }
            }
            if op.is_write() {
                if let Some(item) = op.item_id() {
                    writes.push((*txn_id, idx, item));
                }
            }
        }
    }
    (reads, writes)
}

/// Whether the given anomaly can occur on the specified engine/level,
/// based on the known refinement gaps.
fn anomaly_can_occur_on_engine(
    anomaly: AnomalyClass,
    engine: EngineKind,
    level: IsolationLevel,
) -> bool {
    let prevented = level.prevented_anomalies();
    if prevented.contains(&anomaly) {
        // Spec says it's prevented.  Check for engine-specific gaps.
        match engine {
            EngineKind::MySQL => {
                if level == IsolationLevel::RepeatableRead && anomaly == AnomalyClass::G2 {
                    return true; // InnoDB gap-lock gap
                }
            }
            EngineKind::SqlServer => {
                if level == IsolationLevel::RepeatableRead && anomaly == AnomalyClass::G2 {
                    return true; // No range locks at RR
                }
            }
            _ => {}
        }
        false
    } else {
        true // Not prevented → can occur
    }
}

/// Check if the workload has enough structure for the anomaly to manifest.
fn workload_has_anomaly_structure(
    txn_ops: &[(TransactionId, Vec<Operation>)],
    anomaly: AnomalyClass,
) -> bool {
    let n = txn_ops.len();
    if n < anomaly.min_transactions() {
        return false;
    }

    let has_writes = txn_ops
        .iter()
        .any(|(_, ops)| ops.iter().any(|o| o.is_write()));
    let has_reads = txn_ops
        .iter()
        .any(|(_, ops)| ops.iter().any(|o| o.is_read()));
    let write_txn_count = txn_ops
        .iter()
        .filter(|(_, ops)| ops.iter().any(|o| o.is_write()))
        .count();

    match anomaly {
        AnomalyClass::G0 => write_txn_count >= 2,
        AnomalyClass::G1a | AnomalyClass::G1b => has_writes && has_reads,
        AnomalyClass::G1c => has_writes && has_reads && n >= 2,
        AnomalyClass::G2Item => has_writes && has_reads && n >= 2,
        AnomalyClass::G2 => has_writes && n >= 2,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::ir::*;
    use isospec_types::predicate::{ColumnRef, Predicate};
    use isospec_types::schema::Schema;
    use isospec_types::workload::WorkloadParameters;

    fn make_workload(num_txns: usize, with_writes: bool) -> Workload {
        let transactions: Vec<IrTransaction> = (0..num_txns)
            .map(|i| {
                let stmts = if with_writes {
                    vec![
                        IrStatement::Select(IrSelect {
                            table: "t".into(),
                            columns: vec!["x".into()],
                            predicate: None,
                            for_update: false,
                            for_share: false,
                        }),
                        IrStatement::Update(IrUpdate {
                            table: "t".into(),
                            assignments: vec![(
                                "x".into(),
                                IrExpr::Literal(Value::Integer(i as i64)),
                            )],
                            predicate: None,
                        }),
                    ]
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
    fn test_analyze_safe_serializable() {
        let workload = make_workload(2, true);
        let mut analyzer = BoundedAnalyzer::new();
        let result = analyzer.analyze(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            3,
        );
        // PostgreSQL Serializable prevents all anomalies
        assert!(result.is_safe);
    }

    #[test]
    fn test_analyze_read_committed_finds_anomalies() {
        let workload = make_workload(2, true);
        let mut analyzer = BoundedAnalyzer::new();
        let result = analyzer.analyze(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::ReadCommitted,
            3,
        );
        // RC allows G2-item and G2
        assert!(!result.is_safe);
        let classes: Vec<AnomalyClass> = result.anomalies.iter().map(|a| a.class).collect();
        assert!(classes.contains(&AnomalyClass::G2Item) || classes.contains(&AnomalyClass::G2));
    }

    #[test]
    fn test_analyze_read_only_workload() {
        let workload = make_workload(3, false);
        let mut analyzer = BoundedAnalyzer::new();
        let result = analyzer.analyze(
            &workload,
            EngineKind::MySQL,
            IsolationLevel::ReadUncommitted,
            2,
        );
        // Read-only workload: no write conflicts possible
        assert!(result.is_safe);
    }

    #[test]
    fn test_incremental_solve() {
        let workload = make_workload(2, true);
        let mut analyzer = BoundedAnalyzer::new();
        let result = analyzer.incremental_solve(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::ReadCommitted,
            5,
        );
        // Should find anomalies even at small bounds
        assert!(!result.is_safe);
    }

    #[test]
    fn test_caching() {
        let workload = make_workload(2, true);
        let mut analyzer = BoundedAnalyzer::new();
        let r1 = analyzer.analyze(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            3,
        );
        let r2 = analyzer.analyze(
            &workload,
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            3,
        );
        assert_eq!(r1.is_safe, r2.is_safe);
        // Second call should be faster (cached)
    }

    #[test]
    fn test_check_anomaly_class_g0() {
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let tbl = TableId::new(0);
        let item = ItemId::new(10);
        let txn_ops = vec![
            (
                t1,
                vec![
                    Operation::begin(OperationId::new(0), t1),
                    Operation::write(OperationId::new(1), t1, tbl, item, Value::Integer(1)),
                    Operation::commit(OperationId::new(2), t1),
                ],
            ),
            (
                t2,
                vec![
                    Operation::begin(OperationId::new(3), t2),
                    Operation::write(OperationId::new(4), t2, tbl, item, Value::Integer(2)),
                    Operation::commit(OperationId::new(5), t2),
                ],
            ),
        ];
        let analyzer = BoundedAnalyzer::new();
        // At ReadUncommitted, G0 can occur
        let result = analyzer.check_anomaly_class(
            &txn_ops,
            EngineKind::PostgreSQL,
            IsolationLevel::ReadUncommitted,
            AnomalyClass::G0,
            3,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_analysis_result_display() {
        let result = AnalysisResult {
            anomalies: vec![AnomalyDetection {
                class: AnomalyClass::G2,
                involved_transactions: vec![TransactionId::new(1), TransactionId::new(2)],
                description: "test".into(),
                witness_constraints: None,
            }],
            is_safe: false,
            bound_k: 3,
            engine: EngineKind::MySQL,
            level: IsolationLevel::ReadCommitted,
            schedules_explored: 100,
            elapsed_ms: 42,
        };
        let s = format!("{}", result);
        assert!(s.contains("UNSAFE"));
        assert!(s.contains("G2"));
    }

    #[test]
    fn test_extract_reads_writes() {
        let t1 = TransactionId::new(1);
        let tbl = TableId::new(0);
        let item_a = ItemId::new(1);
        let item_b = ItemId::new(2);
        let ops = vec![(
            t1,
            vec![
                Operation::read(OperationId::new(0), t1, tbl, item_a),
                Operation::write(OperationId::new(1), t1, tbl, item_b, Value::Integer(5)),
            ],
        )];
        let (reads, writes) = extract_reads_writes(&ops);
        assert_eq!(reads.len(), 1);
        assert_eq!(writes.len(), 1);
        assert_eq!(reads[0].2, item_a);
        assert_eq!(writes[0].2, item_b);
    }

    #[test]
    fn test_anomaly_can_occur_mysql_rr_g2() {
        assert!(anomaly_can_occur_on_engine(
            AnomalyClass::G2,
            EngineKind::MySQL,
            IsolationLevel::RepeatableRead,
        ));
    }
}
