//! Workload generators for benchmarking.
//!
//! Generates random, adversarial, and systematic workloads with configurable
//! parameters for use in benchmark experiments.

use std::collections::HashMap;

use isospec_types::identifier::{
    IdAllocator, ItemId, OperationId, TableId, TransactionId, WorkloadId,
};
use isospec_types::ir::{IrExpr, IrInsert, IrProgram, IrSelect, IrStatement, IrTransaction, IrUpdate};
use isospec_types::isolation::IsolationLevel;
use isospec_types::predicate::Predicate;
use isospec_types::value::Value;
use isospec_types::workload::{Workload, WorkloadParameters, WorkloadSuite};
use isospec_types::schema::Schema;

// ---------------------------------------------------------------------------
// Generator configuration
// ---------------------------------------------------------------------------

/// Parameters governing random workload generation.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub txn_count: usize,
    pub ops_per_txn: usize,
    pub table_count: usize,
    pub item_count: usize,
    pub read_write_ratio: f64,
    pub predicate_probability: f64,
    pub insert_probability: f64,
    pub isolation_level: IsolationLevel,
    pub seed: u64,
}

impl GeneratorConfig {
    pub fn new() -> Self {
        Self {
            txn_count: 3,
            ops_per_txn: 4,
            table_count: 2,
            item_count: 5,
            read_write_ratio: 0.5,
            predicate_probability: 0.1,
            insert_probability: 0.05,
            isolation_level: IsolationLevel::ReadCommitted,
            seed: 42,
        }
    }

    pub fn with_txn_count(mut self, n: usize) -> Self { self.txn_count = n; self }
    pub fn with_ops(mut self, n: usize) -> Self { self.ops_per_txn = n; self }
    pub fn with_tables(mut self, n: usize) -> Self { self.table_count = n; self }
    pub fn with_items(mut self, n: usize) -> Self { self.item_count = n; self }
    pub fn with_rw_ratio(mut self, r: f64) -> Self { self.read_write_ratio = r; self }
    pub fn with_isolation(mut self, level: IsolationLevel) -> Self { self.isolation_level = level; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (xorshift64)
// ---------------------------------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 { return 0; }
        (self.next_u64() % bound as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

// ---------------------------------------------------------------------------
// Workload generator
// ---------------------------------------------------------------------------

/// Generates benchmark workloads with various strategies.
pub struct WorkloadGenerator {
    wl_alloc: IdAllocator<WorkloadId>,
}

impl WorkloadGenerator {
    pub fn new() -> Self {
        Self {
            wl_alloc: IdAllocator::new(),
        }
    }

    // ----- random workloads -----

    /// Generate a random workload according to the supplied config.
    pub fn generate_random(&mut self, cfg: &GeneratorConfig) -> Workload {
        let wl_id = self.wl_alloc.allocate();
        let mut rng = Rng::new(cfg.seed);
        let mut txn_alloc: IdAllocator<TransactionId> = IdAllocator::new();
        let mut op_alloc: IdAllocator<OperationId> = IdAllocator::new();

        let table_names: Vec<String> = (0..cfg.table_count)
            .map(|i| format!("t{}", i))
            .collect();
        let col_names = vec!["id".to_string(), "val".to_string(), "amount".to_string()];

        let mut transactions = Vec::with_capacity(cfg.txn_count);

        for _ in 0..cfg.txn_count {
            let txn_id = txn_alloc.allocate();
            let mut stmts = Vec::with_capacity(cfg.ops_per_txn);

            for _ in 0..cfg.ops_per_txn {
                let _op_id = op_alloc.allocate();
                let table = &table_names[rng.next_usize(table_names.len())];
                let item_val = rng.next_usize(cfg.item_count) as i64;
                let roll = rng.next_f64();

                if roll < cfg.insert_probability {
                    stmts.push(IrStatement::Insert(IrInsert {
                        table: table.clone(),
                        columns: col_names.clone(),
                        values: vec![vec![
                            IrExpr::Literal(Value::Integer(item_val)),
                            IrExpr::Literal(Value::Integer(rng.next_u64() as i64 % 1000)),
                            IrExpr::Literal(Value::Integer(rng.next_u64() as i64 % 10000)),
                        ]],
                    }));
                } else if roll < cfg.read_write_ratio + cfg.insert_probability {
                    let pred = Predicate::eq(
                        "id",
                        Value::Integer(item_val),
                    );
                    stmts.push(IrStatement::Select(IrSelect {
                        table: table.clone(),
                        columns: vec!["val".to_string()],
                        predicate: pred,
                        for_update: false,
                        for_share: false,
                    }));
                } else {
                    let pred = Predicate::eq("id", Value::Integer(item_val));
                    stmts.push(IrStatement::Update(IrUpdate {
                        table: table.clone(),
                        assignments: vec![(
                            "val".to_string(),
                            IrExpr::Literal(Value::Integer(rng.next_u64() as i64 % 1000)),
                        )],
                        predicate: pred,
                    }));
                }
            }

            transactions.push(IrTransaction {
                id: txn_id,
                label: format!("T{}", txn_id.as_u64()),
                isolation_level: cfg.isolation_level,
                statements: stmts,
                read_only: false,
            });
        }

        let program = IrProgram {
            id: wl_id,
            name: format!("random_{}t_{}o", cfg.txn_count, cfg.ops_per_txn),
            transactions,
            schema_name: "bench".to_string(),
            metadata: HashMap::new(),
        };

        let schema = build_minimal_schema(&table_names, &col_names);

        Workload {
            id: wl_id,
            name: program.name.clone(),
            program,
            schema,
            parameters: WorkloadParameters {
                transaction_bound: cfg.txn_count,
                operation_bound: cfg.ops_per_txn,
                data_item_bound: cfg.item_count,
                repetitions: 1,
            },
            annotations: HashMap::new(),
        }
    }

    // ----- adversarial workloads -----

    /// Generate a workload designed to trigger specific anomaly classes.
    pub fn generate_adversarial_dirty_write(&mut self) -> Workload {
        let wl_id = self.wl_alloc.allocate();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);

        let w = |table: &str, id_val: i64, new_val: i64| -> IrStatement {
            IrStatement::Update(IrUpdate {
                table: table.to_string(),
                assignments: vec![("val".to_string(), IrExpr::Literal(Value::Integer(new_val)))],
                predicate: Predicate::eq("id", Value::Integer(id_val)),
            })
        };

        let txn1 = IrTransaction {
            id: t1,
            label: "T1_dirty_w".to_string(),
            isolation_level: IsolationLevel::ReadUncommitted,
            statements: vec![w("items", 1, 100), w("items", 2, 200)],
            read_only: false,
        };
        let txn2 = IrTransaction {
            id: t2,
            label: "T2_dirty_w".to_string(),
            isolation_level: IsolationLevel::ReadUncommitted,
            statements: vec![w("items", 1, 300), w("items", 2, 400)],
            read_only: false,
        };

        self.package_workload(wl_id, "adversarial_dirty_write", vec![txn1, txn2])
    }

    /// Workload designed to expose G2-item (anti-dependency cycle) anomalies.
    pub fn generate_adversarial_g2_item(&mut self) -> Workload {
        let wl_id = self.wl_alloc.allocate();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);

        let read = |table: &str, id_val: i64| -> IrStatement {
            IrStatement::Select(IrSelect {
                table: table.to_string(),
                columns: vec!["val".to_string()],
                predicate: Predicate::eq("id", Value::Integer(id_val)),
                for_update: false,
                for_share: false,
            })
        };
        let write = |table: &str, id_val: i64, new_val: i64| -> IrStatement {
            IrStatement::Update(IrUpdate {
                table: table.to_string(),
                assignments: vec![("val".to_string(), IrExpr::Literal(Value::Integer(new_val)))],
                predicate: Predicate::eq("id", Value::Integer(id_val)),
            })
        };

        let txn1 = IrTransaction {
            id: t1,
            label: "T1_g2item".to_string(),
            isolation_level: IsolationLevel::RepeatableRead,
            statements: vec![read("items", 1), write("items", 2, 10)],
            read_only: false,
        };
        let txn2 = IrTransaction {
            id: t2,
            label: "T2_g2item".to_string(),
            isolation_level: IsolationLevel::RepeatableRead,
            statements: vec![read("items", 2), write("items", 1, 20)],
            read_only: false,
        };

        self.package_workload(wl_id, "adversarial_g2_item", vec![txn1, txn2])
    }

    /// Workload with a phantom-read pattern (G2 / predicate-level anti-dependency).
    pub fn generate_adversarial_phantom(&mut self) -> Workload {
        let wl_id = self.wl_alloc.allocate();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);

        let range_read = IrStatement::Select(IrSelect {
            table: "items".to_string(),
            columns: vec!["id".to_string(), "val".to_string()],
            predicate: Predicate::Between(
                isospec_types::predicate::ColumnRef::new("val"),
                Value::Integer(1),
                Value::Integer(100),
            ),
            for_update: false,
            for_share: false,
        });

        let insert = IrStatement::Insert(IrInsert {
            table: "items".to_string(),
            columns: vec!["id".to_string(), "val".to_string()],
            values: vec![vec![
                IrExpr::Literal(Value::Integer(999)),
                IrExpr::Literal(Value::Integer(50)),
            ]],
        });

        let txn1 = IrTransaction {
            id: t1,
            label: "T1_phantom".to_string(),
            isolation_level: IsolationLevel::RepeatableRead,
            statements: vec![range_read.clone(), range_read],
            read_only: true,
        };
        let txn2 = IrTransaction {
            id: t2,
            label: "T2_phantom".to_string(),
            isolation_level: IsolationLevel::RepeatableRead,
            statements: vec![insert],
            read_only: false,
        };

        self.package_workload(wl_id, "adversarial_phantom", vec![txn1, txn2])
    }

    // ----- systematic enumeration -----

    /// Enumerate all workloads with the given transaction/operation counts,
    /// cycling through isolation levels.
    pub fn enumerate_systematic(
        &mut self,
        max_txns: usize,
        max_ops: usize,
        levels: &[IsolationLevel],
    ) -> WorkloadSuite {
        let mut suite = WorkloadSuite::new("systematic_enumeration");
        for &txn_count in &[2, 3] {
            if txn_count > max_txns {
                break;
            }
            for &op_count in &[2, 3, 4] {
                if op_count > max_ops {
                    break;
                }
                for level in levels {
                    let cfg = GeneratorConfig::new()
                        .with_txn_count(txn_count)
                        .with_ops(op_count)
                        .with_isolation(*level)
                        .with_seed(txn_count as u64 * 1000 + op_count as u64 * 100);
                    let wl = self.generate_random(&cfg);
                    suite.add(wl);
                }
            }
        }
        suite
    }

    /// Generate a scaling suite: workloads with increasing transaction counts.
    pub fn generate_scaling_suite(
        &mut self,
        txn_counts: &[usize],
        ops_per_txn: usize,
        level: IsolationLevel,
    ) -> WorkloadSuite {
        let mut suite = WorkloadSuite::new("scaling");
        for (idx, &n) in txn_counts.iter().enumerate() {
            let cfg = GeneratorConfig::new()
                .with_txn_count(n)
                .with_ops(ops_per_txn)
                .with_isolation(level)
                .with_seed(idx as u64 + 1);
            let wl = self.generate_random(&cfg);
            suite.add(wl);
        }
        suite
    }

    // ----- helpers -----

    fn package_workload(
        &self,
        wl_id: WorkloadId,
        name: &str,
        transactions: Vec<IrTransaction>,
    ) -> Workload {
        let txn_count = transactions.len();
        let max_ops = transactions.iter().map(|t| t.statements.len()).max().unwrap_or(0);

        let program = IrProgram {
            id: wl_id,
            name: name.to_string(),
            transactions,
            schema_name: "bench".to_string(),
            metadata: HashMap::new(),
        };

        let schema = build_minimal_schema(
            &["items".to_string()],
            &["id".to_string(), "val".to_string(), "amount".to_string()],
        );

        Workload {
            id: wl_id,
            name: name.to_string(),
            program,
            schema,
            parameters: WorkloadParameters {
                transaction_bound: txn_count,
                operation_bound: max_ops,
                data_item_bound: 10,
                repetitions: 1,
            },
            annotations: HashMap::new(),
        }
    }
}

impl Default for WorkloadGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Schema helper
// ---------------------------------------------------------------------------

fn build_minimal_schema(tables: &[String], columns: &[String]) -> Schema {
    use isospec_types::column::DataType;
    use isospec_types::schema::{ColumnDef, TableSchema};

    let mut schema = Schema::new();
    let mut tid_alloc: IdAllocator<TableId> = IdAllocator::new();

    for table_name in tables {
        let tid = tid_alloc.allocate();
        let mut ts = TableSchema::new(tid, table_name.clone());
        for (ci, col) in columns.iter().enumerate() {
            let mut cd = ColumnDef {
                name: col.clone(),
                data_type: if ci == 0 { DataType::Integer } else { DataType::BigInt },
                nullable: ci != 0,
                primary_key: ci == 0,
                unique: ci == 0,
                default: None,
                auto_increment: false,
                references: None,
            };
            if ci == 0 {
                cd = cd.not_null().primary_key();
            }
            ts = ts.with_column(cd);
        }
        schema.add_table(ts);
    }
    schema
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_basic() {
        let mut gen = WorkloadGenerator::new();
        let cfg = GeneratorConfig::new().with_txn_count(3).with_ops(4);
        let wl = gen.generate_random(&cfg);
        assert_eq!(wl.program.transaction_count(), 3);
        for txn in &wl.program.transactions {
            assert_eq!(txn.statement_count(), 4);
        }
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = WorkloadGenerator::new();
        let mut gen2 = WorkloadGenerator::new();
        let cfg = GeneratorConfig::new().with_seed(123);
        let w1 = gen1.generate_random(&cfg);
        let w2 = gen2.generate_random(&cfg);
        assert_eq!(w1.program.transactions.len(), w2.program.transactions.len());
        for (t1, t2) in w1.program.transactions.iter().zip(w2.program.transactions.iter()) {
            assert_eq!(t1.statements.len(), t2.statements.len());
        }
    }

    #[test]
    fn test_adversarial_dirty_write() {
        let mut gen = WorkloadGenerator::new();
        let wl = gen.generate_adversarial_dirty_write();
        assert_eq!(wl.program.transaction_count(), 2);
        assert!(wl.program.transactions.iter().all(|t| t.has_writes()));
    }

    #[test]
    fn test_adversarial_g2_item() {
        let mut gen = WorkloadGenerator::new();
        let wl = gen.generate_adversarial_g2_item();
        assert_eq!(wl.program.transaction_count(), 2);
        assert_eq!(wl.name, "adversarial_g2_item");
    }

    #[test]
    fn test_adversarial_phantom() {
        let mut gen = WorkloadGenerator::new();
        let wl = gen.generate_adversarial_phantom();
        assert_eq!(wl.program.transaction_count(), 2);
    }

    #[test]
    fn test_systematic_enumeration() {
        let mut gen = WorkloadGenerator::new();
        let suite = gen.enumerate_systematic(
            3,
            4,
            &[IsolationLevel::ReadCommitted, IsolationLevel::Serializable],
        );
        // 2 txn counts × 3 op counts × 2 levels = 12
        assert_eq!(suite.len(), 12);
    }

    #[test]
    fn test_scaling_suite() {
        let mut gen = WorkloadGenerator::new();
        let suite = gen.generate_scaling_suite(
            &[2, 4, 8],
            3,
            IsolationLevel::Snapshot,
        );
        assert_eq!(suite.len(), 3);
    }

    #[test]
    fn test_generator_config_builder() {
        let cfg = GeneratorConfig::new()
            .with_txn_count(5)
            .with_ops(10)
            .with_tables(3)
            .with_items(20)
            .with_rw_ratio(0.7)
            .with_isolation(IsolationLevel::Serializable)
            .with_seed(999);
        assert_eq!(cfg.txn_count, 5);
        assert_eq!(cfg.ops_per_txn, 10);
        assert_eq!(cfg.table_count, 3);
        assert_eq!(cfg.item_count, 20);
        assert!((cfg.read_write_ratio - 0.7).abs() < 1e-9);
        assert_eq!(cfg.seed, 999);
    }

    #[test]
    fn test_rng_determinism() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_schema_construction() {
        let tables = vec!["a".to_string(), "b".to_string()];
        let cols = vec!["id".to_string(), "x".to_string()];
        let schema = build_minimal_schema(&tables, &cols);
        assert_eq!(schema.table_names().len(), 2);
        let ta = schema.get_table("a").unwrap();
        assert_eq!(ta.column_names().len(), 2);
    }
}
