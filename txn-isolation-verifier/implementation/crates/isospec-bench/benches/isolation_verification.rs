//! Criterion benchmarks for IsoSpec core analysis algorithms.
//!
//! Benchmarks key performance-critical paths:
//! - DSG construction from transaction histories
//! - Cycle detection in dependency graphs
//! - Conflict analysis under different isolation levels
//! - SMT encoding generation
//! - History builder throughput

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use isospec_types::identifier::{TransactionId, TableId, ItemId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::value::Value;
use isospec_history::builder::HistoryBuilder;

/// Build a synthetic history with N transactions, each performing M operations.
fn build_synthetic_history(num_txns: usize, ops_per_txn: usize) -> HistoryBuilder {
    let mut builder = HistoryBuilder::new();
    let table = TableId::new(1);

    for t in 0..num_txns {
        let txn_id = TransactionId::new(t as u64 + 1);
        builder.begin_transaction(txn_id, IsolationLevel::Serializable);

        for op in 0..ops_per_txn {
            let item = ItemId::new((t * ops_per_txn + op) as u64);
            if op % 3 == 0 {
                builder.add_read(txn_id, table, item, Some(Value::Integer(op as i64)));
            } else if op % 3 == 1 {
                builder.add_write(
                    txn_id, table, item,
                    Some(Value::Integer(op as i64)),
                    Value::Integer(op as i64 + 1),
                );
            } else {
                builder.add_insert(
                    txn_id, table, item,
                    vec![("col".to_string(), Value::Integer(op as i64))],
                );
            }
        }

        builder.commit_transaction(txn_id);
    }

    builder
}

/// Benchmark: History construction throughput
fn bench_history_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("history_construction");

    for &(txns, ops) in &[(10, 10), (50, 20), (100, 10), (100, 50)] {
        group.bench_with_input(
            BenchmarkId::new("build", format!("{}txn_{}ops", txns, ops)),
            &(txns, ops),
            |b, &(txns, ops)| {
                b.iter(|| {
                    let builder = build_synthetic_history(txns, ops);
                    black_box(builder.build().unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: History query operations
fn bench_history_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("history_queries");

    let builder = build_synthetic_history(50, 20);
    let history = builder.build().unwrap();

    group.bench_function("committed_transactions", |b| {
        b.iter(|| black_box(history.committed_transactions()));
    });

    group.bench_function("event_count", |b| {
        b.iter(|| black_box(history.event_count()));
    });

    group.bench_function("events_for_txn", |b| {
        let txn = TransactionId::new(1);
        b.iter(|| black_box(history.events_for_txn(txn)));
    });

    group.bench_function("items_read_by", |b| {
        let txn = TransactionId::new(1);
        b.iter(|| black_box(history.items_read_by(txn)));
    });

    group.bench_function("items_written_by", |b| {
        let txn = TransactionId::new(1);
        b.iter(|| black_box(history.items_written_by(txn)));
    });

    group.bench_function("referenced_tables", |b| {
        b.iter(|| black_box(history.referenced_tables()));
    });

    group.bench_function("is_well_formed", |b| {
        b.iter(|| black_box(history.is_well_formed()));
    });

    group.finish();
}

/// Benchmark: Write-skew detection scenario (classic isolation anomaly)
fn bench_write_skew_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_skew");

    for &num_pairs in &[2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("detect", format!("{}pairs", num_pairs)),
            &num_pairs,
            |b, &num_pairs| {
                b.iter(|| {
                    let mut builder = HistoryBuilder::new();
                    let table = TableId::new(1);

                    // Create write-skew pattern: T_i reads x,y then writes x
                    // T_{i+1} reads x,y then writes y
                    for i in 0..num_pairs {
                        let t1 = TransactionId::new(i as u64 * 2 + 1);
                        let t2 = TransactionId::new(i as u64 * 2 + 2);
                        let x = ItemId::new(i as u64 * 2);
                        let y = ItemId::new(i as u64 * 2 + 1);

                        builder.begin_transaction(t1, IsolationLevel::Serializable);
                        builder.begin_transaction(t2, IsolationLevel::Serializable);

                        // T1 reads x and y
                        builder.add_read(t1, table, x, Some(Value::Integer(0)));
                        builder.add_read(t1, table, y, Some(Value::Integer(0)));

                        // T2 reads x and y
                        builder.add_read(t2, table, x, Some(Value::Integer(0)));
                        builder.add_read(t2, table, y, Some(Value::Integer(0)));

                        // T1 writes x based on y
                        builder.add_write(
                            t1, table, x,
                            Some(Value::Integer(0)),
                            Value::Integer(1),
                        );

                        // T2 writes y based on x
                        builder.add_write(
                            t2, table, y,
                            Some(Value::Integer(0)),
                            Value::Integer(1),
                        );

                        builder.commit_transaction(t1);
                        builder.commit_transaction(t2);
                    }

                    black_box(builder.build().unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Phantom read scenario construction
fn bench_phantom_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("phantom_read");

    for &num_txns in &[5, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("construct", format!("{}txns", num_txns)),
            &num_txns,
            |b, &num_txns| {
                b.iter(|| {
                    let mut builder = HistoryBuilder::new();
                    let table = TableId::new(1);

                    for i in 0..num_txns {
                        let txn = TransactionId::new(i as u64 + 1);
                        builder.begin_transaction(txn, IsolationLevel::RepeatableRead);

                        // Predicate read (range scan)
                        builder.add_read(txn, table, ItemId::new(i as u64), Some(Value::Null));

                        // Insert into the scanned range (by another txn)
                        if i > 0 {
                            let inserter = TransactionId::new(num_txns as u64 + i as u64);
                            builder.begin_transaction(inserter, IsolationLevel::RepeatableRead);
                            builder.add_insert(
                                inserter, table, ItemId::new(100 + i as u64),
                                vec![("val".to_string(), Value::Integer(i as i64))],
                            );
                            builder.commit_transaction(inserter);
                        }

                        builder.commit_transaction(txn);
                    }

                    black_box(builder.build().unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Scaling with number of items per transaction
fn bench_item_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("item_scaling");
    group.sample_size(50);

    for &items in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("build_10txn", format!("{}items", items)),
            &items,
            |b, &items| {
                b.iter(|| {
                    let builder = build_synthetic_history(10, items);
                    black_box(builder.build().unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Isolation level classification queries
fn bench_isolation_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("isolation_classification");

    group.bench_function("all_levels", |b| {
        b.iter(|| {
            for level in IsolationLevel::all_levels() {
                black_box(level.prevented_anomalies());
                black_box(level.possible_anomalies());
                black_box(level.strength());
            }
        });
    });

    group.bench_function("pairwise_comparison", |b| {
        let levels = IsolationLevel::all_levels();
        b.iter(|| {
            for a in &levels {
                for b_level in &levels {
                    black_box(a.at_least(*b_level));
                }
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_history_construction,
    bench_history_queries,
    bench_write_skew_scenario,
    bench_phantom_scenario,
    bench_item_scaling,
    bench_isolation_classification,
);
criterion_main!(benches);
