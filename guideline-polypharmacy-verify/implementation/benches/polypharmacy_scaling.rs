//! Polypharmacy scaling benchmark — standalone timing harness.
//!
//! Tests scenario setup and pairwise enumeration with 2, 5, 10, 20 drugs.
//! Uses std::time::Instant for measurement (stable Rust compatible).
//!
//! Build and run: `cargo run --example scaling_bench -p guardpharma-cli`
//! Or directly:  `cargo run --bin guardpharma -- benchmark`
//!
//! This file is a reference copy; the runnable version is in examples/scaling_bench.rs.

// This bench is registered as an example in crates/cli/Cargo.toml.
// See examples/scaling_bench.rs for the executable benchmark code.
//
// For Criterion-based benchmarks (optional, requires criterion dependency):
//
//   use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
//
//   fn bench_scenario_setup(c: &mut Criterion) {
//       let mut group = c.benchmark_group("scenario_setup");
//       for n in [2, 5, 10, 20] {
//           group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
//               b.iter(|| {
//                   // build DrugIds, enumerate pairs
//               });
//           });
//       }
//       group.finish();
//   }
//
//   criterion_group!(benches, bench_scenario_setup);
//   criterion_main!(benches);
