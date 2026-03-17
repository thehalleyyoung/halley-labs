//! Criterion microbenchmark for collusion detection.
//!
//! To run:  `cargo bench --bench detection_bench`
//!
//! This is a placeholder. To enable, add the following to the workspace
//! root Cargo.toml:
//!
//! ```toml
//! [[bench]]
//! name = "detection_bench"
//! harness = false
//! ```
//!
//! And add `criterion = "0.5"` to [workspace.dependencies].

// Uncomment when criterion is added as a dependency:
//
// use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
// use shared_types::Price;
// use stat_tests::{CompositeTest, TieredNull};
//
// fn bench_composite_test(c: &mut Criterion) {
//     let mut group = c.benchmark_group("composite_test");
//     let mc = 2.0;
//     let comp_price = Price::new(2.5);
//     let mono_price = Price::new(8.0);
//
//     for &n in &[1_000usize, 10_000, 100_000] {
//         let prices: Vec<Vec<f64>> = (0..2)
//             .map(|_| (0..n).map(|t| 5.0 + (t as f64 * 0.0001).sin() * 0.3).collect())
//             .collect();
//
//         group.bench_with_input(BenchmarkId::new("M1", n), &prices, |b, prices| {
//             let test = CompositeTest::new(0.05, TieredNull::Narrow)
//                 .with_bootstrap(200)
//                 .with_seed(42);
//             b.iter(|| test.run(prices, comp_price, mono_price));
//         });
//     }
//     group.finish();
// }
//
// criterion_group!(benches, bench_composite_test);
// criterion_main!(benches);

fn main() {
    eprintln!(
        "detection_bench: This is a Criterion placeholder. \
         See benches/README.md for setup instructions."
    );
}
