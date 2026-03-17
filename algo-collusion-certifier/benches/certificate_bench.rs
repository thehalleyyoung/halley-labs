//! Criterion microbenchmark for certificate generation and verification.
//!
//! To run:  `cargo bench --bench certificate_bench`
//!
//! This is a placeholder. To enable, add the following to the workspace
//! root Cargo.toml:
//!
//! ```toml
//! [[bench]]
//! name = "certificate_bench"
//! harness = false
//! ```
//!
//! And add `criterion = "0.5"` to [workspace.dependencies].

// Uncomment when criterion is added as a dependency:
//
// use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
// use certificate::{CertificateBuilder, ProofChecker, merkle::{CertMerkleTree, MerkleEvidenceItem}};
// use game_theory::{CollusionPremium, NashEquilibrium};
// use shared_types::{Cost, DemandSystem, GameConfig, MarketType};
// use stat_tests::{TestResult, TestType};
//
// fn bench_build_and_verify(c: &mut Criterion) {
//     let mut group = c.benchmark_group("certificate_cycle");
//     let alpha = 0.05;
//     let gc = GameConfig::symmetric(
//         MarketType::Bertrand,
//         DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
//         2, 0.95, Cost::new(2.0), 100_000,
//     );
//     let nash = NashEquilibrium::pure(vec![0, 0], vec![1.0, 1.0]);
//     let cp = CollusionPremium::compute(5.0, 1.0);
//
//     for &n in &[3usize, 10, 50] {
//         let tests: Vec<TestResult> = (0..n)
//             .map(|i| TestResult::new(TestType::PriceCorrelation, &format!("t{i}"), 3.0, 0.001, alpha))
//             .collect();
//
//         group.bench_with_input(BenchmarkId::new("build_L0", n), &tests, |b, tests| {
//             b.iter(|| {
//                 CertificateBuilder::build_layer0_certificate(
//                     tests, "hash", 2, 100_000, &gc, &nash, &cp, alpha,
//                 )
//             });
//         });
//
//         let cert = CertificateBuilder::build_layer0_certificate(
//             &tests, "hash", 2, 100_000, &gc, &nash, &cp, alpha,
//         );
//         let checker = ProofChecker::new().with_strict_mode(false);
//
//         group.bench_with_input(BenchmarkId::new("verify", n), &cert, |b, cert| {
//             b.iter(|| checker.check_certificate(cert));
//         });
//     }
//     group.finish();
// }
//
// fn bench_merkle(c: &mut Criterion) {
//     let mut group = c.benchmark_group("merkle_tree");
//     for &n in &[10usize, 100, 1_000, 10_000] {
//         let items: Vec<MerkleEvidenceItem> = (0..n)
//             .map(|i| MerkleEvidenceItem::new(&format!("r{i}"), "seg", format!("d{i}").as_bytes()))
//             .collect();
//         group.bench_with_input(BenchmarkId::new("build", n), &items, |b, items| {
//             b.iter(|| CertMerkleTree::build(items));
//         });
//     }
//     group.finish();
// }
//
// criterion_group!(benches, bench_build_and_verify, bench_merkle);
// criterion_main!(benches);

fn main() {
    eprintln!(
        "certificate_bench: This is a Criterion placeholder. \
         See benches/README.md for setup instructions."
    );
}
