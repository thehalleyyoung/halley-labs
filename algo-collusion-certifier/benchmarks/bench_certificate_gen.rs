//! Benchmark: Certificate construction, verification, and Merkle tree.
//!
//! Times the following operations:
//!   1. Certificate construction (Layer 0) from pre-computed test results
//!   2. Certificate verification via the trusted proof checker
//!   3. Merkle evidence tree construction
//!
//! Run via: `cargo run --example bench_certificate_gen --release`

use std::time::Instant;

use certificate::{
    CertificateBuilder, ProofChecker, VerificationResult,
    merkle::{CertMerkleTree, MerkleEvidenceItem},
};
use game_theory::{CollusionPremium, NashEquilibrium};
use shared_types::{
    Cost, DemandSystem, GameConfig, MarketType, OracleAccessLevel, Price,
};
use stat_tests::{TestResult, TestType};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_game_config(num_players: usize) -> GameConfig {
    GameConfig::symmetric(
        MarketType::Bertrand,
        DemandSystem::Linear {
            max_quantity: 10.0,
            slope: 1.0,
        },
        num_players,
        0.95,
        Cost::new(2.0),
        100_000,
    )
}

fn make_test_results(count: usize, alpha: f64) -> Vec<TestResult> {
    (0..count)
        .map(|i| {
            let p_val = 0.001 + 0.001 * i as f64;
            TestResult::new(
                TestType::PriceCorrelation,
                &format!("sub_test_{i}"),
                3.0 + i as f64 * 0.5,
                p_val,
                alpha,
            )
        })
        .collect()
}

fn make_nash(num_players: usize) -> NashEquilibrium {
    NashEquilibrium::pure(vec![0; num_players], vec![1.0; num_players])
}

fn make_merkle_items(count: usize) -> Vec<MerkleEvidenceItem> {
    (0..count)
        .map(|i| {
            let data = format!("evidence_item_{i}");
            MerkleEvidenceItem::new(
                &format!("ref_{i}"),
                "trajectory_segment",
                data.as_bytes(),
            )
        })
        .collect()
}

// ── Benchmark result ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BenchResult {
    operation: String,
    param: String,
    elapsed_us: u128,
    success: bool,
}

impl BenchResult {
    fn csv_header() -> &'static str {
        "operation,param,elapsed_us,success"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{}",
            self.operation, self.param, self.elapsed_us, self.success
        )
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"operation":"{}","param":"{}","elapsed_us":{},"success":{}}}"#,
            self.operation, self.param, self.elapsed_us, self.success
        )
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       Certificate Generation Benchmark Suite            ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let alpha = 0.05;
    let num_players = 2;
    let game_config = make_game_config(num_players);
    let nash = make_nash(num_players);

    let test_counts = [3, 10, 50];
    let merkle_leaf_counts = [10, 100, 1_000, 10_000];

    let mut results = Vec::new();

    // ── 1. Certificate construction ─────────────────────────────────────
    println!("── Certificate Construction (Layer 0) ──────────────────────\n");

    for &n_tests in &test_counts {
        let test_results = make_test_results(n_tests, alpha);
        let cp = CollusionPremium::compute(5.0, 1.0);
        let hash = "abc123def456";

        let start = Instant::now();
        let cert = CertificateBuilder::build_layer0_certificate(
            &test_results,
            hash,
            num_players,
            100_000,
            &game_config,
            &nash,
            &cp,
            alpha,
        );
        let elapsed = start.elapsed().as_micros();

        let steps = cert.body.steps.len();
        println!(
            "  {n_tests:>3} tests → {steps} proof steps   [{elapsed} µs]"
        );

        results.push(BenchResult {
            operation: "cert_build_L0".into(),
            param: format!("{n_tests}_tests"),
            elapsed_us: elapsed,
            success: steps > 0,
        });
    }

    // ── 2. Certificate verification ─────────────────────────────────────
    println!("\n── Certificate Verification ───────────────────────────────\n");

    for &n_tests in &test_counts {
        let test_results = make_test_results(n_tests, alpha);
        let cp = CollusionPremium::compute(5.0, 1.0);
        let hash = "abc123def456";

        let cert = CertificateBuilder::build_layer0_certificate(
            &test_results,
            hash,
            num_players,
            100_000,
            &game_config,
            &nash,
            &cp,
            alpha,
        );

        let checker = ProofChecker::new().with_strict_mode(false);

        let start = Instant::now();
        let result = checker.check_certificate(&cert);
        let elapsed = start.elapsed().as_micros();

        let valid = result.is_valid();
        println!(
            "  {n_tests:>3} tests  valid={valid:<5}   [{elapsed} µs]"
        );

        results.push(BenchResult {
            operation: "cert_verify".into(),
            param: format!("{n_tests}_tests"),
            elapsed_us: elapsed,
            success: valid,
        });
    }

    // ── 3. Merkle tree construction ─────────────────────────────────────
    println!("\n── Merkle Tree Construction ───────────────────────────────\n");

    for &n_leaves in &merkle_leaf_counts {
        let items = make_merkle_items(n_leaves);

        let start = Instant::now();
        let tree = CertMerkleTree::build(&items);
        let elapsed = start.elapsed().as_micros();

        let has_root = tree.root_hash().is_some();
        println!(
            "  {n_leaves:>6} leaves  root={:<5}  [{elapsed} µs]",
            has_root
        );

        results.push(BenchResult {
            operation: "merkle_build".into(),
            param: format!("{n_leaves}_leaves"),
            elapsed_us: elapsed,
            success: has_root,
        });
    }

    // ── Output CSV ──────────────────────────────────────────────────────
    println!("\n── CSV Output ─────────────────────────────────────────────\n");
    println!("{}", BenchResult::csv_header());
    for r in &results {
        println!("{}", r.to_csv());
    }

    // ── Output JSON ─────────────────────────────────────────────────────
    println!("\n── JSON Output ────────────────────────────────────────────\n");
    let json_entries: Vec<String> = results.iter().map(|r| r.to_json()).collect();
    println!("[{}]", json_entries.join(",\n "));

    // ── Summary table ───────────────────────────────────────────────────
    println!("\n── Summary Table ──────────────────────────────────────────\n");
    println!(
        "{:<20} {:>16} {:>12} {:>8}",
        "Operation", "Param", "Time (µs)", "OK"
    );
    println!("{}", "-".repeat(60));
    for r in &results {
        println!(
            "{:<20} {:>16} {:>12} {:>8}",
            r.operation, r.param, r.elapsed_us, r.success
        );
    }

    println!("\n✓ Certificate benchmark complete.");
    Ok(())
}
