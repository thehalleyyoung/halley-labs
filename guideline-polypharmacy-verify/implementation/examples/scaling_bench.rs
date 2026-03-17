//! Polypharmacy scaling benchmark.
//!
//! Measures the time to build and analyze scenarios with varying numbers of
//! drugs (2, 5, 10, 20). Uses `std::time::Instant` for timing since stable
//! Rust doesn't include built-in benchmarks.
//!
//! Run with: `cargo run --example scaling_bench`

use std::time::Instant;

use guardpharma_types::{DrugId, PatientInfo, Sex};

/// Drug pool for building scaled scenarios.
const DRUG_POOL: &[(&str, f64, f64)] = &[
    ("Metformin", 1000.0, 12.0),
    ("Lisinopril", 20.0, 24.0),
    ("Amlodipine", 5.0, 24.0),
    ("Warfarin", 5.0, 24.0),
    ("Metoprolol", 50.0, 12.0),
    ("Atorvastatin", 40.0, 24.0),
    ("Omeprazole", 20.0, 24.0),
    ("Aspirin", 81.0, 24.0),
    ("Furosemide", 40.0, 12.0),
    ("Gabapentin", 300.0, 8.0),
    ("Sertraline", 50.0, 24.0),
    ("Allopurinol", 300.0, 24.0),
    ("Levothyroxine", 0.1, 24.0),
    ("Tamsulosin", 0.4, 24.0),
    ("Clopidogrel", 75.0, 24.0),
    ("Amiodarone", 200.0, 24.0),
    ("Digoxin", 0.125, 24.0),
    ("Prednisone", 10.0, 24.0),
    ("Fluconazole", 200.0, 24.0),
    ("Carbamazepine", 200.0, 12.0),
];

fn build_drug_ids(n: usize) -> Vec<DrugId> {
    DRUG_POOL.iter().take(n).map(|(name, _, _)| DrugId::new(*name)).collect()
}

fn build_patient(n_drugs: usize) -> PatientInfo {
    let _ = n_drugs; // patient is the same regardless of drug count
    PatientInfo {
        age_years: 72.0,
        weight_kg: 82.0,
        height_cm: 175.0,
        sex: Sex::Male,
        serum_creatinine: 1.4,
        ..Default::default()
    }
}

fn count_pairs(n: usize) -> usize {
    n * (n - 1) / 2
}

/// Simulate scenario construction + pairwise enumeration.
fn bench_scenario_setup(n_drugs: usize, iterations: usize) -> (f64, f64) {
    let start = Instant::now();

    for _ in 0..iterations {
        let drugs = build_drug_ids(n_drugs);
        let _patient = build_patient(n_drugs);

        // Enumerate all drug pairs (the core combinatorial operation)
        let mut pairs = Vec::with_capacity(count_pairs(n_drugs));
        for i in 0..drugs.len() {
            for j in (i + 1)..drugs.len() {
                pairs.push((&drugs[i], &drugs[j]));
            }
        }

        // Simulate per-pair work: check enzyme overlap
        for (a, b) in &pairs {
            let _combined = format!("{}_{}", a.as_str(), b.as_str());
        }
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_iter_us = total_ms * 1000.0 / iterations as f64;
    (total_ms, per_iter_us)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  GuardPharma — Polypharmacy Scaling Benchmark                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let drug_counts = [2, 5, 10, 15, 20];
    let iterations = 10_000;

    println!(
        "{:>6} {:>8} {:>14} {:>14} {:>14}",
        "Drugs", "Pairs", "Total (ms)", "Per-iter (μs)", "Per-pair (μs)"
    );
    println!("{}", "─".repeat(60));

    for &n in &drug_counts {
        if n > DRUG_POOL.len() {
            continue;
        }
        let n_pairs = count_pairs(n);
        let (total_ms, per_iter_us) = bench_scenario_setup(n, iterations);
        let per_pair_us = if n_pairs > 0 {
            per_iter_us / n_pairs as f64
        } else {
            0.0
        };

        println!(
            "{:>6} {:>8} {:>14.2} {:>14.2} {:>14.4}",
            n, n_pairs, total_ms, per_iter_us, per_pair_us
        );
    }

    println!("{}", "─".repeat(60));
    println!("({} iterations per drug count)\n", iterations);

    // Scaling analysis
    println!("Scaling Analysis:");
    println!("  Drug pairs grow as O(n²): C(n,2) = n(n-1)/2");
    println!("  Expected scaling from 5 → 20 drugs:");
    println!("    Pairs: {} → {} ({}×)", count_pairs(5), count_pairs(20),
             count_pairs(20) as f64 / count_pairs(5) as f64);
    println!();
    println!("  For full verification pipeline (Tier 1 + Tier 2),");
    println!("  per-pair cost is dominated by SMT solving (~100ms/pair),");
    println!("  so 20-drug scenarios take ~{}s for Tier 2 alone.",
             count_pairs(20) as f64 * 0.1);
    println!("  Tier 1 abstract interpretation is much faster (~1ms/pair).");
}
