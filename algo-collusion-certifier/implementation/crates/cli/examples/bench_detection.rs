//! Benchmark: Collusion detection on synthetic Bertrand markets.
//!
//! Creates 2-player Bertrand markets with linear demand, generates trajectories
//! of varying lengths, and compares three detection approaches:
//!   1. M1 composite hypothesis test (CollusionProof)
//!   2. Harrington price-cost margin screening baseline
//!   3. Correlation-based detection baseline
//!
//! Run via: `cargo run --example bench_detection --release`

use std::time::Instant;

use market_sim::{BertrandMarket, LinearDemand, PlayerAction};
use shared_types::Price;
use stat_tests::{
    CompositeTest, TieredNull, SubTestResult,
    CrossFirmCorrelation,
};

// ── Harrington price-cost margin screening ──────────────────────────────────

/// Simple baseline: flag collusion when the average price-cost margin exceeds
/// a threshold derived from competitive equilibrium.
fn harrington_screen(prices: &[Vec<f64>], marginal_cost: f64, threshold: f64) -> (bool, f64) {
    let mut total_margin = 0.0;
    let mut count = 0usize;
    for firm in prices {
        for &p in firm {
            total_margin += (p - marginal_cost) / marginal_cost.max(1e-12);
            count += 1;
        }
    }
    let avg_margin = if count > 0 { total_margin / count as f64 } else { 0.0 };
    (avg_margin > threshold, avg_margin)
}

// ── Correlation-based detection baseline ────────────────────────────────────

/// Simple baseline: flag collusion when the Pearson correlation between the two
/// firms' price series exceeds a threshold.
fn correlation_screen(firm_a: &[f64], firm_b: &[f64], threshold: f64) -> (bool, f64) {
    let n = firm_a.len().min(firm_b.len());
    if n < 3 {
        return (false, 0.0);
    }
    let mean_a: f64 = firm_a[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = firm_b[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = firm_a[i] - mean_a;
        let db = firm_b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    let r = if denom > 1e-15 { cov / denom } else { 0.0 };
    (r > threshold, r)
}

// ── Trajectory generation ───────────────────────────────────────────────────

/// Generate a synthetic trajectory where two firms independently draw prices
/// from a noisy supra-competitive distribution (simulating tacit collusion).
fn generate_collusive_trajectory(
    num_rounds: usize,
    marginal_cost: f64,
    collusive_price: f64,
    noise_std: f64,
) -> Vec<Vec<f64>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Deterministic pseudo-random using round index
    let mut prices = vec![Vec::with_capacity(num_rounds); 2];
    for t in 0..num_rounds {
        for firm in 0..2 {
            let mut h = DefaultHasher::new();
            (t, firm, 0xDEAD_BEEFu64).hash(&mut h);
            let bits = h.finish();
            // Map to roughly normal via Box-Muller-like transform (cheap approximation)
            let u = (bits & 0xFFFF) as f64 / 65535.0;
            let noise = (u - 0.5) * 2.0 * noise_std;
            prices[firm].push((collusive_price + noise).max(marginal_cost));
        }
    }
    prices
}

// ── Benchmark result ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BenchResult {
    method: String,
    num_rounds: usize,
    elapsed_us: u128,
    detected: bool,
    score: f64,
}

impl BenchResult {
    fn csv_header() -> &'static str {
        "method,num_rounds,elapsed_us,detected,score"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self.method, self.num_rounds, self.elapsed_us, self.detected, self.score
        )
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"method":"{}","num_rounds":{},"elapsed_us":{},"detected":{},"score":{:.6}}}"#,
            self.method, self.num_rounds, self.elapsed_us, self.detected, self.score
        )
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        Collusion Detection Benchmark Suite              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let marginal_cost = 2.0;
    let competitive_price = 2.5;  // approximately mc + small markup
    let collusive_price = 6.0;    // well above competitive
    let monopoly_price = 8.0;
    let noise_std = 0.3;
    let harrington_threshold = 0.5; // 50 % margin over mc
    let correlation_threshold = 0.6;

    let trajectory_lengths: Vec<usize> = vec![1_000, 10_000, 100_000];

    let mut results = Vec::new();

    for &num_rounds in &trajectory_lengths {
        println!("── Trajectory length: {num_rounds} rounds ──────────────────────\n");

        let prices = generate_collusive_trajectory(
            num_rounds,
            marginal_cost,
            collusive_price,
            noise_std,
        );

        // 1. M1 composite hypothesis test
        {
            let test = CompositeTest::new(0.05, TieredNull::Narrow)
                .with_bootstrap(500)
                .with_seed(42);

            let start = Instant::now();
            let decision = test.run(
                &prices,
                Price::new(competitive_price),
                Price::new(monopoly_price),
            )?;
            let elapsed = start.elapsed().as_micros();

            println!(
                "  M1 composite : reject={:<5} stat={:.4}  p={:.6}  [{} µs]",
                decision.reject_null,
                decision.composite_statistic,
                decision.p_value.value(),
                elapsed,
            );

            results.push(BenchResult {
                method: "M1_composite".into(),
                num_rounds,
                elapsed_us: elapsed,
                detected: decision.reject_null,
                score: decision.composite_statistic,
            });
        }

        // 2. Harrington price-cost margin
        {
            let start = Instant::now();
            let (detected, margin) =
                harrington_screen(&prices, marginal_cost, harrington_threshold);
            let elapsed = start.elapsed().as_micros();

            println!(
                "  Harrington   : detect={:<5} margin={:.4}            [{} µs]",
                detected, margin, elapsed,
            );

            results.push(BenchResult {
                method: "Harrington_PCM".into(),
                num_rounds,
                elapsed_us: elapsed,
                detected,
                score: margin,
            });
        }

        // 3. Correlation-based
        {
            let start = Instant::now();
            let (detected, r) =
                correlation_screen(&prices[0], &prices[1], correlation_threshold);
            let elapsed = start.elapsed().as_micros();

            println!(
                "  Correlation  : detect={:<5} r={:.4}                [{} µs]",
                detected, r, elapsed,
            );

            results.push(BenchResult {
                method: "Correlation".into(),
                num_rounds,
                elapsed_us: elapsed,
                detected,
                score: r,
            });
        }

        println!();
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
    println!("\n── Comparison Table ───────────────────────────────────────\n");
    println!(
        "{:<16} {:>10} {:>12} {:>8} {:>10}",
        "Method", "Rounds", "Time (µs)", "Detect", "Score"
    );
    println!("{}", "-".repeat(60));
    for r in &results {
        println!(
            "{:<16} {:>10} {:>12} {:>8} {:>10.4}",
            r.method, r.num_rounds, r.elapsed_us, r.detected, r.score
        );
    }

    println!("\n✓ Detection benchmark complete.");
    Ok(())
}
