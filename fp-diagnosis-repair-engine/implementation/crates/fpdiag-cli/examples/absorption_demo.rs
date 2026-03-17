//! Demonstrates absorption diagnosis with Kahan summation repair.
//!
//! Run with: `cargo run --example absorption_demo`

use fpdiag_analysis::EagBuilder;
use fpdiag_diagnosis::DiagnosisEngine;
use fpdiag_repair::RepairSynthesizer;
use fpdiag_types::{
    expression::FpOp,
    precision::Precision,
    trace::{ExecutionTrace, TraceEvent},
};

/// Naive left-to-right summation.
fn naive_sum(values: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    for &v in values {
        total += v;
    }
    total
}

/// Kahan compensated summation (the standard repair for absorption).
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for &v in values {
        let y = v - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Penumbra: Absorption in Summation Demo                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create test data: one large value followed by many small ones
    let n = 100_000;
    let large = 1e16_f64;
    let mut values = vec![large];
    values.extend(std::iter::repeat(1.0_f64).take(n - 1));

    let true_sum = large + (n as f64 - 1.0);

    let naive = naive_sum(&values);
    let kahan = kahan_sum(&values);

    println!("Summation of 1 × 10^16 + {} × 1.0:", n - 1);
    println!("  True sum:  {:.0}", true_sum);
    println!(
        "  Naive sum: {:.0}  (error: {:.0})",
        naive,
        (naive - true_sum).abs()
    );
    println!(
        "  Kahan sum: {:.0}  (error: {:.0})",
        kahan,
        (kahan - true_sum).abs()
    );
    println!();

    // Build trace of the summation
    println!("Building EAG from summation trace ({} additions)...", n - 1);
    let mut trace = ExecutionTrace::new();

    // Trace a representative subset (first 50 additions)
    let mut acc = large;
    let trace_count = 50.min(n - 1);
    for i in 0..trace_count {
        let old_acc = acc;
        acc += 1.0;
        let shadow_acc = large + ((i + 1) as f64);
        trace.push(TraceEvent::Operation {
            seq: i as u64,
            op: FpOp::Add,
            inputs: vec![old_acc, 1.0],
            output: acc,
            shadow_output: shadow_acc,
            precision: Precision::Double,
            source: None,
            expr_node: None,
        });
    }
    trace.finalize();

    let mut builder = EagBuilder::with_defaults();
    builder.build_from_trace(&trace).expect("EAG build failed");
    let eag = builder.finish();

    println!("  EAG nodes: {}", eag.node_count());
    println!("  EAG edges: {}", eag.edge_count());
    println!();

    // Diagnose
    println!("Running diagnosis...");
    let engine = DiagnosisEngine::with_defaults();
    let report = engine.diagnose(&eag).expect("diagnosis failed");

    let absorption_count = report
        .diagnoses
        .iter()
        .filter(|d| {
            matches!(
                d.category,
                fpdiag_types::diagnosis::DiagnosisCategory::Absorption
            )
        })
        .count();

    println!("  Total diagnoses: {}", report.diagnoses.len());
    println!("  Absorption diagnoses: {}", absorption_count);
    println!();

    // Repair
    println!("Synthesizing repairs...");
    let synth = RepairSynthesizer::with_defaults();
    let repair = synth.synthesize(&eag, &report).unwrap_or_default();

    println!("  Repairs: {}", repair.applied_repairs.len());
    for c in &repair.applied_repairs {
        println!("  → {:?} at nodes {:?}", c.strategy, c.target_nodes);
    }

    println!();
    println!("Repair: Replace naive accumulation with Kahan compensated summation.");
    println!("Error reduction: O(nε) → O(ε), independent of n.");
}
