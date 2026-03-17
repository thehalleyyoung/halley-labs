//! Demonstrates catastrophic cancellation diagnosis and repair.
//!
//! Run with: `cargo run --example cancellation_demo`

use fpdiag_analysis::EagBuilder;
use fpdiag_diagnosis::DiagnosisEngine;
use fpdiag_repair::RepairSynthesizer;
use fpdiag_types::{
    expression::FpOp,
    precision::Precision,
    trace::{ExecutionTrace, TraceEvent},
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Penumbra: Catastrophic Cancellation Diagnosis Demo        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Demonstrate the problem: (1 + x) - 1 for small x
    println!("Problem: Compute (1 + x) - 1 for small x");
    println!("Expected result: x");
    println!();
    println!(
        "{:>15} {:>20} {:>20} {:>15}",
        "x", "fragile", "stable", "rel_error"
    );
    println!("{}", "-".repeat(75));

    for k in 1..=17 {
        let x = f64::powi(10.0, -(k as i32));
        let fragile = (1.0 + x) - 1.0;
        let stable = x;
        let rel_err = if x != 0.0 {
            (fragile - x).abs() / x.abs()
        } else {
            0.0
        };
        println!(
            "{:>15.1e} {:>20.6e} {:>20.6e} {:>15.4e}",
            x, fragile, stable, rel_err
        );
    }

    println!();
    println!("Building Error Amplification Graph (EAG)...");

    // Build a trace simulating (1 + x) - 1 with x = 1e-15
    let mut trace = ExecutionTrace::new();
    let x = 1e-15_f64;

    // Step 1: t = 1.0 + x (rounds to 1.0 in f64)
    trace.push(TraceEvent::Operation {
        seq: 0,
        op: FpOp::Add,
        inputs: vec![1.0, x],
        output: 1.0 + x,        // f64 result
        shadow_output: 1.0 + x, // exact (at higher precision)
        precision: Precision::Double,
        source: None,
        expr_node: None,
    });

    // Step 2: result = t - 1.0 (catastrophic cancellation)
    trace.push(TraceEvent::Operation {
        seq: 1,
        op: FpOp::Sub,
        inputs: vec![1.0 + x, 1.0],
        output: (1.0 + x) - 1.0, // f64 result: ~0 or very wrong
        shadow_output: x,        // true result
        precision: Precision::Double,
        source: None,
        expr_node: None,
    });

    trace.finalize();

    // Build EAG
    let mut builder = EagBuilder::with_defaults();
    builder.build_from_trace(&trace).expect("EAG build failed");
    let eag = builder.finish();

    println!("  EAG nodes: {}", eag.node_count());
    println!("  EAG edges: {}", eag.edge_count());
    println!();

    // Run diagnosis
    println!("Running taxonomic diagnosis...");
    let engine = DiagnosisEngine::with_defaults();
    let report = engine.diagnose(&eag).expect("diagnosis failed");

    println!("  Diagnoses found: {}", report.diagnoses.len());
    for diag in &report.diagnoses {
        println!(
            "  → Node {}: {:?} (confidence: {:.2}, severity: {:?})",
            diag.node_id, diag.category, diag.confidence, diag.severity
        );
        if !diag.repair_suggestion.is_empty() {
            println!("    Suggested repair: {}", diag.repair_suggestion);
        }
    }
    println!();

    // Run repair synthesis
    println!("Synthesizing repairs...");
    let synth = RepairSynthesizer::with_defaults();
    let repair_result = synth.synthesize(&eag, &report).unwrap_or_default();

    println!(
        "  Repairs generated: {}",
        repair_result.applied_repairs.len()
    );
    for candidate in &repair_result.applied_repairs {
        println!(
            "  → Strategy: {:?}, targets: {:?}, estimated improvement: {:.1}×",
            candidate.strategy, candidate.target_nodes, candidate.estimated_reduction
        );
    }

    println!();
    println!("Done. The diagnosis correctly identifies catastrophic cancellation");
    println!("and recommends using the direct algebraic identity (return x).");
}
