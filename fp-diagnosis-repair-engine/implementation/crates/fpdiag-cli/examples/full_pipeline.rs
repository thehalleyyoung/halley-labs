//! Demonstrates the full Penumbra pipeline on an ill-conditioned problem.
//!
//! Run with: `cargo run --example full_pipeline`

use fpdiag_analysis::EagBuilder;
use fpdiag_diagnosis::DiagnosisEngine;
use fpdiag_repair::RepairSynthesizer;
use fpdiag_report::ReportGenerator;
use fpdiag_types::{
    config::OutputFormat,
    expression::FpOp,
    precision::Precision,
    trace::{ExecutionTrace, TraceEvent},
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Penumbra: Full Pipeline Demo                              ║");
    println!("║   trace → EAG → diagnose → repair → report                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Build a complex trace: mixed error patterns
    let mut trace = ExecutionTrace::new();

    // Pattern 1: Catastrophic cancellation — (a - b) where a ≈ b
    let a = 1.0000000000001_f64;
    let b = 1.0_f64;
    trace.push(TraceEvent::Operation {
        seq: 0,
        op: FpOp::Sub,
        inputs: vec![a, b],
        output: a - b,
        shadow_output: 1e-13,
        precision: Precision::Double,
        source: None,
        expr_node: None,
    });

    // Pattern 2: Amplified rounding — multiply cancellation result by large number
    let scale = 1e10_f64;
    let cancel_result = a - b;
    trace.push(TraceEvent::Operation {
        seq: 1,
        op: FpOp::Mul,
        inputs: vec![cancel_result, scale],
        output: cancel_result * scale,
        shadow_output: 1e-13 * scale,
        precision: Precision::Double,
        source: None,
        expr_node: None,
    });

    // Pattern 3: Absorption — add small result to large accumulator
    let acc = 1e16_f64;
    let small = cancel_result * scale;
    trace.push(TraceEvent::Operation {
        seq: 2,
        op: FpOp::Add,
        inputs: vec![acc, small],
        output: acc + small,
        shadow_output: acc + 1e-13 * scale,
        precision: Precision::Double,
        source: None,
        expr_node: None,
    });

    // Pattern 4: exp(x) - 1 for small x (should use expm1)
    let x_small = 1e-14_f64;
    let exp_x = x_small.exp();
    trace.push(TraceEvent::Operation {
        seq: 3,
        op: FpOp::Exp,
        inputs: vec![x_small],
        output: exp_x,
        shadow_output: 1.0 + x_small + x_small * x_small / 2.0,
        precision: Precision::Double,
        source: None,
        expr_node: None,
    });

    trace.push(TraceEvent::Operation {
        seq: 4,
        op: FpOp::Sub,
        inputs: vec![exp_x, 1.0],
        output: exp_x - 1.0,
        shadow_output: x_small,
        precision: Precision::Double,
        source: None,
        expr_node: None,
    });

    trace.finalize();

    // === Stage 1: Build EAG ===
    println!("Stage 1: Building Error Amplification Graph (EAG)");
    let mut builder = EagBuilder::with_defaults();
    builder.build_from_trace(&trace).expect("EAG build failed");
    let eag = builder.finish();

    println!("  Nodes: {}", eag.node_count());
    println!("  Edges: {}", eag.edge_count());
    println!();

    // === Stage 2: Diagnose ===
    println!("Stage 2: Taxonomic Diagnosis");
    let engine = DiagnosisEngine::with_defaults();
    let diag_report = engine.diagnose(&eag).expect("diagnosis failed");

    println!("  Diagnoses: {}", diag_report.diagnoses.len());
    for diag in &diag_report.diagnoses {
        println!(
            "  [{:?}] Node {} — {:?} (conf={:.2})",
            diag.severity, diag.node_id, diag.category, diag.confidence
        );
    }
    println!();

    // === Stage 3: Repair ===
    println!("Stage 3: Repair Synthesis (T4-optimal greedy ordering)");
    let synth = RepairSynthesizer::with_defaults();
    let repair_result = synth.synthesize(&eag, &diag_report).unwrap_or_default();

    println!(
        "  Repair candidates: {}",
        repair_result.applied_repairs.len()
    );
    for (i, c) in repair_result.applied_repairs.iter().enumerate() {
        println!(
            "  [{}] {:?} → nodes {:?} (est. {:.1}× improvement)",
            i + 1,
            c.strategy,
            c.target_nodes,
            c.estimated_reduction
        );
    }
    println!();

    // === Stage 4: Report ===
    println!("Stage 4: Report Generation");
    let gen = ReportGenerator::new(OutputFormat::Human, true);
    let repair = fpdiag_types::repair::RepairResult::new();
    match gen.generate(&eag, &diag_report, &repair) {
        Ok(formatted) => {
            println!("{}", formatted.content);
        }
        Err(e) => println!("  Report generation error: {}", e),
    }

    println!();
    println!(
        "Pipeline complete. Penumbra identified {} error sources",
        diag_report.diagnoses.len()
    );
    println!(
        "and generated {} targeted repair candidates.",
        repair_result.applied_repairs.len()
    );
}
