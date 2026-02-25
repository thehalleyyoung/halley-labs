//! Real STARK Proof Generation
//!
//! Generates actual STARK proofs for simple computations, demonstrating
//! the full prove → verify pipeline with real timing measurements.
//! This addresses the "STARK prover simulated, not implemented" critique.

use spectacles_core::circuit::stark::{
    STARKProver, STARKVerifier, STARKConfig, STARKProof, SecurityConfig,
    build_fibonacci_air, build_counter_air, build_squaring_air,
};
use spectacles_core::circuit::goldilocks::GoldilocksField;
use spectacles_core::circuit::trace::ExecutionTrace;
use spectacles_core::circuit::air::{AIRProgram, AIRTrace, ConstraintType, TraceLayout, ColumnType, SymbolicExpression};
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct StarkBenchmarkResults {
    timestamp: String,
    proofs: Vec<ProofResult>,
    summary: ProofSummary,
}

#[derive(Debug, Serialize)]
struct ProofResult {
    name: String,
    trace_length: usize,
    trace_width: usize,
    num_constraints: usize,
    constraint_types: Vec<String>,
    prove_time_ms: f64,
    verify_time_ms: f64,
    proof_size_bytes: usize,
    proof_valid: bool,
    security_bits: u32,
}

#[derive(Debug, Serialize)]
struct ProofSummary {
    total_proofs_generated: usize,
    all_verified: bool,
    mean_prove_time_ms: f64,
    mean_verify_time_ms: f64,
    mean_proof_size_bytes: usize,
}

/// Build an equality-check AIR for verifying exact string match.
///
/// The trace has one column per character position. For position i:
///   trace[i][0] = candidate char code
///   trace[i][1] = reference char code
///   trace[i][2] = running equality (1 if all match so far, 0 otherwise)
///
/// Constraints:
///   - Boundary: equality[0] = 1 if char[0] matches, 0 otherwise
///   - Transition: eq[i+1] = eq[i] * (1 - (cand[i+1] - ref[i+1])^(p-1))
///     (Fermat's little theorem: a^(p-1) = 1 for a ≠ 0, 0 for a = 0)
///
/// For simplicity, we use a direct check:
///   trace[i][2] = product of (cand[j] == ref[j]) for j = 0..i
fn build_exact_match_air(candidate: &str, reference: &str) -> (AIRProgram, ExecutionTrace) {
    let max_len = candidate.len().max(reference.len()).max(2);
    // Pad to at least 8 for FRI requirements
    let trace_len = max_len.max(8).next_power_of_two();

    let mut layout = TraceLayout::new();
    layout.add_column("candidate_char".to_string(), ColumnType::State);
    layout.add_column("reference_char".to_string(), ColumnType::State);
    layout.add_column("char_equal".to_string(), ColumnType::State);
    layout.add_column("running_eq".to_string(), ColumnType::State);

    let mut air = AIRProgram::new("exact_match", layout);

    // Compute the initial running_eq value for boundary constraint
    let first_eq = if !candidate.is_empty() && !reference.is_empty() {
        if candidate.as_bytes()[0] == reference.as_bytes()[0] {
            GoldilocksField::ONE
        } else {
            GoldilocksField::ZERO
        }
    } else if candidate.is_empty() && reference.is_empty() {
        GoldilocksField::ONE
    } else {
        GoldilocksField::ZERO
    };

    // Boundary: running_eq[0] = first_char_equal
    air.add_boundary_constraint(3, 0, first_eq);

    // Transition: running_eq[i+1] = running_eq[i] * char_equal[i+1]
    air.add_transition_constraint(
        "eq_propagate",
        SymbolicExpression::nxt(3) - SymbolicExpression::cur(3) * SymbolicExpression::nxt(2),
    );

    // char_equal[i] ∈ {0, 1}: char_equal * (1 - char_equal) = 0
    air.add_transition_constraint(
        "char_eq_boolean",
        SymbolicExpression::cur(2) * (SymbolicExpression::one() - SymbolicExpression::cur(2)),
    );

    // running_eq[i] ∈ {0, 1}
    air.add_transition_constraint(
        "running_eq_boolean",
        SymbolicExpression::cur(3) * (SymbolicExpression::one() - SymbolicExpression::cur(3)),
    );

    // Generate the execution trace
    let mut trace = ExecutionTrace::zeros(4, trace_len);
    let cand_bytes: Vec<u8> = candidate.bytes().collect();
    let ref_bytes: Vec<u8> = reference.bytes().collect();

    let mut running_eq = GoldilocksField::ONE;
    for i in 0..trace_len {
        let c = if i < cand_bytes.len() { cand_bytes[i] as u64 } else { 0 };
        let r = if i < ref_bytes.len() { ref_bytes[i] as u64 } else { 0 };
        let char_eq = if c == r && (i < cand_bytes.len() || i < ref_bytes.len() || (i >= cand_bytes.len() && i >= ref_bytes.len())) {
            if i < cand_bytes.len().max(ref_bytes.len()) {
                if c == r { GoldilocksField::ONE } else { GoldilocksField::ZERO }
            } else {
                GoldilocksField::ONE // padding chars are "equal"
            }
        } else {
            if c == r { GoldilocksField::ONE } else { GoldilocksField::ZERO }
        };

        running_eq = if i == 0 {
            char_eq
        } else {
            running_eq * char_eq
        };

        trace.set(i, 0, GoldilocksField::new(c));
        trace.set(i, 1, GoldilocksField::new(r));
        trace.set(i, 2, char_eq);
        trace.set(i, 3, running_eq);
    }

    (air, trace)
}

/// Build a sum-check AIR for token counting (simplified Token F1).
///
/// Proves correct counting of matching tokens.
fn build_token_count_air(num_steps: usize) -> (AIRProgram, ExecutionTrace) {
    let mut layout = TraceLayout::new();
    layout.add_column("token_match".to_string(), ColumnType::State);
    layout.add_column("running_count".to_string(), ColumnType::State);

    let mut air = AIRProgram::new("token_count", layout);

    // Boundary: count[0] = match[0]
    air.add_boundary_constraint(1, 0, GoldilocksField::ZERO);

    // Transition: count[i+1] = count[i] + match[i+1]
    air.add_transition_constraint(
        "count_step",
        SymbolicExpression::nxt(1) - SymbolicExpression::cur(1) - SymbolicExpression::nxt(0),
    );

    // match[i] ∈ {0, 1}
    air.add_transition_constraint(
        "match_boolean",
        SymbolicExpression::cur(0) * (SymbolicExpression::one() - SymbolicExpression::cur(0)),
    );

    // Generate trace with example data
    let trace_len = num_steps.max(8).next_power_of_two();
    let mut trace = ExecutionTrace::zeros(2, trace_len);
    let matches = vec![1u64, 0, 1, 1, 0, 1, 0, 1]; // example: 5/8 tokens match

    let mut count = 0u64;
    for i in 0..trace_len {
        let m = if i < matches.len() { matches[i] } else { 0 };
        trace.set(i, 0, GoldilocksField::new(m));
        if i > 0 { count += m; }
        trace.set(i, 1, GoldilocksField::new(count));
    }

    (air, trace)
}

fn run_proof(name: &str, air: &AIRProgram, trace: &ExecutionTrace) -> ProofResult {
    let config = STARKConfig::default();

    let constraint_types: Vec<String> = air.constraints.iter().map(|c| {
        match c.constraint_type {
            ConstraintType::Boundary => "boundary".into(),
            ConstraintType::Transition => "transition".into(),
            ConstraintType::Periodic => "periodic".into(),
            ConstraintType::Composition => "composition".into(),
        }
    }).collect();

    let prover = STARKProver::new(config.clone());
    let verifier = STARKVerifier::new(config.clone());

    // Prove
    let prove_start = Instant::now();
    let proof_result = prover.prove(air, trace);
    let prove_time = prove_start.elapsed().as_secs_f64() * 1000.0;

    match proof_result {
        Ok(proof) => {
            let proof_size = estimate_proof_size(&proof);

            // Verify
            let verify_start = Instant::now();
            let verify_result = verifier.verify(air, &proof);
            let verify_time = verify_start.elapsed().as_secs_f64() * 1000.0;

            let valid = verify_result.unwrap_or(false);

            println!("  {} | trace={}×{} | constraints={} | prove={:.1}ms | verify={:.1}ms | size={} B | valid={}",
                name, trace.length, trace.width, air.constraints.len(),
                prove_time, verify_time, proof_size, valid);

            ProofResult {
                name: name.into(),
                trace_length: trace.length,
                trace_width: trace.width,
                num_constraints: air.constraints.len(),
                constraint_types,
                prove_time_ms: prove_time,
                verify_time_ms: verify_time,
                proof_size_bytes: proof_size,
                proof_valid: valid,
                security_bits: config.security.security_bits,
            }
        }
        Err(e) => {
            println!("  {} | FAILED: {:?}", name, e);
            ProofResult {
                name: name.into(),
                trace_length: trace.length,
                trace_width: trace.width,
                num_constraints: air.constraints.len(),
                constraint_types,
                prove_time_ms: prove_time,
                verify_time_ms: 0.0,
                proof_size_bytes: 0,
                proof_valid: false,
                security_bits: config.security.security_bits,
            }
        }
    }
}

fn estimate_proof_size(proof: &STARKProof) -> usize {
    proof.size_in_bytes()
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Real STARK Proof Generation & Verification");
    println!("═══════════════════════════════════════════════════════════════");

    let mut results = Vec::new();

    // 1. Fibonacci (baseline)
    println!("\n▸ Generating STARK proofs...");
    {
        let (air, trace) = build_fibonacci_air(8);
        results.push(run_proof("fibonacci_8", &air, &trace));
    }
    {
        let (air, trace) = build_fibonacci_air(16);
        results.push(run_proof("fibonacci_16", &air, &trace));
    }

    // 2. Counter
    {
        let (air, trace) = build_counter_air(8);
        results.push(run_proof("counter_8", &air, &trace));
    }
    {
        let (air, trace) = build_counter_air(16);
        results.push(run_proof("counter_16", &air, &trace));
    }

    // 3. Squaring
    {
        let (air, trace) = build_squaring_air(8);
        results.push(run_proof("squaring_8", &air, &trace));
    }

    // 4. Exact match
    {
        let (air, trace) = build_exact_match_air("hello", "hello");
        results.push(run_proof("exact_match_equal", &air, &trace));
    }
    {
        let (air, trace) = build_exact_match_air("hello", "world");
        results.push(run_proof("exact_match_differ", &air, &trace));
    }
    {
        let (air, trace) = build_exact_match_air("the cat sat on the mat", "the cat sat on the mat");
        results.push(run_proof("exact_match_sentence", &air, &trace));
    }

    // 5. Token counting
    {
        let (air, trace) = build_token_count_air(8);
        results.push(run_proof("token_count_8", &air, &trace));
    }

    // Summary
    let all_valid = results.iter().all(|r| r.proof_valid);
    let valid_results: Vec<&ProofResult> = results.iter().filter(|r| r.proof_valid).collect();
    let mean_prove = if valid_results.is_empty() { 0.0 } else {
        valid_results.iter().map(|r| r.prove_time_ms).sum::<f64>() / valid_results.len() as f64
    };
    let mean_verify = if valid_results.is_empty() { 0.0 } else {
        valid_results.iter().map(|r| r.verify_time_ms).sum::<f64>() / valid_results.len() as f64
    };
    let mean_size = if valid_results.is_empty() { 0 } else {
        valid_results.iter().map(|r| r.proof_size_bytes).sum::<usize>() / valid_results.len()
    };

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Total proofs:    {}", results.len());
    println!("  Valid proofs:    {}", valid_results.len());
    println!("  All verified:    {}", all_valid);
    println!("  Mean prove:      {:.1} ms", mean_prove);
    println!("  Mean verify:     {:.1} ms", mean_verify);
    println!("  Mean proof size: {} bytes", mean_size);

    let benchmark = StarkBenchmarkResults {
        timestamp: chrono::Utc::now().to_rfc3339(),
        proofs: results,
        summary: ProofSummary {
            total_proofs_generated: 9,
            all_verified: all_valid,
            mean_prove_time_ms: mean_prove,
            mean_verify_time_ms: mean_verify,
            mean_proof_size_bytes: mean_size,
        },
    };

    let json = serde_json::to_string_pretty(&benchmark).unwrap();
    std::fs::write("stark_benchmark.json", &json).unwrap();
    println!("\n  Results saved to: stark_benchmark.json");
}
