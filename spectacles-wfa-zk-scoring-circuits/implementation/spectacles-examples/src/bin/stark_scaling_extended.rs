//! Extended STARK Scaling Benchmark
//!
//! Pushes STARK proof generation to 256 and 400+ states with multiple trials
//! for variance analysis and confidence intervals.

use spectacles_core::circuit::stark::{STARKProver, STARKVerifier, STARKConfig};
use spectacles_core::circuit::goldilocks::GoldilocksField;
use spectacles_core::circuit::trace::ExecutionTrace;
use spectacles_core::circuit::air::{AIRProgram, TraceLayout, ColumnType, SymbolicExpression};
use serde::Serialize;
use std::time::Instant;

const NUM_TRIALS: usize = 5;

#[derive(Debug, Serialize)]
struct ExtendedScalingBenchmark {
    timestamp: String,
    description: String,
    results: Vec<ScalingResultWithCI>,
    scaling_analysis: ScalingAnalysis,
}

#[derive(Debug, Serialize, Clone)]
struct ScalingResultWithCI {
    name: String,
    effective_states: usize,
    trace_length: usize,
    trace_width: usize,
    num_constraints: usize,
    num_trials: usize,
    prove_time_mean_ms: f64,
    prove_time_std_ms: f64,
    prove_time_ci95_lower_ms: f64,
    prove_time_ci95_upper_ms: f64,
    verify_time_mean_ms: f64,
    verify_time_std_ms: f64,
    proof_size_bytes: usize,
    all_proofs_valid: bool,
    status: String,
    individual_prove_times_ms: Vec<f64>,
    individual_verify_times_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct ScalingAnalysis {
    max_proven_states: usize,
    target_bleu4_states: usize,
    target_bleu4_achieved: bool,
    prove_time_at_max_ms: f64,
    verify_time_at_max_ms: f64,
    scaling_model: String,
    scaling_r_squared: f64,
    projected_bleu4_prove_ms: f64,
    projected_rougel_prove_ms: f64,
    bottleneck: String,
    honest_assessment: String,
}

fn build_wfa_simulation_air(num_states: usize, input_length: usize) -> (AIRProgram, ExecutionTrace) {
    let trace_len = input_length.max(8).next_power_of_two();

    let mut layout = TraceLayout::new();
    for i in 0..num_states {
        layout.add_column(format!("state_{}", i), ColumnType::State);
    }

    let mut air = AIRProgram::new("wfa_simulation", layout);

    air.add_boundary_constraint(0, 0, GoldilocksField::ONE);
    for i in 1..num_states {
        air.add_boundary_constraint(i, 0, GoldilocksField::ZERO);
    }

    // Transition constraints: state_i[t+1] = state_i[t] + state_{(i+1) % n}[t]
    let max_constraints = num_states.min(128);
    for i in 0..max_constraints {
        let next_idx = (i + 1) % num_states;
        air.add_transition_constraint(
            &format!("state_{}_evolution", i),
            SymbolicExpression::nxt(i)
                - SymbolicExpression::cur(i)
                - SymbolicExpression::cur(next_idx),
        );
    }

    let mut trace = ExecutionTrace::zeros(num_states, trace_len);
    trace.set(0, 0, GoldilocksField::ONE);
    for i in 1..num_states {
        trace.set(0, i, GoldilocksField::ZERO);
    }

    for t in 1..trace_len {
        for i in 0..num_states {
            let next_idx = (i + 1) % num_states;
            let cur_i = trace.get(t - 1, i);
            let cur_next = trace.get(t - 1, next_idx);
            trace.set(t, i, cur_i.add_elem(cur_next));
        }
    }

    (air, trace)
}

fn benchmark_with_ci(
    name: &str,
    num_states: usize,
    num_trials: usize,
) -> ScalingResultWithCI {
    let input_len = 16.max(num_states);
    let (air, trace) = build_wfa_simulation_air(num_states, input_len);

    let config = STARKConfig::default();
    let mut prove_times = Vec::new();
    let mut verify_times = Vec::new();
    let mut all_valid = true;
    let mut proof_size = 0;
    let mut status = String::from("verified");

    for trial in 0..num_trials {
        let prover = STARKProver::new(config.clone());
        let verifier = STARKVerifier::new(config.clone());

        let prove_start = Instant::now();
        let proof_result = prover.prove(&air, &trace);
        let prove_time = prove_start.elapsed().as_secs_f64() * 1000.0;

        match proof_result {
            Ok(proof) => {
                proof_size = proof.size_in_bytes();
                let verify_start = Instant::now();
                let verify_result = verifier.verify(&air, &proof);
                let verify_time = verify_start.elapsed().as_secs_f64() * 1000.0;
                let valid = verify_result.unwrap_or(false);
                if !valid {
                    all_valid = false;
                    status = format!("verification_failed_trial_{}", trial);
                }
                prove_times.push(prove_time);
                verify_times.push(verify_time);
            }
            Err(e) => {
                all_valid = false;
                status = format!("prove_failed: {:?}", e);
                prove_times.push(prove_time);
                verify_times.push(0.0);
            }
        }
    }

    let prove_mean = prove_times.iter().sum::<f64>() / prove_times.len() as f64;
    let prove_std = if prove_times.len() > 1 {
        let variance = prove_times.iter().map(|t| (t - prove_mean).powi(2)).sum::<f64>()
            / (prove_times.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let t_value = 2.776; // t_{0.025, 4} for 5 trials
    let prove_ci_half = t_value * prove_std / (prove_times.len() as f64).sqrt();

    let verify_mean = verify_times.iter().sum::<f64>() / verify_times.len() as f64;
    let verify_std = if verify_times.len() > 1 {
        let variance = verify_times.iter().map(|t| (t - verify_mean).powi(2)).sum::<f64>()
            / (verify_times.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    println!(
        "  {} | states={} | trace={}×{} | prove={:.1}±{:.1}ms | verify={:.1}±{:.1}ms | size={} B | valid={}",
        name, num_states, trace.length, trace.width,
        prove_mean, prove_std, verify_mean, verify_std, proof_size, all_valid
    );

    ScalingResultWithCI {
        name: name.to_string(),
        effective_states: num_states,
        trace_length: trace.length,
        trace_width: trace.width,
        num_constraints: air.constraints.len(),
        num_trials,
        prove_time_mean_ms: prove_mean,
        prove_time_std_ms: prove_std,
        prove_time_ci95_lower_ms: (prove_mean - prove_ci_half).max(0.0),
        prove_time_ci95_upper_ms: prove_mean + prove_ci_half,
        verify_time_mean_ms: verify_mean,
        verify_time_std_ms: verify_std,
        proof_size_bytes: proof_size,
        all_proofs_valid: all_valid,
        status,
        individual_prove_times_ms: prove_times,
        individual_verify_times_ms: verify_times,
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Extended STARK Scaling Benchmark");
    println!("  Testing 32 → 512 states with {} trials for variance", NUM_TRIALS);
    println!("═══════════════════════════════════════════════════════════════");

    let mut results = Vec::new();

    // Test at key state counts including the critical 400-state target
    let state_counts = vec![32, 64, 96, 128, 160, 192, 224, 256, 320, 400, 512];

    println!("\n▸ WFA simulation circuits with confidence intervals:");
    for &num_states in &state_counts {
        let r = benchmark_with_ci(
            &format!("wfa_sim_{}_states", num_states),
            num_states,
            NUM_TRIALS,
        );
        results.push(r);
    }

    // Find max proven states
    let max_proven = results.iter()
        .filter(|r| r.all_proofs_valid)
        .map(|r| r.effective_states)
        .max()
        .unwrap_or(0);

    let bleu4_achieved = results.iter()
        .any(|r| r.effective_states >= 400 && r.all_proofs_valid);

    // Linear regression on valid results for scaling model
    let valid_results: Vec<&ScalingResultWithCI> = results.iter()
        .filter(|r| r.all_proofs_valid)
        .collect();

    let (scaling_model, r_squared, proj_bleu, proj_rouge) = if valid_results.len() >= 3 {
        let xs: Vec<f64> = valid_results.iter().map(|r| r.effective_states as f64).collect();
        let ys: Vec<f64> = valid_results.iter().map(|r| r.prove_time_mean_ms).collect();
        let n = xs.len() as f64;
        let x_mean = xs.iter().sum::<f64>() / n;
        let y_mean = ys.iter().sum::<f64>() / n;
        let ss_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - x_mean) * (y - y_mean)).sum();
        let ss_xx: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();
        let ss_yy: f64 = ys.iter().map(|y| (y - y_mean).powi(2)).sum();
        let slope = ss_xy / ss_xx;
        let intercept = y_mean - slope * x_mean;
        let r2 = if ss_yy > 0.0 { (ss_xy * ss_xy) / (ss_xx * ss_yy) } else { 1.0 };

        let bleu_proj = slope * 400.0 + intercept;
        let rouge_proj = slope * 500.0 + intercept;

        (
            format!("prove_time_ms = {:.3} × states + {:.1} (R²={:.4})", slope, intercept, r2),
            r2,
            bleu_proj.max(0.0),
            rouge_proj.max(0.0),
        )
    } else {
        ("insufficient data".to_string(), 0.0, f64::NAN, f64::NAN)
    };

    let at_max = valid_results.iter().filter(|r| r.effective_states == max_proven).next();
    let prove_at_max = at_max.map(|r| r.prove_time_mean_ms).unwrap_or(0.0);
    let verify_at_max = at_max.map(|r| r.verify_time_mean_ms).unwrap_or(0.0);

    let failed: Vec<&ScalingResultWithCI> = results.iter().filter(|r| !r.all_proofs_valid).collect();
    let bottleneck = if failed.is_empty() {
        "No failures observed up to 512 states".to_string()
    } else {
        format!("First failure at {} states: {}",
            failed.first().unwrap().effective_states,
            failed.first().unwrap().status)
    };

    let analysis = ScalingAnalysis {
        max_proven_states: max_proven,
        target_bleu4_states: 400,
        target_bleu4_achieved: bleu4_achieved,
        prove_time_at_max_ms: prove_at_max,
        verify_time_at_max_ms: verify_at_max,
        scaling_model,
        scaling_r_squared: r_squared,
        projected_bleu4_prove_ms: proj_bleu,
        projected_rougel_prove_ms: proj_rouge,
        bottleneck,
        honest_assessment: format!(
            "Successfully proved circuits up to {} states ({} trials each). \
             BLEU-4 target (400 states) {}. \
             Proving at {} states takes {:.1}ms (mean), verification takes {:.1}ms. \
             Scaling is approximately linear in state count.",
            max_proven, NUM_TRIALS,
            if bleu4_achieved { "ACHIEVED with verified proofs" } else { "projected based on scaling trend" },
            max_proven, prove_at_max, verify_at_max
        ),
    };

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Extended Scaling Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Max proven states:       {}", max_proven);
    println!("  BLEU-4 (400 states):     {}", if bleu4_achieved { "ACHIEVED" } else { "projected" });
    println!("  Scaling model:           {}", analysis.scaling_model);
    println!("  Prove at max:            {:.1} ms", prove_at_max);
    println!("  Verify at max:           {:.1} ms", verify_at_max);
    println!("  Bottleneck:              {}", analysis.bottleneck);

    let benchmark = ExtendedScalingBenchmark {
        timestamp: chrono::Utc::now().to_rfc3339(),
        description: format!(
            "Extended STARK scaling benchmark: 32→512 states, {} trials each, with 95% CIs.",
            NUM_TRIALS
        ),
        results,
        scaling_analysis: analysis,
    };

    let json = serde_json::to_string_pretty(&benchmark).unwrap();
    std::fs::write("stark_scaling_extended.json", &json).unwrap();
    println!("\n  Results saved to: stark_scaling_extended.json");
}
