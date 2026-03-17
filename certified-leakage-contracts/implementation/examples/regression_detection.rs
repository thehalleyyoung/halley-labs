//! # Example: Contract Composition & Regression Detection
//!
//! Demonstrates the compositional contract system and regression detection
//! workflow.  This is the "killer app" of LeakCert: even when absolute
//! leakage bounds are conservative, *changes* between binary versions are
//! precisely captured.
//!
//! The example simulates:
//! 1. Building per-function leakage contracts for a small crypto library
//! 2. Composing them into whole-library bounds
//! 3. Detecting a regression when a "patched" version introduces a leak
//!
//! ```text
//! cargo run --example regression_detection
//! ```

use std::collections::BTreeMap;

use shared_types::{CacheGeometry, FunctionId};
use leak_contract::{
    LeakageContract, CacheTransformer, LeakageBound,
    ContractMetadata, ContractStrength,
    compose_sequential, compose_conditional,
    RegressionSeverity,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Build a mock leakage contract for a function.
fn make_contract(name: &str, id: u32, leakage_bits: f64, speculative: bool) -> LeakageContract {
    let func_id = FunctionId::new(id);
    let transformer = CacheTransformer::identity();
    let bound = LeakageBound::constant(leakage_bits);

    let mut contract = LeakageContract::new(func_id, name, transformer, bound);
    if leakage_bits == 0.0 {
        contract.strength = ContractStrength::Exact;
    } else {
        contract.strength = ContractStrength::UpperBound;
    }
    contract
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   LeakCert — Contract Composition & Regression Demo    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // -----------------------------------------------------------------------
    // Part 1: Build per-function contracts (simulating analysis output)
    // -----------------------------------------------------------------------
    println!("─── Part 1: Per-function Leakage Contracts ───");
    println!();

    // Baseline version (v1.0)
    let v1_contracts = vec![
        make_contract("aes_encrypt_block",     0, 8.0,  false),
        make_contract("aes_key_expand",        1, 0.0,  false),
        make_contract("chacha20_block",        2, 0.0,  false),
        make_contract("poly1305_update",       3, 0.0,  false),
        make_contract("curve25519_scalarmult", 4, 2.4,  false),
        make_contract("sha256_transform",      5, 0.0,  false),
    ];

    println!("  Baseline (v1.0) contracts:");
    for c in &v1_contracts {
        let status = if c.leakage_bound.worst_case_bits == 0.0 { "✓ CT" } else { "⚠ LEAK" };
        println!(
            "    {:<28} {:>6.2} bits  {}",
            c.function_name,
            c.leakage_bound.worst_case_bits,
            status,
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Part 2: Compose into whole-library bound
    // -----------------------------------------------------------------------
    println!("─── Part 2: Compositional Whole-Library Bound ───");
    println!();

    // Sequential: TLS record encryption = AES + Poly1305
    let tls_bits = v1_contracts[0].leakage_bound.worst_case_bits
        + v1_contracts[3].leakage_bound.worst_case_bits;

    // Conditional: key exchange = max(X25519, ECDH) + 1 branch bit
    let kex_bits = 1.0 + v1_contracts[4].leakage_bound.worst_case_bits.max(3.0);

    let total: f64 = v1_contracts.iter().map(|c| c.leakage_bound.worst_case_bits).sum();

    println!("  TLS record encrypt (AES;Poly1305):  {:.2} bits", tls_bits);
    println!("  Key exchange (X25519 | ECDH):        {:.2} bits", kex_bits);
    println!("  Whole-library upper bound:           {:.2} bits", total);
    println!();

    // -----------------------------------------------------------------------
    // Part 3: Regression detection
    // -----------------------------------------------------------------------
    println!("─── Part 3: Regression Detection ───");
    println!();

    // New version (v1.1) — compiler upgrade introduced a leak in ChaCha20
    let v2_contracts = vec![
        make_contract("aes_encrypt_block",     0, 8.0,  false),
        make_contract("aes_key_expand",        1, 0.0,  false),
        make_contract("chacha20_block",        2, 1.5,  false), // REGRESSION!
        make_contract("poly1305_update",       3, 0.0,  false),
        make_contract("curve25519_scalarmult", 4, 2.4,  false),
        make_contract("sha256_transform",      5, 0.0,  false),
        make_contract("aes_gcm_encrypt",       6, 4.2,  true),  // new
    ];

    println!("  Updated (v1.1) contracts:");
    for c in &v2_contracts {
        let status = if c.leakage_bound.worst_case_bits == 0.0 { "✓ CT" } else { "⚠ LEAK" };
        println!(
            "    {:<28} {:>6.2} bits  {}",
            c.function_name,
            c.leakage_bound.worst_case_bits,
            status,
        );
    }
    println!();

    // Compute deltas
    println!("  Regression report (v1.0 → v1.1):");
    println!("  ┌──────────────────────────────┬───────────┬───────────┬─────────┐");
    println!("  │ Function                     │ v1.0 bits │ v1.1 bits │  Δ bits │");
    println!("  ├──────────────────────────────┼───────────┼───────────┼─────────┤");

    let mut regressions_found = 0u32;
    for v2c in &v2_contracts {
        let v1_match = v1_contracts.iter().find(|c| c.function_name == v2c.function_name);
        let (v1_bits_str, delta, marker) = match v1_match {
            Some(v1c) => {
                let d = v2c.leakage_bound.worst_case_bits - v1c.leakage_bound.worst_case_bits;
                let m = if d > 0.01 {
                    regressions_found += 1;
                    " ← REGRESSION"
                } else {
                    ""
                };
                (format!("{:>7.2}", v1c.leakage_bound.worst_case_bits), d, m)
            }
            None => {
                regressions_found += 1;
                ("  (new)".to_string(), v2c.leakage_bound.worst_case_bits, " ← NEW")
            }
        };

        println!(
            "  │ {:<28} │ {}   │ {:>7.2}   │ {:>+6.2}  │{}",
            v2c.function_name,
            v1_bits_str,
            v2c.leakage_bound.worst_case_bits,
            delta,
            marker,
        );
    }
    println!("  └──────────────────────────────┴───────────┴───────────┴─────────┘");
    println!();

    // CI-style summary
    let severity = if regressions_found > 0 {
        RegressionSeverity::Critical
    } else {
        RegressionSeverity::Info
    };

    println!("  CI Summary:");
    println!("    Regressions found:  {}", regressions_found);
    println!("    Severity:           {}", severity.label());
    println!(
        "    Exit code:          {} ({})",
        if regressions_found > 0 { 1 } else { 0 },
        if regressions_found > 0 { "FAIL" } else { "PASS" },
    );
    println!();

    // --- JSON output for CI ---
    println!("─── CI JSON Output ───");
    let ci_json = serde_json::json!({
        "tool": "leakcert",
        "version": "0.1.0",
        "comparison": {
            "baseline": "v1.0",
            "current": "v1.1",
        },
        "regressions": [
            {
                "function": "chacha20_block",
                "baseline_bits": 0.0,
                "current_bits": 1.5,
                "delta_bits": 1.5,
                "severity": "critical",
                "note": "Compiler -O3 converted cmov to conditional branch"
            },
            {
                "function": "aes_gcm_encrypt",
                "baseline_bits": null,
                "current_bits": 4.2,
                "delta_bits": 4.2,
                "severity": "warning",
                "note": "New function without baseline contract"
            }
        ],
        "summary": {
            "total_functions": v2_contracts.len(),
            "regressions": regressions_found,
            "exit_code": 1,
        }
    });
    println!("{}", serde_json::to_string_pretty(&ci_json).unwrap());

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Regression detection completes even when absolute bounds");
    println!("  are conservative — Δ captures the *change* precisely.");
    println!("═══════════════════════════════════════════════════════════");
}
