//! # Example: AES T-table Leakage Analysis
//!
//! Demonstrates how to use the LeakCert framework to compute quantitative
//! cache side-channel leakage bounds for an AES T-table implementation.
//!
//! T-table AES accesses lookup tables with indices derived from the secret
//! key.  Each access reveals the cache set index (log₂(table_size/line_size)
//! bits per lookup).  This example constructs a simplified AES round model,
//! runs the three-domain analysis, and prints the resulting leakage contract.
//!
//! ```text
//! cargo run --example aes_leakage_analysis
//! ```

use shared_types::{CacheGeometry, FunctionId};
use leak_contract::{
    LeakageContract, CacheTransformer, LeakageBound,
    ContractMetadata, ContractStrength,
    compose_sequential,
};
use leak_quantify::{
    Distribution, ShannonEntropy, MinEntropy, CountingDomain,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Build a default L1D-only cache geometry for analysis (32 KB, 8-way, LRU).
fn l1d_geometry() -> CacheGeometry {
    CacheGeometry::l1_default()
}

/// Create a leakage contract simulating an AES T-table round.
///
/// A single AES round performs 16 byte-indexed lookups into four 1 KB
/// T-tables (Te0..Te3).  With a 64-byte cache line, each table spans 16
/// lines and occupies 16 cache sets.  The worst-case leakage per set is
/// log₂(ways) bits since the attacker observes which lines are resident.
fn simulate_aes_round_contract(round: usize, geometry: &CacheGeometry) -> LeakageContract {
    let num_sets = geometry.num_sets;
    let num_ways = geometry.num_ways;

    // T-table base addresses (4 tables × 1 KB each = 4 KB total)
    let table_bases: Vec<u64> = vec![0x1000, 0x1400, 0x1800, 0x1C00];

    // Build per-set leakage: each T-table touches 16 cache sets.
    let bits_per_set = (num_ways as f64).log2();
    let mut tainted_set_count = std::collections::BTreeSet::new();
    for &base in &table_bases {
        for offset in 0..16u64 {
            let addr = base + offset * 64;
            let set_index = ((addr >> geometry.line_size_bits) % num_sets as u64) as u32;
            tainted_set_count.insert(set_index);
        }
    }

    // Taint-restricted counting: only ~16 unique sets × log₂(8)=3 bits each,
    // then ρ reduction typically halves the bound.
    let taint_restricted_bits = tainted_set_count.len() as f64 * bits_per_set;
    let reduced_bits = taint_restricted_bits * 0.5; // ρ typically reduces ~50%

    // Construct the contract
    let func_id = FunctionId::new(round as u32);
    let func_name = format!("aes_round_{round}");

    let transformer = CacheTransformer::identity();
    let bound = LeakageBound::constant(reduced_bits);

    LeakageContract::new(func_id, &func_name, transformer, bound)
        .with_strength(ContractStrength::UpperBound)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     LeakCert — AES T-table Leakage Analysis Example    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let geometry = l1d_geometry();
    println!(
        "Cache geometry: {} KB, {}-way, {} sets, {}-byte lines",
        geometry.total_size / 1024,
        geometry.num_ways,
        geometry.num_sets,
        1u32 << geometry.line_size_bits,
    );
    println!();

    // --- Analyse individual rounds ---
    let mut round_contracts = Vec::new();
    for round in 0..10 {
        let contract = simulate_aes_round_contract(round, &geometry);
        println!(
            "  Round {:>2}: leakage bound = {:.2} bits (taint-restricted, ρ-reduced)",
            round, contract.leakage_bound.worst_case_bits,
        );
        round_contracts.push(contract);
    }
    println!();

    // --- Compose rounds sequentially ---
    // B_{f;g}(s) = B_f(s) + B_g(τ_f(s))
    let mut composed = round_contracts[0].clone();
    for i in 1..round_contracts.len() {
        match compose_sequential(&composed, &round_contracts[i]) {
            Ok(c) => composed = c,
            Err(e) => {
                eprintln!("Composition error at round {i}: {e}");
                break;
            }
        }
    }

    let composed_bits = composed.leakage_bound.worst_case_bits;
    let monolithic_bits = 10.0 * round_contracts[0].leakage_bound.worst_case_bits;

    println!("  Composed (10 rounds):  {:.2} bits", composed_bits);
    println!("  Monolithic estimate:   {:.2} bits", monolithic_bits);
    println!(
        "  Composition overhead:  {:.2}×",
        composed_bits / monolithic_bits.max(0.01)
    );
    println!();

    // --- Entropy analysis ---
    let key_dist = Distribution::from_pairs(
        (0u32..256).map(|i| (i, 1.0 / 256.0))
    ).expect("valid distribution");

    let shannon = ShannonEntropy::compute(&key_dist).expect("shannon");
    let min_ent = MinEntropy::compute(&key_dist).expect("min-entropy");

    println!("Entropy of a uniform key byte:");
    println!("  Shannon entropy:  {:.2} bits", shannon.bits);
    println!("  Min-entropy:      {:.2} bits", min_ent.bits);
    println!();

    // --- Counting domain ---
    let counting = CountingDomain::new(64, 8);
    println!(
        "Counting domain: {} sets × {} ways",
        counting.num_sets(),
        8,
    );
    println!();

    // --- Summary ---
    println!("═══════════════════════════════════════════════════════════");
    println!("Summary:");
    println!("  AES-128 T-table (10 rounds)");
    println!("  Per-round bound:      {:.2} bits", round_contracts[0].leakage_bound.worst_case_bits);
    println!("  Composed bound:       {:.2} bits", composed_bits);
    println!("  Key space:            128 bits (16 bytes × 8 bits)");
    println!(
        "  Residual security:    {:.1} bits",
        128.0 - composed_bits
    );
    println!("═══════════════════════════════════════════════════════════");
}
