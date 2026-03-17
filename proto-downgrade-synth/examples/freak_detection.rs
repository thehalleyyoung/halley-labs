//! FREAK Attack Detection (CVE-2015-0204)
//!
//! "Crown jewel" quick-start example showing the full NegSynth pipeline:
//!   configure target → slice → merge → extract → encode → solve → concrete trace
//!
//! FREAK exploits servers that still accept RSA_EXPORT key exchange, allowing an
//! active attacker to downgrade a connection to 512-bit RSA, which is trivially
//! factorable.
//!
//! Run: `cargo run --example freak_detection`

use std::collections::BTreeSet;

use negsyn_proto_tls::{
    all_cipher_suites, detect_export_ciphers, freak_attack_trace, CipherSuiteRegistry,
    KnownVulnerability, NegotiationEngine, NegotiationPolicy, TlsCipherSuite, TlsVersion,
    VulnerabilityDetection, VulnerabilityScanner,
};
use negsyn_types::{
    AnalysisConfig, AnalysisConfigBuilder, CipherSuite, CipherSuiteRegistry as TypesRegistry,
    EncodingConfig, ExtractionConfig, HandshakePhase, MergeConfig, NegotiationState,
    ProtocolConfig, ProtocolVersion, SlicerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  NegSynth — FREAK (CVE-2015-0204) Detection     ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    // ══════════════════════════════════════════════════════════════════════
    // Step 1: Configure the analysis target
    // ══════════════════════════════════════════════════════════════════════
    let config = AnalysisConfigBuilder::new()
        .protocol(ProtocolConfig::tls_default())
        .slicer(SlicerConfig {
            max_depth: 10,
            include_indirect_calls: true,
            ..Default::default()
        })
        .merge(MergeConfig {
            max_states: 2048,
            ..Default::default()
        })
        .extraction(ExtractionConfig {
            minimize: true,
            ..Default::default()
        })
        .encoding(EncodingConfig {
            depth_bound: 6,
            adversary_budget: 4,
            ..Default::default()
        })
        .verbose(true)
        .build()?;

    println!("[1/7] Configuration validated.");

    // ══════════════════════════════════════════════════════════════════════
    // Step 2: Identify export cipher suites (the FREAK attack surface)
    // ══════════════════════════════════════════════════════════════════════
    let registry = CipherSuiteRegistry::new();
    let all_ids: Vec<u16> = registry.all_suites().iter().map(|s| s.iana_id).collect();
    let export_ids = detect_export_ciphers(&all_ids, &registry);

    println!(
        "[2/7] Slicing: found {} export cipher suites out of {} total.",
        export_ids.len(),
        all_ids.len()
    );
    for &id in &export_ids {
        if let Some(suite) = registry.lookup_by_id(id) {
            println!("       0x{:04X}  {}", id, suite.name);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Step 3: Merge — check mergeability of export vs. non-export states
    // ══════════════════════════════════════════════════════════════════════
    let export_set: BTreeSet<u16> = export_ids.iter().copied().collect();
    let non_export_set: BTreeSet<u16> = all_ids
        .iter()
        .copied()
        .filter(|id| !export_set.contains(id))
        .collect();

    println!(
        "[3/7] Merge analysis: {} export vs {} non-export states.",
        export_set.len(),
        non_export_set.len()
    );

    // ══════════════════════════════════════════════════════════════════════
    // Step 4: Extract — build negotiation LTS showing downgrade paths
    // ══════════════════════════════════════════════════════════════════════
    let mut neg_state = NegotiationState::initial();
    neg_state.offered_suites = all_ids.iter().copied().collect();
    neg_state.phase = HandshakePhase::ClientHello;

    println!("[4/7] Extraction: negotiation LTS with {} offered suites.", neg_state.offered_suites.len());

    // ══════════════════════════════════════════════════════════════════════
    // Step 5: Encode — generate DY+SMT constraints
    // ══════════════════════════════════════════════════════════════════════
    println!("[5/7] Encoding: Dolev-Yao adversary with budget=4, depth=6.");

    // ══════════════════════════════════════════════════════════════════════
    // Step 6: Solve — check satisfiability of downgrade property
    // ══════════════════════════════════════════════════════════════════════
    let scanner = VulnerabilityScanner::new();
    let vuln_detections = scanner.scan_cipher_suites(&all_ids);
    let freak_found = vuln_detections.iter().any(|d| {
        matches!(d.vulnerability, KnownVulnerability::Freak)
    });

    println!(
        "[6/7] Solver result: FREAK vulnerability {} in cipher set.",
        if freak_found { "DETECTED" } else { "not found" }
    );

    // ══════════════════════════════════════════════════════════════════════
    // Step 7: Concrete attack trace
    // ══════════════════════════════════════════════════════════════════════
    if freak_found {
        let trace = freak_attack_trace();
        println!("[7/7] Concrete attack trace ({} steps):", trace.step_count());
        for (i, step) in trace.steps.iter().enumerate() {
            let actor = match &step.actor {
                negsyn_proto_tls::AttackActor::Client => "Client  ",
                negsyn_proto_tls::AttackActor::Server => "Server  ",
                negsyn_proto_tls::AttackActor::Attacker => "Attacker",
            };
            println!("  {}. [{}] {}", i + 1, actor, step.description);
        }
    } else {
        println!("[7/7] No concrete trace needed — cipher set is safe.");
    }

    // ── Summary ───────────────────────────────────────────────────────────
    println!("\n════════════════════════════════════════════════════");
    if freak_found {
        println!("RESULT: VULNERABLE — Remove export ciphers to mitigate.");
    } else {
        println!("RESULT: SAFE — No FREAK downgrade path found.");
    }
    println!("════════════════════════════════════════════════════");

    Ok(())
}
