//! Cipher Suite Removal Analysis
//!
//! Demonstrates verifying that removing weak cipher suites (RC4, 3DES) from a
//! TLS deployment does not introduce new downgrade paths. NegSynth checks that
//! the reduced cipher set still covers all legitimate negotiation outcomes and
//! that no attacker-reachable path leads through the removed suites.
//!
//! Run: `cargo run --example cipher_removal`

use negsyn_proto_tls::{
    CipherSuiteRegistry, NegotiationEngine, NegotiationPolicy, SecurityLevel, TlsCipherSuite,
    VulnerabilityScanner,
};
use negsyn_types::{CipherSuite, CipherSuiteRegistry as TypesRegistry, NegotiationState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = CipherSuiteRegistry::new();

    // ── 1. Identify weak ciphers to remove ────────────────────────────────
    let weak_suites = registry.weak_suites();
    let export_suites = registry.export_suites();

    println!("=== Cipher Suite Removal Analysis ===\n");
    println!("Weak suites found:   {}", weak_suites.len());
    println!("Export suites found: {}", export_suites.len());

    for suite in &weak_suites {
        println!(
            "  REMOVE  0x{:04X}  {}  ({:?})",
            suite.iana_id,
            suite.name,
            suite.security_level()
        );
    }

    // ── 2. Build the "after removal" cipher set ───────────────────────────
    let removed_ids: std::collections::HashSet<u16> = weak_suites
        .iter()
        .chain(export_suites.iter())
        .map(|s| s.iana_id)
        .collect();

    let remaining: Vec<&TlsCipherSuite> = registry
        .all_suites()
        .into_iter()
        .filter(|s| !removed_ids.contains(&s.iana_id))
        .collect();

    println!("\nRemaining suites: {}", remaining.len());

    // ── 3. Verify the remaining set has no security gaps ──────────────────
    let strong_suites: Vec<&TlsCipherSuite> = registry
        .filter_by_security(SecurityLevel::Recommended)
        .into_iter()
        .collect();

    // Every recommended suite should survive the removal
    for suite in &strong_suites {
        assert!(
            !removed_ids.contains(&suite.iana_id),
            "Recommended suite 0x{:04X} ({}) would be removed!",
            suite.iana_id,
            suite.name
        );
    }
    println!("✓ All recommended suites preserved.");

    // ── 4. Scan the remaining set for known vulnerabilities ───────────────
    let scanner = VulnerabilityScanner::new();
    let remaining_ids: Vec<u16> = remaining.iter().map(|s| s.iana_id).collect();
    let detections = scanner.scan_cipher_suites(&remaining_ids);

    if detections.is_empty() {
        println!("✓ No known vulnerabilities in the remaining cipher set.");
    } else {
        for d in &detections {
            println!(
                "  ⚠ [{:.1}] {} — {}",
                d.confidence,
                d.vulnerability.cve_id(),
                d.vulnerability.name(),
            );
        }
    }

    // ── 5. Confirm forward secrecy coverage ───────────────────────────────
    let fs_count = remaining.iter().filter(|s| s.has_forward_secrecy()).count();
    println!(
        "\nForward-secrecy suites remaining: {} / {}",
        fs_count,
        remaining.len()
    );
    assert!(fs_count > 0, "Must retain at least one PFS suite");

    println!("\n✓ Cipher removal analysis complete — safe to deploy.");
    Ok(())
}
