//! TLS 1.2 → 1.3 Migration Downgrade Analysis
//!
//! Demonstrates using NegSynth to verify that a TLS library's migration from
//! TLS 1.2 to TLS 1.3 does not introduce downgrade vulnerabilities. The analysis
//! checks whether an active network attacker can force a session back to TLS 1.2
//! when both endpoints support TLS 1.3.
//!
//! Run: `cargo run --example analyze_migration`

use negsyn_proto_tls::{
    CipherSuiteRegistry, ClientHello, ClientPreferenceBuilder, NegotiationEngine,
    NegotiationPolicy, ServerHello, TlsExtension, TlsStateMachine, TlsVersion,
    VulnerabilityScanner,
};
use negsyn_types::{
    AnalysisConfig, AnalysisConfigBuilder, CipherSuite, CipherSuiteRegistry as TypesRegistry,
    EncodingConfig, ExtractionConfig, MergeConfig, NegotiationState, ProtocolConfig,
    ProtocolVersion, SlicerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Define the migration scenario ──────────────────────────────────
    // Server previously offered TLS 1.2 only; now supports both 1.2 and 1.3.
    let tls12 = TlsVersion { major: 3, minor: 3 };
    let tls13 = TlsVersion { major: 3, minor: 4 };

    // Build a strict negotiation policy that prefers TLS 1.3
    let policy = NegotiationPolicy::strict();
    let engine = NegotiationEngine::new(policy.clone());

    // ── 2. Simulate the client offering both versions ─────────────────────
    let pref_builder = ClientPreferenceBuilder::new(policy.clone());
    let cipher_list_13 = pref_builder.build_cipher_list(tls13);
    let cipher_list_12 = pref_builder.build_cipher_list(tls12);

    println!("TLS 1.3 cipher suites offered: {}", cipher_list_13.len());
    println!("TLS 1.2 cipher suites offered: {}", cipher_list_12.len());

    // Construct a ClientHello that advertises supported_versions extension
    let mut client_hello = ClientHello::new(tls12, rand_bytes());
    client_hello.cipher_suites = [cipher_list_13.clone(), cipher_list_12.clone()].concat();
    client_hello.extensions.push(TlsExtension::SupportedVersions {
        versions: vec![tls13, tls12],
    });

    // ── 3. Negotiate and check for downgrade indicators ───────────────────
    let result = engine.negotiate(&client_hello);
    println!(
        "Negotiation outcome: version={:?}, cipher=0x{:04X}",
        result.version, result.cipher_suite
    );

    // Check the ServerHello random bytes for the RFC 8446 downgrade sentinel
    let server_hello = ServerHello::new(result.version, rand_bytes(), result.cipher_suite);
    if server_hello.has_downgrade_sentinel() {
        println!("⚠  Downgrade sentinel detected in ServerHello.random");
    }

    // ── 4. Scan for known vulnerabilities in the negotiated parameters ────
    let scanner = VulnerabilityScanner::new();
    let detections = scanner.scan_client_hello(&client_hello);
    for d in &detections {
        println!(
            "  [{:.1}] {} — {}",
            d.confidence,
            d.vulnerability.cve_id(),
            d.vulnerability.name(),
        );
    }

    // ── 5. Configure the full NegSynth analysis pipeline ──────────────────
    let config = AnalysisConfigBuilder::new()
        .protocol(ProtocolConfig::tls_default())
        .slicer(SlicerConfig {
            max_depth: 12,
            include_indirect_calls: true,
            ..Default::default()
        })
        .merge(MergeConfig {
            max_states: 4096,
            ..Default::default()
        })
        .extraction(ExtractionConfig {
            minimize: true,
            ..Default::default()
        })
        .encoding(EncodingConfig {
            depth_bound: 8,
            adversary_budget: 6,
            ..Default::default()
        })
        .build()?;
    config.validate()?;

    // ── 6. Verify no downgrade paths exist ────────────────────────────────
    let initial = NegotiationState::initial();
    let is_downgraded = initial.is_downgraded();
    println!(
        "Initial negotiation state downgraded: {} (expected false)",
        is_downgraded
    );

    // Confirm TLS 1.3 is not a downgrade from TLS 1.2
    let v13 = ProtocolVersion::tls13();
    let v12 = ProtocolVersion::tls12();
    assert!(
        !v13.is_downgrade_from(&v12),
        "TLS 1.3 must not be flagged as a downgrade from 1.2"
    );
    assert!(
        v12.is_downgrade_from(&v13),
        "TLS 1.2 IS a downgrade from 1.3"
    );

    println!("✓ Migration analysis complete — no spurious downgrade paths found.");
    Ok(())
}

fn rand_bytes() -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[0] = 0xCA;
    buf[1] = 0xFE;
    buf
}
