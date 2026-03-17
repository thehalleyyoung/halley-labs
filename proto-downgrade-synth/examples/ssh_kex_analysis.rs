//! SSH Key Exchange Analysis
//!
//! Demonstrates using negsyn_proto_ssh to analyse SSH key exchange negotiation,
//! detect algorithm downgrades, and verify that a server configuration is robust
//! against known SSH-layer attacks.
//!
//! Run: `cargo run --example ssh_kex_analysis`

use negsyn_proto_ssh::{
    CompressionAlgorithm, EncryptionAlgorithm, HostKeyAlgorithm, KexAlgorithm, KexInitBuilder,
    MacAlgorithm as SshMac, SecurityClassification, SshNegotiationEngine, SshVulnerability,
    StrictKex, VulnerabilityScanner as SshScanner,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SSH Key Exchange Analysis ===\n");

    // ── 1. Build client KexInit ───────────────────────────────────────────
    let client = KexInitBuilder::new()
        .kex_algorithms_typed(&[
            KexAlgorithm::Curve25519Sha256,
            KexAlgorithm::EcdhSha2Nistp256,
            KexAlgorithm::DhGroup16Sha512,
            KexAlgorithm::DhGroup14Sha256,
        ])
        .host_key_algorithms_typed(&[
            HostKeyAlgorithm::Ed25519,
            HostKeyAlgorithm::EcdsaSha2Nistp256,
            HostKeyAlgorithm::RsaSha2_512,
            HostKeyAlgorithm::RsaSha2_256,
        ])
        .encryption_c2s_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
            EncryptionAlgorithm::Aes256Ctr,
        ])
        .encryption_s2c_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
            EncryptionAlgorithm::Aes256Ctr,
        ])
        .mac_c2s_typed(&[SshMac::HmacSha2_256Etm, SshMac::HmacSha2_512Etm])
        .mac_s2c_typed(&[SshMac::HmacSha2_256Etm, SshMac::HmacSha2_512Etm])
        .with_ext_info_c()
        .with_strict_kex_client()
        .build();

    // ── 2. Build server KexInit ───────────────────────────────────────────
    let server = KexInitBuilder::new()
        .kex_algorithms_typed(&[
            KexAlgorithm::Curve25519Sha256,
            KexAlgorithm::EcdhSha2Nistp256,
        ])
        .host_key_algorithms_typed(&[HostKeyAlgorithm::Ed25519])
        .encryption_c2s_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
        ])
        .encryption_s2c_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
        ])
        .mac_c2s_typed(&[SshMac::HmacSha2_256Etm])
        .mac_s2c_typed(&[SshMac::HmacSha2_256Etm])
        .with_ext_info_s()
        .with_strict_kex_server()
        .build();

    // ── 3. Run negotiation ────────────────────────────────────────────────
    let engine = SshNegotiationEngine::new()
        .with_strict_kex(StrictKex::Both)
        .with_min_security(SecurityClassification::Recommended);

    let negotiated = engine.negotiate(&client, &server)?;
    println!("Negotiated algorithms:");
    println!("  KEX:            {}", negotiated.kex);
    println!("  Host key:       {}", negotiated.host_key);
    println!("  Encryption C→S: {}", negotiated.encryption_c2s);
    println!("  Encryption S→C: {}", negotiated.encryption_s2c);
    println!(
        "  Overall security: {:?}",
        negotiated.overall_security()
    );

    // ── 4. Vulnerability scan ─────────────────────────────────────────────
    let report = SshScanner::scan(&client, &server);
    println!("\nVulnerability scan: {} findings", report.vulnerabilities.len());
    for vuln in &report.vulnerabilities {
        println!(
            "  [{:.1}] {} — {}",
            vuln.severity(),
            vuln.cve().unwrap_or("N/A"),
            vuln.description(),
        );
    }

    // ── 5. Check for downgrade opportunities ──────────────────────────────
    let downgrades = engine.detect_downgrade(&client, &server);
    if downgrades.is_empty() {
        println!("\n✓ No algorithm downgrade opportunities detected.");
    } else {
        for dg in &downgrades {
            println!("  ⚠ Downgrade: {} → {}", dg.strong, dg.weak);
        }
    }

    println!("\n✓ SSH key exchange analysis complete.");
    Ok(())
}
