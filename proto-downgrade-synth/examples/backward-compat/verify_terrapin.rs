//! CVE-2023-48795 (Terrapin) Verification
//!
//! Demonstrates detecting the Terrapin attack, which exploits SSH extension
//! negotiation to manipulate sequence numbers during the handshake. An active
//! MitM can delete the EXT_INFO message and downgrade security features like
//! keystroke timing obfuscation (chacha20-poly1305).
//!
//! Run: `cargo run --example verify_terrapin`

use negsyn_proto_ssh::{
    EncryptionAlgorithm, ExtInfo, HostKeyAlgorithm, KexAlgorithm, KexInitBuilder,
    SecurityClassification, SshNegotiationEngine, SshVulnerability, StrictKex,
    VulnerabilityScanner as SshScanner,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Build realistic client and server KexInit messages ─────────────
    let client_kex = KexInitBuilder::new()
        .kex_algorithms_typed(&[
            KexAlgorithm::Curve25519Sha256,
            KexAlgorithm::EcdhSha2Nistp256,
        ])
        .host_key_algorithms_typed(&[
            HostKeyAlgorithm::Ed25519,
            HostKeyAlgorithm::EcdsaSha2Nistp256,
        ])
        .encryption_c2s_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
        ])
        .encryption_s2c_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
        ])
        .with_ext_info_c()
        .build();

    let server_kex = KexInitBuilder::new()
        .kex_algorithms_typed(&[KexAlgorithm::Curve25519Sha256])
        .host_key_algorithms_typed(&[HostKeyAlgorithm::Ed25519])
        .encryption_c2s_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
        ])
        .encryption_s2c_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
        ])
        .with_ext_info_s()
        .build();

    // ── 2. Scan for Terrapin vulnerability ────────────────────────────────
    let report = SshScanner::scan(&client_kex, &server_kex);
    let terrapin_findings: Vec<_> = report
        .vulnerabilities
        .iter()
        .filter(|v| matches!(v, SshVulnerability::Terrapin))
        .collect();

    println!("Vulnerabilities found: {}", report.vulnerabilities.len());
    for vuln in &report.vulnerabilities {
        println!(
            "  [{:.1}] {} — {}",
            vuln.severity(),
            vuln.cve().unwrap_or("N/A"),
            vuln.description(),
        );
    }

    // ── 3. Verify strict-kex mitigates the attack ─────────────────────────
    let strict_client = KexInitBuilder::new()
        .kex_algorithms_typed(&[KexAlgorithm::Curve25519Sha256])
        .encryption_c2s_typed(&[EncryptionAlgorithm::Chacha20Poly1305])
        .encryption_s2c_typed(&[EncryptionAlgorithm::Chacha20Poly1305])
        .with_strict_kex_client()
        .with_ext_info_c()
        .build();

    let strict_server = KexInitBuilder::new()
        .kex_algorithms_typed(&[KexAlgorithm::Curve25519Sha256])
        .encryption_c2s_typed(&[EncryptionAlgorithm::Chacha20Poly1305])
        .encryption_s2c_typed(&[EncryptionAlgorithm::Chacha20Poly1305])
        .with_strict_kex_server()
        .with_ext_info_s()
        .build();

    assert!(strict_client.has_strict_kex_client());
    assert!(strict_server.has_strict_kex_server());

    let strict_report = SshScanner::scan(&strict_client, &strict_server);
    let still_vulnerable = strict_report
        .vulnerabilities
        .iter()
        .any(|v| matches!(v, SshVulnerability::Terrapin));

    println!(
        "\nWith strict-kex: Terrapin mitigated = {}",
        !still_vulnerable
    );
    println!("✓ Terrapin verification complete.");

    Ok(())
}
