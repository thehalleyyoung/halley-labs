use anyhow::{Context, Result};
use std::path::PathBuf;

use regsynth_certificate::Certificate;

use crate::config::AppConfig;
use crate::output::OutputFormatter;

/// Run the certificate verification command.
pub fn run(
    _config: &AppConfig,
    formatter: &OutputFormatter,
    certificate_path: &PathBuf,
    _deep_verify: bool,
) -> Result<()> {
    formatter.status("Verifying certificate...");

    let content = std::fs::read_to_string(certificate_path)
        .with_context(|| format!("Failed to read {}", certificate_path.display()))?;

    // Try parsing as a wrapped certificate (from certify output) or a bare certificate
    let cert: Certificate = if content.contains("\"certificate\"") {
        let wrapper: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| "Failed to parse certificate file")?;
        serde_json::from_value(
            wrapper
                .get("certificate")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
        .with_context(|| "Failed to extract certificate from wrapper")?
    } else {
        serde_json::from_str(&content).with_context(|| "Failed to parse certificate")?
    };

    let kind_str = &cert.certificate_type;

    formatter.status(&format!("  Certificate type: {}", kind_str));
    formatter.status(&format!("  Issued at:       {}", cert.issued_at));
    formatter.status(&format!("  Fingerprint:     {}…", &cert.fingerprint.hex_digest[..cert.fingerprint.hex_digest.len().min(16)]));
    formatter.status("");

    // Step 1: Integrity verification
    let integrity_valid = cert.verify_integrity();
    formatter.status(&format!(
        "  [{}] Fingerprint integrity check",
        if integrity_valid { "✓" } else { "✗" }
    ));

    let overall_valid = integrity_valid;

    formatter.status("");
    formatter.status(&format!(
        "  Overall: {}",
        if overall_valid {
            "✓ VALID"
        } else {
            "✗ INVALID"
        }
    ));

    formatter.write_certificate_summary(
        kind_str,
        kind_str,
        &cert.fingerprint.hex_digest,
        0,
        overall_valid,
    )?;

    // Full verification result
    let output = serde_json::json!({
        "kind": kind_str,
        "integrity_valid": integrity_valid,
        "overall_valid": overall_valid,
    });
    formatter.write_value(&output)?;

    if !overall_valid {
        anyhow::bail!("Certificate verification failed");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_certificate() {
        let cert = Certificate::wrap("compliance", &"test_data", Vec::new()).unwrap();
        assert!(cert.verify_integrity());
    }

    #[test]
    fn test_tampered_certificate() {
        let mut cert = Certificate::wrap("compliance", &"test_data", Vec::new()).unwrap();
        cert.payload_json = "\"tampered\"".into();
        assert!(!cert.verify_integrity());
    }

    #[test]
    fn test_certificate_to_json_roundtrip() {
        let cert = Certificate::wrap("test", &"hello", Vec::new()).unwrap();
        let json = cert.to_json().unwrap();
        let parsed: Certificate = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.certificate_type, "test");
        assert_eq!(parsed.fingerprint, cert.fingerprint);
    }
}
