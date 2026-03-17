//! Certificate serialization, deserialization, and audit packaging.
//!
//! Provides a unified `Certificate` envelope that wraps any certificate
//! payload with fingerprinting, format metadata, and source references.
//! Supports JSON, plain text, and compact binary (CBOR-style) export.

use serde::{Deserialize, Serialize};

use crate::fingerprint::CertificateFingerprint;

// ─── Certificate Format ─────────────────────────────────────────────────────

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateFormat {
    Json,
    Text,
    Binary,
}

impl CertificateFormat {
    pub fn file_extension(&self) -> &'static str {
        match self {
            CertificateFormat::Json => "json",
            CertificateFormat::Text => "txt",
            CertificateFormat::Binary => "bin",
        }
    }

    pub fn mime_type(&self) -> &'static str {
        match self {
            CertificateFormat::Json => "application/json",
            CertificateFormat::Text => "text/plain",
            CertificateFormat::Binary => "application/octet-stream",
        }
    }
}

// ─── Source Reference ───────────────────────────────────────────────────────

/// A reference to a source file used in certificate generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReference {
    pub file_path: String,
    pub line_range: Option<(usize, usize)>,
    pub content_hash: String,
}

impl SourceReference {
    pub fn new(path: &str, content: &str) -> Self {
        Self {
            file_path: path.to_string(),
            line_range: None,
            content_hash: CertificateFingerprint::compute(content.as_bytes()).hex_digest,
        }
    }

    pub fn with_lines(mut self, start: usize, end: usize) -> Self {
        self.line_range = Some((start, end));
        self
    }
}

// ─── Unified Certificate Envelope ───────────────────────────────────────────

/// A unified certificate envelope wrapping any certificate payload with
/// integrity fingerprinting and audit metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub certificate_type: String,
    pub format: CertificateFormat,
    pub payload_json: String,
    pub fingerprint: CertificateFingerprint,
    pub sources: Vec<SourceReference>,
    pub issued_at: String,
}

impl Certificate {
    /// Wrap a serializable certificate payload into a unified envelope.
    pub fn wrap<T: Serialize>(
        certificate_type: impl Into<String>,
        payload: &T,
        sources: Vec<SourceReference>,
    ) -> crate::Result<Self> {
        let payload_json = serde_json::to_string_pretty(payload)
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
        let fingerprint = CertificateFingerprint::compute(payload_json.as_bytes());

        Ok(Self {
            certificate_type: certificate_type.into(),
            format: CertificateFormat::Json,
            payload_json,
            fingerprint,
            sources,
            issued_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Serialize the entire certificate to JSON.
    pub fn to_json(&self) -> crate::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))
    }

    /// Serialize the certificate to a human-readable text format.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("═══ Certificate: {} ═══\n", self.certificate_type));
        out.push_str(&format!("Issued: {}\n", self.issued_at));
        out.push_str(&format!("Fingerprint: {}\n", self.fingerprint));
        out.push_str(&format!("Format: {:?}\n", self.format));
        if !self.sources.is_empty() {
            out.push_str("Sources:\n");
            for src in &self.sources {
                out.push_str(&format!("  - {}", src.file_path));
                if let Some((s, e)) = src.line_range {
                    out.push_str(&format!(" (L{}-L{})", s, e));
                }
                out.push('\n');
            }
        }
        out.push_str("──── Payload ────\n");
        out.push_str(&self.payload_json);
        out.push('\n');
        out
    }

    /// Serialize the certificate to a compact binary format.
    /// Uses a simple length-prefixed encoding: [type_len][type][payload_len][payload][fp]
    pub fn to_binary(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let type_bytes = self.certificate_type.as_bytes();
        buf.extend_from_slice(&(type_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(type_bytes);

        let payload_bytes = self.payload_json.as_bytes();
        buf.extend_from_slice(&(payload_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(payload_bytes);

        let fp_bytes = self.fingerprint.hex_digest.as_bytes();
        buf.extend_from_slice(&(fp_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(fp_bytes);

        let ts_bytes = self.issued_at.as_bytes();
        buf.extend_from_slice(&(ts_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(ts_bytes);

        buf
    }

    /// Deserialize a certificate from the compact binary format.
    pub fn from_binary(data: &[u8]) -> crate::Result<Self> {
        let mut pos = 0;

        let cert_type = read_length_prefixed(data, &mut pos)?;
        let payload = read_length_prefixed(data, &mut pos)?;
        let fp_hex = read_length_prefixed(data, &mut pos)?;
        let timestamp = read_length_prefixed(data, &mut pos)?;

        Ok(Self {
            certificate_type: cert_type,
            format: CertificateFormat::Binary,
            payload_json: payload,
            fingerprint: CertificateFingerprint {
                hex_digest: fp_hex,
            },
            sources: Vec::new(),
            issued_at: timestamp,
        })
    }

    /// Verify the fingerprint of the payload.
    pub fn verify_integrity(&self) -> bool {
        let recomputed = CertificateFingerprint::compute(self.payload_json.as_bytes());
        recomputed == self.fingerprint
    }
}

/// Read a length-prefixed string from a byte buffer.
fn read_length_prefixed(data: &[u8], pos: &mut usize) -> crate::Result<String> {
    if *pos + 4 > data.len() {
        return Err(crate::CertificateError::Serialization(
            "unexpected end of binary data".into(),
        ));
    }
    let len = u32::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
    ]) as usize;
    *pos += 4;
    if *pos + len > data.len() {
        return Err(crate::CertificateError::Serialization(
            "binary field extends beyond data".into(),
        ));
    }
    let s = String::from_utf8(data[*pos..*pos + len].to_vec())
        .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
    *pos += len;
    Ok(s)
}

// ─── Export for Audit ───────────────────────────────────────────────────────

/// Export a certificate in the requested format.
pub fn export_for_audit(cert: &Certificate, format: CertificateFormat) -> crate::Result<Vec<u8>> {
    match format {
        CertificateFormat::Json => {
            let json = cert.to_json()?;
            Ok(json.into_bytes())
        }
        CertificateFormat::Text => Ok(cert.to_text().into_bytes()),
        CertificateFormat::Binary => Ok(cert.to_binary()),
    }
}

// ─── Audit Package ──────────────────────────────────────────────────────────

/// An audit package bundles multiple certificates with metadata for
/// regulatory audit submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditPackage {
    pub id: String,
    pub created_at: String,
    pub certificates: Vec<Certificate>,
    pub overall_fingerprint: CertificateFingerprint,
    pub notes: String,
}

impl AuditPackage {
    /// Create an audit package from a set of certificates.
    pub fn new(certificates: Vec<Certificate>, notes: impl Into<String>) -> Self {
        let combined: String = certificates
            .iter()
            .map(|c| c.fingerprint.hex_digest.as_str())
            .collect::<Vec<_>>()
            .join("|");
        let overall_fingerprint = CertificateFingerprint::compute(combined.as_bytes());

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            certificates,
            overall_fingerprint,
            notes: notes.into(),
        }
    }

    /// Serialize the audit package to JSON.
    pub fn to_json(&self) -> crate::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))
    }

    /// Verify that all certificates in the package have valid fingerprints.
    pub fn verify_all(&self) -> Vec<(usize, bool)> {
        self.certificates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.verify_integrity()))
            .collect()
    }

    /// Verify the overall package fingerprint.
    pub fn verify_package_integrity(&self) -> bool {
        let combined: String = self
            .certificates
            .iter()
            .map(|c| c.fingerprint.hex_digest.as_str())
            .collect::<Vec<_>>()
            .join("|");
        let expected = CertificateFingerprint::compute(combined.as_bytes());
        expected == self.overall_fingerprint
    }

    pub fn certificate_count(&self) -> usize {
        self.certificates.len()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_certificate() {
        let payload = serde_json::json!({"status": "compliant"});
        let cert = Certificate::wrap("compliance", &payload, Vec::new()).unwrap();
        assert_eq!(cert.certificate_type, "compliance");
        assert!(!cert.fingerprint.hex_digest.is_empty());
        assert!(cert.verify_integrity());
    }

    #[test]
    fn certificate_to_json_and_back() {
        let cert = Certificate::wrap("test", &"hello", Vec::new()).unwrap();
        let json = cert.to_json().unwrap();
        let parsed: Certificate = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.certificate_type, "test");
        assert_eq!(parsed.fingerprint, cert.fingerprint);
    }

    #[test]
    fn certificate_to_text() {
        let cert = Certificate::wrap("compliance", &"data", Vec::new()).unwrap();
        let text = cert.to_text();
        assert!(text.contains("compliance"));
        assert!(text.contains("Fingerprint"));
        assert!(text.contains("Payload"));
    }

    #[test]
    fn certificate_binary_roundtrip() {
        let cert = Certificate::wrap("pareto", &serde_json::json!({"x": 42}), Vec::new()).unwrap();
        let binary = cert.to_binary();
        let restored = Certificate::from_binary(&binary).unwrap();
        assert_eq!(restored.certificate_type, "pareto");
        assert_eq!(restored.payload_json, cert.payload_json);
        assert_eq!(restored.fingerprint, cert.fingerprint);
    }

    #[test]
    fn export_for_audit_json() {
        let cert = Certificate::wrap("test", &"data", Vec::new()).unwrap();
        let bytes = export_for_audit(&cert, CertificateFormat::Json).unwrap();
        let json_str = String::from_utf8(bytes).unwrap();
        assert!(json_str.contains("test"));
    }

    #[test]
    fn export_for_audit_text() {
        let cert = Certificate::wrap("test", &"data", Vec::new()).unwrap();
        let bytes = export_for_audit(&cert, CertificateFormat::Text).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("Certificate"));
    }

    #[test]
    fn export_for_audit_binary() {
        let cert = Certificate::wrap("test", &"data", Vec::new()).unwrap();
        let bytes = export_for_audit(&cert, CertificateFormat::Binary).unwrap();
        assert!(!bytes.is_empty());
        let restored = Certificate::from_binary(&bytes).unwrap();
        assert_eq!(restored.certificate_type, "test");
    }

    #[test]
    fn audit_package_creation() {
        let c1 = Certificate::wrap("compliance", &"data1", Vec::new()).unwrap();
        let c2 = Certificate::wrap("pareto", &"data2", Vec::new()).unwrap();
        let pkg = AuditPackage::new(vec![c1, c2], "audit notes");
        assert_eq!(pkg.certificate_count(), 2);
        assert!(!pkg.overall_fingerprint.hex_digest.is_empty());
        assert!(pkg.verify_package_integrity());
    }

    #[test]
    fn audit_package_verify_all() {
        let c1 = Certificate::wrap("a", &"d1", Vec::new()).unwrap();
        let c2 = Certificate::wrap("b", &"d2", Vec::new()).unwrap();
        let pkg = AuditPackage::new(vec![c1, c2], "notes");
        let results = pkg.verify_all();
        assert!(results.iter().all(|(_, ok)| *ok));
    }

    #[test]
    fn certificate_format_metadata() {
        assert_eq!(CertificateFormat::Json.file_extension(), "json");
        assert_eq!(CertificateFormat::Text.mime_type(), "text/plain");
        assert_eq!(CertificateFormat::Binary.file_extension(), "bin");
    }

    #[test]
    fn source_reference_creation() {
        let src = SourceReference::new("src/main.rs", "fn main() {}").with_lines(1, 5);
        assert_eq!(src.file_path, "src/main.rs");
        assert_eq!(src.line_range, Some((1, 5)));
        assert!(!src.content_hash.is_empty());
    }

    #[test]
    fn tampered_certificate_fails_integrity() {
        let mut cert = Certificate::wrap("test", &"original", Vec::new()).unwrap();
        assert!(cert.verify_integrity());
        cert.payload_json = "\"tampered\"".into();
        assert!(!cert.verify_integrity());
    }

    #[test]
    fn binary_truncated_fails() {
        let cert = Certificate::wrap("x", &"y", Vec::new()).unwrap();
        let binary = cert.to_binary();
        let truncated = &binary[..5];
        assert!(Certificate::from_binary(truncated).is_err());
    }
}
