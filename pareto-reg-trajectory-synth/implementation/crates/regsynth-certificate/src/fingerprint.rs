//! SHA-256 fingerprinting, integrity verification, and certificate chains.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

// ─── Certificate Fingerprint ────────────────────────────────────────────────

/// A SHA-256 fingerprint of certificate content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CertificateFingerprint {
    pub hex_digest: String,
}

impl CertificateFingerprint {
    /// Compute a fingerprint from arbitrary byte content.
    pub fn compute(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        Self {
            hex_digest: hex::encode(hasher.finalize()),
        }
    }

    /// Compute a fingerprint from multiple data fields concatenated with
    /// length-prefixing to avoid ambiguity.
    pub fn compute_fields(fields: &[&[u8]]) -> Self {
        let mut hasher = Sha256::new();
        for field in fields {
            let len = field.len() as u64;
            hasher.update(len.to_le_bytes());
            hasher.update(field);
        }
        Self {
            hex_digest: hex::encode(hasher.finalize()),
        }
    }

    /// Compute a fingerprint from a map of key-value pairs (sorted by key
    /// for determinism).
    pub fn compute_map(entries: &HashMap<String, String>) -> Self {
        let mut keys: Vec<&String> = entries.keys().collect();
        keys.sort();
        let mut hasher = Sha256::new();
        for key in keys {
            let val = &entries[key];
            hasher.update((key.len() as u32).to_le_bytes());
            hasher.update(key.as_bytes());
            hasher.update((val.len() as u32).to_le_bytes());
            hasher.update(val.as_bytes());
        }
        Self {
            hex_digest: hex::encode(hasher.finalize()),
        }
    }

    /// Verify that the fingerprint matches the given data.
    pub fn verify(&self, data: &[u8]) -> bool {
        let expected = Self::compute(data);
        self.hex_digest == expected.hex_digest
    }

    /// Verify that the fingerprint matches the given fields.
    pub fn verify_fields(&self, fields: &[&[u8]]) -> bool {
        let expected = Self::compute_fields(fields);
        self.hex_digest == expected.hex_digest
    }

    /// Return the first `n` hex characters (for display).
    pub fn short(&self, n: usize) -> &str {
        &self.hex_digest[..n.min(self.hex_digest.len())]
    }
}

impl fmt::Display for CertificateFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", &self.hex_digest[..16.min(self.hex_digest.len())])
    }
}

// ─── Certificate Signature ──────────────────────────────────────────────────

/// An HMAC-SHA256 signature for certificate authentication.
/// Uses a shared secret key to produce a message authentication code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateSignature {
    pub algorithm: String,
    pub hex_signature: String,
}

const HMAC_BLOCK_SIZE: usize = 64;

impl CertificateSignature {
    /// Compute an HMAC-SHA256 signature over the data with the given key.
    pub fn sign(key: &[u8], data: &[u8]) -> Self {
        let signature = hmac_sha256(key, data);
        Self {
            algorithm: "HMAC-SHA256".to_string(),
            hex_signature: hex::encode(signature),
        }
    }

    /// Verify the signature against the data and key.
    pub fn verify(&self, key: &[u8], data: &[u8]) -> bool {
        let expected = hmac_sha256(key, data);
        let expected_hex = hex::encode(expected);
        constant_time_eq(self.hex_signature.as_bytes(), expected_hex.as_bytes())
    }
}

/// HMAC-SHA256 implementation using SHA-256 as the underlying hash.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let actual_key = if key.len() > HMAC_BLOCK_SIZE {
        let mut h = Sha256::new();
        h.update(key);
        h.finalize().to_vec()
    } else {
        key.to_vec()
    };

    let mut padded_key = vec![0u8; HMAC_BLOCK_SIZE];
    padded_key[..actual_key.len()].copy_from_slice(&actual_key);

    let mut i_key_pad = vec![0u8; HMAC_BLOCK_SIZE];
    let mut o_key_pad = vec![0u8; HMAC_BLOCK_SIZE];
    for i in 0..HMAC_BLOCK_SIZE {
        i_key_pad[i] = padded_key[i] ^ 0x36;
        o_key_pad[i] = padded_key[i] ^ 0x5c;
    }

    let mut inner_hasher = Sha256::new();
    inner_hasher.update(&i_key_pad);
    inner_hasher.update(data);
    let inner_hash = inner_hasher.finalize();

    let mut outer_hasher = Sha256::new();
    outer_hasher.update(&o_key_pad);
    outer_hasher.update(inner_hash);
    outer_hasher.finalize().to_vec()
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ─── Certificate Chain ──────────────────────────────────────────────────────

/// An entry in a certificate chain linking certificates together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainEntry {
    pub certificate_id: String,
    pub fingerprint: CertificateFingerprint,
    pub previous_fingerprint: Option<CertificateFingerprint>,
    pub timestamp: String,
    pub description: String,
}

/// A chain of certificates where each entry references the fingerprint of the
/// previous entry, forming a tamper-evident linked list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateChain {
    pub entries: Vec<ChainEntry>,
}

impl CertificateChain {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append a new certificate to the chain.
    pub fn append(
        &mut self,
        certificate_id: &str,
        content: &[u8],
        description: &str,
    ) -> CertificateFingerprint {
        let fingerprint = CertificateFingerprint::compute(content);
        let previous_fingerprint = self.entries.last().map(|e| e.fingerprint.clone());
        self.entries.push(ChainEntry {
            certificate_id: certificate_id.to_string(),
            fingerprint: fingerprint.clone(),
            previous_fingerprint,
            timestamp: chrono::Utc::now().to_rfc3339(),
            description: description.to_string(),
        });
        fingerprint
    }

    /// Verify the integrity of the entire chain:
    /// - Each entry's previous_fingerprint matches the preceding entry's fingerprint.
    /// - The first entry has no previous_fingerprint.
    pub fn verify_chain(&self) -> std::result::Result<(), crate::CertificateError> {
        if self.entries.is_empty() {
            return Ok(());
        }

        if self.entries[0].previous_fingerprint.is_some() {
            return Err(crate::CertificateError::ChainVerification {
                index: 0,
                reason: "first entry must not have a previous fingerprint".into(),
            });
        }

        for i in 1..self.entries.len() {
            match (&self.entries[i].previous_fingerprint, &self.entries[i - 1].fingerprint) {
                (Some(prev), expected) if prev == expected => {}
                (Some(prev), expected) => {
                    return Err(crate::CertificateError::ChainVerification {
                        index: i,
                        reason: format!(
                            "previous fingerprint {} does not match entry {} fingerprint {}",
                            prev.hex_digest,
                            i - 1,
                            expected.hex_digest
                        ),
                    });
                }
                (None, _) => {
                    return Err(crate::CertificateError::ChainVerification {
                        index: i,
                        reason: "non-first entry must have a previous fingerprint".into(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the fingerprint of the last entry (chain head).
    pub fn head_fingerprint(&self) -> Option<&CertificateFingerprint> {
        self.entries.last().map(|e| &e.fingerprint)
    }
}

impl Default for CertificateChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Convenience Functions ──────────────────────────────────────────────────

/// Compute a SHA-256 fingerprint of a string.
pub fn compute_fingerprint(content: &str) -> String {
    CertificateFingerprint::compute(content.as_bytes()).hex_digest
}

/// Verify content against a known fingerprint string.
pub fn verify_integrity(content: &str, expected_hex: &str) -> bool {
    let fp = CertificateFingerprint::compute(content.as_bytes());
    fp.hex_digest == expected_hex
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_deterministic() {
        let fp1 = CertificateFingerprint::compute(b"hello world");
        let fp2 = CertificateFingerprint::compute(b"hello world");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprint_different_input() {
        let fp1 = CertificateFingerprint::compute(b"hello");
        let fp2 = CertificateFingerprint::compute(b"world");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn fingerprint_verify() {
        let fp = CertificateFingerprint::compute(b"test data");
        assert!(fp.verify(b"test data"));
        assert!(!fp.verify(b"wrong data"));
    }

    #[test]
    fn fingerprint_fields_length_prefix() {
        // "ab" + "cd" should differ from "a" + "bcd" due to length prefixing
        let fp1 = CertificateFingerprint::compute_fields(&[b"ab", b"cd"]);
        let fp2 = CertificateFingerprint::compute_fields(&[b"a", b"bcd"]);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn fingerprint_map_deterministic() {
        let mut m = HashMap::new();
        m.insert("b".into(), "2".into());
        m.insert("a".into(), "1".into());
        let fp1 = CertificateFingerprint::compute_map(&m);

        let mut m2 = HashMap::new();
        m2.insert("a".into(), "1".into());
        m2.insert("b".into(), "2".into());
        let fp2 = CertificateFingerprint::compute_map(&m2);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn signature_sign_and_verify() {
        let key = b"my-secret-key";
        let data = b"certificate content";
        let sig = CertificateSignature::sign(key, data);
        assert!(sig.verify(key, data));
        assert!(!sig.verify(b"wrong-key", data));
        assert!(!sig.verify(key, b"tampered content"));
    }

    #[test]
    fn chain_append_and_verify() {
        let mut chain = CertificateChain::new();
        chain.append("cert-1", b"first certificate", "compliance");
        chain.append("cert-2", b"second certificate", "infeasibility");
        chain.append("cert-3", b"third certificate", "pareto");
        assert_eq!(chain.len(), 3);
        assert!(chain.verify_chain().is_ok());
    }

    #[test]
    fn chain_tampered_fails() {
        let mut chain = CertificateChain::new();
        chain.append("cert-1", b"first", "a");
        chain.append("cert-2", b"second", "b");
        // Tamper with the previous fingerprint
        chain.entries[1].previous_fingerprint =
            Some(CertificateFingerprint::compute(b"wrong"));
        assert!(chain.verify_chain().is_err());
    }

    #[test]
    fn chain_empty_valid() {
        let chain = CertificateChain::new();
        assert!(chain.verify_chain().is_ok());
    }

    #[test]
    fn convenience_functions() {
        let content = "some audit data";
        let fp = compute_fingerprint(content);
        assert!(verify_integrity(content, &fp));
        assert!(!verify_integrity("tampered", &fp));
    }

    #[test]
    fn fingerprint_short() {
        let fp = CertificateFingerprint::compute(b"test");
        assert_eq!(fp.short(8).len(), 8);
    }

    #[test]
    fn fingerprint_display() {
        let fp = CertificateFingerprint::compute(b"display test");
        let s = format!("{}", fp);
        assert!(s.starts_with("sha256:"));
    }
}
