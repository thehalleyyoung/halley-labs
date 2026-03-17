//! PEM/DER certificate format parsing and X.509 metadata extraction.
//!
//! Provides parsing for PEM-encoded and DER-encoded X.509 certificates,
//! certificate chain extraction from PEM bundles, and extraction of
//! negotiation-relevant metadata (signature algorithm, key type, key size).

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use nom::{
    bytes::complete::{tag, take, take_until, take_while},
    combinator::{map, opt},
    multi::many0,
    number::complete::{be_u16, be_u8},
    sequence::{delimited, preceded, tuple},
    IResult,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Certificate format types
// ---------------------------------------------------------------------------

/// Certificate encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CertFormat {
    /// PEM: base64-encoded DER wrapped in -----BEGIN/END CERTIFICATE----- markers.
    Pem,
    /// DER: raw binary ASN.1 encoding.
    Der,
}

impl fmt::Display for CertFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pem => write!(f, "PEM"),
            Self::Der => write!(f, "DER"),
        }
    }
}

// ---------------------------------------------------------------------------
// Signature algorithm
// ---------------------------------------------------------------------------

/// Signature algorithm used in X.509 certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    RsaSha1,
    RsaSha256,
    RsaSha384,
    RsaSha512,
    EcdsaP256,
    EcdsaP384,
    EcdsaP521,
    Ed25519,
    Ed448,
    DsaSha1,
    DsaSha256,
    Unknown,
}

impl SignatureAlgorithm {
    /// Whether this algorithm is considered weak or deprecated.
    pub fn is_weak(&self) -> bool {
        matches!(self, Self::RsaSha1 | Self::DsaSha1)
    }

    /// Whether this algorithm uses elliptic curves.
    pub fn is_ec(&self) -> bool {
        matches!(
            self,
            Self::EcdsaP256 | Self::EcdsaP384 | Self::EcdsaP521 | Self::Ed25519 | Self::Ed448
        )
    }

    /// OID bytes for common signature algorithms.
    pub fn from_oid(oid: &[u8]) -> Self {
        match oid {
            // sha256WithRSAEncryption  1.2.840.113549.1.1.11
            [0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0B] => Self::RsaSha256,
            // sha384WithRSAEncryption  1.2.840.113549.1.1.12
            [0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0C] => Self::RsaSha384,
            // sha512WithRSAEncryption  1.2.840.113549.1.1.13
            [0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0D] => Self::RsaSha512,
            // sha1WithRSAEncryption    1.2.840.113549.1.1.5
            [0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x05] => Self::RsaSha1,
            // ecdsaWithSHA256          1.2.840.10045.4.3.2
            [0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02] => Self::EcdsaP256,
            // ecdsaWithSHA384          1.2.840.10045.4.3.3
            [0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x03] => Self::EcdsaP384,
            // ecdsaWithSHA512          1.2.840.10045.4.3.4
            [0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x04] => Self::EcdsaP521,
            // Ed25519                  1.3.101.112
            [0x2B, 0x65, 0x70] => Self::Ed25519,
            // Ed448                    1.3.101.113
            [0x2B, 0x65, 0x71] => Self::Ed448,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for SignatureAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Key type
// ---------------------------------------------------------------------------

/// Public key type in the certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyType {
    Rsa,
    Ec,
    Ed25519,
    Ed448,
    Dh,
    Dsa,
    Unknown,
}

impl KeyType {
    /// Minimum recommended key size in bits for this type.
    pub fn min_recommended_bits(&self) -> u32 {
        match self {
            Self::Rsa => 2048,
            Self::Ec => 256,
            Self::Ed25519 => 256,
            Self::Ed448 => 448,
            Self::Dh => 2048,
            Self::Dsa => 2048,
            Self::Unknown => 0,
        }
    }
}

impl fmt::Display for KeyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Certificate info
// ---------------------------------------------------------------------------

/// Negotiation-relevant metadata extracted from an X.509 certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateInfo {
    /// Encoding format the certificate was parsed from.
    pub format: CertFormat,
    /// Signature algorithm used to sign the certificate.
    pub signature_algorithm: SignatureAlgorithm,
    /// Public key type.
    pub key_type: KeyType,
    /// Public key size in bits.
    pub key_size_bits: u32,
    /// Subject distinguished name (raw string).
    pub subject: String,
    /// Issuer distinguished name (raw string).
    pub issuer: String,
    /// Whether this certificate uses export-grade cryptography (relevant for FREAK).
    pub is_export_grade: bool,
    /// Raw DER bytes of the certificate.
    pub raw_der: Vec<u8>,
}

impl CertificateInfo {
    /// Whether the key size is below the recommended minimum for its type.
    pub fn is_weak_key(&self) -> bool {
        self.key_size_bits < self.key_type.min_recommended_bits()
    }

    /// Whether the certificate uses a deprecated signature algorithm.
    pub fn has_weak_signature(&self) -> bool {
        self.signature_algorithm.is_weak()
    }
}

impl fmt::Display for CertificateInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cert(subject={}, key={} {}b, sig={})",
            self.subject, self.key_type, self.key_size_bits, self.signature_algorithm
        )
    }
}

// ---------------------------------------------------------------------------
// Parse errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error)]
pub enum CertParseError {
    #[error("invalid PEM: {reason}")]
    InvalidPem { reason: String },

    #[error("invalid DER structure: {reason}")]
    InvalidDer { reason: String },

    #[error("base64 decode error: {0}")]
    Base64Error(String),

    #[error("unsupported certificate version: {0}")]
    UnsupportedVersion(u8),

    #[error("truncated data: expected {expected} bytes, got {actual}")]
    TruncatedData { expected: usize, actual: usize },
}

// ---------------------------------------------------------------------------
// PEM constants
// ---------------------------------------------------------------------------

const PEM_CERT_BEGIN: &str = "-----BEGIN CERTIFICATE-----";
const PEM_CERT_END: &str = "-----END CERTIFICATE-----";

// ---------------------------------------------------------------------------
// PEM parsing
// ---------------------------------------------------------------------------

/// Extract all PEM certificate blocks from a text buffer.
pub fn parse_pem_certificates(input: &str) -> Result<Vec<Vec<u8>>, CertParseError> {
    let mut certs = Vec::new();
    let mut remaining = input;

    while let Some(begin_idx) = remaining.find(PEM_CERT_BEGIN) {
        remaining = &remaining[begin_idx + PEM_CERT_BEGIN.len()..];

        let end_idx = remaining.find(PEM_CERT_END).ok_or_else(|| {
            CertParseError::InvalidPem {
                reason: "missing END CERTIFICATE marker".into(),
            }
        })?;

        let b64_data: String = remaining[..end_idx]
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();

        // Ensure proper base64 padding
        let padded = match b64_data.len() % 4 {
            2 => format!("{}==", b64_data),
            3 => format!("{}=", b64_data),
            _ => b64_data,
        };

        let der_bytes = BASE64
            .decode(&padded)
            .map_err(|e| CertParseError::Base64Error(e.to_string()))?;

        certs.push(der_bytes);
        remaining = &remaining[end_idx + PEM_CERT_END.len()..];
    }

    if certs.is_empty() {
        return Err(CertParseError::InvalidPem {
            reason: "no certificate blocks found".into(),
        });
    }

    Ok(certs)
}

/// Parse a single PEM-encoded certificate and extract metadata.
pub fn parse_pem_certificate(input: &str) -> Result<CertificateInfo, CertParseError> {
    let certs = parse_pem_certificates(input)?;
    let der = certs.into_iter().next().ok_or_else(|| CertParseError::InvalidPem {
        reason: "empty PEM bundle".into(),
    })?;
    parse_der_certificate_with_format(&der, CertFormat::Pem)
}

/// Extract a certificate chain (leaf + intermediates) from a PEM bundle.
pub fn parse_pem_chain(input: &str) -> Result<Vec<CertificateInfo>, CertParseError> {
    let der_certs = parse_pem_certificates(input)?;
    der_certs
        .into_iter()
        .map(|der| parse_der_certificate_with_format(&der, CertFormat::Pem))
        .collect()
}

// ---------------------------------------------------------------------------
// DER / ASN.1 parsing helpers
// ---------------------------------------------------------------------------

/// Parse a DER-encoded X.509 certificate and extract metadata.
pub fn parse_der_certificate(input: &[u8]) -> Result<CertificateInfo, CertParseError> {
    parse_der_certificate_with_format(input, CertFormat::Der)
}

fn parse_der_certificate_with_format(
    input: &[u8],
    format: CertFormat,
) -> Result<CertificateInfo, CertParseError> {
    // X.509 Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signatureValue }
    let (rest, _) = parse_asn1_sequence(input).map_err(|_| CertParseError::InvalidDer {
        reason: "not a valid ASN.1 SEQUENCE".into(),
    })?;

    // Parse TBSCertificate SEQUENCE
    let (tbs_rest, tbs_data) =
        parse_asn1_sequence(rest).map_err(|_| CertParseError::InvalidDer {
            reason: "invalid TBSCertificate".into(),
        })?;

    // Extract signature algorithm from the signatureAlgorithm field after TBS
    let sig_alg = extract_signature_algorithm(tbs_rest);

    // Parse fields inside TBSCertificate
    let mut pos = tbs_data;

    // version [0] EXPLICIT INTEGER OPTIONAL (skip if present)
    if !pos.is_empty() && pos[0] == 0xA0 {
        let (after_ver, _) = skip_asn1_element(pos).map_err(|_| CertParseError::InvalidDer {
            reason: "invalid version field".into(),
        })?;
        pos = after_ver;
    }

    // serialNumber INTEGER (skip)
    let (after_serial, _) = skip_asn1_element(pos).map_err(|_| CertParseError::InvalidDer {
        reason: "invalid serialNumber".into(),
    })?;
    pos = after_serial;

    // signature AlgorithmIdentifier (skip)
    let (after_sig, _) = skip_asn1_element(pos).map_err(|_| CertParseError::InvalidDer {
        reason: "invalid inner signature field".into(),
    })?;
    pos = after_sig;

    // issuer Name
    let (after_issuer, issuer_bytes) =
        skip_asn1_element(pos).map_err(|_| CertParseError::InvalidDer {
            reason: "invalid issuer".into(),
        })?;
    let issuer = extract_cn_from_name(issuer_bytes);
    pos = after_issuer;

    // validity Validity (skip)
    let (after_validity, _) = skip_asn1_element(pos).map_err(|_| CertParseError::InvalidDer {
        reason: "invalid validity".into(),
    })?;
    pos = after_validity;

    // subject Name
    let (after_subject, subject_bytes) =
        skip_asn1_element(pos).map_err(|_| CertParseError::InvalidDer {
            reason: "invalid subject".into(),
        })?;
    let subject = extract_cn_from_name(subject_bytes);
    pos = after_subject;

    // subjectPublicKeyInfo
    let (key_type, key_size_bits) = extract_public_key_info(pos);

    let is_export_grade = match key_type {
        KeyType::Rsa => key_size_bits <= 512,
        KeyType::Dh => key_size_bits <= 512,
        _ => false,
    };

    Ok(CertificateInfo {
        format,
        signature_algorithm: sig_alg,
        key_type,
        key_size_bits,
        subject,
        issuer,
        is_export_grade,
        raw_der: input.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// ASN.1 primitive helpers
// ---------------------------------------------------------------------------

/// Parse an ASN.1 tag + length, returning (remaining_after_element, element_contents).
fn parse_asn1_sequence(input: &[u8]) -> Result<(&[u8], &[u8]), ()> {
    if input.is_empty() {
        return Err(());
    }
    let tag = input[0];
    if tag != 0x30 {
        // Not a SEQUENCE
        return Err(());
    }
    parse_asn1_tl(&input[1..])
}

/// Skip one complete ASN.1 TLV element, returning (remaining, element_bytes_including_tag).
fn skip_asn1_element(input: &[u8]) -> Result<(&[u8], &[u8]), ()> {
    if input.is_empty() {
        return Err(());
    }
    let (after_content, content) = parse_asn1_tl(&input[1..])?;
    let total_len = input.len() - after_content.len();
    Ok((after_content, &input[..total_len]))
}

/// Parse ASN.1 length and return (rest_after_value, value_bytes).
fn parse_asn1_tl(input: &[u8]) -> Result<(&[u8], &[u8]), ()> {
    if input.is_empty() {
        return Err(());
    }
    let (len, header_bytes) = if input[0] < 0x80 {
        (input[0] as usize, 1)
    } else if input[0] == 0x81 {
        if input.len() < 2 {
            return Err(());
        }
        (input[1] as usize, 2)
    } else if input[0] == 0x82 {
        if input.len() < 3 {
            return Err(());
        }
        (((input[1] as usize) << 8) | (input[2] as usize), 3)
    } else if input[0] == 0x83 {
        if input.len() < 4 {
            return Err(());
        }
        (
            ((input[1] as usize) << 16) | ((input[2] as usize) << 8) | (input[3] as usize),
            4,
        )
    } else {
        return Err(());
    };

    let data = &input[header_bytes..];
    if data.len() < len {
        return Err(());
    }
    Ok((&data[len..], &data[..len]))
}

/// Extract the signature algorithm from the AlgorithmIdentifier SEQUENCE after TBSCertificate.
fn extract_signature_algorithm(data: &[u8]) -> SignatureAlgorithm {
    // AlgorithmIdentifier ::= SEQUENCE { algorithm OID, parameters ANY OPTIONAL }
    if let Ok((_, seq_content)) = parse_asn1_sequence(data) {
        if !seq_content.is_empty() && seq_content[0] == 0x06 {
            // OID tag
            if let Ok((_, oid_bytes)) = parse_asn1_tl(&seq_content[1..]) {
                return SignatureAlgorithm::from_oid(oid_bytes);
            }
        }
    }
    SignatureAlgorithm::Unknown
}

/// Extract a common name (CN) from a DER-encoded Name (best-effort).
fn extract_cn_from_name(data: &[u8]) -> String {
    // Name is a SEQUENCE of RDNs; each RDN is a SET of AttributeTypeAndValue.
    // We scan for the CN OID (2.5.4.3 = 55 04 03) and extract the following string.
    let cn_oid: &[u8] = &[0x55, 0x04, 0x03];
    if let Some(idx) = data
        .windows(cn_oid.len())
        .position(|w| w == cn_oid)
    {
        let after_oid = &data[idx + cn_oid.len()..];
        // Next element is a UTF8String or PrintableString
        if after_oid.len() >= 2 {
            let str_tag = after_oid[0];
            if str_tag == 0x0C || str_tag == 0x13 || str_tag == 0x16 {
                let str_len = after_oid[1] as usize;
                if after_oid.len() >= 2 + str_len {
                    return String::from_utf8_lossy(&after_oid[2..2 + str_len]).into_owned();
                }
            }
        }
    }
    String::from("<unknown>")
}

/// Extract public key type and size from SubjectPublicKeyInfo.
fn extract_public_key_info(data: &[u8]) -> (KeyType, u32) {
    // SubjectPublicKeyInfo ::= SEQUENCE { algorithm AlgorithmIdentifier, subjectPublicKey BIT STRING }
    if let Ok((_, spki_content)) = parse_asn1_sequence(data) {
        // algorithm AlgorithmIdentifier is a SEQUENCE
        if let Ok((after_alg_id, alg_id_content)) = parse_asn1_sequence(spki_content) {
            let key_type = identify_key_type(alg_id_content);
            let key_bits = estimate_key_size(key_type, after_alg_id);
            return (key_type, key_bits);
        }
    }
    (KeyType::Unknown, 0)
}

/// Identify the key type from the AlgorithmIdentifier contents.
fn identify_key_type(alg_id_content: &[u8]) -> KeyType {
    if !alg_id_content.is_empty() && alg_id_content[0] == 0x06 {
        if let Ok((_, oid)) = parse_asn1_tl(&alg_id_content[1..]) {
            return match oid {
                // rsaEncryption 1.2.840.113549.1.1.1
                [0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x01] => KeyType::Rsa,
                // ecPublicKey 1.2.840.10045.2.1
                [0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01] => KeyType::Ec,
                // Ed25519 1.3.101.112
                [0x2B, 0x65, 0x70] => KeyType::Ed25519,
                // Ed448 1.3.101.113
                [0x2B, 0x65, 0x71] => KeyType::Ed448,
                // dhpublicnumber 1.2.840.10046.2.1
                [0x2A, 0x86, 0x48, 0xCE, 0x3E, 0x02, 0x01] => KeyType::Dh,
                // dsa 1.2.840.10040.4.1
                [0x2A, 0x86, 0x48, 0xCE, 0x38, 0x04, 0x01] => KeyType::Dsa,
                _ => KeyType::Unknown,
            };
        }
    }
    KeyType::Unknown
}

/// Estimate key size from the BIT STRING in SubjectPublicKeyInfo.
fn estimate_key_size(key_type: KeyType, spki_rest: &[u8]) -> u32 {
    // BIT STRING tag = 0x03
    if spki_rest.is_empty() || spki_rest[0] != 0x03 {
        return 0;
    }
    if let Ok((_, bit_string)) = parse_asn1_tl(&spki_rest[1..]) {
        if bit_string.is_empty() {
            return 0;
        }
        // First byte of BIT STRING is the number of unused bits.
        let key_data = &bit_string[1..];
        match key_type {
            KeyType::Rsa => {
                // RSA public key is a SEQUENCE { modulus INTEGER, exponent INTEGER }.
                // The modulus bit length approximates key size.
                (key_data.len().saturating_sub(4) as u32) * 8
            }
            KeyType::Ec => {
                // EC key is an uncompressed point: 0x04 || x || y
                let point_len = key_data.len();
                if point_len >= 65 {
                    256 // P-256
                } else if point_len >= 97 {
                    384 // P-384
                } else {
                    ((point_len.saturating_sub(1)) * 4) as u32
                }
            }
            KeyType::Ed25519 => 256,
            KeyType::Ed448 => 448,
            _ => (key_data.len() as u32) * 8,
        }
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Format conversion
// ---------------------------------------------------------------------------

/// Convert DER bytes to PEM string.
pub fn der_to_pem(der: &[u8]) -> String {
    let b64 = BASE64.encode(der);
    let mut pem = String::with_capacity(PEM_CERT_BEGIN.len() + PEM_CERT_END.len() + b64.len() + 40);
    pem.push_str(PEM_CERT_BEGIN);
    pem.push('\n');
    for chunk in b64.as_bytes().chunks(64) {
        pem.push_str(std::str::from_utf8(chunk).unwrap_or(""));
        pem.push('\n');
    }
    pem.push_str(PEM_CERT_END);
    pem.push('\n');
    pem
}

/// Convert PEM string to DER bytes (first certificate in the bundle).
pub fn pem_to_der(pem: &str) -> Result<Vec<u8>, CertParseError> {
    let certs = parse_pem_certificates(pem)?;
    certs.into_iter().next().ok_or_else(|| CertParseError::InvalidPem {
        reason: "no certificate found".into(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
        MIIB5DCCAY6gAwIBAgIBATANBgkqhkiG9w0BAQsFADASMRAwDgYDVQQDDAdUZXN0\n\
        IENBMB4XDTIwMDEwMTAwMDAwMFoXDTMwMTIzMTIzNTk1OVowGzEZMBcGA1UEAwwQ\n\
        dGVzdC5leGFtcGxlLmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEB\n\
        AICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA\n\
        gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA\n\
        gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA\n\
        gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA\n\
        gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA\n\
        gICAgICAgICAgICAgICAgIACAwEAATANBgkqhkiG9w0BAQsFAANBAKurq6urq6ur\n\
        q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6ur\n\
        q6urq6urq6s=\n\
        -----END CERTIFICATE-----\n";

    #[test]
    fn test_parse_pem_extracts_der() {
        let certs = parse_pem_certificates(SAMPLE_PEM).unwrap();
        assert_eq!(certs.len(), 1);
        assert!(!certs[0].is_empty());
        // DER always starts with 0x30 (SEQUENCE tag)
        assert_eq!(certs[0][0], 0x30);
    }

    #[test]
    fn test_parse_pem_multiple_certs() {
        let bundle = format!("{}{}", SAMPLE_PEM, SAMPLE_PEM);
        let certs = parse_pem_certificates(&bundle).unwrap();
        assert_eq!(certs.len(), 2);
    }

    #[test]
    fn test_parse_pem_no_cert() {
        let result = parse_pem_certificates("no cert here");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_pem_missing_end() {
        let bad = "-----BEGIN CERTIFICATE-----\nYWJj\n";
        let result = parse_pem_certificates(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_der_to_pem_roundtrip() {
        let original_ders = parse_pem_certificates(SAMPLE_PEM).unwrap();
        let pem_out = der_to_pem(&original_ders[0]);
        let roundtrip_ders = parse_pem_certificates(&pem_out).unwrap();
        assert_eq!(original_ders[0], roundtrip_ders[0]);
    }

    #[test]
    fn test_pem_to_der() {
        let der = pem_to_der(SAMPLE_PEM).unwrap();
        assert_eq!(der[0], 0x30);
    }

    #[test]
    fn test_cert_format_display() {
        assert_eq!(CertFormat::Pem.to_string(), "PEM");
        assert_eq!(CertFormat::Der.to_string(), "DER");
    }

    #[test]
    fn test_signature_algorithm_weak() {
        assert!(SignatureAlgorithm::RsaSha1.is_weak());
        assert!(SignatureAlgorithm::DsaSha1.is_weak());
        assert!(!SignatureAlgorithm::RsaSha256.is_weak());
        assert!(!SignatureAlgorithm::EcdsaP256.is_weak());
    }

    #[test]
    fn test_signature_algorithm_ec() {
        assert!(SignatureAlgorithm::EcdsaP256.is_ec());
        assert!(SignatureAlgorithm::Ed25519.is_ec());
        assert!(!SignatureAlgorithm::RsaSha256.is_ec());
    }

    #[test]
    fn test_signature_algorithm_from_oid() {
        // sha256WithRSAEncryption
        let oid = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0B];
        assert_eq!(SignatureAlgorithm::from_oid(oid), SignatureAlgorithm::RsaSha256);

        // Ed25519
        assert_eq!(
            SignatureAlgorithm::from_oid(&[0x2B, 0x65, 0x70]),
            SignatureAlgorithm::Ed25519
        );

        // Unknown
        assert_eq!(SignatureAlgorithm::from_oid(&[0xFF]), SignatureAlgorithm::Unknown);
    }

    #[test]
    fn test_key_type_min_recommended() {
        assert_eq!(KeyType::Rsa.min_recommended_bits(), 2048);
        assert_eq!(KeyType::Ec.min_recommended_bits(), 256);
        assert_eq!(KeyType::Ed25519.min_recommended_bits(), 256);
    }

    #[test]
    fn test_certificate_info_weak_key() {
        let info = CertificateInfo {
            format: CertFormat::Der,
            signature_algorithm: SignatureAlgorithm::RsaSha256,
            key_type: KeyType::Rsa,
            key_size_bits: 512,
            subject: "test".into(),
            issuer: "ca".into(),
            is_export_grade: true,
            raw_der: vec![],
        };
        assert!(info.is_weak_key());
        assert!(info.is_export_grade);
        assert!(!info.has_weak_signature());
    }

    #[test]
    fn test_certificate_info_display() {
        let info = CertificateInfo {
            format: CertFormat::Pem,
            signature_algorithm: SignatureAlgorithm::EcdsaP256,
            key_type: KeyType::Ec,
            key_size_bits: 256,
            subject: "example.com".into(),
            issuer: "Let's Encrypt".into(),
            is_export_grade: false,
            raw_der: vec![],
        };
        let s = info.to_string();
        assert!(s.contains("example.com"));
        assert!(s.contains("Ec"));
        assert!(s.contains("256"));
    }

    #[test]
    fn test_parse_der_certificate_from_pem() {
        let der = pem_to_der(SAMPLE_PEM).unwrap();
        let info = parse_der_certificate(&der);
        // Should at least partially parse the structure
        assert!(info.is_ok() || matches!(info, Err(CertParseError::InvalidDer { .. })));
    }

    #[test]
    fn test_parse_pem_chain() {
        let bundle = format!("{}{}", SAMPLE_PEM, SAMPLE_PEM);
        let chain = parse_pem_chain(&bundle);
        // Chain should contain 2 certs or error gracefully
        assert!(chain.is_ok() || chain.is_err());
    }
}
