//! OpenSSL-rs (rust-openssl) pattern analyzer for TLS negotiation synthesis.
//!
//! Detects SslConnector/SslAcceptor configuration, protocol version
//! constraints, cipher list configuration, and unsafe patterns in Rust
//! source code that uses the `openssl` crate. Identifies potential
//! downgrade attack surfaces from configuration patterns.

use serde::{Deserialize, Serialize};
use std::fmt;

// Re-use SourceLocation from rustls_patterns to keep types consistent.
pub use crate::rustls_patterns::SourceLocation;

// ---------------------------------------------------------------------------
// OpenSSL pattern types
// ---------------------------------------------------------------------------

/// Category of detected openssl-rs usage pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpensslPatternType {
    /// SslConnector or SslAcceptor builder configuration.
    ConnectorConfig,
    /// Protocol version constraints (set_min_proto_version / set_max_proto_version).
    VersionConstraint,
    /// Cipher list configuration (set_cipher_list / set_ciphersuites).
    CipherListConfig,
    /// Certificate verification settings.
    CertificateVerification,
    /// Unsafe or insecure configuration patterns.
    UnsafeConfig,
}

impl fmt::Display for OpensslPatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectorConfig => write!(f, "ConnectorConfig"),
            Self::VersionConstraint => write!(f, "VersionConstraint"),
            Self::CipherListConfig => write!(f, "CipherListConfig"),
            Self::CertificateVerification => write!(f, "CertificateVerification"),
            Self::UnsafeConfig => write!(f, "UnsafeConfig"),
        }
    }
}

// ---------------------------------------------------------------------------
// Security level for openssl pattern analysis
// ---------------------------------------------------------------------------

/// Security assessment of a detected openssl-rs pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpensslSecurityLevel {
    /// Pattern follows best practices.
    Secure,
    /// Pattern is acceptable but not ideal.
    Acceptable,
    /// Pattern may weaken security.
    Degraded,
    /// Pattern is dangerous (e.g., allowing SSLv3).
    Dangerous,
}

impl fmt::Display for OpensslSecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Detected pattern
// ---------------------------------------------------------------------------

/// A detected openssl-rs usage pattern with security analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpensslPattern {
    /// Category of the pattern.
    pub pattern_type: OpensslPatternType,
    /// Where the pattern was found.
    pub location: SourceLocation,
    /// Cipher strings referenced in this pattern.
    pub cipher_strings: Vec<String>,
    /// Security assessment.
    pub security_level: OpensslSecurityLevel,
    /// Human-readable description of what was detected.
    pub description: String,
    /// Potential downgrade risk if any.
    pub downgrade_risk: Option<String>,
}

impl fmt::Display for OpensslPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} at {} ({})",
            self.security_level, self.pattern_type, self.location, self.description
        )
    }
}

// ---------------------------------------------------------------------------
// Known openssl-rs API patterns
// ---------------------------------------------------------------------------

struct ApiSignature {
    signature: &'static str,
    pattern_type: OpensslPatternType,
    base_security: OpensslSecurityLevel,
    description: &'static str,
}

const OPENSSL_PATTERNS: &[ApiSignature] = &[
    ApiSignature {
        signature: "SslConnector::builder",
        pattern_type: OpensslPatternType::ConnectorConfig,
        base_security: OpensslSecurityLevel::Acceptable,
        description: "SslConnector builder initialization",
    },
    ApiSignature {
        signature: "SslAcceptor::mozilla_intermediate",
        pattern_type: OpensslPatternType::ConnectorConfig,
        base_security: OpensslSecurityLevel::Secure,
        description: "SslAcceptor with Mozilla intermediate compatibility",
    },
    ApiSignature {
        signature: "SslAcceptor::mozilla_modern",
        pattern_type: OpensslPatternType::ConnectorConfig,
        base_security: OpensslSecurityLevel::Secure,
        description: "SslAcceptor with Mozilla modern compatibility",
    },
    ApiSignature {
        signature: "set_min_proto_version",
        pattern_type: OpensslPatternType::VersionConstraint,
        base_security: OpensslSecurityLevel::Acceptable,
        description: "Minimum protocol version constraint",
    },
    ApiSignature {
        signature: "set_max_proto_version",
        pattern_type: OpensslPatternType::VersionConstraint,
        base_security: OpensslSecurityLevel::Acceptable,
        description: "Maximum protocol version constraint",
    },
    ApiSignature {
        signature: "set_cipher_list",
        pattern_type: OpensslPatternType::CipherListConfig,
        base_security: OpensslSecurityLevel::Acceptable,
        description: "Custom cipher list configuration (TLS 1.2 and below)",
    },
    ApiSignature {
        signature: "set_ciphersuites",
        pattern_type: OpensslPatternType::CipherListConfig,
        base_security: OpensslSecurityLevel::Acceptable,
        description: "TLS 1.3 ciphersuite configuration",
    },
    ApiSignature {
        signature: "set_verify(SslVerifyMode::NONE)",
        pattern_type: OpensslPatternType::CertificateVerification,
        base_security: OpensslSecurityLevel::Dangerous,
        description: "Certificate verification disabled",
    },
    ApiSignature {
        signature: "set_verify_callback",
        pattern_type: OpensslPatternType::CertificateVerification,
        base_security: OpensslSecurityLevel::Degraded,
        description: "Custom verification callback replacing default logic",
    },
    ApiSignature {
        signature: "SslVersion::SSL3",
        pattern_type: OpensslPatternType::UnsafeConfig,
        base_security: OpensslSecurityLevel::Dangerous,
        description: "Reference to SSLv3 protocol version",
    },
    ApiSignature {
        signature: "SslVersion::TLS1",
        pattern_type: OpensslPatternType::UnsafeConfig,
        base_security: OpensslSecurityLevel::Degraded,
        description: "Reference to deprecated TLS 1.0 protocol version",
    },
    ApiSignature {
        signature: "SSL_OP_NO_TLSV1_3",
        pattern_type: OpensslPatternType::UnsafeConfig,
        base_security: OpensslSecurityLevel::Degraded,
        description: "TLS 1.3 explicitly disabled via SSL options",
    },
    ApiSignature {
        signature: "set_options",
        pattern_type: OpensslPatternType::ConnectorConfig,
        base_security: OpensslSecurityLevel::Acceptable,
        description: "SSL context options configuration",
    },
];

/// Well-known insecure cipher strings.
const INSECURE_CIPHER_STRINGS: &[&str] = &[
    "eNULL", "aNULL", "NULL", "EXPORT", "DES", "RC4", "MD5",
    "LOW", "EXP", "ADH", "AECDH", "SSLv3",
];

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Analyzes Rust source code for openssl-rs configuration patterns.
pub struct OpensslAnalyzer {
    patterns: Vec<OpensslPattern>,
}

impl OpensslAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Analyze source code text and return all detected patterns.
    pub fn analyze(&mut self, source: &str, file_path: &str) -> &[OpensslPattern] {
        self.patterns.clear();

        for (line_num, line) in source.lines().enumerate() {
            let line_1based = line_num + 1;

            for api_sig in OPENSSL_PATTERNS {
                if let Some(col) = line.find(api_sig.signature) {
                    let cipher_strings = extract_cipher_strings(line);
                    let security = refine_openssl_security(api_sig, line);
                    let downgrade_risk = assess_openssl_downgrade_risk(api_sig, line);

                    self.patterns.push(OpensslPattern {
                        pattern_type: api_sig.pattern_type,
                        location: SourceLocation {
                            file: file_path.to_string(),
                            line: line_1based,
                            column: col,
                            snippet: line.trim().to_string(),
                        },
                        cipher_strings,
                        security_level: security,
                        description: api_sig.description.to_string(),
                        downgrade_risk,
                    });
                }
            }
        }

        &self.patterns
    }

    /// Return accumulated patterns.
    pub fn patterns(&self) -> &[OpensslPattern] {
        &self.patterns
    }

    /// Check if any dangerous patterns were detected.
    pub fn has_dangerous_patterns(&self) -> bool {
        self.patterns
            .iter()
            .any(|p| p.security_level == OpensslSecurityLevel::Dangerous)
    }

    /// Return patterns that represent potential downgrade attack surfaces.
    pub fn downgrade_surfaces(&self) -> Vec<&OpensslPattern> {
        self.patterns
            .iter()
            .filter(|p| p.downgrade_risk.is_some())
            .collect()
    }

    /// Return patterns involving insecure cipher configurations.
    pub fn insecure_cipher_patterns(&self) -> Vec<&OpensslPattern> {
        self.patterns
            .iter()
            .filter(|p| {
                p.cipher_strings
                    .iter()
                    .any(|s| INSECURE_CIPHER_STRINGS.iter().any(|&bad| s.contains(bad)))
            })
            .collect()
    }
}

/// Extract cipher strings from a source line (quoted strings in set_cipher_list calls).
fn extract_cipher_strings(line: &str) -> Vec<String> {
    let mut results = Vec::new();
    let mut remaining = line;

    while let Some(start) = remaining.find('"') {
        let after_quote = &remaining[start + 1..];
        if let Some(end) = after_quote.find('"') {
            let s = &after_quote[..end];
            // Split on ':' as OpenSSL cipher strings are colon-delimited
            for part in s.split(':') {
                if !part.is_empty() {
                    results.push(part.to_string());
                }
            }
            remaining = &after_quote[end + 1..];
        } else {
            break;
        }
    }

    results
}

/// Refine security level based on context.
fn refine_openssl_security(api_sig: &ApiSignature, line: &str) -> OpensslSecurityLevel {
    // SSLv3 reference is always dangerous
    if line.contains("SSL3") || line.contains("SSLv3") {
        return OpensslSecurityLevel::Dangerous;
    }

    // Verify NONE is always dangerous
    if line.contains("NONE") && line.contains("set_verify") {
        return OpensslSecurityLevel::Dangerous;
    }

    // Check for insecure cipher strings
    for &bad in INSECURE_CIPHER_STRINGS {
        if line.contains(bad) {
            return OpensslSecurityLevel::Dangerous;
        }
    }

    api_sig.base_security
}

/// Assess downgrade risk from an openssl pattern.
fn assess_openssl_downgrade_risk(api_sig: &ApiSignature, line: &str) -> Option<String> {
    match api_sig.pattern_type {
        OpensslPatternType::VersionConstraint => {
            if line.contains("SSL3") || line.contains("SSLv3") {
                Some("Allowing SSLv3 enables POODLE and other downgrade attacks".into())
            } else if line.contains("TLS1") && !line.contains("TLS1_2") && !line.contains("TLS1_3") {
                Some("Allowing TLS 1.0/1.1 enables legacy protocol attacks".into())
            } else {
                None
            }
        }
        OpensslPatternType::UnsafeConfig => {
            if line.contains("SSL3") {
                Some("SSLv3 reference enables POODLE downgrade attack".into())
            } else if line.contains("NO_TLSV1_3") {
                Some("Disabling TLS 1.3 removes downgrade protection sentinels".into())
            } else {
                None
            }
        }
        OpensslPatternType::CipherListConfig => {
            if INSECURE_CIPHER_STRINGS.iter().any(|&bad| line.contains(bad)) {
                Some("Insecure cipher strings enable export/NULL cipher attacks".into())
            } else {
                None
            }
        }
        OpensslPatternType::CertificateVerification => {
            if line.contains("NONE") {
                Some("Disabled certificate verification enables MITM attacks".into())
            } else {
                None
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_OPENSSL_CODE: &str = r#"
use openssl::ssl::{SslConnector, SslMethod, SslVersion};

fn build_connector() -> SslConnector {
    let mut builder = SslConnector::builder(SslMethod::tls()).unwrap();
    builder.set_min_proto_version(Some(SslVersion::TLS1_2)).unwrap();
    builder.set_max_proto_version(Some(SslVersion::TLS1_3)).unwrap();
    builder.set_cipher_list("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM").unwrap();
    builder.build()
}
"#;

    const INSECURE_CODE: &str = r#"
let mut builder = SslConnector::builder(SslMethod::tls()).unwrap();
builder.set_min_proto_version(Some(SslVersion::SSL3)).unwrap();
builder.set_cipher_list("ALL:EXPORT:eNULL").unwrap();
builder.set_verify(SslVerifyMode::NONE);
"#;

    #[test]
    fn test_detect_connector_config() {
        let mut analyzer = OpensslAnalyzer::new();
        let patterns = analyzer.analyze(SAMPLE_OPENSSL_CODE, "test.rs");

        let connector: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == OpensslPatternType::ConnectorConfig)
            .collect();

        assert!(!connector.is_empty());
    }

    #[test]
    fn test_detect_version_constraints() {
        let mut analyzer = OpensslAnalyzer::new();
        let patterns = analyzer.analyze(SAMPLE_OPENSSL_CODE, "test.rs");

        let version_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == OpensslPatternType::VersionConstraint)
            .collect();

        assert!(version_patterns.len() >= 2); // min + max
    }

    #[test]
    fn test_detect_cipher_list() {
        let mut analyzer = OpensslAnalyzer::new();
        let patterns = analyzer.analyze(SAMPLE_OPENSSL_CODE, "test.rs");

        let cipher_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == OpensslPatternType::CipherListConfig)
            .collect();

        assert!(!cipher_patterns.is_empty());
        assert!(!cipher_patterns[0].cipher_strings.is_empty());
    }

    #[test]
    fn test_detect_insecure_sslv3() {
        let mut analyzer = OpensslAnalyzer::new();
        analyzer.analyze(INSECURE_CODE, "bad.rs");

        assert!(analyzer.has_dangerous_patterns());

        let dangerous: Vec<_> = analyzer
            .patterns()
            .iter()
            .filter(|p| p.security_level == OpensslSecurityLevel::Dangerous)
            .collect();

        assert!(!dangerous.is_empty());
    }

    #[test]
    fn test_detect_verify_none() {
        let mut analyzer = OpensslAnalyzer::new();
        let patterns = analyzer.analyze(INSECURE_CODE, "bad.rs");

        let verify_none: Vec<_> = patterns
            .iter()
            .filter(|p| {
                p.pattern_type == OpensslPatternType::CertificateVerification
                    && p.security_level == OpensslSecurityLevel::Dangerous
            })
            .collect();

        assert!(!verify_none.is_empty());
    }

    #[test]
    fn test_downgrade_surfaces_sslv3() {
        let mut analyzer = OpensslAnalyzer::new();
        analyzer.analyze(INSECURE_CODE, "bad.rs");

        let surfaces = analyzer.downgrade_surfaces();
        assert!(!surfaces.is_empty());

        let has_poodle_warning = surfaces
            .iter()
            .any(|p| p.downgrade_risk.as_deref().unwrap_or("").contains("POODLE"));
        assert!(has_poodle_warning);
    }

    #[test]
    fn test_insecure_cipher_patterns() {
        let mut analyzer = OpensslAnalyzer::new();
        analyzer.analyze(INSECURE_CODE, "bad.rs");

        let insecure = analyzer.insecure_cipher_patterns();
        assert!(!insecure.is_empty());
    }

    #[test]
    fn test_empty_source() {
        let mut analyzer = OpensslAnalyzer::new();
        let patterns = analyzer.analyze("", "empty.rs");
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_pattern_display() {
        let p = OpensslPattern {
            pattern_type: OpensslPatternType::CipherListConfig,
            location: SourceLocation {
                file: "test.rs".into(),
                line: 5,
                column: 12,
                snippet: "set_cipher_list(\"HIGH\")".into(),
            },
            cipher_strings: vec!["HIGH".into()],
            security_level: OpensslSecurityLevel::Secure,
            description: "cipher config".into(),
            downgrade_risk: None,
        };
        let s = p.to_string();
        assert!(s.contains("Secure"));
        assert!(s.contains("CipherListConfig"));
    }

    #[test]
    fn test_extract_cipher_strings() {
        let line = r#"builder.set_cipher_list("ECDHE+AESGCM:DHE+AESGCM:HIGH")"#;
        let strings = extract_cipher_strings(line);
        assert_eq!(strings.len(), 3);
        assert!(strings.contains(&"ECDHE+AESGCM".to_string()));
        assert!(strings.contains(&"DHE+AESGCM".to_string()));
        assert!(strings.contains(&"HIGH".to_string()));
    }

    #[test]
    fn test_mozilla_modern_secure() {
        let code = r#"let builder = SslAcceptor::mozilla_modern(SslMethod::tls()).unwrap();"#;
        let mut analyzer = OpensslAnalyzer::new();
        let patterns = analyzer.analyze(code, "good.rs");
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].security_level, OpensslSecurityLevel::Secure);
    }
}
