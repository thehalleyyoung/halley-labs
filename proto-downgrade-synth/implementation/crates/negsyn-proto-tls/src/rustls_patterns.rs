//! Rustls library pattern analyzer for TLS negotiation synthesis.
//!
//! Detects cipher suite configuration, version negotiation, certificate
//! validation, and fallback behavior patterns in Rust source code that
//! uses the `rustls` crate. Identifies potential downgrade attack surfaces
//! from configuration patterns.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Source location
// ---------------------------------------------------------------------------

/// Location of a detected pattern in source code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File path (relative or absolute).
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// Column offset (0-based).
    pub column: usize,
    /// The matched source text snippet.
    pub snippet: String,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

// ---------------------------------------------------------------------------
// Pattern types
// ---------------------------------------------------------------------------

/// Category of detected rustls usage pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Cipher suite configuration (e.g., `with_cipher_suites`).
    CipherSuiteConfig,
    /// TLS version selection (e.g., `with_protocol_versions`).
    VersionNegotiation,
    /// Certificate validation logic (custom verifiers, danger methods).
    CertificateValidation,
    /// Fallback behavior (retry with lower version, relaxed config).
    FallbackBehavior,
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CipherSuiteConfig => write!(f, "CipherSuiteConfig"),
            Self::VersionNegotiation => write!(f, "VersionNegotiation"),
            Self::CertificateValidation => write!(f, "CertificateValidation"),
            Self::FallbackBehavior => write!(f, "FallbackBehavior"),
        }
    }
}

// ---------------------------------------------------------------------------
// Security level for pattern analysis
// ---------------------------------------------------------------------------

/// Security assessment of a detected pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RustlsSecurityLevel {
    /// Pattern follows best practices.
    Secure,
    /// Pattern is acceptable but not ideal.
    Acceptable,
    /// Pattern may weaken security.
    Degraded,
    /// Pattern is dangerous (e.g., disabled verification).
    Dangerous,
}

impl fmt::Display for RustlsSecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Detected pattern
// ---------------------------------------------------------------------------

/// A detected rustls usage pattern with security analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustlsPattern {
    /// Category of the pattern.
    pub pattern_type: PatternType,
    /// Where the pattern was found.
    pub location: SourceLocation,
    /// Cipher suites referenced in this pattern (IANA names).
    pub cipher_suites: Vec<String>,
    /// Security assessment.
    pub security_level: RustlsSecurityLevel,
    /// Human-readable description of what was detected.
    pub description: String,
    /// Potential downgrade risk if any.
    pub downgrade_risk: Option<String>,
}

impl fmt::Display for RustlsPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} at {} ({})",
            self.security_level, self.pattern_type, self.location, self.description
        )
    }
}

// ---------------------------------------------------------------------------
// Known rustls API patterns
// ---------------------------------------------------------------------------

/// Rustls API call signatures that are relevant to negotiation analysis.
struct ApiPattern {
    /// Text pattern to search for in source.
    signature: &'static str,
    /// What type of pattern this matches.
    pattern_type: PatternType,
    /// Base security level for this API usage.
    base_security: RustlsSecurityLevel,
    /// Description template.
    description: &'static str,
}

const RUSTLS_PATTERNS: &[ApiPattern] = &[
    ApiPattern {
        signature: "with_cipher_suites",
        pattern_type: PatternType::CipherSuiteConfig,
        base_security: RustlsSecurityLevel::Acceptable,
        description: "Custom cipher suite configuration via with_cipher_suites()",
    },
    ApiPattern {
        signature: "with_protocol_versions",
        pattern_type: PatternType::VersionNegotiation,
        base_security: RustlsSecurityLevel::Acceptable,
        description: "Protocol version configuration via with_protocol_versions()",
    },
    ApiPattern {
        signature: "dangerous()",
        pattern_type: PatternType::CertificateValidation,
        base_security: RustlsSecurityLevel::Dangerous,
        description: "Dangerous configuration builder bypassing safety checks",
    },
    ApiPattern {
        signature: "with_custom_certificate_verifier",
        pattern_type: PatternType::CertificateValidation,
        base_security: RustlsSecurityLevel::Degraded,
        description: "Custom certificate verifier replacing default validation",
    },
    ApiPattern {
        signature: "ServerCertVerifier",
        pattern_type: PatternType::CertificateValidation,
        base_security: RustlsSecurityLevel::Degraded,
        description: "Custom ServerCertVerifier implementation",
    },
    ApiPattern {
        signature: "ClientConfig::builder()",
        pattern_type: PatternType::CipherSuiteConfig,
        base_security: RustlsSecurityLevel::Secure,
        description: "ClientConfig builder initialization",
    },
    ApiPattern {
        signature: "ServerConfig::builder()",
        pattern_type: PatternType::CipherSuiteConfig,
        base_security: RustlsSecurityLevel::Secure,
        description: "ServerConfig builder initialization",
    },
    ApiPattern {
        signature: "TLS12",
        pattern_type: PatternType::VersionNegotiation,
        base_security: RustlsSecurityLevel::Acceptable,
        description: "Reference to TLS 1.2 protocol version",
    },
    ApiPattern {
        signature: "TLS13",
        pattern_type: PatternType::VersionNegotiation,
        base_security: RustlsSecurityLevel::Secure,
        description: "Reference to TLS 1.3 protocol version",
    },
    ApiPattern {
        signature: "no_client_auth",
        pattern_type: PatternType::CertificateValidation,
        base_security: RustlsSecurityLevel::Acceptable,
        description: "Server configured without client certificate authentication",
    },
];

/// Well-known rustls cipher suite constant names.
const RUSTLS_CIPHER_SUITES: &[(&str, &str)] = &[
    ("TLS13_AES_256_GCM_SHA384", "TLS_AES_256_GCM_SHA384"),
    ("TLS13_AES_128_GCM_SHA256", "TLS_AES_128_GCM_SHA256"),
    ("TLS13_CHACHA20_POLY1305_SHA256", "TLS_CHACHA20_POLY1305_SHA256"),
    ("TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"),
    ("TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256"),
    ("TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256"),
    ("TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"),
    ("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"),
    ("TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256"),
];

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Analyzes Rust source code for rustls configuration patterns.
pub struct RustlsAnalyzer {
    /// Accumulated detected patterns.
    patterns: Vec<RustlsPattern>,
}

impl RustlsAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Analyze source code text and return all detected patterns.
    pub fn analyze(&mut self, source: &str, file_path: &str) -> &[RustlsPattern] {
        self.patterns.clear();

        for (line_num, line) in source.lines().enumerate() {
            let line_1based = line_num + 1;

            for api_pat in RUSTLS_PATTERNS {
                if let Some(col) = line.find(api_pat.signature) {
                    let cipher_suites = extract_cipher_suites_from_line(line);
                    let security = refine_security_level(api_pat, line);
                    let downgrade_risk = assess_downgrade_risk(api_pat, line);

                    self.patterns.push(RustlsPattern {
                        pattern_type: api_pat.pattern_type,
                        location: SourceLocation {
                            file: file_path.to_string(),
                            line: line_1based,
                            column: col,
                            snippet: line.trim().to_string(),
                        },
                        cipher_suites,
                        security_level: security,
                        description: api_pat.description.to_string(),
                        downgrade_risk,
                    });
                }
            }
        }

        &self.patterns
    }

    /// Return accumulated patterns.
    pub fn patterns(&self) -> &[RustlsPattern] {
        &self.patterns
    }

    /// Check if any dangerous patterns were detected.
    pub fn has_dangerous_patterns(&self) -> bool {
        self.patterns
            .iter()
            .any(|p| p.security_level == RustlsSecurityLevel::Dangerous)
    }

    /// Return patterns that represent potential downgrade attack surfaces.
    pub fn downgrade_surfaces(&self) -> Vec<&RustlsPattern> {
        self.patterns
            .iter()
            .filter(|p| p.downgrade_risk.is_some())
            .collect()
    }
}

/// Extract cipher suite constant names referenced on a source line.
fn extract_cipher_suites_from_line(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    for &(rustls_name, iana_name) in RUSTLS_CIPHER_SUITES {
        if line.contains(rustls_name) {
            found.push(iana_name.to_string());
        }
    }
    found
}

/// Refine the security level based on surrounding context.
fn refine_security_level(api_pat: &ApiPattern, line: &str) -> RustlsSecurityLevel {
    // `dangerous()` is always dangerous
    if api_pat.signature == "dangerous()" {
        return RustlsSecurityLevel::Dangerous;
    }

    // If the line explicitly references TLS 1.3 only, it's secure
    if line.contains("TLS13") && !line.contains("TLS12") {
        return RustlsSecurityLevel::Secure;
    }

    // If the line explicitly disables TLS 1.3 (contains only TLS12), degraded
    if line.contains("TLS12") && !line.contains("TLS13")
        && api_pat.pattern_type == PatternType::VersionNegotiation
    {
        return RustlsSecurityLevel::Degraded;
    }

    api_pat.base_security
}

/// Assess downgrade risk from a pattern.
fn assess_downgrade_risk(api_pat: &ApiPattern, line: &str) -> Option<String> {
    match api_pat.pattern_type {
        PatternType::FallbackBehavior => {
            Some("Fallback logic may enable protocol downgrade attacks".into())
        }
        PatternType::VersionNegotiation => {
            if line.contains("TLS12") && !line.contains("TLS13") {
                Some("Restricting to TLS 1.2 only removes TLS 1.3 protections".into())
            } else {
                None
            }
        }
        PatternType::CertificateValidation => {
            if line.contains("dangerous") {
                Some("Disabled certificate validation enables MITM attacks".into())
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

    const SAMPLE_RUSTLS_CODE: &str = r#"
use rustls::{ClientConfig, OwnedTrustAnchor};

fn build_client() -> ClientConfig {
    let config = ClientConfig::builder()
        .with_cipher_suites(&[
            rustls::cipher_suite::TLS13_AES_256_GCM_SHA384,
            rustls::cipher_suite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        ])
        .with_protocol_versions(&[&rustls::version::TLS12, &rustls::version::TLS13])
        .unwrap()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    config
}
"#;

    const DANGEROUS_CODE: &str = r#"
let config = ClientConfig::builder()
    .dangerous()
    .with_custom_certificate_verifier(Arc::new(NoCertVerifier))
    .with_no_client_auth();
"#;

    #[test]
    fn test_detect_cipher_suite_config() {
        let mut analyzer = RustlsAnalyzer::new();
        let patterns = analyzer.analyze(SAMPLE_RUSTLS_CODE, "test.rs");

        let cipher_configs: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::CipherSuiteConfig)
            .collect();

        assert!(!cipher_configs.is_empty());
    }

    #[test]
    fn test_detect_version_negotiation() {
        let mut analyzer = RustlsAnalyzer::new();
        let patterns = analyzer.analyze(SAMPLE_RUSTLS_CODE, "test.rs");

        let version_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::VersionNegotiation)
            .collect();

        assert!(!version_patterns.is_empty());
    }

    #[test]
    fn test_detect_cipher_suites_in_code() {
        let mut analyzer = RustlsAnalyzer::new();
        let patterns = analyzer.analyze(SAMPLE_RUSTLS_CODE, "test.rs");

        let all_suites: Vec<_> = patterns
            .iter()
            .flat_map(|p| p.cipher_suites.iter())
            .collect();

        assert!(all_suites.iter().any(|s| s.contains("AES_256_GCM")));
    }

    #[test]
    fn test_dangerous_pattern_detection() {
        let mut analyzer = RustlsAnalyzer::new();
        analyzer.analyze(DANGEROUS_CODE, "bad.rs");

        assert!(analyzer.has_dangerous_patterns());

        let dangerous: Vec<_> = analyzer
            .patterns()
            .iter()
            .filter(|p| p.security_level == RustlsSecurityLevel::Dangerous)
            .collect();

        assert!(!dangerous.is_empty());
    }

    #[test]
    fn test_custom_verifier_detection() {
        let mut analyzer = RustlsAnalyzer::new();
        let patterns = analyzer.analyze(DANGEROUS_CODE, "bad.rs");

        let cert_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::CertificateValidation)
            .collect();

        assert!(cert_patterns.len() >= 2); // dangerous() + with_custom_certificate_verifier
    }

    #[test]
    fn test_downgrade_surfaces() {
        let code = r#"
let config = ClientConfig::builder()
    .dangerous()
    .with_custom_certificate_verifier(Arc::new(NoCertVerifier));
"#;
        let mut analyzer = RustlsAnalyzer::new();
        analyzer.analyze(code, "vuln.rs");
        let surfaces = analyzer.downgrade_surfaces();
        assert!(!surfaces.is_empty());
    }

    #[test]
    fn test_tls12_only_degraded() {
        let code = "let v = &[&rustls::version::TLS12];";
        let mut analyzer = RustlsAnalyzer::new();
        let patterns = analyzer.analyze(code, "old.rs");

        let degraded: Vec<_> = patterns
            .iter()
            .filter(|p| p.security_level == RustlsSecurityLevel::Degraded)
            .collect();

        assert!(!degraded.is_empty());
    }

    #[test]
    fn test_empty_source() {
        let mut analyzer = RustlsAnalyzer::new();
        let patterns = analyzer.analyze("", "empty.rs");
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_pattern_display() {
        let p = RustlsPattern {
            pattern_type: PatternType::CipherSuiteConfig,
            location: SourceLocation {
                file: "test.rs".into(),
                line: 10,
                column: 5,
                snippet: "with_cipher_suites(&[...])".into(),
            },
            cipher_suites: vec!["TLS_AES_256_GCM_SHA384".into()],
            security_level: RustlsSecurityLevel::Secure,
            description: "cipher config".into(),
            downgrade_risk: None,
        };
        let s = p.to_string();
        assert!(s.contains("Secure"));
        assert!(s.contains("test.rs"));
    }

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation {
            file: "src/main.rs".into(),
            line: 42,
            column: 8,
            snippet: String::new(),
        };
        assert_eq!(loc.to_string(), "src/main.rs:42:8");
    }
}
