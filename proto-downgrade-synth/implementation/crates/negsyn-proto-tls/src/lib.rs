//! `negsyn-proto-tls` — TLS protocol module for the NegSynth downgrade attack synthesis tool.
//!
//! This crate provides RFC-compliant TLS parsing, cipher suite negotiation,
//! extension handling, handshake state machine modeling, known vulnerability
//! patterns, and negotiation logic for protocol versions TLS 1.0 through 1.3.
//!
//! # Modules
//!
//! - [`version`] — TLS version types, ordering, negotiation, fallback SCSV, downgrade sentinels
//! - [`record`] — TLS record layer: framing, content types, fragmentation, reassembly
//! - [`handshake`] — Handshake messages: ClientHello, ServerHello, Certificate, Finished, etc.
//! - [`cipher_suites`] — IANA cipher suite registry with full algorithm decomposition
//! - [`extensions`] — TLS extensions: SNI, ALPN, supported_versions, key_share, etc.
//! - [`parser`] — Unified TLS message parser using nom combinators
//! - [`state_machine`] — TLS handshake finite state machine for 1.2 and 1.3
//! - [`vulnerabilities`] — Known CVE patterns: FREAK, Logjam, POODLE, Terrapin, DROWN, CCS Injection
//! - [`negotiation`] — Negotiation engine with policy-based filtering
//! - [`cert_formats`] — PEM/DER certificate format parsing and X.509 metadata extraction
//! - [`rustls_patterns`] — Rustls library pattern analyzer for downgrade detection
//! - [`openssl_patterns`] — OpenSSL-rs pattern analyzer for downgrade detection
//! - [`message_formats`] — TLS protocol message wire format parsing and serialization

pub mod version;
pub mod record;
pub mod handshake;
pub mod cipher_suites;
pub mod extensions;
pub mod parser;
pub mod state_machine;
pub mod vulnerabilities;
pub mod negotiation;
pub mod cert_formats;
pub mod rustls_patterns;
pub mod openssl_patterns;
pub mod message_formats;

// Re-export key types for convenience.
pub use version::TlsVersion;
pub use record::{ContentType, TlsRecord, RecordLayer, TlsAlert, AlertLevel, AlertDescription};
pub use handshake::{
    HandshakeMessage, HandshakeType, ClientHello, ServerHello,
    CertificateMessage, HandshakeHash, CompressionMethod,
};
pub use cipher_suites::{
    TlsCipherSuite, CipherSuiteRegistry, KeyExchange, Authentication,
    BulkEncryption, MacAlgorithm, SecurityLevel,
};
pub use extensions::{TlsExtension, NamedGroup, SignatureScheme, KeyShareEntry};
pub use parser::{TlsParser, ParsedMessage, ParseError};
pub use state_machine::{TlsStateMachine, TlsState, TlsEvent};
pub use vulnerabilities::{KnownVulnerability, VulnerabilityScanner, VulnerabilityDetection, AttackTrace};
pub use negotiation::{NegotiationEngine, NegotiationPolicy, NegotiationResult, ClientPreferenceBuilder};

// Re-exports from new modules.
pub use cert_formats::{CertFormat, CertificateInfo, SignatureAlgorithm, KeyType, CertParseError};
pub use rustls_patterns::{RustlsPattern, RustlsAnalyzer, PatternType, SourceLocation, RustlsSecurityLevel};
pub use openssl_patterns::{OpensslPattern, OpensslAnalyzer, OpensslPatternType, OpensslSecurityLevel};
pub use message_formats::{
    WireClientHello, WireServerHello, WireCertificateMessage, RawExtension, MessageFormatError,
};
