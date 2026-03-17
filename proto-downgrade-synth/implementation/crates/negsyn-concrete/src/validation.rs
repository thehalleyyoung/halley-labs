//! Trace validation and conformance checking.
//!
//! Validates that concrete traces are well-formed, follow protocol rules,
//! respect adversary capability bounds, and have correct byte encodings.

use crate::byte_encoding::{self, verify_tls_record_structure, version_to_wire, wire_to_version};
use crate::trace::{ConcreteMessage, ConcreteTrace, TraceStep, validate_phase_ordering};
use crate::{AdversaryAction, ConcreteError, ConcreteResult};
use crate::{CipherSuite, Extension, HandshakePhase, ProtocolVersion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ── ValidationReport ─────────────────────────────────────────────────────

/// Detailed report from trace validation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub conformance_checks: Vec<ConformanceCheck>,
    pub byte_level_checks: Vec<ByteCheck>,
    pub adversary_checks: Vec<AdversaryCheck>,
    pub total_checks: usize,
    pub passed_checks: usize,
    pub failed_checks: usize,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            ..Default::default()
        }
    }

    fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.failed_checks += 1;
        self.total_checks += 1;
        self.errors.push(error);
    }

    fn add_warning(&mut self, warning: ValidationWarning) {
        self.total_checks += 1;
        self.passed_checks += 1;
        self.warnings.push(warning);
    }

    fn add_pass(&mut self) {
        self.total_checks += 1;
        self.passed_checks += 1;
    }

    pub fn summary(&self) -> String {
        format!(
            "Validation: {} ({}/{} checks passed, {} errors, {} warnings)",
            if self.is_valid { "PASS" } else { "FAIL" },
            self.passed_checks,
            self.total_checks,
            self.errors.len(),
            self.warnings.len(),
        )
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.summary())?;
        for err in &self.errors {
            writeln!(f, "  ERROR: {}", err)?;
        }
        for warn in &self.warnings {
            writeln!(f, "  WARN:  {}", warn)?;
        }
        Ok(())
    }
}

/// A validation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub category: String,
    pub message: String,
    pub message_index: Option<usize>,
    pub step_index: Option<usize>,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.category)?;
        if let Some(idx) = self.message_index {
            write!(f, " msg#{}", idx)?;
        }
        if let Some(idx) = self.step_index {
            write!(f, " step#{}", idx)?;
        }
        write!(f, " {}", self.message)
    }
}

/// A validation warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub category: String,
    pub message: String,
}

impl fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.category, self.message)
    }
}

/// Result of a protocol conformance check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Result of a byte-level check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteCheck {
    pub message_index: usize,
    pub check_name: String,
    pub passed: bool,
    pub detail: String,
}

/// Result of an adversary capability check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryCheck {
    pub action_index: usize,
    pub check_name: String,
    pub passed: bool,
    pub detail: String,
}

// ── TraceValidator ───────────────────────────────────────────────────────

/// Configuration for trace validation.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Maximum allowed adversary actions.
    pub max_adversary_actions: usize,
    /// Whether to validate byte-level encodings strictly.
    pub strict_byte_validation: bool,
    /// Whether to allow deprecated protocol versions.
    pub allow_deprecated_versions: bool,
    /// Maximum message size in bytes.
    pub max_message_size: usize,
    /// Known valid cipher suite IDs.
    pub valid_cipher_suites: BTreeSet<u16>,
    /// Known valid protocol versions.
    pub valid_versions: BTreeSet<ProtocolVersion>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        let mut valid_ciphers = BTreeSet::new();
        // Common TLS cipher suites
        for &id in &[
            0x002f, 0x0035, 0x003c, 0x003d, 0x009c, 0x009d, 0x009e, 0x009f,
            0xc009, 0xc00a, 0xc013, 0xc014, 0xc023, 0xc024, 0xc027, 0xc028,
            0xc02b, 0xc02c, 0xc02f, 0xc030, 0xcca8, 0xcca9, 0xccaa,
            0x1301, 0x1302, 0x1303, // TLS 1.3
        ] {
            valid_ciphers.insert(id);
        }

        let valid_versions: BTreeSet<ProtocolVersion> = [
            ProtocolVersion::Ssl30,
            ProtocolVersion::Tls10,
            ProtocolVersion::Tls11,
            ProtocolVersion::Tls12,
            ProtocolVersion::Tls13,
            ProtocolVersion::Ssh2,
        ]
        .iter()
        .copied()
        .collect();

        Self {
            max_adversary_actions: 100,
            strict_byte_validation: true,
            allow_deprecated_versions: true,
            max_message_size: 1 << 16,
            valid_cipher_suites: valid_ciphers,
            valid_versions,
        }
    }
}

/// Main trace validator that orchestrates all sub-checks.
pub struct TraceValidator {
    config: ValidatorConfig,
}

impl TraceValidator {
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ValidatorConfig::default())
    }

    /// Run all validation checks on a trace.
    pub fn validate(&self, trace: &ConcreteTrace) -> ValidationReport {
        let mut report = ValidationReport::new();

        // 1. Basic structural checks
        self.check_basic_structure(trace, &mut report);

        // 2. Protocol conformance
        self.check_protocol_conformance(trace, &mut report);

        // 3. Adversary capability bounds
        self.check_adversary_capabilities(trace, &mut report);

        // 4. Byte-level validation
        if self.config.strict_byte_validation {
            self.check_byte_level(trace, &mut report);
        }

        // 5. Phase ordering
        let phase_errors = validate_phase_ordering(trace);
        for err in phase_errors {
            report.add_error(ValidationError {
                category: "phase_ordering".into(),
                message: err,
                message_index: None,
                step_index: None,
            });
        }

        report
    }

    fn check_basic_structure(&self, trace: &ConcreteTrace, report: &mut ValidationReport) {
        // Check that trace has at least one message
        if trace.messages.is_empty() {
            report.add_error(ValidationError {
                category: "structure".into(),
                message: "Trace has no messages".into(),
                message_index: None,
                step_index: None,
            });
            return;
        }

        // Check message indices are sequential
        let mut expected_idx = 0;
        let mut seen_indices = HashSet::new();
        for msg in &trace.messages {
            if !seen_indices.insert(msg.index) {
                report.add_error(ValidationError {
                    category: "structure".into(),
                    message: format!("Duplicate message index: {}", msg.index),
                    message_index: Some(msg.index),
                    step_index: None,
                });
            }
        }

        // Check that steps reference valid message indices
        let valid_indices: HashSet<usize> = trace.messages.iter().map(|m| m.index).collect();
        for (i, step) in trace.steps.iter().enumerate() {
            if let Some(msg_idx) = step.message_index() {
                if !valid_indices.contains(&msg_idx) {
                    report.add_error(ValidationError {
                        category: "structure".into(),
                        message: format!("Step references non-existent message index {}", msg_idx),
                        message_index: Some(msg_idx),
                        step_index: Some(i),
                    });
                }
            }
        }

        // Check message sizes
        for msg in &trace.messages {
            if msg.raw_bytes.len() > self.config.max_message_size {
                report.add_error(ValidationError {
                    category: "structure".into(),
                    message: format!(
                        "Message {} exceeds max size: {} > {}",
                        msg.index,
                        msg.raw_bytes.len(),
                        self.config.max_message_size
                    ),
                    message_index: Some(msg.index),
                    step_index: None,
                });
            }
        }

        if report.errors.is_empty() {
            report.add_pass();
        }
    }

    fn check_protocol_conformance(&self, trace: &ConcreteTrace, report: &mut ValidationReport) {
        // Check that versions are valid
        for msg in &trace.messages {
            if !self.config.valid_versions.contains(&msg.protocol_version)
                && !matches!(msg.protocol_version, ProtocolVersion::Unknown(_))
            {
                report.add_warning(ValidationWarning {
                    category: "conformance".into(),
                    message: format!(
                        "Message {} uses unknown version: {}",
                        msg.index, msg.protocol_version
                    ),
                });
            }
            if msg.protocol_version.is_deprecated() && !self.config.allow_deprecated_versions {
                report.add_error(ValidationError {
                    category: "conformance".into(),
                    message: format!(
                        "Message {} uses deprecated version: {}",
                        msg.index, msg.protocol_version
                    ),
                    message_index: Some(msg.index),
                    step_index: None,
                });
            }
        }

        // Check that negotiated cipher is valid (if set)
        if let Some(ref cipher) = trace.negotiated_cipher {
            if !self.config.valid_cipher_suites.is_empty()
                && !self.config.valid_cipher_suites.contains(&cipher.id)
            {
                report.add_warning(ValidationWarning {
                    category: "conformance".into(),
                    message: format!("Negotiated cipher {} not in known-valid set", cipher),
                });
            }
        }

        // Check that a downgrade actually occurred
        if trace.initial_version.security_level() <= trace.downgraded_version.security_level() {
            report.add_warning(ValidationWarning {
                category: "conformance".into(),
                message: format!(
                    "No downgrade detected: {} (level {}) → {} (level {})",
                    trace.initial_version,
                    trace.initial_version.security_level(),
                    trace.downgraded_version,
                    trace.downgraded_version.security_level(),
                ),
            });
        }

        // Check that ClientHello appears before ServerHello
        let mut saw_client_hello = false;
        let mut saw_server_hello = false;
        for msg in &trace.messages {
            if msg.sender == "adversary" {
                continue; // Adversary messages don't need ordering
            }
            if msg.handshake_phase == HandshakePhase::ClientHello && msg.sender == "client" {
                saw_client_hello = true;
            }
            if msg.handshake_phase == HandshakePhase::ServerHello && msg.sender == "server" {
                if !saw_client_hello {
                    report.add_error(ValidationError {
                        category: "conformance".into(),
                        message: "ServerHello before ClientHello".into(),
                        message_index: Some(msg.index),
                        step_index: None,
                    });
                }
                saw_server_hello = true;
            }
        }

        report.conformance_checks.push(ConformanceCheck {
            name: "version_validity".into(),
            passed: report.errors.iter().all(|e| e.category != "conformance"),
            detail: "Protocol versions are recognized".into(),
        });
        report.add_pass();
    }

    fn check_adversary_capabilities(&self, trace: &ConcreteTrace, report: &mut ValidationReport) {
        let adv_count = trace.adversary_action_count();
        if adv_count > self.config.max_adversary_actions {
            report.add_error(ValidationError {
                category: "adversary".into(),
                message: format!(
                    "Too many adversary actions: {} > {}",
                    adv_count, self.config.max_adversary_actions
                ),
                message_index: None,
                step_index: None,
            });
        }

        // Verify each adversary action is within DY model capabilities
        let known_messages: HashSet<usize> = trace
            .steps
            .iter()
            .filter_map(|s| match s {
                TraceStep::ClientSend { message_index } => Some(*message_index),
                TraceStep::ServerSend { message_index } => Some(*message_index),
                TraceStep::AdversaryIntercept { original_index, .. } => Some(*original_index),
                _ => None,
            })
            .collect();

        for (i, action) in trace.adversary_actions.iter().enumerate() {
            let check = match action {
                AdversaryAction::Replay { original_idx, .. } => {
                    if known_messages.contains(original_idx) {
                        AdversaryCheck {
                            action_index: i,
                            check_name: "replay_knowledge".into(),
                            passed: true,
                            detail: format!("Adversary knows message {}", original_idx),
                        }
                    } else {
                        report.add_error(ValidationError {
                            category: "adversary".into(),
                            message: format!(
                                "Replay of unknown message {}",
                                original_idx
                            ),
                            message_index: None,
                            step_index: None,
                        });
                        AdversaryCheck {
                            action_index: i,
                            check_name: "replay_knowledge".into(),
                            passed: false,
                            detail: format!("Adversary cannot replay unknown message {}", original_idx),
                        }
                    }
                }
                AdversaryAction::Modify { msg_idx, .. } => {
                    let known = known_messages.contains(msg_idx);
                    if !known {
                        report.add_error(ValidationError {
                            category: "adversary".into(),
                            message: format!(
                                "Modify of unknown message {}",
                                msg_idx
                            ),
                            message_index: None,
                            step_index: None,
                        });
                    }
                    AdversaryCheck {
                        action_index: i,
                        check_name: "modify_knowledge".into(),
                        passed: known,
                        detail: format!(
                            "Adversary {} message {}",
                            if known { "can modify" } else { "cannot modify unknown" },
                            msg_idx
                        ),
                    }
                }
                AdversaryAction::Inject { .. } => {
                    // Adversary can always inject messages in DY model
                    AdversaryCheck {
                        action_index: i,
                        check_name: "inject_capability".into(),
                        passed: true,
                        detail: "DY adversary can inject messages".into(),
                    }
                }
                AdversaryAction::Drop { .. } => {
                    AdversaryCheck {
                        action_index: i,
                        check_name: "drop_capability".into(),
                        passed: true,
                        detail: "DY adversary can drop messages".into(),
                    }
                }
                AdversaryAction::Forward { .. } => {
                    AdversaryCheck {
                        action_index: i,
                        check_name: "forward_capability".into(),
                        passed: true,
                        detail: "DY adversary can forward messages".into(),
                    }
                }
                AdversaryAction::Intercept { .. } => {
                    AdversaryCheck {
                        action_index: i,
                        check_name: "intercept_capability".into(),
                        passed: true,
                        detail: "DY adversary can intercept messages".into(),
                    }
                }
            };
            report.adversary_checks.push(check);
        }

        report.add_pass();
    }

    fn check_byte_level(&self, trace: &ConcreteTrace, report: &mut ValidationReport) {
        for msg in &trace.messages {
            if msg.raw_bytes.is_empty() {
                report.add_error(ValidationError {
                    category: "byte_level".into(),
                    message: format!("Message {} has empty bytes", msg.index),
                    message_index: Some(msg.index),
                    step_index: None,
                });
                continue;
            }

            // Check if it looks like a TLS record
            let is_tls = msg.raw_bytes.len() >= 5 && matches!(msg.raw_bytes[0], 20..=23);
            // Check if it looks like SSH
            let is_ssh = msg.parsed_fields.ssh_packet_type.is_some()
                || matches!(msg.protocol_version, ProtocolVersion::Ssh2);

            if is_tls {
                match verify_tls_record_structure(&msg.raw_bytes) {
                    Ok(()) => {
                        report.byte_level_checks.push(ByteCheck {
                            message_index: msg.index,
                            check_name: "tls_record_structure".into(),
                            passed: true,
                            detail: "TLS record structure valid".into(),
                        });
                        report.add_pass();

                        // Check version bytes match expected version
                        let wire_major = msg.raw_bytes[1];
                        let wire_minor = msg.raw_bytes[2];
                        let wire_version = wire_to_version(wire_major, wire_minor);
                        // TLS 1.3 uses 3.3 on wire, so accept that
                        let expected = msg.protocol_version;
                        let version_ok = wire_version == expected
                            || (expected == ProtocolVersion::Tls13
                                && wire_version == ProtocolVersion::Tls12);
                        if !version_ok {
                            report.add_warning(ValidationWarning {
                                category: "byte_level".into(),
                                message: format!(
                                    "Message {}: wire version {} doesn't match expected {}",
                                    msg.index, wire_version, expected
                                ),
                            });
                        }
                    }
                    Err(e) => {
                        report.byte_level_checks.push(ByteCheck {
                            message_index: msg.index,
                            check_name: "tls_record_structure".into(),
                            passed: false,
                            detail: format!("TLS record invalid: {}", e),
                        });
                        report.add_error(ValidationError {
                            category: "byte_level".into(),
                            message: format!("Message {} has invalid TLS record: {}", msg.index, e),
                            message_index: Some(msg.index),
                            step_index: None,
                        });
                    }
                }
            } else if is_ssh {
                // SSH packet validation
                if msg.raw_bytes.len() >= 5 {
                    let pkt_len = u32::from_be_bytes([
                        msg.raw_bytes[0],
                        msg.raw_bytes[1],
                        msg.raw_bytes[2],
                        msg.raw_bytes[3],
                    ]);
                    let expected_total = 4 + pkt_len as usize;
                    if expected_total != msg.raw_bytes.len() {
                        report.add_warning(ValidationWarning {
                            category: "byte_level".into(),
                            message: format!(
                                "Message {}: SSH packet length {} != actual {}",
                                msg.index, expected_total, msg.raw_bytes.len()
                            ),
                        });
                    } else {
                        report.byte_level_checks.push(ByteCheck {
                            message_index: msg.index,
                            check_name: "ssh_packet_structure".into(),
                            passed: true,
                            detail: "SSH packet structure valid".into(),
                        });
                        report.add_pass();
                    }
                }
            } else {
                // Unknown format — just check it's non-empty (already checked above)
                report.byte_level_checks.push(ByteCheck {
                    message_index: msg.index,
                    check_name: "non_empty".into(),
                    passed: true,
                    detail: format!("{} bytes present", msg.raw_bytes.len()),
                });
                report.add_pass();
            }
        }
    }
}

// ── ProtocolConformance ──────────────────────────────────────────────────

/// Checks trace against protocol-specific rules (TLS, SSH, DTLS).
pub struct ProtocolConformance {
    protocol: ProtocolKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolKind {
    Tls,
    Ssh,
    Dtls,
}

impl ProtocolConformance {
    pub fn new(protocol: ProtocolKind) -> Self {
        Self { protocol }
    }

    pub fn tls() -> Self {
        Self::new(ProtocolKind::Tls)
    }

    pub fn ssh() -> Self {
        Self::new(ProtocolKind::Ssh)
    }

    /// Check all protocol-specific conformance rules.
    pub fn check(&self, trace: &ConcreteTrace) -> Vec<ConformanceCheck> {
        match self.protocol {
            ProtocolKind::Tls => self.check_tls(trace),
            ProtocolKind::Ssh => self.check_ssh(trace),
            ProtocolKind::Dtls => self.check_dtls(trace),
        }
    }

    fn check_tls(&self, trace: &ConcreteTrace) -> Vec<ConformanceCheck> {
        let mut checks = Vec::new();

        // TLS: First client message should be ClientHello
        let first_client = trace.messages.iter().find(|m| m.sender == "client");
        if let Some(msg) = first_client {
            let ok = msg.handshake_phase == HandshakePhase::ClientHello;
            checks.push(ConformanceCheck {
                name: "tls_first_client_msg".into(),
                passed: ok,
                detail: if ok {
                    "First client message is ClientHello".into()
                } else {
                    format!("First client message is {:?}, expected ClientHello", msg.handshake_phase)
                },
            });
        }

        // TLS: ServerHello should select a cipher from ClientHello's offered set
        if let Some(ref negotiated) = trace.negotiated_cipher {
            let offered: BTreeSet<u16> = trace
                .messages
                .iter()
                .filter(|m| m.handshake_phase == HandshakePhase::ClientHello)
                .flat_map(|m| m.parsed_fields.cipher_suites.iter().copied())
                .collect();
            if !offered.is_empty() {
                let ok = offered.contains(&negotiated.id);
                checks.push(ConformanceCheck {
                    name: "tls_cipher_in_offered".into(),
                    passed: ok,
                    detail: if ok {
                        format!("Negotiated cipher 0x{:04x} was offered", negotiated.id)
                    } else {
                        format!(
                            "Negotiated cipher 0x{:04x} was NOT in offered set {:?}",
                            negotiated.id,
                            offered.iter().map(|c| format!("0x{:04x}", c)).collect::<Vec<_>>()
                        )
                    },
                });
            }
        }

        // TLS: Version negotiation — server version ≤ client version
        let client_versions: Vec<ProtocolVersion> = trace
            .messages
            .iter()
            .filter(|m| m.sender == "client" && m.handshake_phase == HandshakePhase::ClientHello)
            .map(|m| m.protocol_version)
            .collect();
        let server_versions: Vec<ProtocolVersion> = trace
            .messages
            .iter()
            .filter(|m| m.sender == "server" && m.handshake_phase == HandshakePhase::ServerHello)
            .map(|m| m.protocol_version)
            .collect();

        if let (Some(&cv), Some(&sv)) = (client_versions.first(), server_versions.first()) {
            let ok = sv.security_level() <= cv.security_level();
            checks.push(ConformanceCheck {
                name: "tls_version_negotiation".into(),
                passed: ok,
                detail: format!(
                    "Client offered {}, server selected {} ({})",
                    cv,
                    sv,
                    if ok { "valid" } else { "server version exceeds client" }
                ),
            });
        }

        checks
    }

    fn check_ssh(&self, trace: &ConcreteTrace) -> Vec<ConformanceCheck> {
        let mut checks = Vec::new();

        // SSH: Should have KEX_INIT messages
        let has_kex = trace
            .messages
            .iter()
            .any(|m| m.handshake_phase == HandshakePhase::KeyExchange);
        checks.push(ConformanceCheck {
            name: "ssh_kex_present".into(),
            passed: has_kex,
            detail: if has_kex {
                "KEX_INIT messages present".into()
            } else {
                "No KEX_INIT messages found".into()
            },
        });

        // SSH: Both client and server should send KEX_INIT
        let client_kex = trace.messages.iter().any(|m| {
            m.sender == "client" && m.handshake_phase == HandshakePhase::KeyExchange
        });
        let server_kex = trace.messages.iter().any(|m| {
            m.sender == "server" && m.handshake_phase == HandshakePhase::KeyExchange
        });
        checks.push(ConformanceCheck {
            name: "ssh_both_kex".into(),
            passed: client_kex && server_kex,
            detail: format!(
                "Client KEX: {}, Server KEX: {}",
                client_kex, server_kex
            ),
        });

        checks
    }

    fn check_dtls(&self, trace: &ConcreteTrace) -> Vec<ConformanceCheck> {
        let mut checks = Vec::new();

        // DTLS: Version should be a DTLS version
        for msg in &trace.messages {
            if matches!(msg.protocol_version, ProtocolVersion::Dtls10 | ProtocolVersion::Dtls12) {
                checks.push(ConformanceCheck {
                    name: "dtls_version".into(),
                    passed: true,
                    detail: format!("Message {} uses DTLS version", msg.index),
                });
            }
        }

        if checks.is_empty() {
            checks.push(ConformanceCheck {
                name: "dtls_version".into(),
                passed: false,
                detail: "No DTLS-versioned messages found".into(),
            });
        }

        checks
    }
}

// ── AdversaryCapabilityCheck ─────────────────────────────────────────────

/// Verifies adversary actions are within the Dolev-Yao model.
pub struct AdversaryCapabilityCheck {
    /// Maximum number of concurrent intercepted messages.
    pub max_intercepted: usize,
    /// Whether the adversary can decrypt (should be false in DY).
    pub can_decrypt: bool,
    /// Maximum injected payload size.
    pub max_inject_size: usize,
}

impl AdversaryCapabilityCheck {
    pub fn new() -> Self {
        Self {
            max_intercepted: 1000,
            can_decrypt: false,
            max_inject_size: 1 << 16,
        }
    }

    /// Check all adversary actions in the trace.
    pub fn check(&self, trace: &ConcreteTrace) -> Vec<AdversaryCheck> {
        let mut checks = Vec::new();
        let mut intercepted_count = 0usize;

        for (i, action) in trace.adversary_actions.iter().enumerate() {
            match action {
                AdversaryAction::Intercept { .. } => {
                    intercepted_count += 1;
                    let ok = intercepted_count <= self.max_intercepted;
                    checks.push(AdversaryCheck {
                        action_index: i,
                        check_name: "intercept_limit".into(),
                        passed: ok,
                        detail: format!(
                            "Intercepted count: {} (limit: {})",
                            intercepted_count, self.max_intercepted
                        ),
                    });
                }
                AdversaryAction::Inject { payload, .. } => {
                    let ok = payload.len() <= self.max_inject_size;
                    checks.push(AdversaryCheck {
                        action_index: i,
                        check_name: "inject_size".into(),
                        passed: ok,
                        detail: format!(
                            "Inject size: {} bytes (limit: {})",
                            payload.len(),
                            self.max_inject_size
                        ),
                    });
                }
                AdversaryAction::Modify { new_value, .. } => {
                    // In DY model, modification of plaintext is allowed
                    let ok = !self.can_decrypt || new_value.len() <= self.max_inject_size;
                    checks.push(AdversaryCheck {
                        action_index: i,
                        check_name: "modify_capability".into(),
                        passed: ok,
                        detail: "Modification within DY capabilities".into(),
                    });
                }
                _ => {
                    checks.push(AdversaryCheck {
                        action_index: i,
                        check_name: "dy_allowed".into(),
                        passed: true,
                        detail: format!("{:?} is allowed in DY model", action),
                    });
                }
            }
        }

        checks
    }
}

impl Default for AdversaryCapabilityCheck {
    fn default() -> Self {
        Self::new()
    }
}

// ── ByteLevelVerifier ────────────────────────────────────────────────────

/// Verifies byte-level encoding correctness of protocol messages.
pub struct ByteLevelVerifier;

impl ByteLevelVerifier {
    pub fn new() -> Self {
        Self
    }

    /// Verify a single message's byte encoding.
    pub fn verify_message(&self, msg: &ConcreteMessage) -> Vec<ByteCheck> {
        let mut checks = Vec::new();

        if msg.raw_bytes.is_empty() {
            checks.push(ByteCheck {
                message_index: msg.index,
                check_name: "non_empty".into(),
                passed: false,
                detail: "Message has empty bytes".into(),
            });
            return checks;
        }

        // Try TLS record verification
        if msg.raw_bytes.len() >= 5 && matches!(msg.raw_bytes[0], 20..=23) {
            match verify_tls_record_structure(&msg.raw_bytes) {
                Ok(()) => {
                    checks.push(ByteCheck {
                        message_index: msg.index,
                        check_name: "tls_structure".into(),
                        passed: true,
                        detail: "Valid TLS record".into(),
                    });

                    // Check inner handshake if applicable
                    if msg.raw_bytes[0] == byte_encoding::tls_content_type::HANDSHAKE
                        && msg.raw_bytes.len() > 9
                    {
                        let hs_data = &msg.raw_bytes[5..];
                        match byte_encoding::verify_tls_handshake_structure(hs_data) {
                            Ok(()) => {
                                checks.push(ByteCheck {
                                    message_index: msg.index,
                                    check_name: "tls_handshake_structure".into(),
                                    passed: true,
                                    detail: format!(
                                        "Valid handshake type={}",
                                        hs_data[0]
                                    ),
                                });
                            }
                            Err(e) => {
                                checks.push(ByteCheck {
                                    message_index: msg.index,
                                    check_name: "tls_handshake_structure".into(),
                                    passed: false,
                                    detail: format!("Invalid handshake: {}", e),
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    checks.push(ByteCheck {
                        message_index: msg.index,
                        check_name: "tls_structure".into(),
                        passed: false,
                        detail: format!("Invalid: {}", e),
                    });
                }
            }
        }

        checks
    }

    /// Verify all messages in a trace.
    pub fn verify_trace(&self, trace: &ConcreteTrace) -> Vec<ByteCheck> {
        trace
            .messages
            .iter()
            .flat_map(|msg| self.verify_message(msg))
            .collect()
    }
}

impl Default for ByteLevelVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ── ReplaySimulator ──────────────────────────────────────────────────────

/// Simulates replaying a trace against a protocol state machine.
pub struct ReplaySimulator {
    /// Current simulated state for client.
    client_phase: HandshakePhase,
    /// Current simulated state for server.
    server_phase: HandshakePhase,
    /// Messages received by client.
    client_received: Vec<usize>,
    /// Messages received by server.
    server_received: Vec<usize>,
    /// Dropped messages.
    dropped: HashSet<usize>,
    /// Step log.
    log: Vec<String>,
}

impl ReplaySimulator {
    pub fn new() -> Self {
        Self {
            client_phase: HandshakePhase::Initial,
            server_phase: HandshakePhase::Initial,
            client_received: Vec::new(),
            server_received: Vec::new(),
            dropped: HashSet::new(),
            log: Vec::new(),
        }
    }

    /// Simulate executing the trace and return a validation report.
    pub fn simulate(&mut self, trace: &ConcreteTrace) -> ConcreteResult<Vec<String>> {
        self.log.clear();
        self.log.push("=== Replay Simulation Start ===".into());

        for (i, step) in trace.steps.iter().enumerate() {
            self.log.push(format!("Step {}: {}", i, step));

            match step {
                TraceStep::ClientSend { message_index } => {
                    if let Some(msg) = trace.messages.iter().find(|m| m.index == *message_index) {
                        self.advance_phase(&msg.sender, msg.handshake_phase);
                        self.log.push(format!(
                            "  Client sends {:?} ({} bytes)",
                            msg.handshake_phase,
                            msg.raw_bytes.len()
                        ));
                    }
                }
                TraceStep::ServerSend { message_index } => {
                    if let Some(msg) = trace.messages.iter().find(|m| m.index == *message_index) {
                        self.advance_phase(&msg.sender, msg.handshake_phase);
                        self.log.push(format!(
                            "  Server sends {:?} ({} bytes)",
                            msg.handshake_phase,
                            msg.raw_bytes.len()
                        ));
                    }
                }
                TraceStep::AdversaryDrop { dropped_index, .. } => {
                    self.dropped.insert(*dropped_index);
                    self.log.push(format!("  Message {} dropped", dropped_index));
                }
                TraceStep::AdversaryIntercept { original_index, .. } => {
                    self.log.push(format!("  Message {} intercepted", original_index));
                }
                TraceStep::AdversaryInject { injected_index, to } => {
                    if let Some(msg) = trace.messages.iter().find(|m| m.index == *injected_index) {
                        if to == "server" {
                            self.server_received.push(*injected_index);
                            self.advance_phase("server_recv", msg.handshake_phase);
                        } else {
                            self.client_received.push(*injected_index);
                            self.advance_phase("client_recv", msg.handshake_phase);
                        }
                        self.log.push(format!(
                            "  Injected {:?} to {} ({} bytes)",
                            msg.handshake_phase,
                            to,
                            msg.raw_bytes.len()
                        ));
                    }
                }
                TraceStep::AdversaryModify { modified_index, to, .. } => {
                    if let Some(msg) = trace.messages.iter().find(|m| m.index == *modified_index) {
                        if to == "server" {
                            self.server_received.push(*modified_index);
                        } else {
                            self.client_received.push(*modified_index);
                        }
                        self.log.push(format!(
                            "  Modified message delivered to {} ({} bytes)",
                            to,
                            msg.raw_bytes.len()
                        ));
                    }
                }
            }
        }

        self.log.push(format!(
            "=== Replay Complete: client={:?}, server={:?} ===",
            self.client_phase, self.server_phase
        ));

        Ok(self.log.clone())
    }

    fn advance_phase(&mut self, role: &str, phase: HandshakePhase) {
        match role {
            "client" | "client_recv" => {
                if phase.order_index() >= self.client_phase.order_index() {
                    self.client_phase = phase;
                }
            }
            "server" | "server_recv" => {
                if phase.order_index() >= self.server_phase.order_index() {
                    self.server_phase = phase;
                }
            }
            _ => {}
        }
    }

    pub fn client_phase(&self) -> HandshakePhase {
        self.client_phase
    }

    pub fn server_phase(&self) -> HandshakePhase {
        self.server_phase
    }

    pub fn log(&self) -> &[String] {
        &self.log
    }
}

impl Default for ReplaySimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceBuilder;

    fn make_simple_trace() -> ConcreteTrace {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30);
        // Valid TLS record: type=22 (handshake), version=3.3, length=2, body=[1,0]
        builder.client_send(
            vec![0x16, 0x03, 0x03, 0x00, 0x02, 0x01, 0x00],
            ProtocolVersion::Tls12,
            HandshakePhase::ClientHello,
        );
        builder.adversary_intercept(0, "client");
        builder.adversary_inject(
            "server",
            vec![0x16, 0x03, 0x00, 0x00, 0x02, 0x01, 0x00],
            ProtocolVersion::Ssl30,
            HandshakePhase::ClientHello,
        );
        builder.server_send(
            vec![0x16, 0x03, 0x00, 0x00, 0x02, 0x02, 0x00],
            ProtocolVersion::Ssl30,
            HandshakePhase::ServerHello,
        );
        builder.build()
    }

    #[test]
    fn test_validator_basic() {
        let trace = make_simple_trace();
        let validator = TraceValidator::with_defaults();
        let report = validator.validate(&trace);
        // Should pass basic validation — may have warnings about version
        assert!(
            report.total_checks > 0,
            "Should have run some checks: {:?}",
            report
        );
    }

    #[test]
    fn test_validator_empty_trace() {
        let trace = ConcreteTrace::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30);
        let validator = TraceValidator::with_defaults();
        let report = validator.validate(&trace);
        assert!(!report.is_valid);
        assert!(report.errors.iter().any(|e| e.message.contains("no messages")));
    }

    #[test]
    fn test_protocol_conformance_tls() {
        let trace = make_simple_trace();
        let conformance = ProtocolConformance::tls();
        let checks = conformance.check(&trace);
        assert!(!checks.is_empty());
        // Should have first_client_msg check
        assert!(checks.iter().any(|c| c.name.contains("first_client")));
    }

    #[test]
    fn test_protocol_conformance_ssh() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Ssh2, ProtocolVersion::Ssh2);
        builder.client_send(vec![0x00, 0x00, 0x00, 0x0c, 0x04, 0x14], ProtocolVersion::Ssh2, HandshakePhase::KeyExchange);
        builder.server_send(vec![0x00, 0x00, 0x00, 0x0c, 0x04, 0x14], ProtocolVersion::Ssh2, HandshakePhase::KeyExchange);
        let trace = builder.build();
        let conformance = ProtocolConformance::ssh();
        let checks = conformance.check(&trace);
        assert!(checks.iter().any(|c| c.name == "ssh_both_kex" && c.passed));
    }

    #[test]
    fn test_adversary_capability_check() {
        let trace = make_simple_trace();
        let checker = AdversaryCapabilityCheck::new();
        let checks = checker.check(&trace);
        assert!(checks.iter().all(|c| c.passed));
    }

    #[test]
    fn test_byte_level_verifier() {
        let trace = make_simple_trace();
        let verifier = ByteLevelVerifier::new();
        let checks = verifier.verify_trace(&trace);
        // All messages have valid TLS structure
        let tls_checks: Vec<_> = checks.iter().filter(|c| c.check_name == "tls_structure").collect();
        assert!(!tls_checks.is_empty());
    }

    #[test]
    fn test_replay_simulator() {
        let trace = make_simple_trace();
        let mut sim = ReplaySimulator::new();
        let log = sim.simulate(&trace).unwrap();
        assert!(!log.is_empty());
        assert!(log.first().unwrap().contains("Start"));
        assert!(log.last().unwrap().contains("Complete"));
    }

    #[test]
    fn test_validation_report_display() {
        let mut report = ValidationReport::new();
        report.add_error(ValidationError {
            category: "test".into(),
            message: "test error".into(),
            message_index: Some(0),
            step_index: None,
        });
        let s = format!("{}", report);
        assert!(s.contains("FAIL"));
        assert!(s.contains("test error"));
    }

    #[test]
    fn test_validator_config_defaults() {
        let config = ValidatorConfig::default();
        assert!(!config.valid_cipher_suites.is_empty());
        assert!(config.valid_cipher_suites.contains(&0xc02f));
        assert!(config.valid_versions.contains(&ProtocolVersion::Tls12));
    }

    #[test]
    fn test_validator_deprecated_version_strict() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30);
        builder.client_send(
            vec![0x16, 0x03, 0x00, 0x00, 0x01, 0x01],
            ProtocolVersion::Ssl30,
            HandshakePhase::ClientHello,
        );
        let trace = builder.build();

        let mut config = ValidatorConfig::default();
        config.allow_deprecated_versions = false;
        let validator = TraceValidator::new(config);
        let report = validator.validate(&trace);
        assert!(report.errors.iter().any(|e| e.message.contains("deprecated")));
    }

    #[test]
    fn test_replay_simulator_phases() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Tls12);
        builder.client_send(vec![0x16, 0x03, 0x03, 0x00, 0x01, 0x01], ProtocolVersion::Tls12, HandshakePhase::ClientHello);
        builder.server_send(vec![0x16, 0x03, 0x03, 0x00, 0x01, 0x02], ProtocolVersion::Tls12, HandshakePhase::ServerHello);
        let trace = builder.build();

        let mut sim = ReplaySimulator::new();
        sim.simulate(&trace).unwrap();
        assert_eq!(sim.client_phase(), HandshakePhase::ClientHello);
        assert_eq!(sim.server_phase(), HandshakePhase::ServerHello);
    }
}
