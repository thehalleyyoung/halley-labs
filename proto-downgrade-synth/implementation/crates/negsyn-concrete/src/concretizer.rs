//! Core concretization logic — ALG5: CONCRETIZE.
//!
//! Takes a SAT model from the SMT solver and concretizes it into an
//! executable byte-level attack trace.

use crate::byte_encoding::{
    self, build_client_hello, build_server_hello, build_ssh_kex_init, build_change_cipher_spec,
    build_tls_finished, ByteEncoder, ClientHelloParams, ServerHelloParams, SshKexInitParams,
    SshPacketEncoder, TlsRecordEncoder, version_to_wire, wire_to_version,
};
use crate::trace::{
    ConcreteMessage, ConcreteTrace, FieldModification, ParsedFields, TraceBuilder, TraceStep,
};
use crate::validation::{TraceValidator, ValidatorConfig, ValidationReport};
use crate::{
    AdversaryAction, ConcreteError, ConcreteResult, SmtModel, SmtValue,
};
use crate::{
    CipherSuite, Extension, HandshakePhase, KeyExchange, AuthAlgorithm,
    BulkEncryption, MacAlgorithm, ProtocolVersion,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

// ── ConcretizerConfig ────────────────────────────────────────────────────

/// Configuration for the concretizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcretizerConfig {
    /// Maximum number of protocol messages to produce.
    pub max_messages: usize,
    /// Maximum total byte count.
    pub max_total_bytes: usize,
    /// Whether to validate the trace after construction.
    pub validate_after_build: bool,
    /// Variable name prefix for cipher suite in SMT model.
    pub cipher_var_prefix: String,
    /// Variable name prefix for version in SMT model.
    pub version_var_prefix: String,
    /// Variable name prefix for adversary actions.
    pub adversary_var_prefix: String,
    /// Variable name for step count.
    pub step_count_var: String,
    /// Known cipher suites for ID resolution.
    pub known_ciphers: Vec<CipherSuite>,
    /// Target library name.
    pub library_name: String,
    /// Target library version.
    pub library_version: String,
}

impl Default for ConcretizerConfig {
    fn default() -> Self {
        Self {
            max_messages: 50,
            max_total_bytes: 1 << 20,
            validate_after_build: true,
            cipher_var_prefix: "cipher".into(),
            version_var_prefix: "version".into(),
            adversary_var_prefix: "adv_action".into(),
            step_count_var: "step_count".into(),
            known_ciphers: default_cipher_suites(),
            library_name: String::new(),
            library_version: String::new(),
        }
    }
}

/// Provides a default set of well-known TLS cipher suites.
fn default_cipher_suites() -> Vec<CipherSuite> {
    vec![
        CipherSuite::new(0x002f, "TLS_RSA_WITH_AES_128_CBC_SHA", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Aes128, MacAlgorithm::HmacSha1),
        CipherSuite::new(0x0035, "TLS_RSA_WITH_AES_256_CBC_SHA", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Aes256, MacAlgorithm::HmacSha1),
        CipherSuite::new(0x003c, "TLS_RSA_WITH_AES_128_CBC_SHA256", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Aes128, MacAlgorithm::HmacSha256),
        CipherSuite::new(0x003d, "TLS_RSA_WITH_AES_256_CBC_SHA256", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Aes256, MacAlgorithm::HmacSha256),
        CipherSuite::new(0x009c, "TLS_RSA_WITH_AES_128_GCM_SHA256", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Aes128Gcm, MacAlgorithm::Aead),
        CipherSuite::new(0x009d, "TLS_RSA_WITH_AES_256_GCM_SHA384", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Aes256Gcm, MacAlgorithm::Aead),
        CipherSuite::new(0xc013, "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Aes128, MacAlgorithm::HmacSha1),
        CipherSuite::new(0xc014, "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Aes256, MacAlgorithm::HmacSha1),
        CipherSuite::new(0xc02f, "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Aes128Gcm, MacAlgorithm::Aead),
        CipherSuite::new(0xc030, "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Aes256Gcm, MacAlgorithm::Aead),
        CipherSuite::new(0xcca8, "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Chacha20Poly1305, MacAlgorithm::Aead),
        CipherSuite::new(0x000a, "TLS_RSA_WITH_3DES_EDE_CBC_SHA", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::TripleDes, MacAlgorithm::HmacSha1),
        CipherSuite::new(0x0004, "TLS_RSA_WITH_RC4_128_MD5", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Rc4_128, MacAlgorithm::HmacMd5),
        CipherSuite::new(0x0005, "TLS_RSA_WITH_RC4_128_SHA", KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Rc4_128, MacAlgorithm::HmacSha1),
        CipherSuite::null_suite(),
        // TLS 1.3 suites
        CipherSuite::new(0x1301, "TLS_AES_128_GCM_SHA256", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Aes128Gcm, MacAlgorithm::Aead),
        CipherSuite::new(0x1302, "TLS_AES_256_GCM_SHA384", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Aes256Gcm, MacAlgorithm::Aead),
        CipherSuite::new(0x1303, "TLS_CHACHA20_POLY1305_SHA256", KeyExchange::Ecdhe, AuthAlgorithm::Rsa, BulkEncryption::Chacha20Poly1305, MacAlgorithm::Aead),
    ]
}

// ── Extracted model values ───────────────────────────────────────────────

/// Values extracted from an SMT model relevant to concretization.
#[derive(Debug, Clone)]
struct ExtractedValues {
    /// Negotiated cipher suite ID.
    negotiated_cipher_id: Option<u16>,
    /// Offered cipher suite IDs (from ClientHello).
    offered_cipher_ids: Vec<u16>,
    /// Client's protocol version.
    client_version: Option<ProtocolVersion>,
    /// Server's negotiated version.
    server_version: Option<ProtocolVersion>,
    /// Extension IDs present.
    extensions: Vec<(u16, Vec<u8>)>,
    /// Adversary actions extracted from model.
    adversary_actions: Vec<ExtractedAdversaryAction>,
    /// Number of protocol steps.
    step_count: usize,
    /// Session ID.
    session_id: Option<Vec<u8>>,
    /// Client random.
    client_random: Option<[u8; 32]>,
    /// Server random.
    server_random: Option<[u8; 32]>,
    /// Raw variable assignments for debugging.
    raw_assignments: BTreeMap<String, u64>,
}

#[derive(Debug, Clone)]
struct ExtractedAdversaryAction {
    step: usize,
    action_type: u64,
    target_msg: usize,
    modified_field: Option<String>,
    modified_value: Option<Vec<u8>>,
}

// ── Concretizer ──────────────────────────────────────────────────────────

/// Main concretizer struct — implements ALG5: CONCRETIZE.
///
/// Takes a satisfying SMT model and produces a concrete byte-level attack trace.
pub struct Concretizer {
    config: ConcretizerConfig,
    /// Cipher suite lookup by ID.
    cipher_lookup: HashMap<u16, CipherSuite>,
}

impl Concretizer {
    pub fn new(config: ConcretizerConfig) -> Self {
        let cipher_lookup: HashMap<u16, CipherSuite> = config
            .known_ciphers
            .iter()
            .map(|cs| (cs.id, cs.clone()))
            .collect();
        Self {
            config,
            cipher_lookup,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(ConcretizerConfig::default())
    }

    /// ALG5: CONCRETIZE — main entry point.
    ///
    /// Given a SAT model from the SMT solver, produce a concrete attack trace
    /// with byte-level protocol messages.
    pub fn concretize(&self, model: &SmtModel) -> ConcreteResult<ConcreteTrace> {
        if !model.is_sat {
            return Err(ConcreteError::Concretization(
                "Cannot concretize an UNSAT model".into(),
            ));
        }

        log::debug!("ALG5: Starting concretization from SMT model with {} variables",
            model.assignments.len());

        // Step 1: Extract variable assignments from model
        let extracted = self.extract_values(model)?;
        log::debug!("Extracted: cipher={:?}, client_ver={:?}, server_ver={:?}, {} adv actions",
            extracted.negotiated_cipher_id,
            extracted.client_version,
            extracted.server_version,
            extracted.adversary_actions.len());

        // Step 2: Resolve cipher suite
        let negotiated_cipher = self.resolve_cipher(extracted.negotiated_cipher_id)?;

        // Step 3: Determine versions
        let client_version = extracted.client_version.unwrap_or(ProtocolVersion::Tls12);
        let server_version = extracted.server_version.unwrap_or(ProtocolVersion::Ssl30);

        // Step 4: Reconstruct extensions
        let extensions = self.reconstruct_extensions(&extracted)?;

        // Step 5: Build the message sequence
        let trace = self.build_message_sequence(
            client_version,
            server_version,
            &extracted,
            &negotiated_cipher,
            &extensions,
        )?;

        // Step 6: Validate if configured
        if self.config.validate_after_build {
            let validator = TraceValidator::with_defaults();
            let report = validator.validate(&trace);
            if !report.is_valid {
                log::warn!("Concretized trace has validation issues: {}", report.summary());
            }
        }

        Ok(trace)
    }

    /// Step 1: Extract values from SMT model.
    fn extract_values(&self, model: &SmtModel) -> ConcreteResult<ExtractedValues> {
        let mut values = ExtractedValues {
            negotiated_cipher_id: None,
            offered_cipher_ids: Vec::new(),
            client_version: None,
            server_version: None,
            extensions: Vec::new(),
            adversary_actions: Vec::new(),
            step_count: 0,
            session_id: None,
            client_random: None,
            server_random: None,
            raw_assignments: BTreeMap::new(),
        };

        for (name, val) in &model.assignments {
            // Record raw numeric assignments for debugging
            if let Some(v) = val.as_u64() {
                values.raw_assignments.insert(name.clone(), v);
            }

            // Cipher suite variables
            if name.starts_with(&self.config.cipher_var_prefix) {
                if let Some(v) = val.as_u64() {
                    let cipher_id = v as u16;
                    if name.contains("selected") || name.contains("negotiated") {
                        values.negotiated_cipher_id = Some(cipher_id);
                    } else if name.contains("offered") || name.contains("client") {
                        values.offered_cipher_ids.push(cipher_id);
                    } else {
                        // Default: treat as offered
                        values.offered_cipher_ids.push(cipher_id);
                    }
                }
            }

            // Version variables
            if name.starts_with(&self.config.version_var_prefix) {
                if let Some(v) = val.as_u64() {
                    let version = self.extract_version(v);
                    if name.contains("client") || name.contains("offered") {
                        values.client_version = Some(version);
                    } else if name.contains("server") || name.contains("negotiated") {
                        values.server_version = Some(version);
                    } else {
                        // If only one version var, use as server version (downgraded)
                        if values.server_version.is_none() {
                            values.server_version = Some(version);
                        } else {
                            values.client_version = Some(version);
                        }
                    }
                }
            }

            // Extension variables
            if name.starts_with("ext_") || name.contains("extension") {
                if let Some(bytes) = val.as_bytes() {
                    // Try to parse extension ID from variable name
                    let ext_id = self.parse_extension_id_from_name(name).unwrap_or(0);
                    values.extensions.push((ext_id, bytes.to_vec()));
                } else if let Some(v) = val.as_u64() {
                    // Boolean or numeric extension presence
                    let ext_id = self.parse_extension_id_from_name(name).unwrap_or(0);
                    if v != 0 {
                        values.extensions.push((ext_id, Vec::new()));
                    }
                }
            }

            // Adversary action variables
            if name.starts_with(&self.config.adversary_var_prefix) {
                if let Some(v) = val.as_u64() {
                    let step = self.parse_step_index_from_name(name).unwrap_or(0);
                    values.adversary_actions.push(ExtractedAdversaryAction {
                        step,
                        action_type: v,
                        target_msg: step,
                        modified_field: None,
                        modified_value: None,
                    });
                }
            }

            // Step count
            if name == &self.config.step_count_var || name.contains("step_count") {
                if let Some(v) = val.as_u64() {
                    values.step_count = v as usize;
                }
            }

            // Session ID
            if name.contains("session_id") {
                if let Some(bytes) = val.as_bytes() {
                    values.session_id = Some(bytes.to_vec());
                }
            }

            // Random values
            if name.contains("client_random") {
                if let Some(bytes) = val.as_bytes() {
                    if bytes.len() == 32 {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(bytes);
                        values.client_random = Some(arr);
                    }
                }
            }
            if name.contains("server_random") {
                if let Some(bytes) = val.as_bytes() {
                    if bytes.len() == 32 {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(bytes);
                        values.server_random = Some(arr);
                    }
                }
            }
        }

        // Default step count
        if values.step_count == 0 {
            values.step_count = 4 + values.adversary_actions.len();
        }

        // Default offered ciphers if none extracted
        if values.offered_cipher_ids.is_empty() {
            values.offered_cipher_ids = vec![0xc02f, 0xc030, 0x009e, 0x009f, 0x002f, 0x0035];
            if let Some(neg_id) = values.negotiated_cipher_id {
                if !values.offered_cipher_ids.contains(&neg_id) {
                    values.offered_cipher_ids.push(neg_id);
                }
            }
        }

        Ok(values)
    }

    /// Extract a protocol version from a bitvector value.
    fn extract_version(&self, value: u64) -> ProtocolVersion {
        // Try interpreting as (major << 8) | minor
        let major = ((value >> 8) & 0xff) as u8;
        let minor = (value & 0xff) as u8;
        wire_to_version(major, minor)
    }

    /// Try to parse an extension ID from a variable name like "ext_0x0017".
    fn parse_extension_id_from_name(&self, name: &str) -> Option<u16> {
        // Try patterns: ext_0xXXXX, ext_NNNNN, extension_0xXXXX
        if let Some(hex_part) = name.split("0x").nth(1) {
            let hex_str: String = hex_part.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
            u16::from_str_radix(&hex_str, 16).ok()
        } else {
            // Try numeric suffix
            let digits: String = name.chars().rev().take_while(|c| c.is_ascii_digit()).collect::<String>().chars().rev().collect();
            digits.parse().ok()
        }
    }

    /// Try to parse a step index from a variable name like "adv_action_3".
    fn parse_step_index_from_name(&self, name: &str) -> Option<usize> {
        let digits: String = name
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        digits.parse().ok()
    }

    /// Step 2: Resolve cipher suite from ID.
    fn resolve_cipher(&self, cipher_id: Option<u16>) -> ConcreteResult<CipherSuite> {
        match cipher_id {
            Some(id) => {
                if let Some(cs) = self.cipher_lookup.get(&id) {
                    Ok(cs.clone())
                } else {
                    // Create a generic cipher suite for unknown IDs
                    log::warn!("Unknown cipher suite ID 0x{:04x}, creating generic entry", id);
                    Ok(CipherSuite::new(
                        id,
                        format!("UNKNOWN_0x{:04x}", id),
                        KeyExchange::Rsa,
                        AuthAlgorithm::Rsa,
                        BulkEncryption::Aes128,
                        MacAlgorithm::HmacSha256,
                    ))
                }
            }
            None => {
                // Default to a weak cipher (for downgrade attacks)
                Ok(CipherSuite::new(
                    0x000a,
                    "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
                    KeyExchange::Rsa,
                    AuthAlgorithm::Rsa,
                    BulkEncryption::TripleDes,
                    MacAlgorithm::HmacSha1,
                ))
            }
        }
    }

    /// Step 4: Reconstruct extension data from extracted values.
    fn reconstruct_extensions(
        &self,
        extracted: &ExtractedValues,
    ) -> ConcreteResult<Vec<Extension>> {
        let mut extensions = Vec::new();
        for &(ext_id, ref data) in &extracted.extensions {
            let name = match ext_id {
                0x0000 => "server_name",
                0x000a => "supported_groups",
                0x000b => "ec_point_formats",
                0x000d => "signature_algorithms",
                0x0017 => "extended_master_secret",
                0x0023 => "session_ticket",
                0x002b => "supported_versions",
                0x002d => "psk_key_exchange_modes",
                0x0033 => "key_share",
                0xff01 => "renegotiation_info",
                _ => "unknown",
            };
            let mut ext = Extension::new(ext_id, name, data.clone());
            // Mark renegotiation_info and supported_versions as critical
            ext.is_critical = matches!(ext_id, 0xff01 | 0x002b);
            extensions.push(ext);
        }
        Ok(extensions)
    }

    /// Step 5: Build the message sequence.
    fn build_message_sequence(
        &self,
        client_version: ProtocolVersion,
        server_version: ProtocolVersion,
        extracted: &ExtractedValues,
        negotiated_cipher: &CipherSuite,
        extensions: &[Extension],
    ) -> ConcreteResult<ConcreteTrace> {
        let is_ssh = matches!(client_version, ProtocolVersion::Ssh2);

        if is_ssh {
            self.build_ssh_sequence(extracted, negotiated_cipher)
        } else {
            self.build_tls_sequence(
                client_version,
                server_version,
                extracted,
                negotiated_cipher,
                extensions,
            )
        }
    }

    /// Build a TLS handshake attack trace.
    fn build_tls_sequence(
        &self,
        client_version: ProtocolVersion,
        server_version: ProtocolVersion,
        extracted: &ExtractedValues,
        negotiated_cipher: &CipherSuite,
        extensions: &[Extension],
    ) -> ConcreteResult<ConcreteTrace> {
        let mut builder = TraceBuilder::new(client_version, server_version)
            .with_library(&self.config.library_name, &self.config.library_version);

        let initial_ciphers: Vec<CipherSuite> = extracted
            .offered_cipher_ids
            .iter()
            .map(|&id| {
                self.cipher_lookup
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| CipherSuite::new(id, format!("0x{:04x}", id), KeyExchange::Rsa, AuthAlgorithm::Rsa, BulkEncryption::Aes128, MacAlgorithm::HmacSha256))
            })
            .collect();
        builder = builder.with_initial_ciphers(initial_ciphers);

        // ── Message 0: Client sends ClientHello ──
        let ch_params = ClientHelloParams::new(client_version, extracted.offered_cipher_ids.clone())
            .with_extensions(extensions.to_vec());
        let ch_params = if let Some(ref sid) = extracted.session_id {
            ch_params.with_session_id(sid.clone())
        } else {
            ch_params
        };
        let ch_params = if let Some(random) = extracted.client_random {
            ch_params.with_random(random)
        } else {
            ch_params
        };
        let ch_bytes = build_client_hello(&ch_params)?;
        let ch_idx = builder.client_send(ch_bytes, client_version, HandshakePhase::ClientHello);

        // Set parsed fields for ClientHello
        let mut ch_fields = ParsedFields::default();
        ch_fields.cipher_suites = extracted.offered_cipher_ids.clone();
        ch_fields.extensions = extensions.to_vec();
        ch_fields.content_type = Some(byte_encoding::tls_content_type::HANDSHAKE);
        ch_fields.handshake_type = Some(byte_encoding::tls_handshake_type::CLIENT_HELLO);
        ch_fields.record_version = Some(version_to_wire(client_version));
        builder.set_parsed_fields(ch_idx, ch_fields)?;

        // ── Adversary actions ──
        let has_adversary_actions = !extracted.adversary_actions.is_empty();

        if has_adversary_actions {
            // Determine adversary action type from model
            for adv in &extracted.adversary_actions {
                match adv.action_type {
                    // 0 = intercept + modify (version downgrade)
                    0 | 1 => {
                        // Intercept the ClientHello
                        builder.adversary_intercept(ch_idx, "client");

                        // Build a modified ClientHello with downgraded version
                        let modified_ch_params = ClientHelloParams::new(
                            server_version,
                            extracted.offered_cipher_ids.clone(),
                        );
                        let modified_ch_bytes = build_client_hello(&modified_ch_params)?;

                        let mods = vec![FieldModification::new(
                            "version",
                            {
                                let (maj, min) = version_to_wire(client_version);
                                vec![maj, min]
                            },
                            {
                                let (maj, min) = version_to_wire(server_version);
                                vec![maj, min]
                            },
                            1,
                            2,
                        )];

                        builder.adversary_modify(
                            ch_idx,
                            "client",
                            "server",
                            modified_ch_bytes,
                            server_version,
                            HandshakePhase::ClientHello,
                            mods,
                        );
                    }
                    // 2 = drop + inject
                    2 => {
                        builder.adversary_drop(ch_idx, "client");

                        let injected_params = ClientHelloParams::new(
                            server_version,
                            vec![negotiated_cipher.id],
                        );
                        let injected_bytes = build_client_hello(&injected_params)?;
                        builder.adversary_inject(
                            "server",
                            injected_bytes,
                            server_version,
                            HandshakePhase::ClientHello,
                        );
                    }
                    // 3 = intercept only
                    3 => {
                        builder.adversary_intercept(ch_idx, "client");
                    }
                    _ => {
                        // Default: intercept + modify
                        builder.adversary_intercept(ch_idx, "client");
                        let modified_ch_params = ClientHelloParams::new(
                            server_version,
                            extracted.offered_cipher_ids.clone(),
                        );
                        let modified_ch_bytes = build_client_hello(&modified_ch_params)?;
                        let mods = vec![FieldModification::new(
                            "version",
                            {
                                let (maj, min) = version_to_wire(client_version);
                                vec![maj, min]
                            },
                            {
                                let (maj, min) = version_to_wire(server_version);
                                vec![maj, min]
                            },
                            1,
                            2,
                        )];
                        builder.adversary_modify(
                            ch_idx,
                            "client",
                            "server",
                            modified_ch_bytes,
                            server_version,
                            HandshakePhase::ClientHello,
                            mods,
                        );
                    }
                }
            }
        } else {
            // No explicit adversary actions — construct a default version downgrade:
            // Adversary intercepts ClientHello and modifies the version field.
            builder.adversary_intercept(ch_idx, "client");
            let modified_params = ClientHelloParams::new(
                server_version,
                extracted.offered_cipher_ids.clone(),
            );
            let modified_bytes = build_client_hello(&modified_params)?;
            let mods = vec![FieldModification::new(
                "version",
                {
                    let (maj, min) = version_to_wire(client_version);
                    vec![maj, min]
                },
                {
                    let (maj, min) = version_to_wire(server_version);
                    vec![maj, min]
                },
                1,
                2,
            )];
            builder.adversary_modify(
                ch_idx,
                "client",
                "server",
                modified_bytes,
                server_version,
                HandshakePhase::ClientHello,
                mods,
            );
        }

        // ── Message N: Server responds with ServerHello ──
        let sh_params = ServerHelloParams::new(server_version, negotiated_cipher.id);
        let sh_params = if let Some(random) = extracted.server_random {
            sh_params.with_random(random)
        } else {
            sh_params
        };
        let sh_bytes = build_server_hello(&sh_params)?;
        let sh_idx = builder.server_send(sh_bytes, server_version, HandshakePhase::ServerHello);

        let mut sh_fields = ParsedFields::default();
        sh_fields.selected_cipher = Some(negotiated_cipher.id);
        sh_fields.content_type = Some(byte_encoding::tls_content_type::HANDSHAKE);
        sh_fields.handshake_type = Some(byte_encoding::tls_handshake_type::SERVER_HELLO);
        sh_fields.record_version = Some(version_to_wire(server_version));
        builder.set_parsed_fields(sh_idx, sh_fields)?;

        // ── ChangeCipherSpec (server) ──
        let ccs_bytes = build_change_cipher_spec(server_version)?;
        builder.server_send(ccs_bytes, server_version, HandshakePhase::ChangeCipherSpec);

        // ── Finished (server, simplified) ──
        let hs_hash = byte_encoding::sha256_hash(b"handshake_context_placeholder");
        let fin_bytes = build_tls_finished(server_version, &hs_hash, false)?;
        builder.server_send(fin_bytes, server_version, HandshakePhase::Finished);

        // ── ChangeCipherSpec (client) ──
        let ccs_client = build_change_cipher_spec(server_version)?;
        builder.client_send(ccs_client, server_version, HandshakePhase::ChangeCipherSpec);

        // ── Finished (client) ──
        let fin_client = build_tls_finished(server_version, &hs_hash, true)?;
        builder.client_send(fin_client, server_version, HandshakePhase::Finished);

        builder.set_negotiated_cipher(negotiated_cipher.clone());
        let trace = builder.build();

        self.validate_trace_size(&trace)?;
        Ok(trace)
    }

    /// Build an SSH key exchange attack trace.
    fn build_ssh_sequence(
        &self,
        extracted: &ExtractedValues,
        negotiated_cipher: &CipherSuite,
    ) -> ConcreteResult<ConcreteTrace> {
        let version = ProtocolVersion::Ssh2;
        let mut builder = TraceBuilder::new(version, version)
            .with_library(&self.config.library_name, &self.config.library_version);

        // Client KEX_INIT
        let client_kex = SshKexInitParams::new();
        let client_kex_bytes = build_ssh_kex_init(&client_kex)?;
        let ck_idx = builder.client_send(client_kex_bytes, version, HandshakePhase::KeyExchange);

        // Adversary intercepts and modifies KEX_INIT (downgrades algorithms)
        builder.adversary_intercept(ck_idx, "client");
        let weak_kex = SshKexInitParams::new()
            .with_kex_algorithms(vec!["diffie-hellman-group1-sha1".into()])
            .with_encryption(vec!["3des-cbc".into()]);
        let weak_kex_bytes = build_ssh_kex_init(&weak_kex)?;
        let mods = vec![FieldModification::new(
            "kex_algorithms",
            b"diffie-hellman-group14-sha256".to_vec(),
            b"diffie-hellman-group1-sha1".to_vec(),
            22,
            30,
        )];
        builder.adversary_modify(
            ck_idx,
            "client",
            "server",
            weak_kex_bytes,
            version,
            HandshakePhase::KeyExchange,
            mods,
        );

        // Server KEX_INIT (responds with weak algorithms)
        let server_kex = SshKexInitParams::new()
            .with_kex_algorithms(vec!["diffie-hellman-group1-sha1".into()])
            .with_encryption(vec!["3des-cbc".into()]);
        let server_kex_bytes = build_ssh_kex_init(&server_kex)?;
        builder.server_send(server_kex_bytes, version, HandshakePhase::KeyExchange);

        // NEWKEYS from server
        let newkeys_pkt = SshPacketEncoder::new(byte_encoding::ssh_msg_type::NEWKEYS, Vec::new());
        let newkeys_bytes = newkeys_pkt.encode_to_vec()
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        builder.server_send(newkeys_bytes.clone(), version, HandshakePhase::Finished);

        // NEWKEYS from client
        builder.client_send(newkeys_bytes, version, HandshakePhase::Finished);

        builder.set_negotiated_cipher(negotiated_cipher.clone());
        let trace = builder.build();
        self.validate_trace_size(&trace)?;
        Ok(trace)
    }

    /// Validate trace is within configured bounds.
    fn validate_trace_size(&self, trace: &ConcreteTrace) -> ConcreteResult<()> {
        if trace.message_count() > self.config.max_messages {
            return Err(ConcreteError::Concretization(format!(
                "Trace has {} messages, exceeds limit {}",
                trace.message_count(),
                self.config.max_messages
            )));
        }
        let total_bytes = trace.total_bytes();
        if total_bytes > self.config.max_total_bytes {
            return Err(ConcreteError::Concretization(format!(
                "Trace has {} bytes, exceeds limit {}",
                total_bytes, self.config.max_total_bytes
            )));
        }
        Ok(())
    }

    /// Check if a concrete trace demonstrates a genuine downgrade attack.
    pub fn is_genuine_downgrade(&self, trace: &ConcreteTrace) -> bool {
        // Must have active adversary
        if !trace.has_active_adversary() {
            return false;
        }
        // Must have a version downgrade
        if trace.downgrade_severity() == 0 {
            return false;
        }
        // Negotiated cipher should exist
        if trace.negotiated_cipher.is_none() {
            return false;
        }
        true
    }

    /// Generate a "spurious trace" descriptor from a trace that failed
    /// replay — used to generate refinement predicates.
    pub fn extract_spurious_info(
        &self,
        trace: &ConcreteTrace,
    ) -> BTreeMap<String, u64> {
        let mut info = BTreeMap::new();
        if let Some(ref cs) = trace.negotiated_cipher {
            info.insert("cipher_selected".into(), cs.id as u64);
        }
        let (maj, min) = version_to_wire(trace.downgraded_version);
        info.insert(
            "version_negotiated".into(),
            ((maj as u64) << 8) | min as u64,
        );
        info.insert("adversary_action_count".into(), trace.adversary_action_count() as u64);
        info.insert("message_count".into(), trace.message_count() as u64);
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sat_model() -> SmtModel {
        let mut model = SmtModel::new();
        model.insert("cipher_selected", SmtValue::BitVec(0x002f, 16));
        model.insert("version_client", SmtValue::BitVec(0x0303, 16));
        model.insert("version_server", SmtValue::BitVec(0x0300, 16));
        model.insert("cipher_offered_0", SmtValue::BitVec(0xc02f, 16));
        model.insert("cipher_offered_1", SmtValue::BitVec(0x002f, 16));
        model.insert("adv_action_0", SmtValue::BitVec(0, 8));
        model.insert("step_count", SmtValue::Int(6));
        model
    }

    #[test]
    fn test_concretize_basic() {
        let concretizer = Concretizer::with_defaults();
        let model = make_sat_model();
        let trace = concretizer.concretize(&model).unwrap();

        assert!(trace.message_count() > 0, "Should produce messages");
        assert!(trace.has_active_adversary(), "Should have adversary actions");
        assert_eq!(trace.initial_version, ProtocolVersion::Tls12);
        assert_eq!(trace.downgraded_version, ProtocolVersion::Ssl30);
    }

    #[test]
    fn test_concretize_unsat_model() {
        let concretizer = Concretizer::with_defaults();
        let mut model = SmtModel::new();
        model.is_sat = false;
        assert!(concretizer.concretize(&model).is_err());
    }

    #[test]
    fn test_concretize_minimal_model() {
        let concretizer = Concretizer::with_defaults();
        let model = SmtModel::new(); // empty but SAT
        let trace = concretizer.concretize(&model).unwrap();
        // Should produce a trace with default values
        assert!(trace.message_count() > 0);
    }

    #[test]
    fn test_concretize_with_extensions() {
        let mut model = make_sat_model();
        model.insert("ext_0x0017", SmtValue::BitVec(1, 8));
        model.insert("ext_0xff01", SmtValue::Bytes(vec![0x00]));

        let concretizer = Concretizer::with_defaults();
        let trace = concretizer.concretize(&model).unwrap();
        assert!(trace.message_count() > 0);
    }

    #[test]
    fn test_concretize_ssh() {
        let mut model = SmtModel::new();
        model.insert("version_client", SmtValue::BitVec(0x0200, 16));
        model.insert("version_server", SmtValue::BitVec(0x0200, 16));

        let concretizer = Concretizer::with_defaults();
        let trace = concretizer.concretize(&model).unwrap();
        assert!(trace.message_count() > 0);
    }

    #[test]
    fn test_resolve_known_cipher() {
        let concretizer = Concretizer::with_defaults();
        let cs = concretizer.resolve_cipher(Some(0xc02f)).unwrap();
        assert_eq!(cs.id, 0xc02f);
        assert!(cs.name.contains("ECDHE"));
    }

    #[test]
    fn test_resolve_unknown_cipher() {
        let concretizer = Concretizer::with_defaults();
        let cs = concretizer.resolve_cipher(Some(0xdead)).unwrap();
        assert_eq!(cs.id, 0xdead);
        assert!(cs.name.contains("UNKNOWN"));
    }

    #[test]
    fn test_is_genuine_downgrade() {
        let concretizer = Concretizer::with_defaults();
        let model = make_sat_model();
        let trace = concretizer.concretize(&model).unwrap();
        assert!(concretizer.is_genuine_downgrade(&trace));
    }

    #[test]
    fn test_extract_spurious_info() {
        let concretizer = Concretizer::with_defaults();
        let model = make_sat_model();
        let trace = concretizer.concretize(&model).unwrap();
        let info = concretizer.extract_spurious_info(&trace);
        assert!(info.contains_key("cipher_selected"));
        assert!(info.contains_key("version_negotiated"));
    }

    #[test]
    fn test_concretize_with_drop_inject_action() {
        let mut model = SmtModel::new();
        model.insert("cipher_selected", SmtValue::BitVec(0x002f, 16));
        model.insert("version_client", SmtValue::BitVec(0x0303, 16));
        model.insert("version_server", SmtValue::BitVec(0x0300, 16));
        model.insert("adv_action_0", SmtValue::BitVec(2, 8)); // drop + inject

        let concretizer = Concretizer::with_defaults();
        let trace = concretizer.concretize(&model).unwrap();
        assert!(trace.has_active_adversary());
        // Should have a drop step
        assert!(trace.steps.iter().any(|s| matches!(s, TraceStep::AdversaryDrop { .. })));
    }

    #[test]
    fn test_concretize_trace_validation() {
        let concretizer = Concretizer::with_defaults();
        let model = make_sat_model();
        let trace = concretizer.concretize(&model).unwrap();
        // The trace should have well-formed TLS records
        for msg in &trace.messages {
            assert!(!msg.raw_bytes.is_empty(), "Message {} should have bytes", msg.index);
        }
    }

    #[test]
    fn test_concretize_config() {
        let mut config = ConcretizerConfig::default();
        config.max_messages = 3;
        let concretizer = Concretizer::new(config);
        let model = make_sat_model();
        // This may fail because the trace has more than 3 messages
        let result = concretizer.concretize(&model);
        // Either succeeds with ≤3 messages or fails with size error
        match result {
            Ok(trace) => assert!(trace.message_count() <= 3),
            Err(ConcreteError::Concretization(msg)) => {
                assert!(msg.contains("exceeds limit"));
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_version_extraction() {
        let concretizer = Concretizer::with_defaults();
        assert_eq!(concretizer.extract_version(0x0303), ProtocolVersion::Tls12);
        assert_eq!(concretizer.extract_version(0x0301), ProtocolVersion::Tls10);
        assert_eq!(concretizer.extract_version(0x0300), ProtocolVersion::Ssl30);
    }

    #[test]
    fn test_parse_extension_id() {
        let concretizer = Concretizer::with_defaults();
        assert_eq!(concretizer.parse_extension_id_from_name("ext_0x0017"), Some(0x0017));
        assert_eq!(concretizer.parse_extension_id_from_name("ext_0xff01"), Some(0xff01));
        assert_eq!(concretizer.parse_extension_id_from_name("extension_23"), Some(23));
    }
}
