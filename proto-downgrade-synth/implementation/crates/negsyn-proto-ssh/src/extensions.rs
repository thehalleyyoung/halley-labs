//! SSH extension handling — RFC 8308 (ext-info) and OpenSSH strict-KEX.
//!
//! Extensions are signalled via the pseudo-algorithms `ext-info-c` /
//! `ext-info-s` in the KEX_INIT kex_algorithms list.  After NEWKEYS the
//! server (or client) may send SSH_MSG_EXT_INFO containing one or more
//! extension name/value pairs.
//!
//! The strict-KEX extension (`kex-strict-{c,s}-v00@openssh.com`) mitigates
//! the Terrapin attack (CVE-2023-48795) by enforcing sequence number resets
//! and rejecting unexpected messages during KEX.

use crate::{SshError, SshResult};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::io::{Cursor, Read, Write};

// ---------------------------------------------------------------------------
// SshExtension
// ---------------------------------------------------------------------------

/// Known SSH extension types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SshExtension {
    /// `server-sig-algs` — advertises signature algorithms the server accepts.
    ServerSigAlgs(Vec<String>),
    /// `no-flow-control` — disables SSH flow control.
    NoFlowControl(bool),
    /// `delay-compression` — defers compression until after auth.
    DelayCompression {
        compression_c2s: Vec<String>,
        compression_s2c: Vec<String>,
    },
    /// `ext-info-in-auth` — allows EXT_INFO during auth.
    ExtInfoInAuth,
    /// `global-requests-ok` — peer supports global requests before auth.
    GlobalRequestsOk,
    /// Unknown extension with raw data.
    Unknown {
        name: String,
        value: Vec<u8>,
    },
}

impl SshExtension {
    /// Returns the wire name of this extension.
    pub fn name(&self) -> &str {
        match self {
            Self::ServerSigAlgs(_) => "server-sig-algs",
            Self::NoFlowControl(_) => "no-flow-control",
            Self::DelayCompression { .. } => "delay-compression",
            Self::ExtInfoInAuth => "ext-info-in-auth",
            Self::GlobalRequestsOk => "global-requests-ok",
            Self::Unknown { name, .. } => name,
        }
    }

    /// Encode the extension value to bytes.
    pub fn value_bytes(&self) -> Vec<u8> {
        match self {
            Self::ServerSigAlgs(algs) => {
                let joined = algs.join(",");
                let mut buf = Vec::new();
                buf.write_u32::<BigEndian>(joined.len() as u32).unwrap();
                buf.write_all(joined.as_bytes()).unwrap();
                buf
            }
            Self::NoFlowControl(v) => {
                let s = if *v { "p" } else { "" };
                let mut buf = Vec::new();
                buf.write_u32::<BigEndian>(s.len() as u32).unwrap();
                buf.write_all(s.as_bytes()).unwrap();
                buf
            }
            Self::DelayCompression {
                compression_c2s,
                compression_s2c,
            } => {
                let mut buf = Vec::new();
                let c2s = compression_c2s.join(",");
                let s2c = compression_s2c.join(",");
                buf.write_u32::<BigEndian>(c2s.len() as u32).unwrap();
                buf.write_all(c2s.as_bytes()).unwrap();
                buf.write_u32::<BigEndian>(s2c.len() as u32).unwrap();
                buf.write_all(s2c.as_bytes()).unwrap();
                buf
            }
            Self::ExtInfoInAuth | Self::GlobalRequestsOk => {
                // Empty value
                let mut buf = Vec::new();
                buf.write_u32::<BigEndian>(0).unwrap();
                buf
            }
            Self::Unknown { value, .. } => {
                let mut buf = Vec::new();
                buf.write_u32::<BigEndian>(value.len() as u32).unwrap();
                buf.write_all(value).unwrap();
                buf
            }
        }
    }

    /// Parse an extension from name and raw value bytes.
    pub fn from_name_value(name: &str, value: &[u8]) -> Self {
        match name {
            "server-sig-algs" => {
                let s = std::str::from_utf8(value).unwrap_or("");
                let algs: Vec<String> = if s.is_empty() {
                    Vec::new()
                } else {
                    s.split(',').map(|n| n.to_string()).collect()
                };
                Self::ServerSigAlgs(algs)
            }
            "no-flow-control" => {
                let s = std::str::from_utf8(value).unwrap_or("");
                Self::NoFlowControl(s == "p")
            }
            "delay-compression" => {
                // Try to parse two name-lists
                if let Ok((c2s, s2c)) = parse_two_name_lists(value) {
                    Self::DelayCompression {
                        compression_c2s: c2s,
                        compression_s2c: s2c,
                    }
                } else {
                    Self::Unknown {
                        name: name.to_string(),
                        value: value.to_vec(),
                    }
                }
            }
            "ext-info-in-auth" => Self::ExtInfoInAuth,
            "global-requests-ok" => Self::GlobalRequestsOk,
            _ => Self::Unknown {
                name: name.to_string(),
                value: value.to_vec(),
            },
        }
    }

    /// Returns true if this extension has security implications.
    pub fn is_security_relevant(&self) -> bool {
        matches!(
            self,
            Self::ServerSigAlgs(_) | Self::DelayCompression { .. }
        )
    }

    /// Assess the security impact of this extension.
    pub fn security_impact(&self) -> ExtensionSecurityImpact {
        match self {
            Self::ServerSigAlgs(algs) => {
                let has_weak = algs.iter().any(|a| a == "ssh-rsa" || a == "ssh-dss");
                if has_weak {
                    ExtensionSecurityImpact::WeakAlgorithmsAdvertised
                } else {
                    ExtensionSecurityImpact::Neutral
                }
            }
            Self::NoFlowControl(true) => ExtensionSecurityImpact::Neutral,
            Self::DelayCompression { .. } => ExtensionSecurityImpact::Positive,
            _ => ExtensionSecurityImpact::Neutral,
        }
    }
}

impl fmt::Display for SshExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServerSigAlgs(algs) => {
                write!(f, "server-sig-algs: {}", algs.join(","))
            }
            Self::NoFlowControl(v) => write!(f, "no-flow-control: {}", v),
            Self::DelayCompression {
                compression_c2s,
                compression_s2c,
            } => write!(
                f,
                "delay-compression: c2s={}, s2c={}",
                compression_c2s.join(","),
                compression_s2c.join(",")
            ),
            Self::ExtInfoInAuth => write!(f, "ext-info-in-auth"),
            Self::GlobalRequestsOk => write!(f, "global-requests-ok"),
            Self::Unknown { name, value } => {
                write!(f, "{}: ({} bytes)", name, value.len())
            }
        }
    }
}

/// Security impact of an extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtensionSecurityImpact {
    /// Extension improves security posture.
    Positive,
    /// Extension has no security effect.
    Neutral,
    /// Extension advertises weak algorithms.
    WeakAlgorithmsAdvertised,
    /// Extension could enable an attack.
    Negative,
}

// ---------------------------------------------------------------------------
// ExtInfo
// ---------------------------------------------------------------------------

/// Parsed SSH_MSG_EXT_INFO message (RFC 8308).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtInfo {
    /// Number of extensions in this message.
    pub nr_extensions: u32,
    /// Parsed extension entries.
    pub extensions: Vec<SshExtension>,
}

impl ExtInfo {
    /// Create a new EXT_INFO with the given extensions.
    pub fn new(extensions: Vec<SshExtension>) -> Self {
        Self {
            nr_extensions: extensions.len() as u32,
            extensions,
        }
    }

    /// Parse from raw payload bytes (first byte should be SSH_MSG_EXT_INFO = 7).
    pub fn parse(data: &[u8]) -> SshResult<Self> {
        if data.is_empty() {
            return Err(SshError::ParseError("empty EXT_INFO".into()));
        }

        let mut cursor = Cursor::new(data);
        let msg_type = cursor.read_u8().map_err(|e| SshError::ParseError(e.to_string()))?;
        if msg_type != crate::constants::SSH_MSG_EXT_INFO {
            return Err(SshError::UnexpectedMessage {
                msg_type,
                state: "EXT_INFO parsing".into(),
            });
        }

        let nr_extensions = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| SshError::ParseError(format!("nr_extensions: {}", e)))?;

        let mut extensions = Vec::with_capacity(nr_extensions as usize);
        for i in 0..nr_extensions {
            let name_len = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| SshError::ParseError(format!("ext {} name len: {}", i, e)))?;
            let mut name_buf = vec![0u8; name_len as usize];
            cursor
                .read_exact(&mut name_buf)
                .map_err(|e| SshError::ParseError(format!("ext {} name: {}", i, e)))?;
            let name = std::str::from_utf8(&name_buf)
                .map_err(|e| SshError::ParseError(format!("ext {} name utf8: {}", i, e)))?;

            let value_len = cursor
                .read_u32::<BigEndian>()
                .map_err(|e| SshError::ParseError(format!("ext {} value len: {}", i, e)))?;
            let mut value_buf = vec![0u8; value_len as usize];
            cursor
                .read_exact(&mut value_buf)
                .map_err(|e| SshError::ParseError(format!("ext {} value: {}", i, e)))?;

            extensions.push(SshExtension::from_name_value(name, &value_buf));
        }

        Ok(Self {
            nr_extensions,
            extensions,
        })
    }

    /// Serialize to raw payload bytes (including the message type byte).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.push(crate::constants::SSH_MSG_EXT_INFO);
        buf.write_u32::<BigEndian>(self.nr_extensions).unwrap();

        for ext in &self.extensions {
            let name = ext.name();
            let name_bytes = name.as_bytes();
            buf.write_u32::<BigEndian>(name_bytes.len() as u32).unwrap();
            buf.write_all(name_bytes).unwrap();

            let value = ext.value_bytes();
            // value_bytes already includes its own length prefix for some types,
            // but the EXT_INFO wire format expects raw length + value.
            // We'll produce the raw value without inner length prefix for name-list types.
            let raw_value = match ext {
                SshExtension::ServerSigAlgs(algs) => algs.join(",").into_bytes(),
                SshExtension::NoFlowControl(v) => {
                    if *v { b"p".to_vec() } else { Vec::new() }
                }
                SshExtension::ExtInfoInAuth | SshExtension::GlobalRequestsOk => Vec::new(),
                SshExtension::Unknown { value: v, .. } => v.clone(),
                SshExtension::DelayCompression { .. } => value, // keep as-is
            };
            buf.write_u32::<BigEndian>(raw_value.len() as u32).unwrap();
            buf.write_all(&raw_value).unwrap();
        }

        buf
    }

    /// Look up the server-sig-algs extension.
    pub fn server_sig_algs(&self) -> Option<&[String]> {
        for ext in &self.extensions {
            if let SshExtension::ServerSigAlgs(algs) = ext {
                return Some(algs);
            }
        }
        None
    }

    /// Look up whether no-flow-control is advertised.
    pub fn no_flow_control(&self) -> Option<bool> {
        for ext in &self.extensions {
            if let SshExtension::NoFlowControl(v) = ext {
                return Some(*v);
            }
        }
        None
    }

    /// Returns all security-relevant extensions.
    pub fn security_relevant(&self) -> Vec<&SshExtension> {
        self.extensions.iter().filter(|e| e.is_security_relevant()).collect()
    }

    /// Returns the overall security impact.
    pub fn overall_security_impact(&self) -> ExtensionSecurityImpact {
        let mut worst = ExtensionSecurityImpact::Neutral;
        for ext in &self.extensions {
            let impact = ext.security_impact();
            match impact {
                ExtensionSecurityImpact::Negative => return ExtensionSecurityImpact::Negative,
                ExtensionSecurityImpact::WeakAlgorithmsAdvertised => {
                    worst = ExtensionSecurityImpact::WeakAlgorithmsAdvertised;
                }
                ExtensionSecurityImpact::Positive if worst == ExtensionSecurityImpact::Neutral => {
                    worst = ExtensionSecurityImpact::Positive;
                }
                _ => {}
            }
        }
        worst
    }
}

impl fmt::Display for ExtInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SSH_MSG_EXT_INFO ({} extensions):", self.nr_extensions)?;
        for ext in &self.extensions {
            writeln!(f, "  {}", ext)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StrictKex
// ---------------------------------------------------------------------------

/// Strict-KEX extension state (Terrapin mitigation).
///
/// When both sides include `kex-strict-{c,s}-v00@openssh.com` in their
/// initial KEXINIT, strict mode is activated.  In strict mode:
///
/// 1. Sequence numbers are reset to 0 after NEWKEYS.
/// 2. During initial KEX, the first message after the version exchange MUST
///    be KEXINIT.  Any other message (e.g., SSH_MSG_IGNORE) causes a
///    protocol error.
/// 3. No unexpected messages are allowed between KEXINIT and NEWKEYS.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrictKex {
    /// Whether the client advertised strict-KEX.
    pub client_supported: bool,
    /// Whether the server advertised strict-KEX.
    pub server_supported: bool,
    /// Whether strict-KEX is active (both sides support it).
    pub active: bool,
    /// Whether the initial KEX has completed (strict-KEX rules apply differently
    /// for re-keying).
    pub initial_kex_done: bool,
}

impl StrictKex {
    pub fn new() -> Self {
        Self {
            client_supported: false,
            server_supported: false,
            active: false,
            initial_kex_done: false,
        }
    }

    /// Update from a client's KEXINIT message.
    pub fn process_client_kexinit(&mut self, kex_init: &crate::kex::KexInit) {
        self.client_supported = kex_init.has_strict_kex_client();
        self.update_active();
    }

    /// Update from a server's KEXINIT message.
    pub fn process_server_kexinit(&mut self, kex_init: &crate::kex::KexInit) {
        self.server_supported = kex_init.has_strict_kex_server();
        self.update_active();
    }

    fn update_active(&mut self) {
        self.active = self.client_supported && self.server_supported;
    }

    /// Validate that a message is allowed in strict-KEX mode.
    ///
    /// During the initial KEX:
    ///   - After version exchange, only KEXINIT is allowed.
    ///   - Between KEXINIT and NEWKEYS, only KEX-specific messages are allowed.
    ///
    /// Returns `Ok(())` if the message is allowed, or an error describing the
    /// violation.
    pub fn validate_message(
        &self,
        msg_type: u8,
        phase: StrictKexPhase,
    ) -> SshResult<()> {
        if !self.active || self.initial_kex_done {
            return Ok(()); // strict-KEX only constrains the initial KEX
        }

        match phase {
            StrictKexPhase::BeforeKexInit => {
                // Only KEXINIT allowed
                if msg_type != crate::constants::SSH_MSG_KEXINIT {
                    return Err(SshError::StrictKexViolation(format!(
                        "expected KEXINIT (20) before any other message, got {}",
                        msg_type
                    )));
                }
            }
            StrictKexPhase::DuringKex => {
                // Only KEX messages (20-49) and DISCONNECT(1) allowed
                let allowed = msg_type == crate::constants::SSH_MSG_DISCONNECT
                    || (20..=49).contains(&msg_type);
                if !allowed {
                    return Err(SshError::StrictKexViolation(format!(
                        "unexpected message type {} during KEX",
                        msg_type
                    )));
                }
            }
            StrictKexPhase::AfterNewKeys => {
                // Normal operation resumes
            }
        }

        Ok(())
    }

    /// Mark the initial KEX as complete.
    pub fn mark_initial_kex_done(&mut self) {
        self.initial_kex_done = true;
    }

    /// Check if sequence numbers should be reset after NEWKEYS.
    pub fn should_reset_sequence_numbers(&self) -> bool {
        self.active
    }

    /// Describe the security properties.
    pub fn security_description(&self) -> String {
        if self.active {
            "Strict-KEX active: sequence numbers reset after NEWKEYS, \
             no extraneous messages during initial KEX. \
             Mitigates Terrapin (CVE-2023-48795)."
                .to_string()
        } else if self.client_supported || self.server_supported {
            format!(
                "Strict-KEX partially supported (client={}, server={}). \
                 Terrapin mitigation NOT active.",
                self.client_supported, self.server_supported
            )
        } else {
            "Strict-KEX not supported. Vulnerable to Terrapin (CVE-2023-48795) \
             if using ChaCha20-Poly1305 or CBC+ETM ciphers."
                .to_string()
        }
    }
}

impl Default for StrictKex {
    fn default() -> Self {
        Self::new()
    }
}

/// Phase during the strict-KEX handshake for message validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrictKexPhase {
    /// After version exchange, before KEXINIT is sent/received.
    BeforeKexInit,
    /// Between KEXINIT and NEWKEYS.
    DuringKex,
    /// After NEWKEYS — normal operation.
    AfterNewKeys,
}

// ---------------------------------------------------------------------------
// Extension compatibility checking
// ---------------------------------------------------------------------------

/// Check compatibility between a set of extensions and a negotiated cipher suite.
#[derive(Debug, Clone)]
pub struct ExtensionCompatibility;

impl ExtensionCompatibility {
    /// Check if the `server-sig-algs` extension is consistent with the negotiated
    /// host key algorithm.
    pub fn check_sig_algs_consistency(
        ext_info: &ExtInfo,
        negotiated_host_key: &str,
    ) -> ExtensionCompatibilityResult {
        if let Some(sig_algs) = ext_info.server_sig_algs() {
            if sig_algs.iter().any(|a| a == negotiated_host_key) {
                ExtensionCompatibilityResult::Compatible
            } else {
                ExtensionCompatibilityResult::Inconsistent {
                    reason: format!(
                        "server-sig-algs does not include negotiated host key '{}'",
                        negotiated_host_key
                    ),
                }
            }
        } else {
            ExtensionCompatibilityResult::NotApplicable
        }
    }

    /// Check if delay-compression is compatible with the negotiated compression.
    pub fn check_delay_compression_consistency(
        ext_info: &ExtInfo,
        negotiated_comp_c2s: &str,
        negotiated_comp_s2c: &str,
    ) -> ExtensionCompatibilityResult {
        for ext in &ext_info.extensions {
            if let SshExtension::DelayCompression {
                compression_c2s,
                compression_s2c,
            } = ext
            {
                let c2s_ok = compression_c2s.iter().any(|a| a == negotiated_comp_c2s)
                    || negotiated_comp_c2s == "none";
                let s2c_ok = compression_s2c.iter().any(|a| a == negotiated_comp_s2c)
                    || negotiated_comp_s2c == "none";

                if c2s_ok && s2c_ok {
                    return ExtensionCompatibilityResult::Compatible;
                } else {
                    return ExtensionCompatibilityResult::Inconsistent {
                        reason: format!(
                            "delay-compression does not include negotiated compression \
                             (c2s={}, s2c={})",
                            negotiated_comp_c2s, negotiated_comp_s2c
                        ),
                    };
                }
            }
        }
        ExtensionCompatibilityResult::NotApplicable
    }

    /// Aggregate all compatibility checks.
    pub fn check_all(
        ext_info: &ExtInfo,
        negotiated_host_key: &str,
        negotiated_comp_c2s: &str,
        negotiated_comp_s2c: &str,
    ) -> Vec<(String, ExtensionCompatibilityResult)> {
        vec![
            (
                "sig-algs-consistency".into(),
                Self::check_sig_algs_consistency(ext_info, negotiated_host_key),
            ),
            (
                "delay-compression-consistency".into(),
                Self::check_delay_compression_consistency(
                    ext_info,
                    negotiated_comp_c2s,
                    negotiated_comp_s2c,
                ),
            ),
        ]
    }
}

/// Result of an extension compatibility check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionCompatibilityResult {
    Compatible,
    Inconsistent { reason: String },
    NotApplicable,
}

// ---------------------------------------------------------------------------
// Extension negotiation tracker
// ---------------------------------------------------------------------------

/// Tracks which extensions have been negotiated during a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtensionNegotiationState {
    /// Whether ext-info-c was signalled by the client.
    pub client_ext_info: bool,
    /// Whether ext-info-s was signalled by the server.
    pub server_ext_info: bool,
    /// Extensions received from the server.
    pub server_extensions: Vec<SshExtension>,
    /// Extensions received from the client (rare — only with ext-info-in-auth).
    pub client_extensions: Vec<SshExtension>,
    /// Strict-KEX state.
    pub strict_kex: StrictKex,
}

impl ExtensionNegotiationState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_client_kexinit(&mut self, kex_init: &crate::kex::KexInit) {
        self.client_ext_info = kex_init.has_ext_info_c();
        self.strict_kex.process_client_kexinit(kex_init);
    }

    pub fn process_server_kexinit(&mut self, kex_init: &crate::kex::KexInit) {
        self.server_ext_info = kex_init.has_ext_info_s();
        self.strict_kex.process_server_kexinit(kex_init);
    }

    pub fn process_server_ext_info(&mut self, ext_info: &ExtInfo) {
        self.server_extensions = ext_info.extensions.clone();
    }

    pub fn process_client_ext_info(&mut self, ext_info: &ExtInfo) {
        self.client_extensions = ext_info.extensions.clone();
    }

    /// Retrieve a map of all extension names → values for reporting.
    pub fn extension_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for ext in &self.server_extensions {
            map.insert(format!("server:{}", ext.name()), format!("{}", ext));
        }
        for ext in &self.client_extensions {
            map.insert(format!("client:{}", ext.name()), format!("{}", ext));
        }
        map
    }

    /// Returns true if strict-KEX is active.
    pub fn strict_kex_active(&self) -> bool {
        self.strict_kex.active
    }

    /// Returns the server-sig-algs if available.
    pub fn server_sig_algs(&self) -> Option<Vec<&str>> {
        for ext in &self.server_extensions {
            if let SshExtension::ServerSigAlgs(algs) = ext {
                return Some(algs.iter().map(|s| s.as_str()).collect());
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_two_name_lists(data: &[u8]) -> Result<(Vec<String>, Vec<String>), SshError> {
    let mut cursor = Cursor::new(data);

    let len1 = cursor
        .read_u32::<BigEndian>()
        .map_err(|e| SshError::ParseError(format!("name-list 1 len: {}", e)))?;
    let mut buf1 = vec![0u8; len1 as usize];
    cursor
        .read_exact(&mut buf1)
        .map_err(|e| SshError::ParseError(format!("name-list 1: {}", e)))?;
    let s1 = std::str::from_utf8(&buf1)
        .map_err(|e| SshError::ParseError(format!("name-list 1 utf8: {}", e)))?;

    let len2 = cursor
        .read_u32::<BigEndian>()
        .map_err(|e| SshError::ParseError(format!("name-list 2 len: {}", e)))?;
    let mut buf2 = vec![0u8; len2 as usize];
    cursor
        .read_exact(&mut buf2)
        .map_err(|e| SshError::ParseError(format!("name-list 2: {}", e)))?;
    let s2 = std::str::from_utf8(&buf2)
        .map_err(|e| SshError::ParseError(format!("name-list 2 utf8: {}", e)))?;

    let list1: Vec<String> = if s1.is_empty() {
        Vec::new()
    } else {
        s1.split(',').map(|n| n.to_string()).collect()
    };
    let list2: Vec<String> = if s2.is_empty() {
        Vec::new()
    } else {
        s2.split(',').map(|n| n.to_string()).collect()
    };

    Ok((list1, list2))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_info_roundtrip() {
        let ext_info = ExtInfo::new(vec![
            SshExtension::ServerSigAlgs(vec![
                "ssh-ed25519".into(),
                "rsa-sha2-256".into(),
                "rsa-sha2-512".into(),
            ]),
            SshExtension::NoFlowControl(true),
        ]);

        let bytes = ext_info.to_bytes();
        let parsed = ExtInfo::parse(&bytes).unwrap();

        assert_eq!(parsed.nr_extensions, 2);
        if let SshExtension::ServerSigAlgs(algs) = &parsed.extensions[0] {
            assert_eq!(algs.len(), 3);
            assert_eq!(algs[0], "ssh-ed25519");
        } else {
            panic!("expected ServerSigAlgs");
        }
    }

    #[test]
    fn ext_info_server_sig_algs() {
        let ext_info = ExtInfo::new(vec![SshExtension::ServerSigAlgs(vec![
            "ssh-ed25519".into(),
            "rsa-sha2-256".into(),
        ])]);
        let algs = ext_info.server_sig_algs().unwrap();
        assert_eq!(algs.len(), 2);
    }

    #[test]
    fn strict_kex_activation() {
        let mut sk = StrictKex::new();
        assert!(!sk.active);

        let client_ki = crate::kex::KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .with_strict_kex_client()
            .build();
        sk.process_client_kexinit(&client_ki);
        assert!(sk.client_supported);
        assert!(!sk.active); // server hasn't signalled yet

        let server_ki = crate::kex::KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .with_strict_kex_server()
            .build();
        sk.process_server_kexinit(&server_ki);
        assert!(sk.active);
    }

    #[test]
    fn strict_kex_rejects_ignore_before_kexinit() {
        let mut sk = StrictKex::new();
        sk.client_supported = true;
        sk.server_supported = true;
        sk.active = true;

        // SSH_MSG_IGNORE (2) should be rejected before KEXINIT
        let result = sk.validate_message(2, StrictKexPhase::BeforeKexInit);
        assert!(result.is_err());

        // KEXINIT (20) should be allowed
        let result = sk.validate_message(20, StrictKexPhase::BeforeKexInit);
        assert!(result.is_ok());
    }

    #[test]
    fn strict_kex_allows_kex_messages_during_kex() {
        let mut sk = StrictKex::new();
        sk.active = true;

        // KEX messages (20-49) allowed
        assert!(sk.validate_message(20, StrictKexPhase::DuringKex).is_ok());
        assert!(sk.validate_message(21, StrictKexPhase::DuringKex).is_ok());
        assert!(sk.validate_message(30, StrictKexPhase::DuringKex).is_ok());

        // DISCONNECT allowed
        assert!(sk.validate_message(1, StrictKexPhase::DuringKex).is_ok());

        // SSH_MSG_IGNORE not allowed during KEX
        assert!(sk.validate_message(2, StrictKexPhase::DuringKex).is_err());

        // SERVICE_REQUEST not allowed during KEX
        assert!(sk.validate_message(5, StrictKexPhase::DuringKex).is_err());
    }

    #[test]
    fn strict_kex_inactive_allows_all() {
        let sk = StrictKex::new(); // not active
        assert!(sk.validate_message(2, StrictKexPhase::BeforeKexInit).is_ok());
        assert!(sk.validate_message(5, StrictKexPhase::DuringKex).is_ok());
    }

    #[test]
    fn strict_kex_after_initial_allows_all() {
        let mut sk = StrictKex::new();
        sk.active = true;
        sk.mark_initial_kex_done();
        // After initial KEX, even IGNORE is allowed during rekey
        assert!(sk.validate_message(2, StrictKexPhase::BeforeKexInit).is_ok());
    }

    #[test]
    fn extension_security_impact() {
        let weak_ext =
            SshExtension::ServerSigAlgs(vec!["ssh-rsa".into(), "ssh-ed25519".into()]);
        assert_eq!(
            weak_ext.security_impact(),
            ExtensionSecurityImpact::WeakAlgorithmsAdvertised
        );

        let good_ext = SshExtension::ServerSigAlgs(vec!["ssh-ed25519".into()]);
        assert_eq!(good_ext.security_impact(), ExtensionSecurityImpact::Neutral);
    }

    #[test]
    fn extension_compatibility_check() {
        let ext_info = ExtInfo::new(vec![SshExtension::ServerSigAlgs(vec![
            "ssh-ed25519".into(),
            "rsa-sha2-256".into(),
        ])]);

        let result = ExtensionCompatibility::check_sig_algs_consistency(
            &ext_info,
            "ssh-ed25519",
        );
        assert_eq!(result, ExtensionCompatibilityResult::Compatible);

        let result = ExtensionCompatibility::check_sig_algs_consistency(
            &ext_info,
            "ssh-dss",
        );
        assert!(matches!(
            result,
            ExtensionCompatibilityResult::Inconsistent { .. }
        ));
    }

    #[test]
    fn extension_negotiation_state_tracking() {
        let mut ens = ExtensionNegotiationState::new();

        let client_ki = crate::kex::KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .with_ext_info_c()
            .with_strict_kex_client()
            .build();
        ens.process_client_kexinit(&client_ki);
        assert!(ens.client_ext_info);

        let server_ki = crate::kex::KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .with_ext_info_s()
            .with_strict_kex_server()
            .build();
        ens.process_server_kexinit(&server_ki);
        assert!(ens.server_ext_info);
        assert!(ens.strict_kex_active());
    }

    #[test]
    fn unknown_extension_handling() {
        let ext = SshExtension::from_name_value("custom-extension", b"\x01\x02\x03");
        assert!(matches!(ext, SshExtension::Unknown { .. }));
        assert_eq!(ext.name(), "custom-extension");
    }

    #[test]
    fn overall_security_impact_weak() {
        let ext_info = ExtInfo::new(vec![
            SshExtension::ServerSigAlgs(vec!["ssh-rsa".into()]),
            SshExtension::NoFlowControl(true),
        ]);
        assert_eq!(
            ext_info.overall_security_impact(),
            ExtensionSecurityImpact::WeakAlgorithmsAdvertised
        );
    }

    #[test]
    fn security_description() {
        let mut sk = StrictKex::new();
        sk.active = true;
        let desc = sk.security_description();
        assert!(desc.contains("Terrapin"));
        assert!(desc.contains("CVE-2023-48795"));
    }

    #[test]
    fn ext_info_display() {
        let ext_info = ExtInfo::new(vec![SshExtension::ServerSigAlgs(vec![
            "ssh-ed25519".into(),
        ])]);
        let s = format!("{}", ext_info);
        assert!(s.contains("SSH_MSG_EXT_INFO"));
        assert!(s.contains("server-sig-algs"));
    }

    #[test]
    fn extension_display() {
        let ext = SshExtension::NoFlowControl(true);
        assert_eq!(format!("{}", ext), "no-flow-control: true");
    }
}
