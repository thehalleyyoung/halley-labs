//! Concrete trace representation and builder.
//!
//! Provides [`ConcreteTrace`], [`ConcreteMessage`], [`TraceStep`], and
//! [`TraceBuilder`] — the output of the concretization phase.

use crate::{AdversaryAction, ConcreteError, ConcreteResult};
use crate::{CipherSuite, Extension, HandshakePhase, ProtocolVersion};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Write;

// ── TraceStep ────────────────────────────────────────────────────────────

/// A single step in an attack trace, describing the action taken.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceStep {
    /// Client sends a message to the network (interceptable by adversary).
    ClientSend {
        message_index: usize,
    },
    /// Server sends a message to the network.
    ServerSend {
        message_index: usize,
    },
    /// Adversary intercepts a message in transit.
    AdversaryIntercept {
        original_index: usize,
        from: String,
    },
    /// Adversary injects a new message.
    AdversaryInject {
        injected_index: usize,
        to: String,
    },
    /// Adversary drops a message.
    AdversaryDrop {
        dropped_index: usize,
        from: String,
    },
    /// Adversary modifies a message in transit.
    AdversaryModify {
        original_index: usize,
        modified_index: usize,
        from: String,
        to: String,
        modifications: Vec<FieldModification>,
    },
}

impl TraceStep {
    pub fn is_adversary_action(&self) -> bool {
        !matches!(self, TraceStep::ClientSend { .. } | TraceStep::ServerSend { .. })
    }

    pub fn message_index(&self) -> Option<usize> {
        match self {
            TraceStep::ClientSend { message_index } => Some(*message_index),
            TraceStep::ServerSend { message_index } => Some(*message_index),
            TraceStep::AdversaryIntercept { original_index, .. } => Some(*original_index),
            TraceStep::AdversaryInject { injected_index, .. } => Some(*injected_index),
            TraceStep::AdversaryDrop { dropped_index, .. } => Some(*dropped_index),
            TraceStep::AdversaryModify { modified_index, .. } => Some(*modified_index),
        }
    }

    pub fn step_name(&self) -> &'static str {
        match self {
            TraceStep::ClientSend { .. } => "ClientSend",
            TraceStep::ServerSend { .. } => "ServerSend",
            TraceStep::AdversaryIntercept { .. } => "AdversaryIntercept",
            TraceStep::AdversaryInject { .. } => "AdversaryInject",
            TraceStep::AdversaryDrop { .. } => "AdversaryDrop",
            TraceStep::AdversaryModify { .. } => "AdversaryModify",
        }
    }
}

impl fmt::Display for TraceStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceStep::ClientSend { message_index } => {
                write!(f, "Client → Network  [msg #{}]", message_index)
            }
            TraceStep::ServerSend { message_index } => {
                write!(f, "Server → Network  [msg #{}]", message_index)
            }
            TraceStep::AdversaryIntercept { original_index, from } => {
                write!(f, "Adversary intercepts msg #{} from {}", original_index, from)
            }
            TraceStep::AdversaryInject { injected_index, to } => {
                write!(f, "Adversary injects msg #{} to {}", injected_index, to)
            }
            TraceStep::AdversaryDrop { dropped_index, from } => {
                write!(f, "Adversary drops msg #{} from {}", dropped_index, from)
            }
            TraceStep::AdversaryModify {
                original_index,
                modified_index,
                from,
                to,
                modifications,
            } => {
                write!(
                    f,
                    "Adversary modifies msg #{} → #{} ({} → {}, {} changes)",
                    original_index,
                    modified_index,
                    from,
                    to,
                    modifications.len()
                )
            }
        }
    }
}

/// A field-level modification applied by the adversary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldModification {
    pub field_name: String,
    pub original_value: Vec<u8>,
    pub modified_value: Vec<u8>,
    pub byte_offset: usize,
    pub byte_length: usize,
}

impl FieldModification {
    pub fn new(
        field_name: impl Into<String>,
        original: Vec<u8>,
        modified: Vec<u8>,
        offset: usize,
        length: usize,
    ) -> Self {
        Self {
            field_name: field_name.into(),
            original_value: original,
            modified_value: modified,
            byte_offset: offset,
            byte_length: length,
        }
    }
}

// ── ConcreteMessage ──────────────────────────────────────────────────────

/// A concrete protocol message with raw bytes and parsed fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConcreteMessage {
    pub index: usize,
    pub sender: String,
    pub receiver: String,
    pub raw_bytes: Vec<u8>,
    pub protocol_version: ProtocolVersion,
    pub handshake_phase: HandshakePhase,
    pub parsed_fields: ParsedFields,
    pub timestamp_us: u64,
    pub is_encrypted: bool,
}

impl ConcreteMessage {
    pub fn new(
        index: usize,
        sender: impl Into<String>,
        receiver: impl Into<String>,
        raw_bytes: Vec<u8>,
        version: ProtocolVersion,
        phase: HandshakePhase,
    ) -> Self {
        Self {
            index,
            sender: sender.into(),
            receiver: receiver.into(),
            raw_bytes,
            protocol_version: version,
            handshake_phase: phase,
            parsed_fields: ParsedFields::default(),
            timestamp_us: 0,
            is_encrypted: false,
        }
    }

    pub fn byte_length(&self) -> usize {
        self.raw_bytes.len()
    }

    pub fn hex_dump(&self) -> String {
        self.raw_bytes
            .chunks(16)
            .enumerate()
            .map(|(i, chunk)| {
                let hex: Vec<String> = chunk.iter().map(|b| format!("{:02x}", b)).collect();
                let ascii: String = chunk
                    .iter()
                    .map(|&b| if (0x20..=0x7e).contains(&b) { b as char } else { '.' })
                    .collect();
                format!("{:04x}  {:<48}  {}", i * 16, hex.join(" "), ascii)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl fmt::Display for ConcreteMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[msg #{:03}] {} → {} | {} {:?} | {} bytes{}",
            self.index,
            self.sender,
            self.receiver,
            self.protocol_version,
            self.handshake_phase,
            self.raw_bytes.len(),
            if self.is_encrypted { " [encrypted]" } else { "" },
        )
    }
}

/// Parsed protocol fields extracted from a concrete message.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ParsedFields {
    pub content_type: Option<u8>,
    pub handshake_type: Option<u8>,
    pub cipher_suites: Vec<u16>,
    pub selected_cipher: Option<u16>,
    pub extensions: Vec<Extension>,
    pub session_id: Option<Vec<u8>>,
    pub random: Option<Vec<u8>>,
    pub compression_methods: Vec<u8>,
    pub record_version: Option<(u8, u8)>,
    pub fragment_length: Option<u16>,
    pub ssh_packet_type: Option<u8>,
    pub ssh_algorithms: Vec<Vec<String>>,
}

// ── ConcreteTrace ────────────────────────────────────────────────────────

/// A complete concrete attack trace: an ordered sequence of messages and
/// adversary actions that demonstrate a protocol downgrade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcreteTrace {
    pub messages: Vec<ConcreteMessage>,
    pub steps: Vec<TraceStep>,
    pub adversary_actions: Vec<AdversaryAction>,
    pub initial_version: ProtocolVersion,
    pub downgraded_version: ProtocolVersion,
    pub initial_ciphers: Vec<CipherSuite>,
    pub negotiated_cipher: Option<CipherSuite>,
    pub metadata: TraceMetadata,
}

/// Metadata about a concrete trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub trace_id: String,
    pub library_name: String,
    pub library_version: String,
    pub generation_time_us: u64,
    pub cegar_iterations: usize,
    pub is_validated: bool,
    pub validation_errors: Vec<String>,
}

impl ConcreteTrace {
    pub fn new(initial: ProtocolVersion, downgraded: ProtocolVersion) -> Self {
        Self {
            messages: Vec::new(),
            steps: Vec::new(),
            adversary_actions: Vec::new(),
            initial_version: initial,
            downgraded_version: downgraded,
            initial_ciphers: Vec::new(),
            negotiated_cipher: None,
            metadata: TraceMetadata::default(),
        }
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn adversary_action_count(&self) -> usize {
        self.adversary_actions.len()
    }

    pub fn has_active_adversary(&self) -> bool {
        self.steps.iter().any(|s| s.is_adversary_action())
    }

    pub fn total_bytes(&self) -> usize {
        self.messages.iter().map(|m| m.raw_bytes.len()).sum()
    }

    pub fn downgrade_severity(&self) -> u32 {
        let initial = self.initial_version.security_level();
        let downgraded = self.downgraded_version.security_level();
        initial.saturating_sub(downgraded)
    }

    /// Serialize the trace to a pcap-like binary format.
    ///
    /// Format: global header, then per-packet (timestamp, length, bytes).
    pub fn to_pcap_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24 + self.messages.len() * 64);
        // Global header (simplified pcap magic)
        buf.extend_from_slice(&0xa1b2c3d4u32.to_le_bytes()); // magic
        buf.extend_from_slice(&2u16.to_le_bytes()); // version major
        buf.extend_from_slice(&4u16.to_le_bytes()); // version minor
        buf.extend_from_slice(&0i32.to_le_bytes()); // thiszone
        buf.extend_from_slice(&0u32.to_le_bytes()); // sigfigs
        buf.extend_from_slice(&65535u32.to_le_bytes()); // snaplen
        buf.extend_from_slice(&147u32.to_le_bytes()); // network (user-defined)

        for msg in &self.messages {
            let ts_sec = (msg.timestamp_us / 1_000_000) as u32;
            let ts_usec = (msg.timestamp_us % 1_000_000) as u32;
            let incl_len = msg.raw_bytes.len() as u32;
            let orig_len = incl_len;

            buf.extend_from_slice(&ts_sec.to_le_bytes());
            buf.extend_from_slice(&ts_usec.to_le_bytes());
            buf.extend_from_slice(&incl_len.to_le_bytes());
            buf.extend_from_slice(&orig_len.to_le_bytes());
            buf.extend_from_slice(&msg.raw_bytes);
        }
        buf
    }

    /// Pretty-print the trace for human review.
    pub fn pretty_print(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("╔══════════════════════════════════════════════════════════╗\n");
        out.push_str("║              PROTOCOL DOWNGRADE ATTACK TRACE            ║\n");
        out.push_str("╠══════════════════════════════════════════════════════════╣\n");
        out.push_str(&format!(
            "║ Target:     {} → {}  (severity: {})\n",
            self.initial_version,
            self.downgraded_version,
            self.downgrade_severity()
        ));
        out.push_str(&format!(
            "║ Messages:   {}   Steps: {}   Adversary actions: {}\n",
            self.message_count(),
            self.step_count(),
            self.adversary_action_count()
        ));
        if let Some(ref cs) = self.negotiated_cipher {
            out.push_str(&format!("║ Negotiated: {}\n", cs));
        }
        out.push_str("╠══════════════════════════════════════════════════════════╣\n");

        for (i, step) in self.steps.iter().enumerate() {
            let marker = if step.is_adversary_action() { "⚡" } else { "  " };
            out.push_str(&format!("║ {} Step {:3}: {}\n", marker, i, step));

            if let Some(msg_idx) = step.message_index() {
                if let Some(msg) = self.messages.get(msg_idx) {
                    out.push_str(&format!("║           {}\n", msg));
                    if msg.raw_bytes.len() <= 64 {
                        out.push_str(&format!("║           Hex: {}\n", hex::encode(&msg.raw_bytes)));
                    } else {
                        out.push_str(&format!(
                            "║           Hex: {}... ({} bytes total)\n",
                            hex::encode(&msg.raw_bytes[..32]),
                            msg.raw_bytes.len()
                        ));
                    }
                }
            }
        }

        out.push_str("╚══════════════════════════════════════════════════════════╝\n");
        out
    }

    /// Write the trace as JSON.
    pub fn to_json(&self) -> ConcreteResult<String> {
        serde_json::to_string_pretty(self).map_err(ConcreteError::Json)
    }

    /// Get messages from a specific sender.
    pub fn messages_from(&self, sender: &str) -> Vec<&ConcreteMessage> {
        self.messages.iter().filter(|m| m.sender == sender).collect()
    }

    /// Get all adversary steps.
    pub fn adversary_steps(&self) -> Vec<&TraceStep> {
        self.steps.iter().filter(|s| s.is_adversary_action()).collect()
    }

    /// Write the trace in pcap-like format to a writer.
    pub fn write_pcap<W: Write>(&self, writer: &mut W) -> ConcreteResult<()> {
        let bytes = self.to_pcap_bytes();
        writer.write_all(&bytes).map_err(ConcreteError::Io)
    }
}

impl fmt::Display for ConcreteTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConcreteTrace({} → {}, {} msgs, {} steps, severity={})",
            self.initial_version,
            self.downgraded_version,
            self.message_count(),
            self.step_count(),
            self.downgrade_severity(),
        )
    }
}

// ── TraceBuilder ─────────────────────────────────────────────────────────

/// Builder for constructing [`ConcreteTrace`] step by step.
pub struct TraceBuilder {
    trace: ConcreteTrace,
    next_message_index: usize,
    current_timestamp_us: u64,
    timestamp_step_us: u64,
}

impl TraceBuilder {
    pub fn new(initial: ProtocolVersion, downgraded: ProtocolVersion) -> Self {
        Self {
            trace: ConcreteTrace::new(initial, downgraded),
            next_message_index: 0,
            current_timestamp_us: 0,
            timestamp_step_us: 1000,
        }
    }

    pub fn with_library(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.trace.metadata.library_name = name.into();
        self.trace.metadata.library_version = version.into();
        self
    }

    pub fn with_trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace.metadata.trace_id = id.into();
        self
    }

    pub fn with_timestamp_step(mut self, step_us: u64) -> Self {
        self.timestamp_step_us = step_us;
        self
    }

    pub fn with_initial_ciphers(mut self, ciphers: Vec<CipherSuite>) -> Self {
        self.trace.initial_ciphers = ciphers;
        self
    }

    fn advance_timestamp(&mut self) -> u64 {
        let ts = self.current_timestamp_us;
        self.current_timestamp_us += self.timestamp_step_us;
        ts
    }

    fn allocate_message_index(&mut self) -> usize {
        let idx = self.next_message_index;
        self.next_message_index += 1;
        idx
    }

    /// Add a client-originated message.
    pub fn client_send(
        &mut self,
        raw_bytes: Vec<u8>,
        version: ProtocolVersion,
        phase: HandshakePhase,
    ) -> usize {
        let idx = self.allocate_message_index();
        let ts = self.advance_timestamp();
        let mut msg = ConcreteMessage::new(idx, "client", "server", raw_bytes, version, phase);
        msg.timestamp_us = ts;
        self.trace.messages.push(msg);
        self.trace.steps.push(TraceStep::ClientSend { message_index: idx });
        idx
    }

    /// Add a server-originated message.
    pub fn server_send(
        &mut self,
        raw_bytes: Vec<u8>,
        version: ProtocolVersion,
        phase: HandshakePhase,
    ) -> usize {
        let idx = self.allocate_message_index();
        let ts = self.advance_timestamp();
        let mut msg = ConcreteMessage::new(idx, "server", "client", raw_bytes, version, phase);
        msg.timestamp_us = ts;
        self.trace.messages.push(msg);
        self.trace.steps.push(TraceStep::ServerSend { message_index: idx });
        idx
    }

    /// Record adversary intercepting a message.
    pub fn adversary_intercept(&mut self, original_index: usize, from: impl Into<String>) {
        let from_str = from.into();
        self.trace.steps.push(TraceStep::AdversaryIntercept {
            original_index,
            from: from_str.clone(),
        });
        self.trace.adversary_actions.push(AdversaryAction::Intercept {
            from: from_str,
            msg_idx: original_index,
        });
    }

    /// Record adversary injecting a new message.
    pub fn adversary_inject(
        &mut self,
        to: impl Into<String>,
        raw_bytes: Vec<u8>,
        version: ProtocolVersion,
        phase: HandshakePhase,
    ) -> usize {
        let idx = self.allocate_message_index();
        let ts = self.advance_timestamp();
        let to_str = to.into();
        let mut msg = ConcreteMessage::new(idx, "adversary", &to_str, raw_bytes.clone(), version, phase);
        msg.timestamp_us = ts;
        self.trace.messages.push(msg);
        self.trace.steps.push(TraceStep::AdversaryInject {
            injected_index: idx,
            to: to_str.clone(),
        });
        self.trace.adversary_actions.push(AdversaryAction::Inject {
            to: to_str,
            payload: raw_bytes,
        });
        idx
    }

    /// Record adversary dropping a message.
    pub fn adversary_drop(&mut self, dropped_index: usize, from: impl Into<String>) {
        let from_str = from.into();
        self.trace.steps.push(TraceStep::AdversaryDrop {
            dropped_index,
            from: from_str.clone(),
        });
        self.trace.adversary_actions.push(AdversaryAction::Drop {
            from: from_str,
            msg_idx: dropped_index,
        });
    }

    /// Record adversary modifying a message.
    pub fn adversary_modify(
        &mut self,
        original_index: usize,
        from: impl Into<String>,
        to: impl Into<String>,
        modified_bytes: Vec<u8>,
        version: ProtocolVersion,
        phase: HandshakePhase,
        modifications: Vec<FieldModification>,
    ) -> usize {
        let modified_index = self.allocate_message_index();
        let ts = self.advance_timestamp();
        let from_str = from.into();
        let to_str = to.into();

        let mut msg = ConcreteMessage::new(
            modified_index,
            "adversary",
            &to_str,
            modified_bytes.clone(),
            version,
            phase,
        );
        msg.timestamp_us = ts;
        self.trace.messages.push(msg);

        let field_name = modifications
            .first()
            .map(|m| m.field_name.clone())
            .unwrap_or_default();

        self.trace.steps.push(TraceStep::AdversaryModify {
            original_index,
            modified_index,
            from: from_str.clone(),
            to: to_str.clone(),
            modifications,
        });
        self.trace.adversary_actions.push(AdversaryAction::Modify {
            from: from_str,
            to: to_str,
            msg_idx: original_index,
            field: field_name,
            new_value: modified_bytes,
        });
        modified_index
    }

    /// Set the negotiated cipher suite.
    pub fn set_negotiated_cipher(&mut self, cipher: CipherSuite) {
        self.trace.negotiated_cipher = Some(cipher);
    }

    /// Update parsed fields for a message.
    pub fn set_parsed_fields(&mut self, msg_index: usize, fields: ParsedFields) -> ConcreteResult<()> {
        let msg = self
            .trace
            .messages
            .iter_mut()
            .find(|m| m.index == msg_index)
            .ok_or_else(|| ConcreteError::Concretization(format!("message {} not found", msg_index)))?;
        msg.parsed_fields = fields;
        Ok(())
    }

    /// Build the final trace.
    pub fn build(mut self) -> ConcreteTrace {
        if self.trace.metadata.trace_id.is_empty() {
            self.trace.metadata.trace_id = format!("trace-{:016x}", {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                self.trace.messages.len().hash(&mut h);
                self.trace.steps.len().hash(&mut h);
                self.current_timestamp_us.hash(&mut h);
                h.finish()
            });
        }
        self.trace
    }

    pub fn current_message_count(&self) -> usize {
        self.trace.messages.len()
    }

    pub fn current_step_count(&self) -> usize {
        self.trace.steps.len()
    }
}

// ── Trace validation against protocol state machine ──────────────────────

/// Validate a trace against the expected handshake phase ordering.
pub fn validate_phase_ordering(trace: &ConcreteTrace) -> Vec<String> {
    let mut errors = Vec::new();
    let mut last_client_phase_order = 0u32;
    let mut last_server_phase_order = 0u32;

    for msg in &trace.messages {
        let order = msg.handshake_phase.order_index();

        // Alert and renegotiation can appear at any point
        if msg.handshake_phase == HandshakePhase::Alert
            || msg.handshake_phase == HandshakePhase::Renegotiation
        {
            continue;
        }

        if msg.sender == "client" {
            if order < last_client_phase_order && order != 0 {
                errors.push(format!(
                    "Client phase regression: {:?} (order {}) after order {}",
                    msg.handshake_phase, order, last_client_phase_order
                ));
            }
            last_client_phase_order = order;
        } else if msg.sender == "server" {
            if order < last_server_phase_order && order != 0 {
                errors.push(format!(
                    "Server phase regression: {:?} (order {}) after order {}",
                    msg.handshake_phase, order, last_server_phase_order
                ));
            }
            last_server_phase_order = order;
        }
        // Adversary-injected messages are not validated for phase ordering
        // since the whole point of a downgrade attack may involve phase manipulation.
    }

    errors
}

/// Validate that all message byte lengths are consistent with their headers.
pub fn validate_message_lengths(trace: &ConcreteTrace) -> Vec<String> {
    let mut errors = Vec::new();
    for msg in &trace.messages {
        if msg.raw_bytes.is_empty() {
            errors.push(format!("Message #{} has empty raw bytes", msg.index));
            continue;
        }
        // TLS record: first byte is content type, bytes 3..5 are length
        if msg.raw_bytes.len() >= 5 {
            let record_len =
                ((msg.raw_bytes[3] as usize) << 8) | (msg.raw_bytes[4] as usize);
            let actual_payload = msg.raw_bytes.len().saturating_sub(5);
            if record_len != actual_payload && !msg.parsed_fields.ssh_packet_type.is_some() {
                // Only flag if not SSH (SSH has different framing)
                if record_len > 0 && actual_payload > 0 {
                    // Allow slight mismatches for partial messages
                    let diff = if record_len > actual_payload {
                        record_len - actual_payload
                    } else {
                        actual_payload - record_len
                    };
                    if diff > 4 {
                        errors.push(format!(
                            "Message #{}: record length {} != payload length {}",
                            msg.index, record_len, actual_payload
                        ));
                    }
                }
            }
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_step_display() {
        let step = TraceStep::ClientSend { message_index: 0 };
        assert!(format!("{}", step).contains("Client"));
        assert!(!step.is_adversary_action());

        let step = TraceStep::AdversaryDrop {
            dropped_index: 2,
            from: "server".into(),
        };
        assert!(step.is_adversary_action());
        assert_eq!(step.step_name(), "AdversaryDrop");
    }

    #[test]
    fn test_concrete_message_hex_dump() {
        let msg = ConcreteMessage::new(
            0,
            "client",
            "server",
            vec![0x16, 0x03, 0x01, 0x00, 0x05, 0x01, 0x02, 0x03, 0x04, 0x05],
            ProtocolVersion::Tls12,
            HandshakePhase::ClientHello,
        );
        let dump = msg.hex_dump();
        assert!(dump.contains("16 03 01"));
        assert_eq!(msg.byte_length(), 10);
    }

    #[test]
    fn test_trace_builder_basic() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30)
            .with_library("openssl", "1.1.1")
            .with_trace_id("test-001");

        let ch_idx = builder.client_send(
            vec![0x16, 0x03, 0x03, 0x00, 0x02, 0x01, 0x00],
            ProtocolVersion::Tls12,
            HandshakePhase::ClientHello,
        );
        assert_eq!(ch_idx, 0);

        builder.adversary_intercept(0, "client");

        let inj_idx = builder.adversary_inject(
            "server",
            vec![0x16, 0x03, 0x00, 0x00, 0x02, 0x01, 0x00],
            ProtocolVersion::Ssl30,
            HandshakePhase::ClientHello,
        );
        assert_eq!(inj_idx, 1);

        let sh_idx = builder.server_send(
            vec![0x16, 0x03, 0x00, 0x00, 0x02, 0x02, 0x00],
            ProtocolVersion::Ssl30,
            HandshakePhase::ServerHello,
        );
        assert_eq!(sh_idx, 2);

        let trace = builder.build();
        assert_eq!(trace.message_count(), 3);
        assert_eq!(trace.step_count(), 4);
        assert!(trace.has_active_adversary());
        assert_eq!(trace.downgrade_severity(), 3);
        assert_eq!(trace.metadata.trace_id, "test-001");
    }

    #[test]
    fn test_trace_pcap_serialization() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Tls10);
        builder.client_send(
            vec![0x16, 0x03, 0x03, 0x00, 0x01, 0x01],
            ProtocolVersion::Tls12,
            HandshakePhase::ClientHello,
        );
        let trace = builder.build();
        let pcap = trace.to_pcap_bytes();

        // Check magic number
        assert_eq!(&pcap[0..4], &0xa1b2c3d4u32.to_le_bytes());
        // Should have global header (24 bytes) + packet header (16 bytes) + 6 byte payload
        assert_eq!(pcap.len(), 24 + 16 + 6);
    }

    #[test]
    fn test_trace_json_serialization() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30);
        builder.client_send(vec![0x16], ProtocolVersion::Tls12, HandshakePhase::ClientHello);
        let trace = builder.build();
        let json = trace.to_json().unwrap();
        assert!(json.contains("Tls12"));
        assert!(json.contains("Ssl30"));
    }

    #[test]
    fn test_trace_pretty_print() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30);
        builder.client_send(
            vec![0x16, 0x03, 0x03, 0x00, 0x01, 0x01],
            ProtocolVersion::Tls12,
            HandshakePhase::ClientHello,
        );
        builder.adversary_intercept(0, "client");
        let trace = builder.build();
        let pp = trace.pretty_print();
        assert!(pp.contains("PROTOCOL DOWNGRADE"));
        assert!(pp.contains("severity"));
        assert!(pp.contains("⚡"));
    }

    #[test]
    fn test_trace_adversary_modify() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30);
        let orig = builder.client_send(
            vec![0x16, 0x03, 0x03],
            ProtocolVersion::Tls12,
            HandshakePhase::ClientHello,
        );
        let mods = vec![FieldModification::new(
            "version",
            vec![0x03, 0x03],
            vec![0x03, 0x00],
            1,
            2,
        )];
        let mod_idx = builder.adversary_modify(
            orig,
            "client",
            "server",
            vec![0x16, 0x03, 0x00],
            ProtocolVersion::Ssl30,
            HandshakePhase::ClientHello,
            mods,
        );
        let trace = builder.build();
        assert_eq!(trace.adversary_action_count(), 1);
        assert!(mod_idx > orig);
    }

    #[test]
    fn test_validate_phase_ordering() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Tls10);
        builder.client_send(vec![0x16], ProtocolVersion::Tls12, HandshakePhase::ClientHello);
        builder.server_send(vec![0x16], ProtocolVersion::Tls12, HandshakePhase::ServerHello);
        builder.server_send(vec![0x16], ProtocolVersion::Tls12, HandshakePhase::Certificate);
        let trace = builder.build();
        let errors = validate_phase_ordering(&trace);
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_field_modification() {
        let fm = FieldModification::new("cipher_suite", vec![0xc0, 0x2f], vec![0x00, 0x0a], 44, 2);
        assert_eq!(fm.field_name, "cipher_suite");
        assert_eq!(fm.byte_offset, 44);
        assert_eq!(fm.byte_length, 2);
    }

    #[test]
    fn test_messages_from_filter() {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Tls10);
        builder.client_send(vec![1], ProtocolVersion::Tls12, HandshakePhase::ClientHello);
        builder.server_send(vec![2], ProtocolVersion::Tls12, HandshakePhase::ServerHello);
        builder.client_send(vec![3], ProtocolVersion::Tls12, HandshakePhase::Finished);
        let trace = builder.build();
        assert_eq!(trace.messages_from("client").len(), 2);
        assert_eq!(trace.messages_from("server").len(), 1);
    }
}
