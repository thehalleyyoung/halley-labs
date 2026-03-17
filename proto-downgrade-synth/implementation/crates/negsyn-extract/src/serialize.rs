//! State machine serialization to DOT, JSON, SARIF, and compact binary formats.
//!
//! Also provides import/export of NegotiationLTS and state machine diffing.

use crate::{
    CipherSuite, ExtractError, ExtractResult, HandshakePhase, LtsState, LtsTransition, MessageLabel,
    NegotiationLTS, NegotiationOutcome, Observable, ProtocolVersion, StateId, TransitionId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::io::Write;

// ---------------------------------------------------------------------------
// LtsSerializer
// ---------------------------------------------------------------------------

/// Top-level serializer supporting multiple output formats.
pub struct LtsSerializer;

impl LtsSerializer {
    /// Export the LTS to DOT format for Graphviz visualization.
    pub fn to_dot(lts: &NegotiationLTS) -> String {
        DotExporter::export(lts)
    }

    /// Export the LTS to JSON.
    pub fn to_json(lts: &NegotiationLTS) -> ExtractResult<String> {
        JsonExporter::export(lts)
    }

    /// Export the LTS to JSON with pretty-printing.
    pub fn to_json_pretty(lts: &NegotiationLTS) -> ExtractResult<String> {
        JsonExporter::export_pretty(lts)
    }

    /// Export the LTS to SARIF format for security reporting.
    pub fn to_sarif(lts: &NegotiationLTS, tool_name: &str, tool_version: &str) -> ExtractResult<String> {
        SarifExporter::export(lts, tool_name, tool_version)
    }

    /// Import an LTS from JSON.
    pub fn from_json(json: &str) -> ExtractResult<NegotiationLTS> {
        JsonExporter::import(json)
    }

    /// Export to compact binary format.
    pub fn to_binary(lts: &NegotiationLTS) -> ExtractResult<Vec<u8>> {
        BinarySerializer::serialize(lts)
    }

    /// Import from compact binary format.
    pub fn from_binary(data: &[u8]) -> ExtractResult<NegotiationLTS> {
        BinarySerializer::deserialize(data)
    }

    /// Compute a diff between two LTS versions.
    pub fn diff(old: &NegotiationLTS, new: &NegotiationLTS) -> LtsDiff {
        LtsDiffer::diff(old, new)
    }
}

// ---------------------------------------------------------------------------
// DOT exporter
// ---------------------------------------------------------------------------

struct DotExporter;

impl DotExporter {
    fn export(lts: &NegotiationLTS) -> String {
        let mut out = String::new();
        out.push_str("digraph NegotiationLTS {\n");
        out.push_str("  rankdir=TB;\n");
        out.push_str("  node [shape=circle, fontsize=10];\n");
        out.push_str("  edge [fontsize=8];\n\n");

        // Invisible start node.
        out.push_str("  __start__ [shape=point, width=0.1];\n");

        // States.
        for (&sid, state) in &lts.states {
            let shape = if state.is_terminal {
                "doublecircle"
            } else {
                "circle"
            };
            let color = match state.negotiation.phase {
                HandshakePhase::Init | HandshakePhase::Initial => "lightblue",
                HandshakePhase::ClientHelloSent | HandshakePhase::ClientHello => "lightyellow",
                HandshakePhase::ServerHelloReceived | HandshakePhase::ServerHello => "lightgreen",
                HandshakePhase::Certificate | HandshakePhase::KeyExchange => "lightsalmon",
                HandshakePhase::Negotiated => "lightsalmon",
                HandshakePhase::ChangeCipherSpec | HandshakePhase::Finished => "lightcoral",
                HandshakePhase::Done | HandshakePhase::ApplicationData => "palegreen",
                HandshakePhase::Abort | HandshakePhase::Alert => "lightpink",
                HandshakePhase::Renegotiation => "plum",
            };

            let label = Self::state_label(state);
            out.push_str(&format!(
                "  {} [label=\"{}\", shape={}, style=filled, fillcolor=\"{}\"];\n",
                sid, label, shape, color,
            ));
        }

        // Initial state arrows.
        for &init_id in &lts.initial_states {
            out.push_str(&format!("  __start__ -> {};\n", init_id));
        }

        out.push_str("\n");

        // Transitions.
        for t in &lts.transitions {
            let label = Self::transition_label(t);
            let style = if t.label.is_adversary_action() {
                ", style=dashed, color=red"
            } else if t.label.is_internal() {
                ", style=dotted, color=gray"
            } else {
                ""
            };
            out.push_str(&format!(
                "  {} -> {} [label=\"{}\"{}];\n",
                t.source, t.target, label, style,
            ));
        }

        out.push_str("}\n");
        out
    }

    fn state_label(state: &LtsState) -> String {
        let mut parts = vec![format!("{}", state.graph_id)];
        parts.push(format!("{}", state.negotiation.phase));

        if let Some(ref cipher) = state.negotiation.selected_cipher {
            parts.push(format!("0x{:04x}", cipher.iana_id));
        }
        if state.observation.outcome != NegotiationOutcome::InProgress {
            parts.push(format!("{}", state.observation));
        }

        parts.join("\\n")
    }

    fn transition_label(t: &LtsTransition) -> String {
        let base = t.label.label_name().to_string();
        if t.guard.is_some() {
            format!("{}[guard]", base)
        } else {
            base
        }
    }
}

// ---------------------------------------------------------------------------
// JSON exporter/importer
// ---------------------------------------------------------------------------

/// JSON representation of the LTS.
#[derive(Serialize, Deserialize)]
struct JsonLts {
    states: Vec<JsonState>,
    transitions: Vec<JsonTransition>,
    initial_states: Vec<u64>,
    metadata: JsonMetadata,
}

#[derive(Serialize, Deserialize)]
struct JsonState {
    id: u64,
    phase: String,
    version: String,
    selected_cipher: Option<u16>,
    offered_ciphers: Vec<u16>,
    observation: String,
    is_initial: bool,
    is_terminal: bool,
    source_symbolic_ids: Vec<u64>,
}

#[derive(Serialize, Deserialize)]
struct JsonTransition {
    id: u64,
    source: u64,
    target: u64,
    label: String,
    label_detail: String,
    has_guard: bool,
}

#[derive(Serialize, Deserialize)]
struct JsonMetadata {
    state_count: usize,
    transition_count: usize,
    initial_count: usize,
    terminal_count: usize,
    format_version: String,
}

struct JsonExporter;

impl JsonExporter {
    fn export(lts: &NegotiationLTS) -> ExtractResult<String> {
        let json_lts = Self::to_json_struct(lts);
        serde_json::to_string(&json_lts)
            .map_err(|e| ExtractError::Serialization(e.to_string()))
    }

    fn export_pretty(lts: &NegotiationLTS) -> ExtractResult<String> {
        let json_lts = Self::to_json_struct(lts);
        serde_json::to_string_pretty(&json_lts)
            .map_err(|e| ExtractError::Serialization(e.to_string()))
    }

    fn to_json_struct(lts: &NegotiationLTS) -> JsonLts {
        let states: Vec<JsonState> = lts
            .states
            .iter()
            .map(|(&sid, state)| JsonState {
                id: sid.0 as u64,
                phase: format!("{}", state.negotiation.phase),
                version: state.negotiation.version.as_ref().map(|v| format!("{}", v)).unwrap_or_else(|| "Unknown".to_string()),
                selected_cipher: state.negotiation.selected_cipher.as_ref().map(|c| c.iana_id),
                offered_ciphers: state.negotiation.offered_ciphers.iter().map(|c| c.iana_id).collect(),
                observation: format!("{}", state.observation),
                is_initial: state.is_initial,
                is_terminal: state.is_terminal,
                source_symbolic_ids: state.source_symbolic_ids.clone(),
            })
            .collect();

        let transitions: Vec<JsonTransition> = lts
            .transitions
            .iter()
            .map(|t| JsonTransition {
                id: t.id.0 as u64,
                source: t.source.0 as u64,
                target: t.target.0 as u64,
                label: t.label.label_name().to_string(),
                label_detail: format!("{}", t.label),
                has_guard: t.guard.is_some(),
            })
            .collect();

        let terminal_count = lts
            .states
            .values()
            .filter(|s| s.is_terminal)
            .count();

        JsonLts {
            states,
            transitions,
            initial_states: lts.initial_states.iter().map(|s| s.0 as u64).collect(),
            metadata: JsonMetadata {
                state_count: lts.state_count(),
                transition_count: lts.transition_count(),
                initial_count: lts.initial_states.len(),
                terminal_count,
                format_version: "1.0".to_string(),
            },
        }
    }

    fn import(json: &str) -> ExtractResult<NegotiationLTS> {
        let json_lts: JsonLts = serde_json::from_str(json)
            .map_err(|e| ExtractError::Serialization(e.to_string()))?;

        let mut lts = NegotiationLTS::new();

        for js in &json_lts.states {
            let phase = Self::parse_phase(&js.phase);
            let version = Self::parse_version(&js.version);
            let mut neg = negsyn_types::NegotiationState::new();
            neg.phase = phase;
            neg.version = Some(version);
            neg.selected_cipher = js.selected_cipher.map(|id| CipherSuite::new(
                id,
                format!("IMPORTED_0x{:04x}", id),
                negsyn_types::protocol::KeyExchange::NULL,
                negsyn_types::protocol::AuthAlgorithm::NULL,
                negsyn_types::protocol::EncryptionAlgorithm::NULL,
                negsyn_types::protocol::MacAlgorithm::NULL,
                negsyn_types::SecurityLevel::Standard,
            ));
            neg.offered_ciphers = js.offered_ciphers.iter().map(|&id| CipherSuite::new(
                id,
                format!("IMPORTED_0x{:04x}", id),
                negsyn_types::protocol::KeyExchange::NULL,
                negsyn_types::protocol::AuthAlgorithm::NULL,
                negsyn_types::protocol::EncryptionAlgorithm::NULL,
                negsyn_types::protocol::MacAlgorithm::NULL,
                negsyn_types::SecurityLevel::Standard,
            )).collect();

            let sid = StateId(js.id as u32);
            lts.add_state_with_id(sid, neg);

            if let Some(state) = lts.get_state_mut(sid) {
                state.is_initial = js.is_initial;
                state.is_terminal = js.is_terminal;
                state.source_symbolic_ids = js.source_symbolic_ids.clone();
            }
        }

        for &init_id in &json_lts.initial_states {
            lts.mark_initial(StateId(init_id as u32));
        }

        for jt in &json_lts.transitions {
            lts.add_transition(
                StateId(jt.source as u32),
                StateId(jt.target as u32),
                Self::parse_label(&jt.label),
            );
        }

        Ok(lts)
    }

    fn parse_phase(s: &str) -> HandshakePhase {
        match s {
            "Initial" => HandshakePhase::Initial,
            "ClientHello" => HandshakePhase::ClientHello,
            "ServerHello" => HandshakePhase::ServerHello,
            "Certificate" => HandshakePhase::Certificate,
            "KeyExchange" => HandshakePhase::KeyExchange,
            "ChangeCipherSpec" => HandshakePhase::ChangeCipherSpec,
            "Finished" => HandshakePhase::Finished,
            "ApplicationData" => HandshakePhase::ApplicationData,
            "Alert" => HandshakePhase::Alert,
            "Renegotiation" => HandshakePhase::Renegotiation,
            _ => HandshakePhase::Initial,
        }
    }

    fn parse_version(s: &str) -> ProtocolVersion {
        match s {
            "SSL 3.0" => ProtocolVersion::Ssl30,
            "TLS 1.0" => ProtocolVersion::Tls10,
            "TLS 1.1" => ProtocolVersion::Tls11,
            "TLS 1.2" => ProtocolVersion::Tls12,
            "TLS 1.3" => ProtocolVersion::Tls13,
            "SSH 2.0" => ProtocolVersion::Ssh2,
            "DTLS 1.0" => ProtocolVersion::Dtls10,
            "DTLS 1.2" => ProtocolVersion::Dtls12,
            _ => ProtocolVersion::Unknown,
        }
    }

    fn parse_label(s: &str) -> MessageLabel {
        match s {
            "CH" => MessageLabel::ClientHello {
                offered_ciphers: std::collections::BTreeSet::new(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            "SH" => MessageLabel::ServerHello {
                selected_cipher: 0,
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            "SC" => MessageLabel::ServerCertificate,
            "SKE" => MessageLabel::ServerKeyExchange,
            "SHD" => MessageLabel::ServerHelloDone,
            "CKE" => MessageLabel::ClientKeyExchange,
            "CCCS" => MessageLabel::ClientChangeCipherSpec,
            "SCCS" => MessageLabel::ServerChangeCipherSpec,
            "CF" => MessageLabel::ClientFinished {
                verify_data_hash: 0,
            },
            "SF" => MessageLabel::ServerFinished {
                verify_data_hash: 0,
            },
            "τ" | "Tau" => MessageLabel::Tau,
            _ => MessageLabel::Tau,
        }
    }
}

// ---------------------------------------------------------------------------
// SARIF exporter
// ---------------------------------------------------------------------------

struct SarifExporter;

impl SarifExporter {
    fn export(
        lts: &NegotiationLTS,
        tool_name: &str,
        tool_version: &str,
    ) -> ExtractResult<String> {
        let runs = vec![SarifRun {
            tool: SarifTool {
                driver: SarifDriver {
                    name: tool_name.to_string(),
                    version: tool_version.to_string(),
                    semantic_version: tool_version.to_string(),
                    information_uri: "https://github.com/negsyn/negsyn".to_string(),
                    rules: Self::generate_rules(lts),
                },
            },
            results: Self::generate_results(lts),
        }];

        let sarif = SarifLog {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json".to_string(),
            version: "2.1.0".to_string(),
            runs,
        };

        serde_json::to_string_pretty(&sarif)
            .map_err(|e| ExtractError::Serialization(e.to_string()))
    }

    fn generate_rules(lts: &NegotiationLTS) -> Vec<SarifRule> {
        let mut rules = Vec::new();

        rules.push(SarifRule {
            id: "NEGSYN-001".to_string(),
            name: "CipherDowngrade".to_string(),
            short_description: SarifMessage {
                text: "Potential cipher suite downgrade attack path detected".to_string(),
            },
            default_configuration: SarifRuleConfig {
                level: "error".to_string(),
            },
        });

        rules.push(SarifRule {
            id: "NEGSYN-002".to_string(),
            name: "VersionDowngrade".to_string(),
            short_description: SarifMessage {
                text: "Potential protocol version downgrade attack path detected".to_string(),
            },
            default_configuration: SarifRuleConfig {
                level: "error".to_string(),
            },
        });

        rules.push(SarifRule {
            id: "NEGSYN-003".to_string(),
            name: "ExtensionStripping".to_string(),
            short_description: SarifMessage {
                text: "Potential extension stripping attack path detected".to_string(),
            },
            default_configuration: SarifRuleConfig {
                level: "warning".to_string(),
            },
        });

        rules.push(SarifRule {
            id: "NEGSYN-004".to_string(),
            name: "StateSpace".to_string(),
            short_description: SarifMessage {
                text: "State machine extraction summary".to_string(),
            },
            default_configuration: SarifRuleConfig {
                level: "note".to_string(),
            },
        });

        rules
    }

    fn generate_results(lts: &NegotiationLTS) -> Vec<SarifResult> {
        let mut results = Vec::new();

        // Analyze adversary transitions for potential attacks.
        for t in &lts.transitions {
            if t.label.is_adversary_action() {
                let rule_id = match &t.label {
                    MessageLabel::AdversaryModify { .. } => "NEGSYN-001",
                    MessageLabel::AdversaryDrop { .. } => "NEGSYN-003",
                    MessageLabel::AdversaryInject { .. } => "NEGSYN-001",
                    MessageLabel::AdversaryIntercept { .. } => "NEGSYN-002",
                    _ => "NEGSYN-001",
                };

                results.push(SarifResult {
                    rule_id: rule_id.to_string(),
                    message: SarifMessage {
                        text: format!(
                            "Adversary action {} from state {} to state {}",
                            t.label.label_name(),
                            t.source,
                            t.target,
                        ),
                    },
                    level: "error".to_string(),
                    locations: vec![],
                });
            }
        }

        // Summary result.
        results.push(SarifResult {
            rule_id: "NEGSYN-004".to_string(),
            message: SarifMessage {
                text: format!(
                    "Extracted state machine: {} states, {} transitions, {} initial, {} terminal",
                    lts.state_count(),
                    lts.transition_count(),
                    lts.initial_states.len(),
                    lts.terminal_states().len(),
                ),
            },
            level: "note".to_string(),
            locations: vec![],
        });

        results
    }
}

#[derive(Serialize, Deserialize)]
struct SarifLog {
    #[serde(rename = "$schema")]
    schema: String,
    version: String,
    runs: Vec<SarifRun>,
}

#[derive(Serialize, Deserialize)]
struct SarifRun {
    tool: SarifTool,
    results: Vec<SarifResult>,
}

#[derive(Serialize, Deserialize)]
struct SarifTool {
    driver: SarifDriver,
}

#[derive(Serialize, Deserialize)]
struct SarifDriver {
    name: String,
    version: String,
    #[serde(rename = "semanticVersion")]
    semantic_version: String,
    #[serde(rename = "informationUri")]
    information_uri: String,
    rules: Vec<SarifRule>,
}

#[derive(Serialize, Deserialize)]
struct SarifRule {
    id: String,
    name: String,
    #[serde(rename = "shortDescription")]
    short_description: SarifMessage,
    #[serde(rename = "defaultConfiguration")]
    default_configuration: SarifRuleConfig,
}

#[derive(Serialize, Deserialize)]
struct SarifRuleConfig {
    level: String,
}

#[derive(Serialize, Deserialize)]
struct SarifResult {
    #[serde(rename = "ruleId")]
    rule_id: String,
    message: SarifMessage,
    level: String,
    locations: Vec<SarifLocation>,
}

#[derive(Serialize, Deserialize)]
struct SarifMessage {
    text: String,
}

#[derive(Serialize, Deserialize)]
struct SarifLocation {
    #[serde(rename = "physicalLocation")]
    physical_location: SarifPhysicalLocation,
}

#[derive(Serialize, Deserialize)]
struct SarifPhysicalLocation {
    #[serde(rename = "artifactLocation")]
    artifact_location: SarifArtifactLocation,
}

#[derive(Serialize, Deserialize)]
struct SarifArtifactLocation {
    uri: String,
}

// ---------------------------------------------------------------------------
// Binary serializer
// ---------------------------------------------------------------------------

struct BinarySerializer;

impl BinarySerializer {
    fn serialize(lts: &NegotiationLTS) -> ExtractResult<Vec<u8>> {
        // Use JSON internally for now, compressed with simple encoding.
        let json = serde_json::to_vec(lts)
            .map_err(|e| ExtractError::Serialization(e.to_string()))?;

        // Simple envelope: magic + version + length + data.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"NLTS"); // magic
        buf.push(1); // version
        let len = json.len() as u32;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&json);
        Ok(buf)
    }

    fn deserialize(data: &[u8]) -> ExtractResult<NegotiationLTS> {
        if data.len() < 9 {
            return Err(ExtractError::Serialization("data too short".into()));
        }
        if &data[0..4] != b"NLTS" {
            return Err(ExtractError::Serialization("invalid magic".into()));
        }
        let version = data[4];
        if version != 1 {
            return Err(ExtractError::Serialization(format!(
                "unsupported version: {}",
                version,
            )));
        }
        let len = u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;
        if data.len() < 9 + len {
            return Err(ExtractError::Serialization("truncated data".into()));
        }
        let json_data = &data[9..9 + len];
        serde_json::from_slice(json_data)
            .map_err(|e| ExtractError::Serialization(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// LTS Diffing
// ---------------------------------------------------------------------------

/// Difference between two LTS versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtsDiff {
    pub added_states: Vec<StateId>,
    pub removed_states: Vec<StateId>,
    pub modified_states: Vec<StateId>,
    pub added_transitions: Vec<(StateId, String, StateId)>,
    pub removed_transitions: Vec<(StateId, String, StateId)>,
    pub initial_changed: bool,
}

impl LtsDiff {
    pub fn is_empty(&self) -> bool {
        self.added_states.is_empty()
            && self.removed_states.is_empty()
            && self.modified_states.is_empty()
            && self.added_transitions.is_empty()
            && self.removed_transitions.is_empty()
            && !self.initial_changed
    }

    pub fn summary(&self) -> String {
        format!(
            "+{}/-{} states, ~{} modified, +{}/-{} transitions",
            self.added_states.len(),
            self.removed_states.len(),
            self.modified_states.len(),
            self.added_transitions.len(),
            self.removed_transitions.len(),
        )
    }
}

impl fmt::Display for LtsDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LTS Diff:")?;
        if !self.added_states.is_empty() {
            writeln!(f, "  Added states: {:?}", self.added_states)?;
        }
        if !self.removed_states.is_empty() {
            writeln!(f, "  Removed states: {:?}", self.removed_states)?;
        }
        if !self.modified_states.is_empty() {
            writeln!(f, "  Modified states: {:?}", self.modified_states)?;
        }
        if !self.added_transitions.is_empty() {
            writeln!(f, "  Added transitions: {}", self.added_transitions.len())?;
        }
        if !self.removed_transitions.is_empty() {
            writeln!(
                f,
                "  Removed transitions: {}",
                self.removed_transitions.len(),
            )?;
        }
        if self.initial_changed {
            writeln!(f, "  Initial states changed")?;
        }
        Ok(())
    }
}

struct LtsDiffer;

impl LtsDiffer {
    fn diff(old: &NegotiationLTS, new: &NegotiationLTS) -> LtsDiff {
        let old_state_ids: HashSet<StateId> = old.states.keys().copied().collect();
        let new_state_ids: HashSet<StateId> = new.states.keys().copied().collect();

        let added_states: Vec<StateId> = new_state_ids
            .difference(&old_state_ids)
            .copied()
            .collect();
        let removed_states: Vec<StateId> = old_state_ids
            .difference(&new_state_ids)
            .copied()
            .collect();

        // Modified states: present in both, but observation changed.
        let modified_states: Vec<StateId> = old_state_ids
            .intersection(&new_state_ids)
            .filter(|&&sid| {
                let old_obs = old.obs(sid);
                let new_obs = new.obs(sid);
                old_obs != new_obs
            })
            .copied()
            .collect();

        // Transition diff.
        let old_trans: HashSet<(StateId, String, StateId)> = old
            .transitions
            .iter()
            .map(|t| (t.source, t.label.label_name().to_string(), t.target))
            .collect();
        let new_trans: HashSet<(StateId, String, StateId)> = new
            .transitions
            .iter()
            .map(|t| (t.source, t.label.label_name().to_string(), t.target))
            .collect();

        let added_transitions: Vec<(StateId, String, StateId)> = new_trans
            .difference(&old_trans)
            .cloned()
            .collect();
        let removed_transitions: Vec<(StateId, String, StateId)> = old_trans
            .difference(&new_trans)
            .cloned()
            .collect();

        let initial_changed = old.initial_states != new.initial_states;

        LtsDiff {
            added_states,
            removed_states,
            modified_states,
            added_transitions,
            removed_transitions,
            initial_changed,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use negsyn_types::NegotiationState;
    use std::collections::BTreeSet;

    fn make_neg(phase: HandshakePhase, cipher: Option<u16>) -> NegotiationState {
        let mut ns = NegotiationState::new();
        ns.phase = phase;
        ns.version = Some(ProtocolVersion::Tls12);
        ns.selected_cipher = cipher.map(|id| CipherSuite::new(
            id,
            format!("TEST_0x{:04x}", id),
            negsyn_types::protocol::KeyExchange::NULL,
            negsyn_types::protocol::AuthAlgorithm::NULL,
            negsyn_types::protocol::EncryptionAlgorithm::NULL,
            negsyn_types::protocol::MacAlgorithm::NULL,
            negsyn_types::SecurityLevel::Standard,
        ));
        ns
    }

    fn make_sample_lts() -> NegotiationLTS {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        let s2 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let s3 = lts.add_state(make_neg(HandshakePhase::Alert, None));
        lts.mark_initial(s0);
        lts.add_transition(
            s0,
            s1,
            MessageLabel::ClientHello {
                offered_ciphers: [0x002f, 0x0035].into(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
        );
        lts.add_transition(
            s1,
            s2,
            MessageLabel::ServerHello {
                selected_cipher: 0x002f,
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
        );
        lts.add_transition(
            s1,
            s3,
            MessageLabel::Alert {
                level: 2,
                description: 40,
            },
        );
        lts
    }

    #[test]
    fn test_dot_export() {
        let lts = make_sample_lts();
        let dot = LtsSerializer::to_dot(&lts);
        assert!(dot.contains("digraph NegotiationLTS"));
        assert!(dot.contains("__start__"));
        assert!(dot.contains("->"));
        assert!(dot.contains("CH"));
    }

    #[test]
    fn test_json_export_import() {
        let lts = make_sample_lts();
        let json = LtsSerializer::to_json(&lts).unwrap();
        assert!(json.contains("states"));
        assert!(json.contains("transitions"));

        let imported = LtsSerializer::from_json(&json).unwrap();
        assert_eq!(imported.state_count(), lts.state_count());
        assert_eq!(imported.transition_count(), lts.transition_count());
    }

    #[test]
    fn test_json_pretty() {
        let lts = make_sample_lts();
        let json = LtsSerializer::to_json_pretty(&lts).unwrap();
        assert!(json.contains('\n'));
        assert!(json.contains("  "));
    }

    #[test]
    fn test_sarif_export() {
        let lts = make_sample_lts();
        let sarif = LtsSerializer::to_sarif(&lts, "NegSynth", "0.1.0").unwrap();
        assert!(sarif.contains("$schema"));
        assert!(sarif.contains("2.1.0"));
        assert!(sarif.contains("NegSynth"));
        assert!(sarif.contains("NEGSYN-004"));
    }

    #[test]
    fn test_binary_roundtrip() {
        let lts = make_sample_lts();
        let binary = LtsSerializer::to_binary(&lts).unwrap();
        assert!(binary.starts_with(b"NLTS"));

        let restored = LtsSerializer::from_binary(&binary).unwrap();
        assert_eq!(restored.state_count(), lts.state_count());
        assert_eq!(restored.transition_count(), lts.transition_count());
    }

    #[test]
    fn test_binary_invalid_magic() {
        let result = LtsSerializer::from_binary(b"XXXX\x01\x00\x00\x00\x00");
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_too_short() {
        let result = LtsSerializer::from_binary(b"NLT");
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_identical() {
        let lts = make_sample_lts();
        let diff = LtsSerializer::diff(&lts, &lts);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_added_state() {
        let lts1 = make_sample_lts();
        let mut lts2 = make_sample_lts();
        lts2.add_state(make_neg(HandshakePhase::Renegotiation, None));

        let diff = LtsSerializer::diff(&lts1, &lts2);
        assert_eq!(diff.added_states.len(), 1);
        assert!(diff.removed_states.is_empty());
    }

    #[test]
    fn test_diff_removed_state() {
        let lts1 = make_sample_lts();
        let mut lts2 = make_sample_lts();
        let to_remove = *lts2.states.keys().last().unwrap();
        lts2.remove_state(to_remove);

        let diff = LtsSerializer::diff(&lts1, &lts2);
        assert!(!diff.removed_states.is_empty());
    }

    #[test]
    fn test_diff_display() {
        let lts1 = make_sample_lts();
        let mut lts2 = make_sample_lts();
        lts2.add_state(make_neg(HandshakePhase::Renegotiation, None));

        let diff = LtsSerializer::diff(&lts1, &lts2);
        let s = format!("{}", diff);
        assert!(s.contains("Added states"));
    }

    #[test]
    fn test_diff_summary() {
        let lts1 = make_sample_lts();
        let diff = LtsSerializer::diff(&lts1, &lts1);
        let summary = diff.summary();
        assert!(summary.contains("+0/-0"));
    }

    #[test]
    fn test_dot_adversary_styling() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        lts.mark_initial(s0);
        lts.add_transition(
            s0,
            s1,
            MessageLabel::AdversaryDrop { message_index: 0 },
        );

        let dot = LtsSerializer::to_dot(&lts);
        assert!(dot.contains("dashed"));
        assert!(dot.contains("red"));
    }

    #[test]
    fn test_json_import_handles_unknown() {
        // Test that unknown label names parse to Tau.
        let json = r#"{
            "states": [{"id": 0, "phase": "Initial", "version": "TLS 1.2",
                "selected_cipher": null, "offered_ciphers": [],
                "observation": "IN_PROGRESS", "is_initial": true,
                "is_terminal": false, "source_symbolic_ids": []}],
            "transitions": [{"id": 0, "source": 0, "target": 0,
                "label": "UNKNOWN_LABEL", "label_detail": "unknown",
                "has_guard": false}],
            "initial_states": [0],
            "metadata": {"state_count": 1, "transition_count": 1,
                "initial_count": 1, "terminal_count": 0,
                "format_version": "1.0"}
        }"#;

        let lts = LtsSerializer::from_json(json).unwrap();
        assert_eq!(lts.state_count(), 1);
        assert_eq!(lts.transition_count(), 1);
    }
}
