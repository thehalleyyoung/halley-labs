//! CLI command implementations for NegSynth.
//!
//! Each subcommand lives in its own module and exposes a clap-derived struct
//! with an `execute` method.

pub mod analyze;
pub mod benchmark;
pub mod diff;
pub mod inspect;
pub mod replay;
pub mod verify;

use negsyn_types::{CipherSuite, HandshakePhase, ProtocolVersion};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Protocol type (local — not yet in any upstream crate)
// ---------------------------------------------------------------------------

/// Target protocol family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    Tls,
    Ssh,
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tls => write!(f, "TLS"),
            Self::Ssh => write!(f, "SSH"),
        }
    }
}

impl Protocol {
    pub fn default_port(&self) -> u16 {
        match self {
            Self::Tls => 443,
            Self::Ssh => 22,
        }
    }
}

// ---------------------------------------------------------------------------
// State machine types (compatible with negsyn-eval Lts)
// ---------------------------------------------------------------------------

/// A state in the labeled transition system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub id: u32,
    pub label: String,
    pub phase: HandshakePhase,
    pub version: Option<ProtocolVersion>,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub is_error: bool,
    pub properties: BTreeMap<String, String>,
}

impl State {
    pub fn new(id: u32, label: impl Into<String>, phase: HandshakePhase) -> Self {
        Self {
            id,
            label: label.into(),
            phase,
            version: None,
            is_initial: id == 0,
            is_accepting: false,
            is_error: false,
            properties: BTreeMap::new(),
        }
    }
}

/// A transition in the labeled transition system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub label: String,
    pub guard: Option<String>,
    pub action: Option<String>,
    pub cipher_suite_id: Option<u16>,
    pub is_downgrade: bool,
}

impl Transition {
    pub fn new(id: u32, source: u32, target: u32, label: impl Into<String>) -> Self {
        Self {
            id,
            source,
            target,
            label: label.into(),
            guard: None,
            action: None,
            cipher_suite_id: None,
            is_downgrade: false,
        }
    }
}

/// Labeled transition system (state machine).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    pub states: Vec<State>,
    pub transitions: Vec<Transition>,
    pub initial_state: u32,
    pub library_name: String,
    pub protocol: Protocol,
}

impl StateMachine {
    pub fn new(library_name: impl Into<String>, protocol: Protocol) -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            initial_state: 0,
            library_name: library_name.into(),
            protocol,
        }
    }

    pub fn add_state(&mut self, state: State) {
        self.states.push(state);
    }

    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn get_state(&self, id: u32) -> Option<&State> {
        self.states.iter().find(|s| s.id == id)
    }

    pub fn transitions_from(&self, state_id: u32) -> Vec<&Transition> {
        self.transitions.iter().filter(|t| t.source == state_id).collect()
    }

    pub fn accepting_states(&self) -> Vec<&State> {
        self.states.iter().filter(|s| s.is_accepting).collect()
    }

    pub fn error_states(&self) -> Vec<&State> {
        self.states.iter().filter(|s| s.is_error).collect()
    }

    pub fn downgrade_transitions(&self) -> Vec<&Transition> {
        self.transitions.iter().filter(|t| t.is_downgrade).collect()
    }

    /// Compute reachable states via BFS from the initial state.
    pub fn reachable_states(&self) -> Vec<u32> {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.initial_state);
        visited.insert(self.initial_state);
        while let Some(sid) = queue.pop_front() {
            for t in self.transitions_from(sid) {
                if visited.insert(t.target) {
                    queue.push_back(t.target);
                }
            }
        }
        let mut v: Vec<u32> = visited.into_iter().collect();
        v.sort();
        v
    }

    /// Bisimulation equivalence classes (simple partition by phase + version).
    pub fn bisimulation_classes(&self) -> BTreeMap<String, Vec<u32>> {
        let mut classes: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for s in &self.states {
            let key = format!("{:?}:{:?}", s.phase, s.version);
            classes.entry(key).or_default().push(s.id);
        }
        classes
    }
}

// ---------------------------------------------------------------------------
// Attack trace types (compatible with negsyn-eval)
// ---------------------------------------------------------------------------

/// A single step in an attack trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackStep {
    pub step_number: u32,
    pub action: String,
    pub from_state: u32,
    pub to_state: u32,
    pub message: Option<String>,
    pub cipher_suite_id: Option<u16>,
    pub bytes: Option<Vec<u8>>,
}

/// Complete attack trace showing a downgrade path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTrace {
    pub steps: Vec<AttackStep>,
    pub downgraded_from: u16,
    pub downgraded_to: u16,
    pub adversary_budget: u32,
    pub vulnerability_type: String,
    pub library_name: String,
    pub protocol: Protocol,
}

impl AttackTrace {
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.steps.is_empty() {
            return Err("trace has no steps".into());
        }
        if self.downgraded_from == self.downgraded_to {
            return Err("downgraded_from == downgraded_to — not a downgrade".into());
        }
        for (i, step) in self.steps.iter().enumerate() {
            if step.step_number != i as u32 {
                return Err(format!(
                    "step {} has step_number {}, expected {}",
                    i, step.step_number, i
                ));
            }
        }
        // Check state chain continuity.
        for w in self.steps.windows(2) {
            if w[0].to_state != w[1].from_state {
                return Err(format!(
                    "step {} ends at state {} but step {} starts at {}",
                    w[0].step_number, w[0].to_state, w[1].step_number, w[1].from_state
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Analysis certificate (compatible with negsyn-eval)
// ---------------------------------------------------------------------------

/// Completeness certificate for an analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCertificate {
    pub id: String,
    pub library_name: String,
    pub protocol: Protocol,
    pub timestamp: String,
    pub states_explored: usize,
    pub paths_explored: usize,
    pub coverage_pct: f64,
    pub depth_bound: u32,
    pub action_bound: u32,
    pub vulnerabilities_found: Vec<String>,
    pub attack_traces: Vec<AttackTrace>,
    pub hash: String,
    pub version: String,
}

impl AnalysisCertificate {
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && !self.library_name.is_empty()
            && !self.timestamp.is_empty()
            && self.coverage_pct >= 0.0
            && self.coverage_pct <= 100.0
    }

    pub fn meets_coverage(&self, threshold: f64) -> bool {
        self.coverage_pct >= threshold
    }

    pub fn has_vulnerabilities(&self) -> bool {
        !self.vulnerabilities_found.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Analysis result (aggregate output)
// ---------------------------------------------------------------------------

/// Complete result of an analysis pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub library_name: String,
    pub protocol: Protocol,
    pub state_machine: StateMachine,
    pub certificate: AnalysisCertificate,
    pub attack_traces: Vec<AttackTrace>,
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// Deviation types for differential analysis
// ---------------------------------------------------------------------------

/// A behavioral deviation between two libraries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralDeviation {
    pub library_a: String,
    pub library_b: String,
    pub deviation_type: String,
    pub description: String,
    pub severity: DeviationSeverity,
    pub state_a: Option<u32>,
    pub state_b: Option<u32>,
    pub cipher_suite: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum DeviationSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for DeviationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark types
// ---------------------------------------------------------------------------

/// Result of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u32,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub stddev_ms: f64,
    pub throughput: Option<f64>,
    pub memory_bytes: Option<u64>,
}

// ---------------------------------------------------------------------------
// File detection helpers
// ---------------------------------------------------------------------------

/// Detect protocol type from file contents or extension.
pub fn detect_protocol(path: &Path) -> Option<Protocol> {
    let name = path.file_name()?.to_string_lossy().to_lowercase();
    if name.contains("ssh") || name.contains("kex") {
        return Some(Protocol::Ssh);
    }
    if name.contains("tls") || name.contains("ssl") || name.contains("handshake") {
        return Some(Protocol::Tls);
    }
    // Try reading first few bytes.
    if let Ok(bytes) = std::fs::read(path) {
        let head = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]);
        if head.contains("SSH_MSG") || head.contains("KEX_INIT") || head.contains("kex_algorithm") {
            return Some(Protocol::Ssh);
        }
        if head.contains("ClientHello")
            || head.contains("ServerHello")
            || head.contains("cipher_suite")
            || head.contains("TLS")
        {
            return Some(Protocol::Tls);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_display() {
        assert_eq!(Protocol::Tls.to_string(), "TLS");
        assert_eq!(Protocol::Ssh.to_string(), "SSH");
    }

    #[test]
    fn protocol_default_port() {
        assert_eq!(Protocol::Tls.default_port(), 443);
        assert_eq!(Protocol::Ssh.default_port(), 22);
    }

    #[test]
    fn state_machine_basics() {
        let mut sm = StateMachine::new("openssl", Protocol::Tls);
        sm.add_state(State::new(0, "init", HandshakePhase::Initial));
        sm.add_state(State::new(1, "hello", HandshakePhase::ClientHello));
        sm.add_transition(Transition::new(0, 0, 1, "send_hello"));
        assert_eq!(sm.state_count(), 2);
        assert_eq!(sm.transition_count(), 1);
        assert_eq!(sm.transitions_from(0).len(), 1);
    }

    #[test]
    fn reachable_states() {
        let mut sm = StateMachine::new("test", Protocol::Tls);
        sm.add_state(State::new(0, "s0", HandshakePhase::Initial));
        sm.add_state(State::new(1, "s1", HandshakePhase::ClientHello));
        sm.add_state(State::new(2, "s2", HandshakePhase::ServerHello));
        sm.add_transition(Transition::new(0, 0, 1, "t01"));
        // s2 is unreachable
        let r = sm.reachable_states();
        assert!(r.contains(&0));
        assert!(r.contains(&1));
        assert!(!r.contains(&2));
    }

    #[test]
    fn attack_trace_validation() {
        let trace = AttackTrace {
            steps: vec![
                AttackStep {
                    step_number: 0,
                    action: "forward".into(),
                    from_state: 0,
                    to_state: 1,
                    message: None,
                    cipher_suite_id: None,
                    bytes: None,
                },
                AttackStep {
                    step_number: 1,
                    action: "modify".into(),
                    from_state: 1,
                    to_state: 2,
                    message: None,
                    cipher_suite_id: Some(0x002F),
                    bytes: None,
                },
            ],
            downgraded_from: 0x009E,
            downgraded_to: 0x002F,
            adversary_budget: 2,
            vulnerability_type: "cipher_downgrade".into(),
            library_name: "openssl".into(),
            protocol: Protocol::Tls,
        };
        assert!(trace.validate().is_ok());
    }

    #[test]
    fn attack_trace_empty_invalid() {
        let trace = AttackTrace {
            steps: vec![],
            downgraded_from: 1,
            downgraded_to: 2,
            adversary_budget: 0,
            vulnerability_type: "test".into(),
            library_name: "test".into(),
            protocol: Protocol::Tls,
        };
        assert!(trace.validate().is_err());
    }

    #[test]
    fn attack_trace_discontinuity() {
        let trace = AttackTrace {
            steps: vec![
                AttackStep {
                    step_number: 0, action: "forward".into(),
                    from_state: 0, to_state: 1,
                    message: None, cipher_suite_id: None, bytes: None,
                },
                AttackStep {
                    step_number: 1, action: "modify".into(),
                    from_state: 5, to_state: 6, // gap: 1 != 5
                    message: None, cipher_suite_id: None, bytes: None,
                },
            ],
            downgraded_from: 1, downgraded_to: 2,
            adversary_budget: 2,
            vulnerability_type: "test".into(),
            library_name: "test".into(),
            protocol: Protocol::Tls,
        };
        assert!(trace.validate().is_err());
    }

    #[test]
    fn certificate_validity() {
        let cert = AnalysisCertificate {
            id: "abc-123".into(),
            library_name: "openssl".into(),
            protocol: Protocol::Tls,
            timestamp: "2024-01-01T00:00:00Z".into(),
            states_explored: 100,
            paths_explored: 50,
            coverage_pct: 85.0,
            depth_bound: 64,
            action_bound: 4,
            vulnerabilities_found: vec!["CVE-2014-0224".into()],
            attack_traces: vec![],
            hash: "deadbeef".into(),
            version: "0.1.0".into(),
        };
        assert!(cert.is_valid());
        assert!(cert.meets_coverage(80.0));
        assert!(!cert.meets_coverage(90.0));
        assert!(cert.has_vulnerabilities());
    }

    #[test]
    fn bisimulation_classes() {
        let mut sm = StateMachine::new("test", Protocol::Tls);
        sm.add_state(State::new(0, "s0", HandshakePhase::Initial));
        sm.add_state(State::new(1, "s1", HandshakePhase::Initial));
        sm.add_state(State::new(2, "s2", HandshakePhase::ClientHello));
        let classes = sm.bisimulation_classes();
        // s0 and s1 should be in the same class (both Initial, no version).
        let init_class = classes.values().find(|v| v.contains(&0)).unwrap();
        assert!(init_class.contains(&1));
        assert!(!init_class.contains(&2));
    }

    #[test]
    fn deviation_severity_ordering() {
        assert!(DeviationSeverity::Critical > DeviationSeverity::High);
        assert!(DeviationSeverity::High > DeviationSeverity::Medium);
        assert!(DeviationSeverity::Medium > DeviationSeverity::Low);
    }
}
