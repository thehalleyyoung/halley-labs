//! # negsyn-eval
//!
//! Evaluation harness for the NegSynth protocol downgrade attack synthesis tool.
//! Provides benchmarking, CVE oracle testing, cross-library differential analysis,
//! coverage metrics, and reporting infrastructure.

pub mod benchmark;
pub mod bounded_exhaustive_validator;
pub mod coverage;
pub mod cve_oracle;
pub mod differential;
pub mod harness;
pub mod pipeline;
pub mod report;
pub mod scenario;

pub use benchmark::{
    BenchmarkResult, BenchmarkSuite, MemoryBenchmark, MergeSpeedupBenchmark,
    ScalabilityBenchmark,
};
pub use bounded_exhaustive_validator::{
    Axiom, AxiomCheckResult, AxiomCoverageReport, AxiomValidator, AxiomViolation,
    BoundCalibrationReport, BoundCalibrator, BoundedExhaustiveValidator, SmtPerformanceBenchmark,
    SmtPerformanceReport, ValidationReport,
};
pub use coverage::{
    CoverageAnalyzer, CoverageBoundValidator, PathCoverage, StateCoverage, TransitionCoverage,
};
pub use cve_oracle::{CveEntry, CveOracle, OracleResult, OracleTest};
pub use differential::{
    BehavioralDeviation, CrossLibraryCertificate, DeviationRanker, DifferentialAnalyzer,
    WireProtocolAlignment,
};
pub use harness::{TestCase, TestHarness, TestReport, TestRunner};
pub use pipeline::{AnalysisPipeline, PipelineConfig, PipelineResult};
pub use report::{HumanReport, JsonReport, ReportGenerator, SarifReport, SummaryReport};
pub use scenario::{
    AdversaryScenario, CipherSuiteScenario, ExtensionScenario, ScenarioGenerator,
    VersionScenario,
};

use negsyn_types::{HandshakePhase, ProtocolVersion};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Simulated LTS state for state machine extraction.
/// Compatible with what negsyn-extract would produce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtsState {
    pub id: u32,
    pub label: String,
    pub phase: HandshakePhase,
    pub version: Option<ProtocolVersion>,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub is_error: bool,
    pub properties: BTreeMap<String, String>,
}

impl LtsState {
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

/// Simulated LTS transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtsTransition {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub label: String,
    pub guard: Option<String>,
    pub action: Option<String>,
    pub cipher_suite_id: Option<u16>,
    pub is_downgrade: bool,
}

impl LtsTransition {
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

/// Labeled Transition System representing protocol negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lts {
    pub states: Vec<LtsState>,
    pub transitions: Vec<LtsTransition>,
    pub initial_state: u32,
    pub library_name: String,
}

impl Lts {
    pub fn new(library_name: impl Into<String>) -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            initial_state: 0,
            library_name: library_name.into(),
        }
    }

    pub fn add_state(&mut self, state: LtsState) {
        self.states.push(state);
    }

    pub fn add_transition(&mut self, transition: LtsTransition) {
        self.transitions.push(transition);
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn get_state(&self, id: u32) -> Option<&LtsState> {
        self.states.iter().find(|s| s.id == id)
    }

    pub fn transitions_from(&self, state_id: u32) -> Vec<&LtsTransition> {
        self.transitions
            .iter()
            .filter(|t| t.source == state_id)
            .collect()
    }

    pub fn transitions_to(&self, state_id: u32) -> Vec<&LtsTransition> {
        self.transitions
            .iter()
            .filter(|t| t.target == state_id)
            .collect()
    }

    pub fn accepting_states(&self) -> Vec<&LtsState> {
        self.states.iter().filter(|s| s.is_accepting).collect()
    }

    pub fn error_states(&self) -> Vec<&LtsState> {
        self.states.iter().filter(|s| s.is_error).collect()
    }

    pub fn reachable_states(&self) -> Vec<u32> {
        let mut visited = std::collections::BTreeSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.initial_state);
        visited.insert(self.initial_state);
        while let Some(s) = queue.pop_front() {
            for t in self.transitions_from(s) {
                if visited.insert(t.target) {
                    queue.push_back(t.target);
                }
            }
        }
        visited.into_iter().collect()
    }
}

/// Simulated protocol slice from negsyn-slicer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolSlice {
    pub library_name: String,
    pub function_name: String,
    pub instructions: Vec<SlicedInstruction>,
    pub entry_point: u64,
    pub exit_points: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicedInstruction {
    pub address: u64,
    pub mnemonic: String,
    pub operands: Vec<String>,
    pub is_branch: bool,
    pub is_call: bool,
    pub is_protocol_relevant: bool,
}

/// Simulated SMT encoding result from negsyn-encode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtEncoding {
    pub assertions: Vec<String>,
    pub variables: Vec<SmtVariable>,
    pub check_sat_result: Option<SmtResult>,
    pub node_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtVariable {
    pub name: String,
    pub sort: String,
    pub is_cipher_suite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtResult {
    Sat,
    Unsat,
    Unknown,
    Timeout,
}

impl fmt::Display for SmtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtResult::Sat => write!(f, "SAT"),
            SmtResult::Unsat => write!(f, "UNSAT"),
            SmtResult::Unknown => write!(f, "UNKNOWN"),
            SmtResult::Timeout => write!(f, "TIMEOUT"),
        }
    }
}

/// Simulated CEGAR refinement result from negsyn-concrete.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarResult {
    pub iterations: u32,
    pub is_genuine: bool,
    pub attack_trace: Option<AttackTrace>,
    pub refinements: Vec<CegarRefinement>,
    pub final_smt_result: SmtResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarRefinement {
    pub iteration: u32,
    pub spurious_path: Vec<u32>,
    pub new_predicate: String,
}

/// An attack trace showing a concrete downgrade path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTrace {
    pub steps: Vec<AttackStep>,
    pub downgraded_from: u16,
    pub downgraded_to: u16,
    pub adversary_budget: u32,
    pub vulnerability_type: String,
}

impl AttackTrace {
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackStep {
    pub step_number: u32,
    pub action: String,
    pub from_state: u32,
    pub to_state: u32,
    pub message: Option<String>,
    pub cipher_suite_id: Option<u16>,
}

/// Certificate of analysis completeness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCertificate {
    pub id: String,
    pub library_name: String,
    pub timestamp: String,
    pub states_explored: usize,
    pub paths_explored: usize,
    pub coverage_pct: f64,
    pub vulnerabilities_found: Vec<String>,
    pub hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lts_construction() {
        let mut lts = Lts::new("test-lib");
        lts.add_state(LtsState::new(0, "init", HandshakePhase::Init));
        let mut s1 = LtsState::new(1, "hello", HandshakePhase::ClientHelloSent);
        s1.is_accepting = true;
        lts.add_state(s1);
        lts.add_transition(LtsTransition::new(0, 0, 1, "send_hello"));
        assert_eq!(lts.state_count(), 2);
        assert_eq!(lts.transition_count(), 1);
        assert_eq!(lts.accepting_states().len(), 1);
        assert_eq!(lts.reachable_states(), vec![0, 1]);
    }

    #[test]
    fn test_lts_transitions() {
        let mut lts = Lts::new("test-lib");
        lts.add_state(LtsState::new(0, "s0", HandshakePhase::Init));
        lts.add_state(LtsState::new(1, "s1", HandshakePhase::ClientHelloSent));
        lts.add_state(LtsState::new(2, "s2", HandshakePhase::ServerHelloReceived));
        lts.add_transition(LtsTransition::new(0, 0, 1, "t0"));
        lts.add_transition(LtsTransition::new(1, 0, 2, "t1"));
        lts.add_transition(LtsTransition::new(2, 1, 2, "t2"));
        assert_eq!(lts.transitions_from(0).len(), 2);
        assert_eq!(lts.transitions_to(2).len(), 2);
    }

    #[test]
    fn test_smt_result_display() {
        assert_eq!(format!("{}", SmtResult::Sat), "SAT");
        assert_eq!(format!("{}", SmtResult::Unsat), "UNSAT");
    }

    #[test]
    fn test_attack_trace_steps() {
        let trace = AttackTrace {
            steps: vec![
                AttackStep {
                    step_number: 0,
                    action: "intercept".into(),
                    from_state: 0,
                    to_state: 1,
                    message: Some("ClientHello".into()),
                    cipher_suite_id: Some(0x002F),
                },
                AttackStep {
                    step_number: 1,
                    action: "modify".into(),
                    from_state: 1,
                    to_state: 2,
                    message: Some("ServerHello".into()),
                    cipher_suite_id: Some(0x0001),
                },
            ],
            downgraded_from: 0x002F,
            downgraded_to: 0x0001,
            adversary_budget: 2,
            vulnerability_type: "cipher_downgrade".into(),
        };
        assert_eq!(trace.step_count(), 2);
    }
}
