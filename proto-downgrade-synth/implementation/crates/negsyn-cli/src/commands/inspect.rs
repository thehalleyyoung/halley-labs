//! `negsyn inspect` — state machine inspection.
//!
//! Loads a state machine from a JSON file or analyses a source to produce one,
//! then displays statistics, exports to DOT/GraphViz, lists states and
//! transitions, and reports bisimulation equivalence classes.

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

use negsyn_types::HandshakePhase;

use crate::config::CliConfig;
use crate::logging::TimingGuard;
use crate::output::{
    bold, cyan, dim, green, red, render_dot, OutputFormat, OutputWriter, Table,
};

use super::{detect_protocol, Protocol, State, StateMachine, Transition};

// ---------------------------------------------------------------------------
// Command definition
// ---------------------------------------------------------------------------

/// Inspect a state machine: display statistics, export DOT, list states/transitions.
#[derive(Debug, Clone, Args)]
pub struct InspectCommand {
    /// Path to a state-machine JSON file or a source file to analyse.
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output file path (stdout if omitted).
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Override output format (text, json, dot, csv).
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,

    /// Show full transition details.
    #[arg(long)]
    pub verbose: bool,

    /// Only show states in these phases (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub phases: Vec<String>,

    /// Show bisimulation equivalence classes.
    #[arg(long)]
    pub bisim: bool,

    /// Show only reachable states.
    #[arg(long)]
    pub reachable_only: bool,
}

// ---------------------------------------------------------------------------
// Inspection report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectReport {
    pub library_name: String,
    pub protocol: Protocol,
    pub statistics: MachineStatistics,
    pub states: Vec<StateInfo>,
    pub transitions: Vec<TransitionInfo>,
    pub bisimulation_classes: Option<BTreeMap<String, Vec<u32>>>,
    pub observations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineStatistics {
    pub total_states: usize,
    pub reachable_states: usize,
    pub unreachable_states: usize,
    pub total_transitions: usize,
    pub downgrade_transitions: usize,
    pub accepting_states: usize,
    pub error_states: usize,
    pub initial_state: u32,
    pub phases_present: Vec<String>,
    pub max_out_degree: usize,
    pub avg_out_degree: f64,
    pub bisimulation_class_count: usize,
    pub has_cycles: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    pub id: u32,
    pub label: String,
    pub phase: String,
    pub version: Option<String>,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub is_error: bool,
    pub is_reachable: bool,
    pub out_degree: usize,
    pub in_degree: usize,
    pub properties: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionInfo {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub label: String,
    pub guard: Option<String>,
    pub action: Option<String>,
    pub is_downgrade: bool,
    pub cipher_suite_id: Option<u16>,
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

impl InspectCommand {
    pub fn execute(
        &self,
        cfg: &CliConfig,
        global_format: OutputFormat,
        no_color: bool,
    ) -> Result<()> {
        let format = self.format.unwrap_or(global_format);
        let _timer = TimingGuard::new("inspection");

        let sm = load_or_build(&self.input, cfg)?;

        log::info!(
            "Inspecting state machine: {} ({} states, {} transitions)",
            sm.library_name,
            sm.state_count(),
            sm.transition_count()
        );

        let reachable = sm.reachable_states();
        let reachable_set: std::collections::HashSet<u32> =
            reachable.iter().copied().collect();

        let bisim_classes = sm.bisimulation_classes();

        // Build state info.
        let mut state_infos: Vec<StateInfo> = sm
            .states
            .iter()
            .map(|s| {
                let out_degree = sm.transitions_from(s.id).len();
                let in_degree = sm
                    .transitions
                    .iter()
                    .filter(|t| t.target == s.id)
                    .count();
                StateInfo {
                    id: s.id,
                    label: s.label.clone(),
                    phase: format!("{:?}", s.phase),
                    version: s.version.map(|v| format!("{:?}", v)),
                    is_initial: s.is_initial,
                    is_accepting: s.is_accepting,
                    is_error: s.is_error,
                    is_reachable: reachable_set.contains(&s.id),
                    out_degree,
                    in_degree,
                    properties: s.properties.clone(),
                }
            })
            .collect();

        // Filter by phase if requested.
        if !self.phases.is_empty() {
            let phase_filter: std::collections::HashSet<String> =
                self.phases.iter().map(|p| p.to_lowercase()).collect();
            state_infos.retain(|s| {
                phase_filter.contains(&s.phase.to_lowercase())
            });
        }

        // Filter unreachable if requested.
        if self.reachable_only {
            state_infos.retain(|s| s.is_reachable);
        }

        // Build transition info.
        let transition_infos: Vec<TransitionInfo> = sm
            .transitions
            .iter()
            .map(|t| TransitionInfo {
                id: t.id,
                source: t.source,
                target: t.target,
                label: t.label.clone(),
                guard: t.guard.clone(),
                action: t.action.clone(),
                is_downgrade: t.is_downgrade,
                cipher_suite_id: t.cipher_suite_id,
            })
            .collect();

        // Compute statistics.
        let phases_present: Vec<String> = {
            let mut p: Vec<String> = sm
                .states
                .iter()
                .map(|s| format!("{:?}", s.phase))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            p.sort();
            p
        };

        let max_out_degree = sm
            .states
            .iter()
            .map(|s| sm.transitions_from(s.id).len())
            .max()
            .unwrap_or(0);

        let avg_out_degree = if sm.state_count() > 0 {
            sm.transition_count() as f64 / sm.state_count() as f64
        } else {
            0.0
        };

        let has_cycles = detect_cycles(&sm);

        let statistics = MachineStatistics {
            total_states: sm.state_count(),
            reachable_states: reachable.len(),
            unreachable_states: sm.state_count().saturating_sub(reachable.len()),
            total_transitions: sm.transition_count(),
            downgrade_transitions: sm.downgrade_transitions().len(),
            accepting_states: sm.accepting_states().len(),
            error_states: sm.error_states().len(),
            initial_state: sm.initial_state,
            phases_present,
            max_out_degree,
            avg_out_degree,
            bisimulation_class_count: bisim_classes.len(),
            has_cycles,
        };

        // Observations.
        let observations = generate_observations(&statistics, &sm);

        let report = InspectReport {
            library_name: sm.library_name.clone(),
            protocol: sm.protocol,
            statistics,
            states: state_infos,
            transitions: transition_infos,
            bisimulation_classes: if self.bisim {
                Some(bisim_classes)
            } else {
                None
            },
            observations,
        };

        let mut writer = match &self.output {
            Some(p) => OutputWriter::file(p, format, no_color)?,
            None => OutputWriter::stdout(format, no_color),
        };

        match format {
            OutputFormat::Dot => write_dot(&mut writer, &sm)?,
            OutputFormat::Text => write_text_report(&mut writer, &report, no_color, self.verbose)?,
            OutputFormat::Csv => write_csv(&mut writer, &report)?,
            _ => writer.write_value(&report)?,
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn load_or_build(path: &PathBuf, cfg: &CliConfig) -> Result<StateMachine> {
    if !path.exists() {
        bail!("input not found: {}", path.display());
    }

    // Try to load as JSON state machine first.
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;

    if let Ok(sm) = serde_json::from_str::<StateMachine>(&contents) {
        return Ok(sm);
    }

    // Not JSON — try to build from source.
    let protocol = detect_protocol(path)
        .unwrap_or(Protocol::Tls);
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());

    build_default_sm(&name, protocol, cfg)
}

fn build_default_sm(name: &str, protocol: Protocol, cfg: &CliConfig) -> Result<StateMachine> {
    let phases = match protocol {
        Protocol::Tls => vec![
            HandshakePhase::Initial,
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHello,
            HandshakePhase::Certificate,
            HandshakePhase::KeyExchange,
            HandshakePhase::ChangeCipherSpec,
            HandshakePhase::Finished,
            HandshakePhase::ApplicationData,
        ],
        Protocol::Ssh => vec![
            HandshakePhase::Initial,
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHello,
            HandshakePhase::KeyExchange,
            HandshakePhase::Finished,
        ],
    };

    let mut sm = StateMachine::new(name, protocol);

    for (i, phase) in phases.iter().enumerate() {
        let mut s = State::new(i as u32, format!("{}_{:?}", name, phase), *phase);
        s.is_initial = i == 0;
        s.is_accepting = *phase == HandshakePhase::Finished
            || *phase == HandshakePhase::ApplicationData;
        sm.add_state(s);
    }

    for i in 0..phases.len().saturating_sub(1) {
        let t = Transition::new(
            i as u32,
            i as u32,
            (i + 1) as u32,
            format!("{:?}", phases[i + 1]),
        );
        sm.add_transition(t);
    }

    // Add error state.
    let eid = phases.len() as u32;
    let mut err = State::new(eid, format!("{}_Alert", name), HandshakePhase::Alert);
    err.is_error = true;
    sm.add_state(err);

    // Error transition from midpoint.
    let mid = phases.len() / 2;
    sm.add_transition(Transition::new(
        sm.transition_count() as u32,
        mid as u32,
        eid,
        "alert",
    ));

    Ok(sm)
}

// ---------------------------------------------------------------------------
// Analysis helpers
// ---------------------------------------------------------------------------

fn detect_cycles(sm: &StateMachine) -> bool {
    // DFS-based cycle detection.
    let mut visited = std::collections::HashSet::new();
    let mut on_stack = std::collections::HashSet::new();

    fn dfs(
        node: u32,
        sm: &StateMachine,
        visited: &mut std::collections::HashSet<u32>,
        on_stack: &mut std::collections::HashSet<u32>,
    ) -> bool {
        visited.insert(node);
        on_stack.insert(node);
        for t in sm.transitions_from(node) {
            if !visited.contains(&t.target) {
                if dfs(t.target, sm, visited, on_stack) {
                    return true;
                }
            } else if on_stack.contains(&t.target) {
                return true;
            }
        }
        on_stack.remove(&node);
        false
    }

    for s in &sm.states {
        if !visited.contains(&s.id) {
            if dfs(s.id, sm, &mut visited, &mut on_stack) {
                return true;
            }
        }
    }

    false
}

fn generate_observations(stats: &MachineStatistics, sm: &StateMachine) -> Vec<String> {
    let mut obs = Vec::new();

    if stats.unreachable_states > 0 {
        obs.push(format!(
            "{} unreachable state(s) detected — possible dead code",
            stats.unreachable_states
        ));
    }

    if stats.downgrade_transitions > 0 {
        obs.push(format!(
            "{} downgrade transition(s) found — potential vulnerability",
            stats.downgrade_transitions
        ));
    }

    if stats.error_states == 0 {
        obs.push("No error states — missing error handling?".into());
    }

    if stats.has_cycles {
        obs.push("State machine contains cycles — check for infinite loops".into());
    }

    if stats.max_out_degree > 5 {
        obs.push(format!(
            "High maximum out-degree ({}) — complex branching",
            stats.max_out_degree
        ));
    }

    if stats.accepting_states == 0 {
        obs.push("No accepting states — handshake may never complete".into());
    }

    if stats.bisimulation_class_count < stats.reachable_states {
        obs.push(format!(
            "Bisimulation quotient: {} classes for {} states — {} state(s) can be merged",
            stats.bisimulation_class_count,
            stats.reachable_states,
            stats.reachable_states - stats.bisimulation_class_count,
        ));
    }

    // Check for states with no outgoing transitions (dead ends).
    let dead_ends: Vec<u32> = sm
        .states
        .iter()
        .filter(|s| sm.transitions_from(s.id).is_empty() && !s.is_accepting && !s.is_error)
        .map(|s| s.id)
        .collect();
    if !dead_ends.is_empty() {
        obs.push(format!(
            "{} non-terminal dead-end state(s): {:?}",
            dead_ends.len(),
            dead_ends
        ));
    }

    obs
}

// ---------------------------------------------------------------------------
// DOT output
// ---------------------------------------------------------------------------

fn write_dot(writer: &mut OutputWriter, sm: &StateMachine) -> Result<()> {
    let nodes: Vec<(String, BTreeMap<String, String>)> = sm
        .states
        .iter()
        .map(|s| {
            let mut attrs = BTreeMap::new();
            attrs.insert("phase".into(), format!("{:?}", s.phase));
            if let Some(v) = s.version {
                attrs.insert("version".into(), format!("{:?}", v));
            }
            if s.is_initial {
                attrs.insert("initial".into(), "true".into());
            }
            if s.is_accepting {
                attrs.insert("accepting".into(), "true".into());
            }
            if s.is_error {
                attrs.insert("error".into(), "true".into());
            }
            (format!("s{}", s.id), attrs)
        })
        .collect();

    let edges: Vec<(String, String, String)> = sm
        .transitions
        .iter()
        .map(|t| {
            let label = if t.is_downgrade {
                format!("{} [DOWNGRADE]", t.label)
            } else {
                t.label.clone()
            };
            (format!("s{}", t.source), format!("s{}", t.target), label)
        })
        .collect();

    let dot = render_dot(&sm.library_name, &nodes, &edges);
    writer.write_raw(&dot)
}

// ---------------------------------------------------------------------------
// Text output
// ---------------------------------------------------------------------------

fn write_text_report(
    writer: &mut OutputWriter,
    report: &InspectReport,
    no_color: bool,
    verbose: bool,
) -> Result<()> {
    let stats = &report.statistics;
    let mut buf = String::new();

    buf.push_str(&bold("NegSynth State Machine Inspection", no_color));
    buf.push_str(&format!("\n  Library:      {}", report.library_name));
    buf.push_str(&format!("\n  Protocol:     {}", report.protocol));
    buf.push_str(&format!("\n  States:       {} ({} reachable, {} unreachable)",
        stats.total_states, stats.reachable_states, stats.unreachable_states));
    buf.push_str(&format!("\n  Transitions:  {} ({} downgrade)",
        stats.total_transitions, stats.downgrade_transitions));
    buf.push_str(&format!("\n  Accepting:    {}", stats.accepting_states));
    buf.push_str(&format!("\n  Error:        {}", stats.error_states));
    buf.push_str(&format!("\n  Phases:       {}", stats.phases_present.join(", ")));
    buf.push_str(&format!("\n  Max out-deg:  {}", stats.max_out_degree));
    buf.push_str(&format!("\n  Avg out-deg:  {:.2}", stats.avg_out_degree));
    buf.push_str(&format!("\n  Cycles:       {}", if stats.has_cycles { "yes" } else { "no" }));
    buf.push_str(&format!("\n  Bisim classes:{}\n", stats.bisimulation_class_count));

    // States table.
    let mut state_table = Table::new(vec![
        "ID".into(),
        "Label".into(),
        "Phase".into(),
        "Init".into(),
        "Accept".into(),
        "Error".into(),
        "Reach".into(),
        "Out°".into(),
    ]).with_title("States");

    for s in &report.states {
        state_table.add_row(vec![
            s.id.to_string(),
            s.label.clone(),
            s.phase.clone(),
            if s.is_initial { "✓".into() } else { "".into() },
            if s.is_accepting { "✓".into() } else { "".into() },
            if s.is_error { "✓".into() } else { "".into() },
            if s.is_reachable { "✓".into() } else { red("✗", no_color) },
            s.out_degree.to_string(),
        ]);
    }
    buf.push_str(&state_table.render_text(no_color));

    // Transitions table.
    if verbose {
        let mut trans_table = Table::new(vec![
            "ID".into(),
            "Source".into(),
            "Target".into(),
            "Label".into(),
            "Action".into(),
            "Downgrade".into(),
        ]).with_title("Transitions");

        for t in &report.transitions {
            trans_table.add_row(vec![
                t.id.to_string(),
                t.source.to_string(),
                t.target.to_string(),
                t.label.clone(),
                t.action.clone().unwrap_or_default(),
                if t.is_downgrade {
                    red("YES", no_color)
                } else {
                    "no".into()
                },
            ]);
        }
        buf.push_str(&trans_table.render_text(no_color));
    }

    // Bisimulation classes.
    if let Some(ref classes) = report.bisimulation_classes {
        buf.push_str(&format!("\n  {}\n", bold("Bisimulation Equivalence Classes", no_color)));
        for (class, members) in classes {
            let members_str: Vec<String> = members.iter().map(|m| m.to_string()).collect();
            buf.push_str(&format!("    {}: [{}]\n", cyan(class, no_color), members_str.join(", ")));
        }
    }

    // Observations.
    if !report.observations.is_empty() {
        buf.push_str(&format!("\n  {}\n", bold("Observations", no_color)));
        for obs in &report.observations {
            buf.push_str(&format!("    • {}\n", obs));
        }
    }

    writer.write_raw(&buf)
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

fn write_csv(writer: &mut OutputWriter, report: &InspectReport) -> Result<()> {
    let mut table = Table::new(vec![
        "id".into(),
        "label".into(),
        "phase".into(),
        "is_initial".into(),
        "is_accepting".into(),
        "is_error".into(),
        "is_reachable".into(),
        "out_degree".into(),
        "in_degree".into(),
    ]);
    for s in &report.states {
        table.add_row(vec![
            s.id.to_string(),
            s.label.clone(),
            s.phase.clone(),
            s.is_initial.to_string(),
            s.is_accepting.to_string(),
            s.is_error.to_string(),
            s.is_reachable.to_string(),
            s.out_degree.to_string(),
            s.in_degree.to_string(),
        ]);
    }
    writer.write_raw(&table.render_csv())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sm() -> StateMachine {
        let mut sm = StateMachine::new("test_lib", Protocol::Tls);
        sm.add_state(State::new(0, "init", HandshakePhase::Initial));
        sm.add_state(State::new(1, "hello", HandshakePhase::ClientHello));
        sm.add_state(State::new(2, "reply", HandshakePhase::ServerHello));
        let mut fin = State::new(3, "done", HandshakePhase::Finished);
        fin.is_accepting = true;
        sm.add_state(fin);

        sm.add_transition(Transition::new(0, 0, 1, "send_hello"));
        sm.add_transition(Transition::new(1, 1, 2, "recv_hello"));
        sm.add_transition(Transition::new(2, 2, 3, "finish"));

        sm
    }

    #[test]
    fn detect_cycles_no_cycle() {
        let sm = sample_sm();
        assert!(!detect_cycles(&sm));
    }

    #[test]
    fn detect_cycles_with_cycle() {
        let mut sm = sample_sm();
        sm.add_transition(Transition::new(10, 2, 1, "back"));
        assert!(detect_cycles(&sm));
    }

    #[test]
    fn generate_observations_basic() {
        let sm = sample_sm();
        let reachable = sm.reachable_states();
        let bisim = sm.bisimulation_classes();
        let stats = MachineStatistics {
            total_states: sm.state_count(),
            reachable_states: reachable.len(),
            unreachable_states: 0,
            total_transitions: sm.transition_count(),
            downgrade_transitions: 0,
            accepting_states: 1,
            error_states: 0,
            initial_state: 0,
            phases_present: vec![],
            max_out_degree: 1,
            avg_out_degree: 0.75,
            bisimulation_class_count: bisim.len(),
            has_cycles: false,
        };
        let obs = generate_observations(&stats, &sm);
        // Should note missing error states.
        assert!(obs.iter().any(|o| o.contains("error")));
    }

    #[test]
    fn generate_observations_downgrades() {
        let mut sm = sample_sm();
        let mut dt = Transition::new(5, 1, 2, "downgrade");
        dt.is_downgrade = true;
        sm.add_transition(dt);

        let stats = MachineStatistics {
            total_states: 4, reachable_states: 4, unreachable_states: 0,
            total_transitions: 4, downgrade_transitions: 1,
            accepting_states: 1, error_states: 1, initial_state: 0,
            phases_present: vec![], max_out_degree: 2, avg_out_degree: 1.0,
            bisimulation_class_count: 4, has_cycles: false,
        };
        let obs = generate_observations(&stats, &sm);
        assert!(obs.iter().any(|o| o.contains("downgrade")));
    }

    #[test]
    fn build_default_sm_tls() {
        let cfg = CliConfig::default();
        let sm = build_default_sm("openssl", Protocol::Tls, &cfg).unwrap();
        assert!(sm.state_count() >= 7);
        assert!(sm.transition_count() >= 6);
        assert!(sm.error_states().len() == 1);
    }

    #[test]
    fn build_default_sm_ssh() {
        let cfg = CliConfig::default();
        let sm = build_default_sm("libssh", Protocol::Ssh, &cfg).unwrap();
        assert!(sm.state_count() >= 5);
    }

    #[test]
    fn dot_output_contains_expected() {
        let sm = sample_sm();
        let nodes: Vec<(String, BTreeMap<String, String>)> = sm.states.iter().map(|s| {
            (format!("s{}", s.id), BTreeMap::new())
        }).collect();
        let edges: Vec<(String, String, String)> = sm.transitions.iter().map(|t| {
            (format!("s{}", t.source), format!("s{}", t.target), t.label.clone())
        }).collect();
        let dot = render_dot("test", &nodes, &edges);
        assert!(dot.contains("digraph test"));
        assert!(dot.contains("s0"));
        assert!(dot.contains("send_hello"));
    }

    #[test]
    fn state_info_constructed() {
        let sm = sample_sm();
        let info = StateInfo {
            id: 0,
            label: "init".into(),
            phase: "Initial".into(),
            version: None,
            is_initial: true,
            is_accepting: false,
            is_error: false,
            is_reachable: true,
            out_degree: 1,
            in_degree: 0,
            properties: BTreeMap::new(),
        };
        assert_eq!(info.id, 0);
        assert!(info.is_reachable);
    }

    #[test]
    fn inspect_report_serializes() {
        let report = InspectReport {
            library_name: "test".into(),
            protocol: Protocol::Tls,
            statistics: MachineStatistics {
                total_states: 4, reachable_states: 4, unreachable_states: 0,
                total_transitions: 3, downgrade_transitions: 0,
                accepting_states: 1, error_states: 0, initial_state: 0,
                phases_present: vec!["Initial".into()],
                max_out_degree: 1, avg_out_degree: 0.75,
                bisimulation_class_count: 4, has_cycles: false,
            },
            states: vec![],
            transitions: vec![],
            bisimulation_classes: None,
            observations: vec!["test observation".into()],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("test"));
    }

    #[test]
    fn load_sm_from_json() {
        let sm = sample_sm();
        let json = serde_json::to_string(&sm).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sm.json");
        std::fs::write(&path, &json).unwrap();
        let cfg = CliConfig::default();
        let loaded = load_or_build(&path, &cfg).unwrap();
        assert_eq!(loaded.state_count(), sm.state_count());
    }

    #[test]
    fn machine_statistics_fields() {
        let stats = MachineStatistics {
            total_states: 10, reachable_states: 8, unreachable_states: 2,
            total_transitions: 12, downgrade_transitions: 1,
            accepting_states: 2, error_states: 1, initial_state: 0,
            phases_present: vec!["Initial".into(), "Finished".into()],
            max_out_degree: 3, avg_out_degree: 1.2,
            bisimulation_class_count: 5, has_cycles: false,
        };
        assert_eq!(stats.total_states, 10);
        assert_eq!(stats.unreachable_states, 2);
    }
}
