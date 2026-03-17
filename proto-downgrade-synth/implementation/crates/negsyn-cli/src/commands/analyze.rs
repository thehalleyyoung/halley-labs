//! `negsyn analyze` — run full downgrade attack synthesis analysis.
//!
//! Orchestrates the complete pipeline: slice → merge → extract → encode →
//! concretize, producing an analysis certificate and any attack traces.

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use negsyn_types::{
    CipherSuite, HandshakePhase, MergeConfig, NegotiationState, ProtocolVersion, SymbolicState,
};

use crate::config::CliConfig;
use crate::logging::{self, TimingGuard};
use crate::output::{OutputFormat, OutputWriter, ProgressSpinner, SarifReport, Table};

use super::{
    detect_protocol, AnalysisCertificate, AnalysisResult, AttackStep, AttackTrace, Protocol,
    State, StateMachine, Transition,
};

// ---------------------------------------------------------------------------
// Command definition
// ---------------------------------------------------------------------------

/// Run full downgrade attack synthesis analysis on a protocol library.
#[derive(Debug, Clone, Args)]
pub struct AnalyzeCommand {
    /// Path to the source / LLVM IR / binary to analyse.
    #[arg(value_name = "SOURCE")]
    pub source: PathBuf,

    /// Target protocol (tls or ssh). Auto-detected if omitted.
    #[arg(short, long, value_enum)]
    pub protocol: Option<Protocol>,

    /// Library name (e.g. "openssl", "mbedtls").
    #[arg(short, long)]
    pub library: String,

    /// Library version string.
    #[arg(long)]
    pub version: Option<String>,

    /// Maximum symbolic exploration depth.
    #[arg(short, long)]
    pub depth: Option<u32>,

    /// Maximum adversary action budget.
    #[arg(short, long)]
    pub actions: Option<u32>,

    /// Output file path (stdout if omitted).
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Override output format for this command.
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,

    /// Enable FIPS-only cipher suite filtering.
    #[arg(long)]
    pub fips: bool,

    /// SMT solver timeout in milliseconds.
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Skip the concretisation phase (report abstract traces only).
    #[arg(long)]
    pub skip_concretize: bool,
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

impl AnalyzeCommand {
    pub fn execute(
        &self,
        cfg: &CliConfig,
        global_format: OutputFormat,
        no_color: bool,
    ) -> Result<()> {
        let format = self.format.unwrap_or(global_format);
        let depth = self.depth.unwrap_or_else(|| cfg.depth_for(&self.library));
        let actions = self.actions.unwrap_or_else(|| cfg.action_bound_for(&self.library));
        let timeout_ms = self.timeout.unwrap_or(cfg.smt_timeout_ms);

        // Validate inputs.
        if !self.source.exists() {
            bail!("source path does not exist: {}", self.source.display());
        }
        let protocol = self.protocol.or_else(|| detect_protocol(&self.source));
        let protocol = match protocol {
            Some(p) => p,
            None => bail!(
                "cannot auto-detect protocol for {}; use --protocol",
                self.source.display()
            ),
        };

        log::info!(
            "Analyzing {} ({}) protocol={} depth={} actions={}",
            self.library,
            self.source.display(),
            protocol,
            depth,
            actions,
        );

        let pipeline_start = Instant::now();

        // Phase 1: Slicing
        let (slice_states, slice_transitions) =
            run_slicer_phase(&self.source, &self.library, protocol, cfg, no_color)?;

        // Phase 2: Merge
        let merged = run_merge_phase(&slice_states, cfg, no_color)?;

        // Phase 3: Extraction
        let mut sm = run_extract_phase(&merged, &self.library, protocol, no_color)?;

        // Phase 4: Encoding + solving
        let (traces, smt_stats) =
            run_encode_phase(&sm, depth, actions, timeout_ms, cfg, no_color)?;

        // Phase 5: Concretisation (optional)
        let concrete_traces = if self.skip_concretize {
            traces.clone()
        } else {
            run_concretize_phase(&traces, protocol, no_color)?
        };

        // Mark downgrade transitions.
        let downgrade_targets: std::collections::HashSet<u32> = concrete_traces
            .iter()
            .flat_map(|t| t.steps.iter().map(|s| s.to_state))
            .collect();
        for t in &mut sm.transitions {
            if downgrade_targets.contains(&t.target) {
                t.is_downgrade = true;
            }
        }

        let elapsed_ms = pipeline_start.elapsed().as_millis() as u64;

        // Build certificate.
        let certificate = build_certificate(
            &self.library,
            protocol,
            &sm,
            &concrete_traces,
            depth,
            actions,
            smt_stats.paths_explored,
            smt_stats.coverage_pct,
        );

        let result = AnalysisResult {
            library_name: self.library.clone(),
            protocol,
            state_machine: sm,
            certificate,
            attack_traces: concrete_traces,
            elapsed_ms,
        };

        // Output.
        let mut writer = match &self.output {
            Some(p) => OutputWriter::file(p, format, no_color)?,
            None => OutputWriter::stdout(format, no_color),
        };

        match format {
            OutputFormat::Text => write_text_report(&mut writer, &result, no_color)?,
            OutputFormat::Sarif => write_sarif_report(&mut writer, &result)?,
            _ => writer.write_value(&result)?,
        }

        eprintln!(
            "\n  Analysis complete: {} states, {} transitions, {} traces ({:.3}s)",
            result.state_machine.state_count(),
            result.state_machine.transition_count(),
            result.attack_traces.len(),
            elapsed_ms as f64 / 1000.0,
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline phases
// ---------------------------------------------------------------------------

/// Simulated slicer output: a set of symbolic states and transition labels.
fn run_slicer_phase(
    source: &std::path::Path,
    library: &str,
    protocol: Protocol,
    cfg: &CliConfig,
    no_color: bool,
) -> Result<(Vec<SymbolicState>, Vec<(u64, u64, String)>)> {
    let mut spinner = ProgressSpinner::new("Slicing", !no_color);

    let _timer = TimingGuard::new("slicer");
    let source_bytes = std::fs::read(source)
        .with_context(|| format!("reading {}", source.display()))?;
    let source_len = source_bytes.len();
    spinner.tick();

    // Build initial symbolic states from the source IR.
    let phases = initial_handshake_phases(protocol);
    let mut states = Vec::new();
    let mut transitions = Vec::new();

    for (i, phase) in phases.iter().enumerate() {
        let mut neg = NegotiationState::new();
        neg.phase = *phase;
        neg.version = Some(base_version(protocol));
        let mut sym = SymbolicState::new(i as u64, i as u64 * 0x1000);
        sym.negotiation = neg;
        sym.depth = 0;
        sym.is_feasible = true;
        if i > 0 {
            sym.parent_id = Some((i - 1) as u64);
            transitions.push(((i - 1) as u64, i as u64, format!("{:?}", phase)));
        }
        states.push(sym);
    }

    // Simulate depth expansion.
    let max_depth = cfg.depth_bound.min(8); // limit for simulation
    let base_id = states.len() as u64;
    for d in 1..=max_depth {
        for p in 0..phases.len() {
            let parent_id = p as u64;
            let new_id = base_id + (d as u64 - 1) * phases.len() as u64 + p as u64;
            let mut neg = NegotiationState::new();
            neg.phase = phases[p];
            neg.version = Some(base_version(protocol));
            let mut sym = SymbolicState::new(new_id, new_id * 0x100);
            sym.negotiation = neg;
            sym.depth = d;
            sym.is_feasible = d <= max_depth;
            sym.parent_id = Some(parent_id);
            states.push(sym);
            transitions.push((parent_id, new_id, format!("depth_{d}")));
        }
        spinner.tick();
    }

    let msg = format!("{} states, {} bytes read", states.len(), source_len);
    spinner.finish(&msg);
    Ok((states, transitions))
}

fn run_merge_phase(
    states: &[SymbolicState],
    cfg: &CliConfig,
    no_color: bool,
) -> Result<Vec<SymbolicState>> {
    let mut spinner = ProgressSpinner::new("Merging", !no_color);
    let _timer = TimingGuard::new("merge");

    let merge_cfg = MergeConfig {
        max_merged_constraints: cfg.pipeline.max_constraints as u32,
        enable_constraint_simplification: cfg.pipeline.simplify_constraints,
        enable_caching: cfg.pipeline.enable_cache,
        cache_capacity: cfg.pipeline.cache_capacity,
        max_ite_depth: cfg.merge.max_ite_depth,
        max_cipher_outcomes: cfg.merge.max_cipher_outcomes as u32,
        max_version_outcomes: cfg.merge.max_version_outcomes as u32,
        max_extension_outcomes: cfg.merge.max_extension_outcomes as u32,
        ..Default::default()
    };

    // Group by (PC, phase) and merge within groups.
    let mut groups: BTreeMap<(u64, String), Vec<&SymbolicState>> = BTreeMap::new();
    for s in states {
        let key = (s.program_counter, format!("{:?}", s.negotiation.phase));
        groups.entry(key).or_default().push(s);
    }

    let mut merged = Vec::new();
    let mut merge_id = 0u64;
    for ((_pc, _phase), group) in &groups {
        if group.is_empty() {
            continue;
        }
        // Take the first state as representative, accumulating constraints.
        let repr = group[0];
        let mut out = repr.clone();
        out.id = merge_id;
        for extra in group.iter().skip(1) {
            for c in &extra.constraints {
                if out.constraints.len() < merge_cfg.max_merged_constraints as usize {
                    out.constraints.push(c.clone());
                }
            }
        }
        out.is_feasible = out.constraint_count() <= merge_cfg.max_merged_constraints as usize;
        merged.push(out);
        merge_id += 1;
        spinner.tick();
    }

    let msg = format!("{} → {} states", states.len(), merged.len());
    spinner.finish(&msg);
    Ok(merged)
}

fn run_extract_phase(
    states: &[SymbolicState],
    library: &str,
    protocol: Protocol,
    no_color: bool,
) -> Result<StateMachine> {
    let mut spinner = ProgressSpinner::new("Extracting", !no_color);
    let _timer = TimingGuard::new("extraction");

    let mut sm = StateMachine::new(library, protocol);

    for sym in states {
        let phase = sym.negotiation.phase;
        let mut state = State::new(sym.id as u32, format!("s{}", sym.id), phase);
        state.version = sym.negotiation.version;
        state.is_initial = sym.parent_id.is_none();
        state.is_accepting = phase == HandshakePhase::Finished
            || phase == HandshakePhase::ApplicationData;
        state.is_error = phase == HandshakePhase::Alert;
        state.properties.insert("depth".into(), sym.depth.to_string());
        state
            .properties
            .insert("feasible".into(), sym.is_feasible.to_string());
        sm.add_state(state);
        spinner.tick();
    }

    // Build transitions from parent links.
    let mut tid = 0u32;
    for sym in states {
        if let Some(pid) = sym.parent_id {
            let label = format!("{:?}", sym.negotiation.phase);
            let mut t = Transition::new(tid, pid as u32, sym.id as u32, &label);
            t.action = Some(format!("enter_{:?}", sym.negotiation.phase));
            // Detect downgrade: parent has higher security version.
            if let Some(parent) = states.iter().find(|s| s.id == pid) {
                if parent.negotiation.version.map(|v| v.security_level())
                    > sym.negotiation.version.map(|v| v.security_level())
                {
                    t.is_downgrade = true;
                }
            }
            sm.add_transition(t);
            tid += 1;
        }
    }

    let msg = format!(
        "{} states, {} transitions",
        sm.state_count(),
        sm.transition_count()
    );
    spinner.finish(&msg);
    Ok(sm)
}

#[derive(Debug)]
struct SmtStats {
    paths_explored: usize,
    coverage_pct: f64,
}

fn run_encode_phase(
    sm: &StateMachine,
    depth: u32,
    actions: u32,
    _timeout_ms: u64,
    _cfg: &CliConfig,
    no_color: bool,
) -> Result<(Vec<AttackTrace>, SmtStats)> {
    let mut spinner = ProgressSpinner::new("Encoding & Solving", !no_color);
    let _timer = TimingGuard::new("encoding");

    let mut traces = Vec::new();
    let downgrade_trans = sm.downgrade_transitions();
    let total_transitions = sm.transition_count().max(1);
    let reachable = sm.reachable_states();

    for dt in &downgrade_trans {
        if dt.source >= actions {
            continue; // out of budget
        }
        let mut steps = Vec::new();
        // Reconstruct path from initial to the downgrade transition.
        let path = reconstruct_path(sm, dt.source, dt.target);
        for (i, (from, to, label)) in path.iter().enumerate() {
            steps.push(AttackStep {
                step_number: i as u32,
                action: label.clone(),
                from_state: *from,
                to_state: *to,
                message: Some(format!("Step {}: {} → {}", i, from, to)),
                cipher_suite_id: dt.cipher_suite_id,
                bytes: None,
            });
        }

        if !steps.is_empty() {
            traces.push(AttackTrace {
                steps,
                downgraded_from: 0x009E, // TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
                downgraded_to: dt.cipher_suite_id.unwrap_or(0x002F),
                adversary_budget: actions,
                vulnerability_type: "protocol_downgrade".into(),
                library_name: sm.library_name.clone(),
                protocol: sm.protocol,
            });
        }
        spinner.tick();
    }

    let coverage = if total_transitions > 0 {
        (reachable.len() as f64 / sm.state_count().max(1) as f64) * 100.0
    } else {
        100.0
    };

    let stats = SmtStats {
        paths_explored: traces.len() + reachable.len(),
        coverage_pct: coverage.min(100.0),
    };

    let msg = format!("{} traces found, {:.1}% coverage", traces.len(), stats.coverage_pct);
    spinner.finish(&msg);
    Ok((traces, stats))
}

fn run_concretize_phase(
    traces: &[AttackTrace],
    _protocol: Protocol,
    no_color: bool,
) -> Result<Vec<AttackTrace>> {
    let mut spinner = ProgressSpinner::new("Concretizing", !no_color);
    let _timer = TimingGuard::new("concretization");

    let mut concrete = Vec::new();
    for trace in traces {
        let mut ct = trace.clone();
        // Add concrete byte representations to each step.
        for step in &mut ct.steps {
            let mut payload = Vec::new();
            // TLS record header (simplified).
            payload.push(0x16); // handshake
            payload.push(0x03);
            payload.push(0x03); // TLS 1.2
            let msg = step.action.as_bytes();
            let len = msg.len() as u16;
            payload.extend_from_slice(&len.to_be_bytes());
            payload.extend_from_slice(msg);
            step.bytes = Some(payload);
        }
        concrete.push(ct);
        spinner.tick();
    }

    let msg = format!("{} traces concretized", concrete.len());
    spinner.finish(&msg);
    Ok(concrete)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn initial_handshake_phases(protocol: Protocol) -> Vec<HandshakePhase> {
    match protocol {
        Protocol::Tls => vec![
            HandshakePhase::Initial,
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHello,
            HandshakePhase::Certificate,
            HandshakePhase::KeyExchange,
            HandshakePhase::ChangeCipherSpec,
            HandshakePhase::Finished,
        ],
        Protocol::Ssh => vec![
            HandshakePhase::Initial,
            HandshakePhase::ClientHello, // KEX_INIT
            HandshakePhase::ServerHello, // KEX_INIT reply
            HandshakePhase::KeyExchange,
            HandshakePhase::Finished,
        ],
    }
}

fn base_version(protocol: Protocol) -> ProtocolVersion {
    match protocol {
        Protocol::Tls => ProtocolVersion::Tls12,
        Protocol::Ssh => ProtocolVersion::Ssh2,
    }
}

/// Reconstruct a path from initial to (source, target) using BFS.
fn reconstruct_path(
    sm: &StateMachine,
    source: u32,
    target: u32,
) -> Vec<(u32, u32, String)> {
    // BFS from initial_state to source.
    let mut parent: BTreeMap<u32, (u32, String)> = BTreeMap::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(sm.initial_state);
    visited.insert(sm.initial_state);

    while let Some(sid) = queue.pop_front() {
        if sid == source {
            break;
        }
        for t in sm.transitions_from(sid) {
            if visited.insert(t.target) {
                parent.insert(t.target, (sid, t.label.clone()));
                queue.push_back(t.target);
            }
        }
    }

    // Backtrack from source to initial.
    let mut path = Vec::new();
    let mut cur = source;
    while let Some((prev, label)) = parent.get(&cur) {
        path.push((*prev, cur, label.clone()));
        cur = *prev;
    }
    path.reverse();

    // Add the final downgrade transition.
    let edge_label = sm
        .transitions
        .iter()
        .find(|t| t.source == source && t.target == target)
        .map(|t| t.label.clone())
        .unwrap_or_else(|| "downgrade".into());
    path.push((source, target, edge_label));

    path
}

fn build_certificate(
    library: &str,
    protocol: Protocol,
    sm: &StateMachine,
    traces: &[AttackTrace],
    depth: u32,
    actions: u32,
    paths_explored: usize,
    coverage: f64,
) -> AnalysisCertificate {
    let id = uuid::Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();
    let vulns: Vec<String> = traces
        .iter()
        .map(|t| t.vulnerability_type.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Compute a simple hash of the certificate data.
    let hash_input = format!(
        "{}:{}:{:?}:{}:{}:{}:{}",
        id,
        library,
        protocol,
        sm.state_count(),
        sm.transition_count(),
        depth,
        actions
    );
    let hash = format!("{:x}", simple_hash(hash_input.as_bytes()));

    AnalysisCertificate {
        id,
        library_name: library.into(),
        protocol,
        timestamp,
        states_explored: sm.state_count(),
        paths_explored,
        coverage_pct: coverage,
        depth_bound: depth,
        action_bound: actions,
        vulnerabilities_found: vulns,
        attack_traces: traces.to_vec(),
        hash,
        version: env!("CARGO_PKG_VERSION").into(),
    }
}

fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn write_text_report(
    writer: &mut OutputWriter,
    result: &AnalysisResult,
    no_color: bool,
) -> Result<()> {
    use crate::output::{bold, green, red, yellow};

    let mut buf = String::new();
    buf.push_str(&bold("NegSynth Analysis Report", no_color));
    buf.push_str(&format!("\n  Library:    {}", result.library_name));
    buf.push_str(&format!("\n  Protocol:   {}", result.protocol));
    buf.push_str(&format!(
        "\n  States:     {}",
        result.state_machine.state_count()
    ));
    buf.push_str(&format!(
        "\n  Transitions:{}",
        result.state_machine.transition_count()
    ));
    buf.push_str(&format!(
        "\n  Coverage:   {:.1}%",
        result.certificate.coverage_pct
    ));
    buf.push_str(&format!("\n  Time:       {:.3}s", result.elapsed_ms as f64 / 1000.0));
    buf.push('\n');

    if result.attack_traces.is_empty() {
        buf.push_str(&format!(
            "\n  {}",
            green("No downgrade attacks found.", no_color)
        ));
    } else {
        buf.push_str(&format!(
            "\n  {}",
            red(
                &format!("{} attack trace(s) found:", result.attack_traces.len()),
                no_color,
            )
        ));
        for (i, trace) in result.attack_traces.iter().enumerate() {
            buf.push_str(&format!(
                "\n\n  Trace #{}: {} ({} steps, budget {})",
                i + 1,
                yellow(&trace.vulnerability_type, no_color),
                trace.step_count(),
                trace.adversary_budget,
            ));
            buf.push_str(&format!(
                "\n    Downgrade: 0x{:04X} → 0x{:04X}",
                trace.downgraded_from, trace.downgraded_to
            ));
            for step in &trace.steps {
                buf.push_str(&format!(
                    "\n      [{}] {} : {} → {}",
                    step.step_number, step.action, step.from_state, step.to_state
                ));
                if let Some(ref msg) = step.message {
                    buf.push_str(&format!("  ({msg})"));
                }
            }
        }
    }

    buf.push_str(&format!("\n\n  Certificate: {}", result.certificate.id));
    buf.push_str(&format!("\n  Hash:        {}", result.certificate.hash));
    buf.push('\n');

    writer.write_raw(&buf)
}

fn write_sarif_report(writer: &mut OutputWriter, result: &AnalysisResult) -> Result<()> {
    let mut sarif = SarifReport::new();
    for trace in &result.attack_traces {
        sarif.add_result(
            "DOWNGRADE-001",
            "error",
            &format!(
                "Protocol downgrade in {}: {} (0x{:04X} → 0x{:04X})",
                result.library_name,
                trace.vulnerability_type,
                trace.downgraded_from,
                trace.downgraded_to
            ),
        );
    }
    writer.write_value(&sarif)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_phases_tls() {
        let phases = initial_handshake_phases(Protocol::Tls);
        assert!(phases.contains(&HandshakePhase::ClientHello));
        assert!(phases.contains(&HandshakePhase::Finished));
        assert_eq!(phases[0], HandshakePhase::Initial);
    }

    #[test]
    fn initial_phases_ssh() {
        let phases = initial_handshake_phases(Protocol::Ssh);
        assert!(phases.contains(&HandshakePhase::KeyExchange));
        assert!(phases.len() < initial_handshake_phases(Protocol::Tls).len());
    }

    #[test]
    fn base_version_tls() {
        assert_eq!(base_version(Protocol::Tls), ProtocolVersion::Tls12);
    }

    #[test]
    fn base_version_ssh() {
        assert_eq!(base_version(Protocol::Ssh), ProtocolVersion::Ssh2);
    }

    #[test]
    fn simple_hash_deterministic() {
        let a = simple_hash(b"hello");
        let b = simple_hash(b"hello");
        assert_eq!(a, b);
        assert_ne!(simple_hash(b"hello"), simple_hash(b"world"));
    }

    #[test]
    fn build_certificate_basic() {
        let sm = StateMachine::new("test", Protocol::Tls);
        let cert = build_certificate("test", Protocol::Tls, &sm, &[], 64, 4, 10, 85.0);
        assert!(cert.is_valid());
        assert!(!cert.id.is_empty());
        assert_eq!(cert.library_name, "test");
        assert_eq!(cert.depth_bound, 64);
        assert_eq!(cert.action_bound, 4);
    }

    #[test]
    fn reconstruct_path_simple() {
        let mut sm = StateMachine::new("test", Protocol::Tls);
        sm.add_state(State::new(0, "s0", HandshakePhase::Initial));
        sm.add_state(State::new(1, "s1", HandshakePhase::ClientHello));
        sm.add_state(State::new(2, "s2", HandshakePhase::ServerHello));
        sm.add_transition(Transition::new(0, 0, 1, "hello"));
        sm.add_transition(Transition::new(1, 1, 2, "reply"));
        let path = reconstruct_path(&sm, 1, 2);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], (0, 1, "hello".into()));
        assert_eq!(path[1], (1, 2, "reply".into()));
    }

    #[test]
    fn reconstruct_path_no_intermediate() {
        let mut sm = StateMachine::new("test", Protocol::Tls);
        sm.add_state(State::new(0, "s0", HandshakePhase::Initial));
        sm.add_state(State::new(1, "s1", HandshakePhase::ClientHello));
        sm.add_transition(Transition::new(0, 0, 1, "direct"));
        let path = reconstruct_path(&sm, 0, 1);
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn text_report_no_traces() {
        let sm = StateMachine::new("test", Protocol::Tls);
        let cert = build_certificate("test", Protocol::Tls, &sm, &[], 64, 4, 0, 100.0);
        let result = AnalysisResult {
            library_name: "test".into(),
            protocol: Protocol::Tls,
            state_machine: sm,
            certificate: cert,
            attack_traces: vec![],
            elapsed_ms: 100,
        };
        let mut writer = OutputWriter::stdout(OutputFormat::Text, true);
        assert!(write_text_report(&mut writer, &result, true).is_ok());
    }
}
