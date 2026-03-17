//! Display and formatting implementations for all major types.
//!
//! Provides human-readable, debug, and summary formatting for types
//! across the NegSynth pipeline.

use crate::adversary::{
    AdversaryAction, AdversaryBudget, AdversaryTrace, BoundedDYAdversary, DowngradeInfo,
    KnowledgeSet, MessageTerm,
};
use crate::certificate::{AnalysisResult, AttackTrace, BoundsSpec, Certificate};
use crate::config::{AnalysisConfig, MergeStrategy};
use crate::graph::{BisimulationRelation, QuotientGraph, StateGraph, StateId};
use crate::metrics::{AnalysisMetrics, CoverageMetrics, MergeStatistics, MetricReport};
use crate::protocol::{
    CipherSuite, HandshakePhase, NegotiationLTS, NegotiationOutcome, NegotiationState,
    ProtocolVersion, SecurityLevel,
};
use crate::smt::{SmtExpr, SmtFormula, SmtModel, SmtResult, SmtSort};
use crate::symbolic::{ExecutionTree, PathConstraint, SymbolicState, SymbolicValue};
use std::fmt;

// ── Summary trait ────────────────────────────────────────────────────────

/// Trait for producing compact one-line summaries.
pub trait Summary {
    fn summary(&self) -> String;
}

// ── Protocol summaries ───────────────────────────────────────────────────

impl Summary for CipherSuite {
    fn summary(&self) -> String {
        format!(
            "{}(0x{:04X}) [{}] {}b {}+{}",
            self.name,
            self.iana_id,
            self.security_level,
            self.encryption.key_bits(),
            self.key_exchange,
            self.auth,
        )
    }
}

impl Summary for ProtocolVersion {
    fn summary(&self) -> String {
        format!("{} [{}]", self, self.security_level())
    }
}

impl Summary for NegotiationState {
    fn summary(&self) -> String {
        let cipher_info = match &self.selected_cipher {
            Some(c) => format!("selected={}", c.name),
            None => format!("{} offered", self.offered_ciphers.len()),
        };
        let version_info = match &self.version {
            Some(v) => format!("{}", v),
            None => "no version".into(),
        };
        format!(
            "{}: {} | {} | {} exts",
            self.phase,
            cipher_info,
            version_info,
            self.extensions.len()
        )
    }
}

impl Summary for NegotiationOutcome {
    fn summary(&self) -> String {
        format!(
            "{} + {} [{}]{}",
            self.version,
            self.selected_cipher.name,
            self.security_level(),
            if self.session_resumed { " (resumed)" } else { "" }
        )
    }
}

impl Summary for NegotiationLTS {
    fn summary(&self) -> String {
        let (states, transitions) = self.size();
        let adv = self.adversary_transitions().len();
        format!(
            "LTS: {} states, {} transitions ({} adversary)",
            states, transitions, adv
        )
    }
}

// ── Adversary summaries ──────────────────────────────────────────────────

impl Summary for MessageTerm {
    fn summary(&self) -> String {
        match self {
            MessageTerm::Nonce(n) => format!("nonce({})", n),
            MessageTerm::Key(_) => "key(...)".into(),
            MessageTerm::CipherId(id) => format!("cipher(0x{:04X})", id),
            MessageTerm::Version(maj, min) => format!("v{}.{}", maj, min),
            MessageTerm::Bytes(b) => format!("bytes({})", b.len()),
            MessageTerm::Encrypted { .. } => format!("enc(depth={})", self.depth()),
            MessageTerm::Mac { .. } => format!("mac(depth={})", self.depth()),
            MessageTerm::Hash(_) => format!("hash(depth={})", self.depth()),
            MessageTerm::Pair(_, _) => format!("pair(size={})", self.size()),
            MessageTerm::Record { content_type, .. } => format!("record(type={})", content_type),
            MessageTerm::Packet {
                source,
                destination,
                ..
            } => format!("pkt({}→{})", source, destination),
            MessageTerm::Variable(name) => format!("${}", name),
        }
    }
}

impl Summary for AdversaryAction {
    fn summary(&self) -> String {
        match self {
            AdversaryAction::Intercept { message } => {
                format!("intercept({})", message.summary())
            }
            AdversaryAction::Inject { message } => {
                format!("inject({})", message.summary())
            }
            AdversaryAction::Drop { message } => {
                format!("drop({})", message.summary())
            }
            AdversaryAction::Modify {
                original, modified, ..
            } => format!("modify({}→{})", original.summary(), modified.summary()),
            AdversaryAction::Replay { message } => {
                format!("replay({})", message.summary())
            }
            AdversaryAction::Reorder { indices } => format!("reorder({:?})", indices),
        }
    }
}

impl Summary for AdversaryTrace {
    fn summary(&self) -> String {
        format!(
            "trace: {} actions, cost {}/{}, {} active",
            self.len(),
            self.total_cost(),
            self.budget.action_bound,
            self.active_action_count()
        )
    }
}

impl Summary for BoundedDYAdversary {
    fn summary(&self) -> String {
        format!(
            "{:?} adversary: {} known terms, {}/{} budget, {} actions",
            self.position,
            self.knowledge.size(),
            self.trace.total_cost(),
            self.budget.action_bound,
            self.trace.len()
        )
    }
}

impl Summary for KnowledgeSet {
    fn summary(&self) -> String {
        format!(
            "knowledge: {} base + {} deduced = {} total",
            self.base_size(),
            self.size() - self.base_size(),
            self.size()
        )
    }
}

impl Summary for DowngradeInfo {
    fn summary(&self) -> String {
        format!(
            "{:?}: {} → {} ({})",
            self.kind, self.from_level, self.to_level, self.description
        )
    }
}

// ── Symbolic summaries ───────────────────────────────────────────────────

impl Summary for SymbolicValue {
    fn summary(&self) -> String {
        if self.is_concrete() {
            format!("concrete({})", self)
        } else {
            format!(
                "symbolic(depth={}, nodes={}, vars={})",
                self.depth(),
                self.node_count(),
                self.free_variables().len()
            )
        }
    }
}

impl Summary for PathConstraint {
    fn summary(&self) -> String {
        let vars = self.free_variables();
        format!(
            "PC: {} conditions, {} vars{}",
            self.len(),
            vars.len(),
            if self.is_trivially_unsat() {
                " [UNSAT]"
            } else {
                ""
            }
        )
    }
}

impl Summary for SymbolicState {
    fn summary(&self) -> String {
        format!(
            "state#{} @{:#x}: {} regs, {} constraints, depth={}",
            self.id,
            self.program_counter,
            self.registers.len(),
            self.path_constraint.len(),
            self.depth
        )
    }
}

impl Summary for ExecutionTree {
    fn summary(&self) -> String {
        let active = self.active_leaves().len();
        format!(
            "tree: {} nodes, depth={}, {} active leaves",
            self.total_nodes(),
            self.depth(),
            active
        )
    }
}

// ── SMT summaries ────────────────────────────────────────────────────────

impl Summary for SmtSort {
    fn summary(&self) -> String {
        self.to_smtlib()
    }
}

impl Summary for SmtExpr {
    fn summary(&self) -> String {
        let nc = self.node_count();
        let nv = self.free_variables().len();
        if nc <= 5 {
            self.to_smtlib()
        } else {
            format!("expr({} nodes, {} vars)", nc, nv)
        }
    }
}

impl Summary for SmtFormula {
    fn summary(&self) -> String {
        format!(
            "formula: {} decls, {} assertions, {} nodes, logic={:?}",
            self.declarations.len(),
            self.assertion_count(),
            self.total_nodes(),
            self.logic
        )
    }
}

impl Summary for SmtResult {
    fn summary(&self) -> String {
        format!("{}", self)
    }
}

impl Summary for SmtModel {
    fn summary(&self) -> String {
        format!("model: {} assignments", self.len())
    }
}

// ── Certificate summaries ────────────────────────────────────────────────

impl Summary for Certificate {
    fn summary(&self) -> String {
        format!(
            "cert({}): {} @ k={},n={}",
            &self.id[..8.min(self.id.len())],
            self.library,
            self.bounds.depth_k,
            self.bounds.actions_n
        )
    }
}

impl Summary for AttackTrace {
    fn summary(&self) -> String {
        format!(
            "attack({:?}): {} → {}, {} steps, {} adv actions",
            self.severity(),
            self.downgrade.from_level,
            self.downgrade.to_level,
            self.total_steps(),
            self.adversary_action_count(),
        )
    }
}

impl Summary for AnalysisResult {
    fn summary(&self) -> String {
        match self {
            AnalysisResult::AttackFound(t) => format!("ATTACK: {}", Summary::summary(t)),
            AnalysisResult::CertifiedSafe(c) => format!("SAFE: {}", Summary::summary(c)),
            AnalysisResult::Inconclusive(r) => {
                format!("INCONCLUSIVE: {} ({:.1}%)", r.reason, r.partial_coverage)
            }
        }
    }
}

// ── Graph summaries ──────────────────────────────────────────────────────

impl Summary for StateGraph {
    fn summary(&self) -> String {
        format!("{}", self)
    }
}

impl Summary for BisimulationRelation {
    fn summary(&self) -> String {
        format!("{}", self)
    }
}

impl Summary for QuotientGraph {
    fn summary(&self) -> String {
        format!("{}", self)
    }
}

// ── Metric summaries ─────────────────────────────────────────────────────

impl Summary for AnalysisMetrics {
    fn summary(&self) -> String {
        format!(
            "explored: {} states, {} paths ({} merged), {} solver calls, {} attacks",
            self.states_explored,
            self.paths_explored,
            self.paths_merged,
            self.solver_calls,
            self.attacks_found
        )
    }
}

impl Summary for CoverageMetrics {
    fn summary(&self) -> String {
        format!(
            "coverage: {:.1}% states, {:.1}% transitions, {:.1}% paths",
            self.state_coverage_pct(),
            self.transition_coverage_pct(),
            self.path_coverage_pct()
        )
    }
}

impl Summary for MergeStatistics {
    fn summary(&self) -> String {
        format!(
            "merges: {}/{} success ({:.0}%), {:.0}% reduction",
            self.merge_successes,
            self.merge_attempts,
            self.success_rate() * 100.0,
            self.reduction_pct()
        )
    }
}

impl Summary for MetricReport {
    fn summary(&self) -> String {
        format!(
            "{} | {} | {}",
            self.analysis.summary(),
            self.coverage.summary(),
            self.merge.summary()
        )
    }
}

// ── Table formatting ─────────────────────────────────────────────────────

/// Format a list of cipher suites as a table.
pub fn format_cipher_suite_table(suites: &[CipherSuite]) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{:<6} {:<55} {:<8} {:<6} {:<20} {:<10} {:<8}",
        "ID", "Name", "KX", "Auth", "Enc", "MAC", "Level"
    ));
    lines.push("-".repeat(115));
    for cs in suites {
        lines.push(format!(
            "0x{:04X} {:<55} {:<8} {:<6} {:<20} {:<10} {:<8}",
            cs.iana_id,
            cs.name,
            format!("{}", cs.key_exchange),
            format!("{}", cs.auth),
            format!("{}", cs.encryption),
            format!("{}", cs.mac),
            format!("{}", cs.security_level),
        ));
    }
    lines.join("\n")
}

/// Format an attack trace as a detailed report.
pub fn format_attack_report(attack: &AttackTrace) -> String {
    let mut lines = Vec::new();
    lines.push("╔══════════════════════════════════════════════════════╗".into());
    lines.push(format!(
        "║  DOWNGRADE ATTACK FOUND ({})                  ║",
        attack.severity()
    ));
    lines.push("╚══════════════════════════════════════════════════════╝".into());
    lines.push(String::new());
    lines.push(format!("Downgrade: {}", attack.downgrade));
    lines.push(format!("Expected:  {}", attack.expected_outcome));
    lines.push(format!("Actual:    {}", attack.actual_outcome));
    lines.push(String::new());
    lines.push("Protocol Trace:".into());
    for step in &attack.protocol_steps {
        let actor = match step.actor {
            crate::certificate::StepActor::Client => "C",
            crate::certificate::StepActor::Server => "S",
            crate::certificate::StepActor::Adversary => "A",
        };
        lines.push(format!(
            "  [{:>3}] [{}] {}: {}",
            step.step_number, actor, step.action, step.data_summary
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "Adversary: {} actions, cost {}",
        attack.adversary_trace.len(),
        attack.adversary_trace.total_cost()
    ));
    lines.join("\n")
}

/// Format a metric report as a compact dashboard.
pub fn format_metrics_dashboard(report: &MetricReport) -> String {
    let mut lines = Vec::new();
    lines.push("┌─────────────────────────────────────────┐".into());
    lines.push("│       NegSynth Analysis Dashboard       │".into());
    lines.push("├─────────────────────────────────────────┤".into());
    lines.push(format!(
        "│ States:    {:>8}  Paths:   {:>8} │",
        report.analysis.states_explored, report.analysis.paths_explored
    ));
    lines.push(format!(
        "│ Merged:    {:>8}  Pruned:  {:>8} │",
        report.analysis.paths_merged, report.analysis.paths_pruned
    ));
    lines.push(format!(
        "│ Solver:    {:>8}  Attacks: {:>8} │",
        report.analysis.solver_calls, report.analysis.attacks_found
    ));
    lines.push("├─────────────────────────────────────────┤".into());
    lines.push(format!(
        "│ Coverage:  {:>7.1}%  Merge:   {:>7.1}% │",
        report.coverage.overall_score(),
        report.merge.success_rate() * 100.0
    ));
    lines.push(format!(
        "│ Time:      {:>5}ms  Memory:  {:>5}KB │",
        report.performance.wall_time_ms, report.performance.peak_memory_kb
    ));
    lines.push("└─────────────────────────────────────────┘".into());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::CipherSuiteRegistry;

    #[test]
    fn test_cipher_suite_summary() {
        let cs = CipherSuiteRegistry::lookup(0x1301).unwrap();
        let s = cs.summary();
        assert!(s.contains("TLS_AES_128_GCM_SHA256"));
        assert!(s.contains("HIGH"));
    }

    #[test]
    fn test_protocol_version_summary() {
        let v = ProtocolVersion::tls13();
        let s = v.summary();
        assert!(s.contains("TLSv1.3"));
        assert!(s.contains("HIGH"));
    }

    #[test]
    fn test_symbolic_value_summary() {
        let concrete = SymbolicValue::int_const(42);
        assert!(concrete.summary().contains("concrete"));

        let sym = SymbolicValue::var("x", crate::symbolic::SymSort::Int);
        assert!(sym.summary().contains("symbolic"));
    }

    #[test]
    fn test_path_constraint_summary() {
        let pc = PathConstraint::empty();
        let s = pc.summary();
        assert!(s.contains("0 conditions"));
    }

    #[test]
    fn test_cipher_suite_table() {
        let suites = vec![
            CipherSuiteRegistry::lookup(0x1301).unwrap(),
            CipherSuiteRegistry::lookup(0x0005).unwrap(),
        ];
        let table = format_cipher_suite_table(&suites);
        assert!(table.contains("TLS_AES_128_GCM_SHA256"));
        assert!(table.contains("RC4_128_SHA"));
    }

    #[test]
    fn test_metrics_dashboard() {
        let report = MetricReport::new(
            AnalysisMetrics::new(),
            CoverageMetrics::new(),
            crate::metrics::PerformanceMetrics::new(),
            MergeStatistics::new(),
        );
        let dashboard = format_metrics_dashboard(&report);
        assert!(dashboard.contains("NegSynth"));
        assert!(dashboard.contains("States"));
    }

    #[test]
    fn test_analysis_result_summary() {
        let result = AnalysisResult::Inconclusive(crate::certificate::InconclusiveReason {
            reason: "timeout".into(),
            partial_coverage: 50.0,
            suggestion: "increase budget".into(),
        });
        let s = result.summary();
        assert!(s.contains("INCONCLUSIVE"));
        assert!(s.contains("50.0%"));
    }

    #[test]
    fn test_smt_expr_summary() {
        let simple = SmtExpr::BoolConst(true);
        let s = simple.summary();
        assert_eq!(s, "true");

        let complex = SmtExpr::And(vec![
            SmtExpr::bool_var("a"),
            SmtExpr::bool_var("b"),
            SmtExpr::bool_var("c"),
            SmtExpr::bool_var("d"),
            SmtExpr::bool_var("e"),
            SmtExpr::bool_var("f"),
        ]);
        let s = complex.summary();
        assert!(s.contains("nodes"));
    }

    #[test]
    fn test_knowledge_set_summary() {
        let ks = KnowledgeSet::new();
        let s = ks.summary();
        assert!(s.contains("0 base"));
    }

    #[test]
    fn test_adversary_budget_display() {
        let b = AdversaryBudget::new(5, 20);
        let s = format!("{}", b);
        assert!(s.contains("k=5"));
        assert!(s.contains("n=20"));
    }
}
