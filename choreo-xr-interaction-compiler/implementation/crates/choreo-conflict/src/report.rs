//! Consolidated conflict reporting.
//!
//! Collects results from all analysis passes and produces unified reports
//! in text and JSON formats with severity-based filtering and summaries.

use crate::deadlock::{Deadlock, DeadlockClassification};
use crate::interference::{InterferenceGraph, InterferenceKind};
use crate::race::{RaceCondition, RaceSeverity, TemporalRace};
use crate::unreachable::{StateClass, UnreachableReport};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Severity (unified)
// ---------------------------------------------------------------------------

/// Unified severity level across all analysis types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConflictEntry
// ---------------------------------------------------------------------------

/// A single finding in the conflict report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictEntry {
    pub id: String,
    pub category: ConflictCategory,
    pub severity: Severity,
    pub message: String,
    pub details: String,
    pub affected_states: Vec<u32>,
    pub affected_transitions: Vec<u32>,
}

/// Category of a conflict finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictCategory {
    Deadlock,
    Livelock,
    Race,
    TemporalRace,
    Unreachable,
    DeadTransition,
    DeadGuard,
    Interference,
}

impl fmt::Display for ConflictCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deadlock => write!(f, "deadlock"),
            Self::Livelock => write!(f, "livelock"),
            Self::Race => write!(f, "race"),
            Self::TemporalRace => write!(f, "temporal-race"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::DeadTransition => write!(f, "dead-transition"),
            Self::DeadGuard => write!(f, "dead-guard"),
            Self::Interference => write!(f, "interference"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConflictReport
// ---------------------------------------------------------------------------

/// Full conflict analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub automaton_name: String,
    pub entries: Vec<ConflictEntry>,
}

impl ConflictReport {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            automaton_name: name.into(),
            entries: Vec::new(),
        }
    }

    pub fn total(&self) -> usize {
        self.entries.len()
    }

    pub fn count_by_severity(&self, sev: Severity) -> usize {
        self.entries.iter().filter(|e| e.severity == sev).count()
    }

    pub fn count_by_category(&self, cat: ConflictCategory) -> usize {
        self.entries.iter().filter(|e| e.category == cat).count()
    }

    /// Filter entries to those at or above the given severity.
    pub fn filtered(&self, min_severity: Severity) -> ConflictReport {
        ConflictReport {
            automaton_name: self.automaton_name.clone(),
            entries: self
                .entries
                .iter()
                .filter(|e| e.severity >= min_severity)
                .cloned()
                .collect(),
        }
    }

    /// Generate a summary with counts per category and severity.
    pub fn summary(&self) -> ReportSummary {
        let mut by_category: HashMap<ConflictCategory, usize> = HashMap::new();
        let mut by_severity: HashMap<Severity, usize> = HashMap::new();

        for entry in &self.entries {
            *by_category.entry(entry.category).or_default() += 1;
            *by_severity.entry(entry.severity).or_default() += 1;
        }

        ReportSummary {
            total: self.entries.len(),
            by_category,
            by_severity,
        }
    }

    /// Format the report as plain text.
    pub fn format_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Conflict Report for '{}' ===\n",
            self.automaton_name
        ));
        let summary = self.summary();
        out.push_str(&format!("Total findings: {}\n", summary.total));

        for (sev, count) in &summary.by_severity {
            out.push_str(&format!("  {}: {}\n", sev, count));
        }
        out.push('\n');

        for (cat, count) in &summary.by_category {
            out.push_str(&format!("  {}: {}\n", cat, count));
        }
        out.push('\n');

        for entry in &self.entries {
            out.push_str(&format!(
                "[{}] {} ({}): {}\n",
                entry.severity, entry.id, entry.category, entry.message
            ));
            if !entry.details.is_empty() {
                out.push_str(&format!("  Details: {}\n", entry.details));
            }
            if !entry.affected_states.is_empty() {
                out.push_str(&format!("  States: {:?}\n", entry.affected_states));
            }
            if !entry.affected_transitions.is_empty() {
                out.push_str(&format!(
                    "  Transitions: {:?}\n",
                    entry.affected_transitions
                ));
            }
            out.push('\n');
        }
        out
    }

    /// Format the report as JSON.
    pub fn format_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }
}

// ---------------------------------------------------------------------------
// ReportSummary
// ---------------------------------------------------------------------------

/// Summary statistics for a conflict report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total: usize,
    pub by_category: HashMap<ConflictCategory, usize>,
    pub by_severity: HashMap<Severity, usize>,
}

impl ReportSummary {
    pub fn has_critical(&self) -> bool {
        self.by_severity
            .get(&Severity::Critical)
            .copied()
            .unwrap_or(0)
            > 0
    }

    pub fn has_errors(&self) -> bool {
        self.by_severity
            .get(&Severity::Error)
            .copied()
            .unwrap_or(0)
            > 0
    }

    pub fn is_clean(&self) -> bool {
        self.total == 0
    }
}

// ---------------------------------------------------------------------------
// ReportBuilder
// ---------------------------------------------------------------------------

/// Builder for accumulating findings from multiple analysis passes.
#[derive(Debug)]
pub struct ReportBuilder {
    automaton_name: String,
    entries: Vec<ConflictEntry>,
    next_id: usize,
}

impl ReportBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            automaton_name: name.into(),
            entries: Vec::new(),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self, prefix: &str) -> String {
        let id = format!("{}-{}", prefix, self.next_id);
        self.next_id += 1;
        id
    }

    /// Add deadlock findings.
    pub fn add_deadlocks(&mut self, deadlocks: &[Deadlock]) -> &mut Self {
        for dl in deadlocks {
            let severity = match dl.classification {
                DeadlockClassification::CircularWait => {
                    if dl.is_reachable {
                        Severity::Critical
                    } else {
                        Severity::Warning
                    }
                }
                DeadlockClassification::TerminalTrap => {
                    if dl.is_reachable {
                        Severity::Error
                    } else {
                        Severity::Info
                    }
                }
                DeadlockClassification::GuardBlockage => {
                    if dl.is_reachable {
                        Severity::Error
                    } else {
                        Severity::Info
                    }
                }
                DeadlockClassification::ConditionalDeadlock => Severity::Warning,
            };

            let id = self.alloc_id("DL");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::Deadlock,
                severity,
                message: format!(
                    "{} deadlock involving states {:?}",
                    dl.classification, dl.states
                ),
                details: if dl.spatial_conditions.is_empty() {
                    String::new()
                } else {
                    format!("Spatial conditions: {:?}", dl.spatial_conditions)
                },
                affected_states: dl.states.clone(),
                affected_transitions: vec![],
            });
        }
        self
    }

    /// Add livelock findings.
    pub fn add_livelocks(&mut self, livelocks: &[Vec<u32>]) -> &mut Self {
        for ll in livelocks {
            let id = self.alloc_id("LL");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::Livelock,
                severity: Severity::Warning,
                message: format!("Potential livelock in cycle {:?}", ll),
                details: "Cycle does not contain or reach any accepting state".into(),
                affected_states: ll.clone(),
                affected_transitions: vec![],
            });
        }
        self
    }

    /// Add race condition findings.
    pub fn add_races(&mut self, races: &[RaceCondition]) -> &mut Self {
        for rc in races {
            let severity = match rc.severity {
                RaceSeverity::Critical => Severity::Critical,
                RaceSeverity::Warning => Severity::Warning,
                RaceSeverity::Benign => Severity::Info,
            };
            let id = self.alloc_id("RC");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::Race,
                severity,
                message: format!(
                    "Race at state {} between transitions {:?}",
                    rc.state_id, rc.conflicting
                ),
                details: format!(
                    "{}{}",
                    rc.spatial_condition,
                    if rc.resolved_by_priority {
                        " (resolved by priority)"
                    } else {
                        ""
                    }
                ),
                affected_states: vec![rc.state_id],
                affected_transitions: rc.conflicting.clone(),
            });
        }
        self
    }

    /// Add temporal race findings.
    pub fn add_temporal_races(&mut self, races: &[TemporalRace]) -> &mut Self {
        for tr in races {
            let id = self.alloc_id("TR");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::TemporalRace,
                severity: Severity::Warning,
                message: format!(
                    "Temporal race at state {} between t{} and t{}",
                    tr.state_id, tr.transition_a, tr.transition_b
                ),
                details: tr.overlap_description.clone(),
                affected_states: vec![tr.state_id],
                affected_transitions: vec![tr.transition_a, tr.transition_b],
            });
        }
        self
    }

    /// Add unreachable analysis findings.
    pub fn add_unreachable(&mut self, report: &UnreachableReport) -> &mut Self {
        for &sid in &report.unreachable_states {
            let id = self.alloc_id("UR");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::Unreachable,
                severity: Severity::Info,
                message: format!("State {} is unreachable from the initial state", sid),
                details: String::new(),
                affected_states: vec![sid],
                affected_transitions: vec![],
            });
        }
        for &tid in &report.dead_transitions {
            let id = self.alloc_id("DT");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::DeadTransition,
                severity: Severity::Info,
                message: format!("Transition {} is dead (source unreachable)", tid),
                details: String::new(),
                affected_states: vec![],
                affected_transitions: vec![tid],
            });
        }
        for &tid in &report.dead_guard_transitions {
            let id = self.alloc_id("DG");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::DeadGuard,
                severity: Severity::Warning,
                message: format!(
                    "Transition {} has a guard that can never be satisfied",
                    tid
                ),
                details: String::new(),
                affected_states: vec![],
                affected_transitions: vec![tid],
            });
        }
        self
    }

    /// Add interference graph findings.
    pub fn add_interference(&mut self, graph: &InterferenceGraph) -> &mut Self {
        for &(a, b, kind) in &graph.edges {
            if kind == InterferenceKind::None {
                continue;
            }
            let severity = match kind {
                InterferenceKind::ResourceConflict => Severity::Warning,
                InterferenceKind::SpatialConflict => Severity::Info,
                InterferenceKind::TemporalConflict => Severity::Info,
                InterferenceKind::None => continue,
            };
            let id = self.alloc_id("IF");
            self.entries.push(ConflictEntry {
                id,
                category: ConflictCategory::Interference,
                severity,
                message: format!(
                    "{} interference between states {} and {}",
                    kind, a, b
                ),
                details: String::new(),
                affected_states: vec![a, b],
                affected_transitions: vec![],
            });
        }
        self
    }

    /// Build the final report.
    pub fn build(self) -> ConflictReport {
        ConflictReport {
            automaton_name: self.automaton_name,
            entries: self.entries,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_deadlocks() -> Vec<Deadlock> {
        vec![
            Deadlock {
                states: vec![1, 2],
                spatial_conditions: vec!["inside(e1, r1)".into()],
                is_reachable: true,
                trace: vec![0, 1, 2],
                classification: DeadlockClassification::CircularWait,
            },
            Deadlock {
                states: vec![3],
                spatial_conditions: vec![],
                is_reachable: false,
                trace: vec![],
                classification: DeadlockClassification::TerminalTrap,
            },
        ]
    }

    fn sample_races() -> Vec<RaceCondition> {
        vec![RaceCondition {
            state_id: 0,
            conflicting: vec![0, 1],
            spatial_condition: "guards overlap".into(),
            severity: RaceSeverity::Critical,
            same_event: true,
            resolved_by_priority: false,
        }]
    }

    #[test]
    fn builder_accumulates() {
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_deadlocks(&sample_deadlocks());
        builder.add_races(&sample_races());
        let report = builder.build();
        assert_eq!(report.total(), 3); // 2 deadlocks + 1 race
    }

    #[test]
    fn severity_filtering() {
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_deadlocks(&sample_deadlocks());
        builder.add_races(&sample_races());
        let report = builder.build();
        let filtered = report.filtered(Severity::Warning);
        // Unreachable terminal trap is Info, so filtered out
        assert!(filtered.total() < report.total());
    }

    #[test]
    fn summary_counts() {
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_deadlocks(&sample_deadlocks());
        let report = builder.build();
        let summary = report.summary();
        assert_eq!(summary.total, 2);
        assert_eq!(
            summary.by_category.get(&ConflictCategory::Deadlock).copied().unwrap_or(0),
            2
        );
    }

    #[test]
    fn summary_critical_check() {
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_deadlocks(&sample_deadlocks());
        let report = builder.build();
        let summary = report.summary();
        assert!(summary.has_critical());
    }

    #[test]
    fn clean_report() {
        let builder = ReportBuilder::new("clean");
        let report = builder.build();
        assert!(report.summary().is_clean());
    }

    #[test]
    fn format_text_nonempty() {
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_deadlocks(&sample_deadlocks());
        let report = builder.build();
        let text = report.format_text();
        assert!(text.contains("Conflict Report"));
        assert!(text.contains("deadlock"));
    }

    #[test]
    fn format_json_valid() {
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_deadlocks(&sample_deadlocks());
        let report = builder.build();
        let json = report.format_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());
        assert!(parsed["entries"].is_array());
    }

    #[test]
    fn add_unreachable_entries() {
        let ur = UnreachableReport {
            unreachable_states: vec![5, 6],
            dead_transitions: vec![10],
            dead_guard_transitions: vec![11],
            state_classifications: {
                let mut m = HashMap::new();
                m.insert(5, StateClass::Dead);
                m.insert(6, StateClass::Dead);
                m
            },
        };
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_unreachable(&ur);
        let report = builder.build();
        assert_eq!(report.count_by_category(ConflictCategory::Unreachable), 2);
        assert_eq!(report.count_by_category(ConflictCategory::DeadTransition), 1);
        assert_eq!(report.count_by_category(ConflictCategory::DeadGuard), 1);
    }

    #[test]
    fn add_interference_entries() {
        let mut graph = InterferenceGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_edge(0, 1, InterferenceKind::SpatialConflict);
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_interference(&graph);
        let report = builder.build();
        assert_eq!(report.count_by_category(ConflictCategory::Interference), 1);
    }

    #[test]
    fn add_temporal_races() {
        let trs = vec![TemporalRace {
            state_id: 0,
            transition_a: 1,
            transition_b: 2,
            overlap_description: "[0,5] overlaps [3,8]".into(),
        }];
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_temporal_races(&trs);
        let report = builder.build();
        assert_eq!(report.count_by_category(ConflictCategory::TemporalRace), 1);
    }

    #[test]
    fn add_livelocks() {
        let lls = vec![vec![1, 2, 3]];
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_livelocks(&lls);
        let report = builder.build();
        assert_eq!(report.count_by_category(ConflictCategory::Livelock), 1);
    }

    #[test]
    fn count_by_severity() {
        let mut builder = ReportBuilder::new("test_aut");
        builder.add_deadlocks(&sample_deadlocks());
        builder.add_races(&sample_races());
        let report = builder.build();
        assert!(report.count_by_severity(Severity::Critical) > 0);
    }
}
