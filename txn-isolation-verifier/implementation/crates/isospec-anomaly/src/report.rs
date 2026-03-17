//! Anomaly report formatting and rendering.
//!
//! Provides multiple output formats (text, JSON, DOT, Markdown, HTML) for
//! anomaly detection results, along with pre-written explanations, suggested
//! fixes, and database-migration impact assessments.

use std::collections::HashMap;

use isospec_types::isolation::*;
use isospec_types::dependency::*;
use isospec_types::config::EngineKind;

use crate::classifier::*;
use crate::detector::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Text,
    Json,
    Dot,
    Markdown,
    Html,
}

impl ReportFormat {
    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Json => "json",
            Self::Dot => "dot",
            Self::Markdown => "md",
            Self::Html => "html",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReportConfig {
    pub format: ReportFormat,
    pub include_explanations: bool,
    pub include_suggestions: bool,
    pub include_cycle_details: bool,
    pub include_migration_impact: bool,
    pub max_anomalies_shown: Option<usize>,
    pub target_engine: Option<EngineKind>,
    pub target_isolation: Option<IsolationLevel>,
}

impl ReportConfig {
    pub fn new(format: ReportFormat) -> Self {
        Self { format, include_explanations: true, include_suggestions: true,
            include_cycle_details: false, include_migration_impact: false,
            max_anomalies_shown: None, target_engine: None, target_isolation: None }
    }
    pub fn default_text() -> Self { Self::new(ReportFormat::Text) }
    pub fn default_json() -> Self {
        let mut c = Self::new(ReportFormat::Json); c.include_cycle_details = true; c
    }
    pub fn verbose() -> Self {
        Self { format: ReportFormat::Text, include_explanations: true, include_suggestions: true,
            include_cycle_details: true, include_migration_impact: true,
            max_anomalies_shown: None, target_engine: None, target_isolation: None }
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyExplanation {
    pub anomaly_class: AnomalyClass,
    pub short_desc: String,
    pub detailed_desc: String,
    pub example: String,
    pub sql_example: Option<String>,
}

impl AnomalyExplanation {
    pub fn for_class(class: AnomalyClass) -> Self {
        let (short, detail, ex, sql) = match class {
            AnomalyClass::G0 => (
                "Dirty Write",
                "Two transactions write to the same item without either committing first, \
                 leaving the final state dependent on scheduler timing.",
                "T1 writes x=1, T2 writes x=2, both commit — non-serializable ordering.",
                Some("-- T1: UPDATE accounts SET balance=100 WHERE id=1;\n\
                      -- T2: UPDATE accounts SET balance=200 WHERE id=1;\n-- Both COMMIT;"),
            ),
            AnomalyClass::G1a => (
                "Aborted Read",
                "A committed transaction observed a write from a transaction that later \
                 aborted, acting on state that never logically existed.",
                "T1 writes x=1, T2 reads x=1, T1 aborts — T2 saw a phantom value.",
                Some("-- T1: UPDATE t SET v=0 WHERE id=1;\n\
                      -- T2: SELECT v FROM t WHERE id=1; -- sees 0\n-- T1: ROLLBACK;"),
            ),
            AnomalyClass::G1b => (
                "Intermediate Read",
                "A transaction read an intermediate item version that the writer overwrote \
                 before committing, observing transient state.",
                "T1 writes x=1 then x=2 and commits — T2 read the intermediate value 1.",
                Some("-- T1: UPDATE t SET v=1 WHERE id=1;\n\
                      -- T2: SELECT v FROM t WHERE id=1; -- sees 1\n\
                      -- T1: UPDATE t SET v=2 WHERE id=1; COMMIT;"),
            ),
            AnomalyClass::G1c => (
                "Circular Information Flow",
                "A cycle of write-read and write-write edges among committed transactions \
                 creates information flow that no serial execution can produce.",
                "T1 reads x written by T2, T2 reads y written by T1 — circular dependency.",
                None,
            ),
            AnomalyClass::G2Item => (
                "Item Anti-Dependency Cycle",
                "A cycle contains at least one read-write anti-dependency on a specific item. \
                 One transaction reads an item version that another later overwrites.",
                "T1 reads x, T2 writes x, T2 reads y, T1 writes y — item anti-dep cycle.",
                Some("-- T1: SELECT v FROM t WHERE id=1;\n-- T2: UPDATE t SET v=10 WHERE id=1;\n\
                      -- T2: SELECT v FROM t WHERE id=2;\n-- T1: UPDATE t SET v=20 WHERE id=2;"),
            ),
            AnomalyClass::G2 => (
                "Predicate Anti-Dependency Cycle (Phantom)",
                "A cycle involving a predicate-level anti-dependency: a transaction's predicate \
                 read is invalidated by another transaction's insert or update.",
                "T1 reads WHERE status='open', T2 inserts status='open' — phantom row.",
                Some("-- T1: SELECT COUNT(*) FROM orders WHERE status='open';\n\
                      -- T2: INSERT INTO orders (status) VALUES ('open'); COMMIT;"),
            ),
        };
        Self {
            anomaly_class: class,
            short_desc: short.into(),
            detailed_desc: detail.into(),
            example: ex.into(),
            sql_example: sql.map(Into::into),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuggestedFix {
    pub anomaly_class: AnomalyClass,
    pub fix_description: String,
    pub recommended_level: IsolationLevel,
    pub performance_impact: String,
    pub sql_hint: Option<String>,
}

impl SuggestedFix {
    pub fn for_anomaly(class: AnomalyClass, current_level: Option<IsolationLevel>) -> Self {
        let (rec, desc, perf, hint) = match class {
            AnomalyClass::G0 => (
                IsolationLevel::ReadCommitted,
                "Upgrade to at least Read Committed to prevent dirty writes.",
                "Minimal — short write locks added.",
                Some("SET TRANSACTION ISOLATION LEVEL READ COMMITTED;"),
            ),
            AnomalyClass::G1a | AnomalyClass::G1b => (
                IsolationLevel::ReadCommitted,
                "Use Read Committed or higher to avoid reading uncommitted data.",
                "Low — standard locking prevents dirty and intermediate reads.",
                Some("SET TRANSACTION ISOLATION LEVEL READ COMMITTED;"),
            ),
            AnomalyClass::G1c => (
                IsolationLevel::RepeatableRead,
                "Upgrade to Repeatable Read to break circular information flow.",
                "Moderate — locks held longer, increasing contention.",
                Some("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;"),
            ),
            AnomalyClass::G2Item => (
                IsolationLevel::RepeatableRead,
                "Use Repeatable Read or Snapshot to prevent item anti-dependency cycles.",
                "Moderate — read locks held to commit, or MVCC snapshot overhead.",
                Some("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;"),
            ),
            AnomalyClass::G2 => (
                IsolationLevel::Serializable,
                "Only Serializable prevents predicate-level anti-dependency cycles.",
                "High — predicate locking or SSI required.",
                Some("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"),
            ),
        };

        let fix_description = match current_level {
            Some(lvl) if lvl.strength() >= rec.strength() =>
                format!("Already at {} but {} still detected — check engine semantics. {}", lvl, class.name(), desc),
            Some(lvl) => format!("Upgrade from {} to {}. {}", lvl, rec, desc),
            None => desc.to_string(),
        };
        Self { anomaly_class: class, fix_description, recommended_level: rec,
            performance_impact: perf.to_string(), sql_hint: hint.map(String::from) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel { Low, Medium, High, Critical }

impl RiskLevel {
    pub fn label(self) -> &'static str {
        match self { Self::Low => "Low", Self::Medium => "Medium",
            Self::High => "High", Self::Critical => "Critical" }
    }
}

#[derive(Debug, Clone)]
pub struct MigrationImpact {
    pub source_engine: EngineKind,
    pub target_engine: EngineKind,
    pub new_anomalies: Vec<AnomalyClass>,
    pub resolved_anomalies: Vec<AnomalyClass>,
    pub risk_level: RiskLevel,
    pub notes: Vec<String>,
}

impl MigrationImpact {
    pub fn assess(source: EngineKind, source_level: IsolationLevel,
                  target: EngineKind, target_level: IsolationLevel) -> Self {
        let src = source_level.possible_anomalies();
        let tgt = target_level.possible_anomalies();
        let new_anomalies: Vec<AnomalyClass> = tgt.iter().filter(|a| !src.contains(a)).copied().collect();
        let resolved_anomalies: Vec<AnomalyClass> = src.iter().filter(|a| !tgt.contains(a)).copied().collect();
        let mut notes = Vec::new();

        match (source, target) {
            (EngineKind::PostgreSQL, EngineKind::MySQL) => {
                notes.push("MySQL InnoDB uses next-key locking at RR, unlike PostgreSQL MVCC.".into());
                if target_level == IsolationLevel::RepeatableRead {
                    notes.push("MySQL RR prevents some phantoms via gap locks, unlike standard RR.".into());
                }
            }
            (EngineKind::PostgreSQL, EngineKind::SqlServer) => {
                notes.push("SQL Server uses lock-based isolation by default; consider RCSI for MVCC.".into());
            }
            (EngineKind::MySQL, EngineKind::PostgreSQL) => {
                notes.push("PostgreSQL uses pure MVCC; no gap locking at Repeatable Read.".into());
                if source_level == IsolationLevel::RepeatableRead {
                    notes.push("Phantoms possible under PostgreSQL RR that MySQL RR prevented.".into());
                }
            }
            (EngineKind::MySQL, EngineKind::SqlServer) => {
                notes.push("SQL Server locking model differs significantly from MySQL InnoDB.".into());
            }
            (EngineKind::SqlServer, EngineKind::PostgreSQL) => {
                notes.push("PostgreSQL has no direct equivalent of SQL Server RCSI.".into());
            }
            (EngineKind::SqlServer, EngineKind::MySQL) => {
                notes.push("MySQL InnoDB concurrency differs from SQL Server lock-based model.".into());
            }
            _ => {}
        }

        if source_level.strength() > target_level.strength() {
            notes.push(format!("Target ({}) is weaker than source ({}); expect more anomalies.",
                target_level, source_level));
        }
        let risk_level = if new_anomalies.iter().any(|a| a.severity() == AnomalySeverity::Critical) {
            RiskLevel::Critical
        } else if new_anomalies.len() >= 3 { RiskLevel::High
        } else if !new_anomalies.is_empty() { RiskLevel::Medium
        } else { RiskLevel::Low };

        Self { source_engine: source, target_engine: target,
            new_anomalies, resolved_anomalies, risk_level, notes }
    }
}

pub fn format_severity_indicator(severity: AnomalySeverity) -> String {
    match severity {
        AnomalySeverity::Critical => "\u{1f534} CRITICAL".into(),
        AnomalySeverity::High => "\u{1f7e0} HIGH".into(),
        AnomalySeverity::Medium => "\u{1f7e1} MEDIUM".into(),
        AnomalySeverity::Low => "\u{1f7e2} LOW".into(),
    }
}

pub fn format_cycle_as_text(cycle: &CycleInfo) -> String {
    if cycle.edges.is_empty() { return "(empty cycle)".into(); }
    cycle.edges.iter()
        .map(|e| format!("{} --{}--> {}", e.from_txn, e.dep_type.short_name(), e.to_txn))
        .collect::<Vec<_>>().join(", ")
}

pub fn format_cycle_as_dot(cycle: &CycleInfo, anomaly: AnomalyClass) -> String {
    let color = match anomaly.severity() {
        AnomalySeverity::Critical => "red", AnomalySeverity::High => "orangered",
        AnomalySeverity::Medium => "gold", AnomalySeverity::Low => "green",
    };
    let safe: String = anomaly.name().chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
    let mut out = format!("  subgraph cluster_{} {{\n    label=\"{}\";\n    color=\"{}\";\n",
        safe, anomaly.name(), color);
    for n in &cycle.nodes { out.push_str(&format!("    \"{}\" [label=\"{}\"];\n", n, n)); }
    for e in &cycle.edges {
        out.push_str(&format!("    \"{}\" -> \"{}\" [label=\"{}\" color=\"{}\"];\n",
            e.from_txn, e.to_txn, e.dep_type.short_name(), color));
    }
    out.push_str("  }\n");
    out
}

pub struct AnomalyReporter {
    config: ReportConfig,
}

impl AnomalyReporter {
    pub fn new(config: ReportConfig) -> Self {
        Self { config }
    }

    pub fn format_report(&self, report: &AnomalyReport) -> String {
        match self.config.format {
            ReportFormat::Text => self.format_text(report),
            ReportFormat::Json => self.format_json(report),
            ReportFormat::Dot => self.format_dot(report),
            ReportFormat::Markdown => self.format_markdown(report),
            ReportFormat::Html => self.format_html(report),
        }
    }

    fn format_text(&self, report: &AnomalyReport) -> String {
        let mut out = String::from("=== Anomaly Detection Report ===\n\n");
        let status = if report.is_clean() { "CLEAN" } else { "ANOMALIES DETECTED" };
        out.push_str(&format!("Status: {}\nAnomalies found: {}\nCycles examined: {}\n\
            Detection time: {} ms\n", status, report.anomaly_count(),
            report.cycles_examined, report.detection_time_ms));
        let sev = report.worst_severity().map(format_severity_indicator)
            .unwrap_or_else(|| "N/A".into());
        out.push_str(&format!("Overall severity: {}\n\n", sev));
        let limit = self.config.max_anomalies_shown.unwrap_or(usize::MAX);
        let shown = report.detected.len().min(limit);
        for (i, r) in report.detected.iter().take(shown).enumerate() {
            out.push_str(&format!("--- Anomaly #{} ---\nClass: {}\nSeverity: {}\nConfidence: {}\n",
                i + 1, r.anomaly_class.name(),
                format_severity_indicator(r.anomaly_class.severity()), r.confidence.as_str()));
            if self.config.include_explanations {
                let expl = AnomalyExplanation::for_class(r.anomaly_class);
                out.push_str(&format!("Description: {}\nDetail: {}\n", expl.short_desc, expl.detailed_desc));
            }
            if self.config.include_cycle_details {
                out.push_str(&format!("Cycle: {}\n", format_cycle_as_text(&r.cycle)));
            }
            out.push('\n');
        }
        if shown < report.detected.len() {
            out.push_str(&format!("... and {} more anomalies not shown.\n\n", report.detected.len() - shown));
        }
        if self.config.include_suggestions && !report.is_clean() {
            out.push_str("--- Recommendations ---\n");
            for rec in &report.severity.recommendations { out.push_str(&format!("  * {}\n", rec)); }
            for r in report.detected.iter().take(shown) {
                let fix = SuggestedFix::for_anomaly(r.anomaly_class, self.config.target_isolation);
                out.push_str(&format!("  * [{}] {}\n", r.anomaly_class.name(), fix.fix_description));
            }
            out.push('\n');
        }
        if self.config.include_migration_impact {
            if let (Some(engine), Some(level)) = (self.config.target_engine, self.config.target_isolation) {
                let impact = MigrationImpact::assess(
                    EngineKind::PostgreSQL, IsolationLevel::ReadCommitted, engine, level);
                out.push_str(&format!("--- Migration Impact ---\nRisk level: {}\n", impact.risk_level.label()));
                if !impact.new_anomalies.is_empty() {
                    let names: Vec<String> = impact.new_anomalies.iter().map(|a| a.name().into()).collect();
                    out.push_str(&format!("New anomalies: {}\n", names.join(", ")));
                }
                for note in &impact.notes { out.push_str(&format!("  Note: {}\n", note)); }
                out.push('\n');
            }
        }
        if !report.warnings.is_empty() {
            out.push_str("--- Warnings ---\n");
            for w in &report.warnings { out.push_str(&format!("  ! {}\n", w)); }
        }
        out
    }

    fn format_json(&self, report: &AnomalyReport) -> String {
        let mut out = format!("{{\n  \"clean\": {},\n  \"anomaly_count\": {},\n  \
            \"cycles_examined\": {},\n  \"detection_time_ms\": {},\n",
            report.is_clean(), report.anomaly_count(), report.cycles_examined, report.detection_time_ms);
        let sev_str = report.worst_severity().map(|s| format!("{:?}", s)).unwrap_or_else(|| "null".into());
        out.push_str(&format!("  \"severity\": {{\n    \"overall\": \"{}\",\n", sev_str));
        match &report.severity.worst_anomaly {
            Some(w) => out.push_str(&format!("    \"worst_anomaly\": \"{}\",\n", escape_json(w.name()))),
            None => out.push_str("    \"worst_anomaly\": null,\n"),
        }
        let counts: Vec<String> = report.severity.anomaly_counts.iter()
            .map(|(k, v)| format!("\"{}\": {}", escape_json(k.name()), v)).collect();
        out.push_str(&format!("    \"counts\": {{{}}}\n  }},\n", counts.join(", ")));
        out.push_str("  \"anomalies\": [\n");
        let limit = self.config.max_anomalies_shown.unwrap_or(usize::MAX);
        let items: Vec<String> = report.detected.iter().take(limit).map(|r| {
            let mut item = format!("    {{\n      \"class\": \"{}\",\n      \"severity\": \"{:?}\",\n\
                      \"confidence\": \"{}\",\n      \"explanation\": \"{}\",\n",
                escape_json(r.anomaly_class.name()), r.anomaly_class.severity(),
                r.confidence.as_str(), escape_json(&r.explanation));
            if self.config.include_cycle_details {
                let nodes: Vec<String> = r.cycle.nodes.iter().map(|n| format!("\"{}\"", n)).collect();
                item.push_str(&format!("      \"cycle_nodes\": [{}],\n", nodes.join(", ")));
                let edges: Vec<String> = r.cycle.edges.iter().map(|e|
                    format!("{{\"from\": \"{}\", \"to\": \"{}\", \"type\": \"{}\"}}",
                        e.from_txn, e.to_txn, e.dep_type.short_name())).collect();
                item.push_str(&format!("      \"cycle_edges\": [{}]\n", edges.join(", ")));
            } else {
                item.push_str(&format!("      \"cycle_node_count\": {}\n", r.cycle.nodes.len()));
            }
            item.push_str("    }");
            item
        }).collect();
        out.push_str(&items.join(",\n"));
        out.push_str("\n  ],\n");
        let ws: Vec<String> = report.warnings.iter().map(|w| format!("\"{}\"", escape_json(w))).collect();
        out.push_str(&format!("  \"warnings\": [{}]\n}}", ws.join(", ")));
        out
    }

    fn format_dot(&self, report: &AnomalyReport) -> String {
        let mut out = format!("digraph anomaly_report {{\n  rankdir=LR;\n  \
            node [shape=box style=rounded];\n  label=\"Anomaly Report ({} anomalies)\";\n\n",
            report.anomaly_count());
        let limit = self.config.max_anomalies_shown.unwrap_or(usize::MAX);
        for r in report.detected.iter().take(limit) {
            out.push_str(&format_cycle_as_dot(&r.cycle, r.anomaly_class));
        }
        out.push_str("}\n");
        out
    }

    fn format_markdown(&self, report: &AnomalyReport) -> String {
        let mut out = String::from("# Anomaly Detection Report\n\n");
        let badge = if report.is_clean() { "✅ Clean" } else { "❌ Anomalies Detected" };
        out.push_str(&format!("**Status:** {}\n\n## Summary\n\n", badge));
        out.push_str("| Metric | Value |\n|--------|-------|\n");
        let sev = report.worst_severity().map(format_severity_indicator).unwrap_or_else(|| "N/A".into());
        out.push_str(&format!("| Anomalies | {} |\n| Cycles examined | {} |\n\
            | Detection time | {} ms |\n| Severity | {} |\n\n",
            report.anomaly_count(), report.cycles_examined, report.detection_time_ms, sev));
        if !report.detected.is_empty() {
            out.push_str("## Anomalies\n\n");
            let limit = self.config.max_anomalies_shown.unwrap_or(usize::MAX);
            for (i, r) in report.detected.iter().take(limit).enumerate() {
                out.push_str(&format!("### {}. {}\n\n- **Severity:** {}\n- **Confidence:** {:.0}%\n",
                    i + 1, r.anomaly_class.name(),
                    format_severity_indicator(r.anomaly_class.severity()), r.confidence.as_str()));
                if self.config.include_explanations {
                    out.push_str(&format!("\n> {}\n", AnomalyExplanation::for_class(r.anomaly_class).detailed_desc));
                }
                if self.config.include_cycle_details {
                    out.push_str(&format!("\n**Cycle:** `{}`\n", format_cycle_as_text(&r.cycle)));
                }
                out.push('\n');
            }
        }
        if self.config.include_suggestions && !report.is_clean() {
            out.push_str("## Recommendations\n\n");
            for rec in &report.severity.recommendations { out.push_str(&format!("- {}\n", rec)); }
            out.push('\n');
        }
        if !report.warnings.is_empty() {
            out.push_str("## Warnings\n\n");
            for w in &report.warnings { out.push_str(&format!("- ⚠️ {}\n", w)); }
        }
        out
    }

    fn format_html(&self, report: &AnomalyReport) -> String {
        let status = if report.is_clean() { "Clean" } else { "Anomalies Detected" };
        let mut out = format!("<html><head><title>Anomaly Report</title></head><body>\n\
            <h1>Anomaly Detection Report</h1>\n\
            <p>Status: <strong>{}</strong></p>\n\
            <p>Anomalies: {}, Cycles examined: {}, Time: {} ms</p>\n",
            status, report.anomaly_count(), report.cycles_examined, report.detection_time_ms);
        if !report.detected.is_empty() {
            out.push_str("<h2>Anomalies</h2>\n<ul>\n");
            let limit = self.config.max_anomalies_shown.unwrap_or(usize::MAX);
            for r in report.detected.iter().take(limit) {
                out.push_str(&format!("<li><strong>{}</strong> — Confidence: {}</li>\n",
                    r.anomaly_class.name(), r.confidence.as_str()));
            }
            out.push_str("</ul>\n");
        }
        out.push_str("</body></html>\n");
        out
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::TransactionId;
    use std::collections::HashMap;

    fn make_cycle() -> CycleInfo {
        let (t1, t2, t3) = (TransactionId::new(1), TransactionId::new(2), TransactionId::new(3));
        CycleInfo {
            nodes: vec![t1, t2, t3],
            edges: vec![
                Dependency::new(t1, t2, DependencyType::WriteWrite),
                Dependency::new(t2, t3, DependencyType::ReadWrite),
                Dependency::new(t3, t1, DependencyType::WriteRead),
            ],
        }
    }
    fn make_report() -> AnomalyReport {
        let mut counts = HashMap::new();
        counts.insert(AnomalyClass::G1c, 1);
        AnomalyReport {
            detected: vec![ClassificationResult {
                anomaly_class: AnomalyClass::G1c, cycle: make_cycle(),
                confidence: 0.95, explanation: "Circular information flow detected.".into(),
            }],
            severity: SeverityAssessment {
                overall_severity: AnomalySeverity::High, anomaly_counts: counts,
                worst_anomaly: Some(AnomalyClass::G1c),
                recommendations: vec!["Upgrade to Repeatable Read.".into()],
            },
            detection_time_ms: 42, cycles_examined: 10, config_used: DetectionConfig::default(),
            warnings: vec!["Predicate analysis was skipped.".into()],
        }
    }
    fn make_empty_report() -> AnomalyReport {
        AnomalyReport {
            detected: vec![],
            severity: SeverityAssessment {
                overall_severity: AnomalySeverity::Low, anomaly_counts: HashMap::new(),
                worst_anomaly: None, recommendations: vec![],
            },
            detection_time_ms: 5, cycles_examined: 3,
            config_used: DetectionConfig::default(), warnings: vec![],
        }
    }

    #[test]
    fn text_report_key_sections() {
        let reporter = AnomalyReporter::new(ReportConfig::verbose());
        let text = reporter.format_report(&make_report());
        assert!(text.contains("Anomaly Detection Report"));
        assert!(text.contains("ANOMALIES DETECTED"));
        assert!(text.contains("G1c"));
        assert!(text.contains("Recommendations"));
        assert!(text.contains("Confidence: 95%"));
    }
    #[test]
    fn json_report_structure() {
        let reporter = AnomalyReporter::new(ReportConfig::default_json());
        let json = reporter.format_report(&make_report());
        assert!(json.contains("\"clean\": false"));
        assert!(json.contains("\"anomaly_count\": 1"));
        assert!(json.contains("\"anomalies\": ["));
        assert!(json.contains("\"cycle_nodes\""));
        assert!(json.contains("\"cycle_edges\""));
    }
    #[test]
    fn dot_output_graph_elements() {
        let reporter = AnomalyReporter::new(ReportConfig::new(ReportFormat::Dot));
        let dot = reporter.format_report(&make_report());
        assert!(dot.contains("digraph"));
        assert!(dot.contains("->"));
        assert!(dot.contains("label="));
        assert!(dot.contains("subgraph"));
    }
    #[test]
    fn markdown_output() {
        let cfg = ReportConfig { include_cycle_details: true, ..ReportConfig::new(ReportFormat::Markdown) };
        let md = AnomalyReporter::new(cfg).format_report(&make_report());
        assert!(md.contains("# Anomaly Detection Report"));
        assert!(md.contains("## Anomalies"));
        assert!(md.contains("**Cycle:**"));
    }
    #[test]
    fn empty_report_formats() {
        let text = AnomalyReporter::new(ReportConfig::default_text()).format_report(&make_empty_report());
        assert!(text.contains("CLEAN") && text.contains("Anomalies found: 0"));
        let json = AnomalyReporter::new(ReportConfig::default_json()).format_report(&make_empty_report());
        assert!(json.contains("\"clean\": true"));
    }
    #[test]
    fn explanation_for_every_class() {
        for &cls in &[AnomalyClass::G0, AnomalyClass::G1a, AnomalyClass::G1b,
                       AnomalyClass::G1c, AnomalyClass::G2Item, AnomalyClass::G2] {
            let expl = AnomalyExplanation::for_class(cls);
            assert_eq!(expl.anomaly_class, cls);
            assert!(!expl.short_desc.is_empty() && !expl.detailed_desc.is_empty());
        }
        assert!(AnomalyExplanation::for_class(AnomalyClass::G0).sql_example.unwrap().contains("UPDATE"));
    }
    #[test]
    fn suggested_fix_variants() {
        let up = SuggestedFix::for_anomaly(AnomalyClass::G2, Some(IsolationLevel::ReadCommitted));
        assert_eq!(up.recommended_level, IsolationLevel::Serializable);
        assert!(up.fix_description.contains("Upgrade") && up.sql_hint.is_some());
        let already = SuggestedFix::for_anomaly(AnomalyClass::G0, Some(IsolationLevel::Serializable));
        assert!(already.fix_description.contains("Already at"));
        let none = SuggestedFix::for_anomaly(AnomalyClass::G1a, None);
        assert!(none.fix_description.contains("Read Committed"));
    }
    #[test]
    fn migration_impact_assessment() {
        let weaker = MigrationImpact::assess(
            EngineKind::PostgreSQL, IsolationLevel::Serializable,
            EngineKind::MySQL, IsolationLevel::ReadCommitted,
        );
        assert!(!weaker.new_anomalies.is_empty() && !weaker.notes.is_empty());
        assert_ne!(weaker.risk_level, RiskLevel::Low);
        let same = MigrationImpact::assess(
            EngineKind::PostgreSQL, IsolationLevel::ReadCommitted,
            EngineKind::MySQL, IsolationLevel::ReadCommitted,
        );
        assert!(same.new_anomalies.is_empty());
        assert_eq!(same.risk_level, RiskLevel::Low);
    }
    #[test]
    fn severity_indicators() {
        assert!(format_severity_indicator(AnomalySeverity::Critical).contains("CRITICAL"));
        assert!(format_severity_indicator(AnomalySeverity::Low).contains("LOW"));
    }
    #[test]
    fn cycle_formatting() {
        let text = format_cycle_as_text(&make_cycle());
        assert!(text.contains("--ww-->") && text.contains("--rw-->") && text.contains("--wr-->"));
        assert_eq!(format_cycle_as_text(&CycleInfo { nodes: vec![], edges: vec![] }), "(empty cycle)");
        let dot = format_cycle_as_dot(&make_cycle(), AnomalyClass::G1c);
        assert!(dot.contains("subgraph") && dot.contains("->") && dot.contains("color="));
    }
    #[test]
    fn format_extensions_and_config() {
        assert_eq!(ReportFormat::Text.file_extension(), "txt");
        assert_eq!(ReportFormat::Json.file_extension(), "json");
        assert_eq!(ReportFormat::Dot.file_extension(), "dot");
        assert_eq!(ReportFormat::Markdown.file_extension(), "md");
        assert_eq!(ReportFormat::Html.file_extension(), "html");
        assert_eq!(ReportConfig::default_text().format, ReportFormat::Text);
        assert!(ReportConfig::default_json().include_cycle_details);
        assert!(ReportConfig::verbose().include_migration_impact);
        assert_eq!(RiskLevel::Low.label(), "Low");
        assert_eq!(RiskLevel::Critical.label(), "Critical");
    }
    #[test]
    fn max_anomalies_truncates() {
        let mut report = make_report();
        report.detected.push(ClassificationResult {
            anomaly_class: AnomalyClass::G2, cycle: make_cycle(),
            confidence: 0.80, explanation: "Phantom detected.".into(),
        });
        let mut cfg = ReportConfig::default_text();
        cfg.max_anomalies_shown = Some(1);
        let text = AnomalyReporter::new(cfg).format_report(&report);
        assert!(text.contains("Anomaly #1") && !text.contains("Anomaly #2"));
        assert!(text.contains("1 more anomalies not shown"));
    }
    #[test]
    fn html_report_basic() {
        let html = AnomalyReporter::new(ReportConfig::new(ReportFormat::Html)).format_report(&make_report());
        assert!(html.contains("<html>") && html.contains("Anomalies Detected") && html.contains("<li>"));
    }
}
