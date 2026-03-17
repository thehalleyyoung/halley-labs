use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl Severity {
    pub fn score(&self) -> u8 {
        match self {
            Severity::Critical => 4,
            Severity::High => 3,
            Severity::Medium => 2,
            Severity::Low => 1,
            Severity::Info => 0,
        }
    }

    pub fn from_score(score: u8) -> Self {
        match score {
            4 => Severity::Critical,
            3 => Severity::High,
            2 => Severity::Medium,
            1 => Severity::Low,
            _ => Severity::Info,
        }
    }

    pub fn emoji(&self) -> &str {
        match self {
            Severity::Critical => "🔴",
            Severity::High => "🟠",
            Severity::Medium => "🟡",
            Severity::Low => "🔵",
            Severity::Info => "⚪",
        }
    }
}

impl Ord for Severity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score().cmp(&other.score())
    }
}

impl PartialOrd for Severity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Severity::Critical => "CRITICAL",
            Severity::High => "HIGH",
            Severity::Medium => "MEDIUM",
            Severity::Low => "LOW",
            Severity::Info => "INFO",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// Location
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Location {
    pub file: Option<String>,
    pub service: Option<String>,
    pub edge: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

impl Location {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.file = Some(file.into());
        self
    }

    pub fn with_service(mut self, service: impl Into<String>) -> Self {
        self.service = Some(service.into());
        self
    }

    pub fn with_edge(mut self, edge: impl Into<String>) -> Self {
        self.edge = Some(edge.into());
        self
    }

    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_column(mut self, column: usize) -> Self {
        self.column = Some(column);
        self
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref file) = self.file {
            write!(f, "{}", file)?;
            if let Some(line) = self.line {
                write!(f, ":{}", line)?;
                if let Some(col) = self.column {
                    write!(f, ":{}", col)?;
                }
            }
            return Ok(());
        }
        if let Some(ref svc) = self.service {
            write!(f, "{}", svc)?;
            if let Some(ref edge) = self.edge {
                write!(f, " ({})", edge)?;
            }
            return Ok(());
        }
        if let Some(ref edge) = self.edge {
            return write!(f, "{}", edge);
        }
        write!(f, "<unknown>")
    }
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub description: String,
    pub value: Option<String>,
    pub source: Option<String>,
}

impl Evidence {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            value: None,
            source: None,
        }
    }

    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

impl fmt::Display for Evidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)?;
        if let Some(ref val) = self.value {
            write!(f, " [value: {}]", val)?;
        }
        if let Some(ref src) = self.source {
            write!(f, " (source: {})", src)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CodeFlow / CodeFlowStep
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFlowStep {
    pub location: Location,
    pub message: String,
}

impl CodeFlowStep {
    pub fn new(location: Location, message: impl Into<String>) -> Self {
        Self {
            location,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeFlow {
    pub steps: Vec<CodeFlowStep>,
}

impl CodeFlow {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_step(&mut self, step: CodeFlowStep) {
        self.steps.push(step);
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Finding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub id: String,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub location: Location,
    pub evidence: Vec<Evidence>,
    pub code_flow: Option<CodeFlow>,
    pub remediation: Option<String>,
}

impl Finding {
    pub fn new(
        severity: Severity,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            severity,
            title: title.into(),
            description: description.into(),
            location: Location::default(),
            evidence: Vec::new(),
            code_flow: None,
            remediation: None,
        }
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence.push(evidence);
        self
    }

    pub fn with_code_flow(mut self, code_flow: CodeFlow) -> Self {
        self.code_flow = Some(code_flow);
        self
    }

    pub fn with_remediation(mut self, remediation: impl Into<String>) -> Self {
        self.remediation = Some(remediation.into());
        self
    }
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] {}: {} @ {}",
            self.severity.emoji(),
            self.severity,
            self.title,
            self.description,
            self.location,
        )
    }
}

// ---------------------------------------------------------------------------
// ReportFormat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReportFormat {
    Human,
    Sarif,
    JUnit,
    Json,
    Yaml,
    Markdown,
}

impl ReportFormat {
    pub fn file_extension(&self) -> &str {
        match self {
            ReportFormat::Human => "txt",
            ReportFormat::Sarif => "sarif.json",
            ReportFormat::JUnit => "xml",
            ReportFormat::Json => "json",
            ReportFormat::Yaml => "yaml",
            ReportFormat::Markdown => "md",
        }
    }

    pub fn content_type(&self) -> &str {
        match self {
            ReportFormat::Human => "text/plain",
            ReportFormat::Sarif => "application/json",
            ReportFormat::JUnit => "application/xml",
            ReportFormat::Json => "application/json",
            ReportFormat::Yaml => "application/x-yaml",
            ReportFormat::Markdown => "text/markdown",
        }
    }
}

impl fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            ReportFormat::Human => "Human-Readable",
            ReportFormat::Sarif => "SARIF",
            ReportFormat::JUnit => "JUnit",
            ReportFormat::Json => "JSON",
            ReportFormat::Yaml => "YAML",
            ReportFormat::Markdown => "Markdown",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// ReportMetadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub tool_name: String,
    pub tool_version: String,
    pub analysis_timestamp: DateTime<Utc>,
    pub target: String,
    pub duration_ms: u64,
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self {
            tool_name: "cascade-verify".to_string(),
            tool_version: "0.1.0".to_string(),
            analysis_timestamp: Utc::now(),
            target: String::new(),
            duration_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ReportSummary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_findings: usize,
    pub by_severity: BTreeMap<String, usize>,
    pub services_analyzed: usize,
    pub edges_analyzed: usize,
    pub pass: bool,
}

impl ReportSummary {
    pub fn new() -> Self {
        Self {
            total_findings: 0,
            by_severity: BTreeMap::new(),
            services_analyzed: 0,
            edges_analyzed: 0,
            pass: true,
        }
    }

    pub fn compute(findings: &[Finding]) -> Self {
        let mut by_severity = BTreeMap::new();
        let mut services: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut edges: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut has_critical = false;

        for f in findings {
            *by_severity
                .entry(f.severity.to_string())
                .or_insert(0usize) += 1;
            if let Some(ref svc) = f.location.service {
                services.insert(svc.clone());
            }
            if let Some(ref edge) = f.location.edge {
                edges.insert(edge.clone());
            }
            if f.severity == Severity::Critical {
                has_critical = true;
            }
        }

        Self {
            total_findings: findings.len(),
            by_severity,
            services_analyzed: services.len(),
            edges_analyzed: edges.len(),
            pass: !has_critical,
        }
    }
}

impl Default for ReportSummary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AnalysisReport
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub metadata: ReportMetadata,
    pub findings: Vec<Finding>,
    pub summary: ReportSummary,
    pub raw_data: Option<serde_json::Value>,
}

impl AnalysisReport {
    pub fn new(metadata: ReportMetadata) -> Self {
        Self {
            metadata,
            findings: Vec::new(),
            summary: ReportSummary::new(),
            raw_data: None,
        }
    }

    pub fn add_finding(&mut self, finding: Finding) {
        self.findings.push(finding);
        self.summary = ReportSummary::compute(&self.findings);
    }

    pub fn findings_by_severity(&self, severity: Severity) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| f.severity == severity)
            .collect()
    }

    pub fn critical_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Critical)
            .count()
    }

    pub fn has_critical(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == Severity::Critical)
    }

    pub fn finding_count(&self) -> usize {
        self.findings.len()
    }

    pub fn sorted_findings(&self) -> Vec<&Finding> {
        let mut sorted: Vec<&Finding> = self.findings.iter().collect();
        sorted.sort_by(|a, b| b.severity.cmp(&a.severity));
        sorted
    }
}

impl fmt::Display for AnalysisReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Analysis Report — {} v{}",
            self.metadata.tool_name, self.metadata.tool_version
        )?;
        writeln!(f, "Target: {}", self.metadata.target)?;
        writeln!(f, "Findings: {}", self.finding_count())?;
        writeln!(
            f,
            "Status: {}",
            if self.summary.pass { "PASS" } else { "FAIL" }
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SarifReport
// ---------------------------------------------------------------------------

pub struct SarifReport<'a> {
    report: &'a AnalysisReport,
}

impl<'a> SarifReport<'a> {
    pub fn new(report: &'a AnalysisReport) -> Self {
        Self { report }
    }

    fn severity_to_sarif_level(severity: &Severity) -> &'static str {
        match severity {
            Severity::Critical | Severity::High => "error",
            Severity::Medium => "warning",
            Severity::Low | Severity::Info => "note",
        }
    }

    pub fn to_sarif_json(&self) -> serde_json::Value {
        let results: Vec<serde_json::Value> = self
            .report
            .findings
            .iter()
            .map(|f| {
                let mut loc = serde_json::json!({});
                if let Some(ref file) = f.location.file {
                    let mut region = serde_json::json!({});
                    if let Some(line) = f.location.line {
                        region["startLine"] = serde_json::json!(line);
                    }
                    if let Some(col) = f.location.column {
                        region["startColumn"] = serde_json::json!(col);
                    }
                    loc = serde_json::json!({
                        "physicalLocation": {
                            "artifactLocation": { "uri": file },
                            "region": region
                        }
                    });
                }

                serde_json::json!({
                    "ruleId": f.id,
                    "message": { "text": f.description },
                    "level": Self::severity_to_sarif_level(&f.severity),
                    "locations": [loc]
                })
            })
            .collect();

        serde_json::json!({
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": self.report.metadata.tool_name,
                        "version": self.report.metadata.tool_version
                    }
                },
                "results": results
            }]
        })
    }
}

// ---------------------------------------------------------------------------
// JUnitReport
// ---------------------------------------------------------------------------

pub struct JUnitReport<'a> {
    report: &'a AnalysisReport,
}

impl<'a> JUnitReport<'a> {
    pub fn new(report: &'a AnalysisReport) -> Self {
        Self { report }
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    pub fn to_xml(&self) -> String {
        let failure_count = self
            .report
            .findings
            .iter()
            .filter(|f| f.severity >= Severity::High)
            .count();

        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<testsuite name=\"{}\" tests=\"{}\" failures=\"{}\" timestamp=\"{}\">\n",
            Self::escape_xml(&self.report.metadata.tool_name),
            self.report.findings.len(),
            failure_count,
            self.report.metadata.analysis_timestamp.to_rfc3339(),
        ));

        for finding in &self.report.findings {
            xml.push_str(&format!(
                "  <testcase name=\"{}\" classname=\"{}\"",
                Self::escape_xml(&finding.title),
                Self::escape_xml(&finding.location.to_string()),
            ));

            if finding.severity >= Severity::High {
                xml.push_str(">\n");
                xml.push_str(&format!(
                    "    <failure message=\"{}\" type=\"{}\">{}</failure>\n",
                    Self::escape_xml(&finding.title),
                    finding.severity,
                    Self::escape_xml(&finding.description),
                ));
                xml.push_str("  </testcase>\n");
            } else {
                xml.push_str(" />\n");
            }
        }

        xml.push_str("</testsuite>\n");
        xml
    }
}

// ---------------------------------------------------------------------------
// HumanReadableReport
// ---------------------------------------------------------------------------

pub struct HumanReadableReport<'a> {
    report: &'a AnalysisReport,
}

impl<'a> HumanReadableReport<'a> {
    pub fn new(report: &'a AnalysisReport) -> Self {
        Self { report }
    }

    pub fn render(&self) -> String {
        let mut out = String::new();

        // Header
        out.push_str(&format!(
            "╔══════════════════════════════════════════════════╗\n"
        ));
        out.push_str(&format!(
            "║  {} v{}  ║\n",
            self.report.metadata.tool_name, self.report.metadata.tool_version,
        ));
        out.push_str(&format!(
            "╚══════════════════════════════════════════════════╝\n\n"
        ));

        out.push_str(&format!("Target:   {}\n", self.report.metadata.target));
        out.push_str(&format!(
            "Duration: {}ms\n\n",
            self.report.metadata.duration_ms
        ));

        // Summary table
        out.push_str("── Summary ──────────────────────────────────────\n");
        out.push_str(&format!(
            "  Total findings : {}\n",
            self.report.summary.total_findings
        ));
        for (sev, count) in &self.report.summary.by_severity {
            out.push_str(&format!("  {:12}   : {}\n", sev, count));
        }
        out.push_str(&format!(
            "  Services       : {}\n",
            self.report.summary.services_analyzed
        ));
        out.push_str(&format!(
            "  Edges          : {}\n",
            self.report.summary.edges_analyzed
        ));
        let status = if self.report.summary.pass {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        out.push_str(&format!("  Status         : {}\n\n", status));

        // Findings grouped by severity (descending)
        let sorted = self.report.sorted_findings();
        if sorted.is_empty() {
            out.push_str("No findings.\n");
        } else {
            out.push_str("── Findings ─────────────────────────────────────\n");
            for finding in &sorted {
                out.push_str(&format!(
                    "\n{} [{}] {}\n",
                    finding.severity.emoji(),
                    finding.severity,
                    finding.title,
                ));
                out.push_str(&format!("  Location:    {}\n", finding.location));
                out.push_str(&format!("  Description: {}\n", finding.description));
                for ev in &finding.evidence {
                    out.push_str(&format!("  Evidence:    {}\n", ev));
                }
                if let Some(ref rem) = finding.remediation {
                    out.push_str(&format!("  Remediation: {}\n", rem));
                }
            }
        }

        out
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Severity -----------------------------------------------------------

    #[test]
    fn test_severity_scores() {
        assert_eq!(Severity::Critical.score(), 4);
        assert_eq!(Severity::High.score(), 3);
        assert_eq!(Severity::Medium.score(), 2);
        assert_eq!(Severity::Low.score(), 1);
        assert_eq!(Severity::Info.score(), 0);
    }

    #[test]
    fn test_severity_from_score() {
        assert_eq!(Severity::from_score(4), Severity::Critical);
        assert_eq!(Severity::from_score(3), Severity::High);
        assert_eq!(Severity::from_score(2), Severity::Medium);
        assert_eq!(Severity::from_score(1), Severity::Low);
        assert_eq!(Severity::from_score(0), Severity::Info);
        assert_eq!(Severity::from_score(255), Severity::Info);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
        assert!(Severity::Low > Severity::Info);

        let mut sevs = vec![Severity::Low, Severity::Critical, Severity::Info, Severity::High];
        sevs.sort();
        assert_eq!(
            sevs,
            vec![Severity::Info, Severity::Low, Severity::High, Severity::Critical]
        );
    }

    #[test]
    fn test_severity_display_and_emoji() {
        assert_eq!(format!("{}", Severity::Critical), "CRITICAL");
        assert_eq!(Severity::Critical.emoji(), "🔴");
        assert_eq!(Severity::High.emoji(), "🟠");
        assert_eq!(Severity::Medium.emoji(), "🟡");
        assert_eq!(Severity::Low.emoji(), "🔵");
        assert_eq!(Severity::Info.emoji(), "⚪");
    }

    // -- Location -----------------------------------------------------------

    #[test]
    fn test_location_display_file_line() {
        let loc = Location::new()
            .with_file("config.yaml")
            .with_line(42)
            .with_column(5);
        assert_eq!(format!("{}", loc), "config.yaml:42:5");
    }

    #[test]
    fn test_location_display_service() {
        let loc = Location::new()
            .with_service("api-gateway")
            .with_edge("api-gateway -> auth");
        assert_eq!(format!("{}", loc), "api-gateway (api-gateway -> auth)");
    }

    #[test]
    fn test_location_display_unknown() {
        let loc = Location::new();
        assert_eq!(format!("{}", loc), "<unknown>");
    }

    // -- Evidence -----------------------------------------------------------

    #[test]
    fn test_evidence_display() {
        let ev = Evidence::new("timeout mismatch")
            .with_value("30s vs 10s")
            .with_source("config.yaml");
        let s = format!("{}", ev);
        assert!(s.contains("timeout mismatch"));
        assert!(s.contains("30s vs 10s"));
        assert!(s.contains("config.yaml"));
    }

    // -- CodeFlow -----------------------------------------------------------

    #[test]
    fn test_code_flow() {
        let mut cf = CodeFlow::new();
        assert!(cf.is_empty());
        assert_eq!(cf.len(), 0);

        cf.add_step(CodeFlowStep::new(
            Location::new().with_service("gateway"),
            "request enters gateway",
        ));
        cf.add_step(CodeFlowStep::new(
            Location::new().with_service("auth"),
            "forwarded to auth",
        ));
        assert_eq!(cf.len(), 2);
        assert!(!cf.is_empty());
    }

    // -- Finding builder ----------------------------------------------------

    #[test]
    fn test_finding_builder() {
        let finding = Finding::new(
            Severity::High,
            "Timeout mismatch",
            "Upstream timeout exceeds downstream",
        )
        .with_location(Location::new().with_service("api-gw"))
        .with_evidence(Evidence::new("observed delta").with_value("20s"))
        .with_remediation("Align timeout values");

        assert_eq!(finding.severity, Severity::High);
        assert_eq!(finding.title, "Timeout mismatch");
        assert_eq!(finding.evidence.len(), 1);
        assert!(finding.remediation.is_some());
        assert!(!finding.id.is_empty());
    }

    #[test]
    fn test_finding_display() {
        let finding = Finding::new(Severity::Critical, "Circular dep", "A -> B -> A")
            .with_location(Location::new().with_service("svc-a"));
        let display = format!("{}", finding);
        assert!(display.contains("CRITICAL"));
        assert!(display.contains("Circular dep"));
        assert!(display.contains("svc-a"));
    }

    // -- ReportFormat -------------------------------------------------------

    #[test]
    fn test_report_format_extensions() {
        assert_eq!(ReportFormat::Human.file_extension(), "txt");
        assert_eq!(ReportFormat::Sarif.file_extension(), "sarif.json");
        assert_eq!(ReportFormat::JUnit.file_extension(), "xml");
        assert_eq!(ReportFormat::Json.file_extension(), "json");
        assert_eq!(ReportFormat::Yaml.file_extension(), "yaml");
        assert_eq!(ReportFormat::Markdown.file_extension(), "md");
    }

    #[test]
    fn test_report_format_content_types() {
        assert_eq!(ReportFormat::Human.content_type(), "text/plain");
        assert_eq!(ReportFormat::Sarif.content_type(), "application/json");
        assert_eq!(ReportFormat::JUnit.content_type(), "application/xml");
        assert_eq!(ReportFormat::Json.content_type(), "application/json");
    }

    // -- AnalysisReport -----------------------------------------------------

    fn sample_report() -> AnalysisReport {
        let meta = ReportMetadata {
            target: "test-mesh".to_string(),
            duration_ms: 123,
            ..ReportMetadata::default()
        };
        let mut report = AnalysisReport::new(meta);
        report.add_finding(
            Finding::new(Severity::Critical, "Critical bug", "desc-c")
                .with_location(Location::new().with_service("svc-a").with_edge("a->b")),
        );
        report.add_finding(
            Finding::new(Severity::High, "High issue", "desc-h")
                .with_location(Location::new().with_service("svc-b")),
        );
        report.add_finding(
            Finding::new(Severity::Low, "Low note", "desc-l")
                .with_location(Location::new().with_service("svc-a")),
        );
        report
    }

    #[test]
    fn test_analysis_report_counts() {
        let report = sample_report();
        assert_eq!(report.finding_count(), 3);
        assert_eq!(report.critical_count(), 1);
        assert!(report.has_critical());
    }

    #[test]
    fn test_analysis_report_filter_by_severity() {
        let report = sample_report();
        assert_eq!(report.findings_by_severity(Severity::Critical).len(), 1);
        assert_eq!(report.findings_by_severity(Severity::High).len(), 1);
        assert_eq!(report.findings_by_severity(Severity::Medium).len(), 0);
    }

    #[test]
    fn test_analysis_report_sorted_findings() {
        let report = sample_report();
        let sorted = report.sorted_findings();
        assert_eq!(sorted[0].severity, Severity::Critical);
        assert_eq!(sorted[1].severity, Severity::High);
        assert_eq!(sorted[2].severity, Severity::Low);
    }

    // -- ReportSummary ------------------------------------------------------

    #[test]
    fn test_report_summary_compute() {
        let report = sample_report();
        let summary = &report.summary;
        assert_eq!(summary.total_findings, 3);
        assert_eq!(summary.by_severity.get("CRITICAL"), Some(&1));
        assert_eq!(summary.by_severity.get("HIGH"), Some(&1));
        assert_eq!(summary.by_severity.get("LOW"), Some(&1));
        assert_eq!(summary.services_analyzed, 2); // svc-a, svc-b
        assert_eq!(summary.edges_analyzed, 1); // a->b
        assert!(!summary.pass); // has critical
    }

    // -- SarifReport --------------------------------------------------------

    #[test]
    fn test_sarif_report_structure() {
        let report = sample_report();
        let sarif = SarifReport::new(&report).to_sarif_json();

        assert_eq!(sarif["version"], "2.1.0");
        assert!(sarif["$schema"].as_str().unwrap().contains("sarif"));

        let runs = sarif["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 1);

        let tool = &runs[0]["tool"]["driver"];
        assert_eq!(tool["name"], "cascade-verify");
        assert_eq!(tool["version"], "0.1.0");

        let results = runs[0]["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);

        // Critical -> error level
        assert_eq!(results[0]["level"], "error");
    }

    // -- JUnitReport --------------------------------------------------------

    #[test]
    fn test_junit_report_structure() {
        let report = sample_report();
        let xml = JUnitReport::new(&report).to_xml();

        assert!(xml.starts_with("<?xml"));
        assert!(xml.contains("<testsuite"));
        assert!(xml.contains("failures=\"2\"")); // Critical + High
        assert!(xml.contains("<testcase"));
        assert!(xml.contains("<failure"));
        assert!(xml.contains("</testsuite>"));
    }

    // -- HumanReadableReport ------------------------------------------------

    #[test]
    fn test_human_readable_report() {
        let report = sample_report();
        let rendered = HumanReadableReport::new(&report).render();

        assert!(rendered.contains("cascade-verify"));
        assert!(rendered.contains("test-mesh"));
        assert!(rendered.contains("CRITICAL"));
        assert!(rendered.contains("FAIL"));
        assert!(rendered.contains("Total findings : 3"));
        assert!(rendered.contains("🔴"));
    }

    // -- Serialization round-trip -------------------------------------------

    #[test]
    fn test_severity_serialization_roundtrip() {
        let sev = Severity::High;
        let json = serde_json::to_string(&sev).unwrap();
        let deser: Severity = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, sev);
    }

    #[test]
    fn test_finding_serialization() {
        let finding = Finding::new(Severity::Medium, "Retry storm risk", "Too many retries")
            .with_location(Location::new().with_file("mesh.yaml").with_line(10));
        let json = serde_json::to_string(&finding).unwrap();
        let deser: Finding = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.severity, Severity::Medium);
        assert_eq!(deser.title, "Retry storm risk");
        assert_eq!(deser.location.file.as_deref(), Some("mesh.yaml"));
        assert_eq!(deser.location.line, Some(10));
    }
}
