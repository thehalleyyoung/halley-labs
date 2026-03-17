//! Full localization report generation.
//!
//! Assembles [`LocalizationReport`] documents from localization results,
//! BFI data, and statistical analyses.  Reports consist of a header,
//! ranked findings, evidence chains, methodology descriptions, statistical
//! summaries, recommendations, and appendices.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use localization::LocalizationResult as LocResult;

use crate::bfi::{BFIComputer, BFIInterpretation, BFIResult, BFITrend};

// ── Report format ───────────────────────────────────────────────────────────

/// Supported serialization formats for the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Markdown,
    Html,
    PlainText,
    /// Alias for [`PlainText`](Self::PlainText).
    Plain,
    Csv,
}

impl fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json => write!(f, "JSON"),
            Self::Markdown => write!(f, "Markdown"),
            Self::Html => write!(f, "HTML"),
            Self::PlainText | Self::Plain => write!(f, "PlainText"),
            Self::Csv => write!(f, "CSV"),
        }
    }
}

// ── Severity ────────────────────────────────────────────────────────────────

/// Severity of a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FindingSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl fmt::Display for FindingSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Priority of a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Immediate,
    High,
    Medium,
    Low,
}

impl fmt::Display for RecommendationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ── Building blocks ─────────────────────────────────────────────────────────

/// Report metadata header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportHeader {
    pub title: String,
    pub pipeline_name: String,
    pub generated_at: String,
    pub version: String,
    pub metadata: HashMap<String, String>,
}

/// A piece of evidence supporting a finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub label: String,
    pub value: String,
    /// Optional reference to a metric or data source.
    pub source: Option<String>,
}

/// A localization finding: a suspected fault at a pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub id: String,
    pub stage_name: String,
    pub title: String,
    pub description: String,
    pub severity: FindingSeverity,
    pub suspiciousness: f64,
    pub evidence: Vec<Evidence>,
    pub recommendations: Vec<String>,
}

/// An actionable recommendation derived from the findings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub title: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub related_findings: Vec<String>,
}

/// Methodology section describing how the analysis was run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodologySection {
    pub transformations_used: Vec<String>,
    pub sbfl_metric: String,
    pub distance_metric: String,
    pub test_count: usize,
    pub violation_count: usize,
    pub notes: Vec<String>,
}

/// Statistical summary section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSection {
    /// Per-stage suspiciousness scores.
    pub suspiciousness_scores: Vec<(String, f64)>,
    /// Per-stage BFI values.
    pub bfi_values: Vec<(String, f64)>,
    /// Overall BFI trend across the pipeline.
    pub bfi_trend: String,
    /// Separation ratio between the top suspect and the second.
    pub separation_ratio: f64,
    /// Number of stages examined before finding the fault.
    pub exam_score: f64,
}

/// Appendix – arbitrary supplementary data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Appendix {
    pub title: String,
    pub content: String,
}

// ── LocalizationReport ──────────────────────────────────────────────────────

/// The complete localization report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationReport {
    pub header: ReportHeader,
    pub findings: Vec<Finding>,
    pub methodology: Option<MethodologySection>,
    pub statistical_section: Option<StatisticalSection>,
    pub recommendations: Vec<Recommendation>,
    pub appendices: Vec<Appendix>,
}

impl LocalizationReport {
    /// Top finding (highest suspiciousness).
    pub fn top_finding(&self) -> Option<&Finding> {
        self.findings.first()
    }

    /// Findings filtered by minimum severity.
    pub fn findings_by_severity(&self, min: FindingSeverity) -> Vec<&Finding> {
        let min_ord = severity_order(min);
        self.findings
            .iter()
            .filter(|f| severity_order(f.severity) >= min_ord)
            .collect()
    }

    /// Render the report to a string in the specified format.
    pub fn render(&self, format: ReportFormat) -> String {
        match format {
            ReportFormat::Json => {
                serde_json::to_string_pretty(self).unwrap_or_default()
            }
            ReportFormat::Markdown => render_markdown(self),
            ReportFormat::Html => render_html(self),
            ReportFormat::PlainText | ReportFormat::Plain => render_plaintext(self),
            ReportFormat::Csv => render_csv(self),
        }
    }
}

fn severity_order(s: FindingSeverity) -> u8 {
    match s {
        FindingSeverity::Critical => 4,
        FindingSeverity::High => 3,
        FindingSeverity::Medium => 2,
        FindingSeverity::Low => 1,
        FindingSeverity::Info => 0,
    }
}

// ── ReportGenerator ─────────────────────────────────────────────────────────

/// Generates a [`LocalizationReport`] from raw analysis data.
#[derive(Debug, Clone)]
pub struct ReportGenerator {
    pub pipeline_name: String,
    pub version: String,
    /// Suspiciousness threshold above which a finding is created.
    pub suspiciousness_threshold: f64,
    /// Whether to include a methodology section.
    pub include_methodology: bool,
    /// Whether to include a statistical section.
    pub include_statistics: bool,
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new("default-pipeline")
    }
}

impl ReportGenerator {
    pub fn new(pipeline_name: impl Into<String>) -> Self {
        Self {
            pipeline_name: pipeline_name.into(),
            version: "0.1.0".into(),
            suspiciousness_threshold: 0.3,
            include_methodology: true,
            include_statistics: true,
        }
    }

    /// Generate a report from stage-level suspiciousness scores and BFI data.
    ///
    /// * `stage_names` – ordered pipeline stage names
    /// * `suspiciousness` – per-stage suspiciousness scores (same order)
    /// * `bfi_results` – optional BFI results per stage
    /// * `test_count` – total test cases executed
    /// * `violation_count` – test cases that violated at least one MR
    /// * `transformations` – names of transformations used
    pub fn generate(
        &self,
        stage_names: &[String],
        suspiciousness: &[f64],
        bfi_results: &[BFIResult],
        test_count: usize,
        violation_count: usize,
        transformations: &[String],
    ) -> LocalizationReport {
        // Build ranked findings.
        let mut indexed: Vec<(usize, f64)> = suspiciousness.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut findings = Vec::new();
        for (rank, &(idx, score)) in indexed.iter().enumerate() {
            if score < self.suspiciousness_threshold {
                continue;
            }
            let name = stage_names.get(idx).cloned().unwrap_or_else(|| format!("stage-{idx}"));
            let bfi = bfi_results.get(idx);
            let severity = classify_severity(score, bfi);

            let mut evidence = vec![
                Evidence {
                    label: "Suspiciousness".into(),
                    value: format!("{score:.4}"),
                    source: Some("SBFL".into()),
                },
                Evidence {
                    label: "Rank".into(),
                    value: format!("{}", rank + 1),
                    source: None,
                },
            ];
            if let Some(b) = bfi {
                evidence.push(Evidence {
                    label: "BFI".into(),
                    value: format!("{:.3} ({})", b.bfi_value, b.interpretation),
                    source: Some("BFI computation".into()),
                });
            }

            let mut recs = Vec::new();
            if let Some(b) = bfi {
                if b.interpretation == BFIInterpretation::Amplifying {
                    recs.push(format!(
                        "Investigate why {} amplifies divergence (BFI = {:.2})",
                        name, b.bfi_value
                    ));
                }
            }
            if score > 0.8 {
                recs.push(format!("Prioritize fixing {} – very high suspiciousness", name));
            }

            findings.push(Finding {
                id: format!("F-{}", rank + 1),
                stage_name: name.clone(),
                title: format!("{} fault localization", name),
                description: build_finding_description(&name, score, bfi),
                severity,
                suspiciousness: score,
                evidence,
                recommendations: recs,
            });
        }

        // Global recommendations.
        let recommendations = build_recommendations(&findings);

        // Methodology.
        let methodology = if self.include_methodology {
            Some(MethodologySection {
                transformations_used: transformations.to_vec(),
                sbfl_metric: "Ochiai (continuous-adapted)".into(),
                distance_metric: "Composite".into(),
                test_count,
                violation_count,
                notes: vec![
                    "Differential magnitudes computed per stage via IR distance".into(),
                    "BFI computed as ratio of consecutive stage mean differentials".into(),
                ],
            })
        } else {
            None
        };

        // Statistical section.
        let statistical_section = if self.include_statistics {
            let susp_scores: Vec<(String, f64)> = stage_names
                .iter()
                .zip(suspiciousness.iter())
                .map(|(n, &s)| (n.clone(), s))
                .collect();
            let bfi_vals: Vec<(String, f64)> = bfi_results
                .iter()
                .map(|b| (b.stage_name.clone(), b.bfi_value))
                .collect();
            let trend = BFIComputer::default().trend_analysis(bfi_results);
            let sep = compute_separation_ratio(suspiciousness);
            let exam = compute_exam_score(suspiciousness);

            Some(StatisticalSection {
                suspiciousness_scores: susp_scores,
                bfi_values: bfi_vals,
                bfi_trend: trend.to_string(),
                separation_ratio: sep,
                exam_score: exam,
            })
        } else {
            None
        };

        LocalizationReport {
            header: ReportHeader {
                title: format!("Localization Report – {}", self.pipeline_name),
                pipeline_name: self.pipeline_name.clone(),
                generated_at: chrono::Utc::now().to_rfc3339(),
                version: self.version.clone(),
                metadata: HashMap::new(),
            },
            findings,
            methodology,
            statistical_section,
            recommendations,
            appendices: Vec::new(),
        }
    }

    /// Generate a report from a [`localization::LocalizationResult`].
    ///
    /// Extracts stage names, suspiciousness scores, and BFI data from the
    /// localization result and delegates to [`generate`](Self::generate).
    pub fn generate_report(&self, result: &LocResult) -> LocalizationReport {
        let stage_names: Vec<String> = result.stage_results.iter().map(|s| s.stage_name.clone()).collect();
        let susp: Vec<f64> = result.stage_results.iter().map(|s| s.suspiciousness).collect();
        let stage_diffs: Vec<Vec<f64>> = result.stage_results.iter().map(|s| s.differential_data.clone()).collect();
        let bfi_computer = BFIComputer::default();
        let bfi = bfi_computer.compute_all_bfi(&stage_names, &stage_diffs);
        self.generate(
            &stage_names,
            &susp,
            &bfi,
            result.test_count,
            result.violation_count,
            &result.transformations_used,
        )
    }
}

// ── Public convenience function ─────────────────────────────────────────────

/// Render a [`LocalizationReport`] to a string in the given format.
///
/// This is a free function re-exported for CLI convenience.
pub fn render_report(report: &LocalizationReport, format: ReportFormat) -> String {
    report.render(format)
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn classify_severity(score: f64, bfi: Option<&BFIResult>) -> FindingSeverity {
    let amplifying = bfi
        .map(|b| b.interpretation == BFIInterpretation::Amplifying)
        .unwrap_or(false);

    if score > 0.9 && amplifying {
        FindingSeverity::Critical
    } else if score > 0.8 {
        FindingSeverity::High
    } else if score > 0.5 {
        FindingSeverity::Medium
    } else if score > 0.3 {
        FindingSeverity::Low
    } else {
        FindingSeverity::Info
    }
}

fn build_finding_description(stage: &str, score: f64, bfi: Option<&BFIResult>) -> String {
    let mut desc = format!(
        "Stage '{}' has a suspiciousness score of {:.4}, ",
        stage, score
    );
    if let Some(b) = bfi {
        desc.push_str(&format!(
            "with BFI = {:.3} ({}).",
            b.bfi_value, b.interpretation
        ));
        if b.interpretation == BFIInterpretation::Amplifying {
            desc.push_str(" This stage amplifies divergence introduced by transformations, making it a likely fault origin.");
        } else if b.interpretation == BFIInterpretation::Absorbing {
            desc.push_str(" Despite absorbing some divergence, high suspiciousness indicates it may still be faulty.");
        }
    } else {
        desc.push_str("with no BFI data available.");
    }
    desc
}

fn build_recommendations(findings: &[Finding]) -> Vec<Recommendation> {
    let mut recs = Vec::new();
    let critical: Vec<_> = findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Critical || f.severity == FindingSeverity::High)
        .collect();

    if !critical.is_empty() {
        recs.push(Recommendation {
            title: "Investigate high-suspiciousness stages".into(),
            description: format!(
                "Stages {} have high suspiciousness and should be investigated first.",
                critical
                    .iter()
                    .map(|f| f.stage_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            priority: RecommendationPriority::Immediate,
            related_findings: critical.iter().map(|f| f.id.clone()).collect(),
        });
    }

    if findings.len() > 1 {
        recs.push(Recommendation {
            title: "Check for cascading faults".into(),
            description: "Multiple stages flagged – verify whether a single root cause propagates through the pipeline.".into(),
            priority: RecommendationPriority::High,
            related_findings: findings.iter().map(|f| f.id.clone()).collect(),
        });
    }

    recs
}

fn compute_separation_ratio(scores: &[f64]) -> f64 {
    if scores.len() < 2 {
        return f64::INFINITY;
    }
    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let second = sorted[1].max(1e-9);
    sorted[0] / second
}

fn compute_exam_score(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let examined = scores.iter().filter(|&&s| s >= max).count();
    examined as f64 / scores.len() as f64
}

// ── Renderers (internal) ────────────────────────────────────────────────────

fn render_markdown(report: &LocalizationReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "# {}\n", report.header.title);
    let _ = writeln!(out, "**Pipeline:** {}  ", report.header.pipeline_name);
    let _ = writeln!(out, "**Generated:** {}  ", report.header.generated_at);
    let _ = writeln!(out, "**Version:** {}\n", report.header.version);

    if let Some(ref m) = report.methodology {
        out.push_str("## Methodology\n\n");
        let _ = writeln!(out, "- **SBFL metric:** {}", m.sbfl_metric);
        let _ = writeln!(out, "- **Distance metric:** {}", m.distance_metric);
        let _ = writeln!(out, "- **Test cases:** {} ({} violations)\n", m.test_count, m.violation_count);
    }

    out.push_str("## Findings\n\n");
    for f in &report.findings {
        let _ = writeln!(out, "### {} – {}\n", f.id, f.title);
        let _ = writeln!(out, "**Severity:** {} | **Suspiciousness:** {:.4}\n", f.severity, f.suspiciousness);
        let _ = writeln!(out, "{}\n", f.description);
    }

    if let Some(ref s) = report.statistical_section {
        out.push_str("## Statistics\n\n");
        let _ = writeln!(out, "| Stage | Suspiciousness | BFI |");
        let _ = writeln!(out, "|-------|----------------|-----|");
        for ((name, susp), bfi) in s.suspiciousness_scores.iter().zip(
            s.bfi_values
                .iter()
                .map(Some)
                .chain(std::iter::repeat(None)),
        ) {
            let bfi_str = bfi.map(|(_, v)| format!("{v:.3}")).unwrap_or_default();
            let _ = writeln!(out, "| {name} | {susp:.4} | {bfi_str} |");
        }
        let _ = writeln!(out, "\n**Trend:** {} | **Separation:** {:.2}\n", s.bfi_trend, s.separation_ratio);
    }

    if !report.recommendations.is_empty() {
        out.push_str("## Recommendations\n\n");
        for r in &report.recommendations {
            let _ = writeln!(out, "- **{}** ({:?}): {}", r.title, r.priority, r.description);
        }
    }
    out
}

fn render_html(report: &LocalizationReport) -> String {
    use std::fmt::Write;
    let mut out = String::from("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>");
    out.push_str(&report.header.title);
    out.push_str("</title><style>body{font-family:sans-serif;margin:2em}\
                   .finding{border:1px solid #ddd;padding:1em;margin:1em 0;border-radius:6px}\
                   table{border-collapse:collapse;width:100%}\
                   th,td{border:1px solid #ccc;padding:6px 10px}\
                   th{background:#f5f5f5}</style></head><body>");
    let _ = writeln!(out, "<h1>{}</h1>", report.header.title);
    for f in &report.findings {
        let _ = writeln!(
            out,
            "<div class=\"finding\"><h3>{} – {}</h3>\
             <p><b>Severity:</b> {} | <b>Suspiciousness:</b> {:.4}</p>\
             <p>{}</p></div>",
            f.id, f.title, f.severity, f.suspiciousness, f.description,
        );
    }
    out.push_str("</body></html>");
    out
}

fn render_plaintext(report: &LocalizationReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "=== {} ===\n", report.header.title);
    let _ = writeln!(out, "Pipeline: {}", report.header.pipeline_name);
    let _ = writeln!(out, "Generated: {}\n", report.header.generated_at);

    out.push_str("Findings:\n");
    for f in &report.findings {
        let _ = writeln!(
            out,
            "  [{id}] {title} (stage={stage}, severity={sev}, susp={susp:.4})",
            id = f.id,
            title = f.title,
            stage = f.stage_name,
            sev = f.severity,
            susp = f.suspiciousness,
        );
        let _ = writeln!(out, "    {}\n", f.description);
    }
    out
}

fn render_csv(report: &LocalizationReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    out.push_str("finding_id,stage,severity,suspiciousness,title\n");
    for f in &report.findings {
        let _ = writeln!(
            out,
            "{},{},{},{:.6},\"{}\"",
            f.id,
            f.stage_name,
            f.severity,
            f.suspiciousness,
            f.title.replace('"', "\"\""),
        );
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bfi::{BFIComputer, BFIResult};

    fn sample_bfi_results() -> Vec<BFIResult> {
        let c = BFIComputer::default();
        let names = vec!["tok".into(), "pos".into(), "ner".into()];
        let diffs = vec![
            vec![0.1, 0.2, 0.15],
            vec![0.5, 0.6, 0.55],
            vec![0.2, 0.25, 0.22],
        ];
        c.compute_all_bfi(&names, &diffs)
    }

    #[test]
    fn test_generate_basic() {
        let gen = ReportGenerator::new("test-pipeline");
        let stages = vec!["tok".into(), "pos".into(), "ner".into()];
        let susp = vec![0.2, 0.85, 0.45];
        let bfi = sample_bfi_results();
        let report = gen.generate(&stages, &susp, &bfi, 100, 30, &["passive".into()]);

        assert_eq!(report.header.pipeline_name, "test-pipeline");
        // Only stages above threshold (0.3) should be findings.
        assert_eq!(report.findings.len(), 2); // pos (0.85) and ner (0.45)
        assert_eq!(report.findings[0].stage_name, "pos");
    }

    #[test]
    fn test_finding_severity() {
        let gen = ReportGenerator::new("p");
        let stages = vec!["a".into(), "b".into()];
        let susp = vec![0.95, 0.35];
        let bfi = vec![
            BFIResult {
                stage_name: "a".into(),
                bfi_value: 3.0,
                interpretation: BFIInterpretation::Amplifying,
                confidence_interval: (2.0, 4.0),
                sample_count: 10,
            },
            BFIResult {
                stage_name: "b".into(),
                bfi_value: 0.5,
                interpretation: BFIInterpretation::Absorbing,
                confidence_interval: (0.3, 0.7),
                sample_count: 10,
            },
        ];
        let report = gen.generate(&stages, &susp, &bfi, 50, 20, &[]);
        assert_eq!(report.findings[0].severity, FindingSeverity::Critical);
        assert_eq!(report.findings[1].severity, FindingSeverity::Low);
    }

    #[test]
    fn test_methodology_section() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(
            &["s".into()],
            &[0.5],
            &[],
            42,
            10,
            &["passive".into(), "synonym".into()],
        );
        let m = report.methodology.as_ref().unwrap();
        assert_eq!(m.test_count, 42);
        assert_eq!(m.transformations_used.len(), 2);
    }

    #[test]
    fn test_statistical_section() {
        let gen = ReportGenerator::new("p");
        let bfi = sample_bfi_results();
        let report = gen.generate(
            &["tok".into(), "pos".into(), "ner".into()],
            &[0.3, 0.9, 0.5],
            &bfi,
            100,
            30,
            &[],
        );
        let s = report.statistical_section.as_ref().unwrap();
        assert_eq!(s.suspiciousness_scores.len(), 3);
        assert!(s.separation_ratio > 1.0);
    }

    #[test]
    fn test_recommendations_generated() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(
            &["a".into(), "b".into()],
            &[0.95, 0.5],
            &[],
            10,
            5,
            &[],
        );
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_render_json() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(&["s".into()], &[0.7], &[], 10, 5, &[]);
        let json = report.render(ReportFormat::Json);
        assert!(json.contains("\"title\""));
    }

    #[test]
    fn test_render_markdown() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(&["s".into()], &[0.7], &[], 10, 5, &[]);
        let md = report.render(ReportFormat::Markdown);
        assert!(md.contains("# Localization Report"));
    }

    #[test]
    fn test_render_plaintext() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(&["s".into()], &[0.7], &[], 10, 5, &[]);
        let txt = report.render(ReportFormat::PlainText);
        assert!(txt.contains("==="));
    }

    #[test]
    fn test_render_html() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(&["s".into()], &[0.7], &[], 10, 5, &[]);
        let html = report.render(ReportFormat::Html);
        assert!(html.contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_top_finding() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(
            &["a".into(), "b".into()],
            &[0.5, 0.9],
            &[],
            10,
            5,
            &[],
        );
        let top = report.top_finding().unwrap();
        assert_eq!(top.stage_name, "b");
    }

    #[test]
    fn test_findings_by_severity() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(
            &["a".into(), "b".into()],
            &[0.4, 0.95],
            &[],
            10,
            5,
            &[],
        );
        let high = report.findings_by_severity(FindingSeverity::High);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].stage_name, "b");
    }

    #[test]
    fn test_no_findings_below_threshold() {
        let gen = ReportGenerator::new("p");
        let report = gen.generate(&["a".into()], &[0.1], &[], 10, 0, &[]);
        assert!(report.findings.is_empty());
    }
}
