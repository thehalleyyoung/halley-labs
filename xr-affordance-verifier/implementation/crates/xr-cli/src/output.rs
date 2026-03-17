//! Output formatting for the XR Affordance Verifier CLI.
//!
//! Supports text, JSON, and compact output modes with optional ANSI colors.

use std::collections::HashMap;
use std::time::Duration;

use xr_lint::LintReport;
use xr_types::certificate::{CoverageCertificate, CertificateGrade};
use xr_types::config::VerifierConfig;
use xr_types::error::{Diagnostic, Severity};
use xr_types::scene::SceneModel;

use crate::pipeline::PipelineResult;
use crate::OutputFormat;

// ─── ANSI color codes ──────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const WHITE: &str = "\x1b[37m";
const BOLD_RED: &str = "\x1b[1;31m";
const BOLD_GREEN: &str = "\x1b[1;32m";
const BOLD_YELLOW: &str = "\x1b[1;33m";
const BOLD_BLUE: &str = "\x1b[1;34m";
const BOLD_CYAN: &str = "\x1b[1;36m";

/// Output formatter supporting multiple output modes and optional color.
pub struct OutputFormatter {
    format: OutputFormat,
    color: bool,
}

impl OutputFormatter {
    pub fn new(format: OutputFormat, color: bool) -> Self {
        Self { format, color }
    }

    fn c<'a>(&self, code: &'a str) -> &'a str {
        if self.color { code } else { "" }
    }

    // ─── Status indicators ─────────────────────────────────────────────

    fn pass_indicator(&self) -> String {
        format!("{}✓ PASS{}", self.c(BOLD_GREEN), self.c(RESET))
    }

    fn fail_indicator(&self) -> String {
        format!("{}✗ FAIL{}", self.c(BOLD_RED), self.c(RESET))
    }

    fn warn_indicator(&self) -> String {
        format!("{}⚠ WARN{}", self.c(BOLD_YELLOW), self.c(RESET))
    }

    fn info_indicator(&self) -> String {
        format!("{}ℹ INFO{}", self.c(BOLD_BLUE), self.c(RESET))
    }

    fn severity_indicator(&self, severity: &Severity) -> String {
        match severity {
            Severity::Error | Severity::Critical => self.fail_indicator(),
            Severity::Warning => self.warn_indicator(),
            Severity::Info => self.info_indicator(),
        }
    }

    fn grade_indicator(&self, grade: &CertificateGrade) -> String {
        match grade {
            CertificateGrade::Full => {
                format!("{}★ FULL{}", self.c(BOLD_GREEN), self.c(RESET))
            }
            CertificateGrade::Partial => {
                format!("{}◆ PARTIAL{}", self.c(BOLD_YELLOW), self.c(RESET))
            }
            CertificateGrade::Weak => {
                format!("{}○ WEAK{}", self.c(BOLD_RED), self.c(RESET))
            }
        }
    }

    // ─── Timing ────────────────────────────────────────────────────────

    fn format_duration(&self, d: Duration) -> String {
        if d.as_secs() >= 60 {
            format!("{}m {:.1}s", d.as_secs() / 60, d.as_secs_f64() % 60.0)
        } else if d.as_millis() >= 1000 {
            format!("{:.2}s", d.as_secs_f64())
        } else {
            format!("{}ms", d.as_millis())
        }
    }

    // ─── Table formatting ──────────────────────────────────────────────

    fn format_table(&self, headers: &[&str], rows: &[Vec<String>]) -> String {
        let mut col_widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < col_widths.len() {
                    col_widths[i] = col_widths[i].max(strip_ansi(cell).len());
                }
            }
        }

        let mut out = String::new();

        // Header
        out.push_str(&self.c(BOLD).to_string());
        for (i, header) in headers.iter().enumerate() {
            if i > 0 {
                out.push_str(" │ ");
            }
            out.push_str(&format!("{:<width$}", header, width = col_widths[i]));
        }
        out.push_str(self.c(RESET));
        out.push('\n');

        // Separator
        for (i, &w) in col_widths.iter().enumerate() {
            if i > 0 {
                out.push_str("─┼─");
            }
            out.push_str(&"─".repeat(w));
        }
        out.push('\n');

        // Rows
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i > 0 {
                    out.push_str(" │ ");
                }
                let visible_len = strip_ansi(cell).len();
                let padding = col_widths.get(i).copied().unwrap_or(0).saturating_sub(visible_len);
                out.push_str(cell);
                out.push_str(&" ".repeat(padding));
            }
            out.push('\n');
        }

        out
    }

    // ─── Lint report formatting ────────────────────────────────────────

    pub fn format_lint_report(&self, report: &LintReport, elapsed: Duration) -> String {
        match self.format {
            OutputFormat::Json => self.format_lint_report_json(report, elapsed),
            OutputFormat::Compact => self.format_lint_report_compact(report),
            OutputFormat::Text => self.format_lint_report_text(report, elapsed),
        }
    }

    fn format_lint_report_json(&self, report: &LintReport, elapsed: Duration) -> String {
        let json = serde_json::json!({
            "type": "lint_report",
            "scene": report.scene_name,
            "elements_checked": report.elements_checked,
            "rules_applied": report.rules_applied,
            "findings": report.findings.iter().map(|f| {
                serde_json::json!({
                    "rule": f.rule.to_string(),
                    "severity": format!("{:?}", f.severity),
                    "element": f.element_name,
                    "message": f.message,
                    "suggestion": f.suggestion,
                })
            }).collect::<Vec<_>>(),
            "errors": report.errors().len(),
            "warnings": report.warnings().len(),
            "elapsed_ms": elapsed.as_millis(),
        });
        serde_json::to_string_pretty(&json).unwrap_or_default()
    }

    fn format_lint_report_compact(&self, report: &LintReport) -> String {
        let mut out = String::new();
        for f in &report.findings {
            let sev = match f.severity {
                Severity::Error | Severity::Critical => "E",
                Severity::Warning => "W",
                Severity::Info => "I",
            };
            let elem = f.element_name.as_deref().unwrap_or("-");
            out.push_str(&format!("[{}] {} {}: {}\n", sev, f.rule, elem, f.message));
        }
        out.push_str(&format!(
            "total: {} errors, {} warnings\n",
            report.errors().len(),
            report.warnings().len()
        ));
        out
    }

    fn format_lint_report_text(&self, report: &LintReport, elapsed: Duration) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "\n{}═══ Lint Report: {} ═══{}\n\n",
            self.c(BOLD_CYAN),
            report.scene_name,
            self.c(RESET)
        ));

        if report.findings.is_empty() {
            out.push_str(&format!(
                "  {} No issues found.\n\n",
                self.pass_indicator()
            ));
        } else {
            let mut rows = Vec::new();
            for f in &report.findings {
                rows.push(vec![
                    self.severity_indicator(&f.severity),
                    format!(
                        "{}{}{}",
                        self.c(BOLD),
                        f.rule,
                        self.c(RESET)
                    ),
                    f.element_name.clone().unwrap_or_else(|| "-".into()),
                    f.message.clone(),
                ]);
            }
            out.push_str(&self.format_table(
                &["Status", "Rule", "Element", "Message"],
                &rows,
            ));

            // Suggestions
            let suggestions: Vec<_> = report
                .findings
                .iter()
                .filter_map(|f| {
                    f.suggestion.as_ref().map(|s| {
                        format!(
                            "  {}→{} {}: {}",
                            self.c(DIM),
                            self.c(RESET),
                            f.element_name.as_deref().unwrap_or("scene"),
                            s
                        )
                    })
                })
                .collect();

            if !suggestions.is_empty() {
                out.push_str(&format!(
                    "\n{}Suggestions:{}\n",
                    self.c(BOLD),
                    self.c(RESET)
                ));
                for s in &suggestions {
                    out.push_str(s);
                    out.push('\n');
                }
            }
        }

        out.push_str(&format!(
            "\n{}Summary:{} {} elements checked, {} rules applied, {} errors, {} warnings\n",
            self.c(BOLD),
            self.c(RESET),
            report.elements_checked,
            report.rules_applied,
            report.errors().len(),
            report.warnings().len(),
        ));

        out.push_str(&format!(
            "{}Time:{} {}\n",
            self.c(DIM),
            self.c(RESET),
            self.format_duration(elapsed)
        ));

        out
    }

    // ─── Pipeline result formatting ────────────────────────────────────

    pub fn format_pipeline_result(
        &self,
        result: &PipelineResult,
        elapsed: Duration,
    ) -> String {
        match self.format {
            OutputFormat::Json => self.format_pipeline_result_json(result, elapsed),
            OutputFormat::Compact => self.format_pipeline_result_compact(result),
            OutputFormat::Text => self.format_pipeline_result_text(result, elapsed),
        }
    }

    fn format_pipeline_result_json(
        &self,
        result: &PipelineResult,
        elapsed: Duration,
    ) -> String {
        let json = serde_json::json!({
            "type": "verification_result",
            "stage": format!("{:?}", result.final_stage),
            "lint_errors": result.lint_findings.iter()
                .filter(|f| matches!(f.severity, Severity::Error | Severity::Critical))
                .count(),
            "lint_warnings": result.lint_findings.iter()
                .filter(|f| matches!(f.severity, Severity::Warning))
                .count(),
            "samples_total": result.sample_verdicts.len(),
            "samples_pass": result.sample_verdicts.iter().filter(|s| s.is_pass()).count(),
            "verified_regions": result.verified_regions.len(),
            "violations": result.violations.len(),
            "epsilon_analytical": result.epsilon_analytical,
            "has_errors": result.has_errors(),
            "elapsed_ms": elapsed.as_millis(),
            "stage_timings": result.stage_timings.iter()
                .map(|(k, v)| (format!("{k:?}"), format!("{:.1}ms", v.as_millis())))
                .collect::<HashMap<_, _>>(),
        });
        serde_json::to_string_pretty(&json).unwrap_or_default()
    }

    fn format_pipeline_result_compact(&self, result: &PipelineResult) -> String {
        let pass = result.sample_verdicts.iter().filter(|s| s.is_pass()).count();
        let total = result.sample_verdicts.len();
        let status = if result.has_errors() { "FAIL" } else { "PASS" };
        format!(
            "[{}] samples={}/{} regions={} violations={} lint_errors={}\n",
            status,
            pass,
            total,
            result.verified_regions.len(),
            result.violations.len(),
            result.error_count(),
        )
    }

    fn format_pipeline_result_text(
        &self,
        result: &PipelineResult,
        elapsed: Duration,
    ) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "\n{}═══ Verification Results ═══{}\n\n",
            self.c(BOLD_CYAN),
            self.c(RESET),
        ));

        // Overall status
        let status = if result.has_errors() {
            self.fail_indicator()
        } else {
            self.pass_indicator()
        };
        out.push_str(&format!("  Overall: {}\n\n", status));

        // Lint summary
        let lint_errors = result.lint_findings.iter()
            .filter(|f| matches!(f.severity, Severity::Error | Severity::Critical))
            .count();
        let lint_warnings = result.lint_findings.iter()
            .filter(|f| matches!(f.severity, Severity::Warning))
            .count();
        out.push_str(&format!(
            "  {}Tier 1 Linting:{} {} errors, {} warnings\n",
            self.c(BOLD),
            self.c(RESET),
            lint_errors,
            lint_warnings,
        ));

        // Sampling summary
        let pass = result.sample_verdicts.iter().filter(|s| s.is_pass()).count();
        let total = result.sample_verdicts.len();
        if total > 0 {
            let pct = (pass as f64 / total as f64) * 100.0;
            out.push_str(&format!(
                "  {}Sampling:{} {}/{} pass ({:.1}%)\n",
                self.c(BOLD),
                self.c(RESET),
                pass,
                total,
                pct,
            ));
        }

        // SMT summary
        if !result.verified_regions.is_empty() {
            out.push_str(&format!(
                "  {}SMT Verification:{} {} regions verified\n",
                self.c(BOLD),
                self.c(RESET),
                result.verified_regions.len(),
            ));
        }

        // Violations
        if !result.violations.is_empty() {
            out.push_str(&format!(
                "\n  {}Violations:{}\n",
                self.c(BOLD_RED),
                self.c(RESET),
            ));
            for v in &result.violations {
                out.push_str(&format!(
                    "    {} {}: {} ({} samples)\n",
                    self.fail_indicator(),
                    v.element_id,
                    v.description,
                    v.sample_count(),
                ));
            }
        }

        // Stage timings
        if !result.stage_timings.is_empty() {
            out.push_str(&format!(
                "\n  {}Stage Timings:{}\n",
                self.c(DIM),
                self.c(RESET),
            ));
            for (stage, duration) in &result.stage_timings {
                out.push_str(&format!(
                    "    {:?}: {}\n",
                    stage,
                    self.format_duration(*duration),
                ));
            }
        }

        out.push_str(&format!(
            "\n{}Total time:{} {}\n",
            self.c(DIM),
            self.c(RESET),
            self.format_duration(elapsed),
        ));

        out
    }

    // ─── Certificate summary ───────────────────────────────────────────

    pub fn format_certificate_summary(
        &self,
        cert: &CoverageCertificate,
        elapsed: Duration,
    ) -> String {
        match self.format {
            OutputFormat::Json => {
                let json = serde_json::json!({
                    "type": "certificate_summary",
                    "id": cert.id.to_string(),
                    "grade": format!("{:?}", cert.grade),
                    "kappa": cert.kappa,
                    "epsilon_analytical": cert.epsilon_analytical,
                    "epsilon_estimated": cert.epsilon_estimated,
                    "delta": cert.delta,
                    "samples": cert.samples.len(),
                    "verified_regions": cert.verified_regions.len(),
                    "violations": cert.violations.len(),
                    "elapsed_ms": elapsed.as_millis(),
                });
                serde_json::to_string_pretty(&json).unwrap_or_default()
            }
            OutputFormat::Compact => {
                format!(
                    "cert={} grade={:?} κ={:.4} ε_a={:.6} samples={}\n",
                    cert.id, cert.grade, cert.kappa,
                    cert.epsilon_analytical, cert.samples.len(),
                )
            }
            OutputFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!(
                    "\n{}═══ Coverage Certificate ═══{}\n\n",
                    self.c(BOLD_CYAN),
                    self.c(RESET),
                ));
                out.push_str(&format!(
                    "  {}ID:{} {}\n",
                    self.c(BOLD), self.c(RESET), cert.id,
                ));
                out.push_str(&format!(
                    "  {}Grade:{} {}\n",
                    self.c(BOLD), self.c(RESET), self.grade_indicator(&cert.grade),
                ));
                out.push_str(&format!(
                    "  {}Coverage (κ):{} {:.4}\n",
                    self.c(BOLD), self.c(RESET), cert.kappa,
                ));
                out.push_str(&format!(
                    "  {}ε analytical:{} {:.6}\n",
                    self.c(BOLD), self.c(RESET), cert.epsilon_analytical,
                ));
                out.push_str(&format!(
                    "  {}ε estimated: {} {:.6}\n",
                    self.c(BOLD), self.c(RESET), cert.epsilon_estimated,
                ));
                out.push_str(&format!(
                    "  {}δ (confidence):{} {:.4}\n",
                    self.c(BOLD), self.c(RESET), cert.delta,
                ));
                out.push_str(&format!(
                    "  {}Samples:{} {} ({} pass)\n",
                    self.c(BOLD), self.c(RESET),
                    cert.samples.len(),
                    cert.samples.iter().filter(|s| s.is_pass()).count(),
                ));
                out.push_str(&format!(
                    "  {}Verified regions:{} {}\n",
                    self.c(BOLD), self.c(RESET), cert.verified_regions.len(),
                ));
                out.push_str(&format!(
                    "  {}Violations:{} {}\n",
                    self.c(BOLD), self.c(RESET), cert.violations.len(),
                ));
                out.push_str(&format!(
                    "\n{}Time:{} {}\n",
                    self.c(DIM), self.c(RESET),
                    self.format_duration(elapsed),
                ));
                out
            }
        }
    }

    // ─── Full report ───────────────────────────────────────────────────

    pub fn format_full_report(
        &self,
        cert: &CoverageCertificate,
        elapsed: Duration,
    ) -> String {
        let mut out = self.format_certificate_summary(cert, elapsed);

        if self.format == OutputFormat::Text {
            // Per-element coverage table
            if !cert.element_coverage.is_empty() {
                out.push_str(&format!(
                    "\n{}── Element Coverage ──{}\n\n",
                    self.c(BOLD), self.c(RESET),
                ));

                let mut rows = Vec::new();
                let mut sorted_elements: Vec<_> = cert.element_coverage.iter().collect();
                sorted_elements.sort_by(|a, b| a.0.cmp(b.0));

                for (element_id, coverage) in sorted_elements {
                    let status = if *coverage >= 0.95 {
                        self.pass_indicator()
                    } else if *coverage >= 0.80 {
                        self.warn_indicator()
                    } else {
                        self.fail_indicator()
                    };
                    rows.push(vec![
                        element_id.to_string(),
                        format!("{:.2}%", coverage * 100.0),
                        status,
                    ]);
                }

                out.push_str(&self.format_table(
                    &["Element", "Coverage", "Status"],
                    &rows,
                ));
            }

            // Violations detail
            if !cert.violations.is_empty() {
                out.push_str(&format!(
                    "\n{}── Violations ──{}\n\n",
                    self.c(BOLD_RED), self.c(RESET),
                ));
                for v in &cert.violations {
                    out.push_str(&format!(
                        "  {} {}{}{}: {}\n",
                        self.fail_indicator(),
                        self.c(BOLD),
                        v.element_id,
                        self.c(RESET),
                        v.description,
                    ));
                    out.push_str(&format!(
                        "    Severity: {:?}, Samples: {}, Est. measure: {:.6}\n",
                        v.severity, v.sample_count(), v.estimated_measure,
                    ));
                }
            }
        }

        out
    }

        /// Generate a polished self-contained HTML report from a certificate.
        pub fn generate_html_report(&self, cert: &CoverageCertificate) -> String {
                let summary = cert.summary();
                let grade_class = match cert.grade {
                        CertificateGrade::Full => "full",
                        CertificateGrade::Partial => "partial",
                        CertificateGrade::Weak => "weak",
                };
                let pass_count = cert.samples.iter().filter(|s| s.is_pass()).count();
                let fail_count = cert.samples.len().saturating_sub(pass_count);

                let mut coverage_rows: Vec<_> = cert.element_coverage.iter().collect();
                coverage_rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                let coverage_html = coverage_rows
                        .iter()
                        .map(|(id, cov)| {
                                let class = if **cov >= 0.95 { "pass" } else if **cov >= 0.80 { "warn" } else { "fail" };
                                format!(
                                        r#"<div class="row-item">
    <div class="row-head"><span class="mono">{}</span><span class="badge {}">{:.1}%</span></div>
    <div class="bar"><div class="fill {}" style="width:{:.2}%"></div></div>
</div>"#,
                                        escape_html(&id.to_string()),
                                        class,
                                        *cov * 100.0,
                                        class,
                                        *cov * 100.0,
                                )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                let violations_html = if cert.violations.is_empty() {
                        "<div class=\"empty\">No violation surfaces remain in this certificate.</div>".into()
                } else {
                        cert.violations
                                .iter()
                                .map(|v| {
                                        format!(
                                                r#"<div class="violation">
    <strong>{}</strong>
    <div class="muted">{}</div>
    <div class="tiny muted">Severity: {:?} · Samples: {} · Measure: {:.6}</div>
</div>"#,
                                                escape_html(&v.element_id.to_string()),
                                                escape_html(&v.description),
                                                v.severity,
                                                v.sample_count(),
                                                v.estimated_measure,
                                        )
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                };

                format!(
                        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>XR Coverage Certificate</title>
    <style>
        :root {{ color-scheme: dark; --bg:#06111e; --panel:#0d1728; --line:rgba(148,163,184,.15); --text:#e7eef9; --muted:#9db0cb; --green:#22c55e; --yellow:#f59e0b; --red:#ef4444; --blue:#38bdf8; }}
        * {{ box-sizing:border-box; }}
        body {{ margin:0; padding:28px; font-family:Inter,system-ui,sans-serif; background:linear-gradient(180deg,#06111e,#030712); color:var(--text); }}
        .shell {{ max-width:1180px; margin:0 auto; display:grid; gap:20px; }}
        .hero,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:24px; box-shadow:0 20px 60px rgba(0,0,0,.3); }}
        .hero {{ padding:26px 28px; }}
        .eyebrow {{ color:var(--blue); text-transform:uppercase; letter-spacing:.16em; font-size:12px; font-weight:800; margin-bottom:10px; }}
        h1 {{ margin:0; font-size:40px; }}
        .muted {{ color:var(--muted); }}
        .tiny {{ font-size:12px; }}
        .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; }}
        .card {{ background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.06); border-radius:18px; padding:16px; }}
        .label {{ font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); }}
        .value {{ font-size:30px; font-weight:800; margin-top:10px; }}
        .panel {{ padding:20px; }}
        .columns {{ display:grid; grid-template-columns:1.1fr .9fr; gap:20px; }}
        .badge {{ border-radius:999px; padding:5px 10px; font-size:12px; font-weight:800; }}
        .badge.pass {{ background:rgba(34,197,94,.15); color:#86efac; }}
        .badge.warn {{ background:rgba(245,158,11,.15); color:#fde68a; }}
        .badge.fail {{ background:rgba(239,68,68,.15); color:#fca5a5; }}
        .grade {{ display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:8px 12px; margin-top:14px; font-weight:800; }}
        .grade.full {{ background:rgba(34,197,94,.16); color:#86efac; }}
        .grade.partial {{ background:rgba(245,158,11,.16); color:#fde68a; }}
        .grade.weak {{ background:rgba(239,68,68,.16); color:#fca5a5; }}
        .row-item {{ margin-bottom:14px; }}
        .row-head {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:8px; }}
        .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
        .bar {{ height:10px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; }}
        .fill {{ height:100%; border-radius:inherit; }}
        .fill.pass {{ background:linear-gradient(90deg,#16a34a,#22c55e); }}
        .fill.warn {{ background:linear-gradient(90deg,#d97706,#f59e0b); }}
        .fill.fail {{ background:linear-gradient(90deg,#dc2626,#ef4444); }}
        .violation {{ border-left:3px solid var(--red); padding-left:12px; margin-bottom:12px; }}
        .empty {{ color:var(--muted); font-style:italic; padding:10px 0; }}
        @media (max-width: 980px) {{ .grid, .columns {{ grid-template-columns:1fr; }} }}
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div class="eyebrow">XR Affordance Verifier • HTML Report</div>
            <h1>Coverage Certificate</h1>
            <div class="muted">Certificate ID: <span class="mono">{}</span></div>
            <div class="grade {}">Grade: {}</div>
        </section>
        <section class="grid">
            <div class="card"><div class="label">Coverage κ</div><div class="value">{:.1}%</div><div class="tiny muted">Min element coverage: {:.1}%</div></div>
            <div class="card"><div class="label">Samples</div><div class="value">{}</div><div class="tiny muted">{} pass · {} fail</div></div>
            <div class="card"><div class="label">Verified regions</div><div class="value">{}</div><div class="tiny muted">Violation surfaces: {}</div></div>
            <div class="card"><div class="label">Uncertainty</div><div class="value">εₑ {:.4}</div><div class="tiny muted">εₐ {:.6} · δ {:.4}</div></div>
        </section>
        <section class="panel columns">
            <div>
                <h2>Element Coverage</h2>
                <div class="muted tiny">Sorted from strongest to weakest affordance coverage.</div>
                <div style="margin-top:16px">{}</div>
            </div>
            <div>
                <h2>Violations</h2>
                <div class="muted tiny">Use these callouts to focus remediation or explain known limitations in a demo.</div>
                <div style="margin-top:16px">{}</div>
            </div>
        </section>
    </div>
</body>
</html>"##,
                        escape_html(&cert.id.to_string()),
                        grade_class,
                        escape_html(&format!("{:?}", cert.grade)),
                        cert.kappa * 100.0,
                        cert.min_element_coverage() * 100.0,
                        summary.num_samples,
                        pass_count,
                        fail_count,
                        summary.num_verified_regions,
                        summary.num_violations,
                        cert.epsilon_estimated,
                        cert.epsilon_analytical,
                        cert.delta,
                        coverage_html,
                        violations_html,
                )
        }

    // ─── Inspection formatting ─────────────────────────────────────────

    pub fn format_inspection(
        &self,
        scene: &SceneModel,
        diagnostics: &[Diagnostic],
        show_elements: bool,
        show_deps: bool,
        show_devices: bool,
        elapsed: Duration,
    ) -> String {
        match self.format {
            OutputFormat::Json => {
                let json = serde_json::json!({
                    "type": "inspection",
                    "scene_name": scene.name,
                    "description": scene.description,
                    "version": scene.version,
                    "num_elements": scene.elements.len(),
                    "num_dependencies": scene.dependencies.len(),
                    "num_devices": scene.devices.len(),
                    "bounds": {
                        "min": scene.bounds.min,
                        "max": scene.bounds.max,
                    },
                    "is_dag": scene.is_dag(),
                    "max_depth": scene.max_interaction_depth(),
                    "diagnostics": diagnostics.len(),
                    "elapsed_ms": elapsed.as_millis(),
                });
                serde_json::to_string_pretty(&json).unwrap_or_default()
            }
            OutputFormat::Compact => {
                format!(
                    "scene={} elements={} deps={} devices={} dag={}\n",
                    scene.name,
                    scene.elements.len(),
                    scene.dependencies.len(),
                    scene.devices.len(),
                    scene.is_dag(),
                )
            }
            OutputFormat::Text => {
                let mut out = String::new();

                out.push_str(&format!(
                    "\n{}═══ Scene Inspection: {} ═══{}\n\n",
                    self.c(BOLD_CYAN), scene.name, self.c(RESET),
                ));

                out.push_str(&format!(
                    "  {}Description:{} {}\n",
                    self.c(BOLD), self.c(RESET), scene.description,
                ));
                out.push_str(&format!(
                    "  {}Version:{} {}\n",
                    self.c(BOLD), self.c(RESET), scene.version,
                ));
                out.push_str(&format!(
                    "  {}Elements:{} {}\n",
                    self.c(BOLD), self.c(RESET), scene.elements.len(),
                ));
                out.push_str(&format!(
                    "  {}Dependencies:{} {}\n",
                    self.c(BOLD), self.c(RESET), scene.dependencies.len(),
                ));
                out.push_str(&format!(
                    "  {}Devices:{} {}\n",
                    self.c(BOLD), self.c(RESET), scene.devices.len(),
                ));
                out.push_str(&format!(
                    "  {}Bounds:{} min={:?} max={:?}\n",
                    self.c(BOLD), self.c(RESET), scene.bounds.min, scene.bounds.max,
                ));
                out.push_str(&format!(
                    "  {}DAG:{} {}  {}Max Depth:{} {}\n",
                    self.c(BOLD), self.c(RESET), scene.is_dag(),
                    self.c(BOLD), self.c(RESET), scene.max_interaction_depth(),
                ));

                // Interaction type breakdown
                let count_map = scene.count_by_type();
                if !count_map.is_empty() {
                    out.push_str(&format!(
                        "\n  {}Interaction Types:{}\n",
                        self.c(BOLD), self.c(RESET),
                    ));
                    for (itype, count) in &count_map {
                        out.push_str(&format!("    {:?}: {}\n", itype, count));
                    }
                }

                if show_elements {
                    out.push_str(&format!(
                        "\n{}── Elements ──{}\n\n",
                        self.c(BOLD), self.c(RESET),
                    ));

                    let mut rows = Vec::new();
                    for e in &scene.elements {
                        rows.push(vec![
                            e.name.clone(),
                            format!("{:?}", e.interaction_type),
                            format!(
                                "({:.2}, {:.2}, {:.2})",
                                e.position[0], e.position[1], e.position[2]
                            ),
                            format!("{:?}", e.actuator),
                            e.visual.label.clone().unwrap_or_else(|| "-".into()),
                        ]);
                    }
                    out.push_str(&self.format_table(
                        &["Name", "Interaction", "Position", "Actuator", "Label"],
                        &rows,
                    ));
                }

                if show_deps && !scene.dependencies.is_empty() {
                    out.push_str(&format!(
                        "\n{}── Dependencies ──{}\n\n",
                        self.c(BOLD), self.c(RESET),
                    ));
                    for dep in &scene.dependencies {
                        let src = scene
                            .elements
                            .get(dep.source_index)
                            .map(|e| e.name.as_str())
                            .unwrap_or("?");
                        let tgt = scene
                            .elements
                            .get(dep.target_index)
                            .map(|e| e.name.as_str())
                            .unwrap_or("?");
                        out.push_str(&format!(
                            "  {} {}→{} {} ({:?})\n",
                            src, self.c(DIM), self.c(RESET), tgt, dep.dependency_type,
                        ));
                    }
                }

                if show_devices && !scene.devices.is_empty() {
                    out.push_str(&format!(
                        "\n{}── Devices ──{}\n\n",
                        self.c(BOLD), self.c(RESET),
                    ));
                    for d in &scene.devices {
                        out.push_str(&format!(
                            "  {}{}{}  type={:?}  tracking_precision={:.4}\n",
                            self.c(BOLD), d.name, self.c(RESET),
                            d.device_type, d.tracking_precision,
                        ));
                    }
                }

                if !diagnostics.is_empty() {
                    out.push_str(&format!(
                        "\n{}── Diagnostics ──{}\n\n",
                        self.c(BOLD_YELLOW), self.c(RESET),
                    ));
                    for d in diagnostics {
                        out.push_str(&format!(
                            "  {} [{}] {}\n",
                            self.severity_indicator(&d.severity),
                            d.code,
                            d.message,
                        ));
                    }
                }

                out.push_str(&format!(
                    "\n{}Time:{} {}\n",
                    self.c(DIM), self.c(RESET), self.format_duration(elapsed),
                ));

                out
            }
        }
    }

    // ─── Config formatting ─────────────────────────────────────────────

    pub fn format_config(&self, config: &VerifierConfig) -> String {
        match self.format {
            OutputFormat::Json => {
                config.to_json().unwrap_or_else(|_| "{}".into())
            }
            OutputFormat::Compact => {
                format!(
                    "samples={} smt_timeout={}s tier1={} tier2={}\n",
                    config.sampling.num_samples,
                    config.smt.timeout_s,
                    config.tier1.enabled,
                    config.tier2.enabled,
                )
            }
            OutputFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!(
                    "\n{}═══ Verifier Configuration ═══{}\n\n",
                    self.c(BOLD_CYAN), self.c(RESET),
                ));
                out.push_str(&format!(
                    "  {}Name:{} {}\n",
                    self.c(BOLD), self.c(RESET), config.name,
                ));
                out.push_str(&format!(
                    "\n  {}Tier 1 (Linting):{}\n",
                    self.c(BOLD), self.c(RESET),
                ));
                out.push_str(&format!(
                    "    Enabled: {}  Max time: {}s  Workers: {}\n",
                    config.tier1.enabled, config.tier1.max_time_s, config.tier1.num_workers,
                ));
                out.push_str(&format!(
                    "\n  {}Tier 2 (SMT):{}\n",
                    self.c(BOLD), self.c(RESET),
                ));
                out.push_str(&format!(
                    "    Enabled: {}  Max time: {}s  Max subdivisions: {}\n",
                    config.tier2.enabled, config.tier2.max_time_s, config.tier2.max_subdivisions,
                ));
                out.push_str(&format!(
                    "\n  {}Sampling:{}\n",
                    self.c(BOLD), self.c(RESET),
                ));
                out.push_str(&format!(
                    "    Samples: {}  Strata/dim: {}  δ: {}\n",
                    config.sampling.num_samples, config.sampling.strata_per_dim,
                    config.sampling.confidence_delta,
                ));
                out.push_str(&format!(
                    "    Stratified: {}  Latin Hypercube: {}  Seed: {}\n",
                    config.sampling.use_stratified,
                    config.sampling.use_latin_hypercube,
                    config.sampling.seed,
                ));
                out.push_str(&format!(
                    "\n  {}SMT:{}\n",
                    self.c(BOLD), self.c(RESET),
                ));
                out.push_str(&format!(
                    "    Timeout: {}s  Logic: {}  Incremental: {}\n",
                    config.smt.timeout_s, config.smt.logic, config.smt.incremental,
                ));
                out.push_str(&format!(
                    "\n  {}Population:{}\n",
                    self.c(BOLD), self.c(RESET),
                ));
                out.push_str(&format!(
                    "    Range: {}th–{}th percentile  Seated: {}  Standing: {}\n",
                    (config.population.percentile_low * 100.0) as u32,
                    (config.population.percentile_high * 100.0) as u32,
                    config.population.include_seated,
                    config.population.include_standing,
                ));
                out
            }
        }
    }

    // ─── SVG report generation ─────────────────────────────────────────

    pub fn generate_svg_report(&self, cert: &CoverageCertificate) -> String {
        let summary = cert.summary();
        let width = 600;
        let height = 400;
        let bar_height = 30;
        let bar_width = 400;
        let bar_x = 150;
        let bar_y = 100;

        let kappa_width = (cert.kappa * bar_width as f64) as u32;
        let kappa_color = match cert.grade {
            CertificateGrade::Full => "#22c55e",
            CertificateGrade::Partial => "#eab308",
            CertificateGrade::Weak => "#ef4444",
        };

        let grade_text = match cert.grade {
            CertificateGrade::Full => "FULL",
            CertificateGrade::Partial => "PARTIAL",
            CertificateGrade::Weak => "WEAK",
        };

        let pass_count = cert.samples.iter().filter(|s| s.is_pass()).count();
        let total_count = cert.samples.len();
        let pass_pct = if total_count > 0 {
            (pass_count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };

        // Element coverage bars
        let mut element_bars = String::new();
        let mut sorted_elements: Vec<_> = cert.element_coverage.iter().collect();
        sorted_elements.sort_by(|a, b| a.0.cmp(b.0));

        for (i, (elem_id, coverage)) in sorted_elements.iter().enumerate().take(10) {
            let y = bar_y + 80 + (i as u32) * 35;
            let w = (*coverage * bar_width as f64) as u32;
            let elem_color = if **coverage >= 0.95 {
                "#22c55e"
            } else if **coverage >= 0.80 {
                "#eab308"
            } else {
                "#ef4444"
            };
            let short_id = &elem_id.to_string()[..8];
            element_bars.push_str(&format!(
                r##"    <text x="10" y="{}" font-size="11" fill="#666">{}</text>
    <rect x="{}" y="{}" width="{}" height="20" fill="#e5e7eb" rx="3"/>
    <rect x="{}" y="{}" width="{}" height="20" fill="{}" rx="3"/>
    <text x="{}" y="{}" font-size="11" fill="#333">{:.1}%</text>
"##,
                y + 15,
                short_id,
                bar_x,
                y,
                bar_width,
                bar_x,
                y,
                w,
                elem_color,
                bar_x + bar_width + 10,
                y + 15,
                *coverage * 100.0,
            ));
        }

        let total_height = height + (sorted_elements.len().min(10) as u32) * 35;

        format!(
            r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {total_height}">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  </style>

  <!-- Title -->
  <text x="{}" y="30" font-size="18" font-weight="bold" fill="#1f2937">Coverage Certificate</text>
  <text x="{}" y="50" font-size="12" fill="#6b7280">ID: {}</text>
  <text x="{}" y="65" font-size="12" fill="#6b7280">Grade: {grade_text}</text>

  <!-- Kappa bar -->
  <text x="10" y="{}" font-size="13" font-weight="bold" fill="#374151">Coverage (κ)</text>
  <rect x="{bar_x}" y="{}" width="{bar_width}" height="{bar_height}" fill="#e5e7eb" rx="5"/>
  <rect x="{bar_x}" y="{}" width="{kappa_width}" height="{bar_height}" fill="{kappa_color}" rx="5"/>
  <text x="{}" y="{}" font-size="14" font-weight="bold" fill="#1f2937">{:.2}%</text>

  <!-- Stats -->
  <text x="10" y="{}" font-size="12" fill="#374151">Samples: {total_count} ({pass_pct:.1}% pass)</text>
  <text x="300" y="{}" font-size="12" fill="#374151">Regions: {} | Violations: {}</text>
  <text x="10" y="{}" font-size="12" fill="#374151">ε_a: {:.6}  ε_e: {:.6}  δ: {:.4}</text>

  <!-- Element coverage -->
{element_bars}
</svg>"##,
            width / 2 - 100,
            width / 2 - 100,
            cert.id,
            width / 2 - 100,
            bar_y - 5,
            bar_y - bar_height,
            bar_y - bar_height,
            bar_x + bar_width + 10,
            bar_y - bar_height + 20,
            cert.kappa * 100.0,
            bar_y + 25,
            bar_y + 25,
            summary.num_verified_regions,
            summary.num_violations,
            bar_y + 45,
            cert.epsilon_analytical,
            cert.epsilon_estimated,
            cert.delta,
        )
    }
}

/// Strip ANSI escape codes from a string for width calculation.
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;
    for ch in s.chars() {
        if ch == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if ch == 'm' {
                in_escape = false;
            }
        } else {
            result.push(ch);
        }
    }
    result
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Progress bar for long operations.
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
    label: String,
    color: bool,
}

impl ProgressBar {
    pub fn new(total: usize, label: impl Into<String>, color: bool) -> Self {
        Self {
            total,
            current: 0,
            width: 40,
            label: label.into(),
            color,
        }
    }

    pub fn update(&mut self, current: usize) {
        self.current = current.min(self.total);
        self.render();
    }

    pub fn increment(&mut self) {
        self.current = (self.current + 1).min(self.total);
        self.render();
    }

    pub fn finish(&mut self) {
        self.current = self.total;
        self.render();
        eprintln!();
    }

    fn render(&self) {
        let pct = if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            0.0
        };
        let filled = (pct * self.width as f64) as usize;
        let empty = self.width - filled;

        let bar_color = if self.color { GREEN } else { "" };
        let dim = if self.color { DIM } else { "" };
        let reset = if self.color { RESET } else { "" };

        eprint!(
            "\r  {} [{}{}{}{}{}] {:>3.0}% ({}/{})",
            self.label,
            bar_color,
            "█".repeat(filled),
            dim,
            "░".repeat(empty),
            reset,
            pct * 100.0,
            self.current,
            self.total,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("\x1b[1;31mhello\x1b[0m"), "hello");
        assert_eq!(strip_ansi("no escape"), "no escape");
        assert_eq!(strip_ansi(""), "");
    }

    #[test]
    fn test_output_formatter_text() {
        let f = OutputFormatter::new(OutputFormat::Text, false);
        assert_eq!(f.c(RED), "");
        assert_eq!(f.c(BOLD), "");
    }

    #[test]
    fn test_output_formatter_color() {
        let f = OutputFormatter::new(OutputFormat::Text, true);
        assert_eq!(f.c(RED), RED);
        assert_eq!(f.c(BOLD), BOLD);
    }

    #[test]
    fn test_format_duration() {
        let f = OutputFormatter::new(OutputFormat::Text, false);
        assert_eq!(f.format_duration(Duration::from_millis(50)), "50ms");
        assert_eq!(f.format_duration(Duration::from_millis(1500)), "1.50s");
        assert!(f.format_duration(Duration::from_secs(90)).contains("m"));
    }

    #[test]
    fn test_format_table() {
        let f = OutputFormatter::new(OutputFormat::Text, false);
        let rows = vec![
            vec!["A".into(), "B".into()],
            vec!["Hello".into(), "World".into()],
        ];
        let table = f.format_table(&["Col1", "Col2"], &rows);
        assert!(table.contains("Col1"));
        assert!(table.contains("Hello"));
        assert!(table.contains("World"));
    }

    #[test]
    fn test_lint_report_json_format() {
        let f = OutputFormatter::new(OutputFormat::Json, false);
        let report = LintReport::new("test_scene");
        let output = f.format_lint_report(&report, Duration::from_millis(10));
        assert!(output.contains("\"type\""));
        assert!(output.contains("lint_report"));
    }

    #[test]
    fn test_lint_report_compact_format() {
        let f = OutputFormatter::new(OutputFormat::Compact, false);
        let report = LintReport::new("test_scene");
        let output = f.format_lint_report(&report, Duration::from_millis(10));
        assert!(output.contains("0 errors"));
    }

    #[test]
    fn test_progress_bar() {
        let mut pb = ProgressBar::new(100, "Testing", false);
        pb.update(50);
        assert_eq!(pb.current, 50);
        pb.increment();
        assert_eq!(pb.current, 51);
        pb.update(200); // Should clamp to 100
        assert_eq!(pb.current, 100);
    }

    #[test]
    fn test_grade_indicator() {
        let f = OutputFormatter::new(OutputFormat::Text, false);
        let full = f.grade_indicator(&CertificateGrade::Full);
        let partial = f.grade_indicator(&CertificateGrade::Partial);
        let weak = f.grade_indicator(&CertificateGrade::Weak);
        assert!(full.contains("FULL"));
        assert!(partial.contains("PARTIAL"));
        assert!(weak.contains("WEAK"));
    }

    #[test]
    fn test_svg_report_generation() {
        let cert = CoverageCertificate {
            id: uuid::Uuid::new_v4(),
            timestamp: "2024-01-01T00:00:00Z".into(),
            protocol_version: "0.1.0".into(),
            scene_id: uuid::Uuid::new_v4(),
            samples: vec![],
            verified_regions: vec![],
            violations: vec![],
            epsilon_analytical: 0.001,
            epsilon_estimated: 0.05,
            delta: 0.05,
            kappa: 0.95,
            grade: CertificateGrade::Partial,
            total_time_s: 1.5,
            element_coverage: HashMap::new(),
            metadata: HashMap::new(),
        };
        let f = OutputFormatter::new(OutputFormat::Text, false);
        let svg = f.generate_svg_report(&cert);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Coverage Certificate"));
        assert!(svg.contains("PARTIAL"));
    }

    #[test]
    fn test_html_report_generation() {
        let cert = CoverageCertificate {
            id: uuid::Uuid::new_v4(),
            timestamp: "2024-01-01T00:00:00Z".into(),
            protocol_version: "0.1.0".into(),
            scene_id: uuid::Uuid::new_v4(),
            samples: vec![],
            verified_regions: vec![],
            violations: vec![],
            epsilon_analytical: 0.001,
            epsilon_estimated: 0.05,
            delta: 0.05,
            kappa: 0.95,
            grade: CertificateGrade::Partial,
            total_time_s: 1.5,
            element_coverage: HashMap::new(),
            metadata: HashMap::new(),
        };
        let f = OutputFormatter::new(OutputFormat::Text, false);
        let html = f.generate_html_report(&cert);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Coverage Certificate"));
        assert!(html.contains("Element Coverage"));
    }
}
