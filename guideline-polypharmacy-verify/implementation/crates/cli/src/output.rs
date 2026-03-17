//! Output formatting for GuardPharma CLI.
//!
//! Provides multiple output formats (text, JSON, table) for verification results,
//! conflict reports, recommendations, and safety certificates. Includes ANSI
//! color support and progress reporting.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Write;

use guardpharma_types::{CypEnzyme, DrugId, Severity};

use crate::input::{ActiveMedication, GuidelineDocument, PatientProfile};
use crate::pipeline::{
    AnalysisOutput, ConflictReport, EnzymePathwayDetail, PhaseTiming, PipelineOutput,
    Recommendation, SafetyCertificate, ScreeningOutput, ScreeningResult, TraceStep,
    VerificationVerdict,
};

// ─────────────────────── ANSI Color Helpers ──────────────────────────────

/// ANSI color codes for terminal output.
pub struct ColorOutput {
    pub enabled: bool,
}

impl ColorOutput {
    pub fn new(enabled: bool) -> Self {
        ColorOutput { enabled }
    }

    pub fn red(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[31m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn green(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[32m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn yellow(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[33m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn blue(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[34m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn magenta(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[35m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn cyan(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[36m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn bold(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[1m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn dim(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[2m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    /// Color text based on severity level.
    pub fn severity(&self, severity: Severity) -> String {
        let text = format!("{}", severity);
        match severity {
            Severity::None => self.dim(&text),
            Severity::Minor => self.blue(&text),
            Severity::Moderate => self.yellow(&text),
            Severity::Major => self.red(&text),
            Severity::Contraindicated => self.bold(&self.red(&text)),
        }
    }

    /// Color a verdict.
    pub fn verdict(&self, verdict: &VerificationVerdict) -> String {
        match verdict {
            VerificationVerdict::Safe => self.green("✓ SAFE"),
            VerificationVerdict::ConflictsFound { count } => {
                self.red(&format!("✗ {} CONFLICT(S) FOUND", count))
            }
            VerificationVerdict::Inconclusive { reason } => {
                self.yellow(&format!("? INCONCLUSIVE: {}", reason))
            }
            VerificationVerdict::Error { message } => {
                self.red(&format!("! ERROR: {}", message))
            }
        }
    }
}

// ───────────────────── Output Formatter Trait ────────────────────────────

/// Trait for formatting verification output.
pub trait OutputFormatter {
    /// Format full pipeline verification results.
    fn format_verification(&self, output: &PipelineOutput) -> String;

    /// Format screening (Tier 1) results.
    fn format_screening(&self, output: &ScreeningOutput) -> String;

    /// Format conflict analysis results.
    fn format_conflicts(&self, output: &AnalysisOutput) -> String;

    /// Format recommendations.
    fn format_recommendations(&self, recommendations: &[Recommendation]) -> String;
}

// ─────────────────────── Text Formatter ──────────────────────────────────

/// Human-readable text output formatter.
pub struct TextFormatter {
    color: ColorOutput,
    page_width: usize,
    decimal_places: usize,
    show_timing: bool,
    show_traces: bool,
}

impl TextFormatter {
    pub fn new(color_enabled: bool) -> Self {
        TextFormatter {
            color: ColorOutput::new(color_enabled),
            page_width: 80,
            decimal_places: 4,
            show_timing: true,
            show_traces: false,
        }
    }

    pub fn with_page_width(mut self, width: usize) -> Self {
        self.page_width = width;
        self
    }

    pub fn with_decimal_places(mut self, places: usize) -> Self {
        self.decimal_places = places;
        self
    }

    pub fn with_timing(mut self, show: bool) -> Self {
        self.show_timing = show;
        self
    }

    pub fn with_traces(mut self, show: bool) -> Self {
        self.show_traces = show;
        self
    }

    fn separator(&self) -> String {
        "─".repeat(self.page_width)
    }

    fn double_separator(&self) -> String {
        "═".repeat(self.page_width)
    }

    fn header(&self, title: &str) -> String {
        format!(
            "\n{}\n  {}\n{}",
            self.double_separator(),
            self.color.bold(title),
            self.double_separator()
        )
    }

    fn section(&self, title: &str) -> String {
        format!("\n{}\n  {}\n{}", self.separator(), title, self.separator())
    }

    fn format_patient_summary(&self, patient: &PatientProfile) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "  Age: {:.0} | Sex: {:?} | Weight: {:.1} kg | Height: {:.1} cm\n",
            patient.age(),
            patient.sex(),
            patient.info.weight_kg,
            patient.info.height_cm
        ));
        out.push_str(&format!(
            "  Creatinine: {:.2} mg/dL | eGFR: {} | Renal: {:?}\n",
            patient.info.serum_creatinine,
            patient
                .egfr
                .map(|e| format!("{:.1}", e))
                .unwrap_or_else(|| "N/A".to_string()),
            patient.renal_function()
        ));
        out.push_str(&format!(
            "  Conditions: {} | Medications: {} | Allergies: {}\n",
            patient.conditions.len(),
            patient.medication_count(),
            patient.allergies.len()
        ));
        out
    }

    fn format_timing_section(&self, timings: &[PhaseTiming]) -> String {
        if !self.show_timing || timings.is_empty() {
            return String::new();
        }

        let mut out = self.section("Timing");
        out.push('\n');
        for t in timings {
            out.push_str(&format!("  {:30} {:>8.2}ms\n", t.phase_name, t.duration_ms));
        }
        let total: f64 = timings.iter().map(|t| t.duration_ms).sum();
        out.push_str(&format!("  {:30} {:>8.2}ms\n", "TOTAL", total));
        out
    }
}

impl OutputFormatter for TextFormatter {
    fn format_verification(&self, output: &PipelineOutput) -> String {
        let mut out = String::new();

        out.push_str(&self.header("GuardPharma Verification Report"));
        out.push('\n');
        out.push_str(&format!("  Run ID: {}\n", output.run_id));
        out.push_str(&format!("  Timestamp: {}\n", output.timestamp));

        // Patient summary
        out.push_str(&self.section("Patient Profile"));
        out.push('\n');
        out.push_str(&self.format_patient_summary(&output.patient));

        // Medications
        out.push_str(&self.section("Active Medications"));
        out.push('\n');
        out.push_str(&format_medication_table(&output.patient.medications, &self.color));

        // Verdict
        out.push_str(&self.section("Verification Result"));
        out.push('\n');
        out.push_str(&format!("  Verdict: {}\n", self.color.verdict(&output.verdict)));
        out.push_str(&format!(
            "  Guidelines checked: {}\n",
            output.guidelines_checked
        ));
        out.push_str(&format!("  Drug pairs analyzed: {}\n", output.drug_pairs_checked));

        // Conflicts
        if !output.conflicts.is_empty() {
            out.push_str(&self.section(&format!("Conflicts ({})", output.conflicts.len())));
            out.push('\n');
            out.push_str(&format_conflict_table(&output.conflicts, &self.color));
        }

        // Screening results
        if !output.screening_results.is_empty() {
            out.push_str(&self.section("Tier 1 Screening Results"));
            out.push('\n');
            for result in &output.screening_results {
                out.push_str(&format!(
                    "  {} ↔ {}: {} (confidence: {:.1}%)\n",
                    result.drug_a_name,
                    result.drug_b_name,
                    self.color.severity(result.severity),
                    result.confidence * 100.0
                ));
                if !result.mechanism.is_empty() {
                    out.push_str(&format!("    Mechanism: {}\n", result.mechanism));
                }
            }
        }

        // Traces
        if self.show_traces {
            for conflict in &output.conflicts {
                if !conflict.trace.is_empty() {
                    out.push_str(&self.section(&format!(
                        "Trace: {} ↔ {}",
                        conflict.drug_a_name, conflict.drug_b_name
                    )));
                    out.push('\n');
                    out.push_str(&format_timeline(&conflict.trace, &self.color));
                }
            }
        }

        // Recommendations
        if !output.recommendations.is_empty() {
            out.push_str(&self.section("Recommendations"));
            out.push('\n');
            for (i, rec) in output.recommendations.iter().enumerate() {
                out.push_str(&format!(
                    "  {}. [{}] {}\n",
                    i + 1,
                    self.color.severity(rec.priority),
                    rec.summary
                ));
                if !rec.rationale.is_empty() {
                    out.push_str(&format!("     Rationale: {}\n", rec.rationale));
                }
                if let Some(ref alt) = rec.alternative {
                    out.push_str(&format!("     Alternative: {}\n", alt));
                }
            }
        }

        // Certificate
        if let Some(ref cert) = output.certificate {
            out.push_str(&format_safety_certificate(cert, &self.color));
        }

        // Timing
        out.push_str(&self.format_timing_section(&output.timings));

        out.push('\n');
        out.push_str(&self.double_separator());
        out.push('\n');
        out
    }

    fn format_screening(&self, output: &ScreeningOutput) -> String {
        let mut out = String::new();
        out.push_str(&self.header("GuardPharma Tier 1 Screening Report"));
        out.push('\n');
        out.push_str(&format!("  Timestamp: {}\n", output.timestamp));

        out.push_str(&self.section("Patient Profile"));
        out.push('\n');
        out.push_str(&self.format_patient_summary(&output.patient));

        out.push_str(&self.section(&format!("Screening Results ({})", output.results.len())));
        out.push('\n');

        if output.results.is_empty() {
            out.push_str(&format!(
                "  {}\n",
                self.color.green("No interactions detected at screening level")
            ));
        } else {
            for result in &output.results {
                out.push_str(&format!(
                    "  {} ↔ {}: {} (confidence {:.1}%)\n",
                    result.drug_a_name,
                    result.drug_b_name,
                    self.color.severity(result.severity),
                    result.confidence * 100.0
                ));
                if !result.mechanism.is_empty() {
                    out.push_str(&format!("    Mechanism: {}\n", result.mechanism));
                }
                if !result.affected_enzymes.is_empty() {
                    let enzymes: Vec<String> =
                        result.affected_enzymes.iter().map(|e| format!("{:?}", e)).collect();
                    out.push_str(&format!("    Enzymes: {}\n", enzymes.join(", ")));
                }
            }
        }

        out.push_str(&self.format_timing_section(&output.timings));
        out.push('\n');
        out
    }

    fn format_conflicts(&self, output: &AnalysisOutput) -> String {
        let mut out = String::new();
        out.push_str(&self.header("GuardPharma Interaction Analysis"));
        out.push('\n');
        out.push_str(&format!("  Timestamp: {}\n", output.timestamp));

        out.push_str(&self.section("Patient Profile"));
        out.push('\n');
        out.push_str(&self.format_patient_summary(&output.patient));

        out.push_str(&self.section(&format!("Conflicts ({})", output.conflicts.len())));
        out.push('\n');
        out.push_str(&format_conflict_table(&output.conflicts, &self.color));

        if !output.enzyme_details.is_empty() {
            out.push_str(&self.section("Enzyme Pathway Analysis"));
            out.push('\n');
            for detail in &output.enzyme_details {
                out.push_str(&format!(
                    "  {:?} — Activity: {:.1}% (baseline: 100%)\n",
                    detail.enzyme,
                    detail.net_activity * 100.0
                ));
                for effect in &detail.effects {
                    out.push_str(&format!("    {}: {}\n", effect.0, effect.1));
                }
            }
        }

        out.push_str(&self.format_timing_section(&output.timings));
        out.push('\n');
        out
    }

    fn format_recommendations(&self, recommendations: &[Recommendation]) -> String {
        let mut out = String::new();
        out.push_str(&self.header("GuardPharma Recommendations"));
        out.push('\n');

        if recommendations.is_empty() {
            out.push_str(&format!(
                "  {}\n",
                self.color.green("No recommendations — medication regimen appears safe")
            ));
        } else {
            for (i, rec) in recommendations.iter().enumerate() {
                out.push_str(&format!(
                    "\n  {}. {} — {}\n",
                    i + 1,
                    self.color.severity(rec.priority),
                    self.color.bold(&rec.summary)
                ));
                out.push_str(&format!("     Category: {}\n", rec.category));
                out.push_str(&format!("     Rationale: {}\n", rec.rationale));
                if let Some(ref alt) = rec.alternative {
                    out.push_str(&format!("     Alternative: {}\n", alt));
                }
                if let Some(ref monitoring) = rec.monitoring {
                    out.push_str(&format!("     Monitoring: {}\n", monitoring));
                }
                if !rec.affected_drugs.is_empty() {
                    out.push_str(&format!(
                        "     Affected drugs: {}\n",
                        rec.affected_drugs.join(", ")
                    ));
                }
            }
        }

        out.push('\n');
        out
    }
}

// ─────────────────────── JSON Formatter ──────────────────────────────────

/// JSON output formatter.
pub struct JsonFormatter {
    pretty: bool,
}

impl JsonFormatter {
    pub fn new() -> Self {
        JsonFormatter { pretty: true }
    }

    pub fn compact(mut self) -> Self {
        self.pretty = false;
        self
    }

    fn to_json<T: Serialize>(&self, value: &T) -> String {
        if self.pretty {
            serde_json::to_string_pretty(value).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
        } else {
            serde_json::to_string(value).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
        }
    }
}

impl Default for JsonFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputFormatter for JsonFormatter {
    fn format_verification(&self, output: &PipelineOutput) -> String {
        self.to_json(output)
    }

    fn format_screening(&self, output: &ScreeningOutput) -> String {
        self.to_json(output)
    }

    fn format_conflicts(&self, output: &AnalysisOutput) -> String {
        self.to_json(output)
    }

    fn format_recommendations(&self, recommendations: &[Recommendation]) -> String {
        self.to_json(&recommendations)
    }
}

// ─────────────────────── Table Formatter ─────────────────────────────────

/// Tabular output formatter for structured data display.
pub struct TableFormatter {
    color: ColorOutput,
    page_width: usize,
}

impl TableFormatter {
    pub fn new(color_enabled: bool) -> Self {
        TableFormatter {
            color: ColorOutput::new(color_enabled),
            page_width: 80,
        }
    }

    pub fn with_page_width(mut self, width: usize) -> Self {
        self.page_width = width;
        self
    }

    fn format_row(&self, cells: &[(&str, usize)]) -> String {
        let mut row = String::from("│");
        for (text, width) in cells {
            let display = if text.len() > *width {
                format!("{}…", &text[..*width - 1])
            } else {
                format!("{:width$}", text, width = width)
            };
            row.push_str(&format!(" {} │", display));
        }
        row
    }

    fn format_header_sep(&self, widths: &[usize]) -> String {
        let mut sep = String::from("├");
        for (i, w) in widths.iter().enumerate() {
            sep.push_str(&"─".repeat(w + 2));
            if i < widths.len() - 1 {
                sep.push('┼');
            }
        }
        sep.push('┤');
        sep
    }

    fn format_top_border(&self, widths: &[usize]) -> String {
        let mut sep = String::from("┌");
        for (i, w) in widths.iter().enumerate() {
            sep.push_str(&"─".repeat(w + 2));
            if i < widths.len() - 1 {
                sep.push('┬');
            }
        }
        sep.push('┐');
        sep
    }

    fn format_bottom_border(&self, widths: &[usize]) -> String {
        let mut sep = String::from("└");
        for (i, w) in widths.iter().enumerate() {
            sep.push_str(&"─".repeat(w + 2));
            if i < widths.len() - 1 {
                sep.push('┴');
            }
        }
        sep.push('┘');
        sep
    }
}

impl OutputFormatter for TableFormatter {
    fn format_verification(&self, output: &PipelineOutput) -> String {
        let mut out = String::new();

        // Summary table
        let widths = [30, 40];
        out.push_str(&self.format_top_border(&widths));
        out.push('\n');
        out.push_str(&self.format_row(&[("Property", 30), ("Value", 40)]));
        out.push('\n');
        out.push_str(&self.format_header_sep(&widths));
        out.push('\n');

        let rows: Vec<(String, String)> = vec![
            ("Run ID".to_string(), output.run_id.clone()),
            ("Timestamp".to_string(), output.timestamp.clone()),
            (
                "Verdict".to_string(),
                format!("{:?}", output.verdict),
            ),
            (
                "Guidelines Checked".to_string(),
                output.guidelines_checked.to_string(),
            ),
            (
                "Drug Pairs".to_string(),
                output.drug_pairs_checked.to_string(),
            ),
            ("Conflicts".to_string(), output.conflicts.len().to_string()),
            (
                "Recommendations".to_string(),
                output.recommendations.len().to_string(),
            ),
        ];

        for (key, value) in &rows {
            out.push_str(&self.format_row(&[(key, 30), (value, 40)]));
            out.push('\n');
        }
        out.push_str(&self.format_bottom_border(&widths));
        out.push('\n');

        // Conflict table
        if !output.conflicts.is_empty() {
            out.push('\n');
            out.push_str(&format_conflict_table(&output.conflicts, &self.color));
        }

        out
    }

    fn format_screening(&self, output: &ScreeningOutput) -> String {
        let mut out = String::new();

        let widths = [15, 15, 14, 10, 20];
        out.push_str(&self.format_top_border(&widths));
        out.push('\n');
        out.push_str(&self.format_row(&[
            ("Drug A", 15),
            ("Drug B", 15),
            ("Severity", 14),
            ("Confidence", 10),
            ("Mechanism", 20),
        ]));
        out.push('\n');
        out.push_str(&self.format_header_sep(&widths));
        out.push('\n');

        for result in &output.results {
            out.push_str(&self.format_row(&[
                (&result.drug_a_name, 15),
                (&result.drug_b_name, 15),
                (&format!("{}", result.severity), 14),
                (&format!("{:.1}%", result.confidence * 100.0), 10),
                (&result.mechanism, 20),
            ]));
            out.push('\n');
        }

        out.push_str(&self.format_bottom_border(&widths));
        out.push('\n');
        out
    }

    fn format_conflicts(&self, output: &AnalysisOutput) -> String {
        let mut out = String::new();
        out.push_str(&format_conflict_table(&output.conflicts, &self.color));
        out
    }

    fn format_recommendations(&self, recommendations: &[Recommendation]) -> String {
        let mut out = String::new();

        let widths = [5, 14, 35, 20];
        out.push_str(&self.format_top_border(&widths));
        out.push('\n');
        out.push_str(&self.format_row(&[
            ("#", 5),
            ("Priority", 14),
            ("Summary", 35),
            ("Category", 20),
        ]));
        out.push('\n');
        out.push_str(&self.format_header_sep(&widths));
        out.push('\n');

        for (i, rec) in recommendations.iter().enumerate() {
            out.push_str(&self.format_row(&[
                (&(i + 1).to_string(), 5),
                (&format!("{}", rec.priority), 14),
                (&rec.summary, 35),
                (&rec.category, 20),
            ]));
            out.push('\n');
        }

        out.push_str(&self.format_bottom_border(&widths));
        out.push('\n');
        out
    }
}

// ──────────────── Standalone Formatting Functions ────────────────────────

/// Format a conflict table.
pub fn format_conflict_table(conflicts: &[ConflictReport], color: &ColorOutput) -> String {
    let mut out = String::new();

    if conflicts.is_empty() {
        out.push_str("  No conflicts found.\n");
        return out;
    }

    let max_name_a = conflicts.iter().map(|c| c.drug_a_name.len()).max().unwrap_or(10).max(10);
    let max_name_b = conflicts.iter().map(|c| c.drug_b_name.len()).max().unwrap_or(10).max(10);

    // Header
    out.push_str(&format!(
        "  {:<width_a$}   {:<width_b$}   {:<16}  {:<10}  {}\n",
        "Drug A",
        "Drug B",
        "Severity",
        "Confidence",
        "Mechanism",
        width_a = max_name_a,
        width_b = max_name_b,
    ));
    out.push_str(&format!(
        "  {:<width_a$}   {:<width_b$}   {:<16}  {:<10}  {}\n",
        "─".repeat(max_name_a),
        "─".repeat(max_name_b),
        "─".repeat(16),
        "─".repeat(10),
        "─".repeat(30),
        width_a = max_name_a,
        width_b = max_name_b,
    ));

    for conflict in conflicts {
        out.push_str(&format!(
            "  {:<width_a$}   {:<width_b$}   {:<16}  {:>8.1}%   {}\n",
            conflict.drug_a_name,
            conflict.drug_b_name,
            color.severity(conflict.severity),
            conflict.confidence * 100.0,
            conflict.mechanism,
            width_a = max_name_a,
            width_b = max_name_b,
        ));
    }

    out
}

/// Format a medication table.
pub fn format_medication_table(medications: &[ActiveMedication], color: &ColorOutput) -> String {
    let mut out = String::new();

    if medications.is_empty() {
        out.push_str("  No medications.\n");
        return out;
    }

    let max_name = medications.iter().map(|m| m.name.len()).max().unwrap_or(10).max(10);
    let max_class = medications
        .iter()
        .map(|m| m.drug_class.len())
        .max()
        .unwrap_or(10)
        .max(10);

    out.push_str(&format!(
        "  {:<width_n$}  {:>8}  {:>6}  {:<7}  {:<width_c$}\n",
        "Medication",
        "Dose(mg)",
        "q(h)",
        "Route",
        "Class",
        width_n = max_name,
        width_c = max_class,
    ));
    out.push_str(&format!(
        "  {:<width_n$}  {:>8}  {:>6}  {:<7}  {:<width_c$}\n",
        "─".repeat(max_name),
        "─".repeat(8),
        "─".repeat(6),
        "─".repeat(7),
        "─".repeat(max_class),
        width_n = max_name,
        width_c = max_class,
    ));

    for med in medications {
        out.push_str(&format!(
            "  {:<width_n$}  {:>8.1}  {:>6.0}  {:<7}  {:<width_c$}\n",
            color.cyan(&med.name),
            med.dose_mg,
            med.frequency_hours,
            format!("{:?}", med.route),
            med.drug_class,
            width_n = max_name,
            width_c = max_class,
        ));
    }

    out
}

/// Format a verification trace as a timeline.
pub fn format_timeline(trace: &[TraceStep], color: &ColorOutput) -> String {
    let mut out = String::new();

    if trace.is_empty() {
        out.push_str("  No trace steps.\n");
        return out;
    }

    for (i, step) in trace.iter().enumerate() {
        let connector = if i == trace.len() - 1 { "└" } else { "├" };
        let pipe = if i == trace.len() - 1 { " " } else { "│" };

        out.push_str(&format!(
            "  {} t={:>6.2}h  {}\n",
            connector,
            step.time_hours,
            color.bold(&step.description)
        ));

        if let Some(ref conc_a) = step.concentration_a {
            out.push_str(&format!(
                "  {}   Drug A: [{:.4}, {:.4}] mg/L\n",
                pipe, conc_a.0, conc_a.1
            ));
        }
        if let Some(ref conc_b) = step.concentration_b {
            out.push_str(&format!(
                "  {}   Drug B: [{:.4}, {:.4}] mg/L\n",
                pipe, conc_b.0, conc_b.1
            ));
        }
        if let Some(ref violation) = step.violation {
            out.push_str(&format!(
                "  {}   {} {}\n",
                pipe,
                color.red("VIOLATION:"),
                violation
            ));
        }
    }

    out
}

/// Format a safety certificate.
pub fn format_safety_certificate(cert: &SafetyCertificate, color: &ColorOutput) -> String {
    let mut out = String::new();

    out.push_str("\n");
    out.push_str(&"═".repeat(60));
    out.push_str("\n  SAFETY VERIFICATION CERTIFICATE\n");
    out.push_str(&"═".repeat(60));
    out.push('\n');

    out.push_str(&format!("  Certificate ID: {}\n", cert.certificate_id));
    out.push_str(&format!("  Run ID:         {}\n", cert.run_id));
    out.push_str(&format!("  Timestamp:      {}\n", cert.timestamp));
    out.push_str(&format!("  Patient ID:     {}\n", cert.patient_id));
    out.push('\n');

    out.push_str(&format!(
        "  Verdict: {}\n",
        color.verdict(&cert.verdict)
    ));
    out.push('\n');

    out.push_str(&format!("  Guidelines checked:      {}\n", cert.guidelines_checked));
    out.push_str(&format!("  Drug pairs analyzed:     {}\n", cert.drug_pairs_checked));
    out.push_str(&format!("  Conflicts found:         {}\n", cert.conflicts_found));
    out.push_str(&format!(
        "  Tier 1 (screening):      {}\n",
        if cert.tier1_completed {
            color.green("completed")
        } else {
            color.yellow("skipped")
        }
    ));
    out.push_str(&format!(
        "  Tier 2 (model checking): {}\n",
        if cert.tier2_completed {
            color.green("completed")
        } else {
            color.yellow("skipped")
        }
    ));

    if !cert.warnings.is_empty() {
        out.push_str("\n  Warnings:\n");
        for warning in &cert.warnings {
            out.push_str(&format!("    • {}\n", color.yellow(warning)));
        }
    }

    out.push('\n');
    out.push_str(&"═".repeat(60));
    out.push('\n');

    out
}

// ──────────────────── Progress Reporter ──────────────────────────────────

/// Reports progress of the verification pipeline.
pub struct ProgressReporter {
    color: ColorOutput,
    enabled: bool,
    total_phases: usize,
    current_phase: usize,
}

impl ProgressReporter {
    pub fn new(color_enabled: bool, enabled: bool, total_phases: usize) -> Self {
        ProgressReporter {
            color: ColorOutput::new(color_enabled),
            enabled,
            total_phases,
            current_phase: 0,
        }
    }

    /// Report the start of a new phase.
    pub fn start_phase(&mut self, phase_name: &str) {
        if !self.enabled {
            return;
        }
        self.current_phase += 1;
        let progress = format!("[{}/{}]", self.current_phase, self.total_phases);
        eprintln!(
            "{} {} {}",
            self.color.dim(&progress),
            self.color.cyan("▶"),
            phase_name
        );
    }

    /// Report completion of the current phase.
    pub fn complete_phase(&self, phase_name: &str, duration_ms: f64) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "  {} {} ({:.1}ms)",
            self.color.green("✓"),
            phase_name,
            duration_ms
        );
    }

    /// Report a warning during processing.
    pub fn warn(&self, message: &str) {
        if !self.enabled {
            return;
        }
        eprintln!("  {} {}", self.color.yellow("⚠"), message);
    }

    /// Report an error during processing.
    pub fn error(&self, message: &str) {
        eprintln!("  {} {}", self.color.red("✗"), message);
    }

    /// Report final summary.
    pub fn summary(&self, verdict: &VerificationVerdict, total_ms: f64) {
        if !self.enabled {
            return;
        }
        eprintln!();
        eprintln!("  {}", self.color.verdict(verdict));
        eprintln!(
            "  Total time: {:.1}ms",
            total_ms
        );
    }
}

// ────────────────────── Utility: Output Writer ───────────────────────────

/// Write output to either a file or stdout.
pub fn write_output(content: &str, output_path: Option<&std::path::Path>) -> anyhow::Result<()> {
    match output_path {
        Some(path) => {
            std::fs::write(path, content)
                .with_context(|| format!("Failed to write output to {}", path.display()))?;
            log::info!("Output written to {}", path.display());
            Ok(())
        }
        None => {
            print!("{}", content);
            std::io::stdout().flush()?;
            Ok(())
        }
    }
}

use anyhow::Context;

// ────────────────────────────── Tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::VerificationVerdict;

    #[test]
    fn test_color_output_enabled() {
        let color = ColorOutput::new(true);
        let red = color.red("error");
        assert!(red.contains("\x1b[31m"));
        assert!(red.contains("error"));
    }

    #[test]
    fn test_color_output_disabled() {
        let color = ColorOutput::new(false);
        let red = color.red("error");
        assert_eq!(red, "error");
    }

    #[test]
    fn test_color_severity() {
        let color = ColorOutput::new(true);
        let minor = color.severity(Severity::Minor);
        assert!(minor.contains("Minor"));
        let major = color.severity(Severity::Major);
        assert!(major.contains("Major"));
    }

    #[test]
    fn test_color_verdict_safe() {
        let color = ColorOutput::new(true);
        let v = color.verdict(&VerificationVerdict::Safe);
        assert!(v.contains("SAFE"));
    }

    #[test]
    fn test_color_verdict_conflicts() {
        let color = ColorOutput::new(false);
        let v = color.verdict(&VerificationVerdict::ConflictsFound { count: 3 });
        assert!(v.contains("3"));
        assert!(v.contains("CONFLICT"));
    }

    #[test]
    fn test_text_formatter_header() {
        let tf = TextFormatter::new(false);
        let h = tf.header("Test Header");
        assert!(h.contains("Test Header"));
        assert!(h.contains("═"));
    }

    #[test]
    fn test_text_formatter_section() {
        let tf = TextFormatter::new(false);
        let s = tf.section("Test Section");
        assert!(s.contains("Test Section"));
        assert!(s.contains("─"));
    }

    #[test]
    fn test_format_medication_table_empty() {
        let color = ColorOutput::new(false);
        let result = format_medication_table(&[], &color);
        assert!(result.contains("No medications"));
    }

    #[test]
    fn test_format_medication_table_with_meds() {
        let color = ColorOutput::new(false);
        let meds = vec![
            ActiveMedication::new("Warfarin", 5.0).with_class("anticoagulant"),
            ActiveMedication::new("Metformin", 500.0).with_class("biguanide"),
        ];
        let result = format_medication_table(&meds, &color);
        assert!(result.contains("Warfarin"));
        assert!(result.contains("Metformin"));
        assert!(result.contains("5.0"));
        assert!(result.contains("500.0"));
    }

    #[test]
    fn test_format_conflict_table_empty() {
        let color = ColorOutput::new(false);
        let result = format_conflict_table(&[], &color);
        assert!(result.contains("No conflicts"));
    }

    #[test]
    fn test_format_conflict_table_with_conflicts() {
        let color = ColorOutput::new(false);
        let conflicts = vec![ConflictReport {
            drug_a_name: "Warfarin".to_string(),
            drug_b_name: "Aspirin".to_string(),
            severity: Severity::Major,
            confidence: 0.95,
            mechanism: "Additive anticoagulation".to_string(),
            description: "Increased bleeding risk".to_string(),
            trace: vec![],
            clinical_consequence: String::new(),
        }];
        let result = format_conflict_table(&conflicts, &color);
        assert!(result.contains("Warfarin"));
        assert!(result.contains("Aspirin"));
        assert!(result.contains("95.0%"));
    }

    #[test]
    fn test_format_timeline_empty() {
        let color = ColorOutput::new(false);
        let result = format_timeline(&[], &color);
        assert!(result.contains("No trace"));
    }

    #[test]
    fn test_format_timeline_with_steps() {
        let color = ColorOutput::new(false);
        let steps = vec![
            TraceStep {
                time_hours: 0.0,
                description: "Drug A administered".to_string(),
                concentration_a: Some((0.0, 0.0)),
                concentration_b: None,
                violation: None,
            },
            TraceStep {
                time_hours: 2.0,
                description: "Peak concentration".to_string(),
                concentration_a: Some((5.0, 6.0)),
                concentration_b: Some((1.0, 1.5)),
                violation: Some("Exceeds therapeutic window".to_string()),
            },
        ];
        let result = format_timeline(&steps, &color);
        assert!(result.contains("Drug A administered"));
        assert!(result.contains("Peak concentration"));
        assert!(result.contains("VIOLATION"));
    }

    #[test]
    fn test_format_safety_certificate() {
        let color = ColorOutput::new(false);
        let cert = SafetyCertificate {
            certificate_id: "CERT-001".to_string(),
            run_id: "RUN-001".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            patient_id: "PT-001".to_string(),
            verdict: VerificationVerdict::Safe,
            guidelines_checked: 5,
            drug_pairs_checked: 10,
            conflicts_found: 0,
            tier1_completed: true,
            tier2_completed: true,
            warnings: vec![],
        };
        let result = format_safety_certificate(&cert, &color);
        assert!(result.contains("CERT-001"));
        assert!(result.contains("SAFE"));
        assert!(result.contains("CERTIFICATE"));
    }

    #[test]
    fn test_json_formatter_verification() {
        let formatter = JsonFormatter::new();
        let output = PipelineOutput {
            run_id: "test-run".to_string(),
            timestamp: "2024-01-01".to_string(),
            patient: PatientProfile::default(),
            verdict: VerificationVerdict::Safe,
            guidelines_checked: 1,
            drug_pairs_checked: 0,
            screening_results: vec![],
            conflicts: vec![],
            recommendations: vec![],
            certificate: None,
            timings: vec![],
        };
        let json = formatter.format_verification(&output);
        assert!(json.contains("test-run"));
        assert!(json.contains("Safe"));
    }

    #[test]
    fn test_json_formatter_compact() {
        let formatter = JsonFormatter::new().compact();
        let recs: Vec<Recommendation> = vec![];
        let json = formatter.format_recommendations(&recs);
        assert_eq!(json.trim(), "[]");
    }

    #[test]
    fn test_table_formatter_borders() {
        let tf = TableFormatter::new(false);
        let top = tf.format_top_border(&[10, 15]);
        assert!(top.starts_with('┌'));
        assert!(top.contains('┬'));
        assert!(top.ends_with('┐'));

        let bottom = tf.format_bottom_border(&[10, 15]);
        assert!(bottom.starts_with('└'));
        assert!(bottom.contains('┴'));
        assert!(bottom.ends_with('┘'));
    }

    #[test]
    fn test_table_formatter_row() {
        let tf = TableFormatter::new(false);
        let row = tf.format_row(&[("Hello", 10), ("World", 10)]);
        assert!(row.contains("Hello"));
        assert!(row.contains("World"));
        assert!(row.starts_with('│'));
    }

    #[test]
    fn test_progress_reporter_enabled() {
        let mut pr = ProgressReporter::new(false, true, 3);
        pr.start_phase("Loading");
        assert_eq!(pr.current_phase, 1);
        pr.start_phase("Verifying");
        assert_eq!(pr.current_phase, 2);
    }

    #[test]
    fn test_progress_reporter_disabled() {
        let mut pr = ProgressReporter::new(false, false, 3);
        pr.start_phase("Loading");
        assert_eq!(pr.current_phase, 0); // doesn't increment when disabled
    }

    #[test]
    fn test_text_formatter_screening_no_results() {
        let tf = TextFormatter::new(false);
        let output = ScreeningOutput {
            timestamp: "2024-01-01".to_string(),
            patient: PatientProfile::default(),
            results: vec![],
            timings: vec![],
        };
        let text = tf.format_screening(&output);
        assert!(text.contains("No interactions detected"));
    }

    #[test]
    fn test_text_formatter_recommendations_empty() {
        let tf = TextFormatter::new(false);
        let text = tf.format_recommendations(&[]);
        assert!(text.contains("No recommendations"));
    }

    #[test]
    fn test_text_formatter_recommendations_with_data() {
        let tf = TextFormatter::new(false);
        let recs = vec![Recommendation {
            summary: "Reduce warfarin dose".to_string(),
            priority: Severity::Major,
            category: "Dose adjustment".to_string(),
            rationale: "High bleeding risk".to_string(),
            alternative: Some("Consider DOAC".to_string()),
            monitoring: Some("Check INR weekly".to_string()),
            affected_drugs: vec!["warfarin".to_string()],
        }];
        let text = tf.format_recommendations(&recs);
        assert!(text.contains("Reduce warfarin dose"));
        assert!(text.contains("Dose adjustment"));
        assert!(text.contains("Consider DOAC"));
    }

    #[test]
    fn test_color_bold() {
        let color = ColorOutput::new(true);
        let bold = color.bold("test");
        assert!(bold.contains("\x1b[1m"));

        let color_off = ColorOutput::new(false);
        assert_eq!(color_off.bold("test"), "test");
    }

    #[test]
    fn test_color_all_methods() {
        let color = ColorOutput::new(true);
        assert!(color.green("ok").contains("\x1b[32m"));
        assert!(color.yellow("warn").contains("\x1b[33m"));
        assert!(color.blue("info").contains("\x1b[34m"));
        assert!(color.magenta("debug").contains("\x1b[35m"));
        assert!(color.cyan("trace").contains("\x1b[36m"));
        assert!(color.dim("dim").contains("\x1b[2m"));
    }
}
