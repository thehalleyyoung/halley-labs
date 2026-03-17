//! # fpdiag-report
//!
//! Report generation for Penumbra diagnosis and repair results.
//!
//! Produces human-readable and machine-readable reports from
//! diagnosis reports, repair results, and benchmark metrics.

use fpdiag_types::{
    config::OutputFormat,
    diagnosis::{Diagnosis, DiagnosisReport, DiagnosisSeverity},
    eag::ErrorAmplificationGraph,
    repair::RepairResult,
};
use serde_json;
use std::fmt::Write;
use thiserror::Error;

/// Errors from the report module.
#[derive(Debug, Error)]
pub enum ReportError {
    #[error("formatting error: {0}")]
    FormatError(String),
    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// A formatted report ready for output.
#[derive(Debug, Clone)]
pub struct FormattedReport {
    /// The rendered content.
    pub content: String,
    /// The format used.
    pub format: OutputFormat,
}

/// Report generator.
pub struct ReportGenerator {
    format: OutputFormat,
    verbose: bool,
}

impl ReportGenerator {
    /// Create a new generator.
    pub fn new(format: OutputFormat, verbose: bool) -> Self {
        Self { format, verbose }
    }

    /// Generate a complete report from diagnosis and repair results.
    pub fn generate(
        &self,
        eag: &ErrorAmplificationGraph,
        diagnosis: &DiagnosisReport,
        repair: &RepairResult,
    ) -> Result<FormattedReport, ReportError> {
        match self.format {
            OutputFormat::Human => self.generate_human(eag, diagnosis, repair),
            OutputFormat::Json => self.generate_json(diagnosis, repair),
            OutputFormat::Csv => self.generate_csv(diagnosis),
        }
    }

    /// Generate a human-readable report.
    fn generate_human(
        &self,
        eag: &ErrorAmplificationGraph,
        diagnosis: &DiagnosisReport,
        repair: &RepairResult,
    ) -> Result<FormattedReport, ReportError> {
        let mut out = String::new();

        // Header
        writeln!(out, "╔══════════════════════════════════════════════════╗")
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "║       Penumbra — FP Diagnosis Report            ║")
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "╚══════════════════════════════════════════════════╝")
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out).map_err(|e| ReportError::FormatError(e.to_string()))?;

        // EAG Summary
        writeln!(out, "── Error Amplification Graph ──")
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "  Nodes: {}", eag.node_count())
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "  Edges: {}", eag.edge_count())
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out).map_err(|e| ReportError::FormatError(e.to_string()))?;

        // Diagnosis Summary
        writeln!(out, "── Diagnosis ──").map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(
            out,
            "  Analyzed: {} nodes ({} high-error)",
            diagnosis.total_nodes, diagnosis.high_error_nodes
        )
        .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(
            out,
            "  Overall confidence: {:.0}%",
            diagnosis.overall_confidence * 100.0
        )
        .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "  Time: {}ms", diagnosis.diagnosis_time_ms)
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out).map_err(|e| ReportError::FormatError(e.to_string()))?;

        // Category Breakdown
        if !diagnosis.category_counts.is_empty() {
            writeln!(out, "  Categories:").map_err(|e| ReportError::FormatError(e.to_string()))?;
            for (cat, count) in &diagnosis.category_counts {
                writeln!(out, "    {} — {} node(s)", cat.name(), count)
                    .map_err(|e| ReportError::FormatError(e.to_string()))?;
            }
            writeln!(out).map_err(|e| ReportError::FormatError(e.to_string()))?;
        }

        // Individual Diagnoses
        if self.verbose {
            writeln!(out, "── Detailed Diagnoses ──")
                .map_err(|e| ReportError::FormatError(e.to_string()))?;
            for (i, d) in diagnosis.diagnoses.iter().enumerate() {
                self.format_diagnosis(&mut out, i + 1, d)?;
            }
            writeln!(out).map_err(|e| ReportError::FormatError(e.to_string()))?;
        }

        // Repair Summary
        writeln!(out, "── Repairs ──").map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "  Applied: {} repair(s)", repair.applied_repairs.len())
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "  Overall reduction: {:.1}×", repair.overall_reduction)
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(out, "  Success rate: {:.0}%", repair.success_rate * 100.0)
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        writeln!(
            out,
            "  Fully certified: {}",
            if repair.fully_certified { "yes" } else { "no" }
        )
        .map_err(|e| ReportError::FormatError(e.to_string()))?;

        // Repair Details
        if self.verbose {
            for (i, (r, c)) in repair
                .applied_repairs
                .iter()
                .zip(repair.certifications.iter())
                .enumerate()
            {
                writeln!(out).map_err(|e| ReportError::FormatError(e.to_string()))?;
                writeln!(out, "  Repair #{}: {}", i + 1, r.strategy)
                    .map_err(|e| ReportError::FormatError(e.to_string()))?;
                writeln!(out, "    Targets: {} node(s)", r.target_nodes.len())
                    .map_err(|e| ReportError::FormatError(e.to_string()))?;
                writeln!(
                    out,
                    "    Reduction: {:.1}× ({})",
                    c.reduction_factor,
                    if c.is_formal {
                        "certified"
                    } else {
                        "empirical"
                    }
                )
                .map_err(|e| ReportError::FormatError(e.to_string()))?;
            }
        }

        Ok(FormattedReport {
            content: out,
            format: OutputFormat::Human,
        })
    }

    /// Format a single diagnosis entry.
    fn format_diagnosis(
        &self,
        out: &mut String,
        index: usize,
        diag: &Diagnosis,
    ) -> Result<(), ReportError> {
        let severity_icon = match diag.severity {
            DiagnosisSeverity::Info => "ℹ",
            DiagnosisSeverity::Warning => "⚠",
            DiagnosisSeverity::Error => "✖",
            DiagnosisSeverity::Critical => "🔥",
        };

        writeln!(out, "  {} #{}: {}", severity_icon, index, diag)
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        if !diag.explanation.is_empty() {
            writeln!(out, "      {}", diag.explanation)
                .map_err(|e| ReportError::FormatError(e.to_string()))?;
        }
        if let Some(src) = &diag.source {
            writeln!(
                out,
                "      at {}:{}:{}",
                src.file, src.line_start, src.col_start
            )
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        }
        writeln!(out, "      Suggested repair: {}", diag.repair_suggestion)
            .map_err(|e| ReportError::FormatError(e.to_string()))?;

        Ok(())
    }

    /// Generate a JSON report.
    fn generate_json(
        &self,
        diagnosis: &DiagnosisReport,
        repair: &RepairResult,
    ) -> Result<FormattedReport, ReportError> {
        let report = serde_json::json!({
            "diagnosis": {
                "total_nodes": diagnosis.total_nodes,
                "high_error_nodes": diagnosis.high_error_nodes,
                "overall_confidence": diagnosis.overall_confidence,
                "diagnoses": diagnosis.diagnoses,
                "diagnosis_time_ms": diagnosis.diagnosis_time_ms,
            },
            "repair": {
                "repairs_applied": repair.applied_repairs.len(),
                "overall_reduction": repair.overall_reduction,
                "success_rate": repair.success_rate,
                "fully_certified": repair.fully_certified,
                "repair_time_ms": repair.repair_time_ms,
            }
        });

        Ok(FormattedReport {
            content: serde_json::to_string_pretty(&report)?,
            format: OutputFormat::Json,
        })
    }

    /// Generate a CSV report (diagnoses only).
    fn generate_csv(&self, diagnosis: &DiagnosisReport) -> Result<FormattedReport, ReportError> {
        let mut csv = String::new();
        csv.push_str("node_id,category,severity,confidence,contribution,explanation\n");
        for d in &diagnosis.diagnoses {
            writeln!(
                csv,
                "{},{},{},{:.3},{:.3},\"{}\"",
                d.node_id,
                d.category.code(),
                d.severity.label(),
                d.confidence,
                d.error_contribution,
                d.explanation.replace('"', "\"\""),
            )
            .map_err(|e| ReportError::FormatError(e.to_string()))?;
        }

        Ok(FormattedReport {
            content: csv,
            format: OutputFormat::Csv,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpdiag_types::diagnosis::DiagnosisCategory;

    #[test]
    fn generate_empty_report() {
        let eag = ErrorAmplificationGraph::new();
        let diagnosis = DiagnosisReport::new();
        let repair = RepairResult::new();
        let gen = ReportGenerator::new(OutputFormat::Human, false);
        let report = gen.generate(&eag, &diagnosis, &repair).unwrap();
        assert!(report.content.contains("Penumbra"));
    }
}
