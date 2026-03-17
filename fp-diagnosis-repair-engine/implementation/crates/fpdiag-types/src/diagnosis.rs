//! Diagnosis categories and reports.
//!
//! Defines the five-category taxonomy of floating-point error root causes
//! used by the Penumbra diagnosis engine, plus structured diagnosis reports
//! that connect root causes to repair recommendations.

use crate::eag::EagNodeId;
use crate::source::SourceSpan;
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── DiagnosisCategory ──────────────────────────────────────────────────────

/// Root-cause category from the Penumbra error taxonomy (T3).
///
/// Every high-error node in the EAG is classified into exactly one of these
/// categories, each of which maps to a specific family of repairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiagnosisCategory {
    /// **Catastrophic cancellation**: subtraction of nearly equal values
    /// amplifies relative error by orders of magnitude.
    ///
    /// Classic example: `(1 + x) - 1` for small `x`.
    CatastrophicCancellation,

    /// **Absorption**: small addend is lost when added to a much larger
    /// accumulator, because the sum rounds back to the accumulator value.
    ///
    /// Classic example: `1e16 + 1.0` in f64.
    Absorption,

    /// **Smearing**: alternating-sign summation causes gradual error growth
    /// as partial sums oscillate, each step introducing rounding error
    /// without the catastrophic single-step blowup of cancellation.
    Smearing,

    /// **Amplified rounding**: a high condition number amplifies ordinary
    /// per-operation rounding errors through the computation.
    AmplifiedRounding,

    /// **Ill-conditioned subproblem**: a LAPACK/BLAS call (linear solve,
    /// eigendecomposition, etc.) is numerically ill-conditioned at the
    /// given inputs, amplifying input errors unpredictably.
    IllConditionedSubproblem,
}

impl DiagnosisCategory {
    /// All five categories.
    pub const ALL: [Self; 5] = [
        Self::CatastrophicCancellation,
        Self::Absorption,
        Self::Smearing,
        Self::AmplifiedRounding,
        Self::IllConditionedSubproblem,
    ];

    /// Short human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::CatastrophicCancellation => "catastrophic cancellation",
            Self::Absorption => "absorption",
            Self::Smearing => "smearing",
            Self::AmplifiedRounding => "amplified rounding",
            Self::IllConditionedSubproblem => "ill-conditioned subproblem",
        }
    }

    /// Abbreviated code for reports.
    pub fn code(&self) -> &'static str {
        match self {
            Self::CatastrophicCancellation => "CANCEL",
            Self::Absorption => "ABSORB",
            Self::Smearing => "SMEAR",
            Self::AmplifiedRounding => "AMP_ROUND",
            Self::IllConditionedSubproblem => "ILL_COND",
        }
    }

    /// Suggested repair family for this category.
    pub fn repair_family(&self) -> &'static str {
        match self {
            Self::CatastrophicCancellation => "algebraic rewrite (log-space, compensated form)",
            Self::Absorption => "compensated summation (Kahan, pairwise)",
            Self::Smearing => "reordering or compensated accumulation",
            Self::AmplifiedRounding => "precision promotion on critical path",
            Self::IllConditionedSubproblem => "preconditioning or higher-precision library call",
        }
    }
}

impl fmt::Display for DiagnosisCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ─── DiagnosisSeverity ──────────────────────────────────────────────────────

/// Severity of a diagnosed error pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DiagnosisSeverity {
    /// Informational: error is present but small.
    Info,
    /// Warning: noticeable error (>10 ULPs).
    Warning,
    /// Error: significant precision loss (>1000 ULPs or >4 bits lost).
    Error,
    /// Critical: catastrophic precision loss (>half the significand).
    Critical,
}

impl DiagnosisSeverity {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }
}

impl fmt::Display for DiagnosisSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ─── Diagnosis ──────────────────────────────────────────────────────────────

/// A single diagnosis: a root-cause attribution for a high-error EAG node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    /// The EAG node being diagnosed.
    pub node_id: EagNodeId,
    /// The root-cause category.
    pub category: DiagnosisCategory,
    /// Severity level.
    pub severity: DiagnosisSeverity,
    /// Confidence score in [0, 1].
    pub confidence: f64,
    /// Fraction of total output error attributed to this node via EAG paths.
    pub error_contribution: f64,
    /// Source location of the diagnosed operation.
    pub source: Option<SourceSpan>,
    /// Human-readable explanation of the diagnosis.
    pub explanation: String,
    /// Suggested repair strategy description.
    pub repair_suggestion: String,
}

impl Diagnosis {
    /// Create a new diagnosis.
    pub fn new(
        node_id: EagNodeId,
        category: DiagnosisCategory,
        severity: DiagnosisSeverity,
        confidence: f64,
    ) -> Self {
        Self {
            node_id,
            category,
            severity,
            confidence: confidence.clamp(0.0, 1.0),
            error_contribution: 0.0,
            source: None,
            explanation: String::new(),
            repair_suggestion: category.repair_family().to_string(),
        }
    }

    /// Set the explanation text.
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    /// Set the source span.
    pub fn with_source(mut self, source: SourceSpan) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the error contribution fraction.
    pub fn with_contribution(mut self, contribution: f64) -> Self {
        self.error_contribution = contribution.clamp(0.0, 1.0);
        self
    }
}

impl fmt::Display for Diagnosis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} at {} (confidence: {:.0}%, contribution: {:.1}%)",
            self.severity,
            self.category,
            self.node_id,
            self.confidence * 100.0,
            self.error_contribution * 100.0,
        )
    }
}

// ─── DiagnosisReport ────────────────────────────────────────────────────────

/// A complete diagnosis report for an EAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisReport {
    /// Individual diagnoses, sorted by severity (descending).
    pub diagnoses: Vec<Diagnosis>,
    /// Total number of EAG nodes analyzed.
    pub total_nodes: usize,
    /// Number of nodes classified as high-error.
    pub high_error_nodes: usize,
    /// Breakdown by category.
    pub category_counts: Vec<(DiagnosisCategory, usize)>,
    /// Overall confidence (minimum over individual confidences).
    pub overall_confidence: f64,
    /// Wall-clock time for the diagnosis pass (ms).
    pub diagnosis_time_ms: u64,
}

impl DiagnosisReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self {
            diagnoses: Vec::new(),
            total_nodes: 0,
            high_error_nodes: 0,
            category_counts: Vec::new(),
            overall_confidence: 1.0,
            diagnosis_time_ms: 0,
        }
    }

    /// Add a diagnosis and update statistics.
    pub fn add(&mut self, diag: Diagnosis) {
        if diag.confidence < self.overall_confidence {
            self.overall_confidence = diag.confidence;
        }
        self.diagnoses.push(diag);
        self.high_error_nodes = self.diagnoses.len();
        self.recompute_counts();
    }

    /// Recompute category_counts from diagnoses.
    fn recompute_counts(&mut self) {
        use std::collections::HashMap;
        let mut counts: HashMap<DiagnosisCategory, usize> = HashMap::new();
        for d in &self.diagnoses {
            *counts.entry(d.category).or_insert(0) += 1;
        }
        self.category_counts = counts.into_iter().collect();
        self.category_counts
            .sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    }

    /// Sort diagnoses by severity (critical first), then by contribution.
    pub fn sort(&mut self) {
        self.diagnoses.sort_by(|a, b| {
            b.severity.cmp(&a.severity).then(
                b.error_contribution
                    .partial_cmp(&a.error_contribution)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
    }

    /// Get the most critical diagnosis, if any.
    pub fn most_critical(&self) -> Option<&Diagnosis> {
        self.diagnoses.iter().max_by_key(|d| d.severity)
    }
}

impl Default for DiagnosisReport {
    fn default() -> Self {
        Self::new()
    }
}
