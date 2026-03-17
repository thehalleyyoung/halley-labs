//! Regression detection for leakage contracts.
//!
//! Compares two versions of a contract to detect changes in leakage bounds,
//! transformer behaviour, or strength downgrades. Designed to integrate
//! with CI/CD pipelines.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::contract::LeakageContract;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity of a contract regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Informational – no action required.
    Info,
    /// Warning – bound loosened but still within tolerance.
    Warning,
    /// Error – bound significantly regressed.
    Error,
    /// Critical – soundness property lost.
    Critical,
}

impl RegressionSeverity {
    /// Return a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }

    /// Whether this severity should fail a CI check.
    pub fn is_blocking(self) -> bool {
        matches!(self, Self::Error | Self::Critical)
    }
}

impl fmt::Display for RegressionSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Contract delta
// ---------------------------------------------------------------------------

/// Describes how a single contract changed between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractDelta {
    /// Name of the function whose contract changed.
    pub function_name: String,
    /// Previous worst-case bound (bits).
    pub old_bound: f64,
    /// New worst-case bound (bits).
    pub new_bound: f64,
    /// Absolute change in bits (positive = regression).
    pub bound_delta: f64,
    /// Relative change as a fraction of the old bound.
    pub relative_change: f64,
    /// Whether the strength level changed.
    pub strength_changed: bool,
    /// Whether the transformer changed.
    pub transformer_changed: bool,
    /// Severity of this delta.
    pub severity: RegressionSeverity,
    /// Human-readable explanation.
    pub explanation: String,
}

impl ContractDelta {
    /// Compute the delta between two contract versions.
    pub fn compute(old: &LeakageContract, new: &LeakageContract) -> Self {
        let old_bound = old.worst_case_bits();
        let new_bound = new.worst_case_bits();
        let bound_delta = new_bound - old_bound;
        let relative_change = if old_bound.abs() < f64::EPSILON {
            if new_bound.abs() < f64::EPSILON {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            bound_delta / old_bound
        };

        let strength_changed = old.strength != new.strength;
        let transformer_changed =
            old.cache_transformer.modified_sets() != new.cache_transformer.modified_sets();

        let severity = if !new.is_sound() && old.is_sound() {
            RegressionSeverity::Critical
        } else if bound_delta > 1.0 {
            RegressionSeverity::Error
        } else if bound_delta > 0.0 {
            RegressionSeverity::Warning
        } else {
            RegressionSeverity::Info
        };

        let explanation = if bound_delta <= 0.0 {
            format!(
                "{}: bound improved by {:.2} bits",
                new.function_name,
                bound_delta.abs()
            )
        } else {
            format!(
                "{}: bound regressed by {:.2} bits ({:.1}%)",
                new.function_name,
                bound_delta,
                relative_change * 100.0
            )
        };

        Self {
            function_name: new.function_name.clone(),
            old_bound,
            new_bound,
            bound_delta,
            relative_change,
            strength_changed,
            transformer_changed,
            severity,
            explanation,
        }
    }
}

impl fmt::Display for ContractDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity, self.explanation)
    }
}

// ---------------------------------------------------------------------------
// Regression report
// ---------------------------------------------------------------------------

/// Aggregated regression report across all contracts in a library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Individual deltas for each changed contract.
    pub deltas: Vec<ContractDelta>,
    /// Maximum severity observed.
    pub max_severity: RegressionSeverity,
    /// Number of contracts that regressed.
    pub regressions: usize,
    /// Number of contracts that improved.
    pub improvements: usize,
    /// Number of contracts unchanged.
    pub unchanged: usize,
    /// Total bound change (sum of deltas).
    pub total_delta_bits: f64,
    /// Timestamp of this report.
    pub timestamp: String,
}

impl RegressionReport {
    /// Create an empty report.
    pub fn empty() -> Self {
        Self {
            deltas: Vec::new(),
            max_severity: RegressionSeverity::Info,
            regressions: 0,
            improvements: 0,
            unchanged: 0,
            total_delta_bits: 0.0,
            timestamp: String::new(),
        }
    }

    /// Add a delta to the report.
    pub fn add_delta(&mut self, delta: ContractDelta) {
        self.total_delta_bits += delta.bound_delta;
        if delta.bound_delta > 0.0 {
            self.regressions += 1;
        } else if delta.bound_delta < 0.0 {
            self.improvements += 1;
        } else {
            self.unchanged += 1;
        }
        if delta.severity > self.max_severity {
            self.max_severity = delta.severity;
        }
        self.deltas.push(delta);
    }

    /// Whether the report has any blocking regressions.
    pub fn has_blocking(&self) -> bool {
        self.max_severity.is_blocking()
    }

    /// Summary line.
    pub fn summary(&self) -> String {
        format!(
            "{} regressions, {} improvements, {} unchanged (max severity: {})",
            self.regressions, self.improvements, self.unchanged, self.max_severity
        )
    }
}

impl fmt::Display for RegressionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Regression Report ===")?;
        writeln!(f, "{}", self.summary())?;
        for delta in &self.deltas {
            writeln!(f, "  {}", delta)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Regression analyzer
// ---------------------------------------------------------------------------

/// Compares two sets of contracts and produces a [`RegressionReport`].
#[derive(Debug, Clone)]
pub struct RegressionAnalyzer {
    /// Tolerance for bound changes (deltas below this are Info, not Warning).
    pub tolerance_bits: f64,
    /// Whether to treat strength downgrades as errors.
    pub strict_strength: bool,
}

impl RegressionAnalyzer {
    /// Create a default analyzer with sensible thresholds.
    pub fn new() -> Self {
        Self {
            tolerance_bits: 0.01,
            strict_strength: true,
        }
    }

    /// Analyse the difference between two contract collections.
    pub fn analyze(
        &self,
        old_contracts: &[LeakageContract],
        new_contracts: &[LeakageContract],
    ) -> RegressionReport {
        let old_map: BTreeMap<String, &LeakageContract> = old_contracts
            .iter()
            .map(|c| (c.function_name.clone(), c))
            .collect();
        let new_map: BTreeMap<String, &LeakageContract> = new_contracts
            .iter()
            .map(|c| (c.function_name.clone(), c))
            .collect();

        let mut report = RegressionReport::empty();

        for (name, new_c) in &new_map {
            if let Some(old_c) = old_map.get(name) {
                let delta = ContractDelta::compute(old_c, new_c);
                report.add_delta(delta);
            }
        }

        report
    }
}

impl Default for RegressionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CI report
// ---------------------------------------------------------------------------

/// CI-friendly report suitable for GitHub Actions / GitLab CI output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CIReport {
    /// Whether the overall check passed.
    pub passed: bool,
    /// Exit code suggestion (0 = pass, 1 = fail).
    pub exit_code: i32,
    /// Short one-line summary for CI status.
    pub summary_line: String,
    /// Detailed report body (Markdown-formatted).
    pub markdown_body: String,
    /// The underlying regression report.
    pub regression_report: RegressionReport,
}

impl CIReport {
    /// Build a CI report from a regression report.
    pub fn from_regression(report: RegressionReport) -> Self {
        let passed = !report.has_blocking();
        let exit_code = if passed { 0 } else { 1 };
        let summary_line = if passed {
            format!("✅ Leakage contracts OK: {}", report.summary())
        } else {
            format!("❌ Leakage regression: {}", report.summary())
        };

        let mut md = String::new();
        md.push_str("## Leakage Contract Regression Report\n\n");
        md.push_str(&format!("**Status**: {}\n\n", if passed { "PASS" } else { "FAIL" }));
        md.push_str(&format!("| Function | Old (bits) | New (bits) | Δ | Severity |\n"));
        md.push_str("| --- | --- | --- | --- | --- |\n");
        for d in &report.deltas {
            md.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:+.2} | {} |\n",
                d.function_name, d.old_bound, d.new_bound, d.bound_delta, d.severity
            ));
        }

        Self {
            passed,
            exit_code,
            summary_line,
            markdown_body: md,
            regression_report: report,
        }
    }

    /// Emit as JSON string (for CI integration).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl fmt::Display for CIReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary_line)
    }
}
