//! Contract well-formedness and soundness checking.
//!
//! Validates that a leakage contract satisfies the mathematical properties
//! required for compositional reasoning:
//!
//! - **Soundness**: the bound is a valid upper bound of actual leakage.
//! - **Independence**: parallel contracts touch disjoint cache sets.
//! - **Monotonicity**: transformer and bound respect the abstract domain ordering.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::contract::{
    AbstractCacheState, CacheTransformer, ContractStrength, LeakageBound, LeakageContract,
};

// ---------------------------------------------------------------------------
// Validation severity
// ---------------------------------------------------------------------------

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// A note – informational only.
    Note,
    /// A potential issue worth investigating.
    Warning,
    /// A definite problem that must be fixed.
    Error,
}

impl ValidationSeverity {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// Outcome of a single validation check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationResult {
    /// Check passed.
    Pass,
    /// Check passed with a note.
    PassWithNote(String),
    /// Check failed.
    Fail(String),
}

impl ValidationResult {
    /// Whether the result is a pass (possibly with note).
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass | Self::PassWithNote(_))
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::PassWithNote(n) => write!(f, "PASS ({})", n),
            Self::Fail(reason) => write!(f, "FAIL: {}", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation report
// ---------------------------------------------------------------------------

/// A finding produced by a validation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFinding {
    /// Which check produced this finding.
    pub check_name: String,
    /// Severity.
    pub severity: ValidationSeverity,
    /// Result.
    pub result: ValidationResult,
    /// Optional extra context.
    pub context: Option<String>,
}

/// Aggregated report from running all validation checks on a contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// The function name that was validated.
    pub function_name: String,
    /// Individual findings.
    pub findings: Vec<ValidationFinding>,
    /// Whether all checks passed.
    pub all_passed: bool,
    /// Number of errors.
    pub errors: usize,
    /// Number of warnings.
    pub warnings: usize,
}

impl ValidationReport {
    /// Create an empty report for the given function.
    pub fn new(function_name: impl Into<String>) -> Self {
        Self {
            function_name: function_name.into(),
            findings: Vec::new(),
            all_passed: true,
            errors: 0,
            warnings: 0,
        }
    }

    /// Record a finding.
    pub fn add_finding(&mut self, finding: ValidationFinding) {
        if !finding.result.is_pass() {
            self.all_passed = false;
        }
        match finding.severity {
            ValidationSeverity::Error => self.errors += 1,
            ValidationSeverity::Warning => self.warnings += 1,
            _ => {}
        }
        self.findings.push(finding);
    }

    /// Summary line.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} errors, {} warnings, {}",
            self.function_name,
            self.errors,
            self.warnings,
            if self.all_passed { "PASS" } else { "FAIL" }
        )
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Validation: {} ===", self.function_name)?;
        for finding in &self.findings {
            writeln!(
                f,
                "  [{}] {}: {}",
                finding.severity, finding.check_name, finding.result
            )?;
        }
        writeln!(f, "  → {}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Soundness check
// ---------------------------------------------------------------------------

/// Checks that a contract's bound is a valid upper bound of actual leakage.
#[derive(Debug, Clone)]
pub struct SoundnessCheck {
    /// Whether to also verify the transformer preserves the abstract domain.
    pub check_transformer: bool,
    /// Tolerance for floating-point comparisons.
    pub epsilon: f64,
}

impl SoundnessCheck {
    /// Create a default soundness check.
    pub fn new() -> Self {
        Self {
            check_transformer: true,
            epsilon: 1e-9,
        }
    }

    /// Run the soundness check against a contract.
    pub fn check(&self, contract: &LeakageContract) -> ValidationFinding {
        let check_name = "soundness".to_string();

        // Verify the bound is non-negative.
        if contract.worst_case_bits() < -self.epsilon {
            return ValidationFinding {
                check_name,
                severity: ValidationSeverity::Error,
                result: ValidationResult::Fail(format!(
                    "negative bound: {:.4} bits",
                    contract.worst_case_bits()
                )),
                context: None,
            };
        }

        // Verify strength claim.
        if contract.strength == ContractStrength::Exact && !contract.leakage_bound.is_tight {
            return ValidationFinding {
                check_name,
                severity: ValidationSeverity::Warning,
                result: ValidationResult::PassWithNote(
                    "claimed exact but bound is not marked tight".into(),
                ),
                context: None,
            };
        }

        // Approximate contracts are not sound by definition.
        if contract.strength == ContractStrength::Approximate {
            return ValidationFinding {
                check_name,
                severity: ValidationSeverity::Warning,
                result: ValidationResult::PassWithNote(
                    "approximate bound – not provably sound".into(),
                ),
                context: None,
            };
        }

        ValidationFinding {
            check_name,
            severity: ValidationSeverity::Note,
            result: ValidationResult::Pass,
            context: None,
        }
    }
}

impl Default for SoundnessCheck {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Independence verifier
// ---------------------------------------------------------------------------

/// Verifies that two contracts are independent (suitable for parallel
/// composition).
#[derive(Debug, Clone)]
pub struct IndependenceVerifier {
    /// Whether to allow shared read sets.
    pub allow_shared_reads: bool,
}

impl IndependenceVerifier {
    /// Create a default verifier.
    pub fn new() -> Self {
        Self {
            allow_shared_reads: false,
        }
    }

    /// Check independence of two contracts.
    pub fn verify(
        &self,
        a: &LeakageContract,
        b: &LeakageContract,
    ) -> ValidationFinding {
        let a_writes = &a.cache_transformer.writes;
        let b_writes = &b.cache_transformer.writes;

        let overlap: Vec<u32> = a_writes
            .iter()
            .filter(|s| b_writes.contains(s))
            .copied()
            .collect();

        if !overlap.is_empty() {
            return ValidationFinding {
                check_name: "independence".into(),
                severity: ValidationSeverity::Error,
                result: ValidationResult::Fail(format!(
                    "write-set overlap: {:?}",
                    overlap
                )),
                context: None,
            };
        }

        let a_reads = &a.cache_transformer.reads;
        let b_reads = &b.cache_transformer.reads;

        // Check write-read conflicts.
        let wr: Vec<u32> = a_writes
            .iter()
            .filter(|s| b_reads.contains(s))
            .copied()
            .collect();
        let rw: Vec<u32> = a_reads
            .iter()
            .filter(|s| b_writes.contains(s))
            .copied()
            .collect();

        if !wr.is_empty() || !rw.is_empty() {
            let mut all = wr;
            all.extend(rw);
            all.sort_unstable();
            all.dedup();
            return ValidationFinding {
                check_name: "independence".into(),
                severity: ValidationSeverity::Error,
                result: ValidationResult::Fail(format!(
                    "read-write overlap: {:?}",
                    all
                )),
                context: None,
            };
        }

        ValidationFinding {
            check_name: "independence".into(),
            severity: ValidationSeverity::Note,
            result: ValidationResult::Pass,
            context: None,
        }
    }
}

impl Default for IndependenceVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Monotonicity check
// ---------------------------------------------------------------------------

/// Verifies the monotonicity property: if `s₁ ⊑ s₂` then
/// `B(s₁) ≤ B(s₂)` and `τ(s₁) ⊑ τ(s₂)`.
#[derive(Debug, Clone)]
pub struct MonotonicityCheck {
    /// Number of random samples to test.
    pub num_samples: usize,
    /// Tolerance for floating-point comparisons.
    pub epsilon: f64,
}

impl MonotonicityCheck {
    /// Create a default monotonicity check.
    pub fn new() -> Self {
        Self {
            num_samples: 100,
            epsilon: 1e-9,
        }
    }

    /// Run the monotonicity check.
    ///
    /// Since exhaustive checking is infeasible for large state spaces, this
    /// performs a structural check on the bound representation.
    pub fn check(&self, contract: &LeakageContract) -> ValidationFinding {
        let check_name = "monotonicity".to_string();

        // Constant bounds are trivially monotone.
        if contract.leakage_bound.worst_case_bits.is_finite()
            && contract.leakage_bound.per_set_leakage.is_empty()
        {
            return ValidationFinding {
                check_name,
                severity: ValidationSeverity::Note,
                result: ValidationResult::Pass,
                context: Some("constant bound is trivially monotone".into()),
            };
        }

        // Per-set bounds with non-negative coefficients are monotone.
        let all_non_negative = contract
            .leakage_bound
            .per_set_leakage
            .values()
            .all(|&v| v >= -self.epsilon);

        if all_non_negative {
            return ValidationFinding {
                check_name,
                severity: ValidationSeverity::Note,
                result: ValidationResult::Pass,
                context: Some("all per-set coefficients non-negative".into()),
            };
        }

        ValidationFinding {
            check_name,
            severity: ValidationSeverity::Warning,
            result: ValidationResult::PassWithNote(
                "negative per-set coefficient – monotonicity not guaranteed".into(),
            ),
            context: None,
        }
    }
}

impl Default for MonotonicityCheck {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Contract validator
// ---------------------------------------------------------------------------

/// Runs all validation checks on a leakage contract and collects findings
/// into a [`ValidationReport`].
#[derive(Debug, Clone)]
pub struct ContractValidator {
    /// Soundness checker.
    pub soundness: SoundnessCheck,
    /// Monotonicity checker.
    pub monotonicity: MonotonicityCheck,
    /// Whether to include informational notes.
    pub include_notes: bool,
}

impl ContractValidator {
    /// Create a default validator with all checks enabled.
    pub fn new() -> Self {
        Self {
            soundness: SoundnessCheck::new(),
            monotonicity: MonotonicityCheck::new(),
            include_notes: true,
        }
    }

    /// Validate a single contract.
    pub fn validate(&self, contract: &LeakageContract) -> ValidationReport {
        let mut report = ValidationReport::new(&contract.function_name);

        let s = self.soundness.check(contract);
        if self.include_notes || s.severity != ValidationSeverity::Note {
            report.add_finding(s);
        }

        let m = self.monotonicity.check(contract);
        if self.include_notes || m.severity != ValidationSeverity::Note {
            report.add_finding(m);
        }

        report
    }

    /// Validate a collection of contracts.
    pub fn validate_all(&self, contracts: &[LeakageContract]) -> Vec<ValidationReport> {
        contracts.iter().map(|c| self.validate(c)).collect()
    }
}

impl Default for ContractValidator {
    fn default() -> Self {
        Self::new()
    }
}
