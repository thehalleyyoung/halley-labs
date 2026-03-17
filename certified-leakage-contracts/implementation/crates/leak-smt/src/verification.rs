//! Contract verification orchestration.
//!
//! Ties together the SMT encoding and solver layers to verify whether a
//! binary's leakage contract holds, producing structured reports.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, Instant};

use crate::encoding::LeakageEncoder;
use crate::solver::{SmtSolver, SolverResult};

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// Outcome of a single contract verification query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationResult {
    /// The contract is verified: no leakage beyond the declared contract.
    Verified,
    /// A counterexample was found violating the contract.
    Violated {
        /// Human-readable description of the violation.
        message: String,
    },
    /// The solver could not determine the result within resource limits.
    Inconclusive {
        reason: String,
    },
    /// An error occurred during verification.
    Error {
        message: String,
    },
}

impl VerificationResult {
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified)
    }

    pub fn is_violated(&self) -> bool {
        matches!(self, VerificationResult::Violated { .. })
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationResult::Verified => write!(f, "VERIFIED"),
            VerificationResult::Violated { message } => write!(f, "VIOLATED: {}", message),
            VerificationResult::Inconclusive { reason } => write!(f, "INCONCLUSIVE: {}", reason),
            VerificationResult::Error { message } => write!(f, "ERROR: {}", message),
        }
    }
}

// ---------------------------------------------------------------------------
// VerificationReport
// ---------------------------------------------------------------------------

/// Aggregated report for an entire verification run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Overall result: verified only when *all* sub-queries are verified.
    pub overall: VerificationResult,
    /// Per-obligation results (one entry per assertion group / contract clause).
    pub obligations: Vec<ObligationResult>,
    /// Total wall-clock time for the verification run.
    pub elapsed: Duration,
    /// Number of SMT queries issued.
    pub num_queries: usize,
}

/// Result of a single verification obligation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationResult {
    /// A descriptive label for this obligation.
    pub label: String,
    /// The verification outcome.
    pub result: VerificationResult,
    /// Time spent on this obligation.
    pub elapsed: Duration,
}

impl VerificationReport {
    /// Create a report indicating that all obligations were verified.
    pub fn all_verified(obligations: Vec<ObligationResult>, elapsed: Duration) -> Self {
        Self {
            overall: VerificationResult::Verified,
            num_queries: obligations.len(),
            obligations,
            elapsed,
        }
    }

    /// Create a report with mixed results.
    pub fn from_obligations(obligations: Vec<ObligationResult>, elapsed: Duration) -> Self {
        let overall = if obligations.iter().all(|o| o.result.is_verified()) {
            VerificationResult::Verified
        } else if obligations.iter().any(|o| o.result.is_violated()) {
            VerificationResult::Violated {
                message: "one or more obligations violated".to_string(),
            }
        } else {
            VerificationResult::Inconclusive {
                reason: "some obligations could not be resolved".to_string(),
            }
        };
        Self {
            overall,
            num_queries: obligations.len(),
            obligations,
            elapsed,
        }
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Verification Report ===")?;
        writeln!(f, "Overall: {}", self.overall)?;
        writeln!(f, "Obligations: {}", self.obligations.len())?;
        for (i, o) in self.obligations.iter().enumerate() {
            writeln!(f, "  [{}] {}: {} ({:?})", i, o.label, o.result, o.elapsed)?;
        }
        writeln!(f, "Total time: {:?}", self.elapsed)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ContractVerifier
// ---------------------------------------------------------------------------

/// Top-level entry point for verifying leakage contracts.
///
/// Coordinates encoding, solver invocation, and report generation.
#[derive(Debug)]
pub struct ContractVerifier {
    encoder: LeakageEncoder,
}

impl ContractVerifier {
    /// Create a new verifier with the given encoder.
    pub fn new(encoder: LeakageEncoder) -> Self {
        Self { encoder }
    }

    /// Run verification using the provided solver, returning a full report.
    pub fn verify(&mut self, solver: &mut dyn SmtSolver) -> VerificationReport {
        let start = Instant::now();
        let script = self.encoder.build_script();
        let pool = self.encoder.pool();

        let result = solver.check_sat(&script, pool);

        let vr = match result {
            SolverResult::Unsat => VerificationResult::Verified,
            SolverResult::Sat(_) => VerificationResult::Violated {
                message: "satisfying assignment found – contract violated".to_string(),
            },
            SolverResult::Unknown(reason) => VerificationResult::Inconclusive { reason },
            SolverResult::Error(msg) => VerificationResult::Error { message: msg },
            SolverResult::Timeout => VerificationResult::Inconclusive {
                reason: "solver timeout".to_string(),
            },
        };

        let elapsed = start.elapsed();
        let obligation = ObligationResult {
            label: "main".to_string(),
            result: vr.clone(),
            elapsed,
        };

        VerificationReport::from_obligations(vec![obligation], elapsed)
    }

    /// Access the underlying encoder.
    pub fn encoder(&self) -> &LeakageEncoder {
        &self.encoder
    }

    /// Mutable access to the underlying encoder.
    pub fn encoder_mut(&mut self) -> &mut LeakageEncoder {
        &mut self.encoder
    }
}
