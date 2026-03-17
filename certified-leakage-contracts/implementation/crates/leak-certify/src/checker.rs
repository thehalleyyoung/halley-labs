//! Independent verification of certificates without re-running full analysis.
//!
//! [`CertificateChecker`] walks a certificate's claims and evidence, delegates to
//! the [`WitnessChecker`](crate::WitnessChecker), and produces a [`CheckResult`].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::certificate::Certificate;
use crate::witness::{Witness, WitnessChecker};

// ---------------------------------------------------------------------------
// CheckStatus
// ---------------------------------------------------------------------------

/// Outcome of checking a single certificate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckStatus {
    /// All claims verified successfully.
    Verified,
    /// At least one claim could not be verified.
    Failed,
    /// Checking was skipped (e.g. unsupported witness kind).
    Skipped,
    /// The certificate structure is malformed.
    Malformed,
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verified => write!(f, "verified"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
            Self::Malformed => write!(f, "malformed"),
        }
    }
}

// ---------------------------------------------------------------------------
// CheckResult
// ---------------------------------------------------------------------------

/// Full result of checking a certificate or chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Overall status.
    pub status: CheckStatus,
    /// Per-claim diagnostic messages.
    pub diagnostics: Vec<String>,
    /// When the check was performed.
    pub checked_at: DateTime<Utc>,
    /// Duration of the check in milliseconds.
    pub duration_ms: u64,
}

impl CheckResult {
    /// Convenience constructor for a successful verification.
    pub fn verified() -> Self {
        Self {
            status: CheckStatus::Verified,
            diagnostics: Vec::new(),
            checked_at: Utc::now(),
            duration_ms: 0,
        }
    }

    /// Convenience constructor for a failed verification.
    pub fn failed(diagnostics: Vec<String>) -> Self {
        Self {
            status: CheckStatus::Failed,
            diagnostics,
            checked_at: Utc::now(),
            duration_ms: 0,
        }
    }

    /// Whether the check passed.
    pub fn is_ok(&self) -> bool {
        self.status == CheckStatus::Verified
    }
}

// ---------------------------------------------------------------------------
// CertificateChecker
// ---------------------------------------------------------------------------

/// Configurable checker that validates certificates and their witnesses.
#[derive(Debug, Clone)]
pub struct CertificateChecker {
    /// Inner witness checker.
    witness_checker: WitnessChecker,
    /// Whether to verify content hashes.
    pub verify_hashes: bool,
    /// Whether to verify evidence integrity.
    pub verify_evidence: bool,
}

impl Default for CertificateChecker {
    fn default() -> Self {
        Self {
            witness_checker: WitnessChecker::default(),
            verify_hashes: true,
            verify_evidence: true,
        }
    }
}

impl CertificateChecker {
    /// Create a checker with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the inner witness checker.
    pub fn with_witness_checker(mut self, wc: WitnessChecker) -> Self {
        self.witness_checker = wc;
        self
    }

    /// Check a single certificate, including hash, evidence, and witnesses.
    pub fn check(&self, certificate: &Certificate, witnesses: &[Witness]) -> CheckResult {
        let start = std::time::Instant::now();
        let mut diagnostics = Vec::new();

        // 1. Hash integrity.
        if self.verify_hashes && !certificate.verify_hash() {
            diagnostics.push("certificate hash mismatch".into());
        }

        // 2. Evidence integrity.
        if self.verify_evidence {
            for (i, ev) in certificate.evidence.iter().enumerate() {
                if !ev.verify_integrity() {
                    diagnostics.push(format!("evidence[{}] integrity check failed", i));
                }
            }
        }

        // 3. Witness checks.
        for (i, w) in witnesses.iter().enumerate() {
            let result = self.witness_checker.check(w);
            if !result.passed {
                for msg in result.messages {
                    diagnostics.push(format!("witness[{}]: {}", i, msg));
                }
            }
        }

        // 4. Claims present.
        if certificate.claims.is_empty() {
            diagnostics.push("certificate has no claims".into());
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let status = if diagnostics.is_empty() {
            CheckStatus::Verified
        } else {
            CheckStatus::Failed
        };

        CheckResult {
            status,
            diagnostics,
            checked_at: Utc::now(),
            duration_ms,
        }
    }

    /// Access the inner witness checker.
    pub fn witness_checker(&self) -> &WitnessChecker {
        &self.witness_checker
    }
}
