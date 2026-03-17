//! Verification module for independent certificate checking.
//!
//! Provides tools to independently verify L3, T2, and futility certificates,
//! check dual solutions, and validate partitions.

pub mod bound_checker;
pub mod dual_checker;
pub mod partition_checker;

pub use bound_checker::BoundChecker;
pub use dual_checker::DualChecker;
pub use partition_checker::PartitionChecker;

use serde::{Deserialize, Serialize};

/// Overall verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub all_passed: bool,
    pub num_checks: usize,
    pub num_passed: usize,
    pub num_failed: usize,
    pub num_warnings: usize,
    pub details: Vec<VerificationCheck>,
}

/// A single verification check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub name: String,
    pub passed: bool,
    pub severity: CheckSeverity,
    pub message: String,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
}

/// Severity of a verification check failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckSeverity {
    Error,
    Warning,
    Info,
}

impl VerificationResult {
    pub fn new() -> Self {
        Self {
            all_passed: true,
            num_checks: 0,
            num_passed: 0,
            num_failed: 0,
            num_warnings: 0,
            details: Vec::new(),
        }
    }

    pub fn add_check(&mut self, check: VerificationCheck) {
        self.num_checks += 1;
        if check.passed {
            self.num_passed += 1;
        } else {
            match check.severity {
                CheckSeverity::Error => {
                    self.num_failed += 1;
                    self.all_passed = false;
                }
                CheckSeverity::Warning => {
                    self.num_warnings += 1;
                }
                CheckSeverity::Info => {}
            }
        }
        self.details.push(check);
    }

    pub fn merge(&mut self, other: VerificationResult) {
        for check in other.details {
            self.add_check(check);
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Verification: {}/{} passed, {} failed, {} warnings",
            self.num_passed, self.num_checks, self.num_failed, self.num_warnings
        )
    }
}

impl Default for VerificationResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_new() {
        let result = VerificationResult::new();
        assert!(result.all_passed);
        assert_eq!(result.num_checks, 0);
    }

    #[test]
    fn test_add_passing_check() {
        let mut result = VerificationResult::new();
        result.add_check(VerificationCheck {
            name: "test".to_string(),
            passed: true,
            severity: CheckSeverity::Error,
            message: "OK".to_string(),
            value: None,
            threshold: None,
        });
        assert!(result.all_passed);
        assert_eq!(result.num_passed, 1);
    }

    #[test]
    fn test_add_failing_check() {
        let mut result = VerificationResult::new();
        result.add_check(VerificationCheck {
            name: "test".to_string(),
            passed: false,
            severity: CheckSeverity::Error,
            message: "FAIL".to_string(),
            value: None,
            threshold: None,
        });
        assert!(!result.all_passed);
        assert_eq!(result.num_failed, 1);
    }

    #[test]
    fn test_warning_does_not_fail() {
        let mut result = VerificationResult::new();
        result.add_check(VerificationCheck {
            name: "test".to_string(),
            passed: false,
            severity: CheckSeverity::Warning,
            message: "WARN".to_string(),
            value: None,
            threshold: None,
        });
        assert!(result.all_passed);
        assert_eq!(result.num_warnings, 1);
    }

    #[test]
    fn test_merge() {
        let mut a = VerificationResult::new();
        a.add_check(VerificationCheck {
            name: "a".to_string(),
            passed: true,
            severity: CheckSeverity::Error,
            message: "OK".to_string(),
            value: None,
            threshold: None,
        });
        let mut b = VerificationResult::new();
        b.add_check(VerificationCheck {
            name: "b".to_string(),
            passed: false,
            severity: CheckSeverity::Error,
            message: "FAIL".to_string(),
            value: None,
            threshold: None,
        });
        a.merge(b);
        assert!(!a.all_passed);
        assert_eq!(a.num_checks, 2);
    }

    #[test]
    fn test_summary() {
        let mut result = VerificationResult::new();
        result.add_check(VerificationCheck {
            name: "a".to_string(),
            passed: true,
            severity: CheckSeverity::Error,
            message: "ok".to_string(),
            value: None,
            threshold: None,
        });
        let summary = result.summary();
        assert!(summary.contains("1/1 passed"));
    }
}
