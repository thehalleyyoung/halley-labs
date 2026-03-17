use serde::{Deserialize, Serialize};
use crate::conservation_law::ConservationKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationSeverity::Info => write!(f, "Info"),
            ViolationSeverity::Warning => write!(f, "Warning"),
            ViolationSeverity::Error => write!(f, "Error"),
            ViolationSeverity::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub kind: ConservationKind,
    pub severity: ViolationSeverity,
    pub time: f64,
    pub expected: f64,
    pub actual: f64,
    pub message: String,
}

impl Violation {
    pub fn new(kind: ConservationKind, severity: ViolationSeverity, time: f64, expected: f64, actual: f64) -> Self {
        let message = format!(
            "{} violation at t={:.6}: expected {:.6e}, got {:.6e} (diff={:.6e})",
            kind, time, expected, actual, (actual - expected).abs()
        );
        Self { kind, severity, time, expected, actual, message }
    }

    pub fn error(&self) -> f64 {
        (self.actual - self.expected).abs()
    }

    pub fn relative_error(&self) -> f64 {
        if self.expected.abs() < 1e-15 {
            self.error()
        } else {
            self.error() / self.expected.abs()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationReport {
    pub violations: Vec<Violation>,
    pub summary: String,
}

impl ViolationReport {
    pub fn new(violations: Vec<Violation>) -> Self {
        let summary = format!("Found {} violations", violations.len());
        Self { violations, summary }
    }

    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }
}
