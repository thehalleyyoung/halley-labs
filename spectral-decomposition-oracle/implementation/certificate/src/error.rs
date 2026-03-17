//! Certificate error types for decomposition quality verification.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Primary error type for certificate operations.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum CertificateError {
    #[error("Invalid partition: {reason}")]
    InvalidPartition {
        reason: String,
        num_blocks: Option<usize>,
        num_variables: Option<usize>,
    },

    #[error("Bound verification failed: {reason}")]
    BoundVerificationFailed {
        reason: String,
        claimed_bound: f64,
        computed_bound: f64,
        tolerance: f64,
    },

    #[error("Numerical precision error: {reason}")]
    NumericalPrecision {
        reason: String,
        value: f64,
        threshold: f64,
    },

    #[error("Incomplete data: {missing_field}")]
    IncompleteData {
        missing_field: String,
        context: String,
    },

    #[error("Certificate expired: issued {issued_at}, expired {expired_at}, id: {certificate_id}")]
    CertificateExpired {
        issued_at: String,
        expired_at: String,
        certificate_id: String,
    },

    #[error("Validation failed: {reason}")]
    ValidationFailed {
        reason: String,
        checks_passed: usize,
        checks_total: usize,
        failures: Vec<String>,
    },
}

impl CertificateError {
    pub fn invalid_partition(reason: impl Into<String>) -> Self {
        Self::InvalidPartition {
            reason: reason.into(),
            num_blocks: None,
            num_variables: None,
        }
    }

    pub fn invalid_partition_with_details(
        reason: impl Into<String>,
        num_blocks: usize,
        num_variables: usize,
    ) -> Self {
        Self::InvalidPartition {
            reason: reason.into(),
            num_blocks: Some(num_blocks),
            num_variables: Some(num_variables),
        }
    }

    pub fn bound_verification_failed(
        reason: impl Into<String>,
        claimed: f64,
        computed: f64,
        tolerance: f64,
    ) -> Self {
        Self::BoundVerificationFailed {
            reason: reason.into(),
            claimed_bound: claimed,
            computed_bound: computed,
            tolerance,
        }
    }

    pub fn numerical_precision(reason: impl Into<String>, value: f64, threshold: f64) -> Self {
        Self::NumericalPrecision {
            reason: reason.into(),
            value,
            threshold,
        }
    }

    pub fn incomplete_data(field: impl Into<String>, context: impl Into<String>) -> Self {
        Self::IncompleteData {
            missing_field: field.into(),
            context: context.into(),
        }
    }

    pub fn certificate_expired(
        issued: impl Into<String>,
        expired: impl Into<String>,
        id: impl Into<String>,
    ) -> Self {
        Self::CertificateExpired {
            issued_at: issued.into(),
            expired_at: expired.into(),
            certificate_id: id.into(),
        }
    }

    pub fn validation_failed(
        reason: impl Into<String>,
        passed: usize,
        total: usize,
        failures: Vec<String>,
    ) -> Self {
        Self::ValidationFailed {
            reason: reason.into(),
            checks_passed: passed,
            checks_total: total,
            failures,
        }
    }

    /// Returns a severity score from 0.0 (informational) to 1.0 (critical).
    pub fn severity(&self) -> f64 {
        match self {
            Self::NumericalPrecision { .. } => 0.3,
            Self::IncompleteData { .. } => 0.4,
            Self::CertificateExpired { .. } => 0.5,
            Self::InvalidPartition { .. } => 0.7,
            Self::BoundVerificationFailed { .. } => 0.9,
            Self::ValidationFailed { .. } => 1.0,
        }
    }

    /// Whether this error is recoverable (can potentially be fixed automatically).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::NumericalPrecision { .. }
                | Self::IncompleteData { .. }
                | Self::CertificateExpired { .. }
        )
    }

    /// Short code identifying the error category.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::InvalidPartition { .. } => "CERT-E001",
            Self::BoundVerificationFailed { .. } => "CERT-E002",
            Self::NumericalPrecision { .. } => "CERT-W001",
            Self::IncompleteData { .. } => "CERT-W002",
            Self::CertificateExpired { .. } => "CERT-W003",
            Self::ValidationFailed { .. } => "CERT-E003",
        }
    }

    /// Returns structured details suitable for logging.
    pub fn details(&self) -> ErrorDetails {
        match self {
            Self::InvalidPartition {
                reason,
                num_blocks,
                num_variables,
            } => ErrorDetails {
                code: self.error_code().to_string(),
                message: reason.clone(),
                severity: self.severity(),
                recoverable: self.is_recoverable(),
                context: {
                    let mut ctx = indexmap::IndexMap::new();
                    if let Some(nb) = num_blocks {
                        ctx.insert("num_blocks".to_string(), nb.to_string());
                    }
                    if let Some(nv) = num_variables {
                        ctx.insert("num_variables".to_string(), nv.to_string());
                    }
                    ctx
                },
            },
            Self::BoundVerificationFailed {
                reason,
                claimed_bound,
                computed_bound,
                tolerance,
            } => ErrorDetails {
                code: self.error_code().to_string(),
                message: reason.clone(),
                severity: self.severity(),
                recoverable: self.is_recoverable(),
                context: {
                    let mut ctx = indexmap::IndexMap::new();
                    ctx.insert("claimed_bound".to_string(), claimed_bound.to_string());
                    ctx.insert("computed_bound".to_string(), computed_bound.to_string());
                    ctx.insert("tolerance".to_string(), tolerance.to_string());
                    ctx
                },
            },
            Self::NumericalPrecision {
                reason,
                value,
                threshold,
            } => ErrorDetails {
                code: self.error_code().to_string(),
                message: reason.clone(),
                severity: self.severity(),
                recoverable: self.is_recoverable(),
                context: {
                    let mut ctx = indexmap::IndexMap::new();
                    ctx.insert("value".to_string(), value.to_string());
                    ctx.insert("threshold".to_string(), threshold.to_string());
                    ctx
                },
            },
            Self::IncompleteData {
                missing_field,
                context,
            } => ErrorDetails {
                code: self.error_code().to_string(),
                message: format!("Missing: {}", missing_field),
                severity: self.severity(),
                recoverable: self.is_recoverable(),
                context: {
                    let mut ctx = indexmap::IndexMap::new();
                    ctx.insert("missing_field".to_string(), missing_field.clone());
                    ctx.insert("context".to_string(), context.clone());
                    ctx
                },
            },
            Self::CertificateExpired {
                issued_at,
                expired_at,
                certificate_id,
            } => ErrorDetails {
                code: self.error_code().to_string(),
                message: format!("Expired certificate {}", certificate_id),
                severity: self.severity(),
                recoverable: self.is_recoverable(),
                context: {
                    let mut ctx = indexmap::IndexMap::new();
                    ctx.insert("issued_at".to_string(), issued_at.clone());
                    ctx.insert("expired_at".to_string(), expired_at.clone());
                    ctx.insert("certificate_id".to_string(), certificate_id.clone());
                    ctx
                },
            },
            Self::ValidationFailed {
                reason,
                checks_passed,
                checks_total,
                failures,
            } => ErrorDetails {
                code: self.error_code().to_string(),
                message: reason.clone(),
                severity: self.severity(),
                recoverable: self.is_recoverable(),
                context: {
                    let mut ctx = indexmap::IndexMap::new();
                    ctx.insert("checks_passed".to_string(), checks_passed.to_string());
                    ctx.insert("checks_total".to_string(), checks_total.to_string());
                    ctx.insert("failures".to_string(), failures.join("; "));
                    ctx
                },
            },
        }
    }
}

/// Structured error details for logging and reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub code: String,
    pub message: String,
    pub severity: f64,
    pub recoverable: bool,
    pub context: indexmap::IndexMap<String, String>,
}

impl fmt::Display for ErrorDetails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] (severity={:.1}) {}: {}",
            self.code,
            self.severity,
            if self.recoverable {
                "RECOVERABLE"
            } else {
                "FATAL"
            },
            self.message
        )
    }
}

/// Result type alias for certificate operations.
pub type CertificateResult<T> = Result<T, CertificateError>;

/// Collects multiple certificate errors during batch validation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorCollector {
    errors: Vec<CertificateError>,
    warnings: Vec<CertificateError>,
}

impl ErrorCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_error(&mut self, err: CertificateError) {
        self.errors.push(err);
    }

    pub fn push_warning(&mut self, warn: CertificateError) {
        self.warnings.push(warn);
    }

    pub fn push_auto(&mut self, err: CertificateError) {
        if err.severity() >= 0.6 {
            self.errors.push(err);
        } else {
            self.warnings.push(err);
        }
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty()
    }

    pub fn errors(&self) -> &[CertificateError] {
        &self.errors
    }

    pub fn warnings(&self) -> &[CertificateError] {
        &self.warnings
    }

    pub fn all_issues(&self) -> Vec<&CertificateError> {
        self.errors.iter().chain(self.warnings.iter()).collect()
    }

    /// Converts to a result: Ok if no errors, Err with a summary ValidationFailed otherwise.
    pub fn into_result(self) -> CertificateResult<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            let failures: Vec<String> = self.errors.iter().map(|e| e.to_string()).collect();
            let total = failures.len() + self.warnings.len();
            Err(CertificateError::validation_failed(
                format!("{} errors, {} warnings", self.errors.len(), self.warnings.len()),
                self.warnings.len(),
                total,
                failures,
            ))
        }
    }

    pub fn merge(&mut self, other: ErrorCollector) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }

    pub fn summary(&self) -> String {
        format!(
            "ErrorCollector: {} errors, {} warnings",
            self.errors.len(),
            self.warnings.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_partition_error() {
        let err = CertificateError::invalid_partition("blocks overlap");
        assert!(err.to_string().contains("blocks overlap"));
        assert_eq!(err.error_code(), "CERT-E001");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_invalid_partition_with_details() {
        let err = CertificateError::invalid_partition_with_details("size mismatch", 3, 100);
        if let CertificateError::InvalidPartition {
            num_blocks,
            num_variables,
            ..
        } = &err
        {
            assert_eq!(*num_blocks, Some(3));
            assert_eq!(*num_variables, Some(100));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_bound_verification_failed() {
        let err = CertificateError::bound_verification_failed("gap too large", 1.0, 2.0, 0.01);
        assert!(err.severity() > 0.8);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_numerical_precision() {
        let err = CertificateError::numerical_precision("denormalized", 1e-320, 1e-300);
        assert!(err.is_recoverable());
        assert!(err.severity() < 0.5);
    }

    #[test]
    fn test_incomplete_data() {
        let err = CertificateError::incomplete_data("dual_values", "L3 bound");
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_certificate_expired() {
        let err = CertificateError::certificate_expired("2024-01-01", "2024-06-01", "cert-123");
        assert!(err.is_recoverable());
        assert!(err.to_string().contains("cert-123"));
    }

    #[test]
    fn test_validation_failed() {
        let err = CertificateError::validation_failed(
            "multiple failures",
            8,
            10,
            vec!["check A failed".into(), "check B failed".into()],
        );
        assert_eq!(err.severity(), 1.0);
    }

    #[test]
    fn test_error_details() {
        let err = CertificateError::bound_verification_failed("mismatch", 5.0, 7.0, 0.1);
        let details = err.details();
        assert_eq!(details.code, "CERT-E002");
        assert!(details.context.contains_key("claimed_bound"));
        assert!(details.context.contains_key("computed_bound"));
    }

    #[test]
    fn test_error_collector_empty() {
        let collector = ErrorCollector::new();
        assert!(collector.is_empty());
        assert!(!collector.has_errors());
        assert_eq!(collector.error_count(), 0);
    }

    #[test]
    fn test_error_collector_push_and_count() {
        let mut collector = ErrorCollector::new();
        collector.push_error(CertificateError::invalid_partition("test"));
        collector.push_warning(CertificateError::numerical_precision("warn", 0.0, 1.0));
        assert_eq!(collector.error_count(), 1);
        assert_eq!(collector.warning_count(), 1);
        assert!(collector.has_errors());
        assert!(collector.has_warnings());
    }

    #[test]
    fn test_error_collector_into_result_ok() {
        let collector = ErrorCollector::new();
        assert!(collector.into_result().is_ok());
    }

    #[test]
    fn test_error_collector_into_result_err() {
        let mut collector = ErrorCollector::new();
        collector.push_error(CertificateError::invalid_partition("bad"));
        let result = collector.into_result();
        assert!(result.is_err());
    }

    #[test]
    fn test_error_collector_auto_classify() {
        let mut collector = ErrorCollector::new();
        collector.push_auto(CertificateError::numerical_precision("low sev", 0.0, 1.0));
        collector.push_auto(CertificateError::invalid_partition("high sev"));
        assert_eq!(collector.warning_count(), 1);
        assert_eq!(collector.error_count(), 1);
    }

    #[test]
    fn test_error_collector_merge() {
        let mut a = ErrorCollector::new();
        a.push_error(CertificateError::invalid_partition("a"));
        let mut b = ErrorCollector::new();
        b.push_error(CertificateError::invalid_partition("b"));
        a.merge(b);
        assert_eq!(a.error_count(), 2);
    }

    #[test]
    fn test_error_details_display() {
        let err = CertificateError::numerical_precision("rounding", 1e-16, 1e-12);
        let details = err.details();
        let display = format!("{}", details);
        assert!(display.contains("CERT-W001"));
        assert!(display.contains("RECOVERABLE"));
    }
}
