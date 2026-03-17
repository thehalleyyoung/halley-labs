//! Certificate validation: checks internal consistency of coverage certificates.
//!
//! Validates that sample counts are sufficient per Hoeffding's inequality,
//! kappa computation is correct, verified regions are non-overlapping,
//! and epsilon bounds are valid.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xr_types::certificate::{
    CertificateGrade, CoverageCertificate,
    VerifiedRegion,
};
use xr_types::{ElementId, NUM_BODY_PARAMS};

use crate::hoeffding::HoeffdingBound;

// ──────────────────── Validation Issue ────────────────────────────────────

/// An issue found during certificate validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Category of the issue.
    pub category: IssueCategory,
    /// Human-readable description.
    pub description: String,
    /// The expected value (if applicable).
    pub expected: Option<String>,
    /// The actual value found.
    pub actual: Option<String>,
}

impl ValidationIssue {
    /// Create a new validation issue.
    pub fn new(
        severity: IssueSeverity,
        category: IssueCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            category,
            description: description.into(),
            expected: None,
            actual: None,
        }
    }

    /// Add expected/actual values.
    pub fn with_values(
        mut self,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        self.expected = Some(expected.into());
        self.actual = Some(actual.into());
        self
    }

    /// Whether this issue is a blocker (error or critical).
    pub fn is_blocker(&self) -> bool {
        matches!(
            self.severity,
            IssueSeverity::Error | IssueSeverity::Critical
        )
    }
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:?}] {:?}: {}",
            self.severity, self.category, self.description
        )?;
        if let (Some(exp), Some(act)) = (&self.expected, &self.actual) {
            write!(f, " (expected: {}, actual: {})", exp, act)?;
        }
        Ok(())
    }
}

/// Severity levels for validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Categories of validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueCategory {
    SampleCount,
    KappaComputation,
    EpsilonBound,
    DeltaParameter,
    RegionOverlap,
    RegionVolume,
    GradeConsistency,
    ViolationConsistency,
    MetadataIntegrity,
    SampleIntegrity,
    TimingConsistency,
}

// ──────────────────── Certificate Validator ───────────────────────────────

/// Validates the internal consistency of coverage certificates.
pub struct CertificateValidator {
    /// Minimum acceptable samples for a valid certificate.
    pub min_samples: usize,
    /// Maximum allowed delta.
    pub max_delta: f64,
    /// Maximum allowed epsilon for a useful certificate.
    pub max_useful_epsilon: f64,
    /// Whether to check region overlap (can be expensive).
    pub check_overlap: bool,
}

impl CertificateValidator {
    /// Create a validator with default settings.
    pub fn new() -> Self {
        Self {
            min_samples: 10,
            max_delta: 0.5,
            max_useful_epsilon: 0.5,
            check_overlap: true,
        }
    }

    /// Create with custom minimum sample count.
    pub fn with_min_samples(mut self, n: usize) -> Self {
        self.min_samples = n;
        self
    }

    /// Enable or disable overlap checking.
    pub fn with_overlap_check(mut self, enabled: bool) -> Self {
        self.check_overlap = enabled;
        self
    }

    /// Validate a certificate and return all issues found.
    pub fn validate(&self, cert: &CoverageCertificate) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        self.check_sample_count(cert, &mut issues);
        self.check_kappa(cert, &mut issues);
        self.check_epsilon_bounds(cert, &mut issues);
        self.check_delta(cert, &mut issues);
        self.check_grade_consistency(cert, &mut issues);
        self.check_violations(cert, &mut issues);
        self.check_sample_integrity(cert, &mut issues);
        self.check_region_volumes(cert, &mut issues);
        self.check_metadata(cert, &mut issues);
        self.check_timing(cert, &mut issues);

        if self.check_overlap {
            self.check_region_overlap(cert, &mut issues);
        }

        issues
    }

    /// High-level soundness check: is this certificate trustworthy?
    pub fn check_soundness(&self, cert: &CoverageCertificate) -> bool {
        let issues = self.validate(cert);
        !issues.iter().any(|i| i.is_blocker())
    }

    /// Check that the sample count is sufficient for the claimed epsilon.
    fn check_sample_count(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        let n = cert.samples.len();

        if n == 0 && cert.verified_regions.is_empty() {
            issues.push(ValidationIssue::new(
                IssueSeverity::Error,
                IssueCategory::SampleCount,
                "Certificate has no samples and no verified regions",
            ));
            return;
        }

        if n < self.min_samples && cert.verified_regions.is_empty() {
            issues.push(
                ValidationIssue::new(
                    IssueSeverity::Warning,
                    IssueCategory::SampleCount,
                    format!(
                        "Only {} samples; minimum recommended is {}",
                        n, self.min_samples
                    ),
                )
                .with_values(
                    format!(">= {}", self.min_samples),
                    format!("{}", n),
                ),
            );
        }

        // Check if sample count is sufficient for the claimed epsilon
        if n > 0 && cert.epsilon_estimated < 1.0 {
            let num_elements = cert.element_coverage.len().max(1);
            let num_strata = cert
                .metadata
                .get("num_strata")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1);

            let bound = HoeffdingBound::new(
                cert.delta.max(1e-10).min(1.0 - 1e-10),
                num_strata,
                num_elements,
                1, // assume 1 device
            );

            let achievable_eps = bound.epsilon(n);
            if cert.epsilon_estimated < achievable_eps * 0.5 {
                issues.push(
                    ValidationIssue::new(
                        IssueSeverity::Warning,
                        IssueCategory::EpsilonBound,
                        format!(
                            "Claimed ε_estimated ({:.6}) may be overly optimistic for {} samples \
                             (Hoeffding gives {:.6})",
                            cert.epsilon_estimated, n, achievable_eps
                        ),
                    )
                    .with_values(
                        format!(">= {:.6}", achievable_eps),
                        format!("{:.6}", cert.epsilon_estimated),
                    ),
                );
            }
        }
    }

    /// Verify the kappa computation is consistent with samples.
    fn check_kappa(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if cert.kappa < 0.0 || cert.kappa > 1.0 {
            issues.push(
                ValidationIssue::new(
                    IssueSeverity::Error,
                    IssueCategory::KappaComputation,
                    format!("κ = {} is outside [0, 1]", cert.kappa),
                )
                .with_values("[0, 1]", format!("{}", cert.kappa)),
            );
            return;
        }

        // Recompute kappa from samples
        if !cert.samples.is_empty() {
            let n = cert.samples.len();
            let n_pass = cert.samples.iter().filter(|s| s.is_pass()).count();
            let sampling_kappa = n_pass as f64 / n as f64;

            // kappa should be at least the sampling pass rate (verified regions add more)
            if cert.kappa < sampling_kappa * 0.9 - 0.01 {
                issues.push(
                    ValidationIssue::new(
                        IssueSeverity::Warning,
                        IssueCategory::KappaComputation,
                        format!(
                            "κ ({:.4}) is lower than sampling pass rate ({:.4}); \
                             verified regions should only increase κ",
                            cert.kappa, sampling_kappa
                        ),
                    )
                    .with_values(
                        format!(">= {:.4}", sampling_kappa),
                        format!("{:.4}", cert.kappa),
                    ),
                );
            }
        }
    }

    /// Check epsilon bounds are valid.
    fn check_epsilon_bounds(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if cert.epsilon_analytical < 0.0 {
            issues.push(ValidationIssue::new(
                IssueSeverity::Error,
                IssueCategory::EpsilonBound,
                format!("ε_analytical = {} is negative", cert.epsilon_analytical),
            ));
        }

        if cert.epsilon_estimated < 0.0 {
            issues.push(ValidationIssue::new(
                IssueSeverity::Error,
                IssueCategory::EpsilonBound,
                format!("ε_estimated = {} is negative", cert.epsilon_estimated),
            ));
        }

        if cert.epsilon_analytical > 1.0 {
            issues.push(ValidationIssue::new(
                IssueSeverity::Warning,
                IssueCategory::EpsilonBound,
                format!(
                    "ε_analytical = {:.6} exceeds 1.0; linearization may be unreliable",
                    cert.epsilon_analytical
                ),
            ));
        }

        if cert.epsilon_estimated > self.max_useful_epsilon {
            issues.push(ValidationIssue::new(
                IssueSeverity::Warning,
                IssueCategory::EpsilonBound,
                format!(
                    "ε_estimated = {:.6} exceeds useful threshold {:.6}",
                    cert.epsilon_estimated, self.max_useful_epsilon
                ),
            ));
        }
    }

    /// Check delta parameter validity.
    fn check_delta(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if cert.delta <= 0.0 || cert.delta >= 1.0 {
            issues.push(
                ValidationIssue::new(
                    IssueSeverity::Error,
                    IssueCategory::DeltaParameter,
                    format!("δ = {} is outside (0, 1)", cert.delta),
                )
                .with_values("(0, 1)", format!("{}", cert.delta)),
            );
        }

        if cert.delta > self.max_delta {
            issues.push(ValidationIssue::new(
                IssueSeverity::Warning,
                IssueCategory::DeltaParameter,
                format!(
                    "δ = {:.4} is high (confidence only {:.1}%)",
                    cert.delta,
                    (1.0 - cert.delta) * 100.0
                ),
            ));
        }
    }

    /// Check that the grade is consistent with kappa and violations.
    fn check_grade_consistency(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        let expected_grade =
            CertificateGrade::from_metrics(cert.kappa, cert.violations.len());

        if cert.grade != expected_grade {
            issues.push(
                ValidationIssue::new(
                    IssueSeverity::Warning,
                    IssueCategory::GradeConsistency,
                    format!(
                        "Grade {:?} doesn't match expected {:?} for κ={:.4}, {} violations",
                        cert.grade,
                        expected_grade,
                        cert.kappa,
                        cert.violations.len()
                    ),
                )
                .with_values(format!("{:?}", expected_grade), format!("{:?}", cert.grade)),
            );
        }
    }

    /// Check violation surfaces for consistency.
    fn check_violations(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        for (i, violation) in cert.violations.iter().enumerate() {
            if violation.estimated_measure < 0.0 {
                issues.push(ValidationIssue::new(
                    IssueSeverity::Error,
                    IssueCategory::ViolationConsistency,
                    format!(
                        "Violation {} has negative measure: {}",
                        i, violation.estimated_measure
                    ),
                ));
            }

            if let Some((ref lower, ref upper)) = violation.parameter_bounds {
                for d in 0..lower.len().min(upper.len()) {
                    if lower[d] > upper[d] {
                        issues.push(ValidationIssue::new(
                            IssueSeverity::Error,
                            IssueCategory::ViolationConsistency,
                            format!(
                                "Violation {} has inverted bounds in dimension {}: {} > {}",
                                i, d, lower[d], upper[d]
                            ),
                        ));
                    }
                }
            }
        }
    }

    /// Check sample data integrity.
    fn check_sample_integrity(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        for (i, sample) in cert.samples.iter().enumerate() {
            if sample.body_params.len() != NUM_BODY_PARAMS {
                issues.push(ValidationIssue::new(
                    IssueSeverity::Warning,
                    IssueCategory::SampleIntegrity,
                    format!(
                        "Sample {} has {} body params (expected {})",
                        i,
                        sample.body_params.len(),
                        NUM_BODY_PARAMS
                    ),
                ));
            }

            // Check for NaN or infinity
            for (d, &v) in sample.body_params.iter().enumerate() {
                if v.is_nan() || v.is_infinite() {
                    issues.push(ValidationIssue::new(
                        IssueSeverity::Error,
                        IssueCategory::SampleIntegrity,
                        format!(
                            "Sample {} param {} is {} (invalid)",
                            i, d, v
                        ),
                    ));
                }
            }
        }

        // Check element coverage matches samples
        if !cert.samples.is_empty() {
            let mut sample_elements: HashMap<ElementId, usize> = HashMap::new();
            for s in &cert.samples {
                *sample_elements.entry(s.element_id).or_insert(0) += 1;
            }

            for (eid, _count) in &sample_elements {
                if !cert.element_coverage.contains_key(eid) {
                    issues.push(ValidationIssue::new(
                        IssueSeverity::Info,
                        IssueCategory::SampleIntegrity,
                        format!(
                            "Element {} has samples but no entry in element_coverage",
                            eid
                        ),
                    ));
                }
            }
        }
    }

    /// Check verified region volumes.
    fn check_region_volumes(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        for (_i, region) in cert.verified_regions.iter().enumerate() {
            let vol = region.volume();
            if vol <= 0.0 {
                issues.push(ValidationIssue::new(
                    IssueSeverity::Warning,
                    IssueCategory::RegionVolume,
                    format!("Verified region '{}' has zero or negative volume", region.label),
                ));
            }

            // Check bounds consistency
            for d in 0..region.lower.len().min(region.upper.len()) {
                if region.lower[d] > region.upper[d] {
                    issues.push(ValidationIssue::new(
                        IssueSeverity::Error,
                        IssueCategory::RegionVolume,
                        format!(
                            "Region '{}' has inverted bounds in dim {}: {} > {}",
                            region.label, d, region.lower[d], region.upper[d]
                        ),
                    ));
                }
            }

            // Check linearization error is reasonable
            if region.linearization_error > 0.1 {
                issues.push(ValidationIssue::new(
                    IssueSeverity::Warning,
                    IssueCategory::RegionVolume,
                    format!(
                        "Region '{}' has high linearization error: {:.6}",
                        region.label, region.linearization_error
                    ),
                ));
            }
        }
    }

    /// Check for overlapping verified regions.
    fn check_region_overlap(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        let n = cert.verified_regions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &cert.verified_regions[i];
                let b = &cert.verified_regions[j];
                if regions_overlap(a, b) {
                    let overlap_vol = overlap_volume(a, b);
                    if overlap_vol > 1e-10 {
                        issues.push(ValidationIssue::new(
                            IssueSeverity::Warning,
                            IssueCategory::RegionOverlap,
                            format!(
                                "Regions '{}' and '{}' overlap (volume {:.6})",
                                a.label, b.label, overlap_vol
                            ),
                        ));
                    }
                }
            }
        }
    }

    /// Check metadata for required fields.
    fn check_metadata(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if cert.protocol_version.is_empty() {
            issues.push(ValidationIssue::new(
                IssueSeverity::Warning,
                IssueCategory::MetadataIntegrity,
                "Missing protocol version",
            ));
        }

        if cert.timestamp.is_empty() {
            issues.push(ValidationIssue::new(
                IssueSeverity::Warning,
                IssueCategory::MetadataIntegrity,
                "Missing timestamp",
            ));
        }
    }

    /// Check timing data for consistency.
    fn check_timing(
        &self,
        cert: &CoverageCertificate,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if cert.total_time_s < 0.0 {
            issues.push(ValidationIssue::new(
                IssueSeverity::Error,
                IssueCategory::TimingConsistency,
                format!("Negative total time: {}s", cert.total_time_s),
            ));
        }

        // Check per-sample computation times
        let total_sample_time: f64 = cert
            .samples
            .iter()
            .map(|s| s.computation_time_s)
            .sum();
        if total_sample_time > cert.total_time_s * 2.0 && cert.total_time_s > 0.0 {
            issues.push(ValidationIssue::new(
                IssueSeverity::Info,
                IssueCategory::TimingConsistency,
                format!(
                    "Sum of sample times ({:.2}s) exceeds total time ({:.2}s); \
                     may indicate parallelism",
                    total_sample_time, cert.total_time_s
                ),
            ));
        }
    }
}

impl Default for CertificateValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if two verified regions overlap.
fn regions_overlap(a: &VerifiedRegion, b: &VerifiedRegion) -> bool {
    let dims = a.lower.len().min(b.lower.len());
    for d in 0..dims {
        if a.upper[d] <= b.lower[d] || b.upper[d] <= a.lower[d] {
            return false;
        }
    }
    true
}

/// Compute the overlap volume of two regions.
fn overlap_volume(a: &VerifiedRegion, b: &VerifiedRegion) -> f64 {
    let dims = a.lower.len().min(b.lower.len());
    let mut vol = 1.0;
    for d in 0..dims {
        let lo = a.lower[d].max(b.lower[d]);
        let hi = a.upper[d].min(b.upper[d]);
        if hi <= lo {
            return 0.0;
        }
        vol *= hi - lo;
    }
    vol
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use xr_types::certificate::{CertificateGrade, SampleVerdict, VerifiedRegion, ViolationSeverity, ViolationSurface};

    fn test_element() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_valid_cert() -> CoverageCertificate {
        let eid = test_element();
        let samples: Vec<SampleVerdict> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                SampleVerdict::pass(vec![1.5 + t * 0.4, 0.25 + t * 0.15, 0.35 + t * 0.15, 0.22 + t * 0.11, 0.16 + t * 0.06], eid)
            })
            .collect();

        let mut element_coverage = HashMap::new();
        element_coverage.insert(eid, 1.0);

        CoverageCertificate {
            id: Uuid::new_v4(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            protocol_version: "0.1.0".to_string(),
            scene_id: Uuid::new_v4(),
            samples,
            verified_regions: vec![],
            violations: vec![],
            epsilon_analytical: 0.001,
            epsilon_estimated: 0.1,
            delta: 0.05,
            kappa: 0.95,
            grade: CertificateGrade::Partial,
            total_time_s: 2.5,
            element_coverage,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_valid_certificate() {
        let cert = make_valid_cert();
        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        let blockers: Vec<_> = issues.iter().filter(|i| i.is_blocker()).collect();
        assert!(blockers.is_empty(), "Valid cert should have no blockers: {:?}", blockers);
    }

    #[test]
    fn test_soundness_check() {
        let cert = make_valid_cert();
        let validator = CertificateValidator::new();
        assert!(validator.check_soundness(&cert));
    }

    #[test]
    fn test_empty_certificate() {
        let cert = CoverageCertificate {
            id: Uuid::new_v4(),
            timestamp: "2024-01-01T00:00:00Z".into(),
            protocol_version: "0.1.0".into(),
            scene_id: Uuid::new_v4(),
            samples: vec![],
            verified_regions: vec![],
            violations: vec![],
            epsilon_analytical: 0.0,
            epsilon_estimated: 0.0,
            delta: 0.05,
            kappa: 0.0,
            grade: CertificateGrade::Weak,
            total_time_s: 0.0,
            element_coverage: HashMap::new(),
            metadata: HashMap::new(),
        };

        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::SampleCount)));
    }

    #[test]
    fn test_invalid_delta() {
        let mut cert = make_valid_cert();
        cert.delta = 1.5;
        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::DeltaParameter)));
    }

    #[test]
    fn test_invalid_kappa() {
        let mut cert = make_valid_cert();
        cert.kappa = -0.1;
        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::KappaComputation)));
    }

    #[test]
    fn test_negative_epsilon() {
        let mut cert = make_valid_cert();
        cert.epsilon_estimated = -0.1;
        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| {
            matches!(i.category, IssueCategory::EpsilonBound)
                && i.description.contains("negative")
        }));
    }

    #[test]
    fn test_overlapping_regions() {
        let mut cert = make_valid_cert();
        let eid = test_element();
        cert.verified_regions = vec![
            VerifiedRegion::new("r1", vec![0.0, 0.0], vec![1.0, 1.0], eid),
            VerifiedRegion::new("r2", vec![0.5, 0.5], vec![1.5, 1.5], eid),
        ];

        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::RegionOverlap)));
    }

    #[test]
    fn test_no_overlap_without_check() {
        let mut cert = make_valid_cert();
        let eid = test_element();
        cert.verified_regions = vec![
            VerifiedRegion::new("r1", vec![0.0, 0.0], vec![1.0, 1.0], eid),
            VerifiedRegion::new("r2", vec![0.5, 0.5], vec![1.5, 1.5], eid),
        ];

        let validator = CertificateValidator::new().with_overlap_check(false);
        let issues = validator.validate(&cert);
        assert!(!issues.iter().any(|i| matches!(i.category, IssueCategory::RegionOverlap)));
    }

    #[test]
    fn test_inverted_region_bounds() {
        let mut cert = make_valid_cert();
        let eid = test_element();
        let r = VerifiedRegion::new("bad", vec![1.0, 1.0], vec![0.0, 0.0], eid);
        cert.verified_regions = vec![r];

        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::RegionVolume)));
    }

    #[test]
    fn test_grade_consistency() {
        let mut cert = make_valid_cert();
        cert.kappa = 0.5;
        cert.grade = CertificateGrade::Full; // inconsistent

        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::GradeConsistency)));
    }

    #[test]
    fn test_nan_body_params() {
        let mut cert = make_valid_cert();
        cert.samples.push(SampleVerdict::pass(
            vec![f64::NAN, 0.3, 0.4, 0.25, 0.18],
            test_element(),
        ));

        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| {
            matches!(i.category, IssueCategory::SampleIntegrity)
                && i.severity == IssueSeverity::Error
        }));
    }

    #[test]
    fn test_negative_time() {
        let mut cert = make_valid_cert();
        cert.total_time_s = -1.0;

        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::TimingConsistency)));
    }

    #[test]
    fn test_issue_display() {
        let issue = ValidationIssue::new(
            IssueSeverity::Warning,
            IssueCategory::KappaComputation,
            "Test issue",
        )
        .with_values("0.95", "0.80");

        let display = format!("{}", issue);
        assert!(display.contains("Warning"));
        assert!(display.contains("KappaComputation"));
        assert!(display.contains("Test issue"));
        assert!(display.contains("0.95"));
    }

    #[test]
    fn test_regions_overlap_fn() {
        let eid = test_element();
        let a = VerifiedRegion::new("a", vec![0.0, 0.0], vec![1.0, 1.0], eid);
        let b = VerifiedRegion::new("b", vec![0.5, 0.5], vec![1.5, 1.5], eid);
        let c = VerifiedRegion::new("c", vec![2.0, 2.0], vec![3.0, 3.0], eid);

        assert!(regions_overlap(&a, &b));
        assert!(!regions_overlap(&a, &c));
    }

    #[test]
    fn test_overlap_volume_fn() {
        let eid = test_element();
        let a = VerifiedRegion::new("a", vec![0.0, 0.0], vec![1.0, 1.0], eid);
        let b = VerifiedRegion::new("b", vec![0.5, 0.5], vec![1.5, 1.5], eid);
        let vol = overlap_volume(&a, &b);
        assert!((vol - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_violation_negative_measure() {
        let mut cert = make_valid_cert();
        let mut v = ViolationSurface::new("bad", test_element(), ViolationSeverity::Low);
        v.estimated_measure = -1.0;
        cert.violations.push(v);

        let validator = CertificateValidator::new();
        let issues = validator.validate(&cert);
        assert!(issues.iter().any(|i| matches!(i.category, IssueCategory::ViolationConsistency)));
    }
}
