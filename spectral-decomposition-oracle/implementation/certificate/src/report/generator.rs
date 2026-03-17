//! Certificate report generation.
//!
//! Combines all certificates for an instance into a single comprehensive report
//! with multiple output formats: JSON, human-readable text, and LaTeX.

use crate::futility::certificate::FutilityCertificate;
use crate::l3_bound::partition_bound::L3PartitionCertificate;
use crate::l3_bound::L3BoundSummary;
use crate::report::{classify_tier, DecomposabilityTier};
use crate::spectral_bound::davis_kahan::DavisKahanCertificate;
use crate::spectral_bound::partition_quality::PartitionQualityCertificate;
use crate::spectral_bound::scaling_law::SpectralScalingCertificate;
use crate::spectral_bound::SpectralBoundSummary;
use crate::verification::VerificationResult;
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Verification summary for the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    pub all_verified: bool,
    pub num_checks: usize,
    pub num_passed: usize,
    pub num_failed: usize,
    pub num_warnings: usize,
    pub details: Vec<String>,
}

impl From<&VerificationResult> for VerificationSummary {
    fn from(vr: &VerificationResult) -> Self {
        Self {
            all_verified: vr.all_passed,
            num_checks: vr.num_checks,
            num_passed: vr.num_passed,
            num_failed: vr.num_failed,
            num_warnings: vr.num_warnings,
            details: vr
                .details
                .iter()
                .filter(|c| !c.passed)
                .map(|c| format!("{}: {}", c.name, c.message))
                .collect(),
        }
    }
}

/// Complete certificate report for one instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateReport {
    pub id: String,
    pub created_at: String,
    pub instance_name: String,
    pub structure_type: String,
    pub num_variables: Option<usize>,
    pub num_constraints: Option<usize>,
    pub num_blocks: Option<usize>,
    pub best_method: Option<String>,

    // L3 bound certificates
    pub l3_partition_cert: Option<L3PartitionCertificate>,
    pub l3_summary: Option<L3BoundSummary>,

    // Spectral bound certificates
    pub t2_scaling_cert: Option<SpectralScalingCertificate>,
    pub davis_kahan_cert: Option<DavisKahanCertificate>,
    pub partition_quality_cert: Option<PartitionQualityCertificate>,
    pub spectral_summary: Option<SpectralBoundSummary>,

    // Futility prediction
    pub futility_cert: Option<FutilityCertificate>,

    // Verification
    pub verification: Option<VerificationSummary>,

    // Overall assessment
    pub tier: DecomposabilityTier,
    pub tightest_bound: Option<f64>,
    pub actual_gap: Option<f64>,

    pub metadata: IndexMap<String, String>,
}

impl CertificateReport {
    /// Generate a full report with all available certificates.
    pub fn generate_full_report(
        instance_name: impl Into<String>,
        structure_type: impl Into<String>,
        l3_cert: Option<L3PartitionCertificate>,
        t2_cert: Option<SpectralScalingCertificate>,
        dk_cert: Option<DavisKahanCertificate>,
        pq_cert: Option<PartitionQualityCertificate>,
        futility_cert: Option<FutilityCertificate>,
        verification: Option<&VerificationResult>,
    ) -> Self {
        let l3_summary =
            l3_cert.as_ref().map(|l3| L3BoundSummary::from_certificates(Some(l3), None, None));

        let spectral_summary = SpectralBoundSummary::from_certificates(
            t2_cert.as_ref(),
            dk_cert.as_ref(),
            pq_cert.as_ref(),
        );

        let l3_bound = l3_cert.as_ref().map(|c| c.total_bound);
        let t2_vacuous = t2_cert.as_ref().map_or(true, |c| c.is_vacuous);
        let futility_score = futility_cert.as_ref().map(|c| c.futility_score);
        let actual_gap = l3_cert.as_ref().and_then(|c| c.actual_gap);

        let tier = classify_tier(l3_bound, t2_vacuous, futility_score, actual_gap);

        let tightest_bound = {
            let mut bounds = Vec::new();
            if let Some(b) = l3_bound {
                bounds.push(b);
            }
            if let Some(ref t2) = t2_cert {
                if !t2.is_vacuous {
                    bounds.push(t2.bound_value);
                }
            }
            bounds.into_iter().fold(None, |acc, b| {
                Some(acc.map_or(b, |a: f64| a.min(b)))
            })
        };

        Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            instance_name: instance_name.into(),
            structure_type: structure_type.into(),
            num_variables: l3_cert.as_ref().map(|c| c.partition.num_variables),
            num_constraints: None,
            num_blocks: l3_cert.as_ref().map(|c| c.partition.num_blocks),
            best_method: None,
            l3_partition_cert: l3_cert,
            l3_summary,
            t2_scaling_cert: t2_cert,
            davis_kahan_cert: dk_cert,
            partition_quality_cert: pq_cert,
            spectral_summary: Some(spectral_summary),
            futility_cert,
            verification: verification.map(VerificationSummary::from),
            tier,
            tightest_bound,
            actual_gap,
            metadata: IndexMap::new(),
        }
    }

    /// Generate a summary-only report with key metrics.
    pub fn generate_summary(&self) -> IndexMap<String, String> {
        let mut summary = IndexMap::new();
        summary.insert("instance".to_string(), self.instance_name.clone());
        summary.insert("structure".to_string(), self.structure_type.clone());
        summary.insert("tier".to_string(), self.tier.to_string());

        if let Some(bound) = self.tightest_bound {
            summary.insert("tightest_bound".to_string(), format!("{:.6e}", bound));
        }
        if let Some(gap) = self.actual_gap {
            summary.insert("actual_gap".to_string(), format!("{:.6e}", gap));
        }
        if let Some(ref l3) = self.l3_partition_cert {
            summary.insert("l3_bound".to_string(), format!("{:.6e}", l3.total_bound));
            summary.insert(
                "crossing_edges".to_string(),
                l3.crossing_edges.len().to_string(),
            );
        }
        if let Some(ref t2) = self.t2_scaling_cert {
            summary.insert("t2_bound".to_string(), format!("{:.6e}", t2.bound_value));
            summary.insert("t2_vacuous".to_string(), t2.is_vacuous.to_string());
            summary.insert("spectral_ratio".to_string(), format!("{:.4}", t2.spectral_ratio()));
        }
        if let Some(ref fut) = self.futility_cert {
            summary.insert("futility".to_string(), fut.prediction.to_string());
            summary.insert(
                "futility_confidence".to_string(),
                format!("{:.4}", fut.confidence),
            );
        }
        if let Some(ref ver) = self.verification {
            summary.insert("verified".to_string(), ver.all_verified.to_string());
        }

        summary
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Human-readable text report.
    pub fn to_text(&self) -> String {
        let mut lines = Vec::new();
        let w = 60;
        lines.push("═".repeat(w));
        lines.push(format!("  CERTIFICATE REPORT: {}", self.instance_name));
        lines.push("═".repeat(w));
        lines.push(format!("  ID: {}", self.id));
        lines.push(format!("  Created: {}", self.created_at));
        lines.push(format!("  Structure: {}", self.structure_type));
        lines.push(format!("  Tier: {}", self.tier));
        if let Some(n) = self.num_variables {
            lines.push(format!("  Variables: {}", n));
        }
        if let Some(k) = self.num_blocks {
            lines.push(format!("  Blocks: {}", k));
        }

        lines.push(String::new());
        lines.push("─".repeat(w));
        lines.push("  L3 BOUND".to_string());
        lines.push("─".repeat(w));
        if let Some(ref l3) = self.l3_partition_cert {
            lines.push(format!("  Total bound: {:.6e}", l3.total_bound));
            lines.push(format!("  Crossing edges: {}", l3.crossing_edges.len()));
            lines.push(format!("  Status: {}", l3.verification_status));
            if let Some(gap) = l3.actual_gap {
                lines.push(format!("  Actual gap: {:.6e}", gap));
            }
        } else {
            lines.push("  (not computed)".to_string());
        }

        lines.push(String::new());
        lines.push("─".repeat(w));
        lines.push("  T2 SPECTRAL BOUND".to_string());
        lines.push("─".repeat(w));
        if let Some(ref t2) = self.t2_scaling_cert {
            lines.push(format!("  Bound: {:.6e}", t2.bound_value));
            lines.push(format!("  Vacuous: {}", t2.is_vacuous));
            lines.push(format!("  δ²: {:.6e}", t2.delta_squared));
            lines.push(format!("  γ²: {:.6e}", t2.gamma_squared));
            lines.push(format!("  κ: {:.2e}", t2.kappa));
            lines.push(format!("  Spectral ratio: {:.4}", t2.spectral_ratio()));
        } else {
            lines.push("  (not computed)".to_string());
        }

        lines.push(String::new());
        lines.push("─".repeat(w));
        lines.push("  FUTILITY PREDICTION".to_string());
        lines.push("─".repeat(w));
        if let Some(ref fut) = self.futility_cert {
            lines.push(format!("  Prediction: {}", fut.prediction));
            lines.push(format!("  Score: {:.4}", fut.futility_score));
            lines.push(format!("  Confidence: {:.4}", fut.confidence));
            lines.push("  ⚠ EMPIRICAL (not a formal proof)".to_string());
        } else {
            lines.push("  (not computed)".to_string());
        }

        lines.push(String::new());
        lines.push("─".repeat(w));
        lines.push("  VERIFICATION".to_string());
        lines.push("─".repeat(w));
        if let Some(ref ver) = self.verification {
            lines.push(format!(
                "  Result: {}",
                if ver.all_verified { "PASS" } else { "FAIL" }
            ));
            lines.push(format!(
                "  Checks: {}/{} passed, {} warnings",
                ver.num_passed, ver.num_checks, ver.num_warnings
            ));
            for detail in &ver.details {
                lines.push(format!("  ✗ {}", detail));
            }
        } else {
            lines.push("  (not performed)".to_string());
        }

        lines.push("═".repeat(w));
        lines.join("\n")
    }

    /// LaTeX table row for paper inclusion.
    pub fn to_latex(&self) -> String {
        let mut lines = Vec::new();
        lines.push("% Auto-generated certificate report".to_string());
        lines.push("\\begin{table}[h]".to_string());
        lines.push("\\centering".to_string());
        lines.push("\\caption{Certificate Report}".to_string());
        lines.push("\\begin{tabular}{ll}".to_string());
        lines.push("\\toprule".to_string());
        lines.push("\\textbf{Property} & \\textbf{Value} \\\\".to_string());
        lines.push("\\midrule".to_string());
        lines.push(format!(
            "Instance & \\texttt{{{}}} \\\\",
            self.instance_name.replace('_', "\\_")
        ));
        lines.push(format!("Structure & {} \\\\", self.structure_type));
        lines.push(format!("Tier & {} \\\\", self.tier));

        if let Some(ref l3) = self.l3_partition_cert {
            lines.push(format!(
                "L3 Bound & ${:.4e}$ \\\\",
                l3.total_bound
            ));
        }
        if let Some(ref t2) = self.t2_scaling_cert {
            lines.push(format!(
                "T2 Bound & ${:.4e}$ {} \\\\",
                t2.bound_value,
                if t2.is_vacuous { "(vacuous)" } else { "" }
            ));
            lines.push(format!(
                "$\\delta^2/\\gamma^2$ & ${:.4}$ \\\\",
                t2.spectral_ratio()
            ));
        }
        if let Some(ref fut) = self.futility_cert {
            lines.push(format!(
                "Futility & {} ({:.2}\\%) \\\\",
                fut.prediction,
                fut.confidence * 100.0
            ));
        }
        if let Some(ref ver) = self.verification {
            lines.push(format!(
                "Verified & {}/{} \\\\",
                ver.num_passed, ver.num_checks
            ));
        }

        lines.push("\\bottomrule".to_string());
        lines.push("\\end{tabular}".to_string());
        lines.push("\\end{table}".to_string());
        lines.join("\n")
    }

    /// Set the best decomposition method.
    pub fn set_best_method(&mut self, method: impl Into<String>) {
        self.best_method = Some(method.into());
    }

    /// Add metadata.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l3_bound::partition_bound::{Partition, PartitionMethod};

    fn make_test_report() -> CertificateReport {
        let partition = Partition::new(vec![0, 0, 1, 1], 2, PartitionMethod::Spectral).unwrap();
        let l3 = L3PartitionCertificate::trivial(&partition);
        let t2 =
            SpectralScalingCertificate::compute(0.1, 0.5, 2, 2.0, 10.0, vec![]).unwrap();

        CertificateReport::generate_full_report(
            "test_instance",
            "BlockAngular",
            Some(l3),
            Some(t2),
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn test_generate_full_report() {
        let report = make_test_report();
        assert_eq!(report.instance_name, "test_instance");
        assert!(report.l3_partition_cert.is_some());
        assert!(report.t2_scaling_cert.is_some());
    }

    #[test]
    fn test_generate_summary() {
        let report = make_test_report();
        let summary = report.generate_summary();
        assert!(summary.contains_key("instance"));
        assert!(summary.contains_key("tier"));
    }

    #[test]
    fn test_to_json() {
        let report = make_test_report();
        let json = report.to_json().unwrap();
        assert!(json.contains("test_instance"));
        assert!(json.contains("tier"));
    }

    #[test]
    fn test_to_text() {
        let report = make_test_report();
        let text = report.to_text();
        assert!(text.contains("CERTIFICATE REPORT"));
        assert!(text.contains("test_instance"));
        assert!(text.contains("L3 BOUND"));
        assert!(text.contains("T2 SPECTRAL BOUND"));
    }

    #[test]
    fn test_to_latex() {
        let report = make_test_report();
        let latex = report.to_latex();
        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("test\\_instance"));
    }

    #[test]
    fn test_report_no_certificates() {
        let report = CertificateReport::generate_full_report(
            "empty_instance",
            "Unknown",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(report.l3_partition_cert.is_none());
        assert!(report.tightest_bound.is_none());
    }

    #[test]
    fn test_set_best_method() {
        let mut report = make_test_report();
        report.set_best_method("Benders");
        assert_eq!(report.best_method, Some("Benders".to_string()));
    }

    #[test]
    fn test_add_metadata() {
        let mut report = make_test_report();
        report.add_metadata("solver", "CPLEX");
        assert_eq!(report.metadata.get("solver").unwrap(), "CPLEX");
    }

    #[test]
    fn test_report_with_verification() {
        let mut vr = VerificationResult::new();
        use crate::verification::{CheckSeverity, VerificationCheck};
        vr.add_check(VerificationCheck {
            name: "test".to_string(),
            passed: true,
            severity: CheckSeverity::Error,
            message: "ok".to_string(),
            value: None,
            threshold: None,
        });

        let report = CertificateReport::generate_full_report(
            "test",
            "BlockAngular",
            None,
            None,
            None,
            None,
            None,
            Some(&vr),
        );
        assert!(report.verification.is_some());
        assert!(report.verification.unwrap().all_verified);
    }

    #[test]
    fn test_tier_classification() {
        let report = make_test_report();
        assert!(
            report.tier == DecomposabilityTier::Easy
                || report.tier == DecomposabilityTier::Medium
        );
    }
}
