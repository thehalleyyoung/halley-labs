//! Safety certificate generation, serialization, and formatting.
//!
//! A [`SafetyCertificate`] summarises the outcome of a complete verification
//! run, including the verdict, evidence, methodology, and assumptions.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, DrugId, PatientId, PatientProfile,
    SafetyCertificate, SafetyVerdict, VerificationResult, VerificationTier,
};

// ---------------------------------------------------------------------------
// CertificateEvidence
// ---------------------------------------------------------------------------

/// Evidence supporting (or undermining) a safety certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateEvidence {
    pub tier: VerificationTier,
    pub drug_pair: (DrugId, DrugId),
    pub verdict: SafetyVerdict,
    pub summary: String,
    pub confidence: f64,
    pub trace_available: bool,
}

impl CertificateEvidence {
    pub fn from_result(result: &VerificationResult) -> Self {
        Self {
            tier: result.tier,
            drug_pair: result.drug_pair.clone(),
            verdict: result.verdict,
            summary: if result.has_conflicts() {
                format!(
                    "{} conflict(s) detected; max severity = {}",
                    result.conflict_count(),
                    result
                        .max_severity()
                        .map_or("None".to_string(), |s| s.label().to_string())
                )
            } else {
                "No conflicts detected within analysis bounds.".to_string()
            },
            confidence: if result.conflicts.is_empty() {
                0.90
            } else {
                result
                    .conflicts
                    .iter()
                    .map(|c| c.confidence)
                    .sum::<f64>()
                    / result.conflicts.len() as f64
            },
            trace_available: result.trace.is_some(),
        }
    }

    /// A short one-line summary.
    pub fn one_liner(&self) -> String {
        format!(
            "{} vs {} [{}]: {} (conf={:.0}%)",
            self.drug_pair.0,
            self.drug_pair.1,
            self.tier,
            self.verdict,
            self.confidence * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// AssumptionList
// ---------------------------------------------------------------------------

/// The list of assumptions under which a certificate is valid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionList {
    pub items: Vec<Assumption>,
}

/// A single assumption with a category and description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assumption {
    pub category: AssumptionCategory,
    pub description: String,
    pub impact_if_violated: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssumptionCategory {
    PharmacokineticModel,
    PatientPhysiology,
    DoseCompliance,
    DataCompleteness,
    AnalysisBoundary,
}

impl fmt::Display for AssumptionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::PharmacokineticModel => "PK Model",
            Self::PatientPhysiology => "Patient Physiology",
            Self::DoseCompliance => "Dose Compliance",
            Self::DataCompleteness => "Data Completeness",
            Self::AnalysisBoundary => "Analysis Boundary",
        };
        write!(f, "{}", s)
    }
}

impl AssumptionList {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add(&mut self, assumption: Assumption) {
        self.items.push(assumption);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Standard assumptions for a typical two-tier analysis.
    pub fn standard_assumptions(horizon_hours: f64) -> Self {
        let mut list = Self::new();
        list.add(Assumption {
            category: AssumptionCategory::PharmacokineticModel,
            description: "One-compartment pharmacokinetic model with first-order elimination"
                .to_string(),
            impact_if_violated: "Multi-compartment drugs may have inaccurate peak concentrations"
                .to_string(),
        });
        list.add(Assumption {
            category: AssumptionCategory::PharmacokineticModel,
            description: "Steady-state conditions assumed for all medications".to_string(),
            impact_if_violated:
                "Recently started drugs may not yet be at steady state; concentrations may be lower"
                    .to_string(),
        });
        list.add(Assumption {
            category: AssumptionCategory::PatientPhysiology,
            description: "Inter-patient variability modelled as ±30% on steady-state concentration"
                .to_string(),
            impact_if_violated:
                "Patients with extreme metabolizer phenotypes may fall outside predicted intervals"
                    .to_string(),
        });
        list.add(Assumption {
            category: AssumptionCategory::DoseCompliance,
            description: "Patient takes all medications exactly as prescribed".to_string(),
            impact_if_violated:
                "Non-adherence may reduce or increase risk depending on the missed doses"
                    .to_string(),
        });
        list.add(Assumption {
            category: AssumptionCategory::DataCompleteness,
            description: "All current medications are included in the analysis".to_string(),
            impact_if_violated: "OTC drugs, supplements, or unlisted prescriptions may introduce additional interactions".to_string(),
        });
        list.add(Assumption {
            category: AssumptionCategory::AnalysisBoundary,
            description: format!(
                "Analysis covers a {}‑hour simulation horizon",
                horizon_hours
            ),
            impact_if_violated:
                "Interactions with very long onset times may not be captured".to_string(),
        });
        list
    }

    /// Format assumptions as a bulleted list.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for (i, a) in self.items.iter().enumerate() {
            out.push_str(&format!(
                "  {}. [{}] {}\n     → If violated: {}\n",
                i + 1,
                a.category,
                a.description,
                a.impact_if_violated
            ));
        }
        out
    }
}

impl Default for AssumptionList {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CertificateSerializer
// ---------------------------------------------------------------------------

/// Serializes certificates to various formats.
pub struct CertificateSerializer;

impl CertificateSerializer {
    /// Serialize a certificate to a pretty-printed JSON string.
    pub fn to_json(cert: &SafetyCertificate) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(cert)
    }

    /// Serialize a certificate to a compact JSON string.
    pub fn to_json_compact(cert: &SafetyCertificate) -> Result<String, serde_json::Error> {
        serde_json::to_string(cert)
    }

    /// Deserialize a certificate from a JSON string.
    pub fn from_json(json: &str) -> Result<SafetyCertificate, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize an evidence list to JSON.
    pub fn evidence_to_json(evidence: &[CertificateEvidence]) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(evidence)
    }
}

// ---------------------------------------------------------------------------
// CertificateGenerator
// ---------------------------------------------------------------------------

/// Generates safety certificates from verification results.
pub struct CertificateGenerator {
    horizon_hours: f64,
    min_confidence_for_safe: f64,
}

impl CertificateGenerator {
    pub fn new(horizon_hours: f64) -> Self {
        Self {
            horizon_hours,
            min_confidence_for_safe: 0.80,
        }
    }

    pub fn with_min_confidence(mut self, min_conf: f64) -> Self {
        self.min_confidence_for_safe = min_conf;
        self
    }

    /// Generate a certificate from verification results and patient profile.
    pub fn generate(
        &self,
        patient: &PatientProfile,
        results: &[VerificationResult],
    ) -> GeneratedCertificate {
        let evidence: Vec<CertificateEvidence> =
            results.iter().map(CertificateEvidence::from_result).collect();

        let all_conflicts: Vec<ConfirmedConflict> =
            results.iter().flat_map(|r| r.conflicts.clone()).collect();

        let assumptions = AssumptionList::standard_assumptions(self.horizon_hours);

        let verdict = self.determine_verdict(&all_conflicts, &evidence);
        let confidence = self.compute_overall_confidence(&evidence, &all_conflicts);

        let cert = SafetyCertificate {
            id: format!("cert-{}-{}", patient.id, self.horizon_hours as u64),
            patient_id: patient.id.clone(),
            medications: patient.drug_ids(),
            verdict,
            conflicts: all_conflicts,
            methodology: format!(
                "Two-tier verification pipeline: Tier 1 abstract interpretation + \
                 Tier 2 bounded model checking over a {:.0}-hour horizon.",
                self.horizon_hours
            ),
            assumptions: assumptions
                .items
                .iter()
                .map(|a| a.description.clone())
                .collect(),
            evidence_summary: evidence.iter().map(|e| e.one_liner()).collect(),
            generated_at: "2025-01-01T00:00:00Z".to_string(),
            valid_until: Some("2025-04-01T00:00:00Z".to_string()),
            confidence_score: confidence,
        };

        GeneratedCertificate {
            certificate: cert,
            evidence,
            assumptions,
        }
    }

    /// Format a certificate as human-readable text.
    pub fn format_certificate_text(&self, gen: &GeneratedCertificate) -> String {
        let cert = &gen.certificate;
        let mut out = String::new();

        out.push_str("═══════════════════════════════════════════════════════\n");
        out.push_str("           POLYPHARMACY SAFETY CERTIFICATE\n");
        out.push_str("═══════════════════════════════════════════════════════\n\n");

        out.push_str(&format!("Certificate ID : {}\n", cert.id));
        out.push_str(&format!("Patient        : {}\n", cert.patient_id));
        out.push_str(&format!("Generated      : {}\n", cert.generated_at));
        if let Some(ref until) = cert.valid_until {
            out.push_str(&format!("Valid Until     : {}\n", until));
        }
        out.push('\n');

        // Verdict
        let verdict_indicator = match cert.verdict {
            SafetyVerdict::Safe => "✓ SAFE",
            SafetyVerdict::PossiblySafe => "⚠ POSSIBLY SAFE",
            SafetyVerdict::PossiblyUnsafe => "⚠ POSSIBLY UNSAFE",
            SafetyVerdict::Unsafe => "✗ UNSAFE",
        };
        out.push_str(&format!("VERDICT: {}\n", verdict_indicator));
        out.push_str(&format!(
            "Confidence: {:.1}%\n\n",
            cert.confidence_score * 100.0
        ));

        // Medications
        out.push_str("MEDICATIONS:\n");
        for (i, med) in cert.medications.iter().enumerate() {
            out.push_str(&format!("  {}. {}\n", i + 1, med));
        }
        out.push('\n');

        // Conflicts
        if cert.conflicts.is_empty() {
            out.push_str("CONFLICTS: None detected.\n\n");
        } else {
            out.push_str(&format!("CONFLICTS ({}):\n", cert.conflicts.len()));
            for (i, c) in cert.conflicts.iter().enumerate() {
                let drugs_str: Vec<String> = c.drugs.iter().map(|d| d.to_string()).collect();
                out.push_str(&format!(
                    "  {}. [{}] {} — {}\n     Drugs: {}\n     Recommendation: {}\n",
                    i + 1,
                    c.severity.label(),
                    c.interaction_type,
                    c.mechanism_description,
                    drugs_str.join(", "),
                    c.clinical_recommendation,
                ));
            }
            out.push('\n');
        }

        // Evidence
        out.push_str("EVIDENCE:\n");
        for (i, e) in gen.evidence.iter().enumerate() {
            out.push_str(&format!("  {}. {}\n", i + 1, e.one_liner()));
        }
        out.push('\n');

        // Methodology
        out.push_str("METHODOLOGY:\n");
        out.push_str(&format!("  {}\n\n", cert.methodology));

        // Assumptions
        out.push_str("ASSUMPTIONS:\n");
        out.push_str(&gen.assumptions.to_text());
        out.push('\n');

        out.push_str("═══════════════════════════════════════════════════════\n");

        out
    }

    fn determine_verdict(
        &self,
        conflicts: &[ConfirmedConflict],
        evidence: &[CertificateEvidence],
    ) -> SafetyVerdict {
        if conflicts.iter().any(|c| c.severity == ConflictSeverity::Critical) {
            return SafetyVerdict::Unsafe;
        }
        if conflicts.iter().any(|c| c.severity == ConflictSeverity::Major) {
            return SafetyVerdict::PossiblyUnsafe;
        }
        if !conflicts.is_empty() {
            return SafetyVerdict::PossiblySafe;
        }
        // No conflicts — check if we have enough evidence for a Safe verdict
        let avg_conf = if evidence.is_empty() {
            0.0
        } else {
            evidence.iter().map(|e| e.confidence).sum::<f64>() / evidence.len() as f64
        };
        if avg_conf >= self.min_confidence_for_safe {
            SafetyVerdict::Safe
        } else {
            SafetyVerdict::PossiblySafe
        }
    }

    fn compute_overall_confidence(
        &self,
        evidence: &[CertificateEvidence],
        conflicts: &[ConfirmedConflict],
    ) -> f64 {
        if evidence.is_empty() && conflicts.is_empty() {
            return 0.5; // no data
        }

        let evidence_conf = if evidence.is_empty() {
            0.5
        } else {
            evidence.iter().map(|e| e.confidence).sum::<f64>() / evidence.len() as f64
        };

        let conflict_conf = if conflicts.is_empty() {
            1.0 // no conflicts → high confidence in safety
        } else {
            conflicts.iter().map(|c| c.confidence).sum::<f64>() / conflicts.len() as f64
        };

        // Weighted average: evidence contributes more when no conflicts
        if conflicts.is_empty() {
            evidence_conf * 0.7 + conflict_conf * 0.3
        } else {
            evidence_conf * 0.4 + conflict_conf * 0.6
        }
    }
}

/// Complete output of the certificate generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCertificate {
    pub certificate: SafetyCertificate,
    pub evidence: Vec<CertificateEvidence>,
    pub assumptions: AssumptionList,
}

impl Default for CertificateGenerator {
    fn default() -> Self {
        Self::new(72.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AdministrationRoute, CounterExample, Dosage, DrugId, DrugInfo,
        GuidelineId, InteractionType, MedicationRecord, OrganFunction,
        TraceStep, VerificationTier,
    };

    fn make_patient(n_meds: usize) -> PatientProfile {
        let mut p = PatientProfile::simple("pat-001", 65, 70.0);
        for i in 0..n_meds {
            let drug = DrugInfo::simple(&format!("drug{}", i), &format!("Drug {}", i));
            let med = MedicationRecord::new(
                drug,
                Dosage::new(100.0, 12.0, AdministrationRoute::Oral),
            );
            p.medications.push(med);
        }
        p
    }

    fn make_result(
        a: &str,
        b: &str,
        conflicts: Vec<ConfirmedConflict>,
    ) -> VerificationResult {
        let verdict = if conflicts.is_empty() {
            SafetyVerdict::Safe
        } else {
            SafetyVerdict::PossiblyUnsafe
        };
        VerificationResult {
            drug_pair: (DrugId::new(a), DrugId::new(b)),
            tier: VerificationTier::Tier2ModelCheck,
            verdict,
            conflicts,
            trace: None,
            duration_ms: 42,
            notes: vec![],
        }
    }

    fn make_conflict_obj(sev: ConflictSeverity) -> ConfirmedConflict {
        ConfirmedConflict {
            id: "c-test".to_string(),
            drugs: vec![DrugId::new("a"), DrugId::new("b")],
            interaction_type: InteractionType::CypInhibition {
                enzyme: "3A4".to_string(),
            },
            severity: sev,
            verdict: SafetyVerdict::PossiblyUnsafe,
            mechanism_description: "test".to_string(),
            evidence_tier: VerificationTier::Tier2ModelCheck,
            counter_example: None,
            confidence: 0.88,
            clinical_recommendation: "Monitor closely".to_string(),
            affected_parameters: vec!["AUC".to_string()],
            guideline_references: vec![],
        }
    }

    #[test]
    fn test_generate_safe_certificate() {
        let gen = CertificateGenerator::new(72.0);
        let patient = make_patient(2);
        let results = vec![make_result("drug0", "drug1", vec![])];

        let gc = gen.generate(&patient, &results);
        assert!(gc.certificate.is_safe());
        assert!(gc.certificate.conflicts.is_empty());
        assert_eq!(gc.certificate.medications.len(), 2);
    }

    #[test]
    fn test_generate_unsafe_certificate() {
        let gen = CertificateGenerator::new(72.0);
        let patient = make_patient(2);
        let conflict = make_conflict_obj(ConflictSeverity::Critical);
        let results = vec![make_result("drug0", "drug1", vec![conflict])];

        let gc = gen.generate(&patient, &results);
        assert!(!gc.certificate.is_safe());
        assert_eq!(gc.certificate.verdict, SafetyVerdict::Unsafe);
        assert_eq!(gc.certificate.conflict_count(), 1);
    }

    #[test]
    fn test_certificate_text_formatting() {
        let gen = CertificateGenerator::new(72.0);
        let patient = make_patient(2);
        let results = vec![make_result("drug0", "drug1", vec![])];
        let gc = gen.generate(&patient, &results);
        let text = gen.format_certificate_text(&gc);

        assert!(text.contains("POLYPHARMACY SAFETY CERTIFICATE"));
        assert!(text.contains("pat-001"));
        assert!(text.contains("METHODOLOGY"));
        assert!(text.contains("ASSUMPTIONS"));
    }

    #[test]
    fn test_certificate_evidence_one_liner() {
        let result = make_result("aspirin", "warfarin", vec![]);
        let ev = CertificateEvidence::from_result(&result);
        let line = ev.one_liner();
        assert!(line.contains("aspirin"));
        assert!(line.contains("warfarin"));
        assert!(line.contains("Safe"));
    }

    #[test]
    fn test_assumption_list_standard() {
        let assumptions = AssumptionList::standard_assumptions(72.0);
        assert!(assumptions.len() >= 5);
        let text = assumptions.to_text();
        assert!(text.contains("One-compartment"));
        assert!(text.contains("72"));
    }

    #[test]
    fn test_serializer_roundtrip() {
        let cert = SafetyCertificate {
            id: "test-cert".to_string(),
            patient_id: PatientId::new("p1"),
            medications: vec![DrugId::new("a"), DrugId::new("b")],
            verdict: SafetyVerdict::Safe,
            conflicts: vec![],
            methodology: "Test".to_string(),
            assumptions: vec!["Assumption 1".to_string()],
            evidence_summary: vec!["Evidence 1".to_string()],
            generated_at: "2025-01-01T00:00:00Z".to_string(),
            valid_until: None,
            confidence_score: 0.95,
        };

        let json = CertificateSerializer::to_json(&cert).unwrap();
        let restored = CertificateSerializer::from_json(&json).unwrap();
        assert_eq!(restored.id, cert.id);
        assert_eq!(restored.verdict, cert.verdict);
        assert_eq!(restored.confidence_score, cert.confidence_score);
    }

    #[test]
    fn test_serializer_compact() {
        let cert = SafetyCertificate {
            id: "c".to_string(),
            patient_id: PatientId::new("p"),
            medications: vec![],
            verdict: SafetyVerdict::Safe,
            conflicts: vec![],
            methodology: "m".to_string(),
            assumptions: vec![],
            evidence_summary: vec![],
            generated_at: "now".to_string(),
            valid_until: None,
            confidence_score: 1.0,
        };
        let compact = CertificateSerializer::to_json_compact(&cert).unwrap();
        assert!(!compact.contains('\n'));
    }

    #[test]
    fn test_confidence_no_data() {
        let gen = CertificateGenerator::new(72.0);
        let patient = make_patient(0);
        let gc = gen.generate(&patient, &[]);
        // With no evidence, confidence should be low-ish
        assert!(gc.certificate.confidence_score < 0.9);
    }

    #[test]
    fn test_format_certificate_with_conflicts() {
        let gen = CertificateGenerator::new(72.0);
        let patient = make_patient(2);
        let conflict = make_conflict_obj(ConflictSeverity::Major);
        let results = vec![make_result("drug0", "drug1", vec![conflict])];
        let gc = gen.generate(&patient, &results);
        let text = gen.format_certificate_text(&gc);
        assert!(text.contains("CONFLICTS (1)"));
        assert!(text.contains("Major"));
    }

    #[test]
    fn test_evidence_trace_available() {
        let mut result = make_result("a", "b", vec![]);
        result.trace = Some(vec![TraceStep::new(0.0, "initial")]);
        let ev = CertificateEvidence::from_result(&result);
        assert!(ev.trace_available);
    }

    #[test]
    fn test_assumption_category_display() {
        assert_eq!(
            AssumptionCategory::PharmacokineticModel.to_string(),
            "PK Model"
        );
        assert_eq!(
            AssumptionCategory::DoseCompliance.to_string(),
            "Dose Compliance"
        );
    }
}
