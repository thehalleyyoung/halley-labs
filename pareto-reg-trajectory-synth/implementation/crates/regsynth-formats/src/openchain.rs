//! OpenChain ISO/IEC 5230:2020 conformance support.
//!
//! Maps RegSynth compliance strategies to OpenChain conformance reports
//! covering the six requirement areas defined by the standard:
//!
//! 1. **Program Foundation** — policy and scope
//! 2. **Relevant Competencies** — personnel competence
//! 3. **Awareness** — training and awareness
//! 4. **Scope** — program scope definition
//! 5. **Compliance Artifacts** — artifact delivery
//! 6. **Contributions** — upstream contribution policy
//!
//! Reference: <https://www.openchainproject.org/license>

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use regsynth_types::{ComplianceStrategy, CostVector, Obligation, RegulatoryDomain};

use crate::FormatResult;

// ── Conformance area model ────────────────────────────────────────────

/// The six requirement areas of OpenChain ISO/IEC 5230:2020.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConformanceArea {
    /// §3.1 — Program foundation: policy, competence definition, awareness.
    ProgramFoundation,
    /// §3.2 — Relevant task-specific competencies.
    RelevantCompetencies,
    /// §3.3 — Awareness of open source policy among staff.
    Awareness,
    /// §3.4 — Scope of the compliance program.
    Scope,
    /// §3.5 — Compliance artifact generation and delivery.
    ComplianceArtifacts,
    /// §3.6 — Upstream open source contribution policy.
    Contributions,
}

impl ConformanceArea {
    /// All six areas in specification order.
    pub fn all() -> &'static [ConformanceArea] {
        &[
            Self::ProgramFoundation,
            Self::RelevantCompetencies,
            Self::Awareness,
            Self::Scope,
            Self::ComplianceArtifacts,
            Self::Contributions,
        ]
    }

    /// ISO clause number for this area.
    pub fn clause(&self) -> &'static str {
        match self {
            Self::ProgramFoundation => "3.1",
            Self::RelevantCompetencies => "3.2",
            Self::Awareness => "3.3",
            Self::Scope => "3.4",
            Self::ComplianceArtifacts => "3.5",
            Self::Contributions => "3.6",
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ProgramFoundation => "Program Foundation",
            Self::RelevantCompetencies => "Relevant Competencies",
            Self::Awareness => "Awareness",
            Self::Scope => "Scope",
            Self::ComplianceArtifacts => "Compliance Artifacts",
            Self::Contributions => "Contributions",
        }
    }
}

// ── Area assessment ───────────────────────────────────────────────────

/// Assessment status for a single conformance area.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssessmentStatus {
    /// Fully conformant.
    Conformant,
    /// Partially conformant with known gaps.
    PartiallyConformant,
    /// Not yet assessed or non-conformant.
    NonConformant,
    /// Not applicable to the current scope.
    NotApplicable,
}

/// Detailed assessment of one conformance area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AreaAssessment {
    pub area: ConformanceArea,
    pub status: AssessmentStatus,
    /// Evidence items supporting the assessment.
    pub evidence: Vec<String>,
    /// Identified gaps that need remediation.
    pub gaps: Vec<String>,
    /// Remediation actions planned or completed.
    pub remediation_actions: Vec<String>,
    /// Percentage of sub-requirements satisfied (0.0–1.0).
    pub completion_ratio: f64,
}

impl AreaAssessment {
    /// Create a new assessment for the given area.
    pub fn new(area: ConformanceArea) -> Self {
        Self {
            area,
            status: AssessmentStatus::NonConformant,
            evidence: Vec::new(),
            gaps: Vec::new(),
            remediation_actions: Vec::new(),
            completion_ratio: 0.0,
        }
    }

    /// Mark as conformant with supporting evidence.
    pub fn mark_conformant(mut self, evidence: Vec<String>) -> Self {
        self.status = AssessmentStatus::Conformant;
        self.evidence = evidence;
        self.completion_ratio = 1.0;
        self
    }

    /// Mark as partially conformant, noting gaps.
    pub fn mark_partial(mut self, ratio: f64, evidence: Vec<String>, gaps: Vec<String>) -> Self {
        self.status = AssessmentStatus::PartiallyConformant;
        self.completion_ratio = ratio.clamp(0.0, 1.0);
        self.evidence = evidence;
        self.gaps = gaps;
        self
    }
}

// ── OpenChain conformance record ──────────────────────────────────────

/// Full OpenChain ISO/IEC 5230:2020 conformance record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenChainConformance {
    /// Organization name.
    pub organization: String,
    /// Program name within the organization.
    pub program_name: String,
    /// ISO standard version (always `"ISO/IEC 5230:2020"`).
    pub standard_version: String,
    /// Assessment timestamp.
    pub assessed_at: String,
    /// Per-area assessments.
    pub assessments: HashMap<String, AreaAssessment>,
    /// Overall conformance determination.
    pub overall_status: AssessmentStatus,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl OpenChainConformance {
    /// Create a new conformance record with default non-conformant assessments.
    pub fn new(organization: impl Into<String>, program: impl Into<String>) -> Self {
        let mut assessments = HashMap::new();
        for area in ConformanceArea::all() {
            assessments.insert(area.clause().to_string(), AreaAssessment::new(*area));
        }
        Self {
            organization: organization.into(),
            program_name: program.into(),
            standard_version: "ISO/IEC 5230:2020".into(),
            assessed_at: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            assessments,
            overall_status: AssessmentStatus::NonConformant,
            metadata: HashMap::new(),
        }
    }

    /// Update the assessment for a specific clause.
    pub fn set_assessment(&mut self, clause: &str, assessment: AreaAssessment) {
        self.assessments.insert(clause.to_string(), assessment);
        self.recompute_overall();
    }

    /// Recompute the overall conformance status from individual assessments.
    fn recompute_overall(&mut self) {
        let mut all_conformant = true;
        let mut any_conformant = false;
        for a in self.assessments.values() {
            match a.status {
                AssessmentStatus::Conformant | AssessmentStatus::NotApplicable => {
                    any_conformant = true;
                }
                _ => {
                    all_conformant = false;
                }
            }
        }
        self.overall_status = if all_conformant {
            AssessmentStatus::Conformant
        } else if any_conformant {
            AssessmentStatus::PartiallyConformant
        } else {
            AssessmentStatus::NonConformant
        };
    }

    /// Average completion ratio across all assessed areas.
    pub fn average_completion(&self) -> f64 {
        if self.assessments.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.assessments.values().map(|a| a.completion_ratio).sum();
        sum / self.assessments.len() as f64
    }

    /// Build a conformance record from RegSynth obligations and a strategy.
    ///
    /// The mapping heuristic uses obligation domains and the strategy's
    /// coverage to estimate conformance in each area.
    pub fn from_strategy(
        obligations: &[Obligation],
        strategy: &ComplianceStrategy,
        organization: &str,
        program: &str,
    ) -> Self {
        let mut conf = Self::new(organization, program);

        let coverage = strategy.coverage();
        let has_policy_obs = obligations
            .iter()
            .any(|o| matches!(o.domain, RegulatoryDomain::Documentation));
        let has_transparency = obligations
            .iter()
            .any(|o| matches!(o.domain, RegulatoryDomain::Transparency));

        // §3.1 Program Foundation
        let foundation = if has_policy_obs && coverage > 0.5 {
            AreaAssessment::new(ConformanceArea::ProgramFoundation).mark_conformant(vec![
                "Documentation obligations present".into(),
                format!("Strategy coverage: {:.0}%", coverage * 100.0),
            ])
        } else {
            AreaAssessment::new(ConformanceArea::ProgramFoundation).mark_partial(
                coverage,
                vec![format!("Strategy coverage: {:.0}%", coverage * 100.0)],
                vec!["Missing explicit documentation policy obligations".into()],
            )
        };
        conf.set_assessment("3.1", foundation);

        // §3.2 Relevant Competencies
        let competencies = AreaAssessment::new(ConformanceArea::RelevantCompetencies).mark_partial(
            coverage * 0.8,
            vec!["Derived from strategy assignment breadth".into()],
            vec!["Competency verification not directly modeled".into()],
        );
        conf.set_assessment("3.2", competencies);

        // §3.3 Awareness
        let awareness = if has_transparency {
            AreaAssessment::new(ConformanceArea::Awareness).mark_conformant(vec![
                "Transparency obligations mapped".into(),
            ])
        } else {
            AreaAssessment::new(ConformanceArea::Awareness).mark_partial(
                0.3,
                vec![],
                vec!["No transparency obligations found".into()],
            )
        };
        conf.set_assessment("3.3", awareness);

        // §3.4 Scope
        let scope = AreaAssessment::new(ConformanceArea::Scope).mark_conformant(vec![
            format!("{} obligations in scope", obligations.len()),
        ]);
        conf.set_assessment("3.4", scope);

        // §3.5 Compliance Artifacts
        let artifacts = if coverage > 0.8 {
            AreaAssessment::new(ConformanceArea::ComplianceArtifacts).mark_conformant(vec![
                format!("High strategy coverage: {:.0}%", coverage * 100.0),
            ])
        } else {
            AreaAssessment::new(ConformanceArea::ComplianceArtifacts).mark_partial(
                coverage,
                vec![format!("Strategy coverage: {:.0}%", coverage * 100.0)],
                vec!["Incomplete artifact generation".into()],
            )
        };
        conf.set_assessment("3.5", artifacts);

        // §3.6 Contributions
        let contributions = AreaAssessment::new(ConformanceArea::Contributions).mark_partial(
            0.5,
            vec!["Contribution policy not directly modeled".into()],
            vec!["Upstream contribution rules not encoded".into()],
        );
        conf.set_assessment("3.6", contributions);

        conf
    }
}

// ── OpenChain report ──────────────────────────────────────────────────

/// A complete OpenChain conformance report suitable for external submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenChainReport {
    /// Report title.
    pub title: String,
    /// Report generation timestamp.
    pub generated_at: String,
    /// Conformance data.
    pub conformance: OpenChainConformance,
    /// Executive summary text.
    pub executive_summary: String,
    /// Recommendations for improving conformance.
    pub recommendations: Vec<String>,
    /// Cost estimate for reaching full conformance.
    pub estimated_remediation_cost: Option<CostVector>,
}

impl OpenChainReport {
    /// Generate a report from a conformance record.
    pub fn generate(conformance: OpenChainConformance) -> Self {
        let avg = conformance.average_completion();
        let summary = format!(
            "OpenChain ISO/IEC 5230:2020 conformance assessment for '{}' ({}).\n\
             Overall status: {:?}. Average area completion: {:.0}%.",
            conformance.program_name,
            conformance.organization,
            conformance.overall_status,
            avg * 100.0,
        );

        let mut recommendations = Vec::new();
        for (clause, assessment) in &conformance.assessments {
            if assessment.status != AssessmentStatus::Conformant
                && assessment.status != AssessmentStatus::NotApplicable
            {
                for gap in &assessment.gaps {
                    recommendations.push(format!("[§{}] {}", clause, gap));
                }
            }
        }

        Self {
            title: format!(
                "OpenChain Conformance Report — {}",
                conformance.program_name
            ),
            generated_at: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            conformance,
            executive_summary: summary,
            recommendations,
            estimated_remediation_cost: None,
        }
    }

    /// Serialize the report to JSON.
    pub fn to_json(&self) -> FormatResult<String> {
        serde_json::to_string_pretty(self).map_err(crate::FormatError::Serialization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformance_area_all() {
        assert_eq!(ConformanceArea::all().len(), 6);
    }

    #[test]
    fn new_conformance_is_non_conformant() {
        let conf = OpenChainConformance::new("Acme Corp", "AI Compliance");
        assert_eq!(conf.overall_status, AssessmentStatus::NonConformant);
        assert_eq!(conf.assessments.len(), 6);
    }

    #[test]
    fn set_all_conformant() {
        let mut conf = OpenChainConformance::new("Acme Corp", "AI Compliance");
        for area in ConformanceArea::all() {
            let a = AreaAssessment::new(*area).mark_conformant(vec!["test".into()]);
            conf.set_assessment(area.clause(), a);
        }
        assert_eq!(conf.overall_status, AssessmentStatus::Conformant);
        assert!((conf.average_completion() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn partial_conformance() {
        let mut conf = OpenChainConformance::new("Acme Corp", "AI Compliance");
        let a = AreaAssessment::new(ConformanceArea::ProgramFoundation)
            .mark_conformant(vec!["policy exists".into()]);
        conf.set_assessment("3.1", a);
        assert_eq!(conf.overall_status, AssessmentStatus::PartiallyConformant);
    }

    #[test]
    fn report_generation() {
        let conf = OpenChainConformance::new("TestOrg", "TestProgram");
        let report = OpenChainReport::generate(conf);
        assert!(report.executive_summary.contains("TestProgram"));
        let json = report.to_json().unwrap();
        assert!(json.contains("ISO/IEC 5230:2020"));
    }
}
