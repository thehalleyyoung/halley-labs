//! Conflict classification, severity computation, and prioritization.
//!
//! Provides the [`ConflictAnalyzer`] that takes raw interaction data and
//! produces classified, scored, and prioritized [`ConflictReport`]s.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, CounterExample, DrugId, DrugInfo,
    InteractionType, MedicationRecord, PatientProfile, SafetyVerdict,
    VerificationResult, VerificationTier, GuidelineId,
};

// ---------------------------------------------------------------------------
// ConflictMechanism — detailed mechanism taxonomy
// ---------------------------------------------------------------------------

/// Fine-grained classification of a conflict's pharmacological mechanism.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictMechanism {
    /// CYP enzyme inhibition increases substrate exposure.
    CypEnzymeInhibition { enzyme: String, inhibitor: DrugId, substrate: DrugId },
    /// CYP enzyme induction decreases substrate exposure.
    CypEnzymeInduction { enzyme: String, inducer: DrugId, substrate: DrugId },
    /// Competitive protein-binding displacement.
    ProteinBindingCompetition { displacer: DrugId, displaced: DrugId },
    /// Renal tubular secretion competition.
    RenalTubularCompetition { drug_a: DrugId, drug_b: DrugId },
    /// Additive pharmacodynamic effects (e.g. dual QT prolongation).
    AdditiveToxicity { effect: String },
    /// Synergistic pharmacodynamic effects.
    SynergisticEffect { effect: String },
    /// Pharmacodynamic antagonism reducing efficacy.
    TherapeuticAntagonism { effect: String },
    /// Absorption alteration (e.g. chelation, pH changes).
    AbsorptionInterference { mechanism: String },
    /// Dual QT prolongation leading to torsades risk.
    QtProlongationRisk,
    /// Combined hepatotoxicity.
    HepatotoxicityRisk,
    /// Combined nephrotoxicity.
    NephrotoxicityRisk,
    /// Serotonin syndrome risk from combined serotonergic drugs.
    SerotoninSyndromeRisk,
    /// Bleeding risk from combined anticoagulant/antiplatelet drugs.
    BleedingRisk,
    /// CNS depression from combined sedative drugs.
    CnsDepressionRisk,
    /// Electrolyte imbalance.
    ElectrolyteImbalance { electrolyte: String },
    /// Other / uncategorized.
    Other { description: String },
}

impl ConflictMechanism {
    /// Map an `InteractionType` plus drug context to a `ConflictMechanism`.
    pub fn from_interaction_type(
        itype: &InteractionType,
        drug_a: &DrugId,
        drug_b: &DrugId,
    ) -> Self {
        match itype {
            InteractionType::CypInhibition { enzyme } => Self::CypEnzymeInhibition {
                enzyme: enzyme.clone(),
                inhibitor: drug_a.clone(),
                substrate: drug_b.clone(),
            },
            InteractionType::CypInduction { enzyme } => Self::CypEnzymeInduction {
                enzyme: enzyme.clone(),
                inducer: drug_a.clone(),
                substrate: drug_b.clone(),
            },
            InteractionType::ProteinBindingDisplacement => Self::ProteinBindingCompetition {
                displacer: drug_a.clone(),
                displaced: drug_b.clone(),
            },
            InteractionType::RenalCompetition => Self::RenalTubularCompetition {
                drug_a: drug_a.clone(),
                drug_b: drug_b.clone(),
            },
            InteractionType::PharmacodynamicSynergy => Self::SynergisticEffect {
                effect: "Additive pharmacodynamic effects".to_string(),
            },
            InteractionType::PharmacodynamicAntagonism => Self::TherapeuticAntagonism {
                effect: "Opposing pharmacodynamic effects".to_string(),
            },
            InteractionType::AbsorptionAlteration => Self::AbsorptionInterference {
                mechanism: "GI absorption alteration".to_string(),
            },
            InteractionType::QtProlongation => Self::QtProlongationRisk,
            InteractionType::SerotoninSyndrome => Self::SerotoninSyndromeRisk,
            InteractionType::CnsDepression => Self::CnsDepressionRisk,
        }
    }

    /// Severity weight contribution from the mechanism itself.
    pub fn base_weight(&self) -> f64 {
        match self {
            Self::QtProlongationRisk => 9.0,
            Self::SerotoninSyndromeRisk => 8.5,
            Self::BleedingRisk => 8.0,
            Self::CnsDepressionRisk => 7.5,
            Self::HepatotoxicityRisk => 7.0,
            Self::NephrotoxicityRisk => 7.0,
            Self::CypEnzymeInhibition { .. } => 6.0,
            Self::CypEnzymeInduction { .. } => 5.5,
            Self::SynergisticEffect { .. } => 5.0,
            Self::AdditiveToxicity { .. } => 6.5,
            Self::TherapeuticAntagonism { .. } => 4.5,
            Self::ProteinBindingCompetition { .. } => 4.0,
            Self::RenalTubularCompetition { .. } => 3.5,
            Self::AbsorptionInterference { .. } => 3.0,
            Self::ElectrolyteImbalance { .. } => 5.5,
            Self::Other { .. } => 2.0,
        }
    }

    /// Short human-readable summary.
    pub fn summary(&self) -> String {
        match self {
            Self::CypEnzymeInhibition { enzyme, inhibitor, substrate } => {
                format!("{} inhibits CYP {} metabolism of {}", inhibitor, enzyme, substrate)
            }
            Self::CypEnzymeInduction { enzyme, inducer, substrate } => {
                format!("{} induces CYP {} metabolism of {}", inducer, enzyme, substrate)
            }
            Self::ProteinBindingCompetition { displacer, displaced } => {
                format!("{} displaces {} from plasma proteins", displacer, displaced)
            }
            Self::RenalTubularCompetition { drug_a, drug_b } => {
                format!("{} and {} compete for renal tubular secretion", drug_a, drug_b)
            }
            Self::AdditiveToxicity { effect } => format!("Additive toxicity: {}", effect),
            Self::SynergisticEffect { effect } => format!("Synergistic effect: {}", effect),
            Self::TherapeuticAntagonism { effect } => format!("Therapeutic antagonism: {}", effect),
            Self::AbsorptionInterference { mechanism } => format!("Absorption alteration: {}", mechanism),
            Self::QtProlongationRisk => "Combined QT prolongation → torsades risk".to_string(),
            Self::HepatotoxicityRisk => "Combined hepatotoxicity risk".to_string(),
            Self::NephrotoxicityRisk => "Combined nephrotoxicity risk".to_string(),
            Self::SerotoninSyndromeRisk => "Risk of serotonin syndrome".to_string(),
            Self::BleedingRisk => "Increased bleeding risk".to_string(),
            Self::CnsDepressionRisk => "Additive CNS depression".to_string(),
            Self::ElectrolyteImbalance { electrolyte } => {
                format!("Risk of {} imbalance", electrolyte)
            }
            Self::Other { description } => description.clone(),
        }
    }
}

impl fmt::Display for ConflictMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// ConflictReport
// ---------------------------------------------------------------------------

/// A comprehensive conflict report for a patient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub patient_id: String,
    pub total_medications: usize,
    pub total_pairs_analyzed: usize,
    pub conflicts: Vec<ClassifiedConflict>,
    pub risk_summary: RiskSummary,
    pub recommendations: Vec<String>,
}

impl ConflictReport {
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    pub fn has_critical(&self) -> bool {
        self.conflicts
            .iter()
            .any(|c| c.base.severity == ConflictSeverity::Critical)
    }

    pub fn worst_severity(&self) -> Option<ConflictSeverity> {
        self.conflicts.iter().map(|c| c.base.severity).max()
    }
}

/// A conflict enriched with mechanism classification and priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedConflict {
    pub base: ConfirmedConflict,
    pub mechanism: ConflictMechanism,
    pub priority_rank: usize,
    pub composite_score: f64,
    pub actionability: Actionability,
}

/// How actionable a conflict is for a clinician.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Actionability {
    /// Immediate action required (e.g. stop one drug).
    Immediate,
    /// Action recommended (e.g. dose adjustment, add monitoring).
    Recommended,
    /// Awareness sufficient (monitor but likely okay).
    Awareness,
    /// Informational only.
    Informational,
}

impl Actionability {
    pub fn from_severity(sev: ConflictSeverity) -> Self {
        match sev {
            ConflictSeverity::Critical => Self::Immediate,
            ConflictSeverity::Major => Self::Recommended,
            ConflictSeverity::Moderate => Self::Awareness,
            ConflictSeverity::Minor => Self::Informational,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Immediate => "Immediate Action Required",
            Self::Recommended => "Action Recommended",
            Self::Awareness => "Awareness / Monitor",
            Self::Informational => "Informational",
        }
    }
}

impl fmt::Display for Actionability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// RiskSummary
// ---------------------------------------------------------------------------

/// Aggregate risk summary for a patient's medication regimen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSummary {
    pub overall_verdict: SafetyVerdict,
    pub composite_risk_score: f64,
    pub severity_counts: HashMap<String, usize>,
    pub mechanism_counts: HashMap<String, usize>,
    pub highest_risk_pair: Option<(DrugId, DrugId)>,
    pub total_risk_factors: usize,
}

impl RiskSummary {
    pub fn is_safe(&self) -> bool {
        matches!(
            self.overall_verdict,
            SafetyVerdict::Safe | SafetyVerdict::PossiblySafe
        )
    }
}

// ---------------------------------------------------------------------------
// ConflictPrioritizer
// ---------------------------------------------------------------------------

/// Ranks conflicts by clinical importance using a composite scoring formula.
pub struct ConflictPrioritizer {
    severity_weight: f64,
    confidence_weight: f64,
    mechanism_weight: f64,
    patient_factor_weight: f64,
}

impl ConflictPrioritizer {
    pub fn new() -> Self {
        Self {
            severity_weight: 0.40,
            confidence_weight: 0.25,
            mechanism_weight: 0.20,
            patient_factor_weight: 0.15,
        }
    }

    pub fn with_weights(
        severity: f64,
        confidence: f64,
        mechanism: f64,
        patient: f64,
    ) -> Self {
        let total = severity + confidence + mechanism + patient;
        Self {
            severity_weight: severity / total,
            confidence_weight: confidence / total,
            mechanism_weight: mechanism / total,
            patient_factor_weight: patient / total,
        }
    }

    /// Compute a composite score for a single conflict.
    pub fn score(
        &self,
        conflict: &ConfirmedConflict,
        mechanism: &ConflictMechanism,
        patient: &PatientProfile,
    ) -> f64 {
        let sev = conflict.severity.numeric_score() / 10.0;
        let conf = conflict.confidence;
        let mech = mechanism.base_weight() / 10.0;
        let pf = self.patient_risk_factor(patient, conflict);

        (sev * self.severity_weight
            + conf * self.confidence_weight
            + mech * self.mechanism_weight
            + pf * self.patient_factor_weight)
            * 10.0
    }

    /// Patient-specific risk factor modifier.
    fn patient_risk_factor(
        &self,
        patient: &PatientProfile,
        conflict: &ConfirmedConflict,
    ) -> f64 {
        let mut factor: f64 = 0.5; // baseline

        // Age-based risk
        if patient.age > 75 {
            factor += 0.2;
        } else if patient.age > 65 {
            factor += 0.1;
        }

        // Organ impairment
        let clearance = patient.combined_clearance_factor();
        if clearance < 0.5 {
            factor += 0.3;
        } else if clearance < 0.75 {
            factor += 0.15;
        }

        // Polypharmacy burden
        let med_count = patient.medication_count();
        if med_count > 8 {
            factor += 0.2;
        } else if med_count > 5 {
            factor += 0.1;
        }

        factor.min(1.0)
    }

    /// Prioritize a list of conflicts and return them sorted by composite score.
    pub fn prioritize(
        &self,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> Vec<(ConfirmedConflict, ConflictMechanism, f64)> {
        let mut scored: Vec<(ConfirmedConflict, ConflictMechanism, f64)> = conflicts
            .iter()
            .map(|c| {
                let drug_a = c.drugs.first().cloned().unwrap_or_else(|| DrugId::new("?"));
                let drug_b = c.drugs.get(1).cloned().unwrap_or_else(|| DrugId::new("?"));
                let mechanism =
                    ConflictMechanism::from_interaction_type(&c.interaction_type, &drug_a, &drug_b);
                let score = self.score(c, &mechanism, patient);
                (c.clone(), mechanism, score)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }
}

impl Default for ConflictPrioritizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConflictAnalyzer
// ---------------------------------------------------------------------------

/// Main analyzer that classifies, scores, and reports conflicts.
pub struct ConflictAnalyzer {
    prioritizer: ConflictPrioritizer,
}

impl ConflictAnalyzer {
    pub fn new() -> Self {
        Self {
            prioritizer: ConflictPrioritizer::new(),
        }
    }

    pub fn with_prioritizer(prioritizer: ConflictPrioritizer) -> Self {
        Self { prioritizer }
    }

    /// Classify a single interaction type into a conflict mechanism.
    pub fn classify_conflict(
        &self,
        itype: &InteractionType,
        drug_a: &DrugId,
        drug_b: &DrugId,
    ) -> ConflictMechanism {
        ConflictMechanism::from_interaction_type(itype, drug_a, drug_b)
    }

    /// Compute severity for a conflict based on mechanism, drugs, and patient.
    pub fn compute_severity(
        &self,
        mechanism: &ConflictMechanism,
        drug_a: &DrugInfo,
        drug_b: &DrugInfo,
        patient: &PatientProfile,
    ) -> ConflictSeverity {
        let mut score = mechanism.base_weight();

        // Narrow therapeutic index amplifies risk
        if drug_a.is_narrow_therapeutic_index() || drug_b.is_narrow_therapeutic_index() {
            score += 2.0;
        }

        // Impaired clearance amplifies risk
        let clearance = patient.combined_clearance_factor();
        if clearance < 0.5 {
            score += 1.5;
        } else if clearance < 0.75 {
            score += 0.75;
        }

        // Elderly patients
        if patient.age > 75 {
            score += 1.0;
        } else if patient.age > 65 {
            score += 0.5;
        }

        // High protein-binding combo
        if drug_a.protein_binding > 0.9 && drug_b.protein_binding > 0.9 {
            score += 1.0;
        }

        ConflictSeverity::from_score(score)
    }

    /// Generate a full conflict report for a patient.
    pub fn analyze(&self, patient: &PatientProfile, conflicts: &[ConfirmedConflict]) -> ConflictReport {
        let prioritized = self.prioritizer.prioritize(conflicts, patient);

        let classified: Vec<ClassifiedConflict> = prioritized
            .iter()
            .enumerate()
            .map(|(rank, (conflict, mechanism, score))| {
                ClassifiedConflict {
                    base: conflict.clone(),
                    mechanism: mechanism.clone(),
                    priority_rank: rank + 1,
                    composite_score: *score,
                    actionability: Actionability::from_severity(conflict.severity),
                }
            })
            .collect();

        let risk_summary = self.build_risk_summary(&classified);
        let recommendations = self.generate_recommendations(&classified);

        ConflictReport {
            patient_id: patient.id.to_string(),
            total_medications: patient.medication_count(),
            total_pairs_analyzed: if patient.medication_count() > 1 {
                patient.medication_count() * (patient.medication_count() - 1) / 2
            } else {
                0
            },
            conflicts: classified,
            risk_summary,
            recommendations,
        }
    }

    /// Prioritize conflicts by composite score.
    pub fn prioritize_conflicts(
        &self,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> Vec<ClassifiedConflict> {
        let prioritized = self.prioritizer.prioritize(conflicts, patient);
        prioritized
            .into_iter()
            .enumerate()
            .map(|(rank, (conflict, mechanism, score))| {
                ClassifiedConflict {
                    base: conflict.clone(),
                    mechanism,
                    priority_rank: rank + 1,
                    composite_score: score,
                    actionability: Actionability::from_severity(conflict.severity),
                }
            })
            .collect()
    }

    fn build_risk_summary(&self, classified: &[ClassifiedConflict]) -> RiskSummary {
        let mut sev_counts: HashMap<String, usize> = HashMap::new();
        let mut mech_counts: HashMap<String, usize> = HashMap::new();
        let mut max_score = 0.0f64;
        let mut highest_pair: Option<(DrugId, DrugId)> = None;
        let mut total_risk_factors = 0usize;

        for cc in classified {
            *sev_counts
                .entry(cc.base.severity.label().to_string())
                .or_insert(0) += 1;

            let mech_name = match &cc.mechanism {
                ConflictMechanism::CypEnzymeInhibition { .. } => "CYP Inhibition",
                ConflictMechanism::CypEnzymeInduction { .. } => "CYP Induction",
                ConflictMechanism::ProteinBindingCompetition { .. } => "Protein Binding",
                ConflictMechanism::RenalTubularCompetition { .. } => "Renal Competition",
                ConflictMechanism::QtProlongationRisk => "QT Prolongation",
                ConflictMechanism::BleedingRisk => "Bleeding Risk",
                ConflictMechanism::SerotoninSyndromeRisk => "Serotonin Syndrome",
                ConflictMechanism::CnsDepressionRisk => "CNS Depression",
                ConflictMechanism::HepatotoxicityRisk => "Hepatotoxicity",
                ConflictMechanism::NephrotoxicityRisk => "Nephrotoxicity",
                _ => "Other",
            };
            *mech_counts.entry(mech_name.to_string()).or_insert(0) += 1;

            if cc.composite_score > max_score {
                max_score = cc.composite_score;
                highest_pair = cc.base.drug_pair().map(|(a, b)| (a.clone(), b.clone()));
            }

            total_risk_factors += 1;
        }

        let overall = if classified.iter().any(|c| c.base.severity == ConflictSeverity::Critical) {
            SafetyVerdict::Unsafe
        } else if classified.iter().any(|c| c.base.severity == ConflictSeverity::Major) {
            SafetyVerdict::PossiblyUnsafe
        } else if !classified.is_empty() {
            SafetyVerdict::PossiblySafe
        } else {
            SafetyVerdict::Safe
        };

        let composite = if classified.is_empty() {
            0.0
        } else {
            classified.iter().map(|c| c.composite_score).sum::<f64>() / classified.len() as f64
        };

        RiskSummary {
            overall_verdict: overall,
            composite_risk_score: composite,
            severity_counts: sev_counts,
            mechanism_counts: mech_counts,
            highest_risk_pair: highest_pair,
            total_risk_factors,
        }
    }

    fn generate_recommendations(&self, classified: &[ClassifiedConflict]) -> Vec<String> {
        let mut recs = Vec::new();

        for cc in classified {
            match cc.actionability {
                Actionability::Immediate => {
                    recs.push(format!(
                        "URGENT: {} — consider discontinuing one agent or switching to an alternative.",
                        cc.mechanism.summary()
                    ));
                }
                Actionability::Recommended => {
                    recs.push(format!(
                        "RECOMMENDED: {} — adjust dose or increase monitoring frequency.",
                        cc.mechanism.summary()
                    ));
                }
                Actionability::Awareness => {
                    recs.push(format!(
                        "MONITOR: {} — be aware and monitor for adverse effects.",
                        cc.mechanism.summary()
                    ));
                }
                Actionability::Informational => {
                    recs.push(format!(
                        "INFO: {} — unlikely to be clinically significant at current doses.",
                        cc.mechanism.summary()
                    ));
                }
            }
        }

        if classified.len() > 3 {
            recs.push(
                "NOTE: Multiple interactions detected. Consider comprehensive medication review."
                    .to_string(),
            );
        }

        recs
    }
}

impl Default for ConflictAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AdministrationRoute, Dosage, OrganFunction, PatientId,
    };

    fn make_drug(id: &str, name: &str, class: &str) -> DrugInfo {
        DrugInfo {
            id: DrugId::new(id),
            name: name.to_string(),
            therapeutic_class: class.to_string(),
            cyp_enzymes: vec!["3A4".to_string()],
            half_life_hours: 12.0,
            bioavailability: 0.8,
            protein_binding: 0.6,
            therapeutic_index: None,
        }
    }

    fn make_patient() -> PatientProfile {
        PatientProfile::simple("p1", 70, 75.0)
    }

    fn make_conflict(
        a: &str,
        b: &str,
        itype: InteractionType,
        sev: ConflictSeverity,
    ) -> ConfirmedConflict {
        ConfirmedConflict {
            id: format!("c-{}-{}", a, b),
            drugs: vec![DrugId::new(a), DrugId::new(b)],
            interaction_type: itype,
            severity: sev,
            verdict: if sev >= ConflictSeverity::Major {
                SafetyVerdict::PossiblyUnsafe
            } else {
                SafetyVerdict::PossiblySafe
            },
            mechanism_description: "test mechanism".to_string(),
            evidence_tier: VerificationTier::Tier2ModelCheck,
            counter_example: None,
            confidence: 0.85,
            clinical_recommendation: "test".to_string(),
            affected_parameters: vec!["AUC".to_string()],
            guideline_references: vec![],
        }
    }

    #[test]
    fn test_classify_cyp_inhibition() {
        let analyzer = ConflictAnalyzer::new();
        let itype = InteractionType::CypInhibition {
            enzyme: "3A4".to_string(),
        };
        let mech = analyzer.classify_conflict(
            &itype,
            &DrugId::new("fluconazole"),
            &DrugId::new("warfarin"),
        );
        match mech {
            ConflictMechanism::CypEnzymeInhibition { enzyme, .. } => {
                assert_eq!(enzyme, "3A4");
            }
            _ => panic!("Expected CypEnzymeInhibition"),
        }
    }

    #[test]
    fn test_classify_qt_prolongation() {
        let analyzer = ConflictAnalyzer::new();
        let itype = InteractionType::QtProlongation;
        let mech = analyzer.classify_conflict(&itype, &DrugId::new("a"), &DrugId::new("b"));
        assert!(matches!(mech, ConflictMechanism::QtProlongationRisk));
    }

    #[test]
    fn test_compute_severity_normal_patient() {
        let analyzer = ConflictAnalyzer::new();
        let drug_a = make_drug("a", "A", "ClassX");
        let drug_b = make_drug("b", "B", "ClassY");
        let patient = make_patient();
        let mech = ConflictMechanism::CypEnzymeInhibition {
            enzyme: "3A4".to_string(),
            inhibitor: DrugId::new("a"),
            substrate: DrugId::new("b"),
        };
        let sev = analyzer.compute_severity(&mech, &drug_a, &drug_b, &patient);
        // Base weight 6.0 + age>65 = 0.5 → ~6.5 → Major
        assert!(sev >= ConflictSeverity::Major);
    }

    #[test]
    fn test_compute_severity_impaired_patient() {
        let analyzer = ConflictAnalyzer::new();
        let drug_a = make_drug("a", "A", "X");
        let drug_b = make_drug("b", "B", "Y");
        let mut patient = make_patient();
        patient.renal_function = OrganFunction::SevereImpairment;
        patient.hepatic_function = OrganFunction::SevereImpairment;
        patient.age = 80;

        let mech = ConflictMechanism::QtProlongationRisk;
        let sev = analyzer.compute_severity(&mech, &drug_a, &drug_b, &patient);
        assert_eq!(sev, ConflictSeverity::Critical);
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = ConflictAnalyzer::new();
        let patient = make_patient();
        let report = analyzer.analyze(&patient, &[]);
        assert_eq!(report.conflict_count(), 0);
        assert!(report.risk_summary.is_safe());
        assert_eq!(report.risk_summary.overall_verdict, SafetyVerdict::Safe);
    }

    #[test]
    fn test_analyze_with_conflicts() {
        let analyzer = ConflictAnalyzer::new();
        let patient = make_patient();
        let conflicts = vec![
            make_conflict(
                "war",
                "flu",
                InteractionType::CypInhibition {
                    enzyme: "2C9".to_string(),
                },
                ConflictSeverity::Major,
            ),
            make_conflict(
                "ami",
                "sot",
                InteractionType::QtProlongation,
                ConflictSeverity::Critical,
            ),
        ];

        let report = analyzer.analyze(&patient, &conflicts);
        assert_eq!(report.conflict_count(), 2);
        assert!(report.has_critical());
        assert_eq!(
            report.risk_summary.overall_verdict,
            SafetyVerdict::Unsafe
        );
        // First conflict in priority order should be the Critical one
        assert_eq!(report.conflicts[0].base.severity, ConflictSeverity::Critical);
    }

    #[test]
    fn test_prioritizer_ordering() {
        let prioritizer = ConflictPrioritizer::new();
        let patient = make_patient();
        let conflicts = vec![
            make_conflict("a", "b", InteractionType::RenalCompetition, ConflictSeverity::Minor),
            make_conflict(
                "c",
                "d",
                InteractionType::CypInhibition {
                    enzyme: "3A4".to_string(),
                },
                ConflictSeverity::Major,
            ),
            make_conflict("e", "f", InteractionType::QtProlongation, ConflictSeverity::Critical),
        ];

        let ranked = prioritizer.prioritize(&conflicts, &patient);
        assert_eq!(ranked.len(), 3);
        // Critical should be ranked first
        assert!(ranked[0].2 > ranked[1].2);
        assert!(ranked[1].2 > ranked[2].2);
    }

    #[test]
    fn test_conflict_mechanism_summary() {
        let mech = ConflictMechanism::CypEnzymeInhibition {
            enzyme: "2D6".to_string(),
            inhibitor: DrugId::new("paroxetine"),
            substrate: DrugId::new("codeine"),
        };
        let summary = mech.summary();
        assert!(summary.contains("paroxetine"));
        assert!(summary.contains("2D6"));
        assert!(summary.contains("codeine"));
    }

    #[test]
    fn test_actionability_from_severity() {
        assert_eq!(
            Actionability::from_severity(ConflictSeverity::Critical),
            Actionability::Immediate
        );
        assert_eq!(
            Actionability::from_severity(ConflictSeverity::Major),
            Actionability::Recommended
        );
        assert_eq!(
            Actionability::from_severity(ConflictSeverity::Moderate),
            Actionability::Awareness
        );
        assert_eq!(
            Actionability::from_severity(ConflictSeverity::Minor),
            Actionability::Informational
        );
    }

    #[test]
    fn test_risk_summary_safe() {
        let summary = RiskSummary {
            overall_verdict: SafetyVerdict::Safe,
            composite_risk_score: 0.0,
            severity_counts: HashMap::new(),
            mechanism_counts: HashMap::new(),
            highest_risk_pair: None,
            total_risk_factors: 0,
        };
        assert!(summary.is_safe());
    }

    #[test]
    fn test_recommendations_urgent_message() {
        let analyzer = ConflictAnalyzer::new();
        let patient = make_patient();
        let conflicts = vec![make_conflict(
            "x",
            "y",
            InteractionType::QtProlongation,
            ConflictSeverity::Critical,
        )];
        let report = analyzer.analyze(&patient, &conflicts);
        assert!(report.recommendations.iter().any(|r| r.contains("URGENT")));
    }

    #[test]
    fn test_mechanism_base_weight_ordering() {
        let qt = ConflictMechanism::QtProlongationRisk;
        let renal = ConflictMechanism::RenalTubularCompetition {
            drug_a: DrugId::new("a"),
            drug_b: DrugId::new("b"),
        };
        assert!(qt.base_weight() > renal.base_weight());
    }

    #[test]
    fn test_prioritize_conflicts_ranks() {
        let analyzer = ConflictAnalyzer::new();
        let patient = make_patient();
        let conflicts = vec![
            make_conflict("a", "b", InteractionType::AbsorptionAlteration, ConflictSeverity::Minor),
            make_conflict("c", "d", InteractionType::QtProlongation, ConflictSeverity::Critical),
        ];
        let classified = analyzer.prioritize_conflicts(&conflicts, &patient);
        assert_eq!(classified.len(), 2);
        assert_eq!(classified[0].priority_rank, 1);
        assert_eq!(classified[1].priority_rank, 2);
        assert!(classified[0].composite_score >= classified[1].composite_score);
    }
}
