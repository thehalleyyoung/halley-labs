//! Recommendation synthesis: combines dose, alternative, temporal, and schedule
//! recommendations into a single prioritised clinical action plan.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, DrugClass, DrugId, EvidenceLevel,
    InteractionType, PatientProfile, PkDatabase,
};

use crate::dose_adjustment::{DoseAdjuster, DoseAdjustment};
use crate::alternative::{AlternativeFinder, ScoredAlternative};
use crate::temporal::{TemporalRecommender, TemporalRecommendation};

// ---------------------------------------------------------------------------
// Recommendation types
// ---------------------------------------------------------------------------

/// A single recommendation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Recommendation {
    /// Adjust the dose of a drug.
    AdjustDose {
        drug_id: DrugId,
        current_mg: f64,
        suggested_mg: f64,
        rationale: String,
    },
    /// Switch to an alternative medication.
    SwitchMedication {
        from_drug: DrugId,
        to_drug: DrugId,
        to_class: DrugClass,
        rationale: String,
    },
    /// Change administration timing.
    ChangeTiming {
        drug_id: DrugId,
        suggested_time_h: f64,
        rationale: String,
    },
    /// Add monitoring for a drug or parameter.
    AddMonitoring {
        drug_id: DrugId,
        parameter: String,
        frequency_days: u32,
        rationale: String,
    },
    /// Discontinue a drug.
    Discontinue {
        drug_id: DrugId,
        rationale: String,
    },
    /// No action required (conflict is clinically insignificant).
    NoAction {
        conflict_description: String,
        rationale: String,
    },
}

impl Recommendation {
    /// Priority score (higher = more urgent).
    pub fn priority(&self) -> f64 {
        match self {
            Self::Discontinue { .. } => 5.0,
            Self::SwitchMedication { .. } => 4.0,
            Self::AdjustDose { .. } => 3.0,
            Self::ChangeTiming { .. } => 2.0,
            Self::AddMonitoring { .. } => 1.0,
            Self::NoAction { .. } => 0.0,
        }
    }

    /// Short label for the recommendation type.
    pub fn action_type(&self) -> &str {
        match self {
            Self::AdjustDose { .. } => "Dose Adjustment",
            Self::SwitchMedication { .. } => "Medication Switch",
            Self::ChangeTiming { .. } => "Timing Change",
            Self::AddMonitoring { .. } => "Monitoring",
            Self::Discontinue { .. } => "Discontinue",
            Self::NoAction { .. } => "No Action",
        }
    }

    /// Drug id involved (if any).
    pub fn drug_id(&self) -> Option<&DrugId> {
        match self {
            Self::AdjustDose { drug_id, .. }
            | Self::ChangeTiming { drug_id, .. }
            | Self::AddMonitoring { drug_id, .. }
            | Self::Discontinue { drug_id, .. } => Some(drug_id),
            Self::SwitchMedication { from_drug, .. } => Some(from_drug),
            Self::NoAction { .. } => None,
        }
    }
}

/// A prioritised recommendation with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritisedRecommendation {
    pub recommendation: Recommendation,
    pub priority_score: f64,
    pub evidence_level: EvidenceLevel,
    pub confidence: f64,
    pub source_conflict: Option<String>,
}

/// Clinical action plan combining all recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalActionPlan {
    pub recommendations: Vec<PrioritisedRecommendation>,
    pub summary: String,
    pub risk_level: String,
    pub requires_specialist_review: bool,
}

impl ClinicalActionPlan {
    /// Number of actionable recommendations (excluding NoAction).
    pub fn actionable_count(&self) -> usize {
        self.recommendations
            .iter()
            .filter(|r| !matches!(r.recommendation, Recommendation::NoAction { .. }))
            .count()
    }

    /// Get recommendations for a specific drug.
    pub fn for_drug(&self, drug_id: &DrugId) -> Vec<&PrioritisedRecommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.recommendation.drug_id() == Some(drug_id))
            .collect()
    }

    /// Highest priority recommendation.
    pub fn highest_priority(&self) -> Option<&PrioritisedRecommendation> {
        self.recommendations
            .iter()
            .max_by(|a, b| a.priority_score.partial_cmp(&b.priority_score).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// Full report from the synthesizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationReport {
    pub patient_id: String,
    pub conflicts_analysed: usize,
    pub dose_adjustments: Vec<DoseAdjustment>,
    pub alternatives: Vec<(DrugId, Vec<ScoredAlternative>)>,
    pub temporal: Option<TemporalRecommendation>,
    pub action_plan: ClinicalActionPlan,
}

// ---------------------------------------------------------------------------
// RecommendationSynthesizer
// ---------------------------------------------------------------------------

/// Main synthesizer combining all recommendation engines.
#[derive(Debug)]
pub struct RecommendationSynthesizer {
    dose_adjuster: DoseAdjuster,
    alternative_finder: AlternativeFinder,
    temporal_recommender: TemporalRecommender,
    /// Threshold: if dose change > this %, consider switching instead.
    switch_threshold_pct: f64,
    /// Threshold: severity at or above which discontinuation is considered.
    discontinue_severity: ConflictSeverity,
}

impl RecommendationSynthesizer {
    pub fn new(pk_db: PkDatabase) -> Self {
        RecommendationSynthesizer {
            dose_adjuster: DoseAdjuster::new(pk_db.clone()),
            alternative_finder: AlternativeFinder::new(),
            temporal_recommender: TemporalRecommender::new(pk_db),
            switch_threshold_pct: 50.0,
            discontinue_severity: ConflictSeverity::Critical,
        }
    }

    pub fn with_switch_threshold(mut self, pct: f64) -> Self {
        self.switch_threshold_pct = pct;
        self
    }

    /// Synthesize recommendations from conflicts and patient profile.
    pub fn synthesize(
        &self,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> RecommendationReport {
        // 1. Dose adjustments.
        let dose_adjustments = self.dose_adjuster.suggest_adjustments(conflicts, patient);

        // 2. Alternatives for drugs that need large dose changes.
        let mut alternatives: Vec<(DrugId, Vec<ScoredAlternative>)> = Vec::new();
        for adj in &dose_adjustments {
            if adj.percent_change().abs() > self.switch_threshold_pct {
                if let Some(med) = patient.find_medication(&adj.drug_id) {
                    let alts = self.alternative_finder.find_alternatives(
                        &adj.drug_id,
                        &med.drug_class,
                        conflicts,
                        patient,
                    );
                    if !alts.is_empty() {
                        alternatives.push((adj.drug_id.clone(), alts));
                    }
                }
            }
        }

        // 3. Temporal recommendations.
        let temporal = if !conflicts.is_empty() {
            Some(self.temporal_recommender.recommend_timing(conflicts))
        } else {
            None
        };

        // 4. Build action plan.
        let action_plan = self.build_action_plan(
            conflicts,
            &dose_adjustments,
            &alternatives,
            &temporal,
            patient,
        );

        RecommendationReport {
            patient_id: patient.id.clone(),
            conflicts_analysed: conflicts.len(),
            dose_adjustments,
            alternatives,
            temporal,
            action_plan,
        }
    }

    fn build_action_plan(
        &self,
        conflicts: &[ConfirmedConflict],
        dose_adjustments: &[DoseAdjustment],
        alternatives: &[(DrugId, Vec<ScoredAlternative>)],
        temporal: &Option<TemporalRecommendation>,
        patient: &PatientProfile,
    ) -> ClinicalActionPlan {
        let mut recs: Vec<PrioritisedRecommendation> = Vec::new();

        // Check for critical conflicts that may warrant discontinuation.
        for conflict in conflicts {
            if conflict.severity >= self.discontinue_severity {
                // Recommend discontinuation of the less essential drug if possible.
                let less_essential = self.identify_less_essential(
                    &conflict.drug_a,
                    &conflict.drug_b,
                    patient,
                );
                recs.push(PrioritisedRecommendation {
                    recommendation: Recommendation::Discontinue {
                        drug_id: less_essential.clone(),
                        rationale: format!(
                            "Critical interaction ({}) between {} and {} — consider discontinuing {}",
                            conflict.interaction_type,
                            conflict.drug_a,
                            conflict.drug_b,
                            less_essential,
                        ),
                    },
                    priority_score: 5.0 * conflict.severity.priority_weight(),
                    evidence_level: EvidenceLevel::High,
                    confidence: conflict.confidence,
                    source_conflict: Some(conflict.id.to_string()),
                });
            }
        }

        // Medication switches.
        for (drug_id, alts) in alternatives {
            if let Some(best) = alts.first() {
                recs.push(PrioritisedRecommendation {
                    recommendation: Recommendation::SwitchMedication {
                        from_drug: drug_id.clone(),
                        to_drug: best.alternative.drug_id.clone(),
                        to_class: best.alternative.drug_class.clone(),
                        rationale: best.rationale.clone(),
                    },
                    priority_score: 4.0 * best.score,
                    evidence_level: best.alternative.evidence_level,
                    confidence: best.score,
                    source_conflict: None,
                });
            }
        }

        // Dose adjustments.
        for adj in dose_adjustments {
            if (adj.percent_change().abs()) > 5.0 {
                let severity_weight = conflicts
                    .iter()
                    .filter(|c| c.involves(&adj.drug_id))
                    .map(|c| c.severity.priority_weight())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(1.0);

                recs.push(PrioritisedRecommendation {
                    recommendation: Recommendation::AdjustDose {
                        drug_id: adj.drug_id.clone(),
                        current_mg: adj.current_dose_mg,
                        suggested_mg: adj.suggested_dose_mg,
                        rationale: adj.rationale.clone(),
                    },
                    priority_score: 3.0 * severity_weight / 4.0,
                    evidence_level: adj.evidence_level,
                    confidence: adj.confidence,
                    source_conflict: None,
                });
            }
        }

        // Timing changes.
        if let Some(ref temp) = temporal {
            for timing in &temp.drug_timings {
                if let Some(&time) = timing.recommended_hours.first() {
                    let notes_str = timing.notes.join("; ");
                    recs.push(PrioritisedRecommendation {
                        recommendation: Recommendation::ChangeTiming {
                            drug_id: timing.drug_id.clone(),
                            suggested_time_h: time,
                            rationale: format!(
                                "Optimise administration time to {:.0}:{:02} — {}",
                                time.floor(),
                                ((time.fract()) * 60.0).round() as u32,
                                if notes_str.is_empty() {
                                    "reduce interaction overlap".to_string()
                                } else {
                                    notes_str
                                }
                            ),
                        },
                        priority_score: 2.0 * temp.overall_feasibility,
                        evidence_level: EvidenceLevel::Moderate,
                        confidence: temp.overall_feasibility,
                        source_conflict: None,
                    });
                }
            }
        }

        // Monitoring for all involved drugs.
        let mut monitored: Vec<DrugId> = Vec::new();
        for adj in dose_adjustments {
            if monitored.contains(&adj.drug_id) {
                continue;
            }
            monitored.push(adj.drug_id.clone());
            for param in &adj.monitoring_plan.parameters {
                recs.push(PrioritisedRecommendation {
                    recommendation: Recommendation::AddMonitoring {
                        drug_id: adj.drug_id.clone(),
                        parameter: param.name.clone(),
                        frequency_days: param.frequency_days,
                        rationale: format!(
                            "Monitor {} (target: {:.1}–{:.1} {}) every {} days",
                            param.name,
                            param.target_range.0,
                            param.target_range.1,
                            param.unit,
                            param.frequency_days,
                        ),
                    },
                    priority_score: 1.0,
                    evidence_level: adj.evidence_level,
                    confidence: adj.confidence,
                    source_conflict: None,
                });
            }
        }

        // Minor conflicts → NoAction.
        for conflict in conflicts {
            if conflict.severity == ConflictSeverity::Minor {
                let already_covered = recs.iter().any(|r| {
                    r.recommendation.drug_id() == Some(&conflict.drug_a)
                        || r.recommendation.drug_id() == Some(&conflict.drug_b)
                });
                if !already_covered {
                    recs.push(PrioritisedRecommendation {
                        recommendation: Recommendation::NoAction {
                            conflict_description: format!(
                                "{} ↔ {}: {}",
                                conflict.drug_a, conflict.drug_b, conflict.interaction_type
                            ),
                            rationale: "Minor interaction; clinical significance is low. Continue current regimen with routine monitoring.".to_string(),
                        },
                        priority_score: 0.0,
                        evidence_level: EvidenceLevel::Low,
                        confidence: conflict.confidence,
                        source_conflict: Some(conflict.id.to_string()),
                    });
                }
            }
        }

        // Sort by priority descending.
        recs.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let max_severity = conflicts
            .iter()
            .map(|c| c.severity)
            .max()
            .unwrap_or(ConflictSeverity::Minor);

        let risk_level = match max_severity {
            ConflictSeverity::Critical => "High".to_string(),
            ConflictSeverity::Major => "Moderate-High".to_string(),
            ConflictSeverity::Moderate => "Moderate".to_string(),
            ConflictSeverity::Minor => "Low".to_string(),
        };

        let requires_specialist = max_severity >= ConflictSeverity::Major;

        let summary = self.generate_summary(&recs, conflicts.len());

        ClinicalActionPlan {
            recommendations: recs,
            summary,
            risk_level,
            requires_specialist_review: requires_specialist,
        }
    }

    /// Identify which drug is less essential (heuristic: newer, fewer indications).
    fn identify_less_essential<'a>(
        &self,
        drug_a: &'a DrugId,
        drug_b: &'a DrugId,
        patient: &PatientProfile,
    ) -> &'a DrugId {
        let med_a = patient.find_medication(drug_a);
        let med_b = patient.find_medication(drug_b);

        match (med_a, med_b) {
            (Some(a), Some(b)) => {
                // Prefer keeping drugs with indications specified.
                let a_has_indication = a.indication.is_some();
                let b_has_indication = b.indication.is_some();
                if a_has_indication && !b_has_indication {
                    drug_b
                } else if !a_has_indication && b_has_indication {
                    drug_a
                } else {
                    // Keep the drug that requires monitoring (it's likely critical).
                    if a.drug_class.requires_monitoring() && !b.drug_class.requires_monitoring() {
                        drug_b
                    } else {
                        drug_a
                    }
                }
            }
            (Some(_), None) => drug_b,
            (None, Some(_)) => drug_a,
            (None, None) => drug_a,
        }
    }

    fn generate_summary(
        &self,
        recs: &[PrioritisedRecommendation],
        conflict_count: usize,
    ) -> String {
        let discontinuations = recs
            .iter()
            .filter(|r| matches!(r.recommendation, Recommendation::Discontinue { .. }))
            .count();
        let switches = recs
            .iter()
            .filter(|r| matches!(r.recommendation, Recommendation::SwitchMedication { .. }))
            .count();
        let dose_changes = recs
            .iter()
            .filter(|r| matches!(r.recommendation, Recommendation::AdjustDose { .. }))
            .count();
        let timing_changes = recs
            .iter()
            .filter(|r| matches!(r.recommendation, Recommendation::ChangeTiming { .. }))
            .count();
        let monitoring = recs
            .iter()
            .filter(|r| matches!(r.recommendation, Recommendation::AddMonitoring { .. }))
            .count();

        let mut parts = vec![format!(
            "Analysed {} conflict(s) → {} recommendation(s)",
            conflict_count,
            recs.len()
        )];

        if discontinuations > 0 {
            parts.push(format!("{} discontinuation(s)", discontinuations));
        }
        if switches > 0 {
            parts.push(format!("{} medication switch(es)", switches));
        }
        if dose_changes > 0 {
            parts.push(format!("{} dose adjustment(s)", dose_changes));
        }
        if timing_changes > 0 {
            parts.push(format!("{} timing change(s)", timing_changes));
        }
        if monitoring > 0 {
            parts.push(format!("{} monitoring parameter(s)", monitoring));
        }

        parts.join(". ")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_patient() -> PatientProfile {
        let mut p = PatientProfile::new("pt1", 65, 80.0)
            .with_sex(Sex::Male)
            .with_renal_function(RenalFunction::Normal)
            .with_hepatic_function(HepaticFunction::Normal);
        p.add_medication(
            ActiveMedication::new(
                DrugId::new("atorvastatin"),
                DrugClass::Statin,
                40.0,
                1,
            )
            .with_indication("Hyperlipidemia"),
        );
        p.add_medication(ActiveMedication::new(
            DrugId::new("metoprolol"),
            DrugClass::BetaBlocker,
            50.0,
            2,
        ));
        p.add_medication(ActiveMedication::new(
            DrugId::new("omeprazole"),
            DrugClass::PPI,
            20.0,
            1,
        ));
        p
    }

    fn make_major_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new("itraconazole"),
            DrugId::new("atorvastatin"),
            InteractionType::CypInhibition {
                enzyme: "CYP3A4".to_string(),
            },
            ConflictSeverity::Major,
        )
    }

    fn make_critical_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new("ketoconazole"),
            DrugId::new("atorvastatin"),
            InteractionType::CypInhibition {
                enzyme: "CYP3A4".to_string(),
            },
            ConflictSeverity::Critical,
        )
    }

    fn make_minor_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new("omeprazole"),
            DrugId::new("metoprolol"),
            InteractionType::AbsorptionAlteration,
            ConflictSeverity::Minor,
        )
    }

    #[test]
    fn test_synthesize_basic() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("itraconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let conflict = make_major_conflict();
        let report = synth.synthesize(&[conflict], &patient);
        assert_eq!(report.conflicts_analysed, 1);
        assert!(!report.action_plan.recommendations.is_empty());
    }

    #[test]
    fn test_synthesize_critical_produces_discontinue() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("ketoconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let conflict = make_critical_conflict();
        let report = synth.synthesize(&[conflict], &patient);
        let has_discontinue = report
            .action_plan
            .recommendations
            .iter()
            .any(|r| matches!(r.recommendation, Recommendation::Discontinue { .. }));
        assert!(has_discontinue, "Critical conflict should produce discontinuation recommendation");
    }

    #[test]
    fn test_synthesize_minor_produces_no_action_or_monitoring() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let patient = make_patient();
        let conflict = make_minor_conflict();
        let report = synth.synthesize(&[conflict], &patient);
        assert!(!report.action_plan.recommendations.is_empty());
    }

    #[test]
    fn test_action_plan_sorted_by_priority() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("ketoconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let conflicts = vec![make_critical_conflict(), make_minor_conflict()];
        let report = synth.synthesize(&conflicts, &patient);
        for pair in report.action_plan.recommendations.windows(2) {
            assert!(pair[0].priority_score >= pair[1].priority_score);
        }
    }

    #[test]
    fn test_action_plan_risk_level() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("ketoconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let report = synth.synthesize(&[make_critical_conflict()], &patient);
        assert_eq!(report.action_plan.risk_level, "High");
        assert!(report.action_plan.requires_specialist_review);
    }

    #[test]
    fn test_synthesize_includes_temporal() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("itraconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let report = synth.synthesize(&[make_major_conflict()], &patient);
        assert!(report.temporal.is_some());
    }

    #[test]
    fn test_recommendation_priority_ordering() {
        assert!(Recommendation::Discontinue {
            drug_id: DrugId::new("x"),
            rationale: String::new(),
        }
        .priority()
            > Recommendation::AdjustDose {
                drug_id: DrugId::new("x"),
                current_mg: 0.0,
                suggested_mg: 0.0,
                rationale: String::new(),
            }
            .priority());
    }

    #[test]
    fn test_clinical_action_plan_for_drug() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("itraconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let report = synth.synthesize(&[make_major_conflict()], &patient);
        let atorv_recs = report
            .action_plan
            .for_drug(&DrugId::new("atorvastatin"));
        assert!(!atorv_recs.is_empty());
    }

    #[test]
    fn test_empty_conflicts_produces_empty_plan() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let patient = make_patient();
        let report = synth.synthesize(&[], &patient);
        assert_eq!(report.conflicts_analysed, 0);
        assert_eq!(report.action_plan.actionable_count(), 0);
    }

    #[test]
    fn test_summary_contains_counts() {
        let pk_db = PkDatabase::demo();
        let synth = RecommendationSynthesizer::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("ketoconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let report = synth.synthesize(
            &[make_critical_conflict(), make_minor_conflict()],
            &patient,
        );
        assert!(report.action_plan.summary.contains("2 conflict(s)"));
    }
}
