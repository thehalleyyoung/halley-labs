//! Alternative medication finder.
//!
//! Identifies therapeutic alternatives when a drug interaction cannot be
//! resolved by dose adjustment or schedule changes alone.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, DrugClass, DrugId, EvidenceLevel,
    InteractionType, PatientProfile,
};

// ---------------------------------------------------------------------------
// Therapeutic equivalence
// ---------------------------------------------------------------------------

/// Describes a therapeutic alternative drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeMedication {
    pub drug_id: DrugId,
    pub drug_class: DrugClass,
    pub typical_dose_mg: f64,
    pub doses_per_day: u32,
    /// Relative potency vs. the reference drug in its class (1.0 = equivalent).
    pub relative_potency: f64,
    /// Known interaction risk with the other drugs in the regimen.
    pub interaction_risk: InteractionRisk,
    /// Cost tier (1 = generic/cheapest, 3 = branded/most expensive).
    pub cost_tier: u8,
    /// Evidence level for therapeutic equivalence.
    pub evidence_level: EvidenceLevel,
    /// Free-text notes.
    pub notes: String,
}

/// Interaction risk assessment for an alternative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionRisk {
    None,
    Low,
    Moderate,
    High,
}

impl InteractionRisk {
    pub fn score(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Low => 0.25,
            Self::Moderate => 0.5,
            Self::High => 1.0,
        }
    }
}

/// Mapping of equivalent drugs within a class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TherapeuticEquivalence {
    pub drug_class: DrugClass,
    pub reference_drug: DrugId,
    pub alternatives: Vec<EquivalenceEntry>,
}

/// One entry in an equivalence class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceEntry {
    pub drug_id: DrugId,
    pub relative_potency: f64,
    pub typical_dose_mg: f64,
    pub doses_per_day: u32,
    pub cost_tier: u8,
    pub cyp_profile: Vec<String>,
}

/// A scored alternative recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredAlternative {
    pub alternative: AlternativeMedication,
    /// Overall score in [0, 1] (higher is better).
    pub score: f64,
    pub score_breakdown: ScoreBreakdown,
    pub rationale: String,
}

/// Breakdown of the scoring factors for an alternative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub efficacy_score: f64,
    pub safety_score: f64,
    pub cost_score: f64,
    pub convenience_score: f64,
    pub allergy_score: f64,
}

// ---------------------------------------------------------------------------
// Alternative database
// ---------------------------------------------------------------------------

/// In-memory database of therapeutic alternatives.
#[derive(Debug, Clone)]
pub struct AlternativeDatabase {
    equivalences: Vec<TherapeuticEquivalence>,
}

impl AlternativeDatabase {
    pub fn new() -> Self {
        AlternativeDatabase {
            equivalences: Vec::new(),
        }
    }

    pub fn add_equivalence(&mut self, eq: TherapeuticEquivalence) {
        self.equivalences.push(eq);
    }

    /// Find equivalence entries for a given drug class.
    pub fn find_by_class(&self, class: &DrugClass) -> Vec<&TherapeuticEquivalence> {
        self.equivalences
            .iter()
            .filter(|e| e.drug_class == *class)
            .collect()
    }

    /// Find equivalence entries whose reference matches a drug id.
    pub fn find_by_reference(&self, drug_id: &DrugId) -> Option<&TherapeuticEquivalence> {
        self.equivalences
            .iter()
            .find(|e| {
                e.reference_drug == *drug_id
                    || e.alternatives.iter().any(|a| a.drug_id == *drug_id)
            })
    }

    /// Build a database with well-known therapeutic equivalences.
    pub fn default_database() -> Self {
        let mut db = AlternativeDatabase::new();

        // Statins
        db.add_equivalence(TherapeuticEquivalence {
            drug_class: DrugClass::Statin,
            reference_drug: DrugId::new("atorvastatin"),
            alternatives: vec![
                EquivalenceEntry {
                    drug_id: DrugId::new("rosuvastatin"),
                    relative_potency: 1.5,
                    typical_dose_mg: 10.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2C9".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("simvastatin"),
                    relative_potency: 0.5,
                    typical_dose_mg: 40.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP3A4".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("pravastatin"),
                    relative_potency: 0.25,
                    typical_dose_mg: 40.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec![],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("fluvastatin"),
                    relative_potency: 0.2,
                    typical_dose_mg: 80.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2C9".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("pitavastatin"),
                    relative_potency: 1.0,
                    typical_dose_mg: 2.0,
                    doses_per_day: 1,
                    cost_tier: 2,
                    cyp_profile: vec![],
                },
            ],
        });

        // ACE Inhibitors / ARBs
        db.add_equivalence(TherapeuticEquivalence {
            drug_class: DrugClass::ACEInhibitor,
            reference_drug: DrugId::new("lisinopril"),
            alternatives: vec![
                EquivalenceEntry {
                    drug_id: DrugId::new("enalapril"),
                    relative_potency: 0.75,
                    typical_dose_mg: 10.0,
                    doses_per_day: 2,
                    cost_tier: 1,
                    cyp_profile: vec![],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("ramipril"),
                    relative_potency: 2.0,
                    typical_dose_mg: 5.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec![],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("benazepril"),
                    relative_potency: 1.0,
                    typical_dose_mg: 20.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec![],
                },
            ],
        });

        db.add_equivalence(TherapeuticEquivalence {
            drug_class: DrugClass::ARB,
            reference_drug: DrugId::new("losartan"),
            alternatives: vec![
                EquivalenceEntry {
                    drug_id: DrugId::new("valsartan"),
                    relative_potency: 1.6,
                    typical_dose_mg: 160.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec![],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("irbesartan"),
                    relative_potency: 1.5,
                    typical_dose_mg: 150.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2C9".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("telmisartan"),
                    relative_potency: 2.0,
                    typical_dose_mg: 40.0,
                    doses_per_day: 1,
                    cost_tier: 2,
                    cyp_profile: vec![],
                },
            ],
        });

        // SSRIs
        db.add_equivalence(TherapeuticEquivalence {
            drug_class: DrugClass::SSRI,
            reference_drug: DrugId::new("sertraline"),
            alternatives: vec![
                EquivalenceEntry {
                    drug_id: DrugId::new("escitalopram"),
                    relative_potency: 2.0,
                    typical_dose_mg: 10.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2C19".to_string(), "CYP3A4".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("citalopram"),
                    relative_potency: 1.0,
                    typical_dose_mg: 20.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2C19".to_string(), "CYP3A4".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("fluoxetine"),
                    relative_potency: 1.0,
                    typical_dose_mg: 20.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2D6".to_string(), "CYP2C9".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("paroxetine"),
                    relative_potency: 1.0,
                    typical_dose_mg: 20.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2D6".to_string()],
                },
            ],
        });

        // PPIs
        db.add_equivalence(TherapeuticEquivalence {
            drug_class: DrugClass::PPI,
            reference_drug: DrugId::new("omeprazole"),
            alternatives: vec![
                EquivalenceEntry {
                    drug_id: DrugId::new("esomeprazole"),
                    relative_potency: 1.2,
                    typical_dose_mg: 20.0,
                    doses_per_day: 1,
                    cost_tier: 2,
                    cyp_profile: vec!["CYP2C19".to_string(), "CYP3A4".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("lansoprazole"),
                    relative_potency: 0.9,
                    typical_dose_mg: 30.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2C19".to_string(), "CYP3A4".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("pantoprazole"),
                    relative_potency: 0.7,
                    typical_dose_mg: 40.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2C19".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("rabeprazole"),
                    relative_potency: 0.8,
                    typical_dose_mg: 20.0,
                    doses_per_day: 1,
                    cost_tier: 2,
                    cyp_profile: vec![],
                },
            ],
        });

        // Beta-blockers
        db.add_equivalence(TherapeuticEquivalence {
            drug_class: DrugClass::BetaBlocker,
            reference_drug: DrugId::new("metoprolol"),
            alternatives: vec![
                EquivalenceEntry {
                    drug_id: DrugId::new("atenolol"),
                    relative_potency: 1.0,
                    typical_dose_mg: 50.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec![],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("bisoprolol"),
                    relative_potency: 4.0,
                    typical_dose_mg: 5.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2D6".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("carvedilol"),
                    relative_potency: 0.5,
                    typical_dose_mg: 25.0,
                    doses_per_day: 2,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP2D6".to_string(), "CYP2C9".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("nebivolol"),
                    relative_potency: 4.0,
                    typical_dose_mg: 5.0,
                    doses_per_day: 1,
                    cost_tier: 2,
                    cyp_profile: vec!["CYP2D6".to_string()],
                },
            ],
        });

        // Calcium channel blockers
        db.add_equivalence(TherapeuticEquivalence {
            drug_class: DrugClass::CalciumChannelBlocker,
            reference_drug: DrugId::new("amlodipine"),
            alternatives: vec![
                EquivalenceEntry {
                    drug_id: DrugId::new("nifedipine"),
                    relative_potency: 0.5,
                    typical_dose_mg: 30.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP3A4".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("felodipine"),
                    relative_potency: 1.0,
                    typical_dose_mg: 5.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP3A4".to_string()],
                },
                EquivalenceEntry {
                    drug_id: DrugId::new("diltiazem"),
                    relative_potency: 0.3,
                    typical_dose_mg: 180.0,
                    doses_per_day: 1,
                    cost_tier: 1,
                    cyp_profile: vec!["CYP3A4".to_string()],
                },
            ],
        });

        db
    }
}

impl Default for AlternativeDatabase {
    fn default() -> Self {
        Self::default_database()
    }
}

// ---------------------------------------------------------------------------
// AlternativeFinder
// ---------------------------------------------------------------------------

/// Weights for alternative scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub efficacy: f64,
    pub safety: f64,
    pub cost: f64,
    pub convenience: f64,
    pub allergy: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        ScoringWeights {
            efficacy: 0.30,
            safety: 0.35,
            cost: 0.15,
            convenience: 0.10,
            allergy: 0.10,
        }
    }
}

/// Engine for finding and scoring alternative medications.
#[derive(Debug, Clone)]
pub struct AlternativeFinder {
    db: AlternativeDatabase,
    weights: ScoringWeights,
    /// Enzyme strings that are involved in current conflicts (to avoid).
    conflict_enzymes: Vec<String>,
}

impl AlternativeFinder {
    pub fn new() -> Self {
        AlternativeFinder {
            db: AlternativeDatabase::default_database(),
            weights: ScoringWeights::default(),
            conflict_enzymes: Vec::new(),
        }
    }

    pub fn with_database(db: AlternativeDatabase) -> Self {
        AlternativeFinder {
            db,
            weights: ScoringWeights::default(),
            conflict_enzymes: Vec::new(),
        }
    }

    pub fn with_weights(mut self, w: ScoringWeights) -> Self {
        self.weights = w;
        self
    }

    /// Find and score alternatives for a drug that is part of a conflict.
    pub fn find_alternatives(
        &self,
        drug_id: &DrugId,
        drug_class: &DrugClass,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> Vec<ScoredAlternative> {
        // Collect enzymes from conflicts involving this drug.
        let conflict_enzymes = self.extract_conflict_enzymes(drug_id, conflicts);

        // Collect drugs the patient is currently taking (to exclude).
        let current_drugs: Vec<&DrugId> = patient
            .medications
            .iter()
            .map(|m| &m.drug_id)
            .collect();

        // Find equivalence class.
        let by_class_owned;
        let equiv = match self.db.find_by_reference(drug_id) {
            Some(eq) => eq,
            None => {
                // Try by class.
                by_class_owned = self.db.find_by_class(drug_class);
                match by_class_owned.first() {
                    Some(eq) => eq,
                    None => return Vec::new(),
                }
            }
        };

        let mut scored: Vec<ScoredAlternative> = Vec::new();

        for entry in &equiv.alternatives {
            // Skip the drug itself and drugs the patient is already taking.
            if entry.drug_id == *drug_id || current_drugs.contains(&&entry.drug_id) {
                continue;
            }

            // Check allergy.
            if patient.has_allergy(entry.drug_id.as_str()) {
                continue;
            }

            let interaction_risk = self.assess_interaction_risk(entry, &conflict_enzymes);
            let alt = AlternativeMedication {
                drug_id: entry.drug_id.clone(),
                drug_class: equiv.drug_class.clone(),
                typical_dose_mg: entry.typical_dose_mg,
                doses_per_day: entry.doses_per_day,
                relative_potency: entry.relative_potency,
                interaction_risk,
                cost_tier: entry.cost_tier,
                evidence_level: EvidenceLevel::Moderate,
                notes: String::new(),
            };

            let (score, breakdown) = self.score_alternative(&alt, conflicts, patient);
            let rationale = self.build_rationale(&alt, &breakdown, drug_id);

            scored.push(ScoredAlternative {
                alternative: alt,
                score,
                score_breakdown: breakdown,
                rationale,
            });
        }

        // Also check if reference drug (if different from current) is an option.
        if equiv.reference_drug != *drug_id && !current_drugs.contains(&&equiv.reference_drug) {
            if !patient.has_allergy(equiv.reference_drug.as_str()) {
                let interaction_risk = InteractionRisk::Low;
                let alt = AlternativeMedication {
                    drug_id: equiv.reference_drug.clone(),
                    drug_class: equiv.drug_class.clone(),
                    typical_dose_mg: 20.0,
                    doses_per_day: 1,
                    relative_potency: 1.0,
                    interaction_risk,
                    cost_tier: 1,
                    evidence_level: EvidenceLevel::Moderate,
                    notes: "Reference drug in class".to_string(),
                };
                let (score, breakdown) = self.score_alternative(&alt, conflicts, patient);
                let rationale = self.build_rationale(&alt, &breakdown, drug_id);
                scored.push(ScoredAlternative {
                    alternative: alt,
                    score,
                    score_breakdown: breakdown,
                    rationale,
                });
            }
        }

        // Sort by score descending.
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Score an alternative medication on [0, 1].
    pub fn score_alternative(
        &self,
        alt: &AlternativeMedication,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> (f64, ScoreBreakdown) {
        let efficacy = self.score_efficacy(alt);
        let safety = self.score_safety(alt, conflicts);
        let cost = self.score_cost(alt);
        let convenience = self.score_convenience(alt);
        let allergy = self.score_allergy(alt, patient);

        let total = self.weights.efficacy * efficacy
            + self.weights.safety * safety
            + self.weights.cost * cost
            + self.weights.convenience * convenience
            + self.weights.allergy * allergy;

        let breakdown = ScoreBreakdown {
            efficacy_score: efficacy,
            safety_score: safety,
            cost_score: cost,
            convenience_score: convenience,
            allergy_score: allergy,
        };

        (total.clamp(0.0, 1.0), breakdown)
    }

    fn score_efficacy(&self, alt: &AlternativeMedication) -> f64 {
        // Potency within 0.5-2x of reference is considered equivalent.
        let potency = alt.relative_potency;
        if potency >= 0.8 && potency <= 1.5 {
            1.0
        } else if potency >= 0.5 && potency <= 2.0 {
            0.8
        } else if potency >= 0.2 && potency <= 3.0 {
            0.6
        } else {
            0.4
        }
    }

    fn score_safety(&self, alt: &AlternativeMedication, _conflicts: &[ConfirmedConflict]) -> f64 {
        // Lower interaction risk → higher score.
        1.0 - alt.interaction_risk.score()
    }

    fn score_cost(&self, alt: &AlternativeMedication) -> f64 {
        match alt.cost_tier {
            1 => 1.0,
            2 => 0.7,
            3 => 0.4,
            _ => 0.3,
        }
    }

    fn score_convenience(&self, alt: &AlternativeMedication) -> f64 {
        // Once daily is most convenient.
        match alt.doses_per_day {
            1 => 1.0,
            2 => 0.75,
            3 => 0.5,
            _ => 0.3,
        }
    }

    fn score_allergy(&self, alt: &AlternativeMedication, patient: &PatientProfile) -> f64 {
        // Check for allergy to the alternative or its class.
        if patient.has_allergy(alt.drug_id.as_str()) {
            0.0
        } else if patient.has_allergy(&format!("{}", alt.drug_class)) {
            0.2
        } else {
            1.0
        }
    }

    /// Assess interaction risk for an equivalence entry based on CYP overlap with conflicts.
    fn assess_interaction_risk(
        &self,
        entry: &EquivalenceEntry,
        conflict_enzymes: &[String],
    ) -> InteractionRisk {
        if entry.cyp_profile.is_empty() {
            return InteractionRisk::None;
        }

        let overlap: usize = entry
            .cyp_profile
            .iter()
            .filter(|e| conflict_enzymes.iter().any(|ce| ce == *e))
            .count();

        if overlap == 0 {
            InteractionRisk::None
        } else if overlap == 1 {
            InteractionRisk::Low
        } else if overlap <= entry.cyp_profile.len() / 2 {
            InteractionRisk::Moderate
        } else {
            InteractionRisk::High
        }
    }

    fn extract_conflict_enzymes(
        &self,
        drug_id: &DrugId,
        conflicts: &[ConfirmedConflict],
    ) -> Vec<String> {
        let mut enzymes = Vec::new();
        for c in conflicts {
            if !c.involves(drug_id) {
                continue;
            }
            match &c.interaction_type {
                InteractionType::CypInhibition { enzyme }
                | InteractionType::CypInduction { enzyme } => {
                    if !enzymes.contains(enzyme) {
                        enzymes.push(enzyme.clone());
                    }
                }
                _ => {}
            }
        }
        enzymes
    }

    fn build_rationale(
        &self,
        alt: &AlternativeMedication,
        breakdown: &ScoreBreakdown,
        replacing: &DrugId,
    ) -> String {
        let mut parts = Vec::new();
        parts.push(format!(
            "Replace {} with {} ({})",
            replacing, alt.drug_id, alt.drug_class
        ));
        parts.push(format!(
            "Relative potency {:.1}x, typical dose {:.0}mg {}x/day",
            alt.relative_potency, alt.typical_dose_mg, alt.doses_per_day
        ));
        parts.push(format!(
            "Interaction risk: {:?}, cost tier: {}",
            alt.interaction_risk, alt.cost_tier
        ));
        parts.push(format!(
            "Scores — efficacy: {:.2}, safety: {:.2}, cost: {:.2}, convenience: {:.2}",
            breakdown.efficacy_score,
            breakdown.safety_score,
            breakdown.cost_score,
            breakdown.convenience_score
        ));
        parts.join(". ")
    }
}

impl Default for AlternativeFinder {
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
    use crate::types::*;

    fn make_patient_with_statin() -> PatientProfile {
        let mut p = PatientProfile::new("pt1", 65, 80.0);
        p.add_medication(ActiveMedication::new(
            DrugId::new("atorvastatin"),
            DrugClass::Statin,
            40.0,
            1,
        ));
        p
    }

    fn make_cyp3a4_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new("itraconazole"),
            DrugId::new("atorvastatin"),
            InteractionType::CypInhibition {
                enzyme: "CYP3A4".to_string(),
            },
            ConflictSeverity::Major,
        )
    }

    #[test]
    fn test_find_statin_alternatives() {
        let finder = AlternativeFinder::new();
        let patient = make_patient_with_statin();
        let conflict = make_cyp3a4_conflict();
        let alts = finder.find_alternatives(
            &DrugId::new("atorvastatin"),
            &DrugClass::Statin,
            &[conflict],
            &patient,
        );
        assert!(!alts.is_empty());
        // Pravastatin has no CYP profile → should rank high.
        let pravastatin = alts
            .iter()
            .find(|a| a.alternative.drug_id == DrugId::new("pravastatin"));
        assert!(pravastatin.is_some());
    }

    #[test]
    fn test_alternatives_exclude_current_drugs() {
        let finder = AlternativeFinder::new();
        let mut patient = make_patient_with_statin();
        // Also taking rosuvastatin already.
        patient.add_medication(ActiveMedication::new(
            DrugId::new("rosuvastatin"),
            DrugClass::Statin,
            10.0,
            1,
        ));
        let conflict = make_cyp3a4_conflict();
        let alts = finder.find_alternatives(
            &DrugId::new("atorvastatin"),
            &DrugClass::Statin,
            &[conflict],
            &patient,
        );
        assert!(alts
            .iter()
            .all(|a| a.alternative.drug_id != DrugId::new("rosuvastatin")));
    }

    #[test]
    fn test_alternatives_exclude_allergies() {
        let finder = AlternativeFinder::new();
        let mut patient = make_patient_with_statin();
        patient.add_allergy("pravastatin");
        let conflict = make_cyp3a4_conflict();
        let alts = finder.find_alternatives(
            &DrugId::new("atorvastatin"),
            &DrugClass::Statin,
            &[conflict],
            &patient,
        );
        assert!(alts
            .iter()
            .all(|a| a.alternative.drug_id != DrugId::new("pravastatin")));
    }

    #[test]
    fn test_score_alternative_safety() {
        let finder = AlternativeFinder::new();
        let alt_safe = AlternativeMedication {
            drug_id: DrugId::new("pravastatin"),
            drug_class: DrugClass::Statin,
            typical_dose_mg: 40.0,
            doses_per_day: 1,
            relative_potency: 0.25,
            interaction_risk: InteractionRisk::None,
            cost_tier: 1,
            evidence_level: EvidenceLevel::Moderate,
            notes: String::new(),
        };
        let alt_risky = AlternativeMedication {
            drug_id: DrugId::new("simvastatin"),
            drug_class: DrugClass::Statin,
            typical_dose_mg: 40.0,
            doses_per_day: 1,
            relative_potency: 0.5,
            interaction_risk: InteractionRisk::High,
            cost_tier: 1,
            evidence_level: EvidenceLevel::Moderate,
            notes: String::new(),
        };
        let (score_safe, _) = finder.score_alternative(&alt_safe, &[], &make_patient_with_statin());
        let (score_risky, _) = finder.score_alternative(&alt_risky, &[], &make_patient_with_statin());
        assert!(score_safe > score_risky);
    }

    #[test]
    fn test_alternatives_sorted_by_score() {
        let finder = AlternativeFinder::new();
        let patient = make_patient_with_statin();
        let conflict = make_cyp3a4_conflict();
        let alts = finder.find_alternatives(
            &DrugId::new("atorvastatin"),
            &DrugClass::Statin,
            &[conflict],
            &patient,
        );
        for pair in alts.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn test_beta_blocker_alternatives() {
        let finder = AlternativeFinder::new();
        let mut patient = PatientProfile::new("pt1", 55, 75.0);
        patient.add_medication(ActiveMedication::new(
            DrugId::new("metoprolol"),
            DrugClass::BetaBlocker,
            50.0,
            2,
        ));
        let conflict = ConfirmedConflict::new(
            DrugId::new("fluoxetine"),
            DrugId::new("metoprolol"),
            InteractionType::CypInhibition {
                enzyme: "CYP2D6".to_string(),
            },
            ConflictSeverity::Moderate,
        );
        let alts = finder.find_alternatives(
            &DrugId::new("metoprolol"),
            &DrugClass::BetaBlocker,
            &[conflict],
            &patient,
        );
        assert!(!alts.is_empty());
        // Atenolol has no CYP profile → should score well.
        let atenolol = alts
            .iter()
            .find(|a| a.alternative.drug_id == DrugId::new("atenolol"));
        assert!(atenolol.is_some());
    }

    #[test]
    fn test_ppi_alternatives() {
        let finder = AlternativeFinder::new();
        let mut patient = PatientProfile::new("pt2", 45, 70.0);
        patient.add_medication(ActiveMedication::new(
            DrugId::new("omeprazole"),
            DrugClass::PPI,
            20.0,
            1,
        ));
        let conflict = ConfirmedConflict::new(
            DrugId::new("clopidogrel"),
            DrugId::new("omeprazole"),
            InteractionType::CypInhibition {
                enzyme: "CYP2C19".to_string(),
            },
            ConflictSeverity::Major,
        );
        let alts = finder.find_alternatives(
            &DrugId::new("omeprazole"),
            &DrugClass::PPI,
            &[conflict],
            &patient,
        );
        assert!(!alts.is_empty());
        // Rabeprazole has minimal CYP2C19 involvement.
        let rabeprazole = alts
            .iter()
            .find(|a| a.alternative.drug_id == DrugId::new("rabeprazole"));
        assert!(rabeprazole.is_some());
    }

    #[test]
    fn test_interaction_risk_assessment() {
        let finder = AlternativeFinder::new();
        let entry_no_cyp = EquivalenceEntry {
            drug_id: DrugId::new("pravastatin"),
            relative_potency: 0.25,
            typical_dose_mg: 40.0,
            doses_per_day: 1,
            cost_tier: 1,
            cyp_profile: vec![],
        };
        let entry_cyp3a4 = EquivalenceEntry {
            drug_id: DrugId::new("simvastatin"),
            relative_potency: 0.5,
            typical_dose_mg: 40.0,
            doses_per_day: 1,
            cost_tier: 1,
            cyp_profile: vec!["CYP3A4".to_string()],
        };
        let enzymes = vec!["CYP3A4".to_string()];
        assert_eq!(
            finder.assess_interaction_risk(&entry_no_cyp, &enzymes),
            InteractionRisk::None
        );
        assert_ne!(
            finder.assess_interaction_risk(&entry_cyp3a4, &enzymes),
            InteractionRisk::None
        );
    }

    #[test]
    fn test_no_alternatives_for_unknown_class() {
        let finder = AlternativeFinder::new();
        let patient = PatientProfile::new("pt1", 50, 70.0);
        let alts = finder.find_alternatives(
            &DrugId::new("exotic_drug"),
            &DrugClass::Other("Exotic".to_string()),
            &[],
            &patient,
        );
        assert!(alts.is_empty());
    }

    #[test]
    fn test_score_breakdown_components() {
        let finder = AlternativeFinder::new();
        let alt = AlternativeMedication {
            drug_id: DrugId::new("pravastatin"),
            drug_class: DrugClass::Statin,
            typical_dose_mg: 40.0,
            doses_per_day: 1,
            relative_potency: 1.0,
            interaction_risk: InteractionRisk::None,
            cost_tier: 1,
            evidence_level: EvidenceLevel::High,
            notes: String::new(),
        };
        let patient = PatientProfile::new("pt1", 50, 70.0);
        let (score, breakdown) = finder.score_alternative(&alt, &[], &patient);
        assert!(score > 0.8);
        assert!((breakdown.safety_score - 1.0).abs() < 0.01);
        assert!((breakdown.cost_score - 1.0).abs() < 0.01);
        assert!((breakdown.convenience_score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_alternative_database_find_by_class() {
        let db = AlternativeDatabase::default_database();
        let statins = db.find_by_class(&DrugClass::Statin);
        assert!(!statins.is_empty());
        let ccbs = db.find_by_class(&DrugClass::CalciumChannelBlocker);
        assert!(!ccbs.is_empty());
    }
}
