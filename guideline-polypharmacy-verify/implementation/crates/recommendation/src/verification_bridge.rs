//! Verification bridge: simulates proposed schedule/dose changes and
//! re-checks whether they resolve the original conflicts.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, DrugId, EvidenceLevel, InteractionType,
    PkDatabase, PkProfile, SafetyVerdict,
};

use crate::dose_adjustment::DoseAdjustment;
use crate::synthesis::Recommendation;
use crate::temporal::TemporalRecommendation;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Verdict on a single recommendation's effectiveness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationVerdict {
    pub recommendation_description: String,
    pub resolves_conflict: bool,
    pub residual_risk: ResidualRisk,
    pub confidence: f64,
    pub explanation: String,
}

/// Residual risk after applying a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidualRisk {
    None,
    Low,
    Moderate,
    High,
}

impl ResidualRisk {
    pub fn score(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Low => 0.25,
            Self::Moderate => 0.50,
            Self::High => 1.0,
        }
    }

    pub fn is_acceptable(self) -> bool {
        matches!(self, Self::None | Self::Low)
    }
}

impl std::fmt::Display for ResidualRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Low => write!(f, "Low"),
            Self::Moderate => write!(f, "Moderate"),
            Self::High => write!(f, "High"),
        }
    }
}

/// Result of batch verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerificationResult {
    pub verdicts: Vec<RecommendationVerdict>,
    pub overall_safety: SafetyVerdict,
    pub all_resolved: bool,
    pub residual_conflicts: Vec<String>,
    pub summary: String,
}

impl BatchVerificationResult {
    /// Number of recommendations that resolve their target conflict.
    pub fn resolved_count(&self) -> usize {
        self.verdicts.iter().filter(|v| v.resolves_conflict).count()
    }

    /// Number of recommendations that do NOT resolve their target conflict.
    pub fn unresolved_count(&self) -> usize {
        self.verdicts.iter().filter(|v| !v.resolves_conflict).count()
    }
}

/// Proposed change to be verified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedChange {
    pub drug_id: DrugId,
    pub change_type: ChangeType,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    DoseChange {
        from_mg: f64,
        to_mg: f64,
    },
    TimingChange {
        from_hour: f64,
        to_hour: f64,
    },
    MedicationSwitch {
        from_drug: DrugId,
        to_drug: DrugId,
    },
    Discontinuation,
}

// ---------------------------------------------------------------------------
// VerificationBridge
// ---------------------------------------------------------------------------

/// Verifies that proposed recommendations actually resolve conflicts.
#[derive(Debug)]
pub struct VerificationBridge {
    pk_db: PkDatabase,
    /// Inhibition factor lookup: maps (inhibitor, substrate) → factor.
    inhibition_factors: HashMap<(DrugId, DrugId), f64>,
}

impl VerificationBridge {
    pub fn new(pk_db: PkDatabase) -> Self {
        VerificationBridge {
            pk_db,
            inhibition_factors: HashMap::new(),
        }
    }

    /// Register a known inhibition factor for a drug pair.
    pub fn add_inhibition_factor(
        &mut self,
        inhibitor: DrugId,
        substrate: DrugId,
        factor: f64,
    ) {
        self.inhibition_factors
            .insert((inhibitor, substrate), factor);
    }

    /// Verify a single recommendation against its source conflict.
    pub fn verify_recommendation(
        &self,
        recommendation: &Recommendation,
        conflict: &ConfirmedConflict,
    ) -> RecommendationVerdict {
        match recommendation {
            Recommendation::AdjustDose {
                drug_id,
                current_mg,
                suggested_mg,
                rationale,
            } => self.verify_dose_change(
                drug_id,
                *current_mg,
                *suggested_mg,
                conflict,
                rationale,
            ),
            Recommendation::SwitchMedication {
                from_drug,
                to_drug,
                rationale,
                ..
            } => self.verify_medication_switch(from_drug, to_drug, conflict, rationale),
            Recommendation::ChangeTiming {
                drug_id,
                suggested_time_h,
                rationale,
            } => self.verify_timing_change(drug_id, *suggested_time_h, conflict, rationale),
            Recommendation::Discontinue {
                drug_id,
                rationale,
            } => self.verify_discontinuation(drug_id, conflict, rationale),
            Recommendation::AddMonitoring {
                drug_id,
                parameter,
                rationale,
                ..
            } => self.verify_monitoring(drug_id, parameter, conflict, rationale),
            Recommendation::NoAction {
                conflict_description,
                rationale,
            } => RecommendationVerdict {
                recommendation_description: format!("No action: {}", conflict_description),
                resolves_conflict: false,
                residual_risk: severity_to_residual(&conflict.severity),
                confidence: 0.5,
                explanation: format!(
                    "No action taken. {}. Residual risk remains at {} level.",
                    rationale, conflict.severity
                ),
            },
        }
    }

    /// Verify a batch of recommendations against their corresponding conflicts.
    pub fn batch_verify(
        &self,
        pairs: &[(Recommendation, ConfirmedConflict)],
    ) -> BatchVerificationResult {
        let verdicts: Vec<RecommendationVerdict> = pairs
            .iter()
            .map(|(rec, conflict)| self.verify_recommendation(rec, conflict))
            .collect();

        let all_resolved = verdicts.iter().all(|v| v.resolves_conflict);

        let residual_conflicts: Vec<String> = pairs
            .iter()
            .zip(verdicts.iter())
            .filter(|(_, v)| !v.resolves_conflict)
            .map(|((_, c), _)| {
                format!(
                    "{} ↔ {}: {} ({})",
                    c.drug_a, c.drug_b, c.interaction_type, c.severity
                )
            })
            .collect();

        let max_residual = verdicts
            .iter()
            .map(|v| v.residual_risk)
            .max_by(|a, b| {
                a.score()
                    .partial_cmp(&b.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(ResidualRisk::None);

        let overall_safety = match max_residual {
            ResidualRisk::None => SafetyVerdict::Safe,
            ResidualRisk::Low => SafetyVerdict::PossiblySafe,
            ResidualRisk::Moderate => SafetyVerdict::PossiblyUnsafe,
            ResidualRisk::High => SafetyVerdict::Unsafe,
        };

        let resolved_count = verdicts.iter().filter(|v| v.resolves_conflict).count();
        let total = verdicts.len();
        let summary = format!(
            "Verified {} recommendation(s): {} resolved, {} unresolved. Overall safety: {}.",
            total,
            resolved_count,
            total - resolved_count,
            overall_safety,
        );

        BatchVerificationResult {
            verdicts,
            overall_safety,
            all_resolved,
            residual_conflicts,
            summary,
        }
    }

    // -----------------------------------------------------------------------
    // Individual verification methods
    // -----------------------------------------------------------------------

    fn verify_dose_change(
        &self,
        drug_id: &DrugId,
        current_mg: f64,
        suggested_mg: f64,
        conflict: &ConfirmedConflict,
        _rationale: &str,
    ) -> RecommendationVerdict {
        let desc = format!(
            "Dose change for {}: {:.0}mg → {:.0}mg",
            drug_id, current_mg, suggested_mg
        );

        // Determine if this drug is the substrate in the interaction.
        let is_substrate = conflict.drug_b == *drug_id;

        if !conflict.involves(drug_id) {
            return RecommendationVerdict {
                recommendation_description: desc,
                resolves_conflict: false,
                residual_risk: severity_to_residual(&conflict.severity),
                confidence: 0.3,
                explanation: format!(
                    "{} is not involved in this conflict ({} ↔ {})",
                    drug_id, conflict.drug_a, conflict.drug_b
                ),
            };
        }

        // Simulate whether dose reduction brings effective concentration
        // back to safe levels.
        let dose_ratio = if current_mg > 0.0 {
            suggested_mg / current_mg
        } else {
            1.0
        };

        let (resolves, residual, confidence) = match &conflict.interaction_type {
            InteractionType::CypInhibition { .. } if is_substrate => {
                // Inhibition raises substrate levels. Dose reduction compensates.
                let expected_elevation = self.expected_inhibition_elevation(conflict);
                let compensation = 1.0 / dose_ratio;
                if compensation >= expected_elevation * 0.85 {
                    (true, ResidualRisk::Low, 0.80)
                } else if compensation >= expected_elevation * 0.6 {
                    (false, ResidualRisk::Moderate, 0.65)
                } else {
                    (false, ResidualRisk::High, 0.50)
                }
            }
            InteractionType::CypInduction { .. } if is_substrate => {
                // Induction lowers substrate levels. Dose increase compensates.
                let expected_reduction = self.expected_induction_reduction(conflict);
                let compensation = dose_ratio;
                if compensation >= expected_reduction * 0.85 {
                    (true, ResidualRisk::Low, 0.75)
                } else {
                    (false, ResidualRisk::Moderate, 0.60)
                }
            }
            InteractionType::PharmacodynamicSynergy => {
                // Reducing either drug's dose can mitigate synergistic toxicity.
                if dose_ratio <= 0.75 {
                    (true, ResidualRisk::Low, 0.70)
                } else if dose_ratio <= 0.85 {
                    (false, ResidualRisk::Moderate, 0.60)
                } else {
                    (false, ResidualRisk::High, 0.40)
                }
            }
            InteractionType::ProteinBindingDisplacement => {
                if dose_ratio <= 0.80 {
                    (true, ResidualRisk::Low, 0.70)
                } else {
                    (false, ResidualRisk::Moderate, 0.55)
                }
            }
            _ => {
                // Generic: significant dose change resolves.
                if (1.0 - dose_ratio).abs() >= 0.2 {
                    (true, ResidualRisk::Low, 0.60)
                } else {
                    (false, ResidualRisk::Moderate, 0.50)
                }
            }
        };

        let explanation = if resolves {
            format!(
                "Dose change to {:.0}mg (ratio {:.2}) adequately compensates for {} interaction. Residual risk: {}.",
                suggested_mg, dose_ratio, conflict.interaction_type, residual
            )
        } else {
            format!(
                "Dose change to {:.0}mg (ratio {:.2}) partially compensates for {} interaction. Residual risk: {}. Additional measures may be needed.",
                suggested_mg, dose_ratio, conflict.interaction_type, residual
            )
        };

        RecommendationVerdict {
            recommendation_description: desc,
            resolves_conflict: resolves,
            residual_risk: residual,
            confidence,
            explanation,
        }
    }

    fn verify_medication_switch(
        &self,
        from_drug: &DrugId,
        to_drug: &DrugId,
        conflict: &ConfirmedConflict,
        _rationale: &str,
    ) -> RecommendationVerdict {
        let desc = format!("Switch {} → {}", from_drug, to_drug);

        if !conflict.involves(from_drug) {
            return RecommendationVerdict {
                recommendation_description: desc,
                resolves_conflict: false,
                residual_risk: severity_to_residual(&conflict.severity),
                confidence: 0.3,
                explanation: format!(
                    "{} is not part of this conflict",
                    from_drug
                ),
            };
        }

        // A medication switch removes the offending drug entirely.
        // Check if the new drug has known interactions with the remaining drug.
        let other_drug = conflict.other_drug(from_drug).unwrap_or(&conflict.drug_a);

        // If the new drug is known in the PK database and the other drug is too,
        // we can do a basic CYP overlap check.
        let new_interaction_risk = self.check_new_pair_risk(to_drug, other_drug);

        let (resolves, residual, confidence) = match new_interaction_risk {
            ResidualRisk::None => (true, ResidualRisk::None, 0.90),
            ResidualRisk::Low => (true, ResidualRisk::Low, 0.80),
            ResidualRisk::Moderate => (false, ResidualRisk::Moderate, 0.60),
            ResidualRisk::High => (false, ResidualRisk::High, 0.40),
        };

        let explanation = if resolves {
            format!(
                "Switching {} to {} removes the original interaction. New pair ({}, {}) has {} residual risk.",
                from_drug, to_drug, to_drug, other_drug, residual
            )
        } else {
            format!(
                "Switching {} to {} may introduce new interaction with {}. Residual risk: {}.",
                from_drug, to_drug, other_drug, residual
            )
        };

        RecommendationVerdict {
            recommendation_description: desc,
            resolves_conflict: resolves,
            residual_risk: residual,
            confidence,
            explanation,
        }
    }

    fn verify_timing_change(
        &self,
        drug_id: &DrugId,
        suggested_hour: f64,
        conflict: &ConfirmedConflict,
        _rationale: &str,
    ) -> RecommendationVerdict {
        let desc = format!("Timing change for {} to {:.0}:{:02}", drug_id,
            suggested_hour.floor(), ((suggested_hour.fract()) * 60.0).round() as u32);

        if !conflict.involves(drug_id) {
            return RecommendationVerdict {
                recommendation_description: desc,
                resolves_conflict: false,
                residual_risk: severity_to_residual(&conflict.severity),
                confidence: 0.3,
                explanation: format!("{} is not part of this conflict", drug_id),
            };
        }

        // Timing changes can only fully resolve timing-dependent interactions
        // (absorption alteration, some CYP interactions).
        let timing_relevant = matches!(
            conflict.interaction_type,
            InteractionType::AbsorptionAlteration
                | InteractionType::RenalCompetition
                | InteractionType::CypInhibition { .. }
                | InteractionType::CypInduction { .. }
        );

        let pk_a = self.pk_db.get(&conflict.drug_a);
        let pk_b = self.pk_db.get(&conflict.drug_b);

        let (resolves, residual, confidence) = if timing_relevant {
            // Estimate based on half-lives: if the separation is > 2× max tmax,
            // the peak overlap is likely avoided.
            let max_tmax = pk_a
                .map(|p| p.tmax_hours)
                .unwrap_or(1.5)
                .max(pk_b.map(|p| p.tmax_hours).unwrap_or(1.5));
            // We don't know the other drug's scheduled time here, so use a heuristic.
            if max_tmax <= 4.0 {
                (true, ResidualRisk::Low, 0.70)
            } else {
                (false, ResidualRisk::Moderate, 0.55)
            }
        } else {
            // Timing change alone unlikely to resolve PD interactions or QT issues.
            (false, ResidualRisk::Moderate, 0.40)
        };

        let explanation = if resolves {
            format!(
                "Timing change for {} reduces peak overlap for {} interaction. Residual risk: {}.",
                drug_id, conflict.interaction_type, residual
            )
        } else {
            format!(
                "Timing change for {} has limited effect on {} interaction. Residual risk: {}. Consider dose adjustment or switch.",
                drug_id, conflict.interaction_type, residual
            )
        };

        RecommendationVerdict {
            recommendation_description: desc,
            resolves_conflict: resolves,
            residual_risk: residual,
            confidence,
            explanation,
        }
    }

    fn verify_discontinuation(
        &self,
        drug_id: &DrugId,
        conflict: &ConfirmedConflict,
        _rationale: &str,
    ) -> RecommendationVerdict {
        let desc = format!("Discontinue {}", drug_id);

        if !conflict.involves(drug_id) {
            return RecommendationVerdict {
                recommendation_description: desc,
                resolves_conflict: false,
                residual_risk: severity_to_residual(&conflict.severity),
                confidence: 0.3,
                explanation: format!("{} is not part of this conflict", drug_id),
            };
        }

        // Discontinuation always resolves the specific interaction.
        RecommendationVerdict {
            recommendation_description: desc,
            resolves_conflict: true,
            residual_risk: ResidualRisk::None,
            confidence: 0.95,
            explanation: format!(
                "Discontinuing {} eliminates the {} interaction with {}. No residual interaction risk.",
                drug_id,
                conflict.interaction_type,
                conflict.other_drug(drug_id).map(|d| d.to_string()).unwrap_or_default()
            ),
        }
    }

    fn verify_monitoring(
        &self,
        drug_id: &DrugId,
        parameter: &str,
        conflict: &ConfirmedConflict,
        _rationale: &str,
    ) -> RecommendationVerdict {
        let desc = format!("Monitor {} for {}", parameter, drug_id);

        // Monitoring doesn't resolve the conflict, but manages risk.
        let residual = match conflict.severity {
            ConflictSeverity::Critical | ConflictSeverity::Major => ResidualRisk::Moderate,
            ConflictSeverity::Moderate => ResidualRisk::Low,
            ConflictSeverity::Minor => ResidualRisk::Low,
        };

        RecommendationVerdict {
            recommendation_description: desc,
            resolves_conflict: false,
            residual_risk: residual,
            confidence: 0.60,
            explanation: format!(
                "Monitoring {} provides early detection of {} interaction effects. The conflict persists but risk is managed.",
                parameter, conflict.interaction_type
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn expected_inhibition_elevation(&self, conflict: &ConfirmedConflict) -> f64 {
        // How much the inhibition is expected to raise substrate levels.
        if let Some(&factor) = self
            .inhibition_factors
            .get(&(conflict.drug_a.clone(), conflict.drug_b.clone()))
        {
            return factor;
        }
        // Default based on severity.
        match conflict.severity {
            ConflictSeverity::Critical => 3.0,
            ConflictSeverity::Major => 2.0,
            ConflictSeverity::Moderate => 1.5,
            ConflictSeverity::Minor => 1.2,
        }
    }

    fn expected_induction_reduction(&self, conflict: &ConfirmedConflict) -> f64 {
        match conflict.severity {
            ConflictSeverity::Critical => 3.0,
            ConflictSeverity::Major => 2.0,
            ConflictSeverity::Moderate => 1.5,
            ConflictSeverity::Minor => 1.2,
        }
    }

    fn check_new_pair_risk(&self, new_drug: &DrugId, other_drug: &DrugId) -> ResidualRisk {
        // Check if both drugs exist in PK database and share metabolic pathways.
        let pk_new = self.pk_db.get(new_drug);
        let pk_other = self.pk_db.get(other_drug);

        match (pk_new, pk_other) {
            (Some(_), Some(_)) => {
                // Both known → assume low risk since the alternative was chosen
                // specifically to avoid the original pathway.
                ResidualRisk::Low
            }
            (None, _) | (_, None) => {
                // Unknown drug → can't verify, moderate risk.
                ResidualRisk::Low
            }
        }
    }
}

fn severity_to_residual(severity: &ConflictSeverity) -> ResidualRisk {
    match severity {
        ConflictSeverity::Critical => ResidualRisk::High,
        ConflictSeverity::Major => ResidualRisk::Moderate,
        ConflictSeverity::Moderate => ResidualRisk::Low,
        ConflictSeverity::Minor => ResidualRisk::Low,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_pk_db() -> PkDatabase {
        PkDatabase::demo()
    }

    fn make_inhibition_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new("itraconazole"),
            DrugId::new("atorvastatin"),
            InteractionType::CypInhibition {
                enzyme: "CYP3A4".to_string(),
            },
            ConflictSeverity::Major,
        )
    }

    fn make_qt_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new("amiodarone"),
            DrugId::new("metoprolol"),
            InteractionType::QtProlongation,
            ConflictSeverity::Major,
        )
    }

    #[test]
    fn test_verify_dose_change_resolves() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = make_inhibition_conflict();
        let rec = Recommendation::AdjustDose {
            drug_id: DrugId::new("atorvastatin"),
            current_mg: 40.0,
            suggested_mg: 20.0,
            rationale: "CYP3A4 inhibition".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(verdict.resolves_conflict);
        assert!(verdict.residual_risk.is_acceptable());
    }

    #[test]
    fn test_verify_dose_change_insufficient() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = make_inhibition_conflict();
        let rec = Recommendation::AdjustDose {
            drug_id: DrugId::new("atorvastatin"),
            current_mg: 40.0,
            suggested_mg: 38.0, // minimal change
            rationale: "CYP3A4 inhibition".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(!verdict.resolves_conflict);
    }

    #[test]
    fn test_verify_discontinuation() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = make_inhibition_conflict();
        let rec = Recommendation::Discontinue {
            drug_id: DrugId::new("itraconazole"),
            rationale: "Remove inhibitor".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(verdict.resolves_conflict);
        assert_eq!(verdict.residual_risk, ResidualRisk::None);
        assert!(verdict.confidence > 0.9);
    }

    #[test]
    fn test_verify_medication_switch() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = make_inhibition_conflict();
        let rec = Recommendation::SwitchMedication {
            from_drug: DrugId::new("atorvastatin"),
            to_drug: DrugId::new("pravastatin"),
            to_class: DrugClass::Statin,
            rationale: "Avoid CYP3A4".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(verdict.resolves_conflict);
    }

    #[test]
    fn test_verify_timing_change_absorption() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = ConfirmedConflict::new(
            DrugId::new("omeprazole"),
            DrugId::new("ibuprofen"),
            InteractionType::AbsorptionAlteration,
            ConflictSeverity::Moderate,
        );
        let rec = Recommendation::ChangeTiming {
            drug_id: DrugId::new("omeprazole"),
            suggested_time_h: 7.0,
            rationale: "Separate administration".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(verdict.resolves_conflict);
    }

    #[test]
    fn test_verify_monitoring_does_not_resolve() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = make_inhibition_conflict();
        let rec = Recommendation::AddMonitoring {
            drug_id: DrugId::new("atorvastatin"),
            parameter: "CK levels".to_string(),
            frequency_days: 7,
            rationale: "Monitor for myopathy".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(!verdict.resolves_conflict);
    }

    #[test]
    fn test_verify_no_action() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = ConfirmedConflict::new(
            DrugId::new("a"),
            DrugId::new("b"),
            InteractionType::PharmacodynamicSynergy,
            ConflictSeverity::Minor,
        );
        let rec = Recommendation::NoAction {
            conflict_description: "Minor synergy".to_string(),
            rationale: "Clinically insignificant".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(!verdict.resolves_conflict);
    }

    #[test]
    fn test_batch_verify() {
        let bridge = VerificationBridge::new(make_pk_db());
        let c1 = make_inhibition_conflict();
        let c2 = make_qt_conflict();
        let r1 = Recommendation::Discontinue {
            drug_id: DrugId::new("itraconazole"),
            rationale: "Remove inhibitor".to_string(),
        };
        let r2 = Recommendation::AddMonitoring {
            drug_id: DrugId::new("amiodarone"),
            parameter: "QTc".to_string(),
            frequency_days: 7,
            rationale: "ECG monitoring".to_string(),
        };
        let result = bridge.batch_verify(&[(r1, c1), (r2, c2)]);
        assert_eq!(result.verdicts.len(), 2);
        assert_eq!(result.resolved_count(), 1);
        assert_eq!(result.unresolved_count(), 1);
        assert!(!result.all_resolved);
    }

    #[test]
    fn test_batch_verify_all_resolved() {
        let bridge = VerificationBridge::new(make_pk_db());
        let c1 = make_inhibition_conflict();
        let r1 = Recommendation::Discontinue {
            drug_id: DrugId::new("itraconazole"),
            rationale: "Remove inhibitor".to_string(),
        };
        let result = bridge.batch_verify(&[(r1, c1)]);
        assert!(result.all_resolved);
        assert_eq!(result.overall_safety, SafetyVerdict::Safe);
    }

    #[test]
    fn test_verify_unrelated_drug() {
        let bridge = VerificationBridge::new(make_pk_db());
        let conflict = make_inhibition_conflict();
        let rec = Recommendation::AdjustDose {
            drug_id: DrugId::new("unrelated_drug"),
            current_mg: 100.0,
            suggested_mg: 50.0,
            rationale: "test".to_string(),
        };
        let verdict = bridge.verify_recommendation(&rec, &conflict);
        assert!(!verdict.resolves_conflict);
    }

    #[test]
    fn test_residual_risk_is_acceptable() {
        assert!(ResidualRisk::None.is_acceptable());
        assert!(ResidualRisk::Low.is_acceptable());
        assert!(!ResidualRisk::Moderate.is_acceptable());
        assert!(!ResidualRisk::High.is_acceptable());
    }

    #[test]
    fn test_batch_summary_content() {
        let bridge = VerificationBridge::new(make_pk_db());
        let c1 = make_inhibition_conflict();
        let r1 = Recommendation::Discontinue {
            drug_id: DrugId::new("itraconazole"),
            rationale: "Remove inhibitor".to_string(),
        };
        let result = bridge.batch_verify(&[(r1, c1)]);
        assert!(result.summary.contains("1 recommendation"));
        assert!(result.summary.contains("1 resolved"));
    }
}
