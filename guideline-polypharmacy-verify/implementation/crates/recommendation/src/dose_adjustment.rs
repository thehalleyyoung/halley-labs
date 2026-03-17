//! Dose adjustment recommendations for polypharmacy conflicts.
//!
//! Provides logic to suggest dose modifications when drug interactions
//! alter effective concentrations via CYP inhibition/induction, protein
//! binding displacement, or organ-impairment considerations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    ActiveMedication, ConfirmedConflict, ConflictSeverity, DrugClass, DrugId,
    EvidenceLevel, HepaticFunction, InteractionType, PatientProfile, PkDatabase,
    PkProfile, RenalFunction,
};

// ---------------------------------------------------------------------------
// Monitoring
// ---------------------------------------------------------------------------

/// Monitoring parameter to track after dose adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringParameter {
    pub name: String,
    pub target_range: (f64, f64),
    pub unit: String,
    pub frequency_days: u32,
}

/// Plan for clinical monitoring after a dose adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringPlan {
    pub parameters: Vec<MonitoringParameter>,
    pub initial_check_days: u32,
    pub follow_up_interval_days: u32,
    pub duration_weeks: u32,
    pub notes: Vec<String>,
}

impl MonitoringPlan {
    pub fn empty() -> Self {
        MonitoringPlan {
            parameters: Vec::new(),
            initial_check_days: 7,
            follow_up_interval_days: 14,
            duration_weeks: 4,
            notes: Vec::new(),
        }
    }

    pub fn add_parameter(&mut self, name: &str, low: f64, high: f64, unit: &str, freq: u32) {
        self.parameters.push(MonitoringParameter {
            name: name.to_string(),
            target_range: (low, high),
            unit: unit.to_string(),
            frequency_days: freq,
        });
    }
}

// ---------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------

/// Constraint on an adjusted dose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustmentConstraint {
    pub min_dose_mg: f64,
    pub max_dose_mg: f64,
    pub available_strengths: Vec<f64>,
    pub max_daily_mg: Option<f64>,
}

impl AdjustmentConstraint {
    pub fn new(min: f64, max: f64) -> Self {
        AdjustmentConstraint {
            min_dose_mg: min,
            max_dose_mg: max,
            available_strengths: Vec::new(),
            max_daily_mg: None,
        }
    }

    pub fn with_strengths(mut self, strengths: Vec<f64>) -> Self {
        self.available_strengths = strengths;
        self
    }

    pub fn with_max_daily(mut self, max: f64) -> Self {
        self.max_daily_mg = Some(max);
        self
    }

    /// Clamp the dose to constraint bounds.
    pub fn clamp(&self, dose: f64) -> f64 {
        dose.clamp(self.min_dose_mg, self.max_dose_mg)
    }
}

/// Round a dose to the nearest available tablet strength.
pub fn round_to_available_strength(dose: f64, strengths: &[f64]) -> f64 {
    if strengths.is_empty() {
        return dose;
    }
    let mut best = strengths[0];
    let mut best_diff = (dose - best).abs();
    for &s in &strengths[1..] {
        let diff = (dose - s).abs();
        if diff < best_diff {
            best = s;
            best_diff = diff;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Renal / Hepatic adjustment helpers
// ---------------------------------------------------------------------------

/// Renal dose adjustment details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenalDoseAdjustment {
    pub renal_function: RenalFunction,
    pub original_dose_mg: f64,
    pub adjusted_dose_mg: f64,
    pub adjustment_factor: f64,
    pub rationale: String,
}

impl RenalDoseAdjustment {
    /// Compute renal adjustment given a drug's renal elimination fraction.
    pub fn compute(
        renal_fn: RenalFunction,
        dose_mg: f64,
        renal_elimination_fraction: f64,
    ) -> Self {
        // Only the renally-eliminated fraction is affected by impairment.
        let non_renal_fraction = 1.0 - renal_elimination_fraction;
        let renal_factor = renal_fn.dose_factor();
        let overall_factor = non_renal_fraction + renal_elimination_fraction * renal_factor;
        let adjusted = dose_mg * overall_factor;

        let rationale = format!(
            "Renal function {}: renal elimination fraction {:.0}%, dose factor {:.2} → overall factor {:.2}",
            renal_fn, renal_elimination_fraction * 100.0, renal_factor, overall_factor
        );

        RenalDoseAdjustment {
            renal_function: renal_fn,
            original_dose_mg: dose_mg,
            adjusted_dose_mg: adjusted,
            adjustment_factor: overall_factor,
            rationale,
        }
    }
}

/// Hepatic dose adjustment details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HepaticDoseAdjustment {
    pub hepatic_function: HepaticFunction,
    pub original_dose_mg: f64,
    pub adjusted_dose_mg: f64,
    pub adjustment_factor: f64,
    pub rationale: String,
}

impl HepaticDoseAdjustment {
    /// Compute hepatic adjustment given a drug's hepatic extraction ratio.
    pub fn compute(
        hepatic_fn: HepaticFunction,
        dose_mg: f64,
        hepatic_extraction_ratio: f64,
    ) -> Self {
        let non_hepatic_fraction = 1.0 - hepatic_extraction_ratio;
        let hepatic_factor = hepatic_fn.dose_factor();
        let overall_factor = non_hepatic_fraction + hepatic_extraction_ratio * hepatic_factor;
        let adjusted = dose_mg * overall_factor;

        let rationale = format!(
            "Hepatic function {} (Child-Pugh {}): extraction ratio {:.0}%, factor {:.2} → overall {:.2}",
            hepatic_fn, hepatic_fn.child_pugh_class(),
            hepatic_extraction_ratio * 100.0, hepatic_factor, overall_factor
        );

        HepaticDoseAdjustment {
            hepatic_function: hepatic_fn,
            original_dose_mg: dose_mg,
            adjusted_dose_mg: adjusted,
            adjustment_factor: overall_factor,
            rationale,
        }
    }
}

// ---------------------------------------------------------------------------
// DoseAdjustment result
// ---------------------------------------------------------------------------

/// A recommended dose adjustment for a single drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseAdjustment {
    pub drug_id: DrugId,
    pub current_dose_mg: f64,
    pub suggested_dose_mg: f64,
    pub doses_per_day: u32,
    pub rationale: String,
    pub evidence_level: EvidenceLevel,
    pub monitoring_plan: MonitoringPlan,
    pub renal_adjustment: Option<RenalDoseAdjustment>,
    pub hepatic_adjustment: Option<HepaticDoseAdjustment>,
    pub interaction_factor: f64,
    pub confidence: f64,
}

impl DoseAdjustment {
    /// Percentage change from current to suggested dose.
    pub fn percent_change(&self) -> f64 {
        if self.current_dose_mg == 0.0 {
            return 0.0;
        }
        ((self.suggested_dose_mg - self.current_dose_mg) / self.current_dose_mg) * 100.0
    }

    /// Whether this adjustment represents a dose reduction.
    pub fn is_reduction(&self) -> bool {
        self.suggested_dose_mg < self.current_dose_mg
    }

    /// Whether this adjustment represents a dose increase.
    pub fn is_increase(&self) -> bool {
        self.suggested_dose_mg > self.current_dose_mg
    }

    /// Suggested daily dose.
    pub fn suggested_daily_dose(&self) -> f64 {
        self.suggested_dose_mg * self.doses_per_day as f64
    }
}

// ---------------------------------------------------------------------------
// DoseAdjuster
// ---------------------------------------------------------------------------

/// Main dose adjustment engine.
#[derive(Debug, Clone)]
pub struct DoseAdjuster {
    pk_db: PkDatabase,
    constraints: HashMap<DrugId, AdjustmentConstraint>,
    /// Default available strengths for common drugs.
    strength_catalog: HashMap<DrugId, Vec<f64>>,
}

impl DoseAdjuster {
    pub fn new(pk_db: PkDatabase) -> Self {
        let strength_catalog = Self::build_default_strengths();
        DoseAdjuster {
            pk_db,
            constraints: HashMap::new(),
            strength_catalog,
        }
    }

    /// Register an adjustment constraint for a specific drug.
    pub fn add_constraint(&mut self, drug_id: DrugId, constraint: AdjustmentConstraint) {
        self.constraints.insert(drug_id, constraint);
    }

    /// Suggest dose adjustments for all drugs involved in the conflicts,
    /// taking into account patient organ function.
    pub fn suggest_adjustments(
        &self,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> Vec<DoseAdjustment> {
        let mut adjustments: Vec<DoseAdjustment> = Vec::new();
        let mut processed: HashMap<DrugId, bool> = HashMap::new();

        for conflict in conflicts {
            for drug_id in [&conflict.drug_a, &conflict.drug_b] {
                if processed.contains_key(drug_id) {
                    continue;
                }
                processed.insert(drug_id.clone(), true);

                let med = match patient.find_medication(drug_id) {
                    Some(m) => m,
                    None => continue,
                };

                let pk = self.pk_db.get(drug_id);

                // Compute interaction-based factor.
                let interaction_factor =
                    self.compute_interaction_factor(drug_id, conflicts);

                // Compute organ-function adjustments.
                let renal_adj = pk.map(|p| {
                    RenalDoseAdjustment::compute(
                        patient.renal_function,
                        med.dose_schedule.dose_amount_mg,
                        p.renal_elimination_fraction,
                    )
                });

                let hepatic_adj = pk.map(|p| {
                    HepaticDoseAdjustment::compute(
                        patient.hepatic_function,
                        med.dose_schedule.dose_amount_mg,
                        p.hepatic_extraction_ratio,
                    )
                });

                // Combine factors.
                let renal_factor = renal_adj
                    .as_ref()
                    .map(|r| r.adjustment_factor)
                    .unwrap_or(1.0);
                let hepatic_factor = hepatic_adj
                    .as_ref()
                    .map(|h| h.adjustment_factor)
                    .unwrap_or(1.0);

                let combined_factor = interaction_factor * renal_factor * hepatic_factor;
                let raw_dose = med.dose_schedule.dose_amount_mg * combined_factor;

                // Apply constraints and round to available strength.
                let constrained = if let Some(c) = self.constraints.get(drug_id) {
                    let clamped = c.clamp(raw_dose);
                    if !c.available_strengths.is_empty() {
                        round_to_available_strength(clamped, &c.available_strengths)
                    } else {
                        self.round_with_catalog(drug_id, clamped)
                    }
                } else {
                    self.round_with_catalog(drug_id, raw_dose)
                };

                let suggested = if constrained > 0.0 { constrained } else { raw_dose.max(0.0) };

                // Build rationale.
                let rationale = self.build_rationale(
                    drug_id,
                    med.dose_schedule.dose_amount_mg,
                    suggested,
                    interaction_factor,
                    &renal_adj,
                    &hepatic_adj,
                    conflicts,
                );

                let evidence = self.evidence_for_conflicts(drug_id, conflicts);
                let monitoring = self.generate_monitoring_plan(drug_id, conflicts, patient);
                let confidence = self.compute_confidence(conflicts, pk.is_some());

                adjustments.push(DoseAdjustment {
                    drug_id: drug_id.clone(),
                    current_dose_mg: med.dose_schedule.dose_amount_mg,
                    suggested_dose_mg: suggested,
                    doses_per_day: med.dose_schedule.doses_per_day,
                    rationale,
                    evidence_level: evidence,
                    monitoring_plan: monitoring,
                    renal_adjustment: renal_adj,
                    hepatic_adjustment: hepatic_adj,
                    interaction_factor,
                    confidence,
                });
            }
        }

        adjustments
    }

    /// Compute the net interaction factor for a drug given all conflicts it appears in.
    fn compute_interaction_factor(
        &self,
        drug_id: &DrugId,
        conflicts: &[ConfirmedConflict],
    ) -> f64 {
        let mut factor = 1.0;
        for conflict in conflicts {
            if !conflict.involves(drug_id) {
                continue;
            }
            match &conflict.interaction_type {
                InteractionType::CypInhibition { .. } => {
                    let inhibition_factor = self.adjustment_for_inhibition(
                        &conflict.severity,
                        drug_id,
                        conflict,
                    );
                    factor *= inhibition_factor;
                }
                InteractionType::CypInduction { .. } => {
                    let induction_factor = self.adjustment_for_induction(&conflict.severity);
                    factor *= induction_factor;
                }
                InteractionType::ProteinBindingDisplacement => {
                    factor *= self.adjustment_for_displacement(&conflict.severity);
                }
                InteractionType::PharmacodynamicSynergy => {
                    factor *= self.adjustment_for_pd_synergy(&conflict.severity);
                }
                _ => {
                    // For other interaction types, apply a generic severity-based factor.
                    factor *= self.generic_severity_factor(&conflict.severity);
                }
            }
        }
        factor.clamp(0.1, 3.0)
    }

    /// Dose factor for CYP inhibition: the substrate drug needs LESS dose
    /// because metabolism is inhibited → higher plasma levels.
    pub fn adjustment_for_inhibition(
        &self,
        severity: &ConflictSeverity,
        drug_id: &DrugId,
        conflict: &ConfirmedConflict,
    ) -> f64 {
        // The drug that is the substrate (being metabolised) needs dose reduction.
        // The inhibitor drug generally does NOT need dose change from this interaction.
        // Heuristic: if drug_id is drug_b (typically the affected drug), reduce more.
        let is_substrate = conflict.drug_b == *drug_id;

        if !is_substrate {
            return 1.0; // Inhibitor drug: no change from this mechanism.
        }

        match severity {
            ConflictSeverity::Critical => 0.40,
            ConflictSeverity::Major => 0.50,
            ConflictSeverity::Moderate => 0.65,
            ConflictSeverity::Minor => 0.80,
        }
    }

    /// Dose factor for CYP induction: substrate drug needs MORE dose
    /// because metabolism is induced → lower plasma levels.
    fn adjustment_for_induction(&self, severity: &ConflictSeverity) -> f64 {
        match severity {
            ConflictSeverity::Critical => 2.0,
            ConflictSeverity::Major => 1.75,
            ConflictSeverity::Moderate => 1.50,
            ConflictSeverity::Minor => 1.25,
        }
    }

    /// Dose factor for protein binding displacement: free fraction increases,
    /// so effective dose is higher → reduce dose.
    fn adjustment_for_displacement(&self, severity: &ConflictSeverity) -> f64 {
        match severity {
            ConflictSeverity::Critical => 0.50,
            ConflictSeverity::Major => 0.65,
            ConflictSeverity::Moderate => 0.80,
            ConflictSeverity::Minor => 0.90,
        }
    }

    /// Dose factor for pharmacodynamic synergy: additive/synergistic effects
    /// mean lower doses of each drug may suffice.
    fn adjustment_for_pd_synergy(&self, severity: &ConflictSeverity) -> f64 {
        match severity {
            ConflictSeverity::Critical => 0.50,
            ConflictSeverity::Major => 0.60,
            ConflictSeverity::Moderate => 0.75,
            ConflictSeverity::Minor => 0.85,
        }
    }

    /// Generic severity-based factor for interaction types without specific PK logic.
    fn generic_severity_factor(&self, severity: &ConflictSeverity) -> f64 {
        match severity {
            ConflictSeverity::Critical => 0.50,
            ConflictSeverity::Major => 0.65,
            ConflictSeverity::Moderate => 0.80,
            ConflictSeverity::Minor => 0.90,
        }
    }

    /// Generate a monitoring plan for a drug given its conflicts.
    pub fn generate_monitoring_plan(
        &self,
        drug_id: &DrugId,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> MonitoringPlan {
        let mut plan = MonitoringPlan::empty();

        let max_severity = conflicts
            .iter()
            .filter(|c| c.involves(drug_id))
            .map(|c| c.severity)
            .max()
            .unwrap_or(ConflictSeverity::Minor);

        // Set check intervals based on severity.
        match max_severity {
            ConflictSeverity::Critical => {
                plan.initial_check_days = 2;
                plan.follow_up_interval_days = 7;
                plan.duration_weeks = 12;
            }
            ConflictSeverity::Major => {
                plan.initial_check_days = 3;
                plan.follow_up_interval_days = 7;
                plan.duration_weeks = 8;
            }
            ConflictSeverity::Moderate => {
                plan.initial_check_days = 7;
                plan.follow_up_interval_days = 14;
                plan.duration_weeks = 6;
            }
            ConflictSeverity::Minor => {
                plan.initial_check_days = 14;
                plan.follow_up_interval_days = 28;
                plan.duration_weeks = 4;
            }
        }

        // Add monitoring parameters based on interaction type.
        for conflict in conflicts.iter().filter(|c| c.involves(drug_id)) {
            match &conflict.interaction_type {
                InteractionType::CypInhibition { enzyme } | InteractionType::CypInduction { enzyme } => {
                    plan.add_parameter(
                        &format!("{} drug level", drug_id),
                        0.0,
                        100.0,
                        "ng/mL",
                        plan.follow_up_interval_days,
                    );
                    plan.notes.push(format!(
                        "Monitor {} substrate levels due to {} interaction",
                        enzyme,
                        if matches!(conflict.interaction_type, InteractionType::CypInhibition { .. }) {
                            "inhibition"
                        } else {
                            "induction"
                        }
                    ));
                }
                InteractionType::QtProlongation => {
                    plan.add_parameter("QTc interval", 350.0, 470.0, "ms", 7);
                    plan.notes.push("ECG monitoring for QTc prolongation".to_string());
                }
                InteractionType::RenalCompetition => {
                    plan.add_parameter("Serum creatinine", 0.6, 1.2, "mg/dL", 7);
                    plan.add_parameter("eGFR", 60.0, 150.0, "mL/min/1.73m²", 14);
                    plan.notes.push("Monitor renal function closely".to_string());
                }
                InteractionType::ProteinBindingDisplacement => {
                    plan.add_parameter(
                        &format!("Free {} level", drug_id),
                        0.0,
                        50.0,
                        "ng/mL",
                        plan.follow_up_interval_days,
                    );
                    plan.notes.push("Monitor free (unbound) drug levels".to_string());
                }
                _ => {
                    plan.notes.push(format!(
                        "Monitor clinical response for {} interaction",
                        conflict.interaction_type
                    ));
                }
            }
        }

        // Organ function monitoring.
        if patient.renal_function.requires_dose_adjustment() {
            plan.add_parameter("eGFR", 30.0, 150.0, "mL/min/1.73m²", 14);
        }
        if patient.hepatic_function.requires_dose_adjustment() {
            plan.add_parameter("ALT", 7.0, 56.0, "U/L", 14);
            plan.add_parameter("AST", 10.0, 40.0, "U/L", 14);
        }

        plan
    }

    fn build_rationale(
        &self,
        drug_id: &DrugId,
        current_dose: f64,
        suggested_dose: f64,
        interaction_factor: f64,
        renal_adj: &Option<RenalDoseAdjustment>,
        hepatic_adj: &Option<HepaticDoseAdjustment>,
        conflicts: &[ConfirmedConflict],
    ) -> String {
        let mut parts = Vec::new();

        let pct = if current_dose > 0.0 {
            ((suggested_dose - current_dose) / current_dose * 100.0).round()
        } else {
            0.0
        };

        if (interaction_factor - 1.0).abs() > 0.01 {
            let interaction_descs: Vec<String> = conflicts
                .iter()
                .filter(|c| c.involves(drug_id))
                .map(|c| format!("{}", c.interaction_type))
                .collect();
            parts.push(format!(
                "Interaction adjustment factor {:.2} due to: {}",
                interaction_factor,
                interaction_descs.join("; ")
            ));
        }

        if let Some(ref r) = renal_adj {
            if (r.adjustment_factor - 1.0).abs() > 0.01 {
                parts.push(r.rationale.clone());
            }
        }

        if let Some(ref h) = hepatic_adj {
            if (h.adjustment_factor - 1.0).abs() > 0.01 {
                parts.push(h.rationale.clone());
            }
        }

        if pct.abs() > 0.5 {
            parts.push(format!(
                "Net change: {:.0}mg → {:.0}mg ({:+.0}%)",
                current_dose, suggested_dose, pct
            ));
        } else {
            parts.push("No dose change required".to_string());
        }

        parts.join(". ")
    }

    fn evidence_for_conflicts(
        &self,
        drug_id: &DrugId,
        conflicts: &[ConfirmedConflict],
    ) -> EvidenceLevel {
        let max_severity = conflicts
            .iter()
            .filter(|c| c.involves(drug_id))
            .map(|c| c.severity)
            .max();

        match max_severity {
            Some(ConflictSeverity::Critical) | Some(ConflictSeverity::Major) => EvidenceLevel::High,
            Some(ConflictSeverity::Moderate) => EvidenceLevel::Moderate,
            Some(ConflictSeverity::Minor) => EvidenceLevel::Low,
            None => EvidenceLevel::VeryLow,
        }
    }

    fn compute_confidence(&self, conflicts: &[ConfirmedConflict], has_pk: bool) -> f64 {
        let base = if has_pk { 0.80 } else { 0.60 };
        let avg_conf: f64 = if conflicts.is_empty() {
            0.5
        } else {
            conflicts.iter().map(|c| c.confidence).sum::<f64>() / conflicts.len() as f64
        };
        (base * 0.6 + avg_conf * 0.4).clamp(0.1, 1.0)
    }

    fn round_with_catalog(&self, drug_id: &DrugId, dose: f64) -> f64 {
        if let Some(strengths) = self.strength_catalog.get(drug_id) {
            round_to_available_strength(dose, strengths)
        } else {
            // Round to nearest 5mg for unknown drugs.
            (dose / 5.0).round() * 5.0
        }
    }

    fn build_default_strengths() -> HashMap<DrugId, Vec<f64>> {
        let mut m = HashMap::new();
        m.insert(DrugId::new("atorvastatin"), vec![10.0, 20.0, 40.0, 80.0]);
        m.insert(DrugId::new("rosuvastatin"), vec![5.0, 10.0, 20.0, 40.0]);
        m.insert(DrugId::new("simvastatin"), vec![5.0, 10.0, 20.0, 40.0, 80.0]);
        m.insert(DrugId::new("metoprolol"), vec![25.0, 50.0, 100.0, 200.0]);
        m.insert(DrugId::new("amlodipine"), vec![2.5, 5.0, 10.0]);
        m.insert(DrugId::new("lisinopril"), vec![2.5, 5.0, 10.0, 20.0, 40.0]);
        m.insert(DrugId::new("losartan"), vec![25.0, 50.0, 100.0]);
        m.insert(DrugId::new("omeprazole"), vec![10.0, 20.0, 40.0]);
        m.insert(DrugId::new("sertraline"), vec![25.0, 50.0, 100.0]);
        m.insert(DrugId::new("warfarin"), vec![1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]);
        m.insert(DrugId::new("ibuprofen"), vec![200.0, 400.0, 600.0, 800.0]);
        m
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
        p.add_medication(ActiveMedication::new(
            DrugId::new("atorvastatin"),
            DrugClass::Statin,
            40.0,
            1,
        ));
        p.add_medication(ActiveMedication::new(
            DrugId::new("metoprolol"),
            DrugClass::BetaBlocker,
            50.0,
            2,
        ));
        p
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

    fn make_induction_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            DrugId::new("rifampin"),
            DrugId::new("metoprolol"),
            InteractionType::CypInduction {
                enzyme: "CYP2D6".to_string(),
            },
            ConflictSeverity::Moderate,
        )
    }

    #[test]
    fn test_round_to_available_strength() {
        let strengths = vec![10.0, 20.0, 40.0, 80.0];
        assert!((round_to_available_strength(35.0, &strengths) - 40.0).abs() < 0.01);
        assert!((round_to_available_strength(12.0, &strengths) - 10.0).abs() < 0.01);
        assert!((round_to_available_strength(60.0, &strengths) - 40.0).abs() < 0.01);
        assert!((round_to_available_strength(75.0, &strengths) - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_round_empty_strengths() {
        assert!((round_to_available_strength(35.0, &[]) - 35.0).abs() < 0.01);
    }

    #[test]
    fn test_renal_dose_adjustment_normal() {
        let adj = RenalDoseAdjustment::compute(RenalFunction::Normal, 100.0, 0.5);
        assert!((adj.adjusted_dose_mg - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_renal_dose_adjustment_severe() {
        let adj = RenalDoseAdjustment::compute(RenalFunction::SevereImpairment, 100.0, 0.8);
        assert!(adj.adjusted_dose_mg < 60.0);
        assert!(adj.adjustment_factor < 1.0);
    }

    #[test]
    fn test_hepatic_dose_adjustment_moderate() {
        let adj = HepaticDoseAdjustment::compute(HepaticFunction::ModerateImpairment, 100.0, 0.7);
        assert!(adj.adjusted_dose_mg < 85.0);
    }

    #[test]
    fn test_dose_adjuster_inhibition() {
        let pk_db = PkDatabase::demo();
        let adjuster = DoseAdjuster::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("itraconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let conflict = make_inhibition_conflict();
        let adjustments = adjuster.suggest_adjustments(&[conflict], &patient);
        // Atorvastatin should be reduced (it's the substrate).
        let atorv = adjustments
            .iter()
            .find(|a| a.drug_id == DrugId::new("atorvastatin"))
            .expect("Should have adjustment for atorvastatin");
        assert!(atorv.suggested_dose_mg < 40.0);
        assert!(atorv.is_reduction());
    }

    #[test]
    fn test_dose_adjuster_induction() {
        let pk_db = PkDatabase::demo();
        let adjuster = DoseAdjuster::new(pk_db);
        let mut patient = make_patient();
        patient.add_medication(ActiveMedication::new(
            DrugId::new("rifampin"),
            DrugClass::Antibiotic,
            600.0,
            1,
        ));
        let conflict = make_induction_conflict();
        let adjustments = adjuster.suggest_adjustments(&[conflict], &patient);
        let meto = adjustments
            .iter()
            .find(|a| a.drug_id == DrugId::new("metoprolol"))
            .expect("Should have adjustment for metoprolol");
        // Metoprolol should be increased (CYP2D6 induction → faster clearance).
        assert!(meto.suggested_dose_mg >= 50.0);
    }

    #[test]
    fn test_dose_adjuster_with_renal_impairment() {
        let pk_db = PkDatabase::demo();
        let adjuster = DoseAdjuster::new(pk_db);
        let mut patient = make_patient();
        patient.renal_function = RenalFunction::SevereImpairment;
        patient.add_medication(ActiveMedication::new(
            DrugId::new("itraconazole"),
            DrugClass::Antifungal,
            200.0,
            1,
        ));
        let conflict = make_inhibition_conflict();
        let adjustments = adjuster.suggest_adjustments(&[conflict], &patient);
        let atorv = adjustments
            .iter()
            .find(|a| a.drug_id == DrugId::new("atorvastatin"));
        assert!(atorv.is_some());
    }

    #[test]
    fn test_monitoring_plan_qt() {
        let pk_db = PkDatabase::demo();
        let adjuster = DoseAdjuster::new(pk_db);
        let patient = make_patient();
        let conflict = ConfirmedConflict::new(
            DrugId::new("atorvastatin"),
            DrugId::new("metoprolol"),
            InteractionType::QtProlongation,
            ConflictSeverity::Major,
        );
        let plan = adjuster.generate_monitoring_plan(
            &DrugId::new("atorvastatin"),
            &[conflict],
            &patient,
        );
        assert!(!plan.parameters.is_empty());
        assert!(plan.notes.iter().any(|n| n.contains("QTc")));
    }

    #[test]
    fn test_monitoring_plan_severity_scaling() {
        let pk_db = PkDatabase::demo();
        let adjuster = DoseAdjuster::new(pk_db);
        let patient = make_patient();

        let minor_conflict = ConfirmedConflict::new(
            DrugId::new("atorvastatin"),
            DrugId::new("metoprolol"),
            InteractionType::PharmacodynamicSynergy,
            ConflictSeverity::Minor,
        );
        let critical_conflict = ConfirmedConflict::new(
            DrugId::new("atorvastatin"),
            DrugId::new("metoprolol"),
            InteractionType::PharmacodynamicSynergy,
            ConflictSeverity::Critical,
        );

        let plan_minor = adjuster.generate_monitoring_plan(
            &DrugId::new("atorvastatin"),
            &[minor_conflict],
            &patient,
        );
        let plan_critical = adjuster.generate_monitoring_plan(
            &DrugId::new("atorvastatin"),
            &[critical_conflict],
            &patient,
        );

        assert!(plan_critical.initial_check_days < plan_minor.initial_check_days);
        assert!(plan_critical.duration_weeks > plan_minor.duration_weeks);
    }

    #[test]
    fn test_adjustment_constraint_clamp() {
        let c = AdjustmentConstraint::new(10.0, 80.0);
        assert!((c.clamp(5.0) - 10.0).abs() < 0.01);
        assert!((c.clamp(50.0) - 50.0).abs() < 0.01);
        assert!((c.clamp(100.0) - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_dose_adjustment_percent_change() {
        let adj = DoseAdjustment {
            drug_id: DrugId::new("test"),
            current_dose_mg: 100.0,
            suggested_dose_mg: 50.0,
            doses_per_day: 1,
            rationale: String::new(),
            evidence_level: EvidenceLevel::Moderate,
            monitoring_plan: MonitoringPlan::empty(),
            renal_adjustment: None,
            hepatic_adjustment: None,
            interaction_factor: 0.5,
            confidence: 0.8,
        };
        assert!((adj.percent_change() - (-50.0)).abs() < 0.01);
        assert!(adj.is_reduction());
        assert!(!adj.is_increase());
    }

    #[test]
    fn test_no_adjustment_for_uninvolved_drug() {
        let pk_db = PkDatabase::demo();
        let adjuster = DoseAdjuster::new(pk_db);
        let patient = make_patient();
        let conflict = ConfirmedConflict::new(
            DrugId::new("drug_x"),
            DrugId::new("drug_y"),
            InteractionType::QtProlongation,
            ConflictSeverity::Major,
        );
        let adjustments = adjuster.suggest_adjustments(&[conflict], &patient);
        // Patient doesn't have drug_x or drug_y so no adjustments.
        assert!(adjustments.is_empty());
    }
}
