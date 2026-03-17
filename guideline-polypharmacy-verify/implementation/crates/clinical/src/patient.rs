//! Patient demographics, risk profiles, and builder.
//!
//! Provides a rich [`Patient`] model with demographic data, conditions,
//! medications, lab values, allergies, and computed risk profiles.
//! The [`PatientBuilder`] enables fluent construction and
//! [`compute_risk_profile`](Patient::compute_risk_profile) derives an
//! aggregate [`PatientRiskProfile`] from the patient's clinical data.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::{DrugId, PatientId};

// ═══════════════════════════════════════════════════════════════════════════
// Sex
// ═══════════════════════════════════════════════════════════════════════════

/// Biological sex for pharmacokinetic adjustments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
}

impl Default for Sex {
    fn default() -> Self {
        Sex::Male
    }
}

impl std::fmt::Display for Sex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sex::Male => write!(f, "Male"),
            Sex::Female => write!(f, "Female"),
        }
    }
}

impl std::str::FromStr for Sex {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "male" | "m" => Ok(Sex::Male),
            "female" | "f" => Ok(Sex::Female),
            other => Err(format!("unknown sex: {other}")),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Race
// ═══════════════════════════════════════════════════════════════════════════

/// Race/ethnicity for PK-relevant polymorphism adjustments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Race {
    White,
    Black,
    Asian,
    Hispanic,
    NativeAmerican,
    PacificIslander,
    Other,
}

impl Default for Race {
    fn default() -> Self {
        Race::Other
    }
}

impl std::fmt::Display for Race {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Race::White => "White",
            Race::Black => "Black or African American",
            Race::Asian => "Asian",
            Race::Hispanic => "Hispanic or Latino",
            Race::NativeAmerican => "American Indian or Alaska Native",
            Race::PacificIslander => "Native Hawaiian or Pacific Islander",
            Race::Other => "Other",
        };
        write!(f, "{s}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RenalStage / ChildPughClass
// ═══════════════════════════════════════════════════════════════════════════

/// Renal function classification derived from eGFR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenalStage {
    Normal,
    MildImpairment,
    ModerateImpairment,
    SevereImpairment,
    EndStage,
}

impl RenalStage {
    /// Classify from an eGFR value (mL/min/1.73 m²).
    pub fn from_egfr(egfr: f64) -> Self {
        if egfr >= 90.0 {
            RenalStage::Normal
        } else if egfr >= 60.0 {
            RenalStage::MildImpairment
        } else if egfr >= 30.0 {
            RenalStage::ModerateImpairment
        } else if egfr >= 15.0 {
            RenalStage::SevereImpairment
        } else {
            RenalStage::EndStage
        }
    }

    /// Whether dose adjustments are typically required at this stage.
    pub fn requires_dose_adjustment(&self) -> bool {
        matches!(
            self,
            RenalStage::ModerateImpairment
                | RenalStage::SevereImpairment
                | RenalStage::EndStage
        )
    }

    /// Descriptive label for the stage.
    pub fn label(&self) -> &'static str {
        match self {
            RenalStage::Normal => "Normal (G1)",
            RenalStage::MildImpairment => "Mild (G2)",
            RenalStage::ModerateImpairment => "Moderate (G3)",
            RenalStage::SevereImpairment => "Severe (G4)",
            RenalStage::EndStage => "End-Stage (G5)",
        }
    }
}

impl Default for RenalStage {
    fn default() -> Self {
        RenalStage::Normal
    }
}

/// Child-Pugh hepatic impairment class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChildPughClass {
    A,
    B,
    C,
}

impl ChildPughClass {
    /// Compute the Child-Pugh class from a numerical score (5–15).
    pub fn from_score(score: u32) -> Option<Self> {
        match score {
            5..=6 => Some(ChildPughClass::A),
            7..=9 => Some(ChildPughClass::B),
            10..=15 => Some(ChildPughClass::C),
            _ => None,
        }
    }

    /// Descriptive label.
    pub fn label(&self) -> &'static str {
        match self {
            ChildPughClass::A => "Child-Pugh A (Mild)",
            ChildPughClass::B => "Child-Pugh B (Moderate)",
            ChildPughClass::C => "Child-Pugh C (Severe)",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Allergy
// ═══════════════════════════════════════════════════════════════════════════

/// Severity of an allergic reaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllergySeverity {
    Mild,
    Moderate,
    Severe,
    Anaphylaxis,
}

/// A recorded drug or substance allergy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allergy {
    pub substance: String,
    pub reaction: String,
    pub severity: AllergySeverity,
    pub verified: bool,
}

impl Allergy {
    pub fn new(substance: &str, reaction: &str, severity: AllergySeverity) -> Self {
        Self {
            substance: substance.to_string(),
            reaction: reaction.to_string(),
            severity,
            verified: false,
        }
    }

    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Whether this allergy is potentially life-threatening.
    pub fn is_life_threatening(&self) -> bool {
        matches!(
            self.severity,
            AllergySeverity::Severe | AllergySeverity::Anaphylaxis
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Demographics
// ═══════════════════════════════════════════════════════════════════════════

/// Patient demographics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demographics {
    pub age_years: f64,
    pub sex: Sex,
    pub weight_kg: f64,
    pub height_cm: f64,
    pub race: Race,
    pub smoking_status: bool,
    pub alcohol_use: bool,
    pub serum_creatinine: f64,
}

impl Default for Demographics {
    fn default() -> Self {
        Self {
            age_years: 50.0,
            sex: Sex::Male,
            weight_kg: 70.0,
            height_cm: 170.0,
            race: Race::Other,
            smoking_status: false,
            alcohol_use: false,
            serum_creatinine: 1.0,
        }
    }
}

impl Demographics {
    /// Body mass index in kg/m².
    pub fn bmi(&self) -> f64 {
        let h_m = self.height_cm / 100.0;
        if h_m <= 0.0 {
            return 0.0;
        }
        self.weight_kg / (h_m * h_m)
    }

    /// Body surface area in m² (Mosteller formula).
    pub fn bsa(&self) -> f64 {
        if self.height_cm <= 0.0 || self.weight_kg <= 0.0 {
            return 0.0;
        }
        ((self.height_cm * self.weight_kg) / 3600.0).sqrt()
    }

    /// Ideal body weight in kg (Devine formula).
    pub fn ideal_body_weight(&self) -> f64 {
        let height_inches = self.height_cm / 2.54;
        let base = if self.sex == Sex::Male { 50.0 } else { 45.5 };
        if height_inches <= 60.0 {
            return base;
        }
        base + 2.3 * (height_inches - 60.0)
    }

    /// Adjusted body weight for obese patients (40% excess over IBW).
    pub fn adjusted_body_weight(&self) -> f64 {
        let ibw = self.ideal_body_weight();
        if self.weight_kg <= ibw {
            return self.weight_kg;
        }
        ibw + 0.4 * (self.weight_kg - ibw)
    }

    /// Lean body weight (Boer formula).
    pub fn lean_body_weight(&self) -> f64 {
        let h = self.height_cm;
        let w = self.weight_kg;
        match self.sex {
            Sex::Male => 0.407 * w + 0.267 * h - 19.2,
            Sex::Female => 0.252 * w + 0.473 * h - 48.3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PatientRiskProfile
// ═══════════════════════════════════════════════════════════════════════════

/// Aggregated risk profile for a patient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientRiskProfile {
    pub fall_risk: bool,
    pub bleeding_risk: f64,
    pub renal_stage: RenalStage,
    pub hepatic_class: Option<ChildPughClass>,
    pub polypharmacy_count: usize,
    pub anticholinergic_burden: f64,
    pub qtrisk: f64,
    pub overall_risk_score: f64,
    pub risk_factors: Vec<String>,
}

impl Default for PatientRiskProfile {
    fn default() -> Self {
        Self {
            fall_risk: false,
            bleeding_risk: 0.0,
            renal_stage: RenalStage::Normal,
            hepatic_class: None,
            polypharmacy_count: 0,
            anticholinergic_burden: 0.0,
            qtrisk: 0.0,
            overall_risk_score: 0.0,
            risk_factors: Vec::new(),
        }
    }
}

impl PatientRiskProfile {
    /// Whether the patient has elevated overall risk (score ≥ 0.5).
    pub fn is_high_risk(&self) -> bool {
        self.overall_risk_score >= 0.5
    }

    /// Whether polypharmacy is present (≥ 5 concurrent medications).
    pub fn has_polypharmacy(&self) -> bool {
        self.polypharmacy_count >= 5
    }

    /// Whether excessive polypharmacy is present (≥ 10 medications).
    pub fn has_excessive_polypharmacy(&self) -> bool {
        self.polypharmacy_count >= 10
    }

    /// Summary of all active risk factors.
    pub fn risk_summary(&self) -> String {
        if self.risk_factors.is_empty() {
            return "No significant risk factors identified".to_string();
        }
        self.risk_factors.join("; ")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Patient
// ═══════════════════════════════════════════════════════════════════════════

/// A condition on the patient's problem list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientConditionRecord {
    pub code: String,
    pub name: String,
    pub active: bool,
    pub onset_date: Option<String>,
}

/// A medication in the patient's regimen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientMedicationRecord {
    pub drug_id: String,
    pub name: String,
    pub dose_mg: f64,
    pub interval_hours: f64,
    pub route: String,
    pub active: bool,
}

/// A lab result in the patient's record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientLabRecord {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp: Option<String>,
}

/// Full patient model with demographics and clinical context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patient {
    pub id: String,
    pub demographics: Demographics,
    pub conditions: Vec<PatientConditionRecord>,
    pub medications: Vec<PatientMedicationRecord>,
    pub lab_values: Vec<PatientLabRecord>,
    pub allergies: Vec<Allergy>,
    pub risk_profile: PatientRiskProfile,
    pub notes: Vec<String>,
}

impl Patient {
    /// Create a minimal patient with an ID and default demographics.
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            demographics: Demographics::default(),
            conditions: Vec::new(),
            medications: Vec::new(),
            lab_values: Vec::new(),
            allergies: Vec::new(),
            risk_profile: PatientRiskProfile::default(),
            notes: Vec::new(),
        }
    }

    /// Start a builder for fluent construction.
    pub fn builder(id: &str) -> PatientBuilder {
        PatientBuilder::new(id)
    }

    // ── Computed Properties ───────────────────────────────────────────

    /// Body mass index.
    pub fn calculate_bmi(&self) -> f64 {
        self.demographics.bmi()
    }

    /// Body surface area (Mosteller).
    pub fn calculate_bsa(&self) -> f64 {
        self.demographics.bsa()
    }

    /// Estimated GFR using the Cockcroft-Gault equation (mL/min).
    pub fn calculate_egfr(&self) -> f64 {
        let age = self.demographics.age_years;
        let wt = self.demographics.weight_kg;
        let scr = self.demographics.serum_creatinine;
        if scr <= 0.0 || age <= 0.0 {
            return 0.0;
        }
        let sex_factor = if self.demographics.sex == Sex::Female {
            0.85
        } else {
            1.0
        };
        ((140.0 - age) * wt) / (72.0 * scr) * sex_factor
    }

    /// CKD-EPI 2021 eGFR equation (mL/min/1.73 m²).
    pub fn calculate_ckd_epi_egfr(&self) -> f64 {
        let scr = self.demographics.serum_creatinine;
        let age = self.demographics.age_years;
        if scr <= 0.0 || age <= 0.0 {
            return 0.0;
        }
        let (kappa, alpha, female_mult) = match self.demographics.sex {
            Sex::Female => (0.7, -0.241, 1.012),
            Sex::Male => (0.9, -0.302, 1.0),
        };
        let scr_k = scr / kappa;
        let min_term = scr_k.min(1.0);
        let max_term = scr_k.max(1.0);
        142.0
            * min_term.powf(alpha)
            * max_term.powf(-1.200)
            * 0.9938_f64.powf(age)
            * female_mult
    }

    /// Whether the patient is elderly (≥ 65 years).
    pub fn is_elderly(&self) -> bool {
        self.demographics.age_years >= 65.0
    }

    /// Whether the patient is pediatric (< 18 years).
    pub fn is_pediatric(&self) -> bool {
        self.demographics.age_years < 18.0
    }

    /// Whether the patient has a specific active condition by code.
    pub fn has_condition(&self, code: &str) -> bool {
        self.conditions
            .iter()
            .any(|c| c.active && c.code.eq_ignore_ascii_case(code))
    }

    /// Whether the patient has a condition matching a name substring.
    pub fn has_condition_named(&self, name: &str) -> bool {
        let lc = name.to_lowercase();
        self.conditions
            .iter()
            .any(|c| c.active && c.name.to_lowercase().contains(&lc))
    }

    /// Whether the patient has a specific allergy (case-insensitive).
    pub fn has_allergy(&self, substance: &str) -> bool {
        let lc = substance.to_lowercase();
        self.allergies
            .iter()
            .any(|a| a.substance.to_lowercase().contains(&lc))
    }

    /// Whether the patient has any life-threatening allergies.
    pub fn has_life_threatening_allergy(&self) -> bool {
        self.allergies.iter().any(|a| a.is_life_threatening())
    }

    /// The number of active medications (polypharmacy count).
    pub fn active_medication_count(&self) -> usize {
        self.medications.iter().filter(|m| m.active).count()
    }

    /// Whether the patient is on a specific medication (case-insensitive).
    pub fn is_on_medication(&self, drug: &str) -> bool {
        let lc = drug.to_lowercase();
        self.medications
            .iter()
            .any(|m| m.active && m.drug_id.to_lowercase() == lc)
    }

    /// Find the most recent lab result by name.
    pub fn latest_lab(&self, name: &str) -> Option<&PatientLabRecord> {
        let lc = name.to_lowercase();
        self.lab_values
            .iter()
            .rev()
            .find(|l| l.name.to_lowercase() == lc)
    }

    /// Renal classification derived from Cockcroft-Gault eGFR.
    pub fn renal_stage(&self) -> RenalStage {
        RenalStage::from_egfr(self.calculate_egfr())
    }

    /// Creatinine clearance (mL/min), adjusted for lean body weight
    /// when the patient is obese (BMI > 30).
    pub fn creatinine_clearance_adjusted(&self) -> f64 {
        let age = self.demographics.age_years;
        let scr = self.demographics.serum_creatinine;
        if scr <= 0.0 || age <= 0.0 {
            return 0.0;
        }
        let wt = if self.demographics.bmi() > 30.0 {
            self.demographics.adjusted_body_weight()
        } else {
            self.demographics.weight_kg
        };
        let sex_factor = if self.demographics.sex == Sex::Female {
            0.85
        } else {
            1.0
        };
        ((140.0 - age) * wt) / (72.0 * scr) * sex_factor
    }

    // ── Risk Profile ─────────────────────────────────────────────────

    /// Compute a comprehensive risk profile from the patient's clinical data.
    pub fn compute_risk_profile(&mut self) {
        let mut profile = PatientRiskProfile::default();
        let mut factors = Vec::new();

        if self.is_elderly() {
            factors.push("Age ≥ 65".to_string());
            if self.demographics.age_years >= 80.0 {
                factors.push("Age ≥ 80 (very elderly)".to_string());
            }
        }

        profile.polypharmacy_count = self.active_medication_count();
        if profile.has_polypharmacy() {
            factors.push(format!(
                "Polypharmacy ({} medications)",
                profile.polypharmacy_count
            ));
        }

        let egfr = self.calculate_egfr();
        profile.renal_stage = RenalStage::from_egfr(egfr);
        if profile.renal_stage.requires_dose_adjustment() {
            factors.push(format!(
                "Renal impairment (eGFR {:.0}, {})",
                egfr,
                profile.renal_stage.label()
            ));
        }

        let cns_meds = [
            "benzodiazepine", "opioid", "zolpidem", "trazodone", "gabapentin",
        ];
        let has_cns = self.medications.iter().any(|m| {
            let id = m.drug_id.to_lowercase();
            cns_meds.iter().any(|c| id.contains(c))
        });
        if self.is_elderly() && (has_cns || profile.polypharmacy_count >= 5) {
            profile.fall_risk = true;
            factors.push("Fall risk (elderly + CNS/polypharmacy)".to_string());
        }

        let mut bleed_score: f64 = 0.0;
        if self.demographics.age_years >= 65.0 {
            bleed_score += 1.0;
        }
        if self.has_condition_named("hypertension") {
            bleed_score += 1.0;
        }
        if profile.renal_stage.requires_dose_adjustment() {
            bleed_score += 1.0;
        }
        if self.is_on_medication("warfarin")
            || self.is_on_medication("apixaban")
            || self.is_on_medication("rivaroxaban")
        {
            bleed_score += 1.0;
        }
        if self.is_on_medication("aspirin") || self.is_on_medication("clopidogrel") {
            bleed_score += 1.0;
        }
        if self.demographics.alcohol_use {
            bleed_score += 1.0;
        }
        profile.bleeding_risk = (bleed_score / 6.0).min(1.0);
        if profile.bleeding_risk > 0.5 {
            factors.push(format!(
                "Elevated bleeding risk ({:.0}%)",
                profile.bleeding_risk * 100.0
            ));
        }

        let acb_meds = [
            "diphenhydramine", "oxybutynin", "amitriptyline", "paroxetine",
            "hydroxyzine", "quetiapine", "olanzapine", "chlorpromazine",
        ];
        let acb_count = self
            .medications
            .iter()
            .filter(|m| {
                let id = m.drug_id.to_lowercase();
                acb_meds.iter().any(|a| id.contains(a))
            })
            .count();
        profile.anticholinergic_burden = acb_count as f64;
        if acb_count >= 2 {
            factors.push(format!("High anticholinergic burden ({acb_count} agents)"));
        }

        let bmi = self.calculate_bmi();
        if bmi > 40.0 {
            factors.push(format!("Morbid obesity (BMI {bmi:.1})"));
        } else if bmi > 30.0 {
            factors.push(format!("Obesity (BMI {bmi:.1})"));
        } else if bmi < 18.5 && bmi > 0.0 {
            factors.push(format!("Underweight (BMI {bmi:.1})"));
        }

        if self.demographics.smoking_status {
            factors.push("Active smoker".to_string());
        }

        let age_risk = if self.demographics.age_years >= 80.0 {
            0.3
        } else if self.is_elderly() {
            0.15
        } else {
            0.0
        };
        let renal_risk = match profile.renal_stage {
            RenalStage::SevereImpairment | RenalStage::EndStage => 0.3,
            RenalStage::ModerateImpairment => 0.15,
            _ => 0.0,
        };
        let poly_risk = if profile.polypharmacy_count >= 10 {
            0.3
        } else if profile.polypharmacy_count >= 5 {
            0.15
        } else {
            0.0
        };
        profile.overall_risk_score =
            (age_risk + renal_risk + poly_risk + profile.bleeding_risk * 0.3).min(1.0);

        profile.risk_factors = factors;
        self.risk_profile = profile;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PatientBuilder
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for constructing [`Patient`] objects.
pub struct PatientBuilder {
    patient: Patient,
}

impl PatientBuilder {
    pub fn new(id: &str) -> Self {
        Self {
            patient: Patient::new(id),
        }
    }

    pub fn with_age(mut self, age: f64) -> Self {
        self.patient.demographics.age_years = age;
        self
    }

    pub fn with_sex(mut self, sex: Sex) -> Self {
        self.patient.demographics.sex = sex;
        self
    }

    pub fn with_weight(mut self, kg: f64) -> Self {
        self.patient.demographics.weight_kg = kg;
        self
    }

    pub fn with_height(mut self, cm: f64) -> Self {
        self.patient.demographics.height_cm = cm;
        self
    }

    pub fn with_race(mut self, race: Race) -> Self {
        self.patient.demographics.race = race;
        self
    }

    pub fn with_serum_creatinine(mut self, scr: f64) -> Self {
        self.patient.demographics.serum_creatinine = scr;
        self
    }

    pub fn with_smoking(mut self, smoking: bool) -> Self {
        self.patient.demographics.smoking_status = smoking;
        self
    }

    pub fn with_alcohol_use(mut self, alcohol: bool) -> Self {
        self.patient.demographics.alcohol_use = alcohol;
        self
    }

    pub fn add_condition(mut self, code: &str, name: &str) -> Self {
        self.patient.conditions.push(PatientConditionRecord {
            code: code.to_string(),
            name: name.to_string(),
            active: true,
            onset_date: None,
        });
        self
    }

    pub fn add_inactive_condition(mut self, code: &str, name: &str) -> Self {
        self.patient.conditions.push(PatientConditionRecord {
            code: code.to_string(),
            name: name.to_string(),
            active: false,
            onset_date: None,
        });
        self
    }

    pub fn add_medication(
        mut self,
        drug_id: &str,
        name: &str,
        dose_mg: f64,
        interval_hours: f64,
    ) -> Self {
        self.patient.medications.push(PatientMedicationRecord {
            drug_id: drug_id.to_string(),
            name: name.to_string(),
            dose_mg,
            interval_hours,
            route: "oral".to_string(),
            active: true,
        });
        self
    }

    pub fn add_lab(mut self, name: &str, value: f64, unit: &str) -> Self {
        self.patient.lab_values.push(PatientLabRecord {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            timestamp: None,
        });
        self
    }

    pub fn add_allergy(mut self, allergy: Allergy) -> Self {
        self.patient.allergies.push(allergy);
        self
    }

    pub fn add_allergy_simple(mut self, substance: &str) -> Self {
        self.patient.allergies.push(Allergy {
            substance: substance.to_string(),
            reaction: "Unknown".to_string(),
            severity: AllergySeverity::Moderate,
            verified: false,
        });
        self
    }

    pub fn with_renal_stage(mut self, stage: RenalStage) -> Self {
        self.patient.risk_profile.renal_stage = stage;
        self
    }

    pub fn with_hepatic_class(mut self, class: ChildPughClass) -> Self {
        self.patient.risk_profile.hepatic_class = Some(class);
        self
    }

    /// Build the patient, automatically computing the risk profile.
    pub fn build(mut self) -> Patient {
        self.patient.compute_risk_profile();
        self.patient
    }

    /// Build without computing the risk profile.
    pub fn build_raw(self) -> Patient {
        self.patient
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PatientSummary
// ═══════════════════════════════════════════════════════════════════════════

/// A concise, read-only summary of a patient for display purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientSummary {
    pub id: String,
    pub age: f64,
    pub sex: String,
    pub bmi: f64,
    pub egfr: f64,
    pub active_conditions: usize,
    pub active_medications: usize,
    pub allergy_count: usize,
    pub is_high_risk: bool,
    pub risk_summary: String,
}

impl From<&Patient> for PatientSummary {
    fn from(p: &Patient) -> Self {
        Self {
            id: p.id.clone(),
            age: p.demographics.age_years,
            sex: p.demographics.sex.to_string(),
            bmi: p.calculate_bmi(),
            egfr: p.calculate_egfr(),
            active_conditions: p.conditions.iter().filter(|c| c.active).count(),
            active_medications: p.active_medication_count(),
            allergy_count: p.allergies.len(),
            is_high_risk: p.risk_profile.is_high_risk(),
            risk_summary: p.risk_profile.risk_summary(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PatientComparator
// ═══════════════════════════════════════════════════════════════════════════

/// Utility for comparing two patient snapshots (e.g., across encounters).
pub struct PatientComparator;

impl PatientComparator {
    /// Returns a list of textual differences between two patient states.
    pub fn diff(a: &Patient, b: &Patient) -> Vec<String> {
        let mut diffs = Vec::new();

        if (a.demographics.weight_kg - b.demographics.weight_kg).abs() > 0.5 {
            diffs.push(format!(
                "Weight changed: {:.1} -> {:.1} kg",
                a.demographics.weight_kg, b.demographics.weight_kg
            ));
        }

        let a_codes: HashSet<_> = a
            .conditions
            .iter()
            .filter(|c| c.active)
            .map(|c| c.code.as_str())
            .collect();
        let b_codes: HashSet<_> = b
            .conditions
            .iter()
            .filter(|c| c.active)
            .map(|c| c.code.as_str())
            .collect();
        for code in b_codes.difference(&a_codes) {
            diffs.push(format!("New condition: {code}"));
        }
        for code in a_codes.difference(&b_codes) {
            diffs.push(format!("Resolved condition: {code}"));
        }

        let a_meds: HashSet<_> = a
            .medications
            .iter()
            .filter(|m| m.active)
            .map(|m| m.drug_id.as_str())
            .collect();
        let b_meds: HashSet<_> = b
            .medications
            .iter()
            .filter(|m| m.active)
            .map(|m| m.drug_id.as_str())
            .collect();
        for med in b_meds.difference(&a_meds) {
            diffs.push(format!("New medication: {med}"));
        }
        for med in a_meds.difference(&b_meds) {
            diffs.push(format!("Discontinued medication: {med}"));
        }

        let egfr_a = a.calculate_egfr();
        let egfr_b = b.calculate_egfr();
        if (egfr_a - egfr_b).abs() > 5.0 {
            diffs.push(format!("eGFR changed: {egfr_a:.0} -> {egfr_b:.0}"));
        }

        diffs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn elderly_diabetic() -> Patient {
        Patient::builder("PT-001")
            .with_age(72.0)
            .with_sex(Sex::Male)
            .with_weight(85.0)
            .with_height(175.0)
            .with_serum_creatinine(1.4)
            .add_condition("E11", "Type 2 Diabetes")
            .add_condition("I10", "Hypertension")
            .add_medication("metformin", "Metformin", 500.0, 12.0)
            .add_medication("lisinopril", "Lisinopril", 10.0, 24.0)
            .add_medication("warfarin", "Warfarin", 5.0, 24.0)
            .add_medication("atorvastatin", "Atorvastatin", 40.0, 24.0)
            .add_medication("omeprazole", "Omeprazole", 20.0, 24.0)
            .add_allergy_simple("Penicillin")
            .add_lab("Creatinine", 1.4, "mg/dL")
            .add_lab("HbA1c", 7.2, "%")
            .build()
    }

    #[test]
    fn test_bmi_calculation() {
        let p = Patient::builder("PT-BMI")
            .with_weight(85.0)
            .with_height(175.0)
            .build_raw();
        let bmi = p.calculate_bmi();
        let expected = 85.0 / (1.75 * 1.75);
        assert!((bmi - expected).abs() < 0.01);
    }

    #[test]
    fn test_bmi_zero_height() {
        let p = Patient::builder("PT-0H")
            .with_weight(70.0)
            .with_height(0.0)
            .build_raw();
        assert_eq!(p.calculate_bmi(), 0.0);
    }

    #[test]
    fn test_bsa_calculation() {
        let p = Patient::builder("PT-BSA")
            .with_weight(70.0)
            .with_height(170.0)
            .build_raw();
        let bsa = p.calculate_bsa();
        let expected = ((170.0 * 70.0) / 3600.0_f64).sqrt();
        assert!((bsa - expected).abs() < 0.001);
    }

    #[test]
    fn test_egfr_cockcroft_gault() {
        let p = Patient::builder("PT-GFR")
            .with_age(72.0)
            .with_sex(Sex::Male)
            .with_weight(85.0)
            .with_serum_creatinine(1.4)
            .build_raw();
        let egfr = p.calculate_egfr();
        assert!(egfr > 55.0 && egfr < 60.0, "eGFR = {egfr}");
    }

    #[test]
    fn test_egfr_female_factor() {
        let male = Patient::builder("M")
            .with_age(50.0)
            .with_sex(Sex::Male)
            .with_weight(70.0)
            .with_serum_creatinine(1.0)
            .build_raw();
        let female = Patient::builder("F")
            .with_age(50.0)
            .with_sex(Sex::Female)
            .with_weight(70.0)
            .with_serum_creatinine(1.0)
            .build_raw();
        let ratio = female.calculate_egfr() / male.calculate_egfr();
        assert!((ratio - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_is_elderly() {
        let old = Patient::builder("OLD").with_age(65.0).build_raw();
        assert!(old.is_elderly());
        let young = Patient::builder("YOUNG").with_age(40.0).build_raw();
        assert!(!young.is_elderly());
    }

    #[test]
    fn test_is_pediatric() {
        let child = Patient::builder("KID").with_age(10.0).build_raw();
        assert!(child.is_pediatric());
        let adult = Patient::builder("ADULT").with_age(30.0).build_raw();
        assert!(!adult.is_pediatric());
    }

    #[test]
    fn test_has_condition() {
        let p = elderly_diabetic();
        assert!(p.has_condition("E11"));
        assert!(p.has_condition("I10"));
        assert!(!p.has_condition("F33"));
    }

    #[test]
    fn test_has_condition_named() {
        let p = elderly_diabetic();
        assert!(p.has_condition_named("diabetes"));
        assert!(p.has_condition_named("HYPERTENSION"));
        assert!(!p.has_condition_named("asthma"));
    }

    #[test]
    fn test_has_allergy() {
        let p = elderly_diabetic();
        assert!(p.has_allergy("penicillin"));
        assert!(!p.has_allergy("sulfa"));
    }

    #[test]
    fn test_is_on_medication() {
        let p = elderly_diabetic();
        assert!(p.is_on_medication("warfarin"));
        assert!(!p.is_on_medication("aspirin"));
    }

    #[test]
    fn test_active_medication_count() {
        let p = elderly_diabetic();
        assert_eq!(p.active_medication_count(), 5);
    }

    #[test]
    fn test_latest_lab() {
        let p = elderly_diabetic();
        let cr = p.latest_lab("creatinine").unwrap();
        assert!((cr.value - 1.4).abs() < f64::EPSILON);
        assert!(p.latest_lab("BNP").is_none());
    }

    #[test]
    fn test_renal_stage_from_egfr() {
        assert_eq!(RenalStage::from_egfr(95.0), RenalStage::Normal);
        assert_eq!(RenalStage::from_egfr(75.0), RenalStage::MildImpairment);
        assert_eq!(RenalStage::from_egfr(45.0), RenalStage::ModerateImpairment);
        assert_eq!(RenalStage::from_egfr(20.0), RenalStage::SevereImpairment);
        assert_eq!(RenalStage::from_egfr(10.0), RenalStage::EndStage);
    }

    #[test]
    fn test_child_pugh_from_score() {
        assert_eq!(ChildPughClass::from_score(5), Some(ChildPughClass::A));
        assert_eq!(ChildPughClass::from_score(8), Some(ChildPughClass::B));
        assert_eq!(ChildPughClass::from_score(12), Some(ChildPughClass::C));
        assert_eq!(ChildPughClass::from_score(16), None);
    }

    #[test]
    fn test_risk_profile_polypharmacy() {
        let p = elderly_diabetic();
        assert!(p.risk_profile.has_polypharmacy());
        assert!(!p.risk_profile.has_excessive_polypharmacy());
    }

    #[test]
    fn test_risk_profile_computed() {
        let p = elderly_diabetic();
        assert!(p.risk_profile.renal_stage.requires_dose_adjustment());
        assert!(!p.risk_profile.risk_factors.is_empty());
    }

    #[test]
    fn test_ideal_body_weight() {
        let d = Demographics {
            sex: Sex::Male,
            height_cm: 177.8,
            ..Default::default()
        };
        let ibw = d.ideal_body_weight();
        assert!((ibw - 73.0).abs() < 0.1);
    }

    #[test]
    fn test_adjusted_body_weight() {
        let d = Demographics {
            sex: Sex::Male,
            height_cm: 177.8,
            weight_kg: 120.0,
            ..Default::default()
        };
        let ibw = d.ideal_body_weight();
        let abw = d.adjusted_body_weight();
        let expected = ibw + 0.4 * (120.0 - ibw);
        assert!((abw - expected).abs() < 0.1);
    }

    #[test]
    fn test_patient_summary() {
        let p = elderly_diabetic();
        let summary = PatientSummary::from(&p);
        assert_eq!(summary.age, 72.0);
        assert_eq!(summary.active_medications, 5);
        assert_eq!(summary.allergy_count, 1);
    }

    #[test]
    fn test_patient_comparator() {
        let a = Patient::builder("PT-A")
            .with_age(65.0)
            .with_weight(70.0)
            .with_height(170.0)
            .add_condition("I10", "Hypertension")
            .add_medication("lisinopril", "Lisinopril", 10.0, 24.0)
            .build();
        let b = Patient::builder("PT-A")
            .with_age(65.0)
            .with_weight(75.0)
            .with_height(170.0)
            .add_condition("I10", "Hypertension")
            .add_condition("E11", "Diabetes")
            .add_medication("lisinopril", "Lisinopril", 10.0, 24.0)
            .add_medication("metformin", "Metformin", 500.0, 12.0)
            .build();
        let diffs = PatientComparator::diff(&a, &b);
        assert!(diffs.iter().any(|d| d.contains("Weight changed")));
        assert!(diffs.iter().any(|d| d.contains("New condition")));
        assert!(diffs.iter().any(|d| d.contains("New medication")));
    }

    #[test]
    fn test_allergy_severity() {
        let mild = Allergy::new("Aspirin", "Rash", AllergySeverity::Mild);
        assert!(!mild.is_life_threatening());
        let severe =
            Allergy::new("Penicillin", "Anaphylaxis", AllergySeverity::Anaphylaxis).verified();
        assert!(severe.is_life_threatening());
        assert!(severe.verified);
    }

    #[test]
    fn test_sex_parse() {
        assert_eq!("male".parse::<Sex>().unwrap(), Sex::Male);
        assert_eq!("F".parse::<Sex>().unwrap(), Sex::Female);
        assert!("unknown".parse::<Sex>().is_err());
    }

    #[test]
    fn test_builder_defaults() {
        let p = Patient::builder("PT-DEF").build_raw();
        assert_eq!(p.id, "PT-DEF");
        assert_eq!(p.demographics.age_years, 50.0);
        assert_eq!(p.demographics.sex, Sex::Male);
        assert!(p.conditions.is_empty());
    }

    #[test]
    fn test_lean_body_weight() {
        let d = Demographics {
            sex: Sex::Male,
            weight_kg: 80.0,
            height_cm: 180.0,
            ..Default::default()
        };
        let lbw = d.lean_body_weight();
        assert!((lbw - 61.42).abs() < 0.1);
    }

    #[test]
    fn test_ckd_epi_egfr_nonzero() {
        let p = Patient::builder("EPI")
            .with_age(60.0)
            .with_sex(Sex::Male)
            .with_weight(80.0)
            .with_height(175.0)
            .with_serum_creatinine(1.0)
            .build_raw();
        let egfr = p.calculate_ckd_epi_egfr();
        assert!(egfr > 60.0 && egfr < 120.0, "CKD-EPI eGFR = {egfr}");
    }
}
