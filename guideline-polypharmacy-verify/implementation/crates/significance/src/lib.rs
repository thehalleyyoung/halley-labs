//! # GuardPharma Clinical Significance Filter
//!
//! This crate implements clinical significance filtering for polypharmacy conflicts.
//! It stratifies drug–drug interactions by severity using multiple evidence sources:
//!
//! - **DrugBank** interaction severity data and evidence levels
//! - **Beers Criteria** (2023 AGS update) for potentially inappropriate medications
//! - **FAERS** disproportionality signals (reporting odds ratios, PRR, IC)
//! - **Medicare comorbidity** prevalence-weighted scoring (Charlson, Elixhauser)
//!
//! A composite scoring engine combines these sources with configurable weights
//! to produce a final [`ClinicalSeverity`] classification for each conflict.

pub mod drugbank;
pub mod beers;
pub mod faers;
pub mod comorbidity;
pub mod composite;
pub mod filter;

use serde::{Deserialize, Serialize};
use std::fmt;

// ─────────────────────────── DrugId ──────────────────────────────────────

/// Unique identifier for a drug (canonical lower-case, underscored).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct DrugId(pub String);

impl DrugId {
    pub fn new(name: &str) -> Self {
        DrugId(name.to_lowercase().replace(' ', "_"))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DrugId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for DrugId {
    fn from(s: &str) -> Self { DrugId::new(s) }
}

impl From<String> for DrugId {
    fn from(s: String) -> Self { DrugId(s.to_lowercase().replace(' ', "_")) }
}

// ─────────────────────────── Severity ────────────────────────────────────

/// Base severity classification for drug interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Severity {
    Minor,
    Moderate,
    Major,
    Contraindicated,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Minor => write!(f, "Minor"),
            Severity::Moderate => write!(f, "Moderate"),
            Severity::Major => write!(f, "Major"),
            Severity::Contraindicated => write!(f, "Contraindicated"),
        }
    }
}

// ─────────────────────────── Sex ─────────────────────────────────────────

/// Patient biological sex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex { Male, Female }

// ─────────────────────────── RenalFunction ───────────────────────────────

/// Renal function classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum RenalFunction {
    Normal,
    Mild,
    Moderate,
    Severe,
    EndStage,
}

impl RenalFunction {
    pub fn from_egfr(egfr: f64) -> Self {
        if egfr >= 90.0 { RenalFunction::Normal }
        else if egfr >= 60.0 { RenalFunction::Mild }
        else if egfr >= 30.0 { RenalFunction::Moderate }
        else if egfr >= 15.0 { RenalFunction::Severe }
        else { RenalFunction::EndStage }
    }
}

// ─────────────────────────── Condition ───────────────────────────────────

/// ICD-10 coded clinical condition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Condition {
    pub code: String,
    pub name: String,
    pub active: bool,
}

impl Condition {
    pub fn new(code: &str, name: &str) -> Self {
        Condition { code: code.to_string(), name: name.to_string(), active: true }
    }
    pub fn inactive(code: &str, name: &str) -> Self {
        Condition { code: code.to_string(), name: name.to_string(), active: false }
    }
    pub fn code_starts_with(&self, prefix: &str) -> bool {
        self.code.starts_with(prefix)
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.code)
    }
}

// ─────────────────────────── Medication ──────────────────────────────────

/// A medication a patient is currently taking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Medication {
    pub drug_id: DrugId,
    pub name: String,
    pub drug_class: String,
    pub dose_mg: f64,
    pub frequency_hours: f64,
    pub route: String,
}

impl Medication {
    pub fn new(name: &str, drug_class: &str, dose_mg: f64) -> Self {
        Medication {
            drug_id: DrugId::new(name),
            name: name.to_string(),
            drug_class: drug_class.to_string(),
            dose_mg,
            frequency_hours: 24.0,
            route: "oral".to_string(),
        }
    }
    pub fn with_frequency(mut self, hours: f64) -> Self { self.frequency_hours = hours; self }
    pub fn with_route(mut self, route: &str) -> Self { self.route = route.to_string(); self }
    pub fn canonical_name(&self) -> String { self.name.to_lowercase() }
    pub fn is_class(&self, cls: &str) -> bool {
        self.drug_class.to_lowercase() == cls.to_lowercase()
    }
}

impl fmt::Display for Medication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}mg", self.name, self.dose_mg)
    }
}

// ──────────────────────────── PatientProfile ─────────────────────────────

/// Full patient profile including demographics, conditions, and medications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientProfile {
    pub age_years: f64,
    pub weight_kg: f64,
    pub sex: Sex,
    pub conditions: Vec<Condition>,
    pub medications: Vec<Medication>,
    pub egfr: Option<f64>,
}

impl PatientProfile {
    pub fn new(age: f64, weight: f64, sex: Sex) -> Self {
        PatientProfile {
            age_years: age, weight_kg: weight, sex,
            conditions: Vec::new(), medications: Vec::new(), egfr: None,
        }
    }
    pub fn with_conditions(mut self, c: Vec<Condition>) -> Self { self.conditions = c; self }
    pub fn with_medications(mut self, m: Vec<Medication>) -> Self { self.medications = m; self }
    pub fn with_egfr(mut self, e: f64) -> Self { self.egfr = Some(e); self }
    pub fn age(&self) -> f64 { self.age_years }
    pub fn sex(&self) -> Sex { self.sex }
    pub fn renal_function(&self) -> RenalFunction {
        self.egfr.map(RenalFunction::from_egfr).unwrap_or(RenalFunction::Normal)
    }
    pub fn has_condition_code(&self, prefix: &str) -> bool {
        self.conditions.iter().any(|c| c.active && c.code_starts_with(prefix))
    }
    pub fn has_condition_named(&self, frag: &str) -> bool {
        let lower = frag.to_lowercase();
        self.conditions.iter().any(|c| c.active && c.name.to_lowercase().contains(&lower))
    }
    pub fn active_conditions(&self) -> Vec<&Condition> {
        self.conditions.iter().filter(|c| c.active).collect()
    }
    pub fn medication_count(&self) -> usize { self.medications.len() }
    pub fn is_elderly(&self) -> bool { self.age_years >= 65.0 }
    pub fn is_very_elderly(&self) -> bool { self.age_years >= 85.0 }
}

impl Default for PatientProfile {
    fn default() -> Self {
        PatientProfile::new(70.0, 70.0, Sex::Male)
    }
}

// ──────────────────────────── ConfirmedConflict ──────────────────────────

/// A confirmed drug–drug interaction conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmedConflict {
    pub id: String,
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub drug_a_name: String,
    pub drug_b_name: String,
    pub severity: Severity,
    pub mechanism: String,
    pub description: String,
    pub clinical_consequence: String,
    pub evidence_source: String,
}

impl ConfirmedConflict {
    pub fn new(
        drug_a_name: &str, drug_b_name: &str, severity: Severity,
        mechanism: &str, description: &str,
    ) -> Self {
        let drug_a = DrugId::new(drug_a_name);
        let drug_b = DrugId::new(drug_b_name);
        let id = format!("{}_{}", drug_a.as_str(), drug_b.as_str());
        ConfirmedConflict {
            id, drug_a, drug_b,
            drug_a_name: drug_a_name.to_string(),
            drug_b_name: drug_b_name.to_string(),
            severity, mechanism: mechanism.to_string(),
            description: description.to_string(),
            clinical_consequence: String::new(),
            evidence_source: String::new(),
        }
    }
    pub fn with_consequence(mut self, c: &str) -> Self { self.clinical_consequence = c.to_string(); self }
    pub fn with_source(mut self, s: &str) -> Self { self.evidence_source = s.to_string(); self }
    pub fn pair_key(&self) -> (String, String) {
        let a = self.drug_a.as_str().to_string();
        let b = self.drug_b.as_str().to_string();
        if a <= b { (a, b) } else { (b, a) }
    }
}

impl fmt::Display for ConfirmedConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ↔ {}: {}", self.severity, self.drug_a_name, self.drug_b_name, self.description)
    }
}

// ──────────────────────────── ClinicalConfig ─────────────────────────────

/// Configuration for the clinical significance filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalConfig {
    pub drugbank_weight: f64,
    pub beers_weight: f64,
    pub faers_weight: f64,
    pub comorbidity_weight: f64,
    pub min_severity_threshold: f64,
    pub include_beers: bool,
    pub include_faers: bool,
    pub include_comorbidity: bool,
    pub elderly_age_threshold: f64,
}

impl Default for ClinicalConfig {
    fn default() -> Self {
        ClinicalConfig {
            drugbank_weight: 0.35, beers_weight: 0.25,
            faers_weight: 0.20, comorbidity_weight: 0.20,
            min_severity_threshold: 0.3,
            include_beers: true, include_faers: true, include_comorbidity: true,
            elderly_age_threshold: 65.0,
        }
    }
}

// ──────────────────────────── Re-exports ─────────────────────────────────

pub use drugbank::{
    DrugBankDatabase, DrugBankEntry, DrugBankInteraction, DrugBankSeverity, EvidenceLevel,
};
pub use beers::{
    BeersCategory, BeersCriteria, BeersCriterion, BeersViolation, QualityOfEvidence,
    StrengthOfRecommendation,
};
pub use faers::{
    AdverseEventType, FaersDatabase, FaersSignal, SignalStrength,
};
pub use comorbidity::{
    CharlsonCategory, ComorbidityPrevalence, ElixhauserCategory,
    MedicarePrevalenceData, PopulationImpactScore,
};
pub use composite::{
    ClinicalSeverity, ScoredConflict, SignificanceReport, SignificanceScore,
    SignificanceScorer,
};
pub use filter::{
    FilterConfig, FilterResult, FilterStatistics, SignificanceFilter,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_new() {
        let c = Condition::new("I10", "Hypertension");
        assert_eq!(c.code, "I10");
        assert!(c.active);
    }

    #[test]
    fn test_condition_code_prefix() {
        let c = Condition::new("E11.65", "Type 2 DM with hyperglycemia");
        assert!(c.code_starts_with("E11"));
        assert!(!c.code_starts_with("E12"));
    }

    #[test]
    fn test_medication_canonical() {
        let m = Medication::new("Warfarin", "anticoagulant", 5.0);
        assert_eq!(m.canonical_name(), "warfarin");
    }

    #[test]
    fn test_medication_class_check() {
        let m = Medication::new("Ibuprofen", "NSAID", 400.0);
        assert!(m.is_class("nsaid"));
        assert!(!m.is_class("statin"));
    }

    #[test]
    fn test_patient_profile_defaults() {
        let p = PatientProfile::default();
        assert!(p.is_elderly());
        assert!(!p.is_very_elderly());
    }

    #[test]
    fn test_patient_profile_conditions() {
        let p = PatientProfile::default().with_conditions(vec![
            Condition::new("I10", "Hypertension"),
            Condition::new("E11", "Type 2 Diabetes"),
        ]);
        assert!(p.has_condition_code("I10"));
        assert!(!p.has_condition_code("N18"));
    }

    #[test]
    fn test_confirmed_conflict_pair_key() {
        let c = ConfirmedConflict::new("Warfarin", "Aspirin", Severity::Major, "test", "test");
        let (a, b) = c.pair_key();
        assert!(a <= b);
    }

    #[test]
    fn test_clinical_config_defaults() {
        let cfg = ClinicalConfig::default();
        let total = cfg.drugbank_weight + cfg.beers_weight + cfg.faers_weight + cfg.comorbidity_weight;
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_patient_renal_function() {
        let p = PatientProfile::default().with_egfr(45.0);
        assert_eq!(p.renal_function(), RenalFunction::Moderate);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Minor < Severity::Moderate);
        assert!(Severity::Moderate < Severity::Major);
        assert!(Severity::Major < Severity::Contraindicated);
    }
}
