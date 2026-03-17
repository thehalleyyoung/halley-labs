//! Medicare comorbidity prevalence scoring.
//!
//! Provides Charlson and Elixhauser comorbidity indices, Medicare prevalence
//! data for ~30+ conditions, and prevalence-weighted conflict severity scoring
//! to estimate population-level impact.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::{Condition, PatientProfile};
use crate::Sex;

// ─────────────────────────── Charlson Comorbidity ────────────────────────

/// Categories used in the Charlson Comorbidity Index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CharlsonCategory {
    MyocardialInfarction,
    CongestiveHeartFailure,
    PeripheralVascularDisease,
    CerebrovascularDisease,
    Dementia,
    ChronicPulmonaryDisease,
    RheumatologicDisease,
    PepticUlcerDisease,
    MildLiverDisease,
    DiabetesWithoutComplications,
    DiabetesWithComplications,
    HemiplegiaOrParaplegia,
    RenalDisease,
    MalignancyWithoutMetastasis,
    ModerateSevereLiverDisease,
    MetastaticSolidTumor,
    AIDS,
}

impl CharlsonCategory {
    /// Charlson weight for each category.
    pub fn weight(&self) -> u32 {
        match self {
            CharlsonCategory::MyocardialInfarction => 1,
            CharlsonCategory::CongestiveHeartFailure => 1,
            CharlsonCategory::PeripheralVascularDisease => 1,
            CharlsonCategory::CerebrovascularDisease => 1,
            CharlsonCategory::Dementia => 1,
            CharlsonCategory::ChronicPulmonaryDisease => 1,
            CharlsonCategory::RheumatologicDisease => 1,
            CharlsonCategory::PepticUlcerDisease => 1,
            CharlsonCategory::MildLiverDisease => 1,
            CharlsonCategory::DiabetesWithoutComplications => 1,
            CharlsonCategory::DiabetesWithComplications => 2,
            CharlsonCategory::HemiplegiaOrParaplegia => 2,
            CharlsonCategory::RenalDisease => 2,
            CharlsonCategory::MalignancyWithoutMetastasis => 2,
            CharlsonCategory::ModerateSevereLiverDisease => 3,
            CharlsonCategory::MetastaticSolidTumor => 6,
            CharlsonCategory::AIDS => 6,
        }
    }

    /// ICD-10 code prefixes that map to this category.
    pub fn icd10_prefixes(&self) -> &'static [&'static str] {
        match self {
            CharlsonCategory::MyocardialInfarction => &["I21", "I22", "I25.2"],
            CharlsonCategory::CongestiveHeartFailure => &["I09.9", "I11.0", "I13.0", "I13.2", "I25.5", "I42", "I43", "I50"],
            CharlsonCategory::PeripheralVascularDisease => &["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1", "I79"],
            CharlsonCategory::CerebrovascularDisease => &["G45", "G46", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
            CharlsonCategory::Dementia => &["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"],
            CharlsonCategory::ChronicPulmonaryDisease => &["I27.8", "I27.9", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67"],
            CharlsonCategory::RheumatologicDisease => &["M05", "M06", "M31.5", "M32", "M33", "M34", "M35.1", "M35.3", "M36"],
            CharlsonCategory::PepticUlcerDisease => &["K25", "K26", "K27", "K28"],
            CharlsonCategory::MildLiverDisease => &["B18", "K70.0", "K70.1", "K70.2", "K70.3", "K70.9", "K71.3", "K71.4", "K71.5", "K73", "K74", "K76.0", "K76.2", "K76.3", "K76.4", "K76.8", "K76.9"],
            CharlsonCategory::DiabetesWithoutComplications => &["E10.0", "E10.1", "E10.9", "E11.0", "E11.1", "E11.9", "E13.0", "E13.1", "E13.9"],
            CharlsonCategory::DiabetesWithComplications => &["E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8"],
            CharlsonCategory::HemiplegiaOrParaplegia => &["G04.1", "G11.4", "G80.1", "G80.2", "G81", "G82", "G83.0", "G83.1", "G83.2", "G83.3", "G83.4", "G83.9"],
            CharlsonCategory::RenalDisease => &["I12.0", "I13.1", "N03.2", "N03.7", "N05.2", "N05.7", "N18", "N19", "N25.0", "Z49.0", "Z49.1", "Z49.2", "Z94.0", "Z99.2"],
            CharlsonCategory::MalignancyWithoutMetastasis => &["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"],
            CharlsonCategory::ModerateSevereLiverDisease => &["I85.0", "I85.9", "I86.4", "I98.2", "K70.4", "K71.1", "K72.1", "K72.9", "K76.5", "K76.6", "K76.7"],
            CharlsonCategory::MetastaticSolidTumor => &["C77", "C78", "C79", "C80"],
            CharlsonCategory::AIDS => &["B20", "B21", "B22", "B24"],
        }
    }

    /// All Charlson categories.
    pub fn all() -> &'static [CharlsonCategory] {
        &[
            CharlsonCategory::MyocardialInfarction,
            CharlsonCategory::CongestiveHeartFailure,
            CharlsonCategory::PeripheralVascularDisease,
            CharlsonCategory::CerebrovascularDisease,
            CharlsonCategory::Dementia,
            CharlsonCategory::ChronicPulmonaryDisease,
            CharlsonCategory::RheumatologicDisease,
            CharlsonCategory::PepticUlcerDisease,
            CharlsonCategory::MildLiverDisease,
            CharlsonCategory::DiabetesWithoutComplications,
            CharlsonCategory::DiabetesWithComplications,
            CharlsonCategory::HemiplegiaOrParaplegia,
            CharlsonCategory::RenalDisease,
            CharlsonCategory::MalignancyWithoutMetastasis,
            CharlsonCategory::ModerateSevereLiverDisease,
            CharlsonCategory::MetastaticSolidTumor,
            CharlsonCategory::AIDS,
        ]
    }
}

impl fmt::Display for CharlsonCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            CharlsonCategory::MyocardialInfarction => "Myocardial Infarction",
            CharlsonCategory::CongestiveHeartFailure => "Congestive Heart Failure",
            CharlsonCategory::PeripheralVascularDisease => "Peripheral Vascular Disease",
            CharlsonCategory::CerebrovascularDisease => "Cerebrovascular Disease",
            CharlsonCategory::Dementia => "Dementia",
            CharlsonCategory::ChronicPulmonaryDisease => "Chronic Pulmonary Disease",
            CharlsonCategory::RheumatologicDisease => "Rheumatologic Disease",
            CharlsonCategory::PepticUlcerDisease => "Peptic Ulcer Disease",
            CharlsonCategory::MildLiverDisease => "Mild Liver Disease",
            CharlsonCategory::DiabetesWithoutComplications => "Diabetes (uncomplicated)",
            CharlsonCategory::DiabetesWithComplications => "Diabetes (with complications)",
            CharlsonCategory::HemiplegiaOrParaplegia => "Hemiplegia/Paraplegia",
            CharlsonCategory::RenalDisease => "Renal Disease",
            CharlsonCategory::MalignancyWithoutMetastasis => "Malignancy (non-metastatic)",
            CharlsonCategory::ModerateSevereLiverDisease => "Moderate/Severe Liver Disease",
            CharlsonCategory::MetastaticSolidTumor => "Metastatic Solid Tumor",
            CharlsonCategory::AIDS => "AIDS/HIV",
        };
        write!(f, "{}", name)
    }
}

/// Compute the Charlson Comorbidity Index from a list of conditions.
pub fn compute_charlson(conditions: &[Condition]) -> u32 {
    let mut seen = std::collections::HashSet::new();
    let mut score = 0u32;

    for cat in CharlsonCategory::all() {
        if seen.contains(cat) {
            continue;
        }
        for condition in conditions {
            if !condition.active {
                continue;
            }
            if cat.icd10_prefixes().iter().any(|p| condition.code.starts_with(p)) {
                score += cat.weight();
                seen.insert(*cat);
                break;
            }
        }
    }

    // Age adjustment: +1 per decade over 40, capped at 4
    score
}

/// Compute age-adjusted Charlson score.
pub fn compute_charlson_age_adjusted(conditions: &[Condition], age: f64) -> u32 {
    let base = compute_charlson(conditions);
    let age_points = if age < 50.0 {
        0
    } else if age < 60.0 {
        1
    } else if age < 70.0 {
        2
    } else if age < 80.0 {
        3
    } else {
        4
    };
    base + age_points
}

// ─────────────────────────── Elixhauser Comorbidity ─────────────────────

/// Categories used in the Elixhauser Comorbidity Index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElixhauserCategory {
    CongestiveHeartFailure,
    CardiacArrhythmias,
    ValvularDisease,
    PulmonaryCirculationDisorders,
    PeripheralVascularDisorders,
    HypertensionUncomplicated,
    HypertensionComplicated,
    Paralysis,
    OtherNeurologicalDisorders,
    ChronicPulmonaryDisease,
    DiabetesUncomplicated,
    DiabetesComplicated,
    Hypothyroidism,
    RenalFailure,
    LiverDisease,
    PepticUlcer,
    AIDS,
    Lymphoma,
    MetastaticCancer,
    SolidTumorWithoutMetastasis,
    RheumatoidArthritis,
    Coagulopathy,
    Obesity,
    WeightLoss,
    FluidElectrolyteDisorders,
    BloodLossAnemia,
    DeficiencyAnemia,
    AlcoholAbuse,
    DrugAbuse,
    Psychoses,
    Depression,
}

impl ElixhauserCategory {
    /// Van Walraven weight for the Elixhauser category (for mortality prediction).
    pub fn van_walraven_weight(&self) -> i32 {
        match self {
            ElixhauserCategory::CongestiveHeartFailure => 7,
            ElixhauserCategory::CardiacArrhythmias => 5,
            ElixhauserCategory::ValvularDisease => -1,
            ElixhauserCategory::PulmonaryCirculationDisorders => 4,
            ElixhauserCategory::PeripheralVascularDisorders => 2,
            ElixhauserCategory::HypertensionUncomplicated => 0,
            ElixhauserCategory::HypertensionComplicated => 0,
            ElixhauserCategory::Paralysis => 7,
            ElixhauserCategory::OtherNeurologicalDisorders => 6,
            ElixhauserCategory::ChronicPulmonaryDisease => 3,
            ElixhauserCategory::DiabetesUncomplicated => 0,
            ElixhauserCategory::DiabetesComplicated => 0,
            ElixhauserCategory::Hypothyroidism => 0,
            ElixhauserCategory::RenalFailure => 5,
            ElixhauserCategory::LiverDisease => 11,
            ElixhauserCategory::PepticUlcer => 0,
            ElixhauserCategory::AIDS => 0,
            ElixhauserCategory::Lymphoma => 9,
            ElixhauserCategory::MetastaticCancer => 12,
            ElixhauserCategory::SolidTumorWithoutMetastasis => 4,
            ElixhauserCategory::RheumatoidArthritis => 0,
            ElixhauserCategory::Coagulopathy => 3,
            ElixhauserCategory::Obesity => -4,
            ElixhauserCategory::WeightLoss => 6,
            ElixhauserCategory::FluidElectrolyteDisorders => 5,
            ElixhauserCategory::BloodLossAnemia => -2,
            ElixhauserCategory::DeficiencyAnemia => -2,
            ElixhauserCategory::AlcoholAbuse => 0,
            ElixhauserCategory::DrugAbuse => -7,
            ElixhauserCategory::Psychoses => 0,
            ElixhauserCategory::Depression => -3,
        }
    }

    /// ICD-10 prefixes for this category.
    pub fn icd10_prefixes(&self) -> &'static [&'static str] {
        match self {
            ElixhauserCategory::CongestiveHeartFailure => &["I09.9", "I11.0", "I13.0", "I13.2", "I25.5", "I42", "I43", "I50"],
            ElixhauserCategory::CardiacArrhythmias => &["I44", "I45", "I47", "I48", "I49", "R00.0", "R00.1", "R00.8"],
            ElixhauserCategory::ValvularDisease => &["A52.0", "I05", "I06", "I07", "I08", "I09.1", "I09.8", "I34", "I35", "I36", "I37", "I38", "I39"],
            ElixhauserCategory::PulmonaryCirculationDisorders => &["I26", "I27"],
            ElixhauserCategory::PeripheralVascularDisorders => &["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1", "I79"],
            ElixhauserCategory::HypertensionUncomplicated => &["I10"],
            ElixhauserCategory::HypertensionComplicated => &["I11", "I12", "I13", "I15"],
            ElixhauserCategory::Paralysis => &["G81", "G82", "G83.0", "G83.1", "G83.2", "G83.3", "G83.4"],
            ElixhauserCategory::OtherNeurologicalDisorders => &["G10", "G11", "G12", "G13", "G20", "G21", "G22", "G25.4", "G25.5", "G31.2", "G31.8", "G31.9", "G32", "G35", "G36", "G37", "G40", "G41", "R47.0", "R56"],
            ElixhauserCategory::ChronicPulmonaryDisease => &["I27.8", "I27.9", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67"],
            ElixhauserCategory::DiabetesUncomplicated => &["E10.0", "E10.1", "E10.9", "E11.0", "E11.1", "E11.9"],
            ElixhauserCategory::DiabetesComplicated => &["E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8"],
            ElixhauserCategory::Hypothyroidism => &["E00", "E01", "E02", "E03", "E89.0"],
            ElixhauserCategory::RenalFailure => &["I12.0", "I13.1", "N18", "N19", "N25.0", "Z49.0", "Z49.1", "Z49.2", "Z94.0", "Z99.2"],
            ElixhauserCategory::LiverDisease => &["B18", "I85", "I86.4", "I98.2", "K70", "K71.1", "K71.3", "K71.4", "K71.5", "K72", "K73", "K74", "K76.0", "K76.2", "K76.3", "K76.4", "K76.5", "K76.6", "K76.7", "K76.8", "K76.9"],
            ElixhauserCategory::PepticUlcer => &["K25.7", "K25.9", "K26.7", "K26.9", "K27.7", "K27.9", "K28.7", "K28.9"],
            ElixhauserCategory::AIDS => &["B20", "B21", "B22", "B24"],
            ElixhauserCategory::Lymphoma => &["C81", "C82", "C83", "C84", "C85", "C88", "C96"],
            ElixhauserCategory::MetastaticCancer => &["C77", "C78", "C79", "C80"],
            ElixhauserCategory::SolidTumorWithoutMetastasis => &["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"],
            ElixhauserCategory::RheumatoidArthritis => &["L94.0", "L94.1", "L94.3", "M05", "M06", "M08", "M12.0", "M12.3", "M30", "M31.0", "M31.1", "M31.2", "M31.3", "M32", "M33", "M34", "M35", "M45", "M46.1", "M46.8", "M46.9"],
            ElixhauserCategory::Coagulopathy => &["D65", "D66", "D67", "D68", "D69.1", "D69.3", "D69.4", "D69.5", "D69.6"],
            ElixhauserCategory::Obesity => &["E66"],
            ElixhauserCategory::WeightLoss => &["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R63.4", "R64"],
            ElixhauserCategory::FluidElectrolyteDisorders => &["E22.2", "E86", "E87"],
            ElixhauserCategory::BloodLossAnemia => &["D50.0"],
            ElixhauserCategory::DeficiencyAnemia => &["D50.8", "D50.9", "D51", "D52", "D53"],
            ElixhauserCategory::AlcoholAbuse => &["F10", "E52", "G62.1", "I42.6", "K29.2", "K70.0", "K70.3", "T51", "Z50.2", "Z71.4", "Z72.1"],
            ElixhauserCategory::DrugAbuse => &["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z71.5", "Z72.2"],
            ElixhauserCategory::Psychoses => &["F20", "F22", "F23", "F24", "F25", "F28", "F29", "F30.2", "F31.2", "F31.5"],
            ElixhauserCategory::Depression => &["F20.4", "F31.3", "F31.4", "F31.5", "F32", "F33", "F34.1", "F41.2", "F43.2"],
        }
    }

    /// All Elixhauser categories.
    pub fn all() -> &'static [ElixhauserCategory] {
        &[
            ElixhauserCategory::CongestiveHeartFailure,
            ElixhauserCategory::CardiacArrhythmias,
            ElixhauserCategory::ValvularDisease,
            ElixhauserCategory::PulmonaryCirculationDisorders,
            ElixhauserCategory::PeripheralVascularDisorders,
            ElixhauserCategory::HypertensionUncomplicated,
            ElixhauserCategory::HypertensionComplicated,
            ElixhauserCategory::Paralysis,
            ElixhauserCategory::OtherNeurologicalDisorders,
            ElixhauserCategory::ChronicPulmonaryDisease,
            ElixhauserCategory::DiabetesUncomplicated,
            ElixhauserCategory::DiabetesComplicated,
            ElixhauserCategory::Hypothyroidism,
            ElixhauserCategory::RenalFailure,
            ElixhauserCategory::LiverDisease,
            ElixhauserCategory::PepticUlcer,
            ElixhauserCategory::AIDS,
            ElixhauserCategory::Lymphoma,
            ElixhauserCategory::MetastaticCancer,
            ElixhauserCategory::SolidTumorWithoutMetastasis,
            ElixhauserCategory::RheumatoidArthritis,
            ElixhauserCategory::Coagulopathy,
            ElixhauserCategory::Obesity,
            ElixhauserCategory::WeightLoss,
            ElixhauserCategory::FluidElectrolyteDisorders,
            ElixhauserCategory::BloodLossAnemia,
            ElixhauserCategory::DeficiencyAnemia,
            ElixhauserCategory::AlcoholAbuse,
            ElixhauserCategory::DrugAbuse,
            ElixhauserCategory::Psychoses,
            ElixhauserCategory::Depression,
        ]
    }
}

impl fmt::Display for ElixhauserCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Compute the Elixhauser Comorbidity Index (Van Walraven composite).
pub fn compute_elixhauser(conditions: &[Condition]) -> i32 {
    let mut seen = std::collections::HashSet::new();
    let mut score = 0i32;

    for cat in ElixhauserCategory::all() {
        if seen.contains(cat) {
            continue;
        }
        for condition in conditions {
            if !condition.active {
                continue;
            }
            if cat.icd10_prefixes().iter().any(|p| condition.code.starts_with(p)) {
                score += cat.van_walraven_weight();
                seen.insert(*cat);
                break;
            }
        }
    }

    score
}

// ─────────────────────────── Medicare Prevalence ─────────────────────────

/// Medicare prevalence data for a condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComorbidityPrevalence {
    pub condition: String,
    pub icd10_prefix: String,
    /// Overall prevalence rate (proportion 0..1).
    pub prevalence_rate: f64,
    /// Age-adjusted rate.
    pub age_adjusted_rate: f64,
    /// Sex-specific rates: (male_rate, female_rate).
    pub sex_adjusted_rates: (f64, f64),
    /// Prevalence among Medicare beneficiaries ≥65.
    pub medicare_65_plus_rate: f64,
}

impl ComorbidityPrevalence {
    pub fn new(
        condition: &str,
        icd10_prefix: &str,
        prevalence: f64,
        age_adjusted: f64,
        male: f64,
        female: f64,
        medicare_65_plus: f64,
    ) -> Self {
        ComorbidityPrevalence {
            condition: condition.to_string(),
            icd10_prefix: icd10_prefix.to_string(),
            prevalence_rate: prevalence,
            age_adjusted_rate: age_adjusted,
            sex_adjusted_rates: (male, female),
            medicare_65_plus_rate: medicare_65_plus,
        }
    }

    /// Prevalence for a specific sex.
    pub fn prevalence_for_sex(&self, sex: Sex) -> f64 {
        match sex {
            Sex::Male => self.sex_adjusted_rates.0,
            Sex::Female => self.sex_adjusted_rates.1,
        }
    }
}

impl fmt::Display for ComorbidityPrevalence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}): prevalence {:.1}%",
            self.condition, self.icd10_prefix,
            self.prevalence_rate * 100.0,
        )
    }
}

/// Medicare prevalence data store.
#[derive(Debug, Clone)]
pub struct MedicarePrevalenceData {
    data: HashMap<String, ComorbidityPrevalence>,
}

impl MedicarePrevalenceData {
    pub fn empty() -> Self {
        MedicarePrevalenceData { data: HashMap::new() }
    }

    pub fn with_defaults() -> Self {
        let mut d = Self::empty();
        d.load_defaults();
        d
    }

    pub fn insert(&mut self, prev: ComorbidityPrevalence) {
        self.data.insert(prev.icd10_prefix.clone(), prev);
    }

    pub fn get_by_prefix(&self, prefix: &str) -> Option<&ComorbidityPrevalence> {
        self.data.get(prefix)
    }

    /// Find prevalence data matching a condition's ICD-10 code.
    pub fn get_for_condition(&self, condition: &Condition) -> Option<&ComorbidityPrevalence> {
        for (prefix, prev) in &self.data {
            if condition.code.starts_with(prefix) {
                return Some(prev);
            }
        }
        None
    }

    /// Get the overall prevalence rate for a condition code.
    pub fn prevalence_for_code(&self, code: &str) -> f64 {
        for (prefix, prev) in &self.data {
            if code.starts_with(prefix) {
                return prev.prevalence_rate;
            }
        }
        0.0
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn all(&self) -> Vec<&ComorbidityPrevalence> {
        self.data.values().collect()
    }

    fn load_defaults(&mut self) {
        // Prevalence rates as proportions (e.g. 0.58 = 58%)
        // Sources: CMS Medicare Chronic Conditions Dashboard, CDC NCHS

        self.insert(ComorbidityPrevalence::new(
            "Hypertension", "I10", 0.58, 0.56, 0.56, 0.60, 0.60,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Hyperlipidemia", "E78", 0.49, 0.47, 0.47, 0.51, 0.51,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Ischemic Heart Disease", "I25", 0.29, 0.28, 0.34, 0.24, 0.30,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Diabetes Mellitus", "E11", 0.27, 0.26, 0.28, 0.26, 0.28,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Chronic Kidney Disease", "N18", 0.25, 0.24, 0.26, 0.24, 0.26,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Heart Failure", "I50", 0.14, 0.14, 0.15, 0.13, 0.15,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Atrial Fibrillation", "I48", 0.09, 0.09, 0.11, 0.08, 0.10,
        ));
        self.insert(ComorbidityPrevalence::new(
            "COPD", "J44", 0.11, 0.11, 0.10, 0.12, 0.12,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Depression", "F32", 0.16, 0.15, 0.12, 0.19, 0.17,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Alzheimer / Dementia", "G30", 0.12, 0.11, 0.09, 0.14, 0.12,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Osteoporosis", "M81", 0.07, 0.07, 0.02, 0.11, 0.08,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Rheumatoid Arthritis", "M05", 0.04, 0.04, 0.03, 0.05, 0.04,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Asthma", "J45", 0.05, 0.05, 0.04, 0.06, 0.05,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Stroke / TIA", "I63", 0.04, 0.04, 0.04, 0.04, 0.04,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Cancer (Breast)", "C50", 0.03, 0.03, 0.003, 0.06, 0.03,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Cancer (Prostate)", "C61", 0.03, 0.03, 0.06, 0.0, 0.03,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Cancer (Lung)", "C34", 0.02, 0.02, 0.02, 0.02, 0.02,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Cancer (Colorectal)", "C18", 0.02, 0.02, 0.02, 0.015, 0.02,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Peripheral Vascular Disease", "I70", 0.05, 0.05, 0.06, 0.04, 0.05,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Hypothyroidism", "E03", 0.13, 0.12, 0.07, 0.18, 0.14,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Anemia", "D64", 0.09, 0.09, 0.08, 0.10, 0.10,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Benign Prostatic Hyperplasia", "N40", 0.07, 0.07, 0.14, 0.0, 0.07,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Obesity", "E66", 0.08, 0.08, 0.07, 0.09, 0.08,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Glaucoma", "H40", 0.06, 0.06, 0.05, 0.07, 0.06,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Anxiety Disorders", "F41", 0.08, 0.08, 0.06, 0.10, 0.09,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Osteoarthritis", "M17", 0.14, 0.14, 0.11, 0.16, 0.15,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Gout", "M10", 0.04, 0.04, 0.06, 0.02, 0.04,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Liver Disease (Chronic)", "K74", 0.02, 0.02, 0.025, 0.015, 0.02,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Epilepsy", "G40", 0.02, 0.02, 0.02, 0.02, 0.02,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Schizophrenia", "F20", 0.01, 0.01, 0.012, 0.008, 0.01,
        ));
        self.insert(ComorbidityPrevalence::new(
            "HIV/AIDS", "B20", 0.005, 0.005, 0.007, 0.003, 0.005,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Peptic Ulcer Disease", "K27", 0.015, 0.015, 0.018, 0.012, 0.015,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Parkinson Disease", "G20", 0.02, 0.02, 0.025, 0.015, 0.02,
        ));
        self.insert(ComorbidityPrevalence::new(
            "Deep Vein Thrombosis", "I82", 0.02, 0.02, 0.018, 0.022, 0.02,
        ));
    }
}

// ─────────────────────────── Prevalence-weighted severity ────────────────

/// Weight a conflict's base severity by the prevalence of a condition.
///
/// This amplifies severity for conflicts that affect commonly co-occurring
/// conditions: a moderate interaction is more concerning if the affected
/// condition is present in 25% of the target population.
///
/// Formula: `adjusted = base_severity * (1.0 + log10(1 + prevalence * 100))`
pub fn weight_by_prevalence(conflict_severity: f64, condition_prevalence: f64) -> f64 {
    let prev_boost = (1.0 + condition_prevalence * 100.0).log10();
    conflict_severity * (1.0 + prev_boost)
}

/// Compute a polypharmacy risk score for a patient.
///
/// Combines:
/// - Number of medications (polypharmacy ≥5, excessive ≥10)
/// - Charlson Comorbidity Index
/// - Age
/// - Renal function
///
/// Returns a score in [0, 1].
pub fn compute_polypharmacy_risk_score(patient: &PatientProfile) -> f64 {
    let med_count = patient.medication_count() as f64;
    let med_score = if med_count >= 10.0 {
        1.0
    } else if med_count >= 5.0 {
        0.5 + (med_count - 5.0) * 0.1
    } else {
        med_count * 0.1
    };

    let cci = compute_charlson_age_adjusted(&patient.conditions, patient.age()) as f64;
    let cci_score = (cci / 10.0).min(1.0);

    let age_score = if patient.age() >= 85.0 {
        1.0
    } else if patient.age() >= 75.0 {
        0.7
    } else if patient.age() >= 65.0 {
        0.4
    } else {
        0.1
    };

    let renal_score = match patient.renal_function() {
        crate::RenalFunction::Normal => 0.0,
        crate::RenalFunction::Mild => 0.2,
        crate::RenalFunction::Moderate => 0.5,
        crate::RenalFunction::Severe => 0.8,
        crate::RenalFunction::EndStage => 1.0,
    };

    // Weighted combination
    let raw = 0.30 * med_score + 0.30 * cci_score + 0.25 * age_score + 0.15 * renal_score;
    raw.min(1.0)
}

// ─────────────────────────── Population Impact Score ─────────────────────

/// Estimate the population-level impact of a drug–drug conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationImpactScore {
    pub conflict_id: String,
    /// Number of patients in the population likely affected.
    pub estimated_affected: f64,
    /// Severity-weighted impact.
    pub severity_weighted_impact: f64,
    /// Prevalence of the most relevant comorbidity.
    pub max_comorbidity_prevalence: f64,
    /// QALY impact estimate (negative = harm).
    pub estimated_qaly_impact: f64,
}

impl PopulationImpactScore {
    /// Compute population impact for a conflict given the patient population size.
    pub fn compute(
        conflict_id: &str,
        conflict_severity: f64,
        patient_conditions: &[Condition],
        population_size: f64,
        prevalence_data: &MedicarePrevalenceData,
    ) -> Self {
        let mut max_prev = 0.0_f64;
        for condition in patient_conditions {
            if !condition.active {
                continue;
            }
            let prev = prevalence_data.prevalence_for_code(&condition.code);
            max_prev = max_prev.max(prev);
        }

        // Estimated affected = population × max relevant comorbidity prevalence
        // (this is a rough heuristic; the real computation would use claims data)
        let estimated_affected = population_size * max_prev.max(0.01);

        let severity_weighted = weight_by_prevalence(conflict_severity, max_prev);
        let qaly_impact = -severity_weighted * estimated_affected * 0.001;

        PopulationImpactScore {
            conflict_id: conflict_id.to_string(),
            estimated_affected,
            severity_weighted_impact: severity_weighted,
            max_comorbidity_prevalence: max_prev,
            estimated_qaly_impact: qaly_impact,
        }
    }
}

impl fmt::Display for PopulationImpactScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: affected≈{:.0}, severity_weighted={:.3}, QALY≈{:.2}",
            self.conflict_id, self.estimated_affected,
            self.severity_weighted_impact, self.estimated_qaly_impact,
        )
    }
}

/// Compute the comorbidity component score for the composite scorer.
///
/// Combines polypharmacy risk, Charlson index, and prevalence weighting.
/// Returns a value in [0, 1].
pub fn compute_comorbidity_component(
    patient: &PatientProfile,
    conflict_severity: f64,
    prevalence_data: &MedicarePrevalenceData,
) -> f64 {
    let polypharmacy_risk = compute_polypharmacy_risk_score(patient);

    let mut max_prev = 0.0_f64;
    for condition in &patient.conditions {
        if !condition.active {
            continue;
        }
        let prev = prevalence_data.prevalence_for_code(&condition.code);
        max_prev = max_prev.max(prev);
    }

    let weighted_severity = weight_by_prevalence(conflict_severity, max_prev);

    // Normalize to [0, 1]: combine polypharmacy risk and prevalence-weighted severity
    let raw = 0.5 * polypharmacy_risk + 0.5 * (weighted_severity / 3.0).min(1.0);
    raw.min(1.0)
}

// ──────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Medication;
    

    fn test_patient() -> PatientProfile {
        PatientProfile::new(
            75.0, 70.0, Sex::Male,
        )
        .with_conditions(vec![
            Condition::new("I10", "Hypertension"),
            Condition::new("E11.9", "Type 2 Diabetes without complications"),
            Condition::new("N18.3", "Chronic kidney disease stage 3"),
            Condition::new("I50.9", "Heart failure"),
        ])
        .with_medications(vec![
            Medication::new("Lisinopril", "ACE inhibitor", 10.0),
            Medication::new("Metformin", "biguanide", 500.0),
            Medication::new("Atorvastatin", "statin", 40.0),
            Medication::new("Furosemide", "loop diuretic", 40.0),
            Medication::new("Aspirin", "antiplatelet", 81.0),
        ])
        .with_egfr(45.0)
    }

    #[test]
    fn test_charlson_basic() {
        let conditions = vec![
            Condition::new("I50.9", "Heart failure"),
            Condition::new("E11.9", "Type 2 Diabetes"),
        ];
        let score = compute_charlson(&conditions);
        // CHF (1) + DM uncomplicated (1) = 2
        assert_eq!(score, 2);
    }

    #[test]
    fn test_charlson_complex() {
        let conditions = vec![
            Condition::new("I50.9", "Heart failure"),        // 1
            Condition::new("N18.4", "CKD stage 4"),          // 2
            Condition::new("C78.0", "Metastatic lung cancer"), // 6
            Condition::new("G30", "Alzheimer dementia"),     // 1
        ];
        let score = compute_charlson(&conditions);
        assert_eq!(score, 10);
    }

    #[test]
    fn test_charlson_age_adjusted() {
        let conditions = vec![
            Condition::new("I50.9", "Heart failure"),
        ];
        let base = compute_charlson(&conditions);
        let adjusted = compute_charlson_age_adjusted(&conditions, 75.0);
        assert_eq!(adjusted, base + 3); // age 75 → +3
    }

    #[test]
    fn test_charlson_empty() {
        assert_eq!(compute_charlson(&[]), 0);
    }

    #[test]
    fn test_elixhauser_basic() {
        let conditions = vec![
            Condition::new("I50.9", "Heart failure"),    // +7
            Condition::new("I48", "Atrial fibrillation"), // +5
        ];
        let score = compute_elixhauser(&conditions);
        assert_eq!(score, 12);
    }

    #[test]
    fn test_elixhauser_with_negative_weights() {
        let conditions = vec![
            Condition::new("I50.9", "Heart failure"), // +7
            Condition::new("E66", "Obesity"),          // -4
        ];
        let score = compute_elixhauser(&conditions);
        assert_eq!(score, 3);
    }

    #[test]
    fn test_elixhauser_empty() {
        assert_eq!(compute_elixhauser(&[]), 0);
    }

    #[test]
    fn test_medicare_prevalence_not_empty() {
        let d = MedicarePrevalenceData::with_defaults();
        assert!(d.len() >= 30, "Expected ≥30 conditions, got {}", d.len());
    }

    #[test]
    fn test_hypertension_prevalence() {
        let d = MedicarePrevalenceData::with_defaults();
        let p = d.get_by_prefix("I10").unwrap();
        assert!(p.prevalence_rate > 0.5, "Hypertension prevalence should be >50%");
    }

    #[test]
    fn test_prevalence_for_condition() {
        let d = MedicarePrevalenceData::with_defaults();
        let c = Condition::new("I10", "Hypertension");
        let p = d.get_for_condition(&c);
        assert!(p.is_some());
    }

    #[test]
    fn test_weight_by_prevalence() {
        let base = 0.5;
        let weighted = weight_by_prevalence(base, 0.0);
        // With 0 prevalence: log10(1) = 0, so adjusted = 0.5 * (1 + 0) = 0.5
        assert!((weighted - 0.5).abs() < 1e-10);

        let high = weight_by_prevalence(base, 0.5);
        assert!(high > weighted, "Higher prevalence should increase score");
    }

    #[test]
    fn test_polypharmacy_risk_score() {
        let patient = test_patient();
        let score = compute_polypharmacy_risk_score(&patient);
        assert!(score > 0.0 && score <= 1.0, "Score should be in (0, 1], got {}", score);
    }

    #[test]
    fn test_polypharmacy_high_meds() {
        let patient = PatientProfile::default().with_medications(
            (0..12).map(|i| Medication::new(&format!("Drug{}", i), "misc", 10.0)).collect()
        );
        let score = compute_polypharmacy_risk_score(&patient);
        assert!(score > 0.3, "12 medications should give higher risk");
    }

    #[test]
    fn test_population_impact_score() {
        let d = MedicarePrevalenceData::with_defaults();
        let conditions = vec![
            Condition::new("I10", "Hypertension"),
            Condition::new("E11", "Diabetes"),
        ];
        let impact = PopulationImpactScore::compute(
            "test_conflict",
            0.8,
            &conditions,
            100_000.0,
            &d,
        );
        assert!(impact.estimated_affected > 0.0);
        assert!(impact.estimated_qaly_impact < 0.0); // Negative = harm
    }

    #[test]
    fn test_comorbidity_component() {
        let patient = test_patient();
        let d = MedicarePrevalenceData::with_defaults();
        let score = compute_comorbidity_component(&patient, 0.8, &d);
        assert!(score > 0.0 && score <= 1.0, "Component score should be in (0, 1], got {}", score);
    }

    #[test]
    fn test_charlson_category_weights() {
        assert_eq!(CharlsonCategory::MetastaticSolidTumor.weight(), 6);
        assert_eq!(CharlsonCategory::MyocardialInfarction.weight(), 1);
        assert_eq!(CharlsonCategory::RenalDisease.weight(), 2);
    }

    #[test]
    fn test_prevalence_sex_specific() {
        let d = MedicarePrevalenceData::with_defaults();
        let hyp = d.get_by_prefix("I10").unwrap();
        let male_rate = hyp.prevalence_for_sex(Sex::Male);
        let female_rate = hyp.prevalence_for_sex(Sex::Female);
        assert!(male_rate > 0.0);
        assert!(female_rate > 0.0);
    }

    #[test]
    fn test_inactive_conditions_ignored() {
        let conditions = vec![
            Condition::inactive("I50.9", "Heart failure (resolved)"),
        ];
        let score = compute_charlson(&conditions);
        assert_eq!(score, 0, "Inactive conditions should not contribute");
    }

    #[test]
    fn test_prevalence_display() {
        let p = ComorbidityPrevalence::new("Test", "T00", 0.25, 0.24, 0.26, 0.24, 0.26);
        let s = format!("{}", p);
        assert!(s.contains("25.0%"));
    }
}
