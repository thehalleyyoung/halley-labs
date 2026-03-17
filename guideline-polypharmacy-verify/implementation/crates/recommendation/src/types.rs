//! Local shared types for the recommendation crate.
//!
//! These types are defined locally so modules compile independently without
//! relying on imports from other guardpharma crates.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Newtypes
// ---------------------------------------------------------------------------

/// Drug identifier newtype.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DrugId(String);

impl DrugId {
    /// Create a new drug id, normalising to lowercase with underscores.
    pub fn new(s: impl Into<String>) -> Self {
        let raw: String = s.into();
        let normalised = raw
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
            .collect();
        DrugId(normalised)
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

impl FromStr for DrugId {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(DrugId::new(s))
    }
}

impl From<&str> for DrugId {
    fn from(s: &str) -> Self {
        DrugId::new(s)
    }
}

impl From<String> for DrugId {
    fn from(s: String) -> Self {
        DrugId::new(s)
    }
}

/// Conflict identifier newtype.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConflictId(String);

impl ConflictId {
    pub fn new(s: impl Into<String>) -> Self {
        ConflictId(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConflictId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ConflictId {
    fn from(s: &str) -> Self {
        ConflictId(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Severity of a drug-drug conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Minor = 0,
    Moderate = 1,
    Major = 2,
    Critical = 3,
}

impl ConflictSeverity {
    /// Numeric score (1–4).
    pub fn numeric_score(self) -> u32 {
        match self {
            Self::Minor => 1,
            Self::Moderate => 2,
            Self::Major => 3,
            Self::Critical => 4,
        }
    }

    /// Whether this severity requires immediate clinical action.
    pub fn requires_immediate_action(self) -> bool {
        matches!(self, Self::Critical | Self::Major)
    }

    /// Priority weight for recommendation ordering.
    pub fn priority_weight(self) -> f64 {
        match self {
            Self::Critical => 4.0,
            Self::Major => 3.0,
            Self::Moderate => 2.0,
            Self::Minor => 1.0,
        }
    }
}

impl fmt::Display for ConflictSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minor => write!(f, "Minor"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Major => write!(f, "Major"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

impl FromStr for ConflictSeverity {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "minor" => Ok(Self::Minor),
            "moderate" => Ok(Self::Moderate),
            "major" => Ok(Self::Major),
            "critical" => Ok(Self::Critical),
            _ => Err(format!("Unknown severity: {}", s)),
        }
    }
}

/// Interaction type between two drugs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    CypInhibition { enzyme: String },
    CypInduction { enzyme: String },
    ProteinBindingDisplacement,
    RenalCompetition,
    PharmacodynamicSynergy,
    PharmacodynamicAntagonism,
    AbsorptionAlteration,
    QtProlongation,
}

impl fmt::Display for InteractionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CypInhibition { enzyme } => write!(f, "CYP inhibition ({})", enzyme),
            Self::CypInduction { enzyme } => write!(f, "CYP induction ({})", enzyme),
            Self::ProteinBindingDisplacement => write!(f, "Protein binding displacement"),
            Self::RenalCompetition => write!(f, "Renal competition"),
            Self::PharmacodynamicSynergy => write!(f, "PD synergy"),
            Self::PharmacodynamicAntagonism => write!(f, "PD antagonism"),
            Self::AbsorptionAlteration => write!(f, "Absorption alteration"),
            Self::QtProlongation => write!(f, "QT prolongation"),
        }
    }
}

/// Safety verdict for a drug combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyVerdict {
    Safe,
    PossiblySafe,
    PossiblyUnsafe,
    Unsafe,
}

impl SafetyVerdict {
    /// Whether this verdict allows the combination to proceed (with monitoring).
    pub fn is_acceptable(self) -> bool {
        matches!(self, Self::Safe | Self::PossiblySafe)
    }

    /// Numeric risk score (0 = safe, 3 = unsafe).
    pub fn risk_score(self) -> u32 {
        match self {
            Self::Safe => 0,
            Self::PossiblySafe => 1,
            Self::PossiblyUnsafe => 2,
            Self::Unsafe => 3,
        }
    }
}

impl fmt::Display for SafetyVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safe => write!(f, "Safe"),
            Self::PossiblySafe => write!(f, "Possibly Safe"),
            Self::PossiblyUnsafe => write!(f, "Possibly Unsafe"),
            Self::Unsafe => write!(f, "Unsafe"),
        }
    }
}

/// Drug therapeutic class.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrugClass {
    NSAID,
    Anticoagulant,
    Antihypertensive,
    Antidiabetic,
    Statin,
    Antidepressant,
    SSRI,
    Antibiotic,
    Antiarrhythmic,
    Opioid,
    Benzodiazepine,
    PPI,
    Anticonvulsant,
    Corticosteroid,
    Immunosuppressant,
    Bronchodilator,
    Diuretic,
    ACEInhibitor,
    ARB,
    BetaBlocker,
    CalciumChannelBlocker,
    Antiplatelet,
    Antipsychotic,
    Sedative,
    Antifungal,
    Antiviral,
    Other(String),
}

impl DrugClass {
    /// Whether this drug class requires therapeutic monitoring.
    pub fn requires_monitoring(&self) -> bool {
        matches!(
            self,
            Self::Anticoagulant
                | Self::Antiarrhythmic
                | Self::Anticonvulsant
                | Self::Immunosuppressant
                | Self::Opioid
        )
    }

    /// Whether two drug classes are in the same therapeutic area.
    pub fn same_therapeutic_area(&self, other: &DrugClass) -> bool {
        self.therapeutic_area() == other.therapeutic_area()
    }

    /// Broad therapeutic area name.
    pub fn therapeutic_area(&self) -> &str {
        match self {
            Self::Antihypertensive | Self::ACEInhibitor | Self::ARB
            | Self::BetaBlocker | Self::CalciumChannelBlocker | Self::Diuretic => "Cardiovascular",
            Self::Statin => "Lipid-lowering",
            Self::Antidepressant | Self::SSRI | Self::Antipsychotic
            | Self::Benzodiazepine | Self::Sedative => "CNS/Psychiatry",
            Self::Anticoagulant | Self::Antiplatelet => "Hematology",
            Self::NSAID => "Pain/Inflammation",
            Self::Opioid => "Pain/Analgesia",
            Self::Antibiotic | Self::Antifungal | Self::Antiviral => "Anti-infective",
            Self::PPI => "Gastrointestinal",
            Self::Antidiabetic => "Endocrine",
            Self::Corticosteroid => "Anti-inflammatory/Immune",
            Self::Immunosuppressant => "Immunology",
            Self::Bronchodilator => "Respiratory",
            Self::Anticonvulsant | Self::Antiarrhythmic => "Neurology/Cardiology",
            Self::Other(_) => "Other",
        }
    }

    /// Whether this drug class has CNS activity.
    pub fn is_cns_active(&self) -> bool {
        matches!(
            self,
            Self::Antidepressant
                | Self::SSRI
                | Self::Antipsychotic
                | Self::Benzodiazepine
                | Self::Sedative
                | Self::Opioid
                | Self::Anticonvulsant
        )
    }
}

impl fmt::Display for DrugClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Other(name) => write!(f, "{}", name),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Renal function classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenalFunction {
    Normal,
    MildImpairment,
    ModerateImpairment,
    SevereImpairment,
    EndStage,
}

impl RenalFunction {
    /// eGFR range for this classification.
    pub fn egfr_range(self) -> (f64, f64) {
        match self {
            Self::Normal => (90.0, 150.0),
            Self::MildImpairment => (60.0, 89.0),
            Self::ModerateImpairment => (30.0, 59.0),
            Self::SevereImpairment => (15.0, 29.0),
            Self::EndStage => (0.0, 14.0),
        }
    }

    /// Whether dose adjustment is typically required.
    pub fn requires_dose_adjustment(self) -> bool {
        matches!(
            self,
            Self::ModerateImpairment | Self::SevereImpairment | Self::EndStage
        )
    }

    /// Dose reduction factor (1.0 = no reduction).
    pub fn dose_factor(self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::MildImpairment => 0.9,
            Self::ModerateImpairment => 0.65,
            Self::SevereImpairment => 0.4,
            Self::EndStage => 0.25,
        }
    }
}

impl fmt::Display for RenalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal (eGFR ≥90)"),
            Self::MildImpairment => write!(f, "Mild impairment (eGFR 60–89)"),
            Self::ModerateImpairment => write!(f, "Moderate impairment (eGFR 30–59)"),
            Self::SevereImpairment => write!(f, "Severe impairment (eGFR 15–29)"),
            Self::EndStage => write!(f, "End-stage (eGFR <15)"),
        }
    }
}

/// Hepatic function classification (Child-Pugh based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HepaticFunction {
    Normal,
    MildImpairment,
    ModerateImpairment,
    SevereImpairment,
}

impl HepaticFunction {
    /// Child-Pugh class.
    pub fn child_pugh_class(self) -> &'static str {
        match self {
            Self::Normal => "None",
            Self::MildImpairment => "A",
            Self::ModerateImpairment => "B",
            Self::SevereImpairment => "C",
        }
    }

    /// Whether dose adjustment is typically needed.
    pub fn requires_dose_adjustment(self) -> bool {
        matches!(self, Self::ModerateImpairment | Self::SevereImpairment)
    }

    /// Dose reduction factor (1.0 = no reduction).
    pub fn dose_factor(self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::MildImpairment => 0.85,
            Self::ModerateImpairment => 0.6,
            Self::SevereImpairment => 0.35,
        }
    }
}

impl fmt::Display for HepaticFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::MildImpairment => write!(f, "Mild (Child-Pugh A)"),
            Self::ModerateImpairment => write!(f, "Moderate (Child-Pugh B)"),
            Self::SevereImpairment => write!(f, "Severe (Child-Pugh C)"),
        }
    }
}

/// Biological sex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
    Other,
    Unknown,
}

/// Evidence level for a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceLevel {
    /// Randomised controlled trials / meta-analyses.
    High,
    /// Observational studies / strong clinical consensus.
    Moderate,
    /// Case reports / expert opinion.
    Low,
    /// Theoretical / extrapolated.
    VeryLow,
}

impl EvidenceLevel {
    pub fn confidence_factor(self) -> f64 {
        match self {
            Self::High => 0.95,
            Self::Moderate => 0.75,
            Self::Low => 0.50,
            Self::VeryLow => 0.25,
        }
    }
}

impl fmt::Display for EvidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::High => write!(f, "High"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Low => write!(f, "Low"),
            Self::VeryLow => write!(f, "Very Low"),
        }
    }
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// A confirmed drug-drug conflict (local representation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmedConflict {
    pub id: ConflictId,
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub interaction_type: InteractionType,
    pub severity: ConflictSeverity,
    pub verdict: SafetyVerdict,
    pub mechanism: String,
    pub confidence: f64,
    pub clinical_recommendation: String,
    pub affected_parameters: Vec<String>,
}

impl ConfirmedConflict {
    pub fn new(
        drug_a: DrugId,
        drug_b: DrugId,
        interaction_type: InteractionType,
        severity: ConflictSeverity,
    ) -> Self {
        let id = ConflictId::new(format!("{}_{}", drug_a, drug_b));
        let verdict = match severity {
            ConflictSeverity::Critical => SafetyVerdict::Unsafe,
            ConflictSeverity::Major => SafetyVerdict::PossiblyUnsafe,
            ConflictSeverity::Moderate => SafetyVerdict::PossiblySafe,
            ConflictSeverity::Minor => SafetyVerdict::Safe,
        };
        ConfirmedConflict {
            id,
            drug_a,
            drug_b,
            interaction_type,
            severity,
            verdict,
            mechanism: String::new(),
            confidence: 0.8,
            clinical_recommendation: String::new(),
            affected_parameters: Vec::new(),
        }
    }

    /// Builder: set mechanism description.
    pub fn with_mechanism(mut self, mechanism: &str) -> Self {
        self.mechanism = mechanism.to_string();
        self
    }

    /// Builder: set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Whether this conflict involves a specific drug.
    pub fn involves(&self, drug: &DrugId) -> bool {
        self.drug_a == *drug || self.drug_b == *drug
    }

    /// Get the other drug in the conflict.
    pub fn other_drug(&self, drug: &DrugId) -> Option<&DrugId> {
        if self.drug_a == *drug {
            Some(&self.drug_b)
        } else if self.drug_b == *drug {
            Some(&self.drug_a)
        } else {
            None
        }
    }
}

/// Dosing schedule for a single medication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DosingSchedule {
    pub interval_hours: f64,
    pub dose_amount_mg: f64,
    pub route: String,
    pub max_daily_dose_mg: Option<f64>,
    pub doses_per_day: u32,
}

impl DosingSchedule {
    pub fn new(dose_mg: f64, doses_per_day: u32) -> Self {
        let interval = if doses_per_day > 0 {
            24.0 / doses_per_day as f64
        } else {
            24.0
        };
        DosingSchedule {
            interval_hours: interval,
            dose_amount_mg: dose_mg,
            route: "oral".to_string(),
            max_daily_dose_mg: None,
            doses_per_day,
        }
    }

    pub fn oral(dose_mg: f64, times_per_day: u32) -> Self {
        Self::new(dose_mg, times_per_day)
    }

    pub fn with_max_daily(mut self, max_mg: f64) -> Self {
        self.max_daily_dose_mg = Some(max_mg);
        self
    }

    /// Total daily dose.
    pub fn daily_dose(&self) -> f64 {
        self.dose_amount_mg * self.doses_per_day as f64
    }

    /// Whether the daily dose exceeds the maximum.
    pub fn exceeds_max_daily(&self) -> bool {
        if let Some(max) = self.max_daily_dose_mg {
            self.daily_dose() > max
        } else {
            false
        }
    }
}

/// Active medication on a patient's profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveMedication {
    pub drug_id: DrugId,
    pub drug_class: DrugClass,
    pub dose_schedule: DosingSchedule,
    pub indication: Option<String>,
    pub prescriber: Option<String>,
}

impl ActiveMedication {
    pub fn new(drug_id: DrugId, drug_class: DrugClass, dose_mg: f64, times_per_day: u32) -> Self {
        ActiveMedication {
            drug_id,
            drug_class,
            dose_schedule: DosingSchedule::oral(dose_mg, times_per_day),
            indication: None,
            prescriber: None,
        }
    }

    pub fn with_indication(mut self, indication: &str) -> Self {
        self.indication = Some(indication.to_string());
        self
    }
}

/// Patient clinical profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientProfile {
    pub id: String,
    pub age_years: u32,
    pub weight_kg: f64,
    pub sex: Sex,
    pub renal_function: RenalFunction,
    pub hepatic_function: HepaticFunction,
    pub medications: Vec<ActiveMedication>,
    pub allergies: Vec<String>,
    pub conditions: Vec<String>,
    pub egfr: Option<f64>,
}

impl PatientProfile {
    pub fn new(id: &str, age: u32, weight_kg: f64) -> Self {
        PatientProfile {
            id: id.to_string(),
            age_years: age,
            weight_kg,
            sex: Sex::Unknown,
            renal_function: RenalFunction::Normal,
            hepatic_function: HepaticFunction::Normal,
            medications: Vec::new(),
            allergies: Vec::new(),
            conditions: Vec::new(),
            egfr: None,
        }
    }

    pub fn with_sex(mut self, sex: Sex) -> Self {
        self.sex = sex;
        self
    }

    pub fn with_renal_function(mut self, rf: RenalFunction) -> Self {
        self.renal_function = rf;
        self
    }

    pub fn with_hepatic_function(mut self, hf: HepaticFunction) -> Self {
        self.hepatic_function = hf;
        self
    }

    pub fn add_medication(&mut self, med: ActiveMedication) {
        self.medications.push(med);
    }

    pub fn add_allergy(&mut self, allergy: &str) {
        self.allergies.push(allergy.to_string());
    }

    /// Whether the patient has a specific allergy (case-insensitive).
    pub fn has_allergy(&self, substance: &str) -> bool {
        let lower = substance.to_lowercase();
        self.allergies.iter().any(|a| a.to_lowercase().contains(&lower))
    }

    /// Find medication by drug id.
    pub fn find_medication(&self, drug_id: &DrugId) -> Option<&ActiveMedication> {
        self.medications.iter().find(|m| m.drug_id == *drug_id)
    }

    /// Body surface area (Mosteller formula).
    pub fn bsa(&self) -> f64 {
        // Requires height; approximate from weight if not available.
        let height_cm = 170.0; // default approximation
        ((self.weight_kg * height_cm) / 3600.0).sqrt()
    }

    /// Creatinine clearance (Cockcroft-Gault approximation).
    pub fn estimated_crcl(&self) -> f64 {
        if let Some(egfr) = self.egfr {
            return egfr;
        }
        let sex_factor = match self.sex {
            Sex::Female => 0.85,
            _ => 1.0,
        };
        // Approximate using age and weight; assume serum creatinine ~1.0
        let scr = 1.0;
        ((140.0 - self.age_years as f64) * self.weight_kg * sex_factor) / (72.0 * scr)
    }
}

/// Pharmacokinetic profile for a drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkProfile {
    pub drug_id: DrugId,
    pub half_life_hours: f64,
    pub tmax_hours: f64,
    pub bioavailability: f64,
    pub clearance_l_per_h: f64,
    pub volume_distribution_l: f64,
    pub protein_binding: f64,
    pub renal_elimination_fraction: f64,
    pub hepatic_extraction_ratio: f64,
}

impl PkProfile {
    pub fn new(drug_id: DrugId, half_life: f64, tmax: f64) -> Self {
        PkProfile {
            drug_id,
            half_life_hours: half_life,
            tmax_hours: tmax,
            bioavailability: 0.8,
            clearance_l_per_h: 5.0,
            volume_distribution_l: 50.0,
            protein_binding: 0.9,
            renal_elimination_fraction: 0.3,
            hepatic_extraction_ratio: 0.3,
        }
    }

    /// Elimination rate constant (ke = ln(2) / t½).
    pub fn elimination_rate(&self) -> f64 {
        if self.half_life_hours > 0.0 {
            (2.0_f64).ln() / self.half_life_hours
        } else {
            0.0
        }
    }

    /// Predicted steady-state average concentration for a given dose/interval.
    pub fn predicted_css_avg(&self, dose_mg: f64, interval_h: f64) -> f64 {
        if self.clearance_l_per_h > 0.0 && interval_h > 0.0 {
            (self.bioavailability * dose_mg) / (self.clearance_l_per_h * interval_h)
        } else {
            0.0
        }
    }

    /// Time to reach ~90% steady state (≈3.3 half-lives).
    pub fn time_to_steady_state_hours(&self) -> f64 {
        3.3 * self.half_life_hours
    }
}

/// Lookup of PK profiles by drug id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PkDatabase {
    pub entries: HashMap<DrugId, PkProfile>,
}

impl PkDatabase {
    pub fn new() -> Self {
        PkDatabase {
            entries: HashMap::new(),
        }
    }

    pub fn add(&mut self, profile: PkProfile) {
        self.entries.insert(profile.drug_id.clone(), profile);
    }

    pub fn get(&self, drug_id: &DrugId) -> Option<&PkProfile> {
        self.entries.get(drug_id)
    }

    /// Build a small demonstration database.
    pub fn demo() -> Self {
        let mut db = PkDatabase::new();
        let entries = vec![
            PkProfile {
                drug_id: DrugId::new("atorvastatin"),
                half_life_hours: 14.0,
                tmax_hours: 1.5,
                bioavailability: 0.14,
                clearance_l_per_h: 38.0,
                volume_distribution_l: 381.0,
                protein_binding: 0.98,
                renal_elimination_fraction: 0.02,
                hepatic_extraction_ratio: 0.70,
            },
            PkProfile {
                drug_id: DrugId::new("metoprolol"),
                half_life_hours: 4.0,
                tmax_hours: 1.5,
                bioavailability: 0.50,
                clearance_l_per_h: 60.0,
                volume_distribution_l: 290.0,
                protein_binding: 0.12,
                renal_elimination_fraction: 0.05,
                hepatic_extraction_ratio: 0.70,
            },
            PkProfile {
                drug_id: DrugId::new("amlodipine"),
                half_life_hours: 35.0,
                tmax_hours: 8.0,
                bioavailability: 0.64,
                clearance_l_per_h: 7.0,
                volume_distribution_l: 21.0 * 70.0,
                protein_binding: 0.98,
                renal_elimination_fraction: 0.10,
                hepatic_extraction_ratio: 0.20,
            },
            PkProfile {
                drug_id: DrugId::new("omeprazole"),
                half_life_hours: 1.0,
                tmax_hours: 1.0,
                bioavailability: 0.40,
                clearance_l_per_h: 30.0,
                volume_distribution_l: 17.0,
                protein_binding: 0.95,
                renal_elimination_fraction: 0.20,
                hepatic_extraction_ratio: 0.50,
            },
            PkProfile {
                drug_id: DrugId::new("warfarin"),
                half_life_hours: 40.0,
                tmax_hours: 4.0,
                bioavailability: 1.0,
                clearance_l_per_h: 0.20,
                volume_distribution_l: 10.0,
                protein_binding: 0.99,
                renal_elimination_fraction: 0.02,
                hepatic_extraction_ratio: 0.01,
            },
            PkProfile {
                drug_id: DrugId::new("lisinopril"),
                half_life_hours: 12.0,
                tmax_hours: 7.0,
                bioavailability: 0.25,
                clearance_l_per_h: 3.0,
                volume_distribution_l: 124.0,
                protein_binding: 0.0,
                renal_elimination_fraction: 1.0,
                hepatic_extraction_ratio: 0.0,
            },
            PkProfile {
                drug_id: DrugId::new("sertraline"),
                half_life_hours: 26.0,
                tmax_hours: 6.0,
                bioavailability: 0.44,
                clearance_l_per_h: 96.0,
                volume_distribution_l: 1400.0,
                protein_binding: 0.98,
                renal_elimination_fraction: 0.05,
                hepatic_extraction_ratio: 0.50,
            },
            PkProfile {
                drug_id: DrugId::new("ibuprofen"),
                half_life_hours: 2.0,
                tmax_hours: 1.0,
                bioavailability: 0.95,
                clearance_l_per_h: 3.5,
                volume_distribution_l: 10.0,
                protein_binding: 0.99,
                renal_elimination_fraction: 0.10,
                hepatic_extraction_ratio: 0.40,
            },
        ];
        for e in entries {
            db.add(e);
        }
        db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drug_id_normalisation() {
        let id = DrugId::new("Atorva-Statin");
        assert_eq!(id.as_str(), "atorva_statin");
    }

    #[test]
    fn test_drug_id_equality() {
        assert_eq!(DrugId::new("warfarin"), DrugId::new("Warfarin"));
    }

    #[test]
    fn test_conflict_severity_ordering() {
        assert!(ConflictSeverity::Critical > ConflictSeverity::Major);
        assert!(ConflictSeverity::Major > ConflictSeverity::Moderate);
        assert!(ConflictSeverity::Moderate > ConflictSeverity::Minor);
    }

    #[test]
    fn test_conflict_severity_score() {
        assert_eq!(ConflictSeverity::Critical.numeric_score(), 4);
        assert_eq!(ConflictSeverity::Minor.numeric_score(), 1);
    }

    #[test]
    fn test_confirmed_conflict_involves() {
        let c = ConfirmedConflict::new(
            DrugId::new("a"),
            DrugId::new("b"),
            InteractionType::QtProlongation,
            ConflictSeverity::Major,
        );
        assert!(c.involves(&DrugId::new("a")));
        assert!(c.involves(&DrugId::new("b")));
        assert!(!c.involves(&DrugId::new("c")));
    }

    #[test]
    fn test_patient_allergy_check() {
        let mut p = PatientProfile::new("p1", 65, 80.0);
        p.add_allergy("Penicillin");
        assert!(p.has_allergy("penicillin"));
        assert!(!p.has_allergy("sulfa"));
    }

    #[test]
    fn test_dosing_schedule_daily_dose() {
        let ds = DosingSchedule::oral(250.0, 3);
        assert!((ds.daily_dose() - 750.0).abs() < 0.01);
    }

    #[test]
    fn test_dosing_schedule_exceeds_max() {
        let ds = DosingSchedule::oral(500.0, 4).with_max_daily(1500.0);
        assert!(ds.exceeds_max_daily());
    }

    #[test]
    fn test_pk_profile_elimination_rate() {
        let pk = PkProfile::new(DrugId::new("test"), 10.0, 2.0);
        let ke = pk.elimination_rate();
        assert!((ke - 0.0693).abs() < 0.001);
    }

    #[test]
    fn test_pk_database_demo() {
        let db = PkDatabase::demo();
        assert!(db.get(&DrugId::new("warfarin")).is_some());
        assert!(db.get(&DrugId::new("nonexistent")).is_none());
    }

    #[test]
    fn test_renal_function_dose_factor() {
        assert!((RenalFunction::Normal.dose_factor() - 1.0).abs() < 0.01);
        assert!(RenalFunction::SevereImpairment.dose_factor() < 0.5);
    }

    #[test]
    fn test_hepatic_function_dose_factor() {
        assert!((HepaticFunction::Normal.dose_factor() - 1.0).abs() < 0.01);
        assert!(HepaticFunction::SevereImpairment.dose_factor() < 0.5);
    }

    #[test]
    fn test_drug_class_therapeutic_area() {
        assert_eq!(DrugClass::Statin.therapeutic_area(), "Lipid-lowering");
        assert_eq!(DrugClass::ACEInhibitor.therapeutic_area(), "Cardiovascular");
    }

    #[test]
    fn test_safety_verdict_acceptable() {
        assert!(SafetyVerdict::Safe.is_acceptable());
        assert!(SafetyVerdict::PossiblySafe.is_acceptable());
        assert!(!SafetyVerdict::Unsafe.is_acceptable());
    }

    #[test]
    fn test_evidence_level_confidence() {
        assert!(EvidenceLevel::High.confidence_factor() > EvidenceLevel::Low.confidence_factor());
    }

    #[test]
    fn test_patient_estimated_crcl() {
        let p = PatientProfile::new("p1", 60, 80.0).with_sex(Sex::Male);
        let crcl = p.estimated_crcl();
        assert!(crcl > 50.0);
    }
}
