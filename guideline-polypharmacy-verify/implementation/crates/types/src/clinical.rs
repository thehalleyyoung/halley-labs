//! # Clinical Types
//!
//! Domain types for representing patient clinical data, conditions, lab values,
//! medication status, and clinical predicates used by the polypharmacy
//! verification engine.
//!
//! These types form the clinical context layer that bridges raw patient data
//! with the formal verification tiers. [`ClinicalPredicate`] enables
//! declarative, composable Boolean queries over a [`PatientProfile`] and
//! [`ClinicalState`], driving guideline guard evaluation.

use std::fmt;
use std::str::FromStr;

use chrono::{NaiveDate, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::drug::{DosingSchedule, DrugId, DrugRoute};

// ---------------------------------------------------------------------------
// Sex
// ---------------------------------------------------------------------------

/// Biological sex of a patient, relevant for pharmacokinetic calculations
/// such as Cockcroft-Gault creatinine clearance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
    Other,
    Unknown,
}

impl fmt::Display for Sex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sex::Male => write!(f, "Male"),
            Sex::Female => write!(f, "Female"),
            Sex::Other => write!(f, "Other"),
            Sex::Unknown => write!(f, "Unknown"),
        }
    }
}

impl FromStr for Sex {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "male" | "m" => Ok(Sex::Male),
            "female" | "f" => Ok(Sex::Female),
            "other" => Ok(Sex::Other),
            "unknown" | "u" => Ok(Sex::Unknown),
            _ => Err(format!("unknown sex variant: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// HepaticFunction
// ---------------------------------------------------------------------------

/// Classification of hepatic (liver) function impairment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HepaticFunction {
    Normal,
    MildImpairment,
    ModerateImpairment,
    SevereImpairment,
}

impl HepaticFunction {
    /// Returns the corresponding Child-Pugh class label.
    ///
    /// * Normal → "None"
    /// * MildImpairment → "A"
    /// * ModerateImpairment → "B"
    /// * SevereImpairment → "C"
    pub fn child_pugh_class(&self) -> &str {
        match self {
            HepaticFunction::Normal => "None",
            HepaticFunction::MildImpairment => "A",
            HepaticFunction::ModerateImpairment => "B",
            HepaticFunction::SevereImpairment => "C",
        }
    }
}

impl fmt::Display for HepaticFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HepaticFunction::Normal => write!(f, "Normal"),
            HepaticFunction::MildImpairment => write!(f, "Mild Impairment"),
            HepaticFunction::ModerateImpairment => write!(f, "Moderate Impairment"),
            HepaticFunction::SevereImpairment => write!(f, "Severe Impairment"),
        }
    }
}

impl FromStr for HepaticFunction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "normal" => Ok(HepaticFunction::Normal),
            "mild" | "mild impairment" | "a" => Ok(HepaticFunction::MildImpairment),
            "moderate" | "moderate impairment" | "b" => {
                Ok(HepaticFunction::ModerateImpairment)
            }
            "severe" | "severe impairment" | "c" => Ok(HepaticFunction::SevereImpairment),
            _ => Err(format!("unknown hepatic function: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// RenalFunction
// ---------------------------------------------------------------------------

/// Classification of renal (kidney) function based on eGFR staging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenalFunction {
    /// eGFR ≥ 90 mL/min/1.73 m²
    Normal,
    /// eGFR 60–89
    MildImpairment,
    /// eGFR 30–59
    ModerateImpairment,
    /// eGFR 15–29
    SevereImpairment,
    /// eGFR < 15 or on dialysis
    EndStage,
}

impl RenalFunction {
    /// Returns the inclusive (low, high) eGFR range for this classification.
    pub fn egfr_range(&self) -> (f64, f64) {
        match self {
            RenalFunction::Normal => (90.0, f64::INFINITY),
            RenalFunction::MildImpairment => (60.0, 89.0),
            RenalFunction::ModerateImpairment => (30.0, 59.0),
            RenalFunction::SevereImpairment => (15.0, 29.0),
            RenalFunction::EndStage => (0.0, 14.0),
        }
    }

    /// Whether this level of renal impairment typically requires dose
    /// adjustment of renally cleared drugs.
    pub fn requires_dose_adjustment(&self) -> bool {
        matches!(
            self,
            RenalFunction::ModerateImpairment
                | RenalFunction::SevereImpairment
                | RenalFunction::EndStage
        )
    }
}

impl fmt::Display for RenalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenalFunction::Normal => write!(f, "Normal"),
            RenalFunction::MildImpairment => write!(f, "Mild Impairment"),
            RenalFunction::ModerateImpairment => write!(f, "Moderate Impairment"),
            RenalFunction::SevereImpairment => write!(f, "Severe Impairment"),
            RenalFunction::EndStage => write!(f, "End Stage"),
        }
    }
}

impl FromStr for RenalFunction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "normal" => Ok(RenalFunction::Normal),
            "mild" | "mild impairment" => Ok(RenalFunction::MildImpairment),
            "moderate" | "moderate impairment" => Ok(RenalFunction::ModerateImpairment),
            "severe" | "severe impairment" => Ok(RenalFunction::SevereImpairment),
            "end stage" | "endstage" | "end-stage" | "esrd" => Ok(RenalFunction::EndStage),
            _ => Err(format!("unknown renal function: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// ConditionSeverity
// ---------------------------------------------------------------------------

/// Severity grading for a clinical condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConditionSeverity {
    Mild,
    Moderate,
    Severe,
    Critical,
    Unspecified,
}

impl ConditionSeverity {
    /// Numeric severity score (0–4) useful for threshold comparisons.
    pub fn numeric_score(&self) -> u32 {
        match self {
            ConditionSeverity::Unspecified => 0,
            ConditionSeverity::Mild => 1,
            ConditionSeverity::Moderate => 2,
            ConditionSeverity::Severe => 3,
            ConditionSeverity::Critical => 4,
        }
    }
}

impl fmt::Display for ConditionSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConditionSeverity::Mild => write!(f, "Mild"),
            ConditionSeverity::Moderate => write!(f, "Moderate"),
            ConditionSeverity::Severe => write!(f, "Severe"),
            ConditionSeverity::Critical => write!(f, "Critical"),
            ConditionSeverity::Unspecified => write!(f, "Unspecified"),
        }
    }
}

impl FromStr for ConditionSeverity {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "mild" => Ok(ConditionSeverity::Mild),
            "moderate" => Ok(ConditionSeverity::Moderate),
            "severe" => Ok(ConditionSeverity::Severe),
            "critical" => Ok(ConditionSeverity::Critical),
            "unspecified" | "" => Ok(ConditionSeverity::Unspecified),
            _ => Err(format!("unknown condition severity: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// ClinicalCondition
// ---------------------------------------------------------------------------

/// A diagnosed clinical condition (disease / syndrome) on a patient record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalCondition {
    /// Internal identifier (may be a GUID or human-readable slug).
    pub id: String,
    /// Human-readable condition name (e.g. "Type 2 Diabetes").
    pub name: String,
    /// Optional ICD-10-CM code (e.g. "E11.9").
    pub icd10_code: Option<String>,
    /// Date the condition was first recorded / onset.
    pub onset_date: Option<NaiveDate>,
    /// Severity classification.
    pub severity: ConditionSeverity,
    /// Whether the condition is currently active.
    pub is_active: bool,
    /// Free-text clinician notes.
    pub notes: Option<String>,
}

impl ClinicalCondition {
    /// Create a new active condition with the given id and name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            icd10_code: None,
            onset_date: None,
            severity: ConditionSeverity::Unspecified,
            is_active: true,
            notes: None,
        }
    }

    /// Heuristic: a condition is considered chronic if its onset was more than
    /// 90 days ago.  Returns `None` when the onset date is unknown.
    pub fn is_chronic(&self) -> Option<bool> {
        self.onset_date.map(|onset| {
            let today = Utc::now().date_naive();
            let duration = today.signed_duration_since(onset);
            duration.num_days() > 90
        })
    }

    /// Check whether the condition's ICD-10 code starts with the given prefix
    /// (case-insensitive).
    pub fn matches_icd10_prefix(&self, prefix: &str) -> bool {
        self.icd10_code
            .as_deref()
            .map(|c| c.to_uppercase().starts_with(&prefix.to_uppercase()))
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// LabValueRange
// ---------------------------------------------------------------------------

/// Reference range for a laboratory value, with optional abnormal/critical
/// thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabValueRange {
    /// Lower bound of the normal range (inclusive).
    pub normal_low: f64,
    /// Upper bound of the normal range (inclusive).
    pub normal_high: f64,
    /// Abnormal low threshold (below normal_low but not yet critical).
    pub abnormal_low: Option<f64>,
    /// Abnormal high threshold (above normal_high but not yet critical).
    pub abnormal_high: Option<f64>,
    /// Critical low threshold — immediate clinical action required.
    pub critical_low: Option<f64>,
    /// Critical high threshold — immediate clinical action required.
    pub critical_high: Option<f64>,
}

impl LabValueRange {
    /// Create a new range with just normal bounds; abnormal/critical
    /// thresholds default to `None`.
    pub fn new(normal_low: f64, normal_high: f64) -> Self {
        Self {
            normal_low,
            normal_high,
            abnormal_low: None,
            abnormal_high: None,
            critical_low: None,
            critical_high: None,
        }
    }

    /// Is the value within the normal range?
    pub fn contains(&self, value: f64) -> bool {
        value >= self.normal_low && value <= self.normal_high
    }

    /// Is the value outside the normal range but within abnormal thresholds?
    /// Returns `false` when abnormal thresholds are unset and the value is
    /// outside the normal range.
    pub fn is_abnormal(&self, value: f64) -> bool {
        if self.contains(value) {
            return false;
        }
        // Value is outside normal. Check if it breaches abnormal thresholds.
        if let Some(lo) = self.abnormal_low {
            if value < lo {
                return true;
            }
        }
        if let Some(hi) = self.abnormal_high {
            if value > hi {
                return true;
            }
        }
        // Even without abnormal thresholds, anything outside normal is
        // abnormal by definition.
        true
    }

    /// Is the value in the critical range?
    pub fn is_critical(&self, value: f64) -> bool {
        if let Some(lo) = self.critical_low {
            if value <= lo {
                return true;
            }
        }
        if let Some(hi) = self.critical_high {
            if value >= hi {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// LabValue
// ---------------------------------------------------------------------------

/// A single laboratory measurement for a patient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabValue {
    /// Lab test name (e.g. "Serum Creatinine", "INR").
    pub name: String,
    /// Measured numeric value.
    pub value: f64,
    /// Unit of measure (e.g. "mg/dL", "mmol/L").
    pub unit: String,
    /// Optional reference range for this lab.
    pub reference_range: Option<LabValueRange>,
    /// Timestamp of the measurement.
    pub timestamp: chrono::DateTime<Utc>,
}

impl LabValue {
    /// Is the value within the normal reference range?
    /// Returns `true` when no reference range is available (assumed normal).
    pub fn is_normal(&self) -> bool {
        self.reference_range
            .as_ref()
            .map(|r| r.contains(self.value))
            .unwrap_or(true)
    }

    /// Is the value in the critical range?
    /// Returns `false` when no reference range or critical thresholds exist.
    pub fn is_critical(&self) -> bool {
        self.reference_range
            .as_ref()
            .map(|r| r.is_critical(self.value))
            .unwrap_or(false)
    }

    /// Fractional deviation from the nearest normal bound.
    ///
    /// * Returns `0.0` if within range.
    /// * Returns a positive fraction (e.g. 0.25 = 25 % above upper bound)
    ///   when above the upper normal limit.
    /// * Returns a negative fraction when below the lower normal limit.
    /// * Returns `None` when no reference range is set.
    pub fn deviation_from_normal(&self) -> Option<f64> {
        self.reference_range.as_ref().map(|r| {
            if self.value < r.normal_low {
                if r.normal_low == 0.0 {
                    return self.value - r.normal_low;
                }
                (self.value - r.normal_low) / r.normal_low.abs()
            } else if self.value > r.normal_high {
                if r.normal_high == 0.0 {
                    return self.value - r.normal_high;
                }
                (self.value - r.normal_high) / r.normal_high.abs()
            } else {
                0.0
            }
        })
    }
}

// ---------------------------------------------------------------------------
// MedicationStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a medication order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MedicationStatus {
    Active,
    Discontinued,
    OnHold,
    Planned,
    Completed,
}

impl fmt::Display for MedicationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MedicationStatus::Active => write!(f, "Active"),
            MedicationStatus::Discontinued => write!(f, "Discontinued"),
            MedicationStatus::OnHold => write!(f, "On Hold"),
            MedicationStatus::Planned => write!(f, "Planned"),
            MedicationStatus::Completed => write!(f, "Completed"),
        }
    }
}

impl FromStr for MedicationStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "active" => Ok(MedicationStatus::Active),
            "discontinued" => Ok(MedicationStatus::Discontinued),
            "on hold" | "onhold" | "on_hold" => Ok(MedicationStatus::OnHold),
            "planned" => Ok(MedicationStatus::Planned),
            "completed" => Ok(MedicationStatus::Completed),
            _ => Err(format!("unknown medication status: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// ActiveMedication
// ---------------------------------------------------------------------------

/// A medication entry on a patient's medication list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveMedication {
    /// Drug identifier, linking to the drug knowledge base.
    pub drug_id: DrugId,
    /// Dosing schedule (dose, interval, route).
    pub dose_schedule: DosingSchedule,
    /// Date the medication was started.
    pub start_date: NaiveDate,
    /// Optional planned or actual end date.
    pub end_date: Option<NaiveDate>,
    /// Current lifecycle status.
    pub status: MedicationStatus,
    /// Name of the prescribing clinician.
    pub prescriber: Option<String>,
    /// Clinical indication / reason for prescribing.
    pub indication: Option<String>,
}

impl ActiveMedication {
    /// Create a new active medication with the given drug and schedule.
    pub fn new(drug_id: DrugId, dose_schedule: DosingSchedule, start_date: NaiveDate) -> Self {
        Self {
            drug_id,
            dose_schedule,
            start_date,
            end_date: None,
            status: MedicationStatus::Active,
            prescriber: None,
            indication: None,
        }
    }

    /// Is the medication currently active (status == Active)?
    pub fn is_active(&self) -> bool {
        self.status == MedicationStatus::Active
    }

    /// Number of days from start to end (or today if still active).
    pub fn duration_days(&self) -> i64 {
        let end = self
            .end_date
            .unwrap_or_else(|| Utc::now().date_naive());
        end.signed_duration_since(self.start_date).num_days()
    }

    /// Is the medication current, i.e. active and today falls within the
    /// start..=end window (or open-ended)?
    pub fn is_current(&self) -> bool {
        if self.status != MedicationStatus::Active {
            return false;
        }
        let today = Utc::now().date_naive();
        if today < self.start_date {
            return false;
        }
        match self.end_date {
            Some(end) => today <= end,
            None => true,
        }
    }
}

// ---------------------------------------------------------------------------
// PatientProfile
// ---------------------------------------------------------------------------

/// Comprehensive patient profile aggregating demographics, organ function,
/// conditions, medications, lab values, and allergies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientProfile {
    /// Patient identifier.
    pub id: String,
    /// Age in whole years.
    pub age_years: u32,
    /// Body weight in kilograms.
    pub weight_kg: f64,
    /// Height in centimetres.
    pub height_cm: f64,
    /// Biological sex.
    pub sex: Sex,
    /// Most recent eGFR value (mL/min/1.73 m²), if available.
    pub egfr: Option<f64>,
    /// Hepatic function classification.
    pub hepatic_function: HepaticFunction,
    /// Renal function classification.
    pub renal_function: RenalFunction,
    /// Active and historical clinical conditions.
    pub conditions: Vec<ClinicalCondition>,
    /// Current and past medication entries.
    pub medications: Vec<ActiveMedication>,
    /// Laboratory values (most recent per analyte expected).
    pub lab_values: Vec<LabValue>,
    /// Known drug / substance allergies (free-text list).
    pub allergies: Vec<String>,
}

impl PatientProfile {
    /// Create a minimal patient profile with the required demographics.
    pub fn new(
        id: impl Into<String>,
        age_years: u32,
        weight_kg: f64,
        height_cm: f64,
        sex: Sex,
    ) -> Self {
        Self {
            id: id.into(),
            age_years,
            weight_kg,
            height_cm,
            sex,
            egfr: None,
            hepatic_function: HepaticFunction::Normal,
            renal_function: RenalFunction::Normal,
            conditions: Vec::new(),
            medications: Vec::new(),
            lab_values: Vec::new(),
            allergies: Vec::new(),
        }
    }

    /// Body-mass index (kg / m²).
    pub fn bmi(&self) -> f64 {
        let height_m = self.height_cm / 100.0;
        if height_m <= 0.0 {
            return 0.0;
        }
        self.weight_kg / (height_m * height_m)
    }

    /// Body surface area in m² using the Mosteller formula:
    ///
    /// BSA = sqrt( (height_cm × weight_kg) / 3600 )
    pub fn bsa(&self) -> f64 {
        let product = self.height_cm * self.weight_kg;
        if product <= 0.0 {
            return 0.0;
        }
        (product / 3600.0).sqrt()
    }

    /// Does the patient have an active condition whose name matches (case-
    /// insensitive substring)?
    pub fn has_condition(&self, name: &str) -> bool {
        let needle = name.to_lowercase();
        self.conditions
            .iter()
            .any(|c| c.is_active && c.name.to_lowercase().contains(&needle))
    }

    /// Is there an active medication whose drug id matches?
    pub fn has_active_medication(&self, drug_name: &str) -> bool {
        let needle = drug_name.to_lowercase();
        self.medications
            .iter()
            .any(|m| m.is_active() && m.drug_id.as_str().contains(&needle))
    }

    /// Collect the [`DrugId`]s of all currently active medications.
    pub fn active_drug_ids(&self) -> Vec<DrugId> {
        self.medications
            .iter()
            .filter(|m| m.is_active())
            .map(|m| m.drug_id.clone())
            .collect()
    }

    /// Is the patient considered elderly (age ≥ 65)?
    pub fn is_elderly(&self) -> bool {
        self.age_years >= 65
    }

    /// Estimate creatinine clearance (mL/min) via the Cockcroft-Gault
    /// equation.
    ///
    /// Requires a "Serum Creatinine" lab value in mg/dL.  Returns `None`
    /// when the value is unavailable or zero.
    ///
    /// CrCl = ((140 - age) × weight_kg) / (72 × SCr)  × 0.85 if female
    pub fn creatinine_clearance(&self) -> Option<f64> {
        let scr = self
            .lab_values
            .iter()
            .find(|l| l.name.to_lowercase().contains("creatinine"))?;
        if scr.value <= 0.0 {
            return None;
        }
        let mut crcl =
            ((140.0 - self.age_years as f64) * self.weight_kg) / (72.0 * scr.value);
        if self.sex == Sex::Female {
            crcl *= 0.85;
        }
        Some(crcl)
    }

    /// Derive a [`RenalFunction`] classification from the stored eGFR value.
    /// Falls back to the existing `renal_function` field when eGFR is
    /// unavailable.
    pub fn renal_function_from_egfr(&self) -> RenalFunction {
        match self.egfr {
            Some(val) if val >= 90.0 => RenalFunction::Normal,
            Some(val) if val >= 60.0 => RenalFunction::MildImpairment,
            Some(val) if val >= 30.0 => RenalFunction::ModerateImpairment,
            Some(val) if val >= 15.0 => RenalFunction::SevereImpairment,
            Some(_) => RenalFunction::EndStage,
            None => self.renal_function,
        }
    }
}

// ---------------------------------------------------------------------------
// ClinicalState
// ---------------------------------------------------------------------------

/// A lightweight Boolean-flag map summarising the clinical state for predicate
/// evaluation.  Condition names, medication names, and lab flag names are
/// normalised to lowercase keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalState {
    /// Active conditions (key → present flag).
    pub conditions: IndexMap<String, bool>,
    /// Active medications (key → present flag).
    pub medications: IndexMap<String, bool>,
    /// Lab flags (key → flagged, e.g. "elevated_inr" → true).
    pub lab_flags: IndexMap<String, bool>,
}

impl ClinicalState {
    /// New empty state.
    pub fn new() -> Self {
        Self {
            conditions: IndexMap::new(),
            medications: IndexMap::new(),
            lab_flags: IndexMap::new(),
        }
    }

    /// Alias for [`ClinicalState::new`].
    pub fn empty() -> Self {
        Self::new()
    }

    /// Set a condition flag.
    pub fn set_condition(&mut self, name: impl Into<String>, present: bool) {
        self.conditions.insert(name.into().to_lowercase(), present);
    }

    /// Set a medication flag.
    pub fn set_medication(&mut self, name: impl Into<String>, present: bool) {
        self.medications.insert(name.into().to_lowercase(), present);
    }

    /// Set a lab flag.
    pub fn set_lab_flag(&mut self, name: impl Into<String>, flagged: bool) {
        self.lab_flags.insert(name.into().to_lowercase(), flagged);
    }

    /// Query whether a condition is present.
    pub fn has_condition(&self, name: &str) -> bool {
        self.conditions
            .get(&name.to_lowercase())
            .copied()
            .unwrap_or(false)
    }

    /// Query whether a medication is present.
    pub fn has_medication(&self, name: &str) -> bool {
        self.medications
            .get(&name.to_lowercase())
            .copied()
            .unwrap_or(false)
    }

    /// Query whether a lab flag is set.
    pub fn has_lab_flag(&self, name: &str) -> bool {
        self.lab_flags
            .get(&name.to_lowercase())
            .copied()
            .unwrap_or(false)
    }

    /// Build a [`ClinicalState`] from a [`PatientProfile`].
    ///
    /// * Active conditions → condition flags keyed by lowercased name.
    /// * Active medications → medication flags keyed by drug id string.
    /// * Abnormal lab values → lab flags keyed by `"abnormal_{name}"`.
    pub fn from_patient_profile(profile: &PatientProfile) -> Self {
        let mut state = Self::new();
        for cond in &profile.conditions {
            state.set_condition(&cond.name, cond.is_active);
        }
        for med in &profile.medications {
            state.set_medication(med.drug_id.as_str(), med.is_active());
        }
        for lab in &profile.lab_values {
            if !lab.is_normal() {
                state.set_lab_flag(format!("abnormal_{}", lab.name), true);
            }
        }
        state
    }

    /// Merge another state into this one.  `true` values from `other`
    /// override `false` values in `self`; existing `true` values are
    /// preserved.
    pub fn merge(&mut self, other: &ClinicalState) {
        for (k, &v) in &other.conditions {
            let entry = self.conditions.entry(k.clone()).or_insert(false);
            *entry = *entry || v;
        }
        for (k, &v) in &other.medications {
            let entry = self.medications.entry(k.clone()).or_insert(false);
            *entry = *entry || v;
        }
        for (k, &v) in &other.lab_flags {
            let entry = self.lab_flags.entry(k.clone()).or_insert(false);
            *entry = *entry || v;
        }
    }

    /// All keys across every map, deduplicated and sorted.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .conditions
            .keys()
            .chain(self.medications.keys())
            .chain(self.lab_flags.keys())
            .cloned()
            .collect();
        keys.sort();
        keys.dedup();
        keys
    }
}

impl Default for ClinicalState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ClinicalPredicate
// ---------------------------------------------------------------------------

/// Composable Boolean predicate over patient clinical data.
///
/// Predicates are evaluated against a [`PatientProfile`] and an optional
/// [`ClinicalState`] flag map, enabling declarative guideline guard
/// expressions such as:
///
/// ```text
/// And(
///   HasCondition("atrial_fibrillation"),
///   Not(OnMedication("warfarin"))
/// )
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClinicalPredicate {
    /// Patient has an active condition whose name contains the given string.
    HasCondition(String),
    /// A named lab value falls within [low, high].
    LabInRange {
        name: String,
        low: f64,
        high: f64,
    },
    /// Patient is on a medication whose drug id contains the given string.
    OnMedication(String),
    /// Patient age falls within [min, max].
    AgeRange {
        min: u32,
        max: u32,
    },
    /// Renal function matches the given classification.
    RenalFunction(RenalFunction),
    /// Hepatic function matches the given classification.
    HepaticFunction(HepaticFunction),
    /// Logical conjunction.
    And(Box<ClinicalPredicate>, Box<ClinicalPredicate>),
    /// Logical disjunction.
    Or(Box<ClinicalPredicate>, Box<ClinicalPredicate>),
    /// Logical negation.
    Not(Box<ClinicalPredicate>),
    /// Always true.
    True,
    /// Always false.
    False,
}

impl ClinicalPredicate {
    /// Recursively evaluate this predicate against a patient and clinical
    /// state.
    pub fn evaluate(&self, patient: &PatientProfile, state: &ClinicalState) -> bool {
        match self {
            ClinicalPredicate::HasCondition(name) => {
                // Check both the profile and the state flag map.
                patient.has_condition(name) || state.has_condition(name)
            }
            ClinicalPredicate::LabInRange { name, low, high } => {
                let needle = name.to_lowercase();
                patient
                    .lab_values
                    .iter()
                    .find(|l| l.name.to_lowercase().contains(&needle))
                    .map(|l| l.value >= *low && l.value <= *high)
                    .unwrap_or(false)
            }
            ClinicalPredicate::OnMedication(name) => {
                patient.has_active_medication(name) || state.has_medication(name)
            }
            ClinicalPredicate::AgeRange { min, max } => {
                patient.age_years >= *min && patient.age_years <= *max
            }
            ClinicalPredicate::RenalFunction(rf) => patient.renal_function == *rf,
            ClinicalPredicate::HepaticFunction(hf) => patient.hepatic_function == *hf,
            ClinicalPredicate::And(a, b) => {
                a.evaluate(patient, state) && b.evaluate(patient, state)
            }
            ClinicalPredicate::Or(a, b) => {
                a.evaluate(patient, state) || b.evaluate(patient, state)
            }
            ClinicalPredicate::Not(inner) => !inner.evaluate(patient, state),
            ClinicalPredicate::True => true,
            ClinicalPredicate::False => false,
        }
    }

    /// Combine with another predicate via conjunction.
    pub fn and(self, other: ClinicalPredicate) -> ClinicalPredicate {
        ClinicalPredicate::And(Box::new(self), Box::new(other))
    }

    /// Combine with another predicate via disjunction.
    pub fn or(self, other: ClinicalPredicate) -> ClinicalPredicate {
        ClinicalPredicate::Or(Box::new(self), Box::new(other))
    }

    /// Negate this predicate.
    pub fn negate(self) -> ClinicalPredicate {
        ClinicalPredicate::Not(Box::new(self))
    }

    /// Returns `true` when the predicate is the trivial `True` constant.
    pub fn is_trivially_true(&self) -> bool {
        matches!(self, ClinicalPredicate::True)
    }

    /// Human-readable description of the predicate tree.
    pub fn description(&self) -> String {
        match self {
            ClinicalPredicate::HasCondition(c) => {
                format!("patient has condition '{c}'")
            }
            ClinicalPredicate::LabInRange { name, low, high } => {
                format!("lab '{name}' in [{low}, {high}]")
            }
            ClinicalPredicate::OnMedication(m) => {
                format!("patient is on medication '{m}'")
            }
            ClinicalPredicate::AgeRange { min, max } => {
                format!("patient age in [{min}, {max}]")
            }
            ClinicalPredicate::RenalFunction(rf) => {
                format!("renal function is {rf}")
            }
            ClinicalPredicate::HepaticFunction(hf) => {
                format!("hepatic function is {hf}")
            }
            ClinicalPredicate::And(a, b) => {
                format!("({} AND {})", a.description(), b.description())
            }
            ClinicalPredicate::Or(a, b) => {
                format!("({} OR {})", a.description(), b.description())
            }
            ClinicalPredicate::Not(inner) => {
                format!("NOT ({})", inner.description())
            }
            ClinicalPredicate::True => "TRUE".to_string(),
            ClinicalPredicate::False => "FALSE".to_string(),
        }
    }
}

impl fmt::Display for ClinicalPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drug::{DosingSchedule, DrugId, DrugRoute};

    // -- helpers --

    fn sample_patient() -> PatientProfile {
        let mut p = PatientProfile::new("P-001", 72, 80.0, 175.0, Sex::Male);
        p.egfr = Some(45.0);
        p.renal_function = RenalFunction::ModerateImpairment;
        p.hepatic_function = HepaticFunction::MildImpairment;

        p.conditions.push({
            let mut c = ClinicalCondition::new("c1", "Atrial Fibrillation");
            c.icd10_code = Some("I48.91".to_string());
            c.severity = ConditionSeverity::Moderate;
            c.onset_date = Some(NaiveDate::from_ymd_opt(2020, 1, 15).unwrap());
            c
        });
        p.conditions.push({
            let mut c = ClinicalCondition::new("c2", "Type 2 Diabetes");
            c.icd10_code = Some("E11.9".to_string());
            c.severity = ConditionSeverity::Mild;
            c
        });

        let sched = DosingSchedule::new(24.0, 5.0, DrugRoute::Oral, 1);
        p.medications.push(ActiveMedication::new(
            DrugId::new("warfarin"),
            sched,
            NaiveDate::from_ymd_opt(2023, 6, 1).unwrap(),
        ));

        let met_sched = DosingSchedule::new(12.0, 500.0, DrugRoute::Oral, 2);
        p.medications.push(ActiveMedication::new(
            DrugId::new("metformin"),
            met_sched,
            NaiveDate::from_ymd_opt(2023, 3, 1).unwrap(),
        ));

        p.lab_values.push(LabValue {
            name: "Serum Creatinine".to_string(),
            value: 1.8,
            unit: "mg/dL".to_string(),
            reference_range: Some(LabValueRange {
                normal_low: 0.7,
                normal_high: 1.3,
                abnormal_low: Some(0.4),
                abnormal_high: Some(2.0),
                critical_low: Some(0.2),
                critical_high: Some(4.0),
            }),
            timestamp: Utc::now(),
        });

        p.lab_values.push(LabValue {
            name: "INR".to_string(),
            value: 2.5,
            unit: "ratio".to_string(),
            reference_range: Some(LabValueRange::new(0.8, 1.2)),
            timestamp: Utc::now(),
        });

        p.allergies.push("Penicillin".to_string());
        p
    }

    // -- 1. Patient profile construction --

    #[test]
    fn test_patient_profile_construction() {
        let p = sample_patient();
        assert_eq!(p.id, "P-001");
        assert_eq!(p.age_years, 72);
        assert_eq!(p.conditions.len(), 2);
        assert_eq!(p.medications.len(), 2);
        assert_eq!(p.lab_values.len(), 2);
        assert!(p.is_elderly());
    }

    // -- 2. BMI calculation --

    #[test]
    fn test_bmi_calculation() {
        let p = sample_patient();
        let bmi = p.bmi();
        // 80 / (1.75^2) ≈ 26.12
        assert!((bmi - 26.122).abs() < 0.1, "BMI was {bmi}");
    }

    // -- 3. BSA Mosteller --

    #[test]
    fn test_bsa_mosteller() {
        let p = sample_patient();
        let bsa = p.bsa();
        // sqrt((175 * 80) / 3600) = sqrt(3.8889) ≈ 1.972
        assert!((bsa - 1.972).abs() < 0.01, "BSA was {bsa}");
    }

    // -- 4. ClinicalPredicate simple evaluation --

    #[test]
    fn test_predicate_has_condition() {
        let p = sample_patient();
        let state = ClinicalState::from_patient_profile(&p);

        let pred = ClinicalPredicate::HasCondition("atrial fibrillation".to_string());
        assert!(pred.evaluate(&p, &state));

        let pred2 = ClinicalPredicate::HasCondition("heart failure".to_string());
        assert!(!pred2.evaluate(&p, &state));
    }

    // -- 5. Nested Boolean predicate --

    #[test]
    fn test_predicate_nested_boolean() {
        let p = sample_patient();
        let state = ClinicalState::from_patient_profile(&p);

        // (HasCondition("diabetes") AND OnMedication("warfarin")) => true
        let pred = ClinicalPredicate::HasCondition("diabetes".to_string())
            .and(ClinicalPredicate::OnMedication("warfarin".to_string()));
        assert!(pred.evaluate(&p, &state));

        // NOT (HasCondition("diabetes") AND OnMedication("aspirin")) => true
        let pred2 = ClinicalPredicate::HasCondition("diabetes".to_string())
            .and(ClinicalPredicate::OnMedication("aspirin".to_string()))
            .negate();
        assert!(pred2.evaluate(&p, &state));

        // (AgeRange 60..80 OR False) => true
        let pred3 = ClinicalPredicate::AgeRange { min: 60, max: 80 }
            .or(ClinicalPredicate::False);
        assert!(pred3.evaluate(&p, &state));
    }

    // -- 6. Lab value checking --

    #[test]
    fn test_lab_value_normal_and_critical() {
        let range = LabValueRange {
            normal_low: 0.7,
            normal_high: 1.3,
            abnormal_low: Some(0.4),
            abnormal_high: Some(2.0),
            critical_low: Some(0.2),
            critical_high: Some(4.0),
        };

        assert!(range.contains(1.0));
        assert!(!range.contains(1.8));
        assert!(range.is_abnormal(1.8));
        assert!(!range.is_critical(1.8));
        assert!(range.is_critical(0.1));
        assert!(range.is_critical(5.0));
    }

    // -- 7. Lab deviation --

    #[test]
    fn test_lab_value_deviation() {
        let lab = LabValue {
            name: "Glucose".to_string(),
            value: 130.0,
            unit: "mg/dL".to_string(),
            reference_range: Some(LabValueRange::new(70.0, 100.0)),
            timestamp: Utc::now(),
        };
        assert!(!lab.is_normal());
        let dev = lab.deviation_from_normal().unwrap();
        // (130 - 100) / 100 = 0.3
        assert!((dev - 0.3).abs() < 1e-9, "deviation was {dev}");
    }

    // -- 8. Medication status / duration --

    #[test]
    fn test_medication_status_and_duration() {
        let sched = DosingSchedule::new(24.0, 10.0, DrugRoute::Oral, 1);
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let mut med = ActiveMedication::new(DrugId::new("lisinopril"), sched, start);
        med.end_date = Some(end);

        assert!(med.is_active());
        assert_eq!(med.duration_days(), 30);

        med.status = MedicationStatus::Discontinued;
        assert!(!med.is_active());
        assert!(!med.is_current());
    }

    // -- 9. ClinicalState from_patient_profile --

    #[test]
    fn test_clinical_state_from_profile() {
        let p = sample_patient();
        let state = ClinicalState::from_patient_profile(&p);

        assert!(state.has_condition("atrial fibrillation"));
        assert!(state.has_condition("type 2 diabetes"));
        assert!(state.has_medication("warfarin"));
        assert!(state.has_medication("metformin"));
        // Creatinine 1.8 is above normal range (0.7–1.3) → flagged
        assert!(state.has_lab_flag("abnormal_Serum Creatinine"));
    }

    // -- 10. Serde roundtrip for PatientProfile --

    #[test]
    fn test_serde_roundtrip_patient_profile() {
        let p = sample_patient();
        let json = serde_json::to_string(&p).expect("serialize");
        let p2: PatientProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p2.id, p.id);
        assert_eq!(p2.age_years, p.age_years);
        assert_eq!(p2.conditions.len(), p.conditions.len());
        assert_eq!(p2.medications.len(), p.medications.len());
    }

    // -- 11. Serde roundtrip for ClinicalPredicate --

    #[test]
    fn test_serde_roundtrip_predicate() {
        let pred = ClinicalPredicate::HasCondition("ckd".to_string())
            .and(ClinicalPredicate::OnMedication("metformin".to_string()).negate());
        let json = serde_json::to_string(&pred).expect("serialize");
        let pred2: ClinicalPredicate = serde_json::from_str(&json).expect("deserialize");
        // Structural equality via description.
        assert_eq!(pred.description(), pred2.description());
    }

    // -- 12. Edge cases --

    #[test]
    fn test_edge_cases() {
        // Zero-height BMI
        let p = PatientProfile::new("E-1", 30, 70.0, 0.0, Sex::Unknown);
        assert_eq!(p.bmi(), 0.0);
        assert_eq!(p.bsa(), 0.0);
        assert!(!p.is_elderly());

        // Creatinine clearance without lab
        assert!(p.creatinine_clearance().is_none());

        // Renal function from egfr fallback
        assert_eq!(p.renal_function_from_egfr(), RenalFunction::Normal);

        // Predicate True / False
        let state = ClinicalState::empty();
        assert!(ClinicalPredicate::True.evaluate(&p, &state));
        assert!(!ClinicalPredicate::False.evaluate(&p, &state));
        assert!(ClinicalPredicate::True.is_trivially_true());
        assert!(!ClinicalPredicate::False.is_trivially_true());
    }

    // -- 13. Creatinine clearance Cockcroft-Gault --

    #[test]
    fn test_creatinine_clearance() {
        let p = sample_patient();
        let crcl = p.creatinine_clearance().expect("should have creatinine lab");
        // ((140 - 72) * 80) / (72 * 1.8) = 5440 / 129.6 ≈ 41.975
        assert!((crcl - 41.975).abs() < 0.1, "CrCl was {crcl}");
    }

    // -- 14. Enum Display / FromStr round-trips --

    #[test]
    fn test_enum_display_fromstr() {
        assert_eq!(Sex::from_str("male").unwrap(), Sex::Male);
        assert_eq!(Sex::from_str("F").unwrap(), Sex::Female);
        assert_eq!(format!("{}", Sex::Other), "Other");

        assert_eq!(
            HepaticFunction::from_str("moderate").unwrap(),
            HepaticFunction::ModerateImpairment
        );
        assert_eq!(HepaticFunction::MildImpairment.child_pugh_class(), "A");

        assert_eq!(
            RenalFunction::from_str("end stage").unwrap(),
            RenalFunction::EndStage
        );
        assert!(RenalFunction::SevereImpairment.requires_dose_adjustment());
        assert!(!RenalFunction::MildImpairment.requires_dose_adjustment());

        assert_eq!(
            MedicationStatus::from_str("on hold").unwrap(),
            MedicationStatus::OnHold
        );

        assert_eq!(ConditionSeverity::Critical.numeric_score(), 4);
        assert_eq!(ConditionSeverity::Unspecified.numeric_score(), 0);
    }

    // -- 15. ClinicalState merge --

    #[test]
    fn test_clinical_state_merge() {
        let mut s1 = ClinicalState::new();
        s1.set_condition("hypertension", true);
        s1.set_medication("lisinopril", true);

        let mut s2 = ClinicalState::new();
        s2.set_condition("diabetes", true);
        s2.set_medication("lisinopril", false);
        s2.set_lab_flag("elevated_potassium", true);

        s1.merge(&s2);

        assert!(s1.has_condition("hypertension"));
        assert!(s1.has_condition("diabetes"));
        // true || false → true
        assert!(s1.has_medication("lisinopril"));
        assert!(s1.has_lab_flag("elevated_potassium"));
    }

    // -- 16. Predicate description --

    #[test]
    fn test_predicate_description() {
        let pred = ClinicalPredicate::AgeRange { min: 18, max: 65 }
            .and(ClinicalPredicate::RenalFunction(RenalFunction::Normal));
        let desc = pred.description();
        assert!(desc.contains("age"));
        assert!(desc.contains("renal"));
        assert!(desc.contains("AND"));
    }

    // -- 17. LabInRange predicate --

    #[test]
    fn test_predicate_lab_in_range() {
        let p = sample_patient();
        let state = ClinicalState::empty();

        // INR is 2.5 — within therapeutic range [2.0, 3.0]
        let pred = ClinicalPredicate::LabInRange {
            name: "INR".to_string(),
            low: 2.0,
            high: 3.0,
        };
        assert!(pred.evaluate(&p, &state));

        // Out of range [0.8, 1.2]
        let pred2 = ClinicalPredicate::LabInRange {
            name: "INR".to_string(),
            low: 0.8,
            high: 1.2,
        };
        assert!(!pred2.evaluate(&p, &state));
    }
}
