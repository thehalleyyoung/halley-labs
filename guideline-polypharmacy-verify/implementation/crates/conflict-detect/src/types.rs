//! Core domain types for the conflict detection crate.
//!
//! All types are defined locally so this crate can compile independently
//! of other guardpharma crates. Types include drug identifiers, patient
//! profiles, conflict descriptions, safety verdicts, and certificates.

use std::fmt;

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Newtype identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a drug or active ingredient.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct DrugId(pub String);

impl DrugId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
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
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for DrugId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl PartialOrd for DrugId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DrugId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

/// Unique identifier for a clinical guideline.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GuidelineId(pub String);

impl GuidelineId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GuidelineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for GuidelineId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for GuidelineId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Unique identifier for a patient.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatientId(pub String);

impl PatientId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PatientId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for PatientId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for PatientId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// Severity level of a detected drug–drug conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ConflictSeverity {
    /// Unlikely to be clinically significant.
    Minor = 0,
    /// May require monitoring or dose adjustment.
    Moderate = 1,
    /// Requires intervention; may cause serious adverse effects.
    Major = 2,
    /// Life-threatening; contraindicated combination.
    Critical = 3,
}

impl ConflictSeverity {
    /// Returns a numeric score in [0, 10] for the severity.
    pub fn numeric_score(self) -> f64 {
        match self {
            Self::Minor => 1.0,
            Self::Moderate => 4.0,
            Self::Major => 7.0,
            Self::Critical => 10.0,
        }
    }

    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Minor => "Minor",
            Self::Moderate => "Moderate",
            Self::Major => "Major",
            Self::Critical => "Critical",
        }
    }

    /// Parse from a numeric score (rounds to nearest level).
    pub fn from_score(score: f64) -> Self {
        if score >= 8.5 {
            Self::Critical
        } else if score >= 5.5 {
            Self::Major
        } else if score >= 2.5 {
            Self::Moderate
        } else {
            Self::Minor
        }
    }

    /// Returns all variants in increasing severity order.
    pub fn all() -> &'static [ConflictSeverity] {
        &[Self::Minor, Self::Moderate, Self::Major, Self::Critical]
    }
}

impl fmt::Display for ConflictSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Overall safety verdict for a drug combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyVerdict {
    /// No conflicts detected; combination is safe under stated assumptions.
    Safe,
    /// No confirmed conflicts, but analysis has limited coverage.
    PossiblySafe,
    /// Potential conflicts detected that may be clinically significant.
    PossiblyUnsafe,
    /// Confirmed dangerous interaction; combination is contraindicated.
    Unsafe,
}

impl SafetyVerdict {
    /// Returns a numeric risk score in [0.0, 1.0].
    pub fn risk_score(self) -> f64 {
        match self {
            Self::Safe => 0.0,
            Self::PossiblySafe => 0.25,
            Self::PossiblyUnsafe => 0.65,
            Self::Unsafe => 1.0,
        }
    }

    /// Returns `true` if the verdict indicates any level of concern.
    pub fn is_concerning(self) -> bool {
        matches!(self, Self::PossiblyUnsafe | Self::Unsafe)
    }

    /// Combine two verdicts, keeping the more severe.
    pub fn merge(self, other: Self) -> Self {
        let order = |v: &SafetyVerdict| -> u8 {
            match v {
                SafetyVerdict::Safe => 0,
                SafetyVerdict::PossiblySafe => 1,
                SafetyVerdict::PossiblyUnsafe => 2,
                SafetyVerdict::Unsafe => 3,
            }
        };
        if order(&self) >= order(&other) {
            self
        } else {
            other
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Safe => "Safe",
            Self::PossiblySafe => "Possibly Safe",
            Self::PossiblyUnsafe => "Possibly Unsafe",
            Self::Unsafe => "Unsafe",
        }
    }
}

impl fmt::Display for SafetyVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Which tier of the verification pipeline produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationTier {
    /// Tier 1: abstract interpretation (fast, approximate).
    Tier1Abstract,
    /// Tier 2: bounded model checking (slow, precise).
    Tier2ModelCheck,
}

impl fmt::Display for VerificationTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tier1Abstract => write!(f, "Tier 1 (Abstract Interpretation)"),
            Self::Tier2ModelCheck => write!(f, "Tier 2 (Model Checking)"),
        }
    }
}

/// Type of pharmacokinetic/pharmacodynamic interaction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    /// Drug inhibits a CYP enzyme, raising substrate concentrations.
    CypInhibition { enzyme: String },
    /// Drug induces a CYP enzyme, lowering substrate concentrations.
    CypInduction { enzyme: String },
    /// Drug displaces another from plasma protein binding sites.
    ProteinBindingDisplacement,
    /// Drugs compete for renal elimination pathways.
    RenalCompetition,
    /// Drugs have additive/synergistic pharmacodynamic effects.
    PharmacodynamicSynergy,
    /// Drugs have opposing pharmacodynamic effects.
    PharmacodynamicAntagonism,
    /// One drug alters gastrointestinal absorption of another.
    AbsorptionAlteration,
    /// Both drugs prolong the QT interval on ECG.
    QtProlongation,
    /// Both drugs increase serotonergic activity, risking serotonin syndrome.
    SerotoninSyndrome,
    /// Both drugs cause additive CNS depression.
    CnsDepression,
}

impl InteractionType {
    /// Returns the pharmacokinetic or pharmacodynamic category.
    pub fn category(&self) -> &'static str {
        match self {
            Self::CypInhibition { .. }
            | Self::CypInduction { .. }
            | Self::ProteinBindingDisplacement
            | Self::RenalCompetition
            | Self::AbsorptionAlteration => "Pharmacokinetic",
            Self::PharmacodynamicSynergy
            | Self::PharmacodynamicAntagonism
            | Self::QtProlongation
            | Self::SerotoninSyndrome
            | Self::CnsDepression => "Pharmacodynamic",
        }
    }

    /// Baseline severity weight for this interaction type.
    pub fn base_severity_weight(&self) -> f64 {
        match self {
            Self::QtProlongation => 9.0,
            Self::SerotoninSyndrome => 8.5,
            Self::CnsDepression => 7.0,
            Self::CypInhibition { .. } => 6.0,
            Self::CypInduction { .. } => 5.5,
            Self::PharmacodynamicSynergy => 5.0,
            Self::PharmacodynamicAntagonism => 4.5,
            Self::ProteinBindingDisplacement => 4.0,
            Self::RenalCompetition => 3.5,
            Self::AbsorptionAlteration => 3.0,
        }
    }

    /// Short description of the mechanism.
    pub fn description(&self) -> String {
        match self {
            Self::CypInhibition { enzyme } => {
                format!("CYP {} enzyme inhibition increases substrate plasma levels", enzyme)
            }
            Self::CypInduction { enzyme } => {
                format!("CYP {} enzyme induction decreases substrate plasma levels", enzyme)
            }
            Self::ProteinBindingDisplacement => {
                "Plasma protein binding displacement increases free drug fraction".to_string()
            }
            Self::RenalCompetition => {
                "Competition for renal tubular secretion delays elimination".to_string()
            }
            Self::PharmacodynamicSynergy => {
                "Additive or synergistic pharmacodynamic effects".to_string()
            }
            Self::PharmacodynamicAntagonism => {
                "Opposing pharmacodynamic effects reduce therapeutic efficacy".to_string()
            }
            Self::AbsorptionAlteration => {
                "Altered gastrointestinal absorption changes bioavailability".to_string()
            }
            Self::QtProlongation => {
                "Combined QT prolongation increases risk of torsades de pointes".to_string()
            }
            Self::SerotoninSyndrome => {
                "Combined serotonergic activity risks serotonin syndrome (hyperthermia, rigidity, clonus)".to_string()
            }
            Self::CnsDepression => {
                "Additive CNS depression risks respiratory depression and excessive sedation".to_string()
            }
        }
    }
}

impl fmt::Display for InteractionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Route of drug administration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdministrationRoute {
    Oral,
    Intravenous,
    Intramuscular,
    Subcutaneous,
    Topical,
    Inhalation,
}

impl AdministrationRoute {
    /// Approximate bioavailability multiplier for the route.
    pub fn bioavailability_factor(&self) -> f64 {
        match self {
            Self::Intravenous => 1.0,
            Self::Intramuscular => 0.85,
            Self::Subcutaneous => 0.75,
            Self::Oral => 0.60,
            Self::Inhalation => 0.50,
            Self::Topical => 0.10,
        }
    }
}

impl fmt::Display for AdministrationRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Oral => "Oral",
            Self::Intravenous => "IV",
            Self::Intramuscular => "IM",
            Self::Subcutaneous => "SC",
            Self::Topical => "Topical",
            Self::Inhalation => "Inhalation",
        };
        write!(f, "{}", s)
    }
}

/// Organ function assessment used for dose adjustment decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrganFunction {
    Normal,
    MildImpairment,
    ModerateImpairment,
    SevereImpairment,
}

impl OrganFunction {
    /// Clearance multiplier (1.0 = normal).
    pub fn clearance_factor(self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::MildImpairment => 0.75,
            Self::ModerateImpairment => 0.50,
            Self::SevereImpairment => 0.25,
        }
    }
}

impl fmt::Display for OrganFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Normal => "Normal",
            Self::MildImpairment => "Mild Impairment",
            Self::ModerateImpairment => "Moderate Impairment",
            Self::SevereImpairment => "Severe Impairment",
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// Composite structs
// ---------------------------------------------------------------------------

/// Basic information about a drug or active ingredient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInfo {
    pub id: DrugId,
    pub name: String,
    pub therapeutic_class: String,
    pub cyp_enzymes: Vec<String>,
    pub half_life_hours: f64,
    pub bioavailability: f64,
    pub protein_binding: f64,
    pub therapeutic_index: Option<f64>,
}

impl DrugInfo {
    /// Create a minimal DrugInfo for testing.
    pub fn simple(id: &str, name: &str) -> Self {
        Self {
            id: DrugId::new(id),
            name: name.to_string(),
            therapeutic_class: "Unknown".to_string(),
            cyp_enzymes: Vec::new(),
            half_life_hours: 12.0,
            bioavailability: 0.8,
            protein_binding: 0.5,
            therapeutic_index: None,
        }
    }

    /// Returns `true` if the drug has a narrow therapeutic index.
    pub fn is_narrow_therapeutic_index(&self) -> bool {
        self.therapeutic_index.map_or(false, |ti| ti < 3.0)
    }

    /// Estimated volume of distribution (simplified one-compartment).
    pub fn estimated_vd_liters(&self, weight_kg: f64) -> f64 {
        let base_vd_per_kg = if self.protein_binding > 0.9 {
            0.15
        } else if self.protein_binding > 0.7 {
            0.5
        } else {
            1.0
        };
        base_vd_per_kg * weight_kg
    }

    /// Elimination rate constant (ke = ln(2) / t½).
    pub fn elimination_rate(&self) -> f64 {
        if self.half_life_hours > 0.0 {
            (2.0_f64).ln() / self.half_life_hours
        } else {
            0.0
        }
    }
}

/// Dosage specification for a medication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dosage {
    pub amount_mg: f64,
    pub frequency_hours: f64,
    pub route: AdministrationRoute,
}

impl Dosage {
    pub fn new(amount_mg: f64, frequency_hours: f64, route: AdministrationRoute) -> Self {
        Self {
            amount_mg,
            frequency_hours,
            route,
        }
    }

    /// Daily dose in mg.
    pub fn daily_dose_mg(&self) -> f64 {
        if self.frequency_hours > 0.0 {
            self.amount_mg * (24.0 / self.frequency_hours)
        } else {
            self.amount_mg
        }
    }

    /// Effective absorbed dose per administration.
    pub fn effective_dose_mg(&self) -> f64 {
        self.amount_mg * self.route.bioavailability_factor()
    }
}

/// A single medication entry in a patient record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationRecord {
    pub drug: DrugInfo,
    pub dosage: Dosage,
    pub start_date: Option<String>,
    pub indication: Option<String>,
}

impl MedicationRecord {
    pub fn new(drug: DrugInfo, dosage: Dosage) -> Self {
        Self {
            drug,
            dosage,
            start_date: None,
            indication: None,
        }
    }

    /// Steady-state average plasma concentration (simplified).
    /// Css_avg = (F * D) / (CL * tau) where CL = ke * Vd.
    pub fn steady_state_concentration(&self, weight_kg: f64) -> f64 {
        let f = self.dosage.route.bioavailability_factor() * self.drug.bioavailability;
        let dose = self.dosage.amount_mg;
        let ke = self.drug.elimination_rate();
        let vd = self.drug.estimated_vd_liters(weight_kg);
        let tau = self.dosage.frequency_hours;
        let cl = ke * vd;
        if cl > 0.0 && tau > 0.0 {
            (f * dose) / (cl * tau)
        } else {
            0.0
        }
    }
}

/// Patient demographic and clinical profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientProfile {
    pub id: PatientId,
    pub age: u32,
    pub weight_kg: f64,
    pub medications: Vec<MedicationRecord>,
    pub conditions: Vec<String>,
    pub allergies: Vec<String>,
    pub renal_function: OrganFunction,
    pub hepatic_function: OrganFunction,
}

impl PatientProfile {
    /// Create a minimal patient for testing.
    pub fn simple(id: &str, age: u32, weight_kg: f64) -> Self {
        Self {
            id: PatientId::new(id),
            age,
            weight_kg,
            medications: Vec::new(),
            conditions: Vec::new(),
            allergies: Vec::new(),
            renal_function: OrganFunction::Normal,
            hepatic_function: OrganFunction::Normal,
        }
    }

    /// Estimated creatinine clearance (Cockcroft-Gault).
    pub fn estimated_crcl(&self) -> f64 {
        let age_factor = (140.0 - self.age as f64).max(0.0);
        let base_crcl = (age_factor * self.weight_kg) / 72.0;
        base_crcl * self.renal_function.clearance_factor()
    }

    /// Combined clearance factor accounting for both renal and hepatic function.
    pub fn combined_clearance_factor(&self) -> f64 {
        let renal = self.renal_function.clearance_factor();
        let hepatic = self.hepatic_function.clearance_factor();
        (renal + hepatic) / 2.0
    }

    /// Returns all drug IDs in the patient's medication list.
    pub fn drug_ids(&self) -> Vec<DrugId> {
        self.medications.iter().map(|m| m.drug.id.clone()).collect()
    }

    /// Number of concurrent medications.
    pub fn medication_count(&self) -> usize {
        self.medications.len()
    }
}

// ---------------------------------------------------------------------------
// Verification and conflict types
// ---------------------------------------------------------------------------

/// A single step in a verification trace (state trajectory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub time_hours: f64,
    pub state_description: String,
    pub drug_concentrations: IndexMap<String, f64>,
    pub safety_invariant_holds: bool,
    pub notes: Vec<String>,
}

impl TraceStep {
    pub fn new(time_hours: f64, description: &str) -> Self {
        Self {
            time_hours,
            state_description: description.to_string(),
            drug_concentrations: IndexMap::new(),
            safety_invariant_holds: true,
            notes: Vec::new(),
        }
    }

    pub fn with_concentration(mut self, drug: &str, conc: f64) -> Self {
        self.drug_concentrations.insert(drug.to_string(), conc);
        self
    }

    pub fn with_violation(mut self, note: &str) -> Self {
        self.safety_invariant_holds = false;
        self.notes.push(note.to_string());
        self
    }
}

/// A counterexample demonstrating a safety violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterExample {
    pub trace: Vec<TraceStep>,
    pub violated_property: String,
    pub violation_time_hours: f64,
    pub description: String,
}

impl CounterExample {
    /// Length of the counterexample trace.
    pub fn trace_length(&self) -> usize {
        self.trace.len()
    }

    /// Returns the trace step where the violation occurs.
    pub fn violation_step(&self) -> Option<&TraceStep> {
        self.trace.iter().find(|s| !s.safety_invariant_holds)
    }

    /// Summarize the counterexample in one line.
    pub fn summary(&self) -> String {
        format!(
            "Violation of '{}' at t={:.1}h: {}",
            self.violated_property, self.violation_time_hours, self.description
        )
    }
}

/// A confirmed drug–drug interaction conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmedConflict {
    pub id: String,
    pub drugs: Vec<DrugId>,
    pub interaction_type: InteractionType,
    pub severity: ConflictSeverity,
    pub verdict: SafetyVerdict,
    pub mechanism_description: String,
    pub evidence_tier: VerificationTier,
    pub counter_example: Option<CounterExample>,
    pub confidence: f64,
    pub clinical_recommendation: String,
    pub affected_parameters: Vec<String>,
    pub guideline_references: Vec<GuidelineId>,
}

impl ConfirmedConflict {
    /// Generate a unique ID from the drug pair and interaction type.
    pub fn generate_id(drugs: &[DrugId], interaction: &InteractionType) -> String {
        let mut sorted: Vec<&str> = drugs.iter().map(|d| d.as_str()).collect();
        sorted.sort();
        let drug_part = sorted.join("-");
        let type_part = match interaction {
            InteractionType::CypInhibition { enzyme } => format!("cyp_inh_{}", enzyme),
            InteractionType::CypInduction { enzyme } => format!("cyp_ind_{}", enzyme),
            InteractionType::ProteinBindingDisplacement => "pbd".to_string(),
            InteractionType::RenalCompetition => "renal".to_string(),
            InteractionType::PharmacodynamicSynergy => "pd_syn".to_string(),
            InteractionType::PharmacodynamicAntagonism => "pd_ant".to_string(),
            InteractionType::AbsorptionAlteration => "absorb".to_string(),
            InteractionType::QtProlongation => "qt".to_string(),
            InteractionType::SerotoninSyndrome => "serotonin".to_string(),
            InteractionType::CnsDepression => "cns_dep".to_string(),
        };
        format!("conflict_{}_{}", drug_part, type_part)
    }

    /// Returns `true` if this conflict has a counterexample trace.
    pub fn has_counterexample(&self) -> bool {
        self.counter_example.is_some()
    }

    /// Returns the pair of drug names as a tuple (if exactly two drugs).
    pub fn drug_pair(&self) -> Option<(&DrugId, &DrugId)> {
        if self.drugs.len() == 2 {
            Some((&self.drugs[0], &self.drugs[1]))
        } else {
            None
        }
    }

    /// Weighted risk score combining severity, confidence, and verdict.
    pub fn risk_score(&self) -> f64 {
        let sev = self.severity.numeric_score();
        let verd = self.verdict.risk_score();
        (sev * 0.5 + verd * 10.0 * 0.3 + self.confidence * 10.0 * 0.2).min(10.0)
    }
}

/// Result of verifying a specific drug pair at a given tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub drug_pair: (DrugId, DrugId),
    pub tier: VerificationTier,
    pub verdict: SafetyVerdict,
    pub conflicts: Vec<ConfirmedConflict>,
    pub trace: Option<Vec<TraceStep>>,
    pub duration_ms: u64,
    pub notes: Vec<String>,
}

impl VerificationResult {
    /// Returns `true` if any conflicts were found.
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }

    /// Highest severity among detected conflicts.
    pub fn max_severity(&self) -> Option<ConflictSeverity> {
        self.conflicts.iter().map(|c| c.severity).max()
    }

    /// Total number of conflicts.
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }
}

/// A safety certificate for a drug combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCertificate {
    pub id: String,
    pub patient_id: PatientId,
    pub medications: Vec<DrugId>,
    pub verdict: SafetyVerdict,
    pub conflicts: Vec<ConfirmedConflict>,
    pub methodology: String,
    pub assumptions: Vec<String>,
    pub evidence_summary: Vec<String>,
    pub generated_at: String,
    pub valid_until: Option<String>,
    pub confidence_score: f64,
}

impl SafetyCertificate {
    /// Returns `true` if the certificate indicates a safe combination.
    pub fn is_safe(&self) -> bool {
        matches!(self.verdict, SafetyVerdict::Safe | SafetyVerdict::PossiblySafe)
    }

    /// Number of conflicts recorded.
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    /// Highest severity found across all conflicts.
    pub fn max_severity(&self) -> Option<ConflictSeverity> {
        self.conflicts.iter().map(|c| c.severity).max()
    }
}

/// Concentration interval used in abstract interpretation tier.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConcentrationInterval {
    pub lower: OrderedFloat<f64>,
    pub upper: OrderedFloat<f64>,
}

impl ConcentrationInterval {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self {
            lower: OrderedFloat(lower),
            upper: OrderedFloat(upper),
        }
    }

    pub fn midpoint(&self) -> f64 {
        (*self.lower + *self.upper) / 2.0
    }

    pub fn width(&self) -> f64 {
        *self.upper - *self.lower
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= *self.lower && value <= *self.upper
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        *self.lower <= *other.upper && *other.lower <= *self.upper
    }

    /// Widen this interval by a factor.
    pub fn widen(&self, factor: f64) -> Self {
        let mid = self.midpoint();
        let half = self.width() / 2.0 * factor;
        Self::new(mid - half, mid + half)
    }

    /// Intersect two intervals; returns None if disjoint.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = (*self.lower).max(*other.lower);
        let hi = (*self.upper).min(*other.upper);
        if lo <= hi {
            Some(Self::new(lo, hi))
        } else {
            None
        }
    }
}

impl fmt::Display for ConcentrationInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.3}, {:.3}]", *self.lower, *self.upper)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drug_id_display_and_eq() {
        let id1 = DrugId::new("warfarin");
        let id2 = DrugId::from("warfarin");
        assert_eq!(id1, id2);
        assert_eq!(id1.to_string(), "warfarin");
    }

    #[test]
    fn test_conflict_severity_ordering() {
        assert!(ConflictSeverity::Minor < ConflictSeverity::Moderate);
        assert!(ConflictSeverity::Moderate < ConflictSeverity::Major);
        assert!(ConflictSeverity::Major < ConflictSeverity::Critical);
    }

    #[test]
    fn test_conflict_severity_from_score() {
        assert_eq!(ConflictSeverity::from_score(1.0), ConflictSeverity::Minor);
        assert_eq!(ConflictSeverity::from_score(4.0), ConflictSeverity::Moderate);
        assert_eq!(ConflictSeverity::from_score(7.0), ConflictSeverity::Major);
        assert_eq!(ConflictSeverity::from_score(10.0), ConflictSeverity::Critical);
    }

    #[test]
    fn test_safety_verdict_merge() {
        assert_eq!(
            SafetyVerdict::Safe.merge(SafetyVerdict::Unsafe),
            SafetyVerdict::Unsafe
        );
        assert_eq!(
            SafetyVerdict::PossiblyUnsafe.merge(SafetyVerdict::PossiblySafe),
            SafetyVerdict::PossiblyUnsafe
        );
    }

    #[test]
    fn test_safety_verdict_risk_score() {
        assert!(SafetyVerdict::Safe.risk_score() < SafetyVerdict::PossiblySafe.risk_score());
        assert!(SafetyVerdict::PossiblySafe.risk_score() < SafetyVerdict::PossiblyUnsafe.risk_score());
        assert!(SafetyVerdict::PossiblyUnsafe.risk_score() < SafetyVerdict::Unsafe.risk_score());
    }

    #[test]
    fn test_drug_info_elimination_rate() {
        let drug = DrugInfo::simple("ibuprofen", "Ibuprofen");
        let ke = drug.elimination_rate();
        assert!((ke - 2.0_f64.ln() / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_dosage_daily_dose() {
        let dosage = Dosage::new(500.0, 8.0, AdministrationRoute::Oral);
        assert!((dosage.daily_dose_mg() - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_patient_profile_crcl() {
        let patient = PatientProfile::simple("p1", 60, 70.0);
        let crcl = patient.estimated_crcl();
        let expected = ((140.0 - 60.0) * 70.0) / 72.0;
        assert!((crcl - expected).abs() < 1e-6);
    }

    #[test]
    fn test_concentration_interval_operations() {
        let a = ConcentrationInterval::new(1.0, 5.0);
        let b = ConcentrationInterval::new(3.0, 7.0);
        assert!(a.overlaps(&b));
        let isect = a.intersect(&b).unwrap();
        assert!((isect.midpoint() - 4.0).abs() < 1e-10);
        assert!((isect.width() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_counter_example_summary() {
        let ce = CounterExample {
            trace: vec![
                TraceStep::new(0.0, "initial")
                    .with_concentration("warfarin", 2.0),
                TraceStep::new(4.0, "toxic level")
                    .with_concentration("warfarin", 15.0)
                    .with_violation("warfarin exceeds safe range"),
            ],
            violated_property: "C_max < 10 mg/L".to_string(),
            violation_time_hours: 4.0,
            description: "warfarin concentration exceeded safe range".to_string(),
        };
        assert_eq!(ce.trace_length(), 2);
        assert!(ce.violation_step().is_some());
        assert!(ce.summary().contains("C_max"));
    }

    #[test]
    fn test_interaction_type_metadata() {
        let inh = InteractionType::CypInhibition {
            enzyme: "3A4".to_string(),
        };
        assert_eq!(inh.category(), "Pharmacokinetic");
        assert!(inh.base_severity_weight() > 5.0);

        let qt = InteractionType::QtProlongation;
        assert_eq!(qt.category(), "Pharmacodynamic");
        assert!(qt.base_severity_weight() > 8.0);
    }

    #[test]
    fn test_confirmed_conflict_risk_score() {
        let conflict = ConfirmedConflict {
            id: "test".to_string(),
            drugs: vec![DrugId::new("a"), DrugId::new("b")],
            interaction_type: InteractionType::QtProlongation,
            severity: ConflictSeverity::Critical,
            verdict: SafetyVerdict::Unsafe,
            mechanism_description: "test".to_string(),
            evidence_tier: VerificationTier::Tier2ModelCheck,
            counter_example: None,
            confidence: 0.95,
            clinical_recommendation: "Avoid".to_string(),
            affected_parameters: vec!["QTc".to_string()],
            guideline_references: vec![],
        };
        assert!(conflict.risk_score() > 7.0);
    }

    #[test]
    fn test_organ_function_clearance() {
        assert_eq!(OrganFunction::Normal.clearance_factor(), 1.0);
        assert!(OrganFunction::SevereImpairment.clearance_factor() < 0.5);
    }

    #[test]
    fn test_medication_record_steady_state() {
        let drug = DrugInfo {
            id: DrugId::new("test"),
            name: "Test Drug".to_string(),
            therapeutic_class: "Test".to_string(),
            cyp_enzymes: vec![],
            half_life_hours: 6.0,
            bioavailability: 1.0,
            protein_binding: 0.5,
            therapeutic_index: None,
        };
        let dosage = Dosage::new(100.0, 6.0, AdministrationRoute::Intravenous);
        let rec = MedicationRecord::new(drug, dosage);
        let css = rec.steady_state_concentration(70.0);
        assert!(css > 0.0, "Steady-state concentration should be positive");
    }

    #[test]
    fn test_concentration_interval_disjoint() {
        let a = ConcentrationInterval::new(1.0, 3.0);
        let b = ConcentrationInterval::new(5.0, 7.0);
        assert!(!a.overlaps(&b));
        assert!(a.intersect(&b).is_none());
    }
}
