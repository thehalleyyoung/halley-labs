//! Drug-related types for the GuardPharma polypharmacy verification system.
//!
//! Provides strongly-typed representations for drug identifiers, classifications,
//! dosing schedules, therapeutic windows, and toxicity thresholds.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// DrugId
// ---------------------------------------------------------------------------

/// Unique identifier for a drug, normalized to lowercase with underscores.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DrugId(String);

impl DrugId {
    /// Create a new `DrugId`, normalizing the input.
    pub fn new(s: impl Into<String>) -> Self {
        let raw = s.into();
        Self(raw.trim().to_lowercase().replace(' ', "_"))
    }

    /// Alias for `new` — create from a drug name string.
    pub fn from_name(s: impl Into<String>) -> Self {
        Self::new(s)
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DrugId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for DrugId {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

impl From<&str> for DrugId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for DrugId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

// ---------------------------------------------------------------------------
// DrugName
// ---------------------------------------------------------------------------

/// Human-readable drug name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DrugName(String);

impl DrugName {
    /// Create a new drug name.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return a lowercased version of this name.
    pub fn to_lowercase(&self) -> String {
        self.0.to_lowercase()
    }
}

impl fmt::Display for DrugName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for DrugName {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

impl From<&str> for DrugName {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for DrugName {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

// ---------------------------------------------------------------------------
// DrugClass
// ---------------------------------------------------------------------------

/// Pharmacological classification of a drug.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrugClass {
    NSAID,
    Anticoagulant,
    Antihypertensive,
    Antidiabetic,
    Statin,
    Antidepressant,
    Antibiotic,
    Antiarrhythmic,
    Opioid,
    Benzodiazepine,
    /// Proton pump inhibitor
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
    Analgesic,
    Antiepileptic,
    CardiacGlycoside,
    Antimetabolite,
    /// Catch-all for classes not yet enumerated.
    Other(String),
}

impl DrugClass {
    /// Returns `true` for classes that commonly require therapeutic drug monitoring.
    pub fn requires_monitoring(&self) -> bool {
        matches!(
            self,
            DrugClass::Anticoagulant
                | DrugClass::Antiarrhythmic
                | DrugClass::Anticonvulsant
                | DrugClass::Immunosuppressant
                | DrugClass::Antidepressant
        )
    }

    /// Returns `true` for cardiovascular drug classes.
    pub fn is_cardiovascular(&self) -> bool {
        matches!(
            self,
            DrugClass::Antihypertensive
                | DrugClass::Antiarrhythmic
                | DrugClass::ACEInhibitor
                | DrugClass::ARB
                | DrugClass::BetaBlocker
                | DrugClass::CalciumChannelBlocker
                | DrugClass::Diuretic
                | DrugClass::Statin
                | DrugClass::Antiplatelet
                | DrugClass::Anticoagulant
        )
    }

    /// Returns `true` for drug classes with high abuse/addiction potential.
    pub fn has_abuse_potential(&self) -> bool {
        matches!(
            self,
            DrugClass::Opioid | DrugClass::Benzodiazepine | DrugClass::Sedative
        )
    }

    /// Returns `true` for CNS-active drug classes.
    pub fn is_cns_active(&self) -> bool {
        matches!(
            self,
            DrugClass::Opioid
                | DrugClass::Benzodiazepine
                | DrugClass::Sedative
                | DrugClass::Antidepressant
                | DrugClass::Antipsychotic
                | DrugClass::Anticonvulsant
        )
    }

    /// Returns the broad therapeutic area as a string.
    pub fn therapeutic_area(&self) -> &str {
        match self {
            DrugClass::NSAID | DrugClass::Corticosteroid => "anti-inflammatory",
            DrugClass::Anticoagulant | DrugClass::Antiplatelet => "antithrombotic",
            DrugClass::Antihypertensive
            | DrugClass::ACEInhibitor
            | DrugClass::ARB
            | DrugClass::BetaBlocker
            | DrugClass::CalciumChannelBlocker
            | DrugClass::Diuretic => "cardiovascular",
            DrugClass::Statin => "lipid-lowering",
            DrugClass::Antidiabetic => "metabolic",
            DrugClass::Antidepressant | DrugClass::Antipsychotic => "psychiatric",
            DrugClass::Antibiotic | DrugClass::Antifungal | DrugClass::Antiviral => {
                "anti-infective"
            }
            DrugClass::Antiarrhythmic => "cardiac-electrophysiology",
            DrugClass::Opioid => "analgesic",
            DrugClass::Benzodiazepine | DrugClass::Sedative => "anxiolytic-sedative",
            DrugClass::PPI => "gastrointestinal",
            DrugClass::Anticonvulsant => "neurological",
            DrugClass::Immunosuppressant => "immunology",
            DrugClass::Bronchodilator => "respiratory",
            DrugClass::Analgesic => "analgesic",
            DrugClass::Antiepileptic => "neurological",
            DrugClass::CardiacGlycoside => "cardiovascular",
            DrugClass::Antimetabolite => "oncology",
            DrugClass::Other(_) => "other",
        }
    }
}

impl fmt::Display for DrugClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrugClass::NSAID => write!(f, "NSAID"),
            DrugClass::Anticoagulant => write!(f, "Anticoagulant"),
            DrugClass::Antihypertensive => write!(f, "Antihypertensive"),
            DrugClass::Antidiabetic => write!(f, "Antidiabetic"),
            DrugClass::Statin => write!(f, "Statin"),
            DrugClass::Antidepressant => write!(f, "Antidepressant"),
            DrugClass::Antibiotic => write!(f, "Antibiotic"),
            DrugClass::Antiarrhythmic => write!(f, "Antiarrhythmic"),
            DrugClass::Opioid => write!(f, "Opioid"),
            DrugClass::Benzodiazepine => write!(f, "Benzodiazepine"),
            DrugClass::PPI => write!(f, "PPI"),
            DrugClass::Anticonvulsant => write!(f, "Anticonvulsant"),
            DrugClass::Corticosteroid => write!(f, "Corticosteroid"),
            DrugClass::Immunosuppressant => write!(f, "Immunosuppressant"),
            DrugClass::Bronchodilator => write!(f, "Bronchodilator"),
            DrugClass::Diuretic => write!(f, "Diuretic"),
            DrugClass::ACEInhibitor => write!(f, "ACE Inhibitor"),
            DrugClass::ARB => write!(f, "ARB"),
            DrugClass::BetaBlocker => write!(f, "Beta Blocker"),
            DrugClass::CalciumChannelBlocker => write!(f, "Calcium Channel Blocker"),
            DrugClass::Antiplatelet => write!(f, "Antiplatelet"),
            DrugClass::Antipsychotic => write!(f, "Antipsychotic"),
            DrugClass::Sedative => write!(f, "Sedative"),
            DrugClass::Antifungal => write!(f, "Antifungal"),
            DrugClass::Antiviral => write!(f, "Antiviral"),
            DrugClass::Analgesic => write!(f, "Analgesic"),
            DrugClass::Antiepileptic => write!(f, "Antiepileptic"),
            DrugClass::CardiacGlycoside => write!(f, "Cardiac Glycoside"),
            DrugClass::Antimetabolite => write!(f, "Antimetabolite"),
            DrugClass::Other(s) => write!(f, "Other({})", s),
        }
    }
}

impl FromStr for DrugClass {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s
            .to_lowercase()
            .replace(' ', "")
            .replace('-', "")
            .replace('_', "")
            .as_str()
        {
            "nsaid" => Ok(DrugClass::NSAID),
            "anticoagulant" => Ok(DrugClass::Anticoagulant),
            "antihypertensive" => Ok(DrugClass::Antihypertensive),
            "antidiabetic" => Ok(DrugClass::Antidiabetic),
            "statin" => Ok(DrugClass::Statin),
            "antidepressant" => Ok(DrugClass::Antidepressant),
            "antibiotic" => Ok(DrugClass::Antibiotic),
            "antiarrhythmic" => Ok(DrugClass::Antiarrhythmic),
            "opioid" => Ok(DrugClass::Opioid),
            "benzodiazepine" => Ok(DrugClass::Benzodiazepine),
            "ppi" | "protonpumpinhibitor" => Ok(DrugClass::PPI),
            "anticonvulsant" => Ok(DrugClass::Anticonvulsant),
            "corticosteroid" => Ok(DrugClass::Corticosteroid),
            "immunosuppressant" => Ok(DrugClass::Immunosuppressant),
            "bronchodilator" => Ok(DrugClass::Bronchodilator),
            "diuretic" => Ok(DrugClass::Diuretic),
            "aceinhibitor" => Ok(DrugClass::ACEInhibitor),
            "arb" | "angiotensinreceptorblocker" => Ok(DrugClass::ARB),
            "betablocker" => Ok(DrugClass::BetaBlocker),
            "calciumchannelblocker" => Ok(DrugClass::CalciumChannelBlocker),
            "antiplatelet" => Ok(DrugClass::Antiplatelet),
            "antipsychotic" => Ok(DrugClass::Antipsychotic),
            "sedative" => Ok(DrugClass::Sedative),
            "antifungal" => Ok(DrugClass::Antifungal),
            "antiviral" => Ok(DrugClass::Antiviral),
            "analgesic" => Ok(DrugClass::Analgesic),
            "antiepileptic" => Ok(DrugClass::Antiepileptic),
            "cardiacglycoside" => Ok(DrugClass::CardiacGlycoside),
            "antimetabolite" => Ok(DrugClass::Antimetabolite),
            _ => Ok(DrugClass::Other(s.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// DrugRoute
// ---------------------------------------------------------------------------

/// Route of drug administration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrugRoute {
    Oral,
    Intravenous,
    Intramuscular,
    Subcutaneous,
    Topical,
    Inhalation,
    Sublingual,
    Rectal,
    Transdermal,
    Ophthalmic,
    Nasal,
    Other(String),
}

impl DrugRoute {
    /// Returns typical bioavailability range (low, high) for this route.
    pub fn typical_bioavailability_range(&self) -> (f64, f64) {
        match self {
            DrugRoute::Intravenous => (1.0, 1.0),
            DrugRoute::Oral => (0.05, 0.95),
            DrugRoute::Intramuscular => (0.75, 1.0),
            DrugRoute::Subcutaneous => (0.70, 1.0),
            DrugRoute::Sublingual => (0.30, 0.80),
            DrugRoute::Inhalation => (0.10, 0.90),
            DrugRoute::Rectal => (0.30, 0.80),
            DrugRoute::Transdermal => (0.10, 0.60),
            DrugRoute::Topical => (0.01, 0.10),
            DrugRoute::Ophthalmic => (0.01, 0.10),
            DrugRoute::Nasal => (0.10, 0.80),
            DrugRoute::Other(_) => (0.0, 1.0),
        }
    }

    /// Returns `true` if this is a systemic route (drug enters bloodstream).
    pub fn is_systemic(&self) -> bool {
        !matches!(self, DrugRoute::Topical | DrugRoute::Ophthalmic)
    }

    /// Returns typical onset time range in minutes.
    pub fn typical_onset_minutes(&self) -> (f64, f64) {
        match self {
            DrugRoute::Intravenous => (0.0, 5.0),
            DrugRoute::Sublingual => (1.0, 15.0),
            DrugRoute::Inhalation => (1.0, 15.0),
            DrugRoute::Intramuscular => (10.0, 30.0),
            DrugRoute::Subcutaneous => (15.0, 45.0),
            DrugRoute::Oral => (15.0, 120.0),
            DrugRoute::Rectal => (15.0, 60.0),
            DrugRoute::Nasal => (5.0, 20.0),
            DrugRoute::Transdermal => (60.0, 720.0),
            DrugRoute::Topical => (30.0, 240.0),
            DrugRoute::Ophthalmic => (5.0, 30.0),
            DrugRoute::Other(_) => (0.0, 1440.0),
        }
    }
}

impl fmt::Display for DrugRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrugRoute::Oral => write!(f, "Oral"),
            DrugRoute::Intravenous => write!(f, "Intravenous"),
            DrugRoute::Intramuscular => write!(f, "Intramuscular"),
            DrugRoute::Subcutaneous => write!(f, "Subcutaneous"),
            DrugRoute::Topical => write!(f, "Topical"),
            DrugRoute::Inhalation => write!(f, "Inhalation"),
            DrugRoute::Sublingual => write!(f, "Sublingual"),
            DrugRoute::Rectal => write!(f, "Rectal"),
            DrugRoute::Transdermal => write!(f, "Transdermal"),
            DrugRoute::Ophthalmic => write!(f, "Ophthalmic"),
            DrugRoute::Nasal => write!(f, "Nasal"),
            DrugRoute::Other(s) => write!(f, "Other({})", s),
        }
    }
}

impl FromStr for DrugRoute {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "oral" | "po" => Ok(DrugRoute::Oral),
            "intravenous" | "iv" => Ok(DrugRoute::Intravenous),
            "intramuscular" | "im" => Ok(DrugRoute::Intramuscular),
            "subcutaneous" | "sc" | "sq" | "subq" => Ok(DrugRoute::Subcutaneous),
            "topical" => Ok(DrugRoute::Topical),
            "inhalation" | "inh" => Ok(DrugRoute::Inhalation),
            "sublingual" | "sl" => Ok(DrugRoute::Sublingual),
            "rectal" | "pr" => Ok(DrugRoute::Rectal),
            "transdermal" | "td" => Ok(DrugRoute::Transdermal),
            "ophthalmic" => Ok(DrugRoute::Ophthalmic),
            "nasal" | "nas" => Ok(DrugRoute::Nasal),
            _ => Ok(DrugRoute::Other(s.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// DosingSchedule
// ---------------------------------------------------------------------------

/// Defines how a drug is dosed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DosingSchedule {
    /// Dosing interval in hours (e.g., 8.0 for TID).
    pub interval_hours: f64,
    /// Single dose amount in milligrams.
    pub dose_amount_mg: f64,
    /// Route of administration.
    pub route: DrugRoute,
    /// Maximum allowed daily dose in milligrams, if applicable.
    pub max_daily_dose_mg: Option<f64>,
    /// Loading dose in milligrams, if applicable.
    pub loading_dose_mg: Option<f64>,
    /// Number of discrete doses per day.
    pub doses_per_day: u32,
}

impl DosingSchedule {
    /// Create a new dosing schedule.
    pub fn new(
        interval_hours: f64,
        dose_amount_mg: f64,
        route: DrugRoute,
        doses_per_day: u32,
    ) -> Self {
        Self {
            interval_hours,
            dose_amount_mg,
            route,
            max_daily_dose_mg: None,
            loading_dose_mg: None,
            doses_per_day,
        }
    }

    /// Convenience constructor for oral dosing.
    pub fn oral(dose_mg: f64, times_per_day: u32) -> Self {
        let interval = if times_per_day > 0 {
            24.0 / times_per_day as f64
        } else {
            24.0
        };
        Self::new(interval, dose_mg, DrugRoute::Oral, times_per_day)
    }

    /// Set the maximum daily dose.
    pub fn with_max_daily(mut self, max_mg: f64) -> Self {
        self.max_daily_dose_mg = Some(max_mg);
        self
    }

    /// Set the loading dose.
    pub fn with_loading_dose(mut self, loading_mg: f64) -> Self {
        self.loading_dose_mg = Some(loading_mg);
        self
    }

    /// Set the route of administration.
    pub fn with_route(mut self, route: DrugRoute) -> Self {
        self.route = route;
        self
    }

    /// Calculate the total planned daily dose in milligrams.
    pub fn daily_dose(&self) -> f64 {
        self.dose_amount_mg * self.doses_per_day as f64
    }

    /// Returns `true` if the schedule parameters are within reasonable bounds.
    pub fn is_valid(&self) -> bool {
        self.interval_hours > 0.0
            && self.interval_hours <= 168.0
            && self.dose_amount_mg > 0.0
            && self.doses_per_day > 0
            && self.doses_per_day <= 24
            && self.max_daily_dose_mg.map_or(true, |m| m >= self.daily_dose())
    }

    /// Returns `true` if the daily dose exceeds the maximum, when one is set.
    pub fn exceeds_max_daily(&self) -> bool {
        self.max_daily_dose_mg
            .map_or(false, |m| self.daily_dose() > m)
    }

    /// Calculate the average hourly infusion rate (mg/hr).
    pub fn average_hourly_rate(&self) -> f64 {
        if self.interval_hours > 0.0 {
            self.dose_amount_mg / self.interval_hours
        } else {
            0.0
        }
    }
}

impl fmt::Display for DosingSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1} mg {} q{:.0}h ({}x/day)",
            self.dose_amount_mg, self.route, self.interval_hours, self.doses_per_day
        )
    }
}

// ---------------------------------------------------------------------------
// TherapeuticWindow
// ---------------------------------------------------------------------------

/// Defines the therapeutic concentration window for a drug.
///
/// Concentrations below `min_concentration` may be sub-therapeutic;
/// concentrations above `max_concentration` may produce toxicity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TherapeuticWindow {
    /// Minimum effective concentration.
    pub min_concentration: f64,
    /// Maximum safe concentration.
    pub max_concentration: f64,
    /// Concentration unit (e.g., "mcg/mL").
    pub unit: String,
}

impl TherapeuticWindow {
    /// Create a new therapeutic window.
    pub fn new(min: f64, max: f64, unit: impl Into<String>) -> Self {
        Self {
            min_concentration: min,
            max_concentration: max,
            unit: unit.into(),
        }
    }

    /// Returns `true` if `concentration` falls within the window (inclusive).
    pub fn contains(&self, concentration: f64) -> bool {
        concentration >= self.min_concentration && concentration <= self.max_concentration
    }

    /// Returns the width of the therapeutic window.
    pub fn width(&self) -> f64 {
        self.max_concentration - self.min_concentration
    }

    /// Returns the midpoint of the therapeutic window.
    pub fn midpoint(&self) -> f64 {
        (self.min_concentration + self.max_concentration) / 2.0
    }

    /// Returns `true` if min < max and both are non-negative.
    pub fn is_valid(&self) -> bool {
        self.min_concentration >= 0.0 && self.max_concentration > self.min_concentration
    }

    /// Returns `true` if the window is narrow (width < 2x min),
    /// indicating a narrow therapeutic index drug.
    pub fn is_narrow(&self) -> bool {
        self.is_valid() && self.width() < 2.0 * self.min_concentration
    }

    /// Compute fraction of the window traversed (0.0 at min, 1.0 at max).
    pub fn fraction(&self, concentration: f64) -> f64 {
        let w = self.width();
        if w <= 0.0 {
            return 0.0;
        }
        (concentration - self.min_concentration) / w
    }
}

impl fmt::Display for TherapeuticWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.2}-{:.2}] {}",
            self.min_concentration, self.max_concentration, self.unit
        )
    }
}

// ---------------------------------------------------------------------------
// ToxicThreshold
// ---------------------------------------------------------------------------

/// Concentration threshold above which a drug is considered toxic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToxicThreshold {
    /// Concentration at which toxicity begins.
    pub threshold_concentration: f64,
    /// Unit of concentration.
    pub unit: String,
    /// Severity description (e.g., "hepatotoxicity", "nephrotoxicity").
    pub severity: String,
}

impl ToxicThreshold {
    /// Create a new toxic threshold.
    pub fn new(threshold: f64, unit: impl Into<String>, severity: impl Into<String>) -> Self {
        Self {
            threshold_concentration: threshold,
            unit: unit.into(),
            severity: severity.into(),
        }
    }

    /// Returns `true` if `concentration` exceeds the toxic threshold.
    pub fn is_exceeded(&self, concentration: f64) -> bool {
        concentration > self.threshold_concentration
    }

    /// Returns the margin between the threshold and the given concentration.
    /// Positive means safe (below threshold), negative means toxic.
    pub fn margin_from(&self, concentration: f64) -> f64 {
        self.threshold_concentration - concentration
    }

    /// Returns the fraction of the threshold consumed (0.0 at 0, 1.0 at threshold).
    pub fn utilization(&self, concentration: f64) -> f64 {
        if self.threshold_concentration > 0.0 {
            concentration / self.threshold_concentration
        } else {
            f64::INFINITY
        }
    }
}

impl fmt::Display for ToxicThreshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            ">{:.2} {} ({})",
            self.threshold_concentration, self.unit, self.severity
        )
    }
}

// ---------------------------------------------------------------------------
// DrugInfo
// ---------------------------------------------------------------------------

/// Comprehensive metadata for a drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInfo {
    /// Unique drug identifier.
    pub id: DrugId,
    /// Human-readable drug name.
    pub name: DrugName,
    /// Pharmacological class.
    pub class: DrugClass,
    /// Default dosing schedule.
    pub schedule: DosingSchedule,
    /// Therapeutic window, if defined.
    pub therapeutic_window: Option<TherapeuticWindow>,
    /// Toxicity threshold, if defined.
    pub toxic_threshold: Option<ToxicThreshold>,
    /// Elimination half-life in hours.
    pub half_life_hours: f64,
    /// Fraction of drug reaching systemic circulation (0.0-1.0).
    pub bioavailability: f64,
    /// Fraction bound to plasma proteins (0.0-1.0).
    pub protein_binding_fraction: f64,
    /// Apparent volume of distribution in liters.
    pub volume_of_distribution_l: f64,
    /// Molecular weight in Daltons.
    pub molecular_weight_daltons: f64,
    /// Names of pharmacologically active metabolites.
    pub active_metabolites: Vec<String>,
    /// Known contraindications.
    pub contraindications: Vec<String>,
    /// Alternative names / brand names.
    pub aliases: Vec<String>,
}

impl DrugInfo {
    /// Create a `DrugInfo` with mandatory fields; optional fields default to empty/None.
    pub fn new(
        id: DrugId,
        name: DrugName,
        class: DrugClass,
        schedule: DosingSchedule,
        half_life_hours: f64,
        bioavailability: f64,
        protein_binding_fraction: f64,
        volume_of_distribution_l: f64,
    ) -> Self {
        Self {
            id,
            name,
            class,
            schedule,
            therapeutic_window: None,
            toxic_threshold: None,
            half_life_hours,
            bioavailability,
            protein_binding_fraction,
            volume_of_distribution_l,
            molecular_weight_daltons: 0.0,
            active_metabolites: Vec::new(),
            contraindications: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// Set the therapeutic window.
    pub fn with_therapeutic_window(mut self, tw: TherapeuticWindow) -> Self {
        self.therapeutic_window = Some(tw);
        self
    }

    /// Set the toxic threshold.
    pub fn with_toxic_threshold(mut self, tt: ToxicThreshold) -> Self {
        self.toxic_threshold = Some(tt);
        self
    }

    /// Set the molecular weight.
    pub fn with_molecular_weight(mut self, mw: f64) -> Self {
        self.molecular_weight_daltons = mw;
        self
    }

    /// Returns `true` if this drug is considered to have a narrow therapeutic index.
    pub fn is_narrow_therapeutic_index(&self) -> bool {
        self.therapeutic_window
            .as_ref()
            .map_or(false, |tw| tw.is_narrow())
    }

    /// Compute the effective dose reaching systemic circulation (mg).
    pub fn effective_dose(&self) -> f64 {
        self.schedule.dose_amount_mg * self.bioavailability
    }

    /// Estimate time to reach ~97% of steady-state (5 half-lives) in hours.
    pub fn time_to_steady_state(&self) -> f64 {
        5.0 * self.half_life_hours
    }

    /// Compute the elimination rate constant ke = ln(2) / t_half in per-hour.
    pub fn elimination_rate_constant(&self) -> f64 {
        if self.half_life_hours > 0.0 {
            std::f64::consts::LN_2 / self.half_life_hours
        } else {
            0.0
        }
    }

    /// Compute the clearance CL = ke * Vd in L/hr.
    pub fn clearance_l_per_hr(&self) -> f64 {
        self.elimination_rate_constant() * self.volume_of_distribution_l
    }

    /// Estimate the average steady-state concentration Css_avg in mg/L.
    pub fn estimated_steady_state_avg(&self) -> f64 {
        let cl = self.clearance_l_per_hr();
        if cl > 0.0 && self.schedule.interval_hours > 0.0 {
            (self.bioavailability * self.schedule.dose_amount_mg)
                / (cl * self.schedule.interval_hours)
        } else {
            0.0
        }
    }

    /// Returns the unbound (free) fraction of drug in plasma.
    pub fn free_fraction(&self) -> f64 {
        1.0 - self.protein_binding_fraction
    }

    /// Estimate peak concentration Cmax after a single dose: Cmax ~ (F * Dose) / Vd.
    pub fn estimated_cmax_single_dose(&self) -> f64 {
        if self.volume_of_distribution_l > 0.0 {
            (self.bioavailability * self.schedule.dose_amount_mg)
                / self.volume_of_distribution_l
        } else {
            0.0
        }
    }

    /// Returns `true` if any known contraindication substring-matches `query`.
    pub fn has_contraindication(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.contraindications
            .iter()
            .any(|c| c.to_lowercase().contains(&q))
    }

    /// Returns `true` if the drug has active metabolites.
    pub fn has_active_metabolites(&self) -> bool {
        !self.active_metabolites.is_empty()
    }
}

impl fmt::Display for DrugInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) [{}] t1/2={:.1}h F={:.0}%",
            self.name,
            self.id,
            self.class,
            self.half_life_hours,
            self.bioavailability * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Interaction severity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    None,
    Minor,
    Moderate,
    Major,
    Contraindicated,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::None => write!(f, "None"),
            Severity::Minor => write!(f, "Minor"),
            Severity::Moderate => write!(f, "Moderate"),
            Severity::Major => write!(f, "Major"),
            Severity::Contraindicated => write!(f, "Contraindicated"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sex, ChildPughClass, AscitesGrade
// ---------------------------------------------------------------------------

/// Biological sex for PK covariate modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
}

/// Child-Pugh hepatic function class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChildPughClass {
    A,
    B,
    C,
}

/// Ascites grade for Child-Pugh scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AscitesGrade {
    None,
    Mild,
    ModerateToSevere,
}

// ---------------------------------------------------------------------------
// PatientInfo
// ---------------------------------------------------------------------------

/// Patient demographic and clinical information for PK adjustments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientInfo {
    pub age_years: f64,
    pub weight_kg: f64,
    pub height_cm: f64,
    pub sex: Sex,
    pub serum_creatinine: f64,
    pub bilirubin: Option<f64>,
    pub albumin: Option<f64>,
    pub inr: Option<f64>,
    pub encephalopathy_grade: Option<u8>,
    pub ascites: Option<AscitesGrade>,
}

impl Default for PatientInfo {
    fn default() -> Self {
        Self {
            age_years: 50.0,
            weight_kg: 70.0,
            height_cm: 170.0,
            sex: Sex::Male,
            serum_creatinine: 1.0,
            bilirubin: None,
            albumin: None,
            inr: None,
            encephalopathy_grade: None,
            ascites: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn warfarin_info() -> DrugInfo {
        DrugInfo::new(
            DrugId::new("warfarin"),
            DrugName::new("Warfarin"),
            DrugClass::Anticoagulant,
            DosingSchedule::oral(5.0, 1),
            40.0,
            0.99,
            0.99,
            10.0,
        )
        .with_therapeutic_window(TherapeuticWindow::new(2.0, 3.0, "mcg/mL"))
        .with_toxic_threshold(ToxicThreshold::new(10.0, "mcg/mL", "hemorrhage"))
    }

    #[test]
    fn test_drug_id_normalization() {
        let id = DrugId::new("Warfarin Sodium");
        assert_eq!(id.as_str(), "warfarin_sodium");
    }

    #[test]
    fn test_drug_id_display_fromstr() {
        let id = DrugId::new("aspirin");
        let s = id.to_string();
        assert_eq!(s, "aspirin");
        let parsed: DrugId = s.parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_drug_name_display() {
        let name = DrugName::new("Metformin HCl");
        assert_eq!(name.to_string(), "Metformin HCl");
        assert_eq!(name.to_lowercase(), "metformin hcl");
    }

    #[test]
    fn test_drug_class_display_fromstr() {
        let cls = DrugClass::CalciumChannelBlocker;
        let s = cls.to_string();
        assert_eq!(s, "Calcium Channel Blocker");
        let parsed: DrugClass = "CalciumChannelBlocker".parse().unwrap();
        assert_eq!(parsed, DrugClass::CalciumChannelBlocker);
    }

    #[test]
    fn test_drug_class_properties() {
        assert!(DrugClass::Opioid.has_abuse_potential());
        assert!(DrugClass::Opioid.is_cns_active());
        assert!(!DrugClass::Antibiotic.has_abuse_potential());
        assert!(DrugClass::ACEInhibitor.is_cardiovascular());
        assert!(DrugClass::Anticoagulant.requires_monitoring());
        assert!(!DrugClass::Antibiotic.requires_monitoring());
    }

    #[test]
    fn test_drug_class_other() {
        let parsed: DrugClass = "MuscarnicAntagonist".parse().unwrap();
        assert_eq!(parsed, DrugClass::Other("MuscarnicAntagonist".to_string()));
    }

    #[test]
    fn test_drug_route_display_fromstr() {
        let r: DrugRoute = "iv".parse().unwrap();
        assert_eq!(r, DrugRoute::Intravenous);
        let r2: DrugRoute = "po".parse().unwrap();
        assert_eq!(r2, DrugRoute::Oral);
        let r3: DrugRoute = "subq".parse().unwrap();
        assert_eq!(r3, DrugRoute::Subcutaneous);
    }

    #[test]
    fn test_drug_route_bioavailability() {
        let (lo, hi) = DrugRoute::Intravenous.typical_bioavailability_range();
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 1.0);
        let (lo, _hi) = DrugRoute::Oral.typical_bioavailability_range();
        assert!(lo < 1.0);
        assert!(DrugRoute::Intravenous.is_systemic());
        assert!(!DrugRoute::Topical.is_systemic());
    }

    #[test]
    fn test_dosing_schedule_daily_dose() {
        let sched = DosingSchedule::oral(500.0, 3);
        assert!((sched.daily_dose() - 1500.0).abs() < 1e-9);
        assert!(sched.is_valid());
    }

    #[test]
    fn test_dosing_schedule_exceeds_max() {
        let sched = DosingSchedule::oral(500.0, 3).with_max_daily(1000.0);
        assert!(sched.exceeds_max_daily());
        assert!(!sched.is_valid());
    }

    #[test]
    fn test_dosing_schedule_display() {
        let sched = DosingSchedule::oral(250.0, 2);
        let s = sched.to_string();
        assert!(s.contains("250.0 mg"));
        assert!(s.contains("Oral"));
        assert!(s.contains("2x/day"));
    }

    #[test]
    fn test_therapeutic_window() {
        let tw = TherapeuticWindow::new(1.0, 4.0, "mcg/mL");
        assert!(tw.is_valid());
        assert!(tw.contains(2.5));
        assert!(!tw.contains(0.5));
        assert!(!tw.contains(5.0));
        assert!((tw.width() - 3.0).abs() < 1e-9);
        assert!((tw.midpoint() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_therapeutic_window_narrow() {
        let narrow = TherapeuticWindow::new(2.0, 3.0, "mcg/mL");
        assert!(narrow.is_narrow());
        let wide = TherapeuticWindow::new(2.0, 20.0, "mcg/mL");
        assert!(!wide.is_narrow());
    }

    #[test]
    fn test_toxic_threshold() {
        let tt = ToxicThreshold::new(10.0, "mcg/mL", "hepatotoxicity");
        assert!(!tt.is_exceeded(5.0));
        assert!(tt.is_exceeded(15.0));
        assert!((tt.margin_from(5.0) - 5.0).abs() < 1e-9);
        assert!((tt.utilization(5.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_drug_info_basic() {
        let info = warfarin_info();
        assert_eq!(info.id.as_str(), "warfarin");
        assert!((info.effective_dose() - 4.95).abs() < 1e-2);
        assert!((info.time_to_steady_state() - 200.0).abs() < 1e-9);
        assert!(info.is_narrow_therapeutic_index());
    }

    #[test]
    fn test_drug_info_pk_calculations() {
        let info = warfarin_info();
        let ke = info.elimination_rate_constant();
        assert!((ke - std::f64::consts::LN_2 / 40.0).abs() < 1e-9);
        assert!(info.clearance_l_per_hr() > 0.0);
        assert!(info.estimated_steady_state_avg() > 0.0);
        assert!(info.estimated_cmax_single_dose() > 0.0);
        assert!((info.free_fraction() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_drug_info_contraindications() {
        let mut info = warfarin_info();
        info.contraindications = vec![
            "Active bleeding".to_string(),
            "Pregnancy".to_string(),
        ];
        assert!(info.has_contraindication("bleeding"));
        assert!(info.has_contraindication("PREGNANCY"));
        assert!(!info.has_contraindication("diabetes"));
    }

    #[test]
    fn test_drug_info_display() {
        let info = warfarin_info();
        let s = info.to_string();
        assert!(s.contains("Warfarin"));
        assert!(s.contains("warfarin"));
        assert!(s.contains("Anticoagulant"));
    }

    #[test]
    fn test_serde_roundtrip_drug_info() {
        let info = warfarin_info();
        let json = serde_json::to_string(&info).expect("serialize");
        let parsed: DrugInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.id, info.id);
        assert_eq!(parsed.class, info.class);
        assert!((parsed.half_life_hours - info.half_life_hours).abs() < 1e-9);
    }

    #[test]
    fn test_serde_roundtrip_drug_class() {
        let cls = DrugClass::PPI;
        let json = serde_json::to_string(&cls).expect("serialize");
        let parsed: DrugClass = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, cls);
    }

    #[test]
    fn test_drug_class_therapeutic_area() {
        assert_eq!(DrugClass::NSAID.therapeutic_area(), "anti-inflammatory");
        assert_eq!(DrugClass::Statin.therapeutic_area(), "lipid-lowering");
        assert_eq!(DrugClass::Opioid.therapeutic_area(), "analgesic");
        assert_eq!(DrugClass::PPI.therapeutic_area(), "gastrointestinal");
    }
}
