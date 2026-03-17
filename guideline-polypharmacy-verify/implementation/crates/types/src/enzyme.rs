//! CYP enzyme types, enzyme activity modeling, inhibition/induction kinetics,
//! and drug-enzyme interaction descriptors for the GuardPharma verification engine.
//!
//! This module provides:
//! - [`CypEnzyme`] -- enumeration of Phase I/II metabolic enzymes
//! - [`EnzymeActivity`] -- normalized enzyme activity with inhibition/induction helpers
//! - [`EnzymeActivityInterval`] -- abstract-interpretation interval over enzyme activity
//! - [`InhibitionType`] -- kinetic inhibition mechanism classification
//! - [`InhibitionConstant`] -- Ki with typed inhibition mechanism
//! - [`EnzymeInhibitionEffect`] -- full inhibition descriptor (perpetrator, onset, offset)
//! - [`InductionEffect`] -- enzyme induction descriptor
//! - [`EnzymeMetabolismRoute`] -- which fraction of a drug is metabolized by a given enzyme
//! - [`DrugEnzymeInteraction`] -- composite interaction record between two drugs via an enzyme

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::drug::DrugId;

// ---------------------------------------------------------------------------
// 1. CypEnzyme
// ---------------------------------------------------------------------------

/// Major Phase I and Phase II drug-metabolizing enzymes.
///
/// Despite the name `CypEnzyme`, the enum also covers UGT, MAO, NAT2, COMT and
/// ALDH2 because they participate in clinically significant drug-drug
/// interactions handled by the verification engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum CypEnzyme {
    CYP1A2,
    CYP2B6,
    CYP2C8,
    CYP2C9,
    CYP2C19,
    CYP2D6,
    CYP2E1,
    CYP3A4,
    CYP3A5,
    CYP3A7,
    UGT1A1,
    UGT2B7,
    UGT2B15,
    #[serde(rename = "MAO-A")]
    MAO_A,
    #[serde(rename = "MAO-B")]
    MAO_B,
    NAT2,
    COMT,
    ALDH2,
}

impl CypEnzyme {
    /// All known enzyme variants.
    pub fn all() -> &'static [CypEnzyme] {
        &[
            CypEnzyme::CYP1A2,
            CypEnzyme::CYP2B6,
            CypEnzyme::CYP2C8,
            CypEnzyme::CYP2C9,
            CypEnzyme::CYP2C19,
            CypEnzyme::CYP2D6,
            CypEnzyme::CYP2E1,
            CypEnzyme::CYP3A4,
            CypEnzyme::CYP3A5,
            CypEnzyme::CYP3A7,
            CypEnzyme::UGT1A1,
            CypEnzyme::UGT2B7,
            CypEnzyme::UGT2B15,
            CypEnzyme::MAO_A,
            CypEnzyme::MAO_B,
            CypEnzyme::NAT2,
            CypEnzyme::COMT,
            CypEnzyme::ALDH2,
        ]
    }

    /// Enzyme super-family name (e.g. `"CYP"`, `"UGT"`, `"MAO"`).
    pub fn family(&self) -> &str {
        match self {
            CypEnzyme::CYP1A2
            | CypEnzyme::CYP2B6
            | CypEnzyme::CYP2C8
            | CypEnzyme::CYP2C9
            | CypEnzyme::CYP2C19
            | CypEnzyme::CYP2D6
            | CypEnzyme::CYP2E1
            | CypEnzyme::CYP3A4
            | CypEnzyme::CYP3A5
            | CypEnzyme::CYP3A7 => "CYP",
            CypEnzyme::UGT1A1 | CypEnzyme::UGT2B7 | CypEnzyme::UGT2B15 => "UGT",
            CypEnzyme::MAO_A | CypEnzyme::MAO_B => "MAO",
            CypEnzyme::NAT2 => "NAT",
            CypEnzyme::COMT => "COMT",
            CypEnzyme::ALDH2 => "ALDH",
        }
    }

    /// Returns `true` when the enzyme belongs to the cytochrome P450 family.
    pub fn is_cyp(&self) -> bool {
        self.family() == "CYP"
    }

    /// Returns `true` when the enzyme belongs to the UDP-glucuronosyltransferase family.
    pub fn is_ugt(&self) -> bool {
        self.family() == "UGT"
    }

    /// A short list of well-known substrate drug names for this enzyme.
    pub fn common_substrates(&self) -> &[&str] {
        match self {
            CypEnzyme::CYP1A2 => &["caffeine", "theophylline", "melatonin", "clozapine"],
            CypEnzyme::CYP2B6 => &["efavirenz", "bupropion", "cyclophosphamide"],
            CypEnzyme::CYP2C8 => &["paclitaxel", "repaglinide", "rosiglitazone"],
            CypEnzyme::CYP2C9 => &["warfarin", "phenytoin", "losartan", "celecoxib"],
            CypEnzyme::CYP2C19 => &["omeprazole", "clopidogrel", "diazepam", "escitalopram"],
            CypEnzyme::CYP2D6 => &["codeine", "tramadol", "metoprolol", "fluoxetine"],
            CypEnzyme::CYP2E1 => &["acetaminophen", "ethanol", "isoniazid"],
            CypEnzyme::CYP3A4 => &["midazolam", "simvastatin", "cyclosporine", "tacrolimus"],
            CypEnzyme::CYP3A5 => &["tacrolimus", "midazolam", "cyclosporine"],
            CypEnzyme::CYP3A7 => &["dehydroepiandrosterone", "retinoic_acid"],
            CypEnzyme::UGT1A1 => &["irinotecan", "bilirubin", "atazanavir"],
            CypEnzyme::UGT2B7 => &["morphine", "zidovudine", "valproic_acid"],
            CypEnzyme::UGT2B15 => &["lorazepam", "oxazepam", "sipoglitazar"],
            CypEnzyme::MAO_A => &["serotonin", "norepinephrine", "tyramine"],
            CypEnzyme::MAO_B => &["dopamine", "phenethylamine", "benzylamine"],
            CypEnzyme::NAT2 => &["isoniazid", "hydralazine", "procainamide"],
            CypEnzyme::COMT => &["levodopa", "entacapone", "catechol_estrogens"],
            CypEnzyme::ALDH2 => &["ethanol", "nitroglycerin", "cyclophosphamide_aldehyde"],
        }
    }

    /// Returns `true` when the enzyme is known to be highly polymorphic
    /// in the general population, leading to distinct metabolizer phenotypes.
    pub fn is_polymorphic(&self) -> bool {
        matches!(
            self,
            CypEnzyme::CYP2D6 | CypEnzyme::CYP2C19 | CypEnzyme::NAT2
        )
    }

    /// Typical hepatic abundance relative to CYP3A4 (CYP3A4 = 1.0).
    /// Non-CYP enzymes return 0.0 because the scale is CYP-specific.
    pub fn relative_abundance(&self) -> f64 {
        match self {
            CypEnzyme::CYP1A2 => 0.35,
            CypEnzyme::CYP2B6 => 0.10,
            CypEnzyme::CYP2C8 => 0.20,
            CypEnzyme::CYP2C9 => 0.50,
            CypEnzyme::CYP2C19 => 0.10,
            CypEnzyme::CYP2D6 => 0.07,
            CypEnzyme::CYP2E1 => 0.20,
            CypEnzyme::CYP3A4 => 1.00,
            CypEnzyme::CYP3A5 => 0.50,
            CypEnzyme::CYP3A7 => 0.05,
            _ => 0.0,
        }
    }

    /// Typical enzyme degradation half-life rate constant (h^-1).
    pub fn degradation_rate(&self) -> f64 {
        match self {
            CypEnzyme::CYP1A2 => 0.019,
            CypEnzyme::CYP2B6 => 0.014,
            CypEnzyme::CYP2C8 => 0.014,
            CypEnzyme::CYP2C9 => 0.014,
            CypEnzyme::CYP2C19 => 0.014,
            CypEnzyme::CYP2D6 => 0.014,
            CypEnzyme::CYP2E1 => 0.026,
            CypEnzyme::CYP3A4 => 0.019,
            CypEnzyme::CYP3A5 => 0.019,
            CypEnzyme::CYP3A7 => 0.019,
            CypEnzyme::UGT1A1 => 0.012,
            CypEnzyme::UGT2B7 => 0.012,
            CypEnzyme::UGT2B15 => 0.012,
            CypEnzyme::MAO_A => 0.010,
            CypEnzyme::MAO_B => 0.010,
            CypEnzyme::NAT2 => 0.015,
            CypEnzyme::COMT => 0.020,
            CypEnzyme::ALDH2 => 0.018,
        }
    }
}

impl fmt::Display for CypEnzyme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CypEnzyme::CYP1A2 => "CYP1A2",
            CypEnzyme::CYP2B6 => "CYP2B6",
            CypEnzyme::CYP2C8 => "CYP2C8",
            CypEnzyme::CYP2C9 => "CYP2C9",
            CypEnzyme::CYP2C19 => "CYP2C19",
            CypEnzyme::CYP2D6 => "CYP2D6",
            CypEnzyme::CYP2E1 => "CYP2E1",
            CypEnzyme::CYP3A4 => "CYP3A4",
            CypEnzyme::CYP3A5 => "CYP3A5",
            CypEnzyme::CYP3A7 => "CYP3A7",
            CypEnzyme::UGT1A1 => "UGT1A1",
            CypEnzyme::UGT2B7 => "UGT2B7",
            CypEnzyme::UGT2B15 => "UGT2B15",
            CypEnzyme::MAO_A => "MAO-A",
            CypEnzyme::MAO_B => "MAO-B",
            CypEnzyme::NAT2 => "NAT2",
            CypEnzyme::COMT => "COMT",
            CypEnzyme::ALDH2 => "ALDH2",
        };
        f.write_str(s)
    }
}

/// Error returned when parsing a string into [`CypEnzyme`] fails.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown enzyme: {0}")]
pub struct ParseEnzymeError(String);

impl FromStr for CypEnzyme {
    type Err = ParseEnzymeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let upper = s.to_uppercase().replace('-', "_").replace(' ', "");
        let result = match upper.as_str() {
            "CYP1A2" | "1A2" => Some(CypEnzyme::CYP1A2),
            "CYP2B6" | "2B6" => Some(CypEnzyme::CYP2B6),
            "CYP2C8" | "2C8" => Some(CypEnzyme::CYP2C8),
            "CYP2C9" | "2C9" => Some(CypEnzyme::CYP2C9),
            "CYP2C19" | "2C19" => Some(CypEnzyme::CYP2C19),
            "CYP2D6" | "2D6" => Some(CypEnzyme::CYP2D6),
            "CYP2E1" | "2E1" => Some(CypEnzyme::CYP2E1),
            "CYP3A4" | "3A4" => Some(CypEnzyme::CYP3A4),
            "CYP3A5" | "3A5" => Some(CypEnzyme::CYP3A5),
            "CYP3A7" | "3A7" => Some(CypEnzyme::CYP3A7),
            "UGT1A1" | "1A1" => Some(CypEnzyme::UGT1A1),
            "UGT2B7" | "2B7" => Some(CypEnzyme::UGT2B7),
            "UGT2B15" | "2B15" => Some(CypEnzyme::UGT2B15),
            "MAO_A" | "MAOA" => Some(CypEnzyme::MAO_A),
            "MAO_B" | "MAOB" => Some(CypEnzyme::MAO_B),
            "NAT2" => Some(CypEnzyme::NAT2),
            "COMT" => Some(CypEnzyme::COMT),
            "ALDH2" => Some(CypEnzyme::ALDH2),
            _ => None,
        };
        result.ok_or_else(|| ParseEnzymeError(s.to_string()))
    }
}

// ---------------------------------------------------------------------------
// 2. EnzymeActivity
// ---------------------------------------------------------------------------

/// Normalized enzyme activity state.
///
/// * `current_activity` -- current level in [0.0, max_activity] where 1.0 = full baseline
/// * `max_activity` -- maximum achievable (can exceed 1.0 during induction)
/// * `fraction_active` -- fraction of enzyme pool that is catalytically active [0.0, 1.0]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct EnzymeActivity {
    pub current_activity: f64,
    pub max_activity: f64,
    pub fraction_active: f64,
}

impl EnzymeActivity {
    /// Create a new `EnzymeActivity`, clamping inputs to valid ranges.
    pub fn new(current_activity: f64, max_activity: f64, fraction_active: f64) -> Self {
        let clamped_max = max_activity.max(0.0);
        let upper = clamped_max.max(1.0);
        Self {
            current_activity: current_activity.clamp(0.0, upper),
            max_activity: clamped_max,
            fraction_active: fraction_active.clamp(0.0, 1.0),
        }
    }

    /// Fully-active baseline state.
    pub fn normal() -> Self {
        Self {
            current_activity: 1.0,
            max_activity: 1.0,
            fraction_active: 1.0,
        }
    }

    /// Effective catalytic activity = current * fraction_active.
    pub fn effective_activity(&self) -> f64 {
        self.current_activity * self.fraction_active
    }

    /// True when effective activity is below baseline (1.0).
    pub fn is_inhibited(&self) -> bool {
        self.effective_activity() < 1.0 - f64::EPSILON
    }

    /// True when effective activity exceeds baseline (1.0).
    pub fn is_induced(&self) -> bool {
        self.effective_activity() > 1.0 + f64::EPSILON
    }

    /// Ratio of inhibition: 1.0 - effective_activity, clamped to [0, 1].
    pub fn inhibition_ratio(&self) -> f64 {
        (1.0 - self.effective_activity()).clamp(0.0, 1.0)
    }

    /// Return a new state after applying an inhibition that removes
    /// `fraction` of the activity (clamped to [0, 1]).
    pub fn with_inhibition(&self, fraction: f64) -> Self {
        let frac = fraction.clamp(0.0, 1.0);
        let new_current = (self.current_activity * (1.0 - frac)).max(0.0);
        Self {
            current_activity: new_current,
            max_activity: self.max_activity,
            fraction_active: self.fraction_active,
        }
    }

    /// Return a new state after applying an induction that multiplies
    /// activity by `fold` (>= 1.0). Clamps the result to max_activity * fold.
    pub fn with_induction(&self, fold: f64) -> Self {
        let fold = fold.max(1.0);
        let new_max = self.max_activity * fold;
        let new_current = (self.current_activity * fold).min(new_max);
        Self {
            current_activity: new_current,
            max_activity: new_max,
            fraction_active: self.fraction_active,
        }
    }

    /// Raw inner value (backwards compat with old newtype API).
    pub fn value(&self) -> f64 {
        self.effective_activity()
    }
}

impl Default for EnzymeActivity {
    fn default() -> Self {
        Self::normal()
    }
}

impl fmt::Display for EnzymeActivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}%", self.effective_activity() * 100.0)
    }
}

// ---------------------------------------------------------------------------
// 3. EnzymeActivityInterval -- abstract-interpretation interval [lo, hi]
// ---------------------------------------------------------------------------

/// Interval abstraction over enzyme activity for Tier-1 abstract
/// interpretation.  Values are normalized to [0.0, 1.0] under normal
/// conditions, but `hi` may exceed 1.0 during induction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct EnzymeActivityInterval {
    pub lo: f64,
    pub hi: f64,
}

impl EnzymeActivityInterval {
    /// Create a new interval, swapping bounds if necessary and clamping lo >= 0.
    pub fn new(a: f64, b: f64) -> Self {
        let lo = a.min(b).max(0.0);
        let hi = a.max(b).max(lo);
        Self { lo, hi }
    }

    /// The full-range "top" element [0.0, max].
    pub fn top(max: f64) -> Self {
        Self {
            lo: 0.0,
            hi: max.max(0.0),
        }
    }

    /// Bottom (empty) interval -- represented as lo > hi.
    pub fn bottom() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    /// True when this interval is the bottom element (empty set).
    pub fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    /// Point-containment test.
    pub fn contains(&self, v: f64) -> bool {
        !self.is_bottom() && v >= self.lo && v <= self.hi
    }

    /// Width of the interval (0 for bottom).
    pub fn width(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            (self.lo + self.hi) / 2.0
        }
    }

    /// Least upper bound (join / union).
    pub fn join(&self, other: &Self) -> Self {
        if self.is_bottom() {
            return *other;
        }
        if other.is_bottom() {
            return *self;
        }
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Greatest lower bound (meet / intersection).
    pub fn meet(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            Self::bottom()
        } else {
            Self { lo, hi }
        }
    }

    /// Standard widening operator: if the new interval exceeds the old bound
    /// in a direction, push that bound towards 0 or max respectively.
    pub fn widen(&self, newer: &Self, max: f64) -> Self {
        if self.is_bottom() {
            return *newer;
        }
        if newer.is_bottom() {
            return *self;
        }
        let lo = if newer.lo < self.lo { 0.0 } else { self.lo };
        let hi = if newer.hi > self.hi { max } else { self.hi };
        Self { lo, hi }
    }

    /// Narrowing operator: tighten bounds towards the newer iterate.
    pub fn narrow(&self, newer: &Self) -> Self {
        if self.is_bottom() {
            return *newer;
        }
        if newer.is_bottom() {
            return Self::bottom();
        }
        let lo = if self.lo == 0.0 { newer.lo } else { self.lo };
        let hi = if self.hi == f64::INFINITY {
            newer.hi
        } else {
            self.hi
        };
        Self { lo, hi }
    }

    /// Scale both bounds by a constant factor (clamp lo >= 0).
    pub fn scale(&self, factor: f64) -> Self {
        if self.is_bottom() || factor < 0.0 {
            return Self::bottom();
        }
        Self {
            lo: (self.lo * factor).max(0.0),
            hi: self.hi * factor,
        }
    }

    /// Apply an inhibition whose fractional effect lies in
    /// [`inh_lo`, `inh_hi`] within [0,1].  The remaining activity is
    /// multiplied by `(1 - inhibition)`.
    pub fn apply_inhibition_interval(&self, inh_lo: f64, inh_hi: f64) -> Self {
        if self.is_bottom() {
            return Self::bottom();
        }
        let inh_lo = inh_lo.clamp(0.0, 1.0);
        let inh_hi = inh_hi.clamp(0.0, 1.0);
        let new_lo = (self.lo * (1.0 - inh_hi)).max(0.0);
        let new_hi = self.hi * (1.0 - inh_lo);
        Self::new(new_lo, new_hi)
    }

    /// Apply an induction whose fold-change lies in [`ind_lo`, `ind_hi`]
    /// (both >= 1.0).
    pub fn apply_induction_interval(&self, ind_lo: f64, ind_hi: f64) -> Self {
        if self.is_bottom() {
            return Self::bottom();
        }
        let ind_lo = ind_lo.max(1.0);
        let ind_hi = ind_hi.max(ind_lo);
        Self {
            lo: self.lo * ind_lo,
            hi: self.hi * ind_hi,
        }
    }
}

impl fmt::Display for EnzymeActivityInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() {
            write!(f, "bottom")
        } else {
            write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
        }
    }
}

impl Default for EnzymeActivityInterval {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// 4. InhibitionType
// ---------------------------------------------------------------------------

/// Kinetic mechanism of enzyme inhibition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InhibitionType {
    Competitive,
    NonCompetitive,
    Uncompetitive,
    MechanismBased,
    Mixed,
}

impl fmt::Display for InhibitionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            InhibitionType::Competitive => "Competitive",
            InhibitionType::NonCompetitive => "Non-Competitive",
            InhibitionType::Uncompetitive => "Uncompetitive",
            InhibitionType::MechanismBased => "Mechanism-Based",
            InhibitionType::Mixed => "Mixed",
        };
        f.write_str(s)
    }
}

/// Error returned when parsing a string into [`InhibitionType`] fails.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown inhibition type: {0}")]
pub struct ParseInhibitionTypeError(String);

impl FromStr for InhibitionType {
    type Err = ParseInhibitionTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase().replace(['-', '_', ' '], "");
        match lower.as_str() {
            "competitive" => Ok(InhibitionType::Competitive),
            "noncompetitive" => Ok(InhibitionType::NonCompetitive),
            "uncompetitive" => Ok(InhibitionType::Uncompetitive),
            "mechanismbased" | "mechanism" => Ok(InhibitionType::MechanismBased),
            "mixed" => Ok(InhibitionType::Mixed),
            _ => Err(ParseInhibitionTypeError(s.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// 5. InhibitionConstant
// ---------------------------------------------------------------------------

/// Inhibition constant bundled with its kinetic type.
///
/// `ki_micromolar` is the reversible inhibition constant in uM.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct InhibitionConstant {
    pub ki_micromolar: f64,
    pub inhibition_type: InhibitionType,
}

impl InhibitionConstant {
    pub fn new(ki_micromolar: f64, inhibition_type: InhibitionType) -> Self {
        Self {
            ki_micromolar: ki_micromolar.max(f64::EPSILON),
            inhibition_type,
        }
    }

    /// Compute the **fraction of remaining enzyme activity** given
    /// substrate and inhibitor concentrations (both in uM).
    ///
    /// Returns a value in (0, 1] where 1.0 means no inhibition.
    ///
    /// Kinetic formulae applied per inhibition type:
    ///
    /// - **Competitive**: remaining = (Km + [S]) / (Km*(1 + [I]/Ki) + [S])
    ///   When [S] is unknown we assume [S] = Km so Km is set to Ki as proxy.
    ///
    /// - **NonCompetitive**: remaining = 1 / (1 + [I]/Ki)
    ///
    /// - **Uncompetitive**: remaining = (1 + Km/[S]) / (1 + Km/[S] + [I]/Ki)
    ///   Falls back to 1/(1+[I]/Ki) when [S] is zero.
    ///
    /// - **MechanismBased**: remaining = Ki / (Ki + [I])  (reversible proxy)
    ///
    /// - **Mixed** (alpha = 1): remaining = 1 / (1 + [I]/Ki)
    pub fn effective_inhibition(
        &self,
        substrate_conc: f64,
        inhibitor_conc: f64,
    ) -> f64 {
        let i = inhibitor_conc.max(0.0);
        let ki = self.ki_micromolar;

        if i <= 0.0 {
            return 1.0;
        }

        match self.inhibition_type {
            InhibitionType::Competitive => {
                let km = if substrate_conc > 0.0 {
                    substrate_conc
                } else {
                    ki
                };
                let s = substrate_conc.max(0.0);
                let denom = km * (1.0 + i / ki) + s;
                let numer = km + s;
                if denom <= 0.0 {
                    return 1.0;
                }
                (numer / denom).clamp(0.0, 1.0)
            }
            InhibitionType::NonCompetitive => {
                (1.0 / (1.0 + i / ki)).clamp(0.0, 1.0)
            }
            InhibitionType::Uncompetitive => {
                if substrate_conc > 0.0 {
                    let km_over_s = ki / substrate_conc;
                    let numer = 1.0 + km_over_s;
                    let denom = numer + i / ki;
                    (numer / denom).clamp(0.0, 1.0)
                } else {
                    (1.0 / (1.0 + i / ki)).clamp(0.0, 1.0)
                }
            }
            InhibitionType::MechanismBased => {
                (ki / (ki + i)).clamp(0.0, 1.0)
            }
            InhibitionType::Mixed => {
                (1.0 / (1.0 + i / ki)).clamp(0.0, 1.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. EnzymeInhibitionEffect
// ---------------------------------------------------------------------------

/// Complete description of how one drug inhibits a specific enzyme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeInhibitionEffect {
    pub enzyme: CypEnzyme,
    pub inhibitor_drug: DrugId,
    pub ki: InhibitionConstant,
    /// Maximum fractional inhibition achievable (0.0-1.0).
    pub max_inhibition_fraction: f64,
    /// Time in hours until inhibition onset becomes measurable.
    pub onset_hours: f64,
    /// Time in hours until activity recovers after inhibitor withdrawal.
    pub offset_hours: f64,
}

impl EnzymeInhibitionEffect {
    /// An inhibition is clinically significant when:
    /// - max inhibition fraction >= 0.25 (>= 25% reduction), **or**
    /// - Ki < 1 uM (potent inhibitor).
    pub fn is_clinically_significant(&self) -> bool {
        self.max_inhibition_fraction >= 0.25 || self.ki.ki_micromolar < 1.0
    }

    /// Estimated time (hours) to reach peak inhibition.
    /// For mechanism-based inhibitors this is roughly 3x onset;
    /// for reversible inhibitors it coincides with Cmax (approx onset).
    pub fn time_to_peak_effect(&self) -> f64 {
        match self.ki.inhibition_type {
            InhibitionType::MechanismBased => self.onset_hours * 3.0,
            _ => self.onset_hours,
        }
    }
}

// ---------------------------------------------------------------------------
// 7. InductionEffect
// ---------------------------------------------------------------------------

/// Description of enzyme induction by a drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InductionEffect {
    pub enzyme: CypEnzyme,
    pub inducer_drug: DrugId,
    /// Maximum fold-change in enzyme expression (e.g., 3.0 = 3x baseline).
    pub max_fold_induction: f64,
    /// Concentration producing half-maximal induction (uM).
    pub ec50_micromolar: f64,
    /// Days until induction onset becomes measurable.
    pub onset_days: f64,
    /// Days until activity returns to baseline after inducer withdrawal.
    pub offset_days: f64,
}

impl InductionEffect {
    /// Predicted fold-induction at a given inducer concentration (uM)
    /// using a Hill-type (Emax) model:
    ///
    /// `fold = 1 + (max_fold - 1) * [C] / (EC50 + [C])`
    pub fn fold_induction_at_concentration(&self, conc_micromolar: f64) -> f64 {
        let c = conc_micromolar.max(0.0);
        let emax = self.max_fold_induction - 1.0;
        if emax <= 0.0 || self.ec50_micromolar <= 0.0 {
            return 1.0;
        }
        1.0 + emax * c / (self.ec50_micromolar + c)
    }

    /// An induction is clinically significant when the maximum fold-change
    /// exceeds 2x (i.e., potential for >= 50% increase in victim clearance).
    pub fn is_clinically_significant(&self) -> bool {
        self.max_fold_induction >= 2.0
    }
}

// ---------------------------------------------------------------------------
// 8. EnzymeMetabolismRoute
// ---------------------------------------------------------------------------

/// Fraction of a drug metabolized through a specific enzyme pathway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeMetabolismRoute {
    pub drug_id: DrugId,
    pub enzyme: CypEnzyme,
    /// Fraction of total clearance attributed to this enzyme [0, 1].
    pub fraction_metabolized: f64,
    /// True when this enzyme is the primary metabolic route.
    pub is_primary: bool,
    /// Names of the metabolites produced via this route.
    pub metabolite_names: Vec<String>,
}

impl EnzymeMetabolismRoute {
    /// Construct a new route, clamping `fraction_metabolized` to [0, 1].
    pub fn new(
        drug_id: DrugId,
        enzyme: CypEnzyme,
        fraction_metabolized: f64,
        is_primary: bool,
        metabolite_names: Vec<String>,
    ) -> Self {
        Self {
            drug_id,
            enzyme,
            fraction_metabolized: fraction_metabolized.clamp(0.0, 1.0),
            is_primary,
            metabolite_names,
        }
    }

    /// A pathway is considered "major" when it accounts for > 30% of clearance.
    pub fn is_major_pathway(&self) -> bool {
        self.fraction_metabolized > 0.3
    }
}

// ---------------------------------------------------------------------------
// 9. DrugEnzymeInteraction
// ---------------------------------------------------------------------------

/// A composite record describing how a perpetrator drug affects a victim drug
/// through a shared metabolic enzyme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugEnzymeInteraction {
    pub perpetrator_drug: DrugId,
    pub victim_drug: DrugId,
    pub enzyme: CypEnzyme,
    pub inhibition: Option<EnzymeInhibitionEffect>,
    pub induction: Option<InductionEffect>,
    pub clinical_significance: String,
    pub evidence_level: String,
}

impl DrugEnzymeInteraction {
    /// Estimate the net fold-change on enzyme activity caused by this
    /// interaction at a given perpetrator concentration.
    ///
    /// Returns a value where:
    /// - `< 1.0` means net inhibition (activity decreased)
    /// - `= 1.0` means no change
    /// - `> 1.0` means net induction (activity increased)
    ///
    /// When both inhibition and induction are present, the net effect is
    /// `induction_fold * fraction_remaining`.
    pub fn net_effect_on_activity(&self, perpetrator_conc_um: f64) -> f64 {
        let remaining = self
            .inhibition
            .as_ref()
            .map(|inh| {
                inh.ki.effective_inhibition(0.0, perpetrator_conc_um)
            })
            .unwrap_or(1.0);

        let fold = self
            .induction
            .as_ref()
            .map(|ind| ind.fold_induction_at_concentration(perpetrator_conc_um))
            .unwrap_or(1.0);

        fold * remaining
    }

    /// True when this interaction has both an inhibition and an induction
    /// component.
    pub fn is_bidirectional(&self) -> bool {
        self.inhibition.is_some() && self.induction.is_some()
    }

    /// Returns a human-readable direction label.
    pub fn interaction_direction(&self) -> &str {
        match (self.inhibition.is_some(), self.induction.is_some()) {
            (true, true) => "mixed",
            (true, false) => "inhibition",
            (false, true) => "induction",
            (false, false) => "none",
        }
    }
}

// ===========================================================================
//  Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CypEnzyme Display / FromStr --

    #[test]
    fn test_cyp_enzyme_display() {
        assert_eq!(CypEnzyme::CYP3A4.to_string(), "CYP3A4");
        assert_eq!(CypEnzyme::MAO_A.to_string(), "MAO-A");
        assert_eq!(CypEnzyme::UGT1A1.to_string(), "UGT1A1");
        assert_eq!(CypEnzyme::ALDH2.to_string(), "ALDH2");
    }

    #[test]
    fn test_cyp_enzyme_from_str_canonical() {
        assert_eq!("CYP3A4".parse::<CypEnzyme>().unwrap(), CypEnzyme::CYP3A4);
        assert_eq!("NAT2".parse::<CypEnzyme>().unwrap(), CypEnzyme::NAT2);
        assert_eq!("UGT2B7".parse::<CypEnzyme>().unwrap(), CypEnzyme::UGT2B7);
    }

    #[test]
    fn test_cyp_enzyme_from_str_case_insensitive() {
        assert_eq!("cyp1a2".parse::<CypEnzyme>().unwrap(), CypEnzyme::CYP1A2);
        assert_eq!("Cyp2D6".parse::<CypEnzyme>().unwrap(), CypEnzyme::CYP2D6);
        assert_eq!("comt".parse::<CypEnzyme>().unwrap(), CypEnzyme::COMT);
    }

    #[test]
    fn test_cyp_enzyme_from_str_short_form() {
        assert_eq!("1A2".parse::<CypEnzyme>().unwrap(), CypEnzyme::CYP1A2);
        assert_eq!("3A4".parse::<CypEnzyme>().unwrap(), CypEnzyme::CYP3A4);
        assert_eq!("2C19".parse::<CypEnzyme>().unwrap(), CypEnzyme::CYP2C19);
    }

    #[test]
    fn test_cyp_enzyme_from_str_invalid() {
        assert!("XYZ123".parse::<CypEnzyme>().is_err());
    }

    #[test]
    fn test_cyp_enzyme_family() {
        assert_eq!(CypEnzyme::CYP2D6.family(), "CYP");
        assert_eq!(CypEnzyme::UGT2B15.family(), "UGT");
        assert_eq!(CypEnzyme::MAO_B.family(), "MAO");
        assert_eq!(CypEnzyme::NAT2.family(), "NAT");
        assert_eq!(CypEnzyme::COMT.family(), "COMT");
        assert_eq!(CypEnzyme::ALDH2.family(), "ALDH");
    }

    #[test]
    fn test_cyp_enzyme_polymorphic() {
        assert!(CypEnzyme::CYP2D6.is_polymorphic());
        assert!(CypEnzyme::CYP2C19.is_polymorphic());
        assert!(CypEnzyme::NAT2.is_polymorphic());
        assert!(!CypEnzyme::CYP3A4.is_polymorphic());
    }

    // -- EnzymeActivity --

    #[test]
    fn test_enzyme_activity_effective() {
        let ea = EnzymeActivity::new(0.8, 1.0, 0.5);
        let eff = ea.effective_activity();
        assert!((eff - 0.4).abs() < 1e-9);
        assert!(ea.is_inhibited());
    }

    #[test]
    fn test_enzyme_activity_with_inhibition_and_induction() {
        let normal = EnzymeActivity::normal();
        let inhibited = normal.with_inhibition(0.5);
        assert!((inhibited.effective_activity() - 0.5).abs() < 1e-9);

        let induced = normal.with_induction(2.0);
        assert!((induced.effective_activity() - 2.0).abs() < 1e-9);
        assert!(induced.is_induced());
    }

    // -- InhibitionConstant kinetics --

    #[test]
    fn test_competitive_inhibition_kinetics() {
        let ki = InhibitionConstant::new(1.0, InhibitionType::Competitive);
        // With substrate_conc = 1 uM (= Km proxy) and inhibitor = 1 uM:
        // Km = [S] = 1, denom = 1*(1+1)+1 = 3, numer = 1+1 = 2  -> remaining = 2/3
        let remaining = ki.effective_inhibition(1.0, 1.0);
        assert!((remaining - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_noncompetitive_inhibition_kinetics() {
        let ki = InhibitionConstant::new(5.0, InhibitionType::NonCompetitive);
        // [I] = 5 uM, Ki = 5  -> remaining = 1/(1+1) = 0.5
        let remaining = ki.effective_inhibition(10.0, 5.0);
        assert!((remaining - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mechanism_based_inhibition() {
        let ki = InhibitionConstant::new(2.0, InhibitionType::MechanismBased);
        // [I] = 2, Ki = 2  -> remaining = 2/(2+2) = 0.5
        let remaining = ki.effective_inhibition(0.0, 2.0);
        assert!((remaining - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_no_inhibitor_gives_full_activity() {
        let ki = InhibitionConstant::new(1.0, InhibitionType::NonCompetitive);
        let remaining = ki.effective_inhibition(5.0, 0.0);
        assert!((remaining - 1.0).abs() < 1e-9);
    }

    // -- EnzymeActivityInterval --

    #[test]
    fn test_interval_join_meet() {
        let a = EnzymeActivityInterval::new(0.2, 0.6);
        let b = EnzymeActivityInterval::new(0.4, 0.9);
        let joined = a.join(&b);
        assert!((joined.lo - 0.2).abs() < 1e-9);
        assert!((joined.hi - 0.9).abs() < 1e-9);

        let met = a.meet(&b);
        assert!((met.lo - 0.4).abs() < 1e-9);
        assert!((met.hi - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_interval_bottom() {
        let a = EnzymeActivityInterval::new(0.5, 0.6);
        let b = EnzymeActivityInterval::new(0.7, 0.9);
        let met = a.meet(&b);
        assert!(met.is_bottom());
    }

    #[test]
    fn test_interval_apply_inhibition() {
        let iv = EnzymeActivityInterval::new(0.8, 1.0);
        let result = iv.apply_inhibition_interval(0.2, 0.5);
        // new_lo = 0.8*(1-0.5) = 0.4, new_hi = 1.0*(1-0.2) = 0.8
        assert!((result.lo - 0.4).abs() < 1e-9);
        assert!((result.hi - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_interval_apply_induction() {
        let iv = EnzymeActivityInterval::new(0.9, 1.0);
        let result = iv.apply_induction_interval(1.5, 2.0);
        assert!((result.lo - 1.35).abs() < 1e-9);
        assert!((result.hi - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_interval_widen() {
        let old = EnzymeActivityInterval::new(0.3, 0.7);
        let new_iter = EnzymeActivityInterval::new(0.2, 0.8);
        let widened = old.widen(&new_iter, 2.0);
        assert!((widened.lo - 0.0).abs() < 1e-9);
        assert!((widened.hi - 2.0).abs() < 1e-9);
    }

    // -- Clinical significance --

    #[test]
    fn test_inhibition_effect_clinical_significance() {
        let effect = EnzymeInhibitionEffect {
            enzyme: CypEnzyme::CYP3A4,
            inhibitor_drug: DrugId::new("ketoconazole"),
            ki: InhibitionConstant::new(0.015, InhibitionType::Competitive),
            max_inhibition_fraction: 0.90,
            onset_hours: 1.0,
            offset_hours: 24.0,
        };
        assert!(effect.is_clinically_significant());
        assert!((effect.time_to_peak_effect() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_induction_effect_fold_and_significance() {
        let effect = InductionEffect {
            enzyme: CypEnzyme::CYP3A4,
            inducer_drug: DrugId::new("rifampin"),
            max_fold_induction: 10.0,
            ec50_micromolar: 0.5,
            onset_days: 3.0,
            offset_days: 14.0,
        };
        assert!(effect.is_clinically_significant());
        // At conc = EC50 = 0.5 uM: fold = 1 + 9*0.5/1.0 = 5.5
        let fold = effect.fold_induction_at_concentration(0.5);
        assert!((fold - 5.5).abs() < 1e-9);
    }

    // -- DrugEnzymeInteraction --

    #[test]
    fn test_interaction_direction_inhibition_only() {
        let interaction = DrugEnzymeInteraction {
            perpetrator_drug: DrugId::new("fluconazole"),
            victim_drug: DrugId::new("warfarin"),
            enzyme: CypEnzyme::CYP2C9,
            inhibition: Some(EnzymeInhibitionEffect {
                enzyme: CypEnzyme::CYP2C9,
                inhibitor_drug: DrugId::new("fluconazole"),
                ki: InhibitionConstant::new(7.0, InhibitionType::Competitive),
                max_inhibition_fraction: 0.60,
                onset_hours: 2.0,
                offset_hours: 48.0,
            }),
            induction: None,
            clinical_significance: "Major".to_string(),
            evidence_level: "Established".to_string(),
        };
        assert_eq!(interaction.interaction_direction(), "inhibition");
        assert!(!interaction.is_bidirectional());
        let net = interaction.net_effect_on_activity(7.0);
        assert!(net < 1.0);
    }

    #[test]
    fn test_interaction_direction_mixed() {
        let interaction = DrugEnzymeInteraction {
            perpetrator_drug: DrugId::new("carbamazepine"),
            victim_drug: DrugId::new("midazolam"),
            enzyme: CypEnzyme::CYP3A4,
            inhibition: Some(EnzymeInhibitionEffect {
                enzyme: CypEnzyme::CYP3A4,
                inhibitor_drug: DrugId::new("carbamazepine"),
                ki: InhibitionConstant::new(100.0, InhibitionType::Competitive),
                max_inhibition_fraction: 0.10,
                onset_hours: 1.0,
                offset_hours: 12.0,
            }),
            induction: Some(InductionEffect {
                enzyme: CypEnzyme::CYP3A4,
                inducer_drug: DrugId::new("carbamazepine"),
                max_fold_induction: 3.0,
                ec50_micromolar: 10.0,
                onset_days: 5.0,
                offset_days: 14.0,
            }),
            clinical_significance: "Major".to_string(),
            evidence_level: "Probable".to_string(),
        };
        assert_eq!(interaction.interaction_direction(), "mixed");
        assert!(interaction.is_bidirectional());
    }

    // -- EnzymeMetabolismRoute --

    #[test]
    fn test_metabolism_route_major_pathway() {
        let route = EnzymeMetabolismRoute::new(
            DrugId::new("midazolam"),
            CypEnzyme::CYP3A4,
            0.95,
            true,
            vec!["1-hydroxymidazolam".to_string()],
        );
        assert!(route.is_major_pathway());
        assert!(route.is_primary);
    }

    #[test]
    fn test_metabolism_route_minor_pathway() {
        let route = EnzymeMetabolismRoute::new(
            DrugId::new("midazolam"),
            CypEnzyme::CYP3A5,
            0.05,
            false,
            vec![],
        );
        assert!(!route.is_major_pathway());
    }

    // -- Serde round-trip --

    #[test]
    fn test_cyp_enzyme_serde_roundtrip() {
        for enzyme in CypEnzyme::all() {
            let json = serde_json::to_string(enzyme).unwrap();
            let back: CypEnzyme = serde_json::from_str(&json).unwrap();
            assert_eq!(*enzyme, back);
        }
    }

    #[test]
    fn test_inhibition_constant_serde_roundtrip() {
        let ki = InhibitionConstant::new(3.14, InhibitionType::MechanismBased);
        let json = serde_json::to_string(&ki).unwrap();
        let back: InhibitionConstant = serde_json::from_str(&json).unwrap();
        assert_eq!(ki, back);
    }

    #[test]
    fn test_enzyme_activity_interval_serde_roundtrip() {
        let iv = EnzymeActivityInterval::new(0.25, 0.75);
        let json = serde_json::to_string(&iv).unwrap();
        let back: EnzymeActivityInterval = serde_json::from_str(&json).unwrap();
        assert_eq!(iv, back);
    }

    // -- InhibitionType Display / FromStr --

    #[test]
    fn test_inhibition_type_display_fromstr() {
        let cases = [
            (InhibitionType::Competitive, "Competitive"),
            (InhibitionType::NonCompetitive, "Non-Competitive"),
            (InhibitionType::Uncompetitive, "Uncompetitive"),
            (InhibitionType::MechanismBased, "Mechanism-Based"),
            (InhibitionType::Mixed, "Mixed"),
        ];
        for (variant, expected) in &cases {
            assert_eq!(&variant.to_string(), expected);
        }
        assert_eq!(
            "mechanism-based".parse::<InhibitionType>().unwrap(),
            InhibitionType::MechanismBased
        );
        assert_eq!(
            "non_competitive".parse::<InhibitionType>().unwrap(),
            InhibitionType::NonCompetitive
        );
    }

    // -- All variants count --

    #[test]
    fn test_all_enzymes_count() {
        assert_eq!(CypEnzyme::all().len(), 18);
    }

    #[test]
    fn test_cyp_and_ugt_predicates() {
        assert!(CypEnzyme::CYP3A4.is_cyp());
        assert!(!CypEnzyme::CYP3A4.is_ugt());
        assert!(CypEnzyme::UGT1A1.is_ugt());
        assert!(!CypEnzyme::UGT1A1.is_cyp());
        assert!(!CypEnzyme::COMT.is_cyp());
        assert!(!CypEnzyme::COMT.is_ugt());
    }
}
