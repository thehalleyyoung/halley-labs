//! Abstract domains for pharmacokinetic abstract interpretation.
//!
//! Provides a hierarchy of lattice-based abstract domains used in Tier 1
//! screening of drug–drug interactions. Every domain implements the
//! [`AbstractValue`] trait which supplies the standard lattice operations:
//! join (⊔), meet (⊓), widening (∇), narrowing (Δ), and ordering.
//!
//! # Domain hierarchy
//!
//! ```text
//! ProductDomain
//! ├── ConcentrationAbstractDomain   (DrugId → ConcentrationInterval)
//! ├── EnzymeAbstractDomain          (CypEnzyme → EnzymeActivityInterval)
//! └── ClinicalAbstractDomain        (SetAbstractDomain<String>)
//! ```

use std::collections::BTreeSet;
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ============================================================================
// Local type definitions (avoid guardpharma-types dependency)
// ============================================================================

/// Drug identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DrugId(pub String);

impl DrugId {
    pub fn new(s: impl Into<String>) -> Self {
        DrugId(s.into())
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

/// CYP enzyme enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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
}

impl CypEnzyme {
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
        ]
    }

    pub fn degradation_rate(&self) -> f64 {
        match self {
            CypEnzyme::CYP3A4 => 0.0193,
            CypEnzyme::CYP2D6 => 0.0139,
            CypEnzyme::CYP2C9 => 0.0116,
            CypEnzyme::CYP2C19 => 0.0154,
            CypEnzyme::CYP1A2 => 0.0231,
            CypEnzyme::CYP2B6 => 0.0193,
            CypEnzyme::CYP2C8 => 0.0154,
            CypEnzyme::CYP2E1 => 0.0277,
            CypEnzyme::CYP3A5 => 0.0193,
        }
    }
}

impl fmt::Display for CypEnzyme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Severity levels for drug interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Minor,
    Moderate,
    Major,
    Critical,
}

impl Severity {
    pub fn numeric_score(self) -> u32 {
        match self {
            Severity::Minor => 1,
            Severity::Moderate => 2,
            Severity::Major => 3,
            Severity::Critical => 4,
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Concentration interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConcentrationInterval {
    pub lo: f64,
    pub hi: f64,
}

impl ConcentrationInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        ConcentrationInterval { lo, hi }
    }

    pub fn bottom() -> Self {
        ConcentrationInterval {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    pub fn top() -> Self {
        ConcentrationInterval {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    pub fn point(v: f64) -> Self {
        ConcentrationInterval { lo: v, hi: v }
    }

    pub fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    pub fn is_top(&self) -> bool {
        self.lo == f64::NEG_INFINITY && self.hi == f64::INFINITY
    }

    pub fn contains(&self, v: f64) -> bool {
        !self.is_bottom() && self.lo <= v && v <= self.hi
    }

    pub fn contains_interval(&self, other: &Self) -> bool {
        if other.is_bottom() {
            return true;
        }
        if self.is_bottom() {
            return false;
        }
        self.lo <= other.lo && other.hi <= self.hi
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        other.contains_interval(self)
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        if self.is_bottom() || other.is_bottom() {
            return false;
        }
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn width(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    pub fn midpoint(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            (self.lo + self.hi) / 2.0
        }
    }

    pub fn join(&self, other: &Self) -> Self {
        if self.is_bottom() {
            return *other;
        }
        if other.is_bottom() {
            return *self;
        }
        Self::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    pub fn meet(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            Self::bottom()
        } else {
            Self::new(lo, hi)
        }
    }

    /// Threshold widening: snaps to nearest threshold or ±∞.
    pub fn widen(&self, other: &Self, thresholds: &[f64]) -> Self {
        if self.is_bottom() {
            return *other;
        }
        if other.is_bottom() {
            return *self;
        }
        let lo = if other.lo < self.lo {
            thresholds
                .iter()
                .rev()
                .find(|&&t| t <= other.lo)
                .copied()
                .unwrap_or(f64::NEG_INFINITY)
        } else {
            self.lo
        };
        let hi = if other.hi > self.hi {
            thresholds
                .iter()
                .find(|&&t| t >= other.hi)
                .copied()
                .unwrap_or(f64::INFINITY)
        } else {
            self.hi
        };
        Self::new(lo, hi)
    }

    pub fn narrow(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = if self.lo == f64::NEG_INFINITY {
            other.lo
        } else {
            self.lo
        };
        let hi = if self.hi == f64::INFINITY {
            other.hi
        } else {
            self.hi
        };
        if lo > hi {
            Self::bottom()
        } else {
            Self::new(lo, hi)
        }
    }

    /// Scale both bounds by a non-negative factor.
    pub fn scale(&self, factor: f64) -> Self {
        if self.is_bottom() || factor < 0.0 {
            return Self::bottom();
        }
        Self::new((self.lo * factor).max(0.0), self.hi * factor)
    }

    /// Add a constant offset.
    pub fn add_scalar(&self, offset: f64) -> Self {
        if self.is_bottom() {
            return Self::bottom();
        }
        Self::new(self.lo + offset, self.hi + offset)
    }

    /// Interval addition.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        Self::new(self.lo + other.lo, self.hi + other.hi)
    }

    /// Interval multiplication (handles sign combinations).
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Self::new(lo, hi)
    }
}

impl fmt::Display for ConcentrationInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() {
            write!(f, "⊥")
        } else if self.is_top() {
            write!(f, "⊤")
        } else {
            write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
        }
    }
}

/// Enzyme activity interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnzymeActivityInterval {
    pub lo: f64,
    pub hi: f64,
}

impl EnzymeActivityInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        let lo = lo.max(0.0);
        let hi = hi.max(lo);
        EnzymeActivityInterval { lo, hi }
    }

    pub fn bottom() -> Self {
        EnzymeActivityInterval {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    pub fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }
}

/// Inhibition type for enzyme interactions.
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
        write!(f, "{:?}", self)
    }
}

/// Therapeutic window for a drug.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TherapeuticWindow {
    pub min_concentration: f64,
    pub max_concentration: f64,
}

impl TherapeuticWindow {
    pub fn new(min: f64, max: f64) -> Self {
        TherapeuticWindow {
            min_concentration: min,
            max_concentration: max,
        }
    }

    pub fn contains(&self, concentration: f64) -> bool {
        concentration >= self.min_concentration && concentration <= self.max_concentration
    }

    pub fn width(&self) -> f64 {
        self.max_concentration - self.min_concentration
    }

    pub fn midpoint(&self) -> f64 {
        (self.min_concentration + self.max_concentration) / 2.0
    }

    pub fn to_interval(&self) -> ConcentrationInterval {
        ConcentrationInterval::new(self.min_concentration, self.max_concentration)
    }
}

/// Dosing schedule for a drug.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DosingSchedule {
    pub dose_mg: f64,
    pub interval_hours: f64,
    pub bioavailability: f64,
}

impl DosingSchedule {
    pub fn new(dose_mg: f64, interval_hours: f64) -> Self {
        DosingSchedule {
            dose_mg,
            interval_hours,
            bioavailability: 1.0,
        }
    }

    pub fn with_bioavailability(mut self, f: f64) -> Self {
        self.bioavailability = f.clamp(0.0, 1.0);
        self
    }

    pub fn effective_dose(&self) -> f64 {
        self.dose_mg * self.bioavailability
    }

    pub fn daily_dose(&self) -> f64 {
        if self.interval_hours <= 0.0 {
            return 0.0;
        }
        self.dose_mg * (24.0 / self.interval_hours)
    }
}

/// PK parameters for a single drug.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkParameters {
    pub clearance_l_per_h: f64,
    pub volume_of_distribution_l: f64,
    pub half_life_hours: f64,
    pub absorption_rate: f64,
}

impl PkParameters {
    pub fn new(clearance: f64, vd: f64) -> Self {
        let half_life = if clearance > 0.0 {
            (0.693 * vd) / clearance
        } else {
            f64::INFINITY
        };
        PkParameters {
            clearance_l_per_h: clearance,
            volume_of_distribution_l: vd,
            half_life_hours: half_life,
            absorption_rate: 1.0,
        }
    }

    pub fn elimination_rate(&self) -> f64 {
        if self.half_life_hours > 0.0 {
            0.693 / self.half_life_hours
        } else {
            0.0
        }
    }

    /// Steady-state concentration estimate: F·D / (CL·τ).
    pub fn steady_state_concentration(&self, schedule: &DosingSchedule) -> f64 {
        if self.clearance_l_per_h <= 0.0 || schedule.interval_hours <= 0.0 {
            return 0.0;
        }
        schedule.effective_dose() / (self.clearance_l_per_h * schedule.interval_hours)
    }

    /// Steady-state Cmax estimate.
    pub fn steady_state_cmax(&self, schedule: &DosingSchedule) -> f64 {
        if self.volume_of_distribution_l <= 0.0 {
            return 0.0;
        }
        let ke = self.elimination_rate();
        let tau = schedule.interval_hours;
        if ke <= 0.0 {
            return schedule.effective_dose() / self.volume_of_distribution_l;
        }
        let accumulation = 1.0 / (1.0 - (-ke * tau).exp());
        (schedule.effective_dose() / self.volume_of_distribution_l) * accumulation
    }
}

/// Drug metabolism route through a specific CYP enzyme.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetabolismRoute {
    pub enzyme: CypEnzyme,
    pub fraction_metabolized: f64,
}

impl MetabolismRoute {
    pub fn new(enzyme: CypEnzyme, fraction: f64) -> Self {
        MetabolismRoute {
            enzyme,
            fraction_metabolized: fraction.clamp(0.0, 1.0),
        }
    }
}

/// Drug-specific screening configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrugConfig {
    pub drug_id: DrugId,
    pub schedule: DosingSchedule,
    pub pk_params: PkParameters,
    pub therapeutic_window: Option<TherapeuticWindow>,
    pub metabolism_routes: Vec<MetabolismRoute>,
    pub inhibition_effects: Vec<InhibitionEffect>,
    pub induction_effects: Vec<InductionEffect>,
}

impl DrugConfig {
    pub fn new(drug_id: DrugId, schedule: DosingSchedule, pk: PkParameters) -> Self {
        DrugConfig {
            drug_id,
            schedule,
            pk_params: pk,
            therapeutic_window: None,
            metabolism_routes: Vec::new(),
            inhibition_effects: Vec::new(),
            induction_effects: Vec::new(),
        }
    }

    pub fn with_therapeutic_window(mut self, tw: TherapeuticWindow) -> Self {
        self.therapeutic_window = Some(tw);
        self
    }

    pub fn with_metabolism(mut self, route: MetabolismRoute) -> Self {
        self.metabolism_routes.push(route);
        self
    }

    pub fn with_inhibition(mut self, effect: InhibitionEffect) -> Self {
        self.inhibition_effects.push(effect);
        self
    }

    pub fn with_induction(mut self, effect: InductionEffect) -> Self {
        self.induction_effects.push(effect);
        self
    }

    pub fn baseline_css(&self) -> f64 {
        self.pk_params.steady_state_concentration(&self.schedule)
    }

    pub fn baseline_cmax(&self) -> f64 {
        self.pk_params.steady_state_cmax(&self.schedule)
    }
}

/// Inhibition effect of a perpetrator drug on an enzyme.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InhibitionEffect {
    pub enzyme: CypEnzyme,
    pub inhibition_type: InhibitionType,
    pub ki_micromolar: f64,
}

impl InhibitionEffect {
    pub fn new(enzyme: CypEnzyme, inh_type: InhibitionType, ki: f64) -> Self {
        InhibitionEffect {
            enzyme,
            inhibition_type: inh_type,
            ki_micromolar: ki.max(f64::EPSILON),
        }
    }

    /// Compute AUC ratio for competitive inhibition: 1 + [I]/Ki.
    pub fn auc_ratio(&self, inhibitor_conc: f64) -> f64 {
        match self.inhibition_type {
            InhibitionType::Competitive => 1.0 + inhibitor_conc / self.ki_micromolar,
            InhibitionType::NonCompetitive => 1.0 + inhibitor_conc / self.ki_micromolar,
            InhibitionType::Uncompetitive => 1.0 + inhibitor_conc / (self.ki_micromolar * 2.0),
            InhibitionType::MechanismBased => {
                1.0 + (0.05 * inhibitor_conc) / (self.ki_micromolar * 0.02)
            }
            InhibitionType::Mixed => 1.0 + inhibitor_conc / (self.ki_micromolar * 1.5),
        }
    }
}

/// Induction effect of a perpetrator drug on an enzyme.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InductionEffect {
    pub enzyme: CypEnzyme,
    pub emax: f64,
    pub ec50: f64,
}

impl InductionEffect {
    pub fn new(enzyme: CypEnzyme, emax: f64, ec50: f64) -> Self {
        InductionEffect {
            enzyme,
            emax: emax.max(0.0),
            ec50: ec50.max(f64::EPSILON),
        }
    }

    pub fn fold_induction(&self, conc: f64) -> f64 {
        1.0 + (self.emax * conc) / (self.ec50 + conc)
    }
}

// ============================================================================
// AbstractValue trait
// ============================================================================

/// Core lattice trait for abstract domains.
///
/// Every abstract value forms a complete lattice with join (least upper bound),
/// meet (greatest lower bound), and optional widening / narrowing operators for
/// accelerating or refining fixed-point iterations.
pub trait AbstractValue: Clone + fmt::Debug {
    /// Least upper bound (⊔). Sound over-approximation of the union.
    fn join(&self, other: &Self) -> Self;

    /// Greatest lower bound (⊓). Sound under-approximation of the intersection.
    fn meet(&self, other: &Self) -> Self;

    /// Widening operator (∇). Guarantees termination of ascending chains.
    fn widen(&self, other: &Self) -> Self;

    /// Narrowing operator (Δ). Refines after widening without losing soundness.
    fn narrow(&self, other: &Self) -> Self;

    /// Returns `true` when this element is the bottom (⊥) of the lattice.
    fn is_bottom(&self) -> bool;

    /// Returns `true` when this element is the top (⊤) of the lattice.
    fn is_top(&self) -> bool;

    /// Returns `true` when `other` is contained in (less-than-or-equal to)
    /// `self` in the lattice ordering.
    fn contains(&self, other: &Self) -> bool;

    /// Partial ordering: `self ⊑ other`.
    ///
    /// Default implementation delegates to `other.contains(self)`.
    fn partial_le(&self, other: &Self) -> bool {
        other.contains(self)
    }
}

// ============================================================================
// IntervalDomain<T>
// ============================================================================

/// Generic interval domain `[lo, hi]` over any partially-ordered, cloneable type.
///
/// For `f64` the bottom element is represented by `lo > hi` (NaN-free convention
/// matching [`ConcentrationInterval`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntervalDomain<T: PartialOrd + Clone> {
    pub lo: T,
    pub hi: T,
}

impl<T: PartialOrd + Clone + fmt::Debug> IntervalDomain<T> {
    pub fn new(lo: T, hi: T) -> Self {
        IntervalDomain { lo, hi }
    }
}

impl IntervalDomain<f64> {
    /// The empty interval (⊥).
    pub fn bottom() -> Self {
        IntervalDomain {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    /// The full real line (⊤).
    pub fn top() -> Self {
        IntervalDomain {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// Degenerate interval `[v, v]`.
    pub fn point(v: f64) -> Self {
        IntervalDomain { lo: v, hi: v }
    }

    /// Returns `true` when the scalar `v` is inside `[lo, hi]`.
    pub fn contains_value(&self, v: f64) -> bool {
        !self.is_bottom() && self.lo <= v && v <= self.hi
    }

    /// Width of the interval. Returns 0.0 for bottom.
    pub fn width(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    /// Midpoint, or 0.0 for bottom.
    pub fn midpoint(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            (self.lo + self.hi) / 2.0
        }
    }
}

impl AbstractValue for IntervalDomain<f64> {
    fn join(&self, other: &Self) -> Self {
        if self.is_bottom() {
            return other.clone();
        }
        if other.is_bottom() {
            return self.clone();
        }
        IntervalDomain {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            Self::bottom()
        } else {
            IntervalDomain { lo, hi }
        }
    }

    fn widen(&self, other: &Self) -> Self {
        if self.is_bottom() {
            return other.clone();
        }
        if other.is_bottom() {
            return self.clone();
        }
        let lo = if other.lo < self.lo {
            f64::NEG_INFINITY
        } else {
            self.lo
        };
        let hi = if other.hi > self.hi {
            f64::INFINITY
        } else {
            self.hi
        };
        IntervalDomain { lo, hi }
    }

    fn narrow(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = if self.lo == f64::NEG_INFINITY {
            other.lo
        } else {
            self.lo
        };
        let hi = if self.hi == f64::INFINITY {
            other.hi
        } else {
            self.hi
        };
        if lo > hi {
            Self::bottom()
        } else {
            IntervalDomain { lo, hi }
        }
    }

    fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    fn is_top(&self) -> bool {
        self.lo == f64::NEG_INFINITY && self.hi == f64::INFINITY
    }

    fn contains(&self, other: &Self) -> bool {
        if other.is_bottom() {
            return true;
        }
        if self.is_bottom() {
            return false;
        }
        self.lo <= other.lo && other.hi <= self.hi
    }
}

impl fmt::Display for IntervalDomain<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() {
            write!(f, "⊥")
        } else if self.is_top() {
            write!(f, "⊤")
        } else {
            write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
        }
    }
}

// ============================================================================
// EnzymeActivityAbstractInterval  (wraps EnzymeActivityInterval)
// ============================================================================

/// Abstract domain for enzyme activity levels.
///
/// Wraps [`EnzymeActivityInterval`] from the types crate and adds
/// [`AbstractValue`] conformance. Enzyme activities are non-negative and the
/// widening operator pushes bounds towards 0 (lo) and a configurable ceiling
/// (hi).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnzymeActivityAbstractInterval {
    pub lo: f64,
    pub hi: f64,
}

/// Default widening ceiling for enzyme activity.
const ENZYME_WIDEN_MAX: f64 = 10.0;

impl EnzymeActivityAbstractInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        let lo = lo.max(0.0);
        let hi = hi.max(lo);
        EnzymeActivityAbstractInterval { lo, hi }
    }

    pub fn bottom() -> Self {
        EnzymeActivityAbstractInterval {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    pub fn top(max: f64) -> Self {
        EnzymeActivityAbstractInterval {
            lo: 0.0,
            hi: max,
        }
    }

    pub fn normal() -> Self {
        EnzymeActivityAbstractInterval { lo: 1.0, hi: 1.0 }
    }

    pub fn point(v: f64) -> Self {
        let v = v.max(0.0);
        EnzymeActivityAbstractInterval { lo: v, hi: v }
    }

    pub fn contains_value(&self, v: f64) -> bool {
        !self.is_bottom() && self.lo <= v && v <= self.hi
    }

    pub fn width(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    pub fn midpoint(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            (self.lo + self.hi) / 2.0
        }
    }

    /// Scale both bounds by a non-negative factor.
    pub fn scale(&self, factor: f64) -> Self {
        if self.is_bottom() || factor < 0.0 {
            return Self::bottom();
        }
        Self::new(self.lo * factor, self.hi * factor)
    }

    /// Apply an interval inhibition factor: multiply by `[1 - inh_hi, 1 - inh_lo]`.
    pub fn apply_inhibition_interval(&self, inh_lo: f64, inh_hi: f64) -> Self {
        if self.is_bottom() {
            return Self::bottom();
        }
        let factor_lo = (1.0 - inh_hi).max(0.0);
        let factor_hi = (1.0 - inh_lo).max(0.0);
        Self::new(self.lo * factor_lo, self.hi * factor_hi)
    }

    /// Apply an interval induction fold: multiply by `[ind_lo, ind_hi]`.
    pub fn apply_induction_interval(&self, ind_lo: f64, ind_hi: f64) -> Self {
        if self.is_bottom() {
            return Self::bottom();
        }
        Self::new(self.lo * ind_lo.max(0.0), self.hi * ind_hi.max(0.0))
    }

    /// Convert to the types-crate representation.
    pub fn to_enzyme_activity_interval(&self) -> EnzymeActivityInterval {
        if self.is_bottom() {
            EnzymeActivityInterval::bottom()
        } else {
            EnzymeActivityInterval::new(self.lo, self.hi)
        }
    }

    /// Construct from the types-crate representation.
    pub fn from_enzyme_activity_interval(eai: &EnzymeActivityInterval) -> Self {
        if eai.is_bottom() {
            Self::bottom()
        } else {
            Self::new(eai.lo, eai.hi)
        }
    }
}

impl AbstractValue for EnzymeActivityAbstractInterval {
    fn join(&self, other: &Self) -> Self {
        if self.is_bottom() {
            return other.clone();
        }
        if other.is_bottom() {
            return self.clone();
        }
        EnzymeActivityAbstractInterval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            Self::bottom()
        } else {
            EnzymeActivityAbstractInterval { lo, hi }
        }
    }

    fn widen(&self, other: &Self) -> Self {
        if self.is_bottom() {
            return other.clone();
        }
        if other.is_bottom() {
            return self.clone();
        }
        let lo = if other.lo < self.lo { 0.0 } else { self.lo };
        let hi = if other.hi > self.hi {
            ENZYME_WIDEN_MAX
        } else {
            self.hi
        };
        EnzymeActivityAbstractInterval { lo, hi }
    }

    fn narrow(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = if self.lo == 0.0 && other.lo > 0.0 {
            other.lo
        } else {
            self.lo
        };
        let hi = if (self.hi - ENZYME_WIDEN_MAX).abs() < f64::EPSILON && other.hi < ENZYME_WIDEN_MAX
        {
            other.hi
        } else {
            self.hi
        };
        if lo > hi {
            Self::bottom()
        } else {
            EnzymeActivityAbstractInterval { lo, hi }
        }
    }

    fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    fn is_top(&self) -> bool {
        self.lo == 0.0 && self.hi >= ENZYME_WIDEN_MAX
    }

    fn contains(&self, other: &Self) -> bool {
        if other.is_bottom() {
            return true;
        }
        if self.is_bottom() {
            return false;
        }
        self.lo <= other.lo && other.hi <= self.hi
    }
}

impl Default for EnzymeActivityAbstractInterval {
    fn default() -> Self {
        Self::normal()
    }
}

impl fmt::Display for EnzymeActivityAbstractInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() {
            write!(f, "⊥")
        } else {
            write!(f, "[{:.2}×, {:.2}×]", self.lo, self.hi)
        }
    }
}

// ============================================================================
// BoolAbstractValue
// ============================================================================

/// Three-valued boolean abstract domain with an explicit bottom.
///
/// Lattice structure:
/// ```text
///       Top
///      /   \
///   True   False
///      \   /
///      Bottom
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoolAbstractValue {
    Bottom,
    True,
    False,
    Top,
}

impl BoolAbstractValue {
    /// Construct from a concrete boolean.
    pub fn from_bool(b: bool) -> Self {
        if b {
            BoolAbstractValue::True
        } else {
            BoolAbstractValue::False
        }
    }

    /// Returns `Some(true/false)` when the value is definite, `None` otherwise.
    pub fn to_definite(&self) -> Option<bool> {
        match self {
            BoolAbstractValue::True => Some(true),
            BoolAbstractValue::False => Some(false),
            _ => None,
        }
    }

    /// Abstract negation.
    pub fn negate(&self) -> Self {
        match self {
            BoolAbstractValue::True => BoolAbstractValue::False,
            BoolAbstractValue::False => BoolAbstractValue::True,
            BoolAbstractValue::Top => BoolAbstractValue::Top,
            BoolAbstractValue::Bottom => BoolAbstractValue::Bottom,
        }
    }

    /// Abstract logical AND.
    pub fn abstract_and(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolAbstractValue::Bottom, _) | (_, BoolAbstractValue::Bottom) => {
                BoolAbstractValue::Bottom
            }
            (BoolAbstractValue::False, _) | (_, BoolAbstractValue::False) => {
                BoolAbstractValue::False
            }
            (BoolAbstractValue::True, BoolAbstractValue::True) => BoolAbstractValue::True,
            _ => BoolAbstractValue::Top,
        }
    }

    /// Abstract logical OR.
    pub fn abstract_or(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolAbstractValue::Bottom, _) | (_, BoolAbstractValue::Bottom) => {
                BoolAbstractValue::Bottom
            }
            (BoolAbstractValue::True, _) | (_, BoolAbstractValue::True) => {
                BoolAbstractValue::True
            }
            (BoolAbstractValue::False, BoolAbstractValue::False) => BoolAbstractValue::False,
            _ => BoolAbstractValue::Top,
        }
    }
}

impl AbstractValue for BoolAbstractValue {
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolAbstractValue::Bottom, x) | (x, BoolAbstractValue::Bottom) => *x,
            (BoolAbstractValue::Top, _) | (_, BoolAbstractValue::Top) => BoolAbstractValue::Top,
            (a, b) if a == b => *a,
            _ => BoolAbstractValue::Top,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolAbstractValue::Top, x) | (x, BoolAbstractValue::Top) => *x,
            (BoolAbstractValue::Bottom, _) | (_, BoolAbstractValue::Bottom) => {
                BoolAbstractValue::Bottom
            }
            (a, b) if a == b => *a,
            _ => BoolAbstractValue::Bottom,
        }
    }

    fn widen(&self, other: &Self) -> Self {
        // Finite lattice: join suffices.
        self.join(other)
    }

    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }

    fn is_bottom(&self) -> bool {
        *self == BoolAbstractValue::Bottom
    }

    fn is_top(&self) -> bool {
        *self == BoolAbstractValue::Top
    }

    fn contains(&self, other: &Self) -> bool {
        match (self, other) {
            (_, BoolAbstractValue::Bottom) => true,
            (BoolAbstractValue::Top, _) => true,
            (a, b) => a == b,
        }
    }
}

impl Default for BoolAbstractValue {
    fn default() -> Self {
        BoolAbstractValue::Bottom
    }
}

impl fmt::Display for BoolAbstractValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolAbstractValue::Bottom => write!(f, "⊥"),
            BoolAbstractValue::True => write!(f, "true"),
            BoolAbstractValue::False => write!(f, "false"),
            BoolAbstractValue::Top => write!(f, "⊤"),
        }
    }
}

// ============================================================================
// SetAbstractDomain<T>
// ============================================================================

/// Finite powerset abstract domain with a top element.
///
/// Below a configurable cardinality threshold the domain tracks individual
/// elements; above that threshold the set collapses to ⊤ (universe).
///
/// Bottom is the empty set with `is_top == false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetAbstractDomain<T: Ord + Clone> {
    pub elements: BTreeSet<T>,
    pub is_top: bool,
}

/// Maximum set cardinality before collapsing to ⊤.
const SET_TOP_THRESHOLD: usize = 256;

impl<T: Ord + Clone> SetAbstractDomain<T> {
    /// Construct an empty (bottom) set.
    pub fn empty() -> Self {
        SetAbstractDomain {
            elements: BTreeSet::new(),
            is_top: false,
        }
    }

    /// Construct the top (universal) set.
    pub fn top() -> Self {
        SetAbstractDomain {
            elements: BTreeSet::new(),
            is_top: true,
        }
    }

    /// Construct a singleton set.
    pub fn singleton(v: T) -> Self {
        let mut elements = BTreeSet::new();
        elements.insert(v);
        SetAbstractDomain {
            elements,
            is_top: false,
        }
    }

    /// Construct from an iterator.
    pub fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        let elements: BTreeSet<T> = iter.into_iter().collect();
        if elements.len() > SET_TOP_THRESHOLD {
            Self::top()
        } else {
            SetAbstractDomain {
                elements,
                is_top: false,
            }
        }
    }

    /// Insert an element. May collapse to ⊤.
    pub fn insert(&mut self, v: T) {
        if self.is_top {
            return;
        }
        self.elements.insert(v);
        if self.elements.len() > SET_TOP_THRESHOLD {
            self.is_top = true;
            self.elements.clear();
        }
    }

    /// Remove an element.
    pub fn remove(&mut self, v: &T) {
        if !self.is_top {
            self.elements.remove(v);
        }
    }

    /// Check membership.
    pub fn contains_element(&self, v: &T) -> bool {
        self.is_top || self.elements.contains(v)
    }

    /// Number of tracked elements (`None` when top).
    pub fn len(&self) -> Option<usize> {
        if self.is_top {
            None
        } else {
            Some(self.elements.len())
        }
    }

    /// True when bottom (empty and not top).
    pub fn is_empty(&self) -> bool {
        !self.is_top && self.elements.is_empty()
    }

    /// Iterate over tracked elements. Empty iterator when top.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }
}

impl<T: Ord + Clone + PartialEq> PartialEq for SetAbstractDomain<T> {
    fn eq(&self, other: &Self) -> bool {
        self.is_top == other.is_top && self.elements == other.elements
    }
}

impl<T> AbstractValue for SetAbstractDomain<T>
where
    T: Ord + Clone + fmt::Debug + Serialize + for<'de> Deserialize<'de>,
{
    fn join(&self, other: &Self) -> Self {
        if self.is_top || other.is_top {
            return Self::top();
        }
        let union: BTreeSet<T> = self.elements.union(&other.elements).cloned().collect();
        if union.len() > SET_TOP_THRESHOLD {
            Self::top()
        } else {
            SetAbstractDomain {
                elements: union,
                is_top: false,
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        if self.is_top {
            return other.clone();
        }
        if other.is_top {
            return self.clone();
        }
        let intersection: BTreeSet<T> =
            self.elements.intersection(&other.elements).cloned().collect();
        SetAbstractDomain {
            elements: intersection,
            is_top: false,
        }
    }

    fn widen(&self, other: &Self) -> Self {
        // For finite sets with a threshold, join is already a valid widening.
        self.join(other)
    }

    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }

    fn is_bottom(&self) -> bool {
        !self.is_top && self.elements.is_empty()
    }

    fn is_top(&self) -> bool {
        self.is_top
    }

    fn contains(&self, other: &Self) -> bool {
        if self.is_top {
            return true;
        }
        if other.is_top {
            return false;
        }
        other.elements.is_subset(&self.elements)
    }
}

impl<T: Ord + Clone + fmt::Display> fmt::Display for SetAbstractDomain<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_top {
            write!(f, "⊤")
        } else if self.elements.is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "{{")?;
            for (i, elem) in self.elements.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", elem)?;
            }
            write!(f, "}}")
        }
    }
}

// ============================================================================
// ConcentrationAbstractDomain
// ============================================================================

/// Map-based abstract domain tracking a [`ConcentrationInterval`] per drug.
///
/// The lattice is the point-wise lifting of `ConcentrationInterval` over an
/// `IndexMap` keyed by [`DrugId`]. A drug absent from the map is implicitly ⊥
/// (empty interval).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationAbstractDomain {
    pub intervals: IndexMap<DrugId, ConcentrationInterval>,
}

impl ConcentrationAbstractDomain {
    /// Construct the bottom domain (empty map).
    pub fn bottom() -> Self {
        ConcentrationAbstractDomain {
            intervals: IndexMap::new(),
        }
    }

    /// Construct from an existing map.
    pub fn new(intervals: IndexMap<DrugId, ConcentrationInterval>) -> Self {
        ConcentrationAbstractDomain { intervals }
    }

    /// Get the interval for a drug, defaulting to bottom.
    pub fn get(&self, drug: &DrugId) -> ConcentrationInterval {
        self.intervals
            .get(drug)
            .copied()
            .unwrap_or_else(ConcentrationInterval::bottom)
    }

    /// Set the interval for a drug.
    pub fn set(&mut self, drug: DrugId, interval: ConcentrationInterval) {
        if interval.is_bottom() {
            self.intervals.shift_remove(&drug);
        } else {
            self.intervals.insert(drug, interval);
        }
    }

    /// Number of tracked drugs (with non-bottom intervals).
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Returns `true` when no drugs are tracked.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Iterate over `(DrugId, ConcentrationInterval)` entries.
    pub fn iter(&self) -> impl Iterator<Item = (&DrugId, &ConcentrationInterval)> {
        self.intervals.iter()
    }

    /// All tracked drug identifiers.
    pub fn drugs(&self) -> Vec<DrugId> {
        self.intervals.keys().cloned().collect()
    }

    /// Returns the keys present in either domain.
    fn all_keys(a: &Self, b: &Self) -> Vec<DrugId> {
        let mut keys: Vec<DrugId> = a.intervals.keys().cloned().collect();
        for k in b.intervals.keys() {
            if !a.intervals.contains_key(k) {
                keys.push(k.clone());
            }
        }
        keys
    }
}

impl PartialEq for ConcentrationAbstractDomain {
    fn eq(&self, other: &Self) -> bool {
        if self.intervals.len() != other.intervals.len() {
            return false;
        }
        for (k, v) in &self.intervals {
            match other.intervals.get(k) {
                Some(ov) if ov == v => {}
                _ => return false,
            }
        }
        true
    }
}

impl AbstractValue for ConcentrationAbstractDomain {
    fn join(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let joined = a.join(&b);
            if !joined.is_bottom() {
                result.insert(k, joined);
            }
        }
        ConcentrationAbstractDomain { intervals: result }
    }

    fn meet(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let met = a.meet(&b);
            if !met.is_bottom() {
                result.insert(k, met);
            }
        }
        ConcentrationAbstractDomain { intervals: result }
    }

    fn widen(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        let thresholds: Vec<f64> = vec![
            0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0,
        ];
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let widened = if a.is_bottom() {
                b
            } else if b.is_bottom() {
                a
            } else {
                a.widen(&b, &thresholds)
            };
            if !widened.is_bottom() {
                result.insert(k, widened);
            }
        }
        ConcentrationAbstractDomain { intervals: result }
    }

    fn narrow(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let narrowed = a.narrow(&b);
            if !narrowed.is_bottom() {
                result.insert(k, narrowed);
            }
        }
        ConcentrationAbstractDomain { intervals: result }
    }

    fn is_bottom(&self) -> bool {
        self.intervals.is_empty()
    }

    fn is_top(&self) -> bool {
        // A map domain is only top if every tracked entry is top—
        // but since we can't represent every possible DrugId we define
        // "top" as having at least one top-valued entry.
        self.intervals.values().any(|v| v.is_top())
    }

    fn contains(&self, other: &Self) -> bool {
        for (k, v) in &other.intervals {
            if v.is_bottom() {
                continue;
            }
            match self.intervals.get(k) {
                Some(sv) => {
                    if !sv.contains_interval(v) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }
}

impl fmt::Display for ConcentrationAbstractDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.intervals.is_empty() {
            return write!(f, "⊥");
        }
        write!(f, "{{")?;
        for (i, (drug, iv)) in self.intervals.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", drug, iv)?;
        }
        write!(f, "}}")
    }
}

// ============================================================================
// EnzymeAbstractDomain
// ============================================================================

/// Map-based abstract domain tracking an [`EnzymeActivityAbstractInterval`]
/// per CYP enzyme.
///
/// Enzymes absent from the map are implicitly at normal activity `[1.0, 1.0]`
/// (uninhibited / uninduced), *not* bottom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeAbstractDomain {
    pub activities: IndexMap<CypEnzyme, EnzymeActivityAbstractInterval>,
}

impl EnzymeAbstractDomain {
    /// All enzymes at normal activity.
    pub fn normal() -> Self {
        EnzymeAbstractDomain {
            activities: IndexMap::new(),
        }
    }

    /// Explicit bottom (no information).
    pub fn bottom() -> Self {
        let mut activities = IndexMap::new();
        for &e in CypEnzyme::all() {
            activities.insert(e, EnzymeActivityAbstractInterval::bottom());
        }
        EnzymeAbstractDomain { activities }
    }

    /// Construct from an existing map.
    pub fn new(activities: IndexMap<CypEnzyme, EnzymeActivityAbstractInterval>) -> Self {
        EnzymeAbstractDomain { activities }
    }

    /// Get the activity interval for an enzyme. Absent → normal `[1, 1]`.
    pub fn get(&self, enzyme: &CypEnzyme) -> EnzymeActivityAbstractInterval {
        self.activities
            .get(enzyme)
            .cloned()
            .unwrap_or_else(EnzymeActivityAbstractInterval::normal)
    }

    /// Set the activity interval for an enzyme.
    pub fn set(&mut self, enzyme: CypEnzyme, interval: EnzymeActivityAbstractInterval) {
        // Store even normal values when explicitly set, to track precision.
        self.activities.insert(enzyme, interval);
    }

    /// Number of enzymes with explicitly tracked (non-default) activity.
    pub fn len(&self) -> usize {
        self.activities.len()
    }

    /// Returns true when no enzymes are explicitly tracked.
    pub fn is_empty(&self) -> bool {
        self.activities.is_empty()
    }

    /// Iterate over `(CypEnzyme, EnzymeActivityAbstractInterval)` entries.
    pub fn iter(&self) -> impl Iterator<Item = (&CypEnzyme, &EnzymeActivityAbstractInterval)> {
        self.activities.iter()
    }

    /// All enzyme keys present in either domain.
    fn all_keys(a: &Self, b: &Self) -> Vec<CypEnzyme> {
        let mut keys: Vec<CypEnzyme> = a.activities.keys().copied().collect();
        for &k in b.activities.keys() {
            if !a.activities.contains_key(&k) {
                keys.push(k);
            }
        }
        keys
    }

    /// Check whether any enzyme is significantly inhibited (activity < threshold).
    pub fn has_significant_inhibition(&self, threshold: f64) -> bool {
        self.activities.values().any(|v| !v.is_bottom() && v.lo < threshold)
    }

    /// Check whether any enzyme is significantly induced (activity > threshold).
    pub fn has_significant_induction(&self, threshold: f64) -> bool {
        self.activities.values().any(|v| !v.is_bottom() && v.hi > threshold)
    }
}

impl PartialEq for EnzymeAbstractDomain {
    fn eq(&self, other: &Self) -> bool {
        let keys = Self::all_keys(self, other);
        for k in keys {
            if self.get(&k) != other.get(&k) {
                return false;
            }
        }
        true
    }
}

impl AbstractValue for EnzymeAbstractDomain {
    fn join(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let joined = a.join(&b);
            result.insert(k, joined);
        }
        EnzymeAbstractDomain {
            activities: result,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let met = a.meet(&b);
            result.insert(k, met);
        }
        EnzymeAbstractDomain {
            activities: result,
        }
    }

    fn widen(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let widened = a.widen(&b);
            result.insert(k, widened);
        }
        EnzymeAbstractDomain {
            activities: result,
        }
    }

    fn narrow(&self, other: &Self) -> Self {
        let keys = Self::all_keys(self, other);
        let mut result = IndexMap::new();
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            let narrowed = a.narrow(&b);
            result.insert(k, narrowed);
        }
        EnzymeAbstractDomain {
            activities: result,
        }
    }

    fn is_bottom(&self) -> bool {
        self.activities.values().any(|v| v.is_bottom())
    }

    fn is_top(&self) -> bool {
        // Top when every enzyme tracked is at top, or none are tracked
        // (implicitly all normal, which is not top).
        !self.activities.is_empty() && self.activities.values().all(|v| v.is_top())
    }

    fn contains(&self, other: &Self) -> bool {
        let keys = Self::all_keys(self, other);
        for k in keys {
            let a = self.get(&k);
            let b = other.get(&k);
            if !a.contains(&b) {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for EnzymeAbstractDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.activities.is_empty() {
            return write!(f, "{{normal}}");
        }
        write!(f, "{{")?;
        for (i, (enzyme, iv)) in self.activities.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", enzyme, iv)?;
        }
        write!(f, "}}")
    }
}

// ============================================================================
// ClinicalAbstractDomain
// ============================================================================

/// Abstract domain for clinical flags / conditions.
///
/// Wraps a [`SetAbstractDomain<String>`] representing the set of active
/// clinical conditions, warnings, or guideline labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClinicalAbstractDomain {
    pub conditions: SetAbstractDomain<String>,
}

impl ClinicalAbstractDomain {
    /// Empty set of clinical conditions (⊥).
    pub fn empty() -> Self {
        ClinicalAbstractDomain {
            conditions: SetAbstractDomain::empty(),
        }
    }

    /// Universal set (⊤) — every condition is possible.
    pub fn top() -> Self {
        ClinicalAbstractDomain {
            conditions: SetAbstractDomain::top(),
        }
    }

    /// Construct from a single condition.
    pub fn singleton(condition: impl Into<String>) -> Self {
        ClinicalAbstractDomain {
            conditions: SetAbstractDomain::singleton(condition.into()),
        }
    }

    /// Construct from a list of conditions.
    pub fn from_conditions(conds: impl IntoIterator<Item = String>) -> Self {
        ClinicalAbstractDomain {
            conditions: SetAbstractDomain::from_iter(conds),
        }
    }

    /// Add a clinical condition.
    pub fn add_condition(&mut self, condition: impl Into<String>) {
        self.conditions.insert(condition.into());
    }

    /// Check whether a condition is definitely present.
    pub fn has_condition(&self, condition: &str) -> bool {
        self.conditions.contains_element(&condition.to_string())
    }

    /// Number of tracked conditions (`None` when top).
    pub fn len(&self) -> Option<usize> {
        self.conditions.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Iterate over known conditions.
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.conditions.iter()
    }
}

impl AbstractValue for ClinicalAbstractDomain {
    fn join(&self, other: &Self) -> Self {
        ClinicalAbstractDomain {
            conditions: self.conditions.join(&other.conditions),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        ClinicalAbstractDomain {
            conditions: self.conditions.meet(&other.conditions),
        }
    }

    fn widen(&self, other: &Self) -> Self {
        ClinicalAbstractDomain {
            conditions: self.conditions.widen(&other.conditions),
        }
    }

    fn narrow(&self, other: &Self) -> Self {
        ClinicalAbstractDomain {
            conditions: self.conditions.narrow(&other.conditions),
        }
    }

    fn is_bottom(&self) -> bool {
        self.conditions.is_bottom()
    }

    fn is_top(&self) -> bool {
        self.conditions.is_top()
    }

    fn contains(&self, other: &Self) -> bool {
        self.conditions.contains(&other.conditions)
    }
}

impl Default for ClinicalAbstractDomain {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for ClinicalAbstractDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Clinical({})", self.conditions)
    }
}

// ============================================================================
// ProductDomain
// ============================================================================

/// Product of concentration, enzyme, and clinical abstract domains.
///
/// This is the main abstract state used during fixed-point iteration on the
/// pharmacological timed automaton. All lattice operations are applied
/// component-wise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductDomain {
    pub concentrations: ConcentrationAbstractDomain,
    pub enzymes: EnzymeAbstractDomain,
    pub clinical: ClinicalAbstractDomain,
}

impl ProductDomain {
    /// Construct the bottom product (all components bottom).
    pub fn bottom() -> Self {
        ProductDomain {
            concentrations: ConcentrationAbstractDomain::bottom(),
            enzymes: EnzymeAbstractDomain::bottom(),
            clinical: ClinicalAbstractDomain::empty(),
        }
    }

    /// Construct a product with all-normal enzyme activities and no drugs.
    pub fn initial() -> Self {
        ProductDomain {
            concentrations: ConcentrationAbstractDomain::bottom(),
            enzymes: EnzymeAbstractDomain::normal(),
            clinical: ClinicalAbstractDomain::empty(),
        }
    }

    /// Convenience: get a drug's concentration interval.
    pub fn concentration(&self, drug: &DrugId) -> ConcentrationInterval {
        self.concentrations.get(drug)
    }

    /// Convenience: get an enzyme's activity interval.
    pub fn enzyme_activity(&self, enzyme: &CypEnzyme) -> EnzymeActivityAbstractInterval {
        self.enzymes.get(enzyme)
    }

    /// Convenience: set a drug's concentration interval.
    pub fn set_concentration(&mut self, drug: DrugId, interval: ConcentrationInterval) {
        self.concentrations.set(drug, interval);
    }

    /// Convenience: set an enzyme's activity interval.
    pub fn set_enzyme_activity(
        &mut self,
        enzyme: CypEnzyme,
        interval: EnzymeActivityAbstractInterval,
    ) {
        self.enzymes.set(enzyme, interval);
    }

    /// Convenience: add a clinical condition.
    pub fn add_condition(&mut self, condition: impl Into<String>) {
        self.clinical.add_condition(condition);
    }

    /// Check structural equality (used for convergence testing in fixpoint).
    pub fn structurally_equal(&self, other: &Self) -> bool {
        self.concentrations == other.concentrations
            && self.enzymes == other.enzymes
            && self.clinical.conditions == other.clinical.conditions
    }
}

impl PartialEq for ProductDomain {
    fn eq(&self, other: &Self) -> bool {
        self.structurally_equal(other)
    }
}

impl AbstractValue for ProductDomain {
    fn join(&self, other: &Self) -> Self {
        ProductDomain {
            concentrations: self.concentrations.join(&other.concentrations),
            enzymes: self.enzymes.join(&other.enzymes),
            clinical: self.clinical.join(&other.clinical),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        ProductDomain {
            concentrations: self.concentrations.meet(&other.concentrations),
            enzymes: self.enzymes.meet(&other.enzymes),
            clinical: self.clinical.meet(&other.clinical),
        }
    }

    fn widen(&self, other: &Self) -> Self {
        ProductDomain {
            concentrations: self.concentrations.widen(&other.concentrations),
            enzymes: self.enzymes.widen(&other.enzymes),
            clinical: self.clinical.widen(&other.clinical),
        }
    }

    fn narrow(&self, other: &Self) -> Self {
        ProductDomain {
            concentrations: self.concentrations.narrow(&other.concentrations),
            enzymes: self.enzymes.narrow(&other.enzymes),
            clinical: self.clinical.narrow(&other.clinical),
        }
    }

    fn is_bottom(&self) -> bool {
        self.concentrations.is_bottom()
            && self.enzymes.is_bottom()
            && self.clinical.is_bottom()
    }

    fn is_top(&self) -> bool {
        self.concentrations.is_top()
            && self.enzymes.is_top()
            && self.clinical.is_top()
    }

    fn contains(&self, other: &Self) -> bool {
        self.concentrations.contains(&other.concentrations)
            && self.enzymes.contains(&other.enzymes)
            && self.clinical.contains(&other.clinical)
    }
}

impl Default for ProductDomain {
    fn default() -> Self {
        Self::initial()
    }
}

impl fmt::Display for ProductDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ProductDomain {{")?;
        writeln!(f, "  concentrations: {}", self.concentrations)?;
        writeln!(f, "  enzymes:        {}", self.enzymes)?;
        writeln!(f, "  clinical:       {}", self.clinical)?;
        write!(f, "}}")
    }
}

// ============================================================================
// Standalone product-domain helpers
// ============================================================================

/// Component-wise join of two product domains.
pub fn join_product(a: &ProductDomain, b: &ProductDomain) -> ProductDomain {
    a.join(b)
}

/// Component-wise meet of two product domains.
pub fn meet_product(a: &ProductDomain, b: &ProductDomain) -> ProductDomain {
    a.meet(b)
}

/// Component-wise widening of two product domains.
pub fn widen_product(a: &ProductDomain, b: &ProductDomain) -> ProductDomain {
    a.widen(b)
}

/// Component-wise narrowing of two product domains.
pub fn narrow_product(a: &ProductDomain, b: &ProductDomain) -> ProductDomain {
    a.narrow(b)
}

// ============================================================================
// Conversion helpers
// ============================================================================

impl From<ConcentrationInterval> for IntervalDomain<f64> {
    fn from(ci: ConcentrationInterval) -> Self {
        IntervalDomain {
            lo: ci.lo,
            hi: ci.hi,
        }
    }
}

impl From<IntervalDomain<f64>> for ConcentrationInterval {
    fn from(id: IntervalDomain<f64>) -> Self {
        if id.is_bottom() {
            ConcentrationInterval::bottom()
        } else {
            ConcentrationInterval::new(id.lo, id.hi)
        }
    }
}

impl From<EnzymeActivityInterval> for EnzymeActivityAbstractInterval {
    fn from(eai: EnzymeActivityInterval) -> Self {
        EnzymeActivityAbstractInterval::from_enzyme_activity_interval(&eai)
    }
}

impl From<EnzymeActivityAbstractInterval> for EnzymeActivityInterval {
    fn from(eaai: EnzymeActivityAbstractInterval) -> Self {
        eaai.to_enzyme_activity_interval()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // IntervalDomain<f64>
    // ------------------------------------------------------------------

    #[test]
    fn test_interval_domain_bottom_and_top() {
        let bot = IntervalDomain::<f64>::bottom();
        let top = IntervalDomain::<f64>::top();
        assert!(bot.is_bottom());
        assert!(!bot.is_top());
        assert!(top.is_top());
        assert!(!top.is_bottom());
    }

    #[test]
    fn test_interval_domain_join() {
        let a = IntervalDomain::new(1.0, 3.0);
        let b = IntervalDomain::new(2.0, 5.0);
        let j = a.join(&b);
        assert_eq!(j.lo, 1.0);
        assert_eq!(j.hi, 5.0);

        let bot = IntervalDomain::<f64>::bottom();
        assert_eq!(a.join(&bot), a);
        assert_eq!(bot.join(&a), a);
    }

    #[test]
    fn test_interval_domain_meet() {
        let a = IntervalDomain::new(1.0, 5.0);
        let b = IntervalDomain::new(3.0, 7.0);
        let m = a.meet(&b);
        assert_eq!(m.lo, 3.0);
        assert_eq!(m.hi, 5.0);

        let c = IntervalDomain::new(6.0, 8.0);
        assert!(a.meet(&c).is_bottom());
    }

    #[test]
    fn test_interval_domain_widen() {
        let a = IntervalDomain::new(2.0, 4.0);
        let b = IntervalDomain::new(1.0, 6.0);
        let w = a.widen(&b);
        assert_eq!(w.lo, f64::NEG_INFINITY);
        assert_eq!(w.hi, f64::INFINITY);

        let c = IntervalDomain::new(2.0, 3.0);
        let w2 = a.widen(&c);
        assert_eq!(w2.lo, 2.0);
        assert_eq!(w2.hi, 4.0);
    }

    #[test]
    fn test_interval_domain_narrow() {
        let a = IntervalDomain::new(f64::NEG_INFINITY, f64::INFINITY);
        let b = IntervalDomain::new(1.0, 5.0);
        let n = a.narrow(&b);
        assert_eq!(n.lo, 1.0);
        assert_eq!(n.hi, 5.0);
    }

    #[test]
    fn test_interval_domain_contains() {
        let a = IntervalDomain::new(1.0, 10.0);
        let b = IntervalDomain::new(2.0, 5.0);
        assert!(a.contains(&b));
        assert!(!b.contains(&a));

        let bot = IntervalDomain::<f64>::bottom();
        assert!(a.contains(&bot));
        assert!(!bot.contains(&a));
    }

    #[test]
    fn test_interval_domain_point_and_width() {
        let p = IntervalDomain::point(3.14);
        assert_eq!(p.width(), 0.0);
        assert_eq!(p.midpoint(), 3.14);
        assert!(p.contains_value(3.14));
        assert!(!p.contains_value(4.0));
    }

    // ------------------------------------------------------------------
    // EnzymeActivityAbstractInterval
    // ------------------------------------------------------------------

    #[test]
    fn test_enzyme_interval_basic() {
        let n = EnzymeActivityAbstractInterval::normal();
        assert!(!n.is_bottom());
        assert_eq!(n.lo, 1.0);
        assert_eq!(n.hi, 1.0);
        assert!(n.contains_value(1.0));
        assert!(!n.contains_value(0.5));
    }

    #[test]
    fn test_enzyme_interval_join_meet() {
        let a = EnzymeActivityAbstractInterval::new(0.5, 1.0);
        let b = EnzymeActivityAbstractInterval::new(0.8, 1.5);
        let j = a.join(&b);
        assert_eq!(j.lo, 0.5);
        assert_eq!(j.hi, 1.5);

        let m = a.meet(&b);
        assert_eq!(m.lo, 0.8);
        assert_eq!(m.hi, 1.0);
    }

    #[test]
    fn test_enzyme_interval_widen() {
        let a = EnzymeActivityAbstractInterval::new(0.5, 1.0);
        let b = EnzymeActivityAbstractInterval::new(0.3, 1.2);
        let w = a.widen(&b);
        assert_eq!(w.lo, 0.0);
        assert_eq!(w.hi, ENZYME_WIDEN_MAX);
    }

    #[test]
    fn test_enzyme_inhibition_interval() {
        let a = EnzymeActivityAbstractInterval::new(0.8, 1.2);
        let inh = a.apply_inhibition_interval(0.1, 0.3);
        // factor_lo = 1 - 0.3 = 0.7, factor_hi = 1 - 0.1 = 0.9
        assert!((inh.lo - 0.8 * 0.7).abs() < 1e-10);
        assert!((inh.hi - 1.2 * 0.9).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // BoolAbstractValue
    // ------------------------------------------------------------------

    #[test]
    fn test_bool_lattice() {
        use BoolAbstractValue::*;

        assert_eq!(True.join(&False), Top);
        assert_eq!(True.join(&True), True);
        assert_eq!(Bottom.join(&True), True);
        assert_eq!(True.meet(&False), Bottom);
        assert_eq!(Top.meet(&True), True);

        assert!(Top.contains(&True));
        assert!(Top.contains(&Bottom));
        assert!(!False.contains(&True));

        assert!(Bottom.is_bottom());
        assert!(Top.is_top());
        assert!(!True.is_bottom());
        assert!(!False.is_top());
    }

    #[test]
    fn test_bool_abstract_ops() {
        use BoolAbstractValue::*;

        assert_eq!(True.negate(), False);
        assert_eq!(Top.negate(), Top);
        assert_eq!(True.abstract_and(&True), True);
        assert_eq!(True.abstract_and(&False), False);
        assert_eq!(True.abstract_and(&Top), Top);
        assert_eq!(False.abstract_or(&True), True);
        assert_eq!(False.abstract_or(&False), False);
    }

    // ------------------------------------------------------------------
    // SetAbstractDomain
    // ------------------------------------------------------------------

    #[test]
    fn test_set_domain_basic() {
        let mut s = SetAbstractDomain::<String>::empty();
        assert!(s.is_bottom());
        s.insert("a".to_string());
        assert!(!s.is_bottom());
        assert!(s.contains_element(&"a".to_string()));
        assert!(!s.contains_element(&"b".to_string()));

        let t = SetAbstractDomain::<String>::top();
        assert!(t.is_top());
        assert!(t.contains_element(&"anything".to_string()));
    }

    #[test]
    fn test_set_domain_join_meet() {
        let a = SetAbstractDomain::from_iter(vec!["x".to_string(), "y".to_string()]);
        let b = SetAbstractDomain::from_iter(vec!["y".to_string(), "z".to_string()]);
        let j = a.join(&b);
        assert!(j.contains_element(&"x".to_string()));
        assert!(j.contains_element(&"y".to_string()));
        assert!(j.contains_element(&"z".to_string()));
        assert_eq!(j.len(), Some(3));

        let m = a.meet(&b);
        assert!(!m.contains_element(&"x".to_string()));
        assert!(m.contains_element(&"y".to_string()));
        assert!(!m.contains_element(&"z".to_string()));
    }

    #[test]
    fn test_set_domain_contains() {
        let a = SetAbstractDomain::from_iter(vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ]);
        let b = SetAbstractDomain::from_iter(vec!["x".to_string(), "y".to_string()]);
        assert!(a.contains(&b));
        assert!(!b.contains(&a));

        let top = SetAbstractDomain::<String>::top();
        assert!(top.contains(&a));
        assert!(!a.contains(&top));
    }

    // ------------------------------------------------------------------
    // ConcentrationAbstractDomain
    // ------------------------------------------------------------------

    #[test]
    fn test_concentration_domain() {
        let mut d1 = ConcentrationAbstractDomain::bottom();
        assert!(d1.is_bottom());
        d1.set(
            DrugId::new("warfarin"),
            ConcentrationInterval::new(1.0, 3.0),
        );
        assert!(!d1.is_bottom());
        assert_eq!(d1.len(), 1);

        let ci = d1.get(&DrugId::new("warfarin"));
        assert_eq!(ci.lo, 1.0);
        assert_eq!(ci.hi, 3.0);

        let missing = d1.get(&DrugId::new("aspirin"));
        assert!(missing.is_bottom());

        let mut d2 = ConcentrationAbstractDomain::bottom();
        d2.set(
            DrugId::new("warfarin"),
            ConcentrationInterval::new(2.0, 5.0),
        );
        d2.set(
            DrugId::new("aspirin"),
            ConcentrationInterval::new(0.5, 1.5),
        );

        let joined = d1.join(&d2);
        let w = joined.get(&DrugId::new("warfarin"));
        assert_eq!(w.lo, 1.0);
        assert_eq!(w.hi, 5.0);
        assert!(!joined.get(&DrugId::new("aspirin")).is_bottom());
    }

    #[test]
    fn test_concentration_domain_containment() {
        let mut big = ConcentrationAbstractDomain::bottom();
        big.set(
            DrugId::new("d1"),
            ConcentrationInterval::new(0.0, 10.0),
        );
        let mut small = ConcentrationAbstractDomain::bottom();
        small.set(
            DrugId::new("d1"),
            ConcentrationInterval::new(2.0, 5.0),
        );
        assert!(big.contains(&small));
        assert!(!small.contains(&big));

        let empty = ConcentrationAbstractDomain::bottom();
        assert!(big.contains(&empty));
    }

    // ------------------------------------------------------------------
    // EnzymeAbstractDomain
    // ------------------------------------------------------------------

    #[test]
    fn test_enzyme_domain() {
        let mut ed = EnzymeAbstractDomain::normal();
        assert!(ed.is_empty());
        let default_activity = ed.get(&CypEnzyme::CYP3A4);
        assert_eq!(default_activity, EnzymeActivityAbstractInterval::normal());

        ed.set(
            CypEnzyme::CYP3A4,
            EnzymeActivityAbstractInterval::new(0.3, 0.8),
        );
        assert!(ed.has_significant_inhibition(0.5));
        let a = ed.get(&CypEnzyme::CYP3A4);
        assert_eq!(a.lo, 0.3);
        assert_eq!(a.hi, 0.8);
    }

    // ------------------------------------------------------------------
    // ClinicalAbstractDomain
    // ------------------------------------------------------------------

    #[test]
    fn test_clinical_domain() {
        let mut c = ClinicalAbstractDomain::empty();
        assert!(c.is_bottom());
        c.add_condition("hepatic_impairment");
        assert!(c.has_condition("hepatic_impairment"));
        assert!(!c.has_condition("renal_impairment"));

        let c2 = ClinicalAbstractDomain::singleton("renal_impairment");
        let j = c.join(&c2);
        assert!(j.has_condition("hepatic_impairment"));
        assert!(j.has_condition("renal_impairment"));
    }

    // ------------------------------------------------------------------
    // ProductDomain
    // ------------------------------------------------------------------

    #[test]
    fn test_product_domain_basic() {
        let bot = ProductDomain::bottom();
        assert!(bot.is_bottom());

        let init = ProductDomain::initial();
        assert!(!init.is_bottom());

        let mut p = ProductDomain::initial();
        p.set_concentration(
            DrugId::new("drug_a"),
            ConcentrationInterval::new(1.0, 5.0),
        );
        p.set_enzyme_activity(
            CypEnzyme::CYP2D6,
            EnzymeActivityAbstractInterval::new(0.5, 0.9),
        );
        p.add_condition("elderly_patient");

        assert_eq!(p.concentration(&DrugId::new("drug_a")).lo, 1.0);
        assert_eq!(p.enzyme_activity(&CypEnzyme::CYP2D6).lo, 0.5);
        assert!(p.clinical.has_condition("elderly_patient"));
    }

    #[test]
    fn test_product_domain_join() {
        let mut a = ProductDomain::initial();
        a.set_concentration(
            DrugId::new("d1"),
            ConcentrationInterval::new(1.0, 3.0),
        );
        a.add_condition("cond_a");

        let mut b = ProductDomain::initial();
        b.set_concentration(
            DrugId::new("d1"),
            ConcentrationInterval::new(2.0, 5.0),
        );
        b.add_condition("cond_b");

        let j = join_product(&a, &b);
        let ci = j.concentration(&DrugId::new("d1"));
        assert_eq!(ci.lo, 1.0);
        assert_eq!(ci.hi, 5.0);
        assert!(j.clinical.has_condition("cond_a"));
        assert!(j.clinical.has_condition("cond_b"));
    }

    #[test]
    fn test_product_domain_containment() {
        let mut big = ProductDomain::initial();
        big.set_concentration(
            DrugId::new("d"),
            ConcentrationInterval::new(0.0, 10.0),
        );
        big.add_condition("x");
        big.add_condition("y");

        let mut small = ProductDomain::initial();
        small.set_concentration(
            DrugId::new("d"),
            ConcentrationInterval::new(2.0, 5.0),
        );
        small.add_condition("x");

        assert!(big.contains(&small));
        assert!(!small.contains(&big));
    }

    #[test]
    fn test_standalone_product_functions() {
        let a = ProductDomain::initial();
        let b = ProductDomain::initial();
        let j = join_product(&a, &b);
        let m = meet_product(&a, &b);
        let w = widen_product(&a, &b);
        let n = narrow_product(&a, &b);
        assert!(!j.is_bottom());
        assert!(!m.is_bottom());
        assert!(!w.is_bottom());
        assert!(!n.is_bottom());
    }

    // ------------------------------------------------------------------
    // Conversions
    // ------------------------------------------------------------------

    #[test]
    fn test_conversion_round_trip() {
        let ci = ConcentrationInterval::new(1.0, 5.0);
        let id: IntervalDomain<f64> = ci.into();
        let ci2: ConcentrationInterval = id.into();
        assert_eq!(ci.lo, ci2.lo);
        assert_eq!(ci.hi, ci2.hi);

        let eai = EnzymeActivityInterval::new(0.5, 1.5);
        let eaai: EnzymeActivityAbstractInterval = eai.into();
        let eai2: EnzymeActivityInterval = eaai.into();
        assert_eq!(eai.lo, eai2.lo);
        assert_eq!(eai.hi, eai2.hi);
    }

    // ------------------------------------------------------------------
    // Partial ordering (default impl)
    // ------------------------------------------------------------------

    #[test]
    fn test_partial_le() {
        let a = IntervalDomain::new(2.0, 5.0);
        let b = IntervalDomain::new(1.0, 10.0);
        assert!(a.partial_le(&b));
        assert!(!b.partial_le(&a));

        let bot = IntervalDomain::<f64>::bottom();
        assert!(bot.partial_le(&a));
    }

    // ------------------------------------------------------------------
    // Display implementations
    // ------------------------------------------------------------------

    #[test]
    fn test_display_impls() {
        let iv = IntervalDomain::new(1.0, 2.0);
        assert!(iv.to_string().contains("1.0000"));
        assert!(iv.to_string().contains("2.0000"));

        let bot = IntervalDomain::<f64>::bottom();
        assert_eq!(bot.to_string(), "⊥");

        let top = IntervalDomain::<f64>::top();
        assert_eq!(top.to_string(), "⊤");

        let b = BoolAbstractValue::True;
        assert_eq!(b.to_string(), "true");

        let s = SetAbstractDomain::from_iter(vec!["a".to_string()]);
        assert!(s.to_string().contains("a"));

        let p = ProductDomain::initial();
        let display = p.to_string();
        assert!(display.contains("ProductDomain"));
    }

    // ------------------------------------------------------------------
    // Serialization round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_serde_round_trip() {
        let mut p = ProductDomain::initial();
        p.set_concentration(
            DrugId::new("warfarin"),
            ConcentrationInterval::new(2.0, 4.0),
        );
        p.set_enzyme_activity(
            CypEnzyme::CYP2C9,
            EnzymeActivityAbstractInterval::new(0.4, 0.8),
        );
        p.add_condition("elderly");

        let json = serde_json::to_string(&p).expect("serialize");
        let p2: ProductDomain = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p2.concentration(&DrugId::new("warfarin")).lo, 2.0);
        assert_eq!(p2.enzyme_activity(&CypEnzyme::CYP2C9).lo, 0.4);
        assert!(p2.clinical.has_condition("elderly"));
    }
}
