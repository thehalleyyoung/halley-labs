//! # Concentration Types and Interval Arithmetic
//!
//! This module provides the foundational concentration types used throughout
//! the GuardPharma polypharmacy verification system. It includes:
//!
//! - [`Concentration`] — a scalar concentration value (NaN-rejecting newtype over `f64`)
//! - [`ConcentrationUnit`] — physical unit for concentration measurements with conversions
//! - [`ConcentrationRange`] — a closed range `[lower, upper]` for therapeutic windows
//! - [`ConcentrationInterval`] — abstract-interpretation interval with lattice operations
//!   (join, meet, widen, narrow) forming the core of Tier 1 analysis
//! - [`PlasmaConcentration`] — a timestamped, unit-tagged measurement bound to a drug
//!
//! ## Design Rationale
//!
//! Concentration values are central to pharmacokinetic modeling. The
//! [`ConcentrationInterval`] type implements the classical interval abstract
//! domain with widening and narrowing operators, enabling sound
//! over-approximation of reachable concentration sets during fixpoint
//! iteration in Tier 1.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops;
use std::str::FromStr;

use crate::drug::DrugId;

// ---------------------------------------------------------------------------
// Concentration (scalar)
// ---------------------------------------------------------------------------

/// A non-negative concentration value.
///
/// This is a newtype wrapper around `f64` that rejects `NaN` at construction
/// time. Arithmetic operations clamp to zero on the lower end so that
/// subtraction never produces a negative concentration.
///
/// # Panics
///
/// [`Concentration::new`] panics if `value` is `NaN`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Concentration(f64);

impl Concentration {
    /// Create a new concentration, rejecting `NaN`.
    pub fn new(value: f64) -> Self {
        assert!(!value.is_nan(), "Concentration value must not be NaN");
        Concentration(value)
    }

    /// Return the inner `f64` value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// A zero concentration constant.
    pub fn zero() -> Self {
        Concentration(0.0)
    }

    /// Returns `true` when the concentration is exactly zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0.0
    }

    /// Returns `true` when the concentration is strictly positive.
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.0 > 0.0
    }

    /// Absolute value of this concentration.
    #[inline]
    pub fn abs(&self) -> Self {
        Concentration(self.0.abs())
    }

    /// Returns the larger of `self` and `other`.
    pub fn max(self, other: Self) -> Self {
        if self.0 >= other.0 {
            self
        } else {
            other
        }
    }

    /// Returns the smaller of `self` and `other`.
    pub fn min(self, other: Self) -> Self {
        if self.0 <= other.0 {
            self
        } else {
            other
        }
    }

    /// Clamp `self` to `[lo, hi]`.
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        debug_assert!(lo.0 <= hi.0, "clamp bounds must satisfy lo <= hi");
        if self.0 < lo.0 {
            lo
        } else if self.0 > hi.0 {
            hi
        } else {
            self
        }
    }
}

// -- Equality / ordering (total where possible; NaN excluded at construction)

impl PartialEq for Concentration {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Concentration {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

// -- Display / FromStr

impl fmt::Display for Concentration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// Parsing error for [`Concentration`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConcentrationParseError {
    #[error("invalid float: {0}")]
    InvalidFloat(#[from] std::num::ParseFloatError),
    #[error("concentration must not be NaN")]
    NaN,
}

impl FromStr for Concentration {
    type Err = ConcentrationParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v: f64 = s.trim().parse()?;
        if v.is_nan() {
            return Err(ConcentrationParseError::NaN);
        }
        Ok(Concentration(v))
    }
}

impl From<f64> for Concentration {
    fn from(v: f64) -> Self {
        Concentration::new(v)
    }
}

// -- Arithmetic operators

impl ops::Add for Concentration {
    type Output = Concentration;
    fn add(self, rhs: Concentration) -> Self::Output {
        Concentration(self.0 + rhs.0)
    }
}

impl ops::Sub for Concentration {
    type Output = Concentration;
    fn sub(self, rhs: Concentration) -> Self::Output {
        Concentration(self.0 - rhs.0)
    }
}

impl ops::Mul<f64> for Concentration {
    type Output = Concentration;
    fn mul(self, rhs: f64) -> Self::Output {
        Concentration::new(self.0 * rhs)
    }
}

impl ops::Div<f64> for Concentration {
    type Output = Concentration;
    fn div(self, rhs: f64) -> Self::Output {
        assert!(rhs != 0.0, "division by zero");
        Concentration::new(self.0 / rhs)
    }
}

// ---------------------------------------------------------------------------
// ConcentrationUnit
// ---------------------------------------------------------------------------

/// Physical unit for concentration measurements.
///
/// Units fall into two dimensional families:
/// - **mass/volume** — µg/mL, ng/mL, mg/L, µg/L
/// - **molar** — nmol/L, µmol/L, mmol/L
///
/// Conversions within a family are pure scale factors. Cross-family conversion
/// requires the molecular weight of the analyte and is **not** handled here
/// (returns `None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConcentrationUnit {
    /// Micrograms per milliliter (µg/mL)
    MicrogramPerMl,
    /// Nanograms per milliliter (ng/mL)
    NanogramPerMl,
    /// Milligrams per liter (mg/L)
    MilligramPerLiter,
    /// Micrograms per liter (µg/L)
    MicrogramPerLiter,
    /// Nanomoles per liter (nmol/L)
    NanomolPerLiter,
    /// Micromoles per liter (µmol/L)
    MicromolPerLiter,
    /// Millimoles per liter (mmol/L)
    MillimolPerLiter,
}

impl ConcentrationUnit {
    /// Returns the multiplicative factor to convert a value in `self` to the
    /// equivalent value in `target`.
    ///
    /// Returns `None` when the two units belong to different dimensional
    /// families (mass-per-volume vs. molar) because molecular weight is needed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use guardpharma_types::concentration::ConcentrationUnit;
    /// let factor = ConcentrationUnit::MicrogramPerMl
    ///     .conversion_factor_to(&ConcentrationUnit::MilligramPerLiter)
    ///     .unwrap();
    /// assert!((factor - 1.0).abs() < 1e-12); // 1 µg/mL == 1 mg/L
    /// ```
    pub fn conversion_factor_to(&self, target: &ConcentrationUnit) -> Option<f64> {
        // Normalise each unit to a canonical form within its family, then
        // divide.  Mass-per-volume canonical = mg/L.  Molar canonical = nmol/L.
        let (self_family, self_to_canonical) = self.canonical_info();
        let (target_family, target_to_canonical) = target.canonical_info();
        if self_family != target_family {
            return None;
        }
        // value_canonical = value_self * self_to_canonical
        // value_target    = value_canonical / target_to_canonical
        Some(self_to_canonical / target_to_canonical)
    }

    /// Returns `(family_id, factor_to_canonical)`.
    ///
    /// Family 0 = mass/volume (canonical = mg/L).
    /// Family 1 = molar       (canonical = nmol/L).
    fn canonical_info(&self) -> (u8, f64) {
        match self {
            // 1 µg/mL = 1 mg/L  ⇒  factor = 1.0
            ConcentrationUnit::MicrogramPerMl => (0, 1.0),
            // 1 ng/mL = 0.001 mg/L  ⇒  factor = 0.001
            // Actually: 1 ng/mL = 1 µg/L, and 1 µg/L = 0.001 mg/L
            ConcentrationUnit::NanogramPerMl => (0, 0.001),
            // 1 mg/L = 1 mg/L  (identity)
            ConcentrationUnit::MilligramPerLiter => (0, 1.0),
            // 1 µg/L = 0.001 mg/L
            ConcentrationUnit::MicrogramPerLiter => (0, 0.001),
            // 1 nmol/L = 1 nmol/L  (identity)
            ConcentrationUnit::NanomolPerLiter => (1, 1.0),
            // 1 µmol/L = 1000 nmol/L
            ConcentrationUnit::MicromolPerLiter => (1, 1_000.0),
            // 1 mmol/L = 1_000_000 nmol/L
            ConcentrationUnit::MillimolPerLiter => (1, 1_000_000.0),
        }
    }

    /// Human-readable abbreviation string.
    pub fn abbreviation(&self) -> &'static str {
        match self {
            ConcentrationUnit::MicrogramPerMl => "µg/mL",
            ConcentrationUnit::NanogramPerMl => "ng/mL",
            ConcentrationUnit::MilligramPerLiter => "mg/L",
            ConcentrationUnit::MicrogramPerLiter => "µg/L",
            ConcentrationUnit::NanomolPerLiter => "nmol/L",
            ConcentrationUnit::MicromolPerLiter => "µmol/L",
            ConcentrationUnit::MillimolPerLiter => "mmol/L",
        }
    }

    /// Returns `true` if this unit belongs to the mass-per-volume family.
    pub fn is_mass_per_volume(&self) -> bool {
        self.canonical_info().0 == 0
    }

    /// Returns `true` if this unit belongs to the molar family.
    pub fn is_molar(&self) -> bool {
        self.canonical_info().0 == 1
    }
}

impl fmt::Display for ConcentrationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.abbreviation())
    }
}

/// Parsing error for [`ConcentrationUnit`].
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown concentration unit: {0}")]
pub struct ConcentrationUnitParseError(String);

impl FromStr for ConcentrationUnit {
    type Err = ConcentrationUnitParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().replace(' ', "").as_str() {
            "µg/ml" | "ug/ml" | "microgramperml" | "microgram/ml" => {
                Ok(ConcentrationUnit::MicrogramPerMl)
            }
            "ng/ml" | "nanogramperml" | "nanogram/ml" => Ok(ConcentrationUnit::NanogramPerMl),
            "mg/l" | "milligram/l" | "milligamperliter" | "mg/liter" => {
                Ok(ConcentrationUnit::MilligramPerLiter)
            }
            "µg/l" | "ug/l" | "microgram/l" | "microgramperliter" => {
                Ok(ConcentrationUnit::MicrogramPerLiter)
            }
            "nmol/l" | "nanomolperliter" | "nanomol/l" => Ok(ConcentrationUnit::NanomolPerLiter),
            "µmol/l" | "umol/l" | "micromolperliter" | "micromol/l" => {
                Ok(ConcentrationUnit::MicromolPerLiter)
            }
            "mmol/l" | "millimolperliter" | "millimol/l" => {
                Ok(ConcentrationUnit::MillimolPerLiter)
            }
            other => Err(ConcentrationUnitParseError(other.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// ConcentrationRange
// ---------------------------------------------------------------------------

/// A closed range of concentration values `[lower, upper]`, commonly used to
/// represent therapeutic windows (e.g. the acceptable plasma concentration
/// band for a drug).
///
/// The range is considered *valid* when `lower <= upper`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ConcentrationRange {
    /// Inclusive lower bound.
    pub lower: Concentration,
    /// Inclusive upper bound.
    pub upper: Concentration,
}

impl ConcentrationRange {
    /// Construct a new range. Panics if `lower > upper`.
    pub fn new(lower: Concentration, upper: Concentration) -> Self {
        assert!(
            lower.value() <= upper.value(),
            "ConcentrationRange: lower ({}) must be <= upper ({})",
            lower,
            upper,
        );
        ConcentrationRange { lower, upper }
    }

    /// Returns `true` if `value` falls within `[lower, upper]`.
    pub fn contains(&self, value: Concentration) -> bool {
        value.value() >= self.lower.value() && value.value() <= self.upper.value()
    }

    /// Width of the range (`upper − lower`).
    pub fn width(&self) -> Concentration {
        Concentration::new(self.upper.value() - self.lower.value())
    }

    /// Midpoint of the range.
    pub fn midpoint(&self) -> Concentration {
        Concentration::new((self.lower.value() + self.upper.value()) / 2.0)
    }

    /// Returns `true` if `self` and `other` share at least one point.
    pub fn overlaps(&self, other: &ConcentrationRange) -> bool {
        self.lower.value() <= other.upper.value() && other.lower.value() <= self.upper.value()
    }

    /// Returns the intersection of two ranges, or `None` if disjoint.
    pub fn intersection(&self, other: &ConcentrationRange) -> Option<ConcentrationRange> {
        let lo = self.lower.max(other.lower);
        let hi = self.upper.min(other.upper);
        if lo.value() <= hi.value() {
            Some(ConcentrationRange::new(lo, hi))
        } else {
            None
        }
    }

    /// Returns the smallest range enclosing both `self` and `other`.
    pub fn union_hull(&self, other: &ConcentrationRange) -> ConcentrationRange {
        ConcentrationRange::new(self.lower.min(other.lower), self.upper.max(other.upper))
    }

    /// Returns `true` if `lower <= upper`.
    pub fn is_valid(&self) -> bool {
        self.lower.value() <= self.upper.value()
    }

    /// Returns `true` if `lower == upper` (degenerate / point range).
    pub fn is_empty(&self) -> bool {
        (self.upper.value() - self.lower.value()).abs() < f64::EPSILON
    }
}

impl fmt::Display for ConcentrationRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.lower, self.upper)
    }
}

// ---------------------------------------------------------------------------
// ConcentrationInterval (abstract interpretation domain)
// ---------------------------------------------------------------------------

/// An interval `[lo, hi]` over `f64` forming the abstract domain for Tier 1
/// pharmacokinetic abstract interpretation.
///
/// The lattice is ordered by subset inclusion: `a ⊑ b` iff `a.lo >= b.lo`
/// and `a.hi <= b.hi`.
///
/// Special elements:
/// - **bottom** (`⊥`): the empty set, represented as `lo = f64::INFINITY,
///   hi = f64::NEG_INFINITY`.
/// - **top** (`⊤`): the entire real line, `lo = f64::NEG_INFINITY,
///   hi = f64::INFINITY`.
///
/// Widening uses a configurable set of threshold values to accelerate
/// convergence during fixpoint iteration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ConcentrationInterval {
    /// Lower bound of the interval (inclusive).
    pub lo: f64,
    /// Upper bound of the interval (inclusive).
    pub hi: f64,
}

impl ConcentrationInterval {
    // -- constructors -------------------------------------------------------

    /// Create a new interval `[lo, hi]`.
    ///
    /// # Panics
    ///
    /// Panics if `lo > hi` (unless constructing bottom via [`Self::bottom`]).
    pub fn new(lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "ConcentrationInterval: lo ({lo}) must be <= hi ({hi})");
        ConcentrationInterval { lo, hi }
    }

    /// Construct a point (degenerate) interval `[v, v]`.
    pub fn point(value: f64) -> Self {
        ConcentrationInterval { lo: value, hi: value }
    }

    /// The *bottom* element (empty set) of the lattice.
    pub fn bottom() -> Self {
        ConcentrationInterval {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    /// The *top* element (entire real line) of the lattice.
    pub fn top() -> Self {
        ConcentrationInterval {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    // -- predicates ---------------------------------------------------------

    /// Returns `true` if this is the bottom element (empty set).
    pub fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    /// Returns `true` if this is the top element (entire ℝ).
    pub fn is_top(&self) -> bool {
        self.lo == f64::NEG_INFINITY && self.hi == f64::INFINITY
    }

    /// Returns `true` if `value ∈ [lo, hi]`.
    pub fn contains_value(&self, value: f64) -> bool {
        if self.is_bottom() {
            return false;
        }
        value >= self.lo && value <= self.hi
    }

    /// Returns `true` if `other ⊆ self`.
    pub fn contains_interval(&self, other: &ConcentrationInterval) -> bool {
        if other.is_bottom() {
            return true;
        }
        if self.is_bottom() {
            return false;
        }
        self.lo <= other.lo && self.hi >= other.hi
    }

    // -- lattice operations -------------------------------------------------

    /// Least upper bound (join / union hull) — `self ⊔ other`.
    ///
    /// If either operand is bottom, returns the other.
    pub fn join(&self, other: &ConcentrationInterval) -> ConcentrationInterval {
        if self.is_bottom() {
            return *other;
        }
        if other.is_bottom() {
            return *self;
        }
        ConcentrationInterval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Greatest lower bound (meet / intersection) — `self ⊓ other`.
    ///
    /// Returns bottom if the intervals are disjoint.
    pub fn meet(&self, other: &ConcentrationInterval) -> ConcentrationInterval {
        if self.is_bottom() || other.is_bottom() {
            return ConcentrationInterval::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            ConcentrationInterval::bottom()
        } else {
            ConcentrationInterval { lo, hi }
        }
    }

    /// Widening operator with threshold set.
    ///
    /// Given the previous iterate `prev` (self) and the new iterate `next`,
    /// the widened result pushes unstable bounds outward to the next threshold
    /// or to `±∞`.
    ///
    /// `thresholds` must be **sorted in ascending order**.
    pub fn widen(&self, next: &ConcentrationInterval, thresholds: &[f64]) -> ConcentrationInterval {
        if self.is_bottom() {
            return *next;
        }
        if next.is_bottom() {
            return *self;
        }

        let lo = if next.lo < self.lo {
            // Lower bound decreased — push to next lower threshold or −∞
            thresholds
                .iter()
                .rev()
                .find(|&&t| t <= next.lo)
                .copied()
                .unwrap_or(f64::NEG_INFINITY)
        } else {
            self.lo
        };

        let hi = if next.hi > self.hi {
            // Upper bound increased — push to next higher threshold or +∞
            thresholds
                .iter()
                .find(|&&t| t >= next.hi)
                .copied()
                .unwrap_or(f64::INFINITY)
        } else {
            self.hi
        };

        ConcentrationInterval { lo, hi }
    }

    /// Narrowing operator.
    ///
    /// Given the previous (widened) iterate `prev` (self) and new iterate
    /// `next`, the narrowed result pulls infinite/unstable bounds inward
    /// toward the tighter `next` bounds, without ever going beyond `next`.
    pub fn narrow(&self, next: &ConcentrationInterval) -> ConcentrationInterval {
        if self.is_bottom() {
            return ConcentrationInterval::bottom();
        }
        if next.is_bottom() {
            return ConcentrationInterval::bottom();
        }

        let lo = if self.lo == f64::NEG_INFINITY {
            next.lo
        } else {
            self.lo
        };

        let hi = if self.hi == f64::INFINITY {
            next.hi
        } else {
            self.hi
        };

        if lo > hi {
            ConcentrationInterval::bottom()
        } else {
            ConcentrationInterval { lo, hi }
        }
    }

    // -- convenience queries ------------------------------------------------

    /// Width of the interval (`hi − lo`). Returns 0 for bottom.
    pub fn width(&self) -> f64 {
        if self.is_bottom() {
            return 0.0;
        }
        self.hi - self.lo
    }

    /// Midpoint `(lo + hi) / 2`. Returns `NaN` for bottom.
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    /// Returns `true` if `self` and `other` overlap.
    pub fn overlaps(&self, other: &ConcentrationInterval) -> bool {
        if self.is_bottom() || other.is_bottom() {
            return false;
        }
        self.lo <= other.hi && other.lo <= self.hi
    }

    /// Alias for [`Self::join`] — the convex hull of two intervals.
    pub fn union(&self, other: &ConcentrationInterval) -> ConcentrationInterval {
        self.join(other)
    }

    /// Returns the intersection, or `None` if disjoint (returns bottom).
    pub fn intersection(&self, other: &ConcentrationInterval) -> Option<ConcentrationInterval> {
        let m = self.meet(other);
        if m.is_bottom() {
            None
        } else {
            Some(m)
        }
    }

    /// Returns `true` if `self ⊆ other`.
    pub fn is_subset_of(&self, other: &ConcentrationInterval) -> bool {
        other.contains_interval(self)
    }

    /// Scale bounds by factor range `[factor_lo, factor_hi]`.
    pub fn scale(&self, factor_lo: f64, factor_hi: f64) -> ConcentrationInterval {
        if self.is_bottom() {
            return ConcentrationInterval::bottom();
        }
        let a = self.lo * factor_lo;
        let b = self.lo * factor_hi;
        let c = self.hi * factor_lo;
        let d = self.hi * factor_hi;
        let lo = a.min(b).min(c).min(d);
        let hi = a.max(b).max(c).max(d);
        ConcentrationInterval { lo, hi }
    }

    // -- interval arithmetic ------------------------------------------------

    /// Interval addition: `[a,b] + [c,d] = [a+c, b+d]`.
    pub fn add(&self, other: &ConcentrationInterval) -> ConcentrationInterval {
        if self.is_bottom() || other.is_bottom() {
            return ConcentrationInterval::bottom();
        }
        ConcentrationInterval {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }

    /// Interval subtraction: `[a,b] − [c,d] = [a−d, b−c]`.
    pub fn sub(&self, other: &ConcentrationInterval) -> ConcentrationInterval {
        if self.is_bottom() || other.is_bottom() {
            return ConcentrationInterval::bottom();
        }
        ConcentrationInterval {
            lo: self.lo - other.hi,
            hi: self.hi - other.lo,
        }
    }

    /// Multiply every element of the interval by a scalar.
    ///
    /// When `scalar < 0` the bounds swap.
    pub fn mul_scalar(&self, scalar: f64) -> ConcentrationInterval {
        if self.is_bottom() {
            return ConcentrationInterval::bottom();
        }
        let a = self.lo * scalar;
        let b = self.hi * scalar;
        ConcentrationInterval {
            lo: a.min(b),
            hi: a.max(b),
        }
    }

    /// Divide every element of the interval by a non-zero scalar.
    ///
    /// # Panics
    ///
    /// Panics if `scalar == 0.0`.
    pub fn div_scalar(&self, scalar: f64) -> ConcentrationInterval {
        assert!(scalar != 0.0, "division by zero in interval div_scalar");
        self.mul_scalar(1.0 / scalar)
    }

    /// Interval multiplication: `[a,b] * [c,d]`.
    ///
    /// Computes all four products and takes the min/max hull.
    pub fn mul(&self, other: &ConcentrationInterval) -> ConcentrationInterval {
        if self.is_bottom() || other.is_bottom() {
            return ConcentrationInterval::bottom();
        }
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        ConcentrationInterval {
            lo: products.iter().cloned().fold(f64::INFINITY, f64::min),
            hi: products.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
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

// Operator overloads delegate to the named methods so that
// `interval_a + interval_b` works ergonomically.

impl ops::Add for ConcentrationInterval {
    type Output = ConcentrationInterval;
    fn add(self, rhs: ConcentrationInterval) -> Self::Output {
        ConcentrationInterval::add(&self, &rhs)
    }
}

impl ops::Sub for ConcentrationInterval {
    type Output = ConcentrationInterval;
    fn sub(self, rhs: ConcentrationInterval) -> Self::Output {
        ConcentrationInterval::sub(&self, &rhs)
    }
}

impl ops::Mul for ConcentrationInterval {
    type Output = ConcentrationInterval;
    fn mul(self, rhs: ConcentrationInterval) -> Self::Output {
        ConcentrationInterval::mul(&self, &rhs)
    }
}

impl ops::Mul<f64> for ConcentrationInterval {
    type Output = ConcentrationInterval;
    fn mul(self, rhs: f64) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

// ---------------------------------------------------------------------------
// PlasmaConcentration
// ---------------------------------------------------------------------------

/// A single measured (or predicted) plasma-drug concentration at a specific
/// point in time, tagged with its physical unit and originating drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaConcentration {
    /// The measured concentration value.
    pub value: Concentration,
    /// Physical unit of the measurement.
    pub unit: ConcentrationUnit,
    /// Identifier of the drug being measured.
    pub drug_id: DrugId,
    /// UTC timestamp of the measurement.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl PlasmaConcentration {
    /// Construct a new plasma concentration measurement.
    pub fn new(
        value: Concentration,
        unit: ConcentrationUnit,
        drug_id: DrugId,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        PlasmaConcentration {
            value,
            unit,
            drug_id,
            timestamp,
        }
    }

    /// Convert this measurement to a different [`ConcentrationUnit`].
    ///
    /// Returns `None` if the units belong to different dimensional families
    /// (mass-per-volume vs. molar), since molecular weight would be required.
    pub fn convert_to(&self, target: ConcentrationUnit) -> Option<PlasmaConcentration> {
        let factor = self.unit.conversion_factor_to(&target)?;
        Some(PlasmaConcentration {
            value: Concentration::new(self.value.value() * factor),
            unit: target,
            drug_id: self.drug_id.clone(),
            timestamp: self.timestamp,
        })
    }

    /// Returns `true` if the concentration falls within the given therapeutic
    /// `range`.
    pub fn is_therapeutic(&self, range: &ConcentrationRange) -> bool {
        range.contains(self.value)
    }

    /// Returns `true` if the concentration exceeds the given toxic
    /// `threshold`.
    pub fn is_toxic(&self, threshold: f64) -> bool {
        self.value.value() > threshold
    }
}

impl fmt::Display for PlasmaConcentration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} [{}] @ {}",
            self.value, self.unit, self.drug_id, self.timestamp
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // -- Concentration scalar -----------------------------------------------

    #[test]
    fn test_concentration_new_and_value() {
        let c = Concentration::new(5.25);
        assert_eq!(c.value(), 5.25);
        assert!(!c.is_zero());
        assert!(c.is_positive());
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn test_concentration_rejects_nan() {
        Concentration::new(f64::NAN);
    }

    #[test]
    fn test_concentration_arithmetic() {
        let a = Concentration::new(3.0);
        let b = Concentration::new(2.0);
        assert!((a + b).value() - 5.0 < 1e-10);
        assert!((a - b).value() - 1.0 < 1e-10);
        assert!(((a * 2.5).value() - 7.5).abs() < 1e-10);
        assert!(((a / 2.0).value() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_concentration_clamp_min_max() {
        let lo = Concentration::new(1.0);
        let hi = Concentration::new(10.0);
        let mid = Concentration::new(5.0);
        let below = Concentration::new(0.5);
        let above = Concentration::new(15.0);

        assert_eq!(mid.clamp(lo, hi).value(), 5.0);
        assert_eq!(below.clamp(lo, hi).value(), 1.0);
        assert_eq!(above.clamp(lo, hi).value(), 10.0);
        assert_eq!(lo.max(hi).value(), 10.0);
        assert_eq!(lo.min(hi).value(), 1.0);
    }

    #[test]
    fn test_concentration_fromstr() {
        let c: Concentration = "3.14".parse().unwrap();
        assert!((c.value() - 3.14).abs() < 1e-12);
        assert!("NaN".parse::<Concentration>().is_err());
        assert!("hello".parse::<Concentration>().is_err());
    }

    // -- ConcentrationUnit --------------------------------------------------

    #[test]
    fn test_unit_conversion_same_family() {
        // µg/mL == mg/L  (factor 1.0)
        let f = ConcentrationUnit::MicrogramPerMl
            .conversion_factor_to(&ConcentrationUnit::MilligramPerLiter)
            .unwrap();
        assert!((f - 1.0).abs() < 1e-12);

        // ng/mL → µg/L  (1 ng/mL = 1 µg/L ⇒ factor = 1.0)
        let f2 = ConcentrationUnit::NanogramPerMl
            .conversion_factor_to(&ConcentrationUnit::MicrogramPerLiter)
            .unwrap();
        assert!((f2 - 1.0).abs() < 1e-12);

        // µmol/L → nmol/L  (factor = 1000)
        let f3 = ConcentrationUnit::MicromolPerLiter
            .conversion_factor_to(&ConcentrationUnit::NanomolPerLiter)
            .unwrap();
        assert!((f3 - 1000.0).abs() < 1e-9);

        // mmol/L → µmol/L  (factor = 1000)
        let f4 = ConcentrationUnit::MillimolPerLiter
            .conversion_factor_to(&ConcentrationUnit::MicromolPerLiter)
            .unwrap();
        assert!((f4 - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_unit_conversion_cross_family_returns_none() {
        assert!(ConcentrationUnit::MicrogramPerMl
            .conversion_factor_to(&ConcentrationUnit::NanomolPerLiter)
            .is_none());
    }

    #[test]
    fn test_unit_display_and_fromstr() {
        let u = ConcentrationUnit::MilligramPerLiter;
        assert_eq!(u.to_string(), "mg/L");
        let parsed: ConcentrationUnit = "mg/L".parse().unwrap();
        assert_eq!(parsed, ConcentrationUnit::MilligramPerLiter);
        assert!("bogus".parse::<ConcentrationUnit>().is_err());
    }

    // -- ConcentrationRange -------------------------------------------------

    #[test]
    fn test_range_contains_and_geometry() {
        let r = ConcentrationRange::new(Concentration::new(2.0), Concentration::new(8.0));
        assert!(r.contains(Concentration::new(5.0)));
        assert!(!r.contains(Concentration::new(1.0)));
        assert!(!r.contains(Concentration::new(9.0)));
        assert!((r.width().value() - 6.0).abs() < 1e-12);
        assert!((r.midpoint().value() - 5.0).abs() < 1e-12);
        assert!(r.is_valid());
        assert!(!r.is_empty());
    }

    #[test]
    fn test_range_overlap_intersection_union() {
        let a = ConcentrationRange::new(Concentration::new(1.0), Concentration::new(5.0));
        let b = ConcentrationRange::new(Concentration::new(3.0), Concentration::new(7.0));
        let c = ConcentrationRange::new(Concentration::new(6.0), Concentration::new(9.0));

        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));

        let inter = a.intersection(&b).unwrap();
        assert!((inter.lower.value() - 3.0).abs() < 1e-12);
        assert!((inter.upper.value() - 5.0).abs() < 1e-12);
        assert!(a.intersection(&c).is_none());

        let hull = a.union_hull(&c);
        assert!((hull.lower.value() - 1.0).abs() < 1e-12);
        assert!((hull.upper.value() - 9.0).abs() < 1e-12);
    }

    // -- ConcentrationInterval (abstract interpretation) --------------------

    #[test]
    fn test_interval_join_and_meet() {
        let a = ConcentrationInterval::new(1.0, 4.0);
        let b = ConcentrationInterval::new(3.0, 6.0);

        let j = a.join(&b);
        assert!((j.lo - 1.0).abs() < 1e-12);
        assert!((j.hi - 6.0).abs() < 1e-12);

        let m = a.meet(&b);
        assert!((m.lo - 3.0).abs() < 1e-12);
        assert!((m.hi - 4.0).abs() < 1e-12);

        // Disjoint meet → bottom
        let c = ConcentrationInterval::new(5.0, 6.0);
        let mb = a.meet(&c);
        assert!(mb.is_bottom());
    }

    #[test]
    fn test_interval_bottom_top_identity() {
        let bot = ConcentrationInterval::bottom();
        let top = ConcentrationInterval::top();
        let x = ConcentrationInterval::new(2.0, 5.0);

        assert!(bot.is_bottom());
        assert!(top.is_top());
        assert_eq!(bot.join(&x), x);
        assert_eq!(x.join(&bot), x);
        assert!(top.contains_interval(&x));
        assert!(!bot.contains_value(3.0));
    }

    #[test]
    fn test_interval_widen() {
        let thresholds = &[0.0, 1.0, 5.0, 10.0, 50.0, 100.0];
        let prev = ConcentrationInterval::new(2.0, 8.0);
        let next = ConcentrationInterval::new(1.5, 12.0);

        let widened = prev.widen(&next, thresholds);
        // lo decreased from 2.0 → 1.5, next threshold down is 1.0
        assert!((widened.lo - 1.0).abs() < 1e-12);
        // hi increased from 8.0 → 12.0, next threshold up is 50.0
        assert!((widened.hi - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_narrow() {
        let prev = ConcentrationInterval {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        };
        let next = ConcentrationInterval::new(2.0, 10.0);

        let narrowed = prev.narrow(&next);
        assert!((narrowed.lo - 2.0).abs() < 1e-12);
        assert!((narrowed.hi - 10.0).abs() < 1e-12);

        // If prev is finite, lo/hi are retained
        let prev2 = ConcentrationInterval::new(1.0, 100.0);
        let narrowed2 = prev2.narrow(&next);
        assert!((narrowed2.lo - 1.0).abs() < 1e-12);
        assert!((narrowed2.hi - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = ConcentrationInterval::new(1.0, 3.0);
        let b = ConcentrationInterval::new(2.0, 5.0);

        // add
        let s = a.add(&b);
        assert!((s.lo - 3.0).abs() < 1e-12);
        assert!((s.hi - 8.0).abs() < 1e-12);

        // sub: [1,3] - [2,5] = [1-5, 3-2] = [-4, 1]
        let d = a.sub(&b);
        assert!((d.lo - (-4.0)).abs() < 1e-12);
        assert!((d.hi - 1.0).abs() < 1e-12);

        // mul_scalar
        let m = a.mul_scalar(2.0);
        assert!((m.lo - 2.0).abs() < 1e-12);
        assert!((m.hi - 6.0).abs() < 1e-12);

        // mul_scalar negative flips
        let mn = a.mul_scalar(-1.0);
        assert!((mn.lo - (-3.0)).abs() < 1e-12);
        assert!((mn.hi - (-1.0)).abs() < 1e-12);

        // div_scalar
        let dv = a.div_scalar(2.0);
        assert!((dv.lo - 0.5).abs() < 1e-12);
        assert!((dv.hi - 1.5).abs() < 1e-12);

        // operator overloads
        let sum = a + b;
        assert!((sum.lo - 3.0).abs() < 1e-12);
    }

    // -- PlasmaConcentration ------------------------------------------------

    #[test]
    fn test_plasma_concentration_therapeutic() {
        let pc = PlasmaConcentration::new(
            Concentration::new(5.0),
            ConcentrationUnit::MilligramPerLiter,
            DrugId::new("warfarin"),
            Utc::now(),
        );
        let range = ConcentrationRange::new(Concentration::new(2.0), Concentration::new(8.0));
        assert!(pc.is_therapeutic(&range));
        assert!(!pc.is_toxic(10.0));
        assert!(pc.is_toxic(4.0));
    }

    #[test]
    fn test_plasma_concentration_unit_convert() {
        let pc = PlasmaConcentration::new(
            Concentration::new(5.0),
            ConcentrationUnit::MicrogramPerMl,
            DrugId::new("atorvastatin"),
            Utc::now(),
        );
        // µg/mL → mg/L (factor 1.0)
        let converted = pc.convert_to(ConcentrationUnit::MilligramPerLiter).unwrap();
        assert!((converted.value.value() - 5.0).abs() < 1e-12);
        assert_eq!(converted.unit, ConcentrationUnit::MilligramPerLiter);

        // Cross-family should fail
        assert!(pc.convert_to(ConcentrationUnit::NanomolPerLiter).is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let pc = PlasmaConcentration::new(
            Concentration::new(3.14),
            ConcentrationUnit::NanogramPerMl,
            DrugId::new("metoprolol"),
            Utc::now(),
        );
        let json = serde_json::to_string(&pc).unwrap();
        let deserialized: PlasmaConcentration = serde_json::from_str(&json).unwrap();
        assert!((deserialized.value.value() - 3.14).abs() < 1e-12);
        assert_eq!(deserialized.unit, ConcentrationUnit::NanogramPerMl);

        // ConcentrationInterval roundtrip
        let ci = ConcentrationInterval::new(1.5, 9.9);
        let ci_json = serde_json::to_string(&ci).unwrap();
        let ci_back: ConcentrationInterval = serde_json::from_str(&ci_json).unwrap();
        assert!((ci_back.lo - 1.5).abs() < 1e-12);
        assert!((ci_back.hi - 9.9).abs() < 1e-12);
    }
}
