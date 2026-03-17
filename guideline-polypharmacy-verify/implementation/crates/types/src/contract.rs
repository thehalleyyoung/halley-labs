//! Contract-based compositional verification for polypharmacy guidelines.
//!
//! This module implements a **contract theory** layer for reasoning about the
//! interaction of multiple clinical-practice guidelines. Each guideline is
//! wrapped in a [`GuidelineContract`] that declares:
//!
//! * **Assumptions** — what enzyme activity levels the guideline *expects*.
//! * **Guarantees** — what metabolic load the guideline *promises* to impose.
//!
//! Two contracts are **compatible** when the guarantee of one falls within the
//! assumption of the other for every shared enzyme. Compatible contracts can be
//! **composed** into a single [`ComposedContract`] that soundly over-approximates
//! the combined metabolic effect.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`ContractId`] | Unique identifier for any contract |
//! | [`EnzymeActivityInterval`] | Closed interval of enzyme activity levels |
//! | [`EnzymeLoadInterval`] | Closed interval of metabolic load/throughput |
//! | [`EnzymeContract`] | Single-enzyme assumption/guarantee pair |
//! | [`GuidelineContract`] | Full contract for one clinical guideline |
//! | [`ComposedContract`] | Result of composing multiple guideline contracts |
//! | [`ContractCompatibility`] | Outcome of a pairwise compatibility check |
//! | [`ContractCompositionResult`] | Outcome of an n-way composition attempt |
//! | [`ContractViolation`] | Runtime witness of a contract breach |

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

use crate::enzyme::CypEnzyme;
use crate::identifiers::IdParseError;

// ---------------------------------------------------------------------------
// ContractId
// ---------------------------------------------------------------------------

/// Unique identifier for a contract entity.
///
/// Wraps a [`Uuid`] and renders with a `CT-` prefix in its [`Display`] impl.
///
/// ```text
/// CT-550e8400-e29b-41d4-a716-446655440000
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContractId(Uuid);

impl ContractId {
    /// Create a new random contract identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the nil (all-zeros) identifier.
    pub fn nil() -> Self {
        Self(Uuid::nil())
    }

    /// Create an identifier from an existing [`Uuid`].
    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    /// Return the inner [`Uuid`].
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Returns `true` if this is the nil identifier.
    pub fn is_nil(&self) -> bool {
        self.0.is_nil()
    }

    /// Prefix used in human-readable representations.
    pub fn prefix() -> &'static str {
        "CT"
    }
}

impl Default for ContractId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ContractId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CT-{}", self.0)
    }
}

impl FromStr for ContractId {
    type Err = IdParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let uuid_part = if let Some(stripped) = s.strip_prefix("CT-") {
            stripped
        } else {
            s
        };
        let u = Uuid::parse_str(uuid_part).map_err(|e| IdParseError {
            input: s.to_string(),
            reason: e.to_string(),
        })?;
        Ok(Self(u))
    }
}

impl From<Uuid> for ContractId {
    fn from(u: Uuid) -> Self {
        Self(u)
    }
}

impl From<ContractId> for Uuid {
    fn from(id: ContractId) -> Self {
        id.0
    }
}

impl AsRef<Uuid> for ContractId {
    fn as_ref(&self) -> &Uuid {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// EnzymeActivityInterval
// ---------------------------------------------------------------------------

/// Closed interval `[lo, hi]` representing a range of enzyme activity levels.
///
/// Activity is expressed as a fraction of normal (1.0 = 100 % baseline).
/// An interval is **bottom** (empty) when `lo > hi`.
///
/// The lattice operations are:
/// * **join** (⊔) — smallest interval containing both operands.
/// * **meet** (⊓) — intersection of the two intervals (may be bottom).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnzymeActivityInterval {
    /// Lower bound of the activity range (inclusive).
    pub lo: f64,
    /// Upper bound of the activity range (inclusive).
    pub hi: f64,
}

impl EnzymeActivityInterval {
    /// Create a new activity interval.
    ///
    /// If `lo > hi` the interval is semantically empty (bottom).
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// The full range `[0, +∞)` — any activity is acceptable.
    pub fn top() -> Self {
        Self {
            lo: 0.0,
            hi: f64::INFINITY,
        }
    }

    /// An explicitly empty interval.
    pub fn bottom() -> Self {
        Self { lo: 1.0, hi: 0.0 }
    }

    /// Returns `true` if `v` is contained within `[lo, hi]`.
    pub fn contains_value(&self, v: f64) -> bool {
        !self.is_bottom() && v >= self.lo && v <= self.hi
    }

    /// Lattice join — smallest interval containing both `self` and `other`.
    ///
    /// If either operand is bottom the result is the other operand.
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

    /// Lattice meet — intersection of the two intervals.
    ///
    /// Returns bottom when the intervals are disjoint.
    pub fn meet(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        Self { lo, hi }
    }

    /// Returns `true` if the interval is empty (`lo > hi`).
    pub fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    /// Returns `true` if `self` and `other` share at least one point.
    pub fn overlaps(&self, other: &Self) -> bool {
        !self.meet(other).is_bottom()
    }

    /// Width of the interval. Returns 0.0 for bottom intervals.
    pub fn width(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            self.hi - self.lo
        }
    }
}

impl fmt::Display for EnzymeActivityInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() {
            write!(f, "⊥")
        } else {
            write!(f, "[{:.3}, {:.3}]", self.lo, self.hi)
        }
    }
}

// ---------------------------------------------------------------------------
// EnzymeLoadInterval
// ---------------------------------------------------------------------------

/// Closed interval `[lo, hi]` representing the metabolic load that a guideline
/// imposes on a CYP enzyme.
///
/// Semantically distinct from [`EnzymeActivityInterval`]: activity describes
/// the enzyme's *capacity* while load describes the *demand* placed on it.
///
/// The lattice operations mirror those of `EnzymeActivityInterval`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnzymeLoadInterval {
    /// Lower bound of the load range (inclusive).
    pub lo: f64,
    /// Upper bound of the load range (inclusive).
    pub hi: f64,
}

impl EnzymeLoadInterval {
    /// Create a new load interval.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// The unconstrained interval `[0, +∞)` — "any load is allowed."
    pub fn top() -> Self {
        Self {
            lo: 0.0,
            hi: f64::INFINITY,
        }
    }

    /// An explicitly empty interval.
    pub fn bottom() -> Self {
        Self { lo: 1.0, hi: 0.0 }
    }

    /// Returns `true` if `v` lies within `[lo, hi]`.
    pub fn contains(&self, v: f64) -> bool {
        !self.is_bottom() && v >= self.lo && v <= self.hi
    }

    /// Lattice join — smallest interval containing both operands.
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

    /// Lattice meet — intersection of the two intervals.
    pub fn meet(&self, other: &Self) -> Self {
        if self.is_bottom() || other.is_bottom() {
            return Self::bottom();
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        Self { lo, hi }
    }

    /// Returns `true` if the interval is empty (`lo > hi`).
    pub fn is_bottom(&self) -> bool {
        self.lo > self.hi
    }

    /// Returns `true` if `self` and `other` share at least one point.
    pub fn overlaps(&self, other: &Self) -> bool {
        !self.meet(other).is_bottom()
    }

    /// Width of the interval. Returns 0.0 for bottom intervals.
    pub fn width(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            self.hi - self.lo
        }
    }
}

impl fmt::Display for EnzymeLoadInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bottom() {
            write!(f, "⊥")
        } else {
            write!(f, "[{:.3}, {:.3}]", self.lo, self.hi)
        }
    }
}

// ---------------------------------------------------------------------------
// EnzymeContract
// ---------------------------------------------------------------------------

/// A single-enzyme contract: pairs an **assumption** about the enzyme's
/// activity level with a **guarantee** about the metabolic load imposed.
///
/// In assume/guarantee terminology:
///
/// > *If the enzyme activity lies within `assumption`, then the guideline's
/// > drug regimen will impose a metabolic load within `guarantee`.*
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeContract {
    /// Unique identifier of this enzyme contract.
    pub id: ContractId,
    /// The CYP enzyme this contract concerns.
    pub enzyme: CypEnzyme,
    /// Assumed range of enzyme activity (fraction of normal baseline).
    pub assumption: EnzymeActivityInterval,
    /// Guaranteed range of metabolic load imposed on the enzyme.
    pub guarantee: EnzymeLoadInterval,
}

impl EnzymeContract {
    /// Create a new enzyme contract.
    pub fn new(
        enzyme: CypEnzyme,
        assumption: EnzymeActivityInterval,
        guarantee: EnzymeLoadInterval,
    ) -> Self {
        Self {
            id: ContractId::new(),
            enzyme,
            assumption,
            guarantee,
        }
    }

    /// A contract is **satisfiable** when its assumption is not bottom
    /// (i.e., there exist activity values under which the guarantee applies).
    pub fn is_satisfiable(&self) -> bool {
        !self.assumption.is_bottom()
    }

    /// A contract is **trivial** when its guarantee is top (unconstrained),
    /// meaning the guideline may impose any load.
    pub fn is_trivial(&self) -> bool {
        self.guarantee.lo <= 0.0 && self.guarantee.hi.is_infinite()
    }

    /// Returns `true` if `activity` falls within the assumption interval.
    pub fn assumption_contains(&self, activity: f64) -> bool {
        self.assumption.contains_value(activity)
    }

    /// Returns `true` if `load` falls within the guarantee interval.
    pub fn guarantee_contains(&self, load: f64) -> bool {
        self.guarantee.contains(load)
    }
}

impl fmt::Display for EnzymeContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EnzymeContract({}: assume {} → guarantee {})",
            self.enzyme, self.assumption, self.guarantee
        )
    }
}

// ---------------------------------------------------------------------------
// GuidelineContract
// ---------------------------------------------------------------------------

/// A full contract for a single clinical practice guideline.
///
/// Bundles one or more [`EnzymeContract`]s together with human-readable
/// metadata (preconditions, postconditions, safety descriptions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineContract {
    /// Unique identifier of this guideline contract.
    pub id: ContractId,
    /// Human-readable name of the guideline (e.g., "Beers Criteria 2023").
    pub guideline_name: String,
    /// Per-enzyme sub-contracts.
    pub enzyme_contracts: Vec<EnzymeContract>,
    /// Prose description of the safety property this contract addresses.
    pub safety_property_description: String,
    /// Pre-conditions that must hold for this contract to be applicable
    /// (e.g., "Patient eGFR > 30 mL/min").
    pub preconditions: Vec<String>,
    /// Post-conditions guaranteed when the contract is satisfied
    /// (e.g., "No QT prolongation beyond 500 ms").
    pub postconditions: Vec<String>,
}

impl GuidelineContract {
    /// Build a new guideline contract.
    pub fn new(
        guideline_name: impl Into<String>,
        enzyme_contracts: Vec<EnzymeContract>,
        safety_property_description: impl Into<String>,
        preconditions: Vec<String>,
        postconditions: Vec<String>,
    ) -> Self {
        Self {
            id: ContractId::new(),
            guideline_name: guideline_name.into(),
            enzyme_contracts,
            safety_property_description: safety_property_description.into(),
            preconditions,
            postconditions,
        }
    }

    /// Set of all CYP enzymes referenced by this contract's sub-contracts.
    pub fn enzymes_covered(&self) -> HashSet<CypEnzyme> {
        self.enzyme_contracts.iter().map(|ec| ec.enzyme).collect()
    }

    /// Returns `true` if at least one sub-contract covers `enzyme`.
    pub fn has_contract_for_enzyme(&self, enzyme: CypEnzyme) -> bool {
        self.enzyme_contracts.iter().any(|ec| ec.enzyme == enzyme)
    }

    /// Return the first [`EnzymeContract`] that covers `enzyme`, if any.
    pub fn get_enzyme_contract(&self, enzyme: CypEnzyme) -> Option<&EnzymeContract> {
        self.enzyme_contracts.iter().find(|ec| ec.enzyme == enzyme)
    }

    /// Total number of enzyme sub-contracts.
    pub fn num_enzyme_contracts(&self) -> usize {
        self.enzyme_contracts.len()
    }
}

impl fmt::Display for GuidelineContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let enzymes: Vec<String> = self.enzymes_covered().iter().map(|e| e.to_string()).collect();
        write!(
            f,
            "GuidelineContract({}, enzymes=[{}], {} pre, {} post)",
            self.guideline_name,
            enzymes.join(", "),
            self.preconditions.len(),
            self.postconditions.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// ContractCompatibility
// ---------------------------------------------------------------------------

/// Outcome of checking whether two guideline contracts can coexist.
///
/// Compatible contracts can be safely composed; incompatible ones indicate a
/// potential drug–drug interaction hazard that requires clinical review.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContractCompatibility {
    /// The two contracts are compatible across all shared enzymes.
    Compatible,
    /// At least one enzyme shows an incompatibility.
    Incompatible {
        /// Human-readable explanation of the conflict.
        reason: String,
        /// The enzyme where the conflict was first detected, if applicable.
        enzyme: Option<CypEnzyme>,
    },
}

impl ContractCompatibility {
    /// Returns `true` for the [`Compatible`](Self::Compatible) variant.
    pub fn is_compatible(&self) -> bool {
        matches!(self, ContractCompatibility::Compatible)
    }

    /// Returns the reason string for [`Incompatible`](Self::Incompatible),
    /// or `None` for compatible pairs.
    pub fn reason(&self) -> Option<&str> {
        match self {
            ContractCompatibility::Compatible => None,
            ContractCompatibility::Incompatible { reason, .. } => Some(reason.as_str()),
        }
    }

    /// Returns the conflicting enzyme, if any.
    pub fn enzyme(&self) -> Option<CypEnzyme> {
        match self {
            ContractCompatibility::Compatible => None,
            ContractCompatibility::Incompatible { enzyme, .. } => *enzyme,
        }
    }
}

impl fmt::Display for ContractCompatibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractCompatibility::Compatible => write!(f, "Compatible"),
            ContractCompatibility::Incompatible { reason, enzyme } => {
                if let Some(e) = enzyme {
                    write!(f, "Incompatible({}): {}", e, reason)
                } else {
                    write!(f, "Incompatible: {}", reason)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ComposedContract
// ---------------------------------------------------------------------------

/// The result of successfully composing multiple [`GuidelineContract`]s.
///
/// The composed contract merges per-enzyme sub-contracts by **joining**
/// assumptions (widen) and **meeting** guarantees (tighten) for each shared
/// enzyme, producing a sound over-approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedContract {
    /// Unique identifier of the composed contract.
    pub id: ContractId,
    /// Identifiers of the source guideline contracts that were composed.
    pub source_contracts: Vec<ContractId>,
    /// Merged per-enzyme sub-contracts.
    pub enzyme_contracts: Vec<EnzymeContract>,
    /// Description of the composition method used (e.g., "pairwise-ag-merge").
    pub composition_method: String,
}

impl ComposedContract {
    /// Build a composed contract from pre-merged data.
    pub fn new(
        source_contracts: Vec<ContractId>,
        enzyme_contracts: Vec<EnzymeContract>,
        composition_method: impl Into<String>,
    ) -> Self {
        Self {
            id: ContractId::new(),
            source_contracts,
            enzyme_contracts,
            composition_method: composition_method.into(),
        }
    }

    /// Returns `true` if the composed contract covers `enzyme`.
    pub fn covers_enzyme(&self, enzyme: CypEnzyme) -> bool {
        self.enzyme_contracts.iter().any(|ec| ec.enzyme == enzyme)
    }

    /// Set of all enzymes covered by the composed contract.
    pub fn all_enzymes(&self) -> HashSet<CypEnzyme> {
        self.enzyme_contracts.iter().map(|ec| ec.enzyme).collect()
    }

    /// Number of source contracts that were composed.
    pub fn num_sources(&self) -> usize {
        self.source_contracts.len()
    }
}

impl fmt::Display for ComposedContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let enzymes: Vec<String> = self.all_enzymes().iter().map(|e| e.to_string()).collect();
        write!(
            f,
            "ComposedContract({}, {} sources, enzymes=[{}], method={})",
            self.id,
            self.num_sources(),
            enzymes.join(", "),
            self.composition_method,
        )
    }
}

// ---------------------------------------------------------------------------
// ContractCompositionResult
// ---------------------------------------------------------------------------

/// Outcome of attempting to compose one or more guideline contracts.
///
/// If all pairwise compatibility checks pass, [`composed`](Self::composed)
/// holds the merged contract. Otherwise `compatibility` explains the failure
/// and `diagnostics` may carry additional detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCompositionResult {
    /// The composed contract, present only on success.
    pub composed: Option<ComposedContract>,
    /// Overall compatibility verdict.
    pub compatibility: ContractCompatibility,
    /// Free-form diagnostic messages accumulated during composition.
    pub diagnostics: Vec<String>,
}

impl ContractCompositionResult {
    /// Returns `true` when composition succeeded.
    pub fn is_successful(&self) -> bool {
        self.composed.is_some() && self.compatibility.is_compatible()
    }

    /// Unwrap the composed contract, panicking if composition failed.
    ///
    /// # Panics
    ///
    /// Panics with a message including the incompatibility reason when the
    /// composition was not successful.
    pub fn unwrap_composed(self) -> ComposedContract {
        match self.composed {
            Some(c) => c,
            None => {
                let reason = self
                    .compatibility
                    .reason()
                    .unwrap_or("unknown reason");
                panic!(
                    "called `unwrap_composed()` on a failed composition: {}",
                    reason
                );
            }
        }
    }
}

impl fmt::Display for ContractCompositionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_successful() {
            write!(f, "CompositionResult(success)")
        } else {
            write!(f, "CompositionResult(failed: {})", self.compatibility)
        }
    }
}

// ---------------------------------------------------------------------------
// ContractViolation
// ---------------------------------------------------------------------------

/// Runtime witness of a contract breach: an observed enzyme activity value
/// that falls outside the contract's assumed range.
///
/// Violations are timestamped so they can be correlated with PK simulation
/// time-steps or clinical events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractViolation {
    /// Identifier of the contract that was violated.
    pub contract_id: ContractId,
    /// The enzyme where the violation was observed.
    pub enzyme: CypEnzyme,
    /// The assumption interval that should have held.
    pub expected_activity: EnzymeActivityInterval,
    /// The actual observed activity estimate.
    pub actual_activity_estimate: f64,
    /// Human-readable description of the violation.
    pub violation_description: String,
    /// Severity label (e.g., "critical", "warning", "info").
    pub severity: String,
    /// When the violation was detected.
    pub timestamp: DateTime<Utc>,
}

impl ContractViolation {
    /// Create a new violation record, timestamped to `Utc::now()`.
    pub fn new(
        contract_id: ContractId,
        enzyme: CypEnzyme,
        expected_activity: EnzymeActivityInterval,
        actual_activity_estimate: f64,
        violation_description: impl Into<String>,
        severity: impl Into<String>,
    ) -> Self {
        Self {
            contract_id,
            enzyme,
            expected_activity,
            actual_activity_estimate,
            violation_description: violation_description.into(),
            severity: severity.into(),
            timestamp: Utc::now(),
        }
    }

    /// Returns `true` if the severity is `"critical"`.
    pub fn is_critical(&self) -> bool {
        self.severity.eq_ignore_ascii_case("critical")
    }

    /// Signed distance from the actual value to the nearest interval bound.
    ///
    /// * Negative → the actual value is *below* the interval (`lo`).
    /// * Zero → exactly on a boundary.
    /// * Positive → the actual value is *above* the interval (`hi`).
    ///
    /// If the value is inside the interval the margin is zero (no violation).
    pub fn margin(&self) -> f64 {
        if self.expected_activity.is_bottom() {
            return f64::NAN;
        }
        if self.actual_activity_estimate < self.expected_activity.lo {
            self.actual_activity_estimate - self.expected_activity.lo
        } else if self.actual_activity_estimate > self.expected_activity.hi {
            self.actual_activity_estimate - self.expected_activity.hi
        } else {
            0.0
        }
    }
}

impl fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Violation[{}](enzyme={}, expected={}, actual={:.4}, severity={}, margin={:.4})",
            self.contract_id,
            self.enzyme,
            self.expected_activity,
            self.actual_activity_estimate,
            self.severity,
            self.margin(),
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions — compatibility & composition
// ---------------------------------------------------------------------------

/// Check pairwise compatibility of two guideline contracts.
///
/// For every CYP enzyme covered by **both** contracts the function verifies
/// that the *guarantee* of each contract overlaps with (is containable
/// within) the *assumption* of the other. Specifically, the guarantee load
/// interval of contract A must overlap the activity assumption of contract B
/// in their numeric range, and vice-versa.
///
/// If any shared enzyme fails this check, the function returns
/// [`ContractCompatibility::Incompatible`] with the offending enzyme and a
/// description of the conflict.
///
/// # Algorithm
///
/// For each shared enzyme *e*:
///
/// 1. Let `g_a = a.guarantee` and `asn_b = b.assumption` for enzyme *e*.
/// 2. Check that the load range `g_a` is contained within (or at least
///    overlaps) the activity range `asn_b`. The numeric interpretation is:
///    the guarantee's `[lo, hi]` must overlap `asn_b`'s `[lo, hi]`.
/// 3. Perform the symmetric check for `g_b` against `asn_a`.
///
/// Both directions must pass for the enzyme to be deemed compatible.
pub fn check_pairwise_compatibility(
    a: &GuidelineContract,
    b: &GuidelineContract,
) -> ContractCompatibility {
    let enzymes_a = a.enzymes_covered();
    let enzymes_b = b.enzymes_covered();
    let shared: HashSet<CypEnzyme> = enzymes_a.intersection(&enzymes_b).copied().collect();

    for enzyme in &shared {
        let ec_a = a.get_enzyme_contract(*enzyme).unwrap();
        let ec_b = b.get_enzyme_contract(*enzyme).unwrap();

        // Direction A → B:
        // A's guarantee load must be compatible with B's assumption activity.
        // We interpret this as: the numeric range of A's guarantee must overlap
        // with B's assumption range.
        let ga = &ec_a.guarantee;
        let asn_b = &ec_b.assumption;
        let overlap_ab = EnzymeActivityInterval::new(ga.lo, ga.hi)
            .overlaps(&EnzymeActivityInterval::new(asn_b.lo, asn_b.hi));

        if !overlap_ab {
            return ContractCompatibility::Incompatible {
                reason: format!(
                    "Guarantee of '{}' on {} ({}) does not overlap assumption of '{}' ({})",
                    a.guideline_name, enzyme, ga, b.guideline_name, asn_b,
                ),
                enzyme: Some(*enzyme),
            };
        }

        // Direction B → A:
        let gb = &ec_b.guarantee;
        let asn_a = &ec_a.assumption;
        let overlap_ba = EnzymeActivityInterval::new(gb.lo, gb.hi)
            .overlaps(&EnzymeActivityInterval::new(asn_a.lo, asn_a.hi));

        if !overlap_ba {
            return ContractCompatibility::Incompatible {
                reason: format!(
                    "Guarantee of '{}' on {} ({}) does not overlap assumption of '{}' ({})",
                    b.guideline_name, enzyme, gb, a.guideline_name, asn_a,
                ),
                enzyme: Some(*enzyme),
            };
        }
    }

    ContractCompatibility::Compatible
}

/// Check all pairwise compatibilities among a slice of guideline contracts.
///
/// Returns a vector of `(i, j, result)` triples for every pair `i < j`.
/// The indices correspond to positions in the input slice.
///
/// # Example
///
/// ```ignore
/// let results = check_mutual_compatibility(&[&gc1, &gc2, &gc3]);
/// for (i, j, compat) in &results {
///     println!("({}, {}): {}", i, j, compat);
/// }
/// ```
pub fn check_mutual_compatibility(
    contracts: &[&GuidelineContract],
) -> Vec<(usize, usize, ContractCompatibility)> {
    let n = contracts.len();
    let mut results = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let compat = check_pairwise_compatibility(contracts[i], contracts[j]);
            results.push((i, j, compat));
        }
    }
    results
}

/// Attempt to compose multiple guideline contracts into a single
/// [`ComposedContract`].
///
/// # Composition strategy
///
/// 1. Run [`check_mutual_compatibility`] over all input contracts.
/// 2. If **any** pair is incompatible, return a failure result containing the
///    first incompatibility found and all diagnostic messages.
/// 3. Otherwise, for every CYP enzyme mentioned by at least one input
///    contract, merge the per-enzyme sub-contracts:
///    * **Assumptions** are *joined* (widened) — the composed contract assumes
///      the union of all source assumptions.
///    * **Guarantees** are *met* (tightened) — the composed contract promises
///      the intersection of all source guarantees.
/// 4. If a meet produces a bottom guarantee for any enzyme, emit a diagnostic
///    warning but still include the enzyme contract (it is vacuously safe).
///
/// # Returns
///
/// A [`ContractCompositionResult`] that is either successful (with a
/// [`ComposedContract`]) or failed (with diagnostics).
pub fn compose_contracts(contracts: &[&GuidelineContract]) -> ContractCompositionResult {
    if contracts.is_empty() {
        return ContractCompositionResult {
            composed: None,
            compatibility: ContractCompatibility::Incompatible {
                reason: "No contracts provided for composition".to_string(),
                enzyme: None,
            },
            diagnostics: vec!["Empty input slice".to_string()],
        };
    }

    if contracts.len() == 1 {
        let c = contracts[0];
        let composed = ComposedContract::new(
            vec![c.id],
            c.enzyme_contracts.clone(),
            "identity",
        );
        return ContractCompositionResult {
            composed: Some(composed),
            compatibility: ContractCompatibility::Compatible,
            diagnostics: vec!["Single contract — trivial composition".to_string()],
        };
    }

    // Step 1: check all pairs.
    let mut diagnostics = Vec::new();
    let pairwise = check_mutual_compatibility(contracts);
    for (i, j, ref compat) in &pairwise {
        if !compat.is_compatible() {
            diagnostics.push(format!(
                "Incompatible pair ({}, {}): {}",
                contracts[*i].guideline_name,
                contracts[*j].guideline_name,
                compat,
            ));
        }
    }

    // If any pair failed, return early.
    if let Some((_, _, ref first_fail)) = pairwise.iter().find(|(_, _, c)| !c.is_compatible()) {
        return ContractCompositionResult {
            composed: None,
            compatibility: first_fail.clone(),
            diagnostics,
        };
    }

    // Step 2: collect all enzymes.
    let mut all_enzymes = HashSet::new();
    for c in contracts {
        all_enzymes.extend(c.enzymes_covered());
    }

    // Step 3: merge per-enzyme.
    let mut merged_enzyme_contracts = Vec::new();
    let source_ids: Vec<ContractId> = contracts.iter().map(|c| c.id).collect();

    for enzyme in &all_enzymes {
        let relevant: Vec<&EnzymeContract> = contracts
            .iter()
            .filter_map(|c| c.get_enzyme_contract(*enzyme))
            .collect();

        if relevant.is_empty() {
            continue;
        }

        // Start from the first contract's values and fold.
        let mut merged_assumption = relevant[0].assumption;
        let mut merged_guarantee = relevant[0].guarantee;

        for ec in &relevant[1..] {
            merged_assumption = merged_assumption.join(&ec.assumption);
            merged_guarantee = merged_guarantee.meet(&ec.guarantee);
        }

        if merged_guarantee.is_bottom() {
            diagnostics.push(format!(
                "Warning: merged guarantee for {} is bottom (empty); \
                 the contract is vacuously safe but may indicate over-constraining",
                enzyme,
            ));
        }

        let mut merged_ec = EnzymeContract::new(*enzyme, merged_assumption, merged_guarantee);
        merged_ec.id = ContractId::new();
        merged_enzyme_contracts.push(merged_ec);
    }

    diagnostics.push(format!(
        "Successfully composed {} contracts over {} enzymes",
        contracts.len(),
        all_enzymes.len(),
    ));

    let composed = ComposedContract::new(
        source_ids,
        merged_enzyme_contracts,
        "pairwise-ag-merge",
    );

    ContractCompositionResult {
        composed: Some(composed),
        compatibility: ContractCompatibility::Compatible,
        diagnostics,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    /// Build a minimal enzyme contract for testing.
    fn make_enzyme_contract(
        enzyme: CypEnzyme,
        asn_lo: f64,
        asn_hi: f64,
        guar_lo: f64,
        guar_hi: f64,
    ) -> EnzymeContract {
        EnzymeContract::new(
            enzyme,
            EnzymeActivityInterval::new(asn_lo, asn_hi),
            EnzymeLoadInterval::new(guar_lo, guar_hi),
        )
    }

    /// Build a minimal guideline contract for testing.
    fn make_guideline(
        name: &str,
        enzyme_contracts: Vec<EnzymeContract>,
    ) -> GuidelineContract {
        GuidelineContract::new(
            name,
            enzyme_contracts,
            format!("Safety property for {}", name),
            vec![format!("Pre: {} applicable", name)],
            vec![format!("Post: {} safe", name)],
        )
    }

    // -- ContractId -------------------------------------------------------

    #[test]
    fn test_contract_id_display_and_parse_roundtrip() {
        let id = ContractId::new();
        let s = id.to_string();
        assert!(s.starts_with("CT-"), "Display should start with CT-");
        let parsed: ContractId = s.parse().expect("should round-trip");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_contract_id_nil() {
        let id = ContractId::nil();
        assert!(id.is_nil());
        assert_eq!(
            id.to_string(),
            "CT-00000000-0000-0000-0000-000000000000"
        );
    }

    #[test]
    fn test_contract_id_serde_roundtrip() {
        let id = ContractId::new();
        let json = serde_json::to_string(&id).expect("serialize");
        let parsed: ContractId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(id, parsed);
    }

    // -- EnzymeActivityInterval -------------------------------------------

    #[test]
    fn test_activity_interval_lattice_ops() {
        let a = EnzymeActivityInterval::new(0.5, 1.0);
        let b = EnzymeActivityInterval::new(0.8, 1.5);

        // join should be [0.5, 1.5]
        let j = a.join(&b);
        assert!((j.lo - 0.5).abs() < 1e-12);
        assert!((j.hi - 1.5).abs() < 1e-12);

        // meet should be [0.8, 1.0]
        let m = a.meet(&b);
        assert!((m.lo - 0.8).abs() < 1e-12);
        assert!((m.hi - 1.0).abs() < 1e-12);
        assert!(!m.is_bottom());

        // disjoint → meet is bottom
        let c = EnzymeActivityInterval::new(2.0, 3.0);
        let m2 = a.meet(&c);
        assert!(m2.is_bottom());
    }

    #[test]
    fn test_activity_interval_contains_and_width() {
        let iv = EnzymeActivityInterval::new(0.3, 0.9);
        assert!(iv.contains_value(0.5));
        assert!(iv.contains_value(0.3));
        assert!(iv.contains_value(0.9));
        assert!(!iv.contains_value(0.1));
        assert!(!iv.contains_value(1.0));
        assert!((iv.width() - 0.6).abs() < 1e-12);

        let bot = EnzymeActivityInterval::bottom();
        assert!(!bot.contains_value(0.5));
        assert_eq!(bot.width(), 0.0);
    }

    // -- EnzymeLoadInterval -----------------------------------------------

    #[test]
    fn test_load_interval_operations() {
        let a = EnzymeLoadInterval::new(0.1, 0.5);
        let b = EnzymeLoadInterval::new(0.3, 0.7);

        assert!(a.contains(0.3));
        assert!(!a.contains(0.6));
        assert!(a.overlaps(&b));

        let m = a.meet(&b);
        assert!((m.lo - 0.3).abs() < 1e-12);
        assert!((m.hi - 0.5).abs() < 1e-12);

        let j = a.join(&b);
        assert!((j.lo - 0.1).abs() < 1e-12);
        assert!((j.hi - 0.7).abs() < 1e-12);
    }

    // -- EnzymeContract ---------------------------------------------------

    #[test]
    fn test_enzyme_contract_satisfiability_and_triviality() {
        let satisfiable = make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.1, 0.4);
        assert!(satisfiable.is_satisfiable());
        assert!(!satisfiable.is_trivial());
        assert!(satisfiable.assumption_contains(1.0));
        assert!(!satisfiable.assumption_contains(2.0));
        assert!(satisfiable.guarantee_contains(0.2));

        // Bottom assumption → unsatisfiable.
        let unsat = make_enzyme_contract(CypEnzyme::CYP2D6, 1.0, 0.0, 0.1, 0.4);
        assert!(!unsat.is_satisfiable());

        // Top guarantee → trivial.
        let trivial = EnzymeContract::new(
            CypEnzyme::CYP1A2,
            EnzymeActivityInterval::new(0.5, 1.5),
            EnzymeLoadInterval::top(),
        );
        assert!(trivial.is_trivial());
    }

    // -- GuidelineContract ------------------------------------------------

    #[test]
    fn test_guideline_contract_enzyme_queries() {
        let gc = make_guideline(
            "TestGL",
            vec![
                make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.1, 0.4),
                make_enzyme_contract(CypEnzyme::CYP2D6, 0.4, 1.2, 0.0, 0.3),
            ],
        );

        assert_eq!(gc.enzymes_covered().len(), 2);
        assert!(gc.has_contract_for_enzyme(CypEnzyme::CYP3A4));
        assert!(gc.has_contract_for_enzyme(CypEnzyme::CYP2D6));
        assert!(!gc.has_contract_for_enzyme(CypEnzyme::CYP1A2));
        assert!(gc.get_enzyme_contract(CypEnzyme::CYP3A4).is_some());
        assert!(gc.get_enzyme_contract(CypEnzyme::CYP1A2).is_none());
        assert_eq!(gc.num_enzyme_contracts(), 2);
    }

    // -- Compatibility ----------------------------------------------------

    #[test]
    fn test_compatible_contracts() {
        // Two guidelines sharing CYP3A4.
        // A: assumes activity [0.5, 1.5], guarantees load [0.5, 1.0]
        // B: assumes activity [0.4, 1.2], guarantees load [0.6, 1.1]
        // A's guarantee [0.5, 1.0] overlaps B's assumption [0.4, 1.2] ✓
        // B's guarantee [0.6, 1.1] overlaps A's assumption [0.5, 1.5] ✓
        let gc_a = make_guideline(
            "GuidelineA",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.5, 1.0)],
        );
        let gc_b = make_guideline(
            "GuidelineB",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.4, 1.2, 0.6, 1.1)],
        );

        let compat = check_pairwise_compatibility(&gc_a, &gc_b);
        assert!(compat.is_compatible());
        assert!(compat.reason().is_none());
    }

    #[test]
    fn test_incompatible_contracts() {
        // A: assumes activity [0.5, 1.0], guarantees load [5.0, 10.0]
        // B: assumes activity [0.1, 0.3], guarantees load [0.1, 0.2]
        // A's guarantee [5.0, 10.0] does NOT overlap B's assumption [0.1, 0.3] ✗
        let gc_a = make_guideline(
            "GuidelineA",
            vec![make_enzyme_contract(CypEnzyme::CYP2D6, 0.5, 1.0, 5.0, 10.0)],
        );
        let gc_b = make_guideline(
            "GuidelineB",
            vec![make_enzyme_contract(CypEnzyme::CYP2D6, 0.1, 0.3, 0.1, 0.2)],
        );

        let compat = check_pairwise_compatibility(&gc_a, &gc_b);
        assert!(!compat.is_compatible());
        assert!(compat.reason().is_some());
        assert_eq!(compat.enzyme(), Some(CypEnzyme::CYP2D6));
    }

    #[test]
    fn test_contracts_on_disjoint_enzymes_are_compatible() {
        // A covers CYP3A4, B covers CYP2D6 — no shared enzymes.
        let gc_a = make_guideline(
            "GuidelineA",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.1, 0.4)],
        );
        let gc_b = make_guideline(
            "GuidelineB",
            vec![make_enzyme_contract(CypEnzyme::CYP2D6, 0.4, 1.2, 0.2, 0.6)],
        );

        let compat = check_pairwise_compatibility(&gc_a, &gc_b);
        assert!(compat.is_compatible());
    }

    // -- Mutual compatibility ---------------------------------------------

    #[test]
    fn test_mutual_compatibility_three_contracts() {
        let gc_a = make_guideline(
            "A",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.5, 1.0)],
        );
        let gc_b = make_guideline(
            "B",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.4, 1.2, 0.6, 1.1)],
        );
        let gc_c = make_guideline(
            "C",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.3, 1.4, 0.4, 0.9)],
        );

        let results = check_mutual_compatibility(&[&gc_a, &gc_b, &gc_c]);
        assert_eq!(results.len(), 3); // C(3,2) = 3 pairs
        assert!(results.iter().all(|(_, _, c)| c.is_compatible()));
    }

    // -- Composition ------------------------------------------------------

    #[test]
    fn test_successful_composition() {
        // Two compatible contracts sharing CYP3A4.
        let gc_a = make_guideline(
            "A",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.5, 1.0)],
        );
        let gc_b = make_guideline(
            "B",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.4, 1.2, 0.6, 0.9)],
        );

        let result = compose_contracts(&[&gc_a, &gc_b]);
        assert!(result.is_successful());

        let composed = result.unwrap_composed();
        assert_eq!(composed.num_sources(), 2);
        assert!(composed.covers_enzyme(CypEnzyme::CYP3A4));

        // Merged assumption should be join: [0.4, 1.5]
        let ec = composed
            .enzyme_contracts
            .iter()
            .find(|e| e.enzyme == CypEnzyme::CYP3A4)
            .expect("should have CYP3A4");
        assert!((ec.assumption.lo - 0.4).abs() < 1e-12);
        assert!((ec.assumption.hi - 1.5).abs() < 1e-12);

        // Merged guarantee should be meet: [0.6, 0.9]
        assert!((ec.guarantee.lo - 0.6).abs() < 1e-12);
        assert!((ec.guarantee.hi - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_failed_composition() {
        let gc_a = make_guideline(
            "A",
            vec![make_enzyme_contract(CypEnzyme::CYP2D6, 0.5, 1.0, 5.0, 10.0)],
        );
        let gc_b = make_guideline(
            "B",
            vec![make_enzyme_contract(CypEnzyme::CYP2D6, 0.1, 0.3, 0.1, 0.2)],
        );

        let result = compose_contracts(&[&gc_a, &gc_b]);
        assert!(!result.is_successful());
        assert!(result.composed.is_none());
        assert!(!result.diagnostics.is_empty());
    }

    // -- Violation --------------------------------------------------------

    #[test]
    fn test_contract_violation_margin() {
        let expected = EnzymeActivityInterval::new(0.5, 1.5);
        let v_below = ContractViolation::new(
            ContractId::new(),
            CypEnzyme::CYP3A4,
            expected,
            0.3,
            "Activity too low",
            "critical",
        );
        assert!(v_below.is_critical());
        // 0.3 - 0.5 = -0.2
        assert!((v_below.margin() - (-0.2)).abs() < 1e-12);

        let v_above = ContractViolation::new(
            ContractId::new(),
            CypEnzyme::CYP3A4,
            expected,
            2.0,
            "Activity too high",
            "warning",
        );
        assert!(!v_above.is_critical());
        // 2.0 - 1.5 = 0.5
        assert!((v_above.margin() - 0.5).abs() < 1e-12);

        // Inside the interval → margin 0
        let v_ok = ContractViolation::new(
            ContractId::new(),
            CypEnzyme::CYP3A4,
            expected,
            1.0,
            "Borderline",
            "info",
        );
        assert_eq!(v_ok.margin(), 0.0);
    }

    // -- Serialization round-trip -----------------------------------------

    #[test]
    fn test_guideline_contract_serde_roundtrip() {
        let gc = make_guideline(
            "SerdeTest",
            vec![
                make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.1, 0.4),
                make_enzyme_contract(CypEnzyme::CYP2D6, 0.4, 1.2, 0.0, 0.3),
            ],
        );

        let json = serde_json::to_string_pretty(&gc).expect("serialize");
        let parsed: GuidelineContract =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.guideline_name, gc.guideline_name);
        assert_eq!(parsed.enzyme_contracts.len(), 2);
        assert_eq!(parsed.id, gc.id);
        assert_eq!(parsed.preconditions, gc.preconditions);
        assert_eq!(parsed.postconditions, gc.postconditions);
    }

    #[test]
    fn test_composed_contract_serde_roundtrip() {
        let gc_a = make_guideline(
            "A",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.5, 1.0)],
        );
        let gc_b = make_guideline(
            "B",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.4, 1.2, 0.6, 0.9)],
        );

        let result = compose_contracts(&[&gc_a, &gc_b]);
        let composed = result.unwrap_composed();

        let json = serde_json::to_string(&composed).expect("serialize");
        let parsed: ComposedContract =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.id, composed.id);
        assert_eq!(parsed.source_contracts.len(), 2);
        assert_eq!(parsed.composition_method, "pairwise-ag-merge");
    }

    // -- Edge cases -------------------------------------------------------

    #[test]
    fn test_compose_empty_input() {
        let result = compose_contracts(&[]);
        assert!(!result.is_successful());
    }

    #[test]
    fn test_compose_single_contract() {
        let gc = make_guideline(
            "Solo",
            vec![make_enzyme_contract(CypEnzyme::CYP1A2, 0.5, 1.5, 0.1, 0.4)],
        );
        let result = compose_contracts(&[&gc]);
        assert!(result.is_successful());
        let composed = result.unwrap_composed();
        assert_eq!(composed.num_sources(), 1);
    }

    #[test]
    fn test_compose_multi_enzyme() {
        // A covers CYP3A4 + CYP2D6, B covers CYP3A4 only.
        let gc_a = make_guideline(
            "MultiA",
            vec![
                make_enzyme_contract(CypEnzyme::CYP3A4, 0.5, 1.5, 0.5, 1.0),
                make_enzyme_contract(CypEnzyme::CYP2D6, 0.3, 1.0, 0.1, 0.5),
            ],
        );
        let gc_b = make_guideline(
            "MultiB",
            vec![make_enzyme_contract(CypEnzyme::CYP3A4, 0.4, 1.2, 0.6, 0.9)],
        );

        let result = compose_contracts(&[&gc_a, &gc_b]);
        assert!(result.is_successful());
        let composed = result.unwrap_composed();
        // Should cover both CYP3A4 (merged) and CYP2D6 (from A only).
        assert!(composed.covers_enzyme(CypEnzyme::CYP3A4));
        assert!(composed.covers_enzyme(CypEnzyme::CYP2D6));
        assert_eq!(composed.all_enzymes().len(), 2);
    }

    #[test]
    fn test_violation_display() {
        let v = ContractViolation::new(
            ContractId::nil(),
            CypEnzyme::CYP3A4,
            EnzymeActivityInterval::new(0.5, 1.5),
            0.2,
            "Activity below threshold",
            "critical",
        );
        let display = v.to_string();
        assert!(display.contains("CYP3A4"));
        assert!(display.contains("critical"));
    }
}
