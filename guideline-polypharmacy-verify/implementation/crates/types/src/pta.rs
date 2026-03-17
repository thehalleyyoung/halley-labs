//! # Pharmacological Timed Automata (PTA)
//!
//! This module implements the core PTA formalism used in Tier 2 of the
//! GuardPharma polypharmacy verification system. A PTA extends classical
//! timed automata with:
//!
//! - **Concentration variables** tracking plasma drug levels alongside clocks
//! - **Clinical state predicates** encoding patient conditions, lab flags,
//!   and medication status as Boolean-valued guards
//! - **Dose actions** on edges that model administration events
//!
//! The automaton is used for contract-based compositional model checking:
//! each clinical guideline is compiled into a PTA, and the parallel
//! composition of multiple guideline PTAs is checked for safety properties
//! (e.g., no concurrent contraindicated drugs, concentrations remain within
//! therapeutic windows).
//!
//! ## Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`PTA`] | Top-level automaton with locations, edges, clocks |
//! | [`Location`] | A discrete state (initial, accepting, error) |
//! | [`Guard`] | Boolean predicate over clocks, concentrations, clinical flags |
//! | [`Edge`] | Guarded transition with resets / dose actions |
//! | [`PTAState`] | Concrete valuation (location + clocks + concentrations + flags) |
//! | [`PTARun`] | A witnessed execution trace through the automaton |

use std::collections::VecDeque;
use std::fmt;
use std::str::FromStr;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::drug::DrugId;

// ---------------------------------------------------------------------------
// LocationId
// ---------------------------------------------------------------------------

/// Strongly-typed identifier for a PTA location.
///
/// Wraps a `u32` and provides `Display` / `FromStr` so locations can be
/// referred to by numeric id in traces and diagnostics.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct LocationId(pub u32);

impl LocationId {
    /// Create a new location identifier.
    pub fn new(id: u32) -> Self {
        LocationId(id)
    }
}

impl fmt::Display for LocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

impl FromStr for LocationId {
    type Err = String;

    /// Parse `"L<n>"` or a bare unsigned integer into a [`LocationId`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        let num_part = if trimmed.starts_with('L') || trimmed.starts_with('l') {
            &trimmed[1..]
        } else {
            trimmed
        };
        num_part
            .parse::<u32>()
            .map(LocationId)
            .map_err(|e| format!("invalid LocationId '{}': {}", s, e))
    }
}

// ---------------------------------------------------------------------------
// Location
// ---------------------------------------------------------------------------

/// A discrete location (node) in the PTA.
///
/// Locations may carry an invariant that must hold while the automaton
/// remains in that location (time can only elapse while the invariant is
/// satisfied).  Locations are also tagged as *initial*, *accepting*, or
/// *error* for verification purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    /// Unique location id.
    pub id: LocationId,
    /// Human-readable name (e.g. `"therapeutic"`, `"toxic_risk"`).
    pub name: String,
    /// Optional stay-invariant: time may pass only while this holds.
    pub invariant: Option<Invariant>,
    /// Whether this is the initial location of the automaton.
    pub is_initial: bool,
    /// Whether this is an accepting (safe / goal) location.
    pub is_accepting: bool,
    /// Whether this is an error location representing a safety violation.
    pub is_error: bool,
}

impl Location {
    /// Create a plain (non-special) location.
    pub fn new(id: LocationId, name: impl Into<String>) -> Self {
        Location {
            id,
            name: name.into(),
            invariant: None,
            is_initial: false,
            is_accepting: false,
            is_error: false,
        }
    }

    /// Create and mark as the initial location.
    pub fn initial(id: LocationId, name: impl Into<String>) -> Self {
        Location {
            is_initial: true,
            ..Self::new(id, name)
        }
    }

    /// Create and mark as an error location.
    pub fn error(id: LocationId, name: impl Into<String>) -> Self {
        Location {
            is_error: true,
            ..Self::new(id, name)
        }
    }

    /// Create and mark as an accepting location.
    pub fn accepting(id: LocationId, name: impl Into<String>) -> Self {
        Location {
            is_accepting: true,
            ..Self::new(id, name)
        }
    }

    /// Returns `true` when the location carries a stay-invariant.
    pub fn has_invariant(&self) -> bool {
        self.invariant.is_some()
    }
}

// ---------------------------------------------------------------------------
// ClockVariable
// ---------------------------------------------------------------------------

/// A real-valued clock variable in the PTA.
///
/// Clocks grow at rate 1 with the passage of time and can be reset to zero
/// on edges.  They appear in guards (e.g. `clock ≤ 24 h`) and invariants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClockVariable {
    /// Numeric clock id (used in valuations and guards).
    pub id: u32,
    /// Human-readable name (e.g. `"time_since_dose"`).
    pub name: String,
}

impl ClockVariable {
    /// Create a new clock variable.
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        ClockVariable {
            id,
            name: name.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ClockValuation
// ---------------------------------------------------------------------------

/// Maps each clock id to its current real-valued reading.
///
/// Supports the standard timed-automaton operations: reset individual clocks
/// to zero, advance all clocks uniformly by a time delay δ, and test whether
/// a clock satisfies a comparison bound.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClockValuation {
    values: IndexMap<u32, f64>,
}

impl ClockValuation {
    /// Empty valuation (no clocks registered).
    pub fn new() -> Self {
        ClockValuation {
            values: IndexMap::new(),
        }
    }

    /// Read the current value of clock `id`, returning `0.0` if unknown.
    pub fn get(&self, clock_id: u32) -> f64 {
        self.values.get(&clock_id).copied().unwrap_or(0.0)
    }

    /// Set clock `id` to an explicit value.
    pub fn set(&mut self, clock_id: u32, value: f64) {
        self.values.insert(clock_id, value);
    }

    /// Reset clock `id` to zero.
    pub fn reset(&mut self, clock_id: u32) {
        self.values.insert(clock_id, 0.0);
    }

    /// Reset *all* clocks to zero.
    pub fn reset_all(&mut self) {
        for v in self.values.values_mut() {
            *v = 0.0;
        }
    }

    /// Advance a single clock by `delta` time units.
    pub fn advance(&mut self, clock_id: u32, delta: f64) {
        let cur = self.get(clock_id);
        self.values.insert(clock_id, cur + delta);
    }

    /// Test `clock_id <op> bound`.
    pub fn satisfies_bound(&self, clock_id: u32, op: &GuardOperator, bound: f64) -> bool {
        op.evaluate(self.get(clock_id), bound)
    }

    /// Iterator over all (clock_id, value) pairs.
    pub fn all_clocks(&self) -> impl Iterator<Item = (&u32, &f64)> {
        self.values.iter()
    }

    /// Uniformly advance every registered clock by `delta`.
    pub fn delay(&mut self, delta: f64) {
        for v in self.values.values_mut() {
            *v += delta;
        }
    }
}

impl Default for ClockValuation {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GuardOperator
// ---------------------------------------------------------------------------

/// Relational comparison operator used in clock and concentration guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GuardOperator {
    /// Strictly less than (`<`).
    Lt,
    /// Less than or equal (`≤`).
    Le,
    /// Equal (`=`).
    Eq,
    /// Greater than or equal (`≥`).
    Ge,
    /// Strictly greater than (`>`).
    Gt,
}

impl GuardOperator {
    /// Evaluate `lhs <op> rhs`.
    pub fn evaluate(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            GuardOperator::Lt => lhs < rhs,
            GuardOperator::Le => lhs <= rhs,
            GuardOperator::Eq => (lhs - rhs).abs() < f64::EPSILON,
            GuardOperator::Ge => lhs >= rhs,
            GuardOperator::Gt => lhs > rhs,
        }
    }
}

impl fmt::Display for GuardOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sym = match self {
            GuardOperator::Lt => "<",
            GuardOperator::Le => "≤",
            GuardOperator::Eq => "=",
            GuardOperator::Ge => "≥",
            GuardOperator::Gt => ">",
        };
        write!(f, "{}", sym)
    }
}

// ---------------------------------------------------------------------------
// Guard
// ---------------------------------------------------------------------------

/// A Boolean predicate that labels PTA edges and invariants.
///
/// Guards can reference clocks, drug-concentration variables, or clinical
/// Boolean flags, and can be composed with `And`, `Or`, and `Not`.
///
/// # Examples
///
/// ```ignore
/// // Clock must be ≤ 24 h AND warfarin concentration < 5.0 mg/L
/// let g = Guard::ClockGuard { clock_id: 0, op: GuardOperator::Le, bound: 24.0 }
///     .and(Guard::ConcentrationGuard {
///         drug_id: DrugId::new("warfarin"),
///         op: GuardOperator::Lt,
///         bound: 5.0,
///     });
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Guard {
    /// Compare a clock variable against a constant bound.
    ClockGuard {
        clock_id: u32,
        op: GuardOperator,
        bound: f64,
    },
    /// Compare a drug's plasma concentration against a bound.
    ConcentrationGuard {
        drug_id: DrugId,
        op: GuardOperator,
        bound: f64,
    },
    /// Test a Boolean clinical flag (condition present, lab abnormal, …).
    ClinicalGuard {
        predicate_key: String,
        expected: bool,
    },
    /// Conjunction of two guards.
    And(Box<Guard>, Box<Guard>),
    /// Disjunction of two guards.
    Or(Box<Guard>, Box<Guard>),
    /// Negation of a guard.
    Not(Box<Guard>),
    /// Trivially true (always enabled).
    True,
    /// Trivially false (never enabled).
    False,
}

impl Guard {
    /// Evaluate the guard against a concrete state.
    ///
    /// - `clock_val` — current clock readings
    /// - `concentrations` — map from drug id string to plasma concentration
    /// - `clinical_flags` — map from predicate key to Boolean value
    pub fn evaluate(
        &self,
        clock_val: &ClockValuation,
        concentrations: &IndexMap<String, f64>,
        clinical_flags: &IndexMap<String, bool>,
    ) -> bool {
        match self {
            Guard::ClockGuard {
                clock_id,
                op,
                bound,
            } => clock_val.satisfies_bound(*clock_id, op, *bound),

            Guard::ConcentrationGuard {
                drug_id,
                op,
                bound,
            } => {
                let conc = concentrations
                    .get(drug_id.as_str())
                    .copied()
                    .unwrap_or(0.0);
                op.evaluate(conc, *bound)
            }

            Guard::ClinicalGuard {
                predicate_key,
                expected,
            } => {
                let actual = clinical_flags.get(predicate_key).copied().unwrap_or(false);
                actual == *expected
            }

            Guard::And(lhs, rhs) => {
                lhs.evaluate(clock_val, concentrations, clinical_flags)
                    && rhs.evaluate(clock_val, concentrations, clinical_flags)
            }
            Guard::Or(lhs, rhs) => {
                lhs.evaluate(clock_val, concentrations, clinical_flags)
                    || rhs.evaluate(clock_val, concentrations, clinical_flags)
            }
            Guard::Not(inner) => !inner.evaluate(clock_val, concentrations, clinical_flags),

            Guard::True => true,
            Guard::False => false,
        }
    }

    /// Build the conjunction `self ∧ other`.
    pub fn and(self, other: Guard) -> Guard {
        match (&self, &other) {
            (Guard::True, _) => other,
            (_, Guard::True) => self,
            (Guard::False, _) | (_, Guard::False) => Guard::False,
            _ => Guard::And(Box::new(self), Box::new(other)),
        }
    }

    /// Build the disjunction `self ∨ other`.
    pub fn or(self, other: Guard) -> Guard {
        match (&self, &other) {
            (Guard::False, _) => other,
            (_, Guard::False) => self,
            (Guard::True, _) | (_, Guard::True) => Guard::True,
            _ => Guard::Or(Box::new(self), Box::new(other)),
        }
    }

    /// Build the negation `¬self`.
    pub fn negate(self) -> Guard {
        match self {
            Guard::True => Guard::False,
            Guard::False => Guard::True,
            Guard::Not(inner) => *inner,
            other => Guard::Not(Box::new(other)),
        }
    }

    /// Returns `true` when the guard is syntactically `True`.
    pub fn is_trivially_true(&self) -> bool {
        matches!(self, Guard::True)
    }

    /// Returns `true` when the guard is syntactically `False`.
    pub fn is_trivially_false(&self) -> bool {
        matches!(self, Guard::False)
    }
}

impl fmt::Display for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Guard::ClockGuard {
                clock_id,
                op,
                bound,
            } => write!(f, "x{} {} {}", clock_id, op, bound),
            Guard::ConcentrationGuard {
                drug_id,
                op,
                bound,
            } => write!(f, "[{}] {} {}", drug_id, op, bound),
            Guard::ClinicalGuard {
                predicate_key,
                expected,
            } => {
                if *expected {
                    write!(f, "{}", predicate_key)
                } else {
                    write!(f, "¬{}", predicate_key)
                }
            }
            Guard::And(l, r) => write!(f, "({} ∧ {})", l, r),
            Guard::Or(l, r) => write!(f, "({} ∨ {})", l, r),
            Guard::Not(g) => write!(f, "¬({})", g),
            Guard::True => write!(f, "true"),
            Guard::False => write!(f, "false"),
        }
    }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

/// An action performed when an edge fires.
///
/// Resets can zero a clock, update a concentration variable, toggle a
/// clinical flag, or record a dose-administration event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Reset {
    /// Set a clock back to zero.
    ClockReset { clock_id: u32 },
    /// Overwrite a drug concentration with a new value.
    ConcentrationUpdate { drug_id: String, new_value: f64 },
    /// Update a Boolean clinical-state flag.
    ClinicalStateUpdate { key: String, value: bool },
    /// Record a dose administration (drug, amount, route).
    DoseAction {
        drug_id: String,
        dose_mg: f64,
        route: String,
    },
}

impl Reset {
    /// Apply this reset to a clock valuation (only `ClockReset` has effect).
    pub fn apply_to_clocks(&self, val: &mut ClockValuation) {
        if let Reset::ClockReset { clock_id } = self {
            val.reset(*clock_id);
        }
    }

    /// Apply this reset to the concentration map (only `ConcentrationUpdate`
    /// and `DoseAction` have effect — `DoseAction` is modeled as an additive
    /// bolus scaled by a nominal bioavailability factor of 1.0).
    pub fn apply_to_concentrations(&self, concs: &mut IndexMap<String, f64>) {
        match self {
            Reset::ConcentrationUpdate { drug_id, new_value } => {
                concs.insert(drug_id.clone(), *new_value);
            }
            Reset::DoseAction {
                drug_id, dose_mg, ..
            } => {
                let cur = concs.get(drug_id).copied().unwrap_or(0.0);
                concs.insert(drug_id.clone(), cur + dose_mg);
            }
            _ => {}
        }
    }

    /// Apply this reset to clinical flags (only `ClinicalStateUpdate` has
    /// effect).
    pub fn apply_to_clinical(&self, flags: &mut IndexMap<String, bool>) {
        if let Reset::ClinicalStateUpdate { key, value } = self {
            flags.insert(key.clone(), *value);
        }
    }
}

impl fmt::Display for Reset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reset::ClockReset { clock_id } => write!(f, "x{} := 0", clock_id),
            Reset::ConcentrationUpdate { drug_id, new_value } => {
                write!(f, "[{}] := {}", drug_id, new_value)
            }
            Reset::ClinicalStateUpdate { key, value } => {
                write!(f, "{} := {}", key, value)
            }
            Reset::DoseAction {
                drug_id,
                dose_mg,
                route,
            } => write!(f, "dose({}, {} mg, {})", drug_id, dose_mg, route),
        }
    }
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// A guarded transition between two PTA locations.
///
/// An edge is *enabled* when its guard evaluates to `true` in the current
/// concrete state.  Firing the edge moves the automaton to the target
/// location and applies all resets.  The `priority` field resolves
/// non-determinism: among simultaneously enabled edges, only those with the
/// highest (numerically lowest) priority may fire.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    /// Source location.
    pub source: LocationId,
    /// Target location.
    pub target: LocationId,
    /// Enabling guard.
    pub guard: Guard,
    /// Resets / actions performed on firing.
    pub resets: Vec<Reset>,
    /// Human-readable label (e.g. `"administer_warfarin"`).
    pub label: String,
    /// Priority (lower number = higher priority, 0 is highest).
    pub priority: u32,
}

impl Edge {
    /// Create a new edge with no resets and default priority 0.
    pub fn new(
        source: LocationId,
        target: LocationId,
        guard: Guard,
        label: impl Into<String>,
    ) -> Self {
        Edge {
            source,
            target,
            guard,
            resets: Vec::new(),
            label: label.into(),
            priority: 0,
        }
    }

    /// Check whether this edge is enabled in the given concrete state.
    pub fn is_enabled(
        &self,
        clock_val: &ClockValuation,
        concentrations: &IndexMap<String, f64>,
        clinical_flags: &IndexMap<String, bool>,
    ) -> bool {
        self.guard
            .evaluate(clock_val, concentrations, clinical_flags)
    }

    /// Apply all resets of this edge to the given mutable state components.
    pub fn apply_resets(
        &self,
        clock_val: &mut ClockValuation,
        concentrations: &mut IndexMap<String, f64>,
        clinical_flags: &mut IndexMap<String, bool>,
    ) {
        for r in &self.resets {
            r.apply_to_clocks(clock_val);
            r.apply_to_concentrations(concentrations);
            r.apply_to_clinical(clinical_flags);
        }
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} --[{} | {}]--> {}",
            self.source, self.label, self.guard, self.target
        )
    }
}

// ---------------------------------------------------------------------------
// Invariant
// ---------------------------------------------------------------------------

/// A stay-invariant attached to a PTA location.
///
/// Time may only elapse while the invariant guard evaluates to `true`.
/// Once the invariant is about to be violated, the automaton *must* take
/// an outgoing edge (or the run is invalid / time-blocked).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Invariant {
    guard: Guard,
}

impl Invariant {
    /// Wrap a guard as an invariant.
    pub fn new(guard: Guard) -> Self {
        Invariant { guard }
    }

    /// Check whether the invariant is satisfied by the concrete state.
    pub fn is_satisfied(
        &self,
        clock_val: &ClockValuation,
        concentrations: &IndexMap<String, f64>,
        clinical_flags: &IndexMap<String, bool>,
    ) -> bool {
        self.guard
            .evaluate(clock_val, concentrations, clinical_flags)
    }

    /// Borrow the underlying guard.
    pub fn guard(&self) -> &Guard {
        &self.guard
    }
}

impl fmt::Display for Invariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "inv({})", self.guard)
    }
}

// ---------------------------------------------------------------------------
// PTA (Pharmacological Timed Automaton)
// ---------------------------------------------------------------------------

/// Top-level Pharmacological Timed Automaton.
///
/// A PTA bundles:
/// - a finite set of **locations** (one initial, some accepting, some error),
/// - a set of real-valued **clocks**,
/// - named **concentration variables** (one per drug),
/// - named **clinical variables** (Boolean flags),
/// - a set of guarded **edges** with resets / dose actions.
///
/// The semantics follow standard timed automata extended with the
/// pharmacological dimensions above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PTA {
    /// Automaton name (e.g. `"warfarin_guideline"`).
    pub name: String,
    /// Locations indexed by their [`LocationId`].
    pub locations: IndexMap<LocationId, Location>,
    /// The designated initial location.
    pub initial_location: LocationId,
    /// Clock variables.
    pub clocks: Vec<ClockVariable>,
    /// Transition edges.
    pub edges: Vec<Edge>,
    /// Drug-concentration variable names.
    pub concentration_vars: Vec<String>,
    /// Boolean clinical variable names.
    pub clinical_vars: Vec<String>,
}

impl PTA {
    /// Create a PTA with a single initial location.
    ///
    /// The initial location is automatically inserted into the location map.
    pub fn new(name: impl Into<String>, initial: Location) -> Self {
        let init_id = initial.id;
        let mut locations = IndexMap::new();
        locations.insert(init_id, initial);
        PTA {
            name: name.into(),
            locations,
            initial_location: init_id,
            clocks: Vec::new(),
            edges: Vec::new(),
            concentration_vars: Vec::new(),
            clinical_vars: Vec::new(),
        }
    }

    /// Insert a location.  Panics if a location with the same id exists.
    pub fn add_location(&mut self, loc: Location) {
        assert!(
            !self.locations.contains_key(&loc.id),
            "Duplicate location id: {}",
            loc.id
        );
        self.locations.insert(loc.id, loc);
    }

    /// Append an edge.  Both source and target locations must already exist.
    pub fn add_edge(&mut self, edge: Edge) {
        assert!(
            self.locations.contains_key(&edge.source),
            "Edge source {} not in PTA",
            edge.source
        );
        assert!(
            self.locations.contains_key(&edge.target),
            "Edge target {} not in PTA",
            edge.target
        );
        self.edges.push(edge);
    }

    /// Register a clock variable.
    pub fn add_clock(&mut self, clock: ClockVariable) {
        self.clocks.push(clock);
    }

    /// Look up a location by id.
    pub fn location(&self, id: LocationId) -> Option<&Location> {
        self.locations.get(&id)
    }

    /// Iterate over all locations.
    pub fn locations_iter(&self) -> impl Iterator<Item = (&LocationId, &Location)> {
        self.locations.iter()
    }

    /// Return all edges whose source is `loc`.
    pub fn edges_from(&self, loc: LocationId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.source == loc).collect()
    }

    /// Return edges from `loc` that are enabled in the given concrete state,
    /// sorted by priority (lower number first).
    pub fn enabled_edges(
        &self,
        loc: LocationId,
        clock_val: &ClockValuation,
        concentrations: &IndexMap<String, f64>,
        clinical_flags: &IndexMap<String, bool>,
    ) -> Vec<&Edge> {
        let mut enabled: Vec<&Edge> = self
            .edges_from(loc)
            .into_iter()
            .filter(|e| e.is_enabled(clock_val, concentrations, clinical_flags))
            .collect();
        enabled.sort_by_key(|e| e.priority);
        enabled
    }

    /// Build the initial [`PTAState`] with all clocks at zero and
    /// concentration / clinical variables at their defaults (0.0 / false).
    pub fn initial_state(&self) -> PTAState {
        let mut cv = ClockValuation::new();
        for c in &self.clocks {
            cv.set(c.id, 0.0);
        }
        let mut concentrations = IndexMap::new();
        for var in &self.concentration_vars {
            concentrations.insert(var.clone(), 0.0);
        }
        let mut clinical_flags = IndexMap::new();
        for var in &self.clinical_vars {
            clinical_flags.insert(var.clone(), false);
        }
        PTAState {
            location: self.initial_location,
            clock_valuation: cv,
            concentrations,
            clinical_flags,
        }
    }

    /// Compute all successor states reachable by firing a single enabled
    /// edge from `state`.
    ///
    /// Returns `(successor_state, edge_reference)` pairs.
    pub fn successor_states<'a>(
        &'a self,
        state: &PTAState,
    ) -> Vec<(PTAState, &'a Edge)> {
        let enabled = self.enabled_edges(
            state.location,
            &state.clock_valuation,
            &state.concentrations,
            &state.clinical_flags,
        );
        enabled
            .into_iter()
            .map(|edge| {
                let mut new_cv = state.clock_valuation.clone();
                let mut new_concs = state.concentrations.clone();
                let mut new_flags = state.clinical_flags.clone();
                edge.apply_resets(&mut new_cv, &mut new_concs, &mut new_flags);
                let succ = PTAState {
                    location: edge.target,
                    clock_valuation: new_cv,
                    concentrations: new_concs,
                    clinical_flags: new_flags,
                };
                (succ, edge)
            })
            .collect()
    }

    /// A PTA is *syntactically deterministic* if, for every location, no two
    /// outgoing edges with the same priority have guards that are not
    /// mutually exclusive.
    ///
    /// This is a conservative approximation: we check only that no two edges
    /// from the same source share the same priority and both have
    /// non-trivially-false guards.  Full semantic determinism would require
    /// a satisfiability check on guard conjunctions.
    pub fn is_deterministic(&self) -> bool {
        for (loc_id, _) in &self.locations {
            let outgoing = self.edges_from(*loc_id);
            for i in 0..outgoing.len() {
                for j in (i + 1)..outgoing.len() {
                    let ei = outgoing[i];
                    let ej = outgoing[j];
                    if ei.priority == ej.priority
                        && !ei.guard.is_trivially_false()
                        && !ej.guard.is_trivially_false()
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Compute the set of location ids reachable from the initial location
    /// via a breadth-first traversal of the static edge graph (ignoring
    /// guard semantics).
    pub fn reachable_locations(&self) -> Vec<LocationId> {
        let mut visited = IndexMap::new();
        let mut queue = VecDeque::new();
        visited.insert(self.initial_location, ());
        queue.push_back(self.initial_location);

        while let Some(loc) = queue.pop_front() {
            for edge in self.edges_from(loc) {
                if !visited.contains_key(&edge.target) {
                    visited.insert(edge.target, ());
                    queue.push_back(edge.target);
                }
            }
        }
        visited.keys().copied().collect()
    }
}

impl fmt::Display for PTA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PTA '{}' ({} locations, {} edges, {} clocks)",
            self.name,
            self.locations.len(),
            self.edges.len(),
            self.clocks.len(),
        )?;
        writeln!(f, "  initial: {}", self.initial_location)?;
        for (id, loc) in &self.locations {
            let tags: Vec<&str> = [
                if loc.is_initial { Some("init") } else { None },
                if loc.is_accepting { Some("accept") } else { None },
                if loc.is_error { Some("error") } else { None },
            ]
            .iter()
            .filter_map(|t| *t)
            .collect();
            let tag_str = if tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", tags.join(", "))
            };
            writeln!(f, "  {} '{}'{}", id, loc.name, tag_str)?;
        }
        for edge in &self.edges {
            writeln!(f, "  {}", edge)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PTAState
// ---------------------------------------------------------------------------

/// A concrete state of a PTA: the current location together with valuations
/// of all clocks, drug concentrations, and clinical flags.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PTAState {
    /// Current location.
    pub location: LocationId,
    /// Current clock readings.
    pub clock_valuation: ClockValuation,
    /// Current drug plasma concentrations (drug id → mg/L).
    pub concentrations: IndexMap<String, f64>,
    /// Current clinical Boolean flags.
    pub clinical_flags: IndexMap<String, bool>,
}

impl PTAState {
    /// Build a state from its components.
    pub fn new(
        location: LocationId,
        clock_valuation: ClockValuation,
        concentrations: IndexMap<String, f64>,
        clinical_flags: IndexMap<String, bool>,
    ) -> Self {
        PTAState {
            location,
            clock_valuation,
            concentrations,
            clinical_flags,
        }
    }

    /// Returns `true` if the automaton is currently at `loc`.
    pub fn is_in_location(&self, loc: LocationId) -> bool {
        self.location == loc
    }

    /// Check whether the invariant of the *current* location is satisfied.
    ///
    /// Returns `true` when the location has no invariant.
    pub fn satisfies_invariant(&self, pta: &PTA) -> bool {
        match pta.location(self.location) {
            Some(loc) => match &loc.invariant {
                Some(inv) => inv.is_satisfied(
                    &self.clock_valuation,
                    &self.concentrations,
                    &self.clinical_flags,
                ),
                None => true,
            },
            None => true,
        }
    }
}

impl fmt::Display for PTAState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}", self.location)?;
        for (cid, val) in self.clock_valuation.all_clocks() {
            write!(f, ", x{}={:.2}", cid, val)?;
        }
        for (drug, conc) in &self.concentrations {
            write!(f, ", [{}]={:.2}", drug, conc)?;
        }
        for (key, flag) in &self.clinical_flags {
            if *flag {
                write!(f, ", {}", key)?;
            }
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// PTARun
// ---------------------------------------------------------------------------

/// A witnessed execution (trace) through a PTA.
///
/// A run records the sequence of states visited, the edges taken between
/// them, and a timestamp for each step (cumulative elapsed time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PTARun {
    /// States visited (length = edges.len() + 1).
    pub states: Vec<PTAState>,
    /// Edges fired between consecutive states.
    pub edges: Vec<Edge>,
    /// Cumulative time at each state (same length as `states`).
    pub timestamps: Vec<f64>,
}

impl PTARun {
    /// Start a new run from an initial state at time 0.
    pub fn new(initial_state: PTAState) -> Self {
        PTARun {
            states: vec![initial_state],
            edges: Vec::new(),
            timestamps: vec![0.0],
        }
    }

    /// Extend the run by one discrete step.
    ///
    /// `elapsed` is the *additional* time that passes before the edge fires
    /// (delay transition followed by discrete transition).
    pub fn push_step(&mut self, edge: Edge, successor: PTAState, elapsed: f64) {
        let last_time = self.timestamps.last().copied().unwrap_or(0.0);
        self.timestamps.push(last_time + elapsed);
        self.edges.push(edge);
        self.states.push(successor);
    }

    /// Number of discrete transitions taken so far.
    pub fn length(&self) -> usize {
        self.edges.len()
    }

    /// The most recently visited state.
    pub fn last_state(&self) -> Option<&PTAState> {
        self.states.last()
    }

    /// Returns `true` when the run's final state is in an accepting location.
    pub fn is_accepting(&self, pta: &PTA) -> bool {
        self.last_state()
            .and_then(|s| pta.location(s.location))
            .map_or(false, |loc| loc.is_accepting)
    }

    /// Returns `true` when the run's final state is in an error location.
    pub fn reaches_error(&self, pta: &PTA) -> bool {
        self.last_state()
            .and_then(|s| pta.location(s.location))
            .map_or(false, |loc| loc.is_error)
    }

    /// Produce a human-readable summary of the trace.
    ///
    /// Each line shows `[time] location --label--> location`.
    pub fn trace_summary(&self) -> String {
        let mut lines = Vec::new();
        if let Some(first) = self.states.first() {
            lines.push(format!("[t={:.2}] @ {}", 0.0, first.location));
        }
        for (i, edge) in self.edges.iter().enumerate() {
            let t = self.timestamps.get(i + 1).copied().unwrap_or(0.0);
            lines.push(format!(
                "[t={:.2}] {} --{}--> {}",
                t, edge.source, edge.label, edge.target
            ));
        }
        lines.join("\n")
    }
}

impl fmt::Display for PTARun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.trace_summary())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn empty_concs() -> IndexMap<String, f64> {
        IndexMap::new()
    }

    fn empty_flags() -> IndexMap<String, bool> {
        IndexMap::new()
    }

    fn make_clock_val(pairs: &[(u32, f64)]) -> ClockValuation {
        let mut cv = ClockValuation::new();
        for &(id, v) in pairs {
            cv.set(id, v);
        }
        cv
    }

    // -- 1. Guard evaluation: all atomic variants -------------------------

    #[test]
    fn test_guard_clock() {
        let cv = make_clock_val(&[(0, 10.0)]);
        let g = Guard::ClockGuard {
            clock_id: 0,
            op: GuardOperator::Le,
            bound: 24.0,
        };
        assert!(g.evaluate(&cv, &empty_concs(), &empty_flags()));

        let cv_over = make_clock_val(&[(0, 30.0)]);
        assert!(!g.evaluate(&cv_over, &empty_concs(), &empty_flags()));
    }

    #[test]
    fn test_guard_concentration() {
        let mut concs = IndexMap::new();
        concs.insert("warfarin".to_string(), 3.5);
        let g = Guard::ConcentrationGuard {
            drug_id: DrugId::new("warfarin"),
            op: GuardOperator::Lt,
            bound: 5.0,
        };
        assert!(g.evaluate(&ClockValuation::new(), &concs, &empty_flags()));

        concs.insert("warfarin".to_string(), 6.0);
        assert!(!g.evaluate(&ClockValuation::new(), &concs, &empty_flags()));
    }

    #[test]
    fn test_guard_clinical() {
        let mut flags = IndexMap::new();
        flags.insert("renal_impairment".to_string(), true);
        let g = Guard::ClinicalGuard {
            predicate_key: "renal_impairment".to_string(),
            expected: true,
        };
        assert!(g.evaluate(&ClockValuation::new(), &empty_concs(), &flags));

        let g_neg = Guard::ClinicalGuard {
            predicate_key: "renal_impairment".to_string(),
            expected: false,
        };
        assert!(!g_neg.evaluate(&ClockValuation::new(), &empty_concs(), &flags));
    }

    // -- 2. Nested And / Or / Not guards ----------------------------------

    #[test]
    fn test_guard_and_or_not() {
        let cv = make_clock_val(&[(0, 5.0)]);
        let g_clock = Guard::ClockGuard {
            clock_id: 0,
            op: GuardOperator::Le,
            bound: 10.0,
        };
        let g_false = Guard::False;

        // And: true ∧ false = false
        let g_and = g_clock.clone().and(g_false.clone());
        assert!(!g_and.evaluate(&cv, &empty_concs(), &empty_flags()));

        // Or: true ∨ false = true
        let g_or = g_clock.clone().or(g_false);
        assert!(g_or.evaluate(&cv, &empty_concs(), &empty_flags()));

        // Not: ¬true = false
        let g_not = g_clock.negate();
        assert!(!g_not.evaluate(&cv, &empty_concs(), &empty_flags()));
    }

    #[test]
    fn test_guard_deep_nesting() {
        // (x0 ≤ 10) ∧ (¬(x0 > 20)) — both should be true when x0 = 5
        let cv = make_clock_val(&[(0, 5.0)]);
        let g = Guard::ClockGuard {
            clock_id: 0,
            op: GuardOperator::Le,
            bound: 10.0,
        }
        .and(
            Guard::ClockGuard {
                clock_id: 0,
                op: GuardOperator::Gt,
                bound: 20.0,
            }
            .negate(),
        );
        assert!(g.evaluate(&cv, &empty_concs(), &empty_flags()));
    }

    // -- 3. Trivially true / false simplification -------------------------

    #[test]
    fn test_guard_trivial_simplification() {
        assert!(Guard::True.and(Guard::True).is_trivially_true());
        assert!(Guard::False.or(Guard::False).is_trivially_false());
        assert!(Guard::True.negate().is_trivially_false());
        assert!(Guard::False.negate().is_trivially_true());
        // True ∧ X = X  (not trivially true when X is a clock guard)
        let g = Guard::True.and(Guard::ClockGuard {
            clock_id: 0,
            op: GuardOperator::Lt,
            bound: 1.0,
        });
        assert!(!g.is_trivially_true());
    }

    // -- 4. Clock valuation operations ------------------------------------

    #[test]
    fn test_clock_valuation_delay_and_reset() {
        let mut cv = ClockValuation::new();
        cv.set(0, 0.0);
        cv.set(1, 0.0);
        cv.delay(5.0);
        assert!((cv.get(0) - 5.0).abs() < 1e-10);
        assert!((cv.get(1) - 5.0).abs() < 1e-10);

        cv.reset(0);
        assert!((cv.get(0)).abs() < 1e-10);
        assert!((cv.get(1) - 5.0).abs() < 1e-10);

        cv.reset_all();
        assert!((cv.get(0)).abs() < 1e-10);
        assert!((cv.get(1)).abs() < 1e-10);
    }

    // -- 5. Edge enabling and reset application ---------------------------

    #[test]
    fn test_edge_enabled_and_resets() {
        let cv = make_clock_val(&[(0, 2.0)]);
        let concs = empty_concs();
        let flags = empty_flags();

        let edge = Edge {
            source: LocationId(0),
            target: LocationId(1),
            guard: Guard::ClockGuard {
                clock_id: 0,
                op: GuardOperator::Ge,
                bound: 1.0,
            },
            resets: vec![
                Reset::ClockReset { clock_id: 0 },
                Reset::ConcentrationUpdate {
                    drug_id: "aspirin".to_string(),
                    new_value: 100.0,
                },
                Reset::ClinicalStateUpdate {
                    key: "on_aspirin".to_string(),
                    value: true,
                },
            ],
            label: "administer_aspirin".to_string(),
            priority: 0,
        };

        assert!(edge.is_enabled(&cv, &concs, &flags));

        let mut cv2 = cv.clone();
        let mut concs2 = concs.clone();
        let mut flags2 = flags.clone();
        edge.apply_resets(&mut cv2, &mut concs2, &mut flags2);
        assert!((cv2.get(0)).abs() < 1e-10);
        assert!((concs2["aspirin"] - 100.0).abs() < 1e-10);
        assert_eq!(flags2["on_aspirin"], true);
    }

    // -- 6. PTA construction and structural queries -----------------------

    #[test]
    fn test_pta_construction() {
        let init = Location::initial(LocationId(0), "start");
        let mut pta = PTA::new("test_pta", init);

        let safe = Location::accepting(LocationId(1), "safe");
        let err = Location::error(LocationId(2), "violation");
        pta.add_location(safe);
        pta.add_location(err);

        pta.add_clock(ClockVariable::new(0, "t"));
        pta.concentration_vars.push("warfarin".to_string());
        pta.clinical_vars.push("bleeding_risk".to_string());

        pta.add_edge(Edge::new(
            LocationId(0),
            LocationId(1),
            Guard::True,
            "proceed",
        ));
        pta.add_edge(Edge::new(
            LocationId(0),
            LocationId(2),
            Guard::ClinicalGuard {
                predicate_key: "bleeding_risk".to_string(),
                expected: true,
            },
            "error_path",
        ));

        assert_eq!(pta.locations.len(), 3);
        assert_eq!(pta.edges.len(), 2);
        assert_eq!(pta.edges_from(LocationId(0)).len(), 2);
        assert_eq!(pta.edges_from(LocationId(1)).len(), 0);

        let reachable = pta.reachable_locations();
        assert_eq!(reachable.len(), 3);
    }

    // -- 7. Successor computation -----------------------------------------

    #[test]
    fn test_successor_states() {
        let init = Location::initial(LocationId(0), "start");
        let mut pta = PTA::new("succ_test", init);
        pta.add_location(Location::accepting(LocationId(1), "done"));
        pta.add_clock(ClockVariable::new(0, "t"));

        let mut edge = Edge::new(
            LocationId(0),
            LocationId(1),
            Guard::True,
            "go",
        );
        edge.resets.push(Reset::ClockReset { clock_id: 0 });
        pta.add_edge(edge);

        let mut state = pta.initial_state();
        state.clock_valuation.delay(3.0);
        let succs = pta.successor_states(&state);

        assert_eq!(succs.len(), 1);
        let (ref s, ref e) = succs[0];
        assert_eq!(s.location, LocationId(1));
        // Clock should have been reset
        assert!((s.clock_valuation.get(0)).abs() < 1e-10);
        assert_eq!(e.label, "go");
    }

    // -- 8. PTARun tracking -----------------------------------------------

    #[test]
    fn test_pta_run_tracking() {
        let init = Location::initial(LocationId(0), "start");
        let mut pta = PTA::new("run_test", init);
        let acc = Location::accepting(LocationId(1), "done");
        pta.add_location(acc);

        let state0 = pta.initial_state();
        let mut run = PTARun::new(state0.clone());
        assert_eq!(run.length(), 0);

        let edge = Edge::new(LocationId(0), LocationId(1), Guard::True, "step");
        let state1 = PTAState::new(
            LocationId(1),
            ClockValuation::new(),
            IndexMap::new(),
            IndexMap::new(),
        );
        run.push_step(edge, state1, 1.5);

        assert_eq!(run.length(), 1);
        assert!(run.is_accepting(&pta));
        assert!(!run.reaches_error(&pta));
        assert!((run.timestamps[1] - 1.5).abs() < 1e-10);

        let summary = run.trace_summary();
        assert!(summary.contains("step"));
    }

    // -- 9. Invariant checking --------------------------------------------

    #[test]
    fn test_invariant_checking() {
        let inv = Invariant::new(Guard::ClockGuard {
            clock_id: 0,
            op: GuardOperator::Le,
            bound: 24.0,
        });

        let cv_ok = make_clock_val(&[(0, 12.0)]);
        assert!(inv.is_satisfied(&cv_ok, &empty_concs(), &empty_flags()));

        let cv_bad = make_clock_val(&[(0, 30.0)]);
        assert!(!inv.is_satisfied(&cv_bad, &empty_concs(), &empty_flags()));
    }

    // -- 10. PTAState.satisfies_invariant via PTA -------------------------

    #[test]
    fn test_state_satisfies_invariant() {
        let mut loc = Location::initial(LocationId(0), "guarded");
        loc.invariant = Some(Invariant::new(Guard::ClockGuard {
            clock_id: 0,
            op: GuardOperator::Lt,
            bound: 10.0,
        }));
        let pta = PTA::new("inv_pta", loc);

        let state_ok = PTAState::new(
            LocationId(0),
            make_clock_val(&[(0, 5.0)]),
            IndexMap::new(),
            IndexMap::new(),
        );
        assert!(state_ok.satisfies_invariant(&pta));

        let state_bad = PTAState::new(
            LocationId(0),
            make_clock_val(&[(0, 15.0)]),
            IndexMap::new(),
            IndexMap::new(),
        );
        assert!(!state_bad.satisfies_invariant(&pta));
    }

    // -- 11. Serialization round-trip ------------------------------------

    #[test]
    fn test_serialization_roundtrip() {
        let init = Location::initial(LocationId(0), "start");
        let mut pta = PTA::new("serde_test", init);
        pta.add_location(Location::accepting(LocationId(1), "end"));
        pta.add_clock(ClockVariable::new(0, "timer"));
        pta.concentration_vars.push("drug_a".to_string());

        let mut edge = Edge::new(
            LocationId(0),
            LocationId(1),
            Guard::ClockGuard {
                clock_id: 0,
                op: GuardOperator::Ge,
                bound: 4.0,
            }
            .and(Guard::ConcentrationGuard {
                drug_id: DrugId::new("drug_a"),
                op: GuardOperator::Lt,
                bound: 10.0,
            }),
            "dose_then_proceed",
        );
        edge.resets.push(Reset::DoseAction {
            drug_id: "drug_a".to_string(),
            dose_mg: 50.0,
            route: "oral".to_string(),
        });
        pta.add_edge(edge);

        let json = serde_json::to_string_pretty(&pta).expect("serialize");
        let pta2: PTA = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(pta2.name, "serde_test");
        assert_eq!(pta2.locations.len(), 2);
        assert_eq!(pta2.edges.len(), 1);
        assert_eq!(pta2.clocks[0].name, "timer");
        assert_eq!(pta2.concentration_vars, vec!["drug_a".to_string()]);
    }

    // -- 12. Determinism check -------------------------------------------

    #[test]
    fn test_determinism_check() {
        let init = Location::initial(LocationId(0), "start");
        let mut pta_det = PTA::new("det", init.clone());
        pta_det.add_location(Location::new(LocationId(1), "a"));
        pta_det.add_location(Location::new(LocationId(2), "b"));

        // Different priorities → deterministic
        let mut e1 = Edge::new(LocationId(0), LocationId(1), Guard::True, "go_a");
        e1.priority = 0;
        let mut e2 = Edge::new(LocationId(0), LocationId(2), Guard::True, "go_b");
        e2.priority = 1;
        pta_det.add_edge(e1);
        pta_det.add_edge(e2);
        assert!(pta_det.is_deterministic());

        // Same priority, both non-false → non-deterministic
        let init2 = Location::initial(LocationId(0), "start");
        let mut pta_nondet = PTA::new("nondet", init2);
        pta_nondet.add_location(Location::new(LocationId(1), "a"));
        pta_nondet.add_location(Location::new(LocationId(2), "b"));

        let e3 = Edge::new(LocationId(0), LocationId(1), Guard::True, "go_a");
        let e4 = Edge::new(LocationId(0), LocationId(2), Guard::True, "go_b");
        pta_nondet.add_edge(e3);
        pta_nondet.add_edge(e4);
        assert!(!pta_nondet.is_deterministic());
    }

    // -- 13. LocationId Display and FromStr -------------------------------

    #[test]
    fn test_location_id_display_fromstr() {
        let id = LocationId(42);
        assert_eq!(id.to_string(), "L42");

        let parsed: LocationId = "L42".parse().unwrap();
        assert_eq!(parsed, id);

        let parsed_bare: LocationId = "7".parse().unwrap();
        assert_eq!(parsed_bare, LocationId(7));

        assert!("not_a_number".parse::<LocationId>().is_err());
    }

    // -- 14. GuardOperator evaluation coverage ----------------------------

    #[test]
    fn test_guard_operator_all_variants() {
        assert!(GuardOperator::Lt.evaluate(1.0, 2.0));
        assert!(!GuardOperator::Lt.evaluate(2.0, 2.0));

        assert!(GuardOperator::Le.evaluate(2.0, 2.0));
        assert!(!GuardOperator::Le.evaluate(3.0, 2.0));

        assert!(GuardOperator::Eq.evaluate(2.0, 2.0));
        assert!(!GuardOperator::Eq.evaluate(2.1, 2.0));

        assert!(GuardOperator::Ge.evaluate(2.0, 2.0));
        assert!(!GuardOperator::Ge.evaluate(1.0, 2.0));

        assert!(GuardOperator::Gt.evaluate(3.0, 2.0));
        assert!(!GuardOperator::Gt.evaluate(2.0, 2.0));
    }

    // -- 15. DoseAction additive semantics --------------------------------

    #[test]
    fn test_dose_action_additive() {
        let reset = Reset::DoseAction {
            drug_id: "metformin".to_string(),
            dose_mg: 500.0,
            route: "oral".to_string(),
        };

        let mut concs = IndexMap::new();
        concs.insert("metformin".to_string(), 200.0);
        reset.apply_to_concentrations(&mut concs);
        assert!((concs["metformin"] - 700.0).abs() < 1e-10);

        // From zero
        let mut concs2 = IndexMap::new();
        reset.apply_to_concentrations(&mut concs2);
        assert!((concs2["metformin"] - 500.0).abs() < 1e-10);
    }

    // -- 16. PTA reachable locations with disconnected node ---------------

    #[test]
    fn test_reachable_with_disconnected() {
        let init = Location::initial(LocationId(0), "start");
        let mut pta = PTA::new("reach", init);
        pta.add_location(Location::new(LocationId(1), "connected"));
        pta.add_location(Location::new(LocationId(2), "island"));
        pta.add_edge(Edge::new(
            LocationId(0),
            LocationId(1),
            Guard::True,
            "go",
        ));
        // LocationId(2) has no incoming edge from the initial location.

        let reachable = pta.reachable_locations();
        assert!(reachable.contains(&LocationId(0)));
        assert!(reachable.contains(&LocationId(1)));
        assert!(!reachable.contains(&LocationId(2)));
    }
}
