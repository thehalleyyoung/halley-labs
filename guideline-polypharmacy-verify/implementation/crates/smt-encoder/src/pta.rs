//! Local PTA (Pharmacological Timed Automaton) domain types.
//!
//! These types model the timed-automaton structure that the SMT encoder
//! converts into formulas for bounded model checking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Identifiers
// ═══════════════════════════════════════════════════════════════════════════

/// Unique identifier for a PTA location (control state).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct LocationId(pub String);

impl LocationId {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl fmt::Display for LocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for LocationId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Unique identifier for an edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct EdgeId(pub String);

impl EdgeId {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Variables
// ═══════════════════════════════════════════════════════════════════════════

/// A clock variable in the PTA (measures elapsed time).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClockVariable {
    pub name: String,
}

impl ClockVariable {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }
}

impl fmt::Display for ClockVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A concentration variable tracking a drug's plasma level.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConcentrationVariable {
    pub drug_name: String,
    pub name: String,
}

impl ConcentrationVariable {
    pub fn new(drug_name: &str) -> Self {
        let name = format!("conc_{}", drug_name.to_lowercase().replace(' ', "_"));
        Self { drug_name: drug_name.to_string(), name }
    }
}

impl fmt::Display for ConcentrationVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A generic state variable (boolean or integer valued).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateVariable {
    pub name: String,
    pub kind: StateVariableKind,
}

/// Kind of state variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateVariableKind {
    Bool,
    Int,
    Real,
}

impl StateVariable {
    pub fn bool_var(name: &str) -> Self {
        Self { name: name.to_string(), kind: StateVariableKind::Bool }
    }

    pub fn int_var(name: &str) -> Self {
        Self { name: name.to_string(), kind: StateVariableKind::Int }
    }

    pub fn real_var(name: &str) -> Self {
        Self { name: name.to_string(), kind: StateVariableKind::Real }
    }
}

impl fmt::Display for StateVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Guard
// ═══════════════════════════════════════════════════════════════════════════

/// Comparison operator for guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GuardOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
    Ne,
}

impl fmt::Display for GuardOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuardOp::Lt => write!(f, "<"),
            GuardOp::Le => write!(f, "<="),
            GuardOp::Eq => write!(f, "=="),
            GuardOp::Ge => write!(f, ">="),
            GuardOp::Gt => write!(f, ">"),
            GuardOp::Ne => write!(f, "!="),
        }
    }
}

/// An atomic guard on a PTA edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Guard {
    /// Guard on a clock variable: `clock op value`.
    Clock {
        clock: ClockVariable,
        op: GuardOp,
        value: f64,
    },
    /// Guard on a concentration variable: `conc op threshold`.
    Concentration {
        variable: ConcentrationVariable,
        op: GuardOp,
        threshold: f64,
    },
    /// Guard on a generic state variable.
    State {
        variable: StateVariable,
        op: GuardOp,
        value: f64,
    },
    /// Boolean state variable guard.
    BoolState {
        variable: StateVariable,
        expected: bool,
    },
    /// Compound guard (conjunction/disjunction/negation).
    Compound(CompoundGuard),
    /// Always true.
    True,
}

/// Compound guard combining atomic guards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompoundGuard {
    And(Vec<Guard>),
    Or(Vec<Guard>),
    Not(Box<Guard>),
}

impl Guard {
    pub fn clock_lt(clock: &ClockVariable, value: f64) -> Self {
        Guard::Clock { clock: clock.clone(), op: GuardOp::Lt, value }
    }

    pub fn clock_le(clock: &ClockVariable, value: f64) -> Self {
        Guard::Clock { clock: clock.clone(), op: GuardOp::Le, value }
    }

    pub fn clock_ge(clock: &ClockVariable, value: f64) -> Self {
        Guard::Clock { clock: clock.clone(), op: GuardOp::Ge, value }
    }

    pub fn clock_gt(clock: &ClockVariable, value: f64) -> Self {
        Guard::Clock { clock: clock.clone(), op: GuardOp::Gt, value }
    }

    pub fn clock_eq(clock: &ClockVariable, value: f64) -> Self {
        Guard::Clock { clock: clock.clone(), op: GuardOp::Eq, value }
    }

    pub fn conc_lt(var: &ConcentrationVariable, threshold: f64) -> Self {
        Guard::Concentration { variable: var.clone(), op: GuardOp::Lt, threshold }
    }

    pub fn conc_ge(var: &ConcentrationVariable, threshold: f64) -> Self {
        Guard::Concentration { variable: var.clone(), op: GuardOp::Ge, threshold }
    }

    pub fn conc_gt(var: &ConcentrationVariable, threshold: f64) -> Self {
        Guard::Concentration { variable: var.clone(), op: GuardOp::Gt, threshold }
    }

    pub fn conc_le(var: &ConcentrationVariable, threshold: f64) -> Self {
        Guard::Concentration { variable: var.clone(), op: GuardOp::Le, threshold }
    }

    pub fn state_eq(var: &StateVariable, value: f64) -> Self {
        Guard::State { variable: var.clone(), op: GuardOp::Eq, value }
    }

    pub fn bool_guard(var: &StateVariable, expected: bool) -> Self {
        Guard::BoolState { variable: var.clone(), expected }
    }

    pub fn and(guards: Vec<Guard>) -> Self {
        Guard::Compound(CompoundGuard::And(guards))
    }

    pub fn or(guards: Vec<Guard>) -> Self {
        Guard::Compound(CompoundGuard::Or(guards))
    }

    pub fn not(guard: Guard) -> Self {
        Guard::Compound(CompoundGuard::Not(Box::new(guard)))
    }

    pub fn is_trivially_true(&self) -> bool {
        matches!(self, Guard::True)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reset
// ═══════════════════════════════════════════════════════════════════════════

/// An action performed when an edge is taken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResetAction {
    /// Reset a clock to zero.
    ClockReset(ClockVariable),
    /// Set a concentration to a specific value.
    SetConcentration { variable: ConcentrationVariable, value: f64 },
    /// Add a dose to concentration.
    AddDose { variable: ConcentrationVariable, dose_mg: f64, bioavailability: f64 },
    /// Set a state variable.
    SetState { variable: StateVariable, value: f64 },
    /// Set a boolean state variable.
    SetBool { variable: StateVariable, value: bool },
}

/// Collection of reset actions for an edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reset {
    pub actions: Vec<ResetAction>,
}

impl Reset {
    pub fn new() -> Self {
        Self { actions: Vec::new() }
    }

    pub fn with_action(mut self, action: ResetAction) -> Self {
        self.actions.push(action);
        self
    }

    pub fn clock_reset(mut self, clock: &ClockVariable) -> Self {
        self.actions.push(ResetAction::ClockReset(clock.clone()));
        self
    }

    pub fn add_dose(mut self, var: &ConcentrationVariable, dose_mg: f64, bioavailability: f64) -> Self {
        self.actions.push(ResetAction::AddDose {
            variable: var.clone(),
            dose_mg,
            bioavailability,
        });
        self
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Return the set of clock variables that are reset.
    pub fn reset_clocks(&self) -> Vec<&ClockVariable> {
        self.actions.iter().filter_map(|a| match a {
            ResetAction::ClockReset(c) => Some(c),
            _ => None,
        }).collect()
    }
}

impl Default for Reset {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Invariant
// ═══════════════════════════════════════════════════════════════════════════

/// A single invariant clause for a location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvariantClause {
    /// Clock upper bound: `clock <= value`.
    ClockBound { clock: ClockVariable, bound: f64 },
    /// Concentration within range.
    ConcentrationRange {
        variable: ConcentrationVariable,
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// State condition.
    StateCondition { variable: StateVariable, op: GuardOp, value: f64 },
    /// Boolean condition.
    BoolCondition { variable: StateVariable, expected: bool },
}

/// Location invariant (conjunction of clauses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    pub clauses: Vec<InvariantClause>,
}

impl Invariant {
    pub fn new() -> Self {
        Self { clauses: Vec::new() }
    }

    pub fn with_clause(mut self, clause: InvariantClause) -> Self {
        self.clauses.push(clause);
        self
    }

    pub fn clock_bound(mut self, clock: &ClockVariable, bound: f64) -> Self {
        self.clauses.push(InvariantClause::ClockBound {
            clock: clock.clone(),
            bound,
        });
        self
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }
}

impl Default for Invariant {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Location
// ═══════════════════════════════════════════════════════════════════════════

/// A location (control state) in the PTA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub id: LocationId,
    pub name: String,
    pub invariant: Invariant,
    pub is_initial: bool,
    pub is_urgent: bool,
    pub labels: HashMap<String, String>,
}

impl Location {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: LocationId::new(id),
            name: name.to_string(),
            invariant: Invariant::new(),
            is_initial: false,
            is_urgent: false,
            labels: HashMap::new(),
        }
    }

    pub fn initial(mut self) -> Self {
        self.is_initial = true;
        self
    }

    pub fn urgent(mut self) -> Self {
        self.is_urgent = true;
        self
    }

    pub fn with_invariant(mut self, invariant: Invariant) -> Self {
        self.invariant = invariant;
        self
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge
// ═══════════════════════════════════════════════════════════════════════════

/// An edge in the PTA connecting two locations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: LocationId,
    pub target: LocationId,
    pub guard: Guard,
    pub reset: Reset,
    pub label: Option<String>,
    pub priority: u32,
}

impl Edge {
    pub fn new(id: &str, source: &LocationId, target: &LocationId) -> Self {
        Self {
            id: EdgeId::new(id),
            source: source.clone(),
            target: target.clone(),
            guard: Guard::True,
            reset: Reset::new(),
            label: None,
            priority: 0,
        }
    }

    pub fn with_guard(mut self, guard: Guard) -> Self {
        self.guard = guard;
        self
    }

    pub fn with_reset(mut self, reset: Reset) -> Self {
        self.reset = reset;
        self
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Safety Property
// ═══════════════════════════════════════════════════════════════════════════

/// Condition in a safety property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyCondition {
    /// Concentration must stay within bounds.
    ConcentrationBound {
        variable: ConcentrationVariable,
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// Must never be in a specific location.
    ForbiddenLocation(LocationId),
    /// Clock must not exceed a bound.
    ClockBound { clock: ClockVariable, bound: f64 },
    /// State variable constraint.
    StateConstraint { variable: StateVariable, op: GuardOp, value: f64 },
    /// Boolean flag must hold.
    BoolMustHold { variable: StateVariable, expected: bool },
    /// Conjunction of conditions.
    And(Vec<SafetyCondition>),
    /// Disjunction of conditions.
    Or(Vec<SafetyCondition>),
    /// Negation.
    Not(Box<SafetyCondition>),
    /// Implication: if A then B.
    Implies(Box<SafetyCondition>, Box<SafetyCondition>),
}

/// A safety property that must hold at every reachable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyProperty {
    pub name: String,
    pub description: String,
    pub condition: SafetyCondition,
}

impl SafetyProperty {
    pub fn new(name: &str, description: &str, condition: SafetyCondition) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            condition,
        }
    }

    pub fn concentration_within(
        name: &str,
        var: &ConcentrationVariable,
        lower: f64,
        upper: f64,
    ) -> Self {
        Self::new(
            name,
            &format!("{} in [{}, {}]", var.name, lower, upper),
            SafetyCondition::ConcentrationBound {
                variable: var.clone(),
                lower: Some(lower),
                upper: Some(upper),
            },
        )
    }

    pub fn forbidden_location(name: &str, loc: &LocationId) -> Self {
        Self::new(
            name,
            &format!("never reach {}", loc),
            SafetyCondition::ForbiddenLocation(loc.clone()),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PTA
// ═══════════════════════════════════════════════════════════════════════════

/// Complete Pharmacological Timed Automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PTA {
    pub name: String,
    pub locations: Vec<Location>,
    pub edges: Vec<Edge>,
    pub clocks: Vec<ClockVariable>,
    pub concentration_vars: Vec<ConcentrationVariable>,
    pub state_vars: Vec<StateVariable>,
    pub initial_location: LocationId,
    pub initial_concentrations: HashMap<String, f64>,
    pub safety_properties: Vec<SafetyProperty>,
    pub time_step: f64,
}

impl PTA {
    pub fn new(name: &str, initial_location: &LocationId) -> Self {
        Self {
            name: name.to_string(),
            locations: Vec::new(),
            edges: Vec::new(),
            clocks: Vec::new(),
            concentration_vars: Vec::new(),
            state_vars: Vec::new(),
            initial_location: initial_location.clone(),
            initial_concentrations: HashMap::new(),
            safety_properties: Vec::new(),
            time_step: 1.0,
        }
    }

    pub fn location_by_id(&self, id: &LocationId) -> Option<&Location> {
        self.locations.iter().find(|l| &l.id == id)
    }

    pub fn edges_from(&self, loc: &LocationId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| &e.source == loc).collect()
    }

    pub fn edges_to(&self, loc: &LocationId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| &e.target == loc).collect()
    }

    pub fn location_ids(&self) -> Vec<&LocationId> {
        self.locations.iter().map(|l| &l.id).collect()
    }

    pub fn num_locations(&self) -> usize {
        self.locations.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_clocks(&self) -> usize {
        self.clocks.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PTA Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for constructing PTAs incrementally.
pub struct PTABuilder {
    pta: PTA,
    edge_counter: usize,
}

impl PTABuilder {
    pub fn new(name: &str, initial_location: &str) -> Self {
        let initial_loc_id = LocationId::new(initial_location);
        let mut pta = PTA::new(name, &initial_loc_id);
        pta.locations.push(
            Location::new(initial_location, initial_location).initial(),
        );
        Self { pta, edge_counter: 0 }
    }

    pub fn add_location(mut self, id: &str, name: &str) -> Self {
        self.pta.locations.push(Location::new(id, name));
        self
    }

    pub fn add_location_with_invariant(mut self, id: &str, name: &str, invariant: Invariant) -> Self {
        self.pta.locations.push(Location::new(id, name).with_invariant(invariant));
        self
    }

    pub fn add_clock(mut self, name: &str) -> Self {
        self.pta.clocks.push(ClockVariable::new(name));
        self
    }

    pub fn add_concentration_var(mut self, drug_name: &str) -> Self {
        self.pta.concentration_vars.push(ConcentrationVariable::new(drug_name));
        self
    }

    pub fn add_state_var(mut self, var: StateVariable) -> Self {
        self.pta.state_vars.push(var);
        self
    }

    pub fn set_initial_concentration(mut self, drug_name: &str, value: f64) -> Self {
        self.pta.initial_concentrations.insert(drug_name.to_string(), value);
        self
    }

    pub fn add_edge(mut self, source: &str, target: &str, guard: Guard, reset: Reset) -> Self {
        let edge_id = format!("e{}", self.edge_counter);
        self.edge_counter += 1;
        self.pta.edges.push(Edge {
            id: EdgeId::new(&edge_id),
            source: LocationId::new(source),
            target: LocationId::new(target),
            guard,
            reset,
            label: None,
            priority: 0,
        });
        self
    }

    pub fn add_labeled_edge(
        mut self,
        source: &str,
        target: &str,
        guard: Guard,
        reset: Reset,
        label: &str,
    ) -> Self {
        let edge_id = format!("e{}", self.edge_counter);
        self.edge_counter += 1;
        self.pta.edges.push(Edge {
            id: EdgeId::new(&edge_id),
            source: LocationId::new(source),
            target: LocationId::new(target),
            guard,
            reset,
            label: Some(label.to_string()),
            priority: 0,
        });
        self
    }

    pub fn add_safety_property(mut self, prop: SafetyProperty) -> Self {
        self.pta.safety_properties.push(prop);
        self
    }

    pub fn set_time_step(mut self, dt: f64) -> Self {
        self.pta.time_step = dt;
        self
    }

    pub fn build(self) -> PTA {
        self.pta
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pta() -> PTA {
        let clock = ClockVariable::new("x");
        let conc = ConcentrationVariable::new("warfarin");

        PTABuilder::new("test_pta", "l0")
            .add_location("l1", "dosing")
            .add_location("l2", "monitoring")
            .add_clock("x")
            .add_concentration_var("warfarin")
            .set_initial_concentration("warfarin", 0.0)
            .add_edge(
                "l0", "l1",
                Guard::clock_ge(&clock, 0.0),
                Reset::new().clock_reset(&clock)
                    .add_dose(&conc, 5.0, 0.9),
            )
            .add_edge(
                "l1", "l2",
                Guard::clock_ge(&clock, 1.0),
                Reset::new().clock_reset(&clock),
            )
            .add_edge(
                "l2", "l0",
                Guard::clock_ge(&clock, 8.0),
                Reset::new().clock_reset(&clock),
            )
            .add_safety_property(SafetyProperty::concentration_within(
                "therapeutic_range",
                &conc,
                1.0, 5.0,
            ))
            .set_time_step(0.5)
            .build()
    }

    #[test]
    fn test_pta_builder() {
        let pta = sample_pta();
        assert_eq!(pta.num_locations(), 3);
        assert_eq!(pta.num_edges(), 3);
        assert_eq!(pta.num_clocks(), 1);
    }

    #[test]
    fn test_location_lookup() {
        let pta = sample_pta();
        let loc = pta.location_by_id(&LocationId::new("l0")).unwrap();
        assert!(loc.is_initial);
    }

    #[test]
    fn test_edges_from() {
        let pta = sample_pta();
        let edges = pta.edges_from(&LocationId::new("l0"));
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_edges_to() {
        let pta = sample_pta();
        let edges = pta.edges_to(&LocationId::new("l1"));
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_guard_constructors() {
        let clock = ClockVariable::new("x");
        let g = Guard::clock_lt(&clock, 5.0);
        assert!(!g.is_trivially_true());
        assert!(Guard::True.is_trivially_true());
    }

    #[test]
    fn test_compound_guard() {
        let clock = ClockVariable::new("x");
        let g = Guard::and(vec![
            Guard::clock_ge(&clock, 1.0),
            Guard::clock_le(&clock, 5.0),
        ]);
        assert!(!g.is_trivially_true());
    }

    #[test]
    fn test_reset_clocks() {
        let c1 = ClockVariable::new("x");
        let c2 = ClockVariable::new("y");
        let reset = Reset::new()
            .clock_reset(&c1)
            .clock_reset(&c2);
        assert_eq!(reset.reset_clocks().len(), 2);
    }

    #[test]
    fn test_invariant_builder() {
        let clock = ClockVariable::new("x");
        let inv = Invariant::new()
            .clock_bound(&clock, 10.0)
            .with_clause(InvariantClause::ConcentrationRange {
                variable: ConcentrationVariable::new("aspirin"),
                lower: Some(0.0),
                upper: Some(100.0),
            });
        assert_eq!(inv.clauses.len(), 2);
    }

    #[test]
    fn test_safety_property() {
        let conc = ConcentrationVariable::new("warfarin");
        let prop = SafetyProperty::concentration_within("safe", &conc, 1.0, 5.0);
        assert_eq!(prop.name, "safe");
    }

    #[test]
    fn test_location_labels() {
        let loc = Location::new("l0", "init")
            .with_label("type", "initial")
            .with_label("drug", "warfarin");
        assert_eq!(loc.labels.get("type").unwrap(), "initial");
    }
}
