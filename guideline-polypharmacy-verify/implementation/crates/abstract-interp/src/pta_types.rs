//! Pharmacological Timed Automata (PTA) type definitions.
//!
//! These types model clinical guidelines as timed automata where locations
//! represent clinical states, edges represent medication actions, guards
//! constrain transitions, and clocks track time.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use guardpharma_types::{DrugId, DosingSchedule, CypEnzyme};

// ---------------------------------------------------------------------------
// Location
// ---------------------------------------------------------------------------

/// Unique identifier for a PTA location.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct LocationId(pub String);

impl LocationId {
    pub fn new(name: impl Into<String>) -> Self {
        LocationId(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for LocationId {
    fn from(s: &str) -> Self {
        LocationId::new(s)
    }
}

/// A clock variable in a timed automaton.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClockVariable(pub String);

impl ClockVariable {
    pub fn new(name: impl Into<String>) -> Self {
        ClockVariable(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ClockVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A location (state) in the PTA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub id: LocationId,
    pub name: String,
    pub invariant: Option<Guard>,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub is_urgent: bool,
}

impl Location {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let id_str = id.into();
        Self {
            id: LocationId::new(id_str),
            name: name.into(),
            invariant: None,
            is_initial: false,
            is_accepting: false,
            is_urgent: false,
        }
    }

    pub fn initial(id: impl Into<String>, name: impl Into<String>) -> Self {
        let mut loc = Self::new(id, name);
        loc.is_initial = true;
        loc
    }

    pub fn with_invariant(mut self, guard: Guard) -> Self {
        self.invariant = Some(guard);
        self
    }

    pub fn with_accepting(mut self) -> Self {
        self.is_accepting = true;
        self
    }

    pub fn with_urgent(mut self) -> Self {
        self.is_urgent = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Comparison Operators
// ---------------------------------------------------------------------------

/// Comparison operator for guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Lt,
    Le,
    Ge,
    Gt,
    Eq,
    Ne,
}

impl ComparisonOp {
    pub fn evaluate(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            ComparisonOp::Lt => lhs < rhs,
            ComparisonOp::Le => lhs <= rhs,
            ComparisonOp::Ge => lhs >= rhs,
            ComparisonOp::Gt => lhs > rhs,
            ComparisonOp::Eq => (lhs - rhs).abs() < f64::EPSILON,
            ComparisonOp::Ne => (lhs - rhs).abs() >= f64::EPSILON,
        }
    }

    pub fn negate(&self) -> Self {
        match self {
            ComparisonOp::Lt => ComparisonOp::Ge,
            ComparisonOp::Le => ComparisonOp::Gt,
            ComparisonOp::Ge => ComparisonOp::Lt,
            ComparisonOp::Gt => ComparisonOp::Le,
            ComparisonOp::Eq => ComparisonOp::Ne,
            ComparisonOp::Ne => ComparisonOp::Eq,
        }
    }
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonOp::Lt => write!(f, "<"),
            ComparisonOp::Le => write!(f, "≤"),
            ComparisonOp::Ge => write!(f, "≥"),
            ComparisonOp::Gt => write!(f, ">"),
            ComparisonOp::Eq => write!(f, "="),
            ComparisonOp::Ne => write!(f, "≠"),
        }
    }
}

// ---------------------------------------------------------------------------
// Guard
// ---------------------------------------------------------------------------

/// A guard condition on a PTA transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guard {
    pub kind: GuardKind,
}

impl Guard {
    pub fn new(kind: GuardKind) -> Self {
        Self { kind }
    }

    pub fn true_guard() -> Self {
        Self { kind: GuardKind::True }
    }

    pub fn false_guard() -> Self {
        Self { kind: GuardKind::False }
    }

    pub fn clock(clock: ClockVariable, op: ComparisonOp, value: f64) -> Self {
        Self {
            kind: GuardKind::ClockGuard { clock, op, value },
        }
    }

    pub fn concentration(drug: DrugId, op: ComparisonOp, value: f64) -> Self {
        Self {
            kind: GuardKind::ConcentrationGuard { drug, op, value },
        }
    }

    pub fn enzyme(enzyme: CypEnzyme, op: ComparisonOp, value: f64) -> Self {
        Self {
            kind: GuardKind::EnzymeGuard { enzyme, op, value },
        }
    }

    pub fn and(left: Guard, right: Guard) -> Self {
        Self {
            kind: GuardKind::And(Box::new(left), Box::new(right)),
        }
    }

    pub fn or(left: Guard, right: Guard) -> Self {
        Self {
            kind: GuardKind::Or(Box::new(left), Box::new(right)),
        }
    }

    pub fn not(inner: Guard) -> Self {
        Self {
            kind: GuardKind::Not(Box::new(inner)),
        }
    }

    pub fn is_true(&self) -> bool {
        matches!(self.kind, GuardKind::True)
    }

    pub fn is_false(&self) -> bool {
        matches!(self.kind, GuardKind::False)
    }

    /// Collect all clock variables referenced in this guard.
    pub fn referenced_clocks(&self) -> HashSet<ClockVariable> {
        let mut clocks = HashSet::new();
        self.collect_clocks(&mut clocks);
        clocks
    }

    fn collect_clocks(&self, clocks: &mut HashSet<ClockVariable>) {
        match &self.kind {
            GuardKind::ClockGuard { clock, .. } => {
                clocks.insert(clock.clone());
            }
            GuardKind::And(l, r) | GuardKind::Or(l, r) => {
                l.collect_clocks(clocks);
                r.collect_clocks(clocks);
            }
            GuardKind::Not(inner) => {
                inner.collect_clocks(clocks);
            }
            _ => {}
        }
    }

    /// Collect all drug IDs referenced in this guard.
    pub fn referenced_drugs(&self) -> HashSet<DrugId> {
        let mut drugs = HashSet::new();
        self.collect_drugs(&mut drugs);
        drugs
    }

    fn collect_drugs(&self, drugs: &mut HashSet<DrugId>) {
        match &self.kind {
            GuardKind::ConcentrationGuard { drug, .. } => {
                drugs.insert(drug.clone());
            }
            GuardKind::And(l, r) | GuardKind::Or(l, r) => {
                l.collect_drugs(drugs);
                r.collect_drugs(drugs);
            }
            GuardKind::Not(inner) => {
                inner.collect_drugs(drugs);
            }
            _ => {}
        }
    }
}

/// The kind of guard condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuardKind {
    /// Always true.
    True,
    /// Always false.
    False,
    /// Clock comparison: clock op value.
    ClockGuard {
        clock: ClockVariable,
        op: ComparisonOp,
        value: f64,
    },
    /// Drug concentration comparison.
    ConcentrationGuard {
        drug: DrugId,
        op: ComparisonOp,
        value: f64,
    },
    /// Enzyme activity comparison.
    EnzymeGuard {
        enzyme: CypEnzyme,
        op: ComparisonOp,
        value: f64,
    },
    /// Conjunction.
    And(Box<Guard>, Box<Guard>),
    /// Disjunction.
    Or(Box<Guard>, Box<Guard>),
    /// Negation.
    Not(Box<Guard>),
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// An action on a PTA transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub kind: ActionKind,
}

impl Action {
    pub fn new(kind: ActionKind) -> Self {
        Self { kind }
    }

    pub fn noop() -> Self {
        Self { kind: ActionKind::Noop }
    }

    pub fn start_medication(drug: DrugId, schedule: DosingSchedule) -> Self {
        Self {
            kind: ActionKind::StartMedication { drug, schedule },
        }
    }

    pub fn stop_medication(drug: DrugId) -> Self {
        Self {
            kind: ActionKind::StopMedication { drug },
        }
    }

    pub fn adjust_dose(drug: DrugId, new_dose_mg: f64) -> Self {
        Self {
            kind: ActionKind::AdjustDose { drug, new_dose_mg },
        }
    }

    pub fn time_elapse(hours: f64) -> Self {
        Self {
            kind: ActionKind::TimeElapse { hours },
        }
    }

    pub fn lab_result(test_name: impl Into<String>, value: f64) -> Self {
        Self {
            kind: ActionKind::LabResult {
                test_name: test_name.into(),
                value,
            },
        }
    }

    pub fn is_noop(&self) -> bool {
        matches!(self.kind, ActionKind::Noop)
    }

    /// Collect all drug IDs referenced in this action.
    pub fn referenced_drugs(&self) -> HashSet<DrugId> {
        let mut drugs = HashSet::new();
        match &self.kind {
            ActionKind::StartMedication { drug, .. }
            | ActionKind::StopMedication { drug }
            | ActionKind::AdjustDose { drug, .. } => {
                drugs.insert(drug.clone());
            }
            _ => {}
        }
        drugs
    }
}

/// The kind of action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionKind {
    /// No operation.
    Noop,
    /// Start a medication with a dosing schedule.
    StartMedication {
        drug: DrugId,
        schedule: DosingSchedule,
    },
    /// Stop a medication.
    StopMedication {
        drug: DrugId,
    },
    /// Adjust the dose of an existing medication.
    AdjustDose {
        drug: DrugId,
        new_dose_mg: f64,
    },
    /// Let time elapse (for PK evolution).
    TimeElapse {
        hours: f64,
    },
    /// Incorporate a lab result into the clinical state.
    LabResult {
        test_name: String,
        value: f64,
    },
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// A transition in the PTA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: LocationId,
    pub target: LocationId,
    pub guard: Guard,
    pub action: Action,
    pub clock_resets: Vec<ClockVariable>,
    pub priority: i32,
    pub label: Option<String>,
}

impl Edge {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        guard: Guard,
        action: Action,
    ) -> Self {
        Self {
            source: LocationId::new(source),
            target: LocationId::new(target),
            guard,
            action,
            clock_resets: Vec::new(),
            priority: 0,
            label: None,
        }
    }

    pub fn with_resets(mut self, resets: Vec<ClockVariable>) -> Self {
        self.clock_resets = resets;
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// True if this edge resets any clock.
    pub fn has_resets(&self) -> bool {
        !self.clock_resets.is_empty()
    }
}

// ---------------------------------------------------------------------------
// PTA (Pharmacological Timed Automaton)
// ---------------------------------------------------------------------------

/// A Pharmacological Timed Automaton representing a clinical guideline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PTA {
    pub name: String,
    pub locations: Vec<Location>,
    pub edges: Vec<Edge>,
    pub initial_location: LocationId,
    pub clocks: Vec<ClockVariable>,
    pub drugs: Vec<DrugId>,
}

impl PTA {
    pub fn new(name: impl Into<String>, initial: impl Into<String>) -> Self {
        let init_id = initial.into();
        Self {
            name: name.into(),
            locations: Vec::new(),
            edges: Vec::new(),
            initial_location: LocationId::new(&init_id),
            clocks: Vec::new(),
            drugs: Vec::new(),
        }
    }

    pub fn add_location(&mut self, location: Location) {
        self.locations.push(location);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    pub fn add_clock(&mut self, clock: ClockVariable) {
        self.clocks.push(clock);
    }

    pub fn add_drug(&mut self, drug: DrugId) {
        self.drugs.push(drug);
    }

    pub fn location_ids(&self) -> Vec<&LocationId> {
        self.locations.iter().map(|l| &l.id).collect()
    }

    pub fn get_location(&self, id: &LocationId) -> Option<&Location> {
        self.locations.iter().find(|l| l.id == *id)
    }

    pub fn outgoing_edges(&self, location: &LocationId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.source == *location).collect()
    }

    pub fn incoming_edges(&self, location: &LocationId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.target == *location).collect()
    }

    pub fn successor_locations(&self, location: &LocationId) -> Vec<&LocationId> {
        self.outgoing_edges(location)
            .iter()
            .map(|e| &e.target)
            .collect()
    }

    pub fn predecessor_locations(&self, location: &LocationId) -> Vec<&LocationId> {
        self.incoming_edges(location)
            .iter()
            .map(|e| &e.source)
            .collect()
    }

    pub fn all_referenced_drugs(&self) -> HashSet<DrugId> {
        let mut drugs: HashSet<DrugId> = self.drugs.iter().cloned().collect();
        for edge in &self.edges {
            drugs.extend(edge.guard.referenced_drugs());
            drugs.extend(edge.action.referenced_drugs());
        }
        drugs
    }

    pub fn location_count(&self) -> usize {
        self.locations.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Construct a simple two-location PTA for a single drug.
    pub fn single_drug(drug: DrugId, schedule: DosingSchedule) -> Self {
        let mut pta = PTA::new(format!("pta_{}", drug), "idle");
        let clock = ClockVariable::new("t");
        pta.add_clock(clock.clone());
        pta.add_drug(drug.clone());

        pta.add_location(Location::initial("idle", "Idle"));
        pta.add_location(Location::new("active", "Active medication"));

        pta.add_edge(
            Edge::new("idle", "active", Guard::true_guard(),
                      Action::start_medication(drug.clone(), schedule))
                .with_resets(vec![clock.clone()]),
        );

        pta.add_edge(
            Edge::new("active", "idle", Guard::true_guard(),
                      Action::stop_medication(drug)),
        );

        pta
    }

    /// Compose two PTAs into a product PTA.
    pub fn compose(a: &PTA, b: &PTA) -> PTA {
        let mut composed = PTA::new(
            format!("{}×{}", a.name, b.name),
            format!("{}_{}", a.initial_location, b.initial_location),
        );

        // Product locations
        for la in &a.locations {
            for lb in &b.locations {
                let pid = format!("{}_{}", la.id, lb.id);
                let pname = format!("{}×{}", la.name, lb.name);
                let mut loc = Location::new(&pid, pname);
                if la.is_initial && lb.is_initial {
                    loc.is_initial = true;
                }
                composed.add_location(loc);
            }
        }

        // Interleaving edges from A
        for ea in &a.edges {
            for lb in &b.locations {
                let source = format!("{}_{}", ea.source, lb.id);
                let target = format!("{}_{}", ea.target, lb.id);
                composed.add_edge(Edge::new(source, target, ea.guard.clone(), ea.action.clone())
                    .with_resets(ea.clock_resets.clone()));
            }
        }

        // Interleaving edges from B
        for eb in &b.edges {
            for la in &a.locations {
                let source = format!("{}_{}", la.id, eb.source);
                let target = format!("{}_{}", la.id, eb.target);
                composed.add_edge(Edge::new(source, target, eb.guard.clone(), eb.action.clone())
                    .with_resets(eb.clock_resets.clone()));
            }
        }

        // Merge clocks and drugs
        composed.clocks = a.clocks.iter().chain(b.clocks.iter()).cloned().collect();
        composed.drugs = a.drugs.iter().chain(b.drugs.iter()).cloned().collect();

        composed
    }
}

impl fmt::Display for PTA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PTA({}, {} locations, {} edges)",
               self.name, self.locations.len(), self.edges.len())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_id() {
        let id = LocationId::new("start");
        assert_eq!(id.as_str(), "start");
        assert_eq!(id.to_string(), "start");
    }

    #[test]
    fn test_comparison_op_evaluate() {
        assert!(ComparisonOp::Lt.evaluate(1.0, 2.0));
        assert!(!ComparisonOp::Lt.evaluate(2.0, 1.0));
        assert!(ComparisonOp::Le.evaluate(2.0, 2.0));
        assert!(ComparisonOp::Ge.evaluate(3.0, 2.0));
        assert!(ComparisonOp::Gt.evaluate(3.0, 2.0));
        assert!(ComparisonOp::Eq.evaluate(2.0, 2.0));
        assert!(ComparisonOp::Ne.evaluate(1.0, 2.0));
    }

    #[test]
    fn test_comparison_op_negate() {
        assert_eq!(ComparisonOp::Lt.negate(), ComparisonOp::Ge);
        assert_eq!(ComparisonOp::Le.negate(), ComparisonOp::Gt);
        assert_eq!(ComparisonOp::Ge.negate(), ComparisonOp::Lt);
        assert_eq!(ComparisonOp::Gt.negate(), ComparisonOp::Le);
    }

    #[test]
    fn test_guard_construction() {
        let g = Guard::true_guard();
        assert!(g.is_true());
        let g = Guard::false_guard();
        assert!(g.is_false());
    }

    #[test]
    fn test_guard_clock() {
        let clock = ClockVariable::new("t");
        let g = Guard::clock(clock.clone(), ComparisonOp::Le, 24.0);
        let clocks = g.referenced_clocks();
        assert!(clocks.contains(&clock));
    }

    #[test]
    fn test_guard_and_or() {
        let g1 = Guard::clock(ClockVariable::new("t1"), ComparisonOp::Ge, 0.0);
        let g2 = Guard::clock(ClockVariable::new("t2"), ComparisonOp::Le, 10.0);
        let and_g = Guard::and(g1.clone(), g2.clone());
        let clocks = and_g.referenced_clocks();
        assert_eq!(clocks.len(), 2);
        let or_g = Guard::or(g1, g2);
        assert_eq!(or_g.referenced_clocks().len(), 2);
    }

    #[test]
    fn test_guard_referenced_drugs() {
        let g = Guard::concentration(DrugId::new("warfarin"), ComparisonOp::Le, 4.0);
        let drugs = g.referenced_drugs();
        assert!(drugs.contains(&DrugId::new("warfarin")));
    }

    #[test]
    fn test_action_construction() {
        let a = Action::noop();
        assert!(a.is_noop());
        let a = Action::start_medication(DrugId::new("test"), DosingSchedule::new(5.0, 24.0));
        assert!(!a.is_noop());
        let drugs = a.referenced_drugs();
        assert!(drugs.contains(&DrugId::new("test")));
    }

    #[test]
    fn test_edge_construction() {
        let e = Edge::new("s1", "s2", Guard::true_guard(), Action::noop())
            .with_priority(1)
            .with_label("test_edge");
        assert_eq!(e.source, LocationId::new("s1"));
        assert_eq!(e.target, LocationId::new("s2"));
        assert_eq!(e.priority, 1);
        assert_eq!(e.label.as_deref(), Some("test_edge"));
    }

    #[test]
    fn test_pta_single_drug() {
        let pta = PTA::single_drug(DrugId::new("warfarin"), DosingSchedule::new(5.0, 24.0));
        assert_eq!(pta.location_count(), 2);
        assert_eq!(pta.edge_count(), 2);
        assert!(pta.all_referenced_drugs().contains(&DrugId::new("warfarin")));
    }

    #[test]
    fn test_pta_compose() {
        let a = PTA::single_drug(DrugId::new("drug_a"), DosingSchedule::new(5.0, 24.0));
        let b = PTA::single_drug(DrugId::new("drug_b"), DosingSchedule::new(10.0, 12.0));
        let composed = PTA::compose(&a, &b);
        assert_eq!(composed.location_count(), 4); // 2 * 2
        assert_eq!(composed.edge_count(), 8); // 2*2 + 2*2
    }

    #[test]
    fn test_pta_navigation() {
        let pta = PTA::single_drug(DrugId::new("test"), DosingSchedule::new(5.0, 24.0));
        let initial = &pta.initial_location;
        let outgoing = pta.outgoing_edges(initial);
        assert_eq!(outgoing.len(), 1);
        let successors = pta.successor_locations(initial);
        assert_eq!(successors.len(), 1);
    }

    #[test]
    fn test_pta_display() {
        let pta = PTA::single_drug(DrugId::new("test"), DosingSchedule::new(5.0, 24.0));
        let s = format!("{}", pta);
        assert!(s.contains("PTA"));
        assert!(s.contains("2 locations"));
    }
}
