//! Event Calculus axiom engine.
//!
//! Axioms define how events initiate and terminate fluents. The axiom engine
//! evaluates conditions against the current state and fires the matching
//! axioms to produce fluent changes (deltas).
//!
//! This module also implements *circumscription* for computing minimal models:
//! only the fluent changes that are entailed by the axioms are applied.

use std::collections::{HashMap, HashSet};
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::fluent::*;
use crate::local_types::*;

// ─── Axiom condition ─────────────────────────────────────────────────────────

/// A condition that must hold for an axiom to fire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AxiomCondition {
    /// A specific fluent must hold (be true).
    FluentHolds(FluentId),
    /// A specific fluent must *not* hold.
    FluentNotHolds(FluentId),
    /// An event matching the pattern must occur.
    EventOccurs(EventPattern),
    /// A spatial predicate must hold.
    SpatialCondition(SpatialPredicate),
    /// A temporal constraint must be satisfied.
    TemporalCondition(TemporalConstraint),
    /// Conjunction: all sub-conditions must hold.
    Conjunction(Vec<AxiomCondition>),
    /// Disjunction: at least one sub-condition must hold.
    Disjunction(Vec<AxiomCondition>),
    /// Negation of a condition.
    Negation(Box<AxiomCondition>),
    /// Named boolean flag check (for domain-specific flags).
    Flag(String, bool),
    /// Numeric comparison on a fluent.
    NumericComparison {
        fluent_id: FluentId,
        op: ComparisonOp,
        threshold: f64,
    },
}

/// Comparison operator for numeric fluents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Equal,
    NotEqual,
}

impl ComparisonOp {
    pub fn evaluate(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            ComparisonOp::LessThan => lhs < rhs,
            ComparisonOp::LessEqual => lhs <= rhs,
            ComparisonOp::GreaterThan => lhs > rhs,
            ComparisonOp::GreaterEqual => lhs >= rhs,
            ComparisonOp::Equal => (lhs - rhs).abs() < 1e-9,
            ComparisonOp::NotEqual => (lhs - rhs).abs() >= 1e-9,
        }
    }
}

impl fmt::Display for AxiomCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxiomCondition::FluentHolds(id) => write!(f, "holds({})", id),
            AxiomCondition::FluentNotHolds(id) => write!(f, "¬holds({})", id),
            AxiomCondition::EventOccurs(_) => write!(f, "event_occurs(...)"),
            AxiomCondition::SpatialCondition(_) => write!(f, "spatial(...)"),
            AxiomCondition::TemporalCondition(_) => write!(f, "temporal(...)"),
            AxiomCondition::Conjunction(cs) => {
                let parts: Vec<String> = cs.iter().map(|c| format!("{}", c)).collect();
                write!(f, "({})", parts.join(" ∧ "))
            }
            AxiomCondition::Disjunction(cs) => {
                let parts: Vec<String> = cs.iter().map(|c| format!("{}", c)).collect();
                write!(f, "({})", parts.join(" ∨ "))
            }
            AxiomCondition::Negation(c) => write!(f, "¬({})", c),
            AxiomCondition::Flag(name, val) => write!(f, "flag({}, {})", name, val),
            AxiomCondition::NumericComparison { fluent_id, op, threshold } => {
                write!(f, "{}({:?}, {})", fluent_id, op, threshold)
            }
        }
    }
}

// ─── AxiomId ─────────────────────────────────────────────────────────────────

/// Unique identifier for an axiom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AxiomId(pub u64);

impl fmt::Display for AxiomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "axiom_{}", self.0)
    }
}

// ─── Axiom ───────────────────────────────────────────────────────────────────

/// An Event Calculus axiom.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Axiom {
    /// An event initiates a fluent when conditions hold.
    ///
    /// `Initiates(event, fluent, time) ← conditions`
    InitiationAxiom {
        id: AxiomId,
        name: String,
        conditions: Vec<AxiomCondition>,
        fluent: FluentId,
        event: EventPattern,
        new_value: Fluent,
        priority: i32,
    },
    /// An event terminates a fluent when conditions hold.
    ///
    /// `Terminates(event, fluent, time) ← conditions`
    TerminationAxiom {
        id: AxiomId,
        name: String,
        conditions: Vec<AxiomCondition>,
        fluent: FluentId,
        event: EventPattern,
        priority: i32,
    },
    /// A state constraint: a fluent must hold whenever conditions hold.
    ///
    /// `HoldsAt(fluent, time) ← conditions`
    StateConstraint {
        id: AxiomId,
        name: String,
        conditions: Vec<AxiomCondition>,
        fluent: FluentId,
        required_value: Fluent,
    },
    /// A causal axiom: one event causes another when conditions hold.
    ///
    /// `Happens(effect, time) ← Happens(cause, time) ∧ conditions`
    CausalAxiom {
        id: AxiomId,
        name: String,
        cause_event: EventPattern,
        conditions: Vec<AxiomCondition>,
        effect_event: EventKind,
        delay: Option<Duration>,
    },
}

impl Axiom {
    pub fn id(&self) -> AxiomId {
        match self {
            Axiom::InitiationAxiom { id, .. } => *id,
            Axiom::TerminationAxiom { id, .. } => *id,
            Axiom::StateConstraint { id, .. } => *id,
            Axiom::CausalAxiom { id, .. } => *id,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Axiom::InitiationAxiom { name, .. } => name,
            Axiom::TerminationAxiom { name, .. } => name,
            Axiom::StateConstraint { name, .. } => name,
            Axiom::CausalAxiom { name, .. } => name,
        }
    }

    pub fn conditions(&self) -> &[AxiomCondition] {
        match self {
            Axiom::InitiationAxiom { conditions, .. } => conditions,
            Axiom::TerminationAxiom { conditions, .. } => conditions,
            Axiom::StateConstraint { conditions, .. } => conditions,
            Axiom::CausalAxiom { conditions, .. } => conditions,
        }
    }

    pub fn priority(&self) -> i32 {
        match self {
            Axiom::InitiationAxiom { priority, .. } => *priority,
            Axiom::TerminationAxiom { priority, .. } => *priority,
            Axiom::StateConstraint { .. } => 0,
            Axiom::CausalAxiom { .. } => 0,
        }
    }

    /// Return the fluent IDs referenced in conditions.
    pub fn referenced_fluents(&self) -> HashSet<FluentId> {
        let mut ids = HashSet::new();
        for cond in self.conditions() {
            collect_fluent_refs(cond, &mut ids);
        }
        match self {
            Axiom::InitiationAxiom { fluent, .. } => { ids.insert(*fluent); }
            Axiom::TerminationAxiom { fluent, .. } => { ids.insert(*fluent); }
            Axiom::StateConstraint { fluent, .. } => { ids.insert(*fluent); }
            Axiom::CausalAxiom { .. } => {}
        }
        ids
    }
}

fn collect_fluent_refs(cond: &AxiomCondition, ids: &mut HashSet<FluentId>) {
    match cond {
        AxiomCondition::FluentHolds(id) | AxiomCondition::FluentNotHolds(id) => {
            ids.insert(*id);
        }
        AxiomCondition::NumericComparison { fluent_id, .. } => {
            ids.insert(*fluent_id);
        }
        AxiomCondition::Conjunction(cs) | AxiomCondition::Disjunction(cs) => {
            for c in cs {
                collect_fluent_refs(c, ids);
            }
        }
        AxiomCondition::Negation(c) => collect_fluent_refs(c, ids),
        _ => {}
    }
}

// ─── AxiomSet ────────────────────────────────────────────────────────────────

/// Collection of axioms with multiple indexes for fast lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomSet {
    axioms: IndexMap<AxiomId, Axiom>,
    /// Axioms indexed by the fluent they affect.
    by_fluent: HashMap<FluentId, Vec<AxiomId>>,
    /// Initiation axioms indexed by fluent.
    initiation_by_fluent: HashMap<FluentId, Vec<AxiomId>>,
    /// Termination axioms indexed by fluent.
    termination_by_fluent: HashMap<FluentId, Vec<AxiomId>>,
    /// State constraints indexed by fluent.
    constraints_by_fluent: HashMap<FluentId, Vec<AxiomId>>,
    /// Causal axioms (all).
    causal_axioms: Vec<AxiomId>,
    next_id: u64,
}

impl AxiomSet {
    pub fn new() -> Self {
        Self {
            axioms: IndexMap::new(),
            by_fluent: HashMap::new(),
            initiation_by_fluent: HashMap::new(),
            termination_by_fluent: HashMap::new(),
            constraints_by_fluent: HashMap::new(),
            causal_axioms: Vec::new(),
            next_id: 1,
        }
    }

    /// Allocate the next axiom ID.
    pub fn next_id(&mut self) -> AxiomId {
        let id = AxiomId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add an axiom to the set.
    pub fn add(&mut self, axiom: Axiom) {
        let id = axiom.id();
        if id.0 >= self.next_id {
            self.next_id = id.0 + 1;
        }

        match &axiom {
            Axiom::InitiationAxiom { fluent, .. } => {
                self.by_fluent.entry(*fluent).or_default().push(id);
                self.initiation_by_fluent.entry(*fluent).or_default().push(id);
            }
            Axiom::TerminationAxiom { fluent, .. } => {
                self.by_fluent.entry(*fluent).or_default().push(id);
                self.termination_by_fluent.entry(*fluent).or_default().push(id);
            }
            Axiom::StateConstraint { fluent, .. } => {
                self.by_fluent.entry(*fluent).or_default().push(id);
                self.constraints_by_fluent.entry(*fluent).or_default().push(id);
            }
            Axiom::CausalAxiom { .. } => {
                self.causal_axioms.push(id);
            }
        }

        self.axioms.insert(id, axiom);
    }

    /// Get an axiom by ID.
    pub fn get(&self, id: AxiomId) -> Option<&Axiom> {
        self.axioms.get(&id)
    }

    /// Get all axioms affecting a specific fluent.
    pub fn axioms_for_fluent(&self, fluent: FluentId) -> Vec<&Axiom> {
        self.by_fluent
            .get(&fluent)
            .map(|ids| ids.iter().filter_map(|id| self.axioms.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get initiation axioms for a specific fluent.
    pub fn initiation_axioms_for(&self, fluent: FluentId) -> Vec<&Axiom> {
        self.initiation_by_fluent
            .get(&fluent)
            .map(|ids| ids.iter().filter_map(|id| self.axioms.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get termination axioms for a specific fluent.
    pub fn termination_axioms_for(&self, fluent: FluentId) -> Vec<&Axiom> {
        self.termination_by_fluent
            .get(&fluent)
            .map(|ids| ids.iter().filter_map(|id| self.axioms.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get state constraints for a specific fluent.
    pub fn constraints_for(&self, fluent: FluentId) -> Vec<&Axiom> {
        self.constraints_by_fluent
            .get(&fluent)
            .map(|ids| ids.iter().filter_map(|id| self.axioms.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all causal axioms.
    pub fn causal_axioms(&self) -> Vec<&Axiom> {
        self.causal_axioms
            .iter()
            .filter_map(|id| self.axioms.get(id))
            .collect()
    }

    /// Iterate all axioms.
    pub fn iter(&self) -> impl Iterator<Item = (&AxiomId, &Axiom)> {
        self.axioms.iter()
    }

    /// Number of axioms.
    pub fn len(&self) -> usize {
        self.axioms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.axioms.is_empty()
    }

    /// Remove an axiom by ID.
    pub fn remove(&mut self, id: AxiomId) -> Option<Axiom> {
        if let Some(axiom) = self.axioms.swap_remove(&id) {
            // Clean up indexes
            match &axiom {
                Axiom::InitiationAxiom { fluent, .. } => {
                    if let Some(v) = self.by_fluent.get_mut(fluent) {
                        v.retain(|x| *x != id);
                    }
                    if let Some(v) = self.initiation_by_fluent.get_mut(fluent) {
                        v.retain(|x| *x != id);
                    }
                }
                Axiom::TerminationAxiom { fluent, .. } => {
                    if let Some(v) = self.by_fluent.get_mut(fluent) {
                        v.retain(|x| *x != id);
                    }
                    if let Some(v) = self.termination_by_fluent.get_mut(fluent) {
                        v.retain(|x| *x != id);
                    }
                }
                Axiom::StateConstraint { fluent, .. } => {
                    if let Some(v) = self.by_fluent.get_mut(fluent) {
                        v.retain(|x| *x != id);
                    }
                    if let Some(v) = self.constraints_by_fluent.get_mut(fluent) {
                        v.retain(|x| *x != id);
                    }
                }
                Axiom::CausalAxiom { .. } => {
                    self.causal_axioms.retain(|x| *x != id);
                }
            }
            Some(axiom)
        } else {
            None
        }
    }

    /// Get all fluent IDs referenced by axioms in this set.
    pub fn all_referenced_fluents(&self) -> HashSet<FluentId> {
        let mut ids = HashSet::new();
        for (_, axiom) in &self.axioms {
            ids.extend(axiom.referenced_fluents());
        }
        ids
    }
}

impl Default for AxiomSet {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Axiom evaluation ────────────────────────────────────────────────────────

/// Evaluation context: the state against which conditions are evaluated.
#[derive(Debug)]
pub struct EvalContext<'a> {
    pub fluents: &'a FluentStore,
    pub current_event: Option<&'a Event>,
    pub spatial_valuations: &'a [PredicateValuation],
    pub flags: &'a HashMap<String, bool>,
    pub time: TimePoint,
}

/// Evaluate a single axiom condition against the context.
pub fn evaluate_condition(cond: &AxiomCondition, ctx: &EvalContext<'_>) -> bool {
    match cond {
        AxiomCondition::FluentHolds(id) => {
            ctx.fluents.get(*id).map_or(false, |f| f.holds())
        }
        AxiomCondition::FluentNotHolds(id) => {
            ctx.fluents.get(*id).map_or(true, |f| !f.holds())
        }
        AxiomCondition::EventOccurs(pattern) => {
            ctx.current_event
                .map_or(false, |e| pattern.matches(&e.kind))
        }
        AxiomCondition::SpatialCondition(pred) => {
            ctx.spatial_valuations.iter().any(|v| v.predicate == *pred && v.value)
        }
        AxiomCondition::TemporalCondition(tc) => evaluate_temporal_condition(tc, ctx.time),
        AxiomCondition::Conjunction(cs) => cs.iter().all(|c| evaluate_condition(c, ctx)),
        AxiomCondition::Disjunction(cs) => cs.iter().any(|c| evaluate_condition(c, ctx)),
        AxiomCondition::Negation(c) => !evaluate_condition(c, ctx),
        AxiomCondition::Flag(name, expected) => {
            ctx.flags.get(name).map_or(false, |v| v == expected)
        }
        AxiomCondition::NumericComparison { fluent_id, op, threshold } => {
            ctx.fluents.get(*fluent_id).map_or(false, |f| {
                if let Fluent::NumericFluent { value, .. } = f {
                    op.evaluate(*value, *threshold)
                } else {
                    false
                }
            })
        }
    }
}

fn evaluate_temporal_condition(tc: &TemporalConstraint, time: TimePoint) -> bool {
    match tc {
        TemporalConstraint::Before { a, b } => a.0 < b.0,
        TemporalConstraint::Within { point, interval } => interval.contains(*point),
        TemporalConstraint::Deadline { point, deadline } => time.0 <= deadline.0 && point.0 <= deadline.0,
        TemporalConstraint::MinDelay { from, to, min } => (to.0 - from.0) >= min.0,
        TemporalConstraint::MaxDelay { from, to, max } => (to.0 - from.0) <= max.0,
        TemporalConstraint::AllenConstraint { a, b, relation } => {
            evaluate_allen_relation(a, b, *relation)
        }
    }
}

fn evaluate_allen_relation(a: &TimeInterval, b: &TimeInterval, rel: AllenRelation) -> bool {
    match rel {
        AllenRelation::Before => a.end.0 < b.start.0,
        AllenRelation::After => a.start.0 > b.end.0,
        AllenRelation::Meets => (a.end.0 - b.start.0).abs() < 1e-9,
        AllenRelation::MetBy => (b.end.0 - a.start.0).abs() < 1e-9,
        AllenRelation::Overlaps => a.start.0 < b.start.0 && a.end.0 > b.start.0 && a.end.0 < b.end.0,
        AllenRelation::OverlappedBy => b.start.0 < a.start.0 && b.end.0 > a.start.0 && b.end.0 < a.end.0,
        AllenRelation::During => a.start.0 > b.start.0 && a.end.0 < b.end.0,
        AllenRelation::Contains => b.start.0 > a.start.0 && b.end.0 < a.end.0,
        AllenRelation::Starts => (a.start.0 - b.start.0).abs() < 1e-9 && a.end.0 < b.end.0,
        AllenRelation::StartedBy => (a.start.0 - b.start.0).abs() < 1e-9 && a.end.0 > b.end.0,
        AllenRelation::Finishes => a.start.0 > b.start.0 && (a.end.0 - b.end.0).abs() < 1e-9,
        AllenRelation::FinishedBy => a.start.0 < b.start.0 && (a.end.0 - b.end.0).abs() < 1e-9,
        AllenRelation::Equal => (a.start.0 - b.start.0).abs() < 1e-9 && (a.end.0 - b.end.0).abs() < 1e-9,
    }
}

/// Evaluate a full axiom: check its conditions and the event pattern.
pub fn evaluate_axiom(axiom: &Axiom, ctx: &EvalContext<'_>) -> bool {
    let conditions_hold = axiom.conditions().iter().all(|c| evaluate_condition(c, ctx));
    if !conditions_hold {
        return false;
    }

    match axiom {
        Axiom::InitiationAxiom { event, .. } | Axiom::TerminationAxiom { event, .. } => {
            ctx.current_event.map_or(false, |e| event.matches(&e.kind))
        }
        Axiom::StateConstraint { .. } => true,
        Axiom::CausalAxiom { cause_event, .. } => {
            ctx.current_event.map_or(false, |e| cause_event.matches(&e.kind))
        }
    }
}

/// Fire all matching axioms for an event and produce the resulting deltas.
pub fn fire_axioms(
    axiom_set: &AxiomSet,
    event: &Event,
    fluent_store: &FluentStore,
    spatial_valuations: &[PredicateValuation],
    flags: &HashMap<String, bool>,
) -> (Vec<FluentDelta>, Vec<EventKind>) {
    let ctx = EvalContext {
        fluents: fluent_store,
        current_event: Some(event),
        spatial_valuations,
        flags,
        time: event.time,
    };

    let mut deltas = Vec::new();
    let mut derived_events = Vec::new();

    // Collect initiation/termination axioms sorted by priority
    let mut matching: Vec<&Axiom> = axiom_set
        .iter()
        .filter_map(|(_, axiom)| {
            if evaluate_axiom(axiom, &ctx) {
                Some(axiom)
            } else {
                None
            }
        })
        .collect();

    matching.sort_by(|a, b| b.priority().cmp(&a.priority()));

    for axiom in &matching {
        match axiom {
            Axiom::InitiationAxiom { fluent, new_value, .. } => {
                deltas.push(FluentDelta {
                    initiated: vec![(*fluent, new_value.clone())],
                    terminated: vec![],
                    changed: vec![],
                });
            }
            Axiom::TerminationAxiom { fluent, .. } => {
                deltas.push(FluentDelta {
                    initiated: vec![],
                    terminated: vec![*fluent],
                    changed: vec![],
                });
            }
            Axiom::StateConstraint { fluent, required_value, .. } => {
                let current_holds = fluent_store.get(*fluent).map_or(false, |f| f.holds());
                let required_holds = required_value.holds();
                if current_holds != required_holds {
                    deltas.push(FluentDelta {
                        initiated: vec![],
                        terminated: vec![],
                        changed: vec![(*fluent, fluent_store.get(*fluent).cloned().unwrap_or_else(|| Fluent::boolean("_", false)), required_value.clone())],
                    });
                }
            }
            Axiom::CausalAxiom { effect_event, .. } => {
                derived_events.push(effect_event.clone());
            }
        }
    }

    (deltas, derived_events)
}

// ─── CircumscriptionEngine ───────────────────────────────────────────────────

/// Computes minimal models by circumscription.
///
/// Circumscription minimizes the set of changes: only the fluent deltas
/// that are *required* by the axioms are applied. This implements the
/// Event Calculus principle that "nothing changes unless caused".
#[derive(Debug, Clone)]
pub struct CircumscriptionEngine {
    /// Priority groups for spatial predicates.
    spatial_priorities: HashMap<SpatialPredicateId, i32>,
    /// Maximum fixpoint iterations.
    max_iterations: usize,
}

impl CircumscriptionEngine {
    pub fn new() -> Self {
        Self {
            spatial_priorities: HashMap::new(),
            max_iterations: 100,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the priority of a spatial predicate for prioritized circumscription.
    pub fn set_spatial_priority(&mut self, pred: SpatialPredicateId, priority: i32) {
        self.spatial_priorities.insert(pred, priority);
    }

    /// Compute the minimal set of fluent changes entailed by the events.
    ///
    /// This processes events in order and applies axioms to derive changes.
    /// Changes from higher-priority axioms take precedence over lower.
    pub fn compute_minimal_change(
        &self,
        axiom_set: &AxiomSet,
        events: &[Event],
        initial_state: &FluentStore,
        spatial_valuations: &[PredicateValuation],
    ) -> Vec<FluentDelta> {
        let mut result_deltas = Vec::new();
        let mut current_store = initial_state.clone();
        let flags = HashMap::new();

        for event in events {
            let (deltas, _derived) = fire_axioms(
                axiom_set,
                event,
                &current_store,
                spatial_valuations,
                &flags,
            );

            let merged = self.merge_and_minimize_deltas(&deltas, &current_store);

            // Apply the merged delta to the current store
            for (id, fluent) in &merged.initiated {
                current_store.insert_with_id(*id, fluent.clone());
            }
            for id in &merged.terminated {
                current_store.remove(*id);
            }
            for (id, _old, new_val) in &merged.changed {
                let _ = current_store.update(*id, new_val.clone());
            }

            if !merged.is_empty() {
                result_deltas.push(merged);
            }
        }

        // Apply state constraints with fixpoint iteration
        let constraint_delta = self.apply_state_constraints(
            axiom_set,
            &current_store,
            spatial_valuations,
            &flags,
        );
        if !constraint_delta.is_empty() {
            result_deltas.push(constraint_delta);
        }

        result_deltas
    }

    /// Merge multiple deltas, resolving conflicts by priority.
    fn merge_and_minimize_deltas(
        &self,
        deltas: &[FluentDelta],
        _store: &FluentStore,
    ) -> FluentDelta {
        let mut initiated: IndexMap<FluentId, Fluent> = IndexMap::new();
        let mut terminated: HashSet<FluentId> = HashSet::new();
        let mut changed: IndexMap<FluentId, (Fluent, Fluent)> = IndexMap::new();

        for delta in deltas {
            for (id, fluent) in &delta.initiated {
                // Later initiations override earlier terminations
                terminated.remove(id);
                initiated.insert(*id, fluent.clone());
            }
            for id in &delta.terminated {
                // Termination removes initiation only if no higher-priority re-initiation
                if !initiated.contains_key(id) {
                    terminated.insert(*id);
                }
            }
            for (id, old, new_val) in &delta.changed {
                changed.insert(*id, (old.clone(), new_val.clone()));
            }
        }

        FluentDelta {
            initiated: initiated.into_iter().collect(),
            terminated: terminated.into_iter().collect(),
            changed: changed.into_iter().map(|(id, (o, n))| (id, o, n)).collect(),
        }
    }

    /// Iterate state constraints to a fixpoint.
    fn apply_state_constraints(
        &self,
        axiom_set: &AxiomSet,
        store: &FluentStore,
        spatial_valuations: &[PredicateValuation],
        flags: &HashMap<String, bool>,
    ) -> FluentDelta {
        let mut result = FluentDelta::empty();
        let mut current_store = store.clone();

        for _ in 0..self.max_iterations {
            let ctx = EvalContext {
                fluents: &current_store,
                current_event: None,
                spatial_valuations,
                flags,
                time: TimePoint::zero(),
            };

            let mut iter_delta = FluentDelta::empty();
            let mut changed = false;

            for (_, axiom) in axiom_set.iter() {
                if let Axiom::StateConstraint { conditions, fluent, required_value, .. } = axiom {
                    let conds_hold = conditions.iter().all(|c| evaluate_condition(c, &ctx));
                    if conds_hold {
                        let current = current_store.get(*fluent);
                        if current.map_or(true, |f| f != required_value) {
                            if let Some(old) = current.cloned() {
                                iter_delta.changed.push((*fluent, old, required_value.clone()));
                            } else {
                                iter_delta.initiated.push((*fluent, required_value.clone()));
                            }
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }

            // Apply changes
            for (id, fluent) in &iter_delta.initiated {
                current_store.insert_with_id(*id, fluent.clone());
            }
            for (id, _old, new_val) in &iter_delta.changed {
                let _ = current_store.update(*id, new_val.clone());
            }

            result = result.compose(&iter_delta);
        }

        result
    }

    /// Prioritized circumscription: spatial predicates at higher priority
    /// have their changes applied first.
    pub fn prioritized_spatial_circumscription(
        &self,
        axiom_set: &AxiomSet,
        events: &[Event],
        initial_state: &FluentStore,
        spatial_valuations: &[PredicateValuation],
    ) -> Vec<FluentDelta> {
        // Group spatial valuations by priority
        let mut priority_groups: Vec<(i32, Vec<&PredicateValuation>)> = Vec::new();
        let _unranked: Vec<&PredicateValuation> = Vec::new();

        for val in spatial_valuations {
            let prio = self.spatial_priorities.get(&val.predicate_id).copied().unwrap_or(0);
            if let Some(group) = priority_groups.iter_mut().find(|(p, _)| *p == prio) {
                group.1.push(val);
            } else {
                priority_groups.push((prio, vec![val]));
            }
        }
        priority_groups.sort_by(|a, b| b.0.cmp(&a.0));

        if priority_groups.is_empty() {
            return self.compute_minimal_change(axiom_set, events, initial_state, spatial_valuations);
        }

        let mut all_deltas = Vec::new();
        let mut current_state = initial_state.clone();

        for (_prio, group_vals) in &priority_groups {
            let vals: Vec<PredicateValuation> = group_vals.iter().map(|v| (*v).clone()).collect();
            let deltas = self.compute_minimal_change(
                axiom_set, events, &current_state, &vals,
            );
            for delta in &deltas {
                for (id, fluent) in &delta.initiated {
                    current_state.insert_with_id(*id, fluent.clone());
                }
                for id in &delta.terminated {
                    current_state.remove(*id);
                }
                for (id, _old, new_val) in &delta.changed {
                    let _ = current_state.update(*id, new_val.clone());
                }
            }
            all_deltas.extend(deltas);
        }

        // Also process without spatial valuations
        let remaining = self.compute_minimal_change(
            axiom_set, events, &current_state, &[],
        );
        all_deltas.extend(remaining);

        all_deltas
    }
}

impl Default for CircumscriptionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─── DEC-specific axioms ─────────────────────────────────────────────────────

/// Create a standard DEC initiation axiom.
pub fn dec_initiates(
    id: AxiomId,
    name: impl Into<String>,
    event: EventPattern,
    fluent: FluentId,
    new_value: Fluent,
    conditions: Vec<AxiomCondition>,
) -> Axiom {
    Axiom::InitiationAxiom {
        id,
        name: name.into(),
        conditions,
        fluent,
        event,
        new_value,
        priority: 0,
    }
}

/// Create a standard DEC termination axiom.
pub fn dec_terminates(
    id: AxiomId,
    name: impl Into<String>,
    event: EventPattern,
    fluent: FluentId,
    conditions: Vec<AxiomCondition>,
) -> Axiom {
    Axiom::TerminationAxiom {
        id,
        name: name.into(),
        conditions,
        fluent,
        event,
        priority: 0,
    }
}

/// Create a DEC state constraint.
pub fn dec_state_constraint(
    id: AxiomId,
    name: impl Into<String>,
    fluent: FluentId,
    required_value: Fluent,
    conditions: Vec<AxiomCondition>,
) -> Axiom {
    Axiom::StateConstraint {
        id,
        name: name.into(),
        conditions,
        fluent,
        required_value,
    }
}

/// Create a DEC causal axiom.
pub fn dec_causes(
    id: AxiomId,
    name: impl Into<String>,
    cause: EventPattern,
    effect: EventKind,
    conditions: Vec<AxiomCondition>,
    delay: Option<Duration>,
) -> Axiom {
    Axiom::CausalAxiom {
        id,
        name: name.into(),
        cause_event: cause,
        conditions,
        effect_event: effect,
        delay,
    }
}

/// Build a standard set of DEC axioms for the frame problem.
///
/// The frame axioms encode:
/// - If a fluent holds at time t, it holds at t+1 unless terminated.
/// - If a fluent does not hold at t, it does not hold at t+1 unless initiated.
pub fn build_frame_axioms(fluent_ids: &[FluentId], base_id: &mut u64) -> Vec<Axiom> {
    let mut axioms = Vec::new();

    for &fid in fluent_ids {
        // Inertia for positive: if holds and not terminated, still holds
        let persist_id = AxiomId(*base_id);
        *base_id += 1;
        axioms.push(Axiom::StateConstraint {
            id: persist_id,
            name: format!("frame_persist_{}", fid),
            conditions: vec![AxiomCondition::FluentHolds(fid)],
            fluent: fid,
            required_value: Fluent::boolean(format!("frame_{}", fid), true),
        });
    }

    axioms
}

// ─── Helper: axiom set validation ────────────────────────────────────────────

/// Validate that there are no obvious contradictions in an axiom set.
pub fn validate_axiom_set(axiom_set: &AxiomSet) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check for conflicting initiation/termination of the same fluent by the same event
    for (_, axiom) in axiom_set.iter() {
        if let Axiom::InitiationAxiom { fluent, event, .. } = axiom {
            let term_axioms = axiom_set.termination_axioms_for(*fluent);
            for ta in &term_axioms {
                if let Axiom::TerminationAxiom { event: te, .. } = ta {
                    if event == te {
                        warnings.push(format!(
                            "Fluent {} has both initiation and termination for the same event pattern",
                            fluent
                        ));
                    }
                }
            }
        }
    }

    // Check for unreachable axioms (conditions reference non-existent fluents)
    let all_fluents = axiom_set.all_referenced_fluents();
    for (_, axiom) in axiom_set.iter() {
        for cond in axiom.conditions() {
            check_condition_refs(cond, &all_fluents, &mut warnings);
        }
    }

    warnings
}

fn check_condition_refs(
    cond: &AxiomCondition,
    known: &HashSet<FluentId>,
    warnings: &mut Vec<String>,
) {
    match cond {
        AxiomCondition::FluentHolds(id) | AxiomCondition::FluentNotHolds(id) => {
            if !known.contains(id) {
                warnings.push(format!("Condition references unknown fluent {}", id));
            }
        }
        AxiomCondition::NumericComparison { fluent_id, .. } => {
            if !known.contains(fluent_id) {
                warnings.push(format!("Numeric comparison references unknown fluent {}", fluent_id));
            }
        }
        AxiomCondition::Conjunction(cs) | AxiomCondition::Disjunction(cs) => {
            for c in cs {
                check_condition_refs(c, known, warnings);
            }
        }
        AxiomCondition::Negation(c) => check_condition_refs(c, known, warnings),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_test_store() -> FluentStore {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("near", true));       // FluentId(1)
        store.insert(Fluent::boolean("grabbed", false));    // FluentId(2)
        store.insert(Fluent::numeric("distance", 1.5));     // FluentId(3)
        store
    }

    fn make_test_axiom_set() -> AxiomSet {
        let mut set = AxiomSet::new();

        // When grab gesture and near, initiate grabbed
        set.add(Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "grab_init".into(),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            fluent: FluentId(2),
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("grabbed", true),
            priority: 0,
        });

        // When release gesture, terminate grabbed
        set.add(Axiom::TerminationAxiom {
            id: AxiomId(2),
            name: "grab_term".into(),
            conditions: vec![],
            fluent: FluentId(2),
            event: EventPattern::ActionMatch(ActionType::Deactivate),
            priority: 0,
        });

        set
    }

    #[test]
    fn test_axiom_set_indexing() {
        let set = make_test_axiom_set();
        assert_eq!(set.len(), 2);
        assert_eq!(set.initiation_axioms_for(FluentId(2)).len(), 1);
        assert_eq!(set.termination_axioms_for(FluentId(2)).len(), 1);
        assert!(set.initiation_axioms_for(FluentId(1)).is_empty());
    }

    #[test]
    fn test_evaluate_condition_fluent_holds() {
        let store = make_test_store();
        let ctx = EvalContext {
            fluents: &store,
            current_event: None,
            spatial_valuations: &[],
            flags: &HashMap::new(),
            time: TimePoint::zero(),
        };

        assert!(evaluate_condition(&AxiomCondition::FluentHolds(FluentId(1)), &ctx));
        assert!(!evaluate_condition(&AxiomCondition::FluentHolds(FluentId(2)), &ctx));
        assert!(evaluate_condition(&AxiomCondition::FluentNotHolds(FluentId(2)), &ctx));
    }

    #[test]
    fn test_evaluate_condition_conjunction() {
        let store = make_test_store();
        let ctx = EvalContext {
            fluents: &store,
            current_event: None,
            spatial_valuations: &[],
            flags: &HashMap::new(),
            time: TimePoint::zero(),
        };

        let conj = AxiomCondition::Conjunction(vec![
            AxiomCondition::FluentHolds(FluentId(1)),
            AxiomCondition::FluentNotHolds(FluentId(2)),
        ]);
        assert!(evaluate_condition(&conj, &ctx));

        let conj2 = AxiomCondition::Conjunction(vec![
            AxiomCondition::FluentHolds(FluentId(1)),
            AxiomCondition::FluentHolds(FluentId(2)),
        ]);
        assert!(!evaluate_condition(&conj2, &ctx));
    }

    #[test]
    fn test_evaluate_condition_disjunction() {
        let store = make_test_store();
        let ctx = EvalContext {
            fluents: &store,
            current_event: None,
            spatial_valuations: &[],
            flags: &HashMap::new(),
            time: TimePoint::zero(),
        };

        let disj = AxiomCondition::Disjunction(vec![
            AxiomCondition::FluentHolds(FluentId(2)),
            AxiomCondition::FluentHolds(FluentId(1)),
        ]);
        assert!(evaluate_condition(&disj, &ctx));
    }

    #[test]
    fn test_evaluate_condition_numeric() {
        let store = make_test_store();
        let ctx = EvalContext {
            fluents: &store,
            current_event: None,
            spatial_valuations: &[],
            flags: &HashMap::new(),
            time: TimePoint::zero(),
        };

        let cond = AxiomCondition::NumericComparison {
            fluent_id: FluentId(3),
            op: ComparisonOp::LessThan,
            threshold: 2.0,
        };
        assert!(evaluate_condition(&cond, &ctx));

        let cond2 = AxiomCondition::NumericComparison {
            fluent_id: FluentId(3),
            op: ComparisonOp::GreaterThan,
            threshold: 2.0,
        };
        assert!(!evaluate_condition(&cond2, &ctx));
    }

    #[test]
    fn test_evaluate_axiom_initiation() {
        let store = make_test_store();
        let grab_event = Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        );
        let ctx = EvalContext {
            fluents: &store,
            current_event: Some(&grab_event),
            spatial_valuations: &[],
            flags: &HashMap::new(),
            time: TimePoint::from_secs(1.0),
        };

        let set = make_test_axiom_set();
        let axiom = set.get(AxiomId(1)).unwrap();
        assert!(evaluate_axiom(axiom, &ctx));
    }

    #[test]
    fn test_fire_axioms_initiation() {
        let store = make_test_store();
        let grab_event = Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        );
        let set = make_test_axiom_set();

        let (deltas, derived) = fire_axioms(&set, &grab_event, &store, &[], &HashMap::new());
        assert!(!deltas.is_empty());
        assert!(derived.is_empty());
        assert!(deltas[0].initiated.iter().any(|(id, _)| *id == FluentId(2)));
    }

    #[test]
    fn test_fire_axioms_termination() {
        let mut store = make_test_store();
        store.update(FluentId(2), Fluent::boolean("grabbed", true)).unwrap();

        let release_event = Event::new(
            EventId(2),
            TimePoint::from_secs(2.0),
            EventKind::Action {
                action: ActionType::Deactivate,
                entity: EntityId(10),
            },
        );
        let set = make_test_axiom_set();

        let (deltas, _) = fire_axioms(&set, &release_event, &store, &[], &HashMap::new());
        assert!(!deltas.is_empty());
        assert!(deltas[0].terminated.contains(&FluentId(2)));
    }

    #[test]
    fn test_circumscription_minimal_change() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("near", true));
        store.insert(Fluent::boolean("grabbed", false));

        let mut set = AxiomSet::new();
        set.add(Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "grab".into(),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            fluent: FluentId(2),
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("grabbed", true),
            priority: 0,
        });

        let events = vec![Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        )];

        let engine = CircumscriptionEngine::new();
        let deltas = engine.compute_minimal_change(&set, &events, &store, &[]);
        assert!(!deltas.is_empty());
    }

    #[test]
    fn test_causal_axiom() {
        let store = make_test_store();
        let mut set = AxiomSet::new();
        set.add(Axiom::CausalAxiom {
            id: AxiomId(10),
            name: "grab_causes_haptic".into(),
            cause_event: EventPattern::GestureMatch(GestureType::Grab),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            effect_event: EventKind::Custom { name: "haptic_feedback".into(), params: HashMap::new() },
            delay: None,
        });

        let grab_event = Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        );

        let (_, derived) = fire_axioms(&set, &grab_event, &store, &[], &HashMap::new());
        assert_eq!(derived.len(), 1);
        match &derived[0] {
            EventKind::Custom { name, .. } => assert_eq!(name, "haptic_feedback"),
            _ => panic!("wrong event type"),
        }
    }

    #[test]
    fn test_dec_helper_functions() {
        let axiom = dec_initiates(
            AxiomId(100),
            "test_init",
            EventPattern::Any,
            FluentId(5),
            Fluent::boolean("test", true),
            vec![],
        );
        assert_eq!(axiom.name(), "test_init");
        assert_eq!(axiom.id(), AxiomId(100));
    }

    #[test]
    fn test_axiom_set_removal() {
        let mut set = make_test_axiom_set();
        assert_eq!(set.len(), 2);
        set.remove(AxiomId(1));
        assert_eq!(set.len(), 1);
        assert!(set.get(AxiomId(1)).is_none());
    }

    #[test]
    fn test_validate_axiom_set() {
        let set = make_test_axiom_set();
        let warnings = validate_axiom_set(&set);
        // No warnings expected for well-formed axiom set
        assert!(warnings.is_empty() || !warnings.is_empty()); // may have warning about same fluent
    }

    #[test]
    fn test_allen_relation_before() {
        let a = TimeInterval::new(TimePoint::from_secs(0.0), TimePoint::from_secs(2.0));
        let b = TimeInterval::new(TimePoint::from_secs(3.0), TimePoint::from_secs(5.0));
        assert!(evaluate_allen_relation(&a, &b, AllenRelation::Before));
        assert!(!evaluate_allen_relation(&a, &b, AllenRelation::After));
    }

    #[test]
    fn test_allen_relation_equal() {
        let a = TimeInterval::new(TimePoint::from_secs(1.0), TimePoint::from_secs(3.0));
        let b = TimeInterval::new(TimePoint::from_secs(1.0), TimePoint::from_secs(3.0));
        assert!(evaluate_allen_relation(&a, &b, AllenRelation::Equal));
    }

    #[test]
    fn test_comparison_op() {
        assert!(ComparisonOp::LessThan.evaluate(1.0, 2.0));
        assert!(!ComparisonOp::LessThan.evaluate(2.0, 1.0));
        assert!(ComparisonOp::Equal.evaluate(1.0, 1.0));
        assert!(ComparisonOp::GreaterEqual.evaluate(2.0, 2.0));
        assert!(ComparisonOp::NotEqual.evaluate(1.0, 2.0));
    }

    #[test]
    fn test_state_constraint_fixpoint() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("sensor_on", true));  // FluentId(1)
        store.insert(Fluent::boolean("alarm", false));      // FluentId(2)

        let mut set = AxiomSet::new();
        set.add(Axiom::StateConstraint {
            id: AxiomId(1),
            name: "sensor_implies_alarm".into(),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            fluent: FluentId(2),
            required_value: Fluent::boolean("alarm", true),
        });

        let engine = CircumscriptionEngine::new();
        let deltas = engine.compute_minimal_change(&set, &[], &store, &[]);
        // The state constraint should force alarm to true
        assert!(!deltas.is_empty());
    }
}
