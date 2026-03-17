//! EC-to-automata lowering compiler.
//!
//! This module lowers Event Calculus programs (axiom sets, fluents, spatial
//! predicates) into automaton transition guards, state invariants, and
//! compiled transitions suitable for code generation.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::axioms::*;
use crate::fluent::*;
use crate::local_types::*;

// ─── TransitionGuard ─────────────────────────────────────────────────────────

/// A compiled guard condition on an automaton transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionGuard {
    /// The event must match this pattern.
    EventMatch(EventPattern),
    /// A fluent must hold.
    FluentHolds(FluentId),
    /// A fluent must not hold.
    FluentNotHolds(FluentId),
    /// A spatial predicate must hold.
    SpatialHolds(SpatialPredicateId),
    /// A spatial predicate must not hold.
    SpatialNotHolds(SpatialPredicateId),
    /// A timer must have expired.
    TimerExpired(String),
    /// A numeric comparison on a fluent.
    NumericGuard {
        fluent_id: FluentId,
        op: ComparisonOp,
        threshold: f64,
    },
    /// Conjunction of guards.
    And(Vec<TransitionGuard>),
    /// Disjunction of guards.
    Or(Vec<TransitionGuard>),
    /// Negation of a guard.
    Not(Box<TransitionGuard>),
    /// Always true.
    True,
    /// Always false.
    False,
}

impl TransitionGuard {
    /// Simplify the guard by constant folding and flattening.
    pub fn simplify(&self) -> TransitionGuard {
        match self {
            TransitionGuard::And(children) => {
                let simplified: Vec<TransitionGuard> = children
                    .iter()
                    .map(|c| c.simplify())
                    .collect();

                // If any child is False, the whole thing is False
                if simplified.iter().any(|c| matches!(c, TransitionGuard::False)) {
                    return TransitionGuard::False;
                }

                // Filter out True children
                let filtered: Vec<TransitionGuard> = simplified
                    .into_iter()
                    .filter(|c| !matches!(c, TransitionGuard::True))
                    .collect();

                match filtered.len() {
                    0 => TransitionGuard::True,
                    1 => filtered.into_iter().next().unwrap(),
                    _ => {
                        // Flatten nested Ands
                        let mut flat = Vec::new();
                        for child in filtered {
                            if let TransitionGuard::And(inner) = child {
                                flat.extend(inner);
                            } else {
                                flat.push(child);
                            }
                        }
                        TransitionGuard::And(flat)
                    }
                }
            }
            TransitionGuard::Or(children) => {
                let simplified: Vec<TransitionGuard> = children
                    .iter()
                    .map(|c| c.simplify())
                    .collect();

                // If any child is True, the whole thing is True
                if simplified.iter().any(|c| matches!(c, TransitionGuard::True)) {
                    return TransitionGuard::True;
                }

                // Filter out False children
                let filtered: Vec<TransitionGuard> = simplified
                    .into_iter()
                    .filter(|c| !matches!(c, TransitionGuard::False))
                    .collect();

                match filtered.len() {
                    0 => TransitionGuard::False,
                    1 => filtered.into_iter().next().unwrap(),
                    _ => {
                        let mut flat = Vec::new();
                        for child in filtered {
                            if let TransitionGuard::Or(inner) = child {
                                flat.extend(inner);
                            } else {
                                flat.push(child);
                            }
                        }
                        TransitionGuard::Or(flat)
                    }
                }
            }
            TransitionGuard::Not(child) => {
                let simplified = child.simplify();
                match &simplified {
                    TransitionGuard::True => TransitionGuard::False,
                    TransitionGuard::False => TransitionGuard::True,
                    TransitionGuard::Not(inner) => *inner.clone(),
                    _ => TransitionGuard::Not(Box::new(simplified)),
                }
            }
            other => other.clone(),
        }
    }

    /// Count the number of atomic conditions in this guard.
    pub fn complexity(&self) -> usize {
        match self {
            TransitionGuard::And(cs) | TransitionGuard::Or(cs) => {
                cs.iter().map(|c| c.complexity()).sum()
            }
            TransitionGuard::Not(c) => c.complexity(),
            TransitionGuard::True | TransitionGuard::False => 0,
            _ => 1,
        }
    }

    /// Get all fluent IDs referenced in this guard.
    pub fn referenced_fluents(&self) -> HashSet<FluentId> {
        let mut ids = HashSet::new();
        self.collect_fluent_refs(&mut ids);
        ids
    }

    fn collect_fluent_refs(&self, ids: &mut HashSet<FluentId>) {
        match self {
            TransitionGuard::FluentHolds(id) | TransitionGuard::FluentNotHolds(id) => {
                ids.insert(*id);
            }
            TransitionGuard::NumericGuard { fluent_id, .. } => {
                ids.insert(*fluent_id);
            }
            TransitionGuard::And(cs) | TransitionGuard::Or(cs) => {
                for c in cs {
                    c.collect_fluent_refs(ids);
                }
            }
            TransitionGuard::Not(c) => c.collect_fluent_refs(ids),
            _ => {}
        }
    }

    /// Get all spatial predicate IDs referenced in this guard.
    pub fn referenced_spatial_predicates(&self) -> HashSet<SpatialPredicateId> {
        let mut ids = HashSet::new();
        self.collect_spatial_refs(&mut ids);
        ids
    }

    fn collect_spatial_refs(&self, ids: &mut HashSet<SpatialPredicateId>) {
        match self {
            TransitionGuard::SpatialHolds(id) | TransitionGuard::SpatialNotHolds(id) => {
                ids.insert(*id);
            }
            TransitionGuard::And(cs) | TransitionGuard::Or(cs) => {
                for c in cs {
                    c.collect_spatial_refs(ids);
                }
            }
            TransitionGuard::Not(c) => c.collect_spatial_refs(ids),
            _ => {}
        }
    }
}

impl fmt::Display for TransitionGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransitionGuard::EventMatch(_) => write!(f, "event(...)"),
            TransitionGuard::FluentHolds(id) => write!(f, "holds({})", id),
            TransitionGuard::FluentNotHolds(id) => write!(f, "¬holds({})", id),
            TransitionGuard::SpatialHolds(id) => write!(f, "spatial({})", id),
            TransitionGuard::SpatialNotHolds(id) => write!(f, "¬spatial({})", id),
            TransitionGuard::TimerExpired(name) => write!(f, "timer_expired({})", name),
            TransitionGuard::NumericGuard { fluent_id, op, threshold } => {
                write!(f, "{}({:?}, {})", fluent_id, op, threshold)
            }
            TransitionGuard::And(cs) => {
                let parts: Vec<String> = cs.iter().map(|c| format!("{}", c)).collect();
                write!(f, "({})", parts.join(" ∧ "))
            }
            TransitionGuard::Or(cs) => {
                let parts: Vec<String> = cs.iter().map(|c| format!("{}", c)).collect();
                write!(f, "({})", parts.join(" ∨ "))
            }
            TransitionGuard::Not(c) => write!(f, "¬({})", c),
            TransitionGuard::True => write!(f, "⊤"),
            TransitionGuard::False => write!(f, "⊥"),
        }
    }
}

// ─── StateInvariant ──────────────────────────────────────────────────────────

/// A state-level invariant that must hold while in a particular state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateInvariant {
    pub state_id: StateId,
    pub condition: TransitionGuard,
    pub fluent_requirements: Vec<(FluentId, bool)>,
    pub name: String,
}

impl StateInvariant {
    pub fn new(
        state_id: StateId,
        name: impl Into<String>,
        condition: TransitionGuard,
    ) -> Self {
        let fluent_requirements = extract_fluent_requirements(&condition);
        Self {
            state_id,
            condition,
            fluent_requirements,
            name: name.into(),
        }
    }

    /// Check if the invariant is satisfied given a fluent store.
    pub fn is_satisfied(&self, store: &FluentStore) -> bool {
        for (fid, required) in &self.fluent_requirements {
            let actual = store.get(*fid).map_or(false, |f| f.holds());
            if actual != *required {
                return false;
            }
        }
        true
    }
}

fn extract_fluent_requirements(guard: &TransitionGuard) -> Vec<(FluentId, bool)> {
    let mut reqs = Vec::new();
    match guard {
        TransitionGuard::FluentHolds(id) => reqs.push((*id, true)),
        TransitionGuard::FluentNotHolds(id) => reqs.push((*id, false)),
        TransitionGuard::And(cs) => {
            for c in cs {
                reqs.extend(extract_fluent_requirements(c));
            }
        }
        _ => {}
    }
    reqs
}

// ─── CompiledTransition ──────────────────────────────────────────────────────

/// A fully compiled automaton transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledTransition {
    pub id: TransitionId,
    pub source: StateId,
    pub target: StateId,
    pub guard: TransitionGuard,
    pub actions: Vec<CompiledAction>,
    pub priority: i32,
    pub source_axiom: Option<AxiomId>,
}

/// A compiled action to execute when a transition fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompiledAction {
    /// Set a fluent to a specific value.
    SetFluent { fluent_id: FluentId, value: Fluent },
    /// Remove (terminate) a fluent.
    TerminateFluent { fluent_id: FluentId },
    /// Start a timer.
    StartTimer { name: String, duration: Duration },
    /// Stop a timer.
    StopTimer { name: String },
    /// Emit a derived event.
    EmitEvent(EventKind),
    /// Invoke a named procedure.
    CallProcedure { name: String, params: Vec<Value> },
}

impl CompiledTransition {
    /// Check if this transition is "dead" (guard is always false).
    pub fn is_dead(&self) -> bool {
        matches!(self.guard.simplify(), TransitionGuard::False)
    }

    /// Check if this transition is "unconditional" (guard is always true, only event match).
    pub fn is_unconditional(&self) -> bool {
        match self.guard.simplify() {
            TransitionGuard::True => true,
            TransitionGuard::EventMatch(_) => true,
            _ => false,
        }
    }
}

// ─── AutomatonBlueprint ─────────────────────────────────────────────────────

/// Intermediate representation of an automaton, ready for code generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatonBlueprint {
    pub name: String,
    pub states: Vec<BlueprintState>,
    pub transitions: Vec<CompiledTransition>,
    pub invariants: Vec<StateInvariant>,
    pub initial_state: StateId,
    pub accepting_states: Vec<StateId>,
    pub fluent_declarations: Vec<(FluentId, Fluent)>,
    pub spatial_predicates: Vec<SpatialPredicateId>,
}

/// A state in the blueprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintState {
    pub id: StateId,
    pub name: String,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub on_enter_actions: Vec<CompiledAction>,
    pub on_exit_actions: Vec<CompiledAction>,
}

impl AutomatonBlueprint {
    pub fn new(name: impl Into<String>, initial_state: StateId) -> Self {
        Self {
            name: name.into(),
            states: Vec::new(),
            transitions: Vec::new(),
            invariants: Vec::new(),
            initial_state,
            accepting_states: Vec::new(),
            fluent_declarations: Vec::new(),
            spatial_predicates: Vec::new(),
        }
    }

    /// Add a state.
    pub fn add_state(&mut self, state: BlueprintState) {
        if state.is_accepting {
            self.accepting_states.push(state.id);
        }
        self.states.push(state);
    }

    /// Add a transition.
    pub fn add_transition(&mut self, transition: CompiledTransition) {
        self.transitions.push(transition);
    }

    /// Add an invariant.
    pub fn add_invariant(&mut self, invariant: StateInvariant) {
        self.invariants.push(invariant);
    }

    /// Get transitions from a specific state.
    pub fn transitions_from(&self, state: StateId) -> Vec<&CompiledTransition> {
        self.transitions
            .iter()
            .filter(|t| t.source == state)
            .collect()
    }

    /// Get transitions to a specific state.
    pub fn transitions_to(&self, state: StateId) -> Vec<&CompiledTransition> {
        self.transitions
            .iter()
            .filter(|t| t.target == state)
            .collect()
    }

    /// Eliminate dead transitions (guard is always false).
    pub fn eliminate_dead_transitions(&mut self) {
        self.transitions.retain(|t| !t.is_dead());
    }

    /// Simplify all transition guards.
    pub fn simplify_guards(&mut self) {
        for t in &mut self.transitions {
            t.guard = t.guard.simplify();
        }
    }

    /// Remove unreachable states (no incoming transitions and not initial).
    pub fn remove_unreachable_states(&mut self) {
        let mut reachable: HashSet<StateId> = HashSet::new();
        reachable.insert(self.initial_state);

        let mut changed = true;
        while changed {
            changed = false;
            for t in &self.transitions {
                if reachable.contains(&t.source) && !reachable.contains(&t.target) {
                    reachable.insert(t.target);
                    changed = true;
                }
            }
        }

        self.states.retain(|s| reachable.contains(&s.id));
        self.transitions.retain(|t| {
            reachable.contains(&t.source) && reachable.contains(&t.target)
        });
        self.accepting_states.retain(|s| reachable.contains(s));
    }

    /// Total number of states and transitions.
    pub fn size(&self) -> (usize, usize) {
        (self.states.len(), self.transitions.len())
    }
}

// ─── ECCompiler ──────────────────────────────────────────────────────────────

/// The EC-to-automata compiler.
pub struct ECCompiler {
    next_state_id: u64,
    next_transition_id: u64,
}

impl ECCompiler {
    pub fn new() -> Self {
        Self {
            next_state_id: 1,
            next_transition_id: 1,
        }
    }

    fn alloc_state(&mut self) -> StateId {
        let id = StateId(self.next_state_id);
        self.next_state_id += 1;
        id
    }

    fn alloc_transition(&mut self) -> TransitionId {
        let id = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        id
    }

    /// Compile an axiom condition into a transition guard.
    pub fn compile_condition(&self, cond: &AxiomCondition) -> TransitionGuard {
        match cond {
            AxiomCondition::FluentHolds(id) => TransitionGuard::FluentHolds(*id),
            AxiomCondition::FluentNotHolds(id) => TransitionGuard::FluentNotHolds(*id),
            AxiomCondition::EventOccurs(pattern) => TransitionGuard::EventMatch(pattern.clone()),
            AxiomCondition::SpatialCondition(_pred) => {
                // Map spatial predicate to a SpatialPredicateId
                TransitionGuard::True // simplified; real impl would map to specific ID
            }
            AxiomCondition::TemporalCondition(_tc) => {
                TransitionGuard::True // temporal constraints handled separately
            }
            AxiomCondition::Conjunction(cs) => {
                let guards: Vec<TransitionGuard> = cs
                    .iter()
                    .map(|c| self.compile_condition(c))
                    .collect();
                TransitionGuard::And(guards)
            }
            AxiomCondition::Disjunction(cs) => {
                let guards: Vec<TransitionGuard> = cs
                    .iter()
                    .map(|c| self.compile_condition(c))
                    .collect();
                TransitionGuard::Or(guards)
            }
            AxiomCondition::Negation(c) => {
                TransitionGuard::Not(Box::new(self.compile_condition(c)))
            }
            AxiomCondition::Flag(_name, expected) => {
                if *expected {
                    TransitionGuard::FluentHolds(FluentId(0)) // placeholder
                } else {
                    TransitionGuard::FluentNotHolds(FluentId(0))
                }
            }
            AxiomCondition::NumericComparison { fluent_id, op, threshold } => {
                TransitionGuard::NumericGuard {
                    fluent_id: *fluent_id,
                    op: *op,
                    threshold: *threshold,
                }
            }
        }
    }

    /// Compile all axiom conditions into a combined guard.
    fn compile_conditions(&self, conds: &[AxiomCondition]) -> TransitionGuard {
        if conds.is_empty() {
            return TransitionGuard::True;
        }
        if conds.len() == 1 {
            return self.compile_condition(&conds[0]);
        }
        TransitionGuard::And(
            conds.iter().map(|c| self.compile_condition(c)).collect()
        )
    }

    /// Compile an initiation axiom into a transition guard.
    pub fn compile_initiation_axiom(&self, axiom: &Axiom) -> Option<TransitionGuard> {
        if let Axiom::InitiationAxiom { conditions, event, .. } = axiom {
            let cond_guard = self.compile_conditions(conditions);
            let event_guard = TransitionGuard::EventMatch(event.clone());
            Some(TransitionGuard::And(vec![event_guard, cond_guard]).simplify())
        } else {
            None
        }
    }

    /// Compile a termination axiom into a transition guard.
    pub fn compile_termination_axiom(&self, axiom: &Axiom) -> Option<TransitionGuard> {
        if let Axiom::TerminationAxiom { conditions, event, .. } = axiom {
            let cond_guard = self.compile_conditions(conditions);
            let event_guard = TransitionGuard::EventMatch(event.clone());
            Some(TransitionGuard::And(vec![event_guard, cond_guard]).simplify())
        } else {
            None
        }
    }

    /// Compile a state constraint into a state invariant.
    pub fn compile_state_constraint(
        &self,
        axiom: &Axiom,
        state_id: StateId,
    ) -> Option<StateInvariant> {
        if let Axiom::StateConstraint { conditions, name, .. } = axiom {
            let guard = self.compile_conditions(conditions);
            Some(StateInvariant::new(state_id, name.as_str(), guard))
        } else {
            None
        }
    }

    /// Compile all axiom guards.
    pub fn compile_to_guards(&self, axiom_set: &AxiomSet) -> Vec<TransitionGuard> {
        let mut guards = Vec::new();
        for (_, axiom) in axiom_set.iter() {
            match axiom {
                Axiom::InitiationAxiom { .. } => {
                    if let Some(g) = self.compile_initiation_axiom(axiom) {
                        guards.push(g);
                    }
                }
                Axiom::TerminationAxiom { .. } => {
                    if let Some(g) = self.compile_termination_axiom(axiom) {
                        guards.push(g);
                    }
                }
                _ => {}
            }
        }
        guards
    }

    /// Lower an EC program to compiled transitions.
    pub fn lower_ec_to_transitions(
        &mut self,
        axiom_set: &AxiomSet,
        fluent_store: &FluentStore,
    ) -> Vec<CompiledTransition> {
        let mut transitions = Vec::new();

        // For each fluent that has initiation/termination axioms,
        // create states (fluent-off, fluent-on) and transitions between them.
        let fluent_ids = fluent_store.ids();

        for fid in &fluent_ids {
            let init_axioms = axiom_set.initiation_axioms_for(*fid);
            let term_axioms = axiom_set.termination_axioms_for(*fid);

            if init_axioms.is_empty() && term_axioms.is_empty() {
                continue;
            }

            let off_state = self.alloc_state();
            let on_state = self.alloc_state();

            // Initiation transitions: off -> on
            for axiom in &init_axioms {
                if let Some(guard) = self.compile_initiation_axiom(axiom) {
                    let tid = self.alloc_transition();
                    let actions = vec![CompiledAction::SetFluent {
                        fluent_id: *fid,
                        value: if let Axiom::InitiationAxiom { new_value, .. } = axiom {
                            new_value.clone()
                        } else {
                            Fluent::boolean("_", true)
                        },
                    }];
                    transitions.push(CompiledTransition {
                        id: tid,
                        source: off_state,
                        target: on_state,
                        guard,
                        actions,
                        priority: axiom.priority(),
                        source_axiom: Some(axiom.id()),
                    });
                }
            }

            // Termination transitions: on -> off
            for axiom in &term_axioms {
                if let Some(guard) = self.compile_termination_axiom(axiom) {
                    let tid = self.alloc_transition();
                    let actions = vec![CompiledAction::TerminateFluent { fluent_id: *fid }];
                    transitions.push(CompiledTransition {
                        id: tid,
                        source: on_state,
                        target: off_state,
                        guard,
                        actions,
                        priority: axiom.priority(),
                        source_axiom: Some(axiom.id()),
                    });
                }
            }
        }

        transitions
    }

    /// Thompson-style construction for sequential composition.
    pub fn thompson_sequential(
        &mut self,
        a: &mut AutomatonBlueprint,
        b: &AutomatonBlueprint,
    ) {
        // Connect a's accepting states to b's initial state via epsilon transitions
        let b_initial = b.initial_state;

        // Add b's states (with offset)
        for state in &b.states {
            a.add_state(state.clone());
        }

        // Add b's transitions
        for trans in &b.transitions {
            a.add_transition(trans.clone());
        }

        // Add epsilon transitions from a's accepting to b's initial
        for &acc in &a.accepting_states.clone() {
            let tid = self.alloc_transition();
            a.add_transition(CompiledTransition {
                id: tid,
                source: acc,
                target: b_initial,
                guard: TransitionGuard::True,
                actions: Vec::new(),
                priority: 0,
                source_axiom: None,
            });
        }

        // b's accepting states become the new accepting states
        a.accepting_states = b.accepting_states.clone();
    }

    /// Thompson-style construction for parallel composition.
    pub fn thompson_parallel(
        &mut self,
        a: &AutomatonBlueprint,
        b: &AutomatonBlueprint,
    ) -> AutomatonBlueprint {
        let new_initial = self.alloc_state();
        let new_accept = self.alloc_state();

        let mut blueprint = AutomatonBlueprint::new("parallel", new_initial);

        blueprint.add_state(BlueprintState {
            id: new_initial,
            name: "par_init".into(),
            is_initial: true,
            is_accepting: false,
            on_enter_actions: Vec::new(),
            on_exit_actions: Vec::new(),
        });
        blueprint.add_state(BlueprintState {
            id: new_accept,
            name: "par_accept".into(),
            is_initial: false,
            is_accepting: true,
            on_enter_actions: Vec::new(),
            on_exit_actions: Vec::new(),
        });

        // Fork: new_initial -> a.initial, new_initial -> b.initial
        for bp in [a, b] {
            for state in &bp.states {
                blueprint.add_state(state.clone());
            }
            for trans in &bp.transitions {
                blueprint.add_transition(trans.clone());
            }

            let tid = self.alloc_transition();
            blueprint.add_transition(CompiledTransition {
                id: tid,
                source: new_initial,
                target: bp.initial_state,
                guard: TransitionGuard::True,
                actions: Vec::new(),
                priority: 0,
                source_axiom: None,
            });

            for &acc in &bp.accepting_states {
                let tid = self.alloc_transition();
                blueprint.add_transition(CompiledTransition {
                    id: tid,
                    source: acc,
                    target: new_accept,
                    guard: TransitionGuard::True,
                    actions: Vec::new(),
                    priority: 0,
                    source_axiom: None,
                });
            }
        }

        blueprint.accepting_states = vec![new_accept];
        blueprint
    }

    /// Thompson-style construction for choice composition.
    pub fn thompson_choice(
        &mut self,
        a: &AutomatonBlueprint,
        b: &AutomatonBlueprint,
    ) -> AutomatonBlueprint {
        let new_initial = self.alloc_state();
        let new_accept = self.alloc_state();

        let mut blueprint = AutomatonBlueprint::new("choice", new_initial);

        blueprint.add_state(BlueprintState {
            id: new_initial,
            name: "choice_init".into(),
            is_initial: true,
            is_accepting: false,
            on_enter_actions: Vec::new(),
            on_exit_actions: Vec::new(),
        });
        blueprint.add_state(BlueprintState {
            id: new_accept,
            name: "choice_accept".into(),
            is_initial: false,
            is_accepting: true,
            on_enter_actions: Vec::new(),
            on_exit_actions: Vec::new(),
        });

        // Add both branches
        for bp in [a, b] {
            for state in &bp.states {
                blueprint.add_state(state.clone());
            }
            for trans in &bp.transitions {
                blueprint.add_transition(trans.clone());
            }

            // new_initial -> branch initial
            let tid = self.alloc_transition();
            blueprint.add_transition(CompiledTransition {
                id: tid,
                source: new_initial,
                target: bp.initial_state,
                guard: TransitionGuard::True,
                actions: Vec::new(),
                priority: 0,
                source_axiom: None,
            });

            // branch accepting -> new_accept
            for &acc in &bp.accepting_states {
                let tid = self.alloc_transition();
                blueprint.add_transition(CompiledTransition {
                    id: tid,
                    source: acc,
                    target: new_accept,
                    guard: TransitionGuard::True,
                    actions: Vec::new(),
                    priority: 0,
                    source_axiom: None,
                });
            }
        }

        blueprint.accepting_states = vec![new_accept];
        blueprint
    }

    /// Compile a complete choreography from axioms to an automaton blueprint.
    pub fn compile_choreography(
        &mut self,
        name: impl Into<String>,
        axiom_set: &AxiomSet,
        fluent_store: &FluentStore,
    ) -> AutomatonBlueprint {
        let initial_state = self.alloc_state();
        let mut blueprint = AutomatonBlueprint::new(name, initial_state);

        blueprint.add_state(BlueprintState {
            id: initial_state,
            name: "idle".into(),
            is_initial: true,
            is_accepting: true,
            on_enter_actions: Vec::new(),
            on_exit_actions: Vec::new(),
        });

        // Create states for each fluent
        let mut fluent_states: HashMap<FluentId, (StateId, StateId)> = HashMap::new();
        for fid in fluent_store.ids() {
            let off = self.alloc_state();
            let on = self.alloc_state();
            blueprint.add_state(BlueprintState {
                id: off,
                name: format!("{}_off", fid),
                is_initial: false,
                is_accepting: false,
                on_enter_actions: Vec::new(),
                on_exit_actions: Vec::new(),
            });
            blueprint.add_state(BlueprintState {
                id: on,
                name: format!("{}_on", fid),
                is_initial: false,
                is_accepting: false,
                on_enter_actions: Vec::new(),
                on_exit_actions: Vec::new(),
            });
            fluent_states.insert(fid, (off, on));
        }

        // Compile transitions
        let transitions = self.lower_ec_to_transitions(axiom_set, fluent_store);
        for trans in transitions {
            blueprint.add_transition(trans);
        }

        // Compile state constraints as invariants
        for (_, axiom) in axiom_set.iter() {
            if let Axiom::StateConstraint { .. } = axiom {
                for (_fid, (_off, on)) in &fluent_states {
                    if let Some(inv) = self.compile_state_constraint(axiom, *on) {
                        blueprint.add_invariant(inv);
                    }
                }
            }
        }

        // Record fluent declarations
        for (id, fluent) in fluent_store.iter() {
            blueprint.fluent_declarations.push((id, fluent.clone()));
        }

        // Optimize
        blueprint.simplify_guards();
        blueprint.eliminate_dead_transitions();

        blueprint
    }
}

impl Default for ECCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Optimization passes ─────────────────────────────────────────────────────

/// Merge transitions with identical source, target, and actions.
pub fn merge_identical_transitions(blueprint: &mut AutomatonBlueprint) {
    let mut merged: Vec<CompiledTransition> = Vec::new();
    let mut processed: HashSet<usize> = HashSet::new();

    for i in 0..blueprint.transitions.len() {
        if processed.contains(&i) {
            continue;
        }

        let mut combined_guard = blueprint.transitions[i].guard.clone();

        for j in (i + 1)..blueprint.transitions.len() {
            if processed.contains(&j) {
                continue;
            }
            let ti = &blueprint.transitions[i];
            let tj = &blueprint.transitions[j];
            if ti.source == tj.source && ti.target == tj.target && ti.actions == tj.actions {
                combined_guard = TransitionGuard::Or(vec![combined_guard, tj.guard.clone()]);
                processed.insert(j);
            }
        }

        let mut t = blueprint.transitions[i].clone();
        t.guard = combined_guard.simplify();
        merged.push(t);
        processed.insert(i);
    }

    blueprint.transitions = merged;
}

/// Count total guard complexity across all transitions.
pub fn total_guard_complexity(blueprint: &AutomatonBlueprint) -> usize {
    blueprint.transitions.iter().map(|t| t.guard.complexity()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axioms::AxiomId;

    #[test]
    fn test_guard_simplify_true_and() {
        let guard = TransitionGuard::And(vec![
            TransitionGuard::True,
            TransitionGuard::FluentHolds(FluentId(1)),
        ]);
        let simplified = guard.simplify();
        assert_eq!(simplified, TransitionGuard::FluentHolds(FluentId(1)));
    }

    #[test]
    fn test_guard_simplify_false_and() {
        let guard = TransitionGuard::And(vec![
            TransitionGuard::False,
            TransitionGuard::FluentHolds(FluentId(1)),
        ]);
        assert_eq!(guard.simplify(), TransitionGuard::False);
    }

    #[test]
    fn test_guard_simplify_true_or() {
        let guard = TransitionGuard::Or(vec![
            TransitionGuard::True,
            TransitionGuard::FluentHolds(FluentId(1)),
        ]);
        assert_eq!(guard.simplify(), TransitionGuard::True);
    }

    #[test]
    fn test_guard_simplify_false_or() {
        let guard = TransitionGuard::Or(vec![
            TransitionGuard::False,
            TransitionGuard::FluentHolds(FluentId(1)),
        ]);
        assert_eq!(guard.simplify(), TransitionGuard::FluentHolds(FluentId(1)));
    }

    #[test]
    fn test_guard_simplify_double_negation() {
        let guard = TransitionGuard::Not(Box::new(TransitionGuard::Not(Box::new(
            TransitionGuard::FluentHolds(FluentId(1)),
        ))));
        assert_eq!(guard.simplify(), TransitionGuard::FluentHolds(FluentId(1)));
    }

    #[test]
    fn test_guard_complexity() {
        let guard = TransitionGuard::And(vec![
            TransitionGuard::FluentHolds(FluentId(1)),
            TransitionGuard::FluentNotHolds(FluentId(2)),
            TransitionGuard::EventMatch(EventPattern::Any),
        ]);
        assert_eq!(guard.complexity(), 3);
    }

    #[test]
    fn test_guard_referenced_fluents() {
        let guard = TransitionGuard::And(vec![
            TransitionGuard::FluentHolds(FluentId(1)),
            TransitionGuard::Or(vec![
                TransitionGuard::FluentNotHolds(FluentId(2)),
                TransitionGuard::FluentHolds(FluentId(3)),
            ]),
        ]);
        let refs = guard.referenced_fluents();
        assert!(refs.contains(&FluentId(1)));
        assert!(refs.contains(&FluentId(2)));
        assert!(refs.contains(&FluentId(3)));
    }

    #[test]
    fn test_compile_initiation_axiom() {
        let compiler = ECCompiler::new();
        let axiom = Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "test".into(),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            fluent: FluentId(2),
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("test", true),
            priority: 0,
        };
        let guard = compiler.compile_initiation_axiom(&axiom).unwrap();
        assert!(guard.referenced_fluents().contains(&FluentId(1)));
    }

    #[test]
    fn test_compile_termination_axiom() {
        let compiler = ECCompiler::new();
        let axiom = Axiom::TerminationAxiom {
            id: AxiomId(2),
            name: "test_term".into(),
            conditions: vec![],
            fluent: FluentId(2),
            event: EventPattern::ActionMatch(ActionType::Deactivate),
            priority: 0,
        };
        let guard = compiler.compile_termination_axiom(&axiom).unwrap();
        // Should be just an event match (conditions empty)
        match guard {
            TransitionGuard::EventMatch(_) => {}
            _ => panic!("Expected event match, got {:?}", guard),
        }
    }

    #[test]
    fn test_compile_state_constraint() {
        let compiler = ECCompiler::new();
        let axiom = Axiom::StateConstraint {
            id: AxiomId(3),
            name: "test_constraint".into(),
            conditions: vec![
                AxiomCondition::FluentHolds(FluentId(1)),
                AxiomCondition::FluentNotHolds(FluentId(3)),
            ],
            fluent: FluentId(2),
            required_value: Fluent::boolean("x", true),
        };
        let inv = compiler.compile_state_constraint(&axiom, StateId(10)).unwrap();
        assert_eq!(inv.state_id, StateId(10));
        assert!(!inv.fluent_requirements.is_empty());
    }

    #[test]
    fn test_lower_ec_to_transitions() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("near", true));
        store.insert(Fluent::boolean("grabbed", false));

        let mut axiom_set = AxiomSet::new();
        axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "grab".into(),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            fluent: FluentId(2),
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("grabbed", true),
            priority: 0,
        });

        let mut compiler = ECCompiler::new();
        let transitions = compiler.lower_ec_to_transitions(&axiom_set, &store);
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_compile_choreography() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("near", true));
        store.insert(Fluent::boolean("grabbed", false));

        let mut axiom_set = AxiomSet::new();
        axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "grab".into(),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            fluent: FluentId(2),
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("grabbed", true),
            priority: 0,
        });

        let mut compiler = ECCompiler::new();
        let bp = compiler.compile_choreography("test", &axiom_set, &store);
        assert!(!bp.states.is_empty());
        assert!(!bp.transitions.is_empty());
    }

    #[test]
    fn test_blueprint_dead_transition_elimination() {
        let mut bp = AutomatonBlueprint::new("test", StateId(1));
        bp.add_state(BlueprintState {
            id: StateId(1), name: "s1".into(),
            is_initial: true, is_accepting: false,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        bp.add_state(BlueprintState {
            id: StateId(2), name: "s2".into(),
            is_initial: false, is_accepting: true,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });

        bp.add_transition(CompiledTransition {
            id: TransitionId(1),
            source: StateId(1),
            target: StateId(2),
            guard: TransitionGuard::False,
            actions: vec![],
            priority: 0,
            source_axiom: None,
        });
        bp.add_transition(CompiledTransition {
            id: TransitionId(2),
            source: StateId(1),
            target: StateId(2),
            guard: TransitionGuard::True,
            actions: vec![],
            priority: 0,
            source_axiom: None,
        });

        bp.eliminate_dead_transitions();
        assert_eq!(bp.transitions.len(), 1);
    }

    #[test]
    fn test_thompson_choice() {
        let mut compiler = ECCompiler::new();

        let s1 = compiler.alloc_state();
        let s2 = compiler.alloc_state();
        let mut a = AutomatonBlueprint::new("a", s1);
        a.add_state(BlueprintState {
            id: s1, name: "a_init".into(),
            is_initial: true, is_accepting: false,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        a.add_state(BlueprintState {
            id: s2, name: "a_accept".into(),
            is_initial: false, is_accepting: true,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        a.accepting_states = vec![s2];

        let s3 = compiler.alloc_state();
        let s4 = compiler.alloc_state();
        let mut b = AutomatonBlueprint::new("b", s3);
        b.add_state(BlueprintState {
            id: s3, name: "b_init".into(),
            is_initial: true, is_accepting: false,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        b.add_state(BlueprintState {
            id: s4, name: "b_accept".into(),
            is_initial: false, is_accepting: true,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        b.accepting_states = vec![s4];

        let result = compiler.thompson_choice(&a, &b);
        assert_eq!(result.accepting_states.len(), 1);
        assert!(result.states.len() >= 6); // 2 new + 2 from a + 2 from b
    }

    #[test]
    fn test_state_invariant_satisfied() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("on", true));

        let inv = StateInvariant::new(
            StateId(1),
            "must_be_on",
            TransitionGuard::FluentHolds(FluentId(1)),
        );
        assert!(inv.is_satisfied(&store));

        store.update(FluentId(1), Fluent::boolean("on", false)).unwrap();
        assert!(!inv.is_satisfied(&store));
    }

    #[test]
    fn test_compiled_transition_is_dead() {
        let t = CompiledTransition {
            id: TransitionId(1),
            source: StateId(1),
            target: StateId(2),
            guard: TransitionGuard::And(vec![TransitionGuard::True, TransitionGuard::False]),
            actions: vec![],
            priority: 0,
            source_axiom: None,
        };
        assert!(t.is_dead());
    }

    #[test]
    fn test_merge_identical_transitions() {
        let mut bp = AutomatonBlueprint::new("test", StateId(1));
        bp.add_state(BlueprintState {
            id: StateId(1), name: "s1".into(),
            is_initial: true, is_accepting: false,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        bp.add_state(BlueprintState {
            id: StateId(2), name: "s2".into(),
            is_initial: false, is_accepting: true,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });

        bp.add_transition(CompiledTransition {
            id: TransitionId(1),
            source: StateId(1),
            target: StateId(2),
            guard: TransitionGuard::FluentHolds(FluentId(1)),
            actions: vec![],
            priority: 0,
            source_axiom: None,
        });
        bp.add_transition(CompiledTransition {
            id: TransitionId(2),
            source: StateId(1),
            target: StateId(2),
            guard: TransitionGuard::FluentHolds(FluentId(2)),
            actions: vec![],
            priority: 0,
            source_axiom: None,
        });

        merge_identical_transitions(&mut bp);
        assert_eq!(bp.transitions.len(), 1);
    }

    #[test]
    fn test_blueprint_unreachable_removal() {
        let mut bp = AutomatonBlueprint::new("test", StateId(1));
        bp.add_state(BlueprintState {
            id: StateId(1), name: "s1".into(),
            is_initial: true, is_accepting: false,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        bp.add_state(BlueprintState {
            id: StateId(2), name: "s2".into(),
            is_initial: false, is_accepting: false,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        bp.add_state(BlueprintState {
            id: StateId(3), name: "unreachable".into(),
            is_initial: false, is_accepting: false,
            on_enter_actions: vec![], on_exit_actions: vec![],
        });
        bp.add_transition(CompiledTransition {
            id: TransitionId(1),
            source: StateId(1),
            target: StateId(2),
            guard: TransitionGuard::True,
            actions: vec![],
            priority: 0,
            source_axiom: None,
        });

        bp.remove_unreachable_states();
        assert_eq!(bp.states.len(), 2); // s1 and s2 reachable, s3 removed
    }
}
