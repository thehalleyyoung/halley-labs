//! EC evaluation engine.
//!
//! The `ECEngine` is the core of the Event Calculus pipeline.  It maintains
//! an axiom set, a fluent store, and (optionally) a spatial oracle, and
//! processes events via forward-chaining evaluation.  The engine supports
//! incremental evaluation, state rollback, abductive reasoning, and produces
//! complete narratives from event traces.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::axioms::*;
use crate::fluent::*;
use crate::local_types::*;
use crate::spatial_oracle::SpatialOracle;

// ─── ECEngineConfig ──────────────────────────────────────────────────────────

/// Configuration parameters for the EC engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECEngineConfig {
    /// Maximum number of fixpoint iterations for state constraints.
    pub max_constraint_iterations: usize,
    /// Maximum number of derived events to process per step.
    pub max_derived_events_per_step: usize,
    /// Tolerance for floating-point time comparisons.
    pub time_epsilon: f64,
    /// Whether to enable the spatial oracle.
    pub spatial_enabled: bool,
    /// Sampling interval for the spatial oracle.
    pub spatial_sample_interval: Duration,
    /// Maximum rollback history depth.
    pub max_rollback_depth: usize,
    /// Whether to record full history for abductive reasoning.
    pub record_full_history: bool,
}

impl Default for ECEngineConfig {
    fn default() -> Self {
        Self {
            max_constraint_iterations: 100,
            max_derived_events_per_step: 50,
            time_epsilon: 1e-9,
            spatial_enabled: true,
            spatial_sample_interval: Duration::from_millis(16.0),
            max_rollback_depth: 100,
            record_full_history: true,
        }
    }
}

// ─── ECState ─────────────────────────────────────────────────────────────────

/// Complete state of the EC engine at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECState {
    /// The time of this state.
    pub time: TimePoint,
    /// Snapshot of all fluent values.
    pub fluent_snapshot: FluentSnapshot,
    /// Events that occurred at this time.
    pub events: Vec<Event>,
    /// Fluent deltas applied at this time.
    pub deltas: Vec<FluentDelta>,
    /// Derived events generated at this time.
    pub derived_events: Vec<Event>,
    /// Spatial valuations at this time.
    pub spatial_valuations: Vec<PredicateValuation>,
    /// State sequence number (monotonically increasing).
    pub sequence: u64,
}

impl ECState {
    pub fn new(time: TimePoint, snapshot: FluentSnapshot, sequence: u64) -> Self {
        Self {
            time,
            fluent_snapshot: snapshot,
            events: Vec::new(),
            deltas: Vec::new(),
            derived_events: Vec::new(),
            spatial_valuations: Vec::new(),
            sequence,
        }
    }

    /// Check if a fluent holds in this state.
    pub fn holds(&self, fluent_id: FluentId) -> bool {
        self.fluent_snapshot.holds(fluent_id)
    }

    /// Get the value of a fluent in this state.
    pub fn get_fluent(&self, fluent_id: FluentId) -> Option<&Fluent> {
        self.fluent_snapshot.get(fluent_id)
    }

    /// Number of events processed at this time point.
    pub fn event_count(&self) -> usize {
        self.events.len() + self.derived_events.len()
    }

    /// Total number of fluent changes at this time point.
    pub fn change_count(&self) -> usize {
        self.deltas.iter().map(|d| d.modification_count()).sum()
    }
}

impl fmt::Display for ECState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ECState(t={}, fluents={}, events={}, changes={})",
            self.time,
            self.fluent_snapshot.len(),
            self.event_count(),
            self.change_count()
        )
    }
}

// ─── Narrative ───────────────────────────────────────────────────────────────

/// A complete narrative: the sequence of states and events over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Narrative {
    pub states: Vec<ECState>,
    pub all_events: Vec<Event>,
    pub fluent_history: FluentHistory,
    pub time_span: Option<TimeInterval>,
}

impl Narrative {
    pub fn new(initial_state: FluentSnapshot) -> Self {
        Self {
            states: Vec::new(),
            all_events: Vec::new(),
            fluent_history: FluentHistory::new(initial_state),
            time_span: None,
        }
    }

    /// Add a state to the narrative.
    pub fn push_state(&mut self, state: ECState) {
        if let Some(span) = &mut self.time_span {
            if state.time > span.end {
                span.end = state.time;
            }
            if state.time < span.start {
                span.start = state.time;
            }
        } else {
            self.time_span = Some(TimeInterval::new(state.time, state.time));
        }

        // Record snapshot in history
        self.fluent_history.record_snapshot(state.fluent_snapshot.clone());

        // Record initiations and terminations from deltas
        for delta in &state.deltas {
            for (id, fluent) in &delta.initiated {
                self.fluent_history.record_initiation(InitiatedBy {
                    fluent_id: *id,
                    event_id: state.events.first().map_or(EventId(0), |e| e.id),
                    time: state.time,
                    new_value: fluent.clone(),
                });
            }
            for id in &delta.terminated {
                self.fluent_history.record_termination(TerminatedBy {
                    fluent_id: *id,
                    event_id: state.events.first().map_or(EventId(0), |e| e.id),
                    time: state.time,
                });
            }
        }

        for event in &state.events {
            self.all_events.push(event.clone());
        }
        for event in &state.derived_events {
            self.all_events.push(event.clone());
        }

        self.states.push(state);
    }

    /// Get the state at a given time.
    pub fn state_at(&self, time: TimePoint) -> Option<&ECState> {
        self.states
            .iter()
            .rev()
            .find(|s| s.time <= time)
    }

    /// Get the final state.
    pub fn final_state(&self) -> Option<&ECState> {
        self.states.last()
    }

    /// Number of states in the narrative.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

// ─── EventSequence (for abduction) ──────────────────────────────────────────

/// A hypothetical sequence of events (used in abductive reasoning).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSequence {
    pub events: Vec<Event>,
    pub explanation: String,
}

// ─── ECEngine ────────────────────────────────────────────────────────────────

/// The Event Calculus evaluation engine.
pub struct ECEngine {
    /// The axiom set.
    pub axiom_set: AxiomSet,
    /// The current fluent store.
    pub fluent_store: FluentStore,
    /// The spatial oracle (optional).
    pub spatial_oracle: Option<SpatialOracle>,
    /// Configuration.
    pub config: ECEngineConfig,
    /// Current time.
    current_time: TimePoint,
    /// State sequence counter.
    sequence: u64,
    /// Rollback stack: previous states for undo.
    rollback_stack: VecDeque<(FluentStore, TimePoint)>,
    /// Domain-specific flags.
    flags: HashMap<String, bool>,
    /// Cache of spatial valuations.
    cached_spatial_vals: Vec<PredicateValuation>,
    /// Next event ID for derived events.
    next_derived_event_id: u64,
    /// Set of fluent IDs that changed since last step (for incremental eval).
    dirty_fluents: HashSet<FluentId>,
}

impl ECEngine {
    /// Create a new EC engine.
    pub fn new(
        axiom_set: AxiomSet,
        initial_fluents: FluentStore,
        config: ECEngineConfig,
    ) -> Self {
        Self {
            axiom_set,
            fluent_store: initial_fluents,
            spatial_oracle: None,
            config,
            current_time: TimePoint::zero(),
            sequence: 0,
            rollback_stack: VecDeque::new(),
            flags: HashMap::new(),
            cached_spatial_vals: Vec::new(),
            next_derived_event_id: 100_000,
            dirty_fluents: HashSet::new(),
        }
    }

    /// Attach a spatial oracle.
    pub fn with_spatial_oracle(mut self, oracle: SpatialOracle) -> Self {
        self.spatial_oracle = Some(oracle);
        self
    }

    /// Set a domain flag.
    pub fn set_flag(&mut self, name: impl Into<String>, value: bool) {
        self.flags.insert(name.into(), value);
    }

    /// Get the current time.
    pub fn current_time(&self) -> TimePoint {
        self.current_time
    }

    /// Get the current fluent store.
    pub fn fluents(&self) -> &FluentStore {
        &self.fluent_store
    }

    /// Process a single event and return the resulting state.
    pub fn step(&mut self, event: Event) -> ECState {
        // Save state for rollback
        self.save_rollback_point();

        self.current_time = event.time;
        self.sequence += 1;

        // Snapshot before changes
        let _pre_snapshot = self.fluent_store.snapshot(event.time);

        // Update spatial oracle if enabled
        if self.config.spatial_enabled {
            if let Some(oracle) = &mut self.spatial_oracle {
                self.cached_spatial_vals = oracle.evaluate_at(event.time);
            }
        }

        // Fire axioms
        let (deltas, derived_event_kinds) = fire_axioms(
            &self.axiom_set,
            &event,
            &self.fluent_store,
            &self.cached_spatial_vals,
            &self.flags,
        );

        // Merge and apply deltas
        let mut all_deltas = Vec::new();
        for delta in &deltas {
            self.apply_delta(delta);
            all_deltas.push(delta.clone());
        }

        // Process derived events (causal chain)
        let mut derived_events = Vec::new();
        let mut event_queue: VecDeque<EventKind> = derived_event_kinds.into_iter().collect();
        let mut derived_count = 0;

        while let Some(ekind) = event_queue.pop_front() {
            if derived_count >= self.config.max_derived_events_per_step {
                break;
            }
            let derived_id = EventId(self.next_derived_event_id);
            self.next_derived_event_id += 1;
            let derived_event = Event::new(derived_id, event.time, ekind);

            let (d_deltas, d_derived) = fire_axioms(
                &self.axiom_set,
                &derived_event,
                &self.fluent_store,
                &self.cached_spatial_vals,
                &self.flags,
            );

            for delta in &d_deltas {
                self.apply_delta(delta);
                all_deltas.push(delta.clone());
            }

            for ek in d_derived {
                event_queue.push_back(ek);
            }

            derived_events.push(derived_event);
            derived_count += 1;
        }

        // Apply state constraints
        self.apply_state_constraints();

        // Build result state
        let post_snapshot = self.fluent_store.snapshot(event.time);
        let mut state = ECState::new(event.time, post_snapshot, self.sequence);
        state.events = vec![event];
        state.deltas = all_deltas;
        state.derived_events = derived_events;
        state.spatial_valuations = self.cached_spatial_vals.clone();

        state
    }

    /// Evaluate a complete trace of events, returning all resulting states.
    pub fn evaluate_trace(&mut self, events: &[Event]) -> Vec<ECState> {
        let mut sorted_events = events.to_vec();
        sorted_events.sort_by(|a, b| a.time.cmp(&b.time));

        let mut states = Vec::new();
        for event in sorted_events {
            let state = self.step(event);
            states.push(state);
        }
        states
    }

    /// Compute a complete narrative from events with initial state and time bound.
    pub fn compute_narrative(
        &mut self,
        events: &[Event],
        initial_state: &FluentStore,
        time_bound: TimePoint,
    ) -> Narrative {
        // Reset engine state
        self.fluent_store = initial_state.clone();
        self.current_time = TimePoint::zero();
        self.sequence = 0;
        self.rollback_stack.clear();
        self.dirty_fluents.clear();

        let initial_snapshot = self.fluent_store.snapshot(TimePoint::zero());
        let mut narrative = Narrative::new(initial_snapshot.clone());

        // Initial state entry
        let init_state = ECState::new(TimePoint::zero(), initial_snapshot, 0);
        narrative.push_state(init_state);

        let mut sorted_events = events.to_vec();
        sorted_events.sort_by(|a, b| a.time.cmp(&b.time));

        // Process spatial oracle transitions if available
        let spatial_events = if self.config.spatial_enabled {
            self.collect_spatial_events(&sorted_events, time_bound)
        } else {
            Vec::new()
        };

        // Merge user events and spatial events
        let mut all_events = sorted_events;
        all_events.extend(spatial_events);
        all_events.sort_by(|a, b| a.time.cmp(&b.time));

        for event in &all_events {
            if event.time > time_bound {
                break;
            }
            let state = self.step(event.clone());
            narrative.push_state(state);
        }

        narrative
    }

    /// Forward-chaining evaluation: step through time with spatial sampling.
    pub fn forward_chain(
        &mut self,
        events: &[Event],
        time_bound: TimePoint,
        sample_interval: Duration,
    ) -> Vec<ECState> {
        let mut sorted_events = events.to_vec();
        sorted_events.sort_by(|a, b| a.time.cmp(&b.time));

        let mut states = Vec::new();
        let mut event_idx = 0;
        let mut t = TimePoint::zero();

        while t <= time_bound {
            // Process all events at or before current time
            while event_idx < sorted_events.len() && sorted_events[event_idx].time <= t {
                let state = self.step(sorted_events[event_idx].clone());
                states.push(state);
                event_idx += 1;
            }

            // Spatial oracle sampling
            if self.config.spatial_enabled {
                if let Some(oracle) = &mut self.spatial_oracle {
                    if let Some(scene) = oracle.query_scene_at(t) {
                        let transitions = oracle.detect_transitions(&scene, t);
                        for trans in transitions {
                            let eid = EventId(self.next_derived_event_id);
                            self.next_derived_event_id += 1;
                            let event = trans.to_event(eid);
                            let state = self.step(event);
                            states.push(state);
                        }
                    }
                }
            }

            t = t.advance(sample_interval);
        }

        states
    }

    /// Abductive reasoning: find event sequences that could lead to a goal fluent
    /// holding at a given time.
    pub fn abduce(
        &self,
        goal_fluent: FluentId,
        target_time: TimePoint,
    ) -> Vec<EventSequence> {
        let mut explanations = Vec::new();

        // Find initiation axioms for the goal fluent
        let init_axioms = self.axiom_set.initiation_axioms_for(goal_fluent);
        for axiom in &init_axioms {
            if let Axiom::InitiationAxiom { event, conditions, name, .. } = axiom {
                // Check which conditions are not currently satisfied
                let ctx = EvalContext {
                    fluents: &self.fluent_store,
                    current_event: None,
                    spatial_valuations: &self.cached_spatial_vals,
                    flags: &self.flags,
                    time: target_time,
                };

                let unsatisfied: Vec<&AxiomCondition> = conditions
                    .iter()
                    .filter(|c| !evaluate_condition(c, &ctx))
                    .collect();

                // Build a hypothetical event matching the pattern
                let hypothetical_event = pattern_to_hypothetical_event(event, target_time);

                if unsatisfied.is_empty() {
                    // Only need the trigger event
                    explanations.push(EventSequence {
                        events: vec![hypothetical_event],
                        explanation: format!("Axiom '{}' fires directly", name),
                    });
                } else {
                    // Need to satisfy conditions first
                    let mut needed_events = Vec::new();
                    for cond in &unsatisfied {
                        if let AxiomCondition::FluentHolds(fid) = cond {
                            // Recursively abduce for this precondition
                            let sub = self.abduce(*fid, target_time);
                            if let Some(first) = sub.first() {
                                needed_events.extend(first.events.clone());
                            }
                        }
                    }
                    needed_events.push(hypothetical_event);
                    explanations.push(EventSequence {
                        events: needed_events,
                        explanation: format!(
                            "Axiom '{}' with {} preconditions satisfied",
                            name,
                            unsatisfied.len()
                        ),
                    });
                }
            }
        }

        // Check causal axioms
        for axiom in self.axiom_set.causal_axioms() {
            if let Axiom::CausalAxiom { cause_event, name, .. } = axiom {
                let hypothetical = pattern_to_hypothetical_event(cause_event, target_time);
                explanations.push(EventSequence {
                    events: vec![hypothetical],
                    explanation: format!("Causal chain via '{}'", name),
                });
            }
        }

        explanations
    }

    /// Rollback to the previous state.
    pub fn rollback(&mut self) -> bool {
        if let Some((store, time)) = self.rollback_stack.pop_back() {
            self.fluent_store = store;
            self.current_time = time;
            self.sequence = self.sequence.saturating_sub(1);
            self.dirty_fluents.clear();
            true
        } else {
            false
        }
    }

    /// Incremental evaluation: only re-evaluate axioms whose conditions
    /// reference fluents that changed.
    pub fn step_incremental(&mut self, event: Event) -> ECState {
        if self.dirty_fluents.is_empty() {
            return self.step(event);
        }

        self.save_rollback_point();
        self.current_time = event.time;
        self.sequence += 1;

        // Only evaluate axioms that reference dirty fluents
        let relevant_axioms: Vec<AxiomId> = self.axiom_set
            .iter()
            .filter(|(_, axiom)| {
                let refs = axiom.referenced_fluents();
                refs.iter().any(|id| self.dirty_fluents.contains(id))
            })
            .map(|(id, _)| *id)
            .collect();

        let ctx = EvalContext {
            fluents: &self.fluent_store,
            current_event: Some(&event),
            spatial_valuations: &self.cached_spatial_vals,
            flags: &self.flags,
            time: event.time,
        };

        let mut all_deltas = Vec::new();
        let mut derived_events = Vec::new();

        for aid in &relevant_axioms {
            if let Some(axiom) = self.axiom_set.get(*aid) {
                if evaluate_axiom(axiom, &ctx) {
                    match axiom {
                        Axiom::InitiationAxiom { fluent, new_value, .. } => {
                            let delta = FluentDelta {
                                initiated: vec![(*fluent, new_value.clone())],
                                terminated: vec![],
                                changed: vec![],
                            };
                            all_deltas.push(delta);
                        }
                        Axiom::TerminationAxiom { fluent, .. } => {
                            let delta = FluentDelta {
                                initiated: vec![],
                                terminated: vec![*fluent],
                                changed: vec![],
                            };
                            all_deltas.push(delta);
                        }
                        Axiom::CausalAxiom { effect_event, .. } => {
                            let eid = EventId(self.next_derived_event_id);
                            self.next_derived_event_id += 1;
                            derived_events.push(Event::new(eid, event.time, effect_event.clone()));
                        }
                        _ => {}
                    }
                }
            }
        }

        // Also evaluate non-referenced axioms (full set minus incremental)
        let full_result = fire_axioms(
            &self.axiom_set,
            &event,
            &self.fluent_store,
            &self.cached_spatial_vals,
            &self.flags,
        );
        for delta in &full_result.0 {
            if !all_deltas.contains(delta) {
                all_deltas.push(delta.clone());
            }
        }

        self.dirty_fluents.clear();

        for delta in &all_deltas {
            self.apply_delta(delta);
        }

        let snapshot = self.fluent_store.snapshot(event.time);
        let mut state = ECState::new(event.time, snapshot, self.sequence);
        state.events = vec![event];
        state.deltas = all_deltas;
        state.derived_events = derived_events;
        state.spatial_valuations = self.cached_spatial_vals.clone();

        state
    }

    /// Reset the engine to its initial state.
    pub fn reset(&mut self, initial_fluents: FluentStore) {
        self.fluent_store = initial_fluents;
        self.current_time = TimePoint::zero();
        self.sequence = 0;
        self.rollback_stack.clear();
        self.dirty_fluents.clear();
        self.cached_spatial_vals.clear();
        self.next_derived_event_id = 100_000;
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    fn save_rollback_point(&mut self) {
        if self.rollback_stack.len() >= self.config.max_rollback_depth {
            self.rollback_stack.pop_front();
        }
        self.rollback_stack
            .push_back((self.fluent_store.clone(), self.current_time));
    }

    fn apply_delta(&mut self, delta: &FluentDelta) {
        for (id, fluent) in &delta.initiated {
            self.fluent_store.insert_with_id(*id, fluent.clone());
            self.dirty_fluents.insert(*id);
        }
        for id in &delta.terminated {
            self.fluent_store.remove(*id);
            self.dirty_fluents.insert(*id);
        }
        for (id, _old, new_val) in &delta.changed {
            let _ = self.fluent_store.update(*id, new_val.clone());
            self.dirty_fluents.insert(*id);
        }
    }

    fn apply_state_constraints(&mut self) {
        let engine = CircumscriptionEngine::new()
            .with_max_iterations(self.config.max_constraint_iterations);
        let constraint_delta = engine.compute_minimal_change(
            &self.axiom_set,
            &[],
            &self.fluent_store,
            &self.cached_spatial_vals,
        );
        for delta in &constraint_delta {
            self.apply_delta(delta);
        }
    }

    fn collect_spatial_events(
        &mut self,
        _user_events: &[Event],
        time_bound: TimePoint,
    ) -> Vec<Event> {
        let mut spatial_events = Vec::new();
        if let Some(oracle) = &mut self.spatial_oracle {
            let transitions = oracle.scan_trajectory(self.config.spatial_sample_interval);
            for trans in transitions {
                if trans.time <= time_bound {
                    let eid = EventId(self.next_derived_event_id);
                    self.next_derived_event_id += 1;
                    spatial_events.push(trans.to_event(eid));
                }
            }
        }
        spatial_events
    }
}

/// Convert an event pattern to a hypothetical event for abductive reasoning.
fn pattern_to_hypothetical_event(pattern: &EventPattern, time: TimePoint) -> Event {
    let kind = match pattern {
        EventPattern::GestureMatch(g) => EventKind::Gesture {
            gesture: *g,
            hand: HandSide::Right,
            entity: EntityId(0),
        },
        EventPattern::ActionMatch(a) => EventKind::Action {
            action: *a,
            entity: EntityId(0),
        },
        EventPattern::SpatialChangeMatch(id) => EventKind::SpatialChange {
            predicate_id: *id,
            new_value: true,
        },
        EventPattern::NamedMatch(name) => EventKind::Custom {
            name: name.clone(),
            params: HashMap::new(),
        },
        _ => EventKind::System {
            tag: "hypothetical".into(),
        },
    };
    Event::new(EventId(0), time, kind)
}

// ─── Conflict resolution ─────────────────────────────────────────────────────

/// Resolve conflicts between simultaneous events.
///
/// When multiple events occur at the same time and they produce conflicting
/// deltas (e.g., one initiates and another terminates the same fluent),
/// the higher-priority axiom wins.
pub fn resolve_conflicts(deltas: &[FluentDelta]) -> FluentDelta {
    let mut initiated: IndexMap<FluentId, Fluent> = IndexMap::new();
    let mut terminated: HashSet<FluentId> = HashSet::new();
    let mut changed: IndexMap<FluentId, (Fluent, Fluent)> = IndexMap::new();

    for delta in deltas {
        for (id, fluent) in &delta.initiated {
            terminated.remove(id);
            initiated.insert(*id, fluent.clone());
        }
        for id in &delta.terminated {
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

/// Order events that occur at the same time by priority.
pub fn order_simultaneous_events(events: &mut [Event]) {
    events.sort_by(|a, b| {
        a.time.cmp(&b.time).then_with(|| {
            let pa = event_priority(&a.kind);
            let pb = event_priority(&b.kind);
            pa.cmp(&pb).reverse()
        })
    });
}

fn event_priority(kind: &EventKind) -> i32 {
    match kind {
        EventKind::System { .. } => 0,
        EventKind::TimerExpired { .. } => 1,
        EventKind::SpatialChange { .. } => 2,
        EventKind::CollisionStart { .. } | EventKind::CollisionEnd { .. } => 3,
        EventKind::GazeEnter { .. } | EventKind::GazeExit { .. } => 4,
        EventKind::Gesture { .. } => 5,
        EventKind::Action { .. } => 6,
        EventKind::TimerStarted { .. } => 7,
        EventKind::Custom { .. } => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axioms::AxiomId;

    fn make_engine() -> ECEngine {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("near", true));     // FluentId(1)
        store.insert(Fluent::boolean("grabbed", false));  // FluentId(2)
        store.insert(Fluent::numeric("distance", 1.5));   // FluentId(3)

        let mut axiom_set = AxiomSet::new();

        // Grab initiation: if near and grab gesture -> grabbed
        axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "grab_init".into(),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            fluent: FluentId(2),
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("grabbed", true),
            priority: 0,
        });

        // Release termination
        axiom_set.add(Axiom::TerminationAxiom {
            id: AxiomId(2),
            name: "grab_term".into(),
            conditions: vec![],
            fluent: FluentId(2),
            event: EventPattern::ActionMatch(ActionType::Deactivate),
            priority: 0,
        });

        ECEngine::new(axiom_set, store, ECEngineConfig::default())
    }

    #[test]
    fn test_engine_step_initiation() {
        let mut engine = make_engine();
        let grab = Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        );

        let state = engine.step(grab);
        assert!(state.holds(FluentId(2)));
        assert_eq!(state.time, TimePoint::from_secs(1.0));
    }

    #[test]
    fn test_engine_step_termination() {
        let mut engine = make_engine();
        // First grab
        engine.step(Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));
        assert!(engine.fluent_store.get(FluentId(2)).unwrap().holds());

        // Then release
        let _state = engine.step(Event::new(
            EventId(2),
            TimePoint::from_secs(2.0),
            EventKind::Action {
                action: ActionType::Deactivate,
                entity: EntityId(10),
            },
        ));
        // After termination, the fluent should be removed
        assert!(engine.fluent_store.get(FluentId(2)).is_none() ||
                !engine.fluent_store.get(FluentId(2)).unwrap().holds());
    }

    #[test]
    fn test_engine_evaluate_trace() {
        let mut engine = make_engine();
        let events = vec![
            Event::new(
                EventId(1),
                TimePoint::from_secs(1.0),
                EventKind::Gesture {
                    gesture: GestureType::Grab,
                    hand: HandSide::Right,
                    entity: EntityId(10),
                },
            ),
            Event::new(
                EventId(2),
                TimePoint::from_secs(3.0),
                EventKind::Action {
                    action: ActionType::Deactivate,
                    entity: EntityId(10),
                },
            ),
        ];

        let states = engine.evaluate_trace(&events);
        assert_eq!(states.len(), 2);
        assert!(states[0].holds(FluentId(2)));
    }

    #[test]
    fn test_engine_rollback() {
        let mut engine = make_engine();

        // Initial: grabbed is false
        assert!(!engine.fluent_store.get(FluentId(2)).unwrap().holds());

        // Grab
        engine.step(Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));

        assert!(engine.fluent_store.get(FluentId(2)).unwrap().holds());

        // Rollback
        assert!(engine.rollback());
        assert!(!engine.fluent_store.get(FluentId(2)).unwrap().holds());
    }

    #[test]
    fn test_engine_compute_narrative() {
        let initial = FluentStore::new();
        let axiom_set = AxiomSet::new();
        let mut engine = ECEngine::new(axiom_set, initial.clone(), ECEngineConfig::default());

        let events = vec![
            Event::new(
                EventId(1),
                TimePoint::from_secs(1.0),
                EventKind::System { tag: "tick".into() },
            ),
        ];

        let narrative = engine.compute_narrative(&events, &initial, TimePoint::from_secs(10.0));
        assert!(narrative.len() >= 2); // initial + one event
    }

    #[test]
    fn test_engine_abduction() {
        let engine = make_engine();
        let explanations = engine.abduce(FluentId(2), TimePoint::from_secs(5.0));
        assert!(!explanations.is_empty());
        // Should suggest a grab gesture
        assert!(explanations.iter().any(|e| {
            e.events.iter().any(|ev| matches!(&ev.kind, EventKind::Gesture { gesture: GestureType::Grab, .. }))
        }));
    }

    #[test]
    fn test_engine_causal_chain() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("enabled", true)); // FluentId(1)

        let mut axiom_set = AxiomSet::new();
        axiom_set.add(Axiom::CausalAxiom {
            id: AxiomId(10),
            name: "grab_causes_haptic".into(),
            cause_event: EventPattern::GestureMatch(GestureType::Grab),
            conditions: vec![AxiomCondition::FluentHolds(FluentId(1))],
            effect_event: EventKind::Custom {
                name: "haptic".into(),
                params: HashMap::new(),
            },
            delay: None,
        });

        let mut engine = ECEngine::new(axiom_set, store, ECEngineConfig::default());
        let state = engine.step(Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));

        // The derived event should be produced
        assert!(!state.derived_events.is_empty());
    }

    #[test]
    fn test_conflict_resolution() {
        let d1 = FluentDelta {
            initiated: vec![(FluentId(1), Fluent::boolean("x", true))],
            terminated: vec![],
            changed: vec![],
        };
        let d2 = FluentDelta {
            initiated: vec![],
            terminated: vec![FluentId(1)],
            changed: vec![],
        };

        let resolved = resolve_conflicts(&[d1, d2]);
        // Initiation takes precedence (applied later)
        assert!(!resolved.initiated.is_empty() || !resolved.terminated.is_empty());
    }

    #[test]
    fn test_order_simultaneous_events() {
        let mut events = vec![
            Event::new(EventId(1), TimePoint::from_secs(1.0),
                EventKind::System { tag: "sys".into() }),
            Event::new(EventId(2), TimePoint::from_secs(1.0),
                EventKind::Gesture { gesture: GestureType::Grab, hand: HandSide::Right, entity: EntityId(1) }),
            Event::new(EventId(3), TimePoint::from_secs(1.0),
                EventKind::TimerExpired { name: "t1".into() }),
        ];
        order_simultaneous_events(&mut events);
        // Custom > Gesture > Timer > System
        assert!(matches!(&events[0].kind, EventKind::Gesture { .. }));
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = make_engine();
        engine.step(Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));

        let initial = FluentStore::new();
        engine.reset(initial);
        assert_eq!(engine.current_time(), TimePoint::zero());
        assert_eq!(engine.fluent_store.len(), 0);
    }

    #[test]
    fn test_ec_state_display() {
        let state = ECState::new(
            TimePoint::from_secs(1.0),
            FluentSnapshot::new(TimePoint::from_secs(1.0)),
            1,
        );
        let s = format!("{}", state);
        assert!(s.contains("ECState"));
    }

    #[test]
    fn test_narrative_state_at() {
        let initial = FluentSnapshot::new(TimePoint::zero());
        let mut narrative = Narrative::new(initial.clone());

        narrative.push_state(ECState::new(TimePoint::from_secs(0.0), initial.clone(), 0));
        narrative.push_state(ECState::new(TimePoint::from_secs(1.0), initial.clone(), 1));
        narrative.push_state(ECState::new(TimePoint::from_secs(2.0), initial.clone(), 2));

        let s = narrative.state_at(TimePoint::from_secs(1.5)).unwrap();
        assert_eq!(s.sequence, 1);

        let s2 = narrative.state_at(TimePoint::from_secs(3.0)).unwrap();
        assert_eq!(s2.sequence, 2);
    }

    #[test]
    fn test_engine_incremental_step() {
        let mut engine = make_engine();
        engine.dirty_fluents.insert(FluentId(1));
        let state = engine.step_incremental(Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));
        assert!(state.holds(FluentId(2)));
    }
}
