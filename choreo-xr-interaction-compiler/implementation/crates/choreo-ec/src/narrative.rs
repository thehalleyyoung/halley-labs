//! Narrative management for Event Calculus traces.
//!
//! A narrative captures the complete temporal evolution of an interaction:
//! which events occurred, how fluents changed, and whether the execution
//! respected all causal, temporal, and spatial constraints.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::axioms::*;
use crate::fluent::*;
use crate::local_types::*;

// ─── NarrativeDoc ────────────────────────────────────────────────────────────

/// A complete narrative document: events, fluent history, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeDoc {
    pub name: String,
    pub events: Vec<Event>,
    pub fluent_history: FluentHistory,
    pub metadata: HashMap<String, String>,
    pub time_span: Option<TimeInterval>,
}

impl NarrativeDoc {
    pub fn new(name: impl Into<String>, initial: FluentSnapshot) -> Self {
        Self {
            name: name.into(),
            events: Vec::new(),
            fluent_history: FluentHistory::new(initial),
            metadata: HashMap::new(),
            time_span: None,
        }
    }

    /// Number of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Number of snapshots in history.
    pub fn snapshot_count(&self) -> usize {
        self.fluent_history.snapshot_count()
    }

    /// Get all event kinds that appear in this narrative.
    pub fn event_kinds(&self) -> HashSet<String> {
        self.events
            .iter()
            .map(|e| format!("{:?}", std::mem::discriminant(&e.kind)))
            .collect()
    }

    /// Get events in a time interval.
    pub fn events_in(&self, interval: &TimeInterval) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| interval.contains(e.time))
            .collect()
    }

    /// Get the final fluent values.
    pub fn final_state(&self) -> Option<&FluentSnapshot> {
        self.fluent_history.latest_snapshot()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, ECError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ECError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, ECError> {
        serde_json::from_str(json)
            .map_err(|e| ECError::SerializationError(e.to_string()))
    }
}

// ─── NarrativeBuilder ────────────────────────────────────────────────────────

/// Incremental builder for narratives.
pub struct NarrativeBuilder {
    doc: NarrativeDoc,
    next_event_id: u64,
    current_store: FluentStore,
}

impl NarrativeBuilder {
    pub fn new(name: impl Into<String>, initial_fluents: FluentStore) -> Self {
        let snapshot = initial_fluents.snapshot(TimePoint::zero());
        Self {
            doc: NarrativeDoc::new(name, snapshot),
            next_event_id: 1,
            current_store: initial_fluents,
        }
    }

    /// Add an event and its effects.
    pub fn add_event(&mut self, time: TimePoint, kind: EventKind) -> EventId {
        let id = EventId(self.next_event_id);
        self.next_event_id += 1;
        let event = Event::new(id, time, kind);
        self.doc.events.push(event);
        self.update_time_span(time);
        id
    }

    /// Record a fluent initiation.
    pub fn initiate_fluent(
        &mut self,
        fluent_id: FluentId,
        new_value: Fluent,
        event_id: EventId,
        time: TimePoint,
    ) {
        self.current_store.insert_with_id(fluent_id, new_value.clone());
        self.doc.fluent_history.record_initiation(InitiatedBy {
            fluent_id,
            event_id,
            time,
            new_value,
        });
        self.record_snapshot(time);
    }

    /// Record a fluent termination.
    pub fn terminate_fluent(
        &mut self,
        fluent_id: FluentId,
        event_id: EventId,
        time: TimePoint,
    ) {
        self.current_store.remove(fluent_id);
        self.doc.fluent_history.record_termination(TerminatedBy {
            fluent_id,
            event_id,
            time,
        });
        self.record_snapshot(time);
    }

    /// Record a fluent change.
    pub fn change_fluent(
        &mut self,
        fluent_id: FluentId,
        new_value: Fluent,
        event_id: EventId,
        time: TimePoint,
    ) {
        let _ = self.current_store.update(fluent_id, new_value.clone());
        self.doc.fluent_history.record_initiation(InitiatedBy {
            fluent_id,
            event_id,
            time,
            new_value,
        });
        self.record_snapshot(time);
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.doc.metadata.insert(key.into(), value.into());
    }

    /// Build the final narrative document.
    pub fn build(mut self) -> NarrativeDoc {
        self.doc.events.sort_by(|a, b| a.time.cmp(&b.time));
        self.doc
    }

    fn record_snapshot(&mut self, time: TimePoint) {
        let snapshot = self.current_store.snapshot(time);
        self.doc.fluent_history.record_snapshot(snapshot);
    }

    fn update_time_span(&mut self, time: TimePoint) {
        if let Some(span) = &mut self.doc.time_span {
            if time < span.start { span.start = time; }
            if time > span.end { span.end = time; }
        } else {
            self.doc.time_span = Some(TimeInterval::new(time, time));
        }
    }
}

// ─── NarrativeViolation ──────────────────────────────────────────────────────

/// A violation detected during narrative validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NarrativeViolation {
    /// An event caused a fluent change with no supporting axiom.
    CausalityViolation {
        time: TimePoint,
        event_id: EventId,
        fluent_id: FluentId,
        description: String,
    },
    /// A temporal constraint was violated.
    TemporalViolation {
        time: TimePoint,
        constraint: String,
        description: String,
    },
    /// A spatial predicate was in an inconsistent state.
    SpatialViolation {
        time: TimePoint,
        predicate_id: SpatialPredicateId,
        description: String,
    },
    /// The law of inertia was violated (fluent changed without an event).
    InertiaViolation {
        time: TimePoint,
        fluent_id: FluentId,
        description: String,
    },
    /// A state constraint was not satisfied.
    ConstraintViolation {
        time: TimePoint,
        axiom_name: String,
        description: String,
    },
}

impl NarrativeViolation {
    pub fn time(&self) -> TimePoint {
        match self {
            NarrativeViolation::CausalityViolation { time, .. } => *time,
            NarrativeViolation::TemporalViolation { time, .. } => *time,
            NarrativeViolation::SpatialViolation { time, .. } => *time,
            NarrativeViolation::InertiaViolation { time, .. } => *time,
            NarrativeViolation::ConstraintViolation { time, .. } => *time,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            NarrativeViolation::CausalityViolation { description, .. } => description,
            NarrativeViolation::TemporalViolation { description, .. } => description,
            NarrativeViolation::SpatialViolation { description, .. } => description,
            NarrativeViolation::InertiaViolation { description, .. } => description,
            NarrativeViolation::ConstraintViolation { description, .. } => description,
        }
    }
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate a narrative against an axiom set.
pub fn validate_narrative(
    narrative: &NarrativeDoc,
    axiom_set: &AxiomSet,
) -> Vec<NarrativeViolation> {
    let mut violations = Vec::new();

    // Check inertia: fluent values should only change when an event occurs
    validate_inertia(narrative, &mut violations);

    // Check causality: every fluent change should be supported by an axiom
    validate_causality(narrative, axiom_set, &mut violations);

    // Check temporal ordering
    validate_temporal_ordering(narrative, &mut violations);

    // Check state constraints
    validate_state_constraints(narrative, axiom_set, &mut violations);

    violations.sort_by(|a, b| a.time().cmp(&b.time()));
    violations
}

fn validate_inertia(narrative: &NarrativeDoc, violations: &mut Vec<NarrativeViolation>) {
    let history = &narrative.fluent_history;
    let snapshots: Vec<&FluentSnapshot> = {
        if let Some(span) = &narrative.time_span {
            history.snapshots_in(span)
        } else {
            return;
        }
    };

    for pair in snapshots.windows(2) {
        let prev = pair[0];
        let curr = pair[1];
        let delta = prev.diff(curr);

        for (fid, _old, _new) in &delta.changed {
            // Check if there's an event between these snapshots
            let has_event = narrative.events.iter().any(|e| {
                e.time.0 >= prev.time.0 && e.time.0 <= curr.time.0
            });
            if !has_event {
                violations.push(NarrativeViolation::InertiaViolation {
                    time: curr.time,
                    fluent_id: *fid,
                    description: format!(
                        "Fluent {} changed between {} and {} without an event",
                        fid, prev.time, curr.time
                    ),
                });
            }
        }
    }
}

fn validate_causality(
    narrative: &NarrativeDoc,
    axiom_set: &AxiomSet,
    violations: &mut Vec<NarrativeViolation>,
) {
    for init in &narrative.fluent_history.initiations {
        let matching_event = narrative.events.iter().find(|e| e.id == init.event_id);
        if matching_event.is_none() {
            violations.push(NarrativeViolation::CausalityViolation {
                time: init.time,
                event_id: init.event_id,
                fluent_id: init.fluent_id,
                description: format!(
                    "Fluent {} initiated by event {} which does not exist in narrative",
                    init.fluent_id, init.event_id
                ),
            });
            continue;
        }

        // Check if any axiom supports this initiation
        let init_axioms = axiom_set.initiation_axioms_for(init.fluent_id);
        if init_axioms.is_empty() {
            violations.push(NarrativeViolation::CausalityViolation {
                time: init.time,
                event_id: init.event_id,
                fluent_id: init.fluent_id,
                description: format!(
                    "No initiation axiom exists for fluent {}",
                    init.fluent_id
                ),
            });
        }
    }
}

fn validate_temporal_ordering(
    narrative: &NarrativeDoc,
    violations: &mut Vec<NarrativeViolation>,
) {
    for pair in narrative.events.windows(2) {
        if pair[0].time > pair[1].time {
            violations.push(NarrativeViolation::TemporalViolation {
                time: pair[1].time,
                constraint: "ordering".into(),
                description: format!(
                    "Event {} at {} occurs before event {} at {} but is listed after it",
                    pair[1].id, pair[1].time, pair[0].id, pair[0].time
                ),
            });
        }
    }
}

fn validate_state_constraints(
    narrative: &NarrativeDoc,
    axiom_set: &AxiomSet,
    violations: &mut Vec<NarrativeViolation>,
) {
    let empty_valuations: Vec<PredicateValuation> = Vec::new();
    let empty_flags: HashMap<String, bool> = HashMap::new();

    for (_, axiom) in axiom_set.iter() {
        if let Axiom::StateConstraint { conditions, fluent, required_value, name, .. } = axiom {
            // Check at each snapshot
            if let Some(span) = &narrative.time_span {
                for snapshot in narrative.fluent_history.snapshots_in(span) {
                    let store = FluentStore::from_snapshot(snapshot);
                    let ctx = EvalContext {
                        fluents: &store,
                        current_event: None,
                        spatial_valuations: &empty_valuations,
                        flags: &empty_flags,
                        time: snapshot.time,
                    };
                    let conds_hold = conditions.iter().all(|c| evaluate_condition(c, &ctx));
                    if conds_hold {
                        let actual = snapshot.get(*fluent);
                        let satisfied = actual.map_or(false, |f| f.holds() == required_value.holds());
                        if !satisfied {
                            violations.push(NarrativeViolation::ConstraintViolation {
                                time: snapshot.time,
                                axiom_name: name.clone(),
                                description: format!(
                                    "State constraint '{}' violated at {}",
                                    name, snapshot.time
                                ),
                            });
                        }
                    }
                }
            }
        }
    }
}

// Helper: build FluentStore from a snapshot for validation.
impl FluentStore {
    pub fn from_snapshot(snapshot: &FluentSnapshot) -> Self {
        let mut store = FluentStore::new();
        for (id, fluent) in &snapshot.values {
            store.insert_with_id(*id, fluent.clone());
        }
        store
    }
}

// ─── Narrative merging ───────────────────────────────────────────────────────

/// Merge two narratives (parallel composition).
pub fn merge_narratives(a: &NarrativeDoc, b: &NarrativeDoc) -> NarrativeDoc {
    let initial = a.fluent_history.initial_state().merge(b.fluent_history.initial_state());
    let mut result = NarrativeDoc::new(
        format!("{}+{}", a.name, b.name),
        initial,
    );

    // Merge events by time
    let mut all_events: Vec<Event> = Vec::new();
    all_events.extend(a.events.clone());
    all_events.extend(b.events.clone());
    all_events.sort_by(|x, y| x.time.cmp(&y.time));
    result.events = all_events;

    // Merge time spans
    match (&a.time_span, &b.time_span) {
        (Some(sa), Some(sb)) => {
            result.time_span = Some(TimeInterval::new(
                if sa.start < sb.start { sa.start } else { sb.start },
                if sa.end > sb.end { sa.end } else { sb.end },
            ));
        }
        (Some(s), None) | (None, Some(s)) => result.time_span = Some(*s),
        (None, None) => {}
    }

    // Merge initiations and terminations
    for init in &a.fluent_history.initiations {
        result.fluent_history.record_initiation(init.clone());
    }
    for init in &b.fluent_history.initiations {
        result.fluent_history.record_initiation(init.clone());
    }
    for term in &a.fluent_history.terminations {
        result.fluent_history.record_termination(term.clone());
    }
    for term in &b.fluent_history.terminations {
        result.fluent_history.record_termination(term.clone());
    }

    // Merge metadata
    result.metadata.extend(a.metadata.clone());
    result.metadata.extend(b.metadata.clone());

    result
}

/// Project a narrative onto a subset of fluents.
pub fn project_narrative(
    narrative: &NarrativeDoc,
    fluent_ids: &HashSet<FluentId>,
) -> NarrativeDoc {
    let initial = narrative.fluent_history.initial_state();
    let projected_initial = FluentSnapshot::with_fluents(
        initial.time,
        initial
            .values
            .iter()
            .filter(|(id, _)| fluent_ids.contains(id))
            .map(|(id, f)| (*id, f.clone()))
            .collect(),
    );

    let mut result = NarrativeDoc::new(
        format!("{}_projected", narrative.name),
        projected_initial,
    );

    // Only include events that affect the projected fluents
    let relevant_event_ids: HashSet<EventId> = narrative
        .fluent_history
        .initiations
        .iter()
        .filter(|init| fluent_ids.contains(&init.fluent_id))
        .map(|init| init.event_id)
        .chain(
            narrative
                .fluent_history
                .terminations
                .iter()
                .filter(|term| fluent_ids.contains(&term.fluent_id))
                .map(|term| term.event_id),
        )
        .collect();

    result.events = narrative
        .events
        .iter()
        .filter(|e| relevant_event_ids.contains(&e.id))
        .cloned()
        .collect();

    // Project initiations/terminations
    for init in &narrative.fluent_history.initiations {
        if fluent_ids.contains(&init.fluent_id) {
            result.fluent_history.record_initiation(init.clone());
        }
    }
    for term in &narrative.fluent_history.terminations {
        if fluent_ids.contains(&term.fluent_id) {
            result.fluent_history.record_termination(term.clone());
        }
    }

    result.time_span = narrative.time_span;
    result
}

// ─── NarrativeQuery ──────────────────────────────────────────────────────────

/// Query interface for narratives.
pub struct NarrativeQuery<'a> {
    doc: &'a NarrativeDoc,
}

impl<'a> NarrativeQuery<'a> {
    pub fn new(doc: &'a NarrativeDoc) -> Self {
        Self { doc }
    }

    /// When did a fluent first become true?
    pub fn first_holds(&self, fluent_id: FluentId) -> Option<TimePoint> {
        first_holds_after(&self.doc.fluent_history, fluent_id, TimePoint::zero())
    }

    /// When was a fluent last terminated?
    pub fn last_terminated(&self, fluent_id: FluentId) -> Option<TimePoint> {
        self.doc
            .fluent_history
            .terminations
            .iter()
            .rev()
            .find(|t| t.fluent_id == fluent_id)
            .map(|t| t.time)
    }

    /// How long was a fluent continuously held?
    pub fn max_holding_duration(&self, fluent_id: FluentId) -> Duration {
        let intervals = self.doc.fluent_history.holding_intervals(fluent_id);
        intervals
            .iter()
            .map(|i| i.duration())
            .max()
            .unwrap_or(Duration::zero())
    }

    /// How many times was a fluent initiated?
    pub fn initiation_count(&self, fluent_id: FluentId) -> usize {
        self.doc
            .fluent_history
            .initiations
            .iter()
            .filter(|i| i.fluent_id == fluent_id)
            .count()
    }

    /// How many times was a fluent terminated?
    pub fn termination_count(&self, fluent_id: FluentId) -> usize {
        self.doc
            .fluent_history
            .terminations
            .iter()
            .filter(|t| t.fluent_id == fluent_id)
            .count()
    }

    /// Find all events of a specific kind.
    pub fn events_matching(&self, pattern: &EventPattern) -> Vec<&Event> {
        self.doc
            .events
            .iter()
            .filter(|e| pattern.matches(&e.kind))
            .collect()
    }

    /// Did a fluent hold at a specific time?
    pub fn held_at(&self, fluent_id: FluentId, time: TimePoint) -> bool {
        holds_at(&self.doc.fluent_history, fluent_id, time)
    }

    /// Get the total duration the narrative covers.
    pub fn total_duration(&self) -> Duration {
        self.doc
            .time_span
            .map_or(Duration::zero(), |s| s.duration())
    }

    /// Get all fluent IDs that were ever initiated.
    pub fn initiated_fluents(&self) -> HashSet<FluentId> {
        self.doc
            .fluent_history
            .initiations
            .iter()
            .map(|i| i.fluent_id)
            .collect()
    }

    /// Get all fluent IDs that were ever terminated.
    pub fn terminated_fluents(&self) -> HashSet<FluentId> {
        self.doc
            .fluent_history
            .terminations
            .iter()
            .map(|t| t.fluent_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_narrative() -> NarrativeDoc {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("near", true));    // FluentId(1)
        store.insert(Fluent::boolean("grabbed", false)); // FluentId(2)

        let mut builder = NarrativeBuilder::new("test", store);
        let eid1 = builder.add_event(
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        );
        builder.initiate_fluent(
            FluentId(2),
            Fluent::boolean("grabbed", true),
            eid1,
            TimePoint::from_secs(1.0),
        );

        let eid2 = builder.add_event(
            TimePoint::from_secs(3.0),
            EventKind::Action {
                action: ActionType::Deactivate,
                entity: EntityId(10),
            },
        );
        builder.terminate_fluent(FluentId(2), eid2, TimePoint::from_secs(3.0));

        builder.build()
    }

    #[test]
    fn test_narrative_builder() {
        let doc = make_test_narrative();
        assert_eq!(doc.event_count(), 2);
        assert!(doc.snapshot_count() >= 1);
    }

    #[test]
    fn test_narrative_query_first_holds() {
        let doc = make_test_narrative();
        let query = NarrativeQuery::new(&doc);
        let first = query.first_holds(FluentId(2));
        assert!(first.is_some());
        assert!((first.unwrap().0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_narrative_query_last_terminated() {
        let doc = make_test_narrative();
        let query = NarrativeQuery::new(&doc);
        let last = query.last_terminated(FluentId(2));
        assert!(last.is_some());
        assert!((last.unwrap().0 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_narrative_query_initiation_count() {
        let doc = make_test_narrative();
        let query = NarrativeQuery::new(&doc);
        assert_eq!(query.initiation_count(FluentId(2)), 1);
        assert_eq!(query.termination_count(FluentId(2)), 1);
    }

    #[test]
    fn test_narrative_query_held_at() {
        let doc = make_test_narrative();
        let query = NarrativeQuery::new(&doc);
        assert!(query.held_at(FluentId(2), TimePoint::from_secs(2.0)));
        assert!(!query.held_at(FluentId(2), TimePoint::from_secs(4.0)));
    }

    #[test]
    fn test_narrative_validation_clean() {
        let doc = make_test_narrative();
        let mut axiom_set = AxiomSet::new();
        axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "grab".into(),
            conditions: vec![],
            fluent: FluentId(2),
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("grabbed", true),
            priority: 0,
        });

        let violations = validate_narrative(&doc, &axiom_set);
        // Should have no inertia violations
        let inertia: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, NarrativeViolation::InertiaViolation { .. }))
            .collect();
        assert!(inertia.is_empty());
    }

    #[test]
    fn test_narrative_merge() {
        let doc1 = make_test_narrative();

        let mut store2 = FluentStore::new();
        store2.insert(Fluent::boolean("visible", false));
        let mut builder2 = NarrativeBuilder::new("test2", store2);
        let eid = builder2.add_event(
            TimePoint::from_secs(2.0),
            EventKind::System { tag: "show".into() },
        );
        builder2.initiate_fluent(
            FluentId(3),
            Fluent::boolean("visible", true),
            eid,
            TimePoint::from_secs(2.0),
        );
        let doc2 = builder2.build();

        let merged = merge_narratives(&doc1, &doc2);
        assert_eq!(merged.event_count(), 3);
    }

    #[test]
    fn test_narrative_projection() {
        let doc = make_test_narrative();
        let mut fluents = HashSet::new();
        fluents.insert(FluentId(2));

        let projected = project_narrative(&doc, &fluents);
        // Should only contain events related to FluentId(2)
        assert!(projected.event_count() <= doc.event_count());
    }

    #[test]
    fn test_narrative_serialization() {
        let doc = make_test_narrative();
        let json = doc.to_json().unwrap();
        let deserialized = NarrativeDoc::from_json(&json).unwrap();
        assert_eq!(deserialized.event_count(), doc.event_count());
        assert_eq!(deserialized.name, doc.name);
    }

    #[test]
    fn test_narrative_events_in() {
        let doc = make_test_narrative();
        let interval = TimeInterval::new(
            TimePoint::from_secs(0.5),
            TimePoint::from_secs(2.0),
        );
        let events = doc.events_in(&interval);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_narrative_query_events_matching() {
        let doc = make_test_narrative();
        let query = NarrativeQuery::new(&doc);
        let grabs = query.events_matching(&EventPattern::GestureMatch(GestureType::Grab));
        assert_eq!(grabs.len(), 1);
    }

    #[test]
    fn test_narrative_violation_time() {
        let v = NarrativeViolation::InertiaViolation {
            time: TimePoint::from_secs(5.0),
            fluent_id: FluentId(1),
            description: "test".into(),
        };
        assert_eq!(v.time(), TimePoint::from_secs(5.0));
    }

    #[test]
    fn test_narrative_query_initiated_fluents() {
        let doc = make_test_narrative();
        let query = NarrativeQuery::new(&doc);
        let initiated = query.initiated_fluents();
        assert!(initiated.contains(&FluentId(2)));
    }

    #[test]
    fn test_narrative_builder_metadata() {
        let store = FluentStore::new();
        let mut builder = NarrativeBuilder::new("meta_test", store);
        builder.set_metadata("version", "1.0");
        builder.set_metadata("author", "test");
        let doc = builder.build();
        assert_eq!(doc.metadata.get("version").unwrap(), "1.0");
    }
}
