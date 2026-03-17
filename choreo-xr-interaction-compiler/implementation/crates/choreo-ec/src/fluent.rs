//! Event Calculus fluents — time-varying properties.
//!
//! Fluents are the fundamental state predicates in the Event Calculus.
//! They are initiated and terminated by events, and the *commonsense law of
//! inertia* propagates their values through time unless explicitly changed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::local_types::*;

// ─── FluentId ────────────────────────────────────────────────────────────────

/// Unique identifier for a fluent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FluentId(pub u64);

impl FluentId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for FluentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fluent_{}", self.0)
    }
}

// ─── Fluent ──────────────────────────────────────────────────────────────────

/// A fluent: a time-varying property in the Event Calculus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Fluent {
    /// A named boolean property.
    BooleanFluent {
        name: String,
        value: bool,
    },
    /// A named numeric property.
    NumericFluent {
        name: String,
        value: f64,
    },
    /// A spatial predicate (evaluated by the spatial oracle).
    SpatialFluent {
        predicate: SpatialPredicateId,
        value: bool,
    },
    /// A countdown timer.
    TimerFluent {
        name: String,
        remaining: Duration,
    },
    /// State within a state machine / automaton.
    StateFluent {
        automaton_id: String,
        state: StateId,
    },
    /// Arbitrary dynamic value.
    CustomFluent {
        name: String,
        value: Value,
    },
}

impl Fluent {
    /// Canonical name of the fluent used for indexing and display.
    pub fn name(&self) -> String {
        match self {
            Fluent::BooleanFluent { name, .. } => name.clone(),
            Fluent::NumericFluent { name, .. } => name.clone(),
            Fluent::SpatialFluent { predicate, .. } => format!("spatial_{}", predicate),
            Fluent::TimerFluent { name, .. } => format!("timer_{}", name),
            Fluent::StateFluent { automaton_id, .. } => format!("state_{}", automaton_id),
            Fluent::CustomFluent { name, .. } => name.clone(),
        }
    }

    /// Whether the fluent is considered "truthy" for boolean evaluation.
    pub fn holds(&self) -> bool {
        match self {
            Fluent::BooleanFluent { value, .. } => *value,
            Fluent::NumericFluent { value, .. } => *value != 0.0,
            Fluent::SpatialFluent { value, .. } => *value,
            Fluent::TimerFluent { remaining, .. } => remaining.0 > 0.0,
            Fluent::StateFluent { .. } => true,
            Fluent::CustomFluent { value, .. } => {
                match value {
                    Value::Bool(b) => *b,
                    Value::Int(i) => *i != 0,
                    Value::Float(f) => f.0 != 0.0,
                    Value::String(s) => !s.is_empty(),
                    Value::Entity(_) => true,
                    Value::Null => false,
                }
            }
        }
    }

    /// Create a boolean fluent.
    pub fn boolean(name: impl Into<String>, value: bool) -> Self {
        Fluent::BooleanFluent { name: name.into(), value }
    }

    /// Create a numeric fluent.
    pub fn numeric(name: impl Into<String>, value: f64) -> Self {
        Fluent::NumericFluent { name: name.into(), value }
    }

    /// Create a spatial fluent.
    pub fn spatial(predicate: SpatialPredicateId, value: bool) -> Self {
        Fluent::SpatialFluent { predicate, value }
    }

    /// Create a timer fluent.
    pub fn timer(name: impl Into<String>, remaining: Duration) -> Self {
        Fluent::TimerFluent { name: name.into(), remaining }
    }

    /// Create a state fluent.
    pub fn state(automaton_id: impl Into<String>, state: StateId) -> Self {
        Fluent::StateFluent { automaton_id: automaton_id.into(), state }
    }

    /// Create a custom fluent.
    pub fn custom(name: impl Into<String>, value: Value) -> Self {
        Fluent::CustomFluent { name: name.into(), value }
    }

    /// Negate a boolean or spatial fluent (returns the flipped version).
    pub fn negated(&self) -> Option<Fluent> {
        match self {
            Fluent::BooleanFluent { name, value } => {
                Some(Fluent::BooleanFluent { name: name.clone(), value: !value })
            }
            Fluent::SpatialFluent { predicate, value } => {
                Some(Fluent::SpatialFluent { predicate: *predicate, value: !value })
            }
            _ => None,
        }
    }

    /// Return the set of entity IDs that this fluent references.
    pub fn referenced_entities(&self) -> Vec<EntityId> {
        match self {
            Fluent::CustomFluent { value: Value::Entity(e), .. } => vec![*e],
            _ => vec![],
        }
    }
}

impl fmt::Display for Fluent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fluent::BooleanFluent { name, value } => write!(f, "{}={}", name, value),
            Fluent::NumericFluent { name, value } => write!(f, "{}={:.4}", name, value),
            Fluent::SpatialFluent { predicate, value } => {
                write!(f, "spatial_{}={}", predicate, value)
            }
            Fluent::TimerFluent { name, remaining } => {
                write!(f, "timer_{}={}", name, remaining)
            }
            Fluent::StateFluent { automaton_id, state } => {
                write!(f, "state_{}={}", automaton_id, state)
            }
            Fluent::CustomFluent { name, value } => write!(f, "{}={}", name, value),
        }
    }
}

// ─── InitiatedBy / TerminatedBy ──────────────────────────────────────────────

/// Records that a fluent was initiated by an event at a time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InitiatedBy {
    pub fluent_id: FluentId,
    pub event_id: EventId,
    pub time: TimePoint,
    pub new_value: Fluent,
}

/// Records that a fluent was terminated by an event at a time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminatedBy {
    pub fluent_id: FluentId,
    pub event_id: EventId,
    pub time: TimePoint,
}

// ─── FluentStore ─────────────────────────────────────────────────────────────

/// Indexed collection of fluents with fast lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluentStore {
    /// Primary storage: id → fluent.
    fluents: IndexMap<FluentId, Fluent>,
    /// Name-based index for fast lookup.
    name_index: HashMap<String, FluentId>,
    /// Spatial predicate index.
    spatial_index: HashMap<SpatialPredicateId, FluentId>,
    /// Counter for generating IDs.
    next_id: u64,
}

impl FluentStore {
    pub fn new() -> Self {
        Self {
            fluents: IndexMap::new(),
            name_index: HashMap::new(),
            spatial_index: HashMap::new(),
            next_id: 1,
        }
    }

    /// Insert a fluent, returning its ID.
    pub fn insert(&mut self, fluent: Fluent) -> FluentId {
        let id = FluentId(self.next_id);
        self.next_id += 1;
        let name = fluent.name();
        self.name_index.insert(name, id);
        if let Fluent::SpatialFluent { predicate, .. } = &fluent {
            self.spatial_index.insert(*predicate, id);
        }
        self.fluents.insert(id, fluent);
        id
    }

    /// Insert a fluent with a specific ID.
    pub fn insert_with_id(&mut self, id: FluentId, fluent: Fluent) {
        let name = fluent.name();
        self.name_index.insert(name, id);
        if let Fluent::SpatialFluent { predicate, .. } = &fluent {
            self.spatial_index.insert(*predicate, id);
        }
        self.fluents.insert(id, fluent);
        if id.0 >= self.next_id {
            self.next_id = id.0 + 1;
        }
    }

    /// Get a fluent by ID.
    pub fn get(&self, id: FluentId) -> Option<&Fluent> {
        self.fluents.get(&id)
    }

    /// Get a mutable reference to a fluent by ID.
    pub fn get_mut(&mut self, id: FluentId) -> Option<&mut Fluent> {
        self.fluents.get_mut(&id)
    }

    /// Look up a fluent by name.
    pub fn get_by_name(&self, name: &str) -> Option<(FluentId, &Fluent)> {
        self.name_index.get(name).and_then(|id| {
            self.fluents.get(id).map(|f| (*id, f))
        })
    }

    /// Look up a fluent by spatial predicate.
    pub fn get_by_predicate(&self, pred: SpatialPredicateId) -> Option<(FluentId, &Fluent)> {
        self.spatial_index.get(&pred).and_then(|id| {
            self.fluents.get(id).map(|f| (*id, f))
        })
    }

    /// Update a fluent's value.
    pub fn update(&mut self, id: FluentId, fluent: Fluent) -> ECResult<()> {
        if !self.fluents.contains_key(&id) {
            return Err(ECError::FluentNotFound(format!("{}", id)));
        }
        let old_name = self.fluents[&id].name();
        self.name_index.remove(&old_name);
        let new_name = fluent.name();
        self.name_index.insert(new_name, id);
        if let Fluent::SpatialFluent { predicate, .. } = &fluent {
            self.spatial_index.insert(*predicate, id);
        }
        self.fluents.insert(id, fluent);
        Ok(())
    }

    /// Remove a fluent.
    pub fn remove(&mut self, id: FluentId) -> Option<Fluent> {
        if let Some(fluent) = self.fluents.swap_remove(&id) {
            self.name_index.remove(&fluent.name());
            if let Fluent::SpatialFluent { predicate, .. } = &fluent {
                self.spatial_index.remove(predicate);
            }
            Some(fluent)
        } else {
            None
        }
    }

    /// Iterate over all fluents.
    pub fn iter(&self) -> impl Iterator<Item = (FluentId, &Fluent)> {
        self.fluents.iter().map(|(id, f)| (*id, f))
    }

    /// Number of fluents.
    pub fn len(&self) -> usize {
        self.fluents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fluents.is_empty()
    }

    /// Get all fluent IDs.
    pub fn ids(&self) -> Vec<FluentId> {
        self.fluents.keys().copied().collect()
    }

    /// Check if a fluent ID exists.
    pub fn contains(&self, id: FluentId) -> bool {
        self.fluents.contains_key(&id)
    }

    /// Snapshot the current state.
    pub fn snapshot(&self, time: TimePoint) -> FluentSnapshot {
        FluentSnapshot {
            time,
            values: self.fluents.clone(),
        }
    }

    /// Compute the delta between current state and a previous snapshot.
    pub fn delta_from(&self, previous: &FluentSnapshot) -> FluentDelta {
        let mut initiated = Vec::new();
        let mut terminated = Vec::new();
        let mut changed = Vec::new();

        for (id, fluent) in &self.fluents {
            match previous.values.get(id) {
                None => initiated.push((*id, fluent.clone())),
                Some(prev) => {
                    if prev != fluent {
                        changed.push((*id, prev.clone(), fluent.clone()));
                    }
                }
            }
        }

        for (id, _fluent) in &previous.values {
            if !self.fluents.contains_key(id) {
                terminated.push(*id);
            }
        }

        FluentDelta {
            initiated,
            terminated,
            changed,
        }
    }

    /// Get all boolean fluents that currently hold.
    pub fn holding_fluents(&self) -> Vec<(FluentId, &Fluent)> {
        self.fluents
            .iter()
            .filter(|(_, f)| f.holds())
            .map(|(id, f)| (*id, f))
            .collect()
    }

    /// Get all spatial fluent IDs.
    pub fn spatial_fluent_ids(&self) -> Vec<FluentId> {
        self.fluents
            .iter()
            .filter_map(|(id, f)| {
                if matches!(f, Fluent::SpatialFluent { .. }) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for FluentStore {
    fn default() -> Self {
        Self::new()
    }
}

// ─── FluentSnapshot ──────────────────────────────────────────────────────────

/// Immutable snapshot of all fluent values at a single time point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluentSnapshot {
    pub time: TimePoint,
    pub values: IndexMap<FluentId, Fluent>,
}

impl FluentSnapshot {
    pub fn new(time: TimePoint) -> Self {
        Self {
            time,
            values: IndexMap::new(),
        }
    }

    pub fn with_fluents(time: TimePoint, fluents: Vec<(FluentId, Fluent)>) -> Self {
        let mut values = IndexMap::new();
        for (id, fluent) in fluents {
            values.insert(id, fluent);
        }
        Self { time, values }
    }

    pub fn get(&self, id: FluentId) -> Option<&Fluent> {
        self.values.get(&id)
    }

    pub fn holds(&self, id: FluentId) -> bool {
        self.values.get(&id).map_or(false, |f| f.holds())
    }

    pub fn fluent_ids(&self) -> Vec<FluentId> {
        self.values.keys().copied().collect()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Compute delta from this snapshot to another.
    pub fn diff(&self, other: &FluentSnapshot) -> FluentDelta {
        let mut initiated = Vec::new();
        let mut terminated = Vec::new();
        let mut changed = Vec::new();

        for (id, fluent) in &other.values {
            match self.values.get(id) {
                None => initiated.push((*id, fluent.clone())),
                Some(prev) if prev != fluent => {
                    changed.push((*id, prev.clone(), fluent.clone()));
                }
                _ => {}
            }
        }

        for (id, _) in &self.values {
            if !other.values.contains_key(id) {
                terminated.push(*id);
            }
        }

        FluentDelta { initiated, terminated, changed }
    }

    /// Apply a delta to this snapshot, producing a new snapshot at the given time.
    pub fn apply_delta(&self, delta: &FluentDelta, new_time: TimePoint) -> FluentSnapshot {
        let mut values = self.values.clone();
        for id in &delta.terminated {
            values.swap_remove(id);
        }
        for (id, fluent) in &delta.initiated {
            values.insert(*id, fluent.clone());
        }
        for (id, _old, new_val) in &delta.changed {
            values.insert(*id, new_val.clone());
        }
        FluentSnapshot { time: new_time, values }
    }

    /// Merge another snapshot into this one (other takes precedence on conflict).
    pub fn merge(&self, other: &FluentSnapshot) -> FluentSnapshot {
        let mut values = self.values.clone();
        for (id, fluent) in &other.values {
            values.insert(*id, fluent.clone());
        }
        FluentSnapshot {
            time: if other.time >= self.time { other.time } else { self.time },
            values,
        }
    }
}

impl PartialEq for FluentSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

// ─── FluentDelta ─────────────────────────────────────────────────────────────

/// Changes between two fluent snapshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluentDelta {
    /// Fluents that were created (newly initiated).
    pub initiated: Vec<(FluentId, Fluent)>,
    /// Fluents that were terminated (removed).
    pub terminated: Vec<FluentId>,
    /// Fluents that changed value: (id, old_value, new_value).
    pub changed: Vec<(FluentId, Fluent, Fluent)>,
}

impl FluentDelta {
    pub fn empty() -> Self {
        Self {
            initiated: Vec::new(),
            terminated: Vec::new(),
            changed: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.initiated.is_empty() && self.terminated.is_empty() && self.changed.is_empty()
    }

    /// Number of total modifications.
    pub fn modification_count(&self) -> usize {
        self.initiated.len() + self.terminated.len() + self.changed.len()
    }

    /// All fluent IDs affected by this delta.
    pub fn affected_ids(&self) -> HashSet<FluentId> {
        let mut ids = HashSet::new();
        for (id, _) in &self.initiated {
            ids.insert(*id);
        }
        for id in &self.terminated {
            ids.insert(*id);
        }
        for (id, _, _) in &self.changed {
            ids.insert(*id);
        }
        ids
    }

    /// Compose two deltas: apply `other` after `self`.
    pub fn compose(&self, other: &FluentDelta) -> FluentDelta {
        let mut initiated = self.initiated.clone();
        let mut terminated = self.terminated.clone();
        let mut changed = self.changed.clone();

        // Remove from initiated anything that other terminates
        let other_terminated: HashSet<FluentId> = other.terminated.iter().copied().collect();
        initiated.retain(|(id, _)| !other_terminated.contains(id));

        // Add other's initiations
        for (id, f) in &other.initiated {
            if !initiated.iter().any(|(eid, _)| eid == id) {
                initiated.push((*id, f.clone()));
            }
        }

        // Add other's terminations (if not already in self's initiated, which we removed)
        for id in &other.terminated {
            if !terminated.contains(id) {
                terminated.push(*id);
            }
        }

        // Update changed from other
        for (id, old, new_val) in &other.changed {
            if let Some(pos) = changed.iter().position(|(cid, _, _)| cid == id) {
                changed[pos].2 = new_val.clone();
            } else {
                changed.push((*id, old.clone(), new_val.clone()));
            }
        }

        FluentDelta { initiated, terminated, changed }
    }
}

// ─── Core EC predicates ──────────────────────────────────────────────────────

/// Determine whether a fluent holds at a given time, applying the law of inertia.
///
/// The commonsense law of inertia states that a fluent persists in its current
/// state unless explicitly initiated or terminated by an event.
pub fn holds_at(history: &FluentHistory, fluent_id: FluentId, time: TimePoint) -> bool {
    // Find the most recent snapshot at or before `time`.
    let entry = history.value_at(fluent_id, time);
    match entry {
        Some(fluent) => fluent.holds(),
        None => false,
    }
}

/// Determine whether a fluent was initiated at a given time.
pub fn initiated_at(history: &FluentHistory, fluent_id: FluentId, time: TimePoint) -> bool {
    history.initiations.iter().any(|init| {
        init.fluent_id == fluent_id && (init.time.0 - time.0).abs() < 1e-9
    })
}

/// Determine whether a fluent was terminated at a given time.
pub fn terminated_at(history: &FluentHistory, fluent_id: FluentId, time: TimePoint) -> bool {
    history.terminations.iter().any(|term| {
        term.fluent_id == fluent_id && (term.time.0 - time.0).abs() < 1e-9
    })
}

/// Checks if a fluent holds throughout an entire interval (using inertia).
pub fn holds_during(
    history: &FluentHistory,
    fluent_id: FluentId,
    interval: &TimeInterval,
) -> bool {
    // The fluent must hold at the start and not be terminated during the interval.
    if !holds_at(history, fluent_id, interval.start) {
        return false;
    }
    // Check no termination in the interval
    !history.terminations.iter().any(|term| {
        term.fluent_id == fluent_id
            && term.time.0 >= interval.start.0
            && term.time.0 <= interval.end.0
    })
}

/// Find the first time a fluent becomes true after `from`.
pub fn first_holds_after(
    history: &FluentHistory,
    fluent_id: FluentId,
    from: TimePoint,
) -> Option<TimePoint> {
    for init in &history.initiations {
        if init.fluent_id == fluent_id && init.time >= from && init.new_value.holds() {
            return Some(init.time);
        }
    }
    None
}

/// Find the last time a fluent was terminated before `before`.
pub fn last_terminated_before(
    history: &FluentHistory,
    fluent_id: FluentId,
    before: TimePoint,
) -> Option<TimePoint> {
    history
        .terminations
        .iter()
        .rev()
        .find(|term| term.fluent_id == fluent_id && term.time < before)
        .map(|term| term.time)
}

// ─── FluentHistory ───────────────────────────────────────────────────────────

/// Tracks fluent values over time, implementing the commonsense law of inertia.
///
/// The history stores a sequence of snapshots, initiation records, and
/// termination records. Between explicit changes, fluent values are propagated
/// by inertia — they keep their last known value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluentHistory {
    /// Ordered sequence of snapshots.
    snapshots: BTreeMap<OrderedFloat<f64>, FluentSnapshot>,
    /// All initiation records, sorted by time.
    pub initiations: Vec<InitiatedBy>,
    /// All termination records, sorted by time.
    pub terminations: Vec<TerminatedBy>,
    /// The initial fluent values at time zero.
    initial_state: FluentSnapshot,
}

impl FluentHistory {
    pub fn new(initial_state: FluentSnapshot) -> Self {
        let mut snapshots = BTreeMap::new();
        snapshots.insert(
            OrderedFloat(initial_state.time.0),
            initial_state.clone(),
        );
        Self {
            snapshots,
            initiations: Vec::new(),
            terminations: Vec::new(),
            initial_state,
        }
    }

    /// Record a snapshot at a given time.
    pub fn record_snapshot(&mut self, snapshot: FluentSnapshot) {
        self.snapshots.insert(OrderedFloat(snapshot.time.0), snapshot);
    }

    /// Record an initiation event.
    pub fn record_initiation(&mut self, init: InitiatedBy) {
        let pos = self
            .initiations
            .binary_search_by(|probe| probe.time.cmp(&init.time))
            .unwrap_or_else(|x| x);
        self.initiations.insert(pos, init);
    }

    /// Record a termination event.
    pub fn record_termination(&mut self, term: TerminatedBy) {
        let pos = self
            .terminations
            .binary_search_by(|probe| probe.time.cmp(&term.time))
            .unwrap_or_else(|x| x);
        self.terminations.insert(pos, term);
    }

    /// Get the value of a fluent at a given time via inertia.
    pub fn value_at(&self, fluent_id: FluentId, time: TimePoint) -> Option<Fluent> {
        let key = OrderedFloat(time.0);

        // Walk backwards through snapshots to find the most recent value
        for (_t, snapshot) in self.snapshots.range(..=key).rev() {
            if let Some(fluent) = snapshot.get(fluent_id) {
                // Check if terminated between snapshot time and query time
                let snap_time = snapshot.time.0;
                let was_terminated = self.terminations.iter().any(|term| {
                    term.fluent_id == fluent_id
                        && term.time.0 > snap_time
                        && term.time.0 <= time.0
                });
                if was_terminated {
                    // Check if re-initiated after termination
                    let last_term_time = self
                        .terminations
                        .iter()
                        .filter(|term| {
                            term.fluent_id == fluent_id
                                && term.time.0 > snap_time
                                && term.time.0 <= time.0
                        })
                        .map(|term| term.time)
                        .max()
                        .unwrap();

                    let reinit = self.initiations.iter().find(|init| {
                        init.fluent_id == fluent_id
                            && init.time.0 > last_term_time.0
                            && init.time.0 <= time.0
                    });

                    return reinit.map(|init| init.new_value.clone());
                }

                // Check if a later initiation changes the value
                let last_init = self
                    .initiations
                    .iter()
                    .filter(|init| {
                        init.fluent_id == fluent_id
                            && init.time.0 > snap_time
                            && init.time.0 <= time.0
                    })
                    .last();

                if let Some(init) = last_init {
                    return Some(init.new_value.clone());
                }

                return Some(fluent.clone());
            }
        }
        None
    }

    /// Get the most recent snapshot at or before a time.
    pub fn snapshot_at(&self, time: TimePoint) -> Option<&FluentSnapshot> {
        let key = OrderedFloat(time.0);
        self.snapshots.range(..=key).next_back().map(|(_, s)| s)
    }

    /// Get all snapshots in a time range.
    pub fn snapshots_in(&self, interval: &TimeInterval) -> Vec<&FluentSnapshot> {
        let start = OrderedFloat(interval.start.0);
        let end = OrderedFloat(interval.end.0);
        self.snapshots
            .range(start..=end)
            .map(|(_, s)| s)
            .collect()
    }

    /// Get the time span covered by this history.
    pub fn time_span(&self) -> Option<TimeInterval> {
        let first = self.snapshots.keys().next()?;
        let last = self.snapshots.keys().next_back()?;
        Some(TimeInterval::new(
            TimePoint(first.0),
            TimePoint(last.0),
        ))
    }

    /// Number of snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get the initial state.
    pub fn initial_state(&self) -> &FluentSnapshot {
        &self.initial_state
    }

    /// Get all initiation times for a specific fluent.
    pub fn initiation_times(&self, fluent_id: FluentId) -> Vec<TimePoint> {
        self.initiations
            .iter()
            .filter(|init| init.fluent_id == fluent_id)
            .map(|init| init.time)
            .collect()
    }

    /// Get all termination times for a specific fluent.
    pub fn termination_times(&self, fluent_id: FluentId) -> Vec<TimePoint> {
        self.terminations
            .iter()
            .filter(|term| term.fluent_id == fluent_id)
            .map(|term| term.time)
            .collect()
    }

    /// Compute intervals during which a fluent holds (using inertia).
    pub fn holding_intervals(&self, fluent_id: FluentId) -> Vec<TimeInterval> {
        let mut intervals = Vec::new();
        let mut current_start: Option<TimePoint> = None;

        // Collect all relevant time points
        let mut times: Vec<TimePoint> = Vec::new();
        for (t, _) in &self.snapshots {
            times.push(TimePoint(t.0));
        }
        for init in &self.initiations {
            if init.fluent_id == fluent_id {
                times.push(init.time);
            }
        }
        for term in &self.terminations {
            if term.fluent_id == fluent_id {
                times.push(term.time);
            }
        }
        times.sort();
        times.dedup();

        for t in &times {
            let h = holds_at(self, fluent_id, *t);
            match (current_start, h) {
                (None, true) => current_start = Some(*t),
                (Some(start), false) => {
                    intervals.push(TimeInterval::new(start, *t));
                    current_start = None;
                }
                _ => {}
            }
        }

        if let Some(start) = current_start {
            if let Some(last_t) = times.last() {
                intervals.push(TimeInterval::new(start, *last_t));
            }
        }

        intervals
    }

    /// Clear all history and reset to initial state.
    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.initiations.clear();
        self.terminations.clear();
        self.snapshots.insert(
            OrderedFloat(self.initial_state.time.0),
            self.initial_state.clone(),
        );
    }

    /// Get the latest snapshot.
    pub fn latest_snapshot(&self) -> Option<&FluentSnapshot> {
        self.snapshots.values().next_back()
    }
}

// ─── Inertia implementation ──────────────────────────────────────────────────

/// Apply the commonsense law of inertia: propagate fluent values forward
/// from `previous_snapshot` to `new_time`, applying the given delta.
pub fn apply_inertia(
    previous: &FluentSnapshot,
    delta: &FluentDelta,
    new_time: TimePoint,
) -> FluentSnapshot {
    let mut values = previous.values.clone();

    // Remove terminated fluents
    for id in &delta.terminated {
        values.swap_remove(id);
    }

    // Apply changes
    for (id, _old, new_val) in &delta.changed {
        values.insert(*id, new_val.clone());
    }

    // Add initiated fluents
    for (id, fluent) in &delta.initiated {
        values.insert(*id, fluent.clone());
    }

    // Everything else is inherited by inertia (already in `values`)
    FluentSnapshot { time: new_time, values }
}

/// Compute the minimal delta needed to transition between two states,
/// preserving inertia for unchanged fluents.
pub fn minimal_delta(from: &FluentSnapshot, to: &FluentSnapshot) -> FluentDelta {
    from.diff(to)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> FluentStore {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("grabbed", false));
        store.insert(Fluent::numeric("distance", 5.0));
        store.insert(Fluent::spatial(SpatialPredicateId(1), true));
        store
    }

    #[test]
    fn test_fluent_store_insert_and_lookup() {
        let store = make_store();
        assert_eq!(store.len(), 3);

        let (id, fluent) = store.get_by_name("grabbed").unwrap();
        assert_eq!(id, FluentId(1));
        assert!(!fluent.holds());
    }

    #[test]
    fn test_fluent_store_spatial_index() {
        let store = make_store();
        let (id, f) = store.get_by_predicate(SpatialPredicateId(1)).unwrap();
        assert_eq!(id, FluentId(3));
        assert!(f.holds());
    }

    #[test]
    fn test_fluent_store_update() {
        let mut store = make_store();
        store.update(FluentId(1), Fluent::boolean("grabbed", true)).unwrap();
        assert!(store.get(FluentId(1)).unwrap().holds());
    }

    #[test]
    fn test_fluent_store_remove() {
        let mut store = make_store();
        let removed = store.remove(FluentId(2));
        assert!(removed.is_some());
        assert_eq!(store.len(), 2);
        assert!(store.get(FluentId(2)).is_none());
    }

    #[test]
    fn test_fluent_snapshot_diff() {
        let mut store = FluentStore::new();
        let id1 = store.insert(Fluent::boolean("a", true));
        let _id2 = store.insert(Fluent::boolean("b", false));
        let snap1 = store.snapshot(TimePoint::from_secs(0.0));

        store.update(id1, Fluent::boolean("a", false)).unwrap();
        store.insert(Fluent::boolean("c", true));
        let snap2 = store.snapshot(TimePoint::from_secs(1.0));

        let delta = snap1.diff(&snap2);
        assert_eq!(delta.changed.len(), 1);
        assert_eq!(delta.changed[0].0, id1);
        assert_eq!(delta.initiated.len(), 1);
        assert!(delta.terminated.is_empty());
    }

    #[test]
    fn test_fluent_delta_compose() {
        let d1 = FluentDelta {
            initiated: vec![(FluentId(10), Fluent::boolean("x", true))],
            terminated: vec![],
            changed: vec![],
        };
        let d2 = FluentDelta {
            initiated: vec![],
            terminated: vec![FluentId(10)],
            changed: vec![],
        };
        let composed = d1.compose(&d2);
        assert!(composed.initiated.is_empty());
        assert_eq!(composed.terminated.len(), 1);
    }

    #[test]
    fn test_holds_at_with_inertia() {
        let initial = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![(FluentId(1), Fluent::boolean("on", true))],
        );
        let history = FluentHistory::new(initial);

        // Fluent holds at time 0
        assert!(holds_at(&history, FluentId(1), TimePoint::from_secs(0.0)));
        // By inertia, it holds at time 5 as well
        assert!(holds_at(&history, FluentId(1), TimePoint::from_secs(5.0)));
    }

    #[test]
    fn test_holds_at_after_termination() {
        let initial = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![(FluentId(1), Fluent::boolean("on", true))],
        );
        let mut history = FluentHistory::new(initial);

        history.record_termination(TerminatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(100),
            time: TimePoint::from_secs(3.0),
        });

        assert!(holds_at(&history, FluentId(1), TimePoint::from_secs(2.0)));
        assert!(!holds_at(&history, FluentId(1), TimePoint::from_secs(4.0)));
    }

    #[test]
    fn test_holds_at_reinitiation() {
        let initial = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![(FluentId(1), Fluent::boolean("on", true))],
        );
        let mut history = FluentHistory::new(initial);

        history.record_termination(TerminatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(100),
            time: TimePoint::from_secs(2.0),
        });
        history.record_initiation(InitiatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(101),
            time: TimePoint::from_secs(5.0),
            new_value: Fluent::boolean("on", true),
        });

        assert!(holds_at(&history, FluentId(1), TimePoint::from_secs(1.0)));
        assert!(!holds_at(&history, FluentId(1), TimePoint::from_secs(3.0)));
        assert!(holds_at(&history, FluentId(1), TimePoint::from_secs(6.0)));
    }

    #[test]
    fn test_initiated_at() {
        let initial = FluentSnapshot::new(TimePoint::from_secs(0.0));
        let mut history = FluentHistory::new(initial);
        history.record_initiation(InitiatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(1),
            time: TimePoint::from_secs(3.0),
            new_value: Fluent::boolean("x", true),
        });

        assert!(initiated_at(&history, FluentId(1), TimePoint::from_secs(3.0)));
        assert!(!initiated_at(&history, FluentId(1), TimePoint::from_secs(4.0)));
    }

    #[test]
    fn test_terminated_at() {
        let initial = FluentSnapshot::new(TimePoint::from_secs(0.0));
        let mut history = FluentHistory::new(initial);
        history.record_termination(TerminatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(1),
            time: TimePoint::from_secs(7.0),
        });

        assert!(terminated_at(&history, FluentId(1), TimePoint::from_secs(7.0)));
        assert!(!terminated_at(&history, FluentId(1), TimePoint::from_secs(8.0)));
    }

    #[test]
    fn test_holds_during() {
        let initial = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![(FluentId(1), Fluent::boolean("on", true))],
        );
        let mut history = FluentHistory::new(initial);

        // No termination: holds for the whole interval
        assert!(holds_during(
            &history,
            FluentId(1),
            &TimeInterval::new(TimePoint::from_secs(0.0), TimePoint::from_secs(10.0)),
        ));

        // Termination at t=5: does not hold during [0, 10]
        history.record_termination(TerminatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(1),
            time: TimePoint::from_secs(5.0),
        });
        assert!(!holds_during(
            &history,
            FluentId(1),
            &TimeInterval::new(TimePoint::from_secs(0.0), TimePoint::from_secs(10.0)),
        ));
    }

    #[test]
    fn test_apply_inertia() {
        let snap = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![
                (FluentId(1), Fluent::boolean("a", true)),
                (FluentId(2), Fluent::boolean("b", false)),
            ],
        );
        let delta = FluentDelta {
            initiated: vec![(FluentId(3), Fluent::boolean("c", true))],
            terminated: vec![FluentId(2)],
            changed: vec![(FluentId(1), Fluent::boolean("a", true), Fluent::boolean("a", false))],
        };

        let result = apply_inertia(&snap, &delta, TimePoint::from_secs(1.0));
        assert_eq!(result.values.len(), 2);
        assert!(!result.holds(FluentId(1))); // changed to false
        assert!(result.get(FluentId(2)).is_none()); // terminated
        assert!(result.holds(FluentId(3))); // initiated
    }

    #[test]
    fn test_fluent_history_holding_intervals() {
        let initial = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![(FluentId(1), Fluent::boolean("lamp", true))],
        );
        let mut history = FluentHistory::new(initial);

        history.record_termination(TerminatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(1),
            time: TimePoint::from_secs(3.0),
        });
        history.record_initiation(InitiatedBy {
            fluent_id: FluentId(1),
            event_id: EventId(2),
            time: TimePoint::from_secs(5.0),
            new_value: Fluent::boolean("lamp", true),
        });
        history.record_snapshot(FluentSnapshot::with_fluents(
            TimePoint::from_secs(10.0),
            vec![(FluentId(1), Fluent::boolean("lamp", true))],
        ));

        let intervals = history.holding_intervals(FluentId(1));
        assert!(intervals.len() >= 2);
    }

    #[test]
    fn test_fluent_snapshot_apply_delta() {
        let snap = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![
                (FluentId(1), Fluent::boolean("x", true)),
                (FluentId(2), Fluent::numeric("y", 10.0)),
            ],
        );
        let delta = FluentDelta {
            initiated: vec![],
            terminated: vec![],
            changed: vec![(
                FluentId(2),
                Fluent::numeric("y", 10.0),
                Fluent::numeric("y", 20.0),
            )],
        };
        let new_snap = snap.apply_delta(&delta, TimePoint::from_secs(1.0));
        assert_eq!(new_snap.time, TimePoint::from_secs(1.0));
        match new_snap.get(FluentId(2)).unwrap() {
            Fluent::NumericFluent { value, .. } => assert!((*value - 20.0).abs() < 1e-9),
            _ => panic!("wrong fluent type"),
        }
    }

    #[test]
    fn test_minimal_delta() {
        let from = FluentSnapshot::with_fluents(
            TimePoint::from_secs(0.0),
            vec![
                (FluentId(1), Fluent::boolean("a", true)),
                (FluentId(2), Fluent::boolean("b", false)),
            ],
        );
        let to = FluentSnapshot::with_fluents(
            TimePoint::from_secs(1.0),
            vec![
                (FluentId(1), Fluent::boolean("a", true)),
                (FluentId(3), Fluent::boolean("c", true)),
            ],
        );
        let delta = minimal_delta(&from, &to);
        assert!(delta.initiated.iter().any(|(id, _)| *id == FluentId(3)));
        assert!(delta.terminated.contains(&FluentId(2)));
        assert!(delta.changed.is_empty()); // FluentId(1) unchanged
    }

    #[test]
    fn test_fluent_holds_methods() {
        assert!(Fluent::boolean("x", true).holds());
        assert!(!Fluent::boolean("x", false).holds());
        assert!(Fluent::numeric("x", 1.0).holds());
        assert!(!Fluent::numeric("x", 0.0).holds());
        assert!(Fluent::timer("x", Duration::from_secs(1.0)).holds());
        assert!(!Fluent::timer("x", Duration::from_secs(0.0)).holds());
    }

    #[test]
    fn test_fluent_negation() {
        let f = Fluent::boolean("x", true);
        let neg = f.negated().unwrap();
        assert!(!neg.holds());

        let n = Fluent::numeric("y", 5.0);
        assert!(n.negated().is_none());
    }
}
