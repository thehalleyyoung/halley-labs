//! Fluent builder API for constructing `SpatialEventAutomaton` instances.
//!
//! Provides `AutomatonBuilder` (fluent, validated construction),
//! `AutomatonTemplate` (pre-built interaction patterns), and
//! `from_ec_blueprint` (conversion from Event Calculus IR).

use crate::automaton::{
    SpatialEventAutomaton, State, Transition,
};
use crate::{
    Action, Duration as ChorDuration, ECBlueprint, EntityId, EventKind, Guard,
    RegionId, Result, SpatialPredicate, Span, StateId, TemporalGuardExpr,
    TimerId, TransitionId, VarId, Value, AutomataError,
};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// StateConfig
// ---------------------------------------------------------------------------

/// Configuration for a state to be added via the builder.
#[derive(Debug, Clone)]
pub struct StateConfig {
    pub name: String,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub is_error: bool,
    pub invariant: Option<Guard>,
    pub on_entry: Vec<Action>,
    pub on_exit: Vec<Action>,
    pub metadata: HashMap<String, String>,
}

impl StateConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            is_initial: false,
            is_accepting: false,
            is_error: false,
            invariant: None,
            on_entry: Vec::new(),
            on_exit: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn initial(mut self) -> Self {
        self.is_initial = true;
        self
    }

    pub fn accepting(mut self) -> Self {
        self.is_accepting = true;
        self
    }

    pub fn error(mut self) -> Self {
        self.is_error = true;
        self
    }

    pub fn invariant(mut self, guard: Guard) -> Self {
        self.invariant = Some(guard);
        self
    }

    pub fn on_entry(mut self, actions: Vec<Action>) -> Self {
        self.on_entry = actions;
        self
    }

    pub fn on_exit(mut self, actions: Vec<Action>) -> Self {
        self.on_exit = actions;
        self
    }

    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// AutomatonBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing and validating `SpatialEventAutomaton`.
#[derive(Debug)]
pub struct AutomatonBuilder {
    name: String,
    states: Vec<(StateId, StateConfig)>,
    transitions: Vec<(TransitionId, StateId, StateId, Guard, Vec<Action>, i32)>,
    initial: Option<StateId>,
    accepting: HashSet<StateId>,
    next_state_id: u32,
    next_transition_id: u32,
    source_span: Option<Span>,
    description: String,
    tags: Vec<String>,
    strict_validation: bool,
}

impl AutomatonBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            states: Vec::new(),
            transitions: Vec::new(),
            initial: None,
            accepting: HashSet::new(),
            next_state_id: 0,
            next_transition_id: 0,
            source_span: None,
            description: String::new(),
            tags: Vec::new(),
            strict_validation: true,
        }
    }

    /// Add a state with the given configuration; returns the allocated `StateId`.
    pub fn add_state(&mut self, config: StateConfig) -> StateId {
        let id = StateId(self.next_state_id);
        self.next_state_id += 1;
        if config.is_initial {
            self.initial = Some(id);
        }
        if config.is_accepting {
            self.accepting.insert(id);
        }
        self.states.push((id, config));
        id
    }

    /// Add a transition; returns the allocated `TransitionId`.
    pub fn add_transition(
        &mut self,
        source: StateId,
        target: StateId,
        guard: Guard,
        actions: Vec<Action>,
    ) -> TransitionId {
        let id = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        self.transitions.push((id, source, target, guard, actions, 0));
        id
    }

    /// Add a transition with an explicit priority.
    pub fn add_transition_with_priority(
        &mut self,
        source: StateId,
        target: StateId,
        guard: Guard,
        actions: Vec<Action>,
        priority: i32,
    ) -> TransitionId {
        let id = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        self.transitions
            .push((id, source, target, guard, actions, priority));
        id
    }

    /// Set the initial state (overrides any previous setting).
    pub fn set_initial(&mut self, state_id: StateId) -> &mut Self {
        self.initial = Some(state_id);
        self
    }

    /// Mark a state as accepting.
    pub fn add_accepting(&mut self, state_id: StateId) -> &mut Self {
        self.accepting.insert(state_id);
        self
    }

    /// Attach a source span for diagnostics.
    pub fn source_span(mut self, span: Span) -> Self {
        self.source_span = Some(span);
        self
    }

    /// Set description text.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Disable strict validation (allow unreachable states, etc.).
    pub fn relaxed(mut self) -> Self {
        self.strict_validation = false;
        self
    }

    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------

    /// Validate and build the `SpatialEventAutomaton`.
    pub fn build(self) -> Result<SpatialEventAutomaton> {
        // Must have an initial state
        let initial = self
            .initial
            .ok_or(AutomataError::NoInitialState)?;

        // Collect declared state ids
        let declared: HashSet<StateId> = self.states.iter().map(|(id, _)| *id).collect();

        // Initial state must be declared
        if !declared.contains(&initial) {
            return Err(AutomataError::StateNotFound(initial));
        }

        // All accepting states must be declared
        for &acc in &self.accepting {
            if !declared.contains(&acc) {
                return Err(AutomataError::StateNotFound(acc));
            }
        }

        // All transition endpoints must reference declared states
        for (_tid, src, tgt, _, _, _) in &self.transitions {
            if !declared.contains(src) {
                return Err(AutomataError::InvalidTransition {
                    from: *src,
                    to: *tgt,
                    reason: format!("source state {} not declared", src),
                });
            }
            if !declared.contains(tgt) {
                return Err(AutomataError::InvalidTransition {
                    from: *src,
                    to: *tgt,
                    reason: format!("target state {} not declared", tgt),
                });
            }
        }

        // Build adjacency for reachability
        let mut adj: HashMap<StateId, Vec<StateId>> = HashMap::new();
        for &(_, src, tgt, _, _, _) in &self.transitions {
            adj.entry(src).or_default().push(tgt);
        }

        if self.strict_validation {
            // Reachability from initial
            let reachable = bfs_reachable(initial, &adj);
            let unreachable: Vec<StateId> = declared
                .iter()
                .filter(|s| !reachable.contains(s))
                .copied()
                .collect();
            if !unreachable.is_empty() {
                return Err(AutomataError::UnreachableStates(unreachable));
            }

            // Guard consistency: check that no guard is trivially False
            // on all outgoing transitions of a non-accepting state
            for (sid, cfg) in &self.states {
                if cfg.is_accepting || cfg.is_error {
                    continue;
                }
                let outgoing: Vec<&Guard> = self
                    .transitions
                    .iter()
                    .filter(|(_, src, _, _, _, _)| src == sid)
                    .map(|(_, _, _, g, _, _)| g)
                    .collect();
                if !outgoing.is_empty() && outgoing.iter().all(|g| g.is_trivially_false()) {
                    return Err(AutomataError::GuardInconsistency(format!(
                        "All outgoing guards of state {} are trivially false",
                        sid
                    )));
                }
            }
        }

        // Assemble automaton
        let mut auto = SpatialEventAutomaton::new(&self.name);
        auto.metadata.source_span = self.source_span;
        auto.metadata.description = self.description;
        auto.metadata.tags = self.tags;

        for (id, cfg) in &self.states {
            let mut s = State::new(*id, &cfg.name);
            s.is_initial = Some(*id) == self.initial;
            s.is_accepting = self.accepting.contains(id);
            s.is_error = cfg.is_error;
            s.invariant = cfg.invariant.clone();
            s.on_entry = cfg.on_entry.clone();
            s.on_exit = cfg.on_exit.clone();
            s.metadata = cfg.metadata.clone();
            auto.add_state(s);
        }

        auto.next_state_id = self.next_state_id;
        auto.next_transition_id = self.next_transition_id;

        for (id, src, tgt, guard, actions, priority) in self.transitions {
            let mut t = Transition::new(id, src, tgt, guard, actions);
            t.priority = priority;
            auto.add_transition(t);
        }

        auto.recompute_statistics();
        Ok(auto)
    }
}

/// BFS reachability.
fn bfs_reachable(start: StateId, adj: &HashMap<StateId, Vec<StateId>>) -> HashSet<StateId> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);
    while let Some(s) = queue.pop_front() {
        if let Some(nexts) = adj.get(&s) {
            for &n in nexts {
                if visited.insert(n) {
                    queue.push_back(n);
                }
            }
        }
    }
    visited
}

// ---------------------------------------------------------------------------
// AutomatonTemplate – parameterised templates for common XR patterns
// ---------------------------------------------------------------------------

/// Pre-built templates for common XR interaction patterns.
pub struct AutomatonTemplate;

impl AutomatonTemplate {
    /// Gaze-dwell template: idle → gazing → activated
    ///
    /// Transitions:
    /// - idle --[gaze_enter]--> gazing
    /// - gazing --[dwell_timer_elapsed]--> activated
    /// - gazing --[gaze_exit]--> idle
    /// - activated → (accepting)
    pub fn gaze_dwell(
        target_region: RegionId,
        dwell_duration: ChorDuration,
    ) -> Result<SpatialEventAutomaton> {
        let mut b = AutomatonBuilder::new("gaze_dwell");

        let idle = b.add_state(StateConfig::new("idle").initial());
        let gazing = b.add_state(StateConfig::new("gazing").on_entry(vec![
            Action::StartTimer(TimerId("dwell_timer".into())),
            Action::Highlight {
                entity: EntityId(target_region.0.clone()),
                style: "gaze_highlight".into(),
            },
        ]).on_exit(vec![
            Action::StopTimer(TimerId("dwell_timer".into())),
            Action::ClearHighlight(EntityId(target_region.0.clone())),
        ]));
        let activated = b.add_state(StateConfig::new("activated").accepting().on_entry(vec![
            Action::PlayFeedback("dwell_complete".into()),
        ]));

        b.add_transition(
            idle,
            gazing,
            Guard::Event(EventKind::GazeEnter),
            vec![],
        );
        b.add_transition(
            gazing,
            idle,
            Guard::Event(EventKind::GazeExit),
            vec![],
        );
        b.add_transition(
            gazing,
            activated,
            Guard::And(vec![
                Guard::Event(EventKind::GazeDwell),
                Guard::Temporal(TemporalGuardExpr::TimerElapsed {
                    timer: TimerId("dwell_timer".into()),
                    threshold: dwell_duration,
                }),
            ]),
            vec![],
        );

        b.build()
    }

    /// Grab-release template: idle → grabbed → released
    ///
    /// Models a pick-up / put-down interaction.
    pub fn grab_release(
        hand: EntityId,
        object: EntityId,
    ) -> Result<SpatialEventAutomaton> {
        let mut b = AutomatonBuilder::new("grab_release");

        let idle = b.add_state(
            StateConfig::new("idle")
                .initial()
                .invariant(Guard::Spatial(SpatialPredicate::Not(Box::new(
                    SpatialPredicate::Grasping {
                        hand: hand.clone(),
                        object: object.clone(),
                    },
                )))),
        );
        let grabbed = b.add_state(
            StateConfig::new("grabbed")
                .invariant(Guard::Spatial(SpatialPredicate::Grasping {
                    hand: hand.clone(),
                    object: object.clone(),
                }))
                .on_entry(vec![Action::PlayFeedback("grab_haptic".into())]),
        );
        let released = b.add_state(StateConfig::new("released").accepting().on_entry(vec![
            Action::PlayFeedback("release_haptic".into()),
        ]));

        b.add_transition(
            idle,
            grabbed,
            Guard::And(vec![
                Guard::Event(EventKind::GrabStart),
                Guard::Spatial(SpatialPredicate::Proximity {
                    entity_a: hand.clone(),
                    entity_b: object.clone(),
                    threshold: 0.1,
                }),
            ]),
            vec![],
        );
        b.add_transition(
            grabbed,
            released,
            Guard::Event(EventKind::GrabEnd),
            vec![],
        );
        b.add_transition(released, idle, Guard::True, vec![]);

        b.build()
    }

    /// Proximity-activation template: far → near → activated
    ///
    /// Entity approaches a region; once close enough and dwells, activation occurs.
    pub fn proximity_activation(
        entity: EntityId,
        target: EntityId,
        activation_distance: f64,
        dwell_time: ChorDuration,
    ) -> Result<SpatialEventAutomaton> {
        let mut b = AutomatonBuilder::new("proximity_activation");

        let far = b.add_state(StateConfig::new("far").initial());
        let near = b.add_state(
            StateConfig::new("near")
                .on_entry(vec![
                    Action::StartTimer(TimerId("prox_timer".into())),
                    Action::Highlight {
                        entity: EntityId(target.0.clone()),
                        style: "proximity_glow".into(),
                    },
                ])
                .on_exit(vec![
                    Action::StopTimer(TimerId("prox_timer".into())),
                    Action::ClearHighlight(EntityId(target.0.clone())),
                ]),
        );
        let activated = b.add_state(StateConfig::new("activated").accepting().on_entry(vec![
            Action::PlayFeedback("activated".into()),
        ]));

        b.add_transition(
            far,
            near,
            Guard::And(vec![
                Guard::Event(EventKind::ProximityEnter),
                Guard::Spatial(SpatialPredicate::Proximity {
                    entity_a: entity.clone(),
                    entity_b: target.clone(),
                    threshold: activation_distance,
                }),
            ]),
            vec![],
        );
        b.add_transition(
            near,
            far,
            Guard::Event(EventKind::ProximityExit),
            vec![],
        );
        b.add_transition(
            near,
            activated,
            Guard::Temporal(TemporalGuardExpr::TimerElapsed {
                timer: TimerId("prox_timer".into()),
                threshold: dwell_time,
            }),
            vec![],
        );

        b.build()
    }

    /// Menu interaction template: closed → open → item_hovered → item_selected
    ///
    /// A spatial menu that opens on gaze, shows item hover feedback, and
    /// activates on grab/select.
    pub fn menu_interaction(
        menu_region: RegionId,
        item_count: usize,
    ) -> Result<SpatialEventAutomaton> {
        let mut b = AutomatonBuilder::new("menu_interaction");

        let closed = b.add_state(StateConfig::new("closed").initial());
        let open = b.add_state(
            StateConfig::new("open").on_entry(vec![Action::Custom("show_menu".into())]),
        );
        let item_hovered = b.add_state(
            StateConfig::new("item_hovered")
                .on_entry(vec![Action::PlayFeedback("hover_haptic".into())]),
        );
        let item_selected = b.add_state(
            StateConfig::new("item_selected")
                .accepting()
                .on_entry(vec![Action::PlayFeedback("select_haptic".into())]),
        );

        // closed → open on gaze enter
        b.add_transition(
            closed,
            open,
            Guard::Event(EventKind::GazeEnter),
            vec![],
        );
        // open → closed on gaze exit
        b.add_transition(
            open,
            closed,
            Guard::Event(EventKind::GazeExit),
            vec![Action::Custom("hide_menu".into())],
        );
        // open → item_hovered when gaze lands on an item
        for i in 0..item_count {
            b.add_transition(
                open,
                item_hovered,
                Guard::And(vec![
                    Guard::Event(EventKind::GazeDwell),
                    Guard::Spatial(SpatialPredicate::GazeAt {
                        entity: EntityId("user_gaze".into()),
                        target: RegionId(format!("{}_item_{}", menu_region.0, i)),
                    }),
                ]),
                vec![Action::SetVar {
                    var: VarId("hovered_item".into()),
                    value: Value::Int(i as i64),
                }],
            );
        }
        // item_hovered → item_selected on grab
        b.add_transition(
            item_hovered,
            item_selected,
            Guard::Event(EventKind::GrabStart),
            vec![],
        );
        // item_hovered → open on different gaze (de-hover)
        b.add_transition(
            item_hovered,
            open,
            Guard::Event(EventKind::GazeExit),
            vec![],
        );
        // item_selected → closed (reset)
        b.add_transition(
            item_selected,
            closed,
            Guard::True,
            vec![Action::Custom("hide_menu".into())],
        );

        b.build()
    }

    /// Two-hand manipulation template: idle → one_hand → two_hands → done
    ///
    /// Models bimanual interaction (e.g., resizing an object with two grabs).
    pub fn two_hand_manipulation(
        left_hand: EntityId,
        right_hand: EntityId,
        object: EntityId,
    ) -> Result<SpatialEventAutomaton> {
        let mut b = AutomatonBuilder::new("two_hand_manipulation");

        let idle = b.add_state(StateConfig::new("idle").initial());
        let one_hand = b.add_state(
            StateConfig::new("one_hand_grabbed")
                .on_entry(vec![Action::PlayFeedback("one_hand_grab".into())]),
        );
        let two_hands = b.add_state(
            StateConfig::new("two_hands_grabbed")
                .on_entry(vec![Action::PlayFeedback("two_hand_grab".into())]),
        );
        let done = b.add_state(StateConfig::new("done").accepting());

        // idle → one_hand (either hand grabs)
        b.add_transition(
            idle,
            one_hand,
            Guard::And(vec![
                Guard::Event(EventKind::GrabStart),
                Guard::Or(vec![
                    Guard::Spatial(SpatialPredicate::Proximity {
                        entity_a: left_hand.clone(),
                        entity_b: object.clone(),
                        threshold: 0.15,
                    }),
                    Guard::Spatial(SpatialPredicate::Proximity {
                        entity_a: right_hand.clone(),
                        entity_b: object.clone(),
                        threshold: 0.15,
                    }),
                ]),
            ]),
            vec![],
        );
        // one_hand → two_hands (second hand grabs)
        b.add_transition(
            one_hand,
            two_hands,
            Guard::Event(EventKind::GrabStart),
            vec![],
        );
        // two_hands → one_hand (one hand releases)
        b.add_transition(
            two_hands,
            one_hand,
            Guard::Event(EventKind::GrabEnd),
            vec![],
        );
        // one_hand → done (release)
        b.add_transition(
            one_hand,
            done,
            Guard::Event(EventKind::GrabEnd),
            vec![],
        );
        // done → idle (reset)
        b.add_transition(done, idle, Guard::True, vec![]);

        b.build()
    }
}

// ---------------------------------------------------------------------------
// EC Blueprint conversion
// ---------------------------------------------------------------------------

/// Convert an `ECBlueprint` (from the Event Calculus compiler) into a
/// `SpatialEventAutomaton`.
///
/// Each unique valuation of the blueprint's fluents becomes a state.
/// Event rules produce transitions between valuations.
pub fn from_ec_blueprint(blueprint: &ECBlueprint) -> Result<SpatialEventAutomaton> {
    let n_fluents = blueprint.fluents.len();
    if n_fluents > 20 {
        return Err(AutomataError::BlueprintConversionError(
            "Too many fluents for explicit state enumeration (>20)".into(),
        ));
    }

    let fluent_names: Vec<&str> = blueprint.fluents.iter().map(|f| f.name.as_str()).collect();

    // Build initial valuation
    let mut initial_val = vec![false; n_fluents];
    for (name, val) in &blueprint.initial_fluents {
        if let Some(idx) = fluent_names.iter().position(|n| n == name) {
            initial_val[idx] = *val;
        }
    }

    // We enumerate valuations lazily from the initial one
    let mut builder = AutomatonBuilder::new(&blueprint.name);
    let mut val_to_state: HashMap<Vec<bool>, StateId> = HashMap::new();
    let mut worklist: VecDeque<Vec<bool>> = VecDeque::new();

    let init_id = builder.add_state(
        StateConfig::new(valuation_name(&fluent_names, &initial_val)).initial(),
    );
    val_to_state.insert(initial_val.clone(), init_id);
    worklist.push_back(initial_val);

    while let Some(val) = worklist.pop_front() {
        let src_id = val_to_state[&val];

        for rule in &blueprint.event_rules {
            // Check if the rule's guard is compatible with the current valuation
            if !guard_compatible_with_valuation(&rule.guard, &fluent_names, &val, &blueprint.fluents) {
                continue;
            }

            // Compute new valuation after applying the rule
            let mut new_val = val.clone();
            for initiated in &rule.initiates {
                if let Some(idx) = fluent_names.iter().position(|n| n == initiated) {
                    new_val[idx] = true;
                }
            }
            for terminated in &rule.terminates {
                if let Some(idx) = fluent_names.iter().position(|n| n == terminated) {
                    new_val[idx] = false;
                }
            }

            // Get or create target state
            let tgt_id = if let Some(&existing) = val_to_state.get(&new_val) {
                existing
            } else {
                let nid = builder.add_state(StateConfig::new(valuation_name(
                    &fluent_names,
                    &new_val,
                )));
                val_to_state.insert(new_val.clone(), nid);
                worklist.push_back(new_val.clone());
                nid
            };

            // Build guard combining event trigger and any spatial predicates
            let guard = build_ec_guard(&rule.trigger_event, &rule.guard);
            builder.add_transition(src_id, tgt_id, guard, vec![]);
        }
    }

    // Mark states where all fluents are true as accepting (heuristic)
    for (val, &sid) in &val_to_state {
        if val.iter().all(|v| *v) {
            builder.add_accepting(sid);
        }
    }

    // Fall back: if no accepting states, mark initial as accepting
    if builder.accepting.is_empty() {
        builder.add_accepting(init_id);
    }

    builder.relaxed().build()
}

/// Build a name string from a fluent valuation.
fn valuation_name(names: &[&str], vals: &[bool]) -> String {
    let parts: Vec<String> = names
        .iter()
        .zip(vals.iter())
        .map(|(n, v)| {
            if *v {
                n.to_string()
            } else {
                format!("¬{}", n)
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Check whether a guard is compatible with a given fluent valuation.
fn guard_compatible_with_valuation(
    guard: &Guard,
    fluent_names: &[&str],
    vals: &[bool],
    fluents: &[crate::ECFluent],
) -> bool {
    match guard {
        Guard::True => true,
        Guard::False => false,
        Guard::Spatial(sp) => {
            // Check if any fluent references this predicate
            for (i, fluent) in fluents.iter().enumerate() {
                if let Some(ref fp) = fluent.predicate {
                    if fp == sp {
                        return vals[i];
                    }
                }
            }
            true // Unknown predicates are assumed satisfiable
        }
        Guard::And(gs) => gs
            .iter()
            .all(|g| guard_compatible_with_valuation(g, fluent_names, vals, fluents)),
        Guard::Or(gs) => gs
            .iter()
            .any(|g| guard_compatible_with_valuation(g, fluent_names, vals, fluents)),
        Guard::Not(g) => !guard_compatible_with_valuation(g, fluent_names, vals, fluents),
        _ => true,
    }
}

/// Build a guard from an EC event rule.
fn build_ec_guard(trigger: &EventKind, extra: &Guard) -> Guard {
    let event_guard = Guard::Event(trigger.clone());
    match extra {
        Guard::True => event_guard,
        other => Guard::And(vec![event_guard, other.clone()]),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_builder_simple() {
        let mut b = AutomatonBuilder::new("simple");
        let s0 = b.add_state(StateConfig::new("idle").initial());
        let s1 = b.add_state(StateConfig::new("active").accepting());
        b.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        let auto = b.build().unwrap();
        assert_eq!(auto.state_count(), 2);
        assert_eq!(auto.transition_count(), 1);
        assert!(auto.initial_state.is_some());
    }

    #[test]
    fn test_builder_no_initial() {
        let mut b = AutomatonBuilder::new("no_init");
        let s0 = b.add_state(StateConfig::new("a"));
        let result = b.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_unreachable_strict() {
        let mut b = AutomatonBuilder::new("unreachable");
        let s0 = b.add_state(StateConfig::new("init").initial().accepting());
        let _s1 = b.add_state(StateConfig::new("orphan"));
        let result = b.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_unreachable_relaxed() {
        let mut b = AutomatonBuilder::new("unreachable_relaxed");
        b.strict_validation = false;
        let s0 = b.add_state(StateConfig::new("init").initial().accepting());
        let _s1 = b.add_state(StateConfig::new("orphan"));
        let result = b.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_invalid_transition() {
        let mut b = AutomatonBuilder::new("invalid");
        let s0 = b.add_state(StateConfig::new("init").initial().accepting());
        b.add_transition(s0, StateId(999), Guard::True, vec![]);
        let result = b.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_guard_consistency() {
        let mut b = AutomatonBuilder::new("inconsistent");
        let s0 = b.add_state(StateConfig::new("init").initial());
        let s1 = b.add_state(StateConfig::new("end").accepting());
        b.add_transition(s0, s1, Guard::False, vec![]);
        let result = b.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_state_config_fluent() {
        let cfg = StateConfig::new("test")
            .initial()
            .accepting()
            .error()
            .invariant(Guard::True)
            .on_entry(vec![Action::Noop])
            .on_exit(vec![Action::Noop])
            .meta("key", "value");
        assert!(cfg.is_initial);
        assert!(cfg.is_accepting);
        assert!(cfg.is_error);
        assert!(cfg.invariant.is_some());
        assert_eq!(cfg.on_entry.len(), 1);
        assert_eq!(cfg.on_exit.len(), 1);
        assert_eq!(cfg.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_template_gaze_dwell() {
        let auto = AutomatonTemplate::gaze_dwell(
            RegionId("button_1".into()),
            ChorDuration(2.0),
        )
        .unwrap();
        assert_eq!(auto.state_count(), 3);
        assert!(auto.accepting_states.len() >= 1);
    }

    #[test]
    fn test_template_grab_release() {
        let auto = AutomatonTemplate::grab_release(
            EntityId("left_hand".into()),
            EntityId("cube".into()),
        )
        .unwrap();
        assert_eq!(auto.state_count(), 3);
        assert!(auto.transition_count() >= 3);
    }

    #[test]
    fn test_template_proximity_activation() {
        let auto = AutomatonTemplate::proximity_activation(
            EntityId("user".into()),
            EntityId("door".into()),
            1.5,
            ChorDuration(1.0),
        )
        .unwrap();
        assert_eq!(auto.state_count(), 3);
    }

    #[test]
    fn test_template_menu_interaction() {
        let auto = AutomatonTemplate::menu_interaction(
            RegionId("main_menu".into()),
            3,
        )
        .unwrap();
        assert!(auto.state_count() >= 4);
        assert!(auto.transition_count() >= 6);
    }

    #[test]
    fn test_template_two_hand_manipulation() {
        let auto = AutomatonTemplate::two_hand_manipulation(
            EntityId("left_hand".into()),
            EntityId("right_hand".into()),
            EntityId("panel".into()),
        )
        .unwrap();
        assert_eq!(auto.state_count(), 4);
        assert!(auto.transition_count() >= 4);
    }

    #[test]
    fn test_ec_blueprint_conversion() {
        let bp = ECBlueprint {
            name: "light_switch".into(),
            fluents: vec![
                ECFluent {
                    name: "light_on".into(),
                    predicate: None,
                },
            ],
            event_rules: vec![
                ECEventRule {
                    trigger_event: EventKind::ButtonPress("switch".into()),
                    guard: Guard::True,
                    initiates: vec!["light_on".into()],
                    terminates: vec![],
                },
                ECEventRule {
                    trigger_event: EventKind::ButtonRelease("switch".into()),
                    guard: Guard::True,
                    initiates: vec![],
                    terminates: vec!["light_on".into()],
                },
            ],
            initial_fluents: vec![("light_on".into(), false)],
        };
        let auto = from_ec_blueprint(&bp).unwrap();
        assert!(auto.state_count() >= 2);
        assert!(auto.transition_count() >= 2);
    }

    #[test]
    fn test_ec_blueprint_too_many_fluents() {
        let fluents: Vec<ECFluent> = (0..21)
            .map(|i| ECFluent {
                name: format!("f{}", i),
                predicate: None,
            })
            .collect();
        let bp = ECBlueprint {
            name: "too_big".into(),
            fluents,
            event_rules: vec![],
            initial_fluents: vec![],
        };
        let result = from_ec_blueprint(&bp);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_priority() {
        let mut b = AutomatonBuilder::new("priority");
        let s0 = b.add_state(StateConfig::new("init").initial());
        let s1 = b.add_state(StateConfig::new("high").accepting());
        let s2 = b.add_state(StateConfig::new("low").accepting());
        b.add_transition_with_priority(
            s0,
            s1,
            Guard::Event(EventKind::GrabStart),
            vec![],
            10,
        );
        b.add_transition_with_priority(
            s0,
            s2,
            Guard::Event(EventKind::GrabStart),
            vec![],
            1,
        );
        let auto = b.build().unwrap();
        assert_eq!(auto.transition_count(), 2);
    }

    #[test]
    fn test_set_initial_override() {
        let mut b = AutomatonBuilder::new("override");
        let s0 = b.add_state(StateConfig::new("a").initial().accepting());
        let s1 = b.add_state(StateConfig::new("b").accepting());
        b.add_transition(s0, s1, Guard::True, vec![]);
        b.add_transition(s1, s0, Guard::True, vec![]);
        b.set_initial(s1);
        let auto = b.build().unwrap();
        assert_eq!(auto.initial_state, Some(s1));
    }
}
