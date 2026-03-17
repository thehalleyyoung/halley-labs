//! Runtime executor for spatial-event automata using NFA token-passing.
//!
//! Tracks a set of active states (tokens), evaluates guards against incoming
//! events and the current scene, fires enabled transitions, and records
//! execution history. Supports deterministic (first-match) and
//! non-deterministic (all-matches) modes, checkpointing, and rollback.

use choreo_automata::automaton::SpatialEventAutomaton;
use choreo_automata::{
    Action, EventKind, SceneConfiguration, StateId, TimePoint, TimerId, TransitionId, Value,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

/// Determines how non-determinism is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Fire only the highest-priority enabled transition from each active state.
    FirstMatch,
    /// Fire *all* enabled transitions (NFA token-passing).
    AllMatches,
}

// ---------------------------------------------------------------------------
// ExecutionResult
// ---------------------------------------------------------------------------

/// The result of a single execution step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// New set of active states after the step.
    pub new_states: HashSet<u32>,
    /// Transitions that actually fired.
    pub fired_transitions: Vec<u32>,
    /// Actions produced by firing transitions and entering/exiting states.
    pub emitted_actions: Vec<Action>,
    /// Timer operations requested (start / stop).
    pub timer_updates: Vec<TimerUpdate>,
    /// Variable assignments.
    pub var_updates: Vec<(String, Value)>,
}

/// A timer operation emitted during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimerUpdate {
    Start(String),
    Stop(String),
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A snapshot of executor state for rollback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: usize,
    pub active_states: HashSet<u32>,
    pub timer_values: HashMap<String, f64>,
    pub variables: HashMap<String, Value>,
    pub timestamp: f64,
    pub history_len: usize,
}

// ---------------------------------------------------------------------------
// HistoryEntry
// ---------------------------------------------------------------------------

/// A single entry in the execution history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub step: usize,
    pub timestamp: f64,
    pub event: Option<EventKind>,
    pub fired_transitions: Vec<u32>,
    pub states_before: Vec<u32>,
    pub states_after: Vec<u32>,
}

// ---------------------------------------------------------------------------
// RuntimeExecutor
// ---------------------------------------------------------------------------

/// The core runtime executor.
#[derive(Debug)]
pub struct RuntimeExecutor {
    /// The automaton being executed.
    automaton: SpatialEventAutomaton,
    /// Currently active states (tokens).
    active_states: HashSet<u32>,
    /// Current timer values (timer_id → elapsed seconds).
    timer_values: HashMap<String, f64>,
    /// Current variable bindings.
    variables: HashMap<String, Value>,
    /// Execution mode.
    mode: ExecutionMode,
    /// Execution history.
    history: Vec<HistoryEntry>,
    /// Saved checkpoints.
    checkpoints: Vec<Checkpoint>,
    /// Step counter.
    step_count: usize,
    /// Next checkpoint id.
    next_checkpoint_id: usize,
}

impl RuntimeExecutor {
    /// Create a new executor for the given automaton.
    pub fn new(automaton: SpatialEventAutomaton) -> Self {
        let initial = automaton
            .initial_state
            .map(|s| {
                let mut set = HashSet::new();
                set.insert(s.0);
                set
            })
            .unwrap_or_default();

        Self {
            automaton,
            active_states: initial,
            timer_values: HashMap::new(),
            variables: HashMap::new(),
            mode: ExecutionMode::FirstMatch,
            history: Vec::new(),
            checkpoints: Vec::new(),
            step_count: 0,
            next_checkpoint_id: 0,
        }
    }

    /// Set the execution mode.
    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Get the current set of active states.
    pub fn active_states(&self) -> &HashSet<u32> {
        &self.active_states
    }

    /// Get the execution history.
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Get the current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Check whether the executor has reached an accepting configuration.
    pub fn is_accepting(&self) -> bool {
        self.active_states
            .iter()
            .any(|s| self.automaton.accepting_states.contains(&StateId(*s)))
    }

    /// Check whether the executor is stuck (no active states).
    pub fn is_stuck(&self) -> bool {
        self.active_states.is_empty()
    }

    // -----------------------------------------------------------------------
    // Stepping
    // -----------------------------------------------------------------------

    /// Perform one execution step given an event and scene configuration.
    pub fn step(
        &mut self,
        event: Option<&EventKind>,
        timestamp: f64,
        scene: &SceneConfiguration,
    ) -> ExecutionResult {
        let states_before: Vec<u32> = self.active_states.iter().copied().collect();

        let timer_map: HashMap<TimerId, f64> = self
            .timer_values
            .iter()
            .map(|(k, v)| (TimerId(k.clone()), *v))
            .collect();

        let time = TimePoint(timestamp);

        let mut new_states = HashSet::new();
        let mut fired_transitions = Vec::new();
        let mut all_actions: Vec<Action> = Vec::new();
        let mut timer_updates = Vec::new();
        let mut var_updates = Vec::new();

        for &state_id in &states_before {
            let sid = StateId(state_id);
            let enabled = self.automaton.enabled_transitions(
                sid,
                scene,
                time,
                event,
                &timer_map,
            );

            if enabled.is_empty() {
                // Token stays in current state
                new_states.insert(state_id);
                continue;
            }

            let to_fire: Vec<TransitionId> = match self.mode {
                ExecutionMode::FirstMatch => {
                    // Take only the first (highest-priority)
                    vec![enabled[0]]
                }
                ExecutionMode::AllMatches => enabled,
            };

            for tid in to_fire {
                if let Ok((target, actions)) = self.automaton.fire_transition(tid) {
                    new_states.insert(target.0);
                    fired_transitions.push(tid.0);
                    all_actions.extend(actions);
                }
            }
        }

        // Compute epsilon closure
        let closed = self.epsilon_closure(&new_states, scene, time, &timer_map);
        new_states = closed;

        // Process actions: extract timer and variable updates
        for action in &all_actions {
            match action {
                Action::StartTimer(tid) => {
                    timer_updates.push(TimerUpdate::Start(tid.0.clone()));
                    self.timer_values.insert(tid.0.clone(), 0.0);
                }
                Action::StopTimer(tid) => {
                    timer_updates.push(TimerUpdate::Stop(tid.0.clone()));
                    self.timer_values.remove(&tid.0);
                }
                Action::SetVar { var, value } => {
                    var_updates.push((var.0.clone(), value.clone()));
                    self.variables.insert(var.0.clone(), value.clone());
                }
                _ => {}
            }
        }

        // Record history
        let states_after: Vec<u32> = new_states.iter().copied().collect();
        self.history.push(HistoryEntry {
            step: self.step_count,
            timestamp,
            event: event.cloned(),
            fired_transitions: fired_transitions.clone(),
            states_before,
            states_after: states_after.clone(),
        });
        self.step_count += 1;
        self.active_states = new_states.clone();

        ExecutionResult {
            new_states,
            fired_transitions,
            emitted_actions: all_actions,
            timer_updates,
            var_updates,
        }
    }

    /// Advance timer values by a delta (seconds).
    pub fn advance_timers(&mut self, delta: f64) {
        for v in self.timer_values.values_mut() {
            *v += delta;
        }
    }

    // -----------------------------------------------------------------------
    // Epsilon closure
    // -----------------------------------------------------------------------

    /// Compute the epsilon closure of a set of states.
    fn epsilon_closure(
        &self,
        states: &HashSet<u32>,
        scene: &SceneConfiguration,
        time: TimePoint,
        timer_values: &HashMap<TimerId, f64>,
    ) -> HashSet<u32> {
        let mut result = states.clone();
        let mut queue: VecDeque<u32> = states.iter().copied().collect();

        while let Some(s) = queue.pop_front() {
            let sid = StateId(s);
            let enabled = self.automaton.enabled_transitions(
                sid,
                scene,
                time,
                Some(&EventKind::Epsilon),
                timer_values,
            );
            for tid in enabled {
                if let Some(t) = self.automaton.transition(tid) {
                    if result.insert(t.target.0) {
                        queue.push_back(t.target.0);
                    }
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Checkpointing
    // -----------------------------------------------------------------------

    /// Save a checkpoint of the current state.
    pub fn save_checkpoint(&mut self) -> usize {
        let id = self.next_checkpoint_id;
        self.next_checkpoint_id += 1;
        self.checkpoints.push(Checkpoint {
            id,
            active_states: self.active_states.clone(),
            timer_values: self.timer_values.clone(),
            variables: self.variables.clone(),
            timestamp: self
                .history
                .last()
                .map(|h| h.timestamp)
                .unwrap_or(0.0),
            history_len: self.history.len(),
        });
        id
    }

    /// Restore executor state from a checkpoint.
    pub fn restore_checkpoint(&mut self, id: usize) -> bool {
        if let Some(cp) = self.checkpoints.iter().find(|c| c.id == id).cloned() {
            self.active_states = cp.active_states;
            self.timer_values = cp.timer_values;
            self.variables = cp.variables;
            self.history.truncate(cp.history_len);
            self.step_count = cp.history_len;
            true
        } else {
            false
        }
    }

    /// List all saved checkpoints.
    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Reset the executor to the initial state.
    pub fn reset(&mut self) {
        let initial = self
            .automaton
            .initial_state
            .map(|s| {
                let mut set = HashSet::new();
                set.insert(s.0);
                set
            })
            .unwrap_or_default();
        self.active_states = initial;
        self.timer_values.clear();
        self.variables.clear();
        self.history.clear();
        self.checkpoints.clear();
        self.step_count = 0;
        self.next_checkpoint_id = 0;
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Get the current value of a variable.
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// Get the current value of a timer.
    pub fn get_timer(&self, name: &str) -> Option<f64> {
        self.timer_values.get(name).copied()
    }

    /// Get a reference to the underlying automaton.
    pub fn automaton(&self) -> &SpatialEventAutomaton {
        &self.automaton
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_automata::automaton::{State, Transition};
    use choreo_automata::Guard;
    use choreo_automata::VarId;

    fn make_automaton(
        n_states: u32,
        edges: &[(u32, u32, Guard, Vec<Action>)],
        initial: u32,
        accepting: &[u32],
    ) -> SpatialEventAutomaton {
        let mut aut = SpatialEventAutomaton::new("test_exec");
        for i in 0..n_states {
            let mut s = State::new(StateId(i), format!("s{}", i));
            if i == initial {
                s.is_initial = true;
            }
            if accepting.contains(&i) {
                s.is_accepting = true;
            }
            aut.add_state(s);
        }
        for (idx, (src, tgt, guard, actions)) in edges.iter().enumerate() {
            let t = Transition::new(
                TransitionId(idx as u32),
                StateId(*src),
                StateId(*tgt),
                guard.clone(),
                actions.clone(),
            );
            aut.add_transition(t);
        }
        aut
    }

    fn empty_scene() -> SceneConfiguration {
        SceneConfiguration::empty()
    }

    #[test]
    fn initial_state() {
        let aut = make_automaton(2, &[], 0, &[]);
        let exec = RuntimeExecutor::new(aut);
        assert!(exec.active_states().contains(&0));
        assert_eq!(exec.active_states().len(), 1);
    }

    #[test]
    fn simple_step() {
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart), vec![])],
            0,
            &[1],
        );
        let mut exec = RuntimeExecutor::new(aut);
        let result = exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        assert!(result.new_states.contains(&1));
        assert!(!result.new_states.contains(&0));
        assert!(exec.is_accepting());
    }

    #[test]
    fn no_matching_event_stays() {
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart), vec![])],
            0,
            &[1],
        );
        let mut exec = RuntimeExecutor::new(aut);
        let result = exec.step(Some(&EventKind::TouchStart), 1.0, &empty_scene());
        assert!(result.new_states.contains(&0));
        assert!(!result.new_states.contains(&1));
    }

    #[test]
    fn all_matches_mode() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), vec![]),
                (0, 2, Guard::Event(EventKind::GrabStart), vec![]),
            ],
            0,
            &[1, 2],
        );
        let mut exec = RuntimeExecutor::new(aut).with_mode(ExecutionMode::AllMatches);
        let result = exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        assert!(result.new_states.contains(&1));
        assert!(result.new_states.contains(&2));
    }

    #[test]
    fn first_match_mode() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), vec![]),
                (0, 2, Guard::Event(EventKind::GrabStart), vec![]),
            ],
            0,
            &[1, 2],
        );
        let mut exec = RuntimeExecutor::new(aut).with_mode(ExecutionMode::FirstMatch);
        let result = exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        assert_eq!(result.fired_transitions.len(), 1);
    }

    #[test]
    fn multi_step_sequence() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), vec![]),
                (1, 2, Guard::Event(EventKind::GrabEnd), vec![]),
            ],
            0,
            &[2],
        );
        let mut exec = RuntimeExecutor::new(aut);
        exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        exec.step(Some(&EventKind::GrabEnd), 2.0, &empty_scene());
        assert!(exec.active_states().contains(&2));
        assert!(exec.is_accepting());
        assert_eq!(exec.step_count(), 2);
    }

    #[test]
    fn checkpoint_and_rollback() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), vec![]),
                (1, 2, Guard::Event(EventKind::GrabEnd), vec![]),
            ],
            0,
            &[2],
        );
        let mut exec = RuntimeExecutor::new(aut);
        exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        let cp = exec.save_checkpoint();
        exec.step(Some(&EventKind::GrabEnd), 2.0, &empty_scene());
        assert!(exec.active_states().contains(&2));

        assert!(exec.restore_checkpoint(cp));
        assert!(exec.active_states().contains(&1));
        assert!(!exec.active_states().contains(&2));
    }

    #[test]
    fn timer_actions() {
        let actions = vec![Action::StartTimer(TimerId("t1".into()))];
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart), actions)],
            0,
            &[1],
        );
        let mut exec = RuntimeExecutor::new(aut);
        let result = exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        assert!(result
            .timer_updates
            .iter()
            .any(|u| matches!(u, TimerUpdate::Start(ref s) if s == "t1")));
        assert_eq!(exec.get_timer("t1"), Some(0.0));

        exec.advance_timers(2.5);
        assert_eq!(exec.get_timer("t1"), Some(2.5));
    }

    #[test]
    fn variable_actions() {
        let actions = vec![Action::SetVar {
            var: VarId("score".into()),
            value: Value::Int(42),
        }];
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart), actions)],
            0,
            &[1],
        );
        let mut exec = RuntimeExecutor::new(aut);
        exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        assert_eq!(exec.get_variable("score"), Some(&Value::Int(42)));
    }

    #[test]
    fn reset_clears_state() {
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart), vec![])],
            0,
            &[1],
        );
        let mut exec = RuntimeExecutor::new(aut);
        exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        assert!(exec.active_states().contains(&1));
        exec.reset();
        assert!(exec.active_states().contains(&0));
        assert_eq!(exec.step_count(), 0);
    }

    #[test]
    fn history_recording() {
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart), vec![])],
            0,
            &[1],
        );
        let mut exec = RuntimeExecutor::new(aut);
        exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        assert_eq!(exec.history().len(), 1);
        assert_eq!(exec.history()[0].states_before, vec![0]);
        assert!(exec.history()[0].states_after.contains(&1));
    }

    #[test]
    fn stuck_when_no_active_states() {
        // Create automaton with no transitions and no initial
        let mut aut = SpatialEventAutomaton::new("empty");
        let exec = RuntimeExecutor::new(aut);
        assert!(exec.is_stuck());
    }

    #[test]
    fn epsilon_closure_follows_epsilon() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), vec![]),
                (1, 2, Guard::Event(EventKind::Epsilon), vec![]),
            ],
            0,
            &[2],
        );
        let mut exec = RuntimeExecutor::new(aut);
        let result = exec.step(Some(&EventKind::GrabStart), 1.0, &empty_scene());
        // After firing 0→1, epsilon closure should reach 2
        assert!(result.new_states.contains(&2));
    }
}
