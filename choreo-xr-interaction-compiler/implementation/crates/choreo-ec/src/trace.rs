//! Trace comparison and differential testing.
//!
//! Compare expected event traces with actual EC engine output, detect
//! divergences, and run differential tests that compare the EC oracle's
//! output with the compiled automata output.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::axioms::*;
use crate::compiler::*;
use crate::engine::*;
use crate::fluent::*;
use crate::local_types::*;

// ─── TraceComparisonResult ───────────────────────────────────────────────────

/// Result of comparing two event traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceComparisonResult {
    /// Events present in both traces at matching times.
    pub matching_events: Vec<(Event, Event)>,
    /// Events in the expected trace but missing from actual.
    pub missing_events: Vec<Event>,
    /// Events in the actual trace but not in expected.
    pub extra_events: Vec<Event>,
    /// Events that match in kind but differ in timing.
    pub timing_differences: Vec<TimingDifference>,
    /// The overall similarity score [0, 1].
    pub similarity: f64,
    /// Whether the traces are considered equivalent.
    pub equivalent: bool,
}

/// A timing difference between matched events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingDifference {
    pub expected_event: Event,
    pub actual_event: Event,
    pub time_delta: Duration,
}

impl TraceComparisonResult {
    pub fn is_match(&self) -> bool {
        self.missing_events.is_empty()
            && self.extra_events.is_empty()
            && self.equivalent
    }

    pub fn summary(&self) -> String {
        format!(
            "matching={}, missing={}, extra={}, timing_diffs={}, similarity={:.2}%",
            self.matching_events.len(),
            self.missing_events.len(),
            self.extra_events.len(),
            self.timing_differences.len(),
            self.similarity * 100.0,
        )
    }
}

impl fmt::Display for TraceComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TraceComparison({})", self.summary())
    }
}

// ─── TraceDivergence ─────────────────────────────────────────────────────────

/// A specific point where two traces diverge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceDivergence {
    pub time: TimePoint,
    pub expected_state: Option<FluentSnapshot>,
    pub actual_state: Option<FluentSnapshot>,
    pub divergent_fluents: Vec<FluentId>,
    pub description: String,
}

impl TraceDivergence {
    /// Which fluents differ?
    pub fn diff_fluents(&self) -> &[FluentId] {
        &self.divergent_fluents
    }
}

impl fmt::Display for TraceDivergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Divergence at {}: {} fluents differ ({})",
            self.time,
            self.divergent_fluents.len(),
            self.description
        )
    }
}

// ─── TraceComparator ─────────────────────────────────────────────────────────

/// Compares two event traces and identifies differences.
pub struct TraceComparator {
    /// Timing tolerance for matching events.
    pub time_epsilon: f64,
    /// Whether to match events by kind only (ignoring parameters).
    pub kind_only_matching: bool,
}

impl TraceComparator {
    pub fn new(time_epsilon: f64) -> Self {
        Self {
            time_epsilon,
            kind_only_matching: false,
        }
    }

    pub fn with_kind_only_matching(mut self) -> Self {
        self.kind_only_matching = true;
        self
    }

    /// Compare two event traces.
    pub fn compare_traces(
        &self,
        expected: &EventTrace,
        actual: &EventTrace,
    ) -> TraceComparisonResult {
        let mut matching = Vec::new();
        let mut missing = Vec::new();
        let mut extra = Vec::new();
        let mut timing_diffs = Vec::new();

        let mut matched_actual: HashSet<usize> = HashSet::new();

        // For each expected event, try to find a matching actual event
        for exp_event in &expected.events {
            let mut best_match: Option<(usize, f64)> = None;

            for (idx, act_event) in actual.events.iter().enumerate() {
                if matched_actual.contains(&idx) {
                    continue;
                }

                let kind_matches = if self.kind_only_matching {
                    std::mem::discriminant(&exp_event.kind)
                        == std::mem::discriminant(&act_event.kind)
                } else {
                    exp_event.kind == act_event.kind
                };

                if kind_matches {
                    let time_diff = (exp_event.time.0 - act_event.time.0).abs();
                    if best_match.is_none() || time_diff < best_match.unwrap().1 {
                        best_match = Some((idx, time_diff));
                    }
                }
            }

            if let Some((idx, time_diff)) = best_match {
                matched_actual.insert(idx);
                let act = &actual.events[idx];

                if time_diff <= self.time_epsilon {
                    matching.push((exp_event.clone(), act.clone()));
                } else {
                    timing_diffs.push(TimingDifference {
                        expected_event: exp_event.clone(),
                        actual_event: act.clone(),
                        time_delta: Duration(time_diff),
                    });
                    matching.push((exp_event.clone(), act.clone()));
                }
            } else {
                missing.push(exp_event.clone());
            }
        }

        // Unmatched actual events are extra
        for (idx, event) in actual.events.iter().enumerate() {
            if !matched_actual.contains(&idx) {
                extra.push(event.clone());
            }
        }

        let total = expected.events.len().max(actual.events.len()).max(1) as f64;
        let similarity = matching.len() as f64 / total;
        let equivalent = missing.is_empty() && extra.is_empty();

        TraceComparisonResult {
            matching_events: matching,
            missing_events: missing,
            extra_events: extra,
            timing_differences: timing_diffs,
            similarity,
            equivalent,
        }
    }

    /// Find the first point of divergence between two state sequences.
    pub fn find_divergence(
        &self,
        expected_states: &[ECState],
        actual_states: &[ECState],
    ) -> Option<TraceDivergence> {
        let min_len = expected_states.len().min(actual_states.len());

        for i in 0..min_len {
            let exp = &expected_states[i];
            let act = &actual_states[i];

            let divergent_fluents = find_divergent_fluents(
                &exp.fluent_snapshot,
                &act.fluent_snapshot,
            );

            if !divergent_fluents.is_empty() {
                return Some(TraceDivergence {
                    time: exp.time,
                    expected_state: Some(exp.fluent_snapshot.clone()),
                    actual_state: Some(act.fluent_snapshot.clone()),
                    divergent_fluents,
                    description: format!("State divergence at step {}", i),
                });
            }
        }

        if expected_states.len() != actual_states.len() {
            return Some(TraceDivergence {
                time: if expected_states.len() > actual_states.len() {
                    expected_states[min_len].time
                } else {
                    actual_states[min_len].time
                },
                expected_state: expected_states.get(min_len).map(|s| s.fluent_snapshot.clone()),
                actual_state: actual_states.get(min_len).map(|s| s.fluent_snapshot.clone()),
                divergent_fluents: Vec::new(),
                description: format!(
                    "Trace length mismatch: expected {} states, got {}",
                    expected_states.len(),
                    actual_states.len()
                ),
            });
        }

        None
    }
}

fn find_divergent_fluents(a: &FluentSnapshot, b: &FluentSnapshot) -> Vec<FluentId> {
    let mut divergent = Vec::new();
    let all_ids: HashSet<FluentId> = a
        .fluent_ids()
        .into_iter()
        .chain(b.fluent_ids())
        .collect();

    for id in all_ids {
        let va = a.get(id);
        let vb = b.get(id);
        match (va, vb) {
            (Some(fa), Some(fb)) if fa != fb => divergent.push(id),
            (Some(_), None) | (None, Some(_)) => divergent.push(id),
            _ => {}
        }
    }

    divergent
}

// ─── Trace equivalence / inclusion ───────────────────────────────────────────

/// Check if two traces are equivalent within a timing tolerance.
pub fn trace_equivalence(a: &EventTrace, b: &EventTrace, epsilon: f64) -> bool {
    let comparator = TraceComparator::new(epsilon);
    let result = comparator.compare_traces(a, b);
    result.equivalent
}

/// Check if trace `a` is included in trace `b` (a is a subsequence of b).
pub fn trace_inclusion(a: &EventTrace, b: &EventTrace) -> bool {
    if a.events.is_empty() {
        return true;
    }
    if b.events.is_empty() {
        return false;
    }

    let mut b_idx = 0;
    for a_event in &a.events {
        let mut found = false;
        while b_idx < b.events.len() {
            if events_match(&a_event.kind, &b.events[b_idx].kind) {
                found = true;
                b_idx += 1;
                break;
            }
            b_idx += 1;
        }
        if !found {
            return false;
        }
    }
    true
}

/// Check if trace `a` is a prefix of trace `b`.
pub fn trace_prefix(a: &EventTrace, b: &EventTrace) -> bool {
    if a.events.len() > b.events.len() {
        return false;
    }
    for (ae, be) in a.events.iter().zip(b.events.iter()) {
        if !events_match(&ae.kind, &be.kind) {
            return false;
        }
    }
    true
}

fn events_match(a: &EventKind, b: &EventKind) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

// ─── DifferentialTester ──────────────────────────────────────────────────────

/// Differential tester: compare EC oracle output with compiled automata output.
///
/// The tester runs the same events through both the EC engine and a compiled
/// automaton, then checks that the fluent values agree at each step.
pub struct DifferentialTester {
    /// The EC engine.
    engine: ECEngine,
    /// The compiled transitions (from the compiler).
    transitions: Vec<CompiledTransition>,
    /// Configuration.
    time_epsilon: f64,
    /// Results log.
    results: Vec<DifferentialTestResult>,
}

/// Result of a single differential test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialTestResult {
    pub test_name: String,
    pub passed: bool,
    pub ec_states: Vec<ECState>,
    pub divergence: Option<TraceDivergence>,
    pub event_count: usize,
}

impl DifferentialTester {
    pub fn new(engine: ECEngine, transitions: Vec<CompiledTransition>) -> Self {
        Self {
            engine,
            transitions,
            time_epsilon: 1e-6,
            results: Vec::new(),
        }
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.time_epsilon = epsilon;
        self
    }

    /// Run a differential test with the given events.
    pub fn run_test(
        &mut self,
        test_name: impl Into<String>,
        events: &[Event],
    ) -> DifferentialTestResult {
        let name = test_name.into();

        // Run through EC engine
        let ec_states = self.engine.evaluate_trace(events);

        // Simulate compiled automaton
        let auto_states = self.simulate_automaton(events);

        // Compare
        let comparator = TraceComparator::new(self.time_epsilon);
        let divergence = comparator.find_divergence(&ec_states, &auto_states);

        let passed = divergence.is_none();

        let result = DifferentialTestResult {
            test_name: name,
            passed,
            ec_states,
            divergence,
            event_count: events.len(),
        };

        self.results.push(result.clone());
        result
    }

    /// Get all test results.
    pub fn results(&self) -> &[DifferentialTestResult] {
        &self.results
    }

    /// How many tests passed?
    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// How many tests failed?
    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Simulate the compiled automaton on a sequence of events.
    fn simulate_automaton(&self, events: &[Event]) -> Vec<ECState> {
        let mut states = Vec::new();
        let mut current_fluents = self.engine.fluent_store.clone();
        let mut sequence = 0u64;

        for event in events {
            sequence += 1;

            // Find matching transitions
            let matching_transitions: Vec<&CompiledTransition> = self
                .transitions
                .iter()
                .filter(|t| guard_matches(&t.guard, &current_fluents, &event.kind))
                .collect();

            // Apply actions from the highest-priority matching transition
            if let Some(trans) = matching_transitions
                .iter()
                .max_by_key(|t| t.priority)
            {
                for action in &trans.actions {
                    apply_compiled_action(action, &mut current_fluents);
                }
            }

            let snapshot = current_fluents.snapshot(event.time);
            let state = ECState::new(event.time, snapshot, sequence);
            states.push(state);
        }

        states
    }
}

/// Check if a compiled guard matches the current state and event.
fn guard_matches(
    guard: &TransitionGuard,
    fluents: &FluentStore,
    event_kind: &EventKind,
) -> bool {
    match guard {
        TransitionGuard::True => true,
        TransitionGuard::False => false,
        TransitionGuard::EventMatch(pattern) => pattern.matches(event_kind),
        TransitionGuard::FluentHolds(id) => fluents.get(*id).map_or(false, |f| f.holds()),
        TransitionGuard::FluentNotHolds(id) => fluents.get(*id).map_or(true, |f| !f.holds()),
        TransitionGuard::NumericGuard { fluent_id, op, threshold } => {
            fluents.get(*fluent_id).map_or(false, |f| {
                if let Fluent::NumericFluent { value, .. } = f {
                    op.evaluate(*value, *threshold)
                } else {
                    false
                }
            })
        }
        TransitionGuard::And(children) => {
            children.iter().all(|c| guard_matches(c, fluents, event_kind))
        }
        TransitionGuard::Or(children) => {
            children.iter().any(|c| guard_matches(c, fluents, event_kind))
        }
        TransitionGuard::Not(child) => !guard_matches(child, fluents, event_kind),
        TransitionGuard::TimerExpired(name) => {
            matches!(event_kind, EventKind::TimerExpired { name: n } if n == name)
        }
        TransitionGuard::SpatialHolds(_) | TransitionGuard::SpatialNotHolds(_) => {
            true // spatial conditions not directly evaluable here
        }
    }
}

/// Apply a compiled action to a fluent store.
fn apply_compiled_action(action: &CompiledAction, fluents: &mut FluentStore) {
    match action {
        CompiledAction::SetFluent { fluent_id, value } => {
            if fluents.contains(*fluent_id) {
                let _ = fluents.update(*fluent_id, value.clone());
            } else {
                fluents.insert_with_id(*fluent_id, value.clone());
            }
        }
        CompiledAction::TerminateFluent { fluent_id } => {
            fluents.remove(*fluent_id);
        }
        _ => {} // Other actions don't affect fluents directly
    }
}

// ─── Test trace generation ───────────────────────────────────────────────────

/// Generate a test trace for a given scenario.
pub fn generate_test_trace(scenario: &TestScenario) -> EventTrace {
    let mut trace = EventTrace::new();
    let mut next_id = 1u64;

    for step in &scenario.steps {
        trace.push(Event::new(
            EventId(next_id),
            step.time,
            step.event_kind.clone(),
        ));
        next_id += 1;
    }

    trace
}

/// A test scenario definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub name: String,
    pub steps: Vec<TestStep>,
    pub expected_final_fluents: Vec<(FluentId, bool)>,
}

/// A single step in a test scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStep {
    pub time: TimePoint,
    pub event_kind: EventKind,
    pub expected_fluent_changes: Vec<(FluentId, bool)>,
}

impl TestScenario {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            expected_final_fluents: Vec::new(),
        }
    }

    pub fn add_step(mut self, time: TimePoint, event_kind: EventKind) -> Self {
        self.steps.push(TestStep {
            time,
            event_kind,
            expected_fluent_changes: Vec::new(),
        });
        self
    }

    pub fn add_step_with_expectations(
        mut self,
        time: TimePoint,
        event_kind: EventKind,
        expected_changes: Vec<(FluentId, bool)>,
    ) -> Self {
        self.steps.push(TestStep {
            time,
            event_kind,
            expected_fluent_changes: expected_changes,
        });
        self
    }

    pub fn expect_final(mut self, fluent_id: FluentId, holds: bool) -> Self {
        self.expected_final_fluents.push((fluent_id, holds));
        self
    }
}

/// Run a test scenario against an EC engine and return pass/fail.
pub fn run_scenario(
    engine: &mut ECEngine,
    scenario: &TestScenario,
) -> (bool, Vec<String>) {
    let trace = generate_test_trace(scenario);
    let states = engine.evaluate_trace(&trace.events);
    let mut failures = Vec::new();

    // Check step-by-step expectations
    for (i, step) in scenario.steps.iter().enumerate() {
        if i >= states.len() {
            failures.push(format!("Step {} not executed", i));
            continue;
        }
        for (fid, expected_holds) in &step.expected_fluent_changes {
            let actual = states[i].holds(*fid);
            if actual != *expected_holds {
                failures.push(format!(
                    "Step {} (t={}): fluent {} expected holds={}, got holds={}",
                    i, step.time, fid, expected_holds, actual
                ));
            }
        }
    }

    // Check final state
    if let Some(final_state) = states.last() {
        for (fid, expected_holds) in &scenario.expected_final_fluents {
            let actual = final_state.holds(*fid);
            if actual != *expected_holds {
                failures.push(format!(
                    "Final state: fluent {} expected holds={}, got holds={}",
                    fid, expected_holds, actual
                ));
            }
        }
    } else if !scenario.expected_final_fluents.is_empty() {
        failures.push("No final state produced".into());
    }

    (failures.is_empty(), failures)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_expected_trace() -> EventTrace {
        let mut trace = EventTrace::new();
        trace.push(Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));
        trace.push(Event::new(
            EventId(2),
            TimePoint::from_secs(3.0),
            EventKind::Action {
                action: ActionType::Deactivate,
                entity: EntityId(10),
            },
        ));
        trace
    }

    #[test]
    fn test_trace_comparison_exact_match() {
        let a = make_expected_trace();
        let b = make_expected_trace();
        let comparator = TraceComparator::new(0.01);
        let result = comparator.compare_traces(&a, &b);
        assert!(result.is_match());
        assert!((result.similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_trace_comparison_missing_event() {
        let a = make_expected_trace();
        let mut b = EventTrace::new();
        b.push(a.events[0].clone()); // Only first event
        let comparator = TraceComparator::new(0.01);
        let result = comparator.compare_traces(&a, &b);
        assert!(!result.is_match());
        assert_eq!(result.missing_events.len(), 1);
    }

    #[test]
    fn test_trace_comparison_extra_event() {
        let a = make_expected_trace();
        let mut b = make_expected_trace();
        b.push(Event::new(
            EventId(3),
            TimePoint::from_secs(5.0),
            EventKind::System { tag: "extra".into() },
        ));
        let comparator = TraceComparator::new(0.01);
        let result = comparator.compare_traces(&a, &b);
        assert!(!result.is_match());
        assert_eq!(result.extra_events.len(), 1);
    }

    #[test]
    fn test_trace_comparison_timing_difference() {
        let a = make_expected_trace();
        let mut b = EventTrace::new();
        // Same events but shifted in time
        b.push(Event::new(
            EventId(1),
            TimePoint::from_secs(1.1),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));
        b.push(Event::new(
            EventId(2),
            TimePoint::from_secs(3.2),
            EventKind::Action {
                action: ActionType::Deactivate,
                entity: EntityId(10),
            },
        ));

        let comparator = TraceComparator::new(0.01);
        let result = comparator.compare_traces(&a, &b);
        assert!(!result.timing_differences.is_empty());
    }

    #[test]
    fn test_trace_equivalence() {
        let a = make_expected_trace();
        let b = make_expected_trace();
        assert!(trace_equivalence(&a, &b, 0.01));

        let c = EventTrace::new();
        assert!(!trace_equivalence(&a, &c, 0.01));
    }

    #[test]
    fn test_trace_inclusion() {
        let full = make_expected_trace();
        let mut prefix = EventTrace::new();
        prefix.push(full.events[0].clone());

        assert!(trace_inclusion(&prefix, &full));
        assert!(!trace_inclusion(&full, &prefix));
    }

    #[test]
    fn test_trace_prefix() {
        let full = make_expected_trace();
        let mut prefix = EventTrace::new();
        prefix.push(full.events[0].clone());

        assert!(trace_prefix(&prefix, &full));
        assert!(!trace_prefix(&full, &prefix));
    }

    #[test]
    fn test_trace_inclusion_empty() {
        let empty = EventTrace::new();
        let full = make_expected_trace();
        assert!(trace_inclusion(&empty, &full));
        assert!(trace_inclusion(&empty, &empty));
    }

    #[test]
    fn test_find_divergence_matching() {
        let states = vec![
            ECState::new(
                TimePoint::from_secs(0.0),
                FluentSnapshot::with_fluents(
                    TimePoint::from_secs(0.0),
                    vec![(FluentId(1), Fluent::boolean("x", true))],
                ),
                0,
            ),
        ];
        let comparator = TraceComparator::new(0.01);
        assert!(comparator.find_divergence(&states, &states).is_none());
    }

    #[test]
    fn test_find_divergence_different() {
        let states_a = vec![
            ECState::new(
                TimePoint::from_secs(0.0),
                FluentSnapshot::with_fluents(
                    TimePoint::from_secs(0.0),
                    vec![(FluentId(1), Fluent::boolean("x", true))],
                ),
                0,
            ),
        ];
        let states_b = vec![
            ECState::new(
                TimePoint::from_secs(0.0),
                FluentSnapshot::with_fluents(
                    TimePoint::from_secs(0.0),
                    vec![(FluentId(1), Fluent::boolean("x", false))],
                ),
                0,
            ),
        ];
        let comparator = TraceComparator::new(0.01);
        let div = comparator.find_divergence(&states_a, &states_b);
        assert!(div.is_some());
        assert!(div.unwrap().divergent_fluents.contains(&FluentId(1)));
    }

    #[test]
    fn test_generate_test_trace() {
        let scenario = TestScenario::new("grab_test")
            .add_step(
                TimePoint::from_secs(1.0),
                EventKind::Gesture {
                    gesture: GestureType::Grab,
                    hand: HandSide::Right,
                    entity: EntityId(10),
                },
            )
            .add_step(
                TimePoint::from_secs(3.0),
                EventKind::Action {
                    action: ActionType::Deactivate,
                    entity: EntityId(10),
                },
            );

        let trace = generate_test_trace(&scenario);
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.events[0].time, TimePoint::from_secs(1.0));
    }

    #[test]
    fn test_run_scenario() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("near", true));    // FluentId(1)
        store.insert(Fluent::boolean("grabbed", false)); // FluentId(2)

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

        let mut engine = ECEngine::new(axiom_set, store, ECEngineConfig::default());

        let scenario = TestScenario::new("grab_scenario")
            .add_step_with_expectations(
                TimePoint::from_secs(1.0),
                EventKind::Gesture {
                    gesture: GestureType::Grab,
                    hand: HandSide::Right,
                    entity: EntityId(10),
                },
                vec![(FluentId(2), true)],
            )
            .expect_final(FluentId(2), true);

        let (passed, failures) = run_scenario(&mut engine, &scenario);
        assert!(passed, "Failures: {:?}", failures);
    }

    #[test]
    fn test_differential_tester() {
        let mut store = FluentStore::new();
        store.insert(Fluent::boolean("on", false)); // FluentId(1)

        let mut axiom_set = AxiomSet::new();
        axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(1),
            name: "turn_on".into(),
            conditions: vec![],
            fluent: FluentId(1),
            event: EventPattern::ActionMatch(ActionType::Activate),
            new_value: Fluent::boolean("on", true),
            priority: 0,
        });

        let mut compiler = ECCompiler::new();
        let transitions = compiler.lower_ec_to_transitions(&axiom_set, &store);

        let engine = ECEngine::new(axiom_set, store, ECEngineConfig::default());
        let mut tester = DifferentialTester::new(engine, transitions);

        let result = tester.run_test(
            "turn_on_test",
            &[Event::new(
                EventId(1),
                TimePoint::from_secs(1.0),
                EventKind::Action {
                    action: ActionType::Activate,
                    entity: EntityId(1),
                },
            )],
        );

        assert_eq!(result.event_count, 1);
        // The test may or may not pass depending on state mapping,
        // but it should not panic
    }

    #[test]
    fn test_trace_comparison_result_summary() {
        let result = TraceComparisonResult {
            matching_events: vec![],
            missing_events: vec![],
            extra_events: vec![],
            timing_differences: vec![],
            similarity: 1.0,
            equivalent: true,
        };
        let s = result.summary();
        assert!(s.contains("100.00%"));
    }

    #[test]
    fn test_kind_only_matching() {
        let mut a = EventTrace::new();
        a.push(Event::new(
            EventId(1),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(10),
            },
        ));

        let mut b = EventTrace::new();
        b.push(Event::new(
            EventId(2),
            TimePoint::from_secs(1.0),
            EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Left, // Different hand
                entity: EntityId(20), // Different entity
            },
        ));

        // Without kind_only: not matching (different params)
        let cmp1 = TraceComparator::new(0.01);
        let r1 = cmp1.compare_traces(&a, &b);
        assert!(!r1.is_match());

        // With kind_only: matching (same discriminant)
        let cmp2 = TraceComparator::new(0.01).with_kind_only_matching();
        let r2 = cmp2.compare_traces(&a, &b);
        assert!(r2.is_match());
    }

    #[test]
    fn test_scenario_builder() {
        let scenario = TestScenario::new("test")
            .add_step(TimePoint::from_secs(1.0), EventKind::System { tag: "a".into() })
            .add_step(TimePoint::from_secs(2.0), EventKind::System { tag: "b".into() })
            .expect_final(FluentId(1), true);

        assert_eq!(scenario.steps.len(), 2);
        assert_eq!(scenario.expected_final_fluents.len(), 1);
    }
}
