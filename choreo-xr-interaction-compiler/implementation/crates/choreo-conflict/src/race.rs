//! Race condition detection for spatial-event automata.
//!
//! Identifies states with multiple simultaneously enabled transitions,
//! analyses guard overlap to classify severity, and detects temporal races
//! from overlapping timing windows.

use choreo_automata::automaton::{SpatialEventAutomaton, Transition};
use choreo_automata::{Guard, SpatialPredicate, StateId, TransitionId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// How severe a race condition is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RaceSeverity {
    /// Benign: guards are mutually exclusive in practice.
    Benign,
    /// Warning: guards *may* overlap under certain spatial configurations.
    Warning,
    /// Critical: guards overlap unconditionally.
    Critical,
}

impl fmt::Display for RaceSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Benign => write!(f, "benign"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// RaceCondition
// ---------------------------------------------------------------------------

/// A detected race condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceCondition {
    /// The state in which the race occurs.
    pub state_id: u32,
    /// Ids of the conflicting transitions.
    pub conflicting: Vec<u32>,
    /// Description of the spatial condition leading to the race.
    pub spatial_condition: String,
    /// Severity classification.
    pub severity: RaceSeverity,
    /// Whether this race is on the same event kind.
    pub same_event: bool,
    /// Whether priority ordering resolves the race.
    pub resolved_by_priority: bool,
}

// ---------------------------------------------------------------------------
// Overlap analysis helpers
// ---------------------------------------------------------------------------

/// Classify the overlap between two guards.
fn classify_guard_overlap(g1: &Guard, g2: &Guard) -> RaceSeverity {
    // Check if guards are trivially exclusive
    if are_guards_exclusive(g1, g2) {
        return RaceSeverity::Benign;
    }

    // Check if one guard is the negation of the other
    if is_negation(g1, g2) {
        return RaceSeverity::Benign;
    }

    // Check if both are True or have no spatial component
    let g1_spatial = has_spatial_component(g1);
    let g2_spatial = has_spatial_component(g2);

    if !g1_spatial && !g2_spatial {
        // Both are purely event-based — critical if same event
        if guards_share_event(g1, g2) {
            return RaceSeverity::Critical;
        }
        return RaceSeverity::Benign;
    }

    // Both have spatial components — check for overlap
    let g1_preds = collect_spatial_pred_strings(g1);
    let g2_preds = collect_spatial_pred_strings(g2);
    let shared: HashSet<_> = g1_preds.intersection(&g2_preds).collect();

    if shared.is_empty() {
        // Different spatial conditions — may overlap
        RaceSeverity::Warning
    } else {
        // Same spatial conditions referenced — check polarity
        let g1_neg = collect_negated_spatial_strings(g1);
        let g2_neg = collect_negated_spatial_strings(g2);

        let contradicted = shared.iter().any(|p| {
            (g1_neg.contains(*p) && !g2_neg.contains(*p))
                || (!g1_neg.contains(*p) && g2_neg.contains(*p))
        });

        if contradicted {
            RaceSeverity::Benign
        } else {
            RaceSeverity::Critical
        }
    }
}

fn are_guards_exclusive(g1: &Guard, g2: &Guard) -> bool {
    match (g1, g2) {
        (Guard::True, Guard::False) | (Guard::False, Guard::True) => true,
        (Guard::False, _) | (_, Guard::False) => true,
        (Guard::Event(e1), Guard::Event(e2)) => e1 != e2,
        _ => false,
    }
}

fn is_negation(g1: &Guard, g2: &Guard) -> bool {
    match (g1, g2) {
        (Guard::Not(inner), other) | (other, Guard::Not(inner)) => inner.as_ref() == other,
        _ => false,
    }
}

fn has_spatial_component(guard: &Guard) -> bool {
    match guard {
        Guard::Spatial(_) => true,
        Guard::And(gs) | Guard::Or(gs) => gs.iter().any(has_spatial_component),
        Guard::Not(g) => has_spatial_component(g),
        _ => false,
    }
}

fn guards_share_event(g1: &Guard, g2: &Guard) -> bool {
    let e1 = collect_events(g1);
    let e2 = collect_events(g2);
    e1.intersection(&e2).next().is_some()
}

fn collect_events(guard: &Guard) -> HashSet<String> {
    let mut evts = HashSet::new();
    match guard {
        Guard::Event(ek) => {
            evts.insert(format!("{}", ek));
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                evts.extend(collect_events(g));
            }
        }
        Guard::Not(g) => {
            evts.extend(collect_events(g));
        }
        _ => {}
    }
    evts
}

fn collect_spatial_pred_strings(guard: &Guard) -> HashSet<String> {
    let mut preds = HashSet::new();
    match guard {
        Guard::Spatial(sp) => {
            preds.insert(format!("{:?}", sp));
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                preds.extend(collect_spatial_pred_strings(g));
            }
        }
        Guard::Not(g) => {
            preds.extend(collect_spatial_pred_strings(g));
        }
        _ => {}
    }
    preds
}

fn collect_negated_spatial_strings(guard: &Guard) -> HashSet<String> {
    let mut preds = HashSet::new();
    match guard {
        Guard::Not(inner) => {
            if let Guard::Spatial(sp) = inner.as_ref() {
                preds.insert(format!("{:?}", sp));
            }
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                preds.extend(collect_negated_spatial_strings(g));
            }
        }
        _ => {}
    }
    preds
}

/// Check whether two transitions fire on the same event kind.
fn transitions_share_event(t1: &Transition, t2: &Transition) -> bool {
    guards_share_event(&t1.guard, &t2.guard)
}

/// Check whether priorities resolve a set of conflicting transitions.
fn priorities_resolve(transitions: &[&Transition]) -> bool {
    if transitions.len() <= 1 {
        return true;
    }
    let priorities: Vec<i32> = transitions.iter().map(|t| t.priority).collect();
    let max = priorities.iter().max().copied().unwrap_or(0);
    let count_max = priorities.iter().filter(|&&p| p == max).count();
    count_max == 1
}

// ---------------------------------------------------------------------------
// RaceDetector
// ---------------------------------------------------------------------------

/// Main race condition detector.
#[derive(Debug)]
pub struct RaceDetector {
    min_severity: RaceSeverity,
    ignore_priority_resolved: bool,
}

impl RaceDetector {
    pub fn new() -> Self {
        Self {
            min_severity: RaceSeverity::Benign,
            ignore_priority_resolved: false,
        }
    }

    /// Only report races at or above this severity.
    pub fn min_severity(mut self, sev: RaceSeverity) -> Self {
        self.min_severity = sev;
        self
    }

    /// If set, races that are resolved by transition priority are excluded.
    pub fn ignore_priority_resolved(mut self, yes: bool) -> Self {
        self.ignore_priority_resolved = yes;
        self
    }

    /// Detect race conditions in the given automaton.
    pub fn detect(&self, automaton: &SpatialEventAutomaton) -> Vec<RaceCondition> {
        let states: Vec<StateId> = automaton.state_ids();
        let transitions: Vec<&Transition> = automaton.transitions.values().collect();
        self.detect_races(&states, &transitions)
    }

    /// Core detection from raw states and transitions.
    pub fn detect_races(
        &self,
        states: &[StateId],
        transitions: &[&Transition],
    ) -> Vec<RaceCondition> {
        let mut races = Vec::new();

        // Group transitions by source state.
        let mut by_source: HashMap<u32, Vec<&Transition>> = HashMap::new();
        for t in transitions {
            by_source.entry(t.source.0).or_default().push(t);
        }

        for &state in states {
            let outgoing = match by_source.get(&state.0) {
                Some(ts) => ts,
                None => continue,
            };

            if outgoing.len() < 2 {
                continue;
            }

            // Check all pairs
            for i in 0..outgoing.len() {
                for j in (i + 1)..outgoing.len() {
                    let t1 = outgoing[i];
                    let t2 = outgoing[j];

                    let severity = classify_guard_overlap(&t1.guard, &t2.guard);
                    if severity < self.min_severity {
                        continue;
                    }

                    let same_event = transitions_share_event(t1, t2);
                    let resolved = priorities_resolve(&[t1, t2]);

                    if self.ignore_priority_resolved && resolved {
                        continue;
                    }

                    let spatial_desc = if has_spatial_component(&t1.guard)
                        || has_spatial_component(&t2.guard)
                    {
                        format!(
                            "guards: ({}) vs ({})",
                            t1.guard, t2.guard
                        )
                    } else {
                        "no spatial component".to_string()
                    };

                    races.push(RaceCondition {
                        state_id: state.0,
                        conflicting: vec![t1.id.0, t2.id.0],
                        spatial_condition: spatial_desc,
                        severity,
                        same_event,
                        resolved_by_priority: resolved,
                    });
                }
            }
        }

        races
    }
}

impl Default for RaceDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Temporal race detection
// ---------------------------------------------------------------------------

/// A temporal race: two transitions with overlapping timing windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRace {
    pub state_id: u32,
    pub transition_a: u32,
    pub transition_b: u32,
    pub overlap_description: String,
}

/// Detect temporal races: transitions from the same state whose temporal
/// guards have overlapping windows.
pub fn detect_temporal_races(automaton: &SpatialEventAutomaton) -> Vec<TemporalRace> {
    let mut races = Vec::new();

    let mut by_source: HashMap<u32, Vec<&Transition>> = HashMap::new();
    for t in automaton.transitions.values() {
        by_source.entry(t.source.0).or_default().push(t);
    }

    for (sid, outgoing) in &by_source {
        for i in 0..outgoing.len() {
            for j in (i + 1)..outgoing.len() {
                let t1 = outgoing[i];
                let t2 = outgoing[j];

                let intervals1 = extract_temporal_intervals(&t1.guard);
                let intervals2 = extract_temporal_intervals(&t2.guard);

                for (lo1, hi1) in &intervals1 {
                    for (lo2, hi2) in &intervals2 {
                        if lo1 <= hi2 && lo2 <= hi1 {
                            races.push(TemporalRace {
                                state_id: *sid,
                                transition_a: t1.id.0,
                                transition_b: t2.id.0,
                                overlap_description: format!(
                                    "[{:.2},{:.2}] overlaps [{:.2},{:.2}]",
                                    lo1, hi1, lo2, hi2
                                ),
                            });
                        }
                    }
                }
            }
        }
    }

    races
}

/// Extract timing intervals from a guard's temporal components.
fn extract_temporal_intervals(guard: &Guard) -> Vec<(f64, f64)> {
    let mut intervals = Vec::new();
    match guard {
        Guard::Temporal(te) => {
            extract_from_temporal_expr(te, &mut intervals);
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                intervals.extend(extract_temporal_intervals(g));
            }
        }
        Guard::Not(g) => {
            intervals.extend(extract_temporal_intervals(g));
        }
        _ => {}
    }
    intervals
}

fn extract_from_temporal_expr(
    expr: &choreo_automata::TemporalGuardExpr,
    intervals: &mut Vec<(f64, f64)>,
) {
    use choreo_automata::TemporalGuardExpr;
    match expr {
        TemporalGuardExpr::TimerElapsed { threshold, .. } => {
            intervals.push((threshold.0, f64::INFINITY));
        }
        TemporalGuardExpr::WithinInterval(iv) => {
            intervals.push((iv.start.0, iv.end.0));
        }
        TemporalGuardExpr::And(es) | TemporalGuardExpr::Or(es) => {
            for e in es {
                extract_from_temporal_expr(e, intervals);
            }
        }
        TemporalGuardExpr::Not(inner) => {
            extract_from_temporal_expr(inner, intervals);
        }
        TemporalGuardExpr::Named(_) => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_automata::automaton::{State, Transition};

    fn make_automaton(
        n_states: u32,
        edges: &[(u32, u32, Guard, i32)],
        initial: u32,
    ) -> SpatialEventAutomaton {
        let mut aut = SpatialEventAutomaton::new("test_race");
        for i in 0..n_states {
            let mut s = State::new(StateId(i), format!("s{}", i));
            if i == initial {
                s.is_initial = true;
            }
            aut.add_state(s);
        }
        for (idx, (src, tgt, guard, priority)) in edges.iter().enumerate() {
            let mut t = Transition::new(
                TransitionId(idx as u32),
                StateId(*src),
                StateId(*tgt),
                guard.clone(),
                vec![],
            );
            t.priority = *priority;
            aut.add_transition(t);
        }
        aut
    }

    #[test]
    fn no_race_single_transition() {
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart), 0)],
            0,
        );
        let detector = RaceDetector::new();
        let races = detector.detect(&aut);
        assert!(races.is_empty());
    }

    #[test]
    fn race_on_same_event() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), 0),
                (0, 2, Guard::Event(EventKind::GrabStart), 0),
            ],
            0,
        );
        let detector = RaceDetector::new();
        let races = detector.detect(&aut);
        assert!(!races.is_empty());
        assert!(races[0].same_event);
        assert_eq!(races[0].severity, RaceSeverity::Critical);
    }

    #[test]
    fn no_race_different_events() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), 0),
                (0, 2, Guard::Event(EventKind::TouchStart), 0),
            ],
            0,
        );
        let detector = RaceDetector::new();
        let races = detector.detect(&aut);
        // Different events — benign
        assert!(races.iter().all(|r| r.severity == RaceSeverity::Benign));
    }

    #[test]
    fn race_resolved_by_priority() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), 10),
                (0, 2, Guard::Event(EventKind::GrabStart), 0),
            ],
            0,
        );
        let detector = RaceDetector::new().ignore_priority_resolved(true);
        let races = detector.detect(&aut);
        assert!(races.is_empty());
    }

    #[test]
    fn spatial_race_same_predicate() {
        let sp = SpatialPredicate::Inside {
            entity: choreo_automata::EntityId("e1".into()),
            region: choreo_automata::RegionId("r1".into()),
        };
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Spatial(sp.clone()), 0),
                (0, 2, Guard::Spatial(sp.clone()), 0),
            ],
            0,
        );
        let detector = RaceDetector::new();
        let races = detector.detect(&aut);
        assert!(!races.is_empty());
        assert_eq!(races[0].severity, RaceSeverity::Critical);
    }

    #[test]
    fn spatial_race_opposite_polarity() {
        let sp = SpatialPredicate::Inside {
            entity: choreo_automata::EntityId("e1".into()),
            region: choreo_automata::RegionId("r1".into()),
        };
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Spatial(sp.clone()), 0),
                (
                    0,
                    2,
                    Guard::Not(Box::new(Guard::Spatial(sp.clone()))),
                    0,
                ),
            ],
            0,
        );
        let detector = RaceDetector::new();
        let races = detector.detect(&aut);
        // Opposite polarity → benign
        assert!(races.iter().all(|r| r.severity == RaceSeverity::Benign));
    }

    #[test]
    fn severity_filtering() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart), 0),
                (0, 2, Guard::Event(EventKind::TouchStart), 0),
            ],
            0,
        );
        let detector = RaceDetector::new().min_severity(RaceSeverity::Warning);
        let races = detector.detect(&aut);
        assert!(races.iter().all(|r| r.severity >= RaceSeverity::Warning));
    }

    #[test]
    fn temporal_race_overlapping_windows() {
        use choreo_automata::{Duration, TemporalGuardExpr, TimeInterval, TimePoint, TimerId};
        let te1 = TemporalGuardExpr::WithinInterval(TimeInterval::new(
            TimePoint(0.0),
            TimePoint(5.0),
        ));
        let te2 = TemporalGuardExpr::WithinInterval(TimeInterval::new(
            TimePoint(3.0),
            TimePoint(8.0),
        ));
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Temporal(te1), 0),
                (0, 2, Guard::Temporal(te2), 0),
            ],
            0,
        );
        let temporal_races = detect_temporal_races(&aut);
        assert!(!temporal_races.is_empty());
    }

    #[test]
    fn temporal_race_no_overlap() {
        use choreo_automata::{TemporalGuardExpr, TimeInterval, TimePoint};
        let te1 = TemporalGuardExpr::WithinInterval(TimeInterval::new(
            TimePoint(0.0),
            TimePoint(2.0),
        ));
        let te2 = TemporalGuardExpr::WithinInterval(TimeInterval::new(
            TimePoint(5.0),
            TimePoint(8.0),
        ));
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Temporal(te1), 0),
                (0, 2, Guard::Temporal(te2), 0),
            ],
            0,
        );
        let temporal_races = detect_temporal_races(&aut);
        assert!(temporal_races.is_empty());
    }
}
