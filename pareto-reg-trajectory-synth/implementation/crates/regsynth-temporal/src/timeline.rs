use crate::{ObligationId, RegulatoryEvent};
use chrono::{Duration, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A single event on the timeline with its affected obligations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub timestamp: NaiveDate,
    pub event: RegulatoryEvent,
    pub affected_obligations: BTreeSet<ObligationId>,
}

/// A timeline of regulatory events, maintaining sorted order and state computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    events: Vec<TimelineEvent>,
    states: BTreeMap<NaiveDate, BTreeSet<ObligationId>>,
}

impl Timeline {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            states: BTreeMap::new(),
        }
    }

    /// Add an event to the timeline. Events are kept sorted by timestamp.
    pub fn add_event(&mut self, event: TimelineEvent) {
        self.events.push(event);
        self.events.sort_by_key(|e| e.timestamp);
        self.recompute_states();
    }

    /// Recompute the state map from scratch based on sorted events.
    fn recompute_states(&mut self) {
        self.states.clear();
        let mut current = BTreeSet::new();
        for evt in &self.events {
            match &evt.event {
                RegulatoryEvent::Amendment { .. }
                | RegulatoryEvent::PhaseIn { .. }
                | RegulatoryEvent::GracePeriodEnd { .. } => {
                    for obl in &evt.affected_obligations {
                        current.insert(obl.clone());
                    }
                }
                RegulatoryEvent::Sunset { .. }
                | RegulatoryEvent::Repeal { .. } => {
                    for obl in &evt.affected_obligations {
                        current.remove(obl);
                    }
                }
                RegulatoryEvent::GracePeriodStart { .. } => {
                    // Grace period start: obligations remain active
                }
            }
            self.states.insert(evt.timestamp, current.clone());
        }
    }

    /// Compute the set of active obligations at a given date by looking up
    /// the most recent state snapshot at or before that date.
    pub fn state_at(&self, date: &NaiveDate) -> BTreeSet<ObligationId> {
        self.states
            .range(..=*date)
            .next_back()
            .map(|(_, obls)| obls.clone())
            .unwrap_or_default()
    }

    /// Returns events in the range [start, end] inclusive.
    pub fn transitions_between(&self, start: &NaiveDate, end: &NaiveDate) -> Vec<&TimelineEvent> {
        self.events.iter()
            .filter(|e| e.timestamp >= *start && e.timestamp <= *end)
            .collect()
    }

    /// Returns all unique dates where regulatory changes happen.
    pub fn critical_points(&self) -> Vec<NaiveDate> {
        self.events.iter().map(|e| e.timestamp).collect::<BTreeSet<_>>().into_iter().collect()
    }

    /// Sample the timeline at regular intervals, returning (date, active obligations) pairs.
    pub fn discretize(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        step_days: i64,
    ) -> Vec<(NaiveDate, BTreeSet<ObligationId>)> {
        let mut result = Vec::new();
        let mut current = start;
        while current <= end {
            let obligations = self.state_at(&current);
            result.push((current, obligations));
            current = current + Duration::days(step_days);
        }
        result
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ymd;

    fn make_phase_in(date: NaiveDate, milestone: &str, obligations: &[&str]) -> TimelineEvent {
        TimelineEvent {
            timestamp: date,
            event: RegulatoryEvent::PhaseIn {
                milestone: milestone.to_string(),
                date,
            },
            affected_obligations: obligations.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn make_sunset(date: NaiveDate, desc: &str, obligations: &[&str]) -> TimelineEvent {
        TimelineEvent {
            timestamp: date,
            event: RegulatoryEvent::Sunset {
                description: desc.to_string(),
                date,
            },
            affected_obligations: obligations.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_empty_timeline() {
        let tl = Timeline::new();
        assert!(tl.is_empty());
        assert_eq!(tl.len(), 0);
        assert!(tl.state_at(&ymd(2025, 1, 1)).is_empty());
    }

    #[test]
    fn test_add_events_sorted() {
        let mut tl = Timeline::new();
        tl.add_event(make_phase_in(ymd(2026, 1, 1), "M2", &["b"]));
        tl.add_event(make_phase_in(ymd(2025, 1, 1), "M1", &["a"]));
        assert_eq!(tl.len(), 2);
        let points = tl.critical_points();
        assert_eq!(points[0], ymd(2025, 1, 1));
        assert_eq!(points[1], ymd(2026, 1, 1));
    }

    #[test]
    fn test_state_at() {
        let mut tl = Timeline::new();
        tl.add_event(make_phase_in(ymd(2025, 1, 1), "M1", &["a", "b"]));
        tl.add_event(make_phase_in(ymd(2026, 1, 1), "M2", &["c"]));

        let before = tl.state_at(&ymd(2024, 12, 31));
        assert!(before.is_empty());

        let at_m1 = tl.state_at(&ymd(2025, 6, 1));
        assert!(at_m1.contains("a"));
        assert!(at_m1.contains("b"));
        assert!(!at_m1.contains("c"));

        let at_m2 = tl.state_at(&ymd(2026, 6, 1));
        assert!(at_m2.contains("a"));
        assert!(at_m2.contains("b"));
        assert!(at_m2.contains("c"));
    }

    #[test]
    fn test_sunset_removes_obligations() {
        let mut tl = Timeline::new();
        tl.add_event(make_phase_in(ymd(2025, 1, 1), "M1", &["a", "b"]));
        tl.add_event(make_sunset(ymd(2026, 1, 1), "sunset-a", &["a"]));

        let after_sunset = tl.state_at(&ymd(2026, 6, 1));
        assert!(!after_sunset.contains("a"));
        assert!(after_sunset.contains("b"));
    }

    #[test]
    fn test_transitions_between() {
        let mut tl = Timeline::new();
        tl.add_event(make_phase_in(ymd(2025, 1, 1), "M1", &["a"]));
        tl.add_event(make_phase_in(ymd(2025, 6, 1), "M2", &["b"]));
        tl.add_event(make_phase_in(ymd(2026, 1, 1), "M3", &["c"]));

        let range = tl.transitions_between(&ymd(2025, 1, 1), &ymd(2025, 12, 31));
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn test_critical_points() {
        let mut tl = Timeline::new();
        tl.add_event(make_phase_in(ymd(2025, 1, 1), "M1", &["a"]));
        tl.add_event(make_phase_in(ymd(2025, 1, 1), "M1b", &["b"]));
        tl.add_event(make_phase_in(ymd(2026, 1, 1), "M2", &["c"]));
        let cp = tl.critical_points();
        assert_eq!(cp.len(), 2);
    }

    #[test]
    fn test_discretize() {
        let mut tl = Timeline::new();
        tl.add_event(make_phase_in(ymd(2025, 1, 1), "M1", &["a"]));
        tl.add_event(make_phase_in(ymd(2025, 7, 1), "M2", &["b"]));

        let samples = tl.discretize(ymd(2025, 1, 1), ymd(2025, 12, 31), 90);
        assert!(!samples.is_empty());
        assert!(samples[0].1.contains("a"));
        assert!(!samples[0].1.contains("b"));
        let late = samples.iter().find(|(d, _)| *d >= ymd(2025, 7, 1));
        if let Some((_, obls)) = late {
            assert!(obls.contains("a"));
            assert!(obls.contains("b"));
        }
    }
}
