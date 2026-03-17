//! Event scheduling with a priority queue.
//!
//! Provides a `BinaryHeap`-based scheduler for ordering events by timestamp
//! and priority, with support for batch processing, event coalescing, and
//! statistics tracking.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// ScheduledEvent
// ---------------------------------------------------------------------------

/// An event queued for future delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    /// The event kind (opaque string identifier).
    pub kind: String,
    /// When the event should be delivered (simulation time).
    pub timestamp: f64,
    /// Higher-priority events are delivered first at the same timestamp.
    pub priority: i32,
    /// Sequence number for stable ordering.
    sequence: u64,
    /// Optional group key for coalescing.
    pub group: Option<String>,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so we reverse for earliest-first
        other
            .timestamp
            .partial_cmp(&self.timestamp)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.priority.cmp(&self.priority))
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

// ---------------------------------------------------------------------------
// SchedulerStats
// ---------------------------------------------------------------------------

/// Statistics about the scheduler's operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub total_scheduled: u64,
    pub total_delivered: u64,
    pub total_cancelled: u64,
    pub total_coalesced: u64,
    pub peak_queue_size: usize,
}

// ---------------------------------------------------------------------------
// EventScheduler
// ---------------------------------------------------------------------------

/// Priority-queue based event scheduler.
#[derive(Debug)]
pub struct EventScheduler {
    queue: BinaryHeap<ScheduledEvent>,
    next_sequence: u64,
    /// Whether event coalescing is enabled.
    coalescing_enabled: bool,
    /// Coalescing window: events within this time of each other in the same
    /// group are merged.
    coalescing_window: f64,
    /// Stats
    stats: SchedulerStats,
}

impl EventScheduler {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            next_sequence: 0,
            coalescing_enabled: false,
            coalescing_window: 0.0,
            stats: SchedulerStats::default(),
        }
    }

    /// Enable event coalescing with the given time window.
    pub fn with_coalescing(mut self, window: f64) -> Self {
        self.coalescing_enabled = true;
        self.coalescing_window = window;
        self
    }

    /// Schedule an event for delivery at the given timestamp.
    pub fn schedule_event(
        &mut self,
        kind: impl Into<String>,
        timestamp: f64,
        priority: i32,
    ) {
        self.schedule_event_with_group(kind, timestamp, priority, None);
    }

    /// Schedule an event with an optional coalescing group.
    pub fn schedule_event_with_group(
        &mut self,
        kind: impl Into<String>,
        timestamp: f64,
        priority: i32,
        group: Option<String>,
    ) {
        let kind = kind.into();

        // Check for coalescing
        if self.coalescing_enabled {
            if let Some(ref grp) = group {
                if self.try_coalesce(grp, timestamp) {
                    self.stats.total_coalesced += 1;
                    return;
                }
            }
        }

        let seq = self.next_sequence;
        self.next_sequence += 1;

        self.queue.push(ScheduledEvent {
            kind,
            timestamp,
            priority,
            sequence: seq,
            group,
        });

        self.stats.total_scheduled += 1;
        if self.queue.len() > self.stats.peak_queue_size {
            self.stats.peak_queue_size = self.queue.len();
        }
    }

    /// Get the next event without removing it.
    pub fn peek(&self) -> Option<&ScheduledEvent> {
        self.queue.peek()
    }

    /// Remove and return the next (earliest/highest-priority) event.
    pub fn next_event(&mut self) -> Option<ScheduledEvent> {
        let event = self.queue.pop();
        if event.is_some() {
            self.stats.total_delivered += 1;
        }
        event
    }

    /// Remove and return all events up to (and including) the given timestamp.
    pub fn drain_until(&mut self, timestamp: f64) -> Vec<ScheduledEvent> {
        let mut events = Vec::new();
        while let Some(evt) = self.queue.peek() {
            if evt.timestamp <= timestamp {
                let e = self.queue.pop().unwrap();
                self.stats.total_delivered += 1;
                events.push(e);
            } else {
                break;
            }
        }
        events
    }

    /// Process a batch of events up to the given timestamp, invoking `f`
    /// for each one.
    pub fn process_batch<F>(&mut self, timestamp: f64, mut f: F)
    where
        F: FnMut(&ScheduledEvent),
    {
        let events = self.drain_until(timestamp);
        for evt in &events {
            f(evt);
        }
    }

    /// Number of events in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Is the queue empty?
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clear all scheduled events.
    pub fn clear(&mut self) {
        let cancelled = self.queue.len();
        self.queue.clear();
        self.stats.total_cancelled += cancelled as u64;
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Get the timestamp of the next event, if any.
    pub fn next_event_time(&self) -> Option<f64> {
        self.queue.peek().map(|e| e.timestamp)
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Try to coalesce an event into an existing event in the same group.
    fn try_coalesce(&self, group: &str, timestamp: f64) -> bool {
        // Linear scan (acceptable for small queues).
        // A production implementation would use a HashMap<group, latest_timestamp>.
        for evt in self.queue.iter() {
            if let Some(ref g) = evt.group {
                if g == group && (evt.timestamp - timestamp).abs() <= self.coalescing_window {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for EventScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_scheduling() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("evt_a", 2.0, 0);
        sched.schedule_event("evt_b", 1.0, 0);
        sched.schedule_event("evt_c", 3.0, 0);

        let e1 = sched.next_event().unwrap();
        assert_eq!(e1.kind, "evt_b");
        assert_eq!(e1.timestamp, 1.0);

        let e2 = sched.next_event().unwrap();
        assert_eq!(e2.kind, "evt_a");

        let e3 = sched.next_event().unwrap();
        assert_eq!(e3.kind, "evt_c");
    }

    #[test]
    fn priority_ordering() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("low", 1.0, 0);
        sched.schedule_event("high", 1.0, 10);

        let e = sched.next_event().unwrap();
        assert_eq!(e.kind, "high");
        assert_eq!(e.priority, 10);
    }

    #[test]
    fn stable_ordering() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("first", 1.0, 0);
        sched.schedule_event("second", 1.0, 0);

        let e = sched.next_event().unwrap();
        assert_eq!(e.kind, "first");
    }

    #[test]
    fn drain_until() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("a", 1.0, 0);
        sched.schedule_event("b", 2.0, 0);
        sched.schedule_event("c", 3.0, 0);

        let events = sched.drain_until(2.5);
        assert_eq!(events.len(), 2);
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn process_batch() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("a", 1.0, 0);
        sched.schedule_event("b", 2.0, 0);

        let mut seen = Vec::new();
        sched.process_batch(3.0, |evt| {
            seen.push(evt.kind.clone());
        });
        assert_eq!(seen, vec!["a", "b"]);
        assert!(sched.is_empty());
    }

    #[test]
    fn coalescing() {
        let mut sched = EventScheduler::new().with_coalescing(0.1);
        sched.schedule_event_with_group("rapid", 1.0, 0, Some("group1".into()));
        sched.schedule_event_with_group("rapid", 1.05, 0, Some("group1".into()));
        // Second should be coalesced
        assert_eq!(sched.len(), 1);
        assert_eq!(sched.stats().total_coalesced, 1);
    }

    #[test]
    fn coalescing_different_groups() {
        let mut sched = EventScheduler::new().with_coalescing(0.1);
        sched.schedule_event_with_group("a", 1.0, 0, Some("g1".into()));
        sched.schedule_event_with_group("b", 1.05, 0, Some("g2".into()));
        assert_eq!(sched.len(), 2);
    }

    #[test]
    fn clear_cancels_all() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("a", 1.0, 0);
        sched.schedule_event("b", 2.0, 0);
        sched.clear();
        assert!(sched.is_empty());
        assert_eq!(sched.stats().total_cancelled, 2);
    }

    #[test]
    fn stats_tracking() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("a", 1.0, 0);
        sched.schedule_event("b", 2.0, 0);
        let _ = sched.next_event();
        assert_eq!(sched.stats().total_scheduled, 2);
        assert_eq!(sched.stats().total_delivered, 1);
        assert_eq!(sched.stats().peak_queue_size, 2);
    }

    #[test]
    fn next_event_time() {
        let mut sched = EventScheduler::new();
        assert!(sched.next_event_time().is_none());
        sched.schedule_event("a", 5.0, 0);
        assert_eq!(sched.next_event_time(), Some(5.0));
    }

    #[test]
    fn peek_does_not_remove() {
        let mut sched = EventScheduler::new();
        sched.schedule_event("a", 1.0, 0);
        assert!(sched.peek().is_some());
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn empty_drain() {
        let mut sched = EventScheduler::new();
        let events = sched.drain_until(100.0);
        assert!(events.is_empty());
    }
}
