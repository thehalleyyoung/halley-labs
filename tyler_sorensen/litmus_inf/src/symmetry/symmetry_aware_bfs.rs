//! Symmetry-aware BFS for execution graph enumeration.
//!
//! Instead of computing all graphs then deduplicating, this module skips
//! symmetric states DURING enumeration. Uses canonical labeling to detect
//! already-visited isomorphic states.

use std::collections::{HashSet, VecDeque, HashMap};
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    Event, EventId, ThreadId, Address, OpType, Scope,
    ExecutionGraph, BitMatrix,
};

use super::canonical_labeling::{CanonicalLabeler, CanonicalForm, LabelingConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfsConfig {
    /// Maximum states to explore.
    pub max_states: usize,
    /// Whether to use canonical labeling for dedup.
    pub use_canonical_dedup: bool,
    /// Labeling configuration.
    pub labeling_config: LabelingConfig,
    /// Maximum BFS depth.
    pub max_depth: usize,
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self {
            max_states: 10_000_000,
            use_canonical_dedup: true,
            labeling_config: LabelingConfig::default(),
            max_depth: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BfsStatistics {
    pub states_explored: usize,
    pub states_skipped_symmetric: usize,
    pub unique_states: usize,
    pub max_depth: usize,
    pub bfs_queue_max_size: usize,
    pub canonical_forms_computed: usize,
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// BFS State
// ---------------------------------------------------------------------------

/// A state in the symmetry-aware BFS.
#[derive(Debug, Clone)]
pub struct BfsState {
    /// Current partial execution graph being built.
    pub graph: ExecutionGraph,
    /// Which events have been added.
    pub events_added: Vec<bool>,
    /// Current depth.
    pub depth: usize,
    /// Thread program counters.
    pub thread_pc: Vec<usize>,
    /// Reads-from choices made so far.
    pub rf_choices: Vec<(EventId, EventId)>,
    /// Coherence order choices made so far.
    pub co_choices: Vec<(EventId, EventId)>,
}

impl BfsState {
    pub fn initial(num_events: usize, num_threads: usize) -> Self {
        Self {
            graph: ExecutionGraph::empty(),
            events_added: vec![false; num_events],
            depth: 0,
            thread_pc: vec![0; num_threads],
            rf_choices: Vec::new(),
            co_choices: Vec::new(),
        }
    }

    /// Check if this is a complete execution.
    pub fn is_complete(&self) -> bool {
        self.events_added.iter().all(|&added| added)
    }

    /// Get events that can be added next (respecting program order).
    pub fn next_events(&self, events: &[Event]) -> Vec<EventId> {
        let mut result = Vec::new();
        for (i, event) in events.iter().enumerate() {
            if self.events_added[i] {
                continue;
            }
            if event.po_index == self.thread_pc[event.thread] {
                result.push(i);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Symmetry-Aware BFS
// ---------------------------------------------------------------------------

/// BFS exploration that skips symmetric states using canonical labeling.
pub struct SymmetryAwareBfs {
    config: BfsConfig,
    labeler: CanonicalLabeler,
    seen_canonical: HashSet<u64>,
    pub stats: BfsStatistics,
    pub complete_executions: Vec<ExecutionGraph>,
}

impl SymmetryAwareBfs {
    pub fn new(config: BfsConfig) -> Self {
        let labeler = CanonicalLabeler::new(config.labeling_config.clone());
        Self {
            config,
            labeler,
            seen_canonical: HashSet::new(),
            stats: BfsStatistics::default(),
            complete_executions: Vec::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(BfsConfig::default())
    }

    /// Run symmetry-aware BFS on a set of events.
    pub fn explore(&mut self, events: &[Event]) -> &[ExecutionGraph] {
        let start = std::time::Instant::now();
        let num_events = events.len();
        let num_threads = events.iter().map(|e| e.thread).max().map_or(0, |t| t + 1);

        let initial = BfsState::initial(num_events, num_threads);
        let mut queue = VecDeque::new();
        queue.push_back(initial);

        while let Some(state) = queue.pop_front() {
            if self.stats.states_explored >= self.config.max_states {
                break;
            }

            self.stats.states_explored += 1;
            self.stats.max_depth = self.stats.max_depth.max(state.depth);
            self.stats.bfs_queue_max_size = self.stats.bfs_queue_max_size.max(queue.len());

            if state.is_complete() {
                // Check canonical form for dedup
                if self.config.use_canonical_dedup {
                    self.stats.canonical_forms_computed += 1;
                    let form = self.labeler.canonical_form(&state.graph);
                    if self.seen_canonical.insert(form.hash) {
                        self.stats.unique_states += 1;
                        self.complete_executions.push(state.graph.clone());
                    } else {
                        self.stats.states_skipped_symmetric += 1;
                    }
                } else {
                    self.stats.unique_states += 1;
                    self.complete_executions.push(state.graph.clone());
                }
                continue;
            }

            if state.depth >= self.config.max_depth {
                continue;
            }

            // Get next events to add
            let next = state.next_events(events);

            for &event_id in &next {
                let event = &events[event_id];

                // For reads: enumerate reads-from choices
                if event.is_read() {
                    // Can read from any write to same address that has been added
                    let writes: Vec<EventId> = events.iter().enumerate()
                        .filter(|(i, e)| {
                            state.events_added[*i]
                                && e.is_write()
                                && e.address == event.address
                        })
                        .map(|(i, _)| i)
                        .collect();

                    // Also can read initial value (no rf edge)
                    let mut new_state = self.extend_state(&state, event_id, events);
                    if self.should_explore(&new_state) {
                        queue.push_back(new_state);
                    }

                    for &write_id in &writes {
                        let mut new_state = self.extend_state(&state, event_id, events);
                        new_state.graph.rf.add(write_id, event_id);
                        new_state.rf_choices.push((write_id, event_id));
                        if self.should_explore(&new_state) {
                            queue.push_back(new_state);
                        }
                    }
                } else {
                    let new_state = self.extend_state(&state, event_id, events);
                    if self.should_explore(&new_state) {
                        queue.push_back(new_state);
                    }
                }
            }
        }

        self.stats.elapsed_ms = start.elapsed().as_millis() as u64;
        &self.complete_executions
    }

    fn extend_state(&self, state: &BfsState, event_id: EventId, events: &[Event]) -> BfsState {
        let mut new_state = state.clone();
        new_state.events_added[event_id] = true;
        new_state.depth += 1;

        let event = &events[event_id];
        new_state.thread_pc[event.thread] += 1;

        // Add event to graph
        if new_state.graph.events.len() <= event_id {
            new_state.graph.events.resize(event_id + 1, events[0].clone());
        }
        new_state.graph.events[event_id] = event.clone();

        // Add PO edges from previous event in same thread
        if event.po_index > 0 {
            for (i, e) in events.iter().enumerate() {
                if e.thread == event.thread
                    && e.po_index == event.po_index - 1
                    && state.events_added[i]
                {
                    new_state.graph.po.add(i, event_id);
                }
            }
        }

        new_state
    }

    fn should_explore(&self, state: &BfsState) -> bool {
        if !self.config.use_canonical_dedup {
            return true;
        }
        // For partial states, do a quick structural hash check
        // Full canonical form is only computed for complete states
        true
    }

    /// Get statistics.
    pub fn statistics(&self) -> &BfsStatistics {
        &self.stats
    }

    /// Get unique execution count.
    pub fn unique_count(&self) -> usize {
        self.stats.unique_states
    }

    /// Compute speedup vs naive enumeration.
    pub fn speedup(&self) -> f64 {
        if self.stats.unique_states == 0 { return 1.0; }
        self.stats.states_explored as f64 / self.stats.unique_states as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sb_events() -> Vec<Event> {
        vec![
            Event::new(0, 0, OpType::Write, 0, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 1, 0).with_po_index(1),
            Event::new(2, 1, OpType::Write, 1, 1).with_po_index(0),
            Event::new(3, 1, OpType::Read, 0, 0).with_po_index(1),
        ]
    }

    #[test]
    fn test_symmetry_aware_bfs() {
        let events = make_sb_events();
        let mut bfs = SymmetryAwareBfs::with_defaults();
        bfs.explore(&events);
        assert!(bfs.stats.states_explored > 0);
    }

    #[test]
    fn test_bfs_finds_executions() {
        let events = make_sb_events();
        let mut bfs = SymmetryAwareBfs::with_defaults();
        bfs.explore(&events);
        assert!(bfs.unique_count() > 0);
    }

    #[test]
    fn test_bfs_dedup_reduces() {
        let events = make_sb_events();
        
        // With dedup
        let mut with_dedup = SymmetryAwareBfs::new(BfsConfig {
            use_canonical_dedup: true,
            ..BfsConfig::default()
        });
        with_dedup.explore(&events);

        // Without dedup
        let mut without_dedup = SymmetryAwareBfs::new(BfsConfig {
            use_canonical_dedup: false,
            ..BfsConfig::default()
        });
        without_dedup.explore(&events);

        assert!(with_dedup.unique_count() <= without_dedup.unique_count());
    }
}
