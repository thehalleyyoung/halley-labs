//! Advanced candidate generation strategies for GCHDD.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::subtree::{SubtreeCandidate, SubtreeOp, CandidateHistory};

/// Strategy for selecting candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateStrategy {
    Greedy,
    BreadthFirst,
    DepthFirst,
    SizeGuided,
    RandomSampled,
}

/// A candidate with a computed priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub candidate: SubtreeCandidate,
    pub priority: f64,
    pub estimated_validity_prob: f64,
    pub depth: usize,
}

impl RankedCandidate {
    pub fn new(candidate: SubtreeCandidate, priority: f64, depth: usize) -> Self {
        Self {
            estimated_validity_prob: candidate.estimated_validity,
            candidate,
            priority,
            depth,
        }
    }
}

/// Pool of ranked candidates with a selection strategy.
#[derive(Debug, Clone)]
pub struct CandidatePool {
    pub candidates: Vec<RankedCandidate>,
    pub strategy: CandidateStrategy,
}

impl CandidatePool {
    pub fn new(strategy: CandidateStrategy) -> Self {
        Self {
            candidates: Vec::new(),
            strategy,
        }
    }

    pub fn add(&mut self, candidate: RankedCandidate) {
        self.candidates.push(candidate);
    }

    pub fn add_all(&mut self, candidates: Vec<RankedCandidate>) {
        self.candidates.extend(candidates);
    }

    /// Sort candidates according to the strategy.
    pub fn sort(&mut self) {
        match self.strategy {
            CandidateStrategy::Greedy => {
                self.candidates.sort_by(|a, b| {
                    b.candidate.size_reduction.cmp(&a.candidate.size_reduction)
                });
            }
            CandidateStrategy::BreadthFirst => {
                self.candidates.sort_by(|a, b| a.depth.cmp(&b.depth));
            }
            CandidateStrategy::DepthFirst => {
                self.candidates.sort_by(|a, b| b.depth.cmp(&a.depth));
            }
            CandidateStrategy::SizeGuided => {
                self.candidates.sort_by(|a, b| {
                    b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            CandidateStrategy::RandomSampled => {
                // Shuffle using a simple deterministic approach
                let n = self.candidates.len();
                for i in 0..n {
                    let j = (i * 7 + 3) % n;
                    self.candidates.swap(i, j);
                }
            }
        }
    }

    pub fn pop_best(&mut self) -> Option<RankedCandidate> {
        self.sort();
        if self.candidates.is_empty() {
            None
        } else {
            Some(self.candidates.remove(0))
        }
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn clear(&mut self) {
        self.candidates.clear();
    }
}

/// Computes priority scores for candidates.
pub struct PriorityComputer {
    pub size_weight: f64,
    pub validity_weight: f64,
    pub depth_weight: f64,
    pub history_weight: f64,
}

impl Default for PriorityComputer {
    fn default() -> Self {
        Self {
            size_weight: 0.4,
            validity_weight: 0.3,
            depth_weight: 0.1,
            history_weight: 0.2,
        }
    }
}

impl PriorityComputer {
    pub fn compute(
        &self,
        candidate: &SubtreeCandidate,
        depth: usize,
        max_depth: usize,
        history: &CandidateHistory,
    ) -> f64 {
        let size_score = candidate.size_reduction as f64;
        let validity_score = candidate.estimated_validity;
        let depth_score = if max_depth > 0 {
            1.0 - (depth as f64 / max_depth as f64)
        } else {
            0.5
        };
        let history_score = history.success_rate_for_op(&candidate.operation);

        self.size_weight * size_score
            + self.validity_weight * validity_score
            + self.depth_weight * depth_score
            + self.history_weight * history_score
    }
}

/// Selects candidates greedily (largest reduction first).
pub struct GreedyCandidateSelector;

impl GreedyCandidateSelector {
    pub fn select(candidates: &[SubtreeCandidate]) -> Option<&SubtreeCandidate> {
        candidates.iter().max_by_key(|c| c.size_reduction)
    }
}

/// Selects candidates breadth-first (shallowest first).
pub struct BreadthFirstSelector;

impl BreadthFirstSelector {
    pub fn select_order(candidates: &mut [RankedCandidate]) {
        candidates.sort_by_key(|c| c.depth);
    }
}

/// Selects candidates depth-first (deepest first).
pub struct DepthFirstSelector;

impl DepthFirstSelector {
    pub fn select_order(candidates: &mut [RankedCandidate]) {
        candidates.sort_by(|a, b| b.depth.cmp(&a.depth));
    }
}

/// Selects by estimated size after reduction.
pub struct SizeGuidedSelector;

impl SizeGuidedSelector {
    pub fn select_order(candidates: &mut [RankedCandidate]) {
        candidates.sort_by(|a, b| {
            b.candidate.size_reduction.cmp(&a.candidate.size_reduction)
        });
    }
}

/// Random sampling with replacement probabilities proportional to size reduction.
pub struct RandomSampledSelector {
    pub seed: u64,
}

impl RandomSampledSelector {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn select<'a>(&self, candidates: &'a [SubtreeCandidate]) -> Option<&'a SubtreeCandidate> {
        if candidates.is_empty() {
            return None;
        }
        let total_weight: f64 = candidates.iter().map(|c| c.size_reduction as f64 + 1.0).sum();
        let mut target = (self.seed as f64 * 0.618033988749) % total_weight;
        for c in candidates {
            target -= c.size_reduction as f64 + 1.0;
            if target <= 0.0 {
                return Some(c);
            }
        }
        candidates.last()
    }
}

/// Tracks which types of reductions tend to succeed.
#[derive(Debug, Clone, Default)]
pub struct HistoricalSuccessTracker {
    pub op_stats: HashMap<String, (usize, usize)>,
    pub label_stats: HashMap<String, (usize, usize)>,
}

impl HistoricalSuccessTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_op(&mut self, op: &SubtreeOp, success: bool) {
        let key = format!("{}", op);
        let entry = self.op_stats.entry(key).or_insert((0, 0));
        entry.0 += 1;
        if success { entry.1 += 1; }
    }

    pub fn record_label(&mut self, label: &str, success: bool) {
        let entry = self.label_stats.entry(label.to_string()).or_insert((0, 0));
        entry.0 += 1;
        if success { entry.1 += 1; }
    }

    pub fn op_success_rate(&self, op: &SubtreeOp) -> f64 {
        let key = format!("{}", op);
        self.op_stats.get(&key).map(|(a, s)| {
            if *a == 0 { 0.5 } else { *s as f64 / *a as f64 }
        }).unwrap_or(0.5)
    }

    pub fn label_success_rate(&self, label: &str) -> f64 {
        self.label_stats.get(label).map(|(a, s)| {
            if *a == 0 { 0.5 } else { *s as f64 / *a as f64 }
        }).unwrap_or(0.5)
    }

    pub fn total_attempts(&self) -> usize {
        self.op_stats.values().map(|(a, _)| a).sum()
    }
}

/// Cache validity results to avoid redundant checks.
#[derive(Debug, Clone, Default)]
pub struct CandidateCache {
    pub results: HashMap<String, bool>,
}

impl CandidateCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<bool> {
        self.results.get(key).copied()
    }

    pub fn insert(&mut self, key: String, valid: bool) {
        self.results.insert(key, valid);
    }

    pub fn hit_rate(&self) -> f64 {
        // Approximate: we can't track hits vs misses without extra state
        0.0
    }

    pub fn size(&self) -> usize {
        self.results.len()
    }

    pub fn clear(&mut self) {
        self.results.clear();
    }
}

/// Statistics about candidate generation and selection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandidateStatistics {
    pub total_generated: usize,
    pub total_accepted: usize,
    pub total_rejected: usize,
    pub rejected_validity: usize,
    pub rejected_applicability: usize,
    pub rejected_violation: usize,
    pub cache_hits: usize,
}

impl CandidateStatistics {
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_generated == 0 { return 0.0; }
        self.total_accepted as f64 / self.total_generated as f64
    }

    pub fn rejection_breakdown(&self) -> (f64, f64, f64) {
        let total = self.total_rejected.max(1) as f64;
        (
            self.rejected_validity as f64 / total,
            self.rejected_applicability as f64 / total,
            self.rejected_violation as f64 / total,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(target: usize, reduction: usize) -> SubtreeCandidate {
        SubtreeCandidate::new(target, SubtreeOp::Delete, reduction)
    }

    #[test]
    fn test_candidate_pool_greedy() {
        let mut pool = CandidatePool::new(CandidateStrategy::Greedy);
        pool.add(RankedCandidate::new(make_candidate(0, 5), 5.0, 1));
        pool.add(RankedCandidate::new(make_candidate(1, 10), 10.0, 2));
        pool.add(RankedCandidate::new(make_candidate(2, 3), 3.0, 0));
        let best = pool.pop_best().unwrap();
        assert_eq!(best.candidate.size_reduction, 10);
    }

    #[test]
    fn test_candidate_pool_depth_first() {
        let mut pool = CandidatePool::new(CandidateStrategy::DepthFirst);
        pool.add(RankedCandidate::new(make_candidate(0, 5), 5.0, 1));
        pool.add(RankedCandidate::new(make_candidate(1, 3), 3.0, 5));
        let best = pool.pop_best().unwrap();
        assert_eq!(best.depth, 5);
    }

    #[test]
    fn test_priority_computer() {
        let pc = PriorityComputer::default();
        let c = make_candidate(0, 10);
        let history = CandidateHistory::new();
        let priority = pc.compute(&c, 2, 10, &history);
        assert!(priority > 0.0);
    }

    #[test]
    fn test_greedy_selector() {
        let candidates = vec![make_candidate(0, 3), make_candidate(1, 7), make_candidate(2, 5)];
        let best = GreedyCandidateSelector::select(&candidates).unwrap();
        assert_eq!(best.size_reduction, 7);
    }

    #[test]
    fn test_historical_tracker() {
        let mut tracker = HistoricalSuccessTracker::new();
        tracker.record_op(&SubtreeOp::Delete, true);
        tracker.record_op(&SubtreeOp::Delete, true);
        tracker.record_op(&SubtreeOp::Delete, false);
        let rate = tracker.op_success_rate(&SubtreeOp::Delete);
        assert!((rate - 2.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_candidate_cache() {
        let mut cache = CandidateCache::new();
        cache.insert("test1".into(), true);
        cache.insert("test2".into(), false);
        assert_eq!(cache.get("test1"), Some(true));
        assert_eq!(cache.get("test2"), Some(false));
        assert_eq!(cache.get("test3"), None);
        assert_eq!(cache.size(), 2);
    }

    #[test]
    fn test_candidate_statistics() {
        let stats = CandidateStatistics {
            total_generated: 100,
            total_accepted: 30,
            total_rejected: 70,
            rejected_validity: 40,
            rejected_applicability: 20,
            rejected_violation: 10,
            cache_hits: 5,
        };
        assert!((stats.acceptance_rate() - 0.3).abs() < 0.01);
        let (v, a, vi) = stats.rejection_breakdown();
        assert!(v > a);
    }

    #[test]
    fn test_random_sampled_selector() {
        let selector = RandomSampledSelector::new(42);
        let candidates = vec![make_candidate(0, 10), make_candidate(1, 1)];
        let selected = selector.select(&candidates);
        assert!(selected.is_some());
    }

    #[test]
    fn test_empty_pool() {
        let mut pool = CandidatePool::new(CandidateStrategy::Greedy);
        assert!(pool.is_empty());
        assert!(pool.pop_best().is_none());
    }
}
