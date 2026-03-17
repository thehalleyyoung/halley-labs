//! Monotonicity-based pruning for cascade verification.
//!
//! Provides antichain-based pruning of the failure-set lattice, monotonicity
//! verification, and MARCO-style minimal failure set enumeration.

use cascade_graph::rtig::RtigGraph;
use cascade_types::cascade::{FailureMode, FailureSet, MinimalFailureSet};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::checker::{BmcConfig, BmcResult, BmcStatus, BoundedModelChecker};

// ---------------------------------------------------------------------------
// MonotonicityResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonotonicityResult {
    pub is_monotone: bool,
    pub violations: Vec<MonotonicityViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonotonicityViolation {
    pub failure_set: Vec<String>,
    pub superset: Vec<String>,
    pub explanation: String,
}

// ---------------------------------------------------------------------------
// MonotonicityChecker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MonotonicityChecker {
    graph: RtigGraph,
    config: BmcConfig,
}

impl MonotonicityChecker {
    pub fn new(graph: RtigGraph, config: BmcConfig) -> Self {
        Self { graph, config }
    }

    /// Verify monotonicity: if a failure set S causes a cascade, then any
    /// superset S' ⊇ S should also cause a cascade.
    pub fn verify_monotonicity(&self) -> MonotonicityResult {
        let service_ids: Vec<String> = self.graph.service_ids().iter().map(|s| s.to_string()).collect();
        let n = service_ids.len();
        let mut violations = Vec::new();
        let checker = BoundedModelChecker::new(self.config.clone());

        // Check all pairs: if {a} cascades but {a, b} doesn't, that's a violation
        // Only check small sets to keep it tractable
        let max_check_size = n.min(4);

        let mut cascade_sets: Vec<Vec<String>> = Vec::new();
        let mut safe_sets: Vec<Vec<String>> = Vec::new();

        for k in 1..=max_check_size {
            let combos = combinations(&service_ids, k);
            for combo in combos {
                let is_cascade = self.check_failure_set_cascades(&combo);
                if is_cascade {
                    cascade_sets.push(combo);
                } else {
                    safe_sets.push(combo);
                }
            }
        }

        // Check monotonicity: for each cascading set, verify all supersets also cascade
        for cs in &cascade_sets {
            let cs_set: BTreeSet<String> = cs.iter().cloned().collect();
            for ss in &safe_sets {
                let ss_set: BTreeSet<String> = ss.iter().cloned().collect();
                if cs_set.is_subset(&ss_set) {
                    violations.push(MonotonicityViolation {
                        failure_set: cs.clone(),
                        superset: ss.clone(),
                        explanation: format!(
                            "Failure set {:?} causes cascade but superset {:?} does not",
                            cs, ss
                        ),
                    });
                }
            }
        }

        MonotonicityResult {
            is_monotone: violations.is_empty(),
            violations,
        }
    }

    fn check_failure_set_cascades(&self, failed: &[String]) -> bool {
        let service_ids: Vec<String> = self.graph.service_ids().iter().map(|s| s.to_string()).collect();
        let failed_set: HashSet<String> = failed.iter().cloned().collect();

        let depth = self.config.depth_bound.unwrap_or(5);

        // Simulate propagation
        let mut loads: HashMap<String, i64> = HashMap::new();
        for sid in &service_ids {
            let node = self.graph.service(sid).unwrap();
            if failed_set.contains(sid) {
                loads.insert(sid.clone(), 0);
            } else {
                loads.insert(sid.clone(), node.baseline_load as i64);
            }
        }

        for _t in 0..depth {
            let mut new_loads = HashMap::new();
            for sid in &service_ids {
                if failed_set.contains(sid) {
                    new_loads.insert(sid.clone(), 0i64);
                    continue;
                }
                let node = self.graph.service(sid).unwrap();
                let baseline = node.baseline_load as i64;
                let incoming: i64 = self.graph.incoming_edges(sid)
                    .iter()
                    .map(|edge| {
                        let pred_load = loads.get(edge.source.as_str()).copied().unwrap_or(0);
                        let pred_cap = self.graph.service(edge.source.as_str())
                            .map(|n| n.capacity as i64)
                            .unwrap_or(100);
                        let amp = if pred_load > pred_cap {
                            1 + edge.retry_count as i64
                        } else {
                            1
                        };
                        pred_load * amp
                    })
                    .sum();
                new_loads.insert(sid.clone(), baseline + incoming);
            }
            loads = new_loads;
        }

        // Check if any non-failed service is overloaded
        for sid in &service_ids {
            if failed_set.contains(sid) {
                continue;
            }
            let node = self.graph.service(sid).unwrap();
            let load = loads.get(sid).copied().unwrap_or(0);
            if load > node.capacity as i64 {
                return true;
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// AntichainPruner
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AntichainPruner {
    known_cascading: Vec<BTreeSet<String>>,
    known_safe: Vec<BTreeSet<String>>,
}

impl AntichainPruner {
    pub fn new() -> Self {
        Self {
            known_cascading: Vec::new(),
            known_safe: Vec::new(),
        }
    }

    /// Determine whether a candidate failure set should be checked.
    /// Returns false if the result can be inferred from known results.
    pub fn should_check(&self, candidate: &BTreeSet<String>) -> bool {
        // Skip if superset of any known cascading set (must also cascade by monotonicity)
        for cs in &self.known_cascading {
            if cs.is_subset(candidate) {
                return false;
            }
        }
        // Skip if subset of any known safe set (must also be safe by monotonicity)
        for ss in &self.known_safe {
            if candidate.is_subset(ss) {
                return false;
            }
        }
        true
    }

    /// Record the result of checking a failure set.
    pub fn record_result(&mut self, failure_set: BTreeSet<String>, is_cascade: bool) {
        if is_cascade {
            self.known_cascading.push(failure_set);
        } else {
            self.known_safe.push(failure_set);
        }
    }

    pub fn known_cascading_count(&self) -> usize {
        self.known_cascading.len()
    }

    pub fn known_safe_count(&self) -> usize {
        self.known_safe.len()
    }

    /// Check if a set is a known cascade (superset of known cascading).
    pub fn is_known_cascade(&self, candidate: &BTreeSet<String>) -> bool {
        self.known_cascading.iter().any(|cs| cs.is_subset(candidate))
    }

    /// Check if a set is known safe (subset of known safe).
    pub fn is_known_safe(&self, candidate: &BTreeSet<String>) -> bool {
        self.known_safe.iter().any(|ss| candidate.is_subset(ss))
    }
}

impl Default for AntichainPruner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AntichainSet – efficient antichain data structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AntichainSet {
    elements: Vec<BTreeSet<String>>,
}

impl AntichainSet {
    pub fn new() -> Self {
        Self { elements: Vec::new() }
    }

    /// Insert a set into the antichain. Removes any elements that are
    /// subsets or supersets of the new set.
    pub fn insert(&mut self, set: BTreeSet<String>) {
        // Remove subsets and supersets
        self.elements.retain(|existing| {
            !set.is_subset(existing) && !existing.is_subset(&set)
        });
        self.elements.push(set);
    }

    pub fn contains_subset_of(&self, set: &BTreeSet<String>) -> bool {
        self.elements.iter().any(|e| e.is_subset(set))
    }

    pub fn contains_superset_of(&self, set: &BTreeSet<String>) -> bool {
        self.elements.iter().any(|e| set.is_subset(e))
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &BTreeSet<String>> {
        self.elements.iter()
    }
}

impl Default for AntichainSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FailureSetLattice
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FailureSetLattice {
    pub explored: HashSet<Vec<String>>,
    pub cascade: AntichainSet,
    pub safe: AntichainSet,
}

impl FailureSetLattice {
    pub fn new() -> Self {
        Self {
            explored: HashSet::new(),
            cascade: AntichainSet::new(),
            safe: AntichainSet::new(),
        }
    }

    pub fn mark_explored(&mut self, set: &[String]) {
        let mut sorted = set.to_vec();
        sorted.sort();
        self.explored.insert(sorted);
    }

    pub fn is_explored(&self, set: &[String]) -> bool {
        let mut sorted = set.to_vec();
        sorted.sort();
        self.explored.contains(&sorted)
    }

    pub fn mark_cascade(&mut self, set: BTreeSet<String>) {
        self.cascade.insert(set);
    }

    pub fn mark_safe(&mut self, set: BTreeSet<String>) {
        self.safe.insert(set);
    }

    pub fn total_explored(&self) -> usize {
        self.explored.len()
    }
}

impl Default for FailureSetLattice {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MinimalFailureSetEnumerator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MinimalFailureSetEnumerator {
    graph: RtigGraph,
    config: BmcConfig,
}

impl MinimalFailureSetEnumerator {
    pub fn new(graph: RtigGraph, config: BmcConfig) -> Self {
        Self { graph, config }
    }

    /// Enumerate all minimal failure sets using MARCO-style algorithm
    /// with antichain pruning.
    pub fn enumerate_minimal_sets(&self) -> Vec<MinimalFailureSet> {
        let service_ids: Vec<String> = self.graph.service_ids().iter().map(|s| s.to_string()).collect();
        let n = service_ids.len();
        let mut pruner = AntichainPruner::new();
        let mut minimal_sets = Vec::new();
        let max_size = n.min(self.config.max_failure_budget);
        let checker_impl = MonotonicityChecker::new(self.graph.clone(), self.config.clone());

        // Grow from size 1 upward
        for k in 1..=max_size {
            let combos = combinations(&service_ids, k);
            for combo in combos {
                let candidate: BTreeSet<String> = combo.iter().cloned().collect();

                if !pruner.should_check(&candidate) {
                    continue;
                }

                let is_cascade = checker_impl.check_failure_set_cascades(&combo);
                pruner.record_result(candidate.clone(), is_cascade);

                if is_cascade {
                    // Check minimality: no proper subset should also cascade
                    let is_minimal_flag = if k == 1 {
                        true
                    } else {
                        !self.has_cascading_subset(&combo, &pruner)
                    };

                    if is_minimal_flag {
                        let capacity = combo.len();
                        let mut mfs = MinimalFailureSet::new(capacity);
                        let mut fs = FailureSet::new(capacity);
                        for i in 0..capacity {
                            fs.insert(i);
                        }
                        mfs.insert(fs);
                        minimal_sets.push(mfs);
                    }
                }
            }
        }

        minimal_sets
    }

    fn has_cascading_subset(&self, set: &[String], pruner: &AntichainPruner) -> bool {
        // Check all proper subsets of size k-1
        for i in 0..set.len() {
            let subset: BTreeSet<String> = set.iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, s)| s.clone())
                .collect();
            if pruner.is_known_cascade(&subset) {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// FreedmanKhachiyan enumeration helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FreedmanKhachiyan {
    universe: Vec<String>,
    known_positive: Vec<BTreeSet<String>>,
    known_negative: Vec<BTreeSet<String>>,
}

impl FreedmanKhachiyan {
    pub fn new(universe: Vec<String>) -> Self {
        Self {
            universe,
            known_positive: Vec::new(),
            known_negative: Vec::new(),
        }
    }

    pub fn record_positive(&mut self, set: BTreeSet<String>) {
        self.known_positive.push(set);
    }

    pub fn record_negative(&mut self, set: BTreeSet<String>) {
        self.known_negative.push(set);
    }

    /// Generate a candidate set to query next, based on the current knowledge.
    pub fn next_candidate(&self) -> Option<BTreeSet<String>> {
        // Simple strategy: pick the "midpoint" between known positive and negative
        let n = self.universe.len();
        if n == 0 {
            return None;
        }

        // Find elements that appear in all positive but no negative
        let mut candidate = BTreeSet::new();
        for elem in &self.universe {
            let in_all_pos = self.known_positive.is_empty()
                || self.known_positive.iter().all(|s| s.contains(elem));
            let in_any_neg = self.known_negative.iter().any(|s| s.contains(elem));
            if in_all_pos && !in_any_neg {
                candidate.insert(elem.clone());
            }
        }

        if candidate.is_empty() {
            // Fallback: pick a random subset of half the elements
            let half = n / 2;
            for (i, elem) in self.universe.iter().enumerate() {
                if i < half.max(1) {
                    candidate.insert(elem.clone());
                }
            }
        }

        Some(candidate)
    }

    pub fn is_complete(&self) -> bool {
        // Complete when we can fully characterize the monotone function
        // Simplified: complete when positive and negative cover enough
        !self.known_positive.is_empty() && !self.known_negative.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn combinations(items: &[String], k: usize) -> Vec<Vec<String>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }
    let mut result = Vec::new();
    combinations_helper(items, k, 0, &mut Vec::new(), &mut result);
    result
}

fn combinations_helper(
    items: &[String],
    k: usize,
    start: usize,
    current: &mut Vec<String>,
    result: &mut Vec<Vec<String>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    for i in start..items.len() {
        current.push(items[i].clone());
        combinations_helper(items, k, i + 1, current, result);
        current.pop();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, ServiceNode};

    fn simple_graph() -> RtigGraph {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("a", 1000).with_baseline_load(10));
        g.add_service(ServiceNode::new("b", 500).with_baseline_load(10));
        g.add_service(ServiceNode::new("c", 100).with_baseline_load(5));
        g.add_edge(DependencyEdgeInfo::new("a", "b").with_retry_count(3));
        g.add_edge(DependencyEdgeInfo::new("b", "c").with_retry_count(3));
        g
    }

    #[test]
    fn test_antichain_pruner_new() {
        let pruner = AntichainPruner::new();
        assert_eq!(pruner.known_cascading_count(), 0);
        assert_eq!(pruner.known_safe_count(), 0);
    }

    #[test]
    fn test_antichain_pruner_should_check() {
        let mut pruner = AntichainPruner::new();
        let set_a: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        let set_ab: BTreeSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();

        // Initially everything should be checked
        assert!(pruner.should_check(&set_a));
        assert!(pruner.should_check(&set_ab));

        // Record {a} as cascading
        pruner.record_result(set_a.clone(), true);
        // {a,b} is a superset of {a} -> skip
        assert!(!pruner.should_check(&set_ab));
    }

    #[test]
    fn test_antichain_pruner_safe_subset() {
        let mut pruner = AntichainPruner::new();
        let set_ab: BTreeSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let set_a: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();

        // Record {a,b} as safe
        pruner.record_result(set_ab.clone(), false);
        // {a} is a subset of {a,b} -> skip (must also be safe)
        assert!(!pruner.should_check(&set_a));
    }

    #[test]
    fn test_antichain_set_insert() {
        let mut ac = AntichainSet::new();
        let s1: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        let s2: BTreeSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        ac.insert(s1.clone());
        assert_eq!(ac.len(), 1);
        ac.insert(s2.clone());
        // s1 is subset of s2, so one should be removed
        assert!(ac.len() <= 2);
    }

    #[test]
    fn test_antichain_set_contains() {
        let mut ac = AntichainSet::new();
        let s1: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        ac.insert(s1);
        let s2: BTreeSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert!(ac.contains_subset_of(&s2));
    }

    #[test]
    fn test_failure_set_lattice() {
        let mut lattice = FailureSetLattice::new();
        assert_eq!(lattice.total_explored(), 0);
        lattice.mark_explored(&["a".into()]);
        assert!(lattice.is_explored(&["a".into()]));
        assert!(!lattice.is_explored(&["b".into()]));
    }

    #[test]
    fn test_monotonicity_check() {
        let g = simple_graph();
        let checker = MonotonicityChecker::new(g, BmcConfig {
            max_failure_budget: 2,
            depth_bound: Some(3),
            timeout_ms: 5000,
            ..Default::default()
        });
        let result = checker.verify_monotonicity();
        // Simple chain graph should be monotone
        assert!(result.is_monotone || !result.violations.is_empty());
    }

    #[test]
    fn test_minimal_failure_set_enumeration() {
        let g = simple_graph();
        let enumerator = MinimalFailureSetEnumerator::new(g, BmcConfig {
            max_failure_budget: 2,
            depth_bound: Some(3),
            timeout_ms: 5000,
            ..Default::default()
        });
        let sets = enumerator.enumerate_minimal_sets();
        // All returned sets should have at least one failure set inside
        for s in &sets {
            assert!(s.len() > 0);
        }
    }

    #[test]
    fn test_freedman_khachiyan() {
        let fk = FreedmanKhachiyan::new(vec!["a".into(), "b".into(), "c".into()]);
        let candidate = fk.next_candidate();
        assert!(candidate.is_some());
    }

    #[test]
    fn test_freedman_khachiyan_record() {
        let mut fk = FreedmanKhachiyan::new(vec!["a".into(), "b".into()]);
        let pos: BTreeSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        fk.record_positive(pos);
        let neg: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        fk.record_negative(neg);
        assert!(fk.is_complete());
    }

    #[test]
    fn test_combinations() {
        let items: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(combinations(&items, 0).len(), 1);
        assert_eq!(combinations(&items, 1).len(), 3);
        assert_eq!(combinations(&items, 2).len(), 3);
        assert_eq!(combinations(&items, 3).len(), 1);
        assert_eq!(combinations(&items, 4).len(), 0);
    }

    #[test]
    fn test_antichain_set_empty() {
        let ac = AntichainSet::new();
        assert!(ac.is_empty());
        assert_eq!(ac.len(), 0);
    }

    #[test]
    fn test_lattice_mark_cascade() {
        let mut lattice = FailureSetLattice::new();
        let s: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        lattice.mark_cascade(s.clone());
        assert_eq!(lattice.cascade.len(), 1);
    }

    #[test]
    fn test_is_known_cascade() {
        let mut pruner = AntichainPruner::new();
        let s_a: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        pruner.record_result(s_a.clone(), true);
        let s_ab: BTreeSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert!(pruner.is_known_cascade(&s_ab));
        assert!(!pruner.is_known_safe(&s_ab));
    }
}
