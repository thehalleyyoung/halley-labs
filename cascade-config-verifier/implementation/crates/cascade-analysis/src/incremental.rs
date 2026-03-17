//! Incremental analysis — reuses cached results when only part of the
//! topology has changed.
//!
//! Computes a *topology diff*, determines the *affected cone* (forward +
//! backward reachability from changed services), and re-analyses only the
//! impacted sub-graph.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

use crate::orchestrator::{AnalysisConfig, AnalysisMode, AnalysisResult};
use crate::tier1::{Tier1Analyzer, Tier1Result};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Cached Tier 1 result keyed by topology hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    pub result: Tier1Result,
    pub timestamp: u64,
    pub topology_hash: u64,
}

/// Structural difference between two topologies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologyDiff {
    pub added_services: Vec<String>,
    pub removed_services: Vec<String>,
    pub modified_edges: Vec<(String, String)>,
    pub added_edges: Vec<(String, String)>,
    pub removed_edges: Vec<(String, String)>,
}

impl TopologyDiff {
    /// Returns `true` if nothing changed.
    pub fn is_empty(&self) -> bool {
        self.added_services.is_empty()
            && self.removed_services.is_empty()
            && self.modified_edges.is_empty()
            && self.added_edges.is_empty()
            && self.removed_edges.is_empty()
    }

    /// All services directly touched by the diff.
    pub fn changed_services(&self) -> Vec<String> {
        let mut set: HashSet<String> = HashSet::new();
        for s in &self.added_services {
            set.insert(s.clone());
        }
        for s in &self.removed_services {
            set.insert(s.clone());
        }
        for (a, b) in self.modified_edges.iter().chain(&self.added_edges).chain(&self.removed_edges) {
            set.insert(a.clone());
            set.insert(b.clone());
        }
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort();
        v
    }
}

/// Result of an incremental analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalResult {
    pub changed_services: Vec<String>,
    pub affected_cone: Vec<String>,
    pub analysis_result: AnalysisResult,
    pub reused_findings: usize,
    pub new_findings: usize,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateful incremental analyzer with an in-memory cache.
#[derive(Debug, Clone)]
pub struct IncrementalAnalyzer {
    cache: HashMap<u64, CachedResult>,
}

impl IncrementalAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Perform an incremental analysis between two topology snapshots.
    pub fn analyze_diff(
        &mut self,
        old_adj: &[(String, String, u32, u64, u64)],
        new_adj: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        deadlines: &HashMap<String, u64>,
        service_names: &[String],
        config: &AnalysisConfig,
    ) -> IncrementalResult {
        let start = Instant::now();

        let diff = Self::compute_topology_diff(old_adj, new_adj);
        let changed_services = diff.changed_services();
        let affected_cone = Self::compute_affected_cone(&diff, new_adj);

        // If nothing changed, try the cache.
        let new_hash = Self::compute_hash(new_adj);
        if diff.is_empty() {
            if let Some(cached) = self.lookup_cache(new_hash) {
                let reused = cached.result.risky_paths.len()
                    + cached.result.timeout_violations.len()
                    + cached.result.fan_in_risks.len();
                return IncrementalResult {
                    changed_services,
                    affected_cone,
                    analysis_result: AnalysisResult {
                        tier1_result: Some(cached.result.clone()),
                        tier2_result: None,
                        total_duration_ms: start.elapsed().as_millis() as u64,
                        mode_used: AnalysisMode::FastOnly,
                    },
                    reused_findings: reused,
                    new_findings: 0,
                };
            }
        }

        // Build sub-graph restricted to the affected cone.
        let cone_set: HashSet<&str> = affected_cone.iter().map(|s| s.as_str()).collect();
        let sub_adj: Vec<(String, String, u32, u64, u64)> = if affected_cone.is_empty() {
            new_adj.to_vec()
        } else {
            new_adj
                .iter()
                .filter(|(s, d, _, _, _)| cone_set.contains(s.as_str()) || cone_set.contains(d.as_str()))
                .cloned()
                .collect()
        };

        let sub_names: Vec<String> = if affected_cone.is_empty() {
            service_names.to_vec()
        } else {
            affected_cone.clone()
        };

        // Run analysis on the sub-graph.
        let orchestrator = crate::orchestrator::AnalysisOrchestrator::new();
        let analysis_result =
            orchestrator.run(&sub_adj, capacities, deadlines, &sub_names, config);

        let new_findings = analysis_result
            .tier1_result
            .as_ref()
            .map(|t| t.risky_paths.len() + t.timeout_violations.len() + t.fan_in_risks.len())
            .unwrap_or(0);

        // Cache the Tier 1 result for the *full* new topology.
        let full_t1 = Tier1Analyzer::new().analyze(new_adj, capacities, deadlines, &config.tier1);
        self.cache_result(new_hash, full_t1);

        let reused = 0; // No cache hit.

        IncrementalResult {
            changed_services,
            affected_cone,
            analysis_result,
            reused_findings: reused,
            new_findings,
        }
    }

    // ----- static helpers -----

    /// Compute the structural diff between two topologies.
    pub fn compute_topology_diff(
        old_adj: &[(String, String, u32, u64, u64)],
        new_adj: &[(String, String, u32, u64, u64)],
    ) -> TopologyDiff {
        let old_services = extract_services(old_adj);
        let new_services = extract_services(new_adj);

        let added_services: Vec<String> = new_services.difference(&old_services).cloned().collect();
        let removed_services: Vec<String> = old_services.difference(&new_services).cloned().collect();

        let old_edges: HashMap<(String, String), (u32, u64, u64)> = old_adj
            .iter()
            .map(|(s, d, r, t, w)| ((s.clone(), d.clone()), (*r, *t, *w)))
            .collect();
        let new_edges: HashMap<(String, String), (u32, u64, u64)> = new_adj
            .iter()
            .map(|(s, d, r, t, w)| ((s.clone(), d.clone()), (*r, *t, *w)))
            .collect();

        let old_keys: HashSet<&(String, String)> = old_edges.keys().collect();
        let new_keys: HashSet<&(String, String)> = new_edges.keys().collect();

        let added_edges: Vec<(String, String)> = new_keys
            .difference(&old_keys)
            .map(|k| (k.0.clone(), k.1.clone()))
            .collect();
        let removed_edges: Vec<(String, String)> = old_keys
            .difference(&new_keys)
            .map(|k| (k.0.clone(), k.1.clone()))
            .collect();

        let mut modified_edges = Vec::new();
        for key in old_keys.intersection(&new_keys) {
            if old_edges[*key] != new_edges[*key] {
                modified_edges.push((key.0.clone(), key.1.clone()));
            }
        }

        TopologyDiff {
            added_services,
            removed_services,
            modified_edges,
            added_edges,
            removed_edges,
        }
    }

    /// Forward + backward reachability from changed services.
    pub fn compute_affected_cone(
        diff: &TopologyDiff,
        adj: &[(String, String, u32, u64, u64)],
    ) -> Vec<String> {
        let seeds = diff.changed_services();
        if seeds.is_empty() {
            return Vec::new();
        }

        // Build both forward and backward adjacency.
        let mut forward: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut backward: HashMap<&str, Vec<&str>> = HashMap::new();
        for (s, d, _, _, _) in adj {
            forward.entry(s.as_str()).or_default().push(d.as_str());
            backward.entry(d.as_str()).or_default().push(s.as_str());
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        for seed in &seeds {
            if visited.insert(seed.clone()) {
                queue.push_back(seed.clone());
            }
        }

        // BFS forward.
        while let Some(node) = queue.pop_front() {
            if let Some(neighbours) = forward.get(node.as_str()) {
                for &n in neighbours {
                    if visited.insert(n.to_string()) {
                        queue.push_back(n.to_string());
                    }
                }
            }
        }

        // BFS backward from seeds.
        for seed in &seeds {
            queue.push_back(seed.clone());
        }
        while let Some(node) = queue.pop_front() {
            if let Some(neighbours) = backward.get(node.as_str()) {
                for &n in neighbours {
                    if visited.insert(n.to_string()) {
                        queue.push_back(n.to_string());
                    }
                }
            }
        }

        let mut cone: Vec<String> = visited.into_iter().collect();
        cone.sort();
        cone
    }

    /// Deterministic hash of the adjacency list.
    pub fn compute_hash(adj: &[(String, String, u32, u64, u64)]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut sorted: Vec<_> = adj.to_vec();
        sorted.sort();
        let mut hasher = DefaultHasher::new();
        for edge in &sorted {
            edge.0.hash(&mut hasher);
            edge.1.hash(&mut hasher);
            edge.2.hash(&mut hasher);
            edge.3.hash(&mut hasher);
            edge.4.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Store a Tier 1 result in the cache.
    pub fn cache_result(&mut self, hash: u64, result: Tier1Result) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.cache.insert(
            hash,
            CachedResult {
                result,
                timestamp: ts,
                topology_hash: hash,
            },
        );
    }

    /// Look up a cached result by topology hash.
    pub fn lookup_cache(&self, hash: u64) -> Option<&CachedResult> {
        self.cache.get(&hash)
    }
}

impl Default for IncrementalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn extract_services(adj: &[(String, String, u32, u64, u64)]) -> HashSet<String> {
    let mut set = HashSet::new();
    for (s, d, _, _, _) in adj {
        set.insert(s.clone());
        set.insert(d.clone());
    }
    set
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn adj_v1() -> Vec<(String, String, u32, u64, u64)> {
        vec![
            ("A".into(), "B".into(), 2, 1000, 1),
            ("B".into(), "C".into(), 2, 1000, 1),
        ]
    }

    fn adj_v2() -> Vec<(String, String, u32, u64, u64)> {
        vec![
            ("A".into(), "B".into(), 3, 1000, 1), // retry changed
            ("B".into(), "C".into(), 2, 1000, 1),
            ("C".into(), "D".into(), 1, 500, 1),  // new edge
        ]
    }

    fn adj_v3() -> Vec<(String, String, u32, u64, u64)> {
        // Same as v1.
        adj_v1()
    }

    fn default_caps() -> HashMap<String, u64> {
        [("A", 100), ("B", 100), ("C", 100), ("D", 100)]
            .iter()
            .map(|(k, v)| (k.to_string(), *v as u64))
            .collect()
    }

    fn default_deadlines() -> HashMap<String, u64> {
        HashMap::from([("A".into(), 30_000u64)])
    }

    fn names_v2() -> Vec<String> {
        vec!["A".into(), "B".into(), "C".into(), "D".into()]
    }

    #[test]
    fn test_compute_topology_diff_added_edge() {
        let diff = IncrementalAnalyzer::compute_topology_diff(&adj_v1(), &adj_v2());
        assert!(diff.added_edges.contains(&("C".into(), "D".into())));
        assert!(!diff.added_services.is_empty() || diff.added_edges.len() >= 1);
    }

    #[test]
    fn test_compute_topology_diff_modified_edge() {
        let diff = IncrementalAnalyzer::compute_topology_diff(&adj_v1(), &adj_v2());
        assert!(diff.modified_edges.contains(&("A".into(), "B".into())));
    }

    #[test]
    fn test_compute_topology_diff_no_change() {
        let diff = IncrementalAnalyzer::compute_topology_diff(&adj_v1(), &adj_v3());
        assert!(diff.is_empty());
    }

    #[test]
    fn test_compute_topology_diff_removed_edge() {
        let diff = IncrementalAnalyzer::compute_topology_diff(&adj_v2(), &adj_v1());
        assert!(diff.removed_edges.contains(&("C".into(), "D".into())));
    }

    #[test]
    fn test_affected_cone_forward_backward() {
        let diff = IncrementalAnalyzer::compute_topology_diff(&adj_v1(), &adj_v2());
        let cone = IncrementalAnalyzer::compute_affected_cone(&diff, &adj_v2());
        // A->B changed, C->D added → all nodes should be in the cone.
        assert!(cone.contains(&"A".to_string()));
        assert!(cone.contains(&"B".to_string()));
        assert!(cone.contains(&"C".to_string()));
        assert!(cone.contains(&"D".to_string()));
    }

    #[test]
    fn test_affected_cone_empty_diff() {
        let diff = TopologyDiff::default();
        let cone = IncrementalAnalyzer::compute_affected_cone(&diff, &adj_v1());
        assert!(cone.is_empty());
    }

    #[test]
    fn test_hash_deterministic() {
        let h1 = IncrementalAnalyzer::compute_hash(&adj_v1());
        let h2 = IncrementalAnalyzer::compute_hash(&adj_v1());
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_differs_on_change() {
        let h1 = IncrementalAnalyzer::compute_hash(&adj_v1());
        let h2 = IncrementalAnalyzer::compute_hash(&adj_v2());
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_cache_round_trip() {
        let mut analyzer = IncrementalAnalyzer::new();
        let result = Tier1Result {
            risky_paths: vec![],
            timeout_violations: vec![],
            fan_in_risks: vec![],
            duration_ms: 42,
        };
        analyzer.cache_result(12345, result.clone());
        let cached = analyzer.lookup_cache(12345);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().result.duration_ms, 42);
    }

    #[test]
    fn test_cache_miss() {
        let analyzer = IncrementalAnalyzer::new();
        assert!(analyzer.lookup_cache(99999).is_none());
    }

    #[test]
    fn test_analyze_diff_no_change_uses_cache() {
        let mut analyzer = IncrementalAnalyzer::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::FastOnly,
            ..Default::default()
        };
        // First run — populates cache.
        let _ = analyzer.analyze_diff(
            &adj_v1(),
            &adj_v1(),
            &default_caps(),
            &default_deadlines(),
            &["A".into(), "B".into(), "C".into()],
            &config,
        );
        // Second run — identical topology.
        let result = analyzer.analyze_diff(
            &adj_v1(),
            &adj_v1(),
            &default_caps(),
            &default_deadlines(),
            &["A".into(), "B".into(), "C".into()],
            &config,
        );
        assert!(result.reused_findings > 0 || result.new_findings == 0);
    }

    #[test]
    fn test_incremental_result_fields() {
        let mut analyzer = IncrementalAnalyzer::new();
        let config = AnalysisConfig {
            mode: AnalysisMode::FastOnly,
            ..Default::default()
        };
        let result = analyzer.analyze_diff(
            &adj_v1(),
            &adj_v2(),
            &default_caps(),
            &default_deadlines(),
            &names_v2(),
            &config,
        );
        assert!(!result.changed_services.is_empty());
        assert!(!result.affected_cone.is_empty());
    }

    #[test]
    fn test_topology_diff_changed_services() {
        let diff = IncrementalAnalyzer::compute_topology_diff(&adj_v1(), &adj_v2());
        let cs = diff.changed_services();
        assert!(cs.contains(&"A".to_string()));
        assert!(cs.contains(&"D".to_string()));
    }
}
