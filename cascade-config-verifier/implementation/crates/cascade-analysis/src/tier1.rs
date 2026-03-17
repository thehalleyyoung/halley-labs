//! Tier 1 — fast, graph-based cascade risk analysis.
//!
//! Enumerates simple paths through a service adjacency graph and flags
//! retry-amplification risks, timeout-budget violations, and fan-in storms.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tunable thresholds for Tier 1 analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Config {
    /// Minimum amplification factor to flag as risky (e.g. 8.0).
    pub amplification_threshold: f64,
    /// Per-path timeout budget violation threshold in milliseconds.
    pub timeout_threshold_ms: u64,
    /// Maximum simple-path length to enumerate (limits combinatorial explosion).
    pub max_path_length: usize,
}

impl Default for Tier1Config {
    fn default() -> Self {
        Self {
            amplification_threshold: 8.0,
            timeout_threshold_ms: 30_000,
            max_path_length: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Aggregated result of Tier 1 analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Result {
    pub risky_paths: Vec<AmplificationRisk>,
    pub timeout_violations: Vec<TimeoutViolation>,
    pub fan_in_risks: Vec<FanInRisk>,
    pub duration_ms: u64,
}

/// A path whose cumulative retry amplification exceeds the threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplificationRisk {
    /// Service names along the path (source → … → sink).
    pub path: Vec<String>,
    /// Product of `(1 + retry_count)` across every edge.
    pub amplification_factor: f64,
    /// Minimum service capacity along the path.
    pub capacity: u64,
    /// Human-readable severity label.
    pub severity: String,
}

/// A path whose total worst-case timeout budget exceeds a deadline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutViolation {
    pub path: Vec<String>,
    /// Sum of `timeout_ms * (1 + retry_count)` along the path.
    pub total_timeout_ms: u64,
    /// The deadline against which the total was compared.
    pub deadline_ms: u64,
    /// `total_timeout_ms - deadline_ms`.
    pub excess_ms: u64,
}

/// A service that receives calls from many upstream paths whose combined
/// amplification may overwhelm its capacity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanInRisk {
    pub service: String,
    pub incoming_paths: Vec<Vec<String>>,
    pub combined_amplification: f64,
    pub capacity: u64,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateless Tier 1 analyzer.
#[derive(Debug, Clone)]
pub struct Tier1Analyzer;

impl Tier1Analyzer {
    pub fn new() -> Self {
        Self
    }

    /// Run the full Tier 1 analysis pipeline.
    ///
    /// # Arguments
    /// * `adjacency` – `(src, dst, retry_count, timeout_ms, weight)` tuples.
    /// * `capacities` – mapping from service name to capacity (requests/s).
    /// * `deadlines` – mapping from service name to overall deadline (ms).
    /// * `config` – tunable thresholds.
    pub fn analyze(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        deadlines: &HashMap<String, u64>,
        config: &Tier1Config,
    ) -> Tier1Result {
        let start = Instant::now();

        let risky_paths = self.find_amplification_risks(adjacency, capacities, config);
        let timeout_violations = self.find_timeout_violations(adjacency, deadlines, config);
        let fan_in_risks = self.find_fan_in_risks(adjacency, capacities, config);

        Tier1Result {
            risky_paths,
            timeout_violations,
            fan_in_risks,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    // ----- amplification -----

    pub fn find_amplification_risks(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        config: &Tier1Config,
    ) -> Vec<AmplificationRisk> {
        let adj = build_adjacency_map(adjacency);
        let sources = find_sources(adjacency);
        let mut results = Vec::new();

        for src in &sources {
            let paths = enumerate_simple_paths(&adj, src, config.max_path_length);
            for path in paths {
                let amp = compute_amplification(&path, adjacency);
                if amp >= config.amplification_threshold {
                    let cap = min_capacity_along_path(&path, capacities);
                    let severity = amplification_severity(amp);
                    results.push(AmplificationRisk {
                        path,
                        amplification_factor: amp,
                        capacity: cap,
                        severity,
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            b.amplification_factor
                .partial_cmp(&a.amplification_factor)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    // ----- timeout violations -----

    pub fn find_timeout_violations(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        deadlines: &HashMap<String, u64>,
        config: &Tier1Config,
    ) -> Vec<TimeoutViolation> {
        let adj = build_adjacency_map(adjacency);
        let sources = find_sources(adjacency);
        let mut results = Vec::new();

        for src in &sources {
            let paths = enumerate_simple_paths(&adj, src, config.max_path_length);
            for path in paths {
                let total = compute_timeout_budget(&path, adjacency);
                // Use the source's deadline if available, otherwise the global threshold.
                let deadline = deadlines
                    .get(src)
                    .copied()
                    .unwrap_or(config.timeout_threshold_ms);
                if total > deadline {
                    results.push(TimeoutViolation {
                        path,
                        total_timeout_ms: total,
                        deadline_ms: deadline,
                        excess_ms: total - deadline,
                    });
                }
            }
        }

        results.sort_by(|a, b| b.excess_ms.cmp(&a.excess_ms));
        results
    }

    // ----- fan-in risks -----

    pub fn find_fan_in_risks(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        config: &Tier1Config,
    ) -> Vec<FanInRisk> {
        let adj = build_adjacency_map(adjacency);
        let sources = find_sources(adjacency);

        // Collect every path reaching a given service.
        let mut paths_to_service: HashMap<String, Vec<(Vec<String>, f64)>> = HashMap::new();

        for src in &sources {
            let paths = enumerate_simple_paths(&adj, src, config.max_path_length);
            for path in paths {
                let amp = compute_amplification(&path, adjacency);
                if let Some(last) = path.last() {
                    paths_to_service
                        .entry(last.clone())
                        .or_default()
                        .push((path, amp));
                }
            }
        }

        let mut results = Vec::new();
        for (service, entries) in &paths_to_service {
            if entries.len() < 2 {
                continue;
            }
            let combined: f64 = entries.iter().map(|(_, a)| *a).sum();
            let cap = capacities.get(service).copied().unwrap_or(u64::MAX);
            if combined >= config.amplification_threshold {
                results.push(FanInRisk {
                    service: service.clone(),
                    incoming_paths: entries.iter().map(|(p, _)| p.clone()).collect(),
                    combined_amplification: combined,
                    capacity: cap,
                });
            }
        }

        results.sort_by(|a, b| {
            b.combined_amplification
                .partial_cmp(&a.combined_amplification)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

impl Default for Tier1Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Adjacency map: node → Vec<(neighbour, edge index)>.
type AdjMap = HashMap<String, Vec<(String, usize)>>;

fn build_adjacency_map(adj: &[(String, String, u32, u64, u64)]) -> AdjMap {
    let mut map: AdjMap = HashMap::new();
    for (i, (src, dst, _, _, _)) in adj.iter().enumerate() {
        map.entry(src.clone()).or_default().push((dst.clone(), i));
        // Ensure destination exists in map even if it has no outgoing edges.
        map.entry(dst.clone()).or_default();
    }
    map
}

/// Return nodes that appear as a source but never as a destination.
fn find_sources(adj: &[(String, String, u32, u64, u64)]) -> Vec<String> {
    let srcs: HashSet<&str> = adj.iter().map(|(s, _, _, _, _)| s.as_str()).collect();
    let dsts: HashSet<&str> = adj.iter().map(|(_, d, _, _, _)| d.as_str()).collect();
    let mut sources: Vec<String> = srcs.difference(&dsts).map(|s| s.to_string()).collect();
    if sources.is_empty() {
        // Every node is both source and dest (cycles only) — start from all nodes.
        sources = srcs.into_iter().map(|s| s.to_string()).collect();
    }
    sources.sort();
    sources
}

/// Enumerate all simple (no repeated node) paths from `start` up to
/// `max_length` edges using iterative DFS.
pub fn enumerate_simple_paths(
    adj: &AdjMap,
    start: &str,
    max_length: usize,
) -> Vec<Vec<String>> {
    let mut results = Vec::new();
    // Stack: (current_node, path_so_far, visited_set)
    let mut stack: Vec<(String, Vec<String>, HashSet<String>)> = Vec::new();

    let mut init_visited = HashSet::new();
    init_visited.insert(start.to_string());
    stack.push((start.to_string(), vec![start.to_string()], init_visited));

    while let Some((node, path, visited)) = stack.pop() {
        if let Some(neighbors) = adj.get(&node) {
            let mut has_unvisited_neighbor = false;
            for (next, _) in neighbors {
                if visited.contains(next) {
                    continue;
                }
                if path.len() >= max_length + 1 {
                    continue; // path already at max edges
                }
                has_unvisited_neighbor = true;
                let mut new_path = path.clone();
                new_path.push(next.clone());
                let mut new_visited = visited.clone();
                new_visited.insert(next.clone());
                stack.push((next.clone(), new_path.clone(), new_visited));
                // Also record the extended path as a result.
                results.push(new_path);
            }
            if !has_unvisited_neighbor && path.len() > 1 {
                // Dead end — path already recorded when we extended.
            }
        }
        // Single-node paths are not interesting.
    }

    results
}

/// Product of `(1 + retry_count)` along edges connecting consecutive path nodes.
fn compute_amplification(path: &[String], adj: &[(String, String, u32, u64, u64)]) -> f64 {
    let edge_map = build_edge_lookup(adj);
    let mut amp: f64 = 1.0;
    for w in path.windows(2) {
        if let Some(&(retry, _, _)) = edge_map.get(&(w[0].as_str(), w[1].as_str())) {
            amp *= (1 + retry) as f64;
        }
    }
    amp
}

/// Sum of `timeout_ms * (1 + retry_count)` along the path.
fn compute_timeout_budget(path: &[String], adj: &[(String, String, u32, u64, u64)]) -> u64 {
    let edge_map = build_edge_lookup(adj);
    let mut total: u64 = 0;
    for w in path.windows(2) {
        if let Some(&(retry, timeout, _)) = edge_map.get(&(w[0].as_str(), w[1].as_str())) {
            total += timeout * (1 + retry as u64);
        }
    }
    total
}

/// Build a lookup from (src, dst) → (retry_count, timeout_ms, weight).
fn build_edge_lookup<'a>(
    adj: &'a [(String, String, u32, u64, u64)],
) -> HashMap<(&'a str, &'a str), (u32, u64, u64)> {
    let mut map = HashMap::new();
    for (s, d, r, t, w) in adj {
        map.insert((s.as_str(), d.as_str()), (*r, *t, *w));
    }
    map
}

fn min_capacity_along_path(path: &[String], capacities: &HashMap<String, u64>) -> u64 {
    path.iter()
        .filter_map(|s| capacities.get(s).copied())
        .min()
        .unwrap_or(u64::MAX)
}

fn amplification_severity(amp: f64) -> String {
    if amp >= 100.0 {
        "critical".to_string()
    } else if amp >= 50.0 {
        "high".to_string()
    } else if amp >= 20.0 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_adjacency() -> Vec<(String, String, u32, u64, u64)> {
        // A -> B -> C -> D, retry=2 each, timeout=1000 each
        vec![
            ("A".into(), "B".into(), 2, 1000, 1),
            ("B".into(), "C".into(), 2, 1000, 1),
            ("C".into(), "D".into(), 2, 1000, 1),
        ]
    }

    fn tree_adjacency() -> Vec<(String, String, u32, u64, u64)> {
        //       A
        //      / \
        //     B   C
        //    / \
        //   D   E
        vec![
            ("A".into(), "B".into(), 3, 500, 1),
            ("A".into(), "C".into(), 1, 500, 1),
            ("B".into(), "D".into(), 3, 500, 1),
            ("B".into(), "E".into(), 2, 500, 1),
        ]
    }

    fn mesh_adjacency() -> Vec<(String, String, u32, u64, u64)> {
        // A -> B, A -> C, B -> D, C -> D (diamond)
        vec![
            ("A".into(), "B".into(), 2, 1000, 1),
            ("A".into(), "C".into(), 2, 1000, 1),
            ("B".into(), "D".into(), 2, 1000, 1),
            ("C".into(), "D".into(), 2, 1000, 1),
        ]
    }

    fn default_capacities() -> HashMap<String, u64> {
        let mut m = HashMap::new();
        for s in &["A", "B", "C", "D", "E"] {
            m.insert(s.to_string(), 1000);
        }
        m
    }

    fn default_deadlines() -> HashMap<String, u64> {
        let mut m = HashMap::new();
        m.insert("A".into(), 5000);
        m
    }

    #[test]
    fn test_chain_amplification() {
        // (1+2)^3 = 27 for the full path A->B->C->D
        let adj = chain_adjacency();
        let config = Tier1Config {
            amplification_threshold: 1.0,
            ..Default::default()
        };
        let analyzer = Tier1Analyzer::new();
        let risks = analyzer.find_amplification_risks(&adj, &default_capacities(), &config);
        let full_path_risk = risks
            .iter()
            .find(|r| r.path == vec!["A", "B", "C", "D"]);
        assert!(full_path_risk.is_some());
        let risk = full_path_risk.unwrap();
        assert!((risk.amplification_factor - 27.0).abs() < 0.001);
    }

    #[test]
    fn test_chain_timeout_budget() {
        // Each edge: 1000 * 3 = 3000, three edges → 9000
        let adj = chain_adjacency();
        let deadlines = HashMap::from([("A".into(), 5000u64)]);
        let config = Tier1Config {
            timeout_threshold_ms: 5000,
            ..Default::default()
        };
        let analyzer = Tier1Analyzer::new();
        let violations =
            analyzer.find_timeout_violations(&adj, &deadlines, &config);
        let full_viol = violations.iter().find(|v| v.path == vec!["A", "B", "C", "D"]);
        assert!(full_viol.is_some());
        let v = full_viol.unwrap();
        assert_eq!(v.total_timeout_ms, 9000);
        assert_eq!(v.excess_ms, 4000);
    }

    #[test]
    fn test_tree_paths() {
        let adj = tree_adjacency();
        let adj_map = build_adjacency_map(&adj);
        let paths = enumerate_simple_paths(&adj_map, "A", 10);
        // Paths: A->B, A->C, A->B->D, A->B->E
        assert_eq!(paths.len(), 4);
    }

    #[test]
    fn test_tree_amplification_abd() {
        // A->B (1+3)=4, B->D (1+3)=4, total 16
        let adj = tree_adjacency();
        let config = Tier1Config {
            amplification_threshold: 10.0,
            ..Default::default()
        };
        let analyzer = Tier1Analyzer::new();
        let risks = analyzer.find_amplification_risks(&adj, &default_capacities(), &config);
        let abd = risks.iter().find(|r| r.path == vec!["A", "B", "D"]);
        assert!(abd.is_some());
        assert!((abd.unwrap().amplification_factor - 16.0).abs() < 0.001);
    }

    #[test]
    fn test_mesh_fan_in() {
        // D receives from two paths: A->B->D and A->C->D
        let adj = mesh_adjacency();
        let config = Tier1Config {
            amplification_threshold: 1.0,
            ..Default::default()
        };
        let analyzer = Tier1Analyzer::new();
        let fan_in = analyzer.find_fan_in_risks(&adj, &default_capacities(), &config);
        let d_risk = fan_in.iter().find(|r| r.service == "D");
        assert!(d_risk.is_some());
        assert_eq!(d_risk.unwrap().incoming_paths.len(), 2);
    }

    #[test]
    fn test_no_risks_below_threshold() {
        let adj = vec![("A".into(), "B".into(), 0u32, 100u64, 1u64)];
        let config = Tier1Config {
            amplification_threshold: 8.0,
            timeout_threshold_ms: 30_000,
            max_path_length: 10,
        };
        let analyzer = Tier1Analyzer::new();
        let result = analyzer.analyze(&adj, &default_capacities(), &default_deadlines(), &config);
        assert!(result.risky_paths.is_empty());
        assert!(result.timeout_violations.is_empty());
    }

    #[test]
    fn test_severity_labels() {
        assert_eq!(amplification_severity(150.0), "critical");
        assert_eq!(amplification_severity(60.0), "high");
        assert_eq!(amplification_severity(25.0), "medium");
        assert_eq!(amplification_severity(5.0), "low");
    }

    #[test]
    fn test_full_analysis_returns_duration() {
        let adj = chain_adjacency();
        let analyzer = Tier1Analyzer::new();
        let config = Tier1Config::default();
        let result = analyzer.analyze(&adj, &default_capacities(), &default_deadlines(), &config);
        // duration_ms should be set (≥ 0).
        assert!(result.duration_ms < 10_000);
    }

    #[test]
    fn test_empty_adjacency() {
        let adj: Vec<(String, String, u32, u64, u64)> = vec![];
        let analyzer = Tier1Analyzer::new();
        let config = Tier1Config::default();
        let result = analyzer.analyze(&adj, &HashMap::new(), &HashMap::new(), &config);
        assert!(result.risky_paths.is_empty());
        assert!(result.timeout_violations.is_empty());
        assert!(result.fan_in_risks.is_empty());
    }

    #[test]
    fn test_single_edge_amplification() {
        // retry=9 → amplification = 10
        let adj = vec![("X".into(), "Y".into(), 9, 100, 1)];
        let config = Tier1Config {
            amplification_threshold: 5.0,
            ..Default::default()
        };
        let analyzer = Tier1Analyzer::new();
        let risks = analyzer.find_amplification_risks(&adj, &HashMap::new(), &config);
        assert_eq!(risks.len(), 1);
        assert!((risks[0].amplification_factor - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_max_path_length_limit() {
        // Long chain: 0 -> 1 -> 2 -> ... -> 15
        let adj: Vec<(String, String, u32, u64, u64)> = (0..15)
            .map(|i| (i.to_string(), (i + 1).to_string(), 1, 100, 1))
            .collect();
        let adj_map = build_adjacency_map(&adj);
        let paths = enumerate_simple_paths(&adj_map, "0", 5);
        // No path should have more than 6 nodes (5 edges).
        for p in &paths {
            assert!(p.len() <= 6, "path too long: {:?}", p);
        }
    }

    #[test]
    fn test_min_capacity_along_path() {
        let mut caps = HashMap::new();
        caps.insert("A".into(), 500);
        caps.insert("B".into(), 200);
        caps.insert("C".into(), 1000);
        let path = vec!["A".into(), "B".into(), "C".into()];
        assert_eq!(min_capacity_along_path(&path, &caps), 200);
    }
}
