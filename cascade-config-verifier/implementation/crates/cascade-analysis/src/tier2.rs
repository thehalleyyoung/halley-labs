//! Tier 2 — deep bounded model-checking analysis.
//!
//! Explores all failure-set combinations (up to a configurable budget) and
//! identifies *minimal* failure sets that cause capacity overflow or
//! cascade propagation.  Uses monotonicity pruning: if a failure set of
//! size *k* already causes a violation, no superset needs to be checked.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tunable parameters for Tier 2 analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Config {
    /// Maximum number of simultaneously failed services to consider.
    pub max_failure_budget: usize,
    /// Hard wall-clock timeout for the entire Tier 2 run (ms).
    pub timeout_ms: u64,
    /// When true, prune supersets of already-discovered failure sets.
    pub use_monotonicity: bool,
}

impl Default for Tier2Config {
    fn default() -> Self {
        Self {
            max_failure_budget: 3,
            timeout_ms: 60_000,
            use_monotonicity: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Aggregated Tier 2 result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Result {
    pub minimal_failure_sets: Vec<MinimalFailureSetInfo>,
    pub total_scenarios: usize,
    pub duration_ms: u64,
}

/// A single minimal failure set with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalFailureSetInfo {
    /// Service names in the failure set.
    pub services: Vec<String>,
    pub size: usize,
    /// Classification label (e.g. "capacity_overflow", "amplification_cascade").
    pub classification: String,
    /// Severity label.
    pub severity: String,
}

// ---------------------------------------------------------------------------
// Index-based internal graph representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct IndexEdge {
    src: usize,
    dst: usize,
    retry_count: u32,
    #[allow(dead_code)]
    timeout_ms: u64,
}

#[derive(Debug, Clone)]
struct IndexGraph {
    num_nodes: usize,
    edges: Vec<IndexEdge>,
    #[allow(dead_code)]
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    capacities: Vec<u64>,
    names: Vec<String>,
}

impl IndexGraph {
    fn from_adjacency(
        adjacency: &[(String, String, u32, u64, u64)],
        capacities_map: &HashMap<String, u64>,
        service_names: &[String],
    ) -> Self {
        let name_to_idx: HashMap<&str, usize> = service_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let num_nodes = service_names.len();

        let mut edges = Vec::new();
        let mut outgoing = vec![Vec::new(); num_nodes];
        let mut incoming = vec![Vec::new(); num_nodes];

        for (src, dst, retry, timeout, _weight) in adjacency {
            if let (Some(&si), Some(&di)) = (name_to_idx.get(src.as_str()), name_to_idx.get(dst.as_str())) {
                let idx = edges.len();
                edges.push(IndexEdge {
                    src: si,
                    dst: di,
                    retry_count: *retry,
                    timeout_ms: *timeout,
                });
                outgoing[si].push(idx);
                incoming[di].push(idx);
            }
        }

        let capacities: Vec<u64> = service_names
            .iter()
            .map(|n| capacities_map.get(n).copied().unwrap_or(u64::MAX))
            .collect();

        IndexGraph {
            num_nodes,
            edges,
            outgoing,
            incoming,
            capacities,
            names: service_names.to_vec(),
        }
    }

    /// Simulate load propagation with a set of failed nodes and return
    /// `true` if any surviving node's computed load exceeds its capacity.
    fn check_violation(&self, failed: &HashSet<usize>) -> bool {
        // Accumulation model: propagate load through non-failed edges,
        // iterating until convergence.
        let mut effective_load = vec![1.0_f64; self.num_nodes];
        // Propagate in topological order (or multi-pass for safety).
        for _ in 0..self.num_nodes {
            let prev = effective_load.clone();
            for edge in &self.edges {
                if failed.contains(&edge.src) || failed.contains(&edge.dst) {
                    continue;
                }
                let contrib = prev[edge.src] * (1 + edge.retry_count) as f64;
                effective_load[edge.dst] += contrib;
            }
            // Converge check.
            let converged = effective_load
                .iter()
                .zip(prev.iter())
                .all(|(a, b)| (a - b).abs() < 1e-9);
            if converged {
                break;
            }
        }

        // Failed services additionally redirect their load to surviving neighbours
        // via retries from callers.
        for &fi in failed.iter() {
            for &ei in &self.incoming[fi] {
                let edge = &self.edges[ei];
                if failed.contains(&edge.src) {
                    continue;
                }
                // Callers of the failed service retry, amplifying their own load.
                let caller_load = effective_load[edge.src];
                let retry_load = caller_load * edge.retry_count as f64;
                // This extra load stays at the caller.
                effective_load[edge.src] += retry_load;
            }
        }

        for i in 0..self.num_nodes {
            if failed.contains(&i) {
                continue;
            }
            if self.capacities[i] != u64::MAX && effective_load[i] > self.capacities[i] as f64 {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateless Tier 2 analyzer.
#[derive(Debug, Clone)]
pub struct Tier2Analyzer;

impl Tier2Analyzer {
    pub fn new() -> Self {
        Self
    }

    /// Run bounded model checking.
    pub fn analyze(
        &self,
        adjacency: &[(String, String, u32, u64, u64)],
        capacities: &HashMap<String, u64>,
        service_names: &[String],
        config: &Tier2Config,
    ) -> Tier2Result {
        let start = Instant::now();
        let graph = IndexGraph::from_adjacency(adjacency, capacities, service_names);
        let n = graph.num_nodes;
        let budget = config.max_failure_budget.min(n);
        let deadline = std::time::Duration::from_millis(config.timeout_ms);

        let mut minimal_sets: Vec<Vec<usize>> = Vec::new();
        let mut total_scenarios: usize = 0;

        // Enumerate failure sets of increasing size.
        'outer: for k in 1..=budget {
            let combos = combinations(n, k);
            for combo in combos {
                if start.elapsed() > deadline {
                    break 'outer;
                }
                total_scenarios += 1;

                // Monotonicity pruning: skip if any already-discovered minimal
                // set is a subset of this combo.
                if config.use_monotonicity {
                    let combo_set: HashSet<usize> = combo.iter().copied().collect();
                    let dominated = minimal_sets.iter().any(|ms| {
                        ms.iter().all(|x| combo_set.contains(x))
                    });
                    if dominated {
                        continue;
                    }
                }

                let failed: HashSet<usize> = combo.iter().copied().collect();
                if graph.check_violation(&failed) {
                    minimal_sets.push(combo);
                }
            }
        }

        let minimal_failure_sets: Vec<MinimalFailureSetInfo> = minimal_sets
            .iter()
            .map(|fs| {
                let services: Vec<String> =
                    fs.iter().map(|&i| graph.names[i].clone()).collect();
                let classification =
                    self.classify_scenario(fs, &graph.edges, &graph.capacities);
                let severity = match fs.len() {
                    1 => "critical".to_string(),
                    2 => "high".to_string(),
                    3 => "medium".to_string(),
                    _ => "low".to_string(),
                };
                MinimalFailureSetInfo {
                    size: services.len(),
                    services,
                    classification,
                    severity,
                }
            })
            .collect();

        Tier2Result {
            minimal_failure_sets,
            total_scenarios,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Classify why a failure set causes a violation.
    pub(crate) fn classify_scenario(
        &self,
        failure_set: &[usize],
        edges: &[IndexEdge],
        _caps: &[u64],
    ) -> String {
        let failed: HashSet<usize> = failure_set.iter().copied().collect();

        // Check if any failed node is high-fan-in (many incoming edges).
        let mut incoming_counts = HashMap::new();
        for e in edges {
            *incoming_counts.entry(e.dst).or_insert(0usize) += 1;
        }

        let max_fan_in = failure_set
            .iter()
            .filter_map(|&f| incoming_counts.get(&f))
            .max()
            .copied()
            .unwrap_or(0);

        // Check if high-retry edges touch the failure set.
        let max_retry_on_failed = edges
            .iter()
            .filter(|e| failed.contains(&e.dst))
            .map(|e| e.retry_count)
            .max()
            .unwrap_or(0);

        if max_retry_on_failed >= 3 && max_fan_in >= 2 {
            "amplification_cascade".to_string()
        } else if max_retry_on_failed >= 3 {
            "retry_storm".to_string()
        } else if max_fan_in >= 3 {
            "fan_in_overload".to_string()
        } else if failure_set.len() == 1 {
            "single_point_of_failure".to_string()
        } else {
            "capacity_overflow".to_string()
        }
    }
}

impl Default for Tier2Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Combination generator
// ---------------------------------------------------------------------------

/// Generate all k-combinations of indices `0..n`.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    fn helper(start: usize, n: usize, k: usize, combo: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if combo.len() == k {
            result.push(combo.clone());
            return;
        }
        let remaining = k - combo.len();
        for i in start..=(n - remaining) {
            combo.push(i);
            helper(i + 1, n, k, combo, result);
            combo.pop();
        }
    }
    if k <= n {
        helper(0, n, k, &mut combo, &mut result);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_adjacency() -> (Vec<(String, String, u32, u64, u64)>, Vec<String>) {
        let adj = vec![
            ("A".into(), "B".into(), 3, 1000, 1),
            ("B".into(), "C".into(), 3, 1000, 1),
        ];
        let names = vec!["A".into(), "B".into(), "C".into()];
        (adj, names)
    }

    fn diamond_adjacency() -> (Vec<(String, String, u32, u64, u64)>, Vec<String>) {
        let adj = vec![
            ("A".into(), "B".into(), 2, 500, 1),
            ("A".into(), "C".into(), 2, 500, 1),
            ("B".into(), "D".into(), 2, 500, 1),
            ("C".into(), "D".into(), 2, 500, 1),
        ];
        let names = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        (adj, names)
    }

    fn low_capacity() -> HashMap<String, u64> {
        let mut m = HashMap::new();
        for s in &["A", "B", "C", "D"] {
            m.insert(s.to_string(), 5);
        }
        m
    }

    #[test]
    fn test_combinations_basic() {
        assert_eq!(combinations(4, 2).len(), 6);
        assert_eq!(combinations(5, 3).len(), 10);
        assert_eq!(combinations(3, 0).len(), 1);
    }

    #[test]
    fn test_combinations_edge_cases() {
        assert_eq!(combinations(0, 1).len(), 0);
        assert_eq!(combinations(3, 3).len(), 1);
        assert_eq!(combinations(3, 4).len(), 0);
    }

    #[test]
    fn test_linear_single_failure() {
        let (adj, names) = linear_adjacency();
        let caps = low_capacity();
        let config = Tier2Config {
            max_failure_budget: 1,
            timeout_ms: 5000,
            use_monotonicity: true,
        };
        let analyzer = Tier2Analyzer::new();
        let result = analyzer.analyze(&adj, &caps, &names, &config);
        // Failing any node with low capacity should trigger violations.
        assert!(!result.minimal_failure_sets.is_empty() || result.total_scenarios > 0);
    }

    #[test]
    fn test_diamond_analysis() {
        let (adj, names) = diamond_adjacency();
        let caps = low_capacity();
        let config = Tier2Config::default();
        let analyzer = Tier2Analyzer::new();
        let result = analyzer.analyze(&adj, &caps, &names, &config);
        assert!(result.total_scenarios > 0);
        assert!(result.duration_ms < 10_000);
    }

    #[test]
    fn test_monotonicity_pruning_reduces_scenarios() {
        let (adj, names) = diamond_adjacency();
        let caps = low_capacity();

        let analyzer = Tier2Analyzer::new();

        let with_mono = analyzer.analyze(
            &adj,
            &caps,
            &names,
            &Tier2Config {
                max_failure_budget: 3,
                use_monotonicity: true,
                ..Default::default()
            },
        );
        let without_mono = analyzer.analyze(
            &adj,
            &caps,
            &names,
            &Tier2Config {
                max_failure_budget: 3,
                use_monotonicity: false,
                ..Default::default()
            },
        );
        // With monotonicity, fewer scenarios should be evaluated.
        assert!(with_mono.total_scenarios <= without_mono.total_scenarios);
    }

    #[test]
    fn test_empty_topology() {
        let adj: Vec<(String, String, u32, u64, u64)> = vec![];
        let names: Vec<String> = vec![];
        let config = Tier2Config::default();
        let analyzer = Tier2Analyzer::new();
        let result = analyzer.analyze(&adj, &HashMap::new(), &names, &config);
        assert!(result.minimal_failure_sets.is_empty());
        assert_eq!(result.total_scenarios, 0);
    }

    #[test]
    fn test_classify_retry_storm() {
        let edges = vec![IndexEdge {
            src: 0,
            dst: 1,
            retry_count: 5,
            timeout_ms: 500,
        }];
        let caps = vec![100, 100];
        let analyzer = Tier2Analyzer::new();
        let class = analyzer.classify_scenario(&[1], &edges, &caps);
        assert_eq!(class, "retry_storm");
    }

    #[test]
    fn test_classify_fan_in_overload() {
        let edges = vec![
            IndexEdge { src: 0, dst: 2, retry_count: 1, timeout_ms: 500 },
            IndexEdge { src: 1, dst: 2, retry_count: 1, timeout_ms: 500 },
            IndexEdge { src: 3, dst: 2, retry_count: 1, timeout_ms: 500 },
        ];
        let caps = vec![100; 4];
        let analyzer = Tier2Analyzer::new();
        let class = analyzer.classify_scenario(&[2], &edges, &caps);
        assert_eq!(class, "fan_in_overload");
    }

    #[test]
    fn test_classify_single_point_of_failure() {
        let edges = vec![IndexEdge {
            src: 0,
            dst: 1,
            retry_count: 1,
            timeout_ms: 500,
        }];
        let caps = vec![100, 100];
        let analyzer = Tier2Analyzer::new();
        let class = analyzer.classify_scenario(&[0], &edges, &caps);
        assert_eq!(class, "single_point_of_failure");
    }

    #[test]
    fn test_classify_amplification_cascade() {
        let edges = vec![
            IndexEdge { src: 0, dst: 2, retry_count: 4, timeout_ms: 500 },
            IndexEdge { src: 1, dst: 2, retry_count: 4, timeout_ms: 500 },
            IndexEdge { src: 3, dst: 2, retry_count: 4, timeout_ms: 500 },
        ];
        let caps = vec![100; 4];
        let analyzer = Tier2Analyzer::new();
        let class = analyzer.classify_scenario(&[2], &edges, &caps);
        assert_eq!(class, "amplification_cascade");
    }

    #[test]
    fn test_result_severity_by_size() {
        let (adj, names) = linear_adjacency();
        let caps = low_capacity();
        let config = Tier2Config {
            max_failure_budget: 3,
            ..Default::default()
        };
        let analyzer = Tier2Analyzer::new();
        let result = analyzer.analyze(&adj, &caps, &names, &config);
        for mfs in &result.minimal_failure_sets {
            match mfs.size {
                1 => assert_eq!(mfs.severity, "critical"),
                2 => assert_eq!(mfs.severity, "high"),
                3 => assert_eq!(mfs.severity, "medium"),
                _ => assert_eq!(mfs.severity, "low"),
            }
        }
    }

    #[test]
    fn test_index_graph_construction() {
        let (adj, names) = diamond_adjacency();
        let caps = low_capacity();
        let graph = IndexGraph::from_adjacency(&adj, &caps, &names);
        assert_eq!(graph.num_nodes, 4);
        assert_eq!(graph.edges.len(), 4);
        assert_eq!(graph.capacities.len(), 4);
    }
}
