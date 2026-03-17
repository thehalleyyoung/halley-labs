//! Analysis statistics and topology metrics.
//!
//! Provides [`TopologyStats`] (graph-level metrics), [`RiskSummary`]
//! (severity distribution), histogram construction, and performance metrics.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::tier1::Tier1Result;

// ---------------------------------------------------------------------------
// TopologyStats
// ---------------------------------------------------------------------------

/// Graph-level structural metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologyStats {
    pub service_count: usize,
    pub edge_count: usize,
    pub max_depth: usize,
    pub diameter: usize,
    pub avg_fan_in: f64,
    pub avg_fan_out: f64,
    pub max_amplification: f64,
    pub max_timeout_chain_ms: u64,
}

// ---------------------------------------------------------------------------
// RiskSummary
// ---------------------------------------------------------------------------

/// Severity distribution of findings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskSummary {
    pub critical_count: usize,
    pub high_count: usize,
    pub medium_count: usize,
    pub low_count: usize,
    pub total_risk_score: f64,
}

// ---------------------------------------------------------------------------
// PerformanceMetrics
// ---------------------------------------------------------------------------

/// Timing and resource metrics for an analysis run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub analysis_time_ms: u64,
    pub memory_estimate_bytes: usize,
}

// ---------------------------------------------------------------------------
// AnalysisStatistics
// ---------------------------------------------------------------------------

/// Stateless statistics collector.
#[derive(Debug, Clone)]
pub struct AnalysisStatistics;

impl AnalysisStatistics {
    /// Compute structural metrics of the topology graph.
    pub fn compute_topology_stats(
        adj: &[(String, String, u32, u64, u64)],
        num_services: usize,
    ) -> TopologyStats {
        if adj.is_empty() {
            return TopologyStats {
                service_count: num_services,
                ..Default::default()
            };
        }

        let services = extract_services(adj);
        let service_count = if num_services > 0 {
            num_services
        } else {
            services.len()
        };
        let edge_count = adj.len();

        // Fan-in / fan-out.
        let mut fan_in: HashMap<&str, usize> = HashMap::new();
        let mut fan_out: HashMap<&str, usize> = HashMap::new();
        let mut forward: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut backward: HashMap<&str, Vec<&str>> = HashMap::new();

        for (s, d, _, _, _) in adj {
            *fan_out.entry(s.as_str()).or_insert(0) += 1;
            *fan_in.entry(d.as_str()).or_insert(0) += 1;
            forward.entry(s.as_str()).or_default().push(d.as_str());
            backward.entry(d.as_str()).or_default().push(s.as_str());
        }

        let avg_fan_in = if service_count > 0 {
            fan_in.values().sum::<usize>() as f64 / service_count as f64
        } else {
            0.0
        };
        let avg_fan_out = if service_count > 0 {
            fan_out.values().sum::<usize>() as f64 / service_count as f64
        } else {
            0.0
        };

        // Diameter: longest shortest path between any pair via BFS.
        let diameter = compute_diameter(&services, &forward);

        // Max depth: longest path from any root (node with no incoming edges).
        let roots: Vec<&str> = services
            .iter()
            .filter(|s| !fan_in.contains_key(s.as_str()) || fan_in[s.as_str()] == 0)
            .map(|s| s.as_str())
            .collect();
        let max_depth = compute_max_depth(&roots, &forward);

        // Max amplification across all edges.
        let max_amplification = adj
            .iter()
            .map(|(_, _, r, _, _)| (1 + *r) as f64)
            .fold(1.0_f64, f64::max);

        // Max timeout chain (sum of all edge timeouts weighted by retries).
        let max_timeout_chain_ms = compute_max_timeout_chain(&roots, adj, &forward);

        TopologyStats {
            service_count,
            edge_count,
            max_depth,
            diameter,
            avg_fan_in,
            avg_fan_out,
            max_amplification,
            max_timeout_chain_ms,
        }
    }

    /// Derive a severity-distribution risk summary from Tier 1 results.
    pub fn compute_risk_summary(tier1: &Tier1Result) -> RiskSummary {
        let mut critical = 0usize;
        let mut high = 0usize;
        let mut medium = 0usize;
        let mut low = 0usize;
        let mut total_score = 0.0_f64;

        for r in &tier1.risky_paths {
            match r.severity.as_str() {
                "critical" => {
                    critical += 1;
                    total_score += 10.0;
                }
                "high" => {
                    high += 1;
                    total_score += 7.0;
                }
                "medium" => {
                    medium += 1;
                    total_score += 4.0;
                }
                _ => {
                    low += 1;
                    total_score += 1.0;
                }
            }
        }

        // Timeout violations are at least "medium".
        for v in &tier1.timeout_violations {
            if v.excess_ms > 10_000 {
                critical += 1;
                total_score += 10.0;
            } else if v.excess_ms > 5_000 {
                high += 1;
                total_score += 7.0;
            } else {
                medium += 1;
                total_score += 4.0;
            }
        }

        // Fan-in risks.
        for f in &tier1.fan_in_risks {
            if f.combined_amplification > 100.0 {
                critical += 1;
                total_score += 10.0;
            } else if f.combined_amplification > 50.0 {
                high += 1;
                total_score += 7.0;
            } else {
                medium += 1;
                total_score += 4.0;
            }
        }

        RiskSummary {
            critical_count: critical,
            high_count: high,
            medium_count: medium,
            low_count: low,
            total_risk_score: total_score,
        }
    }

    /// Per-edge amplification distribution.
    pub fn compute_amplification_distribution(
        adj: &[(String, String, u32, u64, u64)],
    ) -> Vec<(String, f64)> {
        let mut result: Vec<(String, f64)> = adj
            .iter()
            .map(|(s, d, r, _, _)| {
                let label = format!("{} → {}", s, d);
                let amp = (1 + *r) as f64;
                (label, amp)
            })
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Per-edge timeout distribution.
    pub fn compute_timeout_distribution(
        adj: &[(String, String, u32, u64, u64)],
    ) -> Vec<(String, u64)> {
        let mut result: Vec<(String, u64)> = adj
            .iter()
            .map(|(s, d, r, t, _)| {
                let label = format!("{} → {}", s, d);
                let total = *t * (1 + *r as u64);
                (label, total)
            })
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

// ---------------------------------------------------------------------------
// HistogramBuilder
// ---------------------------------------------------------------------------

/// Builds equi-width histograms from numeric data.
#[derive(Debug, Clone)]
pub struct HistogramBuilder {
    pub num_buckets: usize,
}

impl Default for HistogramBuilder {
    fn default() -> Self {
        Self { num_buckets: 10 }
    }
}

impl HistogramBuilder {
    pub fn new(num_buckets: usize) -> Self {
        Self {
            num_buckets: num_buckets.max(1),
        }
    }

    /// Build equi-width buckets and return `(bucket_label, count)` pairs.
    pub fn build(&self, values: &[f64]) -> Vec<(String, usize)> {
        if values.is_empty() {
            return Vec::new();
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < f64::EPSILON {
            return vec![(format!("{:.2}", min), values.len())];
        }

        let width = (max - min) / self.num_buckets as f64;
        let mut counts = vec![0usize; self.num_buckets];

        for &v in values {
            let idx = ((v - min) / width).floor() as usize;
            let idx = idx.min(self.num_buckets - 1);
            counts[idx] += 1;
        }

        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let lo = min + width * i as f64;
                let hi = lo + width;
                (format!("[{:.2}, {:.2})", lo, hi), c)
            })
            .collect()
    }

    /// Build a histogram from integer values.
    pub fn build_u64(&self, values: &[u64]) -> Vec<(String, usize)> {
        let floats: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        self.build(&floats)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn extract_services(adj: &[(String, String, u32, u64, u64)]) -> Vec<String> {
    let mut set = HashSet::new();
    for (s, d, _, _, _) in adj {
        set.insert(s.clone());
        set.insert(d.clone());
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

/// BFS diameter: longest shortest path between any pair of reachable nodes.
fn compute_diameter(services: &[String], forward: &HashMap<&str, Vec<&str>>) -> usize {
    let mut diameter = 0usize;
    for src in services {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<(&str, usize)> = VecDeque::new();
        visited.insert(src.as_str());
        queue.push_back((src.as_str(), 0));
        while let Some((node, dist)) = queue.pop_front() {
            diameter = diameter.max(dist);
            if let Some(neighbors) = forward.get(node) {
                for &n in neighbors {
                    if visited.insert(n) {
                        queue.push_back((n, dist + 1));
                    }
                }
            }
        }
    }
    diameter
}

/// Longest path from any root via DFS.
fn compute_max_depth(roots: &[&str], forward: &HashMap<&str, Vec<&str>>) -> usize {
    let mut max_depth = 0usize;
    for &root in roots {
        let depth = dfs_depth(root, forward, &mut HashSet::new());
        max_depth = max_depth.max(depth);
    }
    max_depth
}

fn dfs_depth(node: &str, forward: &HashMap<&str, Vec<&str>>, visited: &mut HashSet<String>) -> usize {
    if !visited.insert(node.to_string()) {
        return 0;
    }
    let mut max_child = 0usize;
    if let Some(neighbors) = forward.get(node) {
        for &n in neighbors {
            let d = dfs_depth(n, forward, visited);
            max_child = max_child.max(d);
        }
    }
    visited.remove(node);
    1 + max_child
}

/// Maximum timeout chain starting from any root (DFS, accumulating timeouts).
fn compute_max_timeout_chain(
    roots: &[&str],
    adj: &[(String, String, u32, u64, u64)],
    forward: &HashMap<&str, Vec<&str>>,
) -> u64 {
    let edge_map: HashMap<(&str, &str), (u32, u64)> = adj
        .iter()
        .map(|(s, d, r, t, _)| ((s.as_str(), d.as_str()), (*r, *t)))
        .collect();

    let mut max_chain = 0u64;
    for &root in roots {
        let chain = dfs_timeout(root, forward, &edge_map, &mut HashSet::new());
        max_chain = max_chain.max(chain);
    }
    max_chain
}

fn dfs_timeout(
    node: &str,
    forward: &HashMap<&str, Vec<&str>>,
    edge_map: &HashMap<(&str, &str), (u32, u64)>,
    visited: &mut HashSet<String>,
) -> u64 {
    if !visited.insert(node.to_string()) {
        return 0;
    }
    let mut max_child = 0u64;
    if let Some(neighbors) = forward.get(node) {
        for &n in neighbors {
            let edge_cost = edge_map
                .get(&(node, n))
                .map(|(r, t)| t * (1 + *r as u64))
                .unwrap_or(0);
            let child_chain = dfs_timeout(n, forward, edge_map, visited);
            max_child = max_child.max(edge_cost + child_chain);
        }
    }
    visited.remove(node);
    max_child
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier1::{AmplificationRisk, TimeoutViolation};

    fn chain_adj() -> Vec<(String, String, u32, u64, u64)> {
        vec![
            ("A".into(), "B".into(), 2, 1000, 1),
            ("B".into(), "C".into(), 2, 1000, 1),
            ("C".into(), "D".into(), 2, 1000, 1),
        ]
    }

    fn diamond_adj() -> Vec<(String, String, u32, u64, u64)> {
        vec![
            ("A".into(), "B".into(), 1, 500, 1),
            ("A".into(), "C".into(), 1, 500, 1),
            ("B".into(), "D".into(), 1, 500, 1),
            ("C".into(), "D".into(), 1, 500, 1),
        ]
    }

    #[test]
    fn test_topology_stats_chain() {
        let stats = AnalysisStatistics::compute_topology_stats(&chain_adj(), 4);
        assert_eq!(stats.service_count, 4);
        assert_eq!(stats.edge_count, 3);
        assert_eq!(stats.max_depth, 4); // A(1)->B(2)->C(3)->D(4)
        assert_eq!(stats.diameter, 3);
    }

    #[test]
    fn test_topology_stats_diamond() {
        let stats = AnalysisStatistics::compute_topology_stats(&diamond_adj(), 4);
        assert_eq!(stats.service_count, 4);
        assert_eq!(stats.edge_count, 4);
        assert!(stats.avg_fan_out > 0.0);
        assert!(stats.avg_fan_in > 0.0);
    }

    #[test]
    fn test_topology_stats_empty() {
        let stats = AnalysisStatistics::compute_topology_stats(&[], 0);
        assert_eq!(stats.service_count, 0);
        assert_eq!(stats.edge_count, 0);
    }

    #[test]
    fn test_risk_summary_counts() {
        let tier1 = Tier1Result {
            risky_paths: vec![
                AmplificationRisk {
                    path: vec!["A".into()],
                    amplification_factor: 200.0,
                    capacity: 10,
                    severity: "critical".into(),
                },
                AmplificationRisk {
                    path: vec!["B".into()],
                    amplification_factor: 60.0,
                    capacity: 100,
                    severity: "high".into(),
                },
            ],
            timeout_violations: vec![TimeoutViolation {
                path: vec!["A".into(), "B".into()],
                total_timeout_ms: 20_000,
                deadline_ms: 5_000,
                excess_ms: 15_000,
            }],
            fan_in_risks: vec![],
            duration_ms: 0,
        };
        let summary = AnalysisStatistics::compute_risk_summary(&tier1);
        assert_eq!(summary.critical_count, 2); // 1 from risky_paths, 1 from timeout (excess > 10k)
        assert_eq!(summary.high_count, 1);
        assert!(summary.total_risk_score > 0.0);
    }

    #[test]
    fn test_risk_summary_empty() {
        let tier1 = Tier1Result {
            risky_paths: vec![],
            timeout_violations: vec![],
            fan_in_risks: vec![],
            duration_ms: 0,
        };
        let summary = AnalysisStatistics::compute_risk_summary(&tier1);
        assert_eq!(summary.critical_count, 0);
        assert_eq!(summary.total_risk_score, 0.0);
    }

    #[test]
    fn test_amplification_distribution() {
        let dist = AnalysisStatistics::compute_amplification_distribution(&chain_adj());
        assert_eq!(dist.len(), 3);
        assert!((dist[0].1 - 3.0).abs() < 0.001); // (1+2) = 3 for each edge
    }

    #[test]
    fn test_timeout_distribution() {
        let dist = AnalysisStatistics::compute_timeout_distribution(&chain_adj());
        assert_eq!(dist.len(), 3);
        // Each: 1000 * (1+2) = 3000
        assert_eq!(dist[0].1, 3000);
    }

    #[test]
    fn test_histogram_builder_basic() {
        let hb = HistogramBuilder::new(3);
        let buckets = hb.build(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(buckets.len(), 3);
        let total: usize = buckets.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_histogram_builder_empty() {
        let hb = HistogramBuilder::new(5);
        let buckets = hb.build(&[]);
        assert!(buckets.is_empty());
    }

    #[test]
    fn test_histogram_builder_single_value() {
        let hb = HistogramBuilder::new(5);
        let buckets = hb.build(&[42.0, 42.0, 42.0]);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].1, 3);
    }

    #[test]
    fn test_histogram_builder_u64() {
        let hb = HistogramBuilder::new(2);
        let buckets = hb.build_u64(&[10, 20, 30, 40]);
        assert_eq!(buckets.len(), 2);
    }

    #[test]
    fn test_max_timeout_chain() {
        let stats = AnalysisStatistics::compute_topology_stats(&chain_adj(), 4);
        // Chain: 3 edges * 1000 * 3 = 9000
        assert_eq!(stats.max_timeout_chain_ms, 9000);
    }

    #[test]
    fn test_max_amplification() {
        let adj = vec![
            ("A".into(), "B".into(), 4, 100, 1),
            ("B".into(), "C".into(), 1, 100, 1),
        ];
        let stats = AnalysisStatistics::compute_topology_stats(&adj, 3);
        assert!((stats.max_amplification - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_performance_metrics_default() {
        let pm = PerformanceMetrics::default();
        assert_eq!(pm.analysis_time_ms, 0);
        assert_eq!(pm.memory_estimate_bytes, 0);
    }
}
