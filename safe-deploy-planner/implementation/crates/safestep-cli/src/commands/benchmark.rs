//! Implementation of the `benchmark` subcommand.
//!
//! Generates synthetic service dependency graphs with configurable topologies,
//! runs simulated deployment planning iterations, collects timing statistics,
//! and optionally compares against saved baseline results.

use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cli::{BenchmarkArgs, TopologyType};
use crate::commands::CommandExecutor;
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A synthetic service dependency graph used for benchmarking.
#[derive(Debug, Clone)]
pub struct SyntheticGraph {
    pub services: usize,
    pub versions: usize,
    pub topology: TopologyType,
    pub edges: Vec<(usize, usize)>,
    pub total_states: u64,
    pub total_edges: u64,
}

/// Timing and metric results from a single benchmark iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationResult {
    pub iteration: usize,
    pub graph_build_ms: u64,
    pub plan_find_ms: u64,
    pub envelope_ms: u64,
    pub total_ms: u64,
    pub plan_steps: usize,
    pub states_explored: u64,
    pub memory_bytes: u64,
}

/// Aggregate results of a full benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub services: usize,
    pub versions: usize,
    pub topology: String,
    pub iterations: usize,
    pub results: Vec<IterationResult>,
    pub summary: BenchmarkSummary,
    pub baseline_comparison: Option<BaselineComparison>,
}

/// Statistical summary over all iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub min_ms: u64,
    pub max_ms: u64,
    pub std_dev_ms: f64,
    pub mean_steps: f64,
    pub total_states: u64,
}

/// Comparison of current results against a previously saved baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_mean_ms: f64,
    pub current_mean_ms: f64,
    pub speedup: f64,
    pub regression: bool,
    pub details: Vec<String>,
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

/// The benchmark subcommand executor.
pub struct BenchmarkCommand {
    args: BenchmarkArgs,
}

impl BenchmarkCommand {
    pub fn new(args: BenchmarkArgs) -> Self {
        Self { args }
    }
}

impl CommandExecutor for BenchmarkCommand {
    fn execute(&self, output: &mut OutputManager) -> Result<()> {
        let args = &self.args;
        info!(
            services = args.services,
            versions = args.versions,
            topology = %args.topology,
            iterations = args.iterations,
            "Starting benchmark"
        );

        // --- Configuration section ---
        output.section("Benchmark Configuration");
        output.writeln(&format!("  Services:   {}", args.services));
        output.writeln(&format!("  Versions:   {}", args.versions));
        output.writeln(&format!("  Topology:   {}", args.topology));
        output.writeln(&format!("  Iterations: {}", args.iterations));
        output.blank_line();

        let graph = generate_graph(args.services, args.versions, args.topology);
        output.writeln(&format!(
            "  Graph edges: {}  |  State space: {}",
            graph.total_edges, graph.total_states
        ));
        output.blank_line();

        // --- Run iterations ---
        output.section("Iteration Results");

        let mut iteration_results: Vec<IterationResult> = Vec::with_capacity(args.iterations);

        for i in 0..args.iterations {
            let iter_start = Instant::now();

            // Phase 1 – graph build (adjacency + compatibility matrix)
            let build_start = Instant::now();
            let seed = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            let compat = generate_compatibility_matrix(args.services, args.versions, seed);
            let _adj = build_adjacency_list(&graph);
            let graph_build_ms = build_start.elapsed().as_millis() as u64;

            // Phase 2 – plan finding (BFS)
            let plan_start = Instant::now();
            let (plan_steps, states_explored) =
                simple_bfs_plan(&graph, &compat, args.versions);
            let plan_find_ms = plan_start.elapsed().as_millis() as u64;

            // Phase 3 – envelope computation (reverse BFS)
            let env_start = Instant::now();
            let _envelope_size =
                compute_envelope(&graph, &compat, args.versions, plan_steps);
            let envelope_ms = env_start.elapsed().as_millis() as u64;

            let total_ms = iter_start.elapsed().as_millis() as u64;

            let memory_bytes = estimate_memory_usage(args.services, args.versions, &graph);

            let result = IterationResult {
                iteration: i + 1,
                graph_build_ms,
                plan_find_ms,
                envelope_ms,
                total_ms,
                plan_steps,
                states_explored,
                memory_bytes,
            };

            output.writeln(&format!(
                "  [iter {:>3}] build={} plan={} envelope={} total={} steps={} explored={} mem={}",
                result.iteration,
                format_duration(result.graph_build_ms),
                format_duration(result.plan_find_ms),
                format_duration(result.envelope_ms),
                format_duration(result.total_ms),
                result.plan_steps,
                result.states_explored,
                format_bytes(result.memory_bytes),
            ));

            iteration_results.push(result);
        }

        output.blank_line();

        // --- Summary ---
        let summary = if iteration_results.is_empty() {
            BenchmarkSummary {
                mean_ms: 0.0,
                median_ms: 0.0,
                p95_ms: 0.0,
                min_ms: 0,
                max_ms: 0,
                std_dev_ms: 0.0,
                mean_steps: 0.0,
                total_states: graph.total_states,
            }
        } else {
            let times: Vec<u64> = iteration_results.iter().map(|r| r.total_ms).collect();
            let steps: Vec<usize> = iteration_results.iter().map(|r| r.plan_steps).collect();
            let mut s = compute_statistics(&times);
            s.total_states = graph.total_states;
            s.mean_steps =
                steps.iter().copied().sum::<usize>() as f64 / steps.len() as f64;
            s
        };

        output.section("Summary");
        let summary_headers: &[&str] = &[
            "Metric", "Value",
        ];
        let summary_rows: Vec<Vec<String>> = vec![
            vec!["Mean".into(), format_duration_f64(summary.mean_ms)],
            vec!["Median".into(), format_duration_f64(summary.median_ms)],
            vec!["P95".into(), format_duration_f64(summary.p95_ms)],
            vec!["Min".into(), format_duration(summary.min_ms)],
            vec!["Max".into(), format_duration(summary.max_ms)],
            vec!["Std Dev".into(), format_duration_f64(summary.std_dev_ms)],
            vec!["Mean Steps".into(), format!("{:.1}", summary.mean_steps)],
            vec!["State Space".into(), summary.total_states.to_string()],
        ];
        output.render_table(summary_headers, &summary_rows);
        output.blank_line();

        // --- Baseline comparison ---
        let baseline_comparison = if let Some(ref baseline_path) = args.baseline {
            let cmp = load_and_compare_baseline(baseline_path, &summary)?;
            output.section("Baseline Comparison");
            output.writeln(&format!(
                "  Baseline mean: {}",
                format_duration_f64(cmp.baseline_mean_ms)
            ));
            output.writeln(&format!(
                "  Current mean:  {}",
                format_duration_f64(cmp.current_mean_ms)
            ));
            output.writeln(&format!("  Speedup:       {:.2}x", cmp.speedup));
            if cmp.regression {
                let colors = output.colors().clone();
                output.writeln(&format!(
                    "  {}",
                    colors.error("REGRESSION DETECTED (>10% slower)")
                ));
            } else {
                let colors = output.colors().clone();
                output.writeln(&format!("  {}", colors.safe("No regression detected")));
            }
            for detail in &cmp.details {
                output.writeln(&format!("  • {detail}"));
            }
            output.blank_line();
            Some(cmp)
        } else {
            None
        };

        // --- Build final result ---
        let bench_result = BenchmarkResult {
            services: args.services,
            versions: args.versions,
            topology: args.topology.to_string(),
            iterations: args.iterations,
            results: iteration_results,
            summary,
            baseline_comparison,
        };

        // --- Save baseline ---
        if let Some(ref save_path) = args.save_baseline {
            let json = serde_json::to_string_pretty(&bench_result)
                .context("Failed to serialize benchmark result")?;
            std::fs::write(save_path, &json)
                .with_context(|| format!("Failed to write baseline to {}", save_path.display()))?;
            output.writeln(&format!("  Baseline saved to {}", save_path.display()));
            output.blank_line();
        }

        // Render as structured value for JSON/YAML formats.
        output.render_value(&bench_result)?;

        info!("Benchmark complete");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Graph generation
// ---------------------------------------------------------------------------

/// Generate a synthetic dependency graph with the requested topology.
pub fn generate_graph(
    services: usize,
    versions: usize,
    topology: TopologyType,
) -> SyntheticGraph {
    let edges = match topology {
        TopologyType::Mesh => generate_mesh_edges(services),
        TopologyType::HubSpoke => generate_hub_spoke_edges(services),
        TopologyType::Chain => generate_chain_edges(services),
        TopologyType::Hierarchical => generate_hierarchical_edges(services),
        TopologyType::Random => generate_random_edges(services),
    };

    let total_states = if services == 0 {
        0
    } else {
        (versions as u64).saturating_pow(services as u32)
    };

    let total_edges = edges.len() as u64;

    SyntheticGraph {
        services,
        versions,
        topology,
        edges,
        total_states,
        total_edges,
    }
}

fn generate_mesh_edges(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n * (n.saturating_sub(1)) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j));
        }
    }
    edges
}

fn generate_hub_spoke_edges(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        edges.push((0, i));
    }
    edges
}

fn generate_chain_edges(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n.saturating_sub(1));
    for i in 0..n.saturating_sub(1) {
        edges.push((i, i + 1));
    }
    edges
}

fn generate_hierarchical_edges(n: usize) -> Vec<(usize, usize)> {
    // Binary-ish tree: each parent i has children at 2i+1 and 2i+2 (+ optional 2i+3
    // when n is large enough to give a ternary branch occasionally).
    let mut edges = Vec::new();
    for i in 0..n {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < n {
            edges.push((i, left));
        }
        if right < n {
            edges.push((i, right));
        }
        // Occasional third child for larger graphs.
        let third = 3 * i + 3;
        if n > 10 && third < n && third != left && third != right {
            edges.push((i, third));
        }
    }
    edges
}

fn generate_random_edges(n: usize) -> Vec<(usize, usize)> {
    // Deterministic pseudo-random using LCG seeded on service count.
    let mut edges = Vec::new();
    let mut rng_state: u64 = (n as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    for i in 0..n {
        for j in (i + 1)..n {
            rng_state = lcg_next(rng_state);
            // ~60% inclusion probability
            if (rng_state >> 33) % 100 < 60 {
                edges.push((i, j));
            }
        }
    }
    edges
}

/// Simple LCG: Knuth's constants.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Build an adjacency list from the edge list.
fn build_adjacency_list(graph: &SyntheticGraph) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); graph.services];
    for &(u, v) in &graph.edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    adj
}

// ---------------------------------------------------------------------------
// Compatibility matrix
// ---------------------------------------------------------------------------

/// Generate a deterministic pseudo-random compatibility matrix.
///
/// Returns `compat[edge_idx][vi][vj]` — whether version `vi` of the source
/// service is compatible with version `vj` of the target service for the given
/// edge.
pub fn generate_compatibility_matrix(
    services: usize,
    versions: usize,
    seed: u64,
) -> Vec<Vec<Vec<bool>>> {
    let max_edges = services * (services.saturating_sub(1)) / 2;
    let mut matrix = Vec::with_capacity(max_edges);
    let mut state = seed;

    for _ in 0..max_edges {
        let mut edge_compat = Vec::with_capacity(versions);
        for _ in 0..versions {
            let mut row = Vec::with_capacity(versions);
            for _ in 0..versions {
                state = lcg_next(state);
                // ~70% compatible to keep plans reachable
                row.push((state >> 33) % 100 < 70);
            }
            edge_compat.push(row);
        }
        matrix.push(edge_compat);
    }
    matrix
}

// ---------------------------------------------------------------------------
// BFS plan finding
// ---------------------------------------------------------------------------

/// Run a BFS across the version-vector state space to find a deployment plan
/// from `(0, 0, …, 0)` to `(versions-1, versions-1, …, versions-1)`.
///
/// Returns `(plan_steps, states_explored)`. If no path is found the step count
/// is 0 with the number of explored states.
pub fn simple_bfs_plan(
    graph: &SyntheticGraph,
    compat: &[Vec<Vec<bool>>],
    versions: usize,
) -> (usize, u64) {
    let n = graph.services;
    if n == 0 || versions == 0 {
        return (0, 0);
    }

    // Encode state as a single u64 in mixed-radix (version per service).
    let encode = |state: &[usize]| -> u64 {
        let mut code: u64 = 0;
        for &v in state.iter().rev() {
            code = code * (versions as u64) + v as u64;
        }
        code
    };

    let start = vec![0usize; n];
    let goal: Vec<usize> = vec![versions - 1; n];

    let start_code = encode(&start);
    let goal_code = encode(&goal);

    if start_code == goal_code {
        return (0, 1);
    }

    // Cap exploration to avoid unbounded memory on large inputs.
    let max_explore: u64 = 200_000;

    let mut visited = std::collections::HashSet::new();
    visited.insert(start_code);

    let mut queue: VecDeque<(Vec<usize>, usize)> = VecDeque::new();
    queue.push_back((start, 0));

    let edge_index = build_edge_index(graph);

    let mut explored: u64 = 0;

    while let Some((current, depth)) = queue.pop_front() {
        explored += 1;
        if explored >= max_explore {
            break;
        }

        // Try upgrading each service by one version.
        for svc in 0..n {
            if current[svc] + 1 >= versions {
                continue;
            }
            let mut next = current.clone();
            next[svc] += 1;

            // Check compatibility along all edges incident to this service.
            let compatible = check_compatibility(svc, &next, &edge_index, compat, n);

            if !compatible {
                continue;
            }

            let code = encode(&next);
            if visited.contains(&code) {
                continue;
            }
            visited.insert(code);

            if code == goal_code {
                return (depth + 1, explored);
            }
            queue.push_back((next, depth + 1));
        }
    }

    (0, explored)
}

/// Map from `(min(u,v), max(u,v))` to edge index in the compatibility matrix.
fn build_edge_index(graph: &SyntheticGraph) -> std::collections::HashMap<(usize, usize), usize> {
    let mut map = std::collections::HashMap::new();
    for (idx, &(u, v)) in graph.edges.iter().enumerate() {
        let key = if u < v { (u, v) } else { (v, u) };
        map.insert(key, idx);
    }
    map
}

/// Check that a state is compatible across all edges involving the given service.
fn check_compatibility(
    svc: usize,
    state: &[usize],
    edge_index: &std::collections::HashMap<(usize, usize), usize>,
    compat: &[Vec<Vec<bool>>],
    n: usize,
) -> bool {
    for other in 0..n {
        if other == svc {
            continue;
        }
        let key = if svc < other {
            (svc, other)
        } else {
            (other, svc)
        };
        if let Some(&idx) = edge_index.get(&key) {
            if idx < compat.len() {
                let vi = state[svc];
                let vj = state[other];
                if vi < compat[idx].len() && vj < compat[idx][vi].len() {
                    if !compat[idx][vi][vj] {
                        return false;
                    }
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Envelope computation (reverse BFS)
// ---------------------------------------------------------------------------

/// Simulate computing the rollback safety envelope by performing a reverse BFS
/// from each step along the plan path back toward the start state.
fn compute_envelope(
    graph: &SyntheticGraph,
    compat: &[Vec<Vec<bool>>],
    versions: usize,
    plan_steps: usize,
) -> u64 {
    let n = graph.services;
    if n == 0 || versions == 0 || plan_steps == 0 {
        return 0;
    }

    let edge_index = build_edge_index(graph);

    // Simulate a plan path: each step upgrades one service in round-robin order.
    let mut path_states: Vec<Vec<usize>> = Vec::with_capacity(plan_steps + 1);
    let mut current = vec![0usize; n];
    path_states.push(current.clone());
    for step in 0..plan_steps {
        let svc = step % n;
        if current[svc] + 1 < versions {
            current[svc] += 1;
        }
        path_states.push(current.clone());
    }

    let encode = |state: &[usize]| -> u64 {
        let mut code: u64 = 0;
        for &v in state.iter().rev() {
            code = code * (versions as u64) + v as u64;
        }
        code
    };

    let max_explore: u64 = 50_000;
    let mut total_envelope: u64 = 0;

    for step_state in &path_states {
        let mut visited = std::collections::HashSet::new();
        let start_code = encode(step_state);
        visited.insert(start_code);

        let mut queue: VecDeque<Vec<usize>> = VecDeque::new();
        queue.push_back(step_state.clone());

        let mut local_explored: u64 = 0;

        while let Some(cur) = queue.pop_front() {
            local_explored += 1;
            if local_explored >= max_explore {
                break;
            }

            // Try downgrading each service by one version (reverse moves).
            for svc in 0..n {
                if cur[svc] == 0 {
                    continue;
                }
                let mut prev = cur.clone();
                prev[svc] -= 1;

                let ok = check_compatibility(svc, &prev, &edge_index, compat, n);
                if !ok {
                    continue;
                }

                let code = encode(&prev);
                if visited.contains(&code) {
                    continue;
                }
                visited.insert(code);
                queue.push_back(prev);
            }
        }

        total_envelope += visited.len() as u64;
    }

    total_envelope
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute timing statistics over a slice of millisecond durations.
pub fn compute_statistics(times: &[u64]) -> BenchmarkSummary {
    if times.is_empty() {
        return BenchmarkSummary {
            mean_ms: 0.0,
            median_ms: 0.0,
            p95_ms: 0.0,
            min_ms: 0,
            max_ms: 0,
            std_dev_ms: 0.0,
            mean_steps: 0.0,
            total_states: 0,
        };
    }

    let n = times.len();
    let sum: u64 = times.iter().copied().sum();
    let mean = sum as f64 / n as f64;

    let mut sorted = times.to_vec();
    sorted.sort_unstable();

    let min_ms = sorted[0];
    let max_ms = sorted[n - 1];

    let median_ms = if n % 2 == 0 {
        (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
    } else {
        sorted[n / 2] as f64
    };

    let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    let p95_ms = sorted[p95_idx] as f64;

    let variance = times
        .iter()
        .map(|&t| {
            let diff = t as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let std_dev_ms = variance.sqrt();

    BenchmarkSummary {
        mean_ms: mean,
        median_ms,
        p95_ms,
        min_ms,
        max_ms,
        std_dev_ms,
        mean_steps: 0.0,
        total_states: 0,
    }
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

fn estimate_memory_usage(services: usize, versions: usize, graph: &SyntheticGraph) -> u64 {
    // Rough estimate: adjacency list + compatibility matrix + BFS visited set overhead.
    let adj_bytes = graph.edges.len() * 2 * std::mem::size_of::<usize>();
    let compat_bytes = graph.edges.len() * versions * versions;
    let visited_estimate = (services * versions * 64).min(200_000 * 8);
    (adj_bytes + compat_bytes + visited_estimate) as u64
}

// ---------------------------------------------------------------------------
// Baseline I/O
// ---------------------------------------------------------------------------

fn load_and_compare_baseline(
    path: &Path,
    current_summary: &BenchmarkSummary,
) -> Result<BaselineComparison> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read baseline from {}", path.display()))?;
    let baseline: BenchmarkResult = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse baseline JSON from {}", path.display()))?;

    let baseline_mean = baseline.summary.mean_ms;
    let current_mean = current_summary.mean_ms;

    let speedup = if current_mean > 0.0 {
        baseline_mean / current_mean
    } else {
        f64::INFINITY
    };

    let regression = current_mean > baseline_mean * 1.10;

    let mut details = Vec::new();

    let pct_change = if baseline_mean > 0.0 {
        ((current_mean - baseline_mean) / baseline_mean) * 100.0
    } else {
        0.0
    };

    if pct_change > 0.0 {
        details.push(format!("Current is {:.1}% slower than baseline", pct_change));
    } else if pct_change < 0.0 {
        details.push(format!(
            "Current is {:.1}% faster than baseline",
            pct_change.abs()
        ));
    } else {
        details.push("Identical to baseline".into());
    }

    if current_summary.p95_ms > baseline.summary.p95_ms * 1.2 {
        details.push(format!(
            "P95 increased: {} → {}",
            format_duration_f64(baseline.summary.p95_ms),
            format_duration_f64(current_summary.p95_ms),
        ));
    }

    if current_summary.std_dev_ms > baseline.summary.std_dev_ms * 1.5 {
        details.push(format!(
            "Variance increased: stddev {} → {}",
            format_duration_f64(baseline.summary.std_dev_ms),
            format_duration_f64(current_summary.std_dev_ms),
        ));
    }

    Ok(BaselineComparison {
        baseline_mean_ms: baseline_mean,
        current_mean_ms: current_mean,
        speedup,
        regression,
        details,
    })
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a millisecond duration into a human-readable string.
pub fn format_duration(ms: u64) -> String {
    if ms == 0 {
        "0ms".into()
    } else if ms >= 1_000 {
        format!("{:.2}s", ms as f64 / 1_000.0)
    } else {
        format!("{}ms", ms)
    }
}

fn format_duration_f64(ms: f64) -> String {
    if ms < 0.001 {
        format!("{:.0}µs", ms * 1_000.0)
    } else if ms >= 1_000.0 {
        format!("{:.2}s", ms / 1_000.0)
    } else {
        format!("{:.2}ms", ms)
    }
}

/// Format a byte count into a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1_024;
    const MIB: u64 = 1_024 * 1_024;
    const GIB: u64 = 1_024 * 1_024 * 1_024;

    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::TopologyType;

    // --- Graph generation tests ---

    #[test]
    fn test_mesh_edges_count() {
        let g = generate_graph(5, 2, TopologyType::Mesh);
        assert_eq!(g.edges.len(), 10); // 5*4/2
        assert_eq!(g.total_edges, 10);
    }

    #[test]
    fn test_mesh_edges_unique() {
        let g = generate_graph(6, 2, TopologyType::Mesh);
        let mut set = std::collections::HashSet::new();
        for &(u, v) in &g.edges {
            assert!(u < v, "edges should be ordered u < v");
            assert!(set.insert((u, v)), "duplicate edge");
        }
    }

    #[test]
    fn test_hub_spoke_edges() {
        let g = generate_graph(5, 2, TopologyType::HubSpoke);
        assert_eq!(g.edges.len(), 4);
        for &(u, _v) in &g.edges {
            assert_eq!(u, 0, "hub should always be node 0");
        }
    }

    #[test]
    fn test_chain_edges() {
        let g = generate_graph(5, 2, TopologyType::Chain);
        assert_eq!(g.edges.len(), 4);
        for (i, &(u, v)) in g.edges.iter().enumerate() {
            assert_eq!(u, i);
            assert_eq!(v, i + 1);
        }
    }

    #[test]
    fn test_hierarchical_edges() {
        let g = generate_graph(7, 2, TopologyType::Hierarchical);
        // Binary tree with 7 nodes: 6 edges (each internal node has 2 children)
        assert!(!g.edges.is_empty());
        for &(u, v) in &g.edges {
            assert!(u < v, "parent index should be smaller");
        }
    }

    #[test]
    fn test_random_edges_deterministic() {
        let g1 = generate_graph(10, 2, TopologyType::Random);
        let g2 = generate_graph(10, 2, TopologyType::Random);
        assert_eq!(g1.edges, g2.edges, "random edges should be deterministic");
    }

    #[test]
    fn test_random_edges_approximately_sixty_percent() {
        let n = 20;
        let g = generate_graph(n, 2, TopologyType::Random);
        let max_edges = n * (n - 1) / 2;
        let ratio = g.edges.len() as f64 / max_edges as f64;
        assert!(
            (0.3..=0.85).contains(&ratio),
            "random edge ratio {ratio:.2} not near 60%"
        );
    }

    #[test]
    fn test_single_service_graph() {
        for topo in [
            TopologyType::Mesh,
            TopologyType::HubSpoke,
            TopologyType::Chain,
            TopologyType::Hierarchical,
            TopologyType::Random,
        ] {
            let g = generate_graph(1, 3, topo);
            assert!(g.edges.is_empty(), "single service should have no edges");
            assert_eq!(g.total_states, 3);
        }
    }

    #[test]
    fn test_zero_services_graph() {
        let g = generate_graph(0, 5, TopologyType::Mesh);
        assert!(g.edges.is_empty());
        assert_eq!(g.total_states, 0);
    }

    #[test]
    fn test_single_version_total_states() {
        let g = generate_graph(4, 1, TopologyType::Chain);
        assert_eq!(g.total_states, 1); // 1^4
    }

    #[test]
    fn test_total_states_computation() {
        let g = generate_graph(3, 4, TopologyType::Mesh);
        assert_eq!(g.total_states, 64); // 4^3
    }

    // --- Statistics tests ---

    #[test]
    fn test_statistics_basic() {
        let times = vec![10, 20, 30, 40, 50];
        let s = compute_statistics(&times);
        assert!((s.mean_ms - 30.0).abs() < 0.001);
        assert!((s.median_ms - 30.0).abs() < 0.001);
        assert_eq!(s.min_ms, 10);
        assert_eq!(s.max_ms, 50);
    }

    #[test]
    fn test_statistics_single_value() {
        let times = vec![42];
        let s = compute_statistics(&times);
        assert!((s.mean_ms - 42.0).abs() < 0.001);
        assert!((s.median_ms - 42.0).abs() < 0.001);
        assert_eq!(s.min_ms, 42);
        assert_eq!(s.max_ms, 42);
        assert!((s.std_dev_ms - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_statistics_even_count_median() {
        let times = vec![10, 20, 30, 40];
        let s = compute_statistics(&times);
        assert!((s.median_ms - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_statistics_empty() {
        let s = compute_statistics(&[]);
        assert!((s.mean_ms - 0.0).abs() < 0.001);
        assert_eq!(s.min_ms, 0);
    }

    #[test]
    fn test_statistics_std_dev() {
        let times = vec![2, 4, 4, 4, 5, 5, 7, 9];
        let s = compute_statistics(&times);
        // mean = 5.0, variance = 4.0, std_dev = 2.0
        assert!((s.mean_ms - 5.0).abs() < 0.001);
        assert!((s.std_dev_ms - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_statistics_p95() {
        let times: Vec<u64> = (1..=100).collect();
        let s = compute_statistics(&times);
        assert!((s.p95_ms - 95.0).abs() < 1.001);
    }

    // --- Compatibility matrix tests ---

    #[test]
    fn test_compat_matrix_dimensions() {
        let m = generate_compatibility_matrix(4, 3, 42);
        // 4 services → max edges = 4*3/2 = 6
        assert_eq!(m.len(), 6);
        for edge in &m {
            assert_eq!(edge.len(), 3);
            for row in edge {
                assert_eq!(row.len(), 3);
            }
        }
    }

    #[test]
    fn test_compat_matrix_deterministic() {
        let m1 = generate_compatibility_matrix(5, 4, 123);
        let m2 = generate_compatibility_matrix(5, 4, 123);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_compat_matrix_different_seeds() {
        let m1 = generate_compatibility_matrix(5, 4, 1);
        let m2 = generate_compatibility_matrix(5, 4, 2);
        assert_ne!(m1, m2, "different seeds should produce different matrices");
    }

    // --- Format helper tests ---

    #[test]
    fn test_format_duration_zero() {
        assert_eq!(format_duration(0), "0ms");
    }

    #[test]
    fn test_format_duration_ms() {
        assert_eq!(format_duration(456), "456ms");
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(1230), "1.23s");
    }

    #[test]
    fn test_format_bytes_b() {
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn test_format_bytes_kib() {
        assert_eq!(format_bytes(1_024 * 256), "256.0 KiB");
    }

    #[test]
    fn test_format_bytes_mib() {
        let result = format_bytes(1_024 * 1_024 + 1_024 * 200);
        assert!(result.contains("MiB"));
    }

    #[test]
    fn test_format_bytes_gib() {
        let result = format_bytes(2 * 1_024 * 1_024 * 1_024);
        assert_eq!(result, "2.0 GiB");
    }

    // --- BFS plan tests ---

    #[test]
    fn test_bfs_trivial_single_service_single_version() {
        let g = generate_graph(1, 1, TopologyType::Chain);
        let compat = generate_compatibility_matrix(1, 1, 0);
        let (steps, explored) = simple_bfs_plan(&g, &compat, 1);
        assert_eq!(steps, 0);
        assert_eq!(explored, 1);
    }

    #[test]
    fn test_bfs_chain_two_services() {
        let g = generate_graph(2, 3, TopologyType::Chain);
        // All compatible
        let compat = vec![vec![vec![true; 3]; 3]];
        let (steps, explored) = simple_bfs_plan(&g, &compat, 3);
        assert!(steps > 0, "should find a plan");
        assert!(explored > 0);
    }

    #[test]
    fn test_bfs_zero_services() {
        let g = generate_graph(0, 5, TopologyType::Mesh);
        let compat: Vec<Vec<Vec<bool>>> = vec![];
        let (steps, explored) = simple_bfs_plan(&g, &compat, 5);
        assert_eq!(steps, 0);
        assert_eq!(explored, 0);
    }

    #[test]
    fn test_bfs_zero_versions() {
        let g = generate_graph(3, 0, TopologyType::Mesh);
        let compat: Vec<Vec<Vec<bool>>> = vec![];
        let (steps, explored) = simple_bfs_plan(&g, &compat, 0);
        assert_eq!(steps, 0);
        assert_eq!(explored, 0);
    }

    // --- Baseline comparison tests ---

    #[test]
    fn test_baseline_regression_detected() {
        let baseline = BenchmarkResult {
            services: 5,
            versions: 3,
            topology: "mesh".into(),
            iterations: 5,
            results: vec![],
            summary: BenchmarkSummary {
                mean_ms: 100.0,
                median_ms: 100.0,
                p95_ms: 120.0,
                min_ms: 90,
                max_ms: 130,
                std_dev_ms: 10.0,
                mean_steps: 5.0,
                total_states: 243,
            },
            baseline_comparison: None,
        };

        let tmpdir = std::env::temp_dir().join("safestep_test_baseline_reg");
        let _ = std::fs::create_dir_all(&tmpdir);
        let path = tmpdir.join("baseline.json");
        std::fs::write(&path, serde_json::to_string_pretty(&baseline).unwrap()).unwrap();

        let current = BenchmarkSummary {
            mean_ms: 150.0, // 50% slower → regression
            median_ms: 140.0,
            p95_ms: 180.0,
            min_ms: 120,
            max_ms: 200,
            std_dev_ms: 20.0,
            mean_steps: 6.0,
            total_states: 243,
        };

        let cmp = load_and_compare_baseline(&path, &current).unwrap();
        assert!(cmp.regression);
        assert!(cmp.speedup < 1.0);
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_baseline_no_regression() {
        let baseline = BenchmarkResult {
            services: 5,
            versions: 3,
            topology: "chain".into(),
            iterations: 3,
            results: vec![],
            summary: BenchmarkSummary {
                mean_ms: 100.0,
                median_ms: 100.0,
                p95_ms: 110.0,
                min_ms: 90,
                max_ms: 120,
                std_dev_ms: 8.0,
                mean_steps: 5.0,
                total_states: 243,
            },
            baseline_comparison: None,
        };

        let tmpdir = std::env::temp_dir().join("safestep_test_baseline_ok");
        let _ = std::fs::create_dir_all(&tmpdir);
        let path = tmpdir.join("baseline.json");
        std::fs::write(&path, serde_json::to_string_pretty(&baseline).unwrap()).unwrap();

        let current = BenchmarkSummary {
            mean_ms: 80.0, // faster
            median_ms: 78.0,
            p95_ms: 95.0,
            min_ms: 70,
            max_ms: 100,
            std_dev_ms: 6.0,
            mean_steps: 4.5,
            total_states: 243,
        };

        let cmp = load_and_compare_baseline(&path, &current).unwrap();
        assert!(!cmp.regression);
        assert!(cmp.speedup > 1.0);
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    // --- Edge index tests ---

    #[test]
    fn test_edge_index_consistency() {
        let g = generate_graph(5, 2, TopologyType::Mesh);
        let idx = build_edge_index(&g);
        assert_eq!(idx.len(), g.edges.len());
        for (i, &(u, v)) in g.edges.iter().enumerate() {
            let key = if u < v { (u, v) } else { (v, u) };
            assert_eq!(idx[&key], i);
        }
    }

    // --- Adjacency list tests ---

    #[test]
    fn test_adjacency_list_undirected() {
        let g = generate_graph(4, 2, TopologyType::Chain);
        let adj = build_adjacency_list(&g);
        assert_eq!(adj.len(), 4);
        assert!(adj[0].contains(&1));
        assert!(adj[1].contains(&0));
        assert!(adj[1].contains(&2));
        assert!(adj[2].contains(&1));
    }

    // --- Envelope test ---

    #[test]
    fn test_envelope_zero_steps() {
        let g = generate_graph(3, 3, TopologyType::Chain);
        let compat = generate_compatibility_matrix(3, 3, 99);
        let result = compute_envelope(&g, &compat, 3, 0);
        assert_eq!(result, 0);
    }
}
