//! Benchmarking utilities for measuring simulation throughput under
//! varying entity counts and scene configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::simulator::{HeadlessSimulator, SimulationConfig};

// ---------------------------------------------------------------------------
// Benchmark scene
// ---------------------------------------------------------------------------

/// A self-contained scene description used as input to a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScene {
    pub name: String,
    pub entity_count: usize,
    pub interaction_count: usize,
    pub entities: Vec<(String, [f64; 3])>,
    pub predicates: Vec<String>,
    pub expected_steps: u64,
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

/// The measured outcome of running a single benchmark scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub elapsed_ms: f64,
    pub steps_per_second: f64,
    pub memory_estimate_bytes: u64,
    pub entity_count: usize,
    pub passed: bool,
    pub errors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------

/// Controls the range and resolution of entity counts to sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkParams {
    pub min_entities: usize,
    pub max_entities: usize,
    pub entity_step: usize,
    pub duration: f64,
    pub time_step: f64,
}

impl Default for BenchmarkParams {
    fn default() -> Self {
        Self {
            min_entities: 4,
            max_entities: 64,
            entity_step: 4,
            duration: 1.0,
            time_step: 1.0 / 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark suite
// ---------------------------------------------------------------------------

/// A collection of benchmark scenes with shared parameters.
pub struct BenchmarkSuite {
    pub scenes: Vec<BenchmarkScene>,
    pub params: BenchmarkParams,
}

impl BenchmarkSuite {
    pub fn new(params: BenchmarkParams) -> Self {
        Self {
            scenes: Vec::new(),
            params,
        }
    }

    /// Auto-generate scenes with entity counts stepping from `min_entities`
    /// to `max_entities`.  Each scene is a grid layout.
    pub fn generate_benchmark_scenes(&mut self) {
        self.scenes.clear();
        let mut n = self.params.min_entities;
        while n <= self.params.max_entities {
            let scene = generate_grid_scene(n, 2.0);
            self.scenes.push(scene);
            n += self.params.entity_step;
        }
    }

    /// Run all scenes and collect results.
    pub fn run_all(&self) -> Vec<BenchmarkResult> {
        self.scenes
            .iter()
            .map(|scene| run_single_benchmark(scene, self.params.time_step, self.params.duration))
            .collect()
    }

    /// Add a custom scene.
    pub fn add_scene(&mut self, scene: BenchmarkScene) {
        self.scenes.push(scene);
    }

    /// Number of scenes in the suite.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }
}

// ---------------------------------------------------------------------------
// Scene generators
// ---------------------------------------------------------------------------

/// Create a square grid of `n × n` entities with the given spacing.
pub fn generate_grid_scene(n: usize, spacing: f64) -> BenchmarkScene {
    let total = n * n;
    let mut entities = Vec::with_capacity(total);
    for row in 0..n {
        for col in 0..n {
            let id = format!("g_{}_{}", row, col);
            let x = col as f64 * spacing;
            let z = row as f64 * spacing;
            entities.push((id, [x, 0.0, z]));
        }
    }

    // Interactions: each entity interacts with its immediate neighbours.
    let interactions = if total > 1 {
        (n * (n - 1) * 2).min(total * (total - 1) / 2)
    } else {
        0
    };

    // Generate some predicate names.
    let predicates: Vec<String> = (0..n.min(8))
        .map(|i| format!("bench_pred_{}", i))
        .collect();

    let expected_steps = (1.0 / (1.0 / 60.0)) as u64; // ~60 steps for 1s

    BenchmarkScene {
        name: format!("grid_{}x{}", n, n),
        entity_count: total,
        interaction_count: interactions,
        entities,
        predicates,
        expected_steps,
    }
}

/// Create `count` entities at pseudo-random positions inside `[-bound, bound]³`
/// using a deterministic LCG seeded with `seed`.
pub fn generate_random_scene(count: usize, bound: f64, seed: u64) -> BenchmarkScene {
    let mut rng = seed;
    let mut entities = Vec::with_capacity(count);
    for i in 0..count {
        let x = lcg_f64(&mut rng, -bound, bound);
        let y = lcg_f64(&mut rng, -bound, bound);
        let z = lcg_f64(&mut rng, -bound, bound);
        entities.push((format!("r_{}", i), [x, y, z]));
    }

    let interactions = count.saturating_sub(1);
    let predicates: Vec<String> = (0..count.min(8))
        .map(|i| format!("rand_pred_{}", i))
        .collect();

    BenchmarkScene {
        name: format!("random_{}", count),
        entity_count: count,
        interaction_count: interactions,
        entities,
        predicates,
        expected_steps: 60,
    }
}

/// Create `clusters` groups of `per_cluster` entities, each cluster centred
/// at a random position with intra-cluster `spread`.
pub fn generate_cluster_scene(
    clusters: usize,
    per_cluster: usize,
    spread: f64,
) -> BenchmarkScene {
    let total = clusters * per_cluster;
    let mut entities = Vec::with_capacity(total);
    let mut rng: u64 = 0xDEAD_BEEF;

    for c in 0..clusters {
        let cx = lcg_f64(&mut rng, -20.0, 20.0);
        let cy = lcg_f64(&mut rng, -20.0, 20.0);
        let cz = lcg_f64(&mut rng, -20.0, 20.0);
        for p in 0..per_cluster {
            let ox = lcg_f64(&mut rng, -spread, spread);
            let oy = lcg_f64(&mut rng, -spread, spread);
            let oz = lcg_f64(&mut rng, -spread, spread);
            let id = format!("c{}_{}", c, p);
            entities.push((id, [cx + ox, cy + oy, cz + oz]));
        }
    }

    // Intra-cluster interactions dominate.
    let interactions = clusters * per_cluster * (per_cluster.saturating_sub(1)) / 2;
    let predicates: Vec<String> = (0..clusters.min(8))
        .map(|i| format!("cluster_pred_{}", i))
        .collect();

    BenchmarkScene {
        name: format!("cluster_{}x{}", clusters, per_cluster),
        entity_count: total,
        interaction_count: interactions,
        entities,
        predicates,
        expected_steps: 60,
    }
}

/// Create a linear chain of `count` entities along the X axis.
pub fn generate_chain_scene(count: usize, spacing: f64) -> BenchmarkScene {
    let entities: Vec<(String, [f64; 3])> = (0..count)
        .map(|i| (format!("chain_{}", i), [i as f64 * spacing, 0.0, 0.0]))
        .collect();
    let interactions = count.saturating_sub(1);
    let predicates: Vec<String> = (0..count.min(8))
        .map(|i| format!("chain_pred_{}", i))
        .collect();

    BenchmarkScene {
        name: format!("chain_{}", count),
        entity_count: count,
        interaction_count: interactions,
        entities,
        predicates,
        expected_steps: 60,
    }
}

// ---------------------------------------------------------------------------
// Running a benchmark
// ---------------------------------------------------------------------------

/// Execute a single benchmark scene: create a simulator, add entities, step
/// for the given duration, and measure wall-clock time.
pub fn run_single_benchmark(
    scene: &BenchmarkScene,
    time_step: f64,
    duration: f64,
) -> BenchmarkResult {
    let cfg = SimulationConfig {
        time_step,
        max_duration: duration,
        random_seed: 42,
        headless: true,
    };
    let mut sim = HeadlessSimulator::new(cfg);

    for (id, pos) in &scene.entities {
        sim.add_entity(id, *pos);
    }

    // Add proximity predicates between the first few entity pairs.
    let ids: Vec<&str> = scene.entities.iter().map(|(id, _)| id.as_str()).collect();
    let pred_limit = scene.predicates.len().min(ids.len().saturating_sub(1));
    for i in 0..pred_limit {
        sim.add_proximity_predicate(
            &scene.predicates[i],
            ids[i],
            ids[i + 1],
            3.0,
        );
    }

    let start = Instant::now();
    let result = sim.run_for_duration(duration);
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

    let steps_per_second = if elapsed_ms > 0.0 {
        result.total_steps as f64 / (elapsed_ms / 1000.0)
    } else {
        0.0
    };

    // Rough memory estimate: entities + predicate map overhead.
    let mem = (scene.entity_count * 200 + scene.predicates.len() * 64) as u64;

    let passed = result.errors.is_empty();

    BenchmarkResult {
        name: scene.name.clone(),
        elapsed_ms,
        steps_per_second,
        memory_estimate_bytes: mem,
        entity_count: scene.entity_count,
        passed,
        errors: result.errors,
    }
}

// ---------------------------------------------------------------------------
// Statistical summary
// ---------------------------------------------------------------------------

/// Descriptive statistics for a set of f64 observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
}

/// Compute descriptive statistics from a slice of values.
///
/// Returns a zeroed summary if `values` is empty.
pub fn compute_statistics(values: &[f64]) -> StatisticalSummary {
    if values.is_empty() {
        return StatisticalSummary {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            p95: 0.0,
        };
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;

    let var: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std_dev = var.sqrt();

    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = *sorted.last().unwrap();
    let median = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let p95_idx = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];

    StatisticalSummary {
        mean,
        median,
        std_dev,
        min,
        max,
        p95,
    }
}

// ---------------------------------------------------------------------------
// Benchmark report
// ---------------------------------------------------------------------------

/// Aggregated benchmark report with per-scene results and a summary of
/// steps-per-second across all scenes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub results: Vec<BenchmarkResult>,
    pub summary: StatisticalSummary,
}

impl BenchmarkReport {
    /// Build a report from a set of benchmark results.
    pub fn generate(results: Vec<BenchmarkResult>) -> Self {
        let sps: Vec<f64> = results.iter().map(|r| r.steps_per_second).collect();
        let summary = compute_statistics(&sps);
        Self { results, summary }
    }

    /// Serialize the report to a pretty-printed JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Produce a human-readable table summarising each scene.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<30} {:>8} {:>12} {:>12} {:>8}\n",
            "Scene", "Entities", "Elapsed(ms)", "Steps/s", "Passed"
        ));
        out.push_str(&"-".repeat(74));
        out.push('\n');

        for r in &self.results {
            out.push_str(&format!(
                "{:<30} {:>8} {:>12.2} {:>12.0} {:>8}\n",
                r.name,
                r.entity_count,
                r.elapsed_ms,
                r.steps_per_second,
                if r.passed { "yes" } else { "NO" },
            ));
        }

        out.push_str(&"-".repeat(74));
        out.push('\n');
        out.push_str(&format!(
            "Summary — mean steps/s: {:.0}  median: {:.0}  p95: {:.0}  min: {:.0}  max: {:.0}\n",
            self.summary.mean,
            self.summary.median,
            self.summary.p95,
            self.summary.min,
            self.summary.max,
        ));
        out
    }

    /// Return the number of benchmark results in the report.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Return the number of scenes that failed.
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Compare two benchmark reports by matching scene names and returning
/// relative speed-ups.
pub fn compare_reports(
    baseline: &BenchmarkReport,
    current: &BenchmarkReport,
) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let base_by_name: HashMap<&str, &BenchmarkResult> =
        baseline.results.iter().map(|r| (r.name.as_str(), r)).collect();
    for r in &current.results {
        if let Some(base) = base_by_name.get(r.name.as_str()) {
            let speedup = if base.steps_per_second > 0.0 {
                r.steps_per_second / base.steps_per_second
            } else {
                0.0
            };
            map.insert(r.name.clone(), speedup);
        }
    }
    map
}

/// Determine whether any scene in a report regressed compared to a baseline
/// by more than `threshold` (e.g. 0.9 means 10% regression).
pub fn has_regression(
    baseline: &BenchmarkReport,
    current: &BenchmarkReport,
    threshold: f64,
) -> bool {
    let ratios = compare_reports(baseline, current);
    ratios.values().any(|&v| v < threshold)
}

// ---------------------------------------------------------------------------
// LCG pseudo-random helpers
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn lcg_f64(state: &mut u64, lo: f64, hi: f64) -> f64 {
    let raw = lcg_next(state);
    let t = (raw >> 11) as f64 / ((1u64 << 53) as f64);
    lo + t * (hi - lo)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // ---- scene generation --------------------------------------------------

    #[test]
    fn grid_scene_entity_count() {
        let scene = generate_grid_scene(4, 1.0);
        assert_eq!(scene.entity_count, 16);
        assert_eq!(scene.entities.len(), 16);
        assert!(scene.name.contains("4x4"));
    }

    #[test]
    fn grid_scene_positions_spaced() {
        let scene = generate_grid_scene(3, 2.0);
        // Last entity in the grid should be at (4.0, 0.0, 4.0).
        let last = &scene.entities[8];
        assert!(approx(last.1[0], 4.0, 1e-9));
        assert!(approx(last.1[2], 4.0, 1e-9));
    }

    #[test]
    fn random_scene_deterministic() {
        let s1 = generate_random_scene(10, 5.0, 42);
        let s2 = generate_random_scene(10, 5.0, 42);
        for i in 0..10 {
            assert_eq!(s1.entities[i].1, s2.entities[i].1);
        }
    }

    #[test]
    fn random_scene_count() {
        let scene = generate_random_scene(25, 10.0, 99);
        assert_eq!(scene.entity_count, 25);
        assert_eq!(scene.entities.len(), 25);
    }

    #[test]
    fn random_scene_within_bounds() {
        let scene = generate_random_scene(50, 5.0, 7);
        for (_, pos) in &scene.entities {
            for &v in pos {
                assert!(v >= -5.0 && v <= 5.0, "out of bounds: {}", v);
            }
        }
    }

    #[test]
    fn cluster_scene_entity_count() {
        let scene = generate_cluster_scene(3, 5, 1.0);
        assert_eq!(scene.entity_count, 15);
        assert_eq!(scene.entities.len(), 15);
    }

    #[test]
    fn chain_scene_entity_count() {
        let scene = generate_chain_scene(10, 1.5);
        assert_eq!(scene.entity_count, 10);
        assert!(approx(scene.entities[9].1[0], 13.5, 1e-9));
    }

    // ---- statistics --------------------------------------------------------

    #[test]
    fn statistics_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_statistics(&vals);
        assert!(approx(stats.mean, 3.0, 1e-9));
        assert!(approx(stats.median, 3.0, 1e-9));
        assert!(approx(stats.min, 1.0, 1e-9));
        assert!(approx(stats.max, 5.0, 1e-9));
    }

    #[test]
    fn statistics_even_count() {
        let vals = vec![1.0, 3.0, 5.0, 7.0];
        let stats = compute_statistics(&vals);
        assert!(approx(stats.median, 4.0, 1e-9));
    }

    #[test]
    fn statistics_single_value() {
        let vals = vec![42.0];
        let stats = compute_statistics(&vals);
        assert!(approx(stats.mean, 42.0, 1e-9));
        assert!(approx(stats.median, 42.0, 1e-9));
        assert!(approx(stats.std_dev, 0.0, 1e-9));
    }

    #[test]
    fn statistics_empty() {
        let stats = compute_statistics(&[]);
        assert!(approx(stats.mean, 0.0, 1e-9));
        assert!(approx(stats.median, 0.0, 1e-9));
    }

    #[test]
    fn statistics_std_dev() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = compute_statistics(&vals);
        // Population std dev ≈ 2.0.
        assert!(approx(stats.mean, 5.0, 1e-9));
        assert!(approx(stats.std_dev, 2.0, 0.01));
    }

    #[test]
    fn statistics_p95() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let stats = compute_statistics(&vals);
        // p95 of [0..99] should be around 94-95.
        assert!(stats.p95 >= 93.0 && stats.p95 <= 95.0);
    }

    // ---- single benchmark --------------------------------------------------

    #[test]
    fn run_single_benchmark_basic() {
        let scene = generate_grid_scene(2, 1.0);
        let result = run_single_benchmark(&scene, 1.0 / 60.0, 0.5);
        assert!(result.passed);
        assert!(result.elapsed_ms > 0.0);
        assert!(result.steps_per_second > 0.0);
        assert_eq!(result.entity_count, 4);
    }

    #[test]
    fn run_single_benchmark_random_scene() {
        let scene = generate_random_scene(8, 5.0, 42);
        let result = run_single_benchmark(&scene, 1.0 / 30.0, 0.5);
        assert!(result.passed);
        assert!(result.entity_count == 8);
    }

    // ---- benchmark suite ---------------------------------------------------

    #[test]
    fn suite_generate_and_run() {
        let params = BenchmarkParams {
            min_entities: 2,
            max_entities: 6,
            entity_step: 2,
            duration: 0.2,
            time_step: 1.0 / 30.0,
        };
        let mut suite = BenchmarkSuite::new(params);
        suite.generate_benchmark_scenes();
        assert!(suite.scene_count() >= 2);

        let results = suite.run_all();
        assert_eq!(results.len(), suite.scene_count());
        for r in &results {
            assert!(r.passed);
        }
    }

    #[test]
    fn suite_add_custom_scene() {
        let mut suite = BenchmarkSuite::new(BenchmarkParams::default());
        let scene = generate_chain_scene(5, 1.0);
        suite.add_scene(scene);
        assert_eq!(suite.scene_count(), 1);
    }

    // ---- report generation -------------------------------------------------

    #[test]
    fn report_generation() {
        let scene = generate_grid_scene(2, 1.0);
        let result = run_single_benchmark(&scene, 1.0 / 60.0, 0.2);
        let report = BenchmarkReport::generate(vec![result]);
        assert_eq!(report.result_count(), 1);
        assert_eq!(report.failure_count(), 0);
    }

    #[test]
    fn report_to_json_round_trip() {
        let scene = generate_grid_scene(2, 1.0);
        let result = run_single_benchmark(&scene, 1.0 / 60.0, 0.1);
        let report = BenchmarkReport::generate(vec![result]);
        let json = report.to_json();
        let decoded: BenchmarkReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.result_count(), 1);
    }

    #[test]
    fn report_to_text() {
        let scenes: Vec<BenchmarkScene> = (2..=4).map(|n| generate_grid_scene(n, 1.0)).collect();
        let results: Vec<BenchmarkResult> = scenes
            .iter()
            .map(|s| run_single_benchmark(s, 1.0 / 30.0, 0.1))
            .collect();
        let report = BenchmarkReport::generate(results);
        let text = report.to_text();
        assert!(text.contains("Scene"));
        assert!(text.contains("Entities"));
        assert!(text.contains("Summary"));
    }

    // ---- comparison --------------------------------------------------------

    #[test]
    fn compare_identical_reports() {
        let scene = generate_grid_scene(2, 1.0);
        let r = run_single_benchmark(&scene, 1.0 / 60.0, 0.1);
        let report = BenchmarkReport::generate(vec![r]);
        let ratios = compare_reports(&report, &report);
        for &v in ratios.values() {
            assert!(approx(v, 1.0, 1e-9));
        }
    }

    #[test]
    fn has_regression_identical() {
        let scene = generate_grid_scene(2, 1.0);
        let r = run_single_benchmark(&scene, 1.0 / 60.0, 0.1);
        let report = BenchmarkReport::generate(vec![r]);
        assert!(!has_regression(&report, &report, 0.9));
    }

    // ---- benchmark params default ------------------------------------------

    #[test]
    fn default_params() {
        let p = BenchmarkParams::default();
        assert!(p.min_entities <= p.max_entities);
        assert!(p.time_step > 0.0);
        assert!(p.duration > 0.0);
    }
}
