//! Performance metrics: solve time, node count, gap statistics, cut statistics,
//! geometric mean computation, shifted geometric mean, and performance profiles.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Cut statistics
// ---------------------------------------------------------------------------

/// Statistics for cuts of a single type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutStats {
    /// Number of cuts generated.
    pub count: u64,
    /// Average violation of the generated cuts.
    pub avg_violation: f64,
    /// Maximum violation.
    pub max_violation: f64,
    /// Total time spent generating cuts of this type.
    pub generation_time_secs: f64,
}

impl CutStats {
    /// Create empty stats.
    pub fn zero() -> Self {
        CutStats {
            count: 0,
            avg_violation: 0.0,
            max_violation: 0.0,
            generation_time_secs: 0.0,
        }
    }

    /// Merge another CutStats into this one.
    pub fn merge(&mut self, other: &CutStats) {
        let total_count = self.count + other.count;
        if total_count > 0 {
            self.avg_violation = (self.avg_violation * self.count as f64
                + other.avg_violation * other.count as f64)
                / total_count as f64;
        }
        self.max_violation = self.max_violation.max(other.max_violation);
        self.count = total_count;
        self.generation_time_secs += other.generation_time_secs;
    }
}

// ---------------------------------------------------------------------------
// Benchmark metrics
// ---------------------------------------------------------------------------

/// Detailed metrics collected during a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Total solve time in seconds.
    pub solve_time_secs: f64,
    /// Number of branch-and-bound nodes.
    pub node_count: u64,
    /// Root gap as a percentage.
    pub root_gap_percent: f64,
    /// Final gap as a percentage.
    pub final_gap_percent: f64,
    /// Root gap closure as a percentage.
    pub root_gap_closure_percent: f64,
    /// Total LP iterations.
    pub iteration_count: u64,
    /// Cut statistics by type.
    pub cuts_by_type: HashMap<String, CutStats>,
    /// Time spent on reformulation.
    pub reformulation_time_secs: f64,
    /// Time spent generating cuts.
    pub cut_generation_time_secs: f64,
    /// Time spent solving LP relaxations.
    pub lp_solve_time_secs: f64,
    /// Time spent on branching decisions.
    pub branching_time_secs: f64,
    /// Remaining time not categorized above.
    pub other_time_secs: f64,
}

impl BenchmarkMetrics {
    /// Create zeroed metrics.
    pub fn zero() -> Self {
        BenchmarkMetrics {
            solve_time_secs: 0.0,
            node_count: 0,
            root_gap_percent: 0.0,
            final_gap_percent: 0.0,
            root_gap_closure_percent: 0.0,
            iteration_count: 0,
            cuts_by_type: HashMap::new(),
            reformulation_time_secs: 0.0,
            cut_generation_time_secs: 0.0,
            lp_solve_time_secs: 0.0,
            branching_time_secs: 0.0,
            other_time_secs: 0.0,
        }
    }

    /// Total number of cuts across all types.
    pub fn total_cuts(&self) -> u64 {
        self.cuts_by_type.values().map(|s| s.count).sum()
    }

    /// Total cut generation time across all types.
    pub fn total_cut_time(&self) -> f64 {
        self.cuts_by_type
            .values()
            .map(|s| s.generation_time_secs)
            .sum()
    }

    /// Fraction of solve time spent on LP relaxations.
    pub fn lp_time_fraction(&self) -> f64 {
        if self.solve_time_secs > 0.0 {
            self.lp_solve_time_secs / self.solve_time_secs
        } else {
            0.0
        }
    }

    /// Fraction of solve time spent on cut generation.
    pub fn cut_time_fraction(&self) -> f64 {
        if self.solve_time_secs > 0.0 {
            self.cut_generation_time_secs / self.solve_time_secs
        } else {
            0.0
        }
    }

    /// Timing breakdown as a HashMap of phase name → seconds.
    pub fn timing_breakdown(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("reformulation".to_string(), self.reformulation_time_secs);
        m.insert("cut_generation".to_string(), self.cut_generation_time_secs);
        m.insert("lp_solve".to_string(), self.lp_solve_time_secs);
        m.insert("branching".to_string(), self.branching_time_secs);
        m.insert("other".to_string(), self.other_time_secs);
        m
    }
}

// ---------------------------------------------------------------------------
// Geometric mean computations
// ---------------------------------------------------------------------------

/// Compute the geometric mean of a slice of positive values.
/// Returns `None` if the slice is empty or contains non-positive values.
pub fn geometric_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut log_sum = 0.0;
    for &v in values {
        if v <= 0.0 {
            return None;
        }
        log_sum += v.ln();
    }
    Some((log_sum / values.len() as f64).exp())
}

/// Compute the shifted geometric mean: geomean(values + shift) - shift.
/// This handles zero values gracefully (common for solve times).
pub fn shifted_geometric_mean(values: &[f64], shift: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let shifted: Vec<f64> = values.iter().map(|v| v + shift).collect();
    geometric_mean(&shifted).map(|gm| gm - shift)
}

/// Compute the arithmetic mean of a slice.
pub fn arithmetic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Compute the median of a slice.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

/// Compute a specified percentile (0–100) of a slice.
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() || p < 0.0 || p > 100.0 {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    Some(sorted[idx])
}

/// Standard deviation of a slice.
pub fn std_dev(values: &[f64]) -> Option<f64> {
    let mean = arithmetic_mean(values)?;
    let n = values.len() as f64;
    if n < 2.0 {
        return Some(0.0);
    }
    let var: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some(var.sqrt())
}

// ---------------------------------------------------------------------------
// Performance profile
// ---------------------------------------------------------------------------

/// A single performance profile point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilePoint {
    /// Performance ratio τ.
    pub tau: f64,
    /// Fraction of instances solved within ratio τ.
    pub fraction: f64,
}

/// Performance profile data for a single configuration (Dolan-Moré style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Configuration label.
    pub config_label: String,
    /// Sorted (τ, fraction) points.
    pub points: Vec<ProfilePoint>,
}

impl PerformanceProfile {
    /// Compute a performance profile from solve times across configurations.
    ///
    /// `times` maps config_label → Vec<f64> of solve times for each instance.
    /// All vectors must have the same length. Times of `f64::INFINITY` denote unsolved.
    pub fn compute_profiles(times: &HashMap<String, Vec<f64>>) -> Vec<PerformanceProfile> {
        if times.is_empty() {
            return Vec::new();
        }

        let configs: Vec<&String> = times.keys().collect();
        let n_instances = times.values().next().map(|v| v.len()).unwrap_or(0);
        if n_instances == 0 {
            return configs
                .iter()
                .map(|c| PerformanceProfile {
                    config_label: c.to_string(),
                    points: vec![],
                })
                .collect();
        }

        // Compute best time for each instance.
        let mut best_times = vec![f64::INFINITY; n_instances];
        for ts in times.values() {
            for (i, &t) in ts.iter().enumerate() {
                if i < n_instances && t < best_times[i] {
                    best_times[i] = t;
                }
            }
        }

        // For each configuration, compute performance ratios.
        let mut profiles = Vec::new();
        for config in &configs {
            let ts = &times[*config];
            let mut ratios: Vec<f64> = Vec::with_capacity(n_instances);
            for (i, &t) in ts.iter().enumerate() {
                if i < n_instances {
                    let best = best_times[i];
                    if best <= 0.0 || best.is_infinite() {
                        if t.is_infinite() {
                            ratios.push(f64::INFINITY);
                        } else {
                            ratios.push(1.0);
                        }
                    } else if t.is_infinite() {
                        ratios.push(f64::INFINITY);
                    } else {
                        ratios.push(t / best);
                    }
                }
            }

            // Build profile: for τ from 1.0 up, fraction ≤ τ.
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = ratios.len() as f64;
            let max_tau = ratios
                .iter()
                .filter(|r| r.is_finite())
                .cloned()
                .last()
                .unwrap_or(1.0)
                .max(1.0);

            let num_points = 200;
            let mut points = Vec::with_capacity(num_points);
            for step in 0..=num_points {
                let tau = 1.0 + (max_tau - 1.0) * (step as f64 / num_points as f64);
                let count = ratios.iter().filter(|&&r| r <= tau + 1e-12).count();
                points.push(ProfilePoint {
                    tau,
                    fraction: count as f64 / n,
                });
            }

            profiles.push(PerformanceProfile {
                config_label: config.to_string(),
                points,
            });
        }

        profiles
    }

    /// Fraction of instances for which this config is the best (τ=1).
    pub fn efficiency(&self) -> f64 {
        self.points.first().map(|p| p.fraction).unwrap_or(0.0)
    }

    /// Fraction of instances solved (τ at the last finite point).
    pub fn robustness(&self) -> f64 {
        self.points.last().map(|p| p.fraction).unwrap_or(0.0)
    }

    /// Area under the performance profile curve (approximated by trapezoidal rule).
    pub fn area_under_curve(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 1..self.points.len() {
            let dt = self.points[i].tau - self.points[i - 1].tau;
            let avg_f = (self.points[i].fraction + self.points[i - 1].fraction) / 2.0;
            area += dt * avg_f;
        }
        area
    }
}

// ---------------------------------------------------------------------------
// Aggregated metrics across a set of benchmark results
// ---------------------------------------------------------------------------

/// Aggregate metrics over a set of benchmark results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Number of instances.
    pub count: usize,
    /// Number solved to optimality.
    pub solved: usize,
    /// Arithmetic mean solve time (solved instances).
    pub mean_time: f64,
    /// Shifted geometric mean solve time (shift = 1.0).
    pub sgm_time: f64,
    /// Median solve time.
    pub median_time: f64,
    /// Arithmetic mean node count.
    pub mean_nodes: f64,
    /// Shifted geometric mean node count (shift = 10).
    pub sgm_nodes: f64,
    /// Mean root gap closure %.
    pub mean_root_gap_closure: f64,
    /// Mean final gap %.
    pub mean_final_gap: f64,
    /// Total cuts generated.
    pub total_cuts: u64,
    /// Mean iteration count.
    pub mean_iterations: f64,
}

impl AggregateMetrics {
    /// Compute aggregate metrics from run results that have detailed metrics.
    pub fn from_results(results: &[crate::runner::RunResult]) -> Self {
        let count = results.len();
        let solved = results
            .iter()
            .filter(|r| r.status == crate::runner::RunStatus::Optimal)
            .count();

        let times: Vec<f64> = results
            .iter()
            .filter(|r| r.status == crate::runner::RunStatus::Optimal)
            .map(|r| r.wall_time_secs)
            .collect();

        let mean_time = arithmetic_mean(&times).unwrap_or(0.0);
        let sgm_time = shifted_geometric_mean(&times, 1.0).unwrap_or(0.0);
        let median_time = median(&times).unwrap_or(0.0);

        let nodes: Vec<f64> = results.iter().map(|r| r.node_count as f64).collect();
        let mean_nodes = arithmetic_mean(&nodes).unwrap_or(0.0);
        let sgm_nodes = shifted_geometric_mean(&nodes, 10.0).unwrap_or(0.0);

        let root_closures: Vec<f64> = results
            .iter()
            .filter_map(|r| r.metrics.as_ref())
            .map(|m| m.root_gap_closure_percent)
            .collect();
        let mean_root_gap_closure = arithmetic_mean(&root_closures).unwrap_or(0.0);

        let final_gaps: Vec<f64> = results
            .iter()
            .filter_map(|r| r.metrics.as_ref())
            .map(|m| m.final_gap_percent)
            .collect();
        let mean_final_gap = arithmetic_mean(&final_gaps).unwrap_or(0.0);

        let total_cuts: u64 = results.iter().map(|r| r.cuts_generated).sum();

        let iterations: Vec<f64> = results.iter().map(|r| r.iteration_count as f64).collect();
        let mean_iterations = arithmetic_mean(&iterations).unwrap_or(0.0);

        AggregateMetrics {
            count,
            solved,
            mean_time,
            sgm_time,
            median_time,
            mean_nodes,
            sgm_nodes,
            mean_root_gap_closure,
            mean_final_gap,
            total_cuts,
            mean_iterations,
        }
    }

    /// Solve rate as a percentage.
    pub fn solve_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.solved as f64 / self.count as f64) * 100.0
        }
    }
}

impl std::fmt::Display for AggregateMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Instances: {}, Solved: {} ({:.1}%)",
            self.count,
            self.solved,
            self.solve_rate()
        )?;
        writeln!(
            f,
            "Time: mean={:.3}s, sgm={:.3}s, median={:.3}s",
            self.mean_time, self.sgm_time, self.median_time
        )?;
        writeln!(
            f,
            "Nodes: mean={:.1}, sgm={:.1}",
            self.mean_nodes, self.sgm_nodes
        )?;
        writeln!(
            f,
            "Root gap closure: {:.1}%, Final gap: {:.1}%",
            self.mean_root_gap_closure, self.mean_final_gap
        )?;
        writeln!(
            f,
            "Total cuts: {}, Mean iterations: {:.0}",
            self.total_cuts, self.mean_iterations
        )
    }
}

// ---------------------------------------------------------------------------
// Metrics aggregator (incremental)
// ---------------------------------------------------------------------------

/// Incremental metrics aggregator for streaming results.
#[derive(Debug, Clone)]
pub struct MetricsAggregator {
    times: Vec<f64>,
    nodes: Vec<f64>,
    root_closures: Vec<f64>,
    final_gaps: Vec<f64>,
    iterations: Vec<f64>,
    total_cuts: u64,
    solved: usize,
}

impl MetricsAggregator {
    /// Create a new empty aggregator.
    pub fn new() -> Self {
        MetricsAggregator {
            times: Vec::new(),
            nodes: Vec::new(),
            root_closures: Vec::new(),
            final_gaps: Vec::new(),
            iterations: Vec::new(),
            total_cuts: 0,
            solved: 0,
        }
    }

    /// Add a run result.
    pub fn add(&mut self, result: &crate::runner::RunResult) {
        if result.status == crate::runner::RunStatus::Optimal {
            self.times.push(result.wall_time_secs);
            self.solved += 1;
        }
        self.nodes.push(result.node_count as f64);
        self.iterations.push(result.iteration_count as f64);
        self.total_cuts += result.cuts_generated;
        if let Some(ref m) = result.metrics {
            self.root_closures.push(m.root_gap_closure_percent);
            self.final_gaps.push(m.final_gap_percent);
        }
    }

    /// Number of results added so far.
    pub fn count(&self) -> usize {
        self.nodes.len()
    }

    /// Build aggregate metrics from collected data.
    pub fn aggregate(&self) -> AggregateMetrics {
        let count = self.nodes.len();
        let mean_time = arithmetic_mean(&self.times).unwrap_or(0.0);
        let sgm_time = shifted_geometric_mean(&self.times, 1.0).unwrap_or(0.0);
        let median_time = median(&self.times).unwrap_or(0.0);
        let mean_nodes = arithmetic_mean(&self.nodes).unwrap_or(0.0);
        let sgm_nodes = shifted_geometric_mean(&self.nodes, 10.0).unwrap_or(0.0);
        let mean_root_gap_closure = arithmetic_mean(&self.root_closures).unwrap_or(0.0);
        let mean_final_gap = arithmetic_mean(&self.final_gaps).unwrap_or(0.0);
        let mean_iterations = arithmetic_mean(&self.iterations).unwrap_or(0.0);

        AggregateMetrics {
            count,
            solved: self.solved,
            mean_time,
            sgm_time,
            median_time,
            mean_nodes,
            sgm_nodes,
            mean_root_gap_closure,
            mean_final_gap,
            total_cuts: self.total_cuts,
            mean_iterations,
        }
    }

    /// Reset the aggregator.
    pub fn reset(&mut self) {
        self.times.clear();
        self.nodes.clear();
        self.root_closures.clear();
        self.final_gaps.clear();
        self.iterations.clear();
        self.total_cuts = 0;
        self.solved = 0;
    }
}

impl Default for MetricsAggregator {
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
    fn test_geometric_mean_basic() {
        let vals = vec![2.0, 8.0];
        let gm = geometric_mean(&vals).unwrap();
        assert!((gm - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_mean_single() {
        let vals = vec![5.0];
        let gm = geometric_mean(&vals).unwrap();
        assert!((gm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_mean_empty() {
        assert!(geometric_mean(&[]).is_none());
    }

    #[test]
    fn test_geometric_mean_negative() {
        assert!(geometric_mean(&[-1.0, 2.0]).is_none());
    }

    #[test]
    fn test_shifted_geometric_mean() {
        let vals = vec![0.0, 1.0, 2.0];
        let sgm = shifted_geometric_mean(&vals, 1.0).unwrap();
        // sgm = geomean([1, 2, 3]) - 1 = (6)^{1/3} - 1 ≈ 0.817
        assert!(sgm > 0.0);
        assert!(sgm < 2.0);
    }

    #[test]
    fn test_median_odd() {
        let vals = vec![1.0, 3.0, 2.0];
        assert_eq!(median(&vals).unwrap(), 2.0);
    }

    #[test]
    fn test_median_even() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&vals).unwrap(), 2.5);
    }

    #[test]
    fn test_percentile_50() {
        let vals = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let p = percentile(&vals, 50.0).unwrap();
        assert!((p - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&vals).unwrap();
        assert!((sd - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_performance_profile() {
        let mut times = HashMap::new();
        times.insert("A".to_string(), vec![1.0, 2.0, 5.0]);
        times.insert("B".to_string(), vec![2.0, 1.0, 3.0]);
        let profiles = PerformanceProfile::compute_profiles(&times);
        assert_eq!(profiles.len(), 2);
        for p in &profiles {
            assert!(!p.points.is_empty());
            assert!(p.efficiency() >= 0.0);
            assert!(p.robustness() >= 0.0);
        }
    }

    #[test]
    fn test_cut_stats_merge() {
        let mut a = CutStats {
            count: 10,
            avg_violation: 0.1,
            max_violation: 0.5,
            generation_time_secs: 1.0,
        };
        let b = CutStats {
            count: 20,
            avg_violation: 0.2,
            max_violation: 0.3,
            generation_time_secs: 2.0,
        };
        a.merge(&b);
        assert_eq!(a.count, 30);
        assert!((a.avg_violation - (0.1 * 10.0 + 0.2 * 20.0) / 30.0).abs() < 1e-10);
        assert_eq!(a.max_violation, 0.5);
        assert_eq!(a.generation_time_secs, 3.0);
    }

    #[test]
    fn test_metrics_aggregator() {
        let mut agg = MetricsAggregator::new();
        let r = crate::runner::RunResult::optimal("test", "cfg", 1.0, 10.0, 5, 100);
        agg.add(&r);
        assert_eq!(agg.count(), 1);
        let am = agg.aggregate();
        assert_eq!(am.solved, 1);
    }
}
