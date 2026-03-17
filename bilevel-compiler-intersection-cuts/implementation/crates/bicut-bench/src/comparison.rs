//! Result comparison: compare across configurations, statistical tests
//! (Wilcoxon signed-rank), performance profile generation, virtual best solver,
//! domination analysis, and Dolan-Moré profiles.

use crate::metrics::{
    arithmetic_mean, geometric_mean, median, shifted_geometric_mean, AggregateMetrics,
    PerformanceProfile,
};
use crate::runner::{RunResult, RunStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Comparison report
// ---------------------------------------------------------------------------

/// Full comparison report across multiple configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Per-configuration aggregate metrics.
    pub config_metrics: HashMap<String, AggregateMetrics>,
    /// Pairwise Wilcoxon tests.
    pub pairwise_tests: Vec<PairwiseTest>,
    /// Domination analysis.
    pub domination: DomainAnalysis,
    /// Virtual best solver results.
    pub virtual_best: VirtualBestSolver,
    /// Performance profile data.
    pub profiles: Vec<PerformanceProfileData>,
}

impl ComparisonReport {
    /// Build a full comparison report from a config→results map.
    pub fn build(results: &HashMap<String, Vec<RunResult>>) -> Self {
        let config_metrics: HashMap<String, AggregateMetrics> = results
            .iter()
            .map(|(label, rs)| (label.clone(), AggregateMetrics::from_results(rs)))
            .collect();

        let configs: Vec<String> = results.keys().cloned().collect();
        let mut pairwise_tests = Vec::new();
        for i in 0..configs.len() {
            for j in (i + 1)..configs.len() {
                let a = &results[&configs[i]];
                let b = &results[&configs[j]];
                let test = PairwiseTest::wilcoxon_signed_rank(&configs[i], &configs[j], a, b);
                pairwise_tests.push(test);
            }
        }

        let domination = DomainAnalysis::compute(results);
        let virtual_best = VirtualBestSolver::compute(results);
        let profiles = PerformanceProfileData::compute_all(results);

        ComparisonReport {
            config_metrics,
            pairwise_tests,
            domination,
            virtual_best,
            profiles,
        }
    }

    /// Rank configurations by shifted geometric mean solve time.
    pub fn rank_by_sgm_time(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = self
            .config_metrics
            .iter()
            .map(|(label, m)| (label.clone(), m.sgm_time))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Rank configurations by solve rate.
    pub fn rank_by_solve_rate(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = self
            .config_metrics
            .iter()
            .map(|(label, m)| (label.clone(), m.solve_rate()))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Identify the best configuration by each metric.
    pub fn best_config_by_metric(&self) -> HashMap<String, String> {
        let mut bests = HashMap::new();

        // Best by solve rate.
        if let Some((label, _)) = self.rank_by_solve_rate().first() {
            bests.insert("solve_rate".to_string(), label.clone());
        }

        // Best by SGM time.
        if let Some((label, _)) = self.rank_by_sgm_time().first() {
            bests.insert("sgm_time".to_string(), label.clone());
        }

        // Best by mean root gap closure.
        if let Some((label, _)) = self.config_metrics.iter().max_by(|a, b| {
            a.1.mean_root_gap_closure
                .partial_cmp(&b.1.mean_root_gap_closure)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            bests.insert("root_gap_closure".to_string(), label.clone());
        }

        bests
    }
}

// ---------------------------------------------------------------------------
// Wilcoxon signed-rank test
// ---------------------------------------------------------------------------

/// Result of a Wilcoxon signed-rank test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilcoxonResult {
    /// Test statistic W.
    pub w_statistic: f64,
    /// Number of non-zero differences.
    pub n_nonzero: usize,
    /// Approximate p-value (normal approximation).
    pub p_value: f64,
    /// Whether the result is significant at α=0.05.
    pub significant: bool,
    /// Direction: +1 if config_a is better, -1 if config_b is better, 0 if tied.
    pub direction: i32,
}

/// Perform a Wilcoxon signed-rank test on paired solve times.
pub fn wilcoxon_signed_rank(times_a: &[f64], times_b: &[f64]) -> WilcoxonResult {
    let n = times_a.len().min(times_b.len());
    if n == 0 {
        return WilcoxonResult {
            w_statistic: 0.0,
            n_nonzero: 0,
            p_value: 1.0,
            significant: false,
            direction: 0,
        };
    }

    // Compute differences.
    let mut diffs: Vec<(f64, f64)> = Vec::new(); // (|diff|, sign)
    for i in 0..n {
        let d = times_a[i] - times_b[i];
        if d.abs() > 1e-12 {
            diffs.push((d.abs(), if d > 0.0 { 1.0 } else { -1.0 }));
        }
    }

    let n_nonzero = diffs.len();
    if n_nonzero == 0 {
        return WilcoxonResult {
            w_statistic: 0.0,
            n_nonzero: 0,
            p_value: 1.0,
            significant: false,
            direction: 0,
        };
    }

    // Sort by absolute difference.
    diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (average ranks for ties).
    let mut ranks = vec![0.0; n_nonzero];
    let mut i = 0;
    while i < n_nonzero {
        let mut j = i;
        while j < n_nonzero && (diffs[j].0 - diffs[i].0).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Compute W+ (sum of ranks for positive differences).
    let w_plus: f64 = ranks
        .iter()
        .zip(diffs.iter())
        .filter(|(_, d)| d.1 > 0.0)
        .map(|(r, _)| r)
        .sum();
    let w_minus: f64 = ranks
        .iter()
        .zip(diffs.iter())
        .filter(|(_, d)| d.1 < 0.0)
        .map(|(r, _)| r)
        .sum();

    let w = w_plus.min(w_minus);

    // Normal approximation for p-value.
    let nn = n_nonzero as f64;
    let mean_w = nn * (nn + 1.0) / 4.0;
    let var_w = nn * (nn + 1.0) * (2.0 * nn + 1.0) / 24.0;
    let p_value = if var_w > 0.0 {
        let z = (w - mean_w).abs() / var_w.sqrt();
        // Two-tailed p-value using standard normal approximation.
        2.0 * standard_normal_cdf(-z)
    } else {
        1.0
    };

    let direction = if w_plus < w_minus {
        1 // config_a had more negative diffs → a is faster
    } else if w_minus < w_plus {
        -1
    } else {
        0
    };

    WilcoxonResult {
        w_statistic: w,
        n_nonzero,
        p_value,
        significant: p_value < 0.05,
        direction,
    }
}

/// Approximate CDF of the standard normal distribution.
fn standard_normal_cdf(x: f64) -> f64 {
    // Abramowitz & Stegun approximation 26.2.17
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * abs_x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let erf_approx =
        1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-abs_x * abs_x).exp();
    0.5 * (1.0 + sign * erf_approx)
}

/// Pairwise comparison between two configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseTest {
    pub config_a: String,
    pub config_b: String,
    pub wilcoxon: WilcoxonResult,
    /// Number of instances where A is strictly better.
    pub a_wins: usize,
    /// Number of instances where B is strictly better.
    pub b_wins: usize,
    /// Ties.
    pub ties: usize,
    /// Geometric mean speedup of A over B (>1 means A is faster).
    pub geo_mean_speedup: f64,
}

impl PairwiseTest {
    /// Run a pairwise comparison using Wilcoxon signed-rank on solve times.
    pub fn wilcoxon_signed_rank(
        config_a: &str,
        config_b: &str,
        results_a: &[RunResult],
        results_b: &[RunResult],
    ) -> Self {
        let n = results_a.len().min(results_b.len());
        let mut times_a = Vec::with_capacity(n);
        let mut times_b = Vec::with_capacity(n);
        let mut a_wins = 0;
        let mut b_wins = 0;
        let mut ties = 0;
        let mut ratios = Vec::new();

        for i in 0..n {
            let ta = results_a[i].wall_time_secs;
            let tb = results_b[i].wall_time_secs;
            times_a.push(ta);
            times_b.push(tb);

            let tol = 1e-6;
            if ta < tb - tol {
                a_wins += 1;
            } else if tb < ta - tol {
                b_wins += 1;
            } else {
                ties += 1;
            }

            if ta > 0.0 {
                ratios.push(tb / ta);
            }
        }

        let wilcoxon = wilcoxon_signed_rank(&times_a, &times_b);
        let geo_mean_speedup = geometric_mean(&ratios).unwrap_or(1.0);

        PairwiseTest {
            config_a: config_a.to_string(),
            config_b: config_b.to_string(),
            wilcoxon,
            a_wins,
            b_wins,
            ties,
            geo_mean_speedup,
        }
    }
}

impl std::fmt::Display for PairwiseTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} vs {}", self.config_a, self.config_b)?;
        writeln!(
            f,
            "  Wins: {} / {} / {} (A/B/tie)",
            self.a_wins, self.b_wins, self.ties
        )?;
        writeln!(f, "  Wilcoxon p-value: {:.4}", self.wilcoxon.p_value)?;
        writeln!(f, "  Significant: {}", self.wilcoxon.significant)?;
        write!(f, "  Geo mean speedup: {:.3}x", self.geo_mean_speedup)
    }
}

// ---------------------------------------------------------------------------
// Domination analysis
// ---------------------------------------------------------------------------

/// Domination analysis across configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAnalysis {
    /// For each config, how many configs does it dominate?
    pub domination_count: HashMap<String, usize>,
    /// For each config, set of configs it dominates.
    pub dominates: HashMap<String, Vec<String>>,
    /// For each config, set of configs that dominate it.
    pub dominated_by: HashMap<String, Vec<String>>,
    /// Pareto-optimal configurations (not dominated by any other).
    pub pareto_optimal: Vec<String>,
}

impl DomainAnalysis {
    /// Compute domination relationships.
    /// Config A dominates config B if A is at least as good on all instances
    /// and strictly better on at least one.
    pub fn compute(results: &HashMap<String, Vec<RunResult>>) -> Self {
        let configs: Vec<String> = results.keys().cloned().collect();
        let mut domination_count = HashMap::new();
        let mut dominates: HashMap<String, Vec<String>> = HashMap::new();
        let mut dominated_by: HashMap<String, Vec<String>> = HashMap::new();

        for c in &configs {
            domination_count.insert(c.clone(), 0);
            dominates.insert(c.clone(), Vec::new());
            dominated_by.insert(c.clone(), Vec::new());
        }

        for i in 0..configs.len() {
            for j in 0..configs.len() {
                if i == j {
                    continue;
                }
                let ra = &results[&configs[i]];
                let rb = &results[&configs[j]];
                if does_dominate(ra, rb) {
                    *domination_count.entry(configs[i].clone()).or_default() += 1;
                    dominates
                        .entry(configs[i].clone())
                        .or_default()
                        .push(configs[j].clone());
                    dominated_by
                        .entry(configs[j].clone())
                        .or_default()
                        .push(configs[i].clone());
                }
            }
        }

        let pareto_optimal: Vec<String> = configs
            .iter()
            .filter(|c| dominated_by.get(*c).map(|v| v.is_empty()).unwrap_or(true))
            .cloned()
            .collect();

        DomainAnalysis {
            domination_count,
            dominates,
            dominated_by,
            pareto_optimal,
        }
    }
}

/// Check if results_a dominates results_b (all ≤ and at least one <).
fn does_dominate(results_a: &[RunResult], results_b: &[RunResult]) -> bool {
    let n = results_a.len().min(results_b.len());
    if n == 0 {
        return false;
    }
    let mut all_leq = true;
    let mut any_strictly_better = false;
    let tol = 1e-6;
    for i in 0..n {
        let ta = results_a[i].wall_time_secs;
        let tb = results_b[i].wall_time_secs;
        if ta > tb + tol {
            all_leq = false;
            break;
        }
        if ta < tb - tol {
            any_strictly_better = true;
        }
    }
    all_leq && any_strictly_better
}

// ---------------------------------------------------------------------------
// Virtual best solver
// ---------------------------------------------------------------------------

/// Virtual best solver: for each instance, take the best result across configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualBestSolver {
    /// Best time for each instance.
    pub best_times: Vec<f64>,
    /// Which configuration achieved the best for each instance.
    pub best_config: Vec<String>,
    /// Instance names.
    pub instance_names: Vec<String>,
    /// Aggregate metrics of the virtual best.
    pub aggregate: AggregateMetrics,
}

impl VirtualBestSolver {
    /// Compute the virtual best solver from a config→results map.
    pub fn compute(results: &HashMap<String, Vec<RunResult>>) -> Self {
        if results.is_empty() {
            return VirtualBestSolver {
                best_times: Vec::new(),
                best_config: Vec::new(),
                instance_names: Vec::new(),
                aggregate: AggregateMetrics {
                    count: 0,
                    solved: 0,
                    mean_time: 0.0,
                    sgm_time: 0.0,
                    median_time: 0.0,
                    mean_nodes: 0.0,
                    sgm_nodes: 0.0,
                    mean_root_gap_closure: 0.0,
                    mean_final_gap: 0.0,
                    total_cuts: 0,
                    mean_iterations: 0.0,
                },
            };
        }

        let configs: Vec<&String> = results.keys().collect();
        let n = results.values().next().map(|v| v.len()).unwrap_or(0);
        let mut best_times = vec![f64::INFINITY; n];
        let mut best_config = vec![String::new(); n];
        let mut instance_names = vec![String::new(); n];

        for (config, rs) in results {
            for (i, r) in rs.iter().enumerate() {
                if i < n {
                    if instance_names[i].is_empty() {
                        instance_names[i] = r.instance_name.clone();
                    }
                    let t = if r.status == RunStatus::Optimal {
                        r.wall_time_secs
                    } else {
                        f64::INFINITY
                    };
                    if t < best_times[i] {
                        best_times[i] = t;
                        best_config[i] = config.clone();
                    }
                }
            }
        }

        // Build virtual best RunResults for aggregate computation.
        let vb_results: Vec<RunResult> = (0..n)
            .map(|i| {
                if best_times[i].is_finite() {
                    RunResult::optimal(&instance_names[i], "virtual_best", best_times[i], 0.0, 0, 0)
                } else {
                    RunResult::timeout(
                        &instance_names[i],
                        "virtual_best",
                        f64::INFINITY,
                        None,
                        None,
                    )
                }
            })
            .collect();

        let aggregate = AggregateMetrics::from_results(&vb_results);

        VirtualBestSolver {
            best_times,
            best_config,
            instance_names,
            aggregate,
        }
    }

    /// How many instances each config is the virtual best for.
    pub fn config_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for c in &self.best_config {
            if !c.is_empty() {
                *counts.entry(c.clone()).or_default() += 1;
            }
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// Performance profile data (Dolan-Moré)
// ---------------------------------------------------------------------------

/// Performance profile data for one configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfileData {
    /// Configuration label.
    pub config_label: String,
    /// Performance ratios (τ values at which steps occur).
    pub ratios: Vec<f64>,
    /// Cumulative fraction at each τ.
    pub fractions: Vec<f64>,
    /// Efficiency (fraction at τ=1).
    pub efficiency: f64,
    /// Robustness (fraction at τ=max).
    pub robustness: f64,
}

impl PerformanceProfileData {
    /// Compute Dolan-Moré performance profiles for all configurations.
    pub fn compute_all(results: &HashMap<String, Vec<RunResult>>) -> Vec<Self> {
        if results.is_empty() {
            return Vec::new();
        }

        let configs: Vec<&String> = results.keys().collect();
        let n = results.values().next().map(|v| v.len()).unwrap_or(0);
        if n == 0 {
            return Vec::new();
        }

        // Extract times (use INFINITY for unsolved).
        let mut times: HashMap<String, Vec<f64>> = HashMap::new();
        for (config, rs) in results {
            let ts: Vec<f64> = rs
                .iter()
                .map(|r| {
                    if r.status == RunStatus::Optimal {
                        r.wall_time_secs.max(1e-6)
                    } else {
                        f64::INFINITY
                    }
                })
                .collect();
            times.insert(config.clone(), ts);
        }

        // Compute best time per instance.
        let mut best = vec![f64::INFINITY; n];
        for ts in times.values() {
            for (i, &t) in ts.iter().enumerate() {
                if i < n && t < best[i] {
                    best[i] = t;
                }
            }
        }

        // Build profiles.
        let mut profiles = Vec::new();
        for config in &configs {
            let ts = &times[*config];
            let mut ratios_raw: Vec<f64> = Vec::with_capacity(n);
            for (i, &t) in ts.iter().enumerate() {
                if i < n {
                    let b = best[i];
                    if b > 0.0 && b.is_finite() && t.is_finite() {
                        ratios_raw.push(t / b);
                    } else if t.is_infinite() {
                        ratios_raw.push(f64::INFINITY);
                    } else {
                        ratios_raw.push(1.0);
                    }
                }
            }

            // Sort ratios for CDF construction.
            let mut finite_ratios: Vec<f64> = ratios_raw
                .iter()
                .filter(|r| r.is_finite())
                .cloned()
                .collect();
            finite_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n_total = ratios_raw.len() as f64;
            let max_ratio = finite_ratios.last().cloned().unwrap_or(1.0);

            // Generate CDF points.
            let num_steps = 500;
            let mut cdf_ratios = Vec::with_capacity(num_steps);
            let mut cdf_fractions = Vec::with_capacity(num_steps);
            for step in 0..=num_steps {
                let tau = 1.0 + (max_ratio - 1.0) * (step as f64 / num_steps as f64);
                let count = ratios_raw.iter().filter(|&&r| r <= tau + 1e-12).count();
                cdf_ratios.push(tau);
                cdf_fractions.push(count as f64 / n_total);
            }

            let efficiency = ratios_raw
                .iter()
                .filter(|&&r| (r - 1.0).abs() < 1e-6)
                .count() as f64
                / n_total;
            let robustness = ratios_raw.iter().filter(|r| r.is_finite()).count() as f64 / n_total;

            profiles.push(PerformanceProfileData {
                config_label: config.to_string(),
                ratios: cdf_ratios,
                fractions: cdf_fractions,
                efficiency,
                robustness,
            });
        }

        profiles
    }
}

// ---------------------------------------------------------------------------
// ConfigComparison: compare exactly two configurations
// ---------------------------------------------------------------------------

/// Direct comparison between two configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigComparison {
    pub config_a: String,
    pub config_b: String,
    pub aggregate_a: AggregateMetrics,
    pub aggregate_b: AggregateMetrics,
    pub pairwise: PairwiseTest,
    /// Per-instance time ratios (a_time / b_time).
    pub time_ratios: Vec<f64>,
    /// Instances where A is uniquely optimal.
    pub a_unique_solves: Vec<String>,
    /// Instances where B is uniquely optimal.
    pub b_unique_solves: Vec<String>,
}

impl ConfigComparison {
    /// Compare two configurations.
    pub fn compare(
        config_a: &str,
        config_b: &str,
        results_a: &[RunResult],
        results_b: &[RunResult],
    ) -> Self {
        let agg_a = AggregateMetrics::from_results(results_a);
        let agg_b = AggregateMetrics::from_results(results_b);
        let pairwise = PairwiseTest::wilcoxon_signed_rank(config_a, config_b, results_a, results_b);

        let n = results_a.len().min(results_b.len());
        let mut time_ratios = Vec::with_capacity(n);
        let mut a_unique = Vec::new();
        let mut b_unique = Vec::new();

        for i in 0..n {
            let ra = &results_a[i];
            let rb = &results_b[i];
            let ta = if ra.status == RunStatus::Optimal {
                ra.wall_time_secs
            } else {
                f64::INFINITY
            };
            let tb = if rb.status == RunStatus::Optimal {
                rb.wall_time_secs
            } else {
                f64::INFINITY
            };

            if ta.is_finite() && tb.is_finite() && tb > 0.0 {
                time_ratios.push(ta / tb);
            } else if ta.is_finite() && tb.is_infinite() {
                time_ratios.push(0.0); // A solved, B didn't
            } else if ta.is_infinite() && tb.is_finite() {
                time_ratios.push(f64::INFINITY); // B solved, A didn't
            }

            if ra.status == RunStatus::Optimal && rb.status != RunStatus::Optimal {
                a_unique.push(ra.instance_name.clone());
            }
            if rb.status == RunStatus::Optimal && ra.status != RunStatus::Optimal {
                b_unique.push(rb.instance_name.clone());
            }
        }

        ConfigComparison {
            config_a: config_a.to_string(),
            config_b: config_b.to_string(),
            aggregate_a: agg_a,
            aggregate_b: agg_b,
            pairwise,
            time_ratios,
            a_unique_solves: a_unique,
            b_unique_solves: b_unique,
        }
    }

    /// Geometric mean speedup of A over B.
    pub fn speedup(&self) -> f64 {
        self.pairwise.geo_mean_speedup
    }

    /// Summary of the comparison.
    pub fn summary(&self) -> String {
        format!(
            "{} vs {}: A wins {}, B wins {}, ties {}, speedup {:.2}x, p={:.4}",
            self.config_a,
            self.config_b,
            self.pairwise.a_wins,
            self.pairwise.b_wins,
            self.pairwise.ties,
            self.speedup(),
            self.pairwise.wilcoxon.p_value,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results(name: &str, times: &[f64]) -> Vec<RunResult> {
        times
            .iter()
            .enumerate()
            .map(|(i, &t)| RunResult::optimal(&format!("inst_{}", i), name, t, 0.0, 0, 0))
            .collect()
    }

    #[test]
    fn test_wilcoxon_equal() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wilcoxon_signed_rank(&a, &b);
        assert_eq!(result.n_nonzero, 0);
        assert!(!result.significant);
    }

    #[test]
    fn test_wilcoxon_different() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let result = wilcoxon_signed_rank(&a, &b);
        assert!(result.n_nonzero > 0);
        // a is consistently better.
        assert_eq!(result.direction, 1);
    }

    #[test]
    fn test_standard_normal_cdf() {
        let cdf0 = standard_normal_cdf(0.0);
        assert!((cdf0 - 0.5).abs() < 0.01);
        assert!(standard_normal_cdf(-10.0) < 0.001);
        assert!(standard_normal_cdf(10.0) > 0.999);
    }

    #[test]
    fn test_domination_analysis() {
        let mut results = HashMap::new();
        results.insert("fast".to_string(), make_results("fast", &[1.0, 2.0, 3.0]));
        results.insert(
            "slow".to_string(),
            make_results("slow", &[10.0, 20.0, 30.0]),
        );
        let dom = DomainAnalysis::compute(&results);
        assert!(dom.pareto_optimal.contains(&"fast".to_string()));
    }

    #[test]
    fn test_virtual_best_solver() {
        let mut results = HashMap::new();
        results.insert("A".to_string(), make_results("A", &[1.0, 5.0, 3.0]));
        results.insert("B".to_string(), make_results("B", &[3.0, 2.0, 4.0]));
        let vbs = VirtualBestSolver::compute(&results);
        assert_eq!(vbs.best_times.len(), 3);
        // Instance 0: A=1, B=3 → best=1
        assert!((vbs.best_times[0] - 1.0).abs() < 1e-10);
        // Instance 1: A=5, B=2 → best=2
        assert!((vbs.best_times[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_performance_profile_data() {
        let mut results = HashMap::new();
        results.insert("A".to_string(), make_results("A", &[1.0, 2.0, 5.0]));
        results.insert("B".to_string(), make_results("B", &[2.0, 1.0, 3.0]));
        let profiles = PerformanceProfileData::compute_all(&results);
        assert_eq!(profiles.len(), 2);
        for p in &profiles {
            assert!(!p.ratios.is_empty());
            assert!(p.efficiency >= 0.0 && p.efficiency <= 1.0);
        }
    }

    #[test]
    fn test_config_comparison() {
        let ra = make_results("A", &[1.0, 2.0, 3.0]);
        let rb = make_results("B", &[3.0, 2.0, 1.0]);
        let cmp = ConfigComparison::compare("A", "B", &ra, &rb);
        assert!(cmp.pairwise.a_wins > 0 || cmp.pairwise.b_wins > 0);
        assert!(!cmp.summary().is_empty());
    }

    #[test]
    fn test_comparison_report() {
        let mut results = HashMap::new();
        results.insert("A".to_string(), make_results("A", &[1.0, 2.0]));
        results.insert("B".to_string(), make_results("B", &[2.0, 1.0]));
        let report = ComparisonReport::build(&results);
        assert_eq!(report.config_metrics.len(), 2);
        assert!(!report.pairwise_tests.is_empty());
    }

    #[test]
    fn test_pairwise_display() {
        let ra = make_results("A", &[1.0, 2.0, 3.0]);
        let rb = make_results("B", &[3.0, 2.0, 1.0]);
        let test = PairwiseTest::wilcoxon_signed_rank("A", "B", &ra, &rb);
        let display = format!("{}", test);
        assert!(display.contains("A vs B"));
    }
}
