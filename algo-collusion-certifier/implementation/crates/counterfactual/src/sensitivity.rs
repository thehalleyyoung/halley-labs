//! Sensitivity analysis across parameter space.
//!
//! Evaluates robustness of counterfactual conclusions to parameter
//! variations: demand specification, discount factor, memory length,
//! and multi-dimensional parameter sweeps.

use shared_types::{
    AlgorithmType, CollusionError, CollusionResult, ConfidenceInterval, DemandSystem,
    GameConfig, MarketType, PlayerId, Price, PriceTrajectory,
    Profit, RoundNumber, SimulationConfig,
};
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::deviation::{deviation_incentive, DeviationProfile, OptimalDeviation};
use crate::market_helper::price_bounds_from_config;

// ── ParameterSweep ──────────────────────────────────────────────────────────

/// Vary one parameter at a time and observe effect on detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSweep {
    pub parameter_name: String,
    pub parameter_values: Vec<f64>,
    pub results: Vec<SweepResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub parameter_value: f64,
    pub deviation_incentive: f64,
    pub max_gain: f64,
    pub self_enforcing: bool,
    pub epsilon_nash_gap: f64,
}

impl ParameterSweep {
    pub fn new(name: &str, values: Vec<f64>) -> Self {
        Self {
            parameter_name: name.to_string(),
            parameter_values: values,
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: SweepResult) {
        self.results.push(result);
    }

    /// Find the critical threshold where detection changes.
    pub fn critical_threshold(&self) -> Option<f64> {
        for i in 1..self.results.len() {
            let prev = &self.results[i - 1];
            let curr = &self.results[i];
            if prev.self_enforcing != curr.self_enforcing {
                return Some((self.parameter_values[i - 1] + self.parameter_values[i]) / 2.0);
            }
        }
        None
    }

    /// Monotonicity: does the metric change monotonically with the parameter?
    pub fn is_monotone(&self) -> bool {
        if self.results.len() < 3 { return true; }
        let gains: Vec<f64> = self.results.iter().map(|r| r.max_gain).collect();
        let increasing = gains.windows(2).all(|w| w[1] >= w[0] - 1e-10);
        let decreasing = gains.windows(2).all(|w| w[1] <= w[0] + 1e-10);
        increasing || decreasing
    }

    /// Summary statistics across the sweep.
    pub fn summary(&self) -> SweepSummary {
        let gains: Vec<f64> = self.results.iter().map(|r| r.max_gain).collect();
        let min_gain = gains.iter().copied().fold(f64::INFINITY, f64::min);
        let max_gain = gains.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_gain = if gains.is_empty() { 0.0 }
            else { gains.iter().sum::<f64>() / gains.len() as f64 };

        let num_enforcing = self.results.iter().filter(|r| r.self_enforcing).count();

        SweepSummary {
            parameter_name: self.parameter_name.clone(),
            num_values: self.results.len(),
            min_gain,
            max_gain,
            mean_gain,
            range: max_gain - min_gain,
            fraction_self_enforcing: if self.results.is_empty() { 0.0 }
                else { num_enforcing as f64 / self.results.len() as f64 },
            is_monotone: self.is_monotone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepSummary {
    pub parameter_name: String,
    pub num_values: usize,
    pub min_gain: f64,
    pub max_gain: f64,
    pub mean_gain: f64,
    pub range: f64,
    pub fraction_self_enforcing: f64,
    pub is_monotone: bool,
}

// ── LatinHypercubeSampling ──────────────────────────────────────────────────

/// Latin Hypercube Sampling for multi-parameter sweeps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatinHypercubeSampling {
    pub dimensions: usize,
    pub num_samples: usize,
    pub bounds: Vec<(f64, f64)>,
    pub samples: Vec<Vec<f64>>,
}

impl LatinHypercubeSampling {
    /// Generate LHS design.
    pub fn generate(bounds: Vec<(f64, f64)>, num_samples: usize, seed: u64) -> Self {
        let dimensions = bounds.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut samples = Vec::with_capacity(num_samples);

        // Generate permutation indices for each dimension
        let mut perms: Vec<Vec<usize>> = (0..dimensions)
            .map(|_| {
                let mut p: Vec<usize> = (0..num_samples).collect();
                p.shuffle(&mut rng);
                p
            })
            .collect();

        for i in 0..num_samples {
            let mut point = Vec::with_capacity(dimensions);
            for d in 0..dimensions {
                let cell = perms[d][i];
                let (lo, hi) = bounds[d];
                let u: f64 = rng.gen();
                let value = lo + (cell as f64 + u) / num_samples as f64 * (hi - lo);
                point.push(value);
            }
            samples.push(point);
        }

        Self {
            dimensions,
            num_samples,
            bounds,
            samples,
        }
    }

    /// Get the i-th sample point.
    pub fn point(&self, i: usize) -> &[f64] {
        &self.samples[i]
    }

    /// Check that coverage is good: each cell in each dimension appears once.
    pub fn coverage_score(&self) -> f64 {
        let mut score = 0.0;
        for d in 0..self.dimensions {
            let mut cells_occupied = vec![false; self.num_samples];
            for sample in &self.samples {
                let (lo, hi) = self.bounds[d];
                let cell = ((sample[d] - lo) / (hi - lo) * self.num_samples as f64) as usize;
                let cell = cell.min(self.num_samples - 1);
                cells_occupied[cell] = true;
            }
            let fraction_covered = cells_occupied.iter().filter(|&&b| b).count() as f64
                / self.num_samples as f64;
            score += fraction_covered;
        }
        score / self.dimensions as f64
    }
}

// ── SobolIndices ────────────────────────────────────────────────────────────

/// Global sensitivity analysis via Sobol' indices.
///
/// Decomposes output variance into contributions from each input parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolIndices {
    /// First-order indices S_i: main effect of parameter i.
    pub first_order: Vec<f64>,
    /// Total-order indices ST_i: total effect including interactions.
    pub total_order: Vec<f64>,
    /// Parameter names.
    pub parameter_names: Vec<String>,
    /// Total output variance.
    pub total_variance: f64,
    /// Number of model evaluations used.
    pub num_evaluations: usize,
}

impl SobolIndices {
    /// Estimate Sobol indices using Saltelli's scheme.
    ///
    /// Requires `model_fn` that maps parameter vector -> scalar output.
    pub fn estimate(
        bounds: &[(f64, f64)],
        parameter_names: Vec<String>,
        model_fn: &dyn Fn(&[f64]) -> f64,
        num_samples: usize,
        seed: u64,
    ) -> Self {
        let k = bounds.len();
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate two independent LHS matrices A and B
        let a_lhs = LatinHypercubeSampling::generate(bounds.to_vec(), num_samples, seed);
        let b_lhs = LatinHypercubeSampling::generate(bounds.to_vec(), num_samples, seed + 1);

        // Evaluate model on A and B
        let f_a: Vec<f64> = a_lhs.samples.iter().map(|s| model_fn(s)).collect();
        let f_b: Vec<f64> = b_lhs.samples.iter().map(|s| model_fn(s)).collect();

        let mean_all: f64 = f_a.iter().chain(f_b.iter()).sum::<f64>()
            / (f_a.len() + f_b.len()) as f64;
        let total_var: f64 = f_a.iter().chain(f_b.iter())
            .map(|y| (y - mean_all).powi(2))
            .sum::<f64>() / (f_a.len() + f_b.len() - 1) as f64;

        let mut first_order = vec![0.0; k];
        let mut total_order = vec![0.0; k];

        // For each parameter i, create matrix AB_i (B with column i from A)
        for i in 0..k {
            let mut f_ab_i = Vec::with_capacity(num_samples);

            for j in 0..num_samples {
                let mut point = b_lhs.samples[j].clone();
                point[i] = a_lhs.samples[j][i]; // Replace column i with A's
                f_ab_i.push(model_fn(&point));
            }

            // First-order estimate: S_i = V_i / V(Y)
            // V_i ≈ (1/N) Σ f_A * (f_AB_i - f_B)
            let v_i: f64 = f_a.iter()
                .zip(f_ab_i.iter())
                .zip(f_b.iter())
                .map(|((fa, fab), fb)| fa * (fab - fb))
                .sum::<f64>() / num_samples as f64;

            first_order[i] = if total_var > 1e-15 { (v_i / total_var).clamp(-1.0, 1.0) } else { 0.0 };

            // Total-order estimate: ST_i = 1 - V_{~i} / V(Y)
            let v_not_i: f64 = f_b.iter()
                .zip(f_ab_i.iter())
                .zip(f_a.iter())
                .map(|((fb, fab), fa)| fb * (fab - fa))
                .sum::<f64>() / num_samples as f64;

            total_order[i] = if total_var > 1e-15 {
                (1.0 - v_not_i / total_var).clamp(0.0, 2.0)
            } else {
                0.0
            };
        }

        let num_evals = num_samples * (k + 2);

        Self {
            first_order,
            total_order,
            parameter_names,
            total_variance: total_var,
            num_evaluations: num_evals,
        }
    }

    /// Which parameter has the largest first-order effect?
    pub fn most_influential(&self) -> Option<(usize, &str, f64)> {
        self.first_order.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (i, self.parameter_names[i].as_str(), s))
    }

    /// Sum of first-order indices (should be ≤ 1 for well-behaved models).
    pub fn first_order_sum(&self) -> f64 {
        self.first_order.iter().sum()
    }

    /// Interaction effects: total - first_order for each parameter.
    pub fn interaction_effects(&self) -> Vec<f64> {
        self.total_order.iter()
            .zip(self.first_order.iter())
            .map(|(t, f)| (t - f).max(0.0))
            .collect()
    }
}

// ── SensitivityAnalyzer ─────────────────────────────────────────────────────

/// Main sensitivity analysis engine.
pub struct SensitivityAnalyzer {
    base_game_config: GameConfig,
    base_sim_config: SimulationConfig,
    grid_size: usize,
    epsilon: f64,
}

impl SensitivityAnalyzer {
    pub fn new(
        game_config: GameConfig,
        sim_config: SimulationConfig,
        grid_size: usize,
        epsilon: f64,
    ) -> Self {
        Self {
            base_game_config: game_config,
            base_sim_config: sim_config,
            grid_size,
            epsilon,
        }
    }

    /// Sweep discount factor and measure detection.
    pub fn discount_factor_sensitivity(
        &self,
        trajectory: &PriceTrajectory,
        delta_values: &[f64],
    ) -> ParameterSweep {
        let mut sweep = ParameterSweep::new("discount_factor", delta_values.to_vec());

        for &delta in delta_values {
            let mut gc = self.base_game_config.clone();
            gc.discount_factor = delta;

            let inc = deviation_incentive(
                PlayerId(0), trajectory, &gc,
                price_bounds_from_config(&gc), self.grid_size,
            );

            let max_gain = inc;
            let self_enforcing = max_gain <= self.epsilon;

            sweep.add_result(SweepResult {
                parameter_value: delta,
                deviation_incentive: inc,
                max_gain,
                self_enforcing,
                epsilon_nash_gap: max_gain,
            });
        }

        sweep
    }

    /// Sweep memory length and measure detection.
    pub fn memory_length_sensitivity(
        &self,
        trajectory: &PriceTrajectory,
        memory_values: &[usize],
    ) -> ParameterSweep {
        let values: Vec<f64> = memory_values.iter().map(|&m| m as f64).collect();
        let mut sweep = ParameterSweep::new("memory_length", values);

        for &mem in memory_values {
            // Truncate trajectory to simulate limited memory
            let effective_len = trajectory.len().min(mem);
            let start_idx = trajectory.len().saturating_sub(effective_len);
            let sub_outcomes: Vec<shared_types::MarketOutcome> =
                trajectory.outcomes[start_idx..].to_vec();
            let trunc = PriceTrajectory::new(
                sub_outcomes,
                trajectory.market_type,
                trajectory.num_players,
                AlgorithmType::QLearning,
                0,
            );

            let inc = deviation_incentive(
                PlayerId(0), &trunc, &self.base_game_config,
                price_bounds_from_config(&self.base_game_config), self.grid_size,
            );

            let self_enforcing = inc <= self.epsilon;

            sweep.add_result(SweepResult {
                parameter_value: mem as f64,
                deviation_incentive: inc,
                max_gain: inc,
                self_enforcing,
                epsilon_nash_gap: inc,
            });
        }

        sweep
    }

    /// Test robustness across different demand specifications.
    pub fn demand_robustness(
        &self,
        trajectory: &PriceTrajectory,
        demand_specs: &[DemandSystem],
    ) -> Vec<SweepResult> {
        let mut results = Vec::new();

        for (idx, demand) in demand_specs.iter().enumerate() {
            let mut gc = self.base_game_config.clone();
            gc.demand_system = demand.clone();

            let inc = deviation_incentive(
                PlayerId(0), trajectory, &gc,
                price_bounds_from_config(&gc), self.grid_size,
            );

            results.push(SweepResult {
                parameter_value: idx as f64,
                deviation_incentive: inc,
                max_gain: inc,
                self_enforcing: inc <= self.epsilon,
                epsilon_nash_gap: inc,
            });
        }

        results
    }

    /// Run full sensitivity analysis across multiple dimensions.
    pub fn full_analysis(
        &self,
        trajectory: &PriceTrajectory,
    ) -> SensitivityReport {
        // Discount factor sweep
        let delta_values: Vec<f64> = (0..11).map(|i| 0.5 + i as f64 * 0.05).collect();
        let discount_sweep = self.discount_factor_sensitivity(trajectory, &delta_values);

        // Memory length sweep
        let mem_values: Vec<usize> = vec![5, 10, 20, 50, 100, 200, 500];
        let memory_sweep = self.memory_length_sensitivity(
            trajectory,
            &mem_values.iter().copied().filter(|&m| m <= trajectory.len()).collect::<Vec<_>>(),
        );

        // Cross-slope sensitivity (demand robustness proxy)
        let cross_slopes: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
        let mut demand_sweep = ParameterSweep::new("cross_slope", cross_slopes.clone());
        for &cs in &cross_slopes {
            let mut gc = self.base_game_config.clone();
            gc.demand_system = DemandSystem::Linear {
                max_quantity: 10.0,
                slope: cs,
            };
            let inc = deviation_incentive(
                PlayerId(0), trajectory, &gc,
                price_bounds_from_config(&gc), self.grid_size,
            );
            demand_sweep.add_result(SweepResult {
                parameter_value: cs,
                deviation_incentive: inc,
                max_gain: inc,
                self_enforcing: inc <= self.epsilon,
                epsilon_nash_gap: inc,
            });
        }

        // Robustness score
        let all_sweeps = vec![&discount_sweep, &memory_sweep, &demand_sweep];
        let robustness = RobustnessScore::compute(&all_sweeps);

        SensitivityReport {
            discount_factor_sweep: discount_sweep,
            memory_length_sweep: memory_sweep,
            demand_sweep,
            robustness_score: robustness,
            num_evaluations: delta_values.len() + mem_values.len() + cross_slopes.len(),
        }
    }
}

// ── SensitivityReport ───────────────────────────────────────────────────────

/// Structured sensitivity analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityReport {
    pub discount_factor_sweep: ParameterSweep,
    pub memory_length_sweep: ParameterSweep,
    pub demand_sweep: ParameterSweep,
    pub robustness_score: RobustnessScore,
    pub num_evaluations: usize,
}

impl SensitivityReport {
    /// Which parameter is most sensitive?
    pub fn most_sensitive_parameter(&self) -> &str {
        let sweeps = [
            (&self.discount_factor_sweep, "discount_factor"),
            (&self.memory_length_sweep, "memory_length"),
            (&self.demand_sweep, "demand"),
        ];

        sweeps.iter()
            .max_by(|a, b| {
                let range_a = a.0.summary().range;
                let range_b = b.0.summary().range;
                range_a.partial_cmp(&range_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, name)| *name)
            .unwrap_or("unknown")
    }

    /// Are conclusions robust across all parameter variations?
    pub fn conclusions_robust(&self, threshold: f64) -> bool {
        self.robustness_score.overall >= threshold
    }
}

// ── RobustnessScore ─────────────────────────────────────────────────────────

/// Aggregate robustness metric across all sensitivity dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessScore {
    pub overall: f64,
    pub per_parameter: Vec<(String, f64)>,
    pub interpretation: String,
}

impl RobustnessScore {
    /// Compute robustness score from a set of parameter sweeps.
    pub fn compute(sweeps: &[&ParameterSweep]) -> Self {
        let mut per_param = Vec::new();

        for sweep in sweeps {
            let summary = sweep.summary();
            // Robustness = fraction of parameter values where conclusion holds
            let score = summary.fraction_self_enforcing;
            per_param.push((summary.parameter_name.clone(), score));
        }

        let overall = if per_param.is_empty() {
            0.0
        } else {
            per_param.iter().map(|(_, s)| s).sum::<f64>() / per_param.len() as f64
        };

        let interpretation = if overall >= 0.9 {
            "Highly robust: conclusions hold across >90% of parameter variations."
        } else if overall >= 0.7 {
            "Moderately robust: conclusions hold across 70-90% of variations."
        } else if overall >= 0.5 {
            "Marginally robust: conclusions are sensitive to parameter choices."
        } else {
            "Not robust: conclusions change significantly with parameters."
        };

        Self {
            overall,
            per_parameter: per_param,
            interpretation: interpretation.to_string(),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{
        AlgorithmConfig, AlgorithmType, Cost, EvaluationMode,
        MarketOutcome, PlayerAction, Price, Profit, Quantity, RoundNumber,
    };

    fn make_trajectory(n_rounds: usize, n_players: usize, price: f64) -> PriceTrajectory {
        let outcomes: Vec<MarketOutcome> = (0..n_rounds)
            .map(|r| {
                let actions: Vec<PlayerAction> = (0..n_players)
                    .map(|p| PlayerAction::new(PlayerId(p), Price(price)))
                    .collect();
                MarketOutcome::new(
                    RoundNumber(r),
                    actions,
                    vec![Price(price); n_players],
                    vec![Quantity(1.0); n_players],
                    vec![Profit((price - 1.0) * 1.0); n_players],
                )
            })
            .collect();
        PriceTrajectory::new(
            outcomes,
            MarketType::Bertrand,
            n_players,
            AlgorithmType::QLearning,
            0,
        )
    }

    fn make_configs() -> (GameConfig, SimulationConfig) {
        let gc = GameConfig {
            market_type: MarketType::Bertrand,
            demand_system: DemandSystem::Linear {
                max_quantity: 10.0,
                slope: 1.0,
            },
            num_players: 2,
            discount_factor: 0.95,
            marginal_costs: vec![Cost(1.0), Cost(1.0)],
            price_grid: None,
            max_rounds: 1000,
            description: String::new(),
        };
        let sc = SimulationConfig::new(
            gc.clone(),
            AlgorithmConfig::new(AlgorithmType::QLearning),
            EvaluationMode::Standard,
        );
        (gc, sc)
    }

    #[test]
    fn test_parameter_sweep_creation() {
        let sweep = ParameterSweep::new("test", vec![1.0, 2.0, 3.0]);
        assert_eq!(sweep.parameter_name, "test");
        assert_eq!(sweep.parameter_values.len(), 3);
    }

    #[test]
    fn test_parameter_sweep_threshold() {
        let mut sweep = ParameterSweep::new("delta", vec![0.5, 0.7, 0.9]);
        sweep.add_result(SweepResult {
            parameter_value: 0.5,
            deviation_incentive: 2.0,
            max_gain: 2.0,
            self_enforcing: false,
            epsilon_nash_gap: 2.0,
        });
        sweep.add_result(SweepResult {
            parameter_value: 0.7,
            deviation_incentive: 0.5,
            max_gain: 0.5,
            self_enforcing: false,
            epsilon_nash_gap: 0.5,
        });
        sweep.add_result(SweepResult {
            parameter_value: 0.9,
            deviation_incentive: 0.001,
            max_gain: 0.001,
            self_enforcing: true,
            epsilon_nash_gap: 0.001,
        });
        let threshold = sweep.critical_threshold();
        assert!(threshold.is_some());
        assert!(threshold.unwrap() > 0.7);
    }

    #[test]
    fn test_sweep_summary() {
        let mut sweep = ParameterSweep::new("test", vec![1.0, 2.0]);
        sweep.add_result(SweepResult {
            parameter_value: 1.0, deviation_incentive: 0.5,
            max_gain: 0.5, self_enforcing: false, epsilon_nash_gap: 0.5,
        });
        sweep.add_result(SweepResult {
            parameter_value: 2.0, deviation_incentive: 0.3,
            max_gain: 0.3, self_enforcing: true, epsilon_nash_gap: 0.3,
        });
        let summary = sweep.summary();
        assert_eq!(summary.num_values, 2);
        assert!(summary.range > 0.0);
    }

    #[test]
    fn test_lhs_generation() {
        let bounds = vec![(0.0, 1.0), (0.0, 10.0), (-5.0, 5.0)];
        let lhs = LatinHypercubeSampling::generate(bounds.clone(), 50, 42);
        assert_eq!(lhs.num_samples, 50);
        assert_eq!(lhs.dimensions, 3);
        // All samples within bounds
        for sample in &lhs.samples {
            for (d, &val) in sample.iter().enumerate() {
                assert!(val >= bounds[d].0 && val <= bounds[d].1);
            }
        }
    }

    #[test]
    fn test_lhs_coverage() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let lhs = LatinHypercubeSampling::generate(bounds, 100, 42);
        let coverage = lhs.coverage_score();
        assert!(coverage > 0.8); // should have good coverage
    }

    #[test]
    fn test_sobol_indices() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let names = vec!["x1".to_string(), "x2".to_string()];
        // f(x1, x2) = x1 (only depends on x1)
        let model = |x: &[f64]| -> f64 { x[0] };
        let sobol = SobolIndices::estimate(&bounds, names, &model, 500, 42);
        assert_eq!(sobol.first_order.len(), 2);
        // x1 should have high first-order index
        assert!(sobol.first_order[0] > sobol.first_order[1]);
    }

    #[test]
    fn test_sobol_most_influential() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let names = vec!["x1".to_string(), "x2".to_string()];
        let model = |x: &[f64]| -> f64 { x[0] * 10.0 + x[1] };
        let sobol = SobolIndices::estimate(&bounds, names, &model, 500, 42);
        let (idx, name, _) = sobol.most_influential().unwrap();
        assert_eq!(name, "x1");
    }

    #[test]
    fn test_sobol_interaction_effects() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let names = vec!["x1".to_string(), "x2".to_string()];
        let model = |x: &[f64]| -> f64 { x[0] + x[1] }; // additive, no interaction
        let sobol = SobolIndices::estimate(&bounds, names, &model, 500, 42);
        let interactions = sobol.interaction_effects();
        // Interactions should be small for additive model
        for ie in &interactions {
            assert!(*ie < 0.5);
        }
    }

    #[test]
    fn test_discount_factor_sensitivity() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, 5.0);
        let analyzer = SensitivityAnalyzer::new(gc, sc, 20, 0.01);
        let sweep = analyzer.discount_factor_sensitivity(&trajectory, &[0.5, 0.7, 0.95]);
        assert_eq!(sweep.results.len(), 3);
    }

    #[test]
    fn test_memory_length_sensitivity() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(100, 2, 5.0);
        let analyzer = SensitivityAnalyzer::new(gc, sc, 20, 0.01);
        let sweep = analyzer.memory_length_sensitivity(&trajectory, &[10, 50, 100]);
        assert_eq!(sweep.results.len(), 3);
    }

    #[test]
    fn test_demand_robustness() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, 5.0);
        let analyzer = SensitivityAnalyzer::new(gc, sc, 20, 0.01);
        let demands = vec![
            DemandSystem::Linear { max_quantity: 10.0, slope: 0.3 },
            DemandSystem::Linear { max_quantity: 10.0, slope: 0.7 },
        ];
        let results = analyzer.demand_robustness(&trajectory, &demands);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_full_analysis() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, 5.0);
        let analyzer = SensitivityAnalyzer::new(gc, sc, 10, 0.01);
        let report = analyzer.full_analysis(&trajectory);
        assert!(!report.discount_factor_sweep.results.is_empty());
        assert!(!report.demand_sweep.results.is_empty());
        assert!(report.robustness_score.overall >= 0.0);
    }

    #[test]
    fn test_robustness_score() {
        let mut sweep1 = ParameterSweep::new("s1", vec![1.0]);
        sweep1.add_result(SweepResult {
            parameter_value: 1.0, deviation_incentive: 0.001,
            max_gain: 0.001, self_enforcing: true, epsilon_nash_gap: 0.001,
        });
        let mut sweep2 = ParameterSweep::new("s2", vec![1.0]);
        sweep2.add_result(SweepResult {
            parameter_value: 1.0, deviation_incentive: 0.5,
            max_gain: 0.5, self_enforcing: false, epsilon_nash_gap: 0.5,
        });
        let score = RobustnessScore::compute(&[&sweep1, &sweep2]);
        assert!((score.overall - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sensitivity_report_most_sensitive() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, 5.0);
        let analyzer = SensitivityAnalyzer::new(gc, sc, 10, 0.01);
        let report = analyzer.full_analysis(&trajectory);
        let most_sensitive = report.most_sensitive_parameter();
        assert!(!most_sensitive.is_empty());
    }

    #[test]
    fn test_conclusions_robust() {
        let score = RobustnessScore {
            overall: 0.95,
            per_parameter: vec![("delta".to_string(), 0.95)],
            interpretation: "Highly robust".to_string(),
        };
        let report = SensitivityReport {
            discount_factor_sweep: ParameterSweep::new("delta", vec![]),
            memory_length_sweep: ParameterSweep::new("memory", vec![]),
            demand_sweep: ParameterSweep::new("demand", vec![]),
            robustness_score: score,
            num_evaluations: 0,
        };
        assert!(report.conclusions_robust(0.9));
        assert!(!report.conclusions_robust(0.99));
    }

    #[test]
    fn test_monotonicity_check() {
        let mut sweep = ParameterSweep::new("test", vec![1.0, 2.0, 3.0]);
        sweep.add_result(SweepResult {
            parameter_value: 1.0, deviation_incentive: 1.0,
            max_gain: 3.0, self_enforcing: false, epsilon_nash_gap: 3.0,
        });
        sweep.add_result(SweepResult {
            parameter_value: 2.0, deviation_incentive: 0.5,
            max_gain: 2.0, self_enforcing: false, epsilon_nash_gap: 2.0,
        });
        sweep.add_result(SweepResult {
            parameter_value: 3.0, deviation_incentive: 0.1,
            max_gain: 1.0, self_enforcing: false, epsilon_nash_gap: 1.0,
        });
        assert!(sweep.is_monotone());
    }
}
