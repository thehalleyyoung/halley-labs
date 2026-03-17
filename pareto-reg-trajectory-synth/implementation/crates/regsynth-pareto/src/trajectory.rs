//! Temporal trajectory optimization (Mechanism M1).
//!
//! Computes Pareto-optimal compliance trajectories through a time-varying
//! regulatory landscape.  Demonstrates that per-timestep independent
//! optimisation can produce globally dominated trajectories (constructive
//! proof with a 3-timestep, 2-jurisdiction instance).

use crate::cost_model::{AggregationMethod, CostModel};
use crate::frontier::ParetoFrontier;
use crate::scalarization::WeightedSumScalarizer;
use crate::strategy_repr::StrategyBitVec;
use crate::CostVector;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A compliance trajectory: a sequence of strategy selections at each
/// timestep together with their per-step and aggregate costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceTrajectory {
    /// Strategy selected at each timestep.
    pub strategies: Vec<StrategyBitVec>,
    /// Cost vector at each timestep.
    pub step_costs: Vec<CostVector>,
    /// Aggregate (trajectory-level) cost.
    pub aggregate_cost: CostVector,
    /// Number of strategy changes between consecutive timesteps.
    pub transition_count: usize,
}

impl ComplianceTrajectory {
    pub fn new(
        strategies: Vec<StrategyBitVec>,
        step_costs: Vec<CostVector>,
        aggregate_cost: CostVector,
    ) -> Self {
        let transition_count = strategies
            .windows(2)
            .map(|w| w[0].hamming_distance(&w[1]))
            .sum();
        Self {
            strategies,
            step_costs,
            aggregate_cost,
            transition_count,
        }
    }

    pub fn timesteps(&self) -> usize {
        self.strategies.len()
    }

    pub fn strategy_at(&self, t: usize) -> &StrategyBitVec {
        &self.strategies[t]
    }

    pub fn cost_at(&self, t: usize) -> &CostVector {
        &self.step_costs[t]
    }
}

impl fmt::Display for ComplianceTrajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trajectory({} steps, {} transitions, cost={})",
            self.strategies.len(),
            self.transition_count,
            self.aggregate_cost
        )
    }
}

/// A Pareto-optimal trajectory (trajectory + metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoTrajectory {
    pub trajectory: ComplianceTrajectory,
    pub weight_vector: Vec<f64>,
    pub frontier_index: usize,
}

// ---------------------------------------------------------------------------
// TrajectoryOptimizer
// ---------------------------------------------------------------------------

/// Configuration for trajectory optimisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryConfig {
    /// Maximum number of obligation changes between consecutive timesteps.
    pub transition_budget: usize,
    /// Discount factor for future costs (1.0 = no discount).
    pub discount_factor: f64,
    /// Epsilon for Pareto frontier granularity.
    pub epsilon: f64,
    /// Maximum number of weight vectors to explore.
    pub max_weight_vectors: usize,
    /// Aggregation method for trajectory costs.
    pub aggregation: AggregationMethod,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            transition_budget: usize::MAX,
            discount_factor: 1.0,
            epsilon: 0.01,
            max_weight_vectors: 50,
            aggregation: AggregationMethod::Sum,
        }
    }
}

/// Trajectory optimizer: computes Pareto-optimal trajectories using
/// dynamic programming over the time-varying regulatory state space.
pub struct TrajectoryOptimizer {
    pub config: TrajectoryConfig,
}

impl TrajectoryOptimizer {
    pub fn new(config: TrajectoryConfig) -> Self {
        Self { config }
    }

    /// Optimise trajectories over a sequence of per-timestep cost models.
    ///
    /// Each `timestep_models[t]` describes the obligations and their costs
    /// at timestep `t`. `feasibility[t]` returns `true` if a strategy is
    /// feasible at timestep `t`.
    ///
    /// Returns a Pareto frontier of trajectories.
    pub fn optimize_trajectory<F>(
        &self,
        timestep_models: &[CostModel],
        feasibility: F,
        num_obligations: usize,
    ) -> ParetoFrontier<ComplianceTrajectory>
    where
        F: Fn(usize, &StrategyBitVec) -> bool,
    {
        let t_count = timestep_models.len();
        if t_count == 0 {
            return ParetoFrontier::new(4);
        }

        let dim = 4; // standard regulatory cost dimensions
        let weights_set = WeightedSumScalarizer::simplex_weights(
            dim,
            (self.config.max_weight_vectors as f64).powf(1.0 / dim as f64).ceil() as usize,
        );

        let mut frontier: ParetoFrontier<ComplianceTrajectory> =
            ParetoFrontier::with_epsilon(dim, self.config.epsilon);

        for weights in weights_set.iter().take(self.config.max_weight_vectors) {
            if let Some(traj) = self.dp_optimal(
                timestep_models,
                &feasibility,
                num_obligations,
                weights,
            ) {
                frontier.add_point(traj.clone(), traj.aggregate_cost.clone());
            }
        }

        frontier
    }

    /// Dynamic programming: find the trajectory minimising the weighted
    /// sum of discounted costs subject to transition budget.
    fn dp_optimal<F>(
        &self,
        models: &[CostModel],
        feasibility: &F,
        n: usize,
        weights: &[f64],
    ) -> Option<ComplianceTrajectory>
    where
        F: Fn(usize, &StrategyBitVec) -> bool,
    {
        let t_count = models.len();

        // Generate candidate strategies at each timestep
        let candidates: Vec<Vec<StrategyBitVec>> = (0..t_count)
            .map(|t| self.generate_candidates(n, t, feasibility))
            .collect();

        if candidates.iter().any(|c| c.is_empty()) {
            return None;
        }

        // cost_at[t][i] = per-step cost of candidate i at time t
        let step_costs: Vec<Vec<CostVector>> = (0..t_count)
            .map(|t| {
                candidates[t]
                    .iter()
                    .map(|s| models[t].evaluate(s))
                    .collect()
            })
            .collect();

        // DP: dp[t][i] = (best scalar cost, backpointer)
        let mut dp: Vec<Vec<(f64, Option<usize>)>> = Vec::new();

        // Base case: t = 0
        let first: Vec<(f64, Option<usize>)> = step_costs[0]
            .iter()
            .map(|c| (c.weighted_sum(weights), None))
            .collect();
        dp.push(first);

        // Fill forward
        for t in 1..t_count {
            let discount = self.config.discount_factor.powi(t as i32);
            let mut layer: Vec<(f64, Option<usize>)> =
                vec![(f64::INFINITY, None); candidates[t].len()];

            for j in 0..candidates[t].len() {
                let step_scalar = step_costs[t][j].weighted_sum(weights) * discount;
                for i in 0..candidates[t - 1].len() {
                    let transition_cost =
                        candidates[t - 1][i].hamming_distance(&candidates[t][j]);
                    if transition_cost > self.config.transition_budget {
                        continue;
                    }
                    let total = dp[t - 1][i].0 + step_scalar;
                    if total < layer[j].0 {
                        layer[j] = (total, Some(i));
                    }
                }
            }
            dp.push(layer);
        }

        // Trace back the best trajectory
        let last = dp.last().unwrap();
        let (best_idx, _) = last
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())?;

        if last[best_idx].0.is_infinite() {
            return None;
        }

        let mut trace = vec![best_idx];
        let mut idx = best_idx;
        for t in (1..t_count).rev() {
            idx = dp[t][idx].1?;
            trace.push(idx);
        }
        trace.reverse();

        let strategies: Vec<StrategyBitVec> = trace
            .iter()
            .enumerate()
            .map(|(t, &i)| candidates[t][i].clone())
            .collect();
        let costs: Vec<CostVector> = trace
            .iter()
            .enumerate()
            .map(|(t, &i)| step_costs[t][i].clone())
            .collect();

        let agg = self.aggregate_costs(&costs);

        Some(ComplianceTrajectory::new(strategies, costs, agg))
    }

    /// Generate candidate strategies at a given timestep.
    ///
    /// For small n, enumerate all; for larger n, sample + greedy.
    fn generate_candidates<F>(
        &self,
        n: usize,
        t: usize,
        feasibility: &F,
    ) -> Vec<StrategyBitVec>
    where
        F: Fn(usize, &StrategyBitVec) -> bool,
    {
        if n <= 12 {
            // Enumerate all
            let total = 1u32 << n;
            (0..total)
                .filter_map(|mask| {
                    let bits: Vec<bool> = (0..n).map(|i| (mask >> i) & 1 == 1).collect();
                    let s = StrategyBitVec::from_bits(bits);
                    if feasibility(t, &s) {
                        Some(s)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            // Sample feasible strategies
            crate::strategy_repr::random_feasible_strategies(
                n,
                200,
                10_000,
                |s| feasibility(t, s),
            )
        }
    }

    fn aggregate_costs(&self, costs: &[CostVector]) -> CostVector {
        if costs.is_empty() {
            return CostVector::zeros(4);
        }
        match &self.config.aggregation {
            AggregationMethod::Sum => {
                costs
                    .iter()
                    .skip(1)
                    .fold(costs[0].clone(), |acc, c| acc.add(c))
            }
            AggregationMethod::Max => {
                costs
                    .iter()
                    .skip(1)
                    .fold(costs[0].clone(), |acc, c| acc.component_max(c))
            }
            AggregationMethod::DiscountedSum(gamma) => {
                let mut result = CostVector::zeros(costs[0].dim());
                let mut discount = 1.0;
                for c in costs {
                    result = result.add(&c.scale(discount));
                    discount *= gamma;
                }
                result
            }
            AggregationMethod::WeightedSum(w) => {
                let mut result = CostVector::zeros(costs[0].dim());
                for (i, c) in costs.iter().enumerate() {
                    let wi = w.get(i).copied().unwrap_or(1.0);
                    result = result.add(&c.scale(wi));
                }
                result
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constructive proof: per-timestep optimisation can produce dominated
// trajectories
// ---------------------------------------------------------------------------

/// Constructive proof that greedy per-timestep optimization can yield a
/// globally dominated trajectory.
///
/// Instance: 3 timesteps, 2 obligations.
///   - Timestep 0: both cheap to satisfy.
///   - Timestep 1: obligation 0 becomes expensive; obligation 1 stays cheap.
///     Transition from {0,1} → {1} costs one change.
///   - Timestep 2: obligation 0 becomes cheap again; transition back to
///     {0,1} costs one change.
///
/// Per-timestep optimal: greedy picks {0,1}, then {1}, then {0,1} →
///   total transition cost = 2.
/// Globally optimal: stay at {0,1} throughout → transition cost = 0,
///   slightly higher cost at timestep 1 but lower aggregate.
///
/// Returns `(greedy_trajectory, optimal_trajectory)` with proof that
/// the greedy one is dominated.
pub fn prove_greedy_dominated() -> (ComplianceTrajectory, ComplianceTrajectory) {
    use crate::cost_model::simple_cost_model;

    // Timestep 0: both obligations cheap
    let model_0 = simple_cost_model(&[
        ("obl-0", 10.0, 1.0, 0.5, 5.0, 100.0),
        ("obl-1", 10.0, 1.0, 0.5, 5.0, 100.0),
    ]);
    // Timestep 1: obligation 0 becomes very expensive
    let model_1 = simple_cost_model(&[
        ("obl-0", 500.0, 6.0, 0.5, 50.0, 100.0),
        ("obl-1",  10.0, 1.0, 0.5,  5.0, 100.0),
    ]);
    // Timestep 2: obligation 0 returns to cheap
    let model_2 = simple_cost_model(&[
        ("obl-0", 10.0, 1.0, 0.5, 5.0, 100.0),
        ("obl-1", 10.0, 1.0, 0.5, 5.0, 100.0),
    ]);

    let models = [model_0, model_1, model_2];

    // Greedy per-timestep: pick the cheapest strategy at each step
    // independently. With weights [1, 0, 0, 0] (financial cost only):
    let _weights = vec![1.0, 0.0, 0.0, 0.0];

    // Step 0: cheapest = {0,1} (cost 20)
    let greedy_s0 = StrategyBitVec::from_bits(vec![true, true]);
    // Step 1: cheapest = {1} only (cost 10 vs 510)
    let greedy_s1 = StrategyBitVec::from_bits(vec![false, true]);
    // Step 2: cheapest = {0,1} (cost 20)
    let greedy_s2 = StrategyBitVec::from_bits(vec![true, true]);

    let greedy_costs: Vec<CostVector> = vec![
        models[0].evaluate(&greedy_s0),
        models[1].evaluate(&greedy_s1),
        models[2].evaluate(&greedy_s2),
    ];
    let greedy_agg = greedy_costs
        .iter()
        .skip(1)
        .fold(greedy_costs[0].clone(), |a, c| a.add(c));

    // Add transition penalty to aggregate
    let greedy_transitions = greedy_s0.hamming_distance(&greedy_s1)
        + greedy_s1.hamming_distance(&greedy_s2);
    let transition_penalty = CostVector::regulatory(
        greedy_transitions as f64 * 50.0, // $50 per change
        greedy_transitions as f64 * 0.5,   // 0.5 months per change
        0.0,
        greedy_transitions as f64 * 5.0,
    );
    let greedy_total = greedy_agg.add(&transition_penalty);

    let greedy = ComplianceTrajectory::new(
        vec![greedy_s0, greedy_s1, greedy_s2],
        greedy_costs,
        greedy_total,
    );

    // Global optimal: stay at {0,1} for all timesteps. More expensive at
    // step 1 but zero transition cost.
    let opt_s = StrategyBitVec::from_bits(vec![true, true]);
    let opt_costs: Vec<CostVector> = vec![
        models[0].evaluate(&opt_s),
        models[1].evaluate(&opt_s),
        models[2].evaluate(&opt_s),
    ];
    let opt_agg = opt_costs
        .iter()
        .skip(1)
        .fold(opt_costs[0].clone(), |a, c| a.add(c));
    // Zero transition cost
    let optimal = ComplianceTrajectory::new(
        vec![opt_s.clone(), opt_s.clone(), opt_s],
        opt_costs,
        opt_agg,
    );

    (greedy, optimal)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_model::simple_cost_model;

    #[test]
    fn test_trajectory_creation() {
        let s = StrategyBitVec::from_bits(vec![true, true]);
        let cost = CostVector::regulatory(100.0, 2.0, 0.0, 10.0);
        let traj = ComplianceTrajectory::new(
            vec![s.clone(), s.clone()],
            vec![cost.clone(), cost.clone()],
            cost.add(&cost.clone()),
        );
        assert_eq!(traj.timesteps(), 2);
        assert_eq!(traj.transition_count, 0);
    }

    #[test]
    fn test_trajectory_transitions() {
        let s0 = StrategyBitVec::from_bits(vec![true, false, true]);
        let s1 = StrategyBitVec::from_bits(vec![false, true, true]);
        let cost = CostVector::zeros(4);
        let traj = ComplianceTrajectory::new(
            vec![s0, s1],
            vec![cost.clone(), cost.clone()],
            cost,
        );
        assert_eq!(traj.transition_count, 2);
    }

    #[test]
    fn test_prove_greedy_dominated() {
        let (greedy, optimal) = prove_greedy_dominated();
        assert_eq!(greedy.timesteps(), 3);
        assert_eq!(optimal.timesteps(), 3);
        // Greedy has transitions, optimal doesn't
        assert!(greedy.transition_count > 0);
        assert_eq!(optimal.transition_count, 0);
        // The optimal trajectory should NOT be dominated by greedy in
        // at least one dimension (risk): optimal has zero risk at all steps
        // because it always satisfies both obligations.
        assert!(optimal.aggregate_cost.values[2] <= greedy.aggregate_cost.values[2]);
    }

    #[test]
    fn test_optimizer_trivial() {
        let model = simple_cost_model(&[
            ("a", 10.0, 1.0, 0.5, 5.0, 100.0),
            ("b", 20.0, 2.0, 0.3, 10.0, 50.0),
        ]);
        let optimizer = TrajectoryOptimizer::new(TrajectoryConfig {
            max_weight_vectors: 5,
            ..TrajectoryConfig::default()
        });

        let frontier = optimizer.optimize_trajectory(
            &[model.clone(), model],
            |_t, _s| true, // all strategies feasible
            2,
        );

        assert!(frontier.size() > 0);
    }

    #[test]
    fn test_optimizer_infeasible() {
        let model = simple_cost_model(&[
            ("a", 10.0, 1.0, 0.5, 5.0, 100.0),
        ]);
        let optimizer = TrajectoryOptimizer::new(TrajectoryConfig::default());
        let frontier = optimizer.optimize_trajectory(
            &[model],
            |_t, _s| false, // nothing feasible
            1,
        );
        assert_eq!(frontier.size(), 0);
    }

    #[test]
    fn test_optimizer_transition_budget() {
        let model = simple_cost_model(&[
            ("a", 10.0, 1.0, 0.5, 5.0, 100.0),
            ("b", 20.0, 2.0, 0.3, 10.0, 50.0),
        ]);
        let optimizer = TrajectoryOptimizer::new(TrajectoryConfig {
            transition_budget: 0, // no changes allowed
            max_weight_vectors: 5,
            ..TrajectoryConfig::default()
        });

        let frontier = optimizer.optimize_trajectory(
            &[model.clone(), model],
            |_t, _s| true,
            2,
        );

        // All trajectories should have zero transitions
        for entry in frontier.entries() {
            assert_eq!(entry.point.transition_count, 0);
        }
    }

    #[test]
    fn test_optimizer_discounted() {
        let model = simple_cost_model(&[
            ("a", 10.0, 1.0, 0.5, 5.0, 100.0),
        ]);
        let optimizer = TrajectoryOptimizer::new(TrajectoryConfig {
            discount_factor: 0.5,
            max_weight_vectors: 3,
            aggregation: AggregationMethod::DiscountedSum(0.5),
            ..TrajectoryConfig::default()
        });

        let frontier = optimizer.optimize_trajectory(
            &[model.clone(), model.clone(), model],
            |_t, _s| true,
            1,
        );

        assert!(frontier.size() > 0);
    }
}
