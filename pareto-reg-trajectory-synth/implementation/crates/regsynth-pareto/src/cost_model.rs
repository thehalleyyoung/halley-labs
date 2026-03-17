//! Multi-objective compliance cost model.
//!
//! Evaluates a [`CostVector`] for a given strategy by aggregating
//! per-obligation cost estimates across multiple dimensions.

use crate::strategy_repr::StrategyBitVec;
use crate::CostVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Per-obligation cost estimate
// ---------------------------------------------------------------------------

/// Cost estimate for a single obligation across all objective dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationCostEstimate {
    pub obligation_id: String,
    /// Financial cost to satisfy this obligation (USD).
    pub financial_cost: f64,
    /// Time required to achieve compliance (months).
    pub time_to_compliance: f64,
    /// Residual regulatory risk if NOT satisfied (probability 0–1).
    pub risk_if_unsatisfied: f64,
    /// Implementation complexity (abstract scale 0–100).
    pub implementation_complexity: f64,
    /// Optional jurisdiction-specific penalty if non-compliant.
    pub non_compliance_penalty: f64,
}

impl ObligationCostEstimate {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            obligation_id: id.into(),
            financial_cost: 0.0,
            time_to_compliance: 0.0,
            risk_if_unsatisfied: 0.0,
            implementation_complexity: 0.0,
            non_compliance_penalty: 0.0,
        }
    }

    pub fn with_financial_cost(mut self, cost: f64) -> Self {
        self.financial_cost = cost;
        self
    }

    pub fn with_time(mut self, months: f64) -> Self {
        self.time_to_compliance = months;
        self
    }

    pub fn with_risk(mut self, risk: f64) -> Self {
        self.risk_if_unsatisfied = risk;
        self
    }

    pub fn with_complexity(mut self, complexity: f64) -> Self {
        self.implementation_complexity = complexity;
        self
    }

    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.non_compliance_penalty = penalty;
        self
    }

    /// Cost vector contribution when this obligation IS satisfied.
    /// Financial and time costs are incurred; risk and penalty are avoided.
    pub fn satisfied_cost(&self) -> CostVector {
        CostVector::regulatory(
            self.financial_cost,
            self.time_to_compliance,
            0.0,
            self.implementation_complexity,
        )
    }

    /// Cost vector contribution when this obligation is NOT satisfied.
    /// Financial and time costs are avoided; risk and penalty are incurred.
    pub fn unsatisfied_cost(&self) -> CostVector {
        CostVector::regulatory(
            self.non_compliance_penalty,
            0.0,
            self.risk_if_unsatisfied,
            0.0,
        )
    }
}

// ---------------------------------------------------------------------------
// CostModel
// ---------------------------------------------------------------------------

/// Cost model: aggregates per-obligation estimates into strategy-level costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    obligations: Vec<ObligationCostEstimate>,
    obligation_index: HashMap<String, usize>,
    /// Discount factor for parallel implementation (0–1).
    /// 1.0 = no parallelism benefit; < 1.0 = time compressed.
    pub parallelism_factor: f64,
    /// Fixed overhead cost independent of obligations.
    pub fixed_overhead: CostVector,
}

impl CostModel {
    pub fn new() -> Self {
        Self {
            obligations: Vec::new(),
            obligation_index: HashMap::new(),
            parallelism_factor: 0.8,
            fixed_overhead: CostVector::zeros(4),
        }
    }

    pub fn with_parallelism_factor(mut self, factor: f64) -> Self {
        self.parallelism_factor = factor.clamp(0.0, 1.0);
        self
    }

    pub fn with_fixed_overhead(mut self, overhead: CostVector) -> Self {
        self.fixed_overhead = overhead;
        self
    }

    pub fn add_obligation(&mut self, estimate: ObligationCostEstimate) {
        let idx = self.obligations.len();
        self.obligation_index
            .insert(estimate.obligation_id.clone(), idx);
        self.obligations.push(estimate);
    }

    pub fn obligation_count(&self) -> usize {
        self.obligations.len()
    }

    pub fn get_estimate(&self, id: &str) -> Option<&ObligationCostEstimate> {
        self.obligation_index.get(id).map(|&i| &self.obligations[i])
    }

    /// Evaluate the cost vector for a strategy given as a bit vector.
    ///
    /// Bit i = true → obligation i is satisfied (incur implementation cost,
    /// avoid risk/penalty).  Bit i = false → obligation i is waived
    /// (incur risk/penalty, avoid implementation cost).
    pub fn evaluate(&self, strategy: &StrategyBitVec) -> CostVector {
        assert_eq!(
            strategy.len(),
            self.obligations.len(),
            "strategy length {} != obligation count {}",
            strategy.len(),
            self.obligations.len()
        );

        let n = self.obligations.len();
        let mut financial = 0.0_f64;
        let mut max_time = 0.0_f64;
        let mut risk = 0.0_f64;
        let mut complexity = 0.0_f64;

        for i in 0..n {
            let est = &self.obligations[i];
            if strategy.get(i) {
                // Obligation satisfied
                financial += est.financial_cost;
                max_time = max_time.max(est.time_to_compliance);
                complexity += est.implementation_complexity;
            } else {
                // Obligation waived
                financial += est.non_compliance_penalty;
                // Risk accumulates: P(any failure) = 1 - ∏(1 - p_i)
                risk = 1.0 - (1.0 - risk) * (1.0 - est.risk_if_unsatisfied);
            }
        }

        // Apply parallelism factor: if many obligations are parallel,
        // time ≈ max_time * factor + (1-factor) * sum_time/count
        let satisfied_count = strategy.count_satisfied();
        let time = if satisfied_count > 0 {
            let sum_time: f64 = (0..n)
                .filter(|&i| strategy.get(i))
                .map(|i| self.obligations[i].time_to_compliance)
                .sum();
            let avg_time = sum_time / satisfied_count as f64;
            self.parallelism_factor * max_time
                + (1.0 - self.parallelism_factor) * avg_time
        } else {
            0.0
        };

        let mut result = CostVector::regulatory(financial, time, risk, complexity);
        result = result.add(&self.fixed_overhead);
        result
    }

    /// Evaluate cost vectors for a trajectory (sequence of strategies).
    pub fn evaluate_trajectory(&self, trajectory: &[StrategyBitVec]) -> Vec<CostVector> {
        trajectory.iter().map(|s| self.evaluate(s)).collect()
    }

    /// Aggregate trajectory cost using a configurable method.
    pub fn aggregate_trajectory_cost(
        &self,
        trajectory: &[StrategyBitVec],
        method: AggregationMethod,
    ) -> CostVector {
        let costs = self.evaluate_trajectory(trajectory);
        if costs.is_empty() {
            return CostVector::zeros(4);
        }
        match method {
            AggregationMethod::Sum => {
                costs.iter().skip(1).fold(costs[0].clone(), |acc, c| acc.add(c))
            }
            AggregationMethod::Max => {
                costs.iter().skip(1).fold(costs[0].clone(), |acc, c| acc.component_max(c))
            }
            AggregationMethod::WeightedSum(ref weights) => {
                let mut result = CostVector::zeros(4);
                for (i, cost) in costs.iter().enumerate() {
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    result = result.add(&cost.scale(w));
                }
                result
            }
            AggregationMethod::DiscountedSum(gamma) => {
                let mut result = CostVector::zeros(4);
                let mut discount = 1.0;
                for cost in &costs {
                    result = result.add(&cost.scale(discount));
                    discount *= gamma;
                }
                result
            }
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Aggregation methods for trajectory costs
// ---------------------------------------------------------------------------

/// Method for aggregating per-timestep costs into a single trajectory cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Sum all timestep costs.
    Sum,
    /// Component-wise max across timesteps.
    Max,
    /// Per-timestep weights.
    WeightedSum(Vec<f64>),
    /// Geometric discounting with factor γ.
    DiscountedSum(f64),
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/// Create a cost model from simple per-obligation cost/risk/time tuples.
pub fn simple_cost_model(
    obligations: &[(
        &str,  // id
        f64,   // financial cost
        f64,   // time (months)
        f64,   // risk if unsatisfied
        f64,   // complexity
        f64,   // penalty
    )],
) -> CostModel {
    let mut model = CostModel::new();
    for &(id, cost, time, risk, complexity, penalty) in obligations {
        model.add_obligation(
            ObligationCostEstimate::new(id)
                .with_financial_cost(cost)
                .with_time(time)
                .with_risk(risk)
                .with_complexity(complexity)
                .with_penalty(penalty),
        );
    }
    model
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> CostModel {
        simple_cost_model(&[
            ("gdpr-art5",  50_000.0, 3.0, 0.3, 20.0, 100_000.0),
            ("gdpr-art6",  30_000.0, 2.0, 0.2, 15.0,  60_000.0),
            ("ai-act-art5", 80_000.0, 6.0, 0.5, 40.0, 200_000.0),
        ])
    }

    #[test]
    fn test_evaluate_all_satisfied() {
        let model = test_model();
        let strategy = StrategyBitVec::from_bits(vec![true, true, true]);
        let cost = model.evaluate(&strategy);
        assert!((cost.values[0] - 160_000.0).abs() < 1e-6); // sum of financial
        assert!(cost.values[2] < 1e-10); // no risk when all satisfied
    }

    #[test]
    fn test_evaluate_none_satisfied() {
        let model = test_model();
        let strategy = StrategyBitVec::from_bits(vec![false, false, false]);
        let cost = model.evaluate(&strategy);
        // Financial = sum of penalties
        assert!((cost.values[0] - 360_000.0).abs() < 1e-6);
        // Risk = 1 - (1-0.3)(1-0.2)(1-0.5) = 1 - 0.28 = 0.72
        assert!((cost.values[2] - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_partial() {
        let model = test_model();
        let strategy = StrategyBitVec::from_bits(vec![true, false, true]);
        let cost = model.evaluate(&strategy);
        // Financial = 50k + 60k(penalty) + 80k = 190k
        assert!((cost.values[0] - 190_000.0).abs() < 1e-6);
        // Risk from gdpr-art6 unsatisfied: 0.2
        assert!((cost.values[2] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_evaluation() {
        let model = test_model();
        let trajectory = vec![
            StrategyBitVec::from_bits(vec![true, false, false]),
            StrategyBitVec::from_bits(vec![true, true, false]),
            StrategyBitVec::from_bits(vec![true, true, true]),
        ];
        let costs = model.evaluate_trajectory(&trajectory);
        assert_eq!(costs.len(), 3);
        // Costs should decrease in risk over time
        assert!(costs[2].values[2] < costs[0].values[2]);
    }

    #[test]
    fn test_aggregate_sum() {
        let model = test_model();
        let trajectory = vec![
            StrategyBitVec::from_bits(vec![true, true, true]),
            StrategyBitVec::from_bits(vec![true, true, true]),
        ];
        let agg = model.aggregate_trajectory_cost(&trajectory, AggregationMethod::Sum);
        let single = model.evaluate(&trajectory[0]);
        assert!((agg.values[0] - 2.0 * single.values[0]).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_discounted() {
        let model = test_model();
        let s = StrategyBitVec::from_bits(vec![true, true, true]);
        let trajectory = vec![s.clone(), s.clone(), s.clone()];
        let agg = model.aggregate_trajectory_cost(&trajectory, AggregationMethod::DiscountedSum(0.9));
        let single = model.evaluate(&s);
        let expected = single.values[0] * (1.0 + 0.9 + 0.81);
        assert!((agg.values[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_simple_cost_model_factory() {
        let model = simple_cost_model(&[
            ("a", 100.0, 1.0, 0.1, 5.0, 200.0),
        ]);
        assert_eq!(model.obligation_count(), 1);
        assert!(model.get_estimate("a").is_some());
        assert!(model.get_estimate("b").is_none());
    }

    #[test]
    fn test_cost_model_with_overhead() {
        let model = CostModel::new()
            .with_fixed_overhead(CostVector::regulatory(1000.0, 1.0, 0.0, 5.0));
        let strategy = StrategyBitVec::new(0);
        let cost = model.evaluate(&strategy);
        assert!((cost.values[0] - 1000.0).abs() < 1e-6);
    }
}
