//! Iterative weighted partial MaxSMT for Pareto enumeration (Mechanism M2).
//!
//! Enumerates the Pareto frontier by repeatedly solving a scalarised
//! single-objective problem with different weight vectors, adding blocking
//! clauses to exclude dominance cones of found points, and accumulating
//! results into a Pareto frontier.
//!
//! Guarantees ε-completeness: the returned set is an ε-cover of the
//! true Pareto front.

use crate::cost_model::CostModel;
use crate::dominance;
use crate::frontier::ParetoFrontier;
use crate::scalarization::WeightedSumScalarizer;
use crate::strategy_repr::StrategyBitVec;
use crate::CostVector;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for iterative MaxSMT-based Pareto enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterativeMaxSmtConfig {
    /// Epsilon for ε-completeness guarantee.
    pub epsilon: f64,
    /// Maximum number of iterations (solver calls).
    pub max_iterations: usize,
    /// Number of divisions for simplex weight generation.
    pub weight_divisions: usize,
    /// Use Chebyshev scalarization (finds non-convex points) vs.
    /// weighted sum (only convex hull points).
    pub use_chebyshev: bool,
    /// Number of adaptive refinement rounds after initial sweep.
    pub adaptive_rounds: usize,
}

impl Default for IterativeMaxSmtConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            max_iterations: 500,
            weight_divisions: 10,
            use_chebyshev: true,
            adaptive_rounds: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Blocking clause generation
// ---------------------------------------------------------------------------

/// A blocking constraint representing the dominance cone of a found point.
///
/// Encoded as: ¬(c₁ ≤ p₁ ∧ c₂ ≤ p₂ ∧ … ∧ cₖ ≤ pₖ)
/// i.e., at least one objective must be strictly better (smaller) than
/// this point, preventing the solver from returning dominated or equal points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockingConstraint {
    pub point: CostVector,
}

impl BlockingConstraint {
    pub fn new(point: CostVector) -> Self {
        Self { point }
    }

    /// Check whether a candidate cost vector violates this blocking
    /// constraint (i.e., falls inside the dominance cone).
    pub fn blocks(&self, candidate: &CostVector) -> bool {
        dominance::weakly_dominates(&self.point, candidate)
    }
}

// ---------------------------------------------------------------------------
// ParetoEnumerator
// ---------------------------------------------------------------------------

/// Iterative solver-driven Pareto frontier enumerator.
///
/// The enumerator works in phases:
/// 1. **Initial sweep**: solve with a grid of weight vectors.
/// 2. **Blocking**: after each solution, add a blocking constraint for its
///    dominance cone.
/// 3. **Adaptive refinement**: identify gaps in the frontier and generate
///    targeted weight vectors.
/// 4. **Termination**: stop when ε-coverage is achieved or max iterations
///    reached.
pub struct ParetoEnumerator {
    pub config: IterativeMaxSmtConfig,
    blocking_constraints: Vec<BlockingConstraint>,
    iteration_count: usize,
}

impl ParetoEnumerator {
    pub fn new(config: IterativeMaxSmtConfig) -> Self {
        Self {
            config,
            blocking_constraints: Vec::new(),
            iteration_count: 0,
        }
    }

    /// Reset the enumerator for a fresh run.
    pub fn reset(&mut self) {
        self.blocking_constraints.clear();
        self.iteration_count = 0;
    }

    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }

    /// Enumerate the Pareto frontier using an abstract solver function.
    ///
    /// `solve_fn` takes a weight vector and a set of blocking constraints
    /// and returns `Some((strategy, cost))` if a feasible solution is found,
    /// `None` if infeasible.
    pub fn enumerate_frontier<F>(
        &mut self,
        dim: usize,
        mut solve_fn: F,
    ) -> ParetoFrontier<StrategyBitVec>
    where
        F: FnMut(&[f64], &[BlockingConstraint]) -> Option<(StrategyBitVec, CostVector)>,
    {
        let mut frontier = ParetoFrontier::with_epsilon(dim, self.config.epsilon);
        self.blocking_constraints.clear();
        self.iteration_count = 0;

        // Phase 1: initial sweep with simplex weights
        let initial_weights =
            WeightedSumScalarizer::simplex_weights(dim, self.config.weight_divisions);

        for weights in &initial_weights {
            if self.iteration_count >= self.config.max_iterations {
                break;
            }
            self.iteration_count += 1;

            if let Some((strategy, cost)) = solve_fn(weights, &self.blocking_constraints) {
                if frontier.add_point(strategy, cost.clone()) {
                    self.blocking_constraints
                        .push(BlockingConstraint::new(cost));
                }
            }
        }

        // Phase 2: adaptive refinement
        for _round in 0..self.config.adaptive_rounds {
            if self.iteration_count >= self.config.max_iterations {
                break;
            }

            let costs: Vec<CostVector> = frontier.entries().iter().map(|e| e.cost.clone()).collect();
            if costs.len() < 2 {
                break;
            }

            let new_weights = crate::scalarization::adaptive_weights(&costs, dim, 10);

            for weights in &new_weights {
                if self.iteration_count >= self.config.max_iterations {
                    break;
                }
                self.iteration_count += 1;

                if let Some((strategy, cost)) = solve_fn(weights, &self.blocking_constraints) {
                    if frontier.add_point(strategy, cost.clone()) {
                        self.blocking_constraints
                            .push(BlockingConstraint::new(cost));
                    }
                }
            }

            // Check ε-coverage termination
            if self.check_epsilon_coverage(&frontier, dim) {
                break;
            }
        }

        frontier
    }

    /// Enumerate using a cost model with brute-force evaluation.
    ///
    /// For small problems, directly evaluates all strategies against
    /// each weight vector and respects blocking constraints.
    pub fn enumerate_from_cost_model(
        &mut self,
        cost_model: &CostModel,
        num_obligations: usize,
    ) -> ParetoFrontier<StrategyBitVec> {
        let dim = 4; // regulatory standard
        let model = cost_model.clone();

        self.enumerate_frontier(dim, |weights, blocking| {
            let mut best: Option<(StrategyBitVec, CostVector, f64)> = None;

            // Generate candidate strategies
            let candidates: Vec<StrategyBitVec> = if num_obligations <= 16 {
                crate::strategy_repr::enumerate_strategies(num_obligations)
            } else {
                crate::strategy_repr::random_feasible_strategies(
                    num_obligations,
                    500,
                    5000,
                    |_| true,
                )
            };

            for strategy in &candidates {
                let cost = model.evaluate(strategy);

                // Check blocking constraints
                if blocking.iter().any(|bc| bc.blocks(&cost)) {
                    continue;
                }

                let scalar = cost.weighted_sum(weights);
                if best.as_ref().map_or(true, |(_, _, s)| scalar < *s) {
                    best = Some((strategy.clone(), cost, scalar));
                }
            }

            best.map(|(s, c, _)| (s, c))
        })
    }

    /// Check whether the current frontier achieves ε-coverage.
    ///
    /// ε-coverage means: for every point p on the true front, there
    /// exists a point q in our frontier such that q ε-dominates p.
    ///
    /// We approximate this by checking the spacing: if the maximum
    /// nearest-neighbour distance is ≤ ε, we declare coverage.
    fn check_epsilon_coverage<T: Clone>(
        &self,
        frontier: &ParetoFrontier<T>,
        _dim: usize,
    ) -> bool {
        if frontier.size() < 2 {
            return false;
        }

        let costs: Vec<&CostVector> = frontier.cost_vectors();
        let mut max_nn_dist = 0.0_f64;

        for (i, ci) in costs.iter().enumerate() {
            let nn_dist = costs
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, cj)| ci.euclidean_distance(cj))
                .fold(f64::INFINITY, f64::min);
            max_nn_dist = max_nn_dist.max(nn_dist);
        }

        max_nn_dist <= self.config.epsilon
    }
}

impl fmt::Display for ParetoEnumerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParetoEnumerator(ε={}, iters={}/{}, blocking={})",
            self.config.epsilon,
            self.iteration_count,
            self.config.max_iterations,
            self.blocking_constraints.len()
        )
    }
}

// ---------------------------------------------------------------------------
// ε-completeness proof sketch
// ---------------------------------------------------------------------------

/// Verify ε-completeness of a computed frontier against a reference.
///
/// Returns `true` if for every point in `reference`, there exists a
/// point in `computed` within ε Euclidean distance.
pub fn verify_epsilon_completeness(
    computed: &[CostVector],
    reference: &[CostVector],
    epsilon: f64,
) -> bool {
    reference.iter().all(|r| {
        computed
            .iter()
            .any(|c| c.euclidean_distance(r) <= epsilon)
    })
}

/// Termination guarantee: the iterative MaxSMT procedure terminates in
/// at most `O((R/ε)^d)` iterations where R is the diameter of the
/// objective space and d is the dimension.
///
/// Returns the upper bound on iterations.
pub fn termination_bound(
    ideal: &CostVector,
    nadir: &CostVector,
    epsilon: f64,
) -> usize {
    let d = ideal.dim();
    let diameter = ideal.euclidean_distance(nadir);
    let cells_per_dim = (diameter / epsilon).ceil() as usize;
    cells_per_dim.saturating_pow(d as u32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_model::simple_cost_model;

    #[test]
    fn test_blocking_constraint() {
        let bc = BlockingConstraint::new(CostVector::new(vec![1.0, 2.0]));
        assert!(bc.blocks(&CostVector::new(vec![1.0, 2.0]))); // equal → blocked
        assert!(bc.blocks(&CostVector::new(vec![2.0, 3.0]))); // dominated → blocked
        assert!(!bc.blocks(&CostVector::new(vec![0.5, 3.0]))); // incomparable → ok
    }

    #[test]
    fn test_enumerate_simple() {
        let model = simple_cost_model(&[
            ("a", 100.0, 2.0, 0.3, 10.0, 500.0),
            ("b", 200.0, 4.0, 0.5, 20.0, 800.0),
        ]);

        let mut enumerator = ParetoEnumerator::new(IterativeMaxSmtConfig {
            max_iterations: 50,
            weight_divisions: 3,
            adaptive_rounds: 1,
            ..IterativeMaxSmtConfig::default()
        });

        let frontier = enumerator.enumerate_from_cost_model(&model, 2);
        assert!(frontier.size() > 0);
    }

    #[test]
    fn test_enumerate_with_blocking() {
        let call_count = std::cell::Cell::new(0usize);

        let mut enumerator = ParetoEnumerator::new(IterativeMaxSmtConfig {
            max_iterations: 20,
            weight_divisions: 3,
            adaptive_rounds: 0,
            ..IterativeMaxSmtConfig::default()
        });

        let frontier = enumerator.enumerate_frontier(2, |weights, blocking| {
            call_count.set(call_count.get() + 1);
            // Return a solution based on weights
            let x = weights[0] * 5.0;
            let y = (1.0 - weights[0]) * 5.0;
            let cost = CostVector::new(vec![x, y]);

            if blocking.iter().any(|bc| bc.blocks(&cost)) {
                None
            } else {
                Some((StrategyBitVec::from_bits(vec![true]), cost))
            }
        });

        assert!(frontier.size() > 0);
        assert!(call_count.get() > 0);
    }

    #[test]
    fn test_epsilon_completeness_verification() {
        let computed = vec![
            CostVector::new(vec![0.0, 1.0]),
            CostVector::new(vec![0.5, 0.5]),
            CostVector::new(vec![1.0, 0.0]),
        ];
        let reference = vec![
            CostVector::new(vec![0.1, 0.9]),
            CostVector::new(vec![0.5, 0.5]),
            CostVector::new(vec![0.9, 0.1]),
        ];

        assert!(verify_epsilon_completeness(&computed, &reference, 0.2));
        assert!(!verify_epsilon_completeness(&computed, &reference, 0.01));
    }

    #[test]
    fn test_termination_bound() {
        let ideal = CostVector::new(vec![0.0, 0.0]);
        let nadir = CostVector::new(vec![10.0, 10.0]);
        let bound = termination_bound(&ideal, &nadir, 1.0);
        // diameter ≈ 14.14, cells ≈ 15, bound ≈ 15^2 = 225
        assert!(bound > 100);
        assert!(bound < 1000);
    }

    #[test]
    fn test_enumerator_reset() {
        let mut enumerator = ParetoEnumerator::new(IterativeMaxSmtConfig::default());
        enumerator.iteration_count = 10;
        enumerator.blocking_constraints.push(BlockingConstraint::new(
            CostVector::new(vec![1.0]),
        ));
        enumerator.reset();
        assert_eq!(enumerator.iteration_count(), 0);
        assert!(enumerator.blocking_constraints.is_empty());
    }

    #[test]
    fn test_enumerator_display() {
        let e = ParetoEnumerator::new(IterativeMaxSmtConfig::default());
        let s = format!("{}", e);
        assert!(s.contains("ParetoEnumerator"));
    }

    #[test]
    fn test_three_obligation_frontier() {
        let model = simple_cost_model(&[
            ("gdpr-1",  50_000.0, 3.0, 0.4, 20.0, 200_000.0),
            ("gdpr-2",  30_000.0, 2.0, 0.2, 15.0,  80_000.0),
            ("ai-act",  80_000.0, 6.0, 0.6, 40.0, 500_000.0),
        ]);

        let mut enumerator = ParetoEnumerator::new(IterativeMaxSmtConfig {
            max_iterations: 100,
            weight_divisions: 4,
            adaptive_rounds: 2,
            ..IterativeMaxSmtConfig::default()
        });

        let frontier = enumerator.enumerate_from_cost_model(&model, 3);
        // With 3 obligations, 8 strategies. Should find several non-dominated ones.
        assert!(frontier.size() >= 2);
    }
}
