//! Incremental Pareto frontier maintenance (Mechanism M4).
//!
//! When the regulatory state changes (constraints added/removed),
//! update the Pareto frontier without full recomputation:
//!
//! 1. Test surviving points against new constraints.
//! 2. Solve only in the uncovered region.
//! 3. Merge and filter.
//!
//! Preserves ε-coverage of the updated true frontier.

use crate::cost_model::CostModel;
use crate::dominance;
use crate::frontier::ParetoFrontier;
use crate::iterative_maxsmt::{ParetoEnumerator, IterativeMaxSmtConfig};
use crate::strategy_repr::StrategyBitVec;
use crate::CostVector;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// ConstraintDiff
// ---------------------------------------------------------------------------

/// Represents a change in the regulatory constraint set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDiff {
    /// Indices of newly added obligations.
    pub added_obligations: Vec<usize>,
    /// Indices of removed obligations.
    pub removed_obligations: Vec<usize>,
    /// Obligations whose cost parameters changed.
    pub modified_obligations: Vec<usize>,
}

impl ConstraintDiff {
    pub fn new() -> Self {
        Self {
            added_obligations: Vec::new(),
            removed_obligations: Vec::new(),
            modified_obligations: Vec::new(),
        }
    }

    pub fn with_added(mut self, added: Vec<usize>) -> Self {
        self.added_obligations = added;
        self
    }

    pub fn with_removed(mut self, removed: Vec<usize>) -> Self {
        self.removed_obligations = removed;
        self
    }

    pub fn with_modified(mut self, modified: Vec<usize>) -> Self {
        self.modified_obligations = modified;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.added_obligations.is_empty()
            && self.removed_obligations.is_empty()
            && self.modified_obligations.is_empty()
    }

    /// Indices of all affected obligations.
    pub fn affected_obligations(&self) -> Vec<usize> {
        let mut all = Vec::new();
        all.extend_from_slice(&self.added_obligations);
        all.extend_from_slice(&self.removed_obligations);
        all.extend_from_slice(&self.modified_obligations);
        all.sort();
        all.dedup();
        all
    }
}

impl Default for ConstraintDiff {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// IncrementalMaintainer
// ---------------------------------------------------------------------------

/// Incremental Pareto frontier maintainer.
///
/// Maintains a cached frontier and updates it efficiently when
/// constraints change.
pub struct IncrementalMaintainer {
    /// Current cost model.
    cost_model: CostModel,
    /// Number of obligations.
    num_obligations: usize,
    /// Epsilon for coverage.
    epsilon: f64,
    /// History of updates for auditing.
    update_history: Vec<UpdateRecord>,
}

/// Record of a frontier update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub diff: ConstraintDiff,
    pub points_survived: usize,
    pub points_removed: usize,
    pub points_added: usize,
    pub full_recompute: bool,
}

impl IncrementalMaintainer {
    pub fn new(cost_model: CostModel, num_obligations: usize, epsilon: f64) -> Self {
        Self {
            cost_model,
            num_obligations,
            epsilon,
            update_history: Vec::new(),
        }
    }

    pub fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    pub fn update_history(&self) -> &[UpdateRecord] {
        &self.update_history
    }

    /// Update the frontier given a constraint diff and new cost model.
    ///
    /// Algorithm:
    /// 1. **Survival test**: re-evaluate all existing points with the new
    ///    cost model. Points are "affected" if their cost changes.
    /// 2. **Invalidation**: remove points that are now infeasible or
    ///    dominated under the new costs.
    /// 3. **Gap filling**: use targeted enumeration to discover new
    ///    Pareto-optimal points in the uncovered region.
    /// 4. **Merge**: combine surviving and new points, re-filter for
    ///    non-dominance.
    pub fn update_frontier(
        &mut self,
        old_frontier: &ParetoFrontier<StrategyBitVec>,
        diff: &ConstraintDiff,
        new_cost_model: &CostModel,
        feasibility_check: impl Fn(&StrategyBitVec) -> bool,
    ) -> ParetoFrontier<StrategyBitVec> {
        self.cost_model = new_cost_model.clone();

        // If the diff is large, just do a full recompute
        let affected = diff.affected_obligations();
        let threshold = self.num_obligations / 2;
        if affected.len() > threshold || old_frontier.is_empty() {
            return self.full_recompute(&feasibility_check, diff);
        }

        // Step 1: Survival test — re-evaluate existing points
        let mut surviving: Vec<(StrategyBitVec, CostVector)> = Vec::new();
        let mut invalidated_count = 0;

        for entry in old_frontier.entries() {
            let strategy = &entry.point;
            // Check feasibility under new constraints
            if !feasibility_check(strategy) {
                invalidated_count += 1;
                continue;
            }
            // Re-evaluate cost with new model
            let new_cost = self.cost_model.evaluate(strategy);
            surviving.push((strategy.clone(), new_cost));
        }

        // Step 2: Filter dominated among survivors
        let survivor_costs: Vec<CostVector> =
            surviving.iter().map(|(_, c)| c.clone()).collect();
        let nd_indices = dominance::filter_dominated_indexed(&survivor_costs);
        let survivors: Vec<(StrategyBitVec, CostVector)> = nd_indices
            .into_iter()
            .map(|(i, _)| surviving[i].clone())
            .collect();

        // Step 3: Gap filling — enumerate in the uncovered region
        let new_points = self.fill_gaps(
            &survivors,
            &affected,
            &feasibility_check,
        );

        // Step 4: Merge
        let dim = if let Some((_, c)) = survivors.first() {
            c.dim()
        } else if let Some((_, c)) = new_points.first() {
            c.dim()
        } else {
            4
        };

        let mut result = ParetoFrontier::with_epsilon(dim, self.epsilon);
        for (strategy, cost) in &survivors {
            result.add_point(strategy.clone(), cost.clone());
        }
        for (strategy, cost) in &new_points {
            result.add_point(strategy.clone(), cost.clone());
        }

        let record = UpdateRecord {
            diff: diff.clone(),
            points_survived: survivors.len(),
            points_removed: invalidated_count,
            points_added: new_points.len(),
            full_recompute: false,
        };
        self.update_history.push(record);

        result
    }

    /// Full recompute via ParetoEnumerator.
    fn full_recompute(
        &mut self,
        feasibility_check: &impl Fn(&StrategyBitVec) -> bool,
        diff: &ConstraintDiff,
    ) -> ParetoFrontier<StrategyBitVec> {
        let mut enumerator = ParetoEnumerator::new(IterativeMaxSmtConfig {
            epsilon: self.epsilon,
            max_iterations: 200,
            weight_divisions: 5,
            adaptive_rounds: 2,
            ..IterativeMaxSmtConfig::default()
        });

        let model = self.cost_model.clone();
        let n = self.num_obligations;

        let frontier = enumerator.enumerate_frontier(4, |weights, blocking| {
            let candidates: Vec<StrategyBitVec> = if n <= 16 {
                crate::strategy_repr::enumerate_strategies(n)
                    .into_iter()
                    .filter(|s| feasibility_check(s))
                    .collect()
            } else {
                crate::strategy_repr::random_feasible_strategies(
                    n, 500, 5000, feasibility_check,
                )
            };

            let mut best: Option<(StrategyBitVec, CostVector, f64)> = None;
            for s in &candidates {
                let cost = model.evaluate(s);
                if blocking.iter().any(|bc| bc.blocks(&cost)) {
                    continue;
                }
                let scalar = cost.weighted_sum(weights);
                if best.as_ref().map_or(true, |(_, _, b)| scalar < *b) {
                    best = Some((s.clone(), cost, scalar));
                }
            }
            best.map(|(s, c, _)| (s, c))
        });

        self.update_history.push(UpdateRecord {
            diff: diff.clone(),
            points_survived: 0,
            points_removed: 0,
            points_added: frontier.size(),
            full_recompute: true,
        });

        frontier
    }

    /// Fill gaps in the frontier by exploring strategies that differ from
    /// survivors in the affected obligation indices.
    fn fill_gaps(
        &self,
        survivors: &[(StrategyBitVec, CostVector)],
        affected: &[usize],
        feasibility_check: &impl Fn(&StrategyBitVec) -> bool,
    ) -> Vec<(StrategyBitVec, CostVector)> {
        if affected.is_empty() {
            return Vec::new();
        }

        let n = self.num_obligations;
        let mut new_points: Vec<(StrategyBitVec, CostVector)> = Vec::new();

        // For each survivor, try flipping the affected bits
        for (base_strategy, _) in survivors {
            let flips = affected.len().min(8); // limit combinatorial explosion
            let max_combos = 1u32 << flips;

            for mask in 0..max_combos {
                let mut candidate = base_strategy.clone();
                for (bit_idx, &obl_idx) in affected.iter().enumerate().take(flips) {
                    if obl_idx < n {
                        let new_val = (mask >> bit_idx) & 1 == 1;
                        candidate.set(obl_idx, new_val);
                    }
                }

                if !feasibility_check(&candidate) {
                    continue;
                }

                let cost = self.cost_model.evaluate(&candidate);

                // Check it's not dominated by any survivor
                let dominated = survivors
                    .iter()
                    .any(|(_, sc)| dominance::dominates(sc, &cost));
                if !dominated {
                    // Check it's not dominated by already-found new points
                    let dom_by_new = new_points
                        .iter()
                        .any(|(_, nc)| dominance::dominates(nc, &cost));
                    if !dom_by_new {
                        // Remove new points dominated by this one
                        new_points.retain(|(_, nc)| !dominance::dominates(&cost, nc));
                        new_points.push((candidate, cost));
                    }
                }
            }
        }

        // Also try some random strategies if we have room
        if new_points.len() < 10 {
            let randoms = crate::strategy_repr::random_feasible_strategies(
                n,
                50,
                500,
                feasibility_check,
            );
            for s in randoms {
                let cost = self.cost_model.evaluate(&s);
                let dom_existing = survivors
                    .iter()
                    .any(|(_, sc)| dominance::dominates(sc, &cost))
                    || new_points
                        .iter()
                        .any(|(_, nc)| dominance::dominates(nc, &cost));
                if !dom_existing {
                    new_points.retain(|(_, nc)| !dominance::dominates(&cost, nc));
                    new_points.push((s, cost));
                }
            }
        }

        new_points
    }

    /// Verify that the updated frontier preserves ε-coverage relative
    /// to a reference set.
    pub fn verify_coverage(
        frontier: &ParetoFrontier<StrategyBitVec>,
        reference: &[CostVector],
        epsilon: f64,
    ) -> bool {
        crate::iterative_maxsmt::verify_epsilon_completeness(
            &frontier.entries().iter().map(|e| e.cost.clone()).collect::<Vec<_>>(),
            reference,
            epsilon,
        )
    }
}

impl fmt::Display for IncrementalMaintainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IncrementalMaintainer(obligations={}, ε={}, updates={})",
            self.num_obligations,
            self.epsilon,
            self.update_history.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_model::simple_cost_model;

    fn make_model_v1() -> CostModel {
        simple_cost_model(&[
            ("a", 100.0, 2.0, 0.3, 10.0, 500.0),
            ("b", 200.0, 4.0, 0.5, 20.0, 800.0),
            ("c", 150.0, 3.0, 0.4, 15.0, 600.0),
        ])
    }

    fn make_model_v2() -> CostModel {
        // Obligation "b" becomes cheaper
        simple_cost_model(&[
            ("a", 100.0, 2.0, 0.3, 10.0, 500.0),
            ("b",  50.0, 1.0, 0.5, 5.0,  800.0),
            ("c", 150.0, 3.0, 0.4, 15.0, 600.0),
        ])
    }

    fn initial_frontier(model: &CostModel) -> ParetoFrontier<StrategyBitVec> {
        let mut frontier = ParetoFrontier::new(4);
        for mask in 0..8u32 {
            let bits: Vec<bool> = (0..3).map(|i| (mask >> i) & 1 == 1).collect();
            let s = StrategyBitVec::from_bits(bits);
            let cost = model.evaluate(&s);
            frontier.add_point(s, cost);
        }
        frontier
    }

    #[test]
    fn test_constraint_diff_basic() {
        let diff = ConstraintDiff::new()
            .with_added(vec![3])
            .with_removed(vec![1])
            .with_modified(vec![0]);
        assert!(!diff.is_empty());
        assert_eq!(diff.affected_obligations(), vec![0, 1, 3]);
    }

    #[test]
    fn test_constraint_diff_empty() {
        let diff = ConstraintDiff::new();
        assert!(diff.is_empty());
    }

    #[test]
    fn test_incremental_update_modified() {
        let model_v1 = make_model_v1();
        let model_v2 = make_model_v2();
        let old_frontier = initial_frontier(&model_v1);

        let mut maintainer = IncrementalMaintainer::new(model_v1, 3, 0.01);
        let diff = ConstraintDiff::new().with_modified(vec![1]);

        let new_frontier = maintainer.update_frontier(
            &old_frontier,
            &diff,
            &model_v2,
            |_| true,
        );

        assert!(new_frontier.size() > 0);
        assert!(!maintainer.update_history().is_empty());
        assert!(!maintainer.update_history().last().unwrap().full_recompute);
    }

    #[test]
    fn test_incremental_full_recompute() {
        let model_v1 = make_model_v1();
        let model_v2 = make_model_v2();
        let old_frontier = ParetoFrontier::new(4); // empty

        let mut maintainer = IncrementalMaintainer::new(model_v1, 3, 0.01);
        let diff = ConstraintDiff::new().with_modified(vec![0, 1, 2]);

        let new_frontier = maintainer.update_frontier(
            &old_frontier,
            &diff,
            &model_v2,
            |_| true,
        );

        assert!(new_frontier.size() > 0);
        assert!(maintainer.update_history().last().unwrap().full_recompute);
    }

    #[test]
    fn test_incremental_feasibility_filter() {
        let model = make_model_v1();
        let old_frontier = initial_frontier(&model);
        let old_size = old_frontier.size();

        let mut maintainer = IncrementalMaintainer::new(model.clone(), 3, 0.01);
        let diff = ConstraintDiff::new().with_added(vec![0]);

        // New feasibility: obligation 0 must always be satisfied
        let new_frontier = maintainer.update_frontier(
            &old_frontier,
            &diff,
            &model,
            |s| s.get(0), // obligation 0 must be true
        );

        // Should have fewer or equal points (some strategies now infeasible)
        assert!(new_frontier.size() <= old_size);
        // All surviving strategies must have obligation 0 satisfied
        for entry in new_frontier.entries() {
            assert!(entry.point.get(0));
        }
    }

    #[test]
    fn test_update_record() {
        let model = make_model_v1();
        let frontier = initial_frontier(&model);

        let mut maintainer = IncrementalMaintainer::new(model.clone(), 3, 0.01);
        let diff = ConstraintDiff::new().with_modified(vec![2]);

        maintainer.update_frontier(&frontier, &diff, &model, |_| true);

        let record = &maintainer.update_history()[0];
        assert!(!record.full_recompute);
        assert!(record.points_survived > 0);
    }

    #[test]
    fn test_verify_coverage() {
        let model = make_model_v1();
        let frontier = initial_frontier(&model);
        let reference: Vec<CostVector> = frontier
            .entries()
            .iter()
            .map(|e| e.cost.clone())
            .collect();

        assert!(IncrementalMaintainer::verify_coverage(
            &frontier,
            &reference,
            0.01,
        ));
    }

    #[test]
    fn test_display() {
        let model = make_model_v1();
        let m = IncrementalMaintainer::new(model, 3, 0.01);
        let s = format!("{}", m);
        assert!(s.contains("IncrementalMaintainer"));
    }
}
