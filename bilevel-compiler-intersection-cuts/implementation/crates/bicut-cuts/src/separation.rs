//! Separation oracle for bilevel intersection cuts.
//!
//! Determines if a point is bilevel-infeasible (c^T y > phi(x)),
//! finds the most violated cut, implements separation heuristics
//! (max violation, weighted), manages separation rounds, and
//! interacts with the cut pool.

use crate::balas::RayLength;
use crate::cut_pool::{CutEntry, CutPool, CutPoolConfig};
use crate::intersection::{GeneratedCut, IntersectionCutConfig, IntersectionCutGenerator};
use crate::{BilevelCut, CutError, CutResult, MIN_EFFICACY, TOLERANCE};
use bicut_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Separation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeparationStrategy {
    /// Find the single most violated cut.
    MaxViolation,
    /// Find multiple cuts ranked by weighted score.
    WeightedScore,
    /// Round-robin across different generators.
    RoundRobin,
    /// Use all generators and pick the best.
    BestOfAll,
}

/// Configuration for the separation oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationConfig {
    pub strategy: SeparationStrategy,
    pub max_cuts_per_round: usize,
    pub min_violation: f64,
    pub min_efficacy: f64,
    pub max_rounds: usize,
    pub efficacy_weight: f64,
    pub orthogonality_weight: f64,
    pub sparsity_weight: f64,
    pub pool_config: CutPoolConfig,
    pub intersection_config: IntersectionCutConfig,
    /// Whether to search the pool before generating new cuts.
    pub search_pool_first: bool,
    /// Maximum fraction of cuts from pool vs. newly generated.
    pub pool_fraction: f64,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            strategy: SeparationStrategy::MaxViolation,
            max_cuts_per_round: 20,
            min_violation: TOLERANCE,
            min_efficacy: MIN_EFFICACY,
            max_rounds: 50,
            efficacy_weight: 1.0,
            orthogonality_weight: 0.1,
            sparsity_weight: 0.01,
            pool_config: CutPoolConfig::default(),
            intersection_config: IntersectionCutConfig::default(),
            search_pool_first: true,
            pool_fraction: 0.3,
        }
    }
}

/// Result of a separation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationResult {
    /// Whether the point is bilevel-infeasible.
    pub is_infeasible: bool,
    /// The bilevel gap c^T y - phi(x).
    pub gap: f64,
    /// Generated cuts (sorted by efficacy, descending).
    pub cuts: Vec<BilevelCut>,
    /// How many cuts came from the pool.
    pub from_pool: usize,
    /// How many cuts were newly generated.
    pub newly_generated: usize,
    /// Separation round number.
    pub round: usize,
    /// Time taken (microseconds).
    pub time_us: u64,
}

/// Separation oracle for bilevel cuts.
#[derive(Debug)]
pub struct SeparationOracle {
    pub config: SeparationConfig,
    pool: CutPool,
    generator: IntersectionCutGenerator,
    n_leader: usize,
    n_follower: usize,
    follower_obj: Vec<f64>,
    round_counter: usize,
    stats: SeparationStats,
}

/// Statistics for separation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeparationStats {
    pub total_calls: usize,
    pub total_infeasible: usize,
    pub total_feasible: usize,
    pub total_cuts_generated: usize,
    pub total_cuts_from_pool: usize,
    pub total_rounds: usize,
    pub avg_gap: f64,
    pub max_gap: f64,
    pub avg_cuts_per_round: f64,
}

impl SeparationOracle {
    pub fn new(
        config: SeparationConfig,
        n_leader: usize,
        n_follower: usize,
        follower_obj: Vec<f64>,
    ) -> Self {
        let pool = CutPool::new(config.pool_config.clone());
        let generator = IntersectionCutGenerator::new(
            config.intersection_config.clone(),
            n_leader,
            n_follower,
            follower_obj.clone(),
        );
        Self {
            config,
            pool,
            generator,
            n_leader,
            n_follower,
            follower_obj,
            round_counter: 0,
            stats: SeparationStats::default(),
        }
    }

    /// Check if a point is bilevel-infeasible.
    pub fn is_bilevel_infeasible(&self, point: &[f64], phi_x: f64) -> bool {
        let cy = self.compute_follower_obj(point);
        cy > phi_x + self.config.min_violation
    }

    /// Compute c^T y from the point.
    fn compute_follower_obj(&self, point: &[f64]) -> f64 {
        self.follower_obj
            .iter()
            .enumerate()
            .map(|(i, &c)| c * point.get(self.n_leader + i).copied().unwrap_or(0.0))
            .sum()
    }

    /// Main separation: find violated cuts for the given point.
    pub fn separate(
        &mut self,
        point: &[f64],
        phi_x: f64,
        basis_status: &[BasisStatus],
        tableau_rows: &[(usize, Vec<f64>)],
        phi_evaluator: &dyn Fn(&[f64]) -> Option<f64>,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> CutResult<SeparationResult> {
        self.stats.total_calls += 1;
        self.round_counter += 1;
        let start = std::time::Instant::now();

        let cy = self.compute_follower_obj(point);
        let gap = cy - phi_x;

        if gap <= self.config.min_violation {
            self.stats.total_feasible += 1;
            return Ok(SeparationResult {
                is_infeasible: false,
                gap,
                cuts: Vec::new(),
                from_pool: 0,
                newly_generated: 0,
                round: self.round_counter,
                time_us: 0,
            });
        }

        self.stats.total_infeasible += 1;
        self.update_gap_stats(gap);

        let mut all_cuts: Vec<(BilevelCut, f64)> = Vec::new();
        let mut from_pool = 0;
        let mut newly_generated = 0;

        // Search pool first.
        if self.config.search_pool_first {
            let pool_max =
                (self.config.max_cuts_per_round as f64 * self.config.pool_fraction) as usize;
            let pool_cuts = self
                .pool
                .get_violated_cuts(point, self.config.min_violation, pool_max);
            for entry in &pool_cuts {
                let eff = entry.cut.compute_efficacy(point);
                if eff >= self.config.min_efficacy {
                    all_cuts.push((entry.cut.clone(), eff));
                    from_pool += 1;
                }
            }
        }

        // Generate new cuts.
        let remaining = self
            .config
            .max_cuts_per_round
            .saturating_sub(all_cuts.len());
        if remaining > 0 {
            match self.generator.generate(
                point,
                basis_status,
                tableau_rows,
                phi_evaluator,
                lower_bounds,
                upper_bounds,
            ) {
                Ok(generated) => {
                    for gc in generated.into_iter().take(remaining) {
                        let eff = gc.cut.compute_efficacy(point);
                        self.pool.add_cut(gc.cut.clone(), point);
                        all_cuts.push((gc.cut, eff));
                        newly_generated += 1;
                    }
                }
                Err(CutError::AlreadyFeasible) => {}
                Err(e) => {
                    log::debug!("Separation cut generation failed: {}", e);
                }
            }
        }

        // Sort and select based on strategy.
        let selected = self.apply_strategy(&mut all_cuts, point);

        self.stats.total_cuts_generated += newly_generated;
        self.stats.total_cuts_from_pool += from_pool;
        self.stats.total_rounds += 1;
        self.update_avg_cuts(selected.len());

        let elapsed = start.elapsed().as_micros() as u64;

        Ok(SeparationResult {
            is_infeasible: true,
            gap,
            cuts: selected,
            from_pool,
            newly_generated,
            round: self.round_counter,
            time_us: elapsed,
        })
    }

    /// Apply the separation strategy to select and order cuts.
    fn apply_strategy(
        &self,
        candidates: &mut Vec<(BilevelCut, f64)>,
        point: &[f64],
    ) -> Vec<BilevelCut> {
        match self.config.strategy {
            SeparationStrategy::MaxViolation => {
                candidates
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates
                    .iter()
                    .take(self.config.max_cuts_per_round)
                    .map(|(c, _)| c.clone())
                    .collect()
            }
            SeparationStrategy::WeightedScore => {
                let mut scored: Vec<(BilevelCut, f64)> = candidates
                    .iter()
                    .map(|(cut, eff)| {
                        let sparsity_score = 1.0 / (cut.nnz as f64 + 1.0);
                        let score = self.config.efficacy_weight * eff
                            + self.config.sparsity_weight * sparsity_score;
                        (cut.clone(), score)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored
                    .into_iter()
                    .take(self.config.max_cuts_per_round)
                    .map(|(c, _)| c)
                    .collect()
            }
            SeparationStrategy::RoundRobin | SeparationStrategy::BestOfAll => {
                candidates
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates
                    .iter()
                    .take(self.config.max_cuts_per_round)
                    .map(|(c, _)| c.clone())
                    .collect()
            }
        }
    }

    /// Update gap statistics.
    fn update_gap_stats(&mut self, gap: f64) {
        let n = self.stats.total_infeasible as f64;
        self.stats.avg_gap = (self.stats.avg_gap * (n - 1.0) + gap) / n;
        self.stats.max_gap = self.stats.max_gap.max(gap);
    }

    /// Update average cuts per round.
    fn update_avg_cuts(&mut self, num_cuts: usize) {
        let n = self.stats.total_rounds as f64;
        self.stats.avg_cuts_per_round =
            (self.stats.avg_cuts_per_round * (n - 1.0) + num_cuts as f64) / n;
    }

    /// Quick separation: just check pool for violated cuts.
    pub fn separate_from_pool(&self, point: &[f64], max_cuts: usize) -> Vec<BilevelCut> {
        self.pool
            .get_violated_cuts(point, self.config.min_violation, max_cuts)
            .into_iter()
            .map(|e| e.cut.clone())
            .collect()
    }

    /// Get the most violated cut from the pool.
    pub fn most_violated_from_pool(&self, point: &[f64]) -> Option<BilevelCut> {
        let violated = self
            .pool
            .get_violated_cuts(point, self.config.min_violation, 1);
        violated.into_iter().next().map(|e| e.cut.clone())
    }

    /// Add an externally generated cut to the pool.
    pub fn add_to_pool(&mut self, cut: BilevelCut, point: &[f64]) {
        self.pool.add_cut(cut, point);
    }

    /// Get the cut pool reference.
    pub fn pool(&self) -> &CutPool {
        &self.pool
    }
    /// Get mutable pool reference.
    pub fn pool_mut(&mut self) -> &mut CutPool {
        &mut self.pool
    }
    /// Get statistics.
    pub fn stats(&self) -> &SeparationStats {
        &self.stats
    }
    /// Reset the oracle state for a new solve.
    pub fn reset(&mut self) {
        self.round_counter = 0;
        self.stats = SeparationStats::default();
        self.pool.clear();
        self.generator.clear_duplicate_cache();
    }

    /// Get the current round number.
    pub fn current_round(&self) -> usize {
        self.round_counter
    }
}

/// Check bilevel feasibility for a batch of points.
pub fn batch_feasibility_check(
    points: &[Vec<f64>],
    n_leader: usize,
    follower_obj: &[f64],
    phi_evaluator: &dyn Fn(&[f64]) -> Option<f64>,
    tolerance: f64,
) -> Vec<bool> {
    points
        .iter()
        .map(|point| {
            let x = &point[..n_leader.min(point.len())];
            let cy: f64 = follower_obj
                .iter()
                .enumerate()
                .map(|(i, &c)| c * point.get(n_leader + i).copied().unwrap_or(0.0))
                .sum();
            match phi_evaluator(x) {
                Some(phi) => cy <= phi + tolerance,
                None => true,
            }
        })
        .collect()
}

/// Compute the maximum violation across a set of cuts.
pub fn max_violation(cuts: &[BilevelCut], point: &[f64]) -> f64 {
    cuts.iter()
        .map(|c| c.violation(point))
        .fold(0.0f64, f64::max)
}

/// Score a cut for selection using a weighted combination.
pub fn score_cut(
    cut: &BilevelCut,
    point: &[f64],
    existing_cuts: &[BilevelCut],
    efficacy_weight: f64,
    orthogonality_weight: f64,
    sparsity_weight: f64,
) -> f64 {
    let efficacy = cut.compute_efficacy(point);
    let sparsity = 1.0 / (cut.nnz as f64 + 1.0);
    let orthogonality = if existing_cuts.is_empty() {
        1.0
    } else {
        existing_cuts
            .iter()
            .map(|ec| crate::cut_orthogonality(&cut.coeffs, &ec.coeffs))
            .fold(f64::INFINITY, f64::min)
    };
    efficacy_weight * efficacy + orthogonality_weight * orthogonality + sparsity_weight * sparsity
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_oracle() -> SeparationOracle {
        SeparationOracle::new(SeparationConfig::default(), 2, 2, vec![1.0, 1.0])
    }

    #[test]
    fn test_config_default() {
        let cfg = SeparationConfig::default();
        assert!(cfg.max_cuts_per_round > 0);
        assert!(cfg.min_violation > 0.0);
    }

    #[test]
    fn test_oracle_creation() {
        let oracle = make_oracle();
        assert_eq!(oracle.current_round(), 0);
    }

    #[test]
    fn test_is_bilevel_infeasible() {
        let oracle = make_oracle();
        // point = (0.5, 0.5, 0.8, 0.8), c = (1,1), cy = 1.6, phi = 0.5
        assert!(oracle.is_bilevel_infeasible(&[0.5, 0.5, 0.8, 0.8], 0.5));
        // point = (0.5, 0.5, 0.2, 0.2), cy = 0.4, phi = 0.5
        assert!(!oracle.is_bilevel_infeasible(&[0.5, 0.5, 0.2, 0.2], 0.5));
    }

    #[test]
    fn test_batch_feasibility() {
        let points = vec![vec![0.5, 0.3], vec![0.5, 0.8]];
        let results = batch_feasibility_check(&points, 1, &[1.0], &|x: &[f64]| Some(x[0]), 1e-8);
        assert!(results[0]); // 0.3 <= 0.5
        assert!(!results[1]); // 0.8 > 0.5
    }

    #[test]
    fn test_max_violation_fn() {
        let cuts = vec![
            BilevelCut::new(vec![(0, 1.0)], 2.0, ConstraintSense::Ge),
            BilevelCut::new(vec![(0, 1.0)], 3.0, ConstraintSense::Ge),
        ];
        let v = max_violation(&cuts, &[0.0]);
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_score_cut_fn() {
        let cut = BilevelCut::new(vec![(0, 1.0)], 1.0, ConstraintSense::Ge);
        let score = score_cut(&cut, &[0.0], &[], 1.0, 0.1, 0.01);
        assert!(score > 0.0);
    }

    #[test]
    fn test_separation_strategy_variants() {
        assert_eq!(
            SeparationStrategy::MaxViolation,
            SeparationStrategy::MaxViolation
        );
        assert_ne!(
            SeparationStrategy::MaxViolation,
            SeparationStrategy::WeightedScore
        );
    }

    #[test]
    fn test_separate_from_pool_empty() {
        let oracle = make_oracle();
        let cuts = oracle.separate_from_pool(&[0.5, 0.5, 0.8, 0.8], 10);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_add_to_pool() {
        let mut oracle = make_oracle();
        let cut = BilevelCut::new(vec![(0, 1.0)], 1.0, ConstraintSense::Ge);
        oracle.add_to_pool(cut, &[0.5]);
        assert_eq!(oracle.pool().len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut oracle = make_oracle();
        oracle.round_counter = 5;
        oracle.reset();
        assert_eq!(oracle.current_round(), 0);
    }

    #[test]
    fn test_stats_default() {
        let stats = SeparationStats::default();
        assert_eq!(stats.total_calls, 0);
    }
}
