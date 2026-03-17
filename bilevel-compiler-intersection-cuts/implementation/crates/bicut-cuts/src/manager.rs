//! Cut round management for bilevel branch-and-cut.
//!
//! Orchestrates cut generation across multiple generators (intersection, Gomory,
//! disjunctive), manages round budgets, tracks gap convergence, detects stalling,
//! and provides statistics for integration with branch-and-cut.

use crate::cut_pool::{CutPool, CutPoolConfig};
use crate::disjunctive::{DisjunctiveConfig, DisjunctiveCut, DisjunctiveCutGenerator};
use crate::gomory::{GomoryConfig, GomoryCut, GomoryCutGenerator};
use crate::intersection::{GeneratedCut, IntersectionCutConfig, IntersectionCutGenerator};
use crate::separation::{SeparationConfig, SeparationOracle};
use crate::strengthening::{CutStrengthener, StrengtheningConfig};
use crate::{BilevelCut, CutError, CutResult, DEFAULT_MAX_CUTS_PER_ROUND, MIN_EFFICACY, TOLERANCE};
use bicut_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the cut manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutManagerConfig {
    /// Maximum number of cut rounds.
    pub max_rounds: usize,
    /// Maximum cuts per round.
    pub max_cuts_per_round: usize,
    /// Minimum improvement in objective to continue rounds.
    pub min_obj_improvement: f64,
    /// Number of stalling rounds before stopping.
    pub max_stall_rounds: usize,
    /// Budget allocation fractions for each generator type.
    pub intersection_fraction: f64,
    pub gomory_fraction: f64,
    pub disjunctive_fraction: f64,
    /// Whether to apply strengthening to generated cuts.
    pub strengthen_cuts: bool,
    /// Whether to use the cut pool.
    pub use_pool: bool,
    /// Pool configuration.
    pub pool_config: CutPoolConfig,
    /// Minimum gap to continue generating cuts.
    pub min_gap: f64,
    /// Whether to generate Gomory cuts.
    pub enable_gomory: bool,
    /// Whether to generate disjunctive cuts.
    pub enable_disjunctive: bool,
}

impl Default for CutManagerConfig {
    fn default() -> Self {
        Self {
            max_rounds: 50,
            max_cuts_per_round: DEFAULT_MAX_CUTS_PER_ROUND,
            min_obj_improvement: 1e-6,
            max_stall_rounds: 5,
            intersection_fraction: 0.6,
            gomory_fraction: 0.3,
            disjunctive_fraction: 0.1,
            strengthen_cuts: true,
            use_pool: true,
            pool_config: CutPoolConfig::default(),
            min_gap: 1e-8,
            enable_gomory: true,
            enable_disjunctive: false,
        }
    }
}

/// Result of a single cut round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResult {
    pub round: usize,
    pub cuts_generated: usize,
    pub cuts_added: usize,
    pub intersection_cuts: usize,
    pub gomory_cuts: usize,
    pub disjunctive_cuts: usize,
    pub objective_before: f64,
    pub objective_after: f64,
    pub gap_before: f64,
    pub gap_after: f64,
    pub gap_closed_fraction: f64,
    pub time_us: u64,
    pub stalling: bool,
}

/// Overall statistics for the cut manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CutManagerStats {
    pub total_rounds: usize,
    pub total_cuts_generated: usize,
    pub total_cuts_added: usize,
    pub total_intersection: usize,
    pub total_gomory: usize,
    pub total_disjunctive: usize,
    pub initial_gap: f64,
    pub final_gap: f64,
    pub gap_closed_fraction: f64,
    pub total_time_us: u64,
    pub stall_rounds: usize,
    pub converged: bool,
    pub round_results: Vec<RoundResult>,
}

/// The cut manager orchestrating bilevel cut generation.
pub struct CutManager {
    pub config: CutManagerConfig,
    pool: CutPool,
    round_counter: usize,
    prev_objective: f64,
    stall_counter: usize,
    stats: CutManagerStats,
    n_leader: usize,
    n_follower: usize,
    follower_obj: Vec<f64>,
}

impl CutManager {
    pub fn new(
        config: CutManagerConfig,
        n_leader: usize,
        n_follower: usize,
        follower_obj: Vec<f64>,
    ) -> Self {
        let pool = CutPool::new(config.pool_config.clone());
        Self {
            config,
            pool,
            round_counter: 0,
            prev_objective: f64::NEG_INFINITY,
            stall_counter: 0,
            stats: CutManagerStats::default(),
            n_leader,
            n_follower,
            follower_obj,
        }
    }

    /// Run a single cut round.
    /// Returns the cuts to add to the LP and the round result.
    pub fn run_round(
        &mut self,
        point: &[f64],
        objective: f64,
        phi_x: f64,
        basis_status: &[BasisStatus],
        tableau_rows_simple: &[(usize, Vec<f64>)],
        tableau_rows_extended: &[(usize, Vec<f64>, f64)],
        phi_evaluator: &dyn Fn(&[f64]) -> Option<f64>,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        integer_vars: &[usize],
    ) -> CutResult<(Vec<BilevelCut>, RoundResult)> {
        let start = std::time::Instant::now();
        self.round_counter += 1;

        // Compute bilevel gap.
        let cy: f64 = self
            .follower_obj
            .iter()
            .enumerate()
            .map(|(i, &c)| c * point.get(self.n_leader + i).copied().unwrap_or(0.0))
            .sum();
        let gap = cy - phi_x;

        if self.round_counter == 1 {
            self.stats.initial_gap = gap;
        }

        // Check convergence.
        if gap <= self.config.min_gap {
            let result = RoundResult {
                round: self.round_counter,
                cuts_generated: 0,
                cuts_added: 0,
                intersection_cuts: 0,
                gomory_cuts: 0,
                disjunctive_cuts: 0,
                objective_before: objective,
                objective_after: objective,
                gap_before: gap,
                gap_after: gap,
                gap_closed_fraction: 1.0,
                time_us: 0,
                stalling: false,
            };
            self.stats.converged = true;
            return Ok((Vec::new(), result));
        }

        // Check stalling.
        let improvement = objective - self.prev_objective;
        let stalling =
            if self.round_counter > 1 && improvement.abs() < self.config.min_obj_improvement {
                self.stall_counter += 1;
                true
            } else {
                self.stall_counter = 0;
                false
            };

        if self.stall_counter >= self.config.max_stall_rounds {
            self.stats.stall_rounds = self.stall_counter;
            let result = RoundResult {
                round: self.round_counter,
                cuts_generated: 0,
                cuts_added: 0,
                intersection_cuts: 0,
                gomory_cuts: 0,
                disjunctive_cuts: 0,
                objective_before: objective,
                objective_after: objective,
                gap_before: gap,
                gap_after: gap,
                gap_closed_fraction: 0.0,
                time_us: 0,
                stalling: true,
            };
            return Ok((Vec::new(), result));
        }

        self.prev_objective = objective;

        // Budget allocation.
        let total_budget = self.config.max_cuts_per_round;
        let intersection_budget =
            (total_budget as f64 * self.config.intersection_fraction) as usize;
        let gomory_budget = (total_budget as f64 * self.config.gomory_fraction) as usize;
        let disjunctive_budget = total_budget.saturating_sub(intersection_budget + gomory_budget);

        let mut all_cuts: Vec<BilevelCut> = Vec::new();
        let mut intersection_count = 0;
        let mut gomory_count = 0;
        let mut disjunctive_count = 0;

        // Generate intersection cuts.
        {
            let mut gen = IntersectionCutGenerator::new(
                IntersectionCutConfig::default(),
                self.n_leader,
                self.n_follower,
                self.follower_obj.clone(),
            );
            match gen.generate(
                point,
                basis_status,
                tableau_rows_simple,
                phi_evaluator,
                lower_bounds,
                upper_bounds,
            ) {
                Ok(generated) => {
                    for gc in generated.into_iter().take(intersection_budget) {
                        all_cuts.push(gc.cut);
                        intersection_count += 1;
                    }
                }
                Err(CutError::AlreadyFeasible) => {}
                Err(e) => {
                    log::debug!("Intersection cut generation failed: {}", e);
                }
            }
        }

        // Generate Gomory cuts.
        if self.config.enable_gomory && !integer_vars.is_empty() {
            let mut gen = GomoryCutGenerator::new(
                GomoryConfig::default(),
                self.n_leader,
                self.n_follower,
                integer_vars.to_vec(),
                self.follower_obj.clone(),
            );
            match gen.generate_gmi_cuts(tableau_rows_extended, basis_status, point) {
                Ok(gmi_cuts) => {
                    for gc in gmi_cuts.into_iter().take(gomory_budget) {
                        all_cuts.push(gc.cut);
                        gomory_count += 1;
                    }
                }
                Err(e) => {
                    log::debug!("Gomory cut generation failed: {}", e);
                }
            }
        }

        // Add to pool and filter.
        for cut in &all_cuts {
            self.pool.add_cut(cut.clone(), point);
        }
        self.pool.advance_round();

        let cuts_generated = all_cuts.len();
        let cuts_added = all_cuts.len();

        self.stats.total_rounds += 1;
        self.stats.total_cuts_generated += cuts_generated;
        self.stats.total_cuts_added += cuts_added;
        self.stats.total_intersection += intersection_count;
        self.stats.total_gomory += gomory_count;
        self.stats.total_disjunctive += disjunctive_count;
        self.stats.final_gap = gap;

        let gap_closed = if self.stats.initial_gap > TOLERANCE {
            1.0 - gap / self.stats.initial_gap
        } else {
            1.0
        };
        self.stats.gap_closed_fraction = gap_closed;

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.total_time_us += elapsed;

        let result = RoundResult {
            round: self.round_counter,
            cuts_generated,
            cuts_added,
            intersection_cuts: intersection_count,
            gomory_cuts: gomory_count,
            disjunctive_cuts: disjunctive_count,
            objective_before: objective,
            objective_after: objective,
            gap_before: gap,
            gap_after: gap,
            gap_closed_fraction: gap_closed,
            time_us: elapsed,
            stalling,
        };
        self.stats.round_results.push(result.clone());

        Ok((all_cuts, result))
    }

    /// Check if the manager should stop generating cuts.
    pub fn should_stop(&self) -> bool {
        self.round_counter >= self.config.max_rounds
            || self.stall_counter >= self.config.max_stall_rounds
            || self.stats.converged
    }

    /// Reset the manager for a new node.
    pub fn reset_for_node(&mut self) {
        self.round_counter = 0;
        self.prev_objective = f64::NEG_INFINITY;
        self.stall_counter = 0;
    }

    /// Full reset.
    pub fn reset(&mut self) {
        self.reset_for_node();
        self.pool.clear();
        self.stats = CutManagerStats::default();
    }

    pub fn stats(&self) -> &CutManagerStats {
        &self.stats
    }
    pub fn pool(&self) -> &CutPool {
        &self.pool
    }
    pub fn pool_mut(&mut self) -> &mut CutPool {
        &mut self.pool
    }
    pub fn current_round(&self) -> usize {
        self.round_counter
    }
    pub fn is_converged(&self) -> bool {
        self.stats.converged
    }
    pub fn stall_count(&self) -> usize {
        self.stall_counter
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        format!(
            "CutManager: {} rounds, {} cuts generated, {} added, gap closed {:.2}%",
            self.stats.total_rounds,
            self.stats.total_cuts_generated,
            self.stats.total_cuts_added,
            self.stats.gap_closed_fraction * 100.0,
        )
    }
}

/// Compute the optimality gap.
pub fn compute_gap(primal_bound: f64, dual_bound: f64) -> f64 {
    if primal_bound.abs() < TOLERANCE {
        return dual_bound.abs();
    }
    ((primal_bound - dual_bound) / primal_bound.abs()).abs()
}

/// Check if a gap is closed to within tolerance.
pub fn gap_is_closed(gap: f64, tolerance: f64) -> bool {
    gap <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> CutManager {
        CutManager::new(CutManagerConfig::default(), 2, 2, vec![1.0, 1.0])
    }

    #[test]
    fn test_config_default() {
        let cfg = CutManagerConfig::default();
        assert!(cfg.max_rounds > 0);
        assert!(cfg.max_cuts_per_round > 0);
        assert!(
            cfg.intersection_fraction + cfg.gomory_fraction + cfg.disjunctive_fraction
                <= 1.0 + 1e-10
        );
    }

    #[test]
    fn test_manager_creation() {
        let mgr = make_manager();
        assert_eq!(mgr.current_round(), 0);
        assert!(!mgr.should_stop());
    }

    #[test]
    fn test_should_stop_max_rounds() {
        let mut mgr = make_manager();
        mgr.round_counter = mgr.config.max_rounds;
        assert!(mgr.should_stop());
    }

    #[test]
    fn test_should_stop_converged() {
        let mut mgr = make_manager();
        mgr.stats.converged = true;
        assert!(mgr.should_stop());
    }

    #[test]
    fn test_should_stop_stalling() {
        let mut mgr = make_manager();
        mgr.stall_counter = mgr.config.max_stall_rounds;
        assert!(mgr.should_stop());
    }

    #[test]
    fn test_reset() {
        let mut mgr = make_manager();
        mgr.round_counter = 10;
        mgr.stall_counter = 3;
        mgr.reset();
        assert_eq!(mgr.current_round(), 0);
        assert_eq!(mgr.stall_count(), 0);
    }

    #[test]
    fn test_compute_gap() {
        let gap = compute_gap(10.0, 8.0);
        assert!((gap - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_gap_is_closed() {
        assert!(gap_is_closed(1e-10, 1e-8));
        assert!(!gap_is_closed(0.1, 1e-8));
    }

    #[test]
    fn test_summary() {
        let mgr = make_manager();
        let s = mgr.summary();
        assert!(s.contains("CutManager"));
    }

    #[test]
    fn test_round_result_struct() {
        let rr = RoundResult {
            round: 1,
            cuts_generated: 5,
            cuts_added: 3,
            intersection_cuts: 2,
            gomory_cuts: 1,
            disjunctive_cuts: 0,
            objective_before: 10.0,
            objective_after: 11.0,
            gap_before: 1.0,
            gap_after: 0.5,
            gap_closed_fraction: 0.5,
            time_us: 1000,
            stalling: false,
        };
        assert_eq!(rr.cuts_generated, 5);
        assert!(!rr.stalling);
    }

    #[test]
    fn test_stats_default() {
        let stats = CutManagerStats::default();
        assert_eq!(stats.total_rounds, 0);
        assert!(!stats.converged);
    }
}
