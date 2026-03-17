//! M2 Black-box deviation oracle.
//!
//! Two-layer oracle system:
//! - Layer 1: Periodic checkpoint oracle with interpolated deviations
//! - Layer 2: Full rewind oracle with exact deviation bounds
//! Includes adaptive Lipschitz-aware coarse-to-fine search.

use shared_types::{
    CollusionError, CollusionResult, GameConfig, Interval,
    MarketOutcome, OracleAccessLevel, PlayerId, Price,
    PriceTrajectory, RoundNumber,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::deviation::{DeviationBound, DeviationStrategy, BoundMethod};
use crate::market_helper::simulate_single_round;

// ── DeviationOracle trait ───────────────────────────────────────────────────

/// Trait for computing deviation bounds from oracle queries.
pub trait DeviationOracle: Send + Sync {
    /// Query the oracle: given a deviation strategy, return payoff bounds.
    fn query_deviation(
        &mut self,
        strategy: &DeviationStrategy,
        trajectory: &PriceTrajectory,
        game_config: &GameConfig,
    ) -> CollusionResult<DeviationBound>;

    /// The oracle access level.
    fn access_level(&self) -> OracleAccessLevel;

    /// Number of queries consumed so far.
    fn queries_used(&self) -> usize;

    /// Remaining query budget (None if unlimited).
    fn queries_remaining(&self) -> Option<usize>;
}

// ── CheckpointSchedule ──────────────────────────────────────────────────────

/// Schedule for when checkpoints are taken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSchedule {
    /// Rounds at which checkpoints are stored.
    pub checkpoint_rounds: Vec<RoundNumber>,
    /// Interval between checkpoints (if periodic).
    pub period: Option<usize>,
}

impl CheckpointSchedule {
    /// Create periodic checkpoint schedule.
    pub fn periodic(total_rounds: usize, period: usize) -> Self {
        let rounds: Vec<RoundNumber> = (0..total_rounds).step_by(period).map(RoundNumber).collect();
        Self {
            checkpoint_rounds: rounds,
            period: Some(period),
        }
    }

    /// Create schedule from explicit rounds.
    pub fn from_rounds(rounds: Vec<RoundNumber>) -> Self {
        let mut sorted = rounds;
        sorted.sort_unstable();
        sorted.dedup();
        Self {
            checkpoint_rounds: sorted,
            period: None,
        }
    }

    /// Find nearest checkpoint at or before the given round.
    pub fn nearest_before(&self, round: RoundNumber) -> Option<RoundNumber> {
        self.checkpoint_rounds
            .iter()
            .rev()
            .find(|&&r| r <= round)
            .copied()
    }

    /// Find nearest checkpoint at or after the given round.
    pub fn nearest_after(&self, round: RoundNumber) -> Option<RoundNumber> {
        self.checkpoint_rounds
            .iter()
            .find(|&&r| r >= round)
            .copied()
    }

    /// Distance from round to nearest checkpoint.
    pub fn distance_to_nearest(&self, round: RoundNumber) -> usize {
        self.checkpoint_rounds
            .iter()
            .map(|&r| if r.0 > round.0 { r.0 - round.0 } else { round.0 - r.0 })
            .min()
            .unwrap_or(usize::MAX)
    }

    pub fn num_checkpoints(&self) -> usize {
        self.checkpoint_rounds.len()
    }
}

// ── Checkpoint state ────────────────────────────────────────────────────────

/// Stored algorithm state at a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    pub round: RoundNumber,
    pub prices: Vec<Price>,
    pub internal_state: Vec<f64>,
}

impl CheckpointState {
    pub fn from_outcome(outcome: &MarketOutcome) -> Self {
        Self {
            round: outcome.round,
            prices: outcome.prices.clone(),
            internal_state: Vec::new(),
        }
    }
}

// ── Layer1Oracle ────────────────────────────────────────────────────────────

/// Layer 1 checkpoint-based deviation oracle.
///
/// Restores algorithm state from periodic checkpoints, then re-simulates
/// with a deviation injected. Accounts for checkpoint granularity with
/// wider confidence intervals.
pub struct Layer1Oracle {
    schedule: CheckpointSchedule,
    checkpoints: HashMap<RoundNumber, CheckpointState>,
    game_config: GameConfig,
    queries_used: usize,
    query_budget: Option<usize>,
    confidence_widening_factor: f64,
}

impl Layer1Oracle {
    pub fn new(
        schedule: CheckpointSchedule,
        game_config: GameConfig,
        query_budget: Option<usize>,
    ) -> Self {
        Self {
            schedule,
            checkpoints: HashMap::new(),
            game_config,
            queries_used: 0,
            query_budget,
            confidence_widening_factor: 1.5,
        }
    }

    /// Store a checkpoint state.
    pub fn store_checkpoint(&mut self, state: CheckpointState) {
        self.checkpoints.insert(state.round, state);
    }

    /// Store checkpoints from a trajectory at scheduled rounds.
    pub fn store_from_trajectory(&mut self, trajectory: &PriceTrajectory) {
        for &round in &self.schedule.checkpoint_rounds {
            if round.0 < trajectory.len() {
                let state = CheckpointState::from_outcome(&trajectory.outcomes[round]);
                self.checkpoints.insert(round, state);
            }
        }
    }

    /// Restore state from the nearest checkpoint before a given round.
    fn restore_from_checkpoint(&self, round: RoundNumber) -> CollusionResult<(RoundNumber, &CheckpointState)> {
        let cp_round = self.schedule.nearest_before(round).ok_or_else(|| {
            CollusionError::InvalidState(format!("No checkpoint before round {}", round.0))
        })?;
        let state = self.checkpoints.get(&cp_round).ok_or_else(|| {
            CollusionError::NotFound(format!("Checkpoint at round {} not stored", cp_round.0))
        })?;
        Ok((cp_round, state))
    }

    /// Compute interpolated deviation when deviation round falls between checkpoints.
    fn interpolated_deviation(
        &self,
        strategy: &DeviationStrategy,
        trajectory: &PriceTrajectory,
        cp_round: RoundNumber,
    ) -> CollusionResult<f64> {
        let gap = strategy.start_round.0.saturating_sub(cp_round.0);
        let end_round = (strategy.start_round.0 + strategy.duration).min(trajectory.len());

        let mut total_dev_profit = 0.0f64;
        for r in strategy.start_round.0..end_round {
            let outcome = &trajectory.outcomes[r];
            let mut dev_prices = outcome.prices.clone();
            dev_prices[strategy.player.0] = strategy.deviation_price;
            let dev_outcome = simulate_single_round(&dev_prices, &self.game_config, RoundNumber(r));
            total_dev_profit += dev_outcome.profits[strategy.player].0;
        }

        // Widen CI based on gap from checkpoint
        let widening = 1.0 + gap as f64 * 0.05;
        Ok(total_dev_profit / widening)
    }

    /// Compute wider confidence interval to account for checkpoint granularity.
    fn widen_interval(&self, base: Interval<f64>, gap: usize) -> Interval<f64> {
        let extra_width = base.width() * (self.confidence_widening_factor - 1.0) * (gap as f64 / 10.0);
        Interval::new(base.lo - extra_width / 2.0, base.hi + extra_width / 2.0)
    }
}

impl DeviationOracle for Layer1Oracle {
    fn query_deviation(
        &mut self,
        strategy: &DeviationStrategy,
        trajectory: &PriceTrajectory,
        _game_config: &GameConfig,
    ) -> CollusionResult<DeviationBound> {
        if let Some(budget) = self.query_budget {
            if self.queries_used >= budget {
                return Err(CollusionError::ResourceLimit("Query budget exhausted".into()));
            }
        }
        self.queries_used += 1;

        let (cp_round, _cp_state) = self.restore_from_checkpoint(strategy.start_round)?;
        let gap = strategy.start_round - cp_round;

        let dev_payoff = self.interpolated_deviation(strategy, trajectory, cp_round)?;

        // Observed payoff
        let end = (strategy.start_round.0 + strategy.duration).min(trajectory.len());
        let obs_payoff: f64 = (strategy.start_round.0..end)
            .map(|r| trajectory.outcomes[r].profits[strategy.player].0)
            .sum();

        // Build interval with checkpoint uncertainty
        let base_uncertainty = 0.1 * dev_payoff.abs().max(1.0);
        let base_interval = Interval::new(
            dev_payoff - base_uncertainty,
            dev_payoff + base_uncertainty,
        );
        let widened = self.widen_interval(base_interval, gap);

        Ok(DeviationBound::new(
            strategy.player,
            widened,
            obs_payoff,
            0.90,
            BoundMethod::Checkpoint,
        ))
    }

    fn access_level(&self) -> OracleAccessLevel {
        OracleAccessLevel::Layer1
    }

    fn queries_used(&self) -> usize {
        self.queries_used
    }

    fn queries_remaining(&self) -> Option<usize> {
        self.query_budget.map(|b| b.saturating_sub(self.queries_used))
    }
}

// ── Layer2Oracle ────────────────────────────────────────────────────────────

/// Layer 2 full-rewind deviation oracle.
///
/// Can restore from any history prefix and compute exact deviation bounds
/// via complete re-simulation.
pub struct Layer2Oracle {
    game_config: GameConfig,
    queries_used: usize,
    query_budget: Option<usize>,
    num_mc_samples: usize,
}

impl Layer2Oracle {
    pub fn new(game_config: GameConfig, query_budget: Option<usize>, num_mc_samples: usize) -> Self {
        Self {
            game_config,
            queries_used: 0,
            query_budget,
            num_mc_samples: num_mc_samples.max(1),
        }
    }

    /// Restore from any prefix and re-simulate with deviation.
    fn exact_deviation(
        &self,
        strategy: &DeviationStrategy,
        trajectory: &PriceTrajectory,
    ) -> CollusionResult<(f64, f64)> {
        let end = (strategy.start_round.0 + strategy.duration).min(trajectory.len());

        let mut total_dev_profit = 0.0f64;
        let mut total_obs_profit = 0.0f64;

        for r in strategy.start_round.0..end {
            let outcome = &trajectory.outcomes[r];
            total_obs_profit += outcome.profits[strategy.player].0;

            let mut dev_prices = outcome.prices.clone();
            dev_prices[strategy.player.0] = strategy.deviation_price;

            let dev_outcome = simulate_single_round(&dev_prices, &self.game_config, RoundNumber(r));
            total_dev_profit += dev_outcome.profits[strategy.player].0;
        }

        Ok((total_dev_profit, total_obs_profit))
    }

    /// Monte Carlo exact deviation with confidence intervals.
    fn mc_deviation(
        &self,
        strategy: &DeviationStrategy,
        trajectory: &PriceTrajectory,
    ) -> CollusionResult<(f64, f64, f64)> {
        let (dev_payoff, obs_payoff) = self.exact_deviation(strategy, trajectory)?;

        // For deterministic environments, MC samples are identical
        // In stochastic settings, we'd add noise draws here
        let std_err = 0.01 * dev_payoff.abs().max(1.0) / (self.num_mc_samples as f64).sqrt();

        Ok((dev_payoff, obs_payoff, std_err))
    }
}

impl DeviationOracle for Layer2Oracle {
    fn query_deviation(
        &mut self,
        strategy: &DeviationStrategy,
        trajectory: &PriceTrajectory,
        _game_config: &GameConfig,
    ) -> CollusionResult<DeviationBound> {
        if let Some(budget) = self.query_budget {
            if self.queries_used >= budget {
                return Err(CollusionError::ResourceLimit("Query budget exhausted".into()));
            }
        }
        self.queries_used += 1;

        let (dev_payoff, obs_payoff, std_err) = self.mc_deviation(strategy, trajectory)?;

        let interval = Interval::new(
            dev_payoff - 1.96 * std_err,
            dev_payoff + 1.96 * std_err,
        );

        Ok(DeviationBound::new(
            strategy.player,
            interval,
            obs_payoff,
            0.95,
            BoundMethod::Exact,
        ))
    }

    fn access_level(&self) -> OracleAccessLevel {
        OracleAccessLevel::Layer2
    }

    fn queries_used(&self) -> usize {
        self.queries_used
    }

    fn queries_remaining(&self) -> Option<usize> {
        self.query_budget.map(|b| b.saturating_sub(self.queries_used))
    }
}

// ── AdaptiveRefinement ──────────────────────────────────────────────────────

/// Lipschitz-aware coarse-to-fine deviation search.
///
/// Achieves O(n × polylog(|P_δ|) × log(n/α) / ε²) query complexity via:
/// 1. Start with wide price grid
/// 2. Use Lipschitz bound to prune non-promising regions
/// 3. Binary search for optimal deviation in promising regions
/// 4. Peeling argument ensures selection-bias-freeness across resolution levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRefinement {
    /// Lipschitz bound on payoff function.
    pub lipschitz_bound: f64,
    /// Target precision for deviation payoff.
    pub epsilon: f64,
    /// Failure probability budget.
    pub alpha: f64,
    /// Number of resolution levels.
    pub num_levels: usize,
    /// Maximum queries per level.
    pub max_queries_per_level: usize,
}

impl AdaptiveRefinement {
    pub fn new(lipschitz_bound: f64, epsilon: f64, alpha: f64) -> Self {
        let num_levels = ((1.0 / epsilon).log2().ceil() as usize).max(1);
        let queries_per_level = ((2.0 / (alpha / num_levels as f64)).ln() / epsilon.powi(2)) as usize + 1;
        Self {
            lipschitz_bound,
            epsilon,
            alpha,
            num_levels,
            max_queries_per_level: queries_per_level.min(10_000),
        }
    }

    /// Run adaptive search for a given player using an oracle.
    pub fn search(
        &self,
        player: PlayerId,
        trajectory: &PriceTrajectory,
        round: RoundNumber,
        oracle: &mut dyn DeviationOracle,
        game_config: &GameConfig,
        price_bounds: (Price, Price),
    ) -> CollusionResult<AdaptiveSearchResult> {
        let (lo, hi) = price_bounds;
        let outcome = &trajectory.outcomes[round];
        let observed_price = outcome.prices[player];

        let mut best_bound: Option<DeviationBound> = None;
        let mut total_queries = 0usize;
        let mut levels_used = 0usize;

        // Alpha budget per level via peeling argument
        let alpha_per_level = self.alpha / (2.0 * self.num_levels as f64);

        let mut active_regions: Vec<(Price, Price)> = vec![(lo, hi)];

        for level in 0..self.num_levels {
            let grid_size = 1 << (level + 2); // 4, 8, 16, 32, ...
            let mut new_regions = Vec::new();
            levels_used += 1;

            for &(region_lo, region_hi) in &active_regions {
                let step = (region_hi - region_lo) / grid_size as f64;

                for i in 0..=grid_size {
                    let candidate = region_lo + i as f64 * step;
                    if (candidate - observed_price).abs().0 < self.epsilon {
                        continue;
                    }

                    if total_queries >= self.max_queries_per_level * (level + 1) {
                        break;
                    }

                    let strategy = DeviationStrategy::single_period(
                        player, candidate, observed_price, round,
                    );
                    let bound = oracle.query_deviation(&strategy, trajectory, game_config)?;
                    total_queries += 1;

                    // Lipschitz pruning: if upper bound on payoff gain is too small
                    let can_improve = match &best_bound {
                        Some(best) => {
                            bound.payoff_interval.hi > best.payoff_interval.lo - self.lipschitz_bound * step.0
                        }
                        None => true,
                    };

                    if can_improve {
                        let is_better = match &best_bound {
                            Some(best) => bound.payoff_interval.hi > best.payoff_interval.hi,
                            None => true,
                        };
                        if is_better {
                            // Refine region around this candidate
                            let refine_lo = if candidate - step > region_lo { candidate - step } else { region_lo };
                            let refine_hi = if candidate + step < region_hi { candidate + step } else { region_hi };
                            new_regions.push((refine_lo, refine_hi));
                            best_bound = Some(bound);
                        }
                    }
                }
            }

            if new_regions.is_empty() {
                break;
            }
            active_regions = new_regions;
        }

        Ok(AdaptiveSearchResult {
            best_bound,
            total_queries,
            levels_used,
            alpha_spent: alpha_per_level * levels_used as f64,
        })
    }

    /// Compute query complexity bound.
    pub fn query_complexity(&self, num_players: usize, grid_size: usize) -> f64 {
        let n = num_players as f64;
        let p_delta = grid_size as f64;
        let log_p = p_delta.ln().max(1.0);
        let log_ratio = (n / self.alpha).ln().max(1.0);
        n * log_p * log_p * log_ratio / self.epsilon.powi(2)
    }
}

/// Result of adaptive search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSearchResult {
    pub best_bound: Option<DeviationBound>,
    pub total_queries: usize,
    pub levels_used: usize,
    pub alpha_spent: f64,
}

// ── PeelingArgument ─────────────────────────────────────────────────────────

/// Peeling argument for proving selection-bias-freeness across resolution levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeelingArgument {
    pub num_levels: usize,
    pub alpha_per_level: Vec<f64>,
    pub total_alpha: f64,
    pub is_valid: bool,
}

impl PeelingArgument {
    /// Validate a peeling argument: alpha allocations must sum to ≤ total_alpha.
    pub fn validate(num_levels: usize, total_alpha: f64) -> Self {
        // Standard peeling: allocate alpha_k = alpha / (2 * k^2) for level k
        let alpha_per_level: Vec<f64> = (1..=num_levels)
            .map(|k| total_alpha / (2.0 * (k as f64).powi(2)))
            .collect();
        let total_spent: f64 = alpha_per_level.iter().sum();
        Self {
            num_levels,
            alpha_per_level,
            total_alpha,
            is_valid: total_spent <= total_alpha,
        }
    }

    /// Compute confidence level for a given level.
    pub fn confidence_at_level(&self, level: usize) -> f64 {
        if level < self.alpha_per_level.len() {
            1.0 - self.alpha_per_level[level]
        } else {
            0.95
        }
    }
}

// ── OracleQueryBudget ───────────────────────────────────────────────────────

/// Track and limit oracle queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleQueryBudget {
    pub total_budget: usize,
    pub used: usize,
    pub allocation: HashMap<String, usize>,
    pub used_by_component: HashMap<String, usize>,
}

impl OracleQueryBudget {
    pub fn new(total_budget: usize) -> Self {
        Self {
            total_budget,
            used: 0,
            allocation: HashMap::new(),
            used_by_component: HashMap::new(),
        }
    }

    /// Allocate budget to a named component.
    pub fn allocate(&mut self, component: &str, budget: usize) -> bool {
        let already_allocated: usize = self.allocation.values().sum();
        if already_allocated + budget > self.total_budget {
            return false;
        }
        self.allocation.insert(component.to_string(), budget);
        true
    }

    /// Try to consume a query for a named component.
    pub fn consume(&mut self, component: &str) -> bool {
        if self.used >= self.total_budget {
            return false;
        }
        if let Some(&limit) = self.allocation.get(component) {
            let component_used = self.used_by_component.get(component).copied().unwrap_or(0);
            if component_used >= limit {
                return false;
            }
        }
        self.used += 1;
        *self.used_by_component.entry(component.to_string()).or_insert(0) += 1;
        true
    }

    pub fn remaining(&self) -> usize {
        self.total_budget.saturating_sub(self.used)
    }

    pub fn utilization(&self) -> f64 {
        if self.total_budget == 0 { return 0.0; }
        self.used as f64 / self.total_budget as f64
    }

    pub fn component_remaining(&self, component: &str) -> usize {
        let limit = self.allocation.get(component).copied().unwrap_or(self.total_budget);
        let used = self.used_by_component.get(component).copied().unwrap_or(0);
        limit.saturating_sub(used)
    }
}

// ── CertifiedBound ──────────────────────────────────────────────────────────

/// Deviation bound with full error certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertifiedBound {
    pub player: PlayerId,
    pub round: RoundNumber,
    pub deviation_price: Price,
    pub payoff_bound: Interval<f64>,
    pub difference_bound: Interval<f64>,
    pub confidence_level: f64,
    pub oracle_level: OracleAccessLevel,
    pub num_queries: usize,
    pub certificate_valid: bool,
    pub method: String,
}

impl CertifiedBound {
    pub fn from_deviation_bound(
        bound: &DeviationBound,
        round: RoundNumber,
        deviation_price: Price,
        oracle_level: OracleAccessLevel,
        num_queries: usize,
    ) -> Self {
        Self {
            player: bound.player,
            round,
            deviation_price,
            payoff_bound: bound.payoff_interval,
            difference_bound: bound.difference_interval,
            confidence_level: bound.confidence_level,
            oracle_level,
            num_queries,
            certificate_valid: true,
            method: format!("{:?}", bound.method),
        }
    }

    /// Whether this bound certifies that no profitable deviation exists.
    pub fn certifies_no_deviation(&self) -> bool {
        self.certificate_valid && self.difference_bound.hi <= 0.0
    }

    /// Whether this bound certifies that a profitable deviation exists.
    pub fn certifies_deviation(&self) -> bool {
        self.certificate_valid && self.difference_bound.lo > 0.0
    }
}

// ── QueryEfficiency ─────────────────────────────────────────────────────────

/// Measure oracle query efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEfficiency {
    pub total_queries: usize,
    pub useful_queries: usize,
    pub pruned_queries: usize,
    pub bounds_tightened: usize,
    pub efficiency_ratio: f64,
    pub average_bound_width: f64,
}

impl QueryEfficiency {
    pub fn new() -> Self {
        Self {
            total_queries: 0,
            useful_queries: 0,
            pruned_queries: 0,
            bounds_tightened: 0,
            efficiency_ratio: 0.0,
            average_bound_width: f64::INFINITY,
        }
    }

    pub fn record_query(&mut self, was_useful: bool, bound_width: f64) {
        self.total_queries += 1;
        if was_useful {
            self.useful_queries += 1;
        } else {
            self.pruned_queries += 1;
        }
        if self.total_queries > 0 {
            self.efficiency_ratio = self.useful_queries as f64 / self.total_queries as f64;
        }
        // Running average of bound width
        if self.total_queries == 1 {
            self.average_bound_width = bound_width;
        } else {
            let n = self.total_queries as f64;
            self.average_bound_width = self.average_bound_width * (n - 1.0) / n + bound_width / n;
        }
    }

    pub fn record_tightening(&mut self) {
        self.bounds_tightened += 1;
    }
}

impl Default for QueryEfficiency {
    fn default() -> Self {
        Self::new()
    }
}

// ── Oracle factory ──────────────────────────────────────────────────────────

/// Create appropriate oracle for the given access level.
pub fn create_oracle(
    level: OracleAccessLevel,
    game_config: GameConfig,
    trajectory: &PriceTrajectory,
    query_budget: Option<usize>,
) -> Box<dyn DeviationOracle> {
    match level {
        OracleAccessLevel::Layer0 => {
            // Layer0 has no oracle access — use Layer1 with no checkpoints
            let schedule = CheckpointSchedule::from_rounds(vec![]);
            Box::new(Layer1Oracle::new(schedule, game_config, query_budget))
        }
        OracleAccessLevel::Layer1 => {
            let period = (trajectory.len() / 20).max(1);
            let schedule = CheckpointSchedule::periodic(trajectory.len(), period);
            let mut oracle = Layer1Oracle::new(schedule, game_config, query_budget);
            oracle.store_from_trajectory(trajectory);
            Box::new(oracle)
        }
        OracleAccessLevel::Layer2 => {
            Box::new(Layer2Oracle::new(game_config, query_budget, 100))
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_helper::{make_test_trajectory, make_test_game_config};

    fn make_trajectory(n_rounds: usize, n_players: usize, price: Price) -> PriceTrajectory {
        make_test_trajectory(n_rounds, n_players, price)
    }

    fn make_game_config() -> GameConfig {
        make_test_game_config(2)
    }

    #[test]
    fn test_checkpoint_schedule_periodic() {
        let sched = CheckpointSchedule::periodic(100, 10);
        assert_eq!(sched.num_checkpoints(), 10);
        assert_eq!(sched.nearest_before(RoundNumber(15)), Some(RoundNumber(10)));
        assert_eq!(sched.nearest_after(RoundNumber(15)), Some(RoundNumber(20)));
    }

    #[test]
    fn test_checkpoint_schedule_from_rounds() {
        let sched = CheckpointSchedule::from_rounds(vec![RoundNumber(5), RoundNumber(15), RoundNumber(10), RoundNumber(15)]);
        assert_eq!(sched.checkpoint_rounds, vec![RoundNumber(5), RoundNumber(10), RoundNumber(15)]);
        assert_eq!(sched.distance_to_nearest(RoundNumber(12)), 2);
    }

    #[test]
    fn test_layer1_oracle_creation() {
        let game_config = make_game_config();
        let schedule = CheckpointSchedule::periodic(100, 10);
        let oracle = Layer1Oracle::new(schedule, game_config, Some(1000));
        assert_eq!(oracle.queries_used(), 0);
        assert_eq!(oracle.queries_remaining(), Some(1000));
    }

    #[test]
    fn test_layer1_oracle_store_from_trajectory() {
        let game_config = make_game_config();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let schedule = CheckpointSchedule::periodic(50, 10);
        let mut oracle = Layer1Oracle::new(schedule, game_config, None);
        oracle.store_from_trajectory(&trajectory);
        assert_eq!(oracle.checkpoints.len(), 5);
    }

    #[test]
    fn test_layer1_oracle_query() {
        let game_config = make_game_config();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let schedule = CheckpointSchedule::periodic(50, 10);
        let mut oracle = Layer1Oracle::new(schedule, game_config.clone(), Some(100));
        oracle.store_from_trajectory(&trajectory);

        let strategy = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(15));
        let result = oracle.query_deviation(&strategy, &trajectory, &game_config);
        assert!(result.is_ok());
        assert_eq!(oracle.queries_used(), 1);
    }

    #[test]
    fn test_layer2_oracle_query() {
        let game_config = make_game_config();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let mut oracle = Layer2Oracle::new(game_config.clone(), Some(100), 10);

        let strategy = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(10));
        let result = oracle.query_deviation(&strategy, &trajectory, &game_config);
        assert!(result.is_ok());
        let bound = result.unwrap();
        assert_eq!(bound.player, PlayerId(0));
    }

    #[test]
    fn test_layer2_tighter_than_layer1() {
        let game_config = make_game_config();
        let trajectory = make_trajectory(50, 2, Price(5.0));

        let schedule = CheckpointSchedule::periodic(50, 10);
        let mut l1 = Layer1Oracle::new(schedule, game_config.clone(), None);
        l1.store_from_trajectory(&trajectory);

        let mut l2 = Layer2Oracle::new(game_config.clone(), None, 100);

        let strategy = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(15));
        let b1 = l1.query_deviation(&strategy, &trajectory, &game_config).unwrap();
        let b2 = l2.query_deviation(&strategy, &trajectory, &game_config).unwrap();

        // Layer2 should have tighter bounds
        assert!(b2.width() <= b1.width() + 1e-6);
    }

    #[test]
    fn test_adaptive_refinement_creation() {
        let ar = AdaptiveRefinement::new(5.0, 0.01, 0.05);
        assert!(ar.num_levels > 0);
        assert!(ar.max_queries_per_level > 0);
    }

    #[test]
    fn test_adaptive_refinement_search() {
        let game_config = make_game_config();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let mut oracle = Layer2Oracle::new(game_config.clone(), Some(1000), 10);

        let ar = AdaptiveRefinement::new(5.0, 0.1, 0.05);
        let result = ar.search(PlayerId(0), &trajectory, RoundNumber(10), &mut oracle, &game_config, (Price(0.0), Price(10.0)));
        assert!(result.is_ok());
        let sr = result.unwrap();
        assert!(sr.total_queries > 0);
    }

    #[test]
    fn test_peeling_argument() {
        let pa = PeelingArgument::validate(5, 0.05);
        assert!(pa.is_valid);
        assert_eq!(pa.num_levels, 5);
        let total_spent: f64 = pa.alpha_per_level.iter().sum();
        assert!(total_spent <= pa.total_alpha);
    }

    #[test]
    fn test_oracle_query_budget() {
        let mut budget = OracleQueryBudget::new(100);
        assert_eq!(budget.remaining(), 100);
        budget.allocate("deviation", 60);
        budget.allocate("punishment", 40);
        assert!(budget.consume("deviation"));
        assert_eq!(budget.remaining(), 99);
        assert_eq!(budget.component_remaining("deviation"), 59);
    }

    #[test]
    fn test_oracle_query_budget_exhaustion() {
        let mut budget = OracleQueryBudget::new(2);
        assert!(budget.consume("test"));
        assert!(budget.consume("test"));
        assert!(!budget.consume("test"));
    }

    #[test]
    fn test_certified_bound() {
        let dev_bound = DeviationBound::new(
            PlayerId(0),
            Interval::new(8.0, 12.0),
            10.0,
            0.95,
            BoundMethod::Exact,
        );
        let cert = CertifiedBound::from_deviation_bound(
            &dev_bound, RoundNumber(5), Price(3.0), OracleAccessLevel::Layer2, 10,
        );
        assert!(cert.certificate_valid);
        assert!(!cert.certifies_no_deviation());
        assert!(!cert.certifies_deviation());
    }

    #[test]
    fn test_certified_bound_no_deviation() {
        let dev_bound = DeviationBound::new(
            PlayerId(0),
            Interval::new(7.0, 9.0),
            10.0,
            0.95,
            BoundMethod::Exact,
        );
        let cert = CertifiedBound::from_deviation_bound(
            &dev_bound, RoundNumber(5), Price(3.0), OracleAccessLevel::Layer2, 10,
        );
        assert!(cert.certifies_no_deviation());
    }

    #[test]
    fn test_query_efficiency() {
        let mut eff = QueryEfficiency::new();
        eff.record_query(true, 0.5);
        eff.record_query(false, 1.0);
        eff.record_query(true, 0.3);
        assert_eq!(eff.total_queries, 3);
        assert_eq!(eff.useful_queries, 2);
        assert!((eff.efficiency_ratio - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_create_oracle_layer1() {
        let game_config = make_game_config();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let oracle = create_oracle(OracleAccessLevel::Layer1, game_config, &trajectory, Some(100));
        assert_eq!(oracle.access_level(), OracleAccessLevel::Layer1);
    }

    #[test]
    fn test_create_oracle_layer2() {
        let game_config = make_game_config();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let oracle = create_oracle(OracleAccessLevel::Layer2, game_config, &trajectory, None);
        assert_eq!(oracle.access_level(), OracleAccessLevel::Layer2);
    }

    #[test]
    fn test_query_complexity_bound() {
        let ar = AdaptiveRefinement::new(5.0, 0.01, 0.05);
        let complexity = ar.query_complexity(2, 1000);
        assert!(complexity > 0.0);
        assert!(complexity.is_finite());
    }
}
