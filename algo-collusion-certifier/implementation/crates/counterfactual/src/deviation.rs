//! Deviation strategy analysis for counterfactual reasoning.
//!
//! Enumerates, evaluates, and organizes single-period and multi-period
//! deviations. Supports Lipschitz-aware pruning and coarse-to-fine refinement.

use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, GameConfig, IntervalF64 as Interval,
    MarketOutcome, PlayerId, Price, PriceTrajectory, Profit, RoundNumber,
    Cost, DemandSystem, PlayerAction, Quantity,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::market_helper::{simulate_single_round, compute_best_response_price, price_bounds_from_config};

// ── DeviationStrategy ───────────────────────────────────────────────────────

/// Specification of a unilateral deviation: who deviates, to what price, from which round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationStrategy {
    /// Which player deviates.
    pub player: PlayerId,
    /// The price the player deviates to.
    pub deviation_price: Price,
    /// The original (observed) price.
    pub original_price: Price,
    /// Round at which deviation begins.
    pub start_round: RoundNumber,
    /// Number of rounds the deviation lasts (1 for single-period).
    pub duration: usize,
}

impl DeviationStrategy {
    pub fn single_period(player: PlayerId, deviation_price: Price, original_price: Price, round: RoundNumber) -> Self {
        Self {
            player,
            deviation_price,
            original_price,
            start_round: round,
            duration: 1,
        }
    }

    pub fn multi_period(player: PlayerId, deviation_price: Price, original_price: Price, start: RoundNumber, duration: usize) -> Self {
        Self {
            player,
            deviation_price,
            original_price,
            start_round: start,
            duration,
        }
    }

    /// Absolute price change magnitude.
    pub fn price_change(&self) -> f64 {
        (self.deviation_price - self.original_price).abs().0
    }

    /// Whether this is a downward deviation (undercutting).
    pub fn is_undercut(&self) -> bool {
        self.deviation_price < self.original_price
    }
}

// ── DeviationResult ─────────────────────────────────────────────────────────

/// Result of evaluating a deviation strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationResult {
    pub player: PlayerId,
    pub strategy: DeviationStrategy,
    pub observed_payoff: f64,
    pub deviation_payoff: f64,
    pub payoff_difference: f64,
    pub is_profitable: bool,
    pub confidence_interval: Option<ConfidenceInterval>,
    pub num_simulations: usize,
}

impl DeviationResult {
    pub fn new(player: PlayerId, strategy: DeviationStrategy, observed: f64, deviation: f64) -> Self {
        let diff = deviation - observed;
        Self {
            player,
            strategy,
            observed_payoff: observed,
            deviation_payoff: deviation,
            payoff_difference: diff,
            is_profitable: diff > 0.0,
            confidence_interval: None,
            num_simulations: 0,
        }
    }

    pub fn with_ci(mut self, ci: ConfidenceInterval) -> Self {
        self.confidence_interval = Some(ci);
        self
    }

    pub fn with_simulations(mut self, n: usize) -> Self {
        self.num_simulations = n;
        self
    }

    /// Relative payoff improvement as fraction of observed payoff.
    pub fn relative_improvement(&self) -> f64 {
        if self.observed_payoff.abs() < 1e-12 {
            return 0.0;
        }
        self.payoff_difference / self.observed_payoff.abs()
    }
}

// ── DeviationBound ──────────────────────────────────────────────────────────

/// Certified upper/lower bound on deviation payoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationBound {
    pub player: PlayerId,
    pub payoff_interval: Interval,
    pub difference_interval: Interval,
    pub confidence_level: f64,
    pub method: BoundMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundMethod {
    Exact,
    MonteCarlo,
    Lipschitz,
    Checkpoint,
}

impl DeviationBound {
    pub fn new(player: PlayerId, payoff_interval: Interval, observed: f64, confidence: f64, method: BoundMethod) -> Self {
        let diff_lo = payoff_interval.lo - observed;
        let diff_hi = payoff_interval.hi - observed;
        Self {
            player,
            payoff_interval,
            difference_interval: Interval::new(diff_lo, diff_hi),
            confidence_level: confidence,
            method,
        }
    }

    /// Whether the bound certifies no profitable deviation.
    pub fn certifies_no_profit(&self) -> bool {
        self.difference_interval.hi <= 0.0
    }

    /// Whether the bound certifies a profitable deviation.
    pub fn certifies_profit(&self) -> bool {
        self.difference_interval.lo > 0.0
    }

    pub fn width(&self) -> f64 {
        self.payoff_interval.width()
    }
}

// ── DeviationEnumerator ─────────────────────────────────────────────────────

/// Configuration for deviation enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationEnumeratorConfig {
    /// Minimum price in grid.
    pub price_min: Price,
    /// Maximum price in grid.
    pub price_max: Price,
    /// Number of points in coarse grid.
    pub coarse_grid_size: usize,
    /// Number of points in fine grid (for refinement).
    pub fine_grid_size: usize,
    /// Lipschitz constant bound for payoff function (if known).
    pub lipschitz_bound: Option<f64>,
    /// Minimum price difference to consider a deviation.
    pub min_price_delta: f64,
    /// Maximum number of deviations to enumerate per player.
    pub max_deviations_per_player: usize,
    /// Number of top deviations to refine.
    pub top_k_refine: usize,
}

impl Default for DeviationEnumeratorConfig {
    fn default() -> Self {
        Self {
            price_min: Price(0.0),
            price_max: Price(10.0),
            coarse_grid_size: 50,
            fine_grid_size: 200,
            lipschitz_bound: None,
            min_price_delta: 0.01,
            max_deviations_per_player: 500,
            top_k_refine: 10,
        }
    }
}

/// Enumerate meaningful single-period deviations for all players.
pub struct DeviationEnumerator {
    config: DeviationEnumeratorConfig,
    game_config: GameConfig,
}

impl DeviationEnumerator {
    pub fn new(config: DeviationEnumeratorConfig, game_config: GameConfig) -> Self {
        Self { config, game_config }
    }

    /// Enumerate all meaningful deviations for a given round.
    pub fn enumerate_deviations(
        &self,
        trajectory: &PriceTrajectory,
        round: RoundNumber,
    ) -> Vec<DeviationStrategy> {
        if round.0 >= trajectory.len() {
            return Vec::new();
        }
        let outcome = &trajectory.outcomes[round];
        let mut deviations = Vec::new();

        for player in 0..trajectory.num_players {
            let observed_price = outcome.prices[player];
            let player_devs = self.enumerate_for_player(PlayerId(player), observed_price, round);
            deviations.extend(player_devs);
        }

        deviations
    }

    /// Enumerate deviations for a single player using coarse grid.
    fn enumerate_for_player(
        &self,
        player: PlayerId,
        observed_price: Price,
        round: RoundNumber,
    ) -> Vec<DeviationStrategy> {
        let step = (self.config.price_max.0 - self.config.price_min.0) / self.config.coarse_grid_size as f64;
        let mut deviations = Vec::new();

        for i in 0..=self.config.coarse_grid_size {
            let candidate = Price(self.config.price_min.0 + i as f64 * step);
            let delta = (candidate - observed_price).abs().0;

            if delta < self.config.min_price_delta {
                continue;
            }

            // Lipschitz pruning: skip if payoff difference is bounded below threshold
            if let Some(lip) = self.config.lipschitz_bound {
                if lip * delta < self.config.min_price_delta {
                    continue;
                }
            }

            deviations.push(DeviationStrategy::single_period(
                player, candidate, observed_price, round,
            ));

            if deviations.len() >= self.config.max_deviations_per_player {
                break;
            }
        }

        deviations
    }

    /// Coarse-to-fine: enumerate coarse, evaluate, refine around top-K.
    pub fn coarse_to_fine(
        &self,
        trajectory: &PriceTrajectory,
        round: RoundNumber,
        evaluator: &dyn Fn(&DeviationStrategy) -> f64,
    ) -> Vec<(DeviationStrategy, f64)> {
        let coarse = self.enumerate_deviations(trajectory, round);

        // Evaluate all coarse deviations
        let mut scored: Vec<(DeviationStrategy, f64)> = coarse
            .into_iter()
            .map(|d| {
                let score = evaluator(&d);
                (d, score)
            })
            .collect();

        // Sort by payoff (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K for refinement
        let top_k: Vec<_> = scored.iter().take(self.config.top_k_refine).cloned().collect();

        let mut refined = scored;

        // Refine around each top-K deviation
        for (dev, _score) in &top_k {
            let center = dev.deviation_price;
            let coarse_step = (self.config.price_max - self.config.price_min)
                / self.config.coarse_grid_size as f64;
            let refine_lo = (center - coarse_step).max(self.config.price_min);
            let refine_hi = (center + coarse_step).min(self.config.price_max);
            let fine_step = (refine_hi - refine_lo) / self.config.fine_grid_size as f64;

            for i in 0..=self.config.fine_grid_size {
                let candidate = refine_lo + i as f64 * fine_step;
                let delta = (candidate - dev.original_price).abs();
                if delta.0 < self.config.min_price_delta {
                    continue;
                }

                let fine_dev = DeviationStrategy::single_period(
                    dev.player,
                    candidate,
                    dev.original_price,
                    dev.start_round,
                );
                let fine_score = evaluator(&fine_dev);
                refined.push((fine_dev, fine_score));
            }
        }

        refined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        refined
    }

    /// Enumerate deviations for ALL rounds.
    pub fn enumerate_all_rounds(
        &self,
        trajectory: &PriceTrajectory,
    ) -> Vec<DeviationStrategy> {
        let mut all = Vec::new();
        for round in 0..trajectory.len() {
            all.extend(self.enumerate_deviations(trajectory, RoundNumber(round)));
        }
        all
    }
}

// ── OptimalDeviation ────────────────────────────────────────────────────────

/// Find the most profitable unilateral single-period deviation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalDeviation {
    pub result: Option<DeviationResult>,
    pub num_evaluated: usize,
    pub search_method: String,
}

impl OptimalDeviation {
    /// Find the optimal deviation for a given player at a given round.
    pub fn find(
        player: PlayerId,
        trajectory: &PriceTrajectory,
        round: RoundNumber,
        game_config: &GameConfig,
        price_bounds: (Price, Price),
        grid_size: usize,
    ) -> CollusionResult<Self> {
        if round.0 >= trajectory.len() {
            return Err(CollusionError::InvalidState(
                format!("Round {} exceeds trajectory length {}", round.0, trajectory.len()),
            ));
        }

        let outcome = &trajectory.outcomes[round];
        let observed_price = outcome.prices[player];
        let observed_profit = outcome.profits[player];

        let mut best_price = observed_price;
        let mut best_profit = observed_profit;
        let mut num_evaluated = 0usize;

        let (lo, hi) = price_bounds;
        let step = (hi - lo) / grid_size as f64;

        for i in 0..=grid_size {
            let candidate = lo + i as f64 * step;
            if (candidate - observed_price).abs().0 < 1e-10 {
                continue;
            }

            let mut dev_prices = outcome.prices.clone();
            dev_prices[player.0] = candidate;

            let dev_outcome = simulate_single_round(&dev_prices, game_config, RoundNumber(0));
            let dev_profit = dev_outcome.profits[player];
            num_evaluated += 1;

            if dev_profit > best_profit {
                best_profit = dev_profit;
                best_price = candidate;
            }
        }

        let strategy = DeviationStrategy::single_period(player, best_price, observed_price, round);
        let result = DeviationResult::new(player, strategy, observed_profit.0, best_profit.0);

        Ok(Self {
            result: Some(result),
            num_evaluated,
            search_method: "grid_search".to_string(),
        })
    }

    /// Binary search for optimal deviation price (requires unimodal payoff).
    pub fn find_binary(
        player: PlayerId,
        trajectory: &PriceTrajectory,
        round: RoundNumber,
        game_config: &GameConfig,
        price_bounds: (Price, Price),
        tolerance: f64,
    ) -> CollusionResult<Self> {
        if round.0 >= trajectory.len() {
            return Err(CollusionError::InvalidState("Round out of range".into()));
        }

        let outcome = &trajectory.outcomes[round];
        let observed_price = outcome.prices[player];
        let observed_profit = outcome.profits[player];

        let evaluate = |p: Price| -> f64 {
            let mut dev_prices = outcome.prices.clone();
            dev_prices[player.0] = p;
            let dev_outcome = simulate_single_round(&dev_prices, game_config, RoundNumber(0));
            dev_outcome.profits[player].0
        };

        let (mut lo, mut hi) = price_bounds;
        let mut num_evaluated = 0usize;

        // Golden section search
        let phi = (5.0_f64.sqrt() + 1.0) / 2.0;
        while (hi - lo).0 > tolerance {
            let m1 = hi - (hi - lo) / phi;
            let m2 = lo + (hi - lo) / phi;
            let f1 = evaluate(m1);
            let f2 = evaluate(m2);
            num_evaluated += 2;

            if f1 < f2 {
                lo = m1;
            } else {
                hi = m2;
            }
        }

        let best_price = (lo + hi) / 2.0;
        let best_profit = evaluate(best_price);
        num_evaluated += 1;

        let strategy = DeviationStrategy::single_period(player, best_price, observed_price, round);
        let result = DeviationResult::new(player, strategy, observed_profit.0, best_profit);

        Ok(Self {
            result: Some(result),
            num_evaluated,
            search_method: "golden_section".to_string(),
        })
    }
}

// ── MultiPeriodDeviation ────────────────────────────────────────────────────

/// Multi-period deviation: deviate for K consecutive periods.
/// Uses dynamic programming for optimal multi-period sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPeriodDeviation {
    pub player: PlayerId,
    pub deviation_prices: Vec<Price>,
    pub start_round: RoundNumber,
    pub duration: usize,
    pub total_observed_payoff: f64,
    pub total_deviation_payoff: f64,
    pub discounted_gain: f64,
    pub discount_factor: f64,
}

impl MultiPeriodDeviation {
    /// Compute optimal multi-period deviation via dynamic programming.
    ///
    /// For each period t in [start, start+K), find the price that maximizes
    /// the discounted sum of deviation payoffs, assuming opponents play their
    /// observed strategies.
    pub fn find_optimal(
        player: PlayerId,
        trajectory: &PriceTrajectory,
        start_round: RoundNumber,
        duration: usize,
        game_config: &GameConfig,
        price_bounds: (Price, Price),
        grid_size: usize,
    ) -> CollusionResult<Self> {
        let end_val = (start_round.0 + duration).min(trajectory.len());
        let actual_duration = end_val - start_round.0;
        if actual_duration == 0 {
            return Err(CollusionError::InvalidState("No rounds available".into()));
        }

        let discount = game_config.discount_factor;
        let (lo, hi) = price_bounds;
        let step = (hi - lo) / grid_size as f64;

        let mut deviation_prices = Vec::with_capacity(actual_duration);
        let mut total_obs_payoff = 0.0f64;
        let mut total_dev_payoff = 0.0f64;

        // Forward pass: greedily select best deviation price per round
        // (simplified DP — for true optimality with opponent response, would need backward induction)
        for t_idx in start_round.0..end_val {
            let t = RoundNumber(t_idx);
            let outcome = &trajectory.outcomes[t];
            let obs_profit = outcome.profits[player];
            let dt = (t_idx - start_round.0) as i32;
            total_obs_payoff += discount.powi(dt) * obs_profit.0;

            let mut best_p = outcome.prices[player];
            let mut best_profit = obs_profit;

            for i in 0..=grid_size {
                let candidate = lo + i as f64 * step;
                let mut dev_prices = outcome.prices.clone();
                dev_prices[player.0] = candidate;

                let dev_outcome = simulate_single_round(&dev_prices, game_config, t);
                let dev_profit = dev_outcome.profits[player];

                if dev_profit > best_profit {
                    best_profit = dev_profit;
                    best_p = candidate;
                }
            }

            deviation_prices.push(best_p);
            total_dev_payoff += discount.powi(dt) * best_profit.0;
        }

        Ok(Self {
            player,
            deviation_prices,
            start_round,
            duration: actual_duration,
            total_observed_payoff: total_obs_payoff,
            total_deviation_payoff: total_dev_payoff,
            discounted_gain: total_dev_payoff - total_obs_payoff,
            discount_factor: discount,
        })
    }

    /// Backward-induction DP for optimal multi-period deviation.
    /// V[t] = max_p { pi(p, p_{-i,t}) + delta * V[t+1] }
    pub fn find_dp(
        player: PlayerId,
        trajectory: &PriceTrajectory,
        start_round: RoundNumber,
        duration: usize,
        game_config: &GameConfig,
        price_bounds: (Price, Price),
        grid_size: usize,
    ) -> CollusionResult<Self> {
        let end_val = (start_round.0 + duration).min(trajectory.len());
        let actual_duration = end_val - start_round.0;
        if actual_duration == 0 {
            return Err(CollusionError::InvalidState("No rounds for DP".into()));
        }

        let discount = game_config.discount_factor;
        let (lo, hi) = price_bounds;
        let step = (hi - lo) / grid_size as f64;

        // value_table[t_offset] = best continuation value from round (start_round + t_offset)
        let mut value_table = vec![0.0f64; actual_duration + 1];
        let mut policy = vec![Price(0.0); actual_duration];

        let evaluate_profit = |t: RoundNumber, p: Price| -> f64 {
            let outcome = &trajectory.outcomes[t];
            let mut dev_prices = outcome.prices.clone();
            dev_prices[player.0] = p;
            let dev_outcome = simulate_single_round(&dev_prices, game_config, t);
            dev_outcome.profits[player].0
        };

        // Backward induction
        for offset in (0..actual_duration).rev() {
            let t = start_round + offset;
            let mut best_value = f64::NEG_INFINITY;
            let mut best_price = lo;

            for i in 0..=grid_size {
                let candidate = lo + i as f64 * step;
                let immediate = evaluate_profit(t, candidate);
                let continuation = discount * value_table[offset + 1];
                let total = immediate + continuation;

                if total > best_value {
                    best_value = total;
                    best_price = candidate;
                }
            }

            value_table[offset] = best_value;
            policy[offset] = best_price;
        }

        let mut total_obs_payoff = 0.0f64;
        for (offset, t_idx) in (start_round.0..end_val).enumerate() {
            let obs_profit = trajectory.outcomes[t_idx].profits[player];
            total_obs_payoff += discount.powi(offset as i32) * obs_profit.0;
        }

        Ok(Self {
            player,
            deviation_prices: policy,
            start_round,
            duration: actual_duration,
            total_observed_payoff: total_obs_payoff,
            total_deviation_payoff: value_table[0],
            discounted_gain: value_table[0] - total_obs_payoff,
            discount_factor: discount,
        })
    }

    pub fn is_profitable(&self) -> bool {
        self.discounted_gain > 0.0
    }
}

// ── DeviationProfile ────────────────────────────────────────────────────────

/// Collection of deviation results for all players.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationProfile {
    pub results: Vec<DeviationResult>,
    pub num_players: usize,
    pub max_deviation_gain: f64,
    pub any_profitable: bool,
}

impl DeviationProfile {
    pub fn new(num_players: usize) -> Self {
        Self {
            results: Vec::new(),
            num_players,
            max_deviation_gain: 0.0,
            any_profitable: false,
        }
    }

    pub fn add_result(&mut self, result: DeviationResult) {
        if result.payoff_difference > self.max_deviation_gain {
            self.max_deviation_gain = result.payoff_difference;
        }
        if result.is_profitable {
            self.any_profitable = true;
        }
        self.results.push(result);
    }

    /// Get the most profitable deviation across all players.
    pub fn most_profitable(&self) -> Option<&DeviationResult> {
        self.results
            .iter()
            .max_by(|a, b| a.payoff_difference.partial_cmp(&b.payoff_difference).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get results for a specific player.
    pub fn for_player(&self, player: PlayerId) -> Vec<&DeviationResult> {
        self.results.iter().filter(|r| r.player == player).collect()
    }

    /// Compute epsilon-Nash threshold: max deviation gain across all players.
    pub fn epsilon_nash(&self) -> f64 {
        self.results
            .iter()
            .map(|r| r.payoff_difference.max(0.0))
            .fold(0.0f64, f64::max)
    }
}

// ── ProfitableDeviation ─────────────────────────────────────────────────────

/// A deviation with strictly positive expected gain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitableDeviation {
    pub strategy: DeviationStrategy,
    pub expected_gain: f64,
    pub gain_ci: ConfidenceInterval,
    pub significance: f64,
}

impl ProfitableDeviation {
    pub fn from_result(result: &DeviationResult, ci: ConfidenceInterval) -> Option<Self> {
        if result.payoff_difference > 0.0 {
            Some(Self {
                strategy: result.strategy.clone(),
                expected_gain: result.payoff_difference,
                gain_ci: ci,
                significance: if ci.lower > 0.0 { 1.0 } else { 0.5 },
            })
        } else {
            None
        }
    }

    /// Whether the gain is statistically significant (CI excludes zero).
    pub fn is_significant(&self) -> bool {
        self.gain_ci.lower > 0.0
    }
}

// ── DeviationCatalog ────────────────────────────────────────────────────────

/// Organized collection of all tested deviations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationCatalog {
    /// All deviation results indexed by (player, round).
    entries: Vec<CatalogEntry>,
    /// Total number of deviations evaluated.
    pub total_evaluated: usize,
    /// Number of profitable deviations found.
    pub num_profitable: usize,
    /// Maximum gain found.
    pub max_gain: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub player: PlayerId,
    pub round: RoundNumber,
    pub results: Vec<DeviationResult>,
    pub best_result: Option<DeviationResult>,
}

impl DeviationCatalog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_evaluated: 0,
            num_profitable: 0,
            max_gain: 0.0,
        }
    }

    pub fn add_entry(&mut self, player: PlayerId, round: RoundNumber, results: Vec<DeviationResult>) {
        let best = results
            .iter()
            .max_by(|a, b| a.payoff_difference.partial_cmp(&b.payoff_difference).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        for r in &results {
            self.total_evaluated += 1;
            if r.is_profitable {
                self.num_profitable += 1;
            }
            if r.payoff_difference > self.max_gain {
                self.max_gain = r.payoff_difference;
            }
        }

        self.entries.push(CatalogEntry {
            player,
            round,
            results,
            best_result: best,
        });
    }

    /// Get all profitable deviations.
    pub fn profitable_deviations(&self) -> Vec<&DeviationResult> {
        self.entries
            .iter()
            .flat_map(|e| e.results.iter())
            .filter(|r| r.is_profitable)
            .collect()
    }

    /// Get the globally optimal deviation.
    pub fn global_optimum(&self) -> Option<&DeviationResult> {
        self.entries
            .iter()
            .flat_map(|e| e.best_result.as_ref())
            .max_by(|a, b| a.payoff_difference.partial_cmp(&b.payoff_difference).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get entries for a specific player.
    pub fn for_player(&self, player: PlayerId) -> Vec<&CatalogEntry> {
        self.entries.iter().filter(|e| e.player == player).collect()
    }

    /// Get entry for a specific player and round.
    pub fn get(&self, player: PlayerId, round: RoundNumber) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.player == player && e.round == round)
    }

    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Summary statistics for the catalog.
    pub fn summary(&self) -> CatalogSummary {
        let gains: Vec<f64> = self.entries
            .iter()
            .flat_map(|e| e.results.iter())
            .map(|r| r.payoff_difference)
            .collect();

        let mean_gain = if gains.is_empty() {
            0.0
        } else {
            gains.iter().sum::<f64>() / gains.len() as f64
        };

        let variance = if gains.len() < 2 {
            0.0
        } else {
            let n = gains.len() as f64;
            gains.iter().map(|g| (g - mean_gain).powi(2)).sum::<f64>() / (n - 1.0)
        };

        CatalogSummary {
            total_evaluated: self.total_evaluated,
            num_profitable: self.num_profitable,
            max_gain: self.max_gain,
            mean_gain,
            std_gain: variance.sqrt(),
            fraction_profitable: if self.total_evaluated > 0 {
                self.num_profitable as f64 / self.total_evaluated as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for DeviationCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSummary {
    pub total_evaluated: usize,
    pub num_profitable: usize,
    pub max_gain: f64,
    pub mean_gain: f64,
    pub std_gain: f64,
    pub fraction_profitable: f64,
}

// ── Deviation incentive computation ─────────────────────────────────────────

/// Compute the deviation incentive for a player given a trajectory and equilibrium.
///
/// Returns the maximum payoff improvement from any single-period deviation
/// minus the on-path payoff, averaged across specified rounds.
pub fn deviation_incentive(
    player: PlayerId,
    trajectory: &PriceTrajectory,
    game_config: &GameConfig,
    price_bounds: (Price, Price),
    grid_size: usize,
) -> f64 {
    if trajectory.is_empty() {
        return 0.0;
    }

    let mut total_incentive = 0.0f64;
    let num_rounds = trajectory.len();

    for round in 0..num_rounds {
        let outcome = &trajectory.outcomes[round];
        let observed_profit = outcome.profits[player];

        // Find best response profit
        let opponent_prices: Vec<Price> = outcome.prices.iter().enumerate()
            .filter(|(i, _)| *i != player)
            .map(|(_, &p)| p)
            .collect();

        let mc = game_config.marginal_costs.get(player.0).copied().unwrap_or(Cost(1.0));
        let best_resp = compute_best_response_price(
            player.0,
            &opponent_prices,
            mc,
            &game_config.demand_system,
            price_bounds,
            grid_size,
        );

        // Compute profit at best response
        let mut dev_prices = outcome.prices.clone();
        dev_prices[player.0] = best_resp;

        let dev_outcome = simulate_single_round(&dev_prices, game_config, RoundNumber(round));
        let dev_profit = dev_outcome.profits[player];

        total_incentive += (dev_profit.0 - observed_profit.0).max(0.0);
    }

    total_incentive / num_rounds as f64
}

/// Compute deviation incentives for all players.
pub fn all_deviation_incentives(
    trajectory: &PriceTrajectory,
    game_config: &GameConfig,
    price_bounds: (Price, Price),
    grid_size: usize,
) -> Vec<f64> {
    (0..trajectory.num_players)
        .map(|player| deviation_incentive(PlayerId(player), trajectory, game_config, price_bounds, grid_size))
        .collect()
}

// ── Lipschitz-aware deviation analysis ──────────────────────────────────────

/// Estimate Lipschitz constant of payoff function for a player.
pub fn estimate_lipschitz_constant(
    player: PlayerId,
    trajectory: &PriceTrajectory,
    game_config: &GameConfig,
    price_bounds: (Price, Price),
    num_samples: usize,
) -> f64 {
    if trajectory.is_empty() || num_samples < 2 {
        return f64::INFINITY;
    }

    let round = trajectory.len() / 2; // Use middle round
    let outcome = &trajectory.outcomes[round];
    let (lo, hi) = price_bounds;
    let step = (hi - lo) / num_samples as f64;

    let evaluate = |p: Price| -> f64 {
        let mut dev_prices = outcome.prices.clone();
        dev_prices[player.0] = p;
        let dev_outcome = simulate_single_round(&dev_prices, game_config, RoundNumber(round));
        dev_outcome.profits[player].0
    };

    let mut max_lip = 0.0f64;
    let mut prev_price = lo;
    let mut prev_profit = evaluate(lo);

    for i in 1..=num_samples {
        let price = lo + i as f64 * step;
        let profit = evaluate(price);
        let dp = (profit - prev_profit).abs();
        let dx = (price - prev_price).0;
        if dx > 1e-15 {
            let lip = dp / dx;
            max_lip = max_lip.max(lip);
        }
        prev_price = price;
        prev_profit = profit;
    }

    max_lip
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::MarketType;
    use crate::market_helper::{make_test_trajectory, make_test_game_config};

    fn make_trajectory(n_rounds: usize, n_players: usize, price: Price) -> PriceTrajectory {
        make_test_trajectory(n_rounds, n_players, price)
    }

    fn make_game_config(n_players: usize) -> GameConfig {
        make_test_game_config(n_players)
    }

    #[test]
    fn test_deviation_strategy_creation() {
        let dev = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(10));
        assert_eq!(dev.player, PlayerId(0));
        assert_eq!(dev.deviation_price, Price(3.0));
        assert_eq!(dev.original_price, Price(5.0));
        assert_eq!(dev.start_round, RoundNumber(10));
        assert_eq!(dev.duration, 1);
        assert!(dev.is_undercut());
        assert!((dev.price_change() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_deviation_strategy_multi_period() {
        let dev = DeviationStrategy::multi_period(PlayerId(1), Price(4.0), Price(5.0), RoundNumber(5), 3);
        assert_eq!(dev.duration, 3);
        assert!(dev.is_undercut());
    }

    #[test]
    fn test_deviation_result_basic() {
        let strategy = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        let result = DeviationResult::new(PlayerId(0), strategy, 10.0, 15.0);
        assert!(result.is_profitable);
        assert!((result.payoff_difference - 5.0).abs() < 1e-10);
        assert!((result.relative_improvement() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_deviation_result_not_profitable() {
        let strategy = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        let result = DeviationResult::new(PlayerId(0), strategy, 15.0, 10.0);
        assert!(!result.is_profitable);
        assert!(result.payoff_difference < 0.0);
    }

    #[test]
    fn test_deviation_bound_no_profit() {
        let bound = DeviationBound::new(PlayerId(0), Interval::new(8.0, 9.0), 10.0, 0.95, BoundMethod::Exact);
        assert!(bound.certifies_no_profit());
        assert!(!bound.certifies_profit());
    }

    #[test]
    fn test_deviation_bound_profitable() {
        let bound = DeviationBound::new(PlayerId(0), Interval::new(12.0, 15.0), 10.0, 0.95, BoundMethod::MonteCarlo);
        assert!(!bound.certifies_no_profit());
        assert!(bound.certifies_profit());
    }

    #[test]
    fn test_deviation_enumerator_basic() {
        let config = DeviationEnumeratorConfig {
            price_min: Price(0.0),
            price_max: Price(10.0),
            coarse_grid_size: 20,
            min_price_delta: 0.1,
            ..Default::default()
        };
        let game_config = make_game_config(2);
        let enumerator = DeviationEnumerator::new(config, game_config);
        let trajectory = make_trajectory(5, 2, Price(5.0));
        let devs = enumerator.enumerate_deviations(&trajectory, RoundNumber(0));
        assert!(!devs.is_empty());
        assert!(devs.iter().all(|d| (d.deviation_price - Price(5.0)).abs().0 >= 0.1));
    }

    #[test]
    fn test_deviation_enumerator_out_of_range() {
        let config = DeviationEnumeratorConfig::default();
        let game_config = make_game_config(2);
        let enumerator = DeviationEnumerator::new(config, game_config);
        let trajectory = make_trajectory(5, 2, Price(5.0));
        let devs = enumerator.enumerate_deviations(&trajectory, RoundNumber(100));
        assert!(devs.is_empty());
    }

    #[test]
    fn test_optimal_deviation_find() {
        let trajectory = make_trajectory(10, 2, Price(5.0));
        let game_config = make_game_config(2);
        let result = OptimalDeviation::find(PlayerId(0), &trajectory, RoundNumber(0), &game_config, (Price(0.0), Price(10.0)), 50);
        assert!(result.is_ok());
        let opt = result.unwrap();
        assert!(opt.num_evaluated > 0);
    }

    #[test]
    fn test_optimal_deviation_binary() {
        let trajectory = make_trajectory(10, 2, Price(5.0));
        let game_config = make_game_config(2);
        let result = OptimalDeviation::find_binary(PlayerId(0), &trajectory, RoundNumber(0), &game_config, (Price(0.0), Price(10.0)), 0.01);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deviation_profile() {
        let mut profile = DeviationProfile::new(2);
        let s1 = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        let s2 = DeviationStrategy::single_period(PlayerId(1), Price(4.0), Price(5.0), RoundNumber(0));
        profile.add_result(DeviationResult::new(PlayerId(0), s1, 10.0, 15.0));
        profile.add_result(DeviationResult::new(PlayerId(1), s2, 10.0, 8.0));
        assert!(profile.any_profitable);
        assert!((profile.max_deviation_gain - 5.0).abs() < 1e-10);
        assert_eq!(profile.most_profitable().unwrap().player, PlayerId(0));
        assert!((profile.epsilon_nash() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_deviation_catalog() {
        let mut catalog = DeviationCatalog::new();
        let s1 = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        let r1 = DeviationResult::new(PlayerId(0), s1, 10.0, 15.0);
        catalog.add_entry(PlayerId(0), RoundNumber(0), vec![r1]);
        assert_eq!(catalog.total_evaluated, 1);
        assert_eq!(catalog.num_profitable, 1);
        assert!(!catalog.profitable_deviations().is_empty());
        let summary = catalog.summary();
        assert!((summary.fraction_profitable - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_catalog_global_optimum() {
        let mut catalog = DeviationCatalog::new();
        let s1 = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        let s2 = DeviationStrategy::single_period(PlayerId(1), Price(2.0), Price(5.0), RoundNumber(1));
        catalog.add_entry(PlayerId(0), RoundNumber(0), vec![DeviationResult::new(PlayerId(0), s1, 10.0, 12.0)]);
        catalog.add_entry(PlayerId(1), RoundNumber(1), vec![DeviationResult::new(PlayerId(1), s2, 10.0, 18.0)]);
        let opt = catalog.global_optimum().unwrap();
        assert_eq!(opt.player, PlayerId(1));
    }

    #[test]
    fn test_profitable_deviation_creation() {
        let s = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        let result = DeviationResult::new(PlayerId(0), s, 10.0, 15.0);
        let ci = ConfidenceInterval::new(3.0, 7.0, 0.95, 5.0);
        let pd = ProfitableDeviation::from_result(&result, ci);
        assert!(pd.is_some());
        assert!(pd.unwrap().is_significant());
    }

    #[test]
    fn test_profitable_deviation_not_profitable() {
        let s = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        let result = DeviationResult::new(PlayerId(0), s, 15.0, 10.0);
        let ci = ConfidenceInterval::new(-7.0, -3.0, 0.95, -5.0);
        let pd = ProfitableDeviation::from_result(&result, ci);
        assert!(pd.is_none());
    }

    #[test]
    fn test_deviation_incentive_computation() {
        let trajectory = make_trajectory(10, 2, Price(5.0));
        let game_config = make_game_config(2);
        let incentive = deviation_incentive(PlayerId(0), &trajectory, &game_config, (Price(0.0), Price(10.0)), 50);
        assert!(incentive >= 0.0);
    }

    #[test]
    fn test_all_deviation_incentives() {
        let trajectory = make_trajectory(5, 2, Price(5.0));
        let game_config = make_game_config(2);
        let incentives = all_deviation_incentives(&trajectory, &game_config, (Price(0.0), Price(10.0)), 50);
        assert_eq!(incentives.len(), 2);
    }

    #[test]
    fn test_lipschitz_estimate() {
        let trajectory = make_trajectory(10, 2, Price(5.0));
        let game_config = make_game_config(2);
        let lip = estimate_lipschitz_constant(PlayerId(0), &trajectory, &game_config, (Price(0.0), Price(10.0)), 50);
        assert!(lip.is_finite());
        assert!(lip >= 0.0);
    }

    #[test]
    fn test_multi_period_deviation() {
        let trajectory = make_trajectory(20, 2, Price(5.0));
        let game_config = make_game_config(2);
        let result = MultiPeriodDeviation::find_optimal(
            PlayerId(0), &trajectory, RoundNumber(5), 5, &game_config, (Price(0.0), Price(10.0)), 20,
        );
        assert!(result.is_ok());
        let mpd = result.unwrap();
        assert_eq!(mpd.duration, 5);
        assert_eq!(mpd.deviation_prices.len(), 5);
    }

    #[test]
    fn test_multi_period_dp() {
        let trajectory = make_trajectory(20, 2, Price(5.0));
        let game_config = make_game_config(2);
        let result = MultiPeriodDeviation::find_dp(
            PlayerId(0), &trajectory, RoundNumber(0), 5, &game_config, (Price(0.0), Price(10.0)), 20,
        );
        assert!(result.is_ok());
        let mpd = result.unwrap();
        assert_eq!(mpd.duration, 5);
    }

    #[test]
    fn test_coarse_to_fine() {
        let config = DeviationEnumeratorConfig {
            coarse_grid_size: 10,
            fine_grid_size: 20,
            top_k_refine: 3,
            ..Default::default()
        };
        let game_config = make_game_config(2);
        let enumerator = DeviationEnumerator::new(config, game_config);
        let trajectory = make_trajectory(5, 2, Price(5.0));
        let evaluator = |_dev: &DeviationStrategy| -> f64 { 1.0 };
        let results = enumerator.coarse_to_fine(&trajectory, RoundNumber(0), &evaluator);
        assert!(!results.is_empty());
    }
}
