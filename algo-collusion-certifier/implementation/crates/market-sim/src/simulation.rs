//! High-performance simulation engine with batch and parallel support.
//!
//! The simulation engine orchestrates market rounds at high throughput:
//! - [`SimulationEngine`]: main engine running a sequence of rounds
//! - [`HotLoop`]: tight inner loop optimised for >100K rounds/sec
//! - [`SimulationResult`]: full trajectory and summary statistics
//! - [`SimulationCheckpoint`]: save/restore mid-simulation state
//! - [`HistoryBuffer`]: memory-efficient circular buffer for trajectories
//! - Batch / Monte Carlo and parallel (rayon) simulation support

use crate::market::{Market, MarketFactory};
use crate::types::*;
use crate::MarketSimResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ════════════════════════════════════════════════════════════════════════════
// SimulationConfig (engine-level)
// ════════════════════════════════════════════════════════════════════════════

/// Configuration for the simulation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationEngineConfig {
    /// Total rounds to simulate.
    pub num_rounds: u64,
    /// Maximum history length to keep in memory (0 = unlimited).
    pub max_history: usize,
    /// Enable performance timing.
    pub enable_timing: bool,
    /// Seed for RNG (None = random).
    pub seed: Option<u64>,
    /// Warmup rounds (excluded from statistics).
    pub warmup_rounds: u64,
    /// Checkpoint interval (0 = no checkpoints).
    pub checkpoint_interval: u64,
    /// Truncated horizon (0 = full).
    pub truncated_horizon: u64,
}

impl Default for SimulationEngineConfig {
    fn default() -> Self {
        Self {
            num_rounds: 1000,
            max_history: 0,
            enable_timing: true,
            seed: Some(42),
            warmup_rounds: 0,
            checkpoint_interval: 0,
            truncated_horizon: 0,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HistoryBuffer
// ════════════════════════════════════════════════════════════════════════════

/// Memory-efficient circular buffer for trajectory storage.
///
/// When `max_len > 0`, only the last `max_len` outcomes are kept.
/// When `max_len == 0`, all outcomes are stored (unbounded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryBuffer {
    buffer: Vec<MarketOutcome>,
    max_len: usize,
    head: usize,
    total_pushed: u64,
    is_full: bool,
}

impl HistoryBuffer {
    pub fn new(max_len: usize) -> Self {
        Self {
            buffer: if max_len > 0 {
                Vec::with_capacity(max_len)
            } else {
                Vec::new()
            },
            max_len,
            head: 0,
            total_pushed: 0,
            is_full: false,
        }
    }

    pub fn unbounded() -> Self {
        Self::new(0)
    }

    /// Push an outcome into the buffer.
    pub fn push(&mut self, outcome: MarketOutcome) {
        if self.max_len == 0 {
            // Unbounded mode
            self.buffer.push(outcome);
        } else if !self.is_full {
            // Still filling up
            self.buffer.push(outcome);
            if self.buffer.len() == self.max_len {
                self.is_full = true;
                self.head = 0;
            }
        } else {
            // Circular overwrite
            self.buffer[self.head] = outcome;
            self.head = (self.head + 1) % self.max_len;
        }
        self.total_pushed += 1;
    }

    /// Number of outcomes currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Total number of outcomes ever pushed.
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    /// Get outcomes in chronological order.
    pub fn to_vec(&self) -> Vec<MarketOutcome> {
        if self.max_len == 0 || !self.is_full {
            return self.buffer.clone();
        }
        // Circular buffer: reconstruct chronological order
        let mut result = Vec::with_capacity(self.max_len);
        for i in 0..self.max_len {
            let idx = (self.head + i) % self.max_len;
            result.push(self.buffer[idx].clone());
        }
        result
    }

    /// Get the most recent outcome.
    pub fn last(&self) -> Option<&MarketOutcome> {
        if self.buffer.is_empty() {
            return None;
        }
        if self.max_len == 0 || !self.is_full {
            self.buffer.last()
        } else {
            let idx = if self.head == 0 { self.max_len - 1 } else { self.head - 1 };
            Some(&self.buffer[idx])
        }
    }

    /// Get the last N outcomes in chronological order.
    pub fn last_n(&self, n: usize) -> Vec<MarketOutcome> {
        let all = self.to_vec();
        if n >= all.len() {
            all
        } else {
            all[all.len() - n..].to_vec()
        }
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.head = 0;
        self.total_pushed = 0;
        self.is_full = false;
    }

    /// Convert to a PriceTrajectory.
    pub fn to_trajectory(&self, num_players: usize) -> PriceTrajectory {
        PriceTrajectory::with_outcomes(num_players, self.to_vec())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SimulationCheckpoint
// ════════════════════════════════════════════════════════════════════════════

/// Snapshot of simulation state for save/restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationCheckpoint {
    pub round: RoundNumber,
    pub recent_outcomes: Vec<MarketOutcome>,
    pub rng_seed_at_checkpoint: u64,
    pub config: SimulationEngineConfig,
    pub cumulative_stats: CumulativeStats,
}

/// Running statistics accumulated during simulation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CumulativeStats {
    pub total_rounds: u64,
    pub price_sums: Vec<f64>,
    pub price_sq_sums: Vec<f64>,
    pub profit_sums: Vec<f64>,
    pub profit_sq_sums: Vec<f64>,
    pub quantity_sums: Vec<f64>,
}

impl CumulativeStats {
    pub fn new(num_players: usize) -> Self {
        Self {
            total_rounds: 0,
            price_sums: vec![0.0; num_players],
            price_sq_sums: vec![0.0; num_players],
            profit_sums: vec![0.0; num_players],
            profit_sq_sums: vec![0.0; num_players],
            quantity_sums: vec![0.0; num_players],
        }
    }

    pub fn update(&mut self, outcome: &MarketOutcome) {
        self.total_rounds += 1;
        let n = self.price_sums.len();
        for i in 0..n {
            self.price_sums[i] += outcome.prices[i];
            self.price_sq_sums[i] += outcome.prices[i] * outcome.prices[i];
            self.profit_sums[i] += outcome.profits[i];
            self.profit_sq_sums[i] += outcome.profits[i] * outcome.profits[i];
            self.quantity_sums[i] += outcome.quantities[i];
        }
    }

    pub fn mean_prices(&self) -> Vec<f64> {
        let t = self.total_rounds as f64;
        if t < 1.0 { return self.price_sums.clone(); }
        self.price_sums.iter().map(|s| s / t).collect()
    }

    pub fn mean_profits(&self) -> Vec<f64> {
        let t = self.total_rounds as f64;
        if t < 1.0 { return self.profit_sums.clone(); }
        self.profit_sums.iter().map(|s| s / t).collect()
    }

    pub fn var_prices(&self) -> Vec<f64> {
        let t = self.total_rounds as f64;
        if t < 2.0 { return vec![0.0; self.price_sums.len()]; }
        self.price_sums
            .iter()
            .zip(self.price_sq_sums.iter())
            .map(|(s, sq)| (sq / t) - (s / t).powi(2))
            .collect()
    }

    pub fn var_profits(&self) -> Vec<f64> {
        let t = self.total_rounds as f64;
        if t < 2.0 { return vec![0.0; self.profit_sums.len()]; }
        self.profit_sums
            .iter()
            .zip(self.profit_sq_sums.iter())
            .map(|(s, sq)| (sq / t) - (s / t).powi(2))
            .collect()
    }

    pub fn mean_quantities(&self) -> Vec<f64> {
        let t = self.total_rounds as f64;
        if t < 1.0 { return self.quantity_sums.clone(); }
        self.quantity_sums.iter().map(|s| s / t).collect()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SimulationResult
// ════════════════════════════════════════════════════════════════════════════

/// Complete result of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub trajectory: PriceTrajectory,
    pub stats: CumulativeStats,
    pub num_players: usize,
    pub total_rounds: u64,
    pub warmup_rounds: u64,
    pub elapsed: Duration,
    pub rounds_per_second: f64,
    pub seed: Option<u64>,
    pub checkpoints: Vec<SimulationCheckpoint>,
}

impl SimulationResult {
    /// Mean prices (post-warmup).
    pub fn mean_prices(&self) -> Vec<f64> {
        self.stats.mean_prices()
    }

    /// Mean profits (post-warmup).
    pub fn mean_profits(&self) -> Vec<f64> {
        self.stats.mean_profits()
    }

    /// Price variance (post-warmup).
    pub fn price_variance(&self) -> Vec<f64> {
        self.stats.var_prices()
    }

    /// Final prices (last round).
    pub fn final_prices(&self) -> Option<Vec<f64>> {
        self.trajectory.last_outcome().map(|o| o.prices.clone())
    }

    /// Final profits (last round).
    pub fn final_profits(&self) -> Option<Vec<f64>> {
        self.trajectory.last_outcome().map(|o| o.profits.clone())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ActionProvider trait
// ════════════════════════════════════════════════════════════════════════════

/// Trait for anything that can provide actions for a simulation round.
pub trait ActionProvider: Send {
    fn provide_actions(
        &mut self,
        round: RoundNumber,
        history: &[MarketOutcome],
        num_players: usize,
    ) -> Vec<PlayerAction>;
}

/// Fixed-action provider: always plays the same actions.
pub struct FixedActions {
    values: Vec<f64>,
}

impl FixedActions {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }
}

impl ActionProvider for FixedActions {
    fn provide_actions(
        &mut self,
        _round: RoundNumber,
        _history: &[MarketOutcome],
        num_players: usize,
    ) -> Vec<PlayerAction> {
        (0..num_players)
            .map(|i| PlayerAction::new(i, self.values[i]))
            .collect()
    }
}

/// Random-action provider: plays uniformly random actions in [lo, hi].
pub struct RandomActions {
    lo: f64,
    hi: f64,
    rng: StdRng,
}

impl RandomActions {
    pub fn new(lo: f64, hi: f64, seed: u64) -> Self {
        Self { lo, hi, rng: StdRng::seed_from_u64(seed) }
    }
}

impl ActionProvider for RandomActions {
    fn provide_actions(
        &mut self,
        _round: RoundNumber,
        _history: &[MarketOutcome],
        num_players: usize,
    ) -> Vec<PlayerAction> {
        (0..num_players)
            .map(|i| {
                let v = self.rng.gen_range(self.lo..=self.hi);
                PlayerAction::new(i, v)
            })
            .collect()
    }
}

/// Best-response provider: each player plays a best response to the last round.
pub struct BestResponseActions {
    market: Box<dyn Market>,
}

impl BestResponseActions {
    pub fn new(market: Box<dyn Market>) -> Self {
        Self { market }
    }
}

impl ActionProvider for BestResponseActions {
    fn provide_actions(
        &mut self,
        _round: RoundNumber,
        history: &[MarketOutcome],
        num_players: usize,
    ) -> Vec<PlayerAction> {
        let n = num_players;
        if history.is_empty() {
            // Start with midpoint actions
            return (0..n).map(|i| PlayerAction::new(i, 5.0)).collect();
        }
        let last = history.last().unwrap();
        let last_values: Vec<f64> = match self.market.market_type() {
            MarketType::Bertrand => last.prices.clone(),
            MarketType::Cournot => last.quantities.clone(),
        };

        let mut actions = Vec::with_capacity(n);
        for i in 0..n {
            let others: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| last_values[j]).collect();
            let br = self.market.best_response(i, &others).unwrap_or(last_values[i]);
            actions.push(PlayerAction::new(i, br));
        }
        actions
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SimulationEngine
// ════════════════════════════════════════════════════════════════════════════

/// Main simulation engine orchestrating market rounds.
pub struct SimulationEngine {
    pub config: SimulationEngineConfig,
    pub game_config: GameConfig,
}

impl SimulationEngine {
    pub fn new(config: SimulationEngineConfig, game_config: GameConfig) -> Self {
        Self { config, game_config }
    }

    /// Run a full simulation with a given action provider.
    pub fn run(
        &self,
        market: &dyn Market,
        action_provider: &mut dyn ActionProvider,
    ) -> MarketSimResult<SimulationResult> {
        let n = market.num_players();
        let num_rounds = if self.config.truncated_horizon > 0 {
            self.config.truncated_horizon
        } else {
            self.config.num_rounds
        };
        let mut buffer = HistoryBuffer::new(self.config.max_history);
        let mut stats = CumulativeStats::new(n);
        let mut checkpoints = Vec::new();
        let start = Instant::now();

        for round in 0..num_rounds {
            let history_slice = buffer.to_vec();
            let actions = action_provider.provide_actions(round, &history_slice, n);
            let outcome = market.simulate_round(&actions, round)?;

            if round >= self.config.warmup_rounds {
                stats.update(&outcome);
            }

            buffer.push(outcome);

            // Checkpoint if needed
            if self.config.checkpoint_interval > 0
                && round > 0
                && round % self.config.checkpoint_interval == 0
            {
                checkpoints.push(SimulationCheckpoint {
                    round,
                    recent_outcomes: buffer.last_n(10),
                    rng_seed_at_checkpoint: self.config.seed.unwrap_or(0).wrapping_add(round),
                    config: self.config.clone(),
                    cumulative_stats: stats.clone(),
                });
            }
        }

        let elapsed = start.elapsed();
        let rps = if elapsed.as_secs_f64() > 0.0 {
            num_rounds as f64 / elapsed.as_secs_f64()
        } else {
            f64::INFINITY
        };

        Ok(SimulationResult {
            trajectory: buffer.to_trajectory(n),
            stats,
            num_players: n,
            total_rounds: num_rounds,
            warmup_rounds: self.config.warmup_rounds,
            elapsed,
            rounds_per_second: rps,
            seed: self.config.seed,
            checkpoints,
        })
    }

    /// Run the simulation using the HotLoop for maximum throughput.
    pub fn run_hot_loop(
        &self,
        market: &dyn Market,
        action_provider: &mut dyn ActionProvider,
    ) -> MarketSimResult<SimulationResult> {
        let hot = HotLoop::new(self.config.clone(), self.game_config.clone());
        hot.execute(market, action_provider)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HotLoop
// ════════════════════════════════════════════════════════════════════════════

/// Tight inner loop optimised for throughput.
///
/// Minimises allocations by pre-allocating buffers and using a circular
/// history buffer. Targets >100K rounds/sec.
pub struct HotLoop {
    config: SimulationEngineConfig,
    #[allow(dead_code)]
    game_config: GameConfig,
}

impl HotLoop {
    pub fn new(config: SimulationEngineConfig, game_config: GameConfig) -> Self {
        Self { config, game_config }
    }

    /// Execute the hot loop. Actions → outcome → record → repeat.
    pub fn execute(
        &self,
        market: &dyn Market,
        action_provider: &mut dyn ActionProvider,
    ) -> MarketSimResult<SimulationResult> {
        let n = market.num_players();
        let num_rounds = if self.config.truncated_horizon > 0 {
            self.config.truncated_horizon
        } else {
            self.config.num_rounds
        };

        // Pre-allocate with bounded history for memory efficiency
        let _history_cap = if self.config.max_history > 0 {
            self.config.max_history
        } else {
            num_rounds as usize
        };
        let mut buffer = HistoryBuffer::new(if self.config.max_history > 0 {
            self.config.max_history
        } else {
            0
        });
        let mut stats = CumulativeStats::new(n);

        // Pre-allocate a small rolling window for the action provider
        let window_size = 64.min(num_rounds as usize);
        let mut rolling_window: Vec<MarketOutcome> = Vec::with_capacity(window_size);

        let start = Instant::now();

        for round in 0..num_rounds {
            // Provide the rolling window instead of full history
            let actions = action_provider.provide_actions(round, &rolling_window, n);
            let outcome = market.simulate_round(&actions, round)?;

            if round >= self.config.warmup_rounds {
                stats.update(&outcome);
            }

            // Maintain rolling window
            if rolling_window.len() >= window_size {
                rolling_window.remove(0);
            }
            rolling_window.push(outcome.clone());

            buffer.push(outcome);
        }

        let elapsed = start.elapsed();
        let rps = if elapsed.as_secs_f64() > 0.0 {
            num_rounds as f64 / elapsed.as_secs_f64()
        } else {
            f64::INFINITY
        };

        Ok(SimulationResult {
            trajectory: buffer.to_trajectory(n),
            stats,
            num_players: n,
            total_rounds: num_rounds,
            warmup_rounds: self.config.warmup_rounds,
            elapsed,
            rounds_per_second: rps,
            seed: self.config.seed,
            checkpoints: Vec::new(),
        })
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Batch / Monte Carlo simulation
// ════════════════════════════════════════════════════════════════════════════

/// Result of a batch (Monte Carlo) simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub results: Vec<SimulationResult>,
    pub num_simulations: usize,
    pub total_elapsed: Duration,
    pub aggregate_mean_prices: Vec<f64>,
    pub aggregate_mean_profits: Vec<f64>,
    pub aggregate_std_prices: Vec<f64>,
    pub aggregate_std_profits: Vec<f64>,
}

impl BatchResult {
    /// Compute aggregate statistics from individual results.
    pub fn compute_aggregates(results: &[SimulationResult], num_players: usize) -> Self {
        let m = results.len();
        let total_elapsed = results.iter().map(|r| r.elapsed).sum();

        let mut price_means_sum = vec![0.0; num_players];
        let mut price_means_sq_sum = vec![0.0; num_players];
        let mut profit_means_sum = vec![0.0; num_players];
        let mut profit_means_sq_sum = vec![0.0; num_players];

        for r in results {
            let mp = r.mean_prices();
            let mpi = r.mean_profits();
            for i in 0..num_players {
                price_means_sum[i] += mp[i];
                price_means_sq_sum[i] += mp[i] * mp[i];
                profit_means_sum[i] += mpi[i];
                profit_means_sq_sum[i] += mpi[i] * mpi[i];
            }
        }

        let mf = m as f64;
        let agg_mean_p: Vec<f64> = price_means_sum.iter().map(|s| s / mf).collect();
        let agg_mean_pi: Vec<f64> = profit_means_sum.iter().map(|s| s / mf).collect();

        let agg_std_p: Vec<f64> = (0..num_players)
            .map(|i| {
                let var = (price_means_sq_sum[i] / mf) - (agg_mean_p[i] * agg_mean_p[i]);
                var.max(0.0).sqrt()
            })
            .collect();
        let agg_std_pi: Vec<f64> = (0..num_players)
            .map(|i| {
                let var = (profit_means_sq_sum[i] / mf) - (agg_mean_pi[i] * agg_mean_pi[i]);
                var.max(0.0).sqrt()
            })
            .collect();

        BatchResult {
            results: results.to_vec(),
            num_simulations: m,
            total_elapsed,
            aggregate_mean_prices: agg_mean_p,
            aggregate_mean_profits: agg_mean_pi,
            aggregate_std_prices: agg_std_p,
            aggregate_std_profits: agg_std_pi,
        }
    }
}

/// Run multiple simulations with different seeds (sequential).
pub fn run_batch_sequential(
    game_config: &GameConfig,
    engine_config: &SimulationEngineConfig,
    seeds: &[u64],
    make_actions: impl Fn(u64) -> Box<dyn ActionProvider>,
) -> MarketSimResult<BatchResult> {
    let n = game_config.num_players;
    let market = MarketFactory::create(game_config)?;
    let mut results = Vec::with_capacity(seeds.len());

    for &seed in seeds {
        let mut cfg = engine_config.clone();
        cfg.seed = Some(seed);
        let engine = SimulationEngine::new(cfg, game_config.clone());
        let mut provider = make_actions(seed);
        let result = engine.run(market.as_ref(), provider.as_mut())?;
        results.push(result);
    }

    Ok(BatchResult::compute_aggregates(&results, n))
}

/// Run multiple simulations in parallel using rayon.
pub fn run_batch_parallel(
    game_config: &GameConfig,
    engine_config: &SimulationEngineConfig,
    seeds: &[u64],
    make_actions: impl Fn(u64) -> Box<dyn ActionProvider> + Send + Sync,
) -> MarketSimResult<BatchResult> {
    use rayon::prelude::*;

    let n = game_config.num_players;

    let results: Vec<MarketSimResult<SimulationResult>> = seeds
        .par_iter()
        .map(|&seed| {
            let market = MarketFactory::create(game_config)?;
            let mut cfg = engine_config.clone();
            cfg.seed = Some(seed);
            let engine = SimulationEngine::new(cfg, game_config.clone());
            let mut provider = make_actions(seed);
            engine.run(market.as_ref(), provider.as_mut())
        })
        .collect();

    let results: Vec<SimulationResult> = results.into_iter().collect::<MarketSimResult<Vec<_>>>()?;
    Ok(BatchResult::compute_aggregates(&results, n))
}

// ════════════════════════════════════════════════════════════════════════════
// Multi-seed convenience
// ════════════════════════════════════════════════════════════════════════════

/// Generate a vector of seeds from a base seed.
pub fn generate_seeds(base_seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(base_seed);
    (0..count).map(|_| rng.gen()).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// Performance metrics
// ════════════════════════════════════════════════════════════════════════════

/// Performance metrics for a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_rounds: u64,
    pub elapsed: Duration,
    pub rounds_per_second: f64,
    pub memory_estimate_bytes: usize,
    pub checkpoints_created: usize,
}

impl PerformanceMetrics {
    pub fn from_result(result: &SimulationResult) -> Self {
        let mem = result.trajectory.outcomes.len()
            * std::mem::size_of::<MarketOutcome>()
            + result.trajectory.outcomes.iter().map(|o| {
                (o.prices.len() + o.quantities.len() + o.profits.len()) * std::mem::size_of::<f64>()
            }).sum::<usize>();
        Self {
            total_rounds: result.total_rounds,
            elapsed: result.elapsed,
            rounds_per_second: result.rounds_per_second,
            memory_estimate_bytes: mem,
            checkpoints_created: result.checkpoints.len(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Round step function
// ════════════════════════════════════════════════════════════════════════════

/// Execute a single round: collect actions → compute outcome → record → notify.
/// This is the core function used in the inner loop.
pub fn round_step(
    market: &dyn Market,
    action_provider: &mut dyn ActionProvider,
    round: RoundNumber,
    history: &[MarketOutcome],
    num_players: usize,
    callback: Option<&dyn Fn(&MarketOutcome)>,
) -> MarketSimResult<MarketOutcome> {
    let actions = action_provider.provide_actions(round, history, num_players);
    let outcome = market.simulate_round(&actions, round)?;

    if let Some(cb) = callback {
        cb(&outcome);
    }

    Ok(outcome)
}

// ════════════════════════════════════════════════════════════════════════════
// Counterfactual simulation
// ════════════════════════════════════════════════════════════════════════════

/// Run a counterfactual simulation starting from a checkpoint.
pub fn run_counterfactual(
    market: &dyn Market,
    checkpoint: &SimulationCheckpoint,
    action_provider: &mut dyn ActionProvider,
    additional_rounds: u64,
) -> MarketSimResult<SimulationResult> {
    let n = market.num_players();
    let mut buffer = HistoryBuffer::new(0);
    let mut stats = CumulativeStats::new(n);

    // Seed the history with checkpoint data
    let mut history = checkpoint.recent_outcomes.clone();
    let start_round = checkpoint.round;
    let start = Instant::now();

    for offset in 0..additional_rounds {
        let round = start_round + offset + 1;
        let actions = action_provider.provide_actions(round, &history, n);
        let outcome = market.simulate_round(&actions, round)?;

        stats.update(&outcome);
        if history.len() >= 64 {
            history.remove(0);
        }
        history.push(outcome.clone());
        buffer.push(outcome);
    }

    let elapsed = start.elapsed();
    let rps = if elapsed.as_secs_f64() > 0.0 {
        additional_rounds as f64 / elapsed.as_secs_f64()
    } else {
        f64::INFINITY
    };

    Ok(SimulationResult {
        trajectory: buffer.to_trajectory(n),
        stats,
        num_players: n,
        total_rounds: additional_rounds,
        warmup_rounds: 0,
        elapsed,
        rounds_per_second: rps,
        seed: Some(checkpoint.rng_seed_at_checkpoint),
        checkpoints: Vec::new(),
    })
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::MarketFactory;

    fn default_game() -> GameConfig {
        GameConfig::default()
    }

    fn make_engine(rounds: u64) -> (SimulationEngine, Box<dyn Market>) {
        let gc = default_game();
        let market = MarketFactory::create(&gc).unwrap();
        let ec = SimulationEngineConfig {
            num_rounds: rounds,
            ..Default::default()
        };
        (SimulationEngine::new(ec, gc), market)
    }

    #[test]
    fn test_history_buffer_unbounded() {
        let mut buf = HistoryBuffer::unbounded();
        for i in 0..100 {
            buf.push(MarketOutcome::new(i, vec![1.0], vec![1.0], vec![1.0]));
        }
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.total_pushed(), 100);
    }

    #[test]
    fn test_history_buffer_bounded() {
        let mut buf = HistoryBuffer::new(10);
        for i in 0..50 {
            buf.push(MarketOutcome::new(i, vec![i as f64], vec![1.0], vec![1.0]));
        }
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.total_pushed(), 50);
        let outcomes = buf.to_vec();
        // Should contain rounds 40–49
        assert_eq!(outcomes[0].round, 40);
        assert_eq!(outcomes[9].round, 49);
    }

    #[test]
    fn test_history_buffer_last() {
        let mut buf = HistoryBuffer::new(5);
        for i in 0..20 {
            buf.push(MarketOutcome::new(i, vec![i as f64], vec![1.0], vec![1.0]));
        }
        let last = buf.last().unwrap();
        assert_eq!(last.round, 19);
    }

    #[test]
    fn test_history_buffer_last_n() {
        let mut buf = HistoryBuffer::new(10);
        for i in 0..30 {
            buf.push(MarketOutcome::new(i, vec![1.0], vec![1.0], vec![1.0]));
        }
        let last3 = buf.last_n(3);
        assert_eq!(last3.len(), 3);
        assert_eq!(last3[0].round, 27);
    }

    #[test]
    fn test_history_buffer_clear() {
        let mut buf = HistoryBuffer::new(10);
        buf.push(MarketOutcome::new(0, vec![1.0], vec![1.0], vec![1.0]));
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.total_pushed(), 0);
    }

    #[test]
    fn test_cumulative_stats() {
        let mut stats = CumulativeStats::new(2);
        stats.update(&MarketOutcome::new(0, vec![3.0, 4.0], vec![5.0, 6.0], vec![10.0, 15.0]));
        stats.update(&MarketOutcome::new(1, vec![5.0, 6.0], vec![3.0, 4.0], vec![12.0, 18.0]));
        let mp = stats.mean_prices();
        assert!((mp[0] - 4.0).abs() < 1e-10);
        assert!((mp[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_actions_provider() {
        let mut provider = FixedActions::new(vec![3.0, 4.0]);
        let actions = provider.provide_actions(0, &[], 2);
        assert_eq!(actions.len(), 2);
        assert!((actions[0].value - 3.0).abs() < 1e-10);
        assert!((actions[1].value - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_actions_provider() {
        let mut provider = RandomActions::new(1.0, 10.0, 42);
        let actions = provider.provide_actions(0, &[], 2);
        assert_eq!(actions.len(), 2);
        assert!(actions[0].value >= 1.0 && actions[0].value <= 10.0);
    }

    #[test]
    fn test_simulation_engine_basic() {
        let (engine, market) = make_engine(100);
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run(market.as_ref(), &mut provider).unwrap();
        assert_eq!(result.total_rounds, 100);
        assert_eq!(result.trajectory.len(), 100);
        assert!(result.rounds_per_second > 0.0);
    }

    #[test]
    fn test_simulation_engine_warmup() {
        let gc = default_game();
        let market = MarketFactory::create(&gc).unwrap();
        let ec = SimulationEngineConfig {
            num_rounds: 100,
            warmup_rounds: 20,
            ..Default::default()
        };
        let engine = SimulationEngine::new(ec, gc);
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run(market.as_ref(), &mut provider).unwrap();
        assert_eq!(result.stats.total_rounds, 80); // 100 - 20 warmup
    }

    #[test]
    fn test_simulation_engine_checkpoints() {
        let gc = default_game();
        let market = MarketFactory::create(&gc).unwrap();
        let ec = SimulationEngineConfig {
            num_rounds: 100,
            checkpoint_interval: 25,
            ..Default::default()
        };
        let engine = SimulationEngine::new(ec, gc);
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run(market.as_ref(), &mut provider).unwrap();
        assert!(result.checkpoints.len() >= 3); // at rounds 25, 50, 75
    }

    #[test]
    fn test_simulation_engine_bounded_history() {
        let gc = default_game();
        let market = MarketFactory::create(&gc).unwrap();
        let ec = SimulationEngineConfig {
            num_rounds: 1000,
            max_history: 50,
            ..Default::default()
        };
        let engine = SimulationEngine::new(ec, gc);
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run(market.as_ref(), &mut provider).unwrap();
        assert!(result.trajectory.len() <= 50);
    }

    #[test]
    fn test_hot_loop() {
        let gc = default_game();
        let market = MarketFactory::create(&gc).unwrap();
        let ec = SimulationEngineConfig {
            num_rounds: 10000,
            max_history: 100,
            ..Default::default()
        };
        let engine = SimulationEngine::new(ec, gc.clone());
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run_hot_loop(market.as_ref(), &mut provider).unwrap();
        assert_eq!(result.total_rounds, 10000);
        // Hot loop should be fast
        assert!(result.rounds_per_second > 1000.0);
    }

    #[test]
    fn test_truncated_horizon() {
        let gc = default_game();
        let market = MarketFactory::create(&gc).unwrap();
        let ec = SimulationEngineConfig {
            num_rounds: 1000,
            truncated_horizon: 50,
            ..Default::default()
        };
        let engine = SimulationEngine::new(ec, gc);
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run(market.as_ref(), &mut provider).unwrap();
        assert_eq!(result.total_rounds, 50);
    }

    #[test]
    fn test_batch_sequential() {
        let gc = default_game();
        let ec = SimulationEngineConfig {
            num_rounds: 50,
            ..Default::default()
        };
        let seeds = vec![1, 2, 3];
        let result = run_batch_sequential(
            &gc,
            &ec,
            &seeds,
            |_seed| Box::new(FixedActions::new(vec![3.0, 3.0])),
        ).unwrap();
        assert_eq!(result.num_simulations, 3);
        assert_eq!(result.aggregate_mean_prices.len(), 2);
    }

    #[test]
    fn test_batch_parallel() {
        let gc = default_game();
        let ec = SimulationEngineConfig {
            num_rounds: 50,
            ..Default::default()
        };
        let seeds = vec![1, 2, 3, 4];
        let result = run_batch_parallel(
            &gc,
            &ec,
            &seeds,
            |_seed| Box::new(FixedActions::new(vec![3.0, 3.0])),
        ).unwrap();
        assert_eq!(result.num_simulations, 4);
    }

    #[test]
    fn test_generate_seeds() {
        let seeds = generate_seeds(42, 10);
        assert_eq!(seeds.len(), 10);
        // All should be distinct with high probability
        let mut unique = seeds.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn test_performance_metrics() {
        let (engine, market) = make_engine(100);
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run(market.as_ref(), &mut provider).unwrap();
        let metrics = PerformanceMetrics::from_result(&result);
        assert_eq!(metrics.total_rounds, 100);
        assert!(metrics.memory_estimate_bytes > 0);
    }

    #[test]
    fn test_round_step() {
        let market = MarketFactory::create(&default_game()).unwrap();
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let outcome = round_step(
            market.as_ref(),
            &mut provider,
            0,
            &[],
            2,
            None,
        ).unwrap();
        assert_eq!(outcome.round, 0);
        assert_eq!(outcome.prices.len(), 2);
    }

    #[test]
    fn test_round_step_with_callback() {
        let market = MarketFactory::create(&default_game()).unwrap();
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let outcome = round_step(
            market.as_ref(),
            &mut provider,
            0,
            &[],
            2,
            Some(&|_o: &MarketOutcome| { /* callback invoked */ }),
        ).unwrap();
        assert_eq!(outcome.round, 0);
    }

    #[test]
    fn test_counterfactual_simulation() {
        let market = MarketFactory::create(&default_game()).unwrap();
        let checkpoint = SimulationCheckpoint {
            round: 50,
            recent_outcomes: vec![
                MarketOutcome::new(49, vec![3.0, 3.0], vec![5.0, 5.0], vec![10.0, 10.0]),
                MarketOutcome::new(50, vec![3.0, 3.0], vec![5.0, 5.0], vec![10.0, 10.0]),
            ],
            rng_seed_at_checkpoint: 42,
            config: SimulationEngineConfig::default(),
            cumulative_stats: CumulativeStats::new(2),
        };
        let mut provider = FixedActions::new(vec![4.0, 4.0]); // different actions
        let result = run_counterfactual(
            market.as_ref(),
            &checkpoint,
            &mut provider,
            50,
        ).unwrap();
        assert_eq!(result.total_rounds, 50);
    }

    #[test]
    fn test_simulation_result_methods() {
        let (engine, market) = make_engine(50);
        let mut provider = FixedActions::new(vec![3.0, 3.0]);
        let result = engine.run(market.as_ref(), &mut provider).unwrap();
        let mp = result.mean_prices();
        assert_eq!(mp.len(), 2);
        assert!((mp[0] - 3.0).abs() < 1e-10);
        let fp = result.final_prices().unwrap();
        assert!((fp[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulation_constant_actions_deterministic() {
        let (engine, market) = make_engine(100);
        let mut p1 = FixedActions::new(vec![3.0, 3.0]);
        let mut p2 = FixedActions::new(vec![3.0, 3.0]);
        let r1 = engine.run(market.as_ref(), &mut p1).unwrap();
        let r2 = engine.run(market.as_ref(), &mut p2).unwrap();
        // Both runs should produce identical results
        assert!((r1.mean_prices()[0] - r2.mean_prices()[0]).abs() < 1e-10);
    }
}
