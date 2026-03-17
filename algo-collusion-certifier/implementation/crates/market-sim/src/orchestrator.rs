//! Game orchestration for market simulations.
//!
//! This module provides high-level orchestration of multi-player market
//! simulation games and tournaments:
//!
//! - [`GameOrchestrator`]: manages the full lifecycle of a single game
//! - [`TournamentOrchestrator`]: runs multiple games across different scenarios
//! - [`ActionValidator`]: validates player actions against market constraints
//! - [`EventCallback`]: trait for hooking into game lifecycle events

use crate::market::{Market, MarketFactory};
use crate::types::*;
use crate::{MarketSimError, MarketSimResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ════════════════════════════════════════════════════════════════════════════
// OrchestratorConfig
// ════════════════════════════════════════════════════════════════════════════

/// Configuration for the game orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Underlying market/game configuration.
    pub game_config: GameConfig,
    /// Number of players in the game.
    pub num_players: usize,
    /// Maximum number of rounds to play.
    pub max_rounds: usize,
    /// Timeout for each player action in milliseconds (0 = no timeout).
    pub action_timeout_ms: u64,
    /// Whether to validate actions before applying them.
    pub validation_enabled: bool,
    /// Whether all players act simultaneously (true) or sequentially (false).
    pub synchronous_timing: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            game_config: GameConfig::default(),
            num_players: 2,
            max_rounds: 1000,
            action_timeout_ms: 0,
            validation_enabled: true,
            synchronous_timing: true,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// RoundResult / GameResult
// ════════════════════════════════════════════════════════════════════════════

/// Result of a single round of play.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResult {
    /// Zero-based round number.
    pub round_number: usize,
    /// Actions submitted by each player.
    pub actions: Vec<PlayerAction>,
    /// Market outcome computed from the actions.
    pub outcome: MarketOutcome,
    /// Wall-clock time to execute this round, in milliseconds.
    pub timing_ms: f64,
}

/// Aggregate result of a complete game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameResult {
    /// Per-round results in chronological order.
    pub rounds: Vec<RoundResult>,
    /// Total number of rounds played.
    pub total_rounds: usize,
    /// Total wall-clock duration of the game in milliseconds.
    pub duration_ms: f64,
    /// Total profit accumulated by each player across all rounds.
    pub per_player_total_profit: Vec<f64>,
}

impl GameResult {
    /// Mean profit per round for each player.
    pub fn mean_profits(&self) -> Vec<f64> {
        if self.total_rounds == 0 {
            return self.per_player_total_profit.clone();
        }
        let t = self.total_rounds as f64;
        self.per_player_total_profit.iter().map(|p| p / t).collect()
    }

    /// Mean price per round for each player.
    pub fn mean_prices(&self) -> Vec<f64> {
        if self.rounds.is_empty() {
            return vec![];
        }
        let n = self.rounds[0].outcome.prices.len();
        let t = self.rounds.len() as f64;
        let mut sums = vec![0.0; n];
        for r in &self.rounds {
            for (i, p) in r.outcome.prices.iter().enumerate() {
                sums[i] += p;
            }
        }
        sums.iter().map(|s| s / t).collect()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PlayerInfo
// ════════════════════════════════════════════════════════════════════════════

/// Metadata for a registered player.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerInfo {
    pub id: PlayerId,
    pub name: String,
    pub algorithm_type: String,
}

// ════════════════════════════════════════════════════════════════════════════
// ActionValidator
// ════════════════════════════════════════════════════════════════════════════

/// Validates player actions against market constraints.
#[derive(Debug, Clone)]
pub struct ActionValidator {
    pub min_price: f64,
    pub max_price: f64,
    pub max_quantity: f64,
}

impl ActionValidator {
    pub fn new(min_price: f64, max_price: f64, max_quantity: f64) -> Self {
        Self { min_price, max_price, max_quantity }
    }

    /// Build a validator from a [`GameConfig`].
    pub fn from_config(config: &GameConfig) -> Self {
        Self {
            min_price: config.price_min,
            max_price: config.price_max,
            max_quantity: config.quantity_max,
        }
    }

    /// Validate a price action.
    pub fn validate_price(&self, price: f64) -> MarketSimResult<()> {
        if !price.is_finite() {
            return Err(MarketSimError::InvalidParameter(
                format!("Price must be finite, got {price}"),
            ));
        }
        if price < self.min_price {
            return Err(MarketSimError::InvalidParameter(
                format!("Price {price} below minimum {}", self.min_price),
            ));
        }
        if price > self.max_price {
            return Err(MarketSimError::InvalidParameter(
                format!("Price {price} above maximum {}", self.max_price),
            ));
        }
        Ok(())
    }

    /// Validate a quantity action.
    pub fn validate_quantity(&self, quantity: f64) -> MarketSimResult<()> {
        if !quantity.is_finite() {
            return Err(MarketSimError::InvalidParameter(
                format!("Quantity must be finite, got {quantity}"),
            ));
        }
        if quantity < 0.0 {
            return Err(MarketSimError::InvalidParameter(
                format!("Quantity {quantity} must be non-negative"),
            ));
        }
        if quantity > self.max_quantity {
            return Err(MarketSimError::InvalidParameter(
                format!("Quantity {quantity} above maximum {}", self.max_quantity),
            ));
        }
        Ok(())
    }

    /// Validate a full set of player actions according to the market type.
    pub fn validate_actions(
        &self,
        actions: &[PlayerAction],
        market_type: MarketType,
    ) -> MarketSimResult<()> {
        for action in actions {
            match market_type {
                MarketType::Bertrand => self.validate_price(action.value)?,
                MarketType::Cournot => self.validate_quantity(action.value)?,
            }
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// EventCallback trait
// ════════════════════════════════════════════════════════════════════════════

/// Trait for receiving game lifecycle events.
pub trait EventCallback: Send {
    /// Called at the start of each round.
    fn on_round_start(&mut self, round: usize);
    /// Called at the end of each round with its result.
    fn on_round_end(&mut self, round: usize, result: &RoundResult);
    /// Called when the game completes.
    fn on_game_end(&mut self, result: &GameResult);
}

/// Default callback that does nothing.
pub struct DefaultCallback;

impl EventCallback for DefaultCallback {
    fn on_round_start(&mut self, _round: usize) {}
    fn on_round_end(&mut self, _round: usize, _result: &RoundResult) {}
    fn on_game_end(&mut self, _result: &GameResult) {}
}

/// Callback that logs events via the `log` crate.
pub struct LoggingCallback {
    /// Only log every Nth round to avoid flooding.
    pub log_interval: usize,
}

impl LoggingCallback {
    pub fn new(log_interval: usize) -> Self {
        Self { log_interval: log_interval.max(1) }
    }
}

impl Default for LoggingCallback {
    fn default() -> Self {
        Self::new(100)
    }
}

impl EventCallback for LoggingCallback {
    fn on_round_start(&mut self, round: usize) {
        if round % self.log_interval == 0 {
            log::debug!("Round {round} starting");
        }
    }

    fn on_round_end(&mut self, round: usize, result: &RoundResult) {
        if round % self.log_interval == 0 {
            log::info!(
                "Round {round} complete in {:.2}ms – profits: {:?}",
                result.timing_ms,
                result.outcome.profits,
            );
        }
    }

    fn on_game_end(&mut self, result: &GameResult) {
        log::info!(
            "Game complete: {} rounds in {:.1}ms, total profits: {:?}",
            result.total_rounds,
            result.duration_ms,
            result.per_player_total_profit,
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ActionSource trait
// ════════════════════════════════════════════════════════════════════════════

/// Trait for anything that can decide a player's action for a round.
///
/// This is the per-player analogue of `simulation::ActionProvider` and allows
/// the orchestrator to query each player individually.
pub trait ActionSource: Send {
    /// Choose an action value (price or quantity) for the given round.
    fn choose_action(
        &mut self,
        player_id: PlayerId,
        round: usize,
        history: &[RoundResult],
    ) -> f64;
}

/// Always plays the same fixed value.
pub struct FixedActionSource {
    pub value: f64,
}

impl FixedActionSource {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl ActionSource for FixedActionSource {
    fn choose_action(&mut self, _player_id: PlayerId, _round: usize, _history: &[RoundResult]) -> f64 {
        self.value
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GameOrchestrator
// ════════════════════════════════════════════════════════════════════════════

/// Orchestrates the full lifecycle of a market simulation game.
///
/// # Example
/// ```ignore
/// let config = OrchestratorConfig::default();
/// let mut orch = GameOrchestrator::new(config)?;
/// let p0 = orch.register_player("Algo-A", "q-learning");
/// let p1 = orch.register_player("Algo-B", "fixed-price");
/// orch.set_action_source(Box::new(FixedActionSource::new(5.0)));
/// let result = orch.run_game(100)?;
/// ```
pub struct GameOrchestrator {
    config: OrchestratorConfig,
    market: Box<dyn Market>,
    players: Vec<PlayerInfo>,
    history: Vec<RoundResult>,
    validator: Option<ActionValidator>,
    callback: Box<dyn EventCallback>,
    action_source: Box<dyn ActionSource>,
    current_round: usize,
}

impl GameOrchestrator {
    /// Create a new orchestrator from the given configuration.
    pub fn new(config: OrchestratorConfig) -> MarketSimResult<Self> {
        let market = MarketFactory::create(&config.game_config)?;
        let validator = if config.validation_enabled {
            Some(ActionValidator::from_config(&config.game_config))
        } else {
            None
        };

        // Default action source uses the midpoint of the price range.
        let midpoint = (config.game_config.price_min + config.game_config.price_max) / 2.0;

        Ok(Self {
            config,
            market,
            players: Vec::new(),
            history: Vec::new(),
            validator,
            callback: Box::new(DefaultCallback),
            action_source: Box::new(FixedActionSource::new(midpoint)),
            current_round: 0,
        })
    }

    /// Register a new player and return their id.
    pub fn register_player(&mut self, name: &str, algorithm_type: &str) -> PlayerId {
        let id = self.players.len();
        self.players.push(PlayerInfo {
            id,
            name: name.to_string(),
            algorithm_type: algorithm_type.to_string(),
        });
        id
    }

    /// Replace the action source that decides player moves each round.
    pub fn set_action_source(&mut self, source: Box<dyn ActionSource>) {
        self.action_source = source;
    }

    /// Replace the event callback.
    pub fn set_callback(&mut self, callback: Box<dyn EventCallback>) {
        self.callback = callback;
    }

    /// Number of registered players.
    pub fn num_players(&self) -> usize {
        self.players.len()
    }

    /// Registered player metadata.
    pub fn players(&self) -> &[PlayerInfo] {
        &self.players
    }

    /// Run a complete game for the specified number of rounds.
    pub fn run_game(&mut self, rounds: usize) -> MarketSimResult<GameResult> {
        let effective_rounds = rounds.min(self.config.max_rounds);
        let game_start = Instant::now();

        for _ in 0..effective_rounds {
            self.run_round()?;
        }

        let duration_ms = game_start.elapsed().as_secs_f64() * 1000.0;
        let result = self.build_game_result(duration_ms);
        self.callback.on_game_end(&result);
        Ok(result)
    }

    /// Execute a single round: collect actions, validate, compute outcome, record.
    pub fn run_round(&mut self) -> MarketSimResult<RoundResult> {
        let round = self.current_round;
        let n = self.effective_num_players();

        self.callback.on_round_start(round);
        let round_start = Instant::now();

        // Collect actions from the action source.
        let actions: Vec<PlayerAction> = (0..n)
            .map(|i| {
                let value = self.action_source.choose_action(i, round, &self.history);
                PlayerAction::new(i, value)
            })
            .collect();

        // Validate if enabled.
        if let Some(ref validator) = self.validator {
            validator.validate_actions(&actions, self.market.market_type())?;
        }

        // Compute market outcome.
        let outcome = self.market.simulate_round(&actions, round as u64)?;

        let timing_ms = round_start.elapsed().as_secs_f64() * 1000.0;
        let result = RoundResult {
            round_number: round,
            actions,
            outcome,
            timing_ms,
        };

        self.history.push(result.clone());
        self.current_round += 1;
        self.callback.on_round_end(round, self.history.last().unwrap());

        Ok(result)
    }

    /// Get the full history of round results.
    pub fn get_history(&self) -> &[RoundResult] {
        &self.history
    }

    /// Reset the orchestrator to its initial state (keeps players and config).
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_round = 0;
    }

    /// Current round number.
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Reference to the orchestrator configuration.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Number of players to use (registered count, falling back to config).
    fn effective_num_players(&self) -> usize {
        if self.players.is_empty() {
            self.config.num_players
        } else {
            self.players.len()
        }
    }

    fn build_game_result(&self, duration_ms: f64) -> GameResult {
        let n = self.effective_num_players();
        let mut per_player_total_profit = vec![0.0; n];
        for r in &self.history {
            for (i, &p) in r.outcome.profits.iter().enumerate() {
                if i < n {
                    per_player_total_profit[i] += p;
                }
            }
        }
        GameResult {
            rounds: self.history.clone(),
            total_rounds: self.history.len(),
            duration_ms,
            per_player_total_profit,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tournament types
// ════════════════════════════════════════════════════════════════════════════

/// Specification for a single tournament match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentMatch {
    /// Human-readable label for this match.
    pub label: String,
    /// First algorithm configuration (name + type + fixed action value).
    pub algorithm_a: AlgorithmSpec,
    /// Second algorithm configuration.
    pub algorithm_b: AlgorithmSpec,
    /// Market / game configuration for this match.
    pub game_config: GameConfig,
    /// Number of rounds to play.
    pub num_rounds: usize,
}

/// Specification of an algorithm in a tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmSpec {
    pub name: String,
    pub algorithm_type: String,
    /// Fixed action value used by the default action source.
    pub action_value: f64,
}

/// Result of a single tournament match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub label: String,
    pub game_result: GameResult,
    pub algorithm_a: AlgorithmSpec,
    pub algorithm_b: AlgorithmSpec,
    /// Index of the winning player (highest total profit), or `None` for a tie.
    pub winner: Option<PlayerId>,
}

/// Aggregate result of a full tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentResult {
    /// Per-match results.
    pub matches: Vec<MatchResult>,
    /// Number of matches played.
    pub total_matches: usize,
    /// Total tournament duration in milliseconds.
    pub duration_ms: f64,
    /// Win count per algorithm name.
    pub win_counts: HashMap<String, usize>,
    /// Total profit per algorithm name across all matches.
    pub total_profits: HashMap<String, f64>,
}

impl TournamentResult {
    /// Average profit per algorithm across all matches.
    pub fn mean_profits(&self) -> HashMap<String, f64> {
        if self.total_matches == 0 {
            return self.total_profits.clone();
        }
        self.total_profits
            .iter()
            .map(|(k, v)| (k.clone(), v / self.total_matches as f64))
            .collect()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PairActionSource (for tournaments)
// ════════════════════════════════════════════════════════════════════════════

/// Action source that maps two players to two different fixed values.
struct PairActionSource {
    value_a: f64,
    value_b: f64,
}

impl PairActionSource {
    fn new(value_a: f64, value_b: f64) -> Self {
        Self { value_a, value_b }
    }
}

impl ActionSource for PairActionSource {
    fn choose_action(&mut self, player_id: PlayerId, _round: usize, _history: &[RoundResult]) -> f64 {
        if player_id == 0 { self.value_a } else { self.value_b }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// TournamentOrchestrator
// ════════════════════════════════════════════════════════════════════════════

/// Orchestrator that runs a series of matches across different scenarios.
pub struct TournamentOrchestrator {
    matches: Vec<TournamentMatch>,
}

impl TournamentOrchestrator {
    /// Create a new tournament from a list of match specifications.
    pub fn new(matches: Vec<TournamentMatch>) -> Self {
        Self { matches }
    }

    /// Number of scheduled matches.
    pub fn num_matches(&self) -> usize {
        self.matches.len()
    }

    /// Run every match and aggregate results.
    pub fn run_tournament(&self) -> MarketSimResult<TournamentResult> {
        let tournament_start = Instant::now();
        let mut match_results = Vec::with_capacity(self.matches.len());
        let mut win_counts: HashMap<String, usize> = HashMap::new();
        let mut total_profits: HashMap<String, f64> = HashMap::new();

        for spec in &self.matches {
            let result = self.run_match(spec)?;

            // Accumulate wins.
            if let Some(winner_id) = result.winner {
                let winner_name = if winner_id == 0 {
                    &result.algorithm_a.name
                } else {
                    &result.algorithm_b.name
                };
                *win_counts.entry(winner_name.clone()).or_insert(0) += 1;
            }

            // Accumulate profits.
            if let Some(profit_a) = result.game_result.per_player_total_profit.first() {
                *total_profits.entry(result.algorithm_a.name.clone()).or_insert(0.0) += profit_a;
            }
            if let Some(profit_b) = result.game_result.per_player_total_profit.get(1) {
                *total_profits.entry(result.algorithm_b.name.clone()).or_insert(0.0) += profit_b;
            }

            match_results.push(result);
        }

        let duration_ms = tournament_start.elapsed().as_secs_f64() * 1000.0;

        Ok(TournamentResult {
            total_matches: match_results.len(),
            matches: match_results,
            duration_ms,
            win_counts,
            total_profits,
        })
    }

    /// Run a single match.
    fn run_match(&self, spec: &TournamentMatch) -> MarketSimResult<MatchResult> {
        let orch_config = OrchestratorConfig {
            game_config: spec.game_config.clone(),
            num_players: 2,
            max_rounds: spec.num_rounds,
            action_timeout_ms: 0,
            validation_enabled: true,
            synchronous_timing: true,
        };

        let mut orch = GameOrchestrator::new(orch_config)?;
        orch.register_player(&spec.algorithm_a.name, &spec.algorithm_a.algorithm_type);
        orch.register_player(&spec.algorithm_b.name, &spec.algorithm_b.algorithm_type);
        orch.set_action_source(Box::new(PairActionSource::new(
            spec.algorithm_a.action_value,
            spec.algorithm_b.action_value,
        )));

        let game_result = orch.run_game(spec.num_rounds)?;

        let winner = determine_winner(&game_result.per_player_total_profit);

        Ok(MatchResult {
            label: spec.label.clone(),
            game_result,
            algorithm_a: spec.algorithm_a.clone(),
            algorithm_b: spec.algorithm_b.clone(),
            winner,
        })
    }
}

/// Determine the winner index from total profits (None if within 1e-9 of a tie).
fn determine_winner(profits: &[f64]) -> Option<PlayerId> {
    if profits.len() < 2 {
        return None;
    }
    let diff = (profits[0] - profits[1]).abs();
    if diff < 1e-9 {
        None
    } else if profits[0] > profits[1] {
        Some(0)
    } else {
        Some(1)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn default_game_config() -> GameConfig {
        GameConfig {
            market_type: MarketType::Bertrand,
            demand_system: DemandSystemType::Linear,
            num_players: 2,
            num_rounds: 100,
            demand_intercept: 10.0,
            demand_slope: 1.0,
            demand_cross_slope: 0.5,
            marginal_costs: vec![1.0, 1.0],
            price_min: 0.0,
            price_max: 20.0,
            price_grid_size: 101,
            ..Default::default()
        }
    }

    fn default_orch_config() -> OrchestratorConfig {
        OrchestratorConfig {
            game_config: default_game_config(),
            num_players: 2,
            max_rounds: 200,
            action_timeout_ms: 0,
            validation_enabled: true,
            synchronous_timing: true,
        }
    }

    // ── OrchestratorConfig tests ────────────────────────────────────────

    #[test]
    fn test_orchestrator_config_default() {
        let cfg = OrchestratorConfig::default();
        assert_eq!(cfg.num_players, 2);
        assert!(cfg.validation_enabled);
        assert!(cfg.synchronous_timing);
        assert_eq!(cfg.action_timeout_ms, 0);
    }

    // ── ActionValidator tests ───────────────────────────────────────────

    #[test]
    fn test_validate_price_in_range() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        assert!(v.validate_price(5.0).is_ok());
        assert!(v.validate_price(0.0).is_ok());
        assert!(v.validate_price(20.0).is_ok());
    }

    #[test]
    fn test_validate_price_below_min() {
        let v = ActionValidator::new(1.0, 20.0, 50.0);
        assert!(v.validate_price(0.5).is_err());
    }

    #[test]
    fn test_validate_price_above_max() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        assert!(v.validate_price(25.0).is_err());
    }

    #[test]
    fn test_validate_price_nan() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        assert!(v.validate_price(f64::NAN).is_err());
    }

    #[test]
    fn test_validate_price_infinity() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        assert!(v.validate_price(f64::INFINITY).is_err());
    }

    #[test]
    fn test_validate_quantity_in_range() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        assert!(v.validate_quantity(10.0).is_ok());
        assert!(v.validate_quantity(0.0).is_ok());
        assert!(v.validate_quantity(50.0).is_ok());
    }

    #[test]
    fn test_validate_quantity_negative() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        assert!(v.validate_quantity(-1.0).is_err());
    }

    #[test]
    fn test_validate_quantity_above_max() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        assert!(v.validate_quantity(100.0).is_err());
    }

    #[test]
    fn test_validate_actions_bertrand() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        let actions = vec![
            PlayerAction::new(0, 5.0),
            PlayerAction::new(1, 10.0),
        ];
        assert!(v.validate_actions(&actions, MarketType::Bertrand).is_ok());
    }

    #[test]
    fn test_validate_actions_cournot_invalid() {
        let v = ActionValidator::new(0.0, 20.0, 50.0);
        let actions = vec![
            PlayerAction::new(0, 5.0),
            PlayerAction::new(1, 100.0), // exceeds max_quantity
        ];
        assert!(v.validate_actions(&actions, MarketType::Cournot).is_err());
    }

    #[test]
    fn test_validator_from_config() {
        let cfg = default_game_config();
        let v = ActionValidator::from_config(&cfg);
        assert!((v.min_price - cfg.price_min).abs() < 1e-15);
        assert!((v.max_price - cfg.price_max).abs() < 1e-15);
        assert!((v.max_quantity - cfg.quantity_max).abs() < 1e-15);
    }

    // ── GameOrchestrator tests ──────────────────────────────────────────

    #[test]
    fn test_register_players() {
        let config = default_orch_config();
        let mut orch = GameOrchestrator::new(config).unwrap();
        let p0 = orch.register_player("Alice", "q-learning");
        let p1 = orch.register_player("Bob", "fixed-price");
        assert_eq!(p0, 0);
        assert_eq!(p1, 1);
        assert_eq!(orch.num_players(), 2);
        assert_eq!(orch.players()[0].name, "Alice");
        assert_eq!(orch.players()[1].algorithm_type, "fixed-price");
    }

    #[test]
    fn test_run_single_round() {
        let config = default_orch_config();
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.register_player("A", "fixed");
        orch.register_player("B", "fixed");
        orch.set_action_source(Box::new(FixedActionSource::new(5.0)));

        let result = orch.run_round().unwrap();
        assert_eq!(result.round_number, 0);
        assert_eq!(result.actions.len(), 2);
        assert_eq!(orch.current_round(), 1);
    }

    #[test]
    fn test_run_game_multiple_rounds() {
        let config = default_orch_config();
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.register_player("A", "fixed");
        orch.register_player("B", "fixed");
        orch.set_action_source(Box::new(FixedActionSource::new(5.0)));

        let result = orch.run_game(10).unwrap();
        assert_eq!(result.total_rounds, 10);
        assert_eq!(result.rounds.len(), 10);
        assert!(result.duration_ms >= 0.0);
        assert_eq!(result.per_player_total_profit.len(), 2);
    }

    #[test]
    fn test_run_game_capped_by_max_rounds() {
        let mut config = default_orch_config();
        config.max_rounds = 5;
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.set_action_source(Box::new(FixedActionSource::new(5.0)));

        let result = orch.run_game(100).unwrap();
        assert_eq!(result.total_rounds, 5);
    }

    #[test]
    fn test_get_history() {
        let config = default_orch_config();
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.set_action_source(Box::new(FixedActionSource::new(5.0)));

        orch.run_round().unwrap();
        orch.run_round().unwrap();
        assert_eq!(orch.get_history().len(), 2);
        assert_eq!(orch.get_history()[0].round_number, 0);
        assert_eq!(orch.get_history()[1].round_number, 1);
    }

    #[test]
    fn test_reset() {
        let config = default_orch_config();
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.register_player("A", "fixed");
        orch.register_player("B", "fixed");
        orch.set_action_source(Box::new(FixedActionSource::new(5.0)));

        orch.run_game(5).unwrap();
        assert_eq!(orch.current_round(), 5);

        orch.reset();
        assert_eq!(orch.current_round(), 0);
        assert!(orch.get_history().is_empty());
        // Players are preserved.
        assert_eq!(orch.num_players(), 2);
    }

    #[test]
    fn test_validation_rejects_invalid_price() {
        let mut config = default_orch_config();
        config.validation_enabled = true;
        config.game_config.price_max = 10.0;
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.set_action_source(Box::new(FixedActionSource::new(15.0)));

        let result = orch.run_round();
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_disabled_allows_anything() {
        let mut config = default_orch_config();
        config.validation_enabled = false;
        let mut orch = GameOrchestrator::new(config).unwrap();
        // Price above the normal max — should succeed with validation off.
        orch.set_action_source(Box::new(FixedActionSource::new(15.0)));

        let result = orch.run_round();
        assert!(result.is_ok());
    }

    #[test]
    fn test_profits_accumulate_correctly() {
        let config = default_orch_config();
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.set_action_source(Box::new(FixedActionSource::new(5.0)));

        let result = orch.run_game(3).unwrap();
        let expected_total: Vec<f64> = {
            let mut totals = vec![0.0; 2];
            for r in &result.rounds {
                for (i, &p) in r.outcome.profits.iter().enumerate() {
                    totals[i] += p;
                }
            }
            totals
        };
        for i in 0..2 {
            assert!((result.per_player_total_profit[i] - expected_total[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_game_result_mean_profits() {
        let config = default_orch_config();
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.set_action_source(Box::new(FixedActionSource::new(5.0)));

        let result = orch.run_game(4).unwrap();
        let means = result.mean_profits();
        for (i, m) in means.iter().enumerate() {
            let expected = result.per_player_total_profit[i] / 4.0;
            assert!((m - expected).abs() < 1e-9);
        }
    }

    // ── EventCallback tests ─────────────────────────────────────────────

    #[test]
    fn test_default_callback_does_nothing() {
        let mut cb = DefaultCallback;
        cb.on_round_start(0);
        cb.on_round_end(0, &RoundResult {
            round_number: 0,
            actions: vec![],
            outcome: MarketOutcome::new(0, vec![], vec![], vec![]),
            timing_ms: 0.0,
        });
        cb.on_game_end(&GameResult {
            rounds: vec![],
            total_rounds: 0,
            duration_ms: 0.0,
            per_player_total_profit: vec![],
        });
        // No panic is the success criterion.
    }

    #[test]
    fn test_logging_callback_interval() {
        let cb = LoggingCallback::new(10);
        assert_eq!(cb.log_interval, 10);
        let cb_min = LoggingCallback::new(0);
        assert_eq!(cb_min.log_interval, 1); // clamped
    }

    // ── determine_winner ────────────────────────────────────────────────

    #[test]
    fn test_determine_winner_clear_winner() {
        assert_eq!(determine_winner(&[100.0, 50.0]), Some(0));
        assert_eq!(determine_winner(&[50.0, 100.0]), Some(1));
    }

    #[test]
    fn test_determine_winner_tie() {
        assert_eq!(determine_winner(&[100.0, 100.0]), None);
    }

    #[test]
    fn test_determine_winner_single_player() {
        assert_eq!(determine_winner(&[100.0]), None);
    }

    // ── TournamentOrchestrator tests ────────────────────────────────────

    fn sample_match(label: &str, va: f64, vb: f64) -> TournamentMatch {
        TournamentMatch {
            label: label.to_string(),
            algorithm_a: AlgorithmSpec {
                name: "AlgoA".to_string(),
                algorithm_type: "fixed".to_string(),
                action_value: va,
            },
            algorithm_b: AlgorithmSpec {
                name: "AlgoB".to_string(),
                algorithm_type: "fixed".to_string(),
                action_value: vb,
            },
            game_config: default_game_config(),
            num_rounds: 10,
        }
    }

    #[test]
    fn test_tournament_single_match() {
        let t = TournamentOrchestrator::new(vec![sample_match("m1", 5.0, 5.0)]);
        assert_eq!(t.num_matches(), 1);
        let result = t.run_tournament().unwrap();
        assert_eq!(result.total_matches, 1);
        assert_eq!(result.matches[0].label, "m1");
    }

    #[test]
    fn test_tournament_multiple_matches() {
        let t = TournamentOrchestrator::new(vec![
            sample_match("m1", 5.0, 6.0),
            sample_match("m2", 6.0, 5.0),
        ]);
        let result = t.run_tournament().unwrap();
        assert_eq!(result.total_matches, 2);
        assert!(result.duration_ms >= 0.0);
    }

    #[test]
    fn test_tournament_win_counts() {
        // With symmetric costs and linear demand, a lower price typically
        // captures more demand and thus more profit if above marginal cost.
        let t = TournamentOrchestrator::new(vec![
            sample_match("m1", 3.0, 8.0),
        ]);
        let result = t.run_tournament().unwrap();
        // Either there is a winner or a tie — the structure must be well-formed.
        assert_eq!(result.matches.len(), 1);
        let total_wins: usize = result.win_counts.values().sum();
        assert!(total_wins <= 1);
    }

    #[test]
    fn test_tournament_mean_profits() {
        let t = TournamentOrchestrator::new(vec![
            sample_match("m1", 5.0, 5.0),
            sample_match("m2", 5.0, 5.0),
        ]);
        let result = t.run_tournament().unwrap();
        let means = result.mean_profits();
        // Both algorithms should have an entry.
        assert!(means.contains_key("AlgoA"));
        assert!(means.contains_key("AlgoB"));
    }

    #[test]
    fn test_tournament_empty() {
        let t = TournamentOrchestrator::new(vec![]);
        let result = t.run_tournament().unwrap();
        assert_eq!(result.total_matches, 0);
        assert!(result.matches.is_empty());
    }

    // ── PairActionSource test ───────────────────────────────────────────

    #[test]
    fn test_pair_action_source() {
        let mut src = PairActionSource::new(3.0, 7.0);
        assert!((src.choose_action(0, 0, &[]) - 3.0).abs() < 1e-15);
        assert!((src.choose_action(1, 0, &[]) - 7.0).abs() < 1e-15);
        assert!((src.choose_action(2, 0, &[]) - 7.0).abs() < 1e-15); // falls to branch b
    }

    // ── Cournot orchestration test ──────────────────────────────────────

    #[test]
    fn test_cournot_orchestration() {
        let game_config = GameConfig {
            market_type: MarketType::Cournot,
            demand_system: DemandSystemType::Linear,
            num_players: 2,
            num_rounds: 10,
            demand_intercept: 10.0,
            demand_slope: 1.0,
            marginal_costs: vec![1.0, 1.0],
            quantity_min: 0.0,
            quantity_max: 10.0,
            quantity_grid_size: 101,
            ..Default::default()
        };
        let config = OrchestratorConfig {
            game_config,
            num_players: 2,
            max_rounds: 10,
            validation_enabled: true,
            ..Default::default()
        };
        let mut orch = GameOrchestrator::new(config).unwrap();
        orch.set_action_source(Box::new(FixedActionSource::new(3.0)));
        let result = orch.run_game(5).unwrap();
        assert_eq!(result.total_rounds, 5);
        for r in &result.rounds {
            assert_eq!(r.outcome.quantities.len(), 2);
        }
    }
}
