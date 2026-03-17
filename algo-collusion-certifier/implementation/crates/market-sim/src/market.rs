//! Unified market interface abstracting over Bertrand and Cournot competition.
//!
//! Provides a [`Market`] trait, a [`MarketFactory`] for constructing markets
//! from configuration, and [`MarketState`] for tracking simulation state.

use crate::bertrand::{self, BertrandMarket};
use crate::cournot::{self, CournotMarket};
use crate::types::*;
use crate::{MarketSimError, MarketSimResult};
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// Market trait
// ════════════════════════════════════════════════════════════════════════════

/// Unified interface for both Bertrand and Cournot markets.
pub trait Market: Send + Sync {
    /// Number of players.
    fn num_players(&self) -> usize;

    /// Type of competition.
    fn market_type(&self) -> MarketType;

    /// Simulate one round given player actions. Returns the resulting outcome.
    fn simulate_round(
        &self,
        actions: &[PlayerAction],
        round: RoundNumber,
    ) -> MarketSimResult<MarketOutcome>;

    /// Best response for player `player` given other players' actions.
    fn best_response(
        &self,
        player: PlayerId,
        other_values: &[f64],
    ) -> MarketSimResult<f64>;

    /// Compute profit for all players given action values.
    fn compute_profits(&self, values: &[f64]) -> MarketSimResult<Vec<f64>>;

    /// Nash equilibrium action values (analytical when available, iterative otherwise).
    fn nash_equilibrium(&self) -> MarketSimResult<Vec<f64>>;

    /// Monopoly / collusive action values.
    fn collusive_values(&self) -> MarketSimResult<Vec<f64>>;

    /// Competitive (marginal-cost pricing / perfect competition) action values.
    fn competitive_values(&self) -> MarketSimResult<Vec<f64>>;
}

// ── Bertrand adapter ────────────────────────────────────────────────────────

/// Wraps a [`BertrandMarket`] to implement the [`Market`] trait.
pub struct BertrandAdapter {
    pub inner: BertrandMarket,
    config: GameConfig,
}

impl BertrandAdapter {
    pub fn new(inner: BertrandMarket, config: GameConfig) -> Self {
        Self { inner, config }
    }
}

impl Market for BertrandAdapter {
    fn num_players(&self) -> usize {
        self.inner.num_players
    }

    fn market_type(&self) -> MarketType {
        MarketType::Bertrand
    }

    fn simulate_round(
        &self,
        actions: &[PlayerAction],
        round: RoundNumber,
    ) -> MarketSimResult<MarketOutcome> {
        self.inner.simulate_round(actions, round)
    }

    fn best_response(
        &self,
        player: PlayerId,
        other_values: &[f64],
    ) -> MarketSimResult<f64> {
        self.inner.best_response_price(player, other_values)
    }

    fn compute_profits(&self, prices: &[f64]) -> MarketSimResult<Vec<f64>> {
        self.inner.compute_profit(prices)
    }

    fn nash_equilibrium(&self) -> MarketSimResult<Vec<f64>> {
        if self.config.num_players == 2
            && self.config.demand_system == DemandSystemType::Linear
        {
            let (p1, p2) = BertrandMarket::compute_nash_equilibrium_linear_2p(
                self.config.demand_intercept,
                self.config.demand_slope,
                self.config.demand_cross_slope,
                self.config.marginal_costs[0],
                self.config.marginal_costs[1],
            )?;
            Ok(vec![p1, p2])
        } else if self.config.demand_system == DemandSystemType::Linear {
            // Symmetric N-player
            let mc = self.config.marginal_costs[0];
            let p = BertrandMarket::compute_nash_equilibrium_symmetric(
                self.config.demand_intercept,
                self.config.demand_slope,
                self.config.demand_cross_slope,
                mc,
                self.config.num_players,
            )?;
            Ok(vec![p; self.config.num_players])
        } else {
            // Iterative for non-linear demand systems
            let initial: Vec<f64> = self.config.marginal_costs.iter().map(|mc| mc * 1.5).collect();
            self.inner.compute_nash_iterative(&initial, 500, 1e-6)
        }
    }

    fn collusive_values(&self) -> MarketSimResult<Vec<f64>> {
        if self.config.demand_system == DemandSystemType::Linear {
            let mc = self.config.marginal_costs[0];
            let p = BertrandMarket::compute_collusive_price_symmetric(
                self.config.demand_intercept,
                self.config.demand_slope,
                self.config.demand_cross_slope,
                mc,
                self.config.num_players,
            )?;
            Ok(vec![p; self.config.num_players])
        } else {
            // For non-linear demand: approximate by searching for joint-profit max
            let grid = &self.inner.price_grid;
            let n = self.config.num_players;
            let mut best_price = grid.min;
            let mut best_total_profit = f64::NEG_INFINITY;

            for k in 0..grid.num_points {
                let p = grid.price_at(k);
                let prices = vec![p; n];
                if let Ok(profits) = self.inner.compute_profit(&prices) {
                    let total: f64 = profits.iter().sum();
                    if total > best_total_profit {
                        best_total_profit = total;
                        best_price = p;
                    }
                }
            }
            Ok(vec![best_price; n])
        }
    }

    fn competitive_values(&self) -> MarketSimResult<Vec<f64>> {
        Ok(self.inner.competitive_prices())
    }
}

// ── Cournot adapter ─────────────────────────────────────────────────────────

/// Wraps a [`CournotMarket`] to implement the [`Market`] trait.
pub struct CournotAdapter {
    pub inner: CournotMarket,
    config: GameConfig,
}

impl CournotAdapter {
    pub fn new(inner: CournotMarket, config: GameConfig) -> Self {
        Self { inner, config }
    }
}

impl Market for CournotAdapter {
    fn num_players(&self) -> usize {
        self.inner.num_players
    }

    fn market_type(&self) -> MarketType {
        MarketType::Cournot
    }

    fn simulate_round(
        &self,
        actions: &[PlayerAction],
        round: RoundNumber,
    ) -> MarketSimResult<MarketOutcome> {
        self.inner.simulate_round(actions, round)
    }

    fn best_response(
        &self,
        player: PlayerId,
        other_values: &[f64],
    ) -> MarketSimResult<f64> {
        self.inner.best_response_quantity(player, other_values)
    }

    fn compute_profits(&self, quantities: &[f64]) -> MarketSimResult<Vec<f64>> {
        self.inner.compute_profit(quantities)
    }

    fn nash_equilibrium(&self) -> MarketSimResult<Vec<f64>> {
        CournotMarket::nash_asymmetric_linear(
            self.config.demand_intercept,
            self.config.demand_slope,
            &self.config.marginal_costs,
        )
    }

    fn collusive_values(&self) -> MarketSimResult<Vec<f64>> {
        let n = self.config.num_players;
        let mc = self.config.marginal_costs[0];
        let q = CournotMarket::collusive_quantity_symmetric(
            self.config.demand_intercept,
            self.config.demand_slope,
            mc,
            n,
        );
        Ok(vec![q; n])
    }

    fn competitive_values(&self) -> MarketSimResult<Vec<f64>> {
        let (_, quantities) = self.inner.competitive_outcome()?;
        Ok(quantities)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MarketFactory
// ════════════════════════════════════════════════════════════════════════════

/// Factory for creating [`Market`] instances from a [`GameConfig`].
pub struct MarketFactory;

impl MarketFactory {
    /// Create a market from configuration.
    pub fn create(config: &GameConfig) -> MarketSimResult<Box<dyn Market>> {
        match config.market_type {
            MarketType::Bertrand => {
                let market = bertrand::bertrand_from_config(config)?;
                Ok(Box::new(BertrandAdapter::new(market, config.clone())))
            }
            MarketType::Cournot => {
                let market = cournot::cournot_from_config(config)?;
                Ok(Box::new(CournotAdapter::new(market, config.clone())))
            }
        }
    }

    /// Create a simple symmetric Bertrand market.
    pub fn simple_bertrand(
        num_players: usize,
        demand_intercept: f64,
        demand_slope: f64,
        cross_slope: f64,
        marginal_cost: f64,
    ) -> MarketSimResult<Box<dyn Market>> {
        let config = GameConfig {
            market_type: MarketType::Bertrand,
            demand_system: DemandSystemType::Linear,
            num_players,
            demand_intercept,
            demand_slope,
            demand_cross_slope: cross_slope,
            marginal_costs: vec![marginal_cost; num_players],
            price_min: 0.0,
            price_max: demand_intercept / demand_slope * 2.0,
            price_grid_size: 1001,
            ..Default::default()
        };
        Self::create(&config)
    }

    /// Create a simple symmetric Cournot market.
    pub fn simple_cournot(
        num_players: usize,
        demand_intercept: f64,
        demand_slope: f64,
        marginal_cost: f64,
    ) -> MarketSimResult<Box<dyn Market>> {
        let config = GameConfig {
            market_type: MarketType::Cournot,
            demand_system: DemandSystemType::Linear,
            num_players,
            demand_intercept,
            demand_slope,
            marginal_costs: vec![marginal_cost; num_players],
            quantity_min: 0.0,
            quantity_max: demand_intercept / demand_slope,
            quantity_grid_size: 1001,
            ..Default::default()
        };
        Self::create(&config)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MarketState
// ════════════════════════════════════════════════════════════════════════════

/// Mutable state tracking the current simulation round and history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub current_round: RoundNumber,
    pub trajectory: PriceTrajectory,
    pub config: GameConfig,
    pub is_terminated: bool,
}

impl MarketState {
    pub fn new(config: GameConfig) -> Self {
        let n = config.num_players;
        Self {
            current_round: 0,
            trajectory: PriceTrajectory::new(n),
            config,
            is_terminated: false,
        }
    }

    /// Record an outcome and advance the round counter.
    pub fn record(&mut self, outcome: MarketOutcome) {
        self.trajectory.push(outcome);
        self.current_round += 1;
    }

    /// Check if the simulation should terminate (reached num_rounds).
    pub fn should_terminate(&self) -> bool {
        self.is_terminated || self.current_round >= self.config.num_rounds
    }

    /// Reset state to round 0.
    pub fn reset(&mut self) {
        self.current_round = 0;
        self.trajectory = PriceTrajectory::new(self.config.num_players);
        self.is_terminated = false;
    }

    /// Number of completed rounds.
    pub fn rounds_completed(&self) -> u64 {
        self.current_round
    }

    /// Get the last N outcomes.
    pub fn recent_outcomes(&self, n: usize) -> &[MarketOutcome] {
        let len = self.trajectory.outcomes.len();
        if n >= len {
            &self.trajectory.outcomes
        } else {
            &self.trajectory.outcomes[len - n..]
        }
    }

    /// Mean price over the last N rounds for a given player.
    pub fn recent_mean_price(&self, player: PlayerId, window: usize) -> Option<f64> {
        let recent = self.recent_outcomes(window);
        if recent.is_empty() {
            return None;
        }
        let sum: f64 = recent.iter().map(|o| o.prices[player]).sum();
        Some(sum / recent.len() as f64)
    }

    /// Mean profit over the last N rounds for a given player.
    pub fn recent_mean_profit(&self, player: PlayerId, window: usize) -> Option<f64> {
        let recent = self.recent_outcomes(window);
        if recent.is_empty() {
            return None;
        }
        let sum: f64 = recent.iter().map(|o| o.profits[player]).sum();
        Some(sum / recent.len() as f64)
    }

    /// Total trajectory length.
    pub fn trajectory_len(&self) -> usize {
        self.trajectory.len()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Market clearing utilities
// ════════════════════════════════════════════════════════════════════════════

/// Compute market-clearing price for Cournot: given quantities, return the price
/// using linear inverse demand from config.
pub fn clear_market_cournot(config: &GameConfig, quantities: &[f64]) -> f64 {
    let total: f64 = quantities.iter().sum();
    (config.demand_intercept - config.demand_slope * total).max(0.0)
}

/// Compute the collusiveness index: how close actual profits are to the collusion frontier.
/// CI = (π_actual - π_nash) / (π_collusive - π_nash)
/// CI = 0 means Nash, CI = 1 means full collusion, CI > 1 means above monopoly.
pub fn collusiveness_index(
    actual_profit: f64,
    nash_profit: f64,
    collusive_profit: f64,
) -> f64 {
    let denom = collusive_profit - nash_profit;
    if denom.abs() < 1e-15 {
        if (actual_profit - nash_profit).abs() < 1e-15 { 0.0 } else { f64::INFINITY }
    } else {
        (actual_profit - nash_profit) / denom
    }
}

/// Compute the collusiveness index from a trajectory, using the last `window` rounds.
pub fn collusiveness_from_trajectory(
    trajectory: &PriceTrajectory,
    nash_profits: &[f64],
    collusive_profits: &[f64],
    window: usize,
) -> Vec<f64> {
    let n = trajectory.num_players;
    let outcomes = &trajectory.outcomes;
    let start = if outcomes.len() > window { outcomes.len() - window } else { 0 };
    let recent = &outcomes[start..];

    let mut indices = vec![0.0; n];
    if recent.is_empty() {
        return indices;
    }

    for i in 0..n {
        let mean_profit: f64 = recent.iter().map(|o| o.profits[i]).sum::<f64>() / recent.len() as f64;
        indices[i] = collusiveness_index(mean_profit, nash_profits[i], collusive_profits[i]);
    }
    indices
}

/// Convert player actions to a values vector, sorted by player_id.
pub fn actions_to_values(actions: &[PlayerAction], num_players: usize) -> MarketSimResult<Vec<f64>> {
    if actions.len() != num_players {
        return Err(MarketSimError::SimulationError(format!(
            "Expected {num_players} actions, got {}",
            actions.len()
        )));
    }
    let mut values = vec![0.0; num_players];
    for a in actions {
        if a.player_id >= num_players {
            return Err(MarketSimError::PlayerError {
                player_id: a.player_id,
                message: format!("Player ID {} out of range [0, {})", a.player_id, num_players),
            });
        }
        values[a.player_id] = a.value;
    }
    Ok(values)
}

/// Create PlayerAction instances from a values vector.
pub fn values_to_actions(values: &[f64]) -> Vec<PlayerAction> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| PlayerAction::new(i, v))
        .collect()
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GameConfig {
        GameConfig::default()
    }

    #[test]
    fn test_market_factory_bertrand() {
        let market = MarketFactory::create(&default_config()).unwrap();
        assert_eq!(market.num_players(), 2);
        assert_eq!(market.market_type(), MarketType::Bertrand);
    }

    #[test]
    fn test_market_factory_cournot() {
        let mut config = default_config();
        config.market_type = MarketType::Cournot;
        let market = MarketFactory::create(&config).unwrap();
        assert_eq!(market.market_type(), MarketType::Cournot);
    }

    #[test]
    fn test_simple_bertrand() {
        let market = MarketFactory::simple_bertrand(2, 10.0, 2.0, 0.5, 1.0).unwrap();
        let ne = market.nash_equilibrium().unwrap();
        assert_eq!(ne.len(), 2);
        assert!(ne[0] > 1.0); // above MC
    }

    #[test]
    fn test_simple_cournot() {
        let market = MarketFactory::simple_cournot(2, 10.0, 1.0, 1.0).unwrap();
        let ne = market.nash_equilibrium().unwrap();
        assert_eq!(ne.len(), 2);
        assert!((ne[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bertrand_ne_above_competitive() {
        let market = MarketFactory::simple_bertrand(2, 10.0, 2.0, 0.5, 1.0).unwrap();
        let ne = market.nash_equilibrium().unwrap();
        let comp = market.competitive_values().unwrap();
        assert!(ne[0] > comp[0]);
    }

    #[test]
    fn test_bertrand_collusive_above_ne() {
        let market = MarketFactory::simple_bertrand(2, 10.0, 2.0, 0.5, 1.0).unwrap();
        let ne = market.nash_equilibrium().unwrap();
        let coll = market.collusive_values().unwrap();
        assert!(coll[0] > ne[0]);
    }

    #[test]
    fn test_market_state_new() {
        let state = MarketState::new(default_config());
        assert_eq!(state.current_round, 0);
        assert!(state.trajectory.is_empty());
        assert!(!state.should_terminate());
    }

    #[test]
    fn test_market_state_record() {
        let mut state = MarketState::new(default_config());
        let outcome = MarketOutcome::new(0, vec![3.0, 3.0], vec![5.5, 5.5], vec![11.0, 11.0]);
        state.record(outcome);
        assert_eq!(state.current_round, 1);
        assert_eq!(state.trajectory_len(), 1);
    }

    #[test]
    fn test_market_state_terminate() {
        let mut config = default_config();
        config.num_rounds = 2;
        let mut state = MarketState::new(config);
        let o1 = MarketOutcome::new(0, vec![3.0, 3.0], vec![5.0, 5.0], vec![10.0, 10.0]);
        let o2 = MarketOutcome::new(1, vec![3.0, 3.0], vec![5.0, 5.0], vec![10.0, 10.0]);
        state.record(o1);
        assert!(!state.should_terminate());
        state.record(o2);
        assert!(state.should_terminate());
    }

    #[test]
    fn test_market_state_recent_mean() {
        let mut state = MarketState::new(default_config());
        for r in 0..10 {
            let p = 2.0 + r as f64 * 0.1;
            let o = MarketOutcome::new(r, vec![p, p], vec![5.0, 5.0], vec![p * 2.0, p * 2.0]);
            state.record(o);
        }
        let mean = state.recent_mean_price(0, 5).unwrap();
        // Last 5 prices: 2.5, 2.6, 2.7, 2.8, 2.9 → mean = 2.7
        assert!((mean - 2.7).abs() < 1e-10);
    }

    #[test]
    fn test_collusiveness_index() {
        assert!((collusiveness_index(10.0, 5.0, 15.0) - 0.5).abs() < 1e-10);
        assert!((collusiveness_index(5.0, 5.0, 15.0) - 0.0).abs() < 1e-10);
        assert!((collusiveness_index(15.0, 5.0, 15.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_actions_to_values() {
        let actions = vec![
            PlayerAction::new(1, 5.0),
            PlayerAction::new(0, 3.0),
        ];
        let values = actions_to_values(&actions, 2).unwrap();
        assert!((values[0] - 3.0).abs() < 1e-10);
        assert!((values[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_values_to_actions() {
        let actions = values_to_actions(&[3.0, 5.0]);
        assert_eq!(actions[0].player_id, 0);
        assert!((actions[0].value - 3.0).abs() < 1e-10);
        assert_eq!(actions[1].player_id, 1);
    }

    #[test]
    fn test_clear_market_cournot() {
        let config = default_config();
        let p = clear_market_cournot(&config, &[3.0, 3.0]);
        // P = 10 - 1*(3+3) = 4
        assert!((p - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_state_reset() {
        let mut state = MarketState::new(default_config());
        let o = MarketOutcome::new(0, vec![3.0, 3.0], vec![5.0, 5.0], vec![10.0, 10.0]);
        state.record(o);
        state.reset();
        assert_eq!(state.current_round, 0);
        assert!(state.trajectory.is_empty());
    }

    #[test]
    fn test_simulate_round_bertrand() {
        let market = MarketFactory::simple_bertrand(2, 10.0, 2.0, 0.5, 1.0).unwrap();
        let actions = vec![PlayerAction::new(0, 3.0), PlayerAction::new(1, 3.0)];
        let outcome = market.simulate_round(&actions, 0).unwrap();
        assert_eq!(outcome.prices.len(), 2);
        assert!(outcome.profits[0] > 0.0);
    }

    #[test]
    fn test_simulate_round_cournot() {
        let market = MarketFactory::simple_cournot(2, 10.0, 1.0, 1.0).unwrap();
        let actions = vec![PlayerAction::new(0, 3.0), PlayerAction::new(1, 3.0)];
        let outcome = market.simulate_round(&actions, 0).unwrap();
        assert_eq!(outcome.quantities.len(), 2);
        assert!(outcome.profits[0] > 0.0);
    }

    #[test]
    fn test_collusiveness_from_trajectory() {
        let mut traj = PriceTrajectory::new(2);
        for r in 0..100 {
            traj.push(MarketOutcome::new(r, vec![5.0, 5.0], vec![3.0, 3.0], vec![12.0, 12.0]));
        }
        let ci = collusiveness_from_trajectory(&traj, &[8.0, 8.0], &[16.0, 16.0], 50);
        // Actual profit = 12, CI = (12-8)/(16-8) = 0.5
        assert!((ci[0] - 0.5).abs() < 1e-10);
    }
}
