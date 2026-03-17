//! Helper utilities replacing non-existent `MarketSimulator` and various
//! game-theory convenience functions.

use shared_types::{
    CollusionResult, Cost, DemandSystem, GameConfig, MarketOutcome, MarketType,
    PlayerId, PlayerAction, Price, Profit, Quantity, RoundNumber,
};

/// Compute a market outcome given prices and game config.
/// Replaces the non-existent `market_sim::MarketSimulator::step()`.
pub(crate) fn simulate_single_round(
    prices: &[Price],
    game_config: &GameConfig,
    round: RoundNumber,
) -> MarketOutcome {
    let n = prices.len();
    let quantities = game_config
        .demand_system
        .compute_quantities(prices, n);
    let profits: Vec<Profit> = (0..n)
        .map(|i| {
            let cost = game_config
                .marginal_costs
                .get(i)
                .map(|c| c.0)
                .unwrap_or(0.0);
            Profit((prices[i].0 - cost) * quantities[i].0)
        })
        .collect();
    let actions: Vec<PlayerAction> = prices
        .iter()
        .enumerate()
        .map(|(i, &p)| PlayerAction::new(PlayerId(i), p))
        .collect();
    MarketOutcome::new(round, actions, prices.to_vec(), quantities, profits)
}

/// Extract price bounds from game config, falling back to defaults.
pub(crate) fn price_bounds_from_config(game_config: &GameConfig) -> (Price, Price) {
    game_config
        .price_grid
        .as_ref()
        .map(|g| (g.min_price, g.max_price))
        .unwrap_or((Price(0.0), Price(10.0)))
}

/// Compute Nash equilibrium using the appropriate solver for the market type.
pub(crate) fn compute_nash(
    game_config: &GameConfig,
) -> CollusionResult<game_theory::NashEquilibrium> {
    match game_config.market_type {
        MarketType::Bertrand => game_theory::BertrandNashSolver::solve(game_config),
        MarketType::Cournot => game_theory::CournotNashSolver::solve(game_config),
    }
}

/// Simple collusive outcome: monopoly-pricing profits.
pub(crate) struct CollusiveOutcome {
    pub payoffs: Vec<f64>,
}

/// Compute collusive (joint-profit-maximizing) outcome.
pub(crate) fn compute_collusive(
    game_config: &GameConfig,
) -> CollusionResult<CollusiveOutcome> {
    let mc = game_config
        .marginal_costs
        .first()
        .copied()
        .unwrap_or(Cost(0.0));
    let monopoly_price = game_config.demand_system.monopoly_price(mc);
    let prices: Vec<Price> = vec![monopoly_price; game_config.num_players];
    let quantities = game_config
        .demand_system
        .compute_quantities(&prices, game_config.num_players);
    let payoffs: Vec<f64> = (0..game_config.num_players)
        .map(|i| {
            let cost = game_config
                .marginal_costs
                .get(i)
                .map(|c| c.0)
                .unwrap_or(mc.0);
            (monopoly_price.0 - cost) * quantities[i].0
        })
        .collect();
    Ok(CollusiveOutcome { payoffs })
}

/// Compute folk theorem minimum discount factor.
/// delta_min = (dev_profit - collusive_profit) / (dev_profit - nash_profit)
pub(crate) fn folk_theorem_min_discount(
    nash_profit: f64,
    collusive_profit: f64,
    deviation_profit: f64,
) -> f64 {
    let numerator = deviation_profit - collusive_profit;
    let denominator = deviation_profit - nash_profit;
    if denominator.abs() < 1e-12 {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

/// Compute best-response price for a player given opponent prices.
pub(crate) fn compute_best_response_price(
    player_idx: usize,
    opponent_prices: &[Price],
    marginal_cost: Cost,
    demand_system: &DemandSystem,
    price_bounds: (Price, Price),
    grid_size: usize,
) -> Price {
    let (lo, hi) = price_bounds;
    let step = (hi.0 - lo.0) / grid_size as f64;
    let mut best_price = lo;
    let mut best_profit = f64::NEG_INFINITY;

    let n_total = opponent_prices.len() + 1;

    for i in 0..=grid_size {
        let candidate_price = Price(lo.0 + i as f64 * step);
        // Build full price vector with the candidate for this player
        let mut all_prices: Vec<Price> = Vec::with_capacity(n_total);
        let mut opp_idx = 0;
        for j in 0..n_total {
            if j == player_idx {
                all_prices.push(candidate_price);
            } else {
                all_prices.push(opponent_prices[opp_idx]);
                opp_idx += 1;
            }
        }
        let quantities = demand_system.compute_quantities(&all_prices, n_total);
        let profit = (candidate_price.0 - marginal_cost.0) * quantities[player_idx].0;
        if profit > best_profit {
            best_profit = profit;
            best_price = candidate_price;
        }
    }
    best_price
}

/// Build a test PriceTrajectory (for use in test modules).
#[cfg(test)]
pub(crate) fn make_test_trajectory(
    n_rounds: usize,
    n_players: usize,
    price: Price,
) -> shared_types::PriceTrajectory {
    use shared_types::{AlgorithmType, PriceTrajectory};

    let outcomes: Vec<MarketOutcome> = (0..n_rounds)
        .map(|r| {
            let prices = vec![price; n_players];
            let quantities = vec![Quantity(1.0); n_players];
            let profits: Vec<Profit> = (0..n_players)
                .map(|_| Profit((price.0 - 1.0) * 1.0))
                .collect();
            let actions: Vec<PlayerAction> = (0..n_players)
                .map(|i| PlayerAction::new(PlayerId(i), price))
                .collect();
            MarketOutcome::new(
                RoundNumber(r),
                actions,
                prices,
                quantities,
                profits,
            )
        })
        .collect();
    PriceTrajectory::new(
        outcomes,
        MarketType::Bertrand,
        n_players,
        AlgorithmType::QLearning,
        42,
    )
}

/// Build a test GameConfig (for use in test modules).
#[cfg(test)]
pub(crate) fn make_test_game_config(n_players: usize) -> GameConfig {
    GameConfig {
        market_type: MarketType::Bertrand,
        demand_system: DemandSystem::Linear {
            max_quantity: 10.0,
            slope: 1.0,
        },
        num_players: n_players,
        discount_factor: 0.95,
        marginal_costs: vec![Cost(1.0); n_players],
        price_grid: None,
        max_rounds: 1000,
        description: String::new(),
    }
}

/// Build a default SimulationConfig for testing.
#[cfg(test)]
pub(crate) fn make_test_sim_config(game_config: &GameConfig) -> shared_types::SimulationConfig {
    use shared_types::{AlgorithmConfig, AlgorithmType, EvaluationMode, SimulationConfig};
    let algo = AlgorithmConfig::new(AlgorithmType::QLearning);
    SimulationConfig::new(game_config.clone(), algo, EvaluationMode::Standard)
}
