//! Bertrand (price) competition model.
//!
//! In Bertrand competition, firms simultaneously choose prices and the market
//! determines quantities through the demand system. This module implements:
//!
//! - Cost models: constant marginal, linear, quadratic
//! - Profit computation and best-response price search
//! - Analytical Nash equilibrium for 2-player linear demand
//! - Monopoly and collusive price computation
//! - Price grid discretization
//! - Full profit landscape computation

use crate::demand::{self, DemandFunction};
use crate::types::*;
use crate::{MarketSimError, MarketSimResult};
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// Cost models
// ════════════════════════════════════════════════════════════════════════════

/// A cost function mapping quantity to total cost.
pub trait CostFunction: Send + Sync + std::fmt::Debug {
    /// Total cost of producing quantity q.
    fn total_cost(&self, q: f64) -> f64;
    /// Marginal cost at quantity q: dC/dq.
    fn marginal_cost(&self, q: f64) -> f64;
    /// Average cost at quantity q: C(q)/q. Returns 0 if q == 0.
    fn average_cost(&self, q: f64) -> f64 {
        if q.abs() < 1e-15 { 0.0 } else { self.total_cost(q) / q }
    }
    fn clone_box(&self) -> Box<dyn CostFunction>;
}

impl Clone for Box<dyn CostFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// C(q) = mc * q  (constant marginal cost).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantMarginalCost {
    pub mc: f64,
}

impl ConstantMarginalCost {
    pub fn new(mc: f64) -> MarketSimResult<Self> {
        if mc < 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Marginal cost must be non-negative".into(),
            ));
        }
        Ok(Self { mc })
    }
}

impl CostFunction for ConstantMarginalCost {
    fn total_cost(&self, q: f64) -> f64 { self.mc * q }
    fn marginal_cost(&self, _q: f64) -> f64 { self.mc }
    fn clone_box(&self) -> Box<dyn CostFunction> { Box::new(self.clone()) }
}

/// C(q) = fixed + mc * q  (linear cost with fixed component).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearCost {
    pub fixed: f64,
    pub mc: f64,
}

impl LinearCost {
    pub fn new(fixed: f64, mc: f64) -> MarketSimResult<Self> {
        if fixed < 0.0 || mc < 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Cost parameters must be non-negative".into(),
            ));
        }
        Ok(Self { fixed, mc })
    }
}

impl CostFunction for LinearCost {
    fn total_cost(&self, q: f64) -> f64 {
        if q > 0.0 { self.fixed + self.mc * q } else { 0.0 }
    }
    fn marginal_cost(&self, _q: f64) -> f64 { self.mc }
    fn clone_box(&self) -> Box<dyn CostFunction> { Box::new(self.clone()) }
}

/// C(q) = fixed + mc * q + quad * q²  (quadratic cost).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticCost {
    pub fixed: f64,
    pub mc: f64,
    pub quad: f64,
}

impl QuadraticCost {
    pub fn new(fixed: f64, mc: f64, quad: f64) -> MarketSimResult<Self> {
        if fixed < 0.0 || mc < 0.0 || quad < 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Cost parameters must be non-negative".into(),
            ));
        }
        Ok(Self { fixed, mc, quad })
    }
}

impl CostFunction for QuadraticCost {
    fn total_cost(&self, q: f64) -> f64 {
        if q > 0.0 {
            self.fixed + self.mc * q + self.quad * q * q
        } else {
            0.0
        }
    }
    fn marginal_cost(&self, q: f64) -> f64 {
        self.mc + 2.0 * self.quad * q
    }
    fn clone_box(&self) -> Box<dyn CostFunction> { Box::new(self.clone()) }
}

// ════════════════════════════════════════════════════════════════════════════
// Price grid
// ════════════════════════════════════════════════════════════════════════════

/// Discretised price grid for best-response search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceGrid {
    pub min: f64,
    pub max: f64,
    pub num_points: usize,
    pub step: f64,
}

impl PriceGrid {
    pub fn new(min: f64, max: f64, num_points: usize) -> MarketSimResult<Self> {
        if min >= max {
            return Err(MarketSimError::InvalidParameter(
                "price_min must be less than price_max".into(),
            ));
        }
        if num_points < 2 {
            return Err(MarketSimError::InvalidParameter(
                "num_points must be at least 2".into(),
            ));
        }
        let step = (max - min) / (num_points - 1) as f64;
        Ok(Self { min, max, num_points, step })
    }

    /// Get the price at grid index k.
    pub fn price_at(&self, k: usize) -> f64 {
        self.min + k as f64 * self.step
    }

    /// Snap a continuous price to the nearest grid point.
    pub fn snap(&self, price: f64) -> f64 {
        let k = ((price - self.min) / self.step).round() as usize;
        let k = k.min(self.num_points - 1);
        self.price_at(k)
    }

    /// Index of the nearest grid point.
    pub fn nearest_index(&self, price: f64) -> usize {
        let k = ((price - self.min) / self.step).round() as i64;
        k.max(0).min((self.num_points - 1) as i64) as usize
    }

    /// Iterator over all grid prices.
    pub fn iter(&self) -> PriceGridIter<'_> {
        PriceGridIter { grid: self, idx: 0 }
    }
}

pub struct PriceGridIter<'a> {
    grid: &'a PriceGrid,
    idx: usize,
}

impl<'a> Iterator for PriceGridIter<'a> {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        if self.idx < self.grid.num_points {
            let p = self.grid.price_at(self.idx);
            self.idx += 1;
            Some(p)
        } else {
            None
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BertrandMarket
// ════════════════════════════════════════════════════════════════════════════

/// A Bertrand competition market where firms set prices.
#[derive(Debug, Clone)]
pub struct BertrandMarket {
    pub demand: Box<dyn DemandFunction>,
    pub costs: Vec<Box<dyn CostFunction>>,
    pub price_grid: PriceGrid,
    pub num_players: usize,
}

impl std::fmt::Debug for Box<dyn DemandFunction> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DemandFunction(n={})", self.num_players())
    }
}

impl Clone for Box<dyn DemandFunction> {
    fn clone(&self) -> Self {
        // This is a placeholder; in production, each DemandFunction impl
        // would implement a clone_box method. For now we panic on clone
        // of trait objects. Users should construct new instances.
        panic!("Clone not supported for boxed DemandFunction; construct a new instance")
    }
}

impl BertrandMarket {
    pub fn new(
        demand: Box<dyn DemandFunction>,
        costs: Vec<Box<dyn CostFunction>>,
        price_grid: PriceGrid,
    ) -> MarketSimResult<Self> {
        let n = demand.num_players();
        if costs.len() != n {
            return Err(MarketSimError::ConfigError(format!(
                "Expected {n} cost functions, got {}",
                costs.len()
            )));
        }
        Ok(Self {
            num_players: n,
            demand,
            costs,
            price_grid,
        })
    }

    /// Construct a Bertrand market with constant marginal costs.
    pub fn with_constant_costs(
        demand: Box<dyn DemandFunction>,
        marginal_costs: &[f64],
        price_min: f64,
        price_max: f64,
        grid_size: usize,
    ) -> MarketSimResult<Self> {
        let costs: Vec<Box<dyn CostFunction>> = marginal_costs
            .iter()
            .map(|&mc| -> Box<dyn CostFunction> {
                Box::new(ConstantMarginalCost { mc })
            })
            .collect();
        let grid = PriceGrid::new(price_min, price_max, grid_size)?;
        Self::new(demand, costs, grid)
    }

    /// Compute profit for each player given a price vector.
    /// π_i = (p_i - AC_i(Q_i)) * Q_i  or equivalently  p_i*Q_i - C_i(Q_i)
    pub fn compute_profit(&self, prices: &[f64]) -> MarketSimResult<Vec<f64>> {
        let q = self.demand.compute_demand(prices)?;
        let mut profits = Vec::with_capacity(self.num_players);
        for i in 0..self.num_players {
            let revenue = prices[i] * q[i];
            let cost = self.costs[i].total_cost(q[i]);
            profits.push(revenue - cost);
        }
        Ok(profits)
    }

    /// Compute profit for a single player.
    pub fn compute_profit_single(
        &self,
        player: PlayerId,
        prices: &[f64],
    ) -> MarketSimResult<f64> {
        let q = self.demand.compute_demand(prices)?;
        let revenue = prices[player] * q[player];
        let cost = self.costs[player].total_cost(q[player]);
        Ok(revenue - cost)
    }

    /// Best-response price for a player given opponents' prices.
    /// Searches over the price grid to find the profit-maximizing price.
    pub fn best_response_price(
        &self,
        player: PlayerId,
        other_prices: &[f64],
    ) -> MarketSimResult<f64> {
        if other_prices.len() != self.num_players - 1 {
            return Err(MarketSimError::InvalidParameter(format!(
                "Expected {} other prices, got {}",
                self.num_players - 1,
                other_prices.len()
            )));
        }

        let mut best_price = self.price_grid.min;
        let mut best_profit = f64::NEG_INFINITY;

        for k in 0..self.price_grid.num_points {
            let p = self.price_grid.price_at(k);
            let mut all_prices = Vec::with_capacity(self.num_players);
            for j in 0..self.num_players {
                if j == player {
                    all_prices.push(p);
                } else {
                    let idx = if j < player { j } else { j - 1 };
                    all_prices.push(other_prices[idx]);
                }
            }

            if let Ok(profit) = self.compute_profit_single(player, &all_prices) {
                if profit > best_profit {
                    best_profit = profit;
                    best_price = p;
                }
            }
        }

        Ok(best_price)
    }

    /// Analytical Nash equilibrium for 2-player linear demand with constant MC.
    ///
    /// For Q_i = a - b*p_i + c*p_j, and constant marginal costs mc_i:
    /// Best response: p_i = (a + c*p_j + b*mc_i) / (2*b)
    /// Solving the system:
    ///   p_1* = (a*(2b+c) + b*(2b*mc_1 + c*mc_2)) / (4b² - c²)  (approx)
    pub fn compute_nash_equilibrium_linear_2p(
        a: f64,
        b: f64,
        c: f64,
        mc1: f64,
        mc2: f64,
    ) -> MarketSimResult<(f64, f64)> {
        let denom = 4.0 * b * b - c * c;
        if denom.abs() < 1e-15 {
            return Err(MarketSimError::EquilibriumNotFound(
                "Singular system (4b²-c²=0)".into(),
            ));
        }
        // From BR_1: p_1 = (a + c*p_2 + b*mc_1) / (2b)
        // From BR_2: p_2 = (a + c*p_1 + b*mc_2) / (2b)
        // Substituting:
        //   p_1 = (a + b*mc_1) / (2b) + c/(2b) * ((a + c*p_1 + b*mc_2) / (2b))
        //   p_1 * (1 - c²/(4b²)) = (a + b*mc_1)/(2b) + c*(a + b*mc_2)/(4b²)
        //   p_1 * (4b² - c²)/(4b²) = (2b(a + b*mc_1) + c(a + b*mc_2)) / (4b²)
        //   p_1 = (2b*a + 2b²*mc_1 + c*a + c*b*mc_2) / (4b² - c²)
        let p1 = (2.0 * b * a + 2.0 * b * b * mc1 + c * a + c * b * mc2) / denom;
        let p2 = (2.0 * b * a + 2.0 * b * b * mc2 + c * a + c * b * mc1) / denom;
        Ok((p1, p2))
    }

    /// Nash equilibrium for N-player symmetric case: all players have same cost.
    /// For symmetric linear demand Q_i = a - b*p_i + c*Σ_{j≠i} p_j with MC = mc:
    /// Symmetric NE price: p* = (a + b*mc) / (2b - (n-1)*c)  ... but need to be careful.
    ///
    /// Best response: p_i = (a + c * Σ_{j≠i} p_j + b*mc) / (2b)
    /// At symmetric NE: p* = (a + (n-1)*c*p* + b*mc) / (2b)
    ///   => p*(2b - (n-1)*c) = a + b*mc
    ///   => p* = (a + b*mc) / (2b - (n-1)*c)
    pub fn compute_nash_equilibrium_symmetric(
        a: f64,
        b: f64,
        c: f64,
        mc: f64,
        num_players: usize,
    ) -> MarketSimResult<f64> {
        let denom = 2.0 * b - (num_players as f64 - 1.0) * c;
        if denom.abs() < 1e-15 {
            return Err(MarketSimError::EquilibriumNotFound(
                "Degenerate symmetric system".into(),
            ));
        }
        Ok((a + b * mc) / denom)
    }

    /// Monopoly price for player `player` (assuming they're the only firm).
    /// For linear demand Q = a - b*p with MC = mc:
    ///   p_monopoly = (a + b*mc) / (2*b)
    pub fn compute_monopoly_price_linear(a: f64, b: f64, mc: f64) -> f64 {
        (a + b * mc) / (2.0 * b)
    }

    /// Collusive (joint-profit-maximizing) prices for N symmetric firms.
    /// Each firm acts as if it's a multi-product monopolist maximizing total profit.
    /// For symmetric linear demand with N firms:
    ///   p_collusive = (a + (b - (n-1)*c)*mc) / (2*(b - (n-1)*c))
    pub fn compute_collusive_price_symmetric(
        a: f64,
        b: f64,
        c: f64,
        mc: f64,
        num_players: usize,
    ) -> MarketSimResult<f64> {
        let effective_slope = b - (num_players as f64 - 1.0) * c;
        if effective_slope <= 0.0 {
            return Err(MarketSimError::EquilibriumNotFound(
                "Collusive price undefined (effective slope non-positive)".into(),
            ));
        }
        Ok((a + effective_slope * mc) / (2.0 * effective_slope))
    }

    /// Homogeneous Bertrand: with identical products, NE is p* = MC.
    pub fn homogeneous_nash_price(mc: f64) -> f64 {
        mc
    }

    /// Simulate one round of Bertrand competition.
    pub fn simulate_round(
        &self,
        actions: &[PlayerAction],
        round: RoundNumber,
    ) -> MarketSimResult<MarketOutcome> {
        if actions.len() != self.num_players {
            return Err(MarketSimError::SimulationError(format!(
                "Expected {} actions, got {}",
                self.num_players,
                actions.len()
            )));
        }
        let prices: Vec<f64> = actions.iter().map(|a| a.value).collect();
        let quantities = self.demand.compute_demand(&prices)?;
        let profits = self.compute_profit(&prices)?;
        Ok(MarketOutcome::new(round, prices, quantities, profits))
    }

    /// Compute the full profit landscape for a player across the price grid,
    /// holding other players' prices fixed.
    pub fn profit_landscape(
        &self,
        player: PlayerId,
        other_prices: &[f64],
    ) -> MarketSimResult<Vec<(f64, f64)>> {
        if other_prices.len() != self.num_players - 1 {
            return Err(MarketSimError::InvalidParameter(
                "other_prices length mismatch".into(),
            ));
        }
        let mut landscape = Vec::with_capacity(self.price_grid.num_points);
        for k in 0..self.price_grid.num_points {
            let p = self.price_grid.price_at(k);
            let mut all_prices = Vec::with_capacity(self.num_players);
            for j in 0..self.num_players {
                if j == player {
                    all_prices.push(p);
                } else {
                    let idx = if j < player { j } else { j - 1 };
                    all_prices.push(other_prices[idx]);
                }
            }
            let profit = self.compute_profit_single(player, &all_prices).unwrap_or(0.0);
            landscape.push((p, profit));
        }
        Ok(landscape)
    }

    /// Compute Nash equilibrium numerically by iterated best response.
    /// Returns the equilibrium prices or an error if convergence fails.
    pub fn compute_nash_iterative(
        &self,
        initial_prices: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> MarketSimResult<Vec<f64>> {
        let n = self.num_players;
        let mut prices = initial_prices.to_vec();

        for _iter in 0..max_iter {
            let old_prices = prices.clone();
            for i in 0..n {
                let others: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| prices[j]).collect();
                prices[i] = self.best_response_price(i, &others)?;
            }
            let max_change: f64 = prices
                .iter()
                .zip(old_prices.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_change < tol {
                return Ok(prices);
            }
        }
        Err(MarketSimError::EquilibriumNotFound(format!(
            "Iterated best response did not converge in {max_iter} iterations"
        )))
    }

    /// Check if given prices constitute a Nash equilibrium (within tolerance).
    pub fn is_nash_equilibrium(&self, prices: &[f64], tol: f64) -> MarketSimResult<bool> {
        let n = self.num_players;
        for i in 0..n {
            let others: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| prices[j]).collect();
            let br = self.best_response_price(i, &others)?;
            if (br - prices[i]).abs() > tol {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Profit gained by deviating from given prices to best response.
    pub fn deviation_profit(
        &self,
        player: PlayerId,
        prices: &[f64],
    ) -> MarketSimResult<f64> {
        let current_profit = self.compute_profit_single(player, prices)?;
        let others: Vec<f64> = (0..self.num_players)
            .filter(|&j| j != player)
            .map(|j| prices[j])
            .collect();
        let br = self.best_response_price(player, &others)?;
        let mut dev_prices = prices.to_vec();
        dev_prices[player] = br;
        let dev_profit = self.compute_profit_single(player, &dev_prices)?;
        Ok(dev_profit - current_profit)
    }

    /// Competitive (marginal-cost) prices.
    pub fn competitive_prices(&self) -> Vec<f64> {
        (0..self.num_players)
            .map(|i| self.costs[i].marginal_cost(0.0))
            .collect()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Builder for BertrandMarket from GameConfig
// ════════════════════════════════════════════════════════════════════════════

/// Build a BertrandMarket from a GameConfig.
pub fn bertrand_from_config(config: &GameConfig) -> MarketSimResult<BertrandMarket> {
    let demand = demand::create_demand_from_config(config)?;
    let costs: Vec<Box<dyn CostFunction>> = config
        .marginal_costs
        .iter()
        .map(|&mc| -> Box<dyn CostFunction> {
            Box::new(ConstantMarginalCost { mc })
        })
        .collect();
    let grid = PriceGrid::new(config.price_min, config.price_max, config.price_grid_size)?;
    BertrandMarket::new(demand, costs, grid)
}

// ════════════════════════════════════════════════════════════════════════════
// Bertrand profit analysis helpers
// ════════════════════════════════════════════════════════════════════════════

/// Compute the consumer surplus for given prices under linear demand.
/// CS = Σ_i ∫_0^{Q_i} P_i^{inv}(q) dq - p_i * Q_i
/// For linear demand this simplifies to: CS_i = Q_i² / (2*b)
pub fn consumer_surplus_linear(a: f64, b: f64, c: f64, prices: &[f64]) -> f64 {
    let n = prices.len();
    let mut cs = 0.0;
    for i in 0..n {
        let cross_sum: f64 = prices.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| *p)
            .sum();
        let q = (a - b * prices[i] + c * cross_sum).max(0.0);
        // CS_i = q²/(2b) approximately
        cs += q * q / (2.0 * b);
    }
    cs
}

/// Total welfare = consumer surplus + total profit.
pub fn total_welfare_linear(
    a: f64,
    b: f64,
    c: f64,
    prices: &[f64],
    marginal_costs: &[f64],
) -> f64 {
    let n = prices.len();
    let cs = consumer_surplus_linear(a, b, c, prices);
    let mut total_profit = 0.0;
    for i in 0..n {
        let cross_sum: f64 = prices.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| *p)
            .sum();
        let q = (a - b * prices[i] + c * cross_sum).max(0.0);
        total_profit += (prices[i] - marginal_costs[i]) * q;
    }
    cs + total_profit
}

/// Markup: (p - mc) / p  (Lerner index).
pub fn lerner_index(price: f64, mc: f64) -> f64 {
    if price.abs() < 1e-15 { 0.0 } else { (price - mc) / price }
}

/// Relative profit gain from collusion: (π_collusive - π_nash) / π_nash.
pub fn collusion_profit_gain(nash_profit: f64, collusive_profit: f64) -> f64 {
    if nash_profit.abs() < 1e-15 {
        if collusive_profit > 0.0 { f64::INFINITY } else { 0.0 }
    } else {
        (collusive_profit - nash_profit) / nash_profit
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demand::LinearDemand;

    fn make_linear_2p() -> (LinearDemand, BertrandMarket) {
        let demand = LinearDemand::new(10.0, 2.0, 0.5, 2).unwrap();
        let costs: Vec<Box<dyn CostFunction>> = vec![
            Box::new(ConstantMarginalCost { mc: 1.0 }),
            Box::new(ConstantMarginalCost { mc: 1.0 }),
        ];
        let grid = PriceGrid::new(0.0, 10.0, 1001).unwrap();
        let market = BertrandMarket::new(
            Box::new(demand.clone()),
            costs,
            grid,
        ).unwrap();
        (demand, market)
    }

    fn make_linear_demand_box(a: f64, b: f64, c: f64, n: usize) -> Box<dyn DemandFunction> {
        Box::new(LinearDemand::new(a, b, c, n).unwrap())
    }

    #[test]
    fn test_constant_marginal_cost() {
        let c = ConstantMarginalCost::new(2.0).unwrap();
        assert!((c.total_cost(5.0) - 10.0).abs() < 1e-10);
        assert!((c.marginal_cost(5.0) - 2.0).abs() < 1e-10);
        assert!((c.average_cost(5.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_cost() {
        let c = LinearCost::new(10.0, 2.0).unwrap();
        assert!((c.total_cost(5.0) - 20.0).abs() < 1e-10);
        assert!((c.total_cost(0.0) - 0.0).abs() < 1e-10); // no cost at zero
        assert!((c.marginal_cost(5.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_cost() {
        let c = QuadraticCost::new(10.0, 2.0, 0.5).unwrap();
        // C(3) = 10 + 2*3 + 0.5*9 = 10 + 6 + 4.5 = 20.5
        assert!((c.total_cost(3.0) - 20.5).abs() < 1e-10);
        // MC(3) = 2 + 2*0.5*3 = 5
        assert!((c.marginal_cost(3.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_price_grid() {
        let g = PriceGrid::new(0.0, 10.0, 101).unwrap();
        assert!((g.step - 0.1).abs() < 1e-10);
        assert!((g.price_at(0) - 0.0).abs() < 1e-10);
        assert!((g.price_at(100) - 10.0).abs() < 1e-10);
        assert!((g.snap(3.14) - 3.1).abs() < 0.05 + 1e-10);
    }

    #[test]
    fn test_price_grid_iter() {
        let g = PriceGrid::new(0.0, 1.0, 11).unwrap();
        let prices: Vec<f64> = g.iter().collect();
        assert_eq!(prices.len(), 11);
        assert!((prices[0] - 0.0).abs() < 1e-10);
        assert!((prices[10] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bertrand_profit() {
        let (_, market) = make_linear_2p();
        let profits = market.compute_profit(&[3.0, 3.0]).unwrap();
        // Q_i = 10 - 2*3 + 0.5*3 = 5.5
        // π_i = (3 - 1) * 5.5 = 11.0
        assert!((profits[0] - 11.0).abs() < 1e-10);
        assert!((profits[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_bertrand_profit_asymmetric() {
        let (_, market) = make_linear_2p();
        let profits = market.compute_profit(&[2.0, 4.0]).unwrap();
        // Q_0 = 10 - 2*2 + 0.5*4 = 8.0, π_0 = (2-1)*8 = 8
        // Q_1 = 10 - 2*4 + 0.5*2 = 3.0, π_1 = (4-1)*3 = 9
        assert!((profits[0] - 8.0).abs() < 1e-10);
        assert!((profits[1] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_nash_equilibrium_2p() {
        let (p1, p2) = BertrandMarket::compute_nash_equilibrium_linear_2p(
            10.0, 2.0, 0.5, 1.0, 1.0,
        ).unwrap();
        // Symmetric: p1 == p2
        assert!((p1 - p2).abs() < 1e-10);
        // p* = (2*2*10 + 2*4*1 + 0.5*10 + 0.5*2*1) / (16 - 0.25)
        //    = (40 + 8 + 5 + 1) / 15.75 = 54 / 15.75
        let expected = (2.0 * 2.0 * 10.0 + 2.0 * 4.0 * 1.0 + 0.5 * 10.0 + 0.5 * 2.0 * 1.0)
            / (4.0 * 4.0 - 0.25);
        assert!((p1 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_nash_symmetric() {
        let p = BertrandMarket::compute_nash_equilibrium_symmetric(
            10.0, 2.0, 0.5, 1.0, 2,
        ).unwrap();
        // p* = (10 + 2*1) / (4 - 0.5) = 12/3.5 ≈ 3.4286
        assert!((p - 12.0 / 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_monopoly_price() {
        let p = BertrandMarket::compute_monopoly_price_linear(10.0, 2.0, 1.0);
        // p_m = (10 + 2) / 4 = 3.0
        assert!((p - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_collusive_price() {
        let p = BertrandMarket::compute_collusive_price_symmetric(
            10.0, 2.0, 0.5, 1.0, 2,
        ).unwrap();
        // effective_slope = 2 - 0.5 = 1.5
        // p_c = (10 + 1.5*1) / (2*1.5) = 11.5 / 3.0 ≈ 3.833
        assert!((p - 11.5 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_collusive_above_nash() {
        let nash = BertrandMarket::compute_nash_equilibrium_symmetric(
            10.0, 2.0, 0.5, 1.0, 2,
        ).unwrap();
        let collusive = BertrandMarket::compute_collusive_price_symmetric(
            10.0, 2.0, 0.5, 1.0, 2,
        ).unwrap();
        // For differentiated Bertrand, the single-product monopoly price
        // (a + b*mc)/(2b) is NOT the upper bound; the multi-product
        // monopoly (= collusive) price can exceed it because firms
        // internalize positive cross-price effects of substitute goods.
        // The correct ordering is: MC < Nash < Collusive
        assert!(1.0 < nash);
        assert!(nash < collusive);
        // Collusive price should still be finite and reasonable
        assert!(collusive < 100.0);
    }

    #[test]
    fn test_homogeneous_nash() {
        let p = BertrandMarket::homogeneous_nash_price(2.0);
        assert!((p - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_best_response() {
        let (_, market) = make_linear_2p();
        let br = market.best_response_price(0, &[3.0]).unwrap();
        // BR_0(p_1=3) = (10 + 0.5*3 + 2*1) / (2*2) = 13.5 / 4 = 3.375
        let analytical = (10.0 + 0.5 * 3.0 + 2.0 * 1.0) / (2.0 * 2.0);
        assert!((br - analytical).abs() < 0.02); // grid tolerance
    }

    #[test]
    fn test_simulate_round() {
        let (_, market) = make_linear_2p();
        let actions = vec![
            PlayerAction::new(0, 3.0),
            PlayerAction::new(1, 3.0),
        ];
        let outcome = market.simulate_round(&actions, 0).unwrap();
        assert_eq!(outcome.round, 0);
        assert_eq!(outcome.prices.len(), 2);
        assert_eq!(outcome.quantities.len(), 2);
        assert!((outcome.profits[0] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_profit_landscape() {
        let (_, market) = make_linear_2p();
        let landscape = market.profit_landscape(0, &[3.0]).unwrap();
        assert_eq!(landscape.len(), 1001);
        // Should have a maximum somewhere around the best response
        let max_entry = landscape
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!(max_entry.0 > 1.0);
        assert!(max_entry.1 > 0.0);
    }

    #[test]
    fn test_is_nash_equilibrium() {
        let (_, market) = make_linear_2p();
        let (p1, p2) = BertrandMarket::compute_nash_equilibrium_linear_2p(
            10.0, 2.0, 0.5, 1.0, 1.0,
        ).unwrap();
        // With grid discretization, allow larger tolerance
        let result = market.is_nash_equilibrium(&[p1, p2], 0.02).unwrap();
        assert!(result);
    }

    #[test]
    fn test_deviation_profit() {
        let (_, market) = make_linear_2p();
        let (p1, p2) = BertrandMarket::compute_nash_equilibrium_linear_2p(
            10.0, 2.0, 0.5, 1.0, 1.0,
        ).unwrap();
        let dev = market.deviation_profit(0, &[p1, p2]).unwrap();
        // At NE, deviation profit should be near zero (within grid tolerance)
        assert!(dev.abs() < 0.1);
    }

    #[test]
    fn test_competitive_prices() {
        let (_, market) = make_linear_2p();
        let cp = market.competitive_prices();
        assert!((cp[0] - 1.0).abs() < 1e-10);
        assert!((cp[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lerner_index_computation() {
        assert!((lerner_index(5.0, 1.0) - 0.8).abs() < 1e-10);
        assert!((lerner_index(2.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_consumer_surplus() {
        let cs = consumer_surplus_linear(10.0, 2.0, 0.5, &[3.0, 3.0]);
        assert!(cs > 0.0);
    }

    #[test]
    fn test_total_welfare() {
        let w = total_welfare_linear(10.0, 2.0, 0.5, &[3.0, 3.0], &[1.0, 1.0]);
        assert!(w > 0.0);
    }

    #[test]
    fn test_bertrand_from_config() {
        let config = GameConfig::default();
        let market = bertrand_from_config(&config).unwrap();
        assert_eq!(market.num_players, 2);
    }

    #[test]
    fn test_asymmetric_costs() {
        let demand = LinearDemand::new(10.0, 2.0, 0.5, 2).unwrap();
        let costs: Vec<Box<dyn CostFunction>> = vec![
            Box::new(ConstantMarginalCost { mc: 1.0 }),
            Box::new(ConstantMarginalCost { mc: 2.0 }),
        ];
        let grid = PriceGrid::new(0.0, 10.0, 1001).unwrap();
        let market = BertrandMarket::new(Box::new(demand), costs, grid).unwrap();

        let (p1, p2) = BertrandMarket::compute_nash_equilibrium_linear_2p(
            10.0, 2.0, 0.5, 1.0, 2.0,
        ).unwrap();
        // Higher-cost firm should have higher NE price
        assert!(p2 > p1);

        let profits = market.compute_profit(&[p1, p2]).unwrap();
        assert!(profits[0] > 0.0);
        assert!(profits[1] > 0.0);
    }

    #[test]
    fn test_negative_cost_rejected() {
        assert!(ConstantMarginalCost::new(-1.0).is_err());
        assert!(LinearCost::new(-1.0, 1.0).is_err());
        assert!(QuadraticCost::new(0.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_wrong_action_count() {
        let (_, market) = make_linear_2p();
        let actions = vec![PlayerAction::new(0, 3.0)]; // only 1
        assert!(market.simulate_round(&actions, 0).is_err());
    }
}
