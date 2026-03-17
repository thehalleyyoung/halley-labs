//! Cournot (quantity) competition model.
//!
//! In Cournot competition, firms simultaneously choose quantities and the market
//! price is determined by inverse demand. This module implements:
//!
//! - Inverse demand functions
//! - Profit computation and best-response quantity search
//! - Analytical Nash equilibrium for N-player symmetric/asymmetric cases
//! - Monopoly and collusive output computation
//! - Reaction functions and stability analysis
//! - Quantity grid discretization

use crate::bertrand::{CostFunction, ConstantMarginalCost};
use crate::types::*;
use crate::{MarketSimError, MarketSimResult};
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// Inverse demand
// ════════════════════════════════════════════════════════════════════════════

/// Inverse demand: maps total quantity to market price.
pub trait InverseDemand: Send + Sync + std::fmt::Debug {
    /// Market price given total industry output.
    fn price(&self, total_quantity: f64) -> f64;

    /// Marginal revenue for a firm producing `own_q` when total output is `total_q`.
    /// MR = P(Q) + q * P'(Q)
    fn marginal_revenue(&self, own_q: f64, total_q: f64) -> f64;

    /// Derivative of inverse demand: dP/dQ.
    fn price_derivative(&self, total_quantity: f64) -> f64;

    /// Maximum total quantity before price hits zero.
    fn max_quantity(&self) -> f64;

    fn clone_box(&self) -> Box<dyn InverseDemand>;
}

impl Clone for Box<dyn InverseDemand> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Linear inverse demand: P = a - b * Q_total, where Q_total = Σ q_i.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearInverseDemand {
    pub intercept: f64,
    pub slope: f64,
}

impl LinearInverseDemand {
    pub fn new(intercept: f64, slope: f64) -> MarketSimResult<Self> {
        if intercept <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Inverse demand intercept must be positive".into(),
            ));
        }
        if slope <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Inverse demand slope must be positive".into(),
            ));
        }
        Ok(Self { intercept, slope })
    }
}

impl InverseDemand for LinearInverseDemand {
    fn price(&self, total_quantity: f64) -> f64 {
        (self.intercept - self.slope * total_quantity).max(0.0)
    }

    fn marginal_revenue(&self, own_q: f64, total_q: f64) -> f64 {
        self.intercept - self.slope * total_q - self.slope * own_q
    }

    fn price_derivative(&self, _total_quantity: f64) -> f64 {
        -self.slope
    }

    fn max_quantity(&self) -> f64 {
        self.intercept / self.slope
    }

    fn clone_box(&self) -> Box<dyn InverseDemand> {
        Box::new(self.clone())
    }
}

/// Iso-elastic inverse demand: P = A * Q^{-1/ε} where ε is demand elasticity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsoelasticInverseDemand {
    pub scale: f64,
    pub elasticity: f64,
}

impl IsoelasticInverseDemand {
    pub fn new(scale: f64, elasticity: f64) -> MarketSimResult<Self> {
        if scale <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Scale must be positive".into(),
            ));
        }
        if elasticity <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Elasticity must be positive".into(),
            ));
        }
        Ok(Self { scale, elasticity })
    }
}

impl InverseDemand for IsoelasticInverseDemand {
    fn price(&self, total_quantity: f64) -> f64 {
        if total_quantity <= 0.0 {
            return f64::INFINITY;
        }
        self.scale * total_quantity.powf(-1.0 / self.elasticity)
    }

    fn marginal_revenue(&self, own_q: f64, total_q: f64) -> f64 {
        if total_q <= 0.0 {
            return f64::INFINITY;
        }
        let p = self.price(total_q);
        p - (own_q / self.elasticity) * self.scale
            * total_q.powf(-1.0 / self.elasticity - 1.0)
    }

    fn price_derivative(&self, total_quantity: f64) -> f64 {
        if total_quantity <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -self.scale / self.elasticity * total_quantity.powf(-1.0 / self.elasticity - 1.0)
    }

    fn max_quantity(&self) -> f64 {
        // Price approaches zero as Q → ∞, but never reaches zero
        f64::INFINITY
    }

    fn clone_box(&self) -> Box<dyn InverseDemand> {
        Box::new(self.clone())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Quantity grid
// ════════════════════════════════════════════════════════════════════════════

/// Discretised quantity grid for best-response search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantityGrid {
    pub min: f64,
    pub max: f64,
    pub num_points: usize,
    pub step: f64,
}

impl QuantityGrid {
    pub fn new(min: f64, max: f64, num_points: usize) -> MarketSimResult<Self> {
        if min > max {
            return Err(MarketSimError::InvalidParameter(
                "quantity_min must be ≤ quantity_max".into(),
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

    pub fn quantity_at(&self, k: usize) -> f64 {
        self.min + k as f64 * self.step
    }

    pub fn snap(&self, quantity: f64) -> f64 {
        let k = ((quantity - self.min) / self.step).round() as usize;
        let k = k.min(self.num_points - 1);
        self.quantity_at(k)
    }

    pub fn nearest_index(&self, quantity: f64) -> usize {
        let k = ((quantity - self.min) / self.step).round() as i64;
        k.max(0).min((self.num_points - 1) as i64) as usize
    }

    pub fn iter(&self) -> QuantityGridIter<'_> {
        QuantityGridIter { grid: self, idx: 0 }
    }
}

pub struct QuantityGridIter<'a> {
    grid: &'a QuantityGrid,
    idx: usize,
}

impl<'a> Iterator for QuantityGridIter<'a> {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        if self.idx < self.grid.num_points {
            let q = self.grid.quantity_at(self.idx);
            self.idx += 1;
            Some(q)
        } else {
            None
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CournotMarket
// ════════════════════════════════════════════════════════════════════════════

/// A Cournot competition market where firms set quantities.
#[derive(Debug, Clone)]
pub struct CournotMarket {
    pub inverse_demand: Box<dyn InverseDemand>,
    pub costs: Vec<Box<dyn CostFunction>>,
    pub quantity_grid: QuantityGrid,
    pub num_players: usize,
}

impl CournotMarket {
    pub fn new(
        inverse_demand: Box<dyn InverseDemand>,
        costs: Vec<Box<dyn CostFunction>>,
        quantity_grid: QuantityGrid,
    ) -> MarketSimResult<Self> {
        let n = costs.len();
        if n < 2 || n > 4 {
            return Err(MarketSimError::ConfigError(
                "num_players must be in [2, 4]".into(),
            ));
        }
        Ok(Self {
            num_players: n,
            inverse_demand,
            costs,
            quantity_grid,
        })
    }

    /// Construct a Cournot market with linear inverse demand and constant MC.
    pub fn linear_with_constant_costs(
        intercept: f64,
        slope: f64,
        marginal_costs: &[f64],
        q_min: f64,
        q_max: f64,
        grid_size: usize,
    ) -> MarketSimResult<Self> {
        let inv_demand = Box::new(LinearInverseDemand::new(intercept, slope)?);
        let costs: Vec<Box<dyn CostFunction>> = marginal_costs
            .iter()
            .map(|&mc| -> Box<dyn CostFunction> {
                Box::new(ConstantMarginalCost { mc })
            })
            .collect();
        let grid = QuantityGrid::new(q_min, q_max, grid_size)?;
        Self::new(inv_demand, costs, grid)
    }

    /// Compute profit for each player given a quantity vector.
    /// π_i = P(Q) * q_i - C_i(q_i)  where Q = Σ q_j
    pub fn compute_profit(&self, quantities: &[f64]) -> MarketSimResult<Vec<f64>> {
        let n = self.num_players;
        if quantities.len() != n {
            return Err(MarketSimError::InvalidParameter(format!(
                "Expected {n} quantities, got {}",
                quantities.len()
            )));
        }
        let total_q: f64 = quantities.iter().sum();
        let price = self.inverse_demand.price(total_q);
        let mut profits = Vec::with_capacity(n);
        for i in 0..n {
            let revenue = price * quantities[i];
            let cost = self.costs[i].total_cost(quantities[i]);
            profits.push(revenue - cost);
        }
        Ok(profits)
    }

    /// Compute profit for a single player.
    pub fn compute_profit_single(
        &self,
        player: PlayerId,
        quantities: &[f64],
    ) -> MarketSimResult<f64> {
        let total_q: f64 = quantities.iter().sum();
        let price = self.inverse_demand.price(total_q);
        let revenue = price * quantities[player];
        let cost = self.costs[player].total_cost(quantities[player]);
        Ok(revenue - cost)
    }

    /// Best-response quantity for a player given opponents' quantities.
    pub fn best_response_quantity(
        &self,
        player: PlayerId,
        other_quantities: &[f64],
    ) -> MarketSimResult<f64> {
        if other_quantities.len() != self.num_players - 1 {
            return Err(MarketSimError::InvalidParameter(format!(
                "Expected {} other quantities, got {}",
                self.num_players - 1,
                other_quantities.len()
            )));
        }

        let others_total: f64 = other_quantities.iter().sum();
        let mut best_q = 0.0;
        let mut best_profit = f64::NEG_INFINITY;

        for k in 0..self.quantity_grid.num_points {
            let q = self.quantity_grid.quantity_at(k);
            let total = others_total + q;
            let price = self.inverse_demand.price(total);
            let revenue = price * q;
            let cost = self.costs[player].total_cost(q);
            let profit = revenue - cost;

            if profit > best_profit {
                best_profit = profit;
                best_q = q;
            }
        }

        Ok(best_q)
    }

    /// Analytical best response for linear inverse demand with constant MC.
    /// BR_i(Q_{-i}) = max(0, (a - b*Q_{-i} - mc_i) / (2*b))
    pub fn best_response_linear(
        intercept: f64,
        slope: f64,
        mc: f64,
        others_total: f64,
    ) -> f64 {
        ((intercept - slope * others_total - mc) / (2.0 * slope)).max(0.0)
    }

    /// Analytical Nash equilibrium for N-player symmetric linear Cournot.
    /// q* = (a - mc) / (b * (N + 1))
    pub fn nash_symmetric_linear(
        intercept: f64,
        slope: f64,
        mc: f64,
        num_players: usize,
    ) -> MarketSimResult<f64> {
        let q = (intercept - mc) / (slope * (num_players as f64 + 1.0));
        if q < 0.0 {
            return Err(MarketSimError::EquilibriumNotFound(
                "Negative NE quantity (mc too high)".into(),
            ));
        }
        Ok(q)
    }

    /// Analytical Nash equilibrium for 2-player asymmetric linear Cournot.
    /// Given P = a - b*Q, C_i(q) = mc_i * q:
    ///   q_1* = (a - 2*mc_1 + mc_2) / (3*b)
    ///   q_2* = (a - 2*mc_2 + mc_1) / (3*b)
    pub fn nash_asymmetric_2p_linear(
        intercept: f64,
        slope: f64,
        mc1: f64,
        mc2: f64,
    ) -> MarketSimResult<(f64, f64)> {
        let q1 = (intercept - 2.0 * mc1 + mc2) / (3.0 * slope);
        let q2 = (intercept - 2.0 * mc2 + mc1) / (3.0 * slope);
        if q1 < 0.0 || q2 < 0.0 {
            return Err(MarketSimError::EquilibriumNotFound(
                "Negative NE quantity in asymmetric case".into(),
            ));
        }
        Ok((q1, q2))
    }

    /// Analytical Nash for N-player asymmetric linear Cournot.
    /// q_i* = (a - (N+1)*mc_i + Σ_j mc_j) / ((N+1)*b)
    ///      = (a + Σ_{j≠i} mc_j - N*mc_i) / ((N+1)*b)
    pub fn nash_asymmetric_linear(
        intercept: f64,
        slope: f64,
        marginal_costs: &[f64],
    ) -> MarketSimResult<Vec<f64>> {
        let n = marginal_costs.len();
        let mc_sum: f64 = marginal_costs.iter().sum();
        let denom = (n as f64 + 1.0) * slope;

        let mut quantities = Vec::with_capacity(n);
        for i in 0..n {
            let q = (intercept + mc_sum - (n as f64 + 1.0) * marginal_costs[i]) / denom;
            if q < 0.0 {
                return Err(MarketSimError::EquilibriumNotFound(format!(
                    "Negative NE quantity for player {i}"
                )));
            }
            quantities.push(q);
        }
        Ok(quantities)
    }

    /// Monopoly output for a single firm: q_m = (a - mc) / (2*b).
    pub fn monopoly_quantity_linear(intercept: f64, slope: f64, mc: f64) -> f64 {
        ((intercept - mc) / (2.0 * slope)).max(0.0)
    }

    /// Collusive (cartel) per-firm output for N symmetric firms.
    /// Total cartel output = (a - mc) / (2*b), each firm produces Q_cartel / N.
    pub fn collusive_quantity_symmetric(
        intercept: f64,
        slope: f64,
        mc: f64,
        num_players: usize,
    ) -> f64 {
        let total = Self::monopoly_quantity_linear(intercept, slope, mc);
        total / num_players as f64
    }

    /// NE market price for symmetric linear Cournot.
    pub fn nash_price_symmetric_linear(
        intercept: f64,
        slope: f64,
        mc: f64,
        num_players: usize,
    ) -> MarketSimResult<f64> {
        let q = Self::nash_symmetric_linear(intercept, slope, mc, num_players)?;
        let total_q = q * num_players as f64;
        Ok((intercept - slope * total_q).max(0.0))
    }

    /// Simulate one round of Cournot competition.
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
        let quantities: Vec<f64> = actions.iter().map(|a| a.value).collect();
        let total_q: f64 = quantities.iter().sum();
        let price = self.inverse_demand.price(total_q);
        let prices = vec![price; self.num_players];
        let profits = self.compute_profit(&quantities)?;
        Ok(MarketOutcome::new(round, prices, quantities, profits))
    }

    /// Compute Nash equilibrium numerically by iterated best response.
    pub fn compute_nash_iterative(
        &self,
        initial_quantities: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> MarketSimResult<Vec<f64>> {
        let n = self.num_players;
        let mut quantities = initial_quantities.to_vec();

        for _iter in 0..max_iter {
            let old_quantities = quantities.clone();
            for i in 0..n {
                let others: Vec<f64> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| quantities[j])
                    .collect();
                quantities[i] = self.best_response_quantity(i, &others)?;
            }
            let max_change: f64 = quantities
                .iter()
                .zip(old_quantities.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_change < tol {
                return Ok(quantities);
            }
        }
        Err(MarketSimError::EquilibriumNotFound(format!(
            "Iterated best response did not converge in {max_iter} iterations"
        )))
    }

    /// Check if given quantities are a Nash equilibrium.
    pub fn is_nash_equilibrium(&self, quantities: &[f64], tol: f64) -> MarketSimResult<bool> {
        let n = self.num_players;
        for i in 0..n {
            let others: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| quantities[j])
                .collect();
            let br = self.best_response_quantity(i, &others)?;
            if (br - quantities[i]).abs() > tol {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Reaction function: maps total opponent output to own best response.
    /// Returns a vector of (Q_{-i}, BR_i(Q_{-i})) pairs.
    pub fn reaction_function(
        &self,
        player: PlayerId,
        grid_points: usize,
    ) -> MarketSimResult<Vec<(f64, f64)>> {
        let max_q = self.inverse_demand.max_quantity();
        let max_others = if max_q.is_finite() { max_q } else { self.quantity_grid.max * (self.num_players as f64 - 1.0) };
        let step = max_others / grid_points as f64;

        let mut points = Vec::with_capacity(grid_points + 1);
        for k in 0..=grid_points {
            let others_total = k as f64 * step;
            // Create a dummy "other_quantities" by splitting equally
            let n_others = self.num_players - 1;
            let each = others_total / n_others as f64;
            let others = vec![each; n_others];
            let br = self.best_response_quantity(player, &others)?;
            points.push((others_total, br));
        }
        Ok(points)
    }

    /// Stability analysis: check if the Cournot equilibrium is stable under
    /// best-response dynamics (spectral radius of reaction function Jacobian < 1).
    /// For symmetric linear Cournot with N players, stability requires
    /// |∂BR_i/∂q_j| = b/(2b) = 1/2, and spectral radius = (N-1)/2.
    /// Stable iff N ≤ 2 for pure best-response, or always stable with dampening.
    pub fn stability_check_linear(
        _slope: f64,
        num_players: usize,
    ) -> StabilityResult {
        // ∂BR_i/∂Q_{-i} = -1/2 for linear Cournot
        // Jacobian of the reaction mapping has off-diagonal entries -1/(2)
        // Spectral radius for symmetric case = (N-1)/2
        let spectral_radius = (num_players as f64 - 1.0) / 2.0;
        StabilityResult {
            spectral_radius,
            is_stable: spectral_radius < 1.0,
            damping_needed: if spectral_radius >= 1.0 {
                Some(1.0 / spectral_radius)
            } else {
                None
            },
        }
    }

    /// Competitive prices and quantities.
    pub fn competitive_outcome(&self) -> MarketSimResult<(f64, Vec<f64>)> {
        // In competitive equilibrium, P = MC for each firm
        // For constant MC, all firms produce where P(Q) = MC
        // If asymmetric, the competitive price is the minimum MC
        // and firms with higher MC produce zero.
        let min_mc = (0..self.num_players)
            .map(|i| self.costs[i].marginal_cost(0.0))
            .fold(f64::INFINITY, f64::min);

        // Find total Q such that P(Q) = min_mc
        // For linear: Q = (a - min_mc) / b
        // Use binary search for general inverse demand
        let mut lo = 0.0;
        let mut hi = self.quantity_grid.max * self.num_players as f64;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            let p = self.inverse_demand.price(mid);
            if p > min_mc {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let total_q = (lo + hi) / 2.0;
        let price = self.inverse_demand.price(total_q);

        // Allocate production to lowest-cost firms
        let mut quantities = vec![0.0; self.num_players];
        let n_active: usize = (0..self.num_players)
            .filter(|&i| (self.costs[i].marginal_cost(0.0) - min_mc).abs() < 1e-10)
            .count();
        let each = if n_active > 0 { total_q / n_active as f64 } else { 0.0 };
        for i in 0..self.num_players {
            if (self.costs[i].marginal_cost(0.0) - min_mc).abs() < 1e-10 {
                quantities[i] = each;
            }
        }

        Ok((price, quantities))
    }

    /// Herfindahl-Hirschman Index (HHI) for given quantities.
    pub fn hhi(quantities: &[f64]) -> f64 {
        let total: f64 = quantities.iter().sum();
        if total.abs() < 1e-15 {
            return 0.0;
        }
        quantities
            .iter()
            .map(|q| {
                let share = q / total;
                share * share
            })
            .sum::<f64>()
            * 10000.0 // Standard HHI scaling
    }

    /// Profit landscape: profit for player `player` as own quantity varies.
    pub fn profit_landscape(
        &self,
        player: PlayerId,
        other_quantities: &[f64],
    ) -> MarketSimResult<Vec<(f64, f64)>> {
        if other_quantities.len() != self.num_players - 1 {
            return Err(MarketSimError::InvalidParameter(
                "other_quantities length mismatch".into(),
            ));
        }
        let others_total: f64 = other_quantities.iter().sum();
        let mut landscape = Vec::with_capacity(self.quantity_grid.num_points);
        for k in 0..self.quantity_grid.num_points {
            let q = self.quantity_grid.quantity_at(k);
            let total = others_total + q;
            let price = self.inverse_demand.price(total);
            let revenue = price * q;
            let cost = self.costs[player].total_cost(q);
            landscape.push((q, revenue - cost));
        }
        Ok(landscape)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// StabilityResult
// ════════════════════════════════════════════════════════════════════════════

/// Result of stability analysis for Cournot equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityResult {
    pub spectral_radius: f64,
    pub is_stable: bool,
    pub damping_needed: Option<f64>,
}

// ════════════════════════════════════════════════════════════════════════════
// Build from GameConfig
// ════════════════════════════════════════════════════════════════════════════

/// Build a CournotMarket from a GameConfig.
pub fn cournot_from_config(config: &GameConfig) -> MarketSimResult<CournotMarket> {
    let inv_demand = Box::new(LinearInverseDemand::new(
        config.demand_intercept,
        config.demand_slope,
    )?);
    let costs: Vec<Box<dyn CostFunction>> = config
        .marginal_costs
        .iter()
        .map(|&mc| -> Box<dyn CostFunction> {
            Box::new(ConstantMarginalCost { mc })
        })
        .collect();
    let grid = QuantityGrid::new(
        config.quantity_min,
        config.quantity_max,
        config.quantity_grid_size,
    )?;
    CournotMarket::new(inv_demand, costs, grid)
}

// ════════════════════════════════════════════════════════════════════════════
// Cournot analysis helpers
// ════════════════════════════════════════════════════════════════════════════

/// Markup in Cournot: (P - MC) / P = s_i / ε  where s_i is market share
/// and ε is market demand elasticity.
pub fn cournot_markup(price: f64, mc: f64) -> f64 {
    if price.abs() < 1e-15 { 0.0 } else { (price - mc) / price }
}

/// Total industry output at NE for N symmetric firms.
pub fn total_output_nash_symmetric(
    intercept: f64,
    slope: f64,
    mc: f64,
    num_players: usize,
) -> f64 {
    let n = num_players as f64;
    (n * (intercept - mc)) / (slope * (n + 1.0))
}

/// Total industry output at monopoly.
pub fn total_output_monopoly(intercept: f64, slope: f64, mc: f64) -> f64 {
    (intercept - mc) / (2.0 * slope)
}

/// Total industry output under perfect competition (P = MC).
pub fn total_output_competitive(intercept: f64, slope: f64, mc: f64) -> f64 {
    (intercept - mc) / slope
}

/// Cournot converges to competitive outcome as N → ∞.
/// Deadweight loss relative to competitive equilibrium.
pub fn deadweight_loss_cournot(
    intercept: f64,
    slope: f64,
    mc: f64,
    num_players: usize,
) -> f64 {
    let q_competitive = total_output_competitive(intercept, slope, mc);
    let q_cournot = total_output_nash_symmetric(intercept, slope, mc, num_players);
    let p_cournot = intercept - slope * q_cournot;
    // DWL = 0.5 * (P_cournot - MC) * (Q_competitive - Q_cournot)
    0.5 * (p_cournot - mc) * (q_competitive - q_cournot)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_2p() -> CournotMarket {
        CournotMarket::linear_with_constant_costs(
            10.0, 1.0, &[1.0, 1.0], 0.0, 10.0, 1001,
        ).unwrap()
    }

    #[test]
    fn test_linear_inverse_demand() {
        let d = LinearInverseDemand::new(10.0, 1.0).unwrap();
        assert!((d.price(3.0) - 7.0).abs() < 1e-10);
        assert!((d.price(10.0) - 0.0).abs() < 1e-10);
        assert!((d.price(15.0) - 0.0).abs() < 1e-10); // clamped
        assert!((d.max_quantity() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_inverse_demand_derivative() {
        let d = LinearInverseDemand::new(10.0, 1.0).unwrap();
        assert!((d.price_derivative(5.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_isoelastic_inverse_demand() {
        let d = IsoelasticInverseDemand::new(100.0, 2.0).unwrap();
        // P = 100 * Q^{-0.5}
        let p = d.price(4.0);
        assert!((p - 50.0).abs() < 1e-10); // 100 * 4^{-0.5} = 100/2 = 50
    }

    #[test]
    fn test_cournot_profit_symmetric() {
        let market = make_linear_2p();
        let profits = market.compute_profit(&[3.0, 3.0]).unwrap();
        // P = 10 - 1*(3+3) = 4, π_i = 4*3 - 1*3 = 9
        assert!((profits[0] - 9.0).abs() < 1e-10);
        assert!((profits[1] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_cournot_profit_asymmetric() {
        let market = make_linear_2p();
        let profits = market.compute_profit(&[2.0, 4.0]).unwrap();
        // P = 10 - 6 = 4, π_0 = 4*2 - 2 = 6, π_1 = 4*4 - 4 = 12
        assert!((profits[0] - 6.0).abs() < 1e-10);
        assert!((profits[1] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_nash_symmetric_2p() {
        let q = CournotMarket::nash_symmetric_linear(10.0, 1.0, 1.0, 2).unwrap();
        // q* = (10-1)/(1*3) = 3.0
        assert!((q - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nash_symmetric_3p() {
        let q = CournotMarket::nash_symmetric_linear(10.0, 1.0, 1.0, 3).unwrap();
        // q* = (10-1)/(1*4) = 2.25
        assert!((q - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_nash_asymmetric_2p() {
        let (q1, q2) = CournotMarket::nash_asymmetric_2p_linear(10.0, 1.0, 1.0, 2.0).unwrap();
        // q1 = (10 - 2 + 2) / 3 = 10/3 ≈ 3.333
        // q2 = (10 - 4 + 1) / 3 = 7/3 ≈ 2.333
        assert!((q1 - 10.0 / 3.0).abs() < 1e-10);
        assert!((q2 - 7.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nash_asymmetric_n_player() {
        let qs = CournotMarket::nash_asymmetric_linear(10.0, 1.0, &[1.0, 1.0, 1.0]).unwrap();
        // Same as symmetric 3-player: q* = 9/4 = 2.25
        for q in &qs {
            assert!((q - 2.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_monopoly_quantity() {
        let q = CournotMarket::monopoly_quantity_linear(10.0, 1.0, 1.0);
        // q_m = (10-1)/(2*1) = 4.5
        assert!((q - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_collusive_quantity() {
        let q = CournotMarket::collusive_quantity_symmetric(10.0, 1.0, 1.0, 2);
        // Total = 4.5, per firm = 2.25
        assert!((q - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_quantity_ordering() {
        // Collusive < Nash < Competitive per firm
        let q_collusive = CournotMarket::collusive_quantity_symmetric(10.0, 1.0, 1.0, 2);
        let q_nash = CournotMarket::nash_symmetric_linear(10.0, 1.0, 1.0, 2).unwrap();
        let q_competitive = total_output_competitive(10.0, 1.0, 1.0) / 2.0;
        assert!(q_collusive < q_nash);
        assert!(q_nash < q_competitive);
    }

    #[test]
    fn test_best_response_analytical() {
        let br = CournotMarket::best_response_linear(10.0, 1.0, 1.0, 3.0);
        // BR(Q_{-i}=3) = (10 - 3 - 1) / 2 = 3.0
        assert!((br - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_best_response_numerical() {
        let market = make_linear_2p();
        let br = market.best_response_quantity(0, &[3.0]).unwrap();
        let analytical = CournotMarket::best_response_linear(10.0, 1.0, 1.0, 3.0);
        assert!((br - analytical).abs() < 0.02);
    }

    #[test]
    fn test_simulate_round() {
        let market = make_linear_2p();
        let actions = vec![
            PlayerAction::new(0, 3.0),
            PlayerAction::new(1, 3.0),
        ];
        let outcome = market.simulate_round(&actions, 0).unwrap();
        assert_eq!(outcome.round, 0);
        // All players see the same price in Cournot
        assert!((outcome.prices[0] - outcome.prices[1]).abs() < 1e-10);
        assert!((outcome.prices[0] - 4.0).abs() < 1e-10); // P = 10-6 = 4
    }

    #[test]
    fn test_stability_2p() {
        let result = CournotMarket::stability_check_linear(1.0, 2);
        assert!(result.is_stable); // spectral radius = 1/2 < 1
        assert!((result.spectral_radius - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stability_3p() {
        let result = CournotMarket::stability_check_linear(1.0, 3);
        assert!(!result.is_stable); // spectral radius = 1.0, not < 1
    }

    #[test]
    fn test_hhi() {
        let hhi = CournotMarket::hhi(&[3.0, 3.0]);
        // Each share = 0.5, HHI = 2 * 0.25 * 10000 = 5000
        assert!((hhi - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn test_hhi_monopoly() {
        let hhi = CournotMarket::hhi(&[10.0]);
        assert!((hhi - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_deadweight_loss() {
        let dwl = deadweight_loss_cournot(10.0, 1.0, 1.0, 2);
        assert!(dwl > 0.0);
        // More players → less DWL
        let dwl3 = deadweight_loss_cournot(10.0, 1.0, 1.0, 3);
        assert!(dwl3 < dwl);
    }

    #[test]
    fn test_cournot_from_config() {
        let mut config = GameConfig::default();
        config.market_type = MarketType::Cournot;
        let market = cournot_from_config(&config).unwrap();
        assert_eq!(market.num_players, 2);
    }

    #[test]
    fn test_reaction_function() {
        let market = make_linear_2p();
        let rf = market.reaction_function(0, 20).unwrap();
        assert!(!rf.is_empty());
        // BR should be decreasing
        assert!(rf[0].1 >= rf.last().unwrap().1);
    }

    #[test]
    fn test_profit_landscape() {
        let market = make_linear_2p();
        let landscape = market.profit_landscape(0, &[3.0]).unwrap();
        assert_eq!(landscape.len(), 1001);
        // Should have a maximum
        let max_profit = landscape.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        assert!(max_profit > 0.0);
    }

    #[test]
    fn test_wrong_quantities_count() {
        let market = make_linear_2p();
        assert!(market.compute_profit(&[1.0]).is_err());
    }

    #[test]
    fn test_quantity_grid() {
        let g = QuantityGrid::new(0.0, 10.0, 101).unwrap();
        assert!((g.step - 0.1).abs() < 1e-10);
        assert!((g.quantity_at(0) - 0.0).abs() < 1e-10);
        assert!((g.quantity_at(100) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_nash_price_symmetric() {
        let p = CournotMarket::nash_price_symmetric_linear(10.0, 1.0, 1.0, 2).unwrap();
        // Total Q = 2*3 = 6, P = 10-6 = 4
        assert!((p - 4.0).abs() < 1e-10);
    }
}
