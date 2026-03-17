//! Demand system implementations for market simulation.
//!
//! Provides [`DemandFunction`] trait and concrete implementations:
//! - [`LinearDemand`]: Q_i = a - b*p_i + c*Σ(p_j, j≠i)
//! - [`CESDemand`]: Constant Elasticity of Substitution demand
//! - [`LogitDemand`]: Multinomial logit with outside option
//!
//! All demand systems support 2–4 players, interval-arithmetic variants,
//! and Lipschitz-constant computation for verification purposes.

use crate::types::{Interval, PlayerId};
use crate::{MarketSimError, MarketSimResult};
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// Trait
// ════════════════════════════════════════════════════════════════════════════

/// Core trait for any demand system.
pub trait DemandFunction: Send + Sync {
    /// Number of players (products) in the market.
    fn num_players(&self) -> usize;

    /// Compute quantity demanded for each player given a price vector.
    fn compute_demand(&self, prices: &[f64]) -> MarketSimResult<Vec<f64>>;

    /// Revenue for each player: R_i = p_i * Q_i(p).
    fn compute_revenue(&self, prices: &[f64]) -> MarketSimResult<Vec<f64>> {
        let q = self.compute_demand(prices)?;
        Ok(prices.iter().zip(q.iter()).map(|(p, q)| p * q).collect())
    }

    /// Own-price elasticity ε_i = (∂Q_i/∂p_i) * (p_i / Q_i).
    fn elasticity(&self, prices: &[f64], player: PlayerId) -> MarketSimResult<f64> {
        let jac = self.jacobian(prices)?;
        let q = self.compute_demand(prices)?;
        if q[player].abs() < 1e-15 {
            return Err(MarketSimError::DemandError(
                "Zero demand – elasticity undefined".into(),
            ));
        }
        Ok(jac[player][player] * prices[player] / q[player])
    }

    /// Full Jacobian matrix: J[i][j] = ∂Q_i / ∂p_j.
    fn jacobian(&self, prices: &[f64]) -> MarketSimResult<Vec<Vec<f64>>>;

    /// Validate that demand is non-negative at the given prices.
    fn validate_non_negative(&self, prices: &[f64]) -> MarketSimResult<()> {
        let q = self.compute_demand(prices)?;
        for (i, &qi) in q.iter().enumerate() {
            if qi < -1e-12 {
                return Err(MarketSimError::DemandError(format!(
                    "Negative demand for player {i}: {qi}"
                )));
            }
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// LinearDemand
// ════════════════════════════════════════════════════════════════════════════

/// Linear differentiated-products demand: Q_i = a - b*p_i + c*Σ_{j≠i} p_j.
///
/// Parameters:
/// - `a`: demand intercept (baseline demand at zero prices)
/// - `b`: own-price sensitivity (> 0)
/// - `c`: cross-price sensitivity (substitution), 0 ≤ c < b for stability
/// - `num_players`: number of products / firms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearDemand {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub num_players: usize,
}

impl LinearDemand {
    pub fn new(a: f64, b: f64, c: f64, num_players: usize) -> MarketSimResult<Self> {
        if a <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Demand intercept `a` must be positive".into(),
            ));
        }
        if b <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Own-price slope `b` must be positive".into(),
            ));
        }
        if c < 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "Cross-price slope `c` must be non-negative".into(),
            ));
        }
        if num_players < 2 || num_players > 4 {
            return Err(MarketSimError::InvalidParameter(
                "num_players must be in [2, 4]".into(),
            ));
        }
        Ok(Self { a, b, c, num_players })
    }

    /// Maximum price at which demand is non-negative (for a single player,
    /// holding others at zero).
    pub fn choke_price(&self) -> f64 {
        self.a / self.b
    }

    /// Inverse demand: given quantities, compute market-clearing prices.
    /// Solves Q_i = a - b*p_i + c*Σ_{j≠i} p_j for p.
    pub fn inverse_demand(&self, quantities: &[f64]) -> MarketSimResult<Vec<f64>> {
        let n = self.num_players;
        if quantities.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "quantities length mismatch".into(),
            ));
        }
        // System: Q = a*1 - B*p  where B = b*I - c*(J-I)
        // B[i][i] = b, B[i][j] = -c  => p = B^{-1} * (a*1 - Q)
        // For symmetric B: B^{-1}[i][i] = (b + (n-2)*c) / det_factor
        //                  B^{-1}[i][j] = c / det_factor
        // where det_factor = (b - c) * (b + (n-1)*c)
        let det = (self.b - self.c) * (self.b + (n as f64 - 1.0) * self.c);
        if det.abs() < 1e-15 {
            return Err(MarketSimError::DemandError(
                "Singular demand matrix".into(),
            ));
        }
        let rhs: Vec<f64> = quantities.iter().map(|q| self.a - q).collect();
        let diag_inv = (self.b + (n as f64 - 2.0) * self.c) / det;
        let off_inv = self.c / det;
        let mut prices = vec![0.0; n];
        for i in 0..n {
            let mut sum = diag_inv * rhs[i];
            for j in 0..n {
                if j != i {
                    sum += off_inv * rhs[j];
                }
            }
            prices[i] = sum;
        }
        Ok(prices)
    }

    /// Compute demand using interval arithmetic for bound propagation.
    pub fn compute_demand_interval(&self, prices: &[Interval]) -> MarketSimResult<Vec<Interval>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let a_iv = Interval::point(self.a);
            let own = Interval::point(self.b) * prices[i];
            let mut cross_sum = Interval::point(0.0);
            for j in 0..n {
                if j != i {
                    cross_sum = cross_sum + prices[j];
                }
            }
            let cross = Interval::point(self.c) * cross_sum;
            let q = a_iv - own + cross;
            // Clamp lower bound to zero
            let q_clamped = Interval::new(q.lo.max(0.0), q.hi.max(0.0));
            result.push(q_clamped);
        }
        Ok(result)
    }
}

impl DemandFunction for LinearDemand {
    fn num_players(&self) -> usize {
        self.num_players
    }

    fn compute_demand(&self, prices: &[f64]) -> MarketSimResult<Vec<f64>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(format!(
                "Expected {n} prices, got {}",
                prices.len()
            )));
        }
        let mut quantities = Vec::with_capacity(n);
        for i in 0..n {
            let cross_sum: f64 = prices.iter().enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| *p)
                .sum();
            let q = self.a - self.b * prices[i] + self.c * cross_sum;
            quantities.push(q.max(0.0));
        }
        Ok(quantities)
    }

    fn jacobian(&self, prices: &[f64]) -> MarketSimResult<Vec<Vec<f64>>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        // Check whether demand is positive (Jacobian is only valid in the interior)
        let q = self.compute_demand(prices)?;
        let mut jac = vec![vec![0.0; n]; n];
        for i in 0..n {
            if q[i] > 0.0 {
                for j in 0..n {
                    jac[i][j] = if i == j { -self.b } else { self.c };
                }
            }
            // If q[i] == 0 the gradient is zero (demand clamped at zero).
        }
        Ok(jac)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CESDemand
// ════════════════════════════════════════════════════════════════════════════

/// Constant Elasticity of Substitution demand system.
///
/// Market share: s_i = p_i^{1-σ} / Σ_j p_j^{1-σ}
/// Quantity:     Q_i = M * s_i / p_i  (Marshallian demand)
///
/// Parameters:
/// - `sigma`: elasticity of substitution (σ > 1 for substitutes)
/// - `market_size`: total expenditure M
/// - `quality`: product quality index a_i (shifts demand)
/// - `num_players`: 2–4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CESDemand {
    pub sigma: f64,
    pub market_size: f64,
    pub quality: Vec<f64>,
    pub num_players: usize,
}

impl CESDemand {
    pub fn new(
        sigma: f64,
        market_size: f64,
        quality: Vec<f64>,
        num_players: usize,
    ) -> MarketSimResult<Self> {
        if sigma <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "sigma must be positive".into(),
            ));
        }
        if market_size <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "market_size must be positive".into(),
            ));
        }
        if quality.len() != num_players {
            return Err(MarketSimError::InvalidParameter(
                "quality vector length must equal num_players".into(),
            ));
        }
        if num_players < 2 || num_players > 4 {
            return Err(MarketSimError::InvalidParameter(
                "num_players must be in [2, 4]".into(),
            ));
        }
        for &q in &quality {
            if q <= 0.0 {
                return Err(MarketSimError::InvalidParameter(
                    "quality must be positive".into(),
                ));
            }
        }
        Ok(Self { sigma, market_size, quality, num_players })
    }

    /// Symmetric CES demand with equal quality across players.
    pub fn symmetric(sigma: f64, market_size: f64, num_players: usize) -> MarketSimResult<Self> {
        Self::new(sigma, market_size, vec![1.0; num_players], num_players)
    }

    /// Market share of player i.
    pub fn market_share(&self, prices: &[f64], player: PlayerId) -> MarketSimResult<f64> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        let exp = 1.0 - self.sigma;
        let numerator = self.quality[player] * prices[player].powf(exp);
        let denominator: f64 = (0..n)
            .map(|j| self.quality[j] * prices[j].powf(exp))
            .sum();
        if denominator.abs() < 1e-15 {
            return Err(MarketSimError::DemandError(
                "CES denominator is zero".into(),
            ));
        }
        Ok(numerator / denominator)
    }

    /// Compute demand using interval arithmetic.
    pub fn compute_demand_interval(&self, prices: &[Interval]) -> MarketSimResult<Vec<Interval>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        let exp = 1.0 - self.sigma;
        let neg_sigma = -self.sigma;

        // Compute p_j^{1-sigma} for each j using interval bounds
        let mut share_terms: Vec<Interval> = Vec::with_capacity(n);
        for j in 0..n {
            let lo_pow = self.quality[j] * prices[j].lo.powf(exp);
            let hi_pow = self.quality[j] * prices[j].hi.powf(exp);
            let (lo, hi) = if exp >= 0.0 {
                (lo_pow.min(hi_pow), lo_pow.max(hi_pow))
            } else {
                (lo_pow.min(hi_pow), lo_pow.max(hi_pow))
            };
            share_terms.push(Interval::new(lo.min(hi), lo.max(hi)));
        }

        let denom = share_terms.iter().copied().fold(
            Interval::point(0.0),
            |acc, x| acc + x,
        );

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            // Q_i = M * a_i * p_i^{-sigma} / denom
            let lo_pow = self.quality[i] * prices[i].lo.powf(neg_sigma);
            let hi_pow = self.quality[i] * prices[i].hi.powf(neg_sigma);
            let num = Interval::new(lo_pow.min(hi_pow), lo_pow.max(hi_pow));
            let q = Interval::point(self.market_size) * num
                / Interval::new(denom.lo.max(1e-15), denom.hi.max(1e-15));
            result.push(Interval::new(q.lo.max(0.0), q.hi.max(0.0)));
        }
        Ok(result)
    }
}

impl DemandFunction for CESDemand {
    fn num_players(&self) -> usize {
        self.num_players
    }

    fn compute_demand(&self, prices: &[f64]) -> MarketSimResult<Vec<f64>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(format!(
                "Expected {n} prices, got {}",
                prices.len()
            )));
        }
        for &p in prices {
            if p <= 0.0 {
                return Err(MarketSimError::DemandError(
                    "CES demand requires strictly positive prices".into(),
                ));
            }
        }
        let exp = 1.0 - self.sigma;
        let denom: f64 = (0..n)
            .map(|j| self.quality[j] * prices[j].powf(exp))
            .sum();
        if denom.abs() < 1e-15 {
            return Err(MarketSimError::DemandError(
                "CES denominator is zero".into(),
            ));
        }
        let mut quantities = Vec::with_capacity(n);
        for i in 0..n {
            let q = self.market_size * self.quality[i] * prices[i].powf(-self.sigma) / denom;
            quantities.push(q.max(0.0));
        }
        Ok(quantities)
    }

    fn jacobian(&self, prices: &[f64]) -> MarketSimResult<Vec<Vec<f64>>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        let q = self.compute_demand(prices)?;
        let shares: Vec<f64> = (0..n)
            .map(|i| self.market_share(prices, i).unwrap_or(0.0))
            .collect();

        let mut jac = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // ∂Q_i/∂p_i = Q_i / p_i * (-σ + (σ-1)*s_i)
                    jac[i][j] = q[i] / prices[i] * (-self.sigma + (self.sigma - 1.0) * shares[i]);
                } else {
                    // ∂Q_i/∂p_j = Q_i * (σ-1) * s_j / p_j
                    jac[i][j] = q[i] * (self.sigma - 1.0) * shares[j] / prices[j];
                }
            }
        }
        Ok(jac)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// LogitDemand
// ════════════════════════════════════════════════════════════════════════════

/// Multinomial logit demand with outside option.
///
/// Utility:      V_i = a_i - α*p_i
/// Market share: s_i = exp(V_i) / (exp(V_0) + Σ_j exp(V_j))
/// Quantity:     Q_i = N * s_i
///
/// Parameters:
/// - `quality`: quality index a_i for each product
/// - `price_sensitivity`: α > 0
/// - `outside_value`: utility of outside option V_0
/// - `market_size`: total number of consumers N
/// - `num_players`: 2–4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitDemand {
    pub quality: Vec<f64>,
    pub price_sensitivity: f64,
    pub outside_value: f64,
    pub market_size: f64,
    pub num_players: usize,
}

impl LogitDemand {
    pub fn new(
        quality: Vec<f64>,
        price_sensitivity: f64,
        outside_value: f64,
        market_size: f64,
        num_players: usize,
    ) -> MarketSimResult<Self> {
        if price_sensitivity <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "price_sensitivity must be positive".into(),
            ));
        }
        if market_size <= 0.0 {
            return Err(MarketSimError::InvalidParameter(
                "market_size must be positive".into(),
            ));
        }
        if quality.len() != num_players {
            return Err(MarketSimError::InvalidParameter(
                "quality vector length must equal num_players".into(),
            ));
        }
        if num_players < 2 || num_players > 4 {
            return Err(MarketSimError::InvalidParameter(
                "num_players must be in [2, 4]".into(),
            ));
        }
        Ok(Self { quality, price_sensitivity, outside_value, market_size, num_players })
    }

    /// Symmetric logit demand with equal quality across players.
    pub fn symmetric(
        quality: f64,
        price_sensitivity: f64,
        outside_value: f64,
        market_size: f64,
        num_players: usize,
    ) -> MarketSimResult<Self> {
        Self::new(
            vec![quality; num_players],
            price_sensitivity,
            outside_value,
            market_size,
            num_players,
        )
    }

    /// Compute indirect utility for player i at price p_i.
    fn utility(&self, player: PlayerId, price: f64) -> f64 {
        self.quality[player] - self.price_sensitivity * price
    }

    /// Market share of player i (softmax).
    pub fn market_share(&self, prices: &[f64], player: PlayerId) -> MarketSimResult<f64> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        // Use log-sum-exp for numerical stability
        let utilities: Vec<f64> = (0..n).map(|j| self.utility(j, prices[j])).collect();
        let max_u = utilities
            .iter()
            .copied()
            .chain(std::iter::once(self.outside_value))
            .fold(f64::NEG_INFINITY, f64::max);

        let denom: f64 = (self.outside_value - max_u).exp()
            + utilities.iter().map(|u| (u - max_u).exp()).sum::<f64>();

        Ok((utilities[player] - max_u).exp() / denom)
    }

    /// Share of consumers choosing the outside option.
    pub fn outside_share(&self, prices: &[f64]) -> MarketSimResult<f64> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        let utilities: Vec<f64> = (0..n).map(|j| self.utility(j, prices[j])).collect();
        let max_u = utilities
            .iter()
            .copied()
            .chain(std::iter::once(self.outside_value))
            .fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = (self.outside_value - max_u).exp()
            + utilities.iter().map(|u| (u - max_u).exp()).sum::<f64>();
        Ok((self.outside_value - max_u).exp() / denom)
    }

    /// Compute demand using interval arithmetic.
    pub fn compute_demand_interval(&self, prices: &[Interval]) -> MarketSimResult<Vec<Interval>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        // For each player, compute bounds on exp(V_i)
        let mut exp_bounds: Vec<Interval> = Vec::with_capacity(n);
        for i in 0..n {
            let v_lo = self.quality[i] - self.price_sensitivity * prices[i].hi;
            let v_hi = self.quality[i] - self.price_sensitivity * prices[i].lo;
            exp_bounds.push(Interval::new(v_lo.exp(), v_hi.exp()));
        }
        let exp_outside = Interval::point(self.outside_value.exp());

        // Denominator bounds: sum of all exp terms
        let denom = exp_bounds.iter().copied().fold(exp_outside, |acc, x| acc + x);

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            // s_i ∈ [exp_lo / denom_hi, exp_hi / denom_lo]
            let s_lo = exp_bounds[i].lo / denom.hi.max(1e-15);
            let s_hi = exp_bounds[i].hi / denom.lo.max(1e-15);
            let q = Interval::new(
                (self.market_size * s_lo).max(0.0),
                (self.market_size * s_hi).max(0.0),
            );
            result.push(q);
        }
        Ok(result)
    }
}

impl DemandFunction for LogitDemand {
    fn num_players(&self) -> usize {
        self.num_players
    }

    fn compute_demand(&self, prices: &[f64]) -> MarketSimResult<Vec<f64>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(format!(
                "Expected {n} prices, got {}",
                prices.len()
            )));
        }
        let mut quantities = Vec::with_capacity(n);
        for i in 0..n {
            let share = self.market_share(prices, i)?;
            quantities.push(self.market_size * share);
        }
        Ok(quantities)
    }

    fn jacobian(&self, prices: &[f64]) -> MarketSimResult<Vec<Vec<f64>>> {
        let n = self.num_players;
        if prices.len() != n {
            return Err(MarketSimError::InvalidParameter(
                "prices length mismatch".into(),
            ));
        }
        let shares: Vec<f64> = (0..n)
            .map(|i| self.market_share(prices, i).unwrap_or(0.0))
            .collect();
        let alpha = self.price_sensitivity;
        let big_n = self.market_size;

        let mut jac = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // ∂Q_i/∂p_i = -α * N * s_i * (1 - s_i)
                    jac[i][j] = -alpha * big_n * shares[i] * (1.0 - shares[i]);
                } else {
                    // ∂Q_i/∂p_j = α * N * s_i * s_j
                    jac[i][j] = alpha * big_n * shares[i] * shares[j];
                }
            }
        }
        Ok(jac)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// LipschitzBound
// ════════════════════════════════════════════════════════════════════════════

/// Lipschitz constant bound for a demand function over a price domain.
///
/// If L is the Lipschitz constant, then for any prices p, p':
///   ||Q(p) - Q(p')|| ≤ L * ||p - p'||
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipschitzBound {
    pub constant: f64,
    pub domain_lo: Vec<f64>,
    pub domain_hi: Vec<f64>,
    pub norm: LipschitzNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LipschitzNorm {
    L1,
    L2,
    LInf,
}

impl LipschitzBound {
    /// Compute the Lipschitz bound for a linear demand system (exact).
    pub fn for_linear(demand: &LinearDemand, _lo: &[f64], _hi: &[f64]) -> Self {
        let n = demand.num_players;
        // The Jacobian is constant: ||J||_inf = max_i Σ_j |J[i][j]|
        //   = b + (n-1)*c
        let linf_bound = demand.b + (n as f64 - 1.0) * demand.c;
        // L2 bound via spectral norm: max singular value
        // For the linear Jacobian B = b*I - c*(J-I):
        //   eigenvalues are b - (n-1)*c (once) and b + c (n-1 times)
        let eig1 = (demand.b - (n as f64 - 1.0) * demand.c).abs();
        let eig2 = (demand.b + demand.c).abs();
        let l2_bound = eig1.max(eig2);
        let _ = l2_bound; // available if needed

        Self {
            constant: linf_bound,
            domain_lo: vec![0.0; n],
            domain_hi: vec![demand.choke_price(); n],
            norm: LipschitzNorm::LInf,
        }
    }

    /// Compute the Lipschitz bound for a CES demand via sampling on a grid.
    pub fn for_ces(demand: &CESDemand, lo: &[f64], hi: &[f64], grid_points: usize) -> Self {
        let n = demand.num_players;
        let mut max_norm = 0.0_f64;

        // Sample the Jacobian at grid points and take the max spectral norm
        let _step: Vec<f64> = (0..n).map(|i| (hi[i] - lo[i]) / grid_points as f64).collect();

        // For efficiency, sample along the diagonal and edges
        for k in 0..=grid_points {
            let frac = k as f64 / grid_points as f64;
            let prices: Vec<f64> = (0..n)
                .map(|i| lo[i] + frac * (hi[i] - lo[i]))
                .collect();

            if prices.iter().all(|&p| p > 0.0) {
                if let Ok(jac) = demand.jacobian(&prices) {
                    let norm = matrix_inf_norm(&jac);
                    max_norm = max_norm.max(norm);
                }
            }
        }

        // Add safety margin for points not sampled
        let safety = 1.1;
        Self {
            constant: max_norm * safety,
            domain_lo: lo.to_vec(),
            domain_hi: hi.to_vec(),
            norm: LipschitzNorm::LInf,
        }
    }

    /// Compute the Lipschitz bound for logit demand via sampling.
    pub fn for_logit(demand: &LogitDemand, lo: &[f64], hi: &[f64], grid_points: usize) -> Self {
        let n = demand.num_players;
        let mut max_norm = 0.0_f64;

        for k in 0..=grid_points {
            let frac = k as f64 / grid_points as f64;
            let prices: Vec<f64> = (0..n)
                .map(|i| lo[i] + frac * (hi[i] - lo[i]))
                .collect();

            if let Ok(jac) = demand.jacobian(&prices) {
                let norm = matrix_inf_norm(&jac);
                max_norm = max_norm.max(norm);
            }
        }

        // Logit Jacobian: max entry bounded by α*N/4
        // (since s*(1-s) ≤ 1/4), so ||J||_inf ≤ α*N*(1/4 + (n-1)/4) = α*N*n/4
        let analytic_bound = demand.price_sensitivity * demand.market_size * n as f64 / 4.0;
        let bound = max_norm.max(analytic_bound);

        Self {
            constant: bound,
            domain_lo: lo.to_vec(),
            domain_hi: hi.to_vec(),
            norm: LipschitzNorm::LInf,
        }
    }

    /// Check if this Lipschitz bound is valid (constant is finite and positive).
    pub fn is_valid(&self) -> bool {
        self.constant.is_finite() && self.constant >= 0.0
    }

    /// Maximum demand change for a given price perturbation.
    pub fn max_demand_change(&self, price_perturbation: f64) -> f64 {
        self.constant * price_perturbation
    }
}

/// Infinity norm of a matrix: max row sum of absolute values.
fn matrix_inf_norm(mat: &[Vec<f64>]) -> f64 {
    mat.iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// Frobenius norm of a matrix.
#[allow(dead_code)]
fn matrix_frobenius_norm(mat: &[Vec<f64>]) -> f64 {
    mat.iter()
        .flat_map(|row| row.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

// ════════════════════════════════════════════════════════════════════════════
// Demand validation utilities
// ════════════════════════════════════════════════════════════════════════════

/// Validate demand parameters for a given demand system type.
pub fn validate_demand_params(
    a: f64,
    b: f64,
    c: f64,
    num_players: usize,
) -> MarketSimResult<()> {
    if a <= 0.0 {
        return Err(MarketSimError::InvalidParameter(
            "intercept must be positive".into(),
        ));
    }
    if b <= 0.0 {
        return Err(MarketSimError::InvalidParameter(
            "own-price slope must be positive".into(),
        ));
    }
    if c < 0.0 || c >= b {
        return Err(MarketSimError::InvalidParameter(
            "cross-price slope must be in [0, b)".into(),
        ));
    }
    if num_players < 2 || num_players > 4 {
        return Err(MarketSimError::InvalidParameter(
            "num_players must be in [2, 4]".into(),
        ));
    }
    Ok(())
}

/// Check that demand satisfies the gross-substitutes property.
/// For linear demand, this means c > 0 (cross-price effect is positive).
pub fn check_gross_substitutes(demand: &dyn DemandFunction, prices: &[f64]) -> MarketSimResult<bool> {
    let jac = demand.jacobian(prices)?;
    let n = demand.num_players();
    for i in 0..n {
        for j in 0..n {
            if i != j && jac[i][j] < -1e-12 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Check that own-price effects dominate cross-price effects (diagonal dominance).
pub fn check_diagonal_dominance(demand: &dyn DemandFunction, prices: &[f64]) -> MarketSimResult<bool> {
    let jac = demand.jacobian(prices)?;
    let n = demand.num_players();
    for i in 0..n {
        let diag = jac[i][i].abs();
        let off_diag_sum: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| jac[i][j].abs())
            .sum();
        if diag < off_diag_sum {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Create a demand function from a GameConfig.
pub fn create_demand_from_config(config: &crate::types::GameConfig) -> MarketSimResult<Box<dyn DemandFunction>> {
    use crate::types::DemandSystemType;
    match config.demand_system {
        DemandSystemType::Linear => {
            let d = LinearDemand::new(
                config.demand_intercept,
                config.demand_slope,
                config.demand_cross_slope,
                config.num_players,
            )?;
            Ok(Box::new(d))
        }
        DemandSystemType::CES => {
            let d = CESDemand::symmetric(
                config.substitution_elasticity,
                config.market_size,
                config.num_players,
            )?;
            Ok(Box::new(d))
        }
        DemandSystemType::Logit => {
            let d = LogitDemand::symmetric(
                1.0,
                config.price_sensitivity,
                config.outside_option_value,
                config.market_size,
                config.num_players,
            )?;
            Ok(Box::new(d))
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_2p() -> LinearDemand {
        LinearDemand::new(10.0, 2.0, 0.5, 2).unwrap()
    }

    fn ces_2p() -> CESDemand {
        CESDemand::symmetric(2.0, 100.0, 2).unwrap()
    }

    fn logit_2p() -> LogitDemand {
        LogitDemand::symmetric(1.0, 0.5, 0.0, 100.0, 2).unwrap()
    }

    #[test]
    fn test_linear_demand_basic() {
        let d = linear_2p();
        let q = d.compute_demand(&[1.0, 1.0]).unwrap();
        // Q_i = 10 - 2*1 + 0.5*1 = 8.5
        assert!((q[0] - 8.5).abs() < 1e-10);
        assert!((q[1] - 8.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_demand_asymmetric() {
        let d = linear_2p();
        let q = d.compute_demand(&[2.0, 3.0]).unwrap();
        // Q_0 = 10 - 2*2 + 0.5*3 = 7.5
        // Q_1 = 10 - 2*3 + 0.5*2 = 5.0
        assert!((q[0] - 7.5).abs() < 1e-10);
        assert!((q[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_demand_clamps_at_zero() {
        let d = linear_2p();
        let q = d.compute_demand(&[100.0, 1.0]).unwrap();
        assert!(q[0] >= 0.0);
    }

    #[test]
    fn test_linear_choke_price() {
        let d = linear_2p();
        assert!((d.choke_price() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_jacobian() {
        let d = linear_2p();
        let jac = d.jacobian(&[1.0, 1.0]).unwrap();
        assert!((jac[0][0] - (-2.0)).abs() < 1e-10);
        assert!((jac[0][1] - 0.5).abs() < 1e-10);
        assert!((jac[1][0] - 0.5).abs() < 1e-10);
        assert!((jac[1][1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_linear_revenue() {
        let d = linear_2p();
        let rev = d.compute_revenue(&[2.0, 2.0]).unwrap();
        // Q = 10 - 2*2 + 0.5*2 = 7.0,  R = 2 * 7 = 14
        assert!((rev[0] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_elasticity() {
        let d = linear_2p();
        let e = d.elasticity(&[2.0, 2.0], 0).unwrap();
        // ε = -b * p / Q = -2 * 2 / 7 ≈ -0.5714
        assert!((e - (-4.0 / 7.0)).abs() < 1e-10);
    }

    #[test]
    fn test_linear_inverse_demand() {
        let d = linear_2p();
        let prices = vec![3.0, 4.0];
        let q = d.compute_demand(&prices).unwrap();
        let p_recovered = d.inverse_demand(&q).unwrap();
        for i in 0..2 {
            assert!((p_recovered[i] - prices[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_linear_3_players() {
        let d = LinearDemand::new(10.0, 2.0, 0.5, 3).unwrap();
        let q = d.compute_demand(&[1.0, 1.0, 1.0]).unwrap();
        // Q_i = 10 - 2*1 + 0.5*(1+1) = 9.0
        assert!((q[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interval_demand() {
        let d = linear_2p();
        let prices = vec![Interval::new(0.9, 1.1), Interval::new(0.9, 1.1)];
        let q = d.compute_demand_interval(&prices).unwrap();
        // Point demand at (1,1) = 8.5, interval should contain it
        assert!(q[0].contains(8.5));
    }

    #[test]
    fn test_ces_demand_symmetric() {
        let d = ces_2p();
        let q = d.compute_demand(&[1.0, 1.0]).unwrap();
        // Symmetric: each gets half the market
        assert!((q[0] - q[1]).abs() < 1e-10);
        assert!((q[0] - 50.0).abs() < 1e-10); // M/2 = 50
    }

    #[test]
    fn test_ces_market_share() {
        let d = ces_2p();
        let s0 = d.market_share(&[1.0, 1.0], 0).unwrap();
        let s1 = d.market_share(&[1.0, 1.0], 1).unwrap();
        assert!((s0 - 0.5).abs() < 1e-10);
        assert!((s0 + s1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ces_higher_price_less_share() {
        let d = ces_2p();
        let s0 = d.market_share(&[2.0, 1.0], 0).unwrap();
        let s1 = d.market_share(&[2.0, 1.0], 1).unwrap();
        assert!(s0 < s1);
    }

    #[test]
    fn test_ces_jacobian_diagonal_negative() {
        let d = ces_2p();
        let jac = d.jacobian(&[2.0, 2.0]).unwrap();
        assert!(jac[0][0] < 0.0); // own-price effect negative
        assert!(jac[0][1] > 0.0); // cross-price effect positive (substitutes)
    }

    #[test]
    fn test_logit_demand_symmetric() {
        let d = logit_2p();
        let q = d.compute_demand(&[1.0, 1.0]).unwrap();
        assert!((q[0] - q[1]).abs() < 1e-10);
    }

    #[test]
    fn test_logit_shares_sum_to_one() {
        let d = logit_2p();
        let prices = &[2.0, 3.0];
        let s0 = d.market_share(prices, 0).unwrap();
        let s1 = d.market_share(prices, 1).unwrap();
        let s_out = d.outside_share(prices).unwrap();
        assert!((s0 + s1 + s_out - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_logit_higher_price_less_demand() {
        let d = logit_2p();
        let q = d.compute_demand(&[1.0, 2.0]).unwrap();
        assert!(q[0] > q[1]);
    }

    #[test]
    fn test_logit_jacobian() {
        let d = logit_2p();
        let jac = d.jacobian(&[1.0, 1.0]).unwrap();
        assert!(jac[0][0] < 0.0); // own-price negative
        assert!(jac[0][1] > 0.0); // cross-price positive
    }

    #[test]
    fn test_lipschitz_linear() {
        let d = linear_2p();
        let lip = LipschitzBound::for_linear(&d, &[0.0, 0.0], &[5.0, 5.0]);
        assert!(lip.is_valid());
        // L_inf = b + (n-1)*c = 2 + 0.5 = 2.5
        assert!((lip.constant - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_lipschitz_ces() {
        let d = ces_2p();
        let lip = LipschitzBound::for_ces(&d, &[0.5, 0.5], &[5.0, 5.0], 50);
        assert!(lip.is_valid());
        assert!(lip.constant > 0.0);
    }

    #[test]
    fn test_lipschitz_logit() {
        let d = logit_2p();
        let lip = LipschitzBound::for_logit(&d, &[0.0, 0.0], &[10.0, 10.0], 50);
        assert!(lip.is_valid());
        assert!(lip.constant > 0.0);
    }

    #[test]
    fn test_gross_substitutes_linear() {
        let d = linear_2p();
        assert!(check_gross_substitutes(&d, &[1.0, 1.0]).unwrap());
    }

    #[test]
    fn test_diagonal_dominance_linear() {
        let d = linear_2p();
        assert!(check_diagonal_dominance(&d, &[1.0, 1.0]).unwrap());
    }

    #[test]
    fn test_demand_validation_rejects_bad_params() {
        assert!(validate_demand_params(-1.0, 1.0, 0.5, 2).is_err());
        assert!(validate_demand_params(10.0, -1.0, 0.5, 2).is_err());
        assert!(validate_demand_params(10.0, 1.0, 1.5, 2).is_err()); // c >= b
        assert!(validate_demand_params(10.0, 1.0, 0.5, 1).is_err()); // too few
        assert!(validate_demand_params(10.0, 1.0, 0.5, 5).is_err()); // too many
    }

    #[test]
    fn test_create_demand_from_config_linear() {
        let config = crate::types::GameConfig::default();
        let demand = create_demand_from_config(&config).unwrap();
        assert_eq!(demand.num_players(), 2);
        let q = demand.compute_demand(&[1.0, 1.0]).unwrap();
        assert!(q[0] > 0.0);
    }

    #[test]
    fn test_invalid_price_count() {
        let d = linear_2p();
        assert!(d.compute_demand(&[1.0]).is_err());
        assert!(d.compute_demand(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_ces_requires_positive_prices() {
        let d = ces_2p();
        assert!(d.compute_demand(&[0.0, 1.0]).is_err());
        assert!(d.compute_demand(&[-1.0, 1.0]).is_err());
    }

    #[test]
    fn test_linear_4_players() {
        let d = LinearDemand::new(10.0, 2.0, 0.3, 4).unwrap();
        let q = d.compute_demand(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        // Q_i = 10 - 2*1 + 0.3*(1+1+1) = 8.9
        assert!((q[0] - 8.9).abs() < 1e-10);
    }
}
