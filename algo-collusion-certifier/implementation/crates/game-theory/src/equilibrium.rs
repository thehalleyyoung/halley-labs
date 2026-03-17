//! Nash equilibrium computation for stage games.
//!
//! Provides analytical solvers for Bertrand and Cournot games,
//! support enumeration for mixed strategies, iterated best response,
//! dominance elimination, and equilibrium verification.

use itertools::Itertools;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use shared_types::*;
use std::collections::HashMap;

// ── Nash Equilibrium ────────────────────────────────────────────────────────

/// A Nash equilibrium of a stage game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibrium {
    /// Strategy profile: for pure NE, the action index per player.
    pub strategy_profile: Vec<usize>,
    /// Equilibrium payoffs per player.
    pub payoffs: Vec<f64>,
    /// Support sets for mixed strategies (empty for pure NE).
    pub support: Vec<Vec<usize>>,
    /// Mixed strategy probabilities (empty for pure NE).
    pub mixed_probabilities: Vec<Vec<f64>>,
    /// Whether this is a symmetric equilibrium.
    pub is_symmetric: bool,
    /// Whether this is a pure strategy NE.
    pub is_pure: bool,
}

impl NashEquilibrium {
    pub fn pure(profile: Vec<usize>, payoffs: Vec<f64>) -> Self {
        let n = profile.len();
        Self {
            strategy_profile: profile.clone(),
            payoffs,
            support: profile.iter().map(|&a| vec![a]).collect(),
            mixed_probabilities: vec![],
            is_symmetric: n > 1 && profile.windows(2).all(|w| w[0] == w[1]),
            is_pure: true,
        }
    }

    pub fn mixed(support: Vec<Vec<usize>>, probs: Vec<Vec<f64>>, payoffs: Vec<f64>) -> Self {
        let n = support.len();
        Self {
            strategy_profile: support.iter().map(|s| s[0]).collect(),
            payoffs,
            support,
            mixed_probabilities: probs,
            is_symmetric: false,
            is_pure: false,
        }
    }

    pub fn num_players(&self) -> usize {
        self.payoffs.len()
    }
}

// ── Payoff Matrix ───────────────────────────────────────────────────────────

/// Normal-form game representation as a payoff matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffMatrix {
    /// Number of players (currently supports 2).
    pub num_players: usize,
    /// Number of actions per player.
    pub num_actions: Vec<usize>,
    /// Payoffs indexed as payoffs[player][action_profile_index].
    /// For 2 players: action_profile_index = a1 * num_actions[1] + a2.
    pub payoffs: Vec<Vec<f64>>,
}

impl PayoffMatrix {
    pub fn new(num_actions: Vec<usize>) -> Self {
        let n = num_actions.len();
        let total: usize = num_actions.iter().product();
        Self {
            num_players: n,
            num_actions,
            payoffs: vec![vec![0.0; total]; n],
        }
    }

    /// Create a 2-player game from payoff pairs.
    pub fn from_bimatrix(rows: usize, cols: usize, payoffs_a: Vec<Vec<f64>>, payoffs_b: Vec<Vec<f64>>) -> Self {
        let total = rows * cols;
        let mut pa = vec![0.0; total];
        let mut pb = vec![0.0; total];
        for i in 0..rows {
            for j in 0..cols {
                pa[i * cols + j] = payoffs_a[i][j];
                pb[i * cols + j] = payoffs_b[i][j];
            }
        }
        PayoffMatrix {
            num_players: 2,
            num_actions: vec![rows, cols],
            payoffs: vec![pa, pb],
        }
    }

    /// Get payoff for a player given a joint action profile.
    pub fn get_payoff(&self, player: usize, actions: &[usize]) -> f64 {
        let idx = self.action_profile_index(actions);
        self.payoffs[player][idx]
    }

    /// Set payoff for a player given a joint action profile.
    pub fn set_payoff(&mut self, player: usize, actions: &[usize], value: f64) {
        let idx = self.action_profile_index(actions);
        self.payoffs[player][idx] = value;
    }

    fn action_profile_index(&self, actions: &[usize]) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        for i in (0..self.num_players).rev() {
            idx += actions[i] * multiplier;
            multiplier *= self.num_actions[i];
        }
        idx
    }

    /// Enumerate all action profiles.
    pub fn all_profiles(&self) -> Vec<Vec<usize>> {
        let ranges: Vec<Vec<usize>> = self.num_actions.iter().map(|&n| (0..n).collect()).collect();
        ranges.into_iter().multi_cartesian_product().collect()
    }

    /// Best response for a player given opponents' actions.
    pub fn best_response(&self, player: usize, opponents_actions: &[usize]) -> (usize, f64) {
        let mut best_action = 0;
        let mut best_payoff = f64::NEG_INFINITY;
        for a in 0..self.num_actions[player] {
            let mut actions = opponents_actions.to_vec();
            actions.insert(player, a);
            let payoff = self.get_payoff(player, &actions);
            if payoff > best_payoff {
                best_payoff = payoff;
                best_action = a;
            }
        }
        (best_action, best_payoff)
    }
}

// ── Bertrand Nash Solver ────────────────────────────────────────────────────

/// Analytical Nash equilibrium solver for differentiated Bertrand competition.
pub struct BertrandNashSolver;

impl BertrandNashSolver {
    /// Solve for Nash equilibrium prices in differentiated Bertrand.
    ///
    /// Linear demand: q_i = a - b*p_i + c*p_j
    /// Profit: pi_i = (p_i - mc_i) * q_i
    /// FOC: a - 2b*p_i + c*p_j + b*mc_i = 0
    /// => p_i = (a + c*p_j + b*mc_i) / (2b)
    pub fn solve(config: &GameConfig) -> CollusionResult<NashEquilibrium> {
        match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => {
                let intercept = *max_quantity;
                let cross_slope = 0.0_f64;
                let a = intercept;
                let b = *slope;
                let c = cross_slope;
                let n = config.num_players;

                if n == 0 {
                    return Err(CollusionError::Config("Need at least 1 player".into()));
                }

                if (2.0 * b).abs() < 1e-12 {
                    return Err(CollusionError::Config("Slope too small".into()));
                }

                if c.abs() < 1e-12 {
                    // Homogeneous Bertrand: p* = mc (Bertrand paradox)
                    let prices: Vec<f64> = config.marginal_costs.iter().map(|c| c.0).collect();
                    let profits = vec![0.0; n];
                    return Ok(NashEquilibrium {
                        strategy_profile: vec![0; n],
                        payoffs: profits,
                        support: vec![vec![0]; n],
                        mixed_probabilities: vec![],
                        is_symmetric: config.marginal_costs.windows(2).all(|w| (w[0].0 - w[1].0).abs() < 1e-12),
                        is_pure: true,
                    });
                }

                if n == 2 {
                    // 2-player system:
                    // p1 = (a + c*p2 + b*mc1) / (2b)
                    // p2 = (a + c*p1 + b*mc2) / (2b)
                    let mc1 = config.marginal_costs[0].0;
                    let mc2 = config.marginal_costs[1].0;
                    let denom = 4.0 * b * b - c * c;
                    if denom.abs() < 1e-12 {
                        return Err(CollusionError::Config("Degenerate system".into()));
                    }
                    let p1 = (2.0 * b * (a + b * mc1) + c * (a + b * mc2)) / denom;
                    let p2 = (2.0 * b * (a + b * mc2) + c * (a + b * mc1)) / denom;
                    let q1 = (a - b * p1 + c * p2).max(0.0);
                    let q2 = (a - b * p2 + c * p1).max(0.0);
                    let pi1 = (p1 - mc1) * q1;
                    let pi2 = (p2 - mc2) * q2;

                    Ok(NashEquilibrium {
                        strategy_profile: vec![0, 0],
                        payoffs: vec![pi1, pi2],
                        support: vec![vec![0], vec![0]],
                        mixed_probabilities: vec![],
                        is_symmetric: (mc1 - mc2).abs() < 1e-12,
                        is_pure: true,
                    })
                } else {
                    // Symmetric N-player: p* = (a + b*mc) / (2b - (n-1)*c)
                    // (assuming symmetric costs)
                    let mc = config.marginal_costs[0].0;
                    let denom = 2.0 * b - (n as f64 - 1.0) * c;
                    if denom.abs() < 1e-12 {
                        return Err(CollusionError::Config("Degenerate N-player system".into()));
                    }
                    let p_star = (a + b * mc) / denom;
                    let q_star = (a - b * p_star + c * p_star * (n as f64 - 1.0)).max(0.0);
                    let pi_star = (p_star - mc) * q_star;

                    Ok(NashEquilibrium {
                        strategy_profile: vec![0; n],
                        payoffs: vec![pi_star; n],
                        support: vec![vec![0]; n],
                        mixed_probabilities: vec![],
                        is_symmetric: true,
                        is_pure: true,
                    })
                }
            }
            DemandSystem::Logit { temperature, outside_option_value, market_size: _ } => {
                let mu = *temperature;
                let a_0 = *outside_option_value;
                // Logit demand: simplified symmetric equilibrium
                let n = config.num_players;
                let mc = config.marginal_costs[0].0;
                // In logit competition, NE price ~ mc + mu (for large n)
                let p_star = mc + mu;
                let share = 1.0 / n as f64;
                let pi_star = (p_star - mc) * share * a_0;

                Ok(NashEquilibrium {
                    strategy_profile: vec![0; n],
                    payoffs: vec![pi_star; n],
                    support: vec![vec![0]; n],
                    mixed_probabilities: vec![],
                    is_symmetric: true,
                    is_pure: true,
                })
            }
            DemandSystem::CES { elasticity_of_substitution, .. } => {
                let sigma = *elasticity_of_substitution;
                let n = config.num_players;
                let mc = config.marginal_costs[0].0;
                // CES markup: p = mc * sigma / (sigma - 1)
                let markup = if sigma > 1.0 { sigma / (sigma - 1.0) } else { 2.0 };
                let p_star = mc * markup;
                let share = 1.0 / n as f64;
                let pi_star = (p_star - mc) * share;

                Ok(NashEquilibrium {
                    strategy_profile: vec![0; n],
                    payoffs: vec![pi_star; n],
                    support: vec![vec![0]; n],
                    mixed_probabilities: vec![],
                    is_symmetric: true,
                    is_pure: true,
                })
            }
        }
    }
}

// ── Cournot Nash Solver ─────────────────────────────────────────────────────

/// Analytical Nash equilibrium solver for N-player Cournot competition.
pub struct CournotNashSolver;

impl CournotNashSolver {
    /// Solve for Nash equilibrium in Cournot with linear inverse demand.
    ///
    /// Inverse demand: P(Q) = a - b*Q where Q = sum(q_i)
    /// Profit: pi_i = (a - b*(q_i + Q_{-i}) - mc_i) * q_i
    /// FOC: a - 2b*q_i - b*Q_{-i} - mc_i = 0
    pub fn solve(config: &GameConfig) -> CollusionResult<NashEquilibrium> {
        let n = config.num_players;
        if n == 0 {
            return Err(CollusionError::Config("Need at least 1 player".into()));
        }

        let (a, b) = match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => (*max_quantity, *slope),
            _ => return Err(CollusionError::Config("Cournot requires linear demand".into())),
        };

        if b.abs() < 1e-12 {
            return Err(CollusionError::Config("Slope must be nonzero".into()));
        }

        // Symmetric case: all firms have same mc
        let all_symmetric = config.marginal_costs.windows(2).all(|w| (w[0].0 - w[1].0).abs() < 1e-12);

        if all_symmetric {
            let mc = config.marginal_costs[0].0;
            // q_i* = (a - mc) / (b * (n + 1))
            let q_star = ((a - mc) / (b * (n as f64 + 1.0))).max(0.0);
            let total_q = q_star * n as f64;
            let price = (a - b * total_q).max(0.0);
            let pi_star = (price - mc) * q_star;

            Ok(NashEquilibrium {
                strategy_profile: vec![0; n],
                payoffs: vec![pi_star; n],
                support: vec![vec![0]; n],
                mixed_probabilities: vec![],
                is_symmetric: true,
                is_pure: true,
            })
        } else {
            // Asymmetric: solve the system
            // q_i = (a - mc_i - b * sum_{j!=i} q_j) / (2b)
            // Iterative solution
            let mut quantities = vec![0.0f64; n];
            let max_iter = 1000;
            let tol = 1e-10;

            for _ in 0..max_iter {
                let mut max_change = 0.0f64;
                for i in 0..n {
                    let mc_i = config.marginal_costs[i].0;
                    let others_q: f64 = quantities.iter().enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, q)| *q)
                        .sum();
                    let new_q = ((a - mc_i - b * others_q) / (2.0 * b)).max(0.0);
                    max_change = max_change.max((new_q - quantities[i]).abs());
                    quantities[i] = new_q;
                }
                if max_change < tol {
                    break;
                }
            }

            let total_q: f64 = quantities.iter().sum();
            let price = (a - b * total_q).max(0.0);
            let profits: Vec<f64> = quantities.iter().enumerate()
                .map(|(i, &q)| (price - config.marginal_costs[i].0) * q)
                .collect();

            Ok(NashEquilibrium {
                strategy_profile: vec![0; n],
                payoffs: profits,
                support: vec![vec![0]; n],
                mixed_probabilities: vec![],
                is_symmetric: false,
                is_pure: true,
            })
        }
    }
}

// ── Mixed Strategy NE ───────────────────────────────────────────────────────

/// Support enumeration algorithm for 2-player finite games.
pub struct MixedStrategyNE;

impl MixedStrategyNE {
    /// Find all Nash equilibria (pure and mixed) for a 2-player game.
    pub fn solve(game: &PayoffMatrix) -> Vec<NashEquilibrium> {
        if game.num_players != 2 {
            return vec![];
        }

        let m = game.num_actions[0];
        let n = game.num_actions[1];
        let mut equilibria = Vec::new();

        // Enumerate all possible support pairs
        for support_size_1 in 1..=m {
            for support_size_2 in 1..=n {
                let supports_1: Vec<Vec<usize>> = (0..m).combinations(support_size_1).collect();
                let supports_2: Vec<Vec<usize>> = (0..n).combinations(support_size_2).collect();

                for s1 in &supports_1 {
                    for s2 in &supports_2 {
                        if let Some(ne) = Self::check_support_pair(game, s1, s2) {
                            equilibria.push(ne);
                        }
                    }
                }
            }
        }

        equilibria
    }

    fn check_support_pair(
        game: &PayoffMatrix,
        support_1: &[usize],
        support_2: &[usize],
    ) -> Option<NashEquilibrium> {
        let m = game.num_actions[0];
        let n = game.num_actions[1];
        let k1 = support_1.len();
        let k2 = support_2.len();

        // For player 2's mixed strategy to make player 1 indifferent over support_1:
        // sum_j q_j * u1(i, j) = v1 for all i in support_1
        // sum_j q_j = 1, q_j >= 0
        // And vice versa.

        // Simple case: pure strategy NE
        if k1 == 1 && k2 == 1 {
            let i = support_1[0];
            let j = support_2[0];
            // Check if (i,j) is a pure NE
            let payoff_1 = game.get_payoff(0, &[i, j]);
            let payoff_2 = game.get_payoff(1, &[i, j]);

            // Check BR for player 1
            for a in 0..m {
                if game.get_payoff(0, &[a, j]) > payoff_1 + 1e-12 {
                    return None;
                }
            }
            // Check BR for player 2
            for b in 0..n {
                if game.get_payoff(1, &[i, b]) > payoff_2 + 1e-12 {
                    return None;
                }
            }

            return Some(NashEquilibrium::pure(vec![i, j], vec![payoff_1, payoff_2]));
        }

        // Mixed NE: solve indifference conditions
        // For 2x2 case, we have a closed-form solution
        if k1 == 2 && k2 == 2 {
            let i0 = support_1[0];
            let i1 = support_1[1];
            let j0 = support_2[0];
            let j1 = support_2[1];

            // Player 2 mixes with prob q on j0 and (1-q) on j1
            // u1(i0, q) = u1(i1, q) =>
            // q*u1(i0,j0) + (1-q)*u1(i0,j1) = q*u1(i1,j0) + (1-q)*u1(i1,j1)
            let a00 = game.get_payoff(0, &[i0, j0]);
            let a01 = game.get_payoff(0, &[i0, j1]);
            let a10 = game.get_payoff(0, &[i1, j0]);
            let a11 = game.get_payoff(0, &[i1, j1]);

            let denom_q = (a00 - a01) - (a10 - a11);
            if denom_q.abs() < 1e-12 {
                return None;
            }
            let q = (a11 - a01) / denom_q;

            // Player 1 mixes with prob p on i0 and (1-p) on i1
            let b00 = game.get_payoff(1, &[i0, j0]);
            let b01 = game.get_payoff(1, &[i0, j1]);
            let b10 = game.get_payoff(1, &[i1, j0]);
            let b11 = game.get_payoff(1, &[i1, j1]);

            let denom_p = (b00 - b10) - (b01 - b11);
            if denom_p.abs() < 1e-12 {
                return None;
            }
            let p = (b11 - b10) / denom_p;

            // Check feasibility
            if q < -1e-12 || q > 1.0 + 1e-12 || p < -1e-12 || p > 1.0 + 1e-12 {
                return None;
            }
            let q = q.clamp(0.0, 1.0);
            let p = p.clamp(0.0, 1.0);

            // Compute expected payoffs
            let v1 = q * a00 + (1.0 - q) * a01;
            let v2 = p * b00 + (1.0 - p) * b01;

            // Verify: no action outside support gives higher payoff
            for a in 0..m {
                if !support_1.contains(&a) {
                    let u = q * game.get_payoff(0, &[a, j0]) + (1.0 - q) * game.get_payoff(0, &[a, j1]);
                    if u > v1 + 1e-10 {
                        return None;
                    }
                }
            }
            for b in 0..n {
                if !support_2.contains(&b) {
                    let u = p * game.get_payoff(1, &[i0, b]) + (1.0 - p) * game.get_payoff(1, &[i1, b]);
                    if u > v2 + 1e-10 {
                        return None;
                    }
                }
            }

            let mut probs_1 = vec![0.0; m];
            probs_1[i0] = p;
            probs_1[i1] = 1.0 - p;
            let mut probs_2 = vec![0.0; n];
            probs_2[j0] = q;
            probs_2[j1] = 1.0 - q;

            return Some(NashEquilibrium::mixed(
                vec![support_1.to_vec(), support_2.to_vec()],
                vec![probs_1, probs_2],
                vec![v1, v2],
            ));
        }

        None // Only handle pure and 2x2 mixed
    }
}

// ── Iterated Best Response ──────────────────────────────────────────────────

/// Iterative best-response dynamics for finding Nash equilibria.
pub struct IteratedBestResponse {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl IteratedBestResponse {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self { max_iterations, tolerance }
    }

    pub fn solve(&self, game: &PayoffMatrix) -> Option<NashEquilibrium> {
        let n = game.num_players;
        let mut actions: Vec<usize> = vec![0; n];

        for iteration in 0..self.max_iterations {
            let mut changed = false;
            for player in 0..n {
                let opponents: Vec<usize> = actions.iter().enumerate()
                    .filter(|(j, _)| *j != player)
                    .map(|(_, &a)| a)
                    .collect();
                let (br, _) = game.best_response(player, &opponents);
                if br != actions[player] {
                    actions[player] = br;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Verify it's actually a NE
        let verifier = EquilibriumVerifier;
        if verifier.is_nash(game, &actions, self.tolerance) {
            let payoffs: Vec<f64> = (0..n).map(|i| game.get_payoff(i, &actions)).collect();
            Some(NashEquilibrium::pure(actions, payoffs))
        } else {
            None
        }
    }
}

// ── Best Response Dynamics ──────────────────────────────────────────────────

/// Compute best-response mappings for discrete action spaces.
pub struct BestResponseDynamics;

impl BestResponseDynamics {
    /// Compute the best response for each player given a price grid and demand system.
    pub fn compute_best_responses(
        config: &GameConfig,
        price_grid: &[f64],
        current_prices: &[f64],
    ) -> Vec<(usize, f64)> {
        let n = config.num_players;
        let mut best_responses = Vec::with_capacity(n);

        for player in 0..n {
            let mc = config.marginal_costs.get(player).map(|c| c.0).unwrap_or(0.0);
            let mut best_price = price_grid[0];
            let mut best_profit = f64::NEG_INFINITY;

            for &p in price_grid {
                let mut trial_prices = current_prices.to_vec();
                trial_prices[player] = p;
                let profit = Self::compute_profit(config, &trial_prices, player, mc);
                if profit > best_profit {
                    best_profit = profit;
                    best_price = p;
                }
            }
            best_responses.push((
                price_grid.iter().position(|&x| (x - best_price).abs() < 1e-12).unwrap_or(0),
                best_price,
            ));
        }
        best_responses
    }

    fn compute_profit(config: &GameConfig, prices: &[f64], player: usize, mc: f64) -> f64 {
        match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => {
                let intercept = *max_quantity;
                let cross_slope = 0.0_f64;
                let n = prices.len();
                let others_avg: f64 = if n > 1 {
                    prices.iter().enumerate()
                        .filter(|(j, _)| *j != player)
                        .map(|(_, &p)| p)
                        .sum::<f64>() / (n - 1) as f64
                } else {
                    0.0
                };
                let q = (intercept - slope * prices[player] + cross_slope * others_avg).max(0.0);
                (prices[player] - mc) * q
            }
            DemandSystem::Logit { temperature, outside_option_value, market_size: _ } => {
                let mu = *temperature;
                let a_0 = *outside_option_value;
                let exp_sum: f64 = prices.iter().map(|&p| ((a_0 - p) / mu).exp()).sum();
                let share = ((a_0 - prices[player]) / mu).exp() / (1.0 + exp_sum);
                (prices[player] - mc) * share
            }
            DemandSystem::CES { elasticity_of_substitution, .. } => {
                let sigma = *elasticity_of_substitution;
                let n = prices.len();
                let price_idx: f64 = prices.iter().map(|&p| p.powf(1.0 - sigma)).sum();
                let share = prices[player].powf(-sigma) / price_idx.max(1e-12);
                (prices[player] - mc) * share
            }
        }
    }
}

// ── Equilibrium Verifier ────────────────────────────────────────────────────

/// Verify whether a strategy profile constitutes an epsilon-Nash equilibrium.
pub struct EquilibriumVerifier;

impl EquilibriumVerifier {
    /// Check if the given action profile is an epsilon-Nash equilibrium.
    pub fn is_nash(&self, game: &PayoffMatrix, actions: &[usize], epsilon: f64) -> bool {
        for player in 0..game.num_players {
            let current_payoff = game.get_payoff(player, actions);
            for alt_action in 0..game.num_actions[player] {
                let mut alt_actions = actions.to_vec();
                alt_actions[player] = alt_action;
                let alt_payoff = game.get_payoff(player, &alt_actions);
                if alt_payoff > current_payoff + epsilon {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the maximum profitable deviation for any player.
    pub fn max_deviation_gain(&self, game: &PayoffMatrix, actions: &[usize]) -> f64 {
        let mut max_gain = 0.0f64;
        for player in 0..game.num_players {
            let current_payoff = game.get_payoff(player, actions);
            for alt_action in 0..game.num_actions[player] {
                let mut alt_actions = actions.to_vec();
                alt_actions[player] = alt_action;
                let alt_payoff = game.get_payoff(player, &alt_actions);
                let gain = alt_payoff - current_payoff;
                max_gain = max_gain.max(gain);
            }
        }
        max_gain
    }

    /// Verify NE for continuous prices using a grid search.
    pub fn verify_continuous_ne(
        config: &GameConfig,
        prices: &[f64],
        epsilon: f64,
        grid_resolution: usize,
    ) -> bool {
        let n = config.num_players;
        let (p_min, p_max) = (0.0, prices.iter().cloned().fold(0.0_f64, f64::max) * 2.0);

        for player in 0..n {
            let mc = config.marginal_costs.get(player).map(|c| c.0).unwrap_or(0.0);
            let current_profit = BestResponseDynamics::compute_profit(config, prices, player, mc);

            for k in 0..=grid_resolution {
                let trial_price = p_min + (p_max - p_min) * k as f64 / grid_resolution as f64;
                let mut trial_prices = prices.to_vec();
                trial_prices[player] = trial_price;
                let trial_profit = BestResponseDynamics::compute_profit(config, &trial_prices, player, mc);
                if trial_profit > current_profit + epsilon {
                    return false;
                }
            }
        }
        true
    }
}

// ── Dominance Elimination ───────────────────────────────────────────────────

/// Iteratively eliminate strictly dominated strategies.
pub struct DominanceElimination;

impl DominanceElimination {
    /// Remove strictly dominated strategies from a 2-player game.
    /// Returns the reduced game and the mapping of remaining actions.
    pub fn eliminate(game: &PayoffMatrix) -> (PayoffMatrix, Vec<Vec<usize>>) {
        if game.num_players != 2 {
            return (game.clone(), (0..game.num_players).map(|i| (0..game.num_actions[i]).collect()).collect());
        }

        let mut remaining = vec![
            (0..game.num_actions[0]).collect::<Vec<usize>>(),
            (0..game.num_actions[1]).collect::<Vec<usize>>(),
        ];
        let mut changed = true;

        while changed {
            changed = false;

            // Check player 0's actions
            let mut to_remove_0 = Vec::new();
            for (idx_i, &action_i) in remaining[0].iter().enumerate() {
                for &action_k in &remaining[0] {
                    if action_k == action_i { continue; }
                    // Check if action_i is strictly dominated by action_k
                    let dominated = remaining[1].iter().all(|&action_j| {
                        game.get_payoff(0, &[action_k, action_j]) > game.get_payoff(0, &[action_i, action_j]) + 1e-12
                    });
                    if dominated {
                        to_remove_0.push(idx_i);
                        break;
                    }
                }
            }
            for &idx in to_remove_0.iter().rev() {
                remaining[0].remove(idx);
                changed = true;
            }

            // Check player 1's actions
            let mut to_remove_1 = Vec::new();
            for (idx_j, &action_j) in remaining[1].iter().enumerate() {
                for &action_l in &remaining[1] {
                    if action_l == action_j { continue; }
                    let dominated = remaining[0].iter().all(|&action_i| {
                        game.get_payoff(1, &[action_i, action_l]) > game.get_payoff(1, &[action_i, action_j]) + 1e-12
                    });
                    if dominated {
                        to_remove_1.push(idx_j);
                        break;
                    }
                }
            }
            for &idx in to_remove_1.iter().rev() {
                remaining[1].remove(idx);
                changed = true;
            }
        }

        // Build reduced game
        let m = remaining[0].len();
        let n = remaining[1].len();
        let mut reduced = PayoffMatrix::new(vec![m, n]);
        for (new_i, &old_i) in remaining[0].iter().enumerate() {
            for (new_j, &old_j) in remaining[1].iter().enumerate() {
                reduced.set_payoff(0, &[new_i, new_j], game.get_payoff(0, &[old_i, old_j]));
                reduced.set_payoff(1, &[new_i, new_j], game.get_payoff(1, &[old_i, old_j]));
            }
        }

        (reduced, remaining)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bertrand_config() -> GameConfig {
        GameConfig {
            num_players: 2,
            discount_factor: 0.95,
            marginal_costs: vec![Cost(1.0), Cost(1.0)],
            demand_system: DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
            market_type: MarketType::Bertrand,
            price_grid: None,
            max_rounds: 1000,
            description: String::new(),
        }
    }

    fn default_cournot_config() -> GameConfig {
        GameConfig {
            num_players: 2,
            discount_factor: 0.95,
            marginal_costs: vec![Cost(1.0), Cost(1.0)],
            demand_system: DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
            market_type: MarketType::Cournot,
            price_grid: None,
            max_rounds: 1000,
            description: String::new(),
        }
    }

    #[test]
    fn test_bertrand_symmetric_ne() {
        let config = default_bertrand_config();
        let ne = BertrandNashSolver::solve(&config).unwrap();
        assert_eq!(ne.payoffs.len(), 2);
        assert!(ne.is_symmetric);
        // Both players should have positive profits with differentiation
        assert!(ne.payoffs[0] > 0.0);
        assert!((ne.payoffs[0] - ne.payoffs[1]).abs() < 1e-10);
    }

    #[test]
    fn test_bertrand_homogeneous() {
        let config = GameConfig {
            demand_system: DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
            ..default_bertrand_config()
        };
        let ne = BertrandNashSolver::solve(&config).unwrap();
        // Bertrand paradox: zero profits
        assert!((ne.payoffs[0]).abs() < 1e-10);
    }

    #[test]
    fn test_bertrand_asymmetric() {
        let config = GameConfig {
            marginal_costs: vec![Cost(1.0), Cost(2.0)],
            ..default_bertrand_config()
        };
        let ne = BertrandNashSolver::solve(&config).unwrap();
        assert!(!ne.is_symmetric);
        assert_eq!(ne.payoffs.len(), 2);
    }

    #[test]
    fn test_cournot_symmetric_ne() {
        let config = default_cournot_config();
        let ne = CournotNashSolver::solve(&config).unwrap();
        assert!(ne.is_symmetric);
        assert!(ne.payoffs[0] > 0.0);
        assert!((ne.payoffs[0] - ne.payoffs[1]).abs() < 1e-10);
    }

    #[test]
    fn test_cournot_asymmetric() {
        let config = GameConfig {
            marginal_costs: vec![Cost(1.0), Cost(3.0)],
            ..default_cournot_config()
        };
        let ne = CournotNashSolver::solve(&config).unwrap();
        // Lower cost firm should earn more
        assert!(ne.payoffs[0] > ne.payoffs[1]);
    }

    #[test]
    fn test_cournot_three_players() {
        let config = GameConfig {
            num_players: 3,
            marginal_costs: vec![Cost(1.0), Cost(1.0), Cost(1.0)],
            ..default_cournot_config()
        };
        let ne = CournotNashSolver::solve(&config).unwrap();
        assert_eq!(ne.payoffs.len(), 3);
        assert!(ne.is_symmetric);
    }

    #[test]
    fn test_payoff_matrix_creation() {
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        );
        assert_eq!(game.get_payoff(0, &[0, 0]), 3.0);
        assert_eq!(game.get_payoff(1, &[0, 1]), 5.0);
    }

    #[test]
    fn test_prisoners_dilemma_pure_ne() {
        // Standard PD: (C,C)=3,3  (C,D)=0,5  (D,C)=5,0  (D,D)=1,1
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        );
        let equilibria = MixedStrategyNE::solve(&game);
        // Should find (D,D) as pure NE
        let pure_ne = equilibria.iter().find(|e| e.is_pure).unwrap();
        assert_eq!(pure_ne.strategy_profile, vec![1, 1]);
    }

    #[test]
    fn test_matching_pennies_mixed_ne() {
        // Matching pennies: (H,H)=1,-1  (H,T)=-1,1  (T,H)=-1,1  (T,T)=1,-1
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
            vec![vec![-1.0, 1.0], vec![1.0, -1.0]],
        );
        let equilibria = MixedStrategyNE::solve(&game);
        let mixed = equilibria.iter().find(|e| !e.is_pure);
        assert!(mixed.is_some());
        let ne = mixed.unwrap();
        // Both should mix 50/50
        assert!((ne.mixed_probabilities[0][0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_iterated_best_response() {
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        );
        let solver = IteratedBestResponse::new(100, 1e-6);
        let ne = solver.solve(&game);
        assert!(ne.is_some());
        assert_eq!(ne.unwrap().strategy_profile, vec![1, 1]);
    }

    #[test]
    fn test_best_response_dynamics() {
        let config = default_bertrand_config();
        let grid: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
        let current = vec![5.0, 5.0];
        let brs = BestResponseDynamics::compute_best_responses(&config, &grid, &current);
        assert_eq!(brs.len(), 2);
    }

    #[test]
    fn test_equilibrium_verifier_pd() {
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        );
        let verifier = EquilibriumVerifier;
        assert!(verifier.is_nash(&game, &[1, 1], 0.0));
        assert!(!verifier.is_nash(&game, &[0, 0], 0.0));
    }

    #[test]
    fn test_equilibrium_verifier_max_deviation() {
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        );
        let verifier = EquilibriumVerifier;
        let gain = verifier.max_deviation_gain(&game, &[0, 0]);
        assert!((gain - 2.0).abs() < 1e-10); // D gives 5 vs C gives 3
    }

    #[test]
    fn test_dominance_elimination_pd() {
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        );
        let (reduced, remaining) = DominanceElimination::eliminate(&game);
        // D strictly dominates C for both players
        assert_eq!(reduced.num_actions, vec![1, 1]);
        assert_eq!(remaining[0], vec![1]);
        assert_eq!(remaining[1], vec![1]);
    }

    #[test]
    fn test_dominance_no_elimination() {
        // Battle of the sexes: no dominated strategies
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![0.0, 2.0]],
            vec![vec![2.0, 0.0], vec![0.0, 3.0]],
        );
        let (reduced, remaining) = DominanceElimination::eliminate(&game);
        assert_eq!(reduced.num_actions, vec![2, 2]);
    }

    #[test]
    fn test_logit_ne() {
        let config = GameConfig {
            demand_system: DemandSystem::Logit { temperature: 0.5, outside_option_value: 5.0, market_size: 100.0 },
            ..default_bertrand_config()
        };
        let ne = BertrandNashSolver::solve(&config).unwrap();
        assert!(ne.payoffs[0] > 0.0);
    }

    #[test]
    fn test_ces_ne() {
        let config = GameConfig {
            demand_system: DemandSystem::CES { elasticity_of_substitution: 3.0, market_size: 100.0, quality_indices: vec![1.0, 1.0] },
            ..default_bertrand_config()
        };
        let ne = BertrandNashSolver::solve(&config).unwrap();
        assert!(ne.payoffs[0] > 0.0);
    }

    #[test]
    fn test_nash_equilibrium_constructors() {
        let pure = NashEquilibrium::pure(vec![0, 1], vec![3.0, 2.0]);
        assert!(pure.is_pure);
        assert!(!pure.is_symmetric);

        let mixed = NashEquilibrium::mixed(
            vec![vec![0, 1], vec![0, 1]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![0.0, 0.0],
        );
        assert!(!mixed.is_pure);
    }

    #[test]
    fn test_payoff_matrix_all_profiles() {
        let game = PayoffMatrix::new(vec![2, 3]);
        let profiles = game.all_profiles();
        assert_eq!(profiles.len(), 6);
    }

    #[test]
    fn test_best_response_in_matrix() {
        let game = PayoffMatrix::from_bimatrix(
            2, 2,
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        );
        let (br, payoff) = game.best_response(0, &[0]); // player 0 vs opp plays 0
        assert_eq!(br, 1); // D is best response
        assert!((payoff - 5.0).abs() < 1e-10);
    }
}
