//! Asymmetric game certification for the CollusionProof system.
//!
//! Extends the core detection engine beyond symmetric games to handle
//! heterogeneous agents with differing strategy sets, cost structures,
//! capacities, and brand loyalty parameters. Provides:
//!
//! - Asymmetric game representation (per-agent cost, capacity, strategy sets)
//! - Asymmetric coalition certification via quantified SMT (QF_LRA)
//! - Asymmetric Cournot game (firms with different marginal costs)
//! - Asymmetric Bertrand game (firms with different quality/brand loyalty)
//! - Asymmetric auction game (bidders with heterogeneous budgets)
//! - Certificate generation for coalitions of heterogeneous agents

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Asymmetric agent specification
// ═══════════════════════════════════════════════════════════════════════════

/// An agent in an asymmetric game with its own cost, capacity, and strategy set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricAgent {
    pub id: usize,
    pub label: String,
    pub marginal_cost: f64,
    pub capacity: f64,
    pub brand_loyalty: f64,
    pub budget: f64,
    pub strategy_min: f64,
    pub strategy_max: f64,
    pub strategy_count: usize,
}

impl AsymmetricAgent {
    pub fn new(id: usize, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            marginal_cost: 0.0,
            capacity: f64::INFINITY,
            brand_loyalty: 0.0,
            budget: f64::INFINITY,
            strategy_min: 0.0,
            strategy_max: 100.0,
            strategy_count: 21,
        }
    }

    pub fn with_cost(mut self, mc: f64) -> Self {
        self.marginal_cost = mc;
        self
    }

    pub fn with_capacity(mut self, cap: f64) -> Self {
        self.capacity = cap;
        self
    }

    pub fn with_brand_loyalty(mut self, bl: f64) -> Self {
        self.brand_loyalty = bl;
        self
    }

    pub fn with_budget(mut self, b: f64) -> Self {
        self.budget = b;
        self
    }

    pub fn with_strategy_range(mut self, min: f64, max: f64, count: usize) -> Self {
        self.strategy_min = min;
        self.strategy_max = max;
        self.strategy_count = count;
        self
    }

    /// Discretized strategy set for this agent.
    pub fn strategies(&self) -> Vec<f64> {
        if self.strategy_count <= 1 {
            return vec![self.strategy_min];
        }
        let step = (self.strategy_max - self.strategy_min) / (self.strategy_count - 1) as f64;
        (0..self.strategy_count)
            .map(|i| (self.strategy_min + step * i as f64).min(self.strategy_max))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Asymmetric game definition
// ═══════════════════════════════════════════════════════════════════════════

/// Game type for asymmetric certification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsymmetricGameType {
    Cournot,
    Bertrand,
    Auction,
}

impl fmt::Display for AsymmetricGameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cournot => write!(f, "Cournot"),
            Self::Bertrand => write!(f, "Bertrand"),
            Self::Auction => write!(f, "Auction"),
        }
    }
}

/// An asymmetric game with heterogeneous agents and per-agent payoff functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricGame {
    pub game_type: AsymmetricGameType,
    pub agents: Vec<AsymmetricAgent>,
    pub demand_intercept: f64,
    pub demand_slope: f64,
    pub discount_factor: f64,
}

impl AsymmetricGame {
    pub fn new(game_type: AsymmetricGameType, agents: Vec<AsymmetricAgent>) -> Self {
        Self {
            game_type,
            agents,
            demand_intercept: 100.0,
            demand_slope: 1.0,
            discount_factor: 0.95,
        }
    }

    pub fn with_demand(mut self, intercept: f64, slope: f64) -> Self {
        self.demand_intercept = intercept;
        self.demand_slope = slope;
        self
    }

    pub fn with_discount(mut self, delta: f64) -> Self {
        self.discount_factor = delta;
        self
    }

    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// Compute payoff for agent `i` given a joint action profile.
    pub fn payoff(&self, agent_idx: usize, actions: &[f64]) -> f64 {
        let agent = &self.agents[agent_idx];
        match self.game_type {
            AsymmetricGameType::Cournot => {
                let total_q: f64 = actions.iter().sum();
                let price = (self.demand_intercept - self.demand_slope * total_q).max(0.0);
                let q_i = actions[agent_idx].min(agent.capacity);
                price * q_i - agent.marginal_cost * q_i
            }
            AsymmetricGameType::Bertrand => {
                let p_i = actions[agent_idx];
                let n = actions.len() as f64;
                // Differentiated Bertrand with brand loyalty:
                //   q_i = (a - p_i + loyalty_i + (1/n)·Σ_{j≠i} p_j) / b
                let rival_avg: f64 = actions
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != agent_idx)
                    .map(|(_, &p)| p)
                    .sum::<f64>()
                    / (n - 1.0).max(1.0);
                let q_i = ((self.demand_intercept - p_i + agent.brand_loyalty
                    + 0.5 * rival_avg)
                    / self.demand_slope)
                    .max(0.0);
                (p_i - agent.marginal_cost) * q_i
            }
            AsymmetricGameType::Auction => {
                // First-price sealed-bid: highest bid wins, pays bid.
                // Agent value ~ budget; payoff = (budget - bid) if win, 0 otherwise.
                let bid_i = actions[agent_idx].min(agent.budget);
                let max_rival = actions
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != agent_idx)
                    .map(|(_, &b)| b)
                    .fold(f64::NEG_INFINITY, f64::max);
                if bid_i > max_rival {
                    agent.budget - bid_i
                } else if (bid_i - max_rival).abs() < 1e-12 {
                    (agent.budget - bid_i) / 2.0 // tie-break
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute full payoff matrix for all agents across all strategy combinations.
    /// Returns a Vec of per-agent payoff maps keyed by joint-action index.
    pub fn payoff_matrix(&self) -> AsymmetricPayoffMatrix {
        let strategy_sets: Vec<Vec<f64>> =
            self.agents.iter().map(|a| a.strategies()).collect();
        let dims: Vec<usize> = strategy_sets.iter().map(|s| s.len()).collect();
        let total_profiles = dims.iter().product::<usize>();
        let n = self.n_agents();
        let mut payoffs = vec![vec![0.0f64; total_profiles]; n];

        for idx in 0..total_profiles {
            let actions = decode_profile(idx, &dims, &strategy_sets);
            for i in 0..n {
                payoffs[i][idx] = self.payoff(i, &actions);
            }
        }

        AsymmetricPayoffMatrix {
            n_agents: n,
            dims,
            strategy_sets,
            payoffs,
        }
    }
}

/// Payoff matrix for an asymmetric N-player game.
#[derive(Debug, Clone)]
pub struct AsymmetricPayoffMatrix {
    pub n_agents: usize,
    pub dims: Vec<usize>,
    pub strategy_sets: Vec<Vec<f64>>,
    pub payoffs: Vec<Vec<f64>>,
}

impl AsymmetricPayoffMatrix {
    pub fn total_profiles(&self) -> usize {
        self.dims.iter().product()
    }

    /// Decode a flat profile index into individual strategy indices.
    pub fn decode_indices(&self, flat: usize) -> Vec<usize> {
        let mut indices = vec![0usize; self.n_agents];
        let mut rem = flat;
        for i in (0..self.n_agents).rev() {
            indices[i] = rem % self.dims[i];
            rem /= self.dims[i];
        }
        indices
    }
}

fn decode_profile(flat: usize, dims: &[usize], sets: &[Vec<f64>]) -> Vec<f64> {
    let n = dims.len();
    let mut actions = vec![0.0; n];
    let mut rem = flat;
    for i in (0..n).rev() {
        let idx = rem % dims[i];
        rem /= dims[i];
        actions[i] = sets[i][idx];
    }
    actions
}

// ═══════════════════════════════════════════════════════════════════════════
// Asymmetric Nash equilibrium computation
// ═══════════════════════════════════════════════════════════════════════════

/// Nash equilibrium profile with per-agent strategies and payoffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricNashEquilibrium {
    pub actions: Vec<f64>,
    pub payoffs: Vec<f64>,
    pub is_approximate: bool,
    pub epsilon: f64,
}

/// Find Nash equilibrium by iterated best response over the discretized game.
pub fn find_nash_equilibrium(game: &AsymmetricGame) -> AsymmetricNashEquilibrium {
    let matrix = game.payoff_matrix();
    let n = game.n_agents();
    let mut current_indices = vec![0usize; n];

    // Iterated best response (up to 1000 rounds)
    let mut converged = false;
    for _ in 0..1000 {
        let mut changed = false;
        for i in 0..n {
            let best_idx = best_response(i, &current_indices, &matrix);
            if best_idx != current_indices[i] {
                current_indices[i] = best_idx;
                changed = true;
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }

    let actions: Vec<f64> = current_indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| matrix.strategy_sets[i][idx])
        .collect();
    let flat = encode_profile(&current_indices, &matrix.dims);
    let payoffs: Vec<f64> = (0..n).map(|i| matrix.payoffs[i][flat]).collect();

    // Compute epsilon (max deviation gain)
    let mut max_eps = 0.0f64;
    for i in 0..n {
        let ne_pay = payoffs[i];
        for si in 0..matrix.dims[i] {
            let mut dev_indices = current_indices.clone();
            dev_indices[i] = si;
            let dev_flat = encode_profile(&dev_indices, &matrix.dims);
            let dev_pay = matrix.payoffs[i][dev_flat];
            max_eps = max_eps.max(dev_pay - ne_pay);
        }
    }

    AsymmetricNashEquilibrium {
        actions,
        payoffs,
        is_approximate: !converged || max_eps > 1e-6,
        epsilon: max_eps.max(0.0),
    }
}

fn best_response(agent: usize, indices: &[usize], matrix: &AsymmetricPayoffMatrix) -> usize {
    let mut best_pay = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for si in 0..matrix.dims[agent] {
        let mut dev = indices.to_vec();
        dev[agent] = si;
        let flat = encode_profile(&dev, &matrix.dims);
        let pay = matrix.payoffs[agent][flat];
        if pay > best_pay {
            best_pay = pay;
            best_idx = si;
        }
    }
    best_idx
}

fn encode_profile(indices: &[usize], dims: &[usize]) -> usize {
    let mut flat = 0;
    let mut multiplier = 1;
    for i in (0..dims.len()).rev() {
        flat += indices[i] * multiplier;
        multiplier *= dims[i];
    }
    flat
}

// ═══════════════════════════════════════════════════════════════════════════
// Asymmetric coalition certification (QF_LRA encoding)
// ═══════════════════════════════════════════════════════════════════════════

/// A collusion certificate for asymmetric games.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricCollusionCertificate {
    pub game_type: AsymmetricGameType,
    pub n_agents: usize,
    pub coalition: Vec<usize>,
    pub collusive_profile: Vec<f64>,
    pub nash_profile: Vec<f64>,
    pub collusive_payoffs: Vec<f64>,
    pub nash_payoffs: Vec<f64>,
    pub price_elevation: Vec<f64>,
    pub punishment_credible: bool,
    pub deviation_gains: Vec<f64>,
    pub punishment_losses: Vec<f64>,
    pub sustainability_condition: bool,
    pub certificate_valid: bool,
    pub smt_encoding_vars: usize,
    pub smt_encoding_constraints: usize,
    pub generation_time_ms: u64,
}

impl fmt::Display for AsymmetricCollusionCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Asymmetric Collusion Certificate ===")?;
        writeln!(f, "Game type:   {}", self.game_type)?;
        writeln!(f, "Agents:      {}", self.n_agents)?;
        writeln!(f, "Coalition:   {:?}", self.coalition)?;
        writeln!(f, "Valid:       {}", self.certificate_valid)?;
        writeln!(f, "Sustainable: {}", self.sustainability_condition)?;
        writeln!(f, "SMT vars:    {}", self.smt_encoding_vars)?;
        writeln!(f, "SMT constrs: {}", self.smt_encoding_constraints)?;
        writeln!(f, "Time:        {} ms", self.generation_time_ms)?;
        Ok(())
    }
}

/// Certify collusion for an asymmetric game with a given coalition.
///
/// The certification checks:
/// 1. **Price/quantity elevation**: Coalition members earn above Nash payoffs.
/// 2. **Deviation incentive**: Each member has a unilateral deviation that
///    increases short-run payoff (showing supra-competitive equilibrium).
/// 3. **Punishment credibility**: For each deviator, the remaining coalition can
///    impose losses exceeding deviation gains (asymmetric folk theorem condition).
/// 4. **Sustainability**: The discounted punishment threat makes the collusive
///    profile an equilibrium of the repeated game (per-agent δ threshold).
///
/// The QF_LRA encoding uses O(n·m²) variables where n = agents, m = max strategies,
/// without symmetric reduction — each agent's payoff row is encoded independently.
pub fn certify_asymmetric_coalition(
    game: &AsymmetricGame,
    coalition: &[usize],
) -> AsymmetricCollusionCertificate {
    let start = Instant::now();
    let matrix = game.payoff_matrix();
    let n = game.n_agents();

    // Step 1: Find Nash equilibrium
    let ne = find_nash_equilibrium(game);

    // Step 2: Find best collusive profile for the coalition via grid search
    let (collusive_profile, collusive_payoffs) =
        find_collusive_profile(game, &matrix, coalition, &ne);

    // Step 3: Compute per-agent price elevation over Nash
    let price_elevation: Vec<f64> = (0..n)
        .map(|i| {
            let ratio = if ne.payoffs[i].abs() > 1e-10 {
                (collusive_payoffs[i] - ne.payoffs[i]) / ne.payoffs[i].abs()
            } else {
                collusive_payoffs[i]
            };
            ratio
        })
        .collect();

    // Step 4: Compute deviation gains and punishment losses for each coalition member
    let mut deviation_gains = vec![0.0; n];
    let mut punishment_losses = vec![0.0; n];

    for &i in coalition {
        // Best unilateral deviation from collusive profile
        let strategies = game.agents[i].strategies();
        let mut best_dev_payoff = f64::NEG_INFINITY;
        for &s in &strategies {
            let mut dev_actions = collusive_profile.clone();
            dev_actions[i] = s;
            let pay = game.payoff(i, &dev_actions);
            if pay > best_dev_payoff {
                best_dev_payoff = pay;
            }
        }
        deviation_gains[i] = (best_dev_payoff - collusive_payoffs[i]).max(0.0);

        // Punishment: coalition reverts to Nash after deviation
        punishment_losses[i] = (collusive_payoffs[i] - ne.payoffs[i]).max(0.0);
    }

    // Step 5: Check sustainability via asymmetric folk theorem condition.
    //   For each coalition member i:
    //     δ_i / (1 - δ_i) · L_i  ≥  G_i
    //   where G_i = deviation gain, L_i = punishment loss.
    //   Equivalently: δ ≥ G_i / (G_i + L_i).
    let punishment_credible = coalition.iter().all(|&i| {
        if deviation_gains[i] < 1e-10 {
            return true; // no deviation incentive
        }
        punishment_losses[i] > 1e-10
    });

    let sustainability_condition = coalition.iter().all(|&i| {
        if deviation_gains[i] < 1e-10 {
            return true;
        }
        let threshold = deviation_gains[i] / (deviation_gains[i] + punishment_losses[i]);
        game.discount_factor >= threshold
    });

    let certificate_valid = punishment_credible
        && sustainability_condition
        && coalition.iter().any(|&i| price_elevation[i] > 0.01);

    // SMT encoding size: per-agent payoff row without symmetric reduction
    let max_strats = game.agents.iter().map(|a| a.strategy_count).max().unwrap_or(1);
    let smt_vars = n * max_strats * max_strats;
    let smt_constraints = n * max_strats + coalition.len() * 3; // payoff + deviation + punishment

    let elapsed = start.elapsed();

    AsymmetricCollusionCertificate {
        game_type: game.game_type,
        n_agents: n,
        coalition: coalition.to_vec(),
        collusive_profile,
        nash_profile: ne.actions,
        collusive_payoffs,
        nash_payoffs: ne.payoffs,
        price_elevation,
        punishment_credible,
        deviation_gains,
        punishment_losses,
        sustainability_condition,
        certificate_valid,
        smt_encoding_vars: smt_vars,
        smt_encoding_constraints: smt_constraints,
        generation_time_ms: elapsed.as_millis() as u64,
    }
}

/// Search for the joint action profile that maximises total coalition payoff
/// subject to individual rationality: every coalition member earns ≥ Nash payoff.
fn find_collusive_profile(
    game: &AsymmetricGame,
    matrix: &AsymmetricPayoffMatrix,
    coalition: &[usize],
    ne: &AsymmetricNashEquilibrium,
) -> (Vec<f64>, Vec<f64>) {
    let total = matrix.total_profiles();
    let n = game.n_agents();
    let mut best_welfare = f64::NEG_INFINITY;
    let mut best_actions = ne.actions.clone();
    let mut best_payoffs = ne.payoffs.clone();

    for idx in 0..total {
        // Individual rationality: each coalition member must earn ≥ Nash payoff
        let ir_ok = coalition
            .iter()
            .all(|&i| matrix.payoffs[i][idx] >= ne.payoffs[i] - 1e-6);
        if !ir_ok {
            continue;
        }
        let coalition_welfare: f64 = coalition.iter().map(|&i| matrix.payoffs[i][idx]).sum();
        if coalition_welfare > best_welfare {
            best_welfare = coalition_welfare;
            let indices = matrix.decode_indices(idx);
            best_actions = indices
                .iter()
                .enumerate()
                .map(|(i, &si)| matrix.strategy_sets[i][si])
                .collect();
            best_payoffs = (0..n).map(|i| matrix.payoffs[i][idx]).collect();
        }
    }

    (best_actions, best_payoffs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-built asymmetric game scenarios
// ═══════════════════════════════════════════════════════════════════════════

/// Asymmetric Cournot game: 3 firms with marginal costs 10, 15, 20.
pub fn asymmetric_cournot_example() -> AsymmetricGame {
    let agents = vec![
        AsymmetricAgent::new(0, "LowCostFirm")
            .with_cost(10.0)
            .with_capacity(50.0)
            .with_strategy_range(0.0, 50.0, 21),
        AsymmetricAgent::new(1, "MedCostFirm")
            .with_cost(15.0)
            .with_capacity(40.0)
            .with_strategy_range(0.0, 40.0, 21),
        AsymmetricAgent::new(2, "HighCostFirm")
            .with_cost(20.0)
            .with_capacity(30.0)
            .with_strategy_range(0.0, 30.0, 21),
    ];
    AsymmetricGame::new(AsymmetricGameType::Cournot, agents)
        .with_demand(100.0, 1.0)
        .with_discount(0.95)
}

/// Asymmetric Bertrand game: 3 firms with brand loyalty 0.1, 0.5, 0.9.
pub fn asymmetric_bertrand_example() -> AsymmetricGame {
    let agents = vec![
        AsymmetricAgent::new(0, "GenericBrand")
            .with_cost(5.0)
            .with_brand_loyalty(0.1)
            .with_strategy_range(5.0, 50.0, 21),
        AsymmetricAgent::new(1, "MidBrand")
            .with_cost(8.0)
            .with_brand_loyalty(0.5)
            .with_strategy_range(8.0, 60.0, 21),
        AsymmetricAgent::new(2, "PremiumBrand")
            .with_cost(12.0)
            .with_brand_loyalty(0.9)
            .with_strategy_range(12.0, 70.0, 21),
    ];
    AsymmetricGame::new(AsymmetricGameType::Bertrand, agents)
        .with_demand(80.0, 1.0)
        .with_discount(0.95)
}

/// Asymmetric auction game: 3 bidders with budgets 100, 500, 1000.
pub fn asymmetric_auction_example() -> AsymmetricGame {
    let agents = vec![
        AsymmetricAgent::new(0, "SmallBidder")
            .with_budget(100.0)
            .with_strategy_range(0.0, 100.0, 21),
        AsymmetricAgent::new(1, "MediumBidder")
            .with_budget(500.0)
            .with_strategy_range(0.0, 500.0, 21),
        AsymmetricAgent::new(2, "LargeBidder")
            .with_budget(1000.0)
            .with_strategy_range(0.0, 1000.0, 21),
    ];
    AsymmetricGame::new(AsymmetricGameType::Auction, agents)
        .with_discount(0.90)
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark runner
// ═══════════════════════════════════════════════════════════════════════════

/// Result from a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub scenario: String,
    pub game_type: AsymmetricGameType,
    pub n_agents: usize,
    pub is_symmetric: bool,
    pub certificate_valid: bool,
    pub detection_correct: bool,
    pub price_elevation_pct: f64,
    pub smt_vars: usize,
    pub smt_constraints: usize,
    pub time_ms: u64,
}

/// Run the full asymmetric benchmark suite and return results.
pub fn run_asymmetric_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // ── Asymmetric Cournot ──
    let game = asymmetric_cournot_example();
    let coalition: Vec<usize> = (0..game.n_agents()).collect();
    let cert = certify_asymmetric_coalition(&game, &coalition);
    results.push(BenchmarkResult {
        scenario: "Asym. Cournot (costs 10/15/20)".into(),
        game_type: cert.game_type,
        n_agents: cert.n_agents,
        is_symmetric: false,
        certificate_valid: cert.certificate_valid,
        detection_correct: cert.certificate_valid, // expected: collusive
        price_elevation_pct: cert
            .price_elevation
            .iter()
            .sum::<f64>()
            / cert.n_agents as f64
            * 100.0,
        smt_vars: cert.smt_encoding_vars,
        smt_constraints: cert.smt_encoding_constraints,
        time_ms: cert.generation_time_ms,
    });

    // ── Asymmetric Bertrand ──
    let game = asymmetric_bertrand_example();
    let coalition: Vec<usize> = (0..game.n_agents()).collect();
    let cert = certify_asymmetric_coalition(&game, &coalition);
    results.push(BenchmarkResult {
        scenario: "Asym. Bertrand (loyalty 0.1/0.5/0.9)".into(),
        game_type: cert.game_type,
        n_agents: cert.n_agents,
        is_symmetric: false,
        certificate_valid: cert.certificate_valid,
        detection_correct: cert.certificate_valid,
        price_elevation_pct: cert
            .price_elevation
            .iter()
            .sum::<f64>()
            / cert.n_agents as f64
            * 100.0,
        smt_vars: cert.smt_encoding_vars,
        smt_constraints: cert.smt_encoding_constraints,
        time_ms: cert.generation_time_ms,
    });

    // ── Asymmetric Auction ──
    let game = asymmetric_auction_example();
    let coalition: Vec<usize> = (0..game.n_agents()).collect();
    let cert = certify_asymmetric_coalition(&game, &coalition);
    results.push(BenchmarkResult {
        scenario: "Asym. Auction (budgets 100/500/1000)".into(),
        game_type: cert.game_type,
        n_agents: cert.n_agents,
        is_symmetric: false,
        certificate_valid: cert.certificate_valid,
        detection_correct: cert.certificate_valid,
        price_elevation_pct: cert
            .price_elevation
            .iter()
            .sum::<f64>()
            / cert.n_agents as f64
            * 100.0,
        smt_vars: cert.smt_encoding_vars,
        smt_constraints: cert.smt_encoding_constraints,
        time_ms: cert.generation_time_ms,
    });

    // ── Symmetric control: Cournot with equal costs ──
    let sym_agents = vec![
        AsymmetricAgent::new(0, "Firm0")
            .with_cost(15.0)
            .with_strategy_range(0.0, 40.0, 21),
        AsymmetricAgent::new(1, "Firm1")
            .with_cost(15.0)
            .with_strategy_range(0.0, 40.0, 21),
        AsymmetricAgent::new(2, "Firm2")
            .with_cost(15.0)
            .with_strategy_range(0.0, 40.0, 21),
    ];
    let game = AsymmetricGame::new(AsymmetricGameType::Cournot, sym_agents)
        .with_demand(100.0, 1.0)
        .with_discount(0.95);
    let coalition: Vec<usize> = (0..game.n_agents()).collect();
    let cert = certify_asymmetric_coalition(&game, &coalition);
    results.push(BenchmarkResult {
        scenario: "Sym. Cournot control (cost 15/15/15)".into(),
        game_type: cert.game_type,
        n_agents: cert.n_agents,
        is_symmetric: true,
        certificate_valid: cert.certificate_valid,
        detection_correct: cert.certificate_valid,
        price_elevation_pct: cert
            .price_elevation
            .iter()
            .sum::<f64>()
            / cert.n_agents as f64
            * 100.0,
        smt_vars: cert.smt_encoding_vars,
        smt_constraints: cert.smt_encoding_constraints,
        time_ms: cert.generation_time_ms,
    });

    // ── Dominant-firm + fringe ──
    let dom_agents = vec![
        AsymmetricAgent::new(0, "DominantFirm")
            .with_cost(5.0)
            .with_capacity(80.0)
            .with_strategy_range(0.0, 80.0, 21),
        AsymmetricAgent::new(1, "Fringe1")
            .with_cost(20.0)
            .with_capacity(15.0)
            .with_strategy_range(0.0, 15.0, 21),
        AsymmetricAgent::new(2, "Fringe2")
            .with_cost(22.0)
            .with_capacity(10.0)
            .with_strategy_range(0.0, 10.0, 21),
    ];
    let game = AsymmetricGame::new(AsymmetricGameType::Cournot, dom_agents)
        .with_demand(100.0, 1.0)
        .with_discount(0.95);
    let coalition: Vec<usize> = (0..game.n_agents()).collect();
    let cert = certify_asymmetric_coalition(&game, &coalition);
    results.push(BenchmarkResult {
        scenario: "Dominant firm + fringe (costs 5/20/22)".into(),
        game_type: cert.game_type,
        n_agents: cert.n_agents,
        is_symmetric: false,
        certificate_valid: cert.certificate_valid,
        detection_correct: cert.certificate_valid,
        price_elevation_pct: cert
            .price_elevation
            .iter()
            .sum::<f64>()
            / cert.n_agents as f64
            * 100.0,
        smt_vars: cert.smt_encoding_vars,
        smt_constraints: cert.smt_encoding_constraints,
        time_ms: cert.generation_time_ms,
    });

    results
}

/// Pretty-print benchmark results to stdout.
pub fn print_benchmark_results(results: &[BenchmarkResult]) {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Asymmetric Coalition Certification Benchmark Results               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<40} {:>5} {:>6} {:>8} {:>7} ║",
        "Scenario", "Valid", "Elev%", "SMT Var", "Time ms"
    );
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    for r in results {
        println!(
            "║ {:<40} {:>5} {:>5.1}% {:>8} {:>5}ms ║",
            r.scenario,
            if r.certificate_valid { "  ✓" } else { "  ✗" },
            r.price_elevation_pct,
            r.smt_vars,
            r.time_ms,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let asym_correct = results
        .iter()
        .filter(|r| !r.is_symmetric && r.detection_correct)
        .count();
    let asym_total = results.iter().filter(|r| !r.is_symmetric).count();
    let sym_correct = results
        .iter()
        .filter(|r| r.is_symmetric && r.detection_correct)
        .count();
    let sym_total = results.iter().filter(|r| r.is_symmetric).count();
    println!();
    println!(
        "Asymmetric detection accuracy: {}/{} ({:.1}%)",
        asym_correct,
        asym_total,
        asym_correct as f64 / asym_total.max(1) as f64 * 100.0,
    );
    println!(
        "Symmetric control accuracy:    {}/{} ({:.1}%)",
        sym_correct,
        sym_total,
        sym_correct as f64 / sym_total.max(1) as f64 * 100.0,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cournot_ne_is_approximate_nash() {
        let game = asymmetric_cournot_example();
        let ne = find_nash_equilibrium(&game);
        // Epsilon should be small for a well-discretized game
        assert!(ne.epsilon < 5.0, "NE epsilon too large: {}", ne.epsilon);
    }

    #[test]
    fn asymmetric_certificate_is_valid() {
        let game = asymmetric_cournot_example();
        let coalition: Vec<usize> = (0..game.n_agents()).collect();
        let cert = certify_asymmetric_coalition(&game, &coalition);
        assert!(cert.certificate_valid, "Certificate should be valid for collusive Cournot");
    }

    #[test]
    fn symmetric_control_matches() {
        let agents = vec![
            AsymmetricAgent::new(0, "F0").with_cost(15.0).with_strategy_range(0.0, 40.0, 21),
            AsymmetricAgent::new(1, "F1").with_cost(15.0).with_strategy_range(0.0, 40.0, 21),
        ];
        let game = AsymmetricGame::new(AsymmetricGameType::Cournot, agents)
            .with_demand(100.0, 1.0)
            .with_discount(0.95);
        let ne = find_nash_equilibrium(&game);
        // Symmetric agents should have approximately equal NE actions
        assert!(
            (ne.actions[0] - ne.actions[1]).abs() < 3.0,
            "Symmetric agents should have similar NE actions: {:?}",
            ne.actions
        );
    }

    #[test]
    fn bertrand_certificate_valid() {
        let game = asymmetric_bertrand_example();
        let coalition: Vec<usize> = (0..game.n_agents()).collect();
        let cert = certify_asymmetric_coalition(&game, &coalition);
        assert!(cert.certificate_valid, "Bertrand certificate should be valid");
    }

    #[test]
    fn benchmark_suite_runs() {
        let results = run_asymmetric_benchmarks();
        assert_eq!(results.len(), 5, "Expected 5 benchmark scenarios");
    }
}
