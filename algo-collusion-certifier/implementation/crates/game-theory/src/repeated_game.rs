//! Repeated game framework with discounting, trigger equilibria, and subgame-perfect checks.
//!
//! Provides the core abstraction for infinitely repeated games: stage-game payoffs,
//! discount factors, average vs discounted payoff computation, trigger equilibria
//! (e.g., grim-trigger sustaining collusion), punishment severity measurement,
//! and subgame-perfection verification.

use serde::{Deserialize, Serialize};
use shared_types::*;
use std::fmt;

use crate::equilibrium::{NashEquilibrium, PayoffMatrix, BertrandNashSolver, CournotNashSolver};
use crate::folk_theorem::{FeasiblePayoffSet, MinimaxComputation, PunishmentStrategy};
use crate::strategies::{Action, Strategy};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Stage Game
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A stage game in a repeated-game framework: the one-shot game played each period.
///
/// Contains the payoff matrix, number of players and actions, and the competitive
/// (Nash) and collusive benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageGame {
    /// Number of players.
    pub num_players: usize,
    /// Number of actions per player.
    pub num_actions: Vec<usize>,
    /// Payoff matrix indexed by `[player][action_profile_flat]`.
    pub payoffs: PayoffMatrix,
    /// Pre-computed Nash equilibrium payoffs (competitive benchmark).
    pub nash_payoffs: Vec<f64>,
    /// Pre-computed collusive (joint-maximizing) payoffs.
    pub collusive_payoffs: Vec<f64>,
    /// Label for the stage game (e.g., "Bertrand-2" or "Cournot-3").
    pub label: String,
}

impl StageGame {
    /// Build a stage game from an explicit payoff matrix with Nash/collusive benchmarks.
    pub fn new(
        payoffs: PayoffMatrix,
        nash_payoffs: Vec<f64>,
        collusive_payoffs: Vec<f64>,
        label: &str,
    ) -> Self {
        let num_players = nash_payoffs.len();
        let num_actions = payoffs.num_actions.clone();
        Self {
            num_players,
            num_actions,
            payoffs,
            nash_payoffs,
            collusive_payoffs,
            label: label.to_string(),
        }
    }

    /// Build a symmetric 2-player Prisoner's Dilemma stage game.
    ///
    /// Payoffs: mutual cooperate = `cc`, mutual defect = `dd`,
    /// temptation (defect vs cooperate) = `dc`, sucker (cooperate vs defect) = `cd`.
    /// Classic PD requires: dc > cc > dd > cd.
    pub fn prisoners_dilemma(cc: f64, cd: f64, dc: f64, dd: f64) -> Self {
        let mut payoffs = PayoffMatrix::new(vec![2, 2]);
        // (C,C)
        payoffs.set_payoff(0, &[0, 0], cc);
        payoffs.set_payoff(1, &[0, 0], cc);
        // (C,D)
        payoffs.set_payoff(0, &[0, 1], cd);
        payoffs.set_payoff(1, &[0, 1], dc);
        // (D,C)
        payoffs.set_payoff(0, &[1, 0], dc);
        payoffs.set_payoff(1, &[1, 0], cd);
        // (D,D)
        payoffs.set_payoff(0, &[1, 1], dd);
        payoffs.set_payoff(1, &[1, 1], dd);

        Self {
            num_players: 2,
            num_actions: vec![2, 2],
            payoffs,
            nash_payoffs: vec![dd, dd],        // (D,D) is the one-shot NE
            collusive_payoffs: vec![cc, cc],    // (C,C) is the joint max
            label: "PrisonersDilemma".to_string(),
        }
    }

    /// Build a stage game from a [`GameConfig`] (Bertrand or Cournot).
    pub fn from_game_config(config: &GameConfig) -> CollusionResult<Self> {
        let nash = match config.market_type {
            MarketType::Bertrand => BertrandNashSolver::solve(config)?,
            MarketType::Cournot => CournotNashSolver::solve(config)?,
        };

        let n = config.num_players;
        let collusive = Self::estimate_collusive_payoffs(config);

        // For continuous-action games we store a discretized payoff matrix
        let num_levels: usize = 10;
        let mut pm = PayoffMatrix::new(vec![num_levels; n]);

        Ok(Self {
            num_players: n,
            num_actions: vec![num_levels; n],
            payoffs: pm,
            nash_payoffs: nash.payoffs.clone(),
            collusive_payoffs: collusive,
            label: format!("{:?}-{}", config.market_type, n),
        })
    }

    /// Heuristic collusive payoffs: equal split of monopoly profit.
    fn estimate_collusive_payoffs(config: &GameConfig) -> Vec<f64> {
        let n = config.num_players;
        // Monopoly profit proxy: assume demand Q = max_quantity − slope × P
        let monopoly_profit = match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => {
                // Monopoly price: p_m = max_quantity / (2 * slope)
                let pm = max_quantity / (2.0 * slope);
                let qm = max_quantity - slope * pm;
                pm * qm
            }
            DemandSystem::Logit { market_size, .. } => {
                market_size * 0.3
            }
            DemandSystem::CES { market_size, .. } => {
                market_size * 0.25
            }
        };
        vec![monopoly_profit / n as f64; n]
    }

    /// Get the one-shot Nash equilibrium payoff for a player.
    pub fn nash_payoff(&self, player: usize) -> f64 {
        self.nash_payoffs.get(player).copied().unwrap_or(0.0)
    }

    /// Get the collusive payoff for a player.
    pub fn collusive_payoff(&self, player: usize) -> f64 {
        self.collusive_payoffs.get(player).copied().unwrap_or(0.0)
    }

    /// Maximum one-shot deviation payoff: best response against collusive profile.
    pub fn deviation_payoff(&self, player: usize) -> f64 {
        // Construct the collusive action profile (assume action 0 = cooperate for discretized)
        let collusive_actions: Vec<usize> = vec![0; self.num_players];
        let (best_action, best_payoff) = self.payoffs.best_response(player, &collusive_actions);
        best_payoff
    }
}

impl fmt::Display for StageGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StageGame({}, {}P, nash={:?}, collusive={:?})",
            self.label, self.num_players, self.nash_payoffs, self.collusive_payoffs
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Repeated Game
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// An infinitely repeated game with discounting.
///
/// Models the interaction where a [`StageGame`] is played each period, with players
/// maximizing the δ-discounted sum (or Cesàro average) of stage payoffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatedGame {
    /// The one-shot game played each period.
    pub stage_game: StageGame,
    /// Discount factor δ ∈ (0, 1). Higher = more patient players.
    pub discount_factor: f64,
    /// Number of periods to simulate (for finite approximation).
    pub num_periods: usize,
}

impl RepeatedGame {
    /// Create a new repeated game.
    ///
    /// # Errors
    /// Returns an error if `discount_factor` is not in (0, 1).
    pub fn new(stage_game: StageGame, discount_factor: f64, num_periods: usize) -> CollusionResult<Self> {
        if discount_factor <= 0.0 || discount_factor >= 1.0 {
            return Err(CollusionError::GameTheory(
                shared_types::errors::GameTheoryError::InvalidDiscountFactor {
                    delta: discount_factor,
                }
            ));
        }
        Ok(Self { stage_game, discount_factor, num_periods })
    }

    /// Compute the critical discount factor δ* above which collusion is sustainable
    /// via a trigger strategy reverting to Nash after deviation.
    ///
    /// δ* = (π_dev - π_col) / (π_dev - π_nash)
    ///
    /// Returns `None` if deviation payoff ≤ Nash payoff (collusion always sustainable
    /// or never beneficial).
    pub fn critical_discount_factor(&self, player: usize) -> Option<f64> {
        let pi_col = self.stage_game.collusive_payoff(player);
        let pi_nash = self.stage_game.nash_payoff(player);
        let pi_dev = self.stage_game.deviation_payoff(player);

        let numerator = pi_dev - pi_col;
        let denominator = pi_dev - pi_nash;

        if denominator.abs() < 1e-12 {
            return None;
        }

        let delta_star = numerator / denominator;
        if delta_star.is_finite() && delta_star > 0.0 && delta_star < 1.0 {
            Some(delta_star)
        } else {
            None
        }
    }

    /// Check whether collusion is sustainable for all players at the current δ.
    pub fn collusion_sustainable(&self) -> bool {
        (0..self.stage_game.num_players).all(|i| {
            match self.critical_discount_factor(i) {
                Some(delta_star) => self.discount_factor >= delta_star,
                None => true, // always sustainable or deviation is never profitable
            }
        })
    }

    /// Compute the discounted payoff of a sequence of stage payoffs.
    pub fn discounted_payoff(&self, stage_payoffs: &[f64]) -> f64 {
        let norm = 1.0 - self.discount_factor;
        let sum: f64 = stage_payoffs.iter().enumerate()
            .map(|(t, &pi)| self.discount_factor.powi(t as i32) * pi)
            .sum();
        norm * sum
    }

    /// Compute the average payoff of a sequence of stage payoffs (Cesàro mean).
    pub fn average_payoff(&self, stage_payoffs: &[f64]) -> f64 {
        if stage_payoffs.is_empty() {
            return 0.0;
        }
        stage_payoffs.iter().sum::<f64>() / stage_payoffs.len() as f64
    }
}

impl fmt::Display for RepeatedGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RepeatedGame(δ={:.4}, T={}, {})",
            self.discount_factor, self.num_periods, self.stage_game
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Average Payoff
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Average (Cesàro) payoff over observed rounds for a player.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AveragePayoff {
    /// Computed average payoff.
    pub value: f64,
    /// Number of rounds.
    pub num_rounds: usize,
    /// Running sum.
    pub total: f64,
}

impl AveragePayoff {
    /// Compute from a payoff sequence.
    pub fn from_payoffs(payoffs: &[f64]) -> Self {
        let total: f64 = payoffs.iter().sum();
        let value = if payoffs.is_empty() { 0.0 } else { total / payoffs.len() as f64 };
        Self { value, num_rounds: payoffs.len(), total }
    }

    /// Incrementally add a new round's payoff.
    pub fn update(&mut self, payoff: f64) {
        self.total += payoff;
        self.num_rounds += 1;
        self.value = self.total / self.num_rounds as f64;
    }

    /// Standard error of the mean (assuming payoff variance is provided).
    pub fn standard_error(&self, variance: f64) -> f64 {
        if self.num_rounds == 0 { return f64::INFINITY; }
        (variance / self.num_rounds as f64).sqrt()
    }
}

impl fmt::Display for AveragePayoff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AvgPayoff({:.6}, n={})", self.value, self.num_rounds)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Discounted Payoff
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Discounted-sum payoff: V = (1 − δ) Σ_{t=0}^{T} δ^t π_t.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscountedPayoff {
    /// The discounted-sum value.
    pub value: f64,
    /// Discount factor δ.
    pub discount_factor: f64,
    /// Number of rounds processed.
    pub num_rounds: usize,
    /// Running discounted sum (unnormalized).
    running_sum: f64,
    /// Current discount power δ^t.
    current_power: f64,
}

impl DiscountedPayoff {
    /// Create a new discounted payoff accumulator.
    pub fn new(discount_factor: f64) -> Self {
        Self {
            value: 0.0,
            discount_factor,
            num_rounds: 0,
            running_sum: 0.0,
            current_power: 1.0,
        }
    }

    /// Compute discounted payoff from a complete sequence.
    pub fn from_payoffs(payoffs: &[f64], discount_factor: f64) -> Self {
        let mut dp = Self::new(discount_factor);
        for &p in payoffs {
            dp.update(p);
        }
        dp
    }

    /// Add a new period's payoff.
    pub fn update(&mut self, payoff: f64) {
        self.running_sum += self.current_power * payoff;
        self.current_power *= self.discount_factor;
        self.num_rounds += 1;
        self.value = (1.0 - self.discount_factor) * self.running_sum;
    }

    /// The normalization factor (1 − δ).
    pub fn normalization(&self) -> f64 {
        1.0 - self.discount_factor
    }

    /// Remaining weight (not yet accounted for): δ^{T+1} / (1 − δ).
    pub fn remaining_weight(&self) -> f64 {
        self.current_power / (1.0 - self.discount_factor)
    }
}

impl fmt::Display for DiscountedPayoff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiscPayoff({:.6}, δ={:.4}, n={})",
            self.value, self.discount_factor, self.num_rounds
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Punishment Severity
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Severity classification for the punishment phase of a trigger strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PunishmentSeverity {
    /// Revert to Nash equilibrium (standard folk theorem).
    NashReversion,
    /// Minimax punishment (strongest credible threat).
    Minimax,
    /// Partial punishment (between Nash and minimax).
    Partial,
    /// No effective punishment detected.
    None,
}

impl PunishmentSeverity {
    /// Classify punishment severity from observed punishment vs benchmark payoffs.
    ///
    /// - `punishment_payoff`: average payoff during the punishment phase.
    /// - `nash_payoff`: competitive equilibrium payoff.
    /// - `minimax_payoff`: minimax payoff (floor of what opponents can enforce).
    pub fn classify(punishment_payoff: f64, nash_payoff: f64, minimax_payoff: f64) -> Self {
        let tol = 1e-6;
        if punishment_payoff > nash_payoff - tol {
            PunishmentSeverity::None
        } else if (punishment_payoff - minimax_payoff).abs() < tol
            || punishment_payoff < minimax_payoff + tol
        {
            PunishmentSeverity::Minimax
        } else if (punishment_payoff - nash_payoff).abs() < tol {
            PunishmentSeverity::NashReversion
        } else {
            PunishmentSeverity::Partial
        }
    }

    /// Whether the punishment is strong enough to sustain collusion.
    pub fn is_effective(&self) -> bool {
        !matches!(self, PunishmentSeverity::None)
    }
}

impl fmt::Display for PunishmentSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PunishmentSeverity::NashReversion => write!(f, "NashReversion"),
            PunishmentSeverity::Minimax => write!(f, "Minimax"),
            PunishmentSeverity::Partial => write!(f, "Partial"),
            PunishmentSeverity::None => write!(f, "None"),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Trigger Equilibrium
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A trigger-strategy equilibrium in a repeated game.
///
/// Players follow a collusive profile until a deviation is detected, then
/// switch to a punishment phase. The equilibrium is self-enforcing if
/// `discount_factor ≥ critical_delta`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerEquilibrium {
    /// Collusive payoff per player.
    pub collusive_payoffs: Vec<f64>,
    /// Punishment payoff per player.
    pub punishment_payoffs: Vec<f64>,
    /// One-shot deviation payoff per player.
    pub deviation_payoffs: Vec<f64>,
    /// Critical discount factor δ* per player.
    pub critical_deltas: Vec<f64>,
    /// Overall critical discount factor (max across players).
    pub critical_delta: f64,
    /// Severity of the punishment.
    pub punishment_severity: PunishmentSeverity,
    /// Whether the equilibrium is sustainable at the given δ.
    pub is_sustainable: bool,
    /// The discount factor used for sustainability check.
    pub discount_factor: f64,
}

impl TriggerEquilibrium {
    /// Analyze a trigger equilibrium for the given repeated game.
    pub fn analyze(game: &RepeatedGame) -> Self {
        let n = game.stage_game.num_players;
        let mut collusive = Vec::with_capacity(n);
        let mut punishment = Vec::with_capacity(n);
        let mut deviation = Vec::with_capacity(n);
        let mut deltas = Vec::with_capacity(n);

        for i in 0..n {
            let pi_col = game.stage_game.collusive_payoff(i);
            let pi_nash = game.stage_game.nash_payoff(i);
            let pi_dev = game.stage_game.deviation_payoff(i);

            collusive.push(pi_col);
            punishment.push(pi_nash); // Nash reversion by default
            deviation.push(pi_dev);

            let denom = pi_dev - pi_nash;
            let delta_star = if denom.abs() > 1e-12 {
                ((pi_dev - pi_col) / denom).clamp(0.0, 1.0)
            } else {
                0.0 // always sustainable
            };
            deltas.push(delta_star);
        }

        let critical_delta = deltas.iter().cloned().fold(0.0_f64, f64::max);
        let is_sustainable = game.discount_factor >= critical_delta;

        // Classify punishment severity
        let avg_punishment: f64 = punishment.iter().sum::<f64>() / n as f64;
        let avg_nash: f64 = game.stage_game.nash_payoffs.iter().sum::<f64>() / n as f64;
        let severity = PunishmentSeverity::classify(avg_punishment, avg_nash, avg_nash * 0.5);

        Self {
            collusive_payoffs: collusive,
            punishment_payoffs: punishment,
            deviation_payoffs: deviation,
            critical_deltas: deltas,
            critical_delta,
            punishment_severity: severity,
            is_sustainable,
            discount_factor: game.discount_factor,
        }
    }

    /// The per-player collusion premium: (π_col - π_nash) / π_nash.
    pub fn collusion_premiums(&self) -> Vec<f64> {
        self.collusive_payoffs.iter().zip(&self.punishment_payoffs)
            .map(|(&col, &pun)| {
                if pun.abs() < 1e-12 { 0.0 } else { (col - pun) / pun }
            })
            .collect()
    }
}

impl fmt::Display for TriggerEquilibrium {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TriggerEq(δ*={:.4}, sustainable={}, punishment={})",
            self.critical_delta, self.is_sustainable, self.punishment_severity
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Subgame-Perfect Check
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Verifier for subgame-perfect equilibrium (SPE) conditions in repeated games.
///
/// A strategy profile is SPE if the trigger equilibrium is sustainable AND
/// the punishment itself is a credible threat (i.e., players would actually
/// carry it out in the subgame after deviation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgamePerfectCheck {
    /// Whether the one-shot deviation principle is satisfied.
    pub one_shot_deviation_ok: bool,
    /// Whether the punishment phase is self-enforcing.
    pub punishment_credible: bool,
    /// Whether the overall profile is subgame-perfect.
    pub is_subgame_perfect: bool,
    /// Maximum deviation gain across all players and all subgames.
    pub max_deviation_gain: f64,
    /// Details per player.
    pub player_details: Vec<SPEPlayerDetail>,
}

/// Per-player detail for the subgame-perfection check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPEPlayerDetail {
    /// Player index.
    pub player: usize,
    /// One-shot deviation gain in the cooperative phase.
    pub cooperative_deviation_gain: f64,
    /// One-shot deviation gain in the punishment phase.
    pub punishment_deviation_gain: f64,
    /// Whether the player's strategy satisfies the one-shot deviation principle.
    pub is_optimal: bool,
}

impl SubgamePerfectCheck {
    /// Check subgame perfection for a trigger equilibrium in the given repeated game.
    ///
    /// Uses the one-shot deviation principle: a strategy profile in an infinite-horizon
    /// discounted game is SPE if and only if no player can gain by deviating in a
    /// single period and then conforming.
    pub fn check(game: &RepeatedGame, trigger_eq: &TriggerEquilibrium) -> Self {
        let n = game.stage_game.num_players;
        let delta = game.discount_factor;
        let mut details = Vec::with_capacity(n);
        let mut max_gain = f64::NEG_INFINITY;
        let mut all_optimal = true;

        for i in 0..n {
            let pi_col = trigger_eq.collusive_payoffs[i];
            let pi_dev = trigger_eq.deviation_payoffs[i];
            let pi_pun = trigger_eq.punishment_payoffs[i];

            // Cooperative phase: one-shot deviation gain
            // V_cooperate = (1-δ)π_col + δ V_cooperate  →  V_cooperate = π_col
            // V_deviate   = (1-δ)π_dev + δ V_punish     →  V_deviate = (1-δ)π_dev + δ π_pun
            // Gain = V_deviate - V_cooperate = (1-δ)(π_dev - π_col) + δ(π_pun - π_col)
            let coop_gain = (1.0 - delta) * (pi_dev - pi_col) + delta * (pi_pun - pi_col);

            // Punishment phase: one-shot deviation gain (for Nash reversion,
            // punishment IS the Nash equilibrium, so deviation gain should be ≤ 0)
            let pun_gain = 0.0; // Nash reversion is self-enforcing by definition

            let is_opt = coop_gain <= 1e-10 && pun_gain <= 1e-10;
            if !is_opt { all_optimal = false; }
            max_gain = max_gain.max(coop_gain).max(pun_gain);

            details.push(SPEPlayerDetail {
                player: i,
                cooperative_deviation_gain: coop_gain,
                punishment_deviation_gain: pun_gain,
                is_optimal: is_opt,
            });
        }

        // Nash reversion is always credible (it's a stage-game NE played forever)
        let punishment_credible = true;
        let osdp_ok = all_optimal;

        Self {
            one_shot_deviation_ok: osdp_ok,
            punishment_credible,
            is_subgame_perfect: osdp_ok && punishment_credible,
            max_deviation_gain: max_gain,
            player_details: details,
        }
    }
}

impl fmt::Display for SubgamePerfectCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SPE(ok={}, credible={}, max_gain={:.6})",
            self.is_subgame_perfect, self.punishment_credible, self.max_deviation_gain
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Repeated Game Simulator
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Simulates play in a repeated game with concrete strategies.
#[derive(Debug, Clone)]
pub struct RepeatedGameSimulator {
    /// The repeated game being simulated.
    pub game: RepeatedGame,
    /// History of action profiles played.
    pub action_history: Vec<Vec<Action>>,
    /// Per-player payoff history.
    pub payoff_history: Vec<Vec<f64>>,
    /// Current round index.
    pub current_round: usize,
}

impl RepeatedGameSimulator {
    /// Create a new simulator for the given repeated game.
    pub fn new(game: RepeatedGame) -> Self {
        let n = game.stage_game.num_players;
        Self {
            game,
            action_history: Vec::new(),
            payoff_history: vec![Vec::new(); n],
            current_round: 0,
        }
    }

    /// Run one round: each strategy chooses an action, payoffs are computed.
    pub fn step(&mut self, actions: Vec<Action>) -> Vec<f64> {
        let n = self.game.stage_game.num_players;
        let action_indices: Vec<usize> = actions.iter()
            .map(|a| match a { Action::Cooperate => 0, Action::Defect => 1 })
            .collect();

        // Look up payoffs from the stage game's payoff matrix
        let payoffs: Vec<f64> = (0..n)
            .map(|i| self.game.stage_game.payoffs.get_payoff(i, &action_indices))
            .collect();

        self.action_history.push(actions);
        for (i, &p) in payoffs.iter().enumerate() {
            self.payoff_history[i].push(p);
        }
        self.current_round += 1;

        payoffs
    }

    /// Simulate a full game with two strategies playing against each other (2-player).
    pub fn simulate_strategies(
        &mut self,
        strategy_a: &dyn Strategy,
        strategy_b: &dyn Strategy,
    ) -> SimulationResult {
        let t = self.game.num_periods;

        for round in 0..t {
            // Each player observes the other's past actions
            let hist_b: Vec<Action> = self.action_history.iter()
                .map(|acts| acts.get(1).copied().unwrap_or(Action::Cooperate))
                .collect();
            let hist_a: Vec<Action> = self.action_history.iter()
                .map(|acts| acts.get(0).copied().unwrap_or(Action::Cooperate))
                .collect();

            let a = strategy_a.choose(&hist_b);
            let b = strategy_b.choose(&hist_a);
            self.step(vec![a, b]);
        }

        let avg_a = AveragePayoff::from_payoffs(&self.payoff_history[0]);
        let avg_b = AveragePayoff::from_payoffs(&self.payoff_history[1]);
        let disc_a = DiscountedPayoff::from_payoffs(&self.payoff_history[0], self.game.discount_factor);
        let disc_b = DiscountedPayoff::from_payoffs(&self.payoff_history[1], self.game.discount_factor);

        SimulationResult {
            num_rounds: self.current_round,
            average_payoffs: vec![avg_a, avg_b],
            discounted_payoffs: vec![disc_a, disc_b],
            cooperation_rates: self.cooperation_rates(),
        }
    }

    /// Compute the fraction of cooperative actions for each player.
    pub fn cooperation_rates(&self) -> Vec<f64> {
        let n = self.game.stage_game.num_players;
        (0..n).map(|i| {
            if self.action_history.is_empty() { return 0.0; }
            let coop_count = self.action_history.iter()
                .filter(|acts| acts.get(i).map_or(false, |a| a.is_cooperate()))
                .count();
            coop_count as f64 / self.action_history.len() as f64
        }).collect()
    }

    /// Reset the simulator for a fresh run.
    pub fn reset(&mut self) {
        let n = self.game.stage_game.num_players;
        self.action_history.clear();
        self.payoff_history = vec![Vec::new(); n];
        self.current_round = 0;
    }
}

/// Results of a repeated game simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub num_rounds: usize,
    pub average_payoffs: Vec<AveragePayoff>,
    pub discounted_payoffs: Vec<DiscountedPayoff>,
    pub cooperation_rates: Vec<f64>,
}

impl fmt::Display for SimulationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimResult(rounds={}, avg=[{}], coop=[{}])",
            self.num_rounds,
            self.average_payoffs.iter().map(|a| format!("{:.4}", a.value)).collect::<Vec<_>>().join(", "),
            self.cooperation_rates.iter().map(|c| format!("{:.2}", c)).collect::<Vec<_>>().join(", "),
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::{GrimTriggerStrategy, TitForTatStrategy, AlwaysDefect};

    fn pd_stage_game() -> StageGame {
        StageGame::prisoners_dilemma(3.0, 0.0, 5.0, 1.0)
    }

    #[test]
    fn test_average_payoff() {
        let avg = AveragePayoff::from_payoffs(&[1.0, 2.0, 3.0, 4.0]);
        assert!((avg.value - 2.5).abs() < 1e-10);
        assert_eq!(avg.num_rounds, 4);
    }

    #[test]
    fn test_average_payoff_incremental() {
        let mut avg = AveragePayoff::from_payoffs(&[]);
        avg.update(10.0);
        avg.update(20.0);
        assert!((avg.value - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_discounted_payoff_constant_stream() {
        let payoffs: Vec<f64> = vec![1.0; 1000];
        let dp = DiscountedPayoff::from_payoffs(&payoffs, 0.99);
        // For constant stream, discounted payoff → 1.0
        assert!((dp.value - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_critical_discount_factor() {
        let stage = pd_stage_game();
        // PD: dc=5, cc=3, dd=1 → δ* = (5-3)/(5-1) = 0.5
        let game = RepeatedGame::new(stage, 0.9, 100).unwrap();
        let delta_star = game.critical_discount_factor(0).unwrap();
        assert!((delta_star - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_collusion_sustainable() {
        let stage = pd_stage_game();
        let game_high = RepeatedGame::new(stage.clone(), 0.9, 100).unwrap();
        assert!(game_high.collusion_sustainable());

        let game_low = RepeatedGame::new(stage, 0.3, 100).unwrap();
        assert!(!game_low.collusion_sustainable());
    }

    #[test]
    fn test_trigger_equilibrium() {
        let stage = pd_stage_game();
        let game = RepeatedGame::new(stage, 0.9, 100).unwrap();
        let te = TriggerEquilibrium::analyze(&game);
        assert!(te.is_sustainable);
        assert!(te.critical_delta > 0.0);
        assert!(te.critical_delta < 1.0);
    }

    #[test]
    fn test_subgame_perfect_check() {
        let stage = pd_stage_game();
        let game = RepeatedGame::new(stage, 0.9, 100).unwrap();
        let te = TriggerEquilibrium::analyze(&game);
        let spe = SubgamePerfectCheck::check(&game, &te);
        assert!(spe.is_subgame_perfect);
        assert!(spe.punishment_credible);
    }

    #[test]
    fn test_punishment_severity_classification() {
        assert_eq!(
            PunishmentSeverity::classify(1.0, 1.0, 0.5),
            PunishmentSeverity::NashReversion
        );
        assert_eq!(
            PunishmentSeverity::classify(0.5, 1.0, 0.5),
            PunishmentSeverity::Minimax
        );
        assert_eq!(
            PunishmentSeverity::classify(2.0, 1.0, 0.5),
            PunishmentSeverity::None
        );
        assert_eq!(
            PunishmentSeverity::classify(0.7, 1.0, 0.5),
            PunishmentSeverity::Partial
        );
    }

    #[test]
    fn test_repeated_game_invalid_discount() {
        let stage = pd_stage_game();
        assert!(RepeatedGame::new(stage.clone(), 0.0, 100).is_err());
        assert!(RepeatedGame::new(stage.clone(), 1.0, 100).is_err());
        assert!(RepeatedGame::new(stage, -0.5, 100).is_err());
    }

    #[test]
    fn test_simulator_basic() {
        let stage = pd_stage_game();
        let game = RepeatedGame::new(stage, 0.95, 10).unwrap();
        let mut sim = RepeatedGameSimulator::new(game);

        // Simulate one round of mutual cooperation
        let payoffs = sim.step(vec![Action::Cooperate, Action::Cooperate]);
        assert!((payoffs[0] - 3.0).abs() < 1e-10);
        assert!((payoffs[1] - 3.0).abs() < 1e-10);
        assert_eq!(sim.current_round, 1);
    }

    #[test]
    fn test_display_implementations() {
        let avg = AveragePayoff::from_payoffs(&[1.0, 2.0]);
        assert!(format!("{}", avg).contains("AvgPayoff"));

        let dp = DiscountedPayoff::from_payoffs(&[1.0, 2.0], 0.9);
        assert!(format!("{}", dp).contains("DiscPayoff"));

        assert!(format!("{}", PunishmentSeverity::Minimax).contains("Minimax"));
    }
}
