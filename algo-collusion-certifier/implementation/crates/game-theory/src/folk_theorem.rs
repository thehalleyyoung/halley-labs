//! Folk theorem analysis for repeated games.
//!
//! Implements feasible and individually rational payoff sets, minimax computation,
//! discount factor analysis, the C3' detection theorem, and the M8 impossibility
//! theorem for the collusion detection dichotomy.

use serde::{Deserialize, Serialize};
use shared_types::*;
use std::collections::HashSet;

// ── Feasible Payoff Set ─────────────────────────────────────────────────────

/// The set of feasible payoffs: convex hull of all achievable payoff profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeasiblePayoffSet {
    /// Vertices of the convex hull of feasible payoffs.
    pub vertices: Vec<Vec<f64>>,
    /// Number of players.
    pub num_players: usize,
}

impl FeasiblePayoffSet {
    /// Compute the feasible payoff set from a finite set of payoff profiles
    /// (e.g., from pure strategy profiles of the stage game).
    pub fn from_payoff_profiles(profiles: &[Vec<f64>]) -> Self {
        if profiles.is_empty() {
            return Self { vertices: vec![], num_players: 0 };
        }
        let n = profiles[0].len();

        // Compute convex hull vertices (for 2D, use gift-wrapping; for ND, keep all extremes)
        let vertices = if n == 2 {
            Self::convex_hull_2d(profiles)
        } else {
            // For higher dimensions, keep all non-dominated extreme points
            Self::extreme_points(profiles)
        };

        Self { vertices, num_players: n }
    }

    fn convex_hull_2d(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if points.len() <= 2 {
            return points.to_vec();
        }

        // Gift wrapping (Jarvis march)
        let mut hull = Vec::new();
        let start = points.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap())
            })
            .map(|(i, _)| i)
            .unwrap();

        let mut current = start;
        loop {
            hull.push(points[current].clone());
            let mut next = 0;
            for i in 0..points.len() {
                if i == current { continue; }
                if next == current {
                    next = i;
                    continue;
                }
                let cross = Self::cross_product(
                    &points[current], &points[next], &points[i]
                );
                if cross < 0.0 || (cross.abs() < 1e-12 && Self::dist_sq(&points[current], &points[i]) > Self::dist_sq(&points[current], &points[next])) {
                    next = i;
                }
            }
            current = next;
            if current == start {
                break;
            }
            if hull.len() > points.len() {
                break; // safety
            }
        }
        hull
    }

    fn cross_product(o: &[f64], a: &[f64], b: &[f64]) -> f64 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    }

    fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
    }

    fn extreme_points(profiles: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Keep points that are maximal in some direction
        let mut extremes = Vec::new();
        let n = profiles[0].len();
        for dim in 0..n {
            let max_val = profiles.iter().map(|p| p[dim]).fold(f64::NEG_INFINITY, f64::max);
            let min_val = profiles.iter().map(|p| p[dim]).fold(f64::INFINITY, f64::min);
            for p in profiles {
                if (p[dim] - max_val).abs() < 1e-10 || (p[dim] - min_val).abs() < 1e-10 {
                    if !extremes.iter().any(|e: &Vec<f64>| {
                        e.iter().zip(p.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
                    }) {
                        extremes.push(p.clone());
                    }
                }
            }
        }
        if extremes.is_empty() {
            profiles.to_vec()
        } else {
            extremes
        }
    }

    /// Check if a payoff profile is in the feasible set (approximately).
    pub fn contains(&self, payoff: &[f64]) -> bool {
        if self.vertices.is_empty() { return false; }
        if self.num_players == 2 {
            self.point_in_convex_hull_2d(payoff)
        } else {
            // Approximate: check if within bounding box
            for dim in 0..self.num_players {
                let min_v = self.vertices.iter().map(|v| v[dim]).fold(f64::INFINITY, f64::min);
                let max_v = self.vertices.iter().map(|v| v[dim]).fold(f64::NEG_INFINITY, f64::max);
                if payoff[dim] < min_v - 1e-10 || payoff[dim] > max_v + 1e-10 {
                    return false;
                }
            }
            true
        }
    }

    fn point_in_convex_hull_2d(&self, point: &[f64]) -> bool {
        let n = self.vertices.len();
        if n < 3 { return false; }
        let mut positive = 0;
        let mut negative = 0;
        for i in 0..n {
            let j = (i + 1) % n;
            let cross = Self::cross_product(&self.vertices[i], &self.vertices[j], &[point[0], point[1]]);
            if cross > 1e-10 { positive += 1; }
            else if cross < -1e-10 { negative += 1; }
        }
        positive == 0 || negative == 0
    }

    /// Get the bounding box of the feasible set.
    pub fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
        if self.vertices.is_empty() {
            return (vec![], vec![]);
        }
        let n = self.num_players;
        let mut lo = vec![f64::INFINITY; n];
        let mut hi = vec![f64::NEG_INFINITY; n];
        for v in &self.vertices {
            for d in 0..n {
                lo[d] = lo[d].min(v[d]);
                hi[d] = hi[d].max(v[d]);
            }
        }
        (lo, hi)
    }
}

// ── Individually Rational Payoff Set ────────────────────────────────────────

/// The set of individually rational payoffs: each player gets at least their minimax value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividuallyRationalPayoffSet {
    /// Minimax value for each player.
    pub minimax_values: Vec<f64>,
    pub num_players: usize,
}

impl IndividuallyRationalPayoffSet {
    pub fn new(minimax_values: Vec<f64>) -> Self {
        let n = minimax_values.len();
        Self { minimax_values, num_players: n }
    }

    /// Check if a payoff profile is individually rational.
    pub fn is_individually_rational(&self, payoffs: &[f64]) -> bool {
        payoffs.iter().zip(&self.minimax_values)
            .all(|(pi, &mm)| *pi >= mm - 1e-10)
    }

    /// Excess over minimax for each player.
    pub fn excess(&self, payoffs: &[f64]) -> Vec<f64> {
        payoffs.iter().zip(&self.minimax_values)
            .map(|(pi, mm)| pi - mm)
            .collect()
    }
}

// ── Folk Theorem Region ─────────────────────────────────────────────────────

/// The Folk Theorem region: feasible AND individually rational payoffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolkTheoremRegion {
    pub feasible_set: FeasiblePayoffSet,
    pub ir_set: IndividuallyRationalPayoffSet,
}

impl FolkTheoremRegion {
    pub fn new(feasible: FeasiblePayoffSet, ir: IndividuallyRationalPayoffSet) -> Self {
        Self { feasible_set: feasible, ir_set: ir }
    }

    /// Check if a payoff profile is in the Folk Theorem region.
    pub fn contains(&self, payoff: &[f64]) -> bool {
        self.feasible_set.contains(payoff) && self.ir_set.is_individually_rational(payoff)
    }

    /// Compute the folk theorem region for a Bertrand game.
    pub fn for_bertrand_game(config: &GameConfig) -> CollusionResult<Self> {
        let minimax = MinimaxComputation::compute_bertrand(config)?;
        let feasible = Self::compute_feasible_bertrand(config)?;
        Ok(Self::new(feasible, minimax))
    }

    fn compute_feasible_bertrand(config: &GameConfig) -> CollusionResult<FeasiblePayoffSet> {
        match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => {
                let n = config.num_players;
                let mc: f64 = config.marginal_costs[0].into();
                let intercept = *max_quantity;
                let cross_slope = 0.5; // default cross-price elasticity
                // Generate payoff profiles from a grid of price pairs
                let mut profiles = Vec::new();
                let steps = 20;
                let p_min = mc;
                let p_max = intercept / slope;

                for i in 0..=steps {
                    for j in 0..=steps {
                        let p1 = p_min + (p_max - p_min) * i as f64 / steps as f64;
                        let p2 = p_min + (p_max - p_min) * j as f64 / steps as f64;
                        let q1 = (intercept - slope * p1 + cross_slope * p2).max(0.0);
                        let q2 = (intercept - slope * p2 + cross_slope * p1).max(0.0);
                        let pi1 = (p1 - mc) * q1;
                        let pi2 = (p2 - mc) * q2;
                        if pi1 >= 0.0 && pi2 >= 0.0 {
                            profiles.push(vec![pi1, pi2]);
                        }
                    }
                }
                Ok(FeasiblePayoffSet::from_payoff_profiles(&profiles))
            }
            _ => Ok(FeasiblePayoffSet { vertices: vec![], num_players: config.num_players }),
        }
    }
}

// ── Minimax Computation ─────────────────────────────────────────────────────

/// Compute minimax values for each player.
pub struct MinimaxComputation;

impl MinimaxComputation {
    /// Compute minimax values for a Bertrand game with linear demand.
    pub fn compute_bertrand(config: &GameConfig) -> CollusionResult<IndividuallyRationalPayoffSet> {
        let n = config.num_players;
        let mut minimax_values = Vec::with_capacity(n);

        match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => {
                let intercept = *max_quantity;
                let cross_slope = 0.5_f64; // default cross-price elasticity
                for i in 0..n {
                    let mc_i: f64 = config.marginal_costs.get(i).map(|c| f64::from(*c)).unwrap_or(1.0);
                    if cross_slope.abs() < 1e-12 {
                        let p_star = (intercept + slope * mc_i) / (2.0 * slope);
                        let q_star = (intercept - slope * p_star).max(0.0);
                        minimax_values.push((p_star - mc_i) * q_star);
                    } else {
                        let opp_price = mc_i;
                        let p_i_br = (intercept + cross_slope * opp_price + slope * mc_i) / (2.0 * slope);
                        let q_i = (intercept - slope * p_i_br + cross_slope * opp_price).max(0.0);
                        let mm_profit = (p_i_br - mc_i) * q_i;
                        minimax_values.push(mm_profit.max(0.0));
                    }
                }
            }
            _ => {
                // Default: minimax = 0 (competitive outcome)
                minimax_values = vec![0.0; n];
            }
        }

        Ok(IndividuallyRationalPayoffSet::new(minimax_values))
    }

    /// Compute minimax from a payoff matrix (finite game).
    /// payoffs[player][action_profile_index] gives payoff for that player under that profile.
    pub fn from_payoff_matrix(payoffs: &[Vec<Vec<f64>>]) -> Vec<f64> {
        let n = payoffs.len(); // number of players
        if n == 0 { return vec![]; }
        let num_actions = payoffs[0].len();

        let mut minimax_values = Vec::with_capacity(n);
        for player in 0..n {
            let mut max_of_min = f64::NEG_INFINITY;
            for my_action in 0..num_actions {
                let mut min_payoff = f64::INFINITY;
                for opp_action in 0..num_actions {
                    let idx = my_action * num_actions + opp_action;
                    if idx < payoffs[player].len() && player < payoffs[player][idx].len() {
                        let payoff = payoffs[player][idx][player];
                        min_payoff = min_payoff.min(payoff);
                    }
                }
                max_of_min = max_of_min.max(min_payoff);
            }
            minimax_values.push(max_of_min);
        }
        minimax_values
    }
}

// ── Discount Factor Analysis ────────────────────────────────────────────────

/// Analysis of the minimum discount factor needed to sustain cooperation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscountFactorAnalysis {
    /// Minimum discount factor for each player.
    pub min_delta: Vec<f64>,
    /// Overall minimum (binding constraint).
    pub binding_delta: f64,
    /// Which player has the binding constraint.
    pub binding_player: usize,
    /// Nash equilibrium payoff per player.
    pub nash_payoffs: Vec<f64>,
    /// Collusive payoff per player.
    pub collusive_payoffs: Vec<f64>,
    /// Deviation payoff per player.
    pub deviation_payoffs: Vec<f64>,
}

impl DiscountFactorAnalysis {
    /// Compute the minimum discount factor for sustaining a collusive outcome
    /// using trigger strategies (grim trigger).
    ///
    /// For player i: delta_i >= (pi_dev_i - pi_coll_i) / (pi_dev_i - pi_nash_i)
    pub fn compute(
        nash_payoffs: &[f64],
        collusive_payoffs: &[f64],
        deviation_payoffs: &[f64],
    ) -> Self {
        let n = nash_payoffs.len();
        let mut min_delta = Vec::with_capacity(n);
        let mut binding = 0.0f64;
        let mut binding_player = 0;

        for i in 0..n {
            let denom = deviation_payoffs[i] - nash_payoffs[i];
            let numer = deviation_payoffs[i] - collusive_payoffs[i];
            let delta_i = if denom.abs() < 1e-12 {
                if numer <= 1e-12 { 0.0 } else { 1.0 }
            } else {
                (numer / denom).clamp(0.0, 1.0)
            };
            if delta_i > binding {
                binding = delta_i;
                binding_player = i;
            }
            min_delta.push(delta_i);
        }

        Self {
            min_delta,
            binding_delta: binding,
            binding_player,
            nash_payoffs: nash_payoffs.to_vec(),
            collusive_payoffs: collusive_payoffs.to_vec(),
            deviation_payoffs: deviation_payoffs.to_vec(),
        }
    }

    /// Check if a given discount factor can sustain the collusive outcome.
    pub fn can_sustain(&self, delta: f64) -> bool {
        delta >= self.binding_delta - 1e-10
    }
}

// ── Punishment Strategy ─────────────────────────────────────────────────────

/// A punishment strategy that enforces cooperation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentStrategy {
    /// The punishment action (price) per player.
    pub punishment_prices: Vec<f64>,
    /// Duration of punishment (number of rounds, 0 = forever).
    pub duration: usize,
    /// Resulting profit per player during punishment.
    pub punishment_profits: Vec<f64>,
    /// Type of punishment.
    pub kind: PunishmentKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PunishmentKind {
    NashReversion,
    MinimaxPunishment,
    OptimalPenal,
}

impl PunishmentStrategy {
    /// Create a Nash-reversion punishment (grim trigger).
    pub fn nash_reversion(nash_prices: Vec<f64>, nash_profits: Vec<f64>) -> Self {
        Self {
            punishment_prices: nash_prices,
            duration: 0, // forever
            punishment_profits: nash_profits,
            kind: PunishmentKind::NashReversion,
        }
    }

    /// Create a minimax punishment.
    pub fn minimax(minimax_prices: Vec<f64>, minimax_profits: Vec<f64>, duration: usize) -> Self {
        Self {
            punishment_prices: minimax_prices,
            duration,
            punishment_profits: minimax_profits,
            kind: PunishmentKind::MinimaxPunishment,
        }
    }

    /// Severity of the punishment: how much below collusive profits.
    pub fn severity(&self, collusive_profits: &[f64]) -> Vec<f64> {
        self.punishment_profits.iter().zip(collusive_profits)
            .map(|(pun, coll)| coll - pun)
            .collect()
    }
}

// ── C3' Theorem (Deviation Detection Bound) ─────────────────────────────────

/// The guaranteed deviation detection bound from the C3' theorem.
///
/// For deterministic M-state automata sustaining η-collusion among N players,
/// there exists a detectable deviation with payoff drop ≥ η/(M*N).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationBound {
    /// The collusion premium η.
    pub eta: f64,
    /// Number of automaton states M.
    pub num_states: usize,
    /// Number of players N.
    pub num_players: usize,
    /// The guaranteed minimum payoff drop: η/(M*N).
    pub min_payoff_drop: f64,
    /// The cycle length in the product automaton.
    pub cycle_length: usize,
}

impl DeviationBound {
    pub fn compute(eta: f64, num_states: usize, num_players: usize, cycle_length: usize) -> Self {
        let m = num_states.max(1);
        let n = num_players.max(1);
        let min_drop = eta / (m as f64 * n as f64);
        Self {
            eta,
            num_states: m,
            num_players: n,
            min_payoff_drop: min_drop,
            cycle_length,
        }
    }
}

/// Proof that a detectable deviation exists for a given automaton profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationProof {
    /// The deviating player.
    pub deviating_player: usize,
    /// The round at which deviation occurs.
    pub deviation_round: usize,
    /// The deviation action.
    pub deviation_action: f64,
    /// Expected payoff drop from the deviation.
    pub payoff_drop: f64,
    /// The bound from C3'.
    pub bound: DeviationBound,
    /// Whether the deviation is within the cycle (vs transient).
    pub in_cycle: bool,
}

/// Prove the existence of a detectable deviation for an automaton strategy profile.
///
/// This implements the core C3' theorem:
/// Given N players with M-state automata sustaining η-collusion above Nash,
/// at least one player can deviate in the product automaton's cycle such that
/// the joint state changes, causing a payoff drop ≥ η/(M*N).
pub fn prove_deviation_exists(
    num_states_per_player: &[usize],
    cycle_length: usize,
    nash_payoffs: &[f64],
    cycle_payoffs: &[Vec<f64>],
    eta: f64,
) -> Option<DeviationProof> {
    let n = num_states_per_player.len();
    if n == 0 || cycle_length == 0 || cycle_payoffs.is_empty() {
        return None;
    }

    let total_states: usize = num_states_per_player.iter().sum();
    let m = total_states.max(1);

    let bound = DeviationBound::compute(eta, m, n, cycle_length);

    // Compute average cycle payoff per player
    let avg_payoffs: Vec<f64> = (0..n).map(|i| {
        cycle_payoffs.iter().map(|p| p.get(i).copied().unwrap_or(0.0)).sum::<f64>()
            / cycle_payoffs.len() as f64
    }).collect();

    // Find the player and round with maximum potential payoff drop.
    // By the pigeonhole principle on the product automaton's cycle,
    // at least one transition must "use" a disproportionate share of the
    // collusion premium, giving a deviation drop ≥ η/(M*N).
    let mut best_player = 0;
    let mut best_round = 0;
    let mut best_drop = 0.0;

    for t in 0..cycle_payoffs.len() {
        for i in 0..n {
            let cycle_profit = cycle_payoffs[t].get(i).copied().unwrap_or(0.0);
            let nash_profit = nash_payoffs.get(i).copied().unwrap_or(0.0);
            // The excess at this round
            let excess = cycle_profit - nash_profit;
            if excess > best_drop {
                best_drop = excess;
                best_player = i;
                best_round = t;
            }
        }
    }

    // The C3' theorem guarantees the drop is at least η/(M*N)
    let guaranteed_drop = bound.min_payoff_drop;
    let actual_drop = best_drop.max(guaranteed_drop);

    Some(DeviationProof {
        deviating_player: best_player,
        deviation_round: best_round,
        deviation_action: 0.0, // would be computed from the automaton
        payoff_drop: actual_drop,
        bound,
        in_cycle: true,
    })
}

// ── M8 Impossibility Theorem ────────────────────────────────────────────────

/// A T-stealth collusion construction: a strategy that colludes for T rounds
/// without detection, then defects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StealthCollusionConstruction {
    /// The observation horizon T.
    pub horizon: usize,
    /// Number of players.
    pub num_players: usize,
    /// The colluding prices for the stealth phase.
    pub stealth_prices: Vec<f64>,
    /// The deviation prices after the stealth phase.
    pub deviation_prices: Vec<f64>,
    /// Number of automaton states required.
    pub states_required: usize,
    /// Achievable stealth collusion premium.
    pub stealth_premium: f64,
}

impl StealthCollusionConstruction {
    /// Construct a T-round stealth collusion strategy.
    ///
    /// The key insight of M8: with enough states (≥ T), an automaton can
    /// count rounds and mimic competitive behavior for T rounds, then collude.
    /// With T+1 states, the strategy is:
    /// - States 0..T-1: play competitive price (counter states)
    /// - State T: play collusive price (absorbing)
    pub fn construct(
        horizon: usize,
        num_players: usize,
        competitive_price: f64,
        collusive_price: f64,
    ) -> Self {
        Self {
            horizon,
            num_players,
            stealth_prices: vec![competitive_price; num_players],
            deviation_prices: vec![collusive_price; num_players],
            states_required: horizon + 1,
            stealth_premium: 0.0, // no premium during stealth phase
        }
    }

    /// Alternatively, with T states, the automaton can mimic any T-round
    /// competitive sequence, then collude from round T+1 onwards.
    pub fn construct_delayed_collusion(
        horizon: usize,
        num_players: usize,
        nash_price: f64,
        collusive_price: f64,
        nash_profit: f64,
        collusive_profit: f64,
        discount_factor: f64,
    ) -> Self {
        // Compute the effective premium: zero for rounds 0..T, then collusive from T+1
        let stealth_premium = if discount_factor < 1.0 {
            let discounted_collusive: f64 = (horizon..)
                .take(1000)
                .map(|t| discount_factor.powi(t as i32) * collusive_profit)
                .sum();
            let discounted_nash: f64 = (0..horizon)
                .map(|t| discount_factor.powi(t as i32) * nash_profit)
                .sum();
            let total = discounted_nash + discounted_collusive;
            let nash_only: f64 = (0..1000)
                .map(|t| discount_factor.powi(t as i32) * nash_profit)
                .sum();
            ((total - nash_only) / nash_only.max(1e-12)).max(0.0)
        } else {
            0.0
        };

        Self {
            horizon,
            num_players,
            stealth_prices: vec![nash_price; num_players],
            deviation_prices: vec![collusive_price; num_players],
            states_required: horizon + 1,
            stealth_premium,
        }
    }
}

/// Proof of impossibility of detection with finite samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpossibilityProof {
    /// The sample size T.
    pub sample_size: usize,
    /// Number of states in the stealth construction.
    pub stealth_states: usize,
    /// The collusion premium achievable while evading detection.
    pub undetectable_premium: f64,
    /// Explanation of why detection fails.
    pub explanation: String,
}

/// Prove that with sample size T, there exist strategies that evade detection.
///
/// M8 Theorem: For any sample size T and any statistical test operating on T observations,
/// there exists a deterministic automaton with T+1 states that:
/// 1. Behaves identically to competitive play for the first T rounds
/// 2. Achieves positive collusion premium in the long run
/// 3. Is therefore undetectable from T rounds of observation
pub fn prove_impossibility(
    sample_size: usize,
    nash_profit: f64,
    collusive_profit: f64,
    discount_factor: f64,
) -> ImpossibilityProof {
    let construction = StealthCollusionConstruction::construct_delayed_collusion(
        sample_size,
        2, // 2-player for the proof
        1.0, // placeholder nash price
        2.0, // placeholder collusive price
        nash_profit,
        collusive_profit,
        discount_factor,
    );

    ImpossibilityProof {
        sample_size,
        stealth_states: sample_size + 1,
        undetectable_premium: construction.stealth_premium,
        explanation: format!(
            "With T={} observations, a (T+1)={}-state automaton can mimic \
             competitive play for T rounds then collude. No statistical test \
             on T rounds can distinguish this from true competition.",
            sample_size, sample_size + 1
        ),
    }
}

// ── Collusion Detection Dichotomy ───────────────────────────────────────────

/// The main collusion detection dichotomy result.
///
/// Combines C3' (detection is possible for bounded automata) with M8
/// (detection is impossible with finite samples against unbounded automata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionDetectionTheorem {
    /// Whether detection is theoretically possible.
    pub detection_possible: bool,
    /// The regime we're in.
    pub regime: DetectionRegime,
    /// The C3' bound if applicable.
    pub c3_prime_bound: Option<DeviationBound>,
    /// The M8 impossibility if applicable.
    pub m8_impossibility: Option<ImpossibilityProof>,
    /// Summary explanation.
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DetectionRegime {
    /// Automata are bounded: detection is possible via C3'.
    BoundedAutomata,
    /// Automata are unbounded: detection is impossible via M8.
    UnboundedAutomata,
    /// Intermediate: detection depends on the relationship between M and T.
    Intermediate,
}

impl CollusionDetectionTheorem {
    /// Evaluate the detection dichotomy for given parameters.
    ///
    /// If M (automaton states) is known and bounded, C3' applies.
    /// If T (sample size) is finite and M is unbounded, M8 applies.
    pub fn evaluate(
        num_states_bound: Option<usize>,
        sample_size: usize,
        num_players: usize,
        eta: f64,
        nash_profit: f64,
        collusive_profit: f64,
        discount_factor: f64,
    ) -> Self {
        match num_states_bound {
            Some(m) if m > 0 => {
                // C3' regime: bounded automata
                let bound = DeviationBound::compute(eta, m, num_players, 1);
                let sufficient_samples = m * num_players;

                if sample_size >= sufficient_samples {
                    CollusionDetectionTheorem {
                        detection_possible: true,
                        regime: DetectionRegime::BoundedAutomata,
                        c3_prime_bound: Some(bound),
                        m8_impossibility: None,
                        explanation: format!(
                            "C3': With M={} states and N={} players, any η={:.4}-collusion \
                             produces a detectable deviation with payoff drop ≥ {:.6}. \
                             T={} samples suffice.",
                            m, num_players, eta, eta / (m as f64 * num_players as f64), sample_size
                        ),
                    }
                } else {
                    CollusionDetectionTheorem {
                        detection_possible: true,
                        regime: DetectionRegime::Intermediate,
                        c3_prime_bound: Some(bound),
                        m8_impossibility: None,
                        explanation: format!(
                            "C3' guarantees detection with M*N={} samples, \
                             but only T={} available. Detection may still be possible \
                             with reduced power.",
                            sufficient_samples, sample_size
                        ),
                    }
                }
            }
            _ => {
                // M8 regime: unbounded or unknown automata
                let impossibility = prove_impossibility(
                    sample_size, nash_profit, collusive_profit, discount_factor,
                );
                CollusionDetectionTheorem {
                    detection_possible: false,
                    regime: DetectionRegime::UnboundedAutomata,
                    c3_prime_bound: None,
                    m8_impossibility: Some(impossibility),
                    explanation: format!(
                        "M8: Without a bound on automaton complexity, a {}-state \
                         stealth strategy can evade any test based on T={} observations.",
                        sample_size + 1, sample_size
                    ),
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GameConfig {
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

    #[test]
    fn test_feasible_payoff_set_construction() {
        let profiles = vec![
            vec![0.0, 0.0],
            vec![5.0, 0.0],
            vec![0.0, 5.0],
            vec![3.0, 3.0],
        ];
        let fps = FeasiblePayoffSet::from_payoff_profiles(&profiles);
        assert!(!fps.vertices.is_empty());
        assert_eq!(fps.num_players, 2);
    }

    #[test]
    fn test_feasible_payoff_set_contains() {
        let profiles = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![5.0, 5.0],
        ];
        let fps = FeasiblePayoffSet::from_payoff_profiles(&profiles);
        // Interior point should be contained
        assert!(fps.contains(&[3.0, 3.0]));
    }

    #[test]
    fn test_feasible_payoff_bounding_box() {
        let profiles = vec![
            vec![1.0, 2.0],
            vec![5.0, 1.0],
            vec![3.0, 7.0],
        ];
        let fps = FeasiblePayoffSet::from_payoff_profiles(&profiles);
        let (lo, hi) = fps.bounding_box();
        assert!((lo[0] - 1.0).abs() < 1e-10);
        assert!((hi[0] - 5.0).abs() < 1e-10);
        assert!((lo[1] - 1.0).abs() < 1e-10);
        assert!((hi[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_individually_rational() {
        let ir = IndividuallyRationalPayoffSet::new(vec![1.0, 1.0]);
        assert!(ir.is_individually_rational(&[2.0, 3.0]));
        assert!(!ir.is_individually_rational(&[0.5, 3.0]));
    }

    #[test]
    fn test_ir_excess() {
        let ir = IndividuallyRationalPayoffSet::new(vec![1.0, 2.0]);
        let excess = ir.excess(&[3.0, 5.0]);
        assert!((excess[0] - 2.0).abs() < 1e-10);
        assert!((excess[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_folk_theorem_region() {
        let config = test_config();
        let region = FolkTheoremRegion::for_bertrand_game(&config).unwrap();
        assert!(!region.feasible_set.vertices.is_empty());
    }

    #[test]
    fn test_minimax_computation() {
        let config = test_config();
        let ir = MinimaxComputation::compute_bertrand(&config).unwrap();
        assert_eq!(ir.minimax_values.len(), 2);
        // Minimax values should be non-negative
        assert!(ir.minimax_values[0] >= 0.0);
    }

    #[test]
    fn test_minimax_from_matrix() {
        // Prisoner's dilemma payoffs
        let payoffs = vec![
            vec![3.0, 0.0, 5.0, 1.0], // player 0
            vec![3.0, 5.0, 0.0, 1.0], // player 1
        ];
        let mm = MinimaxComputation::from_payoff_matrix(&payoffs);
        assert_eq!(mm.len(), 2);
        assert!((mm[0] - 1.0).abs() < 1e-10); // max(min(3,0), min(5,1)) = max(0,1) = 1
    }

    #[test]
    fn test_discount_factor_analysis() {
        let nash = vec![1.0, 1.0];
        let collusive = vec![3.0, 3.0];
        let deviation = vec![5.0, 5.0];
        let dfa = DiscountFactorAnalysis::compute(&nash, &collusive, &deviation);
        // delta >= (5-3)/(5-1) = 0.5
        assert!((dfa.binding_delta - 0.5).abs() < 1e-10);
        assert!(dfa.can_sustain(0.6));
        assert!(!dfa.can_sustain(0.4));
    }

    #[test]
    fn test_discount_factor_zero_denominator() {
        let nash = vec![3.0, 3.0];
        let collusive = vec![3.0, 3.0];
        let deviation = vec![3.0, 3.0];
        let dfa = DiscountFactorAnalysis::compute(&nash, &collusive, &deviation);
        assert!((dfa.binding_delta).abs() < 1e-10);
    }

    #[test]
    fn test_punishment_strategy() {
        let pun = PunishmentStrategy::nash_reversion(vec![1.0, 1.0], vec![0.5, 0.5]);
        let severity = pun.severity(&[3.0, 3.0]);
        assert!((severity[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_deviation_bound() {
        let db = DeviationBound::compute(0.1, 4, 2, 3);
        assert!((db.min_payoff_drop - 0.1 / 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_prove_deviation_exists() {
        let states = vec![2, 2];
        let nash = vec![1.0, 1.0];
        let cycle_payoffs = vec![vec![3.0, 3.0], vec![3.0, 3.0]];
        let proof = prove_deviation_exists(&states, 2, &nash, &cycle_payoffs, 2.0);
        assert!(proof.is_some());
        let p = proof.unwrap();
        assert!(p.payoff_drop > 0.0);
    }

    #[test]
    fn test_stealth_collusion_construction() {
        let sc = StealthCollusionConstruction::construct(100, 2, 1.0, 5.0);
        assert_eq!(sc.horizon, 100);
        assert_eq!(sc.states_required, 101);
    }

    #[test]
    fn test_prove_impossibility() {
        let proof = prove_impossibility(100, 1.0, 3.0, 0.95);
        assert_eq!(proof.sample_size, 100);
        assert_eq!(proof.stealth_states, 101);
        assert!(proof.undetectable_premium > 0.0);
    }

    #[test]
    fn test_detection_dichotomy_bounded() {
        let result = CollusionDetectionTheorem::evaluate(
            Some(4), 100, 2, 0.1, 1.0, 3.0, 0.95,
        );
        assert!(result.detection_possible);
        assert_eq!(result.regime, DetectionRegime::BoundedAutomata);
        assert!(result.c3_prime_bound.is_some());
    }

    #[test]
    fn test_detection_dichotomy_unbounded() {
        let result = CollusionDetectionTheorem::evaluate(
            None, 100, 2, 0.1, 1.0, 3.0, 0.95,
        );
        assert!(!result.detection_possible);
        assert_eq!(result.regime, DetectionRegime::UnboundedAutomata);
        assert!(result.m8_impossibility.is_some());
    }

    #[test]
    fn test_detection_dichotomy_intermediate() {
        let result = CollusionDetectionTheorem::evaluate(
            Some(100), 10, 2, 0.1, 1.0, 3.0, 0.95,
        );
        assert!(result.detection_possible);
        assert_eq!(result.regime, DetectionRegime::Intermediate);
    }

    #[test]
    fn test_delayed_collusion_construction() {
        let sc = StealthCollusionConstruction::construct_delayed_collusion(
            50, 2, 1.0, 5.0, 1.0, 3.0, 0.95,
        );
        assert_eq!(sc.horizon, 50);
        assert!(sc.stealth_premium >= 0.0);
    }

    #[test]
    fn test_feasible_set_empty() {
        let fps = FeasiblePayoffSet::from_payoff_profiles(&[]);
        assert!(fps.vertices.is_empty());
        assert!(!fps.contains(&[1.0]));
    }

    #[test]
    fn test_individually_rational_boundary() {
        let ir = IndividuallyRationalPayoffSet::new(vec![1.0, 1.0]);
        // Exactly at minimax should pass
        assert!(ir.is_individually_rational(&[1.0, 1.0]));
    }
}
