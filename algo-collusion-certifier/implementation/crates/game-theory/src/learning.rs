//! Learning dynamics theory for pricing algorithms.
//!
//! This module characterizes the theoretical properties of learning algorithms
//! rather than implementing the algorithms themselves. It provides regret bounds,
//! convergence rates, Q-value dynamics, and bounds on price correlations.

use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── No-Regret Learner ───────────────────────────────────────────────────────

/// Types of no-regret learning algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LearnerType {
    MultiplicativeWeights,
    FollowTheRegularizedLeader,
    Exp3,
    UCB1,
    EpsilonGreedy,
    BoltzmannExploration,
    TabularQLearning,
}

impl fmt::Display for LearnerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LearnerType::MultiplicativeWeights => write!(f, "MW"),
            LearnerType::FollowTheRegularizedLeader => write!(f, "FTRL"),
            LearnerType::Exp3 => write!(f, "Exp3"),
            LearnerType::UCB1 => write!(f, "UCB1"),
            LearnerType::EpsilonGreedy => write!(f, "ε-Greedy"),
            LearnerType::BoltzmannExploration => write!(f, "Boltzmann"),
            LearnerType::TabularQLearning => write!(f, "Q-Learning"),
        }
    }
}

/// Characterization of a no-regret learner.
pub trait NoRegretLearner: fmt::Debug + Send + Sync {
    /// Type of learner.
    fn learner_type(&self) -> LearnerType;

    /// Number of actions available.
    fn num_actions(&self) -> usize;

    /// Theoretical regret bound after T rounds.
    fn regret_bound(&self, t: usize) -> RegretBound;

    /// Whether this learner has the no-regret property.
    fn is_no_regret(&self) -> bool { true }

    /// Convergence rate (how fast regret/T → 0).
    fn convergence_rate(&self) -> ConvergenceRate;
}

// ── Regret Bound ────────────────────────────────────────────────────────────

/// Theoretical regret bound for a learning algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegretBound {
    /// The regret bound value after T rounds.
    pub bound: f64,
    /// Number of rounds.
    pub rounds: usize,
    /// Per-round regret: bound / T.
    pub per_round: f64,
    /// Asymptotic rate class.
    pub rate: ConvergenceRate,
    /// Description of the bound.
    pub description: String,
}

impl RegretBound {
    pub fn new(bound: f64, rounds: usize, rate: ConvergenceRate, description: &str) -> Self {
        let per_round = if rounds > 0 { bound / rounds as f64 } else { 0.0 };
        Self {
            bound,
            rounds,
            per_round,
            rate,
            description: description.to_string(),
        }
    }
}

// ── Convergence Rate ────────────────────────────────────────────────────────

/// Asymptotic convergence rate classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceRate {
    /// O(1/√T) — standard for adversarial settings.
    InvSqrtT,
    /// O(ln(T)/T) — typical for stochastic bandits.
    LogTOverT,
    /// O(1/T) — fast rate for strongly convex losses.
    InvT,
    /// O(1/T²) — accelerated methods.
    InvTSquared,
    /// No convergence guarantee.
    NoGuarantee,
}

impl ConvergenceRate {
    /// Compute the rate value for a given T.
    pub fn evaluate(&self, t: usize) -> f64 {
        let t = t.max(1) as f64;
        match self {
            ConvergenceRate::InvSqrtT => 1.0 / t.sqrt(),
            ConvergenceRate::LogTOverT => t.ln() / t,
            ConvergenceRate::InvT => 1.0 / t,
            ConvergenceRate::InvTSquared => 1.0 / (t * t),
            ConvergenceRate::NoGuarantee => 1.0,
        }
    }

    /// Is this rate asymptotically vanishing?
    pub fn is_vanishing(&self) -> bool {
        !matches!(self, ConvergenceRate::NoGuarantee)
    }
}

impl fmt::Display for ConvergenceRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvergenceRate::InvSqrtT => write!(f, "O(1/√T)"),
            ConvergenceRate::LogTOverT => write!(f, "O(ln T / T)"),
            ConvergenceRate::InvT => write!(f, "O(1/T)"),
            ConvergenceRate::InvTSquared => write!(f, "O(1/T²)"),
            ConvergenceRate::NoGuarantee => write!(f, "No guarantee"),
        }
    }
}

// ── Concrete Learner Characterizations ──────────────────────────────────────

/// Multiplicative Weights Update characterization.
#[derive(Debug, Clone)]
pub struct MultiplicativeWeightsLearner {
    pub num_actions: usize,
    pub learning_rate: f64,
}

impl MultiplicativeWeightsLearner {
    pub fn new(num_actions: usize, learning_rate: f64) -> Self {
        Self { num_actions, learning_rate }
    }
}

impl NoRegretLearner for MultiplicativeWeightsLearner {
    fn learner_type(&self) -> LearnerType { LearnerType::MultiplicativeWeights }
    fn num_actions(&self) -> usize { self.num_actions }

    fn regret_bound(&self, t: usize) -> RegretBound {
        // MW regret: eta * T + ln(K) / eta
        // Optimal eta = sqrt(ln(K) / T), giving regret = 2 * sqrt(T * ln(K))
        let k = self.num_actions as f64;
        let t_f = t as f64;
        let bound = 2.0 * (t_f * k.ln()).sqrt();
        RegretBound::new(bound, t, ConvergenceRate::InvSqrtT,
            "MW: 2√(T·ln K) with optimal learning rate")
    }

    fn convergence_rate(&self) -> ConvergenceRate { ConvergenceRate::InvSqrtT }
}

/// Epsilon-Greedy learner characterization.
#[derive(Debug, Clone)]
pub struct EpsilonGreedyLearner {
    pub num_actions: usize,
    pub epsilon: f64,
}

impl EpsilonGreedyLearner {
    pub fn new(num_actions: usize, epsilon: f64) -> Self {
        Self { num_actions, epsilon }
    }
}

impl NoRegretLearner for EpsilonGreedyLearner {
    fn learner_type(&self) -> LearnerType { LearnerType::EpsilonGreedy }
    fn num_actions(&self) -> usize { self.num_actions }

    fn regret_bound(&self, t: usize) -> RegretBound {
        // ε-greedy with decaying ε_t = min(1, cK/(d²t)):
        // Expected regret O(K ln T / d²)
        // With fixed ε: regret = ε * T (linear for fixed ε)
        let k = self.num_actions as f64;
        let t_f = t as f64;
        let bound = if self.epsilon > 0.0 {
            self.epsilon * t_f + k * t_f.ln()
        } else {
            k * t_f.ln()
        };
        RegretBound::new(bound, t, ConvergenceRate::LogTOverT,
            "ε-Greedy: ε·T + K·ln(T) for decaying ε")
    }

    fn is_no_regret(&self) -> bool {
        // Only no-regret if ε decays
        self.epsilon < 1.0
    }

    fn convergence_rate(&self) -> ConvergenceRate { ConvergenceRate::LogTOverT }
}

// ── Q-Value Dynamics ────────────────────────────────────────────────────────

/// Theoretical Q-value evolution for tabular Q-learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QValueDynamics {
    /// Number of states.
    pub num_states: usize,
    /// Number of actions.
    pub num_actions: usize,
    /// Discount factor gamma.
    pub gamma: f64,
    /// Learning rate alpha.
    pub alpha: f64,
    /// Current Q-table.
    pub q_table: Vec<Vec<f64>>,
}

impl QValueDynamics {
    pub fn new(num_states: usize, num_actions: usize, gamma: f64, alpha: f64) -> Self {
        Self {
            num_states,
            num_actions,
            gamma,
            alpha,
            q_table: vec![vec![0.0; num_actions]; num_states],
        }
    }

    /// Theoretical fixed point: Q*(s,a) = E[r + γ max_a' Q*(s',a')]
    /// For a single-state (bandit) setting: Q*(a) = r(a) / (1 - γ)
    pub fn compute_fixed_point_bandit(&self, rewards: &[f64]) -> Vec<f64> {
        let denom = 1.0 - self.gamma;
        if denom.abs() < 1e-12 {
            return rewards.to_vec();
        }
        rewards.iter().map(|&r| r / denom).collect()
    }

    /// Simulate Q-learning update for a single step.
    pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        let max_next_q = self.q_table[next_state].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let target = reward + self.gamma * max_next_q;
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action]);
    }

    /// Theoretical convergence time estimate for Q-learning.
    /// T ~ O(S * A / (alpha * (1 - gamma)^5)) for tabular Q-learning.
    pub fn convergence_time_estimate(&self) -> f64 {
        let sa = (self.num_states * self.num_actions) as f64;
        let one_minus_gamma = (1.0 - self.gamma).max(1e-6);
        sa / (self.alpha * one_minus_gamma.powi(5))
    }

    /// Distance to fixed point (max |Q - Q*|).
    pub fn distance_to_fixed_point(&self, q_star: &[Vec<f64>]) -> f64 {
        let mut max_diff = 0.0f64;
        for s in 0..self.num_states {
            for a in 0..self.num_actions {
                let diff = (self.q_table[s][a] - q_star[s][a]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        max_diff
    }

    /// Get the greedy policy from the current Q-table.
    pub fn greedy_policy(&self) -> Vec<usize> {
        self.q_table.iter().map(|q_s| {
            q_s.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }).collect()
    }
}

// ── Learning Equilibrium ────────────────────────────────────────────────────

/// What learning algorithms converge to in repeated games.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEquilibrium {
    /// The type of equilibrium reached.
    pub equilibrium_type: LearningEquilibriumType,
    /// Expected payoffs at convergence.
    pub expected_payoffs: Vec<f64>,
    /// Convergence time estimate (rounds).
    pub convergence_time: f64,
    /// Whether the outcome is Nash.
    pub is_nash: bool,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningEquilibriumType {
    /// Converges to Nash equilibrium.
    Nash,
    /// Converges to correlated equilibrium.
    Correlated,
    /// Converges to coarse correlated equilibrium.
    CoarseCorrelated,
    /// May cycle or not converge.
    Cycling,
    /// Converges to a supra-competitive outcome (possible collusion).
    SupraCompetitive,
}

impl LearningEquilibrium {
    /// Predict the learning outcome for two no-regret learners.
    /// Theorem: if both players use no-regret algorithms, the empirical
    /// distribution of play converges to a coarse correlated equilibrium.
    pub fn predict_no_regret_outcome(
        learner_a: &dyn NoRegretLearner,
        learner_b: &dyn NoRegretLearner,
        nash_payoffs: &[f64],
    ) -> Self {
        let both_no_regret = learner_a.is_no_regret() && learner_b.is_no_regret();

        if both_no_regret {
            LearningEquilibrium {
                equilibrium_type: LearningEquilibriumType::CoarseCorrelated,
                expected_payoffs: nash_payoffs.to_vec(),
                convergence_time: Self::estimate_convergence_time(learner_a, learner_b),
                is_nash: false,
                description: "Both players use no-regret: empirical distribution → CCE".to_string(),
            }
        } else {
            LearningEquilibrium {
                equilibrium_type: LearningEquilibriumType::Cycling,
                expected_payoffs: nash_payoffs.to_vec(),
                convergence_time: f64::INFINITY,
                is_nash: false,
                description: "Not all players are no-regret: convergence not guaranteed".to_string(),
            }
        }
    }

    fn estimate_convergence_time(a: &dyn NoRegretLearner, b: &dyn NoRegretLearner) -> f64 {
        // Estimate T such that regret/T < epsilon
        let epsilon = 0.01;
        let ka = a.num_actions() as f64;
        let kb = b.num_actions() as f64;
        let k = ka.max(kb);
        // For MW: 2*sqrt(T*ln(K)) / T < eps => T > 4*ln(K)/eps^2
        4.0 * k.ln() / (epsilon * epsilon)
    }
}

// ── Price Correlation Bound ─────────────────────────────────────────────────

/// Theoretical bound on cross-firm price correlation under independent learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceCorrelationBound {
    /// Maximum expected correlation under independent learning.
    pub max_correlation: f64,
    /// Number of rounds used in estimation.
    pub sample_size: usize,
    /// Number of actions (price levels).
    pub num_actions: usize,
    /// Description.
    pub description: String,
}

impl PriceCorrelationBound {
    /// Compute the maximum expected price correlation between two independent
    /// no-regret learners.
    ///
    /// Key insight: independent learners choosing from K actions over T rounds
    /// can have spurious correlation up to O(√(K ln K / T)) due to:
    /// 1. Common convergence toward Nash
    /// 2. Finite-sample noise
    pub fn compute(num_actions: usize, sample_size: usize) -> Self {
        let k = num_actions as f64;
        let t = sample_size.max(1) as f64;
        // Upper bound on spurious correlation for independent learners
        let bound = (k * k.ln() / t).sqrt().min(1.0);
        Self {
            max_correlation: bound,
            sample_size,
            num_actions,
            description: format!(
                "Independent learners with K={} actions over T={} rounds: \
                 max spurious correlation ≤ {:.4}",
                num_actions, sample_size, bound
            ),
        }
    }

    /// Is an observed correlation significantly above the independent bound?
    pub fn is_above_bound(&self, observed_correlation: f64) -> bool {
        observed_correlation > self.max_correlation + 0.05 // small margin
    }
}

// ── Independent Learner Model ───────────────────────────────────────────────

/// Model of truly independent learners for the null hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentLearnerModel {
    pub num_players: usize,
    pub num_actions: usize,
    pub learner_type: LearnerType,
    /// Predicted stationary distribution over actions for each player.
    pub predicted_distribution: Vec<Vec<f64>>,
}

impl IndependentLearnerModel {
    pub fn new(num_players: usize, num_actions: usize, learner_type: LearnerType) -> Self {
        // Under independence, each player converges to their best response distribution.
        // For symmetric games with MW, this is often the Nash equilibrium mixed strategy.
        let uniform = vec![1.0 / num_actions as f64; num_actions];
        let predicted_distribution = vec![uniform; num_players];

        Self {
            num_players,
            num_actions,
            learner_type,
            predicted_distribution,
        }
    }

    /// Set the predicted stationary distribution for a player.
    pub fn set_distribution(&mut self, player: usize, dist: Vec<f64>) {
        if player < self.num_players {
            self.predicted_distribution[player] = dist;
        }
    }

    /// Generate a simulated independent trajectory.
    pub fn simulate(&self, rounds: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut trajectory = Vec::with_capacity(rounds);

        for _ in 0..rounds {
            let mut actions = Vec::with_capacity(self.num_players);
            for player in 0..self.num_players {
                let dist = &self.predicted_distribution[player];
                let r: f64 = rng.gen();
                let mut cumsum = 0.0;
                let mut chosen = dist.len() - 1;
                for (a, &p) in dist.iter().enumerate() {
                    cumsum += p;
                    if r < cumsum {
                        chosen = a;
                        break;
                    }
                }
                actions.push(chosen);
            }
            trajectory.push(actions);
        }
        trajectory
    }

    /// Compute expected cross-correlation under independence.
    pub fn expected_correlation(&self) -> f64 {
        // Under true independence, the expected correlation is 0.
        // However, if both converge to similar distributions, there's correlation
        // from the common structure (not communication).
        if self.num_players < 2 { return 0.0; }

        let dist_0 = &self.predicted_distribution[0];
        let dist_1 = &self.predicted_distribution[1];

        // E[X*Y] - E[X]*E[Y] under independence = 0
        // But E[action correlation] depends on action labels
        let mean_0: f64 = dist_0.iter().enumerate().map(|(a, &p)| a as f64 * p).sum();
        let mean_1: f64 = dist_1.iter().enumerate().map(|(a, &p)| a as f64 * p).sum();
        let var_0: f64 = dist_0.iter().enumerate().map(|(a, &p)| (a as f64 - mean_0).powi(2) * p).sum();
        let var_1: f64 = dist_1.iter().enumerate().map(|(a, &p)| (a as f64 - mean_1).powi(2) * p).sum();

        // Under independence, correlation = 0 regardless of marginals
        0.0
    }
}

// ── Correlation Under Competition ───────────────────────────────────────────

/// Maximum achievable price correlation without coordination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationUnderCompetition {
    /// Maximum correlation from common reaction to market conditions.
    pub max_structural_correlation: f64,
    /// Maximum correlation from common convergence to Nash.
    pub max_convergence_correlation: f64,
    /// Total maximum non-collusive correlation.
    pub max_total_correlation: f64,
    pub description: String,
}

impl CorrelationUnderCompetition {
    /// Compute the maximum price correlation achievable without coordination.
    ///
    /// Sources of non-collusive correlation:
    /// 1. Common reaction to demand shocks
    /// 2. Convergence to Nash equilibrium
    /// 3. Similar learning algorithms responding similarly
    pub fn compute(
        demand_volatility: f64,
        cross_price_elasticity: f64,
        num_actions: usize,
        sample_size: usize,
    ) -> Self {
        // Structural correlation from demand cross-elasticity
        let structural = cross_price_elasticity.abs().min(1.0);

        // Convergence correlation: as both converge to NE, prices become similar
        let t = sample_size.max(1) as f64;
        let convergence = (1.0 / t.sqrt()).min(0.5);

        // Demand shock correlation
        let demand_corr = demand_volatility.min(1.0);

        // Total (conservative upper bound)
        let total = (structural + convergence + demand_corr).min(1.0);

        Self {
            max_structural_correlation: structural,
            max_convergence_correlation: convergence,
            max_total_correlation: total,
            description: format!(
                "Non-collusive correlation bound: structural={:.3}, convergence={:.3}, total={:.3}",
                structural, convergence, total
            ),
        }
    }

    /// Is an observed correlation evidence of collusion?
    pub fn suggests_collusion(&self, observed_correlation: f64, margin: f64) -> bool {
        observed_correlation > self.max_total_correlation + margin
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mw_regret_bound() {
        let mw = MultiplicativeWeightsLearner::new(10, 0.1);
        let rb = mw.regret_bound(1000);
        assert!(rb.bound > 0.0);
        assert!(rb.per_round < 1.0);
    }

    #[test]
    fn test_mw_is_no_regret() {
        let mw = MultiplicativeWeightsLearner::new(10, 0.1);
        assert!(mw.is_no_regret());
    }

    #[test]
    fn test_epsilon_greedy_regret() {
        let eg = EpsilonGreedyLearner::new(5, 0.1);
        let rb = eg.regret_bound(1000);
        assert!(rb.bound > 0.0);
    }

    #[test]
    fn test_convergence_rate_ordering() {
        let t = 1000;
        let inv_sqrt = ConvergenceRate::InvSqrtT.evaluate(t);
        let log_over_t = ConvergenceRate::LogTOverT.evaluate(t);
        let inv_t = ConvergenceRate::InvT.evaluate(t);
        assert!(inv_sqrt > log_over_t);
        assert!(log_over_t > inv_t);
    }

    #[test]
    fn test_convergence_rate_vanishing() {
        assert!(ConvergenceRate::InvSqrtT.is_vanishing());
        assert!(!ConvergenceRate::NoGuarantee.is_vanishing());
    }

    #[test]
    fn test_q_value_dynamics_creation() {
        let qvd = QValueDynamics::new(5, 3, 0.99, 0.1);
        assert_eq!(qvd.num_states, 5);
        assert_eq!(qvd.num_actions, 3);
    }

    #[test]
    fn test_q_value_update() {
        let mut qvd = QValueDynamics::new(2, 2, 0.9, 0.1);
        qvd.update(0, 0, 1.0, 0);
        assert!(qvd.q_table[0][0] > 0.0);
    }

    #[test]
    fn test_q_value_fixed_point_bandit() {
        let qvd = QValueDynamics::new(1, 3, 0.9, 0.1);
        let rewards = vec![1.0, 2.0, 3.0];
        let fp = qvd.compute_fixed_point_bandit(&rewards);
        assert!((fp[0] - 10.0).abs() < 1e-10); // 1/(1-0.9) = 10
        assert!((fp[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_q_value_greedy_policy() {
        let mut qvd = QValueDynamics::new(2, 3, 0.9, 0.1);
        qvd.q_table[0] = vec![1.0, 3.0, 2.0];
        qvd.q_table[1] = vec![5.0, 1.0, 2.0];
        let policy = qvd.greedy_policy();
        assert_eq!(policy, vec![1, 0]);
    }

    #[test]
    fn test_convergence_time_estimate() {
        let qvd = QValueDynamics::new(10, 5, 0.95, 0.01);
        let ct = qvd.convergence_time_estimate();
        assert!(ct > 0.0);
    }

    #[test]
    fn test_learning_equilibrium_no_regret() {
        let mw_a = MultiplicativeWeightsLearner::new(5, 0.1);
        let mw_b = MultiplicativeWeightsLearner::new(5, 0.1);
        let nash = vec![1.0, 1.0];
        let eq = LearningEquilibrium::predict_no_regret_outcome(&mw_a, &mw_b, &nash);
        assert_eq!(eq.equilibrium_type, LearningEquilibriumType::CoarseCorrelated);
    }

    #[test]
    fn test_price_correlation_bound() {
        let pcb = PriceCorrelationBound::compute(10, 1000);
        assert!(pcb.max_correlation >= 0.0);
        assert!(pcb.max_correlation <= 1.0);
    }

    #[test]
    fn test_price_correlation_bound_large_sample() {
        let pcb = PriceCorrelationBound::compute(5, 100_000);
        // With large sample, bound should be small
        assert!(pcb.max_correlation < 0.1);
    }

    #[test]
    fn test_correlation_above_bound() {
        let pcb = PriceCorrelationBound::compute(5, 10_000);
        assert!(pcb.is_above_bound(0.9)); // 0.9 is well above any reasonable bound
    }

    #[test]
    fn test_independent_learner_model() {
        let model = IndependentLearnerModel::new(2, 5, LearnerType::MultiplicativeWeights);
        assert_eq!(model.predicted_distribution.len(), 2);
        assert!((model.predicted_distribution[0].iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_independent_learner_simulation() {
        let model = IndependentLearnerModel::new(2, 3, LearnerType::EpsilonGreedy);
        let traj = model.simulate(100, 42);
        assert_eq!(traj.len(), 100);
        for actions in &traj {
            assert_eq!(actions.len(), 2);
            assert!(actions[0] < 3);
            assert!(actions[1] < 3);
        }
    }

    #[test]
    fn test_expected_correlation_independent() {
        let model = IndependentLearnerModel::new(2, 5, LearnerType::MultiplicativeWeights);
        // Under independence, expected correlation is 0
        assert!((model.expected_correlation() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_under_competition() {
        let cuc = CorrelationUnderCompetition::compute(0.1, 0.3, 10, 1000);
        assert!(cuc.max_total_correlation > 0.0);
        assert!(cuc.max_total_correlation <= 1.0);
    }

    #[test]
    fn test_suggests_collusion() {
        let cuc = CorrelationUnderCompetition::compute(0.1, 0.1, 5, 10000);
        assert!(!cuc.suggests_collusion(0.1, 0.05));
        assert!(cuc.suggests_collusion(0.95, 0.05));
    }

    #[test]
    fn test_regret_bound_properties() {
        let mw = MultiplicativeWeightsLearner::new(10, 0.1);
        let rb_100 = mw.regret_bound(100);
        let rb_1000 = mw.regret_bound(1000);
        // Regret grows but per-round decreases
        assert!(rb_1000.bound > rb_100.bound);
        assert!(rb_1000.per_round < rb_100.per_round);
    }

    #[test]
    fn test_distance_to_fixed_point() {
        let mut qvd = QValueDynamics::new(2, 2, 0.9, 0.1);
        qvd.q_table = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let q_star = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((qvd.distance_to_fixed_point(&q_star) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_learner_type_display() {
        assert_eq!(format!("{}", LearnerType::MultiplicativeWeights), "MW");
        assert_eq!(format!("{}", LearnerType::TabularQLearning), "Q-Learning");
    }
}
