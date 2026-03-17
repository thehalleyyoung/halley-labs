//! Multi-armed bandit algorithms adapted for pricing.
//!
//! Implements epsilon-greedy, UCB1, Thompson sampling, and EXP3 bandits.
//! Each arm corresponds to a price level; rewards are profits.

use crate::algorithm::{AlgorithmState, PricingAlgorithm};
use rand::prelude::*;
use rand_distr::Beta;
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, MarketOutcome, PlayerAction, PlayerId, Price, RoundNumber};

// ═══════════════════════════════════════════════════════════════════════════
// Arm statistics
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStatistics {
    pub count: usize,
    pub total_reward: f64,
    pub sum_squared_reward: f64,
}

impl ArmStatistics {
    pub fn new() -> Self {
        Self {
            count: 0,
            total_reward: 0.0,
            sum_squared_reward: 0.0,
        }
    }

    pub fn update(&mut self, reward: f64) {
        self.count += 1;
        self.total_reward += reward;
        self.sum_squared_reward += reward * reward;
    }

    pub fn mean_reward(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.total_reward / self.count as f64
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return f64::MAX;
        }
        let mean = self.mean_reward();
        self.sum_squared_reward / self.count as f64 - mean * mean
    }
}

impl Default for ArmStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Common helper: map arm index to price.
fn arm_to_price(arm: usize, num_arms: usize, price_min: Price, price_max: Price) -> Price {
    if num_arms <= 1 {
        return (price_min + price_max) / 2.0;
    }
    price_min + (arm as f64) * (price_max - price_min) / (num_arms - 1) as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// EpsilonGreedyBandit
// ═══════════════════════════════════════════════════════════════════════════

/// Epsilon-greedy multi-armed bandit: arms = prices, rewards = profits.
pub struct EpsilonGreedyBandit {
    player_id: PlayerId,
    num_arms: usize,
    price_min: Price,
    price_max: Price,
    epsilon: f64,
    arms: Vec<ArmStatistics>,
    last_arm: Option<usize>,
    total_pulls: usize,
    rng: StdRng,
}

impl EpsilonGreedyBandit {
    pub fn new(
        player_id: PlayerId,
        num_arms: usize,
        price_min: Price,
        price_max: Price,
        epsilon: f64,
    ) -> Self {
        Self {
            player_id,
            num_arms,
            price_min,
            price_max,
            epsilon: epsilon.clamp(0.0, 1.0),
            arms: (0..num_arms).map(|_| ArmStatistics::new()).collect(),
            last_arm: None,
            total_pulls: 0,
            rng: StdRng::seed_from_u64(player_id.0 as u64 * 11111 + 3),
        }
    }

    pub fn arm_stats(&self) -> &[ArmStatistics] {
        &self.arms
    }

    pub fn best_arm(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.mean_reward()
                    .partial_cmp(&b.mean_reward())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl PricingAlgorithm for EpsilonGreedyBandit {
    fn observe(&mut self, outcome: &MarketOutcome) {
        if let Some(arm) = self.last_arm {
            let reward = outcome.profits.get(self.player_id.0).map(|p| p.0).unwrap_or(0.0);
            self.arms[arm].update(reward);
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let arm = if self.rng.gen::<f64>() < self.epsilon || self.total_pulls < self.num_arms {
            // Explore: pick random arm (or ensure each arm pulled at least once)
            if self.total_pulls < self.num_arms {
                self.total_pulls
            } else {
                self.rng.gen_range(0..self.num_arms)
            }
        } else {
            self.best_arm()
        };

        self.last_arm = Some(arm);
        self.total_pulls += 1;
        let price = arm_to_price(arm, self.num_arms, self.price_min, self.price_max);
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.arms = (0..self.num_arms).map(|_| ArmStatistics::new()).collect();
        self.last_arm = None;
        self.total_pulls = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::Bandit {
            arm_counts: self.arms.iter().map(|a| a.count).collect(),
            arm_rewards: self.arms.iter().map(|a| a.total_reward).collect(),
            total_pulls: self.total_pulls,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::Bandit {
                arm_counts,
                arm_rewards,
                total_pulls,
            } => {
                for (i, arm) in self.arms.iter_mut().enumerate() {
                    arm.count = arm_counts.get(i).copied().unwrap_or(0);
                    arm.total_reward = arm_rewards.get(i).copied().unwrap_or(0.0);
                }
                self.total_pulls = *total_pulls;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected Bandit state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "EpsilonGreedyBandit"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UCB1Bandit
// ═══════════════════════════════════════════════════════════════════════════

/// Upper Confidence Bound (UCB1) algorithm.
///
/// Selects arm maximizing: mean_reward + c * sqrt(ln(t) / n_i)
pub struct UCB1Bandit {
    player_id: PlayerId,
    num_arms: usize,
    price_min: Price,
    price_max: Price,
    exploration_constant: f64,
    arms: Vec<ArmStatistics>,
    last_arm: Option<usize>,
    total_pulls: usize,
}

impl UCB1Bandit {
    pub fn new(
        player_id: PlayerId,
        num_arms: usize,
        price_min: Price,
        price_max: Price,
        exploration_constant: f64,
    ) -> Self {
        Self {
            player_id,
            num_arms,
            price_min,
            price_max,
            exploration_constant,
            arms: (0..num_arms).map(|_| ArmStatistics::new()).collect(),
            last_arm: None,
            total_pulls: 0,
        }
    }

    pub fn ucb_value(&self, arm: usize) -> f64 {
        let stats = &self.arms[arm];
        if stats.count == 0 {
            return f64::MAX;
        }
        let mean = stats.mean_reward();
        let bonus = self.exploration_constant
            * ((self.total_pulls as f64).ln() / stats.count as f64).sqrt();
        mean + bonus
    }

    pub fn arm_stats(&self) -> &[ArmStatistics] {
        &self.arms
    }
}

impl PricingAlgorithm for UCB1Bandit {
    fn observe(&mut self, outcome: &MarketOutcome) {
        if let Some(arm) = self.last_arm {
            let reward = outcome.profits.get(self.player_id.0).map(|p| p.0).unwrap_or(0.0);
            self.arms[arm].update(reward);
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let arm = if self.total_pulls < self.num_arms {
            self.total_pulls // Explore each arm once
        } else {
            (0..self.num_arms)
                .max_by(|&a, &b| {
                    self.ucb_value(a)
                        .partial_cmp(&self.ucb_value(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0)
        };

        self.last_arm = Some(arm);
        self.total_pulls += 1;
        let price = arm_to_price(arm, self.num_arms, self.price_min, self.price_max);
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.arms = (0..self.num_arms).map(|_| ArmStatistics::new()).collect();
        self.last_arm = None;
        self.total_pulls = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::Bandit {
            arm_counts: self.arms.iter().map(|a| a.count).collect(),
            arm_rewards: self.arms.iter().map(|a| a.total_reward).collect(),
            total_pulls: self.total_pulls,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::Bandit {
                arm_counts,
                arm_rewards,
                total_pulls,
            } => {
                for (i, arm) in self.arms.iter_mut().enumerate() {
                    arm.count = arm_counts.get(i).copied().unwrap_or(0);
                    arm.total_reward = arm_rewards.get(i).copied().unwrap_or(0.0);
                }
                self.total_pulls = *total_pulls;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected Bandit state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "UCB1Bandit"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ThompsonSamplingBandit
// ═══════════════════════════════════════════════════════════════════════════

/// Beta-Bernoulli Thompson Sampling bandit.
///
/// Maintains Beta(alpha, beta) posterior for each arm.
/// Rewards are normalized to [0, 1] for the Beta model.
pub struct ThompsonSamplingBandit {
    player_id: PlayerId,
    num_arms: usize,
    price_min: Price,
    price_max: Price,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    arms: Vec<ArmStatistics>,
    last_arm: Option<usize>,
    total_pulls: usize,
    reward_min: f64,
    reward_max: f64,
    rng: StdRng,
}

impl ThompsonSamplingBandit {
    pub fn new(
        player_id: PlayerId,
        num_arms: usize,
        price_min: Price,
        price_max: Price,
    ) -> Self {
        Self {
            player_id,
            num_arms,
            price_min,
            price_max,
            alphas: vec![1.0; num_arms],
            betas: vec![1.0; num_arms],
            arms: (0..num_arms).map(|_| ArmStatistics::new()).collect(),
            last_arm: None,
            total_pulls: 0,
            reward_min: 0.0,
            reward_max: 1.0,
            rng: StdRng::seed_from_u64(player_id.0 as u64 * 22222 + 5),
        }
    }

    /// Normalize a reward to [0, 1] range.
    fn normalize_reward(&mut self, reward: f64) -> f64 {
        // Adaptively update bounds
        if reward < self.reward_min {
            self.reward_min = reward;
        }
        if reward > self.reward_max {
            self.reward_max = reward;
        }
        let range = self.reward_max - self.reward_min;
        if range < 1e-10 {
            return 0.5;
        }
        ((reward - self.reward_min) / range).clamp(0.0, 1.0)
    }

    pub fn arm_stats(&self) -> &[ArmStatistics] {
        &self.arms
    }
}

impl PricingAlgorithm for ThompsonSamplingBandit {
    fn observe(&mut self, outcome: &MarketOutcome) {
        if let Some(arm) = self.last_arm {
            let raw_reward = outcome.profits.get(self.player_id.0).map(|p| p.0).unwrap_or(0.0);
            self.arms[arm].update(raw_reward);
            let normalized = self.normalize_reward(raw_reward);
            // Update Beta posterior
            self.alphas[arm] += normalized;
            self.betas[arm] += 1.0 - normalized;
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        // Sample from each arm's posterior and pick the highest
        let arm = (0..self.num_arms)
            .max_by(|&a, &b| {
                let sample_a = Beta::new(self.alphas[a], self.betas[a])
                    .map(|d| d.sample(&mut self.rng))
                    .unwrap_or(0.5);
                let sample_b = Beta::new(self.alphas[b], self.betas[b])
                    .map(|d| d.sample(&mut self.rng))
                    .unwrap_or(0.5);
                sample_a
                    .partial_cmp(&sample_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        self.last_arm = Some(arm);
        self.total_pulls += 1;
        let price = arm_to_price(arm, self.num_arms, self.price_min, self.price_max);
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.alphas = vec![1.0; self.num_arms];
        self.betas = vec![1.0; self.num_arms];
        self.arms = (0..self.num_arms).map(|_| ArmStatistics::new()).collect();
        self.last_arm = None;
        self.total_pulls = 0;
        self.reward_min = 0.0;
        self.reward_max = 1.0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::Bandit {
            arm_counts: self.arms.iter().map(|a| a.count).collect(),
            arm_rewards: self.arms.iter().map(|a| a.total_reward).collect(),
            total_pulls: self.total_pulls,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::Bandit {
                arm_counts,
                arm_rewards,
                total_pulls,
            } => {
                for (i, arm) in self.arms.iter_mut().enumerate() {
                    arm.count = arm_counts.get(i).copied().unwrap_or(0);
                    arm.total_reward = arm_rewards.get(i).copied().unwrap_or(0.0);
                }
                self.total_pulls = *total_pulls;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected Bandit state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "ThompsonSamplingBandit"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXP3Bandit
// ═══════════════════════════════════════════════════════════════════════════

/// EXP3 (Exponential-weight algorithm for Exploration and Exploitation).
///
/// Designed for adversarial bandit settings with worst-case guarantees.
pub struct EXP3Bandit {
    player_id: PlayerId,
    num_arms: usize,
    price_min: Price,
    price_max: Price,
    gamma: f64,
    weights: Vec<f64>,
    arms: Vec<ArmStatistics>,
    last_arm: Option<usize>,
    last_probability: f64,
    total_pulls: usize,
    rng: StdRng,
}

impl EXP3Bandit {
    pub fn new(
        player_id: PlayerId,
        num_arms: usize,
        price_min: Price,
        price_max: Price,
        gamma: f64,
    ) -> Self {
        Self {
            player_id,
            num_arms,
            price_min,
            price_max,
            gamma: gamma.clamp(0.0, 1.0),
            weights: vec![1.0; num_arms],
            arms: (0..num_arms).map(|_| ArmStatistics::new()).collect(),
            last_arm: None,
            last_probability: 1.0 / num_arms as f64,
            total_pulls: 0,
            rng: StdRng::seed_from_u64(player_id.0 as u64 * 33333 + 11),
        }
    }

    /// Compute the probability distribution over arms.
    fn probabilities(&self) -> Vec<f64> {
        let total_weight: f64 = self.weights.iter().sum();
        let k = self.num_arms as f64;
        self.weights
            .iter()
            .map(|w| (1.0 - self.gamma) * (w / total_weight) + self.gamma / k)
            .collect()
    }

    pub fn arm_stats(&self) -> &[ArmStatistics] {
        &self.arms
    }
}

impl PricingAlgorithm for EXP3Bandit {
    fn observe(&mut self, outcome: &MarketOutcome) {
        if let Some(arm) = self.last_arm {
            let raw_reward = outcome.profits.get(self.player_id.0).map(|p| p.0).unwrap_or(0.0);
            self.arms[arm].update(raw_reward);

            // Normalize reward to [0, 1]
            let max_possible = self.price_max.0 * 10.0; // rough upper bound
            let normalized = (raw_reward / max_possible.max(1.0)).clamp(0.0, 1.0);

            // Importance-weighted update
            let estimated_reward = normalized / self.last_probability.max(1e-10);
            self.weights[arm] *= (self.gamma * estimated_reward / self.num_arms as f64).exp();

            // Prevent numerical overflow
            let max_weight = self.weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_weight > 1e100 {
                for w in &mut self.weights {
                    *w /= max_weight;
                }
            }
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let probs = self.probabilities();

        // Sample arm according to probability distribution
        let r = self.rng.gen::<f64>();
        let mut cumulative = 0.0;
        let mut arm = self.num_arms - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                arm = i;
                break;
            }
        }

        self.last_arm = Some(arm);
        self.last_probability = probs[arm];
        self.total_pulls += 1;
        let price = arm_to_price(arm, self.num_arms, self.price_min, self.price_max);
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.weights = vec![1.0; self.num_arms];
        self.arms = (0..self.num_arms).map(|_| ArmStatistics::new()).collect();
        self.last_arm = None;
        self.last_probability = 1.0 / self.num_arms as f64;
        self.total_pulls = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::Bandit {
            arm_counts: self.arms.iter().map(|a| a.count).collect(),
            arm_rewards: self.arms.iter().map(|a| a.total_reward).collect(),
            total_pulls: self.total_pulls,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::Bandit {
                arm_counts,
                arm_rewards,
                total_pulls,
            } => {
                for (i, arm) in self.arms.iter_mut().enumerate() {
                    arm.count = arm_counts.get(i).copied().unwrap_or(0);
                    arm.total_reward = arm_rewards.get(i).copied().unwrap_or(0.0);
                }
                self.total_pulls = *total_pulls;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected Bandit state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "EXP3Bandit"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(profits: Vec<f64>) -> MarketOutcome {
        let n = profits.len();
        MarketOutcome::new(0, vec![5.0; n], vec![1.0; n], profits)
    }

    // ── ArmStatistics tests ─────────────────────────────────────────────

    #[test]
    fn test_arm_statistics_new() {
        let arm = ArmStatistics::new();
        assert_eq!(arm.count, 0);
        assert_eq!(arm.mean_reward(), 0.0);
    }

    #[test]
    fn test_arm_statistics_update() {
        let mut arm = ArmStatistics::new();
        arm.update(4.0);
        arm.update(6.0);
        assert_eq!(arm.count, 2);
        assert!((arm.mean_reward() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_arm_statistics_variance() {
        let mut arm = ArmStatistics::new();
        arm.update(4.0);
        arm.update(6.0);
        assert!(arm.variance() >= 0.0);
    }

    #[test]
    fn test_arm_to_price() {
        assert!((arm_to_price(0, 5, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((arm_to_price(4, 5, 0.0, 10.0) - 10.0).abs() < 1e-10);
        assert!((arm_to_price(2, 5, 0.0, 10.0) - 5.0).abs() < 1e-10);
    }

    // ── EpsilonGreedyBandit tests ───────────────────────────────────────

    #[test]
    fn test_epsilon_greedy_creation() {
        let bandit = EpsilonGreedyBandit::new(0, 10, 0.0, 10.0, 0.1);
        assert_eq!(bandit.name(), "EpsilonGreedyBandit");
        assert_eq!(bandit.player_id(), 0);
        assert_eq!(bandit.arm_stats().len(), 10);
    }

    #[test]
    fn test_epsilon_greedy_explore_all_arms() {
        let mut bandit = EpsilonGreedyBandit::new(0, 5, 0.0, 10.0, 0.1);
        for i in 0..5 {
            let action = bandit.act(i);
            assert!(action.price >= 0.0 && action.price <= 10.0);
            bandit.observe(&make_outcome(vec![3.0]));
        }
        // All arms should have been explored
        for arm in bandit.arm_stats() {
            assert_eq!(arm.count, 1);
        }
    }

    #[test]
    fn test_epsilon_greedy_reset() {
        let mut bandit = EpsilonGreedyBandit::new(0, 5, 0.0, 10.0, 0.1);
        bandit.act(0);
        bandit.observe(&make_outcome(vec![3.0]));
        bandit.reset();
        assert_eq!(bandit.arm_stats()[0].count, 0);
    }

    #[test]
    fn test_epsilon_greedy_state_serialization() {
        let mut bandit = EpsilonGreedyBandit::new(0, 5, 0.0, 10.0, 0.1);
        bandit.act(0);
        bandit.observe(&make_outcome(vec![3.0]));
        let state = bandit.get_state();

        let mut bandit2 = EpsilonGreedyBandit::new(0, 5, 0.0, 10.0, 0.1);
        bandit2.set_state(&state).unwrap();
    }

    // ── UCB1 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ucb1_creation() {
        let bandit = UCB1Bandit::new(0, 10, 0.0, 10.0, 2.0);
        assert_eq!(bandit.name(), "UCB1Bandit");
    }

    #[test]
    fn test_ucb1_explores_unpulled_arms() {
        let bandit = UCB1Bandit::new(0, 5, 0.0, 10.0, 2.0);
        // Unpulled arms should have MAX UCB
        assert_eq!(bandit.ucb_value(0), f64::MAX);
    }

    #[test]
    fn test_ucb1_observe_and_act() {
        let mut bandit = UCB1Bandit::new(0, 5, 0.0, 10.0, 2.0);
        for i in 0..10 {
            let action = bandit.act(i);
            assert!(action.price >= 0.0 && action.price <= 10.0);
            bandit.observe(&make_outcome(vec![action.price]));
        }
        assert!(bandit.arm_stats().iter().any(|a| a.count > 0));
    }

    // ── Thompson Sampling tests ─────────────────────────────────────────

    #[test]
    fn test_thompson_creation() {
        let bandit = ThompsonSamplingBandit::new(0, 10, 0.0, 10.0);
        assert_eq!(bandit.name(), "ThompsonSamplingBandit");
        assert_eq!(bandit.arm_stats().len(), 10);
    }

    #[test]
    fn test_thompson_observe_and_act() {
        let mut bandit = ThompsonSamplingBandit::new(0, 5, 0.0, 10.0);
        for i in 0..20 {
            let action = bandit.act(i);
            assert!(action.price >= 0.0 && action.price <= 10.0);
            bandit.observe(&make_outcome(vec![action.price * 0.5]));
        }
    }

    #[test]
    fn test_thompson_reset() {
        let mut bandit = ThompsonSamplingBandit::new(0, 5, 0.0, 10.0);
        bandit.act(0);
        bandit.observe(&make_outcome(vec![3.0]));
        bandit.reset();
        for arm in bandit.arm_stats() {
            assert_eq!(arm.count, 0);
        }
    }

    // ── EXP3 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_exp3_creation() {
        let bandit = EXP3Bandit::new(0, 10, 0.0, 10.0, 0.1);
        assert_eq!(bandit.name(), "EXP3Bandit");
    }

    #[test]
    fn test_exp3_probabilities() {
        let bandit = EXP3Bandit::new(0, 4, 0.0, 10.0, 0.1);
        let probs = bandit.probabilities();
        assert_eq!(probs.len(), 4);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp3_observe_and_act() {
        let mut bandit = EXP3Bandit::new(0, 5, 0.0, 10.0, 0.1);
        for i in 0..20 {
            let action = bandit.act(i);
            assert!(action.price >= 0.0 && action.price <= 10.0);
            bandit.observe(&make_outcome(vec![action.price * 0.3]));
        }
    }

    #[test]
    fn test_exp3_reset() {
        let mut bandit = EXP3Bandit::new(0, 5, 0.0, 10.0, 0.1);
        bandit.act(0);
        bandit.observe(&make_outcome(vec![3.0]));
        bandit.reset();
        assert_eq!(bandit.total_pulls, 0);
    }

    #[test]
    fn test_bandit_wrong_state_variant() {
        let mut bandit = EpsilonGreedyBandit::new(0, 5, 0.0, 10.0, 0.1);
        let state = AlgorithmState::Empty;
        assert!(bandit.set_state(&state).is_err());
    }
}
