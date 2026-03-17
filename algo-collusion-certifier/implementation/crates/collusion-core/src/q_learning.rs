//! Complete Q-learning implementation for pricing algorithms.
//!
//! Provides tabular Q-learning agents with configurable exploration,
//! experience replay, Boltzmann exploration, and convergence detection.

use crate::algorithm::{AlgorithmState, PricingAlgorithm};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, MarketOutcome, PlayerAction, PlayerId, Price, RoundNumber};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Decay schedule for epsilon or learning rate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecaySchedule {
    /// epsilon_t = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * t / total)
    Linear,
    /// epsilon_t = max(epsilon_end, epsilon_start * decay^t)
    Exponential,
    /// epsilon_t = max(epsilon_end, epsilon_start / (1 + decay * t))
    Inverse,
}

impl DecaySchedule {
    pub fn compute(&self, start: f64, end: f64, decay: f64, step: usize, total: usize) -> f64 {
        match self {
            DecaySchedule::Linear => {
                let frac = step as f64 / total.max(1) as f64;
                (start - (start - end) * frac).max(end)
            }
            DecaySchedule::Exponential => {
                (start * decay.powi(step as i32)).max(end)
            }
            DecaySchedule::Inverse => {
                (start / (1.0 + decay * step as f64)).max(end)
            }
        }
    }
}

/// Configuration for a Q-learning agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLearningConfig {
    pub player_id: PlayerId,
    pub num_price_levels: usize,
    pub price_min: Price,
    pub price_max: Price,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub decay_schedule: DecaySchedule,
    pub num_state_bins: usize,
    pub replay_buffer_size: usize,
    pub use_replay: bool,
    pub batch_size: usize,
}

impl Default for QLearningConfig {
    fn default() -> Self {
        Self {
            player_id: PlayerId(0),
            num_price_levels: 15,
            price_min: Price(0.0),
            price_max: Price(10.0),
            learning_rate: 0.1,
            discount_factor: 0.95,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            decay_schedule: DecaySchedule::Exponential,
            num_state_bins: 15,
            replay_buffer_size: 10_000,
            use_replay: false,
            batch_size: 32,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// State discretization
// ═══════════════════════════════════════════════════════════════════════════

/// Discretizes continuous prices into state bins.
#[derive(Debug, Clone)]
struct StateDiscretizer {
    num_bins: usize,
    price_min: Price,
    price_max: Price,
}

impl StateDiscretizer {
    fn new(num_bins: usize, price_min: Price, price_max: Price) -> Self {
        Self { num_bins, price_min, price_max }
    }

    /// Map a continuous price to a discrete bin index.
    fn discretize_price(&self, price: Price) -> usize {
        if self.price_max <= self.price_min || self.num_bins == 0 {
            return 0;
        }
        let clamped = price.clamp(self.price_min, self.price_max);
        let fraction = (clamped - self.price_min) / (self.price_max - self.price_min);
        let bin = (fraction * self.num_bins as f64) as usize;
        bin.min(self.num_bins - 1)
    }

    /// Map all prices into a single discrete state index.
    fn discretize_outcome(&self, prices: &[Price]) -> u64 {
        let mut state: u64 = 0;
        for (i, &p) in prices.iter().enumerate() {
            let bin = self.discretize_price(p) as u64;
            state += bin * (self.num_bins as u64).pow(i as u32);
        }
        state
    }

    /// Convert an action index to a price.
    fn action_to_price(&self, action: usize, num_actions: usize) -> Price {
        if num_actions <= 1 {
            return (self.price_min + self.price_max) / 2.0;
        }
        self.price_min + (action as f64) * (self.price_max - self.price_min) / (num_actions - 1) as f64
    }

    /// Convert a price to the nearest action index.
    fn price_to_action(&self, price: Price, num_actions: usize) -> usize {
        if num_actions <= 1 {
            return 0;
        }
        let clamped = price.clamp(self.price_min, self.price_max);
        let fraction = (clamped - self.price_min) / (self.price_max - self.price_min);
        let action = (fraction * (num_actions - 1) as f64).round() as usize;
        action.min(num_actions - 1)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Experience replay buffer
// ═══════════════════════════════════════════════════════════════════════════

/// Circular buffer for experience replay.
#[derive(Debug, Clone)]
struct ReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
    position: usize,
    len: usize,
}

#[derive(Debug, Clone)]
struct Experience {
    state: u64,
    action: u64,
    reward: f64,
    next_state: u64,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity.min(1024)),
            capacity,
            position: 0,
            len: 0,
        }
    }

    fn push(&mut self, exp: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(exp);
        } else {
            self.buffer[self.position] = exp;
        }
        self.position = (self.position + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<Experience> {
        let n = batch_size.min(self.len);
        let indices: Vec<usize> = (0..n)
            .map(|_| rng.gen_range(0..self.len))
            .collect();
        indices.iter().map(|&i| self.buffer[i].clone()).collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QLearningAgent
// ═══════════════════════════════════════════════════════════════════════════

/// Complete tabular Q-learning agent.
pub struct QLearningAgent {
    config: QLearningConfig,
    q_table: HashMap<(u64, u64), f64>,
    discretizer: StateDiscretizer,
    current_state: Option<u64>,
    last_action: Option<u64>,
    last_reward: Option<f64>,
    epsilon: f64,
    total_steps: usize,
    episode: usize,
    rng: StdRng,
    replay_buffer: ReplayBuffer,
    q_change_history: Vec<f64>,
    cumulative_reward: f64,
}

impl QLearningAgent {
    pub fn new(config: QLearningConfig) -> Self {
        let discretizer = StateDiscretizer::new(
            config.num_state_bins,
            config.price_min,
            config.price_max,
        );
        let epsilon = config.epsilon_start;
        let replay_capacity = if config.use_replay { config.replay_buffer_size } else { 0 };
        let rng = StdRng::seed_from_u64(config.player_id.0 as u64 * 12345 + 42);
        Self {
            config,
            q_table: HashMap::new(),
            discretizer,
            current_state: None,
            last_action: None,
            last_reward: None,
            epsilon,
            total_steps: 0,
            episode: 0,
            rng,
            replay_buffer: ReplayBuffer::new(replay_capacity),
            q_change_history: Vec::new(),
            cumulative_reward: 0.0,
        }
    }

    /// Get Q-value for a (state, action) pair, defaulting to 0.
    pub fn q_value(&self, state: u64, action: u64) -> f64 {
        self.q_table.get(&(state, action)).copied().unwrap_or(0.0)
    }

    /// Set Q-value for a (state, action) pair.
    pub fn set_q_value(&mut self, state: u64, action: u64, value: f64) {
        self.q_table.insert((state, action), value);
    }

    /// Get the best action for a state (argmax Q(s, a)).
    pub fn best_action(&self, state: u64) -> u64 {
        let mut best_a = 0u64;
        let mut best_q = f64::NEG_INFINITY;
        for a in 0..self.config.num_price_levels as u64 {
            let q = self.q_value(state, a);
            if q > best_q {
                best_q = q;
                best_a = a;
            }
        }
        best_a
    }

    /// Get the max Q-value for a state.
    fn max_q(&self, state: u64) -> f64 {
        (0..self.config.num_price_levels as u64)
            .map(|a| self.q_value(state, a))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Epsilon-greedy action selection.
    fn epsilon_greedy_action(&mut self, state: u64) -> u64 {
        if self.rng.gen::<f64>() < self.epsilon {
            self.rng.gen_range(0..self.config.num_price_levels as u64)
        } else {
            self.best_action(state)
        }
    }

    /// Boltzmann (softmax) exploration action selection.
    pub fn boltzmann_action(&mut self, state: u64, temperature: f64) -> u64 {
        let num_actions = self.config.num_price_levels;
        let q_values: Vec<f64> = (0..num_actions as u64)
            .map(|a| self.q_value(state, a))
            .collect();

        let max_q = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = q_values
            .iter()
            .map(|q| ((q - max_q) / temperature.max(1e-10)).exp())
            .collect();

        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return self.rng.gen_range(0..num_actions as u64);
        }

        let dist = WeightedIndex::new(&weights);
        match dist {
            Ok(d) => d.sample(&mut self.rng) as u64,
            Err(_) => self.rng.gen_range(0..num_actions as u64),
        }
    }

    /// Perform a Q-value update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    fn update_q(&mut self, state: u64, action: u64, reward: f64, next_state: u64) {
        let old_q = self.q_value(state, action);
        let max_next_q = self.max_q(next_state);
        let target = reward + self.config.discount_factor * max_next_q;
        let new_q = old_q + self.config.learning_rate * (target - old_q);
        let change = (new_q - old_q).abs();
        self.q_change_history.push(change);
        self.set_q_value(state, action, new_q);
    }

    /// Replay a batch of experiences from the buffer.
    fn replay_batch(&mut self) {
        if self.replay_buffer.len() < self.config.batch_size {
            return;
        }
        let batch = self.replay_buffer.sample(self.config.batch_size, &mut self.rng.clone());
        for exp in batch {
            let old_q = self.q_value(exp.state, exp.action);
            let max_next_q = self.max_q(exp.next_state);
            let target = exp.reward + self.config.discount_factor * max_next_q;
            let new_q = old_q + self.config.learning_rate * (target - old_q);
            self.set_q_value(exp.state, exp.action, new_q);
        }
    }

    /// Update epsilon according to the decay schedule.
    fn decay_epsilon(&mut self) {
        self.epsilon = self.config.decay_schedule.compute(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.epsilon_decay,
            self.total_steps,
            1_000_000,
        );
    }

    /// Check if Q-values have converged (mean change below threshold).
    pub fn has_converged(&self, window: usize, threshold: f64) -> bool {
        if self.q_change_history.len() < window {
            return false;
        }
        let recent = &self.q_change_history[self.q_change_history.len() - window..];
        let mean_change: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        mean_change < threshold
    }

    /// Run a complete training loop: N episodes of M rounds.
    pub fn train<F>(
        &mut self,
        num_episodes: usize,
        rounds_per_episode: usize,
        mut env_step: F,
    ) where
        F: FnMut(Price) -> (f64, Vec<Price>),
    {
        for _ in 0..num_episodes {
            self.reset();
            self.episode += 1;
            let initial_prices = vec![
                (self.config.price_min + self.config.price_max) / 2.0;
                self.config.num_state_bins.max(1)
            ];
            self.current_state = Some(self.discretizer.discretize_outcome(&initial_prices));

            for _ in 0..rounds_per_episode {
                let state = self.current_state.unwrap_or(0);
                let action = self.epsilon_greedy_action(state);
                let price = self.discretizer.action_to_price(
                    action as usize,
                    self.config.num_price_levels,
                );

                let (reward, new_prices) = env_step(price);
                let next_state = self.discretizer.discretize_outcome(&new_prices);

                self.update_q(state, action, reward, next_state);
                self.cumulative_reward += reward;

                if self.config.use_replay {
                    self.replay_buffer.push(Experience {
                        state,
                        action,
                        reward,
                        next_state,
                    });
                    self.replay_batch();
                }

                self.current_state = Some(next_state);
                self.last_action = Some(action);
                self.last_reward = Some(reward);
                self.total_steps += 1;
                self.decay_epsilon();
            }
        }
    }

    /// Get the current epsilon value.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get the Q-table size.
    pub fn q_table_size(&self) -> usize {
        self.q_table.len()
    }

    /// Get cumulative reward.
    pub fn cumulative_reward(&self) -> f64 {
        self.cumulative_reward
    }

    /// Serialize Q-table to JSON.
    pub fn serialize_q_table(&self) -> CollusionResult<String> {
        // Convert to string keys for JSON serialization
        let table: HashMap<String, f64> = self.q_table
            .iter()
            .map(|((s, a), v)| (format!("{},{}", s, a), *v))
            .collect();
        serde_json::to_string_pretty(&table)
            .map_err(|e| CollusionError::Serialization(e.to_string()))
    }

    /// Load Q-table from JSON.
    pub fn deserialize_q_table(&mut self, json: &str) -> CollusionResult<()> {
        let table: HashMap<String, f64> = serde_json::from_str(json)
            .map_err(|e| CollusionError::Serialization(e.to_string()))?;
        self.q_table.clear();
        for (key, value) in table {
            let parts: Vec<&str> = key.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(s), Ok(a)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
                    self.q_table.insert((s, a), value);
                }
            }
        }
        Ok(())
    }
}

impl PricingAlgorithm for QLearningAgent {
    fn observe(&mut self, outcome: &MarketOutcome) {
        let new_state = self.discretizer.discretize_outcome(&outcome.prices);
        let reward = outcome.profits.get(self.config.player_id.0).map(|p| p.0).unwrap_or(0.0);

        if let (Some(state), Some(action)) = (self.current_state, self.last_action) {
            self.update_q(state, action, reward, new_state);
            if self.config.use_replay {
                self.replay_buffer.push(Experience {
                    state,
                    action,
                    reward,
                    next_state: new_state,
                });
                self.replay_batch();
            }
        }

        self.current_state = Some(new_state);
        self.last_reward = Some(reward);
        self.cumulative_reward += reward;
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let state = self.current_state.unwrap_or(0);
        let action = self.epsilon_greedy_action(state);
        self.last_action = Some(action);
        self.total_steps += 1;
        self.decay_epsilon();

        let price = self.discretizer.action_to_price(
            action as usize,
            self.config.num_price_levels,
        );
        PlayerAction::new(self.config.player_id, price)
    }

    fn reset(&mut self) {
        self.current_state = None;
        self.last_action = None;
        self.last_reward = None;
        self.cumulative_reward = 0.0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::QLearning {
            q_table: self.q_table.clone(),
            episode: self.episode,
            epsilon: self.epsilon,
            total_steps: self.total_steps,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::QLearning {
                q_table,
                episode,
                epsilon,
                total_steps,
            } => {
                self.q_table = q_table.clone();
                self.episode = *episode;
                self.epsilon = *epsilon;
                self.total_steps = *total_steps;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected QLearning state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "QLearning"
    }

    fn player_id(&self) -> PlayerId {
        self.config.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-agent independent Q-learning
// ═══════════════════════════════════════════════════════════════════════════

/// Multiple independent Q-learning agents, each with their own Q-table.
pub struct MultiAgentQLearning {
    pub agents: Vec<QLearningAgent>,
}

impl MultiAgentQLearning {
    pub fn new(configs: Vec<QLearningConfig>) -> Self {
        let agents = configs.into_iter().map(QLearningAgent::new).collect();
        Self { agents }
    }

    pub fn symmetric(num_agents: usize, base_config: QLearningConfig) -> Self {
        let configs: Vec<QLearningConfig> = (0..num_agents)
            .map(|i| {
                let mut c = base_config.clone();
                c.player_id = PlayerId(i);
                c
            })
            .collect();
        Self::new(configs)
    }

    pub fn observe_all(&mut self, outcome: &MarketOutcome) {
        for agent in &mut self.agents {
            agent.observe(outcome);
        }
    }

    pub fn act_all(&mut self, round: RoundNumber) -> Vec<PlayerAction> {
        self.agents.iter_mut().map(|a| a.act(round)).collect()
    }

    pub fn reset_all(&mut self) {
        for agent in &mut self.agents {
            agent.reset();
        }
    }

    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    pub fn all_converged(&self, window: usize, threshold: f64) -> bool {
        self.agents.iter().all(|a| a.has_converged(window, threshold))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> QLearningConfig {
        QLearningConfig {
            player_id: 0,
            num_price_levels: 5,
            price_min: 0.0,
            price_max: 10.0,
            learning_rate: 0.1,
            discount_factor: 0.95,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.99,
            decay_schedule: DecaySchedule::Exponential,
            num_state_bins: 5,
            replay_buffer_size: 100,
            use_replay: false,
            batch_size: 4,
        }
    }

    fn make_outcome(prices: Vec<f64>, profits: Vec<f64>) -> MarketOutcome {
        let n = prices.len();
        MarketOutcome::new(0, prices, vec![1.0; n], profits)
    }

    #[test]
    fn test_decay_schedule_linear() {
        let schedule = DecaySchedule::Linear;
        assert!((schedule.compute(1.0, 0.0, 0.0, 0, 100) - 1.0).abs() < 1e-10);
        assert!((schedule.compute(1.0, 0.0, 0.0, 50, 100) - 0.5).abs() < 1e-10);
        assert!((schedule.compute(1.0, 0.0, 0.0, 100, 100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_schedule_exponential() {
        let schedule = DecaySchedule::Exponential;
        let v = schedule.compute(1.0, 0.01, 0.9, 10, 100);
        let expected = (1.0 * 0.9_f64.powi(10)).max(0.01);
        assert!((v - expected).abs() < 1e-10);
    }

    #[test]
    fn test_decay_schedule_inverse() {
        let schedule = DecaySchedule::Inverse;
        let v = schedule.compute(1.0, 0.01, 0.1, 10, 100);
        let expected = (1.0 / (1.0 + 0.1 * 10.0)).max(0.01);
        assert!((v - expected).abs() < 1e-10);
    }

    #[test]
    fn test_state_discretizer() {
        let d = StateDiscretizer::new(10, 0.0, 10.0);
        assert_eq!(d.discretize_price(0.0), 0);
        assert_eq!(d.discretize_price(10.0), 9);
        assert_eq!(d.discretize_price(5.0), 5);
        assert_eq!(d.discretize_price(-1.0), 0);
        assert_eq!(d.discretize_price(11.0), 9);
    }

    #[test]
    fn test_action_to_price() {
        let d = StateDiscretizer::new(5, 0.0, 10.0);
        assert!((d.action_to_price(0, 5) - 0.0).abs() < 1e-10);
        assert!((d.action_to_price(4, 5) - 10.0).abs() < 1e-10);
        assert!((d.action_to_price(2, 5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_price_to_action() {
        let d = StateDiscretizer::new(5, 0.0, 10.0);
        assert_eq!(d.price_to_action(0.0, 5), 0);
        assert_eq!(d.price_to_action(10.0, 5), 4);
        assert_eq!(d.price_to_action(5.0, 5), 2);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buf = ReplayBuffer::new(3);
        assert!(buf.is_empty());

        buf.push(Experience { state: 0, action: 0, reward: 1.0, next_state: 1 });
        buf.push(Experience { state: 1, action: 1, reward: 2.0, next_state: 2 });
        assert_eq!(buf.len(), 2);

        buf.push(Experience { state: 2, action: 2, reward: 3.0, next_state: 0 });
        buf.push(Experience { state: 3, action: 0, reward: 4.0, next_state: 1 });
        assert_eq!(buf.len(), 3); // Circular, so capped at capacity
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.push(Experience {
                state: i,
                action: i % 5,
                reward: i as f64,
                next_state: i + 1,
            });
        }
        let mut rng = StdRng::seed_from_u64(42);
        let batch = buf.sample(10, &mut rng);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_q_learning_agent_creation() {
        let config = default_config();
        let agent = QLearningAgent::new(config);
        assert_eq!(agent.q_table_size(), 0);
        assert!((agent.epsilon() - 1.0).abs() < 1e-10);
        assert_eq!(agent.player_id(), 0);
        assert_eq!(agent.name(), "QLearning");
    }

    #[test]
    fn test_q_learning_observe_and_act() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);

        let outcome = make_outcome(vec![5.0, 5.0], vec![4.0, 4.0]);
        agent.observe(&outcome);

        let action = agent.act(0);
        assert_eq!(action.player_id, 0);
        assert!(action.price >= 0.0 && action.price <= 10.0);
    }

    #[test]
    fn test_q_learning_q_update() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);

        agent.set_q_value(0, 0, 0.0);
        agent.update_q(0, 0, 1.0, 1);

        let q = agent.q_value(0, 0);
        // Q(0,0) += 0.1 * (1.0 + 0.95 * 0.0 - 0.0) = 0.1
        assert!((q - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_q_learning_convergence_detection() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);
        assert!(!agent.has_converged(10, 0.001));

        for _ in 0..20 {
            agent.q_change_history.push(0.0001);
        }
        assert!(agent.has_converged(10, 0.001));
    }

    #[test]
    fn test_q_learning_reset() {
        let mut config = default_config();
        config.epsilon_start = 0.5;
        let mut agent = QLearningAgent::new(config);

        let outcome = make_outcome(vec![5.0, 5.0], vec![4.0, 4.0]);
        agent.observe(&outcome);
        agent.act(0);
        assert!(agent.current_state.is_some());

        agent.reset();
        assert!(agent.current_state.is_none());
        assert!(agent.last_action.is_none());
    }

    #[test]
    fn test_q_learning_state_serialization() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);
        agent.set_q_value(0, 1, 0.5);
        agent.set_q_value(1, 2, 0.3);

        let state = agent.get_state();
        let json = state.to_json().unwrap();

        let mut agent2 = QLearningAgent::new(default_config());
        let restored_state = AlgorithmState::from_json(&json).unwrap();
        agent2.set_state(&restored_state).unwrap();
        assert!((agent2.q_value(0, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_q_table_json_serialization() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);
        agent.set_q_value(0, 0, 1.0);
        agent.set_q_value(1, 2, 3.0);

        let json = agent.serialize_q_table().unwrap();
        let mut agent2 = QLearningAgent::new(default_config());
        agent2.deserialize_q_table(&json).unwrap();

        assert!((agent2.q_value(0, 0) - 1.0).abs() < 1e-10);
        assert!((agent2.q_value(1, 2) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_boltzmann_action() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);
        agent.set_q_value(0, 0, 10.0);
        agent.set_q_value(0, 1, 0.0);
        agent.set_q_value(0, 2, 0.0);
        agent.set_q_value(0, 3, 0.0);
        agent.set_q_value(0, 4, 0.0);

        let mut counts = [0usize; 5];
        for _ in 0..1000 {
            let a = agent.boltzmann_action(0, 0.1);
            counts[a as usize] += 1;
        }
        // Action 0 should be selected most often
        assert!(counts[0] > counts[1]);
    }

    #[test]
    fn test_multi_agent_q_learning() {
        let config = default_config();
        let mut multi = MultiAgentQLearning::symmetric(2, config);
        assert_eq!(multi.num_agents(), 2);

        let outcome = make_outcome(vec![5.0, 5.0], vec![4.0, 4.0]);
        multi.observe_all(&outcome);

        let actions = multi.act_all(0);
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].player_id, 0);
        assert_eq!(actions[1].player_id, 1);
    }

    #[test]
    fn test_multi_agent_convergence() {
        let config = default_config();
        let mut multi = MultiAgentQLearning::symmetric(2, config);
        assert!(!multi.all_converged(10, 0.001));

        for agent in &mut multi.agents {
            for _ in 0..20 {
                agent.q_change_history.push(0.0001);
            }
        }
        assert!(multi.all_converged(10, 0.001));
    }

    #[test]
    fn test_multi_agent_reset() {
        let config = default_config();
        let mut multi = MultiAgentQLearning::symmetric(2, config);
        let outcome = make_outcome(vec![5.0, 5.0], vec![4.0, 4.0]);
        multi.observe_all(&outcome);
        multi.reset_all();
        for agent in &multi.agents {
            assert!(agent.current_state.is_none());
        }
    }

    #[test]
    fn test_q_learning_with_replay() {
        let mut config = default_config();
        config.use_replay = true;
        config.replay_buffer_size = 100;
        config.batch_size = 4;
        let mut agent = QLearningAgent::new(config);

        for i in 0..20 {
            let outcome = make_outcome(
                vec![3.0 + (i as f64) * 0.1, 5.0],
                vec![2.0, 4.0],
            );
            agent.observe(&outcome);
            agent.act(i);
        }
        assert!(agent.q_table_size() > 0);
    }

    #[test]
    fn test_set_state_wrong_variant() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);
        let state = AlgorithmState::GrimTrigger {
            triggered: false,
            trigger_round: None,
            punishment_count: 0,
        };
        assert!(agent.set_state(&state).is_err());
    }

    #[test]
    fn test_train_loop() {
        let config = default_config();
        let mut agent = QLearningAgent::new(config);

        agent.train(2, 10, |_price| {
            (1.0, vec![5.0, 5.0])
        });

        assert!(agent.q_table_size() > 0);
        assert!(agent.total_steps > 0);
    }
}
