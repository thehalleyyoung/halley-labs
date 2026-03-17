//! Deep Q-Network implementation (pure Rust, no PyTorch dependency).
//!
//! Provides a feedforward neural network with backpropagation, Adam optimizer,
//! experience replay, and a DQN agent that uses the network as a Q-function
//! approximator. Includes target network with periodic sync.

use crate::algorithm::{AlgorithmState, PricingAlgorithm};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, MarketOutcome, PlayerAction, PlayerId, Price, RoundNumber};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// Activation functions
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationFn {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl ActivationFn {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFn::ReLU => x.max(0.0),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::Linear => x,
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFn::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFn::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            ActivationFn::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFn::Linear => 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: ActivationFn,
    input_size: usize,
    output_size: usize,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationFn, rng: &mut impl Rng) -> Self {
        // Xavier initialization
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        let biases = vec![0.0; output_size];
        Self { weights, biases, activation, input_size, output_size }
    }

    /// Forward pass: output = activation(W * input + b)
    pub fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut pre_activation = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = self.biases[i];
            for j in 0..self.input_size.min(input.len()) {
                sum += self.weights[i][j] * input[j];
            }
            pre_activation[i] = sum;
        }
        let output: Vec<f64> = pre_activation.iter().map(|&z| self.activation.apply(z)).collect();
        (output, pre_activation)
    }

    pub fn input_size(&self) -> usize { self.input_size }
    pub fn output_size(&self) -> usize { self.output_size }
}

// ═══════════════════════════════════════════════════════════════════════════
// SimpleNeuralNetwork
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleNeuralNetwork {
    pub layers: Vec<Layer>,
}

impl SimpleNeuralNetwork {
    pub fn new(layer_sizes: &[usize], activations: &[ActivationFn], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        assert!(layer_sizes.len() >= 2, "Need at least input and output layers");
        assert_eq!(
            activations.len(),
            layer_sizes.len() - 1,
            "Need one activation per layer transition"
        );

        let layers: Vec<Layer> = (0..layer_sizes.len() - 1)
            .map(|i| Layer::new(layer_sizes[i], layer_sizes[i + 1], activations[i], &mut rng))
            .collect();
        Self { layers }
    }

    /// Forward pass through all layers; returns final output and all intermediate values.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            let (output, _) = layer.forward(&current);
            current = output;
        }
        current
    }

    /// Full forward pass returning all intermediate activations and pre-activations.
    fn forward_detailed(&self, input: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut activations = vec![input.to_vec()];
        let mut pre_activations = Vec::new();
        let mut current = input.to_vec();

        for layer in &self.layers {
            let (output, pre_act) = layer.forward(&current);
            pre_activations.push(pre_act);
            activations.push(output.clone());
            current = output;
        }
        (activations, pre_activations)
    }

    /// Backpropagation: compute gradients given input, target output, and loss.
    pub fn backpropagate(
        &self,
        input: &[f64],
        target: &[f64],
    ) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let (activations, pre_activations) = self.forward_detailed(input);
        let num_layers = self.layers.len();
        let mut weight_grads: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_layers);
        let mut bias_grads: Vec<Vec<f64>> = Vec::with_capacity(num_layers);

        // Initialize gradients storage
        for layer in &self.layers {
            weight_grads.push(vec![vec![0.0; layer.input_size()]; layer.output_size()]);
            bias_grads.push(vec![0.0; layer.output_size()]);
        }

        // Output layer delta: dL/dz = (output - target) * activation'(z)
        let output = &activations[num_layers];
        let mut delta: Vec<f64> = output
            .iter()
            .zip(target.iter())
            .zip(pre_activations[num_layers - 1].iter())
            .map(|((o, t), z)| (o - t) * self.layers[num_layers - 1].activation.derivative(*z))
            .collect();

        // Compute gradients for output layer
        for i in 0..self.layers[num_layers - 1].output_size() {
            bias_grads[num_layers - 1][i] = delta[i];
            for j in 0..self.layers[num_layers - 1].input_size() {
                weight_grads[num_layers - 1][i][j] = delta[i] * activations[num_layers - 1][j];
            }
        }

        // Backpropagate through hidden layers
        for l in (0..num_layers - 1).rev() {
            let mut new_delta = vec![0.0; self.layers[l].output_size()];
            for i in 0..self.layers[l].output_size() {
                let mut sum = 0.0;
                for k in 0..self.layers[l + 1].output_size() {
                    sum += self.layers[l + 1].weights[k][i] * delta[k];
                }
                new_delta[i] = sum * self.layers[l].activation.derivative(pre_activations[l][i]);
            }
            delta = new_delta;

            for i in 0..self.layers[l].output_size() {
                bias_grads[l][i] = delta[i];
                for j in 0..self.layers[l].input_size() {
                    weight_grads[l][i][j] = delta[i] * activations[l][j];
                }
            }
        }

        (weight_grads, bias_grads)
    }

    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum()
    }

    /// Get serializable weight/bias data.
    pub fn get_weights(&self) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let weights = self.layers.iter().map(|l| l.weights.clone()).collect();
        let biases = self.layers.iter().map(|l| l.biases.clone()).collect();
        (weights, biases)
    }

    /// Set weights/biases from serialized data.
    pub fn set_weights(&mut self, weights: &[Vec<Vec<f64>>], biases: &[Vec<f64>]) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < weights.len() {
                layer.weights = weights[i].clone();
            }
            if i < biases.len() {
                layer.biases = biases[i].clone();
            }
        }
    }

    /// Copy weights from another network (for target network sync).
    pub fn copy_from(&mut self, other: &SimpleNeuralNetwork) {
        let (w, b) = other.get_weights();
        self.set_weights(&w, &b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Adam Optimizer
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    m_weights: Vec<Vec<Vec<f64>>>,
    v_weights: Vec<Vec<Vec<f64>>>,
    m_biases: Vec<Vec<f64>>,
    v_biases: Vec<Vec<f64>>,
}

impl AdamOptimizer {
    pub fn new(network: &SimpleNeuralNetwork, learning_rate: f64) -> Self {
        let m_weights: Vec<Vec<Vec<f64>>> = network
            .layers
            .iter()
            .map(|l| vec![vec![0.0; l.input_size()]; l.output_size()])
            .collect();
        let v_weights = m_weights.clone();
        let m_biases: Vec<Vec<f64>> = network
            .layers
            .iter()
            .map(|l| vec![0.0; l.output_size()])
            .collect();
        let v_biases = m_biases.clone();

        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
        }
    }

    pub fn step(
        &mut self,
        network: &mut SimpleNeuralNetwork,
        weight_grads: &[Vec<Vec<f64>>],
        bias_grads: &[Vec<f64>],
    ) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (l, layer) in network.layers.iter_mut().enumerate() {
            // Update weights
            for i in 0..layer.output_size() {
                for j in 0..layer.input_size() {
                    let g = weight_grads[l][i][j];
                    self.m_weights[l][i][j] = self.beta1 * self.m_weights[l][i][j] + (1.0 - self.beta1) * g;
                    self.v_weights[l][i][j] = self.beta2 * self.v_weights[l][i][j] + (1.0 - self.beta2) * g * g;
                    let m_hat = self.m_weights[l][i][j] / bc1;
                    let v_hat = self.v_weights[l][i][j] / bc2;
                    layer.weights[i][j] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }

            // Update biases
            for i in 0..layer.output_size() {
                let g = bias_grads[l][i];
                self.m_biases[l][i] = self.beta1 * self.m_biases[l][i] + (1.0 - self.beta1) * g;
                self.v_biases[l][i] = self.beta2 * self.v_biases[l][i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m_biases[l][i] / bc1;
                let v_hat = self.v_biases[l][i] / bc2;
                layer.biases[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Experience replay for DQN
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct DQNExperience {
    state: Vec<f64>,
    action: usize,
    reward: f64,
    next_state: Vec<f64>,
    done: bool,
}

#[derive(Debug, Clone)]
struct PrioritizedReplayBuffer {
    buffer: VecDeque<(DQNExperience, f64)>,
    capacity: usize,
}

impl PrioritizedReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, exp: DQNExperience, priority: f64) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back((exp, priority.max(1e-6)));
    }

    fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<&DQNExperience> {
        let n = batch_size.min(self.buffer.len());
        if n == 0 {
            return Vec::new();
        }

        let total_priority: f64 = self.buffer.iter().map(|(_, p)| *p).sum();
        if total_priority <= 0.0 {
            // Uniform sampling fallback
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..self.buffer.len())).collect();
            return indices.iter().map(|&i| &self.buffer[i].0).collect();
        }

        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let threshold = rng.gen::<f64>() * total_priority;
            let mut cumulative = 0.0;
            for (exp, p) in &self.buffer {
                cumulative += p;
                if cumulative >= threshold {
                    samples.push(exp);
                    break;
                }
            }
            if samples.len() < n {
                // Fallback: use last
                if let Some((exp, _)) = self.buffer.back() {
                    samples.push(exp);
                }
            }
        }
        samples.truncate(n);
        samples
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Epsilon schedule for DQN
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpsilonSchedule {
    pub start: f64,
    pub end: f64,
    pub decay: f64,
}

impl EpsilonSchedule {
    pub fn new(start: f64, end: f64, decay: f64) -> Self {
        Self { start, end, decay }
    }

    pub fn value(&self, step: usize) -> f64 {
        (self.start * self.decay.powi(step as i32)).max(self.end)
    }
}

impl Default for EpsilonSchedule {
    fn default() -> Self {
        Self { start: 1.0, end: 0.01, decay: 0.995 }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DQN Configuration
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DQNConfig {
    pub player_id: PlayerId,
    pub num_actions: usize,
    pub price_min: Price,
    pub price_max: Price,
    pub history_length: usize,
    pub hidden_sizes: Vec<usize>,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub target_update_freq: usize,
    pub replay_buffer_size: usize,
    pub batch_size: usize,
    pub num_players: usize,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            player_id: PlayerId(0),
            num_actions: 15,
            price_min: Price(0.0),
            price_max: Price(10.0),
            history_length: 10,
            hidden_sizes: vec![64, 32],
            learning_rate: 0.001,
            discount_factor: 0.95,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            target_update_freq: 100,
            replay_buffer_size: 10_000,
            batch_size: 32,
            num_players: 2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DQNAgent
// ═══════════════════════════════════════════════════════════════════════════

/// Deep Q-Network agent using a feedforward neural network.
pub struct DQNAgent {
    config: DQNConfig,
    q_network: SimpleNeuralNetwork,
    target_network: SimpleNeuralNetwork,
    optimizer: AdamOptimizer,
    replay_buffer: PrioritizedReplayBuffer,
    epsilon_schedule: EpsilonSchedule,
    price_history: VecDeque<Vec<Price>>,
    current_state: Option<Vec<f64>>,
    last_action: Option<usize>,
    total_steps: usize,
    episode: usize,
    rng: StdRng,
    training_losses: Vec<f64>,
}

impl DQNAgent {
    pub fn new(config: DQNConfig) -> Self {
        let input_size = config.history_length * config.num_players;
        let mut layer_sizes = vec![input_size];
        layer_sizes.extend_from_slice(&config.hidden_sizes);
        layer_sizes.push(config.num_actions);

        let mut activations: Vec<ActivationFn> = config
            .hidden_sizes
            .iter()
            .map(|_| ActivationFn::ReLU)
            .collect();
        activations.push(ActivationFn::Linear);

        let seed = config.player_id.0 as u64 * 99999 + 31;
        let q_network = SimpleNeuralNetwork::new(&layer_sizes, &activations, seed);
        let target_network = SimpleNeuralNetwork::new(&layer_sizes, &activations, seed);

        let optimizer = AdamOptimizer::new(&q_network, config.learning_rate);
        let replay_buffer = PrioritizedReplayBuffer::new(config.replay_buffer_size);
        let epsilon_schedule = EpsilonSchedule::new(
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        );

        Self {
            config,
            q_network,
            target_network,
            optimizer,
            replay_buffer,
            epsilon_schedule,
            price_history: VecDeque::new(),
            current_state: None,
            last_action: None,
            total_steps: 0,
            episode: 0,
            rng: StdRng::seed_from_u64(seed),
            training_losses: Vec::new(),
        }
    }

    /// Encode recent price history into a feature vector.
    fn encode_state(&self) -> Vec<f64> {
        let total_features = self.config.history_length * self.config.num_players;
        let mut features = vec![0.0; total_features];
        let price_range = (self.config.price_max - self.config.price_min).0;
        let scale = if price_range > 0.0 { 1.0 / price_range } else { 1.0 };

        for (t, prices) in self.price_history.iter().enumerate() {
            for (p_idx, &price) in prices.iter().enumerate() {
                let idx = t * self.config.num_players + p_idx;
                if idx < total_features {
                    features[idx] = (price.0 - self.config.price_min.0) * scale;
                }
            }
        }
        features
    }

    /// Map action index to price.
    fn action_to_price(&self, action: usize) -> Price {
        if self.config.num_actions <= 1 {
            return Price((self.config.price_min.0 + self.config.price_max.0) / 2.0);
        }
        Price(self.config.price_min.0
            + (action as f64) * (self.config.price_max.0 - self.config.price_min.0)
                / (self.config.num_actions - 1) as f64)
    }

    /// Select action using epsilon-greedy on Q-network output.
    fn select_action(&mut self, state: &[f64]) -> usize {
        let epsilon = self.epsilon_schedule.value(self.total_steps);
        if self.rng.gen::<f64>() < epsilon {
            self.rng.gen_range(0..self.config.num_actions)
        } else {
            let q_values = self.q_network.forward(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Train on a batch from the replay buffer.
    fn train_batch(&mut self) {
        if self.replay_buffer.len() < self.config.batch_size {
            return;
        }

        let batch: Vec<DQNExperience> = {
            let refs = self.replay_buffer.sample(self.config.batch_size, &mut self.rng.clone());
            refs.into_iter().cloned().collect()
        };

        let mut total_loss = 0.0;

        for exp in &batch {
            let q_values = self.q_network.forward(&exp.state);
            let target_q_values = self.target_network.forward(&exp.next_state);

            let max_next_q = target_q_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let target = if exp.done {
                exp.reward
            } else {
                exp.reward + self.config.discount_factor * max_next_q
            };

            // Create target vector (same as current Q-values except for the taken action)
            let mut target_vec = q_values.clone();
            target_vec[exp.action] = target;

            let loss = (q_values[exp.action] - target).powi(2);
            total_loss += loss;

            let (weight_grads, bias_grads) = self.q_network.backpropagate(&exp.state, &target_vec);
            self.optimizer.step(&mut self.q_network, &weight_grads, &bias_grads);
        }

        self.training_losses.push(total_loss / batch.len() as f64);

        // Sync target network periodically
        if self.total_steps % self.config.target_update_freq == 0 {
            self.target_network.copy_from(&self.q_network);
        }
    }

    /// Get current epsilon.
    pub fn epsilon(&self) -> f64 {
        self.epsilon_schedule.value(self.total_steps)
    }

    /// Get mean training loss.
    pub fn mean_loss(&self) -> f64 {
        if self.training_losses.is_empty() {
            return 0.0;
        }
        self.training_losses.iter().sum::<f64>() / self.training_losses.len() as f64
    }

    /// Get total parameter count.
    pub fn num_parameters(&self) -> usize {
        self.q_network.num_parameters()
    }
}

impl PricingAlgorithm for DQNAgent {
    fn observe(&mut self, outcome: &MarketOutcome) {
        // Update price history
        self.price_history.push_back(outcome.prices.clone());
        if self.price_history.len() > self.config.history_length {
            self.price_history.pop_front();
        }

        let new_state = self.encode_state();
        let reward = outcome.profits.get(self.config.player_id.0).map(|p| p.0).unwrap_or(0.0);

        // Store experience in replay buffer
        if let (Some(prev_state), Some(action)) = (&self.current_state, self.last_action) {
            let td_error = {
                let q_val = self.q_network.forward(prev_state)[action];
                let max_next = self.target_network.forward(&new_state)
                    .iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (reward + self.config.discount_factor * max_next - q_val).abs()
            };

            self.replay_buffer.push(
                DQNExperience {
                    state: prev_state.clone(),
                    action,
                    reward,
                    next_state: new_state.clone(),
                    done: false,
                },
                td_error + 1e-6,
            );
        }

        self.current_state = Some(new_state);
        self.train_batch();
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let state = self.current_state.clone().unwrap_or_else(|| self.encode_state());
        let action = self.select_action(&state);
        self.last_action = Some(action);
        self.total_steps += 1;
        let price = self.action_to_price(action);
        PlayerAction::new(self.config.player_id, price)
    }

    fn reset(&mut self) {
        self.price_history.clear();
        self.current_state = None;
        self.last_action = None;
    }

    fn get_state(&self) -> AlgorithmState {
        let (weights, biases) = self.q_network.get_weights();
        AlgorithmState::DQN {
            weights,
            biases,
            episode: self.episode,
            epsilon: self.epsilon(),
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::DQN {
                weights,
                biases,
                episode,
                epsilon: _,
            } => {
                self.q_network.set_weights(weights, biases);
                self.target_network.copy_from(&self.q_network);
                self.episode = *episode;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected DQN state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "DQN"
    }

    fn player_id(&self) -> PlayerId {
        self.config.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(prices: Vec<f64>, profits: Vec<f64>) -> MarketOutcome {
        let n = prices.len();
        MarketOutcome::new(0, prices, vec![1.0; n], profits)
    }

    // ── Activation function tests ───────────────────────────────────────

    #[test]
    fn test_relu() {
        assert_eq!(ActivationFn::ReLU.apply(-1.0), 0.0);
        assert_eq!(ActivationFn::ReLU.apply(0.0), 0.0);
        assert_eq!(ActivationFn::ReLU.apply(2.0), 2.0);
        assert_eq!(ActivationFn::ReLU.derivative(-1.0), 0.0);
        assert_eq!(ActivationFn::ReLU.derivative(1.0), 1.0);
    }

    #[test]
    fn test_sigmoid() {
        let s = ActivationFn::Sigmoid.apply(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let d = ActivationFn::Sigmoid.derivative(0.0);
        assert!((d - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tanh() {
        let t = ActivationFn::Tanh.apply(0.0);
        assert!(t.abs() < 1e-10);
        let d = ActivationFn::Tanh.derivative(0.0);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear() {
        assert_eq!(ActivationFn::Linear.apply(3.14), 3.14);
        assert_eq!(ActivationFn::Linear.derivative(3.14), 1.0);
    }

    // ── Layer tests ─────────────────────────────────────────────────────

    #[test]
    fn test_layer_forward_dimensions() {
        let mut rng = StdRng::seed_from_u64(42);
        let layer = Layer::new(4, 3, ActivationFn::ReLU, &mut rng);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (output, pre_act) = layer.forward(&input);
        assert_eq!(output.len(), 3);
        assert_eq!(pre_act.len(), 3);
    }

    #[test]
    fn test_layer_sizes() {
        let mut rng = StdRng::seed_from_u64(42);
        let layer = Layer::new(5, 3, ActivationFn::ReLU, &mut rng);
        assert_eq!(layer.input_size(), 5);
        assert_eq!(layer.output_size(), 3);
    }

    // ── Neural network tests ────────────────────────────────────────────

    #[test]
    fn test_network_forward() {
        let net = SimpleNeuralNetwork::new(
            &[4, 8, 3],
            &[ActivationFn::ReLU, ActivationFn::Linear],
            42,
        );
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = net.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_network_backprop() {
        let net = SimpleNeuralNetwork::new(
            &[2, 4, 1],
            &[ActivationFn::Sigmoid, ActivationFn::Linear],
            42,
        );
        let input = vec![0.5, 0.3];
        let target = vec![1.0];
        let (w_grads, b_grads) = net.backpropagate(&input, &target);
        assert_eq!(w_grads.len(), 2);
        assert_eq!(b_grads.len(), 2);
    }

    #[test]
    fn test_network_learning() {
        let mut net = SimpleNeuralNetwork::new(
            &[1, 8, 1],
            &[ActivationFn::ReLU, ActivationFn::Linear],
            42,
        );
        let mut optimizer = AdamOptimizer::new(&net, 0.01);

        let initial_output = net.forward(&[0.5])[0];

        // Train to output 1.0 for input 0.5
        for _ in 0..100 {
            let (w_g, b_g) = net.backpropagate(&[0.5], &[1.0]);
            optimizer.step(&mut net, &w_g, &b_g);
        }

        let final_output = net.forward(&[0.5])[0];
        // Should be closer to target
        assert!((final_output - 1.0).abs() < (initial_output - 1.0).abs());
    }

    #[test]
    fn test_network_copy() {
        let net1 = SimpleNeuralNetwork::new(
            &[2, 4, 2],
            &[ActivationFn::ReLU, ActivationFn::Linear],
            42,
        );
        let mut net2 = SimpleNeuralNetwork::new(
            &[2, 4, 2],
            &[ActivationFn::ReLU, ActivationFn::Linear],
            99,
        );

        let input = vec![1.0, 2.0];
        let out1_before = net1.forward(&input);
        let out2_before = net2.forward(&input);
        // Different seeds -> different outputs
        assert!((out1_before[0] - out2_before[0]).abs() > 1e-10 || (out1_before[1] - out2_before[1]).abs() > 1e-10);

        net2.copy_from(&net1);
        let out2_after = net2.forward(&input);
        assert!((out1_before[0] - out2_after[0]).abs() < 1e-10);
        assert!((out1_before[1] - out2_after[1]).abs() < 1e-10);
    }

    #[test]
    fn test_num_parameters() {
        let net = SimpleNeuralNetwork::new(
            &[4, 8, 3],
            &[ActivationFn::ReLU, ActivationFn::Linear],
            42,
        );
        // Layer 1: 4*8 + 8 = 40, Layer 2: 8*3 + 3 = 27
        assert_eq!(net.num_parameters(), 67);
    }

    // ── Epsilon schedule tests ──────────────────────────────────────────

    #[test]
    fn test_epsilon_schedule() {
        let sched = EpsilonSchedule::new(1.0, 0.01, 0.99);
        assert!((sched.value(0) - 1.0).abs() < 1e-10);
        assert!(sched.value(100) < 1.0);
        assert!(sched.value(1000) >= 0.01);
    }

    // ── DQN agent tests ─────────────────────────────────────────────────

    #[test]
    fn test_dqn_agent_creation() {
        let config = DQNConfig::default();
        let agent = DQNAgent::new(config);
        assert_eq!(agent.name(), "DQN");
        assert_eq!(agent.player_id(), 0);
        assert!(agent.num_parameters() > 0);
    }

    #[test]
    fn test_dqn_observe_and_act() {
        let config = DQNConfig {
            num_players: 2,
            history_length: 3,
            num_actions: 5,
            ..Default::default()
        };
        let mut agent = DQNAgent::new(config);

        let outcome = make_outcome(vec![5.0, 5.0], vec![4.0, 4.0]);
        agent.observe(&outcome);
        let action = agent.act(0);
        assert_eq!(action.player_id, 0);
        assert!(action.price >= 0.0 && action.price <= 10.0);
    }

    #[test]
    fn test_dqn_multiple_steps() {
        let config = DQNConfig {
            num_players: 2,
            history_length: 3,
            num_actions: 5,
            replay_buffer_size: 50,
            batch_size: 4,
            ..Default::default()
        };
        let mut agent = DQNAgent::new(config);

        for i in 0..20 {
            let outcome = make_outcome(vec![5.0, 5.0], vec![4.0, 4.0]);
            agent.observe(&outcome);
            agent.act(i);
        }
        assert!(agent.total_steps > 0);
    }

    #[test]
    fn test_dqn_reset() {
        let config = DQNConfig::default();
        let mut agent = DQNAgent::new(config);
        let outcome = make_outcome(vec![5.0, 5.0], vec![4.0, 4.0]);
        agent.observe(&outcome);
        agent.reset();
        assert!(agent.price_history.is_empty());
        assert!(agent.current_state.is_none());
    }

    #[test]
    fn test_dqn_state_serialization() {
        let config = DQNConfig {
            num_players: 2,
            num_actions: 5,
            hidden_sizes: vec![8, 4],
            ..Default::default()
        };
        let agent = DQNAgent::new(config.clone());
        let state = agent.get_state();

        let mut agent2 = DQNAgent::new(config);
        agent2.set_state(&state).unwrap();
    }

    #[test]
    fn test_dqn_wrong_state_variant() {
        let config = DQNConfig::default();
        let mut agent = DQNAgent::new(config);
        let state = AlgorithmState::Empty;
        assert!(agent.set_state(&state).is_err());
    }

    #[test]
    fn test_action_to_price_mapping() {
        let config = DQNConfig {
            price_min: 0.0,
            price_max: 10.0,
            num_actions: 11,
            ..Default::default()
        };
        let agent = DQNAgent::new(config);
        assert!((agent.action_to_price(0) - 0.0).abs() < 1e-10);
        assert!((agent.action_to_price(5) - 5.0).abs() < 1e-10);
        assert!((agent.action_to_price(10) - 10.0).abs() < 1e-10);
    }

    // ── Prioritized replay tests ────────────────────────────────────────

    #[test]
    fn test_replay_buffer_push_and_sample() {
        let mut buf = PrioritizedReplayBuffer::new(10);
        for i in 0..5 {
            buf.push(
                DQNExperience {
                    state: vec![i as f64],
                    action: 0,
                    reward: 1.0,
                    next_state: vec![(i + 1) as f64],
                    done: false,
                },
                1.0,
            );
        }
        assert_eq!(buf.len(), 5);
        let mut rng = StdRng::seed_from_u64(42);
        let batch = buf.sample(3, &mut rng);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_replay_buffer_overflow() {
        let mut buf = PrioritizedReplayBuffer::new(3);
        for i in 0..5 {
            buf.push(
                DQNExperience {
                    state: vec![i as f64],
                    action: 0,
                    reward: 1.0,
                    next_state: vec![(i + 1) as f64],
                    done: false,
                },
                1.0,
            );
        }
        assert_eq!(buf.len(), 3);
    }
}
