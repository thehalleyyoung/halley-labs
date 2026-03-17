//! Black-box algorithm interface for the CollusionProof system.
//!
//! Provides the core `PricingAlgorithm` trait, sandboxing, oracle layers,
//! and batch execution infrastructure.

use serde::{Deserialize, Serialize};
use shared_types::{
    AlgorithmConfig, AlgorithmType, CollusionError, CollusionResult, MarketOutcome, MarketType,
    OracleAccessLevel, PlayerAction, PlayerId, Price, PriceTrajectory, RoundNumber,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// PricingAlgorithm trait
// ═══════════════════════════════════════════════════════════════════════════

/// Core trait that every pricing algorithm must implement.
///
/// The interface is deliberately minimal to support black-box analysis:
/// algorithms receive market outcomes and produce pricing actions.
pub trait PricingAlgorithm: Send + Sync {
    /// Observe a market outcome (prices, quantities, profits from last round).
    fn observe(&mut self, outcome: &MarketOutcome);

    /// Choose an action (price) given the current internal state.
    fn act(&mut self, round: RoundNumber) -> PlayerAction;

    /// Reset internal state for a fresh episode.
    fn reset(&mut self);

    /// Snapshot the current internal state for serialization.
    fn get_state(&self) -> AlgorithmState;

    /// Restore internal state from a snapshot.
    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()>;

    /// Unique name identifying this algorithm instance.
    fn name(&self) -> &str;

    /// The player id this algorithm controls.
    fn player_id(&self) -> PlayerId;
}

// ═══════════════════════════════════════════════════════════════════════════
// AlgorithmState
// ═══════════════════════════════════════════════════════════════════════════

/// Serializable state snapshot for any algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmState {
    /// Q-learning state: Q-table entries, episode count, epsilon.
    QLearning {
        q_table: HashMap<(u64, u64), f64>,
        episode: usize,
        epsilon: f64,
        total_steps: usize,
    },
    /// DQN state: serialized weight matrices.
    DQN {
        weights: Vec<Vec<Vec<f64>>>,
        biases: Vec<Vec<f64>>,
        episode: usize,
        epsilon: f64,
    },
    /// Grim trigger state: triggered flag and round of trigger.
    GrimTrigger {
        triggered: bool,
        trigger_round: Option<RoundNumber>,
        punishment_count: usize,
    },
    /// Tit-for-tat state: memory of recent rounds.
    TitForTat {
        memory: Vec<Vec<Price>>,
        defection_count: usize,
    },
    /// Bandit state: arm statistics.
    Bandit {
        arm_counts: Vec<usize>,
        arm_rewards: Vec<f64>,
        total_pulls: usize,
    },
    /// Generic JSON state for custom algorithms.
    Custom {
        algorithm_name: String,
        data: serde_json::Value,
    },
    /// Empty / initial state.
    Empty,
}

impl AlgorithmState {
    pub fn is_empty(&self) -> bool {
        matches!(self, AlgorithmState::Empty)
    }

    pub fn to_json(&self) -> CollusionResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| CollusionError::Serialization(e.to_string()))
    }

    pub fn from_json(s: &str) -> CollusionResult<Self> {
        serde_json::from_str(s).map_err(|e| CollusionError::Serialization(e.to_string()))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AlgorithmSandbox
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for algorithm sandboxing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub max_step_time: Duration,
    pub max_memory_bytes: usize,
    pub max_total_steps: usize,
    pub enforce_determinism: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_step_time: Duration::from_secs(1),
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            max_total_steps: 1_000_000,
            enforce_determinism: false,
        }
    }
}

/// Wraps an algorithm with resource limits (time, memory, step count).
pub struct AlgorithmSandbox {
    algorithm: Box<dyn PricingAlgorithm>,
    config: SandboxConfig,
    metrics: AlgorithmMetrics,
    step_count: usize,
}

impl AlgorithmSandbox {
    pub fn new(algorithm: Box<dyn PricingAlgorithm>, config: SandboxConfig) -> Self {
        Self {
            algorithm,
            config,
            metrics: AlgorithmMetrics::new(),
            step_count: 0,
        }
    }

    pub fn with_defaults(algorithm: Box<dyn PricingAlgorithm>) -> Self {
        Self::new(algorithm, SandboxConfig::default())
    }

    /// Execute a single observe+act step with resource enforcement.
    pub fn execute_step(
        &mut self,
        outcome: &MarketOutcome,
        round: RoundNumber,
    ) -> CollusionResult<SandboxedExecution> {
        if self.step_count >= self.config.max_total_steps {
            return Err(CollusionError::ResourceLimit(format!(
                "Exceeded max total steps: {}",
                self.config.max_total_steps
            )));
        }

        let start = Instant::now();

        // Observe phase
        self.algorithm.observe(outcome);
        let observe_time = start.elapsed();

        if observe_time > self.config.max_step_time {
            return Err(CollusionError::Timeout(format!(
                "Observe phase exceeded time limit: {:?} > {:?}",
                observe_time, self.config.max_step_time
            )));
        }

        // Act phase
        let act_start = Instant::now();
        let action = self.algorithm.act(round);
        let act_time = act_start.elapsed();

        let total_time = start.elapsed();
        if total_time > self.config.max_step_time {
            return Err(CollusionError::Timeout(format!(
                "Total step time exceeded limit: {:?} > {:?}",
                total_time, self.config.max_step_time
            )));
        }

        self.step_count += 1;

        // Estimate memory (rough heuristic based on state serialization size)
        let mem_estimate = self.estimate_memory();
        if mem_estimate > self.config.max_memory_bytes {
            return Err(CollusionError::ResourceLimit(format!(
                "Estimated memory {} > limit {}",
                mem_estimate, self.config.max_memory_bytes
            )));
        }

        let exec = SandboxedExecution {
            action,
            observe_time,
            act_time,
            total_time,
            memory_estimate: mem_estimate,
            step_number: self.step_count,
        };

        self.metrics.record_step(&exec);
        Ok(exec)
    }

    /// Reset the sandboxed algorithm.
    pub fn reset(&mut self) {
        self.algorithm.reset();
        self.step_count = 0;
        self.metrics = AlgorithmMetrics::new();
    }

    /// Get the algorithm name.
    pub fn name(&self) -> &str {
        self.algorithm.name()
    }

    /// Get algorithm player id.
    pub fn player_id(&self) -> PlayerId {
        self.algorithm.player_id()
    }

    /// Get collected metrics.
    pub fn metrics(&self) -> &AlgorithmMetrics {
        &self.metrics
    }

    /// Get a state snapshot of the inner algorithm.
    pub fn get_state(&self) -> AlgorithmState {
        self.algorithm.get_state()
    }

    /// Restore inner algorithm state.
    pub fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        self.algorithm.set_state(state)
    }

    fn estimate_memory(&self) -> usize {
        let state = self.algorithm.get_state();
        match &state {
            AlgorithmState::QLearning { q_table, .. } => {
                q_table.len() * (std::mem::size_of::<(u64, u64)>() + 8) + 256
            }
            AlgorithmState::DQN { weights, biases, .. } => {
                let w_size: usize = weights.iter()
                    .map(|layer| layer.iter().map(|row| row.len() * 8).sum::<usize>())
                    .sum();
                let b_size: usize = biases.iter().map(|b| b.len() * 8).sum();
                w_size + b_size + 256
            }
            AlgorithmState::Bandit { arm_counts, .. } => arm_counts.len() * 24 + 256,
            _ => 1024,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SandboxedExecution
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a single sandboxed algorithm step.
#[derive(Debug, Clone)]
pub struct SandboxedExecution {
    pub action: PlayerAction,
    pub observe_time: Duration,
    pub act_time: Duration,
    pub total_time: Duration,
    pub memory_estimate: usize,
    pub step_number: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// AlgorithmMetrics
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks per-step performance metrics for an algorithm.
#[derive(Debug, Clone)]
pub struct AlgorithmMetrics {
    pub step_times: Vec<Duration>,
    pub memory_usage: Vec<usize>,
    pub total_steps: usize,
    pub max_step_time: Duration,
    pub min_step_time: Duration,
    pub total_time: Duration,
    pub peak_memory: usize,
}

impl AlgorithmMetrics {
    pub fn new() -> Self {
        Self {
            step_times: Vec::new(),
            memory_usage: Vec::new(),
            total_steps: 0,
            max_step_time: Duration::ZERO,
            min_step_time: Duration::MAX,
            total_time: Duration::ZERO,
            peak_memory: 0,
        }
    }

    pub fn record_step(&mut self, exec: &SandboxedExecution) {
        self.step_times.push(exec.total_time);
        self.memory_usage.push(exec.memory_estimate);
        self.total_steps += 1;
        self.total_time += exec.total_time;
        if exec.total_time > self.max_step_time {
            self.max_step_time = exec.total_time;
        }
        if exec.total_time < self.min_step_time {
            self.min_step_time = exec.total_time;
        }
        if exec.memory_estimate > self.peak_memory {
            self.peak_memory = exec.memory_estimate;
        }
    }

    pub fn mean_step_time(&self) -> Duration {
        if self.total_steps == 0 {
            return Duration::ZERO;
        }
        self.total_time / self.total_steps as u32
    }

    pub fn mean_memory(&self) -> f64 {
        if self.memory_usage.is_empty() {
            return 0.0;
        }
        self.memory_usage.iter().sum::<usize>() as f64 / self.memory_usage.len() as f64
    }

    pub fn step_time_variance(&self) -> f64 {
        if self.step_times.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_step_time().as_secs_f64();
        let var: f64 = self.step_times.iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.step_times.len() - 1) as f64;
        var
    }
}

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AlgorithmFactory
// ═══════════════════════════════════════════════════════════════════════════

/// Factory for creating algorithm instances by type and config.
pub struct AlgorithmFactory;

impl AlgorithmFactory {
    /// Create an algorithm from a config.
    pub fn create(
        config: &AlgorithmConfig,
        player_id: PlayerId,
        num_actions: usize,
        price_min: Price,
        price_max: Price,
    ) -> CollusionResult<Box<dyn PricingAlgorithm>> {
        match &config.algorithm_type {
            AlgorithmType::QLearning => {
                let ql_config = crate::q_learning::QLearningConfig {
                    player_id,
                    num_price_levels: num_actions,
                    price_min,
                    price_max,
                    learning_rate: config.get_param("learning_rate").unwrap_or(0.1),
                    discount_factor: config.get_param("discount_factor").unwrap_or(0.95),
                    epsilon_start: config.get_param("epsilon_start").unwrap_or(1.0),
                    epsilon_end: config.get_param("epsilon_end").unwrap_or(0.01),
                    epsilon_decay: config.get_param("epsilon_decay").unwrap_or(0.995),
                    decay_schedule: crate::q_learning::DecaySchedule::Exponential,
                    num_state_bins: config.get_param("num_state_bins").unwrap_or(15.0) as usize,
                    replay_buffer_size: config.get_param("replay_buffer_size").unwrap_or(10000.0) as usize,
                    use_replay: config.get_param("use_replay").unwrap_or(0.0) > 0.5,
                    batch_size: config.get_param("batch_size").unwrap_or(32.0) as usize,
                };
                Ok(Box::new(crate::q_learning::QLearningAgent::new(ql_config)))
            }
            AlgorithmType::DQN => {
                let dqn_config = crate::dqn::DQNConfig {
                    player_id,
                    num_actions,
                    price_min,
                    price_max,
                    history_length: config.get_param("history_length").unwrap_or(10.0) as usize,
                    hidden_sizes: vec![
                        config.get_param("hidden_size_1").unwrap_or(64.0) as usize,
                        config.get_param("hidden_size_2").unwrap_or(32.0) as usize,
                    ],
                    learning_rate: config.get_param("learning_rate").unwrap_or(0.001),
                    discount_factor: config.get_param("discount_factor").unwrap_or(0.95),
                    epsilon_start: config.get_param("epsilon_start").unwrap_or(1.0),
                    epsilon_end: config.get_param("epsilon_end").unwrap_or(0.01),
                    epsilon_decay: config.get_param("epsilon_decay").unwrap_or(0.995),
                    target_update_freq: config.get_param("target_update_freq").unwrap_or(100.0) as usize,
                    replay_buffer_size: config.get_param("replay_buffer_size").unwrap_or(10000.0) as usize,
                    batch_size: config.get_param("batch_size").unwrap_or(32.0) as usize,
                    num_players: config.get_param("num_players").unwrap_or(2.0) as usize,
                };
                Ok(Box::new(crate::dqn::DQNAgent::new(dqn_config)))
            }
            AlgorithmType::GrimTrigger => {
                let gt_config = crate::grim_trigger::GrimTriggerConfig {
                    player_id,
                    cooperative_price: Price(config.get_param("cooperative_price").unwrap_or(5.0)),
                    punishment_price: Price(config.get_param("punishment_price").unwrap_or(1.0)),
                    cooperation_threshold: config.get_param("cooperation_threshold").unwrap_or(0.1),
                    num_players: config.get_param("num_players").unwrap_or(2.0) as usize,
                };
                Ok(Box::new(crate::grim_trigger::GrimTriggerAgent::new(gt_config)))
            }
            AlgorithmType::TitForTat => {
                let num_players = config.get_param("num_players").unwrap_or(2.0) as usize;
                let base_price = Price(config.get_param("base_price").unwrap_or(5.0));
                let punishment_price = Price(config.get_param("punishment_price").unwrap_or(1.0));
                Ok(Box::new(crate::tit_for_tat::TitForTatAgent::new(
                    player_id,
                    num_players,
                    base_price,
                    punishment_price,
                )))
            }
            AlgorithmType::Bandit => {
                let epsilon = config.get_param("epsilon").unwrap_or(0.1);
                Ok(Box::new(crate::bandit::EpsilonGreedyBandit::new(
                    player_id,
                    num_actions,
                    price_min,
                    price_max,
                    epsilon,
                )))
            }
            AlgorithmType::NashEquilibrium | AlgorithmType::MyopicBestResponse => {
                // Default to Q-learning with very low exploration
                let ql_config = crate::q_learning::QLearningConfig {
                    player_id,
                    num_price_levels: num_actions,
                    price_min,
                    price_max,
                    learning_rate: 0.5,
                    discount_factor: 0.0,
                    epsilon_start: 0.0,
                    epsilon_end: 0.0,
                    epsilon_decay: 1.0,
                    decay_schedule: crate::q_learning::DecaySchedule::Linear,
                    num_state_bins: 1,
                    replay_buffer_size: 0,
                    use_replay: false,
                    batch_size: 1,
                };
                Ok(Box::new(crate::q_learning::QLearningAgent::new(ql_config)))
            }
            AlgorithmType::Custom(name) => {
                Err(CollusionError::NotFound(format!("Custom algorithm '{}' not registered", name)))
            }
            other => {
                Err(CollusionError::NotFound(format!("Algorithm type '{:?}' not yet implemented", other)))
            }
        }
    }

    /// Create a pair of algorithms for a two-player scenario.
    pub fn create_pair(
        config_a: &AlgorithmConfig,
        config_b: &AlgorithmConfig,
        num_actions: usize,
        price_min: Price,
        price_max: Price,
    ) -> CollusionResult<(Box<dyn PricingAlgorithm>, Box<dyn PricingAlgorithm>)> {
        let a = Self::create(config_a, PlayerId(0), num_actions, price_min, price_max)?;
        let b = Self::create(config_b, PlayerId(1), num_actions, price_min, price_max)?;
        Ok((a, b))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BatchOracle
// ═══════════════════════════════════════════════════════════════════════════

/// Buffers N steps for batch evaluation (useful for PyO3 efficiency).
pub struct BatchOracle {
    buffer: Vec<(MarketOutcome, RoundNumber)>,
    capacity: usize,
    results: Vec<PlayerAction>,
}

impl BatchOracle {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            results: Vec::new(),
        }
    }

    /// Add a step to the buffer.
    pub fn push(&mut self, outcome: MarketOutcome, round: RoundNumber) -> bool {
        self.buffer.push((outcome, round));
        self.buffer.len() >= self.capacity
    }

    /// Check if the buffer is full and ready for batch execution.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Execute all buffered steps against an algorithm.
    pub fn flush(&mut self, algorithm: &mut dyn PricingAlgorithm) -> Vec<PlayerAction> {
        self.results.clear();
        for (outcome, round) in self.buffer.drain(..) {
            algorithm.observe(&outcome);
            self.results.push(algorithm.act(round));
        }
        self.results.clone()
    }

    /// Number of buffered items.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer without executing.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.results.clear();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OracleInterface trait and Layer implementations
// ═══════════════════════════════════════════════════════════════════════════

/// Oracle interface for Layer 0/1/2 access patterns.
pub trait OracleInterface: Send + Sync {
    /// Access level this oracle provides.
    fn access_level(&self) -> OracleAccessLevel;

    /// Feed a trajectory for observation.
    fn observe_trajectory(&mut self, trajectory: &PriceTrajectory);

    /// Get the current observed trajectory.
    fn get_trajectory(&self) -> &PriceTrajectory;

    /// Check if a deviation was performed at the given round.
    fn deviation_at(&self, round: RoundNumber) -> Option<&DeviationRecord>;

    /// Get all state checkpoints.
    fn checkpoints(&self) -> &[StateCheckpoint];

    /// Rewind to a specific round (Layer 2 only).
    fn rewind_to(&mut self, round: RoundNumber) -> CollusionResult<()>;

    /// Insert a deviation at a specific round (Layer 1+ only).
    fn insert_deviation(
        &mut self,
        round: RoundNumber,
        player: PlayerId,
        deviation_price: Price,
    ) -> CollusionResult<()>;
}

/// Record of a deliberate price deviation for testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationRecord {
    pub round: RoundNumber,
    pub player: PlayerId,
    pub original_price: Price,
    pub deviation_price: Price,
    pub response_observed: bool,
    pub punishment_detected: bool,
}

/// Snapshot of algorithm state at a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCheckpoint {
    pub round: RoundNumber,
    pub states: Vec<AlgorithmState>,
    pub market_outcome: Option<MarketOutcome>,
}

// ═══════════════════════════════════════════════════════════════════════════
// PassiveOracle (Layer 0)
// ═══════════════════════════════════════════════════════════════════════════

/// Layer 0: Observe-only oracle with no intervention capability.
pub struct PassiveOracle {
    trajectory: PriceTrajectory,
}

impl PassiveOracle {
    pub fn new(num_players: usize) -> Self {
        Self {
            trajectory: PriceTrajectory::new(Vec::new(), MarketType::Bertrand, num_players, AlgorithmType::QLearning, 0),
        }
    }
    fn access_level(&self) -> OracleAccessLevel {
        OracleAccessLevel::Layer0
    }

    fn observe_trajectory(&mut self, trajectory: &PriceTrajectory) {
        self.trajectory = trajectory.clone();
    }

    fn get_trajectory(&self) -> &PriceTrajectory {
        &self.trajectory
    }

    fn deviation_at(&self, _round: RoundNumber) -> Option<&DeviationRecord> {
        None // Layer 0 cannot perform deviations
    }

    fn checkpoints(&self) -> &[StateCheckpoint] {
        &[] // Layer 0 has no checkpoints
    }

    fn rewind_to(&mut self, _round: RoundNumber) -> CollusionResult<()> {
        Err(CollusionError::InvalidState(
            "PassiveOracle (Layer 0) does not support rewind".into(),
        ))
    }

    fn insert_deviation(
        &mut self,
        _round: RoundNumber,
        _player: PlayerId,
        _deviation_price: Price,
    ) -> CollusionResult<()> {
        Err(CollusionError::InvalidState(
            "PassiveOracle (Layer 0) does not support deviation insertion".into(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CheckpointOracle (Layer 1)
// ═══════════════════════════════════════════════════════════════════════════

/// Layer 1: Periodic state snapshots + deviation insertion.
pub struct CheckpointOracle {
    trajectory: PriceTrajectory,
    checkpoints_list: Vec<StateCheckpoint>,
    deviations: Vec<DeviationRecord>,
    checkpoint_interval: usize,
}

impl CheckpointOracle {
    pub fn new(num_players: usize, checkpoint_interval: usize) -> Self {
        Self {
            trajectory: PriceTrajectory::new(Vec::new(), MarketType::Bertrand, num_players, AlgorithmType::QLearning, 0),
            checkpoints_list: Vec::new(),
            deviations: Vec::new(),
            checkpoint_interval,
        }
    }

    /// Add a checkpoint at the current state.
    pub fn add_checkpoint(&mut self, checkpoint: StateCheckpoint) {
        self.checkpoints_list.push(checkpoint);
    }

    /// Check if a checkpoint should be taken at this round.
    pub fn should_checkpoint(&self, round: RoundNumber) -> bool {
        self.checkpoint_interval > 0 && round % self.checkpoint_interval == 0
    }

    /// Get the nearest checkpoint before or at the given round.
    pub fn nearest_checkpoint(&self, round: RoundNumber) -> Option<&StateCheckpoint> {
        self.checkpoints_list
            .iter()
            .rev()
            .find(|cp| cp.round <= round)
    }

    /// Get all deviation records.
    pub fn deviations(&self) -> &[DeviationRecord] {
        &self.deviations
    }
}

impl OracleInterface for CheckpointOracle {
    fn access_level(&self) -> OracleAccessLevel {
        OracleAccessLevel::Layer1
    }

    fn observe_trajectory(&mut self, trajectory: &PriceTrajectory) {
        self.trajectory = trajectory.clone();
    }

    fn get_trajectory(&self) -> &PriceTrajectory {
        &self.trajectory
    }

    fn deviation_at(&self, round: RoundNumber) -> Option<&DeviationRecord> {
        self.deviations.iter().find(|d| d.round == round)
    }

    fn checkpoints(&self) -> &[StateCheckpoint] {
        &self.checkpoints_list
    }

    fn rewind_to(&mut self, _round: RoundNumber) -> CollusionResult<()> {
        Err(CollusionError::InvalidState(
            "CheckpointOracle (Layer 1) does not support arbitrary rewind".into(),
        ))
    }

    fn insert_deviation(
        &mut self,
        round: RoundNumber,
        player: PlayerId,
        deviation_price: Price,
    ) -> CollusionResult<()> {
        let original_price = self
            .trajectory
            .outcomes
            .get(round.0)
            .and_then(|o| o.prices.get(player.0))
            .copied()
            .unwrap_or(Price::ZERO);

        self.deviations.push(DeviationRecord {
            round,
            player,
            original_price,
            deviation_price,
            response_observed: false,
            punishment_detected: false,
        });
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RewindOracle (Layer 2)
// ═══════════════════════════════════════════════════════════════════════════

/// Layer 2: Full state restore from any prefix.
pub struct RewindOracle {
    trajectory: PriceTrajectory,
    checkpoints_list: Vec<StateCheckpoint>,
    deviations: Vec<DeviationRecord>,
    full_history: Vec<Vec<AlgorithmState>>,
    current_round: RoundNumber,
}

impl RewindOracle {
    pub fn new(num_players: usize) -> Self {
        Self {
            trajectory: PriceTrajectory::new(Vec::new(), MarketType::Bertrand, num_players, AlgorithmType::QLearning, 0),
            checkpoints_list: Vec::new(),
            deviations: Vec::new(),
            full_history: Vec::new(),
            current_round: RoundNumber(0),
        }
    }

    /// Record algorithm states for a round.
    pub fn record_states(&mut self, round: RoundNumber, states: Vec<AlgorithmState>) {
        while self.full_history.len() <= round.0 {
            self.full_history.push(Vec::new());
        }
        self.full_history[round.0] = states.clone();

        self.checkpoints_list.push(StateCheckpoint {
            round,
            states,
            market_outcome: self.trajectory.outcomes.get(round.0).cloned(),
        });
    }

    /// Get the state history at a specific round.
    pub fn states_at(&self, round: RoundNumber) -> Option<&Vec<AlgorithmState>> {
        self.full_history.get(round.0)
    }

    /// Get all deviation records.
    pub fn deviations(&self) -> &[DeviationRecord] {
        &self.deviations
    }

    pub fn current_round(&self) -> RoundNumber {
        self.current_round
    }
}

impl OracleInterface for RewindOracle {
    fn access_level(&self) -> OracleAccessLevel {
        OracleAccessLevel::Layer2
    }

    fn observe_trajectory(&mut self, trajectory: &PriceTrajectory) {
        self.trajectory = trajectory.clone();
    }

    fn get_trajectory(&self) -> &PriceTrajectory {
        &self.trajectory
    }

    fn deviation_at(&self, round: RoundNumber) -> Option<&DeviationRecord> {
        self.deviations.iter().find(|d| d.round == round)
    }

    fn checkpoints(&self) -> &[StateCheckpoint] {
        &self.checkpoints_list
    }

    fn rewind_to(&mut self, round: RoundNumber) -> CollusionResult<()> {
        if round.0 >= self.full_history.len() {
            return Err(CollusionError::InvalidState(format!(
                "Cannot rewind to round {} (history only has {} entries)",
                round,
                self.full_history.len()
            )));
        }
        self.current_round = round;
        // Truncate trajectory to the rewind point
        self.trajectory.outcomes.truncate(round.0 + 1);
        Ok(())
    }

    fn insert_deviation(
        &mut self,
        round: RoundNumber,
        player: PlayerId,
        deviation_price: Price,
    ) -> CollusionResult<()> {
        let original_price = self
            .trajectory
            .outcomes
            .get(round.0)
            .and_then(|o| o.prices.get(player.0))
            .copied()
            .unwrap_or(Price::ZERO);

        self.deviations.push(DeviationRecord {
            round,
            player,
            original_price,
            deviation_price,
            response_observed: false,
            punishment_detected: false,
        });
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{Profit, Quantity};

    fn make_outcome(round: usize, prices: Vec<f64>) -> MarketOutcome {
        let n = prices.len();
        let actions: Vec<PlayerAction> = prices
            .iter()
            .enumerate()
            .map(|(i, &p)| PlayerAction::new(PlayerId(i), Price(p)))
            .collect();
        let price_vec: Vec<Price> = prices.iter().map(|&p| Price(p)).collect();
        let quantities: Vec<Quantity> = vec![Quantity(1.0); n];
        let profits: Vec<Profit> = prices.iter().map(|&p| Profit(p - 1.0)).collect();
        MarketOutcome::new(
            RoundNumber(round),
            actions,
            price_vec,
            quantities,
            profits,
        )
    }

    fn make_trajectory(rounds: usize, price: f64) -> PriceTrajectory {
        let outcomes: Vec<MarketOutcome> = (0..rounds)
            .map(|r| make_outcome(r, vec![price, price]))
            .collect();
        PriceTrajectory::new(outcomes, MarketType::Bertrand, 2, AlgorithmType::QLearning, 0)
    }

    #[test]
    fn test_algorithm_state_serialization() {
        let state = AlgorithmState::QLearning {
            q_table: HashMap::from([((0, 1), 0.5), ((1, 2), 0.3)]),
            episode: 10,
            epsilon: 0.1,
            total_steps: 100,
        };
        let json = state.to_json().unwrap();
        let restored = AlgorithmState::from_json(&json).unwrap();
        match restored {
            AlgorithmState::QLearning { episode, .. } => assert_eq!(episode, 10),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_algorithm_state_empty() {
        let state = AlgorithmState::Empty;
        assert!(state.is_empty());
    }

    #[test]
    fn test_sandbox_config_default() {
        let config = SandboxConfig::default();
        assert_eq!(config.max_step_time, Duration::from_secs(1));
        assert_eq!(config.max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(config.max_total_steps, 1_000_000);
    }

    #[test]
    fn test_algorithm_metrics_new() {
        let metrics = AlgorithmMetrics::new();
        assert_eq!(metrics.total_steps, 0);
        assert_eq!(metrics.mean_step_time(), Duration::ZERO);
        assert_eq!(metrics.mean_memory(), 0.0);
    }

    #[test]
    fn test_algorithm_metrics_record() {
        let mut metrics = AlgorithmMetrics::new();
        let exec = SandboxedExecution {
            action: PlayerAction::new(PlayerId(0), Price(5.0)),
            observe_time: Duration::from_millis(1),
            act_time: Duration::from_millis(2),
            total_time: Duration::from_millis(3),
            memory_estimate: 1024,
            step_number: 1,
        };
        metrics.record_step(&exec);
        assert_eq!(metrics.total_steps, 1);
        assert_eq!(metrics.peak_memory, 1024);
        assert_eq!(metrics.max_step_time, Duration::from_millis(3));
    }

    #[test]
    fn test_batch_oracle() {
        let mut oracle = BatchOracle::new(3);
        assert!(oracle.is_empty());
        assert!(!oracle.is_full());

        oracle.push(make_outcome(0, vec![5.0, 5.0]), RoundNumber(0));
        oracle.push(make_outcome(1, vec![5.0, 5.0]), RoundNumber(1));
        assert_eq!(oracle.len(), 2);

        let full = oracle.push(make_outcome(2, vec![5.0, 5.0]), RoundNumber(2));
        assert!(full);
        assert!(oracle.is_full());
    }

    #[test]
    fn test_passive_oracle() {
        let mut oracle = PassiveOracle::new(2);
        assert_eq!(oracle.access_level(), OracleAccessLevel::Layer0);

        let traj = make_trajectory(10, 5.0);
        oracle.observe_trajectory(&traj);
        assert_eq!(oracle.get_trajectory().len(), 10);

        assert!(oracle.deviation_at(RoundNumber(0)).is_none());
        assert!(oracle.checkpoints().is_empty());
        assert!(oracle.rewind_to(RoundNumber(0)).is_err());
        assert!(oracle.insert_deviation(RoundNumber(0), PlayerId(0), Price(1.0)).is_err());
    }

    #[test]
    fn test_checkpoint_oracle() {
        let mut oracle = CheckpointOracle::new(2, 5);
        assert_eq!(oracle.access_level(), OracleAccessLevel::Layer1);

        let traj = make_trajectory(10, 5.0);
        oracle.observe_trajectory(&traj);

        assert!(oracle.should_checkpoint(RoundNumber(0)));
        assert!(!oracle.should_checkpoint(RoundNumber(3)));
        assert!(oracle.should_checkpoint(RoundNumber(5)));

        oracle.add_checkpoint(StateCheckpoint {
            round: RoundNumber(5),
            states: vec![AlgorithmState::Empty],
            market_outcome: None,
        });
        assert_eq!(oracle.checkpoints().len(), 1);
        assert_eq!(oracle.nearest_checkpoint(RoundNumber(7)).unwrap().round, RoundNumber(5));
    }

    #[test]
    fn test_checkpoint_oracle_deviation() {
        let mut oracle = CheckpointOracle::new(2, 5);
        let traj = make_trajectory(10, 5.0);
        oracle.observe_trajectory(&traj);

        oracle.insert_deviation(RoundNumber(3), PlayerId(0), Price(1.0)).unwrap();
        let dev = oracle.deviation_at(RoundNumber(3)).unwrap();
        assert_eq!(dev.player, 0);
        assert_eq!(dev.deviation_price, Price(1.0));
        assert_eq!(dev.original_price, Price(5.0));
    }

    #[test]
    fn test_rewind_oracle() {
        let mut oracle = RewindOracle::new(2);
        assert_eq!(oracle.access_level(), OracleAccessLevel::Layer2);

        let traj = make_trajectory(10, 5.0);
        oracle.observe_trajectory(&traj);

        oracle.record_states(RoundNumber(0), vec![AlgorithmState::Empty, AlgorithmState::Empty]);
        oracle.record_states(RoundNumber(5), vec![AlgorithmState::Empty, AlgorithmState::Empty]);

        assert_eq!(oracle.states_at(RoundNumber(0)).unwrap().len(), 2);

        oracle.rewind_to(RoundNumber(5)).unwrap();
        assert_eq!(oracle.current_round(), RoundNumber(5));
        assert_eq!(oracle.get_trajectory().len(), 6);
    }

    #[test]
    fn test_rewind_oracle_deviation() {
        let mut oracle = RewindOracle::new(2);
        let traj = make_trajectory(10, 5.0);
        oracle.observe_trajectory(&traj);

        oracle.insert_deviation(RoundNumber(3), PlayerId(1), Price(2.0)).unwrap();
        let dev = oracle.deviation_at(RoundNumber(3)).unwrap();
        assert_eq!(dev.player, 1);
        assert_eq!(dev.deviation_price, Price(2.0));
    }

    #[test]
    fn test_rewind_oracle_out_of_bounds() {
        let mut oracle = RewindOracle::new(2);
        assert!(oracle.rewind_to(RoundNumber(100)).is_err());
    }

    #[test]
    fn test_deviation_record_fields() {
        let record = DeviationRecord {
            round: RoundNumber(42),
            player: PlayerId(0),
            original_price: Price(5.0),
            deviation_price: Price(1.0),
            response_observed: true,
            punishment_detected: false,
        };
        assert_eq!(record.round, RoundNumber(42));
        assert!(record.response_observed);
        assert!(!record.punishment_detected);
    }

    #[test]
    fn test_state_checkpoint_serialization() {
        let cp = StateCheckpoint {
            round: RoundNumber(10),
            states: vec![AlgorithmState::Empty],
            market_outcome: Some(make_outcome(10, vec![5.0, 5.0])),
        };
        let json = serde_json::to_string(&cp).unwrap();
        let restored: StateCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.round, RoundNumber(10));
    }

    #[test]
    fn test_batch_oracle_clear() {
        let mut oracle = BatchOracle::new(5);
        oracle.push(make_outcome(0, vec![5.0]), RoundNumber(0));
        oracle.push(make_outcome(1, vec![5.0]), RoundNumber(1));
        assert_eq!(oracle.len(), 2);
        oracle.clear();
        assert!(oracle.is_empty());
    }

    #[test]
    fn test_metrics_variance_single_step() {
        let mut metrics = AlgorithmMetrics::new();
        let exec = SandboxedExecution {
            action: PlayerAction::new(PlayerId(0), Price(5.0)),
            observe_time: Duration::from_millis(1),
            act_time: Duration::from_millis(1),
            total_time: Duration::from_millis(2),
            memory_estimate: 512,
            step_number: 1,
        };
        metrics.record_step(&exec);
        assert_eq!(metrics.step_time_variance(), 0.0);
    }

    #[test]
    fn test_algorithm_state_grim_trigger() {
        let state = AlgorithmState::GrimTrigger {
            triggered: true,
            trigger_round: Some(RoundNumber(42)),
            punishment_count: 10,
        };
        let json = state.to_json().unwrap();
        assert!(json.contains("42"));
        assert!(!state.is_empty());
    }
}
