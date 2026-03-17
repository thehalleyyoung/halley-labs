//! Evaluation scenario definitions for the CollusionProof system.
//!
//! Provides a library of 30 scenarios covering known-collusive,
//! known-competitive, boundary, and adversarial cases.

use crate::algorithm::PricingAlgorithm;
use crate::bandit::EpsilonGreedyBandit;
use crate::detector::{DetectionConfig, Verdict};
use crate::dqn::DQNConfig;
use crate::grim_trigger::{GrimTriggerAgent, GrimTriggerConfig};
use crate::q_learning::{DecaySchedule, QLearningConfig};
use crate::tit_for_tat::{GenerousTitForTat, SuspiciousTitForTat, TitForTatAgent, TitForTwoTats};
use serde::{Deserialize, Serialize};
use shared_types::{
    AlgorithmConfig, AlgorithmType, DemandSystem, MarketType, PlayerId, Price, SimulationConfig,
};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// Scenario definition
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    pub expected_verdict: ExpectedVerdict,
    pub algorithm_configs: Vec<AlgorithmConfig>,
    pub simulation_config: SimulationConfig,
    pub detection_config: DetectionConfig,
    pub category: ScenarioCategory,
    pub difficulty: Difficulty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedVerdict {
    Collusive,
    Competitive,
    Inconclusive,
}

impl ExpectedVerdict {
    pub fn matches(&self, verdict: &Verdict) -> bool {
        match (self, verdict) {
            (ExpectedVerdict::Collusive, Verdict::Collusive) => true,
            (ExpectedVerdict::Competitive, Verdict::Competitive) => true,
            (ExpectedVerdict::Inconclusive, Verdict::Inconclusive) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioCategory {
    KnownCollusive,
    KnownCompetitive,
    Boundary,
    Adversarial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario Library
// ═══════════════════════════════════════════════════════════════════════════

pub struct ScenarioLibrary {
    scenarios: Vec<Scenario>,
}

impl ScenarioLibrary {
    pub fn new() -> Self {
        let mut lib = Self { scenarios: Vec::new() };
        lib.populate_collusive();
        lib.populate_competitive();
        lib.populate_boundary();
        lib.populate_adversarial();
        lib
    }

    pub fn all(&self) -> &[Scenario] {
        &self.scenarios
    }

    pub fn by_category(&self, category: ScenarioCategory) -> Vec<&Scenario> {
        self.scenarios.iter().filter(|s| s.category == category).collect()
    }

    pub fn by_name(&self, name: &str) -> Option<&Scenario> {
        self.scenarios.iter().find(|s| s.name == name)
    }

    pub fn by_difficulty(&self, difficulty: Difficulty) -> Vec<&Scenario> {
        self.scenarios.iter().filter(|s| s.difficulty == difficulty).collect()
    }

    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    // ── Known-collusive scenarios (10) ──────────────────────────────────

    fn populate_collusive(&mut self) {
        let base_sim = SimulationConfig::default();
        let base_det = DetectionConfig::default();

        // 1. Grim trigger pair
        self.scenarios.push(Scenario {
            name: "grim_trigger_pair".into(),
            description: "Two grim trigger agents with cooperative price".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("punishment_price", 1.0),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("punishment_price", 1.0),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Easy,
        });

        // 2. TFT pair
        self.scenarios.push(Scenario {
            name: "tft_pair".into(),
            description: "Two tit-for-tat agents".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::TitForTat)
                    .with_param("base_price", 5.0)
                    .with_param("punishment_price", 1.0),
                AlgorithmConfig::new(AlgorithmType::TitForTat)
                    .with_param("base_price", 5.0)
                    .with_param("punishment_price", 1.0),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Easy,
        });

        // 3. Q-learning converging to collusion
        self.scenarios.push(Scenario {
            name: "q_learning_collusion".into(),
            description: "Q-learning agents converging to supracompetitive prices".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("learning_rate", 0.1)
                    .with_param("discount_factor", 0.95)
                    .with_param("epsilon_start", 0.5),
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("learning_rate", 0.1)
                    .with_param("discount_factor", 0.95)
                    .with_param("epsilon_start", 0.5),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Medium,
        });

        // 4. DQN collusion
        self.scenarios.push(Scenario {
            name: "dqn_collusion".into(),
            description: "DQN agents learning collusive pricing".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::DQN)
                    .with_param("learning_rate", 0.001)
                    .with_param("discount_factor", 0.95),
                AlgorithmConfig::new(AlgorithmType::DQN)
                    .with_param("learning_rate", 0.001)
                    .with_param("discount_factor", 0.95),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Medium,
        });

        // 5. Grim trigger asymmetric costs
        self.scenarios.push(Scenario {
            name: "grim_trigger_asymmetric".into(),
            description: "Grim trigger with asymmetric marginal costs".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.5)
                    .with_param("punishment_price", 1.5),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.5)
                    .with_param("punishment_price", 2.0),
            ],
            simulation_config: {
                let mut sim = base_sim.clone();
                sim.game.marginal_costs = vec![shared_types::Cost(1.0), shared_types::Cost(1.5)];
                sim
            },
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Medium,
        });

        // 6. Three-player grim trigger
        self.scenarios.push(Scenario {
            name: "three_player_grim".into(),
            description: "Three grim trigger agents colluding".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 5.0),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 5.0),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 5.0),
            ],
            simulation_config: {
                let mut sim = base_sim.clone();
                sim.game.num_players = 3;
                sim.game.marginal_costs = vec![shared_types::Cost(1.0), shared_types::Cost(1.0), shared_types::Cost(1.0)];
                sim
            },
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Medium,
        });

        // 7. TFT with generous variant
        self.scenarios.push(Scenario {
            name: "generous_tft_pair".into(),
            description: "Two generous TFT agents with 30% forgiveness".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::TitForTat)
                    .with_param("generosity", 0.3)
                    .with_param("base_price", 5.0),
                AlgorithmConfig::new(AlgorithmType::TitForTat)
                    .with_param("generosity", 0.3)
                    .with_param("base_price", 5.0),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Medium,
        });

        // 8. Q-learning with high discount
        self.scenarios.push(Scenario {
            name: "q_learning_high_discount".into(),
            description: "Q-learning with delta=0.99 (strong collusion incentive)".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("discount_factor", 0.99)
                    .with_param("learning_rate", 0.05),
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("discount_factor", 0.99)
                    .with_param("learning_rate", 0.05),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Hard,
        });

        // 9. Mixed: grim trigger + TFT
        self.scenarios.push(Scenario {
            name: "mixed_grim_tft".into(),
            description: "One grim trigger + one TFT".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 5.0),
                AlgorithmConfig::new(AlgorithmType::TitForTat).with_param("base_price", 5.0),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Medium,
        });

        // 10. Logit demand collusion
        self.scenarios.push(Scenario {
            name: "logit_demand_collusion".into(),
            description: "Grim trigger under logit demand".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 4.0),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 4.0),
            ],
            simulation_config: {
                let mut sim = base_sim.clone();
                sim.game.demand_system = DemandSystem::Logit { temperature: 0.5, outside_option_value: 0.0, market_size: 1.0 };
                sim
            },
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCollusive,
            difficulty: Difficulty::Hard,
        });
    }

    // ── Known-competitive scenarios (8) ─────────────────────────────────

    fn populate_competitive(&mut self) {
        let base_sim = SimulationConfig::default();
        let base_det = DetectionConfig::default();

        // 11. Nash equilibrium play
        self.scenarios.push(Scenario {
            name: "nash_equilibrium".into(),
            description: "Both players play Nash equilibrium prices".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::NashEquilibrium),
                AlgorithmConfig::new(AlgorithmType::NashEquilibrium),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Easy,
        });

        // 12. Myopic best response
        self.scenarios.push(Scenario {
            name: "myopic_best_response".into(),
            description: "Each player myopically best-responds each round".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::MyopicBestResponse),
                AlgorithmConfig::new(AlgorithmType::MyopicBestResponse),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Easy,
        });

        // 13. Independent epsilon-greedy bandits
        self.scenarios.push(Scenario {
            name: "independent_bandits".into(),
            description: "Epsilon-greedy bandits learning independently".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::Bandit).with_param("epsilon", 0.1),
                AlgorithmConfig::new(AlgorithmType::Bandit).with_param("epsilon", 0.1),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Easy,
        });

        // 14. Q-learning with zero discount (myopic)
        self.scenarios.push(Scenario {
            name: "q_learning_myopic".into(),
            description: "Q-learning with gamma=0 (purely myopic)".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("discount_factor", 0.0)
                    .with_param("learning_rate", 0.1),
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("discount_factor", 0.0)
                    .with_param("learning_rate", 0.1),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Medium,
        });

        // 15. High exploration Q-learning
        self.scenarios.push(Scenario {
            name: "high_exploration_q".into(),
            description: "Q-learning with very high epsilon (mostly random)".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("epsilon_start", 0.9)
                    .with_param("epsilon_end", 0.9),
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("epsilon_start", 0.9)
                    .with_param("epsilon_end", 0.9),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Easy,
        });

        // 16. UCB1 bandits
        self.scenarios.push(Scenario {
            name: "ucb1_bandits".into(),
            description: "UCB1 bandits competing on price".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::Bandit).with_param("ucb_constant", 2.0),
                AlgorithmConfig::new(AlgorithmType::Bandit).with_param("ucb_constant", 2.0),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Easy,
        });

        // 17. Three competitive players
        self.scenarios.push(Scenario {
            name: "three_competitive".into(),
            description: "Three players at Nash equilibrium".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::NashEquilibrium),
                AlgorithmConfig::new(AlgorithmType::NashEquilibrium),
                AlgorithmConfig::new(AlgorithmType::NashEquilibrium),
            ],
            simulation_config: {
                let mut sim = base_sim.clone();
                sim.game.num_players = 3;
                sim.game.marginal_costs = vec![shared_types::Cost(1.0), shared_types::Cost(1.0), shared_types::Cost(1.0)];
                sim
            },
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Easy,
        });

        // 18. Suspicious TFT vs competitive
        self.scenarios.push(Scenario {
            name: "suspicious_tft_vs_nash".into(),
            description: "Suspicious TFT vs Nash player — settles to competitive".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::TitForTat).with_param("suspicious", 1.0),
                AlgorithmConfig::new(AlgorithmType::NashEquilibrium),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::KnownCompetitive,
            difficulty: Difficulty::Medium,
        });
    }

    // ── Boundary scenarios (8) ──────────────────────────────────────────

    fn populate_boundary(&mut self) {
        let base_sim = SimulationConfig::default();
        let base_det = DetectionConfig::default();

        // 19. Partial collusion (CP ~ 0.3)
        self.scenarios.push(Scenario {
            name: "partial_collusion".into(),
            description: "Prices between Nash and monopoly (~30% CP)".into(),
            expected_verdict: ExpectedVerdict::Inconclusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("discount_factor", 0.7),
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("discount_factor", 0.7),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Hard,
        });

        // 20. Asymmetric algorithms
        self.scenarios.push(Scenario {
            name: "asymmetric_algorithms".into(),
            description: "One Q-learner vs one bandit".into(),
            expected_verdict: ExpectedVerdict::Inconclusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning).with_param("discount_factor", 0.95),
                AlgorithmConfig::new(AlgorithmType::Bandit).with_param("epsilon", 0.1),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Hard,
        });

        // 21. Mixed strategies
        self.scenarios.push(Scenario {
            name: "mixed_strategies".into(),
            description: "Agents mixing cooperative and competitive play".into(),
            expected_verdict: ExpectedVerdict::Inconclusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("epsilon_end", 0.3),
                AlgorithmConfig::new(AlgorithmType::QLearning)
                    .with_param("epsilon_end", 0.3),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Hard,
        });

        // 22. Edgeworth price cycles
        self.scenarios.push(Scenario {
            name: "edgeworth_cycles".into(),
            description: "Cyclical pricing patterns (competitive but unusual)".into(),
            expected_verdict: ExpectedVerdict::Inconclusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::Custom("edgeworth_cycle".into())),
                AlgorithmConfig::new(AlgorithmType::Custom("edgeworth_cycle".into())),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Expert,
        });

        // 23. Transient collusion (collude then break down)
        self.scenarios.push(Scenario {
            name: "transient_collusion".into(),
            description: "Collusion for first half, then competitive breakdown".into(),
            expected_verdict: ExpectedVerdict::Inconclusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 5.0),
                AlgorithmConfig::new(AlgorithmType::QLearning).with_param("epsilon_end", 0.5),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Hard,
        });

        // 24. Near-Nash pricing
        self.scenarios.push(Scenario {
            name: "near_nash".into(),
            description: "Prices slightly above Nash (CP ~ 0.1)".into(),
            expected_verdict: ExpectedVerdict::Competitive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::QLearning).with_param("discount_factor", 0.5),
                AlgorithmConfig::new(AlgorithmType::QLearning).with_param("discount_factor", 0.5),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Hard,
        });

        // 25. Near-monopoly pricing
        self.scenarios.push(Scenario {
            name: "near_monopoly".into(),
            description: "Prices very close to monopoly (CP ~ 0.95)".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 5.4),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger).with_param("cooperative_price", 5.4),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Medium,
        });

        // 26. Alternating cooperation/punishment
        self.scenarios.push(Scenario {
            name: "alternating_phases".into(),
            description: "Regular alternation between cooperative and punishment phases".into(),
            expected_verdict: ExpectedVerdict::Inconclusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::Custom("alternating".into())),
                AlgorithmConfig::new(AlgorithmType::Custom("alternating".into())),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Boundary,
            difficulty: Difficulty::Expert,
        });
    }

    // ── Adversarial scenarios (4) ───────────────────────────────────────

    fn populate_adversarial(&mut self) {
        let base_sim = SimulationConfig::default();
        let base_det = DetectionConfig::default();

        // 27. Noise-masked collusion
        self.scenarios.push(Scenario {
            name: "noise_masked_collusion".into(),
            description: "Collusive pricing with added noise to mask detection".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("noise_std", 0.3),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("noise_std", 0.3),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Adversarial,
            difficulty: Difficulty::Expert,
        });

        // 28. Randomized punishment timing
        self.scenarios.push(Scenario {
            name: "randomized_punishment".into(),
            description: "Punishment triggered with random delay to evade detection".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("random_delay", 5.0),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("random_delay", 5.0),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Adversarial,
            difficulty: Difficulty::Expert,
        });

        // 29. Frequency-modulated pricing
        self.scenarios.push(Scenario {
            name: "frequency_modulated".into(),
            description: "Price signals encoded in frequency domain".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::Custom("freq_mod".into())),
                AlgorithmConfig::new(AlgorithmType::Custom("freq_mod".into())),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Adversarial,
            difficulty: Difficulty::Expert,
        });

        // 30. Delayed retaliation
        self.scenarios.push(Scenario {
            name: "delayed_retaliation".into(),
            description: "Punishment occurs many rounds after deviation".into(),
            expected_verdict: ExpectedVerdict::Collusive,
            algorithm_configs: vec![
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("delay", 10.0),
                AlgorithmConfig::new(AlgorithmType::GrimTrigger)
                    .with_param("cooperative_price", 5.0)
                    .with_param("delay", 10.0),
            ],
            simulation_config: base_sim.clone(),
            detection_config: base_det.clone(),
            category: ScenarioCategory::Adversarial,
            difficulty: Difficulty::Expert,
        });
    }
}

impl Default for ScenarioLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario loading
// ═══════════════════════════════════════════════════════════════════════════

pub fn load_scenario(name: &str) -> Option<Scenario> {
    let library = ScenarioLibrary::new();
    library.by_name(name).cloned()
}

// ═══════════════════════════════════════════════════════════════════════════
// ScenarioGenerator (parametric sweep)
// ═══════════════════════════════════════════════════════════════════════════

pub struct ScenarioGenerator;

impl ScenarioGenerator {
    /// Generate scenarios by sweeping a parameter.
    pub fn parametric_sweep(
        base: &Scenario,
        param_name: &str,
        values: &[f64],
    ) -> Vec<Scenario> {
        values
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let mut s = base.clone();
                s.name = format!("{}_{}_sweep_{}", base.name, param_name, i);
                for config in &mut s.algorithm_configs {
                    config.extra_params.insert(param_name.to_string(), val);
                }
                s
            })
            .collect()
    }

    /// Generate discount factor sweep.
    pub fn discount_sweep(base: &Scenario, discounts: &[f64]) -> Vec<Scenario> {
        Self::parametric_sweep(base, "discount_factor", discounts)
    }

    /// Generate epsilon sweep.
    pub fn epsilon_sweep(base: &Scenario, epsilons: &[f64]) -> Vec<Scenario> {
        Self::parametric_sweep(base, "epsilon_start", epsilons)
    }

    /// Generate player count sweep.
    pub fn player_count_sweep(base: &Scenario, counts: &[usize]) -> Vec<Scenario> {
        counts
            .iter()
            .map(|&n| {
                let mut s = base.clone();
                s.name = format!("{}_{}p", base.name, n);
                s.simulation_config.game.num_players = n;
                s.simulation_config.game.marginal_costs = vec![shared_types::Cost(1.0); n];
                while s.algorithm_configs.len() < n {
                    if let Some(last) = s.algorithm_configs.last().cloned() {
                        s.algorithm_configs.push(last);
                    } else {
                        break;
                    }
                }
                s.algorithm_configs.truncate(n);
                s
            })
            .collect()
    }
}

/// Parametric sweep configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricSweep {
    pub base_scenario: String,
    pub parameter: String,
    pub values: Vec<f64>,
    pub label: String,
}

impl ParametricSweep {
    pub fn new(base_scenario: &str, parameter: &str, values: Vec<f64>) -> Self {
        Self {
            base_scenario: base_scenario.to_string(),
            parameter: parameter.to_string(),
            values,
            label: format!("{}_sweep_{}", base_scenario, parameter),
        }
    }

    pub fn generate(&self) -> Vec<Scenario> {
        let lib = ScenarioLibrary::new();
        if let Some(base) = lib.by_name(&self.base_scenario) {
            ScenarioGenerator::parametric_sweep(base, &self.parameter, &self.values)
        } else {
            Vec::new()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_library_has_30() {
        let lib = ScenarioLibrary::new();
        assert_eq!(lib.len(), 30);
    }

    #[test]
    fn test_scenario_library_categories() {
        let lib = ScenarioLibrary::new();
        assert_eq!(lib.by_category(ScenarioCategory::KnownCollusive).len(), 10);
        assert_eq!(lib.by_category(ScenarioCategory::KnownCompetitive).len(), 8);
        assert_eq!(lib.by_category(ScenarioCategory::Boundary).len(), 8);
        assert_eq!(lib.by_category(ScenarioCategory::Adversarial).len(), 4);
    }

    #[test]
    fn test_scenario_by_name() {
        let lib = ScenarioLibrary::new();
        let scenario = lib.by_name("grim_trigger_pair");
        assert!(scenario.is_some());
        assert_eq!(scenario.unwrap().expected_verdict, ExpectedVerdict::Collusive);
    }

    #[test]
    fn test_scenario_by_name_not_found() {
        let lib = ScenarioLibrary::new();
        assert!(lib.by_name("nonexistent").is_none());
    }

    #[test]
    fn test_load_scenario() {
        let s = load_scenario("nash_equilibrium");
        assert!(s.is_some());
        assert_eq!(s.unwrap().expected_verdict, ExpectedVerdict::Competitive);
    }

    #[test]
    fn test_expected_verdict_matches() {
        assert!(ExpectedVerdict::Collusive.matches(&Verdict::Collusive));
        assert!(ExpectedVerdict::Competitive.matches(&Verdict::Competitive));
        assert!(!ExpectedVerdict::Collusive.matches(&Verdict::Competitive));
    }

    #[test]
    fn test_difficulty_levels() {
        let lib = ScenarioLibrary::new();
        assert!(!lib.by_difficulty(Difficulty::Easy).is_empty());
        assert!(!lib.by_difficulty(Difficulty::Expert).is_empty());
    }

    #[test]
    fn test_parametric_sweep() {
        let lib = ScenarioLibrary::new();
        let base = lib.by_name("grim_trigger_pair").unwrap();
        let sweep = ScenarioGenerator::parametric_sweep(base, "cooperative_price", &[4.0, 5.0, 6.0]);
        assert_eq!(sweep.len(), 3);
        assert!(sweep[0].name.contains("sweep"));
    }

    #[test]
    fn test_discount_sweep() {
        let lib = ScenarioLibrary::new();
        let base = lib.by_name("q_learning_collusion").unwrap();
        let sweep = ScenarioGenerator::discount_sweep(base, &[0.5, 0.7, 0.9, 0.95, 0.99]);
        assert_eq!(sweep.len(), 5);
    }

    #[test]
    fn test_player_count_sweep() {
        let lib = ScenarioLibrary::new();
        let base = lib.by_name("grim_trigger_pair").unwrap();
        let sweep = ScenarioGenerator::player_count_sweep(base, &[2, 3, 4, 5]);
        assert_eq!(sweep.len(), 4);
        assert_eq!(sweep[2].simulation_config.num_players, 4);
        assert_eq!(sweep[2].algorithm_configs.len(), 4);
    }

    #[test]
    fn test_parametric_sweep_struct() {
        let sweep = ParametricSweep::new("grim_trigger_pair", "cooperative_price", vec![4.0, 5.0, 6.0]);
        let scenarios = sweep.generate();
        assert_eq!(scenarios.len(), 3);
    }

    #[test]
    fn test_parametric_sweep_missing_base() {
        let sweep = ParametricSweep::new("nonexistent", "param", vec![1.0]);
        let scenarios = sweep.generate();
        assert!(scenarios.is_empty());
    }

    #[test]
    fn test_all_scenarios_have_algorithms() {
        let lib = ScenarioLibrary::new();
        for scenario in lib.all() {
            assert!(
                !scenario.algorithm_configs.is_empty(),
                "Scenario {} has no algorithm configs",
                scenario.name
            );
        }
    }

    #[test]
    fn test_all_collusive_scenarios_expect_collusive() {
        let lib = ScenarioLibrary::new();
        for s in lib.by_category(ScenarioCategory::KnownCollusive) {
            assert_eq!(
                s.expected_verdict,
                ExpectedVerdict::Collusive,
                "Collusive scenario {} has wrong expected verdict",
                s.name
            );
        }
    }

    #[test]
    fn test_scenario_serialization() {
        let lib = ScenarioLibrary::new();
        let s = lib.by_name("grim_trigger_pair").unwrap();
        let json = serde_json::to_string(s).unwrap();
        let restored: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, "grim_trigger_pair");
    }

    #[test]
    fn test_library_default() {
        let lib = ScenarioLibrary::default();
        assert_eq!(lib.len(), 30);
    }
}
