//! Grim trigger strategy and variants for pricing algorithms.
//!
//! Implements the classic grim trigger strategy where cooperation is maintained
//! until a defection is detected, after which the agent punishes forever
//! (or with variants that allow forgiveness or gradual escalation).

use crate::algorithm::{AlgorithmState, PricingAlgorithm};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, MarketOutcome, PlayerAction, PlayerId, Price, RoundNumber};

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrimTriggerConfig {
    pub player_id: PlayerId,
    pub cooperative_price: Price,
    pub punishment_price: Price,
    pub cooperation_threshold: f64,
    pub num_players: usize,
}

impl Default for GrimTriggerConfig {
    fn default() -> Self {
        Self {
            player_id: PlayerId(0),
            cooperative_price: Price(5.0),
            punishment_price: Price(1.0),
            cooperation_threshold: 0.1,
            num_players: 2,
        }
    }
}

/// State of the grim trigger machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerState {
    Cooperating,
    Punishing,
}

// ═══════════════════════════════════════════════════════════════════════════
// GrimTriggerAgent
// ═══════════════════════════════════════════════════════════════════════════

/// Classic grim trigger: cooperate until defection, then punish forever.
pub struct GrimTriggerAgent {
    config: GrimTriggerConfig,
    state: TriggerState,
    trigger_round: Option<RoundNumber>,
    punishment_count: usize,
    rounds_observed: usize,
    opponent_prices: Vec<Vec<Price>>,
}

impl GrimTriggerAgent {
    pub fn new(config: GrimTriggerConfig) -> Self {
        Self {
            config,
            state: TriggerState::Cooperating,
            trigger_round: None,
            punishment_count: 0,
            rounds_observed: 0,
            opponent_prices: Vec::new(),
        }
    }

    pub fn current_state(&self) -> TriggerState {
        self.state
    }

    pub fn is_triggered(&self) -> bool {
        self.state == TriggerState::Punishing
    }

    pub fn trigger_round(&self) -> Option<RoundNumber> {
        self.trigger_round
    }

    pub fn punishment_count(&self) -> usize {
        self.punishment_count
    }

    /// Check if any opponent defected (price below cooperative threshold).
    fn detect_defection(&self, prices: &[Price]) -> bool {
        for (i, &price) in prices.iter().enumerate() {
            if i == self.config.player_id {
                continue;
            }
            if price < self.config.cooperative_price - self.config.cooperation_threshold {
                return true;
            }
        }
        false
    }
}

impl PricingAlgorithm for GrimTriggerAgent {
    fn observe(&mut self, outcome: &MarketOutcome) {
        self.opponent_prices.push(outcome.prices.clone());
        self.rounds_observed += 1;

        if self.state == TriggerState::Cooperating && self.detect_defection(&outcome.prices) {
            self.state = TriggerState::Punishing;
            self.trigger_round = Some(outcome.round);
        }

        if self.state == TriggerState::Punishing {
            self.punishment_count += 1;
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let price = match self.state {
            TriggerState::Cooperating => self.config.cooperative_price,
            TriggerState::Punishing => self.config.punishment_price,
        };
        PlayerAction::new(self.config.player_id, price)
    }

    fn reset(&mut self) {
        self.state = TriggerState::Cooperating;
        self.trigger_round = None;
        self.punishment_count = 0;
        self.rounds_observed = 0;
        self.opponent_prices.clear();
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::GrimTrigger {
            triggered: self.is_triggered(),
            trigger_round: self.trigger_round,
            punishment_count: self.punishment_count,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::GrimTrigger {
                triggered,
                trigger_round,
                punishment_count,
            } => {
                self.state = if *triggered {
                    TriggerState::Punishing
                } else {
                    TriggerState::Cooperating
                };
                self.trigger_round = *trigger_round;
                self.punishment_count = *punishment_count;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected GrimTrigger state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "GrimTrigger"
    }

    fn player_id(&self) -> PlayerId {
        self.config.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ForgivingGrimTrigger
// ═══════════════════════════════════════════════════════════════════════════

/// Variant of grim trigger that forgives with some probability each round.
pub struct ForgivingGrimTrigger {
    config: GrimTriggerConfig,
    forgiveness_probability: f64,
    state: TriggerState,
    trigger_round: Option<RoundNumber>,
    punishment_count: usize,
    rounds_observed: usize,
    rng: StdRng,
}

impl ForgivingGrimTrigger {
    pub fn new(config: GrimTriggerConfig, forgiveness_probability: f64) -> Self {
        let seed = config.player_id.0 as u64 * 67890 + 17;
        Self {
            config,
            forgiveness_probability: forgiveness_probability.clamp(0.0, 1.0),
            state: TriggerState::Cooperating,
            trigger_round: None,
            punishment_count: 0,
            rounds_observed: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn forgiveness_probability(&self) -> f64 {
        self.forgiveness_probability
    }

    pub fn current_state(&self) -> TriggerState {
        self.state
    }

    fn detect_defection(&self, prices: &[Price]) -> bool {
        for (i, &price) in prices.iter().enumerate() {
            if i == self.config.player_id {
                continue;
            }
            if price < self.config.cooperative_price - self.config.cooperation_threshold {
                return true;
            }
        }
        false
    }
}

impl PricingAlgorithm for ForgivingGrimTrigger {
    fn observe(&mut self, outcome: &MarketOutcome) {
        self.rounds_observed += 1;

        match self.state {
            TriggerState::Cooperating => {
                if self.detect_defection(&outcome.prices) {
                    self.state = TriggerState::Punishing;
                    self.trigger_round = Some(outcome.round);
                }
            }
            TriggerState::Punishing => {
                self.punishment_count += 1;
                // Try to forgive
                if self.rng.gen::<f64>() < self.forgiveness_probability {
                    self.state = TriggerState::Cooperating;
                }
            }
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let price = match self.state {
            TriggerState::Cooperating => self.config.cooperative_price,
            TriggerState::Punishing => self.config.punishment_price,
        };
        PlayerAction::new(self.config.player_id, price)
    }

    fn reset(&mut self) {
        self.state = TriggerState::Cooperating;
        self.trigger_round = None;
        self.punishment_count = 0;
        self.rounds_observed = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::GrimTrigger {
            triggered: self.state == TriggerState::Punishing,
            trigger_round: self.trigger_round,
            punishment_count: self.punishment_count,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::GrimTrigger {
                triggered,
                trigger_round,
                punishment_count,
            } => {
                self.state = if *triggered {
                    TriggerState::Punishing
                } else {
                    TriggerState::Cooperating
                };
                self.trigger_round = *trigger_round;
                self.punishment_count = *punishment_count;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected GrimTrigger state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "ForgivingGrimTrigger"
    }

    fn player_id(&self) -> PlayerId {
        self.config.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GradualTrigger
// ═══════════════════════════════════════════════════════════════════════════

/// Gradual trigger that escalates punishment incrementally.
///
/// On each defection, the punishment duration increases. After serving
/// the punishment, the agent returns to cooperation but remembers.
pub struct GradualTrigger {
    config: GrimTriggerConfig,
    state: TriggerState,
    defection_count: usize,
    punishment_rounds_remaining: usize,
    cooperation_rounds_after_punishment: usize,
    cool_off_remaining: usize,
    rounds_observed: usize,
    total_punishment: usize,
}

impl GradualTrigger {
    pub fn new(config: GrimTriggerConfig) -> Self {
        Self {
            config,
            state: TriggerState::Cooperating,
            defection_count: 0,
            punishment_rounds_remaining: 0,
            cooperation_rounds_after_punishment: 2,
            cool_off_remaining: 0,
            rounds_observed: 0,
            total_punishment: 0,
        }
    }

    pub fn defection_count(&self) -> usize {
        self.defection_count
    }

    pub fn current_state(&self) -> TriggerState {
        self.state
    }

    pub fn punishment_remaining(&self) -> usize {
        self.punishment_rounds_remaining
    }

    fn detect_defection(&self, prices: &[Price]) -> bool {
        for (i, &price) in prices.iter().enumerate() {
            if i == self.config.player_id {
                continue;
            }
            if price < self.config.cooperative_price - self.config.cooperation_threshold {
                return true;
            }
        }
        false
    }
}

impl PricingAlgorithm for GradualTrigger {
    fn observe(&mut self, outcome: &MarketOutcome) {
        self.rounds_observed += 1;

        match self.state {
            TriggerState::Cooperating => {
                if self.cool_off_remaining > 0 {
                    self.cool_off_remaining -= 1;
                } else if self.detect_defection(&outcome.prices) {
                    self.defection_count += 1;
                    // Punish for defection_count rounds
                    self.punishment_rounds_remaining = self.defection_count;
                    self.state = TriggerState::Punishing;
                }
            }
            TriggerState::Punishing => {
                self.total_punishment += 1;
                if self.punishment_rounds_remaining > 0 {
                    self.punishment_rounds_remaining -= 1;
                }
                if self.punishment_rounds_remaining == 0 {
                    self.state = TriggerState::Cooperating;
                    self.cool_off_remaining = self.cooperation_rounds_after_punishment;
                }
            }
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let price = match self.state {
            TriggerState::Cooperating => self.config.cooperative_price,
            TriggerState::Punishing => self.config.punishment_price,
        };
        PlayerAction::new(self.config.player_id, price)
    }

    fn reset(&mut self) {
        self.state = TriggerState::Cooperating;
        self.defection_count = 0;
        self.punishment_rounds_remaining = 0;
        self.cool_off_remaining = 0;
        self.rounds_observed = 0;
        self.total_punishment = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::GrimTrigger {
            triggered: self.state == TriggerState::Punishing,
            trigger_round: None,
            punishment_count: self.total_punishment,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::GrimTrigger {
                triggered,
                punishment_count,
                ..
            } => {
                self.state = if *triggered {
                    TriggerState::Punishing
                } else {
                    TriggerState::Cooperating
                };
                self.total_punishment = *punishment_count;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected GrimTrigger state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "GradualTrigger"
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

    fn default_config() -> GrimTriggerConfig {
        GrimTriggerConfig {
            player_id: 0,
            cooperative_price: 5.0,
            punishment_price: 1.0,
            cooperation_threshold: 0.1,
            num_players: 2,
        }
    }

    fn make_outcome(round: usize, prices: Vec<f64>) -> MarketOutcome {
        let n = prices.len();
        MarketOutcome::new(round, prices.clone(), vec![1.0; n], prices.iter().map(|p| p - 1.0).collect())
    }

    // ── GrimTriggerAgent tests ──────────────────────────────────────────

    #[test]
    fn test_grim_trigger_cooperates_initially() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);
        let action = agent.act(0);
        assert!((action.price - 5.0).abs() < 1e-10);
        assert_eq!(agent.current_state(), TriggerState::Cooperating);
    }

    #[test]
    fn test_grim_trigger_cooperates_when_opponent_cooperates() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);
        agent.observe(&make_outcome(0, vec![5.0, 5.0]));
        let action = agent.act(1);
        assert!((action.price - 5.0).abs() < 1e-10);
        assert_eq!(agent.current_state(), TriggerState::Cooperating);
    }

    #[test]
    fn test_grim_trigger_punishes_on_defection() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);

        // Opponent defects
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        assert!(agent.is_triggered());
        assert_eq!(agent.trigger_round(), Some(0));

        let action = agent.act(1);
        assert!((action.price - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grim_trigger_stays_punishing() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);

        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        // Opponent returns to cooperation
        agent.observe(&make_outcome(1, vec![5.0, 5.0]));
        assert!(agent.is_triggered()); // Still punishing forever

        let action = agent.act(2);
        assert!((action.price - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grim_trigger_within_threshold() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);

        // Opponent price is within threshold (4.95 >= 5.0 - 0.1 = 4.9)
        agent.observe(&make_outcome(0, vec![5.0, 4.95]));
        assert!(!agent.is_triggered());
    }

    #[test]
    fn test_grim_trigger_reset() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);

        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        assert!(agent.is_triggered());

        agent.reset();
        assert!(!agent.is_triggered());
        assert_eq!(agent.trigger_round(), None);
        assert_eq!(agent.punishment_count(), 0);
    }

    #[test]
    fn test_grim_trigger_state_serialization() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));

        let state = agent.get_state();
        let mut agent2 = GrimTriggerAgent::new(default_config());
        agent2.set_state(&state).unwrap();
        assert!(agent2.is_triggered());
    }

    #[test]
    fn test_grim_trigger_wrong_state_variant() {
        let config = default_config();
        let mut agent = GrimTriggerAgent::new(config);
        let state = AlgorithmState::Empty;
        assert!(agent.set_state(&state).is_err());
    }

    // ── ForgivingGrimTrigger tests ──────────────────────────────────────

    #[test]
    fn test_forgiving_cooperates_initially() {
        let config = default_config();
        let mut agent = ForgivingGrimTrigger::new(config, 0.5);
        let action = agent.act(0);
        assert!((action.price - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_forgiving_triggers_on_defection() {
        let config = default_config();
        let mut agent = ForgivingGrimTrigger::new(config, 0.5);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        assert_eq!(agent.current_state(), TriggerState::Punishing);
    }

    #[test]
    fn test_forgiving_eventually_forgives() {
        let config = default_config();
        let mut agent = ForgivingGrimTrigger::new(config, 1.0); // Always forgive
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        assert_eq!(agent.current_state(), TriggerState::Punishing);

        // With forgiveness_probability = 1.0, should forgive on next observation
        agent.observe(&make_outcome(1, vec![5.0, 5.0]));
        assert_eq!(agent.current_state(), TriggerState::Cooperating);
    }

    #[test]
    fn test_forgiving_never_forgives() {
        let config = default_config();
        let mut agent = ForgivingGrimTrigger::new(config, 0.0); // Never forgive
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));

        for r in 1..100 {
            agent.observe(&make_outcome(r, vec![5.0, 5.0]));
        }
        assert_eq!(agent.current_state(), TriggerState::Punishing);
    }

    #[test]
    fn test_forgiving_name() {
        let config = default_config();
        let agent = ForgivingGrimTrigger::new(config, 0.5);
        assert_eq!(agent.name(), "ForgivingGrimTrigger");
        assert!((agent.forgiveness_probability() - 0.5).abs() < 1e-10);
    }

    // ── GradualTrigger tests ────────────────────────────────────────────

    #[test]
    fn test_gradual_cooperates_initially() {
        let config = default_config();
        let mut agent = GradualTrigger::new(config);
        let action = agent.act(0);
        assert!((action.price - 5.0).abs() < 1e-10);
        assert_eq!(agent.current_state(), TriggerState::Cooperating);
    }

    #[test]
    fn test_gradual_first_defection_one_round_punishment() {
        let config = default_config();
        let mut agent = GradualTrigger::new(config);

        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        assert_eq!(agent.current_state(), TriggerState::Punishing);
        assert_eq!(agent.defection_count(), 1);

        let action = agent.act(1);
        assert!((action.price - 1.0).abs() < 1e-10);

        // After 1 round of punishment, observe to process end of punishment
        agent.observe(&make_outcome(1, vec![1.0, 5.0]));
        assert_eq!(agent.current_state(), TriggerState::Cooperating);
    }

    #[test]
    fn test_gradual_second_defection_two_rounds_punishment() {
        let config = default_config();
        let mut agent = GradualTrigger::new(config);

        // First defection
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![1.0, 5.0])); // 1 punishment round
        assert_eq!(agent.current_state(), TriggerState::Cooperating);

        // Cool off period
        agent.observe(&make_outcome(2, vec![5.0, 5.0]));
        agent.observe(&make_outcome(3, vec![5.0, 5.0]));

        // Second defection
        agent.observe(&make_outcome(4, vec![5.0, 2.0]));
        assert_eq!(agent.defection_count(), 2);
        assert_eq!(agent.punishment_remaining(), 2);
    }

    #[test]
    fn test_gradual_reset() {
        let config = default_config();
        let mut agent = GradualTrigger::new(config);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.reset();
        assert_eq!(agent.defection_count(), 0);
        assert_eq!(agent.current_state(), TriggerState::Cooperating);
    }

    #[test]
    fn test_gradual_name() {
        let config = default_config();
        let agent = GradualTrigger::new(config);
        assert_eq!(agent.name(), "GradualTrigger");
        assert_eq!(agent.player_id(), 0);
    }
}
