//! Tit-for-tat family of pricing strategies.
//!
//! Implements TFT and variants: classic TFT, TitForTwoTats, Suspicious,
//! Generous, and bounded-memory TFT.

use crate::algorithm::{AlgorithmState, PricingAlgorithm};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, MarketOutcome, PlayerAction, PlayerId, Price, RoundNumber};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// TitForTatAgent — Classic
// ═══════════════════════════════════════════════════════════════════════════

/// Classic tit-for-tat: mirror the opponent's last price.
///
/// If any opponent prices below a threshold, play punishment next round.
/// Otherwise play the cooperative price.
pub struct TitForTatAgent {
    player_id: PlayerId,
    num_players: usize,
    cooperative_price: Price,
    punishment_price: Price,
    last_opponent_prices: Vec<Price>,
    defection_count: usize,
    round: usize,
    cooperation_threshold: f64,
}

impl TitForTatAgent {
    pub fn new(
        player_id: PlayerId,
        num_players: usize,
        cooperative_price: Price,
        punishment_price: Price,
    ) -> Self {
        Self {
            player_id,
            num_players,
            cooperative_price,
            punishment_price,
            last_opponent_prices: Vec::new(),
            defection_count: 0,
            round: 0,
            cooperation_threshold: cooperative_price.0 * 0.95,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.cooperation_threshold = threshold;
        self
    }

    pub fn defection_count(&self) -> usize {
        self.defection_count
    }

    fn opponents_defected(&self) -> bool {
        self.last_opponent_prices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.player_id)
            .any(|(_, &p)| p.0 < self.cooperation_threshold)
    }
}

impl PricingAlgorithm for TitForTatAgent {
    fn observe(&mut self, outcome: &MarketOutcome) {
        self.last_opponent_prices = outcome.prices.clone();
        self.round += 1;

        if self.opponents_defected() {
            self.defection_count += 1;
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        // First round: cooperate
        if self.last_opponent_prices.is_empty() {
            return PlayerAction::new(self.player_id, self.cooperative_price);
        }

        let price = if self.opponents_defected() {
            self.punishment_price
        } else {
            self.cooperative_price
        };
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.last_opponent_prices.clear();
        self.defection_count = 0;
        self.round = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::TitForTat {
            memory: vec![self.last_opponent_prices.clone()],
            defection_count: self.defection_count,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::TitForTat {
                memory,
                defection_count,
            } => {
                if let Some(last) = memory.last() {
                    self.last_opponent_prices = last.clone();
                }
                self.defection_count = *defection_count;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected TitForTat state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "TitForTat"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TitForTwoTats
// ═══════════════════════════════════════════════════════════════════════════

/// Punish only after two consecutive rounds where any opponent defects.
pub struct TitForTwoTats {
    player_id: PlayerId,
    cooperative_price: Price,
    punishment_price: Price,
    consecutive_defections: usize,
    defection_count: usize,
    round: usize,
    cooperation_threshold: f64,
    last_prices: Vec<Vec<Price>>,
}

impl TitForTwoTats {
    pub fn new(
        player_id: PlayerId,
        num_players: usize,
        cooperative_price: Price,
        punishment_price: Price,
    ) -> Self {
        let _ = num_players;
        Self {
            player_id,
            cooperative_price,
            punishment_price,
            consecutive_defections: 0,
            defection_count: 0,
            round: 0,
            cooperation_threshold: cooperative_price.0 * 0.95,
            last_prices: Vec::new(),
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.cooperation_threshold = threshold;
        self
    }

    fn any_opponent_defected(&self, prices: &[Price]) -> bool {
        prices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.player_id)
            .any(|(_, &p)| p.0 < self.cooperation_threshold)
    }
}

impl PricingAlgorithm for TitForTwoTats {
    fn observe(&mut self, outcome: &MarketOutcome) {
        self.last_prices.push(outcome.prices.clone());
        self.round += 1;

        if self.any_opponent_defected(&outcome.prices) {
            self.consecutive_defections += 1;
            self.defection_count += 1;
        } else {
            self.consecutive_defections = 0;
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let price = if self.consecutive_defections >= 2 {
            self.punishment_price
        } else {
            self.cooperative_price
        };
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.consecutive_defections = 0;
        self.defection_count = 0;
        self.round = 0;
        self.last_prices.clear();
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::TitForTat {
            memory: self.last_prices.clone(),
            defection_count: self.defection_count,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::TitForTat {
                memory,
                defection_count,
            } => {
                self.last_prices = memory.clone();
                self.defection_count = *defection_count;
                // Recompute consecutive defections from last 2 entries
                self.consecutive_defections = 0;
                let recent: Vec<_> = self.last_prices.iter().rev().take(2).collect();
                for prices in &recent {
                    if self.any_opponent_defected(prices) {
                        self.consecutive_defections += 1;
                    } else {
                        break;
                    }
                }
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected TitForTat state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "TitForTwoTats"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SuspiciousTitForTat
// ═══════════════════════════════════════════════════════════════════════════

/// Starts with punishment, then mirrors opponent's last price.
pub struct SuspiciousTitForTat {
    player_id: PlayerId,
    cooperative_price: Price,
    punishment_price: Price,
    last_opponent_prices: Vec<Price>,
    defection_count: usize,
    round: usize,
    cooperation_threshold: f64,
}

impl SuspiciousTitForTat {
    pub fn new(
        player_id: PlayerId,
        _num_players: usize,
        cooperative_price: Price,
        punishment_price: Price,
    ) -> Self {
        Self {
            player_id,
            cooperative_price,
            punishment_price,
            last_opponent_prices: Vec::new(),
            defection_count: 0,
            round: 0,
            cooperation_threshold: cooperative_price.0 * 0.95,
        }
    }

    fn opponents_defected(&self) -> bool {
        self.last_opponent_prices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.player_id)
            .any(|(_, &p)| p.0 < self.cooperation_threshold)
    }
}

impl PricingAlgorithm for SuspiciousTitForTat {
    fn observe(&mut self, outcome: &MarketOutcome) {
        self.last_opponent_prices = outcome.prices.clone();
        self.round += 1;
        if self.opponents_defected() {
            self.defection_count += 1;
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        // First round: defect (suspicious)
        if self.last_opponent_prices.is_empty() {
            return PlayerAction::new(self.player_id, self.punishment_price);
        }

        let price = if self.opponents_defected() {
            self.punishment_price
        } else {
            self.cooperative_price
        };
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.last_opponent_prices.clear();
        self.defection_count = 0;
        self.round = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::TitForTat {
            memory: vec![self.last_opponent_prices.clone()],
            defection_count: self.defection_count,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::TitForTat {
                memory,
                defection_count,
            } => {
                if let Some(last) = memory.last() {
                    self.last_opponent_prices = last.clone();
                }
                self.defection_count = *defection_count;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected TitForTat state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "SuspiciousTitForTat"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GenerousTitForTat
// ═══════════════════════════════════════════════════════════════════════════

/// Sometimes cooperates even after opponent defects.
pub struct GenerousTitForTat {
    player_id: PlayerId,
    cooperative_price: Price,
    punishment_price: Price,
    generosity: f64,
    last_opponent_prices: Vec<Price>,
    defection_count: usize,
    round: usize,
    cooperation_threshold: f64,
    rng: StdRng,
}

impl GenerousTitForTat {
    pub fn new(
        player_id: PlayerId,
        _num_players: usize,
        cooperative_price: Price,
        punishment_price: Price,
        generosity: f64,
    ) -> Self {
        Self {
            player_id,
            cooperative_price,
            punishment_price,
            generosity: generosity.clamp(0.0, 1.0),
            last_opponent_prices: Vec::new(),
            defection_count: 0,
            round: 0,
            cooperation_threshold: cooperative_price.0 * 0.95,
            rng: StdRng::seed_from_u64(player_id.0 as u64 * 54321 + 7),
        }
    }

    pub fn generosity(&self) -> f64 {
        self.generosity
    }

    fn opponents_defected(&self) -> bool {
        self.last_opponent_prices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.player_id)
            .any(|(_, &p)| p.0 < self.cooperation_threshold)
    }
}

impl PricingAlgorithm for GenerousTitForTat {
    fn observe(&mut self, outcome: &MarketOutcome) {
        self.last_opponent_prices = outcome.prices.clone();
        self.round += 1;
        if self.opponents_defected() {
            self.defection_count += 1;
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        if self.last_opponent_prices.is_empty() {
            return PlayerAction::new(self.player_id, self.cooperative_price);
        }

        let price = if self.opponents_defected() {
            // Be generous sometimes
            if self.rng.gen::<f64>() < self.generosity {
                self.cooperative_price
            } else {
                self.punishment_price
            }
        } else {
            self.cooperative_price
        };
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.last_opponent_prices.clear();
        self.defection_count = 0;
        self.round = 0;
    }

    fn get_state(&self) -> AlgorithmState {
        AlgorithmState::TitForTat {
            memory: vec![self.last_opponent_prices.clone()],
            defection_count: self.defection_count,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::TitForTat {
                memory,
                defection_count,
            } => {
                if let Some(last) = memory.last() {
                    self.last_opponent_prices = last.clone();
                }
                self.defection_count = *defection_count;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected TitForTat state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "GenerousTitForTat"
    }

    fn player_id(&self) -> PlayerId {
        self.player_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BoundedMemoryTFT
// ═══════════════════════════════════════════════════════════════════════════

/// Tit-for-tat considering the last M rounds.
///
/// If more than half of the last M rounds had opponent defections,
/// play the punishment price.
pub struct BoundedMemoryTFT {
    player_id: PlayerId,
    cooperative_price: Price,
    punishment_price: Price,
    memory_size: usize,
    history: VecDeque<bool>,
    defection_count: usize,
    round: usize,
    cooperation_threshold: f64,
    all_prices_history: Vec<Vec<Price>>,
}

impl BoundedMemoryTFT {
    pub fn new(
        player_id: PlayerId,
        _num_players: usize,
        cooperative_price: Price,
        punishment_price: Price,
        memory_size: usize,
    ) -> Self {
        Self {
            player_id,
            cooperative_price,
            punishment_price,
            memory_size: memory_size.max(1),
            history: VecDeque::with_capacity(memory_size + 1),
            defection_count: 0,
            round: 0,
            cooperation_threshold: cooperative_price.0 * 0.95,
            all_prices_history: Vec::new(),
        }
    }

    pub fn memory_size(&self) -> usize {
        self.memory_size
    }

    /// Fraction of recent rounds with defection.
    pub fn defection_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let defections = self.history.iter().filter(|&&d| d).count();
        defections as f64 / self.history.len() as f64
    }

    fn any_opponent_defected(&self, prices: &[Price]) -> bool {
        prices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.player_id)
            .any(|(_, &p)| p.0 < self.cooperation_threshold)
    }
}

impl PricingAlgorithm for BoundedMemoryTFT {
    fn observe(&mut self, outcome: &MarketOutcome) {
        let defected = self.any_opponent_defected(&outcome.prices);
        self.history.push_back(defected);
        if self.history.len() > self.memory_size {
            self.history.pop_front();
        }
        self.all_prices_history.push(outcome.prices.clone());
        self.round += 1;
        if defected {
            self.defection_count += 1;
        }
    }

    fn act(&mut self, _round: RoundNumber) -> PlayerAction {
        let price = if self.defection_rate() > 0.5 {
            self.punishment_price
        } else {
            self.cooperative_price
        };
        PlayerAction::new(self.player_id, price)
    }

    fn reset(&mut self) {
        self.history.clear();
        self.defection_count = 0;
        self.round = 0;
        self.all_prices_history.clear();
    }

    fn get_state(&self) -> AlgorithmState {
        let memory: Vec<Vec<Price>> = self.all_prices_history
            .iter()
            .rev()
            .take(self.memory_size)
            .rev()
            .cloned()
            .collect();
        AlgorithmState::TitForTat {
            memory,
            defection_count: self.defection_count,
        }
    }

    fn set_state(&mut self, state: &AlgorithmState) -> CollusionResult<()> {
        match state {
            AlgorithmState::TitForTat {
                memory,
                defection_count,
            } => {
                self.history.clear();
                self.all_prices_history = memory.clone();
                for prices in memory {
                    let defected = self.any_opponent_defected(prices);
                    self.history.push_back(defected);
                    if self.history.len() > self.memory_size {
                        self.history.pop_front();
                    }
                }
                self.defection_count = *defection_count;
                Ok(())
            }
            _ => Err(CollusionError::InvalidState(
                "Expected TitForTat state variant".into(),
            )),
        }
    }

    fn name(&self) -> &str {
        "BoundedMemoryTFT"
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

    fn make_outcome(round: usize, prices: Vec<f64>) -> MarketOutcome {
        let n = prices.len();
        MarketOutcome::new(round, prices.clone(), vec![1.0; n], prices.iter().map(|p| p - 1.0).collect())
    }

    // ── TitForTatAgent tests ────────────────────────────────────────────

    #[test]
    fn test_tft_cooperates_first_round() {
        let mut agent = TitForTatAgent::new(0, 2, 5.0, 1.0);
        let action = agent.act(0);
        assert!((action.price - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tft_cooperates_after_cooperation() {
        let mut agent = TitForTatAgent::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 5.0]));
        let action = agent.act(1);
        assert!((action.price - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tft_punishes_after_defection() {
        let mut agent = TitForTatAgent::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        let action = agent.act(1);
        assert!((action.price - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tft_forgives_after_cooperation_resumes() {
        let mut agent = TitForTatAgent::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![1.0, 5.0]));
        let action = agent.act(2);
        assert!((action.price - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tft_reset() {
        let mut agent = TitForTatAgent::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.reset();
        let action = agent.act(0);
        assert!((action.price - 5.0).abs() < 1e-10);
        assert_eq!(agent.defection_count(), 0);
    }

    // ── TitForTwoTats tests ────────────────────────────────────────────

    #[test]
    fn test_t42t_cooperates_after_single_defection() {
        let mut agent = TitForTwoTats::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        let action = agent.act(1);
        assert!((action.price - 5.0).abs() < 1e-10); // Still cooperating
    }

    #[test]
    fn test_t42t_punishes_after_two_consecutive_defections() {
        let mut agent = TitForTwoTats::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![5.0, 2.0]));
        let action = agent.act(2);
        assert!((action.price - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_t42t_resets_on_cooperation() {
        let mut agent = TitForTwoTats::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![5.0, 5.0])); // Cooperation resets
        agent.observe(&make_outcome(2, vec![5.0, 2.0]));
        let action = agent.act(3);
        assert!((action.price - 5.0).abs() < 1e-10); // Only 1 consecutive
    }

    // ── SuspiciousTitForTat tests ───────────────────────────────────────

    #[test]
    fn test_suspicious_defects_first_round() {
        let mut agent = SuspiciousTitForTat::new(0, 2, 5.0, 1.0);
        let action = agent.act(0);
        assert!((action.price - 1.0).abs() < 1e-10); // Starts with punishment
    }

    #[test]
    fn test_suspicious_cooperates_after_opponent_cooperates() {
        let mut agent = SuspiciousTitForTat::new(0, 2, 5.0, 1.0);
        agent.observe(&make_outcome(0, vec![1.0, 5.0]));
        let action = agent.act(1);
        assert!((action.price - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_suspicious_name() {
        let agent = SuspiciousTitForTat::new(0, 2, 5.0, 1.0);
        assert_eq!(agent.name(), "SuspiciousTitForTat");
    }

    // ── GenerousTitForTat tests ─────────────────────────────────────────

    #[test]
    fn test_generous_cooperates_first() {
        let mut agent = GenerousTitForTat::new(0, 2, 5.0, 1.0, 0.5);
        let action = agent.act(0);
        assert!((action.price - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_generous_always_generous() {
        let mut agent = GenerousTitForTat::new(0, 2, 5.0, 1.0, 1.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        let action = agent.act(1);
        assert!((action.price - 5.0).abs() < 1e-10); // Always generous
    }

    #[test]
    fn test_generous_never_generous() {
        let mut agent = GenerousTitForTat::new(0, 2, 5.0, 1.0, 0.0);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        let action = agent.act(1);
        assert!((action.price - 1.0).abs() < 1e-10); // Never generous
    }

    #[test]
    fn test_generous_generosity_accessor() {
        let agent = GenerousTitForTat::new(0, 2, 5.0, 1.0, 0.3);
        assert!((agent.generosity() - 0.3).abs() < 1e-10);
    }

    // ── BoundedMemoryTFT tests ──────────────────────────────────────────

    #[test]
    fn test_bounded_cooperates_when_few_defections() {
        let mut agent = BoundedMemoryTFT::new(0, 2, 5.0, 1.0, 4);
        agent.observe(&make_outcome(0, vec![5.0, 5.0]));
        agent.observe(&make_outcome(1, vec![5.0, 2.0])); // 1 defection out of 2
        let action = agent.act(2);
        assert!((action.price - 5.0).abs() < 1e-10); // < 50% defections
    }

    #[test]
    fn test_bounded_punishes_when_many_defections() {
        let mut agent = BoundedMemoryTFT::new(0, 2, 5.0, 1.0, 4);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![5.0, 2.0]));
        agent.observe(&make_outcome(2, vec![5.0, 2.0]));
        let action = agent.act(3);
        assert!((action.price - 1.0).abs() < 1e-10); // 100% defections
    }

    #[test]
    fn test_bounded_memory_sliding_window() {
        let mut agent = BoundedMemoryTFT::new(0, 2, 5.0, 1.0, 3);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![5.0, 2.0]));
        agent.observe(&make_outcome(2, vec![5.0, 5.0]));
        agent.observe(&make_outcome(3, vec![5.0, 5.0]));
        agent.observe(&make_outcome(4, vec![5.0, 5.0]));
        // Window: [coop, coop, coop] => defection_rate = 0
        let action = agent.act(5);
        assert!((action.price - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounded_defection_rate() {
        let mut agent = BoundedMemoryTFT::new(0, 2, 5.0, 1.0, 4);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![5.0, 5.0]));
        assert!((agent.defection_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bounded_reset() {
        let mut agent = BoundedMemoryTFT::new(0, 2, 5.0, 1.0, 4);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.reset();
        assert!((agent.defection_rate() - 0.0).abs() < 1e-10);
        assert_eq!(agent.memory_size(), 4);
    }

    #[test]
    fn test_bounded_state_serialization() {
        let mut agent = BoundedMemoryTFT::new(0, 2, 5.0, 1.0, 3);
        agent.observe(&make_outcome(0, vec![5.0, 2.0]));
        agent.observe(&make_outcome(1, vec![5.0, 5.0]));

        let state = agent.get_state();
        let mut agent2 = BoundedMemoryTFT::new(0, 2, 5.0, 1.0, 3);
        agent2.set_state(&state).unwrap();
        assert!((agent2.defection_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tft_state_wrong_variant() {
        let mut agent = TitForTatAgent::new(0, 2, 5.0, 1.0);
        let state = AlgorithmState::Empty;
        assert!(agent.set_state(&state).is_err());
    }
}
