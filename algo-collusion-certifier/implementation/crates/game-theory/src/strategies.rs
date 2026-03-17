//! Concrete strategy implementations as finite-state automata.
//!
//! Provides classic repeated-game strategies: Grim Trigger, Tit-for-Tat,
//! Win-Stay-Lose-Shift, bounded recall, and more.

use crate::automaton::{AutomatonBuilder, DiscretizedPrice, FiniteStateStrategy, MealyMachine, AutomatonState};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

// ── Strategy trait ──────────────────────────────────────────────────────────

/// Actions in a simplified cooperate/defect framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Cooperate,
    Defect,
}

impl Action {
    pub fn is_cooperate(&self) -> bool {
        matches!(self, Action::Cooperate)
    }
    pub fn is_defect(&self) -> bool {
        matches!(self, Action::Defect)
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Cooperate => write!(f, "C"),
            Action::Defect => write!(f, "D"),
        }
    }
}

/// Common interface for all repeated-game strategies.
pub trait Strategy: fmt::Debug + Send + Sync {
    /// Name of the strategy.
    fn name(&self) -> &str;

    /// Choose an action given the history of opponent actions.
    fn choose(&self, opponent_history: &[Action]) -> Action;

    /// Number of internal states.
    fn num_states(&self) -> usize;

    /// Reset to initial state (for stateful implementations).
    fn reset(&mut self) {}

    /// Convert to a finite-state strategy with the given price discretization.
    fn to_finite_state(&self, num_price_levels: u32, min_price: f64, max_price: f64) -> FiniteStateStrategy;

    /// Run the strategy against a given opponent history, returning full action sequence.
    fn play_sequence(&self, opponent_history: &[Action]) -> Vec<Action> {
        let mut actions = Vec::with_capacity(opponent_history.len() + 1);
        // First move has empty history
        actions.push(self.choose(&[]));
        for t in 0..opponent_history.len() {
            actions.push(self.choose(&opponent_history[..=t]));
        }
        actions
    }
}

// Helper: map cooperate/defect to price levels.
fn coop_price(num_levels: u32) -> DiscretizedPrice {
    DiscretizedPrice(num_levels.saturating_sub(1))
}
fn defect_price(_num_levels: u32) -> DiscretizedPrice {
    DiscretizedPrice(0)
}
// Threshold: actions at or above this are "cooperative"
fn coop_threshold(num_levels: u32) -> u32 {
    num_levels / 2
}

// ── Grim Trigger ────────────────────────────────────────────────────────────

/// Cooperate until opponent defects once, then defect forever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrimTriggerStrategy;

impl Strategy for GrimTriggerStrategy {
    fn name(&self) -> &str { "GrimTrigger" }
    fn num_states(&self) -> usize { 2 }

    fn choose(&self, opponent_history: &[Action]) -> Action {
        if opponent_history.iter().any(|a| a.is_defect()) {
            Action::Defect
        } else {
            Action::Cooperate
        }
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        let thresh = coop_threshold(num_levels);
        AutomatonBuilder::new("GrimTrigger", num_levels)
            .price_range(min_p, max_p)
            .add_state("cooperate")
            .add_state("punish")
            .initial(0)
            // In cooperate state: high opponent price (>= thresh) stays; low triggers punish
            .transition_threshold(0, thresh, 1, 0)
            // In punish state: stay forever
            .transition_all(1, 1)
            .output(0, num_levels - 1) // cooperate = high price
            .output(1, 0)              // punish = low price
            .build()
    }
}

// ── Tit for Tat ─────────────────────────────────────────────────────────────

/// Mirror the opponent's last action. Start by cooperating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitForTatStrategy;

impl Strategy for TitForTatStrategy {
    fn name(&self) -> &str { "TitForTat" }
    fn num_states(&self) -> usize { 2 }

    fn choose(&self, opponent_history: &[Action]) -> Action {
        match opponent_history.last() {
            None | Some(Action::Cooperate) => Action::Cooperate,
            Some(Action::Defect) => Action::Defect,
        }
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        let thresh = coop_threshold(num_levels);
        AutomatonBuilder::new("TitForTat", num_levels)
            .price_range(min_p, max_p)
            .add_state("cooperate")
            .add_state("defect")
            .initial(0)
            // cooperate state: high opp -> stay cooperate; low opp -> defect
            .transition_threshold(0, thresh, 1, 0)
            // defect state: high opp -> cooperate; low opp -> stay defect
            .transition_threshold(1, thresh, 1, 0)
            .output(0, num_levels - 1)
            .output(1, 0)
            .build()
    }
}

// ── Tit for Two Tats ────────────────────────────────────────────────────────

/// Only punish after two consecutive defections by the opponent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitForTwoTatsStrategy;

impl Strategy for TitForTwoTatsStrategy {
    fn name(&self) -> &str { "TitForTwoTats" }
    fn num_states(&self) -> usize { 3 }

    fn choose(&self, opponent_history: &[Action]) -> Action {
        let len = opponent_history.len();
        if len >= 2
            && opponent_history[len - 1].is_defect()
            && opponent_history[len - 2].is_defect()
        {
            Action::Defect
        } else {
            Action::Cooperate
        }
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        let thresh = coop_threshold(num_levels);
        // States: 0 = no recent defections, 1 = one recent defection, 2 = punish
        let mut strategy = FiniteStateStrategy::new("TitForTwoTats", 3, num_levels, min_p, max_p);
        strategy.initial_state = 0;
        strategy.set_output(0, coop_price(num_levels));
        strategy.set_output(1, coop_price(num_levels));
        strategy.set_output(2, defect_price(num_levels));

        for inp in 0..num_levels {
            if inp >= thresh {
                // Opponent cooperated
                strategy.set_transition(0, inp, 0);
                strategy.set_transition(1, inp, 0);
                strategy.set_transition(2, inp, 0);
            } else {
                // Opponent defected
                strategy.set_transition(0, inp, 1);
                strategy.set_transition(1, inp, 2);
                strategy.set_transition(2, inp, 2);
            }
        }
        strategy
    }
}

// ── Win-Stay Lose-Shift (Pavlov) ────────────────────────────────────────────

/// Cooperate if last round's payoff was "good" (both cooperated or both defected
/// in the standard interpretation). Two states based on payoff comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinStayLoseShift;

impl Strategy for WinStayLoseShift {
    fn name(&self) -> &str { "WinStayLoseShift" }
    fn num_states(&self) -> usize { 2 }

    fn choose(&self, opponent_history: &[Action]) -> Action {
        if opponent_history.is_empty() {
            return Action::Cooperate;
        }
        // Count defections in last round to determine "win" or "loss".
        // In Prisoner's Dilemma: CC and DD are "win" (repeat), CD and DC are "loss" (switch).
        // We simulate: current action is the one we would play.
        // With only opponent history, we reconstruct: if opp cooperated, we "won" so stay.
        // If opp defected, we "lost" so shift.
        let last_opp = opponent_history.last().unwrap();
        // Determine our last action by replaying
        let my_last = if opponent_history.len() <= 1 {
            Action::Cooperate // our first move
        } else {
            self.choose(&opponent_history[..opponent_history.len() - 1])
        };

        // Win = mutual cooperation or mutual defection -> stay
        // Lose = asymmetric -> shift
        if my_last == *last_opp {
            my_last // stay
        } else {
            // shift
            match my_last {
                Action::Cooperate => Action::Defect,
                Action::Defect => Action::Cooperate,
            }
        }
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        let thresh = coop_threshold(num_levels);
        // State 0: cooperating, State 1: defecting
        // Pavlov rule: if outcome was "good" (same action), stay. Otherwise switch.
        let mut strategy = FiniteStateStrategy::new("WinStayLoseShift", 2, num_levels, min_p, max_p);
        strategy.initial_state = 0;
        strategy.set_output(0, coop_price(num_levels));
        strategy.set_output(1, defect_price(num_levels));

        for inp in 0..num_levels {
            if inp >= thresh {
                // Opponent cooperated
                // State 0 (I cooperated, opp cooperated) -> win -> stay 0
                strategy.set_transition(0, inp, 0);
                // State 1 (I defected, opp cooperated) -> lose -> switch to 0
                strategy.set_transition(1, inp, 0);
            } else {
                // Opponent defected
                // State 0 (I cooperated, opp defected) -> lose -> switch to 1
                strategy.set_transition(0, inp, 1);
                // State 1 (I defected, opp defected) -> win -> stay 1
                strategy.set_transition(1, inp, 1);
            }
        }
        strategy
    }
}

// ── Always Defect ───────────────────────────────────────────────────────────

/// Always play the lowest price (defect). 1 state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlwaysDefect;

impl Strategy for AlwaysDefect {
    fn name(&self) -> &str { "AlwaysDefect" }
    fn num_states(&self) -> usize { 1 }

    fn choose(&self, _opponent_history: &[Action]) -> Action {
        Action::Defect
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        AutomatonBuilder::new("AlwaysDefect", num_levels)
            .price_range(min_p, max_p)
            .add_state("defect")
            .initial(0)
            .transition_all(0, 0)
            .output(0, 0)
            .build()
    }
}

// ── Always Cooperate ────────────────────────────────────────────────────────

/// Always play the highest price (cooperate). 1 state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlwaysCooperate;

impl Strategy for AlwaysCooperate {
    fn name(&self) -> &str { "AlwaysCooperate" }
    fn num_states(&self) -> usize { 1 }

    fn choose(&self, _opponent_history: &[Action]) -> Action {
        Action::Cooperate
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        AutomatonBuilder::new("AlwaysCooperate", num_levels)
            .price_range(min_p, max_p)
            .add_state("cooperate")
            .initial(0)
            .transition_all(0, 0)
            .output(0, num_levels - 1)
            .build()
    }
}

// ── Soft Majority ───────────────────────────────────────────────────────────

/// Cooperate if opponent cooperated in at least half of the last M rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftMajority {
    pub window: usize,
}

impl SoftMajority {
    pub fn new(window: usize) -> Self {
        Self { window: window.max(1) }
    }
}

impl Strategy for SoftMajority {
    fn name(&self) -> &str { "SoftMajority" }

    fn num_states(&self) -> usize {
        // States encode the number of cooperations in the window: 0..=window
        self.window + 1
    }

    fn choose(&self, opponent_history: &[Action]) -> Action {
        let start = opponent_history.len().saturating_sub(self.window);
        let coop_count = opponent_history[start..]
            .iter()
            .filter(|a| a.is_cooperate())
            .count();
        let total = opponent_history.len().min(self.window);
        if total == 0 || coop_count * 2 >= total {
            Action::Cooperate
        } else {
            Action::Defect
        }
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        let thresh = coop_threshold(num_levels);
        let m = self.window;
        let num_states = m + 1; // state = number of cooperations in window
        let mut strategy = FiniteStateStrategy::new("SoftMajority", num_states, num_levels, min_p, max_p);
        strategy.initial_state = m; // assume full cooperation initially

        for state in 0..num_states {
            // Output: cooperate if coop_count >= ceil(m/2)
            if state * 2 >= m {
                strategy.set_output(state, coop_price(num_levels));
            } else {
                strategy.set_output(state, defect_price(num_levels));
            }

            for inp in 0..num_levels {
                let opp_cooperated = inp >= thresh;
                // Sliding window approximation: we track coop count.
                // When a new round arrives, we add the new observation and remove the oldest.
                // In steady state (window full), new_count = state - (oldest_was_coop ? 1 : 0) + new.
                // Since we cannot track the oldest exactly with a single counter,
                // we approximate: new_state = clamp(state + if new_coop {1} else {0} - state/m, 0, m).
                // For exact tracking, we'd need 2^m states, but this is the bounded approximation.
                let increment = if opp_cooperated { 1i32 } else { 0 };
                // Remove approximate oldest: assume uniform distribution
                let expected_removal = if m > 0 { (state as f64 / m as f64).round() as i32 } else { 0 };
                let new_state = ((state as i32) + increment - expected_removal)
                    .max(0)
                    .min(m as i32) as usize;
                strategy.set_transition(state, inp, new_state);
            }
        }
        strategy
    }
}

// ── Bounded Recall Strategy ─────────────────────────────────────────────────

/// Generic M-state bounded recall strategy. The state encodes a summary of
/// the last M opponent actions (as a sliding counter of cooperations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedRecallStrategy {
    pub recall: usize,
    pub threshold_fraction: f64,
    name: String,
}

impl BoundedRecallStrategy {
    pub fn new(recall: usize, threshold_fraction: f64) -> Self {
        Self {
            recall: recall.max(1),
            threshold_fraction: threshold_fraction.clamp(0.0, 1.0),
            name: format!("BoundedRecall(M={})", recall),
        }
    }
}

impl Strategy for BoundedRecallStrategy {
    fn name(&self) -> &str { &self.name }

    fn num_states(&self) -> usize {
        self.recall + 1
    }

    fn choose(&self, opponent_history: &[Action]) -> Action {
        let start = opponent_history.len().saturating_sub(self.recall);
        let window = &opponent_history[start..];
        let total = window.len();
        if total == 0 {
            return Action::Cooperate;
        }
        let coop_count = window.iter().filter(|a| a.is_cooperate()).count();
        if (coop_count as f64 / total as f64) >= self.threshold_fraction {
            Action::Cooperate
        } else {
            Action::Defect
        }
    }

    fn to_finite_state(&self, num_levels: u32, min_p: f64, max_p: f64) -> FiniteStateStrategy {
        let thresh = coop_threshold(num_levels);
        let m = self.recall;
        let num_states = m + 1;
        let mut strategy = FiniteStateStrategy::new(&self.name, num_states, num_levels, min_p, max_p);
        strategy.initial_state = m;

        let coop_threshold_count = (m as f64 * self.threshold_fraction).ceil() as usize;

        for state in 0..num_states {
            if state >= coop_threshold_count {
                strategy.set_output(state, coop_price(num_levels));
            } else {
                strategy.set_output(state, defect_price(num_levels));
            }

            for inp in 0..num_levels {
                let opp_cooperated = inp >= thresh;
                let increment = if opp_cooperated { 1i32 } else { 0 };
                let expected_removal = if m > 0 {
                    (state as f64 / m as f64).round() as i32
                } else {
                    0
                };
                let new_state = ((state as i32) + increment - expected_removal)
                    .max(0)
                    .min(m as i32) as usize;
                strategy.set_transition(state, inp, new_state);
            }
        }
        strategy
    }
}

// ── Conversion utilities ────────────────────────────────────────────────────

/// Convert a Strategy trait object into a MealyMachine representation.
pub fn strategy_to_mealy(
    strategy: &dyn Strategy,
    num_price_levels: u32,
    min_price: f64,
    max_price: f64,
) -> MealyMachine<usize, u32, DiscretizedPrice> {
    let fs = strategy.to_finite_state(num_price_levels, min_price, max_price);
    fs.to_mealy()
}

/// Classify an observed price sequence into cooperate/defect actions.
pub fn classify_price_actions(
    prices: &[f64],
    cooperative_price: f64,
    competitive_price: f64,
) -> Vec<Action> {
    let midpoint = (cooperative_price + competitive_price) / 2.0;
    prices
        .iter()
        .map(|&p| {
            if p >= midpoint {
                Action::Cooperate
            } else {
                Action::Defect
            }
        })
        .collect()
}

/// Simulate a two-player iterated game between two strategies.
pub fn simulate_match(
    strategy_a: &dyn Strategy,
    strategy_b: &dyn Strategy,
    rounds: usize,
) -> (Vec<Action>, Vec<Action>) {
    let mut history_a = Vec::with_capacity(rounds);
    let mut history_b = Vec::with_capacity(rounds);

    for t in 0..rounds {
        let action_a = strategy_a.choose(&history_b[..t]);
        let action_b = strategy_b.choose(&history_a[..t]);
        history_a.push(action_a);
        history_b.push(action_b);
    }

    (history_a, history_b)
}

/// Compute cooperation rate from a sequence of actions.
pub fn cooperation_rate(actions: &[Action]) -> f64 {
    if actions.is_empty() {
        return 0.0;
    }
    let coop = actions.iter().filter(|a| a.is_cooperate()).count();
    coop as f64 / actions.len() as f64
}

/// Identify the strategy type from observed behavior (simple heuristic).
pub fn identify_strategy(my_actions: &[Action], opp_actions: &[Action]) -> String {
    if my_actions.is_empty() {
        return "Unknown".to_string();
    }

    // Check AlwaysCooperate
    if my_actions.iter().all(|a| a.is_cooperate()) {
        return "AlwaysCooperate".to_string();
    }

    // Check AlwaysDefect
    if my_actions.iter().all(|a| a.is_defect()) {
        return "AlwaysDefect".to_string();
    }

    // Check GrimTrigger: cooperate until first opp defect, then all defect
    let first_opp_defect = opp_actions.iter().position(|a| a.is_defect());
    if let Some(idx) = first_opp_defect {
        let grim_match = my_actions[..=idx].iter().all(|a| a.is_cooperate())
            && (idx + 2 >= my_actions.len()
                || my_actions[idx + 2..].iter().all(|a| a.is_defect()));
        if grim_match {
            return "GrimTrigger".to_string();
        }
    }

    // Check TitForTat: my_action[t] should match opp_action[t-1]
    let tft_match = my_actions
        .iter()
        .skip(1)
        .zip(opp_actions.iter())
        .all(|(my, opp)| my == opp);
    if tft_match && my_actions[0].is_cooperate() {
        return "TitForTat".to_string();
    }

    "Unknown".to_string()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grim_trigger_cooperates_initially() {
        let gt = GrimTriggerStrategy;
        assert_eq!(gt.choose(&[]), Action::Cooperate);
    }

    #[test]
    fn test_grim_trigger_punishes_after_defection() {
        let gt = GrimTriggerStrategy;
        assert_eq!(gt.choose(&[Action::Cooperate, Action::Defect]), Action::Defect);
        assert_eq!(
            gt.choose(&[Action::Cooperate, Action::Defect, Action::Cooperate]),
            Action::Defect
        );
    }

    #[test]
    fn test_tit_for_tat_mirrors() {
        let tft = TitForTatStrategy;
        assert_eq!(tft.choose(&[]), Action::Cooperate);
        assert_eq!(tft.choose(&[Action::Defect]), Action::Defect);
        assert_eq!(tft.choose(&[Action::Defect, Action::Cooperate]), Action::Cooperate);
    }

    #[test]
    fn test_tit_for_two_tats() {
        let tft2 = TitForTwoTatsStrategy;
        assert_eq!(tft2.choose(&[Action::Defect]), Action::Cooperate);
        assert_eq!(tft2.choose(&[Action::Defect, Action::Defect]), Action::Defect);
        assert_eq!(
            tft2.choose(&[Action::Defect, Action::Cooperate, Action::Defect]),
            Action::Cooperate
        );
    }

    #[test]
    fn test_always_defect() {
        let ad = AlwaysDefect;
        assert_eq!(ad.choose(&[Action::Cooperate, Action::Cooperate]), Action::Defect);
    }

    #[test]
    fn test_always_cooperate() {
        let ac = AlwaysCooperate;
        assert_eq!(ac.choose(&[Action::Defect, Action::Defect]), Action::Cooperate);
    }

    #[test]
    fn test_win_stay_lose_shift_initial() {
        let wsls = WinStayLoseShift;
        assert_eq!(wsls.choose(&[]), Action::Cooperate);
    }

    #[test]
    fn test_win_stay_lose_shift_mutual_coop() {
        let wsls = WinStayLoseShift;
        // Opponent cooperated on first round, we cooperated -> win -> stay cooperate
        assert_eq!(wsls.choose(&[Action::Cooperate]), Action::Cooperate);
    }

    #[test]
    fn test_win_stay_lose_shift_exploited() {
        let wsls = WinStayLoseShift;
        // Opponent defected, we cooperated -> lose -> switch to defect
        assert_eq!(wsls.choose(&[Action::Defect]), Action::Defect);
    }

    #[test]
    fn test_soft_majority() {
        let sm = SoftMajority::new(3);
        assert_eq!(sm.choose(&[Action::Cooperate, Action::Cooperate, Action::Defect]), Action::Cooperate);
        assert_eq!(sm.choose(&[Action::Defect, Action::Defect, Action::Cooperate]), Action::Defect);
    }

    #[test]
    fn test_soft_majority_empty_history() {
        let sm = SoftMajority::new(3);
        assert_eq!(sm.choose(&[]), Action::Cooperate);
    }

    #[test]
    fn test_bounded_recall() {
        let br = BoundedRecallStrategy::new(2, 0.5);
        assert_eq!(br.choose(&[Action::Cooperate, Action::Cooperate]), Action::Cooperate);
        assert_eq!(br.choose(&[Action::Defect, Action::Defect]), Action::Defect);
    }

    #[test]
    fn test_bounded_recall_threshold() {
        let br = BoundedRecallStrategy::new(4, 0.75);
        // 3 out of 4 = 0.75, so this should cooperate
        let history = vec![Action::Cooperate, Action::Cooperate, Action::Cooperate, Action::Defect];
        assert_eq!(br.choose(&history), Action::Cooperate);
    }

    #[test]
    fn test_strategy_num_states() {
        assert_eq!(GrimTriggerStrategy.num_states(), 2);
        assert_eq!(TitForTatStrategy.num_states(), 2);
        assert_eq!(TitForTwoTatsStrategy.num_states(), 3);
        assert_eq!(WinStayLoseShift.num_states(), 2);
        assert_eq!(AlwaysDefect.num_states(), 1);
        assert_eq!(AlwaysCooperate.num_states(), 1);
        assert_eq!(SoftMajority::new(5).num_states(), 6);
    }

    #[test]
    fn test_strategy_to_finite_state() {
        let gt = GrimTriggerStrategy;
        let fs = gt.to_finite_state(4, 0.0, 10.0);
        assert_eq!(fs.num_states, 2);
        assert_eq!(fs.num_price_levels, 4);
    }

    #[test]
    fn test_strategy_to_mealy() {
        let tft = TitForTatStrategy;
        let mealy = strategy_to_mealy(&tft, 4, 0.0, 10.0);
        assert_eq!(mealy.num_states(), 2);
    }

    #[test]
    fn test_classify_price_actions() {
        let prices = vec![5.0, 2.0, 8.0, 1.0];
        let actions = classify_price_actions(&prices, 8.0, 2.0);
        assert_eq!(actions, vec![Action::Cooperate, Action::Defect, Action::Cooperate, Action::Defect]);
    }

    #[test]
    fn test_simulate_match_tft_vs_tft() {
        let (a, b) = simulate_match(&TitForTatStrategy, &TitForTatStrategy, 10);
        // TFT vs TFT should result in mutual cooperation
        assert!(a.iter().all(|x| x.is_cooperate()));
        assert!(b.iter().all(|x| x.is_cooperate()));
    }

    #[test]
    fn test_simulate_match_grim_vs_always_defect() {
        let (a, b) = simulate_match(&GrimTriggerStrategy, &AlwaysDefect, 10);
        // Grim cooperates first round, then sees defect and punishes
        assert_eq!(a[0], Action::Cooperate);
        assert!(a[2..].iter().all(|x| x.is_defect()));
        assert!(b.iter().all(|x| x.is_defect()));
    }

    #[test]
    fn test_cooperation_rate() {
        let actions = vec![Action::Cooperate, Action::Cooperate, Action::Defect, Action::Cooperate];
        assert!((cooperation_rate(&actions) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_cooperation_rate_empty() {
        assert!((cooperation_rate(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_identify_always_cooperate() {
        let my = vec![Action::Cooperate; 10];
        let opp = vec![Action::Defect; 10];
        assert_eq!(identify_strategy(&my, &opp), "AlwaysCooperate");
    }

    #[test]
    fn test_identify_always_defect() {
        let my = vec![Action::Defect; 10];
        let opp = vec![Action::Cooperate; 10];
        assert_eq!(identify_strategy(&my, &opp), "AlwaysDefect");
    }

    #[test]
    fn test_play_sequence() {
        let gt = GrimTriggerStrategy;
        let opp = vec![Action::Cooperate, Action::Cooperate, Action::Defect, Action::Cooperate];
        let seq = gt.play_sequence(&opp);
        assert_eq!(seq.len(), 5);
        assert_eq!(seq[0], Action::Cooperate);
        assert_eq!(seq[3], Action::Cooperate);
        assert_eq!(seq[4], Action::Defect);
    }

    #[test]
    fn test_grim_trigger_finite_state() {
        let gt = GrimTriggerStrategy;
        let fs = gt.to_finite_state(4, 0.0, 10.0);
        // Verify initial state outputs high price
        assert_eq!(fs.output_action[0], DiscretizedPrice(3));
        // Verify punishment state outputs low price
        assert_eq!(fs.output_action[1], DiscretizedPrice(0));
    }

    #[test]
    fn test_always_defect_finite_state() {
        let ad = AlwaysDefect;
        let fs = ad.to_finite_state(4, 0.0, 10.0);
        assert_eq!(fs.num_states, 1);
        assert_eq!(fs.output_action[0], DiscretizedPrice(0));
    }
}
