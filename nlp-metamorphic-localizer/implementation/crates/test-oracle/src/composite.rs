//! Composite oracle combining multiple oracle strategies via voting.

use crate::threshold::OracleDecision;
use serde::{Deserialize, Serialize};

/// A vote from a single oracle in the composite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleVote {
    pub oracle_name: String,
    pub decision: OracleDecision,
    pub weight: f64,
}

/// Strategy for combining votes from multiple oracles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Majority vote (> 50% must agree).
    Majority,
    /// Unanimous agreement required.
    Unanimous,
    /// Weighted majority using oracle weights.
    WeightedMajority,
    /// Any single oracle detecting a violation triggers it.
    AnyViolation,
    /// Minimum number of oracles must agree.
    MinimumQuorum { quorum: usize },
}

/// A composite oracle that combines decisions from multiple sub-oracles.
pub struct CompositeOracle {
    strategy: VotingStrategy,
    oracle_names: Vec<String>,
    weights: Vec<f64>,
    history: Vec<CompositeDecision>,
}

/// Result from a composite oracle decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeDecision {
    pub is_violation: bool,
    pub votes: Vec<OracleVote>,
    pub agreement_ratio: f64,
    pub weighted_score: f64,
    pub strategy: String,
    pub explanation: String,
}

impl CompositeOracle {
    pub fn new(strategy: VotingStrategy) -> Self {
        Self {
            strategy,
            oracle_names: Vec::new(),
            weights: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Register a named oracle with a weight.
    pub fn register_oracle(&mut self, name: impl Into<String>, weight: f64) {
        self.oracle_names.push(name.into());
        self.weights.push(weight);
    }

    /// Make a composite decision from individual oracle votes.
    pub fn decide(&mut self, votes: Vec<OracleVote>) -> CompositeDecision {
        let violation_count = votes.iter().filter(|v| v.decision.is_violation).count();
        let total = votes.len();

        let agreement_ratio = if total == 0 {
            0.0
        } else {
            let max_side = violation_count.max(total - violation_count);
            max_side as f64 / total as f64
        };

        let weighted_sum: f64 = votes
            .iter()
            .map(|v| {
                if v.decision.is_violation {
                    v.weight
                } else {
                    0.0
                }
            })
            .sum();
        let total_weight: f64 = votes.iter().map(|v| v.weight).sum();
        let weighted_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        let is_violation = match &self.strategy {
            VotingStrategy::Majority => violation_count > total / 2,
            VotingStrategy::Unanimous => violation_count == total,
            VotingStrategy::WeightedMajority => weighted_score > 0.5,
            VotingStrategy::AnyViolation => violation_count > 0,
            VotingStrategy::MinimumQuorum { quorum } => violation_count >= *quorum,
        };

        let strategy_name = match &self.strategy {
            VotingStrategy::Majority => "majority",
            VotingStrategy::Unanimous => "unanimous",
            VotingStrategy::WeightedMajority => "weighted_majority",
            VotingStrategy::AnyViolation => "any_violation",
            VotingStrategy::MinimumQuorum { quorum } => "minimum_quorum",
        };

        let explanation = format!(
            "{}/{} oracles detected violation (agreement: {:.1}%, weighted: {:.3})",
            violation_count,
            total,
            agreement_ratio * 100.0,
            weighted_score,
        );

        let decision = CompositeDecision {
            is_violation,
            votes,
            agreement_ratio,
            weighted_score,
            strategy: strategy_name.to_string(),
            explanation,
        };

        self.history.push(decision.clone());
        decision
    }

    /// Get the history of composite decisions.
    pub fn history(&self) -> &[CompositeDecision] {
        &self.history
    }

    /// Clear decision history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Compute the overall violation rate from history.
    pub fn violation_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let violations = self.history.iter().filter(|d| d.is_violation).count();
        violations as f64 / self.history.len() as f64
    }

    /// Compute per-oracle agreement rates.
    pub fn oracle_agreement_rates(&self) -> Vec<(String, f64)> {
        if self.history.is_empty() {
            return Vec::new();
        }

        let n = self.history.len() as f64;
        let mut rates = Vec::new();

        for (i, name) in self.oracle_names.iter().enumerate() {
            let agreed = self
                .history
                .iter()
                .filter(|d| {
                    d.votes
                        .get(i)
                        .map(|v| v.decision.is_violation == d.is_violation)
                        .unwrap_or(false)
                })
                .count();
            rates.push((name.clone(), agreed as f64 / n));
        }

        rates
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vote(name: &str, is_violation: bool, weight: f64) -> OracleVote {
        OracleVote {
            oracle_name: name.to_string(),
            decision: OracleDecision {
                is_violation,
                distance: if is_violation { 0.8 } else { 0.2 },
                threshold: 0.5,
                confidence: 0.9,
                explanation: String::new(),
            },
            weight,
        }
    }

    #[test]
    fn test_majority_voting() {
        let mut oracle = CompositeOracle::new(VotingStrategy::Majority);
        let decision = oracle.decide(vec![
            make_vote("a", true, 1.0),
            make_vote("b", true, 1.0),
            make_vote("c", false, 1.0),
        ]);
        assert!(decision.is_violation); // 2/3 say violation
    }

    #[test]
    fn test_majority_no_violation() {
        let mut oracle = CompositeOracle::new(VotingStrategy::Majority);
        let decision = oracle.decide(vec![
            make_vote("a", false, 1.0),
            make_vote("b", true, 1.0),
            make_vote("c", false, 1.0),
        ]);
        assert!(!decision.is_violation); // 1/3 say violation
    }

    #[test]
    fn test_unanimous_voting() {
        let mut oracle = CompositeOracle::new(VotingStrategy::Unanimous);

        let d1 = oracle.decide(vec![
            make_vote("a", true, 1.0),
            make_vote("b", true, 1.0),
        ]);
        assert!(d1.is_violation);

        let d2 = oracle.decide(vec![
            make_vote("a", true, 1.0),
            make_vote("b", false, 1.0),
        ]);
        assert!(!d2.is_violation);
    }

    #[test]
    fn test_weighted_majority() {
        let mut oracle = CompositeOracle::new(VotingStrategy::WeightedMajority);

        // Heavy-weighted oracle says violation, two light ones say no.
        let decision = oracle.decide(vec![
            make_vote("a", true, 5.0),
            make_vote("b", false, 1.0),
            make_vote("c", false, 1.0),
        ]);
        assert!(decision.is_violation); // 5/(5+1+1) = 0.714 > 0.5
    }

    #[test]
    fn test_any_violation() {
        let mut oracle = CompositeOracle::new(VotingStrategy::AnyViolation);

        let d1 = oracle.decide(vec![
            make_vote("a", false, 1.0),
            make_vote("b", true, 1.0),
            make_vote("c", false, 1.0),
        ]);
        assert!(d1.is_violation);

        let d2 = oracle.decide(vec![
            make_vote("a", false, 1.0),
            make_vote("b", false, 1.0),
        ]);
        assert!(!d2.is_violation);
    }

    #[test]
    fn test_quorum_voting() {
        let mut oracle = CompositeOracle::new(VotingStrategy::MinimumQuorum { quorum: 2 });

        let d1 = oracle.decide(vec![
            make_vote("a", true, 1.0),
            make_vote("b", true, 1.0),
            make_vote("c", false, 1.0),
        ]);
        assert!(d1.is_violation);

        let d2 = oracle.decide(vec![
            make_vote("a", true, 1.0),
            make_vote("b", false, 1.0),
            make_vote("c", false, 1.0),
        ]);
        assert!(!d2.is_violation);
    }

    #[test]
    fn test_violation_rate() {
        let mut oracle = CompositeOracle::new(VotingStrategy::Majority);
        oracle.decide(vec![make_vote("a", true, 1.0)]);
        oracle.decide(vec![make_vote("a", false, 1.0)]);
        oracle.decide(vec![make_vote("a", true, 1.0)]);

        assert!((oracle.violation_rate() - 2.0 / 3.0).abs() < 0.01);
    }
}
