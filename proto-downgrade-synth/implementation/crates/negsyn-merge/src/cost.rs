//! Merge cost estimation and adaptive merge policies.
//!
//! Provides cost estimation for protocol-aware merges, cost-benefit analysis,
//! and an adaptive merge policy that tunes thresholds based on historical data.

use std::collections::VecDeque;
use std::fmt;

use serde::{Deserialize, Serialize};

use negsyn_types::{MergeConfig, SymbolicState};

// ---------------------------------------------------------------------------
// Merge cost
// ---------------------------------------------------------------------------

/// Quantified cost of a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeCost {
    pub constraint_complexity: f64,
    pub memory_overlap: f64,
    pub register_divergence: f64,
    pub ite_depth_penalty: f64,
    pub cipher_conflict_cost: f64,
    pub extension_conflict_cost: f64,
}

impl MergeCost {
    /// Aggregate scalar score (lower is cheaper).
    pub fn score(&self) -> f64 {
        self.constraint_complexity
            + self.memory_overlap
            + self.register_divergence
            + self.ite_depth_penalty
            + self.cipher_conflict_cost
            + self.extension_conflict_cost
    }

    /// Does this cost exceed the given budget?
    pub fn exceeds_budget(&self, budget: f64) -> bool {
        self.score() > budget
    }

    /// Combine two costs (e.g. for chained merges).
    pub fn add(&self, other: &MergeCost) -> MergeCost {
        MergeCost {
            constraint_complexity: self.constraint_complexity + other.constraint_complexity,
            memory_overlap: self.memory_overlap + other.memory_overlap,
            register_divergence: self.register_divergence + other.register_divergence,
            ite_depth_penalty: self.ite_depth_penalty + other.ite_depth_penalty,
            cipher_conflict_cost: self.cipher_conflict_cost + other.cipher_conflict_cost,
            extension_conflict_cost: self.extension_conflict_cost + other.extension_conflict_cost,
        }
    }

    /// Zero cost.
    pub fn zero() -> Self {
        Self {
            constraint_complexity: 0.0,
            memory_overlap: 0.0,
            register_divergence: 0.0,
            ite_depth_penalty: 0.0,
            cipher_conflict_cost: 0.0,
            extension_conflict_cost: 0.0,
        }
    }
}

impl fmt::Display for MergeCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MergeCost(score={:.2})", self.score())
    }
}

impl Default for MergeCost {
    fn default() -> Self {
        Self::zero()
    }
}

// ---------------------------------------------------------------------------
// Cost estimator
// ---------------------------------------------------------------------------

/// Estimates the cost of merging two symbolic states.
pub struct CostEstimator {
    config: MergeConfig,
    constraint_weight: f64,
    memory_weight: f64,
    register_weight: f64,
    ite_depth_weight: f64,
    cipher_weight: f64,
    extension_weight: f64,
}

impl CostEstimator {
    pub fn new(config: MergeConfig) -> Self {
        Self {
            config,
            constraint_weight: 1.0,
            memory_weight: 0.5,
            register_weight: 0.3,
            ite_depth_weight: 2.0,
            cipher_weight: 1.5,
            extension_weight: 0.8,
        }
    }

    /// Estimate merge cost for a pair of `SymbolicState` values.
    pub fn estimate(&self, left: &SymbolicState, right: &SymbolicState) -> MergeCost {
        let constraint_complexity = (left.constraints.len() + right.constraints.len()) as f64
            * self.constraint_weight;

        let left_ciphers: std::collections::BTreeSet<u16> = left.negotiation.offered_ciphers.iter().map(|c| c.iana_id).collect();
        let right_ciphers: std::collections::BTreeSet<u16> = right.negotiation.offered_ciphers.iter().map(|c| c.iana_id).collect();
        let cipher_conflict = if left_ciphers == right_ciphers {
            0.0
        } else {
            let union_size = left_ciphers.union(&right_ciphers).count();
            let intersection_size = left_ciphers.intersection(&right_ciphers).count();
            if union_size == 0 {
                0.0
            } else {
                (1.0 - intersection_size as f64 / union_size as f64) * self.cipher_weight
            }
        };

        let total_constraint_nodes: usize = left.constraints
            .iter()
            .chain(right.constraints.iter())
            .map(|c| c.node_count())
            .sum();

        let max_ite_depth = self.config.max_ite_depth as usize;
        let ite_depth_penalty = if total_constraint_nodes > max_ite_depth {
            (total_constraint_nodes - max_ite_depth) as f64 * self.ite_depth_weight
        } else {
            0.0
        };

        MergeCost {
            constraint_complexity,
            memory_overlap: 0.0,
            register_divergence: 0.0,
            ite_depth_penalty,
            cipher_conflict_cost: cipher_conflict,
            extension_conflict_cost: 0.0,
        }
    }

    /// More detailed cost estimate on concrete `SymbolicState` values.
    pub fn estimate_symbolic(&self, left: &SymbolicState, right: &SymbolicState) -> MergeCost {
        let constraint_complexity = (left.constraints.len() + right.constraints.len()) as f64
            * self.constraint_weight;

        let left_regions: std::collections::BTreeSet<String> = left.memory.keys().into_iter().map(|s| s.to_string()).collect();
        let right_regions: std::collections::BTreeSet<String> = right.memory.keys().into_iter().map(|s| s.to_string()).collect();
        let shared_regions = left_regions.intersection(&right_regions).count();
        let memory_overlap = shared_regions as f64 * self.memory_weight;

        let mut divergent = 0usize;
        let all_regs: std::collections::BTreeSet<&String> =
            left.registers.keys().chain(right.registers.keys()).collect();
        for reg in &all_regs {
            match (left.registers.get(*reg), right.registers.get(*reg)) {
                (Some(lv), Some(rv)) if lv != rv => divergent += 1,
                (Some(_), None) | (None, Some(_)) => divergent += 1,
                _ => {}
            }
        }
        let register_divergence = divergent as f64 * self.register_weight;

        let total_nodes: usize = left
            .constraints
            .iter()
            .chain(right.constraints.iter())
            .map(|c| c.node_count())
            .sum();
        let max_ite_depth = self.config.max_ite_depth as usize;
        let ite_depth_penalty = if total_nodes > max_ite_depth {
            (total_nodes - max_ite_depth) as f64 * self.ite_depth_weight
        } else {
            0.0
        };

        let cipher_conflict =
            if left.negotiation.selected_cipher != right.negotiation.selected_cipher {
                self.cipher_weight
            } else {
                0.0
            };

        let left_ext_ids: std::collections::BTreeSet<u16> =
            left.negotiation.extensions.iter().map(|e| e.id).collect();
        let right_ext_ids: std::collections::BTreeSet<u16> =
            right.negotiation.extensions.iter().map(|e| e.id).collect();
        let ext_diff = left_ext_ids.symmetric_difference(&right_ext_ids).count();
        let extension_conflict_cost = ext_diff as f64 * self.extension_weight;

        MergeCost {
            constraint_complexity,
            memory_overlap,
            register_divergence,
            ite_depth_penalty,
            cipher_conflict_cost: cipher_conflict,
            extension_conflict_cost,
        }
    }
}

// ---------------------------------------------------------------------------
// Cost-benefit analysis
// ---------------------------------------------------------------------------

/// Evaluates whether merging is beneficial by comparing cost to expected gain.
pub struct CostBenefitAnalysis {
    pub state_reduction: f64,
    pub solver_savings: f64,
    pub merge_cost: MergeCost,
}

impl CostBenefitAnalysis {
    pub fn new(merge_cost: MergeCost, state_reduction: f64, solver_savings: f64) -> Self {
        Self {
            state_reduction,
            solver_savings,
            merge_cost,
        }
    }

    /// Benefit-to-cost ratio. Values > 1.0 indicate a worthwhile merge.
    pub fn compute_benefit_ratio(&self) -> f64 {
        let cost = self.merge_cost.score();
        if cost <= f64::EPSILON {
            return f64::MAX;
        }
        (self.state_reduction + self.solver_savings) / cost
    }

    /// Whether the analysis recommends performing the merge.
    pub fn is_beneficial(&self) -> bool {
        self.compute_benefit_ratio() > 1.0
    }

    /// Fraction of the benefit from protocol-specific advantages.
    pub fn protocol_advantage_ratio(&self) -> f64 {
        let total = self.state_reduction + self.solver_savings;
        if total <= f64::EPSILON {
            return 0.0;
        }
        self.solver_savings / total
    }
}

// ---------------------------------------------------------------------------
// Adaptive merge policy
// ---------------------------------------------------------------------------

const HISTORY_WINDOW: usize = 64;

/// Adaptive policy that tunes merge thresholds based on recent outcomes.
pub struct AdaptiveMergePolicy {
    config: MergeConfig,
    cost_estimator: CostEstimator,
    budget: f64,
    history: VecDeque<HistoricalMerge>,
}

#[derive(Debug, Clone)]
struct HistoricalMerge {
    cost: MergeCost,
    success: bool,
}

impl AdaptiveMergePolicy {
    pub fn new(config: MergeConfig) -> Self {
        let cost_estimator = CostEstimator::new(config.clone());
        Self {
            config,
            cost_estimator,
            budget: 100.0,
            history: VecDeque::with_capacity(HISTORY_WINDOW),
        }
    }

    /// Should we proceed with the merge of `left` and `right`?
    pub fn should_merge(&self, left: &SymbolicState, right: &SymbolicState) -> bool {
        let cost = self.cost_estimator.estimate(left, right);
        !cost.exceeds_budget(self.budget)
    }

    /// Record an observed merge outcome to refine future decisions.
    pub fn track_historical_cost(&mut self, cost: MergeCost, success: bool) {
        if self.history.len() >= HISTORY_WINDOW {
            self.history.pop_front();
        }
        self.history.push_back(HistoricalMerge { cost, success });
        self.adapt_budget();
    }

    pub fn current_budget(&self) -> f64 {
        self.budget
    }

    pub fn recent_success_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let successes = self.history.iter().filter(|h| h.success).count();
        successes as f64 / self.history.len() as f64
    }

    fn adapt_budget(&mut self) {
        let rate = self.recent_success_rate();
        if rate > 0.8 {
            self.budget *= 1.1;
        } else if rate < 0.4 {
            self.budget *= 0.8;
        }
        self.budget = self.budget.clamp(10.0, 10_000.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use negsyn_types::{
        HandshakePhase, NegotiationState, PathConstraint, ProtocolVersion, SymbolicValue,
    };

    fn make_test_state(id: u64, phase: HandshakePhase, ciphers: &[u16]) -> SymbolicState {
        let mut neg = NegotiationState::new(phase, ProtocolVersion::Tls12);
        neg.offered_ciphers = ciphers.iter().copied().collect();
        SymbolicState::new(id, 0x1000, neg)
    }

    #[test]
    fn test_merge_cost_score() {
        let cost = MergeCost {
            constraint_complexity: 2.0,
            memory_overlap: 1.0,
            register_divergence: 0.5,
            ite_depth_penalty: 0.0,
            cipher_conflict_cost: 1.5,
            extension_conflict_cost: 0.0,
        };
        assert!((cost.score() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_cost_exceeds_budget() {
        let cost = MergeCost {
            constraint_complexity: 10.0,
            ..MergeCost::zero()
        };
        assert!(cost.exceeds_budget(5.0));
        assert!(!cost.exceeds_budget(15.0));
    }

    #[test]
    fn test_merge_cost_add() {
        let a = MergeCost {
            constraint_complexity: 1.0,
            memory_overlap: 2.0,
            ..MergeCost::zero()
        };
        let b = MergeCost {
            constraint_complexity: 3.0,
            register_divergence: 4.0,
            ..MergeCost::zero()
        };
        let sum = a.add(&b);
        assert!((sum.constraint_complexity - 4.0).abs() < f64::EPSILON);
        assert!((sum.memory_overlap - 2.0).abs() < f64::EPSILON);
        assert!((sum.register_divergence - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_cost_zero() {
        let z = MergeCost::zero();
        assert!((z.score()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_cost_display() {
        let cost = MergeCost::zero();
        let s = format!("{}", cost);
        assert!(s.contains("MergeCost"));
    }

    #[test]
    fn test_estimator_identical_states() {
        let config = MergeConfig::default();
        let est = CostEstimator::new(config);
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let cost = est.estimate(&s1, &s2);
        assert!((cost.cipher_conflict_cost).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimator_symbolic_registers() {
        let config = MergeConfig::default();
        let est = CostEstimator::new(config);
        let mut s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let mut s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        s1.registers
            .insert("rax".to_string(), SymbolicValue::Concrete(1));
        s2.registers
            .insert("rax".to_string(), SymbolicValue::Concrete(2));
        let cost = est.estimate_symbolic(&s1, &s2);
        assert!(cost.register_divergence > 0.0);
    }

    #[test]
    fn test_estimator_constraint_complexity() {
        let config = MergeConfig::default();
        let est = CostEstimator::new(config);
        let mut s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        s1.constraints
            .push(PathConstraint::new(SymbolicValue::BoolConst(true)));
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let cost = est.estimate_symbolic(&s1, &s2);
        assert!(cost.constraint_complexity > 0.0);
    }

    #[test]
    fn test_cba_beneficial() {
        let cost = MergeCost {
            constraint_complexity: 1.0,
            ..MergeCost::zero()
        };
        let cba = CostBenefitAnalysis::new(cost, 5.0, 3.0);
        assert!(cba.is_beneficial());
        assert!(cba.compute_benefit_ratio() > 1.0);
    }

    #[test]
    fn test_cba_not_beneficial() {
        let cost = MergeCost {
            constraint_complexity: 100.0,
            ..MergeCost::zero()
        };
        let cba = CostBenefitAnalysis::new(cost, 1.0, 0.5);
        assert!(!cba.is_beneficial());
    }

    #[test]
    fn test_cba_protocol_advantage() {
        let cost = MergeCost::zero();
        let cba = CostBenefitAnalysis::new(cost, 2.0, 8.0);
        assert!((cba.protocol_advantage_ratio() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cba_zero_cost() {
        let cost = MergeCost::zero();
        let cba = CostBenefitAnalysis::new(cost, 5.0, 5.0);
        assert_eq!(cba.compute_benefit_ratio(), f64::MAX);
    }

    #[test]
    fn test_adaptive_initial_should_merge() {
        let config = MergeConfig::default();
        let policy = AdaptiveMergePolicy::new(config);
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        assert!(policy.should_merge(&s1, &s2));
    }

    #[test]
    fn test_adaptive_budget_increases_on_success() {
        let config = MergeConfig::default();
        let mut policy = AdaptiveMergePolicy::new(config);
        let initial = policy.current_budget();
        for _ in 0..10 {
            policy.track_historical_cost(MergeCost::zero(), true);
        }
        assert!(policy.current_budget() > initial);
    }

    #[test]
    fn test_adaptive_budget_decreases_on_failure() {
        let config = MergeConfig::default();
        let mut policy = AdaptiveMergePolicy::new(config);
        let initial = policy.current_budget();
        for _ in 0..10 {
            policy.track_historical_cost(MergeCost::zero(), false);
        }
        assert!(policy.current_budget() < initial);
    }

    #[test]
    fn test_adaptive_recent_success_rate() {
        let config = MergeConfig::default();
        let mut policy = AdaptiveMergePolicy::new(config);
        policy.track_historical_cost(MergeCost::zero(), true);
        policy.track_historical_cost(MergeCost::zero(), false);
        assert!((policy.recent_success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_empty_history() {
        let config = MergeConfig::default();
        let policy = AdaptiveMergePolicy::new(config);
        assert!((policy.recent_success_rate() - 1.0).abs() < f64::EPSILON);
    }
}
