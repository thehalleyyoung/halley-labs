//! Strategy representation and manipulation.
//!
//! Encode compliance strategies as bit vectors (one bit per obligation:
//! satisfied or waived), decode from solver assignments, compute diffs,
//! and provide enumeration / sampling helpers.

use crate::CostVector;
use regsynth_types::{Cost, Id};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// ObligationEntry
// ---------------------------------------------------------------------------

/// Reference to an obligation within a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationEntry {
    pub obligation_id: Id,
    pub name: String,
    pub estimated_cost: Option<Cost>,
}

// ---------------------------------------------------------------------------
// ComplianceStrategy
// ---------------------------------------------------------------------------

/// A compliance strategy: a selection of obligations to satisfy (or waive)
/// together with its aggregate cost information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStrategy {
    pub id: Id,
    pub name: String,
    pub obligation_entries: Vec<ObligationEntry>,
    pub waived_obligations: Vec<Id>,
    pub total_cost: Cost,
    pub compliance_score: f64,
    pub risk_score: f64,
    pub cost_vector: CostVector,
}

impl ComplianceStrategy {
    pub fn new(name: impl Into<String>, entries: Vec<ObligationEntry>) -> Self {
        let total: f64 = entries
            .iter()
            .filter_map(|o| o.estimated_cost.as_ref())
            .map(|c| c.amount)
            .sum();
        Self {
            id: Id::new(),
            name: name.into(),
            obligation_entries: entries,
            waived_obligations: Vec::new(),
            total_cost: Cost {
                amount: total,
                currency: "USD".into(),
            },
            compliance_score: 1.0,
            risk_score: 0.0,
            cost_vector: CostVector::new(vec![total, 0.0, 0.0, 1.0]),
        }
    }

    pub fn with_cost_vector(mut self, cv: CostVector) -> Self {
        self.cost_vector = cv;
        self
    }

    pub fn with_waived(mut self, waived: Vec<Id>) -> Self {
        self.waived_obligations = waived;
        self
    }

    pub fn obligation_count(&self) -> usize {
        self.obligation_entries.len()
    }

    pub fn waived_count(&self) -> usize {
        self.waived_obligations.len()
    }
}

impl fmt::Display for ComplianceStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Strategy({}, {} obligations, cost={})",
            self.name,
            self.obligation_entries.len(),
            self.total_cost.amount
        )
    }
}

// ---------------------------------------------------------------------------
// StrategyBitVec
// ---------------------------------------------------------------------------

/// Bit-vector representation of a strategy.
///
/// Bit `i` = 1 means obligation `i` is *satisfied* (included in the
/// strategy); bit `i` = 0 means it is *waived* / not covered.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StrategyBitVec {
    bits: Vec<bool>,
}

impl StrategyBitVec {
    pub fn new(size: usize) -> Self {
        Self {
            bits: vec![false; size],
        }
    }

    pub fn from_bits(bits: Vec<bool>) -> Self {
        Self { bits }
    }

    /// Create from a list of active (true) indices.
    pub fn from_active(size: usize, active: &[usize]) -> Self {
        let mut bits = vec![false; size];
        for &idx in active {
            if idx < size {
                bits[idx] = true;
            }
        }
        Self { bits }
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    pub fn get(&self, idx: usize) -> bool {
        self.bits[idx]
    }

    pub fn set(&mut self, idx: usize, val: bool) {
        self.bits[idx] = val;
    }

    pub fn toggle(&mut self, idx: usize) {
        self.bits[idx] = !self.bits[idx];
    }

    /// Number of satisfied (true) obligations.
    pub fn count_satisfied(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    /// Number of waived (false) obligations.
    pub fn count_waived(&self) -> usize {
        self.bits.iter().filter(|&&b| !b).count()
    }

    /// Indices of satisfied obligations.
    pub fn satisfied_indices(&self) -> Vec<usize> {
        self.bits
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect()
    }

    /// Indices of waived obligations.
    pub fn waived_indices(&self) -> Vec<usize> {
        self.bits
            .iter()
            .enumerate()
            .filter(|(_, &b)| !b)
            .map(|(i, _)| i)
            .collect()
    }

    /// Decode a strategy from a solver variable assignment.
    ///
    /// `var_mapping` maps obligation index → solver variable id.
    /// `assignment` maps variable id → bool.
    pub fn from_solver_assignment(
        size: usize,
        var_mapping: &[u32],
        assignment: &std::collections::HashMap<u32, bool>,
    ) -> Self {
        let mut bits = vec![false; size];
        for (i, &var) in var_mapping.iter().enumerate() {
            if i < size {
                bits[i] = assignment.get(&var).copied().unwrap_or(false);
            }
        }
        Self { bits }
    }

    /// Compute the symmetric difference (XOR) between two strategies.
    pub fn diff(&self, other: &StrategyBitVec) -> StrategyDiff {
        assert_eq!(self.len(), other.len());
        let mut added = Vec::new();
        let mut removed = Vec::new();
        for i in 0..self.len() {
            match (self.bits[i], other.bits[i]) {
                (false, true) => added.push(i),
                (true, false) => removed.push(i),
                _ => {}
            }
        }
        StrategyDiff { added, removed }
    }

    /// Hamming distance to another strategy.
    pub fn hamming_distance(&self, other: &StrategyBitVec) -> usize {
        assert_eq!(self.len(), other.len());
        self.bits
            .iter()
            .zip(other.bits.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Union (OR) of two strategies.
    pub fn union(&self, other: &StrategyBitVec) -> StrategyBitVec {
        assert_eq!(self.len(), other.len());
        StrategyBitVec {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(&a, &b)| a || b)
                .collect(),
        }
    }

    /// Intersection (AND) of two strategies.
    pub fn intersection(&self, other: &StrategyBitVec) -> StrategyBitVec {
        assert_eq!(self.len(), other.len());
        StrategyBitVec {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(&a, &b)| a && b)
                .collect(),
        }
    }
}

impl fmt::Display for StrategyBitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &b in &self.bits {
            write!(f, "{}", if b { '1' } else { '0' })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StrategyDiff
// ---------------------------------------------------------------------------

/// Difference between two strategies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategyDiff {
    /// Obligations added (were waived, now satisfied).
    pub added: Vec<usize>,
    /// Obligations removed (were satisfied, now waived).
    pub removed: Vec<usize>,
}

impl StrategyDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    pub fn total_changes(&self) -> usize {
        self.added.len() + self.removed.len()
    }
}

// ---------------------------------------------------------------------------
// Strategy enumeration / sampling
// ---------------------------------------------------------------------------

/// Enumerate all 2^n strategies for a problem of size `n`.
///
/// Only practical for very small instances (n ≤ ~20).
pub fn enumerate_strategies(n: usize) -> Vec<StrategyBitVec> {
    assert!(n <= 24, "enumerate_strategies: n={} too large (max 24)", n);
    let total = 1u32 << n;
    (0..total)
        .map(|mask| {
            let bits: Vec<bool> = (0..n).map(|i| (mask >> i) & 1 == 1).collect();
            StrategyBitVec::from_bits(bits)
        })
        .collect()
}

/// Sample `count` random feasible strategies.
///
/// `feasibility_check` returns true if the strategy satisfies all hard
/// constraints. Retries up to `max_attempts` per sample.
pub fn random_feasible_strategies<F>(
    n: usize,
    count: usize,
    max_attempts: usize,
    mut feasibility_check: F,
) -> Vec<StrategyBitVec>
where
    F: FnMut(&StrategyBitVec) -> bool,
{
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut results = Vec::with_capacity(count);
    let mut attempts = 0;

    while results.len() < count && attempts < max_attempts {
        let bits: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
        let strategy = StrategyBitVec::from_bits(bits);
        if feasibility_check(&strategy) {
            results.push(strategy);
        }
        attempts += 1;
    }
    results
}

/// Greedy strategy construction: start from an empty strategy and
/// iteratively add the obligation that improves the weighted-sum cost
/// the most.
///
/// `cost_fn` evaluates the cost vector of a given strategy.
/// `weights` define the scalarization for greedy selection.
pub fn greedy_strategy<F>(
    n: usize,
    weights: &[f64],
    mut cost_fn: F,
) -> StrategyBitVec
where
    F: FnMut(&StrategyBitVec) -> CostVector,
{
    let mut current = StrategyBitVec::new(n);
    let mut current_cost = cost_fn(&current).weighted_sum(weights);

    loop {
        let mut best_idx = None;
        let mut best_cost = current_cost;

        for i in 0..n {
            if current.get(i) {
                continue;
            }
            let mut candidate = current.clone();
            candidate.set(i, true);
            let candidate_cost = cost_fn(&candidate).weighted_sum(weights);
            if candidate_cost < best_cost {
                best_cost = candidate_cost;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(idx) => {
                current.set(idx, true);
                current_cost = best_cost;
            }
            None => break,
        }
    }

    current
}

/// Greedy strategy starting from all obligations satisfied and greedily
/// removing the obligation whose removal improves cost the most.
pub fn greedy_strategy_removal<F>(
    n: usize,
    weights: &[f64],
    mut cost_fn: F,
) -> StrategyBitVec
where
    F: FnMut(&StrategyBitVec) -> CostVector,
{
    let mut current = StrategyBitVec::from_bits(vec![true; n]);
    let mut current_cost = cost_fn(&current).weighted_sum(weights);

    loop {
        let mut best_idx = None;
        let mut best_cost = current_cost;

        for i in 0..n {
            if !current.get(i) {
                continue;
            }
            let mut candidate = current.clone();
            candidate.set(i, false);
            let candidate_cost = cost_fn(&candidate).weighted_sum(weights);
            if candidate_cost < best_cost {
                best_cost = candidate_cost;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(idx) => {
                current.set(idx, false);
                current_cost = best_cost;
            }
            None => break,
        }
    }

    current
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitvec_basic() {
        let mut s = StrategyBitVec::new(5);
        assert_eq!(s.count_satisfied(), 0);
        s.set(1, true);
        s.set(3, true);
        assert_eq!(s.count_satisfied(), 2);
        assert_eq!(s.satisfied_indices(), vec![1, 3]);
    }

    #[test]
    fn test_bitvec_from_active() {
        let s = StrategyBitVec::from_active(5, &[0, 2, 4]);
        assert!(s.get(0));
        assert!(!s.get(1));
        assert!(s.get(2));
        assert!(!s.get(3));
        assert!(s.get(4));
    }

    #[test]
    fn test_bitvec_toggle() {
        let mut s = StrategyBitVec::new(3);
        s.toggle(1);
        assert!(s.get(1));
        s.toggle(1);
        assert!(!s.get(1));
    }

    #[test]
    fn test_strategy_diff() {
        let a = StrategyBitVec::from_bits(vec![true, false, true, false]);
        let b = StrategyBitVec::from_bits(vec![true, true, false, false]);
        let diff = a.diff(&b);
        assert_eq!(diff.added, vec![1]);
        assert_eq!(diff.removed, vec![2]);
    }

    #[test]
    fn test_hamming_distance() {
        let a = StrategyBitVec::from_bits(vec![true, false, true]);
        let b = StrategyBitVec::from_bits(vec![false, false, true]);
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn test_union_intersection() {
        let a = StrategyBitVec::from_bits(vec![true, false, true]);
        let b = StrategyBitVec::from_bits(vec![false, true, true]);
        let u = a.union(&b);
        let i = a.intersection(&b);
        assert_eq!(u.bits, vec![true, true, true]);
        assert_eq!(i.bits, vec![false, false, true]);
    }

    #[test]
    fn test_enumerate_strategies() {
        let all = enumerate_strategies(3);
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_random_feasible() {
        let strategies = random_feasible_strategies(5, 3, 1000, |_| true);
        assert_eq!(strategies.len(), 3);
    }

    #[test]
    fn test_greedy_strategy() {
        // 3 obligations with costs [10, 5, 3]
        // Greedy with equal weights should add the one that reduces cost most
        let costs = vec![10.0, 5.0, 3.0];
        let result = greedy_strategy(3, &[1.0], |s| {
            let total: f64 = (0..3)
                .filter(|&i| s.get(i))
                .map(|i| costs[i])
                .sum();
            // Cost is -total (maximize coverage = minimize negative)
            CostVector::new(vec![-total])
        });
        // Should satisfy all three
        assert_eq!(result.count_satisfied(), 3);
    }

    #[test]
    fn test_greedy_removal() {
        let costs = vec![10.0, 5.0, 3.0];
        let result = greedy_strategy_removal(3, &[1.0], |s| {
            let total: f64 = (0..3)
                .filter(|&i| s.get(i))
                .map(|i| costs[i])
                .sum();
            CostVector::new(vec![total])
        });
        // Greedy removal to minimize cost: should remove all
        assert_eq!(result.count_satisfied(), 0);
    }

    #[test]
    fn test_display_bitvec() {
        let s = StrategyBitVec::from_bits(vec![true, false, true, true]);
        assert_eq!(format!("{}", s), "1011");
    }

    #[test]
    fn test_from_solver_assignment() {
        let mut assignment = std::collections::HashMap::new();
        assignment.insert(10u32, true);
        assignment.insert(11u32, false);
        assignment.insert(12u32, true);
        let s = StrategyBitVec::from_solver_assignment(3, &[10, 11, 12], &assignment);
        assert!(s.get(0));
        assert!(!s.get(1));
        assert!(s.get(2));
    }

    #[test]
    fn test_compliance_strategy_construction() {
        let entry = ObligationEntry {
            obligation_id: Id::new(),
            name: "GDPR Art.5".into(),
            estimated_cost: Some(Cost {
                amount: 50000.0,
                currency: "EUR".into(),
            }),
        };
        let strat = ComplianceStrategy::new("Basic", vec![entry]);
        assert_eq!(strat.obligation_count(), 1);
        assert!((strat.total_cost.amount - 50000.0).abs() < 1e-10);
    }
}
