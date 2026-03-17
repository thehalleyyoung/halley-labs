//! Aggregated mutation-testing statistics and reports.

use shared_types::{MutantDescriptor, MutantStatus, MutationOperator};
use std::collections::HashMap;

/// Aggregated statistics for a mutation testing run.
#[derive(Debug, Clone, Default)]
pub struct MutationStatistics {
    pub total_mutants: usize,
    pub killed: usize,
    pub alive: usize,
    pub equivalent: usize,
    pub timeout: usize,
    pub errors: usize,
    pub by_operator: HashMap<String, OperatorStats>,
}

/// Per-operator statistics.
#[derive(Debug, Clone, Default)]
pub struct OperatorStats {
    pub total: usize,
    pub killed: usize,
    pub alive: usize,
    pub equivalent: usize,
}

impl MutationStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute statistics from a collection of mutant descriptors.
    pub fn from_descriptors(descriptors: &[MutantDescriptor]) -> Self {
        let mut stats = Self::new();
        stats.total_mutants = descriptors.len();
        for d in descriptors {
            match &d.status {
                MutantStatus::Killed => stats.killed += 1,
                MutantStatus::Alive => stats.alive += 1,
                MutantStatus::Equivalent => stats.equivalent += 1,
                MutantStatus::Timeout => stats.timeout += 1,
                MutantStatus::Error(_) => stats.errors += 1,
            }
            let op_name = d.operator.to_string();
            let entry = stats.by_operator.entry(op_name).or_default();
            entry.total += 1;
            match &d.status {
                MutantStatus::Killed => entry.killed += 1,
                MutantStatus::Alive => entry.alive += 1,
                MutantStatus::Equivalent => entry.equivalent += 1,
                _ => {}
            }
        }
        stats
    }

    /// Mutation score: killed / (total - equivalent).
    pub fn mutation_score(&self) -> f64 {
        let denominator = self.total_mutants - self.equivalent;
        if denominator == 0 {
            1.0
        } else {
            self.killed as f64 / denominator as f64
        }
    }
}
