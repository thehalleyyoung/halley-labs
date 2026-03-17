//! Coverage tracking and goal management for test generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A coverage goal specifying target counts per transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageGoal {
    pub targets: HashMap<String, usize>,
    pub min_total: usize,
    pub min_per_transformation: usize,
}

impl CoverageGoal {
    /// Create a uniform coverage goal: same count for all transformations.
    pub fn uniform(transformations: &[String], count_per_transformation: usize) -> Self {
        let targets = transformations
            .iter()
            .map(|t| (t.clone(), count_per_transformation))
            .collect();
        Self {
            targets,
            min_total: transformations.len() * count_per_transformation,
            min_per_transformation: count_per_transformation,
        }
    }

    /// Create a weighted coverage goal.
    pub fn weighted(weights: HashMap<String, usize>) -> Self {
        let min = weights.values().cloned().min().unwrap_or(0);
        let total = weights.values().sum();
        Self {
            targets: weights,
            min_total: total,
            min_per_transformation: min,
        }
    }
}

/// Per-transformation coverage data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationCoverage {
    pub transformation_name: String,
    pub current_count: usize,
    pub target_count: usize,
    pub is_satisfied: bool,
    pub deficit: usize,
}

/// Tracks coverage progress during test generation.
#[derive(Debug, Clone)]
pub struct CoverageTracker {
    goal: CoverageGoal,
    current: HashMap<String, usize>,
    total_generated: usize,
}

impl CoverageTracker {
    pub fn new(goal: CoverageGoal) -> Self {
        let current = goal.targets.keys().map(|k| (k.clone(), 0)).collect();
        Self {
            goal,
            current,
            total_generated: 0,
        }
    }

    /// Record that a test case was generated for a transformation.
    pub fn record(&mut self, transformation: &str) {
        *self.current.entry(transformation.to_string()).or_insert(0) += 1;
        self.total_generated += 1;
    }

    /// Check if all coverage goals are satisfied.
    pub fn is_satisfied(&self) -> bool {
        self.goal.targets.iter().all(|(t, &target)| {
            self.current.get(t).copied().unwrap_or(0) >= target
        })
    }

    /// Get the transformation with the largest coverage deficit.
    pub fn most_needed_transformation(&self) -> Option<String> {
        self.goal
            .targets
            .iter()
            .map(|(t, &target)| {
                let current = self.current.get(t).copied().unwrap_or(0);
                let deficit = target.saturating_sub(current);
                (t.clone(), deficit)
            })
            .filter(|(_, deficit)| *deficit > 0)
            .max_by_key(|(_, deficit)| *deficit)
            .map(|(t, _)| t)
    }

    /// Get coverage report for all transformations.
    pub fn report(&self) -> Vec<TransformationCoverage> {
        self.goal
            .targets
            .iter()
            .map(|(t, &target)| {
                let current = self.current.get(t).copied().unwrap_or(0);
                TransformationCoverage {
                    transformation_name: t.clone(),
                    current_count: current,
                    target_count: target,
                    is_satisfied: current >= target,
                    deficit: target.saturating_sub(current),
                }
            })
            .collect()
    }

    /// Get overall coverage ratio (0.0 to 1.0).
    pub fn coverage_ratio(&self) -> f64 {
        if self.goal.targets.is_empty() {
            return 1.0;
        }
        let satisfied = self
            .goal
            .targets
            .iter()
            .filter(|(t, &target)| self.current.get(*t).copied().unwrap_or(0) >= target)
            .count();
        satisfied as f64 / self.goal.targets.len() as f64
    }

    /// Get total generated count.
    pub fn total_generated(&self) -> usize {
        self.total_generated
    }

    /// Get remaining count needed to satisfy all goals.
    pub fn remaining(&self) -> usize {
        self.goal
            .targets
            .iter()
            .map(|(t, &target)| {
                target.saturating_sub(self.current.get(t).copied().unwrap_or(0))
            })
            .sum()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_goal() {
        let transforms = vec!["passivization".into(), "clefting".into()];
        let goal = CoverageGoal::uniform(&transforms, 10);
        assert_eq!(goal.targets.len(), 2);
        assert_eq!(goal.min_total, 20);
    }

    #[test]
    fn test_coverage_tracking() {
        let transforms = vec!["passivization".into(), "clefting".into()];
        let goal = CoverageGoal::uniform(&transforms, 2);
        let mut tracker = CoverageTracker::new(goal);

        assert!(!tracker.is_satisfied());
        assert_eq!(tracker.remaining(), 4);

        tracker.record("passivization");
        tracker.record("passivization");
        assert!(!tracker.is_satisfied());
        assert_eq!(tracker.remaining(), 2);

        tracker.record("clefting");
        tracker.record("clefting");
        assert!(tracker.is_satisfied());
        assert_eq!(tracker.remaining(), 0);
        assert!((tracker.coverage_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_most_needed() {
        let transforms = vec!["a".into(), "b".into()];
        let goal = CoverageGoal::uniform(&transforms, 5);
        let mut tracker = CoverageTracker::new(goal);

        tracker.record("a");
        tracker.record("a");
        tracker.record("a");

        assert_eq!(tracker.most_needed_transformation(), Some("b".to_string()));
    }

    #[test]
    fn test_coverage_report() {
        let transforms = vec!["passivization".into()];
        let goal = CoverageGoal::uniform(&transforms, 3);
        let mut tracker = CoverageTracker::new(goal);
        tracker.record("passivization");

        let report = tracker.report();
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].current_count, 1);
        assert_eq!(report[0].target_count, 3);
        assert!(!report[0].is_satisfied);
        assert_eq!(report[0].deficit, 2);
    }
}
