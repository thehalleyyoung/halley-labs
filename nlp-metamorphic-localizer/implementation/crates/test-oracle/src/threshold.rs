//! Threshold-based oracle for metamorphic relation checking.
//!
//! Determines whether a metamorphic relation holds by comparing a distance
//! metric against a configurable threshold.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for threshold-based oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub initial_threshold: f64,
    pub min_threshold: f64,
    pub max_threshold: f64,
    pub adaptation_rate: f64,
    pub window_size: usize,
    pub significance_level: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            initial_threshold: 0.5,
            min_threshold: 0.1,
            max_threshold: 0.95,
            adaptation_rate: 0.05,
            window_size: 100,
            significance_level: 0.05,
        }
    }
}

/// A threshold-based oracle that checks if distance exceeds a threshold.
#[derive(Debug, Clone)]
pub struct ThresholdOracle {
    config: ThresholdConfig,
    current_threshold: f64,
    history: Vec<f64>,
    violation_count: usize,
    total_checks: usize,
}

impl ThresholdOracle {
    pub fn new(config: ThresholdConfig) -> Self {
        let threshold = config.initial_threshold;
        Self {
            config,
            current_threshold: threshold,
            history: Vec::new(),
            violation_count: 0,
            total_checks: 0,
        }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self::new(ThresholdConfig {
            initial_threshold: threshold,
            ..Default::default()
        })
    }

    /// Check if the given distance constitutes a violation.
    pub fn check(&mut self, distance: f64) -> OracleDecision {
        self.total_checks += 1;
        self.history.push(distance);

        let is_violation = distance > self.current_threshold;
        if is_violation {
            self.violation_count += 1;
        }

        let confidence = self.compute_confidence(distance);

        OracleDecision {
            is_violation,
            distance,
            threshold: self.current_threshold,
            confidence,
            explanation: if is_violation {
                format!(
                    "Distance {:.4} exceeds threshold {:.4} (confidence: {:.2}%)",
                    distance,
                    self.current_threshold,
                    confidence * 100.0
                )
            } else {
                format!(
                    "Distance {:.4} within threshold {:.4}",
                    distance, self.current_threshold
                )
            },
        }
    }

    /// Compute confidence based on how far the distance is from the threshold.
    fn compute_confidence(&self, distance: f64) -> f64 {
        if self.history.len() < 5 {
            return 0.5;
        }

        let mean: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let std_dev: f64 = {
            let variance: f64 = self
                .history
                .iter()
                .map(|d| (d - mean).powi(2))
                .sum::<f64>()
                / self.history.len() as f64;
            variance.sqrt()
        };

        if std_dev < f64::EPSILON {
            return if (distance - self.current_threshold).abs() < f64::EPSILON {
                0.5
            } else if distance > self.current_threshold {
                0.95
            } else {
                0.95
            };
        }

        // Z-score relative to threshold.
        let z = (distance - self.current_threshold) / std_dev;
        // Sigmoid to map to [0, 1].
        1.0 / (1.0 + (-z * 2.0).exp())
    }

    /// Get the current violation rate.
    pub fn violation_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.violation_count as f64 / self.total_checks as f64
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> f64 {
        self.current_threshold
    }

    /// Manually set the threshold.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.current_threshold = threshold
            .max(self.config.min_threshold)
            .min(self.config.max_threshold);
    }

    /// Get statistics about the oracle's history.
    pub fn statistics(&self) -> OracleStatistics {
        let n = self.history.len();
        if n == 0 {
            return OracleStatistics {
                total_checks: 0,
                violations: 0,
                violation_rate: 0.0,
                mean_distance: 0.0,
                std_distance: 0.0,
                min_distance: 0.0,
                max_distance: 0.0,
                current_threshold: self.current_threshold,
            };
        }

        let mean = self.history.iter().sum::<f64>() / n as f64;
        let variance: f64 = self.history.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
        let min = self.history.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        OracleStatistics {
            total_checks: self.total_checks,
            violations: self.violation_count,
            violation_rate: self.violation_rate(),
            mean_distance: mean,
            std_distance: variance.sqrt(),
            min_distance: min,
            max_distance: max,
            current_threshold: self.current_threshold,
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.violation_count = 0;
        self.total_checks = 0;
        self.current_threshold = self.config.initial_threshold;
    }
}

/// An adaptive threshold that adjusts based on observed distances.
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    config: ThresholdConfig,
    current_threshold: f64,
    window: VecDeque<f64>,
    violation_window: VecDeque<bool>,
}

impl AdaptiveThreshold {
    pub fn new(config: ThresholdConfig) -> Self {
        let threshold = config.initial_threshold;
        let window_size = config.window_size;
        Self {
            config,
            current_threshold: threshold,
            window: VecDeque::with_capacity(window_size),
            violation_window: VecDeque::with_capacity(window_size),
        }
    }

    /// Update the adaptive threshold with a new observation and return a decision.
    pub fn observe(&mut self, distance: f64) -> OracleDecision {
        // Add to window.
        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
            self.violation_window.pop_front();
        }

        let is_violation = distance > self.current_threshold;
        self.window.push_back(distance);
        self.violation_window.push_back(is_violation);

        // Adapt threshold based on the window.
        self.adapt();

        let confidence = self.compute_adaptive_confidence(distance);

        OracleDecision {
            is_violation,
            distance,
            threshold: self.current_threshold,
            confidence,
            explanation: format!(
                "Adaptive threshold: {:.4}, distance: {:.4}, window_size: {}",
                self.current_threshold,
                distance,
                self.window.len()
            ),
        }
    }

    /// Adapt the threshold based on the current window.
    fn adapt(&mut self) {
        if self.window.len() < 10 {
            return;
        }

        let mean: f64 = self.window.iter().sum::<f64>() / self.window.len() as f64;
        let std_dev: f64 = {
            let variance: f64 = self
                .window
                .iter()
                .map(|d| (d - mean).powi(2))
                .sum::<f64>()
                / self.window.len() as f64;
            variance.sqrt()
        };

        // Set threshold at mean + 2*std_dev (approximately 95th percentile for normal).
        let target = mean + 2.0 * std_dev;

        // Smooth adaptation.
        let rate = self.config.adaptation_rate;
        self.current_threshold = (1.0 - rate) * self.current_threshold + rate * target;
        self.current_threshold = self
            .current_threshold
            .max(self.config.min_threshold)
            .min(self.config.max_threshold);
    }

    /// Compute confidence using the adaptive window.
    fn compute_adaptive_confidence(&self, distance: f64) -> f64 {
        if self.window.len() < 10 {
            return 0.5;
        }

        let mean: f64 = self.window.iter().sum::<f64>() / self.window.len() as f64;
        let std_dev: f64 = {
            let variance: f64 = self
                .window
                .iter()
                .map(|d| (d - mean).powi(2))
                .sum::<f64>()
                / self.window.len() as f64;
            variance.sqrt()
        };

        if std_dev < f64::EPSILON {
            return 0.5;
        }

        let z = (distance - self.current_threshold) / std_dev;
        1.0 / (1.0 + (-z * 2.0).exp())
    }

    pub fn threshold(&self) -> f64 {
        self.current_threshold
    }

    pub fn window_size(&self) -> usize {
        self.window.len()
    }
}

/// A decision made by an oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleDecision {
    pub is_violation: bool,
    pub distance: f64,
    pub threshold: f64,
    pub confidence: f64,
    pub explanation: String,
}

/// Statistics about an oracle's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleStatistics {
    pub total_checks: usize,
    pub violations: usize,
    pub violation_rate: f64,
    pub mean_distance: f64,
    pub std_distance: f64,
    pub min_distance: f64,
    pub max_distance: f64,
    pub current_threshold: f64,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_oracle_basic() {
        let mut oracle = ThresholdOracle::with_threshold(0.5);

        let decision = oracle.check(0.3);
        assert!(!decision.is_violation);

        let decision = oracle.check(0.7);
        assert!(decision.is_violation);
    }

    #[test]
    fn test_threshold_oracle_violation_rate() {
        let mut oracle = ThresholdOracle::with_threshold(0.5);

        oracle.check(0.3);
        oracle.check(0.7);
        oracle.check(0.4);
        oracle.check(0.8);

        assert!((oracle.violation_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_threshold_oracle_statistics() {
        let mut oracle = ThresholdOracle::with_threshold(0.5);

        for i in 0..10 {
            oracle.check(i as f64 * 0.1);
        }

        let stats = oracle.statistics();
        assert_eq!(stats.total_checks, 10);
        assert!(stats.mean_distance > 0.0);
        assert!(stats.min_distance < 0.01);
        assert!(stats.max_distance > 0.8);
    }

    #[test]
    fn test_threshold_oracle_reset() {
        let mut oracle = ThresholdOracle::with_threshold(0.5);
        oracle.check(0.3);
        oracle.check(0.7);
        oracle.reset();

        let stats = oracle.statistics();
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.violations, 0);
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = ThresholdConfig {
            initial_threshold: 0.5,
            adaptation_rate: 0.1,
            window_size: 50,
            ..Default::default()
        };
        let mut adaptive = AdaptiveThreshold::new(config);

        // Feed mostly low distances.
        for _ in 0..30 {
            adaptive.observe(0.1);
        }

        // Threshold should have adapted down.
        assert!(adaptive.threshold() < 0.5);

        // Now a larger distance should be flagged.
        let decision = adaptive.observe(0.4);
        assert!(decision.is_violation || adaptive.threshold() < 0.4);
    }

    #[test]
    fn test_adaptive_threshold_window() {
        let config = ThresholdConfig {
            window_size: 10,
            ..Default::default()
        };
        let mut adaptive = AdaptiveThreshold::new(config);

        for i in 0..20 {
            adaptive.observe(i as f64 * 0.05);
        }

        assert_eq!(adaptive.window_size(), 10);
    }

    #[test]
    fn test_confidence_increases_with_history() {
        let mut oracle = ThresholdOracle::with_threshold(0.5);

        // Build up history of low distances.
        for _ in 0..20 {
            oracle.check(0.1);
        }

        // A high distance should have high confidence.
        let decision = oracle.check(0.9);
        assert!(decision.confidence > 0.5);
    }
}
