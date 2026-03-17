//! Distance-based oracle for computing semantic distances between IRs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Available distance metrics for comparing intermediate representations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance between feature vectors.
    Cosine,
    /// Euclidean distance in feature space.
    Euclidean,
    /// Jaccard distance for set-based comparisons.
    Jaccard,
    /// Edit distance for sequence comparisons.
    EditDistance,
    /// Tree edit distance for parse tree comparisons.
    TreeEditDistance,
    /// Weighted combination of multiple metrics.
    Weighted(Vec<(DistanceMetric, f64)>),
    /// Custom metric identified by name.
    Custom(String),
}

/// A distance-based oracle that computes distance and checks against a threshold.
#[derive(Debug, Clone)]
pub struct DistanceOracle {
    metric: DistanceMetric,
    threshold: f64,
    per_stage_thresholds: HashMap<String, f64>,
    normalization: NormalizationStrategy,
}

/// How to normalize distance values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationStrategy {
    /// No normalization.
    None,
    /// Normalize to [0, 1] using min-max over observed values.
    MinMax,
    /// Z-score normalization using observed mean and std.
    ZScore,
    /// Log normalization.
    Log,
}

/// Result of a distance computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceResult {
    pub metric_name: String,
    pub raw_distance: f64,
    pub normalized_distance: f64,
    pub is_violation: bool,
    pub components: Vec<DistanceComponent>,
}

/// A component of a composite distance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceComponent {
    pub name: String,
    pub value: f64,
    pub weight: f64,
}

impl DistanceOracle {
    pub fn new(metric: DistanceMetric, threshold: f64) -> Self {
        Self {
            metric,
            threshold,
            per_stage_thresholds: HashMap::new(),
            normalization: NormalizationStrategy::None,
        }
    }

    pub fn with_normalization(mut self, strategy: NormalizationStrategy) -> Self {
        self.normalization = strategy;
        self
    }

    pub fn set_stage_threshold(&mut self, stage: impl Into<String>, threshold: f64) {
        self.per_stage_thresholds.insert(stage.into(), threshold);
    }

    /// Get the threshold for a specific stage, falling back to the default.
    pub fn threshold_for_stage(&self, stage: &str) -> f64 {
        self.per_stage_thresholds
            .get(stage)
            .copied()
            .unwrap_or(self.threshold)
    }

    /// Compute cosine distance between two vectors.
    pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 1.0;
        }
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let denom = norm_a * norm_b;
        if denom < f64::EPSILON {
            return 1.0;
        }
        1.0 - (dot / denom)
    }

    /// Compute Euclidean distance between two vectors.
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute Jaccard distance between two sets (represented as sorted slices).
    pub fn jaccard_distance(a: &[String], b: &[String]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let set_a: std::collections::HashSet<&String> = a.iter().collect();
        let set_b: std::collections::HashSet<&String> = b.iter().collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            return 0.0;
        }
        1.0 - (intersection as f64 / union as f64)
    }

    /// Compute Levenshtein edit distance between two strings.
    pub fn edit_distance(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        let mut dp = vec![vec![0usize; n + 1]; m + 1];

        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[m][n]
    }

    /// Compute normalized edit distance (0.0 to 1.0).
    pub fn normalized_edit_distance(a: &str, b: &str) -> f64 {
        let dist = Self::edit_distance(a, b);
        let max_len = a.len().max(b.len());
        if max_len == 0 {
            return 0.0;
        }
        dist as f64 / max_len as f64
    }

    /// Compute distance based on the configured metric for feature vectors.
    pub fn compute_vector_distance(&self, a: &[f64], b: &[f64]) -> DistanceResult {
        let (metric_name, raw_distance) = match &self.metric {
            DistanceMetric::Cosine => ("cosine".to_string(), Self::cosine_distance(a, b)),
            DistanceMetric::Euclidean => {
                ("euclidean".to_string(), Self::euclidean_distance(a, b))
            }
            DistanceMetric::Weighted(components) => {
                let mut total = 0.0;
                let mut total_weight = 0.0;
                let mut detail_components = Vec::new();
                for (sub_metric, weight) in components {
                    let d = match sub_metric {
                        DistanceMetric::Cosine => Self::cosine_distance(a, b),
                        DistanceMetric::Euclidean => Self::euclidean_distance(a, b),
                        _ => 0.0,
                    };
                    total += d * weight;
                    total_weight += weight;
                    detail_components.push(DistanceComponent {
                        name: format!("{:?}", sub_metric),
                        value: d,
                        weight: *weight,
                    });
                }
                let dist = if total_weight > 0.0 {
                    total / total_weight
                } else {
                    0.0
                };
                return DistanceResult {
                    metric_name: "weighted".to_string(),
                    raw_distance: dist,
                    normalized_distance: dist,
                    is_violation: dist > self.threshold,
                    components: detail_components,
                };
            }
            _ => ("unknown".to_string(), 0.0),
        };

        let normalized = self.normalize(raw_distance);

        DistanceResult {
            metric_name,
            raw_distance,
            normalized_distance: normalized,
            is_violation: normalized > self.threshold,
            components: Vec::new(),
        }
    }

    /// Apply normalization strategy.
    fn normalize(&self, distance: f64) -> f64 {
        match &self.normalization {
            NormalizationStrategy::None => distance,
            NormalizationStrategy::MinMax => distance.min(1.0).max(0.0),
            NormalizationStrategy::ZScore => distance,
            NormalizationStrategy::Log => (1.0 + distance).ln(),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = DistanceOracle::cosine_distance(&a, &a);
        assert!(d.abs() < 0.001);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = DistanceOracle::cosine_distance(&a, &b);
        assert!((d - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = DistanceOracle::euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_distance() {
        let a = vec!["a".into(), "b".into(), "c".into()];
        let b = vec!["b".into(), "c".into(), "d".into()];
        let d = DistanceOracle::jaccard_distance(&a, &b);
        // intersection = {b, c} = 2, union = {a, b, c, d} = 4, jaccard = 1 - 2/4 = 0.5
        assert!((d - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(DistanceOracle::edit_distance("kitten", "sitting"), 3);
        assert_eq!(DistanceOracle::edit_distance("", "abc"), 3);
        assert_eq!(DistanceOracle::edit_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_normalized_edit_distance() {
        let d = DistanceOracle::normalized_edit_distance("abc", "abc");
        assert!(d.abs() < f64::EPSILON);

        let d2 = DistanceOracle::normalized_edit_distance("abc", "xyz");
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_distance_oracle_cosine() {
        let oracle = DistanceOracle::new(DistanceMetric::Cosine, 0.3);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = oracle.compute_vector_distance(&a, &b);
        assert!(!result.is_violation);
        assert!(result.raw_distance < 0.01);
    }

    #[test]
    fn test_distance_oracle_violation() {
        let oracle = DistanceOracle::new(DistanceMetric::Cosine, 0.3);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0];
        let result = oracle.compute_vector_distance(&a, &b);
        assert!(result.is_violation);
    }

    #[test]
    fn test_per_stage_thresholds() {
        let mut oracle = DistanceOracle::new(DistanceMetric::Cosine, 0.5);
        oracle.set_stage_threshold("tagger", 0.3);
        oracle.set_stage_threshold("ner", 0.7);

        assert!((oracle.threshold_for_stage("tagger") - 0.3).abs() < f64::EPSILON);
        assert!((oracle.threshold_for_stage("ner") - 0.7).abs() < f64::EPSILON);
        assert!((oracle.threshold_for_stage("parser") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_distance() {
        let metric = DistanceMetric::Weighted(vec![
            (DistanceMetric::Cosine, 2.0),
            (DistanceMetric::Euclidean, 1.0),
        ]);
        let oracle = DistanceOracle::new(metric, 0.5);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = oracle.compute_vector_distance(&a, &b);
        assert_eq!(result.components.len(), 2);
        assert!(result.raw_distance > 0.0);
    }
}
