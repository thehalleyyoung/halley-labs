// Futility predictor: binary classifier predicting when no k-block decomposition
// improves the dual bound by more than epsilon.

use crate::error::{OracleError, OracleResult};
use crate::futility::threshold::{ThresholdCalibrator, ThresholdStrategy};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Features used for futility prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutilityFeatures {
    pub spectral_gap: f64,
    pub spectral_ratio: f64,
    pub effective_dimension: f64,
    pub block_separability: f64,
    pub constraint_density: f64,
    pub variable_count: usize,
    pub constraint_count: usize,
    pub nonzero_density: f64,
}

impl FutilityFeatures {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.spectral_gap,
            self.spectral_ratio,
            self.effective_dimension,
            self.block_separability,
            self.constraint_density,
            self.variable_count as f64,
            self.constraint_count as f64,
            self.nonzero_density,
        ]
    }

    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "spectral_gap",
            "spectral_ratio",
            "effective_dimension",
            "block_separability",
            "constraint_density",
            "variable_count",
            "constraint_count",
            "nonzero_density",
        ]
    }
}

/// Result of a futility prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutilityPrediction {
    pub is_futile: bool,
    pub futility_score: f64,
    pub confidence: f64,
    pub threshold_used: f64,
    pub contributing_features: Vec<(String, f64)>,
}

/// Logistic-regression-based futility predictor with asymmetric loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutilityPredictor {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub threshold: f64,
    pub threshold_strategy: ThresholdStrategy,
    pub target_precision: f64,
    pub false_negative_weight: f64,
    pub feature_means: Vec<f64>,
    pub feature_stds: Vec<f64>,
    pub learning_rate: f64,
    pub max_iter: usize,
    pub lambda: f64,
    pub training_precision: f64,
    pub training_recall: f64,
    trained: bool,
}

impl FutilityPredictor {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            threshold: 0.5,
            threshold_strategy: ThresholdStrategy::TargetPrecision { target: 0.8 },
            target_precision: 0.8,
            false_negative_weight: 2.0,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
            learning_rate: 0.01,
            max_iter: 500,
            lambda: 0.01,
            training_precision: 0.0,
            training_recall: 0.0,
            trained: false,
        }
    }

    pub fn with_target_precision(mut self, precision: f64) -> Self {
        self.target_precision = precision;
        self.threshold_strategy = ThresholdStrategy::TargetPrecision { target: precision };
        self
    }

    pub fn with_false_negative_weight(mut self, weight: f64) -> Self {
        self.false_negative_weight = weight;
        self
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn standardize_features(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i < self.feature_means.len() && self.feature_stds[i] > 1e-12 {
                    (x - self.feature_means[i]) / self.feature_stds[i]
                } else {
                    x
                }
            })
            .collect()
    }

    fn fit_standardizer(&mut self, features: &[Vec<f64>]) {
        if features.is_empty() {
            return;
        }
        let p = features[0].len();
        let n = features.len() as f64;

        self.feature_means = vec![0.0; p];
        self.feature_stds = vec![0.0; p];

        for feat in features {
            for j in 0..p {
                self.feature_means[j] += feat[j];
            }
        }
        for j in 0..p {
            self.feature_means[j] /= n;
        }

        for feat in features {
            for j in 0..p {
                let diff = feat[j] - self.feature_means[j];
                self.feature_stds[j] += diff * diff;
            }
        }
        for j in 0..p {
            self.feature_stds[j] = (self.feature_stds[j] / n).sqrt();
            if self.feature_stds[j] < 1e-12 {
                self.feature_stds[j] = 1.0;
            }
        }
    }

    fn compute_score(&self, features: &[f64]) -> f64 {
        let std_feat = self.standardize_features(features);
        let logit: f64 = std_feat
            .iter()
            .zip(self.weights.iter())
            .map(|(&x, &w)| x * w)
            .sum::<f64>()
            + self.bias;
        Self::sigmoid(logit)
    }

    /// Train on binary labels: true = futile, false = not futile.
    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        labels: &[bool],
    ) -> OracleResult<()> {
        if features.is_empty() || features.len() != labels.len() {
            return Err(OracleError::invalid_input("invalid training data"));
        }

        let p = features[0].len();
        self.fit_standardizer(features);

        // Standardize training data
        let std_features: Vec<Vec<f64>> = features
            .iter()
            .map(|f| self.standardize_features(f))
            .collect();

        // Initialize weights
        self.weights = vec![0.0; p];
        self.bias = 0.0;
        let n = features.len() as f64;

        // Gradient descent with asymmetric loss
        for _iter in 0..self.max_iter {
            let mut grad_w = vec![0.0_f64; p];
            let mut grad_b = 0.0_f64;

            for (i, feat) in std_features.iter().enumerate() {
                let logit: f64 = feat
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&x, &w)| x * w)
                    .sum::<f64>()
                    + self.bias;
                let pred = Self::sigmoid(logit);
                let y = if labels[i] { 1.0 } else { 0.0 };

                // Asymmetric weighting: penalize false negatives more
                let sample_weight = if !labels[i] {
                    self.false_negative_weight // false negative: predicted futile but actually useful
                } else {
                    1.0
                };

                let error = (pred - y) * sample_weight;
                for j in 0..p {
                    grad_w[j] += error * feat[j];
                }
                grad_b += error;
            }

            for j in 0..p {
                grad_w[j] = grad_w[j] / n + self.lambda * self.weights[j];
                self.weights[j] -= self.learning_rate * grad_w[j];
            }
            self.bias -= self.learning_rate * grad_b / n;
        }

        // Calibrate threshold
        let scores: Vec<f64> = features.iter().map(|f| self.compute_score(f)).collect();
        let calibrator = ThresholdCalibrator::new();
        self.threshold = calibrator.calibrate(&scores, labels, &self.threshold_strategy);

        // Compute training precision and recall
        let (precision, recall) = self.compute_precision_recall(&scores, labels);
        self.training_precision = precision;
        self.training_recall = recall;

        self.trained = true;
        Ok(())
    }

    fn compute_precision_recall(&self, scores: &[f64], labels: &[bool]) -> (f64, f64) {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for (i, &score) in scores.iter().enumerate() {
            let predicted_positive = score >= self.threshold;
            let actual_positive = labels[i];

            match (predicted_positive, actual_positive) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                _ => {}
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };

        (precision, recall)
    }

    /// Predict futility for a single instance.
    pub fn predict(&self, features: &FutilityFeatures) -> OracleResult<FutilityPrediction> {
        if !self.trained {
            return Err(OracleError::model_not_trained("futility predictor not trained"));
        }

        let feat_vec = features.to_vec();
        let score = self.compute_score(&feat_vec);
        let is_futile = score >= self.threshold;

        // Compute feature contributions
        let std_feat = self.standardize_features(&feat_vec);
        let names = FutilityFeatures::feature_names();
        let mut contributions: Vec<(String, f64)> = std_feat
            .iter()
            .zip(self.weights.iter())
            .enumerate()
            .map(|(i, (&x, &w))| {
                let name = if i < names.len() {
                    names[i].to_string()
                } else {
                    format!("feature_{}", i)
                };
                (name, x * w)
            })
            .collect();
        contributions.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let confidence = if is_futile {
            score
        } else {
            1.0 - score
        };

        Ok(FutilityPrediction {
            is_futile,
            futility_score: score,
            confidence,
            threshold_used: self.threshold,
            contributing_features: contributions,
        })
    }

    /// Predict futility for multiple instances.
    pub fn predict_batch(
        &self,
        features_list: &[FutilityFeatures],
    ) -> OracleResult<Vec<FutilityPrediction>> {
        features_list.iter().map(|f| self.predict(f)).collect()
    }

    /// Cross-validate the futility predictor.
    pub fn cross_validate(
        features: &[Vec<f64>],
        labels: &[bool],
        n_folds: usize,
        seed: u64,
    ) -> OracleResult<FutilityCVResults> {
        if features.len() < n_folds || features.len() != labels.len() {
            return Err(OracleError::invalid_input("invalid CV data"));
        }

        let n = features.len();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let fold_size = n / n_folds;
        let mut fold_precisions = Vec::new();
        let mut fold_recalls = Vec::new();
        let mut fold_thresholds = Vec::new();

        for fold in 0..n_folds {
            let val_start = fold * fold_size;
            let val_end = if fold == n_folds - 1 { n } else { val_start + fold_size };

            let val_indices: Vec<usize> = indices[val_start..val_end].to_vec();
            let train_indices: Vec<usize> = indices[..val_start]
                .iter()
                .chain(indices[val_end..].iter())
                .copied()
                .collect();

            let train_feats: Vec<Vec<f64>> = train_indices.iter().map(|&i| features[i].clone()).collect();
            let train_labs: Vec<bool> = train_indices.iter().map(|&i| labels[i]).collect();
            let val_feats: Vec<Vec<f64>> = val_indices.iter().map(|&i| features[i].clone()).collect();
            let val_labs: Vec<bool> = val_indices.iter().map(|&i| labels[i]).collect();

            let mut predictor = FutilityPredictor::new();
            predictor.train(&train_feats, &train_labs)?;

            let scores: Vec<f64> = val_feats.iter().map(|f| predictor.compute_score(f)).collect();
            let (precision, recall) = predictor.compute_precision_recall(&scores, &val_labs);

            fold_precisions.push(precision);
            fold_recalls.push(recall);
            fold_thresholds.push(predictor.threshold);
        }

        Ok(FutilityCVResults {
            fold_precisions,
            fold_recalls,
            fold_thresholds,
        })
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Cross-validation results for the futility predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutilityCVResults {
    pub fold_precisions: Vec<f64>,
    pub fold_recalls: Vec<f64>,
    pub fold_thresholds: Vec<f64>,
}

impl FutilityCVResults {
    pub fn mean_precision(&self) -> f64 {
        let n = self.fold_precisions.len() as f64;
        if n == 0.0 { return 0.0; }
        self.fold_precisions.iter().sum::<f64>() / n
    }

    pub fn mean_recall(&self) -> f64 {
        let n = self.fold_recalls.len() as f64;
        if n == 0.0 { return 0.0; }
        self.fold_recalls.iter().sum::<f64>() / n
    }

    pub fn std_precision(&self) -> f64 {
        let mean = self.mean_precision();
        let n = self.fold_precisions.len() as f64;
        if n <= 1.0 { return 0.0; }
        let var = self.fold_precisions.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    }

    pub fn mean_threshold(&self) -> f64 {
        let n = self.fold_thresholds.len() as f64;
        if n == 0.0 { return 0.5; }
        self.fold_thresholds.iter().sum::<f64>() / n
    }

    pub fn threshold_stability(&self) -> f64 {
        let mean = self.mean_threshold();
        let n = self.fold_thresholds.len() as f64;
        if n <= 1.0 { return 0.0; }
        let var = self.fold_thresholds.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_futility_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<bool>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for _ in 0..n {
            let gap: f64 = rng.gen_range(0.0..2.0);
            let ratio: f64 = rng.gen_range(0.0..5.0);
            let eff_dim: f64 = rng.gen_range(1.0..50.0);
            let separability: f64 = rng.gen_range(0.0..1.0);

            // Futile when gap is large and separability is low
            let score = gap * 0.8 - separability * 1.5 + rng.gen_range(-0.3..0.3);
            let is_futile = score > 0.5;

            features.push(vec![gap, ratio, eff_dim, separability]);
            labels.push(is_futile);
        }
        (features, labels)
    }

    #[test]
    fn test_futility_features_to_vec() {
        let f = FutilityFeatures {
            spectral_gap: 0.5,
            spectral_ratio: 1.2,
            effective_dimension: 10.0,
            block_separability: 0.8,
            constraint_density: 0.1,
            variable_count: 100,
            constraint_count: 50,
            nonzero_density: 0.05,
        };
        let v = f.to_vec();
        assert_eq!(v.len(), 8);
        assert_eq!(v[0], 0.5);
    }

    #[test]
    fn test_sigmoid() {
        assert!((FutilityPredictor::sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(FutilityPredictor::sigmoid(10.0) > 0.99);
        assert!(FutilityPredictor::sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_futility_train() {
        let (features, labels) = make_futility_data(200, 42);
        let mut predictor = FutilityPredictor::new();
        predictor.train(&features, &labels).unwrap();
        assert!(predictor.is_trained());
    }

    #[test]
    fn test_futility_predict() {
        let (features, labels) = make_futility_data(200, 42);
        let mut predictor = FutilityPredictor::new();
        predictor.train(&features, &labels).unwrap();

        let ff = FutilityFeatures {
            spectral_gap: 1.5,
            spectral_ratio: 3.0,
            effective_dimension: 5.0,
            block_separability: 0.1,
            constraint_density: 0.2,
            variable_count: 100,
            constraint_count: 50,
            nonzero_density: 0.05,
        };
        let pred = predictor.predict(&ff).unwrap();
        assert!(pred.futility_score >= 0.0 && pred.futility_score <= 1.0);
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_futility_predict_batch() {
        let (features, labels) = make_futility_data(100, 42);
        let mut predictor = FutilityPredictor::new();
        predictor.train(&features, &labels).unwrap();

        let batch: Vec<FutilityFeatures> = (0..5)
            .map(|i| FutilityFeatures {
                spectral_gap: i as f64 * 0.3,
                spectral_ratio: 1.0,
                effective_dimension: 10.0,
                block_separability: 0.5,
                constraint_density: 0.1,
                variable_count: 100,
                constraint_count: 50,
                nonzero_density: 0.05,
            })
            .collect();
        let preds = predictor.predict_batch(&batch).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_futility_untrained() {
        let predictor = FutilityPredictor::new();
        let ff = FutilityFeatures {
            spectral_gap: 0.5,
            spectral_ratio: 1.0,
            effective_dimension: 10.0,
            block_separability: 0.5,
            constraint_density: 0.1,
            variable_count: 100,
            constraint_count: 50,
            nonzero_density: 0.05,
        };
        assert!(predictor.predict(&ff).is_err());
    }

    #[test]
    fn test_futility_cross_validate() {
        let (features, labels) = make_futility_data(100, 42);
        let results = FutilityPredictor::cross_validate(&features, &labels, 3, 42).unwrap();
        assert_eq!(results.fold_precisions.len(), 3);
        assert!(results.mean_precision() >= 0.0);
    }

    #[test]
    fn test_cv_results_statistics() {
        let results = FutilityCVResults {
            fold_precisions: vec![0.8, 0.85, 0.75],
            fold_recalls: vec![0.6, 0.65, 0.55],
            fold_thresholds: vec![0.5, 0.52, 0.48],
        };
        assert!((results.mean_precision() - 0.8).abs() < 1e-10);
        assert!(results.std_precision() > 0.0);
        assert!(results.threshold_stability() > 0.0);
    }

    #[test]
    fn test_futility_contributing_features() {
        let (features, labels) = make_futility_data(200, 42);
        let mut predictor = FutilityPredictor::new();
        predictor.train(&features, &labels).unwrap();

        let ff = FutilityFeatures {
            spectral_gap: 1.0,
            spectral_ratio: 2.0,
            effective_dimension: 20.0,
            block_separability: 0.5,
            constraint_density: 0.1,
            variable_count: 100,
            constraint_count: 50,
            nonzero_density: 0.05,
        };
        let pred = predictor.predict(&ff).unwrap();
        assert!(!pred.contributing_features.is_empty());
    }

    #[test]
    fn test_futility_target_precision() {
        let predictor = FutilityPredictor::new().with_target_precision(0.9);
        assert_eq!(predictor.target_precision, 0.9);
    }

    #[test]
    fn test_futility_empty_data() {
        let mut predictor = FutilityPredictor::new();
        assert!(predictor.train(&[], &[]).is_err());
    }
}
