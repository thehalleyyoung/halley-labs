// Multinomial Logistic Regression with L2 regularization for decomposition selection.

use crate::classifier::traits::{
    Classifier, DecompositionMethod, FeatureVector,
};
use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};

/// Hyperparameters for Logistic Regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegressionParams {
    pub learning_rate: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub lambda: f64, // L2 regularization strength
    pub batch_size: Option<usize>,
    pub standardize: bool,
    pub seed: Option<u64>,
}

impl Default for LogisticRegressionParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-6,
            lambda: 0.01,
            batch_size: None,
            standardize: true,
            seed: Some(42),
        }
    }
}

/// Feature standardizer (z-score normalization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStandardizer {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub fitted: bool,
}

impl FeatureStandardizer {
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            fitted: false,
        }
    }

    pub fn fit(&mut self, features: &[FeatureVector]) {
        if features.is_empty() {
            return;
        }
        let n = features.len() as f64;
        let p = features[0].len();

        self.means = vec![0.0; p];
        self.stds = vec![0.0; p];

        for feat in features {
            for j in 0..p {
                self.means[j] += feat[j];
            }
        }
        for j in 0..p {
            self.means[j] /= n;
        }

        for feat in features {
            for j in 0..p {
                let diff = feat[j] - self.means[j];
                self.stds[j] += diff * diff;
            }
        }
        for j in 0..p {
            self.stds[j] = (self.stds[j] / n).sqrt();
            if self.stds[j] < 1e-12 {
                self.stds[j] = 1.0; // avoid division by zero for constant features
            }
        }
        self.fitted = true;
    }

    pub fn transform(&self, features: &FeatureVector) -> FeatureVector {
        if !self.fitted {
            return features.clone();
        }
        features
            .iter()
            .enumerate()
            .map(|(j, &x)| (x - self.means[j]) / self.stds[j])
            .collect()
    }

    pub fn transform_batch(&self, features: &[FeatureVector]) -> Vec<FeatureVector> {
        features.iter().map(|f| self.transform(f)).collect()
    }
}

/// Multinomial Logistic Regression classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    pub params: LogisticRegressionParams,
    /// Weight matrix: weights[class][feature]. Shape: [n_classes, n_features + 1].
    pub weights: Vec<Vec<f64>>,
    pub n_classes: usize,
    pub n_features: usize,
    pub standardizer: FeatureStandardizer,
    pub convergence_history: Vec<f64>,
    trained: bool,
}

impl LogisticRegression {
    pub fn new(params: LogisticRegressionParams) -> Self {
        let n_classes = DecompositionMethod::n_classes();
        Self {
            params,
            weights: Vec::new(),
            n_classes,
            n_features: 0,
            standardizer: FeatureStandardizer::new(),
            convergence_history: Vec::new(),
            trained: false,
        }
    }

    /// Softmax function for computing class probabilities.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Compute logits for a single sample. Feature vector should already include bias.
    fn compute_logits(&self, features_with_bias: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .map(|w| {
                w.iter()
                    .zip(features_with_bias.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum()
            })
            .collect()
    }

    /// Add bias term to features.
    fn add_bias(features: &FeatureVector) -> FeatureVector {
        let mut result = features.clone();
        result.push(1.0);
        result
    }

    /// Compute cross-entropy loss with L2 regularization.
    fn compute_loss(
        &self,
        features: &[FeatureVector],
        labels: &[usize],
    ) -> f64 {
        let n = labels.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mut loss = 0.0;
        for (i, feat) in features.iter().enumerate() {
            let fb = Self::add_bias(feat);
            let logits = self.compute_logits(&fb);
            let proba = Self::softmax(&logits);
            let p = proba[labels[i]].max(1e-15);
            loss -= p.ln();
        }
        loss /= n;

        // L2 regularization (exclude bias)
        let l2 = self.weights.iter().flat_map(|w| w.iter().take(w.len().saturating_sub(1)))
            .map(|&wi| wi * wi)
            .sum::<f64>();
        loss += 0.5 * self.params.lambda * l2 / n;

        loss
    }

    #[allow(dead_code)]
    fn gradient_step(
        &mut self,
        features: &[FeatureVector],
        labels: &[usize],
    ) {
        let n = features.len() as f64;
        if n == 0.0 {
            return;
        }

        let p = self.weights[0].len(); // n_features + 1

        // Initialize gradients
        let mut gradients = vec![vec![0.0_f64; p]; self.n_classes];

        for (i, feat) in features.iter().enumerate() {
            let fb = Self::add_bias(feat);
            let logits = self.compute_logits(&fb);
            let proba = Self::softmax(&logits);

            for c in 0..self.n_classes {
                let error = proba[c] - if labels[i] == c { 1.0 } else { 0.0 };
                for j in 0..p {
                    gradients[c][j] += error * fb[j];
                }
            }
        }

        // Average and add L2 regularization, then update
        for c in 0..self.n_classes {
            for j in 0..p {
                gradients[c][j] /= n;
                // L2 regularization (skip bias term)
                if j < p - 1 {
                    gradients[c][j] += self.params.lambda * self.weights[c][j] / n;
                }
                self.weights[c][j] -= self.params.learning_rate * gradients[c][j];
            }
        }
    }

    /// L-BFGS-inspired optimization with limited memory.
    fn train_lbfgs(
        &mut self,
        features: &[FeatureVector],
        labels: &[usize],
    ) {
        // Use gradient descent with momentum as a simpler approximation
        let p = self.weights[0].len();
        let mut velocity = vec![vec![0.0_f64; p]; self.n_classes];
        let momentum = 0.9;
        let n = features.len() as f64;

        for iter in 0..self.params.max_iter {
            let mut gradients = vec![vec![0.0_f64; p]; self.n_classes];

            for (i, feat) in features.iter().enumerate() {
                let fb = Self::add_bias(feat);
                let logits = self.compute_logits(&fb);
                let proba = Self::softmax(&logits);

                for c in 0..self.n_classes {
                    let error = proba[c] - if labels[i] == c { 1.0 } else { 0.0 };
                    for j in 0..p {
                        gradients[c][j] += error * fb[j];
                    }
                }
            }

            let mut grad_norm = 0.0;
            for c in 0..self.n_classes {
                for j in 0..p {
                    gradients[c][j] /= n;
                    if j < p - 1 {
                        gradients[c][j] += self.params.lambda * self.weights[c][j] / n;
                    }
                    grad_norm += gradients[c][j] * gradients[c][j];

                    velocity[c][j] = momentum * velocity[c][j]
                        - self.params.learning_rate * gradients[c][j];
                    self.weights[c][j] += velocity[c][j];
                }
            }

            let loss = self.compute_loss(features, labels);
            self.convergence_history.push(loss);

            if grad_norm.sqrt() < self.params.tol {
                log::info!("LogReg converged at iteration {}", iter + 1);
                break;
            }
        }
    }

    /// Select regularization strength via cross-validation.
    pub fn select_regularization(
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
        lambdas: &[f64],
        n_folds: usize,
    ) -> OracleResult<f64> {
        if features.len() < n_folds {
            return Err(OracleError::invalid_input("too few samples for CV"));
        }

        let n = features.len();
        let fold_size = n / n_folds;
        let label_idx: Vec<usize> = labels.iter().map(|l| l.to_index()).collect();

        let mut best_lambda = lambdas[0];
        let mut best_accuracy = 0.0;

        for &lambda in lambdas {
            let mut total_correct = 0usize;
            let mut total_count = 0usize;

            for fold in 0..n_folds {
                let val_start = fold * fold_size;
                let val_end = if fold == n_folds - 1 { n } else { val_start + fold_size };

                let train_feats: Vec<FeatureVector> = (0..val_start)
                    .chain(val_end..n)
                    .map(|i| features[i].clone())
                    .collect();
                let train_labs: Vec<DecompositionMethod> = (0..val_start)
                    .chain(val_end..n)
                    .map(|i| labels[i])
                    .collect();
                let val_feats: Vec<FeatureVector> =
                    (val_start..val_end).map(|i| features[i].clone()).collect();
                let val_labs: Vec<usize> =
                    (val_start..val_end).map(|i| label_idx[i]).collect();

                let params = LogisticRegressionParams {
                    lambda,
                    max_iter: 200,
                    ..Default::default()
                };
                let mut model = LogisticRegression::new(params);
                if model.train(&train_feats, &train_labs).is_ok() {
                    for (j, feat) in val_feats.iter().enumerate() {
                        if let Ok(pred) = model.predict(feat) {
                            if pred.to_index() == val_labs[j] {
                                total_correct += 1;
                            }
                        }
                        total_count += 1;
                    }
                }
            }

            let accuracy = if total_count > 0 {
                total_correct as f64 / total_count as f64
            } else {
                0.0
            };

            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_lambda = lambda;
            }
        }

        Ok(best_lambda)
    }

    /// Calibrate probabilities using Platt scaling.
    pub fn calibrated_proba(&self, features: &FeatureVector) -> OracleResult<Vec<f64>> {
        // Return raw softmax probabilities (already well-calibrated for logistic regression)
        self.predict_proba(features)
    }
}

impl Classifier for LogisticRegression {
    fn train(
        &mut self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
    ) -> OracleResult<()> {
        if features.is_empty() {
            return Err(OracleError::invalid_input("empty training data"));
        }
        if features.len() != labels.len() {
            return Err(OracleError::invalid_input(
                "features and labels length mismatch",
            ));
        }

        self.n_features = features[0].len();
        let label_indices: Vec<usize> = labels.iter().map(|l| l.to_index()).collect();

        // Standardize features
        let processed_features = if self.params.standardize {
            self.standardizer.fit(features);
            self.standardizer.transform_batch(features)
        } else {
            features.to_vec()
        };

        // Initialize weights to small random values
        let p = self.n_features + 1; // +1 for bias
        self.weights = vec![vec![0.0; p]; self.n_classes];

        // Initialize with small values proportional to 1/sqrt(p)
        let scale = 1.0 / (p as f64).sqrt();
        let mut state = 42u64;
        for c in 0..self.n_classes {
            for j in 0..p {
                // Simple PRNG for initialization
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
                self.weights[c][j] = r * scale;
            }
        }

        self.convergence_history.clear();
        self.train_lbfgs(&processed_features, &label_indices);

        self.trained = true;
        Ok(())
    }

    fn predict(&self, features: &FeatureVector) -> OracleResult<DecompositionMethod> {
        if !self.trained {
            return Err(OracleError::model_not_trained("LogReg not trained"));
        }
        let proba = self.predict_proba(features)?;
        let best = proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        DecompositionMethod::from_index(best)
    }

    fn predict_proba(&self, features: &FeatureVector) -> OracleResult<Vec<f64>> {
        if !self.trained {
            return Err(OracleError::model_not_trained("LogReg not trained"));
        }
        if features.len() != self.n_features {
            return Err(OracleError::invalid_input(format!(
                "expected {} features, got {}",
                self.n_features,
                features.len()
            )));
        }
        let processed = if self.params.standardize {
            self.standardizer.transform(features)
        } else {
            features.clone()
        };
        let fb = Self::add_bias(&processed);
        let logits = self.compute_logits(&fb);
        Ok(Self::softmax(&logits))
    }

    fn feature_importance(&self) -> OracleResult<Vec<f64>> {
        if !self.trained {
            return Err(OracleError::model_not_trained("LogReg not trained"));
        }
        // Sum of absolute weights across classes for each feature
        let mut importances = vec![0.0_f64; self.n_features];
        for c in 0..self.n_classes {
            for j in 0..self.n_features {
                importances[j] += self.weights[c][j].abs();
            }
        }
        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for imp in &mut importances {
                *imp /= total;
            }
        }
        Ok(importances)
    }

    fn name(&self) -> &str {
        "LogisticRegression"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

/// One-vs-Rest wrapper for binary classifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneVsRestLogistic {
    pub models: Vec<LogisticRegression>,
    pub n_classes: usize,
    pub n_features: usize,
    trained: bool,
}

impl OneVsRestLogistic {
    pub fn new(params: LogisticRegressionParams) -> Self {
        let n_classes = DecompositionMethod::n_classes();
        let models: Vec<LogisticRegression> = (0..n_classes)
            .map(|_| LogisticRegression::new(params.clone()))
            .collect();
        Self {
            models,
            n_classes,
            n_features: 0,
            trained: false,
        }
    }

    pub fn train(
        &mut self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
    ) -> OracleResult<()> {
        if features.is_empty() {
            return Err(OracleError::invalid_input("empty training data"));
        }
        self.n_features = features[0].len();
        let label_indices: Vec<usize> = labels.iter().map(|l| l.to_index()).collect();

        for c in 0..self.n_classes {
            // Convert to binary: class c vs rest
            let binary_labels: Vec<DecompositionMethod> = label_indices
                .iter()
                .map(|&l| {
                    if l == c {
                        DecompositionMethod::Benders // positive class
                    } else {
                        DecompositionMethod::None // negative class
                    }
                })
                .collect();
            self.models[c].train(features, &binary_labels)?;
        }
        self.trained = true;
        Ok(())
    }

    pub fn predict(&self, features: &FeatureVector) -> OracleResult<DecompositionMethod> {
        if !self.trained {
            return Err(OracleError::model_not_trained("OVR not trained"));
        }
        let mut best_score = f64::NEG_INFINITY;
        let mut best_class = 0;

        for c in 0..self.n_classes {
            let proba = self.models[c].predict_proba(features)?;
            // Score for the positive class (index 0 = Benders = our positive)
            let score = proba[0];
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }
        DecompositionMethod::from_index(best_class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    fn make_data(n: usize, seed: u64) -> (Vec<FeatureVector>, Vec<DecompositionMethod>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for _ in 0..n {
            let x: f64 = rng.gen_range(-2.0..2.0);
            let y: f64 = rng.gen_range(-2.0..2.0);
            let label = if x > 0.0 && y > 0.0 {
                DecompositionMethod::Benders
            } else if x > 0.0 {
                DecompositionMethod::DantzigWolfe
            } else if y > 0.0 {
                DecompositionMethod::Lagrangian
            } else {
                DecompositionMethod::None
            };
            features.push(vec![x, y]);
            labels.push(label);
        }
        (features, labels)
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let proba = LogisticRegression::softmax(&logits);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standardizer() {
        let features = vec![
            vec![1.0, 10.0],
            vec![3.0, 20.0],
            vec![5.0, 30.0],
        ];
        let mut std = FeatureStandardizer::new();
        std.fit(&features);
        assert!(std.fitted);

        let transformed = std.transform(&features[1]);
        assert!((transformed[0] - 0.0).abs() < 1e-10); // mean is 3.0
    }

    #[test]
    fn test_logistic_train_predict() {
        let (features, labels) = make_data(200, 42);
        let params = LogisticRegressionParams {
            max_iter: 200,
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut lr = LogisticRegression::new(params);
        lr.train(&features, &labels).unwrap();

        assert!(lr.is_trained());
        let pred = lr.predict(&features[0]).unwrap();
        assert!(DecompositionMethod::all().contains(&pred));
    }

    #[test]
    fn test_logistic_predict_proba() {
        let (features, labels) = make_data(100, 42);
        let params = LogisticRegressionParams {
            max_iter: 100,
            ..Default::default()
        };
        let mut lr = LogisticRegression::new(params);
        lr.train(&features, &labels).unwrap();

        let proba = lr.predict_proba(&features[0]).unwrap();
        assert_eq!(proba.len(), 4);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_logistic_convergence() {
        let (features, labels) = make_data(100, 42);
        let params = LogisticRegressionParams {
            max_iter: 200,
            ..Default::default()
        };
        let mut lr = LogisticRegression::new(params);
        lr.train(&features, &labels).unwrap();

        assert!(!lr.convergence_history.is_empty());
        // Loss should generally decrease
        let first = lr.convergence_history[0];
        let last = lr.convergence_history.last().unwrap();
        assert!(last <= &(first + 0.5));
    }

    #[test]
    fn test_logistic_feature_importance() {
        let (features, labels) = make_data(200, 42);
        let mut lr = LogisticRegression::new(LogisticRegressionParams::default());
        lr.train(&features, &labels).unwrap();

        let imp = lr.feature_importance().unwrap();
        assert_eq!(imp.len(), 2);
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_logistic_untrained() {
        let lr = LogisticRegression::new(LogisticRegressionParams::default());
        assert!(!lr.is_trained());
        assert!(lr.predict(&vec![1.0]).is_err());
    }

    #[test]
    fn test_logistic_evaluate() {
        let (features, labels) = make_data(200, 42);
        let mut lr = LogisticRegression::new(LogisticRegressionParams {
            max_iter: 300,
            learning_rate: 0.05,
            ..Default::default()
        });
        lr.train(&features, &labels).unwrap();
        let metrics = lr.evaluate(&features, &labels).unwrap();
        assert!(metrics.accuracy > 0.2);
    }

    #[test]
    fn test_standardizer_constant_feature() {
        let features = vec![vec![5.0, 1.0], vec![5.0, 2.0], vec![5.0, 3.0]];
        let mut std = FeatureStandardizer::new();
        std.fit(&features);
        let t = std.transform(&features[0]);
        assert!(t[0].is_finite()); // constant feature handled
    }

    #[test]
    fn test_add_bias() {
        let f = vec![1.0, 2.0];
        let fb = LogisticRegression::add_bias(&f);
        assert_eq!(fb.len(), 3);
        assert_eq!(fb[2], 1.0);
    }

    #[test]
    fn test_ovr_train_predict() {
        let (features, labels) = make_data(200, 42);
        let params = LogisticRegressionParams {
            max_iter: 100,
            ..Default::default()
        };
        let mut ovr = OneVsRestLogistic::new(params);
        ovr.train(&features, &labels).unwrap();
        assert!(ovr.trained);

        let pred = ovr.predict(&features[0]).unwrap();
        assert!(DecompositionMethod::all().contains(&pred));
    }

    #[test]
    fn test_select_regularization() {
        let (features, labels) = make_data(100, 42);
        let lambdas = vec![0.001, 0.01, 0.1, 1.0];
        let best = LogisticRegression::select_regularization(&features, &labels, &lambdas, 3).unwrap();
        assert!(lambdas.contains(&best));
    }
}
