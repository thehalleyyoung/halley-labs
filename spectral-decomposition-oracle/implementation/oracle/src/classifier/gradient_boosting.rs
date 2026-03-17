// Gradient Boosting classifier for multi-class decomposition method prediction.
// Uses regression trees as base learners with softmax loss.

use crate::classifier::traits::{
    Classifier, DecompositionMethod, FeatureVector,
};
use crate::error::{OracleError, OracleResult};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Hyperparameters for Gradient Boosting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingParams {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub subsample: f64,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub early_stopping_rounds: Option<usize>,
    pub validation_fraction: f64,
    pub seed: Option<u64>,
}

impl Default for GradientBoostingParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 4,
            subsample: 0.8,
            min_samples_leaf: 5,
            min_samples_split: 10,
            early_stopping_rounds: Some(10),
            validation_fraction: 0.1,
            seed: Some(42),
        }
    }
}

/// A regression tree node for gradient boosting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionNode {
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<RegressionNode>,
        right: Box<RegressionNode>,
        n_samples: usize,
    },
    Leaf {
        value: f64,
        n_samples: usize,
    },
}

impl RegressionNode {
    pub fn predict(&self, features: &[f64]) -> f64 {
        match self {
            RegressionNode::Leaf { value, .. } => *value,
            RegressionNode::Split {
                feature_index,
                threshold,
                left,
                right,
                ..
            } => {
                if features[*feature_index] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            RegressionNode::Leaf { .. } => 0,
            RegressionNode::Split { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            RegressionNode::Leaf { .. } => 1,
            RegressionNode::Split { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }
        }
    }

    pub fn feature_importances(&self, importances: &mut Vec<f64>, weight: f64) {
        match self {
            RegressionNode::Leaf { .. } => {}
            RegressionNode::Split {
                feature_index,
                left,
                right,
                n_samples,
                ..
            } => {
                if *feature_index < importances.len() {
                    importances[*feature_index] += weight * (*n_samples as f64);
                }
                left.feature_importances(importances, weight);
                right.feature_importances(importances, weight);
            }
        }
    }
}

/// A regression tree trained on pseudo-residuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTree {
    pub root: Option<RegressionNode>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
}

impl RegressionTree {
    pub fn new(max_depth: usize, min_samples_leaf: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_leaf,
            min_samples_split,
        }
    }

    pub fn fit(&mut self, features: &[FeatureVector], targets: &[f64]) {
        let indices: Vec<usize> = (0..features.len()).collect();
        self.root = Some(self.build_node(features, targets, &indices, 0));
    }

    fn build_node(
        &self,
        features: &[FeatureVector],
        targets: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> RegressionNode {
        let n = indices.len();
        let mean_val = indices.iter().map(|&i| targets[i]).sum::<f64>() / n.max(1) as f64;

        if depth >= self.max_depth || n < self.min_samples_split || n < 2 {
            return RegressionNode::Leaf {
                value: mean_val,
                n_samples: n,
            };
        }

        if let Some((feat_idx, threshold, left_idx, right_idx)) =
            self.find_best_split(features, targets, indices)
        {
            if left_idx.len() < self.min_samples_leaf
                || right_idx.len() < self.min_samples_leaf
            {
                return RegressionNode::Leaf {
                    value: mean_val,
                    n_samples: n,
                };
            }

            let left = self.build_node(features, targets, &left_idx, depth + 1);
            let right = self.build_node(features, targets, &right_idx, depth + 1);

            RegressionNode::Split {
                feature_index: feat_idx,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
                n_samples: n,
            }
        } else {
            RegressionNode::Leaf {
                value: mean_val,
                n_samples: n,
            }
        }
    }

    fn find_best_split(
        &self,
        features: &[FeatureVector],
        targets: &[f64],
        indices: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        if features.is_empty() || indices.is_empty() {
            return None;
        }

        let n_features = features[0].len();
        let n = indices.len() as f64;
        let total_sum: f64 = indices.iter().map(|&i| targets[i]).sum();
        let total_sq_sum: f64 = indices.iter().map(|&i| targets[i] * targets[i]).sum();
        let parent_variance = total_sq_sum / n - (total_sum / n).powi(2);

        let mut best_gain = 0.0;
        let mut best_result: Option<(usize, f64, Vec<usize>, Vec<usize>)> = None;

        for feat_idx in 0..n_features {
            let mut sorted: Vec<(f64, f64)> = indices
                .iter()
                .map(|&i| (features[i][feat_idx], targets[i]))
                .collect();
            sorted.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut left_sum = 0.0;
            let mut left_sq = 0.0;
            let mut left_n = 0.0;

            for i in 0..sorted.len() - 1 {
                let (val, target) = sorted[i];
                left_sum += target;
                left_sq += target * target;
                left_n += 1.0;

                let right_n = n - left_n;
                let right_sum = total_sum - left_sum;

                let next_val = sorted[i + 1].0;
                if (next_val - val).abs() < 1e-15 {
                    continue;
                }

                let left_var = if left_n > 0.0 {
                    left_sq / left_n - (left_sum / left_n).powi(2)
                } else {
                    0.0
                };
                let right_sq = total_sq_sum - left_sq;
                let right_var = if right_n > 0.0 {
                    right_sq / right_n - (right_sum / right_n).powi(2)
                } else {
                    0.0
                };

                let weighted_var =
                    (left_n / n) * left_var + (right_n / n) * right_var;
                let gain = parent_variance - weighted_var;

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (val + next_val) / 2.0;
                    let left_idx: Vec<usize> = indices
                        .iter()
                        .filter(|&&idx| features[idx][feat_idx] <= threshold)
                        .copied()
                        .collect();
                    let right_idx: Vec<usize> = indices
                        .iter()
                        .filter(|&&idx| features[idx][feat_idx] > threshold)
                        .copied()
                        .collect();
                    best_result = Some((feat_idx, threshold, left_idx, right_idx));
                }
            }
        }

        best_result
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        match &self.root {
            Some(root) => root.predict(features),
            None => 0.0,
        }
    }
}

/// Multi-class gradient boosting classifier using softmax loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingClassifier {
    pub params: GradientBoostingParams,
    /// trees[round][class] -> regression tree
    pub trees: Vec<Vec<RegressionTree>>,
    /// Initial log-odds for each class (class prior probabilities).
    pub init_scores: Vec<f64>,
    pub n_classes: usize,
    pub n_features: usize,
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub best_n_estimators: usize,
    pub feature_importances: Vec<f64>,
    trained: bool,
}

impl GradientBoostingClassifier {
    pub fn new(params: GradientBoostingParams) -> Self {
        let n_classes = DecompositionMethod::n_classes();
        Self {
            params,
            trees: Vec::new(),
            init_scores: vec![0.0; n_classes],
            n_classes,
            n_features: 0,
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            best_n_estimators: 0,
            feature_importances: Vec::new(),
            trained: false,
        }
    }

    /// Compute softmax probabilities from raw scores.
    fn softmax(scores: &[f64]) -> Vec<f64> {
        let max_score = scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Compute cross-entropy loss for a set of samples.
    fn cross_entropy_loss(scores: &[Vec<f64>], labels: &[usize]) -> f64 {
        let n = labels.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let mut total_loss = 0.0;
        for (i, &label) in labels.iter().enumerate() {
            let proba = Self::softmax(&scores[i]);
            let p = proba[label].max(1e-15);
            total_loss -= p.ln();
        }
        total_loss / n
    }

    /// Compute raw scores for a sample given current model state.
    fn compute_scores(
        &self,
        features: &FeatureVector,
        n_rounds: usize,
    ) -> Vec<f64> {
        let mut scores = self.init_scores.clone();
        let rounds = n_rounds.min(self.trees.len());
        for round in 0..rounds {
            for c in 0..self.n_classes {
                scores[c] +=
                    self.params.learning_rate * self.trees[round][c].predict(features);
            }
        }
        scores
    }

    /// Compute raw scores for all samples.
    fn compute_all_scores(
        &self,
        features: &[FeatureVector],
        n_rounds: usize,
    ) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|f| self.compute_scores(f, n_rounds))
            .collect()
    }

    /// Split data into train and validation sets.
    fn train_val_split(
        n: usize,
        val_frac: f64,
        rng: &mut ChaCha8Rng,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(rng);
        let val_size = ((n as f64) * val_frac).ceil() as usize;
        let val_size = val_size.min(n - 1).max(1);
        let val_idx = indices[..val_size].to_vec();
        let train_idx = indices[val_size..].to_vec();
        (train_idx, val_idx)
    }

    /// Subsample indices for stochastic gradient boosting.
    fn subsample(indices: &[usize], fraction: f64, rng: &mut ChaCha8Rng) -> Vec<usize> {
        if fraction >= 1.0 {
            return indices.to_vec();
        }
        let n = ((indices.len() as f64) * fraction).ceil() as usize;
        let mut shuffled = indices.to_vec();
        shuffled.shuffle(rng);
        shuffled.truncate(n);
        shuffled
    }
}

impl Classifier for GradientBoostingClassifier {
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
                "features and labels must have the same length",
            ));
        }

        self.n_features = features[0].len();
        let label_indices: Vec<usize> = labels.iter().map(|l| l.to_index()).collect();

        let mut rng = match self.params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        // Split train/validation for early stopping
        let (train_idx, val_idx) = if self.params.early_stopping_rounds.is_some()
            && features.len() > 20
        {
            Self::train_val_split(features.len(), self.params.validation_fraction, &mut rng)
        } else {
            let all: Vec<usize> = (0..features.len()).collect();
            (all, Vec::new())
        };

        let train_features: Vec<FeatureVector> =
            train_idx.iter().map(|&i| features[i].clone()).collect();
        let train_labels: Vec<usize> =
            train_idx.iter().map(|&i| label_indices[i]).collect();
        let val_features: Vec<FeatureVector> =
            val_idx.iter().map(|&i| features[i].clone()).collect();
        let val_labels: Vec<usize> =
            val_idx.iter().map(|&i| label_indices[i]).collect();

        // Initialize with class priors (log-odds)
        let n = train_labels.len() as f64;
        let mut class_counts = vec![0.0_f64; self.n_classes];
        for &l in &train_labels {
            class_counts[l] += 1.0;
        }
        self.init_scores = class_counts
            .iter()
            .map(|&c| (c / n).max(1e-10).ln())
            .collect();

        // Current scores for training samples
        let mut current_scores: Vec<Vec<f64>> = train_features
            .iter()
            .map(|_| self.init_scores.clone())
            .collect();

        self.trees.clear();
        self.train_losses.clear();
        self.val_losses.clear();

        let mut best_val_loss = f64::INFINITY;
        let mut rounds_without_improvement = 0;
        self.best_n_estimators = self.params.n_estimators;

        for round in 0..self.params.n_estimators {
            // Compute pseudo-residuals for each class
            let probas: Vec<Vec<f64>> = current_scores
                .iter()
                .map(|s| Self::softmax(s))
                .collect();

            let sample_idx =
                Self::subsample(&(0..train_features.len()).collect::<Vec<_>>(), self.params.subsample, &mut rng);

            let mut round_trees = Vec::new();

            for c in 0..self.n_classes {
                // Negative gradient for class c: y_c - p_c
                let residuals: Vec<f64> = (0..train_features.len())
                    .map(|i| {
                        let y = if train_labels[i] == c { 1.0 } else { 0.0 };
                        y - probas[i][c]
                    })
                    .collect();

                // Fit regression tree to residuals on subsample
                let sub_features: Vec<FeatureVector> =
                    sample_idx.iter().map(|&i| train_features[i].clone()).collect();
                let sub_residuals: Vec<f64> =
                    sample_idx.iter().map(|&i| residuals[i]).collect();

                let mut tree = RegressionTree::new(
                    self.params.max_depth,
                    self.params.min_samples_leaf,
                    self.params.min_samples_split,
                );
                tree.fit(&sub_features, &sub_residuals);

                // Update current scores
                for i in 0..train_features.len() {
                    let update = tree.predict(&train_features[i]);
                    current_scores[i][c] += self.params.learning_rate * update;
                }

                round_trees.push(tree);
            }

            self.trees.push(round_trees);

            // Compute losses
            let train_loss = Self::cross_entropy_loss(&current_scores, &train_labels);
            self.train_losses.push(train_loss);

            // Validation loss for early stopping
            if !val_features.is_empty() {
                let val_scores = self.compute_all_scores(&val_features, round + 1);
                let val_loss = Self::cross_entropy_loss(&val_scores, &val_labels);
                self.val_losses.push(val_loss);

                if val_loss < best_val_loss - 1e-6 {
                    best_val_loss = val_loss;
                    rounds_without_improvement = 0;
                    self.best_n_estimators = round + 1;
                } else {
                    rounds_without_improvement += 1;
                }

                if let Some(patience) = self.params.early_stopping_rounds {
                    if rounds_without_improvement >= patience {
                        log::info!(
                            "GBT: early stopping at round {} (best: {})",
                            round + 1,
                            self.best_n_estimators
                        );
                        break;
                    }
                }
            } else {
                self.best_n_estimators = round + 1;
            }
        }

        // Compute feature importances
        self.feature_importances = self.compute_gbt_feature_importances();
        self.trained = true;
        Ok(())
    }

    fn predict(&self, features: &FeatureVector) -> OracleResult<DecompositionMethod> {
        if !self.trained {
            return Err(OracleError::model_not_trained("GBT not trained"));
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
            return Err(OracleError::model_not_trained("GBT not trained"));
        }
        if features.len() != self.n_features {
            return Err(OracleError::invalid_input(format!(
                "expected {} features, got {}",
                self.n_features,
                features.len()
            )));
        }
        let scores = self.compute_scores(features, self.best_n_estimators);
        Ok(Self::softmax(&scores))
    }

    fn feature_importance(&self) -> OracleResult<Vec<f64>> {
        if !self.trained {
            return Err(OracleError::model_not_trained("GBT not trained"));
        }
        Ok(self.feature_importances.clone())
    }

    fn name(&self) -> &str {
        "GradientBoosting"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

impl GradientBoostingClassifier {
    fn compute_gbt_feature_importances(&self) -> Vec<f64> {
        let mut importances = vec![0.0_f64; self.n_features];
        let rounds = self.best_n_estimators.min(self.trees.len());

        for round in 0..rounds {
            for tree in &self.trees[round] {
                if let Some(ref root) = tree.root {
                    root.feature_importances(&mut importances, 1.0);
                }
            }
        }

        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for imp in &mut importances {
                *imp /= total;
            }
        }
        importances
    }

    /// Return the learning curves (train/val loss per round).
    pub fn learning_curves(&self) -> (&[f64], &[f64]) {
        (&self.train_losses, &self.val_losses)
    }

    /// Return the optimal number of boosting rounds.
    pub fn optimal_n_estimators(&self) -> usize {
        self.best_n_estimators
    }
}

/// Staged prediction: yield predictions after each boosting round.
pub fn staged_predict(
    model: &GradientBoostingClassifier,
    features: &FeatureVector,
) -> Vec<DecompositionMethod> {
    let rounds = model.best_n_estimators.min(model.trees.len());
    let mut results = Vec::with_capacity(rounds);

    for r in 1..=rounds {
        let scores = model.compute_scores(features, r);
        let proba = GradientBoostingClassifier::softmax(&scores);
        let best = proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if let Ok(method) = DecompositionMethod::from_index(best) {
            results.push(method);
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, seed: u64) -> (Vec<FeatureVector>, Vec<DecompositionMethod>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for _ in 0..n {
            let x: f64 = rng.gen_range(-3.0..3.0);
            let y: f64 = rng.gen_range(-3.0..3.0);
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
    fn test_softmax_basic() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let proba = GradientBoostingClassifier::softmax(&scores);
        assert_eq!(proba.len(), 4);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(proba[3] > proba[0]);
    }

    #[test]
    fn test_softmax_equal() {
        let scores = vec![1.0, 1.0, 1.0, 1.0];
        let proba = GradientBoostingClassifier::softmax(&scores);
        for &p in &proba {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        let scores = vec![1000.0, 1001.0, 999.0, 998.0];
        let proba = GradientBoostingClassifier::softmax(&scores);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regression_tree_fit() {
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 1.0],
            vec![4.0, 2.0],
        ];
        let targets = vec![0.5, 0.6, -0.3, -0.2];
        let mut tree = RegressionTree::new(3, 1, 2);
        tree.fit(&features, &targets);
        assert!(tree.root.is_some());
    }

    #[test]
    fn test_gbt_train_predict() {
        let (features, labels) = make_data(200, 42);
        let params = GradientBoostingParams {
            n_estimators: 20,
            learning_rate: 0.1,
            max_depth: 3,
            early_stopping_rounds: None,
            seed: Some(42),
            ..Default::default()
        };
        let mut gbt = GradientBoostingClassifier::new(params);
        gbt.train(&features, &labels).unwrap();

        assert!(gbt.is_trained());
        let pred = gbt.predict(&features[0]).unwrap();
        assert!(DecompositionMethod::all().contains(&pred));
    }

    #[test]
    fn test_gbt_predict_proba() {
        let (features, labels) = make_data(100, 42);
        let params = GradientBoostingParams {
            n_estimators: 10,
            learning_rate: 0.2,
            max_depth: 3,
            early_stopping_rounds: None,
            seed: Some(42),
            ..Default::default()
        };
        let mut gbt = GradientBoostingClassifier::new(params);
        gbt.train(&features, &labels).unwrap();

        let proba = gbt.predict_proba(&features[0]).unwrap();
        assert_eq!(proba.len(), 4);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gbt_early_stopping() {
        let (features, labels) = make_data(200, 42);
        let params = GradientBoostingParams {
            n_estimators: 100,
            learning_rate: 0.3,
            max_depth: 3,
            early_stopping_rounds: Some(5),
            validation_fraction: 0.2,
            seed: Some(42),
            ..Default::default()
        };
        let mut gbt = GradientBoostingClassifier::new(params);
        gbt.train(&features, &labels).unwrap();

        assert!(gbt.optimal_n_estimators() <= 100);
    }

    #[test]
    fn test_gbt_feature_importance() {
        let (features, labels) = make_data(200, 42);
        let params = GradientBoostingParams {
            n_estimators: 20,
            max_depth: 3,
            early_stopping_rounds: None,
            seed: Some(42),
            ..Default::default()
        };
        let mut gbt = GradientBoostingClassifier::new(params);
        gbt.train(&features, &labels).unwrap();

        let imp = gbt.feature_importance().unwrap();
        assert_eq!(imp.len(), 2);
    }

    #[test]
    fn test_gbt_untrained() {
        let gbt = GradientBoostingClassifier::new(GradientBoostingParams::default());
        assert!(!gbt.is_trained());
        assert!(gbt.predict(&vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn test_gbt_evaluate() {
        let (features, labels) = make_data(200, 42);
        let params = GradientBoostingParams {
            n_estimators: 30,
            max_depth: 4,
            early_stopping_rounds: None,
            seed: Some(42),
            ..Default::default()
        };
        let mut gbt = GradientBoostingClassifier::new(params);
        gbt.train(&features, &labels).unwrap();
        let metrics = gbt.evaluate(&features, &labels).unwrap();
        assert!(metrics.accuracy > 0.3);
    }

    #[test]
    fn test_gbt_learning_curves() {
        let (features, labels) = make_data(100, 42);
        let params = GradientBoostingParams {
            n_estimators: 10,
            early_stopping_rounds: None,
            seed: Some(42),
            ..Default::default()
        };
        let mut gbt = GradientBoostingClassifier::new(params);
        gbt.train(&features, &labels).unwrap();

        let (train_losses, _) = gbt.learning_curves();
        assert_eq!(train_losses.len(), 10);
        // Training loss should generally decrease
        assert!(train_losses[9] <= train_losses[0] + 0.5);
    }

    #[test]
    fn test_staged_predict() {
        let (features, labels) = make_data(100, 42);
        let params = GradientBoostingParams {
            n_estimators: 5,
            early_stopping_rounds: None,
            seed: Some(42),
            ..Default::default()
        };
        let mut gbt = GradientBoostingClassifier::new(params);
        gbt.train(&features, &labels).unwrap();

        let staged = staged_predict(&gbt, &features[0]);
        assert_eq!(staged.len(), 5);
    }

    #[test]
    fn test_regression_node_depth() {
        let leaf = RegressionNode::Leaf {
            value: 0.5,
            n_samples: 1,
        };
        assert_eq!(leaf.depth(), 0);
        assert_eq!(leaf.node_count(), 1);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let scores = vec![vec![10.0, 0.0, 0.0, 0.0]];
        let labels = vec![0];
        let loss = GradientBoostingClassifier::cross_entropy_loss(&scores, &labels);
        assert!(loss < 0.01);
    }

    #[test]
    fn test_subsample() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let indices: Vec<usize> = (0..100).collect();
        let sub = GradientBoostingClassifier::subsample(&indices, 0.5, &mut rng);
        assert!(sub.len() <= 51);
        assert!(sub.len() >= 49);
    }
}
