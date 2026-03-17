// Random Forest classifier built from scratch for decomposition method selection.

use crate::classifier::traits::{
    Classifier, DecompositionMethod, FeatureVector,
};
use crate::error::{OracleError, OracleResult};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hyperparameters for the Random Forest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestParams {
    pub n_trees: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
    pub bootstrap: bool,
    pub oob_score: bool,
    pub seed: Option<u64>,
}

impl Default for RandomForestParams {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: Some(15),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            bootstrap: true,
            oob_score: true,
            seed: Some(42),
        }
    }
}

/// Strategy for selecting the number of features at each split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaxFeatures {
    Sqrt,
    Log2,
    Fraction(f64),
    Fixed(usize),
    All,
}

impl MaxFeatures {
    pub fn resolve(&self, n_features: usize) -> usize {
        let m = match self {
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2().ceil() as usize,
            MaxFeatures::Fraction(f) => ((n_features as f64) * f).ceil() as usize,
            MaxFeatures::Fixed(k) => *k,
            MaxFeatures::All => n_features,
        };
        m.max(1).min(n_features)
    }
}

/// A node in a decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        impurity_decrease: f64,
        n_samples: usize,
    },
    Leaf {
        class_distribution: Vec<f64>,
        predicted_class: usize,
        n_samples: usize,
    },
}

impl TreeNode {
    /// Predict class probabilities by traversing the tree.
    pub fn predict_proba(&self, features: &[f64]) -> &[f64] {
        match self {
            TreeNode::Leaf {
                class_distribution, ..
            } => class_distribution,
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
                ..
            } => {
                if features[*feature_index] <= *threshold {
                    left.predict_proba(features)
                } else {
                    right.predict_proba(features)
                }
            }
        }
    }

    /// Return the total impurity decrease contribution across all split nodes.
    pub fn feature_importances(&self, importances: &mut Vec<f64>) {
        match self {
            TreeNode::Leaf { .. } => {}
            TreeNode::Split {
                feature_index,
                impurity_decrease,
                left,
                right,
                ..
            } => {
                if *feature_index < importances.len() {
                    importances[*feature_index] += *impurity_decrease;
                }
                left.feature_importances(importances);
                right.feature_importances(importances);
            }
        }
    }

    /// Count total nodes in this subtree.
    pub fn node_count(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => 1 + left.node_count() + right.node_count(),
        }
    }

    /// Maximum depth of this subtree.
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 0,
            TreeNode::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }
}

/// A single decision tree trained with random feature subsets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub root: Option<TreeNode>,
    pub n_features: usize,
    pub n_classes: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features_per_split: usize,
}

impl DecisionTree {
    pub fn new(
        n_features: usize,
        n_classes: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features_per_split: usize,
    ) -> Self {
        Self {
            root: None,
            n_features,
            n_classes,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features_per_split,
        }
    }

    /// Train the decision tree on a bootstrapped sample.
    pub fn fit(
        &mut self,
        features: &[FeatureVector],
        labels: &[usize],
        rng: &mut ChaCha8Rng,
    ) {
        let indices: Vec<usize> = (0..features.len()).collect();
        self.root = Some(self.build_tree(features, labels, &indices, 0, rng));
    }

    fn build_tree(
        &self,
        features: &[FeatureVector],
        labels: &[usize],
        indices: &[usize],
        depth: usize,
        rng: &mut ChaCha8Rng,
    ) -> TreeNode {
        let n_samples = indices.len();
        let class_dist = self.class_distribution(labels, indices);
        let predicted_class = class_dist
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Check stopping criteria
        let max_depth_reached = self.max_depth.map(|d| depth >= d).unwrap_or(false);
        let is_pure = class_dist.iter().filter(|&&c| c > 0.0).count() <= 1;
        let too_few_samples = n_samples < self.min_samples_split;

        if max_depth_reached || is_pure || too_few_samples {
            return TreeNode::Leaf {
                class_distribution: class_dist,
                predicted_class,
                n_samples,
            };
        }

        // Select random feature subset
        let feature_subset = self.random_feature_subset(rng);

        // Find best split
        if let Some((best_feature, best_threshold, best_impurity_decrease, left_idx, right_idx)) =
            self.find_best_split(features, labels, indices, &feature_subset)
        {
            if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                return TreeNode::Leaf {
                    class_distribution: class_dist,
                    predicted_class,
                    n_samples,
                };
            }

            let left = self.build_tree(features, labels, &left_idx, depth + 1, rng);
            let right = self.build_tree(features, labels, &right_idx, depth + 1, rng);

            TreeNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
                impurity_decrease: best_impurity_decrease,
                n_samples,
            }
        } else {
            TreeNode::Leaf {
                class_distribution: class_dist,
                predicted_class,
                n_samples,
            }
        }
    }

    fn random_feature_subset(&self, rng: &mut ChaCha8Rng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.n_features).collect();
        indices.shuffle(rng);
        indices.truncate(self.max_features_per_split);
        indices
    }

    fn find_best_split(
        &self,
        features: &[FeatureVector],
        labels: &[usize],
        indices: &[usize],
        feature_subset: &[usize],
    ) -> Option<(usize, f64, f64, Vec<usize>, Vec<usize>)> {
        let parent_gini = self.gini_impurity(labels, indices);
        let n = indices.len() as f64;

        let mut best_gain = 0.0;
        let mut best_result: Option<(usize, f64, f64, Vec<usize>, Vec<usize>)> = None;

        for &feat_idx in feature_subset {
            // Gather values and sort by feature
            let mut value_label: Vec<(f64, usize)> = indices
                .iter()
                .map(|&i| (features[i][feat_idx], labels[i]))
                .collect();
            value_label.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Initialize class counts for left and right
            let mut left_counts = vec![0usize; self.n_classes];
            let mut right_counts = vec![0usize; self.n_classes];
            for &(_, label) in &value_label {
                right_counts[label] += 1;
            }

            let mut left_n = 0usize;
            let mut right_n = value_label.len();

            // Scan through sorted values to find best threshold
            for i in 0..value_label.len() - 1 {
                let (val, label) = value_label[i];
                left_counts[label] += 1;
                right_counts[label] -= 1;
                left_n += 1;
                right_n -= 1;

                let next_val = value_label[i + 1].0;
                if (next_val - val).abs() < 1e-15 {
                    continue;
                }

                let threshold = (val + next_val) / 2.0;
                let left_gini = gini_from_counts(&left_counts, left_n);
                let right_gini = gini_from_counts(&right_counts, right_n);

                let weighted_gini =
                    (left_n as f64 / n) * left_gini + (right_n as f64 / n) * right_gini;
                let gain = parent_gini - weighted_gini;

                if gain > best_gain {
                    best_gain = gain;

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

                    best_result = Some((feat_idx, threshold, gain * n, left_idx, right_idx));
                }
            }
        }

        best_result
    }

    fn class_distribution(&self, labels: &[usize], indices: &[usize]) -> Vec<f64> {
        let mut counts = vec![0.0_f64; self.n_classes];
        let n = indices.len() as f64;
        if n == 0.0 {
            return counts;
        }
        for &i in indices {
            if labels[i] < self.n_classes {
                counts[labels[i]] += 1.0;
            }
        }
        for c in &mut counts {
            *c /= n;
        }
        counts
    }

    fn gini_impurity(&self, labels: &[usize], indices: &[usize]) -> f64 {
        let mut counts = vec![0usize; self.n_classes];
        for &i in indices {
            if labels[i] < self.n_classes {
                counts[labels[i]] += 1;
            }
        }
        gini_from_counts(&counts, indices.len())
    }

    pub fn predict_proba(&self, features: &[f64]) -> OracleResult<Vec<f64>> {
        match &self.root {
            Some(root) => Ok(root.predict_proba(features).to_vec()),
            None => Err(OracleError::model_not_trained("decision tree has no root")),
        }
    }
}

/// Compute Gini impurity from class counts.
fn gini_from_counts(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    1.0 - counts.iter().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

/// Random Forest classifier for decomposition method prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForest {
    pub params: RandomForestParams,
    pub trees: Vec<DecisionTree>,
    pub n_features: usize,
    pub n_classes: usize,
    pub feature_importances: Vec<f64>,
    pub oob_error: Option<f64>,
    pub oob_predictions: Option<Vec<Vec<f64>>>,
    trained: bool,
}

impl RandomForest {
    pub fn new(params: RandomForestParams) -> Self {
        Self {
            params,
            trees: Vec::new(),
            n_features: 0,
            n_classes: DecompositionMethod::n_classes(),
            feature_importances: Vec::new(),
            oob_error: None,
            oob_predictions: None,
            trained: false,
        }
    }

    /// Bootstrap sample indices from the dataset.
    fn bootstrap_sample(n: usize, rng: &mut ChaCha8Rng) -> (Vec<usize>, Vec<bool>) {
        let mut in_bag = vec![false; n];
        let mut sample = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sample.push(idx);
            in_bag[idx] = true;
        }
        (sample, in_bag)
    }

    /// Compute out-of-bag error.
    fn compute_oob_error(
        &self,
        features: &[FeatureVector],
        labels: &[usize],
        oob_masks: &[Vec<bool>],
    ) -> (f64, Vec<Vec<f64>>) {
        let n = features.len();
        let mut oob_votes = vec![vec![0.0_f64; self.n_classes]; n];
        let mut oob_counts = vec![0usize; n];

        for (tree_idx, tree) in self.trees.iter().enumerate() {
            for i in 0..n {
                if !oob_masks[tree_idx][i] {
                    // i was out-of-bag for this tree
                    if let Ok(proba) = tree.predict_proba(&features[i]) {
                        for (c, &p) in proba.iter().enumerate() {
                            oob_votes[i][c] += p;
                        }
                        oob_counts[i] += 1;
                    }
                }
            }
        }

        let mut correct = 0usize;
        let mut total = 0usize;
        for i in 0..n {
            if oob_counts[i] > 0 {
                let pred = oob_votes[i]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(c, _)| c)
                    .unwrap_or(0);
                if pred == labels[i] {
                    correct += 1;
                }
                total += 1;

                // Normalize to probabilities
                let sum: f64 = oob_votes[i].iter().sum();
                if sum > 0.0 {
                    for c in &mut oob_votes[i] {
                        *c /= sum;
                    }
                }
            }
        }

        let oob_accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        (1.0 - oob_accuracy, oob_votes)
    }

    /// Compute feature importances from all trees (Gini importance).
    fn compute_feature_importances(&self) -> Vec<f64> {
        let mut importances = vec![0.0_f64; self.n_features];

        for tree in &self.trees {
            if let Some(ref root) = tree.root {
                root.feature_importances(&mut importances);
            }
        }

        // Normalize
        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for imp in &mut importances {
                *imp /= total;
            }
        }

        importances
    }

    /// Get the out-of-bag score (1 - oob_error).
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_error.map(|e| 1.0 - e)
    }

    /// Get tree statistics.
    pub fn tree_stats(&self) -> TreeStats {
        if self.trees.is_empty() {
            return TreeStats {
                n_trees: 0,
                avg_depth: 0.0,
                avg_nodes: 0.0,
                max_depth: 0,
                min_depth: 0,
            };
        }
        let depths: Vec<usize> = self
            .trees
            .iter()
            .filter_map(|t| t.root.as_ref().map(|r| r.depth()))
            .collect();
        let nodes: Vec<usize> = self
            .trees
            .iter()
            .filter_map(|t| t.root.as_ref().map(|r| r.node_count()))
            .collect();

        TreeStats {
            n_trees: self.trees.len(),
            avg_depth: depths.iter().sum::<usize>() as f64 / depths.len().max(1) as f64,
            avg_nodes: nodes.iter().sum::<usize>() as f64 / nodes.len().max(1) as f64,
            max_depth: depths.iter().copied().max().unwrap_or(0),
            min_depth: depths.iter().copied().min().unwrap_or(0),
        }
    }
}

/// Tree statistics for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStats {
    pub n_trees: usize,
    pub avg_depth: f64,
    pub avg_nodes: f64,
    pub max_depth: usize,
    pub min_depth: usize,
}

impl Classifier for RandomForest {
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
        let max_features_per_split = self.params.max_features.resolve(self.n_features);

        let mut rng = match self.params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        self.trees.clear();
        let mut oob_masks = Vec::new();

        for _ in 0..self.params.n_trees {
            let (sample_indices, in_bag) = if self.params.bootstrap {
                Self::bootstrap_sample(features.len(), &mut rng)
            } else {
                let all: Vec<usize> = (0..features.len()).collect();
                let mask = vec![true; features.len()];
                (all, mask)
            };

            if self.params.oob_score {
                oob_masks.push(in_bag);
            }

            let boot_features: Vec<FeatureVector> =
                sample_indices.iter().map(|&i| features[i].clone()).collect();
            let boot_labels: Vec<usize> =
                sample_indices.iter().map(|&i| label_indices[i]).collect();

            let mut tree = DecisionTree::new(
                self.n_features,
                self.n_classes,
                self.params.max_depth,
                self.params.min_samples_split,
                self.params.min_samples_leaf,
                max_features_per_split,
            );

            let all_indices: Vec<usize> = (0..boot_features.len()).collect();
            tree.root = Some(tree.build_tree(
                &boot_features,
                &boot_labels,
                &all_indices,
                0,
                &mut rng,
            ));
            self.trees.push(tree);
        }

        // Compute OOB error
        if self.params.oob_score && self.params.bootstrap {
            let (oob_error, oob_preds) =
                self.compute_oob_error(features, &label_indices, &oob_masks);
            self.oob_error = Some(oob_error);
            self.oob_predictions = Some(oob_preds);
        }

        self.feature_importances = self.compute_feature_importances();
        self.trained = true;
        Ok(())
    }

    fn predict(&self, features: &FeatureVector) -> OracleResult<DecompositionMethod> {
        if !self.trained {
            return Err(OracleError::model_not_trained("random forest not trained"));
        }
        let proba = self.predict_proba(features)?;
        let best_class = proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        DecompositionMethod::from_index(best_class)
    }

    fn predict_proba(&self, features: &FeatureVector) -> OracleResult<Vec<f64>> {
        if !self.trained {
            return Err(OracleError::model_not_trained("random forest not trained"));
        }
        if features.len() != self.n_features {
            return Err(OracleError::invalid_input(format!(
                "expected {} features, got {}",
                self.n_features,
                features.len()
            )));
        }

        let mut avg_proba = vec![0.0_f64; self.n_classes];
        let mut count = 0;

        for tree in &self.trees {
            if let Ok(proba) = tree.predict_proba(features) {
                for (i, &p) in proba.iter().enumerate() {
                    if i < avg_proba.len() {
                        avg_proba[i] += p;
                    }
                }
                count += 1;
            }
        }

        if count > 0 {
            for p in &mut avg_proba {
                *p /= count as f64;
            }
        }

        Ok(avg_proba)
    }

    fn feature_importance(&self) -> OracleResult<Vec<f64>> {
        if !self.trained {
            return Err(OracleError::model_not_trained("random forest not trained"));
        }
        Ok(self.feature_importances.clone())
    }

    fn name(&self) -> &str {
        "RandomForest"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Permutation importance: measure prediction error increase when a feature is shuffled.
pub fn permutation_importance(
    model: &dyn Classifier,
    features: &[FeatureVector],
    labels: &[DecompositionMethod],
    n_repeats: usize,
    seed: u64,
) -> OracleResult<Vec<f64>> {
    if features.is_empty() {
        return Err(OracleError::invalid_input("empty features"));
    }

    let n_features = features[0].len();
    let base_metrics = model.evaluate(features, labels)?;
    let base_acc = base_metrics.accuracy;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut importances = vec![0.0_f64; n_features];

    for feat_idx in 0..n_features {
        let mut acc_drop_sum = 0.0;
        for _ in 0..n_repeats {
            let mut shuffled_features: Vec<FeatureVector> =
                features.iter().cloned().collect();
            let mut indices: Vec<usize> = (0..features.len()).collect();
            indices.shuffle(&mut rng);
            for (i, &shuffled_i) in indices.iter().enumerate() {
                shuffled_features[i][feat_idx] = features[shuffled_i][feat_idx];
            }
            let shuffled_metrics = model.evaluate(&shuffled_features, labels)?;
            acc_drop_sum += base_acc - shuffled_metrics.accuracy;
        }
        importances[feat_idx] = acc_drop_sum / n_repeats as f64;
    }

    Ok(importances)
}

/// Top-k feature indices by importance.
pub fn top_k_features(importances: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = importances.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Compute class weights inversely proportional to class frequency.
pub fn compute_class_weights(labels: &[DecompositionMethod]) -> HashMap<DecompositionMethod, f64> {
    let n = labels.len() as f64;
    let mut counts = HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0usize) += 1;
    }
    let n_classes = counts.len() as f64;
    let mut weights = HashMap::new();
    for (&class, &count) in &counts {
        weights.insert(class, n / (n_classes * count as f64));
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_separable_data(n: usize, seed: u64) -> (Vec<FeatureVector>, Vec<DecompositionMethod>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for _ in 0..n {
            let x: f64 = rng.gen_range(-2.0..2.0);
            let y: f64 = rng.gen_range(-2.0..2.0);
            let noise: f64 = rng.gen_range(-0.1..0.1);
            let label = if x + noise > 0.0 && y + noise > 0.0 {
                DecompositionMethod::Benders
            } else if x + noise > 0.0 {
                DecompositionMethod::DantzigWolfe
            } else if y + noise > 0.0 {
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
    fn test_gini_pure() {
        let counts = vec![10, 0, 0, 0];
        assert!((gini_from_counts(&counts, 10) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gini_uniform() {
        let counts = vec![5, 5];
        let gini = gini_from_counts(&counts, 10);
        assert!((gini - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gini_four_class() {
        let counts = vec![25, 25, 25, 25];
        let gini = gini_from_counts(&counts, 100);
        assert!((gini - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_max_features_sqrt() {
        assert_eq!(MaxFeatures::Sqrt.resolve(16), 4);
        assert_eq!(MaxFeatures::Sqrt.resolve(100), 10);
    }

    #[test]
    fn test_max_features_log2() {
        assert_eq!(MaxFeatures::Log2.resolve(16), 4);
    }

    #[test]
    fn test_max_features_fraction() {
        assert_eq!(MaxFeatures::Fraction(0.5).resolve(10), 5);
    }

    #[test]
    fn test_random_forest_train_predict() {
        let (features, labels) = make_separable_data(200, 42);
        let params = RandomForestParams {
            n_trees: 20,
            max_depth: Some(5),
            seed: Some(42),
            ..Default::default()
        };
        let mut rf = RandomForest::new(params);
        rf.train(&features, &labels).unwrap();

        assert!(rf.is_trained());
        let pred = rf.predict(&features[0]).unwrap();
        assert!(DecompositionMethod::all().contains(&pred));
    }

    #[test]
    fn test_random_forest_predict_proba() {
        let (features, labels) = make_separable_data(100, 42);
        let params = RandomForestParams {
            n_trees: 10,
            max_depth: Some(4),
            seed: Some(42),
            ..Default::default()
        };
        let mut rf = RandomForest::new(params);
        rf.train(&features, &labels).unwrap();

        let proba = rf.predict_proba(&features[0]).unwrap();
        assert_eq!(proba.len(), 4);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_random_forest_oob_score() {
        let (features, labels) = make_separable_data(200, 42);
        let params = RandomForestParams {
            n_trees: 30,
            max_depth: Some(6),
            oob_score: true,
            seed: Some(42),
            ..Default::default()
        };
        let mut rf = RandomForest::new(params);
        rf.train(&features, &labels).unwrap();

        assert!(rf.oob_score().is_some());
        let score = rf.oob_score().unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_random_forest_feature_importance() {
        let (features, labels) = make_separable_data(200, 42);
        let params = RandomForestParams {
            n_trees: 20,
            max_depth: Some(5),
            seed: Some(42),
            ..Default::default()
        };
        let mut rf = RandomForest::new(params);
        rf.train(&features, &labels).unwrap();

        let imp = rf.feature_importance().unwrap();
        assert_eq!(imp.len(), 2);
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_random_forest_untrained() {
        let rf = RandomForest::new(RandomForestParams::default());
        assert!(!rf.is_trained());
        assert!(rf.predict(&vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn test_random_forest_evaluate() {
        let (features, labels) = make_separable_data(200, 42);
        let params = RandomForestParams {
            n_trees: 30,
            max_depth: Some(6),
            seed: Some(42),
            ..Default::default()
        };
        let mut rf = RandomForest::new(params);
        rf.train(&features, &labels).unwrap();
        let metrics = rf.evaluate(&features, &labels).unwrap();
        assert!(metrics.accuracy > 0.5);
    }

    #[test]
    fn test_tree_stats() {
        let (features, labels) = make_separable_data(100, 42);
        let params = RandomForestParams {
            n_trees: 5,
            max_depth: Some(4),
            seed: Some(42),
            ..Default::default()
        };
        let mut rf = RandomForest::new(params);
        rf.train(&features, &labels).unwrap();
        let stats = rf.tree_stats();
        assert_eq!(stats.n_trees, 5);
        assert!(stats.avg_depth > 0.0);
    }

    #[test]
    fn test_bootstrap_sample() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (sample, in_bag) = RandomForest::bootstrap_sample(100, &mut rng);
        assert_eq!(sample.len(), 100);
        assert!(in_bag.iter().any(|&b| !b)); // Some out-of-bag
    }

    #[test]
    fn test_top_k_features() {
        let importances = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let top = top_k_features(&importances, 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_class_weights() {
        let labels = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        let weights = compute_class_weights(&labels);
        assert!(weights[&DecompositionMethod::DantzigWolfe] > weights[&DecompositionMethod::Benders]);
    }

    #[test]
    fn test_decision_tree_depth() {
        let leaf = TreeNode::Leaf {
            class_distribution: vec![1.0],
            predicted_class: 0,
            n_samples: 1,
        };
        assert_eq!(leaf.depth(), 0);
        assert_eq!(leaf.node_count(), 1);
    }
}
