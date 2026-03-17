// Ensemble methods: voting, stacking, and automatic weight optimization.

use crate::classifier::traits::{
    Classifier, DecompositionMethod, FeatureVector,
};
use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};

/// Voting strategy for ensemble classifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingStrategy {
    Hard,
    Soft,
}

/// Voting classifier combining multiple base classifiers.
pub struct VotingClassifier {
    pub classifiers: Vec<Box<dyn Classifier>>,
    pub weights: Vec<f64>,
    pub strategy: VotingStrategy,
    pub n_features: usize,
    trained: bool,
}

impl VotingClassifier {
    pub fn new(strategy: VotingStrategy) -> Self {
        Self {
            classifiers: Vec::new(),
            weights: Vec::new(),
            strategy,
            n_features: 0,
            trained: false,
        }
    }

    pub fn add_classifier(&mut self, classifier: Box<dyn Classifier>, weight: f64) {
        self.classifiers.push(classifier);
        self.weights.push(weight);
    }

    /// Optimize weights based on cross-validation performance.
    pub fn optimize_weights(
        &mut self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
        n_folds: usize,
    ) -> OracleResult<()> {
        if self.classifiers.is_empty() {
            return Err(OracleError::invalid_input("no classifiers added"));
        }
        if features.len() < n_folds {
            return Err(OracleError::invalid_input("too few samples for CV"));
        }

        let n = features.len();
        let fold_size = n / n_folds;

        // Evaluate each classifier individually
        let mut cv_scores = vec![0.0_f64; self.classifiers.len()];

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
            let val_labs: Vec<DecompositionMethod> =
                (val_start..val_end).map(|i| labels[i]).collect();

            for (clf_idx, clf) in self.classifiers.iter_mut().enumerate() {
                let mut temp_score = 0.0;
                if clf.train(&train_feats, &train_labs).is_ok() {
                    if let Ok(metrics) = clf.evaluate(&val_feats, &val_labs) {
                        temp_score = metrics.accuracy;
                    }
                }
                cv_scores[clf_idx] += temp_score;
            }
        }

        // Normalize to get weights
        for score in &mut cv_scores {
            *score /= n_folds as f64;
        }
        let total: f64 = cv_scores.iter().sum();
        if total > 0.0 {
            self.weights = cv_scores.iter().map(|&s| s / total).collect();
        }

        Ok(())
    }

    /// Compute diversity between classifiers (disagreement rate).
    pub fn diversity_score(
        &self,
        features: &[FeatureVector],
    ) -> OracleResult<f64> {
        if self.classifiers.len() < 2 || features.is_empty() {
            return Ok(0.0);
        }

        let n = features.len();
        let k = self.classifiers.len();
        let mut predictions: Vec<Vec<DecompositionMethod>> = Vec::new();

        for clf in &self.classifiers {
            let preds: Vec<DecompositionMethod> = features
                .iter()
                .map(|f| clf.predict(f).unwrap_or(DecompositionMethod::None))
                .collect();
            predictions.push(preds);
        }

        let mut disagreements = 0usize;
        let mut pairs = 0usize;

        for i in 0..k {
            for j in (i + 1)..k {
                for s in 0..n {
                    if predictions[i][s] != predictions[j][s] {
                        disagreements += 1;
                    }
                }
                pairs += n;
            }
        }

        Ok(if pairs > 0 {
            disagreements as f64 / pairs as f64
        } else {
            0.0
        })
    }
}

impl Classifier for VotingClassifier {
    fn train(
        &mut self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
    ) -> OracleResult<()> {
        if self.classifiers.is_empty() {
            return Err(OracleError::invalid_input("no classifiers added"));
        }
        self.n_features = features.first().map(|f| f.len()).unwrap_or(0);

        for clf in &mut self.classifiers {
            clf.train(features, labels)?;
        }
        self.trained = true;
        Ok(())
    }

    fn predict(&self, features: &FeatureVector) -> OracleResult<DecompositionMethod> {
        if !self.trained {
            return Err(OracleError::model_not_trained("voting classifier not trained"));
        }

        match self.strategy {
            VotingStrategy::Hard => {
                let n_classes = DecompositionMethod::n_classes();
                let mut votes = vec![0.0_f64; n_classes];

                for (clf, &weight) in self.classifiers.iter().zip(self.weights.iter()) {
                    if let Ok(pred) = clf.predict(features) {
                        votes[pred.to_index()] += weight;
                    }
                }

                let best = votes
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                DecompositionMethod::from_index(best)
            }
            VotingStrategy::Soft => {
                let proba = self.predict_proba(features)?;
                let best = proba
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                DecompositionMethod::from_index(best)
            }
        }
    }

    fn predict_proba(&self, features: &FeatureVector) -> OracleResult<Vec<f64>> {
        if !self.trained {
            return Err(OracleError::model_not_trained("voting classifier not trained"));
        }

        let n_classes = DecompositionMethod::n_classes();
        let mut avg_proba = vec![0.0_f64; n_classes];
        let mut total_weight = 0.0;

        for (clf, &weight) in self.classifiers.iter().zip(self.weights.iter()) {
            if let Ok(proba) = clf.predict_proba(features) {
                for (i, &p) in proba.iter().enumerate() {
                    avg_proba[i] += weight * p;
                }
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            for p in &mut avg_proba {
                *p /= total_weight;
            }
        }

        Ok(avg_proba)
    }

    fn feature_importance(&self) -> OracleResult<Vec<f64>> {
        if !self.trained || self.classifiers.is_empty() {
            return Err(OracleError::model_not_trained("voting classifier not trained"));
        }

        let n_features = self.n_features;
        let mut avg_imp = vec![0.0_f64; n_features];
        let mut count = 0;

        for (clf, &weight) in self.classifiers.iter().zip(self.weights.iter()) {
            if let Ok(imp) = clf.feature_importance() {
                for (i, &v) in imp.iter().enumerate() {
                    if i < avg_imp.len() {
                        avg_imp[i] += weight * v;
                    }
                }
                count += 1;
            }
        }

        if count > 0 {
            let total: f64 = avg_imp.iter().sum();
            if total > 0.0 {
                for v in &mut avg_imp {
                    *v /= total;
                }
            }
        }

        Ok(avg_imp)
    }

    fn name(&self) -> &str {
        match self.strategy {
            VotingStrategy::Hard => "VotingClassifier(hard)",
            VotingStrategy::Soft => "VotingClassifier(soft)",
        }
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Stacking classifier: use base classifier outputs as features for a meta-learner.
pub struct StackingClassifier {
    pub base_classifiers: Vec<Box<dyn Classifier>>,
    pub meta_learner: Box<dyn Classifier>,
    pub use_probas: bool,
    pub n_features: usize,
    trained: bool,
}

impl StackingClassifier {
    pub fn new(meta_learner: Box<dyn Classifier>, use_probas: bool) -> Self {
        Self {
            base_classifiers: Vec::new(),
            meta_learner,
            use_probas,
            n_features: 0,
            trained: false,
        }
    }

    pub fn add_base(&mut self, classifier: Box<dyn Classifier>) {
        self.base_classifiers.push(classifier);
    }

    /// Generate meta-features from base classifier predictions.
    fn generate_meta_features(&self, features: &FeatureVector) -> OracleResult<FeatureVector> {
        let mut meta_features = Vec::new();

        for clf in &self.base_classifiers {
            if self.use_probas {
                let proba = clf.predict_proba(features)?;
                meta_features.extend(proba);
            } else {
                let pred = clf.predict(features)?;
                let n_classes = DecompositionMethod::n_classes();
                let mut one_hot = vec![0.0; n_classes];
                one_hot[pred.to_index()] = 1.0;
                meta_features.extend(one_hot);
            }
        }

        // Also include original features
        meta_features.extend(features.iter());

        Ok(meta_features)
    }

    /// Generate stacked training data using cross-validation predictions.
    fn generate_stacked_train_data(
        &mut self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
        n_folds: usize,
    ) -> OracleResult<Vec<FeatureVector>> {
        let n = features.len();
        let fold_size = n / n_folds.max(1);
        let n_classes = DecompositionMethod::n_classes();

        // Initialize out-of-fold predictions
        let n_base = self.base_classifiers.len();
        let meta_dim = if self.use_probas {
            n_base * n_classes + features[0].len()
        } else {
            n_base * n_classes + features[0].len()
        };
        let mut meta_features = vec![vec![0.0_f64; meta_dim]; n];

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

            for (clf_idx, clf) in self.base_classifiers.iter_mut().enumerate() {
                clf.train(&train_feats, &train_labs)?;

                for i in val_start..val_end {
                    let offset = clf_idx * n_classes;
                    if self.use_probas {
                        if let Ok(proba) = clf.predict_proba(&features[i]) {
                            for (c, &p) in proba.iter().enumerate() {
                                meta_features[i][offset + c] = p;
                            }
                        }
                    } else {
                        if let Ok(pred) = clf.predict(&features[i]) {
                            meta_features[i][offset + pred.to_index()] = 1.0;
                        }
                    }
                }
            }

            // Add original features
            for i in val_start..val_end {
                let feat_offset = n_base * n_classes;
                for (j, &v) in features[i].iter().enumerate() {
                    meta_features[i][feat_offset + j] = v;
                }
            }
        }

        Ok(meta_features)
    }
}

impl Classifier for StackingClassifier {
    fn train(
        &mut self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
    ) -> OracleResult<()> {
        if self.base_classifiers.is_empty() {
            return Err(OracleError::invalid_input("no base classifiers added"));
        }
        self.n_features = features.first().map(|f| f.len()).unwrap_or(0);

        // Generate stacked training data
        let n_folds = 3.min(features.len());
        let meta_features = self.generate_stacked_train_data(features, labels, n_folds)?;

        // Re-train all base classifiers on full data
        for clf in &mut self.base_classifiers {
            clf.train(features, labels)?;
        }

        // Train meta-learner
        self.meta_learner.train(&meta_features, labels)?;

        self.trained = true;
        Ok(())
    }

    fn predict(&self, features: &FeatureVector) -> OracleResult<DecompositionMethod> {
        if !self.trained {
            return Err(OracleError::model_not_trained("stacking not trained"));
        }
        let meta_features = self.generate_meta_features(features)?;
        self.meta_learner.predict(&meta_features)
    }

    fn predict_proba(&self, features: &FeatureVector) -> OracleResult<Vec<f64>> {
        if !self.trained {
            return Err(OracleError::model_not_trained("stacking not trained"));
        }
        let meta_features = self.generate_meta_features(features)?;
        self.meta_learner.predict_proba(&meta_features)
    }

    fn feature_importance(&self) -> OracleResult<Vec<f64>> {
        self.meta_learner.feature_importance()
    }

    fn name(&self) -> &str {
        "StackingClassifier"
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Compute pairwise diversity between two sets of predictions.
pub fn pairwise_disagreement(
    preds_a: &[DecompositionMethod],
    preds_b: &[DecompositionMethod],
) -> f64 {
    if preds_a.len() != preds_b.len() || preds_a.is_empty() {
        return 0.0;
    }
    let disagreements = preds_a
        .iter()
        .zip(preds_b.iter())
        .filter(|(a, b)| a != b)
        .count();
    disagreements as f64 / preds_a.len() as f64
}

/// Q-statistic for measuring diversity between two classifiers.
pub fn q_statistic(
    preds_a: &[DecompositionMethod],
    preds_b: &[DecompositionMethod],
    true_labels: &[DecompositionMethod],
) -> f64 {
    if preds_a.len() != preds_b.len() || preds_a.len() != true_labels.len() || preds_a.is_empty() {
        return 0.0;
    }

    let mut n11 = 0.0_f64; // both correct
    let mut n00 = 0.0_f64; // both wrong
    let mut n10 = 0.0_f64; // a correct, b wrong
    let mut n01 = 0.0_f64; // a wrong, b correct

    for i in 0..preds_a.len() {
        let a_correct = preds_a[i] == true_labels[i];
        let b_correct = preds_b[i] == true_labels[i];
        match (a_correct, b_correct) {
            (true, true) => n11 += 1.0,
            (false, false) => n00 += 1.0,
            (true, false) => n10 += 1.0,
            (false, true) => n01 += 1.0,
        }
    }

    let denom = n11 * n00 + n10 * n01;
    if denom.abs() < 1e-12 {
        return 0.0;
    }

    (n11 * n00 - n10 * n01) / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::random_forest::{RandomForest, RandomForestParams};
    use crate::classifier::logistic::{LogisticRegression, LogisticRegressionParams};
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
    fn test_voting_hard() {
        let (features, labels) = make_data(200, 42);

        let rf = RandomForest::new(RandomForestParams {
            n_trees: 10,
            max_depth: Some(4),
            seed: Some(42),
            ..Default::default()
        });
        let lr = LogisticRegression::new(LogisticRegressionParams {
            max_iter: 100,
            ..Default::default()
        });

        let mut voting = VotingClassifier::new(VotingStrategy::Hard);
        voting.add_classifier(Box::new(rf), 1.0);
        voting.add_classifier(Box::new(lr), 1.0);
        voting.train(&features, &labels).unwrap();

        let pred = voting.predict(&features[0]).unwrap();
        assert!(DecompositionMethod::all().contains(&pred));
    }

    #[test]
    fn test_voting_soft() {
        let (features, labels) = make_data(200, 42);

        let rf = RandomForest::new(RandomForestParams {
            n_trees: 10,
            max_depth: Some(4),
            seed: Some(42),
            ..Default::default()
        });

        let mut voting = VotingClassifier::new(VotingStrategy::Soft);
        voting.add_classifier(Box::new(rf), 1.0);
        voting.train(&features, &labels).unwrap();

        let proba = voting.predict_proba(&features[0]).unwrap();
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_voting_untrained() {
        let voting = VotingClassifier::new(VotingStrategy::Hard);
        assert!(!voting.is_trained());
    }

    #[test]
    fn test_voting_no_classifiers() {
        let mut voting = VotingClassifier::new(VotingStrategy::Hard);
        assert!(voting.train(&[vec![1.0]], &[DecompositionMethod::Benders]).is_err());
    }

    #[test]
    fn test_diversity_score() {
        let (features, labels) = make_data(100, 42);

        let rf1 = RandomForest::new(RandomForestParams {
            n_trees: 5,
            max_depth: Some(3),
            seed: Some(1),
            ..Default::default()
        });
        let rf2 = RandomForest::new(RandomForestParams {
            n_trees: 5,
            max_depth: Some(3),
            seed: Some(99),
            ..Default::default()
        });

        let mut voting = VotingClassifier::new(VotingStrategy::Hard);
        voting.add_classifier(Box::new(rf1), 1.0);
        voting.add_classifier(Box::new(rf2), 1.0);
        voting.train(&features, &labels).unwrap();

        let diversity = voting.diversity_score(&features).unwrap();
        assert!(diversity >= 0.0 && diversity <= 1.0);
    }

    #[test]
    fn test_pairwise_disagreement() {
        let a = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
            DecompositionMethod::Benders,
            DecompositionMethod::None,
        ];
        let b = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::None,
        ];
        let d = pairwise_disagreement(&a, &b);
        assert!((d - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_q_statistic_identical() {
        let preds = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        let labels = preds.clone();
        let q = q_statistic(&preds, &preds, &labels);
        // Both always correct: n11 = 2, all others 0
        // Q = (2*0 - 0*0) / (2*0 + 0*0) = 0/0 -> 0
        assert!(q.is_finite());
    }

    #[test]
    fn test_stacking_basic() {
        let (features, labels) = make_data(100, 42);

        let rf = RandomForest::new(RandomForestParams {
            n_trees: 5,
            max_depth: Some(3),
            seed: Some(42),
            ..Default::default()
        });

        let meta = LogisticRegression::new(LogisticRegressionParams {
            max_iter: 50,
            ..Default::default()
        });

        let mut stacking = StackingClassifier::new(Box::new(meta), true);
        stacking.add_base(Box::new(rf));
        stacking.train(&features, &labels).unwrap();

        let pred = stacking.predict(&features[0]).unwrap();
        assert!(DecompositionMethod::all().contains(&pred));
    }

    #[test]
    fn test_pairwise_empty() {
        let d = pairwise_disagreement(&[], &[]);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_pairwise_mismatched_lengths() {
        let a = vec![DecompositionMethod::Benders];
        let b = vec![DecompositionMethod::Benders, DecompositionMethod::None];
        let d = pairwise_disagreement(&a, &b);
        assert_eq!(d, 0.0);
    }
}
