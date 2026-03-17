// Cross-validation: stratified k-fold, nested CV, and hyperparameter tuning.

use crate::classifier::traits::{
    Classifier, ClassificationMetrics, DecompositionMethod, FeatureVector,
};
use crate::error::{OracleError, OracleResult};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResults {
    pub fold_metrics: Vec<ClassificationMetrics>,
    pub mean_accuracy: f64,
    pub std_accuracy: f64,
    pub mean_macro_f1: f64,
    pub std_macro_f1: f64,
    pub mean_kappa: f64,
    pub n_folds: usize,
}

impl CVResults {
    pub fn from_fold_metrics(metrics: Vec<ClassificationMetrics>) -> Self {
        let n = metrics.len() as f64;
        let accuracies: Vec<f64> = metrics.iter().map(|m| m.accuracy).collect();
        let f1s: Vec<f64> = metrics.iter().map(|m| m.macro_f1).collect();
        let kappas: Vec<f64> = metrics.iter().map(|m| m.cohen_kappa).collect();

        let mean_acc = accuracies.iter().sum::<f64>() / n;
        let std_acc = if n > 1.0 {
            (accuracies.iter().map(|&a| (a - mean_acc).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        let mean_f1 = f1s.iter().sum::<f64>() / n;
        let std_f1 = if n > 1.0 {
            (f1s.iter().map(|&f| (f - mean_f1).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        let mean_kappa = kappas.iter().sum::<f64>() / n;

        Self {
            n_folds: metrics.len(),
            fold_metrics: metrics,
            mean_accuracy: mean_acc,
            std_accuracy: std_acc,
            mean_macro_f1: mean_f1,
            std_macro_f1: std_f1,
            mean_kappa,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "CV({} folds): acc={:.4}±{:.4}, F1={:.4}±{:.4}, κ={:.4}",
            self.n_folds, self.mean_accuracy, self.std_accuracy,
            self.mean_macro_f1, self.std_macro_f1, self.mean_kappa
        )
    }
}

/// Stratified k-fold split generator.
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    pub n_folds: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl StratifiedKFold {
    pub fn new(n_folds: usize, seed: u64) -> Self {
        Self {
            n_folds,
            shuffle: true,
            seed,
        }
    }

    /// Generate fold indices: returns Vec of (train_indices, test_indices).
    pub fn split(
        &self,
        labels: &[DecompositionMethod],
    ) -> OracleResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let n = labels.len();
        if n < self.n_folds {
            return Err(OracleError::invalid_input(format!(
                "cannot create {} folds from {} samples",
                self.n_folds, n
            )));
        }

        // Group indices by class
        let mut class_indices: HashMap<DecompositionMethod, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            class_indices.entry(label).or_default().push(i);
        }

        // Shuffle within each class if requested
        if self.shuffle {
            let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
            for indices in class_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Assign each sample to a fold (stratified)
        let mut fold_assignments = vec![0usize; n];
        for indices in class_indices.values() {
            for (i, &idx) in indices.iter().enumerate() {
                fold_assignments[idx] = i % self.n_folds;
            }
        }

        // Generate train/test splits for each fold
        let mut folds = Vec::new();
        for fold in 0..self.n_folds {
            let test_idx: Vec<usize> = (0..n).filter(|&i| fold_assignments[i] == fold).collect();
            let train_idx: Vec<usize> = (0..n).filter(|&i| fold_assignments[i] != fold).collect();
            folds.push((train_idx, test_idx));
        }

        Ok(folds)
    }
}

/// Leave-one-out cross-validation split.
pub fn leave_one_out(n: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    (0..n)
        .map(|i| {
            let train: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            let test = vec![i];
            (train, test)
        })
        .collect()
}

/// Repeated stratified k-fold.
pub fn repeated_stratified_kfold(
    labels: &[DecompositionMethod],
    n_folds: usize,
    n_repeats: usize,
    base_seed: u64,
) -> OracleResult<Vec<(Vec<usize>, Vec<usize>)>> {
    let mut all_folds = Vec::new();
    for rep in 0..n_repeats {
        let skf = StratifiedKFold::new(n_folds, base_seed + rep as u64);
        let folds = skf.split(labels)?;
        all_folds.extend(folds);
    }
    Ok(all_folds)
}

/// Nested cross-validation with hyperparameter tuning.
pub struct NestedCV {
    pub outer_folds: usize,
    pub inner_folds: usize,
    pub seed: u64,
}

impl NestedCV {
    pub fn new(outer_folds: usize, inner_folds: usize, seed: u64) -> Self {
        Self {
            outer_folds,
            inner_folds,
            seed,
        }
    }

    /// Run nested CV with a classifier factory and hyperparameter grid.
    pub fn evaluate<F>(
        &self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
        classifier_factory: F,
        param_grid: &[HashMap<String, f64>],
    ) -> OracleResult<NestedCVResults>
    where
        F: Fn(&HashMap<String, f64>) -> Box<dyn Classifier>,
    {
        let outer_skf = StratifiedKFold::new(self.outer_folds, self.seed);
        let outer_folds = outer_skf.split(labels)?;

        let mut outer_metrics = Vec::new();
        let mut best_params_per_fold = Vec::new();

        for (fold_idx, (train_idx, test_idx)) in outer_folds.iter().enumerate() {
            let train_features: Vec<FeatureVector> =
                train_idx.iter().map(|&i| features[i].clone()).collect();
            let train_labels: Vec<DecompositionMethod> =
                train_idx.iter().map(|&i| labels[i]).collect();
            let test_features: Vec<FeatureVector> =
                test_idx.iter().map(|&i| features[i].clone()).collect();
            let test_labels: Vec<DecompositionMethod> =
                test_idx.iter().map(|&i| labels[i]).collect();

            // Inner CV for hyperparameter tuning
            let best_params = self.inner_cv_select(
                &train_features,
                &train_labels,
                &classifier_factory,
                param_grid,
                fold_idx,
            )?;

            // Train on full outer train set with best params
            let mut clf = classifier_factory(&best_params);
            clf.train(&train_features, &train_labels)?;

            // Evaluate on outer test set
            let metrics = clf.evaluate(&test_features, &test_labels)?;
            outer_metrics.push(metrics);
            best_params_per_fold.push(best_params);
        }

        let cv_results = CVResults::from_fold_metrics(outer_metrics);

        Ok(NestedCVResults {
            cv_results,
            best_params_per_fold,
        })
    }

    fn inner_cv_select<F>(
        &self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
        classifier_factory: &F,
        param_grid: &[HashMap<String, f64>],
        outer_fold: usize,
    ) -> OracleResult<HashMap<String, f64>>
    where
        F: Fn(&HashMap<String, f64>) -> Box<dyn Classifier>,
    {
        if param_grid.is_empty() {
            return Ok(HashMap::new());
        }

        let inner_skf =
            StratifiedKFold::new(self.inner_folds, self.seed + 1000 + outer_fold as u64);
        let inner_folds = inner_skf.split(labels)?;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = param_grid[0].clone();

        for params in param_grid {
            let mut fold_scores = Vec::new();

            for (inner_train, inner_test) in &inner_folds {
                let inner_train_feat: Vec<FeatureVector> =
                    inner_train.iter().map(|&i| features[i].clone()).collect();
                let inner_train_lab: Vec<DecompositionMethod> =
                    inner_train.iter().map(|&i| labels[i]).collect();
                let inner_test_feat: Vec<FeatureVector> =
                    inner_test.iter().map(|&i| features[i].clone()).collect();
                let inner_test_lab: Vec<DecompositionMethod> =
                    inner_test.iter().map(|&i| labels[i]).collect();

                let mut clf = classifier_factory(params);
                if clf.train(&inner_train_feat, &inner_train_lab).is_ok() {
                    if let Ok(metrics) = clf.evaluate(&inner_test_feat, &inner_test_lab) {
                        fold_scores.push(metrics.accuracy);
                    }
                }
            }

            let mean_score = if fold_scores.is_empty() {
                0.0
            } else {
                fold_scores.iter().sum::<f64>() / fold_scores.len() as f64
            };

            if mean_score > best_score {
                best_score = mean_score;
                best_params = params.clone();
            }
        }

        Ok(best_params)
    }
}

/// Results from nested cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedCVResults {
    pub cv_results: CVResults,
    pub best_params_per_fold: Vec<HashMap<String, f64>>,
}

impl NestedCVResults {
    pub fn summary(&self) -> String {
        format!(
            "Nested CV: {}\n  Best params consistency: {} unique configs across {} folds",
            self.cv_results.summary(),
            self.unique_param_configs(),
            self.best_params_per_fold.len()
        )
    }

    pub fn unique_param_configs(&self) -> usize {
        let mut seen = Vec::new();
        for params in &self.best_params_per_fold {
            let mut sorted: Vec<(String, i64)> = params
                .iter()
                .map(|(k, v)| (k.clone(), (v * 1000.0) as i64))
                .collect();
            sorted.sort();
            if !seen.contains(&sorted) {
                seen.push(sorted);
            }
        }
        seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_labels(n: usize) -> Vec<DecompositionMethod> {
        (0..n)
            .map(|i| match i % 4 {
                0 => DecompositionMethod::Benders,
                1 => DecompositionMethod::DantzigWolfe,
                2 => DecompositionMethod::Lagrangian,
                _ => DecompositionMethod::None,
            })
            .collect()
    }

    #[test]
    fn test_stratified_kfold_basic() {
        let labels = make_labels(100);
        let skf = StratifiedKFold::new(5, 42);
        let folds = skf.split(&labels).unwrap();
        assert_eq!(folds.len(), 5);

        for (train, test) in &folds {
            assert!(!train.is_empty());
            assert!(!test.is_empty());
            assert_eq!(train.len() + test.len(), 100);
        }
    }

    #[test]
    fn test_stratified_kfold_no_overlap() {
        let labels = make_labels(20);
        let skf = StratifiedKFold::new(4, 42);
        let folds = skf.split(&labels).unwrap();

        for (train, test) in &folds {
            let train_set: std::collections::HashSet<usize> = train.iter().copied().collect();
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            assert_eq!(train_set.intersection(&test_set).count(), 0);
        }
    }

    #[test]
    fn test_stratified_kfold_too_few_samples() {
        let labels = vec![DecompositionMethod::Benders; 2];
        let skf = StratifiedKFold::new(5, 42);
        assert!(skf.split(&labels).is_err());
    }

    #[test]
    fn test_leave_one_out() {
        let folds = leave_one_out(5);
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.len(), 4);
            assert_eq!(test.len(), 1);
        }
    }

    #[test]
    fn test_repeated_stratified() {
        let labels = make_labels(20);
        let folds = repeated_stratified_kfold(&labels, 4, 3, 42).unwrap();
        assert_eq!(folds.len(), 12); // 4 folds * 3 repeats
    }

    #[test]
    fn test_cv_results_from_metrics() {
        let metrics = vec![
            ClassificationMetrics::compute(
                &[DecompositionMethod::Benders, DecompositionMethod::DantzigWolfe],
                &[DecompositionMethod::Benders, DecompositionMethod::DantzigWolfe],
            ),
            ClassificationMetrics::compute(
                &[DecompositionMethod::Benders, DecompositionMethod::Benders],
                &[DecompositionMethod::Benders, DecompositionMethod::DantzigWolfe],
            ),
        ];
        let cv = CVResults::from_fold_metrics(metrics);
        assert_eq!(cv.n_folds, 2);
        assert!(cv.mean_accuracy > 0.0);
    }

    #[test]
    fn test_cv_results_summary() {
        let metrics = vec![ClassificationMetrics::compute(
            &[DecompositionMethod::Benders],
            &[DecompositionMethod::Benders],
        )];
        let cv = CVResults::from_fold_metrics(metrics);
        let summary = cv.summary();
        assert!(summary.contains("CV"));
    }

    #[test]
    fn test_nested_cv_basic() {
        use crate::classifier::random_forest::{RandomForest, RandomForestParams};

        let features: Vec<FeatureVector> = (0..40)
            .map(|i| vec![i as f64, (i * 2) as f64])
            .collect();
        let labels = make_labels(40);

        let nested = NestedCV::new(3, 2, 42);
        let param_grid = vec![{
            let mut m = HashMap::new();
            m.insert("n_trees".to_string(), 10.0);
            m
        }];

        let factory = |_params: &HashMap<String, f64>| -> Box<dyn Classifier> {
            Box::new(RandomForest::new(RandomForestParams {
                n_trees: 10,
                max_depth: Some(3),
                seed: Some(42),
                ..Default::default()
            }))
        };

        let results = nested.evaluate(&features, &labels, factory, &param_grid).unwrap();
        assert_eq!(results.cv_results.n_folds, 3);
    }

    #[test]
    fn test_nested_cv_results_summary() {
        let cv = CVResults {
            fold_metrics: vec![],
            mean_accuracy: 0.8,
            std_accuracy: 0.05,
            mean_macro_f1: 0.75,
            std_macro_f1: 0.06,
            mean_kappa: 0.7,
            n_folds: 5,
        };
        let nested = NestedCVResults {
            cv_results: cv,
            best_params_per_fold: vec![HashMap::new(); 5],
        };
        let summary = nested.summary();
        assert!(summary.contains("Nested CV"));
    }

    #[test]
    fn test_unique_param_configs() {
        let nested = NestedCVResults {
            cv_results: CVResults {
                fold_metrics: vec![],
                mean_accuracy: 0.8,
                std_accuracy: 0.05,
                mean_macro_f1: 0.75,
                std_macro_f1: 0.06,
                mean_kappa: 0.7,
                n_folds: 3,
            },
            best_params_per_fold: vec![
                {
                    let mut m = HashMap::new();
                    m.insert("n_trees".to_string(), 100.0);
                    m
                },
                {
                    let mut m = HashMap::new();
                    m.insert("n_trees".to_string(), 100.0);
                    m
                },
                {
                    let mut m = HashMap::new();
                    m.insert("n_trees".to_string(), 200.0);
                    m
                },
            ],
        };
        assert_eq!(nested.unique_param_configs(), 2);
    }
}
