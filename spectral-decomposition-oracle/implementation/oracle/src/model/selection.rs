// Model selection: compare classifiers, AutoML-lite, and model ranking.

use crate::classifier::traits::{
    Classifier, DecompositionMethod, FeatureVector,
};
use crate::classifier::random_forest::{RandomForest, RandomForestParams};
use crate::classifier::gradient_boosting::{GradientBoostingClassifier, GradientBoostingParams};
use crate::classifier::logistic::{LogisticRegression, LogisticRegressionParams};
use crate::evaluation::cross_validation::{StratifiedKFold, CVResults};
use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for AutoML-lite model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLConfig {
    pub n_folds: usize,
    pub seed: u64,
    pub include_rf: bool,
    pub include_gbt: bool,
    pub include_logreg: bool,
    pub try_default_params: bool,
    pub try_tuned_params: bool,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            seed: 42,
            include_rf: true,
            include_gbt: true,
            include_logreg: true,
            try_default_params: true,
            try_tuned_params: true,
        }
    }
}

/// Comparison result for a single model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_name: String,
    pub params_description: String,
    pub cv_results: CVResults,
    pub rank: usize,
}

impl ModelComparison {
    pub fn summary(&self) -> String {
        format!(
            "#{} {:<25} acc={:.4}±{:.4} F1={:.4}±{:.4}",
            self.rank,
            self.model_name,
            self.cv_results.mean_accuracy,
            self.cv_results.std_accuracy,
            self.cv_results.mean_macro_f1,
            self.cv_results.std_macro_f1
        )
    }
}

/// Result of model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub comparisons: Vec<ModelComparison>,
    pub best_model_name: String,
    pub best_cv_accuracy: f64,
    pub best_cv_f1: f64,
}

impl SelectionResult {
    pub fn report(&self) -> String {
        let mut s = format!(
            "Model Selection Report (best: {})\n\n",
            self.best_model_name
        );
        for comp in &self.comparisons {
            s.push_str(&comp.summary());
            s.push('\n');
        }
        s
    }
}

/// Model selector: run AutoML-lite to find the best classifier.
pub struct ModelSelector {
    pub config: AutoMLConfig,
}

impl ModelSelector {
    pub fn new(config: AutoMLConfig) -> Self {
        Self { config }
    }

    /// Run model selection on the given data.
    pub fn select(
        &self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
    ) -> OracleResult<SelectionResult> {
        if features.is_empty() || features.len() != labels.len() {
            return Err(OracleError::invalid_input("invalid data for model selection"));
        }

        let mut comparisons = Vec::new();

        // Generate model configurations
        let configs = self.generate_configurations();

        let skf = StratifiedKFold::new(self.config.n_folds, self.config.seed);
        let folds = skf.split(labels)?;

        for (name, factory) in &configs {
            match self.evaluate_model(features, labels, &folds, factory) {
                Ok(cv_results) => {
                    comparisons.push(ModelComparison {
                        model_name: name.clone(),
                        params_description: name.clone(),
                        cv_results,
                        rank: 0,
                    });
                }
                Err(e) => {
                    log::warn!("Model {} failed: {}", name, e);
                }
            }
        }

        // Sort by mean accuracy descending
        comparisons.sort_by(|a, b| {
            b.cv_results
                .mean_accuracy
                .partial_cmp(&a.cv_results.mean_accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, comp) in comparisons.iter_mut().enumerate() {
            comp.rank = i + 1;
        }

        let best = comparisons.first();
        let best_name = best
            .map(|c| c.model_name.clone())
            .unwrap_or_else(|| "None".to_string());
        let best_acc = best.map(|c| c.cv_results.mean_accuracy).unwrap_or(0.0);
        let best_f1 = best.map(|c| c.cv_results.mean_macro_f1).unwrap_or(0.0);

        Ok(SelectionResult {
            comparisons,
            best_model_name: best_name,
            best_cv_accuracy: best_acc,
            best_cv_f1: best_f1,
        })
    }

    fn generate_configurations(&self) -> Vec<(String, Box<dyn Fn() -> Box<dyn Classifier> + '_>)> {
        let mut configs: Vec<(String, Box<dyn Fn() -> Box<dyn Classifier> + '_>)> = Vec::new();

        if self.config.include_rf {
            if self.config.try_default_params {
                configs.push((
                    "RF-default".to_string(),
                    Box::new(|| {
                        Box::new(RandomForest::new(RandomForestParams {
                            n_trees: 100,
                            max_depth: Some(10),
                            seed: Some(42),
                            ..Default::default()
                        }))
                    }),
                ));
            }
            if self.config.try_tuned_params {
                configs.push((
                    "RF-tuned".to_string(),
                    Box::new(|| {
                        Box::new(RandomForest::new(RandomForestParams {
                            n_trees: 200,
                            max_depth: Some(15),
                            min_samples_leaf: 2,
                            seed: Some(42),
                            ..Default::default()
                        }))
                    }),
                ));
            }
        }

        if self.config.include_gbt {
            if self.config.try_default_params {
                configs.push((
                    "GBT-default".to_string(),
                    Box::new(|| {
                        Box::new(GradientBoostingClassifier::new(GradientBoostingParams {
                            n_estimators: 100,
                            learning_rate: 0.1,
                            max_depth: 4,
                            seed: Some(42),
                            ..Default::default()
                        }))
                    }),
                ));
            }
            if self.config.try_tuned_params {
                configs.push((
                    "GBT-tuned".to_string(),
                    Box::new(|| {
                        Box::new(GradientBoostingClassifier::new(GradientBoostingParams {
                            n_estimators: 200,
                            learning_rate: 0.05,
                            max_depth: 5,
                            subsample: 0.8,
                            seed: Some(42),
                            ..Default::default()
                        }))
                    }),
                ));
            }
        }

        if self.config.include_logreg {
            if self.config.try_default_params {
                configs.push((
                    "LogReg-default".to_string(),
                    Box::new(|| {
                        Box::new(LogisticRegression::new(LogisticRegressionParams {
                            max_iter: 500,
                            ..Default::default()
                        }))
                    }),
                ));
            }
            if self.config.try_tuned_params {
                configs.push((
                    "LogReg-tuned".to_string(),
                    Box::new(|| {
                        Box::new(LogisticRegression::new(LogisticRegressionParams {
                            max_iter: 1000,
                            learning_rate: 0.05,
                            lambda: 0.001,
                            ..Default::default()
                        }))
                    }),
                ));
            }
        }

        configs
    }

    fn evaluate_model(
        &self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
        folds: &[(Vec<usize>, Vec<usize>)],
        factory: &dyn Fn() -> Box<dyn Classifier>,
    ) -> OracleResult<CVResults> {
        let mut fold_metrics = Vec::new();

        for (train_idx, test_idx) in folds {
            let train_feat: Vec<FeatureVector> =
                train_idx.iter().map(|&i| features[i].clone()).collect();
            let train_lab: Vec<DecompositionMethod> =
                train_idx.iter().map(|&i| labels[i]).collect();
            let test_feat: Vec<FeatureVector> =
                test_idx.iter().map(|&i| features[i].clone()).collect();
            let test_lab: Vec<DecompositionMethod> =
                test_idx.iter().map(|&i| labels[i]).collect();

            let mut clf = factory();
            clf.train(&train_feat, &train_lab)?;
            let metrics = clf.evaluate(&test_feat, &test_lab)?;
            fold_metrics.push(metrics);
        }

        Ok(CVResults::from_fold_metrics(fold_metrics))
    }

    /// Build the best model (train on full data with best config).
    pub fn build_best(
        &self,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
    ) -> OracleResult<(Box<dyn Classifier>, SelectionResult)> {
        let result = self.select(features, labels)?;

        let configs = self.generate_configurations();
        let best_factory = configs
            .iter()
            .find(|(name, _)| name == &result.best_model_name)
            .map(|(_, f)| f);

        if let Some(factory) = best_factory {
            let mut clf = factory();
            clf.train(features, labels)?;
            Ok((clf, result))
        } else {
            // Fallback to RF default
            let mut clf = Box::new(RandomForest::new(RandomForestParams::default()));
            clf.train(features, labels)?;
            Ok((clf as Box<dyn Classifier>, result))
        }
    }
}

/// Compare models on specific subsets (e.g., per structure type).
pub fn compare_on_subsets(
    models: &[(&str, &dyn Classifier)],
    features: &[FeatureVector],
    labels: &[DecompositionMethod],
    subset_indices: &HashMap<String, Vec<usize>>,
) -> HashMap<String, Vec<(String, f64)>> {
    let mut results = HashMap::new();

    for (subset_name, indices) in subset_indices {
        let sub_feat: Vec<FeatureVector> = indices.iter().map(|&i| features[i].clone()).collect();
        let sub_lab: Vec<DecompositionMethod> = indices.iter().map(|&i| labels[i]).collect();

        let mut subset_results = Vec::new();
        for (name, model) in models {
            if let Ok(metrics) = model.evaluate(&sub_feat, &sub_lab) {
                subset_results.push((name.to_string(), metrics.accuracy));
            }
        }
        results.insert(subset_name.clone(), subset_results);
    }

    results
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
    fn test_automl_config_default() {
        let config = AutoMLConfig::default();
        assert!(config.include_rf);
        assert!(config.include_gbt);
        assert!(config.include_logreg);
    }

    #[test]
    fn test_model_selector_basic() {
        let (features, labels) = make_data(100, 42);
        let config = AutoMLConfig {
            n_folds: 3,
            include_gbt: false,
            include_logreg: false,
            try_tuned_params: false,
            ..Default::default()
        };
        let selector = ModelSelector::new(config);
        let result = selector.select(&features, &labels).unwrap();

        assert!(!result.comparisons.is_empty());
        assert!(!result.best_model_name.is_empty());
        assert!(result.best_cv_accuracy >= 0.0);
    }

    #[test]
    fn test_model_selector_multiple_models() {
        let (features, labels) = make_data(100, 42);
        let config = AutoMLConfig {
            n_folds: 3,
            try_tuned_params: false,
            ..Default::default()
        };
        let selector = ModelSelector::new(config);
        let result = selector.select(&features, &labels).unwrap();

        assert!(result.comparisons.len() >= 2);
        // Check ranking
        for (i, comp) in result.comparisons.iter().enumerate() {
            assert_eq!(comp.rank, i + 1);
        }
    }

    #[test]
    fn test_model_selector_empty_data() {
        let selector = ModelSelector::new(AutoMLConfig::default());
        assert!(selector.select(&[], &[]).is_err());
    }

    #[test]
    fn test_model_comparison_summary() {
        let cv = CVResults {
            fold_metrics: vec![],
            mean_accuracy: 0.8,
            std_accuracy: 0.05,
            mean_macro_f1: 0.75,
            std_macro_f1: 0.06,
            mean_kappa: 0.7,
            n_folds: 5,
        };
        let comp = ModelComparison {
            model_name: "RF-default".to_string(),
            params_description: "default".to_string(),
            cv_results: cv,
            rank: 1,
        };
        let summary = comp.summary();
        assert!(summary.contains("#1"));
        assert!(summary.contains("RF-default"));
    }

    #[test]
    fn test_selection_result_report() {
        let result = SelectionResult {
            comparisons: vec![ModelComparison {
                model_name: "RF".to_string(),
                params_description: "default".to_string(),
                cv_results: CVResults {
                    fold_metrics: vec![],
                    mean_accuracy: 0.85,
                    std_accuracy: 0.03,
                    mean_macro_f1: 0.8,
                    std_macro_f1: 0.04,
                    mean_kappa: 0.75,
                    n_folds: 5,
                },
                rank: 1,
            }],
            best_model_name: "RF".to_string(),
            best_cv_accuracy: 0.85,
            best_cv_f1: 0.8,
        };
        let report = result.report();
        assert!(report.contains("RF"));
    }

    #[test]
    fn test_build_best() {
        let (features, labels) = make_data(80, 42);
        let config = AutoMLConfig {
            n_folds: 3,
            include_gbt: false,
            include_logreg: false,
            try_tuned_params: false,
            ..Default::default()
        };
        let selector = ModelSelector::new(config);
        let (clf, result) = selector.build_best(&features, &labels).unwrap();
        assert!(clf.is_trained());
        assert!(!result.best_model_name.is_empty());
    }

    #[test]
    fn test_generate_configurations() {
        let config = AutoMLConfig::default();
        let selector = ModelSelector::new(config);
        let configs = selector.generate_configurations();
        assert!(configs.len() >= 4); // 2 RF + 2 GBT + 2 LogReg (some may be less)
    }

    #[test]
    fn test_compare_on_subsets() {
        let (features, labels) = make_data(40, 42);
        let mut rf = RandomForest::new(RandomForestParams {
            n_trees: 5,
            max_depth: Some(3),
            seed: Some(42),
            ..Default::default()
        });
        rf.train(&features, &labels).unwrap();

        let models: Vec<(&str, &dyn Classifier)> = vec![("RF", &rf)];
        let mut subsets = HashMap::new();
        subsets.insert("first_half".to_string(), (0..20).collect());
        subsets.insert("second_half".to_string(), (20..40).collect());

        let results = compare_on_subsets(&models, &features, &labels, &subsets);
        assert_eq!(results.len(), 2);
    }
}
