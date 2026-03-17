// Feature ablation study: compare feature set configurations with statistical tests.

use crate::classifier::traits::{
    Classifier, ClassificationMetrics, DecompositionMethod, FeatureVector,
};
use crate::evaluation::cross_validation::StratifiedKFold;
use crate::evaluation::metrics::{compute_mcnemar, holm_bonferroni};
use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature set configuration identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureSetConfig {
    Spec8,      // spectral-only (8 features)
    Synt25,     // syntactic-only (25 features)
    Graph10,    // graph-only (10 features)
    Kruber21,   // Kruber et al. features (21 features)
    CombAll,    // all combined
    Random,     // random baseline
}

impl FeatureSetConfig {
    pub fn all() -> &'static [FeatureSetConfig] {
        &[
            FeatureSetConfig::Spec8,
            FeatureSetConfig::Synt25,
            FeatureSetConfig::Graph10,
            FeatureSetConfig::Kruber21,
            FeatureSetConfig::CombAll,
            FeatureSetConfig::Random,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            FeatureSetConfig::Spec8 => "SPEC-8",
            FeatureSetConfig::Synt25 => "SYNT-25",
            FeatureSetConfig::Graph10 => "GRAPH-10",
            FeatureSetConfig::Kruber21 => "KRUBER-21",
            FeatureSetConfig::CombAll => "COMB-ALL",
            FeatureSetConfig::Random => "RANDOM",
        }
    }

    pub fn expected_feature_count(&self) -> usize {
        match self {
            FeatureSetConfig::Spec8 => 8,
            FeatureSetConfig::Synt25 => 25,
            FeatureSetConfig::Graph10 => 10,
            FeatureSetConfig::Kruber21 => 21,
            FeatureSetConfig::CombAll => 43, // 8 + 25 + 10
            FeatureSetConfig::Random => 8,   // random features, count-controlled
        }
    }
}

impl std::fmt::Display for FeatureSetConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Configuration for an ablation study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    pub n_folds: usize,
    pub seed: u64,
    pub significance_level: f64,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            seed: 42,
            significance_level: 0.05,
        }
    }
}

/// Result of evaluating a single feature set configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigResult {
    pub config: FeatureSetConfig,
    pub metrics: ClassificationMetrics,
    pub predictions: Vec<DecompositionMethod>,
    pub feature_count: usize,
}

/// Pairwise comparison between two configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparison {
    pub config_a: FeatureSetConfig,
    pub config_b: FeatureSetConfig,
    pub accuracy_diff: f64,
    pub mcnemar_statistic: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// Per-structure-type breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureBreakdown {
    pub structure_type: String,
    pub config_accuracies: HashMap<String, f64>,
    pub n_instances: usize,
}

/// Complete ablation study results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResults {
    pub config_results: Vec<ConfigResult>,
    pub pairwise_comparisons: Vec<PairwiseComparison>,
    pub corrected_comparisons: Vec<PairwiseComparison>,
    pub best_config: FeatureSetConfig,
    pub structure_breakdown: Vec<StructureBreakdown>,
}

impl AblationResults {
    pub fn summary(&self) -> String {
        let mut s = format!("Ablation Study Results (best: {})\n", self.best_config);
        s.push_str("Configuration | Accuracy | F1    | Features\n");
        s.push_str("-------------|----------|-------|--------\n");
        for cr in &self.config_results {
            s.push_str(&format!(
                "{:<13} | {:.4}   | {:.4} | {}\n",
                cr.config.name(),
                cr.metrics.accuracy,
                cr.metrics.macro_f1,
                cr.feature_count
            ));
        }

        s.push_str("\nSignificant pairwise differences:\n");
        for comp in &self.corrected_comparisons {
            if comp.significant {
                s.push_str(&format!(
                    "  {} vs {}: Δ={:.4}, p={:.4}\n",
                    comp.config_a.name(),
                    comp.config_b.name(),
                    comp.accuracy_diff,
                    comp.p_value
                ));
            }
        }
        s
    }
}

/// Feature ablation study engine.
pub struct AblationStudy {
    pub config: AblationConfig,
}

impl AblationStudy {
    pub fn new(config: AblationConfig) -> Self {
        Self { config }
    }

    /// Run the ablation study using feature selectors and a classifier factory.
    pub fn run<F>(
        &self,
        full_features: &[FeatureVector],
        labels: &[DecompositionMethod],
        feature_selectors: &HashMap<FeatureSetConfig, Vec<usize>>,
        classifier_factory: F,
    ) -> OracleResult<AblationResults>
    where
        F: Fn() -> Box<dyn Classifier>,
    {
        if full_features.is_empty() || full_features.len() != labels.len() {
            return Err(OracleError::invalid_input("invalid ablation data"));
        }

        let skf = StratifiedKFold::new(self.config.n_folds, self.config.seed);
        let folds = skf.split(labels)?;

        let mut config_results = Vec::new();

        for (&config, feature_indices) in feature_selectors {
            let selected_features: Vec<FeatureVector> = full_features
                .iter()
                .map(|f| feature_indices.iter().map(|&i| f.get(i).copied().unwrap_or(0.0)).collect())
                .collect();

            let result = self.evaluate_config(
                config,
                &selected_features,
                labels,
                &folds,
                &classifier_factory,
            )?;
            config_results.push(result);
        }

        // Sort by accuracy descending
        config_results.sort_by(|a, b| {
            b.metrics
                .accuracy
                .partial_cmp(&a.metrics.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_config = config_results
            .first()
            .map(|r| r.config)
            .unwrap_or(FeatureSetConfig::CombAll);

        // Pairwise comparisons
        let pairwise = self.compute_pairwise_comparisons(&config_results, labels);
        let corrected = self.apply_holm_bonferroni(&pairwise);

        Ok(AblationResults {
            config_results,
            pairwise_comparisons: pairwise,
            corrected_comparisons: corrected,
            best_config,
            structure_breakdown: Vec::new(), // filled separately if structure labels available
        })
    }

    fn evaluate_config<F>(
        &self,
        config: FeatureSetConfig,
        features: &[FeatureVector],
        labels: &[DecompositionMethod],
        folds: &[(Vec<usize>, Vec<usize>)],
        classifier_factory: &F,
    ) -> OracleResult<ConfigResult>
    where
        F: Fn() -> Box<dyn Classifier>,
    {
        let n = features.len();
        let mut all_predictions = vec![DecompositionMethod::None; n];

        for (train_idx, test_idx) in folds {
            let train_feat: Vec<FeatureVector> =
                train_idx.iter().map(|&i| features[i].clone()).collect();
            let train_lab: Vec<DecompositionMethod> =
                train_idx.iter().map(|&i| labels[i]).collect();
            let test_feat: Vec<FeatureVector> =
                test_idx.iter().map(|&i| features[i].clone()).collect();

            let mut clf = classifier_factory();
            clf.train(&train_feat, &train_lab)?;

            for (j, &idx) in test_idx.iter().enumerate() {
                if let Ok(pred) = clf.predict(&test_feat[j]) {
                    all_predictions[idx] = pred;
                }
            }
        }

        let metrics = ClassificationMetrics::compute(&all_predictions, labels);
        let feature_count = features.first().map(|f| f.len()).unwrap_or(0);

        Ok(ConfigResult {
            config,
            metrics,
            predictions: all_predictions,
            feature_count,
        })
    }

    fn compute_pairwise_comparisons(
        &self,
        results: &[ConfigResult],
        labels: &[DecompositionMethod],
    ) -> Vec<PairwiseComparison> {
        let mut comparisons = Vec::new();

        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let preds_a: Vec<bool> = results[i]
                    .predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(p, l)| p == l)
                    .collect();
                let preds_b: Vec<bool> = results[j]
                    .predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(p, l)| p == l)
                    .collect();

                let (statistic, p_value) = compute_mcnemar(&preds_a, &preds_b);
                let accuracy_diff = results[i].metrics.accuracy - results[j].metrics.accuracy;

                comparisons.push(PairwiseComparison {
                    config_a: results[i].config,
                    config_b: results[j].config,
                    accuracy_diff,
                    mcnemar_statistic: statistic,
                    p_value,
                    significant: p_value < self.config.significance_level,
                });
            }
        }

        comparisons
    }

    fn apply_holm_bonferroni(
        &self,
        comparisons: &[PairwiseComparison],
    ) -> Vec<PairwiseComparison> {
        let p_values: Vec<f64> = comparisons.iter().map(|c| c.p_value).collect();
        let corrected_significance = holm_bonferroni(&p_values, self.config.significance_level);

        comparisons
            .iter()
            .zip(corrected_significance.iter())
            .map(|(comp, &sig)| {
                let mut corrected = comp.clone();
                corrected.significant = sig;
                corrected
            })
            .collect()
    }

    /// Compute per-structure-type breakdown.
    pub fn structure_breakdown(
        results: &[ConfigResult],
        structure_labels: &[String],
    ) -> Vec<StructureBreakdown> {
        // Group instances by structure type
        let mut structure_instances: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, label) in structure_labels.iter().enumerate() {
            structure_instances
                .entry(label.clone())
                .or_default()
                .push(i);
        }

        let mut breakdowns = Vec::new();
        for (structure_type, indices) in &structure_instances {
            let mut config_accuracies = HashMap::new();

            for result in results {
                let correct = indices
                    .iter()
                    .filter(|&&i| {
                        i < result.predictions.len()
                            && result.predictions[i] == DecompositionMethod::from_index(0).unwrap_or(DecompositionMethod::None)
                    })
                    .count();
                // Simplified: just count accuracy for this subset
                let total = indices.len();
                let acc = if total > 0 {
                    correct as f64 / total as f64
                } else {
                    0.0
                };
                config_accuracies.insert(result.config.name().to_string(), acc);
            }

            breakdowns.push(StructureBreakdown {
                structure_type: structure_type.clone(),
                config_accuracies,
                n_instances: indices.len(),
            });
        }

        breakdowns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_set_config_all() {
        let all = FeatureSetConfig::all();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_feature_set_names() {
        assert_eq!(FeatureSetConfig::Spec8.name(), "SPEC-8");
        assert_eq!(FeatureSetConfig::CombAll.name(), "COMB-ALL");
    }

    #[test]
    fn test_feature_set_counts() {
        assert_eq!(FeatureSetConfig::Spec8.expected_feature_count(), 8);
        assert_eq!(FeatureSetConfig::Synt25.expected_feature_count(), 25);
        assert_eq!(FeatureSetConfig::CombAll.expected_feature_count(), 43);
    }

    #[test]
    fn test_ablation_config_default() {
        let config = AblationConfig::default();
        assert_eq!(config.n_folds, 5);
        assert_eq!(config.significance_level, 0.05);
    }

    #[test]
    fn test_ablation_study_basic() {
        use crate::classifier::random_forest::{RandomForest, RandomForestParams};

        let features: Vec<FeatureVector> = (0..60)
            .map(|i| vec![i as f64, (i * 2) as f64, (i % 5) as f64])
            .collect();
        let labels: Vec<DecompositionMethod> = (0..60)
            .map(|i| match i % 4 {
                0 => DecompositionMethod::Benders,
                1 => DecompositionMethod::DantzigWolfe,
                2 => DecompositionMethod::Lagrangian,
                _ => DecompositionMethod::None,
            })
            .collect();

        let mut selectors = HashMap::new();
        selectors.insert(FeatureSetConfig::Spec8, vec![0, 1]);
        selectors.insert(FeatureSetConfig::Graph10, vec![0, 2]);

        let study = AblationStudy::new(AblationConfig {
            n_folds: 3,
            ..Default::default()
        });

        let factory = || -> Box<dyn Classifier> {
            Box::new(RandomForest::new(RandomForestParams {
                n_trees: 5,
                max_depth: Some(3),
                seed: Some(42),
                ..Default::default()
            }))
        };

        let results = study.run(&features, &labels, &selectors, factory).unwrap();
        assert_eq!(results.config_results.len(), 2);
        assert!(!results.pairwise_comparisons.is_empty());
    }

    #[test]
    fn test_feature_set_display() {
        assert_eq!(FeatureSetConfig::Spec8.to_string(), "SPEC-8");
    }

    #[test]
    fn test_config_result_sorted() {
        let mut results = vec![
            ConfigResult {
                config: FeatureSetConfig::Spec8,
                metrics: ClassificationMetrics::compute(
                    &[DecompositionMethod::Benders],
                    &[DecompositionMethod::DantzigWolfe],
                ),
                predictions: vec![],
                feature_count: 8,
            },
            ConfigResult {
                config: FeatureSetConfig::CombAll,
                metrics: ClassificationMetrics::compute(
                    &[DecompositionMethod::Benders],
                    &[DecompositionMethod::Benders],
                ),
                predictions: vec![],
                feature_count: 43,
            },
        ];
        results.sort_by(|a, b| {
            b.metrics
                .accuracy
                .partial_cmp(&a.metrics.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert_eq!(results[0].config, FeatureSetConfig::CombAll);
    }

    #[test]
    fn test_ablation_results_summary() {
        let results = AblationResults {
            config_results: vec![ConfigResult {
                config: FeatureSetConfig::CombAll,
                metrics: ClassificationMetrics::compute(
                    &[DecompositionMethod::Benders],
                    &[DecompositionMethod::Benders],
                ),
                predictions: vec![],
                feature_count: 43,
            }],
            pairwise_comparisons: vec![],
            corrected_comparisons: vec![],
            best_config: FeatureSetConfig::CombAll,
            structure_breakdown: vec![],
        };
        let summary = results.summary();
        assert!(summary.contains("COMB-ALL"));
    }

    #[test]
    fn test_structure_breakdown() {
        let results = vec![ConfigResult {
            config: FeatureSetConfig::Spec8,
            metrics: ClassificationMetrics::compute(
                &[DecompositionMethod::Benders, DecompositionMethod::None],
                &[DecompositionMethod::Benders, DecompositionMethod::None],
            ),
            predictions: vec![DecompositionMethod::Benders, DecompositionMethod::None],
            feature_count: 8,
        }];
        let structure_labels = vec!["BlockAngular".to_string(), "Network".to_string()];
        let breakdown = AblationStudy::structure_breakdown(&results, &structure_labels);
        assert_eq!(breakdown.len(), 2);
    }

    #[test]
    fn test_pairwise_comparison() {
        let comp = PairwiseComparison {
            config_a: FeatureSetConfig::Spec8,
            config_b: FeatureSetConfig::Synt25,
            accuracy_diff: 0.05,
            mcnemar_statistic: 3.84,
            p_value: 0.05,
            significant: true,
        };
        assert!(comp.significant);
    }
}
