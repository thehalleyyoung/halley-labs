// Classifier traits and shared types for the decomposition selection oracle.

use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Feature vector: a dense vector of f64 values.
pub type FeatureVector = Vec<f64>;

/// Decomposition method that the oracle recommends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum DecompositionMethod {
    Benders,
    DantzigWolfe,
    Lagrangian,
    None,
}

impl DecompositionMethod {
    pub fn all() -> &'static [DecompositionMethod] {
        &[
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
            DecompositionMethod::Lagrangian,
            DecompositionMethod::None,
        ]
    }

    pub fn to_index(self) -> usize {
        match self {
            DecompositionMethod::Benders => 0,
            DecompositionMethod::DantzigWolfe => 1,
            DecompositionMethod::Lagrangian => 2,
            DecompositionMethod::None => 3,
        }
    }

    pub fn from_index(idx: usize) -> OracleResult<Self> {
        match idx {
            0 => Ok(DecompositionMethod::Benders),
            1 => Ok(DecompositionMethod::DantzigWolfe),
            2 => Ok(DecompositionMethod::Lagrangian),
            3 => Ok(DecompositionMethod::None),
            _ => Err(OracleError::invalid_input(format!(
                "invalid class index: {}",
                idx
            ))),
        }
    }

    pub fn n_classes() -> usize {
        4
    }
}

impl fmt::Display for DecompositionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecompositionMethod::Benders => write!(f, "Benders"),
            DecompositionMethod::DantzigWolfe => write!(f, "Dantzig-Wolfe"),
            DecompositionMethod::Lagrangian => write!(f, "Lagrangian"),
            DecompositionMethod::None => write!(f, "None"),
        }
    }
}

/// Core classifier trait for decomposition method prediction.
pub trait Classifier: Send + Sync {
    /// Train the classifier on labeled data.
    fn train(&mut self, features: &[FeatureVector], labels: &[DecompositionMethod]) -> OracleResult<()>;

    /// Predict the best decomposition method for a feature vector.
    fn predict(&self, features: &FeatureVector) -> OracleResult<DecompositionMethod>;

    /// Predict class probabilities for each decomposition method.
    fn predict_proba(&self, features: &FeatureVector) -> OracleResult<Vec<f64>>;

    /// Evaluate the classifier on test data and return classification metrics.
    fn evaluate(
        &self,
        test_features: &[FeatureVector],
        test_labels: &[DecompositionMethod],
    ) -> OracleResult<ClassificationMetrics> {
        if test_features.len() != test_labels.len() {
            return Err(OracleError::invalid_input(
                "test_features and test_labels must have the same length",
            ));
        }
        if test_features.is_empty() {
            return Err(OracleError::invalid_input("test data is empty"));
        }

        let predictions: Vec<DecompositionMethod> = test_features
            .iter()
            .map(|f| self.predict(f))
            .collect::<OracleResult<Vec<_>>>()?;

        Ok(ClassificationMetrics::compute(&predictions, test_labels))
    }

    /// Return feature importance scores (higher = more important).
    fn feature_importance(&self) -> OracleResult<Vec<f64>>;

    /// Return the name of this classifier.
    fn name(&self) -> &str;

    /// Whether the classifier has been trained.
    fn is_trained(&self) -> bool;
}

/// Confusion matrix for multi-class classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// matrix[true_class][predicted_class] = count
    pub matrix: Vec<Vec<usize>>,
    pub class_names: Vec<String>,
}

impl ConfusionMatrix {
    pub fn new(n_classes: usize) -> Self {
        let class_names: Vec<String> = DecompositionMethod::all()
            .iter()
            .take(n_classes)
            .map(|m| m.to_string())
            .collect();
        Self {
            matrix: vec![vec![0; n_classes]; n_classes],
            class_names,
        }
    }

    pub fn add(&mut self, true_class: usize, predicted_class: usize) {
        if true_class < self.matrix.len() && predicted_class < self.matrix[0].len() {
            self.matrix[true_class][predicted_class] += 1;
        }
    }

    pub fn total(&self) -> usize {
        self.matrix.iter().flat_map(|row| row.iter()).sum()
    }

    pub fn correct(&self) -> usize {
        (0..self.matrix.len()).map(|i| self.matrix[i][i]).sum()
    }

    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        self.correct() as f64 / total as f64
    }

    /// True positives for a given class.
    pub fn tp(&self, class: usize) -> usize {
        self.matrix[class][class]
    }

    /// False positives for a given class.
    pub fn fp(&self, class: usize) -> usize {
        (0..self.matrix.len())
            .filter(|&i| i != class)
            .map(|i| self.matrix[i][class])
            .sum()
    }

    /// False negatives for a given class.
    pub fn fn_count(&self, class: usize) -> usize {
        (0..self.matrix[class].len())
            .filter(|&j| j != class)
            .map(|j| self.matrix[class][j])
            .sum()
    }

    /// True negatives for a given class.
    pub fn tn(&self, class: usize) -> usize {
        let total = self.total();
        total - self.tp(class) - self.fp(class) - self.fn_count(class)
    }
}

/// Per-class metrics for a single decomposition method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerClassMetrics {
    pub class_name: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

/// Comprehensive classification metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub weighted_accuracy: f64,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub micro_precision: f64,
    pub micro_recall: f64,
    pub micro_f1: f64,
    pub cohen_kappa: f64,
    pub confusion_matrix: ConfusionMatrix,
    pub per_class: Vec<PerClassMetrics>,
}

impl ClassificationMetrics {
    /// Compute all classification metrics from predictions and true labels.
    pub fn compute(
        predictions: &[DecompositionMethod],
        true_labels: &[DecompositionMethod],
    ) -> Self {
        let n_classes = DecompositionMethod::n_classes();
        let mut cm = ConfusionMatrix::new(n_classes);

        for (pred, truth) in predictions.iter().zip(true_labels.iter()) {
            cm.add(truth.to_index(), pred.to_index());
        }

        let accuracy = cm.accuracy();

        // Per-class metrics
        let mut per_class = Vec::new();
        let mut total_support = 0usize;
        let mut weighted_acc_sum = 0.0;

        for c in 0..n_classes {
            let tp = cm.tp(c);
            let fp = cm.fp(c);
            let fn_ = cm.fn_count(c);
            let support = tp + fn_;

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = if support > 0 {
                tp as f64 / support as f64
            } else {
                0.0
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            weighted_acc_sum += recall * support as f64;
            total_support += support;

            let class_name = DecompositionMethod::from_index(c)
                .map(|m| m.to_string())
                .unwrap_or_else(|_| format!("Class {}", c));

            per_class.push(PerClassMetrics {
                class_name,
                precision,
                recall,
                f1,
                support,
                true_positives: tp,
                false_positives: fp,
                false_negatives: fn_,
            });
        }

        let weighted_accuracy = if total_support > 0 {
            weighted_acc_sum / total_support as f64
        } else {
            0.0
        };

        // Macro-averaged metrics
        let active_classes = per_class.iter().filter(|pc| pc.support > 0).count() as f64;
        let macro_precision = if active_classes > 0.0 {
            per_class
                .iter()
                .filter(|pc| pc.support > 0)
                .map(|pc| pc.precision)
                .sum::<f64>()
                / active_classes
        } else {
            0.0
        };
        let macro_recall = if active_classes > 0.0 {
            per_class
                .iter()
                .filter(|pc| pc.support > 0)
                .map(|pc| pc.recall)
                .sum::<f64>()
                / active_classes
        } else {
            0.0
        };
        let macro_f1 = if active_classes > 0.0 {
            per_class
                .iter()
                .filter(|pc| pc.support > 0)
                .map(|pc| pc.f1)
                .sum::<f64>()
                / active_classes
        } else {
            0.0
        };

        // Micro-averaged metrics
        let total_tp: usize = per_class.iter().map(|pc| pc.true_positives).sum();
        let total_fp: usize = per_class.iter().map(|pc| pc.false_positives).sum();
        let total_fn: usize = per_class.iter().map(|pc| pc.false_negatives).sum();

        let micro_precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else {
            0.0
        };
        let micro_recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else {
            0.0
        };
        let micro_f1 = if micro_precision + micro_recall > 0.0 {
            2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        } else {
            0.0
        };

        // Cohen's kappa
        let n = predictions.len() as f64;
        let cohen_kappa = if n > 0.0 {
            let p_o = accuracy;
            let mut p_e = 0.0;
            for c in 0..n_classes {
                let actual_c = (cm.tp(c) + cm.fn_count(c)) as f64;
                let pred_c = (cm.tp(c) + cm.fp(c)) as f64;
                p_e += (actual_c / n) * (pred_c / n);
            }
            if (1.0 - p_e).abs() < 1e-12 {
                if (p_o - 1.0).abs() < 1e-12 {
                    1.0
                } else {
                    0.0
                }
            } else {
                (p_o - p_e) / (1.0 - p_e)
            }
        } else {
            0.0
        };

        ClassificationMetrics {
            accuracy,
            weighted_accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            micro_precision,
            micro_recall,
            micro_f1,
            cohen_kappa,
            confusion_matrix: cm,
            per_class,
        }
    }

    /// Returns a formatted summary string.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Accuracy:          {:.4}\n", self.accuracy));
        s.push_str(&format!("Weighted Accuracy: {:.4}\n", self.weighted_accuracy));
        s.push_str(&format!("Macro F1:          {:.4}\n", self.macro_f1));
        s.push_str(&format!("Micro F1:          {:.4}\n", self.micro_f1));
        s.push_str(&format!("Cohen's Kappa:     {:.4}\n", self.cohen_kappa));
        s.push_str("\nPer-class:\n");
        for pc in &self.per_class {
            s.push_str(&format!(
                "  {:<15} P={:.3} R={:.3} F1={:.3} (n={})\n",
                pc.class_name, pc.precision, pc.recall, pc.f1, pc.support
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_method_roundtrip() {
        for &m in DecompositionMethod::all() {
            let idx = m.to_index();
            let back = DecompositionMethod::from_index(idx).unwrap();
            assert_eq!(m, back);
        }
    }

    #[test]
    fn test_decomposition_method_invalid_index() {
        assert!(DecompositionMethod::from_index(99).is_err());
    }

    #[test]
    fn test_confusion_matrix_basic() {
        let mut cm = ConfusionMatrix::new(4);
        cm.add(0, 0);
        cm.add(0, 1);
        cm.add(1, 1);
        cm.add(1, 1);
        assert_eq!(cm.total(), 4);
        assert_eq!(cm.correct(), 3);
        assert!((cm.accuracy() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_tp_fp_fn() {
        let mut cm = ConfusionMatrix::new(2);
        cm.add(0, 0); // TP for class 0
        cm.add(0, 1); // FN for class 0, FP for class 1
        cm.add(1, 0); // FP for class 0, FN for class 1
        cm.add(1, 1); // TP for class 1
        assert_eq!(cm.tp(0), 1);
        assert_eq!(cm.fp(0), 1);
        assert_eq!(cm.fn_count(0), 1);
        assert_eq!(cm.tn(0), 1);
    }

    #[test]
    fn test_classification_metrics_perfect() {
        let labels = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
            DecompositionMethod::Lagrangian,
            DecompositionMethod::None,
        ];
        let preds = labels.clone();
        let metrics = ClassificationMetrics::compute(&preds, &labels);
        assert!((metrics.accuracy - 1.0).abs() < 1e-10);
        assert!((metrics.macro_f1 - 1.0).abs() < 1e-10);
        assert!((metrics.cohen_kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_classification_metrics_random() {
        let labels = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
            DecompositionMethod::DantzigWolfe,
        ];
        let preds = vec![
            DecompositionMethod::DantzigWolfe,
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        let metrics = ClassificationMetrics::compute(&preds, &labels);
        assert!((metrics.accuracy - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_per_class_metrics() {
        let labels = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        let preds = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        let metrics = ClassificationMetrics::compute(&preds, &labels);
        let benders = &metrics.per_class[DecompositionMethod::Benders.to_index()];
        assert!((benders.precision - 1.0).abs() < 1e-10);
        assert!((benders.recall - 1.0).abs() < 1e-10);
        assert_eq!(benders.support, 2);
    }

    #[test]
    fn test_decomposition_display() {
        assert_eq!(DecompositionMethod::Benders.to_string(), "Benders");
        assert_eq!(DecompositionMethod::DantzigWolfe.to_string(), "Dantzig-Wolfe");
    }

    #[test]
    fn test_metrics_summary() {
        let labels = vec![DecompositionMethod::Benders; 10];
        let preds = labels.clone();
        let metrics = ClassificationMetrics::compute(&preds, &labels);
        let summary = metrics.summary();
        assert!(summary.contains("Accuracy"));
        assert!(summary.contains("Benders"));
    }

    #[test]
    fn test_cohen_kappa_chance() {
        // All same class predictions = kappa should be 0
        let labels = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        let preds = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
        ];
        let metrics = ClassificationMetrics::compute(&preds, &labels);
        assert!(metrics.cohen_kappa < 1.0);
    }

    #[test]
    fn test_n_classes() {
        assert_eq!(DecompositionMethod::n_classes(), 4);
    }

    #[test]
    fn test_serde_roundtrip() {
        let method = DecompositionMethod::Lagrangian;
        let json = serde_json::to_string(&method).unwrap();
        let back: DecompositionMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(method, back);
    }
}
