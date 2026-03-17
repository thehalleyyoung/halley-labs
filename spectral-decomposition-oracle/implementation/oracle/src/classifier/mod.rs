// Classifier submodule: traits, random forest, gradient boosting, logistic, ensemble.

pub mod traits;
pub mod random_forest;
pub mod gradient_boosting;
pub mod logistic;
pub mod ensemble;

pub use traits::{
    Classifier, ClassificationMetrics, PerClassMetrics, ConfusionMatrix,
    DecompositionMethod, FeatureVector,
};
pub use random_forest::{RandomForest, RandomForestParams};
pub use gradient_boosting::{GradientBoostingClassifier, GradientBoostingParams};
pub use logistic::{LogisticRegression, LogisticRegressionParams};
pub use ensemble::{VotingClassifier, StackingClassifier, VotingStrategy};

use crate::error::OracleResult;
use serde::{Deserialize, Serialize};

/// A labeled dataset for training classifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub features: Vec<FeatureVector>,
    pub labels: Vec<DecompositionMethod>,
    pub instance_names: Vec<String>,
}

impl Dataset {
    pub fn new(
        features: Vec<FeatureVector>,
        labels: Vec<DecompositionMethod>,
        instance_names: Vec<String>,
    ) -> OracleResult<Self> {
        if features.len() != labels.len() || features.len() != instance_names.len() {
            return Err(crate::error::OracleError::invalid_input(
                "features, labels, and instance_names must have the same length",
            ));
        }
        if features.is_empty() {
            return Err(crate::error::OracleError::invalid_input("dataset is empty"));
        }
        Ok(Self {
            features,
            labels,
            instance_names,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.features.len()
    }

    pub fn n_features(&self) -> usize {
        self.features.first().map(|f| f.len()).unwrap_or(0)
    }

    /// Split dataset into train and test by indices.
    pub fn split(&self, train_idx: &[usize], test_idx: &[usize]) -> (Dataset, Dataset) {
        let train = Dataset {
            features: train_idx.iter().map(|&i| self.features[i].clone()).collect(),
            labels: train_idx.iter().map(|&i| self.labels[i]).collect(),
            instance_names: train_idx.iter().map(|&i| self.instance_names[i].clone()).collect(),
        };
        let test = Dataset {
            features: test_idx.iter().map(|&i| self.features[i].clone()).collect(),
            labels: test_idx.iter().map(|&i| self.labels[i]).collect(),
            instance_names: test_idx.iter().map(|&i| self.instance_names[i].clone()).collect(),
        };
        (train, test)
    }

    /// Class distribution as a map from method to count.
    pub fn class_distribution(&self) -> indexmap::IndexMap<DecompositionMethod, usize> {
        let mut dist = indexmap::IndexMap::new();
        for &label in &self.labels {
            *dist.entry(label).or_insert(0) += 1;
        }
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(n: usize) -> Dataset {
        let features: Vec<FeatureVector> = (0..n).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let labels: Vec<DecompositionMethod> = (0..n)
            .map(|i| if i % 2 == 0 { DecompositionMethod::Benders } else { DecompositionMethod::DantzigWolfe })
            .collect();
        let names: Vec<String> = (0..n).map(|i| format!("inst_{}", i)).collect();
        Dataset::new(features, labels, names).unwrap()
    }

    #[test]
    fn test_dataset_creation() {
        let ds = make_dataset(10);
        assert_eq!(ds.n_samples(), 10);
        assert_eq!(ds.n_features(), 2);
    }

    #[test]
    fn test_dataset_empty_error() {
        let result = Dataset::new(vec![], vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_mismatched_lengths() {
        let result = Dataset::new(vec![vec![1.0]], vec![], vec!["a".into()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_split() {
        let ds = make_dataset(10);
        let (train, test) = ds.split(&[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9]);
        assert_eq!(train.n_samples(), 5);
        assert_eq!(test.n_samples(), 5);
    }

    #[test]
    fn test_class_distribution() {
        let ds = make_dataset(10);
        let dist = ds.class_distribution();
        assert_eq!(*dist.get(&DecompositionMethod::Benders).unwrap(), 5);
        assert_eq!(*dist.get(&DecompositionMethod::DantzigWolfe).unwrap(), 5);
    }
}
