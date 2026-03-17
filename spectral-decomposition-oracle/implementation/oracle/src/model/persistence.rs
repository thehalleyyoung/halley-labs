// Model serialization, versioning, and persistence.

use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata for a trained model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub model_type: String,
    pub version: String,
    pub training_date: String,
    pub feature_set: Vec<String>,
    pub n_training_samples: usize,
    pub cv_score: f64,
    pub hyperparameters: HashMap<String, String>,
    pub notes: String,
}

impl ModelMetadata {
    pub fn new(model_type: &str, version: &str) -> Self {
        Self {
            model_id: uuid::Uuid::new_v4().to_string(),
            model_type: model_type.to_string(),
            version: version.to_string(),
            training_date: chrono::Utc::now().to_rfc3339(),
            feature_set: Vec::new(),
            n_training_samples: 0,
            cv_score: 0.0,
            hyperparameters: HashMap::new(),
            notes: String::new(),
        }
    }

    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.feature_set = features;
        self
    }

    pub fn with_cv_score(mut self, score: f64) -> Self {
        self.cv_score = score;
        self
    }

    pub fn with_training_samples(mut self, n: usize) -> Self {
        self.n_training_samples = n;
        self
    }

    pub fn with_hyperparameter(mut self, key: &str, value: &str) -> Self {
        self.hyperparameters.insert(key.to_string(), value.to_string());
        self
    }

    pub fn summary(&self) -> String {
        format!(
            "Model {} ({}): v{} | {} features | CV={:.4} | trained {}",
            self.model_id.chars().take(8).collect::<String>(),
            self.model_type,
            self.version,
            self.feature_set.len(),
            self.cv_score,
            self.training_date
        )
    }
}

/// A serialized model with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    pub metadata: ModelMetadata,
    pub model_data: String, // JSON-encoded model
}

/// Model checkpoint during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_score: f64,
    pub model_data: String,
    pub timestamp: String,
}

impl ModelCheckpoint {
    pub fn new(epoch: usize, train_loss: f64, val_score: f64, model_data: String) -> Self {
        Self {
            epoch,
            train_loss,
            val_score,
            model_data,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Model store for managing multiple model versions.
#[derive(Debug, Clone)]
pub struct ModelStore {
    models: Vec<SerializedModel>,
    checkpoints: Vec<ModelCheckpoint>,
}

impl ModelStore {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Save a model with metadata.
    pub fn save(&mut self, metadata: ModelMetadata, model_json: &str) -> OracleResult<String> {
        let id = metadata.model_id.clone();
        self.models.push(SerializedModel {
            metadata,
            model_data: model_json.to_string(),
        });
        Ok(id)
    }

    /// Load a model by ID.
    pub fn load(&self, model_id: &str) -> OracleResult<&SerializedModel> {
        self.models
            .iter()
            .find(|m| m.metadata.model_id == model_id)
            .ok_or_else(|| OracleError::model_not_trained(format!("model {} not found", model_id)))
    }

    /// Load the latest model of a given type.
    pub fn load_latest(&self, model_type: &str) -> OracleResult<&SerializedModel> {
        self.models
            .iter()
            .rev()
            .find(|m| m.metadata.model_type == model_type)
            .ok_or_else(|| {
                OracleError::model_not_trained(format!("no {} model found", model_type))
            })
    }

    /// Load the best model by CV score.
    pub fn load_best(&self) -> OracleResult<&SerializedModel> {
        self.models
            .iter()
            .max_by(|a, b| {
                a.metadata
                    .cv_score
                    .partial_cmp(&b.metadata.cv_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| OracleError::model_not_trained("no models in store"))
    }

    /// List all models.
    pub fn list(&self) -> Vec<&ModelMetadata> {
        self.models.iter().map(|m| &m.metadata).collect()
    }

    /// Delete a model by ID.
    pub fn delete(&mut self, model_id: &str) -> bool {
        let len_before = self.models.len();
        self.models.retain(|m| m.metadata.model_id != model_id);
        self.models.len() < len_before
    }

    /// Save a checkpoint.
    pub fn save_checkpoint(&mut self, checkpoint: ModelCheckpoint) {
        self.checkpoints.push(checkpoint);
    }

    /// Get the best checkpoint by validation score.
    pub fn best_checkpoint(&self) -> Option<&ModelCheckpoint> {
        self.checkpoints.iter().max_by(|a, b| {
            a.val_score
                .partial_cmp(&b.val_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all checkpoints.
    pub fn checkpoints(&self) -> &[ModelCheckpoint] {
        &self.checkpoints
    }

    /// Number of stored models.
    pub fn count(&self) -> usize {
        self.models.len()
    }

    /// Export all models to JSON.
    pub fn export_json(&self) -> OracleResult<String> {
        let export: Vec<&SerializedModel> = self.models.iter().collect();
        serde_json::to_string_pretty(&export).map_err(|e| OracleError::serialization(e.to_string()))
    }

    /// Import models from JSON.
    pub fn import_json(&mut self, json: &str) -> OracleResult<usize> {
        let imported: Vec<SerializedModel> =
            serde_json::from_str(json).map_err(|e| OracleError::serialization(e.to_string()))?;
        let count = imported.len();
        self.models.extend(imported);
        Ok(count)
    }
}

/// Save a classifier model to JSON string.
pub fn serialize_random_forest(
    model: &crate::classifier::random_forest::RandomForest,
) -> OracleResult<String> {
    serde_json::to_string(model).map_err(|e| OracleError::serialization(e.to_string()))
}

/// Load a random forest from JSON string.
pub fn deserialize_random_forest(
    json: &str,
) -> OracleResult<crate::classifier::random_forest::RandomForest> {
    serde_json::from_str(json).map_err(|e| OracleError::serialization(e.to_string()))
}

/// Save a gradient boosting model to JSON.
pub fn serialize_gbt(
    model: &crate::classifier::gradient_boosting::GradientBoostingClassifier,
) -> OracleResult<String> {
    serde_json::to_string(model).map_err(|e| OracleError::serialization(e.to_string()))
}

/// Load a gradient boosting model from JSON.
pub fn deserialize_gbt(
    json: &str,
) -> OracleResult<crate::classifier::gradient_boosting::GradientBoostingClassifier> {
    serde_json::from_str(json).map_err(|e| OracleError::serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_creation() {
        let meta = ModelMetadata::new("RandomForest", "1.0.0");
        assert_eq!(meta.model_type, "RandomForest");
        assert!(!meta.model_id.is_empty());
    }

    #[test]
    fn test_model_metadata_builder() {
        let meta = ModelMetadata::new("GBT", "1.0.0")
            .with_features(vec!["f1".to_string(), "f2".to_string()])
            .with_cv_score(0.85)
            .with_training_samples(500)
            .with_hyperparameter("n_trees", "100");

        assert_eq!(meta.feature_set.len(), 2);
        assert_eq!(meta.cv_score, 0.85);
        assert_eq!(meta.n_training_samples, 500);
        assert_eq!(meta.hyperparameters["n_trees"], "100");
    }

    #[test]
    fn test_model_store_save_load() {
        let mut store = ModelStore::new();
        let meta = ModelMetadata::new("RF", "1.0.0");
        let id = meta.model_id.clone();
        store.save(meta, r#"{"data": "test"}"#).unwrap();

        let loaded = store.load(&id).unwrap();
        assert_eq!(loaded.metadata.model_type, "RF");
    }

    #[test]
    fn test_model_store_load_latest() {
        let mut store = ModelStore::new();
        store
            .save(ModelMetadata::new("RF", "1.0.0"), "model1")
            .unwrap();
        store
            .save(ModelMetadata::new("RF", "2.0.0"), "model2")
            .unwrap();

        let latest = store.load_latest("RF").unwrap();
        assert_eq!(latest.metadata.version, "2.0.0");
    }

    #[test]
    fn test_model_store_load_best() {
        let mut store = ModelStore::new();
        store
            .save(
                ModelMetadata::new("RF", "1.0.0").with_cv_score(0.8),
                "m1",
            )
            .unwrap();
        store
            .save(
                ModelMetadata::new("GBT", "1.0.0").with_cv_score(0.9),
                "m2",
            )
            .unwrap();

        let best = store.load_best().unwrap();
        assert_eq!(best.metadata.model_type, "GBT");
    }

    #[test]
    fn test_model_store_delete() {
        let mut store = ModelStore::new();
        let meta = ModelMetadata::new("RF", "1.0.0");
        let id = meta.model_id.clone();
        store.save(meta, "data").unwrap();

        assert_eq!(store.count(), 1);
        assert!(store.delete(&id));
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_model_store_list() {
        let mut store = ModelStore::new();
        store.save(ModelMetadata::new("RF", "1.0.0"), "d1").unwrap();
        store
            .save(ModelMetadata::new("GBT", "1.0.0"), "d2")
            .unwrap();

        let list = store.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_model_store_not_found() {
        let store = ModelStore::new();
        assert!(store.load("nonexistent").is_err());
    }

    #[test]
    fn test_checkpoint() {
        let cp = ModelCheckpoint::new(10, 0.5, 0.85, "model_data".to_string());
        assert_eq!(cp.epoch, 10);
        assert_eq!(cp.val_score, 0.85);
    }

    #[test]
    fn test_checkpoint_management() {
        let mut store = ModelStore::new();
        store.save_checkpoint(ModelCheckpoint::new(1, 1.0, 0.7, "cp1".into()));
        store.save_checkpoint(ModelCheckpoint::new(2, 0.8, 0.8, "cp2".into()));
        store.save_checkpoint(ModelCheckpoint::new(3, 0.6, 0.75, "cp3".into()));

        let best = store.best_checkpoint().unwrap();
        assert_eq!(best.epoch, 2);
        assert_eq!(store.checkpoints().len(), 3);
    }

    #[test]
    fn test_export_import_json() {
        let mut store = ModelStore::new();
        store.save(ModelMetadata::new("RF", "1.0"), "data").unwrap();

        let json = store.export_json().unwrap();
        let mut store2 = ModelStore::new();
        let count = store2.import_json(&json).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_serialize_random_forest() {
        use crate::classifier::random_forest::{RandomForest, RandomForestParams};

        let rf = RandomForest::new(RandomForestParams::default());
        let json = serialize_random_forest(&rf).unwrap();
        let loaded = deserialize_random_forest(&json).unwrap();
        assert_eq!(loaded.params.n_trees, rf.params.n_trees);
    }

    #[test]
    fn test_metadata_summary() {
        let meta = ModelMetadata::new("RF", "1.0.0").with_cv_score(0.85);
        let summary = meta.summary();
        assert!(summary.contains("RF"));
        assert!(summary.contains("0.8500"));
    }
}
