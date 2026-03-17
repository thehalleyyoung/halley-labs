// Main oracle pipeline: end-to-end prediction from instance to recommendation.

use crate::classifier::traits::{DecompositionMethod, FeatureVector};
use crate::error::{OracleError, OracleResult};
use crate::futility::predictor::{FutilityFeatures, FutilityPrediction, FutilityPredictor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub enable_futility_check: bool,
    pub enable_structure_detection: bool,
    pub generate_partition: bool,
    pub timeout_secs: f64,
    pub log_stages: bool,
    pub batch_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_futility_check: true,
            enable_structure_detection: true,
            generate_partition: true,
            timeout_secs: 300.0,
            log_stages: true,
            batch_size: 1,
        }
    }
}

/// Result of a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub stage_name: String,
    pub elapsed_secs: f64,
    pub success: bool,
    pub message: String,
    pub data: HashMap<String, String>,
}

impl StageResult {
    pub fn success(name: &str, elapsed: f64, msg: &str) -> Self {
        Self {
            stage_name: name.to_string(),
            elapsed_secs: elapsed,
            success: true,
            message: msg.to_string(),
            data: HashMap::new(),
        }
    }

    pub fn failure(name: &str, elapsed: f64, msg: &str) -> Self {
        Self {
            stage_name: name.to_string(),
            elapsed_secs: elapsed,
            success: false,
            message: msg.to_string(),
            data: HashMap::new(),
        }
    }

    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.data.insert(key.to_string(), value.to_string());
        self
    }
}

/// Complete pipeline result for one instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub instance_name: String,
    pub recommended_method: Option<DecompositionMethod>,
    pub confidence: f64,
    pub class_probabilities: Vec<f64>,
    pub is_futile: Option<bool>,
    pub futility_score: Option<f64>,
    pub features: Option<FeatureVector>,
    pub stages: Vec<StageResult>,
    pub total_elapsed_secs: f64,
    pub partial: bool, // true if some stages failed
}

impl PipelineResult {
    pub fn summary(&self) -> String {
        let method_str = self
            .recommended_method
            .map(|m| m.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        format!(
            "{}: {} (conf={:.3}, futile={:?}, time={:.2}s, partial={})",
            self.instance_name,
            method_str,
            self.confidence,
            self.is_futile,
            self.total_elapsed_secs,
            self.partial
        )
    }

    /// Whether the pipeline completed all stages successfully.
    pub fn is_complete(&self) -> bool {
        !self.partial && self.recommended_method.is_some()
    }

    /// List failed stages.
    pub fn failed_stages(&self) -> Vec<&StageResult> {
        self.stages.iter().filter(|s| !s.success).collect()
    }
}

/// Spectral features extracted from an instance (local type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    pub eigenvalues: Vec<f64>,
    pub spectral_gap: f64,
    pub fiedler_value: f64,
    pub effective_dimension: f64,
    pub spectral_ratio: f64,
    pub algebraic_connectivity: f64,
    pub normalized_laplacian_gap: f64,
    pub entropy: f64,
}

impl SpectralFeatures {
    pub fn to_feature_vector(&self) -> FeatureVector {
        vec![
            self.spectral_gap,
            self.fiedler_value,
            self.effective_dimension,
            self.spectral_ratio,
            self.algebraic_connectivity,
            self.normalized_laplacian_gap,
            self.entropy,
            self.eigenvalues.len() as f64,
        ]
    }
}

/// Instance metadata for the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub name: String,
    pub n_variables: usize,
    pub n_constraints: usize,
    pub n_nonzeros: usize,
    pub has_integers: bool,
    pub density: f64,
}

impl InstanceInfo {
    pub fn syntactic_features(&self) -> FeatureVector {
        vec![
            self.n_variables as f64,
            self.n_constraints as f64,
            self.n_nonzeros as f64,
            if self.has_integers { 1.0 } else { 0.0 },
            self.density,
            self.n_constraints as f64 / self.n_variables.max(1) as f64,
            (self.n_nonzeros as f64) / (self.n_variables as f64 * self.n_constraints as f64).max(1.0),
        ]
    }
}

/// The main oracle pipeline.
pub struct OraclePipeline {
    pub config: PipelineConfig,
    pub classifier: Option<Box<dyn crate::classifier::traits::Classifier>>,
    pub futility_predictor: Option<FutilityPredictor>,
}

impl OraclePipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            classifier: None,
            futility_predictor: None,
        }
    }

    pub fn with_classifier(mut self, classifier: Box<dyn crate::classifier::traits::Classifier>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    pub fn with_futility_predictor(mut self, predictor: FutilityPredictor) -> Self {
        self.futility_predictor = Some(predictor);
        self
    }

    /// Run the full pipeline for a single instance.
    pub fn run(
        &self,
        instance: &InstanceInfo,
        features: &FeatureVector,
    ) -> PipelineResult {
        let start = Instant::now();
        let mut stages = Vec::new();
        let mut partial = false;

        // Stage 1: Feature extraction/validation
        let stage_start = Instant::now();
        let feature_result = self.validate_features(features);
        let elapsed = stage_start.elapsed().as_secs_f64();
        match &feature_result {
            Ok(_) => {
                stages.push(StageResult::success(
                    "feature_validation",
                    elapsed,
                    &format!("{} features validated", features.len()),
                ));
            }
            Err(e) => {
                stages.push(StageResult::failure("feature_validation", elapsed, &e.to_string()));
                return PipelineResult {
                    instance_name: instance.name.clone(),
                    recommended_method: None,
                    confidence: 0.0,
                    class_probabilities: vec![],
                    is_futile: None,
                    futility_score: None,
                    features: Some(features.clone()),
                    stages,
                    total_elapsed_secs: start.elapsed().as_secs_f64(),
                    partial: true,
                };
            }
        }

        // Stage 2: Futility check
        let mut is_futile = None;
        let mut futility_score = None;

        if self.config.enable_futility_check {
            let stage_start = Instant::now();
            let futility_result = self.run_futility_check(instance, features);
            let elapsed = stage_start.elapsed().as_secs_f64();

            match futility_result {
                Ok(pred) => {
                    is_futile = Some(pred.is_futile);
                    futility_score = Some(pred.futility_score);
                    stages.push(
                        StageResult::success(
                            "futility_check",
                            elapsed,
                            &format!("futile={}, score={:.3}", pred.is_futile, pred.futility_score),
                        )
                        .with_data("is_futile", &pred.is_futile.to_string()),
                    );

                    if pred.is_futile {
                        return PipelineResult {
                            instance_name: instance.name.clone(),
                            recommended_method: Some(DecompositionMethod::None),
                            confidence: pred.confidence,
                            class_probabilities: vec![0.0, 0.0, 0.0, 1.0],
                            is_futile,
                            futility_score,
                            features: Some(features.clone()),
                            stages,
                            total_elapsed_secs: start.elapsed().as_secs_f64(),
                            partial: false,
                        };
                    }
                }
                Err(e) => {
                    stages.push(StageResult::failure("futility_check", elapsed, &e.to_string()));
                    partial = true;
                    // Continue despite futility check failure
                }
            }
        }

        // Stage 3: Classification
        let stage_start = Instant::now();
        let classification_result = self.run_classification(features);
        let elapsed = stage_start.elapsed().as_secs_f64();

        let (recommended_method, confidence, class_proba) = match classification_result {
            Ok((method, conf, proba)) => {
                stages.push(
                    StageResult::success(
                        "classification",
                        elapsed,
                        &format!("predicted={}, confidence={:.3}", method, conf),
                    )
                    .with_data("method", &method.to_string()),
                );
                (Some(method), conf, proba)
            }
            Err(e) => {
                stages.push(StageResult::failure("classification", elapsed, &e.to_string()));
                partial = true;
                (None, 0.0, vec![])
            }
        };

        PipelineResult {
            instance_name: instance.name.clone(),
            recommended_method,
            confidence,
            class_probabilities: class_proba,
            is_futile,
            futility_score,
            features: Some(features.clone()),
            stages,
            total_elapsed_secs: start.elapsed().as_secs_f64(),
            partial,
        }
    }

    /// Run the pipeline in batch mode.
    pub fn run_batch(
        &self,
        instances: &[(InstanceInfo, FeatureVector)],
    ) -> Vec<PipelineResult> {
        instances
            .iter()
            .map(|(info, features)| self.run(info, features))
            .collect()
    }

    fn validate_features(&self, features: &FeatureVector) -> OracleResult<()> {
        if features.is_empty() {
            return Err(OracleError::feature_extraction("empty feature vector"));
        }
        for (i, &val) in features.iter().enumerate() {
            if val.is_nan() {
                return Err(OracleError::feature_extraction(format!(
                    "NaN at feature index {}",
                    i
                )));
            }
            if val.is_infinite() {
                return Err(OracleError::feature_extraction(format!(
                    "Inf at feature index {}",
                    i
                )));
            }
        }
        Ok(())
    }

    fn run_futility_check(
        &self,
        instance: &InstanceInfo,
        features: &FeatureVector,
    ) -> OracleResult<FutilityPrediction> {
        let predictor = self
            .futility_predictor
            .as_ref()
            .ok_or_else(|| OracleError::model_not_trained("no futility predictor configured"))?;

        let ff = FutilityFeatures {
            spectral_gap: features.get(0).copied().unwrap_or(0.0),
            spectral_ratio: features.get(1).copied().unwrap_or(0.0),
            effective_dimension: features.get(2).copied().unwrap_or(0.0),
            block_separability: features.get(3).copied().unwrap_or(0.0),
            constraint_density: instance.density,
            variable_count: instance.n_variables,
            constraint_count: instance.n_constraints,
            nonzero_density: instance.density,
        };

        predictor.predict(&ff)
    }

    fn run_classification(
        &self,
        features: &FeatureVector,
    ) -> OracleResult<(DecompositionMethod, f64, Vec<f64>)> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| OracleError::model_not_trained("no classifier configured"))?;

        let method = classifier.predict(features)?;
        let proba = classifier.predict_proba(features)?;
        let confidence = proba
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);

        Ok((method, confidence, proba))
    }

    /// Get timing breakdown by stage.
    pub fn timing_breakdown(result: &PipelineResult) -> HashMap<String, f64> {
        let mut timings = HashMap::new();
        for stage in &result.stages {
            timings.insert(stage.stage_name.clone(), stage.elapsed_secs);
        }
        timings
    }
}

/// Aggregate statistics for batch results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub total_instances: usize,
    pub complete_count: usize,
    pub partial_count: usize,
    pub futile_count: usize,
    pub method_distribution: HashMap<String, usize>,
    pub avg_confidence: f64,
    pub avg_elapsed_secs: f64,
    pub total_elapsed_secs: f64,
}

impl BatchStatistics {
    pub fn compute(results: &[PipelineResult]) -> Self {
        let total = results.len();
        let complete = results.iter().filter(|r| r.is_complete()).count();
        let partial = results.iter().filter(|r| r.partial).count();
        let futile = results
            .iter()
            .filter(|r| r.is_futile == Some(true))
            .count();

        let mut method_dist = HashMap::new();
        for result in results {
            if let Some(method) = result.recommended_method {
                *method_dist.entry(method.to_string()).or_insert(0) += 1;
            }
        }

        let total_elapsed: f64 = results.iter().map(|r| r.total_elapsed_secs).sum();
        let avg_confidence = if total > 0 {
            results.iter().map(|r| r.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };
        let avg_elapsed = if total > 0 {
            total_elapsed / total as f64
        } else {
            0.0
        };

        BatchStatistics {
            total_instances: total,
            complete_count: complete,
            partial_count: partial,
            futile_count: futile,
            method_distribution: method_dist,
            avg_confidence,
            avg_elapsed_secs: avg_elapsed,
            total_elapsed_secs: total_elapsed,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Batch: {}/{} complete, {} futile | avg conf={:.3}, avg time={:.3}s",
            self.complete_count,
            self.total_instances,
            self.futile_count,
            self.avg_confidence,
            self.avg_elapsed_secs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instance() -> InstanceInfo {
        InstanceInfo {
            name: "test_instance".to_string(),
            n_variables: 100,
            n_constraints: 50,
            n_nonzeros: 500,
            has_integers: true,
            density: 0.1,
        }
    }

    fn make_features() -> FeatureVector {
        vec![0.5, 1.2, 10.0, 0.8, 0.1, 100.0, 50.0, 0.05]
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.enable_futility_check);
        assert_eq!(config.timeout_secs, 300.0);
    }

    #[test]
    fn test_pipeline_no_classifier() {
        let config = PipelineConfig {
            enable_futility_check: false,
            ..Default::default()
        };
        let pipeline = OraclePipeline::new(config);
        let result = pipeline.run(&make_instance(), &make_features());
        assert!(result.partial);
        assert!(result.recommended_method.is_none());
    }

    #[test]
    fn test_pipeline_nan_features() {
        let pipeline = OraclePipeline::new(PipelineConfig::default());
        let result = pipeline.run(&make_instance(), &vec![f64::NAN, 1.0]);
        assert!(result.partial);
    }

    #[test]
    fn test_pipeline_empty_features() {
        let pipeline = OraclePipeline::new(PipelineConfig::default());
        let result = pipeline.run(&make_instance(), &vec![]);
        assert!(result.partial);
    }

    #[test]
    fn test_stage_result() {
        let stage = StageResult::success("test", 0.5, "ok")
            .with_data("key", "value");
        assert!(stage.success);
        assert_eq!(stage.data["key"], "value");
    }

    #[test]
    fn test_instance_syntactic_features() {
        let info = make_instance();
        let feats = info.syntactic_features();
        assert_eq!(feats.len(), 7);
        assert_eq!(feats[0], 100.0);
    }

    #[test]
    fn test_spectral_features_to_vector() {
        let sf = SpectralFeatures {
            eigenvalues: vec![1.0, 0.5, 0.1],
            spectral_gap: 0.5,
            fiedler_value: 0.3,
            effective_dimension: 5.0,
            spectral_ratio: 2.0,
            algebraic_connectivity: 0.3,
            normalized_laplacian_gap: 0.4,
            entropy: 1.5,
        };
        let v = sf.to_feature_vector();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pipeline_result_summary() {
        let result = PipelineResult {
            instance_name: "test".to_string(),
            recommended_method: Some(DecompositionMethod::Benders),
            confidence: 0.85,
            class_probabilities: vec![0.85, 0.1, 0.03, 0.02],
            is_futile: Some(false),
            futility_score: Some(0.2),
            features: None,
            stages: vec![],
            total_elapsed_secs: 1.5,
            partial: false,
        };
        let summary = result.summary();
        assert!(summary.contains("Benders"));
        assert!(summary.contains("0.850"));
    }

    #[test]
    fn test_pipeline_result_complete() {
        let result = PipelineResult {
            instance_name: "test".to_string(),
            recommended_method: Some(DecompositionMethod::Benders),
            confidence: 0.9,
            class_probabilities: vec![],
            is_futile: None,
            futility_score: None,
            features: None,
            stages: vec![],
            total_elapsed_secs: 1.0,
            partial: false,
        };
        assert!(result.is_complete());
    }

    #[test]
    fn test_batch_statistics() {
        let results = vec![
            PipelineResult {
                instance_name: "a".to_string(),
                recommended_method: Some(DecompositionMethod::Benders),
                confidence: 0.9,
                class_probabilities: vec![],
                is_futile: Some(false),
                futility_score: None,
                features: None,
                stages: vec![],
                total_elapsed_secs: 1.0,
                partial: false,
            },
            PipelineResult {
                instance_name: "b".to_string(),
                recommended_method: Some(DecompositionMethod::None),
                confidence: 0.8,
                class_probabilities: vec![],
                is_futile: Some(true),
                futility_score: None,
                features: None,
                stages: vec![],
                total_elapsed_secs: 0.5,
                partial: false,
            },
        ];
        let stats = BatchStatistics::compute(&results);
        assert_eq!(stats.total_instances, 2);
        assert_eq!(stats.futile_count, 1);
        assert!((stats.avg_confidence - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_timing_breakdown() {
        let result = PipelineResult {
            instance_name: "test".to_string(),
            recommended_method: None,
            confidence: 0.0,
            class_probabilities: vec![],
            is_futile: None,
            futility_score: None,
            features: None,
            stages: vec![
                StageResult::success("stage1", 0.5, "ok"),
                StageResult::success("stage2", 1.0, "ok"),
            ],
            total_elapsed_secs: 1.5,
            partial: false,
        };
        let timings = OraclePipeline::timing_breakdown(&result);
        assert_eq!(*timings.get("stage1").unwrap(), 0.5);
    }

    #[test]
    fn test_batch_statistics_summary() {
        let stats = BatchStatistics {
            total_instances: 10,
            complete_count: 8,
            partial_count: 2,
            futile_count: 3,
            method_distribution: HashMap::new(),
            avg_confidence: 0.75,
            avg_elapsed_secs: 1.2,
            total_elapsed_secs: 12.0,
        };
        let summary = stats.summary();
        assert!(summary.contains("8/10"));
    }
}
