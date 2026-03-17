//! Core metamorphic testing framework types and traits.
//!
//! Re-exports fundamental traits from `shared_types` and provides additional
//! framework types for differential testing and fault localisation.

pub mod calibration;
pub mod composition;
pub mod execution;
pub mod property;
pub mod relation;
pub mod test_case;

pub use shared_types::{
    ConfidenceInterval, DistanceComputer, DistanceMetric, DistanceValue, IRSequence, IRSnapshot,
    IRType, IntermediateRepresentation, LocalizerError, MRCheckDetail, MetamorphicRelation,
    PipelineStage, Result, RunningStats, Sentence, StageDistance, StageId, Token,
    Transformation,
};

// Re-export submodule key types.
pub use calibration::{
    CalibrationBaseline, CalibrationCorpus, CalibrationEngine, CalibrationReport,
    CalibrationSample,
};
pub use composition::{
    ComposedTransformation, CompositionConfidence, CompositionOptimizer, CompositionValidator,
    CoverageMatrix, PositionOverlap, SyntacticPosition,
};
pub use execution::{
    BatchExecutor, DefaultPipelineRunner, ExecutionCache, ExecutionContext, ExecutionEngine,
    ExecutionPlan, ParallelExecutor, PipelineExecution, PipelineRunner, StageExecutionResult,
};
pub use property::{
    CompositionRule, DisjointPositionChecker, MetamorphicProperty, PropertyChecker,
    PropertyGenerator, PropertyResult, PropertyScope,
};
pub use relation::{
    EntityPreservationMR, MRCheckResult, MRDefinition, MRRegistry, MRType, NegationFlipMR,
    SemanticEquivalenceMR, SentimentPreservationMR, SyntacticConsistencyMR,
};
pub use test_case::{
    CoverageTracker, TestCase, TestCaseGenerator, TestResult, TestSuite, TestSuiteBuilder,
    ViolationRecord, ViolationSeverity,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of running a differential test (original vs. transformed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialResult {
    pub original_trace: IRSequence,
    pub transformed_trace: IRSequence,
    pub stage_distances: Vec<StageDistance>,
    pub suspicious_stages: Vec<StageId>,
}

/// Localisation verdict for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationVerdict {
    pub stage_id: StageId,
    pub score: f64,
    pub rank: usize,
    pub evidence: Vec<String>,
}

/// Full localisation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationReport {
    pub verdicts: Vec<LocalizationVerdict>,
    pub summary: String,
    pub metadata: HashMap<String, String>,
}

/// Engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub distance_threshold: f64,
    pub top_k_suspects: usize,
    pub normalize_distances: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 0.1,
            top_k_suspects: 3,
            normalize_distances: true,
        }
    }
}

/// A stage wrapper that records IRs during execution.
pub struct InstrumentedStage {
    inner: Box<dyn PipelineStage>,
    pub recorded: std::sync::Mutex<Vec<IntermediateRepresentation>>,
}

impl InstrumentedStage {
    pub fn new(stage: Box<dyn PipelineStage>) -> Self {
        Self {
            inner: stage,
            recorded: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn process_and_record(
        &self,
        input: &IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation> {
        let output = self.inner.process(input)?;
        if let Ok(mut rec) = self.recorded.lock() {
            rec.push(output.clone());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_default() {
        let cfg = EngineConfig::default();
        assert_eq!(cfg.top_k_suspects, 3);
        assert!(cfg.normalize_distances);
    }

    #[test]
    fn test_localization_verdict() {
        let v = LocalizationVerdict {
            stage_id: StageId::new("tok"),
            score: 0.8,
            rank: 1,
            evidence: vec!["high distance".into()],
        };
        assert_eq!(v.rank, 1);
    }

    #[test]
    fn test_differential_result() {
        let dr = DifferentialResult {
            original_trace: IRSequence::new(),
            transformed_trace: IRSequence::new(),
            stage_distances: vec![],
            suspicious_stages: vec![StageId::new("tok")],
        };
        assert_eq!(dr.suspicious_stages.len(), 1);
    }

    #[test]
    fn test_localization_report() {
        let r = LocalizationReport {
            verdicts: vec![],
            summary: "ok".into(),
            metadata: HashMap::new(),
        };
        assert!(r.verdicts.is_empty());
    }
}
