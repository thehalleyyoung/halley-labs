//! Core traits for pipeline stages, transformations, and oracles.

use crate::distance::{DistanceMetric, DistanceValue};
use crate::error::Result;
use crate::ir::{IRType, IntermediateRepresentation};
use crate::types::StageId;
use serde::{Deserialize, Serialize};

/// A single stage in an NLP processing pipeline.
pub trait PipelineStage: Send + Sync {
    fn id(&self) -> &StageId;
    fn name(&self) -> &str;
    fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation>;
    fn input_type(&self) -> IRType;
    fn output_type(&self) -> IRType;
}

/// Detail record from checking a metamorphic relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRCheckDetail {
    pub passed: bool,
    pub violation_magnitude: f64,
    pub expected: String,
    pub actual: String,
    pub explanation: String,
}

/// A metamorphic relation that can be checked between original and transformed IRs.
pub trait MetamorphicRelation: Send + Sync {
    fn name(&self) -> &str;
    fn check(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> Result<bool>;
    fn check_with_detail(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> Result<MRCheckDetail>;
    fn tolerance(&self) -> f64 {
        0.0
    }
}

/// A text-level transformation that produces a semantically related input.
pub trait Transformation: Send + Sync {
    fn name(&self) -> &str;
    fn transform(&self, input: &str) -> Result<String>;
    fn expected_invariant(&self) -> &str;
}

/// Computes distance between two intermediate representations.
pub trait DistanceComputer: Send + Sync {
    /// Returns the distance metric used by this computer.
    fn metric(&self) -> DistanceMetric {
        DistanceMetric::Custom("unknown".into())
    }

    /// Compute the distance between two intermediate representations.
    fn compute(
        &self,
        a: &IntermediateRepresentation,
        b: &IntermediateRepresentation,
    ) -> Result<DistanceValue>;
}

/// Checks whether a sentence is grammatically / semantically valid.
pub trait ValidityOracle: Send + Sync {
    fn is_valid(&self, sentence: &str) -> Result<bool>;
}
