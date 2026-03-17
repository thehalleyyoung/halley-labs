//! Intermediate representation types for NLP pipeline stages.

use crate::types::{Sentence, StageId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The type of intermediate representation produced by a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IRType {
    RawText,
    Tokenized,
    PosTagged,
    Parsed,
    DependencyParsed,
    EntityRecognized,
    EntityAnnotated,
    SentimentScored,
    FeatureVector,
    Custom(String),
}

/// A single intermediate representation at one pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateRepresentation {
    pub ir_type: IRType,
    pub sentence: Sentence,
    pub data: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub labels: Vec<String>,
}

impl IntermediateRepresentation {
    pub fn new(ir_type: IRType, sentence: Sentence) -> Self {
        Self {
            ir_type,
            sentence,
            data: HashMap::new(),
            confidence: None,
            labels: Vec::new(),
        }
    }

    pub fn with_data(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// A snapshot of the IR at a specific pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRSnapshot {
    pub stage_id: StageId,
    pub stage_name: String,
    pub ir: IntermediateRepresentation,
}

/// An ordered sequence of IR snapshots through a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRSequence {
    pub snapshots: Vec<IRSnapshot>,
}

impl IRSequence {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    pub fn push(&mut self, snapshot: IRSnapshot) {
        self.snapshots.push(snapshot);
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&IRSnapshot> {
        self.snapshots.get(index)
    }
}

impl Default for IRSequence {
    fn default() -> Self {
        Self::new()
    }
}
