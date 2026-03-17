//! Pipeline definition, management, adapters, and registry.

use shared_types::{
    IRSequence, IRSnapshot, IRType, IntermediateRepresentation, LocalizerError,
    PipelineStage, Result, Sentence, StageId,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::stage::{
    DependencyParserStage, EmbedderStage, NERStage, PosTaggerStage,
    SentimentClassifierStage, TokenizerStage,
};

// ── PipelineTrace ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTrace {
    pub input: String,
    pub per_stage_irs: Vec<IRSnapshot>,
    pub total_time_ms: u64,
    pub success: bool,
}

// ── Pipeline ────────────────────────────────────────────────────────────────

/// An ordered pipeline of stages with metadata.
pub struct Pipeline {
    pub id: String,
    pub name: String,
    stages: Vec<Box<dyn PipelineStage>>,
    pub metadata: HashMap<String, String>,
}

impl Pipeline {
    /// Execute the full pipeline on raw text.
    pub fn execute(&self, input: &str) -> Result<PipelineTrace> {
        let start = Instant::now();
        let sentence = Sentence::from_text(input);
        let mut ir = IntermediateRepresentation::new(IRType::RawText, sentence);
        let mut snapshots = Vec::new();

        for stage in &self.stages {
            ir = stage.process(&ir)?;
            snapshots.push(IRSnapshot { stage_id: stage.id().clone(), stage_name: stage.name().to_string(), ir: ir.clone() });
        }

        Ok(PipelineTrace {
            input: input.to_string(),
            per_stage_irs: snapshots,
            total_time_ms: start.elapsed().as_millis() as u64,
            success: true,
        })
    }

    /// Execute up to (and including) the stage with the given id.
    pub fn execute_prefix(&self, input: &str, up_to_stage: &StageId) -> Result<PipelineTrace> {
        let start = Instant::now();
        let sentence = Sentence::from_text(input);
        let mut ir = IntermediateRepresentation::new(IRType::RawText, sentence);
        let mut snapshots = Vec::new();

        for stage in &self.stages {
            ir = stage.process(&ir)?;
            snapshots.push(IRSnapshot { stage_id: stage.id().clone(), stage_name: stage.name().to_string(), ir: ir.clone() });
            if stage.id() == up_to_stage {
                break;
            }
        }

        Ok(PipelineTrace {
            input: input.to_string(),
            per_stage_irs: snapshots,
            total_time_ms: start.elapsed().as_millis() as u64,
            success: true,
        })
    }

    /// Execute from an existing IR through the remaining stages starting at `from_stage`.
    pub fn execute_from(
        &self,
        ir: &IntermediateRepresentation,
        from_stage: &StageId,
    ) -> Result<PipelineTrace> {
        let start = Instant::now();
        let mut current = ir.clone();
        let mut snapshots = Vec::new();
        let mut started = false;

        for stage in &self.stages {
            if stage.id() == from_stage {
                started = true;
            }
            if started {
                current = stage.process(&current)?;
                snapshots.push(IRSnapshot { stage_id: stage.id().clone(), stage_name: stage.name().to_string(), ir: current.clone() });
            }
        }

        if !started {
            return Err(LocalizerError::validation("not_found", format!(
                "Stage '{}' not found in pipeline",
                from_stage
            )));
        }

        Ok(PipelineTrace {
            input: String::new(),
            per_stage_irs: snapshots,
            total_time_ms: start.elapsed().as_millis() as u64,
            success: true,
        })
    }

    /// Look up a stage by id.
    pub fn get_stage(&self, id: &StageId) -> Option<&dyn PipelineStage> {
        self.stages.iter().find(|s| s.id() == id).map(|s| s.as_ref())
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.name().to_string()).collect()
    }

    pub fn stage_ids(&self) -> Vec<StageId> {
        self.stages.iter().map(|s| s.id().clone()).collect()
    }
}

// ── PipelineBuilder ─────────────────────────────────────────────────────────

pub struct PipelineBuilder {
    id: String,
    name: String,
    stages: Vec<Box<dyn PipelineStage>>,
    metadata: HashMap<String, String>,
}

impl PipelineBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        let n = name.into();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: n,
            stages: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> Pipeline {
        Pipeline {
            id: self.id,
            name: self.name,
            stages: self.stages,
            metadata: self.metadata,
        }
    }
}

// ── PipelineAdapter trait ───────────────────────────────────────────────────

/// Adapter that wraps a particular NLP framework's pipeline.
pub trait PipelineAdapter: Send + Sync {
    fn name(&self) -> &str;
    fn stages(&self) -> Vec<StageId>;
    fn execute(&self, input: &str) -> Result<PipelineTrace>;
    fn execute_prefix(&self, input: &str, up_to_stage: &StageId) -> Result<PipelineTrace>;
    fn execute_from(&self, ir: &IntermediateRepresentation, from_stage: &StageId) -> Result<PipelineTrace>;
    fn get_ir_at_stage(&self, input: &str, stage_id: &StageId) -> Result<IntermediateRepresentation>;
}

// ── SpacyLikeAdapter ────────────────────────────────────────────────────────

/// Models a spaCy-like 4-stage pipeline: tokenize → tag → parse → ner.
pub struct SpacyLikeAdapter {
    pipeline: Pipeline,
}

impl SpacyLikeAdapter {
    pub fn new() -> Self {
        let pipeline = PipelineBuilder::new("spacy-like")
            .id("spacy")
            .add_stage(Box::new(TokenizerStage::new().with_id("spacy_tok")))
            .add_stage(Box::new(PosTaggerStage::new().with_id("spacy_tag")))
            .add_stage(Box::new(DependencyParserStage::new().with_id("spacy_parse")))
            .add_stage(Box::new(NERStage::new().with_id("spacy_ner")))
            .metadata("framework", "spacy")
            .build();
        Self { pipeline }
    }
}

impl Default for SpacyLikeAdapter {
    fn default() -> Self { Self::new() }
}

impl PipelineAdapter for SpacyLikeAdapter {
    fn name(&self) -> &str { "spacy-like" }

    fn stages(&self) -> Vec<StageId> {
        self.pipeline.stage_ids()
    }

    fn execute(&self, input: &str) -> Result<PipelineTrace> {
        self.pipeline.execute(input)
    }

    fn execute_prefix(&self, input: &str, up_to_stage: &StageId) -> Result<PipelineTrace> {
        self.pipeline.execute_prefix(input, up_to_stage)
    }

    fn execute_from(&self, ir: &IntermediateRepresentation, from_stage: &StageId) -> Result<PipelineTrace> {
        self.pipeline.execute_from(ir, from_stage)
    }

    fn get_ir_at_stage(&self, input: &str, stage_id: &StageId) -> Result<IntermediateRepresentation> {
        let trace = self.pipeline.execute_prefix(input, stage_id)?;
        trace.per_stage_irs.last()
            .map(|s| s.ir.clone())
            .ok_or_else(|| LocalizerError::ValidationError { context: "not_found".into(), message: "Stage produced no output".into() })
    }
}

// ── HuggingFaceLikeAdapter ──────────────────────────────────────────────────

/// Models a HuggingFace-like 3-stage pipeline: tokenize → encode → classify.
pub struct HuggingFaceLikeAdapter {
    pipeline: Pipeline,
}

impl HuggingFaceLikeAdapter {
    pub fn new() -> Self {
        let pipeline = PipelineBuilder::new("huggingface-like")
            .id("hf")
            .add_stage(Box::new(TokenizerStage::new().with_id("hf_tok")))
            .add_stage(Box::new(EmbedderStage::new(128).with_id("hf_enc")))
            .add_stage(Box::new(SentimentClassifierStage::new().with_id("hf_cls")))
            .metadata("framework", "huggingface")
            .build();
        Self { pipeline }
    }
}

impl Default for HuggingFaceLikeAdapter {
    fn default() -> Self { Self::new() }
}

impl PipelineAdapter for HuggingFaceLikeAdapter {
    fn name(&self) -> &str { "huggingface-like" }

    fn stages(&self) -> Vec<StageId> {
        self.pipeline.stage_ids()
    }

    fn execute(&self, input: &str) -> Result<PipelineTrace> {
        self.pipeline.execute(input)
    }

    fn execute_prefix(&self, input: &str, up_to_stage: &StageId) -> Result<PipelineTrace> {
        self.pipeline.execute_prefix(input, up_to_stage)
    }

    fn execute_from(&self, ir: &IntermediateRepresentation, from_stage: &StageId) -> Result<PipelineTrace> {
        self.pipeline.execute_from(ir, from_stage)
    }

    fn get_ir_at_stage(&self, input: &str, stage_id: &StageId) -> Result<IntermediateRepresentation> {
        let trace = self.pipeline.execute_prefix(input, stage_id)?;
        trace.per_stage_irs.last()
            .map(|s| s.ir.clone())
            .ok_or_else(|| LocalizerError::ValidationError { context: "not_found".into(), message: "Stage produced no output".into() })
    }
}

// ── StanzaLikeAdapter ───────────────────────────────────────────────────────

/// Models a Stanza-like 4-stage pipeline: tokenize → tag → parse → ner.
pub struct StanzaLikeAdapter {
    pipeline: Pipeline,
}

impl StanzaLikeAdapter {
    pub fn new() -> Self {
        let pipeline = PipelineBuilder::new("stanza-like")
            .id("stanza")
            .add_stage(Box::new(TokenizerStage::new().with_id("stanza_tok")))
            .add_stage(Box::new(PosTaggerStage::new().with_id("stanza_tag")))
            .add_stage(Box::new(DependencyParserStage::new().with_id("stanza_parse")))
            .add_stage(Box::new(NERStage::new().with_id("stanza_ner")))
            .metadata("framework", "stanza")
            .build();
        Self { pipeline }
    }
}

impl Default for StanzaLikeAdapter {
    fn default() -> Self { Self::new() }
}

impl PipelineAdapter for StanzaLikeAdapter {
    fn name(&self) -> &str { "stanza-like" }

    fn stages(&self) -> Vec<StageId> {
        self.pipeline.stage_ids()
    }

    fn execute(&self, input: &str) -> Result<PipelineTrace> {
        self.pipeline.execute(input)
    }

    fn execute_prefix(&self, input: &str, up_to_stage: &StageId) -> Result<PipelineTrace> {
        self.pipeline.execute_prefix(input, up_to_stage)
    }

    fn execute_from(&self, ir: &IntermediateRepresentation, from_stage: &StageId) -> Result<PipelineTrace> {
        self.pipeline.execute_from(ir, from_stage)
    }

    fn get_ir_at_stage(&self, input: &str, stage_id: &StageId) -> Result<IntermediateRepresentation> {
        let trace = self.pipeline.execute_prefix(input, stage_id)?;
        trace.per_stage_irs.last()
            .map(|s| s.ir.clone())
            .ok_or_else(|| LocalizerError::ValidationError { context: "not_found".into(), message: "Stage produced no output".into() })
    }
}

// ── AdapterRegistry ─────────────────────────────────────────────────────────

/// Stores and retrieves pipeline adapters by name.
pub struct AdapterRegistry {
    adapters: HashMap<String, Box<dyn PipelineAdapter>>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }

    pub fn register(&mut self, adapter: Box<dyn PipelineAdapter>) {
        self.adapters.insert(adapter.name().to_string(), adapter);
    }

    pub fn get(&self, name: &str) -> Option<&dyn PipelineAdapter> {
        self.adapters.get(name).map(|a| a.as_ref())
    }

    pub fn names(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    pub fn count(&self) -> usize {
        self.adapters.len()
    }

    /// Pre-populate with all built-in adapters.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(SpacyLikeAdapter::new()));
        registry.register(Box::new(HuggingFaceLikeAdapter::new()));
        registry.register(Box::new(StanzaLikeAdapter::new()));
        registry
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_execute_full() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Box::new(TokenizerStage::new()))
            .add_stage(Box::new(PosTaggerStage::new()))
            .build();
        let trace = pipeline.execute("The cat sat").unwrap();
        assert!(trace.success);
        assert_eq!(trace.per_stage_irs.len(), 2);
    }

    #[test]
    fn test_pipeline_execute_prefix() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Box::new(TokenizerStage::new()))
            .add_stage(Box::new(PosTaggerStage::new()))
            .build();
        let tok_id = StageId::new("tokenizer");
        let trace = pipeline.execute_prefix("The cat sat", &tok_id).unwrap();
        assert_eq!(trace.per_stage_irs.len(), 1);
    }

    #[test]
    fn test_pipeline_execute_from() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Box::new(TokenizerStage::new()))
            .add_stage(Box::new(PosTaggerStage::new()))
            .build();
        let full = pipeline.execute("The cat sat").unwrap();
        let tok_ir = &full.per_stage_irs[0].ir;
        let pos_id = StageId::new("pos_tagger");
        let trace = pipeline.execute_from(tok_ir, &pos_id).unwrap();
        assert_eq!(trace.per_stage_irs.len(), 1);
    }

    #[test]
    fn test_pipeline_stage_count() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Box::new(TokenizerStage::new()))
            .add_stage(Box::new(PosTaggerStage::new()))
            .add_stage(Box::new(DependencyParserStage::new()))
            .build();
        assert_eq!(pipeline.stage_count(), 3);
    }

    #[test]
    fn test_pipeline_get_stage() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Box::new(TokenizerStage::new()))
            .build();
        let id = StageId::new("tokenizer");
        assert!(pipeline.get_stage(&id).is_some());
        let missing = StageId::new("missing");
        assert!(pipeline.get_stage(&missing).is_none());
    }

    #[test]
    fn test_pipeline_stage_names() {
        let pipeline = PipelineBuilder::new("test")
            .add_stage(Box::new(TokenizerStage::new()))
            .add_stage(Box::new(PosTaggerStage::new()))
            .build();
        let names = pipeline.stage_names();
        assert!(names.contains(&"tokenizer".to_string()));
        assert!(names.contains(&"pos_tagger".to_string()));
    }

    #[test]
    fn test_spacy_adapter() {
        let adapter = SpacyLikeAdapter::new();
        assert_eq!(adapter.name(), "spacy-like");
        let trace = adapter.execute("John lives in London").unwrap();
        assert!(trace.success);
        assert_eq!(trace.per_stage_irs.len(), 4);
    }

    #[test]
    fn test_huggingface_adapter() {
        let adapter = HuggingFaceLikeAdapter::new();
        assert_eq!(adapter.name(), "huggingface-like");
        let trace = adapter.execute("This is great").unwrap();
        assert!(trace.success);
        assert_eq!(trace.per_stage_irs.len(), 3);
    }

    #[test]
    fn test_stanza_adapter() {
        let adapter = StanzaLikeAdapter::new();
        let trace = adapter.execute("Dogs run fast").unwrap();
        assert!(trace.success);
        assert_eq!(trace.per_stage_irs.len(), 4);
    }

    #[test]
    fn test_adapter_get_ir_at_stage() {
        let adapter = SpacyLikeAdapter::new();
        let stages = adapter.stages();
        let ir = adapter.get_ir_at_stage("The cat sat", &stages[0]).unwrap();
        assert_eq!(ir.ir_type, IRType::Tokenized);
    }

    #[test]
    fn test_adapter_registry() {
        let registry = AdapterRegistry::with_defaults();
        assert_eq!(registry.count(), 3);
        assert!(registry.get("spacy-like").is_some());
        assert!(registry.get("huggingface-like").is_some());
        assert!(registry.get("stanza-like").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_adapter_registry_register() {
        let mut registry = AdapterRegistry::new();
        assert_eq!(registry.count(), 0);
        registry.register(Box::new(SpacyLikeAdapter::new()));
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_pipeline_builder_metadata() {
        let pipeline = PipelineBuilder::new("test")
            .metadata("version", "1.0")
            .add_stage(Box::new(TokenizerStage::new()))
            .build();
        assert_eq!(pipeline.metadata.get("version").unwrap(), "1.0");
    }

    #[test]
    fn test_pipeline_trace_structure() {
        let adapter = SpacyLikeAdapter::new();
        let trace = adapter.execute("Hello world").unwrap();
        assert_eq!(trace.input, "Hello world");
        assert!(trace.total_time_ms < 10000);
        for snapshot in &trace.per_stage_irs {
            assert!(!snapshot.stage_id.0.is_empty());
        }
    }
}
