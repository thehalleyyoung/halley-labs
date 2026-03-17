//! NLP pipeline models for the metamorphic localizer.
//!
//! This crate provides concrete NLP pipeline stages (tokenizer, POS tagger,
//! dependency parser, NER), intermediate-representation utilities, distance
//! computers, and pipeline adapters that model real-world NLP frameworks.

pub mod distance;
pub mod formats;
pub mod ir;
pub mod ner_model;
pub mod parser_model;
pub mod pipeline;
pub mod stage;
pub mod tagger;
pub mod tokenizer;

// Re-export key public types for ergonomic downstream use.
pub use distance::{
    ClassificationDistanceComputer, DependencyDistanceComputer, EmbeddingDistanceComputer,
    EntityDistanceComputer, NormalizedDistanceComputer, PosTagDistanceComputer,
    StageDistanceFactory, TokenDistanceComputer,
};
pub use ir::{
    DependencyTreeIR, EntityAnnotationIR, IRAligner, IRDiff, IRSerializer, LemmaAlignment,
    TokenSequenceIR,
};
pub use ner_model::RuleBasedNER;
pub use parser_model::RuleBasedParser;
pub use pipeline::{
    AdapterRegistry, HuggingFaceLikeAdapter, Pipeline, PipelineAdapter, PipelineBuilder,
    PipelineTrace, SpacyLikeAdapter, StanzaLikeAdapter,
};
pub use stage::{
    DependencyParserStage, EmbedderStage, NERStage, PosTaggerStage, SentimentClassifierStage,
    StageMetadata, StageType, TokenizerStage,
};
pub use tagger::RuleBasedTagger;
pub use tokenizer::SimpleTokenizer;
