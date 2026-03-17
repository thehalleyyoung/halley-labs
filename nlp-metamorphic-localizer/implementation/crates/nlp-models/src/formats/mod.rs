//! File-format readers and writers for NLP data exchange.
//!
//! Provides I/O support for:
//! - **CoNLL-U** (`conll`): Universal Dependencies column format.
//! - **JSONL** (`jsonl`): Line-delimited JSON records.
//! - **HuggingFace** (`huggingface`): HuggingFace model/tokenizer configs.
//! - **spaCy** (`spacy`): spaCy model metadata and pipeline configs.

pub mod conll;
pub mod huggingface;
pub mod jsonl;
pub mod spacy;

pub use conll::{CoNLLReader, CoNLLWriter};
pub use huggingface::{
    HuggingFaceModelConfig, HuggingFaceTokenizerConfig, load_model_config,
    load_tokenizer_config,
};
pub use jsonl::{JsonlReader, JsonlRecord, JsonlWriter};
pub use spacy::{SpacyMeta, SpacyPipelineConfig, load_pipeline_config, load_spacy_meta};
