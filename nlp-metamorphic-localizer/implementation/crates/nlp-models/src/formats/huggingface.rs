//! HuggingFace model and tokenizer configuration parsing.
//!
//! Provides types and loaders for the two JSON configuration files commonly
//! found in HuggingFace model directories:
//!
//! - `config.json` → [`HuggingFaceModelConfig`]
//! - `tokenizer_config.json` → [`HuggingFaceTokenizerConfig`]

use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};
use std::collections::HashMap;
use std::path::Path;

// ── Model types ─────────────────────────────────────────────────────────────

/// Well-known HuggingFace model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// BERT-family models.
    #[serde(rename = "bert")]
    Bert,
    /// RoBERTa-family models.
    #[serde(rename = "roberta")]
    Roberta,
    /// DistilBERT-family models.
    #[serde(rename = "distilbert")]
    DistilBert,
    /// Any model type not explicitly enumerated.
    #[serde(other)]
    Other,
}

// ── Model config ────────────────────────────────────────────────────────────

/// Represents a HuggingFace `config.json`.
///
/// Only the most commonly used fields are captured as typed members; everything
/// else is collected in [`extra`](HuggingFaceModelConfig::extra).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceModelConfig {
    /// Architecture identifier (e.g. `"bert"`, `"roberta"`).
    #[serde(default)]
    pub model_type: Option<ModelType>,

    /// Number of self-attention layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,

    /// Dimensionality of the hidden representations.
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of attention heads per layer.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,

    /// Size of the feed-forward intermediate layer.
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Maximum sequence length the model was trained with.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// Human-readable architecture list (e.g. `["BertForMaskedLM"]`).
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Mapping from string label ids to display names.
    #[serde(default)]
    pub id2label: HashMap<String, String>,

    /// Mapping from display names to integer label ids.
    #[serde(default)]
    pub label2id: HashMap<String, usize>,

    /// Catch-all for extra / model-specific keys.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl HuggingFaceModelConfig {
    /// The model type as a string, falling back to `"unknown"`.
    pub fn model_type_str(&self) -> &str {
        match &self.model_type {
            Some(ModelType::Bert) => "bert",
            Some(ModelType::Roberta) => "roberta",
            Some(ModelType::DistilBert) => "distilbert",
            Some(ModelType::Other) | None => "unknown",
        }
    }

    /// Number of classification labels, if `id2label` is populated.
    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }
}

// ── Tokenizer config ────────────────────────────────────────────────────────

/// Represents a HuggingFace `tokenizer_config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceTokenizerConfig {
    /// Tokenizer class name (e.g. `"BertTokenizerFast"`).
    #[serde(default)]
    pub tokenizer_class: Option<String>,

    /// Whether the tokenizer should lowercase input.
    #[serde(default)]
    pub do_lower_case: Option<bool>,

    /// Maximum input length.
    #[serde(default)]
    pub model_max_length: Option<usize>,

    /// Padding side (`"right"` or `"left"`).
    #[serde(default)]
    pub padding_side: Option<String>,

    /// Special tokens used by the tokenizer.
    #[serde(default)]
    pub special_tokens_map: HashMap<String, serde_json::Value>,

    /// Catch-all for extra keys.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// ── Loaders ─────────────────────────────────────────────────────────────────

/// Load a `config.json` from a HuggingFace model directory.
pub fn load_model_config(dir: impl AsRef<Path>) -> Result<HuggingFaceModelConfig> {
    let path = dir.as_ref().join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to read model config: {e}"),
        )
    })?;
    serde_json::from_str(&text).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to parse model config: {e}"),
        )
    })
}

/// Load a `tokenizer_config.json` from a HuggingFace model directory.
pub fn load_tokenizer_config(dir: impl AsRef<Path>) -> Result<HuggingFaceTokenizerConfig> {
    let path = dir.as_ref().join("tokenizer_config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to read tokenizer config: {e}"),
        )
    })?;
    serde_json::from_str(&text).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to parse tokenizer config: {e}"),
        )
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BERT_CONFIG: &str = r#"{
        "model_type": "bert",
        "architectures": ["BertForMaskedLM"],
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "vocab_size": 30522,
        "id2label": {"0": "NEGATIVE", "1": "POSITIVE"},
        "label2id": {"NEGATIVE": 0, "POSITIVE": 1}
    }"#;

    const ROBERTA_CONFIG: &str = r#"{
        "model_type": "roberta",
        "architectures": ["RobertaForSequenceClassification"],
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "vocab_size": 50265
    }"#;

    const DISTILBERT_CONFIG: &str = r#"{
        "model_type": "distilbert",
        "architectures": ["DistilBertForSequenceClassification"],
        "num_hidden_layers": 6,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "vocab_size": 30522
    }"#;

    const TOKENIZER_CONFIG: &str = r#"{
        "tokenizer_class": "BertTokenizerFast",
        "do_lower_case": true,
        "model_max_length": 512,
        "padding_side": "right"
    }"#;

    #[test]
    fn test_parse_bert_config() {
        let config: HuggingFaceModelConfig = serde_json::from_str(BERT_CONFIG).unwrap();
        assert_eq!(config.model_type, Some(ModelType::Bert));
        assert_eq!(config.model_type_str(), "bert");
        assert_eq!(config.num_hidden_layers, Some(12));
        assert_eq!(config.hidden_size, Some(768));
        assert_eq!(config.num_attention_heads, Some(12));
        assert_eq!(config.intermediate_size, Some(3072));
        assert_eq!(config.max_position_embeddings, Some(512));
        assert_eq!(config.vocab_size, Some(30522));
        assert_eq!(config.architectures, vec!["BertForMaskedLM"]);
        assert_eq!(config.num_labels(), 2);
    }

    #[test]
    fn test_parse_roberta_config() {
        let config: HuggingFaceModelConfig = serde_json::from_str(ROBERTA_CONFIG).unwrap();
        assert_eq!(config.model_type, Some(ModelType::Roberta));
        assert_eq!(config.model_type_str(), "roberta");
        assert_eq!(config.vocab_size, Some(50265));
    }

    #[test]
    fn test_parse_distilbert_config() {
        let config: HuggingFaceModelConfig = serde_json::from_str(DISTILBERT_CONFIG).unwrap();
        assert_eq!(config.model_type, Some(ModelType::DistilBert));
        assert_eq!(config.model_type_str(), "distilbert");
        assert_eq!(config.num_hidden_layers, Some(6));
    }

    #[test]
    fn test_parse_tokenizer_config() {
        let config: HuggingFaceTokenizerConfig = serde_json::from_str(TOKENIZER_CONFIG).unwrap();
        assert_eq!(
            config.tokenizer_class.as_deref(),
            Some("BertTokenizerFast")
        );
        assert_eq!(config.do_lower_case, Some(true));
        assert_eq!(config.model_max_length, Some(512));
        assert_eq!(config.padding_side.as_deref(), Some("right"));
    }

    #[test]
    fn test_unknown_model_type() {
        let json = r#"{"model_type": "gpt2", "hidden_size": 768}"#;
        let config: HuggingFaceModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, Some(ModelType::Other));
        assert_eq!(config.model_type_str(), "unknown");
    }

    #[test]
    fn test_extra_fields_preserved() {
        let json = r#"{"model_type": "bert", "hidden_size": 768, "custom_key": 42}"#;
        let config: HuggingFaceModelConfig = serde_json::from_str(json).unwrap();
        assert!(config.extra.contains_key("custom_key"));
    }

    #[test]
    fn test_load_model_config_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), BERT_CONFIG).unwrap();
        let config = load_model_config(dir.path()).unwrap();
        assert_eq!(config.model_type, Some(ModelType::Bert));
    }

    #[test]
    fn test_load_tokenizer_config_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), TOKENIZER_CONFIG).unwrap();
        let config = load_tokenizer_config(dir.path()).unwrap();
        assert_eq!(
            config.tokenizer_class.as_deref(),
            Some("BertTokenizerFast")
        );
    }

    #[test]
    fn test_load_missing_config_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_model_config(dir.path());
        assert!(result.is_err());
    }
}
