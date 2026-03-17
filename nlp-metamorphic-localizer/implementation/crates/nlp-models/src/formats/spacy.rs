//! spaCy model metadata and pipeline configuration parsing.
//!
//! Provides types and loaders for the two JSON files typically found in a
//! spaCy model package directory:
//!
//! - `meta.json` → [`SpacyMeta`]
//! - `config.cfg` (as JSON) → [`SpacyPipelineConfig`]

use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};
use std::collections::HashMap;
use std::path::Path;

// ── Model metadata ──────────────────────────────────────────────────────────

/// Represents a spaCy model's `meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacyMeta {
    /// Human-readable model name (e.g. `"en_core_web_sm"`).
    #[serde(default)]
    pub name: Option<String>,

    /// Full model name including language prefix.
    #[serde(default)]
    pub lang: Option<String>,

    /// Model version string.
    #[serde(default)]
    pub version: Option<String>,

    /// spaCy version the model was trained with.
    #[serde(default)]
    pub spacy_version: Option<String>,

    /// Short description of the model.
    #[serde(default)]
    pub description: Option<String>,

    /// Author / publisher.
    #[serde(default)]
    pub author: Option<String>,

    /// License identifier (e.g. `"MIT"`).
    #[serde(default)]
    pub license: Option<String>,

    /// Ordered list of pipeline component names.
    #[serde(default)]
    pub pipeline: Vec<String>,

    /// Named-entity labels the model can predict.
    #[serde(default)]
    pub labels: HashMap<String, Vec<String>>,

    /// Accuracy / performance metrics from training.
    #[serde(default)]
    pub performance: HashMap<String, serde_json::Value>,

    /// Sizes of model vectors, if any.
    #[serde(default)]
    pub vectors: HashMap<String, serde_json::Value>,

    /// Catch-all for extra keys.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl SpacyMeta {
    /// True when the model includes a NER component.
    pub fn has_ner(&self) -> bool {
        self.pipeline.iter().any(|c| c == "ner")
    }

    /// True when the model includes a parser component.
    pub fn has_parser(&self) -> bool {
        self.pipeline.iter().any(|c| c == "parser")
    }

    /// True when the model includes a tagger component.
    pub fn has_tagger(&self) -> bool {
        self.pipeline.iter().any(|c| c == "tagger")
    }

    /// Return NER labels if available.
    pub fn ner_labels(&self) -> Option<&[String]> {
        self.labels.get("ner").map(|v| v.as_slice())
    }
}

// ── Pipeline config ─────────────────────────────────────────────────────────

/// Represents a spaCy pipeline configuration.
///
/// spaCy v3+ uses `config.cfg` (INI-like), but many tools export it as JSON.
/// This type captures the common sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacyPipelineConfig {
    /// NLP pipeline component settings keyed by component name.
    #[serde(default)]
    pub components: HashMap<String, SpacyComponentConfig>,

    /// Training hyper-parameters (optional).
    #[serde(default)]
    pub training: HashMap<String, serde_json::Value>,

    /// General NLP settings (tokenizer, etc).
    #[serde(default)]
    pub nlp: HashMap<String, serde_json::Value>,

    /// Catch-all.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Configuration for a single spaCy pipeline component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacyComponentConfig {
    /// Factory name that creates this component (e.g. `"ner"`, `"tagger"`).
    #[serde(default)]
    pub factory: Option<String>,

    /// Model architecture description.
    #[serde(default)]
    pub model: HashMap<String, serde_json::Value>,

    /// Catch-all.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl SpacyPipelineConfig {
    /// Ordered list of component names.
    pub fn component_names(&self) -> Vec<&str> {
        self.components.keys().map(|s| s.as_str()).collect()
    }
}

// ── Loaders ─────────────────────────────────────────────────────────────────

/// Load `meta.json` from a spaCy model directory.
pub fn load_spacy_meta(dir: impl AsRef<Path>) -> Result<SpacyMeta> {
    let path = dir.as_ref().join("meta.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to read spaCy meta: {e}"),
        )
    })?;
    serde_json::from_str(&text).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to parse spaCy meta: {e}"),
        )
    })
}

/// Load a spaCy pipeline config from a JSON file named `config.json` in the
/// given directory.
///
/// Note: spaCy v3 normally uses `config.cfg` (INI format).  This loader
/// expects a JSON export — use `spacy init fill-config --to-json` or similar
/// to produce one.
pub fn load_pipeline_config(dir: impl AsRef<Path>) -> Result<SpacyPipelineConfig> {
    let path = dir.as_ref().join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to read pipeline config: {e}"),
        )
    })?;
    serde_json::from_str(&text).map_err(|e| {
        LocalizerError::config(
            path.display().to_string(),
            format!("failed to parse pipeline config: {e}"),
        )
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_META: &str = r#"{
        "name": "en_core_web_sm",
        "lang": "en",
        "version": "3.7.0",
        "spacy_version": ">=3.7.0,<3.8.0",
        "description": "English pipeline optimized for CPU",
        "author": "Explosion",
        "license": "MIT",
        "pipeline": ["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"],
        "labels": {
            "ner": ["PERSON", "ORG", "GPE", "DATE", "MONEY"],
            "tagger": ["NN", "VB", "JJ"]
        },
        "performance": {
            "ents_f": 0.85,
            "tag_acc": 0.97
        }
    }"#;

    const SAMPLE_CONFIG: &str = r#"{
        "nlp": {
            "lang": "en",
            "pipeline": ["tagger", "parser", "ner"]
        },
        "components": {
            "tagger": {
                "factory": "tagger",
                "model": {"@architectures": "spacy.Tagger.v2"}
            },
            "parser": {
                "factory": "parser",
                "model": {"@architectures": "spacy.TransitionBasedParser.v2"}
            },
            "ner": {
                "factory": "ner",
                "model": {"@architectures": "spacy.TransitionBasedParser.v2"}
            }
        },
        "training": {
            "batcher": {"size": 1000}
        }
    }"#;

    #[test]
    fn test_parse_meta() {
        let meta: SpacyMeta = serde_json::from_str(SAMPLE_META).unwrap();
        assert_eq!(meta.name.as_deref(), Some("en_core_web_sm"));
        assert_eq!(meta.lang.as_deref(), Some("en"));
        assert_eq!(meta.version.as_deref(), Some("3.7.0"));
        assert_eq!(meta.pipeline.len(), 6);
        assert!(meta.has_ner());
        assert!(meta.has_parser());
        assert!(meta.has_tagger());
        assert_eq!(meta.ner_labels().unwrap().len(), 5);
    }

    #[test]
    fn test_parse_pipeline_config() {
        let config: SpacyPipelineConfig = serde_json::from_str(SAMPLE_CONFIG).unwrap();
        assert_eq!(config.components.len(), 3);
        assert!(config.components.contains_key("ner"));

        let ner = &config.components["ner"];
        assert_eq!(ner.factory.as_deref(), Some("ner"));
    }

    #[test]
    fn test_component_names() {
        let config: SpacyPipelineConfig = serde_json::from_str(SAMPLE_CONFIG).unwrap();
        let names = config.component_names();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn test_meta_no_ner() {
        let json = r#"{"pipeline": ["tagger", "parser"]}"#;
        let meta: SpacyMeta = serde_json::from_str(json).unwrap();
        assert!(!meta.has_ner());
        assert!(meta.has_parser());
        assert!(meta.ner_labels().is_none());
    }

    #[test]
    fn test_load_spacy_meta_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("meta.json"), SAMPLE_META).unwrap();
        let meta = load_spacy_meta(dir.path()).unwrap();
        assert_eq!(meta.name.as_deref(), Some("en_core_web_sm"));
    }

    #[test]
    fn test_load_pipeline_config_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), SAMPLE_CONFIG).unwrap();
        let config = load_pipeline_config(dir.path()).unwrap();
        assert_eq!(config.components.len(), 3);
    }

    #[test]
    fn test_load_missing_meta_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_spacy_meta(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_fields_preserved() {
        let json = r#"{"name": "test", "pipeline": [], "custom_key": true}"#;
        let meta: SpacyMeta = serde_json::from_str(json).unwrap();
        assert!(meta.extra.contains_key("custom_key"));
    }
}
