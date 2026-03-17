//! Pipeline stage implementations that wrap the model modules and implement
//! the `PipelineStage` trait from `shared_types`.

use shared_types::{
    DependencyRelation, EntityLabel, IRType, IntermediateRepresentation, LocalizerError,
    PipelineStage, PosTag, Result, Sentence, SentenceFeatures, StageId, Token,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ner_model::RuleBasedNER;
use crate::parser_model::RuleBasedParser;
use crate::tagger::RuleBasedTagger;
use crate::tokenizer::SimpleTokenizer;

// ── StageType ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageType {
    Tokenizer,
    PosTagger,
    DependencyParser,
    NER,
    SentimentClassifier,
    Encoder,
    Embedder,
    Custom(String),
}

impl std::fmt::Display for StageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageType::Custom(s) => write!(f, "Custom({})", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetadata {
    pub stage_type: StageType,
    pub input_ir_type: IRType,
    pub output_ir_type: IRType,
    pub description: String,
    pub version: String,
}

// ── TokenizerStage ──────────────────────────────────────────────────────────

pub struct TokenizerStage {
    id: StageId,
    tok: SimpleTokenizer,
}

impl TokenizerStage {
    pub fn new() -> Self {
        Self {
            id: StageId::new("tokenizer"),
            tok: SimpleTokenizer::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = StageId::new(&id.into());
        self
    }

    pub fn metadata(&self) -> StageMetadata {
        StageMetadata {
            stage_type: StageType::Tokenizer,
            input_ir_type: IRType::RawText,
            output_ir_type: IRType::Tokenized,
            description: "Whitespace/punctuation/contraction tokenizer".into(),
            version: "1.0.0".into(),
        }
    }
}

impl Default for TokenizerStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for TokenizerStage {
    fn id(&self) -> &StageId { &self.id }
    fn name(&self) -> &str { "tokenizer" }
    fn input_type(&self) -> IRType { IRType::RawText }
    fn output_type(&self) -> IRType { IRType::Tokenized }

    fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
        let text = &input.sentence.raw_text;
        let tokens = self.tok.tokenize(text);
        let mut sentence = input.sentence.clone();
        sentence.tokens = tokens;
        Ok(IntermediateRepresentation::new(IRType::Tokenized, sentence))
    }
}

// ── PosTaggerStage ──────────────────────────────────────────────────────────

pub struct PosTaggerStage {
    id: StageId,
    tagger: RuleBasedTagger,
}

impl PosTaggerStage {
    pub fn new() -> Self {
        Self {
            id: StageId::new("pos_tagger"),
            tagger: RuleBasedTagger::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = StageId::new(&id.into());
        self
    }

    pub fn metadata(&self) -> StageMetadata {
        StageMetadata {
            stage_type: StageType::PosTagger,
            input_ir_type: IRType::Tokenized,
            output_ir_type: IRType::PosTagged,
            description: "Rule-based POS tagger with lexicon and suffix rules".into(),
            version: "1.0.0".into(),
        }
    }
}

impl Default for PosTaggerStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for PosTaggerStage {
    fn id(&self) -> &StageId { &self.id }
    fn name(&self) -> &str { "pos_tagger" }
    fn input_type(&self) -> IRType { IRType::Tokenized }
    fn output_type(&self) -> IRType { IRType::PosTagged }

    fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
        let mut sentence = input.sentence.clone();
        if sentence.tokens.is_empty() {
            return Err(LocalizerError::pipeline("execution",
                "POS tagger received input with no tokens",
            ));
        }
        self.tagger.tag(&mut sentence.tokens);
        Ok(IntermediateRepresentation::new(IRType::PosTagged, sentence))
    }
}

// ── DependencyParserStage ───────────────────────────────────────────────────

pub struct DependencyParserStage {
    id: StageId,
    parser: RuleBasedParser,
}

impl DependencyParserStage {
    pub fn new() -> Self {
        Self {
            id: StageId::new("dep_parser"),
            parser: RuleBasedParser::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = StageId::new(&id.into());
        self
    }

    pub fn metadata(&self) -> StageMetadata {
        StageMetadata {
            stage_type: StageType::DependencyParser,
            input_ir_type: IRType::PosTagged,
            output_ir_type: IRType::Parsed,
            description: "Arc-eager-like rule-based dependency parser".into(),
            version: "1.0.0".into(),
        }
    }
}

impl Default for DependencyParserStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for DependencyParserStage {
    fn id(&self) -> &StageId { &self.id }
    fn name(&self) -> &str { "dep_parser" }
    fn input_type(&self) -> IRType { IRType::PosTagged }
    fn output_type(&self) -> IRType { IRType::Parsed }

    fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
        let mut sentence = input.sentence.clone();
        if sentence.tokens.is_empty() {
            return Err(LocalizerError::pipeline("execution",
                "Dependency parser received input with no tokens",
            ));
        }
        let edges = self.parser.parse(&sentence.tokens);
        sentence.dependency_edges = edges;
        Ok(IntermediateRepresentation::new(IRType::Parsed, sentence))
    }
}

// ── NERStage ────────────────────────────────────────────────────────────────

pub struct NERStage {
    id: StageId,
    ner: RuleBasedNER,
}

impl NERStage {
    pub fn new() -> Self {
        Self {
            id: StageId::new("ner"),
            ner: RuleBasedNER::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = StageId::new(&id.into());
        self
    }

    pub fn metadata(&self) -> StageMetadata {
        StageMetadata {
            stage_type: StageType::NER,
            input_ir_type: IRType::PosTagged,
            output_ir_type: IRType::EntityAnnotated,
            description: "Rule-based NER with gazetteers".into(),
            version: "1.0.0".into(),
        }
    }
}

impl Default for NERStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for NERStage {
    fn id(&self) -> &StageId { &self.id }
    fn name(&self) -> &str { "ner" }
    fn input_type(&self) -> IRType { IRType::PosTagged }
    fn output_type(&self) -> IRType { IRType::EntityAnnotated }

    fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
        let mut sentence = input.sentence.clone();
        let entities = self.ner.recognize(&sentence.tokens);
        sentence.entities = entities;
        Ok(IntermediateRepresentation::new(IRType::EntityAnnotated, sentence))
    }
}

// ── SentimentClassifierStage ────────────────────────────────────────────────

pub struct SentimentClassifierStage {
    id: StageId,
    positive_words: HashMap<String, f64>,
    negative_words: HashMap<String, f64>,
}

impl SentimentClassifierStage {
    pub fn new() -> Self {
        Self {
            id: StageId::new("sentiment"),
            positive_words: build_positive_lexicon(),
            negative_words: build_negative_lexicon(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = StageId::new(&id.into());
        self
    }

    fn score_tokens(&self, tokens: &[Token]) -> f64 {
        let mut total = 0.0;
        let mut count = 0;
        let mut negation = false;
        for tok in tokens {
            let lower = tok.text.to_lowercase();
            if matches!(lower.as_str(), "not" | "no" | "never" | "n't" | "neither" | "nor") {
                negation = true;
                continue;
            }
            let mut val = 0.0;
            if let Some(&v) = self.positive_words.get(&lower) {
                val = v;
            } else if let Some(&v) = self.negative_words.get(&lower) {
                val = -v;
            }
            if negation && val != 0.0 {
                val = -val;
                negation = false;
            }
            if val != 0.0 {
                total += val;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total / count as f64 }
    }
}

impl Default for SentimentClassifierStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for SentimentClassifierStage {
    fn id(&self) -> &StageId { &self.id }
    fn name(&self) -> &str { "sentiment_classifier" }
    fn input_type(&self) -> IRType { IRType::Tokenized }
    fn output_type(&self) -> IRType { IRType::SentimentScored }

    fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
        let mut sentence = input.sentence.clone();
        let score = self.score_tokens(&sentence.tokens);
        let label = if score > 0.1 {
            "positive"
        } else if score < -0.1 {
            "negative"
        } else {
            "neutral"
        };
        let features = SentenceFeatures {
            sentiment_score: Some(score),
            sentiment_label: Some(label.to_string()),
            ..SentenceFeatures::default()
        };
        sentence.features = Some(features);
        Ok(IntermediateRepresentation::new(IRType::SentimentScored, sentence))
    }
}

// ── EmbedderStage ───────────────────────────────────────────────────────────

pub struct EmbedderStage {
    id: StageId,
    vocab: HashMap<String, usize>,
    dim: usize,
}

impl EmbedderStage {
    pub fn new(dim: usize) -> Self {
        Self {
            id: StageId::new("embedder"),
            vocab: HashMap::new(),
            dim,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = StageId::new(&id.into());
        self
    }

    /// Produce a bag-of-words embedding vector.
    pub fn embed(&self, tokens: &[Token]) -> Vec<f64> {
        let mut vec = vec![0.0f64; self.dim];
        for tok in tokens {
            let lower = tok.text.to_lowercase();
            let hash = simple_hash(&lower) % self.dim;
            vec[hash] += 1.0;
        }
        // L2 normalise
        let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

impl Default for EmbedderStage {
    fn default() -> Self {
        Self::new(128)
    }
}

impl PipelineStage for EmbedderStage {
    fn id(&self) -> &StageId { &self.id }
    fn name(&self) -> &str { "embedder" }
    fn input_type(&self) -> IRType { IRType::Tokenized }
    fn output_type(&self) -> IRType { IRType::FeatureVector }

    fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
        let mut sentence = input.sentence.clone();
        let embedding = self.embed(&sentence.tokens);
        let features = sentence.features.get_or_insert_with(SentenceFeatures::default);
        features.extra.insert("embedding_dim".into(), self.dim.to_string());
        // Store embedding in data map
        let mut ir = IntermediateRepresentation::new(IRType::FeatureVector, sentence);
        ir.data.insert(
            "embedding".into(),
            serde_json::to_value(&embedding).unwrap_or_default(),
        );
        Ok(ir)
    }
}

fn simple_hash(s: &str) -> usize {
    let mut h: usize = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as usize);
    }
    h
}

// ── Sentiment lexicons ──────────────────────────────────────────────────────

fn build_positive_lexicon() -> HashMap<String, f64> {
    let words = [
        ("good", 0.7), ("great", 0.9), ("excellent", 1.0), ("wonderful", 0.9),
        ("fantastic", 0.95), ("amazing", 0.9), ("love", 0.8), ("happy", 0.8),
        ("best", 0.9), ("beautiful", 0.8), ("nice", 0.6), ("awesome", 0.85),
        ("brilliant", 0.9), ("perfect", 1.0), ("superb", 0.95),
        ("outstanding", 0.95), ("delightful", 0.85), ("pleasant", 0.6),
        ("enjoy", 0.7), ("enjoyed", 0.7), ("impressive", 0.8),
        ("magnificent", 0.9), ("marvelous", 0.85), ("terrific", 0.85),
        ("remarkable", 0.8), ("exceptional", 0.9), ("splendid", 0.85),
        ("joyful", 0.8), ("grateful", 0.7), ("exciting", 0.75),
        ("fabulous", 0.85), ("glorious", 0.8), ("cheerful", 0.7),
        ("positive", 0.6), ("like", 0.5), ("liked", 0.5),
        ("better", 0.6), ("fine", 0.4), ("well", 0.4),
        ("recommend", 0.7), ("recommended", 0.7),
    ];
    words.iter().map(|(w, v)| (w.to_string(), *v)).collect()
}

fn build_negative_lexicon() -> HashMap<String, f64> {
    let words = [
        ("bad", 0.7), ("terrible", 0.95), ("horrible", 0.95), ("awful", 0.9),
        ("worst", 1.0), ("hate", 0.9), ("ugly", 0.7), ("poor", 0.6),
        ("disappointing", 0.7), ("disappointed", 0.7), ("boring", 0.6),
        ("dull", 0.5), ("stupid", 0.8), ("annoying", 0.7), ("angry", 0.7),
        ("sad", 0.6), ("unhappy", 0.7), ("miserable", 0.85),
        ("dreadful", 0.9), ("pathetic", 0.8), ("lousy", 0.7),
        ("disgusting", 0.9), ("atrocious", 0.95), ("abysmal", 0.9),
        ("inferior", 0.6), ("mediocre", 0.5), ("nasty", 0.7),
        ("unpleasant", 0.6), ("painful", 0.6), ("frustrating", 0.7),
        ("irritating", 0.65), ("depressing", 0.7), ("wretched", 0.8),
        ("dislike", 0.6), ("negative", 0.5), ("worse", 0.7),
        ("fail", 0.7), ("failed", 0.7), ("failure", 0.75),
    ];
    words.iter().map(|(w, v)| (w.to_string(), *v)).collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_ir(text: &str) -> IntermediateRepresentation {
        IntermediateRepresentation::new(IRType::RawText, Sentence::from_text(text))
    }

    fn tokenized_ir(text: &str) -> IntermediateRepresentation {
        let tok_stage = TokenizerStage::new();
        tok_stage.process(&raw_ir(text)).unwrap()
    }

    fn tagged_ir(text: &str) -> IntermediateRepresentation {
        let tir = tokenized_ir(text);
        let pos_stage = PosTaggerStage::new();
        pos_stage.process(&tir).unwrap()
    }

    #[test]
    fn test_tokenizer_stage() {
        let stage = TokenizerStage::new();
        let ir = raw_ir("Hello world");
        let out = stage.process(&ir).unwrap();
        assert_eq!(out.ir_type, IRType::Tokenized);
        assert_eq!(out.sentence.tokens.len(), 2);
    }

    #[test]
    fn test_pos_tagger_stage() {
        let ir = tokenized_ir("The cat sat");
        let stage = PosTaggerStage::new();
        let out = stage.process(&ir).unwrap();
        assert_eq!(out.ir_type, IRType::PosTagged);
        assert!(out.sentence.tokens.iter().all(|t| t.pos_tag.is_some()));
    }

    #[test]
    fn test_dep_parser_stage() {
        let ir = tagged_ir("The cat sat");
        let stage = DependencyParserStage::new();
        let out = stage.process(&ir).unwrap();
        assert_eq!(out.ir_type, IRType::Parsed);
        assert!(!out.sentence.dependency_edges.is_empty());
    }

    #[test]
    fn test_ner_stage() {
        let ir = tagged_ir("John lives in London");
        let stage = NERStage::new();
        let out = stage.process(&ir).unwrap();
        assert_eq!(out.ir_type, IRType::EntityAnnotated);
        assert!(!out.sentence.entities.is_empty());
    }

    #[test]
    fn test_sentiment_positive() {
        let ir = tokenized_ir("This movie is great and wonderful");
        let stage = SentimentClassifierStage::new();
        let out = stage.process(&ir).unwrap();
        let features = out.sentence.features.unwrap();
        assert!(features.sentiment_score.unwrap() > 0.0);
        assert_eq!(features.sentiment_label.unwrap(), "positive");
    }

    #[test]
    fn test_sentiment_negative() {
        let ir = tokenized_ir("This movie is terrible and awful");
        let stage = SentimentClassifierStage::new();
        let out = stage.process(&ir).unwrap();
        let features = out.sentence.features.unwrap();
        assert!(features.sentiment_score.unwrap() < 0.0);
        assert_eq!(features.sentiment_label.unwrap(), "negative");
    }

    #[test]
    fn test_sentiment_negation() {
        let ir = tokenized_ir("This is not bad");
        let stage = SentimentClassifierStage::new();
        let out = stage.process(&ir).unwrap();
        let features = out.sentence.features.unwrap();
        // "not bad" → positive
        assert!(features.sentiment_score.unwrap() > 0.0);
    }

    #[test]
    fn test_embedder_stage() {
        let ir = tokenized_ir("The quick brown fox");
        let stage = EmbedderStage::new(64);
        let out = stage.process(&ir).unwrap();
        assert_eq!(out.ir_type, IRType::FeatureVector);
        assert!(out.data.contains_key("embedding"));
    }

    #[test]
    fn test_embedder_normalisation() {
        let stage = EmbedderStage::new(64);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("world", 1),
        ];
        let emb = stage.embed(&tokens);
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stage_metadata() {
        let m = TokenizerStage::new().metadata();
        assert_eq!(m.stage_type, StageType::Tokenizer);
        assert_eq!(m.input_ir_type, IRType::RawText);
    }

    #[test]
    fn test_pos_tagger_empty_tokens_error() {
        let stage = PosTaggerStage::new();
        let ir = IntermediateRepresentation::new(
            IRType::Tokenized,
            Sentence { raw_text: "".into(), tokens: vec![], entities: vec![], dependency_edges: vec![], parse_tree: None, features: None },
        );
        assert!(stage.process(&ir).is_err());
    }
}
