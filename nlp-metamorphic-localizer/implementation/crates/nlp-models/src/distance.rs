//! Type-specific distance computation for each IR representation.

use shared_types::{
    DependencyEdge, DependencyRelation, DistanceComputer, DistanceMetric, DistanceValue,
    EntityLabel, EntitySpan, IRType, IntermediateRepresentation, LocalizerError, PosTag,
    Result, Token,
};
use std::collections::{HashMap, HashSet};

// ── TokenDistanceComputer ───────────────────────────────────────────────────

/// Levenshtein-based distance on token sequences, normalised to [0, 1].
pub struct TokenDistanceComputer;

impl TokenDistanceComputer {
    pub fn new() -> Self {
        Self
    }

    fn token_texts(ir: &IntermediateRepresentation) -> Vec<String> {
        ir.sentence.tokens.iter().map(|t| t.text.to_lowercase()).collect()
    }

    fn levenshtein_distance(a: &[String], b: &[String]) -> usize {
        let n = a.len();
        let m = b.len();
        let mut dp = vec![vec![0usize; m + 1]; n + 1];
        for i in 0..=n { dp[i][0] = i; }
        for j in 0..=m { dp[0][j] = j; }
        for i in 1..=n {
            for j in 1..=m {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[n][m]
    }
}

impl Default for TokenDistanceComputer {
    fn default() -> Self { Self::new() }
}

impl DistanceComputer for TokenDistanceComputer {
    fn metric(&self) -> DistanceMetric {
        DistanceMetric::EditDistance
    }

    fn compute(&self, a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> Result<DistanceValue> {
        let ta = Self::token_texts(a);
        let tb = Self::token_texts(b);
        let dist = Self::levenshtein_distance(&ta, &tb);
        let max_len = ta.len().max(tb.len());
        let normalised = if max_len == 0 { 0.0 } else { dist as f64 / max_len as f64 };
        Ok(DistanceValue::normalized(normalised, DistanceMetric::EditDistance))
    }
}

// ── PosTagDistanceComputer ──────────────────────────────────────────────────

/// Weighted Hamming distance on POS-tag sequences.
pub struct PosTagDistanceComputer {
    weights: HashMap<PosTag, f64>,
}

impl PosTagDistanceComputer {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert(PosTag::Verb, 2.0);
        weights.insert(PosTag::Noun, 1.5);
        weights.insert(PosTag::Adj, 1.2);
        weights.insert(PosTag::Adv, 1.0);
        weights.insert(PosTag::Pron, 1.3);
        weights.insert(PosTag::Det, 0.5);
        weights.insert(PosTag::Prep, 0.8);
        weights.insert(PosTag::Conj, 0.6);
        weights.insert(PosTag::Punct, 0.2);
        weights.insert(PosTag::Num, 1.0);
        weights.insert(PosTag::Other, 0.3);
        weights.insert(PosTag::Intj, 0.5);
        weights.insert(PosTag::Other, 0.5);
        Self { weights }
    }

    fn weight(&self, tag: PosTag) -> f64 {
        *self.weights.get(&tag).unwrap_or(&1.0)
    }
}

impl Default for PosTagDistanceComputer {
    fn default() -> Self { Self::new() }
}

impl DistanceComputer for PosTagDistanceComputer {
    fn metric(&self) -> DistanceMetric {
        DistanceMetric::EditDistance
    }

    fn compute(&self, a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> Result<DistanceValue> {
        let tags_a: Vec<PosTag> = a.sentence.tokens.iter().filter_map(|t| t.pos_tag).collect();
        let tags_b: Vec<PosTag> = b.sentence.tokens.iter().filter_map(|t| t.pos_tag).collect();
        let min_len = tags_a.len().min(tags_b.len());
        let max_len = tags_a.len().max(tags_b.len());
        if max_len == 0 {
            return Ok(DistanceValue::new(0.0, DistanceMetric::EditDistance));
        }
        let mut weighted_diff = 0.0;
        let mut total_weight = 0.0;
        for i in 0..min_len {
            let w = self.weight(tags_a[i]).max(self.weight(tags_b[i]));
            total_weight += w;
            if tags_a[i] != tags_b[i] {
                weighted_diff += w;
            }
        }
        // Unmatched positions count as full-weight differences
        let extra = (max_len - min_len) as f64;
        weighted_diff += extra;
        total_weight += extra;

        let normalised = if total_weight == 0.0 { 0.0 } else { weighted_diff / total_weight };
        Ok(DistanceValue::normalized(normalised, DistanceMetric::EditDistance))
    }
}

// ── DependencyDistanceComputer ──────────────────────────────────────────────

/// Tree edit distance approximation: Jaccard over edge sets, plus label
/// agreement and unlabeled/labeled attachment scores.
pub struct DependencyDistanceComputer;

impl DependencyDistanceComputer {
    pub fn new() -> Self { Self }

    fn edge_set_unlabeled(edges: &[DependencyEdge]) -> HashSet<(usize, usize)> {
        edges.iter().map(|e| (e.head_index, e.dependent_index)).collect()
    }

    fn edge_set_labeled(edges: &[DependencyEdge]) -> HashSet<(usize, usize, String)> {
        edges.iter().map(|e| (e.head_index, e.dependent_index, format!("{:?}", e.relation))).collect()
    }

    /// Unlabeled attachment score.
    pub fn uas(a: &[DependencyEdge], b: &[DependencyEdge]) -> f64 {
        let set_a = Self::edge_set_unlabeled(a);
        let set_b = Self::edge_set_unlabeled(b);
        if set_a.is_empty() && set_b.is_empty() {
            return 1.0;
        }
        let intersection = set_a.intersection(&set_b).count();
        let union_size = set_a.len().max(set_b.len());
        if union_size == 0 { 1.0 } else { intersection as f64 / union_size as f64 }
    }

    /// Labeled attachment score.
    pub fn las(a: &[DependencyEdge], b: &[DependencyEdge]) -> f64 {
        let set_a = Self::edge_set_labeled(a);
        let set_b = Self::edge_set_labeled(b);
        if set_a.is_empty() && set_b.is_empty() {
            return 1.0;
        }
        let intersection = set_a.intersection(&set_b).count();
        let union_size = set_a.len().max(set_b.len());
        if union_size == 0 { 1.0 } else { intersection as f64 / union_size as f64 }
    }

    /// Jaccard distance over unlabeled edges.
    pub fn jaccard(a: &[DependencyEdge], b: &[DependencyEdge]) -> f64 {
        let set_a = Self::edge_set_unlabeled(a);
        let set_b = Self::edge_set_unlabeled(b);
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 { 0.0 } else { 1.0 - (intersection as f64 / union as f64) }
    }
}

impl Default for DependencyDistanceComputer {
    fn default() -> Self { Self::new() }
}

impl DistanceComputer for DependencyDistanceComputer {
    fn metric(&self) -> DistanceMetric {
        DistanceMetric::TreeEditDistance
    }

    fn compute(&self, a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> Result<DistanceValue> {
        let edges_a = &a.sentence.dependency_edges;
        let edges_b = &b.sentence.dependency_edges;
        let jaccard = Self::jaccard(edges_a, edges_b);
        let uas = Self::uas(edges_a, edges_b);
        let las = Self::las(edges_a, edges_b);
        // Combine: higher distance = more divergent
        let combined = (jaccard + (1.0 - uas) + (1.0 - las)) / 3.0;
        Ok(DistanceValue::normalized(combined, DistanceMetric::TreeEditDistance))
    }
}

// ── EntityDistanceComputer ──────────────────────────────────────────────────

/// Span-overlap F1, label agreement, and boundary distance.
pub struct EntityDistanceComputer;

impl EntityDistanceComputer {
    pub fn new() -> Self { Self }

    /// Compute span-level F1.
    pub fn span_f1(a: &[EntitySpan], b: &[EntitySpan]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let key = |e: &EntitySpan| (e.start, e.end, format!("{:?}", e.label));
        let set_a: HashSet<_> = a.iter().map(|e| key(e)).collect();
        let set_b: HashSet<_> = b.iter().map(|e| key(e)).collect();
        let tp = set_a.intersection(&set_b).count() as f64;
        let precision = if set_b.is_empty() { 0.0 } else { tp / set_b.len() as f64 };
        let recall = if set_a.is_empty() { 0.0 } else { tp / set_a.len() as f64 };
        if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) }
    }

    /// Label agreement among matched spans.
    pub fn label_agreement(a: &[EntitySpan], b: &[EntitySpan]) -> f64 {
        let mut matches = 0;
        let mut total = 0;
        for ea in a {
            for eb in b {
                if ea.start == eb.start && ea.end == eb.end {
                    total += 1;
                    if ea.label == eb.label {
                        matches += 1;
                    }
                }
            }
        }
        if total == 0 { 1.0 } else { matches as f64 / total as f64 }
    }

    /// Average boundary distance for overlapping entity pairs.
    pub fn boundary_distance(a: &[EntitySpan], b: &[EntitySpan]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let mut total_dist = 0.0;
        let mut count = 0;
        for ea in a {
            for eb in b {
                if ea.label == eb.label {
                    let d = ((ea.start as f64 - eb.start as f64).abs()
                        + (ea.end as f64 - eb.end as f64).abs())
                        / 2.0;
                    total_dist += d;
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { total_dist / count as f64 }
    }
}

impl Default for EntityDistanceComputer {
    fn default() -> Self { Self::new() }
}

impl DistanceComputer for EntityDistanceComputer {
    fn metric(&self) -> DistanceMetric {
        DistanceMetric::Jaccard
    }

    fn compute(&self, a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> Result<DistanceValue> {
        let ents_a = &a.sentence.entities;
        let ents_b = &b.sentence.entities;
        let f1 = Self::span_f1(ents_a, ents_b);
        let label_agr = Self::label_agreement(ents_a, ents_b);
        let boundary = Self::boundary_distance(ents_a, ents_b);
        let combined = 1.0 - (f1 * 0.5 + label_agr * 0.3 + (1.0 / (1.0 + boundary)) * 0.2);
        Ok(DistanceValue::normalized(combined.clamp(0.0, 1.0), DistanceMetric::Jaccard))
    }
}

// ── ClassificationDistanceComputer ──────────────────────────────────────────

/// Label disagreement and confidence divergence for classification outputs.
pub struct ClassificationDistanceComputer;

impl ClassificationDistanceComputer {
    pub fn new() -> Self { Self }
}

impl Default for ClassificationDistanceComputer {
    fn default() -> Self { Self::new() }
}

impl DistanceComputer for ClassificationDistanceComputer {
    fn metric(&self) -> DistanceMetric {
        DistanceMetric::Exact
    }

    fn compute(&self, a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> Result<DistanceValue> {
        let feat_a = a.sentence.features.as_ref();
        let feat_b = b.sentence.features.as_ref();
        let label_a = feat_a.and_then(|f| f.sentiment_label.as_deref()).unwrap_or("");
        let label_b = feat_b.and_then(|f| f.sentiment_label.as_deref()).unwrap_or("");
        let label_dist = if label_a == label_b { 0.0 } else { 1.0 };

        let conf_a = feat_a.and_then(|f| f.sentiment_score).unwrap_or(0.0);
        let conf_b = feat_b.and_then(|f| f.sentiment_score).unwrap_or(0.0);
        let conf_div = (conf_a - conf_b).abs();

        // Combine
        let combined = label_dist * 0.7 + conf_div.min(1.0) * 0.3;
        Ok(DistanceValue::normalized(combined.clamp(0.0, 1.0), DistanceMetric::Exact))
    }
}

// ── EmbeddingDistanceComputer ───────────────────────────────────────────────

/// Cosine distance between embedding vectors stored in IR metadata.
pub struct EmbeddingDistanceComputer;

impl EmbeddingDistanceComputer {
    pub fn new() -> Self { Self }

    fn extract_embedding(ir: &IntermediateRepresentation) -> Option<Vec<f64>> {
        ir.data
            .get("embedding")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 1.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        let similarity = dot / (norm_a * norm_b);
        1.0 - similarity.clamp(-1.0, 1.0)
    }
}

impl Default for EmbeddingDistanceComputer {
    fn default() -> Self { Self::new() }
}

impl DistanceComputer for EmbeddingDistanceComputer {
    fn metric(&self) -> DistanceMetric {
        DistanceMetric::Cosine
    }

    fn compute(&self, a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> Result<DistanceValue> {
        let emb_a = Self::extract_embedding(a)
            .ok_or_else(|| LocalizerError::pipeline("internal", "No embedding in IR a"))?;
        let emb_b = Self::extract_embedding(b)
            .ok_or_else(|| LocalizerError::pipeline("internal", "No embedding in IR b"))?;
        let dist = Self::cosine_distance(&emb_a, &emb_b);
        Ok(DistanceValue::normalized(dist.clamp(0.0, 1.0), DistanceMetric::Cosine))
    }
}

// ── StageDistanceFactory ────────────────────────────────────────────────────

/// Creates the appropriate `DistanceComputer` based on `IRType`.
pub struct StageDistanceFactory;

impl StageDistanceFactory {
    pub fn create(ir_type: &IRType) -> Box<dyn DistanceComputer> {
        match ir_type {
            IRType::RawText | IRType::Tokenized => Box::new(TokenDistanceComputer::new()),
            IRType::PosTagged => Box::new(PosTagDistanceComputer::new()),
            IRType::Parsed => Box::new(DependencyDistanceComputer::new()),
            IRType::EntityAnnotated => Box::new(EntityDistanceComputer::new()),
            IRType::SentimentScored => Box::new(ClassificationDistanceComputer::new()),
            IRType::FeatureVector => Box::new(EmbeddingDistanceComputer::new()),
            IRType::Custom(_) => Box::new(TokenDistanceComputer::new()),
            IRType::DependencyParsed => Box::new(DependencyDistanceComputer::new()),
            IRType::EntityRecognized => Box::new(EntityDistanceComputer::new()),
        }
    }
}

// ── NormalizedDistanceComputer ──────────────────────────────────────────────

/// Wraps any `DistanceComputer` and forces the result into [0, 1].
pub struct NormalizedDistanceComputer {
    inner: Box<dyn DistanceComputer>,
}

impl NormalizedDistanceComputer {
    pub fn new(inner: Box<dyn DistanceComputer>) -> Self {
        Self { inner }
    }
}

impl DistanceComputer for NormalizedDistanceComputer {
    fn metric(&self) -> DistanceMetric {
        self.inner.metric()
    }

    fn compute(&self, a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> Result<DistanceValue> {
        let mut dv = self.inner.compute(a, b)?;
        dv.value = dv.value.clamp(0.0, 1.0);
        dv.normalized = true;
        Ok(dv)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{Sentence, SentenceFeatures};

    fn ir_with_tokens(words: &[&str]) -> IntermediateRepresentation {
        let mut s = Sentence::from_text(&words.join(" "));
        for (i, tok) in s.tokens.iter_mut().enumerate() {
            tok.pos_tag = Some(PosTag::Noun);
        }
        IntermediateRepresentation::new(IRType::Tokenized, s)
    }

    fn ir_with_tags(pairs: &[(&str, PosTag)]) -> IntermediateRepresentation {
        let text = pairs.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" ");
        let mut s = Sentence::from_text(&text);
        for (i, (_, tag)) in pairs.iter().enumerate() {
            if i < s.tokens.len() {
                s.tokens[i].pos_tag = Some(*tag);
            }
        }
        IntermediateRepresentation::new(IRType::PosTagged, s)
    }

    #[test]
    fn test_token_distance_identical() {
        let a = ir_with_tokens(&["the", "cat", "sat"]);
        let b = ir_with_tokens(&["the", "cat", "sat"]);
        let d = TokenDistanceComputer::new().compute(&a, &b).unwrap();
        assert_eq!(d.value, 0.0);
        assert_eq!(d.value, 0.0);
    }

    #[test]
    fn test_token_distance_different() {
        let a = ir_with_tokens(&["the", "cat", "sat"]);
        let b = ir_with_tokens(&["a", "dog", "ran"]);
        let d = TokenDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value > 0.0);
    }

    #[test]
    fn test_pos_tag_distance_identical() {
        let a = ir_with_tags(&[("the", PosTag::Det), ("cat", PosTag::Noun)]);
        let b = ir_with_tags(&[("the", PosTag::Det), ("cat", PosTag::Noun)]);
        let d = PosTagDistanceComputer::new().compute(&a, &b).unwrap();
        assert_eq!(d.value, 0.0);
    }

    #[test]
    fn test_pos_tag_distance_different() {
        let a = ir_with_tags(&[("run", PosTag::Verb)]);
        let b = ir_with_tags(&[("run", PosTag::Noun)]);
        let d = PosTagDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value > 0.0);
    }

    #[test]
    fn test_dep_distance_identical() {
        let mut a = ir_with_tokens(&["cat", "sat"]);
        a.sentence.dependency_edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 2, relation: DependencyRelation::Root },
            DependencyEdge { head_index: 2, dependent_index: 1, relation: DependencyRelation::Nsubj },
        ];
        let b = a.clone();
        let d = DependencyDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value < 0.01);
    }

    #[test]
    fn test_dep_distance_different() {
        let mut a = ir_with_tokens(&["cat", "sat"]);
        a.sentence.dependency_edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 1, relation: DependencyRelation::Root },
        ];
        let mut b = ir_with_tokens(&["cat", "sat"]);
        b.sentence.dependency_edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 2, relation: DependencyRelation::Root },
        ];
        let d = DependencyDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value > 0.0);
    }

    #[test]
    fn test_entity_distance_identical() {
        let mut a = ir_with_tokens(&["John", "lives"]);
        a.sentence.entities = vec![EntitySpan {
            start: 0, end: 1, text: "John".into(), label: EntityLabel::Person, confidence: 0.9,
        }];
        let b = a.clone();
        let d = EntityDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value < 0.3);
    }

    #[test]
    fn test_entity_distance_different() {
        let mut a = ir_with_tokens(&["John", "lives"]);
        a.sentence.entities = vec![EntitySpan {
            start: 0, end: 1, text: "John".into(), label: EntityLabel::Person, confidence: 0.9,
        }];
        let b = ir_with_tokens(&["John", "lives"]);
        let d = EntityDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value > 0.0);
    }

    #[test]
    fn test_classification_distance_same_label() {
        let mut a = ir_with_tokens(&["great"]);
        a.sentence.features = Some(SentenceFeatures {
            sentiment_label: Some("positive".into()),
            sentiment_score: Some(0.8),
            ..SentenceFeatures::default()
        });
        let b = a.clone();
        let d = ClassificationDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value < 0.01);
    }

    #[test]
    fn test_classification_distance_different_label() {
        let mut a = ir_with_tokens(&["great"]);
        a.sentence.features = Some(SentenceFeatures {
            sentiment_label: Some("positive".into()),
            sentiment_score: Some(0.8),
            ..SentenceFeatures::default()
        });
        let mut b = ir_with_tokens(&["terrible"]);
        b.sentence.features = Some(SentenceFeatures {
            sentiment_label: Some("negative".into()),
            sentiment_score: Some(-0.8),
            ..SentenceFeatures::default()
        });
        let d = ClassificationDistanceComputer::new().compute(&a, &b).unwrap();
        assert!(d.value > 0.5);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = EmbeddingDistanceComputer::cosine_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        let d_val = EmbeddingDistanceComputer::cosine_distance(&a, &c);
        assert!(d_val < 1e-6);
    }

    #[test]
    fn test_factory_creates_correct_computer() {
        let c = StageDistanceFactory::create(&IRType::Tokenized);
        assert_eq!(c.metric(), DistanceMetric::EditDistance);

        let c2 = StageDistanceFactory::create(&IRType::Parsed);
        assert_eq!(c2.metric(), DistanceMetric::TreeEditDistance);
    }

    #[test]
    fn test_normalized_wrapper() {
        let inner = Box::new(TokenDistanceComputer::new());
        let wrapped = NormalizedDistanceComputer::new(inner);
        let a = ir_with_tokens(&["hello"]);
        let b = ir_with_tokens(&["world"]);
        let d = wrapped.compute(&a, &b).unwrap();
        assert!(d.value >= 0.0 && d.value <= 1.0);
    }
}
