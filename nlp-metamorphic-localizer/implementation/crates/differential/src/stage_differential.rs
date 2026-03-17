//! Per-stage differential computation.
//!
//! Computes Δ_k(x, τ) = d_k(s_k(prefix_k(x)), s_k(prefix_k(τ(x)))) for each pipeline stage k.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use shared_types::{
    DependencyEdge, DistanceMetric, DistanceValue, EntitySpan, IRType,
    IntermediateRepresentation, LocalizerError, PosTag, Result, Sentence, StageId, Token,
};
use std::collections::{HashMap, HashSet};

// ── StageDifferential ───────────────────────────────────────────────────────

/// The measured differential for a single stage on a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDifferential {
    pub stage_id: StageId,
    pub stage_index: usize,
    pub delta_value: f64,
    pub metric_used: DistanceMetric,
    pub normalized: bool,
    pub raw_components: Vec<(String, f64)>,
}

impl StageDifferential {
    pub fn new(stage_id: StageId, stage_index: usize, delta_value: f64, metric: DistanceMetric) -> Self {
        Self {
            stage_id,
            stage_index,
            delta_value,
            metric_used: metric,
            normalized: false,
            raw_components: Vec::new(),
        }
    }

    pub fn with_components(mut self, components: Vec<(String, f64)>) -> Self {
        self.raw_components = components;
        self
    }

    pub fn mark_normalized(mut self) -> Self {
        self.normalized = true;
        self
    }
}

// ── DistanceConfig ──────────────────────────────────────────────────────────

/// Per-IR-type distance configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceConfig {
    pub metric: DistanceMetric,
    pub weight: f64,
}

impl Default for DistanceConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::EditDistance,
            weight: 1.0,
        }
    }
}

// ── NormalizationParams ─────────────────────────────────────────────────────

/// Per-stage calibration baseline for z-score normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub stage_id: StageId,
    pub mean: f64,
    pub std_dev: f64,
    pub sample_count: usize,
}

impl NormalizationParams {
    pub fn new(stage_id: StageId, mean: f64, std_dev: f64, sample_count: usize) -> Self {
        Self {
            stage_id,
            mean,
            std_dev,
            sample_count,
        }
    }

    /// Z-score normalize a raw differential value.
    pub fn normalize(&self, raw: f64) -> f64 {
        if self.std_dev < 1e-12 {
            raw - self.mean
        } else {
            (raw - self.mean) / self.std_dev
        }
    }
}

// ── DifferentialComputer ────────────────────────────────────────────────────

/// Main engine for computing per-stage differentials.
pub struct DifferentialComputer {
    pub distance_configs: HashMap<String, DistanceConfig>,
    pub normalization_params: HashMap<StageId, NormalizationParams>,
}

impl DifferentialComputer {
    pub fn new() -> Self {
        let mut distance_configs = HashMap::new();
        distance_configs.insert(
            "TokenSequence".to_string(),
            DistanceConfig {
                metric: DistanceMetric::EditDistance,
                weight: 1.0,
            },
        );
        distance_configs.insert(
            "PosTagged".to_string(),
            DistanceConfig {
                metric: DistanceMetric::Exact,
                weight: 1.0,
            },
        );
        distance_configs.insert(
            "Parsed".to_string(),
            DistanceConfig {
                metric: DistanceMetric::TreeEditDistance,
                weight: 1.0,
            },
        );
        distance_configs.insert(
            "EntityAnnotated".to_string(),
            DistanceConfig {
                metric: DistanceMetric::Jaccard,
                weight: 1.0,
            },
        );
        distance_configs.insert(
            "SentimentScored".to_string(),
            DistanceConfig {
                metric: DistanceMetric::Custom("custom".into()),
                weight: 1.0,
            },
        );
        Self {
            distance_configs,
            normalization_params: HashMap::new(),
        }
    }

    pub fn with_normalization(mut self, params: Vec<NormalizationParams>) -> Self {
        for p in params {
            self.normalization_params.insert(p.stage_id.clone(), p);
        }
        self
    }

    /// Compute the differential between original and transformed IRs at one stage.
    pub fn compute_stage_differential(
        &self,
        stage_id: &StageId,
        stage_index: usize,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> Result<StageDifferential> {
        let ir_type_key = ir_type_key(&original.ir_type);
        let config = self
            .distance_configs
            .get(&ir_type_key)
            .cloned()
            .unwrap_or_default();

        let (delta, components) =
            compute_type_specific_distance(&original.ir_type, original, transformed)?;

        let weighted_delta = delta * config.weight;

        let mut diff = StageDifferential::new(
            stage_id.clone(),
            stage_index,
            weighted_delta,
            config.metric,
        )
        .with_components(components);

        if let Some(params) = self.normalization_params.get(stage_id) {
            diff.delta_value = params.normalize(diff.delta_value);
            diff.normalized = true;
        }

        Ok(diff)
    }

    /// Normalize a precomputed differential using calibration baselines.
    pub fn normalize_differential(&self, diff: &mut StageDifferential) {
        if let Some(params) = self.normalization_params.get(&diff.stage_id) {
            diff.delta_value = params.normalize(diff.delta_value);
            diff.normalized = true;
        }
    }

    /// Compute differentials for a batch of test cases at a single stage.
    pub fn batch_compute(
        &self,
        stage_id: &StageId,
        stage_index: usize,
        pairs: &[(IntermediateRepresentation, IntermediateRepresentation)],
    ) -> Result<Vec<StageDifferential>> {
        let mut results = Vec::with_capacity(pairs.len());
        for (original, transformed) in pairs {
            results.push(self.compute_stage_differential(
                stage_id,
                stage_index,
                original,
                transformed,
            )?);
        }
        Ok(results)
    }
}

impl Default for DifferentialComputer {
    fn default() -> Self {
        Self::new()
    }
}

// ── DifferentialTimeSeries ──────────────────────────────────────────────────

/// Tracks how differentials evolve across pipeline stages for a single test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialTimeSeries {
    pub test_id: String,
    pub stage_diffs: Vec<StageDifferential>,
}

impl DifferentialTimeSeries {
    pub fn new(test_id: impl Into<String>) -> Self {
        Self {
            test_id: test_id.into(),
            stage_diffs: Vec::new(),
        }
    }

    pub fn push(&mut self, diff: StageDifferential) {
        self.stage_diffs.push(diff);
    }

    pub fn len(&self) -> usize {
        self.stage_diffs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stage_diffs.is_empty()
    }

    /// The stage index where the largest jump in delta occurs.
    pub fn max_jump_stage(&self) -> Option<usize> {
        if self.stage_diffs.len() < 2 {
            return self.stage_diffs.first().map(|d| d.stage_index);
        }
        let mut max_jump = f64::NEG_INFINITY;
        let mut max_idx = 0;
        for i in 1..self.stage_diffs.len() {
            let jump = self.stage_diffs[i].delta_value - self.stage_diffs[i - 1].delta_value;
            if jump > max_jump {
                max_jump = jump;
                max_idx = self.stage_diffs[i].stage_index;
            }
        }
        Some(max_idx)
    }

    /// Cumulative delta values across stages.
    pub fn cumulative_deltas(&self) -> Vec<f64> {
        let mut cum = Vec::with_capacity(self.stage_diffs.len());
        let mut total = 0.0;
        for d in &self.stage_diffs {
            total += d.delta_value;
            cum.push(total);
        }
        cum
    }

    /// Behavioral Fragility Index per stage: ratio of Δ_{k} to Δ_{k-1}.
    pub fn fragility_indices(&self) -> Vec<(usize, f64)> {
        let mut indices = Vec::new();
        for i in 0..self.stage_diffs.len() {
            let prev = if i == 0 {
                1.0 // sentinel: first stage amplification is its own delta
            } else {
                let p = self.stage_diffs[i - 1].delta_value;
                if p.abs() < 1e-12 {
                    1.0
                } else {
                    p
                }
            };
            let bfi = self.stage_diffs[i].delta_value / prev;
            indices.push((self.stage_diffs[i].stage_index, bfi));
        }
        indices
    }
}

// ── Type-specific distance functions ────────────────────────────────────────

fn ir_type_key(ir_type: &IRType) -> String {
    match ir_type {
        IRType::RawText => "RawText".into(),
        IRType::Tokenized => "TokenSequence".into(),
        IRType::PosTagged => "PosTagged".into(),
        IRType::Parsed => "Parsed".into(),
        IRType::EntityAnnotated => "EntityAnnotated".into(),
        IRType::FeatureVector => "FeatureVector".into(),
        IRType::SentimentScored => "SentimentScored".into(),
        IRType::Custom(s) => s.clone(),
        _ => "Unknown".into(),
    }
}

fn compute_type_specific_distance(
    ir_type: &IRType,
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
) -> Result<(f64, Vec<(String, f64)>)> {
    match ir_type {
        IRType::RawText => compute_raw_text_distance(original, transformed),
        IRType::Tokenized => compute_token_edit_distance(original, transformed),
        IRType::PosTagged => compute_pos_hamming_distance(original, transformed),
        IRType::Parsed => compute_tree_edit_distance(original, transformed),
        IRType::EntityAnnotated => compute_entity_span_f1(original, transformed),
        IRType::SentimentScored | IRType::FeatureVector => {
            compute_classification_distance(original, transformed)
        }
        IRType::Custom(_) => compute_token_edit_distance(original, transformed),
        _ => compute_token_edit_distance(original, transformed),
    }
}

/// Raw text distance using character-level edit distance.
fn compute_raw_text_distance(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
) -> Result<(f64, Vec<(String, f64)>)> {
    let a = &original.sentence.raw_text;
    let b = &transformed.sentence.raw_text;
    let dist = levenshtein_distance(a.as_bytes(), b.as_bytes());
    let max_len = a.len().max(b.len()).max(1);
    let normalized = dist as f64 / max_len as f64;
    Ok((normalized, vec![("char_edit_distance".into(), dist as f64)]))
}

/// Token-level edit distance for TokenSequence IRs.
fn compute_token_edit_distance(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
) -> Result<(f64, Vec<(String, f64)>)> {
    let orig_tokens: Vec<&str> = original.sentence.tokens.iter().map(|t| t.text.as_str()).collect();
    let trans_tokens: Vec<&str> = transformed.sentence.tokens.iter().map(|t| t.text.as_str()).collect();

    let dist = levenshtein_distance_generic(&orig_tokens, &trans_tokens);
    let max_len = orig_tokens.len().max(trans_tokens.len()).max(1);
    let normalized = dist as f64 / max_len as f64;

    let jaccard = token_jaccard(&orig_tokens, &trans_tokens);

    Ok((
        normalized,
        vec![
            ("token_edit_distance".into(), dist as f64),
            ("jaccard_distance".into(), 1.0 - jaccard),
        ],
    ))
}

/// Hamming-style distance for POS-tagged IRs (fraction of tag mismatches).
fn compute_pos_hamming_distance(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
) -> Result<(f64, Vec<(String, f64)>)> {
    let orig_tags: Vec<Option<PosTag>> = original.sentence.tokens.iter().map(|t| t.pos_tag).collect();
    let trans_tags: Vec<Option<PosTag>> = transformed.sentence.tokens.iter().map(|t| t.pos_tag).collect();

    let max_len = orig_tags.len().max(trans_tags.len());
    if max_len == 0 {
        return Ok((0.0, vec![("hamming_distance".into(), 0.0)]));
    }

    let mut mismatches = 0usize;
    for i in 0..max_len {
        let a = orig_tags.get(i).copied().flatten();
        let b = trans_tags.get(i).copied().flatten();
        if a != b {
            mismatches += 1;
        }
    }

    let normalized = mismatches as f64 / max_len as f64;
    Ok((
        normalized,
        vec![
            ("hamming_mismatches".into(), mismatches as f64),
            ("sequence_length".into(), max_len as f64),
        ],
    ))
}

/// Tree edit distance approximation for parsed IRs.
/// Uses a fast heuristic: compares dependency edge sets plus structural depth.
fn compute_tree_edit_distance(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
) -> Result<(f64, Vec<(String, f64)>)> {
    let orig_edges = &original.sentence.dependency_edges;
    let trans_edges = &transformed.sentence.dependency_edges;

    let orig_set: HashSet<String> = orig_edges
        .iter()
        .map(|e| format!("{}->{}:{:?}", e.head_index, e.dependent_index, e.relation))
        .collect();
    let trans_set: HashSet<String> = trans_edges
        .iter()
        .map(|e| format!("{}->{}:{:?}", e.head_index, e.dependent_index, e.relation))
        .collect();

    let union_size = orig_set.union(&trans_set).count().max(1);
    let intersection_size = orig_set.intersection(&trans_set).count();
    let edge_jaccard = intersection_size as f64 / union_size as f64;
    let edge_distance = 1.0 - edge_jaccard;

    // Structural depth difference
    let orig_depth = compute_tree_depth(&original.sentence);
    let trans_depth = compute_tree_depth(&transformed.sentence);
    let depth_diff = (orig_depth as f64 - trans_depth as f64).abs();
    let max_depth = orig_depth.max(trans_depth).max(1);
    let depth_distance = depth_diff / max_depth as f64;

    // Weighted combination
    let combined = 0.7 * edge_distance + 0.3 * depth_distance;

    Ok((
        combined,
        vec![
            ("edge_jaccard_distance".into(), edge_distance),
            ("depth_distance".into(), depth_distance),
            ("orig_edge_count".into(), orig_edges.len() as f64),
            ("trans_edge_count".into(), trans_edges.len() as f64),
        ],
    ))
}

/// Span-F1 distance for entity-annotated IRs.
fn compute_entity_span_f1(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
) -> Result<(f64, Vec<(String, f64)>)> {
    let orig_spans = &original.sentence.entities;
    let trans_spans = &transformed.sentence.entities;

    if orig_spans.is_empty() && trans_spans.is_empty() {
        return Ok((
            0.0,
            vec![
                ("precision".into(), 1.0),
                ("recall".into(), 1.0),
                ("f1".into(), 1.0),
            ],
        ));
    }
    if orig_spans.is_empty() || trans_spans.is_empty() {
        return Ok((
            1.0,
            vec![
                ("precision".into(), 0.0),
                ("recall".into(), 0.0),
                ("f1".into(), 0.0),
            ],
        ));
    }

    // Exact span+label match
    let orig_set: HashSet<(usize, usize, String)> = orig_spans
        .iter()
        .map(|e| (e.start, e.end, format!("{:?}", e.label)))
        .collect();
    let trans_set: HashSet<(usize, usize, String)> = trans_spans
        .iter()
        .map(|e| (e.start, e.end, format!("{:?}", e.label)))
        .collect();

    let tp = orig_set.intersection(&trans_set).count() as f64;
    let precision = if trans_set.is_empty() {
        0.0
    } else {
        tp / trans_set.len() as f64
    };
    let recall = if orig_set.is_empty() {
        0.0
    } else {
        tp / orig_set.len() as f64
    };
    let f1 = if precision + recall < 1e-12 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok((
        1.0 - f1,
        vec![
            ("precision".into(), precision),
            ("recall".into(), recall),
            ("f1".into(), f1),
        ],
    ))
}

/// Classification distance: label disagreement + confidence gap.
fn compute_classification_distance(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
) -> Result<(f64, Vec<(String, f64)>)> {
    let orig_label = original
        .sentence
        .features
        .as_ref()
        .and_then(|f| f.sentiment_label.as_deref())
        .unwrap_or("none");
    let trans_label = transformed
        .sentence
        .features
        .as_ref()
        .and_then(|f| f.sentiment_label.as_deref())
        .unwrap_or("none");

    let label_disagree = if orig_label != trans_label { 1.0 } else { 0.0 };

    let orig_conf = original.confidence.unwrap_or(0.5);
    let trans_conf = transformed.confidence.unwrap_or(0.5);
    let confidence_gap = (orig_conf - trans_conf).abs();

    let combined = 0.6 * label_disagree + 0.4 * confidence_gap;

    Ok((
        combined,
        vec![
            ("label_disagreement".into(), label_disagree),
            ("confidence_gap".into(), confidence_gap),
        ],
    ))
}

// ── Helper functions ────────────────────────────────────────────────────────

fn levenshtein_distance(a: &[u8], b: &[u8]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

fn levenshtein_distance_generic<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

fn token_jaccard(a: &[&str], b: &[&str]) -> f64 {
    let set_a: HashSet<&str> = a.iter().copied().collect();
    let set_b: HashSet<&str> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn compute_tree_depth(sentence: &Sentence) -> usize {
    if sentence.dependency_edges.is_empty() {
        return 0;
    }
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut root = 0usize;
    for edge in &sentence.dependency_edges {
        if edge.relation == shared_types::DependencyRelation::Root {
            root = edge.dependent_index;
        }
        children.entry(edge.head_index).or_default().push(edge.dependent_index);
    }
    fn depth_helper(node: usize, children: &HashMap<usize, Vec<usize>>, visited: &mut HashSet<usize>) -> usize {
        if !visited.insert(node) {
            return 0;
        }
        match children.get(&node) {
            Some(kids) => {
                1 + kids
                    .iter()
                    .map(|&k| depth_helper(k, children, visited))
                    .max()
                    .unwrap_or(0)
            }
            None => 1,
        }
    }
    let mut visited = HashSet::new();
    depth_helper(root, &children, &mut visited)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyRelation, EntityLabel, SentenceFeatures, Token};

    fn make_token(text: &str, idx: usize) -> Token {
        Token::new(text, idx, idx * 5, idx * 5 + text.len())
    }

    fn make_sentence(words: &[&str]) -> Sentence {
        let tokens: Vec<Token> = words.iter().enumerate().map(|(i, w)| make_token(w, i)).collect();
        let text = words.join(" ");
        Sentence {
            text,
            tokens,
            entities: Vec::new(),
            dependencies: Vec::new(),
            parse_tree: None,
            features: None,
        }
    }

    fn make_ir(ir_type: IRType, sentence: Sentence) -> IntermediateRepresentation {
        IntermediateRepresentation::new(ir_type, sentence)
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance(b"hello", b"hello"), 0);
    }

    #[test]
    fn test_levenshtein_different() {
        assert_eq!(levenshtein_distance(b"kitten", b"sitting"), 3);
    }

    #[test]
    fn test_token_edit_distance() {
        let orig = make_ir(IRType::Tokenized, make_sentence(&["the", "cat", "sat"]));
        let trans = make_ir(IRType::Tokenized, make_sentence(&["the", "dog", "sat"]));
        let (dist, _) = compute_token_edit_distance(&orig, &trans).unwrap();
        assert!(dist > 0.0 && dist < 1.0);
    }

    #[test]
    fn test_pos_hamming_identical() {
        let mut s1 = make_sentence(&["the", "cat"]);
        s1.tokens[0].pos_tag = Some(PosTag::Det);
        s1.tokens[1].pos_tag = Some(PosTag::Noun);
        let mut s2 = make_sentence(&["the", "cat"]);
        s2.tokens[0].pos_tag = Some(PosTag::Det);
        s2.tokens[1].pos_tag = Some(PosTag::Noun);

        let (dist, _) = compute_pos_hamming_distance(
            &make_ir(IRType::PosTagged, s1),
            &make_ir(IRType::PosTagged, s2),
        )
        .unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_pos_hamming_different() {
        let mut s1 = make_sentence(&["the", "cat"]);
        s1.tokens[0].pos_tag = Some(PosTag::Det);
        s1.tokens[1].pos_tag = Some(PosTag::Noun);
        let mut s2 = make_sentence(&["the", "cat"]);
        s2.tokens[0].pos_tag = Some(PosTag::Det);
        s2.tokens[1].pos_tag = Some(PosTag::Verb);

        let (dist, _) = compute_pos_hamming_distance(
            &make_ir(IRType::PosTagged, s1),
            &make_ir(IRType::PosTagged, s2),
        )
        .unwrap();
        assert!((dist - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_entity_span_f1_identical() {
        let mut s1 = make_sentence(&["John", "Smith"]);
        s1.entities.push(EntitySpan {
            text: "John Smith".into(),
            label: EntityLabel::Person,
            start: 0,
            end: 2,
            confidence: 0.9,
        });
        let s2 = s1.clone();
        let (dist, comps) = compute_entity_span_f1(
            &make_ir(IRType::EntityAnnotated, s1),
            &make_ir(IRType::EntityAnnotated, s2),
        )
        .unwrap();
        assert_eq!(dist, 0.0);
        let f1 = comps.iter().find(|(n, _)| n == "f1").unwrap().1;
        assert_eq!(f1, 1.0);
    }

    #[test]
    fn test_entity_span_f1_disjoint() {
        let mut s1 = make_sentence(&["John", "Smith"]);
        s1.entities.push(EntitySpan {
            text: "John".into(),
            label: EntityLabel::Person,
            start: 0,
            end: 1,
            confidence: 0.9,
        });
        let mut s2 = make_sentence(&["John", "Smith"]);
        s2.entities.push(EntitySpan {
            text: "Smith".into(),
            label: EntityLabel::Person,
            start: 1,
            end: 2,
            confidence: 0.9,
        });
        let (dist, _) = compute_entity_span_f1(
            &make_ir(IRType::EntityAnnotated, s1),
            &make_ir(IRType::EntityAnnotated, s2),
        )
        .unwrap();
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_classification_distance_same_label() {
        let mut s1 = make_sentence(&["great"]);
        s1.features = Some(SentenceFeatures {
            sentiment_label: Some("positive".into()),
            ..SentenceFeatures::default()
        });
        let mut s2 = make_sentence(&["great"]);
        s2.features = Some(SentenceFeatures {
            sentiment_label: Some("positive".into()),
            ..SentenceFeatures::default()
        });
        let ir1 = make_ir(IRType::SentimentScored, s1).with_confidence(0.9);
        let ir2 = make_ir(IRType::SentimentScored, s2).with_confidence(0.85);
        let (dist, _) = compute_classification_distance(&ir1, &ir2).unwrap();
        assert!(dist < 0.05);
    }

    #[test]
    fn test_classification_distance_different_label() {
        let mut s1 = make_sentence(&["great"]);
        s1.features = Some(SentenceFeatures {
            sentiment_label: Some("positive".into()),
            ..SentenceFeatures::default()
        });
        let mut s2 = make_sentence(&["terrible"]);
        s2.features = Some(SentenceFeatures {
            sentiment_label: Some("negative".into()),
            ..SentenceFeatures::default()
        });
        let ir1 = make_ir(IRType::SentimentScored, s1).with_confidence(0.9);
        let ir2 = make_ir(IRType::SentimentScored, s2).with_confidence(0.1);
        let (dist, _) = compute_classification_distance(&ir1, &ir2).unwrap();
        assert!(dist > 0.5);
    }

    #[test]
    fn test_differential_computer_roundtrip() {
        let computer = DifferentialComputer::new();
        let orig = make_ir(IRType::Tokenized, make_sentence(&["hello", "world"]));
        let trans = make_ir(IRType::Tokenized, make_sentence(&["hello", "earth"]));
        let diff = computer
            .compute_stage_differential(&StageId::new("tok"), 0, &orig, &trans)
            .unwrap();
        assert!(diff.delta_value > 0.0);
        assert_eq!(diff.stage_index, 0);
    }

    #[test]
    fn test_normalization_params() {
        let params = NormalizationParams::new(StageId::new("tok"), 0.5, 0.1, 100);
        let normalized = params.normalize(0.7);
        assert!((normalized - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_differential_time_series_max_jump() {
        let mut ts = DifferentialTimeSeries::new("test1");
        ts.push(StageDifferential::new(
            StageId::new("s0"),
            0,
            0.1,
            DistanceMetric::EditDistance,
        ));
        ts.push(StageDifferential::new(
            StageId::new("s1"),
            1,
            0.5,
            DistanceMetric::EditDistance,
        ));
        ts.push(StageDifferential::new(
            StageId::new("s2"),
            2,
            0.6,
            DistanceMetric::EditDistance,
        ));
        assert_eq!(ts.max_jump_stage(), Some(1));
    }

    #[test]
    fn test_differential_time_series_fragility() {
        let mut ts = DifferentialTimeSeries::new("test1");
        ts.push(StageDifferential::new(
            StageId::new("s0"),
            0,
            0.2,
            DistanceMetric::EditDistance,
        ));
        ts.push(StageDifferential::new(
            StageId::new("s1"),
            1,
            0.6,
            DistanceMetric::EditDistance,
        ));
        let fragilities = ts.fragility_indices();
        assert_eq!(fragilities.len(), 2);
        assert!((fragilities[1].1 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_batch_compute() {
        let computer = DifferentialComputer::new();
        let pairs = vec![
            (
                make_ir(IRType::Tokenized, make_sentence(&["a", "b"])),
                make_ir(IRType::Tokenized, make_sentence(&["a", "c"])),
            ),
            (
                make_ir(IRType::Tokenized, make_sentence(&["x"])),
                make_ir(IRType::Tokenized, make_sentence(&["y"])),
            ),
        ];
        let results = computer.batch_compute(&StageId::new("s"), 0, &pairs).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.delta_value > 0.0);
        }
    }

    #[test]
    fn test_tree_edit_distance_identical() {
        let mut s = make_sentence(&["the", "cat", "sat"]);
        s.dependency_edges.push(DependencyEdge {
            head_index: 2,
            dependent_index: 2,
            relation: DependencyRelation::Root,
        });
        s.dependency_edges.push(DependencyEdge {
            head_index: 2,
            dependent_index: 1,
            relation: DependencyRelation::Nsubj,
        });
        let ir1 = make_ir(IRType::Parsed, s.clone());
        let ir2 = make_ir(IRType::Parsed, s);
        let (dist, _) = compute_tree_edit_distance(&ir1, &ir2).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_token_jaccard() {
        let a = vec!["the", "cat", "sat"];
        let b = vec!["the", "dog", "sat"];
        let j = token_jaccard(&a, &b);
        assert!((j - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_differential_with_normalization() {
        let params = vec![NormalizationParams::new(StageId::new("tok"), 0.3, 0.1, 50)];
        let computer = DifferentialComputer::new().with_normalization(params);
        let orig = make_ir(IRType::Tokenized, make_sentence(&["a", "b"]));
        let trans = make_ir(IRType::Tokenized, make_sentence(&["a", "c"]));
        let diff = computer
            .compute_stage_differential(&StageId::new("tok"), 0, &orig, &trans)
            .unwrap();
        assert!(diff.normalized);
    }
}
