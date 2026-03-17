//! Metamorphic relation definitions and checking.

use serde::{Deserialize, Serialize};
use shared_types::{
    EntitySpan, IntermediateRepresentation, MRCheckDetail, MetamorphicRelation, Result, Sentence,
    SentenceFeatures, TransformationId,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MRType
// ---------------------------------------------------------------------------

/// Enumeration of metamorphic relation kinds.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MRType {
    SemanticEquivalence,
    SentimentPreservation,
    EntityPreservation,
    SyntacticConsistency,
    NegationFlip,
    TenseConsistency,
    VoiceInvariance,
    LabelPreservation,
    ConfidenceStability,
    Custom(String),
}

impl std::fmt::Display for MRType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MRType::SemanticEquivalence => write!(f, "SemanticEquivalence"),
            MRType::SentimentPreservation => write!(f, "SentimentPreservation"),
            MRType::EntityPreservation => write!(f, "EntityPreservation"),
            MRType::SyntacticConsistency => write!(f, "SyntacticConsistency"),
            MRType::NegationFlip => write!(f, "NegationFlip"),
            MRType::TenseConsistency => write!(f, "TenseConsistency"),
            MRType::VoiceInvariance => write!(f, "VoiceInvariance"),
            MRType::LabelPreservation => write!(f, "LabelPreservation"),
            MRType::ConfidenceStability => write!(f, "ConfidenceStability"),
            MRType::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

// ---------------------------------------------------------------------------
// MRDefinition
// ---------------------------------------------------------------------------

/// A full metamorphic-relation definition that can be stored, serialized, and checked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRDefinition {
    pub id: String,
    pub name: String,
    pub mr_type: MRType,
    pub tolerance: f64,
    pub description: String,
    pub applicable_transformations: Vec<TransformationId>,
}

impl MRDefinition {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        mr_type: MRType,
        tolerance: f64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            mr_type,
            tolerance,
            description: description.into(),
            applicable_transformations: Vec::new(),
        }
    }

    pub fn with_transformations(mut self, ts: Vec<TransformationId>) -> Self {
        self.applicable_transformations = ts;
        self
    }

    /// Dispatch to the concrete checker for this MR type.
    fn dispatch_check(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> MRCheckResult {
        match &self.mr_type {
            MRType::SemanticEquivalence => {
                let mr = SemanticEquivalenceMR::new(self.tolerance, 0.3, 0.3);
                mr.check_result(original, transformed)
            }
            MRType::EntityPreservation => {
                let mr = EntityPreservationMR::new(self.tolerance);
                mr.check_result(original, transformed)
            }
            MRType::SentimentPreservation => {
                let mr = SentimentPreservationMR::new(self.tolerance);
                mr.check_result(original, transformed)
            }
            MRType::SyntacticConsistency => {
                let mr = SyntacticConsistencyMR::new(self.tolerance);
                mr.check_result(original, transformed)
            }
            MRType::NegationFlip => {
                let mr = NegationFlipMR::new();
                mr.check_result(original, transformed)
            }
            MRType::TenseConsistency => {
                check_tense_consistency(original, transformed, self.tolerance)
            }
            MRType::VoiceInvariance => {
                check_voice_invariance(original, transformed, self.tolerance)
            }
            MRType::LabelPreservation => {
                check_label_preservation(original, transformed, self.tolerance)
            }
            MRType::ConfidenceStability => {
                check_confidence_stability(original, transformed, self.tolerance)
            }
            MRType::Custom(_) => MRCheckResult {
                passed: true,
                violation_magnitude: 0.0,
                expected: "custom".into(),
                actual: "custom".into(),
                explanation: "Custom MR – no built-in checker".into(),
            },
        }
    }
}

impl MetamorphicRelation for MRDefinition {
    fn check(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> Result<bool> {
        Ok(self.dispatch_check(original, transformed).passed)
    }

    fn check_with_detail(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> Result<MRCheckDetail> {
        let r = self.dispatch_check(original, transformed);
        Ok(MRCheckDetail {
            passed: r.passed,
            violation_magnitude: r.violation_magnitude,
            expected: r.expected,
            actual: r.actual,
            explanation: r.explanation,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

// ---------------------------------------------------------------------------
// MRCheckResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRCheckResult {
    pub passed: bool,
    pub violation_magnitude: f64,
    pub expected: String,
    pub actual: String,
    pub explanation: String,
}

impl MRCheckResult {
    pub fn pass(explanation: impl Into<String>) -> Self {
        Self {
            passed: true,
            violation_magnitude: 0.0,
            expected: String::new(),
            actual: String::new(),
            explanation: explanation.into(),
        }
    }

    pub fn fail(
        magnitude: f64,
        expected: impl Into<String>,
        actual: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            passed: false,
            violation_magnitude: magnitude,
            expected: expected.into(),
            actual: actual.into(),
            explanation: explanation.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: Jaccard similarity over token texts
// ---------------------------------------------------------------------------

fn jaccard_tokens(a: &Sentence, b: &Sentence) -> f64 {
    let set_a: std::collections::HashSet<&str> = a.tokens.iter().map(|t| t.text.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.tokens.iter().map(|t| t.text.as_str()).collect();
    let inter = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

fn entity_overlap(a: &[EntitySpan], b: &[EntitySpan]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let matched = a
        .iter()
        .filter(|ea| {
            b.iter()
                .any(|eb| ea.label == eb.label && ea.text == eb.text)
        })
        .count();
    let total = a.len().max(b.len());
    matched as f64 / total as f64
}

fn dep_edge_overlap(a: &Sentence, b: &Sentence) -> f64 {
    if a.dependency_edges.is_empty() && b.dependency_edges.is_empty() {
        return 1.0;
    }
    if a.dependency_edges.is_empty() || b.dependency_edges.is_empty() {
        return 0.0;
    }
    let matched = a
        .dependency_edges
        .iter()
        .filter(|da| {
            b.dependency_edges
                .iter()
                .any(|db| da.relation == db.relation && da.head_index == db.head_index && da.dependent_index == db.dependent_index)
        })
        .count();
    let total = a.dependency_edges.len().max(b.dependency_edges.len());
    matched as f64 / total as f64
}

fn parse_tree_similarity(a: &Sentence, b: &Sentence) -> f64 {
    // No parse_tree field on Sentence; approximate via dependency edge overlap.
    dep_edge_overlap(a, b)
}

fn features_or_default(s: &Sentence) -> SentenceFeatures {
    s.features.clone().unwrap_or_else(|| s.compute_features())
}

// ---------------------------------------------------------------------------
// SemanticEquivalenceMR
// ---------------------------------------------------------------------------

/// Checks that NLP outputs are equivalent under meaning-preserving transformations.
/// Combines token-level Jaccard, entity overlap and label overlap.
#[derive(Debug, Clone)]
pub struct SemanticEquivalenceMR {
    pub tolerance: f64,
    pub entity_weight: f64,
    pub label_weight: f64,
}

impl SemanticEquivalenceMR {
    pub fn new(tolerance: f64, entity_weight: f64, label_weight: f64) -> Self {
        Self {
            tolerance,
            entity_weight,
            label_weight,
        }
    }

    pub fn check_result(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> MRCheckResult {
        let token_sim = jaccard_tokens(&original.sentence, &transformed.sentence);
        let entity_sim = entity_overlap(&original.sentence.entities, &transformed.sentence.entities);
        let dep_sim = dep_edge_overlap(&original.sentence, &transformed.sentence);

        let token_w = 1.0 - self.entity_weight - self.label_weight;
        let combined = token_w * token_sim + self.entity_weight * entity_sim + self.label_weight * dep_sim;
        let distance = 1.0 - combined;

        if distance <= self.tolerance {
            MRCheckResult::pass(format!(
                "Semantic equivalence holds (distance={:.4}, tol={:.4})",
                distance, self.tolerance
            ))
        } else {
            MRCheckResult::fail(
                distance,
                format!("distance <= {:.4}", self.tolerance),
                format!("distance = {:.4}", distance),
                format!(
                    "token_sim={:.3}, entity_sim={:.3}, dep_sim={:.3}",
                    token_sim, entity_sim, dep_sim
                ),
            )
        }
    }
}

impl MetamorphicRelation for SemanticEquivalenceMR {
    fn check(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<bool> {
        Ok(self.check_result(orig, trans).passed)
    }
    fn check_with_detail(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<MRCheckDetail> {
        let r = self.check_result(orig, trans);
        Ok(MRCheckDetail {
            passed: r.passed,
            violation_magnitude: r.violation_magnitude,
            expected: r.expected,
            actual: r.actual,
            explanation: r.explanation,
        })
    }
    fn name(&self) -> &str {
        "SemanticEquivalence"
    }
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

// ---------------------------------------------------------------------------
// EntityPreservationMR
// ---------------------------------------------------------------------------

/// Checks that all entities in the original output are preserved after transformation.
#[derive(Debug, Clone)]
pub struct EntityPreservationMR {
    pub tolerance: f64,
}

impl EntityPreservationMR {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn check_result(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> MRCheckResult {
        let overlap = entity_overlap(
            &original.sentence.entities,
            &transformed.sentence.entities,
        );
        let distance = 1.0 - overlap;
        if distance <= self.tolerance {
            MRCheckResult::pass(format!(
                "Entity preservation holds (overlap={:.4})",
                overlap
            ))
        } else {
            let orig_ents: Vec<String> = original
                .sentence
                .entities
                .iter()
                .map(|e| format!("{}:{}", e.text, e.label))
                .collect();
            let trans_ents: Vec<String> = transformed
                .sentence
                .entities
                .iter()
                .map(|e| format!("{}:{}", e.text, e.label))
                .collect();
            MRCheckResult::fail(
                distance,
                format!("entities: {:?}", orig_ents),
                format!("entities: {:?}", trans_ents),
                format!("Entity overlap={:.4}, needed>={:.4}", overlap, 1.0 - self.tolerance),
            )
        }
    }
}

impl MetamorphicRelation for EntityPreservationMR {
    fn check(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<bool> {
        Ok(self.check_result(orig, trans).passed)
    }
    fn check_with_detail(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<MRCheckDetail> {
        let r = self.check_result(orig, trans);
        Ok(MRCheckDetail {
            passed: r.passed,
            violation_magnitude: r.violation_magnitude,
            expected: r.expected,
            actual: r.actual,
            explanation: r.explanation,
        })
    }
    fn name(&self) -> &str {
        "EntityPreservation"
    }
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

// ---------------------------------------------------------------------------
// SentimentPreservationMR
// ---------------------------------------------------------------------------

/// Checks that the sentiment label/score is preserved under transformation.
#[derive(Debug, Clone)]
pub struct SentimentPreservationMR {
    pub tolerance: f64,
}

impl SentimentPreservationMR {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn check_result(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> MRCheckResult {
        let orig_label = original.data.get("sentiment_label").and_then(|v| v.as_str()).map(String::from);
        let trans_label = transformed.data.get("sentiment_label").and_then(|v| v.as_str()).map(String::from);
        let orig_score = original.data.get("sentiment_score").and_then(|v| v.as_f64());
        let trans_score = transformed.data.get("sentiment_score").and_then(|v| v.as_f64());

        // Check label match.
        let label_match = match (&orig_label, &trans_label) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false,
        };

        // Check score distance.
        let score_dist = match (orig_score, trans_score) {
            (Some(a), Some(b)) => (a - b).abs(),
            (None, None) => 0.0,
            _ => 1.0,
        };

        if label_match && score_dist <= self.tolerance {
            MRCheckResult::pass(format!(
                "Sentiment preserved (label_match={}, score_dist={:.4})",
                label_match, score_dist
            ))
        } else {
            MRCheckResult::fail(
                score_dist.max(if label_match { 0.0 } else { 1.0 }),
                format!(
                    "label={:?} score={:?}",
                    orig_label, orig_score
                ),
                format!(
                    "label={:?} score={:?}",
                    trans_label, trans_score
                ),
                format!(
                    "label_match={}, score_dist={:.4}",
                    label_match, score_dist
                ),
            )
        }
    }
}

impl MetamorphicRelation for SentimentPreservationMR {
    fn check(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<bool> {
        Ok(self.check_result(orig, trans).passed)
    }
    fn check_with_detail(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<MRCheckDetail> {
        let r = self.check_result(orig, trans);
        Ok(MRCheckDetail {
            passed: r.passed,
            violation_magnitude: r.violation_magnitude,
            expected: r.expected,
            actual: r.actual,
            explanation: r.explanation,
        })
    }
    fn name(&self) -> &str {
        "SentimentPreservation"
    }
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

// ---------------------------------------------------------------------------
// SyntacticConsistencyMR
// ---------------------------------------------------------------------------

/// Checks that syntactic parse structure is consistent after transformation.
#[derive(Debug, Clone)]
pub struct SyntacticConsistencyMR {
    pub tolerance: f64,
}

impl SyntacticConsistencyMR {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn check_result(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> MRCheckResult {
        let dep_sim = dep_edge_overlap(&original.sentence, &transformed.sentence);
        let tree_sim = parse_tree_similarity(&original.sentence, &transformed.sentence);
        let combined = (dep_sim + tree_sim) / 2.0;
        let distance = 1.0 - combined;

        if distance <= self.tolerance {
            MRCheckResult::pass(format!(
                "Syntactic consistency holds (dep_sim={:.3}, tree_sim={:.3})",
                dep_sim, tree_sim
            ))
        } else {
            MRCheckResult::fail(
                distance,
                format!("distance <= {:.4}", self.tolerance),
                format!("distance = {:.4}", distance),
                format!("dep_sim={:.3}, tree_sim={:.3}", dep_sim, tree_sim),
            )
        }
    }
}

impl MetamorphicRelation for SyntacticConsistencyMR {
    fn check(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<bool> {
        Ok(self.check_result(orig, trans).passed)
    }
    fn check_with_detail(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<MRCheckDetail> {
        let r = self.check_result(orig, trans);
        Ok(MRCheckDetail {
            passed: r.passed,
            violation_magnitude: r.violation_magnitude,
            expected: r.expected,
            actual: r.actual,
            explanation: r.explanation,
        })
    }
    fn name(&self) -> &str {
        "SyntacticConsistency"
    }
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

// ---------------------------------------------------------------------------
// NegationFlipMR
// ---------------------------------------------------------------------------

/// Checks that sentiment/label flips correctly under negation transformation.
#[derive(Debug, Clone)]
pub struct NegationFlipMR;

impl NegationFlipMR {
    pub fn new() -> Self {
        Self
    }

    fn flip_label(label: &str) -> String {
        match label.to_lowercase().as_str() {
            "positive" => "negative".into(),
            "negative" => "positive".into(),
            "neutral" => "neutral".into(),
            other => format!("not-{}", other),
        }
    }

    pub fn check_result(
        &self,
        original: &IntermediateRepresentation,
        transformed: &IntermediateRepresentation,
    ) -> MRCheckResult {
        let fo = features_or_default(&original.sentence);
        let ft = features_or_default(&transformed.sentence);

        let orig_label = original.data.get("sentiment_label").and_then(|v| v.as_str()).map(String::from);
        let trans_label = transformed.data.get("sentiment_label").and_then(|v| v.as_str()).map(String::from);
        let orig_score = original.data.get("sentiment_score").and_then(|v| v.as_f64());
        let trans_score = transformed.data.get("sentiment_score").and_then(|v| v.as_f64());

        // Expect the transformed sentence to be marked as negated relative to original.
        let negation_present = fo.has_negation != ft.has_negation;

        // Check that sentiment label is flipped.
        let label_flipped = match (&orig_label, &trans_label) {
            (Some(orig_l), Some(trans_l)) => {
                let expected = Self::flip_label(orig_l);
                trans_l.to_lowercase() == expected.to_lowercase()
            }
            (None, None) => true,
            _ => false,
        };

        // Check score is roughly inverted (if on [-1,1] scale).
        let score_flipped = match (orig_score, trans_score) {
            (Some(a), Some(b)) => {
                let expected = -a;
                (expected - b).abs() < 0.5
            }
            (None, None) => true,
            _ => false,
        };

        let ok = (negation_present || label_flipped || score_flipped) && (label_flipped || score_flipped);

        if ok {
            MRCheckResult::pass("Negation flip verified")
        } else {
            let mag = if !label_flipped { 1.0 } else { 0.5 };
            MRCheckResult::fail(
                mag,
                format!(
                    "flipped_label={:?}, flipped_score",
                    orig_label.as_deref().map(Self::flip_label)
                ),
                format!(
                    "label={:?}, score={:?}",
                    trans_label, trans_score
                ),
                format!(
                    "negation_present={}, label_flipped={}, score_flipped={}",
                    negation_present, label_flipped, score_flipped
                ),
            )
        }
    }
}

impl MetamorphicRelation for NegationFlipMR {
    fn check(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<bool> {
        Ok(self.check_result(orig, trans).passed)
    }
    fn check_with_detail(&self, orig: &IntermediateRepresentation, trans: &IntermediateRepresentation) -> Result<MRCheckDetail> {
        let r = self.check_result(orig, trans);
        Ok(MRCheckDetail {
            passed: r.passed,
            violation_magnitude: r.violation_magnitude,
            expected: r.expected,
            actual: r.actual,
            explanation: r.explanation,
        })
    }
    fn name(&self) -> &str {
        "NegationFlip"
    }
    fn tolerance(&self) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Standalone checkers for remaining MR types
// ---------------------------------------------------------------------------

fn check_tense_consistency(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
    _tolerance: f64,
) -> MRCheckResult {
    let fo = features_or_default(&original.sentence);
    let ft = features_or_default(&transformed.sentence);
    if fo.tense == ft.tense {
        MRCheckResult::pass(format!("Tense consistent: {:?}", fo.tense))
    } else {
        MRCheckResult::fail(
            1.0,
            format!("{:?}", fo.tense),
            format!("{:?}", ft.tense),
            "Tense changed unexpectedly",
        )
    }
}

fn check_voice_invariance(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
    _tolerance: f64,
) -> MRCheckResult {
    let fo = features_or_default(&original.sentence);
    let ft = features_or_default(&transformed.sentence);
    if fo.voice == ft.voice {
        MRCheckResult::pass(format!("Voice invariant: {:?}", fo.voice))
    } else {
        MRCheckResult::fail(
            1.0,
            format!("{:?}", fo.voice),
            format!("{:?}", ft.voice),
            "Voice changed",
        )
    }
}

fn check_label_preservation(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
    tolerance: f64,
) -> MRCheckResult {
    let orig_label = original.data.get("sentiment_label").and_then(|v| v.as_str());
    let trans_label = transformed.data.get("sentiment_label").and_then(|v| v.as_str());
    let label_match = orig_label == trans_label;
    let all_keys: std::collections::HashSet<&String> =
        original.data.keys().chain(transformed.data.keys()).collect();
    let extra_match_ratio = if all_keys.is_empty() {
        1.0
    } else {
        let matching = all_keys
            .iter()
            .filter(|k| original.data.get(**k) == transformed.data.get(**k))
            .count();
        matching as f64 / all_keys.len() as f64
    };
    let distance = if label_match { 0.0 } else { 1.0 } * 0.6 + (1.0 - extra_match_ratio) * 0.4;
    if distance <= tolerance {
        MRCheckResult::pass(format!(
            "Labels preserved (label_match={}, extra_ratio={:.3})",
            label_match, extra_match_ratio
        ))
    } else {
        MRCheckResult::fail(
            distance,
            format!("label={:?}", orig_label),
            format!("label={:?}", trans_label),
            format!(
                "label_match={}, extra_match_ratio={:.3}",
                label_match, extra_match_ratio
            ),
        )
    }
}

fn check_confidence_stability(
    original: &IntermediateRepresentation,
    transformed: &IntermediateRepresentation,
    tolerance: f64,
) -> MRCheckResult {
    let orig_conf = original.confidence.or_else(|| original.data.get("confidence").and_then(|v| v.as_f64()));
    let trans_conf = transformed.confidence.or_else(|| transformed.data.get("confidence").and_then(|v| v.as_f64()));
    match (orig_conf, trans_conf) {
        (Some(a), Some(b)) => {
            let diff = (a - b).abs();
            if diff <= tolerance {
                MRCheckResult::pass(format!(
                    "Confidence stable (diff={:.4}, tol={:.4})",
                    diff, tolerance
                ))
            } else {
                MRCheckResult::fail(
                    diff,
                    format!("{:.4}", a),
                    format!("{:.4}", b),
                    format!("Confidence drifted by {:.4}", diff),
                )
            }
        }
        (None, None) => MRCheckResult::pass("No confidence values, trivially stable"),
        _ => MRCheckResult::fail(1.0, "confidence present", "confidence absent", "Confidence info mismatch"),
    }
}

// ---------------------------------------------------------------------------
// MRRegistry
// ---------------------------------------------------------------------------

/// Registry of named metamorphic-relation definitions.
#[derive(Debug, Clone, Default)]
pub struct MRRegistry {
    relations: HashMap<String, MRDefinition>,
}

impl MRRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, def: MRDefinition) {
        self.relations.insert(def.id.clone(), def);
    }

    pub fn get(&self, id: &str) -> Option<&MRDefinition> {
        self.relations.get(id)
    }

    pub fn list(&self) -> Vec<&MRDefinition> {
        self.relations.values().collect()
    }

    pub fn len(&self) -> usize {
        self.relations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    pub fn remove(&mut self, id: &str) -> Option<MRDefinition> {
        self.relations.remove(id)
    }

    /// Create a registry pre-populated with standard MR definitions.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(MRDefinition::new(
            "sem-equiv",
            "Semantic Equivalence",
            MRType::SemanticEquivalence,
            0.15,
            "Checks NLP outputs are semantically equivalent under meaning-preserving transformations",
        ));
        reg.register(MRDefinition::new(
            "entity-pres",
            "Entity Preservation",
            MRType::EntityPreservation,
            0.1,
            "Checks that named entities are preserved",
        ));
        reg.register(MRDefinition::new(
            "sentiment-pres",
            "Sentiment Preservation",
            MRType::SentimentPreservation,
            0.1,
            "Checks that sentiment label and score are preserved",
        ));
        reg.register(MRDefinition::new(
            "syntax-con",
            "Syntactic Consistency",
            MRType::SyntacticConsistency,
            0.2,
            "Checks parse structure consistency",
        ));
        reg.register(MRDefinition::new(
            "neg-flip",
            "Negation Flip",
            MRType::NegationFlip,
            0.0,
            "Checks sentiment flips under negation",
        ));
        reg.register(MRDefinition::new(
            "tense-con",
            "Tense Consistency",
            MRType::TenseConsistency,
            0.0,
            "Checks tense is preserved",
        ));
        reg.register(MRDefinition::new(
            "voice-inv",
            "Voice Invariance",
            MRType::VoiceInvariance,
            0.0,
            "Checks voice is preserved",
        ));
        reg.register(MRDefinition::new(
            "label-pres",
            "Label Preservation",
            MRType::LabelPreservation,
            0.1,
            "Checks labels are preserved",
        ));
        reg.register(MRDefinition::new(
            "conf-stab",
            "Confidence Stability",
            MRType::ConfidenceStability,
            0.1,
            "Checks confidence scores are stable",
        ));
        reg
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::*;

    fn make_ir(text: &str) -> IntermediateRepresentation {
        IntermediateRepresentation::new(IRType::RawText, Sentence::from_text(text))
    }

    fn make_ir_with_entities(text: &str, ents: Vec<EntitySpan>) -> IntermediateRepresentation {
        let mut s = Sentence::from_text(text);
        s.entities = ents;
        IntermediateRepresentation::new(IRType::EntityRecognized, s)
    }

    fn make_ir_with_sentiment(
        text: &str,
        label: &str,
        score: f64,
        _negated: bool,
    ) -> IntermediateRepresentation {
        let s = Sentence::from_text(text);
        let mut ir = IntermediateRepresentation::new(IRType::Custom("SentimentScored".into()), s);
        ir.data.insert("sentiment_label".into(), serde_json::Value::String(label.into()));
        ir.data.insert("sentiment_score".into(), serde_json::json!(score));
        ir
    }

    fn entity(text: &str, label: EntityLabel) -> EntitySpan {
        EntitySpan {
            text: text.into(),
            label,
            start: 0,
            end: text.len(),
            confidence: 0.9,
        }
    }

    #[test]
    fn test_semantic_equivalence_pass() {
        let orig = make_ir("The quick brown fox jumps");
        let trans = make_ir("The quick brown fox leaps");
        let mr = SemanticEquivalenceMR::new(0.5, 0.3, 0.3);
        let r = mr.check_result(&orig, &trans);
        assert!(r.passed, "Expected pass: {}", r.explanation);
    }

    #[test]
    fn test_semantic_equivalence_fail() {
        let orig = make_ir("alpha beta gamma");
        let trans = make_ir("delta epsilon zeta");
        let mr = SemanticEquivalenceMR::new(0.05, 0.3, 0.3);
        let r = mr.check_result(&orig, &trans);
        assert!(!r.passed);
    }

    #[test]
    fn test_entity_preservation_pass() {
        let ents = vec![entity("Google", EntityLabel::Organization)];
        let orig = make_ir_with_entities("Google is great", ents.clone());
        let trans = make_ir_with_entities("Google is wonderful", ents);
        let mr = EntityPreservationMR::new(0.1);
        assert!(mr.check_result(&orig, &trans).passed);
    }

    #[test]
    fn test_entity_preservation_fail() {
        let orig = make_ir_with_entities(
            "Google is great",
            vec![entity("Google", EntityLabel::Organization)],
        );
        let trans = make_ir_with_entities(
            "Apple is great",
            vec![entity("Apple", EntityLabel::Organization)],
        );
        let mr = EntityPreservationMR::new(0.1);
        assert!(!mr.check_result(&orig, &trans).passed);
    }

    #[test]
    fn test_sentiment_preservation_pass() {
        let orig = make_ir_with_sentiment("I love this", "positive", 0.9, false);
        let trans = make_ir_with_sentiment("I adore this", "positive", 0.85, false);
        let mr = SentimentPreservationMR::new(0.1);
        assert!(mr.check_result(&orig, &trans).passed);
    }

    #[test]
    fn test_sentiment_preservation_fail() {
        let orig = make_ir_with_sentiment("I love this", "positive", 0.9, false);
        let trans = make_ir_with_sentiment("I hate this", "negative", -0.8, false);
        let mr = SentimentPreservationMR::new(0.1);
        assert!(!mr.check_result(&orig, &trans).passed);
    }

    #[test]
    fn test_negation_flip_pass() {
        let orig = make_ir_with_sentiment("I love this", "positive", 0.9, false);
        let trans = make_ir_with_sentiment("I do not love this", "negative", -0.7, true);
        let mr = NegationFlipMR::new();
        assert!(mr.check_result(&orig, &trans).passed);
    }

    #[test]
    fn test_negation_flip_fail() {
        let orig = make_ir_with_sentiment("I love this", "positive", 0.9, false);
        let trans = make_ir_with_sentiment("I do not love this", "positive", 0.8, true);
        let mr = NegationFlipMR::new();
        assert!(!mr.check_result(&orig, &trans).passed);
    }

    #[test]
    fn test_syntactic_consistency_identical() {
        let orig = make_ir("The cat sat on the mat");
        let trans = make_ir("The cat sat on the mat");
        let mr = SyntacticConsistencyMR::new(0.1);
        assert!(mr.check_result(&orig, &trans).passed);
    }

    #[test]
    fn test_registry_with_defaults() {
        let reg = MRRegistry::with_defaults();
        assert!(reg.len() >= 9);
        assert!(reg.get("sem-equiv").is_some());
        assert!(reg.get("neg-flip").is_some());
    }

    #[test]
    fn test_registry_crud() {
        let mut reg = MRRegistry::new();
        let def = MRDefinition::new("x", "X", MRType::Custom("x".into()), 0.1, "test");
        reg.register(def);
        assert_eq!(reg.len(), 1);
        assert!(reg.get("x").is_some());
        reg.remove("x");
        assert!(reg.is_empty());
    }

    #[test]
    fn test_mr_definition_dispatch() {
        let def = MRDefinition::new(
            "conf",
            "Confidence Stability",
            MRType::ConfidenceStability,
            0.1,
            "test",
        );
        let mut orig = make_ir("hello");
        orig.data.insert("confidence".into(), serde_json::json!(0.9));
        let mut trans = make_ir("hello");
        trans.data.insert("confidence".into(), serde_json::json!(0.88));
        let r = def.dispatch_check(&orig, &trans);
        assert!(r.passed);
    }

    #[test]
    fn test_tense_consistency() {
        let orig = make_ir("He runs");
        let trans = make_ir("He runs fast");
        let r = check_tense_consistency(&orig, &trans, 0.0);
        assert!(r.passed);
    }

    #[test]
    fn test_voice_invariance() {
        let orig = make_ir("The cat ate the fish");
        let trans = make_ir("The cat ate the fish quickly");
        let r = check_voice_invariance(&orig, &trans, 0.0);
        assert!(r.passed);
    }

    #[test]
    fn test_confidence_stability_fail() {
        let mut orig = make_ir("hello");
        orig.data.insert("confidence".into(), serde_json::json!(0.95));
        let mut trans = make_ir("hello");
        trans.data.insert("confidence".into(), serde_json::json!(0.5));
        let r = check_confidence_stability(&orig, &trans, 0.1);
        assert!(!r.passed);
    }
}
