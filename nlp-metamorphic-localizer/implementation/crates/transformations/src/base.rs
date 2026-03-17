//! Base transformation infrastructure: enums, metadata, error types, and the
//! `BaseTransformation` trait that all 15 concrete transformations implement.

use shared_types::{
    DependencyEdge, DependencyRelation, PosTag, Sentence, Token, TransformationId,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── TransformationKind ──────────────────────────────────────────────────────

/// Enumerates every transformation shipped with this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformationKind {
    Passivization,
    Clefting,
    Topicalization,
    RelativeClauseInsertion,
    RelativeClauseDeletion,
    TenseChange,
    AgreementPerturbation,
    SynonymSubstitution,
    NegationInsertion,
    CoordinatedNpReorder,
    PpAttachmentVariation,
    AdverbRepositioning,
    ThereInsertion,
    DativeAlternation,
    EmbeddingDepthChange,
}

impl TransformationKind {
    pub fn all() -> &'static [TransformationKind] {
        &[
            Self::Passivization,
            Self::Clefting,
            Self::Topicalization,
            Self::RelativeClauseInsertion,
            Self::RelativeClauseDeletion,
            Self::TenseChange,
            Self::AgreementPerturbation,
            Self::SynonymSubstitution,
            Self::NegationInsertion,
            Self::CoordinatedNpReorder,
            Self::PpAttachmentVariation,
            Self::AdverbRepositioning,
            Self::ThereInsertion,
            Self::DativeAlternation,
            Self::EmbeddingDepthChange,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Passivization => "Passivization",
            Self::Clefting => "Clefting",
            Self::Topicalization => "Topicalization",
            Self::RelativeClauseInsertion => "Relative Clause Insertion",
            Self::RelativeClauseDeletion => "Relative Clause Deletion",
            Self::TenseChange => "Tense Change",
            Self::AgreementPerturbation => "Agreement Perturbation",
            Self::SynonymSubstitution => "Synonym Substitution",
            Self::NegationInsertion => "Negation Insertion",
            Self::CoordinatedNpReorder => "Coordinated NP Reorder",
            Self::PpAttachmentVariation => "PP Attachment Variation",
            Self::AdverbRepositioning => "Adverb Repositioning",
            Self::ThereInsertion => "There-Insertion",
            Self::DativeAlternation => "Dative Alternation",
            Self::EmbeddingDepthChange => "Embedding Depth Change",
        }
    }
}

impl fmt::Display for TransformationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ── Precondition ────────────────────────────────────────────────────────────

/// A precondition that a sentence must satisfy before a transformation applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precondition {
    HasActiveVoice,
    HasTransitiveVerb,
    HasRelativeClause,
    HasCoordination,
    HasPrepPhrase,
    HasAdverb,
    HasIndefiniteSubject,
    HasDitransitiveVerb,
    HasEmbeddedClause,
    MinTokenCount(usize),
    HasNominalSubject,
    Custom(String),
}

/// Check whether `sentence` satisfies `precondition`.
pub fn check_precondition(sentence: &Sentence, precondition: &Precondition) -> bool {
    match precondition {
        Precondition::HasActiveVoice => {
            let feats = sentence.compute_features();
            feats.voice == Some(shared_types::Voice::Active)
        }
        Precondition::HasTransitiveVerb => {
            sentence
                .dependency_edges
                .iter()
                .any(|e| e.relation == DependencyRelation::Dobj)
        }
        Precondition::HasRelativeClause => {
            sentence
                .dependency_edges
                .iter()
                .any(|e| e.relation == DependencyRelation::Relcl)
        }
        Precondition::HasCoordination => {
            sentence
                .dependency_edges
                .iter()
                .any(|e| e.relation == DependencyRelation::Conj)
        }
        Precondition::HasPrepPhrase => {
            sentence
                .dependency_edges
                .iter()
                .any(|e| e.relation == DependencyRelation::Prep)
        }
        Precondition::HasAdverb => {
            sentence.tokens.iter().any(|t| t.pos_tag == Some(PosTag::Adv))
        }
        Precondition::HasIndefiniteSubject => {
            if let Some(subj_edge) = sentence
                .dependency_edges
                .iter()
                .find(|e| e.relation == DependencyRelation::Nsubj)
            {
                let subj_idx = subj_edge.dependent_index;
                sentence.dependency_edges.iter().any(|e| {
                    e.head_index == subj_idx
                        && e.relation == DependencyRelation::Det
                        && sentence
                            .tokens
                            .get(e.dependent_index)
                            .map_or(false, |t| {
                                let low = t.text.to_lowercase();
                                low == "a" || low == "an" || low == "some"
                            })
                })
            } else {
                false
            }
        }
        Precondition::HasDitransitiveVerb => {
            let has_dobj = sentence
                .dependency_edges
                .iter()
                .any(|e| e.relation == DependencyRelation::Dobj);
            let has_iobj = sentence
                .dependency_edges
                .iter()
                .any(|e| e.relation == DependencyRelation::Iobj);
            has_dobj && has_iobj
        }
        Precondition::HasEmbeddedClause => {
            sentence.dependency_edges.iter().any(|e| {
                e.relation == DependencyRelation::Ccomp
                    || e.relation == DependencyRelation::Xcomp
            })
        }
        Precondition::MinTokenCount(n) => sentence.tokens.len() >= *n,
        Precondition::HasNominalSubject => {
            sentence
                .dependency_edges
                .iter()
                .any(|e| e.relation == DependencyRelation::Nsubj)
        }
        Precondition::Custom(name) => {
            match name.as_str() {
                "has_copular_verb" => sentence.tokens.iter().any(|t| {
                    let low = t.text.to_lowercase();
                    (low == "is" || low == "are" || low == "was" || low == "were" || low == "be")
                        && (t.pos_tag == Some(PosTag::Verb) || t.pos_tag == Some(PosTag::Aux))
                }),
                "has_auxiliary" => sentence
                    .tokens
                    .iter()
                    .any(|t| t.pos_tag == Some(PosTag::Aux)),
                _ => true,
            }
        }
    }
}

// ── TransformationMetadata ──────────────────────────────────────────────────

/// Descriptive metadata attached to every transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationMetadata {
    pub kind: TransformationKind,
    pub name: String,
    pub description: String,
    pub preconditions: Vec<Precondition>,
    pub meaning_preserving: bool,
    pub applicable_mr_types: Vec<String>,
}

impl TransformationMetadata {
    pub fn new(kind: TransformationKind) -> Self {
        Self {
            kind,
            name: kind.name().to_string(),
            description: String::new(),
            preconditions: Vec::new(),
            meaning_preserving: true,
            applicable_mr_types: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_preconditions(mut self, pre: Vec<Precondition>) -> Self {
        self.preconditions = pre;
        self
    }

    pub fn with_meaning_preserving(mut self, mp: bool) -> Self {
        self.meaning_preserving = mp;
        self
    }

    pub fn with_mr_types(mut self, types: Vec<String>) -> Self {
        self.applicable_mr_types = types;
        self
    }
}

// ── SyntacticPosition ───────────────────────────────────────────────────────

/// Records a position in the token sequence that was modified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntacticPosition {
    pub start: usize,
    pub end: usize,
    pub role: String,
}

impl SyntacticPosition {
    pub fn new(start: usize, end: usize, role: impl Into<String>) -> Self {
        Self {
            start,
            end,
            role: role.into(),
        }
    }

    pub fn span_len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

// ── TransformationResult ────────────────────────────────────────────────────

/// The outcome of applying a transformation to a sentence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationResult {
    pub success: bool,
    pub original: Sentence,
    pub transformed: Sentence,
    pub positions_modified: Vec<(usize, usize)>,
    pub explanation: String,
}

impl TransformationResult {
    pub fn ok(
        original: Sentence,
        transformed: Sentence,
        positions: Vec<(usize, usize)>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            success: true,
            original,
            transformed,
            positions_modified: positions,
            explanation: explanation.into(),
        }
    }

    pub fn fail(original: Sentence, explanation: impl Into<String>) -> Self {
        Self {
            success: false,
            original: original.clone(),
            transformed: original,
            positions_modified: Vec::new(),
            explanation: explanation.into(),
        }
    }
}

// ── TransformationError ─────────────────────────────────────────────────────

/// Errors specific to transformation application.
#[derive(Debug, thiserror::Error)]
pub enum TransformationError {
    #[error("precondition not met: {0}")]
    PreconditionNotMet(String),

    #[error("no applicable site found in sentence")]
    NoApplicableSite,

    #[error("structural error: {0}")]
    StructuralError(String),

    #[error("morphological error: {0}")]
    MorphologicalError(String),

    #[error("inverse not possible: {0}")]
    InverseNotPossible(String),

    #[error("internal error: {0}")]
    Internal(String),
}

// ── BaseTransformation ──────────────────────────────────────────────────────

/// The core trait that all 15 transformations implement.
pub trait BaseTransformation: Send + Sync {
    /// Unique kind discriminant.
    fn kind(&self) -> TransformationKind;

    /// Whether this transformation is applicable to `sentence`.
    fn is_applicable(&self, sentence: &Sentence) -> bool;

    /// Apply the transformation, producing a `TransformationResult`.
    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError>;

    /// Whether the transformation preserves meaning.
    fn is_meaning_preserving(&self) -> bool;

    // ── default helpers ─────────────────────────────────────────────────

    /// Human-readable name.
    fn name(&self) -> &str {
        self.kind().name()
    }

    /// Full metadata.
    fn metadata(&self) -> TransformationMetadata {
        TransformationMetadata::new(self.kind())
            .with_meaning_preserving(self.is_meaning_preserving())
    }
}

// ── Sentence-building helpers ───────────────────────────────────────────────

/// Rebuild a `Sentence` from a new token list, rewriting `raw_text`.
pub fn rebuild_sentence(tokens: Vec<Token>, original: &Sentence) -> Sentence {
    let raw = tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" ");
    Sentence {
        tokens,
        dependency_edges: Vec::new(),
        entities: Vec::new(),
        raw_text: raw,
        parse_tree: None,
        features: None,
    }
}

/// Re-index tokens so that `token.index` matches the position in the vec.
pub fn reindex_tokens(tokens: &mut [Token]) {
    for (i, t) in tokens.iter_mut().enumerate() {
        t.index = i;
    }
}

/// Find the index of the first token with the given dependency relation as dependent.
pub fn find_dep_index(sentence: &Sentence, rel: DependencyRelation) -> Option<usize> {
    sentence
        .dependency_edges
        .iter()
        .find(|e| e.relation == rel)
        .map(|e| e.dependent_index)
}

/// Collect all dependent indices for a head with a given relation.
pub fn find_all_deps(sentence: &Sentence, head_index: usize, rel: DependencyRelation) -> Vec<usize> {
    sentence
        .dependency_edges
        .iter()
        .filter(|e| e.head_index == head_index && e.relation == rel)
        .map(|e| e.dependent_index)
        .collect()
}

/// Return the full span (min_idx..=max_idx) of a subtree rooted at `root_idx`.
pub fn subtree_span(sentence: &Sentence, root_idx: usize) -> (usize, usize) {
    let mut min = root_idx;
    let mut max = root_idx;
    let mut stack = vec![root_idx];
    while let Some(idx) = stack.pop() {
        if idx < min {
            min = idx;
        }
        if idx > max {
            max = idx;
        }
        for dep in sentence.dependents_of(idx) {
            stack.push(dep);
        }
    }
    (min, max)
}

/// Collect all token indices in the subtree rooted at `root_idx`, sorted.
pub fn subtree_indices(sentence: &Sentence, root_idx: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut stack = vec![root_idx];
    while let Some(idx) = stack.pop() {
        result.push(idx);
        for dep in sentence.dependents_of(idx) {
            stack.push(dep);
        }
    }
    result.sort();
    result
}

/// Extract tokens for a subtree, preserving surface order.
pub fn subtree_tokens(sentence: &Sentence, root_idx: usize) -> Vec<Token> {
    let indices = subtree_indices(sentence, root_idx);
    indices
        .iter()
        .filter_map(|&i| sentence.tokens.get(i).cloned())
        .collect()
}

/// Capitalize the first character of a string.
pub fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Lower-case the first character of a string.
pub fn decapitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn sample_sentence() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("cat", 1).with_pos(PosTag::Noun),
            Token::new("chased", 2).with_pos(PosTag::Verb),
            Token::new("the", 3).with_pos(PosTag::Det),
            Token::new("mouse", 4).with_pos(PosTag::Noun),
            Token::new(".", 5).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 4, DependencyRelation::Dobj),
            DependencyEdge::new(4, 3, DependencyRelation::Det),
            DependencyEdge::new(2, 5, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The cat chased the mouse.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_transformation_kind_all() {
        assert_eq!(TransformationKind::all().len(), 15);
    }

    #[test]
    fn test_transformation_kind_name() {
        assert_eq!(
            TransformationKind::Passivization.name(),
            "Passivization"
        );
        assert_eq!(
            TransformationKind::DativeAlternation.name(),
            "Dative Alternation"
        );
    }

    #[test]
    fn test_precondition_has_transitive_verb() {
        let s = sample_sentence();
        assert!(check_precondition(&s, &Precondition::HasTransitiveVerb));
        assert!(check_precondition(&s, &Precondition::HasNominalSubject));
        assert!(!check_precondition(&s, &Precondition::HasRelativeClause));
    }

    #[test]
    fn test_precondition_min_token_count() {
        let s = sample_sentence();
        assert!(check_precondition(&s, &Precondition::MinTokenCount(3)));
        assert!(!check_precondition(&s, &Precondition::MinTokenCount(100)));
    }

    #[test]
    fn test_precondition_has_adverb() {
        let s = sample_sentence();
        assert!(!check_precondition(&s, &Precondition::HasAdverb));
    }

    #[test]
    fn test_subtree_span() {
        let s = sample_sentence();
        let (lo, hi) = subtree_span(&s, 4);
        assert_eq!(lo, 3);
        assert_eq!(hi, 4);
    }

    #[test]
    fn test_subtree_indices() {
        let s = sample_sentence();
        let idx = subtree_indices(&s, 1);
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(decapitalize("Hello"), "hello");
        assert_eq!(capitalize(""), "");
    }

    #[test]
    fn test_transformation_result_ok() {
        let s = sample_sentence();
        let r = TransformationResult::ok(s.clone(), s.clone(), vec![(0, 2)], "test");
        assert!(r.success);
    }

    #[test]
    fn test_transformation_result_fail() {
        let s = sample_sentence();
        let r = TransformationResult::fail(s, "not applicable");
        assert!(!r.success);
    }

    #[test]
    fn test_reindex_tokens() {
        let mut toks = vec![
            Token::new("a", 5),
            Token::new("b", 9),
        ];
        reindex_tokens(&mut toks);
        assert_eq!(toks[0].index, 0);
        assert_eq!(toks[1].index, 1);
    }

    #[test]
    fn test_find_dep_index() {
        let s = sample_sentence();
        assert_eq!(find_dep_index(&s, DependencyRelation::Dobj), Some(4));
        assert_eq!(find_dep_index(&s, DependencyRelation::Relcl), None);
    }

    #[test]
    fn test_metadata() {
        let m = TransformationMetadata::new(TransformationKind::Clefting)
            .with_description("cleft")
            .with_meaning_preserving(true);
        assert_eq!(m.kind, TransformationKind::Clefting);
        assert!(m.meaning_preserving);
    }

    #[test]
    fn test_syntactic_position() {
        let sp = SyntacticPosition::new(2, 5, "verb_phrase");
        assert_eq!(sp.span_len(), 3);
    }
}
