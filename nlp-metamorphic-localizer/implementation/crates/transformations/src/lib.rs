//! Linguistically-grounded NLP transformations for metamorphic testing.
//!
//! This crate provides 15 tree-transduction transformations on dependency parse
//! trees, used to generate metamorphic test inputs for NLP pipeline fault
//! localisation.

pub mod base;
pub mod passivization;
pub mod clefting;
pub mod topicalization;
pub mod relative_clause;
pub mod tense_change;
pub mod agreement_perturbation;
pub mod synonym_substitution;
pub mod negation;
pub mod coordination;
pub mod pp_attachment;
pub mod adverb_repositioning;
pub mod there_insertion;
pub mod dative_alternation;
pub mod embedding_depth;
pub mod registry;

pub use base::{
    BaseTransformation, Precondition, SyntacticPosition, TransformationError,
    TransformationKind, TransformationMetadata, TransformationResult,
};
pub use passivization::PassivizationTransform;
pub use clefting::CleftTransform;
pub use topicalization::TopicalizationTransform;
pub use relative_clause::{RelativeClauseDeletionTransform, RelativeClauseInsertTransform};
pub use tense_change::TenseChangeTransform;
pub use agreement_perturbation::AgreementPerturbationTransform;
pub use synonym_substitution::SynonymSubstitutionTransform;
pub use negation::NegationInsertionTransform;
pub use coordination::CoordinatedNpReorderTransform;
pub use pp_attachment::PpAttachmentTransform;
pub use adverb_repositioning::AdverbRepositionTransform;
pub use there_insertion::ThereInsertionTransform;
pub use dative_alternation::DativeAlternationTransform;
pub use embedding_depth::EmbeddingDepthTransform;
pub use registry::TransformationRegistry;

/// Helper to build a fully annotated sentence for testing.
pub fn make_test_sentence(
    words: &[(&str, shared_types::PosTag)],
    edges: &[(usize, usize, shared_types::DependencyRelation)],
) -> shared_types::Sentence {
    let raw = words.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" ");
    let tokens: Vec<shared_types::Token> = words
        .iter()
        .enumerate()
        .map(|(i, (w, pos))| shared_types::Token::new(*w, i).with_pos(*pos))
        .collect();
    let dep_edges: Vec<shared_types::DependencyEdge> = edges
        .iter()
        .map(|(h, d, r)| shared_types::DependencyEdge::new(*h, *d, *r))
        .collect();
    let mut s = shared_types::Sentence::from_tokens(tokens, raw);
    s.dependency_edges = dep_edges;
    s
}
