//! Transformation registry: stores all transformations and provides lookup,
//! applicability queries, and coverage analysis.

use crate::base::{BaseTransformation, TransformationKind, TransformationResult, TransformationError};
use crate::{
    adverb_repositioning::AdverbRepositionTransform,
    agreement_perturbation::AgreementPerturbationTransform,
    clefting::CleftTransform,
    coordination::CoordinatedNpReorderTransform,
    dative_alternation::DativeAlternationTransform,
    embedding_depth::EmbeddingDepthTransform,
    negation::NegationInsertionTransform,
    passivization::PassivizationTransform,
    pp_attachment::PpAttachmentTransform,
    relative_clause::{RelativeClauseDeletionTransform, RelativeClauseInsertTransform},
    synonym_substitution::SynonymSubstitutionTransform,
    tense_change::TenseChangeTransform,
    there_insertion::ThereInsertionTransform,
    topicalization::TopicalizationTransform,
};
use shared_types::Sentence;
use std::collections::HashMap;

/// A boxed transformation.
type BoxedTransformation = Box<dyn BaseTransformation>;

/// Registry holding all transformations keyed by `TransformationKind`.
pub struct TransformationRegistry {
    transformations: HashMap<TransformationKind, BoxedTransformation>,
}

impl TransformationRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            transformations: HashMap::new(),
        }
    }

    /// Register a transformation.
    pub fn register(&mut self, transformation: BoxedTransformation) {
        let kind = transformation.kind();
        self.transformations.insert(kind, transformation);
    }

    /// Retrieve a transformation by kind.
    pub fn get(&self, kind: &TransformationKind) -> Option<&dyn BaseTransformation> {
        self.transformations.get(kind).map(|b| b.as_ref())
    }

    /// List all registered transformation kinds.
    pub fn list(&self) -> Vec<TransformationKind> {
        let mut kinds: Vec<TransformationKind> = self.transformations.keys().copied().collect();
        kinds.sort_by_key(|k| format!("{:?}", k));
        kinds
    }

    /// Return all transformations that are applicable to the given sentence.
    pub fn get_applicable(&self, sentence: &Sentence) -> Vec<TransformationKind> {
        self.transformations
            .iter()
            .filter(|(_, t)| t.is_applicable(sentence))
            .map(|(k, _)| *k)
            .collect()
    }

    /// Return transformations applicable to a sentence, sorted by kind name.
    pub fn get_transformations_for_sentence(
        &self,
        sentence: &Sentence,
    ) -> Vec<(&TransformationKind, &dyn BaseTransformation)> {
        let mut applicable: Vec<_> = self
            .transformations
            .iter()
            .filter(|(_, t)| t.is_applicable(sentence))
            .map(|(k, t)| (k, t.as_ref()))
            .collect();
        applicable.sort_by_key(|(k, _)| format!("{:?}", k));
        applicable
    }

    /// Coverage analysis: for each sentence, how many transformations are applicable.
    pub fn coverage_analysis(&self, sentences: &[Sentence]) -> CoverageReport {
        let mut per_sentence: Vec<usize> = Vec::new();
        let mut per_transformation: HashMap<TransformationKind, usize> = HashMap::new();

        for sentence in sentences {
            let applicable = self.get_applicable(sentence);
            per_sentence.push(applicable.len());
            for kind in &applicable {
                *per_transformation.entry(*kind).or_insert(0) += 1;
            }
        }

        let total_sentences = sentences.len();
        let avg_coverage = if total_sentences > 0 {
            per_sentence.iter().sum::<usize>() as f64 / total_sentences as f64
        } else {
            0.0
        };

        CoverageReport {
            total_sentences,
            total_transformations: self.transformations.len(),
            average_applicable_per_sentence: avg_coverage,
            per_sentence_counts: per_sentence,
            per_transformation_counts: per_transformation,
        }
    }

    /// Number of registered transformations.
    pub fn len(&self) -> usize {
        self.transformations.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.transformations.is_empty()
    }

    /// Apply all applicable transformations to a sentence and collect results.
    pub fn apply_all_applicable(
        &self,
        sentence: &Sentence,
    ) -> Vec<(TransformationKind, Result<TransformationResult, TransformationError>)> {
        self.transformations
            .iter()
            .filter(|(_, t)| t.is_applicable(sentence))
            .map(|(k, t)| (*k, t.apply(sentence)))
            .collect()
    }
}

impl Default for TransformationRegistry {
    fn default() -> Self {
        create_default_registry()
    }
}

/// Coverage analysis report.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    pub total_sentences: usize,
    pub total_transformations: usize,
    pub average_applicable_per_sentence: f64,
    pub per_sentence_counts: Vec<usize>,
    pub per_transformation_counts: HashMap<TransformationKind, usize>,
}

impl CoverageReport {
    /// Transformation kinds that were never applicable.
    pub fn uncovered_transformations(
        &self,
        all_kinds: &[TransformationKind],
    ) -> Vec<TransformationKind> {
        all_kinds
            .iter()
            .filter(|k| self.per_transformation_counts.get(k).copied().unwrap_or(0) == 0)
            .copied()
            .collect()
    }

    /// Sentences with zero applicable transformations.
    pub fn uncovered_sentences(&self) -> Vec<usize> {
        self.per_sentence_counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == 0)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Create a registry pre-populated with all 15 transformations.
pub fn create_default_registry() -> TransformationRegistry {
    let mut reg = TransformationRegistry::new();

    reg.register(Box::new(PassivizationTransform::new()));
    reg.register(Box::new(CleftTransform::new()));
    reg.register(Box::new(TopicalizationTransform::new()));
    reg.register(Box::new(RelativeClauseInsertTransform::new()));
    reg.register(Box::new(RelativeClauseDeletionTransform::new()));
    reg.register(Box::new(TenseChangeTransform::new()));
    reg.register(Box::new(AgreementPerturbationTransform::new()));
    reg.register(Box::new(SynonymSubstitutionTransform::new()));
    reg.register(Box::new(NegationInsertionTransform::new()));
    reg.register(Box::new(CoordinatedNpReorderTransform::new()));
    reg.register(Box::new(PpAttachmentTransform::new()));
    reg.register(Box::new(AdverbRepositionTransform::new()));
    reg.register(Box::new(ThereInsertionTransform::new()));
    reg.register(Box::new(DativeAlternationTransform::new()));
    reg.register(Box::new(EmbeddingDepthTransform::new()));

    reg
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn active_transitive() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("cat", 1).with_pos(PosTag::Noun),
            Token::new("chased", 2).with_pos(PosTag::Verb).with_lemma("chase"),
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
    fn test_default_registry_has_15() {
        let reg = create_default_registry();
        assert_eq!(reg.len(), 15);
    }

    #[test]
    fn test_list() {
        let reg = create_default_registry();
        let kinds = reg.list();
        assert_eq!(kinds.len(), 15);
    }

    #[test]
    fn test_get() {
        let reg = create_default_registry();
        assert!(reg.get(&TransformationKind::Passivization).is_some());
        assert!(reg.get(&TransformationKind::Clefting).is_some());
    }

    #[test]
    fn test_get_applicable() {
        let reg = create_default_registry();
        let s = active_transitive();
        let applicable = reg.get_applicable(&s);
        // Should include at least passivization, clefting, topicalization
        assert!(applicable.contains(&TransformationKind::Passivization));
        assert!(applicable.contains(&TransformationKind::Topicalization));
    }

    #[test]
    fn test_coverage_analysis() {
        let reg = create_default_registry();
        let sentences = vec![active_transitive()];
        let report = reg.coverage_analysis(&sentences);
        assert_eq!(report.total_sentences, 1);
        assert_eq!(report.total_transformations, 15);
        assert!(report.average_applicable_per_sentence > 0.0);
    }

    #[test]
    fn test_uncovered() {
        let reg = create_default_registry();
        let sentences = vec![active_transitive()];
        let report = reg.coverage_analysis(&sentences);
        let uncovered = report.uncovered_sentences();
        // The active transitive sentence should have at least some coverage
        assert!(uncovered.is_empty());
    }

    #[test]
    fn test_apply_all_applicable() {
        let reg = create_default_registry();
        let s = active_transitive();
        let results = reg.apply_all_applicable(&s);
        assert!(!results.is_empty());
        for (kind, result) in &results {
            match result {
                Ok(r) => assert!(r.success, "{:?} failed: {}", kind, r.explanation),
                Err(e) => panic!("{:?} errored: {}", kind, e),
            }
        }
    }

    #[test]
    fn test_empty_registry() {
        let reg = TransformationRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_register() {
        let mut reg = TransformationRegistry::new();
        reg.register(Box::new(PassivizationTransform::new()));
        assert_eq!(reg.len(), 1);
        assert!(reg.get(&TransformationKind::Passivization).is_some());
    }
}
