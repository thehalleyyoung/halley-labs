//! Agreement perturbation: deliberately breaks subject-verb agreement for
//! metamorphic testing of error-detection pipelines.

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// Flip number on a copula / auxiliary.
fn flip_copula(word: &str) -> Option<&'static str> {
    match word.to_lowercase().as_str() {
        "is" => Some("are"),
        "are" => Some("is"),
        "was" => Some("were"),
        "were" => Some("was"),
        "has" => Some("have"),
        "have" => Some("has"),
        "does" => Some("do"),
        "do" => Some("does"),
        "am" => Some("are"),
        _ => None,
    }
}

/// Flip 3rd-person -s on a main verb.
fn flip_verb_agreement(word: &str) -> String {
    let low = word.to_lowercase();
    if low.ends_with("ies") {
        // carries → carry
        format!("{}y", &low[..low.len() - 3])
    } else if low.ends_with("es") && (low.ends_with("shes") || low.ends_with("ches") || low.ends_with("xes") || low.ends_with("zes") || low.ends_with("sses")) {
        // washes → wash
        low[..low.len() - 2].to_string()
    } else if low.ends_with('s') && !low.ends_with("ss") {
        // runs → run
        low[..low.len() - 1].to_string()
    } else {
        // run → runs (add -s)
        if low.ends_with("sh") || low.ends_with("ch") || low.ends_with('x')
            || low.ends_with('z') || low.ends_with('s')
        {
            format!("{}es", low)
        } else if low.ends_with('y')
            && low.len() > 2
            && !matches!(low.as_bytes()[low.len() - 2], b'a' | b'e' | b'i' | b'o' | b'u')
        {
            format!("{}ies", &low[..low.len() - 1])
        } else {
            format!("{}s", low)
        }
    }
}

/// Find subject-verb pairs in the sentence.
fn find_subject_verb_pairs(sentence: &Sentence) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for edge in &sentence.dependency_edges {
        if edge.relation == DependencyRelation::Nsubj {
            let verb_idx = edge.head_index;
            let subj_idx = edge.dependent_index;
            if sentence.tokens.get(verb_idx).map_or(false, |t| {
                t.pos_tag == Some(PosTag::Verb) || t.pos_tag == Some(PosTag::Aux)
            }) {
                pairs.push((subj_idx, verb_idx));
            }
        }
    }
    pairs
}

pub struct AgreementPerturbationTransform;

impl AgreementPerturbationTransform {
    pub fn new() -> Self {
        Self
    }

    /// Fix the agreement (inverse of perturbation).
    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Applying the same perturbation again should flip back (it's a toggle)
        self.apply(sentence)
    }
}

impl BaseTransformation for AgreementPerturbationTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::AgreementPerturbation
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        !find_subject_verb_pairs(sentence).is_empty()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        let pairs = find_subject_verb_pairs(sentence);
        if pairs.is_empty() {
            return Err(TransformationError::PreconditionNotMet(
                "No subject-verb pair found".into(),
            ));
        }

        let mut new_tokens: Vec<Token> = sentence.tokens.clone();
        let mut modified = Vec::new();

        // Perturb the first subject-verb pair
        let (_subj_idx, verb_idx) = pairs[0];
        let verb_tok = &sentence.tokens[verb_idx];

        if let Some(flipped) = flip_copula(&verb_tok.text) {
            new_tokens[verb_idx] = Token::new(flipped, verb_idx).with_pos(verb_tok.pos_tag.unwrap_or(PosTag::Verb));
            modified.push((verb_idx, verb_idx + 1));
        } else {
            let flipped = flip_verb_agreement(&verb_tok.text);
            new_tokens[verb_idx] = Token::new(&flipped, verb_idx).with_pos(verb_tok.pos_tag.unwrap_or(PosTag::Verb));
            modified.push((verb_idx, verb_idx + 1));
        }

        // Also check for auxiliaries of this verb and flip them
        for edge in &sentence.dependency_edges {
            if edge.head_index == verb_idx && edge.relation == DependencyRelation::Aux {
                let aux_idx = edge.dependent_index;
                if let Some(aux_tok) = sentence.tokens.get(aux_idx) {
                    if let Some(flipped) = flip_copula(&aux_tok.text) {
                        new_tokens[aux_idx] = Token::new(flipped, aux_idx).with_pos(PosTag::Aux);
                        modified.push((aux_idx, aux_idx + 1));
                    }
                }
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            modified,
            "Perturbed subject-verb agreement".to_string(),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        false // introduces grammatical errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn copula_sentence() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("cat", 1).with_pos(PosTag::Noun),
            Token::new("is", 2).with_pos(PosTag::Verb),
            Token::new("happy", 3).with_pos(PosTag::Adj),
            Token::new(".", 4).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 3, DependencyRelation::Amod),
            DependencyEdge::new(2, 4, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The cat is happy.");
        s.dependency_edges = edges;
        s
    }

    fn main_verb_sentence() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("cat", 1).with_pos(PosTag::Noun),
            Token::new("runs", 2).with_pos(PosTag::Verb),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 3, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The cat runs.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = AgreementPerturbationTransform::new();
        assert_eq!(t.kind(), TransformationKind::AgreementPerturbation);
    }

    #[test]
    fn test_is_applicable() {
        let t = AgreementPerturbationTransform::new();
        assert!(t.is_applicable(&copula_sentence()));
        assert!(t.is_applicable(&main_verb_sentence()));
    }

    #[test]
    fn test_flip_copula_is_to_are() {
        let t = AgreementPerturbationTransform::new();
        let result = t.apply(&copula_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("are"));
    }

    #[test]
    fn test_flip_main_verb() {
        let t = AgreementPerturbationTransform::new();
        let result = t.apply(&main_verb_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("run") && !text.contains("runs"));
    }

    #[test]
    fn test_not_meaning_preserving() {
        let t = AgreementPerturbationTransform::new();
        assert!(!t.is_meaning_preserving());
    }

    #[test]
    fn test_flip_verb_agreement_helpers() {
        assert_eq!(flip_copula("is"), Some("are"));
        assert_eq!(flip_copula("was"), Some("were"));
        assert_eq!(flip_copula("has"), Some("have"));
        assert_eq!(flip_copula("do"), Some("does"));
    }

    #[test]
    fn test_flip_regular_verb() {
        assert_eq!(flip_verb_agreement("runs"), "run");
        assert_eq!(flip_verb_agreement("run"), "runs");
        assert_eq!(flip_verb_agreement("carries"), "carry");
    }

    #[test]
    fn test_not_applicable_no_subj() {
        let t = AgreementPerturbationTransform::new();
        let tokens = vec![Token::new("Run", 0).with_pos(PosTag::Verb)];
        let s = Sentence::from_tokens(tokens, "Run");
        assert!(!t.is_applicable(&s));
    }
}
