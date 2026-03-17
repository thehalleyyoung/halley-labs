//! Adverb repositioning: "Quickly John ran" ↔ "John quickly ran" ↔ "John ran
//! quickly".

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    capitalize, decapitalize, rebuild_sentence, reindex_tokens,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// An adverb site: the adverb index and its head verb index.
#[derive(Debug)]
struct AdverbSite {
    adv_idx: usize,
    head_idx: usize,
}

/// Find adverbs and their head verbs.
fn find_adverb_sites(sentence: &Sentence) -> Vec<AdverbSite> {
    let mut sites = Vec::new();
    for edge in &sentence.dependency_edges {
        if edge.relation == DependencyRelation::Advmod {
            let adv = edge.dependent_index;
            let head = edge.head_index;
            if sentence.tokens.get(adv).map_or(false, |t| t.pos_tag == Some(PosTag::Adv)) {
                sites.push(AdverbSite { adv_idx: adv, head_idx: head });
            }
        }
    }
    sites
}

/// Determine adverb position: initial, medial, or final.
fn adverb_position(sentence: &Sentence, adv_idx: usize, head_idx: usize) -> &'static str {
    let non_punct_indices: Vec<usize> = sentence
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.pos_tag != Some(PosTag::Punct))
        .map(|(i, _)| i)
        .collect();

    if non_punct_indices.first() == Some(&adv_idx) {
        "initial"
    } else if non_punct_indices.last() == Some(&adv_idx) {
        "final"
    } else {
        "medial"
    }
}

/// Choose next position: initial → medial → final → initial.
fn next_position(current: &str) -> &'static str {
    match current {
        "initial" => "medial",
        "medial" => "final",
        "final" => "initial",
        _ => "medial",
    }
}

pub struct AdverbRepositionTransform;

impl AdverbRepositionTransform {
    pub fn new() -> Self {
        Self
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Applying again cycles through positions
        self.apply(sentence)
    }
}

impl BaseTransformation for AdverbRepositionTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::AdverbRepositioning
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        if sentence.tokens.len() < 3 {
            return false;
        }
        !find_adverb_sites(sentence).is_empty()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        let sites = find_adverb_sites(sentence);
        if sites.is_empty() {
            return Err(TransformationError::PreconditionNotMet(
                "No adverb found".into(),
            ));
        }

        let site = &sites[0];
        let current_pos = adverb_position(sentence, site.adv_idx, site.head_idx);
        let target_pos = next_position(current_pos);

        let adv_token = sentence.tokens[site.adv_idx].clone();

        // Build tokens without the adverb
        let mut tokens_without_adv: Vec<Token> = sentence
            .tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != site.adv_idx)
            .map(|(_, t)| t.clone())
            .collect();

        // Find insertion point
        let verb_idx_in_new = tokens_without_adv
            .iter()
            .position(|t| {
                t.pos_tag == Some(PosTag::Verb)
                    || t.pos_tag == Some(PosTag::Aux)
            })
            .unwrap_or(0);

        let mut new_tokens: Vec<Token> = Vec::new();
        match target_pos {
            "initial" => {
                let mut adv = adv_token;
                adv.text = capitalize(&adv.text);
                new_tokens.push(adv);
                for (i, t) in tokens_without_adv.iter().enumerate() {
                    let mut tok = t.clone();
                    if i == 0 {
                        tok.text = decapitalize(&tok.text);
                    }
                    new_tokens.push(tok);
                }
            }
            "medial" => {
                // Insert just before the verb
                for (i, t) in tokens_without_adv.iter().enumerate() {
                    if i == verb_idx_in_new {
                        let mut adv = adv_token.clone();
                        adv.text = decapitalize(&adv.text);
                        new_tokens.push(adv);
                    }
                    let mut tok = t.clone();
                    if i == 0 && current_pos == "initial" {
                        tok.text = capitalize(&tok.text);
                    }
                    new_tokens.push(tok);
                }
            }
            "final" => {
                // Insert before final punctuation
                let has_punct = tokens_without_adv.last().map_or(false, |t| {
                    t.pos_tag == Some(PosTag::Punct)
                });
                for (i, t) in tokens_without_adv.iter().enumerate() {
                    if has_punct && i == tokens_without_adv.len() - 1 {
                        let mut adv = adv_token.clone();
                        adv.text = decapitalize(&adv.text);
                        new_tokens.push(adv);
                    }
                    let mut tok = t.clone();
                    if i == 0 && current_pos == "initial" {
                        tok.text = capitalize(&tok.text);
                    }
                    new_tokens.push(tok);
                }
                if !has_punct {
                    let mut adv = adv_token;
                    adv.text = decapitalize(&adv.text);
                    new_tokens.push(adv);
                }
            }
            _ => return Err(TransformationError::Internal("unknown position".into())),
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, sentence.tokens.len())],
            format!("Moved adverb from {} to {} position", current_pos, target_pos),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        true // adverb repositioning preserves core meaning
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn final_adverb() -> Sentence {
        // "John ran quickly."
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("ran", 1).with_pos(PosTag::Verb),
            Token::new("quickly", 2).with_pos(PosTag::Adv),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Advmod),
            DependencyEdge::new(1, 3, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John ran quickly.");
        s.dependency_edges = edges;
        s
    }

    fn initial_adverb() -> Sentence {
        // "Quickly John ran."
        let tokens = vec![
            Token::new("Quickly", 0).with_pos(PosTag::Adv),
            Token::new("John", 1).with_pos(PosTag::Noun),
            Token::new("ran", 2).with_pos(PosTag::Verb),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(2, 0, DependencyRelation::Advmod),
            DependencyEdge::new(2, 3, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "Quickly John ran.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = AdverbRepositionTransform::new();
        assert_eq!(t.kind(), TransformationKind::AdverbRepositioning);
    }

    #[test]
    fn test_is_applicable() {
        let t = AdverbRepositionTransform::new();
        assert!(t.is_applicable(&final_adverb()));
        assert!(t.is_applicable(&initial_adverb()));
    }

    #[test]
    fn test_final_to_initial() {
        let t = AdverbRepositionTransform::new();
        let result = t.apply(&final_adverb()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        // "final" → "initial"
        let quickly_pos = text.to_lowercase().find("quickly").unwrap_or(usize::MAX);
        let john_pos = text.find("John").unwrap_or(text.find("john").unwrap_or(usize::MAX));
        assert!(quickly_pos < john_pos, "adverb should be initial: {}", text);
    }

    #[test]
    fn test_initial_to_medial() {
        let t = AdverbRepositionTransform::new();
        let result = t.apply(&initial_adverb()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        // "initial" → "medial" (before verb)
        let quickly_pos = text.find("quickly").unwrap_or(usize::MAX);
        let john_pos = text.find("john").unwrap_or(0);
        assert!(quickly_pos > john_pos, "adverb should be medial: {}", text);
    }

    #[test]
    fn test_meaning_preserving() {
        let t = AdverbRepositionTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_no_adverb() {
        let t = AdverbRepositionTransform::new();
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("ran", 1).with_pos(PosTag::Verb),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, "John ran.");
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_not_applicable_short() {
        let t = AdverbRepositionTransform::new();
        let tokens = vec![
            Token::new("Go", 0).with_pos(PosTag::Verb),
        ];
        let s = Sentence::from_tokens(tokens, "Go");
        assert!(!t.is_applicable(&s));
    }
}
