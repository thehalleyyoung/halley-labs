//! There-insertion: "A book is on the table" → "There is a book on the table".

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    capitalize, decapitalize, rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// Copular / unaccusative verbs that license there-insertion.
fn is_there_verb(word: &str) -> bool {
    let low = word.to_lowercase();
    matches!(
        low.as_str(),
        "is" | "are" | "was" | "were" | "be"
            | "exist" | "exists" | "existed"
            | "appear" | "appears" | "appeared"
            | "remain" | "remains" | "remained"
            | "seem" | "seems" | "seemed"
            | "lie" | "lies" | "lay"
            | "stand" | "stands" | "stood"
            | "sit" | "sits" | "sat"
    )
}

/// Check if a determiner is indefinite.
fn is_indefinite(word: &str) -> bool {
    let low = word.to_lowercase();
    matches!(low.as_str(), "a" | "an" | "some" | "many" | "several" | "few" | "no")
}

/// Find the subject and check if it has an indefinite determiner.
fn has_indefinite_subject(sentence: &Sentence) -> Option<(usize, usize)> {
    let root = sentence.root_index()?;
    let subj_edge = sentence.dependency_edges.iter().find(|e| {
        e.head_index == root && e.relation == DependencyRelation::Nsubj
    })?;
    let subj_idx = subj_edge.dependent_index;

    // Find determiner of subject
    let det_idx = sentence.dependency_edges.iter().find(|e| {
        e.head_index == subj_idx && e.relation == DependencyRelation::Det
    }).map(|e| e.dependent_index);

    if let Some(di) = det_idx {
        if sentence.tokens.get(di).map_or(false, |t| is_indefinite(&t.text)) {
            return Some((subj_idx, di));
        }
    }

    None
}

pub struct ThereInsertionTransform;

impl ThereInsertionTransform {
    pub fn new() -> Self {
        Self
    }

    /// Remove there-insertion (inverse).
    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Check if sentence starts with "there"
        if sentence.tokens.first().map_or(true, |t| t.text.to_lowercase() != "there") {
            return Err(TransformationError::InverseNotPossible(
                "sentence does not start with 'there'".into(),
            ));
        }

        if sentence.tokens.len() < 4 {
            return Err(TransformationError::InverseNotPossible("sentence too short".into()));
        }

        // Structure: "There" + verb + NP + rest
        // Find the verb (token after "there")
        let verb_idx = 1;
        let verb_tok = sentence.tokens.get(verb_idx)
            .ok_or(TransformationError::InverseNotPossible("no verb after there".into()))?;

        if !is_there_verb(&verb_tok.text) {
            return Err(TransformationError::InverseNotPossible(
                "token after 'there' is not a copular/unaccusative verb".into(),
            ));
        }

        // Everything after the verb until the next PP/punct is the subject NP
        let mut new_tokens: Vec<Token> = Vec::new();

        // Skip "there", keep everything else; put NP before verb
        // Simple heuristic: tokens[2..] until we hit a preposition or punct
        let mut np_end = sentence.tokens.len();
        for i in 2..sentence.tokens.len() {
            if sentence.tokens[i].pos_tag == Some(PosTag::Prep)
                || sentence.tokens[i].pos_tag == Some(PosTag::Punct)
            {
                np_end = i;
                break;
            }
        }

        // NP tokens (capitalize first)
        for i in 2..np_end {
            let mut tok = sentence.tokens[i].clone();
            if i == 2 {
                tok.text = capitalize(&tok.text);
            }
            new_tokens.push(tok);
        }

        // Verb
        new_tokens.push(verb_tok.clone());

        // Rest (PP, etc.)
        for i in np_end..sentence.tokens.len() {
            new_tokens.push(sentence.tokens[i].clone());
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, sentence.tokens.len())],
            "Removed there-insertion".to_string(),
        ))
    }
}

impl BaseTransformation for ThereInsertionTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::ThereInsertion
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        if sentence.tokens.len() < 3 {
            return false;
        }
        // Already has there-insertion?
        if sentence.tokens.first().map_or(false, |t| t.text.to_lowercase() == "there") {
            return false;
        }
        // Need indefinite subject + copular/unaccusative verb
        let root = match sentence.root_index() {
            Some(r) => r,
            None => return false,
        };
        let verb = match sentence.tokens.get(root) {
            Some(t) => t,
            None => return false,
        };
        if !is_there_verb(&verb.text) {
            return false;
        }
        has_indefinite_subject(sentence).is_some()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "Need indefinite subject + copular/unaccusative verb".into(),
            ));
        }

        let root_idx = sentence.root_index().unwrap();
        let (subj_idx, _det_idx) = has_indefinite_subject(sentence).unwrap();
        let subj_span = subtree_indices(sentence, subj_idx);

        let verb_token = &sentence.tokens[root_idx];

        let mut new_tokens: Vec<Token> = Vec::new();

        // "There" + verb
        new_tokens.push(Token::new("There", 0).with_pos(PosTag::Pron));
        new_tokens.push(verb_token.clone());

        // Subject NP (moved after verb)
        for &i in &subj_span {
            if let Some(t) = sentence.tokens.get(i) {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
        }

        // Remaining tokens (not subject, not verb, not punct)
        for (i, tok) in sentence.tokens.iter().enumerate() {
            if i == root_idx || subj_span.contains(&i) || tok.pos_tag == Some(PosTag::Punct) {
                continue;
            }
            new_tokens.push(tok.clone());
        }

        // Final punct
        if let Some(last) = sentence.tokens.last() {
            if last.pos_tag == Some(PosTag::Punct) {
                new_tokens.push(last.clone());
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, sentence.tokens.len())],
            "Inserted 'there' as expletive subject".to_string(),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn there_eligible() -> Sentence {
        // "A book is on the table."
        let tokens = vec![
            Token::new("A", 0).with_pos(PosTag::Det),
            Token::new("book", 1).with_pos(PosTag::Noun),
            Token::new("is", 2).with_pos(PosTag::Verb),
            Token::new("on", 3).with_pos(PosTag::Prep),
            Token::new("the", 4).with_pos(PosTag::Det),
            Token::new("table", 5).with_pos(PosTag::Noun),
            Token::new(".", 6).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 3, DependencyRelation::Prep),
            DependencyEdge::new(3, 5, DependencyRelation::Pobj),
            DependencyEdge::new(5, 4, DependencyRelation::Det),
            DependencyEdge::new(2, 6, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "A book is on the table.");
        s.dependency_edges = edges;
        s
    }

    fn definite_subject() -> Sentence {
        // "The book is on the table."
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("book", 1).with_pos(PosTag::Noun),
            Token::new("is", 2).with_pos(PosTag::Verb),
            Token::new("on", 3).with_pos(PosTag::Prep),
            Token::new("the", 4).with_pos(PosTag::Det),
            Token::new("table", 5).with_pos(PosTag::Noun),
            Token::new(".", 6).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 3, DependencyRelation::Prep),
            DependencyEdge::new(3, 5, DependencyRelation::Pobj),
            DependencyEdge::new(5, 4, DependencyRelation::Det),
            DependencyEdge::new(2, 6, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The book is on the table.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = ThereInsertionTransform::new();
        assert_eq!(t.kind(), TransformationKind::ThereInsertion);
    }

    #[test]
    fn test_is_applicable() {
        let t = ThereInsertionTransform::new();
        assert!(t.is_applicable(&there_eligible()));
    }

    #[test]
    fn test_not_applicable_definite() {
        let t = ThereInsertionTransform::new();
        assert!(!t.is_applicable(&definite_subject()));
    }

    #[test]
    fn test_apply() {
        let t = ThereInsertionTransform::new();
        let result = t.apply(&there_eligible()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.starts_with("There"));
        assert!(text.to_lowercase().contains("is"));
        assert!(text.to_lowercase().contains("book"));
    }

    #[test]
    fn test_meaning_preserving() {
        let t = ThereInsertionTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_inverse() {
        let t = ThereInsertionTransform::new();
        let tokens = vec![
            Token::new("There", 0).with_pos(PosTag::Pron),
            Token::new("is", 1).with_pos(PosTag::Verb),
            Token::new("a", 2).with_pos(PosTag::Det),
            Token::new("book", 3).with_pos(PosTag::Noun),
            Token::new("on", 4).with_pos(PosTag::Prep),
            Token::new("the", 5).with_pos(PosTag::Det),
            Token::new("table", 6).with_pos(PosTag::Noun),
            Token::new(".", 7).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, "There is a book on the table.");
        let inv = t.inverse(&s).unwrap();
        assert!(inv.success);
        let text = inv.transformed.surface_text();
        assert!(!text.starts_with("There"));
    }

    #[test]
    fn test_not_applicable_already_there() {
        let t = ThereInsertionTransform::new();
        let tokens = vec![
            Token::new("There", 0).with_pos(PosTag::Pron),
            Token::new("is", 1).with_pos(PosTag::Verb),
            Token::new("a", 2).with_pos(PosTag::Det),
            Token::new("cat", 3).with_pos(PosTag::Noun),
        ];
        let s = Sentence::from_tokens(tokens, "There is a cat");
        assert!(!t.is_applicable(&s));
    }
}
