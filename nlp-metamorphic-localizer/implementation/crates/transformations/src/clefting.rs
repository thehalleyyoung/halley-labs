//! Cleft-sentence transformation: "John ate the cake" → "It was John who ate
//! the cake" (subject cleft) or "It was the cake that John ate" (object cleft).

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    capitalize, decapitalize, rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// Whether a token likely refers to a human/animate entity.
fn is_animate(token: &Token) -> bool {
    let low = token.text.to_lowercase();
    // Pronouns referring to people
    let animate_pronouns = [
        "he", "she", "they", "who", "whom", "i", "we", "you",
    ];
    if animate_pronouns.contains(&low.as_str()) {
        return true;
    }
    // Proper nouns are often animate (heuristic)
    if token.text.chars().next().map_or(false, |c| c.is_uppercase())
        && token.pos_tag == Some(PosTag::Noun)
    {
        return true;
    }
    false
}

/// Select the relative pronoun for a cleft based on animacy.
fn relative_pronoun(animate: bool, is_subject: bool) -> &'static str {
    if is_subject {
        if animate { "who" } else { "that" }
    } else {
        "that"
    }
}

pub struct CleftTransform;

impl CleftTransform {
    pub fn new() -> Self {
        Self
    }

    /// Attempt to de-cleft a sentence (inverse).
    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Detect cleft: starts with "It" + copula
        if sentence.tokens.len() < 5 {
            return Err(TransformationError::InverseNotPossible("sentence too short".into()));
        }
        let first = sentence.tokens[0].text.to_lowercase();
        if first != "it" {
            return Err(TransformationError::InverseNotPossible("not a cleft sentence".into()));
        }
        let second = sentence.tokens[1].text.to_lowercase();
        if second != "was" && second != "is" && second != "were" && second != "are" {
            return Err(TransformationError::InverseNotPossible("no copula after 'it'".into()));
        }

        // Find the relative pronoun position
        let rel_pos = sentence.tokens.iter().position(|t| {
            let low = t.text.to_lowercase();
            low == "who" || low == "that" || low == "which"
        });
        let rel_idx = rel_pos.ok_or_else(|| {
            TransformationError::InverseNotPossible("no relative pronoun found".into())
        })?;

        // Focus = tokens[2..rel_idx], clause = tokens[rel_idx+1..end-punct]
        let focus_tokens: Vec<Token> = sentence.tokens[2..rel_idx].to_vec();
        let end = if sentence.tokens.last().map_or(false, |t| t.pos_tag == Some(PosTag::Punct)) {
            sentence.tokens.len() - 1
        } else {
            sentence.tokens.len()
        };
        let clause_tokens: Vec<Token> = sentence.tokens[rel_idx + 1..end].to_vec();

        // Try to determine if subject or object cleft
        // Subject cleft: focus is the subject → focus + clause
        // Object cleft: focus is the object → clause_subj + verb + focus
        let is_subject_cleft = clause_tokens.first().map_or(true, |t| {
            t.pos_tag == Some(PosTag::Verb) || t.pos_tag == Some(PosTag::Aux)
        });

        let mut new_tokens: Vec<Token> = Vec::new();
        if is_subject_cleft {
            for t in &focus_tokens {
                new_tokens.push(t.clone());
            }
            for t in &clause_tokens {
                new_tokens.push(t.clone());
            }
        } else {
            // Object cleft: find where the verb is in clause, split subj/verb
            let verb_pos = clause_tokens.iter().position(|t| {
                t.pos_tag == Some(PosTag::Verb)
            }).unwrap_or(0);
            // subject = clause[..=verb_pos-1?], verb = clause[verb_pos], rest
            for t in &clause_tokens[..=verb_pos] {
                new_tokens.push(t.clone());
            }
            for t in &focus_tokens {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
            for t in &clause_tokens[verb_pos + 1..] {
                new_tokens.push(t.clone());
            }
        }

        // Capitalize first, add punct
        if let Some(first) = new_tokens.first_mut() {
            first.text = capitalize(&first.text);
        }
        if let Some(punct) = sentence.tokens.last() {
            if punct.pos_tag == Some(PosTag::Punct) {
                new_tokens.push(punct.clone());
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, sentence.tokens.len())],
            "De-clefted sentence to canonical form",
        ))
    }
}

impl BaseTransformation for CleftTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::Clefting
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        if sentence.tokens.len() < 3 {
            return false;
        }
        let root = match sentence.root_index() {
            Some(r) => r,
            None => return false,
        };
        // Need a subject or object to serve as focus
        let has_subj = sentence.dependency_edges.iter().any(|e| {
            e.head_index == root && e.relation == DependencyRelation::Nsubj
        });
        let verb_ok = sentence.tokens.get(root).map_or(false, |t| {
            t.pos_tag == Some(PosTag::Verb)
        });
        // Don't cleft sentences that are already clefts
        let already_cleft = sentence.tokens.first().map_or(false, |t| {
            t.text.to_lowercase() == "it"
        }) && sentence.tokens.get(1).map_or(false, |t| {
            let l = t.text.to_lowercase();
            l == "is" || l == "was"
        });
        has_subj && verb_ok && !already_cleft
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "Need declarative sentence with clear subject".into(),
            ));
        }

        let root_idx = sentence.root_index().unwrap();
        let subj_edge = sentence.dependency_edges.iter().find(|e| {
            e.head_index == root_idx && e.relation == DependencyRelation::Nsubj
        }).unwrap();
        let subj_idx = subj_edge.dependent_index;

        let obj_edge = sentence.dependency_edges.iter().find(|e| {
            e.head_index == root_idx && e.relation == DependencyRelation::Dobj
        });

        // Decide: subject cleft by default, object cleft if object exists and subject is a pronoun
        let subj_token = &sentence.tokens[subj_idx];
        let do_object_cleft = obj_edge.is_some()
            && subj_token.pos_tag == Some(PosTag::Pron);

        if do_object_cleft {
            let obj_idx = obj_edge.unwrap().dependent_index;
            self.apply_object_cleft(sentence, root_idx, subj_idx, obj_idx)
        } else {
            self.apply_subject_cleft(sentence, root_idx, subj_idx)
        }
    }

    fn is_meaning_preserving(&self) -> bool {
        true
    }
}

impl CleftTransform {
    fn apply_subject_cleft(
        &self,
        sentence: &Sentence,
        root_idx: usize,
        subj_idx: usize,
    ) -> Result<TransformationResult, TransformationError> {
        let subj_span = subtree_indices(sentence, subj_idx);
        let subj_token = &sentence.tokens[subj_idx];
        let animate = is_animate(subj_token);
        let rel_pron = relative_pronoun(animate, true);

        // Determine tense of copula from the main verb
        let verb = &sentence.tokens[root_idx];
        let copula = if verb.text.to_lowercase().ends_with("ed")
            || verb.features.get("Tense").map_or(false, |v| v == "Past")
        {
            "was"
        } else {
            "is"
        };

        let mut new_tokens: Vec<Token> = Vec::new();

        // "It" + copula
        new_tokens.push(Token::new("It", 0).with_pos(PosTag::Pron));
        new_tokens.push(Token::new(copula, 0).with_pos(PosTag::Aux));

        // Focus (subject span)
        for &i in &subj_span {
            if let Some(t) = sentence.tokens.get(i) {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
        }

        // Relative pronoun
        new_tokens.push(Token::new(rel_pron, 0).with_pos(PosTag::Pron));

        // Rest of the clause (everything except subject span and punct)
        for (i, tok) in sentence.tokens.iter().enumerate() {
            if !subj_span.contains(&i) && tok.pos_tag != Some(PosTag::Punct) {
                new_tokens.push(tok.clone());
            }
        }

        // Punct
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
            format!("Subject cleft with '{}'", rel_pron),
        ))
    }

    fn apply_object_cleft(
        &self,
        sentence: &Sentence,
        root_idx: usize,
        subj_idx: usize,
        obj_idx: usize,
    ) -> Result<TransformationResult, TransformationError> {
        let obj_span = subtree_indices(sentence, obj_idx);
        let subj_span = subtree_indices(sentence, subj_idx);
        let obj_token = &sentence.tokens[obj_idx];
        let animate = is_animate(obj_token);
        let rel_pron = relative_pronoun(animate, false);

        let verb = &sentence.tokens[root_idx];
        let copula = if verb.text.to_lowercase().ends_with("ed")
            || verb.features.get("Tense").map_or(false, |v| v == "Past")
        {
            "was"
        } else {
            "is"
        };

        let mut new_tokens: Vec<Token> = Vec::new();

        // "It" + copula
        new_tokens.push(Token::new("It", 0).with_pos(PosTag::Pron));
        new_tokens.push(Token::new(copula, 0).with_pos(PosTag::Aux));

        // Focus (object span)
        for &i in &obj_span {
            if let Some(t) = sentence.tokens.get(i) {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
        }

        // Relative pronoun
        new_tokens.push(Token::new(rel_pron, 0).with_pos(PosTag::Pron));

        // Subject span
        for &i in &subj_span {
            if let Some(t) = sentence.tokens.get(i) {
                new_tokens.push(t.clone());
            }
        }

        // Verb and remaining (skip subject, object, punct)
        for (i, tok) in sentence.tokens.iter().enumerate() {
            if !subj_span.contains(&i) && !obj_span.contains(&i) && tok.pos_tag != Some(PosTag::Punct) {
                new_tokens.push(tok.clone());
            }
        }

        // Punct
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
            format!("Object cleft with '{}'", rel_pron),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn simple_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("ate", 1).with_pos(PosTag::Verb).with_lemma("eat"),
            Token::new("the", 2).with_pos(PosTag::Det),
            Token::new("cake", 3).with_pos(PosTag::Noun),
            Token::new(".", 4).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 3, DependencyRelation::Dobj),
            DependencyEdge::new(3, 2, DependencyRelation::Det),
            DependencyEdge::new(1, 4, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John ate the cake.");
        s.dependency_edges = edges;
        s
    }

    fn pronoun_subject_sentence() -> Sentence {
        let tokens = vec![
            Token::new("He", 0).with_pos(PosTag::Pron),
            Token::new("ate", 1).with_pos(PosTag::Verb).with_lemma("eat"),
            Token::new("the", 2).with_pos(PosTag::Det),
            Token::new("cake", 3).with_pos(PosTag::Noun),
            Token::new(".", 4).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 3, DependencyRelation::Dobj),
            DependencyEdge::new(3, 2, DependencyRelation::Det),
            DependencyEdge::new(1, 4, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "He ate the cake.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = CleftTransform::new();
        assert_eq!(t.kind(), TransformationKind::Clefting);
    }

    #[test]
    fn test_is_applicable() {
        let t = CleftTransform::new();
        assert!(t.is_applicable(&simple_sentence()));
    }

    #[test]
    fn test_subject_cleft() {
        let t = CleftTransform::new();
        let result = t.apply(&simple_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.starts_with("It"));
        assert!(text.contains("who") || text.contains("that"));
    }

    #[test]
    fn test_object_cleft() {
        let t = CleftTransform::new();
        let result = t.apply(&pronoun_subject_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.starts_with("It"));
        assert!(text.contains("that"));
    }

    #[test]
    fn test_meaning_preserving() {
        let t = CleftTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_too_short() {
        let t = CleftTransform::new();
        let tokens = vec![Token::new("Go", 0).with_pos(PosTag::Verb)];
        let s = Sentence::from_tokens(tokens, "Go");
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_inverse() {
        let t = CleftTransform::new();
        let tokens = vec![
            Token::new("It", 0).with_pos(PosTag::Pron),
            Token::new("was", 1).with_pos(PosTag::Aux),
            Token::new("John", 2).with_pos(PosTag::Noun),
            Token::new("who", 3).with_pos(PosTag::Pron),
            Token::new("ate", 4).with_pos(PosTag::Verb),
            Token::new("the", 5).with_pos(PosTag::Det),
            Token::new("cake", 6).with_pos(PosTag::Noun),
            Token::new(".", 7).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, "It was John who ate the cake.");
        let inv = t.inverse(&s).unwrap();
        assert!(inv.success);
    }

    #[test]
    fn test_animate_detection() {
        assert!(is_animate(&Token::new("John", 0).with_pos(PosTag::Noun)));
        assert!(is_animate(&Token::new("she", 0).with_pos(PosTag::Pron)));
        assert!(!is_animate(&Token::new("cake", 0).with_pos(PosTag::Noun)));
    }
}
