//! Relative clause insertion and deletion transformations.

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Whether a token likely refers to a human/animate entity.
fn is_animate(token: &Token) -> bool {
    let low = token.text.to_lowercase();
    let animate = [
        "he", "she", "they", "who", "whom", "i", "we", "you",
        "man", "woman", "boy", "girl", "person", "child", "student",
        "teacher", "doctor", "friend", "brother", "sister",
    ];
    if animate.contains(&low.as_str()) {
        return true;
    }
    token.text.chars().next().map_or(false, |c| c.is_uppercase())
        && token.pos_tag == Some(PosTag::Noun)
}

/// Select relative pronoun based on animacy.
fn select_relative_pronoun(animate: bool) -> &'static str {
    if animate { "who" } else { "which" }
}

/// Generic predicates for relative clause generation.
fn generic_predicate(animate: bool) -> (&'static str, PosTag) {
    if animate {
        ("arrived", PosTag::Verb)
    } else {
        ("appeared", PosTag::Verb)
    }
}

/// Find noun heads that can take a relative clause modifier.
fn find_modifiable_nouns(sentence: &Sentence) -> Vec<usize> {
    let mut candidates = Vec::new();
    for (i, tok) in sentence.tokens.iter().enumerate() {
        if tok.pos_tag != Some(PosTag::Noun) && tok.pos_tag != Some(PosTag::Pron) {
            continue;
        }
        // Skip if it already has a relative clause
        let already_has_relcl = sentence.dependency_edges.iter().any(|e| {
            e.head_index == i && e.relation == DependencyRelation::Relcl
        });
        if !already_has_relcl {
            candidates.push(i);
        }
    }
    candidates
}

/// Find the head noun and span of an existing relative clause.
fn find_relative_clause(sentence: &Sentence) -> Option<(usize, usize)> {
    sentence.dependency_edges.iter().find(|e| {
        e.relation == DependencyRelation::Relcl
    }).map(|e| (e.head_index, e.dependent_index))
}

// ── RelativeClauseInsertTransform ───────────────────────────────────────────

pub struct RelativeClauseInsertTransform;

impl RelativeClauseInsertTransform {
    pub fn new() -> Self {
        Self
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Delegate to deletion
        let del = RelativeClauseDeletionTransform::new();
        del.apply(sentence)
    }
}

impl BaseTransformation for RelativeClauseInsertTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::RelativeClauseInsertion
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        if sentence.tokens.len() < 3 {
            return false;
        }
        !find_modifiable_nouns(sentence).is_empty()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "No noun phrase available for relative clause insertion".into(),
            ));
        }

        let candidates = find_modifiable_nouns(sentence);
        // Prefer the subject noun
        let root = sentence.root_index().unwrap_or(0);
        let subj_idx = sentence.dependency_edges.iter().find(|e| {
            e.head_index == root && e.relation == DependencyRelation::Nsubj
        }).map(|e| e.dependent_index);

        let target = if let Some(si) = subj_idx {
            if candidates.contains(&si) { si } else { candidates[0] }
        } else {
            candidates[0]
        };

        let target_token = &sentence.tokens[target];
        let animate = is_animate(target_token);
        let rel_pron = select_relative_pronoun(animate);
        let (predicate, pred_pos) = generic_predicate(animate);

        // Find the end of the target noun's span
        let target_span = subtree_indices(sentence, target);
        let insert_after = *target_span.last().unwrap_or(&target);

        let mut new_tokens: Vec<Token> = Vec::new();
        for (i, tok) in sentence.tokens.iter().enumerate() {
            new_tokens.push(tok.clone());
            if i == insert_after {
                // Insert: ", who/which <predicate>,"
                // Only add commas if the noun isn't at the end
                let is_sentence_end = i + 1 >= sentence.tokens.len()
                    || sentence.tokens[i + 1].pos_tag == Some(PosTag::Punct);

                if !is_sentence_end {
                    new_tokens.push(Token::new(",", 0).with_pos(PosTag::Punct));
                }
                new_tokens.push(Token::new(rel_pron, 0).with_pos(PosTag::Pron));
                new_tokens.push(Token::new(predicate, 0).with_pos(pred_pos));
                if !is_sentence_end {
                    new_tokens.push(Token::new(",", 0).with_pos(PosTag::Punct));
                }
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(insert_after + 1, insert_after + 4)],
            format!(
                "Inserted relative clause '{} {}' after '{}'",
                rel_pron, predicate, target_token.text
            ),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        false // adding information changes semantics
    }
}

// ── RelativeClauseDeletionTransform ─────────────────────────────────────────

pub struct RelativeClauseDeletionTransform;

impl RelativeClauseDeletionTransform {
    pub fn new() -> Self {
        Self
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Delegate to insertion
        let ins = RelativeClauseInsertTransform::new();
        ins.apply(sentence)
    }
}

impl BaseTransformation for RelativeClauseDeletionTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::RelativeClauseDeletion
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        find_relative_clause(sentence).is_some()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "No relative clause found in sentence".into(),
            ));
        }

        let (head_idx, relcl_idx) = find_relative_clause(sentence).unwrap();
        let relcl_span = subtree_indices(sentence, relcl_idx);

        // Also remove the relative pronoun if it precedes the relcl head
        let mut remove_set: Vec<usize> = relcl_span.clone();

        // Find relative pronoun (token right before the relcl verb, or dependent of relcl)
        for &idx in &relcl_span {
            if idx > 0 {
                let prev = &sentence.tokens[idx - 1];
                let low = prev.text.to_lowercase();
                if (low == "who" || low == "which" || low == "that" || low == "whom")
                    && !remove_set.contains(&(idx - 1))
                {
                    remove_set.push(idx - 1);
                }
            }
        }

        // Also remove surrounding commas
        let min_rc = *remove_set.iter().min().unwrap_or(&0);
        let max_rc = *remove_set.iter().max().unwrap_or(&0);
        if min_rc > 0 && sentence.tokens[min_rc - 1].text == "," {
            remove_set.push(min_rc - 1);
        }
        if max_rc + 1 < sentence.tokens.len() && sentence.tokens[max_rc + 1].text == "," {
            remove_set.push(max_rc + 1);
        }

        remove_set.sort();
        remove_set.dedup();

        let mut new_tokens: Vec<Token> = Vec::new();
        for (i, tok) in sentence.tokens.iter().enumerate() {
            if !remove_set.contains(&i) {
                new_tokens.push(tok.clone());
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(min_rc, max_rc + 1)],
            format!("Deleted relative clause modifying token at index {}", head_idx),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        false // removing a relative clause loses information
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn simple_sentence() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("man", 1).with_pos(PosTag::Noun),
            Token::new("ran", 2).with_pos(PosTag::Verb),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 3, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The man ran.");
        s.dependency_edges = edges;
        s
    }

    fn sentence_with_relcl() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("man", 1).with_pos(PosTag::Noun),
            Token::new(",", 2).with_pos(PosTag::Punct),
            Token::new("who", 3).with_pos(PosTag::Pron),
            Token::new("arrived", 4).with_pos(PosTag::Verb),
            Token::new(",", 5).with_pos(PosTag::Punct),
            Token::new("ran", 6).with_pos(PosTag::Verb),
            Token::new(".", 7).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(6, 6, DependencyRelation::Root),
            DependencyEdge::new(6, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(1, 4, DependencyRelation::Relcl),
            DependencyEdge::new(4, 3, DependencyRelation::Nsubj),
            DependencyEdge::new(6, 7, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The man, who arrived, ran.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_insert_new() {
        let t = RelativeClauseInsertTransform::new();
        assert_eq!(t.kind(), TransformationKind::RelativeClauseInsertion);
    }

    #[test]
    fn test_insert_applicable() {
        let t = RelativeClauseInsertTransform::new();
        assert!(t.is_applicable(&simple_sentence()));
    }

    #[test]
    fn test_insert_apply() {
        let t = RelativeClauseInsertTransform::new();
        let result = t.apply(&simple_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.contains("who") || text.contains("which"));
    }

    #[test]
    fn test_delete_new() {
        let t = RelativeClauseDeletionTransform::new();
        assert_eq!(t.kind(), TransformationKind::RelativeClauseDeletion);
    }

    #[test]
    fn test_delete_applicable() {
        let t = RelativeClauseDeletionTransform::new();
        assert!(!t.is_applicable(&simple_sentence()));
        assert!(t.is_applicable(&sentence_with_relcl()));
    }

    #[test]
    fn test_delete_apply() {
        let t = RelativeClauseDeletionTransform::new();
        let result = t.apply(&sentence_with_relcl()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(!text.contains("who"));
        assert!(!text.contains("arrived"));
    }

    #[test]
    fn test_insert_not_meaning_preserving() {
        let t = RelativeClauseInsertTransform::new();
        assert!(!t.is_meaning_preserving());
    }

    #[test]
    fn test_delete_not_meaning_preserving() {
        let t = RelativeClauseDeletionTransform::new();
        assert!(!t.is_meaning_preserving());
    }

    #[test]
    fn test_insert_with_inanimate() {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("book", 1).with_pos(PosTag::Noun),
            Token::new("fell", 2).with_pos(PosTag::Verb),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 3, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The book fell.");
        s.dependency_edges = edges;

        let t = RelativeClauseInsertTransform::new();
        let result = t.apply(&s).unwrap();
        let text = result.transformed.surface_text();
        assert!(text.contains("which"));
    }
}
