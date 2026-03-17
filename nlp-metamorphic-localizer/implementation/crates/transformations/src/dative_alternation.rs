//! Dative alternation: "John gave Mary a book" ↔ "John gave a book to Mary".

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// Ditransitive verbs that participate in the dative alternation.
fn is_ditransitive(word: &str) -> bool {
    let low = word.to_lowercase();
    let lemmas: &[&str] = &[
        "give", "gives", "gave", "given", "giving",
        "send", "sends", "sent", "sending",
        "show", "shows", "showed", "shown", "showing",
        "tell", "tells", "told", "telling",
        "offer", "offers", "offered", "offering",
        "bring", "brings", "brought", "bringing",
        "hand", "hands", "handed", "handing",
        "pass", "passes", "passed", "passing",
        "teach", "teaches", "taught", "teaching",
        "write", "writes", "wrote", "written", "writing",
        "read", "reads", "reading",
        "sell", "sells", "sold", "selling",
        "buy", "buys", "bought", "buying",
        "lend", "lends", "lent", "lending",
        "pay", "pays", "paid", "paying",
        "throw", "throws", "threw", "thrown", "throwing",
        "toss", "tosses", "tossed", "tossing",
        "award", "awards", "awarded", "awarding",
        "grant", "grants", "granted", "granting",
    ];
    lemmas.contains(&low.as_str())
}

/// Detect whether the sentence uses double-object or PP form.
#[derive(Debug, PartialEq)]
enum DativeForm {
    DoubleObject { dobj_idx: usize, iobj_idx: usize },
    PrepPhrase { dobj_idx: usize, prep_idx: usize, pobj_idx: usize },
}

fn detect_dative_form(sentence: &Sentence) -> Option<(usize, DativeForm)> {
    let root = sentence.root_index()?;
    let verb = sentence.tokens.get(root)?;
    if !is_ditransitive(&verb.text) {
        return None;
    }

    // Check for double object: both dobj and iobj
    let dobj = sentence.dependency_edges.iter().find(|e| {
        e.head_index == root && e.relation == DependencyRelation::Dobj
    });
    let iobj = sentence.dependency_edges.iter().find(|e| {
        e.head_index == root && e.relation == DependencyRelation::Iobj
    });

    if let (Some(dobj_e), Some(iobj_e)) = (dobj, iobj) {
        return Some((root, DativeForm::DoubleObject {
            dobj_idx: dobj_e.dependent_index,
            iobj_idx: iobj_e.dependent_index,
        }));
    }

    // Check for PP form: dobj + prep "to" + pobj
    if let Some(dobj_e) = dobj {
        // Look for a "to" PP
        for edge in &sentence.dependency_edges {
            if edge.head_index == root && edge.relation == DependencyRelation::Prep {
                let prep_idx = edge.dependent_index;
                if sentence.tokens.get(prep_idx).map_or(false, |t| {
                    t.text.to_lowercase() == "to"
                }) {
                    let pobj = sentence.dependency_edges.iter().find(|e| {
                        e.head_index == prep_idx && e.relation == DependencyRelation::Pobj
                    });
                    if let Some(pobj_e) = pobj {
                        return Some((root, DativeForm::PrepPhrase {
                            dobj_idx: dobj_e.dependent_index,
                            prep_idx,
                            pobj_idx: pobj_e.dependent_index,
                        }));
                    }
                }
            }
        }
    }

    None
}

pub struct DativeAlternationTransform;

impl DativeAlternationTransform {
    pub fn new() -> Self {
        Self
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Applying again alternates back
        self.apply(sentence)
    }
}

impl BaseTransformation for DativeAlternationTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::DativeAlternation
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        detect_dative_form(sentence).is_some()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        let (root_idx, form) = detect_dative_form(sentence).ok_or(
            TransformationError::PreconditionNotMet(
                "No ditransitive verb with dative structure found".into(),
            ),
        )?;

        match form {
            DativeForm::DoubleObject { dobj_idx, iobj_idx } => {
                self.double_to_pp(sentence, root_idx, dobj_idx, iobj_idx)
            }
            DativeForm::PrepPhrase { dobj_idx, prep_idx, pobj_idx } => {
                self.pp_to_double(sentence, root_idx, dobj_idx, prep_idx, pobj_idx)
            }
        }
    }

    fn is_meaning_preserving(&self) -> bool {
        true
    }
}

impl DativeAlternationTransform {
    /// "gave Mary a book" → "gave a book to Mary"
    fn double_to_pp(
        &self,
        sentence: &Sentence,
        root_idx: usize,
        dobj_idx: usize,
        iobj_idx: usize,
    ) -> Result<TransformationResult, TransformationError> {
        let iobj_span = subtree_indices(sentence, iobj_idx);
        let dobj_span = subtree_indices(sentence, dobj_idx);

        // Build: tokens_before_verb + verb + dobj + "to" + iobj + rest
        let mut new_tokens: Vec<Token> = Vec::new();

        // Tokens before the verb (subject etc.)
        for i in 0..root_idx {
            if !iobj_span.contains(&i) && !dobj_span.contains(&i) {
                new_tokens.push(sentence.tokens[i].clone());
            }
        }

        // Verb
        new_tokens.push(sentence.tokens[root_idx].clone());

        // Direct object
        for &i in &dobj_span {
            if let Some(t) = sentence.tokens.get(i) {
                new_tokens.push(t.clone());
            }
        }

        // "to"
        new_tokens.push(Token::new("to", 0).with_pos(PosTag::Prep));

        // Indirect object
        for &i in &iobj_span {
            if let Some(t) = sentence.tokens.get(i) {
                new_tokens.push(t.clone());
            }
        }

        // Remaining tokens (modifiers after verb, except dobj/iobj)
        for i in (root_idx + 1)..sentence.tokens.len() {
            if !iobj_span.contains(&i) && !dobj_span.contains(&i)
                && sentence.tokens[i].pos_tag != Some(PosTag::Punct)
            {
                new_tokens.push(sentence.tokens[i].clone());
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
            vec![(root_idx, sentence.tokens.len())],
            "Double-object → PP dative".to_string(),
        ))
    }

    /// "gave a book to Mary" → "gave Mary a book"
    fn pp_to_double(
        &self,
        sentence: &Sentence,
        root_idx: usize,
        dobj_idx: usize,
        prep_idx: usize,
        pobj_idx: usize,
    ) -> Result<TransformationResult, TransformationError> {
        let dobj_span = subtree_indices(sentence, dobj_idx);
        let pobj_span = subtree_indices(sentence, pobj_idx);

        let mut skip = dobj_span.clone();
        skip.extend(&pobj_span);
        skip.push(prep_idx);
        skip.sort();

        // Build: tokens_before_verb + verb + pobj (new iobj) + dobj + rest
        let mut new_tokens: Vec<Token> = Vec::new();

        // Tokens before verb
        for i in 0..root_idx {
            if !skip.contains(&i) {
                new_tokens.push(sentence.tokens[i].clone());
            }
        }

        // Verb
        new_tokens.push(sentence.tokens[root_idx].clone());

        // pobj becomes indirect object
        for &i in &pobj_span {
            if let Some(t) = sentence.tokens.get(i) {
                new_tokens.push(t.clone());
            }
        }

        // dobj stays
        for &i in &dobj_span {
            if let Some(t) = sentence.tokens.get(i) {
                new_tokens.push(t.clone());
            }
        }

        // Remaining (skip what we've used)
        for i in (root_idx + 1)..sentence.tokens.len() {
            if !skip.contains(&i) && i != root_idx && sentence.tokens[i].pos_tag != Some(PosTag::Punct) {
                new_tokens.push(sentence.tokens[i].clone());
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
            vec![(root_idx, sentence.tokens.len())],
            "PP dative → double-object".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn double_object_sentence() -> Sentence {
        // "John gave Mary a book."
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("gave", 1).with_pos(PosTag::Verb),
            Token::new("Mary", 2).with_pos(PosTag::Noun),
            Token::new("a", 3).with_pos(PosTag::Det),
            Token::new("book", 4).with_pos(PosTag::Noun),
            Token::new(".", 5).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Iobj),
            DependencyEdge::new(1, 4, DependencyRelation::Dobj),
            DependencyEdge::new(4, 3, DependencyRelation::Det),
            DependencyEdge::new(1, 5, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John gave Mary a book.");
        s.dependency_edges = edges;
        s
    }

    fn pp_dative_sentence() -> Sentence {
        // "John gave a book to Mary."
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("gave", 1).with_pos(PosTag::Verb),
            Token::new("a", 2).with_pos(PosTag::Det),
            Token::new("book", 3).with_pos(PosTag::Noun),
            Token::new("to", 4).with_pos(PosTag::Prep),
            Token::new("Mary", 5).with_pos(PosTag::Noun),
            Token::new(".", 6).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 3, DependencyRelation::Dobj),
            DependencyEdge::new(3, 2, DependencyRelation::Det),
            DependencyEdge::new(1, 4, DependencyRelation::Prep),
            DependencyEdge::new(4, 5, DependencyRelation::Pobj),
            DependencyEdge::new(1, 6, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John gave a book to Mary.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = DativeAlternationTransform::new();
        assert_eq!(t.kind(), TransformationKind::DativeAlternation);
    }

    #[test]
    fn test_is_applicable_double_object() {
        let t = DativeAlternationTransform::new();
        assert!(t.is_applicable(&double_object_sentence()));
    }

    #[test]
    fn test_is_applicable_pp() {
        let t = DativeAlternationTransform::new();
        assert!(t.is_applicable(&pp_dative_sentence()));
    }

    #[test]
    fn test_double_to_pp() {
        let t = DativeAlternationTransform::new();
        let result = t.apply(&double_object_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.contains("to"), "Should contain 'to': {}", text);
        let book_pos = text.find("book").unwrap_or(0);
        let mary_pos = text.find("Mary").unwrap_or(0);
        assert!(book_pos < mary_pos, "book should come before Mary: {}", text);
    }

    #[test]
    fn test_pp_to_double() {
        let t = DativeAlternationTransform::new();
        let result = t.apply(&pp_dative_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(!text.contains(" to "), "Should not contain 'to': {}", text);
        let mary_pos = text.find("Mary").unwrap_or(usize::MAX);
        let book_pos = text.find("book").unwrap_or(usize::MAX);
        assert!(mary_pos < book_pos, "Mary should come before book: {}", text);
    }

    #[test]
    fn test_meaning_preserving() {
        let t = DativeAlternationTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_no_ditransitive() {
        let t = DativeAlternationTransform::new();
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("ran", 1).with_pos(PosTag::Verb),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John ran.");
        s.dependency_edges = edges;
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_ditransitive_verb_check() {
        assert!(is_ditransitive("give"));
        assert!(is_ditransitive("gave"));
        assert!(is_ditransitive("sends"));
        assert!(is_ditransitive("taught"));
        assert!(!is_ditransitive("run"));
        assert!(!is_ditransitive("sleep"));
    }
}
