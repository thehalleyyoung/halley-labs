//! Negation insertion: "John runs" → "John does not run".

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// Check if the sentence is already negated.
fn is_negated(sentence: &Sentence) -> bool {
    sentence.tokens.iter().any(|t| {
        let low = t.text.to_lowercase();
        low == "not" || low == "n't" || low == "never" || low == "no"
            || low == "neither" || low == "nor" || low == "cannot"
    })
}

/// Find the main verb (root) and any auxiliary.
fn find_verb_and_aux(sentence: &Sentence) -> Option<(usize, Option<usize>)> {
    let root = sentence.root_index()?;
    let aux = sentence.dependency_edges.iter().find(|e| {
        e.head_index == root && e.relation == DependencyRelation::Aux
    }).map(|e| e.dependent_index);
    Some((root, aux))
}

/// Strip 3rd-person -s from a verb to get base form for do-support.
fn strip_3sg(verb: &str) -> String {
    let low = verb.to_lowercase();
    if low.ends_with("ies") {
        format!("{}y", &low[..low.len() - 3])
    } else if low.ends_with("es")
        && (low.ends_with("shes") || low.ends_with("ches")
            || low.ends_with("xes") || low.ends_with("zes")
            || low.ends_with("sses"))
    {
        low[..low.len() - 2].to_string()
    } else if low.ends_with('s') && !low.ends_with("ss") {
        low[..low.len() - 1].to_string()
    } else {
        low
    }
}

/// Negate an auxiliary: "can" → "cannot", "will" → "will not", etc.
fn negate_auxiliary(aux: &str) -> (String, bool) {
    let low = aux.to_lowercase();
    match low.as_str() {
        "can" => ("cannot".to_string(), true),
        "could" => ("could not".to_string(), true),
        "will" => ("will not".to_string(), true),
        "would" => ("would not".to_string(), true),
        "shall" => ("shall not".to_string(), true),
        "should" => ("should not".to_string(), true),
        "may" => ("may not".to_string(), true),
        "might" => ("might not".to_string(), true),
        "must" => ("must not".to_string(), true),
        "is" => ("is not".to_string(), true),
        "are" => ("are not".to_string(), true),
        "was" => ("was not".to_string(), true),
        "were" => ("were not".to_string(), true),
        "am" => ("am not".to_string(), true),
        "has" => ("has not".to_string(), true),
        "have" => ("have not".to_string(), true),
        "had" => ("had not".to_string(), true),
        "do" => ("do not".to_string(), true),
        "does" => ("does not".to_string(), true),
        "did" => ("did not".to_string(), true),
        _ => (format!("{} not", low), true),
    }
}

/// Check if a verb form is past tense (heuristic).
fn is_past_tense(verb: &str) -> bool {
    let low = verb.to_lowercase();
    low.ends_with("ed") || [
        "went", "came", "saw", "did", "had", "made", "said", "took", "got",
        "gave", "found", "knew", "thought", "told", "became", "left", "felt",
        "ran", "brought", "began", "kept", "held", "wrote", "stood", "heard",
        "meant", "met", "paid", "sat", "spoke", "led", "lost", "spent",
        "built", "sent", "fell", "ate", "drew", "fought", "threw", "broke",
        "chose", "grew", "hung", "shook", "wore", "bit", "blew", "bore",
    ].contains(&low.as_str())
}

pub struct NegationInsertionTransform;

impl NegationInsertionTransform {
    pub fn new() -> Self {
        Self
    }

    /// Remove negation (inverse).
    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !is_negated(sentence) {
            return Err(TransformationError::InverseNotPossible(
                "sentence is not negated".into(),
            ));
        }

        let mut new_tokens: Vec<Token> = Vec::new();
        let mut i = 0;
        let tokens = &sentence.tokens;

        while i < tokens.len() {
            let low = tokens[i].text.to_lowercase();

            if low == "cannot" {
                new_tokens.push(Token::new("can", 0).with_pos(PosTag::Aux));
                i += 1;
                continue;
            }
            if low == "not" || low == "n't" {
                // Check if preceded by do-support
                if i > 0 {
                    let prev_low = tokens[i - 1].text.to_lowercase();
                    if prev_low == "do" || prev_low == "does" || prev_low == "did" {
                        // Remove the "do/does/did" and "not", adjust following verb
                        new_tokens.pop(); // remove the do/does/did we just added
                        // The next token should be the base verb; conjugate it
                        if i + 1 < tokens.len() && tokens[i + 1].pos_tag == Some(PosTag::Verb) {
                            let verb = &tokens[i + 1].text;
                            let conjugated = if prev_low == "does" {
                                // Restore 3sg
                                let base = verb.to_lowercase();
                                if base.ends_with("sh") || base.ends_with("ch")
                                    || base.ends_with('x') || base.ends_with('z')
                                    || base.ends_with('s')
                                {
                                    format!("{}es", base)
                                } else {
                                    format!("{}s", base)
                                }
                            } else if prev_low == "did" {
                                // Restore past
                                format!("{}ed", verb.to_lowercase())
                            } else {
                                verb.clone()
                            };
                            new_tokens.push(Token::new(&conjugated, 0).with_pos(PosTag::Verb));
                            i += 2;
                            continue;
                        }
                    }
                }
                // Just skip "not"
                i += 1;
                continue;
            }

            new_tokens.push(tokens[i].clone());
            i += 1;
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, sentence.tokens.len())],
            "Removed negation".to_string(),
        ))
    }
}

impl BaseTransformation for NegationInsertionTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::NegationInsertion
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        if sentence.tokens.len() < 2 {
            return false;
        }
        // Must not already be negated
        if is_negated(sentence) {
            return false;
        }
        // Must have a verb
        sentence.tokens.iter().any(|t| {
            t.pos_tag == Some(PosTag::Verb) || t.pos_tag == Some(PosTag::Aux)
        })
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "Sentence must be affirmative with a verb".into(),
            ));
        }

        let (root_idx, aux_idx) = find_verb_and_aux(sentence)
            .ok_or(TransformationError::NoApplicableSite)?;

        let mut new_tokens: Vec<Token> = Vec::new();

        if let Some(ai) = aux_idx {
            // Has auxiliary: negate it
            let aux_tok = &sentence.tokens[ai];
            let (negated, _) = negate_auxiliary(&aux_tok.text);

            for (i, tok) in sentence.tokens.iter().enumerate() {
                if i == ai {
                    for part in negated.split_whitespace() {
                        new_tokens.push(Token::new(part, 0).with_pos(PosTag::Aux));
                    }
                } else {
                    new_tokens.push(tok.clone());
                }
            }
        } else {
            // No auxiliary: do-support
            let verb_tok = &sentence.tokens[root_idx];
            let is_copula = {
                let low = verb_tok.text.to_lowercase();
                low == "is" || low == "are" || low == "was" || low == "were"
                    || low == "am" || low == "be"
            };

            if is_copula {
                // Negate copula directly
                let (negated, _) = negate_auxiliary(&verb_tok.text);
                for (i, tok) in sentence.tokens.iter().enumerate() {
                    if i == root_idx {
                        for part in negated.split_whitespace() {
                            new_tokens.push(Token::new(part, 0).with_pos(PosTag::Verb));
                        }
                    } else {
                        new_tokens.push(tok.clone());
                    }
                }
            } else {
                // Insert "do/does/did not" before verb, change verb to base form
                let do_form = if is_past_tense(&verb_tok.text) {
                    "did"
                } else if verb_tok.text.to_lowercase().ends_with('s')
                    && !verb_tok.text.to_lowercase().ends_with("ss")
                {
                    "does"
                } else {
                    "do"
                };

                let base_verb = if do_form == "does" {
                    strip_3sg(&verb_tok.text)
                } else if do_form == "did" {
                    verb_tok.lemma.as_deref()
                        .map(|l| l.to_lowercase())
                        .unwrap_or_else(|| {
                            let low = verb_tok.text.to_lowercase();
                            if low.ends_with("ed") {
                                low[..low.len() - 2].to_string()
                            } else {
                                low
                            }
                        })
                } else {
                    verb_tok.text.to_lowercase()
                };

                for (i, tok) in sentence.tokens.iter().enumerate() {
                    if i == root_idx {
                        new_tokens.push(Token::new(do_form, 0).with_pos(PosTag::Aux));
                        new_tokens.push(Token::new("not", 0).with_pos(PosTag::Part));
                        new_tokens.push(Token::new(&base_verb, 0).with_pos(PosTag::Verb));
                    } else {
                        new_tokens.push(tok.clone());
                    }
                }
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(root_idx, root_idx + 1)],
            "Inserted negation".to_string(),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        false // negation flips meaning
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn simple_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("runs", 1).with_pos(PosTag::Verb).with_lemma("run"),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John runs.");
        s.dependency_edges = edges;
        s
    }

    fn copula_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("is", 1).with_pos(PosTag::Verb),
            Token::new("happy", 2).with_pos(PosTag::Adj),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Amod),
            DependencyEdge::new(1, 3, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John is happy.");
        s.dependency_edges = edges;
        s
    }

    fn aux_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("can", 1).with_pos(PosTag::Aux),
            Token::new("swim", 2).with_pos(PosTag::Verb),
            Token::new(".", 3).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(2, 1, DependencyRelation::Aux),
            DependencyEdge::new(2, 3, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John can swim.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = NegationInsertionTransform::new();
        assert_eq!(t.kind(), TransformationKind::NegationInsertion);
    }

    #[test]
    fn test_is_applicable() {
        let t = NegationInsertionTransform::new();
        assert!(t.is_applicable(&simple_sentence()));
        assert!(t.is_applicable(&copula_sentence()));
    }

    #[test]
    fn test_do_support() {
        let t = NegationInsertionTransform::new();
        let result = t.apply(&simple_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("does not run") || text.contains("do not run"));
    }

    #[test]
    fn test_copula_negation() {
        let t = NegationInsertionTransform::new();
        let result = t.apply(&copula_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("is not"));
    }

    #[test]
    fn test_auxiliary_negation() {
        let t = NegationInsertionTransform::new();
        let result = t.apply(&aux_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("cannot") || text.contains("can not"));
    }

    #[test]
    fn test_not_meaning_preserving() {
        let t = NegationInsertionTransform::new();
        assert!(!t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_already_negated() {
        let t = NegationInsertionTransform::new();
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("does", 1).with_pos(PosTag::Aux),
            Token::new("not", 2).with_pos(PosTag::Part),
            Token::new("run", 3).with_pos(PosTag::Verb),
            Token::new(".", 4).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, "John does not run.");
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_inverse() {
        let t = NegationInsertionTransform::new();
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("is", 1).with_pos(PosTag::Verb),
            Token::new("not", 2).with_pos(PosTag::Part),
            Token::new("happy", 3).with_pos(PosTag::Adj),
            Token::new(".", 4).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, "John is not happy.");
        let inv = t.inverse(&s).unwrap();
        assert!(inv.success);
        let text = inv.transformed.surface_text().to_lowercase();
        assert!(!text.contains("not"));
    }
}
