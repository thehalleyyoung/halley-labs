//! Topicalization: moves a constituent to sentence-initial position.
//! "John ate the cake" → "The cake, John ate"

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    capitalize, decapitalize, rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// Find a topicalizable constituent: prefer objects, then PPs, then adverbials.
fn find_topicalizable(sentence: &Sentence) -> Option<(usize, &'static str)> {
    let root = sentence.root_index()?;

    // Try direct object
    if let Some(edge) = sentence.dependency_edges.iter().find(|e| {
        e.head_index == root && e.relation == DependencyRelation::Dobj
    }) {
        return Some((edge.dependent_index, "object"));
    }

    // Try prepositional phrase
    if let Some(edge) = sentence.dependency_edges.iter().find(|e| {
        e.head_index == root && e.relation == DependencyRelation::Prep
    }) {
        return Some((edge.dependent_index, "pp"));
    }

    // Try adverbial modifier
    if let Some(edge) = sentence.dependency_edges.iter().find(|e| {
        e.head_index == root && e.relation == DependencyRelation::Advmod
    }) {
        return Some((edge.dependent_index, "adverbial"));
    }

    None
}

pub struct TopicalizationTransform;

impl TopicalizationTransform {
    pub fn new() -> Self {
        Self
    }

    /// Inverse: restore canonical word order by moving topic back to original position.
    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Heuristic: if sentence has a comma near the front, the material before the
        // comma is the topicalized constituent.
        let comma_pos = sentence.tokens.iter().position(|t| t.text == ",");
        let comma_idx = comma_pos.ok_or_else(|| {
            TransformationError::InverseNotPossible("no comma found for topic boundary".into())
        })?;

        if comma_idx == 0 || comma_idx >= sentence.tokens.len() - 2 {
            return Err(TransformationError::InverseNotPossible(
                "comma position not consistent with topicalization".into(),
            ));
        }

        let topic: Vec<Token> = sentence.tokens[..comma_idx].to_vec();
        let end = if sentence.tokens.last().map_or(false, |t| t.pos_tag == Some(PosTag::Punct)) {
            sentence.tokens.len() - 1
        } else {
            sentence.tokens.len()
        };
        let rest: Vec<Token> = sentence.tokens[comma_idx + 1..end].to_vec();

        // Reconstruct: find verb in rest, insert topic after verb
        let verb_pos = rest.iter().position(|t| {
            t.pos_tag == Some(PosTag::Verb) || t.pos_tag == Some(PosTag::Aux)
        });

        let mut new_tokens: Vec<Token> = Vec::new();
        if let Some(vp) = verb_pos {
            // subj...verb + topic + rest_after_verb
            for t in &rest[..=vp] {
                new_tokens.push(t.clone());
            }
            for t in &topic {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
            for t in &rest[vp + 1..] {
                new_tokens.push(t.clone());
            }
        } else {
            // Just put rest + topic
            for t in &rest {
                new_tokens.push(t.clone());
            }
            for t in &topic {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
        }

        if let Some(first) = new_tokens.first_mut() {
            first.text = capitalize(&first.text);
        }
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
            "De-topicalized: restored canonical word order",
        ))
    }
}

impl BaseTransformation for TopicalizationTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::Topicalization
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        if sentence.tokens.len() < 4 {
            return false;
        }
        find_topicalizable(sentence).is_some()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "No topicalizable constituent found".into(),
            ));
        }

        let (topic_head, role) = find_topicalizable(sentence).unwrap();
        let topic_span = subtree_indices(sentence, topic_head);

        let mut new_tokens: Vec<Token> = Vec::new();

        // Topicalized constituent first (capitalized)
        let mut first_topic = true;
        for &i in &topic_span {
            if let Some(t) = sentence.tokens.get(i) {
                let mut tok = t.clone();
                if first_topic {
                    tok.text = capitalize(&tok.text);
                    first_topic = false;
                }
                new_tokens.push(tok);
            }
        }

        // Insert comma after topicalized element
        new_tokens.push(Token::new(",", 0).with_pos(PosTag::Punct));

        // Rest of sentence (excluding topic span and final punct)
        for (i, tok) in sentence.tokens.iter().enumerate() {
            if !topic_span.contains(&i) && tok.pos_tag != Some(PosTag::Punct) {
                let mut t = tok.clone();
                // Decapitalize the old sentence-initial word if needed
                if i == 0 || (i < topic_span.first().copied().unwrap_or(usize::MAX)) {
                    t.text = decapitalize(&t.text);
                }
                new_tokens.push(t);
            }
        }

        // Final punctuation
        if let Some(last) = sentence.tokens.last() {
            if last.pos_tag == Some(PosTag::Punct) {
                new_tokens.push(last.clone());
            }
        }

        reindex_tokens(&mut new_tokens);
        let new_tokens_len = new_tokens.len();
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, topic_span.len()), (topic_span.len() + 1, new_tokens_len)],
            format!("Topicalized {} constituent", role),
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

    fn transitive_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("ate", 1).with_pos(PosTag::Verb),
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

    fn pp_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("sat", 1).with_pos(PosTag::Verb),
            Token::new("on", 2).with_pos(PosTag::Prep),
            Token::new("the", 3).with_pos(PosTag::Det),
            Token::new("chair", 4).with_pos(PosTag::Noun),
            Token::new(".", 5).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Prep),
            DependencyEdge::new(2, 4, DependencyRelation::Pobj),
            DependencyEdge::new(4, 3, DependencyRelation::Det),
            DependencyEdge::new(1, 5, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John sat on the chair.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = TopicalizationTransform::new();
        assert_eq!(t.kind(), TransformationKind::Topicalization);
    }

    #[test]
    fn test_is_applicable() {
        let t = TopicalizationTransform::new();
        assert!(t.is_applicable(&transitive_sentence()));
        assert!(t.is_applicable(&pp_sentence()));
    }

    #[test]
    fn test_apply_object_topicalization() {
        let t = TopicalizationTransform::new();
        let result = t.apply(&transitive_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        // "The cake" should appear before "John"
        let cake_pos = text.find("cake").unwrap_or(usize::MAX);
        let john_pos = text.find("John").unwrap_or(text.find("john").unwrap_or(usize::MAX));
        assert!(cake_pos < john_pos);
    }

    #[test]
    fn test_apply_pp_topicalization() {
        let t = TopicalizationTransform::new();
        let result = t.apply(&pp_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.contains(","));
    }

    #[test]
    fn test_meaning_preserving() {
        let t = TopicalizationTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_short() {
        let t = TopicalizationTransform::new();
        let s = Sentence::from_tokens(vec![Token::new("Go", 0)], "Go");
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_inverse() {
        let t = TopicalizationTransform::new();
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("cake", 1).with_pos(PosTag::Noun),
            Token::new(",", 2).with_pos(PosTag::Punct),
            Token::new("John", 3).with_pos(PosTag::Noun),
            Token::new("ate", 4).with_pos(PosTag::Verb),
            Token::new(".", 5).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, "The cake, John ate.");
        let inv = t.inverse(&s).unwrap();
        assert!(inv.success);
        let text = inv.transformed.surface_text();
        let john_pos = text.find("John").unwrap_or(text.find("john").unwrap_or(0));
        let cake_pos = text.find("cake").unwrap_or(usize::MAX);
        assert!(john_pos < cake_pos);
    }
}
