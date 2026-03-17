//! Embedding depth change: wraps/unwraps a sentence in clausal embedding.
//! "John left" → "I think that John left" → "She said that I think that John left"

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    capitalize, decapitalize, rebuild_sentence, reindex_tokens,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// Embedding frames: (subject, verb, complementizer).
const EMBEDDING_FRAMES: &[(&str, &str, &str)] = &[
    ("I", "think", "that"),
    ("She", "said", "that"),
    ("He", "believes", "that"),
    ("They", "know", "that"),
    ("We", "assume", "that"),
    ("He", "reported", "that"),
    ("She", "claims", "that"),
    ("They", "suggest", "that"),
];

/// Detect if a sentence already starts with an embedding.
fn detect_embedding(sentence: &Sentence) -> Option<usize> {
    if sentence.tokens.len() < 4 {
        return None;
    }

    // Look for "that" complementizer
    for (i, tok) in sentence.tokens.iter().enumerate() {
        if tok.text.to_lowercase() == "that"
            && (tok.pos_tag == Some(PosTag::Part) || tok.pos_tag == Some(PosTag::Conj)
                || tok.pos_tag == Some(PosTag::Other) || tok.pos_tag.is_none())
        {
            // Verify there's a matrix clause before "that"
            if i >= 2 {
                let prev = &sentence.tokens[i - 1];
                let low = prev.text.to_lowercase();
                let is_embed_verb = [
                    "think", "thinks", "thought", "say", "says", "said",
                    "believe", "believes", "believed", "know", "knows", "knew",
                    "assume", "assumes", "assumed", "report", "reports", "reported",
                    "claim", "claims", "claimed", "suggest", "suggests", "suggested",
                ].contains(&low.as_str());
                if is_embed_verb {
                    return Some(i);
                }
            }
        }
    }
    None
}

pub struct EmbeddingDepthTransform {
    pub target_depth: i32, // +1 to add embedding, -1 to remove one
}

impl EmbeddingDepthTransform {
    pub fn new() -> Self {
        Self { target_depth: 1 }
    }

    pub fn with_depth(depth: i32) -> Self {
        Self { target_depth: depth }
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        let reverse = EmbeddingDepthTransform::with_depth(-self.target_depth);
        reverse.apply(sentence)
    }

    /// Add one layer of embedding.
    fn add_embedding(
        &self,
        sentence: &Sentence,
        depth: usize,
    ) -> Result<TransformationResult, TransformationError> {
        let frame_idx = depth.min(EMBEDDING_FRAMES.len() - 1);
        let (subj, verb, comp) = EMBEDDING_FRAMES[frame_idx];

        let mut new_tokens: Vec<Token> = Vec::new();

        // Matrix clause: "I think that"
        new_tokens.push(Token::new(subj, 0).with_pos(PosTag::Pron));
        new_tokens.push(Token::new(verb, 0).with_pos(PosTag::Verb));
        new_tokens.push(Token::new(comp, 0).with_pos(PosTag::Part));

        // Embedded clause (decapitalize first token)
        for (i, tok) in sentence.tokens.iter().enumerate() {
            if tok.pos_tag == Some(PosTag::Punct) && i == sentence.tokens.len() - 1 {
                continue; // skip final punct for now
            }
            let mut t = tok.clone();
            if i == 0 {
                t.text = decapitalize(&t.text);
            }
            new_tokens.push(t);
        }

        // Add final punct
        if let Some(last) = sentence.tokens.last() {
            if last.pos_tag == Some(PosTag::Punct) {
                new_tokens.push(last.clone());
            } else {
                new_tokens.push(Token::new(".", 0).with_pos(PosTag::Punct));
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, 3)],
            format!("Added embedding: '{} {} {}'", subj, verb, comp),
        ))
    }

    /// Remove one layer of embedding.
    fn remove_embedding(
        &self,
        sentence: &Sentence,
    ) -> Result<TransformationResult, TransformationError> {
        let that_idx = detect_embedding(sentence).ok_or(
            TransformationError::NoApplicableSite,
        )?;

        // Everything after "that" is the embedded clause
        let mut new_tokens: Vec<Token> = Vec::new();
        for i in (that_idx + 1)..sentence.tokens.len() {
            let mut tok = sentence.tokens[i].clone();
            if new_tokens.is_empty() {
                tok.text = capitalize(&tok.text);
            }
            new_tokens.push(tok);
        }

        // Ensure final punct
        if new_tokens.last().map_or(true, |t| t.pos_tag != Some(PosTag::Punct)) {
            new_tokens.push(Token::new(".", 0).with_pos(PosTag::Punct));
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(0, that_idx + 1)],
            "Removed one layer of clausal embedding".to_string(),
        ))
    }
}

impl BaseTransformation for EmbeddingDepthTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::EmbeddingDepthChange
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        if sentence.tokens.len() < 2 {
            return false;
        }
        if self.target_depth > 0 {
            // Can always add embedding
            true
        } else {
            // Can only remove if there's an existing embedding
            detect_embedding(sentence).is_some()
        }
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "Cannot apply embedding change".into(),
            ));
        }

        if self.target_depth > 0 {
            // Add embedding(s)
            let mut current = sentence.clone();
            let mut last_result = None;
            for i in 0..self.target_depth as usize {
                let r = self.add_embedding(&current, i)?;
                current = r.transformed.clone();
                last_result = Some(r);
            }
            let mut final_result = last_result.unwrap();
            final_result.original = sentence.clone();
            Ok(final_result)
        } else {
            // Remove embedding(s)
            let mut current = sentence.clone();
            let mut last_result = None;
            for _ in 0..(-self.target_depth as usize) {
                match self.remove_embedding(&current) {
                    Ok(r) => {
                        current = r.transformed.clone();
                        last_result = Some(r);
                    }
                    Err(_) => break,
                }
            }
            match last_result {
                Some(mut r) => {
                    r.original = sentence.clone();
                    Ok(r)
                }
                None => Err(TransformationError::NoApplicableSite),
            }
        }
    }

    fn is_meaning_preserving(&self) -> bool {
        false // embedding adds a matrix clause / attitude report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn simple_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("left", 1).with_pos(PosTag::Verb),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John left.");
        s.dependency_edges = edges;
        s
    }

    fn embedded_sentence() -> Sentence {
        let tokens = vec![
            Token::new("I", 0).with_pos(PosTag::Pron),
            Token::new("think", 1).with_pos(PosTag::Verb),
            Token::new("that", 2).with_pos(PosTag::Part),
            Token::new("John", 3).with_pos(PosTag::Noun),
            Token::new("left", 4).with_pos(PosTag::Verb),
            Token::new(".", 5).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 4, DependencyRelation::Ccomp),
            DependencyEdge::new(4, 2, DependencyRelation::Mark),
            DependencyEdge::new(4, 3, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 5, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "I think that John left.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = EmbeddingDepthTransform::new();
        assert_eq!(t.kind(), TransformationKind::EmbeddingDepthChange);
    }

    #[test]
    fn test_add_embedding() {
        let t = EmbeddingDepthTransform::new();
        let result = t.apply(&simple_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.contains("think") || text.contains("said") || text.contains("believes"));
        assert!(text.contains("that"));
        assert!(text.to_lowercase().contains("john"));
    }

    #[test]
    fn test_remove_embedding() {
        let t = EmbeddingDepthTransform::with_depth(-1);
        let result = t.apply(&embedded_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        assert!(text.starts_with("John") || text.starts_with("john"));
        assert!(!text.contains("think"));
    }

    #[test]
    fn test_double_embedding() {
        let t = EmbeddingDepthTransform::with_depth(2);
        let result = t.apply(&simple_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        // Should have two "that" occurrences
        let count = text.matches("that").count();
        assert!(count >= 2, "Expected ≥2 'that', got {}: {}", count, text);
    }

    #[test]
    fn test_not_applicable_remove_no_embed() {
        let t = EmbeddingDepthTransform::with_depth(-1);
        assert!(!t.is_applicable(&simple_sentence()));
    }

    #[test]
    fn test_is_applicable_add() {
        let t = EmbeddingDepthTransform::new();
        assert!(t.is_applicable(&simple_sentence()));
    }

    #[test]
    fn test_not_meaning_preserving() {
        let t = EmbeddingDepthTransform::new();
        assert!(!t.is_meaning_preserving());
    }

    #[test]
    fn test_inverse() {
        let t = EmbeddingDepthTransform::new();
        let result = t.apply(&simple_sentence()).unwrap();
        let inv = t.inverse(&result.transformed).unwrap();
        assert!(inv.success);
        let text = inv.transformed.surface_text();
        assert!(text.contains("John") || text.contains("john"));
    }
}
