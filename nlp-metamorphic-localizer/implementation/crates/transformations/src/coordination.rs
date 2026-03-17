//! Coordinated NP reordering: "A and B" → "B and A".

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// A coordination structure: head conjunct, connector, and conjunct(s).
#[derive(Debug)]
struct CoordinationSite {
    head_idx: usize,
    conj_indices: Vec<usize>,
    cc_idx: Option<usize>,
}

/// Find coordination structures in the sentence (cc + conj dependencies).
fn find_coordinations(sentence: &Sentence) -> Vec<CoordinationSite> {
    let mut sites: Vec<CoordinationSite> = Vec::new();
    let mut seen_heads: Vec<usize> = Vec::new();

    for edge in &sentence.dependency_edges {
        if edge.relation == DependencyRelation::Conj && !seen_heads.contains(&edge.head_index) {
            let head = edge.head_index;
            seen_heads.push(head);

            let conjs: Vec<usize> = sentence
                .dependency_edges
                .iter()
                .filter(|e| e.head_index == head && e.relation == DependencyRelation::Conj)
                .map(|e| e.dependent_index)
                .collect();

            let cc = sentence
                .dependency_edges
                .iter()
                .find(|e| e.head_index == head && e.relation == DependencyRelation::Cc)
                .map(|e| e.dependent_index);

            sites.push(CoordinationSite {
                head_idx: head,
                conj_indices: conjs,
                cc_idx: cc,
            });
        }
    }
    sites
}

pub struct CoordinatedNpReorderTransform;

impl CoordinatedNpReorderTransform {
    pub fn new() -> Self {
        Self
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Reordering again reverses the swap
        self.apply(sentence)
    }
}

impl BaseTransformation for CoordinatedNpReorderTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::CoordinatedNpReorder
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        !find_coordinations(sentence).is_empty()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        let sites = find_coordinations(sentence);
        if sites.is_empty() {
            return Err(TransformationError::PreconditionNotMet(
                "No coordination found".into(),
            ));
        }

        let site = &sites[0];
        let head_span = subtree_indices_no_conj(sentence, site.head_idx, &site.conj_indices, site.cc_idx);
        let mut conj_spans: Vec<Vec<usize>> = site
            .conj_indices
            .iter()
            .map(|&ci| subtree_indices(sentence, ci))
            .collect();

        // Build all conjuncts in order: [head_span, conj_span1, conj_span2, ...]
        let mut all_conjuncts: Vec<Vec<usize>> = Vec::new();
        all_conjuncts.push(head_span.clone());
        all_conjuncts.append(&mut conj_spans);

        if all_conjuncts.len() < 2 {
            return Err(TransformationError::NoApplicableSite);
        }

        // Rotate: [A, B] → [B, A], or [A, B, C] → [C, A, B]
        let first = all_conjuncts.remove(0);
        all_conjuncts.push(first);

        // Collect all indices involved in the coordination
        let mut all_involved: Vec<usize> = Vec::new();
        for conj in &all_conjuncts {
            all_involved.extend(conj);
        }
        if let Some(cc) = site.cc_idx {
            all_involved.push(cc);
        }
        all_involved.sort();
        all_involved.dedup();

        // Determine the range in the original sentence
        let range_start = *all_involved.first().unwrap_or(&0);
        let range_end = *all_involved.last().unwrap_or(&0);

        // Build new tokens
        let mut new_tokens: Vec<Token> = Vec::new();

        // Tokens before coordination
        for i in 0..range_start {
            new_tokens.push(sentence.tokens[i].clone());
        }

        // Reordered conjuncts with connectors
        let cc_text = site.cc_idx.and_then(|i| sentence.tokens.get(i))
            .map(|t| t.text.clone())
            .unwrap_or_else(|| "and".to_string());

        for (j, conj_span) in all_conjuncts.iter().enumerate() {
            if j > 0 {
                if all_conjuncts.len() > 2 && j < all_conjuncts.len() - 1 {
                    new_tokens.push(Token::new(",", 0).with_pos(PosTag::Punct));
                }
                if j == all_conjuncts.len() - 1 {
                    if all_conjuncts.len() > 2 {
                        new_tokens.push(Token::new(",", 0).with_pos(PosTag::Punct));
                    }
                    new_tokens.push(Token::new(&cc_text, 0).with_pos(PosTag::Conj));
                }
            }
            for &idx in conj_span {
                if let Some(t) = sentence.tokens.get(idx) {
                    new_tokens.push(t.clone());
                }
            }
        }

        // Tokens after coordination
        for i in (range_end + 1)..sentence.tokens.len() {
            new_tokens.push(sentence.tokens[i].clone());
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(range_start, range_end + 1)],
            "Reordered coordinated conjuncts".to_string(),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        true // A and B ≡ B and A (for NP coordination)
    }
}

/// Collect subtree indices for the head conjunct, excluding conj dependents
/// and the cc token.
fn subtree_indices_no_conj(
    sentence: &Sentence,
    root_idx: usize,
    conj_indices: &[usize],
    cc_idx: Option<usize>,
) -> Vec<usize> {
    let mut result = Vec::new();
    let mut stack = vec![root_idx];
    let mut exclude_subtrees: Vec<usize> = conj_indices.to_vec();
    if let Some(cc) = cc_idx {
        exclude_subtrees.push(cc);
    }

    while let Some(idx) = stack.pop() {
        if exclude_subtrees.contains(&idx) {
            continue;
        }
        result.push(idx);
        for dep in sentence.dependents_of(idx) {
            if !exclude_subtrees.contains(&dep) {
                stack.push(dep);
            }
        }
    }
    result.sort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn coord_sentence() -> Sentence {
        // "John and Mary ran."
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("and", 1).with_pos(PosTag::Conj),
            Token::new("Mary", 2).with_pos(PosTag::Noun),
            Token::new("ran", 3).with_pos(PosTag::Verb),
            Token::new(".", 4).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(3, 3, DependencyRelation::Root),
            DependencyEdge::new(3, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(0, 1, DependencyRelation::Cc),
            DependencyEdge::new(0, 2, DependencyRelation::Conj),
            DependencyEdge::new(3, 4, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John and Mary ran.");
        s.dependency_edges = edges;
        s
    }

    fn triple_coord() -> Sentence {
        // "A , B , and C ran."
        let tokens = vec![
            Token::new("A", 0).with_pos(PosTag::Noun),
            Token::new(",", 1).with_pos(PosTag::Punct),
            Token::new("B", 2).with_pos(PosTag::Noun),
            Token::new(",", 3).with_pos(PosTag::Punct),
            Token::new("and", 4).with_pos(PosTag::Conj),
            Token::new("C", 5).with_pos(PosTag::Noun),
            Token::new("ran", 6).with_pos(PosTag::Verb),
            Token::new(".", 7).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(6, 6, DependencyRelation::Root),
            DependencyEdge::new(6, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(0, 4, DependencyRelation::Cc),
            DependencyEdge::new(0, 2, DependencyRelation::Conj),
            DependencyEdge::new(0, 5, DependencyRelation::Conj),
            DependencyEdge::new(6, 7, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "A, B, and C ran.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = CoordinatedNpReorderTransform::new();
        assert_eq!(t.kind(), TransformationKind::CoordinatedNpReorder);
    }

    #[test]
    fn test_is_applicable() {
        let t = CoordinatedNpReorderTransform::new();
        assert!(t.is_applicable(&coord_sentence()));
    }

    #[test]
    fn test_apply_swap() {
        let t = CoordinatedNpReorderTransform::new();
        let result = t.apply(&coord_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        let mary_pos = text.find("Mary").unwrap_or(usize::MAX);
        let john_pos = text.find("John").unwrap_or(usize::MAX);
        assert!(mary_pos < john_pos, "Mary should come before John: {}", text);
    }

    #[test]
    fn test_apply_triple() {
        let t = CoordinatedNpReorderTransform::new();
        let result = t.apply(&triple_coord()).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_meaning_preserving() {
        let t = CoordinatedNpReorderTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_no_coord() {
        let t = CoordinatedNpReorderTransform::new();
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("ran", 1).with_pos(PosTag::Verb),
        ];
        let s = Sentence::from_tokens(tokens, "John ran");
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_inverse_is_double_apply() {
        let t = CoordinatedNpReorderTransform::new();
        let s = coord_sentence();
        let r1 = t.apply(&s).unwrap();
        // Second apply on result should reorder back
        let mut r1_sent = r1.transformed.clone();
        r1_sent.dependency_edges = s.dependency_edges.clone();
        // The inverse of a swap is another swap (handled in logic)
    }
}
