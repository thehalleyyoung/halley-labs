//! PP attachment variation: moves a prepositional phrase to an alternative
//! attachment site (VP-attach ↔ NP-attach).

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};

/// A PP site: the preposition index, its object subtree, and its current head.
#[derive(Debug)]
struct PpSite {
    prep_idx: usize,
    pobj_idx: Option<usize>,
    current_head_index: usize,
}

/// Find prepositional phrases in the sentence.
fn find_pp_sites(sentence: &Sentence) -> Vec<PpSite> {
    let mut sites = Vec::new();
    for edge in &sentence.dependency_edges {
        if edge.relation == DependencyRelation::Prep {
            let prep_idx = edge.dependent_index;
            let head = edge.head_index;
            let pobj = sentence.dependency_edges.iter().find(|e| {
                e.head_index == prep_idx && e.relation == DependencyRelation::Pobj
            }).map(|e| e.dependent_index);
            sites.push(PpSite {
                prep_idx,
                pobj_idx: pobj,
                current_head_index: head,
            });
        }
    }
    sites
}

/// Find alternative attachment sites for a PP.
fn find_alternative_heads(sentence: &Sentence, pp: &PpSite) -> Vec<usize> {
    let mut candidates = Vec::new();
    let root = sentence.root_index().unwrap_or(0);

    // If currently VP-attached, try NP-attachment (to object or subject)
    if sentence.tokens.get(pp.current_head_index).map_or(false, |t| t.pos_tag == Some(PosTag::Verb)) {
        // Find nouns that are dependents of the verb
        for edge in &sentence.dependency_edges {
            if edge.head_index == pp.current_head_index
                && (edge.relation == DependencyRelation::Dobj
                    || edge.relation == DependencyRelation::Nsubj)
            {
                let dep = edge.dependent_index;
                if sentence.tokens.get(dep).map_or(false, |t| t.pos_tag == Some(PosTag::Noun)) {
                    candidates.push(dep);
                }
            }
        }
    }

    // If currently NP-attached, try VP-attachment
    if sentence.tokens.get(pp.current_head_index).map_or(false, |t| t.pos_tag == Some(PosTag::Noun)) {
        // Attach to the verb that governs this NP
        if let Some(head_edge) = sentence.dependency_edges.iter().find(|e| {
            e.dependent_index == pp.current_head_index
        }) {
            if sentence.tokens.get(head_edge.head_index).map_or(false, |t| {
                t.pos_tag == Some(PosTag::Verb)
            }) {
                candidates.push(head_edge.head_index);
            }
        }
        // Also try root verb
        if sentence.tokens.get(root).map_or(false, |t| t.pos_tag == Some(PosTag::Verb))
            && !candidates.contains(&root)
        {
            candidates.push(root);
        }
    }

    candidates
}

pub struct PpAttachmentTransform;

impl PpAttachmentTransform {
    pub fn new() -> Self {
        Self
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Re-applying moves back (the alternative attachment becomes current)
        self.apply(sentence)
    }
}

impl BaseTransformation for PpAttachmentTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::PpAttachmentVariation
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        let sites = find_pp_sites(sentence);
        sites.iter().any(|pp| !find_alternative_heads(sentence, pp).is_empty())
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        let sites = find_pp_sites(sentence);
        let (pp, new_head) = sites
            .iter()
            .find_map(|pp| {
                let alts = find_alternative_heads(sentence, pp);
                alts.first().map(|&h| (pp, h))
            })
            .ok_or(TransformationError::NoApplicableSite)?;

        // Collect PP span
        let pp_tokens_indices = if let Some(pobj) = pp.pobj_idx {
            let mut indices = subtree_indices(sentence, pobj);
            indices.push(pp.prep_idx);
            indices.sort();
            indices
        } else {
            vec![pp.prep_idx]
        };

        let pp_token_list: Vec<Token> = pp_tokens_indices
            .iter()
            .filter_map(|&i| sentence.tokens.get(i).cloned())
            .collect();

        // Find the new head's subtree end to know where to insert
        let new_head_span = subtree_indices(sentence, new_head);
        let insert_after = *new_head_span.last().unwrap_or(&new_head);

        // Build new sentence: remove PP from old position, insert at new position
        let mut new_tokens: Vec<Token> = Vec::new();
        let mut inserted = false;

        for (i, tok) in sentence.tokens.iter().enumerate() {
            if pp_tokens_indices.contains(&i) {
                continue; // skip old PP position
            }
            new_tokens.push(tok.clone());

            if i == insert_after && !inserted {
                for pt in &pp_token_list {
                    new_tokens.push(pt.clone());
                }
                inserted = true;
            }
        }

        // If insert_after was in the removed set, append at end (before punct)
        if !inserted {
            let punct_at_end = new_tokens.last().map_or(false, |t| t.pos_tag == Some(PosTag::Punct));
            if punct_at_end {
                let punct = new_tokens.pop().unwrap();
                for pt in &pp_token_list {
                    new_tokens.push(pt.clone());
                }
                new_tokens.push(punct);
            } else {
                for pt in &pp_token_list {
                    new_tokens.push(pt.clone());
                }
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(pp.prep_idx, pp.prep_idx + pp_tokens_indices.len())],
            format!(
                "Moved PP '{}' from head {} to head {}",
                sentence.tokens.get(pp.prep_idx).map(|t| t.text.as_str()).unwrap_or("?"),
                pp.current_head_index,
                new_head
            ),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        false // different attachment = different meaning
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn vp_attached_pp() -> Sentence {
        // "John saw the man with binoculars."  (PP attached to verb "saw")
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("saw", 1).with_pos(PosTag::Verb),
            Token::new("the", 2).with_pos(PosTag::Det),
            Token::new("man", 3).with_pos(PosTag::Noun),
            Token::new("with", 4).with_pos(PosTag::Prep),
            Token::new("binoculars", 5).with_pos(PosTag::Noun),
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
        let mut s = Sentence::from_tokens(tokens, "John saw the man with binoculars.");
        s.dependency_edges = edges;
        s
    }

    fn np_attached_pp() -> Sentence {
        // "John saw the man with a hat."  (PP attached to noun "man")
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("saw", 1).with_pos(PosTag::Verb),
            Token::new("the", 2).with_pos(PosTag::Det),
            Token::new("man", 3).with_pos(PosTag::Noun),
            Token::new("with", 4).with_pos(PosTag::Prep),
            Token::new("a", 5).with_pos(PosTag::Det),
            Token::new("hat", 6).with_pos(PosTag::Noun),
            Token::new(".", 7).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 3, DependencyRelation::Dobj),
            DependencyEdge::new(3, 2, DependencyRelation::Det),
            DependencyEdge::new(3, 4, DependencyRelation::Prep),
            DependencyEdge::new(4, 6, DependencyRelation::Pobj),
            DependencyEdge::new(6, 5, DependencyRelation::Det),
            DependencyEdge::new(1, 7, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John saw the man with a hat.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = PpAttachmentTransform::new();
        assert_eq!(t.kind(), TransformationKind::PpAttachmentVariation);
    }

    #[test]
    fn test_is_applicable_vp() {
        let t = PpAttachmentTransform::new();
        assert!(t.is_applicable(&vp_attached_pp()));
    }

    #[test]
    fn test_is_applicable_np() {
        let t = PpAttachmentTransform::new();
        assert!(t.is_applicable(&np_attached_pp()));
    }

    #[test]
    fn test_apply_vp_to_np() {
        let t = PpAttachmentTransform::new();
        let result = t.apply(&vp_attached_pp()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text();
        // "with binoculars" should now be near "man"
        let man_pos = text.find("man").unwrap_or(0);
        let with_pos = text.find("with").unwrap_or(usize::MAX);
        assert!(with_pos > man_pos, "PP should follow the noun: {}", text);
    }

    #[test]
    fn test_apply_np_to_vp() {
        let t = PpAttachmentTransform::new();
        let result = t.apply(&np_attached_pp()).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_not_meaning_preserving() {
        let t = PpAttachmentTransform::new();
        assert!(!t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_no_pp() {
        let t = PpAttachmentTransform::new();
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
}
