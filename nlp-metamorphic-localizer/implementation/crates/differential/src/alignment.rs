//! Token-to-tree alignment between original and transformed sentences.
//!
//! Provides alignment maps that track which tokens in the original correspond
//! to which tokens in the transformed output, enabling fine-grained differential
//! analysis even when transformations add, remove, or reorder tokens.

use serde::{Deserialize, Serialize};
use shared_types::{
    DependencyEdge, LocalizerError, Result, Sentence, Token,
};
use std::collections::{HashMap, HashSet};

// ── AlignmentMap ────────────────────────────────────────────────────────────

/// Bidirectional mapping between token indices in the original and transformed sentences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentMap {
    pub original_to_transformed: HashMap<usize, usize>,
    pub transformed_to_original: HashMap<usize, usize>,
    pub original_len: usize,
    pub transformed_len: usize,
}

impl AlignmentMap {
    pub fn new(original_len: usize, transformed_len: usize) -> Self {
        Self {
            original_to_transformed: HashMap::new(),
            transformed_to_original: HashMap::new(),
            original_len,
            transformed_len,
        }
    }

    /// Add a bidirectional alignment between original index `o` and transformed index `t`.
    pub fn add(&mut self, o: usize, t: usize) {
        self.original_to_transformed.insert(o, t);
        self.transformed_to_original.insert(t, o);
    }

    /// Get the transformed index for an original index.
    pub fn get_transformed(&self, original_idx: usize) -> Option<usize> {
        self.original_to_transformed.get(&original_idx).copied()
    }

    /// Get the original index for a transformed index.
    pub fn get_original(&self, transformed_idx: usize) -> Option<usize> {
        self.transformed_to_original.get(&transformed_idx).copied()
    }

    /// Indices in original that have no alignment.
    pub fn unaligned_original(&self) -> Vec<usize> {
        (0..self.original_len)
            .filter(|i| !self.original_to_transformed.contains_key(i))
            .collect()
    }

    /// Indices in transformed that have no alignment.
    pub fn unaligned_transformed(&self) -> Vec<usize> {
        (0..self.transformed_len)
            .filter(|i| !self.transformed_to_original.contains_key(i))
            .collect()
    }

    /// Number of aligned pairs.
    pub fn aligned_count(&self) -> usize {
        self.original_to_transformed.len()
    }

    /// Coverage: fraction of original tokens that are aligned.
    pub fn original_coverage(&self) -> f64 {
        if self.original_len == 0 {
            return 1.0;
        }
        self.original_to_transformed.len() as f64 / self.original_len as f64
    }

    /// Coverage: fraction of transformed tokens that are aligned.
    pub fn transformed_coverage(&self) -> f64 {
        if self.transformed_len == 0 {
            return 1.0;
        }
        self.transformed_to_original.len() as f64 / self.transformed_len as f64
    }

    /// Check if the alignment preserves ordering (no crossing alignments).
    pub fn is_monotonic(&self) -> bool {
        let mut pairs: Vec<(usize, usize)> = self.original_to_transformed.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by_key(|&(o, _)| o);
        for i in 1..pairs.len() {
            if pairs[i].1 <= pairs[i - 1].1 {
                return false;
            }
        }
        true
    }
}

// ── AlignmentQuality ────────────────────────────────────────────────────────

/// Quality assessment of an alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentQuality {
    pub coverage: f64,
    pub confidence: f64,
    pub ambiguity_count: usize,
    pub is_monotonic: bool,
}

impl AlignmentQuality {
    pub fn is_high_quality(&self) -> bool {
        self.coverage > 0.8 && self.confidence > 0.7 && self.ambiguity_count == 0
    }
}

/// Compute alignment quality metrics.
pub fn compute_alignment_quality(alignment: &AlignmentMap) -> AlignmentQuality {
    let coverage = (alignment.original_coverage() + alignment.transformed_coverage()) / 2.0;
    let is_monotonic = alignment.is_monotonic();

    // Confidence based on coverage and monotonicity
    let confidence = if is_monotonic {
        coverage
    } else {
        coverage * 0.8 // Penalty for non-monotonic alignment
    };

    AlignmentQuality {
        coverage,
        confidence,
        ambiguity_count: 0, // No ambiguity in final alignment
        is_monotonic,
    }
}

// ── AlignmentFailure ────────────────────────────────────────────────────────

/// Describes why an alignment attempt failed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentFailure {
    pub reason: String,
    pub unmatched_original: Vec<usize>,
    pub unmatched_transformed: Vec<usize>,
    pub partial_alignment: Option<AlignmentMap>,
}

impl AlignmentFailure {
    pub fn new(
        reason: impl Into<String>,
        unmatched_orig: Vec<usize>,
        unmatched_trans: Vec<usize>,
    ) -> Self {
        Self {
            reason: reason.into(),
            unmatched_original: unmatched_orig,
            unmatched_transformed: unmatched_trans,
            partial_alignment: None,
        }
    }

    pub fn with_partial(mut self, alignment: AlignmentMap) -> Self {
        self.partial_alignment = Some(alignment);
        self
    }
}

// ── LemmaAligner ────────────────────────────────────────────────────────────

/// Aligns tokens by their lemma (transformation-invariant for most transformations).
pub struct LemmaAligner {
    pub min_coverage: f64,
}

impl LemmaAligner {
    pub fn new() -> Self {
        Self { min_coverage: 0.5 }
    }

    pub fn with_min_coverage(mut self, min_coverage: f64) -> Self {
        self.min_coverage = min_coverage;
        self
    }

    /// Align by lemma, resolving ambiguities by position proximity.
    pub fn align_by_lemma(
        &self,
        original: &Sentence,
        transformed: &Sentence,
    ) -> std::result::Result<AlignmentMap, AlignmentFailure> {
        let orig_tokens = &original.tokens;
        let trans_tokens = &transformed.tokens;
        let mut alignment = AlignmentMap::new(orig_tokens.len(), trans_tokens.len());

        // Build lemma→indices map for transformed tokens
        let mut trans_lemma_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, tok) in trans_tokens.iter().enumerate() {
            let lemma = get_lemma(tok);
            trans_lemma_map.entry(lemma).or_default().push(i);
        }

        let mut used_trans: HashSet<usize> = HashSet::new();

        for (orig_idx, orig_tok) in orig_tokens.iter().enumerate() {
            let orig_lemma = get_lemma(orig_tok);
            if let Some(candidates) = trans_lemma_map.get(&orig_lemma) {
                // Find closest unused candidate by position
                let best = candidates
                    .iter()
                    .filter(|&&t| !used_trans.contains(&t))
                    .min_by_key(|&&t| {
                        (orig_idx as isize - t as isize).unsigned_abs()
                    })
                    .copied();

                if let Some(trans_idx) = best {
                    alignment.add(orig_idx, trans_idx);
                    used_trans.insert(trans_idx);
                }
            }
        }

        let coverage = alignment.original_coverage();
        if coverage < self.min_coverage {
            return Err(AlignmentFailure::new(
                format!(
                    "Lemma alignment coverage {:.2} below threshold {:.2}",
                    coverage, self.min_coverage
                ),
                alignment.unaligned_original(),
                alignment.unaligned_transformed(),
            )
            .with_partial(alignment));
        }

        Ok(alignment)
    }

    /// Fallback: align by exact surface form match.
    pub fn align_by_surface_form(
        &self,
        original: &Sentence,
        transformed: &Sentence,
    ) -> std::result::Result<AlignmentMap, AlignmentFailure> {
        let orig_tokens = &original.tokens;
        let trans_tokens = &transformed.tokens;
        let mut alignment = AlignmentMap::new(orig_tokens.len(), trans_tokens.len());

        let mut trans_text_map: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, tok) in trans_tokens.iter().enumerate() {
            trans_text_map.entry(tok.text.as_str()).or_default().push(i);
        }

        let mut used_trans: HashSet<usize> = HashSet::new();

        for (orig_idx, orig_tok) in orig_tokens.iter().enumerate() {
            if let Some(candidates) = trans_text_map.get(orig_tok.text.as_str()) {
                let best = candidates
                    .iter()
                    .filter(|&&t| !used_trans.contains(&t))
                    .min_by_key(|&&t| {
                        (orig_idx as isize - t as isize).unsigned_abs()
                    })
                    .copied();

                if let Some(trans_idx) = best {
                    alignment.add(orig_idx, trans_idx);
                    used_trans.insert(trans_idx);
                }
            }
        }

        let coverage = alignment.original_coverage();
        if coverage < self.min_coverage {
            return Err(AlignmentFailure::new(
                format!(
                    "Surface form alignment coverage {:.2} below threshold {:.2}",
                    coverage, self.min_coverage
                ),
                alignment.unaligned_original(),
                alignment.unaligned_transformed(),
            )
            .with_partial(alignment));
        }

        Ok(alignment)
    }

    /// Positional alignment with offset tracking.
    pub fn align_by_position(
        &self,
        original: &Sentence,
        transformed: &Sentence,
    ) -> AlignmentMap {
        let orig_len = original.tokens.len();
        let trans_len = transformed.tokens.len();
        let mut alignment = AlignmentMap::new(orig_len, trans_len);

        let min_len = orig_len.min(trans_len);
        for i in 0..min_len {
            alignment.add(i, i);
        }
        alignment
    }

    /// Multi-strategy alignment: try lemma first, then surface form, then position.
    pub fn align(
        &self,
        original: &Sentence,
        transformed: &Sentence,
    ) -> (AlignmentMap, AlignmentQuality) {
        // Try lemma alignment first
        if let Ok(map) = self.align_by_lemma(original, transformed) {
            let quality = compute_alignment_quality(&map);
            if quality.is_high_quality() {
                return (map, quality);
            }
        }

        // Try surface form alignment
        if let Ok(map) = self.align_by_surface_form(original, transformed) {
            let quality = compute_alignment_quality(&map);
            if quality.coverage > 0.5 {
                return (map, quality);
            }
        }

        // Fallback to positional
        let map = self.align_by_position(original, transformed);
        let quality = compute_alignment_quality(&map);
        (map, quality)
    }
}

impl Default for LemmaAligner {
    fn default() -> Self {
        Self::new()
    }
}

// ── TransformationSpecificAligner ───────────────────────────────────────────

/// For transformations that change the lemma inventory (synonym, negation,
/// agreement, embedding depth), use explicit token-level maps.
pub struct TransformationSpecificAligner {
    pub explicit_map: HashMap<usize, usize>,
}

impl TransformationSpecificAligner {
    pub fn new() -> Self {
        Self {
            explicit_map: HashMap::new(),
        }
    }

    /// Build from an explicit mapping of original→transformed token indices.
    pub fn from_map(map: HashMap<usize, usize>) -> Self {
        Self { explicit_map: map }
    }

    /// Apply the explicit map, filling gaps with lemma or positional alignment.
    pub fn align(
        &self,
        original: &Sentence,
        transformed: &Sentence,
    ) -> AlignmentMap {
        let mut alignment = AlignmentMap::new(original.tokens.len(), transformed.tokens.len());

        // First: apply explicit mappings
        for (&orig_idx, &trans_idx) in &self.explicit_map {
            if orig_idx < original.tokens.len() && trans_idx < transformed.tokens.len() {
                alignment.add(orig_idx, trans_idx);
            }
        }

        // Fill gaps with lemma matching
        let used_orig: HashSet<usize> = alignment.original_to_transformed.keys().copied().collect();
        let used_trans: HashSet<usize> = alignment.transformed_to_original.keys().copied().collect();

        let mut trans_lemma_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, tok) in transformed.tokens.iter().enumerate() {
            if !used_trans.contains(&i) {
                let lemma = get_lemma(tok);
                trans_lemma_map.entry(lemma).or_default().push(i);
            }
        }

        let mut newly_used: HashSet<usize> = HashSet::new();
        for (orig_idx, orig_tok) in original.tokens.iter().enumerate() {
            if used_orig.contains(&orig_idx) {
                continue;
            }
            let orig_lemma = get_lemma(orig_tok);
            if let Some(candidates) = trans_lemma_map.get(&orig_lemma) {
                let best = candidates
                    .iter()
                    .filter(|&&t| !newly_used.contains(&t))
                    .min_by_key(|&&t| (orig_idx as isize - t as isize).unsigned_abs())
                    .copied();
                if let Some(trans_idx) = best {
                    alignment.add(orig_idx, trans_idx);
                    newly_used.insert(trans_idx);
                }
            }
        }

        alignment
    }
}

impl Default for TransformationSpecificAligner {
    fn default() -> Self {
        Self::new()
    }
}

// ── TreeAlignment ───────────────────────────────────────────────────────────

/// Aligns dependency tree nodes between original and transformed sentences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeAlignment {
    pub node_map: HashMap<usize, usize>,
    pub edge_matches: Vec<(DependencyEdge, DependencyEdge)>,
    pub unmatched_original_edges: Vec<DependencyEdge>,
    pub unmatched_transformed_edges: Vec<DependencyEdge>,
}

impl TreeAlignment {
    pub fn new() -> Self {
        Self {
            node_map: HashMap::new(),
            edge_matches: Vec::new(),
            unmatched_original_edges: Vec::new(),
            unmatched_transformed_edges: Vec::new(),
        }
    }

    /// Match tree nodes using an existing token alignment plus structural similarity.
    pub fn align_trees(
        token_alignment: &AlignmentMap,
        original: &Sentence,
        transformed: &Sentence,
    ) -> Self {
        let mut tree_align = TreeAlignment::new();

        // Node alignment from token alignment
        tree_align.node_map = token_alignment.original_to_transformed.clone();

        // Edge matching
        let trans_edges: HashMap<(usize, usize), &DependencyEdge> = transformed
            .dependency_edges
            .iter()
            .map(|e| ((e.head_index, e.dependent_index), e))
            .collect();

        let mut matched_trans_edges: HashSet<(usize, usize)> = HashSet::new();

        for orig_edge in &original.dependency_edges {
            let mapped_head = token_alignment.get_transformed(orig_edge.head_index);
            let mapped_dep = token_alignment.get_transformed(orig_edge.dependent_index);

            match (mapped_head, mapped_dep) {
                (Some(h), Some(d)) => {
                    if let Some(trans_edge) = trans_edges.get(&(h, d)) {
                        tree_align
                            .edge_matches
                            .push((orig_edge.clone(), (*trans_edge).clone()));
                        matched_trans_edges.insert((h, d));
                    } else {
                        tree_align.unmatched_original_edges.push(orig_edge.clone());
                    }
                }
                _ => {
                    tree_align.unmatched_original_edges.push(orig_edge.clone());
                }
            }
        }

        // Find unmatched transformed edges
        for trans_edge in &transformed.dependency_edges {
            if !matched_trans_edges.contains(&(trans_edge.head_index, trans_edge.dependent_index)) {
                tree_align
                    .unmatched_transformed_edges
                    .push(trans_edge.clone());
            }
        }

        tree_align
    }

    /// Fraction of original edges that were matched.
    pub fn edge_recall(&self, total_original_edges: usize) -> f64 {
        if total_original_edges == 0 {
            return 1.0;
        }
        self.edge_matches.len() as f64 / total_original_edges as f64
    }

    /// Fraction of transformed edges that were matched.
    pub fn edge_precision(&self, total_transformed_edges: usize) -> f64 {
        if total_transformed_edges == 0 {
            return 1.0;
        }
        self.edge_matches.len() as f64 / total_transformed_edges as f64
    }
}

impl Default for TreeAlignment {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn get_lemma(token: &Token) -> String {
    token
        .lemma
        .as_deref()
        .unwrap_or(&token.text)
        .to_lowercase()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyRelation, PosTag};

    fn make_token(text: &str, idx: usize) -> Token {
        Token::new(text, idx, idx * 5, idx * 5 + text.len())
    }

    fn make_token_with_lemma(text: &str, lemma: &str, idx: usize) -> Token {
        let mut t = make_token(text, idx);
        t.lemma = Some(lemma.to_string());
        t
    }

    fn make_sentence(words: &[&str]) -> Sentence {
        let tokens: Vec<Token> = words.iter().enumerate().map(|(i, w)| make_token(w, i)).collect();
        Sentence {
            text: words.join(" "),
            tokens,
            entities: Vec::new(),
            dependencies: Vec::new(),
            parse_tree: None,
            features: None,
        }
    }

    fn make_sentence_with_lemmas(pairs: &[(&str, &str)]) -> Sentence {
        let tokens: Vec<Token> = pairs
            .iter()
            .enumerate()
            .map(|(i, (text, lemma))| make_token_with_lemma(text, lemma, i))
            .collect();
        let text = pairs.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" ");
        Sentence {
            text,
            tokens,
            entities: Vec::new(),
            dependencies: Vec::new(),
            parse_tree: None,
            features: None,
        }
    }

    #[test]
    fn test_alignment_map_basic() {
        let mut map = AlignmentMap::new(3, 3);
        map.add(0, 0);
        map.add(1, 1);
        map.add(2, 2);
        assert_eq!(map.get_transformed(0), Some(0));
        assert_eq!(map.get_original(1), Some(1));
        assert_eq!(map.aligned_count(), 3);
        assert!(map.unaligned_original().is_empty());
    }

    #[test]
    fn test_alignment_map_partial() {
        let mut map = AlignmentMap::new(4, 3);
        map.add(0, 0);
        map.add(2, 1);
        assert_eq!(map.aligned_count(), 2);
        assert_eq!(map.unaligned_original(), vec![1, 3]);
        assert_eq!(map.unaligned_transformed(), vec![2]);
        assert!((map.original_coverage() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_alignment_monotonic() {
        let mut map = AlignmentMap::new(3, 3);
        map.add(0, 0);
        map.add(1, 1);
        map.add(2, 2);
        assert!(map.is_monotonic());
    }

    #[test]
    fn test_alignment_non_monotonic() {
        let mut map = AlignmentMap::new(3, 3);
        map.add(0, 2);
        map.add(1, 0);
        map.add(2, 1);
        assert!(!map.is_monotonic());
    }

    #[test]
    fn test_lemma_aligner_identical() {
        let aligner = LemmaAligner::new();
        let orig = make_sentence_with_lemmas(&[("The", "the"), ("cat", "cat"), ("sat", "sit")]);
        let trans = make_sentence_with_lemmas(&[("The", "the"), ("cat", "cat"), ("sat", "sit")]);
        let result = aligner.align_by_lemma(&orig, &trans);
        assert!(result.is_ok());
        let map = result.unwrap();
        assert_eq!(map.aligned_count(), 3);
    }

    #[test]
    fn test_lemma_aligner_synonym_swap() {
        let aligner = LemmaAligner::new().with_min_coverage(0.3);
        let orig = make_sentence_with_lemmas(&[("The", "the"), ("cat", "cat"), ("sat", "sit")]);
        let trans = make_sentence_with_lemmas(&[("The", "the"), ("dog", "dog"), ("sat", "sit")]);
        let result = aligner.align_by_lemma(&orig, &trans);
        assert!(result.is_ok());
        let map = result.unwrap();
        assert_eq!(map.aligned_count(), 2); // "The" and "sat" match, "cat"→"dog" doesn't
        assert_eq!(map.get_transformed(0), Some(0));
        assert_eq!(map.get_transformed(2), Some(2));
        assert_eq!(map.get_transformed(1), None);
    }

    #[test]
    fn test_surface_form_aligner() {
        let aligner = LemmaAligner::new();
        let orig = make_sentence(&["the", "cat", "sat"]);
        let trans = make_sentence(&["the", "cat", "sat"]);
        let result = aligner.align_by_surface_form(&orig, &trans);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().aligned_count(), 3);
    }

    #[test]
    fn test_positional_alignment() {
        let aligner = LemmaAligner::new();
        let orig = make_sentence(&["a", "b", "c"]);
        let trans = make_sentence(&["x", "y"]);
        let map = aligner.align_by_position(&orig, &trans);
        assert_eq!(map.aligned_count(), 2);
        assert_eq!(map.get_transformed(0), Some(0));
        assert_eq!(map.get_transformed(1), Some(1));
        assert_eq!(map.get_transformed(2), None);
    }

    #[test]
    fn test_multi_strategy_align() {
        let aligner = LemmaAligner::new();
        let orig = make_sentence_with_lemmas(&[("Running", "run"), ("fast", "fast")]);
        let trans = make_sentence_with_lemmas(&[("Ran", "run"), ("quickly", "quick")]);
        let (map, quality) = aligner.align(&orig, &trans);
        assert!(map.aligned_count() >= 1); // At least "run" matches
        assert!(quality.coverage > 0.0);
    }

    #[test]
    fn test_transformation_specific_aligner() {
        let mut explicit = HashMap::new();
        explicit.insert(0, 0); // "the" → "the"
        explicit.insert(1, 2); // "cat" → "big" moved to pos 2
        let aligner = TransformationSpecificAligner::from_map(explicit);
        let orig = make_sentence(&["the", "cat", "sat"]);
        let trans = make_sentence(&["the", "sat", "cat"]);
        let map = aligner.align(&orig, &trans);
        assert_eq!(map.get_transformed(0), Some(0));
        assert_eq!(map.get_transformed(1), Some(2));
    }

    #[test]
    fn test_tree_alignment_identical() {
        let mut orig = make_sentence(&["the", "cat", "sat"]);
        orig.dependency_edges = vec![
            DependencyEdge {
                head_index: 2,
                dependent_index: 2,
                relation: DependencyRelation::Root,
            },
            DependencyEdge {
                head_index: 2,
                dependent_index: 1,
                relation: DependencyRelation::Nsubj,
            },
            DependencyEdge {
                head_index: 1,
                dependent_index: 0,
                relation: DependencyRelation::Det,
            },
        ];
        let trans = orig.clone();

        let mut token_align = AlignmentMap::new(3, 3);
        token_align.add(0, 0);
        token_align.add(1, 1);
        token_align.add(2, 2);

        let tree_align = TreeAlignment::align_trees(&token_align, &orig, &trans);
        assert_eq!(tree_align.edge_matches.len(), 3);
        assert!(tree_align.unmatched_original_edges.is_empty());
    }

    #[test]
    fn test_tree_alignment_partial() {
        let mut orig = make_sentence(&["the", "cat", "sat"]);
        orig.dependency_edges = vec![
            DependencyEdge {
                head_index: 2,
                dependent_index: 1,
                relation: DependencyRelation::Nsubj,
            },
            DependencyEdge {
                head_index: 1,
                dependent_index: 0,
                relation: DependencyRelation::Det,
            },
        ];
        let mut trans = make_sentence(&["a", "dog", "ran"]);
        trans.dependency_edges = vec![
            DependencyEdge {
                head_index: 2,
                dependent_index: 1,
                relation: DependencyRelation::Nsubj,
            },
            DependencyEdge {
                head_index: 1,
                dependent_index: 0,
                relation: DependencyRelation::Amod,
            },
        ];

        let mut token_align = AlignmentMap::new(3, 3);
        token_align.add(0, 0);
        token_align.add(1, 1);
        token_align.add(2, 2);

        let tree_align = TreeAlignment::align_trees(&token_align, &orig, &trans);
        assert_eq!(tree_align.edge_matches.len(), 1); // Subject edge matches
        assert_eq!(tree_align.unmatched_original_edges.len(), 1);
    }

    #[test]
    fn test_alignment_quality_high() {
        let mut map = AlignmentMap::new(3, 3);
        map.add(0, 0);
        map.add(1, 1);
        map.add(2, 2);
        let quality = compute_alignment_quality(&map);
        assert!(quality.is_high_quality());
        assert!((quality.coverage - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_alignment_quality_low() {
        let mut map = AlignmentMap::new(10, 10);
        map.add(0, 0);
        let quality = compute_alignment_quality(&map);
        assert!(!quality.is_high_quality());
    }

    #[test]
    fn test_alignment_failure() {
        let failure = AlignmentFailure::new("too few matches", vec![1, 2], vec![3]);
        assert_eq!(failure.unmatched_original.len(), 2);
        assert_eq!(failure.unmatched_transformed.len(), 1);
        assert!(failure.partial_alignment.is_none());
    }

    #[test]
    fn test_tree_alignment_edge_recall() {
        let tree_align = TreeAlignment {
            node_map: HashMap::new(),
            edge_matches: vec![(
                DependencyEdge {
                    head_index: 0,
                    dependent_index: 1,
                    relation: DependencyRelation::Nsubj,
                },
                DependencyEdge {
                    head_index: 0,
                    dependent_index: 1,
                    relation: DependencyRelation::Nsubj,
                },
            )],
            unmatched_original_edges: vec![DependencyEdge {
                head_index: 0,
                dependent_index: 2,
                relation: DependencyRelation::Dobj,
            }],
            unmatched_transformed_edges: Vec::new(),
        };
        assert!((tree_align.edge_recall(2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_lemma_aligner_with_insertion() {
        let aligner = LemmaAligner::new().with_min_coverage(0.3);
        let orig = make_sentence_with_lemmas(&[("the", "the"), ("cat", "cat")]);
        let trans = make_sentence_with_lemmas(&[
            ("the", "the"),
            ("big", "big"),
            ("cat", "cat"),
        ]);
        let result = aligner.align_by_lemma(&orig, &trans);
        assert!(result.is_ok());
        let map = result.unwrap();
        assert_eq!(map.get_transformed(0), Some(0)); // "the" → "the"
        assert_eq!(map.get_transformed(1), Some(2)); // "cat" → "cat" (at index 2)
    }

    #[test]
    fn test_empty_sentence_alignment() {
        let aligner = LemmaAligner::new().with_min_coverage(0.0);
        let orig = make_sentence(&[]);
        let trans = make_sentence(&[]);
        let (map, quality) = aligner.align(&orig, &trans);
        assert_eq!(map.aligned_count(), 0);
        assert!((quality.coverage - 1.0).abs() < 1e-9);
    }
}
