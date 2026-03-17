//! Extended IR operations: conversion, alignment, diffing, and serialisation.

use shared_types::{
    DependencyEdge, DependencyRelation, EntityLabel, EntitySpan, IRType,
    IntermediateRepresentation, LocalizerError, PosTag, Result, Sentence, StageId, Token,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── IRConverter trait ───────────────────────────────────────────────────────

/// Convert between different IR formats.
pub trait IRConverter {
    fn can_convert(&self, from: &IRType, to: &IRType) -> bool;
    fn convert(
        &self,
        ir: &IntermediateRepresentation,
        target: &IRType,
    ) -> Result<IntermediateRepresentation>;
}

/// Default converter supporting common conversions.
pub struct DefaultIRConverter;

impl IRConverter for DefaultIRConverter {
    fn can_convert(&self, from: &IRType, to: &IRType) -> bool {
        matches!(
            (from, to),
            (IRType::RawText, IRType::Tokenized)
                | (IRType::Tokenized, IRType::RawText)
                | (IRType::PosTagged, IRType::Tokenized)
                | (IRType::Parsed, IRType::PosTagged)
                | (IRType::EntityAnnotated, IRType::PosTagged)
        )
    }

    fn convert(
        &self,
        ir: &IntermediateRepresentation,
        target: &IRType,
    ) -> Result<IntermediateRepresentation> {
        if !self.can_convert(&ir.ir_type, target) {
            return Err(LocalizerError::pipeline("internal", format!(
                "Cannot convert {:?} to {:?}",
                ir.ir_type, target
            )));
        }
        let sentence = ir.sentence.clone();
        Ok(IntermediateRepresentation::new(target.clone(), sentence))
    }
}

// ── TokenSequenceIR ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSequenceIR {
    pub tokens: Vec<Token>,
}

impl TokenSequenceIR {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens }
    }

    pub fn from_ir(ir: &IntermediateRepresentation) -> Self {
        Self {
            tokens: ir.sentence.tokens.clone(),
        }
    }

    pub fn to_text(&self) -> String {
        self.tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn from_text(text: &str) -> Self {
        let tokens: Vec<Token> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Token::new(w, i))
            .collect();
        Self { tokens }
    }

    /// Align with another token sequence using LCS-based alignment.
    pub fn align_with(&self, other: &TokenSequenceIR) -> Vec<(Option<usize>, Option<usize>)> {
        let a: Vec<&str> = self.tokens.iter().map(|t| t.text.as_str()).collect();
        let b: Vec<&str> = other.tokens.iter().map(|t| t.text.as_str()).collect();
        lcs_alignment(&a, &b)
    }

    /// Token-level diff: returns (removed_indices, added_indices, changed_indices).
    pub fn diff_tokens(&self, other: &TokenSequenceIR) -> TokenDiff {
        let alignment = self.align_with(other);
        let mut removed = Vec::new();
        let mut added = Vec::new();
        let mut changed = Vec::new();
        for &(a, b) in &alignment {
            match (a, b) {
                (Some(ai), None) => removed.push(ai),
                (None, Some(bi)) => added.push(bi),
                (Some(ai), Some(bi)) => {
                    if self.tokens[ai].text != other.tokens[bi].text {
                        changed.push((ai, bi));
                    }
                }
                _ => {}
            }
        }
        TokenDiff { removed, added, changed }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDiff {
    pub removed: Vec<usize>,
    pub added: Vec<usize>,
    pub changed: Vec<(usize, usize)>,
}

// ── DependencyTreeIR ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyTreeIR {
    pub tokens: Vec<Token>,
    pub edges: Vec<DependencyEdge>,
}

impl DependencyTreeIR {
    pub fn new(tokens: Vec<Token>, edges: Vec<DependencyEdge>) -> Self {
        Self { tokens, edges }
    }

    pub fn from_ir(ir: &IntermediateRepresentation) -> Self {
        Self {
            tokens: ir.sentence.tokens.clone(),
            edges: ir.sentence.dependency_edges.clone(),
        }
    }

    /// Serialise to CoNLL-like format.
    pub fn to_conll(&self) -> String {
        let mut heads: HashMap<usize, (usize, &DependencyRelation)> = HashMap::new();
        for e in &self.edges {
            heads.insert(e.dependent_index, (e.head_index, &e.relation));
        }
        let mut lines = Vec::new();
        for (i, tok) in self.tokens.iter().enumerate() {
            let idx = i + 1;
            let (head, rel) = heads.get(&idx).map(|(h, r)| (*h, format!("{:?}", r))).unwrap_or((0, "dep".into()));
            let pos = tok.pos_tag.map(|p| format!("{:?}", p)).unwrap_or_else(|| "_".into());
            let lemma = tok.lemma.as_deref().unwrap_or("_");
            lines.push(format!("{}\t{}\t{}\t{}\t{}\t{}", idx, tok.text, lemma, pos, head, rel));
        }
        lines.join("\n")
    }

    /// Parse from CoNLL-like format.
    pub fn from_conll(text: &str) -> Result<Self> {
        let mut tokens = Vec::new();
        let mut edges = Vec::new();
        for line in text.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 6 {
                continue;
            }
            let idx: usize = parts[0]
                .parse()
                .map_err(|_| LocalizerError::PipelineError { stage: "internal".into(), message: "Bad index".into(), source: None })?;
            let word = parts[1];
            let head_index: usize = parts[4]
                .parse()
                .map_err(|_| LocalizerError::PipelineError { stage: "internal".into(), message: "Bad head".into(), source: None })?;
            let rel_str = parts[5];
            let tok = Token::new(word, idx - 1);
            tokens.push(tok);
            edges.push(DependencyEdge {
                head_index,
                dependent_index: idx,
                relation: parse_dep_relation(rel_str),
            });
        }
        Ok(Self { tokens, edges })
    }

    /// Get the subtree rooted at a given token index.
    pub fn subtree_at(&self, root_idx: usize) -> Vec<usize> {
        let children = build_children_map(&self.edges);
        let mut result = Vec::new();
        let mut stack = vec![root_idx];
        while let Some(node) = stack.pop() {
            result.push(node);
            if let Some(ch) = children.get(&node) {
                for &c in ch {
                    stack.push(c);
                }
            }
        }
        result.sort_unstable();
        result
    }

    /// Find the path between two token indices.
    pub fn path_between(&self, a: usize, b: usize) -> Vec<usize> {
        let heads = build_head_map(&self.edges);
        let path_a = path_to_root(a, &heads);
        let path_b = path_to_root(b, &heads);
        // Find LCA
        let set_a: HashSet<usize> = path_a.iter().copied().collect();
        let mut lca = 0;
        for &node in &path_b {
            if set_a.contains(&node) {
                lca = node;
                break;
            }
        }
        let mut path = Vec::new();
        for &n in &path_a {
            path.push(n);
            if n == lca {
                break;
            }
        }
        let mut from_b = Vec::new();
        for &n in &path_b {
            if n == lca {
                break;
            }
            from_b.push(n);
        }
        from_b.reverse();
        path.extend(from_b);
        path
    }

    /// Check if tree is projective.
    pub fn is_projective(&self) -> bool {
        crate::parser_model::RuleBasedParser::is_projective(&self.edges)
    }
}

// ── EntityAnnotationIR ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAnnotationIR {
    pub entities: Vec<EntitySpan>,
}

impl EntityAnnotationIR {
    pub fn new(entities: Vec<EntitySpan>) -> Self {
        Self { entities }
    }

    pub fn from_ir(ir: &IntermediateRepresentation) -> Self {
        Self {
            entities: ir.sentence.entities.clone(),
        }
    }

    /// Return entities whose spans overlap with the given range.
    pub fn overlapping_entities(&self, start: usize, end: usize) -> Vec<&EntitySpan> {
        self.entities
            .iter()
            .filter(|e| e.start < end && e.end > start)
            .collect()
    }

    /// Merge overlapping or adjacent entities of the same type.
    pub fn merge_entities(&self) -> Self {
        let mut sorted = self.entities.clone();
        sorted.sort_by_key(|e| (e.start, e.end));
        let mut merged: Vec<EntitySpan> = Vec::new();
        for ent in sorted {
            if let Some(last) = merged.last_mut() {
                if last.label == ent.label && last.end >= ent.start {
                    last.end = last.end.max(ent.end);
                    last.text = format!("{} {}", last.text, ent.text);
                    continue;
                }
            }
            merged.push(ent);
        }
        Self { entities: merged }
    }

    /// Keep only entities with the given label.
    pub fn filter_by_label(&self, label: EntityLabel) -> Self {
        Self {
            entities: self.entities.iter().filter(|e| e.label == label).cloned().collect(),
        }
    }
}

// ── IRAligner ───────────────────────────────────────────────────────────────

/// Aligns IRs between original and transformed executions at the lemma level.
#[derive(Debug, Clone)]
pub struct IRAligner;

impl IRAligner {
    /// Align at the lemma (or text) level.
    pub fn align_lemmas(original: &[Token], transformed: &[Token]) -> LemmaAlignment {
        let orig_lemmas: Vec<String> = original
            .iter()
            .map(|t| t.lemma.clone().unwrap_or_else(|| t.text.to_lowercase()))
            .collect();
        let trans_lemmas: Vec<String> = transformed
            .iter()
            .map(|t| t.lemma.clone().unwrap_or_else(|| t.text.to_lowercase()))
            .collect();
        let pairs = lcs_alignment(
            &orig_lemmas.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &trans_lemmas.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );
        LemmaAlignment {
            pairs,
            original_count: original.len(),
            transformed_count: transformed.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LemmaAlignment {
    pub pairs: Vec<(Option<usize>, Option<usize>)>,
    pub original_count: usize,
    pub transformed_count: usize,
}

impl LemmaAlignment {
    /// Fraction of original tokens that were matched.
    pub fn coverage(&self) -> f64 {
        if self.original_count == 0 {
            return 1.0;
        }
        let matched = self.pairs.iter().filter(|(a, b)| a.is_some() && b.is_some()).count();
        matched as f64 / self.original_count as f64
    }
}

// ── IRDiff ──────────────────────────────────────────────────────────────────

/// Represents the difference between two IRs of the same type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRDiff {
    pub ir_type: IRType,
    pub summary: String,
    pub token_diff: Option<TokenDiff>,
    pub entity_diff: Option<EntityDiff>,
    pub dependency_diff: Option<DependencyDiff>,
    pub pos_diff: Option<PosDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDiff {
    pub added: Vec<EntitySpan>,
    pub removed: Vec<EntitySpan>,
    pub label_changes: Vec<(EntitySpan, EntitySpan)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyDiff {
    pub added_edges: Vec<DependencyEdge>,
    pub removed_edges: Vec<DependencyEdge>,
    pub changed_labels: Vec<(DependencyEdge, DependencyEdge)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosDiff {
    pub changes: Vec<(usize, PosTag, PosTag)>,
}

/// Compute the diff between two IRs.
pub fn compute_ir_diff(a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> IRDiff {
    let token_diff = compute_token_diff(&a.sentence, &b.sentence);
    let entity_diff = compute_entity_diff(&a.sentence, &b.sentence);
    let dependency_diff = compute_dependency_diff(&a.sentence, &b.sentence);
    let pos_diff = compute_pos_diff(&a.sentence, &b.sentence);

    let n_tok_changes = token_diff.as_ref().map_or(0, |d| d.removed.len() + d.added.len() + d.changed.len());
    let n_ent_changes = entity_diff.as_ref().map_or(0, |d| d.added.len() + d.removed.len() + d.label_changes.len());
    let n_dep_changes = dependency_diff.as_ref().map_or(0, |d| d.added_edges.len() + d.removed_edges.len());
    let n_pos_changes = pos_diff.as_ref().map_or(0, |d| d.changes.len());

    IRDiff {
        ir_type: a.ir_type.clone(),
        summary: format!(
            "tok:{} ent:{} dep:{} pos:{}",
            n_tok_changes, n_ent_changes, n_dep_changes, n_pos_changes
        ),
        token_diff,
        entity_diff,
        dependency_diff,
        pos_diff,
    }
}

fn compute_token_diff(a: &Sentence, b: &Sentence) -> Option<TokenDiff> {
    if a.tokens.is_empty() && b.tokens.is_empty() {
        return None;
    }
    let ir_a = TokenSequenceIR::new(a.tokens.clone());
    let ir_b = TokenSequenceIR::new(b.tokens.clone());
    Some(ir_a.diff_tokens(&ir_b))
}

fn compute_entity_diff(a: &Sentence, b: &Sentence) -> Option<EntityDiff> {
    if a.entities.is_empty() && b.entities.is_empty() {
        return None;
    }
    let set_a: HashSet<String> = a.entities.iter().map(|e| format!("{}:{}:{}", e.start, e.end, e.text)).collect();
    let set_b: HashSet<String> = b.entities.iter().map(|e| format!("{}:{}:{}", e.start, e.end, e.text)).collect();

    let added: Vec<EntitySpan> = b.entities.iter().filter(|e| !set_a.contains(&format!("{}:{}:{}", e.start, e.end, e.text))).cloned().collect();
    let removed: Vec<EntitySpan> = a.entities.iter().filter(|e| !set_b.contains(&format!("{}:{}:{}", e.start, e.end, e.text))).cloned().collect();

    // Label changes: same span text but different label
    let mut label_changes = Vec::new();
    for ea in &a.entities {
        for eb in &b.entities {
            if ea.text == eb.text && ea.start == eb.start && ea.label != eb.label {
                label_changes.push((ea.clone(), eb.clone()));
            }
        }
    }

    Some(EntityDiff { added, removed, label_changes })
}

fn compute_dependency_diff(a: &Sentence, b: &Sentence) -> Option<DependencyDiff> {
    if a.dependency_edges.is_empty() && b.dependency_edges.is_empty() {
        return None;
    }
    let key = |e: &DependencyEdge| (e.head_index, e.dependent_index);
    let a_map: HashMap<(usize, usize), &DependencyEdge> = a.dependency_edges.iter().map(|e| (key(e), e)).collect();
    let b_map: HashMap<(usize, usize), &DependencyEdge> = b.dependency_edges.iter().map(|e| (key(e), e)).collect();

    let added: Vec<DependencyEdge> = b.dependency_edges.iter().filter(|e| !a_map.contains_key(&key(e))).cloned().collect();
    let removed: Vec<DependencyEdge> = a.dependency_edges.iter().filter(|e| !b_map.contains_key(&key(e))).cloned().collect();

    let mut changed_labels = Vec::new();
    for (k, ea) in &a_map {
        if let Some(eb) = b_map.get(k) {
            if ea.relation != eb.relation {
                changed_labels.push(((*ea).clone(), (*eb).clone()));
            }
        }
    }

    Some(DependencyDiff { added_edges: added, removed_edges: removed, changed_labels })
}

fn compute_pos_diff(a: &Sentence, b: &Sentence) -> Option<PosDiff> {
    let min_len = a.tokens.len().min(b.tokens.len());
    if min_len == 0 {
        return None;
    }
    let mut changes = Vec::new();
    for i in 0..min_len {
        if let (Some(pa), Some(pb)) = (a.tokens[i].pos_tag, b.tokens[i].pos_tag) {
            if pa != pb {
                changes.push((i, pa, pb));
            }
        }
    }
    if changes.is_empty() {
        None
    } else {
        Some(PosDiff { changes })
    }
}

// ── IRSerializer ────────────────────────────────────────────────────────────

pub struct IRSerializer;

impl IRSerializer {
    pub fn to_json(ir: &IntermediateRepresentation) -> Result<String> {
        serde_json::to_string_pretty(ir)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Serialization failed: {}", e), source: None })
    }

    pub fn from_json(json: &str) -> Result<IntermediateRepresentation> {
        serde_json::from_str(json)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Deserialization failed: {}", e), source: None })
    }

    pub fn token_seq_to_json(ts: &TokenSequenceIR) -> Result<String> {
        serde_json::to_string_pretty(ts)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Serialization failed: {}", e), source: None })
    }

    pub fn token_seq_from_json(json: &str) -> Result<TokenSequenceIR> {
        serde_json::from_str(json)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Deserialization failed: {}", e), source: None })
    }

    pub fn dep_tree_to_json(dt: &DependencyTreeIR) -> Result<String> {
        serde_json::to_string_pretty(dt)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Serialization failed: {}", e), source: None })
    }

    pub fn dep_tree_from_json(json: &str) -> Result<DependencyTreeIR> {
        serde_json::from_str(json)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Deserialization failed: {}", e), source: None })
    }

    pub fn entity_ir_to_json(eir: &EntityAnnotationIR) -> Result<String> {
        serde_json::to_string_pretty(eir)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Serialization failed: {}", e), source: None })
    }

    pub fn entity_ir_from_json(json: &str) -> Result<EntityAnnotationIR> {
        serde_json::from_str(json)
            .map_err(|e| LocalizerError::PipelineError { stage: "internal".into(), message: format!("Deserialization failed: {}", e), source: None })
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn lcs_alignment<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(Option<usize>, Option<usize>)> {
    let n = a.len();
    let m = b.len();
    // LCS table
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    // Backtrack
    let mut alignment: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let (mut i, mut j) = (n, m);
    let mut pairs = Vec::new();
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            pairs.push((Some(i - 1), Some(j - 1)));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            pairs.push((Some(i - 1), None));
            i -= 1;
        } else {
            pairs.push((None, Some(j - 1)));
            j -= 1;
        }
    }
    while i > 0 {
        pairs.push((Some(i - 1), None));
        i -= 1;
    }
    while j > 0 {
        pairs.push((None, Some(j - 1)));
        j -= 1;
    }
    pairs.reverse();
    pairs
}

fn build_children_map(edges: &[DependencyEdge]) -> HashMap<usize, Vec<usize>> {
    let mut m: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in edges {
        m.entry(e.head_index).or_default().push(e.dependent_index);
    }
    m
}

fn build_head_map(edges: &[DependencyEdge]) -> HashMap<usize, usize> {
    edges.iter().map(|e| (e.dependent_index, e.head_index)).collect()
}

fn path_to_root(node: usize, heads: &HashMap<usize, usize>) -> Vec<usize> {
    let mut path = vec![node];
    let mut cur = node;
    let mut visited = HashSet::new();
    while let Some(&h) = heads.get(&cur) {
        if !visited.insert(h) {
            break;
        }
        path.push(h);
        if h == 0 {
            break;
        }
        cur = h;
    }
    path
}

fn parse_dep_relation(s: &str) -> DependencyRelation {
    match s.to_lowercase().as_str() {
        "root" => DependencyRelation::Root,
        "subject" | "nsubj" => DependencyRelation::Nsubj,
        "object" | "dobj" => DependencyRelation::Dobj,
        "indirectobject" | "iobj" => DependencyRelation::Iobj,
        "modifier" | "amod" | "advmod" => DependencyRelation::Amod,
        "complement" | "xcomp" => DependencyRelation::Ccomp,
        "determiner" | "det" => DependencyRelation::Det,
        "conjunction" | "cc" => DependencyRelation::Conj,
        "preposition" | "prep" => DependencyRelation::Prep,
        "auxiliary" | "aux" => DependencyRelation::Aux,
        "negation" | "neg" => DependencyRelation::Advmod,
        "punctuation" | "punct" => DependencyRelation::Punct,
        _ => DependencyRelation::Other,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sentence(words: &[&str]) -> Sentence {
        Sentence::from_text(&words.join(" "))
    }

    #[test]
    fn test_token_sequence_to_text() {
        let ts = TokenSequenceIR::from_text("Hello world");
        assert_eq!(ts.to_text(), "Hello world");
    }

    #[test]
    fn test_token_sequence_diff_identical() {
        let a = TokenSequenceIR::from_text("The cat sat");
        let b = TokenSequenceIR::from_text("The cat sat");
        let diff = a.diff_tokens(&b);
        assert!(diff.removed.is_empty());
        assert!(diff.added.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_token_sequence_diff_insertion() {
        let a = TokenSequenceIR::from_text("The cat");
        let b = TokenSequenceIR::from_text("The big cat");
        let diff = a.diff_tokens(&b);
        assert!(!diff.added.is_empty() || !diff.changed.is_empty());
    }

    #[test]
    fn test_dep_tree_conll_roundtrip() {
        let tokens = vec![
            Token::new("The", 0, 0, 3),
            Token::new("cat", 1, 4, 7),
            Token::new("sat", 2, 8, 11),
        ];
        let edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 3, relation: DependencyRelation::Root },
            DependencyEdge { head_index: 3, dependent_index: 1, relation: DependencyRelation::Det },
            DependencyEdge { head_index: 3, dependent_index: 2, relation: DependencyRelation::Nsubj },
        ];
        let dt = DependencyTreeIR::new(tokens, edges);
        let conll = dt.to_conll();
        assert!(conll.contains("cat"));
        let parsed = DependencyTreeIR::from_conll(&conll).unwrap();
        assert_eq!(parsed.tokens.len(), 3);
    }

    #[test]
    fn test_subtree() {
        let tokens = vec![
            Token::new("The", 0, 0, 3),
            Token::new("cat", 1, 4, 7),
            Token::new("sat", 2, 8, 11),
        ];
        let edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 3, relation: DependencyRelation::Root },
            DependencyEdge { head_index: 3, dependent_index: 1, relation: DependencyRelation::Det },
            DependencyEdge { head_index: 3, dependent_index: 2, relation: DependencyRelation::Nsubj },
        ];
        let dt = DependencyTreeIR::new(tokens, edges);
        let subtree = dt.subtree_at(3);
        assert!(subtree.contains(&1));
        assert!(subtree.contains(&2));
        assert!(subtree.contains(&3));
    }

    #[test]
    fn test_entity_overlap() {
        let eir = EntityAnnotationIR::new(vec![
            EntitySpan { start: 0, end: 2, text: "New York".into(), label: EntityLabel::Location, confidence: 0.9 },
            EntitySpan { start: 5, end: 6, text: "Google".into(), label: EntityLabel::Organization, confidence: 0.9 },
        ]);
        let overlapping = eir.overlapping_entities(0, 3);
        assert_eq!(overlapping.len(), 1);
        assert_eq!(overlapping[0].label, EntityLabel::Location);
    }

    #[test]
    fn test_entity_filter_by_label() {
        let eir = EntityAnnotationIR::new(vec![
            EntitySpan { start: 0, end: 1, text: "John".into(), label: EntityLabel::Person, confidence: 0.9 },
            EntitySpan { start: 3, end: 4, text: "London".into(), label: EntityLabel::Location, confidence: 0.9 },
        ]);
        let persons = eir.filter_by_label(EntityLabel::Person);
        assert_eq!(persons.entities.len(), 1);
    }

    #[test]
    fn test_lemma_alignment() {
        let orig = vec![Token::new("the", 0, 0, 3), Token::new("cat", 1, 4, 7)];
        let trans = vec![Token::new("the", 0, 0, 3), Token::new("big", 1, 4, 7), Token::new("cat", 2, 8, 11)];
        let alignment = IRAligner::align_lemmas(&orig, &trans);
        assert!(alignment.coverage() > 0.5);
    }

    #[test]
    fn test_ir_diff_identical() {
        let s = make_sentence(&["The", "cat", "sat"]);
        let ir_a = IntermediateRepresentation::new(IRType::Tokenized, s.clone());
        let ir_b = IntermediateRepresentation::new(IRType::Tokenized, s);
        let diff = compute_ir_diff(&ir_a, &ir_b);
        let td = diff.token_diff.unwrap();
        assert!(td.removed.is_empty() && td.added.is_empty() && td.changed.is_empty());
    }

    #[test]
    fn test_ir_serializer_roundtrip() {
        let s = Sentence::from_text("hello");
        let ir = IntermediateRepresentation::new(IRType::RawText, s);
        let json = IRSerializer::to_json(&ir).unwrap();
        let ir2 = IRSerializer::from_json(&json).unwrap();
        assert_eq!(ir.sentence.text, ir2.sentence.text);
    }

    #[test]
    fn test_default_converter() {
        let conv = DefaultIRConverter;
        assert!(conv.can_convert(&IRType::RawText, &IRType::Tokenized));
        assert!(!conv.can_convert(&IRType::RawText, &IRType::FeatureVector));
    }

    #[test]
    fn test_path_between() {
        let tokens = vec![
            Token::new("A", 0, 0, 1),
            Token::new("B", 1, 2, 3),
            Token::new("C", 2, 4, 5),
        ];
        let edges = vec![
            DependencyEdge { head_index: 0, dependent_index: 2, relation: DependencyRelation::Root },
            DependencyEdge { head_index: 2, dependent_index: 1, relation: DependencyRelation::Nsubj },
            DependencyEdge { head_index: 2, dependent_index: 3, relation: DependencyRelation::Dobj },
        ];
        let dt = DependencyTreeIR::new(tokens, edges);
        let path = dt.path_between(1, 3);
        assert!(path.contains(&2)); // LCA
    }
}
