//! Clause type definitions and classification.
//!
//! Identifies clause types (main, subordinate, relative, etc.) from
//! dependency/constituency information and validates their internal structure.

use crate::features::{
    Feature, FeatureBundle, FinitenessValue, MoodValue,
};
use shared_types::{DependencyRelation, PosTag, Sentence};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── ClauseType ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClauseType {
    Main,
    Subordinate,
    Relative,
    Adverbial,
    Nominal,
    Infinitival,
    Gerundive,
    SmallClause,
    Existential,
    Cleft,
    Interrogative,
    Imperative,
    Exclamative,
    Conditional,
    Comparative,
}

impl ClauseType {
    pub fn is_finite(&self) -> bool {
        matches!(
            self,
            ClauseType::Main
                | ClauseType::Subordinate
                | ClauseType::Relative
                | ClauseType::Adverbial
                | ClauseType::Nominal
                | ClauseType::Existential
                | ClauseType::Cleft
                | ClauseType::Interrogative
                | ClauseType::Imperative
                | ClauseType::Exclamative
                | ClauseType::Conditional
                | ClauseType::Comparative
        )
    }

    pub fn is_embedded(&self) -> bool {
        !matches!(self, ClauseType::Main)
    }

    pub fn is_non_finite(&self) -> bool {
        matches!(
            self,
            ClauseType::Infinitival | ClauseType::Gerundive | ClauseType::SmallClause
        )
    }
}

impl fmt::Display for ClauseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

// ── ClauseProperties ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClauseProperties {
    pub clause_type: ClauseType,
    pub finiteness: FinitenessValue,
    pub mood: Option<MoodValue>,
    pub has_complementizer: bool,
    pub has_subject: bool,
    pub has_overt_verb: bool,
}

impl ClauseProperties {
    pub fn new(clause_type: ClauseType) -> Self {
        let finiteness = if clause_type.is_finite() {
            FinitenessValue::Finite
        } else {
            match clause_type {
                ClauseType::Infinitival => FinitenessValue::Infinitive,
                ClauseType::Gerundive => FinitenessValue::Gerund,
                _ => FinitenessValue::NonFinite,
            }
        };
        Self {
            clause_type,
            finiteness,
            mood: None,
            has_complementizer: false,
            has_subject: true,
            has_overt_verb: true,
        }
    }
}

// ── ClauseRelation ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClauseRelation {
    Coordination,
    Subordination,
    Complementation,
    Relativization,
    Adjunction,
}

impl fmt::Display for ClauseRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

// ── ClauseInfo (internal struct for clause instances) ────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClauseInfo {
    /// Index of the clause's predicate/verb in the sentence token list.
    pub head_index: usize,
    pub properties: ClauseProperties,
    /// Token indices that belong to this clause.
    pub token_span: (usize, usize),
}

// ── ClauseStructure ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ClauseStructure {
    pub clauses: Vec<ClauseInfo>,
    pub relations: Vec<(usize, usize, ClauseRelation)>,
}

impl ClauseStructure {
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            relations: Vec::new(),
        }
    }
}

impl Default for ClauseStructure {
    fn default() -> Self {
        Self::new()
    }
}

// ── ClauseClassifier ────────────────────────────────────────────────────────

pub struct ClauseClassifier;

impl ClauseClassifier {
    /// Classify the clause anchored at a given verb index.
    pub fn classify_clause(sentence: &Sentence, verb_index: usize) -> ClauseProperties {
        let head_edge = sentence
            .dependency_edges
            .iter()
            .find(|e| e.dependent_index == verb_index);

        let is_root = head_edge.map_or(false, |e| e.relation == DependencyRelation::Root);

        // Check for complementiser (mark)
        let has_comp = sentence.dependency_edges.iter().any(|e| {
            e.head_index == verb_index && e.relation == DependencyRelation::Mark
        });

        // Check for subject
        let has_subj = sentence.dependency_edges.iter().any(|e| {
            e.head_index == verb_index && e.relation == DependencyRelation::Nsubj
        });

        // Check for expletive
        let has_expl = sentence.dependency_edges.iter().any(|e| {
            e.head_index == verb_index && e.relation == DependencyRelation::Expl
        });

        // Detect relative clause
        let is_relcl = head_edge.map_or(false, |e| e.relation == DependencyRelation::Relcl);

        // Detect adverbial clause
        let is_advcl = head_edge.map_or(false, |e| e.relation == DependencyRelation::Advcl);

        // Detect complement clause
        let is_ccomp = head_edge.map_or(false, |e| e.relation == DependencyRelation::Ccomp);
        let is_xcomp = head_edge.map_or(false, |e| e.relation == DependencyRelation::Xcomp);

        // Detect coordination
        let is_conj = head_edge.map_or(false, |e| e.relation == DependencyRelation::Conj);

        // Check verb form for finiteness
        let verb_token = &sentence.tokens[verb_index];
        let verb_low = verb_token.text.to_lowercase();

        let is_gerund = verb_low.ends_with("ing")
            || verb_token.features.get("VerbForm").map_or(false, |v| v == "Ger");
        let is_infinitive = verb_index > 0
            && sentence.tokens.get(verb_index - 1).map_or(false, |t| {
                t.text.to_lowercase() == "to"
            });
        let is_imperative = !has_subj && is_root && verb_token.pos_tag == Some(PosTag::Verb);

        // Check for question mark
        let is_interrogative = sentence
            .tokens
            .last()
            .map_or(false, |t| t.text == "?");

        // Check for existential "there"
        let is_existential = has_expl
            && sentence.tokens.iter().any(|t| t.text.to_lowercase() == "there");

        // Determine clause type
        let clause_type = if is_relcl {
            ClauseType::Relative
        } else if is_advcl {
            if has_comp {
                let comp_idx = sentence.dependency_edges.iter()
                    .find(|e| e.head_index == verb_index && e.relation == DependencyRelation::Mark)
                    .map(|e| e.dependent_index);
                if let Some(ci) = comp_idx {
                    let comp_text = sentence.tokens[ci].text.to_lowercase();
                    if comp_text == "if" || comp_text == "unless" {
                        ClauseType::Conditional
                    } else if comp_text == "than" || comp_text == "as" {
                        ClauseType::Comparative
                    } else {
                        ClauseType::Adverbial
                    }
                } else {
                    ClauseType::Adverbial
                }
            } else {
                ClauseType::Adverbial
            }
        } else if is_ccomp {
            ClauseType::Nominal
        } else if is_xcomp {
            if is_gerund {
                ClauseType::Gerundive
            } else if is_infinitive {
                ClauseType::Infinitival
            } else if !has_subj {
                ClauseType::SmallClause
            } else {
                ClauseType::Subordinate
            }
        } else if is_root {
            if is_existential {
                ClauseType::Existential
            } else if is_interrogative {
                ClauseType::Interrogative
            } else if is_imperative {
                ClauseType::Imperative
            } else {
                ClauseType::Main
            }
        } else if is_conj {
            // Coordinated clause — treat as same type as parent.
            ClauseType::Main
        } else {
            ClauseType::Subordinate
        };

        let finiteness = if is_gerund {
            FinitenessValue::Gerund
        } else if is_infinitive {
            FinitenessValue::Infinitive
        } else if clause_type.is_non_finite() {
            FinitenessValue::NonFinite
        } else {
            FinitenessValue::Finite
        };

        let mood = if clause_type == ClauseType::Imperative {
            Some(MoodValue::Imperative)
        } else if clause_type == ClauseType::Interrogative {
            Some(MoodValue::Interrogative)
        } else if clause_type.is_finite() {
            Some(MoodValue::Indicative)
        } else {
            None
        };

        ClauseProperties {
            clause_type,
            finiteness,
            mood,
            has_complementizer: has_comp,
            has_subject: has_subj || has_expl,
            has_overt_verb: true,
        }
    }

    /// Find all clauses in a sentence.
    pub fn find_clauses(sentence: &Sentence) -> Vec<ClauseInfo> {
        let mut clauses = Vec::new();
        // Every verb/auxiliary that heads dependents can anchor a clause.
        let verb_indices: Vec<usize> = sentence
            .tokens
            .iter()
            .filter(|t| matches!(t.pos_tag, Some(PosTag::Verb) | Some(PosTag::Aux)))
            .map(|t| t.index)
            .collect();

        for vi in &verb_indices {
            // Skip auxiliaries that are dependents of another verb.
            let is_aux_dep = sentence.dependency_edges.iter().any(|e| {
                e.dependent_index == *vi && e.relation == DependencyRelation::Aux
            });
            if is_aux_dep {
                continue;
            }
            let props = Self::classify_clause(sentence, *vi);
            let span_start = *vi;
            let span_end = sentence
                .dependents_of(*vi)
                .into_iter()
                .max()
                .unwrap_or(*vi)
                + 1;

            clauses.push(ClauseInfo {
                head_index: *vi,
                properties: props,
                token_span: (span_start, span_end.min(sentence.tokens.len())),
            });
        }
        clauses
    }
}

// ── Clause relation classification ──────────────────────────────────────────

/// Determine how two clauses are related.
pub fn classify_clause_relation(
    sentence: &Sentence,
    clause1_head_index: usize,
    clause2_head_index: usize,
) -> Option<ClauseRelation> {
    for edge in &sentence.dependency_edges {
        if edge.head_index == clause1_head_index && edge.dependent_index == clause2_head_index {
            return Some(match edge.relation {
                DependencyRelation::Conj => ClauseRelation::Coordination,
                DependencyRelation::Ccomp => ClauseRelation::Complementation,
                DependencyRelation::Xcomp => ClauseRelation::Complementation,
                DependencyRelation::Relcl => ClauseRelation::Relativization,
                DependencyRelation::Advcl => ClauseRelation::Adjunction,
                DependencyRelation::Parataxis => ClauseRelation::Coordination,
                _ => ClauseRelation::Subordination,
            });
        }
        if edge.head_index == clause2_head_index && edge.dependent_index == clause1_head_index {
            return Some(match edge.relation {
                DependencyRelation::Conj => ClauseRelation::Coordination,
                _ => ClauseRelation::Subordination,
            });
        }
    }
    None
}

/// Compute the maximum embedding depth of the sentence.
pub fn compute_embedding_depth(sentence: &Sentence) -> u32 {
    let clause_rels = [
        DependencyRelation::Relcl,
        DependencyRelation::Advcl,
        DependencyRelation::Ccomp,
        DependencyRelation::Xcomp,
    ];

    let root = sentence.root_index().unwrap_or(0);
    let mut max_depth: u32 = 0;
    let mut stack: Vec<(usize, u32)> = vec![(root, 0)];
    let mut visited = vec![false; sentence.tokens.len()];

    while let Some((node, depth)) = stack.pop() {
        if node >= visited.len() || visited[node] {
            continue;
        }
        visited[node] = true;
        if depth > max_depth {
            max_depth = depth;
        }
        for edge in &sentence.dependency_edges {
            if edge.head_index == node {
                let is_clause = clause_rels.contains(&edge.relation);
                stack.push((
                    edge.dependent_index,
                    if is_clause { depth + 1 } else { depth },
                ));
            }
        }
    }
    max_depth
}

/// Find the main (root) clause.
pub fn find_main_clause(sentence: &Sentence) -> Option<ClauseInfo> {
    let clauses = ClauseClassifier::find_clauses(sentence);
    clauses
        .into_iter()
        .find(|c| c.properties.clause_type == ClauseType::Main)
}

/// Find all embedded clauses.
pub fn find_embedded_clauses(sentence: &Sentence) -> Vec<ClauseInfo> {
    ClauseClassifier::find_clauses(sentence)
        .into_iter()
        .filter(|c| c.properties.clause_type.is_embedded())
        .collect()
}

/// Extract a FeatureBundle from clause properties.
pub fn clause_to_features(props: &ClauseProperties) -> FeatureBundle {
    let mut fb = FeatureBundle::new();
    fb.set("Finiteness", Feature::Finiteness(props.finiteness));
    if let Some(m) = props.mood {
        fb.set("Mood", Feature::Mood(m));
    }
    if props.clause_type == ClauseType::Existential {
        fb.set(
            "Definiteness",
            Feature::Definiteness(crate::features::DefinitenessValue::Indefinite),
        );
    }
    fb
}

/// Validate that clause structure is well-formed.
pub fn validate_clause_structure(sentence: &Sentence) -> Vec<String> {
    let mut issues = Vec::new();
    let clauses = ClauseClassifier::find_clauses(sentence);

    for clause in &clauses {
        let props = &clause.properties;
        // Finite clauses should have a subject (except imperatives).
        if props.finiteness == FinitenessValue::Finite
            && !props.has_subject
            && props.clause_type != ClauseType::Imperative
        {
            issues.push(format!(
                "Finite {:?} clause at index {} has no subject",
                props.clause_type, clause.head_index
            ));
        }
        // Non-finite clauses with xcomp should not have a complementiser.
        if props.clause_type == ClauseType::SmallClause && props.has_complementizer {
            issues.push(format!(
                "Small clause at index {} has unexpected complementiser",
                clause.head_index
            ));
        }
    }
    issues
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, Token};

    fn make_sentence(words: &[(&str, PosTag)], edges: Vec<DependencyEdge>) -> Sentence {
        let tokens: Vec<Token> = words
            .iter()
            .enumerate()
            .map(|(i, (w, pos))| Token::new(*w, i).with_pos(*pos))
            .collect();
        Sentence {
            raw_text: words.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" "),
            tokens,
            dependency_edges: edges,
            entities: vec![],
        }
    }

    #[test]
    fn test_classify_main_clause() {
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let props = ClauseClassifier::classify_clause(&s, 1);
        assert_eq!(props.clause_type, ClauseType::Main);
        assert_eq!(props.finiteness, FinitenessValue::Finite);
        assert!(props.has_subject);
    }

    #[test]
    fn test_classify_relative_clause() {
        // "the cat that sleeps" — sleeps is relcl of cat
        let s = make_sentence(
            &[
                ("the", PosTag::Det),
                ("cat", PosTag::Noun),
                ("that", PosTag::Pron),
                ("sleeps", PosTag::Verb),
            ],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Det),
                DependencyEdge::new(1, 3, DependencyRelation::Relcl),
                DependencyEdge::new(3, 2, DependencyRelation::Nsubj),
            ],
        );
        let props = ClauseClassifier::classify_clause(&s, 3);
        assert_eq!(props.clause_type, ClauseType::Relative);
    }

    #[test]
    fn test_classify_interrogative() {
        let s = make_sentence(
            &[("do", PosTag::Aux), ("cats", PosTag::Noun), ("eat", PosTag::Verb), ("?", PosTag::Punct)],
            vec![
                DependencyEdge::new(2, 0, DependencyRelation::Aux),
                DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
                DependencyEdge::new(2, 2, DependencyRelation::Root),
                DependencyEdge::new(2, 3, DependencyRelation::Punct),
            ],
        );
        let props = ClauseClassifier::classify_clause(&s, 2);
        assert_eq!(props.clause_type, ClauseType::Interrogative);
    }

    #[test]
    fn test_find_clauses() {
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let clauses = ClauseClassifier::find_clauses(&s);
        assert_eq!(clauses.len(), 1);
    }

    #[test]
    fn test_compute_embedding_depth_flat() {
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        assert_eq!(compute_embedding_depth(&s), 0);
    }

    #[test]
    fn test_compute_embedding_depth_one() {
        let s = make_sentence(
            &[
                ("I", PosTag::Pron),
                ("think", PosTag::Verb),
                ("cats", PosTag::Noun),
                ("eat", PosTag::Verb),
            ],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
                DependencyEdge::new(1, 3, DependencyRelation::Ccomp),
                DependencyEdge::new(3, 2, DependencyRelation::Nsubj),
            ],
        );
        assert_eq!(compute_embedding_depth(&s), 1);
    }

    #[test]
    fn test_clause_to_features() {
        let props = ClauseProperties::new(ClauseType::Main);
        let fb = clause_to_features(&props);
        assert!(fb.contains("Finiteness"));
    }

    #[test]
    fn test_classify_clause_relation() {
        let s = make_sentence(
            &[
                ("I", PosTag::Pron),
                ("think", PosTag::Verb),
                ("cats", PosTag::Noun),
                ("eat", PosTag::Verb),
            ],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
                DependencyEdge::new(1, 3, DependencyRelation::Ccomp),
                DependencyEdge::new(3, 2, DependencyRelation::Nsubj),
            ],
        );
        let rel = classify_clause_relation(&s, 1, 3);
        assert_eq!(rel, Some(ClauseRelation::Complementation));
    }

    #[test]
    fn test_validate_clause_structure_ok() {
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let issues = validate_clause_structure(&s);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_existential_clause() {
        let s = make_sentence(
            &[("there", PosTag::Pron), ("is", PosTag::Verb), ("a", PosTag::Det), ("cat", PosTag::Noun)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Expl),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
                DependencyEdge::new(3, 2, DependencyRelation::Det),
                DependencyEdge::new(1, 3, DependencyRelation::Nsubj),
            ],
        );
        let props = ClauseClassifier::classify_clause(&s, 1);
        assert_eq!(props.clause_type, ClauseType::Existential);
    }

    #[test]
    fn test_clause_type_helpers() {
        assert!(ClauseType::Main.is_finite());
        assert!(!ClauseType::Main.is_embedded());
        assert!(ClauseType::Relative.is_embedded());
        assert!(ClauseType::Infinitival.is_non_finite());
    }
}
