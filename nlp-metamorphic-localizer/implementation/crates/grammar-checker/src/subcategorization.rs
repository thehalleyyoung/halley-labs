//! Verb subcategorization frame checking.
//!
//! A *subcategorization frame* describes the argument structure a verb requires.
//! This module provides a built-in lexicon of ~120 English verbs, classifies
//! them by transitivity, and checks whether the arguments present in a
//! dependency parse satisfy the verb's frame.

use crate::features::{
    FeatureBundle, TransitivityValue,
};
use shared_types::{DependencyRelation, PosTag, Sentence};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── ArgumentSlot ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArgumentRole {
    Subject,
    DirectObject,
    IndirectObject,
    Complement,
    Adjunct,
}

impl fmt::Display for ArgumentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgumentSlot {
    pub role: ArgumentRole,
    /// Expected syntactic category (NP, PP, S, VP, AP, …).
    pub category: String,
    pub features: FeatureBundle,
    pub required: bool,
}

impl ArgumentSlot {
    pub fn required(role: ArgumentRole, category: impl Into<String>) -> Self {
        Self {
            role,
            category: category.into(),
            features: FeatureBundle::new(),
            required: true,
        }
    }

    pub fn optional(role: ArgumentRole, category: impl Into<String>) -> Self {
        Self {
            role,
            category: category.into(),
            features: FeatureBundle::new(),
            required: false,
        }
    }
}

// ── SubcatFrame ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubcatFrame {
    pub verb_class: TransitivityValue,
    pub required_args: Vec<ArgumentSlot>,
    pub optional_args: Vec<ArgumentSlot>,
}

impl SubcatFrame {
    pub fn transitive() -> Self {
        Self {
            verb_class: TransitivityValue::Transitive,
            required_args: vec![
                ArgumentSlot::required(ArgumentRole::Subject, "NP"),
                ArgumentSlot::required(ArgumentRole::DirectObject, "NP"),
            ],
            optional_args: vec![],
        }
    }

    pub fn intransitive() -> Self {
        Self {
            verb_class: TransitivityValue::Intransitive,
            required_args: vec![ArgumentSlot::required(ArgumentRole::Subject, "NP")],
            optional_args: vec![],
        }
    }

    pub fn ditransitive() -> Self {
        Self {
            verb_class: TransitivityValue::Ditransitive,
            required_args: vec![
                ArgumentSlot::required(ArgumentRole::Subject, "NP"),
                ArgumentSlot::required(ArgumentRole::DirectObject, "NP"),
                ArgumentSlot::required(ArgumentRole::IndirectObject, "NP"),
            ],
            optional_args: vec![],
        }
    }

    pub fn copular() -> Self {
        Self {
            verb_class: TransitivityValue::Copular,
            required_args: vec![
                ArgumentSlot::required(ArgumentRole::Subject, "NP"),
                ArgumentSlot::required(ArgumentRole::Complement, "AP"),
            ],
            optional_args: vec![],
        }
    }

    pub fn unaccusative() -> Self {
        Self {
            verb_class: TransitivityValue::Unaccusative,
            required_args: vec![ArgumentSlot::required(ArgumentRole::Subject, "NP")],
            optional_args: vec![],
        }
    }

    /// Does this frame permit passivisation?
    pub fn allows_passive(&self) -> bool {
        matches!(
            self.verb_class,
            TransitivityValue::Transitive | TransitivityValue::Ditransitive
        )
    }

    /// Does this frame permit there-insertion?
    pub fn allows_there_insertion(&self) -> bool {
        matches!(
            self.verb_class,
            TransitivityValue::Unaccusative | TransitivityValue::Copular
        )
    }

    /// Does this frame permit dative alternation?
    pub fn allows_dative_alternation(&self) -> bool {
        self.verb_class == TransitivityValue::Ditransitive
    }
}

// ── SubcatViolation ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubcatViolation {
    pub verb: String,
    pub expected_frame: TransitivityValue,
    pub actual_arguments: Vec<String>,
    pub missing_args: Vec<String>,
    pub extra_args: Vec<String>,
}

impl SubcatViolation {
    pub fn new(
        verb: impl Into<String>,
        expected_frame: TransitivityValue,
        actual_arguments: Vec<String>,
        missing_args: Vec<String>,
        extra_args: Vec<String>,
    ) -> Self {
        Self {
            verb: verb.into(),
            expected_frame,
            actual_arguments,
            missing_args,
            extra_args,
        }
    }
}

impl fmt::Display for SubcatViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Subcat violation for '{}': expected {:?}, missing [{}], extra [{}]",
            self.verb,
            self.expected_frame,
            self.missing_args.join(", "),
            self.extra_args.join(", "),
        )
    }
}

// ── SubcatChecker ───────────────────────────────────────────────────────────

/// Verb subcategorization checker with a built-in lexicon.
#[derive(Debug, Clone)]
pub struct SubcatChecker {
    pub frame_lexicon: HashMap<String, Vec<SubcatFrame>>,
}

impl SubcatChecker {
    /// Build the checker with the default ~120-verb lexicon.
    pub fn new() -> Self {
        let mut lex: HashMap<String, Vec<SubcatFrame>> = HashMap::new();

        // ── Transitive verbs ────────────────────────────────────────────
        let transitive_verbs = [
            "eat", "read", "write", "see", "take", "make", "find", "use", "know",
            "want", "need", "like", "love", "hate", "break", "build", "buy", "catch",
            "choose", "clean", "close", "cook", "cut", "describe", "design",
            "destroy", "discuss", "draw", "drink", "drive", "enjoy", "examine",
            "explain", "fix", "follow", "forget", "help", "hit", "hold", "include",
            "keep", "kick", "leave", "lift", "lose", "meet", "move", "open", "pay",
            "pick", "play", "pull", "push", "put", "reach", "remove", "replace",
            "save", "sell", "stop", "study", "support", "throw", "touch", "turn",
            "understand", "visit", "wash", "watch", "wear", "win",
        ];
        for v in &transitive_verbs {
            lex.entry(v.to_string())
                .or_default()
                .push(SubcatFrame::transitive());
        }

        // ── Intransitive verbs ──────────────────────────────────────────
        let intransitive_verbs = [
            "sleep", "arrive", "exist", "die", "fall", "laugh", "cry", "run",
            "walk", "swim", "fly", "jump", "sit", "stand", "lie", "wait", "work",
            "go", "come", "stay", "live", "breathe", "sneeze", "cough", "shiver",
            "tremble", "yawn", "smile", "frown", "nod",
        ];
        for v in &intransitive_verbs {
            lex.entry(v.to_string())
                .or_default()
                .push(SubcatFrame::intransitive());
        }

        // ── Ditransitive verbs ──────────────────────────────────────────
        let ditransitive_verbs = [
            "give", "send", "show", "tell", "offer", "bring", "hand", "pass",
            "teach", "write", "lend", "sell", "buy", "read", "make", "cook",
            "build", "pay", "owe", "promise",
        ];
        for v in &ditransitive_verbs {
            lex.entry(v.to_string())
                .or_default()
                .push(SubcatFrame::ditransitive());
        }

        // ── Copular verbs ───────────────────────────────────────────────
        let copular_verbs = [
            "be", "seem", "become", "appear", "look", "feel", "sound", "taste",
            "smell", "remain", "stay", "grow", "turn", "get", "prove",
        ];
        for v in &copular_verbs {
            lex.entry(v.to_string())
                .or_default()
                .push(SubcatFrame::copular());
        }

        // ── Unaccusative verbs ──────────────────────────────────────────
        let unaccusative_verbs = [
            "arrive", "exist", "appear", "happen", "remain", "emerge", "occur",
            "vanish", "disappear", "fall", "rise", "grow", "melt", "freeze",
            "break", "sink", "float",
        ];
        for v in &unaccusative_verbs {
            lex.entry(v.to_string())
                .or_default()
                .push(SubcatFrame::unaccusative());
        }

        Self { frame_lexicon: lex }
    }

    /// Look up the frames for a given lemma.
    pub fn frames_for(&self, lemma: &str) -> Vec<&SubcatFrame> {
        self.frame_lexicon
            .get(&lemma.to_lowercase())
            .map(|fs| fs.iter().collect())
            .unwrap_or_default()
    }

    /// Is the verb attested in the lexicon?
    pub fn knows_verb(&self, lemma: &str) -> bool {
        self.frame_lexicon.contains_key(&lemma.to_lowercase())
    }

    /// Check whether the arguments found in `sentence` for the verb at
    /// `verb_index` satisfy at least one of the verb's frames.
    pub fn check_subcategorization(
        &self,
        sentence: &Sentence,
        verb_index: usize,
    ) -> Vec<SubcatViolation> {
        let token = match sentence.tokens.get(verb_index) {
            Some(t) => t,
            None => return vec![],
        };
        let lemma = token.normalized_form();
        let frames = match self.frame_lexicon.get(&lemma) {
            Some(f) => f,
            None => return vec![], // unknown verb – nothing to check
        };

        let actual = find_arguments(sentence, verb_index);
        let actual_roles: Vec<String> = actual.iter().map(|(r, _)| format!("{r:?}")).collect();

        let mut best_violation: Option<SubcatViolation> = None;
        let mut best_missing = usize::MAX;

        for frame in frames {
            let mut missing = Vec::new();
            let mut extra = Vec::new();

            // Check required args
            for slot in &frame.required_args {
                let role_str = format!("{:?}", slot.role);
                if !actual.iter().any(|(r, _)| format!("{r:?}") == role_str) {
                    missing.push(role_str);
                }
            }

            // Check for arguments not in any slot
            let all_roles: Vec<String> = frame
                .required_args
                .iter()
                .chain(frame.optional_args.iter())
                .map(|s| format!("{:?}", s.role))
                .collect();
            for (role, _) in &actual {
                let role_str = format!("{role:?}");
                if !all_roles.contains(&role_str) {
                    extra.push(role_str);
                }
            }

            if missing.is_empty() && extra.is_empty() {
                return vec![]; // at least one frame is satisfied
            }

            if missing.len() < best_missing {
                best_missing = missing.len();
                best_violation = Some(SubcatViolation::new(
                    &lemma,
                    frame.verb_class,
                    actual_roles.clone(),
                    missing,
                    extra,
                ));
            }
        }

        best_violation.into_iter().collect()
    }

    /// Check whether passivisation is valid for the verb at `verb_index`.
    pub fn check_passivization_requirement(
        &self,
        sentence: &Sentence,
        verb_index: usize,
    ) -> Option<SubcatViolation> {
        let token = sentence.tokens.get(verb_index)?;
        let lemma = token.normalized_form();
        let frames = self.frame_lexicon.get(&lemma)?;
        if frames.iter().any(|f| f.allows_passive()) {
            None
        } else {
            Some(SubcatViolation::new(
                &lemma,
                TransitivityValue::Intransitive,
                vec![],
                vec![],
                vec!["Passive requires transitive/ditransitive frame".into()],
            ))
        }
    }

    /// Check whether there-insertion is valid for the verb at `verb_index`.
    pub fn check_there_insertion_requirement(
        &self,
        sentence: &Sentence,
        verb_index: usize,
    ) -> Option<SubcatViolation> {
        let token = sentence.tokens.get(verb_index)?;
        let lemma = token.normalized_form();
        let frames = self.frame_lexicon.get(&lemma)?;
        if frames.iter().any(|f| f.allows_there_insertion()) {
            None
        } else {
            Some(SubcatViolation::new(
                &lemma,
                TransitivityValue::Transitive,
                vec![],
                vec![],
                vec!["There-insertion requires unaccusative/copular frame".into()],
            ))
        }
    }

    /// Check whether dative alternation is valid for the verb at `verb_index`.
    pub fn check_dative_requirement(
        &self,
        sentence: &Sentence,
        verb_index: usize,
    ) -> Option<SubcatViolation> {
        let token = sentence.tokens.get(verb_index)?;
        let lemma = token.normalized_form();
        let frames = self.frame_lexicon.get(&lemma)?;
        if frames.iter().any(|f| f.allows_dative_alternation()) {
            None
        } else {
            Some(SubcatViolation::new(
                &lemma,
                TransitivityValue::Transitive,
                vec![],
                vec![],
                vec!["Dative alternation requires ditransitive frame".into()],
            ))
        }
    }
}

impl Default for SubcatChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Argument extraction from dependency tree ────────────────────────────────

/// Extract the argument roles present for the verb at `verb_index`.
pub fn find_arguments(
    sentence: &Sentence,
    verb_index: usize,
) -> Vec<(ArgumentRole, usize)> {
    let mut args = Vec::new();
    for edge in &sentence.dependency_edges {
        if edge.head_index != verb_index {
            continue;
        }
        let role = match edge.relation {
            DependencyRelation::Nsubj => Some(ArgumentRole::Subject),
            DependencyRelation::Dobj => Some(ArgumentRole::DirectObject),
            DependencyRelation::Iobj => Some(ArgumentRole::IndirectObject),
            DependencyRelation::Ccomp | DependencyRelation::Xcomp => {
                Some(ArgumentRole::Complement)
            }
            DependencyRelation::Advmod
            | DependencyRelation::Prep
            | DependencyRelation::Nmod
            | DependencyRelation::Advcl => Some(ArgumentRole::Adjunct),
            _ => None,
        };
        if let Some(r) = role {
            args.push((r, edge.dependent_index));
        }
    }
    args
}

/// Check whether the token at `dep_index` has the expected POS category.
pub fn check_argument_category(
    sentence: &Sentence,
    dep_index: usize,
    expected_category: &str,
) -> bool {
    let token = match sentence.tokens.get(dep_index) {
        Some(t) => t,
        None => return false,
    };
    match expected_category {
        "NP" => matches!(token.pos_tag, Some(PosTag::Noun) | Some(PosTag::Pron)),
        "VP" => matches!(token.pos_tag, Some(PosTag::Verb) | Some(PosTag::Aux)),
        "AP" | "AdjP" => token.pos_tag == Some(PosTag::Adj),
        "AdvP" => token.pos_tag == Some(PosTag::Adv),
        "PP" => token.pos_tag == Some(PosTag::Prep),
        "S" | "CP" => matches!(token.pos_tag, Some(PosTag::Verb) | Some(PosTag::Aux)),
        _ => true,
    }
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
            .map(|(i, (w, pos))| Token::new(*w, i).with_pos(*pos).with_lemma(w.to_lowercase()))
            .collect();
        Sentence {
            raw_text: words.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" "),
            tokens,
            dependency_edges: edges,
            entities: vec![],
        }
    }

    #[test]
    fn test_lexicon_knows_common_verbs() {
        let checker = SubcatChecker::new();
        assert!(checker.knows_verb("eat"));
        assert!(checker.knows_verb("sleep"));
        assert!(checker.knows_verb("give"));
        assert!(checker.knows_verb("be"));
        assert!(checker.knows_verb("arrive"));
    }

    #[test]
    fn test_lexicon_transitive_has_frame() {
        let checker = SubcatChecker::new();
        let frames = checker.frames_for("eat");
        assert!(!frames.is_empty());
        assert!(frames.iter().any(|f| f.verb_class == TransitivityValue::Transitive));
    }

    #[test]
    fn test_lexicon_ditransitive() {
        let checker = SubcatChecker::new();
        let frames = checker.frames_for("give");
        assert!(frames.iter().any(|f| f.verb_class == TransitivityValue::Ditransitive));
    }

    #[test]
    fn test_find_arguments_basic() {
        // "cats eat fish"
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb), ("fish", PosTag::Noun)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 2, DependencyRelation::Dobj),
            ],
        );
        let args = find_arguments(&s, 1);
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn test_check_subcategorization_ok() {
        let checker = SubcatChecker::new();
        // "cats eat fish" — transitive satisfied
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb), ("fish", PosTag::Noun)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 2, DependencyRelation::Dobj),
            ],
        );
        let v = checker.check_subcategorization(&s, 1);
        assert!(v.is_empty(), "Expected no violation, got: {:?}", v);
    }

    #[test]
    fn test_check_passivization_transitive_ok() {
        let checker = SubcatChecker::new();
        let s = make_sentence(&[("eat", PosTag::Verb)], vec![]);
        assert!(checker.check_passivization_requirement(&s, 0).is_none());
    }

    #[test]
    fn test_check_passivization_intransitive_fails() {
        let checker = SubcatChecker::new();
        let s = make_sentence(&[("sleep", PosTag::Verb)], vec![]);
        assert!(checker.check_passivization_requirement(&s, 0).is_some());
    }

    #[test]
    fn test_check_there_insertion_copular_ok() {
        let checker = SubcatChecker::new();
        let s = make_sentence(&[("be", PosTag::Verb)], vec![]);
        assert!(checker.check_there_insertion_requirement(&s, 0).is_none());
    }

    #[test]
    fn test_check_dative_ditransitive_ok() {
        let checker = SubcatChecker::new();
        let s = make_sentence(&[("give", PosTag::Verb)], vec![]);
        assert!(checker.check_dative_requirement(&s, 0).is_none());
    }

    #[test]
    fn test_check_dative_intransitive_fails() {
        let checker = SubcatChecker::new();
        let s = make_sentence(&[("sleep", PosTag::Verb)], vec![]);
        assert!(checker.check_dative_requirement(&s, 0).is_some());
    }

    #[test]
    fn test_check_argument_category() {
        let s = make_sentence(
            &[("cat", PosTag::Noun), ("big", PosTag::Adj)],
            vec![],
        );
        assert!(check_argument_category(&s, 0, "NP"));
        assert!(!check_argument_category(&s, 0, "VP"));
        assert!(check_argument_category(&s, 1, "AP"));
    }
}
