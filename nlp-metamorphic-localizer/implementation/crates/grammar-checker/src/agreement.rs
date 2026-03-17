//! Subject–verb agreement checking.
//!
//! Given a `Sentence` with dependency annotations, extract every subject–verb
//! pair and verify that number and person features agree.

use crate::features::{
    Feature, FeatureBundle, NumberValue, PersonValue, TenseValue,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── AgreementViolation ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgreementViolation {
    pub subject_features: FeatureBundle,
    pub verb_features: FeatureBundle,
    pub violated_feature: String,
    pub explanation: String,
}

impl AgreementViolation {
    pub fn new(
        subject_features: FeatureBundle,
        verb_features: FeatureBundle,
        violated_feature: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            subject_features,
            verb_features,
            violated_feature: violated_feature.into(),
            explanation: explanation.into(),
        }
    }
}

impl fmt::Display for AgreementViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agreement violation ({}): {}", self.violated_feature, self.explanation)
    }
}

// ── AgreementRule ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AgreementRule {
    pub subject_features: FeatureBundle,
    pub verb_features: FeatureBundle,
    pub required_agreement: Vec<String>,
}

impl AgreementRule {
    pub fn new(required: Vec<&str>) -> Self {
        Self {
            subject_features: FeatureBundle::new(),
            verb_features: FeatureBundle::new(),
            required_agreement: required.into_iter().map(String::from).collect(),
        }
    }
}

// ── SubjectFinder ───────────────────────────────────────────────────────────

pub struct SubjectFinder;

impl SubjectFinder {
    /// Find the token index of the subject for a given verb index.
    pub fn find_subject(sentence: &Sentence, verb_index: usize) -> Option<usize> {
        sentence
            .dependency_edges
            .iter()
            .find(|e| {
                e.head_index == verb_index && e.relation == DependencyRelation::Nsubj
            })
            .map(|e| e.dependent_index)
    }

    /// Find the expletive subject ("there" / "it") for a verb.
    pub fn find_expletive(sentence: &Sentence, verb_index: usize) -> Option<usize> {
        sentence
            .dependency_edges
            .iter()
            .find(|e| {
                e.head_index == verb_index
                    && e.relation == DependencyRelation::Expl
            })
            .map(|e| e.dependent_index)
    }

    /// In existential constructions ("there is X"), the real subject is the
    /// post-verbal NP.  Return its token index.
    pub fn find_post_verbal_np(sentence: &Sentence, verb_index: usize) -> Option<usize> {
        // Look for a dobj or nsubj after the verb position.
        sentence
            .dependency_edges
            .iter()
            .find(|e| {
                e.head_index == verb_index
                    && (e.relation == DependencyRelation::Dobj
                        || e.relation == DependencyRelation::Nsubj)
                    && e.dependent_index > verb_index
            })
            .map(|e| e.dependent_index)
    }
}

// ── VerbFinder ──────────────────────────────────────────────────────────────

pub struct VerbFinder;

impl VerbFinder {
    /// Find the main (root) verb index.
    pub fn find_main_verb(sentence: &Sentence) -> Option<usize> {
        sentence
            .dependency_edges
            .iter()
            .find(|e| e.relation == DependencyRelation::Root)
            .map(|e| e.dependent_index)
    }

    /// Find all verb indices in the sentence.
    pub fn find_all_verbs(sentence: &Sentence) -> Vec<usize> {
        sentence
            .tokens
            .iter()
            .filter(|t| {
                matches!(t.pos_tag, Some(PosTag::Verb) | Some(PosTag::Aux))
            })
            .map(|t| t.index)
            .collect()
    }

    /// Extract the features of a verb from its surface form.
    pub fn verb_features(token: &Token) -> FeatureBundle {
        let mut fb = FeatureBundle::new();
        let low = token.text.to_lowercase();

        // Tense heuristic
        if let Some(t) = token.features.get("Tense") {
            match t.as_str() {
                "Past" => fb.set("Tense", Feature::Tense(TenseValue::Past)),
                "Pres" => fb.set("Tense", Feature::Tense(TenseValue::Present)),
                "Fut" => fb.set("Tense", Feature::Tense(TenseValue::Future)),
                _ => {}
            }
        } else if low.ends_with("ed") {
            fb.set("Tense", Feature::Tense(TenseValue::Past));
        } else {
            fb.set("Tense", Feature::Tense(TenseValue::Present));
        }

        // Number + person from morphology
        let (num, per) = verb_number_person(&low);
        if let Some(n) = num {
            fb.set("Number", Feature::Number(n));
        }
        if let Some(p) = per {
            fb.set("Person", Feature::Person(p));
        }
        fb
    }
}

/// Infer number/person from the surface form of a present-tense verb.
fn verb_number_person(form: &str) -> (Option<NumberValue>, Option<PersonValue>) {
    match form {
        "am" => (Some(NumberValue::Singular), Some(PersonValue::First)),
        "is" => (Some(NumberValue::Singular), Some(PersonValue::Third)),
        "are" => (Some(NumberValue::Plural), None),
        "was" => (Some(NumberValue::Singular), None),
        "were" => (Some(NumberValue::Plural), None),
        "has" => (Some(NumberValue::Singular), Some(PersonValue::Third)),
        "have" => (Some(NumberValue::Plural), None),
        "does" => (Some(NumberValue::Singular), Some(PersonValue::Third)),
        "do" => (Some(NumberValue::Plural), None),
        _ if form.ends_with('s') && !form.ends_with("ss") => {
            (Some(NumberValue::Singular), Some(PersonValue::Third))
        }
        _ => (None, None),
    }
}

// ── Feature extraction helpers ──────────────────────────────────────────────

/// Determine the number of an NP rooted at `token_index` in `sentence`.
pub fn extract_number(sentence: &Sentence, token_index: usize) -> Option<NumberValue> {
    let token = sentence.tokens.get(token_index)?;

    // Check explicit feature annotation
    if let Some(num) = token.features.get("Number") {
        return match num.as_str() {
            "Sing" => Some(NumberValue::Singular),
            "Plur" => Some(NumberValue::Plural),
            _ => None,
        };
    }

    let low = token.text.to_lowercase();

    // Coordination → plural
    let has_conj = sentence.dependency_edges.iter().any(|e| {
        e.head_index == token_index && e.relation == DependencyRelation::Conj
    });
    if has_conj {
        return Some(NumberValue::Plural);
    }

    // Pronouns
    match low.as_str() {
        "i" | "me" | "he" | "him" | "she" | "her" | "it" | "myself" | "himself" | "herself"
        | "itself" => return Some(NumberValue::Singular),
        "we" | "us" | "they" | "them" | "ourselves" | "themselves" => {
            return Some(NumberValue::Plural)
        }
        "you" | "yourself" | "yourselves" => return None, // ambiguous
        _ => {}
    }

    // Uncountable common nouns
    static UNCOUNTABLE: &[&str] = &[
        "information", "advice", "furniture", "luggage", "equipment", "knowledge",
        "research", "evidence", "music", "water", "air", "rice", "sugar", "salt",
        "bread", "news", "homework", "software", "hardware", "traffic",
    ];
    if UNCOUNTABLE.contains(&low.as_str()) {
        return Some(NumberValue::Uncountable);
    }

    // Check determiner
    let det = sentence.dependency_edges.iter().find(|e| {
        e.head_index == token_index && e.relation == DependencyRelation::Det
    });
    if let Some(det_edge) = det {
        if let Some(det_tok) = sentence.tokens.get(det_edge.dependent_index) {
            let d = det_tok.text.to_lowercase();
            if d == "this" || d == "that" || d == "a" || d == "an" || d == "each" || d == "every" {
                return Some(NumberValue::Singular);
            }
            if d == "these" || d == "those" || d == "many" || d == "few" || d == "several" {
                return Some(NumberValue::Plural);
            }
        }
    }

    // Morphological heuristic: trailing -s
    if let Some(pos) = &token.pos_tag {
        if *pos == PosTag::Noun {
            if low.ends_with('s')
                && !low.ends_with("ss")
                && !low.ends_with("us")
                && !low.ends_with("is")
            {
                return Some(NumberValue::Plural);
            }
            return Some(NumberValue::Singular);
        }
    }

    None
}

/// Determine the person of the NP at `token_index`.
pub fn extract_person(sentence: &Sentence, token_index: usize) -> Option<PersonValue> {
    let token = sentence.tokens.get(token_index)?;

    if let Some(per) = token.features.get("Person") {
        return match per.as_str() {
            "1" => Some(PersonValue::First),
            "2" => Some(PersonValue::Second),
            "3" => Some(PersonValue::Third),
            _ => None,
        };
    }

    let low = token.text.to_lowercase();
    match low.as_str() {
        "i" | "me" | "my" | "mine" | "myself" | "we" | "us" | "our" | "ours" | "ourselves" => {
            Some(PersonValue::First)
        }
        "you" | "your" | "yours" | "yourself" | "yourselves" => Some(PersonValue::Second),
        "he" | "him" | "his" | "himself" | "she" | "her" | "hers" | "herself" | "it" | "its"
        | "itself" | "they" | "them" | "their" | "theirs" | "themselves" => {
            Some(PersonValue::Third)
        }
        _ => {
            // Non-pronominal NPs are 3rd person by default.
            if token.pos_tag == Some(PosTag::Noun) || token.pos_tag == Some(PosTag::Other) {
                Some(PersonValue::Third)
            } else {
                None
            }
        }
    }
}

// ── Copula, auxiliary, modal agreement helpers ───────────────────────────────

/// Check "be" form against subject features.
pub fn check_copula_agreement(
    subject_number: Option<NumberValue>,
    subject_person: Option<PersonValue>,
    copula_form: &str,
) -> Option<AgreementViolation> {
    let low = copula_form.to_lowercase();
    let ok = match low.as_str() {
        "am" => {
            subject_person == Some(PersonValue::First)
                && subject_number == Some(NumberValue::Singular)
        }
        "is" => subject_number == Some(NumberValue::Singular) && subject_person != Some(PersonValue::First),
        "are" => {
            subject_number == Some(NumberValue::Plural)
                || subject_person == Some(PersonValue::Second)
        }
        "was" => subject_number == Some(NumberValue::Singular) || subject_number.is_none(),
        "were" => subject_number == Some(NumberValue::Plural) || subject_number.is_none(),
        "be" | "been" | "being" => true, // non-finite
        _ => true,
    };
    if ok {
        None
    } else {
        let mut sfb = FeatureBundle::new();
        if let Some(n) = subject_number {
            sfb.set("Number", Feature::Number(n));
        }
        if let Some(p) = subject_person {
            sfb.set("Person", Feature::Person(p));
        }
        let mut vfb = FeatureBundle::new();
        let (vn, vp) = verb_number_person(&low);
        if let Some(n) = vn {
            vfb.set("Number", Feature::Number(n));
        }
        if let Some(p) = vp {
            vfb.set("Person", Feature::Person(p));
        }
        Some(AgreementViolation::new(
            sfb,
            vfb,
            "Number/Person",
            format!("Copula '{low}' does not agree with subject"),
        ))
    }
}

/// Check has/have, does/do agreement.
pub fn check_auxiliary_agreement(
    subject_number: Option<NumberValue>,
    _subject_person: Option<PersonValue>,
    aux_form: &str,
) -> Option<AgreementViolation> {
    let low = aux_form.to_lowercase();
    let ok = match low.as_str() {
        "has" | "does" => subject_number == Some(NumberValue::Singular) || subject_number.is_none(),
        "have" | "do" => subject_number == Some(NumberValue::Plural) || subject_number.is_none(),
        "had" | "did" => true,
        _ => true,
    };
    if ok {
        None
    } else {
        let mut sfb = FeatureBundle::new();
        if let Some(n) = subject_number {
            sfb.set("Number", Feature::Number(n));
        }
        let mut vfb = FeatureBundle::new();
        let (vn, _) = verb_number_person(&low);
        if let Some(n) = vn {
            vfb.set("Number", Feature::Number(n));
        }
        Some(AgreementViolation::new(
            sfb,
            vfb,
            "Number",
            format!("Auxiliary '{low}' does not agree with subject number"),
        ))
    }
}

/// Modals never inflect – always return `None` (no violation).
pub fn check_modal_agreement(
    _subject_number: Option<NumberValue>,
    _subject_person: Option<PersonValue>,
    modal_form: &str,
) -> Option<AgreementViolation> {
    static MODALS: &[&str] = &[
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
        "ought",
    ];
    let low = modal_form.to_lowercase();
    if MODALS.contains(&low.as_str()) {
        None
    } else {
        // Not a modal — no opinion.
        None
    }
}

/// "There is/are" must agree with the post-verbal NP.
pub fn check_existential_agreement(
    sentence: &Sentence,
    verb_index: usize,
) -> Option<AgreementViolation> {
    let verb = sentence.tokens.get(verb_index)?;
    let low = verb.text.to_lowercase();

    let post_np_idx = SubjectFinder::find_post_verbal_np(sentence, verb_index)?;
    let np_number = extract_number(sentence, post_np_idx);

    let verb_singular = matches!(low.as_str(), "is" | "was" | "has");
    let verb_plural = matches!(low.as_str(), "are" | "were" | "have");

    let mismatch = (verb_singular && np_number == Some(NumberValue::Plural))
        || (verb_plural && np_number == Some(NumberValue::Singular));

    if mismatch {
        let mut sfb = FeatureBundle::new();
        if let Some(n) = np_number {
            sfb.set("Number", Feature::Number(n));
        }
        let mut vfb = FeatureBundle::new();
        let (vn, vp) = verb_number_person(&low);
        if let Some(n) = vn {
            vfb.set("Number", Feature::Number(n));
        }
        if let Some(p) = vp {
            vfb.set("Person", Feature::Person(p));
        }
        Some(AgreementViolation::new(
            sfb,
            vfb,
            "Number",
            format!("Existential verb '{low}' does not agree with post-verbal NP"),
        ))
    } else {
        None
    }
}

// ── AgreementChecker ────────────────────────────────────────────────────────

/// Top-level agreement checker.
#[derive(Debug, Clone)]
pub struct AgreementChecker {
    pub rules: Vec<AgreementRule>,
}

impl AgreementChecker {
    pub fn new() -> Self {
        let mut rules = Vec::new();
        // Default English agreement: number and person must match between
        // subject and finite verb.
        rules.push(AgreementRule::new(vec!["Number", "Person"]));
        Self { rules }
    }

    /// Check a single subject–verb pair.
    pub fn check_subject_verb_agreement(
        &self,
        sentence: &Sentence,
        subject_index: usize,
        verb_index: usize,
    ) -> Vec<AgreementViolation> {
        let mut violations = Vec::new();

        let subj_number = extract_number(sentence, subject_index);
        let subj_person = extract_person(sentence, subject_index);

        let verb_token = match sentence.tokens.get(verb_index) {
            Some(t) => t,
            None => return violations,
        };
        let low_verb = verb_token.text.to_lowercase();

        // Copula?
        static COPULA_FORMS: &[&str] = &["am", "is", "are", "was", "were", "be", "been", "being"];
        if COPULA_FORMS.contains(&low_verb.as_str()) {
            if let Some(v) = check_copula_agreement(subj_number, subj_person, &low_verb) {
                violations.push(v);
            }
            return violations;
        }

        // Auxiliary?
        static AUX_FORMS: &[&str] = &["has", "have", "had", "does", "do", "did"];
        if AUX_FORMS.contains(&low_verb.as_str()) {
            if let Some(v) = check_auxiliary_agreement(subj_number, subj_person, &low_verb) {
                violations.push(v);
            }
            return violations;
        }

        // Modal?
        static MODALS: &[&str] = &[
            "can", "could", "will", "would", "shall", "should", "may", "might", "must", "ought",
        ];
        if MODALS.contains(&low_verb.as_str()) {
            if let Some(v) = check_modal_agreement(subj_number, subj_person, &low_verb) {
                violations.push(v);
            }
            return violations;
        }

        // General present-tense verb: 3sg has -s suffix.
        let verb_fb = VerbFinder::verb_features(verb_token);
        if let Some(Feature::Tense(TenseValue::Present)) = verb_fb.get("Tense") {
            let verb_is_3sg = low_verb.ends_with('s') && !low_verb.ends_with("ss");
            let subj_is_3sg = subj_number == Some(NumberValue::Singular)
                && subj_person == Some(PersonValue::Third);
            let subj_is_plural = subj_number == Some(NumberValue::Plural);

            if verb_is_3sg && subj_is_plural {
                let mut sfb = FeatureBundle::new();
                sfb.set("Number", Feature::Number(NumberValue::Plural));
                violations.push(AgreementViolation::new(
                    sfb,
                    verb_fb.clone(),
                    "Number",
                    format!(
                        "Plural subject with 3sg verb form '{}'",
                        verb_token.text
                    ),
                ));
            } else if !verb_is_3sg && subj_is_3sg {
                let mut sfb = FeatureBundle::new();
                sfb.set("Number", Feature::Number(NumberValue::Singular));
                sfb.set("Person", Feature::Person(PersonValue::Third));
                violations.push(AgreementViolation::new(
                    sfb,
                    verb_fb.clone(),
                    "Number",
                    format!(
                        "3sg subject with non-3sg verb form '{}'",
                        verb_token.text
                    ),
                ));
            }
        }
        // Past-tense verbs in English don't inflect for number (except "be").
        violations
    }

    /// Find all subject–verb pairs in a sentence including embedded clauses.
    pub fn find_all_agreement_pairs(&self, sentence: &Sentence) -> Vec<(usize, usize)> {
        find_all_agreement_pairs(sentence)
    }

    /// Run agreement checks over every subject–verb pair.
    pub fn check_all(&self, sentence: &Sentence) -> Vec<AgreementViolation> {
        let mut violations = Vec::new();
        let pairs = self.find_all_agreement_pairs(sentence);
        for (subj, verb) in &pairs {
            violations.extend(self.check_subject_verb_agreement(sentence, *subj, *verb));
        }

        // Existential constructions
        for (i, tok) in sentence.tokens.iter().enumerate() {
            if tok.text.to_lowercase() == "there" {
                if let Some(head_edge) = sentence.head_of(i) {
                    if head_edge.relation == DependencyRelation::Expl {
                        if let Some(v) =
                            check_existential_agreement(sentence, head_edge.head_index)
                        {
                            violations.push(v);
                        }
                    }
                }
            }
        }
        violations
    }
}

impl Default for AgreementChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Find all (subject_index, verb_index) pairs in a sentence.
pub fn find_all_agreement_pairs(sentence: &Sentence) -> Vec<(usize, usize)> {
    sentence
        .dependency_edges
        .iter()
        .filter(|e| e.relation == DependencyRelation::Nsubj)
        .map(|e| (e.dependent_index, e.head_index))
        .collect()
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
    fn test_extract_number_singular() {
        let s = make_sentence(
            &[("cat", PosTag::Noun)],
            vec![],
        );
        assert_eq!(extract_number(&s, 0), Some(NumberValue::Singular));
    }

    #[test]
    fn test_extract_number_plural() {
        let s = make_sentence(
            &[("cats", PosTag::Noun)],
            vec![],
        );
        assert_eq!(extract_number(&s, 0), Some(NumberValue::Plural));
    }

    #[test]
    fn test_extract_person_pronoun() {
        let s = make_sentence(&[("he", PosTag::Pron)], vec![]);
        assert_eq!(extract_person(&s, 0), Some(PersonValue::Third));
        let s2 = make_sentence(&[("I", PosTag::Pron)], vec![]);
        assert_eq!(extract_person(&s2, 0), Some(PersonValue::First));
    }

    #[test]
    fn test_copula_am_first_singular() {
        assert!(check_copula_agreement(
            Some(NumberValue::Singular),
            Some(PersonValue::First),
            "am"
        )
        .is_none());
    }

    #[test]
    fn test_copula_is_with_plural_fails() {
        let v = check_copula_agreement(
            Some(NumberValue::Plural),
            Some(PersonValue::Third),
            "is",
        );
        assert!(v.is_some());
    }

    #[test]
    fn test_modal_never_violates() {
        assert!(check_modal_agreement(
            Some(NumberValue::Plural),
            Some(PersonValue::First),
            "can"
        )
        .is_none());
    }

    #[test]
    fn test_auxiliary_has_singular() {
        assert!(check_auxiliary_agreement(
            Some(NumberValue::Singular),
            Some(PersonValue::Third),
            "has"
        )
        .is_none());
    }

    #[test]
    fn test_auxiliary_has_with_plural_fails() {
        let v = check_auxiliary_agreement(
            Some(NumberValue::Plural),
            Some(PersonValue::Third),
            "has",
        );
        assert!(v.is_some());
    }

    #[test]
    fn test_find_all_agreement_pairs() {
        // "cats eat" with nsubj edge 1->0
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let pairs = find_all_agreement_pairs(&s);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_checker_no_violations() {
        let checker = AgreementChecker::new();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let v = checker.check_all(&s);
        assert!(v.is_empty(), "Expected no violations, got: {:?}", v);
    }

    #[test]
    fn test_checker_3sg_violation() {
        let checker = AgreementChecker::new();
        // "cats eats" – plural subject with 3sg verb
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eats", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let v = checker.check_all(&s);
        assert!(!v.is_empty());
    }

    #[test]
    fn test_extract_number_coordination() {
        let s = make_sentence(
            &[("cat", PosTag::Noun), ("and", PosTag::Conj), ("dog", PosTag::Noun)],
            vec![
                DependencyEdge::new(0, 2, DependencyRelation::Conj),
                DependencyEdge::new(0, 1, DependencyRelation::Cc),
            ],
        );
        assert_eq!(extract_number(&s, 0), Some(NumberValue::Plural));
    }
}
