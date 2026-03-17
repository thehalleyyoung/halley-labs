//! Passivization transformation: active voice → passive voice and back.

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    capitalize, decapitalize, rebuild_sentence, reindex_tokens, subtree_indices,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Token};
use std::collections::HashMap;

/// Table mapping base form → past participle for irregular verbs.
fn irregular_participle_table() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    let entries: &[(&str, &str)] = &[
        ("arise", "arisen"), ("awake", "awoken"), ("be", "been"), ("bear", "borne"),
        ("beat", "beaten"), ("become", "become"), ("begin", "begun"), ("bend", "bent"),
        ("bet", "bet"), ("bind", "bound"), ("bite", "bitten"), ("bleed", "bled"),
        ("blow", "blown"), ("break", "broken"), ("breed", "bred"), ("bring", "brought"),
        ("build", "built"), ("burn", "burnt"), ("burst", "burst"), ("buy", "bought"),
        ("catch", "caught"), ("choose", "chosen"), ("come", "come"), ("cost", "cost"),
        ("cut", "cut"), ("deal", "dealt"), ("dig", "dug"), ("do", "done"),
        ("draw", "drawn"), ("dream", "dreamt"), ("drink", "drunk"), ("drive", "driven"),
        ("eat", "eaten"), ("fall", "fallen"), ("feed", "fed"), ("feel", "felt"),
        ("fight", "fought"), ("find", "found"), ("fly", "flown"), ("forbid", "forbidden"),
        ("forget", "forgotten"), ("forgive", "forgiven"), ("freeze", "frozen"),
        ("get", "gotten"), ("give", "given"), ("go", "gone"), ("grow", "grown"),
        ("hang", "hung"), ("have", "had"), ("hear", "heard"), ("hide", "hidden"),
        ("hit", "hit"), ("hold", "held"), ("hurt", "hurt"), ("keep", "kept"),
        ("kneel", "knelt"), ("know", "known"), ("lay", "laid"), ("lead", "led"),
        ("leave", "left"), ("lend", "lent"), ("let", "let"), ("lie", "lain"),
        ("light", "lit"), ("lose", "lost"), ("make", "made"), ("mean", "meant"),
        ("meet", "met"), ("pay", "paid"), ("put", "put"), ("read", "read"),
        ("ride", "ridden"), ("ring", "rung"), ("rise", "risen"), ("run", "run"),
        ("say", "said"), ("see", "seen"), ("seek", "sought"), ("sell", "sold"),
        ("send", "sent"), ("set", "set"), ("shake", "shaken"), ("shine", "shone"),
        ("shoot", "shot"), ("show", "shown"), ("shut", "shut"), ("sing", "sung"),
        ("sink", "sunk"), ("sit", "sat"), ("sleep", "slept"), ("slide", "slid"),
        ("speak", "spoken"), ("spend", "spent"), ("spin", "spun"), ("stand", "stood"),
        ("steal", "stolen"), ("stick", "stuck"), ("sting", "stung"), ("strike", "struck"),
        ("swear", "sworn"), ("sweep", "swept"), ("swim", "swum"), ("swing", "swung"),
        ("take", "taken"), ("teach", "taught"), ("tear", "torn"), ("tell", "told"),
        ("think", "thought"), ("throw", "thrown"), ("understand", "understood"),
        ("wake", "woken"), ("wear", "worn"), ("weave", "woven"), ("win", "won"),
        ("wind", "wound"), ("write", "written"),
    ];
    for &(base, pp) in entries {
        m.insert(base, pp);
    }
    m
}

/// Reverse lookup: past participle → base form.
fn reverse_participle_table() -> HashMap<&'static str, &'static str> {
    let fwd = irregular_participle_table();
    let mut rev = HashMap::new();
    for (base, pp) in fwd {
        rev.insert(pp, base);
    }
    rev
}

/// Convert a verb to its past-participle form.
fn to_past_participle(verb: &str) -> String {
    let low = verb.to_lowercase();
    let table = irregular_participle_table();
    if let Some(&pp) = table.get(low.as_str()) {
        return pp.to_string();
    }
    // Regular: add -ed (handling doubling and silent-e)
    if low.ends_with('e') {
        format!("{}d", low)
    } else if low.ends_with('y') && low.len() > 2 && !is_vowel(low.as_bytes()[low.len() - 2]) {
        format!("{}ied", &low[..low.len() - 1])
    } else if should_double_final(&low) {
        format!("{}{}ed", low, low.chars().last().unwrap())
    } else {
        format!("{}ed", low)
    }
}

fn is_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')
}

fn should_double_final(word: &str) -> bool {
    let bytes = word.as_bytes();
    if bytes.len() < 3 {
        return false;
    }
    let last = bytes[bytes.len() - 1];
    let penult = bytes[bytes.len() - 2];
    let antepenult = bytes[bytes.len() - 3];
    !is_vowel(last) && is_vowel(penult) && !is_vowel(antepenult) && last != b'w' && last != b'x' && last != b'y'
}

/// Choose the correct form of "be" for passive auxiliary given the original verb tense.
fn passive_auxiliary(original_verb: &str) -> &'static str {
    let low = original_verb.to_lowercase();
    if low.ends_with("ed") || is_past_form(&low) {
        "was"
    } else if low.ends_with('s') && !low.ends_with("ss") {
        "is"
    } else {
        "is"
    }
}

fn is_past_form(verb: &str) -> bool {
    let past_forms: &[&str] = &[
        "went", "came", "saw", "did", "had", "made", "said", "took", "got",
        "gave", "found", "knew", "thought", "told", "became", "left", "felt",
        "put", "ran", "brought", "began", "kept", "held", "wrote", "stood",
        "heard", "let", "meant", "set", "met", "paid", "read", "sat",
        "spoke", "lay", "led", "lost", "spent", "built", "sent", "fell",
        "hit", "cut", "caught", "drove", "ate", "drew", "fought", "threw",
        "broke", "chose", "grew", "hung", "shook", "wore", "bit", "blew",
        "bore", "dealt", "dug", "flew", "froze", "hid", "rose", "sang",
        "sank", "slept", "slid", "stole", "struck", "swam", "swore", "tore",
        "woke", "wound",
    ];
    past_forms.contains(&verb)
}

/// Find the subject token index (nsubj dependent of the root).
fn find_subject(sentence: &Sentence) -> Option<usize> {
    let root = sentence.root_index()?;
    sentence
        .dependency_edges
        .iter()
        .find(|e| e.head_index == root && e.relation == DependencyRelation::Nsubj)
        .map(|e| e.dependent_index)
}

/// Find the direct object token index.
fn find_object(sentence: &Sentence) -> Option<usize> {
    let root = sentence.root_index()?;
    sentence
        .dependency_edges
        .iter()
        .find(|e| e.head_index == root && e.relation == DependencyRelation::Dobj)
        .map(|e| e.dependent_index)
}

/// Find the main verb token index.
fn find_main_verb(sentence: &Sentence) -> Option<usize> {
    sentence.root_index()
}

/// Check if there is a modal auxiliary.
fn find_modal(sentence: &Sentence) -> Option<usize> {
    let root = sentence.root_index()?;
    sentence
        .dependency_edges
        .iter()
        .find(|e| {
            e.head_index == root
                && e.relation == DependencyRelation::Aux
                && sentence
                    .tokens
                    .get(e.dependent_index)
                    .map_or(false, |t| {
                        let low = t.text.to_lowercase();
                        low == "can" || low == "could" || low == "may" || low == "might"
                            || low == "shall" || low == "should" || low == "will"
                            || low == "would" || low == "must"
                    })
        })
        .map(|e| e.dependent_index)
}

// ── PassivizationTransform ──────────────────────────────────────────────────

pub struct PassivizationTransform;

impl PassivizationTransform {
    pub fn new() -> Self {
        Self
    }

    /// Apply inverse: passive → active.
    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Look for "by" PP to recover agent
        let root_idx = sentence.root_index().ok_or(TransformationError::NoApplicableSite)?;
        let verb_token = sentence.tokens.get(root_idx)
            .ok_or(TransformationError::NoApplicableSite)?;

        // Find the passive subject (nsubj of root)
        let pass_subj_idx = find_subject(sentence)
            .ok_or(TransformationError::InverseNotPossible("no subject found".into()))?;

        // Find "by" prep dependent
        let by_idx = sentence.dependency_edges.iter().find(|e| {
            e.head_index == root_idx
                && (e.relation == DependencyRelation::Prep || e.relation == DependencyRelation::Case)
                && sentence.tokens.get(e.dependent_index)
                    .map_or(false, |t| t.text.to_lowercase() == "by")
        }).map(|e| e.dependent_index);

        let agent_idx = by_idx.and_then(|bi| {
            sentence.dependency_edges.iter().find(|e| {
                e.head_index == bi && e.relation == DependencyRelation::Pobj
            }).map(|e| e.dependent_index)
        });

        // Find the auxiliary "be" form
        let aux_be_idx = sentence.dependency_edges.iter().find(|e| {
            e.head_index == root_idx && e.relation == DependencyRelation::Aux
                && sentence.tokens.get(e.dependent_index).map_or(false, |t| {
                    let low = t.text.to_lowercase();
                    low == "was" || low == "were" || low == "is" || low == "are"
                        || low == "been" || low == "be" || low == "being"
                })
        }).map(|e| e.dependent_index);

        // Reconstruct active form
        let subj_span = subtree_indices(sentence, pass_subj_idx);
        let agent_span = agent_idx.map(|ai| subtree_indices(sentence, ai)).unwrap_or_default();

        let rev_table = reverse_participle_table();
        let verb_text = &verb_token.text.to_lowercase();
        let active_verb = rev_table.get(verb_text.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                if verb_text.ends_with("ed") {
                    verb_text[..verb_text.len() - 2].to_string()
                } else {
                    verb_text.clone()
                }
            });

        // Build: agent + active_verb + passive_subject (now object)
        let mut new_tokens: Vec<Token> = Vec::new();

        // Agent as new subject (or "someone" if no by-phrase)
        if !agent_span.is_empty() {
            for &i in &agent_span {
                if let Some(t) = sentence.tokens.get(i) {
                    new_tokens.push(t.clone());
                }
            }
        } else {
            new_tokens.push(Token::new("someone", 0).with_pos(PosTag::Pron));
        }
        if let Some(first) = new_tokens.first_mut() {
            first.text = capitalize(&first.text);
        }

        // Verb
        let tense_verb = if aux_be_idx.map_or(false, |i| {
            sentence.tokens.get(i).map_or(false, |t| {
                let l = t.text.to_lowercase();
                l == "was" || l == "were"
            })
        }) {
            // Past tense
            if let Some(&pp) = irregular_participle_table().get(active_verb.as_str()) {
                // Use simple past - approximate with past participle for irregulars
                let past_table: HashMap<&str, &str> = [
                    ("go", "went"), ("take", "took"), ("give", "gave"), ("see", "saw"),
                    ("eat", "ate"), ("write", "wrote"), ("break", "broke"), ("speak", "spoke"),
                    ("drive", "drove"), ("choose", "chose"), ("know", "knew"),
                    ("grow", "grew"), ("throw", "threw"), ("draw", "drew"),
                    ("begin", "began"), ("drink", "drank"), ("sing", "sang"),
                    ("swim", "swam"), ("ring", "rang"), ("run", "ran"),
                    ("come", "came"), ("become", "became"),
                ].into_iter().collect();
                past_table.get(active_verb.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{}ed", active_verb))
            } else {
                format!("{}ed", active_verb)
            }
        } else {
            active_verb.clone()
        };
        new_tokens.push(Token::new(&tense_verb, 0).with_pos(PosTag::Verb));

        // Old passive subject becomes object
        for &i in &subj_span {
            if let Some(t) = sentence.tokens.get(i) {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
        }

        // Punct
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
            "Converted passive to active voice",
        ))
    }
}

impl BaseTransformation for PassivizationTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::Passivization
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        let feats = sentence.compute_features();
        if feats.voice != Some(shared_types::Voice::Active) {
            return false;
        }
        let root = match sentence.root_index() {
            Some(r) => r,
            None => return false,
        };
        let has_subj = sentence.dependency_edges.iter().any(|e| {
            e.head_index == root && e.relation == DependencyRelation::Nsubj
        });
        let has_obj = sentence.dependency_edges.iter().any(|e| {
            e.head_index == root && e.relation == DependencyRelation::Dobj
        });
        let verb_ok = sentence.tokens.get(root).map_or(false, |t| {
            t.pos_tag == Some(PosTag::Verb)
        });
        has_subj && has_obj && verb_ok
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "Sentence must be active voice with transitive verb".into(),
            ));
        }

        let root_idx = sentence.root_index().unwrap();
        let subj_idx = find_subject(sentence).unwrap();
        let obj_idx = find_object(sentence).unwrap();
        let verb_token = &sentence.tokens[root_idx];

        let subj_span = subtree_indices(sentence, subj_idx);
        let obj_span = subtree_indices(sentence, obj_idx);

        // Determine past participle
        let lemma = verb_token.lemma.as_deref()
            .unwrap_or(&verb_token.text);
        let pp = to_past_participle(lemma);

        // Determine auxiliary
        let modal_idx = find_modal(sentence);
        let aux_form = if modal_idx.is_some() {
            "be"
        } else {
            passive_auxiliary(&verb_token.text)
        };

        // Build new token sequence: Object + aux + past_participle + "by" + Subject
        let mut new_tokens: Vec<Token> = Vec::new();

        // Object tokens become subject (capitalize first)
        for (i, &idx) in obj_span.iter().enumerate() {
            if let Some(t) = sentence.tokens.get(idx) {
                let mut tok = t.clone();
                if i == 0 {
                    tok.text = capitalize(&tok.text);
                }
                new_tokens.push(tok);
            }
        }

        // Modal if present
        if let Some(mi) = modal_idx {
            if let Some(t) = sentence.tokens.get(mi) {
                new_tokens.push(t.clone());
            }
        }

        // Auxiliary "be"
        new_tokens.push(Token::new(aux_form, 0).with_pos(PosTag::Aux));

        // Past participle
        new_tokens.push(Token::new(&pp, 0).with_pos(PosTag::Verb));

        // "by" + subject tokens
        new_tokens.push(Token::new("by", 0).with_pos(PosTag::Prep));
        for &idx in &subj_span {
            if let Some(t) = sentence.tokens.get(idx) {
                let mut tok = t.clone();
                tok.text = decapitalize(&tok.text);
                new_tokens.push(tok);
            }
        }

        // Copy remaining tokens (adverbs, PPs, etc.) that aren't subj/obj/verb/modal/punct
        let mut skip: Vec<usize> = Vec::new();
        skip.extend(&subj_span);
        skip.extend(&obj_span);
        skip.push(root_idx);
        if let Some(mi) = modal_idx {
            skip.push(mi);
        }
        // Also skip existing auxiliaries of the root
        for e in &sentence.dependency_edges {
            if e.head_index == root_idx && e.relation == DependencyRelation::Aux {
                skip.push(e.dependent_index);
            }
        }

        for (i, tok) in sentence.tokens.iter().enumerate() {
            if !skip.contains(&i) && tok.pos_tag != Some(PosTag::Punct) {
                new_tokens.push(tok.clone());
            }
        }

        // Trailing punctuation
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
            format!(
                "Passivized: promoted object, inserted aux '{}', verb→'{}'",
                aux_form, pp
            ),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        true
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn active_sentence() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("cat", 1).with_pos(PosTag::Noun),
            Token::new("chased", 2).with_pos(PosTag::Verb).with_lemma("chase"),
            Token::new("the", 3).with_pos(PosTag::Det),
            Token::new("mouse", 4).with_pos(PosTag::Noun),
            Token::new(".", 5).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 4, DependencyRelation::Dobj),
            DependencyEdge::new(4, 3, DependencyRelation::Det),
            DependencyEdge::new(2, 5, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The cat chased the mouse.");
        s.dependency_edges = edges;
        s
    }

    fn modal_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("can", 1).with_pos(PosTag::Aux),
            Token::new("write", 2).with_pos(PosTag::Verb).with_lemma("write"),
            Token::new("a", 3).with_pos(PosTag::Det),
            Token::new("letter", 4).with_pos(PosTag::Noun),
            Token::new(".", 5).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(2, 2, DependencyRelation::Root),
            DependencyEdge::new(2, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(2, 1, DependencyRelation::Aux),
            DependencyEdge::new(2, 4, DependencyRelation::Dobj),
            DependencyEdge::new(4, 3, DependencyRelation::Det),
            DependencyEdge::new(2, 5, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John can write a letter.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = PassivizationTransform::new();
        assert_eq!(t.kind(), TransformationKind::Passivization);
    }

    #[test]
    fn test_is_applicable_active() {
        let t = PassivizationTransform::new();
        assert!(t.is_applicable(&active_sentence()));
    }

    #[test]
    fn test_not_applicable_no_object() {
        let t = PassivizationTransform::new();
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("slept", 1).with_pos(PosTag::Verb),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John slept.");
        s.dependency_edges = edges;
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_apply_basic() {
        let t = PassivizationTransform::new();
        let result = t.apply(&active_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("chased"));
        assert!(text.contains("by"));
    }

    #[test]
    fn test_apply_modal() {
        let t = PassivizationTransform::new();
        let result = t.apply(&modal_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("can"));
        assert!(text.contains("be"));
        assert!(text.contains("written"));
    }

    #[test]
    fn test_irregular_participles() {
        assert_eq!(to_past_participle("go"), "gone");
        assert_eq!(to_past_participle("take"), "taken");
        assert_eq!(to_past_participle("write"), "written");
        assert_eq!(to_past_participle("break"), "broken");
        assert_eq!(to_past_participle("eat"), "eaten");
    }

    #[test]
    fn test_regular_participles() {
        assert_eq!(to_past_participle("chase"), "chased");
        assert_eq!(to_past_participle("walk"), "walked");
        assert_eq!(to_past_participle("stop"), "stopped");
        assert_eq!(to_past_participle("carry"), "carried");
    }

    #[test]
    fn test_meaning_preserving() {
        let t = PassivizationTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_inverse() {
        let t = PassivizationTransform::new();
        // First passivize, then try inverse
        let result = t.apply(&active_sentence()).unwrap();
        // Build a passive sentence manually for inverse test
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("mouse", 1).with_pos(PosTag::Noun),
            Token::new("was", 2).with_pos(PosTag::Aux),
            Token::new("chased", 3).with_pos(PosTag::Verb).with_lemma("chase"),
            Token::new("by", 4).with_pos(PosTag::Prep),
            Token::new("the", 5).with_pos(PosTag::Det),
            Token::new("cat", 6).with_pos(PosTag::Noun),
            Token::new(".", 7).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(3, 3, DependencyRelation::Root),
            DependencyEdge::new(3, 1, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 0, DependencyRelation::Det),
            DependencyEdge::new(3, 2, DependencyRelation::Aux),
            DependencyEdge::new(3, 4, DependencyRelation::Prep),
            DependencyEdge::new(4, 6, DependencyRelation::Pobj),
            DependencyEdge::new(6, 5, DependencyRelation::Det),
            DependencyEdge::new(3, 7, DependencyRelation::Punct),
        ];
        let mut passive = Sentence::from_tokens(tokens, "The mouse was chased by the cat.");
        passive.dependency_edges = edges;
        let inv = t.inverse(&passive).unwrap();
        assert!(inv.success);
        let text = inv.transformed.surface_text().to_lowercase();
        assert!(text.contains("cat"));
    }
}
