//! Tense-change transformation: past↔present↔future with irregular verb
//! handling.

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens,
};
use shared_types::{DependencyRelation, PosTag, Sentence, Tense, Token};
use std::collections::HashMap;

/// Irregular verb table: (base, past, past_participle, present_3sg).
fn irregular_verb_table() -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
    vec![
        ("arise", "arose", "arisen", "arises"),
        ("be", "was", "been", "is"),
        ("bear", "bore", "borne", "bears"),
        ("beat", "beat", "beaten", "beats"),
        ("become", "became", "become", "becomes"),
        ("begin", "began", "begun", "begins"),
        ("bend", "bent", "bent", "bends"),
        ("bet", "bet", "bet", "bets"),
        ("bind", "bound", "bound", "binds"),
        ("bite", "bit", "bitten", "bites"),
        ("bleed", "bled", "bled", "bleeds"),
        ("blow", "blew", "blown", "blows"),
        ("break", "broke", "broken", "breaks"),
        ("breed", "bred", "bred", "breeds"),
        ("bring", "brought", "brought", "brings"),
        ("build", "built", "built", "builds"),
        ("burn", "burnt", "burnt", "burns"),
        ("burst", "burst", "burst", "bursts"),
        ("buy", "bought", "bought", "buys"),
        ("catch", "caught", "caught", "catches"),
        ("choose", "chose", "chosen", "chooses"),
        ("come", "came", "come", "comes"),
        ("cost", "cost", "cost", "costs"),
        ("cut", "cut", "cut", "cuts"),
        ("deal", "dealt", "dealt", "deals"),
        ("dig", "dug", "dug", "digs"),
        ("do", "did", "done", "does"),
        ("draw", "drew", "drawn", "draws"),
        ("dream", "dreamt", "dreamt", "dreams"),
        ("drink", "drank", "drunk", "drinks"),
        ("drive", "drove", "driven", "drives"),
        ("eat", "ate", "eaten", "eats"),
        ("fall", "fell", "fallen", "falls"),
        ("feed", "fed", "fed", "feeds"),
        ("feel", "felt", "felt", "feels"),
        ("fight", "fought", "fought", "fights"),
        ("find", "found", "found", "finds"),
        ("fly", "flew", "flown", "flies"),
        ("forbid", "forbade", "forbidden", "forbids"),
        ("forget", "forgot", "forgotten", "forgets"),
        ("forgive", "forgave", "forgiven", "forgives"),
        ("freeze", "froze", "frozen", "freezes"),
        ("get", "got", "gotten", "gets"),
        ("give", "gave", "given", "gives"),
        ("go", "went", "gone", "goes"),
        ("grow", "grew", "grown", "grows"),
        ("hang", "hung", "hung", "hangs"),
        ("have", "had", "had", "has"),
        ("hear", "heard", "heard", "hears"),
        ("hide", "hid", "hidden", "hides"),
        ("hit", "hit", "hit", "hits"),
        ("hold", "held", "held", "holds"),
        ("hurt", "hurt", "hurt", "hurts"),
        ("keep", "kept", "kept", "keeps"),
        ("kneel", "knelt", "knelt", "kneels"),
        ("know", "knew", "known", "knows"),
        ("lay", "laid", "laid", "lays"),
        ("lead", "led", "led", "leads"),
        ("leave", "left", "left", "leaves"),
        ("lend", "lent", "lent", "lends"),
        ("let", "let", "let", "lets"),
        ("lie", "lay", "lain", "lies"),
        ("light", "lit", "lit", "lights"),
        ("lose", "lost", "lost", "loses"),
        ("make", "made", "made", "makes"),
        ("mean", "meant", "meant", "means"),
        ("meet", "met", "met", "meets"),
        ("pay", "paid", "paid", "pays"),
        ("put", "put", "put", "puts"),
        ("read", "read", "read", "reads"),
        ("ride", "rode", "ridden", "rides"),
        ("ring", "rang", "rung", "rings"),
        ("rise", "rose", "risen", "rises"),
        ("run", "ran", "run", "runs"),
        ("say", "said", "said", "says"),
        ("see", "saw", "seen", "sees"),
        ("seek", "sought", "sought", "seeks"),
        ("sell", "sold", "sold", "sells"),
        ("send", "sent", "sent", "sends"),
        ("set", "set", "set", "sets"),
        ("shake", "shook", "shaken", "shakes"),
        ("shine", "shone", "shone", "shines"),
        ("shoot", "shot", "shot", "shoots"),
        ("show", "showed", "shown", "shows"),
        ("shut", "shut", "shut", "shuts"),
        ("sing", "sang", "sung", "sings"),
        ("sink", "sank", "sunk", "sinks"),
        ("sit", "sat", "sat", "sits"),
        ("sleep", "slept", "slept", "sleeps"),
        ("slide", "slid", "slid", "slides"),
        ("speak", "spoke", "spoken", "speaks"),
        ("spend", "spent", "spent", "spends"),
        ("spin", "spun", "spun", "spins"),
        ("stand", "stood", "stood", "stands"),
        ("steal", "stole", "stolen", "steals"),
        ("stick", "stuck", "stuck", "sticks"),
        ("sting", "stung", "stung", "stings"),
        ("strike", "struck", "struck", "strikes"),
        ("swear", "swore", "sworn", "swears"),
        ("sweep", "swept", "swept", "sweeps"),
        ("swim", "swam", "swum", "swims"),
        ("swing", "swung", "swung", "swings"),
        ("take", "took", "taken", "takes"),
        ("teach", "taught", "taught", "teaches"),
        ("tear", "tore", "torn", "tears"),
        ("tell", "told", "told", "tells"),
        ("think", "thought", "thought", "thinks"),
        ("throw", "threw", "thrown", "throws"),
        ("understand", "understood", "understood", "understands"),
        ("wake", "woke", "woken", "wakes"),
        ("wear", "wore", "worn", "wears"),
        ("weave", "wove", "woven", "weaves"),
        ("win", "won", "won", "wins"),
        ("wind", "wound", "wound", "winds"),
        ("write", "wrote", "written", "writes"),
    ]
}

/// Build lookup maps from the irregular table.
struct VerbLookup {
    /// base → (past, pp, 3sg)
    base_to_forms: HashMap<String, (String, String, String)>,
    /// any_form → base
    form_to_base: HashMap<String, String>,
}

impl VerbLookup {
    fn new() -> Self {
        let table = irregular_verb_table();
        let mut base_to_forms = HashMap::new();
        let mut form_to_base = HashMap::new();
        for (base, past, pp, sg3) in &table {
            base_to_forms.insert(
                base.to_string(),
                (past.to_string(), pp.to_string(), sg3.to_string()),
            );
            form_to_base.insert(base.to_string(), base.to_string());
            form_to_base.insert(past.to_string(), base.to_string());
            form_to_base.insert(pp.to_string(), base.to_string());
            form_to_base.insert(sg3.to_string(), base.to_string());
        }
        Self { base_to_forms, form_to_base }
    }

    fn get_base(&self, form: &str) -> Option<&str> {
        self.form_to_base.get(form).map(|s| s.as_str())
    }

    fn get_forms(&self, base: &str) -> Option<&(String, String, String)> {
        self.base_to_forms.get(base)
    }
}

/// Detect the tense of a single verb token.
fn detect_token_tense(token: &Token) -> Tense {
    if let Some(t) = token.features.get("Tense") {
        return match t.as_str() {
            "Past" => Tense::Past,
            "Pres" => Tense::Present,
            "Fut" => Tense::Future,
            _ => Tense::Unknown,
        };
    }
    let low = token.text.to_lowercase();
    let lookup = VerbLookup::new();
    if let Some(base) = lookup.get_base(&low) {
        if let Some((past, _pp, _sg3)) = lookup.get_forms(base) {
            if low == *past {
                return Tense::Past;
            }
        }
    }
    if low.ends_with("ed") {
        return Tense::Past;
    }
    if low.ends_with('s') && !low.ends_with("ss") {
        return Tense::Present;
    }
    Tense::Present
}

/// Convert a verb to a target tense.
fn conjugate(verb_text: &str, lemma: Option<&str>, target: Tense) -> String {
    let low = verb_text.to_lowercase();
    let lookup = VerbLookup::new();
    let base = lemma
        .map(|l| l.to_lowercase())
        .or_else(|| lookup.get_base(&low).map(|s| s.to_string()))
        .unwrap_or_else(|| {
            // Guess base form from regular past
            if low.ends_with("ed") {
                if low.ends_with("ied") {
                    format!("{}y", &low[..low.len() - 3])
                } else if low.ends_with("eed") {
                    low[..low.len() - 2].to_string()
                } else {
                    low[..low.len() - 2].to_string()
                }
            } else if low.ends_with('s') && !low.ends_with("ss") {
                if low.ends_with("ies") {
                    format!("{}y", &low[..low.len() - 3])
                } else if low.ends_with("es") {
                    low[..low.len() - 2].to_string()
                } else {
                    low[..low.len() - 1].to_string()
                }
            } else {
                low.clone()
            }
        });

    match target {
        Tense::Past | Tense::PastPerfect | Tense::PastProgressive => {
            if let Some((past, _pp, _sg3)) = lookup.get_forms(&base) {
                past.clone()
            } else {
                regular_past(&base)
            }
        }
        Tense::Present | Tense::PresentPerfect | Tense::PresentProgressive => {
            // Use base form (or 3sg if needed - we use base as default)
            if let Some((_past, _pp, sg3)) = lookup.get_forms(&base) {
                sg3.clone()
            } else {
                regular_present_3sg(&base)
            }
        }
        Tense::Future | Tense::FuturePerfect | Tense::FutureProgressive => {
            // Return base form; caller inserts "will"
            base
        }
        Tense::Unknown => base,
    }
}

fn regular_past(base: &str) -> String {
    if base.ends_with('e') {
        format!("{}d", base)
    } else if base.ends_with('y') && base.len() > 2 && !is_vowel_byte(base.as_bytes()[base.len() - 2]) {
        format!("{}ied", &base[..base.len() - 1])
    } else {
        format!("{}ed", base)
    }
}

fn regular_present_3sg(base: &str) -> String {
    if base.ends_with('s') || base.ends_with("sh") || base.ends_with("ch")
        || base.ends_with('x') || base.ends_with('z')
    {
        format!("{}es", base)
    } else if base.ends_with('y') && base.len() > 2 && !is_vowel_byte(base.as_bytes()[base.len() - 2]) {
        format!("{}ies", &base[..base.len() - 1])
    } else {
        format!("{}s", base)
    }
}

fn is_vowel_byte(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')
}

/// Handle auxiliary chain conversion for tense change.
fn convert_auxiliary(aux_text: &str, target: Tense) -> String {
    let low = aux_text.to_lowercase();
    match target {
        Tense::Past | Tense::PastPerfect | Tense::PastProgressive => {
            match low.as_str() {
                "is" | "am" => "was".to_string(),
                "are" => "were".to_string(),
                "has" => "had".to_string(),
                "have" => "had".to_string(),
                "do" | "does" => "did".to_string(),
                "will" | "shall" => "would".to_string(),
                "can" => "could".to_string(),
                "may" => "might".to_string(),
                _ => low,
            }
        }
        Tense::Present | Tense::PresentPerfect | Tense::PresentProgressive => {
            match low.as_str() {
                "was" => "is".to_string(),
                "were" => "are".to_string(),
                "had" => "has".to_string(),
                "did" => "does".to_string(),
                "would" => "will".to_string(),
                "could" => "can".to_string(),
                "might" => "may".to_string(),
                _ => low,
            }
        }
        Tense::Future | Tense::FuturePerfect | Tense::FutureProgressive => {
            match low.as_str() {
                "is" | "am" | "was" | "are" | "were" => "will be".to_string(),
                "has" | "had" | "have" => "will have".to_string(),
                "does" | "did" | "do" => "will".to_string(),
                _ => low,
            }
        }
        _ => low,
    }
}

pub struct TenseChangeTransform {
    pub target_tense: Tense,
}

impl TenseChangeTransform {
    pub fn new() -> Self {
        Self {
            target_tense: Tense::Past,
        }
    }

    pub fn with_target(target: Tense) -> Self {
        Self { target_tense: target }
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Inverse: determine original tense and convert back
        let original_tense = sentence.compute_features().tense.unwrap_or(Tense::Present);
        let reverse = TenseChangeTransform::with_target(original_tense);
        reverse.apply(sentence)
    }
}

impl BaseTransformation for TenseChangeTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::TenseChange
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        sentence.tokens.iter().any(|t| {
            t.pos_tag == Some(PosTag::Verb) || t.pos_tag == Some(PosTag::Aux)
        })
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        if !self.is_applicable(sentence) {
            return Err(TransformationError::PreconditionNotMet(
                "No tensed verb found".into(),
            ));
        }

        let root_idx = sentence.root_index();
        let mut new_tokens: Vec<Token> = Vec::new();
        let mut modified = Vec::new();
        let mut will_inserted = false;

        // For future tense, we may need to insert "will" before the main verb
        let needs_will = matches!(
            self.target_tense,
            Tense::Future | Tense::FuturePerfect | Tense::FutureProgressive
        );
        let has_aux = sentence.tokens.iter().any(|t| t.pos_tag == Some(PosTag::Aux));

        for (i, tok) in sentence.tokens.iter().enumerate() {
            if tok.pos_tag == Some(PosTag::Aux) {
                let converted = convert_auxiliary(&tok.text, self.target_tense);
                // Handle multi-word auxiliaries (e.g., "will be")
                for (j, part) in converted.split_whitespace().enumerate() {
                    new_tokens.push(Token::new(part, 0).with_pos(PosTag::Aux));
                }
                modified.push((i, i + 1));
                if needs_will && converted.contains("will") {
                    will_inserted = true;
                }
            } else if tok.pos_tag == Some(PosTag::Verb) && Some(i) == root_idx {
                // Insert "will" before main verb for future tense if no aux
                if needs_will && !will_inserted && !has_aux {
                    new_tokens.push(Token::new("will", 0).with_pos(PosTag::Aux));
                    will_inserted = true;
                    // Use base form for verb after "will"
                    let lookup = VerbLookup::new();
                    let low = tok.text.to_lowercase();
                    let base = tok.lemma.as_deref()
                        .map(|l| l.to_string())
                        .or_else(|| lookup.get_base(&low).map(|s| s.to_string()))
                        .unwrap_or_else(|| {
                            if low.ends_with("ed") {
                                low[..low.len()-2].to_string()
                            } else {
                                low
                            }
                        });
                    new_tokens.push(Token::new(&base, 0).with_pos(PosTag::Verb));
                } else {
                    let conjugated = conjugate(
                        &tok.text,
                        tok.lemma.as_deref(),
                        self.target_tense,
                    );
                    new_tokens.push(Token::new(&conjugated, 0).with_pos(PosTag::Verb));
                }
                modified.push((i, i + 1));
            } else {
                new_tokens.push(tok.clone());
            }
        }

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            modified,
            format!("Changed tense to {:?}", self.target_tense),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        false // changing tense changes temporal reference
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn past_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("walked", 1).with_pos(PosTag::Verb).with_lemma("walk"),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John walked.");
        s.dependency_edges = edges;
        s
    }

    fn present_sentence() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("walks", 1).with_pos(PosTag::Verb).with_lemma("walk"),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John walks.");
        s.dependency_edges = edges;
        s
    }

    fn irregular_sentence() -> Sentence {
        let tokens = vec![
            Token::new("She", 0).with_pos(PosTag::Pron),
            Token::new("went", 1).with_pos(PosTag::Verb).with_lemma("go"),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "She went.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = TenseChangeTransform::new();
        assert_eq!(t.kind(), TransformationKind::TenseChange);
    }

    #[test]
    fn test_past_to_present() {
        let t = TenseChangeTransform::with_target(Tense::Present);
        let result = t.apply(&past_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("walks") || text.contains("walk"));
    }

    #[test]
    fn test_present_to_past() {
        let t = TenseChangeTransform::with_target(Tense::Past);
        let result = t.apply(&present_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("walked"));
    }

    #[test]
    fn test_irregular_past_to_present() {
        let t = TenseChangeTransform::with_target(Tense::Present);
        let result = t.apply(&irregular_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("goes") || text.contains("go"));
    }

    #[test]
    fn test_to_future() {
        let t = TenseChangeTransform::with_target(Tense::Future);
        let result = t.apply(&past_sentence()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("will"));
    }

    #[test]
    fn test_not_meaning_preserving() {
        let t = TenseChangeTransform::new();
        assert!(!t.is_meaning_preserving());
    }

    #[test]
    fn test_not_applicable_no_verb() {
        let t = TenseChangeTransform::new();
        let tokens = vec![
            Token::new("Hello", 0).with_pos(PosTag::Intj),
            Token::new(".", 1).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, "Hello.");
        assert!(!t.is_applicable(&s));
    }

    #[test]
    fn test_auxiliary_conversion() {
        assert_eq!(convert_auxiliary("is", Tense::Past), "was");
        assert_eq!(convert_auxiliary("was", Tense::Present), "is");
        assert_eq!(convert_auxiliary("has", Tense::Past), "had");
        assert_eq!(convert_auxiliary("can", Tense::Past), "could");
    }
}
