//! English locale-specific rules and transformations.

use crate::morphology::{InflectionTable, InflectionType, MorphologicalRule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// English-specific locale with comprehensive morphological data.
pub struct EnglishLocale {
    irregular_verbs: HashMap<String, IrregularVerb>,
    irregular_plurals: HashMap<String, String>,
    copular_verbs: Vec<String>,
    auxiliary_verbs: Vec<String>,
    modal_verbs: Vec<String>,
    indefinite_determiners: Vec<String>,
    subcategorization_frames: HashMap<String, Vec<SubcatFrame>>,
}

/// Irregular verb forms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrregularVerb {
    pub base: String,
    pub third_singular: String,
    pub past: String,
    pub past_participle: String,
    pub present_participle: String,
}

/// Subcategorization frame for a verb.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubcatFrame {
    Intransitive,
    Transitive,
    Ditransitive,
    Copular,
    Unaccusative,
    SententialComplement,
    InfinitivalComplement,
}

impl EnglishLocale {
    pub fn new() -> Self {
        let mut locale = Self {
            irregular_verbs: HashMap::new(),
            irregular_plurals: HashMap::new(),
            copular_verbs: vec![
                "be".into(), "seem".into(), "appear".into(), "become".into(),
                "remain".into(), "look".into(), "sound".into(), "feel".into(),
                "taste".into(), "smell".into(),
            ],
            auxiliary_verbs: vec![
                "be".into(), "have".into(), "do".into(),
                "will".into(), "would".into(), "shall".into(), "should".into(),
                "can".into(), "could".into(), "may".into(), "might".into(),
                "must".into(),
            ],
            modal_verbs: vec![
                "can".into(), "could".into(), "may".into(), "might".into(),
                "shall".into(), "should".into(), "will".into(), "would".into(),
                "must".into(),
            ],
            indefinite_determiners: vec![
                "a".into(), "an".into(), "some".into(), "any".into(),
                "several".into(), "many".into(), "few".into(), "no".into(),
            ],
            subcategorization_frames: HashMap::new(),
        };
        locale.load_irregular_verbs();
        locale.load_irregular_plurals();
        locale.load_subcategorization();
        locale
    }

    fn load_irregular_verbs(&mut self) {
        let verbs = vec![
            ("be", "is", "was", "been", "being"),
            ("have", "has", "had", "had", "having"),
            ("do", "does", "did", "done", "doing"),
            ("go", "goes", "went", "gone", "going"),
            ("say", "says", "said", "said", "saying"),
            ("get", "gets", "got", "gotten", "getting"),
            ("make", "makes", "made", "made", "making"),
            ("know", "knows", "knew", "known", "knowing"),
            ("think", "thinks", "thought", "thought", "thinking"),
            ("take", "takes", "took", "taken", "taking"),
            ("see", "sees", "saw", "seen", "seeing"),
            ("come", "comes", "came", "come", "coming"),
            ("give", "gives", "gave", "given", "giving"),
            ("find", "finds", "found", "found", "finding"),
            ("tell", "tells", "told", "told", "telling"),
            ("write", "writes", "wrote", "written", "writing"),
            ("sit", "sits", "sat", "sat", "sitting"),
            ("stand", "stands", "stood", "stood", "standing"),
            ("run", "runs", "ran", "run", "running"),
            ("read", "reads", "read", "read", "reading"),
            ("eat", "eats", "ate", "eaten", "eating"),
            ("drink", "drinks", "drank", "drunk", "drinking"),
            ("break", "breaks", "broke", "broken", "breaking"),
            ("drive", "drives", "drove", "driven", "driving"),
            ("speak", "speaks", "spoke", "spoken", "speaking"),
            ("choose", "chooses", "chose", "chosen", "choosing"),
            ("begin", "begins", "began", "begun", "beginning"),
            ("swim", "swims", "swam", "swum", "swimming"),
            ("sing", "sings", "sang", "sung", "singing"),
            ("ring", "rings", "rang", "rung", "ringing"),
            ("throw", "throws", "threw", "thrown", "throwing"),
            ("grow", "grows", "grew", "grown", "growing"),
            ("draw", "draws", "drew", "drawn", "drawing"),
            ("fly", "flies", "flew", "flown", "flying"),
            ("show", "shows", "showed", "shown", "showing"),
            ("blow", "blows", "blew", "blown", "blowing"),
            ("freeze", "freezes", "froze", "frozen", "freezing"),
            ("steal", "steals", "stole", "stolen", "stealing"),
            ("wear", "wears", "wore", "worn", "wearing"),
            ("tear", "tears", "tore", "torn", "tearing"),
            ("bite", "bites", "bit", "bitten", "biting"),
            ("hide", "hides", "hid", "hidden", "hiding"),
            ("fall", "falls", "fell", "fallen", "falling"),
            ("catch", "catches", "caught", "caught", "catching"),
            ("teach", "teaches", "taught", "taught", "teaching"),
            ("buy", "buys", "bought", "bought", "buying"),
            ("bring", "brings", "brought", "brought", "bringing"),
            ("fight", "fights", "fought", "fought", "fighting"),
            ("hold", "holds", "held", "held", "holding"),
            ("lead", "leads", "led", "led", "leading"),
            ("lose", "loses", "lost", "lost", "losing"),
            ("pay", "pays", "paid", "paid", "paying"),
            ("build", "builds", "built", "built", "building"),
            ("send", "sends", "sent", "sent", "sending"),
            ("spend", "spends", "spent", "spent", "spending"),
            ("leave", "leaves", "left", "left", "leaving"),
            ("keep", "keeps", "kept", "kept", "keeping"),
            ("let", "lets", "let", "let", "letting"),
            ("put", "puts", "put", "put", "putting"),
            ("cut", "cuts", "cut", "cut", "cutting"),
            ("set", "sets", "set", "set", "setting"),
            ("hit", "hits", "hit", "hit", "hitting"),
            ("shut", "shuts", "shut", "shut", "shutting"),
        ];

        for (base, third, past, pp, pres_p) in verbs {
            self.irregular_verbs.insert(
                base.to_string(),
                IrregularVerb {
                    base: base.to_string(),
                    third_singular: third.to_string(),
                    past: past.to_string(),
                    past_participle: pp.to_string(),
                    present_participle: pres_p.to_string(),
                },
            );
        }
    }

    fn load_irregular_plurals(&mut self) {
        let plurals = vec![
            ("child", "children"), ("man", "men"), ("woman", "women"),
            ("foot", "feet"), ("tooth", "teeth"), ("goose", "geese"),
            ("mouse", "mice"), ("person", "people"), ("ox", "oxen"),
            ("sheep", "sheep"), ("fish", "fish"), ("deer", "deer"),
            ("series", "series"), ("species", "species"), ("aircraft", "aircraft"),
            ("knife", "knives"), ("life", "lives"), ("wife", "wives"),
            ("leaf", "leaves"), ("half", "halves"), ("self", "selves"),
            ("cactus", "cacti"), ("focus", "foci"), ("fungus", "fungi"),
            ("nucleus", "nuclei"), ("stimulus", "stimuli"),
            ("analysis", "analyses"), ("basis", "bases"), ("crisis", "crises"),
            ("hypothesis", "hypotheses"), ("thesis", "theses"),
            ("criterion", "criteria"), ("phenomenon", "phenomena"),
            ("datum", "data"), ("medium", "media"),
        ];

        for (singular, plural) in plurals {
            self.irregular_plurals
                .insert(singular.to_string(), plural.to_string());
        }
    }

    fn load_subcategorization(&mut self) {
        let frames = vec![
            ("give", vec![SubcatFrame::Ditransitive, SubcatFrame::Transitive]),
            ("send", vec![SubcatFrame::Ditransitive, SubcatFrame::Transitive]),
            ("tell", vec![SubcatFrame::Ditransitive, SubcatFrame::SententialComplement]),
            ("show", vec![SubcatFrame::Ditransitive, SubcatFrame::Transitive]),
            ("run", vec![SubcatFrame::Intransitive]),
            ("walk", vec![SubcatFrame::Intransitive]),
            ("sleep", vec![SubcatFrame::Intransitive]),
            ("appear", vec![SubcatFrame::Unaccusative, SubcatFrame::Copular]),
            ("seem", vec![SubcatFrame::Copular, SubcatFrame::InfinitivalComplement]),
            ("become", vec![SubcatFrame::Copular]),
            ("be", vec![SubcatFrame::Copular]),
            ("believe", vec![SubcatFrame::SententialComplement, SubcatFrame::Transitive]),
            ("think", vec![SubcatFrame::SententialComplement, SubcatFrame::Intransitive]),
            ("know", vec![SubcatFrame::SententialComplement, SubcatFrame::Transitive]),
            ("eat", vec![SubcatFrame::Transitive, SubcatFrame::Intransitive]),
            ("read", vec![SubcatFrame::Transitive, SubcatFrame::Intransitive]),
            ("write", vec![SubcatFrame::Transitive, SubcatFrame::Ditransitive]),
            ("break", vec![SubcatFrame::Transitive, SubcatFrame::Intransitive]),
            ("open", vec![SubcatFrame::Transitive, SubcatFrame::Intransitive]),
            ("close", vec![SubcatFrame::Transitive, SubcatFrame::Intransitive]),
        ];

        for (verb, subcats) in frames {
            self.subcategorization_frames
                .insert(verb.to_string(), subcats);
        }
    }

    /// Get the past participle form of a verb.
    pub fn past_participle(&self, verb: &str) -> String {
        if let Some(irregular) = self.irregular_verbs.get(verb) {
            return irregular.past_participle.clone();
        }
        // Regular: add -ed (simplified).
        Self::regular_past_participle(verb)
    }

    /// Get the past tense form of a verb.
    pub fn past_tense(&self, verb: &str) -> String {
        if let Some(irregular) = self.irregular_verbs.get(verb) {
            return irregular.past.clone();
        }
        Self::regular_past(verb)
    }

    /// Get the third person singular present form.
    pub fn third_singular(&self, verb: &str) -> String {
        if let Some(irregular) = self.irregular_verbs.get(verb) {
            return irregular.third_singular.clone();
        }
        Self::regular_third_singular(verb)
    }

    /// Get the present participle form.
    pub fn present_participle(&self, verb: &str) -> String {
        if let Some(irregular) = self.irregular_verbs.get(verb) {
            return irregular.present_participle.clone();
        }
        Self::regular_present_participle(verb)
    }

    /// Get the plural form of a noun.
    pub fn pluralize(&self, noun: &str) -> String {
        if let Some(irregular) = self.irregular_plurals.get(noun) {
            return irregular.clone();
        }
        Self::regular_plural(noun)
    }

    /// Check if a verb is copular.
    pub fn is_copular(&self, verb: &str) -> bool {
        self.copular_verbs.contains(&verb.to_string())
    }

    /// Check if a verb is a modal.
    pub fn is_modal(&self, verb: &str) -> bool {
        self.modal_verbs.contains(&verb.to_string())
    }

    /// Check if a determiner is indefinite.
    pub fn is_indefinite(&self, det: &str) -> bool {
        self.indefinite_determiners.contains(&det.to_lowercase())
    }

    /// Get subcategorization frames for a verb.
    pub fn get_subcat_frames(&self, verb: &str) -> Vec<&SubcatFrame> {
        self.subcategorization_frames
            .get(verb)
            .map(|frames| frames.iter().collect())
            .unwrap_or_default()
    }

    /// Check if a verb can be passivized (must have a transitive or ditransitive frame).
    pub fn can_passivize(&self, verb: &str) -> bool {
        self.subcategorization_frames
            .get(verb)
            .map(|frames| {
                frames.iter().any(|f| {
                    matches!(f, SubcatFrame::Transitive | SubcatFrame::Ditransitive)
                })
            })
            .unwrap_or(true) // Default: assume transitive if unknown.
    }

    /// Check if a verb supports dative alternation.
    pub fn supports_dative_alternation(&self, verb: &str) -> bool {
        self.subcategorization_frames
            .get(verb)
            .map(|frames| frames.iter().any(|f| matches!(f, SubcatFrame::Ditransitive)))
            .unwrap_or(false)
    }

    // ── Regular inflection rules ────────────────────────────────────────

    fn regular_past_participle(verb: &str) -> String {
        Self::regular_past(verb)
    }

    fn regular_past(verb: &str) -> String {
        if verb.ends_with('e') {
            format!("{}d", verb)
        } else if verb.ends_with('y')
            && verb.len() > 2
            && !Self::is_vowel(verb.chars().rev().nth(1).unwrap_or('a'))
        {
            format!("{}ied", &verb[..verb.len() - 1])
        } else if Self::should_double_final(verb) {
            format!("{}{}ed", verb, verb.chars().last().unwrap())
        } else {
            format!("{}ed", verb)
        }
    }

    fn regular_third_singular(verb: &str) -> String {
        if verb.ends_with('s')
            || verb.ends_with('x')
            || verb.ends_with('z')
            || verb.ends_with("ch")
            || verb.ends_with("sh")
        {
            format!("{}es", verb)
        } else if verb.ends_with('y')
            && verb.len() > 1
            && !Self::is_vowel(verb.chars().rev().nth(1).unwrap_or('a'))
        {
            format!("{}ies", &verb[..verb.len() - 1])
        } else {
            format!("{}s", verb)
        }
    }

    fn regular_present_participle(verb: &str) -> String {
        if verb.ends_with('e') && !verb.ends_with("ee") {
            format!("{}ing", &verb[..verb.len() - 1])
        } else if Self::should_double_final(verb) {
            format!("{}{}ing", verb, verb.chars().last().unwrap())
        } else {
            format!("{}ing", verb)
        }
    }

    fn regular_plural(noun: &str) -> String {
        if noun.ends_with('s')
            || noun.ends_with('x')
            || noun.ends_with('z')
            || noun.ends_with("ch")
            || noun.ends_with("sh")
        {
            format!("{}es", noun)
        } else if noun.ends_with('y')
            && noun.len() > 1
            && !Self::is_vowel(noun.chars().rev().nth(1).unwrap_or('a'))
        {
            format!("{}ies", &noun[..noun.len() - 1])
        } else if noun.ends_with('f') {
            format!("{}ves", &noun[..noun.len() - 1])
        } else if noun.ends_with("fe") {
            format!("{}ves", &noun[..noun.len() - 2])
        } else {
            format!("{}s", noun)
        }
    }

    fn is_vowel(c: char) -> bool {
        matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
    }

    fn should_double_final(word: &str) -> bool {
        if word.len() < 3 {
            return false;
        }
        let chars: Vec<char> = word.chars().collect();
        let last = chars[chars.len() - 1];
        let second_last = chars[chars.len() - 2];
        let third_last = chars[chars.len() - 3];

        last.is_ascii_alphabetic()
            && !Self::is_vowel(last)
            && Self::is_vowel(second_last)
            && !Self::is_vowel(third_last)
            && !matches!(last, 'w' | 'x' | 'y')
    }
}

impl Default for EnglishLocale {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_irregular_past_participle() {
        let locale = EnglishLocale::new();
        assert_eq!(locale.past_participle("write"), "written");
        assert_eq!(locale.past_participle("go"), "gone");
        assert_eq!(locale.past_participle("be"), "been");
        assert_eq!(locale.past_participle("eat"), "eaten");
    }

    #[test]
    fn test_regular_past_participle() {
        let locale = EnglishLocale::new();
        assert_eq!(locale.past_participle("walk"), "walked");
        assert_eq!(locale.past_participle("chase"), "chased");
        assert_eq!(locale.past_participle("carry"), "carried");
    }

    #[test]
    fn test_irregular_past_tense() {
        let locale = EnglishLocale::new();
        assert_eq!(locale.past_tense("go"), "went");
        assert_eq!(locale.past_tense("see"), "saw");
        assert_eq!(locale.past_tense("run"), "ran");
    }

    #[test]
    fn test_third_singular() {
        let locale = EnglishLocale::new();
        assert_eq!(locale.third_singular("have"), "has");
        assert_eq!(locale.third_singular("be"), "is");
        assert_eq!(locale.third_singular("walk"), "walks");
        assert_eq!(locale.third_singular("watch"), "watches");
        assert_eq!(locale.third_singular("carry"), "carries");
    }

    #[test]
    fn test_pluralization() {
        let locale = EnglishLocale::new();
        assert_eq!(locale.pluralize("child"), "children");
        assert_eq!(locale.pluralize("mouse"), "mice");
        assert_eq!(locale.pluralize("cat"), "cats");
        assert_eq!(locale.pluralize("box"), "boxes");
        assert_eq!(locale.pluralize("baby"), "babies");
        assert_eq!(locale.pluralize("leaf"), "leaves");
    }

    #[test]
    fn test_copular_verbs() {
        let locale = EnglishLocale::new();
        assert!(locale.is_copular("be"));
        assert!(locale.is_copular("seem"));
        assert!(!locale.is_copular("run"));
    }

    #[test]
    fn test_modal_verbs() {
        let locale = EnglishLocale::new();
        assert!(locale.is_modal("can"));
        assert!(locale.is_modal("must"));
        assert!(!locale.is_modal("run"));
    }

    #[test]
    fn test_indefinite_determiners() {
        let locale = EnglishLocale::new();
        assert!(locale.is_indefinite("a"));
        assert!(locale.is_indefinite("an"));
        assert!(locale.is_indefinite("some"));
        assert!(!locale.is_indefinite("the"));
    }

    #[test]
    fn test_passivization_check() {
        let locale = EnglishLocale::new();
        assert!(locale.can_passivize("give"));
        assert!(locale.can_passivize("eat"));
        assert!(!locale.can_passivize("run"));
    }

    #[test]
    fn test_dative_alternation_check() {
        let locale = EnglishLocale::new();
        assert!(locale.supports_dative_alternation("give"));
        assert!(locale.supports_dative_alternation("send"));
        assert!(!locale.supports_dative_alternation("eat"));
    }

    #[test]
    fn test_present_participle() {
        let locale = EnglishLocale::new();
        assert_eq!(locale.present_participle("write"), "writing");
        assert_eq!(locale.present_participle("go"), "going");
        assert_eq!(locale.present_participle("run"), "running");
        assert_eq!(locale.present_participle("walk"), "walking");
    }
}
