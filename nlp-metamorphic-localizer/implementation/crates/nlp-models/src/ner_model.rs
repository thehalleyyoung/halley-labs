//! Rule-based Named Entity Recognition with gazetteers and pattern rules.

use shared_types::{EntityLabel, EntitySpan, PosTag, Token};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── Types ───────────────────────────────────────────────────────────────────

/// BIO tag for internal tagging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BioTag {
    B(EntityLabel),
    I(EntityLabel),
    O,
}

impl BioTag {
    pub fn label(&self) -> Option<EntityLabel> {
        match self {
            BioTag::B(l) | BioTag::I(l) => Some(*l),
            BioTag::O => None,
        }
    }
}

/// A single NER pattern rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NERRule {
    pub name: String,
    pub capitalized: Option<bool>,
    pub preceding_lower: Vec<String>,
    pub following_lower: Vec<String>,
    pub entity_type: EntityLabel,
    pub priority: u8,
}

/// A gazetteer mapping known entity strings to labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gazetteer {
    pub entries: HashMap<EntityLabel, HashSet<String>>,
}

impl Default for Gazetteer {
    fn default() -> Self {
        Self::new()
    }
}

impl Gazetteer {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn add(&mut self, label: EntityLabel, name: &str) {
        self.entries
            .entry(label)
            .or_default()
            .insert(name.to_lowercase());
    }

    pub fn lookup(&self, text: &str) -> Option<EntityLabel> {
        let lower = text.to_lowercase();
        for (label, set) in &self.entries {
            if set.contains(&lower) {
                return Some(*label);
            }
        }
        None
    }

    pub fn lookup_phrase(&self, tokens: &[Token], start: usize, max_len: usize) -> Option<(EntityLabel, usize)> {
        for len in (1..=max_len.min(tokens.len() - start)).rev() {
            let phrase: String = tokens[start..start + len]
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            for (label, set) in &self.entries {
                if set.contains(&phrase) {
                    return Some((*label, len));
                }
            }
        }
        None
    }
}

// ── RuleBasedNER ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RuleBasedNER {
    gazetteer: Gazetteer,
    rules: Vec<NERRule>,
}

impl Default for RuleBasedNER {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleBasedNER {
    pub fn new() -> Self {
        Self {
            gazetteer: build_default_gazetteer(),
            rules: build_default_rules(),
        }
    }

    /// Tag tokens, returning entity spans.
    pub fn recognize(&self, tokens: &[Token]) -> Vec<EntitySpan> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut bio_tags = vec![BioTag::O; tokens.len()];

        // Phase 1: gazetteer matching (longest first)
        let mut i = 0;
        while i < tokens.len() {
            if let Some((label, len)) = self.gazetteer.lookup_phrase(tokens, i, 4) {
                bio_tags[i] = BioTag::B(label);
                for j in 1..len {
                    bio_tags[i + j] = BioTag::I(label);
                }
                i += len;
                continue;
            }
            i += 1;
        }

        // Phase 2: pattern rules for untagged tokens
        for i in 0..tokens.len() {
            if bio_tags[i] != BioTag::O {
                continue;
            }
            if let Some(label) = self.apply_rules(tokens, i) {
                bio_tags[i] = BioTag::B(label);
            }
        }

        // Phase 3: capitalization heuristic for remaining unknowns
        for i in 0..tokens.len() {
            if bio_tags[i] != BioTag::O {
                continue;
            }
            let word = &tokens[i].text;
            if i > 0 && is_capitalized(word) && word.len() > 1 {
                // Not sentence-initial → likely entity
                let label = self.disambiguate_by_context(tokens, i);
                bio_tags[i] = BioTag::B(label);
            }
        }

        // Phase 4: merge adjacent same-type B/I into spans
        self.merge_bio_to_spans(tokens, &bio_tags)
    }

    /// Get BIO tags for the sequence.
    pub fn tag_bio(&self, tokens: &[Token]) -> Vec<BioTag> {
        let spans = self.recognize(tokens);
        let mut tags = vec![BioTag::O; tokens.len()];
        for span in &spans {
            if span.start < tokens.len() {
                tags[span.start] = BioTag::B(span.label);
                for j in (span.start + 1)..span.end.min(tokens.len()) {
                    tags[j] = BioTag::I(span.label);
                }
            }
        }
        tags
    }

    // ── internals ───────────────────────────────────────────────────────────

    fn apply_rules(&self, tokens: &[Token], idx: usize) -> Option<EntityLabel> {
        let word = &tokens[idx].text;
        let prev_lower = if idx > 0 {
            Some(tokens[idx - 1].text.to_lowercase())
        } else {
            None
        };
        let next_lower = if idx + 1 < tokens.len() {
            Some(tokens[idx + 1].text.to_lowercase())
        } else {
            None
        };

        let mut best: Option<(EntityLabel, u8)> = None;
        for rule in &self.rules {
            // Check capitalization constraint
            if let Some(cap) = rule.capitalized {
                if cap != is_capitalized(word) {
                    continue;
                }
            }
            // Check preceding context
            if !rule.preceding_lower.is_empty() {
                match &prev_lower {
                    Some(pw) if rule.preceding_lower.contains(pw) => {}
                    _ => continue,
                }
            }
            // Check following context
            if !rule.following_lower.is_empty() {
                match &next_lower {
                    Some(nw) if rule.following_lower.contains(nw) => {}
                    _ => continue,
                }
            }
            match best {
                Some((_, bp)) if rule.priority > bp => {
                    best = Some((rule.entity_type, rule.priority));
                }
                None => best = Some((rule.entity_type, rule.priority)),
                _ => {}
            }
        }
        best.map(|(l, _)| l)
    }

    fn disambiguate_by_context(&self, tokens: &[Token], idx: usize) -> EntityLabel {
        let prev_lower = if idx > 0 {
            tokens[idx - 1].text.to_lowercase()
        } else {
            String::new()
        };

        // Title → Person
        if matches!(prev_lower.as_str(), "mr" | "mr." | "mrs" | "mrs."
            | "ms" | "ms." | "dr" | "dr." | "prof" | "prof."
            | "president" | "senator" | "judge" | "sir" | "lady")
        {
            return EntityLabel::Person;
        }

        // Location indicators
        if matches!(prev_lower.as_str(), "in" | "at" | "from" | "near" | "to") {
            return EntityLabel::Location;
        }

        // If POS tag is available and is Noun-like, default to Organization
        if let Some(tag) = tokens[idx].pos_tag {
            if tag == PosTag::Noun {
                return EntityLabel::Organization;
            }
        }

        EntityLabel::Person
    }

    fn merge_bio_to_spans(&self, tokens: &[Token], bio: &[BioTag]) -> Vec<EntitySpan> {
        let mut spans = Vec::new();
        let mut i = 0;
        while i < bio.len() {
            if let BioTag::B(label) = bio[i] {
                let start = i;
                i += 1;
                while i < bio.len() {
                    match bio[i] {
                        BioTag::I(l) if l == label => i += 1,
                        _ => break,
                    }
                }
                let text: String = tokens[start..i]
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                spans.push(EntitySpan {
                    start,
                    end: i,
                    text,
                    label,
                    confidence: 0.8,
                });
            } else {
                i += 1;
            }
        }
        // Merge adjacent same-type spans
        merge_adjacent_spans(&mut spans)
    }
}

fn merge_adjacent_spans(spans: &mut Vec<EntitySpan>) -> Vec<EntitySpan> {
    if spans.is_empty() {
        return Vec::new();
    }
    let mut merged: Vec<EntitySpan> = Vec::new();
    for span in spans.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.label == span.label && last.end == span.start {
                last.end = span.end;
                last.text = format!("{} {}", last.text, span.text);
                continue;
            }
        }
        merged.push(span);
    }
    merged
}

fn is_capitalized(word: &str) -> bool {
    word.chars().next().map_or(false, |c| c.is_uppercase())
}

// ── Default gazetteers ──────────────────────────────────────────────────────

fn build_default_gazetteer() -> Gazetteer {
    let mut g = Gazetteer::new();

    // Person names (50+)
    let persons = [
        "james", "john", "robert", "michael", "william", "david", "richard",
        "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
        "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
        "susan", "jessica", "sarah", "karen", "nancy", "lisa", "betty",
        "margaret", "sandra", "ashley", "dorothy", "kimberly", "emily",
        "donna", "michelle", "carol", "amanda", "melissa", "deborah",
        "stephanie", "rebecca", "sharon", "laura", "cynthia", "alice",
        "obama", "trump", "biden", "einstein", "newton", "shakespeare",
    ];
    for name in &persons {
        g.add(EntityLabel::Person, name);
    }

    // Locations (50+)
    let locations = [
        "new york", "los angeles", "chicago", "houston", "phoenix",
        "philadelphia", "san antonio", "san diego", "dallas", "san jose",
        "london", "paris", "tokyo", "berlin", "madrid", "rome", "moscow",
        "beijing", "sydney", "toronto", "california", "texas", "florida",
        "ohio", "michigan", "georgia", "virginia", "washington",
        "massachusetts", "arizona", "colorado", "minnesota", "wisconsin",
        "oregon", "nevada", "utah", "montana", "hawaii", "alaska",
        "america", "europe", "asia", "africa", "australia", "canada",
        "mexico", "brazil", "india", "china", "japan", "france",
        "germany", "italy", "spain", "russia", "england", "ireland",
        "scotland", "wales", "manhattan", "brooklyn", "boston",
    ];
    for loc in &locations {
        g.add(EntityLabel::Location, loc);
    }

    // Organizations (50+)
    let orgs = [
        "google", "apple", "microsoft", "amazon", "facebook", "meta",
        "tesla", "netflix", "twitter", "uber", "airbnb", "spotify",
        "ibm", "intel", "oracle", "cisco", "samsung", "sony", "honda",
        "toyota", "bmw", "ford", "nasa", "fbi", "cia", "nsa", "un",
        "nato", "who", "imf", "mit", "harvard", "stanford", "yale",
        "princeton", "oxford", "cambridge", "berkeley", "columbia",
        "nyu", "ucla", "congress", "parliament", "senate", "pentagon",
        "reuters", "bloomberg", "cnn", "bbc", "nbc", "abc", "cbs",
        "fox", "disney", "warner", "paramount", "nike", "adidas",
    ];
    for org in &orgs {
        g.add(EntityLabel::Organization, org);
    }

    // Date patterns
    let months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "oct", "nov", "dec", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday",
    ];
    for m in &months {
        g.add(EntityLabel::Date, m);
    }

    g
}

// ── Default pattern rules ───────────────────────────────────────────────────

fn build_default_rules() -> Vec<NERRule> {
    vec![
        // Mr./Dr./Prof. + Capitalized → Person
        NERRule {
            name: "title_person".into(),
            capitalized: Some(true),
            preceding_lower: vec![
                "mr".into(), "mr.".into(), "mrs".into(), "mrs.".into(),
                "ms".into(), "ms.".into(), "dr".into(), "dr.".into(),
                "prof".into(), "prof.".into(), "sir".into(), "lady".into(),
            ],
            following_lower: vec![],
            entity_type: EntityLabel::Person,
            priority: 10,
        },
        // "in"/"at"/"from" + Capitalized → Location
        NERRule {
            name: "prep_location".into(),
            capitalized: Some(true),
            preceding_lower: vec![
                "in".into(), "at".into(), "from".into(), "near".into(),
                "to".into(), "across".into(), "through".into(),
            ],
            following_lower: vec![],
            entity_type: EntityLabel::Location,
            priority: 8,
        },
        // Capitalized + "Inc"/"Corp"/"Ltd" → Organization
        NERRule {
            name: "org_suffix".into(),
            capitalized: Some(true),
            preceding_lower: vec![],
            following_lower: vec![
                "inc".into(), "inc.".into(), "corp".into(), "corp.".into(),
                "ltd".into(), "ltd.".into(), "llc".into(), "co".into(),
                "co.".into(), "group".into(), "foundation".into(),
            ],
            entity_type: EntityLabel::Organization,
            priority: 9,
        },
        // "President"/"Senator" + Cap → Person
        NERRule {
            name: "political_title_person".into(),
            capitalized: Some(true),
            preceding_lower: vec![
                "president".into(), "senator".into(), "governor".into(),
                "minister".into(), "chancellor".into(), "king".into(),
                "queen".into(), "prince".into(), "princess".into(),
                "mayor".into(), "judge".into(),
            ],
            following_lower: vec![],
            entity_type: EntityLabel::Person,
            priority: 9,
        },
        // "$" + number → Money
        NERRule {
            name: "dollar_money".into(),
            capitalized: None,
            preceding_lower: vec!["$".into()],
            following_lower: vec![],
            entity_type: EntityLabel::Money,
            priority: 10,
        },
        // number + "%" → Percent
        NERRule {
            name: "pct_percent".into(),
            capitalized: None,
            preceding_lower: vec![],
            following_lower: vec!["%".into(), "percent".into(), "pct".into()],
            entity_type: EntityLabel::Percent,
            priority: 10,
        },
    ]
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(words: &[&str]) -> Vec<Token> {
        words
            .iter()
            .enumerate()
            .map(|(i, w)| Token::new(*w, i, 0, w.len()))
            .collect()
    }

    #[test]
    fn test_gazetteer_person() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["I", "met", "John", "yesterday"]);
        let spans = ner.recognize(&tokens);
        assert!(spans.iter().any(|s| s.label == EntityLabel::Person && s.text.to_lowercase().contains("john")));
    }

    #[test]
    fn test_gazetteer_location() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["She", "lives", "in", "London"]);
        let spans = ner.recognize(&tokens);
        assert!(spans.iter().any(|s| s.label == EntityLabel::Location && s.text.to_lowercase().contains("london")));
    }

    #[test]
    fn test_gazetteer_organization() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["He", "works", "at", "Google"]);
        let spans = ner.recognize(&tokens);
        assert!(spans.iter().any(|s| s.label == EntityLabel::Organization && s.text.to_lowercase().contains("google")));
    }

    #[test]
    fn test_multi_word_location() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["I", "visited", "New", "York"]);
        let spans = ner.recognize(&tokens);
        assert!(spans.iter().any(|s| s.label == EntityLabel::Location));
    }

    #[test]
    fn test_title_person_rule() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["Dr.", "Smith", "arrived"]);
        let spans = ner.recognize(&tokens);
        assert!(spans.iter().any(|s| s.label == EntityLabel::Person));
    }

    #[test]
    fn test_capitalization_heuristic() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["I", "saw", "Xanadu", "yesterday"]);
        let spans = ner.recognize(&tokens);
        // "Xanadu" is capitalized and not sentence-initial
        assert!(!spans.is_empty());
    }

    #[test]
    fn test_bio_tags() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["John", "works", "here"]);
        let bio = ner.tag_bio(&tokens);
        assert_eq!(bio.len(), 3);
        assert!(matches!(bio[0], BioTag::B(EntityLabel::Person)));
    }

    #[test]
    fn test_no_entities_in_common_text() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["the", "cat", "sat", "on", "the", "mat"]);
        let spans = ner.recognize(&tokens);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_date_recognition() {
        let ner = RuleBasedNER::new();
        let tokens = make_tokens(&["It", "happened", "in", "January"]);
        let spans = ner.recognize(&tokens);
        assert!(spans.iter().any(|s| s.label == EntityLabel::Date));
    }

    #[test]
    fn test_empty_input() {
        let ner = RuleBasedNER::new();
        let spans = ner.recognize(&[]);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_merge_adjacent() {
        let mut spans = vec![
            EntitySpan { start: 0, end: 1, text: "New".into(), label: EntityLabel::Location, confidence: 0.8 },
            EntitySpan { start: 1, end: 2, text: "York".into(), label: EntityLabel::Location, confidence: 0.8 },
        ];
        let merged = merge_adjacent_spans(&mut spans);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "New York");
    }
}
