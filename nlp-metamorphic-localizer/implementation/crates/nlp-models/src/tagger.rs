//! Rule-based POS tagger with lexicon, suffix rules, and bigram probabilities.

use shared_types::{PosTag, Token};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Rule types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuffixRule {
    pub suffix: String,
    pub pos_tag: PosTag,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualRule {
    pub prev_tag: Option<PosTag>,
    pub current_lower: Option<String>,
    pub next_lower: Option<String>,
    pub assigned_tag: PosTag,
    pub priority: u8,
}

// ── RuleBasedTagger ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RuleBasedTagger {
    lexicon: HashMap<String, PosTag>,
    suffix_rules: Vec<SuffixRule>,
    contextual_rules: Vec<ContextualRule>,
    bigram_probs: HashMap<(PosTag, PosTag), f64>,
}

impl Default for RuleBasedTagger {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleBasedTagger {
    pub fn new() -> Self {
        let lexicon = build_lexicon();
        let suffix_rules = build_suffix_rules();
        let contextual_rules = build_contextual_rules();
        let bigram_probs = build_bigram_probs();
        Self { lexicon, suffix_rules, contextual_rules, bigram_probs }
    }

    /// Tag a sequence of tokens in place, filling in `tok.pos_tag`.
    pub fn tag(&self, tokens: &mut [Token]) {
        // Pass 1: lexicon + suffix lookup
        for tok in tokens.iter_mut() {
            let lower = tok.text.to_lowercase();
            if let Some(tag) = self.lexicon.get(&lower) {
                tok.pos_tag = Some(*tag);
            } else {
                tok.pos_tag = Some(self.tag_by_suffix(&lower));
            }
        }

        // Pass 2: contextual rules
        let snapshot: Vec<Option<PosTag>> = tokens.iter().map(|t| t.pos_tag).collect();
        for i in 0..tokens.len() {
            let prev = if i > 0 { snapshot[i - 1] } else { None };
            let next_lower = if i + 1 < tokens.len() {
                Some(tokens[i + 1].text.to_lowercase())
            } else {
                None
            };
            let cur_lower = tokens[i].text.to_lowercase();
            if let Some(tag) = self.apply_contextual(&prev, &cur_lower, &next_lower) {
                tokens[i].pos_tag = Some(tag);
            }
        }

        // Pass 3: Viterbi-like bigram disambiguation
        self.viterbi_smooth(tokens);
    }

    /// Tag a single word in isolation (no context).
    pub fn tag_word(&self, word: &str) -> PosTag {
        let lower = word.to_lowercase();
        if let Some(tag) = self.lexicon.get(&lower) {
            return *tag;
        }
        self.tag_by_suffix(&lower)
    }

    // ── internals ───────────────────────────────────────────────────────────

    fn tag_by_suffix(&self, lower: &str) -> PosTag {
        let mut best: Option<(&SuffixRule, usize)> = None;
        for rule in &self.suffix_rules {
            if lower.ends_with(&rule.suffix) {
                let len = rule.suffix.len();
                match &best {
                    Some((_, blen)) if len > *blen => best = Some((rule, len)),
                    Some((br, blen)) if len == *blen && rule.priority > br.priority => {
                        best = Some((rule, len));
                    }
                    None => best = Some((rule, len)),
                    _ => {}
                }
            }
        }
        if let Some((rule, _)) = best {
            return rule.pos_tag;
        }
        // Fallback heuristics
        if lower.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',') {
            return PosTag::Num;
        }
        if lower.chars().next().map_or(false, |c| c.is_uppercase()) {
            return PosTag::Noun;
        }
        PosTag::Noun
    }

    fn apply_contextual(
        &self,
        prev_tag: &Option<PosTag>,
        cur_lower: &str,
        next_lower: &Option<String>,
    ) -> Option<PosTag> {
        let mut best: Option<(PosTag, u8)> = None;
        for rule in &self.contextual_rules {
            let prev_ok = match &rule.prev_tag {
                Some(pt) => prev_tag.as_ref() == Some(pt),
                None => true,
            };
            let cur_ok = match &rule.current_lower {
                Some(cw) => cur_lower == cw.as_str(),
                None => true,
            };
            let next_ok = match &rule.next_lower {
                Some(nw) => next_lower.as_deref() == Some(nw.as_str()),
                None => true,
            };
            if prev_ok && cur_ok && next_ok {
                match best {
                    Some((_, bp)) if rule.priority > bp => {
                        best = Some((rule.assigned_tag, rule.priority));
                    }
                    None => best = Some((rule.assigned_tag, rule.priority)),
                    _ => {}
                }
            }
        }
        best.map(|(t, _)| t)
    }

    fn viterbi_smooth(&self, tokens: &mut [Token]) {
        if tokens.len() < 2 {
            return;
        }
        // A simplified one-pass forward smoothing:
        // If current tag has a very low bigram probability with previous, and
        // there is an alternative that is much better, switch.
        let candidates = [
            PosTag::Noun, PosTag::Verb, PosTag::Adj, PosTag::Adv,
            PosTag::Det, PosTag::Prep, PosTag::Pron,
            PosTag::Conj, PosTag::Punct, PosTag::Num,
        ];
        for i in 1..tokens.len() {
            let prev_tag = match tokens[i - 1].pos_tag {
                Some(t) => t,
                None => continue,
            };
            let cur_tag = match tokens[i].pos_tag {
                Some(t) => t,
                None => continue,
            };
            let cur_prob = self.bigram_prob(prev_tag, cur_tag);
            // Only override if current probability is very low
            if cur_prob >= 0.05 {
                continue;
            }
            let lower = tokens[i].text.to_lowercase();
            let mut best_tag = cur_tag;
            let mut best_prob = cur_prob;
            for &cand in &candidates {
                // The candidate must also be plausible from the lexicon/suffix
                let lex_tag = self.lexicon.get(&lower).copied();
                let suffix_tag = self.tag_by_suffix(&lower);
                if cand != cur_tag
                    && (lex_tag == Some(cand) || suffix_tag == cand)
                {
                    let p = self.bigram_prob(prev_tag, cand);
                    if p > best_prob {
                        best_prob = p;
                        best_tag = cand;
                    }
                }
            }
            if best_tag != cur_tag {
                tokens[i].pos_tag = Some(best_tag);
            }
        }
    }

    fn bigram_prob(&self, prev: PosTag, cur: PosTag) -> f64 {
        *self.bigram_probs.get(&(prev, cur)).unwrap_or(&0.01)
    }
}

// ── Lexicon builder (200+ entries) ──────────────────────────────────────────

fn build_lexicon() -> HashMap<String, PosTag> {
    let mut m = HashMap::new();
    // Determiners
    for w in &["the", "a", "an", "this", "that", "these", "those", "every",
               "each", "some", "any", "no", "all", "both", "either", "neither",
               "my", "your", "his", "her", "its", "our", "their"] {
        m.insert(w.to_string(), PosTag::Det);
    }
    // Prepositions
    for w in &["in", "on", "at", "to", "for", "with", "by", "from", "of",
               "about", "into", "through", "during", "before", "after", "above",
               "below", "between", "under", "over", "upon", "within", "without",
               "along", "across", "behind", "beyond", "near", "among", "around",
               "against", "toward", "towards"] {
        m.insert(w.to_string(), PosTag::Prep);
    }
    // Conjunctions
    for w in &["and", "but", "or", "nor", "yet", "so", "for", "because",
               "although", "while", "if", "when", "since", "unless", "until",
               "whereas", "whether", "though"] {
        m.insert(w.to_string(), PosTag::Conj);
    }
    // Override "for" – more often preposition than conjunction
    m.insert("for".to_string(), PosTag::Prep);
    // Pronouns
    for w in &["i", "me", "we", "us", "you", "he", "him", "she", "her",
               "it", "they", "them", "myself", "yourself", "himself", "herself",
               "itself", "ourselves", "themselves", "who", "whom", "whose",
               "which", "what", "whoever", "whatever", "everyone", "someone",
               "anyone", "nobody", "everybody", "something", "anything",
               "nothing", "everything", "one", "ones"] {
        m.insert(w.to_string(), PosTag::Pron);
    }
    // Common verbs
    for w in &["is", "am", "are", "was", "were", "be", "been", "being",
               "have", "has", "had", "do", "does", "did", "will", "would",
               "shall", "should", "may", "might", "can", "could", "must",
               "go", "goes", "went", "gone", "going", "come", "came",
               "say", "said", "says", "get", "gets", "got", "getting",
               "make", "made", "makes", "making", "know", "knew", "known",
               "think", "thought", "take", "took", "taken", "see", "saw",
               "seen", "want", "wanted", "give", "gave", "given", "use",
               "used", "find", "found", "tell", "told", "ask", "asked",
               "work", "worked", "seem", "seemed", "feel", "felt", "try",
               "tried", "leave", "left", "call", "called", "need", "needed",
               "keep", "kept", "let", "begin", "began", "begun", "show",
               "showed", "shown", "hear", "heard", "play", "played", "run",
               "ran", "move", "moved", "live", "lived", "believe", "believed",
               "bring", "brought", "happen", "happened", "write", "wrote",
               "written", "provide", "provided", "sit", "sat", "stand",
               "stood", "lose", "lost", "pay", "paid", "meet", "met",
               "include", "included", "continue", "continued", "set",
               "learn", "learned", "change", "changed", "lead", "led",
               "understand", "understood", "watch", "watched", "follow",
               "followed", "stop", "stopped", "create", "created",
               "speak", "spoke", "spoken", "read", "allow", "allowed",
               "add", "added", "grow", "grew", "grown", "open", "opened",
               "walk", "walked", "win", "won", "offer", "offered",
               "remember", "remembered", "love", "loved", "consider",
               "considered", "appear", "appeared", "buy", "bought",
               "wait", "waited", "serve", "served", "die", "died",
               "send", "sent", "expect", "expected", "build", "built",
               "stay", "stayed", "fall", "fell", "fallen", "cut",
               "reach", "reached", "kill", "killed", "remain", "remained"] {
        m.insert(w.to_string(), PosTag::Verb);
    }
    // Common nouns
    for w in &["time", "year", "people", "way", "day", "man", "woman",
               "child", "world", "life", "hand", "part", "place", "case",
               "week", "company", "system", "program", "question", "work",
               "government", "number", "night", "point", "home", "water",
               "room", "mother", "area", "money", "story", "fact", "month",
               "lot", "right", "study", "book", "eye", "job", "word",
               "business", "issue", "side", "kind", "head", "house",
               "service", "friend", "father", "power", "hour", "game",
               "line", "end", "members", "city", "community", "name",
               "president", "team", "minute", "idea", "body", "information",
               "back", "parent", "face", "others", "level", "office",
               "door", "health", "person", "art", "war", "history",
               "party", "result", "morning", "reason", "research",
               "girl", "guy", "moment", "air", "teacher", "force",
               "education", "dog", "cat", "car", "food", "music",
               "language", "problem", "school", "country", "student",
               "group", "family", "court", "market", "data", "plan",
               "table", "form", "class", "report", "model", "process"] {
        m.insert(w.to_string(), PosTag::Noun);
    }
    // Common adjectives
    for w in &["good", "new", "first", "last", "long", "great", "little",
               "own", "other", "old", "right", "big", "high", "different",
               "small", "large", "next", "early", "young", "important",
               "few", "public", "bad", "same", "able", "free", "full",
               "sure", "real", "best", "better", "true", "whole", "clear",
               "happy", "sad", "beautiful", "nice", "simple", "hard",
               "fast", "slow", "easy", "difficult", "strong", "weak",
               "hot", "cold", "warm", "cool", "rich", "poor", "dark",
               "light", "deep", "wide", "short", "tall", "thick", "thin",
               "red", "blue", "green", "white", "black", "brown", "yellow"] {
        m.insert(w.to_string(), PosTag::Adj);
    }
    // Common adverbs
    for w in &["not", "also", "very", "often", "however", "too", "usually",
               "really", "already", "always", "never", "sometimes", "together",
               "likely", "simply", "generally", "instead", "actually",
               "just", "now", "then", "here", "there", "still", "well",
               "quite", "perhaps", "ever", "soon", "almost", "enough",
               "far", "only", "even", "again", "once", "away", "today",
               "finally", "certainly", "probably", "definitely", "absolutely",
               "completely", "extremely", "rather", "nearly", "merely"] {
        m.insert(w.to_string(), PosTag::Adv);
    }
    // Interjections
    for w in &["oh", "wow", "hey", "hello", "hi", "yes", "no", "yeah",
               "okay", "ok", "please", "thanks", "sorry", "well"] {
        m.insert(w.to_string(), PosTag::Intj);
    }
    // Override "well" – adverb more common than interjection
    m.insert("well".to_string(), PosTag::Adv);
    m
}

// ── Suffix rules ────────────────────────────────────────────────────────────

fn build_suffix_rules() -> Vec<SuffixRule> {
    vec![
        SuffixRule { suffix: "ing".into(), pos_tag: PosTag::Verb, priority: 5 },
        SuffixRule { suffix: "tion".into(), pos_tag: PosTag::Noun, priority: 7 },
        SuffixRule { suffix: "sion".into(), pos_tag: PosTag::Noun, priority: 7 },
        SuffixRule { suffix: "ment".into(), pos_tag: PosTag::Noun, priority: 6 },
        SuffixRule { suffix: "ness".into(), pos_tag: PosTag::Noun, priority: 7 },
        SuffixRule { suffix: "ity".into(), pos_tag: PosTag::Noun, priority: 6 },
        SuffixRule { suffix: "ence".into(), pos_tag: PosTag::Noun, priority: 6 },
        SuffixRule { suffix: "ance".into(), pos_tag: PosTag::Noun, priority: 6 },
        SuffixRule { suffix: "able".into(), pos_tag: PosTag::Adj, priority: 6 },
        SuffixRule { suffix: "ible".into(), pos_tag: PosTag::Adj, priority: 6 },
        SuffixRule { suffix: "ful".into(), pos_tag: PosTag::Adj, priority: 6 },
        SuffixRule { suffix: "less".into(), pos_tag: PosTag::Adj, priority: 6 },
        SuffixRule { suffix: "ous".into(), pos_tag: PosTag::Adj, priority: 6 },
        SuffixRule { suffix: "ive".into(), pos_tag: PosTag::Adj, priority: 5 },
        SuffixRule { suffix: "al".into(), pos_tag: PosTag::Adj, priority: 4 },
        SuffixRule { suffix: "ial".into(), pos_tag: PosTag::Adj, priority: 5 },
        SuffixRule { suffix: "ical".into(), pos_tag: PosTag::Adj, priority: 6 },
        SuffixRule { suffix: "ly".into(), pos_tag: PosTag::Adv, priority: 6 },
        SuffixRule { suffix: "ed".into(), pos_tag: PosTag::Verb, priority: 4 },
        SuffixRule { suffix: "en".into(), pos_tag: PosTag::Verb, priority: 3 },
        SuffixRule { suffix: "ize".into(), pos_tag: PosTag::Verb, priority: 6 },
        SuffixRule { suffix: "ise".into(), pos_tag: PosTag::Verb, priority: 6 },
        SuffixRule { suffix: "ate".into(), pos_tag: PosTag::Verb, priority: 5 },
        SuffixRule { suffix: "ify".into(), pos_tag: PosTag::Verb, priority: 6 },
        SuffixRule { suffix: "er".into(), pos_tag: PosTag::Noun, priority: 3 },
        SuffixRule { suffix: "or".into(), pos_tag: PosTag::Noun, priority: 3 },
        SuffixRule { suffix: "ist".into(), pos_tag: PosTag::Noun, priority: 5 },
        SuffixRule { suffix: "ism".into(), pos_tag: PosTag::Noun, priority: 5 },
        SuffixRule { suffix: "ship".into(), pos_tag: PosTag::Noun, priority: 6 },
        SuffixRule { suffix: "dom".into(), pos_tag: PosTag::Noun, priority: 5 },
        SuffixRule { suffix: "hood".into(), pos_tag: PosTag::Noun, priority: 6 },
        SuffixRule { suffix: "ward".into(), pos_tag: PosTag::Adv, priority: 5 },
        SuffixRule { suffix: "wards".into(), pos_tag: PosTag::Adv, priority: 5 },
        SuffixRule { suffix: "wise".into(), pos_tag: PosTag::Adv, priority: 5 },
    ]
}

// ── Contextual rules ────────────────────────────────────────────────────────

fn build_contextual_rules() -> Vec<ContextualRule> {
    vec![
        // DT + ? → Noun or Adjective (handled: DT + unknown → Noun)
        ContextualRule {
            prev_tag: Some(PosTag::Det),
            current_lower: None,
            next_lower: None,
            assigned_tag: PosTag::Noun,
            priority: 2,
        },
        // Preposition + ? → Noun
        ContextualRule {
            prev_tag: Some(PosTag::Prep),
            current_lower: None,
            next_lower: None,
            assigned_tag: PosTag::Noun,
            priority: 1,
        },
        // "to" before a verb-like word → keep verb
        ContextualRule {
            prev_tag: Some(PosTag::Prep),
            current_lower: None,
            next_lower: None,
            assigned_tag: PosTag::Noun,
            priority: 1,
        },
        // Adjective + Adjective can happen (compound), but typically next is Noun
        // "very" + ? → Adjective
        ContextualRule {
            prev_tag: Some(PosTag::Adv),
            current_lower: None,
            next_lower: None,
            assigned_tag: PosTag::Adj,
            priority: 1,
        },
    ]
}

// ── Bigram probabilities ────────────────────────────────────────────────────

fn build_bigram_probs() -> HashMap<(PosTag, PosTag), f64> {
    let mut m = HashMap::new();
    // Common transitions
    let entries: Vec<(PosTag, PosTag, f64)> = vec![
        (PosTag::Det, PosTag::Noun, 0.50),
        (PosTag::Det, PosTag::Adj, 0.35),
        (PosTag::Det, PosTag::Adv, 0.05),
        (PosTag::Adj, PosTag::Noun, 0.60),
        (PosTag::Adj, PosTag::Adj, 0.10),
        (PosTag::Adj, PosTag::Conj, 0.05),
        (PosTag::Noun, PosTag::Verb, 0.35),
        (PosTag::Noun, PosTag::Prep, 0.20),
        (PosTag::Noun, PosTag::Conj, 0.10),
        (PosTag::Noun, PosTag::Noun, 0.10),
        (PosTag::Noun, PosTag::Punct, 0.10),
        (PosTag::Verb, PosTag::Det, 0.20),
        (PosTag::Verb, PosTag::Noun, 0.15),
        (PosTag::Verb, PosTag::Adv, 0.15),
        (PosTag::Verb, PosTag::Adj, 0.10),
        (PosTag::Verb, PosTag::Prep, 0.15),
        (PosTag::Verb, PosTag::Pron, 0.10),
        (PosTag::Verb, PosTag::Verb, 0.05),
        (PosTag::Prep, PosTag::Det, 0.30),
        (PosTag::Prep, PosTag::Noun, 0.35),
        (PosTag::Prep, PosTag::Pron, 0.10),
        (PosTag::Prep, PosTag::Adj, 0.10),
        (PosTag::Pron, PosTag::Verb, 0.50),
        (PosTag::Pron, PosTag::Adv, 0.10),
        (PosTag::Pron, PosTag::Noun, 0.05),
        (PosTag::Adv, PosTag::Verb, 0.30),
        (PosTag::Adv, PosTag::Adj, 0.25),
        (PosTag::Adv, PosTag::Adv, 0.10),
        (PosTag::Conj, PosTag::Det, 0.20),
        (PosTag::Conj, PosTag::Noun, 0.20),
        (PosTag::Conj, PosTag::Pron, 0.15),
        (PosTag::Conj, PosTag::Verb, 0.10),
        (PosTag::Conj, PosTag::Adj, 0.10),
        (PosTag::Punct, PosTag::Det, 0.15),
        (PosTag::Punct, PosTag::Noun, 0.15),
        (PosTag::Punct, PosTag::Pron, 0.10),
        (PosTag::Punct, PosTag::Conj, 0.10),
        (PosTag::Num, PosTag::Noun, 0.30),
        (PosTag::Num, PosTag::Punct, 0.15),
    ];
    for (a, b, p) in entries {
        m.insert((a, b), p);
    }
    m
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
    fn test_tag_simple_sentence() {
        let tagger = RuleBasedTagger::new();
        let mut tokens = make_tokens(&["The", "cat", "sat"]);
        tagger.tag(&mut tokens);
        assert_eq!(tokens[0].pos_tag, Some(PosTag::Det));
        assert_eq!(tokens[1].pos_tag, Some(PosTag::Noun));
        assert_eq!(tokens[2].pos_tag, Some(PosTag::Verb));
    }

    #[test]
    fn test_tag_adjective_noun() {
        let tagger = RuleBasedTagger::new();
        let mut tokens = make_tokens(&["The", "big", "dog"]);
        tagger.tag(&mut tokens);
        assert_eq!(tokens[1].pos_tag, Some(PosTag::Adj));
        assert_eq!(tokens[2].pos_tag, Some(PosTag::Noun));
    }

    #[test]
    fn test_suffix_ing() {
        let tagger = RuleBasedTagger::new();
        assert_eq!(tagger.tag_word("running"), PosTag::Verb);
    }

    #[test]
    fn test_suffix_tion() {
        let tagger = RuleBasedTagger::new();
        assert_eq!(tagger.tag_word("education"), PosTag::Noun);
    }

    #[test]
    fn test_suffix_ly() {
        let tagger = RuleBasedTagger::new();
        assert_eq!(tagger.tag_word("quickly"), PosTag::Adv);
    }

    #[test]
    fn test_suffix_able() {
        let tagger = RuleBasedTagger::new();
        assert_eq!(tagger.tag_word("reachable"), PosTag::Adj);
    }

    #[test]
    fn test_suffix_ness() {
        let tagger = RuleBasedTagger::new();
        assert_eq!(tagger.tag_word("happiness"), PosTag::Noun);
    }

    #[test]
    fn test_pronoun_verb_sequence() {
        let tagger = RuleBasedTagger::new();
        let mut tokens = make_tokens(&["I", "think"]);
        tagger.tag(&mut tokens);
        assert_eq!(tokens[0].pos_tag, Some(PosTag::Pron));
        assert_eq!(tokens[1].pos_tag, Some(PosTag::Verb));
    }

    #[test]
    fn test_preposition_noun() {
        let tagger = RuleBasedTagger::new();
        let mut tokens = make_tokens(&["in", "the", "house"]);
        tagger.tag(&mut tokens);
        assert_eq!(tokens[0].pos_tag, Some(PosTag::Prep));
    }

    #[test]
    fn test_number_detection() {
        let tagger = RuleBasedTagger::new();
        assert_eq!(tagger.tag_word("42"), PosTag::Num);
    }

    #[test]
    fn test_unknown_word_gets_tagged() {
        let tagger = RuleBasedTagger::new();
        let tag = tagger.tag_word("xyzzy");
        // Should fall back to Noun (default)
        assert_eq!(tag, PosTag::Noun);
    }

    #[test]
    fn test_contextual_det_adj_noun() {
        let tagger = RuleBasedTagger::new();
        let mut tokens = make_tokens(&["The", "beautiful", "garden"]);
        tagger.tag(&mut tokens);
        assert_eq!(tokens[0].pos_tag, Some(PosTag::Det));
        assert_eq!(tokens[1].pos_tag, Some(PosTag::Adj));
        assert_eq!(tokens[2].pos_tag, Some(PosTag::Noun));
    }
}
