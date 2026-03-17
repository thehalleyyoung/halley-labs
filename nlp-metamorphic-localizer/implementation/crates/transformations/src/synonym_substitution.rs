//! Synonym substitution transformation: replaces words with synonyms while
//! maintaining morphological form.

use crate::base::{
    BaseTransformation, TransformationError, TransformationKind, TransformationResult,
    rebuild_sentence, reindex_tokens,
};
use shared_types::{PosTag, Sentence, Token};
use std::collections::HashMap;

/// Build the built-in synonym table (≥100 pairs).
fn build_synonym_table() -> HashMap<&'static str, Vec<(&'static str, Option<PosTag>)>> {
    let mut m: HashMap<&str, Vec<(&str, Option<PosTag>)>> = HashMap::new();

    let pairs: &[(&str, &str, Option<PosTag>)] = &[
        // Adjectives
        ("big", "large", Some(PosTag::Adj)),
        ("large", "big", Some(PosTag::Adj)),
        ("small", "tiny", Some(PosTag::Adj)),
        ("tiny", "small", Some(PosTag::Adj)),
        ("fast", "quick", Some(PosTag::Adj)),
        ("quick", "fast", Some(PosTag::Adj)),
        ("happy", "glad", Some(PosTag::Adj)),
        ("glad", "happy", Some(PosTag::Adj)),
        ("sad", "unhappy", Some(PosTag::Adj)),
        ("unhappy", "sad", Some(PosTag::Adj)),
        ("angry", "furious", Some(PosTag::Adj)),
        ("furious", "angry", Some(PosTag::Adj)),
        ("beautiful", "gorgeous", Some(PosTag::Adj)),
        ("gorgeous", "beautiful", Some(PosTag::Adj)),
        ("ugly", "hideous", Some(PosTag::Adj)),
        ("hideous", "ugly", Some(PosTag::Adj)),
        ("smart", "intelligent", Some(PosTag::Adj)),
        ("intelligent", "smart", Some(PosTag::Adj)),
        ("brave", "courageous", Some(PosTag::Adj)),
        ("courageous", "brave", Some(PosTag::Adj)),
        ("easy", "simple", Some(PosTag::Adj)),
        ("simple", "easy", Some(PosTag::Adj)),
        ("hard", "difficult", Some(PosTag::Adj)),
        ("difficult", "hard", Some(PosTag::Adj)),
        ("old", "ancient", Some(PosTag::Adj)),
        ("ancient", "old", Some(PosTag::Adj)),
        ("new", "novel", Some(PosTag::Adj)),
        ("rich", "wealthy", Some(PosTag::Adj)),
        ("wealthy", "rich", Some(PosTag::Adj)),
        ("poor", "impoverished", Some(PosTag::Adj)),
        ("strong", "powerful", Some(PosTag::Adj)),
        ("powerful", "strong", Some(PosTag::Adj)),
        ("weak", "feeble", Some(PosTag::Adj)),
        ("feeble", "weak", Some(PosTag::Adj)),
        ("hot", "scorching", Some(PosTag::Adj)),
        ("cold", "frigid", Some(PosTag::Adj)),
        ("clean", "spotless", Some(PosTag::Adj)),
        ("dirty", "filthy", Some(PosTag::Adj)),
        ("tall", "towering", Some(PosTag::Adj)),
        ("short", "brief", Some(PosTag::Adj)),
        ("calm", "peaceful", Some(PosTag::Adj)),
        ("peaceful", "calm", Some(PosTag::Adj)),
        ("loud", "noisy", Some(PosTag::Adj)),
        ("noisy", "loud", Some(PosTag::Adj)),
        ("quiet", "silent", Some(PosTag::Adj)),
        ("silent", "quiet", Some(PosTag::Adj)),
        ("bright", "luminous", Some(PosTag::Adj)),
        ("dark", "dim", Some(PosTag::Adj)),
        ("strange", "peculiar", Some(PosTag::Adj)),
        ("peculiar", "strange", Some(PosTag::Adj)),
        // Verbs
        ("run", "sprint", Some(PosTag::Verb)),
        ("sprint", "run", Some(PosTag::Verb)),
        ("walk", "stroll", Some(PosTag::Verb)),
        ("stroll", "walk", Some(PosTag::Verb)),
        ("eat", "consume", Some(PosTag::Verb)),
        ("consume", "eat", Some(PosTag::Verb)),
        ("drink", "sip", Some(PosTag::Verb)),
        ("say", "state", Some(PosTag::Verb)),
        ("state", "say", Some(PosTag::Verb)),
        ("tell", "inform", Some(PosTag::Verb)),
        ("inform", "tell", Some(PosTag::Verb)),
        ("ask", "inquire", Some(PosTag::Verb)),
        ("inquire", "ask", Some(PosTag::Verb)),
        ("see", "observe", Some(PosTag::Verb)),
        ("observe", "see", Some(PosTag::Verb)),
        ("look", "gaze", Some(PosTag::Verb)),
        ("gaze", "look", Some(PosTag::Verb)),
        ("hear", "listen", Some(PosTag::Verb)),
        ("make", "create", Some(PosTag::Verb)),
        ("create", "make", Some(PosTag::Verb)),
        ("build", "construct", Some(PosTag::Verb)),
        ("construct", "build", Some(PosTag::Verb)),
        ("break", "shatter", Some(PosTag::Verb)),
        ("shatter", "break", Some(PosTag::Verb)),
        ("fix", "repair", Some(PosTag::Verb)),
        ("repair", "fix", Some(PosTag::Verb)),
        ("help", "assist", Some(PosTag::Verb)),
        ("assist", "help", Some(PosTag::Verb)),
        ("start", "begin", Some(PosTag::Verb)),
        ("begin", "start", Some(PosTag::Verb)),
        ("finish", "complete", Some(PosTag::Verb)),
        ("complete", "finish", Some(PosTag::Verb)),
        ("buy", "purchase", Some(PosTag::Verb)),
        ("purchase", "buy", Some(PosTag::Verb)),
        ("sell", "vend", Some(PosTag::Verb)),
        ("give", "provide", Some(PosTag::Verb)),
        ("provide", "give", Some(PosTag::Verb)),
        ("take", "grab", Some(PosTag::Verb)),
        ("grab", "take", Some(PosTag::Verb)),
        ("throw", "hurl", Some(PosTag::Verb)),
        ("hurl", "throw", Some(PosTag::Verb)),
        ("hit", "strike", Some(PosTag::Verb)),
        ("strike", "hit", Some(PosTag::Verb)),
        ("close", "shut", Some(PosTag::Verb)),
        ("shut", "close", Some(PosTag::Verb)),
        ("open", "unlock", Some(PosTag::Verb)),
        // Nouns
        ("house", "home", Some(PosTag::Noun)),
        ("home", "house", Some(PosTag::Noun)),
        ("car", "automobile", Some(PosTag::Noun)),
        ("automobile", "car", Some(PosTag::Noun)),
        ("child", "kid", Some(PosTag::Noun)),
        ("kid", "child", Some(PosTag::Noun)),
        ("man", "gentleman", Some(PosTag::Noun)),
        ("woman", "lady", Some(PosTag::Noun)),
        ("lady", "woman", Some(PosTag::Noun)),
        ("dog", "hound", Some(PosTag::Noun)),
        ("hound", "dog", Some(PosTag::Noun)),
        ("cat", "feline", Some(PosTag::Noun)),
        ("feline", "cat", Some(PosTag::Noun)),
        ("road", "path", Some(PosTag::Noun)),
        ("path", "road", Some(PosTag::Noun)),
        ("forest", "woods", Some(PosTag::Noun)),
        ("woods", "forest", Some(PosTag::Noun)),
        ("ocean", "sea", Some(PosTag::Noun)),
        ("sea", "ocean", Some(PosTag::Noun)),
        ("stone", "rock", Some(PosTag::Noun)),
        ("rock", "stone", Some(PosTag::Noun)),
        // Adverbs
        ("quickly", "rapidly", Some(PosTag::Adv)),
        ("rapidly", "quickly", Some(PosTag::Adv)),
        ("slowly", "gradually", Some(PosTag::Adv)),
        ("gradually", "slowly", Some(PosTag::Adv)),
        ("often", "frequently", Some(PosTag::Adv)),
        ("frequently", "often", Some(PosTag::Adv)),
        ("rarely", "seldom", Some(PosTag::Adv)),
        ("seldom", "rarely", Some(PosTag::Adv)),
        ("always", "constantly", Some(PosTag::Adv)),
        ("constantly", "always", Some(PosTag::Adv)),
    ];

    for &(word, syn, pos) in pairs {
        m.entry(word).or_default().push((syn, pos));
    }
    m
}

/// Attempt to adapt a synonym to the morphological form of the original.
fn adapt_morphology(original: &str, synonym_base: &str) -> String {
    let low = original.to_lowercase();

    // Past tense -ed
    if low.ends_with("ed") {
        if synonym_base.ends_with('e') {
            return format!("{}d", synonym_base);
        }
        return format!("{}ed", synonym_base);
    }
    // Progressive -ing
    if low.ends_with("ing") {
        if synonym_base.ends_with('e') {
            return format!("{}ing", &synonym_base[..synonym_base.len() - 1]);
        }
        return format!("{}ing", synonym_base);
    }
    // 3rd person -s
    if low.ends_with('s') && !low.ends_with("ss") {
        if synonym_base.ends_with("sh") || synonym_base.ends_with("ch")
            || synonym_base.ends_with('x') || synonym_base.ends_with('z')
            || synonym_base.ends_with('s')
        {
            return format!("{}es", synonym_base);
        }
        if synonym_base.ends_with('y')
            && synonym_base.len() > 2
            && !matches!(
                synonym_base.as_bytes()[synonym_base.len() - 2],
                b'a' | b'e' | b'i' | b'o' | b'u'
            )
        {
            return format!("{}ies", &synonym_base[..synonym_base.len() - 1]);
        }
        return format!("{}s", synonym_base);
    }
    // Comparative -er
    if low.ends_with("er") && !low.ends_with("ler") {
        return format!("{}er", synonym_base);
    }
    // Superlative -est
    if low.ends_with("est") {
        return format!("{}est", synonym_base);
    }
    // Plural nouns
    if low.ends_with('s') {
        return format!("{}s", synonym_base);
    }

    synonym_base.to_string()
}

/// Preserve capitalization pattern from original.
fn match_capitalization(original: &str, replacement: &str) -> String {
    if original.chars().next().map_or(false, |c| c.is_uppercase()) {
        let mut chars = replacement.chars();
        match chars.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
        }
    } else {
        replacement.to_string()
    }
}

pub struct SynonymSubstitutionTransform {
    synonym_table: HashMap<String, Vec<(String, Option<PosTag>)>>,
}

impl SynonymSubstitutionTransform {
    pub fn new() -> Self {
        let raw = build_synonym_table();
        let table = raw
            .into_iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    v.into_iter().map(|(s, p)| (s.to_string(), p)).collect(),
                )
            })
            .collect();
        Self { synonym_table: table }
    }

    /// Find the first substitutable token.
    fn find_substitutable(&self, sentence: &Sentence) -> Option<(usize, String)> {
        for (i, tok) in sentence.tokens.iter().enumerate() {
            let low = tok.text.to_lowercase();
            let lemma = tok.lemma.as_deref().unwrap_or(&low).to_lowercase();
            // Try lemma first, then surface form
            for key in [&lemma, &low] {
                if let Some(synonyms) = self.synonym_table.get(key.as_str()) {
                    // POS-aware: prefer synonym matching the token's POS
                    let best = synonyms
                        .iter()
                        .find(|(_, pos)| pos.is_none() || *pos == tok.pos_tag)
                        .or_else(|| synonyms.first());
                    if let Some((syn, _)) = best {
                        let adapted = adapt_morphology(&tok.text, syn);
                        let final_form = match_capitalization(&tok.text, &adapted);
                        return Some((i, final_form));
                    }
                }
            }
        }
        None
    }

    pub fn inverse(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        // Since the table is bijective for many pairs, applying again may reverse.
        self.apply(sentence)
    }
}

impl BaseTransformation for SynonymSubstitutionTransform {
    fn kind(&self) -> TransformationKind {
        TransformationKind::SynonymSubstitution
    }

    fn is_applicable(&self, sentence: &Sentence) -> bool {
        self.find_substitutable(sentence).is_some()
    }

    fn apply(&self, sentence: &Sentence) -> Result<TransformationResult, TransformationError> {
        let (idx, replacement) = self.find_substitutable(sentence).ok_or(
            TransformationError::NoApplicableSite,
        )?;

        let mut new_tokens: Vec<Token> = sentence.tokens.clone();
        let orig_text = new_tokens[idx].text.clone();
        new_tokens[idx].text = replacement.clone();

        reindex_tokens(&mut new_tokens);
        let result = rebuild_sentence(new_tokens, sentence);
        Ok(TransformationResult::ok(
            sentence.clone(),
            result,
            vec![(idx, idx + 1)],
            format!("Substituted '{}' → '{}'", orig_text, replacement),
        ))
    }

    fn is_meaning_preserving(&self) -> bool {
        true // synonyms preserve meaning (approximately)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};

    fn sentence_with_adj() -> Sentence {
        let tokens = vec![
            Token::new("The", 0).with_pos(PosTag::Det),
            Token::new("big", 1).with_pos(PosTag::Adj),
            Token::new("cat", 2).with_pos(PosTag::Noun),
            Token::new("sat", 3).with_pos(PosTag::Verb),
            Token::new(".", 4).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(3, 3, DependencyRelation::Root),
            DependencyEdge::new(3, 2, DependencyRelation::Nsubj),
            DependencyEdge::new(2, 0, DependencyRelation::Det),
            DependencyEdge::new(2, 1, DependencyRelation::Amod),
            DependencyEdge::new(3, 4, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "The big cat sat.");
        s.dependency_edges = edges;
        s
    }

    fn sentence_with_verb() -> Sentence {
        let tokens = vec![
            Token::new("John", 0).with_pos(PosTag::Noun),
            Token::new("runs", 1).with_pos(PosTag::Verb).with_lemma("run"),
            Token::new(".", 2).with_pos(PosTag::Punct),
        ];
        let edges = vec![
            DependencyEdge::new(1, 1, DependencyRelation::Root),
            DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
            DependencyEdge::new(1, 2, DependencyRelation::Punct),
        ];
        let mut s = Sentence::from_tokens(tokens, "John runs.");
        s.dependency_edges = edges;
        s
    }

    #[test]
    fn test_new() {
        let t = SynonymSubstitutionTransform::new();
        assert_eq!(t.kind(), TransformationKind::SynonymSubstitution);
    }

    #[test]
    fn test_is_applicable() {
        let t = SynonymSubstitutionTransform::new();
        assert!(t.is_applicable(&sentence_with_adj()));
    }

    #[test]
    fn test_apply_adjective() {
        let t = SynonymSubstitutionTransform::new();
        let result = t.apply(&sentence_with_adj()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        assert!(text.contains("large") || text.contains("feline"));
    }

    #[test]
    fn test_apply_verb_morphology() {
        let t = SynonymSubstitutionTransform::new();
        let result = t.apply(&sentence_with_verb()).unwrap();
        assert!(result.success);
        let text = result.transformed.surface_text().to_lowercase();
        // "runs" (lemma "run") → "sprints"
        assert!(text.contains("sprints") || text.contains("sprint"));
    }

    #[test]
    fn test_meaning_preserving() {
        let t = SynonymSubstitutionTransform::new();
        assert!(t.is_meaning_preserving());
    }

    #[test]
    fn test_adapt_morphology() {
        assert_eq!(adapt_morphology("walked", "stroll"), "strolled");
        assert_eq!(adapt_morphology("running", "sprint"), "sprinting");
        assert_eq!(adapt_morphology("bigger", "large"), "larger");
    }

    #[test]
    fn test_match_capitalization() {
        assert_eq!(match_capitalization("Hello", "world"), "World");
        assert_eq!(match_capitalization("hello", "world"), "world");
    }

    #[test]
    fn test_not_applicable_no_synonyms() {
        let t = SynonymSubstitutionTransform::new();
        let tokens = vec![
            Token::new(".", 0).with_pos(PosTag::Punct),
        ];
        let s = Sentence::from_tokens(tokens, ".");
        assert!(!t.is_applicable(&s));
    }
}
