//! Comprehensive rule-based tokenizer for English text.

use shared_types::{LocalizerError, Token};
use serde::{Deserialize, Serialize};

// ── Tokenizer rule types ────────────────────────────────────────────────────

/// What to do when a rule fires.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenAction {
    Split,
    Merge,
    Keep,
}

/// A named tokenization rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerRule {
    pub name: String,
    pub pattern: String,
    pub action: TokenAction,
}

/// Character-level span tracking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharacterSpan {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

// ── SimpleTokenizer ─────────────────────────────────────────────────────────

/// A rule-based tokenizer that handles whitespace, punctuation, contractions,
/// possessives, numbers, URLs, and e-mail addresses.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    rules: Vec<TokenizerRule>,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let rules = vec![
            TokenizerRule {
                name: "whitespace_split".into(),
                pattern: r"\s+".into(),
                action: TokenAction::Split,
            },
            TokenizerRule {
                name: "contraction_split".into(),
                pattern: "n't|'re|'ve|'ll|'d|'s|'m".into(),
                action: TokenAction::Split,
            },
            TokenizerRule {
                name: "punctuation_split".into(),
                pattern: r#"[.,!?;:\-"()\[\]{}]"#.into(),
                action: TokenAction::Split,
            },
        ];
        Self { rules }
    }

    /// Main entry point – tokenize a string into [`Token`] values with spans.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let spans = self.split_to_spans(text);
        spans
            .into_iter()
            .enumerate()
            .map(|(i, sp)| Token::new(&sp.text, i))
            .collect()
    }

    /// Tokenize and also return the raw character spans.
    pub fn tokenize_with_spans(&self, text: &str) -> (Vec<Token>, Vec<CharacterSpan>) {
        let spans = self.split_to_spans(text);
        let tokens: Vec<Token> = spans
            .iter()
            .enumerate()
            .map(|(i, sp)| Token::new(&sp.text, i))
            .collect();
        (tokens, spans)
    }

    /// Convert a token sequence back to text.
    pub fn detokenize(&self, tokens: &[Token]) -> String {
        if tokens.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(tokens.last().unwrap().text.len() * 2);
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 && !is_punct_no_space(&tok.text) && !is_closing_bracket(&tok.text) {
                let prev = &tokens[i - 1].text;
                if !is_opening_bracket(prev) {
                    out.push(' ');
                }
            }
            // Contractions should attach directly
            if tok.text.starts_with('\'') || tok.text == "n't" {
                // Remove trailing space if we just added one
                if out.ends_with(' ') {
                    out.pop();
                }
            }
            out.push_str(&tok.text);
        }
        out
    }

    // ── internal helpers ────────────────────────────────────────────────────

    fn split_to_spans(&self, text: &str) -> Vec<CharacterSpan> {
        // Phase 1: whitespace split into raw chunks
        let raw_chunks = whitespace_split(text);
        // Phase 2: further split each chunk
        let mut spans = Vec::new();
        for chunk in raw_chunks {
            let sub = self.split_chunk(&chunk);
            spans.extend(sub);
        }
        spans
    }

    fn split_chunk(&self, chunk: &CharacterSpan) -> Vec<CharacterSpan> {
        let word = &chunk.text;
        let base = chunk.start;

        // URL / email detection – keep as single token
        if is_url(word) || is_email(word) {
            return vec![chunk.clone()];
        }

        // Number with punctuation (e.g. "3.14", "1,000")
        if is_number(word) {
            return vec![chunk.clone()];
        }

        // Try contraction splitting
        if let Some(parts) = split_contraction(word) {
            let mut out = Vec::new();
            let mut off = base;
            for p in &parts {
                out.push(CharacterSpan {
                    start: off,
                    end: off + p.len(),
                    text: p.clone(),
                });
                off += p.len();
            }
            return out;
        }

        // Possessive splitting: word ending with 's
        if word.ends_with("'s") && word.len() > 2 {
            let stem = &word[..word.len() - 2];
            return vec![
                CharacterSpan { start: base, end: base + stem.len(), text: stem.to_string() },
                CharacterSpan {
                    start: base + stem.len(),
                    end: base + word.len(),
                    text: "'s".to_string(),
                },
            ];
        }

        // Punctuation splitting from edges
        let mut result = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;

        // Leading punctuation
        while i < chars.len() && is_split_punct(chars[i]) {
            let c = chars[i];
            result.push(CharacterSpan {
                start: base + byte_offset(word, i),
                end: base + byte_offset(word, i) + c.len_utf8(),
                text: c.to_string(),
            });
            i += 1;
        }

        // Trailing punctuation
        let mut j = chars.len();
        let mut trailing = Vec::new();
        while j > i && is_split_punct(chars[j - 1]) {
            j -= 1;
            let c = chars[j];
            trailing.push(CharacterSpan {
                start: base + byte_offset(word, j),
                end: base + byte_offset(word, j) + c.len_utf8(),
                text: c.to_string(),
            });
        }

        // Core word
        if i < j {
            let core: String = chars[i..j].iter().collect();
            let s = base + byte_offset(word, i);
            result.push(CharacterSpan {
                start: s,
                end: s + core.len(),
                text: core,
            });
        }

        trailing.reverse();
        result.extend(trailing);
        result
    }

    pub fn rules(&self) -> &[TokenizerRule] {
        &self.rules
    }
}

// ── TokenAligner ────────────────────────────────────────────────────────────

/// Aligns tokens between original and transformed text using character offsets
/// and Levenshtein-based matching.
#[derive(Debug, Clone)]
pub struct TokenAligner;

impl TokenAligner {
    /// Align two token sequences, returning pairs (orig_idx, trans_idx).
    /// Uses a simple greedy text-match approach.
    pub fn align(original: &[Token], transformed: &[Token]) -> Vec<(Option<usize>, Option<usize>)> {
        let mut alignment = Vec::new();
        let mut j = 0;
        for (i, orig_tok) in original.iter().enumerate() {
            if j < transformed.len() {
                if orig_tok.text.to_lowercase() == transformed[j].text.to_lowercase() {
                    alignment.push((Some(i), Some(j)));
                    j += 1;
                } else {
                    // Try skipping in transformed to find a match
                    let mut found = false;
                    for k in j..transformed.len().min(j + 3) {
                        if orig_tok.text.to_lowercase() == transformed[k].text.to_lowercase() {
                            // Emit unmatched transformed tokens first
                            for m in j..k {
                                alignment.push((None, Some(m)));
                            }
                            alignment.push((Some(i), Some(k)));
                            j = k + 1;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        alignment.push((Some(i), None));
                    }
                }
            } else {
                alignment.push((Some(i), None));
            }
        }
        // Remaining transformed tokens
        while j < transformed.len() {
            alignment.push((None, Some(j)));
            j += 1;
        }
        alignment
    }

    /// Compute Levenshtein distance between two strings.
    pub fn levenshtein(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let n = a_chars.len();
        let m = b_chars.len();
        let mut dp = vec![vec![0usize; m + 1]; n + 1];
        for i in 0..=n {
            dp[i][0] = i;
        }
        for j in 0..=m {
            dp[0][j] = j;
        }
        for i in 1..=n {
            for j in 1..=m {
                let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[n][m]
    }
}

// ── Free helpers ────────────────────────────────────────────────────────────

fn whitespace_split(text: &str) -> Vec<CharacterSpan> {
    let mut spans = Vec::new();
    let mut start = None;
    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start {
                spans.push(CharacterSpan {
                    start: s,
                    end: i,
                    text: text[s..i].to_string(),
                });
                start = None;
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push(CharacterSpan {
            start: s,
            end: text.len(),
            text: text[s..].to_string(),
        });
    }
    spans
}

fn split_contraction(word: &str) -> Option<Vec<String>> {
    let lower = word.to_lowercase();
    let contractions = [
        ("don't", vec!["do", "n't"]),
        ("doesn't", vec!["does", "n't"]),
        ("didn't", vec!["did", "n't"]),
        ("won't", vec!["wo", "n't"]),
        ("wouldn't", vec!["would", "n't"]),
        ("couldn't", vec!["could", "n't"]),
        ("shouldn't", vec!["should", "n't"]),
        ("isn't", vec!["is", "n't"]),
        ("aren't", vec!["are", "n't"]),
        ("wasn't", vec!["was", "n't"]),
        ("weren't", vec!["were", "n't"]),
        ("hasn't", vec!["has", "n't"]),
        ("haven't", vec!["have", "n't"]),
        ("hadn't", vec!["had", "n't"]),
        ("can't", vec!["ca", "n't"]),
        ("i'm", vec!["I", "'m"]),
        ("i've", vec!["I", "'ve"]),
        ("i'll", vec!["I", "'ll"]),
        ("i'd", vec!["I", "'d"]),
        ("he's", vec!["he", "'s"]),
        ("she's", vec!["she", "'s"]),
        ("it's", vec!["it", "'s"]),
        ("we're", vec!["we", "'re"]),
        ("they're", vec!["they", "'re"]),
        ("you're", vec!["you", "'re"]),
        ("we've", vec!["we", "'ve"]),
        ("they've", vec!["they", "'ve"]),
        ("you've", vec!["you", "'ve"]),
        ("we'll", vec!["we", "'ll"]),
        ("they'll", vec!["they", "'ll"]),
        ("you'll", vec!["you", "'ll"]),
        ("he'd", vec!["he", "'d"]),
        ("she'd", vec!["she", "'d"]),
        ("we'd", vec!["we", "'d"]),
        ("they'd", vec!["they", "'d"]),
        ("you'd", vec!["you", "'d"]),
    ];
    for (pat, parts) in &contractions {
        if lower == *pat {
            return Some(parts.iter().map(|s| s.to_string()).collect());
        }
    }
    // Generic n't splitting
    if lower.ends_with("n't") && lower.len() > 3 {
        let stem_end = word.len() - 3;
        return Some(vec![word[..stem_end].to_string(), "n't".to_string()]);
    }
    // Generic 're, 've, 'll, 'd, 'm
    for suffix in &["'re", "'ve", "'ll", "'d", "'m", "'s"] {
        if lower.ends_with(suffix) && word.len() > suffix.len() {
            let stem_end = word.len() - suffix.len();
            return Some(vec![word[..stem_end].to_string(), word[stem_end..].to_string()]);
        }
    }
    None
}

fn is_url(word: &str) -> bool {
    word.starts_with("http://")
        || word.starts_with("https://")
        || word.starts_with("www.")
        || word.starts_with("ftp://")
}

fn is_email(word: &str) -> bool {
    let at = word.find('@');
    let dot = word.rfind('.');
    matches!((at, dot), (Some(a), Some(d)) if a > 0 && d > a + 1 && d < word.len() - 1)
}

fn is_number(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let mut has_digit = false;
    for ch in word.chars() {
        if ch.is_ascii_digit() {
            has_digit = true;
        } else if ch != '.' && ch != ',' && ch != '-' && ch != '+' {
            return false;
        }
    }
    has_digit
}

fn is_split_punct(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}'
            | '-' | '–' | '—'
    )
}

fn is_punct_no_space(s: &str) -> bool {
    matches!(s, "." | "," | "!" | "?" | ";" | ":" | ")" | "]" | "}" | "'" | "\"")
}

fn is_opening_bracket(s: &str) -> bool {
    matches!(s, "(" | "[" | "{")
}

fn is_closing_bracket(s: &str) -> bool {
    matches!(s, ")" | "]" | "}")
}

fn byte_offset(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> SimpleTokenizer {
        SimpleTokenizer::new()
    }

    fn texts(tokens: &[Token]) -> Vec<&str> {
        tokens.iter().map(|t| t.text.as_str()).collect()
    }

    #[test]
    fn test_simple_whitespace() {
        let toks = tok().tokenize("Hello world");
        assert_eq!(texts(&toks), vec!["Hello", "world"]);
    }

    #[test]
    fn test_punctuation_split() {
        let toks = tok().tokenize("Hello, world!");
        assert_eq!(texts(&toks), vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn test_contraction_dont() {
        let toks = tok().tokenize("I don't know");
        assert_eq!(texts(&toks), vec!["I", "do", "n't", "know"]);
    }

    #[test]
    fn test_contraction_im() {
        let toks = tok().tokenize("I'm happy");
        assert!(texts(&toks).contains(&"I"));
        assert!(texts(&toks).contains(&"'m"));
    }

    #[test]
    fn test_possessive() {
        let toks = tok().tokenize("John's book");
        assert!(texts(&toks).contains(&"John"));
        assert!(texts(&toks).contains(&"'s"));
    }

    #[test]
    fn test_url_kept_whole() {
        let toks = tok().tokenize("Visit https://example.com today");
        assert!(texts(&toks).contains(&"https://example.com"));
    }

    #[test]
    fn test_email_kept_whole() {
        let toks = tok().tokenize("Mail user@example.com please");
        assert!(texts(&toks).contains(&"user@example.com"));
    }

    #[test]
    fn test_number_kept_whole() {
        let toks = tok().tokenize("The price is 3.14");
        assert!(texts(&toks).contains(&"3.14"));
    }

    #[test]
    fn test_character_offsets() {
        let text = "Hello world";
        let toks = tok().tokenize(text);
        assert_eq!(toks[0].start, 0);
        assert_eq!(toks[0].end, 5);
        assert_eq!(toks[1].start, 6);
        assert_eq!(toks[1].end, 11);
    }

    #[test]
    fn test_detokenize_roundtrip() {
        let text = "Hello world";
        let toks = tok().tokenize(text);
        let recovered = tok().detokenize(&toks);
        assert_eq!(recovered, text);
    }

    #[test]
    fn test_token_aligner_identical() {
        let t = tok();
        let a = t.tokenize("The cat sat");
        let b = t.tokenize("The cat sat");
        let alignment = TokenAligner::align(&a, &b);
        for (orig, trans) in &alignment {
            assert!(orig.is_some() && trans.is_some());
        }
    }

    #[test]
    fn test_token_aligner_insertion() {
        let t = tok();
        let a = t.tokenize("The cat");
        let b = t.tokenize("The big cat");
        let alignment = TokenAligner::align(&a, &b);
        assert!(alignment.len() >= 3);
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(TokenAligner::levenshtein("kitten", "sitting"), 3);
        assert_eq!(TokenAligner::levenshtein("", "abc"), 3);
        assert_eq!(TokenAligner::levenshtein("same", "same"), 0);
    }

    #[test]
    fn test_empty_input() {
        let toks = tok().tokenize("");
        assert!(toks.is_empty());
    }

    #[test]
    fn test_multiple_spaces() {
        let toks = tok().tokenize("Hello   world");
        assert_eq!(texts(&toks), vec!["Hello", "world"]);
    }
}
