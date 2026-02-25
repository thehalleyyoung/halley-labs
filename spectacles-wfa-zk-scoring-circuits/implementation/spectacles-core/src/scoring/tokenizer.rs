//! Tokenizer implementations for scoring functions.
//!
//! Provides multiple tokenization strategies:
//! - WhitespaceTokenizer: splits on whitespace
//! - WordPieceTokenizer: subword tokenization
//! - CharacterTokenizer: character-level tokenization
//! - NGramTokenizer: n-gram extraction

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A single token with metadata
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub id: u32,
    pub start: usize,
    pub end: usize,
}

impl Token {
    pub fn new(text: String, id: u32, start: usize, end: usize) -> Self {
        Self { text, id, start, end }
    }
}

/// Normalization options for tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub lowercase: bool,
    pub strip_punctuation: bool,
    pub strip_whitespace: bool,
    pub strip_accents: bool,
    pub collapse_whitespace: bool,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            strip_punctuation: false,
            strip_whitespace: false,
            strip_accents: false,
            collapse_whitespace: true,
        }
    }
}

/// Core tokenizer trait
pub trait Tokenizer: Send + Sync {
    /// Tokenize a string into tokens
    fn tokenize(&self, text: &str) -> Vec<Token>;
    
    /// Convert tokens back to a string
    fn detokenize(&self, tokens: &[Token]) -> String;
    
    /// Get the vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Look up a token ID by text
    fn token_to_id(&self, token: &str) -> Option<u32>;
    
    /// Look up token text by ID
    fn id_to_token(&self, id: u32) -> Option<&str>;
    
    /// Normalize text before tokenization
    fn normalize(&self, text: &str) -> String;
}

/// Token vocabulary with bidirectional mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    special_tokens: HashMap<String, u32>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            special_tokens: HashMap::new(),
        }
    }
    
    pub fn with_special_tokens(tokens: &[&str]) -> Self {
        let mut vocab = Self::new();
        for &token in tokens {
            vocab.add_special_token(token);
        }
        vocab
    }
    
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        id
    }
    
    pub fn add_special_token(&mut self, token: &str) -> u32 {
        let id = self.add_token(token);
        self.special_tokens.insert(token.to_string(), id);
        id
    }
    
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
    
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }
    
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }
    
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }
    
    pub fn is_special(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }
    
    pub fn special_token_ids(&self) -> Vec<u32> {
        self.special_tokens.values().copied().collect()
    }
    
    /// Build vocabulary from a corpus of texts
    pub fn build_from_corpus(texts: &[&str], min_freq: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for word in text.split_whitespace() {
                *freq.entry(word.to_string()).or_insert(0) += 1;
            }
        }
        
        let mut vocab = Self::with_special_tokens(&["[PAD]", "[UNK]", "[BOS]", "[EOS]"]);
        
        let mut sorted: Vec<_> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        
        for (token, count) in sorted {
            if count >= min_freq {
                vocab.add_token(&token);
            }
        }
        
        vocab
    }
    
    /// Merge two vocabularies
    pub fn merge(&mut self, other: &Vocabulary) {
        for token in &other.id_to_token {
            self.add_token(token);
        }
        for (token, _) in &other.special_tokens {
            self.special_tokens.entry(token.clone())
                .or_insert_with(|| *self.token_to_id.get(token).unwrap());
        }
    }
}

// ============================================================
// Text normalization utilities
// ============================================================

fn normalize_text(text: &str, config: &NormalizationConfig) -> String {
    let mut result = text.to_string();
    
    if config.lowercase {
        result = result.to_lowercase();
    }
    
    if config.strip_accents {
        result = strip_accents(&result);
    }
    
    if config.strip_punctuation {
        result = result.chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();
    }
    
    if config.collapse_whitespace {
        let mut prev_space = false;
        result = result.chars().filter(|c| {
            if c.is_whitespace() {
                if prev_space { return false; }
                prev_space = true;
            } else {
                prev_space = false;
            }
            true
        }).collect();
    }
    
    if config.strip_whitespace {
        result = result.trim().to_string();
    }
    
    result
}

fn strip_accents(s: &str) -> String {
    // Simple ASCII approximation of accent stripping
    s.chars().map(|c| match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ñ' => 'n',
        'ç' => 'c',
        'ý' | 'ÿ' => 'y',
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'A',
        'È' | 'É' | 'Ê' | 'Ë' => 'E',
        'Ì' | 'Í' | 'Î' | 'Ï' => 'I',
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'O',
        'Ù' | 'Ú' | 'Û' | 'Ü' => 'U',
        'Ñ' => 'N',
        'Ç' => 'C',
        'Ý' => 'Y',
        _ => c,
    }).collect()
}

fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || matches!(c, 
        '–' | '—' | '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}' | '…' | '•' | '·'
    )
}

// ============================================================
// WhitespaceTokenizer
// ============================================================

/// Simple whitespace-based tokenizer
#[derive(Debug, Clone)]
pub struct WhitespaceTokenizer {
    vocab: Vocabulary,
    config: NormalizationConfig,
    unk_id: u32,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        let mut vocab = Vocabulary::with_special_tokens(&["[PAD]", "[UNK]", "[BOS]", "[EOS]"]);
        let unk_id = vocab.get_id("[UNK]").unwrap();
        Self {
            vocab,
            config: NormalizationConfig::default(),
            unk_id,
        }
    }
    
    pub fn with_config(config: NormalizationConfig) -> Self {
        let mut t = Self::new();
        t.config = config;
        t
    }
    
    /// Build tokenizer from a corpus, adding all words to vocabulary
    pub fn from_corpus(texts: &[&str], config: NormalizationConfig) -> Self {
        let vocab = Vocabulary::build_from_corpus(texts, 1);
        let unk_id = vocab.get_id("[UNK]").unwrap();
        Self { vocab, config, unk_id }
    }
    
    pub fn add_word(&mut self, word: &str) -> u32 {
        self.vocab.add_token(word)
    }
    
    /// Build vocabulary from iterator of words
    pub fn build_vocab<'a>(&mut self, words: impl Iterator<Item = &'a str>) {
        for word in words {
            let normalized = if self.config.lowercase {
                word.to_lowercase()
            } else {
                word.to_string()
            };
            self.vocab.add_token(&normalized);
        }
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let normalized = self.normalize(text);
        let mut tokens = Vec::new();
        let mut pos = 0;
        
        for word in normalized.split_whitespace() {
            let start = normalized[pos..].find(word).map(|i| i + pos).unwrap_or(pos);
            let end = start + word.len();
            
            let id = self.vocab.get_id(word).unwrap_or(self.unk_id);
            tokens.push(Token::new(word.to_string(), id, start, end));
            pos = end;
        }
        
        tokens
    }
    
    fn detokenize(&self, tokens: &[Token]) -> String {
        tokens.iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }
    
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get_token(id)
    }
    
    fn normalize(&self, text: &str) -> String {
        normalize_text(text, &self.config)
    }
}

// ============================================================
// WordPieceTokenizer
// ============================================================

/// A simplified WordPiece tokenizer.
///
/// Uses a greedy longest-match-first strategy to break words into subword units.
/// Subword continuations are prefixed with "##".
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    vocab: Vocabulary,
    config: NormalizationConfig,
    max_word_len: usize,
    unk_id: u32,
    continuation_prefix: String,
}

impl WordPieceTokenizer {
    pub fn new() -> Self {
        let mut vocab = Vocabulary::with_special_tokens(&["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]);
        let unk_id = vocab.get_id("[UNK]").unwrap();
        Self {
            vocab,
            config: NormalizationConfig::default(),
            max_word_len: 200,
            unk_id,
            continuation_prefix: "##".to_string(),
        }
    }
    
    /// Build a WordPiece vocabulary from corpus using a simple BPE-like approach
    pub fn train(texts: &[&str], vocab_size: usize, min_freq: usize) -> Self {
        let mut tokenizer = Self::new();
        
        // First pass: collect all characters
        let mut char_freq: HashMap<char, usize> = HashMap::new();
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        
        for text in texts {
            let normalized = normalize_text(text, &tokenizer.config);
            for word in normalized.split_whitespace() {
                *word_freq.entry(word.to_string()).or_insert(0) += 1;
                for c in word.chars() {
                    *char_freq.entry(c).or_insert(0) += 1;
                }
            }
        }
        
        // Add all characters as initial vocabulary
        let mut sorted_chars: Vec<_> = char_freq.into_iter().collect();
        sorted_chars.sort_by(|a, b| b.1.cmp(&a.1));
        for (c, _) in &sorted_chars {
            tokenizer.vocab.add_token(&c.to_string());
        }
        
        // Add continuation versions
        for (c, _) in &sorted_chars {
            tokenizer.vocab.add_token(&format!("##{}", c));
        }
        
        // BPE-like merge iterations
        // Represent each word as a sequence of subword units
        let mut word_splits: HashMap<String, Vec<String>> = HashMap::new();
        for (word, _) in &word_freq {
            let chars: Vec<String> = word.chars().enumerate().map(|(i, c)| {
                if i == 0 {
                    c.to_string()
                } else {
                    format!("##{}", c)
                }
            }).collect();
            word_splits.insert(word.clone(), chars);
        }
        
        // Iteratively merge most frequent adjacent pairs
        while tokenizer.vocab.size() < vocab_size {
            // Count pair frequencies
            let mut pair_freq: HashMap<(String, String), usize> = HashMap::new();
            for (word, freq) in &word_freq {
                if let Some(splits) = word_splits.get(word) {
                    for i in 0..splits.len().saturating_sub(1) {
                        let pair = (splits[i].clone(), splits[i + 1].clone());
                        *pair_freq.entry(pair).or_insert(0) += freq;
                    }
                }
            }
            
            if pair_freq.is_empty() {
                break;
            }
            
            // Find most frequent pair
            let best_pair = pair_freq.into_iter()
                .filter(|(_, freq)| *freq >= min_freq)
                .max_by_key(|(_, freq)| *freq);
            
            let (best_a, best_b) = match best_pair {
                Some((pair, _)) => pair,
                None => break,
            };
            
            // Merge the pair
            let merged = if best_b.starts_with("##") {
                format!("{}{}", best_a, &best_b[2..])
            } else {
                format!("{}{}", best_a, best_b)
            };
            tokenizer.vocab.add_token(&merged);
            
            // Update all word splits
            for (word, _) in &word_freq {
                if let Some(splits) = word_splits.get_mut(word) {
                    let mut new_splits = Vec::new();
                    let mut i = 0;
                    while i < splits.len() {
                        if i + 1 < splits.len() && splits[i] == best_a && splits[i + 1] == best_b {
                            new_splits.push(merged.clone());
                            i += 2;
                        } else {
                            new_splits.push(splits[i].clone());
                            i += 1;
                        }
                    }
                    *splits = new_splits;
                }
            }
        }
        
        tokenizer
    }
    
    /// Tokenize a single word using greedy longest-match-first
    fn tokenize_word(&self, word: &str, offset: usize) -> Vec<Token> {
        if word.len() > self.max_word_len {
            return vec![Token::new("[UNK]".to_string(), self.unk_id, offset, offset + word.len())];
        }
        
        let mut tokens = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut is_first = true;
        
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;
            
            while end > start {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if is_first {
                    substr.clone()
                } else {
                    format!("{}{}", self.continuation_prefix, substr)
                };
                
                if let Some(id) = self.vocab.get_id(&candidate) {
                    let byte_start = offset + chars[..start].iter().map(|c| c.len_utf8()).sum::<usize>();
                    let byte_end = offset + chars[..end].iter().map(|c| c.len_utf8()).sum::<usize>();
                    tokens.push(Token::new(candidate, id, byte_start, byte_end));
                    start = end;
                    is_first = false;
                    found = true;
                    break;
                }
                end -= 1;
            }
            
            if !found {
                // Character not in vocab
                let byte_start = offset + chars[..start].iter().map(|c| c.len_utf8()).sum::<usize>();
                let byte_end = offset + chars[..start + 1].iter().map(|c| c.len_utf8()).sum::<usize>();
                tokens.push(Token::new("[UNK]".to_string(), self.unk_id, byte_start, byte_end));
                start += 1;
                is_first = false;
            }
        }
        
        tokens
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let normalized = self.normalize(text);
        let mut all_tokens = Vec::new();
        let mut pos = 0;
        
        for word in normalized.split_whitespace() {
            let start = normalized[pos..].find(word).map(|i| i + pos).unwrap_or(pos);
            let word_tokens = self.tokenize_word(word, start);
            all_tokens.extend(word_tokens);
            pos = start + word.len();
        }
        
        all_tokens
    }
    
    fn detokenize(&self, tokens: &[Token]) -> String {
        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if token.text.starts_with("##") {
                result.push_str(&token.text[2..]);
            } else {
                if i > 0 {
                    result.push(' ');
                }
                result.push_str(&token.text);
            }
        }
        result
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }
    
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get_token(id)
    }
    
    fn normalize(&self, text: &str) -> String {
        normalize_text(text, &self.config)
    }
}

// ============================================================
// CharacterTokenizer
// ============================================================

/// Character-level tokenizer that maps each character to a token ID
#[derive(Debug, Clone)]
pub struct CharacterTokenizer {
    vocab: Vocabulary,
    config: NormalizationConfig,
    unk_id: u32,
    include_whitespace: bool,
}

impl CharacterTokenizer {
    pub fn new(include_whitespace: bool) -> Self {
        let mut vocab = Vocabulary::with_special_tokens(&["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SPACE]"]);
        let unk_id = vocab.get_id("[UNK]").unwrap();
        
        // Pre-populate with ASCII printable characters
        for c in 32u8..=126 {
            vocab.add_token(&(c as char).to_string());
        }
        
        Self {
            vocab,
            config: NormalizationConfig {
                lowercase: false,
                strip_punctuation: false,
                strip_whitespace: false,
                strip_accents: false,
                collapse_whitespace: false,
            },
            unk_id,
            include_whitespace,
        }
    }
    
    /// Create a character tokenizer from a specific alphabet
    pub fn from_alphabet(chars: &[char]) -> Self {
        let mut vocab = Vocabulary::with_special_tokens(&["[PAD]", "[UNK]", "[BOS]", "[EOS]"]);
        let unk_id = vocab.get_id("[UNK]").unwrap();
        
        for &c in chars {
            vocab.add_token(&c.to_string());
        }
        
        Self {
            vocab,
            config: NormalizationConfig {
                lowercase: false,
                strip_punctuation: false,
                strip_whitespace: false,
                strip_accents: false,
                collapse_whitespace: false,
            },
            unk_id,
            include_whitespace: true,
        }
    }
    
    pub fn add_char(&mut self, c: char) -> u32 {
        self.vocab.add_token(&c.to_string())
    }
}

impl Tokenizer for CharacterTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let normalized = self.normalize(text);
        let mut tokens = Vec::new();
        
        for (i, c) in normalized.char_indices() {
            if c.is_whitespace() && !self.include_whitespace {
                continue;
            }
            
            let text_repr = if c == ' ' && self.vocab.contains("[SPACE]") {
                "[SPACE]".to_string()
            } else {
                c.to_string()
            };
            
            let id = self.vocab.get_id(&text_repr)
                .or_else(|| self.vocab.get_id(&c.to_string()))
                .unwrap_or(self.unk_id);
            
            tokens.push(Token::new(c.to_string(), id, i, i + c.len_utf8()));
        }
        
        tokens
    }
    
    fn detokenize(&self, tokens: &[Token]) -> String {
        tokens.iter().map(|t| t.text.as_str()).collect()
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }
    
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get_token(id)
    }
    
    fn normalize(&self, text: &str) -> String {
        normalize_text(text, &self.config)
    }
}

// ============================================================
// NGramTokenizer
// ============================================================

/// N-gram tokenizer that extracts character or word n-grams
#[derive(Debug, Clone)]
pub struct NGramTokenizer {
    vocab: Vocabulary,
    config: NormalizationConfig,
    n: usize,
    mode: NGramMode,
    unk_id: u32,
}

/// Whether to extract character-level or word-level n-grams
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NGramMode {
    Character,
    Word,
}

impl NGramTokenizer {
    pub fn new(n: usize, mode: NGramMode) -> Self {
        assert!(n > 0, "N-gram size must be positive");
        let mut vocab = Vocabulary::with_special_tokens(&["[PAD]", "[UNK]"]);
        let unk_id = vocab.get_id("[UNK]").unwrap();
        Self {
            vocab,
            config: NormalizationConfig::default(),
            n,
            mode,
            unk_id,
        }
    }
    
    /// Build vocabulary from corpus
    pub fn train(texts: &[&str], n: usize, mode: NGramMode, min_freq: usize) -> Self {
        let mut tokenizer = Self::new(n, mode);
        let mut freq: HashMap<String, usize> = HashMap::new();
        
        for text in texts {
            let normalized = normalize_text(text, &tokenizer.config);
            let ngrams = match mode {
                NGramMode::Character => extract_char_ngrams(&normalized, n),
                NGramMode::Word => extract_word_ngrams(&normalized, n),
            };
            for ng in ngrams {
                *freq.entry(ng).or_insert(0) += 1;
            }
        }
        
        let mut sorted: Vec<_> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        
        for (ngram, count) in sorted {
            if count >= min_freq {
                tokenizer.vocab.add_token(&ngram);
            }
        }
        
        tokenizer
    }
    
    /// Extract n-grams from text without vocabulary lookup
    pub fn extract_ngrams(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize(text);
        match self.mode {
            NGramMode::Character => extract_char_ngrams(&normalized, self.n),
            NGramMode::Word => extract_word_ngrams(&normalized, self.n),
        }
    }
    
    /// Extract n-grams with their counts
    pub fn extract_ngram_counts(&self, text: &str) -> HashMap<String, usize> {
        let ngrams = self.extract_ngrams(text);
        let mut counts: HashMap<String, usize> = HashMap::new();
        for ng in ngrams {
            *counts.entry(ng).or_insert(0) += 1;
        }
        counts
    }
}

fn extract_char_ngrams(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return Vec::new();
    }
    (0..=chars.len() - n)
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

fn extract_word_ngrams(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return Vec::new();
    }
    (0..=words.len() - n)
        .map(|i| words[i..i + n].join(" "))
        .collect()
}

impl Tokenizer for NGramTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let normalized = self.normalize(text);
        let ngrams = match self.mode {
            NGramMode::Character => extract_char_ngrams(&normalized, self.n),
            NGramMode::Word => extract_word_ngrams(&normalized, self.n),
        };
        
        let mut tokens = Vec::new();
        let mut pos = 0;
        for ng in ngrams {
            let start = normalized[pos..].find(&ng).map(|i| i + pos).unwrap_or(pos);
            let end = start + ng.len();
            let id = self.vocab.get_id(&ng).unwrap_or(self.unk_id);
            tokens.push(Token::new(ng, id, start, end));
            // For overlapping n-grams, advance by the unit size
            match self.mode {
                NGramMode::Character => {
                    pos = start + normalized[start..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                }
                NGramMode::Word => {
                    // Advance past the first word in this n-gram
                    if let Some(space_pos) = normalized[start..].find(' ') {
                        pos = start + space_pos + 1;
                    } else {
                        pos = end;
                    }
                }
            }
        }
        
        tokens
    }
    
    fn detokenize(&self, tokens: &[Token]) -> String {
        // For n-grams, detokenization reconstructs from overlapping segments
        if tokens.is_empty() {
            return String::new();
        }
        
        match self.mode {
            NGramMode::Character => {
                let mut result = tokens[0].text.clone();
                for token in &tokens[1..] {
                    let chars: Vec<char> = token.text.chars().collect();
                    if let Some(last) = chars.last() {
                        result.push(*last);
                    }
                }
                result
            }
            NGramMode::Word => {
                let mut words: Vec<String> = Vec::new();
                for token in tokens {
                    let token_words: Vec<&str> = token.text.split_whitespace().collect();
                    if words.is_empty() {
                        words.extend(token_words.iter().map(|s| s.to_string()));
                    } else if let Some(last) = token_words.last() {
                        words.push(last.to_string());
                    }
                }
                words.join(" ")
            }
        }
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }
    
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get_token(id)
    }
    
    fn normalize(&self, text: &str) -> String {
        normalize_text(text, &self.config)
    }
}

// ============================================================
// Tokenizer Pipeline: compose normalization + tokenization
// ============================================================

/// A pipeline that chains normalization and tokenization steps
#[derive(Debug, Clone)]
pub struct TokenizerPipeline {
    pre_tokenizer: PreTokenizer,
    main_tokenizer: TokenizerKind,
    post_processor: PostProcessor,
}

#[derive(Debug, Clone)]
enum TokenizerKind {
    Whitespace(WhitespaceTokenizer),
    WordPiece(WordPieceTokenizer),
    Character(CharacterTokenizer),
    NGram(NGramTokenizer),
}

#[derive(Debug, Clone)]
struct PreTokenizer {
    split_on_punctuation: bool,
    split_digits: bool,
}

impl Default for PreTokenizer {
    fn default() -> Self {
        Self {
            split_on_punctuation: false,
            split_digits: false,
        }
    }
}

impl PreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<(String, usize)> {
        if !self.split_on_punctuation && !self.split_digits {
            return vec![(text.to_string(), 0)];
        }
        
        let mut segments = Vec::new();
        let mut current = String::new();
        let mut current_start = 0;
        
        for (i, c) in text.char_indices() {
            let should_split = (self.split_on_punctuation && is_punctuation(c))
                || (self.split_digits && c.is_ascii_digit());
            
            if should_split {
                if !current.is_empty() {
                    segments.push((current.clone(), current_start));
                    current.clear();
                }
                segments.push((c.to_string(), i));
                current_start = i + c.len_utf8();
            } else {
                if current.is_empty() {
                    current_start = i;
                }
                current.push(c);
            }
        }
        
        if !current.is_empty() {
            segments.push((current, current_start));
        }
        
        segments
    }
}

#[derive(Debug, Clone)]
struct PostProcessor {
    add_bos: bool,
    add_eos: bool,
    max_length: Option<usize>,
    truncation_strategy: TruncationStrategy,
}

#[derive(Debug, Clone, Copy)]
enum TruncationStrategy {
    TruncateEnd,
    TruncateStart,
}

impl Default for PostProcessor {
    fn default() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
            max_length: None,
            truncation_strategy: TruncationStrategy::TruncateEnd,
        }
    }
}

impl PostProcessor {
    fn process(&self, mut tokens: Vec<Token>, tokenizer: &dyn Tokenizer) -> Vec<Token> {
        if self.add_bos {
            if let Some(bos_id) = tokenizer.token_to_id("[BOS]") {
                tokens.insert(0, Token::new("[BOS]".to_string(), bos_id, 0, 0));
            }
        }
        
        if self.add_eos {
            if let Some(eos_id) = tokenizer.token_to_id("[EOS]") {
                let end = tokens.last().map(|t| t.end).unwrap_or(0);
                tokens.push(Token::new("[EOS]".to_string(), eos_id, end, end));
            }
        }
        
        if let Some(max_len) = self.max_length {
            if tokens.len() > max_len {
                match self.truncation_strategy {
                    TruncationStrategy::TruncateEnd => {
                        tokens.truncate(max_len);
                    }
                    TruncationStrategy::TruncateStart => {
                        let skip = tokens.len() - max_len;
                        tokens = tokens.into_iter().skip(skip).collect();
                    }
                }
            }
        }
        
        tokens
    }
}

impl TokenizerPipeline {
    pub fn whitespace() -> Self {
        Self {
            pre_tokenizer: PreTokenizer::default(),
            main_tokenizer: TokenizerKind::Whitespace(WhitespaceTokenizer::new()),
            post_processor: PostProcessor::default(),
        }
    }
    
    pub fn wordpiece(tokenizer: WordPieceTokenizer) -> Self {
        Self {
            pre_tokenizer: PreTokenizer { split_on_punctuation: true, split_digits: false },
            main_tokenizer: TokenizerKind::WordPiece(tokenizer),
            post_processor: PostProcessor::default(),
        }
    }
    
    pub fn character(include_whitespace: bool) -> Self {
        Self {
            pre_tokenizer: PreTokenizer::default(),
            main_tokenizer: TokenizerKind::Character(CharacterTokenizer::new(include_whitespace)),
            post_processor: PostProcessor::default(),
        }
    }
    
    pub fn ngram(n: usize, mode: NGramMode) -> Self {
        Self {
            pre_tokenizer: PreTokenizer::default(),
            main_tokenizer: TokenizerKind::NGram(NGramTokenizer::new(n, mode)),
            post_processor: PostProcessor::default(),
        }
    }
    
    pub fn with_bos_eos(mut self) -> Self {
        self.post_processor.add_bos = true;
        self.post_processor.add_eos = true;
        self
    }
    
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.post_processor.max_length = Some(max_length);
        self
    }
    
    fn get_tokenizer(&self) -> &dyn Tokenizer {
        match &self.main_tokenizer {
            TokenizerKind::Whitespace(t) => t,
            TokenizerKind::WordPiece(t) => t,
            TokenizerKind::Character(t) => t,
            TokenizerKind::NGram(t) => t,
        }
    }
    
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let tokenizer = self.get_tokenizer();
        let segments = self.pre_tokenizer.pre_tokenize(text);
        
        let mut all_tokens = Vec::new();
        for (segment, _offset) in segments {
            let tokens = tokenizer.tokenize(&segment);
            all_tokens.extend(tokens);
        }
        
        self.post_processor.process(all_tokens, tokenizer)
    }
    
    pub fn detokenize(&self, tokens: &[Token]) -> String {
        let tokenizer = self.get_tokenizer();
        // Filter out special tokens
        let filtered: Vec<Token> = tokens.iter()
            .filter(|t| !t.text.starts_with('[') || !t.text.ends_with(']'))
            .cloned()
            .collect();
        tokenizer.detokenize(&filtered)
    }
}

// ============================================================
// Batch tokenization utilities
// ============================================================

/// Tokenize a batch of texts, optionally with padding
pub fn batch_tokenize(tokenizer: &dyn Tokenizer, texts: &[&str], pad_to_max: bool) -> Vec<Vec<Token>> {
    let mut batches: Vec<Vec<Token>> = texts.iter()
        .map(|text| tokenizer.tokenize(text))
        .collect();
    
    if pad_to_max {
        let max_len = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
        
        for batch in &mut batches {
            while batch.len() < max_len {
                let pos = batch.last().map(|t| t.end).unwrap_or(0);
                batch.push(Token::new("[PAD]".to_string(), pad_id, pos, pos));
            }
        }
    }
    
    batches
}

/// Extract token IDs from a list of tokens
pub fn tokens_to_ids(tokens: &[Token]) -> Vec<u32> {
    tokens.iter().map(|t| t.id).collect()
}

/// Compute token overlap between two sequences
pub fn token_overlap(a: &[Token], b: &[Token]) -> (usize, usize, usize) {
    let a_set: std::collections::HashSet<&str> = a.iter().map(|t| t.text.as_str()).collect();
    let b_set: std::collections::HashSet<&str> = b.iter().map(|t| t.text.as_str()).collect();
    
    let intersection = a_set.intersection(&b_set).count();
    (intersection, a_set.len(), b_set.len())
}

/// Count n-gram occurrences in a token sequence
pub fn count_token_ngrams(tokens: &[Token], n: usize) -> HashMap<Vec<u32>, usize> {
    let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let mut counts: HashMap<Vec<u32>, usize> = HashMap::new();
    
    if ids.len() >= n {
        for i in 0..=ids.len() - n {
            let ngram = ids[i..i + n].to_vec();
            *counts.entry(ngram).or_insert(0) += 1;
        }
    }
    
    counts
}

/// Compute clipped n-gram counts (for BLEU)
pub fn clipped_ngram_counts(
    candidate_counts: &HashMap<Vec<u32>, usize>,
    reference_counts: &HashMap<Vec<u32>, usize>,
) -> HashMap<Vec<u32>, usize> {
    let mut clipped = HashMap::new();
    for (ngram, &cand_count) in candidate_counts {
        let ref_count = reference_counts.get(ngram).copied().unwrap_or(0);
        let clipped_count = cand_count.min(ref_count);
        if clipped_count > 0 {
            clipped.insert(ngram.clone(), clipped_count);
        }
    }
    clipped
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_whitespace_tokenizer_basic() {
        let mut tokenizer = WhitespaceTokenizer::new();
        tokenizer.add_word("hello");
        tokenizer.add_word("world");
        
        let tokens = tokenizer.tokenize("Hello World");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }
    
    #[test]
    fn test_whitespace_tokenizer_unknown() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("unknown words here");
        assert_eq!(tokens.len(), 3);
        // All should be UNK
        for token in &tokens {
            assert_eq!(token.id, tokenizer.token_to_id("[UNK]").unwrap());
        }
    }
    
    #[test]
    fn test_whitespace_detokenize() {
        let mut tokenizer = WhitespaceTokenizer::new();
        tokenizer.add_word("hello");
        tokenizer.add_word("world");
        
        let tokens = tokenizer.tokenize("hello world");
        let reconstructed = tokenizer.detokenize(&tokens);
        assert_eq!(reconstructed, "hello world");
    }
    
    #[test]
    fn test_whitespace_from_corpus() {
        let texts = vec!["the cat sat", "the dog ran", "a cat ran"];
        let tokenizer = WhitespaceTokenizer::from_corpus(&texts, NormalizationConfig::default());
        
        assert!(tokenizer.token_to_id("the").is_some());
        assert!(tokenizer.token_to_id("cat").is_some());
        assert!(tokenizer.token_to_id("ran").is_some());
    }
    
    #[test]
    fn test_wordpiece_tokenizer_known_words() {
        let texts = vec!["hello world testing tokenization"];
        let tokenizer = WordPieceTokenizer::train(&texts, 100, 1);
        
        let tokens = tokenizer.tokenize("hello world");
        assert!(!tokens.is_empty());
        let reconstructed = tokenizer.detokenize(&tokens);
        assert_eq!(reconstructed.replace(" ", "").replace("##", ""), 
                   "helloworld");
    }
    
    #[test]
    fn test_wordpiece_subword_splitting() {
        let mut tokenizer = WordPieceTokenizer::new();
        // Add individual characters and some subwords
        for c in 'a'..='z' {
            tokenizer.vocab.add_token(&c.to_string());
            tokenizer.vocab.add_token(&format!("##{}", c));
        }
        tokenizer.vocab.add_token("un");
        tokenizer.vocab.add_token("##known");
        
        let tokens = tokenizer.tokenize("unknown");
        assert!(tokens.len() >= 1);
    }
    
    #[test]
    fn test_character_tokenizer() {
        let tokenizer = CharacterTokenizer::new(false);
        let tokens = tokenizer.tokenize("abc");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "a");
        assert_eq!(tokens[1].text, "b");
        assert_eq!(tokens[2].text, "c");
    }
    
    #[test]
    fn test_character_tokenizer_with_whitespace() {
        let tokenizer = CharacterTokenizer::new(true);
        let tokens = tokenizer.tokenize("a b");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1].text, " ");
    }
    
    #[test]
    fn test_character_tokenizer_from_alphabet() {
        let tokenizer = CharacterTokenizer::from_alphabet(&['a', 'b', 'c']);
        let tokens = tokenizer.tokenize("abcd");
        assert_eq!(tokens.len(), 4);
        // 'd' may or may not be in vocabulary depending on implementation
        // just verify we get 4 tokens
        assert_eq!(tokens[0].text, "a");
    }
    
    #[test]
    fn test_character_detokenize() {
        let tokenizer = CharacterTokenizer::new(true);
        let tokens = tokenizer.tokenize("hello");
        let result = tokenizer.detokenize(&tokens);
        assert_eq!(result, "hello");
    }
    
    #[test]
    fn test_ngram_character() {
        let tokenizer = NGramTokenizer::new(2, NGramMode::Character);
        let ngrams = tokenizer.extract_ngrams("abcd");
        assert_eq!(ngrams, vec!["ab", "bc", "cd"]);
    }
    
    #[test]
    fn test_ngram_word() {
        let tokenizer = NGramTokenizer::new(2, NGramMode::Word);
        let ngrams = tokenizer.extract_ngrams("the cat sat on the mat");
        assert_eq!(ngrams.len(), 5);
        assert_eq!(ngrams[0], "the cat");
        assert_eq!(ngrams[4], "the mat");
    }
    
    #[test]
    fn test_ngram_counts() {
        let tokenizer = NGramTokenizer::new(1, NGramMode::Word);
        let counts = tokenizer.extract_ngram_counts("the cat and the dog");
        assert_eq!(counts.get("the"), Some(&2));
        assert_eq!(counts.get("cat"), Some(&1));
    }
    
    #[test]
    fn test_ngram_empty() {
        let tokenizer = NGramTokenizer::new(5, NGramMode::Word);
        let ngrams = tokenizer.extract_ngrams("too short");
        assert!(ngrams.is_empty());
    }
    
    #[test]
    fn test_ngram_train() {
        let texts = vec!["hello world", "hello there", "world hello"];
        let tokenizer = NGramTokenizer::train(&texts, 2, NGramMode::Word, 1);
        
        assert!(tokenizer.token_to_id("hello world").is_some());
        assert!(tokenizer.token_to_id("hello there").is_some());
    }
    
    #[test]
    fn test_vocabulary_operations() {
        let mut vocab = Vocabulary::new();
        let id1 = vocab.add_token("hello");
        let id2 = vocab.add_token("world");
        let id3 = vocab.add_token("hello"); // duplicate
        
        assert_eq!(id1, id3); // same token, same ID
        assert_ne!(id1, id2);
        assert_eq!(vocab.size(), 2);
        assert_eq!(vocab.get_token(id1), Some("hello"));
        assert_eq!(vocab.get_id("world"), Some(id2));
    }
    
    #[test]
    fn test_vocabulary_special_tokens() {
        let vocab = Vocabulary::with_special_tokens(&["[PAD]", "[UNK]"]);
        assert!(vocab.is_special("[PAD]"));
        assert!(vocab.is_special("[UNK]"));
        assert!(!vocab.is_special("hello"));
        assert_eq!(vocab.special_token_ids().len(), 2);
    }
    
    #[test]
    fn test_vocabulary_merge() {
        let mut vocab1 = Vocabulary::new();
        vocab1.add_token("hello");
        vocab1.add_token("world");
        
        let mut vocab2 = Vocabulary::new();
        vocab2.add_token("world");
        vocab2.add_token("rust");
        
        vocab1.merge(&vocab2);
        assert_eq!(vocab1.size(), 3);
        assert!(vocab1.contains("hello"));
        assert!(vocab1.contains("world"));
        assert!(vocab1.contains("rust"));
    }
    
    #[test]
    fn test_normalization_lowercase() {
        let config = NormalizationConfig {
            lowercase: true,
            ..Default::default()
        };
        assert_eq!(normalize_text("Hello WORLD", &config), "hello world");
    }
    
    #[test]
    fn test_normalization_strip_punctuation() {
        let config = NormalizationConfig {
            strip_punctuation: true,
            lowercase: false,
            ..Default::default()
        };
        assert_eq!(normalize_text("Hello, World!", &config).trim(), "Hello World");
    }
    
    #[test]
    fn test_normalization_collapse_whitespace() {
        let config = NormalizationConfig {
            collapse_whitespace: true,
            lowercase: false,
            ..Default::default()
        };
        assert_eq!(normalize_text("hello   world", &config), "hello world");
    }
    
    #[test]
    fn test_normalization_strip_accents() {
        let config = NormalizationConfig {
            strip_accents: true,
            lowercase: false,
            ..Default::default()
        };
        assert_eq!(normalize_text("café résumé", &config), "cafe resume");
    }
    
    #[test]
    fn test_batch_tokenize() {
        let mut tokenizer = WhitespaceTokenizer::new();
        tokenizer.add_word("hello");
        tokenizer.add_word("world");
        tokenizer.add_word("hi");
        
        let texts = vec!["hello world", "hi"];
        let batches = batch_tokenize(&tokenizer, &texts, true);
        
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), batches[1].len()); // padded to same length
    }
    
    #[test]
    fn test_tokens_to_ids() {
        let tokens = vec![
            Token::new("a".to_string(), 1, 0, 1),
            Token::new("b".to_string(), 2, 2, 3),
        ];
        assert_eq!(tokens_to_ids(&tokens), vec![1, 2]);
    }
    
    #[test]
    fn test_token_overlap() {
        let a = vec![
            Token::new("the".to_string(), 1, 0, 3),
            Token::new("cat".to_string(), 2, 4, 7),
            Token::new("sat".to_string(), 3, 8, 11),
        ];
        let b = vec![
            Token::new("the".to_string(), 1, 0, 3),
            Token::new("dog".to_string(), 4, 4, 7),
            Token::new("sat".to_string(), 3, 8, 11),
        ];
        let (overlap, a_len, b_len) = token_overlap(&a, &b);
        assert_eq!(overlap, 2); // "the" and "sat"
        assert_eq!(a_len, 3);
        assert_eq!(b_len, 3);
    }
    
    #[test]
    fn test_count_token_ngrams() {
        let tokens = vec![
            Token::new("the".to_string(), 1, 0, 3),
            Token::new("cat".to_string(), 2, 4, 7),
            Token::new("sat".to_string(), 3, 8, 11),
            Token::new("the".to_string(), 1, 12, 15),
        ];
        
        let unigrams = count_token_ngrams(&tokens, 1);
        assert_eq!(unigrams.get(&vec![1u32]), Some(&2)); // "the" appears twice
        
        let bigrams = count_token_ngrams(&tokens, 2);
        assert_eq!(bigrams.len(), 3);
    }
    
    #[test]
    fn test_clipped_ngram_counts() {
        let mut cand = HashMap::new();
        cand.insert(vec![1u32], 3);
        cand.insert(vec![2u32], 1);
        
        let mut refs = HashMap::new();
        refs.insert(vec![1u32], 2);
        refs.insert(vec![2u32], 5);
        
        let clipped = clipped_ngram_counts(&cand, &refs);
        assert_eq!(clipped.get(&vec![1u32]), Some(&2)); // clipped from 3 to 2
        assert_eq!(clipped.get(&vec![2u32]), Some(&1)); // stays at 1
    }
    
    #[test]
    fn test_pipeline_whitespace() {
        let pipeline = TokenizerPipeline::whitespace();
        let tokens = pipeline.tokenize("hello world");
        assert_eq!(tokens.len(), 2);
    }
    
    #[test]
    fn test_pipeline_with_max_length() {
        let pipeline = TokenizerPipeline::character(true)
            .with_max_length(5);
        let tokens = pipeline.tokenize("hello world");
        assert!(tokens.len() <= 5);
    }
    
    #[test]
    fn test_pipeline_ngram() {
        let pipeline = TokenizerPipeline::ngram(2, NGramMode::Character);
        let tokens = pipeline.tokenize("abcd");
        assert_eq!(tokens.len(), 3); // "ab", "bc", "cd"
    }
    
    #[test]
    fn test_pre_tokenizer_punctuation_split() {
        let pre = PreTokenizer {
            split_on_punctuation: true,
            split_digits: false,
        };
        let segments = pre.pre_tokenize("hello, world!");
        assert!(segments.len() >= 3);
    }
    
    #[test]
    fn test_pre_tokenizer_digit_split() {
        let pre = PreTokenizer {
            split_on_punctuation: false,
            split_digits: true,
        };
        let segments = pre.pre_tokenize("test123");
        assert!(segments.len() >= 4); // "test", "1", "2", "3"
    }
}
