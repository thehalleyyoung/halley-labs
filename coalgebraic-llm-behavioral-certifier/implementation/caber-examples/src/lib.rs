//! Shared types and utilities for CABER examples.
//!
//! Provides simplified versions of core types sufficient to demonstrate
//! L*-style automaton learning, property checking, and certificate generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

/// Opaque state identifier within a learned automaton.
pub type StateId = usize;

/// A symbol drawn from the input alphabet.
pub type Symbol = String;

/// A word is a sequence of symbols (i.e., a prompt or prompt prefix).
pub type Word = Vec<Symbol>;

// ---------------------------------------------------------------------------
// Automaton types
// ---------------------------------------------------------------------------

/// A single state in a learned automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatonState {
    pub id: StateId,
    pub label: String,
    pub is_accepting: bool,
}

/// A transition between two automaton states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatonTransition {
    pub from: StateId,
    pub symbol: Symbol,
    pub to: StateId,
}

/// A deterministic finite automaton learned via L*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedAutomaton {
    pub states: Vec<AutomatonState>,
    pub transitions: Vec<AutomatonTransition>,
    pub initial_state: StateId,
    pub alphabet: Vec<Symbol>,
}

impl LearnedAutomaton {
    pub fn new(alphabet: Vec<Symbol>) -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            initial_state: 0,
            alphabet,
        }
    }

    pub fn add_state(&mut self, label: &str, accepting: bool) -> StateId {
        let id = self.states.len();
        self.states.push(AutomatonState {
            id,
            label: label.to_string(),
            is_accepting: accepting,
        });
        id
    }

    pub fn add_transition(&mut self, from: StateId, symbol: &str, to: StateId) {
        self.transitions.push(AutomatonTransition {
            from,
            symbol: symbol.to_string(),
            to,
        });
    }

    /// Return the target state for a given (state, symbol) pair, if defined.
    pub fn step(&self, state: StateId, symbol: &str) -> Option<StateId> {
        self.transitions
            .iter()
            .find(|t| t.from == state && t.symbol == symbol)
            .map(|t| t.to)
    }

    /// Run the automaton on a word, returning the final state (or None).
    pub fn run(&self, word: &[String]) -> Option<StateId> {
        let mut current = self.initial_state;
        for sym in word {
            current = self.step(current, sym)?;
        }
        Some(current)
    }
}

// ---------------------------------------------------------------------------
// Mock LLM
// ---------------------------------------------------------------------------

/// A deterministic mock LLM that maps prompt keywords to response labels.
#[derive(Debug, Clone)]
pub struct MockLLM {
    pub name: String,
    /// Maps a keyword (found anywhere in the prompt) to a response label.
    pub response_map: Vec<(String, String)>,
    /// Default response when no keyword matches.
    pub default_response: String,
}

impl MockLLM {
    pub fn new(name: &str, default_response: &str) -> Self {
        Self {
            name: name.to_string(),
            response_map: Vec::new(),
            default_response: default_response.to_string(),
        }
    }

    pub fn add_rule(&mut self, keyword: &str, response: &str) {
        self.response_map
            .push((keyword.to_lowercase(), response.to_string()));
    }

    /// Query the mock model. Returns the response label for the first matching
    /// keyword found in `prompt`.
    pub fn query(&self, prompt: &str) -> String {
        let lower = prompt.to_lowercase();
        for (kw, resp) in &self.response_map {
            if lower.contains(kw.as_str()) {
                return resp.clone();
            }
        }
        self.default_response.clone()
    }
}

// ---------------------------------------------------------------------------
// Property checking
// ---------------------------------------------------------------------------

/// Outcome of checking a single property on an automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyVerdict {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

// ---------------------------------------------------------------------------
// Certificate
// ---------------------------------------------------------------------------

/// A simplified behavioral certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub model_id: String,
    pub issued_at: String,
    pub num_states: usize,
    pub num_transitions: usize,
    pub properties: Vec<PropertyVerdict>,
    pub total_queries: usize,
    pub signature: String,
}

// ---------------------------------------------------------------------------
// Observation table (L* learning)
// ---------------------------------------------------------------------------

/// Observation table used in L*-style learning.
#[derive(Debug, Clone)]
pub struct ObservationTable {
    /// Rows for short prefixes (S).
    pub short_prefixes: Vec<Word>,
    /// Rows for long prefixes (SA = S · Σ).
    pub long_prefixes: Vec<Word>,
    /// Column suffixes (E).
    pub suffixes: Vec<Word>,
    /// Table entries: maps (prefix, suffix) → response label.
    pub entries: HashMap<(Word, Word), String>,
}

impl ObservationTable {
    pub fn new() -> Self {
        Self {
            short_prefixes: vec![vec![]], // ε
            long_prefixes: Vec::new(),
            suffixes: vec![vec![]], // ε
            entries: HashMap::new(),
        }
    }

    /// Row signature for a given prefix.
    pub fn row(&self, prefix: &Word) -> Vec<String> {
        self.suffixes
            .iter()
            .map(|suf| {
                self.entries
                    .get(&(prefix.clone(), suf.clone()))
                    .cloned()
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Collect the set of distinct row signatures among short prefixes.
    pub fn distinct_rows(&self) -> Vec<Vec<String>> {
        let mut seen: Vec<Vec<String>> = Vec::new();
        for sp in &self.short_prefixes {
            let r = self.row(sp);
            if !seen.contains(&r) {
                seen.push(r);
            }
        }
        seen
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

pub fn print_separator() {
    println!("{}", "═".repeat(72));
}

pub fn print_header(title: &str) {
    println!();
    print_separator();
    println!("  {}", title);
    print_separator();
}
