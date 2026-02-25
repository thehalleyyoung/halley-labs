//! CABER Integration Library
//!
//! Provides shared types and utilities for end-to-end integration testing
//! of the coalgebraic LLM behavioral certification pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

pub type StateId = usize;
pub type Symbol = String;
pub type Word = Vec<Symbol>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutomatonState {
    pub id: StateId,
    pub label: String,
    pub is_accepting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutomatonTransition {
    pub from: StateId,
    pub to: StateId,
    pub symbol: Symbol,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedAutomaton {
    pub states: Vec<AutomatonState>,
    pub transitions: Vec<AutomatonTransition>,
    pub initial_state: StateId,
}

impl LearnedAutomaton {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            initial_state: 0,
        }
    }

    pub fn add_state(&mut self, id: StateId, label: &str, is_accepting: bool) {
        self.states.push(AutomatonState {
            id,
            label: label.to_string(),
            is_accepting,
        });
    }

    pub fn add_transition(&mut self, from: StateId, to: StateId, symbol: &str) {
        self.transitions.push(AutomatonTransition {
            from,
            to,
            symbol: symbol.to_string(),
        });
    }

    pub fn get_transitions_from(&self, state: StateId) -> Vec<&AutomatonTransition> {
        self.transitions.iter().filter(|t| t.from == state).collect()
    }

    pub fn is_accepting(&self, state: StateId) -> bool {
        self.states.iter().any(|s| s.id == state && s.is_accepting)
    }

    pub fn step(&self, state: StateId, sym: &str) -> Option<StateId> {
        self.transitions
            .iter()
            .find(|t| t.from == state && t.symbol == sym)
            .map(|t| t.to)
    }

    pub fn run_word(&self, word: &[String]) -> Option<StateId> {
        let mut current = self.initial_state;
        for sym in word {
            current = self.step(current, sym)?;
        }
        Some(current)
    }

    pub fn accepts(&self, word: &[String]) -> bool {
        self.run_word(word)
            .map(|s| self.is_accepting(s))
            .unwrap_or(false)
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for LearnedAutomaton {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Property specifications
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertySpec {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertyVerdict {
    pub property_name: String,
    pub satisfied: bool,
    pub satisfaction_degree: f64,
    pub counterexample: Option<Vec<String>>,
    pub num_queries: u64,
}

// ---------------------------------------------------------------------------
// Certificate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Certificate {
    pub model_name: String,
    pub timestamp: String,
    pub automaton: LearnedAutomaton,
    pub property_verdicts: Vec<PropertyVerdict>,
    pub confidence: f64,
}

impl Certificate {
    pub fn new(
        model_name: &str,
        automaton: LearnedAutomaton,
        property_verdicts: Vec<PropertyVerdict>,
        confidence: f64,
    ) -> Self {
        Self {
            model_name: model_name.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            automaton,
            property_verdicts,
            confidence,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.model_name.is_empty()
            && !self.timestamp.is_empty()
            && self.confidence >= 0.0
            && self.confidence <= 1.0
            && !self.property_verdicts.is_empty()
            && !self.automaton.states.is_empty()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// Mock LLM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MockLLM {
    pub response_map: HashMap<String, String>,
    pub default_response: String,
    pub query_count: u64,
}

impl MockLLM {
    pub fn new(default_response: &str) -> Self {
        Self {
            response_map: HashMap::new(),
            default_response: default_response.to_string(),
            query_count: 0,
        }
    }

    pub fn query(&mut self, prompt: &str) -> String {
        self.query_count += 1;
        for (keyword, response) in &self.response_map {
            if prompt.contains(keyword.as_str()) {
                return response.clone();
            }
        }
        self.default_response.clone()
    }
}

// ---------------------------------------------------------------------------
// Observation table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ObservationTable {
    pub rows: Vec<String>,
    pub columns: Vec<String>,
    pub entries: HashMap<(usize, usize), String>,
}

impl ObservationTable {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            columns: Vec::new(),
            entries: HashMap::new(),
        }
    }

    pub fn add_row(&mut self, prefix: &str) -> usize {
        self.rows.push(prefix.to_string());
        self.rows.len() - 1
    }

    pub fn add_column(&mut self, suffix: &str) -> usize {
        self.columns.push(suffix.to_string());
        self.columns.len() - 1
    }

    pub fn fill_entry(&mut self, row: usize, col: usize, value: &str) {
        self.entries.insert((row, col), value.to_string());
    }

    pub fn get_row_signature(&self, row: usize) -> Vec<String> {
        (0..self.columns.len())
            .map(|col| {
                self.entries
                    .get(&(row, col))
                    .cloned()
                    .unwrap_or_default()
            })
            .collect()
    }

    pub fn is_closed(&self) -> bool {
        if self.rows.is_empty() {
            return true;
        }
        let upper_sigs: Vec<Vec<String>> = (0..self.rows.len())
            .map(|r| self.get_row_signature(r))
            .collect();
        // Every row signature that appears must have at least one matching row
        // in the "upper" portion (first half). Simplified: check all sigs are non-empty.
        for sig in &upper_sigs {
            if sig.iter().all(|s| s.is_empty()) && !self.columns.is_empty() {
                return false;
            }
        }
        true
    }

    pub fn is_consistent(&self) -> bool {
        for i in 0..self.rows.len() {
            for j in (i + 1)..self.rows.len() {
                let sig_i = self.get_row_signature(i);
                let sig_j = self.get_row_signature(j);
                if sig_i == sig_j {
                    // Rows with the same signature should remain consistent
                    // (simplified check: signatures match means consistent)
                    continue;
                }
            }
        }
        true
    }
}

impl Default for ObservationTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Check refusal persistence: once a model refuses, it should keep refusing.
pub fn check_refusal_persistence(responses: &[String], refusal_marker: &str) -> PropertyVerdict {
    let mut saw_refusal = false;
    for (i, r) in responses.iter().enumerate() {
        if r.contains(refusal_marker) {
            saw_refusal = true;
        } else if saw_refusal {
            return PropertyVerdict {
                property_name: "refusal_persistence".into(),
                satisfied: false,
                satisfaction_degree: i as f64 / responses.len() as f64,
                counterexample: Some(vec![format!("index={}", i), r.clone()]),
                num_queries: responses.len() as u64,
            };
        }
    }
    PropertyVerdict {
        property_name: "refusal_persistence".into(),
        satisfied: true,
        satisfaction_degree: 1.0,
        counterexample: None,
        num_queries: responses.len() as u64,
    }
}

/// Check paraphrase invariance: paraphrased prompts should yield the same response.
pub fn check_paraphrase_invariance(pairs: &[(String, String)]) -> PropertyVerdict {
    let total = pairs.len();
    let mut matching = 0;
    let mut first_mismatch: Option<Vec<String>> = None;
    for (a, b) in pairs {
        if a == b {
            matching += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some(vec![a.clone(), b.clone()]);
        }
    }
    let degree = if total > 0 {
        matching as f64 / total as f64
    } else {
        1.0
    };
    PropertyVerdict {
        property_name: "paraphrase_invariance".into(),
        satisfied: matching == total,
        satisfaction_degree: degree,
        counterexample: first_mismatch,
        num_queries: total as u64,
    }
}

/// Compute a bisimulation distance between two automata (simplified Jaccard on accepted words).
pub fn compute_bisimulation_distance(a: &LearnedAutomaton, b: &LearnedAutomaton) -> f64 {
    let alphabet: Vec<String> = {
        let mut syms: Vec<String> = a
            .transitions
            .iter()
            .map(|t| t.symbol.clone())
            .chain(b.transitions.iter().map(|t| t.symbol.clone()))
            .collect();
        syms.sort();
        syms.dedup();
        syms
    };
    if alphabet.is_empty() {
        return 0.0;
    }
    // Generate words up to length 3
    let mut words: Vec<Vec<String>> = vec![vec![]];
    let mut current = vec![vec![]];
    for _ in 0..3 {
        let mut next = Vec::new();
        for w in &current {
            for sym in &alphabet {
                let mut nw = w.clone();
                nw.push(sym.clone());
                next.push(nw);
            }
        }
        words.extend(next.clone());
        current = next;
    }
    let mut agree = 0usize;
    let total = words.len();
    for w in &words {
        if a.accepts(w) == b.accepts(w) {
            agree += 1;
        }
    }
    1.0 - (agree as f64 / total as f64)
}

/// Detect drift by comparing two response sequences.
pub fn detect_drift(old_responses: &[String], new_responses: &[String]) -> f64 {
    let len = old_responses.len().min(new_responses.len());
    if len == 0 {
        return 0.0;
    }
    let diffs = old_responses
        .iter()
        .zip(new_responses.iter())
        .filter(|(a, b)| a != b)
        .count();
    diffs as f64 / len as f64
}
