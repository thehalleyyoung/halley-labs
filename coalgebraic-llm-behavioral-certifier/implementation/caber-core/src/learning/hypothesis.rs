//! Hypothesis automaton construction and refinement.
//!
//! The hypothesis is the current conjecture about the target system,
//! constructed from a closed and consistent observation table.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Local type aliases (to be swapped for coalgebra module types later)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
    pub fn epsilon() -> Self { Self(String::new()) }
    pub fn is_epsilon(&self) -> bool { self.0.is_empty() }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_epsilon() { write!(f, "ε") } else { write!(f, "{}", self.0) }
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self { Self(s.to_string()) }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Word {
    pub symbols: Vec<Symbol>,
}

impl Word {
    pub fn empty() -> Self { Self { symbols: Vec::new() } }
    pub fn singleton(sym: Symbol) -> Self { Self { symbols: vec![sym] } }
    pub fn from_symbols(symbols: Vec<Symbol>) -> Self { Self { symbols } }
    pub fn from_str_slice(parts: &[&str]) -> Self {
        Self { symbols: parts.iter().map(|s| Symbol::new(*s)).collect() }
    }
    pub fn len(&self) -> usize { self.symbols.len() }
    pub fn is_empty(&self) -> bool { self.symbols.is_empty() }
    pub fn concat(&self, other: &Word) -> Word {
        let mut syms = self.symbols.clone();
        syms.extend(other.symbols.iter().cloned());
        Word { symbols: syms }
    }
    pub fn prefix(&self, n: usize) -> Word {
        Word { symbols: self.symbols[..n.min(self.symbols.len())].to_vec() }
    }
    pub fn suffix_from(&self, start: usize) -> Word {
        if start >= self.symbols.len() { return Word::empty(); }
        Word { symbols: self.symbols[start..].to_vec() }
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() { write!(f, "ε") }
        else {
            let parts: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
            write!(f, "{}", parts.join("·"))
        }
    }
}

/// Sub-distribution over outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubDistribution {
    pub weights: HashMap<String, f64>,
}

impl SubDistribution {
    pub fn new() -> Self { Self { weights: HashMap::new() } }
    pub fn singleton(key: String, prob: f64) -> Self {
        let mut w = HashMap::new();
        w.insert(key, prob);
        Self { weights: w }
    }
    pub fn from_map(weights: HashMap<String, f64>) -> Self { Self { weights } }

    pub fn total_mass(&self) -> f64 { self.weights.values().sum() }

    pub fn get(&self, key: &str) -> f64 {
        self.weights.get(key).copied().unwrap_or(0.0)
    }

    pub fn set(&mut self, key: String, prob: f64) {
        if prob > 1e-15 { self.weights.insert(key, prob); }
        else { self.weights.remove(&key); }
    }

    pub fn normalize(&mut self) {
        let total = self.total_mass();
        if total > 1e-15 {
            for v in self.weights.values_mut() {
                *v /= total;
            }
        }
    }

    pub fn tv_distance(&self, other: &SubDistribution) -> f64 {
        let mut all_keys: HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());
        let mut dist = 0.0;
        for key in all_keys {
            dist += (self.get(key) - other.get(key)).abs();
        }
        dist / 2.0
    }

    pub fn linf_distance(&self, other: &SubDistribution) -> f64 {
        let mut all_keys: HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());
        let mut max_diff = 0.0f64;
        for key in all_keys {
            max_diff = max_diff.max((self.get(key) - other.get(key)).abs());
        }
        max_diff
    }

    pub fn support(&self) -> Vec<&String> {
        self.weights.keys().filter(|k| self.weights[*k] > 1e-15).collect()
    }

    pub fn support_size(&self) -> usize {
        self.support().len()
    }

    pub fn merge(&self, other: &SubDistribution, alpha: f64) -> SubDistribution {
        let mut result = SubDistribution::new();
        let mut all_keys: HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());
        for key in all_keys {
            let v = alpha * self.get(key) + (1.0 - alpha) * other.get(key);
            result.set(key.clone(), v);
        }
        result
    }
}

impl Default for SubDistribution {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Hypothesis state
// ---------------------------------------------------------------------------

/// Unique identifier for a hypothesis state.
pub type HypothesisStateId = usize;

/// A state in the hypothesis automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisState {
    /// State identifier
    pub id: HypothesisStateId,
    /// Access string: the word that reaches this state from initial
    pub access_string: Word,
    /// Row signature: the observation table row for this state
    pub signature: Vec<SubDistribution>,
    /// Whether this is an accepting state
    pub is_accepting: bool,
    /// Output distribution at this state
    pub output: SubDistribution,
    /// Optional label
    pub label: Option<String>,
}

impl HypothesisState {
    pub fn new(id: HypothesisStateId, access_string: Word, signature: Vec<SubDistribution>) -> Self {
        let output = if signature.is_empty() {
            SubDistribution::new()
        } else {
            signature[0].clone() // first column (epsilon suffix) is the output
        };
        let is_accepting = output.total_mass() > 0.5;

        Self {
            id,
            access_string,
            signature,
            is_accepting,
            output,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Compare signatures with statistical tolerance.
    pub fn signature_equivalent(&self, other: &HypothesisState, tolerance: f64) -> bool {
        if self.signature.len() != other.signature.len() {
            return false;
        }
        for (a, b) in self.signature.iter().zip(other.signature.iter()) {
            if a.tv_distance(b) > tolerance {
                return false;
            }
        }
        true
    }

    /// Compute the maximum TV-distance between corresponding signature entries.
    pub fn signature_distance(&self, other: &HypothesisState) -> f64 {
        if self.signature.len() != other.signature.len() {
            return f64::INFINITY;
        }
        let mut max_dist = 0.0f64;
        for (a, b) in self.signature.iter().zip(other.signature.iter()) {
            max_dist = max_dist.max(a.tv_distance(b));
        }
        max_dist
    }
}

impl fmt::Display for HypothesisState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q{}[{}]", self.id, self.access_string)
    }
}

// ---------------------------------------------------------------------------
// Hypothesis transition
// ---------------------------------------------------------------------------

/// A transition in the hypothesis automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTransition {
    /// Source state
    pub from: HypothesisStateId,
    /// Input symbol
    pub symbol: Symbol,
    /// Target distribution over states
    pub target_distribution: SubDistribution,
    /// Deterministic target (most likely state)
    pub deterministic_target: HypothesisStateId,
    /// Confidence in this transition
    pub confidence: f64,
}

impl HypothesisTransition {
    pub fn new(
        from: HypothesisStateId,
        symbol: Symbol,
        target_distribution: SubDistribution,
        deterministic_target: HypothesisStateId,
        confidence: f64,
    ) -> Self {
        Self { from, symbol, target_distribution, deterministic_target, confidence }
    }

    /// Create a deterministic transition.
    pub fn deterministic(from: HypothesisStateId, symbol: Symbol, to: HypothesisStateId) -> Self {
        let mut dist = SubDistribution::new();
        dist.set(format!("q{}", to), 1.0);
        Self {
            from,
            symbol,
            target_distribution: dist,
            deterministic_target: to,
            confidence: 1.0,
        }
    }

    pub fn is_deterministic(&self) -> bool {
        self.target_distribution.support_size() <= 1
    }
}

impl fmt::Display for HypothesisTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q{} --{}-> q{} (conf={:.3})",
            self.from, self.symbol, self.deterministic_target, self.confidence)
    }
}

// ---------------------------------------------------------------------------
// HypothesisAutomaton
// ---------------------------------------------------------------------------

/// The hypothesis automaton — the current conjecture about the target system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisAutomaton {
    /// Unique identifier for this hypothesis
    pub id: String,
    /// States
    pub states: Vec<HypothesisState>,
    /// Transitions indexed by (from_state, symbol)
    pub transitions: HashMap<(HypothesisStateId, String), HypothesisTransition>,
    /// Initial state
    pub initial_state: HypothesisStateId,
    /// Alphabet
    pub alphabet: Vec<Symbol>,
    /// Iteration number when this hypothesis was created
    pub iteration: usize,
    /// Version counter
    pub version: usize,
    /// Tolerance used for construction
    pub tolerance: f64,
    /// State index by access string
    access_string_index: HashMap<Word, HypothesisStateId>,
}

impl HypothesisAutomaton {
    /// Create an empty hypothesis.
    pub fn new(alphabet: Vec<Symbol>, tolerance: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            states: Vec::new(),
            transitions: HashMap::new(),
            initial_state: 0,
            alphabet,
            iteration: 0,
            version: 0,
            tolerance,
            access_string_index: HashMap::new(),
        }
    }

    /// Build a hypothesis from a closed, consistent observation table.
    ///
    /// The observation table provides:
    ///   - S: set of access strings (state-identifying prefixes)
    ///   - E: set of suffixes (columns)
    ///   - T: table entries T(s·e) for each s ∈ S, e ∈ E
    ///
    /// We construct one state per equivalence class of rows in S,
    /// and compute transitions from the table structure.
    pub fn from_observation_table(
        access_strings: &[Word],
        suffixes: &[Word],
        table: &HashMap<Word, Vec<SubDistribution>>,
        alphabet: &[Symbol],
        tolerance: f64,
        iteration: usize,
    ) -> Self {
        let mut hypothesis = Self::new(alphabet.to_vec(), tolerance);
        hypothesis.iteration = iteration;

        if access_strings.is_empty() {
            return hypothesis;
        }

        // Step 1: identify distinct states (equivalence classes of rows)
        let mut representatives: Vec<Word> = Vec::new();
        let mut state_map: HashMap<Word, HypothesisStateId> = HashMap::new();

        for s in access_strings {
            let row = match table.get(s) {
                Some(r) => r.clone(),
                None => vec![SubDistribution::new(); suffixes.len()],
            };

            // Check if this row is equivalent to an existing representative
            let mut found_match = false;
            for (rep_idx, rep) in representatives.iter().enumerate() {
                let rep_row = table.get(rep).cloned()
                    .unwrap_or_else(|| vec![SubDistribution::new(); suffixes.len()]);

                if rows_equivalent(&row, &rep_row, tolerance) {
                    state_map.insert(s.clone(), rep_idx);
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                let new_id = representatives.len();
                representatives.push(s.clone());
                state_map.insert(s.clone(), new_id);
            }
        }

        // Step 2: create states
        for (id, rep) in representatives.iter().enumerate() {
            let row = table.get(rep).cloned()
                .unwrap_or_else(|| vec![SubDistribution::new(); suffixes.len()]);

            let state = HypothesisState::new(id, rep.clone(), row);
            hypothesis.states.push(state);
            hypothesis.access_string_index.insert(rep.clone(), id);
        }

        // Step 3: compute transitions
        for (id, rep) in representatives.iter().enumerate() {
            for sym in alphabet {
                let extended = rep.concat(&Word::singleton(sym.clone()));

                // Find which state the extended access string maps to
                let target_id = if let Some(&sid) = state_map.get(&extended) {
                    sid
                } else {
                    // Find the closest matching state based on row similarity
                    let ext_row = table.get(&extended).cloned()
                        .unwrap_or_else(|| vec![SubDistribution::new(); suffixes.len()]);

                    find_closest_state(&hypothesis.states, &ext_row, tolerance)
                };

                // Compute transition distribution
                let target_dist = compute_transition_distribution(
                    table, rep, sym, suffixes, &representatives, tolerance,
                );

                let confidence = compute_transition_confidence(
                    table, rep, sym, suffixes,
                );

                let transition = HypothesisTransition::new(
                    id, sym.clone(), target_dist, target_id, confidence,
                );

                hypothesis.transitions.insert((id, sym.0.clone()), transition);
            }
        }

        // Initial state is the one with empty access string
        hypothesis.initial_state = state_map.get(&Word::empty()).copied().unwrap_or(0);
        hypothesis.version += 1;

        hypothesis
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Number of transitions.
    pub fn num_transitions(&self) -> usize {
        self.transitions.len()
    }

    /// Get a state by ID.
    pub fn state(&self, id: HypothesisStateId) -> Option<&HypothesisState> {
        self.states.get(id)
    }

    /// Get the initial state.
    pub fn initial(&self) -> Option<&HypothesisState> {
        self.states.get(self.initial_state)
    }

    /// Get transition for a given state and symbol.
    pub fn transition(&self, from: HypothesisStateId, sym: &Symbol) -> Option<&HypothesisTransition> {
        self.transitions.get(&(from, sym.0.clone()))
    }

    /// Run the hypothesis on a word, returning the final state.
    pub fn run(&self, word: &Word) -> Option<HypothesisStateId> {
        let mut current = self.initial_state;
        for sym in &word.symbols {
            if let Some(trans) = self.transition(current, sym) {
                current = trans.deterministic_target;
            } else {
                return None;
            }
        }
        Some(current)
    }

    /// Run the hypothesis on a word, returning the output distribution.
    pub fn output_for(&self, word: &Word) -> SubDistribution {
        match self.run(word) {
            Some(state_id) => {
                self.states.get(state_id)
                    .map(|s| s.output.clone())
                    .unwrap_or_default()
            }
            None => SubDistribution::new(),
        }
    }

    /// Get the trace (sequence of states) for a word.
    pub fn trace(&self, word: &Word) -> Vec<HypothesisStateId> {
        let mut trace = vec![self.initial_state];
        let mut current = self.initial_state;
        for sym in &word.symbols {
            if let Some(trans) = self.transition(current, sym) {
                current = trans.deterministic_target;
                trace.push(current);
            } else {
                break;
            }
        }
        trace
    }

    /// Find state by access string.
    pub fn state_for_access_string(&self, word: &Word) -> Option<HypothesisStateId> {
        self.access_string_index.get(word).copied()
    }

    /// Check well-formedness of the hypothesis.
    pub fn validate(&self) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Check initial state exists
        if self.initial_state >= self.states.len() {
            issues.push(ValidationIssue::InvalidInitialState(self.initial_state));
        }

        // Check all transitions reference valid states
        for ((from, sym_str), trans) in &self.transitions {
            if *from >= self.states.len() {
                issues.push(ValidationIssue::InvalidSourceState(*from, sym_str.clone()));
            }
            if trans.deterministic_target >= self.states.len() {
                issues.push(ValidationIssue::InvalidTargetState(
                    trans.deterministic_target,
                    *from,
                    sym_str.clone(),
                ));
            }
        }

        // Check completeness: every state should have a transition for every symbol
        for state in &self.states {
            for sym in &self.alphabet {
                if !self.transitions.contains_key(&(state.id, sym.0.clone())) {
                    issues.push(ValidationIssue::MissingTransition(state.id, sym.clone()));
                }
            }
        }

        // Check state signatures have consistent lengths
        if let Some(first) = self.states.first() {
            let sig_len = first.signature.len();
            for state in &self.states {
                if state.signature.len() != sig_len {
                    issues.push(ValidationIssue::InconsistentSignatureLength(
                        state.id,
                        state.signature.len(),
                        sig_len,
                    ));
                }
            }
        }

        // Check transition distributions sum to ≤ 1
        for ((from, sym_str), trans) in &self.transitions {
            let mass = trans.target_distribution.total_mass();
            if mass > 1.0 + 1e-6 {
                issues.push(ValidationIssue::InvalidTransitionDistribution(
                    *from,
                    sym_str.clone(),
                    mass,
                ));
            }
        }

        issues
    }

    /// Is the hypothesis well-formed?
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Refine the hypothesis with a new suffix (from counter-example decomposition).
    ///
    /// This adds a column to the observation table and may split states.
    pub fn refine_with_suffix(
        &mut self,
        new_suffix: &Word,
        table: &HashMap<Word, Vec<SubDistribution>>,
        suffixes: &[Word],
        alphabet: &[Symbol],
    ) {
        // Rebuild the hypothesis from the updated table
        let access_strings: Vec<Word> = self.states.iter()
            .map(|s| s.access_string.clone())
            .collect();

        let new_hypothesis = Self::from_observation_table(
            &access_strings,
            suffixes,
            table,
            alphabet,
            self.tolerance,
            self.iteration + 1,
        );

        // Update self
        self.states = new_hypothesis.states;
        self.transitions = new_hypothesis.transitions;
        self.initial_state = new_hypothesis.initial_state;
        self.access_string_index = new_hypothesis.access_string_index;
        self.version += 1;
        self.iteration += 1;
    }

    /// Compare with another hypothesis: which states were added/removed/changed?
    pub fn diff(&self, other: &HypothesisAutomaton) -> HypothesisDiff {
        let self_access: HashSet<&Word> = self.states.iter()
            .map(|s| &s.access_string)
            .collect();
        let other_access: HashSet<&Word> = other.states.iter()
            .map(|s| &s.access_string)
            .collect();

        let added_states: Vec<Word> = other_access.difference(&self_access)
            .map(|w| (*w).clone())
            .collect();
        let removed_states: Vec<Word> = self_access.difference(&other_access)
            .map(|w| (*w).clone())
            .collect();
        let common_states: Vec<Word> = self_access.intersection(&other_access)
            .map(|w| (*w).clone())
            .collect();

        let mut changed_states = Vec::new();
        for w in &common_states {
            let self_id = self.access_string_index.get(w);
            let other_id = other.access_string_index.get(w);
            if let (Some(&sid), Some(&oid)) = (self_id, other_id) {
                if let (Some(ss), Some(os)) = (self.states.get(sid), other.states.get(oid)) {
                    if !ss.signature_equivalent(os, self.tolerance) {
                        changed_states.push(w.clone());
                    }
                }
            }
        }

        // Count transition changes
        let mut transition_changes = 0;
        for ((from, sym), trans) in &self.transitions {
            if let Some(other_trans) = other.transitions.get(&(*from, sym.clone())) {
                if trans.deterministic_target != other_trans.deterministic_target {
                    transition_changes += 1;
                }
            } else {
                transition_changes += 1;
            }
        }
        for ((from, sym), _) in &other.transitions {
            if !self.transitions.contains_key(&(*from, sym.clone())) {
                transition_changes += 1;
            }
        }

        HypothesisDiff {
            added_states,
            removed_states,
            changed_states,
            transition_changes,
            old_state_count: self.num_states(),
            new_state_count: other.num_states(),
        }
    }

    /// Export as an adjacency list representation.
    pub fn to_adjacency_list(&self) -> HashMap<HypothesisStateId, Vec<(Symbol, HypothesisStateId)>> {
        let mut adj: HashMap<HypothesisStateId, Vec<(Symbol, HypothesisStateId)>> = HashMap::new();
        for state in &self.states {
            adj.insert(state.id, Vec::new());
        }
        for ((from, _), trans) in &self.transitions {
            adj.entry(*from)
                .or_default()
                .push((trans.symbol.clone(), trans.deterministic_target));
        }
        adj
    }

    /// Export as a transition matrix (state × state) for each symbol.
    pub fn to_transition_matrices(&self) -> HashMap<String, Vec<Vec<f64>>> {
        let n = self.states.len();
        let mut matrices: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

        for sym in &self.alphabet {
            let mut mat = vec![vec![0.0; n]; n];
            for state in &self.states {
                if let Some(trans) = self.transition(state.id, sym) {
                    // Use the target distribution
                    for (key, &prob) in &trans.target_distribution.weights {
                        // Parse state id from key like "q3"
                        if let Some(target_id) = key.strip_prefix('q')
                            .and_then(|s| s.parse::<usize>().ok())
                        {
                            if target_id < n {
                                mat[state.id][target_id] = prob;
                            }
                        }
                    }
                    // Also ensure the deterministic target has weight
                    if trans.deterministic_target < n && trans.is_deterministic() {
                        mat[state.id][trans.deterministic_target] = 1.0;
                    }
                }
            }
            matrices.insert(sym.0.clone(), mat);
        }

        matrices
    }

    /// Compute minimum transition confidence across all transitions.
    pub fn min_confidence(&self) -> f64 {
        self.transitions.values()
            .map(|t| t.confidence)
            .fold(f64::INFINITY, f64::min)
    }

    /// Compute average transition confidence.
    pub fn avg_confidence(&self) -> f64 {
        if self.transitions.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.transitions.values().map(|t| t.confidence).sum();
        sum / self.transitions.len() as f64
    }

    /// Get all reachable states from the initial state.
    pub fn reachable_states(&self) -> HashSet<HypothesisStateId> {
        let mut reachable = HashSet::new();
        let mut stack = vec![self.initial_state];

        while let Some(state) = stack.pop() {
            if reachable.insert(state) {
                for sym in &self.alphabet {
                    if let Some(trans) = self.transition(state, sym) {
                        stack.push(trans.deterministic_target);
                    }
                }
            }
        }

        reachable
    }

    /// Remove unreachable states.
    pub fn minimize(&mut self) {
        let reachable = self.reachable_states();

        if reachable.len() == self.states.len() {
            return; // already minimal
        }

        // Build mapping from old IDs to new IDs
        let mut id_map: HashMap<HypothesisStateId, HypothesisStateId> = HashMap::new();
        let mut new_states = Vec::new();

        for state in &self.states {
            if reachable.contains(&state.id) {
                let new_id = new_states.len();
                id_map.insert(state.id, new_id);
                let mut new_state = state.clone();
                new_state.id = new_id;
                new_states.push(new_state);
            }
        }

        // Rebuild transitions
        let mut new_transitions = HashMap::new();
        for ((from, sym), trans) in &self.transitions {
            if let (Some(&new_from), Some(&new_to)) = (
                id_map.get(from),
                id_map.get(&trans.deterministic_target),
            ) {
                let mut new_trans = trans.clone();
                new_trans.from = new_from;
                new_trans.deterministic_target = new_to;
                new_transitions.insert((new_from, sym.clone()), new_trans);
            }
        }

        // Rebuild index
        let mut new_index = HashMap::new();
        for state in &new_states {
            new_index.insert(state.access_string.clone(), state.id);
        }

        self.states = new_states;
        self.transitions = new_transitions;
        self.access_string_index = new_index;
        self.initial_state = id_map.get(&self.initial_state).copied().unwrap_or(0);
        self.version += 1;
    }
}

impl fmt::Display for HypothesisAutomaton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hypothesis[v{}, {} states, {} transitions, init=q{}, confidence={:.3}]",
            self.version,
            self.num_states(),
            self.num_transitions(),
            self.initial_state,
            self.avg_confidence(),
        )
    }
}

// ---------------------------------------------------------------------------
// Validation issues
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ValidationIssue {
    InvalidInitialState(HypothesisStateId),
    InvalidSourceState(HypothesisStateId, String),
    InvalidTargetState(HypothesisStateId, HypothesisStateId, String),
    MissingTransition(HypothesisStateId, Symbol),
    InconsistentSignatureLength(HypothesisStateId, usize, usize),
    InvalidTransitionDistribution(HypothesisStateId, String, f64),
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInitialState(id) =>
                write!(f, "Invalid initial state: q{}", id),
            Self::InvalidSourceState(id, sym) =>
                write!(f, "Invalid source state q{} on symbol {}", id, sym),
            Self::InvalidTargetState(target, from, sym) =>
                write!(f, "Invalid target state q{} from q{} on {}", target, from, sym),
            Self::MissingTransition(id, sym) =>
                write!(f, "Missing transition from q{} on {}", id, sym),
            Self::InconsistentSignatureLength(id, got, expected) =>
                write!(f, "State q{} has signature length {}, expected {}", id, got, expected),
            Self::InvalidTransitionDistribution(id, sym, mass) =>
                write!(f, "Transition from q{} on {} has mass {:.4} > 1", id, sym, mass),
        }
    }
}

// ---------------------------------------------------------------------------
// Hypothesis diff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisDiff {
    pub added_states: Vec<Word>,
    pub removed_states: Vec<Word>,
    pub changed_states: Vec<Word>,
    pub transition_changes: usize,
    pub old_state_count: usize,
    pub new_state_count: usize,
}

impl HypothesisDiff {
    pub fn is_empty(&self) -> bool {
        self.added_states.is_empty()
            && self.removed_states.is_empty()
            && self.changed_states.is_empty()
            && self.transition_changes == 0
    }

    pub fn total_changes(&self) -> usize {
        self.added_states.len()
            + self.removed_states.len()
            + self.changed_states.len()
            + self.transition_changes
    }
}

impl fmt::Display for HypothesisDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Diff(+{} states, -{} states, ~{} states, {} trans changes, {} → {} states)",
            self.added_states.len(),
            self.removed_states.len(),
            self.changed_states.len(),
            self.transition_changes,
            self.old_state_count,
            self.new_state_count,
        )
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Check if two rows are equivalent within tolerance.
fn rows_equivalent(row1: &[SubDistribution], row2: &[SubDistribution], tolerance: f64) -> bool {
    if row1.len() != row2.len() {
        return false;
    }
    for (a, b) in row1.iter().zip(row2.iter()) {
        if a.tv_distance(b) > tolerance {
            return false;
        }
    }
    true
}

/// Find the closest matching state for a given row.
fn find_closest_state(
    states: &[HypothesisState],
    row: &[SubDistribution],
    _tolerance: f64,
) -> HypothesisStateId {
    if states.is_empty() {
        return 0;
    }

    let mut best_id = 0;
    let mut best_dist = f64::INFINITY;

    for state in states {
        if state.signature.len() != row.len() {
            continue;
        }
        let mut max_d = 0.0f64;
        for (a, b) in state.signature.iter().zip(row.iter()) {
            max_d = max_d.max(a.tv_distance(b));
        }
        if max_d < best_dist {
            best_dist = max_d;
            best_id = state.id;
        }
    }

    best_id
}

/// Compute the transition distribution from observation table data.
fn compute_transition_distribution(
    table: &HashMap<Word, Vec<SubDistribution>>,
    access_string: &Word,
    symbol: &Symbol,
    suffixes: &[Word],
    representatives: &[Word],
    tolerance: f64,
) -> SubDistribution {
    let extended = access_string.concat(&Word::singleton(symbol.clone()));
    let ext_row = match table.get(&extended) {
        Some(r) => r,
        None => return SubDistribution::singleton(format!("q0"), 1.0),
    };

    // Find matching representative
    let mut dist = SubDistribution::new();
    let mut best_match = 0;
    let mut best_distance = f64::INFINITY;

    for (i, rep) in representatives.iter().enumerate() {
        let rep_row = match table.get(rep) {
            Some(r) => r,
            None => continue,
        };

        if rep_row.len() != ext_row.len() {
            continue;
        }

        let mut max_d = 0.0f64;
        for (a, b) in rep_row.iter().zip(ext_row.iter()) {
            max_d = max_d.max(a.tv_distance(b));
        }

        if max_d < best_distance {
            best_distance = max_d;
            best_match = i;
        }
    }

    dist.set(format!("q{}", best_match), 1.0);
    dist
}

/// Compute confidence in a transition.
fn compute_transition_confidence(
    table: &HashMap<Word, Vec<SubDistribution>>,
    access_string: &Word,
    symbol: &Symbol,
    suffixes: &[Word],
) -> f64 {
    let extended = access_string.concat(&Word::singleton(symbol.clone()));
    match table.get(&extended) {
        Some(row) => {
            // Confidence based on total mass in the row entries
            if row.is_empty() {
                return 0.0;
            }
            let avg_mass: f64 = row.iter().map(|d| d.total_mass()).sum::<f64>() / row.len() as f64;
            avg_mass.min(1.0)
        }
        None => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Advanced hypothesis analysis and operations
// ---------------------------------------------------------------------------

/// Comparison result between two hypothesis automata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisComparison {
    /// Number of states in first hypothesis
    pub states_a: usize,
    /// Number of states in second hypothesis
    pub states_b: usize,
    /// Number of transitions in first
    pub transitions_a: usize,
    /// Number of transitions in second
    pub transitions_b: usize,
    /// Number of matched state pairs (bisimilar states)
    pub matched_pairs: usize,
    /// Maximum transition probability difference across matched states
    pub max_transition_diff: f64,
    /// Average transition probability difference
    pub avg_transition_diff: f64,
    /// Whether the hypotheses are structurally equivalent
    pub structurally_equivalent: bool,
}

impl HypothesisComparison {
    /// Compare two hypotheses.
    pub fn compare(h1: &HypothesisAutomaton, h2: &HypothesisAutomaton) -> Self {
        let states_a = h1.states.len();
        let states_b = h2.states.len();
        let transitions_a = h1.transitions.len();
        let transitions_b = h2.transitions.len();

        let mut matched_pairs = 0;
        let mut max_diff = 0.0f64;
        let mut total_diff = 0.0;
        let mut diff_count = 0;

        // Simple greedy matching: match states by access string similarity
        let mut matched_b: HashSet<usize> = HashSet::new();
        for (i, sa) in h1.states.iter().enumerate() {
            let mut best_match = None;
            let mut best_dist = f64::INFINITY;

            for (j, sb) in h2.states.iter().enumerate() {
                if matched_b.contains(&j) {
                    continue;
                }
                // Compare output distributions
                let dist = sa.output.tv_distance(&sb.output);
                if dist < best_dist {
                    best_dist = dist;
                    best_match = Some(j);
                }
            }

            if let Some(j) = best_match {
                if best_dist < 0.5 {
                    matched_pairs += 1;
                    matched_b.insert(j);
                    max_diff = max_diff.max(best_dist);
                    total_diff += best_dist;
                    diff_count += 1;
                }
            }
        }

        let avg_diff = if diff_count > 0 {
            total_diff / diff_count as f64
        } else {
            0.0
        };

        let structurally_equivalent = states_a == states_b
            && matched_pairs == states_a
            && max_diff < 0.01;

        HypothesisComparison {
            states_a,
            states_b,
            transitions_a,
            transitions_b,
            matched_pairs,
            max_transition_diff: max_diff,
            avg_transition_diff: avg_diff,
            structurally_equivalent,
        }
    }

    pub fn similarity_score(&self) -> f64 {
        if self.states_a == 0 && self.states_b == 0 {
            return 1.0;
        }
        let max_states = self.states_a.max(self.states_b) as f64;
        let match_ratio = self.matched_pairs as f64 / max_states;
        let dist_penalty = 1.0 - self.avg_transition_diff;
        (match_ratio * dist_penalty).clamp(0.0, 1.0)
    }
}

impl fmt::Display for HypothesisComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HypComp({}↔{} states, matched={}, max_diff={:.4}, struct_eq={})",
            self.states_a, self.states_b,
            self.matched_pairs,
            self.max_transition_diff,
            self.structurally_equivalent,
        )
    }
}

/// Hypothesis history tracker for monitoring convergence.
#[derive(Debug, Clone)]
pub struct HypothesisHistory {
    pub snapshots: Vec<HypothesisSnapshot>,
    pub comparisons: Vec<HypothesisComparison>,
    pub max_history: usize,
}

/// A lightweight snapshot of a hypothesis at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisSnapshot {
    pub iteration: usize,
    pub num_states: usize,
    pub num_transitions: usize,
    pub avg_output_entropy: f64,
    pub timestamp: u64,
}

impl HypothesisHistory {
    pub fn new(max_history: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            comparisons: Vec::new(),
            max_history,
        }
    }

    pub fn record(&mut self, hypothesis: &HypothesisAutomaton, iteration: usize) {
        let avg_entropy = if hypothesis.states.is_empty() {
            0.0
        } else {
            let total: f64 = hypothesis
                .states
                .iter()
                .map(|s| {
                    let mass = s.output.total_mass();
                    if mass < 1e-15 {
                        return 0.0;
                    }
                    let mut h = 0.0;
                    for &v in s.output.weights.values() {
                        let p = v / mass;
                        if p > 1e-15 {
                            h -= p * p.ln();
                        }
                    }
                    h
                })
                .sum();
            total / hypothesis.states.len() as f64
        };

        let snapshot = HypothesisSnapshot {
            iteration,
            num_states: hypothesis.states.len(),
            num_transitions: hypothesis.transitions.len(),
            avg_output_entropy: avg_entropy,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        self.snapshots.push(snapshot);

        // Keep bounded history
        if self.snapshots.len() > self.max_history {
            self.snapshots.remove(0);
        }
    }

    /// Check if hypothesis has stabilized (same structure for last n iterations).
    pub fn is_stable(&self, window: usize) -> bool {
        if self.snapshots.len() < window {
            return false;
        }
        let recent = &self.snapshots[self.snapshots.len() - window..];
        let first = &recent[0];
        recent
            .iter()
            .all(|s| s.num_states == first.num_states && s.num_transitions == first.num_transitions)
    }

    /// Get the state count trend over time.
    pub fn state_count_trend(&self) -> Vec<(usize, usize)> {
        self.snapshots
            .iter()
            .map(|s| (s.iteration, s.num_states))
            .collect()
    }

    /// Get the latest snapshot.
    pub fn latest(&self) -> Option<&HypothesisSnapshot> {
        self.snapshots.last()
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }
}

/// Automaton product construction for hypothesis comparison.
pub fn product_automaton(
    h1: &HypothesisAutomaton,
    h2: &HypothesisAutomaton,
) -> HypothesisAutomaton {
    let mut product = HypothesisAutomaton::new(h1.alphabet.clone(), h1.tolerance);

    // Product states: (s1, s2) for each pair
    for (i, s1) in h1.states.iter().enumerate() {
        for (j, s2) in h2.states.iter().enumerate() {
            // Product output distribution: element-wise minimum (for safety)
            let mut output = SubDistribution::new();
            let all_keys: HashSet<&String> = s1
                .output
                .weights
                .keys()
                .chain(s2.output.weights.keys())
                .collect();
            for key in all_keys {
                let v1 = s1.output.get(key);
                let v2 = s2.output.get(key);
                output.set(key.clone(), v1.min(v2));
            }

            let state = HypothesisState {
                id: i * h2.states.len() + j,
                access_string: s1.access_string.concat(&s2.access_string),
                signature: Vec::new(),
                output: output,
                is_accepting: s1.is_accepting && s2.is_accepting,
                label: None,
            };
            product.states.push(state);
        }
    }

    // Set initial state
    if !h1.states.is_empty() && !h2.states.is_empty() {
        product.initial_state =
            h1.initial_state * h2.states.len()
                + h2.initial_state;
    }

    product
}

/// Minimize a hypothesis automaton by merging bisimilar states.
pub fn minimize_hypothesis(hypothesis: &HypothesisAutomaton, tolerance: f64) -> HypothesisAutomaton {
    if hypothesis.states.len() <= 1 {
        return hypothesis.clone();
    }

    // Partition refinement algorithm
    let n = hypothesis.states.len();
    let mut partition: Vec<usize> = vec![0; n];
    let mut num_classes = 1;

    // Initial partition by output distribution similarity
    for i in 1..n {
        let mut found_class = false;
        for j in 0..i {
            if partition[j] < num_classes {
                let dist = hypothesis.states[i]
                    .output
                    .tv_distance(&hypothesis.states[j].output);
                if dist <= tolerance {
                    partition[i] = partition[j];
                    found_class = true;
                    break;
                }
            }
        }
        if !found_class {
            partition[i] = num_classes;
            num_classes += 1;
        }
    }

    // Iteratively refine until stable
    let mut changed = true;
    let max_iterations = 100;
    let mut iteration = 0;
    while changed && iteration < max_iterations {
        changed = false;
        iteration += 1;

        for i in 0..n {
            for j in (i + 1)..n {
                if partition[i] != partition[j] {
                    continue;
                }

                // Check if transitions lead to the same partition classes
                let mut should_split = false;
                for ((from_id, sym), trans_i) in &hypothesis.transitions {
                    if *from_id != hypothesis.states[i].id {
                        continue;
                    }
                    // Find corresponding transition from j
                    let matching_trans = hypothesis.transitions.get(
                        &(hypothesis.states[j].id, sym.clone())
                    );

                    if let Some(trans_j) = matching_trans {
                        let target_i_class = hypothesis
                            .states
                            .iter()
                            .position(|s| s.id == trans_i.deterministic_target)
                            .map(|idx| partition[idx]);
                        let target_j_class = hypothesis
                            .states
                            .iter()
                            .position(|s| s.id == trans_j.deterministic_target)
                            .map(|idx| partition[idx]);

                        if target_i_class != target_j_class {
                            should_split = true;
                            break;
                        }

                        if trans_i.target_distribution.tv_distance(&trans_j.target_distribution) > tolerance {
                            should_split = true;
                            break;
                        }
                    } else {
                        should_split = true;
                        break;
                    }
                }

                if should_split {
                    partition[j] = num_classes;
                    num_classes += 1;
                    changed = true;
                }
            }
        }
    }

    // Build minimized automaton
    let mut minimized = HypothesisAutomaton::new(hypothesis.alphabet.clone(), hypothesis.tolerance);
    let mut class_to_state: HashMap<usize, usize> = HashMap::new();

    for class in 0..num_classes {
        let representative = (0..n).find(|&i| partition[i] == class);
        if let Some(rep_idx) = representative {
            let new_id = minimized.states.len();
            class_to_state.insert(class, new_id);
            let mut state = hypothesis.states[rep_idx].clone();
            state.id = new_id;
            minimized.states.push(state);
        }
    }

    // Remap transitions
    for ((from_id, sym), trans) in &hypothesis.transitions {
        let from_idx = hypothesis.states.iter().position(|s| s.id == *from_id);
        let to_idx = hypothesis.states.iter().position(|s| s.id == trans.deterministic_target);

        if let (Some(fi), Some(ti)) = (from_idx, to_idx) {
            let from_class = partition[fi];
            let to_class = partition[ti];

            if let (Some(&new_from), Some(&new_to)) =
                (class_to_state.get(&from_class), class_to_state.get(&to_class))
            {
                let new_key = (new_from, sym.clone());
                if !minimized.transitions.contains_key(&new_key) {
                    let mut new_trans = trans.clone();
                    new_trans.from = new_from;
                    new_trans.deterministic_target = new_to;
                    minimized.transitions.insert(new_key, new_trans);
                }
            }
        }
    }

    {
        let init = hypothesis.initial_state;
        let init_idx = hypothesis.states.iter().position(|s| s.id == init).unwrap_or(0);
        let init_class = partition[init_idx];
        minimized.initial_state = class_to_state.get(&init_class).copied().unwrap_or(0);
    }

    minimized
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_alphabet() -> Vec<Symbol> {
        vec![Symbol::new("a"), Symbol::new("b")]
    }

    fn make_simple_table() -> (Vec<Word>, Vec<Word>, HashMap<Word, Vec<SubDistribution>>) {
        let access_strings = vec![
            Word::empty(),
            Word::from_str_slice(&["a"]),
        ];
        let suffixes = vec![
            Word::empty(),
            Word::from_str_slice(&["b"]),
        ];

        let mut table: HashMap<Word, Vec<SubDistribution>> = HashMap::new();

        // State q0 (empty): output = {yes: 0.8}, on suffix "b" = {no: 0.7}
        table.insert(Word::empty(), vec![
            SubDistribution::singleton("yes".to_string(), 0.8),
            SubDistribution::singleton("no".to_string(), 0.7),
        ]);

        // State q1 ("a"): output = {no: 0.9}, on suffix "b" = {yes: 0.6}
        table.insert(Word::from_str_slice(&["a"]), vec![
            SubDistribution::singleton("no".to_string(), 0.9),
            SubDistribution::singleton("yes".to_string(), 0.6),
        ]);

        // Extensions
        table.insert(Word::from_str_slice(&["b"]), vec![
            SubDistribution::singleton("yes".to_string(), 0.75),
            SubDistribution::singleton("no".to_string(), 0.65),
        ]);
        table.insert(Word::from_str_slice(&["a", "a"]), vec![
            SubDistribution::singleton("no".to_string(), 0.85),
            SubDistribution::singleton("yes".to_string(), 0.55),
        ]);
        table.insert(Word::from_str_slice(&["a", "b"]), vec![
            SubDistribution::singleton("yes".to_string(), 0.82),
            SubDistribution::singleton("no".to_string(), 0.71),
        ]);

        (access_strings, suffixes, table)
    }

    #[test]
    fn test_hypothesis_from_table() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        assert_eq!(hyp.num_states(), 2);
        assert!(hyp.num_transitions() > 0);
        assert!(hyp.is_valid());
    }

    #[test]
    fn test_hypothesis_run() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        // Empty word should reach initial state
        let state = hyp.run(&Word::empty());
        assert_eq!(state, Some(hyp.initial_state));

        // "a" should reach some state
        let state = hyp.run(&Word::from_str_slice(&["a"]));
        assert!(state.is_some());
    }

    #[test]
    fn test_hypothesis_trace() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let trace = hyp.trace(&Word::from_str_slice(&["a", "b"]));
        assert!(trace.len() >= 1);
        assert_eq!(trace[0], hyp.initial_state);
    }

    #[test]
    fn test_hypothesis_output_for() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let output = hyp.output_for(&Word::empty());
        assert!(output.total_mass() > 0.0);
    }

    #[test]
    fn test_hypothesis_validation() {
        let hyp = HypothesisAutomaton::new(make_alphabet(), 0.1);
        // Empty hypothesis with no states should have initial state issue
        let issues = hyp.validate();
        // Initial state 0 doesn't exist
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_hypothesis_valid_automaton() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        assert!(hyp.is_valid());
    }

    #[test]
    fn test_hypothesis_diff_identical() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp1 = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );
        let hyp2 = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 1,
        );

        let diff = hyp1.diff(&hyp2);
        assert!(diff.added_states.is_empty());
        assert!(diff.removed_states.is_empty());
    }

    #[test]
    fn test_hypothesis_diff_with_change() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp1 = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        // Create a different table
        let mut table2 = table.clone();
        let mut more_strings = access_strings.clone();
        more_strings.push(Word::from_str_slice(&["b"]));
        // Ensure "b" row is very different
        table2.insert(Word::from_str_slice(&["b"]), vec![
            SubDistribution::singleton("z".to_string(), 0.1),
            SubDistribution::singleton("w".to_string(), 0.2),
        ]);

        let hyp2 = HypothesisAutomaton::from_observation_table(
            &more_strings, &suffixes, &table2, &alphabet, 0.1, 1,
        );

        let diff = hyp1.diff(&hyp2);
        assert!(diff.total_changes() > 0 || hyp2.num_states() != hyp1.num_states());
    }

    #[test]
    fn test_hypothesis_reachable_states() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let reachable = hyp.reachable_states();
        assert!(!reachable.is_empty());
        assert!(reachable.contains(&hyp.initial_state));
    }

    #[test]
    fn test_hypothesis_minimize() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let mut hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let original_states = hyp.num_states();
        hyp.minimize();
        // Should be same or fewer states
        assert!(hyp.num_states() <= original_states);
    }

    #[test]
    fn test_hypothesis_adjacency_list() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let adj = hyp.to_adjacency_list();
        assert_eq!(adj.len(), hyp.num_states());
    }

    #[test]
    fn test_hypothesis_transition_matrices() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let matrices = hyp.to_transition_matrices();
        assert_eq!(matrices.len(), alphabet.len());
        for (_, mat) in &matrices {
            assert_eq!(mat.len(), hyp.num_states());
        }
    }

    #[test]
    fn test_hypothesis_confidence() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let min_conf = hyp.min_confidence();
        let avg_conf = hyp.avg_confidence();
        assert!(min_conf >= 0.0);
        assert!(avg_conf >= 0.0);
    }

    #[test]
    fn test_hypothesis_state_equivalent() {
        let sig1 = vec![SubDistribution::singleton("a".to_string(), 0.5)];
        let sig2 = vec![SubDistribution::singleton("a".to_string(), 0.52)];
        let sig3 = vec![SubDistribution::singleton("b".to_string(), 0.9)];

        let s1 = HypothesisState::new(0, Word::empty(), sig1);
        let s2 = HypothesisState::new(1, Word::from_str_slice(&["a"]), sig2);
        let s3 = HypothesisState::new(2, Word::from_str_slice(&["b"]), sig3);

        assert!(s1.signature_equivalent(&s2, 0.1));
        assert!(!s1.signature_equivalent(&s3, 0.1));
    }

    #[test]
    fn test_hypothesis_state_distance() {
        let sig1 = vec![SubDistribution::singleton("a".to_string(), 0.5)];
        let sig2 = vec![SubDistribution::singleton("a".to_string(), 0.7)];

        let s1 = HypothesisState::new(0, Word::empty(), sig1);
        let s2 = HypothesisState::new(1, Word::from_str_slice(&["a"]), sig2);

        let dist = s1.signature_distance(&s2);
        assert!(dist > 0.0);
        assert!(dist < 1.0);
    }

    #[test]
    fn test_hypothesis_display() {
        let (access_strings, suffixes, table) = make_simple_table();
        let alphabet = make_alphabet();

        let hyp = HypothesisAutomaton::from_observation_table(
            &access_strings, &suffixes, &table, &alphabet, 0.1, 0,
        );

        let s = format!("{}", hyp);
        assert!(s.contains("Hypothesis"));
        assert!(s.contains("states"));
    }

    #[test]
    fn test_transition_deterministic() {
        let t = HypothesisTransition::deterministic(0, Symbol::new("a"), 1);
        assert!(t.is_deterministic());
        assert_eq!(t.deterministic_target, 1);
        assert_eq!(t.confidence, 1.0);
    }

    #[test]
    fn test_transition_display() {
        let t = HypothesisTransition::deterministic(0, Symbol::new("x"), 2);
        let s = format!("{}", t);
        assert!(s.contains("q0"));
        assert!(s.contains("q2"));
    }

    #[test]
    fn test_rows_equivalent() {
        let r1 = vec![SubDistribution::singleton("a".to_string(), 0.5)];
        let r2 = vec![SubDistribution::singleton("a".to_string(), 0.55)];
        let r3 = vec![SubDistribution::singleton("b".to_string(), 0.9)];

        assert!(rows_equivalent(&r1, &r2, 0.1));
        assert!(!rows_equivalent(&r1, &r3, 0.1));
    }

    #[test]
    fn test_hypothesis_diff_display() {
        let diff = HypothesisDiff {
            added_states: vec![Word::from_str_slice(&["x"])],
            removed_states: vec![],
            changed_states: vec![],
            transition_changes: 2,
            old_state_count: 3,
            new_state_count: 4,
        };

        let s = format!("{}", diff);
        assert!(s.contains("+1 states"));
        assert!(s.contains("3 → 4"));
    }

    #[test]
    fn test_validation_issue_display() {
        let issue = ValidationIssue::MissingTransition(0, Symbol::new("a"));
        let s = format!("{}", issue);
        assert!(s.contains("Missing"));
    }

    #[test]
    fn test_sub_distribution_merge() {
        let d1 = SubDistribution::singleton("a".to_string(), 0.6);
        let d2 = SubDistribution::singleton("a".to_string(), 0.4);

        let merged = d1.merge(&d2, 0.5);
        assert!((merged.get("a") - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sub_distribution_normalize() {
        let mut d = SubDistribution::from_map(
            [("a".to_string(), 2.0), ("b".to_string(), 3.0)]
                .into_iter().collect()
        );
        d.normalize();
        assert!((d.total_mass() - 1.0).abs() < 0.001);
        assert!((d.get("a") - 0.4).abs() < 0.001);
        assert!((d.get("b") - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_hypothesis_comparison_identical() {
        let h1 = make_simple_hypothesis();
        let cmp = HypothesisComparison::compare(&h1, &h1);
        assert!(cmp.structurally_equivalent);
        assert!((cmp.similarity_score() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hypothesis_comparison_different() {
        let h1 = make_simple_hypothesis();
        let mut h2 = make_simple_hypothesis();
        h2.states[0].output = SubDistribution::singleton("no".to_string(), 1.0);
        let cmp = HypothesisComparison::compare(&h1, &h2);
        assert!(cmp.max_transition_diff > 0.0);
    }

    #[test]
    fn test_hypothesis_comparison_display() {
        let h = make_simple_hypothesis();
        let cmp = HypothesisComparison::compare(&h, &h);
        let s = format!("{}", cmp);
        assert!(s.contains("HypComp"));
    }

    #[test]
    fn test_hypothesis_history() {
        let mut history = HypothesisHistory::new(10);
        let h = make_simple_hypothesis();

        for i in 0..5 {
            history.record(&h, i);
        }

        assert_eq!(history.len(), 5);
        assert!(!history.is_stable(3)); // Might not be stable at first check
        let latest = history.latest().unwrap();
        assert_eq!(latest.iteration, 4);
        assert_eq!(latest.num_states, 1);
    }

    #[test]
    fn test_hypothesis_history_stability() {
        let mut history = HypothesisHistory::new(10);
        let h = make_simple_hypothesis();

        // Record same hypothesis multiple times
        for i in 0..5 {
            history.record(&h, i);
        }

        assert!(history.is_stable(3));
        assert!(history.is_stable(5));
        assert!(!history.is_stable(6)); // Not enough data
    }

    #[test]
    fn test_hypothesis_history_state_count_trend() {
        let mut history = HypothesisHistory::new(10);
        let h = make_simple_hypothesis();
        history.record(&h, 0);
        history.record(&h, 1);

        let trend = history.state_count_trend();
        assert_eq!(trend.len(), 2);
        assert_eq!(trend[0], (0, 1));
        assert_eq!(trend[1], (1, 1));
    }

    #[test]
    fn test_minimize_hypothesis_single_state() {
        let h = make_simple_hypothesis();
        let minimized = minimize_hypothesis(&h, 0.1);
        assert_eq!(minimized.states.len(), 1);
    }

    #[test]
    fn test_minimize_hypothesis_merge_equivalent() {
        let mut h = HypothesisAutomaton::new(Vec::new(), 0.05);
        // Two states with same output
        h.states.push(HypothesisState {
            id: 0,
            access_string: Word::empty(),
            signature: Vec::new(),
            output: SubDistribution::singleton("yes".to_string(), 0.8),
            is_accepting: true,
            label: None,
        });
        h.states.push(HypothesisState {
            id: 1,
            access_string: Word::from_str_slice(&["a"]),
            signature: Vec::new(),
            output: SubDistribution::singleton("yes".to_string(), 0.81),
            is_accepting: true,
            label: None,
        });
        h.initial_state = 0;

        let minimized = minimize_hypothesis(&h, 0.05);
        assert_eq!(minimized.states.len(), 1); // Should merge
    }

    #[test]
    fn test_minimize_hypothesis_keep_different() {
        let mut h = HypothesisAutomaton::new(Vec::new(), 0.05);
        h.states.push(HypothesisState {
            id: 0,
            access_string: Word::empty(),
            signature: Vec::new(),
            output: SubDistribution::singleton("yes".to_string(), 0.9),
            is_accepting: true,
            label: None,
        });
        h.states.push(HypothesisState {
            id: 1,
            access_string: Word::from_str_slice(&["a"]),
            signature: Vec::new(),
            output: SubDistribution::singleton("no".to_string(), 0.9),
            is_accepting: false,
            label: None,
        });
        h.initial_state = 0;

        let minimized = minimize_hypothesis(&h, 0.05);
        assert_eq!(minimized.states.len(), 2);
    }

    #[test]
    fn test_product_automaton() {
        let h1 = make_simple_hypothesis();
        let h2 = make_simple_hypothesis();
        let product = product_automaton(&h1, &h2);
        assert_eq!(product.states.len(), 1); // 1 * 1 = 1
    }

    fn make_simple_hypothesis() -> HypothesisAutomaton {
        let (access_strings, suffixes, data) = make_simple_table();
        let alphabet = make_alphabet();
        HypothesisAutomaton::from_table(&access_strings, &suffixes, &data, &alphabet, 0.1)
    }
}
