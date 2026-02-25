//! Core weighted finite automaton (WFA) implementation.
//!
//! Provides a generic `WeightedFiniteAutomaton<S>` parameterized over any
//! semiring `S`.  Includes forward/backward weight computation, closure
//! operations (union, concatenation, Kleene star, intersection, complement,
//! reverse), determinization via weighted subset construction, epsilon
//! removal, trimming, basic regex‑to‑WFA conversion, DOT export, and
//! serde serialization.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use indexmap::IndexSet;
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::semiring::{
    BooleanSemiring, CountingSemiring, RealSemiring, Semiring, SemiringMatrix, StarSemiring,
    TropicalSemiring,
};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during WFA construction or manipulation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum WfaError {
    #[error("invalid state index {0} (automaton has {1} states)")]
    InvalidState(usize, usize),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("automaton is empty (zero states)")]
    EmptyAutomaton,

    #[error("invalid transition: {0}")]
    InvalidTransition(String),

    #[error("no accepting path found")]
    NoPath,

    #[error("alphabet mismatch between operands")]
    AlphabetMismatch,

    #[error("automaton is non‑deterministic")]
    NonDeterministic,

    #[error("invalid weight: {0}")]
    InvalidWeight(String),

    #[error("invalid symbol index {0} (alphabet size {1})")]
    InvalidSymbol(usize, usize),

    #[error("regex parse error: {0}")]
    RegexParse(String),

    #[error("operation not supported: {0}")]
    Unsupported(String),
}

pub type WfaResult<T> = Result<T, WfaError>;

// ---------------------------------------------------------------------------
// Symbol
// ---------------------------------------------------------------------------

/// An element of an alphabet.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Symbol {
    Char(char),
    Byte(u8),
    Token(String),
    Epsilon,
    Wildcard,
    Id(usize),
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Symbol::Char(c) => write!(f, "{}", c),
            Symbol::Byte(b) => write!(f, "0x{:02x}", b),
            Symbol::Token(t) => write!(f, "\"{}\"", t),
            Symbol::Epsilon => write!(f, "ε"),
            Symbol::Wildcard => write!(f, "*"),
            Symbol::Id(id) => write!(f, "#{}", id),
        }
    }
}

// ---------------------------------------------------------------------------
// Alphabet
// ---------------------------------------------------------------------------

/// Ordered set of symbols with a bijection to indices `0..n`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Alphabet {
    symbols: IndexSet<Symbol>,
    /// If true the alphabet contains an explicit epsilon column.  Epsilon is
    /// always stored at index 0 when present.
    has_epsilon: bool,
}

impl Alphabet {
    /// Create an empty alphabet.
    pub fn new() -> Self {
        Alphabet {
            symbols: IndexSet::new(),
            has_epsilon: false,
        }
    }

    /// Create an alphabet from a sequence of `char`s.
    pub fn from_chars(chars: &[char]) -> Self {
        let mut a = Self::new();
        for &c in chars {
            a.insert(Symbol::Char(c));
        }
        a
    }

    /// Create an alphabet from a sequence of string tokens.
    pub fn from_strings(tokens: &[&str]) -> Self {
        let mut a = Self::new();
        for t in tokens {
            a.insert(Symbol::Token(t.to_string()));
        }
        a
    }

    /// Create an alphabet from `Symbol::Id` values `0..n`.
    pub fn from_range(n: usize) -> Self {
        let mut a = Self::new();
        for i in 0..n {
            a.insert(Symbol::Id(i));
        }
        a
    }

    /// Create an alphabet that includes an explicit epsilon symbol at index 0.
    pub fn with_epsilon(mut self) -> Self {
        if !self.has_epsilon {
            // Shift existing symbols: insert epsilon at front.
            let old: Vec<Symbol> = self.symbols.into_iter().collect();
            self.symbols = IndexSet::new();
            self.symbols.insert(Symbol::Epsilon);
            for s in old {
                self.symbols.insert(s);
            }
            self.has_epsilon = true;
        }
        self
    }

    /// Insert a symbol.  Returns the index.
    pub fn insert(&mut self, sym: Symbol) -> usize {
        let (idx, _) = self.symbols.insert_full(sym);
        idx
    }

    /// Whether the alphabet contains the given symbol.
    pub fn contains(&self, sym: &Symbol) -> bool {
        self.symbols.contains(sym)
    }

    /// Return the index of a symbol, if present.
    pub fn index_of(&self, sym: &Symbol) -> Option<usize> {
        self.symbols.get_index_of(sym)
    }

    /// Return the symbol at a given index.
    pub fn symbol_at(&self, idx: usize) -> Option<&Symbol> {
        self.symbols.get_index(idx)
    }

    /// Number of symbols (including epsilon if present).
    pub fn size(&self) -> usize {
        self.symbols.len()
    }

    /// Iterate over `(index, &Symbol)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &Symbol)> {
        self.symbols.iter().enumerate()
    }

    /// Whether an explicit epsilon column is present.
    pub fn has_epsilon(&self) -> bool {
        self.has_epsilon
    }

    /// Index of the epsilon symbol, if present.
    pub fn epsilon_index(&self) -> Option<usize> {
        if self.has_epsilon {
            self.symbols.get_index_of(&Symbol::Epsilon)
        } else {
            None
        }
    }

    /// Union of two alphabets (preserving order of `self` first).
    pub fn union(&self, other: &Alphabet) -> Alphabet {
        let mut result = self.clone();
        for sym in other.symbols.iter() {
            result.insert(sym.clone());
        }
        result.has_epsilon = self.has_epsilon || other.has_epsilon;
        result
    }

    /// Intersection of two alphabets.
    pub fn intersection(&self, other: &Alphabet) -> Alphabet {
        let mut result = Alphabet::new();
        for sym in self.symbols.iter() {
            if other.contains(sym) {
                result.insert(sym.clone());
            }
        }
        result.has_epsilon = self.has_epsilon && other.has_epsilon;
        result
    }

    /// Build a mapping from indices in `self` to indices in `target`.
    /// Returns `None` for symbols not present in `target`.
    pub fn index_mapping(&self, target: &Alphabet) -> Vec<Option<usize>> {
        self.symbols
            .iter()
            .map(|sym| target.index_of(sym))
            .collect()
    }
}

impl Default for Alphabet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Alphabet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, sym) in self.symbols.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", sym)?;
        }
        write!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A single weighted transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition<S: Semiring> {
    pub from_state: usize,
    pub to_state: usize,
    /// Index into the alphabet.
    pub symbol: usize,
    pub weight: S,
}

impl<S: Semiring> Transition<S> {
    pub fn new(from: usize, to: usize, symbol: usize, weight: S) -> Self {
        Transition {
            from_state: from,
            to_state: to,
            symbol,
            weight,
        }
    }
}

impl<S: Semiring + fmt::Display> fmt::Display for Transition<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} --[sym={}, w={}]--> {})",
            self.from_state, self.symbol, self.weight, self.to_state
        )
    }
}

// ---------------------------------------------------------------------------
// Cached metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CachedMeta {
    transition_count: Option<usize>,
    is_deterministic: Option<bool>,
    is_trim: Option<bool>,
}

// ---------------------------------------------------------------------------
// WeightedFiniteAutomaton
// ---------------------------------------------------------------------------

/// A weighted finite automaton over a semiring `S`.
///
/// States are numbered `0..num_states`.  Transitions are stored as
/// `transitions[from_state][symbol_index] -> Vec<(to_state, weight)>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedFiniteAutomaton<S: Semiring> {
    pub num_states: usize,
    pub alphabet: Alphabet,
    /// Weight for each state being initial; `initial_weights[q]` is the
    /// weight of starting in state `q`.
    pub initial_weights: Vec<S>,
    /// Weight for each state being final.
    pub final_weights: Vec<S>,
    /// `transitions[q][a]` is a `Vec<(r, w)>` of transitions from state `q`
    /// reading symbol index `a` to state `r` with weight `w`.
    pub transitions: Vec<Vec<Vec<(usize, S)>>>,
    /// Optional state labels for display / debugging.
    #[serde(default)]
    pub state_labels: Vec<Option<String>>,
    #[serde(skip)]
    pub cache: CachedMeta,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    // ── basic constructors ─────────────────────────────────────────────

    /// Create a new WFA with `num_states` states and the given `alphabet`.
    /// All initial, final and transition weights are zero.
    pub fn new(num_states: usize, alphabet: Alphabet) -> Self {
        let alpha_size = alphabet.size();
        let transitions = vec![vec![Vec::new(); alpha_size]; num_states];
        WeightedFiniteAutomaton {
            num_states,
            alphabet,
            initial_weights: vec![S::zero(); num_states],
            final_weights: vec![S::zero(); num_states],
            transitions,
            state_labels: vec![None; num_states],
            cache: CachedMeta::default(),
        }
    }

    /// Create a new empty WFA (zero states) with the given alphabet.
    pub fn from_alphabet(alphabet: Alphabet) -> Self {
        Self::new(0, alphabet)
    }

    /// Create a new WFA pre‑allocated for `num_states` and `alphabet_size`.
    pub fn with_capacity(num_states: usize, alphabet_size: usize) -> Self {
        let alphabet = Alphabet::from_range(alphabet_size);
        Self::new(num_states, alphabet)
    }

    /// Build a WFA directly from component vectors.
    pub fn from_transitions(
        num_states: usize,
        alphabet: Alphabet,
        initial_weights: Vec<S>,
        final_weights: Vec<S>,
        trans: Vec<Transition<S>>,
    ) -> WfaResult<Self> {
        if initial_weights.len() != num_states {
            return Err(WfaError::DimensionMismatch {
                expected: num_states,
                got: initial_weights.len(),
            });
        }
        if final_weights.len() != num_states {
            return Err(WfaError::DimensionMismatch {
                expected: num_states,
                got: final_weights.len(),
            });
        }
        let mut wfa = Self::new(num_states, alphabet);
        wfa.initial_weights = initial_weights;
        wfa.final_weights = final_weights;
        for t in trans {
            wfa.add_transition(t.from_state, t.symbol, t.to_state, t.weight)?;
        }
        Ok(wfa)
    }

    /// WFA that recognises a single symbol with the given weight.
    /// Two states: 0 (initial, weight one) → 1 (final, weight one) on the
    /// given symbol with the given weight.
    pub fn single_symbol(symbol: Symbol, weight: S) -> Self {
        let mut alphabet = Alphabet::new();
        let idx = alphabet.insert(symbol);
        let mut wfa = Self::new(2, alphabet);
        wfa.initial_weights[0] = S::one();
        wfa.final_weights[1] = S::one();
        wfa.transitions[0][idx].push((1, weight));
        wfa
    }

    /// WFA that accepts only the empty string with the given weight.
    /// Single state that is both initial and final.
    pub fn epsilon_wfa(weight: S) -> Self {
        let alphabet = Alphabet::new();
        let mut wfa = Self::new(1, alphabet);
        wfa.initial_weights[0] = S::one();
        wfa.final_weights[0] = weight;
        wfa
    }

    /// WFA that accepts nothing (has no accepting paths).
    pub fn empty() -> Self {
        let alphabet = Alphabet::new();
        Self::new(0, alphabet)
    }

    /// WFA that accepts every string over the given alphabet with the
    /// given weight per symbol transition.
    pub fn universal(alphabet: Alphabet, weight: S) -> Self {
        let alpha_size = alphabet.size();
        let mut wfa = Self::new(1, alphabet);
        wfa.initial_weights[0] = S::one();
        wfa.final_weights[0] = S::one();
        for a in 0..alpha_size {
            wfa.transitions[0][a].push((0, weight.clone()));
        }
        wfa
    }

    // ── mutators ────────────────────────────────────────────────────────

    /// Set the initial weight of `state`.
    pub fn set_initial_weight(&mut self, state: usize, weight: S) {
        assert!(state < self.num_states, "state index out of bounds");
        self.initial_weights[state] = weight;
        self.invalidate_cache();
    }

    /// Set the final weight of `state`.
    pub fn set_final_weight(&mut self, state: usize, weight: S) {
        assert!(state < self.num_states, "state index out of bounds");
        self.final_weights[state] = weight;
        self.invalidate_cache();
    }

    /// Add a transition.
    pub fn add_transition(
        &mut self,
        from: usize,
        symbol: usize,
        to: usize,
        weight: S,
    ) -> WfaResult<()> {
        if from >= self.num_states {
            return Err(WfaError::InvalidState(from, self.num_states));
        }
        if to >= self.num_states {
            return Err(WfaError::InvalidState(to, self.num_states));
        }
        if symbol >= self.alphabet.size() {
            return Err(WfaError::InvalidSymbol(symbol, self.alphabet.size()));
        }
        self.transitions[from][symbol].push((to, weight));
        self.invalidate_cache();
        Ok(())
    }

    /// Add a new state and return its index.
    pub fn add_state(&mut self) -> usize {
        let idx = self.num_states;
        self.num_states += 1;
        self.initial_weights.push(S::zero());
        self.final_weights.push(S::zero());
        self.transitions
            .push(vec![Vec::new(); self.alphabet.size()]);
        self.state_labels.push(None);
        self.invalidate_cache();
        idx
    }

    /// Set a human‑readable label for a state.
    pub fn set_state_label(&mut self, state: usize, label: String) {
        if state < self.state_labels.len() {
            self.state_labels[state] = Some(label);
        }
    }

    /// Get the label of a state.
    pub fn state_label(&self, state: usize) -> Option<&str> {
        self.state_labels.get(state).and_then(|l| l.as_deref())
    }

    fn invalidate_cache(&mut self) {
        self.cache = CachedMeta::default();
    }

    // ── accessors ───────────────────────────────────────────────────────

    /// Number of states.
    pub fn state_count(&self) -> usize {
        self.num_states
    }

    /// Alias for `state_count()`.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Reference to the alphabet.
    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    /// Reference to the initial‑weight vector.
    pub fn initial_weights(&self) -> &[S] {
        &self.initial_weights
    }

    /// Reference to the final‑weight vector.
    pub fn final_weights(&self) -> &[S] {
        &self.final_weights
    }

    /// Iterate over all transitions.
    pub fn all_transitions(&self) -> Vec<Transition<S>> {
        let mut result = Vec::new();
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[from][sym] {
                    result.push(Transition::new(from, to, sym, w.clone()));
                }
            }
        }
        result
    }

    /// Transitions from a given state on a given symbol.
    pub fn transitions_from(&self, state: usize, symbol: usize) -> &[(usize, S)] {
        &self.transitions[state][symbol]
    }

    /// Reference to the full transitions table.
    pub fn transitions(&self) -> &Vec<Vec<Vec<(usize, S)>>> {
        &self.transitions
    }

    /// Get the total transition weight from `from` reading `symbol` going to `to`.
    /// Returns `S::zero()` if no such transition exists.
    pub fn transition_weight(&self, from: usize, symbol: usize, to: usize) -> S {
        let mut total = S::zero();
        for &(dest, ref w) in &self.transitions[from][symbol] {
            if dest == to {
                total.add_assign(w);
            }
        }
        total
    }

    /// Total number of transitions.
    pub fn num_transitions(&self) -> usize {
        if let Some(count) = self.cache.transition_count {
            return count;
        }
        let count: usize = self
            .transitions
            .iter()
            .flat_map(|per_state| per_state.iter())
            .map(|v| v.len())
            .sum();
        count
    }

    /// True when the automaton has zero states.
    pub fn is_empty_automaton(&self) -> bool {
        self.num_states == 0
    }

    /// True when no string has non‑zero weight (no accepting run).
    pub fn is_empty(&self) -> bool {
        if self.num_states == 0 {
            return true;
        }
        // Quick: if no initial or no final state has a non‑zero weight,
        // there is no accepting path.
        let has_initial = self.initial_weights.iter().any(|w| !w.is_zero());
        let has_final = self.final_weights.iter().any(|w| !w.is_zero());
        if !has_initial || !has_final {
            return true;
        }
        // Heavier: BFS from initial states to see if any final state is
        // reachable.
        let reachable = self.reachable_states();
        let productive = self.productive_states();
        let useful: HashSet<usize> = reachable.intersection(&productive).copied().collect();
        useful.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Weight computation
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Compute the weight (value in the semiring) that this WFA assigns to
    /// the input string given as a sequence of symbol indices.
    ///
    /// Uses the *forward algorithm*: we propagate weight vectors from left to
    /// right.
    ///
    /// `weight(w) = α^T · M(w_1) · M(w_2) · … · M(w_n) · β`
    ///
    /// where α is the initial‑weight vector, β the final‑weight vector, and
    /// M(a) the transition matrix for symbol a.
    pub fn compute_weight(&self, input: &[usize]) -> S {
        if self.num_states == 0 {
            return S::zero();
        }
        // forward[q] holds the accumulated weight of being in state q after
        // reading input[0..t].
        let mut forward: Vec<S> = self.initial_weights.clone();

        for &sym in input {
            if sym >= self.alphabet.size() {
                return S::zero();
            }
            let mut next = vec![S::zero(); self.num_states];
            for from in 0..self.num_states {
                if forward[from].is_zero() {
                    continue;
                }
                for &(to, ref w) in &self.transitions[from][sym] {
                    let contribution = forward[from].mul(w);
                    next[to].add_assign(&contribution);
                }
            }
            forward = next;
        }

        // Collect: sum_q forward[q] * final_weights[q]
        let mut total = S::zero();
        for q in 0..self.num_states {
            if !forward[q].is_zero() && !self.final_weights[q].is_zero() {
                let contribution = forward[q].mul(&self.final_weights[q]);
                total.add_assign(&contribution);
            }
        }
        total
    }

    /// Return all forward vectors: `result[t]` is the weight vector after
    /// reading `input[0..t]`.  `result[0]` equals the initial‑weight vector.
    pub fn forward_vectors(&self, input: &[usize]) -> Vec<Vec<S>> {
        let n = self.num_states;
        let mut vectors: Vec<Vec<S>> = Vec::with_capacity(input.len() + 1);
        vectors.push(self.initial_weights.clone());

        for &sym in input {
            let prev = vectors.last().unwrap();
            let mut next = vec![S::zero(); n];
            if sym < self.alphabet.size() {
                for from in 0..n {
                    if prev[from].is_zero() {
                        continue;
                    }
                    for &(to, ref w) in &self.transitions[from][sym] {
                        let c = prev[from].mul(w);
                        next[to].add_assign(&c);
                    }
                }
            }
            vectors.push(next);
        }
        vectors
    }

    /// Return all backward vectors: `result[t]` is the weight of reaching a
    /// final state from each state when reading `input[t..]`.
    /// `result[input.len()]` equals the final‑weight vector.
    pub fn backward_vectors(&self, input: &[usize]) -> Vec<Vec<S>> {
        let n = self.num_states;
        let len = input.len();
        let mut vectors: Vec<Vec<S>> = vec![vec![S::zero(); n]; len + 1];
        // Base: backward[len] = final_weights
        vectors[len] = self.final_weights.clone();

        for t in (0..len).rev() {
            let sym = input[t];
            if sym >= self.alphabet.size() {
                // vectors[t] stays zero
                continue;
            }
            for from in 0..n {
                for &(to, ref w) in &self.transitions[from][sym] {
                    if vectors[t + 1][to].is_zero() {
                        continue;
                    }
                    let c = w.mul(&vectors[t + 1][to]);
                    vectors[t][from].add_assign(&c);
                }
            }
        }
        vectors
    }

    /// Whether the WFA assigns a non‑zero weight to `input`.
    pub fn accepts(&self, input: &[usize]) -> bool {
        !self.compute_weight(input).is_zero()
    }

    /// Return the transition matrix for a single symbol.
    pub fn weight_matrix(&self, symbol: usize) -> SemiringMatrix<S> {
        let n = self.num_states;
        let mut mat = SemiringMatrix::zeros(n, n);
        if symbol >= self.alphabet.size() {
            return mat;
        }
        for from in 0..n {
            for &(to, ref w) in &self.transitions[from][symbol] {
                let cur = mat.get(from, to).unwrap().clone();
                let _ = mat.set(from, to, cur.add(w));
            }
        }
        mat
    }

    /// Compute the weight of a string via matrix multiplication.
    pub fn compute_weight_matrix(&self, input: &[usize]) -> S {
        if self.num_states == 0 {
            return S::zero();
        }
        let n = self.num_states;

        // Build 1×n row vector for initial weights.
        let mut row = SemiringMatrix::zeros(1, n);
        for q in 0..n {
            let _ = row.set(0, q, self.initial_weights[q].clone());
        }

        for &sym in input {
            let m = self.weight_matrix(sym);
            row = row.mul(&m).unwrap();
        }

        // Multiply by final‑weight column vector.
        let mut total = S::zero();
        for q in 0..n {
            let c = row.get(0, q).unwrap().mul(&self.final_weights[q]);
            total.add_assign(&c);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Closure operations
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Union of two WFAs.
    ///
    /// The result automaton has `|Q1| + |Q2|` states; its weight for a
    /// string w equals `weight_1(w) + weight_2(w)` in the semiring.
    pub fn union(&self, other: &Self) -> WfaResult<Self> {
        let combined_alpha = self.alphabet.union(&other.alphabet);
        let n1 = self.num_states;
        let n2 = other.num_states;
        let total = n1 + n2;
        let mut wfa = Self::new(total, combined_alpha.clone());

        // Initial weights.
        for q in 0..n1 {
            wfa.initial_weights[q] = self.initial_weights[q].clone();
        }
        for q in 0..n2 {
            wfa.initial_weights[n1 + q] = other.initial_weights[q].clone();
        }

        // Final weights.
        for q in 0..n1 {
            wfa.final_weights[q] = self.final_weights[q].clone();
        }
        for q in 0..n2 {
            wfa.final_weights[n1 + q] = other.final_weights[q].clone();
        }

        // Transitions from self.
        let self_map = self.alphabet.index_mapping(&combined_alpha);
        for from in 0..n1 {
            for sym in 0..self.alphabet.size() {
                if let Some(new_sym) = self_map[sym] {
                    for &(to, ref w) in &self.transitions[from][sym] {
                        wfa.add_transition(from, new_sym, to, w.clone())?;
                    }
                }
            }
        }

        // Transitions from other.
        let other_map = other.alphabet.index_mapping(&combined_alpha);
        for from in 0..n2 {
            for sym in 0..other.alphabet.size() {
                if let Some(new_sym) = other_map[sym] {
                    for &(to, ref w) in &other.transitions[from][sym] {
                        wfa.add_transition(n1 + from, new_sym, n1 + to, w.clone())?;
                    }
                }
            }
        }

        Ok(wfa)
    }

    /// Concatenation of two WFAs.
    ///
    /// The result recognises `{uv | u ∈ L(self), v ∈ L(other)}` with weight
    /// being the semiring product of the individual weights, summed over all
    /// factorisations.
    ///
    /// Construction: states = Q1 ∪ Q2.  For every pair (q, r) where q is a
    /// final state of A1 and r is an initial state of A2 we add "linking"
    /// transitions that fold final(q) · initial(r) into the weight.
    pub fn concatenation(&self, other: &Self) -> WfaResult<Self> {
        let combined_alpha = self.alphabet.union(&other.alphabet);
        let n1 = self.num_states;
        let n2 = other.num_states;
        let total = n1 + n2;
        let mut wfa = Self::new(total, combined_alpha.clone());

        // Initial weights: from self only.
        for q in 0..n1 {
            wfa.initial_weights[q] = self.initial_weights[q].clone();
        }

        // Final weights: from other only (shifted).
        for q in 0..n2 {
            wfa.final_weights[n1 + q] = other.final_weights[q].clone();
        }

        // Copy transitions of self.
        let self_map = self.alphabet.index_mapping(&combined_alpha);
        for from in 0..n1 {
            for sym in 0..self.alphabet.size() {
                if let Some(new_sym) = self_map[sym] {
                    for &(to, ref w) in &self.transitions[from][sym] {
                        wfa.add_transition(from, new_sym, to, w.clone())?;
                    }
                }
            }
        }

        // Copy transitions of other (shifted).
        let other_map = other.alphabet.index_mapping(&combined_alpha);
        for from in 0..n2 {
            for sym in 0..other.alphabet.size() {
                if let Some(new_sym) = other_map[sym] {
                    for &(to, ref w) in &other.transitions[from][sym] {
                        wfa.add_transition(n1 + from, new_sym, n1 + to, w.clone())?;
                    }
                }
            }
        }

        // Linking transitions: for every final state q of self and every
        // initial state r of other, for each transition (r, a, s, w) in
        // other, add transition (q, a, n1+s, final[q]·init[r]·w) in the
        // result.
        for q in 0..n1 {
            if self.final_weights[q].is_zero() {
                continue;
            }
            for r in 0..n2 {
                if other.initial_weights[r].is_zero() {
                    continue;
                }
                let link_weight = self.final_weights[q].mul(&other.initial_weights[r]);
                for sym in 0..other.alphabet.size() {
                    if let Some(new_sym) = other_map[sym] {
                        for &(to, ref w) in &other.transitions[r][sym] {
                            let tw = link_weight.mul(w);
                            wfa.add_transition(q, new_sym, n1 + to, tw)?;
                        }
                    }
                }
            }
        }

        // Handle case where other accepts the empty string: propagate
        // self's final weights directly to the result.
        for q in 0..n1 {
            if self.final_weights[q].is_zero() {
                continue;
            }
            for r in 0..n2 {
                if other.initial_weights[r].is_zero() || other.final_weights[r].is_zero() {
                    continue;
                }
                let extra = self
                    .final_weights[q]
                    .mul(&other.initial_weights[r])
                    .mul(&other.final_weights[r]);
                wfa.final_weights[q].add_assign(&extra);
            }
        }

        Ok(wfa)
    }

    /// Product / Hadamard construction (intersection for Boolean, point‑wise
    /// product for general semirings).
    ///
    /// States of the result are pairs (q1, q2).  The alphabets must be
    /// compatible (we use their intersection).
    pub fn intersection(&self, other: &Self) -> WfaResult<Self> {
        let common_alpha = self.alphabet.intersection(&other.alphabet);
        if common_alpha.size() == 0 && self.alphabet.size() > 0 && other.alphabet.size() > 0 {
            return Err(WfaError::AlphabetMismatch);
        }

        let n1 = self.num_states;
        let n2 = other.num_states;
        let total = n1 * n2;
        let mut wfa = Self::new(total, common_alpha.clone());

        let self_map = self.alphabet.index_mapping(&common_alpha);
        let other_map = other.alphabet.index_mapping(&common_alpha);

        // Build reverse maps: for each symbol in the common alphabet, find
        // its index in self and other.
        let mut self_rev: Vec<Option<usize>> = vec![None; common_alpha.size()];
        let mut other_rev: Vec<Option<usize>> = vec![None; common_alpha.size()];
        for (i, mi) in self_map.iter().enumerate() {
            if let Some(ci) = mi {
                self_rev[*ci] = Some(i);
            }
        }
        for (i, mi) in other_map.iter().enumerate() {
            if let Some(ci) = mi {
                other_rev[*ci] = Some(i);
            }
        }

        let pair_to_state = |q1: usize, q2: usize| -> usize { q1 * n2 + q2 };

        // Initial and final weights.
        for q1 in 0..n1 {
            for q2 in 0..n2 {
                let s = pair_to_state(q1, q2);
                wfa.initial_weights[s] = self.initial_weights[q1].mul(&other.initial_weights[q2]);
                wfa.final_weights[s] = self.final_weights[q1].mul(&other.final_weights[q2]);
            }
        }

        // Transitions.
        for csym in 0..common_alpha.size() {
            let s1_idx = match self_rev[csym] {
                Some(i) => i,
                None => continue,
            };
            let s2_idx = match other_rev[csym] {
                Some(i) => i,
                None => continue,
            };
            for q1 in 0..n1 {
                for &(r1, ref w1) in &self.transitions[q1][s1_idx] {
                    for q2 in 0..n2 {
                        for &(r2, ref w2) in &other.transitions[q2][s2_idx] {
                            let from = pair_to_state(q1, q2);
                            let to = pair_to_state(r1, r2);
                            let weight = w1.mul(w2);
                            wfa.add_transition(from, csym, to, weight)?;
                        }
                    }
                }
            }
        }

        Ok(wfa)
    }

    /// Reverse the automaton: swap initial and final, reverse transitions.
    pub fn reverse(&self) -> Self {
        let mut rev = Self::new(self.num_states, self.alphabet.clone());
        rev.initial_weights = self.final_weights.clone();
        rev.final_weights = self.initial_weights.clone();

        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[from][sym] {
                    rev.transitions[to][sym].push((from, w.clone()));
                }
            }
        }
        rev
    }
}

// Kleene star requires StarSemiring.
impl<S: StarSemiring> WeightedFiniteAutomaton<S> {
    /// Kleene star (closure).
    ///
    /// We add a new initial/final state and link it to the original
    /// automaton.  The new state is both initial (weight one) and final
    /// (weight one), giving the empty‑string contribution.
    pub fn kleene_star(&self) -> Self {
        if self.num_states == 0 {
            // L* of the empty language is {ε}.
            return Self::epsilon_wfa(S::one());
        }

        let n = self.num_states;
        let new_n = n + 1;
        let new_state = n;
        let mut wfa = Self::new(new_n, self.alphabet.clone());

        // The new state is the sole initial state (weight one) and final
        // (weight one).
        wfa.initial_weights[new_state] = S::one();
        wfa.final_weights[new_state] = S::one();

        // Copy final weights of the original automaton.
        for q in 0..n {
            wfa.final_weights[q] = self.final_weights[q].clone();
        }

        // Copy original transitions.
        for from in 0..n {
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[from][sym] {
                    wfa.transitions[from][sym].push((to, w.clone()));
                }
            }
        }

        // From new_state, add transitions mimicking the initial distribution:
        // for every original initial state r with weight α_r, and for every
        // transition (r, a, s, w), add (new_state, a, s, α_r · w).
        for r in 0..n {
            if self.initial_weights[r].is_zero() {
                continue;
            }
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[r][sym] {
                    let weight = self.initial_weights[r].mul(w);
                    wfa.transitions[new_state][sym].push((to, weight));
                }
            }
        }

        // From every final state q, add transitions mimicking the initial
        // distribution so that after completing one pass we can start
        // another: for every final state q, initial state r, and
        // transition (r, a, s, w), add (q, a, s, final[q] · init[r] · w).
        for q in 0..n {
            if self.final_weights[q].is_zero() {
                continue;
            }
            for r in 0..n {
                if self.initial_weights[r].is_zero() {
                    continue;
                }
                let link = self.final_weights[q].mul(&self.initial_weights[r]);
                for sym in 0..self.alphabet.size() {
                    for &(to, ref w) in &self.transitions[r][sym] {
                        let weight = link.mul(w);
                        wfa.transitions[q][sym].push((to, weight));
                    }
                }
            }
        }

        wfa
    }
}

// Complement is only well‑defined for Boolean‑weighted deterministic WFAs.
impl WeightedFiniteAutomaton<BooleanSemiring> {
    /// Complement: flip final / non‑final for a *deterministic* WFA.
    ///
    /// If the automaton is not deterministic it is determinized first.
    pub fn complement(&self) -> Self {
        let det = if self.is_deterministic() {
            self.clone()
        } else {
            self.determinize()
        };
        let mut comp = det.clone();
        for q in 0..comp.num_states {
            comp.final_weights[q] = if det.final_weights[q].is_zero() {
                BooleanSemiring::one()
            } else {
                BooleanSemiring::zero()
            };
        }
        comp.invalidate_cache();
        comp
    }
}

// ---------------------------------------------------------------------------
// Determinization
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S>
where
    S: std::hash::Hash + Eq + Ord,
{
    /// Whether the WFA is deterministic.
    ///
    /// A WFA is deterministic when (a) exactly one state has non‑zero initial
    /// weight, and (b) for every state and symbol there is at most one
    /// transition.
    pub fn is_deterministic(&self) -> bool {
        let initial_count = self
            .initial_weights
            .iter()
            .filter(|w| !w.is_zero())
            .count();
        if initial_count > 1 {
            return false;
        }
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                if self.transitions[from][sym].len() > 1 {
                    return false;
                }
            }
        }
        true
    }

    /// Determinize the WFA via weighted subset construction.
    ///
    /// Each macro‑state is a mapping from original states to weights
    /// (normalised so that the first non‑zero weight is `one`).
    pub fn determinize(&self) -> Self {
        if self.num_states == 0 {
            return Self::empty();
        }

        // A macro‑state is a BTreeMap<usize, S> from state → weight.
        type MacroState<W> = BTreeMap<usize, W>;

        fn normalize_macro<W: Semiring + Ord + Clone>(ms: &MacroState<W>) -> (MacroState<W>, W) {
            // Find the "first" non-zero weight to use as the normalization factor.
            let mut factor = W::one();
            let mut found = false;
            for (_q, w) in ms.iter() {
                if !w.is_zero() {
                    factor = w.clone();
                    found = true;
                    break;
                }
            }
            if !found {
                return (ms.clone(), W::one());
            }
            // For general semirings we cannot divide, so we leave normalization
            // as an identity and just use the raw map as the key.
            (ms.clone(), W::one())
        }

        // Initial macro‑state.
        let mut init_macro: MacroState<S> = BTreeMap::new();
        for q in 0..self.num_states {
            if !self.initial_weights[q].is_zero() {
                init_macro.insert(q, self.initial_weights[q].clone());
            }
        }

        if init_macro.is_empty() {
            return Self::empty();
        }

        let alpha_size = self.alphabet.size();

        // Map from canonical macro‑state → new state id.
        let mut macro_to_id: HashMap<BTreeMap<usize, S>, usize> = HashMap::new();
        let mut id_to_macro: Vec<MacroState<S>> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();

        let (norm_init, _) = normalize_macro(&init_macro);
        macro_to_id.insert(norm_init.clone(), 0);
        id_to_macro.push(norm_init.clone());
        queue.push_back(0);

        // Deterministic transition function for the new automaton.
        let mut det_transitions: Vec<Vec<Option<(usize, S)>>> = Vec::new();
        det_transitions.push(vec![None; alpha_size]);

        while let Some(cur_id) = queue.pop_front() {
            let cur_macro = id_to_macro[cur_id].clone();

            for sym in 0..alpha_size {
                // Compute successor macro‑state.
                let mut next_macro: MacroState<S> = BTreeMap::new();
                for (&q, w_q) in cur_macro.iter() {
                    for &(to, ref w_t) in &self.transitions[q][sym] {
                        let contribution = w_q.mul(w_t);
                        next_macro
                            .entry(to)
                            .and_modify(|existing: &mut S| existing.add_assign(&contribution))
                            .or_insert(contribution);
                    }
                }
                // Remove zero entries.
                next_macro.retain(|_, w| !w.is_zero());

                if next_macro.is_empty() {
                    continue;
                }

                let (norm_next, _) = normalize_macro(&next_macro);

                let next_id = if let Some(&id) = macro_to_id.get(&norm_next) {
                    id
                } else {
                    let id = id_to_macro.len();
                    macro_to_id.insert(norm_next.clone(), id);
                    id_to_macro.push(norm_next);
                    det_transitions.push(vec![None; alpha_size]);
                    queue.push_back(id);
                    id
                };

                det_transitions[cur_id][sym] = Some((next_id, S::one()));
            }
        }

        // Build the deterministic WFA.
        let new_n = id_to_macro.len();
        let mut wfa = Self::new(new_n, self.alphabet.clone());
        wfa.initial_weights[0] = S::one();

        // Final weights.
        for (id, ms) in id_to_macro.iter().enumerate() {
            let mut fw = S::zero();
            for (&q, w_q) in ms.iter() {
                let c = w_q.mul(&self.final_weights[q]);
                fw.add_assign(&c);
            }
            wfa.final_weights[id] = fw;
        }

        // Transitions.
        for from in 0..new_n {
            for sym in 0..alpha_size {
                if let Some((to, ref w)) = det_transitions[from][sym] {
                    wfa.transitions[from][sym].push((to, w.clone()));
                }
            }
        }

        wfa
    }
}

// Provide is_deterministic for S that doesn't have Hash+Eq+Ord via a
// separate inherent block so we don't require those bounds on the
// main impl.  (Rust selects the most specific impl.)
impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Check determinism (available for all semirings, no Hash/Eq/Ord needed).
    pub fn check_deterministic(&self) -> bool {
        let initial_count = self
            .initial_weights
            .iter()
            .filter(|w| !w.is_zero())
            .count();
        if initial_count > 1 {
            return false;
        }
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                if self.transitions[from][sym].len() > 1 {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Epsilon handling
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Whether the WFA has any epsilon transitions.
    pub fn has_epsilon_transitions(&self) -> bool {
        let eps_idx = match self.alphabet.epsilon_index() {
            Some(i) => i,
            None => return false,
        };
        for from in 0..self.num_states {
            if !self.transitions[from][eps_idx].is_empty() {
                return true;
            }
        }
        false
    }

    /// Remove epsilon transitions by computing the epsilon‑closure via a
    /// transitive‑closure style fixed‑point computation.
    ///
    /// Returns a new WFA without epsilon transitions (and without the
    /// epsilon symbol in the alphabet).
    pub fn remove_epsilon(&self) -> Self {
        let eps_idx = match self.alphabet.epsilon_index() {
            Some(i) => i,
            None => return self.clone(),
        };

        let n = self.num_states;
        // Compute epsilon‑closure matrix: eps_closure[q][r] is the total
        // weight of reaching r from q using only epsilon transitions.
        // We compute this as the reflexive‑transitive closure of the
        // epsilon‑transition matrix.
        let mut eps_closure: Vec<Vec<S>> = vec![vec![S::zero(); n]; n];
        // Reflexive: eps_closure[q][q] = one.
        for q in 0..n {
            eps_closure[q][q] = S::one();
        }
        // Add direct epsilon transitions.
        for from in 0..n {
            for &(to, ref w) in &self.transitions[from][eps_idx] {
                eps_closure[from][to].add_assign(w);
            }
        }
        // Fixed‑point: Warshall‑style.
        for k in 0..n {
            for i in 0..n {
                if eps_closure[i][k].is_zero() {
                    continue;
                }
                for j in 0..n {
                    if eps_closure[k][j].is_zero() {
                        continue;
                    }
                    let product = eps_closure[i][k].mul(&eps_closure[k][j]);
                    eps_closure[i][j].add_assign(&product);
                }
            }
        }

        // Build new alphabet without epsilon.
        let mut new_alpha = Alphabet::new();
        let mut old_to_new: Vec<Option<usize>> = vec![None; self.alphabet.size()];
        for (i, sym) in self.alphabet.symbols.iter().enumerate() {
            if i == eps_idx {
                continue;
            }
            let new_idx = new_alpha.insert(sym.clone());
            old_to_new[i] = Some(new_idx);
        }

        let mut wfa = Self::new(n, new_alpha);

        // New initial weights: α'[q] = Σ_r α[r] · ε*(r, q).
        for q in 0..n {
            let mut w = S::zero();
            for r in 0..n {
                if self.initial_weights[r].is_zero() || eps_closure[r][q].is_zero() {
                    continue;
                }
                let c = self.initial_weights[r].mul(&eps_closure[r][q]);
                w.add_assign(&c);
            }
            wfa.initial_weights[q] = w;
        }

        // Final weights stay the same.
        wfa.final_weights = self.final_weights.clone();

        // New transitions: for every non‑epsilon transition (p, a, r, w),
        // and for every state q reachable from some state via epsilon to p,
        // and for every state s reachable from r via epsilon, add a new
        // transition (q, a, s, ε*(q,p) · w · ε*(r,s)).
        //
        // Optimised: we fold the epsilon‑closure into the transitions from
        // each state.
        //
        // For each original non-eps transition (p, a, r, w):
        //   For each q such that eps_closure[q][p] != 0:
        //     For each s such that eps_closure[r][s] != 0:
        //       add (q, a', s, eps_closure[q][p] * w * eps_closure[r][s])
        for p in 0..n {
            for old_sym in 0..self.alphabet.size() {
                if old_sym == eps_idx {
                    continue;
                }
                let new_sym = match old_to_new[old_sym] {
                    Some(ns) => ns,
                    None => continue,
                };
                for &(r, ref w) in &self.transitions[p][old_sym] {
                    for q in 0..n {
                        if eps_closure[q][p].is_zero() {
                            continue;
                        }
                        let left = eps_closure[q][p].mul(w);
                        for s in 0..n {
                            if eps_closure[r][s].is_zero() {
                                continue;
                            }
                            let weight = left.mul(&eps_closure[r][s]);
                            if !weight.is_zero() {
                                wfa.transitions[q][new_sym].push((s, weight));
                            }
                        }
                    }
                }
            }
        }

        // Update final weights through epsilon closure:
        // β'[q] = Σ_r ε*(q,r) · β[r]
        for q in 0..n {
            let mut fw = S::zero();
            for r in 0..n {
                if eps_closure[q][r].is_zero() || self.final_weights[r].is_zero() {
                    continue;
                }
                let c = eps_closure[q][r].mul(&self.final_weights[r]);
                fw.add_assign(&c);
            }
            wfa.final_weights[q] = fw;
        }

        wfa
    }
}

// ---------------------------------------------------------------------------
// Trimming & reachability
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Compute the set of states reachable from any initial state via
    /// forward BFS.
    pub fn reachable_states(&self) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for q in 0..self.num_states {
            if !self.initial_weights[q].is_zero() {
                visited.insert(q);
                queue.push_back(q);
            }
        }
        while let Some(q) = queue.pop_front() {
            for sym in 0..self.alphabet.size() {
                for &(to, ref _w) in &self.transitions[q][sym] {
                    if visited.insert(to) {
                        queue.push_back(to);
                    }
                }
            }
        }
        visited
    }

    /// Compute the set of productive states, i.e. states from which a final
    /// state is reachable via backward BFS.
    pub fn productive_states(&self) -> HashSet<usize> {
        // Build reverse adjacency list.
        let mut rev_adj: Vec<HashSet<usize>> = vec![HashSet::new(); self.num_states];
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, ref _w) in &self.transitions[from][sym] {
                    rev_adj[to].insert(from);
                }
            }
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for q in 0..self.num_states {
            if !self.final_weights[q].is_zero() {
                visited.insert(q);
                queue.push_back(q);
            }
        }
        while let Some(q) = queue.pop_front() {
            for &pred in &rev_adj[q] {
                if visited.insert(pred) {
                    queue.push_back(pred);
                }
            }
        }
        visited
    }

    /// Trim the automaton: keep only states that are both reachable and
    /// productive, renumber them contiguously.
    pub fn trim(&self) -> Self {
        let reachable = self.reachable_states();
        let productive = self.productive_states();
        let useful: BTreeSet<usize> = reachable.intersection(&productive).copied().collect();

        if useful.len() == self.num_states {
            return self.clone();
        }
        if useful.is_empty() {
            return Self::new(0, self.alphabet.clone());
        }

        // Build old→new mapping.
        let mut old_to_new: Vec<Option<usize>> = vec![None; self.num_states];
        let mut new_to_old: Vec<usize> = Vec::new();
        for &q in &useful {
            old_to_new[q] = Some(new_to_old.len());
            new_to_old.push(q);
        }

        let new_n = new_to_old.len();
        let mut wfa = Self::new(new_n, self.alphabet.clone());

        for (new_q, &old_q) in new_to_old.iter().enumerate() {
            wfa.initial_weights[new_q] = self.initial_weights[old_q].clone();
            wfa.final_weights[new_q] = self.final_weights[old_q].clone();
            for sym in 0..self.alphabet.size() {
                for &(old_to, ref w) in &self.transitions[old_q][sym] {
                    if let Some(new_to) = old_to_new[old_to] {
                        wfa.transitions[new_q][sym].push((new_to, w.clone()));
                    }
                }
            }
        }

        wfa
    }

    /// Whether the automaton is already trim.
    pub fn is_trim(&self) -> bool {
        if self.num_states == 0 {
            return true;
        }
        let reachable = self.reachable_states();
        let productive = self.productive_states();
        let useful: HashSet<usize> = reachable.intersection(&productive).copied().collect();
        useful.len() == self.num_states
    }

    /// Return the sub‑automaton induced by a given set of states.
    pub fn subautomaton(&self, states: &HashSet<usize>) -> Self {
        let ordered: BTreeSet<usize> = states.iter().copied().collect();
        let mut old_to_new: Vec<Option<usize>> = vec![None; self.num_states];
        let mut new_to_old: Vec<usize> = Vec::new();
        for &q in &ordered {
            if q < self.num_states {
                old_to_new[q] = Some(new_to_old.len());
                new_to_old.push(q);
            }
        }

        let new_n = new_to_old.len();
        let mut wfa = Self::new(new_n, self.alphabet.clone());

        for (new_q, &old_q) in new_to_old.iter().enumerate() {
            wfa.initial_weights[new_q] = self.initial_weights[old_q].clone();
            wfa.final_weights[new_q] = self.final_weights[old_q].clone();
            for sym in 0..self.alphabet.size() {
                for &(old_to, ref w) in &self.transitions[old_q][sym] {
                    if let Some(new_to) = old_to_new[old_to] {
                        wfa.transitions[new_q][sym].push((new_to, w.clone()));
                    }
                }
            }
        }

        wfa
    }

    /// Rename states according to the given mapping.
    /// `mapping[old_state] = new_state`.
    pub fn rename_states(&self, mapping: &[usize]) -> Self {
        assert_eq!(mapping.len(), self.num_states);
        let new_n = mapping.iter().copied().max().map_or(0, |m| m + 1);
        let mut wfa = Self::new(new_n, self.alphabet.clone());

        for q in 0..self.num_states {
            let new_q = mapping[q];
            wfa.initial_weights[new_q].add_assign(&self.initial_weights[q]);
            wfa.final_weights[new_q].add_assign(&self.final_weights[q]);
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[q][sym] {
                    let new_to = mapping[to];
                    wfa.transitions[new_q][sym].push((new_to, w.clone()));
                }
            }
        }

        wfa
    }
}

// ---------------------------------------------------------------------------
// Regex → WFA (Thompson's construction, Boolean semiring)
// ---------------------------------------------------------------------------

/// Internal parsed regex AST for Thompson's construction.
#[derive(Debug, Clone)]
enum RegexAst {
    Literal(Symbol),
    Concatenation(Vec<RegexAst>),
    Alternation(Vec<RegexAst>),
    KleeneStar(Box<RegexAst>),
    KleenePlus(Box<RegexAst>),
    Optional(Box<RegexAst>),
    Epsilon,
    Empty,
    Dot, // wildcard
    CharClass(Vec<char>, bool), // chars, negated?
}

impl WeightedFiniteAutomaton<BooleanSemiring> {
    /// Build a WFA from a simple regex string over the given alphabet.
    ///
    /// Supported syntax:
    /// - Literal characters
    /// - `|` alternation
    /// - `*` Kleene star
    /// - `+` Kleene plus
    /// - `?` optional
    /// - `.` wildcard (matches any symbol in the alphabet)
    /// - `(`, `)` grouping
    /// - `[abc]`, `[^abc]` character classes
    /// - `\` escape
    pub fn from_regex_str(pattern: &str, alphabet: &Alphabet) -> WfaResult<Self> {
        let ast = parse_regex(pattern)?;
        let mut builder = ThompsonBuilder::new(alphabet.clone());
        let (start, end) = builder.build_fragment(&ast)?;
        let mut wfa = builder.finalize(start, end);
        wfa
    }
}

/// Parser for simple regex patterns.
fn parse_regex(pattern: &str) -> WfaResult<RegexAst> {
    let chars: Vec<char> = pattern.chars().collect();
    let (ast, pos) = parse_alternation(&chars, 0)?;
    if pos != chars.len() {
        return Err(WfaError::RegexParse(format!(
            "unexpected character '{}' at position {}",
            chars[pos], pos
        )));
    }
    Ok(ast)
}

fn parse_alternation(chars: &[char], mut pos: usize) -> WfaResult<(RegexAst, usize)> {
    let mut branches = Vec::new();
    let (first, new_pos) = parse_concatenation(chars, pos)?;
    branches.push(first);
    pos = new_pos;

    while pos < chars.len() && chars[pos] == '|' {
        pos += 1; // skip '|'
        let (branch, new_pos) = parse_concatenation(chars, pos)?;
        branches.push(branch);
        pos = new_pos;
    }

    if branches.len() == 1 {
        Ok((branches.remove(0), pos))
    } else {
        Ok((RegexAst::Alternation(branches), pos))
    }
}

fn parse_concatenation(chars: &[char], mut pos: usize) -> WfaResult<(RegexAst, usize)> {
    let mut parts = Vec::new();

    while pos < chars.len() && chars[pos] != '|' && chars[pos] != ')' {
        let (atom, new_pos) = parse_quantified(chars, pos)?;
        parts.push(atom);
        pos = new_pos;
    }

    if parts.is_empty() {
        Ok((RegexAst::Epsilon, pos))
    } else if parts.len() == 1 {
        Ok((parts.remove(0), pos))
    } else {
        Ok((RegexAst::Concatenation(parts), pos))
    }
}

fn parse_quantified(chars: &[char], pos: usize) -> WfaResult<(RegexAst, usize)> {
    let (atom, mut pos) = parse_atom(chars, pos)?;

    if pos < chars.len() {
        match chars[pos] {
            '*' => {
                pos += 1;
                Ok((RegexAst::KleeneStar(Box::new(atom)), pos))
            }
            '+' => {
                pos += 1;
                Ok((RegexAst::KleenePlus(Box::new(atom)), pos))
            }
            '?' => {
                pos += 1;
                Ok((RegexAst::Optional(Box::new(atom)), pos))
            }
            _ => Ok((atom, pos)),
        }
    } else {
        Ok((atom, pos))
    }
}

fn parse_atom(chars: &[char], mut pos: usize) -> WfaResult<(RegexAst, usize)> {
    if pos >= chars.len() {
        return Err(WfaError::RegexParse("unexpected end of pattern".into()));
    }

    match chars[pos] {
        '(' => {
            pos += 1; // skip '('
            let (inner, new_pos) = parse_alternation(chars, pos)?;
            pos = new_pos;
            if pos >= chars.len() || chars[pos] != ')' {
                return Err(WfaError::RegexParse("unmatched '('".into()));
            }
            pos += 1; // skip ')'
            Ok((inner, pos))
        }
        '.' => {
            pos += 1;
            Ok((RegexAst::Dot, pos))
        }
        '[' => {
            pos += 1;
            let negated = if pos < chars.len() && chars[pos] == '^' {
                pos += 1;
                true
            } else {
                false
            };
            let mut class_chars = Vec::new();
            while pos < chars.len() && chars[pos] != ']' {
                if chars[pos] == '\\' && pos + 1 < chars.len() {
                    pos += 1;
                    class_chars.push(chars[pos]);
                } else if pos + 2 < chars.len() && chars[pos + 1] == '-' {
                    let from = chars[pos];
                    let to = chars[pos + 2];
                    for c in from..=to {
                        class_chars.push(c);
                    }
                    pos += 2;
                } else {
                    class_chars.push(chars[pos]);
                }
                pos += 1;
            }
            if pos >= chars.len() {
                return Err(WfaError::RegexParse("unmatched '['".into()));
            }
            pos += 1; // skip ']'
            Ok((RegexAst::CharClass(class_chars, negated), pos))
        }
        '\\' => {
            pos += 1;
            if pos >= chars.len() {
                return Err(WfaError::RegexParse("trailing '\\'".into()));
            }
            let c = chars[pos];
            pos += 1;
            Ok((RegexAst::Literal(Symbol::Char(c)), pos))
        }
        c if c == '*' || c == '+' || c == '?' => {
            Err(WfaError::RegexParse(format!(
                "unexpected quantifier '{}' at position {}",
                c,
                pos
            )))
        }
        c => {
            pos += 1;
            Ok((RegexAst::Literal(Symbol::Char(c)), pos))
        }
    }
}

/// Thompson's NFA construction produces an NFA with epsilon transitions.
struct ThompsonBuilder {
    pub alphabet: Alphabet,
    pub num_states: usize,
    transitions: Vec<Vec<Vec<(usize, BooleanSemiring)>>>,
}

impl ThompsonBuilder {
    fn new(alphabet: Alphabet) -> Self {
        // Ensure the alphabet has an epsilon column.
        let alphabet = if alphabet.has_epsilon() {
            alphabet
        } else {
            alphabet.with_epsilon()
        };
        ThompsonBuilder {
            alphabet,
            num_states: 0,
            transitions: Vec::new(),
        }
    }

    fn new_state(&mut self) -> usize {
        let id = self.num_states;
        self.num_states += 1;
        self.transitions
            .push(vec![Vec::new(); self.alphabet.size()]);
        id
    }

    fn add_epsilon(&mut self, from: usize, to: usize) {
        let eps_idx = self.alphabet.epsilon_index().unwrap();
        self.transitions[from][eps_idx].push((to, BooleanSemiring::one()));
    }

    fn add_symbol_transition(&mut self, from: usize, sym_idx: usize, to: usize) {
        self.transitions[from][sym_idx].push((to, BooleanSemiring::one()));
    }

    /// Build a fragment (start_state, end_state) for the given AST node.
    fn build_fragment(&mut self, ast: &RegexAst) -> WfaResult<(usize, usize)> {
        match ast {
            RegexAst::Literal(sym) => {
                let idx = self
                    .alphabet
                    .index_of(sym)
                    .ok_or_else(|| WfaError::RegexParse(format!("symbol {} not in alphabet", sym)))?;
                let s = self.new_state();
                let e = self.new_state();
                self.add_symbol_transition(s, idx, e);
                Ok((s, e))
            }
            RegexAst::Epsilon => {
                let s = self.new_state();
                let e = self.new_state();
                self.add_epsilon(s, e);
                Ok((s, e))
            }
            RegexAst::Empty => {
                let s = self.new_state();
                let e = self.new_state();
                // No transition → accepts nothing.
                Ok((s, e))
            }
            RegexAst::Dot => {
                let s = self.new_state();
                let e = self.new_state();
                // Transition on every non‑epsilon symbol.
                let syms: Vec<(usize, Symbol)> = self.alphabet.symbols.iter().enumerate()
                    .map(|(i, sym)| (i, sym.clone())).collect();
                for (i, sym) in &syms {
                    if *sym != Symbol::Epsilon {
                        self.add_symbol_transition(s, *i, e);
                    }
                }
                Ok((s, e))
            }
            RegexAst::CharClass(chars, negated) => {
                let s = self.new_state();
                let e = self.new_state();
                let char_set: HashSet<char> = chars.iter().copied().collect();
                let syms: Vec<(usize, Symbol)> = self.alphabet.symbols.iter().enumerate()
                    .map(|(i, sym)| (i, sym.clone())).collect();
                for (i, sym) in &syms {
                    if let Symbol::Char(c) = sym {
                        let in_class = char_set.contains(c);
                        if in_class != *negated {
                            self.add_symbol_transition(s, *i, e);
                        }
                    }
                }
                Ok((s, e))
            }
            RegexAst::Concatenation(parts) => {
                if parts.is_empty() {
                    return self.build_fragment(&RegexAst::Epsilon);
                }
                let (mut start, mut end) = self.build_fragment(&parts[0])?;
                for part in &parts[1..] {
                    let (s, e) = self.build_fragment(part)?;
                    self.add_epsilon(end, s);
                    end = e;
                }
                Ok((start, end))
            }
            RegexAst::Alternation(branches) => {
                let s = self.new_state();
                let e = self.new_state();
                for branch in branches {
                    let (bs, be) = self.build_fragment(branch)?;
                    self.add_epsilon(s, bs);
                    self.add_epsilon(be, e);
                }
                Ok((s, e))
            }
            RegexAst::KleeneStar(inner) => {
                let s = self.new_state();
                let e = self.new_state();
                let (is, ie) = self.build_fragment(inner)?;
                self.add_epsilon(s, is);
                self.add_epsilon(s, e); // skip
                self.add_epsilon(ie, is); // repeat
                self.add_epsilon(ie, e); // done
                Ok((s, e))
            }
            RegexAst::KleenePlus(inner) => {
                let s = self.new_state();
                let e = self.new_state();
                let (is, ie) = self.build_fragment(inner)?;
                self.add_epsilon(s, is);
                self.add_epsilon(ie, is); // repeat
                self.add_epsilon(ie, e); // done
                Ok((s, e))
            }
            RegexAst::Optional(inner) => {
                let s = self.new_state();
                let e = self.new_state();
                let (is, ie) = self.build_fragment(inner)?;
                self.add_epsilon(s, is);
                self.add_epsilon(s, e); // skip
                self.add_epsilon(ie, e);
                Ok((s, e))
            }
        }
    }

    fn finalize(
        self,
        start: usize,
        end: usize,
    ) -> WfaResult<WeightedFiniteAutomaton<BooleanSemiring>> {
        let mut wfa =
            WeightedFiniteAutomaton::<BooleanSemiring>::new(self.num_states, self.alphabet);
        wfa.initial_weights[start] = BooleanSemiring::one();
        wfa.final_weights[end] = BooleanSemiring::one();
        wfa.transitions = self.transitions;

        // Remove epsilon transitions to get a clean NFA.
        let clean = wfa.remove_epsilon();
        Ok(clean)
    }
}

// ---------------------------------------------------------------------------
// DOT export & Display
// ---------------------------------------------------------------------------

impl<S: Semiring + fmt::Display> WeightedFiniteAutomaton<S> {
    /// Export the WFA to Graphviz DOT format.
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph WFA {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=circle];\n");

        // Mark initial states with an invisible incoming arrow.
        for q in 0..self.num_states {
            if !self.initial_weights[q].is_zero() {
                let label = self
                    .state_label(q)
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| format!("q{}", q));
                dot.push_str(&format!(
                    "  __init_{} [shape=point, width=0, height=0];\n",
                    q
                ));
                dot.push_str(&format!(
                    "  __init_{} -> {} [label=\"{}\"];\n",
                    q, q, self.initial_weights[q]
                ));
            }
        }

        // Mark final states with double circles and show final weight.
        for q in 0..self.num_states {
            let label = self
                .state_label(q)
                .map(|l| l.to_string())
                .unwrap_or_else(|| format!("q{}", q));
            if !self.final_weights[q].is_zero() {
                dot.push_str(&format!(
                    "  {} [label=\"{}\\n(f={})\", shape=doublecircle];\n",
                    q, label, self.final_weights[q]
                ));
            } else {
                dot.push_str(&format!("  {} [label=\"{}\"];\n", q, label));
            }
        }

        // Transitions.
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                let sym_label = self
                    .alphabet
                    .symbol_at(sym)
                    .map(|s| format!("{}", s))
                    .unwrap_or_else(|| format!("{}", sym));
                for &(to, ref w) in &self.transitions[from][sym] {
                    dot.push_str(&format!(
                        "  {} -> {} [label=\"{}/{}\"];\n",
                        from, to, sym_label, w
                    ));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }
}

impl<S: Semiring + fmt::Display> fmt::Display for WeightedFiniteAutomaton<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "WFA({} states, {} symbols)", self.num_states, self.alphabet.size())?;
        writeln!(f, "  Alphabet: {}", self.alphabet)?;

        write!(f, "  Initial: [")?;
        for (i, w) in self.initial_weights.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", w)?;
        }
        writeln!(f, "]")?;

        write!(f, "  Final:   [")?;
        for (i, w) in self.final_weights.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", w)?;
        }
        writeln!(f, "]")?;

        writeln!(f, "  Transitions:")?;
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[from][sym] {
                    let sym_label = self
                        .alphabet
                        .symbol_at(sym)
                        .map(|s| format!("{}", s))
                        .unwrap_or_else(|| format!("{}", sym));
                    writeln!(f, "    {} --[{}]--> {} (w={})", from, sym_label, to, w)?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JSON serialization helpers
// ---------------------------------------------------------------------------

impl<S: Semiring + Serialize> WeightedFiniteAutomaton<S> {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl<S: Semiring + for<'de> Deserialize<'de>> WeightedFiniteAutomaton<S> {
    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// Additional utility methods
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Collect the set of states that have at least one outgoing transition.
    pub fn active_states(&self) -> HashSet<usize> {
        let mut active = HashSet::new();
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                if !self.transitions[from][sym].is_empty() {
                    active.insert(from);
                    break;
                }
            }
        }
        active
    }

    /// Collect the set of symbols that actually appear on transitions.
    pub fn used_symbols(&self) -> HashSet<usize> {
        let mut used = HashSet::new();
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                if !self.transitions[from][sym].is_empty() {
                    used.insert(sym);
                }
            }
        }
        used
    }

    /// Compute the in‑degree of every state.
    pub fn in_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.num_states];
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, _) in &self.transitions[from][sym] {
                    deg[to] += 1;
                }
            }
        }
        deg
    }

    /// Compute the out‑degree of every state.
    pub fn out_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.num_states];
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                deg[from] += self.transitions[from][sym].len();
            }
        }
        deg
    }

    /// Return a topological ordering of the states if the transition graph
    /// is a DAG, otherwise `None`.
    pub fn topological_order(&self) -> Option<Vec<usize>> {
        let n = self.num_states;
        let mut in_deg = vec![0usize; n];
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for from in 0..n {
            for sym in 0..self.alphabet.size() {
                for &(to, _) in &self.transitions[from][sym] {
                    if adj[from].insert(to) {
                        in_deg[to] += 1;
                    }
                }
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for q in 0..n {
            if in_deg[q] == 0 {
                queue.push_back(q);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(q) = queue.pop_front() {
            order.push(q);
            for &r in &adj[q] {
                in_deg[r] -= 1;
                if in_deg[r] == 0 {
                    queue.push_back(r);
                }
            }
        }

        if order.len() == n {
            Some(order)
        } else {
            None
        }
    }

    /// Whether the automaton's transition graph is acyclic.
    pub fn is_acyclic(&self) -> bool {
        self.topological_order().is_some()
    }

    /// Compute the shortest‑path length from any initial state to every
    /// other state (unweighted BFS distance).
    pub fn bfs_distances(&self) -> Vec<Option<usize>> {
        let n = self.num_states;
        let mut dist: Vec<Option<usize>> = vec![None; n];
        let mut queue = VecDeque::new();

        for q in 0..n {
            if !self.initial_weights[q].is_zero() {
                dist[q] = Some(0);
                queue.push_back(q);
            }
        }

        while let Some(q) = queue.pop_front() {
            let d = dist[q].unwrap();
            for sym in 0..self.alphabet.size() {
                for &(to, _) in &self.transitions[q][sym] {
                    if dist[to].is_none() {
                        dist[to] = Some(d + 1);
                        queue.push_back(to);
                    }
                }
            }
        }
        dist
    }

    /// Enumerate all strings (as sequences of symbol indices) accepted by
    /// this WFA up to the given length, along with their weights.
    ///
    /// Warning: the number of strings can be exponential.
    pub fn enumerate_strings(&self, max_length: usize) -> Vec<(Vec<usize>, S)> {
        let mut results = Vec::new();
        if self.num_states == 0 {
            return results;
        }

        // BFS / DFS with memoization wouldn't help in the general case, so
        // we just do DFS with backtracking.
        struct Dfs<'a, S: Semiring> {
            wfa: &'a WeightedFiniteAutomaton<S>,
            max_len: usize,
            results: Vec<(Vec<usize>, S)>,
            path: Vec<usize>,
        }

        impl<'a, S: Semiring> Dfs<'a, S> {
            fn search(&mut self, state_weights: Vec<S>) {
                // Collect weight for current path.
                let mut total = S::zero();
                for q in 0..self.wfa.num_states {
                    if !state_weights[q].is_zero() && !self.wfa.final_weights[q].is_zero() {
                        let c = state_weights[q].mul(&self.wfa.final_weights[q]);
                        total.add_assign(&c);
                    }
                }
                if !total.is_zero() {
                    self.results.push((self.path.clone(), total));
                }

                if self.path.len() >= self.max_len {
                    return;
                }

                for sym in 0..self.wfa.alphabet.size() {
                    // Skip epsilon in enumeration.
                    if let Some(s) = self.wfa.alphabet.symbol_at(sym) {
                        if *s == Symbol::Epsilon {
                            continue;
                        }
                    }

                    let mut next_weights = vec![S::zero(); self.wfa.num_states];
                    let mut any_nonzero = false;
                    for from in 0..self.wfa.num_states {
                        if state_weights[from].is_zero() {
                            continue;
                        }
                        for &(to, ref w) in &self.wfa.transitions[from][sym] {
                            let c = state_weights[from].mul(w);
                            next_weights[to].add_assign(&c);
                            any_nonzero = true;
                        }
                    }

                    if any_nonzero {
                        self.path.push(sym);
                        self.search(next_weights);
                        self.path.pop();
                    }
                }
            }
        }

        let mut dfs = Dfs {
            wfa: self,
            max_len: max_length,
            results: Vec::new(),
            path: Vec::new(),
        };

        dfs.search(self.initial_weights.clone());
        dfs.results
    }

    /// Apply a function to every transition weight.
    pub fn map_weights<F>(&self, f: F) -> Self
    where
        F: Fn(&S) -> S,
    {
        let mut wfa = Self::new(self.num_states, self.alphabet.clone());
        wfa.initial_weights = self.initial_weights.iter().map(&f).collect();
        wfa.final_weights = self.final_weights.iter().map(&f).collect();
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[from][sym] {
                    wfa.transitions[from][sym].push((to, f(w)));
                }
            }
        }
        wfa
    }

    /// Return the "support" automaton over BooleanSemiring: every non‑zero
    /// weight becomes `true`.
    pub fn support(&self) -> WeightedFiniteAutomaton<BooleanSemiring> {
        let mut wfa =
            WeightedFiniteAutomaton::<BooleanSemiring>::new(self.num_states, self.alphabet.clone());
        for q in 0..self.num_states {
            if !self.initial_weights[q].is_zero() {
                wfa.initial_weights[q] = BooleanSemiring::one();
            }
            if !self.final_weights[q].is_zero() {
                wfa.final_weights[q] = BooleanSemiring::one();
            }
        }
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[from][sym] {
                    if !w.is_zero() {
                        wfa.transitions[from][sym].push((to, BooleanSemiring::one()));
                    }
                }
            }
        }
        wfa
    }

    /// Merge parallel transitions (same from, symbol, to) by summing their
    /// weights.
    pub fn merge_parallel_transitions(&mut self) {
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                let trans = &mut self.transitions[from][sym];
                if trans.len() <= 1 {
                    continue;
                }
                let mut merged: BTreeMap<usize, S> = BTreeMap::new();
                for (to, w) in trans.drain(..) {
                    merged
                        .entry(to)
                        .and_modify(|existing: &mut S| existing.add_assign(&w))
                        .or_insert(w);
                }
                for (to, w) in merged {
                    if !w.is_zero() {
                        trans.push((to, w));
                    }
                }
            }
        }
        self.invalidate_cache();
    }

    /// Remove transitions whose weight is zero.
    pub fn remove_zero_transitions(&mut self) {
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                self.transitions[from][sym].retain(|(_, w)| !w.is_zero());
            }
        }
        self.invalidate_cache();
    }

    /// Copy another WFA's transitions into this one at a given state offset
    /// and symbol mapping.  Used internally by construction operations.
    fn embed_transitions(
        &mut self,
        other: &Self,
        state_offset: usize,
        sym_map: &[Option<usize>],
    ) {
        for from in 0..other.num_states {
            for sym in 0..other.alphabet.size() {
                if let Some(new_sym) = sym_map[sym] {
                    for &(to, ref w) in &other.transitions[from][sym] {
                        self.transitions[state_offset + from][new_sym]
                            .push((state_offset + to, w.clone()));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strongly‑connected components (used for cycle detection / analysis)
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Compute the strongly‑connected components using Tarjan's algorithm.
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.num_states;
        struct TarjanState {
            index_counter: usize,
            stack: Vec<usize>,
            on_stack: Vec<bool>,
            indices: Vec<Option<usize>>,
            lowlinks: Vec<usize>,
            sccs: Vec<Vec<usize>>,
        }

        let mut state = TarjanState {
            index_counter: 0,
            stack: Vec::new(),
            on_stack: vec![false; n],
            indices: vec![None; n],
            lowlinks: vec![0; n],
            sccs: Vec::new(),
        };

        fn strongconnect<S2: Semiring>(
            wfa: &WeightedFiniteAutomaton<S2>,
            v: usize,
            s: &mut TarjanState,
        ) {
            s.indices[v] = Some(s.index_counter);
            s.lowlinks[v] = s.index_counter;
            s.index_counter += 1;
            s.stack.push(v);
            s.on_stack[v] = true;

            for sym in 0..wfa.alphabet.size() {
                for &(w, _) in &wfa.transitions[v][sym] {
                    if s.indices[w].is_none() {
                        strongconnect(wfa, w, s);
                        s.lowlinks[v] = s.lowlinks[v].min(s.lowlinks[w]);
                    } else if s.on_stack[w] {
                        s.lowlinks[v] = s.lowlinks[v].min(s.indices[w].unwrap());
                    }
                }
            }

            if s.lowlinks[v] == s.indices[v].unwrap() {
                let mut scc = Vec::new();
                loop {
                    let w = s.stack.pop().unwrap();
                    s.on_stack[w] = false;
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                s.sccs.push(scc);
            }
        }

        for v in 0..n {
            if state.indices[v].is_none() {
                strongconnect(self, v, &mut state);
            }
        }

        state.sccs
    }

    /// Number of strongly‑connected components.
    pub fn num_sccs(&self) -> usize {
        self.strongly_connected_components().len()
    }
}

// ---------------------------------------------------------------------------
// Path enumeration (shortest / all paths)
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Find a shortest accepting path (by number of transitions) via BFS,
    /// and return the sequence of symbol indices, or `Err(NoPath)`.
    pub fn shortest_accepting_path(&self) -> WfaResult<Vec<usize>> {
        if self.num_states == 0 {
            return Err(WfaError::NoPath);
        }

        // BFS states: (state, path_so_far).
        let mut queue: VecDeque<(usize, Vec<usize>)> = VecDeque::new();
        let mut visited = HashSet::new();

        // Seed with initial states.
        for q in 0..self.num_states {
            if !self.initial_weights[q].is_zero() {
                // Check if initial state is also final (empty string).
                if !self.final_weights[q].is_zero() {
                    return Ok(Vec::new());
                }
                visited.insert(q);
                queue.push_back((q, Vec::new()));
            }
        }

        while let Some((state, path)) = queue.pop_front() {
            for sym in 0..self.alphabet.size() {
                for &(to, ref _w) in &self.transitions[state][sym] {
                    let mut new_path = path.clone();
                    new_path.push(sym);
                    if !self.final_weights[to].is_zero() {
                        return Ok(new_path);
                    }
                    if visited.insert(to) {
                        queue.push_back((to, new_path));
                    }
                }
            }
        }

        Err(WfaError::NoPath)
    }

    /// Find the k shortest accepting paths (by number of transitions).
    pub fn k_shortest_paths(&self, k: usize) -> Vec<(Vec<usize>, S)> {
        if self.num_states == 0 || k == 0 {
            return Vec::new();
        }

        let mut results: Vec<(Vec<usize>, S)> = Vec::new();
        // Priority‑queue BFS: (path_length, state, path, accumulated_weight).
        // We use a Vec and sort to simulate a min‑heap by path length.
        let mut queue: VecDeque<(usize, Vec<usize>, Vec<S>)> = VecDeque::new();

        // Seed: initial forward vector.
        queue.push_back((0, Vec::new(), self.initial_weights.clone()));

        let mut expansions = 0usize;
        let max_expansions = 100_000; // Safety limit.

        while let Some((_state, path, forward)) = queue.pop_front() {
            expansions += 1;
            if expansions > max_expansions {
                break;
            }

            // Check if this path is accepting.
            let mut total = S::zero();
            for q in 0..self.num_states {
                if !forward[q].is_zero() && !self.final_weights[q].is_zero() {
                    let c = forward[q].mul(&self.final_weights[q]);
                    total.add_assign(&c);
                }
            }
            if !total.is_zero() {
                results.push((path.clone(), total));
                if results.len() >= k {
                    break;
                }
            }

            // Expand.
            for sym in 0..self.alphabet.size() {
                let mut next_forward = vec![S::zero(); self.num_states];
                let mut any = false;
                for from in 0..self.num_states {
                    if forward[from].is_zero() {
                        continue;
                    }
                    for &(to, ref w) in &self.transitions[from][sym] {
                        let c = forward[from].mul(w);
                        next_forward[to].add_assign(&c);
                        any = true;
                    }
                }
                if any {
                    let mut new_path = path.clone();
                    new_path.push(sym);
                    queue.push_back((0, new_path, next_forward));
                }
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Morphism / homomorphism application
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Apply a symbol‑to‑symbol homomorphism.
    /// `mapping[old_sym_idx] = new_sym_idx` in the target alphabet.
    pub fn apply_symbol_map(
        &self,
        target_alphabet: &Alphabet,
        mapping: &[usize],
    ) -> WfaResult<Self> {
        if mapping.len() != self.alphabet.size() {
            return Err(WfaError::DimensionMismatch {
                expected: self.alphabet.size(),
                got: mapping.len(),
            });
        }

        let mut wfa = Self::new(self.num_states, target_alphabet.clone());
        wfa.initial_weights = self.initial_weights.clone();
        wfa.final_weights = self.final_weights.clone();

        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                let new_sym = mapping[sym];
                if new_sym >= target_alphabet.size() {
                    return Err(WfaError::InvalidSymbol(new_sym, target_alphabet.size()));
                }
                for &(to, ref w) in &self.transitions[from][sym] {
                    wfa.transitions[from][new_sym].push((to, w.clone()));
                }
            }
        }

        Ok(wfa)
    }

    /// Project the WFA onto a sub‑alphabet (keep only transitions on
    /// symbols in `keep`).
    pub fn project(&self, keep: &HashSet<usize>) -> Self {
        let mut new_alpha = Alphabet::new();
        let mut old_to_new: Vec<Option<usize>> = vec![None; self.alphabet.size()];
        for &sym in keep {
            if sym < self.alphabet.size() {
                if let Some(s) = self.alphabet.symbol_at(sym) {
                    let new_idx = new_alpha.insert(s.clone());
                    old_to_new[sym] = Some(new_idx);
                }
            }
        }

        let mut wfa = Self::new(self.num_states, new_alpha);
        wfa.initial_weights = self.initial_weights.clone();
        wfa.final_weights = self.final_weights.clone();

        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                if let Some(new_sym) = old_to_new[sym] {
                    for &(to, ref w) in &self.transitions[from][sym] {
                        wfa.transitions[from][new_sym].push((to, w.clone()));
                    }
                }
            }
        }

        wfa
    }
}

// ---------------------------------------------------------------------------
// Equivalence check (basic)
// ---------------------------------------------------------------------------

impl<S: Semiring + PartialEq> WeightedFiniteAutomaton<S> {
    /// Heuristic equivalence check: test both WFAs on all strings up to
    /// `max_len` and compare weights.
    ///
    /// This is NOT a complete algorithm for arbitrary semirings; it is
    /// useful for testing.
    pub fn equiv_check_bounded(&self, other: &Self, max_len: usize) -> bool {
        // We need a common alphabet.
        let common = self.alphabet.union(&other.alphabet);
        let self_map = self.alphabet.index_mapping(&common);
        let other_map = other.alphabet.index_mapping(&common);

        fn gen_strings(alpha_size: usize, max_len: usize) -> Vec<Vec<usize>> {
            let mut result = vec![vec![]]; // empty string
            let mut frontier = vec![vec![]];
            for _ in 0..max_len {
                let mut next_frontier = Vec::new();
                for s in &frontier {
                    for a in 0..alpha_size {
                        let mut ns = s.clone();
                        ns.push(a);
                        next_frontier.push(ns.clone());
                        result.push(ns);
                    }
                }
                frontier = next_frontier;
            }
            result
        }

        // Translate a string in the common alphabet to the original.
        fn translate(
            string: &[usize],
            mapping: &[Option<usize>],
            alpha_size: usize,
        ) -> Option<Vec<usize>> {
            // Build reverse mapping.
            let mut rev: Vec<Option<usize>> = vec![None; alpha_size];
            for (old, new_opt) in mapping.iter().enumerate() {
                if let Some(new) = new_opt {
                    rev[*new] = Some(old);
                }
            }
            string.iter().map(|&s| rev.get(s).copied().flatten()).collect()
        }

        let strings = gen_strings(common.size(), max_len);
        for s in &strings {
            let self_input = match translate(s, &self_map, common.size()) {
                Some(v) => v,
                None => continue,
            };
            let other_input = match translate(s, &other_map, common.size()) {
                Some(v) => v,
                None => continue,
            };
            let w1 = self.compute_weight(&self_input);
            let w2 = other.compute_weight(&other_input);
            if w1 != w2 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Statistics / debug helpers
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Summary statistics.
    pub fn stats(&self) -> WfaStats {
        let trans_count = self.num_transitions();
        let initial_count = self
            .initial_weights
            .iter()
            .filter(|w| !w.is_zero())
            .count();
        let final_count = self
            .final_weights
            .iter()
            .filter(|w| !w.is_zero())
            .count();
        let det = self.check_deterministic();
        let trim = self.is_trim();
        WfaStats {
            num_states: self.num_states,
            alphabet_size: self.alphabet.size(),
            num_transitions: trans_count,
            num_initial: initial_count,
            num_final: final_count,
            is_deterministic: det,
            is_trim: trim,
        }
    }
}

/// Summary statistics for a WFA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfaStats {
    pub num_states: usize,
    pub alphabet_size: usize,
    pub num_transitions: usize,
    pub num_initial: usize,
    pub num_final: usize,
    pub is_deterministic: bool,
    pub is_trim: bool,
}

impl fmt::Display for WfaStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "States: {}, Alphabet: {}, Transitions: {}, Initial: {}, Final: {}, Det: {}, Trim: {}",
            self.num_states,
            self.alphabet_size,
            self.num_transitions,
            self.num_initial,
            self.num_final,
            self.is_deterministic,
            self.is_trim,
        )
    }
}

// ---------------------------------------------------------------------------
// Builder pattern
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a `WeightedFiniteAutomaton`.
pub struct WfaBuilder<S: Semiring> {
    pub num_states: usize,
    pub alphabet: Alphabet,
    pub initial_weights: Vec<S>,
    pub final_weights: Vec<S>,
    transitions: Vec<Transition<S>>,
    pub state_labels: Vec<Option<String>>,
}

impl<S: Semiring> WfaBuilder<S> {
    /// Start building a WFA with `num_states` states.
    pub fn new(num_states: usize, alphabet: Alphabet) -> Self {
        WfaBuilder {
            num_states,
            alphabet,
            initial_weights: vec![S::zero(); num_states],
            final_weights: vec![S::zero(); num_states],
            transitions: Vec::new(),
            state_labels: vec![None; num_states],
        }
    }

    pub fn initial(mut self, state: usize, weight: S) -> Self {
        self.initial_weights[state] = weight;
        self
    }

    pub fn final_state(mut self, state: usize, weight: S) -> Self {
        self.final_weights[state] = weight;
        self
    }

    pub fn transition(mut self, from: usize, symbol: usize, to: usize, weight: S) -> Self {
        self.transitions
            .push(Transition::new(from, to, symbol, weight));
        self
    }

    pub fn label(mut self, state: usize, label: &str) -> Self {
        self.state_labels[state] = Some(label.to_string());
        self
    }

    pub fn build(self) -> WfaResult<WeightedFiniteAutomaton<S>> {
        let mut wfa = WeightedFiniteAutomaton::from_transitions(
            self.num_states,
            self.alphabet,
            self.initial_weights,
            self.final_weights,
            self.transitions,
        )?;
        wfa.state_labels = self.state_labels;
        Ok(wfa)
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers for common semirings
// ---------------------------------------------------------------------------

impl WeightedFiniteAutomaton<BooleanSemiring> {
    /// Convert from a Boolean WFA to a counting WFA (true→1, false→0).
    pub fn to_counting(&self) -> WeightedFiniteAutomaton<CountingSemiring> {
        let convert = |b: &BooleanSemiring| -> CountingSemiring {
            if b.is_zero() {
                CountingSemiring::zero()
            } else {
                CountingSemiring::one()
            }
        };

        let mut wfa =
            WeightedFiniteAutomaton::<CountingSemiring>::new(self.num_states, self.alphabet.clone());
        for q in 0..self.num_states {
            wfa.initial_weights[q] = convert(&self.initial_weights[q]);
            wfa.final_weights[q] = convert(&self.final_weights[q]);
        }
        for from in 0..self.num_states {
            for sym in 0..self.alphabet.size() {
                for &(to, ref w) in &self.transitions[from][sym] {
                    wfa.transitions[from][sym].push((to, convert(w)));
                }
            }
        }
        wfa
    }
}

impl WeightedFiniteAutomaton<CountingSemiring> {
    /// Convert from a counting WFA to a Boolean WFA (nonzero → true).
    pub fn to_boolean(&self) -> WeightedFiniteAutomaton<BooleanSemiring> {
        self.support()
    }
}

// ---------------------------------------------------------------------------
// Misc: composition with weight transformation
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    /// Multiply all transition weights by a scalar.
    pub fn scale_weights(&self, scalar: &S) -> Self {
        self.map_weights(|w| w.mul(scalar))
    }

    /// Shift all final weights by adding a constant.
    pub fn shift_final_weights(&self, delta: &S) -> Self {
        let mut wfa = self.clone();
        for q in 0..wfa.num_states {
            wfa.final_weights[q].add_assign(delta);
        }
        wfa
    }
}

// ---------------------------------------------------------------------------
// Deep equality (structural)
// ---------------------------------------------------------------------------

impl<S: Semiring + PartialEq> PartialEq for WeightedFiniteAutomaton<S> {
    fn eq(&self, other: &Self) -> bool {
        if self.num_states != other.num_states {
            return false;
        }
        if self.alphabet != other.alphabet {
            return false;
        }
        if self.initial_weights != other.initial_weights {
            return false;
        }
        if self.final_weights != other.final_weights {
            return false;
        }
        self.transitions == other.transitions
    }
}

impl<S: Semiring + Eq> Eq for WeightedFiniteAutomaton<S> {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: build a simple two‑state Boolean WFA accepting "ab" ────
    fn make_ab_wfa() -> WeightedFiniteAutomaton<BooleanSemiring> {
        // States: 0 (init) --a--> 1 --b--> 2 (final)
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a_idx = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b_idx = alpha.index_of(&Symbol::Char('b')).unwrap();

        WfaBuilder::new(3, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(2, BooleanSemiring::one())
            .transition(0, a_idx, 1, BooleanSemiring::one())
            .transition(1, b_idx, 2, BooleanSemiring::one())
            .build()
            .unwrap()
    }

    fn sym_indices(alpha: &Alphabet, chars: &[char]) -> Vec<usize> {
        chars
            .iter()
            .map(|c| alpha.index_of(&Symbol::Char(*c)).unwrap())
            .collect()
    }

    // ── Construction & basic weight computation ────────────────────────

    #[test]
    fn test_basic_construction() {
        let wfa = make_ab_wfa();
        assert_eq!(wfa.state_count(), 3);
        assert_eq!(wfa.alphabet().size(), 2);
    }

    #[test]
    fn test_accepts_ab() {
        let wfa = make_ab_wfa();
        let input = sym_indices(wfa.alphabet(), &['a', 'b']);
        assert!(wfa.accepts(&input));
    }

    #[test]
    fn test_rejects_ba() {
        let wfa = make_ab_wfa();
        let input = sym_indices(wfa.alphabet(), &['b', 'a']);
        assert!(!wfa.accepts(&input));
    }

    #[test]
    fn test_rejects_empty() {
        let wfa = make_ab_wfa();
        assert!(!wfa.accepts(&[]));
    }

    #[test]
    fn test_rejects_a_only() {
        let wfa = make_ab_wfa();
        let input = sym_indices(wfa.alphabet(), &['a']);
        assert!(!wfa.accepts(&input));
    }

    #[test]
    fn test_epsilon_wfa() {
        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::epsilon_wfa(BooleanSemiring::one());
        assert!(wfa.accepts(&[]));
        assert_eq!(wfa.state_count(), 1);
    }

    #[test]
    fn test_empty_wfa() {
        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::empty();
        assert_eq!(wfa.state_count(), 0);
        assert!(!wfa.accepts(&[]));
        assert!(wfa.is_empty());
    }

    #[test]
    fn test_single_symbol() {
        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::single_symbol(
            Symbol::Char('x'),
            BooleanSemiring::one(),
        );
        let x_idx = wfa.alphabet().index_of(&Symbol::Char('x')).unwrap();
        assert!(wfa.accepts(&[x_idx]));
        assert!(!wfa.accepts(&[]));
    }

    // ── Counting semiring ──────────────────────────────────────────────

    #[test]
    fn test_counting_semiring_weight() {
        // WFA with two paths from 0 to 2 on the same input "a":
        //   0 --a(1)--> 1 --a(1)--> 2
        //   0 --a(1)--> 3 --a(1)--> 2
        // Weight of "aa" should be 2 (two paths).
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<CountingSemiring>::new(4, alpha)
            .initial(0, CountingSemiring::one())
            .final_state(2, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .transition(1, a, 2, CountingSemiring::one())
            .transition(0, a, 3, CountingSemiring::one())
            .transition(3, a, 2, CountingSemiring::one())
            .build()
            .unwrap();

        let input = vec![a, a];
        let w = wfa.compute_weight(&input);
        // CountingSemiring: two paths each of weight 1, total = 2
        assert_eq!(w, CountingSemiring::one().add(&CountingSemiring::one()));
    }

    // ── Forward / backward vectors ─────────────────────────────────────

    #[test]
    fn test_forward_vectors() {
        let wfa = make_ab_wfa();
        let input = sym_indices(wfa.alphabet(), &['a', 'b']);
        let fwd = wfa.forward_vectors(&input);

        // fwd[0] = initial = [1, 0, 0]
        assert!(!fwd[0][0].is_zero());
        assert!(fwd[0][1].is_zero());
        assert!(fwd[0][2].is_zero());

        // fwd[1] = after 'a' = [0, 1, 0]
        assert!(fwd[1][0].is_zero());
        assert!(!fwd[1][1].is_zero());
        assert!(fwd[1][2].is_zero());

        // fwd[2] = after 'ab' = [0, 0, 1]
        assert!(fwd[2][0].is_zero());
        assert!(fwd[2][1].is_zero());
        assert!(!fwd[2][2].is_zero());
    }

    #[test]
    fn test_backward_vectors() {
        let wfa = make_ab_wfa();
        let input = sym_indices(wfa.alphabet(), &['a', 'b']);
        let bwd = wfa.backward_vectors(&input);

        // bwd[2] = final weights = [0, 0, 1]
        assert!(bwd[2][0].is_zero());
        assert!(bwd[2][1].is_zero());
        assert!(!bwd[2][2].is_zero());

        // bwd[1] = from state 1, reading 'b', can reach final
        assert!(bwd[1][0].is_zero());
        assert!(!bwd[1][1].is_zero());
        assert!(bwd[1][2].is_zero()); // state 2 has no outgoing 'b' to final

        // bwd[0] = from state 0, reading "ab", can reach final
        assert!(!bwd[0][0].is_zero());
    }

    #[test]
    fn test_forward_backward_consistency() {
        // For any position t, Σ_q forward[t][q] · backward[t][q] should
        // equal the total weight.
        let wfa = make_ab_wfa();
        let input = sym_indices(wfa.alphabet(), &['a', 'b']);
        let fwd = wfa.forward_vectors(&input);
        let bwd = wfa.backward_vectors(&input);
        let total = wfa.compute_weight(&input);

        for t in 0..=input.len() {
            let mut sum = BooleanSemiring::zero();
            for q in 0..wfa.state_count() {
                let c = fwd[t][q].mul(&bwd[t][q]);
                sum.add_assign(&c);
            }
            assert_eq!(sum, total);
        }
    }

    // ── Matrix weight computation ──────────────────────────────────────

    #[test]
    fn test_weight_matrix_computation() {
        let wfa = make_ab_wfa();
        let input = sym_indices(wfa.alphabet(), &['a', 'b']);
        let w1 = wfa.compute_weight(&input);
        let w2 = wfa.compute_weight_matrix(&input);
        assert_eq!(w1, w2);
    }

    // ── Union ──────────────────────────────────────────────────────────

    #[test]
    fn test_union() {
        let wfa1 = make_ab_wfa(); // accepts "ab"
        // Build a WFA that accepts "ba".
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a_idx = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b_idx = alpha.index_of(&Symbol::Char('b')).unwrap();

        let wfa2 = WfaBuilder::new(3, alpha.clone())
            .initial(0, BooleanSemiring::one())
            .final_state(2, BooleanSemiring::one())
            .transition(0, b_idx, 1, BooleanSemiring::one())
            .transition(1, a_idx, 2, BooleanSemiring::one())
            .build()
            .unwrap();

        let u = wfa1.union(&wfa2).unwrap();

        // Alphabet of union.
        let ua = u.alphabet();
        let ua_a = ua.index_of(&Symbol::Char('a')).unwrap();
        let ua_b = ua.index_of(&Symbol::Char('b')).unwrap();

        assert!(u.accepts(&[ua_a, ua_b])); // "ab"
        assert!(u.accepts(&[ua_b, ua_a])); // "ba"
        assert!(!u.accepts(&[ua_a, ua_a])); // "aa" not accepted
    }

    // ── Concatenation ──────────────────────────────────────────────────

    #[test]
    fn test_concatenation() {
        // A accepts "a", B accepts "b".  Concat should accept "ab".
        let wfa_a = WeightedFiniteAutomaton::<BooleanSemiring>::single_symbol(
            Symbol::Char('a'),
            BooleanSemiring::one(),
        );
        let wfa_b = WeightedFiniteAutomaton::<BooleanSemiring>::single_symbol(
            Symbol::Char('b'),
            BooleanSemiring::one(),
        );
        let cat = wfa_a.concatenation(&wfa_b).unwrap();

        let ca = cat.alphabet();
        let ca_a = ca.index_of(&Symbol::Char('a')).unwrap();
        let ca_b = ca.index_of(&Symbol::Char('b')).unwrap();

        assert!(cat.accepts(&[ca_a, ca_b])); // "ab"
        assert!(!cat.accepts(&[ca_a])); // "a" alone
        assert!(!cat.accepts(&[ca_b])); // "b" alone
        assert!(!cat.accepts(&[])); // empty
    }

    // ── Kleene star ────────────────────────────────────────────────────

    #[test]
    fn test_kleene_star() {
        let wfa_a = WeightedFiniteAutomaton::<BooleanSemiring>::single_symbol(
            Symbol::Char('a'),
            BooleanSemiring::one(),
        );
        let star = wfa_a.kleene_star();

        let sa = star.alphabet();
        let sa_a = sa.index_of(&Symbol::Char('a')).unwrap();

        // a* accepts: ε, a, aa, aaa, ...
        assert!(star.accepts(&[]));
        assert!(star.accepts(&[sa_a]));
        assert!(star.accepts(&[sa_a, sa_a]));
        assert!(star.accepts(&[sa_a, sa_a, sa_a]));
    }

    #[test]
    fn test_kleene_star_empty() {
        let empty = WeightedFiniteAutomaton::<BooleanSemiring>::empty();
        let star = empty.kleene_star();
        assert!(star.accepts(&[])); // ε is always in L*
    }

    // ── Intersection ───────────────────────────────────────────────────

    #[test]
    fn test_intersection() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a_idx = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b_idx = alpha.index_of(&Symbol::Char('b')).unwrap();

        // WFA1: accepts strings starting with 'a'.
        let wfa1 = WfaBuilder::new(2, alpha.clone())
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a_idx, 1, BooleanSemiring::one())
            .transition(1, a_idx, 1, BooleanSemiring::one())
            .transition(1, b_idx, 1, BooleanSemiring::one())
            .build()
            .unwrap();

        // WFA2: accepts strings ending with 'b'.
        let wfa2 = WfaBuilder::new(2, alpha.clone())
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a_idx, 0, BooleanSemiring::one())
            .transition(0, b_idx, 1, BooleanSemiring::one())
            .transition(1, a_idx, 0, BooleanSemiring::one())
            .transition(1, b_idx, 1, BooleanSemiring::one())
            .build()
            .unwrap();

        let inter = wfa1.intersection(&wfa2).unwrap();
        let ia = inter.alphabet();
        let ia_a = ia.index_of(&Symbol::Char('a')).unwrap();
        let ia_b = ia.index_of(&Symbol::Char('b')).unwrap();

        assert!(inter.accepts(&[ia_a, ia_b])); // "ab" starts with a, ends with b
        assert!(inter.accepts(&[ia_a, ia_a, ia_b])); // "aab"
        assert!(!inter.accepts(&[ia_b, ia_a])); // "ba" doesn't start with a
        assert!(!inter.accepts(&[ia_a, ia_a])); // "aa" doesn't end with b
    }

    // ── Complement ─────────────────────────────────────────────────────

    #[test]
    fn test_complement() {
        let wfa = make_ab_wfa();
        let comp = wfa.complement();

        let ca = comp.alphabet();
        let ca_a = ca.index_of(&Symbol::Char('a')).unwrap();
        let ca_b = ca.index_of(&Symbol::Char('b')).unwrap();

        // "ab" was accepted, should be rejected in complement.
        assert!(!comp.accepts(&[ca_a, ca_b]));
        // Other strings should be accepted.
        assert!(comp.accepts(&[])); // ε
        assert!(comp.accepts(&[ca_a])); // "a"
    }

    // ── Reverse ────────────────────────────────────────────────────────

    #[test]
    fn test_reverse() {
        let wfa = make_ab_wfa(); // accepts "ab"
        let rev = wfa.reverse();

        let ra = rev.alphabet();
        let ra_a = ra.index_of(&Symbol::Char('a')).unwrap();
        let ra_b = ra.index_of(&Symbol::Char('b')).unwrap();

        assert!(rev.accepts(&[ra_b, ra_a])); // "ba" is reverse of "ab"
        assert!(!rev.accepts(&[ra_a, ra_b])); // "ab" not in reverse
    }

    // ── Determinization ────────────────────────────────────────────────

    #[test]
    fn test_determinize_already_det() {
        let wfa = make_ab_wfa();
        assert!(wfa.is_deterministic());
        let det = wfa.determinize();
        assert!(det.is_deterministic());

        let da = det.alphabet();
        let da_a = da.index_of(&Symbol::Char('a')).unwrap();
        let da_b = da.index_of(&Symbol::Char('b')).unwrap();
        assert!(det.accepts(&[da_a, da_b]));
        assert!(!det.accepts(&[da_b, da_a]));
    }

    #[test]
    fn test_determinize_nondeterministic() {
        // NFA: 0 --a--> 1, 0 --a--> 2, 1 final, 2 final.
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(3, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .final_state(2, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(0, a, 2, BooleanSemiring::one())
            .build()
            .unwrap();

        assert!(!wfa.is_deterministic());
        let det = wfa.determinize();
        assert!(det.is_deterministic());

        let da = det.alphabet();
        let da_a = da.index_of(&Symbol::Char('a')).unwrap();
        assert!(det.accepts(&[da_a]));
        assert!(!det.accepts(&[]));
    }

    // ── Epsilon removal ────────────────────────────────────────────────

    #[test]
    fn test_epsilon_removal() {
        // Build WFA with epsilon: 0 --ε--> 1 --a--> 2 (final).
        let alpha = Alphabet::from_chars(&['a']).with_epsilon();
        let eps_idx = alpha.epsilon_index().unwrap();
        let a_idx = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(3, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(2, BooleanSemiring::one())
            .transition(0, eps_idx, 1, BooleanSemiring::one())
            .transition(1, a_idx, 2, BooleanSemiring::one())
            .build()
            .unwrap();

        assert!(wfa.has_epsilon_transitions());
        let clean = wfa.remove_epsilon();
        assert!(!clean.has_epsilon_transitions());

        // Should still accept "a".
        let ca = clean.alphabet();
        let ca_a = ca.index_of(&Symbol::Char('a')).unwrap();
        assert!(clean.accepts(&[ca_a]));
    }

    #[test]
    fn test_epsilon_removal_chain() {
        // 0 --ε--> 1 --ε--> 2 --a--> 3 (final)
        let alpha = Alphabet::from_chars(&['a']).with_epsilon();
        let eps_idx = alpha.epsilon_index().unwrap();
        let a_idx = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(4, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(3, BooleanSemiring::one())
            .transition(0, eps_idx, 1, BooleanSemiring::one())
            .transition(1, eps_idx, 2, BooleanSemiring::one())
            .transition(2, a_idx, 3, BooleanSemiring::one())
            .build()
            .unwrap();

        let clean = wfa.remove_epsilon();
        let ca = clean.alphabet();
        let ca_a = ca.index_of(&Symbol::Char('a')).unwrap();
        assert!(clean.accepts(&[ca_a]));
    }

    // ── Trim ───────────────────────────────────────────────────────────

    #[test]
    fn test_trim_removes_unreachable() {
        // State 3 is unreachable.
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(4, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            // State 2 → 3 exists but 2 is not reachable from 0.
            .transition(2, a, 3, BooleanSemiring::one())
            .build()
            .unwrap();

        assert!(!wfa.is_trim());
        let trimmed = wfa.trim();
        assert!(trimmed.is_trim());
        assert_eq!(trimmed.state_count(), 2); // only 0 and 1
    }

    #[test]
    fn test_trim_removes_unproductive() {
        // State 2 has no path to a final state.
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(3, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(0, b, 2, BooleanSemiring::one())
            // State 2 has no outgoing to anything final.
            .build()
            .unwrap();

        let trimmed = wfa.trim();
        // State 2 is reachable but not productive → removed.
        assert_eq!(trimmed.state_count(), 2);
    }

    // ── Product construction ───────────────────────────────────────────

    #[test]
    fn test_product_construction_counting() {
        // Two WFAs over {'a'} that each count paths.
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        // WFA1: 1 path for "a"
        let wfa1 = WfaBuilder::<CountingSemiring>::new(2, alpha.clone())
            .initial(0, CountingSemiring::one())
            .final_state(1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .build()
            .unwrap();

        // WFA2: 1 path for "a"
        let wfa2 = WfaBuilder::<CountingSemiring>::new(2, alpha.clone())
            .initial(0, CountingSemiring::one())
            .final_state(1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .build()
            .unwrap();

        let prod = wfa1.intersection(&wfa2).unwrap();
        let pa = prod.alphabet();
        let pa_a = pa.index_of(&Symbol::Char('a')).unwrap();

        let w = prod.compute_weight(&[pa_a]);
        // Product should give 1*1 = 1.
        assert_eq!(w, CountingSemiring::one());
    }

    // ── Tropical semiring ──────────────────────────────────────────────

    #[test]
    fn test_tropical_semiring() {
        // Shortest‑path computation: tropical add = min, mul = +.
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        // Two paths from 0 to 2 for "ab":
        //   Path 1: 0 --a(3)--> 1 --b(2)--> 2  total = 3+2 = 5
        //   Path 2: 0 --a(1)--> 3 --b(1)--> 2  total = 1+1 = 2
        let wfa = WfaBuilder::<TropicalSemiring>::new(4, alpha)
            .initial(0, TropicalSemiring::one()) // one() = 0 in tropical
            .final_state(2, TropicalSemiring::one())
            .transition(0, a, 1, TropicalSemiring::from_value(3.0))
            .transition(1, b, 2, TropicalSemiring::from_value(2.0))
            .transition(0, a, 3, TropicalSemiring::from_value(1.0))
            .transition(3, b, 2, TropicalSemiring::from_value(1.0))
            .build()
            .unwrap();

        let input = vec![a, b];
        let w = wfa.compute_weight(&input);
        // min(5, 2) = 2
        assert_eq!(w, TropicalSemiring::from_value(2.0));
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_single_state_accepting() {
        let alpha = Alphabet::from_chars(&['a']);
        let wfa = WfaBuilder::<BooleanSemiring>::new(1, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(0, BooleanSemiring::one())
            .build()
            .unwrap();

        assert!(wfa.accepts(&[]));
        assert!(!wfa.accepts(&[0])); // no transitions
    }

    #[test]
    fn test_self_loop() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(1, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(0, BooleanSemiring::one())
            .transition(0, a, 0, BooleanSemiring::one())
            .build()
            .unwrap();

        assert!(wfa.accepts(&[]));
        assert!(wfa.accepts(&[a]));
        assert!(wfa.accepts(&[a, a, a, a]));
    }

    #[test]
    fn test_universal_wfa() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::universal(
            alpha.clone(),
            BooleanSemiring::one(),
        );

        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        assert!(wfa.accepts(&[]));
        assert!(wfa.accepts(&[a]));
        assert!(wfa.accepts(&[b]));
        assert!(wfa.accepts(&[a, b, a, b]));
    }

    #[test]
    fn test_num_transitions() {
        let wfa = make_ab_wfa();
        assert_eq!(wfa.num_transitions(), 2); // a→1, b→2
    }

    // ── DOT export ─────────────────────────────────────────────────────

    #[test]
    fn test_to_dot() {
        let wfa = make_ab_wfa();
        let dot = wfa.to_dot();
        assert!(dot.contains("digraph WFA"));
        assert!(dot.contains("doublecircle"));
    }

    // ── Enumerate strings ──────────────────────────────────────────────

    #[test]
    fn test_enumerate_strings() {
        let wfa = make_ab_wfa();
        let strings = wfa.enumerate_strings(3);
        let alpha = wfa.alphabet();
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        assert_eq!(strings.len(), 1);
        assert_eq!(strings[0].0, vec![a, b]);
    }

    // ── Reachable / productive / is_empty ──────────────────────────────

    #[test]
    fn test_reachable_states() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(4, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(2, a, 3, BooleanSemiring::one())
            .build()
            .unwrap();

        let reachable = wfa.reachable_states();
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
        assert!(!reachable.contains(&2));
        assert!(!reachable.contains(&3));
    }

    #[test]
    fn test_productive_states() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(3, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(0, a, 2, BooleanSemiring::one())
            .build()
            .unwrap();

        let productive = wfa.productive_states();
        assert!(productive.contains(&0));
        assert!(productive.contains(&1));
        assert!(!productive.contains(&2)); // 2 cannot reach final
    }

    #[test]
    fn test_is_empty_accepts_nothing() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(2, alpha)
            .initial(0, BooleanSemiring::one())
            // No final state!
            .transition(0, a, 1, BooleanSemiring::one())
            .build()
            .unwrap();

        assert!(wfa.is_empty());
    }

    // ── SCC ────────────────────────────────────────────────────────────

    #[test]
    fn test_sccs() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        // Cycle: 0 → 1 → 0, plus 2 isolated.
        let wfa = WfaBuilder::<BooleanSemiring>::new(3, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(0, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(1, a, 0, BooleanSemiring::one())
            .build()
            .unwrap();

        let sccs = wfa.strongly_connected_components();
        // Should have 2 SCCs: {0,1} and {2}.
        assert_eq!(sccs.len(), 2);
    }

    // ── Topological order & acyclicity ─────────────────────────────────

    #[test]
    fn test_acyclic_dag() {
        let wfa = make_ab_wfa();
        assert!(wfa.is_acyclic());
        let order = wfa.topological_order().unwrap();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_cyclic() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(2, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(0, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(1, a, 0, BooleanSemiring::one())
            .build()
            .unwrap();

        assert!(!wfa.is_acyclic());
        assert!(wfa.topological_order().is_none());
    }

    // ── Shortest path ──────────────────────────────────────────────────

    #[test]
    fn test_shortest_accepting_path() {
        let wfa = make_ab_wfa();
        let path = wfa.shortest_accepting_path().unwrap();
        let alpha = wfa.alphabet();
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();
        assert_eq!(path, vec![a, b]);
    }

    #[test]
    fn test_shortest_path_empty_string() {
        let wfa =
            WeightedFiniteAutomaton::<BooleanSemiring>::epsilon_wfa(BooleanSemiring::one());
        let path = wfa.shortest_accepting_path().unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn test_shortest_path_no_path() {
        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::empty();
        assert!(wfa.shortest_accepting_path().is_err());
    }

    // ── k shortest paths ───────────────────────────────────────────────

    #[test]
    fn test_k_shortest_paths() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(2, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(0, b, 0, BooleanSemiring::one())
            .build()
            .unwrap();

        let paths = wfa.k_shortest_paths(3);
        assert!(paths.len() >= 2); // at least ε and "a"
    }

    // ── Merge parallel transitions ─────────────────────────────────────

    #[test]
    fn test_merge_parallel() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let mut wfa = WfaBuilder::<CountingSemiring>::new(2, alpha)
            .initial(0, CountingSemiring::one())
            .final_state(1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .build()
            .unwrap();

        assert_eq!(wfa.num_transitions(), 2);
        wfa.merge_parallel_transitions();
        assert_eq!(wfa.num_transitions(), 1);

        // Weight should be 2 (sum of the two parallel transitions).
        let w = wfa.compute_weight(&[a]);
        let two = CountingSemiring::one().add(&CountingSemiring::one());
        assert_eq!(w, two);
    }

    // ── Builder pattern ────────────────────────────────────────────────

    #[test]
    fn test_builder() {
        let alpha = Alphabet::from_chars(&['x']);
        let x = alpha.index_of(&Symbol::Char('x')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(2, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, x, 1, BooleanSemiring::one())
            .label(0, "start")
            .label(1, "end")
            .build()
            .unwrap();

        assert_eq!(wfa.state_label(0), Some("start"));
        assert_eq!(wfa.state_label(1), Some("end"));
        assert!(wfa.accepts(&[x]));
    }

    // ── Rename states ──────────────────────────────────────────────────

    #[test]
    fn test_rename_states() {
        let wfa = make_ab_wfa();
        // Reverse the state numbering: 0→2, 1→1, 2→0
        let mapping = vec![2, 1, 0];
        let renamed = wfa.rename_states(&mapping);
        assert_eq!(renamed.state_count(), 3);

        // State 2 should now be initial, state 0 should be final.
        assert!(!renamed.initial_weights()[2].is_zero());
        assert!(!renamed.final_weights()[0].is_zero());
    }

    // ── Subautomaton ───────────────────────────────────────────────────

    #[test]
    fn test_subautomaton() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(4, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .final_state(3, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(0, b, 2, BooleanSemiring::one())
            .transition(2, a, 3, BooleanSemiring::one())
            .build()
            .unwrap();

        let states: HashSet<usize> = [0, 1].iter().copied().collect();
        let sub = wfa.subautomaton(&states);
        assert_eq!(sub.state_count(), 2);
    }

    // ── BFS distances ──────────────────────────────────────────────────

    #[test]
    fn test_bfs_distances() {
        let wfa = make_ab_wfa();
        let dist = wfa.bfs_distances();
        assert_eq!(dist[0], Some(0));
        assert_eq!(dist[1], Some(1));
        assert_eq!(dist[2], Some(2));
    }

    // ── Support ────────────────────────────────────────────────────────

    #[test]
    fn test_support() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<CountingSemiring>::new(2, alpha)
            .initial(0, CountingSemiring::one())
            .final_state(1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .build()
            .unwrap();

        let support = wfa.support();
        assert!(support.accepts(&[a]));
        assert!(!support.accepts(&[]));
    }

    // ── Alphabet ───────────────────────────────────────────────────────

    #[test]
    fn test_alphabet_basics() {
        let alpha = Alphabet::from_chars(&['a', 'b', 'c']);
        assert_eq!(alpha.size(), 3);
        assert!(alpha.contains(&Symbol::Char('a')));
        assert!(!alpha.contains(&Symbol::Char('z')));
        assert_eq!(alpha.index_of(&Symbol::Char('b')), Some(1));
        assert_eq!(alpha.symbol_at(2), Some(&Symbol::Char('c')));
    }

    #[test]
    fn test_alphabet_union_intersection() {
        let a1 = Alphabet::from_chars(&['a', 'b']);
        let a2 = Alphabet::from_chars(&['b', 'c']);
        let u = a1.union(&a2);
        assert_eq!(u.size(), 3); // a, b, c
        let i = a1.intersection(&a2);
        assert_eq!(i.size(), 1); // b
        assert!(i.contains(&Symbol::Char('b')));
    }

    #[test]
    fn test_alphabet_with_epsilon() {
        let alpha = Alphabet::from_chars(&['a', 'b']).with_epsilon();
        assert!(alpha.has_epsilon());
        assert_eq!(alpha.epsilon_index(), Some(0));
        assert_eq!(alpha.size(), 3);
    }

    // ── Display ────────────────────────────────────────────────────────

    #[test]
    fn test_display() {
        let wfa = make_ab_wfa();
        let s = format!("{}", wfa);
        assert!(s.contains("WFA(3 states"));
    }

    // ── Stats ──────────────────────────────────────────────────────────

    #[test]
    fn test_stats() {
        let wfa = make_ab_wfa();
        let stats = wfa.stats();
        assert_eq!(stats.num_states, 3);
        assert_eq!(stats.num_transitions, 2);
        assert_eq!(stats.num_initial, 1);
        assert_eq!(stats.num_final, 1);
        assert!(stats.is_deterministic);
    }

    // ── Serialization round‑trip ───────────────────────────────────────

    #[test]
    fn test_json_roundtrip() {
        let wfa = make_ab_wfa();
        let json = wfa.to_json().unwrap();
        let wfa2 = WeightedFiniteAutomaton::<BooleanSemiring>::from_json(&json).unwrap();
        assert_eq!(wfa, wfa2);
    }

    // ── Scale / shift weights ──────────────────────────────────────────

    #[test]
    fn test_map_weights() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WfaBuilder::<CountingSemiring>::new(2, alpha)
            .initial(0, CountingSemiring::one())
            .final_state(1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .build()
            .unwrap();

        // Scale all weights by 2 (add to itself).
        let doubled = wfa.map_weights(|w| w.add(w));
        let w = doubled.compute_weight(&[a]);
        // initial(2) * transition(2) * final(2) = 8
        let eight = CountingSemiring::one()
            .add(&CountingSemiring::one())
            .mul(&CountingSemiring::one().add(&CountingSemiring::one()))
            .mul(&CountingSemiring::one().add(&CountingSemiring::one()));
        assert_eq!(w, eight);
    }

    // ── Equivalence check (bounded) ────────────────────────────────────

    #[test]
    fn test_equiv_check() {
        let wfa = make_ab_wfa();
        let wfa2 = make_ab_wfa();
        assert!(wfa.equiv_check_bounded(&wfa2, 3));
    }

    // ── from_transitions constructor ───────────────────────────────────

    #[test]
    fn test_from_transitions() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::from_transitions(
            2,
            alpha,
            vec![BooleanSemiring::one(), BooleanSemiring::zero()],
            vec![BooleanSemiring::zero(), BooleanSemiring::one()],
            vec![Transition::new(0, 1, a, BooleanSemiring::one())],
        )
        .unwrap();

        assert!(wfa.accepts(&[a]));
    }

    #[test]
    fn test_from_transitions_dimension_error() {
        let alpha = Alphabet::from_chars(&['a']);
        let result = WeightedFiniteAutomaton::<BooleanSemiring>::from_transitions(
            2,
            alpha,
            vec![BooleanSemiring::one()], // wrong size
            vec![BooleanSemiring::zero(), BooleanSemiring::one()],
            vec![],
        );
        assert!(result.is_err());
    }

    // ── Invalid state / symbol errors ──────────────────────────────────

    #[test]
    fn test_invalid_state_error() {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(2, alpha);
        let result = wfa.add_transition(5, 0, 0, BooleanSemiring::one());
        assert!(matches!(result, Err(WfaError::InvalidState(5, 2))));
    }

    #[test]
    fn test_invalid_symbol_error() {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(2, alpha);
        let result = wfa.add_transition(0, 99, 1, BooleanSemiring::one());
        assert!(matches!(result, Err(WfaError::InvalidSymbol(99, 1))));
    }

    // ── Regex → WFA ────────────────────────────────────────────────────

    #[test]
    fn test_regex_literal() {
        let alpha = Alphabet::from_chars(&['a', 'b', 'c']);
        let wfa =
            WeightedFiniteAutomaton::<BooleanSemiring>::from_regex_str("ab", &alpha).unwrap();

        let wa = wfa.alphabet();
        let a = wa.index_of(&Symbol::Char('a')).unwrap();
        let b = wa.index_of(&Symbol::Char('b')).unwrap();
        let c = wa.index_of(&Symbol::Char('c')).unwrap();

        assert!(wfa.accepts(&[a, b]));
        assert!(!wfa.accepts(&[a, c]));
        assert!(!wfa.accepts(&[a]));
    }

    #[test]
    fn test_regex_alternation() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let wfa =
            WeightedFiniteAutomaton::<BooleanSemiring>::from_regex_str("a|b", &alpha).unwrap();

        let wa = wfa.alphabet();
        let a = wa.index_of(&Symbol::Char('a')).unwrap();
        let b = wa.index_of(&Symbol::Char('b')).unwrap();

        assert!(wfa.accepts(&[a]));
        assert!(wfa.accepts(&[b]));
        assert!(!wfa.accepts(&[a, b]));
    }

    #[test]
    fn test_regex_star() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let wfa =
            WeightedFiniteAutomaton::<BooleanSemiring>::from_regex_str("a*", &alpha).unwrap();

        let wa = wfa.alphabet();
        let a = wa.index_of(&Symbol::Char('a')).unwrap();

        assert!(wfa.accepts(&[]));
        assert!(wfa.accepts(&[a]));
        assert!(wfa.accepts(&[a, a, a]));
    }

    // ── Conversion helpers ─────────────────────────────────────────────

    #[test]
    fn test_boolean_to_counting() {
        let wfa = make_ab_wfa();
        let counting = wfa.to_counting();
        let alpha = counting.alphabet();
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        let w = counting.compute_weight(&[a, b]);
        assert_eq!(w, CountingSemiring::one());
    }

    // ── Add state ──────────────────────────────────────────────────────

    #[test]
    fn test_add_state() {
        let mut wfa = make_ab_wfa();
        assert_eq!(wfa.state_count(), 3);
        let new_id = wfa.add_state();
        assert_eq!(new_id, 3);
        assert_eq!(wfa.state_count(), 4);
    }

    // ── In/out degrees ─────────────────────────────────────────────────

    #[test]
    fn test_degrees() {
        let wfa = make_ab_wfa();
        let in_deg = wfa.in_degrees();
        let out_deg = wfa.out_degrees();

        assert_eq!(in_deg[0], 0); // no incoming
        assert_eq!(in_deg[1], 1); // one incoming (from 0)
        assert_eq!(in_deg[2], 1); // one incoming (from 1)

        assert_eq!(out_deg[0], 1); // one outgoing
        assert_eq!(out_deg[1], 1);
        assert_eq!(out_deg[2], 0); // no outgoing
    }

    // ── Used symbols / active states ───────────────────────────────────

    #[test]
    fn test_used_symbols() {
        let wfa = make_ab_wfa();
        let used = wfa.used_symbols();
        assert_eq!(used.len(), 2);
    }

    #[test]
    fn test_active_states() {
        let wfa = make_ab_wfa();
        let active = wfa.active_states();
        assert!(active.contains(&0));
        assert!(active.contains(&1));
        assert!(!active.contains(&2)); // state 2 has no outgoing
    }

    // ── Remove zero transitions ────────────────────────────────────────

    #[test]
    fn test_remove_zero_transitions() {
        let alpha = Alphabet::from_chars(&['a']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();

        let mut wfa = WfaBuilder::<CountingSemiring>::new(2, alpha)
            .initial(0, CountingSemiring::one())
            .final_state(1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::one())
            .transition(0, a, 1, CountingSemiring::zero())
            .build()
            .unwrap();

        assert_eq!(wfa.num_transitions(), 2);
        wfa.remove_zero_transitions();
        assert_eq!(wfa.num_transitions(), 1);
    }

    // ── Symbol Display ─────────────────────────────────────────────────

    #[test]
    fn test_symbol_display() {
        assert_eq!(format!("{}", Symbol::Char('a')), "a");
        assert_eq!(format!("{}", Symbol::Epsilon), "ε");
        assert_eq!(format!("{}", Symbol::Wildcard), "*");
        assert_eq!(format!("{}", Symbol::Token("hello".into())), "\"hello\"");
        assert_eq!(format!("{}", Symbol::Id(42)), "#42");
    }

    // ── Alphabet display ───────────────────────────────────────────────

    #[test]
    fn test_alphabet_display() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let s = format!("{}", alpha);
        assert!(s.contains('a'));
        assert!(s.contains('b'));
    }

    // ── WfaStats display ───────────────────────────────────────────────

    #[test]
    fn test_stats_display() {
        let wfa = make_ab_wfa();
        let stats = wfa.stats();
        let s = format!("{}", stats);
        assert!(s.contains("States: 3"));
    }

    // ── Error display ──────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = WfaError::InvalidState(5, 3);
        assert!(format!("{}", e).contains("invalid state index 5"));

        let e2 = WfaError::DimensionMismatch {
            expected: 3,
            got: 5,
        };
        assert!(format!("{}", e2).contains("dimension mismatch"));
    }

    // ── Project ────────────────────────────────────────────────────────

    #[test]
    fn test_project() {
        let alpha = Alphabet::from_chars(&['a', 'b', 'c']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();
        let c = alpha.index_of(&Symbol::Char('c')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(2, alpha)
            .initial(0, BooleanSemiring::one())
            .final_state(1, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(0, b, 1, BooleanSemiring::one())
            .transition(0, c, 1, BooleanSemiring::one())
            .build()
            .unwrap();

        let keep: HashSet<usize> = [a, b].iter().copied().collect();
        let projected = wfa.project(&keep);
        assert_eq!(projected.alphabet().size(), 2);
    }

    // ── Apply symbol map ───────────────────────────────────────────────

    #[test]
    fn test_apply_symbol_map() {
        let alpha_src = Alphabet::from_chars(&['a', 'b']);
        let alpha_tgt = Alphabet::from_chars(&['x', 'y']);
        let a = alpha_src.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha_src.index_of(&Symbol::Char('b')).unwrap();
        let x = alpha_tgt.index_of(&Symbol::Char('x')).unwrap();
        let y = alpha_tgt.index_of(&Symbol::Char('y')).unwrap();

        let wfa = WfaBuilder::<BooleanSemiring>::new(3, alpha_src)
            .initial(0, BooleanSemiring::one())
            .final_state(2, BooleanSemiring::one())
            .transition(0, a, 1, BooleanSemiring::one())
            .transition(1, b, 2, BooleanSemiring::one())
            .build()
            .unwrap();

        // Map a→x, b→y.
        let mapping = vec![x, y];
        let mapped = wfa.apply_symbol_map(&alpha_tgt, &mapping).unwrap();
        assert!(mapped.accepts(&[x, y]));
        assert!(!mapped.accepts(&[y, x]));
    }

    // ── Verify forward == matrix computation on various inputs ─────────

    #[test]
    fn test_forward_vs_matrix_tropical() {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let b = alpha.index_of(&Symbol::Char('b')).unwrap();

        let wfa = WfaBuilder::<TropicalSemiring>::new(3, alpha)
            .initial(0, TropicalSemiring::one())
            .final_state(2, TropicalSemiring::one())
            .transition(0, a, 1, TropicalSemiring::from_value(2.0))
            .transition(1, b, 2, TropicalSemiring::from_value(3.0))
            .transition(0, b, 2, TropicalSemiring::from_value(10.0))
            .build()
            .unwrap();

        let inputs: Vec<Vec<usize>> = vec![
            vec![a, b],
            vec![b],
            vec![a],
            vec![],
            vec![a, a],
        ];

        for input in &inputs {
            let w1 = wfa.compute_weight(input);
            let w2 = wfa.compute_weight_matrix(input);
            assert_eq!(w1, w2, "mismatch on input {:?}", input);
        }
    }

    // ── Transition struct ──────────────────────────────────────────────

    #[test]
    fn test_transition_new() {
        let t = Transition::<BooleanSemiring>::new(0, 1, 2, BooleanSemiring::one());
        assert_eq!(t.from_state, 0);
        assert_eq!(t.to_state, 1);
        assert_eq!(t.symbol, 2);
    }

    // ── All transitions ────────────────────────────────────────────────

    #[test]
    fn test_all_transitions() {
        let wfa = make_ab_wfa();
        let all = wfa.all_transitions();
        assert_eq!(all.len(), 2);
    }

    // ── Transitions from ───────────────────────────────────────────────

    #[test]
    fn test_transitions_from() {
        let wfa = make_ab_wfa();
        let alpha = wfa.alphabet();
        let a = alpha.index_of(&Symbol::Char('a')).unwrap();
        let trans = wfa.transitions_from(0, a);
        assert_eq!(trans.len(), 1);
        assert_eq!(trans[0].0, 1); // to state 1
    }
}
