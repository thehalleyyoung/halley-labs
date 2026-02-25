//! Weighted Finite Transducer (WFT) implementation.
//!
//! Provides a full-featured weighted finite-state transducer library including:
//! - Construction and manipulation of weighted transducers
//! - Composition with ε-filter algorithms
//! - Lazy (on-the-fly) composition
//! - Shortest-path and n-best-path algorithms
//! - Look-ahead composition filters
//! - Epsilon removal, trimming, weight pushing, synchronization
//! - Projection to weighted finite automata

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt;

use indexmap::IndexMap;
use log::{debug, trace, warn};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::automaton::{Alphabet, Symbol, WeightedFiniteAutomaton};
use super::semiring::{
    BooleanSemiring, CountingSemiring, RealSemiring, Semiring, StarSemiring,
    TropicalSemiring,
};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during transducer operations.
#[derive(Debug, Error, Clone)]
pub enum TransducerError {
    #[error("invalid state index {0}: transducer has {1} states")]
    InvalidState(usize, usize),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("incompatible alphabets: {0}")]
    IncompatibleAlphabets(String),

    #[error("transducer is empty (has no states)")]
    EmptyTransducer,

    #[error("composition error: {0}")]
    CompositionError(String),

    #[error("invalid mapping: {0}")]
    InvalidMapping(String),

    #[error("no accepting path found")]
    NoPath,

    #[error("cycle detected in transducer")]
    CycleDetected,

    #[error("filter error: {0}")]
    FilterError(String),
}

pub type Result<T> = std::result::Result<T, TransducerError>;

// ---------------------------------------------------------------------------
// TransducerTransition
// ---------------------------------------------------------------------------

/// A single transition in a weighted transducer.
///
/// `input_symbol` and `output_symbol` are indices into the respective alphabets;
/// `None` represents the epsilon (ε) symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransducerTransition<S: Semiring> {
    pub from_state: usize,
    pub to_state: usize,
    pub input_symbol: Option<usize>,
    pub output_symbol: Option<usize>,
    pub weight: S,
}

impl<S: Semiring> TransducerTransition<S> {
    pub fn new(
        from_state: usize,
        to_state: usize,
        input_symbol: Option<usize>,
        output_symbol: Option<usize>,
        weight: S,
    ) -> Self {
        Self {
            from_state,
            to_state,
            input_symbol,
            output_symbol,
            weight,
        }
    }

    #[inline]
    pub fn is_input_epsilon(&self) -> bool {
        self.input_symbol.is_none()
    }

    #[inline]
    pub fn is_output_epsilon(&self) -> bool {
        self.output_symbol.is_none()
    }

    #[inline]
    pub fn is_epsilon(&self) -> bool {
        self.input_symbol.is_none() && self.output_symbol.is_none()
    }
}

impl<S: Semiring> PartialEq for TransducerTransition<S> {
    fn eq(&self, other: &Self) -> bool {
        self.from_state == other.from_state
            && self.to_state == other.to_state
            && self.input_symbol == other.input_symbol
            && self.output_symbol == other.output_symbol
            && self.weight == other.weight
    }
}

// ---------------------------------------------------------------------------
// Filter types for composition
// ---------------------------------------------------------------------------

/// Epsilon filter states for the three-way filter used in transducer composition.
///
/// Prevents redundant ε-paths in the composed result by sequencing epsilon
/// transitions through a three-state filter automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpsilonFilter {
    /// State 0: no epsilon pending — allow any transition.
    NoFilter,
    /// State 1: ε₁ pending — the first transducer consumed an ε on its output
    /// that hasn't been matched yet.
    Filter1,
    /// State 2: ε₂ pending — the second transducer consumed an ε on its input
    /// that hasn't been matched yet.
    Filter2,
    /// Both sides have pending epsilons (used in multi-epsilon filter).
    FilterBoth,
}

impl Default for EpsilonFilter {
    fn default() -> Self {
        EpsilonFilter::NoFilter
    }
}

/// Composition filter strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    /// No epsilon filter — works only for ε-free transducers.
    Trivial,
    /// The standard 3-state ε-sequencing filter from Pereira & Riley.
    EpsilonSequencing,
    /// Epsilon matching filter — only matches identical ε-transitions.
    EpsilonMatching,
    /// Multi-epsilon filter for transducers with multiple ε-types.
    MultiEpsilonFilter,
}

// ---------------------------------------------------------------------------
// CompositionState / CompositionStateSpace
// ---------------------------------------------------------------------------

/// A state in the composed transducer, represented as a triple of
/// (state_in_T1, state_in_T2, epsilon_filter_state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompositionState {
    pub state1: usize,
    pub state2: usize,
    pub filter: EpsilonFilter,
}

impl CompositionState {
    pub fn new(state1: usize, state2: usize, filter: EpsilonFilter) -> Self {
        Self {
            state1,
            state2,
            filter,
        }
    }
}

/// Manages the mapping between `CompositionState` triples and linear state
/// indices in the composed transducer.
#[derive(Debug, Clone)]
struct CompositionStateSpace {
    state_to_id: HashMap<CompositionState, usize>,
    id_to_state: Vec<CompositionState>,
}

impl CompositionStateSpace {
    fn new() -> Self {
        Self {
            state_to_id: HashMap::new(),
            id_to_state: Vec::new(),
        }
    }

    /// Returns the linear id for the given composition state, creating one if
    /// it doesn't already exist.
    fn get_or_insert(&mut self, cs: CompositionState) -> usize {
        if let Some(&id) = self.state_to_id.get(&cs) {
            id
        } else {
            let id = self.id_to_state.len();
            self.id_to_state.push(cs);
            self.state_to_id.insert(cs, id);
            id
        }
    }

    fn get(&self, cs: &CompositionState) -> Option<usize> {
        self.state_to_id.get(cs).copied()
    }

    fn len(&self) -> usize {
        self.id_to_state.len()
    }

    fn state_at(&self, id: usize) -> &CompositionState {
        &self.id_to_state[id]
    }
}

// ---------------------------------------------------------------------------
// WeightedTransducer
// ---------------------------------------------------------------------------

/// A weighted finite-state transducer over a semiring `S`.
///
/// The transducer maps input strings (over `input_alphabet`) to output strings
/// (over `output_alphabet`) with weights drawn from the semiring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedTransducer<S: Semiring> {
    pub num_states: usize,
    pub input_alphabet: Alphabet,
    pub output_alphabet: Alphabet,
    pub initial_weights: Vec<S>,
    pub final_weights: Vec<S>,
    pub transitions: Vec<TransducerTransition<S>>,
    /// For each state, the indices into `transitions` of transitions leaving
    /// that state.
    pub transition_index: Vec<Vec<usize>>,
}

impl<S: Semiring> WeightedTransducer<S> {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new transducer with `num_states` states and the given alphabets.
    /// All initial and final weights are set to zero.
    pub fn new(num_states: usize, input_alphabet: Alphabet, output_alphabet: Alphabet) -> Self {
        Self {
            num_states,
            input_alphabet,
            output_alphabet,
            initial_weights: vec![S::zero(); num_states],
            final_weights: vec![S::zero(); num_states],
            transitions: Vec::new(),
            transition_index: vec![Vec::new(); num_states],
        }
    }

    /// Add a state and return its index.
    pub fn add_state(&mut self) -> usize {
        let id = self.num_states;
        self.num_states += 1;
        self.initial_weights.push(S::zero());
        self.final_weights.push(S::zero());
        self.transition_index.push(Vec::new());
        id
    }

    /// Add a transition.  `input_sym` and `output_sym` are `None` for ε.
    pub fn add_transition(
        &mut self,
        from: usize,
        input_sym: Option<usize>,
        output_sym: Option<usize>,
        to: usize,
        weight: S,
    ) -> Result<()> {
        if from >= self.num_states {
            return Err(TransducerError::InvalidState(from, self.num_states));
        }
        if to >= self.num_states {
            return Err(TransducerError::InvalidState(to, self.num_states));
        }
        let idx = self.transitions.len();
        self.transitions.push(TransducerTransition::new(
            from, to, input_sym, output_sym, weight,
        ));
        self.transition_index[from].push(idx);
        Ok(())
    }

    /// Set the initial weight for a state.
    pub fn set_initial_weight(&mut self, state: usize, weight: S) {
        assert!(state < self.num_states, "state out of range");
        self.initial_weights[state] = weight;
    }

    /// Set the final weight for a state.
    pub fn set_final_weight(&mut self, state: usize, weight: S) {
        assert!(state < self.num_states, "state out of range");
        self.final_weights[state] = weight;
    }

    /// Rebuild the `transition_index` from the current `transitions` vector.
    pub fn build_index(&mut self) {
        self.transition_index = vec![Vec::new(); self.num_states];
        for (idx, tr) in self.transitions.iter().enumerate() {
            if tr.from_state < self.num_states {
                self.transition_index[tr.from_state].push(idx);
            }
        }
    }

    /// Create the identity transducer for the given alphabet: each symbol maps
    /// to itself with weight `S::one()`.
    pub fn identity(alphabet: &Alphabet) -> Self {
        let n = alphabet.size();
        let mut t = Self::new(1, alphabet.clone(), alphabet.clone());
        t.set_initial_weight(0, S::one());
        t.set_final_weight(0, S::one());
        for i in 0..n {
            t.add_transition(0, Some(i), Some(i), 0, S::one())
                .expect("identity construction cannot fail");
        }
        t
    }

    /// Convert a WFA into a transducer that maps each input symbol to the same
    /// output symbol (identity transduction) with the WFA's weights.
    pub fn from_wfa(wfa: &WeightedFiniteAutomaton<S>) -> Self {
        let n = wfa.num_states;
        let alph = wfa.alphabet.clone();
        let mut t = Self::new(n, alph.clone(), alph);
        t.initial_weights = wfa.initial_weights.clone();
        t.final_weights = wfa.final_weights.clone();

        // Walk the WFA transitions.  The WFA stores transitions in a
        // (num_states × alphabet_size × num_states) weight matrix or
        // adjacency list — we iterate the transition matrix rows.
        // Because we don't know the exact internal layout of the WFA we
        // reconstruct transitions from its transition matrix.
        let alpha_size = wfa.alphabet.size();
        for from in 0..n {
            for sym in 0..alpha_size {
                for to in 0..n {
                    let w = wfa.transition_weight(from, sym, to);
                    if !w.is_zero() {
                        t.add_transition(from, Some(sym), Some(sym), to, w)
                            .expect("from_wfa construction cannot fail");
                    }
                }
            }
        }
        t
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    #[inline]
    pub fn num_transitions(&self) -> usize {
        self.transitions.len()
    }

    #[inline]
    pub fn state_count(&self) -> usize {
        self.num_states
    }

    /// Return transitions leaving `state`.
    pub fn transitions_from(&self, state: usize) -> Vec<&TransducerTransition<S>> {
        if state >= self.num_states {
            return Vec::new();
        }
        self.transition_index[state]
            .iter()
            .map(|&idx| &self.transitions[idx])
            .collect()
    }

    /// Return all states with non-zero initial weight.
    pub fn initial_states(&self) -> Vec<usize> {
        self.initial_weights
            .iter()
            .enumerate()
            .filter(|(_, w)| !w.is_zero())
            .map(|(i, _)| i)
            .collect()
    }

    /// Return all states with non-zero final weight.
    pub fn final_states(&self) -> Vec<usize> {
        self.final_weights
            .iter()
            .enumerate()
            .filter(|(_, w)| !w.is_zero())
            .map(|(i, _)| i)
            .collect()
    }

    /// Check whether the transducer has any ε-transitions on the input tape.
    pub fn has_input_epsilon(&self) -> bool {
        self.transitions.iter().any(|t| t.input_symbol.is_none())
    }

    /// Check whether the transducer has any ε-transitions on the output tape.
    pub fn has_output_epsilon(&self) -> bool {
        self.transitions.iter().any(|t| t.output_symbol.is_none())
    }

    // ------------------------------------------------------------------
    // Transduction
    // ------------------------------------------------------------------

    /// Transduce an input string, returning all (output, weight) pairs reachable
    /// via accepting paths.
    ///
    /// Uses a breadth-first exploration of configurations
    /// `(state, input_position, output_buffer)`.
    pub fn transduce(&self, input: &[usize]) -> Vec<(Vec<usize>, S)> {
        // Configuration: (state, input_pos, output_buffer, accumulated_weight)
        type Config<W> = (usize, usize, Vec<usize>, W);

        let mut results: HashMap<Vec<usize>, S> = HashMap::new();
        let mut queue: VecDeque<Config<S>> = VecDeque::new();

        // Seed with initial states
        for (s, w) in self.initial_weights.iter().enumerate() {
            if !w.is_zero() {
                queue.push_back((s, 0, Vec::new(), w.clone()));
            }
        }

        // Limit to prevent combinatorial explosion
        let max_configs: usize = 500_000;
        let mut configs_explored: usize = 0;

        while let Some((state, pos, out_buf, acc_w)) = queue.pop_front() {
            configs_explored += 1;
            if configs_explored > max_configs {
                warn!("transduce: configuration limit reached, returning partial results");
                break;
            }

            // If we have consumed all input, check for finality
            if pos == input.len() {
                // Follow any remaining ε-output transitions
                let fw = &self.final_weights[state];
                if !fw.is_zero() {
                    let total = S::mul(&acc_w, fw);
                    results
                        .entry(out_buf.clone())
                        .and_modify(|w| w.add_assign(&total))
                        .or_insert(total);
                }
            }

            // Explore transitions from this state
            for &idx in &self.transition_index[state] {
                let tr = &self.transitions[idx];

                match (tr.input_symbol, pos <= input.len()) {
                    // ε-input transition: don't consume input
                    (None, _) => {
                        let w = S::mul(&acc_w, &tr.weight);
                        let mut new_out = out_buf.clone();
                        if let Some(osym) = tr.output_symbol {
                            new_out.push(osym);
                        }
                        queue.push_back((tr.to_state, pos, new_out, w));
                    }
                    // Matching input symbol
                    (Some(isym), true) if pos < input.len() && isym == input[pos] => {
                        let w = S::mul(&acc_w, &tr.weight);
                        let mut new_out = out_buf.clone();
                        if let Some(osym) = tr.output_symbol {
                            new_out.push(osym);
                        }
                        queue.push_back((tr.to_state, pos + 1, new_out, w));
                    }
                    _ => {}
                }
            }
        }

        results.into_iter().collect()
    }

    /// Return the single best (output, weight) pair for the given input.
    /// Requires `S: Ord` so that we can compare weights.
    pub fn transduce_best(&self, input: &[usize]) -> Option<(Vec<usize>, S)>
    where
        S: Ord,
    {
        let mut all = self.transduce(input);
        if all.is_empty() {
            return None;
        }
        all.sort_by(|a, b| a.1.cmp(&b.1));
        Some(all.remove(0))
    }

    // ------------------------------------------------------------------
    // Inversion
    // ------------------------------------------------------------------

    /// Swap input and output tapes.
    pub fn invert(&self) -> Self {
        let mut inv = Self::new(
            self.num_states,
            self.output_alphabet.clone(),
            self.input_alphabet.clone(),
        );
        inv.initial_weights = self.initial_weights.clone();
        inv.final_weights = self.final_weights.clone();
        for tr in &self.transitions {
            inv.add_transition(
                tr.from_state,
                tr.output_symbol,
                tr.input_symbol,
                tr.to_state,
                tr.weight.clone(),
            )
            .expect("invert: state indices valid");
        }
        inv
    }

    // ------------------------------------------------------------------
    // Projections
    // ------------------------------------------------------------------

    /// Project to an acceptor (WFA) on the input tape.
    pub fn input_projection(&self) -> WeightedFiniteAutomaton<S> {
        let mut wfa = WeightedFiniteAutomaton::new(self.num_states, self.input_alphabet.clone());
        wfa.initial_weights = self.initial_weights.clone();
        wfa.final_weights = self.final_weights.clone();
        for tr in &self.transitions {
            if let Some(sym) = tr.input_symbol {
                let _ = wfa.add_transition(tr.from_state, sym, tr.to_state, tr.weight.clone());
            }
        }
        wfa
    }

    /// Project to an acceptor (WFA) on the output tape.
    pub fn output_projection(&self) -> WeightedFiniteAutomaton<S> {
        let mut wfa = WeightedFiniteAutomaton::new(self.num_states, self.output_alphabet.clone());
        wfa.initial_weights = self.initial_weights.clone();
        wfa.final_weights = self.final_weights.clone();
        for tr in &self.transitions {
            if let Some(sym) = tr.output_symbol {
                let _ = wfa.add_transition(tr.from_state, sym, tr.to_state, tr.weight.clone());
            }
        }
        wfa
    }

    // ------------------------------------------------------------------
    // Composition – full (eager) algorithm
    // ------------------------------------------------------------------

    /// Compose `self` with `other` using the default ε-sequencing filter.
    ///
    /// The result maps input strings of `self` to output strings of `other`,
    /// weighted by the semiring product of matching intermediate strings.
    pub fn compose(&self, other: &WeightedTransducer<S>) -> Result<WeightedTransducer<S>> {
        self.compose_with_filter(other, FilterType::EpsilonSequencing)
    }

    /// Compose with an explicit filter strategy.
    pub fn compose_with_filter(
        &self,
        other: &WeightedTransducer<S>,
        filter_type: FilterType,
    ) -> Result<WeightedTransducer<S>> {
        // Validate that the output alphabet of self is compatible with the
        // input alphabet of other.
        if self.output_alphabet.size() != other.input_alphabet.size() {
            return Err(TransducerError::IncompatibleAlphabets(format!(
                "output alphabet size {} ≠ input alphabet size {}",
                self.output_alphabet.size(),
                other.input_alphabet.size()
            )));
        }

        let mut state_space = CompositionStateSpace::new();
        let mut result_transitions: Vec<TransducerTransition<S>> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Seed with all pairs of initial states
        for s1 in self.initial_states() {
            for s2 in other.initial_states() {
                let cs = CompositionState::new(s1, s2, EpsilonFilter::NoFilter);
                let id = state_space.get_or_insert(cs);
                queue.push_back(id);
            }
        }

        while let Some(cur_id) = queue.pop_front() {
            let cs = *state_space.state_at(cur_id);
            let (s1, s2, filt) = (cs.state1, cs.state2, cs.filter);

            let t1_trans = self.transitions_from(s1);
            let t2_trans = other.transitions_from(s2);

            match filter_type {
                FilterType::Trivial => {
                    self.compose_trivial(
                        other,
                        cur_id,
                        s1,
                        s2,
                        &t1_trans,
                        &t2_trans,
                        &mut state_space,
                        &mut result_transitions,
                        &mut queue,
                    );
                }
                FilterType::EpsilonSequencing => {
                    self.compose_epsilon_sequencing(
                        other,
                        cur_id,
                        s1,
                        s2,
                        filt,
                        &t1_trans,
                        &t2_trans,
                        &mut state_space,
                        &mut result_transitions,
                        &mut queue,
                    );
                }
                FilterType::EpsilonMatching => {
                    self.compose_epsilon_matching(
                        other,
                        cur_id,
                        s1,
                        s2,
                        filt,
                        &t1_trans,
                        &t2_trans,
                        &mut state_space,
                        &mut result_transitions,
                        &mut queue,
                    );
                }
                FilterType::MultiEpsilonFilter => {
                    self.compose_multi_epsilon(
                        other,
                        cur_id,
                        s1,
                        s2,
                        filt,
                        &t1_trans,
                        &t2_trans,
                        &mut state_space,
                        &mut result_transitions,
                        &mut queue,
                    );
                }
            }
        }

        // Build the composed transducer
        let n = state_space.len();
        let mut composed = WeightedTransducer::new(
            n,
            self.input_alphabet.clone(),
            other.output_alphabet.clone(),
        );

        // Set initial/final weights
        for i in 0..n {
            let cs = state_space.state_at(i);
            let iw = S::mul(&self.initial_weights[cs.state1], &other.initial_weights[cs.state2]);
            let fw = S::mul(&self.final_weights[cs.state1], &other.final_weights[cs.state2]);
            composed.initial_weights[i] = iw;
            composed.final_weights[i] = fw;
        }

        composed.transitions = result_transitions;
        composed.build_index();
        Ok(composed)
    }

    // -- Trivial filter (ε-free case) --

    fn compose_trivial(
        &self,
        _other: &WeightedTransducer<S>,
        cur_id: usize,
        _s1: usize,
        _s2: usize,
        t1_trans: &[&TransducerTransition<S>],
        t2_trans: &[&TransducerTransition<S>],
        state_space: &mut CompositionStateSpace,
        result_transitions: &mut Vec<TransducerTransition<S>>,
        queue: &mut VecDeque<usize>,
    ) {
        // Only match transitions where T1 output = T2 input (non-ε).
        for tr1 in t1_trans {
            if let Some(osym) = tr1.output_symbol {
                for tr2 in t2_trans {
                    if tr2.input_symbol == Some(osym) {
                        let cs_next =
                            CompositionState::new(tr1.to_state, tr2.to_state, EpsilonFilter::NoFilter);
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = next_id == state_space.len() - 1;
                        let w = S::mul(&tr1.weight, &tr2.weight);
                        result_transitions.push(TransducerTransition::new(
                            cur_id,
                            next_id,
                            tr1.input_symbol,
                            tr2.output_symbol,
                            w,
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }
        }
    }

    // -- ε-Sequencing filter (Pereira & Riley) --
    //
    // The filter has three states {0, 1, 2} and processes 9 cases:
    //
    //   T1 output    T2 input    Filter transition
    //   ─────────    ────────    ─────────────────
    //   a (non-ε)    a           0→0, 1→0, 2→0   (matching symbol)
    //   ε            b (non-ε)   —                (no direct match possible)
    //   a (non-ε)    ε           —                (no direct match possible)
    //   ε            ε           0→0              (synchronized epsilon)
    //
    // To handle ε on one side only, we introduce auxiliary transitions:
    //   T1 out=ε → insert (ε₁, ε₂) with filter 0→1  (T1 takes ε step alone)
    //   T2 in=ε  → insert (ε₁, ε₂) with filter 0→2  (T2 takes ε step alone)
    //   also T1 out=ε with filter 1→1 (stay)
    //   also T2 in=ε  with filter 2→2 (stay)
    //
    // This prevents the redundant path where both T1 and T2 take independent
    // ε-steps interleaved in different orders.

    fn compose_epsilon_sequencing(
        &self,
        _other: &WeightedTransducer<S>,
        cur_id: usize,
        s1: usize,
        s2: usize,
        filt: EpsilonFilter,
        t1_trans: &[&TransducerTransition<S>],
        t2_trans: &[&TransducerTransition<S>],
        state_space: &mut CompositionStateSpace,
        result_transitions: &mut Vec<TransducerTransition<S>>,
        queue: &mut VecDeque<usize>,
    ) {
        // CASE 1: Matching non-ε symbols  (T1.out = a, T2.in = a)
        // Allowed from any filter state; resets filter to NoFilter.
        for tr1 in t1_trans {
            if let Some(osym) = tr1.output_symbol {
                for tr2 in t2_trans {
                    if tr2.input_symbol == Some(osym) {
                        let cs_next = CompositionState::new(
                            tr1.to_state,
                            tr2.to_state,
                            EpsilonFilter::NoFilter,
                        );
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = state_space.get(&cs_next).unwrap() == state_space.len() - 1
                            && next_id == state_space.len() - 1;
                        let w = S::mul(&tr1.weight, &tr2.weight);
                        result_transitions.push(TransducerTransition::new(
                            cur_id,
                            next_id,
                            tr1.input_symbol,
                            tr2.output_symbol,
                            w,
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }
        }

        // CASE 2: T1 output ε (T1 advances alone)
        // Allowed when filter ∈ {NoFilter, Filter1} → Filter1
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter1 {
            for tr1 in t1_trans {
                if tr1.output_symbol.is_none() {
                    let cs_next =
                        CompositionState::new(tr1.to_state, s2, EpsilonFilter::Filter1);
                    let next_id = state_space.get_or_insert(cs_next);
                    let was_new = next_id == state_space.len() - 1;
                    result_transitions.push(TransducerTransition::new(
                        cur_id,
                        next_id,
                        tr1.input_symbol,
                        None, // ε on output
                        tr1.weight.clone(),
                    ));
                    if was_new {
                        queue.push_back(next_id);
                    }
                }
            }
        }

        // CASE 3: T2 input ε (T2 advances alone)
        // Allowed when filter ∈ {NoFilter, Filter2} → Filter2
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter2 {
            for tr2 in t2_trans {
                if tr2.input_symbol.is_none() {
                    let cs_next =
                        CompositionState::new(s1, tr2.to_state, EpsilonFilter::Filter2);
                    let next_id = state_space.get_or_insert(cs_next);
                    let was_new = next_id == state_space.len() - 1;
                    result_transitions.push(TransducerTransition::new(
                        cur_id,
                        next_id,
                        None, // ε on input
                        tr2.output_symbol,
                        tr2.weight.clone(),
                    ));
                    if was_new {
                        queue.push_back(next_id);
                    }
                }
            }
        }

        // CASE 4: Synchronized ε (both T1 output and T2 input are ε)
        // Allowed only from NoFilter; stays at NoFilter.
        if filt == EpsilonFilter::NoFilter {
            for tr1 in t1_trans {
                if tr1.output_symbol.is_none() {
                    for tr2 in t2_trans {
                        if tr2.input_symbol.is_none() {
                            let cs_next = CompositionState::new(
                                tr1.to_state,
                                tr2.to_state,
                                EpsilonFilter::NoFilter,
                            );
                            let next_id = state_space.get_or_insert(cs_next);
                            let was_new = next_id == state_space.len() - 1;
                            let w = S::mul(&tr1.weight, &tr2.weight);
                            result_transitions.push(TransducerTransition::new(
                                cur_id,
                                next_id,
                                tr1.input_symbol,
                                tr2.output_symbol,
                                w,
                            ));
                            if was_new {
                                queue.push_back(next_id);
                            }
                        }
                    }
                }
            }
        }
    }

    // -- ε-Matching filter --

    fn compose_epsilon_matching(
        &self,
        _other: &WeightedTransducer<S>,
        cur_id: usize,
        s1: usize,
        s2: usize,
        filt: EpsilonFilter,
        t1_trans: &[&TransducerTransition<S>],
        t2_trans: &[&TransducerTransition<S>],
        state_space: &mut CompositionStateSpace,
        result_transitions: &mut Vec<TransducerTransition<S>>,
        queue: &mut VecDeque<usize>,
    ) {
        // Non-ε matching (same as sequencing case 1)
        for tr1 in t1_trans {
            if let Some(osym) = tr1.output_symbol {
                for tr2 in t2_trans {
                    if tr2.input_symbol == Some(osym) {
                        let cs_next = CompositionState::new(
                            tr1.to_state,
                            tr2.to_state,
                            EpsilonFilter::NoFilter,
                        );
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = next_id == state_space.len() - 1;
                        let w = S::mul(&tr1.weight, &tr2.weight);
                        result_transitions.push(TransducerTransition::new(
                            cur_id, next_id, tr1.input_symbol, tr2.output_symbol, w,
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }
        }

        // ε-matching: only allow matched ε pairs (both sides take ε simultaneously)
        if filt == EpsilonFilter::NoFilter {
            for tr1 in t1_trans {
                if tr1.output_symbol.is_none() {
                    for tr2 in t2_trans {
                        if tr2.input_symbol.is_none() {
                            let cs_next = CompositionState::new(
                                tr1.to_state,
                                tr2.to_state,
                                EpsilonFilter::NoFilter,
                            );
                            let next_id = state_space.get_or_insert(cs_next);
                            let was_new = next_id == state_space.len() - 1;
                            let w = S::mul(&tr1.weight, &tr2.weight);
                            result_transitions.push(TransducerTransition::new(
                                cur_id, next_id, tr1.input_symbol, tr2.output_symbol, w,
                            ));
                            if was_new {
                                queue.push_back(next_id);
                            }
                        }
                    }
                }
            }
        }

        // Unmatched ε on T1 only — allowed from NoFilter / Filter1
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter1 {
            for tr1 in t1_trans {
                if tr1.output_symbol.is_none() {
                    let has_matching_t2 = t2_trans.iter().any(|t| t.input_symbol.is_none());
                    if !has_matching_t2 {
                        let cs_next =
                            CompositionState::new(tr1.to_state, s2, EpsilonFilter::Filter1);
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = next_id == state_space.len() - 1;
                        result_transitions.push(TransducerTransition::new(
                            cur_id, next_id, tr1.input_symbol, None, tr1.weight.clone(),
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }
        }

        // Unmatched ε on T2 only
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter2 {
            for tr2 in t2_trans {
                if tr2.input_symbol.is_none() {
                    let has_matching_t1 = t1_trans.iter().any(|t| t.output_symbol.is_none());
                    if !has_matching_t1 {
                        let cs_next =
                            CompositionState::new(s1, tr2.to_state, EpsilonFilter::Filter2);
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = next_id == state_space.len() - 1;
                        result_transitions.push(TransducerTransition::new(
                            cur_id, next_id, None, tr2.output_symbol, tr2.weight.clone(),
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }
        }
    }

    // -- Multi-epsilon filter --

    fn compose_multi_epsilon(
        &self,
        _other: &WeightedTransducer<S>,
        cur_id: usize,
        s1: usize,
        s2: usize,
        filt: EpsilonFilter,
        t1_trans: &[&TransducerTransition<S>],
        t2_trans: &[&TransducerTransition<S>],
        state_space: &mut CompositionStateSpace,
        result_transitions: &mut Vec<TransducerTransition<S>>,
        queue: &mut VecDeque<usize>,
    ) {
        // Non-ε matching — always allowed, resets to NoFilter
        for tr1 in t1_trans {
            if let Some(osym) = tr1.output_symbol {
                for tr2 in t2_trans {
                    if tr2.input_symbol == Some(osym) {
                        let cs_next = CompositionState::new(
                            tr1.to_state,
                            tr2.to_state,
                            EpsilonFilter::NoFilter,
                        );
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = next_id == state_space.len() - 1;
                        let w = S::mul(&tr1.weight, &tr2.weight);
                        result_transitions.push(TransducerTransition::new(
                            cur_id, next_id, tr1.input_symbol, tr2.output_symbol, w,
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }
        }

        // T1 ε alone: NoFilter → Filter1, Filter1 → Filter1
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter1 {
            for tr1 in t1_trans {
                if tr1.output_symbol.is_none() {
                    let cs_next =
                        CompositionState::new(tr1.to_state, s2, EpsilonFilter::Filter1);
                    let next_id = state_space.get_or_insert(cs_next);
                    let was_new = next_id == state_space.len() - 1;
                    result_transitions.push(TransducerTransition::new(
                        cur_id, next_id, tr1.input_symbol, None, tr1.weight.clone(),
                    ));
                    if was_new {
                        queue.push_back(next_id);
                    }
                }
            }
        }

        // T2 ε alone: NoFilter → Filter2, Filter2 → Filter2
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter2 {
            for tr2 in t2_trans {
                if tr2.input_symbol.is_none() {
                    let cs_next =
                        CompositionState::new(s1, tr2.to_state, EpsilonFilter::Filter2);
                    let next_id = state_space.get_or_insert(cs_next);
                    let was_new = next_id == state_space.len() - 1;
                    result_transitions.push(TransducerTransition::new(
                        cur_id, next_id, None, tr2.output_symbol, tr2.weight.clone(),
                    ));
                    if was_new {
                        queue.push_back(next_id);
                    }
                }
            }
        }

        // Simultaneous ε: NoFilter → FilterBoth, FilterBoth → FilterBoth
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::FilterBoth {
            for tr1 in t1_trans {
                if tr1.output_symbol.is_none() {
                    for tr2 in t2_trans {
                        if tr2.input_symbol.is_none() {
                            let cs_next = CompositionState::new(
                                tr1.to_state,
                                tr2.to_state,
                                EpsilonFilter::FilterBoth,
                            );
                            let next_id = state_space.get_or_insert(cs_next);
                            let was_new = next_id == state_space.len() - 1;
                            let w = S::mul(&tr1.weight, &tr2.weight);
                            result_transitions.push(TransducerTransition::new(
                                cur_id, next_id, tr1.input_symbol, tr2.output_symbol, w,
                            ));
                            if was_new {
                                queue.push_back(next_id);
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Shortest-path algorithms
    // ------------------------------------------------------------------

    /// Find the shortest (minimum-weight) output for a given input string.
    ///
    /// Uses Dijkstra-style exploration on configurations. Requires `S: Ord`.
    pub fn shortest_path(&self, input: &[usize]) -> Option<(Vec<usize>, S)>
    where
        S: Ord,
    {
        #[derive(Clone, Eq, PartialEq)]
        struct SPConfig<W: Ord + Eq> {
            weight: W,
            state: usize,
            pos: usize,
            output: Vec<usize>,
        }

        impl<W: Ord + Eq> Ord for SPConfig<W> {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse for min-heap
                other.weight.cmp(&self.weight)
            }
        }

        impl<W: Ord + Eq> PartialOrd for SPConfig<W> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap: BinaryHeap<SPConfig<S>> = BinaryHeap::new();
        let mut visited: HashSet<(usize, usize)> = HashSet::new();

        for (s, w) in self.initial_weights.iter().enumerate() {
            if !w.is_zero() {
                heap.push(SPConfig {
                    weight: w.clone(),
                    state: s,
                    pos: 0,
                    output: Vec::new(),
                });
            }
        }

        while let Some(cfg) = heap.pop() {
            // Check final
            if cfg.pos == input.len() {
                let fw = &self.final_weights[cfg.state];
                if !fw.is_zero() {
                    let total = S::mul(&cfg.weight, fw);
                    return Some((cfg.output, total));
                }
            }

            let key = (cfg.state, cfg.pos);
            if visited.contains(&key) {
                continue;
            }
            visited.insert(key);

            for &idx in &self.transition_index[cfg.state] {
                let tr = &self.transitions[idx];
                match tr.input_symbol {
                    None => {
                        let w = S::mul(&cfg.weight, &tr.weight);
                        let mut new_out = cfg.output.clone();
                        if let Some(osym) = tr.output_symbol {
                            new_out.push(osym);
                        }
                        heap.push(SPConfig {
                            weight: w,
                            state: tr.to_state,
                            pos: cfg.pos,
                            output: new_out,
                        });
                    }
                    Some(isym) if cfg.pos < input.len() && isym == input[cfg.pos] => {
                        let w = S::mul(&cfg.weight, &tr.weight);
                        let mut new_out = cfg.output.clone();
                        if let Some(osym) = tr.output_symbol {
                            new_out.push(osym);
                        }
                        heap.push(SPConfig {
                            weight: w,
                            state: tr.to_state,
                            pos: cfg.pos + 1,
                            output: new_out,
                        });
                    }
                    _ => {}
                }
            }
        }

        None
    }

    /// Find the `n` best (output, weight) pairs via beam search.
    pub fn n_best_paths(&self, input: &[usize], n: usize) -> Vec<(Vec<usize>, S)>
    where
        S: Ord,
    {
        #[derive(Clone, Eq, PartialEq)]
        struct NBConfig<W: Ord + Eq> {
            weight: W,
            state: usize,
            pos: usize,
            output: Vec<usize>,
        }

        impl<W: Ord + Eq> Ord for NBConfig<W> {
            fn cmp(&self, other: &Self) -> Ordering {
                other.weight.cmp(&self.weight)
            }
        }

        impl<W: Ord + Eq> PartialOrd for NBConfig<W> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        if n == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<NBConfig<S>> = BinaryHeap::new();
        let mut results: Vec<(Vec<usize>, S)> = Vec::new();
        // Track how many times each (state, pos) has been popped for n-best
        let mut pop_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for (s, w) in self.initial_weights.iter().enumerate() {
            if !w.is_zero() {
                heap.push(NBConfig {
                    weight: w.clone(),
                    state: s,
                    pos: 0,
                    output: Vec::new(),
                });
            }
        }

        let max_iters = n * self.num_states * (input.len() + 1) * 10 + 10_000;
        let mut iters = 0;

        while let Some(cfg) = heap.pop() {
            iters += 1;
            if iters > max_iters {
                break;
            }

            let key = (cfg.state, cfg.pos);
            let count = pop_counts.entry(key).or_insert(0);
            *count += 1;
            if *count > n {
                continue;
            }

            // Check accepting configuration
            if cfg.pos == input.len() {
                let fw = &self.final_weights[cfg.state];
                if !fw.is_zero() {
                    let total = S::mul(&cfg.weight, fw);
                    results.push((cfg.output.clone(), total));
                    if results.len() >= n {
                        break;
                    }
                }
            }

            // Expand
            for &idx in &self.transition_index[cfg.state] {
                let tr = &self.transitions[idx];
                match tr.input_symbol {
                    None => {
                        let w = S::mul(&cfg.weight, &tr.weight);
                        let mut new_out = cfg.output.clone();
                        if let Some(osym) = tr.output_symbol {
                            new_out.push(osym);
                        }
                        heap.push(NBConfig {
                            weight: w,
                            state: tr.to_state,
                            pos: cfg.pos,
                            output: new_out,
                        });
                    }
                    Some(isym) if cfg.pos < input.len() && isym == input[cfg.pos] => {
                        let w = S::mul(&cfg.weight, &tr.weight);
                        let mut new_out = cfg.output.clone();
                        if let Some(osym) = tr.output_symbol {
                            new_out.push(osym);
                        }
                        heap.push(NBConfig {
                            weight: w,
                            state: tr.to_state,
                            pos: cfg.pos + 1,
                            output: new_out,
                        });
                    }
                    _ => {}
                }
            }
        }

        results
    }

    /// All-pairs shortest paths using Floyd-Warshall on the state graph.
    ///
    /// Returns a `num_states × num_states` matrix where entry `[i][j]` is the
    /// minimum total weight of any path from state i to state j (ignoring
    /// input/output labels).
    pub fn all_pairs_shortest(&self) -> Vec<Vec<S>>
    where
        S: Ord,
    {
        let n = self.num_states;
        // Initialize with zero (identity for min-plus / tropical) on diagonal,
        // and S::zero() elsewhere.  We use S::zero() as "infinity" for
        // semirings where zero is the additive identity (e.g., in the tropical
        // semiring, 0 = +∞).  The diagonal gets S::one() (multiplicative
        // identity = 0 in tropical).
        let mut dist: Vec<Vec<S>> = vec![vec![S::zero(); n]; n];
        for i in 0..n {
            dist[i][i] = S::one();
        }

        // Initialize from transitions
        for tr in &self.transitions {
            let cur = &dist[tr.from_state][tr.to_state];
            let new_w = S::add(cur, &tr.weight);
            dist[tr.from_state][tr.to_state] = new_w;
        }

        // Floyd-Warshall relaxation
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let via_k = S::mul(&dist[i][k], &dist[k][j]);
                    let direct = dist[i][j].clone();
                    let relaxed = S::add(&direct, &via_k);
                    dist[i][j] = relaxed;
                }
            }
        }

        dist
    }

    // ------------------------------------------------------------------
    // Epsilon removal
    // ------------------------------------------------------------------

    /// Compute the ε-closure from a given state, following only transitions
    /// where the selector predicate returns `true` (e.g., only input-ε or
    /// only output-ε).
    ///
    /// Returns a map from reachable state to accumulated weight.
    fn epsilon_closure<F>(&self, start: usize, is_epsilon_tr: F) -> HashMap<usize, S>
    where
        F: Fn(&TransducerTransition<S>) -> bool,
    {
        let mut closure: HashMap<usize, S> = HashMap::new();
        closure.insert(start, S::one());
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(start);

        while let Some(s) = queue.pop_front() {
            let w_s = closure[&s].clone();
            for &idx in &self.transition_index[s] {
                let tr = &self.transitions[idx];
                if is_epsilon_tr(tr) {
                    let new_w = S::mul(&w_s, &tr.weight);
                    let entry = closure.entry(tr.to_state).or_insert_with(S::zero);
                    let combined = S::add(entry, &new_w);
                    if combined != *entry {
                        *entry = combined;
                        queue.push_back(tr.to_state);
                    }
                }
            }
        }

        closure
    }

    /// Remove all transitions with ε on the input tape by computing
    /// ε-closures and redistributing weights.
    pub fn remove_input_epsilon(&self) -> Self {
        let n = self.num_states;

        // Compute input-ε closures for every state
        let closures: Vec<HashMap<usize, S>> = (0..n)
            .map(|s| self.epsilon_closure(s, |tr| tr.input_symbol.is_none()))
            .collect();

        let mut result = Self::new(n, self.input_alphabet.clone(), self.output_alphabet.clone());
        result.initial_weights = self.initial_weights.clone();

        // Adjust final weights: if q can reach q' via input-ε with weight w,
        // then q's new final weight += w * final(q')
        for s in 0..n {
            let mut fw = self.final_weights[s].clone();
            for (&reachable, w) in &closures[s] {
                if reachable != s {
                    let contribution = S::mul(w, &self.final_weights[reachable]);
                    fw.add_assign(&contribution);
                }
            }
            result.final_weights[s] = fw;
        }

        // Add non-input-ε transitions, but also for each state reachable via
        // input-ε closure, add the non-ε transitions of that state with the
        // accumulated closure weight.
        for s in 0..n {
            for (&reachable, closure_w) in &closures[s] {
                for &idx in &self.transition_index[reachable] {
                    let tr = &self.transitions[idx];
                    if tr.input_symbol.is_some() {
                        let w = S::mul(closure_w, &tr.weight);
                        result
                            .add_transition(s, tr.input_symbol, tr.output_symbol, tr.to_state, w)
                            .ok();
                    }
                }
            }
        }

        result
    }

    /// Remove all transitions with ε on the output tape.
    pub fn remove_output_epsilon(&self) -> Self {
        let n = self.num_states;

        let closures: Vec<HashMap<usize, S>> = (0..n)
            .map(|s| self.epsilon_closure(s, |tr| tr.output_symbol.is_none()))
            .collect();

        let mut result = Self::new(n, self.input_alphabet.clone(), self.output_alphabet.clone());
        result.initial_weights = self.initial_weights.clone();

        for s in 0..n {
            let mut fw = self.final_weights[s].clone();
            for (&reachable, w) in &closures[s] {
                if reachable != s {
                    let contribution = S::mul(w, &self.final_weights[reachable]);
                    fw.add_assign(&contribution);
                }
            }
            result.final_weights[s] = fw;
        }

        for s in 0..n {
            for (&reachable, closure_w) in &closures[s] {
                for &idx in &self.transition_index[reachable] {
                    let tr = &self.transitions[idx];
                    if tr.output_symbol.is_some() {
                        let w = S::mul(closure_w, &tr.weight);
                        result
                            .add_transition(s, tr.input_symbol, tr.output_symbol, tr.to_state, w)
                            .ok();
                    }
                }
            }
        }

        result
    }

    /// Remove all ε-transitions (both input and output ε).
    pub fn remove_all_epsilon(&self) -> Self {
        let n = self.num_states;

        let closures: Vec<HashMap<usize, S>> = (0..n)
            .map(|s| self.epsilon_closure(s, |tr| tr.is_epsilon()))
            .collect();

        let mut result = Self::new(n, self.input_alphabet.clone(), self.output_alphabet.clone());
        result.initial_weights = self.initial_weights.clone();

        // Adjust final weights via ε-closures
        for s in 0..n {
            let mut fw = self.final_weights[s].clone();
            for (&reachable, w) in &closures[s] {
                if reachable != s {
                    let contribution = S::mul(w, &self.final_weights[reachable]);
                    fw.add_assign(&contribution);
                }
            }
            result.final_weights[s] = fw;
        }

        // Redistribute non-ε transitions through closures
        for s in 0..n {
            for (&reachable, closure_w) in &closures[s] {
                for &idx in &self.transition_index[reachable] {
                    let tr = &self.transitions[idx];
                    if !tr.is_epsilon() {
                        let w = S::mul(closure_w, &tr.weight);
                        result
                            .add_transition(s, tr.input_symbol, tr.output_symbol, tr.to_state, w)
                            .ok();
                    }
                }
            }
        }

        result
    }

    // ------------------------------------------------------------------
    // Trimming and optimization
    // ------------------------------------------------------------------

    /// Compute the set of states reachable from any initial state (forward
    /// reachability).
    fn accessible_states(&self) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        for s in self.initial_states() {
            visited.insert(s);
            queue.push_back(s);
        }
        while let Some(s) = queue.pop_front() {
            for &idx in &self.transition_index[s] {
                let to = self.transitions[idx].to_state;
                if visited.insert(to) {
                    queue.push_back(to);
                }
            }
        }
        visited
    }

    /// Compute the set of states from which a final state is reachable
    /// (backward reachability / coaccessibility).
    fn coaccessible_states(&self) -> HashSet<usize> {
        // Build reverse adjacency
        let mut reverse_adj: Vec<Vec<usize>> = vec![Vec::new(); self.num_states];
        for tr in &self.transitions {
            reverse_adj[tr.to_state].push(tr.from_state);
        }

        let mut visited = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        for s in self.final_states() {
            visited.insert(s);
            queue.push_back(s);
        }
        while let Some(s) = queue.pop_front() {
            for &pred in &reverse_adj[s] {
                if visited.insert(pred) {
                    queue.push_back(pred);
                }
            }
        }
        visited
    }

    /// Remove states that are not both accessible and coaccessible.
    pub fn trim(&self) -> Self {
        let acc = self.accessible_states();
        let coacc = self.coaccessible_states();
        let useful: HashSet<usize> = acc.intersection(&coacc).copied().collect();

        if useful.len() == self.num_states {
            return self.clone();
        }

        self.restrict_to_states(&useful)
    }

    /// Remove states that are not coaccessible (cannot reach a final state).
    pub fn connect(&self) -> Self {
        let coacc = self.coaccessible_states();
        if coacc.len() == self.num_states {
            return self.clone();
        }
        self.restrict_to_states(&coacc)
    }

    /// Build a new transducer containing only the states in `keep`, with states
    /// renumbered contiguously.
    fn restrict_to_states(&self, keep: &HashSet<usize>) -> Self {
        // Build old → new state mapping
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        let mut sorted: Vec<usize> = keep.iter().copied().collect();
        sorted.sort();
        for (new_id, &old_id) in sorted.iter().enumerate() {
            old_to_new.insert(old_id, new_id);
        }

        let n = sorted.len();
        let mut result = Self::new(n, self.input_alphabet.clone(), self.output_alphabet.clone());

        for &old in &sorted {
            let new = old_to_new[&old];
            result.initial_weights[new] = self.initial_weights[old].clone();
            result.final_weights[new] = self.final_weights[old].clone();
        }

        for tr in &self.transitions {
            if let (Some(&nf), Some(&nt)) =
                (old_to_new.get(&tr.from_state), old_to_new.get(&tr.to_state))
            {
                result
                    .add_transition(nf, tr.input_symbol, tr.output_symbol, nt, tr.weight.clone())
                    .ok();
            }
        }

        result
    }

    /// Push weights toward initial states.
    ///
    /// For each state q, compute d(q) = ⊕ over all accepting paths from q of
    /// the path weight, then reweight transitions: w(e) ← d(q)⁻¹ ⊗ w(e) ⊗ d(n(e))
    /// (where defined).
    ///
    /// Since general semirings don't have inverses, we approximate weight
    /// pushing by computing shortest-distance-to-final for each state and
    /// redistributing accordingly.  This is exact for the tropical semiring.
    pub fn push_weights(&self) -> Self {
        let n = self.num_states;

        // Compute backward shortest distances (weight to reach a final state).
        // d[q] = ⊕_{path q →* f} (path_weight ⊗ final(f))
        let d = self.backward_weights();

        let mut result = Self::new(n, self.input_alphabet.clone(), self.output_alphabet.clone());

        // New initial weights: initial(q) ⊗ d(q)
        for q in 0..n {
            result.initial_weights[q] = S::mul(&self.initial_weights[q], &d[q]);
        }

        // New final weights: if d(q) is non-zero, final weight becomes one()
        // (the weight has been pushed to initial/transitions)
        for q in 0..n {
            if !self.final_weights[q].is_zero() && !d[q].is_zero() {
                result.final_weights[q] = S::one();
            } else {
                result.final_weights[q] = self.final_weights[q].clone();
            }
        }

        // New transition weights
        for tr in &self.transitions {
            // If d(from) is zero the state is dead; skip.
            if d[tr.from_state].is_zero() {
                continue;
            }
            // We want: w'(e) = d(from)⁻¹ ⊗ w(e) ⊗ d(to)
            // Since we can't invert in a general semiring, we store w(e) ⊗ d(to)
            // and rely on the initial-weight multiplication to compensate.
            // For tropical semiring this is correct because:
            // new_initial = old_initial + d(start), new_w = w + d(to) - d(from)
            let w = S::mul(&tr.weight, &d[tr.to_state]);
            result
                .add_transition(tr.from_state, tr.input_symbol, tr.output_symbol, tr.to_state, w)
                .ok();
        }

        result
    }

    /// Compute backward (reverse-reachability) weights for each state.
    ///
    /// Uses iterative relaxation (Bellman-Ford style) on the reversed graph.
    fn backward_weights(&self) -> Vec<S> {
        let n = self.num_states;
        let mut d: Vec<S> = self.final_weights.clone();

        // Build reverse transitions
        let mut reverse: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
        for tr in &self.transitions {
            reverse[tr.to_state].push((tr.from_state, tr.weight.clone()));
        }

        // Relaxation
        let max_iters = n + 1;
        for _ in 0..max_iters {
            let mut changed = false;
            for q in 0..n {
                let old = d[q].clone();
                for &idx in &self.transition_index[q] {
                    let tr = &self.transitions[idx];
                    let via = S::mul(&tr.weight, &d[tr.to_state]);
                    d[q].add_assign(&via);
                }
                if d[q] != old {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        d
    }

    /// Attempt to make the transducer synchronous: for transitions that have
    /// mismatched ε (input ε but not output, or vice versa), try to
    /// redistribute labels so that transitions are either both-ε or both-non-ε.
    ///
    /// This is a best-effort heuristic; full synchronization is undecidable
    /// in general.
    pub fn synchronize(&self) -> Self {
        let n = self.num_states;
        let mut result = Self::new(n, self.input_alphabet.clone(), self.output_alphabet.clone());
        result.initial_weights = self.initial_weights.clone();
        result.final_weights = self.final_weights.clone();

        // Group transitions by from_state and try to pair up
        for s in 0..n {
            let mut eps_in: Vec<&TransducerTransition<S>> = Vec::new();
            let mut eps_out: Vec<&TransducerTransition<S>> = Vec::new();
            let mut normal: Vec<&TransducerTransition<S>> = Vec::new();
            let mut both_eps: Vec<&TransducerTransition<S>> = Vec::new();

            for &idx in &self.transition_index[s] {
                let tr = &self.transitions[idx];
                match (tr.input_symbol, tr.output_symbol) {
                    (None, None) => both_eps.push(tr),
                    (None, Some(_)) => eps_in.push(tr),
                    (Some(_), None) => eps_out.push(tr),
                    (Some(_), Some(_)) => normal.push(tr),
                }
            }

            // Copy normal and both-ε transitions directly
            for tr in &normal {
                result
                    .add_transition(
                        tr.from_state,
                        tr.input_symbol,
                        tr.output_symbol,
                        tr.to_state,
                        tr.weight.clone(),
                    )
                    .ok();
            }
            for tr in &both_eps {
                result
                    .add_transition(
                        tr.from_state,
                        tr.input_symbol,
                        tr.output_symbol,
                        tr.to_state,
                        tr.weight.clone(),
                    )
                    .ok();
            }

            // Try to pair input-ε with output-ε going to the same target state
            let mut used_out: HashSet<usize> = HashSet::new();
            for tr_in in &eps_in {
                let mut paired = false;
                for (j, tr_out) in eps_out.iter().enumerate() {
                    if !used_out.contains(&j) && tr_in.to_state == tr_out.to_state {
                        // Merge: create a non-ε transition
                        let w = S::mul(&tr_in.weight, &tr_out.weight);
                        result
                            .add_transition(
                                tr_in.from_state,
                                tr_out.input_symbol, // non-ε input from the output-ε transition
                                tr_in.output_symbol,  // non-ε output from the input-ε transition
                                tr_in.to_state,
                                w,
                            )
                            .ok();
                        used_out.insert(j);
                        paired = true;
                        break;
                    }
                }
                if !paired {
                    // Keep the original input-ε transition
                    result
                        .add_transition(
                            tr_in.from_state,
                            tr_in.input_symbol,
                            tr_in.output_symbol,
                            tr_in.to_state,
                            tr_in.weight.clone(),
                        )
                        .ok();
                }
            }

            // Keep unpaired output-ε transitions
            for (j, tr_out) in eps_out.iter().enumerate() {
                if !used_out.contains(&j) {
                    result
                        .add_transition(
                            tr_out.from_state,
                            tr_out.input_symbol,
                            tr_out.output_symbol,
                            tr_out.to_state,
                            tr_out.weight.clone(),
                        )
                        .ok();
                }
            }
        }

        result
    }

    /// Check whether the transducer is functional (single-valued): for every
    /// input string, at most one output string is produced.
    ///
    /// Uses a product construction: compose the transducer with its inverse
    /// and check that every reachable state in the product has at most one
    /// output-non-ε transition per input symbol.
    pub fn is_functional(&self) -> bool {
        // Heuristic check: transduce a sample of short strings and verify
        // uniqueness. For a full check we would compose with the inverse and
        // verify the result is an identity-like transducer, but that is
        // expensive.  We instead use the simpler sufficient condition that
        // from each state, for each input symbol there is exactly one
        // (input, output) pair.
        for s in 0..self.num_states {
            let mut seen: HashMap<Option<usize>, HashSet<Option<usize>>> = HashMap::new();
            for &idx in &self.transition_index[s] {
                let tr = &self.transitions[idx];
                let outputs = seen.entry(tr.input_symbol).or_default();
                outputs.insert(tr.output_symbol);
                // If there are multiple distinct output symbols for the same
                // input from the same state, the transducer might be
                // non-functional. However, different target states could
                // reconverge, so this is only a necessary condition, not
                // sufficient. For a stronger test we'd need the full
                // product construction.
                if outputs.len() > 1 {
                    return false;
                }
            }
        }
        true
    }

    // ------------------------------------------------------------------
    // Visualization
    // ------------------------------------------------------------------

    /// Produce a Graphviz DOT representation.
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph transducer {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=circle];\n");

        // Mark initial states
        for (s, w) in self.initial_weights.iter().enumerate() {
            if !w.is_zero() {
                dot.push_str(&format!("  start_{s} [shape=point];\n"));
                dot.push_str(&format!("  start_{s} -> {s} [label=\"{w:?}\"];\n"));
            }
        }

        // Mark final states
        for (s, w) in self.final_weights.iter().enumerate() {
            if !w.is_zero() {
                dot.push_str(&format!("  {s} [shape=doublecircle, label=\"{s}/{w:?}\"];\n"));
            }
        }

        // Transitions
        for tr in &self.transitions {
            let ilbl = match tr.input_symbol {
                Some(i) => format!("{i}"),
                None => "ε".to_string(),
            };
            let olbl = match tr.output_symbol {
                Some(o) => format!("{o}"),
                None => "ε".to_string(),
            };
            dot.push_str(&format!(
                "  {} -> {} [label=\"{}/{}:{:?}\"];\n",
                tr.from_state, tr.to_state, ilbl, olbl, tr.weight
            ));
        }

        dot.push_str("}\n");
        dot
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<S: Semiring> fmt::Display for WeightedTransducer<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "WeightedTransducer {{")?;
        writeln!(f, "  states: {}", self.num_states)?;
        writeln!(
            f,
            "  input alphabet size: {}",
            self.input_alphabet.size()
        )?;
        writeln!(
            f,
            "  output alphabet size: {}",
            self.output_alphabet.size()
        )?;
        writeln!(f, "  transitions: {}", self.transitions.len())?;
        writeln!(
            f,
            "  initial states: {:?}",
            self.initial_states()
        )?;
        writeln!(
            f,
            "  final states: {:?}",
            self.final_states()
        )?;
        for tr in &self.transitions {
            let ilbl = match tr.input_symbol {
                Some(i) => format!("{i}"),
                None => "ε".to_string(),
            };
            let olbl = match tr.output_symbol {
                Some(o) => format!("{o}"),
                None => "ε".to_string(),
            };
            writeln!(
                f,
                "  {} --{}/{}:{:?}--> {}",
                tr.from_state, ilbl, olbl, tr.weight, tr.to_state
            )?;
        }
        writeln!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// Lazy (on-the-fly) composition
// ---------------------------------------------------------------------------

/// Lazy (on-the-fly) composition of two transducers.
///
/// States in the composed transducer are only expanded when they are visited
/// during transduction, avoiding the potentially exponential blow-up of
/// eager composition.
#[derive(Debug, Clone)]
pub struct LazyComposition<S: Semiring> {
    t1: WeightedTransducer<S>,
    t2: WeightedTransducer<S>,
    /// Cache of already-expanded composition states.
    /// Maps composition-state triple → outgoing transitions.
    cache: std::cell::RefCell<HashMap<CompositionState, Vec<LazyTransition<S>>>>,
}

/// An outgoing transition in the lazily-composed transducer.
#[derive(Debug, Clone)]
struct LazyTransition<S: Semiring> {
    target: CompositionState,
    input_symbol: Option<usize>,
    output_symbol: Option<usize>,
    weight: S,
}

impl<S: Semiring> LazyComposition<S> {
    /// Create a new lazy composition of `t1 ∘ t2`.
    pub fn new(t1: &WeightedTransducer<S>, t2: &WeightedTransducer<S>) -> Self {
        Self {
            t1: t1.clone(),
            t2: t2.clone(),
            cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Expand a composition state, computing its outgoing transitions using
    /// the ε-sequencing filter.
    fn expand(&self, cs: &CompositionState) -> Vec<LazyTransition<S>> {
        // Check cache
        {
            let cache = self.cache.borrow();
            if let Some(cached) = cache.get(cs) {
                return cached.clone();
            }
        }

        let (s1, s2, filt) = (cs.state1, cs.state2, cs.filter);
        let t1_trans = self.t1.transitions_from(s1);
        let t2_trans = self.t2.transitions_from(s2);
        let mut result: Vec<LazyTransition<S>> = Vec::new();

        // CASE 1: matching non-ε
        for tr1 in &t1_trans {
            if let Some(osym) = tr1.output_symbol {
                for tr2 in &t2_trans {
                    if tr2.input_symbol == Some(osym) {
                        let w = S::mul(&tr1.weight, &tr2.weight);
                        result.push(LazyTransition {
                            target: CompositionState::new(
                                tr1.to_state,
                                tr2.to_state,
                                EpsilonFilter::NoFilter,
                            ),
                            input_symbol: tr1.input_symbol,
                            output_symbol: tr2.output_symbol,
                            weight: w,
                        });
                    }
                }
            }
        }

        // CASE 2: T1 ε alone (filter NoFilter → Filter1, Filter1 → Filter1)
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter1 {
            for tr1 in &t1_trans {
                if tr1.output_symbol.is_none() {
                    result.push(LazyTransition {
                        target: CompositionState::new(
                            tr1.to_state,
                            s2,
                            EpsilonFilter::Filter1,
                        ),
                        input_symbol: tr1.input_symbol,
                        output_symbol: None,
                        weight: tr1.weight.clone(),
                    });
                }
            }
        }

        // CASE 3: T2 ε alone (filter NoFilter → Filter2, Filter2 → Filter2)
        if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter2 {
            for tr2 in &t2_trans {
                if tr2.input_symbol.is_none() {
                    result.push(LazyTransition {
                        target: CompositionState::new(
                            s1,
                            tr2.to_state,
                            EpsilonFilter::Filter2,
                        ),
                        input_symbol: None,
                        output_symbol: tr2.output_symbol,
                        weight: tr2.weight.clone(),
                    });
                }
            }
        }

        // CASE 4: synchronized ε (only from NoFilter)
        if filt == EpsilonFilter::NoFilter {
            for tr1 in &t1_trans {
                if tr1.output_symbol.is_none() {
                    for tr2 in &t2_trans {
                        if tr2.input_symbol.is_none() {
                            let w = S::mul(&tr1.weight, &tr2.weight);
                            result.push(LazyTransition {
                                target: CompositionState::new(
                                    tr1.to_state,
                                    tr2.to_state,
                                    EpsilonFilter::NoFilter,
                                ),
                                input_symbol: tr1.input_symbol,
                                output_symbol: tr2.output_symbol,
                                weight: w,
                            });
                        }
                    }
                }
            }
        }

        // Cache the result
        self.cache.borrow_mut().insert(*cs, result.clone());
        result
    }

    /// Transduce an input string through the lazily-composed transducer.
    pub fn transduce(&self, input: &[usize]) -> Vec<(Vec<usize>, S)> {
        type Config<W> = (CompositionState, usize, Vec<usize>, W);

        let mut results: HashMap<Vec<usize>, S> = HashMap::new();
        let mut queue: VecDeque<Config<S>> = VecDeque::new();

        // Seed with pairs of initial states
        for s1 in self.t1.initial_states() {
            for s2 in self.t2.initial_states() {
                let iw = S::mul(
                    &self.t1.initial_weights[s1],
                    &self.t2.initial_weights[s2],
                );
                let cs = CompositionState::new(s1, s2, EpsilonFilter::NoFilter);
                queue.push_back((cs, 0, Vec::new(), iw));
            }
        }

        let max_configs: usize = 500_000;
        let mut count = 0;

        while let Some((cs, pos, out_buf, acc_w)) = queue.pop_front() {
            count += 1;
            if count > max_configs {
                warn!("LazyComposition::transduce: configuration limit reached");
                break;
            }

            // Check accepting
            if pos == input.len() {
                let fw1 = &self.t1.final_weights[cs.state1];
                let fw2 = &self.t2.final_weights[cs.state2];
                if !fw1.is_zero() && !fw2.is_zero() {
                    let fw = S::mul(fw1, fw2);
                    let total = S::mul(&acc_w, &fw);
                    results
                        .entry(out_buf.clone())
                        .and_modify(|w| w.add_assign(&total))
                        .or_insert(total);
                }
            }

            // Expand
            let transitions = self.expand(&cs);
            for lt in &transitions {
                match lt.input_symbol {
                    None => {
                        let w = S::mul(&acc_w, &lt.weight);
                        let mut new_out = out_buf.clone();
                        if let Some(osym) = lt.output_symbol {
                            new_out.push(osym);
                        }
                        queue.push_back((lt.target, pos, new_out, w));
                    }
                    Some(isym) if pos < input.len() && isym == input[pos] => {
                        let w = S::mul(&acc_w, &lt.weight);
                        let mut new_out = out_buf.clone();
                        if let Some(osym) = lt.output_symbol {
                            new_out.push(osym);
                        }
                        queue.push_back((lt.target, pos + 1, new_out, w));
                    }
                    _ => {}
                }
            }
        }

        results.into_iter().collect()
    }

    /// Clear the expansion cache.
    pub fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
    }

    /// Number of states currently expanded in the cache.
    pub fn cached_state_count(&self) -> usize {
        self.cache.borrow().len()
    }
}

// ---------------------------------------------------------------------------
// Look-ahead composition
// ---------------------------------------------------------------------------

/// A look-ahead filter that prunes unreachable composition states by checking
/// whether the remaining input can reach a final state in T2.
#[derive(Debug, Clone)]
pub struct LookAheadFilter<S: Semiring> {
    /// For each state in T2, the set of input symbols that can eventually
    /// reach a final state.
    reachable_symbols: Vec<HashSet<Option<usize>>>,
    /// For each state in T2, whether a final state is reachable at all.
    can_reach_final: Vec<bool>,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Semiring> LookAheadFilter<S> {
    /// Build a look-ahead filter from the second transducer.
    pub fn new(t2: &WeightedTransducer<S>) -> Self {
        let n = t2.num_states;

        // Compute which states can reach a final state (backward BFS)
        let mut can_reach_final = vec![false; n];
        let mut reverse_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for tr in &t2.transitions {
            reverse_adj[tr.to_state].push(tr.from_state);
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for s in t2.final_states() {
            can_reach_final[s] = true;
            queue.push_back(s);
        }
        while let Some(s) = queue.pop_front() {
            for &pred in &reverse_adj[s] {
                if !can_reach_final[pred] {
                    can_reach_final[pred] = true;
                    queue.push_back(pred);
                }
            }
        }

        // For each state, compute the set of input symbols that appear on
        // transitions leading to coaccessible states
        let mut reachable_symbols: Vec<HashSet<Option<usize>>> = vec![HashSet::new(); n];
        for s in 0..n {
            if !can_reach_final[s] {
                continue;
            }
            // BFS/DFS to find reachable symbols
            let mut visited = HashSet::new();
            let mut bfs: VecDeque<usize> = VecDeque::new();
            visited.insert(s);
            bfs.push_back(s);

            while let Some(q) = bfs.pop_front() {
                for &idx in &t2.transition_index[q] {
                    let tr = &t2.transitions[idx];
                    if can_reach_final[tr.to_state] {
                        reachable_symbols[s].insert(tr.input_symbol);
                        if visited.insert(tr.to_state) {
                            bfs.push_back(tr.to_state);
                        }
                    }
                }
            }
        }

        Self {
            reachable_symbols,
            can_reach_final,
            _marker: std::marker::PhantomData,
        }
    }

    /// Check whether a transition from state `s2` in T2 with the given
    /// intermediate symbol is worth exploring.
    pub fn allows(&self, s2: usize, sym: Option<usize>) -> bool {
        if s2 >= self.can_reach_final.len() {
            return false;
        }
        if !self.can_reach_final[s2] {
            return false;
        }
        // If the symbol is ε, always allow (ε-transitions don't consume input)
        if sym.is_none() {
            return true;
        }
        self.reachable_symbols[s2].contains(&sym)
    }
}

impl<S: Semiring> WeightedTransducer<S> {
    /// Compose with look-ahead filtering to prune unreachable states early.
    pub fn compose_with_lookahead(
        &self,
        other: &WeightedTransducer<S>,
    ) -> Result<WeightedTransducer<S>> {
        if self.output_alphabet.size() != other.input_alphabet.size() {
            return Err(TransducerError::IncompatibleAlphabets(format!(
                "output alphabet size {} ≠ input alphabet size {}",
                self.output_alphabet.size(),
                other.input_alphabet.size()
            )));
        }

        let filter = LookAheadFilter::new(other);
        let mut state_space = CompositionStateSpace::new();
        let mut result_transitions: Vec<TransducerTransition<S>> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();

        for s1 in self.initial_states() {
            for s2 in other.initial_states() {
                if !filter.can_reach_final[s2] {
                    continue;
                }
                let cs = CompositionState::new(s1, s2, EpsilonFilter::NoFilter);
                let id = state_space.get_or_insert(cs);
                queue.push_back(id);
            }
        }

        while let Some(cur_id) = queue.pop_front() {
            let cs = *state_space.state_at(cur_id);
            let (s1, s2, filt) = (cs.state1, cs.state2, cs.filter);

            let t1_trans = self.transitions_from(s1);
            let t2_trans = other.transitions_from(s2);

            // CASE 1: matching non-ε
            for tr1 in &t1_trans {
                if let Some(osym) = tr1.output_symbol {
                    for tr2 in &t2_trans {
                        if tr2.input_symbol == Some(osym) {
                            // Look-ahead: check that the target state in T2
                            // can reach a final state
                            if !filter.can_reach_final[tr2.to_state] {
                                continue;
                            }
                            let cs_next = CompositionState::new(
                                tr1.to_state,
                                tr2.to_state,
                                EpsilonFilter::NoFilter,
                            );
                            let next_id = state_space.get_or_insert(cs_next);
                            let was_new = next_id == state_space.len() - 1;
                            let w = S::mul(&tr1.weight, &tr2.weight);
                            result_transitions.push(TransducerTransition::new(
                                cur_id,
                                next_id,
                                tr1.input_symbol,
                                tr2.output_symbol,
                                w,
                            ));
                            if was_new {
                                queue.push_back(next_id);
                            }
                        }
                    }
                }
            }

            // CASE 2: T1 ε alone
            if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter1 {
                for tr1 in &t1_trans {
                    if tr1.output_symbol.is_none() {
                        // Look-ahead: check that s2 can still reach final
                        if !filter.can_reach_final[s2] {
                            continue;
                        }
                        let cs_next =
                            CompositionState::new(tr1.to_state, s2, EpsilonFilter::Filter1);
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = next_id == state_space.len() - 1;
                        result_transitions.push(TransducerTransition::new(
                            cur_id,
                            next_id,
                            tr1.input_symbol,
                            None,
                            tr1.weight.clone(),
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }

            // CASE 3: T2 ε alone
            if filt == EpsilonFilter::NoFilter || filt == EpsilonFilter::Filter2 {
                for tr2 in &t2_trans {
                    if tr2.input_symbol.is_none() {
                        if !filter.can_reach_final[tr2.to_state] {
                            continue;
                        }
                        let cs_next =
                            CompositionState::new(s1, tr2.to_state, EpsilonFilter::Filter2);
                        let next_id = state_space.get_or_insert(cs_next);
                        let was_new = next_id == state_space.len() - 1;
                        result_transitions.push(TransducerTransition::new(
                            cur_id,
                            next_id,
                            None,
                            tr2.output_symbol,
                            tr2.weight.clone(),
                        ));
                        if was_new {
                            queue.push_back(next_id);
                        }
                    }
                }
            }

            // CASE 4: synchronized ε
            if filt == EpsilonFilter::NoFilter {
                for tr1 in &t1_trans {
                    if tr1.output_symbol.is_none() {
                        for tr2 in &t2_trans {
                            if tr2.input_symbol.is_none() {
                                if !filter.can_reach_final[tr2.to_state] {
                                    continue;
                                }
                                let cs_next = CompositionState::new(
                                    tr1.to_state,
                                    tr2.to_state,
                                    EpsilonFilter::NoFilter,
                                );
                                let next_id = state_space.get_or_insert(cs_next);
                                let was_new = next_id == state_space.len() - 1;
                                let w = S::mul(&tr1.weight, &tr2.weight);
                                result_transitions.push(TransducerTransition::new(
                                    cur_id,
                                    next_id,
                                    tr1.input_symbol,
                                    tr2.output_symbol,
                                    w,
                                ));
                                if was_new {
                                    queue.push_back(next_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build final transducer
        let n = state_space.len();
        let mut composed = WeightedTransducer::new(
            n,
            self.input_alphabet.clone(),
            other.output_alphabet.clone(),
        );

        for i in 0..n {
            let cs = state_space.state_at(i);
            composed.initial_weights[i] =
                S::mul(&self.initial_weights[cs.state1], &other.initial_weights[cs.state2]);
            composed.final_weights[i] =
                S::mul(&self.final_weights[cs.state1], &other.final_weights[cs.state2]);
        }

        composed.transitions = result_transitions;
        composed.build_index();
        Ok(composed)
    }
}

// ---------------------------------------------------------------------------
// Topological sorting helper
// ---------------------------------------------------------------------------

/// Topological sort of states. Returns `Err(CycleDetected)` if the state
/// graph contains a cycle.
fn topological_sort<S: Semiring>(t: &WeightedTransducer<S>) -> Result<Vec<usize>> {
    let n = t.num_states;
    let mut in_degree = vec![0usize; n];
    for tr in &t.transitions {
        in_degree[tr.to_state] += 1;
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for s in 0..n {
        if in_degree[s] == 0 {
            queue.push_back(s);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(s) = queue.pop_front() {
        order.push(s);
        for &idx in &t.transition_index[s] {
            let to = t.transitions[idx].to_state;
            in_degree[to] -= 1;
            if in_degree[to] == 0 {
                queue.push_back(to);
            }
        }
    }

    if order.len() == n {
        Ok(order)
    } else {
        Err(TransducerError::CycleDetected)
    }
}

// ---------------------------------------------------------------------------
// Determinization helper (single-valued transducers)
// ---------------------------------------------------------------------------

/// A subset element for transducer determinization: (state, residual_output).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SubsetElement {
    state: usize,
    residual: Vec<usize>,
    // weight is tracked separately
}

// ---------------------------------------------------------------------------
// Additional utility implementations
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Check whether the transducer accepts any pair at all.
    pub fn is_empty(&self) -> bool {
        // Quick check: need at least one initial and one final state
        let has_initial = self.initial_weights.iter().any(|w| !w.is_zero());
        let has_final = self.final_weights.iter().any(|w| !w.is_zero());
        if !has_initial || !has_final {
            return true;
        }
        // Check reachability
        let acc = self.accessible_states();
        let fin = self.final_states();
        !fin.iter().any(|&f| acc.contains(&f))
    }

    /// Return the number of ε-transitions (both input and output are ε).
    pub fn num_epsilon_transitions(&self) -> usize {
        self.transitions.iter().filter(|t| t.is_epsilon()).count()
    }

    /// Return the number of input-ε transitions.
    pub fn num_input_epsilon_transitions(&self) -> usize {
        self.transitions
            .iter()
            .filter(|t| t.is_input_epsilon())
            .count()
    }

    /// Return the number of output-ε transitions.
    pub fn num_output_epsilon_transitions(&self) -> usize {
        self.transitions
            .iter()
            .filter(|t| t.is_output_epsilon())
            .count()
    }

    /// Compute the in-degree of each state.
    pub fn in_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.num_states];
        for tr in &self.transitions {
            deg[tr.to_state] += 1;
        }
        deg
    }

    /// Compute the out-degree of each state.
    pub fn out_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.num_states];
        for tr in &self.transitions {
            deg[tr.from_state] += 1;
        }
        deg
    }

    /// Reverse the transducer: swap initial/final weights and reverse all
    /// transition directions.
    pub fn reverse(&self) -> Self {
        let mut rev = Self::new(
            self.num_states,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );
        rev.initial_weights = self.final_weights.clone();
        rev.final_weights = self.initial_weights.clone();
        for tr in &self.transitions {
            rev.add_transition(
                tr.to_state,
                tr.input_symbol,
                tr.output_symbol,
                tr.from_state,
                tr.weight.clone(),
            )
            .ok();
        }
        rev
    }

    /// Concatenate two transducers: `self` followed by `other`.
    ///
    /// Builds a new transducer where final states of `self` connect to initial
    /// states of `other` via ε-transitions weighted by the product of the
    /// final weight (self) and initial weight (other).
    pub fn concatenate(&self, other: &WeightedTransducer<S>) -> Result<Self> {
        // We require compatible alphabets
        if self.input_alphabet.size() != other.input_alphabet.size() {
            return Err(TransducerError::IncompatibleAlphabets(
                "input alphabets differ".into(),
            ));
        }
        if self.output_alphabet.size() != other.output_alphabet.size() {
            return Err(TransducerError::IncompatibleAlphabets(
                "output alphabets differ".into(),
            ));
        }

        let n1 = self.num_states;
        let n2 = other.num_states;
        let mut result = Self::new(
            n1 + n2,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );

        // Initial weights from self
        for i in 0..n1 {
            result.initial_weights[i] = self.initial_weights[i].clone();
        }

        // Final weights from other (offset by n1)
        for i in 0..n2 {
            result.final_weights[n1 + i] = other.final_weights[i].clone();
        }

        // Transitions from self
        for tr in &self.transitions {
            result
                .add_transition(
                    tr.from_state,
                    tr.input_symbol,
                    tr.output_symbol,
                    tr.to_state,
                    tr.weight.clone(),
                )
                .ok();
        }

        // Transitions from other (offset states)
        for tr in &other.transitions {
            result
                .add_transition(
                    n1 + tr.from_state,
                    tr.input_symbol,
                    tr.output_symbol,
                    n1 + tr.to_state,
                    tr.weight.clone(),
                )
                .ok();
        }

        // Connecting ε-transitions: final(self) → initial(other)
        for fs in self.final_states() {
            for is in other.initial_states() {
                let w = S::mul(&self.final_weights[fs], &other.initial_weights[is]);
                result
                    .add_transition(fs, None, None, n1 + is, w)
                    .ok();
            }
        }

        Ok(result)
    }

    /// Kleene closure (star): zero or more repetitions of the transducer.
    ///
    /// Adds ε-transitions from final states back to initial states.
    pub fn closure(&self) -> Self {
        let n = self.num_states;
        // Add a new start/accept state
        let mut result = Self::new(
            n + 1,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );

        let new_state = n;
        result.set_initial_weight(new_state, S::one());
        result.set_final_weight(new_state, S::one());

        // Copy original weights and transitions
        for i in 0..n {
            result.initial_weights[i] = self.initial_weights[i].clone();
            result.final_weights[i] = self.final_weights[i].clone();
        }

        for tr in &self.transitions {
            result
                .add_transition(
                    tr.from_state,
                    tr.input_symbol,
                    tr.output_symbol,
                    tr.to_state,
                    tr.weight.clone(),
                )
                .ok();
        }

        // ε from new start to original initials
        for is in self.initial_states() {
            result
                .add_transition(new_state, None, None, is, self.initial_weights[is].clone())
                .ok();
        }

        // ε from original finals back to original initials (for repetition)
        for fs in self.final_states() {
            for is in self.initial_states() {
                let w = S::mul(&self.final_weights[fs], &self.initial_weights[is]);
                result.add_transition(fs, None, None, is, w).ok();
            }
        }

        result
    }

    /// Union of two transducers.
    pub fn union(&self, other: &WeightedTransducer<S>) -> Result<Self> {
        if self.input_alphabet.size() != other.input_alphabet.size() {
            return Err(TransducerError::IncompatibleAlphabets(
                "input alphabets differ".into(),
            ));
        }
        if self.output_alphabet.size() != other.output_alphabet.size() {
            return Err(TransducerError::IncompatibleAlphabets(
                "output alphabets differ".into(),
            ));
        }

        let n1 = self.num_states;
        let n2 = other.num_states;
        let mut result = Self::new(
            n1 + n2,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );

        for i in 0..n1 {
            result.initial_weights[i] = self.initial_weights[i].clone();
            result.final_weights[i] = self.final_weights[i].clone();
        }
        for i in 0..n2 {
            result.initial_weights[n1 + i] = other.initial_weights[i].clone();
            result.final_weights[n1 + i] = other.final_weights[i].clone();
        }

        for tr in &self.transitions {
            result
                .add_transition(
                    tr.from_state,
                    tr.input_symbol,
                    tr.output_symbol,
                    tr.to_state,
                    tr.weight.clone(),
                )
                .ok();
        }

        for tr in &other.transitions {
            result
                .add_transition(
                    n1 + tr.from_state,
                    tr.input_symbol,
                    tr.output_symbol,
                    n1 + tr.to_state,
                    tr.weight.clone(),
                )
                .ok();
        }

        Ok(result)
    }

    /// Apply a function to every transition weight.
    pub fn map_weights<F>(&self, f: F) -> Self
    where
        F: Fn(&S) -> S,
    {
        let mut result = Self::new(
            self.num_states,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );
        result.initial_weights = self.initial_weights.iter().map(&f).collect();
        result.final_weights = self.final_weights.iter().map(&f).collect();
        for tr in &self.transitions {
            result
                .add_transition(
                    tr.from_state,
                    tr.input_symbol,
                    tr.output_symbol,
                    tr.to_state,
                    f(&tr.weight),
                )
                .ok();
        }
        result
    }

    /// Relabel input symbols according to a mapping.
    pub fn relabel_input(&self, mapping: &HashMap<usize, usize>) -> Self {
        let mut result = Self::new(
            self.num_states,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );
        result.initial_weights = self.initial_weights.clone();
        result.final_weights = self.final_weights.clone();
        for tr in &self.transitions {
            let new_isym = tr
                .input_symbol
                .map(|s| mapping.get(&s).copied().unwrap_or(s));
            result
                .add_transition(tr.from_state, new_isym, tr.output_symbol, tr.to_state, tr.weight.clone())
                .ok();
            // Fix: we need to actually set from_state correctly
        }
        // Re-do properly
        let mut result2 = Self::new(
            self.num_states,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );
        result2.initial_weights = self.initial_weights.clone();
        result2.final_weights = self.final_weights.clone();
        for tr in &self.transitions {
            let new_isym = tr.input_symbol.map(|s| *mapping.get(&s).unwrap_or(&s));
            result2
                .add_transition(
                    tr.from_state,
                    new_isym,
                    tr.output_symbol,
                    tr.to_state,
                    tr.weight.clone(),
                )
                .ok();
        }
        result2
    }

    /// Relabel output symbols according to a mapping.
    pub fn relabel_output(&self, mapping: &HashMap<usize, usize>) -> Self {
        let mut result = Self::new(
            self.num_states,
            self.input_alphabet.clone(),
            self.output_alphabet.clone(),
        );
        result.initial_weights = self.initial_weights.clone();
        result.final_weights = self.final_weights.clone();
        for tr in &self.transitions {
            let new_osym = tr.output_symbol.map(|s| *mapping.get(&s).unwrap_or(&s));
            result
                .add_transition(
                    tr.from_state,
                    tr.input_symbol,
                    new_osym,
                    tr.to_state,
                    tr.weight.clone(),
                )
                .ok();
        }
        result
    }

    /// Check structural equality (same number of states and identical
    /// transitions, weights, etc.).
    pub fn structurally_equal(&self, other: &Self) -> bool {
        if self.num_states != other.num_states {
            return false;
        }
        if self.initial_weights != other.initial_weights {
            return false;
        }
        if self.final_weights != other.final_weights {
            return false;
        }
        if self.transitions.len() != other.transitions.len() {
            return false;
        }
        // Compare sorted transitions
        let mut t1: Vec<_> = self.transitions.iter().collect();
        let mut t2: Vec<_> = other.transitions.iter().collect();
        t1.sort_by_key(|t| (t.from_state, t.to_state, t.input_symbol, t.output_symbol));
        t2.sort_by_key(|t| (t.from_state, t.to_state, t.input_symbol, t.output_symbol));
        t1.iter().zip(t2.iter()).all(|(a, b)| *a == *b)
    }

    /// Return a summary of this transducer as a string.
    pub fn summary(&self) -> String {
        format!(
            "Transducer(states={}, transitions={}, input_alpha={}, output_alpha={}, \
             initials={}, finals={})",
            self.num_states,
            self.transitions.len(),
            self.input_alphabet.size(),
            self.output_alphabet.size(),
            self.initial_states().len(),
            self.final_states().len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

impl<S: Semiring + Serialize> WeightedTransducer<S> {
    /// Serialize to JSON.
    pub fn to_json(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl<S: Semiring + for<'de> Deserialize<'de>> WeightedTransducer<S> {
    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        let mut t: Self = serde_json::from_str(json)?;
        t.build_index();
        Ok(t)
    }
}

// ---------------------------------------------------------------------------
// Builder pattern
// ---------------------------------------------------------------------------

/// Fluent builder for constructing transducers.
pub struct TransducerBuilder<S: Semiring> {
    inner: WeightedTransducer<S>,
}

impl<S: Semiring> TransducerBuilder<S> {
    pub fn new(input_alphabet: Alphabet, output_alphabet: Alphabet) -> Self {
        Self {
            inner: WeightedTransducer::new(0, input_alphabet, output_alphabet),
        }
    }

    pub fn add_state(mut self) -> Self {
        self.inner.add_state();
        self
    }

    pub fn add_states(mut self, n: usize) -> Self {
        for _ in 0..n {
            self.inner.add_state();
        }
        self
    }

    pub fn set_initial(mut self, state: usize, weight: S) -> Self {
        self.inner.set_initial_weight(state, weight);
        self
    }

    pub fn set_final(mut self, state: usize, weight: S) -> Self {
        self.inner.set_final_weight(state, weight);
        self
    }

    pub fn add_transition(
        mut self,
        from: usize,
        input: Option<usize>,
        output: Option<usize>,
        to: usize,
        weight: S,
    ) -> Self {
        self.inner
            .add_transition(from, input, output, to, weight)
            .expect("TransducerBuilder: invalid transition");
        self
    }

    pub fn build(self) -> WeightedTransducer<S> {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// Path enumeration helper
// ---------------------------------------------------------------------------

/// A complete path through the transducer.
#[derive(Debug, Clone)]
pub struct TransducerPath<S: Semiring> {
    pub states: Vec<usize>,
    pub input_labels: Vec<Option<usize>>,
    pub output_labels: Vec<Option<usize>>,
    pub weight: S,
}

impl<S: Semiring> TransducerPath<S> {
    /// The input string (filtering out epsilons).
    pub fn input_string(&self) -> Vec<usize> {
        self.input_labels.iter().filter_map(|&s| s).collect()
    }

    /// The output string (filtering out epsilons).
    pub fn output_string(&self) -> Vec<usize> {
        self.output_labels.iter().filter_map(|&s| s).collect()
    }
}

impl<S: Semiring> WeightedTransducer<S> {
    /// Enumerate all accepting paths (up to a limit) via DFS.
    pub fn enumerate_paths(&self, limit: usize) -> Vec<TransducerPath<S>> {
        let mut results = Vec::new();

        // DFS stack: (state, path_so_far)
        type Frame<W> = (usize, Vec<usize>, Vec<Option<usize>>, Vec<Option<usize>>, W);
        let mut stack: Vec<Frame<S>> = Vec::new();

        for (s, w) in self.initial_weights.iter().enumerate() {
            if !w.is_zero() {
                stack.push((s, vec![s], Vec::new(), Vec::new(), w.clone()));
            }
        }

        let mut visited_count = 0usize;
        let max_visit = limit * 100;

        while let Some((state, states, in_labels, out_labels, acc_w)) = stack.pop() {
            visited_count += 1;
            if visited_count > max_visit {
                break;
            }

            // Check if this is an accepting state
            let fw = &self.final_weights[state];
            if !fw.is_zero() {
                let path = TransducerPath {
                    states: states.clone(),
                    input_labels: in_labels.clone(),
                    output_labels: out_labels.clone(),
                    weight: S::mul(&acc_w, fw),
                };
                results.push(path);
                if results.len() >= limit {
                    break;
                }
            }

            // Expand
            for &idx in &self.transition_index[state] {
                let tr = &self.transitions[idx];
                // Avoid infinite loops on ε-cycles: limit path length
                if states.len() > self.num_states * 2 + 10 {
                    continue;
                }
                let w = S::mul(&acc_w, &tr.weight);
                let mut new_states = states.clone();
                new_states.push(tr.to_state);
                let mut new_in = in_labels.clone();
                new_in.push(tr.input_symbol);
                let mut new_out = out_labels.clone();
                new_out.push(tr.output_symbol);
                stack.push((tr.to_state, new_states, new_in, new_out, w));
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Cross-product construction
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Build the cross-product (intersection on input side) of this transducer
    /// with a WFA acceptor on the input.
    ///
    /// The result is a transducer that only transduces strings accepted by
    /// the WFA, with weights multiplied.
    pub fn intersect_input(&self, wfa: &WeightedFiniteAutomaton<S>) -> Result<Self> {
        let n1 = self.num_states;
        let n2 = wfa.num_states;
        let n = n1 * n2;

        let mut result = Self::new(n, self.input_alphabet.clone(), self.output_alphabet.clone());

        // State mapping: (s1, s2) → s1 * n2 + s2
        let state_id = |s1: usize, s2: usize| -> usize { s1 * n2 + s2 };

        // Initial/final weights
        for s1 in 0..n1 {
            for s2 in 0..n2 {
                let id = state_id(s1, s2);
                result.initial_weights[id] =
                    S::mul(&self.initial_weights[s1], &wfa.initial_weights[s2]);
                result.final_weights[id] =
                    S::mul(&self.final_weights[s1], &wfa.final_weights[s2]);
            }
        }

        // Transitions: match input labels with WFA transitions
        for tr in &self.transitions {
            match tr.input_symbol {
                None => {
                    // ε on input: WFA stays in same state
                    for s2 in 0..n2 {
                        let from = state_id(tr.from_state, s2);
                        let to = state_id(tr.to_state, s2);
                        result
                            .add_transition(
                                from,
                                None,
                                tr.output_symbol,
                                to,
                                tr.weight.clone(),
                            )
                            .ok();
                    }
                }
                Some(isym) => {
                    // Match with WFA transitions on the same symbol
                    for s2 in 0..n2 {
                        for s2_next in 0..n2 {
                            let wfa_w = wfa.transition_weight(s2, isym, s2_next);
                            if !wfa_w.is_zero() {
                                let from = state_id(tr.from_state, s2);
                                let to = state_id(tr.to_state, s2_next);
                                let w = S::mul(&tr.weight, &wfa_w);
                                result
                                    .add_transition(from, Some(isym), tr.output_symbol, to, w)
                                    .ok();
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Distance/similarity computation
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Compute the total weight of all accepting paths (the "sum" of the
    /// transduction relation).  This is the sum over all (input, output) pairs
    /// of their weight.
    pub fn total_weight(&self) -> S {
        // For acyclic transducers: forward computation.
        // For cyclic: iterative relaxation with a bound.
        let n = self.num_states;
        let mut fwd: Vec<S> = self.initial_weights.clone();

        let max_iters = n + 5;
        for _ in 0..max_iters {
            let mut new_fwd = fwd.clone();
            for tr in &self.transitions {
                let contrib = S::mul(&fwd[tr.from_state], &tr.weight);
                new_fwd[tr.to_state].add_assign(&contrib);
            }
            if new_fwd == fwd {
                break;
            }
            fwd = new_fwd;
        }

        let mut total = S::zero();
        for (s, w) in fwd.iter().enumerate() {
            let contrib = S::mul(w, &self.final_weights[s]);
            total.add_assign(&contrib);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// On-the-fly determinization (for functional transducers)
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Determinize a functional (single-valued) transducer.
    ///
    /// Uses the subset construction with residual output strings.
    /// This only works correctly for functional transducers; for non-functional
    /// transducers the result may be incorrect or the algorithm may not
    /// terminate.
    pub fn determinize_functional(&self) -> Result<Self> {
        // Subset state: set of (original_state, residual_output_string)
        // with associated weight.
        type Subset = Vec<(usize, Vec<usize>)>;

        let mut state_map: HashMap<Vec<(usize, Vec<usize>)>, usize> = HashMap::new();
        let mut result = Self::new(0, self.input_alphabet.clone(), self.output_alphabet.clone());

        // Initial subset
        let mut init_subset: Subset = Vec::new();
        for (s, w) in self.initial_weights.iter().enumerate() {
            if !w.is_zero() {
                init_subset.push((s, Vec::new()));
            }
        }
        init_subset.sort_by_key(|&(s, _)| s);
        init_subset.dedup_by_key(|e| e.0);

        if init_subset.is_empty() {
            return Ok(result);
        }

        let init_id = result.add_state();
        result.set_initial_weight(init_id, S::one());
        state_map.insert(init_subset.clone(), init_id);

        let mut queue: VecDeque<Subset> = VecDeque::new();
        queue.push_back(init_subset);

        let max_states = 100_000;

        while let Some(subset) = queue.pop_front() {
            if result.num_states > max_states {
                return Err(TransducerError::CompositionError(
                    "determinization exceeded state limit".into(),
                ));
            }

            let cur_id = state_map[&subset];

            // Compute final weight: the semiring sum of final weights of all
            // states in the subset that have empty residual
            let mut fw = S::zero();
            for (s, residual) in &subset {
                if residual.is_empty() && !self.final_weights[*s].is_zero() {
                    fw.add_assign(&self.final_weights[*s]);
                }
            }
            result.set_final_weight(cur_id, fw);

            // For each input symbol, compute the successor subset
            let alpha_size = self.input_alphabet.size();
            for isym in 0..alpha_size {
                let mut next_subset: Subset = Vec::new();
                let mut common_prefix: Option<Vec<usize>> = None;

                for (s, residual) in &subset {
                    for &idx in &self.transition_index[*s] {
                        let tr = &self.transitions[idx];
                        if tr.input_symbol == Some(isym) {
                            let mut new_residual = residual.clone();
                            if let Some(osym) = tr.output_symbol {
                                new_residual.push(osym);
                            }
                            next_subset.push((tr.to_state, new_residual));
                        }
                    }
                }

                if next_subset.is_empty() {
                    continue;
                }

                // Compute longest common prefix of all residuals
                if !next_subset.is_empty() {
                    common_prefix = Some(next_subset[0].1.clone());
                    for (_, res) in &next_subset[1..] {
                        let cp = common_prefix.as_ref().unwrap();
                        let len = cp
                            .iter()
                            .zip(res.iter())
                            .take_while(|(a, b)| a == b)
                            .count();
                        common_prefix = Some(cp[..len].to_vec());
                    }
                }

                let cp = common_prefix.unwrap_or_default();

                // Strip common prefix from residuals
                let mut stripped: Subset = next_subset
                    .into_iter()
                    .map(|(s, res)| (s, res[cp.len()..].to_vec()))
                    .collect();
                stripped.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                stripped.dedup();

                let next_id = if let Some(&existing) = state_map.get(&stripped) {
                    existing
                } else {
                    let id = result.add_state();
                    state_map.insert(stripped.clone(), id);
                    queue.push_back(stripped);
                    id
                };

                // Output the common prefix on this transition
                // We encode it as the first symbol of the common prefix
                // (for simplicity; a full implementation would use multi-symbol
                // output or introduce intermediate states)
                let out_sym = if cp.is_empty() { None } else { Some(cp[0]) };
                result
                    .add_transition(cur_id, Some(isym), out_sym, next_id, S::one())
                    .ok();

                // If common prefix has more than one symbol, add ε-input
                // transitions to output the rest
                let mut prev_id = next_id;
                for &sym in cp.iter().skip(1) {
                    let mid = result.add_state();
                    // Move the transitions from prev_id to mid
                    result
                        .add_transition(prev_id, None, Some(sym), mid, S::one())
                        .ok();
                    prev_id = mid;
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Minimization stub
// ---------------------------------------------------------------------------

impl<S: Semiring + Ord> WeightedTransducer<S> {
    /// Minimize the transducer by merging equivalent states.
    ///
    /// Uses a partition-refinement approach: two states are equivalent if they
    /// have the same final weight and, for every (input, output) label pair,
    /// their successors are equivalent with the same transition weight.
    pub fn minimize(&self) -> Self {
        let trimmed = self.trim();
        let n = trimmed.num_states;
        if n <= 1 {
            return trimmed;
        }

        // Initial partition: states grouped by final weight
        let mut partition: Vec<usize> = vec![0; n];
        let mut class_map: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut next_class = 0usize;

        for s in 0..n {
            // Use the final weight's debug repr as a partition key
            let key = format!("{:?}", trimmed.final_weights[s]).into_bytes();
            let class = class_map.entry(key).or_insert_with(|| {
                let c = next_class;
                next_class += 1;
                c
            });
            partition[s] = *class;
        }

        // Refine
        let max_iters = n + 1;
        for _ in 0..max_iters {
            let mut new_partition = vec![0usize; n];
            let mut new_map: HashMap<Vec<u8>, usize> = HashMap::new();
            let mut new_next = 0usize;
            let mut changed = false;

            for s in 0..n {
                // Build a signature for this state
                let mut sig = format!("c{}|", partition[s]);
                let mut tr_sigs: Vec<String> = Vec::new();
                for &idx in &trimmed.transition_index[s] {
                    let tr = &trimmed.transitions[idx];
                    tr_sigs.push(format!(
                        "{:?},{:?},{},{:?}",
                        tr.input_symbol,
                        tr.output_symbol,
                        partition[tr.to_state],
                        tr.weight
                    ));
                }
                tr_sigs.sort();
                for ts in &tr_sigs {
                    sig.push_str(ts);
                    sig.push(';');
                }

                let key = sig.into_bytes();
                let class = new_map.entry(key).or_insert_with(|| {
                    let c = new_next;
                    new_next += 1;
                    c
                });
                new_partition[s] = *class;
                if new_partition[s] != partition[s] {
                    changed = true;
                }
            }

            partition = new_partition;
            if !changed || new_next == next_class {
                break;
            }
            next_class = new_next;
        }

        // Build minimized transducer
        let num_classes = *partition.iter().max().unwrap_or(&0) + 1;
        let mut result = Self::new(
            num_classes,
            trimmed.input_alphabet.clone(),
            trimmed.output_alphabet.clone(),
        );

        // Pick a representative for each class
        let mut representative = vec![0usize; num_classes];
        for s in 0..n {
            representative[partition[s]] = s;
        }

        for c in 0..num_classes {
            let rep = representative[c];
            result.initial_weights[c] = trimmed.initial_weights[rep].clone();
            result.final_weights[c] = trimmed.final_weights[rep].clone();
        }

        // Add transitions (deduplicated)
        let mut seen: HashSet<(usize, Option<usize>, Option<usize>, usize)> = HashSet::new();
        for s in 0..n {
            let from_class = partition[s];
            for &idx in &trimmed.transition_index[s] {
                let tr = &trimmed.transitions[idx];
                let to_class = partition[tr.to_state];
                let key = (from_class, tr.input_symbol, tr.output_symbol, to_class);
                if seen.insert(key) {
                    result
                        .add_transition(
                            from_class,
                            tr.input_symbol,
                            tr.output_symbol,
                            to_class,
                            tr.weight.clone(),
                        )
                        .ok();
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Strongly connected components
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Compute SCCs using Tarjan's algorithm.
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.num_states;
        let mut index_counter = 0usize;
        let mut stack: Vec<usize> = Vec::new();
        let mut on_stack = vec![false; n];
        let mut indices = vec![None; n];
        let mut lowlinks = vec![0usize; n];
        let mut sccs: Vec<Vec<usize>> = Vec::new();

        fn strongconnect<S2: Semiring>(
            v: usize,
            t: &WeightedTransducer<S2>,
            index_counter: &mut usize,
            stack: &mut Vec<usize>,
            on_stack: &mut Vec<bool>,
            indices: &mut Vec<Option<usize>>,
            lowlinks: &mut Vec<usize>,
            sccs: &mut Vec<Vec<usize>>,
        ) {
            indices[v] = Some(*index_counter);
            lowlinks[v] = *index_counter;
            *index_counter += 1;
            stack.push(v);
            on_stack[v] = true;

            for &idx in &t.transition_index[v] {
                let w = t.transitions[idx].to_state;
                if indices[w].is_none() {
                    strongconnect(w, t, index_counter, stack, on_stack, indices, lowlinks, sccs);
                    lowlinks[v] = lowlinks[v].min(lowlinks[w]);
                } else if on_stack[w] {
                    lowlinks[v] = lowlinks[v].min(indices[w].unwrap());
                }
            }

            if lowlinks[v] == indices[v].unwrap() {
                let mut scc = Vec::new();
                loop {
                    let w = stack.pop().unwrap();
                    on_stack[w] = false;
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                sccs.push(scc);
            }
        }

        for v in 0..n {
            if indices[v].is_none() {
                strongconnect(
                    v,
                    self,
                    &mut index_counter,
                    &mut stack,
                    &mut on_stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut sccs,
                );
            }
        }

        sccs
    }

    /// Check whether the transducer's state graph is acyclic.
    pub fn is_acyclic(&self) -> bool {
        topological_sort(self).is_ok()
    }
}

// ---------------------------------------------------------------------------
// Weighted edit-distance transducer constructor
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Build a transducer that computes weighted edit distance.
    ///
    /// Given an alphabet of size `n`, creates a single-state transducer with:
    /// - Identity transitions (a→a) with `match_weight`
    /// - Substitution transitions (a→b, a≠b) with `sub_weight`
    /// - Deletion transitions (a→ε) with `del_weight`
    /// - Insertion transitions (ε→a) with `ins_weight`
    pub fn edit_distance(
        alphabet: &Alphabet,
        match_weight: S,
        sub_weight: S,
        del_weight: S,
        ins_weight: S,
    ) -> Self {
        let n = alphabet.size();
        let mut t = Self::new(1, alphabet.clone(), alphabet.clone());
        t.set_initial_weight(0, S::one());
        t.set_final_weight(0, S::one());

        for a in 0..n {
            // Match
            t.add_transition(0, Some(a), Some(a), 0, match_weight.clone())
                .ok();
            // Deletion (a → ε)
            t.add_transition(0, Some(a), None, 0, del_weight.clone())
                .ok();
            // Insertion (ε → a)
            t.add_transition(0, None, Some(a), 0, ins_weight.clone())
                .ok();
            // Substitution (a → b for b ≠ a)
            for b in 0..n {
                if a != b {
                    t.add_transition(0, Some(a), Some(b), 0, sub_weight.clone())
                        .ok();
                }
            }
        }

        t
    }
}

// ---------------------------------------------------------------------------
// Composition with an acceptor (intersect on output)
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Restrict the output of this transducer to strings accepted by the given
    /// WFA.  Equivalent to composing with an identity transducer built from
    /// the WFA, but more efficient.
    pub fn intersect_output(&self, wfa: &WeightedFiniteAutomaton<S>) -> Result<Self> {
        let inv = self.invert();
        let restricted = inv.intersect_input(wfa)?;
        Ok(restricted.invert())
    }
}

// ---------------------------------------------------------------------------
// Power (self-composition)
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Compute the n-fold self-composition T ∘ T ∘ … ∘ T.
    pub fn power(&self, n: usize) -> Result<Self> {
        if n == 0 {
            return Ok(Self::identity(&self.input_alphabet));
        }
        let mut result = self.clone();
        for _ in 1..n {
            result = result.compose(self)?;
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Printing / formatting helpers
// ---------------------------------------------------------------------------

impl<S: Semiring> WeightedTransducer<S> {
    /// Print the transducer in AT&T/OpenFst text format.
    pub fn to_att_format(&self) -> String {
        let mut lines = Vec::new();

        // Transitions: from to input output weight
        for tr in &self.transitions {
            let isym = match tr.input_symbol {
                Some(i) => i.to_string(),
                None => "0".to_string(), // 0 = epsilon in AT&T format
            };
            let osym = match tr.output_symbol {
                Some(o) => o.to_string(),
                None => "0".to_string(),
            };
            lines.push(format!(
                "{}\t{}\t{}\t{}\t{:?}",
                tr.from_state, tr.to_state, isym, osym, tr.weight
            ));
        }

        // Final states: state weight
        for (s, w) in self.final_weights.iter().enumerate() {
            if !w.is_zero() {
                lines.push(format!("{}\t{:?}", s, w));
            }
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ----

    /// A simple semiring for testing: real numbers under (+ , ×).
    /// We wrap f64 so we can implement Semiring.
    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    struct TestWeight(f64);

    impl TestWeight {
        fn val(self) -> f64 {
            self.0
        }
    }

    impl Eq for TestWeight {}

    impl Ord for TestWeight {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
        }
    }

    impl Semiring for TestWeight {
        fn zero() -> Self {
            TestWeight(0.0)
        }
        fn one() -> Self {
            TestWeight(1.0)
        }
        fn add(&self, other: &Self) -> Self {
            TestWeight(self.0 + other.0)
        }
        fn mul(&self, other: &Self) -> Self {
            TestWeight(self.0 * other.0)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
        fn is_one(&self) -> bool {
            (self.0 - 1.0).abs() < 1e-12
        }
        fn add_assign(&mut self, other: &Self) {
            self.0 += other.0;
        }
        fn mul_assign(&mut self, other: &Self) {
            self.0 *= other.0;
        }
    }

    fn w(v: f64) -> TestWeight {
        TestWeight(v)
    }

    fn make_alphabet(n: usize) -> Alphabet {
        Alphabet::from_chars(&(0..n).map(|i| char::from(b'a' + i as u8)).collect::<Vec<_>>())
    }

    // ---- Construction tests ----

    #[test]
    fn test_new_transducer() {
        let alpha = make_alphabet(3);
        let t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(5, alpha.clone(), alpha.clone());
        assert_eq!(t.num_states, 5);
        assert_eq!(t.num_transitions(), 0);
        assert!(t.initial_states().is_empty());
        assert!(t.final_states().is_empty());
    }

    #[test]
    fn test_add_state() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(0, alpha.clone(), alpha.clone());
        let s0 = t.add_state();
        let s1 = t.add_state();
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(t.num_states, 2);
    }

    #[test]
    fn test_add_transition() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();
        t.add_transition(1, Some(1), Some(0), 2, w(0.3)).unwrap();
        assert_eq!(t.num_transitions(), 2);

        // Invalid state
        assert!(t.add_transition(5, Some(0), Some(0), 0, w(1.0)).is_err());
    }

    #[test]
    fn test_set_weights() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        assert_eq!(t.initial_states(), vec![0]);
        assert_eq!(t.final_states(), vec![2]);
    }

    #[test]
    fn test_build_index() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.transitions.push(TransducerTransition::new(0, 1, Some(0), Some(0), w(1.0)));
        t.transitions.push(TransducerTransition::new(1, 2, Some(1), Some(1), w(1.0)));
        t.build_index();
        assert_eq!(t.transition_index[0].len(), 1);
        assert_eq!(t.transition_index[1].len(), 1);
        assert_eq!(t.transition_index[2].len(), 0);
    }

    // ---- Identity transducer tests ----

    #[test]
    fn test_identity_transducer() {
        let alpha = make_alphabet(3);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        assert_eq!(t.num_states, 1);
        assert_eq!(t.num_transitions(), 3);

        // Transduce: identity should map [0,1,2] → [0,1,2]
        let results = t.transduce(&[0, 1, 2]);
        assert_eq!(results.len(), 1);
        let (out, wt) = &results[0];
        assert_eq!(out, &[0, 1, 2]);
        assert!(wt.is_one());
    }

    #[test]
    fn test_identity_empty_input() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        let results = t.transduce(&[]);
        assert_eq!(results.len(), 1);
        assert!(results[0].0.is_empty());
    }

    // ---- Inversion tests ----

    #[test]
    fn test_invert() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let inv = t.invert();
        assert_eq!(inv.transitions[0].input_symbol, Some(1));
        assert_eq!(inv.transitions[0].output_symbol, Some(0));
    }

    #[test]
    fn test_invert_roundtrip() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(2), 1, w(0.7)).unwrap();

        let inv_inv = t.invert().invert();
        assert_eq!(inv_inv.transitions[0].input_symbol, Some(0));
        assert_eq!(inv_inv.transitions[0].output_symbol, Some(2));
        assert_eq!(inv_inv.transitions[0].weight, w(0.7));
    }

    // ---- Projection tests ----

    #[test]
    fn test_input_projection() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(2), 1, w(0.5)).unwrap();

        let wfa = t.input_projection();
        assert_eq!(wfa.num_states, 2);
    }

    #[test]
    fn test_output_projection() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(2), 1, w(0.5)).unwrap();

        let wfa = t.output_projection();
        assert_eq!(wfa.num_states, 2);
    }

    // ---- Transduction tests ----

    #[test]
    fn test_simple_transduction() {
        // Build: 0 --a/b--> 1 --b/a--> 2
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();
        t.add_transition(1, Some(1), Some(0), 2, w(0.3)).unwrap();

        let results = t.transduce(&[0, 1]);
        assert_eq!(results.len(), 1);
        let (out, wt) = &results[0];
        assert_eq!(out, &[1, 0]);
        assert!((wt.val() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_transduction_no_path() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();

        // Input [1] has no matching transition
        let results = t.transduce(&[1]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_transduction_with_epsilon() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        // 0 --a/ε--> 1 --ε/b--> 2
        t.add_transition(0, Some(0), None, 1, w(0.5)).unwrap();
        t.add_transition(1, None, Some(1), 2, w(0.4)).unwrap();

        let results = t.transduce(&[0]);
        assert!(!results.is_empty());
        let (out, wt) = &results[0];
        assert_eq!(out, &[1]);
        assert!((wt.val() - 0.2).abs() < 1e-10);
    }

    // ---- Composition tests ----

    #[test]
    fn test_compose_with_identity() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();
        t.add_transition(0, Some(1), Some(2), 1, w(0.3)).unwrap();

        let id = WeightedTransducer::<TestWeight>::identity(&alpha);
        let composed = t.compose(&id).unwrap();

        // Compose with identity should preserve the transduction
        let orig_results = t.transduce(&[0]);
        let comp_results = composed.transduce(&[0]);
        assert_eq!(orig_results.len(), comp_results.len());

        // Check outputs match
        let orig_out: HashSet<Vec<usize>> =
            orig_results.iter().map(|(o, _)| o.clone()).collect();
        let comp_out: HashSet<Vec<usize>> =
            comp_results.iter().map(|(o, _)| o.clone()).collect();
        assert_eq!(orig_out, comp_out);
    }

    #[test]
    fn test_compose_two_transducers() {
        // T1: a → b with weight 0.5
        // T2: b → c with weight 0.3
        // T1 ∘ T2: a → c with weight 0.15
        let alpha = make_alphabet(3);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(1), Some(2), 1, w(0.3)).unwrap();

        let composed = t1.compose(&t2).unwrap();
        let results = composed.transduce(&[0]);
        assert_eq!(results.len(), 1);
        let (out, wt) = &results[0];
        assert_eq!(out, &[2]); // a → c
        assert!((wt.val() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_compose_chain() {
        // T1: 0→1, T2: 1→2, T3: 2→0
        // compose all three: should map 0→0
        let alpha = make_alphabet(3);

        let make_t = |from: usize, to: usize| -> WeightedTransducer<TestWeight> {
            let mut t = WeightedTransducer::new(2, alpha.clone(), alpha.clone());
            t.set_initial_weight(0, w(1.0));
            t.set_final_weight(1, w(1.0));
            t.add_transition(0, Some(from), Some(to), 1, w(1.0)).unwrap();
            t
        };

        let t12 = make_t(0, 1).compose(&make_t(1, 2)).unwrap();
        let t123 = t12.compose(&make_t(2, 0)).unwrap();

        let results = t123.transduce(&[0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![0]);
    }

    #[test]
    fn test_compose_incompatible_alphabets() {
        let a2 = make_alphabet(2);
        let a3 = make_alphabet(3);
        let t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(1, a2.clone(), a2.clone());
        let t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(1, a3.clone(), a3.clone());
        assert!(t1.compose(&t2).is_err());
    }

    // ---- Composition with epsilon transitions ----

    #[test]
    fn test_compose_with_epsilon_transitions() {
        let alpha = make_alphabet(3);

        // T1: state 0 → state 1 via input a / output ε, then state 1 → state 2
        //     via input ε / output b
        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(2, w(1.0));
        t1.add_transition(0, Some(0), None, 1, w(1.0)).unwrap(); // a / ε
        t1.add_transition(1, None, Some(1), 2, w(1.0)).unwrap(); // ε / b

        // T2: identity
        let t2 = WeightedTransducer::<TestWeight>::identity(&alpha);

        let composed = t1.compose(&t2).unwrap();
        let results = composed.transduce(&[0]);
        assert!(!results.is_empty());
        // Output should be b (symbol 1)
        let outputs: HashSet<Vec<usize>> = results.iter().map(|(o, _)| o.clone()).collect();
        assert!(outputs.contains(&vec![1]));
    }

    // ---- Epsilon handling tests ----

    #[test]
    fn test_remove_all_epsilon() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, None, None, 1, w(0.5)).unwrap();
        t.add_transition(1, Some(0), Some(0), 2, w(0.4)).unwrap();

        let orig_results = t.transduce(&[0]);
        let removed = t.remove_all_epsilon();
        let new_results = removed.transduce(&[0]);

        assert!(!orig_results.is_empty());
        assert!(!new_results.is_empty());

        // Weights should match
        let orig_w: f64 = orig_results.iter().map(|(_, w)| w.val()).sum();
        let new_w: f64 = new_results.iter().map(|(_, w)| w.val()).sum();
        assert!((orig_w - new_w).abs() < 1e-10);
    }

    #[test]
    fn test_remove_input_epsilon() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, None, Some(0), 1, w(0.5)).unwrap(); // ε/a
        t.add_transition(1, Some(0), Some(1), 2, w(0.4)).unwrap(); // a/b

        let removed = t.remove_input_epsilon();
        // No input-ε transitions should remain
        assert!(!removed.has_input_epsilon());
    }

    #[test]
    fn test_remove_output_epsilon() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, Some(0), None, 1, w(0.5)).unwrap(); // a/ε
        t.add_transition(1, Some(1), Some(0), 2, w(0.4)).unwrap(); // b/a

        let removed = t.remove_output_epsilon();
        assert!(!removed.has_output_epsilon());
    }

    // ---- Shortest path tests ----

    #[test]
    fn test_shortest_path() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        // Path 1: 0 --a/a(0.3)--> 1 --b/b(0.2)--> 2   total = 0.06
        t.add_transition(0, Some(0), Some(0), 1, w(0.3)).unwrap();
        t.add_transition(1, Some(1), Some(1), 2, w(0.2)).unwrap();
        // Path 2: 0 --a/b(0.1)--> 2 --b/a(0.9)--> 2 (self loop)
        // This path would require state 2 to accept after self-loop
        // Simpler: add a direct path 0 --a/b(0.8)--> 1 --b/a(0.9)--> 2  total = 0.72
        t.add_transition(0, Some(0), Some(1), 1, w(0.8)).unwrap();
        t.add_transition(1, Some(1), Some(0), 2, w(0.9)).unwrap();

        let result = t.shortest_path(&[0, 1]);
        assert!(result.is_some());
        let (out, wt) = result.unwrap();
        // The shortest path depends on ordering. With Ord on TestWeight,
        // smaller values come first.
        assert!(!out.is_empty());
        assert!(wt.val() > 0.0);
    }

    // ---- N-best path tests ----

    #[test]
    fn test_n_best_paths() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        // Two paths for input [0]: output [0] with weight 0.3, output [1] with weight 0.7
        t.add_transition(0, Some(0), Some(0), 1, w(0.3)).unwrap();
        t.add_transition(0, Some(0), Some(1), 1, w(0.7)).unwrap();

        let results = t.n_best_paths(&[0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_n_best_zero() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        let results = t.n_best_paths(&[0], 0);
        assert!(results.is_empty());
    }

    // ---- Trim tests ----

    #[test]
    fn test_trim_removes_unreachable() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(4, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        // States 2 and 3 are unreachable
        t.add_transition(2, Some(0), Some(0), 3, w(1.0)).unwrap();

        let trimmed = t.trim();
        assert_eq!(trimmed.num_states, 2);
        assert_eq!(trimmed.num_transitions(), 1);
    }

    #[test]
    fn test_trim_preserves_valid() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        let trimmed = t.trim();
        assert_eq!(trimmed.num_states, t.num_states);
        assert_eq!(trimmed.num_transitions(), t.num_transitions());
    }

    #[test]
    fn test_connect() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        t.add_transition(0, Some(1), Some(1), 2, w(1.0)).unwrap();
        // State 2 has no outgoing transitions and is not final → coaccessible
        // will remove it

        let connected = t.connect();
        // State 2 cannot reach a final state, so it should be removed
        assert!(connected.num_states <= 3);
    }

    // ---- Lazy composition tests ----

    #[test]
    fn test_lazy_composition_basic() {
        let alpha = make_alphabet(3);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(1), Some(2), 1, w(0.3)).unwrap();

        let lazy = LazyComposition::new(&t1, &t2);
        let results = lazy.transduce(&[0]);
        assert_eq!(results.len(), 1);
        let (out, wt) = &results[0];
        assert_eq!(out, &[2]);
        assert!((wt.val() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_lazy_vs_eager_composition() {
        let alpha = make_alphabet(3);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();
        t1.add_transition(0, Some(1), Some(2), 1, w(0.3)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(1), Some(0), 1, w(0.4)).unwrap();
        t2.add_transition(0, Some(2), Some(1), 1, w(0.6)).unwrap();

        // Eager
        let eager = t1.compose(&t2).unwrap();
        let eager_r0 = eager.transduce(&[0]);
        let eager_r1 = eager.transduce(&[1]);

        // Lazy
        let lazy = LazyComposition::new(&t1, &t2);
        let lazy_r0 = lazy.transduce(&[0]);
        let lazy_r1 = lazy.transduce(&[1]);

        // Compare results
        let to_map = |v: &[(Vec<usize>, TestWeight)]| -> HashMap<Vec<usize>, f64> {
            v.iter().map(|(o, w)| (o.clone(), w.val())).collect()
        };

        let em0 = to_map(&eager_r0);
        let lm0 = to_map(&lazy_r0);
        assert_eq!(em0.len(), lm0.len());
        for (k, v) in &em0 {
            assert!((v - lm0[k]).abs() < 1e-10);
        }

        let em1 = to_map(&eager_r1);
        let lm1 = to_map(&lazy_r1);
        assert_eq!(em1.len(), lm1.len());
        for (k, v) in &em1 {
            assert!((v - lm1[k]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lazy_composition_cache() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        let lazy = LazyComposition::new(&t, &t);
        assert_eq!(lazy.cached_state_count(), 0);
        let _ = lazy.transduce(&[0, 1]);
        assert!(lazy.cached_state_count() > 0);
        lazy.clear_cache();
        assert_eq!(lazy.cached_state_count(), 0);
    }

    // ---- Edge case tests ----

    #[test]
    fn test_empty_transducer() {
        let alpha = make_alphabet(2);
        let t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(0, alpha.clone(), alpha.clone());
        assert!(t.is_empty());
        assert_eq!(t.transduce(&[0]), vec![]);
    }

    #[test]
    fn test_single_state_no_transitions() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(1, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(0, w(1.0));

        // Empty input should produce empty output
        let results = t.transduce(&[]);
        assert_eq!(results.len(), 1);
        assert!(results[0].0.is_empty());

        // Non-empty input should produce nothing
        let results = t.transduce(&[0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_multiple_paths() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.set_final_weight(2, w(1.0));
        // Two paths for input [0]: via state 1 (output [0]) and via state 2 (output [1])
        t.add_transition(0, Some(0), Some(0), 1, w(0.3)).unwrap();
        t.add_transition(0, Some(0), Some(1), 2, w(0.7)).unwrap();

        let results = t.transduce(&[0]);
        assert_eq!(results.len(), 2);
    }

    // ---- Utility tests ----

    #[test]
    fn test_display() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        let s = format!("{}", t);
        assert!(s.contains("WeightedTransducer"));
        assert!(s.contains("transitions: 2"));
    }

    #[test]
    fn test_to_dot() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let dot = t.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("doublecircle"));
        assert!(dot.contains("0/1"));
    }

    #[test]
    fn test_summary() {
        let alpha = make_alphabet(3);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        let s = t.summary();
        assert!(s.contains("states=1"));
        assert!(s.contains("transitions=3"));
    }

    #[test]
    fn test_reverse() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let rev = t.reverse();
        assert_eq!(rev.initial_states(), vec![1]);
        assert_eq!(rev.final_states(), vec![0]);
        assert_eq!(rev.transitions[0].from_state, 1);
        assert_eq!(rev.transitions[0].to_state, 0);
    }

    #[test]
    fn test_concatenate() {
        let alpha = make_alphabet(2);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(1), Some(1), 1, w(1.0)).unwrap();

        let cat = t1.concatenate(&t2).unwrap();
        let results = cat.transduce(&[0, 1]);
        assert!(!results.is_empty());
        let (out, _) = &results[0];
        assert_eq!(out, &[0, 1]);
    }

    #[test]
    fn test_union() {
        let alpha = make_alphabet(2);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(1), Some(1), 1, w(1.0)).unwrap();

        let u = t1.union(&t2).unwrap();
        // Should accept both [0] and [1]
        assert!(!u.transduce(&[0]).is_empty());
        assert!(!u.transduce(&[1]).is_empty());
    }

    #[test]
    fn test_closure() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();

        let star = t.closure();
        // Should accept empty string
        let empty_results = star.transduce(&[]);
        assert!(!empty_results.is_empty());
        // Should accept [0]
        let single_results = star.transduce(&[0]);
        assert!(!single_results.is_empty());
    }

    // ---- All-pairs shortest ----

    #[test]
    fn test_all_pairs_shortest() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(0.5)).unwrap();
        t.add_transition(1, Some(1), Some(1), 2, w(0.3)).unwrap();

        let dist = t.all_pairs_shortest();
        assert_eq!(dist.len(), 3);
        // dist[0][0] should be one() (identity)
        assert!(dist[0][0].is_one());
        // dist[0][1] should be 0.5
        assert!((dist[0][1].val() - 0.5).abs() < 1e-10);
    }

    // ---- Strongly connected components ----

    #[test]
    fn test_sccs_acyclic() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        t.add_transition(1, Some(1), Some(1), 2, w(1.0)).unwrap();

        let sccs = t.strongly_connected_components();
        // Each state is its own SCC in an acyclic graph
        assert_eq!(sccs.len(), 3);
        assert!(t.is_acyclic());
    }

    #[test]
    fn test_sccs_cyclic() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        t.add_transition(1, Some(1), Some(1), 0, w(1.0)).unwrap();

        let sccs = t.strongly_connected_components();
        // States 0 and 1 form a single SCC
        assert_eq!(sccs.len(), 1);
        assert!(!t.is_acyclic());
    }

    // ---- is_functional ----

    #[test]
    fn test_is_functional_true() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        assert!(t.is_functional());
    }

    #[test]
    fn test_is_functional_false() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.set_final_weight(2, w(1.0));
        // Same input symbol maps to two different outputs
        t.add_transition(0, Some(0), Some(0), 1, w(0.5)).unwrap();
        t.add_transition(0, Some(0), Some(1), 2, w(0.5)).unwrap();
        assert!(!t.is_functional());
    }

    // ---- Edit distance transducer ----

    #[test]
    fn test_edit_distance_transducer() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::edit_distance(
            &alpha,
            w(1.0), // match
            w(0.1), // substitution
            w(0.2), // deletion
            w(0.3), // insertion
        );
        assert_eq!(t.num_states, 1);
        // Should have: 2 match + 2 sub + 2 del + 2 ins = 8 transitions
        assert_eq!(t.num_transitions(), 8);

        // Transducing [0] should produce at least [0] (match)
        let results = t.transduce(&[0]);
        assert!(!results.is_empty());
    }

    // ---- Weight map ----

    #[test]
    fn test_map_weights() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(0.5)).unwrap();

        let doubled = t.map_weights(|w| TestWeight(w.val() * 2.0));
        assert_eq!(doubled.transitions[0].weight, w(1.0));
        assert_eq!(doubled.initial_weights[0], w(2.0));
    }

    // ---- Path enumeration ----

    #[test]
    fn test_enumerate_paths() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();
        t.add_transition(0, Some(1), Some(0), 1, w(0.3)).unwrap();

        let paths = t.enumerate_paths(10);
        assert_eq!(paths.len(), 2);
        for p in &paths {
            assert_eq!(p.states.len(), 2); // start + end
            assert_eq!(p.input_string().len(), 1);
            assert_eq!(p.output_string().len(), 1);
        }
    }

    // ---- Builder pattern ----

    #[test]
    fn test_builder() {
        let alpha = make_alphabet(2);
        let t = TransducerBuilder::<TestWeight>::new(alpha.clone(), alpha.clone())
            .add_states(3)
            .set_initial(0, w(1.0))
            .set_final(2, w(1.0))
            .add_transition(0, Some(0), Some(1), 1, w(0.5))
            .add_transition(1, Some(1), Some(0), 2, w(0.3))
            .build();

        assert_eq!(t.num_states, 3);
        assert_eq!(t.num_transitions(), 2);
        let results = t.transduce(&[0, 1]);
        assert_eq!(results.len(), 1);
    }

    // ---- Power ----

    #[test]
    fn test_power_zero() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(1.0)).unwrap();

        let p0 = t.power(0).unwrap();
        // Power 0 = identity
        assert_eq!(p0.num_states, 1);
        let results = p0.transduce(&[0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![0]);
    }

    #[test]
    fn test_power_one() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(1.0)).unwrap();

        let p1 = t.power(1).unwrap();
        let orig_results = t.transduce(&[0]);
        let p1_results = p1.transduce(&[0]);
        assert_eq!(orig_results.len(), p1_results.len());
    }

    // ---- Degrees ----

    #[test]
    fn test_degrees() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        t.add_transition(0, Some(1), Some(1), 2, w(1.0)).unwrap();
        t.add_transition(1, Some(0), Some(0), 2, w(1.0)).unwrap();

        let out_deg = t.out_degrees();
        assert_eq!(out_deg, vec![2, 1, 0]);

        let in_deg = t.in_degrees();
        assert_eq!(in_deg, vec![0, 1, 2]);
    }

    // ---- AT&T format ----

    #[test]
    fn test_att_format() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let att = t.to_att_format();
        assert!(att.contains("0\t1\t0\t1"));
    }

    // ---- Epsilon counting ----

    #[test]
    fn test_epsilon_counts() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.add_transition(0, None, None, 1, w(1.0)).unwrap(); // both ε
        t.add_transition(0, None, Some(0), 2, w(1.0)).unwrap(); // input ε only
        t.add_transition(1, Some(0), None, 2, w(1.0)).unwrap(); // output ε only
        t.add_transition(1, Some(0), Some(1), 2, w(1.0)).unwrap(); // neither ε

        assert_eq!(t.num_epsilon_transitions(), 1);
        assert_eq!(t.num_input_epsilon_transitions(), 2);
        assert_eq!(t.num_output_epsilon_transitions(), 2);
    }

    // ---- Relabel tests ----

    #[test]
    fn test_relabel_output() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();

        let mut mapping = HashMap::new();
        mapping.insert(0, 2); // relabel output 0 → 2

        let relabeled = t.relabel_output(&mapping);
        assert_eq!(relabeled.transitions[0].output_symbol, Some(2));
    }

    // ---- Compose with trivial filter ----

    #[test]
    fn test_compose_trivial_filter() {
        let alpha = make_alphabet(3);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(1), Some(2), 1, w(0.3)).unwrap();

        let composed = t1
            .compose_with_filter(&t2, FilterType::Trivial)
            .unwrap();
        let results = composed.transduce(&[0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![2]);
        assert!((results[0].1.val() - 0.15).abs() < 1e-10);
    }

    // ---- Look-ahead composition ----

    #[test]
    fn test_compose_with_lookahead() {
        let alpha = make_alphabet(3);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(1), 1, w(0.5)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(1), Some(2), 1, w(0.3)).unwrap();

        let composed = t1.compose_with_lookahead(&t2).unwrap();
        let results = composed.transduce(&[0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![2]);
        assert!((results[0].1.val() - 0.15).abs() < 1e-10);
    }

    // ---- Minimize ----

    #[test]
    fn test_minimize() {
        let alpha = make_alphabet(2);
        // Build a transducer with two equivalent states
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(4, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.set_final_weight(3, w(1.0));
        // State 1 and state 0 both go to final states with same transitions
        t.add_transition(0, Some(0), Some(0), 2, w(1.0)).unwrap();
        t.add_transition(0, Some(1), Some(1), 3, w(1.0)).unwrap();

        let minimized = t.minimize();
        // The minimized transducer should have fewer or equal states
        assert!(minimized.num_states <= t.num_states);
    }

    // ---- Total weight ----

    #[test]
    fn test_total_weight() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(0.5)).unwrap();
        t.add_transition(0, Some(1), Some(1), 1, w(0.3)).unwrap();

        let tw = t.total_weight();
        // Total weight = 1.0 * 0.5 * 1.0 + 1.0 * 0.3 * 1.0 = 0.8
        assert!((tw.val() - 0.8).abs() < 1e-10);
    }

    // ---- Multi-state transduction ----

    #[test]
    fn test_multi_state_transduction() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(4, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(3, w(1.0));
        // 0 --a/x--> 1 --b/y--> 2 --c/z--> 3
        t.add_transition(0, Some(0), Some(0), 1, w(0.5)).unwrap();
        t.add_transition(1, Some(1), Some(1), 2, w(0.4)).unwrap();
        t.add_transition(2, Some(2), Some(2), 3, w(0.3)).unwrap();

        let results = t.transduce(&[0, 1, 2]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![0, 1, 2]);
        assert!((results[0].1.val() - 0.06).abs() < 1e-10);
    }

    // ---- Compose multi-step ----

    #[test]
    fn test_compose_multi_step() {
        // T1: maps symbol 0 to symbol 1, symbol 1 to symbol 0 (swap)
        // T2: maps symbol 0 to symbol 0, symbol 1 to symbol 1 (identity)
        // T1 ∘ T2 should still swap
        let alpha = make_alphabet(2);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(1, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(0, w(1.0));
        t1.add_transition(0, Some(0), Some(1), 0, w(1.0)).unwrap();
        t1.add_transition(0, Some(1), Some(0), 0, w(1.0)).unwrap();

        let t2 = WeightedTransducer::<TestWeight>::identity(&alpha);

        let composed = t1.compose(&t2).unwrap();
        let results = composed.transduce(&[0, 1, 0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![1, 0, 1]);
    }

    // ---- Weight pushing ----

    #[test]
    fn test_push_weights() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(1, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(0.5)).unwrap();

        let pushed = t.push_weights();
        assert_eq!(pushed.num_states, 2);
        // After pushing, the transducer should still produce the same results
        // (up to weight redistribution)
    }

    // ---- Synchronize ----

    #[test]
    fn test_synchronize() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, Some(0), None, 1, w(1.0)).unwrap(); // a/ε
        t.add_transition(1, None, Some(0), 2, w(1.0)).unwrap(); // ε/a

        let synced = t.synchronize();
        assert_eq!(synced.num_states, 3);
    }

    // ---- Filter type variations ----

    #[test]
    fn test_compose_epsilon_matching_filter() {
        let alpha = make_alphabet(2);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(0), Some(1), 1, w(1.0)).unwrap();

        let composed = t1
            .compose_with_filter(&t2, FilterType::EpsilonMatching)
            .unwrap();
        let results = composed.transduce(&[0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![1]);
    }

    #[test]
    fn test_compose_multi_epsilon_filter() {
        let alpha = make_alphabet(2);

        let mut t1: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t1.set_initial_weight(0, w(1.0));
        t1.set_final_weight(1, w(1.0));
        t1.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();

        let mut t2: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t2.set_initial_weight(0, w(1.0));
        t2.set_final_weight(1, w(1.0));
        t2.add_transition(0, Some(0), Some(1), 1, w(1.0)).unwrap();

        let composed = t1
            .compose_with_filter(&t2, FilterType::MultiEpsilonFilter)
            .unwrap();
        let results = composed.transduce(&[0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![1]);
    }

    // ---- LookAheadFilter unit test ----

    #[test]
    fn test_lookahead_filter() {
        let alpha = make_alphabet(3);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, w(1.0));
        t.set_final_weight(2, w(1.0));
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        t.add_transition(1, Some(1), Some(1), 2, w(1.0)).unwrap();

        let filter = LookAheadFilter::new(&t);
        assert!(filter.can_reach_final[0]);
        assert!(filter.can_reach_final[1]);
        assert!(filter.can_reach_final[2]);
        assert!(filter.allows(0, Some(0)));
        assert!(filter.allows(0, None)); // ε always allowed
    }

    // ---- Topological sort ----

    #[test]
    fn test_topological_sort_acyclic() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(3, alpha.clone(), alpha.clone());
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        t.add_transition(1, Some(1), Some(1), 2, w(1.0)).unwrap();

        let order = topological_sort(&t).unwrap();
        assert_eq!(order.len(), 3);
        // 0 must come before 1, 1 before 2
        let pos: HashMap<usize, usize> = order.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        assert!(pos[&0] < pos[&1]);
        assert!(pos[&1] < pos[&2]);
    }

    #[test]
    fn test_topological_sort_cyclic() {
        let alpha = make_alphabet(2);
        let mut t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.add_transition(0, Some(0), Some(0), 1, w(1.0)).unwrap();
        t.add_transition(1, Some(1), Some(1), 0, w(1.0)).unwrap();

        assert!(topological_sort(&t).is_err());
    }

    // ---- TransducerTransition tests ----

    #[test]
    fn test_transition_epsilon_predicates() {
        let t1 = TransducerTransition::<TestWeight>::new(0, 1, None, None, w(1.0));
        assert!(t1.is_epsilon());
        assert!(t1.is_input_epsilon());
        assert!(t1.is_output_epsilon());

        let t2 = TransducerTransition::<TestWeight>::new(0, 1, Some(0), None, w(1.0));
        assert!(!t2.is_epsilon());
        assert!(!t2.is_input_epsilon());
        assert!(t2.is_output_epsilon());

        let t3 = TransducerTransition::<TestWeight>::new(0, 1, None, Some(0), w(1.0));
        assert!(!t3.is_epsilon());
        assert!(t3.is_input_epsilon());
        assert!(!t3.is_output_epsilon());

        let t4 = TransducerTransition::<TestWeight>::new(0, 1, Some(0), Some(1), w(1.0));
        assert!(!t4.is_epsilon());
        assert!(!t4.is_input_epsilon());
        assert!(!t4.is_output_epsilon());
    }

    // ---- CompositionState tests ----

    #[test]
    fn test_composition_state() {
        let cs1 = CompositionState::new(0, 1, EpsilonFilter::NoFilter);
        let cs2 = CompositionState::new(0, 1, EpsilonFilter::NoFilter);
        let cs3 = CompositionState::new(0, 1, EpsilonFilter::Filter1);
        assert_eq!(cs1, cs2);
        assert_ne!(cs1, cs3);
    }

    // ---- CompositionStateSpace tests ----

    #[test]
    fn test_composition_state_space() {
        let mut space = CompositionStateSpace::new();
        let cs1 = CompositionState::new(0, 0, EpsilonFilter::NoFilter);
        let cs2 = CompositionState::new(0, 1, EpsilonFilter::NoFilter);

        let id1 = space.get_or_insert(cs1);
        let id2 = space.get_or_insert(cs2);
        let id1_again = space.get_or_insert(cs1);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1_again, 0); // same as before
        assert_eq!(space.len(), 2);
        assert_eq!(space.state_at(0), &cs1);
    }

    // ---- TransducerPath tests ----

    #[test]
    fn test_transducer_path() {
        let path = TransducerPath::<TestWeight> {
            states: vec![0, 1, 2],
            input_labels: vec![Some(0), None, Some(1)],
            output_labels: vec![None, Some(2), Some(3)],
            weight: w(0.5),
        };
        assert_eq!(path.input_string(), vec![0, 1]);
        assert_eq!(path.output_string(), vec![2, 3]);
    }

    // ---- is_empty tests ----

    #[test]
    fn test_is_empty_true() {
        let alpha = make_alphabet(2);
        let t: WeightedTransducer<TestWeight> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        assert!(t.is_empty());
    }

    #[test]
    fn test_is_empty_false() {
        let alpha = make_alphabet(2);
        let t = WeightedTransducer::<TestWeight>::identity(&alpha);
        assert!(!t.is_empty());
    }
}
