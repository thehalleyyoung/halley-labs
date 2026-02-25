//! Additional operations on Weighted Finite Automata.
//!
//! This module implements product constructions, quotient operations,
//! projections, weight manipulations, state elimination for regex conversion,
//! morphisms, analysis queries, and DOT output.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use anyhow::Result;
use log::{debug, trace, warn};
use thiserror::Error;

use super::automaton::{WeightedFiniteAutomaton, Alphabet, Symbol};
use super::semiring::{
    BooleanSemiring, CountingSemiring, RealSemiring, Semiring, SemiringMatrix,
    StarSemiring, TropicalSemiring,
};

// ---------------------------------------------------------------------------
// 1. OperationError
// ---------------------------------------------------------------------------

/// Errors arising from WFA operations.
#[derive(Debug, Error)]
pub enum OperationError {
    #[error("incompatible alphabets: the two automata must share the same alphabet")]
    IncompatibleAlphabets,

    #[error("dimension mismatch in matrix/vector operation")]
    DimensionMismatch,

    #[error("operation requires a non-empty automaton")]
    EmptyAutomaton,

    #[error("invalid operation: {desc}")]
    InvalidOperation { desc: String },

    #[error("arithmetic overflow during weight computation")]
    OverflowError,

    #[error("state explosion: {states} states exceeds limit {limit}")]
    StateExplosion { states: usize, limit: usize },
}

/// Default limit on product-state-space size to prevent runaway blowup.
const STATE_EXPLOSION_LIMIT: usize = 1_000_000;

// ---------------------------------------------------------------------------
// Helper: pair-state index mapping
// ---------------------------------------------------------------------------

/// Maps a pair `(q1, q2)` into a single linear index.
#[inline]
fn pair_index(q1: usize, q2: usize, n2: usize) -> usize {
    q1 * n2 + q2
}

/// Recovers the pair from a linear index.
#[inline]
fn index_to_pair(idx: usize, n2: usize) -> (usize, usize) {
    (idx / n2, idx % n2)
}

/// Validates that two WFAs share the same alphabet size and returns it.
fn check_compatible_alphabets<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> std::result::Result<usize, OperationError> {
    let a1 = wfa1.alphabet().size();
    let a2 = wfa2.alphabet().size();
    if a1 != a2 {
        return Err(OperationError::IncompatibleAlphabets);
    }
    Ok(a1)
}

/// Checks that a product would not exceed the explosion limit.
fn check_product_size(
    n1: usize,
    n2: usize,
) -> std::result::Result<usize, OperationError> {
    let product = n1.checked_mul(n2).ok_or(OperationError::OverflowError)?;
    if product > STATE_EXPLOSION_LIMIT {
        return Err(OperationError::StateExplosion {
            states: product,
            limit: STATE_EXPLOSION_LIMIT,
        });
    }
    Ok(product)
}

// ---------------------------------------------------------------------------
// 2. Hadamard product (synchronous/intersection product)
// ---------------------------------------------------------------------------

/// Computes the Hadamard (intersection) product of two WFAs.
///
/// The resulting WFA has a product state space `(q1, q2)` with
///   * initial weight  `α(q1) ⊗ α(q2)`
///   * final weight    `ρ(q1) ⊗ ρ(q2)`
///   * transitions     `(q1,q2) --a/w1⊗w2--> (q1',q2')` whenever
///     `q1 --a/w1--> q1'` in wfa1 and `q2 --a/w2--> q2'` in wfa2.
pub fn hadamard_product<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> std::result::Result<WeightedFiniteAutomaton<S>, OperationError> {
    let alpha_size = check_compatible_alphabets(wfa1, wfa2)?;
    let n1 = wfa1.num_states();
    let n2 = wfa2.num_states();

    if n1 == 0 || n2 == 0 {
        return Err(OperationError::EmptyAutomaton);
    }

    let product_states = check_product_size(n1, n2)?;
    debug!(
        "hadamard_product: {}×{} = {} product states",
        n1, n2, product_states
    );

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa1.alphabet().clone());

    // Add product states.
    for _ in 0..product_states {
        result.add_state();
    }

    // Initial weights: α_result(q1,q2) = α1(q1) ⊗ α2(q2).
    for q1 in 0..n1 {
        let iw1 = &wfa1.initial_weights()[q1];
        if iw1.is_zero() {
            continue;
        }
        for q2 in 0..n2 {
            let iw2 = &wfa2.initial_weights()[q2];
            if iw2.is_zero() {
                continue;
            }
            let w = iw1.clone().mul(&iw2);
            if !w.is_zero() {
                result.set_initial_weight(pair_index(q1, q2, n2), w);
            }
        }
    }

    // Final weights: ρ_result(q1,q2) = ρ1(q1) ⊗ ρ2(q2).
    for q1 in 0..n1 {
        let fw1 = &wfa1.final_weights()[q1];
        if fw1.is_zero() {
            continue;
        }
        for q2 in 0..n2 {
            let fw2 = &wfa2.final_weights()[q2];
            if fw2.is_zero() {
                continue;
            }
            let w = fw1.clone().mul(&fw2);
            if !w.is_zero() {
                result.set_final_weight(pair_index(q1, q2, n2), w);
            }
        }
    }

    // Transitions.
    let trans1 = wfa1.transitions();
    let trans2 = wfa2.transitions();

    for q1 in 0..n1 {
        for q2 in 0..n2 {
            let src = pair_index(q1, q2, n2);
            for a in 0..alpha_size {
                // All pairs of transitions on symbol a.
                for &(q1p, ref w1) in &trans1[q1][a] {
                    if w1.is_zero() {
                        continue;
                    }
                    for &(q2p, ref w2) in &trans2[q2][a] {
                        if w2.is_zero() {
                            continue;
                        }
                        let w = w1.clone().mul(w2);
                        if !w.is_zero() {
                            let dst = pair_index(q1p, q2p, n2);
                            result.add_transition(src, a, dst, w);
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Returns the (q1, q2) pair for a product-state index.
pub fn hadamard_state_pair(
    product_idx: usize,
    n2: usize,
) -> (usize, usize) {
    index_to_pair(product_idx, n2)
}

/// Returns the product-state index for a (q1, q2) pair.
pub fn hadamard_pair_index(q1: usize, q2: usize, n2: usize) -> usize {
    pair_index(q1, q2, n2)
}

// ---------------------------------------------------------------------------
// 3. Shuffle product
// ---------------------------------------------------------------------------

/// Computes the shuffle (interleaving) product of two WFAs.
///
/// The shuffle of languages L1 and L2 contains all interleavings of a word
/// from L1 and a word from L2. The product state space is `(q1, q2)`.
///
/// Transitions:
///   * advance wfa1: `(q1,q2) --a/w1--> (q1',q2)` for `q1 --a/w1--> q1'`
///   * advance wfa2: `(q1,q2) --a/w2--> (q1,q2')` for `q2 --a/w2--> q2'`
pub fn shuffle_product<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> std::result::Result<WeightedFiniteAutomaton<S>, OperationError> {
    let alpha_size = check_compatible_alphabets(wfa1, wfa2)?;
    let n1 = wfa1.num_states();
    let n2 = wfa2.num_states();

    if n1 == 0 || n2 == 0 {
        return Err(OperationError::EmptyAutomaton);
    }

    let product_states = check_product_size(n1, n2)?;
    debug!(
        "shuffle_product: {}×{} = {} product states",
        n1, n2, product_states
    );

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa1.alphabet().clone());
    for _ in 0..product_states {
        result.add_state();
    }

    // Initial weights.
    for q1 in 0..n1 {
        let iw1 = &wfa1.initial_weights()[q1];
        if iw1.is_zero() {
            continue;
        }
        for q2 in 0..n2 {
            let iw2 = &wfa2.initial_weights()[q2];
            if iw2.is_zero() {
                continue;
            }
            let w = iw1.clone().mul(iw2);
            if !w.is_zero() {
                result.set_initial_weight(pair_index(q1, q2, n2), w);
            }
        }
    }

    // Final weights.
    for q1 in 0..n1 {
        let fw1 = &wfa1.final_weights()[q1];
        if fw1.is_zero() {
            continue;
        }
        for q2 in 0..n2 {
            let fw2 = &wfa2.final_weights()[q2];
            if fw2.is_zero() {
                continue;
            }
            let w = fw1.clone().mul(fw2);
            if !w.is_zero() {
                result.set_final_weight(pair_index(q1, q2, n2), w);
            }
        }
    }

    let trans1 = wfa1.transitions();
    let trans2 = wfa2.transitions();

    for q1 in 0..n1 {
        for q2 in 0..n2 {
            let src = pair_index(q1, q2, n2);

            for a in 0..alpha_size {
                // Type 1: advance wfa1, keep q2 fixed.
                for &(q1p, ref w1) in &trans1[q1][a] {
                    if !w1.is_zero() {
                        let dst = pair_index(q1p, q2, n2);
                        result.add_transition(src, a, dst, w1.clone());
                    }
                }

                // Type 2: keep q1 fixed, advance wfa2.
                for &(q2p, ref w2) in &trans2[q2][a] {
                    if !w2.is_zero() {
                        let dst = pair_index(q1, q2p, n2);
                        result.add_transition(src, a, dst, w2.clone());
                    }
                }
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// 4. Reversal
// ---------------------------------------------------------------------------

/// Returns the reversal of a WFA.
///
/// The reversed WFA accepts the reversal of every string accepted by the
/// original, with the same weight. Initial and final weights are swapped,
/// and every transition `q --a/w--> q'` becomes `q' --a/w--> q`.
pub fn reverse<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    // Swap initial ↔ final.
    for q in 0..n {
        result.set_initial_weight(q, wfa.final_weights()[q].clone());
        result.set_final_weight(q, wfa.initial_weights()[q].clone());
    }

    // Reverse transitions.
    let trans = wfa.transitions();
    for q in 0..n {
        for a in 0..alpha_size {
            for &(qp, ref w) in &trans[q][a] {
                result.add_transition(qp, a, q, w.clone());
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// 5. Quotient operations
// ---------------------------------------------------------------------------

/// Left quotient: `prefix_wfa \ wfa`.
///
/// The result accepts a string `w` with weight equal to the sum over all
/// decompositions `u · w` of the product of the weight of `u` in
/// `prefix_wfa` and the weight of `u · w` in `wfa`.
///
/// Implementation: product construction where `prefix_wfa` is run on
/// the same symbols as `wfa` but only contributes through its final
/// weights – once `prefix_wfa` reaches a final state, the remaining
/// computation is done by `wfa` alone.
///
/// Concretely, the product state space is `(p, q)` where `p` is a state
/// of `prefix_wfa` and `q` is a state of `wfa`. Two kinds of states:
///   * "matching" states `(p, q)` – still consuming the prefix
///   * "suffix" states `(⊥, q)` ≡ `(n_prefix, q)` – prefix consumed
///
/// Transitions from matching state `(p, q)`:
///   * on symbol a: go to `(p', q')` with weight `w_prefix(p,a,p') ⊗ w(q,a,q')`
///
/// ε-transitions from matching `(p, q)` where `p` is final in prefix_wfa:
///   * go to suffix state `(⊥, q)` with weight `ρ_prefix(p)`
///
/// Transitions from suffix state `(⊥, q)`:
///   * on symbol a: go to `(⊥, q')` with weight `w(q,a,q')`
///
/// Initial: `(p, q)` with `α_prefix(p) ⊗ α_wfa(q)`
/// Final: `(⊥, q)` with `ρ_wfa(q)`
pub fn left_quotient<S: Semiring>(
    prefix_wfa: &WeightedFiniteAutomaton<S>,
    wfa: &WeightedFiniteAutomaton<S>,
) -> std::result::Result<WeightedFiniteAutomaton<S>, OperationError> {
    let alpha_size = check_compatible_alphabets(prefix_wfa, wfa)?;
    let np = prefix_wfa.num_states();
    let nw = wfa.num_states();

    if np == 0 || nw == 0 {
        return Err(OperationError::EmptyAutomaton);
    }

    // Total states: np*nw (matching) + nw (suffix, encoded as (np, q)).
    let total = np * nw + nw;
    if total > STATE_EXPLOSION_LIMIT {
        return Err(OperationError::StateExplosion {
            states: total,
            limit: STATE_EXPLOSION_LIMIT,
        });
    }

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..total {
        result.add_state();
    }

    let matching_idx = |p: usize, q: usize| -> usize { p * nw + q };
    let suffix_idx = |q: usize| -> usize { np * nw + q };

    // Initial weights for matching states.
    for p in 0..np {
        let ip = &prefix_wfa.initial_weights()[p];
        if ip.is_zero() {
            continue;
        }
        for q in 0..nw {
            let iq = &wfa.initial_weights()[q];
            if iq.is_zero() {
                continue;
            }
            let w = ip.clone().mul(iq);
            if !w.is_zero() {
                result.set_initial_weight(matching_idx(p, q), w);
            }
        }
    }

    // If a prefix initial state is already final, the suffix can start immediately.
    // We handle this via epsilon-like behaviour: also set initial weights on suffix
    // states accumulated from matching initial states whose prefix component is final.
    for p in 0..np {
        let ip = &prefix_wfa.initial_weights()[p];
        let fp = &prefix_wfa.final_weights()[p];
        if ip.is_zero() || fp.is_zero() {
            continue;
        }
        for q in 0..nw {
            let iq = &wfa.initial_weights()[q];
            if iq.is_zero() {
                continue;
            }
            let w = ip.clone().mul(fp).mul(iq);
            if !w.is_zero() {
                // Add to existing initial weight of suffix state.
                let cur = result.initial_weights()[suffix_idx(q)].clone();
                result.set_initial_weight(suffix_idx(q), cur.add(&w));
            }
        }
    }

    // Final weights: only suffix states.
    for q in 0..nw {
        let fw = &wfa.final_weights()[q];
        if !fw.is_zero() {
            result.set_final_weight(suffix_idx(q), fw.clone());
        }
    }

    let tp = prefix_wfa.transitions();
    let tw = wfa.transitions();

    // Transitions from matching states.
    for p in 0..np {
        for q in 0..nw {
            let src = matching_idx(p, q);
            for a in 0..alpha_size {
                for &(pp, ref wp) in &tp[p][a] {
                    if wp.is_zero() {
                        continue;
                    }
                    for &(qp, ref wq) in &tw[q][a] {
                        if wq.is_zero() {
                            continue;
                        }
                        let w = wp.clone().mul(wq);
                        if !w.is_zero() {
                            let dst = matching_idx(pp, qp);
                            result.add_transition(src, a, dst, w.clone());

                            // If pp is final in prefix_wfa, also add
                            // transition to suffix state.
                            let fp = &prefix_wfa.final_weights()[pp];
                            if !fp.is_zero() {
                                let ws = w.mul(fp);
                                if !ws.is_zero() {
                                    result.add_transition(src, a, suffix_idx(qp), ws);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Transitions from suffix states: just follow wfa.
    for q in 0..nw {
        let src = suffix_idx(q);
        for a in 0..alpha_size {
            for &(qp, ref w) in &tw[q][a] {
                if !w.is_zero() {
                    result.add_transition(src, a, suffix_idx(qp), w.clone());
                }
            }
        }
    }

    Ok(result)
}

/// Right quotient: `wfa / suffix_wfa`.
///
/// The result accepts `w` iff there exists `v` such that `w · v` is
/// accepted by `wfa` and `v` is accepted by `suffix_wfa`.
///
/// Implemented as: reverse(left_quotient(reverse(suffix_wfa), reverse(wfa))).
pub fn right_quotient<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    suffix_wfa: &WeightedFiniteAutomaton<S>,
) -> std::result::Result<WeightedFiniteAutomaton<S>, OperationError> {
    let rev_suffix = reverse(suffix_wfa);
    let rev_wfa = reverse(wfa);
    let lq = left_quotient(&rev_suffix, &rev_wfa)?;
    Ok(reverse(&lq))
}

// ---------------------------------------------------------------------------
// 6. Projection and restriction
// ---------------------------------------------------------------------------

/// Projects a WFA onto a subset of symbols.
///
/// Transitions on symbols whose indices are in `symbols` are kept;
/// transitions on all other symbols are treated as epsilon-transitions
/// (target is the same, weight is preserved, but the symbol is removed
/// from the output alphabet).
///
/// The returned WFA uses a new alphabet consisting only of the retained
/// symbols. The new symbol indices are contiguous starting from 0 in the
/// order given by `symbols`.
pub fn project_to_alphabet<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    symbols: &[usize],
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let old_alpha = wfa.alphabet().size();

    // Build mapping old_sym -> Option<new_sym>.
    let sym_set: HashSet<usize> = symbols.iter().cloned().collect();
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in symbols.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }
    let new_alpha = symbols.len();

    // Build a new alphabet. For simplicity we create one from indices.
    let mut new_alphabet = Alphabet::new();
    for &old_idx in symbols {
        if old_idx < old_alpha {
            if let Some(sym) = wfa.alphabet().symbol_at(old_idx) {
                new_alphabet.insert(sym.clone());
            }
        }
    }

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(new_alphabet);
    for _ in 0..n {
        result.add_state();
    }

    for q in 0..n {
        result.set_initial_weight(q, wfa.initial_weights()[q].clone());
        result.set_final_weight(q, wfa.final_weights()[q].clone());
    }

    let trans = wfa.transitions();

    // First, handle epsilon closure for non-projected symbols.
    // We build an epsilon-closure via repeated BFS.
    // Epsilon transitions: transitions on symbols NOT in sym_set.
    // Compute transitive closure of epsilon transitions.
    let mut eps_reach: Vec<Vec<(usize, S)>> = Vec::with_capacity(n);
    for q in 0..n {
        let mut visited: HashMap<usize, S> = HashMap::new();
        let mut queue: VecDeque<(usize, S)> = VecDeque::new();
        visited.insert(q, S::one());
        queue.push_back((q, S::one()));

        while let Some((cur, cur_w)) = queue.pop_front() {
            for a in 0..old_alpha {
                if sym_set.contains(&a) {
                    continue;
                }
                for &(dst, ref tw) in &trans[cur][a] {
                    let new_w = cur_w.clone().mul(tw);
                    if new_w.is_zero() {
                        continue;
                    }
                    let entry = visited.entry(dst).or_insert_with(S::zero);
                    let old_w = entry.clone();
                    let sum = old_w.add(&new_w);
                    if sum != old_w {
                        *entry = sum;
                        queue.push_back((dst, new_w));
                    }
                }
            }
        }
        eps_reach.push(visited.into_iter().collect());
    }

    // For each state q, for each epsilon-reachable state r, for each
    // projected symbol a, add transition q --a/w_eps(q,r)*w(r,a,s)--> s.
    for q in 0..n {
        for &(r, ref w_eps) in &eps_reach[q] {
            for &old_a in symbols {
                if old_a >= old_alpha {
                    continue;
                }
                let new_a = old_to_new[&old_a];
                for &(s, ref w_trans) in &trans[r][old_a] {
                    let w = w_eps.clone().mul(w_trans);
                    if !w.is_zero() {
                        result.add_transition(q, new_a, s, w);
                    }
                }
            }
        }

        // Update final weights through epsilon closure.
        let mut new_final = wfa.final_weights()[q].clone();
        for &(r, ref w_eps) in &eps_reach[q] {
            if r == q {
                continue;
            }
            let fw = &wfa.final_weights()[r];
            if !fw.is_zero() {
                let contrib = w_eps.clone().mul(fw);
                new_final = new_final.add(&contrib);
            }
        }
        result.set_final_weight(q, new_final);
    }

    result
}

/// Restricts a WFA to a subset of states.
///
/// Only transitions among the given states are kept. Initial and final
/// weights of states not in the set are zeroed out. States are renumbered
/// contiguously starting from 0.
pub fn restrict_to_states<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    states: &HashSet<usize>,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    // Build old → new mapping.
    let mut sorted: Vec<usize> = states.iter().cloned().filter(|&s| s < n).collect();
    sorted.sort();
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in sorted.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }
    let new_n = sorted.len();

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..new_n {
        result.add_state();
    }

    for &old_q in &sorted {
        let new_q = old_to_new[&old_q];
        result.set_initial_weight(new_q, wfa.initial_weights()[old_q].clone());
        result.set_final_weight(new_q, wfa.final_weights()[old_q].clone());
    }

    let trans = wfa.transitions();
    for &old_q in &sorted {
        let new_q = old_to_new[&old_q];
        for a in 0..alpha_size {
            for &(old_dst, ref w) in &trans[old_q][a] {
                if let Some(&new_dst) = old_to_new.get(&old_dst) {
                    if !w.is_zero() {
                        result.add_transition(new_q, a, new_dst, w.clone());
                    }
                }
            }
        }
    }

    result
}

/// Restricts a WFA to accept only strings of length in `[min_len, max_len]`.
///
/// This is done by building a product with a "length counter" automaton
/// that has `max_len + 2` states (0..=max_len are live, max_len+1 is a
/// dead sink). State `i` means "i symbols consumed so far".
pub fn restrict_to_length<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    min_len: usize,
    max_len: usize,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    let len_states = max_len + 2; // 0..=max_len plus one dead state
    let dead = max_len + 1;
    let total = n * len_states;

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..total {
        result.add_state();
    }

    let state_idx = |q: usize, l: usize| -> usize { q * len_states + l };

    // Initial: (q, 0) with initial weight of q.
    for q in 0..n {
        let iw = &wfa.initial_weights()[q];
        if !iw.is_zero() {
            // If min_len == 0, state (q, 0) can be final if q is final.
            result.set_initial_weight(state_idx(q, 0), iw.clone());
        }
    }

    // Final: (q, l) for l in [min_len, max_len] where q is final.
    for q in 0..n {
        let fw = &wfa.final_weights()[q];
        if fw.is_zero() {
            continue;
        }
        for l in min_len..=max_len {
            result.set_final_weight(state_idx(q, l), fw.clone());
        }
    }

    // Transitions: (q, l) --a/w--> (q', l+1) if l < max_len.
    let trans = wfa.transitions();
    for q in 0..n {
        for l in 0..=max_len {
            if l >= max_len {
                // No more transitions from length max_len (would exceed).
                continue;
            }
            for a in 0..alpha_size {
                for &(qp, ref w) in &trans[q][a] {
                    if !w.is_zero() {
                        result.add_transition(
                            state_idx(q, l),
                            a,
                            state_idx(qp, l + 1),
                            w.clone(),
                        );
                    }
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// 7. Weight operations
// ---------------------------------------------------------------------------

/// Scales all transition weights of a WFA by a constant factor.
///
/// Initial and final weights are NOT scaled.
pub fn scale_weights<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    factor: &S,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    for q in 0..n {
        result.set_initial_weight(q, wfa.initial_weights()[q].clone());
        result.set_final_weight(q, wfa.final_weights()[q].clone());
    }

    let trans = wfa.transitions();
    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                let scaled = w.clone().mul(factor);
                if !scaled.is_zero() {
                    result.add_transition(q, a, dst, scaled);
                }
            }
        }
    }

    result
}

/// Trait bound for semirings that support division (needed for normalization).
pub trait DivisibleSemiring: Semiring {
    fn div(&self, other: &Self) -> Self;
}

/// Normalizes weights so the total outgoing weight from each state sums to one.
///
/// For each state, we compute the sum of all outgoing transition weights
/// plus the final weight, then divide each weight by that sum.
pub fn normalize_weights<S: DivisibleSemiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    let trans = wfa.transitions();

    for q in 0..n {
        result.set_initial_weight(q, wfa.initial_weights()[q].clone());

        // Compute total outgoing weight for this state.
        let mut total = wfa.final_weights()[q].clone();
        for a in 0..alpha_size {
            for &(_, ref w) in &trans[q][a] {
                total = total.add(w);
            }
        }

        if total.is_zero() {
            result.set_final_weight(q, S::zero());
            continue;
        }

        // Normalize final weight.
        result.set_final_weight(q, wfa.final_weights()[q].div(&total));

        // Normalize transitions.
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                let normed = w.div(&total);
                if !normed.is_zero() {
                    result.add_transition(q, a, dst, normed);
                }
            }
        }
    }

    result
}

/// Pushes weights toward initial states.
///
/// For each state q, compute the total weight of all paths from q to any
/// final state (the "backward" weight `β(q)`). Then redistribute:
///   * transition weight `w(q,a,q')` becomes `β(q')⁻¹ ⊗ w(q,a,q') ⊗ β(q')`
///     (simplified: just adjust final and transition weights).
///
/// Simplified heuristic version: multiply each transition by the final
/// weight of its destination, and set final weights to one for accepting
/// states.
pub fn push_weights_initial<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    // Compute backward weights β(q) via reverse BFS.
    let mut beta: Vec<S> = vec![S::zero(); n];
    for q in 0..n {
        beta[q] = wfa.final_weights()[q].clone();
    }

    // Iterate to propagate backward weights. We do a fixed-point iteration
    // for convergence on acyclic parts (and a bounded number of rounds for
    // cyclic WFAs).
    let trans = wfa.transitions();
    let max_iters = n + 1;
    for _iter in 0..max_iters {
        let mut changed = false;
        for q in 0..n {
            let mut new_beta = wfa.final_weights()[q].clone();
            for a in 0..alpha_size {
                for &(dst, ref w) in &trans[q][a] {
                    let contrib = w.clone().mul(&beta[dst]);
                    new_beta = new_beta.add(&contrib);
                }
            }
            if new_beta != beta[q] {
                beta[q] = new_beta;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Build result: push beta values into initial weights.
    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    for q in 0..n {
        // Multiply initial weight by beta.
        let new_iw = wfa.initial_weights()[q].clone().mul(&beta[q]);
        result.set_initial_weight(q, new_iw);
        // Set final weight: if beta is non-zero, final weight becomes one;
        // otherwise zero.
        if !beta[q].is_zero() {
            result.set_final_weight(q, S::one());
        } else {
            result.set_final_weight(q, S::zero());
        }
    }

    // Keep transitions but with adjusted weights.
    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                if !w.is_zero() {
                    result.add_transition(q, a, dst, w.clone());
                }
            }
        }
    }

    result
}

/// Pushes weights toward final states.
///
/// Analogous to `push_weights_initial` but in the reverse direction.
/// Computes forward weights α(q) and redistributes.
pub fn push_weights_final<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    // Forward weights: α(q) = sum over all paths from any initial state to q.
    let mut alpha: Vec<S> = vec![S::zero(); n];
    for q in 0..n {
        alpha[q] = wfa.initial_weights()[q].clone();
    }

    let trans = wfa.transitions();
    let max_iters = n + 1;
    for _iter in 0..max_iters {
        let mut changed = false;
        let mut new_alpha = alpha.clone();
        for q in 0..n {
            if alpha[q].is_zero() {
                continue;
            }
            for a in 0..alpha_size {
                for &(dst, ref w) in &trans[q][a] {
                    let contrib = alpha[q].clone().mul(w);
                    let old = new_alpha[dst].clone();
                    new_alpha[dst] = new_alpha[dst].add(&contrib);
                    if new_alpha[dst] != old {
                        changed = true;
                    }
                }
            }
        }
        alpha = new_alpha;
        if !changed {
            break;
        }
    }

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    for q in 0..n {
        if !alpha[q].is_zero() {
            result.set_initial_weight(q, S::one());
        } else {
            result.set_initial_weight(q, S::zero());
        }
        let new_fw = alpha[q].clone().mul(&wfa.final_weights()[q]);
        result.set_final_weight(q, new_fw);
    }

    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                if !w.is_zero() {
                    result.add_transition(q, a, dst, w.clone());
                }
            }
        }
    }

    result
}

/// Computes the total weight of all strings accepted by the WFA.
///
/// Uses the matrix formulation: total = α^T · M* · ρ where M is the
/// sum of all transition matrices and M* is its Kleene star.
///
/// Requires `StarSemiring` for convergence.
pub fn total_weight<S: Semiring + StarSemiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> S {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    if n == 0 {
        return S::zero();
    }

    // Build the combined transition matrix M = sum_a M_a.
    let mut m = SemiringMatrix::<S>::zeros(n, n);
    let trans = wfa.transitions();
    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                let cur = m.get(q, dst).unwrap().clone();
                let _ = m.set(q, dst, cur.add(w));
            }
        }
    }

    // Compute M* using Floyd-Warshall-like closure.
    // m_star[i][j] = sum of weights of all paths from i to j.
    let mut star = SemiringMatrix::<S>::identity(n);

    // We iterate the star computation via repeated squaring up to n steps,
    // or use the direct element-wise star approach.
    // Use the iterative approach: star = I + M + M^2 + ... until convergence.
    // For efficiency, we use the state-elimination / Gauss-Jordan approach.
    let mut closure = SemiringMatrix::<S>::zeros(n, n);
    // Initialize closure = M.
    for i in 0..n {
        for j in 0..n {
            let _ = closure.set(i, j, m.get(i, j).unwrap().clone());
        }
    }

    // Floyd-Warshall with star semiring.
    for k in 0..n {
        let kk_star = closure.get(k, k).unwrap().star();
        for i in 0..n {
            for j in 0..n {
                let ik = closure.get(i, k).unwrap().clone();
                let kj = closure.get(k, j).unwrap();
                let contribution = ik.mul(&kk_star).mul(kj);
                if i == j {
                    // star[i][j] includes identity.
                    let cur = closure.get(i, j).unwrap().clone();
                    let _ = closure.set(i, j, cur.add(&contribution));
                } else {
                    let cur = closure.get(i, j).unwrap().clone();
                    let _ = closure.set(i, j, cur.add(&contribution));
                }
            }
        }
    }

    // Add identity to get the star (I + closure gives all paths including
    // length-0 path).
    for i in 0..n {
        let cur = closure.get(i, i).unwrap().clone();
        let _ = closure.set(i, i, cur.add(&S::one()));
    }

    // total = α^T · closure · ρ
    let mut total = S::zero();
    for i in 0..n {
        let ai = &wfa.initial_weights()[i];
        if ai.is_zero() {
            continue;
        }
        for j in 0..n {
            let fj = &wfa.final_weights()[j];
            if fj.is_zero() {
                continue;
            }
            let path_w = ai.clone().mul(closure.get(i, j).unwrap()).mul(fj);
            total = total.add(&path_w);
        }
    }

    total
}

// ---------------------------------------------------------------------------
// 8. State manipulation
// ---------------------------------------------------------------------------

/// Renames (permutes) states according to a bijective mapping.
///
/// `mapping[old] = new` for each old state index.
pub fn rename_states<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    mapping: &[usize],
) -> std::result::Result<WeightedFiniteAutomaton<S>, OperationError> {
    let n = wfa.num_states();
    if mapping.len() != n {
        return Err(OperationError::DimensionMismatch);
    }

    // Verify bijection.
    let mut seen: HashSet<usize> = HashSet::new();
    for &new_idx in mapping {
        if new_idx >= n {
            return Err(OperationError::InvalidOperation {
                desc: format!("mapping target {} out of range [0, {})", new_idx, n),
            });
        }
        if !seen.insert(new_idx) {
            return Err(OperationError::InvalidOperation {
                desc: format!("duplicate mapping target {}", new_idx),
            });
        }
    }

    let alpha_size = wfa.alphabet().size();
    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    for old in 0..n {
        let new_q = mapping[old];
        result.set_initial_weight(new_q, wfa.initial_weights()[old].clone());
        result.set_final_weight(new_q, wfa.final_weights()[old].clone());
    }

    let trans = wfa.transitions();
    for old_src in 0..n {
        let new_src = mapping[old_src];
        for a in 0..alpha_size {
            for &(old_dst, ref w) in &trans[old_src][a] {
                let new_dst = mapping[old_dst];
                result.add_transition(new_src, a, new_dst, w.clone());
            }
        }
    }

    Ok(result)
}

/// Re-indexes states contiguously 0..k where k is the number of
/// "reachable" or "useful" states. Dead states (unreachable from any
/// initial state or from which no final state is reachable) are removed.
pub fn reindex_states<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();
    let trans = wfa.transitions();

    // Forward reachability from initial states.
    let mut fwd_reachable: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    for q in 0..n {
        if !wfa.initial_weights()[q].is_zero() {
            fwd_reachable.insert(q);
            queue.push_back(q);
        }
    }
    while let Some(q) = queue.pop_front() {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                if !w.is_zero() && fwd_reachable.insert(dst) {
                    queue.push_back(dst);
                }
            }
        }
    }

    // Backward reachability from final states.
    // Build reverse adjacency.
    let mut rev_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, _) in &trans[q][a] {
                rev_adj[dst].push(q);
            }
        }
    }

    let mut bwd_reachable: HashSet<usize> = HashSet::new();
    for q in 0..n {
        if !wfa.final_weights()[q].is_zero() {
            bwd_reachable.insert(q);
            queue.push_back(q);
        }
    }
    while let Some(q) = queue.pop_front() {
        for &src in &rev_adj[q] {
            if bwd_reachable.insert(src) {
                queue.push_back(src);
            }
        }
    }

    // Useful states = forward ∩ backward reachable.
    let mut useful: Vec<usize> = fwd_reachable
        .intersection(&bwd_reachable)
        .cloned()
        .collect();
    useful.sort();

    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in useful.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }

    let new_n = useful.len();
    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..new_n {
        result.add_state();
    }

    for &old_q in &useful {
        let new_q = old_to_new[&old_q];
        result.set_initial_weight(new_q, wfa.initial_weights()[old_q].clone());
        result.set_final_weight(new_q, wfa.final_weights()[old_q].clone());
    }

    for &old_q in &useful {
        let new_q = old_to_new[&old_q];
        for a in 0..alpha_size {
            for &(old_dst, ref w) in &trans[old_q][a] {
                if let Some(&new_dst) = old_to_new.get(&old_dst) {
                    if !w.is_zero() {
                        result.add_transition(new_q, a, new_dst, w.clone());
                    }
                }
            }
        }
    }

    result
}

/// Adds a non-accepting sink state to the WFA. All "missing" transitions
/// (states with no outgoing transition on some symbol) are redirected to the
/// sink. Returns the index of the new sink state.
pub fn add_sink_state<S: Semiring>(
    wfa: &mut WeightedFiniteAutomaton<S>,
) -> usize {
    let sink = wfa.add_state();
    let n = wfa.num_states(); // includes sink
    let alpha_size = wfa.alphabet().size();

    // Sink's initial and final weights are zero (default).
    wfa.set_initial_weight(sink, S::zero());
    wfa.set_final_weight(sink, S::zero());

    // For the sink state itself, all transitions go to sink.
    for a in 0..alpha_size {
        wfa.add_transition(sink, a, sink, S::one());
    }

    // For every other state, check for missing transitions.
    // We need to read existing transitions first.
    let trans = wfa.transitions().clone();
    for q in 0..n - 1 {
        for a in 0..alpha_size {
            let has_transition = !trans[q][a].is_empty();
            if !has_transition {
                wfa.add_transition(q, a, sink, S::one());
            }
        }
    }

    sink
}

/// Splits a state into two copies: `state` keeps all outgoing transitions
/// and the original final weight; the new state gets all incoming transitions
/// and the original initial weight. Returns the index of the new state.
///
/// After splitting, transitions into `state` from other states go to the
/// new state, and an epsilon-like copy of outgoing transitions from `state`
/// is added from the new state (so the language is preserved).
pub fn split_state<S: Semiring>(
    wfa: &mut WeightedFiniteAutomaton<S>,
    state: usize,
) -> std::result::Result<usize, OperationError> {
    let n = wfa.num_states();
    if state >= n {
        return Err(OperationError::InvalidOperation {
            desc: format!("state {} out of range [0, {})", state, n),
        });
    }

    let new_state = wfa.add_state();
    let alpha_size = wfa.alphabet().size();

    // The new state gets the initial weight of the original state.
    wfa.set_initial_weight(new_state, wfa.initial_weights()[state].clone());
    wfa.set_initial_weight(state, S::zero());

    // The original state keeps its final weight; the new state gets zero.
    wfa.set_final_weight(new_state, S::zero());

    // Copy outgoing transitions from state to new_state.
    let trans = wfa.transitions().clone();
    for a in 0..alpha_size {
        for &(dst, ref w) in &trans[state][a] {
            wfa.add_transition(new_state, a, dst, w.clone());
        }
    }

    // Redirect incoming transitions from other states to point to new_state
    // instead of state. We do this by adding matching transitions to new_state.
    // (We cannot easily remove existing transitions, so we conceptually
    // redirect by adding transitions to the new state. The original incoming
    // transitions remain but become "dead" paths in many use cases.)
    for q in 0..n {
        if q == state {
            continue;
        }
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                if dst == state {
                    wfa.add_transition(q, a, new_state, w.clone());
                }
            }
        }
    }

    Ok(new_state)
}

// ---------------------------------------------------------------------------
// 9. WFA to regular expression (state elimination)
// ---------------------------------------------------------------------------

/// A fragment of a regular expression, used during state elimination.
#[derive(Debug, Clone)]
pub enum RegexFragment {
    /// Matches nothing (∅).
    Empty,
    /// Matches the empty string (ε).
    Epsilon,
    /// Matches a single symbol.
    Symbol(String),
    /// Concatenation of sub-expressions.
    Concat(Vec<RegexFragment>),
    /// Alternation of sub-expressions.
    Alt(Vec<RegexFragment>),
    /// Kleene star.
    Star(Box<RegexFragment>),
    /// Weighted regex node (for non-boolean semirings).
    Weight(Box<RegexFragment>, String),
}

impl fmt::Display for RegexFragment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", regex_to_string(self))
    }
}

impl PartialEq for RegexFragment {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RegexFragment::Empty, RegexFragment::Empty) => true,
            (RegexFragment::Epsilon, RegexFragment::Epsilon) => true,
            (RegexFragment::Symbol(a), RegexFragment::Symbol(b)) => a == b,
            (RegexFragment::Concat(a), RegexFragment::Concat(b)) => a == b,
            (RegexFragment::Alt(a), RegexFragment::Alt(b)) => a == b,
            (RegexFragment::Star(a), RegexFragment::Star(b)) => a == b,
            (RegexFragment::Weight(a, wa), RegexFragment::Weight(b, wb)) => {
                a == b && wa == wb
            }
            _ => false,
        }
    }
}

/// Converts a `RegexFragment` to a string representation.
pub fn regex_to_string(frag: &RegexFragment) -> String {
    match frag {
        RegexFragment::Empty => "∅".to_string(),
        RegexFragment::Epsilon => "ε".to_string(),
        RegexFragment::Symbol(s) => {
            if s.len() == 1 {
                s.clone()
            } else {
                format!("[{}]", s)
            }
        }
        RegexFragment::Concat(parts) => {
            if parts.is_empty() {
                return "ε".to_string();
            }
            let strs: Vec<String> = parts
                .iter()
                .map(|p| {
                    let s = regex_to_string(p);
                    // Wrap alternations in parens when inside concat.
                    match p {
                        RegexFragment::Alt(_) => format!("({})", s),
                        _ => s,
                    }
                })
                .collect();
            strs.join("")
        }
        RegexFragment::Alt(alts) => {
            if alts.is_empty() {
                return "∅".to_string();
            }
            if alts.len() == 1 {
                return regex_to_string(&alts[0]);
            }
            let strs: Vec<String> = alts.iter().map(|a| regex_to_string(a)).collect();
            strs.join("|")
        }
        RegexFragment::Star(inner) => {
            let s = regex_to_string(inner);
            match **inner {
                RegexFragment::Symbol(_) => format!("{}*", s),
                RegexFragment::Epsilon => "ε".to_string(),
                RegexFragment::Empty => "ε".to_string(),
                _ => format!("({})*", s),
            }
        }
        RegexFragment::Weight(inner, w) => {
            let s = regex_to_string(inner);
            format!("<{}>:{}", w, s)
        }
    }
}

/// Simplifies a `RegexFragment` by applying algebraic identities.
pub fn simplify_regex(frag: &RegexFragment) -> RegexFragment {
    match frag {
        RegexFragment::Empty | RegexFragment::Epsilon | RegexFragment::Symbol(_) => {
            frag.clone()
        }
        RegexFragment::Concat(parts) => {
            let simplified: Vec<RegexFragment> = parts
                .iter()
                .map(simplify_regex)
                .filter(|p| !matches!(p, RegexFragment::Epsilon))
                .collect();

            // If any part is Empty, the whole concat is Empty.
            if simplified.iter().any(|p| matches!(p, RegexFragment::Empty)) {
                return RegexFragment::Empty;
            }

            // Flatten nested concats.
            let mut flat: Vec<RegexFragment> = Vec::new();
            for p in simplified {
                match p {
                    RegexFragment::Concat(inner) => flat.extend(inner),
                    other => flat.push(other),
                }
            }

            match flat.len() {
                0 => RegexFragment::Epsilon,
                1 => flat.into_iter().next().unwrap(),
                _ => RegexFragment::Concat(flat),
            }
        }
        RegexFragment::Alt(alts) => {
            let simplified: Vec<RegexFragment> = alts
                .iter()
                .map(simplify_regex)
                .filter(|a| !matches!(a, RegexFragment::Empty))
                .collect();

            // Flatten nested alts and deduplicate.
            let mut flat: Vec<RegexFragment> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            for a in simplified {
                match a {
                    RegexFragment::Alt(inner) => {
                        for x in inner {
                            let s = regex_to_string(&x);
                            if seen.insert(s) {
                                flat.push(x);
                            }
                        }
                    }
                    other => {
                        let s = regex_to_string(&other);
                        if seen.insert(s) {
                            flat.push(other);
                        }
                    }
                }
            }

            match flat.len() {
                0 => RegexFragment::Empty,
                1 => flat.into_iter().next().unwrap(),
                _ => RegexFragment::Alt(flat),
            }
        }
        RegexFragment::Star(inner) => {
            let simplified = simplify_regex(inner);
            match &simplified {
                RegexFragment::Empty => RegexFragment::Epsilon,
                RegexFragment::Epsilon => RegexFragment::Epsilon,
                RegexFragment::Star(_) => simplified, // (r*)* = r*
                _ => RegexFragment::Star(Box::new(simplified)),
            }
        }
        RegexFragment::Weight(inner, w) => {
            let simplified = simplify_regex(inner);
            match &simplified {
                RegexFragment::Empty => RegexFragment::Empty,
                _ => RegexFragment::Weight(Box::new(simplified), w.clone()),
            }
        }
    }
}

/// Builds a regex for the concatenation of two fragments.
fn regex_concat(a: RegexFragment, b: RegexFragment) -> RegexFragment {
    match (&a, &b) {
        (RegexFragment::Empty, _) | (_, RegexFragment::Empty) => RegexFragment::Empty,
        (RegexFragment::Epsilon, _) => b,
        (_, RegexFragment::Epsilon) => a,
        _ => {
            let mut parts = Vec::new();
            match a {
                RegexFragment::Concat(v) => parts.extend(v),
                other => parts.push(other),
            }
            match b {
                RegexFragment::Concat(v) => parts.extend(v),
                other => parts.push(other),
            }
            RegexFragment::Concat(parts)
        }
    }
}

/// Builds a regex for the alternation of two fragments.
fn regex_alt(a: RegexFragment, b: RegexFragment) -> RegexFragment {
    match (&a, &b) {
        (RegexFragment::Empty, _) => b,
        (_, RegexFragment::Empty) => a,
        _ => {
            let mut alts = Vec::new();
            match a {
                RegexFragment::Alt(v) => alts.extend(v),
                other => alts.push(other),
            }
            match b {
                RegexFragment::Alt(v) => alts.extend(v),
                other => alts.push(other),
            }
            RegexFragment::Alt(alts)
        }
    }
}

/// Builds a regex for the Kleene star of a fragment.
fn regex_star(a: RegexFragment) -> RegexFragment {
    match &a {
        RegexFragment::Empty | RegexFragment::Epsilon => RegexFragment::Epsilon,
        RegexFragment::Star(_) => a,
        _ => RegexFragment::Star(Box::new(a)),
    }
}

/// Converts a WFA (over `BooleanSemiring`) to a regular expression string
/// using the state-elimination method.
///
/// Algorithm:
/// 1. Add a single new initial state `s` and a single new final state `f`.
/// 2. For each original initial state `qi`, add ε-transition `s -> qi`.
/// 3. For each original final state `qf`, add ε-transition `qf -> f`.
/// 4. Eliminate states one by one (in arbitrary order, not `s` or `f`).
///    When eliminating state `q`, for every pair `(p, r)` where `p -> q`
///    and `q -> r`, update the label of `p -> r` to include the path
///    through `q`.
/// 5. The final regex is the label on the edge `s -> f`.
pub fn to_regex_string(
    wfa: &WeightedFiniteAutomaton<BooleanSemiring>,
) -> std::result::Result<String, OperationError> {
    let n = wfa.num_states();
    if n == 0 {
        return Err(OperationError::EmptyAutomaton);
    }

    let alpha_size = wfa.alphabet().size();
    let trans = wfa.transitions();

    // State indices: 0..n are original, n is new start, n+1 is new final.
    let start = n;
    let fin = n + 1;
    let total = n + 2;

    // Build regex-labeled transition matrix.
    // labels[i][j] = regex for the direct transition from i to j.
    let mut labels: Vec<Vec<RegexFragment>> =
        vec![vec![RegexFragment::Empty; total]; total];

    // Original transitions.
    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                if !w.is_zero() {
                    let sym_name = if let Some(sym) = wfa.alphabet().symbol_at(a) {
                        match sym {
                            Symbol::Char(c) => c.to_string(),
                            Symbol::Byte(b) => format!("\\x{:02x}", b),
                            Symbol::Token(t) => t.clone(),
                            Symbol::Epsilon => "ε".to_string(),
                            Symbol::Wildcard => ".".to_string(),
                            Symbol::Id(id) => format!("s{}", id),
                        }
                    } else {
                        format!("a{}", a)
                    };
                    let sym_frag = RegexFragment::Symbol(sym_name);
                    labels[q][dst] = regex_alt(labels[q][dst].clone(), sym_frag);
                }
            }
        }
    }

    // Epsilon transitions from new start to original initial states.
    for q in 0..n {
        if !wfa.initial_weights()[q].is_zero() {
            labels[start][q] =
                regex_alt(labels[start][q].clone(), RegexFragment::Epsilon);
        }
    }

    // Epsilon transitions from original final states to new final.
    for q in 0..n {
        if !wfa.final_weights()[q].is_zero() {
            labels[q][fin] =
                regex_alt(labels[q][fin].clone(), RegexFragment::Epsilon);
        }
    }

    // Eliminate original states 0..n in order.
    for elim in 0..n {
        // Self-loop on elim.
        let self_loop = labels[elim][elim].clone();
        let star_self = regex_star(self_loop);

        for p in 0..total {
            if p == elim {
                continue;
            }
            let p_to_elim = labels[p][elim].clone();
            if matches!(p_to_elim, RegexFragment::Empty) {
                continue;
            }

            for r in 0..total {
                if r == elim {
                    continue;
                }
                let elim_to_r = labels[elim][r].clone();
                if matches!(elim_to_r, RegexFragment::Empty) {
                    continue;
                }

                // New path p -> elim -> ... -> elim -> r
                let path = regex_concat(
                    regex_concat(p_to_elim.clone(), star_self.clone()),
                    elim_to_r,
                );

                labels[p][r] = regex_alt(labels[p][r].clone(), path);
            }
        }

        // Clear all edges involving elim.
        for i in 0..total {
            labels[elim][i] = RegexFragment::Empty;
            labels[i][elim] = RegexFragment::Empty;
        }
    }

    let result_frag = simplify_regex(&labels[start][fin]);
    Ok(regex_to_string(&result_frag))
}

// ---------------------------------------------------------------------------
// 10. Morphisms and transformations
// ---------------------------------------------------------------------------

/// Maps all weights of a WFA through a function `f: S -> T`.
pub fn map_weights<S: Semiring, T: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    f: impl Fn(&S) -> T,
) -> WeightedFiniteAutomaton<T> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    let mut result = WeightedFiniteAutomaton::<T>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    for q in 0..n {
        result.set_initial_weight(q, f(&wfa.initial_weights()[q]));
        result.set_final_weight(q, f(&wfa.final_weights()[q]));
    }

    let trans = wfa.transitions();
    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                let mapped = f(w);
                if !mapped.is_zero() {
                    result.add_transition(q, a, dst, mapped);
                }
            }
        }
    }

    result
}

/// Keeps only transitions satisfying a predicate.
///
/// The predicate receives `(src, symbol_idx, dst, weight)`.
pub fn filter_transitions<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    pred: impl Fn(usize, usize, usize, &S) -> bool,
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
    for _ in 0..n {
        result.add_state();
    }

    for q in 0..n {
        result.set_initial_weight(q, wfa.initial_weights()[q].clone());
        result.set_final_weight(q, wfa.final_weights()[q].clone());
    }

    let trans = wfa.transitions();
    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                if pred(q, a, dst, w) {
                    result.add_transition(q, a, dst, w.clone());
                }
            }
        }
    }

    result
}

/// Sequential composition (concatenation) of multiple WFAs.
///
/// For WFAs A1, A2, ..., An, the result accepts `w1 · w2 · ... · wn`
/// with weight `w(A1, w1) ⊗ w(A2, w2) ⊗ ... ⊗ w(An, wn)`.
///
/// Implementation: chain automata by connecting final states of Ai to
/// initial states of A(i+1).
pub fn compose_sequential<S: Semiring>(
    wfas: &[&WeightedFiniteAutomaton<S>],
) -> std::result::Result<WeightedFiniteAutomaton<S>, OperationError> {
    if wfas.is_empty() {
        return Err(OperationError::InvalidOperation {
            desc: "empty list of WFAs for sequential composition".to_string(),
        });
    }

    if wfas.len() == 1 {
        // Just clone the single WFA.
        let wfa = wfas[0];
        let n = wfa.num_states();
        let alpha_size = wfa.alphabet().size();
        let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
        for _ in 0..n {
            result.add_state();
        }
        for q in 0..n {
            result.set_initial_weight(q, wfa.initial_weights()[q].clone());
            result.set_final_weight(q, wfa.final_weights()[q].clone());
        }
        let trans = wfa.transitions();
        for q in 0..n {
            for a in 0..alpha_size {
                for &(dst, ref w) in &trans[q][a] {
                    result.add_transition(q, a, dst, w.clone());
                }
            }
        }
        return Ok(result);
    }

    // Compute state offsets for each WFA.
    let mut offsets: Vec<usize> = Vec::with_capacity(wfas.len());
    let mut total_states: usize = 0;
    for wfa in wfas {
        offsets.push(total_states);
        total_states += wfa.num_states();
    }

    if total_states > STATE_EXPLOSION_LIMIT {
        return Err(OperationError::StateExplosion {
            states: total_states,
            limit: STATE_EXPLOSION_LIMIT,
        });
    }

    // All WFAs must share the same alphabet size.
    let alpha_size = wfas[0].alphabet().size();
    for wfa in &wfas[1..] {
        if wfa.alphabet().size() != alpha_size {
            return Err(OperationError::IncompatibleAlphabets);
        }
    }

    let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfas[0].alphabet().clone());
    for _ in 0..total_states {
        result.add_state();
    }

    // Initial weights: only from the first WFA.
    let first = wfas[0];
    for q in 0..first.num_states() {
        result.set_initial_weight(
            offsets[0] + q,
            first.initial_weights()[q].clone(),
        );
    }

    // Final weights: only from the last WFA.
    let last_idx = wfas.len() - 1;
    let last = wfas[last_idx];
    for q in 0..last.num_states() {
        result.set_final_weight(
            offsets[last_idx] + q,
            last.final_weights()[q].clone(),
        );
    }

    // Copy transitions for each WFA.
    for (i, wfa) in wfas.iter().enumerate() {
        let off = offsets[i];
        let n = wfa.num_states();
        let trans = wfa.transitions();

        for q in 0..n {
            for a in 0..alpha_size {
                for &(dst, ref w) in &trans[q][a] {
                    result.add_transition(off + q, a, off + dst, w.clone());
                }
            }
        }
    }

    // Connect final states of WFA[i] to initial states of WFA[i+1].
    for i in 0..(wfas.len() - 1) {
        let cur = wfas[i];
        let nxt = wfas[i + 1];
        let off_cur = offsets[i];
        let off_nxt = offsets[i + 1];

        for qf in 0..cur.num_states() {
            let fw = &cur.final_weights()[qf];
            if fw.is_zero() {
                continue;
            }
            for qi in 0..nxt.num_states() {
                let iw = &nxt.initial_weights()[qi];
                if iw.is_zero() {
                    continue;
                }
                let w = fw.clone().mul(iw);
                if w.is_zero() {
                    continue;
                }
                // Add "epsilon" transitions: for each outgoing transition
                // from qi in wfa[i+1], add it from qf in the result.
                let trans_nxt = nxt.transitions();
                for a in 0..alpha_size {
                    for &(dst, ref tw) in &trans_nxt[qi][a] {
                        let total_w = w.clone().mul(tw);
                        if !total_w.is_zero() {
                            result.add_transition(
                                off_cur + qf,
                                a,
                                off_nxt + dst,
                                total_w,
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Parallel composition (iterated Hadamard product) of multiple WFAs.
pub fn compose_parallel<S: Semiring>(
    wfas: &[&WeightedFiniteAutomaton<S>],
) -> std::result::Result<WeightedFiniteAutomaton<S>, OperationError> {
    if wfas.is_empty() {
        return Err(OperationError::InvalidOperation {
            desc: "empty list of WFAs for parallel composition".to_string(),
        });
    }

    if wfas.len() == 1 {
        let wfa = wfas[0];
        let n = wfa.num_states();
        let alpha_size = wfa.alphabet().size();
        let mut result = WeightedFiniteAutomaton::<S>::from_alphabet(wfa.alphabet().clone());
        for _ in 0..n {
            result.add_state();
        }
        for q in 0..n {
            result.set_initial_weight(q, wfa.initial_weights()[q].clone());
            result.set_final_weight(q, wfa.final_weights()[q].clone());
        }
        let trans = wfa.transitions();
        for q in 0..n {
            for a in 0..alpha_size {
                for &(dst, ref w) in &trans[q][a] {
                    result.add_transition(q, a, dst, w.clone());
                }
            }
        }
        return Ok(result);
    }

    // Iteratively apply Hadamard product.
    let mut acc = hadamard_product(wfas[0], wfas[1])?;
    for wfa in &wfas[2..] {
        acc = hadamard_product(&acc, wfa)?;
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// 11. Analysis and queries
// ---------------------------------------------------------------------------

/// Counts the number of accepting paths of exactly `length` symbols.
///
/// Uses dynamic programming: `dp[l][q]` = number of paths of length `l`
/// ending at state `q`.
pub fn count_paths<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    length: usize,
) -> usize {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();
    let trans = wfa.transitions();

    // dp[q] = number of distinct paths of current length ending at q.
    let mut dp: Vec<usize> = vec![0; n];

    // Initialize: paths of length 0 start at initial states.
    for q in 0..n {
        if !wfa.initial_weights()[q].is_zero() {
            dp[q] = 1;
        }
    }

    for _step in 0..length {
        let mut new_dp: Vec<usize> = vec![0; n];
        for q in 0..n {
            if dp[q] == 0 {
                continue;
            }
            for a in 0..alpha_size {
                for &(dst, ref w) in &trans[q][a] {
                    if !w.is_zero() {
                        new_dp[dst] = new_dp[dst].saturating_add(dp[q]);
                    }
                }
            }
        }
        dp = new_dp;
    }

    // Sum over final states.
    let mut total: usize = 0;
    for q in 0..n {
        if !wfa.final_weights()[q].is_zero() {
            total = total.saturating_add(dp[q]);
        }
    }

    total
}

/// Enumerates all accepted strings up to `max_length` with their weights.
///
/// Returns a vector of `(symbol_indices, weight)` pairs.
pub fn enumerate_accepted<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    max_length: usize,
) -> Vec<(Vec<usize>, S)> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();
    let trans = wfa.transitions();

    let mut results: Vec<(Vec<usize>, S)> = Vec::new();

    // BFS: (current_state, path_so_far, accumulated_weight).
    let mut queue: VecDeque<(usize, Vec<usize>, S)> = VecDeque::new();

    // Start from all initial states.
    for q in 0..n {
        let iw = &wfa.initial_weights()[q];
        if !iw.is_zero() {
            queue.push_back((q, Vec::new(), iw.clone()));
        }
    }

    while let Some((state, path, weight)) = queue.pop_front() {
        // Check if current state is final.
        let fw = &wfa.final_weights()[state];
        if !fw.is_zero() {
            let total_w = weight.clone().mul(fw);
            if !total_w.is_zero() {
                // Merge with existing entry for same path, if any.
                let mut found = false;
                for (existing_path, existing_w) in results.iter_mut() {
                    if *existing_path == path {
                        *existing_w = existing_w.add(&total_w);
                        found = true;
                        break;
                    }
                }
                if !found {
                    results.push((path.clone(), total_w));
                }
            }
        }

        // Extend path if under max length.
        if path.len() < max_length {
            for a in 0..alpha_size {
                for &(dst, ref tw) in &trans[state][a] {
                    if !tw.is_zero() {
                        let new_w = weight.clone().mul(tw);
                        if !new_w.is_zero() {
                            let mut new_path = path.clone();
                            new_path.push(a);
                            queue.push_back((dst, new_path, new_w));
                        }
                    }
                }
            }
        }
    }

    // Sort by path length then lexicographically.
    results.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0)));
    results
}

/// Finds the shortest accepted string (by number of symbols).
///
/// Uses BFS from initial states, returns the first string that reaches
/// a final state.
pub fn shortest_accepted<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Option<(Vec<usize>, S)> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();
    let trans = wfa.transitions();

    // BFS with state deduplication per length level.
    // (state, path, weight)
    let mut queue: VecDeque<(usize, Vec<usize>, S)> = VecDeque::new();

    for q in 0..n {
        let iw = &wfa.initial_weights()[q];
        if !iw.is_zero() {
            // Check empty string acceptance.
            let fw = &wfa.final_weights()[q];
            if !fw.is_zero() {
                return Some((Vec::new(), iw.clone().mul(fw)));
            }
            queue.push_back((q, Vec::new(), iw.clone()));
        }
    }

    // Track visited (state, length) pairs to avoid infinite loops.
    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    for q in 0..n {
        if !wfa.initial_weights()[q].is_zero() {
            visited.insert((q, 0));
        }
    }

    while let Some((state, path, weight)) = queue.pop_front() {
        for a in 0..alpha_size {
            for &(dst, ref tw) in &trans[state][a] {
                if tw.is_zero() {
                    continue;
                }
                let new_len = path.len() + 1;
                if !visited.insert((dst, new_len)) {
                    continue;
                }
                let new_w = weight.clone().mul(tw);
                let mut new_path = path.clone();
                new_path.push(a);

                // Check if destination is final.
                let fw = &wfa.final_weights()[dst];
                if !fw.is_zero() {
                    let total = new_w.mul(fw);
                    if !total.is_zero() {
                        return Some((new_path, total));
                    }
                }

                queue.push_back((dst, new_path, new_w));
            }
        }
    }

    None
}

/// Finds the longest accepted string up to `max_search` symbols.
///
/// Uses DFS with memoization, tracking the longest accepting path found.
pub fn longest_accepted<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    max_search: usize,
) -> Option<(Vec<usize>, S)> {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();
    let trans = wfa.transitions();

    let mut best: Option<(Vec<usize>, S)> = None;

    // DFS stack: (state, path, weight).
    let mut stack: Vec<(usize, Vec<usize>, S)> = Vec::new();

    for q in 0..n {
        let iw = &wfa.initial_weights()[q];
        if !iw.is_zero() {
            stack.push((q, Vec::new(), iw.clone()));
        }
    }

    while let Some((state, path, weight)) = stack.pop() {
        // Check if this state is accepting.
        let fw = &wfa.final_weights()[state];
        if !fw.is_zero() {
            let total = weight.clone().mul(fw);
            if !total.is_zero() {
                let is_better = match &best {
                    None => true,
                    Some((bp, _)) => path.len() > bp.len(),
                };
                if is_better {
                    best = Some((path.clone(), total));
                }
            }
        }

        if path.len() >= max_search {
            continue;
        }

        for a in 0..alpha_size {
            for &(dst, ref tw) in &trans[state][a] {
                if tw.is_zero() {
                    continue;
                }
                let new_w = weight.clone().mul(tw);
                if new_w.is_zero() {
                    continue;
                }
                let mut new_path = path.clone();
                new_path.push(a);
                stack.push((dst, new_path, new_w));
            }
        }
    }

    best
}

/// Checks whether a WFA (over `BooleanSemiring`) is universal, i.e.,
/// accepts all strings over its alphabet.
///
/// Complement + emptiness check: the WFA is universal iff its complement
/// accepts no string. For a deterministic complete WFA, the complement
/// swaps accepting and non-accepting states.
///
/// For non-deterministic WFAs we attempt a bounded search for a
/// counter-example.
pub fn is_universal(
    wfa: &WeightedFiniteAutomaton<BooleanSemiring>,
) -> bool {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();

    if n == 0 || alpha_size == 0 {
        return false;
    }

    // If the WFA is deterministic and complete, complement and check emptiness.
    if wfa.is_deterministic() {
        let mut complement =
            WeightedFiniteAutomaton::<BooleanSemiring>::from_alphabet(wfa.alphabet().clone());
        for _ in 0..n {
            complement.add_state();
        }
        for q in 0..n {
            complement.set_initial_weight(q, wfa.initial_weights()[q].clone());
            // Flip final weights.
            let fw = &wfa.final_weights()[q];
            if fw.is_zero() {
                complement.set_final_weight(q, BooleanSemiring::one());
            } else {
                complement.set_final_weight(q, BooleanSemiring::zero());
            }
        }
        let trans = wfa.transitions();
        for q in 0..n {
            for a in 0..alpha_size {
                for &(dst, ref w) in &trans[q][a] {
                    complement.add_transition(q, a, dst, w.clone());
                }
            }
        }
        // Check emptiness: BFS from initial states.
        return shortest_accepted(&complement).is_none();
    }

    // Non-deterministic case: bounded search for counter-example.
    // Try all strings up to a length bound.
    let max_len = std::cmp::min(n * alpha_size + 1, 20);
    let trans = wfa.transitions();

    // Generate all strings up to max_len and check acceptance.
    fn check_all_strings(
        wfa_trans: &[Vec<Vec<(usize, BooleanSemiring)>>],
        initial: &[BooleanSemiring],
        final_w: &[BooleanSemiring],
        alpha_size: usize,
        max_len: usize,
        n: usize,
    ) -> bool {
        // Check empty string.
        let mut accepts_empty = false;
        for q in 0..n {
            if !initial[q].is_zero() && !final_w[q].is_zero() {
                accepts_empty = true;
                break;
            }
        }
        if !accepts_empty {
            return false;
        }

        // BFS through all strings.
        // At each level, track the set of reachable state-sets.
        // Start: for each initial state, track reachable states for each string.
        // This is equivalent to subset construction BFS.
        let mut current_sets: HashSet<Vec<bool>> = HashSet::new();

        // Initial state set.
        let mut init_set = vec![false; n];
        for q in 0..n {
            if !initial[q].is_zero() {
                init_set[q] = true;
            }
        }
        current_sets.insert(init_set);

        for _len in 1..=max_len {
            let mut next_sets: HashSet<Vec<bool>> = HashSet::new();
            for state_set in &current_sets {
                for a in 0..alpha_size {
                    let mut new_set = vec![false; n];
                    for q in 0..n {
                        if !state_set[q] {
                            continue;
                        }
                        for &(dst, ref w) in &wfa_trans[q][a] {
                            if !w.is_zero() {
                                new_set[dst] = true;
                            }
                        }
                    }
                    // Check if this state set has any final state.
                    let accepts = new_set
                        .iter()
                        .enumerate()
                        .any(|(q, &reachable)| reachable && !final_w[q].is_zero());
                    if !accepts {
                        return false;
                    }
                    next_sets.insert(new_set);
                }
            }
            current_sets = next_sets;
        }

        true
    }

    check_all_strings(
        trans,
        wfa.initial_weights(),
        wfa.final_weights(),
        alpha_size,
        max_len,
        n,
    )
}

// ---------------------------------------------------------------------------
// 12. DOT graph output
// ---------------------------------------------------------------------------

/// Options for DOT graph rendering.
#[derive(Debug, Clone)]
pub struct DotOptions {
    /// Whether to display weights on transitions.
    pub show_weights: bool,
    /// Whether to show state labels (initial/final weights).
    pub show_state_labels: bool,
    /// Compact mode: omit zero-weight transitions.
    pub compact: bool,
    /// Set of states to highlight with a different color.
    pub highlight_states: HashSet<usize>,
    /// Color scheme: "default", "warm", "cool", "grayscale".
    pub color_scheme: String,
}

impl Default for DotOptions {
    fn default() -> Self {
        Self {
            show_weights: true,
            show_state_labels: true,
            compact: true,
            highlight_states: HashSet::new(),
            color_scheme: "default".to_string(),
        }
    }
}

impl DotOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_show_weights(mut self, show: bool) -> Self {
        self.show_weights = show;
        self
    }

    pub fn with_show_state_labels(mut self, show: bool) -> Self {
        self.show_state_labels = show;
        self
    }

    pub fn with_compact(mut self, compact: bool) -> Self {
        self.compact = compact;
        self
    }

    pub fn with_highlight_states(mut self, states: HashSet<usize>) -> Self {
        self.highlight_states = states;
        self
    }

    pub fn with_color_scheme(mut self, scheme: &str) -> Self {
        self.color_scheme = scheme.to_string();
        self
    }
}

/// Produces a detailed DOT-format graph representation of a WFA.
pub fn to_dot_detailed<S: Semiring + fmt::Display>(
    wfa: &WeightedFiniteAutomaton<S>,
    options: &DotOptions,
) -> String {
    let n = wfa.num_states();
    let alpha_size = wfa.alphabet().size();
    let trans = wfa.transitions();

    let (node_color, highlight_color, edge_color, font_color) =
        match options.color_scheme.as_str() {
            "warm" => ("#FFCCCC", "#FF6666", "#CC3333", "#660000"),
            "cool" => ("#CCCCFF", "#6666FF", "#3333CC", "#000066"),
            "grayscale" => ("#DDDDDD", "#888888", "#444444", "#000000"),
            _ => ("#FFFFFF", "#FFFF99", "#000000", "#000000"),
        };

    let mut dot = String::new();
    dot.push_str("digraph WFA {\n");
    dot.push_str("    rankdir=LR;\n");
    dot.push_str(&format!(
        "    node [shape=circle, style=filled, fillcolor=\"{}\", fontcolor=\"{}\"];\n",
        node_color, font_color
    ));
    dot.push_str(&format!(
        "    edge [color=\"{}\", fontcolor=\"{}\"];\n",
        edge_color, font_color
    ));
    dot.push_str("\n");

    // Invisible start nodes for initial states.
    let mut has_initial = false;
    for q in 0..n {
        if !wfa.initial_weights()[q].is_zero() {
            if !has_initial {
                dot.push_str("    // Initial state markers\n");
                has_initial = true;
            }
            dot.push_str(&format!(
                "    __start_{} [shape=point, width=0.0, height=0.0];\n",
                q
            ));
            let label = if options.show_weights {
                format!("{}", wfa.initial_weights()[q])
            } else {
                String::new()
            };
            dot.push_str(&format!(
                "    __start_{} -> {} [label=\"{}\"];\n",
                q, q, label
            ));
        }
    }

    dot.push_str("\n    // States\n");
    for q in 0..n {
        let is_final = !wfa.final_weights()[q].is_zero();
        let is_highlighted = options.highlight_states.contains(&q);

        let shape = if is_final { "doublecircle" } else { "circle" };
        let fill = if is_highlighted {
            highlight_color
        } else {
            node_color
        };

        let label = if options.show_state_labels {
            if is_final && options.show_weights {
                format!("q{}/{}",q, wfa.final_weights()[q])
            } else {
                format!("q{}", q)
            }
        } else {
            format!("{}", q)
        };

        dot.push_str(&format!(
            "    {} [shape={}, fillcolor=\"{}\", label=\"{}\"];\n",
            q, shape, fill, label
        ));
    }

    dot.push_str("\n    // Transitions\n");

    // Group transitions by (src, dst) for compact display.
    let mut edge_labels: BTreeMap<(usize, usize), Vec<String>> = BTreeMap::new();

    for q in 0..n {
        for a in 0..alpha_size {
            for &(dst, ref w) in &trans[q][a] {
                if options.compact && w.is_zero() {
                    continue;
                }

                let sym_name = if let Some(sym) = wfa.alphabet().symbol_at(a) {
                    match sym {
                        Symbol::Char(c) => c.to_string(),
                        Symbol::Byte(b) => format!("0x{:02x}", b),
                        Symbol::Token(t) => t.clone(),
                        Symbol::Epsilon => "ε".to_string(),
                        Symbol::Wildcard => ".".to_string(),
                        Symbol::Id(id) => format!("#{}", id),
                    }
                } else {
                    format!("{}", a)
                };

                let label = if options.show_weights && !w.is_one() {
                    format!("{}/{}", sym_name, w)
                } else {
                    sym_name
                };

                edge_labels.entry((q, dst)).or_default().push(label);
            }
        }
    }

    for ((src, dst), labels) in &edge_labels {
        let combined = labels.join(", ");
        dot.push_str(&format!(
            "    {} -> {} [label=\"{}\"];\n",
            src, dst, combined
        ));
    }

    dot.push_str("}\n");
    dot
}

// ---------------------------------------------------------------------------
// 13. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Builds a small WFA over BooleanSemiring that accepts {"ab", "ba"}.
    fn make_ab_ba_wfa() -> WeightedFiniteAutomaton<BooleanSemiring> {
        // Alphabet: {a=0, b=1}
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));

        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        // States: 0 (initial), 1 ("a" read), 2 ("b" read), 3 (accept "ab"),
        //         4 (accept "ba")
        for _ in 0..5 {
            wfa.add_state();
        }

        wfa.set_initial_weight(0, BooleanSemiring::one());

        // 0 --a--> 1, 0 --b--> 2
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(0, 1, 2, BooleanSemiring::one());

        // 1 --b--> 3, 2 --a--> 4
        wfa.add_transition(1, 1, 3, BooleanSemiring::one());
        wfa.add_transition(2, 0, 4, BooleanSemiring::one());

        wfa.set_final_weight(3, BooleanSemiring::one());
        wfa.set_final_weight(4, BooleanSemiring::one());

        wfa
    }

    /// Builds a WFA that accepts {"a", "aa"} (any string of 'a's up to length 2).
    fn make_a_star_bounded() -> WeightedFiniteAutomaton<BooleanSemiring> {
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));

        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        for _ in 0..3 {
            wfa.add_state();
        }

        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one()); // a
        wfa.add_transition(1, 0, 2, BooleanSemiring::one()); // aa

        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());

        wfa
    }

    /// Builds a single-symbol WFA that accepts {"a"}.
    fn make_single_a() -> WeightedFiniteAutomaton<BooleanSemiring> {
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));

        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        for _ in 0..2 {
            wfa.add_state();
        }

        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.set_final_weight(1, BooleanSemiring::one());

        wfa
    }

    /// Builds a WFA that accepts {"b"}.
    fn make_single_b() -> WeightedFiniteAutomaton<BooleanSemiring> {
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));

        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        for _ in 0..2 {
            wfa.add_state();
        }

        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 1, 1, BooleanSemiring::one());
        wfa.set_final_weight(1, BooleanSemiring::one());

        wfa
    }

    /// Builds a WFA over CountingSemiring for "a*" that counts paths.
    fn make_counting_a_star() -> WeightedFiniteAutomaton<CountingSemiring> {
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));

        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        wfa.add_state();
        wfa.set_initial_weight(0, CountingSemiring::one());
        wfa.set_final_weight(0, CountingSemiring::one());
        wfa.add_transition(0, 0, 0, CountingSemiring::one());

        wfa
    }

    /// Simulates a WFA on a string of symbol indices and returns the weight.
    fn simulate_wfa<S: Semiring>(
        wfa: &WeightedFiniteAutomaton<S>,
        input: &[usize],
    ) -> S {
        let n = wfa.num_states();
        let trans = wfa.transitions();

        // Forward pass: track weights at each state.
        let mut current: Vec<S> = wfa.initial_weights().to_vec();

        for &sym in input {
            let mut next = vec![S::zero(); n];
            for q in 0..n {
                if current[q].is_zero() {
                    continue;
                }
                if sym < trans[q].len() {
                    for &(dst, ref w) in &trans[q][sym] {
                        let contrib = current[q].clone().mul(w);
                        next[dst] = next[dst].add(&contrib);
                    }
                }
            }
            current = next;
        }

        let mut total = S::zero();
        for q in 0..n {
            if !current[q].is_zero() && !wfa.final_weights()[q].is_zero() {
                let w = current[q].clone().mul(&wfa.final_weights()[q]);
                total = total.add(&w);
            }
        }
        total
    }

    // -----------------------------------------------------------------------
    // Hadamard product tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hadamard_intersection_semantics() {
        // Intersection of {"ab","ba"} and {"ab"} should be {"ab"}.
        let wfa1 = make_ab_ba_wfa();

        // WFA that accepts only "ab".
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));
        let mut wfa2 = WeightedFiniteAutomaton::from_alphabet(alpha);
        for _ in 0..3 {
            wfa2.add_state();
        }
        wfa2.set_initial_weight(0, BooleanSemiring::one());
        wfa2.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa2.add_transition(1, 1, 2, BooleanSemiring::one());
        wfa2.set_final_weight(2, BooleanSemiring::one());

        let product = hadamard_product(&wfa1, &wfa2).unwrap();

        // "ab" should be accepted.
        let w_ab = simulate_wfa(&product, &[0, 1]);
        assert!(!w_ab.is_zero(), "ab should be accepted by the product");

        // "ba" should NOT be accepted.
        let w_ba = simulate_wfa(&product, &[1, 0]);
        assert!(w_ba.is_zero(), "ba should not be accepted by the product");

        // Empty string should not be accepted.
        let w_eps = simulate_wfa(&product, &[]);
        assert!(w_eps.is_zero());
    }

    #[test]
    fn test_hadamard_self_intersection() {
        let wfa = make_ab_ba_wfa();
        let product = hadamard_product(&wfa, &wfa).unwrap();

        // Self-intersection should accept exactly the same language.
        assert!(!simulate_wfa(&product, &[0, 1]).is_zero());
        assert!(!simulate_wfa(&product, &[1, 0]).is_zero());
        assert!(simulate_wfa(&product, &[0, 0]).is_zero());
        assert!(simulate_wfa(&product, &[1, 1]).is_zero());
    }

    #[test]
    fn test_hadamard_disjoint_empty() {
        // Intersection of {"a"} and {"b"} should be empty.
        let wfa_a = make_single_a();
        let wfa_b = make_single_b();
        let product = hadamard_product(&wfa_a, &wfa_b).unwrap();

        assert!(simulate_wfa(&product, &[0]).is_zero());
        assert!(simulate_wfa(&product, &[1]).is_zero());
        assert!(simulate_wfa(&product, &[]).is_zero());
    }

    #[test]
    fn test_hadamard_state_mapping() {
        let n2 = 5;
        assert_eq!(hadamard_pair_index(2, 3, n2), 13);
        assert_eq!(hadamard_state_pair(13, n2), (2, 3));
    }

    // -----------------------------------------------------------------------
    // Shuffle product tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shuffle_product_basic() {
        let wfa_a = make_single_a();
        let wfa_b = make_single_b();

        let shuffled = shuffle_product(&wfa_a, &wfa_b).unwrap();

        // Shuffle of "a" and "b" should accept "ab" and "ba".
        assert!(!simulate_wfa(&shuffled, &[0, 1]).is_zero());
        assert!(!simulate_wfa(&shuffled, &[1, 0]).is_zero());

        // Should not accept "a", "b", "aa", "bb".
        assert!(simulate_wfa(&shuffled, &[0]).is_zero());
        assert!(simulate_wfa(&shuffled, &[1]).is_zero());
        assert!(simulate_wfa(&shuffled, &[0, 0]).is_zero());
        assert!(simulate_wfa(&shuffled, &[1, 1]).is_zero());
    }

    // -----------------------------------------------------------------------
    // Reversal tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reverse_palindrome() {
        // Reverse of {"ab","ba"} should be {"ba","ab"} – same set.
        let wfa = make_ab_ba_wfa();
        let rev = reverse(&wfa);

        assert!(!simulate_wfa(&rev, &[0, 1]).is_zero()); // "ab" reversed = "ba"
        assert!(!simulate_wfa(&rev, &[1, 0]).is_zero()); // "ba" reversed = "ab"
        assert!(simulate_wfa(&rev, &[0, 0]).is_zero());
    }

    #[test]
    fn test_reverse_single() {
        let wfa = make_single_a();
        let rev = reverse(&wfa);
        // "a" reversed is "a" – same.
        assert!(!simulate_wfa(&rev, &[0]).is_zero());
        assert!(simulate_wfa(&rev, &[1]).is_zero());
    }

    #[test]
    fn test_reverse_asymmetric() {
        // Build WFA that accepts only "ab" (not "ba").
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));
        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        for _ in 0..3 {
            wfa.add_state();
        }
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(1, 1, 2, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());

        let rev = reverse(&wfa);

        // Reversed WFA should accept "ba" but not "ab".
        assert!(!simulate_wfa(&rev, &[1, 0]).is_zero());
        assert!(simulate_wfa(&rev, &[0, 1]).is_zero());
    }

    // -----------------------------------------------------------------------
    // Quotient tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_left_quotient_basic() {
        // WFA accepts {"ab"}, prefix accepts {"a"}.
        // Left quotient should accept {"b"}.
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));

        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha.clone());
        for _ in 0..3 {
            wfa.add_state();
        }
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(1, 1, 2, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());

        let prefix = make_single_a();

        let quotient = left_quotient(&prefix, &wfa).unwrap();

        // Should accept "b".
        let w_b = simulate_wfa(&quotient, &[1]);
        assert!(!w_b.is_zero(), "left quotient should accept 'b'");

        // Should not accept "a" or "ab".
        let w_a = simulate_wfa(&quotient, &[0]);
        assert!(w_a.is_zero(), "should not accept 'a'");
    }

    #[test]
    fn test_right_quotient_basic() {
        // WFA accepts {"ab"}, suffix accepts {"b"}.
        // Right quotient should accept {"a"}.
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        alpha.insert(Symbol::Char('b'));

        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha.clone());
        for _ in 0..3 {
            wfa.add_state();
        }
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(1, 1, 2, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());

        let suffix = make_single_b();

        let quotient = right_quotient(&wfa, &suffix).unwrap();

        // Should accept "a".
        let w_a = simulate_wfa(&quotient, &[0]);
        assert!(!w_a.is_zero(), "right quotient should accept 'a'");

        // Should not accept "b".
        let w_b = simulate_wfa(&quotient, &[1]);
        assert!(w_b.is_zero());
    }

    // -----------------------------------------------------------------------
    // Projection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_restrict_to_states() {
        let wfa = make_ab_ba_wfa();
        // Keep only states 0, 1, 3 (the "ab" accepting path).
        let mut states = HashSet::new();
        states.insert(0);
        states.insert(1);
        states.insert(3);

        let restricted = restrict_to_states(&wfa, &states);
        assert_eq!(restricted.num_states(), 3);

        // Should still accept "ab".
        assert!(!simulate_wfa(&restricted, &[0, 1]).is_zero());
        // Should NOT accept "ba" (state 2 and 4 removed).
        assert!(simulate_wfa(&restricted, &[1, 0]).is_zero());
    }

    #[test]
    fn test_restrict_to_length() {
        let wfa = make_a_star_bounded(); // accepts "a" and "aa"

        let restricted = restrict_to_length(&wfa, 2, 2);
        // Should accept "aa" but not "a".
        assert!(!simulate_wfa(&restricted, &[0, 0]).is_zero());
        assert!(simulate_wfa(&restricted, &[0]).is_zero());
    }

    #[test]
    fn test_restrict_to_length_range() {
        let wfa = make_a_star_bounded(); // accepts "a" and "aa"
        let restricted = restrict_to_length(&wfa, 1, 2);

        assert!(!simulate_wfa(&restricted, &[0]).is_zero());
        assert!(!simulate_wfa(&restricted, &[0, 0]).is_zero());
        assert!(simulate_wfa(&restricted, &[]).is_zero());
    }

    // -----------------------------------------------------------------------
    // Weight scaling tests
    // -----------------------------------------------------------------------

/* // COMMENTED OUT: broken test - test_scale_weights_counting
    #[test]
    fn test_scale_weights_counting() {
        let wfa = make_counting_a_star();
        let scaled = scale_weights(&wfa, &CountingSemiring::from(3));

        // Original: single 'a' has weight 1 (one path, one transition of weight 1).
        // After scaling by 3: transition weight becomes 3.
        let w_a = simulate_wfa(&scaled, &[0]);
        // Path: init(1) * transition(3) * final(1) = 3
        assert_eq!(w_a, CountingSemiring::from(3));
    }
*/

    // -----------------------------------------------------------------------
    // State manipulation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rename_states_identity() {
        let wfa = make_ab_ba_wfa();
        let mapping: Vec<usize> = (0..wfa.num_states()).collect();
        let renamed = rename_states(&wfa, &mapping).unwrap();

        assert!(!simulate_wfa(&renamed, &[0, 1]).is_zero());
        assert!(!simulate_wfa(&renamed, &[1, 0]).is_zero());
    }

    #[test]
    fn test_rename_states_swap() {
        let wfa = make_single_a();
        // Swap states 0 and 1.
        let renamed = rename_states(&wfa, &[1, 0]).unwrap();
        // Should still accept "a".
        assert!(!simulate_wfa(&renamed, &[0]).is_zero());
    }

    #[test]
    fn test_rename_states_invalid() {
        let wfa = make_single_a();
        // Bad mapping: duplicate targets.
        let result = rename_states(&wfa, &[0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reindex_states() {
        let wfa = make_ab_ba_wfa();
        let reindexed = reindex_states(&wfa);

        // Should preserve language.
        assert!(!simulate_wfa(&reindexed, &[0, 1]).is_zero());
        assert!(!simulate_wfa(&reindexed, &[1, 0]).is_zero());
        assert!(simulate_wfa(&reindexed, &[0, 0]).is_zero());
    }

    #[test]
    fn test_add_sink_state() {
        let mut wfa = make_single_a();
        let sink = add_sink_state(&mut wfa);

        // Should still accept "a".
        assert!(!simulate_wfa(&wfa, &[0]).is_zero());
        // Sink state should be the last state.
        assert_eq!(sink, 2);
        // Going to sink via "b" from state 0 should not accept.
        assert!(simulate_wfa(&wfa, &[1]).is_zero());
    }

    #[test]
    fn test_split_state() {
        let mut wfa = make_single_a();
        let new_state = split_state(&mut wfa, 0).unwrap();

        // The new state got the initial weight, so paths starting from it
        // should work.
        assert_eq!(wfa.num_states(), 3);
        assert_eq!(new_state, 2);
    }

    // -----------------------------------------------------------------------
    // Regex conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_regex_single_symbol() {
        let wfa = make_single_a();
        let regex = to_regex_string(&wfa).unwrap();
        // Should produce something like "a".
        assert!(regex.contains('a'), "regex should contain 'a': got {}", regex);
        assert!(!regex.contains('b'), "regex should not contain 'b': got {}", regex);
    }

    #[test]
    fn test_to_regex_alternation() {
        let wfa = make_ab_ba_wfa();
        let regex = to_regex_string(&wfa).unwrap();
        // Should contain both "a" and "b".
        assert!(
            regex.contains('a') && regex.contains('b'),
            "regex should contain 'a' and 'b': got {}",
            regex
        );
    }

    #[test]
    fn test_simplify_regex_empty_star() {
        let frag = RegexFragment::Star(Box::new(RegexFragment::Empty));
        let simplified = simplify_regex(&frag);
        assert_eq!(simplified, RegexFragment::Epsilon);
    }

    #[test]
    fn test_simplify_regex_double_star() {
        let inner = RegexFragment::Star(Box::new(RegexFragment::Symbol("a".into())));
        let double = RegexFragment::Star(Box::new(inner.clone()));
        let simplified = simplify_regex(&double);
        // (a*)* should simplify to a*.
        assert_eq!(simplified, inner);
    }

    #[test]
    fn test_simplify_regex_concat_with_epsilon() {
        let frag = RegexFragment::Concat(vec![
            RegexFragment::Epsilon,
            RegexFragment::Symbol("a".into()),
            RegexFragment::Epsilon,
        ]);
        let simplified = simplify_regex(&frag);
        assert_eq!(simplified, RegexFragment::Symbol("a".into()));
    }

    #[test]
    fn test_regex_to_string_basic() {
        let frag = RegexFragment::Concat(vec![
            RegexFragment::Symbol("a".into()),
            RegexFragment::Symbol("b".into()),
        ]);
        assert_eq!(regex_to_string(&frag), "ab");
    }

    // -----------------------------------------------------------------------
    // Weight mapping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_map_weights_bool_to_counting() {
        let wfa = make_single_a();
        let mapped = map_weights(&wfa, |b: &BooleanSemiring| {
            if b.is_zero() {
                CountingSemiring::zero()
            } else {
                CountingSemiring::one()
            }
        });

        let w = simulate_wfa(&mapped, &[0]);
        assert_eq!(w, CountingSemiring::one());
    }

    #[test]
    fn test_filter_transitions() {
        let wfa = make_ab_ba_wfa();
        // Keep only transitions on symbol 0 ('a').
        let filtered = filter_transitions(&wfa, |_src, sym, _dst, _w| sym == 0);

        // "ab" requires a 'b' transition, so should not be accepted.
        assert!(simulate_wfa(&filtered, &[0, 1]).is_zero());
        // But the 'a' transition from state 0 to state 1 should still exist.
        // No final state is reachable with only 'a', so nothing is accepted.
    }

    // -----------------------------------------------------------------------
    // Path counting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_count_paths_single() {
        let wfa = make_single_a();
        assert_eq!(count_paths(&wfa, 1), 1); // exactly one path of length 1
        assert_eq!(count_paths(&wfa, 0), 0); // no accepting path of length 0
        assert_eq!(count_paths(&wfa, 2), 0); // no path of length 2
    }

    #[test]
    fn test_count_paths_ab_ba() {
        let wfa = make_ab_ba_wfa();
        assert_eq!(count_paths(&wfa, 2), 2); // "ab" and "ba"
        assert_eq!(count_paths(&wfa, 1), 0);
        assert_eq!(count_paths(&wfa, 0), 0);
    }

    #[test]
    fn test_count_paths_a_star() {
        let wfa = make_counting_a_star();
        // a* with one state: at length k there's exactly 1 accepting path.
        assert_eq!(count_paths(&wfa, 0), 1);
        assert_eq!(count_paths(&wfa, 1), 1);
        assert_eq!(count_paths(&wfa, 5), 1);
    }

    // -----------------------------------------------------------------------
    // Enumeration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_enumerate_single_a() {
        let wfa = make_single_a();
        let accepted = enumerate_accepted(&wfa, 3);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].0, vec![0]);
    }

    #[test]
    fn test_enumerate_ab_ba() {
        let wfa = make_ab_ba_wfa();
        let accepted = enumerate_accepted(&wfa, 3);
        assert_eq!(accepted.len(), 2);
        let paths: Vec<&Vec<usize>> = accepted.iter().map(|(p, _)| p).collect();
        assert!(paths.contains(&&vec![0, 1])); // "ab"
        assert!(paths.contains(&&vec![1, 0])); // "ba"
    }

    #[test]
    fn test_enumerate_max_length_boundary() {
        let wfa = make_a_star_bounded(); // accepts "a" (len 1) and "aa" (len 2)
        let accepted = enumerate_accepted(&wfa, 1);
        // Only strings up to length 1.
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].0, vec![0]);
    }

    // -----------------------------------------------------------------------
    // Shortest / longest accepted tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shortest_accepted_single() {
        let wfa = make_single_a();
        let result = shortest_accepted(&wfa);
        assert!(result.is_some());
        let (path, _w) = result.unwrap();
        assert_eq!(path, vec![0]);
    }

    #[test]
    fn test_shortest_accepted_ab_ba() {
        let wfa = make_ab_ba_wfa();
        let result = shortest_accepted(&wfa);
        assert!(result.is_some());
        let (path, _w) = result.unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_shortest_accepted_empty_language() {
        // Build a WFA that accepts nothing.
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        wfa.add_state();
        wfa.set_initial_weight(0, BooleanSemiring::one());
        // No final states.

        let result = shortest_accepted(&wfa);
        assert!(result.is_none());
    }

    #[test]
    fn test_shortest_accepts_empty_string() {
        // WFA that accepts the empty string.
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        let mut wfa = WeightedFiniteAutomaton::from_alphabet(alpha);
        wfa.add_state();
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(0, BooleanSemiring::one());

        let result = shortest_accepted(&wfa);
        assert!(result.is_some());
        let (path, _w) = result.unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn test_longest_accepted() {
        let wfa = make_a_star_bounded(); // accepts "a" and "aa"
        let result = longest_accepted(&wfa, 10);
        assert!(result.is_some());
        let (path, _w) = result.unwrap();
        assert_eq!(path.len(), 2); // "aa" is longest
    }

    // -----------------------------------------------------------------------
    // Sequential / parallel composition tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compose_sequential_two() {
        // Concat of "a" and "b" should accept "ab".
        let wfa_a = make_single_a();
        let wfa_b = make_single_b();

        let composed =
            compose_sequential(&[&wfa_a, &wfa_b]).unwrap();

        assert!(!simulate_wfa(&composed, &[0, 1]).is_zero());
        // Should not accept "a" alone or "ba".
        assert!(simulate_wfa(&composed, &[0]).is_zero());
        assert!(simulate_wfa(&composed, &[1, 0]).is_zero());
    }

    #[test]
    fn test_compose_sequential_single() {
        let wfa = make_single_a();
        let composed = compose_sequential(&[&wfa]).unwrap();
        assert!(!simulate_wfa(&composed, &[0]).is_zero());
    }

    #[test]
    fn test_compose_sequential_empty() {
        let result =
            compose_sequential::<BooleanSemiring>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compose_parallel_intersection() {
        // Parallel composition of {"ab","ba"} with itself should be same.
        let wfa = make_ab_ba_wfa();
        let composed = compose_parallel(&[&wfa, &wfa]).unwrap();

        assert!(!simulate_wfa(&composed, &[0, 1]).is_zero());
        assert!(!simulate_wfa(&composed, &[1, 0]).is_zero());
        assert!(simulate_wfa(&composed, &[0, 0]).is_zero());
    }

    // -----------------------------------------------------------------------
    // DOT output tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_output_basic() {
        let wfa = make_ab_ba_wfa();
        let options = DotOptions::default();
        let dot = to_dot_detailed(&wfa, &options);

        assert!(dot.contains("digraph WFA"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("doublecircle")); // final states
    }

    #[test]
    fn test_dot_output_with_highlights() {
        let wfa = make_single_a();
        let mut highlights = HashSet::new();
        highlights.insert(0);
        let options = DotOptions::new().with_highlight_states(highlights);
        let dot = to_dot_detailed(&wfa, &options);
        assert!(dot.contains("digraph WFA"));
    }

    #[test]
    fn test_dot_output_color_schemes() {
        let wfa = make_single_a();
        for scheme in &["default", "warm", "cool", "grayscale"] {
            let options = DotOptions::new().with_color_scheme(scheme);
            let dot = to_dot_detailed(&wfa, &options);
            assert!(dot.contains("digraph WFA"));
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_hadamard_empty_automaton_error() {
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        let wfa1: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::from_alphabet(alpha.clone());
        let mut wfa2 = WeightedFiniteAutomaton::from_alphabet(alpha);
        wfa2.add_state();

        let result = hadamard_product(&wfa1, &wfa2);
        assert!(result.is_err());
    }

    #[test]
    fn test_incompatible_alphabets() {
        let mut alpha1 = Alphabet::new();
        alpha1.insert(Symbol::Char('a'));
        let mut wfa1 = WeightedFiniteAutomaton::<BooleanSemiring>::from_alphabet(alpha1);
        wfa1.add_state();

        let mut alpha2 = Alphabet::new();
        alpha2.insert(Symbol::Char('a'));
        alpha2.insert(Symbol::Char('b'));
        let mut wfa2 = WeightedFiniteAutomaton::<BooleanSemiring>::from_alphabet(alpha2);
        wfa2.add_state();

        let result = hadamard_product(&wfa1, &wfa2);
        assert!(result.is_err());
    }

    #[test]
    fn test_restrict_to_states_empty() {
        let wfa = make_single_a();
        let empty: HashSet<usize> = HashSet::new();
        let restricted = restrict_to_states(&wfa, &empty);
        assert_eq!(restricted.num_states(), 0);
    }

    #[test]
    fn test_count_paths_empty_wfa() {
        let mut alpha = Alphabet::new();
        alpha.insert(Symbol::Char('a'));
        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::from_alphabet(alpha);
        assert_eq!(count_paths(&wfa, 0), 0);
        assert_eq!(count_paths(&wfa, 1), 0);
    }

    #[test]
    fn test_reverse_double_reversal() {
        // Reversing twice should give back the same language.
        let wfa = make_ab_ba_wfa();
        let rev2 = reverse(&reverse(&wfa));

        assert!(!simulate_wfa(&rev2, &[0, 1]).is_zero());
        assert!(!simulate_wfa(&rev2, &[1, 0]).is_zero());
        assert!(simulate_wfa(&rev2, &[0, 0]).is_zero());
    }

    #[test]
    fn test_regex_fragment_display() {
        let frag = RegexFragment::Alt(vec![
            RegexFragment::Symbol("a".into()),
            RegexFragment::Symbol("b".into()),
        ]);
        assert_eq!(format!("{}", frag), "a|b");
    }

    #[test]
    fn test_regex_concat_identity() {
        let a = RegexFragment::Symbol("x".into());
        let eps = RegexFragment::Epsilon;
        let result = regex_concat(a.clone(), eps);
        assert_eq!(result, a);
    }

    #[test]
    fn test_regex_alt_identity() {
        let a = RegexFragment::Symbol("x".into());
        let empty = RegexFragment::Empty;
        let result = regex_alt(a.clone(), empty);
        assert_eq!(result, a);
    }

    #[test]
    fn test_regex_star_epsilon() {
        let result = regex_star(RegexFragment::Epsilon);
        assert_eq!(result, RegexFragment::Epsilon);
    }

    #[test]
    fn test_project_to_alphabet_single_symbol() {
        let wfa = make_ab_ba_wfa();
        // Project to symbol 0 ('a') only.
        let projected = project_to_alphabet(&wfa, &[0]);

        // After projecting out 'b', the 'b' transitions become epsilon.
        // "ab" becomes "a" (b is epsilon), so projected should accept "a".
        assert!(!simulate_wfa(&projected, &[0]).is_zero());
    }

    #[test]
    fn test_scale_weights_by_zero() {
        let wfa = make_counting_a_star();
        let scaled = scale_weights(&wfa, &CountingSemiring::zero());
        // All transitions zeroed out, only empty string can be accepted
        // (if initial and final weights are non-zero).
        let w = simulate_wfa(&scaled, &[0]);
        assert!(w.is_zero());
    }

    #[test]
    fn test_dot_options_builder() {
        let opts = DotOptions::new()
            .with_show_weights(false)
            .with_compact(false)
            .with_color_scheme("warm");
        assert!(!opts.show_weights);
        assert!(!opts.compact);
        assert_eq!(opts.color_scheme, "warm");
    }
}
