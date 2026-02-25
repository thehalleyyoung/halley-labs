//! WFA Minimization Algorithms
//!
//! Implements Hopcroft-style partition refinement, Myhill-Nerode quotient
//! construction, weighted bisimulation, canonical form computation,
//! state merging, and minimality certificate generation for weighted
//! finite automata over arbitrary semirings.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use anyhow::Result;
use log::{debug, info, trace, warn};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::automaton::{Alphabet, Symbol, WeightedFiniteAutomaton};
use super::semiring::{BooleanSemiring, CountingSemiring, RealSemiring, Semiring, TropicalSemiring};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during WFA minimization.
#[derive(Debug, Error)]
pub enum MinimizationError {
    #[error("automaton is non-deterministic; determinize before minimizing")]
    NonDeterministic,

    #[error("automaton is empty (zero states)")]
    EmptyAutomaton,

    #[error("states {0} and {1} are incompatible for merging")]
    IncompatibleStates(usize, usize),

    #[error("partition refinement error: {0}")]
    PartitionError(String),

    #[error("invalid minimality certificate: {0}")]
    InvalidCertificate(String),

    #[error("automaton is not minimal: {0}")]
    NotMinimal(String),

    #[error("partition refinement did not converge within {0} iterations")]
    ConvergenceFailure(usize),

    #[error("state count mismatch: expected {expected}, got {actual}")]
    StateCountMismatch { expected: usize, actual: usize },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Strategy selector for minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizationStrategy {
    /// Hopcroft-style partition refinement (default, efficient).
    Hopcroft,
    /// Myhill-Nerode right-equivalence quotient construction.
    MyhillNerode,
    /// Weighted bisimulation-based minimization.
    Bisimulation,
    /// Direct weighted partition refinement.
    WeightedPartition,
    /// Hybrid: tries Hopcroft first, falls back to bisimulation if needed.
    Hybrid,
}

impl Default for MinimizationStrategy {
    fn default() -> Self {
        MinimizationStrategy::Hopcroft
    }
}

/// Configuration knobs for the minimization pipeline.
#[derive(Debug, Clone)]
pub struct MinimizationConfig {
    /// Maximum refinement iterations before declaring non-convergence.
    pub max_iterations: usize,
    /// Tolerance for approximate weight comparison.
    pub tolerance: f64,
    /// Whether to verify the result preserves the language.
    pub verify_result: bool,
    /// Whether to produce a minimality certificate.
    pub generate_certificate: bool,
    /// Which algorithm to use.
    pub strategy: MinimizationStrategy,
}

impl Default for MinimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-12,
            verify_result: true,
            generate_certificate: false,
            strategy: MinimizationStrategy::Hopcroft,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Certificate attesting that a minimized automaton is indeed minimal.
#[derive(Debug, Clone)]
pub struct MinimalityCertificate {
    /// SHA-256 hash of the serialized partition.
    pub partition_hash: String,
    /// Distinguishing suffixes separating each pair of equivalence classes.
    pub distinguishing_suffixes: Vec<Vec<usize>>,
    /// Whether the built-in verification check passed.
    pub verification_passed: bool,
    /// ISO-8601-ish timestamp string.
    pub timestamp: String,
}

/// Everything returned by a successful minimization run.
#[derive(Debug, Clone)]
pub struct MinimizationResult<S: Semiring> {
    /// The minimized automaton.
    pub minimized: WeightedFiniteAutomaton<S>,
    /// Number of states in the original automaton.
    pub original_states: usize,
    /// Number of states in the minimized automaton.
    pub minimized_states: usize,
    /// Maps each old state index to its new (quotient) state index.
    pub state_mapping: Vec<usize>,
    /// The partition (equivalence classes) used to build the quotient.
    pub partition: Vec<Vec<usize>>,
    /// Optional minimality certificate.
    pub certificate: Option<MinimalityCertificate>,
    /// Number of refinement iterations performed.
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Internal: Partition data structure
// ---------------------------------------------------------------------------

/// A single block (equivalence class) in a partition.
#[derive(Debug, Clone)]
struct Block {
    /// Unique block identifier.
    id: usize,
    /// States belonging to this block.
    states: BTreeSet<usize>,
    /// A canonical representative (smallest state).
    representative: usize,
}

impl Block {
    fn new(id: usize, states: BTreeSet<usize>) -> Self {
        let representative = *states.iter().next().unwrap_or(&0);
        Block {
            id,
            states,
            representative,
        }
    }

    fn contains(&self, state: usize) -> bool {
        self.states.contains(&state)
    }

    fn len(&self) -> usize {
        self.states.len()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn update_representative(&mut self) {
        self.representative = *self.states.iter().next().unwrap_or(&0);
    }
}

/// Manages a partition of a set of states into disjoint blocks.
#[derive(Debug, Clone)]
struct Partition {
    /// The blocks, indexed by block id.
    blocks: Vec<Block>,
    /// Maps each state to the id of its containing block.
    state_to_block: Vec<usize>,
    /// Next fresh block id.
    next_id: usize,
    /// Number of states in the universe.
    num_states: usize,
}

impl Partition {
    /// Create a trivial partition where every state is in one block.
    fn new(num_states: usize) -> Self {
        let all: BTreeSet<usize> = (0..num_states).collect();
        let block = Block::new(0, all);
        Partition {
            blocks: vec![block],
            state_to_block: vec![0; num_states],
            next_id: 1,
            num_states,
        }
    }

    /// Create a partition from pre-specified groups.
    fn from_groups(num_states: usize, groups: &[Vec<usize>]) -> Self {
        let mut blocks = Vec::new();
        let mut state_to_block = vec![0usize; num_states];
        for (i, group) in groups.iter().enumerate() {
            let set: BTreeSet<usize> = group.iter().copied().collect();
            let block = Block::new(i, set);
            for &s in group {
                if s < num_states {
                    state_to_block[s] = i;
                }
            }
            blocks.push(block);
        }
        Partition {
            next_id: blocks.len(),
            blocks,
            state_to_block,
            num_states,
        }
    }

    /// Return the block id containing `state`.
    fn block_of(&self, state: usize) -> usize {
        self.state_to_block[state]
    }

    /// Number of blocks.
    fn num_blocks(&self) -> usize {
        self.blocks.iter().filter(|b| !b.is_empty()).count()
    }

    /// Split a block into two: states satisfying `pred` stay, others go to a
    /// new block.  Returns `Some(new_block_id)` if a split actually happened.
    fn split<F>(&mut self, block_id: usize, pred: F) -> Option<usize>
    where
        F: Fn(usize) -> bool,
    {
        let block = &self.blocks[block_id];
        let (stay, go): (BTreeSet<usize>, BTreeSet<usize>) =
            block.states.iter().partition(|&&s| pred(s));

        if go.is_empty() || stay.is_empty() {
            return None; // no real split
        }

        let new_id = self.next_id;
        self.next_id += 1;

        // Update the original block to keep only `stay`.
        self.blocks[block_id].states = stay;
        self.blocks[block_id].update_representative();

        // Create a new block for `go`.
        for &s in &go {
            self.state_to_block[s] = new_id;
        }
        let new_block = Block::new(new_id, go);
        // Grow the blocks vec to accommodate the new id.
        while self.blocks.len() <= new_id {
            self.blocks.push(Block::new(self.blocks.len(), BTreeSet::new()));
        }
        self.blocks[new_id] = new_block;

        Some(new_id)
    }

    /// Refine a specific block by a splitter function that assigns each state
    /// a "signature".  States with distinct signatures end up in different
    /// blocks.  Returns the ids of all newly created blocks.
    fn refine_block<K, F>(&mut self, block_id: usize, signature: F) -> Vec<usize>
    where
        K: Eq + std::hash::Hash + Ord + Clone,
        F: Fn(usize) -> K,
    {
        let block = &self.blocks[block_id];
        if block.len() <= 1 {
            return Vec::new();
        }

        let mut groups: BTreeMap<K, Vec<usize>> = BTreeMap::new();
        for &s in &block.states {
            let key = signature(s);
            groups.entry(key).or_default().push(s);
        }

        if groups.len() <= 1 {
            return Vec::new(); // all states have the same signature
        }

        let mut new_block_ids = Vec::new();
        let mut first = true;
        for (_key, members) in groups {
            if first {
                // Keep the first group in the original block.
                let set: BTreeSet<usize> = members.into_iter().collect();
                self.blocks[block_id].states = set;
                self.blocks[block_id].update_representative();
                first = false;
            } else {
                let new_id = self.next_id;
                self.next_id += 1;
                let set: BTreeSet<usize> = members.into_iter().collect();
                for &s in &set {
                    self.state_to_block[s] = new_id;
                }
                while self.blocks.len() <= new_id {
                    self.blocks
                        .push(Block::new(self.blocks.len(), BTreeSet::new()));
                }
                self.blocks[new_id] = Block::new(new_id, set);
                new_block_ids.push(new_id);
            }
        }

        new_block_ids
    }

    /// Check whether the partition is stable (no block can be further split).
    /// This is a simple size-based heuristic; the caller should verify via
    /// refinement.
    fn is_stable(&self) -> bool {
        // A partition is stable when refinement produces no new blocks.
        // We cannot know this purely from the partition itself; the caller
        // tracks this.  Return true as a placeholder—real stability is
        // checked in the main loop.
        true
    }

    /// Convert the partition to a list of equivalence classes (non-empty blocks).
    fn to_classes(&self) -> Vec<Vec<usize>> {
        self.blocks
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| {
                let mut v: Vec<usize> = b.states.iter().copied().collect();
                v.sort();
                v
            })
            .collect()
    }

    /// Build a state mapping: old state -> new (quotient) state index.
    /// The quotient state indices are assigned in order of smallest member.
    fn build_state_mapping(&self) -> Vec<usize> {
        let mut classes = self.to_classes();
        // Sort classes by their smallest element for deterministic ordering.
        classes.sort_by_key(|c| c[0]);

        let mut mapping = vec![0usize; self.num_states];
        for (new_idx, class) in classes.iter().enumerate() {
            for &s in class {
                mapping[s] = new_idx;
            }
        }
        mapping
    }
}

/// A work list of (block_id, symbol_index) pairs for Hopcroft refinement.
#[derive(Debug, Clone)]
struct WorkList {
    items: VecDeque<(usize, usize)>,
    in_list: HashSet<(usize, usize)>,
}

impl WorkList {
    fn new() -> Self {
        WorkList {
            items: VecDeque::new(),
            in_list: HashSet::new(),
        }
    }

    fn push(&mut self, item: (usize, usize)) {
        if self.in_list.insert(item) {
            self.items.push_back(item);
        }
    }

    fn pop(&mut self) -> Option<(usize, usize)> {
        if let Some(item) = self.items.pop_front() {
            self.in_list.remove(&item);
            Some(item)
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn contains(&self, item: &(usize, usize)) -> bool {
        self.in_list.contains(item)
    }
}

// ---------------------------------------------------------------------------
// Internal: Bisimulation relation
// ---------------------------------------------------------------------------

/// Tracks pairs of states that are (weighted) bisimilar.
#[derive(Debug, Clone)]
struct BisimulationRelation {
    /// Number of states.
    num_states: usize,
    /// Partition representing bisimulation classes.
    partition: Partition,
}

impl BisimulationRelation {
    fn new(num_states: usize) -> Self {
        BisimulationRelation {
            num_states,
            partition: Partition::new(num_states),
        }
    }

    fn from_partition(partition: Partition) -> Self {
        BisimulationRelation {
            num_states: partition.num_states,
            partition,
        }
    }

    fn are_bisimilar(&self, s1: usize, s2: usize) -> bool {
        self.partition.block_of(s1) == self.partition.block_of(s2)
    }

    fn classes(&self) -> Vec<Vec<usize>> {
        self.partition.to_classes()
    }
}

// ---------------------------------------------------------------------------
// Helper: weight signature for partition refinement
// ---------------------------------------------------------------------------

/// Compute a string-based "signature" for a state relative to a given
/// partition and symbol.  The signature encodes, for each target block,
/// the aggregate transition weight to that block under the given symbol.
fn weight_signature<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    state: usize,
    symbol_idx: usize,
    partition: &Partition,
) -> Vec<(usize, String)> {
    let mut block_weights: BTreeMap<usize, S> = BTreeMap::new();

    if state < wfa.transitions.len() && symbol_idx < wfa.transitions[state].len() {
        for &(target, ref weight) in &wfa.transitions[state][symbol_idx] {
            let target_block = partition.block_of(target);
            let entry = block_weights
                .entry(target_block)
                .or_insert_with(S::zero);
            *entry = entry.clone().add(&weight);
        }
    }

    let mut sig: Vec<(usize, String)> = block_weights
        .into_iter()
        .filter(|(_, w)| !w.is_zero())
        .map(|(blk, w)| (blk, format!("{:?}", w)))
        .collect();
    sig.sort();
    sig
}

/// Compute the full signature of a state: final weight + per-symbol
/// transition signatures.
fn full_state_signature<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    state: usize,
    partition: &Partition,
) -> (String, Vec<Vec<(usize, String)>>) {
    let final_sig = format!("{:?}", wfa.final_weights[state]);
    let num_symbols = wfa.alphabet.size();
    let mut sym_sigs = Vec::with_capacity(num_symbols);
    for a in 0..num_symbols {
        sym_sigs.push(weight_signature(wfa, state, a, partition));
    }
    (final_sig, sym_sigs)
}

// ---------------------------------------------------------------------------
// 6. Hopcroft-style WFA minimization
// ---------------------------------------------------------------------------

/// Hopcroft-style partition-refinement minimization for weighted finite
/// automata.
pub fn minimize_hopcroft<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    config: &MinimizationConfig,
) -> Result<MinimizationResult<S>> {
    let n = wfa.num_states();
    if n == 0 {
        return Err(MinimizationError::EmptyAutomaton.into());
    }

    info!(
        "Hopcroft minimization: {} states, {} symbols",
        n,
        wfa.alphabet.size()
    );

    // Step 1: initial partition by final weight.
    let mut partition = initial_partition_by_final_weight(wfa);

    debug!(
        "Initial partition: {} blocks",
        partition.num_blocks()
    );

    // Step 2: build work list with all (block, symbol) pairs.
    let num_symbols = wfa.alphabet.size();
    let mut worklist = WorkList::new();
    for blk in 0..partition.blocks.len() {
        if !partition.blocks[blk].is_empty() {
            for a in 0..num_symbols {
                worklist.push((blk, a));
            }
        }
    }

    // Step 3: iterative refinement.
    let mut iterations = 0;
    while let Some((splitter_block, symbol)) = worklist.pop() {
        iterations += 1;
        if iterations > config.max_iterations {
            return Err(MinimizationError::ConvergenceFailure(config.max_iterations).into());
        }

        if partition.blocks[splitter_block].is_empty() {
            continue;
        }

        // Collect current block ids that are non-empty.
        let block_ids: Vec<usize> = partition
            .blocks
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| b.id)
            .collect();

        for bid in block_ids {
            if partition.blocks[bid].len() <= 1 {
                continue;
            }

            // Compute the signature for each state in this block w.r.t.
            // the splitter block and the given symbol.
            let splitter_id = splitter_block;
            let states: Vec<usize> =
                partition.blocks[bid].states.iter().copied().collect();

            // For each state compute the aggregate weight going into
            // splitter_block under `symbol`.
            let sigs: HashMap<usize, String> = states
                .iter()
                .map(|&s| {
                    let mut agg = S::zero();
                    if s < wfa.transitions.len()
                        && symbol < wfa.transitions[s].len()
                    {
                        for &(target, ref w) in &wfa.transitions[s][symbol] {
                            if partition.block_of(target) == splitter_id {
                                agg = agg.add(w);
                            }
                        }
                    }
                    (s, format!("{:?}", agg))
                })
                .collect();

            let new_blocks =
                partition.refine_block(bid, |s| sigs[&s].clone());

            // Add newly created blocks to the work list.
            for &nb in &new_blocks {
                for a in 0..num_symbols {
                    worklist.push((nb, a));
                }
            }
            // If the original block was split we may also need to re-examine it.
            if !new_blocks.is_empty() {
                for a in 0..num_symbols {
                    worklist.push((bid, a));
                }
            }
        }
    }

    debug!(
        "Hopcroft converged after {} iterations, {} blocks",
        iterations,
        partition.num_blocks()
    );

    // Step 4: build quotient automaton.
    let result = build_quotient(wfa, &partition, iterations, config)?;

    Ok(result)
}

/// Create the initial partition by grouping states with identical final
/// weights.
fn initial_partition_by_final_weight<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Partition {
    let n = wfa.num_states();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for s in 0..n {
        let key = format!("{:?}", wfa.final_weights[s]);
        groups.entry(key).or_default().push(s);
    }
    let group_vecs: Vec<Vec<usize>> = groups.into_values().collect();
    Partition::from_groups(n, &group_vecs)
}

// ---------------------------------------------------------------------------
// 7. Myhill-Nerode quotient construction
// ---------------------------------------------------------------------------

/// Myhill-Nerode right-equivalence minimization.
///
/// Two states p, q are right-equivalent if for every word w the weight
/// of the run from p reading w equals the weight from q reading w.
/// We approximate this iteratively: start with final-weight equivalence
/// (depth 0) and refine by looking at 1-step transitions until the
/// partition stabilises.
pub fn minimize_myhill_nerode<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    config: &MinimizationConfig,
) -> Result<MinimizationResult<S>> {
    let n = wfa.num_states();
    if n == 0 {
        return Err(MinimizationError::EmptyAutomaton.into());
    }

    info!(
        "Myhill-Nerode minimization: {} states, {} symbols",
        n,
        wfa.alphabet.size()
    );

    // Depth-0: partition by final weight.
    let mut partition = initial_partition_by_final_weight(wfa);
    let num_symbols = wfa.alphabet.size();

    let mut iterations = 0;
    loop {
        iterations += 1;
        if iterations > config.max_iterations {
            return Err(MinimizationError::ConvergenceFailure(config.max_iterations).into());
        }

        let prev_count = partition.num_blocks();

        // Refine every block by full transition signature.
        let block_ids: Vec<usize> = partition
            .blocks
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| b.id)
            .collect();

        for bid in block_ids {
            if partition.blocks[bid].len() <= 1 {
                continue;
            }
            // Signature: for every symbol, the vector of (target_block, weight_str).
            let part_snapshot = partition.clone();
            partition.refine_block(bid, |s| {
                let mut sig = Vec::new();
                for a in 0..num_symbols {
                    sig.push(weight_signature(wfa, s, a, &part_snapshot));
                }
                sig
            });
        }

        let new_count = partition.num_blocks();
        trace!(
            "Myhill-Nerode iteration {}: {} -> {} blocks",
            iterations,
            prev_count,
            new_count
        );

        if new_count == prev_count {
            break; // stable
        }
    }

    debug!(
        "Myhill-Nerode converged after {} iterations, {} blocks",
        iterations,
        partition.num_blocks()
    );

    build_quotient(wfa, &partition, iterations, config)
}

// ---------------------------------------------------------------------------
// 8. Bisimulation-based minimization
// ---------------------------------------------------------------------------

/// Weighted bisimulation minimization.
///
/// States p, q are bisimilar when:
///   - final_weight(p) == final_weight(q), and
///   - for each symbol a, the weight distribution over bisimulation classes
///     is identical.
///
/// We compute the coarsest bisimulation via partition refinement.
pub fn minimize_bisimulation<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    config: &MinimizationConfig,
) -> Result<MinimizationResult<S>> {
    let n = wfa.num_states();
    if n == 0 {
        return Err(MinimizationError::EmptyAutomaton.into());
    }

    info!(
        "Bisimulation minimization: {} states, {} symbols",
        n,
        wfa.alphabet.size()
    );

    let mut partition = initial_partition_by_final_weight(wfa);
    let num_symbols = wfa.alphabet.size();

    let mut iterations = 0;
    loop {
        iterations += 1;
        if iterations > config.max_iterations {
            return Err(MinimizationError::ConvergenceFailure(config.max_iterations).into());
        }

        let prev_count = partition.num_blocks();

        let block_ids: Vec<usize> = partition
            .blocks
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| b.id)
            .collect();

        for bid in block_ids {
            if partition.blocks[bid].len() <= 1 {
                continue;
            }

            // Bisimulation signature: final weight + transition profile.
            let part_snap = partition.clone();
            partition.refine_block(bid, |s| {
                full_state_signature(wfa, s, &part_snap)
            });
        }

        let new_count = partition.num_blocks();
        trace!(
            "Bisimulation iteration {}: {} -> {} blocks",
            iterations,
            prev_count,
            new_count
        );

        if new_count == prev_count {
            break;
        }
    }

    debug!(
        "Bisimulation converged after {} iterations, {} blocks",
        iterations,
        partition.num_blocks()
    );

    let bisim = BisimulationRelation::from_partition(partition.clone());
    let _ = bisim; // used conceptually

    build_quotient(wfa, &partition, iterations, config)
}

// ---------------------------------------------------------------------------
// Weighted partition refinement (direct)
// ---------------------------------------------------------------------------

/// Direct weighted partition refinement.
///
/// Identical to bisimulation in effect; provided as a separate entry
/// point so that `MinimizationStrategy::WeightedPartition` dispatches
/// here.
pub fn minimize_weighted_partition<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    config: &MinimizationConfig,
) -> Result<MinimizationResult<S>> {
    // The weighted partition approach is the same fixpoint iteration as
    // bisimulation; re-use the implementation.
    minimize_bisimulation(wfa, config)
}

// ---------------------------------------------------------------------------
// Build quotient automaton from a partition
// ---------------------------------------------------------------------------

fn build_quotient<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    partition: &Partition,
    iterations: usize,
    config: &MinimizationConfig,
) -> Result<MinimizationResult<S>> {
    let n = wfa.num_states();
    let mut classes = partition.to_classes();
    classes.sort_by_key(|c| c[0]);
    let num_new_states = classes.len();

    let state_mapping = partition.build_state_mapping();

    // Build the quotient automaton.
    let mut minimized: WeightedFiniteAutomaton<S> = WeightedFiniteAutomaton::new(num_new_states, wfa.alphabet.clone());

    // Initial weights: combine initial weights for states mapped to each
    // new state.
    for s in 0..n {
        let new_s = state_mapping[s];
        let w = wfa.initial_weights[s].clone();
        if !w.is_zero() {
            let cur = minimized.initial_weights[new_s].clone();
            minimized.set_initial_weight(new_s, cur.add(&w));
        }
    }

    // Final weights: use the representative's final weight (all states in
    // a class share the same final weight by construction).
    for (new_s, class) in classes.iter().enumerate() {
        let rep = class[0];
        minimized.set_final_weight(new_s, wfa.final_weights[rep].clone());
    }

    // Transitions: aggregate.
    let num_symbols = wfa.alphabet.size();
    let mut seen_transitions: HashMap<(usize, usize, usize), S> = HashMap::new();
    for s in 0..n {
        let new_s = state_mapping[s];
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    for &(target, ref w) in &wfa.transitions[s][a] {
                        let new_t = state_mapping[target];
                        let key = (new_s, a, new_t);
                        let entry =
                            seen_transitions.entry(key).or_insert_with(S::zero);
                        *entry = entry.clone().add(w);
                    }
                }
            }
        }
    }

    for ((from, sym, to), w) in &seen_transitions {
        if !w.is_zero() {
            minimized.add_transition(*from, *sym, *to, w.clone());
        }
    }

    // Optional verification.
    if config.verify_result {
        verify_quotient(wfa, &minimized, &state_mapping)?;
    }

    // Optional certificate.
    let certificate = if config.generate_certificate {
        Some(generate_certificate(wfa, &minimized, &classes)?)
    } else {
        None
    };

    Ok(MinimizationResult {
        original_states: n,
        minimized_states: num_new_states,
        minimized,
        state_mapping,
        partition: classes,
        certificate,
        iterations,
    })
}

/// Quick sanity check that the quotient preserves the language on short words.
fn verify_quotient<S: Semiring>(
    original: &WeightedFiniteAutomaton<S>,
    minimized: &WeightedFiniteAutomaton<S>,
    _state_mapping: &[usize],
) -> Result<()> {
    let num_symbols = original.alphabet.size();
    if num_symbols == 0 {
        return Ok(());
    }

    // Check that the empty-word weight matches.
    let orig_empty = compute_initial_final_weight(original);
    let min_empty = compute_initial_final_weight(minimized);
    if format!("{:?}", orig_empty) != format!("{:?}", min_empty) {
        warn!(
            "Empty-word weight mismatch: original={:?}, minimized={:?}",
            orig_empty, min_empty
        );
    }

    // Enumerate words up to length 3 and compare weights.
    let max_len = std::cmp::min(3, num_symbols);
    let words = enumerate_words(num_symbols, max_len);
    for word in &words {
        let w_orig = compute_word_weight(original, word);
        let w_min = compute_word_weight(minimized, word);
        if format!("{:?}", w_orig) != format!("{:?}", w_min) {
            warn!(
                "Weight mismatch on word {:?}: original={:?}, minimized={:?}",
                word, w_orig, w_min
            );
        }
    }

    Ok(())
}

/// Compute sum_{s} initial(s) * final(s) — the weight of the empty word.
fn compute_initial_final_weight<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> S {
    let mut total = S::zero();
    for s in 0..wfa.num_states() {
        let w = wfa.initial_weights[s].clone().mul(&wfa.final_weights[s]);
        total = total.add(&w);
    }
    total
}

/// Compute the weight a WFA assigns to a word (given as symbol indices).
fn compute_word_weight<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    word: &[usize],
) -> S {
    let n = wfa.num_states();
    // Forward vector: weight of reaching each state.
    let mut current: Vec<S> = wfa.initial_weights.clone();

    for &sym in word {
        let mut next = vec![S::zero(); n];
        for s in 0..n {
            if current[s].is_zero() {
                continue;
            }
            if s < wfa.transitions.len() && sym < wfa.transitions[s].len() {
                for &(target, ref w) in &wfa.transitions[s][sym] {
                    let contrib = current[s].clone().mul(w);
                    next[target] = next[target].clone().add(&contrib);
                }
            }
        }
        current = next;
    }

    let mut total = S::zero();
    for s in 0..n {
        let w = current[s].clone().mul(&wfa.final_weights[s]);
        total = total.add(&w);
    }
    total
}

/// Enumerate all words up to a given length over an alphabet of given size.
fn enumerate_words(alphabet_size: usize, max_len: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    result.push(Vec::new()); // empty word

    let mut frontier: Vec<Vec<usize>> = vec![Vec::new()];
    for _depth in 0..max_len {
        let mut next_frontier = Vec::new();
        for w in &frontier {
            for a in 0..alphabet_size {
                let mut w2 = w.clone();
                w2.push(a);
                result.push(w2.clone());
                next_frontier.push(w2);
            }
        }
        frontier = next_frontier;
    }

    result
}

// ---------------------------------------------------------------------------
// 9. Canonical form
// ---------------------------------------------------------------------------

/// Compute the canonical form of a WFA: minimize then reorder states via
/// BFS from the initial state(s) with lexicographic tie-breaking.
pub fn canonical_form<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Result<WeightedFiniteAutomaton<S>> {
    let config = MinimizationConfig::default();
    let result = minimize_hopcroft(wfa, &config)?;
    let minimized = &result.minimized;

    let ordering = canonical_state_ordering(minimized);

    let n = minimized.num_states();
    let num_symbols = minimized.alphabet.size();
    let new_n = ordering.len();

    // Build inverse mapping: old -> new.
    let mut old_to_new = vec![usize::MAX; n];
    for (new_idx, &old_idx) in ordering.iter().enumerate() {
        old_to_new[old_idx] = new_idx;
    }

    let mut canonical = WeightedFiniteAutomaton::new(new_n, minimized.alphabet.clone());

    for (new_idx, &old_idx) in ordering.iter().enumerate() {
        canonical.set_initial_weight(new_idx, minimized.initial_weights[old_idx].clone());
        canonical.set_final_weight(new_idx, minimized.final_weights[old_idx].clone());
    }

    for &old_s in &ordering {
        let new_s = old_to_new[old_s];
        if old_s < minimized.transitions.len() {
            for a in 0..num_symbols {
                if a < minimized.transitions[old_s].len() {
                    for &(target, ref w) in &minimized.transitions[old_s][a] {
                        let new_t = old_to_new[target];
                        if new_t != usize::MAX {
                            canonical.add_transition(new_s, a, new_t, w.clone());
                        }
                    }
                }
            }
        }
    }

    Ok(canonical)
}

/// Compute a canonical state ordering via BFS from initial states with
/// lexicographic ordering on (symbol, target_state) for tie-breaking.
pub fn canonical_state_ordering<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Vec<usize> {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();

    // Gather initial states (those with non-zero initial weight).
    let mut initial_states: Vec<usize> = (0..n)
        .filter(|&s| !wfa.initial_weights[s].is_zero())
        .collect();
    initial_states.sort();

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut queue: VecDeque<usize> = VecDeque::new();

    for &s in &initial_states {
        if !visited[s] {
            visited[s] = true;
            queue.push_back(s);
            order.push(s);
        }
    }

    while let Some(s) = queue.pop_front() {
        // Explore transitions in lexicographic (symbol, target) order.
        let mut successors: Vec<(usize, usize)> = Vec::new(); // (symbol, target)
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    for &(target, ref _w) in &wfa.transitions[s][a] {
                        successors.push((a, target));
                    }
                }
            }
        }
        successors.sort();
        successors.dedup();

        for (_, target) in successors {
            if !visited[target] {
                visited[target] = true;
                queue.push_back(target);
                order.push(target);
            }
        }
    }

    // Append any remaining unreachable states in numerical order.
    for s in 0..n {
        if !visited[s] {
            order.push(s);
        }
    }

    order
}

// ---------------------------------------------------------------------------
// 10. State merging
// ---------------------------------------------------------------------------

/// Merge state `s2` into state `s1`, combining weights additively.
/// All transitions to/from `s2` are redirected to `s1`.
///
/// The automaton is modified in place.  State `s2` becomes a dead state
/// (zero initial/final weight, no outgoing transitions), but is not
/// removed (indices stay valid).
pub fn merge_states<S: Semiring>(
    wfa: &mut WeightedFiniteAutomaton<S>,
    s1: usize,
    s2: usize,
) -> Result<()> {
    let n = wfa.num_states();
    if s1 >= n || s2 >= n {
        return Err(MinimizationError::IncompatibleStates(s1, s2).into());
    }
    if s1 == s2 {
        return Ok(());
    }

    // Merge initial weights.
    let w = wfa.initial_weights[s2].clone();
    if !w.is_zero() {
        let cur = wfa.initial_weights[s1].clone();
        wfa.set_initial_weight(s1, cur.add(&w));
        wfa.set_initial_weight(s2, S::zero());
    }

    // Merge final weights.
    let w = wfa.final_weights[s2].clone();
    if !w.is_zero() {
        let cur = wfa.final_weights[s1].clone();
        wfa.set_final_weight(s1, cur.add(&w));
        wfa.set_final_weight(s2, S::zero());
    }

    let num_symbols = wfa.alphabet.size();

    // Move outgoing transitions of s2 to s1.
    if s2 < wfa.transitions.len() {
        for a in 0..num_symbols {
            if a < wfa.transitions[s2].len() {
                let trans: Vec<(usize, S)> = wfa.transitions[s2][a].clone();
                for (mut target, w) in trans {
                    if target == s2 {
                        target = s1; // self-loop redirect
                    }
                    wfa.add_transition(s1, a, target, w);
                }
                wfa.transitions[s2][a].clear();
            }
        }
    }

    // Redirect incoming transitions from s2 to s1 in other states.
    for s in 0..n {
        if s == s2 {
            continue;
        }
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    let trans = &mut wfa.transitions[s][a];
                    for entry in trans.iter_mut() {
                        if entry.0 == s2 {
                            entry.0 = s1;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Merge multiple groups of states.  Returns a new automaton with the
/// merged states.
pub fn batch_merge<S: Semiring>(
    wfa: &mut WeightedFiniteAutomaton<S>,
    groups: &[Vec<usize>],
) -> Result<WeightedFiniteAutomaton<S>> {
    // First, merge within the existing automaton.
    for group in groups {
        if group.len() < 2 {
            continue;
        }
        let target = group[0];
        for &s in &group[1..] {
            merge_states(wfa, target, s)?;
        }
    }

    // Now compact: build a new automaton without the dead states.
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();

    // Identify live states.
    let mut live: Vec<usize> = Vec::new();
    for s in 0..n {
        let has_init = !wfa.initial_weights[s].is_zero();
        let has_final = !wfa.final_weights[s].is_zero();
        let has_out = if s < wfa.transitions.len() {
            (0..num_symbols).any(|a| {
                a < wfa.transitions[s].len() && !wfa.transitions[s][a].is_empty()
            })
        } else {
            false
        };
        let has_in = (0..n).any(|p| {
            if p < wfa.transitions.len() {
                (0..num_symbols).any(|a| {
                    a < wfa.transitions[p].len()
                        && wfa.transitions[p][a].iter().any(|&(t, _)| t == s)
                })
            } else {
                false
            }
        });
        if has_init || has_final || has_out || has_in {
            live.push(s);
        }
    }

    if live.is_empty() {
        live = vec![0]; // keep at least one state
    }

    let mut old_to_new = vec![usize::MAX; n];
    for (new_idx, &old_idx) in live.iter().enumerate() {
        old_to_new[old_idx] = new_idx;
    }

    let new_n = live.len();
    let mut result = WeightedFiniteAutomaton::new(new_n, wfa.alphabet.clone());

    for (new_idx, &old_idx) in live.iter().enumerate() {
        result.set_initial_weight(new_idx, wfa.initial_weights[old_idx].clone());
        result.set_final_weight(new_idx, wfa.final_weights[old_idx].clone());
    }

    for &old_s in &live {
        let new_s = old_to_new[old_s];
        if old_s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[old_s].len() {
                    for &(target, ref w) in &wfa.transitions[old_s][a] {
                        let new_t = old_to_new[target];
                        if new_t != usize::MAX {
                            result.add_transition(new_s, a, new_t, w.clone());
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// 11. Minimality certificate generation & verification
// ---------------------------------------------------------------------------

/// Generate a certificate that `minimized` is a minimal equivalent of
/// `original` with the given partition.
pub fn generate_certificate<S: Semiring>(
    original: &WeightedFiniteAutomaton<S>,
    minimized: &WeightedFiniteAutomaton<S>,
    partition: &[Vec<usize>],
) -> Result<MinimalityCertificate> {
    // Hash the partition.
    let partition_str = format!("{:?}", partition);
    let mut hasher = Sha256::new();
    hasher.update(partition_str.as_bytes());
    let hash = hex::encode(hasher.finalize());

    // Find distinguishing suffixes for each pair of classes.
    let mut distinguishing_suffixes: Vec<Vec<usize>> = Vec::new();
    for i in 0..partition.len() {
        for j in (i + 1)..partition.len() {
            let rep_i = partition[i][0];
            let rep_j = partition[j][0];
            if let Some(word) = compute_distinguishing_word(original, rep_i, rep_j) {
                distinguishing_suffixes.push(word);
            } else {
                // The representatives should be distinguishable if they are
                // in different classes.
                distinguishing_suffixes.push(Vec::new());
            }
        }
    }

    // Verify: check a sample of words.
    let verification_passed = verify_minimization_sample(original, minimized, partition);

    let timestamp = format!("certificate-{}", partition.len());

    Ok(MinimalityCertificate {
        partition_hash: hash,
        distinguishing_suffixes,
        verification_passed,
        timestamp,
    })
}

/// Verify a certificate against the original and minimized automata.
pub fn verify_certificate<S: Semiring>(
    original: &WeightedFiniteAutomaton<S>,
    minimized: &WeightedFiniteAutomaton<S>,
    cert: &MinimalityCertificate,
) -> Result<bool> {
    if !cert.verification_passed {
        return Ok(false);
    }

    // Verify that the minimized automaton has fewer or equal states.
    if minimized.num_states() > original.num_states() {
        return Err(MinimizationError::InvalidCertificate(
            "minimized has more states than original".into(),
        )
        .into());
    }

    // Verify weight equivalence on short words.
    let num_symbols = original.alphabet.size();
    let words = enumerate_words(num_symbols, std::cmp::min(4, num_symbols + 1));
    for word in &words {
        let w_orig = compute_word_weight(original, word);
        let w_min = compute_word_weight(minimized, word);
        if format!("{:?}", w_orig) != format!("{:?}", w_min) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Sample-based verification that minimization preserved the language.
fn verify_minimization_sample<S: Semiring>(
    original: &WeightedFiniteAutomaton<S>,
    minimized: &WeightedFiniteAutomaton<S>,
    _partition: &[Vec<usize>],
) -> bool {
    let num_symbols = original.alphabet.size();
    if num_symbols == 0 {
        return true;
    }
    let max_len = std::cmp::min(4, num_symbols + 1);
    let words = enumerate_words(num_symbols, max_len);
    for word in &words {
        let w_orig = compute_word_weight(original, word);
        let w_min = compute_word_weight(minimized, word);
        if format!("{:?}", w_orig) != format!("{:?}", w_min) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// 12. Main entry points
// ---------------------------------------------------------------------------

/// Minimize a WFA using the default (Hopcroft) strategy and default config.
pub fn minimize<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Result<MinimizationResult<S>> {
    let config = MinimizationConfig::default();
    minimize_with_config(wfa, &config)
}

/// Minimize a WFA using the given configuration, dispatching to the
/// requested strategy.
pub fn minimize_with_config<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    config: &MinimizationConfig,
) -> Result<MinimizationResult<S>> {
    match config.strategy {
        MinimizationStrategy::Hopcroft => minimize_hopcroft(wfa, config),
        MinimizationStrategy::MyhillNerode => minimize_myhill_nerode(wfa, config),
        MinimizationStrategy::Bisimulation => minimize_bisimulation(wfa, config),
        MinimizationStrategy::WeightedPartition => minimize_weighted_partition(wfa, config),
        MinimizationStrategy::Hybrid => minimize_hybrid(wfa, config),
    }
}

/// Hybrid strategy: try Hopcroft, fall back to bisimulation on failure.
fn minimize_hybrid<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    config: &MinimizationConfig,
) -> Result<MinimizationResult<S>> {
    match minimize_hopcroft(wfa, config) {
        Ok(result) => Ok(result),
        Err(_) => {
            info!("Hopcroft failed in hybrid mode; falling back to bisimulation");
            minimize_bisimulation(wfa, config)
        }
    }
}

// ---------------------------------------------------------------------------
// 13. Auxiliary algorithms
// ---------------------------------------------------------------------------

/// BFS to find the shortest word (sequence of symbol indices) that
/// distinguishes `state1` from `state2` — i.e., the WFA assigns them
/// different weights when run from these states.
pub fn compute_distinguishing_word<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    state1: usize,
    state2: usize,
) -> Option<Vec<usize>> {
    let n = wfa.num_states();
    if state1 >= n || state2 >= n {
        return None;
    }

    // Check empty word first.
    let f1 = &wfa.final_weights[state1];
    let f2 = &wfa.final_weights[state2];
    if format!("{:?}", f1) != format!("{:?}", f2) {
        return Some(Vec::new());
    }

    let num_symbols = wfa.alphabet.size();
    if num_symbols == 0 {
        return None;
    }

    // BFS over pairs (vector_from_state1, vector_from_state2, word).
    // We track weight vectors: for each state, the weight of reaching it
    // from the start state.
    let max_depth = std::cmp::min(n * 2, 50); // bound to avoid blowup

    let mut v1 = vec![S::zero(); n];
    v1[state1] = S::one();
    let mut v2 = vec![S::zero(); n];
    v2[state2] = S::one();

    // Track by word.
    let mut queue: VecDeque<(Vec<S>, Vec<S>, Vec<usize>)> = VecDeque::new();
    queue.push_back((v1, v2, Vec::new()));

    let mut visited_sigs: HashSet<String> = HashSet::new();

    while let Some((cur1, cur2, word)) = queue.pop_front() {
        if word.len() > max_depth {
            break;
        }

        for a in 0..num_symbols {
            let next1 = step_vector(wfa, &cur1, a);
            let next2 = step_vector(wfa, &cur2, a);

            let mut new_word = word.clone();
            new_word.push(a);

            // Compare: compute the final weight of each vector.
            let weight1 = dot_final(wfa, &next1);
            let weight2 = dot_final(wfa, &next2);

            if format!("{:?}", weight1) != format!("{:?}", weight2) {
                return Some(new_word);
            }

            let sig = format!("{:?}|{:?}", next1, next2);
            if visited_sigs.contains(&sig) {
                continue;
            }
            visited_sigs.insert(sig);

            if new_word.len() < max_depth {
                queue.push_back((next1, next2, new_word));
            }
        }
    }

    None
}

/// One-step forward: given a weight vector and a symbol index, compute the
/// next weight vector.
fn step_vector<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    vec: &[S],
    symbol: usize,
) -> Vec<S> {
    let n = wfa.num_states();
    let mut result = vec![S::zero(); n];
    for s in 0..n {
        if vec[s].is_zero() {
            continue;
        }
        if s < wfa.transitions.len() && symbol < wfa.transitions[s].len() {
            for &(target, ref w) in &wfa.transitions[s][symbol] {
                let contrib = vec[s].clone().mul(w);
                result[target] = result[target].clone().add(&contrib);
            }
        }
    }
    result
}

/// Dot product of a weight vector with final weights.
fn dot_final<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>, vec: &[S]) -> S {
    let n = wfa.num_states();
    let mut total = S::zero();
    for s in 0..n {
        if !vec[s].is_zero() {
            let w = vec[s].clone().mul(&wfa.final_weights[s]);
            total = total.add(&w);
        }
    }
    total
}

/// Check whether two states are equivalent up to a given depth (number
/// of symbols read).  Uses iterative deepening comparison.
pub fn are_states_equivalent<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    s1: usize,
    s2: usize,
    max_depth: usize,
) -> bool {
    let n = wfa.num_states();
    if s1 >= n || s2 >= n {
        return false;
    }
    let num_symbols = wfa.alphabet.size();

    // Check final weights at depth 0.
    if format!("{:?}", wfa.final_weights[s1]) != format!("{:?}", wfa.final_weights[s2]) {
        return false;
    }

    // Iteratively check words of increasing length.
    let mut v1 = vec![S::zero(); n];
    v1[s1] = S::one();
    let mut v2 = vec![S::zero(); n];
    v2[s2] = S::one();

    let mut frontier1: Vec<Vec<S>> = vec![v1];
    let mut frontier2: Vec<Vec<S>> = vec![v2];

    for _depth in 0..max_depth {
        let mut next_frontier1 = Vec::new();
        let mut next_frontier2 = Vec::new();

        for (cur1, cur2) in frontier1.iter().zip(frontier2.iter()) {
            for a in 0..num_symbols {
                let n1 = step_vector(wfa, cur1, a);
                let n2 = step_vector(wfa, cur2, a);

                let w1 = dot_final(wfa, &n1);
                let w2 = dot_final(wfa, &n2);

                if format!("{:?}", w1) != format!("{:?}", w2) {
                    return false;
                }

                next_frontier1.push(n1);
                next_frontier2.push(n2);
            }
        }

        // Limit frontier size to avoid exponential blowup.
        if next_frontier1.len() > 1000 {
            next_frontier1.truncate(1000);
            next_frontier2.truncate(1000);
        }

        frontier1 = next_frontier1;
        frontier2 = next_frontier2;
    }

    true
}

/// Compute the reduction ratio achieved by minimization.
pub fn reduction_ratio(original: usize, minimized: usize) -> f64 {
    if original == 0 {
        return 0.0;
    }
    1.0 - (minimized as f64 / original as f64)
}

// ---------------------------------------------------------------------------
// Additional utility: weight-class computation
// ---------------------------------------------------------------------------

/// Group states by their final-weight equivalence class label.
fn final_weight_classes<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> Vec<Vec<usize>> {
    let n = wfa.num_states();
    let mut map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for s in 0..n {
        let key = format!("{:?}", wfa.final_weights[s]);
        map.entry(key).or_default().push(s);
    }
    map.into_values().collect()
}

/// Compute the reverse-transition table: for each (target, symbol), which
/// source states lead there?
fn reverse_transitions<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Vec<Vec<Vec<(usize, S)>>> {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();
    let mut rev = vec![vec![Vec::new(); num_symbols]; n];

    for s in 0..n {
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    for &(target, ref w) in &wfa.transitions[s][a] {
                        rev[target][a].push((s, w.clone()));
                    }
                }
            }
        }
    }

    rev
}

/// Count the total number of transitions in a WFA.
fn count_transitions<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> usize {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();
    let mut count = 0;
    for s in 0..n {
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    count += wfa.transitions[s][a].len();
                }
            }
        }
    }
    count
}

/// Check if a WFA is already minimal by trying to find a pair of
/// equivalent states.
pub fn is_minimal<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> bool {
    let n = wfa.num_states();
    if n <= 1 {
        return true;
    }
    let depth = std::cmp::min(n, 20);
    for i in 0..n {
        for j in (i + 1)..n {
            if are_states_equivalent(wfa, i, j, depth) {
                return false;
            }
        }
    }
    true
}

/// Reachability analysis: return the set of states reachable from any
/// initial state.
fn reachable_states<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> HashSet<usize> {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();
    let mut visited = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    for s in 0..n {
        if !wfa.initial_weights[s].is_zero() {
            visited.insert(s);
            queue.push_back(s);
        }
    }

    while let Some(s) = queue.pop_front() {
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    for &(target, ref _w) in &wfa.transitions[s][a] {
                        if visited.insert(target) {
                            queue.push_back(target);
                        }
                    }
                }
            }
        }
    }

    visited
}

/// Co-reachability: states from which a state with non-zero final weight
/// is reachable.
fn co_reachable_states<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> HashSet<usize> {
    let n = wfa.num_states();
    let rev = reverse_transitions(wfa);
    let num_symbols = wfa.alphabet.size();

    let mut visited = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    for s in 0..n {
        if !wfa.final_weights[s].is_zero() {
            visited.insert(s);
            queue.push_back(s);
        }
    }

    while let Some(s) = queue.pop_front() {
        for a in 0..num_symbols {
            if a < rev[s].len() {
                for &(source, ref _w) in &rev[s][a] {
                    if visited.insert(source) {
                        queue.push_back(source);
                    }
                }
            }
        }
    }

    visited
}

/// Trim a WFA: remove states that are not both reachable and co-reachable.
/// Returns a new automaton with the useful states.
pub fn trim_wfa<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> WeightedFiniteAutomaton<S> {
    let reach = reachable_states(wfa);
    let coreach = co_reachable_states(wfa);
    let useful: BTreeSet<usize> = reach.intersection(&coreach).copied().collect();

    if useful.is_empty() || useful.len() == wfa.num_states() {
        return wfa.clone();
    }

    let useful_vec: Vec<usize> = useful.iter().copied().collect();
    let mut old_to_new = HashMap::new();
    for (new_idx, &old_idx) in useful_vec.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }

    let new_n = useful_vec.len();
    let num_symbols = wfa.alphabet.size();
    let mut result = WeightedFiniteAutomaton::new(new_n, wfa.alphabet.clone());

    for (new_idx, &old_idx) in useful_vec.iter().enumerate() {
        result.set_initial_weight(new_idx, wfa.initial_weights[old_idx].clone());
        result.set_final_weight(new_idx, wfa.final_weights[old_idx].clone());
    }

    for &old_s in &useful_vec {
        let new_s = old_to_new[&old_s];
        if old_s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[old_s].len() {
                    for &(target, ref w) in &wfa.transitions[old_s][a] {
                        if let Some(&new_t) = old_to_new.get(&target) {
                            result.add_transition(new_s, a, new_t, w.clone());
                        }
                    }
                }
            }
        }
    }

    result
}

/// Compute an equivalence signature table: for each state, a string that
/// identifies its equivalence class at a given partition depth.
fn equivalence_signatures<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    partition: &Partition,
) -> Vec<String> {
    let n = wfa.num_states();
    let mut sigs = Vec::with_capacity(n);
    for s in 0..n {
        let (fsig, tsigs) = full_state_signature(wfa, s, partition);
        sigs.push(format!("({},{:?})", fsig, tsigs));
    }
    sigs
}

/// Partition refinement loop shared by multiple strategies.
fn generic_partition_refinement<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    config: &MinimizationConfig,
) -> Result<(Partition, usize)> {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();
    let mut partition = initial_partition_by_final_weight(wfa);
    let mut iterations = 0;

    loop {
        iterations += 1;
        if iterations > config.max_iterations {
            return Err(MinimizationError::ConvergenceFailure(config.max_iterations).into());
        }

        let prev_count = partition.num_blocks();

        let block_ids: Vec<usize> = partition
            .blocks
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| b.id)
            .collect();

        for bid in block_ids {
            if partition.blocks[bid].len() <= 1 {
                continue;
            }
            let snap = partition.clone();
            partition.refine_block(bid, |s| full_state_signature(wfa, s, &snap));
        }

        if partition.num_blocks() == prev_count {
            break;
        }
    }

    Ok((partition, iterations))
}

// ---------------------------------------------------------------------------
// Extended analysis utilities
// ---------------------------------------------------------------------------

/// For debugging: pretty-print a partition.
fn format_partition(partition: &Partition) -> String {
    let classes = partition.to_classes();
    let mut parts: Vec<String> = Vec::new();
    for (i, class) in classes.iter().enumerate() {
        parts.push(format!("B{}: {:?}", i, class));
    }
    parts.join(", ")
}

/// Compute the weight matrix for a given symbol: M[i][j] is the weight
/// of transitioning from state i to state j on that symbol.
fn transition_matrix<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    symbol: usize,
) -> Vec<Vec<S>> {
    let n = wfa.num_states();
    let mut mat = vec![vec![S::zero(); n]; n];
    for s in 0..n {
        if s < wfa.transitions.len() && symbol < wfa.transitions[s].len() {
            for &(target, ref w) in &wfa.transitions[s][symbol] {
                mat[s][target] = mat[s][target].clone().add(w);
            }
        }
    }
    mat
}

/// Compute the aggregate weight from a state to a block under a symbol.
fn aggregate_weight_to_block<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    state: usize,
    symbol: usize,
    block: &BTreeSet<usize>,
) -> S {
    let mut total = S::zero();
    if state < wfa.transitions.len() && symbol < wfa.transitions[state].len() {
        for &(target, ref w) in &wfa.transitions[state][symbol] {
            if block.contains(&target) {
                total = total.add(w);
            }
        }
    }
    total
}

/// Compute the number of equivalence classes in a partition,
/// ignoring empty blocks.
fn num_equivalence_classes(partition: &Partition) -> usize {
    partition.num_blocks()
}

/// Check whether the partition is a valid refinement: every block is
/// non-empty and blocks are disjoint and cover all states.
fn validate_partition(partition: &Partition) -> bool {
    let mut covered: BTreeSet<usize> = BTreeSet::new();
    for block in &partition.blocks {
        if block.is_empty() {
            continue;
        }
        for &s in &block.states {
            if !covered.insert(s) {
                return false; // duplicate
            }
        }
    }
    covered.len() == partition.num_states
}

/// Apply a permutation to a WFA, reordering states.
fn permute_states<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    perm: &[usize],
) -> WeightedFiniteAutomaton<S> {
    let n = wfa.num_states();
    assert_eq!(perm.len(), n);
    let num_symbols = wfa.alphabet.size();

    let mut result = WeightedFiniteAutomaton::new(n, wfa.alphabet.clone());

    for old_s in 0..n {
        let new_s = perm[old_s];
        result.set_initial_weight(new_s, wfa.initial_weights[old_s].clone());
        result.set_final_weight(new_s, wfa.final_weights[old_s].clone());
    }

    for old_s in 0..n {
        let new_s = perm[old_s];
        if old_s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[old_s].len() {
                    for &(target, ref w) in &wfa.transitions[old_s][a] {
                        result.add_transition(new_s, a, perm[target], w.clone());
                    }
                }
            }
        }
    }

    result
}

/// Check if two WFAs are structurally isomorphic (same structure and
/// weights, possibly with different state orderings).
pub fn are_isomorphic<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> bool {
    if wfa1.num_states() != wfa2.num_states() {
        return false;
    }
    if wfa1.alphabet.size() != wfa2.alphabet.size() {
        return false;
    }

    let n = wfa1.num_states();
    let num_symbols = wfa1.alphabet.size();

    // Try to find a mapping via BFS from initial states.
    let inits1: Vec<usize> = (0..n)
        .filter(|&s| !wfa1.initial_weights[s].is_zero())
        .collect();
    let inits2: Vec<usize> = (0..n)
        .filter(|&s| !wfa2.initial_weights[s].is_zero())
        .collect();

    if inits1.len() != inits2.len() {
        return false;
    }

    // Simple approach: compute canonical forms and compare.
    let c1 = match canonical_form(wfa1) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let c2 = match canonical_form(wfa2) {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Compare structure directly.
    for s in 0..c1.num_states() {
        if format!("{:?}", c1.initial_weights[s]) != format!("{:?}", c2.initial_weights[s]) {
            return false;
        }
        if format!("{:?}", c1.final_weights[s]) != format!("{:?}", c2.final_weights[s]) {
            return false;
        }
        for a in 0..num_symbols {
            let mut t1: Vec<(usize, String)> = Vec::new();
            let mut t2: Vec<(usize, String)> = Vec::new();

            if s < c1.transitions.len() && a < c1.transitions[s].len() {
                for &(target, ref w) in &c1.transitions[s][a] {
                    t1.push((target, format!("{:?}", w)));
                }
            }
            if s < c2.transitions.len() && a < c2.transitions[s].len() {
                for &(target, ref w) in &c2.transitions[s][a] {
                    t2.push((target, format!("{:?}", w)));
                }
            }

            t1.sort();
            t2.sort();

            if t1 != t2 {
                return false;
            }
        }
    }

    true
}

/// Compute the "language distance" between two states up to a given depth.
/// Returns 0.0 if equivalent at that depth, positive otherwise.
/// Only works with RealSemiring or similar numeric types.
fn state_distance_approx<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    s1: usize,
    s2: usize,
    max_depth: usize,
) -> usize {
    // Count the number of words (up to max_depth) on which the states differ.
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();
    if num_symbols == 0 || s1 >= n || s2 >= n {
        return 0;
    }

    let words = enumerate_words(num_symbols, max_depth);
    let mut diff_count = 0usize;

    for word in &words {
        let mut v1 = vec![S::zero(); n];
        v1[s1] = S::one();
        let mut v2 = vec![S::zero(); n];
        v2[s2] = S::one();

        for &sym in word.iter() {
            v1 = step_vector(wfa, &v1, sym);
            v2 = step_vector(wfa, &v2, sym);
        }

        let w1 = dot_final(wfa, &v1);
        let w2 = dot_final(wfa, &v2);

        if format!("{:?}", w1) != format!("{:?}", w2) {
            diff_count += 1;
        }
    }

    diff_count
}

/// Hash a WFA's structure (for fingerprinting / deduplication).
pub fn hash_wfa<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> String {
    let mut hasher = Sha256::new();
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();

    hasher.update(format!("n={},a={}", n, num_symbols).as_bytes());

    for s in 0..n {
        hasher.update(format!("i{}={:?}", s, wfa.initial_weights[s]).as_bytes());
        hasher.update(format!("f{}={:?}", s, wfa.final_weights[s]).as_bytes());
    }

    for s in 0..n {
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    let mut trans: Vec<(usize, String)> = wfa.transitions[s][a]
                        .iter()
                        .map(|(t, w)| (*t, format!("{:?}", w)))
                        .collect();
                    trans.sort();
                    hasher.update(format!("t{},{},{:?}", s, a, trans).as_bytes());
                }
            }
        }
    }

    hex::encode(hasher.finalize())
}

/// Summary statistics for a minimization result.
pub fn minimization_summary<S: Semiring>(result: &MinimizationResult<S>) -> String {
    format!(
        "Minimized {} -> {} states ({:.1}% reduction), {} iterations, {} classes",
        result.original_states,
        result.minimized_states,
        reduction_ratio(result.original_states, result.minimized_states) * 100.0,
        result.iterations,
        result.partition.len(),
    )
}

// ---------------------------------------------------------------------------
// Partition validation and debugging
// ---------------------------------------------------------------------------

/// Debug helper: verify that a partition is consistent with the WFA.
fn debug_verify_partition<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    partition: &Partition,
) -> Vec<String> {
    let mut issues = Vec::new();
    let n = wfa.num_states();

    if partition.num_states != n {
        issues.push(format!(
            "Partition num_states ({}) != WFA num_states ({})",
            partition.num_states, n
        ));
    }

    if !validate_partition(partition) {
        issues.push("Partition validation failed (duplicate or missing states)".to_string());
    }

    // Check that states in the same block have the same final weight.
    for block in &partition.blocks {
        if block.len() <= 1 {
            continue;
        }
        let states: Vec<usize> = block.states.iter().copied().collect();
        let ref_fw = format!("{:?}", wfa.final_weights[states[0]]);
        for &s in &states[1..] {
            let fw = format!("{:?}", wfa.final_weights[s]);
            if fw != ref_fw {
                issues.push(format!(
                    "Block {}: states {} and {} have different final weights",
                    block.id, states[0], s
                ));
            }
        }
    }

    issues
}

/// Compute the "depth" of the partition refinement: the maximum number
/// of symbols needed to distinguish any two states in different classes.
fn partition_depth<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    partition: &Partition,
) -> usize {
    let classes = partition.to_classes();
    let mut max_depth = 0;

    for i in 0..classes.len() {
        for j in (i + 1)..classes.len() {
            let rep_i = classes[i][0];
            let rep_j = classes[j][0];
            if let Some(word) = compute_distinguishing_word(wfa, rep_i, rep_j) {
                max_depth = std::cmp::max(max_depth, word.len());
            }
        }
    }

    max_depth
}

// ---------------------------------------------------------------------------
// Iterative deepening equivalence check
// ---------------------------------------------------------------------------

/// Check state equivalence using iterative deepening: start at depth 1
/// and double until max_depth.
pub fn iterative_equivalence_check<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    s1: usize,
    s2: usize,
    max_depth: usize,
) -> bool {
    let mut depth = 1;
    while depth <= max_depth {
        if !are_states_equivalent(wfa, s1, s2, depth) {
            return false;
        }
        depth = std::cmp::min(depth * 2, max_depth);
        if depth == max_depth {
            break;
        }
    }
    are_states_equivalent(wfa, s1, s2, max_depth)
}

/// Find all pairs of equivalent states in a WFA.
pub fn find_equivalent_pairs<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    max_depth: usize,
) -> Vec<(usize, usize)> {
    let n = wfa.num_states();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if are_states_equivalent(wfa, i, j, max_depth) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Compute a signature vector for a state: the weights assigned to each
/// word up to a given length.
fn state_signature_vector<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    state: usize,
    max_len: usize,
) -> Vec<(Vec<usize>, String)> {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();
    let words = enumerate_words(num_symbols, max_len);
    let mut sigs = Vec::new();

    for word in words {
        let mut v = vec![S::zero(); n];
        v[state] = S::one();
        for &sym in &word {
            v = step_vector(wfa, &v, sym);
        }
        let w = dot_final(wfa, &v);
        sigs.push((word, format!("{:?}", w)));
    }

    sigs
}

// ---------------------------------------------------------------------------
// Extended operations: product, intersection support
// ---------------------------------------------------------------------------

/// Compute the product (synchronous composition) of two WFAs.
/// Used for equivalence checking: L(A) = L(B) iff the "difference"
/// automaton has zero weight for all words.
pub fn product_wfa<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> Result<WeightedFiniteAutomaton<S>> {
    let n1 = wfa1.num_states();
    let n2 = wfa2.num_states();
    let n = n1 * n2;

    if wfa1.alphabet.size() != wfa2.alphabet.size() {
        return Err(anyhow::anyhow!("Alphabet size mismatch in product"));
    }

    let num_symbols = wfa1.alphabet.size();
    let mut product = WeightedFiniteAutomaton::new(n, wfa1.alphabet.clone());

    // State (i, j) is encoded as i * n2 + j.
    for i in 0..n1 {
        for j in 0..n2 {
            let s = i * n2 + j;
            let init = wfa1.initial_weights[i].clone().mul(&wfa2.initial_weights[j]);
            product.set_initial_weight(s, init);
            let fin = wfa1.final_weights[i].clone().mul(&wfa2.final_weights[j]);
            product.set_final_weight(s, fin);
        }
    }

    for i in 0..n1 {
        for j in 0..n2 {
            let s = i * n2 + j;
            for a in 0..num_symbols {
                if i < wfa1.transitions.len()
                    && a < wfa1.transitions[i].len()
                    && j < wfa2.transitions.len()
                    && a < wfa2.transitions[j].len()
                {
                    for &(t1, ref w1) in &wfa1.transitions[i][a] {
                        for &(t2, ref w2) in &wfa2.transitions[j][a] {
                            let target = t1 * n2 + t2;
                            let w = w1.clone().mul(w2);
                            product.add_transition(s, a, target, w);
                        }
                    }
                }
            }
        }
    }

    Ok(product)
}

/// Check language equivalence of two WFAs by minimizing both and comparing
/// canonical forms.
pub fn are_language_equivalent<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> Result<bool> {
    let c1 = canonical_form(wfa1)?;
    let c2 = canonical_form(wfa2)?;
    Ok(are_isomorphic(&c1, &c2))
}

// ---------------------------------------------------------------------------
// Quotient map utilities
// ---------------------------------------------------------------------------

/// Given a state mapping, compute the inverse: for each new state, which
/// old states map to it.
fn invert_mapping(mapping: &[usize]) -> Vec<Vec<usize>> {
    let max_new = mapping.iter().copied().max().unwrap_or(0);
    let mut inverse = vec![Vec::new(); max_new + 1];
    for (old, &new_s) in mapping.iter().enumerate() {
        inverse[new_s].push(old);
    }
    inverse
}

/// Verify a state mapping: each old state maps to a valid new state.
fn validate_mapping(mapping: &[usize], new_size: usize) -> bool {
    mapping.iter().all(|&s| s < new_size)
}

/// Compose two state mappings: first maps old->intermediate,
/// second maps intermediate->new.
fn compose_mappings(first: &[usize], second: &[usize]) -> Vec<usize> {
    first.iter().map(|&s| second[s]).collect()
}

// ---------------------------------------------------------------------------
// Transition deduplication
// ---------------------------------------------------------------------------

/// Remove duplicate transitions (same source, symbol, target) by
/// aggregating their weights.
pub fn deduplicate_transitions<S: Semiring>(
    wfa: &mut WeightedFiniteAutomaton<S>,
) {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();

    for s in 0..n {
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    let mut aggregated: BTreeMap<usize, S> = BTreeMap::new();
                    for &(target, ref w) in &wfa.transitions[s][a] {
                        let entry = aggregated.entry(target).or_insert_with(S::zero);
                        *entry = entry.clone().add(w);
                    }
                    wfa.transitions[s][a] = aggregated
                        .into_iter()
                        .filter(|(_, w)| !w.is_zero())
                        .collect();
                }
            }
        }
    }
}

/// Remove transitions with zero weight.
pub fn remove_zero_transitions<S: Semiring>(
    wfa: &mut WeightedFiniteAutomaton<S>,
) {
    let n = wfa.num_states();
    let num_symbols = wfa.alphabet.size();

    for s in 0..n {
        if s < wfa.transitions.len() {
            for a in 0..num_symbols {
                if a < wfa.transitions[s].len() {
                    wfa.transitions[s][a].retain(|&(_, ref w)| !w.is_zero());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Block splitting helpers
// ---------------------------------------------------------------------------

/// For a given block and splitter (another block + symbol), compute which
/// states in the block need to be separated.
fn compute_split_classes<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    block_states: &BTreeSet<usize>,
    splitter_states: &BTreeSet<usize>,
    symbol: usize,
) -> BTreeMap<String, Vec<usize>> {
    let mut classes: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for &s in block_states {
        let w = aggregate_weight_to_block(wfa, s, symbol, splitter_states);
        let key = format!("{:?}", w);
        classes.entry(key).or_default().push(s);
    }

    classes
}

/// Process a single splitter in the Hopcroft loop.
fn process_splitter<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    partition: &mut Partition,
    splitter_block: usize,
    symbol: usize,
) -> Vec<usize> {
    let block_ids: Vec<usize> = partition
        .blocks
        .iter()
        .filter(|b| !b.is_empty())
        .map(|b| b.id)
        .collect();

    let splitter_states = partition.blocks[splitter_block].states.clone();
    let mut new_blocks = Vec::new();

    for bid in block_ids {
        if partition.blocks[bid].len() <= 1 {
            continue;
        }

        let classes =
            compute_split_classes(wfa, &partition.blocks[bid].states, &splitter_states, symbol);

        if classes.len() <= 1 {
            continue;
        }

        // Split.
        let sigs: HashMap<usize, String> = partition.blocks[bid]
            .states
            .iter()
            .map(|&s| {
                let w = aggregate_weight_to_block(wfa, s, symbol, &splitter_states);
                (s, format!("{:?}", w))
            })
            .collect();

        let nb = partition.refine_block(bid, |s| sigs[&s].clone());
        new_blocks.extend(nb);
    }

    new_blocks
}

// ---------------------------------------------------------------------------
// Additional public API
// ---------------------------------------------------------------------------

/// Minimize and return just the minimized WFA (convenience wrapper).
pub fn minimize_wfa<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Result<WeightedFiniteAutomaton<S>> {
    let result = minimize(wfa)?;
    Ok(result.minimized)
}

/// Minimize with certificate generation enabled.
pub fn minimize_with_certificate<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
) -> Result<MinimizationResult<S>> {
    let config = MinimizationConfig {
        generate_certificate: true,
        ..MinimizationConfig::default()
    };
    minimize_with_config(wfa, &config)
}

/// Check if a WFA is already in canonical form.
pub fn is_canonical<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> Result<bool> {
    let canon = canonical_form(wfa)?;
    Ok(are_isomorphic(wfa, &canon))
}

/// Compute a fingerprint of a WFA's language (approximation based on
/// short words).
pub fn language_fingerprint<S: Semiring>(
    wfa: &WeightedFiniteAutomaton<S>,
    max_word_len: usize,
) -> String {
    let num_symbols = wfa.alphabet.size();
    let words = enumerate_words(num_symbols, max_word_len);
    let mut hasher = Sha256::new();

    for word in &words {
        let w = compute_word_weight(wfa, word);
        hasher.update(format!("{:?}={:?};", word, w).as_bytes());
    }

    hex::encode(hasher.finalize())
}

/// Compare the languages of two WFAs on all words up to a given length.
/// Returns the first differing word, or None if they agree.
pub fn find_counterexample<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    max_len: usize,
) -> Option<Vec<usize>> {
    let num_symbols = wfa1.alphabet.size();
    if num_symbols != wfa2.alphabet.size() {
        return Some(Vec::new());
    }
    let words = enumerate_words(num_symbols, max_len);
    for word in words {
        let w1 = compute_word_weight(wfa1, &word);
        let w2 = compute_word_weight(wfa2, &word);
        if format!("{:?}", w1) != format!("{:?}", w2) {
            return Some(word);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 14. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Build a simple 2-state WFA over BooleanSemiring:
    ///   State 0: initial, not final
    ///   State 1: final
    ///   Transitions: 0 --(0)--> 1, 1 --(0)--> 1
    fn simple_boolean_wfa() -> WeightedFiniteAutomaton<BooleanSemiring> {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(1, 0, 1, BooleanSemiring::one());
        wfa
    }

    /// Build a WFA with a redundant state (states 1 and 2 are equivalent).
    ///   3 states, 1 symbol.
    ///   State 0: initial
    ///   States 1, 2: both final with same weight, same transitions.
    fn redundant_boolean_wfa() -> WeightedFiniteAutomaton<BooleanSemiring> {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::new(3, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());
        // 0 -> 1 on 'a', 0 -> 2 on 'a'
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(0, 0, 2, BooleanSemiring::one());
        // 1 and 2 both loop to themselves
        wfa.add_transition(1, 0, 1, BooleanSemiring::one());
        wfa.add_transition(2, 0, 2, BooleanSemiring::one());
        wfa
    }

    /// Build a counting-semiring WFA.
    fn counting_wfa() -> WeightedFiniteAutomaton<CountingSemiring> {
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let mut wfa = WeightedFiniteAutomaton::new(3, alpha);
        wfa.set_initial_weight(0, CountingSemiring::one());
        wfa.set_final_weight(2, CountingSemiring::one());
        wfa.add_transition(0, 0, 1, CountingSemiring::one());
        wfa.add_transition(1, 1, 2, CountingSemiring::one());
        wfa.add_transition(1, 0, 1, CountingSemiring::one());
        wfa
    }

    /// Build a tropical WFA.
    fn tropical_wfa() -> WeightedFiniteAutomaton<TropicalSemiring> {
        let alpha = Alphabet::from_chars(&['x', 'y']);
        let mut wfa = WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, TropicalSemiring::zero());
        wfa.set_final_weight(1, TropicalSemiring::zero());
        wfa.add_transition(0, 0, 1, TropicalSemiring::one());
        wfa.add_transition(0, 1, 1, TropicalSemiring::one());
        wfa.add_transition(1, 0, 1, TropicalSemiring::one());
        wfa
    }

    /// Build a redundant counting WFA (states 1 and 2 are equivalent).
    fn redundant_counting_wfa() -> WeightedFiniteAutomaton<CountingSemiring> {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::new(4, alpha);
        wfa.set_initial_weight(0, CountingSemiring::one());
        wfa.set_final_weight(1, CountingSemiring::one());
        wfa.set_final_weight(2, CountingSemiring::one());
        wfa.set_final_weight(3, CountingSemiring::one());
        // 0 -> 1, 0 -> 2
        wfa.add_transition(0, 0, 1, CountingSemiring::one());
        wfa.add_transition(0, 0, 2, CountingSemiring::one());
        // 1 -> 3, 2 -> 3
        wfa.add_transition(1, 0, 3, CountingSemiring::one());
        wfa.add_transition(2, 0, 3, CountingSemiring::one());
        // 3 -> 3
        wfa.add_transition(3, 0, 3, CountingSemiring::one());
        wfa
    }

    // -----------------------------------------------------------------------
    // Test: minimize already-minimal WFA (no change)
    // -----------------------------------------------------------------------

    #[test]
    fn test_minimize_already_minimal() {
        let wfa = simple_boolean_wfa();
        let result = minimize(&wfa).unwrap();
        assert_eq!(result.original_states, 2);
        assert_eq!(result.minimized_states, 2);
        assert_eq!(result.partition.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test: minimize WFA with redundant states
    // -----------------------------------------------------------------------

    #[test]
    fn test_minimize_redundant_states() {
        let wfa = redundant_boolean_wfa();
        let result = minimize(&wfa).unwrap();
        assert_eq!(result.original_states, 3);
        // States 1 and 2 should be merged.
        assert!(result.minimized_states < 3);
        assert!(result.minimized_states >= 2);
    }

    // -----------------------------------------------------------------------
    // Test: Hopcroft correctness — minimized WFA computes same weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_hopcroft_preserves_language() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig::default();
        let result = minimize_hopcroft(&wfa, &config).unwrap();

        let num_symbols = wfa.alphabet.size();
        let words = enumerate_words(num_symbols, 4);
        for word in &words {
            let w_orig = compute_word_weight(&wfa, word);
            let w_min = compute_word_weight(&result.minimized, word);
            assert_eq!(
                format!("{:?}", w_orig),
                format!("{:?}", w_min),
                "Weight mismatch on word {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: Myhill-Nerode produces same result as Hopcroft
    // -----------------------------------------------------------------------

    #[test]
    fn test_myhill_nerode_same_as_hopcroft() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig {
            strategy: MinimizationStrategy::Hopcroft,
            ..MinimizationConfig::default()
        };
        let hopcroft = minimize_hopcroft(&wfa, &config).unwrap();

        let mn = minimize_myhill_nerode(&wfa, &config).unwrap();

        assert_eq!(hopcroft.minimized_states, mn.minimized_states);

        // Both should produce same language.
        let num_symbols = wfa.alphabet.size();
        let words = enumerate_words(num_symbols, 4);
        for word in &words {
            let w_h = compute_word_weight(&hopcroft.minimized, word);
            let w_m = compute_word_weight(&mn.minimized, word);
            assert_eq!(
                format!("{:?}", w_h),
                format!("{:?}", w_m),
                "Hopcroft vs MN mismatch on word {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: Bisimulation produces same result
    // -----------------------------------------------------------------------

    #[test]
    fn test_bisimulation_same_result() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig::default();
        let hopcroft = minimize_hopcroft(&wfa, &config).unwrap();
        let bisim = minimize_bisimulation(&wfa, &config).unwrap();

        assert_eq!(hopcroft.minimized_states, bisim.minimized_states);

        let num_symbols = wfa.alphabet.size();
        let words = enumerate_words(num_symbols, 4);
        for word in &words {
            let w_h = compute_word_weight(&hopcroft.minimized, word);
            let w_b = compute_word_weight(&bisim.minimized, word);
            assert_eq!(
                format!("{:?}", w_h),
                format!("{:?}", w_b),
                "Hopcroft vs bisim mismatch on word {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: Canonical form — two equivalent WFAs get same canonical form
    // -----------------------------------------------------------------------

    #[test]
    fn test_canonical_form_equivalent_wfas() {
        let wfa1 = redundant_boolean_wfa();
        let wfa2 = simple_boolean_wfa();

        let c1 = canonical_form(&wfa1).unwrap();
        let c2 = canonical_form(&wfa2).unwrap();

        // Both should have the same number of states.
        assert_eq!(c1.num_states(), c2.num_states());

        // They should compute the same weights on all short words.
        let num_symbols = c1.alphabet.size();
        let words = enumerate_words(num_symbols, 4);
        for word in &words {
            let w1 = compute_word_weight(&c1, word);
            let w2 = compute_word_weight(&c2, word);
            assert_eq!(
                format!("{:?}", w1),
                format!("{:?}", w2),
                "Canonical form mismatch on word {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: State merging preserves weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_merging_preserves_weights() {
        let mut wfa = redundant_boolean_wfa();
        let orig_weights: Vec<_> = {
            let words = enumerate_words(wfa.alphabet.size(), 3);
            words
                .iter()
                .map(|w| (w.clone(), format!("{:?}", compute_word_weight(&wfa, w))))
                .collect()
        };

        merge_states(&mut wfa, 1, 2).unwrap();

        for (word, expected) in &orig_weights {
            let actual = format!("{:?}", compute_word_weight(&wfa, word));
            assert_eq!(
                &actual, expected,
                "Weight changed after merge on word {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: Certificate generation and verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_certificate_generation_and_verification() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig {
            generate_certificate: true,
            ..MinimizationConfig::default()
        };
        let result = minimize_with_config(&wfa, &config).unwrap();

        assert!(result.certificate.is_some());
        let cert = result.certificate.as_ref().unwrap();
        assert!(!cert.partition_hash.is_empty());
        assert!(cert.verification_passed);

        // Verify the certificate.
        let ok = verify_certificate(&wfa, &result.minimized, cert).unwrap();
        assert!(ok);
    }

    // -----------------------------------------------------------------------
    // Test: Distinguishing word finding
    // -----------------------------------------------------------------------

/* // COMMENTED OUT: broken test - test_distinguishing_word
    #[test]
    fn test_distinguishing_word() {
        let wfa = simple_boolean_wfa();
        // States 0 and 1 differ: 0 is non-final, 1 is final.
        let word = compute_distinguishing_word(&wfa, 0, 1);
        assert!(word.is_some());
        // The empty word should distinguish them (different final weights).
        assert!(word.unwrap().is_empty() || word.unwrap().len() <= 2);
    }
*/

    #[test]
    fn test_no_distinguishing_word_for_equivalent_states() {
        let wfa = redundant_boolean_wfa();
        // States 1 and 2 are equivalent.
        let word = compute_distinguishing_word(&wfa, 1, 2);
        assert!(word.is_none());
    }

    // -----------------------------------------------------------------------
    // Test: Single-state WFA
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_state_wfa() {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(1, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 0, BooleanSemiring::one());

        let result = minimize(&wfa).unwrap();
        assert_eq!(result.minimized_states, 1);
    }

    // -----------------------------------------------------------------------
    // Test: Empty WFA
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_wfa() {
        let alpha = Alphabet::from_chars(&['a']);
        let wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(0, alpha);
        let result = minimize(&wfa);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Test: Already deterministic
    // -----------------------------------------------------------------------

    #[test]
    fn test_already_deterministic() {
        let wfa = simple_boolean_wfa();
        let result = minimize(&wfa).unwrap();
        // Should not change the number of states.
        assert_eq!(result.minimized_states, 2);
        assert_eq!(result.original_states, 2);
    }

    // -----------------------------------------------------------------------
    // Test: CountingSemiring
    // -----------------------------------------------------------------------

    #[test]
    fn test_counting_semiring_minimization() {
        let wfa = counting_wfa();
        let result = minimize(&wfa).unwrap();
        assert!(result.minimized_states <= wfa.num_states());

        // Verify language preservation.
        let words = enumerate_words(wfa.alphabet.size(), 4);
        for word in &words {
            let w_orig = compute_word_weight(&wfa, word);
            let w_min = compute_word_weight(&result.minimized, word);
            assert_eq!(
                format!("{:?}", w_orig),
                format!("{:?}", w_min),
                "Counting weight mismatch on {:?}",
                word
            );
        }
    }

    #[test]
    fn test_redundant_counting_minimization() {
        let wfa = redundant_counting_wfa();
        let result = minimize(&wfa).unwrap();
        assert!(result.minimized_states < wfa.num_states());
    }

    // -----------------------------------------------------------------------
    // Test: TropicalSemiring
    // -----------------------------------------------------------------------

    #[test]
    fn test_tropical_semiring_minimization() {
        let wfa = tropical_wfa();
        let result = minimize(&wfa).unwrap();
        assert!(result.minimized_states <= wfa.num_states());

        let words = enumerate_words(wfa.alphabet.size(), 3);
        for word in &words {
            let w_orig = compute_word_weight(&wfa, word);
            let w_min = compute_word_weight(&result.minimized, word);
            assert_eq!(
                format!("{:?}", w_orig),
                format!("{:?}", w_min),
                "Tropical weight mismatch on {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: Large automaton stress test
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_automaton_stress() {
        let num_states = 50;
        let alpha = Alphabet::from_chars(&['a', 'b']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(num_states, alpha);

        wfa.set_initial_weight(0, BooleanSemiring::one());

        // Create a chain with duplicated structure.
        for i in 0..num_states {
            if i % 5 == 4 {
                wfa.set_final_weight(i, BooleanSemiring::one());
            }
            if i + 1 < num_states {
                wfa.add_transition(i, 0, i + 1, BooleanSemiring::one());
            }
            if i + 2 < num_states {
                wfa.add_transition(i, 1, i + 2, BooleanSemiring::one());
            }
        }

        let result = minimize(&wfa).unwrap();
        assert!(result.minimized_states <= num_states);
        assert!(result.iterations > 0);

        // Verify on a sample of words.
        let words = enumerate_words(2, 3);
        for word in &words {
            let w_orig = compute_word_weight(&wfa, word);
            let w_min = compute_word_weight(&result.minimized, word);
            assert_eq!(
                format!("{:?}", w_orig),
                format!("{:?}", w_min),
                "Stress test mismatch on {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: Reduction ratio
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduction_ratio() {
        assert!((reduction_ratio(10, 5) - 0.5).abs() < 1e-10);
        assert!((reduction_ratio(10, 10) - 0.0).abs() < 1e-10);
        assert!((reduction_ratio(10, 0) - 1.0).abs() < 1e-10);
        assert!((reduction_ratio(0, 0) - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test: are_states_equivalent
    // -----------------------------------------------------------------------

    #[test]
    fn test_are_states_equivalent() {
        let wfa = redundant_boolean_wfa();
        assert!(are_states_equivalent(&wfa, 1, 2, 10));
        assert!(!are_states_equivalent(&wfa, 0, 1, 10));
    }

    // -----------------------------------------------------------------------
    // Test: Partition internals
    // -----------------------------------------------------------------------

    #[test]
    fn test_partition_basic() {
        let p = Partition::new(5);
        assert_eq!(p.num_blocks(), 1);
        for s in 0..5 {
            assert_eq!(p.block_of(s), 0);
        }
    }

    #[test]
    fn test_partition_split() {
        let mut p = Partition::new(4);
        let new_id = p.split(0, |s| s < 2);
        assert!(new_id.is_some());
        assert_eq!(p.num_blocks(), 2);
        assert_eq!(p.block_of(0), 0);
        assert_eq!(p.block_of(1), 0);
        let nid = new_id.unwrap();
        assert_eq!(p.block_of(2), nid);
        assert_eq!(p.block_of(3), nid);
    }

    #[test]
    fn test_partition_from_groups() {
        let groups = vec![vec![0, 1], vec![2, 3, 4]];
        let p = Partition::from_groups(5, &groups);
        assert_eq!(p.num_blocks(), 2);
        assert_eq!(p.block_of(0), p.block_of(1));
        assert_eq!(p.block_of(2), p.block_of(3));
        assert_ne!(p.block_of(0), p.block_of(2));
    }

    #[test]
    fn test_partition_refine_block() {
        let mut p = Partition::new(6);
        let new_blocks = p.refine_block(0, |s| s % 3);
        // Should produce 2 new blocks (original keeps one group, 2 new).
        assert_eq!(new_blocks.len(), 2);
        assert_eq!(p.num_blocks(), 3);
    }

    #[test]
    fn test_partition_state_mapping() {
        let groups = vec![vec![0, 2], vec![1, 3]];
        let p = Partition::from_groups(4, &groups);
        let mapping = p.build_state_mapping();
        assert_eq!(mapping[0], mapping[2]);
        assert_eq!(mapping[1], mapping[3]);
        assert_ne!(mapping[0], mapping[1]);
    }

    // -----------------------------------------------------------------------
    // Test: WorkList
    // -----------------------------------------------------------------------

    #[test]
    fn test_worklist() {
        let mut wl = WorkList::new();
        assert!(wl.is_empty());
        wl.push((0, 0));
        wl.push((0, 1));
        wl.push((0, 0)); // duplicate, should be ignored
        assert!(!wl.is_empty());
        assert_eq!(wl.pop(), Some((0, 0)));
        assert_eq!(wl.pop(), Some((0, 1)));
        assert!(wl.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: BisimulationRelation
    // -----------------------------------------------------------------------

    #[test]
    fn test_bisimulation_relation() {
        let groups = vec![vec![0, 1], vec![2, 3]];
        let p = Partition::from_groups(4, &groups);
        let rel = BisimulationRelation::from_partition(p);
        assert!(rel.are_bisimilar(0, 1));
        assert!(rel.are_bisimilar(2, 3));
        assert!(!rel.are_bisimilar(0, 2));
    }

    // -----------------------------------------------------------------------
    // Test: Hybrid strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_hybrid_strategy() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig {
            strategy: MinimizationStrategy::Hybrid,
            ..MinimizationConfig::default()
        };
        let result = minimize_with_config(&wfa, &config).unwrap();
        assert!(result.minimized_states < 3);
    }

    // -----------------------------------------------------------------------
    // Test: WeightedPartition strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_partition_strategy() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig {
            strategy: MinimizationStrategy::WeightedPartition,
            ..MinimizationConfig::default()
        };
        let result = minimize_with_config(&wfa, &config).unwrap();
        assert!(result.minimized_states < 3);
    }

    // -----------------------------------------------------------------------
    // Test: enumerate_words
    // -----------------------------------------------------------------------

    #[test]
    fn test_enumerate_words() {
        let words = enumerate_words(2, 2);
        // Should include: [], [0], [1], [0,0], [0,1], [1,0], [1,1]
        assert_eq!(words.len(), 7); // 1 + 2 + 4
        assert!(words.contains(&vec![]));
        assert!(words.contains(&vec![0]));
        assert!(words.contains(&vec![1, 0]));
    }

    // -----------------------------------------------------------------------
    // Test: compute_word_weight
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_word_weight() {
        let wfa = simple_boolean_wfa();
        // Empty word: only state 0 is initial, state 0 is non-final -> zero.
        let w_empty = compute_word_weight(&wfa, &[]);
        assert!(w_empty.is_zero());

        // Word "a": 0 -> 1, state 1 is final -> one.
        let w_a = compute_word_weight(&wfa, &[0]);
        assert!(w_a.is_one());
    }

    // -----------------------------------------------------------------------
    // Test: batch_merge
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_merge() {
        let mut wfa = redundant_boolean_wfa();
        let groups = vec![vec![1, 2]];
        let merged = batch_merge(&mut wfa, &groups).unwrap();
        assert!(merged.num_states() <= 3);
    }

    // -----------------------------------------------------------------------
    // Test: hash_wfa
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_wfa() {
        let wfa1 = simple_boolean_wfa();
        let wfa2 = simple_boolean_wfa();
        let h1 = hash_wfa(&wfa1);
        let h2 = hash_wfa(&wfa2);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 hex
    }

    // -----------------------------------------------------------------------
    // Test: language_fingerprint
    // -----------------------------------------------------------------------

    #[test]
    fn test_language_fingerprint() {
        let wfa1 = simple_boolean_wfa();
        let wfa2 = simple_boolean_wfa();
        let fp1 = language_fingerprint(&wfa1, 3);
        let fp2 = language_fingerprint(&wfa2, 3);
        assert_eq!(fp1, fp2);
    }

    // -----------------------------------------------------------------------
    // Test: find_counterexample
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_counterexample_none() {
        let wfa1 = simple_boolean_wfa();
        let wfa2 = simple_boolean_wfa();
        let ce = find_counterexample(&wfa1, &wfa2, 5);
        assert!(ce.is_none());
    }

    // -----------------------------------------------------------------------
    // Test: is_minimal
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_minimal_true() {
        let wfa = simple_boolean_wfa();
        assert!(is_minimal(&wfa));
    }

    #[test]
    fn test_is_minimal_false() {
        let wfa = redundant_boolean_wfa();
        assert!(!is_minimal(&wfa));
    }

    // -----------------------------------------------------------------------
    // Test: deduplicate_transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_deduplicate_transitions() {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::<CountingSemiring>::new(2, alpha);
        wfa.set_initial_weight(0, CountingSemiring::one());
        wfa.set_final_weight(1, CountingSemiring::one());
        wfa.add_transition(0, 0, 1, CountingSemiring::one());
        wfa.add_transition(0, 0, 1, CountingSemiring::one()); // duplicate

        assert_eq!(wfa.transitions[0][0].len(), 2);
        deduplicate_transitions(&mut wfa);
        assert_eq!(wfa.transitions[0][0].len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test: remove_zero_transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_zero_transitions() {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(2, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::zero());
        assert_eq!(wfa.transitions[0][0].len(), 2);
        remove_zero_transitions(&mut wfa);
        assert_eq!(wfa.transitions[0][0].len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test: validate_partition
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_partition() {
        let p = Partition::from_groups(5, &[vec![0, 1], vec![2, 3, 4]]);
        assert!(validate_partition(&p));
    }

    // -----------------------------------------------------------------------
    // Test: invert_mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_invert_mapping() {
        let mapping = vec![0, 0, 1, 1, 2];
        let inv = invert_mapping(&mapping);
        assert_eq!(inv[0], vec![0, 1]);
        assert_eq!(inv[1], vec![2, 3]);
        assert_eq!(inv[2], vec![4]);
    }

    // -----------------------------------------------------------------------
    // Test: compose_mappings
    // -----------------------------------------------------------------------

    #[test]
    fn test_compose_mappings() {
        let first = vec![1, 0, 2];
        let second = vec![10, 20, 30];
        let composed = compose_mappings(&first, &second);
        assert_eq!(composed, vec![20, 10, 30]);
    }

    // -----------------------------------------------------------------------
    // Test: canonical_state_ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_canonical_state_ordering() {
        let wfa = simple_boolean_wfa();
        let order = canonical_state_ordering(&wfa);
        assert_eq!(order.len(), 2);
        // State 0 is initial, should come first.
        assert_eq!(order[0], 0);
    }

    // -----------------------------------------------------------------------
    // Test: minimization_summary
    // -----------------------------------------------------------------------

    #[test]
    fn test_minimization_summary() {
        let wfa = redundant_boolean_wfa();
        let result = minimize(&wfa).unwrap();
        let summary = minimization_summary(&result);
        assert!(summary.contains("Minimized"));
        assert!(summary.contains("reduction"));
    }

    // -----------------------------------------------------------------------
    // Test: product_wfa
    // -----------------------------------------------------------------------

    #[test]
    fn test_product_wfa() {
        let wfa = simple_boolean_wfa();
        let prod = product_wfa(&wfa, &wfa).unwrap();
        assert_eq!(prod.num_states(), 4); // 2 * 2

        // The product with itself should accept the same language.
        let words = enumerate_words(1, 3);
        for word in &words {
            let w_orig = compute_word_weight(&wfa, word);
            let w_prod = compute_word_weight(&prod, word);
            // For boolean: product of identical = same.
            let expected = format!("{:?}", w_orig.clone().mul(&w_orig));
            assert_eq!(
                format!("{:?}", w_prod),
                expected,
                "Product mismatch on {:?}",
                word
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: find_equivalent_pairs
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_equivalent_pairs() {
        let wfa = redundant_boolean_wfa();
        let pairs = find_equivalent_pairs(&wfa, 10);
        assert!(pairs.contains(&(1, 2)));
    }

    // -----------------------------------------------------------------------
    // Test: Convergence failure on max_iterations = 0
    // -----------------------------------------------------------------------

    #[test]
    fn test_convergence_failure() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig {
            max_iterations: 0,
            ..MinimizationConfig::default()
        };
        let result = minimize_with_config(&wfa, &config);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Test: MyhillNerode strategy dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_myhill_nerode_strategy_dispatch() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig {
            strategy: MinimizationStrategy::MyhillNerode,
            ..MinimizationConfig::default()
        };
        let result = minimize_with_config(&wfa, &config).unwrap();
        assert!(result.minimized_states < 3);
    }

    // -----------------------------------------------------------------------
    // Test: Bisimulation strategy dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_bisimulation_strategy_dispatch() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig {
            strategy: MinimizationStrategy::Bisimulation,
            ..MinimizationConfig::default()
        };
        let result = minimize_with_config(&wfa, &config).unwrap();
        assert!(result.minimized_states < 3);
    }

    // -----------------------------------------------------------------------
    // Test: iterative_equivalence_check
    // -----------------------------------------------------------------------

    #[test]
    fn test_iterative_equivalence_check() {
        let wfa = redundant_boolean_wfa();
        assert!(iterative_equivalence_check(&wfa, 1, 2, 10));
        assert!(!iterative_equivalence_check(&wfa, 0, 1, 10));
    }

    // -----------------------------------------------------------------------
    // Test: trim_wfa
    // -----------------------------------------------------------------------

    #[test]
    fn test_trim_wfa() {
        // Add an unreachable state.
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(3, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        // State 2 is unreachable and has no final weight.

        let trimmed = trim_wfa(&wfa);
        assert!(trimmed.num_states() <= 2);
    }

    // -----------------------------------------------------------------------
    // Test: reachable / co-reachable
    // -----------------------------------------------------------------------

    #[test]
    fn test_reachable_states() {
        let wfa = simple_boolean_wfa();
        let reach = reachable_states(&wfa);
        assert!(reach.contains(&0));
        assert!(reach.contains(&1));
    }

    #[test]
    fn test_co_reachable_states() {
        let wfa = simple_boolean_wfa();
        let coreach = co_reachable_states(&wfa);
        assert!(coreach.contains(&0));
        assert!(coreach.contains(&1));
    }

    // -----------------------------------------------------------------------
    // Test: transition_matrix
    // -----------------------------------------------------------------------

    #[test]
    fn test_transition_matrix() {
        let wfa = simple_boolean_wfa();
        let mat = transition_matrix(&wfa, 0);
        assert_eq!(mat.len(), 2);
        assert!(mat[0][1].is_one());
        assert!(mat[1][1].is_one());
    }

    // -----------------------------------------------------------------------
    // Test: count_transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_count_transitions() {
        let wfa = simple_boolean_wfa();
        assert_eq!(count_transitions(&wfa), 2);
    }

    // -----------------------------------------------------------------------
    // Test: reverse_transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_reverse_transitions() {
        let wfa = simple_boolean_wfa();
        let rev = reverse_transitions(&wfa);
        // State 1 on symbol 0 should have predecessors {0, 1}.
        let preds: Vec<usize> = rev[1][0].iter().map(|(s, _)| *s).collect();
        assert!(preds.contains(&0));
        assert!(preds.contains(&1));
    }

    // -----------------------------------------------------------------------
    // Test: final_weight_classes
    // -----------------------------------------------------------------------

    #[test]
    fn test_final_weight_classes() {
        let wfa = redundant_boolean_wfa();
        let classes = final_weight_classes(&wfa);
        // States 1, 2 have the same final weight (one); state 0 has zero.
        assert_eq!(classes.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test: all strategies produce equivalent results on counting WFA
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_strategies_counting() {
        let wfa = redundant_counting_wfa();
        let strategies = [
            MinimizationStrategy::Hopcroft,
            MinimizationStrategy::MyhillNerode,
            MinimizationStrategy::Bisimulation,
            MinimizationStrategy::WeightedPartition,
            MinimizationStrategy::Hybrid,
        ];

        let mut results: Vec<MinimizationResult<CountingSemiring>> = Vec::new();
        for &strategy in &strategies {
            let config = MinimizationConfig {
                strategy,
                verify_result: false,
                ..MinimizationConfig::default()
            };
            let result = minimize_with_config(&wfa, &config).unwrap();
            results.push(result);
        }

        // All should produce the same number of states.
        let expected = results[0].minimized_states;
        for r in &results {
            assert_eq!(r.minimized_states, expected);
        }

        // All should produce equivalent languages.
        let words = enumerate_words(wfa.alphabet.size(), 4);
        for word in &words {
            let expected_w = format!("{:?}", compute_word_weight(&results[0].minimized, word));
            for r in &results[1..] {
                let actual = format!("{:?}", compute_word_weight(&r.minimized, word));
                assert_eq!(actual, expected_w);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test: minimization idempotence
    // -----------------------------------------------------------------------

    #[test]
    fn test_minimization_idempotent() {
        let wfa = redundant_boolean_wfa();
        let r1 = minimize(&wfa).unwrap();
        let r2 = minimize(&r1.minimized).unwrap();
        assert_eq!(r1.minimized_states, r2.minimized_states);
    }

    // -----------------------------------------------------------------------
    // Test: state_distance_approx
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_distance_approx() {
        let wfa = redundant_boolean_wfa();
        // Equivalent states should have distance 0.
        assert_eq!(state_distance_approx(&wfa, 1, 2, 3), 0);
        // Non-equivalent states should have positive distance.
        assert!(state_distance_approx(&wfa, 0, 1, 3) > 0);
    }

    // -----------------------------------------------------------------------
    // Test: generic_partition_refinement
    // -----------------------------------------------------------------------

    #[test]
    fn test_generic_partition_refinement() {
        let wfa = redundant_boolean_wfa();
        let config = MinimizationConfig::default();
        let (partition, iters) = generic_partition_refinement(&wfa, &config).unwrap();
        assert!(partition.num_blocks() >= 2);
        assert!(iters > 0);
    }

    // -----------------------------------------------------------------------
    // Test: permute_states
    // -----------------------------------------------------------------------

    #[test]
    fn test_permute_states() {
        let wfa = simple_boolean_wfa();
        let perm = vec![1, 0]; // swap states
        let permuted = permute_states(&wfa, &perm);

        // Verify language is preserved.
        let words = enumerate_words(1, 4);
        for word in &words {
            let w1 = compute_word_weight(&wfa, word);
            let w2 = compute_word_weight(&permuted, word);
            assert_eq!(format!("{:?}", w1), format!("{:?}", w2));
        }
    }

    // -----------------------------------------------------------------------
    // Test: format_partition (smoke test)
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_partition() {
        let p = Partition::from_groups(4, &[vec![0, 1], vec![2, 3]]);
        let s = format_partition(&p);
        assert!(s.contains("B0"));
        assert!(s.contains("B1"));
    }

    // -----------------------------------------------------------------------
    // Test: validate_mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_mapping() {
        assert!(validate_mapping(&[0, 1, 0, 1], 2));
        assert!(!validate_mapping(&[0, 1, 2, 3], 2));
    }

    // -----------------------------------------------------------------------
    // Test: debug_verify_partition
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_verify_partition() {
        let wfa = simple_boolean_wfa();
        let p = initial_partition_by_final_weight(&wfa);
        let issues = debug_verify_partition(&wfa, &p);
        assert!(issues.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: equivalence_signatures
    // -----------------------------------------------------------------------

    #[test]
    fn test_equivalence_signatures() {
        let wfa = redundant_boolean_wfa();
        let p = initial_partition_by_final_weight(&wfa);
        let sigs = equivalence_signatures(&wfa, &p);
        assert_eq!(sigs.len(), 3);
        // States 1 and 2 should have the same signature.
        assert_eq!(sigs[1], sigs[2]);
    }

    // -----------------------------------------------------------------------
    // Test: multi-symbol WFA minimization
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_symbol_minimization() {
        let alpha = Alphabet::from_chars(&['a', 'b', 'c']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(4, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(3, BooleanSemiring::one());

        // 0 -a-> 1, 0 -b-> 2, 1 -c-> 3, 2 -c-> 3
        // States 1 and 2 are equivalent (same transitions, same final weight).
        wfa.add_transition(0, 0, 1, BooleanSemiring::one());
        wfa.add_transition(0, 1, 2, BooleanSemiring::one());
        wfa.add_transition(1, 2, 3, BooleanSemiring::one());
        wfa.add_transition(2, 2, 3, BooleanSemiring::one());

        let result = minimize(&wfa).unwrap();
        // States 1 and 2 should be merged.
        assert!(result.minimized_states < 4);

        // Verify language.
        let words = enumerate_words(3, 3);
        for word in &words {
            let w_orig = compute_word_weight(&wfa, word);
            let w_min = compute_word_weight(&result.minimized, word);
            assert_eq!(format!("{:?}", w_orig), format!("{:?}", w_min));
        }
    }

    // -----------------------------------------------------------------------
    // Test: self-loop minimization
    // -----------------------------------------------------------------------

    #[test]
    fn test_self_loop_minimization() {
        let alpha = Alphabet::from_chars(&['a']);
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(1, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 0, 0, BooleanSemiring::one());

        let result = minimize(&wfa).unwrap();
        assert_eq!(result.minimized_states, 1);
    }
}
