//! WFA equivalence checking algorithms.
//!
//! Provides multiple algorithms for checking whether two weighted finite
//! automata define the same formal power series:
//!
//! - **Minimization-based**: minimize both WFAs, compare canonical forms.
//! - **Coalgebraic bisimulation**: explore the product state space.
//! - **Hopcroft–Karp style**: union-find on paired states.
//! - **Bounded-depth**: enumerate all strings up to a given length.
//! - **Random sampling**: probabilistic equivalence test.
//! - **Approximate**: check closeness within a tolerance.
//!
//! All algorithms work generically over any [`Semiring`].

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use log::{debug, trace, warn};
use rand::Rng;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::automaton::{Alphabet, Symbol, Transition, WeightedFiniteAutomaton, WfaError};
use super::semiring::{
    BooleanSemiring, CountingSemiring, RealSemiring, Semiring, SemiringMatrix, StarSemiring,
    TropicalSemiring,
};

// ===========================================================================
// Error types
// ===========================================================================

/// Errors specific to equivalence checking.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EquivalenceError {
    #[error("incompatible alphabets: {desc}")]
    IncompatibleAlphabets { desc: String },

    #[error("empty automaton")]
    EmptyAutomaton,

    #[error("computation timed out after {steps} steps")]
    ComputationTimeout { steps: usize },

    #[error("internal error: {message}")]
    InternalError { message: String },

    #[error("automaton is non-deterministic: {desc}")]
    NonDeterministic { desc: String },

    #[error("invalid certificate: {reason}")]
    InvalidCertificate { reason: String },
}

impl From<WfaError> for EquivalenceError {
    fn from(e: WfaError) -> Self {
        EquivalenceError::InternalError {
            message: e.to_string(),
        }
    }
}

pub type EquivResult<T> = Result<T, EquivalenceError>;

// ===========================================================================
// Core result / certificate types
// ===========================================================================

/// Method used for the equivalence check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquivalenceMethod {
    Minimization,
    Bisimulation,
    BoundedCheck,
    RandomSampling,
    Coalgebraic,
    HopcroftKarp,
}

impl fmt::Display for EquivalenceMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquivalenceMethod::Minimization => write!(f, "Minimization"),
            EquivalenceMethod::Bisimulation => write!(f, "Bisimulation"),
            EquivalenceMethod::BoundedCheck => write!(f, "BoundedCheck"),
            EquivalenceMethod::RandomSampling => write!(f, "RandomSampling"),
            EquivalenceMethod::Coalgebraic => write!(f, "Coalgebraic"),
            EquivalenceMethod::HopcroftKarp => write!(f, "HopcroftKarp"),
        }
    }
}

/// A word that distinguishes two non-equivalent WFAs.
#[derive(Debug, Clone, PartialEq)]
pub struct DistinguishingWord {
    /// The word as a sequence of symbol indices.
    pub word: Vec<usize>,
    /// Formatted weight of the word in the first WFA.
    pub weight_in_first: String,
    /// Formatted weight of the word in the second WFA.
    pub weight_in_second: String,
    /// Length of the word.
    pub length: usize,
}

impl DistinguishingWord {
    pub fn new(word: Vec<usize>, w1: String, w2: String) -> Self {
        let length = word.len();
        DistinguishingWord {
            word,
            weight_in_first: w1,
            weight_in_second: w2,
            length,
        }
    }
}

impl fmt::Display for DistinguishingWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "word {:?} (len {}): wfa1={}, wfa2={}",
            self.word, self.length, self.weight_in_first, self.weight_in_second
        )
    }
}

/// Certificate proving two WFAs are equivalent.
#[derive(Debug, Clone, PartialEq)]
pub struct EquivalenceCertificate {
    /// Algorithm used.
    pub method: EquivalenceMethod,
    /// Optional bisimulation relation (pairs of state indices).
    pub bisimulation_relation: Option<Vec<(usize, usize)>>,
    /// Depth to which the check was performed.
    pub checked_depth: usize,
    /// Total state-pair count explored.
    pub total_pairs_checked: usize,
    /// SHA-256 hash of the certificate data.
    pub hash: String,
    /// ISO-8601 timestamp (or monotonic counter in test mode).
    pub timestamp: String,
}

impl fmt::Display for EquivalenceCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Certificate(method={}, depth={}, pairs={}, hash={})",
            self.method, self.checked_depth, self.total_pairs_checked, self.hash
        )
    }
}

/// Outcome of an equivalence check.
#[derive(Debug, Clone, PartialEq)]
pub enum EquivalenceResult {
    /// The two WFAs are equivalent.
    Equivalent {
        certificate: EquivalenceCertificate,
    },
    /// The two WFAs are *not* equivalent; a witness word is provided.
    NotEquivalent {
        witness: DistinguishingWord,
    },
}

impl EquivalenceResult {
    pub fn is_equivalent(&self) -> bool {
        matches!(self, EquivalenceResult::Equivalent { .. })
    }

    pub fn is_not_equivalent(&self) -> bool {
        matches!(self, EquivalenceResult::NotEquivalent { .. })
    }

    pub fn witness(&self) -> Option<&DistinguishingWord> {
        match self {
            EquivalenceResult::NotEquivalent { witness } => Some(witness),
            _ => None,
        }
    }

    pub fn certificate(&self) -> Option<&EquivalenceCertificate> {
        match self {
            EquivalenceResult::Equivalent { certificate } => Some(certificate),
            _ => None,
        }
    }
}

/// Result of an approximate equivalence check.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproximateResult {
    /// Whether the WFAs are approximately equivalent within the tolerance.
    pub is_approximate: bool,
    /// Maximum weight difference observed.
    pub max_difference: f64,
    /// Word witnessing the maximum difference.
    pub worst_word: Vec<usize>,
    /// Number of samples checked.
    pub samples_checked: usize,
}

// ===========================================================================
// Configuration
// ===========================================================================

/// Configuration for equivalence checking.
#[derive(Debug, Clone)]
pub struct EquivalenceConfig {
    /// Which algorithm to use.
    pub method: EquivalenceMethod,
    /// Maximum BFS / enumeration depth.
    pub max_depth: usize,
    /// Maximum product-state-space size before aborting.
    pub max_states: usize,
    /// Number of random samples (for `RandomSampling`).
    pub num_samples: usize,
    /// Maximum word length for random sampling.
    pub max_word_length: usize,
    /// Step budget before timeout.
    pub timeout_steps: usize,
    /// Tolerance for approximate comparison.
    pub tolerance: f64,
}

impl Default for EquivalenceConfig {
    fn default() -> Self {
        EquivalenceConfig {
            method: EquivalenceMethod::Bisimulation,
            max_depth: 1000,
            max_states: 100_000,
            num_samples: 10_000,
            max_word_length: 50,
            timeout_steps: 1_000_000,
            tolerance: 1e-9,
        }
    }
}

impl EquivalenceConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_method(mut self, m: EquivalenceMethod) -> Self {
        self.method = m;
        self
    }

    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    pub fn with_max_states(mut self, s: usize) -> Self {
        self.max_states = s;
        self
    }

    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    pub fn with_max_word_length(mut self, l: usize) -> Self {
        self.max_word_length = l;
        self
    }

    pub fn with_timeout_steps(mut self, t: usize) -> Self {
        self.timeout_steps = t;
        self
    }

    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn compute_certificate_hash(
    method: &EquivalenceMethod,
    pairs: usize,
    depth: usize,
    relation: &Option<Vec<(usize, usize)>>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}", method).as_bytes());
    hasher.update(pairs.to_le_bytes());
    hasher.update(depth.to_le_bytes());
    if let Some(ref rel) = relation {
        for &(a, b) in rel {
            hasher.update(a.to_le_bytes());
            hasher.update(b.to_le_bytes());
        }
    }
    let digest = hasher.finalize();
    hex::encode(digest)
}

fn timestamp_now() -> String {
    // Monotonic counter for deterministic tests; real code would use
    // chrono or std::time::SystemTime.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("T{}", c)
}

/// Validate that two WFAs share a compatible alphabet.
fn validate_alphabets<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> EquivResult<()> {
    if wfa1.alphabet().size() != wfa2.alphabet().size() {
        return Err(EquivalenceError::IncompatibleAlphabets {
            desc: format!(
                "alphabet sizes differ: {} vs {}",
                wfa1.alphabet().size(),
                wfa2.alphabet().size()
            ),
        });
    }
    Ok(())
}

/// Build a distinguishing word from a product-state BFS trace.
fn trace_to_word(parent: &HashMap<(usize, usize), ((usize, usize), usize)>, target: (usize, usize)) -> Vec<usize> {
    let mut word = Vec::new();
    let mut cur = target;
    while let Some(&(prev, sym)) = parent.get(&cur) {
        word.push(sym);
        cur = prev;
    }
    word.reverse();
    word
}

// ===========================================================================
// Certificate generation & verification
// ===========================================================================

/// Generate a certificate attesting equivalence.
pub fn generate_equivalence_certificate<S: Semiring>(
    _wfa1: &WeightedFiniteAutomaton<S>,
    _wfa2: &WeightedFiniteAutomaton<S>,
    method: EquivalenceMethod,
    relation: &[(usize, usize)],
) -> EquivalenceCertificate {
    let bisim = if relation.is_empty() {
        None
    } else {
        Some(relation.to_vec())
    };
    let pairs = relation.len();
    let depth = relation.iter().map(|&(a, b)| a.max(b)).max().unwrap_or(0);
    let hash = compute_certificate_hash(&method, pairs, depth, &bisim);
    EquivalenceCertificate {
        method,
        bisimulation_relation: bisim,
        checked_depth: depth,
        total_pairs_checked: pairs,
        hash,
        timestamp: timestamp_now(),
    }
}

/// Verify that a certificate is internally consistent and matches the WFAs.
///
/// This re-checks the bisimulation relation (if present) against the actual
/// transition structure of both WFAs.
pub fn verify_equivalence_certificate<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    cert: &EquivalenceCertificate,
) -> EquivResult<bool> {
    // 1. Re-compute hash
    let expected_hash = compute_certificate_hash(
        &cert.method,
        cert.total_pairs_checked,
        cert.checked_depth,
        &cert.bisimulation_relation,
    );
    if expected_hash != cert.hash {
        return Err(EquivalenceError::InvalidCertificate {
            reason: "hash mismatch".into(),
        });
    }

    // 2. If there is a bisimulation relation, verify it
    if let Some(ref relation) = cert.bisimulation_relation {
        let n1 = wfa1.state_count();
        let n2 = wfa2.state_count();
        let alpha = wfa1.alphabet().size();

        for &(s1, s2) in relation {
            if s1 >= n1 || s2 >= n2 {
                return Err(EquivalenceError::InvalidCertificate {
                    reason: format!("state pair ({}, {}) out of bounds", s1, s2),
                });
            }
            // Final weights must agree
            if wfa1.final_weights()[s1] != wfa2.final_weights()[s2] {
                return Ok(false);
            }
            // For each symbol, the sum of successor weights must be
            // consistent (in a bisimulation the successors must themselves
            // be related, but checking full closure requires the relation
            // to be stored as a set for lookup).
            let rel_set: HashSet<(usize, usize)> = relation.iter().copied().collect();
            for a in 0..alpha {
                let targets1 = wfa1.transitions_from(s1, a);
                let targets2 = wfa2.transitions_from(s2, a);
                // Every target of wfa1 must be paired with some target of wfa2
                for &(t1, ref _w1) in targets1 {
                    let has_match = targets2
                        .iter()
                        .any(|&(t2, ref _w2)| rel_set.contains(&(t1, t2)));
                    if !has_match {
                        return Ok(false);
                    }
                }
            }
        }
    }

    Ok(true)
}

// ===========================================================================
// Top-level API
// ===========================================================================

/// Check equivalence of two WFAs using the default (bisimulation) method.
pub fn check_equivalence<S: Semiring + Ord>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> EquivResult<EquivalenceResult> {
    let config = EquivalenceConfig::default();
    check_equivalence_with_config(wfa1, wfa2, &config)
}

/// Check equivalence with a specific configuration.
pub fn check_equivalence_with_config<S: Semiring + Ord>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    config: &EquivalenceConfig,
) -> EquivResult<EquivalenceResult> {
    validate_alphabets(wfa1, wfa2)?;

    match config.method {
        EquivalenceMethod::Bisimulation | EquivalenceMethod::Coalgebraic => {
            check_via_bisimulation(wfa1, wfa2, config)
        }
        EquivalenceMethod::HopcroftKarp => check_via_hopcroft_karp(wfa1, wfa2, config),
        EquivalenceMethod::BoundedCheck => check_bounded(wfa1, wfa2, config.max_depth),
        EquivalenceMethod::RandomSampling => check_random_sampling(wfa1, wfa2, config),
        EquivalenceMethod::Minimization => check_via_minimization(wfa1, wfa2, config),
    }
}

// ===========================================================================
// 1. Coalgebraic bisimulation
// ===========================================================================

/// State in the product automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ProductState {
    /// State index in the first WFA.
    s1: usize,
    /// State index in the second WFA.
    s2: usize,
}

impl ProductState {
    fn new(s1: usize, s2: usize) -> Self {
        ProductState { s1, s2 }
    }
}

/// Internal checker that explores the product-state space via BFS.
struct BisimulationChecker<'a, S: Semiring> {
    wfa1: &'a WeightedFiniteAutomaton<S>,
    wfa2: &'a WeightedFiniteAutomaton<S>,
    config: &'a EquivalenceConfig,
    /// States already visited.
    visited: HashSet<ProductState>,
    /// BFS queue.
    queue: VecDeque<ProductState>,
    /// Parent pointers for extracting a distinguishing word.
    parent: HashMap<ProductState, (ProductState, usize)>,
    /// The bisimulation relation (pairs that were found compatible).
    relation: Vec<(usize, usize)>,
    /// Total steps (for timeout).
    steps: usize,
    /// Maximum depth reached.
    max_depth: usize,
    /// Depth per product state.
    depth: HashMap<ProductState, usize>,
}

impl<'a, S: Semiring> BisimulationChecker<'a, S> {
    fn new(
        wfa1: &'a WeightedFiniteAutomaton<S>,
        wfa2: &'a WeightedFiniteAutomaton<S>,
        config: &'a EquivalenceConfig,
    ) -> Self {
        BisimulationChecker {
            wfa1,
            wfa2,
            config,
            visited: HashSet::new(),
            queue: VecDeque::new(),
            parent: HashMap::new(),
            relation: Vec::new(),
            steps: 0,
            max_depth: 0,
            depth: HashMap::new(),
        }
    }

    /// Seed the BFS with initial-state pairs whose initial weights are
    /// compatible.
    fn seed_initial_pairs(&mut self) -> EquivResult<Option<DistinguishingWord>> {
        let n1 = self.wfa1.state_count();
        let n2 = self.wfa2.state_count();

        for i in 0..n1 {
            if self.wfa1.initial_weights()[i].is_zero() {
                continue;
            }
            for j in 0..n2 {
                if self.wfa2.initial_weights()[j].is_zero() {
                    continue;
                }
                // Both states are initial.  Check that initial weights agree.
                if self.wfa1.initial_weights()[i] != self.wfa2.initial_weights()[j] {
                    // Non-equivalent on the empty word if these are the only
                    // initial states with non-zero weight.  However, the
                    // overall weight on the empty word is
                    //   sum_i  alpha1[i] * rho1[i]   vs   sum_j  alpha2[j] * rho2[j]
                    // so we defer the empty-word check to check_empty_word().
                }
                let ps = ProductState::new(i, j);
                if self.visited.insert(ps) {
                    self.queue.push_back(ps);
                    self.depth.insert(ps, 0);
                }
            }
        }

        // If no initial pairs at all, check if both WFAs assign zero to
        // everything.
        if self.queue.is_empty() {
            let any_init1 = self.wfa1.initial_weights().iter().any(|w| !w.is_zero());
            let any_init2 = self.wfa2.initial_weights().iter().any(|w| !w.is_zero());
            if any_init1 != any_init2 {
                return Ok(Some(DistinguishingWord::new(
                    vec![],
                    format!("has_initial={}", any_init1),
                    format!("has_initial={}", any_init2),
                )));
            }
        }

        Ok(None)
    }

    /// Check whether the two WFAs agree on the empty string.
    fn check_empty_word(&self) -> Option<DistinguishingWord> {
        let w1 = self.compute_empty_weight(self.wfa1);
        let w2 = self.compute_empty_weight(self.wfa2);
        if w1 != w2 {
            Some(DistinguishingWord::new(
                vec![],
                format!("{:?}", w1),
                format!("{:?}", w2),
            ))
        } else {
            None
        }
    }

    fn compute_empty_weight(&self, wfa: &WeightedFiniteAutomaton<S>) -> S {
        let mut total = S::zero();
        for i in 0..wfa.state_count() {
            let contrib = wfa.initial_weights()[i].mul(&wfa.final_weights()[i]);
            total.add_assign(&contrib);
        }
        total
    }

    /// Run the BFS exploration.
    fn run(&mut self) -> EquivResult<EquivalenceResult> {
        // 1. Check empty word
        if let Some(dw) = self.check_empty_word() {
            return Ok(EquivalenceResult::NotEquivalent { witness: dw });
        }

        // 2. Seed initial pairs
        if let Some(dw) = self.seed_initial_pairs()? {
            return Ok(EquivalenceResult::NotEquivalent { witness: dw });
        }

        let alpha_size = self.wfa1.alphabet().size();

        // 3. BFS
        while let Some(ps) = self.queue.pop_front() {
            self.steps += 1;
            if self.steps > self.config.timeout_steps {
                return Err(EquivalenceError::ComputationTimeout {
                    steps: self.steps,
                });
            }

            let cur_depth = self.depth.get(&ps).copied().unwrap_or(0);
            if cur_depth > self.max_depth {
                self.max_depth = cur_depth;
            }

            if cur_depth >= self.config.max_depth {
                continue;
            }

            // Check final weights
            if self.wfa1.final_weights()[ps.s1] != self.wfa2.final_weights()[ps.s2] {
                // Extract distinguishing word
                let word = self.extract_word(ps);
                let w1 = self.wfa1.compute_weight(&word);
                let w2 = self.wfa2.compute_weight(&word);
                return Ok(EquivalenceResult::NotEquivalent {
                    witness: DistinguishingWord::new(word, format!("{:?}", w1), format!("{:?}", w2)),
                });
            }

            self.relation.push((ps.s1, ps.s2));

            // Explore successors for each symbol
            for sym in 0..alpha_size {
                let targets1 = self.wfa1.transitions_from(ps.s1, sym);
                let targets2 = self.wfa2.transitions_from(ps.s2, sym);

                // Build aggregated successor weights:
                // For each pair of successor states, the combined weight
                // must be consistent.  We pair successors with matching
                // weights.
                let mut succ1_map: BTreeMap<usize, S> = BTreeMap::new();
                for &(t, ref w) in targets1 {
                    succ1_map
                        .entry(t)
                        .and_modify(|existing| existing.add_assign(w))
                        .or_insert_with(|| w.clone());
                }

                let mut succ2_map: BTreeMap<usize, S> = BTreeMap::new();
                for &(t, ref w) in targets2 {
                    succ2_map
                        .entry(t)
                        .and_modify(|existing| existing.add_assign(w))
                        .or_insert_with(|| w.clone());
                }

                // Enqueue all pairs of successors
                for (&t1, _w1) in &succ1_map {
                    for (&t2, _w2) in &succ2_map {
                        let next = ProductState::new(t1, t2);
                        if self.visited.insert(next) {
                            if self.visited.len() > self.config.max_states {
                                return Err(EquivalenceError::ComputationTimeout {
                                    steps: self.steps,
                                });
                            }
                            self.queue.push_back(next);
                            self.parent.insert(next, (ps, sym));
                            self.depth.insert(next, cur_depth + 1);
                        }
                    }
                }

                // If one side has successors and the other doesn't, we need
                // to check that the missing side contributes zero weight.
                if !succ1_map.is_empty() && succ2_map.is_empty() {
                    // wfa2 has no successors on this symbol from ps.s2.
                    // Check whether wfa1's successors can lead to a
                    // non-zero weight word.
                    for (&t1, _w1) in &succ1_map {
                        // Pair t1 with a "dead" state in wfa2.  We represent
                        // this by checking whether any extension of the
                        // current word gives non-zero weight in wfa1 but
                        // zero in wfa2.
                        let mut test_word = self.extract_word(ps);
                        test_word.push(sym);
                        let tw1 = self.wfa1.compute_weight(&test_word);
                        let tw2 = self.wfa2.compute_weight(&test_word);
                        if tw1 != tw2 {
                            return Ok(EquivalenceResult::NotEquivalent {
                                witness: DistinguishingWord::new(
                                    test_word,
                                    format!("{:?}", tw1),
                                    format!("{:?}", tw2),
                                ),
                            });
                        }
                        // Try extending one more symbol
                        for sym2 in 0..alpha_size {
                            let mut extended = test_word.clone();
                            extended.push(sym2);
                            let ew1 = self.wfa1.compute_weight(&extended);
                            let ew2 = self.wfa2.compute_weight(&extended);
                            if ew1 != ew2 {
                                return Ok(EquivalenceResult::NotEquivalent {
                                    witness: DistinguishingWord::new(
                                        extended,
                                        format!("{:?}", ew1),
                                        format!("{:?}", ew2),
                                    ),
                                });
                            }
                        }
                    }
                }
                if succ1_map.is_empty() && !succ2_map.is_empty() {
                    for (&t2, _w2) in &succ2_map {
                        let mut test_word = self.extract_word(ps);
                        test_word.push(sym);
                        let tw1 = self.wfa1.compute_weight(&test_word);
                        let tw2 = self.wfa2.compute_weight(&test_word);
                        if tw1 != tw2 {
                            return Ok(EquivalenceResult::NotEquivalent {
                                witness: DistinguishingWord::new(
                                    test_word,
                                    format!("{:?}", tw1),
                                    format!("{:?}", tw2),
                                ),
                            });
                        }
                    }
                }
            }
        }

        // All reachable product states are consistent.
        let relation = std::mem::take(&mut self.relation);
        let cert = generate_equivalence_certificate(
            self.wfa1,
            self.wfa2,
            EquivalenceMethod::Bisimulation,
            &relation,
        );
        Ok(EquivalenceResult::Equivalent { certificate: cert })
    }

    /// Reconstruct the word leading to `ps` by following parent pointers.
    fn extract_word(&self, ps: ProductState) -> Vec<usize> {
        let mut word = Vec::new();
        let mut cur = ps;
        while let Some(&(prev, sym)) = self.parent.get(&cur) {
            word.push(sym);
            cur = prev;
        }
        word.reverse();
        word
    }
}

/// Check equivalence via coalgebraic bisimulation (BFS on the product
/// automaton).
pub fn check_via_bisimulation<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    config: &EquivalenceConfig,
) -> EquivResult<EquivalenceResult> {
    validate_alphabets(wfa1, wfa2)?;

    if wfa1.state_count() == 0 && wfa2.state_count() == 0 {
        let cert = generate_equivalence_certificate(
            wfa1,
            wfa2,
            EquivalenceMethod::Bisimulation,
            &[],
        );
        return Ok(EquivalenceResult::Equivalent { certificate: cert });
    }

    let mut checker = BisimulationChecker::new(wfa1, wfa2, config);
    checker.run()
}

// ===========================================================================
// 2. Hopcroft–Karp style (union-find)
// ===========================================================================

/// Weighted union-find data structure.
///
/// Each element belongs to a class.  When two elements are unified we check
/// that their associated final weights are consistent.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path splitting
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false; // already in same set
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
        true
    }

    fn same(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}

/// Extended union-find tracking weights for Hopcroft–Karp equivalence.
struct WeightedUnionFind<S: Semiring> {
    uf: UnionFind,
    /// Final weight associated with each *representative*.
    final_weight: Vec<S>,
}

impl<S: Semiring> WeightedUnionFind<S> {
    fn new(final_weights: Vec<S>) -> Self {
        let n = final_weights.len();
        WeightedUnionFind {
            uf: UnionFind::new(n),
            final_weight: final_weights,
        }
    }

    fn find(&mut self, x: usize) -> usize {
        self.uf.find(x)
    }

    /// Attempt to union `a` and `b`.
    /// Returns `Err` if they have different final weights (and thus are
    /// distinguishable).
    fn union(&mut self, a: usize, b: usize) -> Result<bool, (usize, usize)> {
        let ra = self.uf.find(a);
        let rb = self.uf.find(b);
        if ra == rb {
            return Ok(false);
        }
        // Check final-weight compatibility
        if self.final_weight[ra] != self.final_weight[rb] {
            return Err((a, b));
        }
        self.uf.union(ra, rb);
        // Propagate the final weight to the new representative
        let new_rep = self.uf.find(ra);
        self.final_weight[new_rep] = self.final_weight[ra].clone();
        Ok(true)
    }

    fn same(&mut self, a: usize, b: usize) -> bool {
        self.uf.same(a, b)
    }
}

/// Hopcroft–Karp style equivalence checking via union-find on a combined
/// state space.
pub fn check_via_hopcroft_karp<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    config: &EquivalenceConfig,
) -> EquivResult<EquivalenceResult> {
    validate_alphabets(wfa1, wfa2)?;

    let n1 = wfa1.state_count();
    let n2 = wfa2.state_count();
    let n = n1 + n2;
    let alpha_size = wfa1.alphabet().size();

    if n == 0 {
        let cert = generate_equivalence_certificate(
            wfa1,
            wfa2,
            EquivalenceMethod::HopcroftKarp,
            &[],
        );
        return Ok(EquivalenceResult::Equivalent { certificate: cert });
    }

    // Build combined final-weight vector.
    // States 0..n1 belong to wfa1, states n1..n1+n2 belong to wfa2.
    let mut final_weights: Vec<S> = Vec::with_capacity(n);
    final_weights.extend_from_slice(wfa1.final_weights());
    final_weights.extend_from_slice(wfa2.final_weights());

    let mut wuf = WeightedUnionFind::new(final_weights);

    // Worklist: pairs to process
    let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();
    // Parent tracking for witness extraction
    let mut parent_map: HashMap<(usize, usize), ((usize, usize), usize)> = HashMap::new();
    let mut steps: usize = 0;

    // Seed with initial-state pairs
    // First check empty word
    let empty_w1 = {
        let mut total = S::zero();
        for i in 0..n1 {
            let c = wfa1.initial_weights()[i].mul(&wfa1.final_weights()[i]);
            total.add_assign(&c);
        }
        total
    };
    let empty_w2 = {
        let mut total = S::zero();
        for j in 0..n2 {
            let c = wfa2.initial_weights()[j].mul(&wfa2.final_weights()[j]);
            total.add_assign(&c);
        }
        total
    };
    if empty_w1 != empty_w2 {
        return Ok(EquivalenceResult::NotEquivalent {
            witness: DistinguishingWord::new(
                vec![],
                format!("{:?}", empty_w1),
                format!("{:?}", empty_w2),
            ),
        });
    }

    for i in 0..n1 {
        if wfa1.initial_weights()[i].is_zero() {
            continue;
        }
        for j in 0..n2 {
            if wfa2.initial_weights()[j].is_zero() {
                continue;
            }
            if wfa1.initial_weights()[i] == wfa2.initial_weights()[j] {
                let pair = (i, n1 + j);
                worklist.push_back(pair);
            }
        }
    }

    // Process worklist
    let mut relation: Vec<(usize, usize)> = Vec::new();

    while let Some((a, b)) = worklist.pop_front() {
        steps += 1;
        if steps > config.timeout_steps {
            return Err(EquivalenceError::ComputationTimeout { steps });
        }

        if wuf.same(a, b) {
            continue;
        }

        // Try to union
        match wuf.union(a, b) {
            Ok(true) => {
                // Successfully unified; record relation
                let s1 = if a < n1 { a } else { a - n1 };
                let s2 = if b < n1 { b } else { b - n1 };
                relation.push((s1, s2));
            }
            Ok(false) => continue, // already same
            Err((_ea, _eb)) => {
                // Final weights differ → not equivalent.
                // Extract the word from parent_map
                let word = extract_hk_word(&parent_map, (a, b));
                let w1 = wfa1.compute_weight(&word);
                let w2 = wfa2.compute_weight(&word);
                return Ok(EquivalenceResult::NotEquivalent {
                    witness: DistinguishingWord::new(
                        word,
                        format!("{:?}", w1),
                        format!("{:?}", w2),
                    ),
                });
            }
        }

        // Enqueue successor pairs for each symbol
        for sym in 0..alpha_size {
            // Gather successors in the combined space
            let succs_a: Vec<(usize, S)> = if a < n1 {
                wfa1.transitions_from(a, sym)
                    .iter()
                    .map(|&(t, ref w)| (t, w.clone()))
                    .collect()
            } else {
                wfa2.transitions_from(a - n1, sym)
                    .iter()
                    .map(|&(t, ref w)| (n1 + t, w.clone()))
                    .collect()
            };
            let succs_b: Vec<(usize, S)> = if b < n1 {
                wfa1.transitions_from(b, sym)
                    .iter()
                    .map(|&(t, ref w)| (t, w.clone()))
                    .collect()
            } else {
                wfa2.transitions_from(b - n1, sym)
                    .iter()
                    .map(|&(t, ref w)| (n1 + t, w.clone()))
                    .collect()
            };

            for &(ta, ref _wa) in &succs_a {
                for &(tb, ref _wb) in &succs_b {
                    if !wuf.same(ta, tb) {
                        let pair = (ta, tb);
                        if !parent_map.contains_key(&pair) {
                            parent_map.insert(pair, ((a, b), sym));
                        }
                        worklist.push_back(pair);
                    }
                }
            }
        }
    }

    let cert = generate_equivalence_certificate(
        wfa1,
        wfa2,
        EquivalenceMethod::HopcroftKarp,
        &relation,
    );
    Ok(EquivalenceResult::Equivalent { certificate: cert })
}

/// Extract a word from the Hopcroft–Karp parent map.
fn extract_hk_word(
    parent: &HashMap<(usize, usize), ((usize, usize), usize)>,
    target: (usize, usize),
) -> Vec<usize> {
    let mut word = Vec::new();
    let mut cur = target;
    while let Some(&(prev, sym)) = parent.get(&cur) {
        word.push(sym);
        cur = prev;
    }
    word.reverse();
    word
}

// ===========================================================================
// 3. Minimization-based equivalence
// ===========================================================================

/// Check equivalence by minimizing both WFAs and testing isomorphism of
/// the resulting canonical forms.
///
/// Requires `S: Ord` so that states can be sorted into a canonical order.
pub fn check_via_minimization<S: Semiring + Ord>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    config: &EquivalenceConfig,
) -> EquivResult<EquivalenceResult> {
    validate_alphabets(wfa1, wfa2)?;

    // Minimize both (trim, then merge indistinguishable states).
    let min1 = minimize_wfa(wfa1, config)?;
    let min2 = minimize_wfa(wfa2, config)?;

    // Check isomorphism of minimized WFAs.
    match check_isomorphism(&min1, &min2)? {
        Some(mapping) => {
            let relation: Vec<(usize, usize)> =
                mapping.iter().enumerate().map(|(i, &j)| (i, j)).collect();
            let cert = generate_equivalence_certificate(
                wfa1,
                wfa2,
                EquivalenceMethod::Minimization,
                &relation,
            );
            Ok(EquivalenceResult::Equivalent { certificate: cert })
        }
        None => {
            // Not isomorphic → not equivalent. Find a distinguishing word.
            match find_distinguishing_word(wfa1, wfa2) {
                Some(dw) => Ok(EquivalenceResult::NotEquivalent { witness: dw }),
                None => {
                    // Shouldn't happen if minimization is correct, but the
                    // automata might still agree on all words up to a depth
                    // due to algorithmic limitations.
                    Err(EquivalenceError::InternalError {
                        message: "minimized automata not isomorphic but no distinguishing word found"
                            .into(),
                    })
                }
            }
        }
    }
}

/// Partition-refinement minimization.
///
/// Groups states into equivalence classes based on their future behaviour
/// (final weights and transition structure).
fn minimize_wfa<S: Semiring + Ord>(
    wfa: &WeightedFiniteAutomaton<S>,
    _config: &EquivalenceConfig,
) -> EquivResult<WeightedFiniteAutomaton<S>> {
    let n = wfa.state_count();
    if n == 0 {
        return Ok(wfa.clone());
    }

    let alpha_size = wfa.alphabet().size();

    // 1. Initial partition: group states by final weight.
    let mut partition: Vec<usize> = vec![0; n];
    let mut class_map: BTreeMap<S, usize> = BTreeMap::new();
    let mut next_class: usize = 0;

    for i in 0..n {
        let fw = &wfa.final_weights()[i];
        let cls = class_map.entry(fw.clone()).or_insert_with(|| {
            let c = next_class;
            next_class += 1;
            c
        });
        partition[i] = *cls;
    }

    // 2. Refine until stable.
    let mut changed = true;
    let mut iteration = 0;
    let max_iterations = n * n;

    while changed && iteration < max_iterations {
        changed = false;
        iteration += 1;

        // For each state, compute a signature = (current class, for each
        // symbol: sorted list of (target class, weight)).
        let mut sig_map: BTreeMap<Vec<(usize, Vec<(usize, S)>)>, usize> = BTreeMap::new();
        let mut new_partition = vec![0usize; n];
        let mut new_next = 0usize;

        // Group by (current partition, signature)
        type Sig<S> = (usize, Vec<(usize, Vec<(usize, S)>)>);
        let mut full_sigs: BTreeMap<Sig<S>, usize> = BTreeMap::new();

        for state in 0..n {
            let cur_class = partition[state];
            let mut sym_sigs: Vec<(usize, Vec<(usize, S)>)> = Vec::with_capacity(alpha_size);
            for sym in 0..alpha_size {
                let mut targets: Vec<(usize, S)> = wfa
                    .transitions_from(state, sym)
                    .iter()
                    .map(|&(t, ref w)| (partition[t], w.clone()))
                    .collect();
                targets.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                sym_sigs.push((sym, targets));
            }

            let full_sig: Sig<S> = (cur_class, sym_sigs);
            let cls = full_sigs.entry(full_sig).or_insert_with(|| {
                let c = new_next;
                new_next += 1;
                c
            });
            new_partition[state] = *cls;
        }

        if new_partition != partition {
            changed = true;
            partition = new_partition;
        }
    }

    // 3. Build minimized WFA.
    let num_classes = *partition.iter().max().unwrap_or(&0) + 1;
    let mut min_wfa: WeightedFiniteAutomaton<S> = WeightedFiniteAutomaton::new(num_classes, wfa.alphabet().clone());

    // Pick one representative per class.
    let mut repr: Vec<Option<usize>> = vec![None; num_classes];
    for i in 0..n {
        let cls = partition[i];
        if repr[cls].is_none() {
            repr[cls] = Some(i);
        }
    }

    // Set initial weights: sum over states in each class
    for i in 0..n {
        let cls = partition[i];
        let old_init = &wfa.initial_weights()[i];
        if !old_init.is_zero() {
            let cur = min_wfa.initial_weights()[cls].add(old_init);
            min_wfa.set_initial_weight(cls, cur);
        }
    }

    // Set final weights from representative
    for cls in 0..num_classes {
        if let Some(rep) = repr[cls] {
            min_wfa.set_final_weight(cls, wfa.final_weights()[rep].clone());
        }
    }

    // Set transitions from representative of each class
    for cls in 0..num_classes {
        if let Some(rep) = repr[cls] {
            for sym in 0..alpha_size {
                // Aggregate transitions to the same target class
                let mut agg: BTreeMap<usize, S> = BTreeMap::new();
                for &(t, ref w) in wfa.transitions_from(rep, sym) {
                    let tcls = partition[t];
                    agg.entry(tcls)
                        .and_modify(|existing| existing.add_assign(w))
                        .or_insert_with(|| w.clone());
                }
                for (tcls, w) in agg {
                    let _ = min_wfa.add_transition(cls, sym, tcls, w);
                }
            }
        }
    }

    Ok(min_wfa)
}

/// Check whether two WFAs are isomorphic (same structure up to state
/// renaming).
///
/// Returns a state mapping `wfa1_state -> wfa2_state` if they are
/// isomorphic, or `None`.
///
/// Uses backtracking search with constraint propagation.
pub fn check_isomorphism<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> EquivResult<Option<Vec<usize>>> {
    let n1 = wfa1.state_count();
    let n2 = wfa2.state_count();

    if n1 != n2 {
        return Ok(None);
    }
    if n1 == 0 {
        return Ok(Some(vec![]));
    }

    let n = n1;
    let alpha_size = wfa1.alphabet().size();

    // Constraint propagation: for each state in wfa1, compute its
    // "fingerprint" (final weight, out-degree per symbol, in-degree).
    // Only states with matching fingerprints can be mapped to each other.

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Fingerprint<S: Semiring> {
        final_weight: S,
        out_degrees: Vec<usize>,
        initial_weight: S,
    }

    let fingerprint = |wfa: &WeightedFiniteAutomaton<S>, state: usize| -> Fingerprint<S> {
        let mut out_degrees = Vec::with_capacity(alpha_size);
        for sym in 0..alpha_size {
            out_degrees.push(wfa.transitions_from(state, sym).len());
        }
        Fingerprint {
            final_weight: wfa.final_weights()[state].clone(),
            out_degrees,
            initial_weight: wfa.initial_weights()[state].clone(),
        }
    };

    // Build candidate lists
    let fps1: Vec<Fingerprint<S>> = (0..n).map(|i| fingerprint(wfa1, i)).collect();
    let fps2: Vec<Fingerprint<S>> = (0..n).map(|i| fingerprint(wfa2, i)).collect();

    let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut cands = Vec::new();
        for j in 0..n {
            if fps1[i] == fps2[j] {
                cands.push(j);
            }
        }
        if cands.is_empty() {
            return Ok(None); // No valid mapping for state i
        }
        candidates.push(cands);
    }

    // Backtracking search
    let mut mapping: Vec<Option<usize>> = vec![None; n];
    let mut used: Vec<bool> = vec![false; n];

    fn backtrack<S: Semiring>(
        wfa1: &WeightedFiniteAutomaton<S>,
        wfa2: &WeightedFiniteAutomaton<S>,
        candidates: &[Vec<usize>],
        mapping: &mut Vec<Option<usize>>,
        used: &mut Vec<bool>,
        state: usize,
        n: usize,
        alpha_size: usize,
    ) -> bool {
        if state == n {
            return true;
        }

        for &cand in &candidates[state] {
            if used[cand] {
                continue;
            }

            // Check consistency with already-mapped states
            let mut consistent = true;
            for sym in 0..alpha_size {
                let trans1 = wfa1.transitions_from(state, sym);
                let trans2 = wfa2.transitions_from(cand, sym);

                // For transitions to already-mapped states, check that
                // the target mapping and weight are correct.
                for &(t1, ref w1) in trans1 {
                    if let Some(mapped_t1) = mapping[t1] {
                        let found = trans2.iter().any(|&(t2, ref w2)| {
                            t2 == mapped_t1 && w1 == w2
                        });
                        if !found {
                            consistent = false;
                            break;
                        }
                    }
                }
                if !consistent {
                    break;
                }
            }

            if !consistent {
                continue;
            }

            mapping[state] = Some(cand);
            used[cand] = true;

            if backtrack(wfa1, wfa2, candidates, mapping, used, state + 1, n, alpha_size) {
                return true;
            }

            mapping[state] = None;
            used[cand] = false;
        }

        false
    }

    if backtrack(wfa1, wfa2, &candidates, &mut mapping, &mut used, 0, n, alpha_size) {
        let result: Vec<usize> = mapping.into_iter().map(|o| o.unwrap()).collect();

        // Final verification: check all transitions
        for i in 0..n {
            let j = result[i];
            for sym in 0..alpha_size {
                let mut trans1: Vec<(usize, &S)> = wfa1
                    .transitions_from(i, sym)
                    .iter()
                    .map(|&(t, ref w)| (result[t], w))
                    .collect();
                let mut trans2: Vec<(usize, &S)> = wfa2
                    .transitions_from(j, sym)
                    .iter()
                    .map(|&(t, ref w)| (t, w))
                    .collect();

                trans1.sort_by_key(|&(t, _)| t);
                trans2.sort_by_key(|&(t, _)| t);

                if trans1.len() != trans2.len() {
                    return Ok(None);
                }
                for (a, b) in trans1.iter().zip(trans2.iter()) {
                    if a.0 != b.0 || a.1 != b.1 {
                        return Ok(None);
                    }
                }
            }
        }

        Ok(Some(result))
    } else {
        Ok(None)
    }
}

// ===========================================================================
// 4. Bounded-depth equivalence
// ===========================================================================

/// Check equivalence by enumerating all strings up to `max_depth`.
///
/// Guarantees that the WFAs agree on all words of length ≤ max_depth.
pub fn check_bounded<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    max_depth: usize,
) -> EquivResult<EquivalenceResult> {
    validate_alphabets(wfa1, wfa2)?;

    let alpha_size = wfa1.alphabet().size();
    if alpha_size == 0 {
        // Only the empty word
        let w1 = wfa1.compute_weight(&[]);
        let w2 = wfa2.compute_weight(&[]);
        if w1 == w2 {
            let cert = generate_equivalence_certificate(
                wfa1,
                wfa2,
                EquivalenceMethod::BoundedCheck,
                &[],
            );
            return Ok(EquivalenceResult::Equivalent { certificate: cert });
        } else {
            return Ok(EquivalenceResult::NotEquivalent {
                witness: DistinguishingWord::new(
                    vec![],
                    format!("{:?}", w1),
                    format!("{:?}", w2),
                ),
            });
        }
    }

    // BFS enumeration using forward vectors for efficiency.
    // Instead of re-computing weights from scratch for each word, we
    // propagate forward weight vectors incrementally.

    // Each entry in the queue is (word, forward_vec_wfa1, forward_vec_wfa2).
    let n1 = wfa1.state_count();
    let n2 = wfa2.state_count();

    let init_fwd1: Vec<S> = wfa1.initial_weights().to_vec();
    let init_fwd2: Vec<S> = wfa2.initial_weights().to_vec();

    // Check empty word
    let empty_w1 = dot_product(&init_fwd1, wfa1.final_weights());
    let empty_w2 = dot_product(&init_fwd2, wfa2.final_weights());
    if empty_w1 != empty_w2 {
        return Ok(EquivalenceResult::NotEquivalent {
            witness: DistinguishingWord::new(
                vec![],
                format!("{:?}", empty_w1),
                format!("{:?}", empty_w2),
            ),
        });
    }

    // BFS with incremental forward vectors
    struct BfsEntry<S: Semiring> {
        word: Vec<usize>,
        fwd1: Vec<S>,
        fwd2: Vec<S>,
    }

    let mut queue: VecDeque<BfsEntry<S>> = VecDeque::new();
    queue.push_back(BfsEntry {
        word: vec![],
        fwd1: init_fwd1,
        fwd2: init_fwd2,
    });

    let mut total_checked: usize = 0;

    while let Some(entry) = queue.pop_front() {
        total_checked += 1;

        if entry.word.len() >= max_depth {
            continue;
        }

        for sym in 0..alpha_size {
            // Advance forward vectors by one symbol
            let new_fwd1 = advance_forward(&entry.fwd1, wfa1, sym);
            let new_fwd2 = advance_forward(&entry.fwd2, wfa2, sym);

            // Compute weight on this new word
            let w1 = dot_product(&new_fwd1, wfa1.final_weights());
            let w2 = dot_product(&new_fwd2, wfa2.final_weights());

            let mut new_word = entry.word.clone();
            new_word.push(sym);

            if w1 != w2 {
                return Ok(EquivalenceResult::NotEquivalent {
                    witness: DistinguishingWord::new(
                        new_word,
                        format!("{:?}", w1),
                        format!("{:?}", w2),
                    ),
                });
            }

            // Only continue BFS if the forward vectors are non-zero
            let any_nonzero1 = new_fwd1.iter().any(|w| !w.is_zero());
            let any_nonzero2 = new_fwd2.iter().any(|w| !w.is_zero());
            if any_nonzero1 || any_nonzero2 {
                queue.push_back(BfsEntry {
                    word: new_word,
                    fwd1: new_fwd1,
                    fwd2: new_fwd2,
                });
            }
        }
    }

    // All words up to max_depth agree
    let cert = EquivalenceCertificate {
        method: EquivalenceMethod::BoundedCheck,
        bisimulation_relation: None,
        checked_depth: max_depth,
        total_pairs_checked: total_checked,
        hash: compute_certificate_hash(
            &EquivalenceMethod::BoundedCheck,
            total_checked,
            max_depth,
            &None,
        ),
        timestamp: timestamp_now(),
    };
    Ok(EquivalenceResult::Equivalent { certificate: cert })
}

/// Advance a forward weight vector by one symbol.
fn advance_forward<S: Semiring>(
    fwd: &[S],
    wfa: &WeightedFiniteAutomaton<S>,
    sym: usize,
) -> Vec<S> {
    let n = wfa.state_count();
    let mut new_fwd = vec![S::zero(); n];
    for from in 0..n {
        if fwd[from].is_zero() {
            continue;
        }
        for &(to, ref w) in wfa.transitions_from(from, sym) {
            let contrib = fwd[from].mul(w);
            new_fwd[to].add_assign(&contrib);
        }
    }
    new_fwd
}

/// Dot product of two weight vectors.
fn dot_product<S: Semiring>(a: &[S], b: &[S]) -> S {
    let mut total = S::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let prod = x.mul(y);
        total.add_assign(&prod);
    }
    total
}

// ===========================================================================
// 5. Random sampling equivalence
// ===========================================================================

/// Random word generator with configurable length distribution.
pub struct RandomWordGenerator {
    /// Maximum word length.
    pub max_length: usize,
    /// Alphabet size.
    pub alphabet_size: usize,
}

impl RandomWordGenerator {
    pub fn new(alphabet_size: usize, max_length: usize) -> Self {
        RandomWordGenerator {
            max_length,
            alphabet_size,
        }
    }

    /// Generate a single random word using the provided RNG.
    pub fn generate(&self, rng: &mut impl Rng) -> Vec<usize> {
        generate_random_word(self.alphabet_size, self.max_length, rng)
    }

    /// Generate `count` random words.
    pub fn generate_batch(&self, count: usize, rng: &mut impl Rng) -> Vec<Vec<usize>> {
        (0..count).map(|_| self.generate(rng)).collect()
    }
}

/// Generate a random word over an alphabet of the given size.
///
/// The word length is chosen uniformly at random in `[0, max_length]`.
pub fn generate_random_word(alphabet_size: usize, max_length: usize, rng: &mut impl Rng) -> Vec<usize> {
    if alphabet_size == 0 {
        return vec![];
    }
    let len = rng.gen_range(0..=max_length);
    (0..len).map(|_| rng.gen_range(0..alphabet_size)).collect()
}

/// Probabilistic equivalence check via random sampling.
pub fn check_random_sampling<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    config: &EquivalenceConfig,
) -> EquivResult<EquivalenceResult> {
    validate_alphabets(wfa1, wfa2)?;

    let alpha_size = wfa1.alphabet().size();
    let mut rng = rand::thread_rng();
    let gen = RandomWordGenerator::new(alpha_size, config.max_word_length);

    // Check empty word first
    let w1_empty = wfa1.compute_weight(&[]);
    let w2_empty = wfa2.compute_weight(&[]);
    if w1_empty != w2_empty {
        return Ok(EquivalenceResult::NotEquivalent {
            witness: DistinguishingWord::new(
                vec![],
                format!("{:?}", w1_empty),
                format!("{:?}", w2_empty),
            ),
        });
    }

    for _ in 0..config.num_samples {
        let word = gen.generate(&mut rng);
        let w1 = wfa1.compute_weight(&word);
        let w2 = wfa2.compute_weight(&word);

        if w1 != w2 {
            return Ok(EquivalenceResult::NotEquivalent {
                witness: DistinguishingWord::new(
                    word.clone(),
                    format!("{:?}", w1),
                    format!("{:?}", w2),
                ),
            });
        }
    }

    // All samples matched → probably equivalent (probabilistic result).
    let cert = EquivalenceCertificate {
        method: EquivalenceMethod::RandomSampling,
        bisimulation_relation: None,
        checked_depth: config.max_word_length,
        total_pairs_checked: config.num_samples,
        hash: compute_certificate_hash(
            &EquivalenceMethod::RandomSampling,
            config.num_samples,
            config.max_word_length,
            &None,
        ),
        timestamp: timestamp_now(),
    };
    Ok(EquivalenceResult::Equivalent { certificate: cert })
}

// ===========================================================================
// 6. Approximate equivalence
// ===========================================================================

/// Check whether two WFAs are approximately equivalent within a tolerance.
///
/// Uses random sampling to estimate the maximum weight difference across
/// all words.
pub fn check_approximate<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    tolerance: f64,
    config: &EquivalenceConfig,
) -> EquivResult<ApproximateResult> {
    validate_alphabets(wfa1, wfa2)?;

    let alpha_size = wfa1.alphabet().size();
    let mut rng = rand::thread_rng();
    let gen = RandomWordGenerator::new(alpha_size, config.max_word_length);

    let mut max_diff: f64 = 0.0;
    let mut worst_word: Vec<usize> = vec![];
    let mut samples_checked: usize = 0;

    // Check empty word
    let w1 = wfa1.compute_weight(&[]);
    let w2 = wfa2.compute_weight(&[]);
    let diff = weight_distance(&w1, &w2);
    if diff > max_diff {
        max_diff = diff;
        worst_word = vec![];
    }
    samples_checked += 1;

    // Also do bounded enumeration for short words
    let short_depth = config.max_word_length.min(5);
    let mut short_words: Vec<Vec<usize>> = vec![vec![]];
    for _depth in 0..short_depth {
        let mut next = Vec::new();
        for word in &short_words {
            for sym in 0..alpha_size {
                let mut w = word.clone();
                w.push(sym);
                next.push(w);
            }
        }
        short_words.extend(next.clone());
    }
    for word in &short_words {
        let w1 = wfa1.compute_weight(word);
        let w2 = wfa2.compute_weight(word);
        let diff = weight_distance(&w1, &w2);
        if diff > max_diff {
            max_diff = diff;
            worst_word = word.clone();
        }
        samples_checked += 1;
    }

    // Random sampling for longer words
    for _ in 0..config.num_samples {
        let word = gen.generate(&mut rng);
        let w1 = wfa1.compute_weight(&word);
        let w2 = wfa2.compute_weight(&word);
        let diff = weight_distance(&w1, &w2);
        if diff > max_diff {
            max_diff = diff;
            worst_word = word;
        }
        samples_checked += 1;
    }

    Ok(ApproximateResult {
        is_approximate: max_diff <= tolerance,
        max_difference: max_diff,
        worst_word,
        samples_checked,
    })
}

/// Compute a numeric distance between two semiring weights.
///
/// Uses the `Debug` formatting to extract a number; falls back to 0.0 if
/// the weights are equal, 1.0 if they differ.
fn weight_distance<S: Semiring>(a: &S, b: &S) -> f64 {
    if a == b {
        return 0.0;
    }
    // Try to parse the debug representation as f64
    let sa = format!("{:?}", a);
    let sb = format!("{:?}", b);

    // Attempt to extract numeric values
    let va = extract_numeric(&sa);
    let vb = extract_numeric(&sb);

    match (va, vb) {
        (Some(fa), Some(fb)) => (fa - fb).abs(),
        _ => 1.0, // Non-numeric weights that differ
    }
}

/// Try to extract a numeric value from a Debug-formatted semiring element.
fn extract_numeric(s: &str) -> Option<f64> {
    // Common patterns: "RealSemiring(3.14)", "CountingSemiring(42)", bare "3.14"
    let trimmed = s.trim();

    // Try to find a number inside parentheses
    if let Some(start) = trimmed.find('(') {
        if let Some(end) = trimmed.rfind(')') {
            if start < end {
                let inner = &trimmed[start + 1..end];
                if let Ok(v) = inner.parse::<f64>() {
                    return Some(v);
                }
            }
        }
    }

    // Try direct parse
    if let Ok(v) = trimmed.parse::<f64>() {
        return Some(v);
    }

    None
}

// ===========================================================================
// 7. Distinguishing word generation
// ===========================================================================

/// Find a distinguishing word using BFS on the product state space.
///
/// Returns `None` if the WFAs are equivalent.
pub fn find_distinguishing_word<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> Option<DistinguishingWord> {
    let config = EquivalenceConfig::default();
    find_distinguishing_word_impl(wfa1, wfa2, &config)
}

fn find_distinguishing_word_impl<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    config: &EquivalenceConfig,
) -> Option<DistinguishingWord> {
    if wfa1.alphabet().size() != wfa2.alphabet().size() {
        return None;
    }

    let alpha_size = wfa1.alphabet().size();
    let n1 = wfa1.state_count();
    let n2 = wfa2.state_count();

    // Check empty word
    let w1_empty = wfa1.compute_weight(&[]);
    let w2_empty = wfa2.compute_weight(&[]);
    if w1_empty != w2_empty {
        return Some(DistinguishingWord::new(
            vec![],
            format!("{:?}", w1_empty),
            format!("{:?}", w2_empty),
        ));
    }

    // BFS through product space using forward vectors.
    // We store (fwd1, fwd2, word) at each BFS level.
    struct Entry<S: Semiring> {
        fwd1: Vec<S>,
        fwd2: Vec<S>,
        word: Vec<usize>,
    }

    let init_fwd1: Vec<S> = wfa1.initial_weights().to_vec();
    let init_fwd2: Vec<S> = wfa2.initial_weights().to_vec();

    let mut queue: VecDeque<Entry<S>> = VecDeque::new();
    queue.push_back(Entry {
        fwd1: init_fwd1,
        fwd2: init_fwd2,
        word: vec![],
    });

    let mut steps: usize = 0;

    while let Some(entry) = queue.pop_front() {
        steps += 1;
        if steps > config.timeout_steps {
            break;
        }

        if entry.word.len() >= config.max_depth {
            continue;
        }

        for sym in 0..alpha_size {
            let new_fwd1 = advance_forward(&entry.fwd1, wfa1, sym);
            let new_fwd2 = advance_forward(&entry.fwd2, wfa2, sym);

            let w1 = dot_product(&new_fwd1, wfa1.final_weights());
            let w2 = dot_product(&new_fwd2, wfa2.final_weights());

            let mut new_word = entry.word.clone();
            new_word.push(sym);

            if w1 != w2 {
                return Some(DistinguishingWord::new(
                    new_word,
                    format!("{:?}", w1),
                    format!("{:?}", w2),
                ));
            }

            let any1 = new_fwd1.iter().any(|w| !w.is_zero());
            let any2 = new_fwd2.iter().any(|w| !w.is_zero());
            if any1 || any2 {
                queue.push_back(Entry {
                    fwd1: new_fwd1,
                    fwd2: new_fwd2,
                    word: new_word,
                });
            }
        }
    }

    None
}

/// Find the shortest distinguishing word.
///
/// Guaranteed shortest by BFS (breadth-first search processes words in
/// length order).
pub fn find_shortest_distinguishing_word<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> Option<DistinguishingWord> {
    // BFS already finds shortest; just call the implementation with a
    // generous depth bound.
    let config = EquivalenceConfig {
        max_depth: 1000,
        timeout_steps: 5_000_000,
        ..EquivalenceConfig::default()
    };
    find_distinguishing_word_impl(wfa1, wfa2, &config)
}

/// Find up to `count` distinguishing words.
pub fn find_distinguishing_set<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    count: usize,
) -> Vec<DistinguishingWord> {
    if count == 0 {
        return vec![];
    }
    if wfa1.alphabet().size() != wfa2.alphabet().size() {
        return vec![];
    }

    let alpha_size = wfa1.alphabet().size();
    let config = EquivalenceConfig::default();
    let mut results: Vec<DistinguishingWord> = Vec::new();

    // Check empty word
    let w1_empty = wfa1.compute_weight(&[]);
    let w2_empty = wfa2.compute_weight(&[]);
    if w1_empty != w2_empty {
        results.push(DistinguishingWord::new(
            vec![],
            format!("{:?}", w1_empty),
            format!("{:?}", w2_empty),
        ));
        if results.len() >= count {
            return results;
        }
    }

    // BFS
    struct Entry<S: Semiring> {
        fwd1: Vec<S>,
        fwd2: Vec<S>,
        word: Vec<usize>,
    }

    let init_fwd1: Vec<S> = wfa1.initial_weights().to_vec();
    let init_fwd2: Vec<S> = wfa2.initial_weights().to_vec();

    let mut queue: VecDeque<Entry<S>> = VecDeque::new();
    queue.push_back(Entry {
        fwd1: init_fwd1,
        fwd2: init_fwd2,
        word: vec![],
    });

    let mut steps: usize = 0;

    while let Some(entry) = queue.pop_front() {
        steps += 1;
        if steps > config.timeout_steps || results.len() >= count {
            break;
        }
        if entry.word.len() >= config.max_depth {
            continue;
        }

        for sym in 0..alpha_size {
            let new_fwd1 = advance_forward(&entry.fwd1, wfa1, sym);
            let new_fwd2 = advance_forward(&entry.fwd2, wfa2, sym);

            let w1 = dot_product(&new_fwd1, wfa1.final_weights());
            let w2 = dot_product(&new_fwd2, wfa2.final_weights());

            let mut new_word = entry.word.clone();
            new_word.push(sym);

            if w1 != w2 {
                results.push(DistinguishingWord::new(
                    new_word.clone(),
                    format!("{:?}", w1),
                    format!("{:?}", w2),
                ));
                if results.len() >= count {
                    return results;
                }
            }

            let any1 = new_fwd1.iter().any(|w| !w.is_zero());
            let any2 = new_fwd2.iter().any(|w| !w.is_zero());
            if any1 || any2 {
                queue.push_back(Entry {
                    fwd1: new_fwd1,
                    fwd2: new_fwd2,
                    word: new_word,
                });
            }
        }
    }

    results
}

// ===========================================================================
// 8. Language / series inclusion
// ===========================================================================

/// Check whether the formal power series of `wfa1` is point-wise ≤ that
/// of `wfa2`.
///
/// This is checked via bounded enumeration and random sampling; for finite
/// automata over the Boolean semiring it reduces to language inclusion.
///
/// NOTE: this is a *heuristic* for general semirings—proving inclusion over
/// all strings is undecidable in general.
pub fn check_inclusion<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> EquivResult<bool> {
    validate_alphabets(wfa1, wfa2)?;

    let alpha_size = wfa1.alphabet().size();
    let config = EquivalenceConfig::default();
    let mut rng = rand::thread_rng();
    let gen = RandomWordGenerator::new(alpha_size, config.max_word_length);

    // Check a bounded set of short words
    let short_depth = 8.min(config.max_depth);
    let mut words: Vec<Vec<usize>> = vec![vec![]];
    for _d in 0..short_depth {
        let mut next = Vec::new();
        for word in &words {
            for sym in 0..alpha_size {
                let mut w = word.clone();
                w.push(sym);
                next.push(w);
            }
        }
        // Limit size to prevent exponential blowup
        if next.len() > 100_000 {
            break;
        }
        words.extend(next);
    }

    for word in &words {
        let w1 = wfa1.compute_weight(word);
        let w2 = wfa2.compute_weight(word);
        if !weight_leq(&w1, &w2) {
            return Ok(false);
        }
    }

    // Random sampling for longer words
    for _ in 0..config.num_samples.min(10_000) {
        let word = gen.generate(&mut rng);
        let w1 = wfa1.compute_weight(&word);
        let w2 = wfa2.compute_weight(&word);
        if !weight_leq(&w1, &w2) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Check strict inclusion: wfa1 ≤ wfa2 and wfa1 ≠ wfa2.
pub fn check_strict_inclusion<S: Semiring + Ord>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
) -> EquivResult<bool> {
    if !check_inclusion(wfa1, wfa2)? {
        return Ok(false);
    }

    // Check that there exists at least one word where the weights differ.
    let config = EquivalenceConfig::default();
    let result = check_equivalence_with_config(wfa1, wfa2, &config)?;
    Ok(result.is_not_equivalent())
}

/// Heuristic ≤ comparison for semiring weights.
///
/// Uses debug formatting to extract numeric values; for non-numeric
/// semirings falls back to equality (i.e., `a ≤ b iff a == b`).
fn weight_leq<S: Semiring>(a: &S, b: &S) -> bool {
    if a == b {
        return true;
    }
    let sa = format!("{:?}", a);
    let sb = format!("{:?}", b);
    match (extract_numeric(&sa), extract_numeric(&sb)) {
        (Some(fa), Some(fb)) => fa <= fb,
        _ => false,
    }
}

// ===========================================================================
// 9. Builder-pattern checker
// ===========================================================================

/// Builder-pattern struct for configuring and running equivalence checks.
pub struct EquivalenceChecker<'a, S: Semiring> {
    wfa1: &'a WeightedFiniteAutomaton<S>,
    wfa2: &'a WeightedFiniteAutomaton<S>,
    config: EquivalenceConfig,
}

impl<'a, S: Semiring + Ord> EquivalenceChecker<'a, S> {
    /// Create a new checker for two WFAs.
    pub fn new(
        wfa1: &'a WeightedFiniteAutomaton<S>,
        wfa2: &'a WeightedFiniteAutomaton<S>,
    ) -> Self {
        EquivalenceChecker {
            wfa1,
            wfa2,
            config: EquivalenceConfig::default(),
        }
    }

    /// Set the equivalence-checking method.
    pub fn method(mut self, m: EquivalenceMethod) -> Self {
        self.config.method = m;
        self
    }

    /// Set the maximum depth for bounded checks.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.config.max_depth = d;
        self
    }

    /// Set the number of random samples.
    pub fn num_samples(mut self, n: usize) -> Self {
        self.config.num_samples = n;
        self
    }

    /// Set the maximum word length for sampling.
    pub fn max_word_length(mut self, l: usize) -> Self {
        self.config.max_word_length = l;
        self
    }

    /// Set the timeout step budget.
    pub fn timeout_steps(mut self, t: usize) -> Self {
        self.config.timeout_steps = t;
        self
    }

    /// Set the tolerance for approximate checks.
    pub fn tolerance(mut self, t: f64) -> Self {
        self.config.tolerance = t;
        self
    }

    /// Run the configured equivalence check.
    pub fn check(self) -> EquivResult<EquivalenceResult> {
        check_equivalence_with_config(self.wfa1, self.wfa2, &self.config)
    }

    /// Run an approximate equivalence check.
    pub fn check_approximate(self) -> EquivResult<ApproximateResult> {
        check_approximate(self.wfa1, self.wfa2, self.config.tolerance, &self.config)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::automaton::{Alphabet, Symbol, WeightedFiniteAutomaton};
    use super::super::semiring::{
        BooleanSemiring, CountingSemiring, RealSemiring, TropicalSemiring,
    };

    // ── helpers ────────────────────────────────────────────────────────

    /// Build a tiny 2-state WFA over the Boolean semiring that accepts
    /// words starting with symbol 0.
    fn bool_wfa_accepts_starts_with_0() -> WeightedFiniteAutomaton<BooleanSemiring> {
        // states: 0 (initial), 1 (accepting)
        // alphabet: {0, 1}
        let alpha = Alphabet::from_range(2);
        let mut wfa = WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, BooleanSemiring(true));
        wfa.set_final_weight(1, BooleanSemiring(true));
        let _ = wfa.add_transition(0, 0, 1, BooleanSemiring(true)); // 0 --[0]--> 1
        let _ = wfa.add_transition(1, 0, 1, BooleanSemiring(true)); // loop on 0
        let _ = wfa.add_transition(1, 1, 1, BooleanSemiring(true)); // loop on 1
        wfa
    }

    /// Build a WFA that accepts all words (over {0,1}).
    fn bool_wfa_accepts_all() -> WeightedFiniteAutomaton<BooleanSemiring> {
        let alpha = Alphabet::from_range(2);
        let mut wfa = WeightedFiniteAutomaton::new(1, alpha);
        wfa.set_initial_weight(0, BooleanSemiring(true));
        wfa.set_final_weight(0, BooleanSemiring(true));
        let _ = wfa.add_transition(0, 0, 0, BooleanSemiring(true));
        let _ = wfa.add_transition(0, 1, 0, BooleanSemiring(true));
        wfa
    }

    /// Build two identical counting WFAs.
    fn counting_wfa_pair() -> (
        WeightedFiniteAutomaton<CountingSemiring>,
        WeightedFiniteAutomaton<CountingSemiring>,
    ) {
        let alpha = Alphabet::from_range(2);
        let mut wfa1 = WeightedFiniteAutomaton::new(2, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 1, 0, CountingSemiring(1));
        let _ = wfa1.add_transition(1, 0, 1, CountingSemiring(1));
        let _ = wfa1.add_transition(1, 1, 1, CountingSemiring(1));

        let wfa2 = wfa1.clone();
        (wfa1, wfa2)
    }

    /// Build two counting WFAs that differ on symbol sequence [0].
    fn counting_wfa_different() -> (
        WeightedFiniteAutomaton<CountingSemiring>,
        WeightedFiniteAutomaton<CountingSemiring>,
    ) {
        let alpha = Alphabet::from_range(2);
        let mut wfa1 = WeightedFiniteAutomaton::new(2, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(2, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(1, CountingSemiring(1));
        let _ = wfa2.add_transition(0, 0, 1, CountingSemiring(2)); // weight 2 vs 1

        (wfa1, wfa2)
    }

    /// Single-state WFA with self-loops (counting semiring).
    fn single_state_counting_wfa(w0: u64, w1: u64) -> WeightedFiniteAutomaton<CountingSemiring> {
        let alpha = Alphabet::from_range(2);
        let mut wfa = WeightedFiniteAutomaton::new(1, alpha);
        wfa.set_initial_weight(0, CountingSemiring(1));
        wfa.set_final_weight(0, CountingSemiring(1));
        let _ = wfa.add_transition(0, 0, 0, CountingSemiring(w0));
        let _ = wfa.add_transition(0, 1, 0, CountingSemiring(w1));
        wfa
    }

    /// Build a real-semiring WFA.
    fn real_wfa(
        init: f64,
        fin_w: f64,
        t_weight: f64,
    ) -> WeightedFiniteAutomaton<RealSemiring> {
        let alpha = Alphabet::from_range(1);
        let mut wfa = WeightedFiniteAutomaton::new(1, alpha);
        wfa.set_initial_weight(0, RealSemiring(init));
        wfa.set_final_weight(0, RealSemiring(fin_w));
        let _ = wfa.add_transition(0, 0, 0, RealSemiring(t_weight));
        wfa
    }

    /// Build a tropical WFA.
    fn tropical_wfa(
        init: f64,
        fin_w: f64,
        t_weight: f64,
    ) -> WeightedFiniteAutomaton<TropicalSemiring> {
        let alpha = Alphabet::from_range(1);
        let mut wfa = WeightedFiniteAutomaton::new(1, alpha);
        wfa.set_initial_weight(0, TropicalSemiring(init));
        wfa.set_final_weight(0, TropicalSemiring(fin_w));
        let _ = wfa.add_transition(0, 0, 0, TropicalSemiring(t_weight));
        wfa
    }

    // ── tests: identical WFAs are equivalent ──────────────────────────

    #[test]
    fn test_identical_wfas_bisimulation() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::Bisimulation);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_equivalent(), "Identical WFAs should be equivalent");
    }

    #[test]
    fn test_identical_wfas_hopcroft_karp() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::HopcroftKarp);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_identical_wfas_bounded() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let result = check_bounded(&wfa1, &wfa2, 6).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_identical_wfas_random_sampling() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let config = EquivalenceConfig::new()
            .with_method(EquivalenceMethod::RandomSampling)
            .with_num_samples(500);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_equivalent());
    }

    // ── tests: different WFAs are not equivalent ─────────────────────

    #[test]
    fn test_different_wfas_bisimulation() {
        let (wfa1, wfa2) = counting_wfa_different();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::Bisimulation);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_not_equivalent());

        let witness = result.witness().unwrap();
        assert_eq!(witness.word, vec![0]); // They differ on [0]
    }

    #[test]
    fn test_different_wfas_hopcroft_karp() {
        let (wfa1, wfa2) = counting_wfa_different();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::HopcroftKarp);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_not_equivalent());
    }

    #[test]
    fn test_different_wfas_bounded() {
        let (wfa1, wfa2) = counting_wfa_different();
        let result = check_bounded(&wfa1, &wfa2, 5).unwrap();
        assert!(result.is_not_equivalent());

        let witness = result.witness().unwrap();
        assert!(witness.length <= 5);
    }

    #[test]
    fn test_different_wfas_random_finds_difference() {
        let (wfa1, wfa2) = counting_wfa_different();
        let config = EquivalenceConfig::new()
            .with_method(EquivalenceMethod::RandomSampling)
            .with_num_samples(1000);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        // Random sampling should very likely find the difference on [0]
        assert!(result.is_not_equivalent());
    }

    // ── tests: bisimulation finds correct relation ───────────────────

    #[test]
    fn test_bisimulation_relation() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::Bisimulation);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        let cert = result.certificate().unwrap();
        assert_eq!(cert.method, EquivalenceMethod::Bisimulation);
        assert!(cert.bisimulation_relation.is_some());
        let relation = cert.bisimulation_relation.as_ref().unwrap();
        assert!(!relation.is_empty());
    }

    // ── tests: bounded check matches exact check on small examples ───

    #[test]
    fn test_bounded_agrees_with_bisimulation() {
        // Equivalent pair
        let (wfa1, wfa2) = counting_wfa_pair();
        let bisim = check_via_bisimulation(
            &wfa1,
            &wfa2,
            &EquivalenceConfig::default(),
        )
        .unwrap();
        let bounded = check_bounded(&wfa1, &wfa2, 8).unwrap();
        assert_eq!(bisim.is_equivalent(), bounded.is_equivalent());

        // Non-equivalent pair
        let (wfa1, wfa2) = counting_wfa_different();
        let bisim = check_via_bisimulation(
            &wfa1,
            &wfa2,
            &EquivalenceConfig::default(),
        )
        .unwrap();
        let bounded = check_bounded(&wfa1, &wfa2, 8).unwrap();
        assert_eq!(bisim.is_equivalent(), bounded.is_equivalent());
    }

    // ── tests: certificate generation & verification ─────────────────

    #[test]
    fn test_certificate_generation() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let relation = vec![(0, 0), (1, 1)];
        let cert = generate_equivalence_certificate(
            &wfa1,
            &wfa2,
            EquivalenceMethod::Bisimulation,
            &relation,
        );
        assert_eq!(cert.method, EquivalenceMethod::Bisimulation);
        assert!(!cert.hash.is_empty());
        assert_eq!(cert.total_pairs_checked, 2);
    }

    #[test]
    fn test_certificate_verification_valid() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::Bisimulation);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        let cert = result.certificate().unwrap();

        let valid = verify_equivalence_certificate(&wfa1, &wfa2, cert).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_certificate_verification_bad_hash() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let mut cert = generate_equivalence_certificate(
            &wfa1,
            &wfa2,
            EquivalenceMethod::Bisimulation,
            &[(0, 0)],
        );
        cert.hash = "bad_hash".into();
        let result = verify_equivalence_certificate(&wfa1, &wfa2, &cert);
        assert!(result.is_err());
    }

    // ── tests: approximate equivalence ───────────────────────────────

    #[test]
    fn test_approximate_equivalent() {
        let wfa1 = real_wfa(1.0, 1.0, 0.5);
        let wfa2 = real_wfa(1.0, 1.0, 0.5);
        let config = EquivalenceConfig::new().with_num_samples(100);
        let result = check_approximate(&wfa1, &wfa2, 0.01, &config).unwrap();
        assert!(result.is_approximate);
        assert_eq!(result.max_difference, 0.0);
    }

    #[test]
    fn test_approximate_close_but_not_exact() {
        let wfa1 = real_wfa(1.0, 1.0, 0.5);
        let wfa2 = real_wfa(1.0, 1.0, 0.500001);
        let config = EquivalenceConfig::new()
            .with_num_samples(200)
            .with_max_word_length(3);
        let result = check_approximate(&wfa1, &wfa2, 0.01, &config).unwrap();
        // Differences should be tiny
        assert!(result.max_difference < 0.01);
    }

    #[test]
    fn test_approximate_not_close() {
        let wfa1 = real_wfa(1.0, 1.0, 0.5);
        let wfa2 = real_wfa(1.0, 1.0, 2.0);
        let config = EquivalenceConfig::new()
            .with_num_samples(100)
            .with_max_word_length(3);
        let result = check_approximate(&wfa1, &wfa2, 0.01, &config).unwrap();
        assert!(!result.is_approximate);
        assert!(result.max_difference > 0.01);
    }

    // ── tests: shortest distinguishing word ──────────────────────────

    #[test]
    fn test_shortest_distinguishing_word() {
        let (wfa1, wfa2) = counting_wfa_different();
        let dw = find_shortest_distinguishing_word(&wfa1, &wfa2);
        assert!(dw.is_some());
        let dw = dw.unwrap();
        assert_eq!(dw.word, vec![0]);
        assert_eq!(dw.length, 1);
    }

    #[test]
    fn test_no_distinguishing_word_for_equivalent() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let dw = find_shortest_distinguishing_word(&wfa1, &wfa2);
        assert!(dw.is_none());
    }

    #[test]
    fn test_distinguishing_set() {
        let alpha = Alphabet::from_range(2);
        let mut wfa1 = WeightedFiniteAutomaton::new(2, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 1, 1, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(2, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(1, CountingSemiring(1));
        let _ = wfa2.add_transition(0, 0, 1, CountingSemiring(2)); // differs on sym 0
        let _ = wfa2.add_transition(0, 1, 1, CountingSemiring(3)); // differs on sym 1

        let set = find_distinguishing_set(&wfa1, &wfa2, 5);
        assert!(set.len() >= 2);
    }

    // ── tests: inclusion checking ────────────────────────────────────

    #[test]
    fn test_inclusion_identical() {
        let (wfa1, wfa2) = counting_wfa_pair();
        assert!(check_inclusion(&wfa1, &wfa2).unwrap());
    }

    #[test]
    fn test_inclusion_subset() {
        // wfa1: weight 1 on symbol 0
        // wfa2: weight 2 on symbol 0
        // => wfa1 ≤ wfa2
        let alpha = Alphabet::from_range(1);
        let mut wfa1 = WeightedFiniteAutomaton::new(2, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(2, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(1, CountingSemiring(1));
        let _ = wfa2.add_transition(0, 0, 1, CountingSemiring(2));

        assert!(check_inclusion(&wfa1, &wfa2).unwrap());
        assert!(!check_inclusion(&wfa2, &wfa1).unwrap());
    }

    #[test]
    fn test_strict_inclusion() {
        let alpha = Alphabet::from_range(1);
        let mut wfa1 = WeightedFiniteAutomaton::new(2, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(2, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(1, CountingSemiring(1));
        let _ = wfa2.add_transition(0, 0, 1, CountingSemiring(2));

        assert!(check_strict_inclusion(&wfa1, &wfa2).unwrap());
        assert!(!check_strict_inclusion(&wfa2, &wfa1).unwrap());
    }

    // ── tests: edge cases ────────────────────────────────────────────

    #[test]
    fn test_empty_wfas() {
        let alpha = Alphabet::from_range(2);
        let wfa1: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(0, alpha.clone());
        let wfa2: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(0, alpha);
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_single_state_equivalent() {
        let wfa1 = single_state_counting_wfa(1, 1);
        let wfa2 = single_state_counting_wfa(1, 1);
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_single_state_different() {
        let wfa1 = single_state_counting_wfa(1, 1);
        let wfa2 = single_state_counting_wfa(1, 2);
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_not_equivalent());
    }

    #[test]
    fn test_incompatible_alphabets() {
        let wfa1: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(1, Alphabet::from_range(2));
        let wfa2: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(1, Alphabet::from_range(3));
        let result = check_equivalence(&wfa1, &wfa2);
        assert!(result.is_err());
        match result.unwrap_err() {
            EquivalenceError::IncompatibleAlphabets { .. } => {}
            e => panic!("Expected IncompatibleAlphabets, got {:?}", e),
        }
    }

    // ── tests: Boolean semiring ──────────────────────────────────────

    #[test]
    fn test_boolean_different_languages() {
        let wfa1 = bool_wfa_accepts_starts_with_0();
        let wfa2 = bool_wfa_accepts_all();
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_not_equivalent());
        // The witness should be a word starting with symbol 1
        let dw = result.witness().unwrap();
        assert!(dw.word.first() == Some(&1) || dw.word.is_empty());
    }

    #[test]
    fn test_boolean_identical() {
        let wfa1 = bool_wfa_accepts_all();
        let wfa2 = bool_wfa_accepts_all();
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
    }

    // ── tests: Tropical semiring ─────────────────────────────────────

    #[test]
    fn test_tropical_equivalent() {
        let wfa1 = tropical_wfa(0.0, 0.0, 1.0);
        let wfa2 = tropical_wfa(0.0, 0.0, 1.0);
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_tropical_different() {
        let wfa1 = tropical_wfa(0.0, 0.0, 1.0);
        let wfa2 = tropical_wfa(0.0, 0.0, 2.0);
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_not_equivalent());
    }

    // ── tests: Real semiring ─────────────────────────────────────────

    #[test]
    fn test_real_equivalent() {
        let wfa1 = real_wfa(1.0, 1.0, 0.5);
        let wfa2 = real_wfa(1.0, 1.0, 0.5);
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_real_different() {
        let wfa1 = real_wfa(1.0, 1.0, 0.5);
        let wfa2 = real_wfa(1.0, 1.0, 0.75);
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_not_equivalent());
    }

    // ── tests: builder pattern ───────────────────────────────────────

    #[test]
    fn test_builder_pattern_equivalent() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let result = EquivalenceChecker::new(&wfa1, &wfa2)
            .method(EquivalenceMethod::Bisimulation)
            .max_depth(100)
            .check()
            .unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_builder_pattern_not_equivalent() {
        let (wfa1, wfa2) = counting_wfa_different();
        let result = EquivalenceChecker::new(&wfa1, &wfa2)
            .method(EquivalenceMethod::HopcroftKarp)
            .check()
            .unwrap();
        assert!(result.is_not_equivalent());
    }

    #[test]
    fn test_builder_approximate() {
        let wfa1 = real_wfa(1.0, 1.0, 0.5);
        let wfa2 = real_wfa(1.0, 1.0, 0.5);
        let result = EquivalenceChecker::new(&wfa1, &wfa2)
            .tolerance(0.01)
            .num_samples(100)
            .check_approximate()
            .unwrap();
        assert!(result.is_approximate);
    }

    // ── tests: Hopcroft-Karp correctness ─────────────────────────────

    #[test]
    fn test_hopcroft_karp_multi_state() {
        // Two 3-state WFAs that are equivalent but have different structure
        let alpha = Alphabet::from_range(2);

        let mut wfa1 = WeightedFiniteAutomaton::new(3, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(2, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));
        let _ = wfa1.add_transition(1, 0, 2, CountingSemiring(1));
        let _ = wfa1.add_transition(1, 1, 2, CountingSemiring(1));

        // wfa2 is structurally identical
        let wfa2 = wfa1.clone();

        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::HopcroftKarp);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_hopcroft_karp_finds_difference() {
        let alpha = Alphabet::from_range(2);
        let mut wfa1 = WeightedFiniteAutomaton::new(2, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(2, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(1, CountingSemiring(3));
        let _ = wfa2.add_transition(0, 0, 1, CountingSemiring(1));

        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::HopcroftKarp);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_not_equivalent());
    }

    // ── tests: minimization-based ────────────────────────────────────

    #[test]
    fn test_minimization_equivalent() {
        // Two identical WFAs over CountingSemiring (which is Ord)
        let (wfa1, wfa2) = counting_wfa_pair();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::Minimization);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_minimization_not_equivalent() {
        let (wfa1, wfa2) = counting_wfa_different();
        let config = EquivalenceConfig::new().with_method(EquivalenceMethod::Minimization);
        let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
        assert!(result.is_not_equivalent());
    }

    // ── tests: isomorphism ───────────────────────────────────────────

    #[test]
    fn test_isomorphism_identical() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let mapping = check_isomorphism(&wfa1, &wfa2).unwrap();
        assert!(mapping.is_some());
        let m = mapping.unwrap();
        // Identity mapping
        assert_eq!(m, vec![0, 1]);
    }

    #[test]
    fn test_isomorphism_different_size() {
        let wfa1: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(2, Alphabet::from_range(1));
        let wfa2: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(3, Alphabet::from_range(1));
        let mapping = check_isomorphism(&wfa1, &wfa2).unwrap();
        assert!(mapping.is_none());
    }

    #[test]
    fn test_isomorphism_empty() {
        let wfa1: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(0, Alphabet::from_range(1));
        let wfa2: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(0, Alphabet::from_range(1));
        let mapping = check_isomorphism(&wfa1, &wfa2).unwrap();
        assert_eq!(mapping, Some(vec![]));
    }

    // ── tests: random word generator ─────────────────────────────────

    #[test]
    fn test_random_word_generator() {
        let gen = RandomWordGenerator::new(3, 10);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let word = gen.generate(&mut rng);
            assert!(word.len() <= 10);
            for &sym in &word {
                assert!(sym < 3);
            }
        }
    }

    #[test]
    fn test_random_word_empty_alphabet() {
        let mut rng = rand::thread_rng();
        let word = generate_random_word(0, 10, &mut rng);
        assert!(word.is_empty());
    }

    // ── tests: union-find ────────────────────────────────────────────

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert!(!uf.same(0, 1));
        uf.union(0, 1);
        assert!(uf.same(0, 1));
        assert!(!uf.same(0, 2));
        uf.union(1, 2);
        assert!(uf.same(0, 2));
    }

    #[test]
    fn test_union_find_single() {
        let mut uf = UnionFind::new(1);
        assert!(uf.same(0, 0));
    }

    #[test]
    fn test_weighted_union_find() {
        let weights = vec![
            CountingSemiring(1),
            CountingSemiring(1),
            CountingSemiring(2),
        ];
        let mut wuf = WeightedUnionFind::new(weights);

        // Same final weight → should succeed
        assert!(wuf.union(0, 1).is_ok());
        assert!(wuf.same(0, 1));

        // Different final weight → should fail
        assert!(wuf.union(0, 2).is_err());
        assert!(!wuf.same(0, 2));
    }

    // ── tests: default method ────────────────────────────────────────

    #[test]
    fn test_default_method_equivalent() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_default_method_not_equivalent() {
        let (wfa1, wfa2) = counting_wfa_different();
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_not_equivalent());
    }

    // ── tests: EquivalenceResult methods ─────────────────────────────

    #[test]
    fn test_result_accessors() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
        assert!(!result.is_not_equivalent());
        assert!(result.certificate().is_some());
        assert!(result.witness().is_none());
    }

    #[test]
    fn test_result_accessors_not_equiv() {
        let (wfa1, wfa2) = counting_wfa_different();
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(!result.is_equivalent());
        assert!(result.is_not_equivalent());
        assert!(result.certificate().is_none());
        assert!(result.witness().is_some());
    }

    // ── tests: Display impls ─────────────────────────────────────────

    #[test]
    fn test_display_equivalence_method() {
        assert_eq!(format!("{}", EquivalenceMethod::Bisimulation), "Bisimulation");
        assert_eq!(format!("{}", EquivalenceMethod::HopcroftKarp), "HopcroftKarp");
    }

    #[test]
    fn test_display_distinguishing_word() {
        let dw = DistinguishingWord::new(vec![0, 1], "3".into(), "5".into());
        let s = format!("{}", dw);
        assert!(s.contains("[0, 1]"));
        assert!(s.contains("wfa1=3"));
        assert!(s.contains("wfa2=5"));
    }

    #[test]
    fn test_display_certificate() {
        let cert = EquivalenceCertificate {
            method: EquivalenceMethod::Bisimulation,
            bisimulation_relation: None,
            checked_depth: 10,
            total_pairs_checked: 42,
            hash: "abc123".into(),
            timestamp: "T0".into(),
        };
        let s = format!("{}", cert);
        assert!(s.contains("Bisimulation"));
        assert!(s.contains("42"));
    }

    // ── tests: config builder ────────────────────────────────────────

    #[test]
    fn test_config_builder() {
        let config = EquivalenceConfig::new()
            .with_method(EquivalenceMethod::HopcroftKarp)
            .with_max_depth(500)
            .with_max_states(50_000)
            .with_num_samples(2000)
            .with_max_word_length(30)
            .with_timeout_steps(500_000)
            .with_tolerance(0.001);

        assert_eq!(config.method, EquivalenceMethod::HopcroftKarp);
        assert_eq!(config.max_depth, 500);
        assert_eq!(config.max_states, 50_000);
        assert_eq!(config.num_samples, 2000);
        assert_eq!(config.max_word_length, 30);
        assert_eq!(config.timeout_steps, 500_000);
        assert!((config.tolerance - 0.001).abs() < 1e-10);
    }

    // ── tests: multi-path WFA equivalence ────────────────────────────

    #[test]
    fn test_multipath_equivalent() {
        // wfa1: two paths from 0 to 2, each with weight 1 → total = 2
        // wfa2: one path from 0 to 1 with weight 2
        // They should agree on the word [0].
        let alpha = Alphabet::from_range(1);

        let mut wfa1 = WeightedFiniteAutomaton::new(3, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(2, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 2, CountingSemiring(1)); // two paths on sym 0
        let _ = wfa1.add_transition(1, 0, 2, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(3, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(2, CountingSemiring(1));
        let _ = wfa2.add_transition(0, 0, 1, CountingSemiring(1));
        let _ = wfa2.add_transition(0, 0, 2, CountingSemiring(1));
        let _ = wfa2.add_transition(1, 0, 2, CountingSemiring(1));

        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());
    }

    // ── tests: all methods agree on simple cases ─────────────────────

    #[test]
    fn test_all_methods_agree_equivalent() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let methods = [
            EquivalenceMethod::Bisimulation,
            EquivalenceMethod::HopcroftKarp,
            EquivalenceMethod::BoundedCheck,
            EquivalenceMethod::RandomSampling,
        ];
        for method in &methods {
            let config = EquivalenceConfig::new()
                .with_method(*method)
                .with_max_depth(10)
                .with_num_samples(500);
            let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
            assert!(
                result.is_equivalent(),
                "Method {:?} should find equivalence",
                method
            );
        }
    }

    #[test]
    fn test_all_methods_agree_not_equivalent() {
        let (wfa1, wfa2) = counting_wfa_different();
        let methods = [
            EquivalenceMethod::Bisimulation,
            EquivalenceMethod::HopcroftKarp,
            EquivalenceMethod::BoundedCheck,
            EquivalenceMethod::RandomSampling,
        ];
        for method in &methods {
            let config = EquivalenceConfig::new()
                .with_method(*method)
                .with_max_depth(10)
                .with_num_samples(500);
            let result = check_equivalence_with_config(&wfa1, &wfa2, &config).unwrap();
            assert!(
                result.is_not_equivalent(),
                "Method {:?} should find non-equivalence",
                method
            );
        }
    }

    // ── tests: WFA that differs only on long words ───────────────────

    #[test]
    fn test_difference_at_depth() {
        // Build two WFAs that agree on all words of length < 3 but differ
        // on a word of length 3.
        let alpha = Alphabet::from_range(1);

        let mut wfa1 = WeightedFiniteAutomaton::new(4, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(3, CountingSemiring(1));
        let _ = wfa1.add_transition(0, 0, 1, CountingSemiring(1));
        let _ = wfa1.add_transition(1, 0, 2, CountingSemiring(1));
        let _ = wfa1.add_transition(2, 0, 3, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(4, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(3, CountingSemiring(1));
        let _ = wfa2.add_transition(0, 0, 1, CountingSemiring(1));
        let _ = wfa2.add_transition(1, 0, 2, CountingSemiring(1));
        let _ = wfa2.add_transition(2, 0, 3, CountingSemiring(2)); // differs at depth 3

        // Bounded check with depth 2 should say equivalent
        let result = check_bounded(&wfa1, &wfa2, 2).unwrap();
        assert!(result.is_equivalent());

        // Bounded check with depth 3 should find the difference
        let result = check_bounded(&wfa1, &wfa2, 3).unwrap();
        assert!(result.is_not_equivalent());
        let witness = result.witness().unwrap();
        assert_eq!(witness.word, vec![0, 0, 0]);
    }

    // ── tests: empty word only ───────────────────────────────────────

    #[test]
    fn test_differ_on_empty_word() {
        let alpha = Alphabet::from_range(1);
        let mut wfa1 = WeightedFiniteAutomaton::new(1, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring(1));
        wfa1.set_final_weight(0, CountingSemiring(1));

        let mut wfa2 = WeightedFiniteAutomaton::new(1, alpha);
        wfa2.set_initial_weight(0, CountingSemiring(1));
        wfa2.set_final_weight(0, CountingSemiring(2));

        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_not_equivalent());
        let witness = result.witness().unwrap();
        assert_eq!(witness.word, vec![]);
    }

    // ── tests: certificate hash determinism ──────────────────────────

    #[test]
    fn test_certificate_hash_deterministic() {
        let h1 = compute_certificate_hash(
            &EquivalenceMethod::Bisimulation,
            10,
            5,
            &Some(vec![(0, 0), (1, 1)]),
        );
        let h2 = compute_certificate_hash(
            &EquivalenceMethod::Bisimulation,
            10,
            5,
            &Some(vec![(0, 0), (1, 1)]),
        );
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_certificate_hash_different_inputs() {
        let h1 = compute_certificate_hash(
            &EquivalenceMethod::Bisimulation,
            10,
            5,
            &None,
        );
        let h2 = compute_certificate_hash(
            &EquivalenceMethod::HopcroftKarp,
            10,
            5,
            &None,
        );
        assert_ne!(h1, h2);
    }

    // ── tests: weight_distance helper ────────────────────────────────

    #[test]
    fn test_weight_distance_zero() {
        let a = RealSemiring(3.0);
        let b = RealSemiring(3.0);
        assert_eq!(weight_distance(&a, &b), 0.0);
    }

    #[test]
    fn test_weight_distance_nonzero() {
        let a = RealSemiring(3.0);
        let b = RealSemiring(5.0);
        let d = weight_distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_weight_distance_boolean() {
        let a = BooleanSemiring(true);
        let b = BooleanSemiring(false);
        let d = weight_distance(&a, &b);
        // Non-numeric; should be 1.0
        assert_eq!(d, 1.0);
    }

    // ── tests: advance_forward / dot_product ─────────────────────────

    #[test]
    fn test_advance_forward_simple() {
        let alpha = Alphabet::from_range(1);
        let mut wfa = WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, CountingSemiring(1));
        wfa.set_final_weight(1, CountingSemiring(1));
        let _ = wfa.add_transition(0, 0, 1, CountingSemiring(3));

        let fwd = vec![CountingSemiring(1), CountingSemiring(0)];
        let new_fwd = advance_forward(&fwd, &wfa, 0);
        assert_eq!(new_fwd[0], CountingSemiring(0));
        assert_eq!(new_fwd[1], CountingSemiring(3));
    }

    #[test]
    fn test_dot_product_simple() {
        let a = vec![CountingSemiring(2), CountingSemiring(3)];
        let b = vec![CountingSemiring(4), CountingSemiring(5)];
        // 2*4 + 3*5 = 8 + 15 = 23
        let result = dot_product(&a, &b);
        assert_eq!(result, CountingSemiring(23));
    }

    // ── tests: extract_numeric helper ────────────────────────────────

    #[test]
    fn test_extract_numeric_float() {
        assert_eq!(extract_numeric("3.14"), Some(3.14));
    }

    #[test]
    fn test_extract_numeric_wrapped() {
        assert_eq!(extract_numeric("RealSemiring(2.5)"), Some(2.5));
    }

    #[test]
    fn test_extract_numeric_non_numeric() {
        assert_eq!(extract_numeric("hello"), None);
    }

    // ── integration: check_equivalence round-trip ────────────────────

    #[test]
    fn test_round_trip_check_and_verify() {
        let (wfa1, wfa2) = counting_wfa_pair();
        let result = check_equivalence(&wfa1, &wfa2).unwrap();
        assert!(result.is_equivalent());

        let cert = result.certificate().unwrap();
        let verified = verify_equivalence_certificate(&wfa1, &wfa2, cert).unwrap();
        assert!(verified);
    }
}
