//! Query oracle abstractions and implementations.
//!
//! Provides traits and concrete implementations for membership and equivalence
//! queries used in the PCL* active learning algorithm.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ordered_float::OrderedFloat;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Local type aliases
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

/// Sub-distribution over string-keyed outcomes.
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
    pub fn get(&self, key: &str) -> f64 { self.weights.get(key).copied().unwrap_or(0.0) }
    pub fn set(&mut self, key: String, prob: f64) {
        if prob > 1e-15 { self.weights.insert(key, prob); }
        else { self.weights.remove(&key); }
    }

    pub fn tv_distance(&self, other: &SubDistribution) -> f64 {
        let mut all_keys: std::collections::HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());
        let mut dist = 0.0;
        for key in all_keys {
            dist += (self.get(key) - other.get(key)).abs();
        }
        dist / 2.0
    }

    pub fn support_size(&self) -> usize { self.weights.len() }

    pub fn normalize(&mut self) {
        let total = self.total_mass();
        if total > 1e-15 {
            for v in self.weights.values_mut() { *v /= total; }
        }
    }

    /// Sample from the distribution (returns the key).
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<String> {
        let total = self.total_mass();
        if total < 1e-15 { return None; }

        let threshold = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        for (key, &prob) in &self.weights {
            cumulative += prob;
            if cumulative >= threshold {
                return Some(key.clone());
            }
        }
        self.weights.keys().last().cloned()
    }

    /// Mixture of two distributions: result = alpha * self + (1 - alpha) * other.
    pub fn mixture(&self, other: &Self, alpha: f64) -> Result<Self, String> {
        let mut result = HashMap::new();
        for (key, &prob) in &self.weights {
            *result.entry(key.clone()).or_insert(0.0) += alpha * prob;
        }
        for (key, &prob) in &other.weights {
            *result.entry(key.clone()).or_insert(0.0) += (1.0 - alpha) * prob;
        }
        Ok(Self { weights: result })
    }
}

impl Default for SubDistribution {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Query types
// ---------------------------------------------------------------------------

/// A membership query: "what is the output distribution for this word?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipQuery {
    /// The word to query
    pub word: Word,
    /// Requested number of samples
    pub num_samples: usize,
    /// Query identifier
    pub id: u64,
    /// Priority (higher = more important)
    pub priority: i32,
}

impl MembershipQuery {
    pub fn new(word: Word, num_samples: usize) -> Self {
        Self {
            word,
            num_samples,
            id: rand::thread_rng().gen(),
            priority: 0,
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Result of a membership query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipResult {
    /// The queried word
    pub word: Word,
    /// Estimated output distribution
    pub distribution: SubDistribution,
    /// Number of samples used
    pub num_samples: usize,
    /// Standard errors for each entry
    pub standard_errors: HashMap<String, f64>,
    /// Total wall-clock time (ms)
    pub time_ms: u64,
}

impl MembershipResult {
    pub fn new(word: Word, distribution: SubDistribution, num_samples: usize) -> Self {
        // Compute standard errors assuming multinomial sampling
        let mut se = HashMap::new();
        let n = num_samples as f64;
        for (key, &prob) in &distribution.weights {
            let std_err = (prob * (1.0 - prob) / n).sqrt();
            se.insert(key.clone(), std_err);
        }

        Self {
            word,
            distribution,
            num_samples,
            standard_errors: se,
            time_ms: 0,
        }
    }

    /// Margin of error at given confidence level.
    pub fn margin_of_error(&self, key: &str, z: f64) -> f64 {
        let se = self.standard_errors.get(key).copied().unwrap_or(0.0);
        z * se
    }
}

/// An equivalence query: "is this hypothesis correct?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceQuery {
    /// Number of random tests to perform
    pub num_tests: usize,
    /// Maximum word length for test words
    pub max_word_length: usize,
    /// Samples per test word
    pub samples_per_test: usize,
    /// Tolerance for declaring mismatch
    pub tolerance: f64,
}

impl EquivalenceQuery {
    pub fn new(num_tests: usize, max_word_length: usize) -> Self {
        Self {
            num_tests,
            max_word_length,
            samples_per_test: 50,
            tolerance: 0.05,
        }
    }
}

/// Result of an equivalence query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceResult {
    /// Whether the hypothesis passed all tests
    pub is_equivalent: bool,
    /// Counter-example word (if found)
    pub counterexample: Option<Word>,
    /// The observed mismatch distance (if CE found)
    pub mismatch_distance: Option<f64>,
    /// System distribution at CE (if found)
    pub system_distribution: Option<SubDistribution>,
    /// Hypothesis distribution at CE (if found)
    pub hypothesis_distribution: Option<SubDistribution>,
    /// Number of tests performed
    pub tests_performed: usize,
    /// Total wall-clock time (ms)
    pub time_ms: u64,
}

impl EquivalenceResult {
    pub fn equivalent(tests_performed: usize) -> Self {
        Self {
            is_equivalent: true,
            counterexample: None,
            mismatch_distance: None,
            system_distribution: None,
            hypothesis_distribution: None,
            tests_performed,
            time_ms: 0,
        }
    }

    pub fn counterexample(
        word: Word,
        distance: f64,
        sys_dist: SubDistribution,
        hyp_dist: SubDistribution,
        tests_performed: usize,
    ) -> Self {
        Self {
            is_equivalent: false,
            counterexample: Some(word),
            mismatch_distance: Some(distance),
            system_distribution: Some(sys_dist),
            hypothesis_distribution: Some(hyp_dist),
            tests_performed,
            time_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// QueryOracle trait
// ---------------------------------------------------------------------------

/// Trait for query oracles that can answer membership and equivalence queries.
pub trait QueryOracle: Send + Sync {
    /// Answer a membership query.
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult;

    /// Answer an equivalence query given a hypothesis.
    fn equivalence_query(
        &self,
        eq_query: &EquivalenceQuery,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult;

    /// Reset internal state.
    fn reset(&mut self);

    /// Number of queries answered so far.
    fn query_count(&self) -> usize;

    /// Get oracle statistics.
    fn stats(&self) -> OracleStats;
}

/// Statistics for an oracle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OracleStats {
    pub membership_queries: usize,
    pub equivalence_queries: usize,
    pub total_samples: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_time_ms: u64,
}

// ---------------------------------------------------------------------------
// StatisticalMembershipOracle
// ---------------------------------------------------------------------------

/// A membership oracle that generates multiple samples and estimates distributions.
pub struct StatisticalMembershipOracle<F>
where
    F: Fn(&Word) -> String + Send + Sync,
{
    /// The sampling function: given a word, return a single sample outcome
    sample_fn: F,
    /// Default number of samples per query
    default_samples: usize,
    /// Stats
    stats: Mutex<OracleStats>,
}

impl<F> StatisticalMembershipOracle<F>
where
    F: Fn(&Word) -> String + Send + Sync,
{
    pub fn new(sample_fn: F, default_samples: usize) -> Self {
        Self {
            sample_fn,
            default_samples,
            stats: Mutex::new(OracleStats::default()),
        }
    }

    /// Run multiple samples and estimate the distribution.
    fn estimate_distribution(&self, word: &Word, num_samples: usize) -> SubDistribution {
        let mut counts: HashMap<String, usize> = HashMap::new();

        for _ in 0..num_samples {
            let outcome = (self.sample_fn)(word);
            *counts.entry(outcome).or_insert(0) += 1;
        }

        let n = num_samples as f64;
        let mut dist = SubDistribution::new();
        for (key, count) in counts {
            dist.set(key, count as f64 / n);
        }
        dist
    }
}

impl<F> QueryOracle for StatisticalMembershipOracle<F>
where
    F: Fn(&Word) -> String + Send + Sync,
{
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        let num_samples = if query.num_samples > 0 {
            query.num_samples
        } else {
            self.default_samples
        };

        let start = std::time::Instant::now();
        let dist = self.estimate_distribution(&query.word, num_samples);
        let elapsed = start.elapsed().as_millis() as u64;

        let mut stats = self.stats.lock().unwrap();
        stats.membership_queries += 1;
        stats.total_samples += num_samples;
        stats.total_time_ms += elapsed;

        let mut result = MembershipResult::new(query.word.clone(), dist, num_samples);
        result.time_ms = elapsed;
        result
    }

    fn equivalence_query(
        &self,
        _eq_query: &EquivalenceQuery,
        _hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        // Statistical membership oracles can't do equivalence queries directly
        EquivalenceResult::equivalent(0)
    }

    fn reset(&mut self) {
        *self.stats.lock().unwrap() = OracleStats::default();
    }

    fn query_count(&self) -> usize {
        let stats = self.stats.lock().unwrap();
        stats.membership_queries + stats.equivalence_queries
    }

    fn stats(&self) -> OracleStats {
        self.stats.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// ApproximateEquivalenceOracle
// ---------------------------------------------------------------------------

/// Equivalence oracle that uses random testing.
pub struct ApproximateEquivalenceOracle<F>
where
    F: Fn(&Word) -> SubDistribution + Send + Sync,
{
    /// System query function
    system_fn: F,
    /// Alphabet for generating random words
    alphabet: Vec<Symbol>,
    /// Statistics
    stats: Mutex<OracleStats>,
}

impl<F> ApproximateEquivalenceOracle<F>
where
    F: Fn(&Word) -> SubDistribution + Send + Sync,
{
    pub fn new(system_fn: F, alphabet: Vec<Symbol>) -> Self {
        Self {
            system_fn,
            alphabet,
            stats: Mutex::new(OracleStats::default()),
        }
    }

    /// Generate a random word up to the given length.
    fn random_word<R: Rng>(&self, max_length: usize, rng: &mut R) -> Word {
        if self.alphabet.is_empty() {
            return Word::empty();
        }
        let len = rng.gen_range(0..=max_length);
        let symbols: Vec<Symbol> = (0..len)
            .map(|_| self.alphabet[rng.gen_range(0..self.alphabet.len())].clone())
            .collect();
        Word::from_symbols(symbols)
    }
}

impl<F> QueryOracle for ApproximateEquivalenceOracle<F>
where
    F: Fn(&Word) -> SubDistribution + Send + Sync,
{
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        let dist = (self.system_fn)(&query.word);
        let mut stats = self.stats.lock().unwrap();
        stats.membership_queries += 1;
        MembershipResult::new(query.word.clone(), dist, 1)
    }

    fn equivalence_query(
        &self,
        eq_query: &EquivalenceQuery,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        let start = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        let mut stats = self.stats.lock().unwrap();
        stats.equivalence_queries += 1;
        drop(stats);

        for i in 0..eq_query.num_tests {
            let word = self.random_word(eq_query.max_word_length, &mut rng);
            let sys_dist = (self.system_fn)(&word);
            let hyp_dist = hypothesis(&word);

            let distance = sys_dist.tv_distance(&hyp_dist);
            if distance > eq_query.tolerance {
                let mut result = EquivalenceResult::counterexample(
                    word, distance, sys_dist, hyp_dist, i + 1,
                );
                result.time_ms = start.elapsed().as_millis() as u64;
                return result;
            }
        }

        let mut result = EquivalenceResult::equivalent(eq_query.num_tests);
        result.time_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn reset(&mut self) {
        *self.stats.lock().unwrap() = OracleStats::default();
    }

    fn query_count(&self) -> usize {
        let stats = self.stats.lock().unwrap();
        stats.membership_queries + stats.equivalence_queries
    }

    fn stats(&self) -> OracleStats {
        self.stats.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// CachedOracle
// ---------------------------------------------------------------------------

/// A memoization layer over any oracle.
pub struct CachedOracle<O: QueryOracle> {
    inner: O,
    cache: Mutex<HashMap<Word, MembershipResult>>,
    max_cache_size: usize,
    stats: Mutex<OracleStats>,
}

impl<O: QueryOracle> CachedOracle<O> {
    pub fn new(inner: O, max_cache_size: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
            max_cache_size,
            stats: Mutex::new(OracleStats::default()),
        }
    }

    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    fn evict_if_needed(cache: &mut HashMap<Word, MembershipResult>, max_size: usize) {
        if cache.len() >= max_size {
            let to_remove = max_size / 10;
            let keys: Vec<Word> = cache.keys().take(to_remove).cloned().collect();
            for key in keys {
                cache.remove(&key);
            }
        }
    }
}

impl<O: QueryOracle> QueryOracle for CachedOracle<O> {
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        // Check cache
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&query.word) {
                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
                return cached.clone();
            }
        }

        // Cache miss
        let mut stats = self.stats.lock().unwrap();
        stats.cache_misses += 1;
        stats.membership_queries += 1;
        drop(stats);

        let result = self.inner.membership_query(query);

        // Store in cache
        let mut cache = self.cache.lock().unwrap();
        Self::evict_if_needed(&mut cache, self.max_cache_size);
        cache.insert(query.word.clone(), result.clone());

        result
    }

    fn equivalence_query(
        &self,
        eq_query: &EquivalenceQuery,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        let mut stats = self.stats.lock().unwrap();
        stats.equivalence_queries += 1;
        drop(stats);
        self.inner.equivalence_query(eq_query, hypothesis)
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.cache.lock().unwrap().clear();
        *self.stats.lock().unwrap() = OracleStats::default();
    }

    fn query_count(&self) -> usize {
        self.inner.query_count()
    }

    fn stats(&self) -> OracleStats {
        let mut stats = self.stats.lock().unwrap().clone();
        let inner_stats = self.inner.stats();
        stats.total_samples = inner_stats.total_samples;
        stats.total_time_ms = inner_stats.total_time_ms;
        stats
    }
}

// ---------------------------------------------------------------------------
// BatchOracle
// ---------------------------------------------------------------------------

/// Batches queries for improved throughput.
pub struct BatchOracle<O: QueryOracle> {
    inner: O,
    batch_size: usize,
    pending: Mutex<Vec<MembershipQuery>>,
    results: Mutex<HashMap<u64, MembershipResult>>,
    stats: Mutex<OracleStats>,
}

impl<O: QueryOracle> BatchOracle<O> {
    pub fn new(inner: O, batch_size: usize) -> Self {
        Self {
            inner,
            batch_size,
            pending: Mutex::new(Vec::new()),
            results: Mutex::new(HashMap::new()),
            stats: Mutex::new(OracleStats::default()),
        }
    }

    /// Submit a query for batching.
    pub fn submit(&self, query: MembershipQuery) -> u64 {
        let id = query.id;
        self.pending.lock().unwrap().push(query);
        id
    }

    /// Flush pending queries and execute them.
    pub fn flush(&self) {
        let queries: Vec<MembershipQuery> = {
            let mut pending = self.pending.lock().unwrap();
            std::mem::take(&mut *pending)
        };

        if queries.is_empty() {
            return;
        }

        let mut stats = self.stats.lock().unwrap();
        stats.membership_queries += queries.len();
        drop(stats);

        let mut results = self.results.lock().unwrap();
        for query in queries {
            let id = query.id;
            let result = self.inner.membership_query(&query);
            results.insert(id, result);
        }
    }

    /// Get the result for a previously submitted query.
    pub fn get_result(&self, query_id: u64) -> Option<MembershipResult> {
        self.results.lock().unwrap().remove(&query_id)
    }

    /// Submit and immediately execute a batch of queries.
    pub fn execute_batch(&self, queries: Vec<MembershipQuery>) -> Vec<MembershipResult> {
        let mut stats = self.stats.lock().unwrap();
        stats.membership_queries += queries.len();
        drop(stats);

        queries
            .iter()
            .map(|q| self.inner.membership_query(q))
            .collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }
}

impl<O: QueryOracle> QueryOracle for BatchOracle<O> {
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        let mut stats = self.stats.lock().unwrap();
        stats.membership_queries += 1;
        drop(stats);
        self.inner.membership_query(query)
    }

    fn equivalence_query(
        &self,
        eq_query: &EquivalenceQuery,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        self.inner.equivalence_query(eq_query, hypothesis)
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.pending.lock().unwrap().clear();
        self.results.lock().unwrap().clear();
        *self.stats.lock().unwrap() = OracleStats::default();
    }

    fn query_count(&self) -> usize {
        self.inner.query_count()
    }

    fn stats(&self) -> OracleStats {
        self.stats.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// MockOracle
// ---------------------------------------------------------------------------

/// A deterministic oracle for testing, constructed from a known automaton.
pub struct MockOracle {
    /// Deterministic transition function: (state, symbol) → state
    transitions: HashMap<(usize, String), usize>,
    /// Output for each state
    outputs: HashMap<usize, SubDistribution>,
    /// Initial state
    initial_state: usize,
    /// Alphabet
    alphabet: Vec<Symbol>,
    /// Stats
    stats: Mutex<OracleStats>,
}

impl MockOracle {
    pub fn new(
        transitions: HashMap<(usize, String), usize>,
        outputs: HashMap<usize, SubDistribution>,
        initial_state: usize,
        alphabet: Vec<Symbol>,
    ) -> Self {
        Self {
            transitions,
            outputs,
            initial_state,
            alphabet,
            stats: Mutex::new(OracleStats::default()),
        }
    }

    /// Create a simple two-state automaton for testing.
    pub fn two_state(alphabet: Vec<Symbol>) -> Self {
        let mut transitions = HashMap::new();
        let mut outputs = HashMap::new();

        // State 0: on first symbol → state 1, else → state 0
        // State 1: always → state 0
        if let Some(first_sym) = alphabet.first() {
            transitions.insert((0, first_sym.0.clone()), 1);
            for sym in alphabet.iter().skip(1) {
                transitions.insert((0, sym.0.clone()), 0);
            }
            for sym in &alphabet {
                transitions.insert((1, sym.0.clone()), 0);
            }
        }

        outputs.insert(0, SubDistribution::singleton("accept".to_string(), 0.8));
        outputs.insert(1, SubDistribution::singleton("reject".to_string(), 0.9));

        Self::new(transitions, outputs, 0, alphabet)
    }

    /// Create an n-state cycle automaton.
    pub fn cycle(n: usize, alphabet: Vec<Symbol>) -> Self {
        let mut transitions = HashMap::new();
        let mut outputs = HashMap::new();

        for state in 0..n {
            if let Some(first_sym) = alphabet.first() {
                transitions.insert((state, first_sym.0.clone()), (state + 1) % n);
            }
            for sym in alphabet.iter().skip(1) {
                transitions.insert((state, sym.0.clone()), state);
            }
            let p = (state as f64 + 1.0) / (n as f64 + 1.0);
            outputs.insert(
                state,
                SubDistribution::singleton("out".to_string(), p),
            );
        }

        Self::new(transitions, outputs, 0, alphabet)
    }

    /// Run the automaton on a word.
    fn run_word(&self, word: &Word) -> usize {
        let mut state = self.initial_state;
        for sym in &word.symbols {
            if let Some(&next) = self.transitions.get(&(state, sym.0.clone())) {
                state = next;
            }
            // If no transition, stay in same state
        }
        state
    }

    /// Get output distribution for a word.
    fn output_for(&self, word: &Word) -> SubDistribution {
        let state = self.run_word(word);
        self.outputs
            .get(&state)
            .cloned()
            .unwrap_or_default()
    }

    pub fn num_states(&self) -> usize {
        let mut max_state = self.initial_state;
        for (&(s, _), &t) in &self.transitions {
            max_state = max_state.max(s).max(t);
        }
        max_state + 1
    }
}

impl QueryOracle for MockOracle {
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        let mut stats = self.stats.lock().unwrap();
        stats.membership_queries += 1;
        stats.total_samples += query.num_samples;
        drop(stats);

        let dist = self.output_for(&query.word);
        MembershipResult::new(query.word.clone(), dist, query.num_samples)
    }

    fn equivalence_query(
        &self,
        eq_query: &EquivalenceQuery,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        let mut stats = self.stats.lock().unwrap();
        stats.equivalence_queries += 1;
        drop(stats);

        let mut rng = rand::thread_rng();

        for i in 0..eq_query.num_tests {
            // Generate random word
            let len = rng.gen_range(0..=eq_query.max_word_length);
            let symbols: Vec<Symbol> = (0..len)
                .map(|_| {
                    if self.alphabet.is_empty() {
                        Symbol::new("a")
                    } else {
                        self.alphabet[rng.gen_range(0..self.alphabet.len())].clone()
                    }
                })
                .collect();
            let word = Word::from_symbols(symbols);

            let sys_dist = self.output_for(&word);
            let hyp_dist = hypothesis(&word);

            let distance = sys_dist.tv_distance(&hyp_dist);
            if distance > eq_query.tolerance {
                return EquivalenceResult::counterexample(
                    word, distance, sys_dist, hyp_dist, i + 1,
                );
            }
        }

        EquivalenceResult::equivalent(eq_query.num_tests)
    }

    fn reset(&mut self) {
        *self.stats.lock().unwrap() = OracleStats::default();
    }

    fn query_count(&self) -> usize {
        let stats = self.stats.lock().unwrap();
        stats.membership_queries + stats.equivalence_queries
    }

    fn stats(&self) -> OracleStats {
        self.stats.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// StochasticOracle
// ---------------------------------------------------------------------------

/// Noise model for the stochastic oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseModel {
    /// Gaussian noise added to probabilities
    Gaussian { std_dev: f64 },
    /// Uniform noise in [-width, width]
    Uniform { width: f64 },
    /// Beta distribution noise (concentration parameter)
    Dirichlet { concentration: f64 },
    /// No noise
    None,
}

/// An oracle that wraps another oracle and adds configurable noise.
pub struct StochasticOracle {
    /// The underlying deterministic system
    system: HashMap<Word, SubDistribution>,
    /// Noise model
    noise_model: NoiseModel,
    /// Alphabet
    alphabet: Vec<Symbol>,
    /// Stats
    stats: Mutex<OracleStats>,
}

impl StochasticOracle {
    pub fn new(
        system: HashMap<Word, SubDistribution>,
        noise_model: NoiseModel,
        alphabet: Vec<Symbol>,
    ) -> Self {
        Self {
            system,
            noise_model,
            alphabet,
            stats: Mutex::new(OracleStats::default()),
        }
    }

    /// Create from a mock oracle.
    pub fn from_mock(mock: &MockOracle, noise_model: NoiseModel) -> Self {
        let mut system = HashMap::new();

        // Populate system with outputs for all words up to length 3
        fn enumerate_words(alphabet: &[Symbol], max_len: usize) -> Vec<Word> {
            let mut words = vec![Word::empty()];
            let mut current = vec![Word::empty()];

            for _ in 0..max_len {
                let mut next = Vec::new();
                for w in &current {
                    for sym in alphabet {
                        next.push(w.concat(&Word::singleton(sym.clone())));
                    }
                }
                words.extend(next.iter().cloned());
                current = next;
            }
            words
        }

        for word in enumerate_words(&mock.alphabet, 3) {
            let dist = mock.output_for(&word);
            system.insert(word, dist);
        }

        Self::new(system, noise_model, mock.alphabet.clone())
    }

    fn add_noise(&self, dist: &SubDistribution) -> SubDistribution {
        let mut rng = rand::thread_rng();
        let mut noisy = SubDistribution::new();

        match &self.noise_model {
            NoiseModel::Gaussian { std_dev } => {
                let normal = Normal::new(0.0, *std_dev).unwrap_or(Normal::new(0.0, 0.01).unwrap());
                for (key, &prob) in &dist.weights {
                    let noise: f64 = normal.sample(&mut rng);
                    let noisy_prob = (prob + noise).clamp(0.0, 1.0);
                    noisy.set(key.clone(), noisy_prob);
                }
            }
            NoiseModel::Uniform { width } => {
                let uniform = Uniform::new(-width, *width);
                for (key, &prob) in &dist.weights {
                    let noise: f64 = uniform.sample(&mut rng);
                    let noisy_prob = (prob + noise).clamp(0.0, 1.0);
                    noisy.set(key.clone(), noisy_prob);
                }
            }
            NoiseModel::Dirichlet { concentration } => {
                // Approximate Dirichlet by adding gamma-distributed noise
                for (key, &prob) in &dist.weights {
                    let shape = prob * concentration;
                    // Use gamma approximation: mean = shape/rate = prob
                    let noise_factor = if shape > 0.01 {
                        let std = (shape).sqrt() / concentration;
                        let n = Normal::new(1.0, std).unwrap_or(Normal::new(1.0, 0.01).unwrap());
                        n.sample(&mut rng).max(0.0)
                    } else {
                        1.0
                    };
                    noisy.set(key.clone(), (prob * noise_factor).clamp(0.0, 1.0));
                }
            }
            NoiseModel::None => {
                return dist.clone();
            }
        }

        // Renormalize if total mass exceeds 1
        let total = noisy.total_mass();
        if total > 1.0 {
            noisy.normalize();
        }

        noisy
    }

    fn output_for(&self, word: &Word) -> SubDistribution {
        let base = self.system.get(word)
            .cloned()
            .unwrap_or_default();
        self.add_noise(&base)
    }
}

impl QueryOracle for StochasticOracle {
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        let mut stats = self.stats.lock().unwrap();
        stats.membership_queries += 1;
        stats.total_samples += query.num_samples;
        drop(stats);

        // Average over multiple noisy samples
        let mut accumulated = SubDistribution::new();
        let n = query.num_samples.max(1);

        for _ in 0..n {
            let sample = self.output_for(&query.word);
            for (key, &prob) in &sample.weights {
                let current = accumulated.get(key);
                accumulated.set(key.clone(), current + prob / n as f64);
            }
        }

        MembershipResult::new(query.word.clone(), accumulated, n)
    }

    fn equivalence_query(
        &self,
        eq_query: &EquivalenceQuery,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        let mut stats = self.stats.lock().unwrap();
        stats.equivalence_queries += 1;
        drop(stats);

        let mut rng = rand::thread_rng();

        for i in 0..eq_query.num_tests {
            let len = rng.gen_range(0..=eq_query.max_word_length);
            let symbols: Vec<Symbol> = (0..len)
                .map(|_| {
                    if self.alphabet.is_empty() {
                        Symbol::new("a")
                    } else {
                        self.alphabet[rng.gen_range(0..self.alphabet.len())].clone()
                    }
                })
                .collect();
            let word = Word::from_symbols(symbols);

            // Average over samples
            let mut sys_avg = SubDistribution::new();
            for _ in 0..eq_query.samples_per_test {
                let sample = self.output_for(&word);
                for (key, &prob) in &sample.weights {
                    let current = sys_avg.get(key);
                    sys_avg.set(key.clone(), current + prob / eq_query.samples_per_test as f64);
                }
            }

            let hyp_dist = hypothesis(&word);
            let distance = sys_avg.tv_distance(&hyp_dist);

            if distance > eq_query.tolerance {
                return EquivalenceResult::counterexample(
                    word, distance, sys_avg, hyp_dist, i + 1,
                );
            }
        }

        EquivalenceResult::equivalent(eq_query.num_tests)
    }

    fn reset(&mut self) {
        *self.stats.lock().unwrap() = OracleStats::default();
    }

    fn query_count(&self) -> usize {
        let stats = self.stats.lock().unwrap();
        stats.membership_queries + stats.equivalence_queries
    }

    fn stats(&self) -> OracleStats {
        self.stats.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// Query scheduling and prioritization
// ---------------------------------------------------------------------------

/// Priority queue for queries.
pub struct QueryScheduler {
    /// Pending queries sorted by priority
    queue: Vec<MembershipQuery>,
    /// Maximum queue size
    max_size: usize,
}

impl QueryScheduler {
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Vec::new(),
            max_size,
        }
    }

    /// Add a query to the schedule.
    pub fn enqueue(&mut self, query: MembershipQuery) {
        if self.queue.len() >= self.max_size {
            // Remove lowest priority
            if let Some(min_idx) = self.queue.iter()
                .enumerate()
                .min_by_key(|(_, q)| q.priority)
                .map(|(i, _)| i)
            {
                if self.queue[min_idx].priority < query.priority {
                    self.queue.swap_remove(min_idx);
                } else {
                    return; // New query has lower priority, skip
                }
            }
        }
        self.queue.push(query);
    }

    /// Get the next highest-priority query.
    pub fn dequeue(&mut self) -> Option<MembershipQuery> {
        if self.queue.is_empty() {
            return None;
        }

        let max_idx = self.queue.iter()
            .enumerate()
            .max_by_key(|(_, q)| q.priority)
            .map(|(i, _)| i)
            .unwrap();

        Some(self.queue.swap_remove(max_idx))
    }

    /// Dequeue up to `n` highest-priority queries.
    pub fn dequeue_batch(&mut self, n: usize) -> Vec<MembershipQuery> {
        let mut batch = Vec::new();
        for _ in 0..n {
            match self.dequeue() {
                Some(q) => batch.push(q),
                None => break,
            }
        }
        batch
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

// ---------------------------------------------------------------------------
// Sample size calculation
// ---------------------------------------------------------------------------

/// Calculate the number of samples needed for a given confidence and accuracy.
pub fn samples_for_confidence(
    epsilon: f64,
    delta: f64,
    support_size: usize,
) -> usize {
    if epsilon <= 0.0 || delta <= 0.0 {
        return usize::MAX;
    }

    // Hoeffding: m ≥ (1/(2ε²)) * ln(2k/δ) where k = support size
    let base = 1.0 / (2.0 * epsilon * epsilon);
    let log_term = (2.0 * support_size as f64 / delta).ln().max(1.0);
    (base * log_term).ceil() as usize
}

/// Calculate confidence given sample size and accuracy.
pub fn confidence_from_samples(
    sample_size: usize,
    epsilon: f64,
    support_size: usize,
) -> f64 {
    // δ = 2k * exp(-2mε²)
    let delta = 2.0 * support_size as f64
        * (-2.0 * sample_size as f64 * epsilon * epsilon).exp();
    (1.0 - delta).max(0.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Advanced oracle infrastructure
// ---------------------------------------------------------------------------

/// Composite oracle that chains multiple oracles for fallback behavior.
pub struct CompositeOracle {
    oracles: Vec<Box<dyn QueryOracle>>,
    strategy: CompositeStrategy,
}

/// Strategy for combining multiple oracle results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeStrategy {
    /// Use first oracle that succeeds
    Fallback,
    /// Average results from all oracles
    Average,
    /// Use majority vote (for discrete outcomes)
    MajorityVote,
}

impl CompositeOracle {
    pub fn new(oracles: Vec<Box<dyn QueryOracle>>, strategy: CompositeStrategy) -> Self {
        Self { oracles, strategy }
    }

    pub fn fallback(oracles: Vec<Box<dyn QueryOracle>>) -> Self {
        Self::new(oracles, CompositeStrategy::Fallback)
    }

    pub fn averaging(oracles: Vec<Box<dyn QueryOracle>>) -> Self {
        Self::new(oracles, CompositeStrategy::Average)
    }
}

impl QueryOracle for CompositeOracle {
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        match self.strategy {
            CompositeStrategy::Fallback => {
                for oracle in &self.oracles {
                    let result = oracle.membership_query(query);
                    if result.distribution.support_size() > 0 {
                        return result;
                    }
                }
                MembershipResult::new(
                    query.word.clone(),
                    SubDistribution::new(),
                    0,
                )
            }
            CompositeStrategy::Average => {
                let mut combined = SubDistribution::new();
                let mut total_samples = 0usize;
                let mut count = 0usize;
                for oracle in &self.oracles {
                    let result = oracle.membership_query(query);
                    if result.distribution.support_size() > 0 {
                        let alpha = if count == 0 { 1.0 } else { 1.0 / (count + 1) as f64 };
                        combined = combined.mixture(&result.distribution, 1.0 - alpha).unwrap_or(combined);
                        total_samples += result.num_samples;
                        count += 1;
                    }
                }
                MembershipResult::new(
                    query.word.clone(),
                    combined,
                    total_samples,
                )
            }
            CompositeStrategy::MajorityVote => {
                let mut vote_counts: HashMap<String, usize> = HashMap::new();
                let mut total = 0usize;
                for oracle in &self.oracles {
                    let result = oracle.membership_query(query);
                    if let Some((key, _)) = result.distribution.weights.iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        *vote_counts.entry(key.clone()).or_insert(0) += 1;
                        total += 1;
                    }
                }
                let mut result_dist = SubDistribution::new();
                for (key, count) in &vote_counts {
                    result_dist.set(key.clone(), *count as f64 / total.max(1) as f64);
                }
                MembershipResult::new(
                    query.word.clone(),
                    result_dist,
                    total,
                )
            }
        }
    }

    fn equivalence_query(
        &self,
        query: &EquivalenceQuery,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        // For equivalence, use first oracle (or any that finds a counterexample)
        for oracle in &self.oracles {
            let result = oracle.equivalence_query(query, hypothesis);
            if let Some(ref _ce) = result.counterexample {
                return result;
            }
        }
        EquivalenceResult::equivalent(0)
    }

    fn reset(&mut self) {
        for oracle in &mut self.oracles {
            oracle.reset();
        }
    }

    fn query_count(&self) -> usize {
        self.oracles.iter().map(|o| o.query_count()).sum()
    }

    fn stats(&self) -> OracleStats {
        OracleStats::default()
    }
}

/// Replay oracle that replays recorded query-response pairs.
pub struct ReplayOracle {
    pub recordings: HashMap<String, SubDistribution>,
    pub query_log: Vec<(MembershipQuery, MembershipResult)>,
    pub fallback_distribution: SubDistribution,
}

impl ReplayOracle {
    pub fn new() -> Self {
        Self {
            recordings: HashMap::new(),
            query_log: Vec::new(),
            fallback_distribution: SubDistribution::new(),
        }
    }

    pub fn with_fallback(fallback: SubDistribution) -> Self {
        Self {
            recordings: HashMap::new(),
            query_log: Vec::new(),
            fallback_distribution: fallback,
        }
    }

    pub fn record(&mut self, word: &Word, dist: SubDistribution) {
        self.recordings.insert(format!("{}", word), dist);
    }

    pub fn num_recordings(&self) -> usize {
        self.recordings.len()
    }

    pub fn num_replayed(&self) -> usize {
        self.query_log.len()
    }
}

impl QueryOracle for ReplayOracle {
    fn membership_query(&self, query: &MembershipQuery) -> MembershipResult {
        let key = format!("{}", query.word);
        let dist = self.recordings.get(&key)
            .cloned()
            .unwrap_or_else(|| self.fallback_distribution.clone());
        MembershipResult::new(query.word.clone(), dist, query.num_samples)
    }

    fn equivalence_query(
        &self,
        _query: &EquivalenceQuery,
        _hypothesis: &dyn Fn(&Word) -> SubDistribution,
    ) -> EquivalenceResult {
        EquivalenceResult::equivalent(0)
    }

    fn reset(&mut self) {
        self.query_log.clear();
    }

    fn query_count(&self) -> usize {
        self.query_log.len()
    }

    fn stats(&self) -> OracleStats {
        OracleStats::default()
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OracleMetrics {
    pub total_membership_queries: usize,
    pub total_equivalence_queries: usize,
    pub total_samples: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub avg_query_time_ms: f64,
    pub max_query_time_ms: f64,
    pub counterexamples_found: usize,
}

impl OracleMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { return 0.0; }
        self.cache_hits as f64 / total as f64
    }

    pub fn queries_per_counterexample(&self) -> f64 {
        if self.counterexamples_found == 0 {
            return f64::INFINITY;
        }
        self.total_equivalence_queries as f64 / self.counterexamples_found as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "OracleMetrics(MQ={}, EQ={}, samples={}, cache_hit={:.1}%, CE={})",
            self.total_membership_queries,
            self.total_equivalence_queries,
            self.total_samples,
            self.cache_hit_rate() * 100.0,
            self.counterexamples_found,
        )
    }
}

/// Weighted sample size calculator for different confidence requirements.
pub struct SampleSizeCalculator;

impl SampleSizeCalculator {
    /// Hoeffding bound: P(|X̄ - μ| ≥ ε) ≤ 2exp(-2nε²)
    pub fn hoeffding(epsilon: f64, delta: f64) -> usize {
        if epsilon <= 0.0 || delta <= 0.0 || delta >= 1.0 {
            return 1;
        }
        let n = (2.0 / delta).ln() / (2.0 * epsilon * epsilon);
        n.ceil() as usize
    }

    /// Chernoff bound for multiplicative error.
    pub fn chernoff_multiplicative(epsilon: f64, delta: f64, p: f64) -> usize {
        if epsilon <= 0.0 || delta <= 0.0 || p <= 0.0 {
            return 1;
        }
        let n = (2.0 / delta).ln() * 3.0 / (epsilon * epsilon * p);
        n.ceil() as usize
    }

    /// Sample size for estimating a probability to within ε with confidence 1-δ.
    pub fn for_probability_estimation(epsilon: f64, delta: f64) -> usize {
        Self::hoeffding(epsilon, delta)
    }

    /// Sample size for KS test to distinguish distributions separated by d.
    pub fn for_ks_test(min_distance: f64, significance: f64) -> usize {
        if min_distance <= 0.0 {
            return 10000;
        }
        // KS critical value scales as c(α) / sqrt(n)
        let c_alpha = if significance <= 0.01 { 1.63 } else if significance <= 0.05 { 1.36 } else { 1.22 };
        let n = (c_alpha / min_distance).powi(2);
        n.ceil() as usize
    }

    /// Minimum samples for PAC learning with given parameters.
    pub fn for_pac_learning(epsilon: f64, delta: f64, vc_dimension: usize) -> usize {
        if epsilon <= 0.0 || delta <= 0.0 {
            return 1;
        }
        let d = vc_dimension as f64;
        let n = (4.0 / epsilon) * (2.0 * d * (2.0 / epsilon).ln() + (2.0 / delta).ln());
        n.ceil() as usize
    }

    /// Sample size using the DKW inequality for distribution estimation.
    pub fn dkw_inequality(epsilon: f64, delta: f64) -> usize {
        if epsilon <= 0.0 || delta <= 0.0 {
            return 1;
        }
        let n = (2.0 / delta).ln() / (2.0 * epsilon * epsilon);
        n.ceil() as usize
    }
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

    #[test]
    fn test_membership_query_creation() {
        let q = MembershipQuery::new(Word::from_str_slice(&["a", "b"]), 100);
        assert_eq!(q.word.len(), 2);
        assert_eq!(q.num_samples, 100);
        assert_eq!(q.priority, 0);

        let q2 = q.with_priority(5);
        assert_eq!(q2.priority, 5);
    }

    #[test]
    fn test_membership_result() {
        let dist = SubDistribution::singleton("yes".to_string(), 0.7);
        let result = MembershipResult::new(Word::from_str_slice(&["a"]), dist, 100);

        assert_eq!(result.num_samples, 100);
        assert!(!result.standard_errors.is_empty());

        let margin = result.margin_of_error("yes", 1.96);
        assert!(margin > 0.0);
        assert!(margin < 0.5);
    }

    #[test]
    fn test_equivalence_result_equivalent() {
        let result = EquivalenceResult::equivalent(1000);
        assert!(result.is_equivalent);
        assert!(result.counterexample.is_none());
    }

    #[test]
    fn test_equivalence_result_counterexample() {
        let sys = SubDistribution::singleton("a".to_string(), 0.9);
        let hyp = SubDistribution::singleton("b".to_string(), 0.8);
        let result = EquivalenceResult::counterexample(
            Word::from_str_slice(&["x"]),
            0.5,
            sys,
            hyp,
            42,
        );
        assert!(!result.is_equivalent);
        assert!(result.counterexample.is_some());
        assert_eq!(result.tests_performed, 42);
    }

    #[test]
    fn test_statistical_membership_oracle() {
        let oracle = StatisticalMembershipOracle::new(
            |word: &Word| {
                if word.is_empty() {
                    "yes".to_string()
                } else {
                    "no".to_string()
                }
            },
            50,
        );

        let query = MembershipQuery::new(Word::empty(), 50);
        let result = oracle.membership_query(&query);

        assert!((result.distribution.get("yes") - 1.0).abs() < 0.001);
        assert_eq!(oracle.query_count(), 1);
    }

    #[test]
    fn test_approximate_equivalence_oracle_pass() {
        let oracle = ApproximateEquivalenceOracle::new(
            |_word: &Word| SubDistribution::singleton("ok".to_string(), 1.0),
            make_alphabet(),
        );

        let eq_query = EquivalenceQuery::new(100, 3);
        let hypothesis = |_: &Word| SubDistribution::singleton("ok".to_string(), 1.0);
        let result = oracle.equivalence_query(&eq_query, &hypothesis);

        assert!(result.is_equivalent);
    }

    #[test]
    fn test_approximate_equivalence_oracle_fail() {
        let oracle = ApproximateEquivalenceOracle::new(
            |_word: &Word| SubDistribution::singleton("yes".to_string(), 0.9),
            make_alphabet(),
        );

        let eq_query = EquivalenceQuery::new(100, 3);
        let hypothesis = |_: &Word| SubDistribution::singleton("no".to_string(), 0.8);
        let result = oracle.equivalence_query(&eq_query, &hypothesis);

        assert!(!result.is_equivalent);
    }

    #[test]
    fn test_mock_oracle_two_state() {
        let alphabet = make_alphabet();
        let oracle = MockOracle::two_state(alphabet);

        // Empty word → state 0
        let q = MembershipQuery::new(Word::empty(), 10);
        let r = oracle.membership_query(&q);
        assert!(r.distribution.get("accept") > 0.5);

        // "a" → state 1
        let q = MembershipQuery::new(Word::from_str_slice(&["a"]), 10);
        let r = oracle.membership_query(&q);
        assert!(r.distribution.get("reject") > 0.5);

        // "a" "b" → state 0
        let q = MembershipQuery::new(Word::from_str_slice(&["a", "b"]), 10);
        let r = oracle.membership_query(&q);
        assert!(r.distribution.get("accept") > 0.5);
    }

    #[test]
    fn test_mock_oracle_cycle() {
        let alphabet = vec![Symbol::new("a")];
        let oracle = MockOracle::cycle(3, alphabet);

        assert_eq!(oracle.num_states(), 3);

        // State 0 → state 1 → state 2 → state 0
        let q = MembershipQuery::new(Word::empty(), 10);
        let r = oracle.membership_query(&q);
        assert!(r.distribution.get("out") > 0.0);
    }

    #[test]
    fn test_mock_oracle_equivalence() {
        let alphabet = make_alphabet();
        let oracle = MockOracle::two_state(alphabet);

        let eq_query = EquivalenceQuery::new(100, 3);

        // Hypothesis that matches
        let mock_ref = &oracle;
        let hypothesis = |word: &Word| -> SubDistribution {
            mock_ref.output_for(word)
        };
        let result = oracle.equivalence_query(&eq_query, &hypothesis);
        assert!(result.is_equivalent);
    }

    #[test]
    fn test_cached_oracle() {
        let inner = MockOracle::two_state(make_alphabet());
        let oracle = CachedOracle::new(inner, 100);

        let q = MembershipQuery::new(Word::empty(), 10);

        // First query: cache miss
        let r1 = oracle.membership_query(&q);
        assert_eq!(oracle.cache_size(), 1);

        // Second query: cache hit
        let r2 = oracle.membership_query(&q);
        assert_eq!(oracle.cache_size(), 1);

        // Results should be the same
        assert!((r1.distribution.get("accept") - r2.distribution.get("accept")).abs() < 0.001);

        let stats = oracle.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_cached_oracle_clear() {
        let inner = MockOracle::two_state(make_alphabet());
        let oracle = CachedOracle::new(inner, 100);

        let q = MembershipQuery::new(Word::empty(), 10);
        oracle.membership_query(&q);
        assert_eq!(oracle.cache_size(), 1);

        oracle.clear_cache();
        assert_eq!(oracle.cache_size(), 0);
    }

    #[test]
    fn test_batch_oracle() {
        let inner = MockOracle::two_state(make_alphabet());
        let oracle = BatchOracle::new(inner, 10);

        let queries = vec![
            MembershipQuery::new(Word::empty(), 10),
            MembershipQuery::new(Word::from_str_slice(&["a"]), 10),
            MembershipQuery::new(Word::from_str_slice(&["b"]), 10),
        ];

        let results = oracle.execute_batch(queries);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_oracle_submit_flush() {
        let inner = MockOracle::two_state(make_alphabet());
        let oracle = BatchOracle::new(inner, 10);

        let q1 = MembershipQuery::new(Word::empty(), 10);
        let id1 = oracle.submit(q1);

        let q2 = MembershipQuery::new(Word::from_str_slice(&["a"]), 10);
        let id2 = oracle.submit(q2);

        assert_eq!(oracle.pending_count(), 2);

        oracle.flush();

        assert!(oracle.get_result(id1).is_some());
        assert!(oracle.get_result(id2).is_some());
    }

    #[test]
    fn test_stochastic_oracle_no_noise() {
        let mut system = HashMap::new();
        system.insert(
            Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
        );

        let oracle = StochasticOracle::new(system, NoiseModel::None, make_alphabet());

        let q = MembershipQuery::new(Word::empty(), 1);
        let r = oracle.membership_query(&q);

        assert!((r.distribution.get("yes") - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_stochastic_oracle_gaussian_noise() {
        let mut system = HashMap::new();
        system.insert(
            Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
        );

        let oracle = StochasticOracle::new(
            system,
            NoiseModel::Gaussian { std_dev: 0.01 },
            make_alphabet(),
        );

        let q = MembershipQuery::new(Word::empty(), 100);
        let r = oracle.membership_query(&q);

        // With many samples and small noise, should be close to true value
        assert!((r.distribution.get("yes") - 0.8).abs() < 0.1);
    }

    #[test]
    fn test_stochastic_oracle_uniform_noise() {
        let mut system = HashMap::new();
        system.insert(
            Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.5),
        );

        let oracle = StochasticOracle::new(
            system,
            NoiseModel::Uniform { width: 0.05 },
            make_alphabet(),
        );

        let q = MembershipQuery::new(Word::empty(), 100);
        let r = oracle.membership_query(&q);

        assert!(r.distribution.get("yes") > 0.0);
    }

    #[test]
    fn test_query_scheduler() {
        let mut scheduler = QueryScheduler::new(100);

        scheduler.enqueue(MembershipQuery::new(Word::empty(), 10).with_priority(1));
        scheduler.enqueue(MembershipQuery::new(Word::from_str_slice(&["a"]), 10).with_priority(5));
        scheduler.enqueue(MembershipQuery::new(Word::from_str_slice(&["b"]), 10).with_priority(3));

        assert_eq!(scheduler.len(), 3);

        // Should dequeue highest priority first
        let q = scheduler.dequeue().unwrap();
        assert_eq!(q.priority, 5);

        let q = scheduler.dequeue().unwrap();
        assert_eq!(q.priority, 3);

        let q = scheduler.dequeue().unwrap();
        assert_eq!(q.priority, 1);

        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_query_scheduler_max_size() {
        let mut scheduler = QueryScheduler::new(2);

        scheduler.enqueue(MembershipQuery::new(Word::empty(), 10).with_priority(1));
        scheduler.enqueue(MembershipQuery::new(Word::from_str_slice(&["a"]), 10).with_priority(3));

        // This should replace priority-1 query
        scheduler.enqueue(MembershipQuery::new(Word::from_str_slice(&["b"]), 10).with_priority(5));

        assert_eq!(scheduler.len(), 2);

        let q1 = scheduler.dequeue().unwrap();
        assert_eq!(q1.priority, 5);
    }

    #[test]
    fn test_query_scheduler_batch_dequeue() {
        let mut scheduler = QueryScheduler::new(100);

        for i in 0..5 {
            scheduler.enqueue(MembershipQuery::new(Word::empty(), 10).with_priority(i));
        }

        let batch = scheduler.dequeue_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(scheduler.len(), 2);
    }

    #[test]
    fn test_samples_for_confidence() {
        let n = samples_for_confidence(0.1, 0.05, 10);
        assert!(n > 0);

        // Tighter bounds need more samples
        let n2 = samples_for_confidence(0.01, 0.05, 10);
        assert!(n2 > n);
    }

    #[test]
    fn test_confidence_from_samples() {
        let conf = confidence_from_samples(1000, 0.1, 10);
        assert!(conf > 0.0);
        assert!(conf <= 1.0);

        // More samples → higher confidence
        let conf2 = confidence_from_samples(10000, 0.1, 10);
        assert!(conf2 >= conf);
    }

    #[test]
    fn test_sub_distribution_sample() {
        let mut rng = rand::thread_rng();
        let dist = SubDistribution::from_map(
            [("a".to_string(), 0.5), ("b".to_string(), 0.5)]
                .into_iter().collect(),
        );

        let sample = dist.sample(&mut rng);
        assert!(sample.is_some());
        let s = sample.unwrap();
        assert!(s == "a" || s == "b");
    }

    #[test]
    fn test_sub_distribution_sample_empty() {
        let mut rng = rand::thread_rng();
        let dist = SubDistribution::new();
        assert!(dist.sample(&mut rng).is_none());
    }

    #[test]
    fn test_oracle_stats_default() {
        let stats = OracleStats::default();
        assert_eq!(stats.membership_queries, 0);
        assert_eq!(stats.equivalence_queries, 0);
    }

    #[test]
    fn test_mock_oracle_stats() {
        let mut oracle = MockOracle::two_state(make_alphabet());

        let q = MembershipQuery::new(Word::empty(), 10);
        oracle.membership_query(&q);
        oracle.membership_query(&q);

        assert_eq!(oracle.query_count(), 2);

        oracle.reset();
        assert_eq!(oracle.query_count(), 0);
    }

    #[test]
    fn test_word_concat() {
        let w1 = Word::from_str_slice(&["a"]);
        let w2 = Word::from_str_slice(&["b"]);
        let w3 = w1.concat(&w2);
        assert_eq!(w3.len(), 2);
    }

    #[test]
    fn test_query_scheduler_priority() {
        let mut sched = QueryScheduler::with_defaults();

        sched.schedule_membership(
            MembershipQuery::new(Word::from_str_slice(&["a"]), 10),
            1.0,
        );
        sched.schedule_membership(
            MembershipQuery::new(Word::from_str_slice(&["b"]), 10),
            5.0,
        );
        sched.schedule_membership(
            MembershipQuery::new(Word::from_str_slice(&["c"]), 10),
            3.0,
        );

        assert_eq!(sched.pending_count(), 3);
        assert!(sched.has_pending());

        let batch = sched.next_membership_batch();
        assert_eq!(batch.len(), 3);
        // Highest priority first
        assert_eq!(batch[0].word.symbols[0].0, "b");
    }

    #[test]
    fn test_query_scheduler_utilization() {
        let mut sched = QueryScheduler::with_defaults();
        sched.schedule_membership(
            MembershipQuery::new(Word::empty(), 5),
            1.0,
        );

        let _ = sched.next_membership_batch();
        assert!((sched.utilization() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_query_scheduler_clear() {
        let mut sched = QueryScheduler::with_defaults();
        sched.schedule_membership(MembershipQuery::new(Word::empty(), 5), 1.0);
        sched.schedule_equivalence(EquivalenceQuery::random_testing(100, 5), 1.0);
        assert_eq!(sched.pending_count(), 2);

        sched.clear();
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn test_replay_oracle() {
        let mut oracle = ReplayOracle::new();
        oracle.record(
            &Word::from_str_slice(&["a"]),
            SubDistribution::singleton("yes".to_string(), 0.8),
        );

        assert_eq!(oracle.num_recordings(), 1);

        let result = oracle.membership_query(
            &MembershipQuery::new(Word::from_str_slice(&["a"]), 10),
        );
        assert!((result.distribution.get("yes") - 0.8).abs() < 0.01);
        assert_eq!(oracle.num_replayed(), 1);
    }

    #[test]
    fn test_replay_oracle_fallback() {
        let mut oracle = ReplayOracle::with_fallback(
            SubDistribution::singleton("default".to_string(), 0.5),
        );
        let result = oracle.membership_query(
            &MembershipQuery::new(Word::from_str_slice(&["unknown"]), 10),
        );
        assert!((result.distribution.get("default") - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_oracle_metrics() {
        let mut metrics = OracleMetrics::new();
        metrics.total_membership_queries = 100;
        metrics.total_equivalence_queries = 10;
        metrics.cache_hits = 30;
        metrics.cache_misses = 70;
        metrics.counterexamples_found = 5;

        assert!((metrics.cache_hit_rate() - 0.3).abs() < 0.01);
        assert!((metrics.queries_per_counterexample() - 2.0).abs() < 0.01);
        let summary = metrics.summary();
        assert!(summary.contains("MQ=100"));
    }

    #[test]
    fn test_sample_size_hoeffding() {
        let n = SampleSizeCalculator::hoeffding(0.05, 0.05);
        assert!(n > 100);
        let n2 = SampleSizeCalculator::hoeffding(0.01, 0.01);
        assert!(n2 > n);
    }

    #[test]
    fn test_sample_size_ks_test() {
        let n = SampleSizeCalculator::for_ks_test(0.1, 0.05);
        assert!(n > 50);
    }

    #[test]
    fn test_sample_size_pac() {
        let n = SampleSizeCalculator::for_pac_learning(0.1, 0.1, 5);
        assert!(n > 10);
    }

    #[test]
    fn test_sample_size_dkw() {
        let n = SampleSizeCalculator::dkw_inequality(0.05, 0.05);
        assert!(n > 100);
    }
}
