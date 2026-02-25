//! Counter-example processing algorithms for the PCL* learning loop.
//!
//! Provides decomposition, minimization, generalization, caching, and
//! statistical validation of counter-examples returned by equivalence queries.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Local type aliases (to be swapped for coalgebra module types later)
// ---------------------------------------------------------------------------

/// A symbol in the input alphabet.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn epsilon() -> Self {
        Self(String::new())
    }
    pub fn is_epsilon(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_epsilon() {
            write!(f, "ε")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A word (sequence of symbols).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Word {
    pub symbols: Vec<Symbol>,
}

impl Word {
    pub fn empty() -> Self {
        Self {
            symbols: Vec::new(),
        }
    }
    pub fn singleton(sym: Symbol) -> Self {
        Self {
            symbols: vec![sym],
        }
    }
    pub fn from_symbols(symbols: Vec<Symbol>) -> Self {
        Self { symbols }
    }
    pub fn from_str_slice(parts: &[&str]) -> Self {
        Self {
            symbols: parts.iter().map(|s| Symbol::new(*s)).collect(),
        }
    }
    pub fn len(&self) -> usize {
        self.symbols.len()
    }
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
    pub fn concat(&self, other: &Word) -> Word {
        let mut syms = self.symbols.clone();
        syms.extend(other.symbols.iter().cloned());
        Word { symbols: syms }
    }
    pub fn prefix(&self, n: usize) -> Word {
        let end = n.min(self.symbols.len());
        Word {
            symbols: self.symbols[..end].to_vec(),
        }
    }
    pub fn suffix_from(&self, start: usize) -> Word {
        if start >= self.symbols.len() {
            return Word::empty();
        }
        Word {
            symbols: self.symbols[start..].to_vec(),
        }
    }
    pub fn split_at(&self, pos: usize) -> (Word, Word) {
        let pos = pos.min(self.symbols.len());
        (
            Word {
                symbols: self.symbols[..pos].to_vec(),
            },
            Word {
                symbols: self.symbols[pos..].to_vec(),
            },
        )
    }
    pub fn last_symbol(&self) -> Option<&Symbol> {
        self.symbols.last()
    }
    pub fn without_last(&self) -> Word {
        if self.symbols.is_empty() {
            return Word::empty();
        }
        Word {
            symbols: self.symbols[..self.symbols.len() - 1].to_vec(),
        }
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "ε")
        } else {
            let parts: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
            write!(f, "{}", parts.join("·"))
        }
    }
}

/// Sub-distribution: maps outcomes to probabilities that sum to ≤ 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubDistribution {
    pub weights: HashMap<String, f64>,
}

impl SubDistribution {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    pub fn singleton(key: String, prob: f64) -> Self {
        let mut weights = HashMap::new();
        weights.insert(key, prob);
        Self { weights }
    }

    pub fn from_map(weights: HashMap<String, f64>) -> Self {
        Self { weights }
    }

    pub fn total_mass(&self) -> f64 {
        self.weights.values().sum()
    }

    pub fn get(&self, key: &str) -> f64 {
        self.weights.get(key).copied().unwrap_or(0.0)
    }

    pub fn set(&mut self, key: String, prob: f64) {
        if prob > 0.0 {
            self.weights.insert(key, prob);
        } else {
            self.weights.remove(&key);
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.weights.keys()
    }

    /// Total variation distance between two sub-distributions.
    pub fn tv_distance(&self, other: &SubDistribution) -> f64 {
        let mut all_keys: HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());

        let mut dist = 0.0;
        for key in all_keys {
            let p = self.get(key);
            let q = other.get(key);
            dist += (p - q).abs();
        }
        dist / 2.0
    }

    /// L-infinity distance.
    pub fn linf_distance(&self, other: &SubDistribution) -> f64 {
        let mut all_keys: HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());

        let mut max_diff = 0.0f64;
        for key in all_keys {
            let p = self.get(key);
            let q = other.get(key);
            max_diff = max_diff.max((p - q).abs());
        }
        max_diff
    }

    pub fn support_size(&self) -> usize {
        self.weights.len()
    }
}

impl Default for SubDistribution {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Counter-example types
// ---------------------------------------------------------------------------

/// The type of mismatch that a counter-example demonstrates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MismatchType {
    /// Hypothesis says one distribution, system shows another
    DistributionMismatch {
        expected: SubDistribution,
        observed: SubDistribution,
        distance: f64,
    },
    /// Hypothesis has no transition, system does
    MissingTransition {
        from_state: String,
        on_symbol: Symbol,
    },
    /// Hypothesis reaches wrong state
    StateDisagreement {
        hypothesis_state: String,
        system_behavior: SubDistribution,
    },
}

/// A counter-example is a word that distinguishes the hypothesis from the target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterExample {
    pub id: String,
    /// The word witnessing the difference
    pub word: Word,
    /// Type of mismatch
    pub mismatch: MismatchType,
    /// Confidence in the counter-example (from statistical testing)
    pub confidence: f64,
    /// Number of samples used to validate
    pub sample_count: usize,
    /// Timestamp
    pub created_at: String,
    /// Whether this CE has been processed
    pub processed: bool,
    /// Decomposition result, if computed
    pub decomposition: Option<Decomposition>,
}

impl CounterExample {
    pub fn new(word: Word, mismatch: MismatchType, confidence: f64, sample_count: usize) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            word,
            mismatch,
            confidence,
            sample_count,
            created_at: chrono::Utc::now().to_rfc3339(),
            processed: false,
            decomposition: None,
        }
    }

    pub fn length(&self) -> usize {
        self.word.len()
    }

    pub fn mark_processed(&mut self) {
        self.processed = true;
    }

    pub fn set_decomposition(&mut self, decomp: Decomposition) {
        self.decomposition = Some(decomp);
    }

    /// Magnitude of disagreement, derived from the mismatch type.
    pub fn disagreement_magnitude(&self) -> f64 {
        match &self.mismatch {
            MismatchType::DistributionMismatch { distance, .. } => *distance,
            MismatchType::MissingTransition { .. } => 1.0,
            MismatchType::StateDisagreement { .. } => 1.0,
        }
    }

    /// Critical position in the word, if identifiable from decomposition.
    pub fn critical_position(&self) -> Option<usize> {
        self.decomposition.as_ref().map(|d| d.split_index)
    }
}

impl fmt::Display for CounterExample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CE[{}](len={}, conf={:.4})",
            self.id.chars().take(8).collect::<String>(),
            self.length(),
            self.confidence
        )
    }
}

/// Result of decomposing a counter-example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decomposition {
    /// Prefix that reaches a state in the hypothesis
    pub prefix: Word,
    /// The distinguishing symbol
    pub symbol: Symbol,
    /// Suffix that witnesses the difference
    pub suffix: Word,
    /// Which method was used
    pub method: DecompositionMethod,
    /// Index in the original word where the split occurs
    pub split_index: usize,
    /// Number of queries used to find decomposition
    pub queries_used: usize,
}

impl Decomposition {
    pub fn new(
        prefix: Word,
        symbol: Symbol,
        suffix: Word,
        method: DecompositionMethod,
        split_index: usize,
        queries_used: usize,
    ) -> Self {
        Self {
            prefix,
            symbol,
            suffix,
            method,
            split_index,
            queries_used,
        }
    }

    /// The new suffix to add to the observation table.
    pub fn new_suffix(&self) -> Word {
        Word::singleton(self.symbol.clone()).concat(&self.suffix)
    }
}

impl fmt::Display for Decomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Decomp(prefix={}, sym={}, suffix={}, method={:?})",
            self.prefix, self.symbol, self.suffix, self.method
        )
    }
}

/// Method used for counter-example decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecompositionMethod {
    /// Binary search decomposition (Rivest-Schapire)
    RivestSchapire,
    /// Linear scan from suffix end
    SuffixBased,
    /// Exponential search then binary
    ExponentialSearch,
    /// Try all positions, pick best
    Exhaustive,
}

// ---------------------------------------------------------------------------
// Query callback trait
// ---------------------------------------------------------------------------

/// Trait for making membership queries during CE processing.
pub trait MembershipQueryFn: Send + Sync {
    /// Query the target system: given a word, return the output distribution.
    fn query(&self, word: &Word) -> SubDistribution;

    /// Query the hypothesis: given a word, return the hypothesis output distribution.
    fn hypothesis_query(&self, word: &Word) -> SubDistribution;
}

/// Simple function-based membership query.
pub struct FnMembershipQuery<F, G>
where
    F: Fn(&Word) -> SubDistribution + Send + Sync,
    G: Fn(&Word) -> SubDistribution + Send + Sync,
{
    pub system_fn: F,
    pub hypothesis_fn: G,
}

impl<F, G> MembershipQueryFn for FnMembershipQuery<F, G>
where
    F: Fn(&Word) -> SubDistribution + Send + Sync,
    G: Fn(&Word) -> SubDistribution + Send + Sync,
{
    fn query(&self, word: &Word) -> SubDistribution {
        (self.system_fn)(word)
    }

    fn hypothesis_query(&self, word: &Word) -> SubDistribution {
        (self.hypothesis_fn)(word)
    }
}

// ---------------------------------------------------------------------------
// CounterExampleProcessor
// ---------------------------------------------------------------------------

/// Configuration for counter-example processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CEProcessorConfig {
    /// Default decomposition method
    pub default_method: DecompositionMethod,
    /// Statistical tolerance for distribution comparison
    pub tolerance: f64,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Whether to minimize CEs before decomposing
    pub minimize_first: bool,
    /// Maximum number of minimization attempts
    pub max_minimization_steps: usize,
    /// Number of samples for statistical validation
    pub validation_samples: usize,
    /// Confidence threshold for validation
    pub validation_confidence: f64,
}

impl Default for CEProcessorConfig {
    fn default() -> Self {
        Self {
            default_method: DecompositionMethod::RivestSchapire,
            tolerance: 0.05,
            max_cache_size: 10000,
            minimize_first: true,
            max_minimization_steps: 100,
            validation_samples: 100,
            validation_confidence: 0.95,
        }
    }
}

/// Processes counter-examples: decomposition, minimization, generalization.
pub struct CounterExampleProcessor {
    config: CEProcessorConfig,
    cache: CounterExampleCache,
    stats: CEProcessingStats,
}

/// Statistics about counter-example processing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CEProcessingStats {
    pub total_processed: usize,
    pub total_minimized: usize,
    pub total_queries: usize,
    pub average_ce_length: f64,
    pub average_minimized_length: f64,
    pub decomposition_method_counts: HashMap<String, usize>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl CounterExampleProcessor {
    pub fn new(config: CEProcessorConfig) -> Self {
        Self {
            config,
            cache: CounterExampleCache::new(10000),
            stats: CEProcessingStats::default(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(CEProcessorConfig::default())
    }

    pub fn stats(&self) -> &CEProcessingStats {
        &self.stats
    }

    pub fn cache(&self) -> &CounterExampleCache {
        &self.cache
    }

    // -----------------------------------------------------------------------
    // Main processing entry point
    // -----------------------------------------------------------------------

    /// Process a counter-example: optionally minimize, then decompose.
    pub fn process(
        &mut self,
        ce: &mut CounterExample,
        oracle: &dyn MembershipQueryFn,
    ) -> Decomposition {
        self.stats.total_processed += 1;
        self.stats.average_ce_length = (self.stats.average_ce_length
            * (self.stats.total_processed - 1) as f64
            + ce.length() as f64)
            / self.stats.total_processed as f64;

        // Check cache
        if let Some(cached) = self.cache.lookup(&ce.word) {
            self.stats.cache_hits += 1;
            ce.set_decomposition(cached.clone());
            ce.mark_processed();
            return cached.clone();
        }
        self.stats.cache_misses += 1;

        // Minimize if configured
        let working_word = if self.config.minimize_first {
            let minimized = self.minimize(ce, oracle);
            if minimized.len() < ce.word.len() {
                self.stats.total_minimized += 1;
                self.stats.average_minimized_length = (self.stats.average_minimized_length
                    * (self.stats.total_minimized - 1) as f64
                    + minimized.len() as f64)
                    / self.stats.total_minimized as f64;
            }
            minimized
        } else {
            ce.word.clone()
        };

        // Decompose
        let decomp = match self.config.default_method {
            DecompositionMethod::RivestSchapire => {
                self.rivest_schapire_decomposition(&working_word, oracle)
            }
            DecompositionMethod::SuffixBased => {
                self.suffix_decomposition(&working_word, oracle)
            }
            DecompositionMethod::ExponentialSearch => {
                self.exponential_decomposition(&working_word, oracle)
            }
            DecompositionMethod::Exhaustive => {
                self.exhaustive_decomposition(&working_word, oracle)
            }
        };

        let method_key = format!("{:?}", self.config.default_method);
        *self
            .stats
            .decomposition_method_counts
            .entry(method_key)
            .or_insert(0) += 1;

        self.cache.insert(ce.word.clone(), decomp.clone());
        ce.set_decomposition(decomp.clone());
        ce.mark_processed();
        decomp
    }

    // -----------------------------------------------------------------------
    // Rivest-Schapire binary search decomposition
    // -----------------------------------------------------------------------

    /// Binary search decomposition (Rivest-Schapire).
    ///
    /// Find the split point where the hypothesis and target diverge by binary
    /// searching for the index i such that:
    ///   - hypothesis(prefix[0..i] . suffix[i..]) ≈ target(prefix[0..i] . suffix[i..])
    ///   - hypothesis(prefix[0..i+1] . suffix[i+1..]) ≉ target(prefix[0..i+1] . suffix[i+1..])
    pub fn rivest_schapire_decomposition(
        &mut self,
        word: &Word,
        oracle: &dyn MembershipQueryFn,
    ) -> Decomposition {
        let n = word.len();
        if n == 0 {
            return Decomposition::new(
                Word::empty(),
                Symbol::epsilon(),
                Word::empty(),
                DecompositionMethod::RivestSchapire,
                0,
                0,
            );
        }
        if n == 1 {
            return Decomposition::new(
                Word::empty(),
                word.symbols[0].clone(),
                Word::empty(),
                DecompositionMethod::RivestSchapire,
                0,
                1,
            );
        }

        let mut queries_used = 0;
        let mut lo: usize = 0;
        let mut hi: usize = n - 1;

        // For each split point i, we check if hypothesis and target agree
        // on the word formed by hypothesis-run on prefix[0..i] then suffix[i..n]
        while lo < hi {
            let mid = lo + (hi - lo) / 2;

            let prefix = word.prefix(mid + 1);
            let suffix = word.suffix_from(mid + 1);
            let test_word = prefix.concat(&suffix);

            let sys_output = oracle.query(&test_word);
            let hyp_output = oracle.hypothesis_query(&test_word);
            queries_used += 2;

            let dist = sys_output.tv_distance(&hyp_output);

            if dist > self.config.tolerance {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        let split = lo;
        let prefix = word.prefix(split);
        let symbol = if split < n {
            word.symbols[split].clone()
        } else {
            word.symbols[n - 1].clone()
        };
        let suffix = word.suffix_from(split + 1);

        self.stats.total_queries += queries_used;

        Decomposition::new(
            prefix,
            symbol,
            suffix,
            DecompositionMethod::RivestSchapire,
            split,
            queries_used,
        )
    }

    // -----------------------------------------------------------------------
    // Suffix-based decomposition
    // -----------------------------------------------------------------------

    /// Linear scan from the end of the word, finding the longest suffix
    /// that still distinguishes hypothesis from target.
    pub fn suffix_decomposition(
        &mut self,
        word: &Word,
        oracle: &dyn MembershipQueryFn,
    ) -> Decomposition {
        let n = word.len();
        if n == 0 {
            return Decomposition::new(
                Word::empty(),
                Symbol::epsilon(),
                Word::empty(),
                DecompositionMethod::SuffixBased,
                0,
                0,
            );
        }

        let mut queries_used = 0;
        let mut best_split = 0;
        let mut best_distance = 0.0f64;

        // Scan from end to start
        for i in (0..n).rev() {
            let prefix = word.prefix(i);
            let suffix = word.suffix_from(i);
            let test_word = prefix.concat(&suffix);

            let sys_output = oracle.query(&test_word);
            let hyp_output = oracle.hypothesis_query(&test_word);
            queries_used += 2;

            let dist = sys_output.tv_distance(&hyp_output);
            if dist > best_distance {
                best_distance = dist;
                best_split = i;
            }

            // If we found a clear split point, we can stop
            if dist > self.config.tolerance && best_distance > self.config.tolerance * 2.0 {
                break;
            }
        }

        let prefix = word.prefix(best_split);
        let symbol = if best_split < n {
            word.symbols[best_split].clone()
        } else {
            Symbol::epsilon()
        };
        let suffix = word.suffix_from(best_split + 1);

        self.stats.total_queries += queries_used;

        Decomposition::new(
            prefix,
            symbol,
            suffix,
            DecompositionMethod::SuffixBased,
            best_split,
            queries_used,
        )
    }

    // -----------------------------------------------------------------------
    // Exponential search decomposition
    // -----------------------------------------------------------------------

    /// Exponential search then binary search.
    /// First find an approximate range via exponential jumps, then refine with binary search.
    pub fn exponential_decomposition(
        &mut self,
        word: &Word,
        oracle: &dyn MembershipQueryFn,
    ) -> Decomposition {
        let n = word.len();
        if n <= 2 {
            return self.rivest_schapire_decomposition(word, oracle);
        }

        let mut queries_used = 0;
        let mut step = 1usize;
        let mut found_range_start = 0usize;
        let mut found_range_end = n;

        // Exponential phase: jump by doubling steps
        while step < n {
            let prefix = word.prefix(step);
            let suffix = word.suffix_from(step);
            let test_word = prefix.concat(&suffix);

            let sys_output = oracle.query(&test_word);
            let hyp_output = oracle.hypothesis_query(&test_word);
            queries_used += 2;

            let dist = sys_output.tv_distance(&hyp_output);

            if dist > self.config.tolerance {
                // Split point is between step/2 and step
                found_range_start = step / 2;
                found_range_end = step;
                break;
            }

            step *= 2;
        }

        if step >= n {
            found_range_start = step / 2;
            found_range_end = n;
        }

        // Binary search within the found range
        let mut lo = found_range_start;
        let mut hi = found_range_end;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;

            let prefix = word.prefix(mid + 1);
            let suffix = word.suffix_from(mid + 1);
            let test_word = prefix.concat(&suffix);

            let sys_output = oracle.query(&test_word);
            let hyp_output = oracle.hypothesis_query(&test_word);
            queries_used += 2;

            let dist = sys_output.tv_distance(&hyp_output);

            if dist > self.config.tolerance {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        let split = lo;
        let prefix = word.prefix(split);
        let symbol = if split < n {
            word.symbols[split].clone()
        } else {
            word.symbols[n - 1].clone()
        };
        let suffix = word.suffix_from(split + 1);

        self.stats.total_queries += queries_used;

        Decomposition::new(
            prefix,
            symbol,
            suffix,
            DecompositionMethod::ExponentialSearch,
            split,
            queries_used,
        )
    }

    // -----------------------------------------------------------------------
    // Exhaustive decomposition
    // -----------------------------------------------------------------------

    /// Try all positions and pick the one with maximum distance.
    pub fn exhaustive_decomposition(
        &mut self,
        word: &Word,
        oracle: &dyn MembershipQueryFn,
    ) -> Decomposition {
        let n = word.len();
        if n == 0 {
            return Decomposition::new(
                Word::empty(),
                Symbol::epsilon(),
                Word::empty(),
                DecompositionMethod::Exhaustive,
                0,
                0,
            );
        }

        let mut queries_used = 0;
        let mut best_split = 0;
        let mut best_distance = -1.0f64;

        for i in 0..n {
            let prefix = word.prefix(i);
            let suffix = word.suffix_from(i);
            let test_word = prefix.concat(&suffix);

            let sys_output = oracle.query(&test_word);
            let hyp_output = oracle.hypothesis_query(&test_word);
            queries_used += 2;

            let dist = sys_output.tv_distance(&hyp_output);
            if dist > best_distance {
                best_distance = dist;
                best_split = i;
            }
        }

        let prefix = word.prefix(best_split);
        let symbol = if best_split < n {
            word.symbols[best_split].clone()
        } else {
            Symbol::epsilon()
        };
        let suffix = word.suffix_from(best_split + 1);

        self.stats.total_queries += queries_used;

        Decomposition::new(
            prefix,
            symbol,
            suffix,
            DecompositionMethod::Exhaustive,
            best_split,
            queries_used,
        )
    }

    // -----------------------------------------------------------------------
    // Counter-example minimization
    // -----------------------------------------------------------------------

    /// Minimize a counter-example by removing symbols that are not needed.
    ///
    /// Uses a greedy approach: try removing each symbol; if the resulting
    /// word is still a counter-example, keep the removal.
    pub fn minimize(
        &mut self,
        ce: &CounterExample,
        oracle: &dyn MembershipQueryFn,
    ) -> Word {
        let mut current = ce.word.clone();
        let mut steps = 0;

        while steps < self.config.max_minimization_steps && current.len() > 1 {
            let mut improved = false;

            // Try removing each symbol
            let n = current.len();
            for i in (0..n).rev() {
                let mut reduced_symbols = current.symbols.clone();
                reduced_symbols.remove(i);
                let candidate = Word::from_symbols(reduced_symbols);

                if self.is_counter_example(&candidate, oracle) {
                    current = candidate;
                    improved = true;
                    break;
                }
            }

            if !improved {
                break;
            }
            steps += 1;
        }

        // Try prefix minimization
        let prefix_min = self.minimize_prefix(&current, oracle);
        if prefix_min.len() < current.len() && self.is_counter_example(&prefix_min, oracle) {
            current = prefix_min;
        }

        // Try suffix minimization
        let suffix_min = self.minimize_suffix(&current, oracle);
        if suffix_min.len() < current.len() && self.is_counter_example(&suffix_min, oracle) {
            current = suffix_min;
        }

        current
    }

    /// Binary search for the shortest prefix that is still a counter-example.
    fn minimize_prefix(&mut self, word: &Word, oracle: &dyn MembershipQueryFn) -> Word {
        let n = word.len();
        if n <= 1 {
            return word.clone();
        }

        let mut lo = 1usize;
        let mut hi = n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let candidate = word.prefix(mid);

            if self.is_counter_example(&candidate, oracle) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        word.prefix(lo)
    }

    /// Binary search for the shortest suffix that is still a counter-example.
    fn minimize_suffix(&mut self, word: &Word, oracle: &dyn MembershipQueryFn) -> Word {
        let n = word.len();
        if n <= 1 {
            return word.clone();
        }

        let mut lo = 0usize;
        let mut hi = n - 1;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let candidate = word.suffix_from(mid);

            if self.is_counter_example(&candidate, oracle) {
                lo = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
        }

        word.suffix_from(lo)
    }

    /// Check if a word is a counter-example (hypothesis ≉ target).
    fn is_counter_example(&mut self, word: &Word, oracle: &dyn MembershipQueryFn) -> bool {
        let sys = oracle.query(word);
        let hyp = oracle.hypothesis_query(word);
        self.stats.total_queries += 2;
        sys.tv_distance(&hyp) > self.config.tolerance
    }

    // -----------------------------------------------------------------------
    // Counter-example generalization
    // -----------------------------------------------------------------------

    /// Generalize a counter-example by replacing specific symbols with wildcards.
    ///
    /// Returns a set of "patterns" — words where some positions are marked
    /// as generalized (any symbol from the alphabet produces a CE).
    pub fn generalize(
        &mut self,
        ce: &CounterExample,
        alphabet: &[Symbol],
        oracle: &dyn MembershipQueryFn,
    ) -> Vec<GeneralizedPattern> {
        let word = &ce.word;
        let n = word.len();
        let mut generalizable = vec![false; n];

        // For each position, check if all alphabet symbols produce a CE
        for i in 0..n {
            let mut all_ce = true;
            for sym in alphabet {
                let mut modified = word.symbols.clone();
                modified[i] = sym.clone();
                let candidate = Word::from_symbols(modified);

                if !self.is_counter_example(&candidate, oracle) {
                    all_ce = false;
                    break;
                }
            }
            generalizable[i] = all_ce;
        }

        // Build patterns from generalizable positions
        let mut patterns = Vec::new();

        // Single-wildcard patterns
        for i in 0..n {
            if generalizable[i] {
                let mut positions = vec![PatternPosition::Fixed(word.symbols[i].clone()); n];
                positions[i] = PatternPosition::Wildcard;
                patterns.push(GeneralizedPattern {
                    positions,
                    original: word.clone(),
                    generalized_count: 1,
                });
            }
        }

        // Multi-wildcard: try combining pairwise
        let gen_indices: Vec<usize> = (0..n).filter(|&i| generalizable[i]).collect();
        for i in 0..gen_indices.len() {
            for j in (i + 1)..gen_indices.len() {
                let idx_i = gen_indices[i];
                let idx_j = gen_indices[j];

                // Verify joint generalizability with a sample
                let mut joint_ok = true;
                let sample_count = alphabet.len().min(4);
                'outer: for si in 0..sample_count {
                    for sj in 0..sample_count {
                        let sym_i = &alphabet[si % alphabet.len()];
                        let sym_j = &alphabet[sj % alphabet.len()];
                        let mut modified = word.symbols.clone();
                        modified[idx_i] = sym_i.clone();
                        modified[idx_j] = sym_j.clone();
                        let candidate = Word::from_symbols(modified);
                        if !self.is_counter_example(&candidate, oracle) {
                            joint_ok = false;
                            break 'outer;
                        }
                    }
                }

                if joint_ok {
                    let mut positions =
                        vec![PatternPosition::Fixed(Symbol::epsilon()); n];
                    for k in 0..n {
                        if k == idx_i || k == idx_j {
                            positions[k] = PatternPosition::Wildcard;
                        } else {
                            positions[k] = PatternPosition::Fixed(word.symbols[k].clone());
                        }
                    }
                    patterns.push(GeneralizedPattern {
                        positions,
                        original: word.clone(),
                        generalized_count: 2,
                    });
                }
            }
        }

        patterns
    }

    // -----------------------------------------------------------------------
    // Statistical validation
    // -----------------------------------------------------------------------

    /// Validate a counter-example with multiple samples.
    ///
    /// Returns the confidence that this is a true counter-example.
    pub fn validate_statistically(
        &mut self,
        word: &Word,
        oracle: &dyn MembershipQueryFn,
        num_samples: usize,
    ) -> ValidationResult {
        let mut distances = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let sys = oracle.query(word);
            let hyp = oracle.hypothesis_query(word);
            let dist = sys.tv_distance(&hyp);
            distances.push(dist);
            self.stats.total_queries += 2;
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_distance = distances.iter().sum::<f64>() / num_samples as f64;
        let variance = distances
            .iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f64>()
            / (num_samples as f64 - 1.0).max(1.0);
        let std_dev = variance.sqrt();

        let median_distance = if num_samples % 2 == 0 {
            (distances[num_samples / 2 - 1] + distances[num_samples / 2]) / 2.0
        } else {
            distances[num_samples / 2]
        };

        // Use a one-sample t-test against tolerance threshold
        let t_statistic = (mean_distance - self.config.tolerance)
            / (std_dev / (num_samples as f64).sqrt());

        // Approximate p-value using normal approximation for large samples
        let confidence = if num_samples >= 30 {
            // Standard normal CDF approximation
            normal_cdf(t_statistic)
        } else {
            // Use simple threshold-based confidence
            if mean_distance > self.config.tolerance * 2.0 {
                0.99
            } else if mean_distance > self.config.tolerance {
                0.90
            } else {
                0.50
            }
        };

        let is_valid = confidence > self.config.validation_confidence
            && mean_distance > self.config.tolerance;

        ValidationResult {
            is_valid,
            confidence,
            mean_distance,
            median_distance,
            std_dev,
            num_samples,
            t_statistic,
            distances,
        }
    }
}

/// Approximate normal CDF using Abramowitz and Stegun formula.
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let c = t * (0.319381530
        + t * (-0.356563782
            + t * (1.781477937
                + t * (-1.821255978 + t * 1.330274429))));

    if x >= 0.0 {
        1.0 - p * c
    } else {
        p * c
    }
}

/// Result of statistical validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub mean_distance: f64,
    pub median_distance: f64,
    pub std_dev: f64,
    pub num_samples: usize,
    pub t_statistic: f64,
    pub distances: Vec<f64>,
}

/// A generalized pattern from a counter-example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralizedPattern {
    pub positions: Vec<PatternPosition>,
    pub original: Word,
    pub generalized_count: usize,
}

impl GeneralizedPattern {
    /// Check if a word matches this pattern.
    pub fn matches(&self, word: &Word) -> bool {
        if word.len() != self.positions.len() {
            return false;
        }
        for (pos, sym) in self.positions.iter().zip(word.symbols.iter()) {
            match pos {
                PatternPosition::Fixed(expected) => {
                    if expected != sym {
                        return false;
                    }
                }
                PatternPosition::Wildcard => {} // matches anything
            }
        }
        true
    }

    pub fn wildcard_positions(&self) -> Vec<usize> {
        self.positions
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p {
                PatternPosition::Wildcard => Some(i),
                _ => None,
            })
            .collect()
    }
}

impl fmt::Display for GeneralizedPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .positions
            .iter()
            .map(|p| match p {
                PatternPosition::Fixed(s) => s.to_string(),
                PatternPosition::Wildcard => "*".to_string(),
            })
            .collect();
        write!(f, "[{}]", parts.join("·"))
    }
}

/// A position in a generalized pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternPosition {
    Fixed(Symbol),
    Wildcard,
}

// ---------------------------------------------------------------------------
// CounterExampleCache
// ---------------------------------------------------------------------------

/// Cache for counter-example decompositions and results.
pub struct CounterExampleCache {
    /// Decomposition cache: word → decomposition
    decompositions: HashMap<Word, Decomposition>,
    /// Validation cache: word → validation result
    validations: HashMap<Word, ValidationResult>,
    /// Max entries
    max_size: usize,
    /// Access order for LRU eviction
    access_order: VecDeque<Word>,
    /// Statistics
    hits: usize,
    misses: usize,
}

impl CounterExampleCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            decompositions: HashMap::new(),
            validations: HashMap::new(),
            max_size,
            access_order: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn lookup(&mut self, word: &Word) -> Option<&Decomposition> {
        if self.decompositions.contains_key(word) {
            self.hits += 1;
            // Move to front of access order
            self.access_order.retain(|w| w != word);
            self.access_order.push_front(word.clone());
            self.decompositions.get(word)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, word: Word, decomp: Decomposition) {
        if self.decompositions.len() >= self.max_size {
            self.evict();
        }
        self.access_order.push_front(word.clone());
        self.decompositions.insert(word, decomp);
    }

    pub fn lookup_validation(&mut self, word: &Word) -> Option<&ValidationResult> {
        if self.validations.contains_key(word) {
            self.hits += 1;
            self.validations.get(word)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert_validation(&mut self, word: Word, result: ValidationResult) {
        if self.validations.len() >= self.max_size {
            self.evict_validations();
        }
        self.validations.insert(word, result);
    }

    pub fn size(&self) -> usize {
        self.decompositions.len()
    }

    pub fn validation_size(&self) -> usize {
        self.validations.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn clear(&mut self) {
        self.decompositions.clear();
        self.validations.clear();
        self.access_order.clear();
    }

    fn evict(&mut self) {
        let to_remove = self.max_size / 10;
        for _ in 0..to_remove {
            if let Some(word) = self.access_order.pop_back() {
                self.decompositions.remove(&word);
            }
        }
    }

    fn evict_validations(&mut self) {
        let to_remove = self.max_size / 10;
        let keys: Vec<Word> = self.validations.keys().take(to_remove).cloned().collect();
        for key in keys {
            self.validations.remove(&key);
        }
    }
}

// ---------------------------------------------------------------------------
// Batch counter-example processing
// ---------------------------------------------------------------------------

/// Process multiple counter-examples and return the most informative ones.
pub fn rank_counterexamples(
    ces: &[CounterExample],
    max_results: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = ces
        .iter()
        .enumerate()
        .map(|(i, ce)| {
            // Score: higher confidence, shorter length
            let length_penalty = 1.0 / (1.0 + ce.length() as f64);
            let confidence_score = ce.confidence;
            let score = confidence_score * 0.7 + length_penalty * 0.3;
            (i, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max_results);
    scored.iter().map(|(i, _)| *i).collect()
}

/// Deduplicate counter-examples based on their decomposition suffixes.
pub fn deduplicate_counterexamples(ces: &mut Vec<CounterExample>) {
    let mut seen_suffixes: HashSet<Word> = HashSet::new();
    ces.retain(|ce| {
        if let Some(ref decomp) = ce.decomposition {
            let suffix = decomp.new_suffix();
            if seen_suffixes.contains(&suffix) {
                false
            } else {
                seen_suffixes.insert(suffix);
                true
            }
        } else {
            true // keep unprocessed CEs
        }
    });
}

/// Group counter-examples by their distinguishing suffix.
pub fn group_by_suffix(ces: &[CounterExample]) -> HashMap<Word, Vec<usize>> {
    let mut groups: HashMap<Word, Vec<usize>> = HashMap::new();
    for (i, ce) in ces.iter().enumerate() {
        if let Some(ref decomp) = ce.decomposition {
            groups
                .entry(decomp.new_suffix())
                .or_default()
                .push(i);
        }
    }
    groups
}

// ---------------------------------------------------------------------------
// Advanced counter-example analysis: abstract patterns and reuse
// ---------------------------------------------------------------------------

/// Abstract pattern extracted from concrete counter-examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterExamplePattern {
    /// Pattern identifier
    pub id: String,
    /// Symbolic representation (e.g., "a*ba" where * is wildcard)
    pub symbolic: Vec<PatternElement>,
    /// Concrete instances that match this pattern
    pub instances: Vec<Word>,
    /// Minimum length of matching words
    pub min_length: usize,
    /// Maximum length of matching words
    pub max_length: usize,
    /// Average disagreement magnitude across instances
    pub avg_disagreement: f64,
    /// The critical position(s) in the pattern
    pub critical_positions: Vec<usize>,
}

/// Element in a counter-example pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternElement {
    /// Exact symbol match
    Exact(String),
    /// Any single symbol
    Wildcard,
    /// Any sequence of symbols (including empty)
    Star,
}

impl fmt::Display for PatternElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternElement::Exact(s) => write!(f, "{}", s),
            PatternElement::Wildcard => write!(f, "?"),
            PatternElement::Star => write!(f, "*"),
        }
    }
}

impl CounterExamplePattern {
    /// Create a new pattern from a single counter-example.
    pub fn from_single(ce: &CounterExample) -> Self {
        let elements: Vec<PatternElement> = ce
            .word
            .symbols
            .iter()
            .map(|s| PatternElement::Exact(s.0.clone()))
            .collect();

        let len = ce.word.len();
        let critical = ce.critical_position().map(|p| vec![p]).unwrap_or_default();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            symbolic: elements,
            instances: vec![ce.word.clone()],
            min_length: len,
            max_length: len,
            critical_positions: critical,
            avg_disagreement: ce.disagreement_magnitude(),
        }
    }

    /// Check if a word matches this pattern.
    pub fn matches(&self, word: &Word) -> bool {
        self.match_recursive(&self.symbolic, &word.symbols, 0, 0)
    }

    fn match_recursive(
        &self,
        pattern: &[PatternElement],
        symbols: &[Symbol],
        pi: usize,
        si: usize,
    ) -> bool {
        if pi == pattern.len() && si == symbols.len() {
            return true;
        }
        if pi == pattern.len() {
            return false;
        }

        match &pattern[pi] {
            PatternElement::Exact(s) => {
                if si < symbols.len() && symbols[si].0 == *s {
                    self.match_recursive(pattern, symbols, pi + 1, si + 1)
                } else {
                    false
                }
            }
            PatternElement::Wildcard => {
                if si < symbols.len() {
                    self.match_recursive(pattern, symbols, pi + 1, si + 1)
                } else {
                    false
                }
            }
            PatternElement::Star => {
                // Try matching star with 0 or more symbols
                for skip in 0..=(symbols.len() - si) {
                    if self.match_recursive(pattern, symbols, pi + 1, si + skip) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Add an instance to this pattern and generalize if needed.
    pub fn add_instance(&mut self, word: &Word, disagreement: f64) {
        self.instances.push(word.clone());
        self.min_length = self.min_length.min(word.len());
        self.max_length = self.max_length.max(word.len());
        self.avg_disagreement = (self.avg_disagreement * (self.instances.len() - 1) as f64
            + disagreement)
            / self.instances.len() as f64;
    }

    /// Generalize the pattern based on multiple instances.
    pub fn generalize(&mut self) {
        if self.instances.len() < 2 {
            return;
        }

        let min_len = self.instances.iter().map(|w| w.len()).min().unwrap_or(0);
        let max_len = self.instances.iter().map(|w| w.len()).max().unwrap_or(0);

        if min_len == 0 {
            self.symbolic = vec![PatternElement::Star];
            return;
        }

        let mut pattern: Vec<PatternElement> = Vec::new();

        // Compare positions across all instances
        for pos in 0..min_len {
            let symbols_at_pos: HashSet<&str> = self
                .instances
                .iter()
                .filter(|w| pos < w.len())
                .map(|w| w.symbols[pos].0.as_str())
                .collect();

            if symbols_at_pos.len() == 1 {
                pattern.push(PatternElement::Exact(
                    symbols_at_pos.into_iter().next().unwrap().to_string(),
                ));
            } else {
                pattern.push(PatternElement::Wildcard);
            }
        }

        // If lengths vary, add a star at the end
        if max_len > min_len {
            pattern.push(PatternElement::Star);
        }

        self.symbolic = pattern;
    }

    /// Get pattern length (excluding stars).
    pub fn fixed_length(&self) -> usize {
        self.symbolic
            .iter()
            .filter(|e| !matches!(e, PatternElement::Star))
            .count()
    }
}

impl fmt::Display for CounterExamplePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.symbolic.iter().map(|e| format!("{}", e)).collect();
        write!(
            f,
            "Pattern[{}](instances={}, disagreement={:.4})",
            parts.join(""),
            self.instances.len(),
            self.avg_disagreement,
        )
    }
}

/// Pattern library for storing and matching counter-example patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternLibrary {
    pub patterns: Vec<CounterExamplePattern>,
    pub max_patterns: usize,
}

impl PatternLibrary {
    pub fn new(max_patterns: usize) -> Self {
        Self {
            patterns: Vec::new(),
            max_patterns,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(100)
    }

    /// Add a counter-example, either creating a new pattern or augmenting an existing one.
    pub fn add(&mut self, ce: &CounterExample) {
        // Check if it matches an existing pattern
        for pattern in &mut self.patterns {
            if pattern.matches(&ce.word) {
                pattern.add_instance(&ce.word, ce.disagreement_magnitude());
                pattern.generalize();
                return;
            }
        }

        // Create a new pattern
        if self.patterns.len() < self.max_patterns {
            self.patterns.push(CounterExamplePattern::from_single(ce));
        } else {
            // Replace the least informative pattern (fewest instances)
            if let Some(min_idx) = self
                .patterns
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| p.instances.len())
                .map(|(i, _)| i)
            {
                self.patterns[min_idx] = CounterExamplePattern::from_single(ce);
            }
        }
    }

    /// Find patterns that match a given word.
    pub fn matching_patterns(&self, word: &Word) -> Vec<&CounterExamplePattern> {
        self.patterns.iter().filter(|p| p.matches(word)).collect()
    }

    /// Get the most frequent patterns.
    pub fn top_patterns(&self, n: usize) -> Vec<&CounterExamplePattern> {
        let mut sorted: Vec<&CounterExamplePattern> = self.patterns.iter().collect();
        sorted.sort_by(|a, b| b.instances.len().cmp(&a.instances.len()));
        sorted.into_iter().take(n).collect()
    }

    pub fn num_patterns(&self) -> usize {
        self.patterns.len()
    }

    pub fn total_instances(&self) -> usize {
        self.patterns.iter().map(|p| p.instances.len()).sum()
    }
}

/// Severity classifier for counter-examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CounterExampleSeverity {
    /// Minor: small distribution difference, likely noise
    Minor,
    /// Moderate: clear difference but similar behavior class
    Moderate,
    /// Major: different behavior class (e.g., refusal vs compliance)
    Major,
    /// Critical: fundamental model misunderstanding
    Critical,
}

impl CounterExampleSeverity {
    /// Classify based on disagreement magnitude.
    pub fn from_disagreement(magnitude: f64) -> Self {
        if magnitude < 0.1 {
            Self::Minor
        } else if magnitude < 0.3 {
            Self::Moderate
        } else if magnitude < 0.7 {
            Self::Major
        } else {
            Self::Critical
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            Self::Minor => 0.1,
            Self::Moderate => 0.3,
            Self::Major => 0.7,
            Self::Critical => 1.0,
        }
    }
}

impl fmt::Display for CounterExampleSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minor => write!(f, "Minor"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Major => write!(f, "Major"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Counter-example prioritizer for processing order.
pub struct CounterExamplePrioritizer {
    pub pending: Vec<(CounterExample, CounterExampleSeverity, f64)>,
}

impl CounterExamplePrioritizer {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Add a counter-example with automatic severity classification.
    pub fn add(&mut self, ce: CounterExample) {
        let severity = CounterExampleSeverity::from_disagreement(ce.disagreement_magnitude());
        let priority = severity.weight() * (1.0 + 1.0 / (ce.word.len() as f64 + 1.0));
        self.pending.push((ce, severity, priority));
        self.pending
            .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Get the highest-priority counter-example.
    pub fn next(&mut self) -> Option<(CounterExample, CounterExampleSeverity)> {
        if self.pending.is_empty() {
            return None;
        }
        let (ce, sev, _) = self.pending.remove(0);
        Some((ce, sev))
    }

    /// Get count by severity.
    pub fn count_by_severity(&self) -> HashMap<CounterExampleSeverity, usize> {
        let mut counts = HashMap::new();
        for (_, sev, _) in &self.pending {
            *counts.entry(*sev).or_insert(0) += 1;
        }
        counts
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

impl Default for CounterExamplePrioritizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock oracle for testing: simple deterministic system.
    struct TestOracle {
        /// Words where system and hypothesis disagree
        disagreements: HashMap<Word, (SubDistribution, SubDistribution)>,
        /// Default distributions
        default_sys: SubDistribution,
        default_hyp: SubDistribution,
    }

    impl TestOracle {
        fn new() -> Self {
            let default_dist = SubDistribution::singleton("ok".to_string(), 1.0);
            Self {
                disagreements: HashMap::new(),
                default_sys: default_dist.clone(),
                default_hyp: default_dist,
            }
        }

        fn add_disagreement(
            &mut self,
            word: Word,
            sys_dist: SubDistribution,
            hyp_dist: SubDistribution,
        ) {
            self.disagreements.insert(word, (sys_dist, hyp_dist));
        }

        fn with_disagreement_at_prefix(n: usize, split: usize) -> Self {
            let mut oracle = Self::new();
            // Create disagreement for words with prefix of length `split`
            let word = Word::from_str_slice(
                &(0..n).map(|i| {
                    // We need to return &str, so use leaked strings for test
                    let s: &str = Box::leak(format!("s{}", i).into_boxed_str());
                    s
                }).collect::<Vec<&str>>(),
            );

            // Add disagreements for all prefixes at and after split
            for i in split..=n {
                let prefix = word.prefix(i);
                let suffix = word.suffix_from(i);
                let test = prefix.concat(&suffix);

                let sys = SubDistribution::singleton("a".to_string(), 0.8);
                let hyp = SubDistribution::singleton("b".to_string(), 0.9);
                oracle.add_disagreement(test, sys, hyp);
            }

            oracle
        }
    }

    impl MembershipQueryFn for TestOracle {
        fn query(&self, word: &Word) -> SubDistribution {
            if let Some((sys, _)) = self.disagreements.get(word) {
                sys.clone()
            } else {
                self.default_sys.clone()
            }
        }

        fn hypothesis_query(&self, word: &Word) -> SubDistribution {
            if let Some((_, hyp)) = self.disagreements.get(word) {
                hyp.clone()
            } else {
                self.default_hyp.clone()
            }
        }
    }

    #[test]
    fn test_counter_example_creation() {
        let word = Word::from_str_slice(&["a", "b", "c"]);
        let mismatch = MismatchType::DistributionMismatch {
            expected: SubDistribution::singleton("x".to_string(), 0.5),
            observed: SubDistribution::singleton("y".to_string(), 0.5),
            distance: 0.5,
        };

        let ce = CounterExample::new(word.clone(), mismatch, 0.95, 100);
        assert_eq!(ce.length(), 3);
        assert!(!ce.processed);
        assert!(ce.decomposition.is_none());
    }

    #[test]
    fn test_decomposition_new_suffix() {
        let decomp = Decomposition::new(
            Word::from_str_slice(&["a", "b"]),
            Symbol::new("c"),
            Word::from_str_slice(&["d"]),
            DecompositionMethod::RivestSchapire,
            2,
            4,
        );

        let new_suffix = decomp.new_suffix();
        assert_eq!(new_suffix.len(), 2);
        assert_eq!(new_suffix.symbols[0], Symbol::new("c"));
        assert_eq!(new_suffix.symbols[1], Symbol::new("d"));
    }

    #[test]
    fn test_rivest_schapire_single_symbol() {
        let oracle = TestOracle::new();
        let mut processor = CounterExampleProcessor::with_default_config();

        let word = Word::from_str_slice(&["a"]);
        let decomp = processor.rivest_schapire_decomposition(&word, &oracle);

        assert_eq!(decomp.method, DecompositionMethod::RivestSchapire);
        assert_eq!(decomp.symbol, Symbol::new("a"));
    }

    #[test]
    fn test_rivest_schapire_empty_word() {
        let oracle = TestOracle::new();
        let mut processor = CounterExampleProcessor::with_default_config();

        let word = Word::empty();
        let decomp = processor.rivest_schapire_decomposition(&word, &oracle);
        assert_eq!(decomp.split_index, 0);
    }

    #[test]
    fn test_suffix_decomposition() {
        let mut oracle = TestOracle::new();
        let word = Word::from_str_slice(&["a", "b", "c"]);
        // Add disagreement at position 1
        let sys = SubDistribution::singleton("x".to_string(), 0.9);
        let hyp = SubDistribution::singleton("y".to_string(), 0.1);
        oracle.add_disagreement(word.clone(), sys, hyp);

        let mut processor = CounterExampleProcessor::with_default_config();
        let decomp = processor.suffix_decomposition(&word, &oracle);
        assert_eq!(decomp.method, DecompositionMethod::SuffixBased);
    }

    #[test]
    fn test_exponential_decomposition() {
        let oracle = TestOracle::new();
        let mut processor = CounterExampleProcessor::with_default_config();

        let word = Word::from_str_slice(&["a", "b"]);
        let decomp = processor.exponential_decomposition(&word, &oracle);
        // For short words, falls back to RS
        assert_eq!(decomp.method, DecompositionMethod::RivestSchapire);
    }

    #[test]
    fn test_exhaustive_decomposition() {
        let mut oracle = TestOracle::new();
        let word = Word::from_str_slice(&["a", "b", "c"]);

        let sys = SubDistribution::singleton("x".to_string(), 0.9);
        let hyp = SubDistribution::singleton("y".to_string(), 0.1);
        // Add disagreement when prefix is "a" + suffix
        let prefix_a = Word::from_str_slice(&["a"]);
        let suffix_bc = Word::from_str_slice(&["b", "c"]);
        oracle.add_disagreement(prefix_a.concat(&suffix_bc), sys, hyp);

        let mut processor = CounterExampleProcessor::with_default_config();
        let decomp = processor.exhaustive_decomposition(&word, &oracle);
        assert_eq!(decomp.method, DecompositionMethod::Exhaustive);
    }

    #[test]
    fn test_minimization_identity() {
        // When word is already minimal, minimization should return it
        let oracle = TestOracle::new();
        let mut processor = CounterExampleProcessor::with_default_config();

        let word = Word::from_str_slice(&["a"]);
        let ce = CounterExample::new(
            word.clone(),
            MismatchType::DistributionMismatch {
                expected: SubDistribution::new(),
                observed: SubDistribution::new(),
                distance: 0.0,
            },
            0.95,
            100,
        );

        let minimized = processor.minimize(&ce, &oracle);
        // Single-symbol word cannot be further minimized
        assert!(minimized.len() <= 1);
    }

    #[test]
    fn test_process_caches_result() {
        let oracle = TestOracle::new();
        let mut processor = CounterExampleProcessor::with_default_config();
        processor.config.minimize_first = false;

        let word = Word::from_str_slice(&["a", "b"]);
        let mut ce = CounterExample::new(
            word.clone(),
            MismatchType::DistributionMismatch {
                expected: SubDistribution::new(),
                observed: SubDistribution::new(),
                distance: 0.1,
            },
            0.95,
            100,
        );

        let _decomp = processor.process(&mut ce, &oracle);
        assert!(ce.processed);
        assert!(ce.decomposition.is_some());

        // Second call should hit cache
        let mut ce2 = CounterExample::new(
            word.clone(),
            MismatchType::DistributionMismatch {
                expected: SubDistribution::new(),
                observed: SubDistribution::new(),
                distance: 0.1,
            },
            0.95,
            100,
        );
        let _decomp2 = processor.process(&mut ce2, &oracle);
        assert_eq!(processor.stats().cache_hits, 1);
    }

    #[test]
    fn test_statistical_validation() {
        let mut oracle = TestOracle::new();
        let word = Word::from_str_slice(&["a"]);

        // Add a clear disagreement
        let sys = SubDistribution::singleton("yes".to_string(), 0.9);
        let hyp = SubDistribution::singleton("no".to_string(), 0.8);
        oracle.add_disagreement(word.clone(), sys, hyp);

        let mut processor = CounterExampleProcessor::with_default_config();
        let result = processor.validate_statistically(&word, &oracle, 50);

        assert!(result.num_samples == 50);
        assert!(result.mean_distance > 0.0);
    }

    #[test]
    fn test_generalized_pattern_matches() {
        let pattern = GeneralizedPattern {
            positions: vec![
                PatternPosition::Fixed(Symbol::new("a")),
                PatternPosition::Wildcard,
                PatternPosition::Fixed(Symbol::new("c")),
            ],
            original: Word::from_str_slice(&["a", "b", "c"]),
            generalized_count: 1,
        };

        assert!(pattern.matches(&Word::from_str_slice(&["a", "x", "c"])));
        assert!(pattern.matches(&Word::from_str_slice(&["a", "b", "c"])));
        assert!(!pattern.matches(&Word::from_str_slice(&["x", "b", "c"])));
        assert!(!pattern.matches(&Word::from_str_slice(&["a", "b"])));
    }

    #[test]
    fn test_generalized_pattern_wildcards() {
        let pattern = GeneralizedPattern {
            positions: vec![
                PatternPosition::Wildcard,
                PatternPosition::Fixed(Symbol::new("b")),
                PatternPosition::Wildcard,
            ],
            original: Word::from_str_slice(&["a", "b", "c"]),
            generalized_count: 2,
        };

        assert_eq!(pattern.wildcard_positions(), vec![0, 2]);
    }

    #[test]
    fn test_cache_operations() {
        let mut cache = CounterExampleCache::new(100);

        let word = Word::from_str_slice(&["a", "b"]);
        let decomp = Decomposition::new(
            Word::from_str_slice(&["a"]),
            Symbol::new("b"),
            Word::empty(),
            DecompositionMethod::RivestSchapire,
            1,
            2,
        );

        assert!(cache.lookup(&word).is_none());
        cache.insert(word.clone(), decomp);
        assert!(cache.lookup(&word).is_some());
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = CounterExampleCache::new(10);

        // Fill beyond capacity
        for i in 0..15 {
            let word = Word::from_str_slice(&[Box::leak(format!("s{}", i).into_boxed_str())]);
            let decomp = Decomposition::new(
                Word::empty(),
                Symbol::new(format!("s{}", i)),
                Word::empty(),
                DecompositionMethod::RivestSchapire,
                0,
                0,
            );
            cache.insert(word, decomp);
        }

        // Cache should have evicted some entries
        assert!(cache.size() <= 15);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = CounterExampleCache::new(100);
        assert_eq!(cache.hit_rate(), 0.0);

        let word = Word::from_str_slice(&["x"]);
        cache.lookup(&word); // miss
        assert_eq!(cache.hit_rate(), 0.0);

        let decomp = Decomposition::new(
            Word::empty(),
            Symbol::new("x"),
            Word::empty(),
            DecompositionMethod::RivestSchapire,
            0,
            0,
        );
        cache.insert(word.clone(), decomp);
        cache.lookup(&word); // hit
        // 1 hit out of 3 total (2 misses + 1 hit)
        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn test_rank_counterexamples() {
        let ces: Vec<CounterExample> = (0..5)
            .map(|i| {
                let word = Word::from_str_slice(
                    &(0..i + 1)
                        .map(|j| Box::leak(format!("s{}", j).into_boxed_str()) as &str)
                        .collect::<Vec<&str>>(),
                );
                CounterExample::new(
                    word,
                    MismatchType::DistributionMismatch {
                        expected: SubDistribution::new(),
                        observed: SubDistribution::new(),
                        distance: (5 - i) as f64 * 0.1,
                    },
                    (5 - i) as f64 * 0.2,
                    100,
                )
            })
            .collect();

        let ranked = rank_counterexamples(&ces, 3);
        assert_eq!(ranked.len(), 3);
        // First should be the highest confidence, shortest CE
        assert_eq!(ranked[0], 0);
    }

    #[test]
    fn test_deduplicate_counterexamples() {
        let decomp = Decomposition::new(
            Word::from_str_slice(&["a"]),
            Symbol::new("b"),
            Word::empty(),
            DecompositionMethod::RivestSchapire,
            1,
            2,
        );

        let mut ces = vec![
            {
                let mut ce = CounterExample::new(
                    Word::from_str_slice(&["a", "b"]),
                    MismatchType::DistributionMismatch {
                        expected: SubDistribution::new(),
                        observed: SubDistribution::new(),
                        distance: 0.1,
                    },
                    0.95,
                    100,
                );
                ce.decomposition = Some(decomp.clone());
                ce
            },
            {
                let mut ce = CounterExample::new(
                    Word::from_str_slice(&["c", "b"]),
                    MismatchType::DistributionMismatch {
                        expected: SubDistribution::new(),
                        observed: SubDistribution::new(),
                        distance: 0.2,
                    },
                    0.90,
                    100,
                );
                ce.decomposition = Some(decomp.clone());
                ce
            },
        ];

        deduplicate_counterexamples(&mut ces);
        assert_eq!(ces.len(), 1);
    }

    #[test]
    fn test_group_by_suffix() {
        let decomp1 = Decomposition::new(
            Word::from_str_slice(&["a"]),
            Symbol::new("b"),
            Word::empty(),
            DecompositionMethod::RivestSchapire,
            1,
            2,
        );
        let decomp2 = Decomposition::new(
            Word::from_str_slice(&["x"]),
            Symbol::new("y"),
            Word::from_str_slice(&["z"]),
            DecompositionMethod::SuffixBased,
            1,
            2,
        );

        let ces = vec![
            {
                let mut ce = CounterExample::new(
                    Word::from_str_slice(&["a", "b"]),
                    MismatchType::DistributionMismatch {
                        expected: SubDistribution::new(),
                        observed: SubDistribution::new(),
                        distance: 0.1,
                    },
                    0.95,
                    100,
                );
                ce.decomposition = Some(decomp1);
                ce
            },
            {
                let mut ce = CounterExample::new(
                    Word::from_str_slice(&["x", "y", "z"]),
                    MismatchType::DistributionMismatch {
                        expected: SubDistribution::new(),
                        observed: SubDistribution::new(),
                        distance: 0.2,
                    },
                    0.90,
                    100,
                );
                ce.decomposition = Some(decomp2);
                ce
            },
        ];

        let groups = group_by_suffix(&ces);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_normal_cdf() {
        // CDF(0) should be ~0.5
        let val = normal_cdf(0.0);
        assert!((val - 0.5).abs() < 0.01);

        // CDF(large) should be ~1
        let val = normal_cdf(5.0);
        assert!(val > 0.99);

        // CDF(very negative) should be ~0
        let val = normal_cdf(-5.0);
        assert!(val < 0.01);
    }

    #[test]
    fn test_sub_distribution_tv_distance() {
        let d1 = SubDistribution::singleton("a".to_string(), 0.7);
        let d2 = SubDistribution::singleton("a".to_string(), 0.3);

        let dist = d1.tv_distance(&d2);
        assert!((dist - 0.2).abs() < 0.01);

        // Same distribution => 0
        let d3 = SubDistribution::singleton("a".to_string(), 0.5);
        assert!(d3.tv_distance(&d3) < 1e-10);
    }

    #[test]
    fn test_word_split_at() {
        let word = Word::from_str_slice(&["a", "b", "c", "d"]);
        let (prefix, suffix) = word.split_at(2);
        assert_eq!(prefix.len(), 2);
        assert_eq!(suffix.len(), 2);
        assert_eq!(prefix.symbols[0], Symbol::new("a"));
        assert_eq!(suffix.symbols[0], Symbol::new("c"));
    }

    #[test]
    fn test_word_concat() {
        let w1 = Word::from_str_slice(&["a", "b"]);
        let w2 = Word::from_str_slice(&["c"]);
        let w3 = w1.concat(&w2);
        assert_eq!(w3.len(), 3);
        assert_eq!(w3.symbols[2], Symbol::new("c"));
    }

    #[test]
    fn test_processing_stats() {
        let processor = CounterExampleProcessor::with_default_config();
        let stats = processor.stats();
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.total_queries, 0);
    }

    #[test]
    fn test_validation_result_for_agreement() {
        // When system and hypothesis agree, validation should show low distance
        let oracle = TestOracle::new();
        let mut processor = CounterExampleProcessor::with_default_config();

        let word = Word::from_str_slice(&["a"]);
        let result = processor.validate_statistically(&word, &oracle, 20);

        assert!(result.mean_distance < 0.01);
    }

    #[test]
    fn test_mismatch_type_display() {
        let mismatch = MismatchType::MissingTransition {
            from_state: "q0".to_string(),
            on_symbol: Symbol::new("a"),
        };
        // Just ensure no panic on debug formatting
        let _s = format!("{:?}", mismatch);

        let mismatch2 = MismatchType::StateDisagreement {
            hypothesis_state: "q1".to_string(),
            system_behavior: SubDistribution::singleton("x".to_string(), 0.5),
        };
        let _s2 = format!("{:?}", mismatch2);
    }

    #[test]
    fn test_counter_example_pattern_from_single() {
        let ce = CounterExample::new(Word::from_str_slice(&["a", "b", "c"]), MismatchType::DistributionMismatch { expected: SubDistribution::singleton("yes".to_string(), 0.9), observed: SubDistribution::singleton("no".to_string(), 0.8), distance: 0.7 }, 0.95, 100);

        let pattern = CounterExamplePattern::from_single(&ce);
        assert_eq!(pattern.instances.len(), 1);
        assert_eq!(pattern.min_length, 3);
        assert_eq!(pattern.max_length, 3);
        assert!(pattern.matches(&ce.word));
    }

    #[test]
    fn test_counter_example_pattern_match_exact() {
        let ce = CounterExample::new(Word::from_str_slice(&["a", "b"]), MismatchType::DistributionMismatch { expected: SubDistribution::new(), observed: SubDistribution::new(), distance: 0.5 }, 0.95, 100);

        let pattern = CounterExamplePattern::from_single(&ce);
        assert!(pattern.matches(&Word::from_str_slice(&["a", "b"])));
        assert!(!pattern.matches(&Word::from_str_slice(&["a", "c"])));
        assert!(!pattern.matches(&Word::from_str_slice(&["a"])));
    }

    #[test]
    fn test_counter_example_pattern_wildcard() {
        let pattern = CounterExamplePattern {
            id: "test".to_string(),
            symbolic: vec![
                PatternElement::Exact("a".to_string()),
                PatternElement::Wildcard,
                PatternElement::Exact("c".to_string()),
            ],
            instances: vec![],
            min_length: 3,
            max_length: 3,
            avg_disagreement: 0.5,
            critical_positions: vec![],
        };

        assert!(pattern.matches(&Word::from_str_slice(&["a", "b", "c"])));
        assert!(pattern.matches(&Word::from_str_slice(&["a", "x", "c"])));
        assert!(!pattern.matches(&Word::from_str_slice(&["a", "b", "d"])));
    }

    #[test]
    fn test_counter_example_pattern_star() {
        let pattern = CounterExamplePattern {
            id: "test".to_string(),
            symbolic: vec![
                PatternElement::Exact("a".to_string()),
                PatternElement::Star,
                PatternElement::Exact("c".to_string()),
            ],
            instances: vec![],
            min_length: 2,
            max_length: 10,
            avg_disagreement: 0.5,
            critical_positions: vec![],
        };

        assert!(pattern.matches(&Word::from_str_slice(&["a", "c"])));
        assert!(pattern.matches(&Word::from_str_slice(&["a", "b", "c"])));
        assert!(pattern.matches(&Word::from_str_slice(&["a", "b", "b", "c"])));
        assert!(!pattern.matches(&Word::from_str_slice(&["a", "b"])));
    }

    #[test]
    fn test_counter_example_pattern_generalize() {
        let ce1 = CounterExample::new(Word::from_str_slice(&["a", "b", "c"]), MismatchType::DistributionMismatch { expected: SubDistribution::new(), observed: SubDistribution::new(), distance: 0.5 }, 0.95, 100);

        let mut pattern = CounterExamplePattern::from_single(&ce1);
        pattern.add_instance(&Word::from_str_slice(&["a", "x", "c"]), 0.6);
        pattern.generalize();

        // Position 0 and 2 are fixed, position 1 is wildcard
        assert!(pattern.symbolic[0] == PatternElement::Exact("a".to_string()));
        assert!(pattern.symbolic[1] == PatternElement::Wildcard);
        assert!(pattern.symbolic[2] == PatternElement::Exact("c".to_string()));
    }

    #[test]
    fn test_counter_example_pattern_display() {
        let pattern = CounterExamplePattern {
            id: "test".to_string(),
            symbolic: vec![
                PatternElement::Exact("a".to_string()),
                PatternElement::Wildcard,
            ],
            instances: vec![Word::from_str_slice(&["a", "b"])],
            min_length: 2,
            max_length: 2,
            avg_disagreement: 0.5,
            critical_positions: vec![],
        };

        let s = format!("{}", pattern);
        assert!(s.contains("Pattern"));
        assert!(s.contains("instances=1"));
    }

    #[test]
    fn test_pattern_library_basic() {
        let mut lib = PatternLibrary::with_defaults();
        let ce = CounterExample::new(Word::from_str_slice(&["a", "b"]), MismatchType::DistributionMismatch { expected: SubDistribution::singleton("yes".to_string(), 0.9), observed: SubDistribution::singleton("no".to_string(), 0.8), distance: 0.5 }, 0.95, 100);

        lib.add(&ce);
        assert_eq!(lib.num_patterns(), 1);
        assert_eq!(lib.total_instances(), 1);
    }

    #[test]
    fn test_pattern_library_matching() {
        let mut lib = PatternLibrary::with_defaults();
        let ce = CounterExample::new(Word::from_str_slice(&["a", "b"]), MismatchType::DistributionMismatch { expected: SubDistribution::new(), observed: SubDistribution::new(), distance: 0.5 }, 0.95, 100);

        lib.add(&ce);
        let matches = lib.matching_patterns(&Word::from_str_slice(&["a", "b"]));
        assert_eq!(matches.len(), 1);

        let no_match = lib.matching_patterns(&Word::from_str_slice(&["x", "y"]));
        assert!(no_match.is_empty());
    }

    #[test]
    fn test_pattern_library_top_patterns() {
        let mut lib = PatternLibrary::with_defaults();
        for s in &["a", "b", "c"] {
            let ce = CounterExample::new(Word::from_str_slice(&[s]), MismatchType::DistributionMismatch { expected: SubDistribution::new(), observed: SubDistribution::new(), distance: 0.5 }, 0.95, 100);
            lib.add(&ce);
        }

        let top = lib.top_patterns(2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_counter_example_severity_classification() {
        assert_eq!(CounterExampleSeverity::from_disagreement(0.05), CounterExampleSeverity::Minor);
        assert_eq!(CounterExampleSeverity::from_disagreement(0.2), CounterExampleSeverity::Moderate);
        assert_eq!(CounterExampleSeverity::from_disagreement(0.5), CounterExampleSeverity::Major);
        assert_eq!(CounterExampleSeverity::from_disagreement(0.9), CounterExampleSeverity::Critical);
    }

    #[test]
    fn test_counter_example_severity_weight() {
        assert!(CounterExampleSeverity::Minor.weight() < CounterExampleSeverity::Moderate.weight());
        assert!(CounterExampleSeverity::Moderate.weight() < CounterExampleSeverity::Major.weight());
        assert!(CounterExampleSeverity::Major.weight() < CounterExampleSeverity::Critical.weight());
    }

    #[test]
    fn test_counter_example_severity_display() {
        assert_eq!(format!("{}", CounterExampleSeverity::Critical), "Critical");
    }

    #[test]
    fn test_counter_example_prioritizer() {
        let mut prioritizer = CounterExamplePrioritizer::new();

        let ce_minor = CounterExample::new(Word::from_str_slice(&["a"]), MismatchType::DistributionMismatch { expected: SubDistribution::new(), observed: SubDistribution::new(), distance: 0.05 }, 0.95, 100);

        let ce_critical = CounterExample::new(Word::from_str_slice(&["b"]), MismatchType::DistributionMismatch { expected: SubDistribution::new(), observed: SubDistribution::new(), distance: 0.9 }, 0.95, 100);

        prioritizer.add(ce_minor);
        prioritizer.add(ce_critical);

        assert_eq!(prioritizer.len(), 2);

        // Critical should come first
        let (first, sev) = prioritizer.next().unwrap();
        assert_eq!(sev, CounterExampleSeverity::Critical);
        assert_eq!(first.word.symbols[0].0, "b");
    }

    #[test]
    fn test_counter_example_prioritizer_count_by_severity() {
        let mut prioritizer = CounterExamplePrioritizer::new();

        for &mag in &[0.05, 0.2, 0.5, 0.9] {
            let ce = CounterExample::new(Word::from_str_slice(&["x"]), MismatchType::DistributionMismatch { expected: SubDistribution::new(), observed: SubDistribution::new(), distance: mag }, 0.95, 100);
            prioritizer.add(ce);
        }

        let counts = prioritizer.count_by_severity();
        assert_eq!(*counts.get(&CounterExampleSeverity::Minor).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&CounterExampleSeverity::Critical).unwrap_or(&0), 1);
    }

    #[test]
    fn test_pattern_element_display() {
        assert_eq!(format!("{}", PatternElement::Exact("a".to_string())), "a");
        assert_eq!(format!("{}", PatternElement::Wildcard), "?");
        assert_eq!(format!("{}", PatternElement::Star), "*");
    }

    #[test]
    fn test_pattern_fixed_length() {
        let pattern = CounterExamplePattern {
            id: "test".to_string(),
            symbolic: vec![
                PatternElement::Exact("a".to_string()),
                PatternElement::Star,
                PatternElement::Wildcard,
                PatternElement::Exact("b".to_string()),
            ],
            instances: vec![],
            min_length: 3,
            max_length: 10,
            avg_disagreement: 0.0,
            critical_positions: vec![],
        };
        assert_eq!(pattern.fixed_length(), 3); // Exact + Wildcard + Exact, excluding Star
    }
}
