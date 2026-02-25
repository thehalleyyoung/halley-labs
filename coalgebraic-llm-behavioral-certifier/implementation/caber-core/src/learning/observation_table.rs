//! Angluin-style observation table generalized to coalgebras.
//!
//! The observation table is the central data structure of the L* algorithm.
//! Rows are indexed by state-access strings (prefixes), columns by suffixes.
//! Table entries are sub-distributions rather than single Boolean values,
//! enabling learning of probabilistic and weighted automata.

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

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
    pub fn last_symbol(&self) -> Option<&Symbol> { self.symbols.last() }
    pub fn without_last(&self) -> Word {
        if self.symbols.is_empty() { return Word::empty(); }
        Word { symbols: self.symbols[..self.symbols.len() - 1].to_vec() }
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
        let mut w = HashMap::new(); w.insert(key, prob); Self { weights: w }
    }
    pub fn from_map(weights: HashMap<String, f64>) -> Self { Self { weights } }
    pub fn total_mass(&self) -> f64 { self.weights.values().sum() }
    pub fn get(&self, key: &str) -> f64 { self.weights.get(key).copied().unwrap_or(0.0) }
    pub fn set(&mut self, key: String, prob: f64) {
        if prob > 1e-15 { self.weights.insert(key, prob); }
        else { self.weights.remove(&key); }
    }
    pub fn support_size(&self) -> usize { self.weights.len() }
    pub fn is_empty_dist(&self) -> bool { self.weights.is_empty() || self.total_mass() < 1e-15 }

    pub fn tv_distance(&self, other: &SubDistribution) -> f64 {
        let mut all_keys: HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());
        let mut dist = 0.0;
        for key in all_keys { dist += (self.get(key) - other.get(key)).abs(); }
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

    pub fn merge_weighted(&self, other: &SubDistribution, alpha: f64) -> SubDistribution {
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
// Table entry
// ---------------------------------------------------------------------------

/// A single entry in the observation table.
///
/// Contains the estimated sub-distribution, sample count, and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEntry {
    /// Estimated output sub-distribution
    pub distribution: SubDistribution,
    /// Number of samples used to estimate
    pub sample_count: usize,
    /// Confidence in the estimate (based on sample size)
    pub confidence: f64,
    /// Whether this entry has been filled
    pub filled: bool,
    /// Timestamp of last update
    pub last_updated: u64,
}

impl TableEntry {
    pub fn new() -> Self {
        Self {
            distribution: SubDistribution::new(),
            sample_count: 0,
            confidence: 0.0,
            filled: false,
            last_updated: 0,
        }
    }

    pub fn from_distribution(dist: SubDistribution, samples: usize) -> Self {
        let confidence = if samples > 0 {
            1.0 - 1.0 / (samples as f64).sqrt()
        } else {
            0.0
        };

        Self {
            distribution: dist,
            sample_count: samples,
            confidence,
            filled: true,
            last_updated: timestamp_now(),
        }
    }

    /// Update with new samples (running average).
    pub fn update(&mut self, new_dist: &SubDistribution, new_samples: usize) {
        if self.sample_count == 0 {
            self.distribution = new_dist.clone();
            self.sample_count = new_samples;
        } else {
            let total = self.sample_count + new_samples;
            let alpha = self.sample_count as f64 / total as f64;
            self.distribution = self.distribution.merge_weighted(new_dist, alpha);
            self.sample_count = total;
        }

        self.confidence = 1.0 - 1.0 / (self.sample_count as f64).sqrt().max(1.0);
        self.filled = true;
        self.last_updated = timestamp_now();
    }

    /// Standard error of the distribution estimate.
    pub fn standard_error(&self) -> f64 {
        if self.sample_count < 2 {
            return f64::INFINITY;
        }
        // Approximate: SE ≈ 1 / sqrt(2n) for distribution estimation
        1.0 / (2.0 * self.sample_count as f64).sqrt()
    }
}

impl Default for TableEntry {
    fn default() -> Self { Self::new() }
}

fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Closedness and Consistency results
// ---------------------------------------------------------------------------

/// Result of checking closedness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClosednessResult {
    /// Table is closed
    Closed,
    /// Table is not closed; contains the rows that need to be promoted
    NotClosed {
        /// Extended rows that have no matching upper row
        unclosed_rows: Vec<Word>,
    },
}

impl ClosednessResult {
    pub fn is_closed(&self) -> bool {
        matches!(self, ClosednessResult::Closed)
    }
}

/// Result of checking consistency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyResult {
    /// Table is consistent
    Consistent,
    /// Table is not consistent; contains the distinguishing evidence
    NotConsistent {
        /// The two rows that are equivalent but have inconsistent extensions
        row1: Word,
        row2: Word,
        /// The symbol on which they disagree
        symbol: Symbol,
        /// The suffix that distinguishes them
        distinguishing_suffix: Word,
    },
}

impl ConsistencyResult {
    pub fn is_consistent(&self) -> bool {
        matches!(self, ConsistencyResult::Consistent)
    }
}

// ---------------------------------------------------------------------------
// ObservationTable
// ---------------------------------------------------------------------------

/// Configuration for the observation table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationTableConfig {
    /// Statistical tolerance for row equivalence
    pub tolerance: f64,
    /// Minimum samples per entry for confidence
    pub min_samples: usize,
    /// Maximum number of rows
    pub max_rows: usize,
    /// Maximum number of columns
    pub max_columns: usize,
    /// Whether to use hypothesis testing for equivalence
    pub use_hypothesis_test: bool,
    /// Significance level for hypothesis tests
    pub significance_level: f64,
}

impl Default for ObservationTableConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.05,
            min_samples: 30,
            max_rows: 10000,
            max_columns: 1000,
            use_hypothesis_test: true,
            significance_level: 0.05,
        }
    }
}

/// The observation table generalized to coalgebras.
///
/// Rows are indexed by prefixes (access strings), split into:
///   - S: upper rows (states in the hypothesis)
///   - SA: lower rows (one-symbol extensions of S)
///
/// Columns are indexed by suffixes E.
///
/// Entries T(s, e) are sub-distributions estimated from samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationTable<S: Clone + Eq + std::hash::Hash + Ord + Serialize> {
    /// Upper rows: state-access strings (S)
    pub upper_rows: Vec<S>,
    /// Lower rows: extensions of upper rows (S·A)
    pub lower_rows: Vec<S>,
    /// Columns: suffixes (E)
    pub columns: Vec<S>,
    /// Table entries indexed by (row, column)
    pub entries: HashMap<(S, S), TableEntry>,
    /// Configuration
    pub config: ObservationTableConfig,
    /// Statistics
    pub stats: TableStats,
}

/// Statistics about the observation table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TableStats {
    pub total_entries: usize,
    pub filled_entries: usize,
    pub total_samples: usize,
    pub closedness_checks: usize,
    pub consistency_checks: usize,
    pub rows_promoted: usize,
    pub columns_added: usize,
    pub compactions_performed: usize,
}

// Implement for Word specifically
impl ObservationTable<Word> {
    /// Create a new observation table with empty word as initial row and column.
    pub fn new(config: ObservationTableConfig) -> Self {
        let upper_rows = vec![Word::empty()];
        let columns = vec![Word::empty()];

        Self {
            upper_rows,
            lower_rows: Vec::new(),
            columns,
            entries: HashMap::new(),
            config,
            stats: TableStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ObservationTableConfig::default())
    }

    /// Initialize the table with the given alphabet.
    pub fn initialize(&mut self, alphabet: &[Symbol]) {
        // Add one-symbol extensions as lower rows
        for sym in alphabet {
            let word = Word::singleton(sym.clone());
            if !self.lower_rows.contains(&word) && !self.upper_rows.contains(&word) {
                self.lower_rows.push(word);
            }
        }
    }

    /// Get the number of upper rows (states).
    pub fn num_states(&self) -> usize {
        self.upper_rows.len()
    }

    /// Get the number of columns (suffixes).
    pub fn num_suffixes(&self) -> usize {
        self.columns.len()
    }

    /// Get total number of rows (upper + lower).
    pub fn num_rows(&self) -> usize {
        self.upper_rows.len() + self.lower_rows.len()
    }

    /// Get a table entry.
    pub fn get_entry(&self, row: &Word, col: &Word) -> Option<&TableEntry> {
        self.entries.get(&(row.clone(), col.clone()))
    }

    /// Get the distribution for a cell.
    pub fn get_distribution(&self, row: &Word, col: &Word) -> SubDistribution {
        self.entries
            .get(&(row.clone(), col.clone()))
            .map(|e| e.distribution.clone())
            .unwrap_or_default()
    }

    /// Set a table entry.
    pub fn set_entry(&mut self, row: Word, col: Word, entry: TableEntry) {
        if entry.filled {
            self.stats.filled_entries += 1;
            self.stats.total_samples += entry.sample_count;
        }
        self.stats.total_entries += 1;
        self.entries.insert((row, col), entry);
    }

    /// Set entry from distribution and samples.
    pub fn set_distribution(&mut self, row: Word, col: Word, dist: SubDistribution, samples: usize) {
        let entry = TableEntry::from_distribution(dist, samples);
        self.set_entry(row, col, entry);
    }

    /// Update an existing entry with new samples.
    pub fn update_entry(&mut self, row: &Word, col: &Word, dist: &SubDistribution, samples: usize) {
        let key = (row.clone(), col.clone());
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.update(dist, samples);
            self.stats.total_samples += samples;
        } else {
            self.set_distribution(row.clone(), col.clone(), dist.clone(), samples);
        }
    }

    /// Get the row signature for a prefix (all column values).
    pub fn row_signature(&self, row: &Word) -> Vec<SubDistribution> {
        self.columns.iter()
            .map(|col| self.get_distribution(row, col))
            .collect()
    }

    /// Check if two rows are equivalent within tolerance.
    pub fn rows_equivalent(&self, row1: &Word, row2: &Word) -> bool {
        let sig1 = self.row_signature(row1);
        let sig2 = self.row_signature(row2);

        if sig1.len() != sig2.len() {
            return false;
        }

        if self.config.use_hypothesis_test {
            self.rows_equivalent_hypothesis_test(&sig1, &sig2, row1, row2)
        } else {
            self.rows_equivalent_tolerance(&sig1, &sig2)
        }
    }

    /// Simple tolerance-based row equivalence.
    fn rows_equivalent_tolerance(
        &self,
        sig1: &[SubDistribution],
        sig2: &[SubDistribution],
    ) -> bool {
        for (d1, d2) in sig1.iter().zip(sig2.iter()) {
            if d1.tv_distance(d2) > self.config.tolerance {
                return false;
            }
        }
        true
    }

    /// Hypothesis-test-based row equivalence.
    ///
    /// Uses a two-sample KS-like test: rows are equivalent if the
    /// maximum entry-wise distance is below a threshold derived
    /// from the sample sizes.
    fn rows_equivalent_hypothesis_test(
        &self,
        sig1: &[SubDistribution],
        sig2: &[SubDistribution],
        row1: &Word,
        row2: &Word,
    ) -> bool {
        for (i, (d1, d2)) in sig1.iter().zip(sig2.iter()).enumerate() {
            let col = &self.columns[i];

            let n1 = self.entries.get(&(row1.clone(), col.clone()))
                .map(|e| e.sample_count)
                .unwrap_or(0);
            let n2 = self.entries.get(&(row2.clone(), col.clone()))
                .map(|e| e.sample_count)
                .unwrap_or(0);

            if n1 == 0 || n2 == 0 {
                continue;
            }

            let tv_dist = d1.tv_distance(d2);

            // Critical value for two-sample test:
            // D_crit = c(α) * sqrt((n1+n2)/(n1*n2))
            // where c(0.05) ≈ 1.36
            let c_alpha = match OrderedFloat(self.config.significance_level) {
                x if x <= OrderedFloat(0.01) => 1.63,
                x if x <= OrderedFloat(0.05) => 1.36,
                x if x <= OrderedFloat(0.10) => 1.22,
                _ => 1.07,
            };

            let critical_value = c_alpha * ((n1 + n2) as f64 / (n1 * n2) as f64).sqrt();

            if tv_dist > critical_value.max(self.config.tolerance) {
                return false;
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Closedness checking
    // -----------------------------------------------------------------------

    /// Check if the table is closed.
    ///
    /// The table is closed if every lower row has an equivalent upper row.
    pub fn check_closedness(&mut self) -> ClosednessResult {
        self.stats.closedness_checks += 1;

        let mut unclosed = Vec::new();

        for lower in &self.lower_rows {
            let mut has_match = false;
            for upper in &self.upper_rows {
                if self.rows_equivalent(lower, upper) {
                    has_match = true;
                    break;
                }
            }
            if !has_match {
                unclosed.push(lower.clone());
            }
        }

        if unclosed.is_empty() {
            ClosednessResult::Closed
        } else {
            ClosednessResult::NotClosed {
                unclosed_rows: unclosed,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Consistency checking
    // -----------------------------------------------------------------------

    /// Check if the table is consistent.
    ///
    /// The table is consistent if for all s1, s2 in S:
    ///   row(s1) ≈ row(s2) ⟹ row(s1·a) ≈ row(s2·a) for all a ∈ Σ
    pub fn check_consistency(&mut self, alphabet: &[Symbol]) -> ConsistencyResult {
        self.stats.consistency_checks += 1;

        for i in 0..self.upper_rows.len() {
            for j in (i + 1)..self.upper_rows.len() {
                let s1 = &self.upper_rows[i];
                let s2 = &self.upper_rows[j];

                if !self.rows_equivalent(s1, s2) {
                    continue;
                }

                // Check extensions
                for sym in alphabet {
                    let ext1 = s1.concat(&Word::singleton(sym.clone()));
                    let ext2 = s2.concat(&Word::singleton(sym.clone()));

                    let sig1 = self.row_signature(&ext1);
                    let sig2 = self.row_signature(&ext2);

                    // Find distinguishing suffix
                    for (k, (d1, d2)) in sig1.iter().zip(sig2.iter()).enumerate() {
                        let dist = d1.tv_distance(d2);
                        if dist > self.config.tolerance {
                            let new_suffix = Word::singleton(sym.clone())
                                .concat(&self.columns[k]);

                            return ConsistencyResult::NotConsistent {
                                row1: s1.clone(),
                                row2: s2.clone(),
                                symbol: sym.clone(),
                                distinguishing_suffix: new_suffix,
                            };
                        }
                    }
                }
            }
        }

        ConsistencyResult::Consistent
    }

    // -----------------------------------------------------------------------
    // Table extension operations
    // -----------------------------------------------------------------------

    /// Add a new upper row (promote from lower or add new).
    pub fn add_upper_row(&mut self, word: Word, alphabet: &[Symbol]) {
        if self.upper_rows.contains(&word) {
            return;
        }
        if self.upper_rows.len() >= self.config.max_rows {
            return;
        }

        // Remove from lower rows if present
        self.lower_rows.retain(|r| r != &word);

        self.upper_rows.push(word.clone());
        self.stats.rows_promoted += 1;

        // Add extensions to lower rows
        for sym in alphabet {
            let ext = word.concat(&Word::singleton(sym.clone()));
            if !self.upper_rows.contains(&ext) && !self.lower_rows.contains(&ext) {
                self.lower_rows.push(ext);
            }
        }
    }

    /// Add a new column (suffix).
    pub fn add_column(&mut self, suffix: Word) -> bool {
        if self.columns.contains(&suffix) {
            return false;
        }
        if self.columns.len() >= self.config.max_columns {
            return false;
        }

        self.columns.push(suffix);
        self.stats.columns_added += 1;
        true
    }

    /// Get all cells that need to be filled.
    pub fn unfilled_cells(&self) -> Vec<(Word, Word)> {
        let mut cells = Vec::new();

        let all_rows: Vec<&Word> = self.upper_rows.iter()
            .chain(self.lower_rows.iter())
            .collect();

        for row in all_rows {
            for col in &self.columns {
                let key = (row.clone(), col.clone());
                if !self.entries.contains_key(&key) || !self.entries[&key].filled {
                    cells.push((row.clone(), col.clone()));
                }
            }
        }

        cells
    }

    /// Get cells that have insufficient samples.
    pub fn low_confidence_cells(&self) -> Vec<(Word, Word, usize)> {
        let mut cells = Vec::new();

        let all_rows: Vec<&Word> = self.upper_rows.iter()
            .chain(self.lower_rows.iter())
            .collect();

        for row in all_rows {
            for col in &self.columns {
                let key = (row.clone(), col.clone());
                if let Some(entry) = self.entries.get(&key) {
                    if entry.sample_count < self.config.min_samples {
                        let needed = self.config.min_samples - entry.sample_count;
                        cells.push((row.clone(), col.clone(), needed));
                    }
                }
            }
        }

        cells
    }

    // -----------------------------------------------------------------------
    // Counter-example processing
    // -----------------------------------------------------------------------

    /// Process a counter-example using Rivest-Schapire decomposition.
    ///
    /// Given a counter-example word w, find a decomposition w = u·a·v
    /// such that the suffix a·v is added as a new column to the table.
    pub fn process_counterexample(
        &mut self,
        ce_word: &Word,
        alphabet: &[Symbol],
        query_fn: &dyn Fn(&Word) -> SubDistribution,
    ) -> Word {
        let n = ce_word.len();
        if n == 0 {
            return Word::empty();
        }
        if n == 1 {
            let suffix = ce_word.clone();
            self.add_column(suffix.clone());
            return suffix;
        }

        // Binary search for the break point
        let mut lo = 0usize;
        let mut hi = n - 1;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;

            // Build access string by running hypothesis on prefix
            let prefix = ce_word.prefix(mid + 1);
            let suffix = ce_word.suffix_from(mid + 1);

            // Query the system and hypothesis for the combined word
            let test_word = prefix.concat(&suffix);
            let sys_dist = query_fn(&test_word);

            // Find the hypothesis row matching the prefix
            let hyp_dist = self.hypothesis_output_for_word(&prefix, &suffix);

            let dist = sys_dist.tv_distance(&hyp_dist);

            if dist > self.config.tolerance {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        // New suffix: symbol at lo concatenated with remaining suffix
        let new_suffix = ce_word.suffix_from(lo);

        self.add_column(new_suffix.clone());

        // Also add prefixes of the CE as new rows if needed
        for i in 0..=n {
            let prefix = ce_word.prefix(i);
            if !self.upper_rows.contains(&prefix) && !self.lower_rows.contains(&prefix) {
                self.lower_rows.push(prefix);
            }
        }

        new_suffix
    }

    /// Get hypothesis output by looking up the observation table.
    fn hypothesis_output_for_word(&self, prefix: &Word, suffix: &Word) -> SubDistribution {
        // Find the upper row that matches this prefix
        let mut best_match = &self.upper_rows[0];
        let mut best_dist = f64::INFINITY;

        let prefix_sig = self.row_signature(prefix);

        for upper in &self.upper_rows {
            let upper_sig = self.row_signature(upper);
            let dist = signature_distance(&prefix_sig, &upper_sig);
            if dist < best_dist {
                best_dist = dist;
                best_match = upper;
            }
        }

        // Get the entry for best_match concatenated with suffix
        let query_word = best_match.concat(suffix);
        self.get_distribution(&query_word, &Word::empty())
    }

    // -----------------------------------------------------------------------
    // Table compaction
    // -----------------------------------------------------------------------

    /// Compact the table by merging statistically equivalent rows.
    ///
    /// Finds pairs of upper rows that are equivalent and merges them,
    /// keeping the shorter access string.
    pub fn compact(&mut self) -> usize {
        let mut merged = 0;
        let mut to_remove: HashSet<Word> = HashSet::new();

        let n = self.upper_rows.len();
        for i in 0..n {
            if to_remove.contains(&self.upper_rows[i]) {
                continue;
            }
            for j in (i + 1)..n {
                if to_remove.contains(&self.upper_rows[j]) {
                    continue;
                }

                let r1 = &self.upper_rows[i];
                let r2 = &self.upper_rows[j];

                if self.rows_equivalent(r1, r2) {
                    // Keep the shorter one
                    let remove = if r1.len() <= r2.len() { r2 } else { r1 };
                    to_remove.insert(remove.clone());
                    merged += 1;
                }
            }
        }

        // Remove merged rows and move them to lower rows
        for word in &to_remove {
            self.upper_rows.retain(|r| r != word);
            if !self.lower_rows.contains(word) {
                self.lower_rows.push(word.clone());
            }
        }

        self.stats.compactions_performed += 1;
        merged
    }

    // -----------------------------------------------------------------------
    // Table export (for hypothesis construction)
    // -----------------------------------------------------------------------

    /// Export the table data for hypothesis construction.
    pub fn export_for_hypothesis(&self) -> (Vec<Word>, Vec<Word>, HashMap<Word, Vec<SubDistribution>>) {
        let access_strings = self.upper_rows.clone();
        let suffixes = self.columns.clone();

        let mut table = HashMap::new();

        let all_rows: Vec<&Word> = self.upper_rows.iter()
            .chain(self.lower_rows.iter())
            .collect();

        for row in all_rows {
            let sig = self.row_signature(row);
            table.insert(row.clone(), sig);
        }

        (access_strings, suffixes, table)
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serialize the table to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize the table from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Create a checkpoint of the table.
    pub fn checkpoint(&self) -> TableCheckpoint {
        TableCheckpoint {
            upper_row_count: self.upper_rows.len(),
            lower_row_count: self.lower_rows.len(),
            column_count: self.columns.len(),
            entry_count: self.entries.len(),
            filled_count: self.stats.filled_entries,
            total_samples: self.stats.total_samples,
            stats: self.stats.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Display and debugging
    // -----------------------------------------------------------------------

    /// Format the table as a human-readable string.
    pub fn display_table(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("ObservationTable:\n");
        output.push_str(&format!(
            "  {} upper rows, {} lower rows, {} columns\n",
            self.upper_rows.len(),
            self.lower_rows.len(),
            self.columns.len(),
        ));

        // Column headers
        output.push_str("  ");
        output.push_str(&format!("{:>15} |", "Row \\ Col"));
        for col in &self.columns {
            output.push_str(&format!(" {:>12} |", format!("{}", col)));
        }
        output.push('\n');
        output.push_str(&"-".repeat(17 + 15 * self.columns.len()));
        output.push('\n');

        // Upper rows
        output.push_str("  S:\n");
        for row in &self.upper_rows {
            output.push_str(&format!("  {:>15} |", format!("{}", row)));
            for col in &self.columns {
                let dist = self.get_distribution(row, col);
                let mass = dist.total_mass();
                output.push_str(&format!(" {:>12.4} |", mass));
            }
            output.push('\n');
        }

        // Lower rows
        output.push_str("  SA:\n");
        for row in self.lower_rows.iter().take(10) {
            output.push_str(&format!("  {:>15} |", format!("{}", row)));
            for col in &self.columns {
                let dist = self.get_distribution(row, col);
                let mass = dist.total_mass();
                output.push_str(&format!(" {:>12.4} |", mass));
            }
            output.push('\n');
        }
        if self.lower_rows.len() > 10 {
            output.push_str(&format!("  ... and {} more lower rows\n",
                self.lower_rows.len() - 10));
        }

        output
    }
}

/// Compute distance between two signatures.
fn signature_distance(sig1: &[SubDistribution], sig2: &[SubDistribution]) -> f64 {
    if sig1.len() != sig2.len() {
        return f64::INFINITY;
    }
    let mut max_dist = 0.0f64;
    for (a, b) in sig1.iter().zip(sig2.iter()) {
        max_dist = max_dist.max(a.tv_distance(b));
    }
    max_dist
}

/// Table checkpoint for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCheckpoint {
    pub upper_row_count: usize,
    pub lower_row_count: usize,
    pub column_count: usize,
    pub entry_count: usize,
    pub filled_count: usize,
    pub total_samples: usize,
    pub stats: TableStats,
}

impl fmt::Display for TableCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Checkpoint(S={}, SA={}, E={}, entries={}/{})",
            self.upper_row_count,
            self.lower_row_count,
            self.column_count,
            self.filled_count,
            self.entry_count,
        )
    }
}

// ---------------------------------------------------------------------------
// Advanced table operations: partitioning, stratified sampling, multi-table
// ---------------------------------------------------------------------------

/// Partition information for a row in the observation table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowPartition {
    /// Which partition (equivalence class) this row belongs to
    pub partition_id: usize,
    /// The representative row for this partition
    pub representative: Word,
    /// Distance to the representative
    pub distance_to_representative: f64,
}

/// Result of table partitioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TablePartitioning {
    /// Number of partitions found
    pub num_partitions: usize,
    /// Mapping from row to partition ID
    pub assignments: HashMap<String, usize>,
    /// Representatives for each partition
    pub representatives: Vec<Word>,
    /// Average intra-partition distance
    pub avg_intra_distance: f64,
    /// Minimum inter-partition distance
    pub min_inter_distance: f64,
    /// Silhouette score for the partitioning quality
    pub silhouette_score: f64,
}

/// Stratified sampling configuration for the observation table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedSamplingConfig {
    /// Minimum samples per stratum
    pub min_samples_per_stratum: usize,
    /// Maximum total samples across all strata
    pub max_total_samples: usize,
    /// Whether to use proportional allocation
    pub proportional_allocation: bool,
    /// Confidence level for stratum boundaries
    pub confidence_level: f64,
    /// Number of strata for continuous outcomes
    pub num_strata: usize,
}

impl Default for StratifiedSamplingConfig {
    fn default() -> Self {
        Self {
            min_samples_per_stratum: 10,
            max_total_samples: 10000,
            proportional_allocation: true,
            confidence_level: 0.95,
            num_strata: 5,
        }
    }
}

/// A stratum in the stratified sampling scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stratum {
    pub id: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub count: usize,
    pub mean: f64,
    pub variance: f64,
    pub allocated_samples: usize,
}

impl Stratum {
    pub fn new(id: usize, lower: f64, upper: f64) -> Self {
        Self {
            id,
            lower_bound: lower,
            upper_bound: upper,
            count: 0,
            mean: 0.0,
            variance: 0.0,
            allocated_samples: 0,
        }
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower_bound && value < self.upper_bound
    }

    pub fn add_observation(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.variance += delta * delta2;
    }

    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        self.variance / (self.count - 1) as f64
    }

    pub fn standard_deviation(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    pub fn weight(&self, total: usize) -> f64 {
        if total == 0 { return 0.0; }
        self.count as f64 / total as f64
    }
}

/// Diff between two observation table snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableDiff {
    pub added_upper_rows: Vec<Word>,
    pub added_lower_rows: Vec<Word>,
    pub added_columns: Vec<Word>,
    pub changed_entries: Vec<(Word, Word, f64)>,
    pub total_distribution_drift: f64,
    pub max_entry_change: f64,
}

impl TableDiff {
    pub fn is_empty(&self) -> bool {
        self.added_upper_rows.is_empty()
            && self.added_lower_rows.is_empty()
            && self.added_columns.is_empty()
            && self.changed_entries.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "TableDiff(+{}S, +{}SA, +{}E, {}Δ, max_change={:.6})",
            self.added_upper_rows.len(),
            self.added_lower_rows.len(),
            self.added_columns.len(),
            self.changed_entries.len(),
            self.max_entry_change,
        )
    }
}

/// Multi-resolution observation table that maintains tables at different granularities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiResolutionTable {
    /// Tables at different tolerance levels
    pub tables: Vec<ObservationTable<Word>>,
    /// Tolerance levels (coarse to fine)
    pub tolerance_levels: Vec<f64>,
    /// Current active resolution index
    pub active_level: usize,
    /// Whether to automatically select resolution
    pub auto_select: bool,
}

impl MultiResolutionTable {
    pub fn new(tolerance_levels: Vec<f64>) -> Self {
        let tables = tolerance_levels
            .iter()
            .map(|&tol| {
                let mut config = ObservationTableConfig::default();
                config.tolerance = tol;
                ObservationTable::new(config)
            })
            .collect();
        let n = tolerance_levels.len();
        Self {
            tables,
            tolerance_levels,
            active_level: 0,
            auto_select: true,
        }
    }

    pub fn default_levels() -> Self {
        Self::new(vec![0.2, 0.1, 0.05, 0.02, 0.01])
    }

    pub fn active_table(&self) -> &ObservationTable<Word> {
        &self.tables[self.active_level]
    }

    pub fn active_table_mut(&mut self) -> &mut ObservationTable<Word> {
        &mut self.tables[self.active_level]
    }

    pub fn num_levels(&self) -> usize {
        self.tolerance_levels.len()
    }

    /// Propagate an entry to all resolution levels.
    pub fn propagate_entry(&mut self, row: Word, col: Word, dist: SubDistribution, samples: usize) {
        for table in &mut self.tables {
            table.set_distribution(row.clone(), col.clone(), dist.clone(), samples);
        }
    }

    /// Initialize all tables with the same alphabet.
    pub fn initialize_all(&mut self, alphabet: &[Symbol]) {
        for table in &mut self.tables {
            table.initialize(alphabet);
        }
    }

    /// Select the appropriate resolution level based on current state counts.
    pub fn auto_select_level(&mut self) -> usize {
        if !self.auto_select {
            return self.active_level;
        }

        let mut best_level = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (i, table) in self.tables.iter().enumerate() {
            let n_states = table.num_states() as f64;
            let tol = self.tolerance_levels[i];
            let fill_rate = if table.num_rows() * table.num_suffixes() > 0 {
                table.stats.filled_entries as f64
                    / (table.num_rows() * table.num_suffixes()) as f64
            } else {
                0.0
            };

            // Score: prefer tables with moderate state count and high fill rate
            // Penalize both too few states (underfitting) and too many (overfitting)
            let state_penalty = if n_states < 2.0 {
                -1.0
            } else if n_states > 50.0 {
                -(n_states - 50.0) * 0.1
            } else {
                0.0
            };
            let score = fill_rate * 10.0 + state_penalty - tol * 5.0;

            if score > best_score {
                best_score = score;
                best_level = i;
            }
        }

        self.active_level = best_level;
        best_level
    }

    /// Get state counts at each resolution level.
    pub fn state_counts(&self) -> Vec<(f64, usize)> {
        self.tolerance_levels
            .iter()
            .zip(self.tables.iter())
            .map(|(&tol, table)| (tol, table.num_states()))
            .collect()
    }

    /// Compute the diff between tables at two resolution levels.
    pub fn resolution_diff(&self, level1: usize, level2: usize) -> Option<TableDiff> {
        if level1 >= self.tables.len() || level2 >= self.tables.len() {
            return None;
        }
        Some(self.tables[level1].diff(&self.tables[level2]))
    }
}

impl ObservationTable<Word> {
    /// Compute the difference between this table and another.
    pub fn diff(&self, other: &ObservationTable<Word>) -> TableDiff {
        let self_upper: HashSet<&Word> = self.upper_rows.iter().collect();
        let other_upper: HashSet<&Word> = other.upper_rows.iter().collect();
        let self_lower: HashSet<&Word> = self.lower_rows.iter().collect();
        let other_lower: HashSet<&Word> = other.lower_rows.iter().collect();
        let self_cols: HashSet<&Word> = self.columns.iter().collect();
        let other_cols: HashSet<&Word> = other.columns.iter().collect();

        let added_upper: Vec<Word> = other_upper.difference(&self_upper).map(|w| (*w).clone()).collect();
        let added_lower: Vec<Word> = other_lower.difference(&self_lower).map(|w| (*w).clone()).collect();
        let added_columns: Vec<Word> = other_cols.difference(&self_cols).map(|w| (*w).clone()).collect();

        let mut changed_entries = Vec::new();
        let mut total_drift = 0.0;
        let mut max_change = 0.0f64;

        for row in self.upper_rows.iter().chain(self.lower_rows.iter()) {
            for col in &self.columns {
                let d1 = self.get_distribution(row, col);
                let d2 = other.get_distribution(row, col);
                let dist = d1.tv_distance(&d2);
                if dist > 1e-10 {
                    changed_entries.push((row.clone(), col.clone(), dist));
                    total_drift += dist;
                    max_change = max_change.max(dist);
                }
            }
        }

        TableDiff {
            added_upper_rows: added_upper,
            added_lower_rows: added_lower,
            added_columns: added_columns,
            changed_entries,
            total_distribution_drift: total_drift,
            max_entry_change: max_change,
        }
    }

    /// Partition upper rows into equivalence classes based on row signatures.
    pub fn partition_rows(&self) -> TablePartitioning {
        let mut partition_map: HashMap<usize, Vec<Word>> = HashMap::new();
        let mut row_to_partition: HashMap<String, usize> = HashMap::new();
        let mut next_partition = 0usize;
        let mut representatives: Vec<Word> = Vec::new();
        let signatures: Vec<(Word, Vec<SubDistribution>)> = self
            .upper_rows
            .iter()
            .map(|r| (r.clone(), self.row_signature(r)))
            .collect();

        for (row, sig) in &signatures {
            let mut found_partition = None;
            for (pid, rep) in representatives.iter().enumerate() {
                let rep_sig = self.row_signature(rep);
                if signature_distance(sig, &rep_sig) <= self.config.tolerance {
                    found_partition = Some(pid);
                    break;
                }
            }

            let pid = match found_partition {
                Some(p) => p,
                None => {
                    let p = next_partition;
                    next_partition += 1;
                    representatives.push(row.clone());
                    p
                }
            };

            partition_map.entry(pid).or_default().push(row.clone());
            row_to_partition.insert(format!("{}", row), pid);
        }

        // Compute quality metrics
        let mut intra_distances: Vec<f64> = Vec::new();
        let mut inter_distances: Vec<f64> = Vec::new();

        for (pid, members) in &partition_map {
            let rep_sig = self.row_signature(&representatives[*pid]);
            for m in members {
                let m_sig = self.row_signature(m);
                let d = signature_distance(&m_sig, &rep_sig);
                if d.is_finite() {
                    intra_distances.push(d);
                }
            }
        }

        for i in 0..representatives.len() {
            for j in (i + 1)..representatives.len() {
                let si = self.row_signature(&representatives[i]);
                let sj = self.row_signature(&representatives[j]);
                let d = signature_distance(&si, &sj);
                if d.is_finite() {
                    inter_distances.push(d);
                }
            }
        }

        let avg_intra = if intra_distances.is_empty() {
            0.0
        } else {
            intra_distances.iter().sum::<f64>() / intra_distances.len() as f64
        };
        let min_inter = inter_distances
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        // Silhouette score approximation
        let silhouette = if min_inter.is_finite() && min_inter > 0.0 {
            (min_inter - avg_intra) / min_inter.max(avg_intra).max(1e-15)
        } else if avg_intra < 1e-15 {
            1.0
        } else {
            0.0
        };

        TablePartitioning {
            num_partitions: representatives.len(),
            assignments: row_to_partition,
            representatives,
            avg_intra_distance: avg_intra,
            min_inter_distance: if min_inter.is_infinite() { 0.0 } else { min_inter },
            silhouette_score: silhouette.clamp(-1.0, 1.0),
        }
    }

    /// Perform stratified sampling to estimate entry distributions more efficiently.
    pub fn stratified_sample_allocation(
        &self,
        config: &StratifiedSamplingConfig,
    ) -> Vec<(Word, Word, usize)> {
        let mut allocations = Vec::new();
        let all_rows: Vec<&Word> = self
            .upper_rows
            .iter()
            .chain(self.lower_rows.iter())
            .collect();

        for row in &all_rows {
            for col in &self.columns {
                let entry = self.entries.get(&((*row).clone(), col.clone()));
                let current_samples = entry.map(|e| e.sample_count).unwrap_or(0);

                if current_samples >= config.min_samples_per_stratum * config.num_strata {
                    continue;
                }

                // Compute variance-based allocation
                let variance = entry
                    .map(|e| {
                        let dist = &e.distribution;
                        let n = dist.support_size().max(1) as f64;
                        let mean = dist.total_mass() / n;
                        dist.weights
                            .values()
                            .map(|&v| (v - mean).powi(2))
                            .sum::<f64>()
                            / n
                    })
                    .unwrap_or(0.25); // Maximum variance for unknown

                // Neyman allocation: n_h proportional to N_h * S_h
                let needed = ((variance.sqrt() * config.num_strata as f64 * 4.0) as usize)
                    .max(config.min_samples_per_stratum)
                    .min(config.max_total_samples);

                let additional = if needed > current_samples {
                    needed - current_samples
                } else {
                    0
                };

                if additional > 0 {
                    allocations.push(((*row).clone(), col.clone(), additional));
                }
            }
        }

        // Sort by priority (higher variance entries first)
        allocations.sort_by(|a, b| b.2.cmp(&a.2));

        // Enforce total budget
        let mut total = 0usize;
        allocations.retain(|(_r, _c, n)| {
            if total + n <= config.max_total_samples {
                total += n;
                true
            } else {
                false
            }
        });

        allocations
    }

    /// Compute the entropy of each row's signature for information-theoretic analysis.
    pub fn row_entropies(&self) -> Vec<(Word, f64)> {
        let mut result = Vec::new();
        for row in self.upper_rows.iter().chain(self.lower_rows.iter()) {
            let sig = self.row_signature(row);
            let entropy = signature_entropy(&sig);
            result.push((row.clone(), entropy));
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Find the most informative suffix to add next (maximum entropy reduction).
    pub fn suggest_next_suffix(&self, candidates: &[Word]) -> Option<Word> {
        let mut best_suffix = None;
        let mut best_info_gain = f64::NEG_INFINITY;

        for candidate in candidates {
            if self.columns.contains(candidate) {
                continue;
            }

            // Estimate information gain by checking how many row pairs this suffix would distinguish
            let mut distinguishing_pairs = 0usize;
            let mut total_pairs = 0usize;

            for i in 0..self.upper_rows.len() {
                for j in (i + 1)..self.upper_rows.len() {
                    total_pairs += 1;
                    let row_i = &self.upper_rows[i];
                    let row_j = &self.upper_rows[j];

                    // Check if rows are currently equivalent
                    if self.rows_equivalent(row_i, row_j) {
                        // This suffix might distinguish them
                        let wi = row_i.concat(candidate);
                        let wj = row_j.concat(candidate);
                        let di = self
                            .entries
                            .get(&(wi, Word::empty()))
                            .map(|e| e.distribution.clone())
                            .unwrap_or_default();
                        let dj = self
                            .entries
                            .get(&(wj, Word::empty()))
                            .map(|e| e.distribution.clone())
                            .unwrap_or_default();

                        if di.tv_distance(&dj) > self.config.tolerance {
                            distinguishing_pairs += 1;
                        }
                    }
                }
            }

            let info_gain = if total_pairs > 0 {
                distinguishing_pairs as f64 / total_pairs as f64
            } else {
                0.0
            };

            if info_gain > best_info_gain {
                best_info_gain = info_gain;
                best_suffix = Some(candidate.clone());
            }
        }

        best_suffix
    }

    /// Compute the fill rate of the table.
    pub fn fill_rate(&self) -> f64 {
        let total_cells = self.num_rows() * self.num_suffixes();
        if total_cells == 0 {
            return 0.0;
        }
        let filled = self.entries.values().filter(|e| e.filled).count();
        filled as f64 / total_cells as f64
    }

    /// Get rows that need more samples (low confidence).
    pub fn low_confidence_cells_by_threshold(&self, threshold: f64) -> Vec<(Word, Word, f64)> {
        let mut cells = Vec::new();
        for row in self.upper_rows.iter().chain(self.lower_rows.iter()) {
            for col in &self.columns {
                let entry = self.entries.get(&(row.clone(), col.clone()));
                let confidence = entry.map(|e| e.confidence).unwrap_or(0.0);
                if confidence < threshold {
                    cells.push((row.clone(), col.clone(), confidence));
                }
            }
        }
        cells.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        cells
    }

    /// Compute the effective dimension of the table (number of distinct row signatures).
    pub fn effective_dimension(&self) -> usize {
        let partitioning = self.partition_rows();
        partitioning.num_partitions
    }

    /// Extract a sub-table containing only the specified rows and columns.
    pub fn sub_table(&self, rows: &[Word], cols: &[Word]) -> ObservationTable<Word> {
        let mut sub = ObservationTable::new(self.config.clone());
        sub.upper_rows = rows.to_vec();
        sub.columns = cols.to_vec();
        sub.lower_rows = Vec::new();

        for row in rows {
            for col in cols {
                if let Some(entry) = self.entries.get(&(row.clone(), col.clone())) {
                    sub.entries.insert((row.clone(), col.clone()), entry.clone());
                }
            }
        }

        sub
    }

    /// Merge two tables, combining their entries using weighted averaging.
    pub fn merge_with(&mut self, other: &ObservationTable<Word>) {
        // Add any new upper rows from other
        for row in &other.upper_rows {
            if !self.upper_rows.contains(row) && !self.lower_rows.contains(row) {
                self.upper_rows.push(row.clone());
            }
        }

        // Add any new columns from other
        for col in &other.columns {
            if !self.columns.contains(col) {
                self.columns.push(col.clone());
            }
        }

        // Merge entries
        for ((row, col), entry) in &other.entries {
            if let Some(existing) = self.entries.get_mut(&(row.clone(), col.clone())) {
                existing.update(&entry.distribution, entry.sample_count);
            } else {
                self.entries.insert((row.clone(), col.clone()), entry.clone());
            }
        }
    }

    /// Rank rows by their discriminative power (how different they are from other rows).
    pub fn rank_rows_by_discriminative_power(&self) -> Vec<(Word, f64)> {
        let mut scores: Vec<(Word, f64)> = Vec::new();

        for row in &self.upper_rows {
            let sig = self.row_signature(row);
            let mut min_distance = f64::INFINITY;
            let mut total_distance = 0.0;
            let mut count = 0;

            for other_row in &self.upper_rows {
                if row == other_row {
                    continue;
                }
                let other_sig = self.row_signature(other_row);
                let dist = signature_distance(&sig, &other_sig);
                if dist.is_finite() {
                    min_distance = min_distance.min(dist);
                    total_distance += dist;
                    count += 1;
                }
            }

            let avg_distance = if count > 0 { total_distance / count as f64 } else { 0.0 };
            let power = if min_distance.is_finite() {
                min_distance * 0.3 + avg_distance * 0.7
            } else {
                avg_distance
            };
            scores.push((row.clone(), power));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Validate table integrity: ensure all entries are well-formed.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Check that all upper rows are unique
        let mut seen_upper: HashSet<&Word> = HashSet::new();
        for row in &self.upper_rows {
            if !seen_upper.insert(row) {
                errors.push(format!("Duplicate upper row: {}", row));
            }
        }

        // Check no overlap between upper and lower rows
        let upper_set: HashSet<&Word> = self.upper_rows.iter().collect();
        for row in &self.lower_rows {
            if upper_set.contains(row) {
                errors.push(format!("Row {} appears in both upper and lower rows", row));
            }
        }

        // Check that all entries have valid distributions
        for ((row, col), entry) in &self.entries {
            if entry.filled {
                let mass = entry.distribution.total_mass();
                if mass > 1.0 + 1e-6 {
                    errors.push(format!(
                        "Entry ({}, {}) has mass {} > 1.0",
                        row, col, mass
                    ));
                }
                for (key, &val) in &entry.distribution.weights {
                    if val < -1e-10 {
                        errors.push(format!(
                            "Entry ({}, {}) has negative weight {} for key {}",
                            row, col, val, key
                        ));
                    }
                }
            }
        }

        // Check epsilon is in upper rows
        if !self.upper_rows.contains(&Word::empty()) {
            errors.push("Empty word (ε) not in upper rows".to_string());
        }

        // Check epsilon is in columns
        if !self.columns.contains(&Word::empty()) {
            errors.push("Empty word (ε) not in columns".to_string());
        }

        errors
    }

    /// Compute the hash of the table state (for change detection).
    pub fn state_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.upper_rows.len().hash(&mut hasher);
        self.lower_rows.len().hash(&mut hasher);
        self.columns.len().hash(&mut hasher);

        for row in &self.upper_rows {
            row.hash(&mut hasher);
        }
        for col in &self.columns {
            col.hash(&mut hasher);
        }

        // Hash a summary of entries
        let mut entry_summary = 0u64;
        for ((row, col), entry) in &self.entries {
            if entry.filled {
                let mass_bits = (entry.distribution.total_mass() * 1000.0) as u64;
                entry_summary = entry_summary.wrapping_add(mass_bits);
            }
        }
        entry_summary.hash(&mut hasher);
        hasher.finish()
    }

    /// Find the nearest upper row to a given row based on signature distance.
    pub fn nearest_upper_row(&self, row: &Word) -> Option<(Word, f64)> {
        let target_sig = self.row_signature(row);
        let mut best: Option<(Word, f64)> = None;

        for upper in &self.upper_rows {
            let upper_sig = self.row_signature(upper);
            let dist = signature_distance(&target_sig, &upper_sig);
            if dist.is_finite() {
                match &best {
                    None => best = Some((upper.clone(), dist)),
                    Some((_, best_dist)) if dist < *best_dist => {
                        best = Some((upper.clone(), dist));
                    }
                    _ => {}
                }
            }
        }

        best
    }

    /// Compute the Kantorovich-style distance matrix between all upper row pairs.
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.upper_rows.len();
        let mut matrix = vec![vec![0.0; n]; n];

        let sigs: Vec<Vec<SubDistribution>> = self
            .upper_rows
            .iter()
            .map(|r| self.row_signature(r))
            .collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = signature_distance(&sigs[i], &sigs[j]);
                let d = if dist.is_finite() { dist } else { 1.0 };
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }

        matrix
    }

    /// Perform agglomerative clustering on upper rows and return dendogram as nested pairs.
    pub fn hierarchical_clustering(&self) -> Vec<(usize, usize, f64)> {
        let n = self.upper_rows.len();
        if n <= 1 {
            return Vec::new();
        }

        let mut matrix = self.distance_matrix();
        let mut active: Vec<bool> = vec![true; n];
        let mut merges: Vec<(usize, usize, f64)> = Vec::new();

        for _ in 0..(n - 1) {
            // Find closest pair among active clusters
            let mut min_dist = f64::INFINITY;
            let mut min_i = 0;
            let mut min_j = 1;

            for i in 0..n {
                if !active[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !active[j] {
                        continue;
                    }
                    if matrix[i][j] < min_dist {
                        min_dist = matrix[i][j];
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            merges.push((min_i, min_j, min_dist));
            active[min_j] = false;

            // Update distances using average linkage
            for k in 0..n {
                if !active[k] || k == min_i {
                    continue;
                }
                matrix[min_i][k] = (matrix[min_i][k] + matrix[min_j][k]) / 2.0;
                matrix[k][min_i] = matrix[min_i][k];
            }
        }

        merges
    }

    /// Export the table as a CSV-like string for external analysis.
    pub fn export_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("row_type,row");
        for col in &self.columns {
            csv.push_str(&format!(",\"{}\"", col));
        }
        csv.push('\n');

        // Upper rows
        for row in &self.upper_rows {
            csv.push_str(&format!("S,\"{}\"", row));
            for col in &self.columns {
                let dist = self.get_distribution(row, col);
                csv.push_str(&format!(",{:.6}", dist.total_mass()));
            }
            csv.push('\n');
        }

        // Lower rows
        for row in &self.lower_rows {
            csv.push_str(&format!("SA,\"{}\"", row));
            for col in &self.columns {
                let dist = self.get_distribution(row, col);
                csv.push_str(&format!(",{:.6}", dist.total_mass()));
            }
            csv.push('\n');
        }

        csv
    }

    /// Import entries from a CSV-like string.
    pub fn import_csv(&mut self, csv: &str) -> Result<usize, String> {
        let lines: Vec<&str> = csv.lines().collect();
        if lines.is_empty() {
            return Err("Empty CSV".to_string());
        }

        // Parse header to get columns
        let header_parts: Vec<&str> = lines[0].split(',').collect();
        if header_parts.len() < 3 {
            return Err("Header too short".to_string());
        }

        let mut imported = 0;
        for line in &lines[1..] {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 3 {
                continue;
            }

            let row_str = parts[1].trim_matches('"');
            let row = if row_str == "ε" || row_str.is_empty() {
                Word::empty()
            } else {
                Word::from_str_slice(&row_str.split('·').collect::<Vec<&str>>())
            };

            for (i, col) in self.columns.clone().iter().enumerate() {
                if i + 2 < parts.len() {
                    if let Ok(mass) = parts[i + 2].trim().parse::<f64>() {
                        let dist = SubDistribution::singleton("_total".to_string(), mass);
                        self.set_distribution(row.clone(), col.clone(), dist, 1);
                        imported += 1;
                    }
                }
            }
        }

        Ok(imported)
    }
}

/// Compute the Shannon entropy of a signature (vector of sub-distributions).
fn signature_entropy(sig: &[SubDistribution]) -> f64 {
    let mut total_entropy = 0.0;
    for dist in sig {
        let mass = dist.total_mass();
        if mass < 1e-15 {
            continue;
        }
        for &prob in dist.weights.values() {
            let p = prob / mass;
            if p > 1e-15 {
                total_entropy -= p * p.ln();
            }
        }
    }
    total_entropy
}

/// Compute the Hellinger distance between two sub-distributions.
pub fn hellinger_distance(d1: &SubDistribution, d2: &SubDistribution) -> f64 {
    let mut all_keys: HashSet<&String> = d1.weights.keys().collect();
    all_keys.extend(d2.weights.keys());

    let m1 = d1.total_mass().max(1e-15);
    let m2 = d2.total_mass().max(1e-15);

    let mut sum_sq = 0.0;
    for key in all_keys {
        let p = d1.get(key) / m1;
        let q = d2.get(key) / m2;
        sum_sq += (p.sqrt() - q.sqrt()).powi(2);
    }

    (sum_sq / 2.0).sqrt()
}

/// Compute the Jensen-Shannon divergence between two sub-distributions.
pub fn jensen_shannon_divergence(d1: &SubDistribution, d2: &SubDistribution) -> f64 {
    let m = d1.merge_weighted(d2, 0.5);
    let kl1 = kl_divergence(d1, &m);
    let kl2 = kl_divergence(d2, &m);
    (kl1 + kl2) / 2.0
}

/// Compute the KL divergence D_KL(P || Q).
pub fn kl_divergence(p: &SubDistribution, q: &SubDistribution) -> f64 {
    let mp = p.total_mass().max(1e-15);
    let mq = q.total_mass().max(1e-15);

    let mut kl = 0.0;
    for (key, &pv) in &p.weights {
        let pp = pv / mp;
        let qp = q.get(key) / mq;
        if pp > 1e-15 && qp > 1e-15 {
            kl += pp * (pp / qp).ln();
        } else if pp > 1e-15 {
            kl += pp * (pp / 1e-15).ln();
        }
    }

    kl.max(0.0)
}

/// Chi-squared test statistic between observed and expected distributions.
pub fn chi_squared_statistic(observed: &SubDistribution, expected: &SubDistribution) -> f64 {
    let mut all_keys: HashSet<&String> = observed.weights.keys().collect();
    all_keys.extend(expected.weights.keys());

    let n_obs = observed.total_mass();
    let n_exp = expected.total_mass();

    if n_obs < 1e-15 || n_exp < 1e-15 {
        return 0.0;
    }

    let mut chi2 = 0.0;
    for key in all_keys {
        let o = observed.get(key) / n_obs;
        let e = expected.get(key) / n_exp;
        if e > 1e-15 {
            chi2 += (o - e).powi(2) / e;
        }
    }

    chi2
}

/// Two-sample Kolmogorov-Smirnov test statistic between two distributions.
pub fn ks_statistic(d1: &SubDistribution, d2: &SubDistribution) -> f64 {
    let mut all_keys: BTreeSet<&String> = d1.weights.keys().collect();
    all_keys.extend(d2.weights.keys());

    let m1 = d1.total_mass().max(1e-15);
    let m2 = d2.total_mass().max(1e-15);

    let mut cdf1 = 0.0;
    let mut cdf2 = 0.0;
    let mut max_diff = 0.0f64;

    for key in all_keys {
        cdf1 += d1.get(key) / m1;
        cdf2 += d2.get(key) / m2;
        max_diff = max_diff.max((cdf1 - cdf2).abs());
    }

    max_diff
}

/// Cramér-von Mises test statistic between two distributions.
pub fn cramer_von_mises_statistic(d1: &SubDistribution, d2: &SubDistribution) -> f64 {
    let mut all_keys: BTreeSet<&String> = d1.weights.keys().collect();
    all_keys.extend(d2.weights.keys());

    let m1 = d1.total_mass().max(1e-15);
    let m2 = d2.total_mass().max(1e-15);

    let mut cdf1 = 0.0;
    let mut cdf2 = 0.0;
    let mut sum_sq = 0.0;

    for key in all_keys {
        cdf1 += d1.get(key) / m1;
        cdf2 += d2.get(key) / m2;
        sum_sq += (cdf1 - cdf2).powi(2);
    }

    sum_sq
}

/// Anderson-Darling test statistic (weighted KS emphasizing tails).
pub fn anderson_darling_statistic(d1: &SubDistribution, d2: &SubDistribution) -> f64 {
    let mut all_keys: BTreeSet<&String> = d1.weights.keys().collect();
    all_keys.extend(d2.weights.keys());

    let m1 = d1.total_mass().max(1e-15);
    let m2 = d2.total_mass().max(1e-15);
    let n = all_keys.len() as f64;
    if n < 1.0 {
        return 0.0;
    }

    let mut cdf1 = 0.0;
    let mut cdf2 = 0.0;
    let mut ad = 0.0;

    for key in &all_keys {
        cdf1 += d1.get(*key) / m1;
        cdf2 += d2.get(*key) / m2;
        let combined = (cdf1 + cdf2) / 2.0;
        if combined > 1e-15 && combined < 1.0 - 1e-15 {
            ad += (cdf1 - cdf2).powi(2) / (combined * (1.0 - combined));
        }
    }

    ad / n
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

    fn make_table_with_data() -> ObservationTable<Word> {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        // Fill in some data
        // Upper row: epsilon
        table.set_distribution(
            Word::empty(),
            Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );

        // Lower rows: "a", "b"
        table.set_distribution(
            Word::from_str_slice(&["a"]),
            Word::empty(),
            SubDistribution::singleton("no".to_string(), 0.9),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["b"]),
            Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.75),
            100,
        );

        table
    }

    #[test]
    fn test_table_creation() {
        let table = ObservationTable::<Word>::with_defaults();
        assert_eq!(table.num_states(), 1);
        assert_eq!(table.num_suffixes(), 1);
    }

    #[test]
    fn test_table_initialize() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        assert_eq!(table.num_states(), 1); // Just epsilon
        assert_eq!(table.lower_rows.len(), 2); // "a" and "b"
    }

    #[test]
    fn test_table_set_get_entry() {
        let mut table = ObservationTable::with_defaults();

        let row = Word::empty();
        let col = Word::empty();
        let dist = SubDistribution::singleton("yes".to_string(), 0.8);

        table.set_distribution(row.clone(), col.clone(), dist.clone(), 100);

        let entry = table.get_entry(&row, &col);
        assert!(entry.is_some());
        assert!(entry.unwrap().filled);
        assert_eq!(entry.unwrap().sample_count, 100);
    }

    #[test]
    fn test_table_update_entry() {
        let mut table = ObservationTable::with_defaults();

        let row = Word::empty();
        let col = Word::empty();

        // Initial entry
        table.set_distribution(
            row.clone(), col.clone(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            50,
        );

        // Update with more samples
        let new_dist = SubDistribution::singleton("yes".to_string(), 0.7);
        table.update_entry(&row, &col, &new_dist, 50);

        let entry = table.get_entry(&row, &col).unwrap();
        assert_eq!(entry.sample_count, 100);
        // Distribution should be averaged
        let val = entry.distribution.get("yes");
        assert!((val - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_row_signature() {
        let mut table = make_table_with_data();

        let sig = table.row_signature(&Word::empty());
        assert_eq!(sig.len(), 1); // One column
        assert!(sig[0].get("yes") > 0.5);
    }

    #[test]
    fn test_rows_equivalent_identical() {
        let mut table = ObservationTable::with_defaults();
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.82),
            100,
        );

        assert!(table.rows_equivalent(&Word::empty(), &Word::from_str_slice(&["a"])));
    }

    #[test]
    fn test_rows_not_equivalent() {
        let mut table = ObservationTable::with_defaults();
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("no".to_string(), 0.9),
            100,
        );

        assert!(!table.rows_equivalent(&Word::empty(), &Word::from_str_slice(&["a"])));
    }

    #[test]
    fn test_closedness_closed() {
        let mut table = ObservationTable::with_defaults();
        table.initialize(&make_alphabet());

        // Make lower rows equivalent to upper rows
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.81),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["b"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.79),
            100,
        );

        let result = table.check_closedness();
        assert!(result.is_closed());
    }

    #[test]
    fn test_closedness_not_closed() {
        let mut table = ObservationTable::with_defaults();
        table.config.use_hypothesis_test = false;
        table.initialize(&make_alphabet());

        // Make lower row "a" different from all upper rows
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("no".to_string(), 0.9),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["b"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.78),
            100,
        );

        let result = table.check_closedness();
        assert!(!result.is_closed());
        if let ClosednessResult::NotClosed { unclosed_rows } = result {
            assert!(unclosed_rows.contains(&Word::from_str_slice(&["a"])));
        }
    }

    #[test]
    fn test_consistency_consistent() {
        let mut table = ObservationTable::with_defaults();
        table.config.use_hypothesis_test = false;
        let alphabet = make_alphabet();

        // Single upper row (epsilon) is trivially consistent
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );

        let result = table.check_consistency(&alphabet);
        assert!(result.is_consistent());
    }

    #[test]
    fn test_consistency_with_multiple_rows() {
        let mut table = ObservationTable::with_defaults();
        table.config.use_hypothesis_test = false;
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        // Two equivalent upper rows
        table.upper_rows.push(Word::from_str_slice(&["a"]));
        table.lower_rows.retain(|r| r != &Word::from_str_slice(&["a"]));

        // Make them equivalent
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.81),
            100,
        );

        // Their extensions should also be equivalent
        table.set_distribution(
            Word::from_str_slice(&["a", "a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.79),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["b"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.80),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a", "b"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.82),
            100,
        );

        let result = table.check_consistency(&alphabet);
        assert!(result.is_consistent());
    }

    #[test]
    fn test_add_upper_row() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        assert_eq!(table.num_states(), 1);

        table.add_upper_row(Word::from_str_slice(&["a"]), &alphabet);

        assert_eq!(table.num_states(), 2);
        // "a" should no longer be in lower rows
        assert!(!table.lower_rows.contains(&Word::from_str_slice(&["a"])));
        // "aa" and "ab" should be in lower rows
        assert!(table.lower_rows.contains(&Word::from_str_slice(&["a", "a"])));
        assert!(table.lower_rows.contains(&Word::from_str_slice(&["a", "b"])));
    }

    #[test]
    fn test_add_column() {
        let mut table = ObservationTable::with_defaults();

        assert_eq!(table.num_suffixes(), 1);

        let added = table.add_column(Word::from_str_slice(&["a"]));
        assert!(added);
        assert_eq!(table.num_suffixes(), 2);

        // Adding duplicate should fail
        let added2 = table.add_column(Word::from_str_slice(&["a"]));
        assert!(!added2);
        assert_eq!(table.num_suffixes(), 2);
    }

    #[test]
    fn test_unfilled_cells() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        let unfilled = table.unfilled_cells();
        // All cells should be unfilled initially
        assert!(!unfilled.is_empty());
    }

    #[test]
    fn test_low_confidence_cells() {
        let mut table = ObservationTable::with_defaults();
        table.config.min_samples = 50;

        // Add entry with few samples
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            10, // Less than min_samples
        );

        let low_conf = table.low_confidence_cells();
        assert!(!low_conf.is_empty());
        assert_eq!(low_conf[0].2, 40); // Need 40 more samples
    }

    #[test]
    fn test_compact() {
        let mut table = ObservationTable::with_defaults();
        table.config.use_hypothesis_test = false;
        let alphabet = make_alphabet();

        // Add two equivalent upper rows
        table.upper_rows.push(Word::from_str_slice(&["a"]));

        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.81),
            100,
        );

        let merged = table.compact();
        assert_eq!(merged, 1);
        assert_eq!(table.num_states(), 1);
    }

    #[test]
    fn test_export_for_hypothesis() {
        let table = make_table_with_data();

        let (access_strings, suffixes, data) = table.export_for_hypothesis();

        assert_eq!(access_strings.len(), 1); // Just epsilon
        assert_eq!(suffixes.len(), 1); // Just epsilon
        assert!(!data.is_empty());
    }

    #[test]
    fn test_process_counterexample() {
        let mut table = make_table_with_data();
        let alphabet = make_alphabet();

        let ce = Word::from_str_slice(&["a", "b"]);
        let query_fn = |_word: &Word| -> SubDistribution {
            SubDistribution::singleton("yes".to_string(), 0.5)
        };

        let new_suffix = table.process_counterexample(&ce, &alphabet, &query_fn);
        assert!(!new_suffix.is_empty() || table.columns.len() > 1);
    }

    #[test]
    fn test_process_counterexample_single() {
        let mut table = make_table_with_data();
        let alphabet = make_alphabet();

        let ce = Word::from_str_slice(&["a"]);
        let query_fn = |_word: &Word| -> SubDistribution {
            SubDistribution::singleton("yes".to_string(), 0.5)
        };

        let new_suffix = table.process_counterexample(&ce, &alphabet, &query_fn);
        assert!(table.columns.len() >= 1);
    }

    #[test]
    fn test_table_entry_creation() {
        let entry = TableEntry::new();
        assert!(!entry.filled);
        assert_eq!(entry.sample_count, 0);
        assert_eq!(entry.confidence, 0.0);
    }

    #[test]
    fn test_table_entry_from_distribution() {
        let dist = SubDistribution::singleton("yes".to_string(), 0.8);
        let entry = TableEntry::from_distribution(dist, 100);

        assert!(entry.filled);
        assert_eq!(entry.sample_count, 100);
        assert!(entry.confidence > 0.0);
    }

    #[test]
    fn test_table_entry_update() {
        let mut entry = TableEntry::from_distribution(
            SubDistribution::singleton("a".to_string(), 0.6),
            50,
        );

        let new_dist = SubDistribution::singleton("a".to_string(), 0.4);
        entry.update(&new_dist, 50);

        assert_eq!(entry.sample_count, 100);
        let val = entry.distribution.get("a");
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_table_entry_standard_error() {
        let entry = TableEntry::from_distribution(
            SubDistribution::singleton("a".to_string(), 0.5),
            100,
        );

        let se = entry.standard_error();
        assert!(se > 0.0);
        assert!(se < 1.0);

        // More samples → smaller SE
        let entry2 = TableEntry::from_distribution(
            SubDistribution::singleton("a".to_string(), 0.5),
            1000,
        );
        assert!(entry2.standard_error() < se);
    }

    #[test]
    fn test_closedness_result_api() {
        let closed = ClosednessResult::Closed;
        assert!(closed.is_closed());

        let not_closed = ClosednessResult::NotClosed {
            unclosed_rows: vec![Word::from_str_slice(&["a"])],
        };
        assert!(!not_closed.is_closed());
    }

    #[test]
    fn test_consistency_result_api() {
        let consistent = ConsistencyResult::Consistent;
        assert!(consistent.is_consistent());

        let not_consistent = ConsistencyResult::NotConsistent {
            row1: Word::empty(),
            row2: Word::from_str_slice(&["a"]),
            symbol: Symbol::new("b"),
            distinguishing_suffix: Word::from_str_slice(&["b"]),
        };
        assert!(!not_consistent.is_consistent());
    }

    #[test]
    fn test_table_serialization() {
        let table = make_table_with_data();

        let json = table.to_json().unwrap();
        assert!(!json.is_empty());

        let restored = ObservationTable::<Word>::from_json(&json).unwrap();
        assert_eq!(restored.num_states(), table.num_states());
        assert_eq!(restored.num_suffixes(), table.num_suffixes());
    }

    #[test]
    fn test_table_checkpoint() {
        let table = make_table_with_data();
        let cp = table.checkpoint();

        assert_eq!(cp.upper_row_count, 1);
        assert_eq!(cp.column_count, 1);
        assert!(cp.filled_count > 0);
    }

    #[test]
    fn test_table_display() {
        let table = make_table_with_data();
        let display = table.display_table();

        assert!(display.contains("ObservationTable"));
        assert!(display.contains("upper rows"));
    }

    #[test]
    fn test_signature_distance_identical() {
        let sig = vec![SubDistribution::singleton("a".to_string(), 0.5)];
        assert!(signature_distance(&sig, &sig) < 1e-10);
    }

    #[test]
    fn test_signature_distance_different() {
        let sig1 = vec![SubDistribution::singleton("a".to_string(), 0.5)];
        let sig2 = vec![SubDistribution::singleton("b".to_string(), 0.5)];
        assert!(signature_distance(&sig1, &sig2) > 0.0);
    }

    #[test]
    fn test_signature_distance_different_lengths() {
        let sig1 = vec![SubDistribution::singleton("a".to_string(), 0.5)];
        let sig2 = vec![
            SubDistribution::singleton("a".to_string(), 0.5),
            SubDistribution::singleton("b".to_string(), 0.5),
        ];
        assert_eq!(signature_distance(&sig1, &sig2), f64::INFINITY);
    }

    #[test]
    fn test_sub_distribution_merge_weighted() {
        let d1 = SubDistribution::singleton("a".to_string(), 0.8);
        let d2 = SubDistribution::singleton("a".to_string(), 0.4);

        let merged = d1.merge_weighted(&d2, 0.5);
        assert!((merged.get("a") - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_sub_distribution_linf() {
        let d1 = SubDistribution::from_map(
            [("a".to_string(), 0.5), ("b".to_string(), 0.3)]
                .into_iter().collect(),
        );
        let d2 = SubDistribution::from_map(
            [("a".to_string(), 0.2), ("b".to_string(), 0.3)]
                .into_iter().collect(),
        );

        let dist = d1.linf_distance(&d2);
        assert!((dist - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_table_stats() {
        let table = make_table_with_data();
        assert!(table.stats.total_entries > 0);
        assert!(table.stats.filled_entries > 0);
        assert!(table.stats.total_samples > 0);
    }

    #[test]
    fn test_table_config_default() {
        let config = ObservationTableConfig::default();
        assert!(config.tolerance > 0.0);
        assert!(config.min_samples > 0);
        assert!(config.max_rows > 0);
        assert!(config.max_columns > 0);
    }

    #[test]
    fn test_add_upper_row_duplicate() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();

        let before = table.num_states();
        table.add_upper_row(Word::empty(), &alphabet); // epsilon already exists
        assert_eq!(table.num_states(), before);
    }

    #[test]
    fn test_table_row_count() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        assert_eq!(table.num_rows(), 3); // epsilon + a + b
    }

    #[test]
    fn test_checkpoint_display() {
        let cp = TableCheckpoint {
            upper_row_count: 5,
            lower_row_count: 10,
            column_count: 3,
            entry_count: 45,
            filled_count: 40,
            total_samples: 4000,
            stats: TableStats::default(),
        };

        let s = format!("{}", cp);
        assert!(s.contains("S=5"));
        assert!(s.contains("SA=10"));
    }

    #[test]
    fn test_hypothesis_test_equivalence() {
        let mut table = ObservationTable::with_defaults();
        table.config.use_hypothesis_test = true;
        table.config.significance_level = 0.05;

        // With high samples and similar distributions, should be equivalent
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.80),
            1000,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.81),
            1000,
        );

        assert!(table.rows_equivalent(&Word::empty(), &Word::from_str_slice(&["a"])));
    }

    #[test]
    fn test_hypothesis_test_not_equivalent() {
        let mut table = ObservationTable::with_defaults();
        table.config.use_hypothesis_test = true;

        // With high samples and very different distributions, should not be equivalent
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.80),
            1000,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("no".to_string(), 0.90),
            1000,
        );

        assert!(!table.rows_equivalent(&Word::empty(), &Word::from_str_slice(&["a"])));
    }

    #[test]
    fn test_sub_distribution_is_empty() {
        let d = SubDistribution::new();
        assert!(d.is_empty_dist());

        let d2 = SubDistribution::singleton("a".to_string(), 0.5);
        assert!(!d2.is_empty_dist());
    }

    #[test]
    fn test_word_without_last() {
        let w = Word::from_str_slice(&["a", "b", "c"]);
        let wl = w.without_last();
        assert_eq!(wl.len(), 2);
        assert_eq!(wl.symbols[0], Symbol::new("a"));
        assert_eq!(wl.symbols[1], Symbol::new("b"));

        let empty = Word::empty();
        assert!(empty.without_last().is_empty());
    }

    #[test]
    fn test_word_last_symbol() {
        let w = Word::from_str_slice(&["a", "b"]);
        assert_eq!(w.last_symbol(), Some(&Symbol::new("b")));

        let empty = Word::empty();
        assert_eq!(empty.last_symbol(), None);
    }

    #[test]
    fn test_partition_rows_single_state() {
        let table = make_table_with_data();
        let partitioning = table.partition_rows();
        assert!(partitioning.num_partitions >= 1);
        assert!(!partitioning.representatives.is_empty());
    }

    #[test]
    fn test_partition_rows_multiple_states() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        // Make epsilon and "a" clearly different
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.9),
            100,
        );
        table.add_upper_row(Word::from_str_slice(&["a"]), &alphabet);
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("no".to_string(), 0.9),
            100,
        );

        let partitioning = table.partition_rows();
        assert_eq!(partitioning.num_partitions, 2);
    }

    #[test]
    fn test_effective_dimension() {
        let table = make_table_with_data();
        let dim = table.effective_dimension();
        assert!(dim >= 1);
    }

    #[test]
    fn test_fill_rate() {
        let table = make_table_with_data();
        let rate = table.fill_rate();
        assert!(rate >= 0.0);
        assert!(rate <= 1.0);
    }

    #[test]
    fn test_fill_rate_empty_table() {
        let table = ObservationTable::with_defaults();
        let rate = table.fill_rate();
        assert!(rate >= 0.0);
    }

    #[test]
    fn test_low_confidence_cells() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        // Add one entry with high confidence
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            1000,
        );

        let cells = table.low_confidence_cells_by_threshold(0.9);
        // Should find unfilled cells
        assert!(!cells.is_empty() || table.fill_rate() > 0.5);
    }

    #[test]
    fn test_validate_good_table() {
        let table = make_table_with_data();
        let errors = table.validate();
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_validate_bad_table() {
        let mut table = ObservationTable::with_defaults();
        // Don't add epsilon to columns - should be there by default
        // Add duplicate upper row
        table.upper_rows.push(Word::empty());
        let errors = table.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_table_diff_identical() {
        let table = make_table_with_data();
        let diff = table.diff(&table);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_table_diff_different() {
        let table1 = make_table_with_data();
        let mut table2 = table1.clone();
        table2.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("no".to_string(), 0.9),
            100,
        );
        let diff = table1.diff(&table2);
        assert!(!diff.changed_entries.is_empty());
        assert!(diff.max_entry_change > 0.0);
    }

    #[test]
    fn test_distance_matrix() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);
        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.add_upper_row(Word::from_str_slice(&["a"]), &alphabet);
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.3),
            100,
        );

        let dm = table.distance_matrix();
        assert_eq!(dm.len(), 2);
        assert_eq!(dm[0].len(), 2);
        assert_eq!(dm[0][0], 0.0);
        assert_eq!(dm[1][1], 0.0);
        assert!(dm[0][1] > 0.0);
        assert!((dm[0][1] - dm[1][0]).abs() < 1e-10); // Symmetric
    }

    #[test]
    fn test_hierarchical_clustering_single() {
        let table = make_table_with_data();
        let merges = table.hierarchical_clustering();
        assert!(merges.is_empty()); // Single state → no merges
    }

    #[test]
    fn test_export_csv() {
        let table = make_table_with_data();
        let csv = table.export_csv();
        assert!(csv.contains("row_type,row"));
        assert!(csv.contains("S,"));
    }

    #[test]
    fn test_state_hash_deterministic() {
        let table = make_table_with_data();
        let h1 = table.state_hash();
        let h2 = table.state_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_state_hash_changes_with_data() {
        let table1 = make_table_with_data();
        let mut table2 = table1.clone();
        table2.add_column(Word::from_str_slice(&["x"]));
        assert_ne!(table1.state_hash(), table2.state_hash());
    }

    #[test]
    fn test_nearest_upper_row() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.79),
            100,
        );

        let nearest = table.nearest_upper_row(&Word::from_str_slice(&["a"]));
        assert!(nearest.is_some());
        let (row, dist) = nearest.unwrap();
        assert_eq!(row, Word::empty());
        assert!(dist < 0.1);
    }

    #[test]
    fn test_row_entropies() {
        let table = make_table_with_data();
        let entropies = table.row_entropies();
        assert!(!entropies.is_empty());
    }

    #[test]
    fn test_sub_table() {
        let table = make_table_with_data();
        let sub = table.sub_table(
            &[Word::empty()],
            &[Word::empty()],
        );
        assert_eq!(sub.num_states(), 1);
        assert_eq!(sub.num_suffixes(), 1);
    }

    #[test]
    fn test_merge_tables() {
        let mut table1 = make_table_with_data();
        let mut table2 = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table2.initialize(&alphabet);
        table2.set_distribution(
            Word::from_str_slice(&["b"]), Word::empty(),
            SubDistribution::singleton("maybe".to_string(), 0.5),
            50,
        );

        let states_before = table1.num_states();
        table1.merge_with(&table2);
        // Should not have fewer states
        assert!(table1.num_states() >= states_before);
    }

    #[test]
    fn test_rank_rows_by_discriminative_power() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        table.set_distribution(
            Word::empty(), Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.9),
            100,
        );
        table.add_upper_row(Word::from_str_slice(&["a"]), &alphabet);
        table.set_distribution(
            Word::from_str_slice(&["a"]), Word::empty(),
            SubDistribution::singleton("no".to_string(), 0.9),
            100,
        );

        let ranked = table.rank_rows_by_discriminative_power();
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].1 >= ranked[1].1);
    }

    #[test]
    fn test_multi_resolution_table() {
        let mut mrt = MultiResolutionTable::default_levels();
        let alphabet = vec![Symbol::new("a"), Symbol::new("b")];
        mrt.initialize_all(&alphabet);

        assert_eq!(mrt.num_levels(), 5);
        assert!(mrt.active_table().num_states() >= 1);
    }

    #[test]
    fn test_multi_resolution_propagate() {
        let mut mrt = MultiResolutionTable::default_levels();
        let alphabet = vec![Symbol::new("a")];
        mrt.initialize_all(&alphabet);

        mrt.propagate_entry(
            Word::empty(),
            Word::empty(),
            SubDistribution::singleton("yes".to_string(), 0.8),
            100,
        );

        // All levels should have the entry
        for table in &mrt.tables {
            let d = table.get_distribution(&Word::empty(), &Word::empty());
            assert!((d.get("yes") - 0.8).abs() < 0.01);
        }
    }

    #[test]
    fn test_multi_resolution_auto_select() {
        let mut mrt = MultiResolutionTable::default_levels();
        let alphabet = vec![Symbol::new("a")];
        mrt.initialize_all(&alphabet);

        let level = mrt.auto_select_level();
        assert!(level < mrt.num_levels());
    }

    #[test]
    fn test_multi_resolution_state_counts() {
        let mrt = MultiResolutionTable::default_levels();
        let counts = mrt.state_counts();
        assert_eq!(counts.len(), 5);
        for (tol, count) in &counts {
            assert!(*tol > 0.0);
            assert!(*count >= 1);
        }
    }

    #[test]
    fn test_hellinger_distance_identical() {
        let d = SubDistribution::singleton("a".to_string(), 0.5);
        assert!(hellinger_distance(&d, &d) < 1e-10);
    }

    #[test]
    fn test_hellinger_distance_different() {
        let d1 = SubDistribution::singleton("a".to_string(), 1.0);
        let d2 = SubDistribution::singleton("b".to_string(), 1.0);
        let h = hellinger_distance(&d1, &d2);
        assert!(h > 0.9); // Completely disjoint
    }

    #[test]
    fn test_jensen_shannon_divergence() {
        let d1 = SubDistribution::singleton("a".to_string(), 0.5);
        let d2 = SubDistribution::singleton("a".to_string(), 0.5);
        let jsd = jensen_shannon_divergence(&d1, &d2);
        assert!(jsd < 1e-10); // Identical distributions
    }

    #[test]
    fn test_kl_divergence_identical() {
        let d = SubDistribution::from_map(
            [("a".to_string(), 0.5), ("b".to_string(), 0.5)]
                .into_iter().collect()
        );
        let kl = kl_divergence(&d, &d);
        assert!(kl < 1e-10);
    }

    #[test]
    fn test_chi_squared_statistic() {
        let obs = SubDistribution::from_map(
            [("a".to_string(), 0.5), ("b".to_string(), 0.5)]
                .into_iter().collect()
        );
        let chi2 = chi_squared_statistic(&obs, &obs);
        assert!(chi2 < 1e-10);
    }

    #[test]
    fn test_ks_statistic_identical() {
        let d = SubDistribution::from_map(
            [("a".to_string(), 0.3), ("b".to_string(), 0.7)]
                .into_iter().collect()
        );
        assert!(ks_statistic(&d, &d) < 1e-10);
    }

    #[test]
    fn test_ks_statistic_different() {
        let d1 = SubDistribution::singleton("a".to_string(), 1.0);
        let d2 = SubDistribution::singleton("b".to_string(), 1.0);
        assert!(ks_statistic(&d1, &d2) > 0.5);
    }

    #[test]
    fn test_cramer_von_mises() {
        let d = SubDistribution::singleton("a".to_string(), 0.5);
        assert!(cramer_von_mises_statistic(&d, &d) < 1e-10);
    }

    #[test]
    fn test_anderson_darling() {
        let d = SubDistribution::singleton("a".to_string(), 0.5);
        assert!(anderson_darling_statistic(&d, &d) < 1e-10);
    }

    #[test]
    fn test_stratified_sample_allocation() {
        let mut table = ObservationTable::with_defaults();
        let alphabet = make_alphabet();
        table.initialize(&alphabet);

        let config = StratifiedSamplingConfig::default();
        let allocations = table.stratified_sample_allocation(&config);
        // Should suggest samples for unfilled cells
        assert!(!allocations.is_empty());
    }

    #[test]
    fn test_suggest_next_suffix() {
        let table = make_table_with_data();
        let candidates = vec![
            Word::from_str_slice(&["a"]),
            Word::from_str_slice(&["b"]),
            Word::from_str_slice(&["a", "b"]),
        ];
        let suggestion = table.suggest_next_suffix(&candidates);
        // Should suggest something (or None if all already columns)
        // Just verify it doesn't crash
        let _ = suggestion;
    }

    #[test]
    fn test_table_diff_summary() {
        let diff = TableDiff {
            added_upper_rows: vec![Word::from_str_slice(&["x"])],
            added_lower_rows: vec![],
            added_columns: vec![],
            changed_entries: vec![(Word::empty(), Word::empty(), 0.1)],
            total_distribution_drift: 0.1,
            max_entry_change: 0.1,
        };
        let summary = diff.summary();
        assert!(summary.contains("+1S"));
        assert!(summary.contains("1Δ"));
    }

    #[test]
    fn test_stratum() {
        let mut s = Stratum::new(0, 0.0, 0.5);
        assert!(s.contains(0.25));
        assert!(!s.contains(0.75));

        s.add_observation(0.1);
        s.add_observation(0.2);
        s.add_observation(0.3);
        assert_eq!(s.count, 3);
        assert!((s.mean - 0.2).abs() < 0.01);
        assert!(s.sample_variance() > 0.0);
        assert!(s.standard_deviation() > 0.0);
    }

    #[test]
    fn test_stratum_weight() {
        let mut s = Stratum::new(0, 0.0, 1.0);
        s.count = 50;
        assert!((s.weight(100) - 0.5).abs() < 0.01);
        assert_eq!(s.weight(0), 0.0);
    }

    #[test]
    fn test_import_csv() {
        let mut table = ObservationTable::with_defaults();
        let csv = "row_type,row,\"ε\"\nS,\"ε\",0.800000\n";
        let result = table.import_csv(csv);
        assert!(result.is_ok());
    }

    #[test]
    fn test_import_csv_empty() {
        let mut table = ObservationTable::with_defaults();
        let result = table.import_csv("");
        assert!(result.is_err());
    }
}
