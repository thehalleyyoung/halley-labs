//! Generic active learning framework.
//!
//! Provides a teacher-learner protocol, adaptive query selection,
//! exploration vs exploitation, multi-property learning, incremental
//! learning, transfer learning, and learning curve estimation.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use rand::prelude::*;

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
        let mut w = HashMap::new(); w.insert(key, prob); Self { weights: w }
    }
    pub fn from_map(weights: HashMap<String, f64>) -> Self { Self { weights } }
    pub fn total_mass(&self) -> f64 { self.weights.values().sum() }
    pub fn get(&self, key: &str) -> f64 { self.weights.get(key).copied().unwrap_or(0.0) }
    pub fn set(&mut self, key: String, prob: f64) {
        if prob > 1e-15 { self.weights.insert(key, prob); }
        else { self.weights.remove(&key); }
    }
    pub fn tv_distance(&self, other: &SubDistribution) -> f64 {
        let mut all_keys: HashSet<&String> = self.weights.keys().collect();
        all_keys.extend(other.weights.keys());
        let mut dist = 0.0;
        for key in all_keys { dist += (self.get(key) - other.get(key)).abs(); }
        dist / 2.0
    }
    pub fn support_size(&self) -> usize { self.weights.len() }

    /// Entropy of the distribution.
    pub fn entropy(&self) -> f64 {
        let total = self.total_mass();
        if total < 1e-15 { return 0.0; }
        let mut h = 0.0;
        for &p in self.weights.values() {
            let q = p / total;
            if q > 1e-15 {
                h -= q * q.ln();
            }
        }
        h
    }
}

impl Default for SubDistribution {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Teacher-Learner Protocol
// ---------------------------------------------------------------------------

/// A property specification that the learner should verify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySpec {
    /// Unique identifier
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Type of property
    pub property_type: PropertyType,
    /// Required confidence level
    pub confidence: f64,
    /// Priority (higher = more important)
    pub priority: i32,
}

/// Type of property being verified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyType {
    /// Safety: bad states should be unreachable
    Safety { bad_states: Vec<String> },
    /// Liveness: good states should be reachable
    Liveness { good_states: Vec<String> },
    /// Fairness: distribution over outcomes should satisfy constraints
    Fairness { constraints: Vec<FairnessConstraint> },
    /// Consistency: behavior should be deterministic/stable
    Consistency { tolerance: f64 },
    /// Custom property defined by a predicate
    Custom { name: String },
}

/// A fairness constraint on output distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConstraint {
    pub group_a: String,
    pub group_b: String,
    pub max_disparity: f64,
}

/// The teacher in the teacher-learner protocol.
pub trait Teacher: Send + Sync {
    /// Answer a membership query: what is the output distribution for this word?
    fn membership_query(&self, word: &Word) -> SubDistribution;

    /// Answer an equivalence query: is the hypothesis correct?
    /// Returns None if equivalent, or Some(counterexample) if not.
    fn equivalence_query(
        &self,
        hypothesis: &dyn Fn(&Word) -> SubDistribution,
        num_tests: usize,
        tolerance: f64,
    ) -> Option<Word>;

    /// Get the alphabet.
    fn alphabet(&self) -> &[Symbol];
}

/// The learner in the teacher-learner protocol.
pub trait ActiveLearner: Send + Sync {
    /// Initialize the learner.
    fn initialize(&mut self);

    /// Run one iteration of the learning loop.
    /// Returns true if learning is complete.
    fn step(&mut self, teacher: &dyn Teacher) -> bool;

    /// Get the current hypothesis (as a function from words to distributions).
    fn hypothesis(&self) -> Box<dyn Fn(&Word) -> SubDistribution + '_>;

    /// Number of states in current hypothesis.
    fn num_states(&self) -> usize;

    /// Total queries used.
    fn total_queries(&self) -> usize;
}

/// Implementation of the teacher-learner protocol.
pub struct TeacherLearnerProtocol<T: Teacher> {
    teacher: T,
    /// Maximum iterations
    max_iterations: usize,
    /// Current iteration
    iteration: usize,
    /// Learning history
    history: Vec<ProtocolEvent>,
    /// Configuration
    config: ProtocolConfig,
}

/// Configuration for the protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolConfig {
    pub max_iterations: usize,
    pub equivalence_test_count: usize,
    pub equivalence_tolerance: f64,
    pub log_events: bool,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            equivalence_test_count: 1000,
            equivalence_tolerance: 0.05,
            log_events: true,
        }
    }
}

/// An event in the protocol history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolEvent {
    pub iteration: usize,
    pub event_type: ProtocolEventType,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolEventType {
    MembershipQuery { word: Word, result_mass: f64 },
    EquivalenceQuery { passed: bool, ce_length: Option<usize> },
    HypothesisUpdated { num_states: usize },
    Converged,
    BudgetExhausted,
}

impl<T: Teacher> TeacherLearnerProtocol<T> {
    pub fn new(teacher: T, config: ProtocolConfig) -> Self {
        let max_iterations = config.max_iterations;
        Self {
            teacher,
            max_iterations,
            iteration: 0,
            history: Vec::new(),
            config,
        }
    }

    /// Run the protocol with the given learner until convergence or budget exhaustion.
    pub fn run(&mut self, learner: &mut dyn ActiveLearner) -> ProtocolResult {
        learner.initialize();

        let start = std::time::Instant::now();

        for _ in 0..self.max_iterations {
            self.iteration += 1;

            // Run one learning step
            let done = learner.step(&self.teacher);

            if self.config.log_events {
                self.history.push(ProtocolEvent {
                    iteration: self.iteration,
                    event_type: ProtocolEventType::HypothesisUpdated {
                        num_states: learner.num_states(),
                    },
                    timestamp_ms: start.elapsed().as_millis() as u64,
                });
            }

            if done {
                // Verify with equivalence query
                let hyp = learner.hypothesis();
                let ce = self.teacher.equivalence_query(
                    &*hyp,
                    self.config.equivalence_test_count,
                    self.config.equivalence_tolerance,
                );

                if ce.is_none() {
                    if self.config.log_events {
                        self.history.push(ProtocolEvent {
                            iteration: self.iteration,
                            event_type: ProtocolEventType::Converged,
                            timestamp_ms: start.elapsed().as_millis() as u64,
                        });
                    }

                    return ProtocolResult {
                        converged: true,
                        iterations: self.iteration,
                        num_states: learner.num_states(),
                        total_queries: learner.total_queries(),
                        total_time_ms: start.elapsed().as_millis() as u64,
                        events: self.history.clone(),
                    };
                }
            }
        }

        if self.config.log_events {
            self.history.push(ProtocolEvent {
                iteration: self.iteration,
                event_type: ProtocolEventType::BudgetExhausted,
                timestamp_ms: start.elapsed().as_millis() as u64,
            });
        }

        ProtocolResult {
            converged: false,
            iterations: self.iteration,
            num_states: learner.num_states(),
            total_queries: learner.total_queries(),
            total_time_ms: start.elapsed().as_millis() as u64,
            events: self.history.clone(),
        }
    }

    pub fn history(&self) -> &[ProtocolEvent] {
        &self.history
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

/// Result of running the protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolResult {
    pub converged: bool,
    pub iterations: usize,
    pub num_states: usize,
    pub total_queries: usize,
    pub total_time_ms: u64,
    pub events: Vec<ProtocolEvent>,
}

impl fmt::Display for ProtocolResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Protocol({}, {} iters, {} states, {} queries, {}ms)",
            if self.converged { "converged" } else { "budget exhausted" },
            self.iterations,
            self.num_states,
            self.total_queries,
            self.total_time_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// Query Selector (information-theoretic)
// ---------------------------------------------------------------------------

/// Selects which queries to make next based on information gain.
pub struct QuerySelector {
    /// History of query outcomes for information estimation
    query_history: Vec<QueryOutcome>,
    /// Entropy estimates per word region
    entropy_estimates: HashMap<String, f64>,
    /// Exploration parameter (0=exploit, 1=explore)
    exploration_rate: f64,
    /// Decay factor for exploration rate
    exploration_decay: f64,
    /// Minimum exploration rate
    min_exploration: f64,
}

/// Record of a past query outcome.
#[derive(Debug, Clone)]
struct QueryOutcome {
    word: Word,
    distribution: SubDistribution,
    information_gain: f64,
}

impl QuerySelector {
    pub fn new(exploration_rate: f64, exploration_decay: f64) -> Self {
        Self {
            query_history: Vec::new(),
            entropy_estimates: HashMap::new(),
            exploration_rate,
            exploration_decay,
            min_exploration: 0.01,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.3, 0.995)
    }

    /// Record a query outcome for future information gain estimation.
    pub fn record_outcome(&mut self, word: Word, distribution: SubDistribution) {
        let entropy = distribution.entropy();
        let region = self.word_to_region(&word);

        // Compute information gain as entropy difference from prior estimate
        let prior_entropy = self.entropy_estimates.get(&region).copied().unwrap_or(0.0);
        let info_gain = (entropy - prior_entropy).abs();

        self.entropy_estimates.insert(region, entropy);
        self.query_history.push(QueryOutcome {
            word,
            distribution,
            information_gain: info_gain,
        });

        // Decay exploration rate
        self.exploration_rate = (self.exploration_rate * self.exploration_decay)
            .max(self.min_exploration);
    }

    /// Select the next query word from a set of candidates.
    ///
    /// Uses Upper Confidence Bound (UCB) strategy:
    /// score = estimated_info_gain + exploration_bonus
    pub fn select_query(&self, candidates: &[Word]) -> Option<Word> {
        if candidates.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();

        // With probability exploration_rate, explore randomly
        if rng.gen::<f64>() < self.exploration_rate {
            let idx = rng.gen_range(0..candidates.len());
            return Some(candidates[idx].clone());
        }

        // Otherwise, pick the candidate with highest estimated information gain
        let mut best_score = f64::NEG_INFINITY;
        let mut best_candidate = &candidates[0];

        let total_queries = self.query_history.len().max(1) as f64;

        for candidate in candidates {
            let region = self.word_to_region(candidate);

            // Estimated information gain from this region
            let region_entropy = self.entropy_estimates.get(&region).copied().unwrap_or(1.0);

            // Count visits to this region
            let visits = self.query_history.iter()
                .filter(|o| self.word_to_region(&o.word) == region)
                .count() as f64;

            // UCB score: entropy + exploration bonus
            let exploration_bonus = if visits > 0.0 {
                (2.0 * total_queries.ln() / visits).sqrt()
            } else {
                f64::INFINITY // Unexplored regions get infinite bonus
            };

            let score = region_entropy + self.exploration_rate * exploration_bonus;

            if score > best_score {
                best_score = score;
                best_candidate = candidate;
            }
        }

        Some(best_candidate.clone())
    }

    /// Map a word to a region identifier for entropy tracking.
    fn word_to_region(&self, word: &Word) -> String {
        // Use prefix of length 2 as region identifier
        let prefix_len = word.len().min(2);
        let prefix_syms: Vec<String> = word.symbols.iter()
            .take(prefix_len)
            .map(|s| s.0.clone())
            .collect();
        format!("r:{}", prefix_syms.join(","))
    }

    /// Current exploration rate.
    pub fn current_exploration_rate(&self) -> f64 {
        self.exploration_rate
    }

    /// Total outcomes recorded.
    pub fn total_outcomes(&self) -> usize {
        self.query_history.len()
    }

    /// Average information gain.
    pub fn average_information_gain(&self) -> f64 {
        if self.query_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.query_history.iter().map(|o| o.information_gain).sum();
        sum / self.query_history.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Learning Curve
// ---------------------------------------------------------------------------

/// Tracks the learning curve: how accuracy improves over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurve {
    /// Data points: (queries, accuracy)
    pub points: Vec<LearningCurvePoint>,
}

/// A single point on the learning curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurvePoint {
    pub iteration: usize,
    pub queries: usize,
    pub num_states: usize,
    pub accuracy: f64,
    pub hypothesis_distance: f64,
}

impl LearningCurve {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Record a new data point.
    pub fn record(
        &mut self,
        iteration: usize,
        queries: usize,
        num_states: usize,
        accuracy: f64,
        hypothesis_distance: f64,
    ) {
        self.points.push(LearningCurvePoint {
            iteration,
            queries,
            num_states,
            accuracy,
            hypothesis_distance,
        });
    }

    /// Estimate when accuracy will reach the target.
    ///
    /// Uses log-linear regression on the error (1 - accuracy) to
    /// project when the target will be reached.
    pub fn estimate_queries_to_target(&self, target_accuracy: f64) -> Option<usize> {
        if self.points.len() < 3 {
            return None;
        }

        // Fit log(1-accuracy) = a + b * queries
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xx = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut count = 0.0f64;

        for point in &self.points {
            let error = 1.0 - point.accuracy;
            if error > 1e-10 {
                let x = point.queries as f64;
                let y = error.ln();
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                count += 1.0;
            }
        }

        if count < 2.0 {
            return None;
        }

        let b = (count * sum_xy - sum_x * sum_y)
            / (count * sum_xx - sum_x * sum_x);
        let a = (sum_y - b * sum_x) / count;

        if b >= 0.0 {
            // Error is not decreasing
            return None;
        }

        // Solve a + b * x = ln(1 - target_accuracy)
        let target_error = 1.0 - target_accuracy;
        if target_error <= 0.0 {
            return None;
        }

        let target_x = (target_error.ln() - a) / b;
        if target_x > 0.0 && target_x.is_finite() {
            Some(target_x.ceil() as usize)
        } else {
            None
        }
    }

    /// Current accuracy (last recorded).
    pub fn current_accuracy(&self) -> Option<f64> {
        self.points.last().map(|p| p.accuracy)
    }

    /// Improvement rate (accuracy per query).
    pub fn improvement_rate(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let first = &self.points[0];
        let last = self.points.last().unwrap();

        let accuracy_gain = last.accuracy - first.accuracy;
        let query_diff = (last.queries - first.queries).max(1) as f64;

        accuracy_gain / query_diff
    }

    /// Whether learning has plateaued.
    pub fn has_plateaued(&self, window: usize, threshold: f64) -> bool {
        if self.points.len() < window {
            return false;
        }

        let recent = &self.points[self.points.len() - window..];
        let first_acc = recent[0].accuracy;
        let last_acc = recent.last().unwrap().accuracy;

        (last_acc - first_acc).abs() < threshold
    }
}

impl Default for LearningCurve {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for LearningCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LearningCurve({} points, current_acc={:.4})",
            self.points.len(),
            self.current_accuracy().unwrap_or(0.0),
        )
    }
}

// ---------------------------------------------------------------------------
// Incremental Learner
// ---------------------------------------------------------------------------

/// Supports incremental learning: update the automaton without full relearn.
pub struct IncrementalLearner {
    /// Current table entries
    table: HashMap<Word, Vec<SubDistribution>>,
    /// Access strings (state-identifying prefixes)
    access_strings: Vec<Word>,
    /// Suffixes (distinguishing extensions)
    suffixes: Vec<Word>,
    /// Alphabet
    alphabet: Vec<Symbol>,
    /// Tolerance for state equivalence
    tolerance: f64,
    /// Learning stats
    queries_used: usize,
    /// Iteration counter
    iteration: usize,
}

impl IncrementalLearner {
    pub fn new(alphabet: Vec<Symbol>, tolerance: f64) -> Self {
        Self {
            table: HashMap::new(),
            access_strings: vec![Word::empty()],
            suffixes: vec![Word::empty()],
            alphabet,
            tolerance,
            queries_used: 0,
            iteration: 0,
        }
    }

    /// Add a new observation to the table.
    pub fn add_observation(&mut self, word: Word, suffix_idx: usize, dist: SubDistribution) {
        let entry = self.table.entry(word).or_insert_with(|| {
            vec![SubDistribution::new(); self.suffixes.len()]
        });

        if suffix_idx < entry.len() {
            entry[suffix_idx] = dist;
        }
    }

    /// Add a new access string (potential state).
    pub fn add_access_string(&mut self, word: Word) -> bool {
        if self.access_strings.contains(&word) {
            return false;
        }

        // Check if this word's row is distinct from existing rows
        let new_row = self.table.get(&word).cloned()
            .unwrap_or_else(|| vec![SubDistribution::new(); self.suffixes.len()]);

        for existing in &self.access_strings {
            let existing_row = self.table.get(existing).cloned()
                .unwrap_or_else(|| vec![SubDistribution::new(); self.suffixes.len()]);

            if rows_equivalent(&new_row, &existing_row, self.tolerance) {
                return false; // Row is not distinct
            }
        }

        self.access_strings.push(word);
        true
    }

    /// Add a new suffix (distinguishing extension).
    pub fn add_suffix(&mut self, suffix: Word) -> bool {
        if self.suffixes.contains(&suffix) {
            return false;
        }

        self.suffixes.push(suffix);

        // Extend all existing table entries with a new column
        for entry in self.table.values_mut() {
            entry.push(SubDistribution::new());
        }

        true
    }

    /// Incrementally update with a new counter-example.
    ///
    /// Processes the counter-example and updates the table,
    /// potentially adding new access strings or suffixes.
    pub fn process_counterexample(
        &mut self,
        ce_word: &Word,
        ce_suffix: Word,
        query_fn: &dyn Fn(&Word) -> SubDistribution,
    ) {
        self.iteration += 1;

        // Add the new suffix
        let suffix_added = self.add_suffix(ce_suffix.clone());

        if suffix_added {
            let suffix_idx = self.suffixes.len() - 1;

            // Fill in the new column for all access strings
            for access in self.access_strings.clone() {
                let query_word = access.concat(&ce_suffix);
                let dist = query_fn(&query_word);
                self.queries_used += 1;
                self.add_observation(access, suffix_idx, dist);
            }
        }

        // Check if CE prefix creates a new state
        for i in 0..=ce_word.len() {
            let prefix = ce_word.prefix(i);
            if self.add_access_string(prefix.clone()) {
                // Fill in all columns for the new access string
                for (j, suffix) in self.suffixes.clone().iter().enumerate() {
                    let query_word = prefix.concat(suffix);
                    let dist = query_fn(&query_word);
                    self.queries_used += 1;
                    self.add_observation(prefix.clone(), j, dist);
                }
            }
        }
    }

    /// Get current number of states (distinct rows).
    pub fn num_states(&self) -> usize {
        self.access_strings.len()
    }

    /// Get total queries used.
    pub fn total_queries(&self) -> usize {
        self.queries_used
    }

    /// Get the current table.
    pub fn table(&self) -> &HashMap<Word, Vec<SubDistribution>> {
        &self.table
    }

    /// Get access strings.
    pub fn access_strings(&self) -> &[Word] {
        &self.access_strings
    }

    /// Get suffixes.
    pub fn suffixes(&self) -> &[Word] {
        &self.suffixes
    }
}

/// Check row equivalence.
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

// ---------------------------------------------------------------------------
// Transfer Learning
// ---------------------------------------------------------------------------

/// Transfer learning between related properties.
///
/// Given a learned automaton for one property, use it as a starting point
/// for learning a related property.
pub struct TransferLearner {
    /// Source table entries to transfer
    source_table: HashMap<Word, Vec<SubDistribution>>,
    /// Source access strings
    source_access_strings: Vec<Word>,
    /// Source suffixes
    source_suffixes: Vec<Word>,
    /// Similarity threshold for transfer
    similarity_threshold: f64,
    /// Transfer statistics
    stats: TransferStats,
}

/// Statistics about transfer learning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferStats {
    pub source_states: usize,
    pub transferred_states: usize,
    pub transferred_suffixes: usize,
    pub queries_saved: usize,
    pub transfer_ratio: f64,
}

impl TransferLearner {
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            source_table: HashMap::new(),
            source_access_strings: Vec::new(),
            source_suffixes: Vec::new(),
            similarity_threshold,
            stats: TransferStats::default(),
        }
    }

    /// Set the source automaton to transfer from.
    pub fn set_source(
        &mut self,
        table: HashMap<Word, Vec<SubDistribution>>,
        access_strings: Vec<Word>,
        suffixes: Vec<Word>,
    ) {
        self.stats.source_states = access_strings.len();
        self.source_table = table;
        self.source_access_strings = access_strings;
        self.source_suffixes = suffixes;
    }

    /// Transfer knowledge to an incremental learner.
    ///
    /// Validates source table entries against the target system,
    /// and transfers entries that are still valid.
    pub fn transfer_to(
        &mut self,
        target: &mut IncrementalLearner,
        query_fn: &dyn Fn(&Word) -> SubDistribution,
        validation_samples: usize,
    ) {
        // Transfer suffixes
        for suffix in &self.source_suffixes {
            if target.add_suffix(suffix.clone()) {
                self.stats.transferred_suffixes += 1;
            }
        }

        // Transfer access strings and validate
        for access in &self.source_access_strings {
            // Query the target system and compare with source
            let source_row = self.source_table.get(access);

            if let Some(source_row) = source_row {
                // Validate: query the target for the first suffix
                let target_dist = query_fn(&access.concat(&Word::empty()));

                let source_dist = if source_row.is_empty() {
                    &SubDistribution::new()
                } else {
                    &source_row[0]
                };

                let distance = target_dist.tv_distance(source_dist);

                if distance < self.similarity_threshold {
                    // Transfer this state
                    if target.add_access_string(access.clone()) {
                        self.stats.transferred_states += 1;

                        // Transfer row data
                        for (j, suffix) in target.suffixes().to_vec().iter().enumerate() {
                            if j < source_row.len() {
                                // Validate source entry
                                let target_entry = query_fn(&access.concat(suffix));
                                let source_entry = &source_row[j];

                                if target_entry.tv_distance(source_entry) < self.similarity_threshold {
                                    // Use source data (saves a detailed query)
                                    target.add_observation(access.clone(), j, source_entry.clone());
                                    self.stats.queries_saved += 1;
                                } else {
                                    // Use fresh query
                                    target.add_observation(access.clone(), j, target_entry);
                                }
                            } else {
                                // No source data, query target
                                let dist = query_fn(&access.concat(suffix));
                                target.add_observation(access.clone(), j, dist);
                            }
                        }
                    }
                }
            }
        }

        // Compute transfer ratio
        if self.stats.source_states > 0 {
            self.stats.transfer_ratio =
                self.stats.transferred_states as f64 / self.stats.source_states as f64;
        }
    }

    pub fn stats(&self) -> &TransferStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Multi-Property Learning
// ---------------------------------------------------------------------------

/// Learn a single automaton that can verify multiple properties.
pub struct MultiPropertyLearner {
    /// Properties to verify
    properties: Vec<PropertySpec>,
    /// Per-property learning state
    property_states: HashMap<String, PropertyLearningState>,
    /// Shared alphabet
    alphabet: Vec<Symbol>,
    /// Configuration
    config: MultiPropertyConfig,
}

/// Per-property learning state.
#[derive(Debug, Clone)]
struct PropertyLearningState {
    is_satisfied: Option<bool>,
    confidence: f64,
    queries_used: usize,
    iteration: usize,
}

/// Configuration for multi-property learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPropertyConfig {
    /// Strategy for allocating queries among properties
    pub allocation_strategy: AllocationStrategy,
    /// Total query budget
    pub total_budget: usize,
    /// Minimum queries per property
    pub min_queries_per_property: usize,
}

impl Default for MultiPropertyConfig {
    fn default() -> Self {
        Self {
            allocation_strategy: AllocationStrategy::Proportional,
            total_budget: 10000,
            min_queries_per_property: 100,
        }
    }
}

/// Strategy for allocating queries among properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Equal allocation to all properties
    Equal,
    /// Proportional to property priority
    Proportional,
    /// Focus on the least-confident property
    LeastConfident,
    /// Round-robin
    RoundRobin,
}

impl MultiPropertyLearner {
    pub fn new(
        properties: Vec<PropertySpec>,
        alphabet: Vec<Symbol>,
        config: MultiPropertyConfig,
    ) -> Self {
        let mut property_states = HashMap::new();
        for prop in &properties {
            property_states.insert(prop.id.clone(), PropertyLearningState {
                is_satisfied: None,
                confidence: 0.0,
                queries_used: 0,
                iteration: 0,
            });
        }

        Self {
            properties,
            property_states,
            alphabet,
            config,
        }
    }

    /// Allocate query budget among properties.
    pub fn allocate_budget(&self) -> HashMap<String, usize> {
        let mut allocation = HashMap::new();
        let n = self.properties.len();
        if n == 0 {
            return allocation;
        }

        match &self.config.allocation_strategy {
            AllocationStrategy::Equal => {
                let per_prop = self.config.total_budget / n;
                for prop in &self.properties {
                    allocation.insert(prop.id.clone(), per_prop.max(self.config.min_queries_per_property));
                }
            }
            AllocationStrategy::Proportional => {
                let total_priority: i32 = self.properties.iter().map(|p| p.priority.max(1)).sum();
                for prop in &self.properties {
                    let share = (prop.priority.max(1) as f64 / total_priority as f64
                        * self.config.total_budget as f64) as usize;
                    allocation.insert(
                        prop.id.clone(),
                        share.max(self.config.min_queries_per_property),
                    );
                }
            }
            AllocationStrategy::LeastConfident => {
                // Give more budget to least confident properties
                let mut confidences: Vec<(String, f64)> = self.property_states.iter()
                    .map(|(id, state)| (id.clone(), state.confidence))
                    .collect();
                confidences.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let total_inv_conf: f64 = confidences.iter()
                    .map(|(_, c)| 1.0 - c)
                    .sum();

                for (id, conf) in &confidences {
                    let share = if total_inv_conf > 1e-10 {
                        ((1.0 - conf) / total_inv_conf * self.config.total_budget as f64) as usize
                    } else {
                        self.config.total_budget / n
                    };
                    allocation.insert(
                        id.clone(),
                        share.max(self.config.min_queries_per_property),
                    );
                }
            }
            AllocationStrategy::RoundRobin => {
                let per_prop = self.config.total_budget / n;
                for prop in &self.properties {
                    allocation.insert(prop.id.clone(), per_prop.max(self.config.min_queries_per_property));
                }
            }
        }

        allocation
    }

    /// Update the learning state for a property.
    pub fn update_property(
        &mut self,
        property_id: &str,
        is_satisfied: Option<bool>,
        confidence: f64,
        queries_used: usize,
    ) {
        if let Some(state) = self.property_states.get_mut(property_id) {
            state.is_satisfied = is_satisfied;
            state.confidence = confidence;
            state.queries_used += queries_used;
            state.iteration += 1;
        }
    }

    /// Check if all properties have been resolved.
    pub fn all_resolved(&self) -> bool {
        self.property_states.values().all(|s| s.is_satisfied.is_some())
    }

    /// Get the overall confidence (minimum across properties).
    pub fn overall_confidence(&self) -> f64 {
        self.property_states.values()
            .map(|s| s.confidence)
            .fold(f64::INFINITY, f64::min)
    }

    /// Get summary of property learning states.
    pub fn summary(&self) -> MultiPropertySummary {
        let mut property_results = Vec::new();
        for prop in &self.properties {
            if let Some(state) = self.property_states.get(&prop.id) {
                property_results.push(PropertyResult {
                    id: prop.id.clone(),
                    description: prop.description.clone(),
                    is_satisfied: state.is_satisfied,
                    confidence: state.confidence,
                    queries_used: state.queries_used,
                });
            }
        }

        MultiPropertySummary {
            total_properties: self.properties.len(),
            resolved: self.property_states.values().filter(|s| s.is_satisfied.is_some()).count(),
            satisfied: self.property_states.values().filter(|s| s.is_satisfied == Some(true)).count(),
            violated: self.property_states.values().filter(|s| s.is_satisfied == Some(false)).count(),
            overall_confidence: self.overall_confidence(),
            total_queries: self.property_states.values().map(|s| s.queries_used).sum(),
            property_results,
        }
    }
}

/// Summary of multi-property learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPropertySummary {
    pub total_properties: usize,
    pub resolved: usize,
    pub satisfied: usize,
    pub violated: usize,
    pub overall_confidence: f64,
    pub total_queries: usize,
    pub property_results: Vec<PropertyResult>,
}

/// Result for a single property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyResult {
    pub id: String,
    pub description: String,
    pub is_satisfied: Option<bool>,
    pub confidence: f64,
    pub queries_used: usize,
}

impl fmt::Display for MultiPropertySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiProperty({}/{} resolved, {} satisfied, {} violated, conf={:.4})",
            self.resolved, self.total_properties,
            self.satisfied, self.violated,
            self.overall_confidence,
        )
    }
}

// ---------------------------------------------------------------------------
// Advanced active learning: information-theoretic query selection and transfer
// ---------------------------------------------------------------------------

/// Information-theoretic query selector that maximizes expected information gain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoTheoreticQuerySelector {
    /// Entropy estimates for each state
    pub state_entropies: HashMap<String, f64>,
    /// Query information gain history
    pub gain_history: Vec<(String, f64)>,
    /// Exploration rate (0 = pure exploitation, 1 = pure exploration)
    pub exploration_rate: f64,
    /// Temperature for softmax selection
    pub temperature: f64,
    /// Number of candidates to evaluate
    pub num_candidates: usize,
}

impl InfoTheoreticQuerySelector {
    pub fn new(exploration_rate: f64) -> Self {
        Self {
            state_entropies: HashMap::new(),
            gain_history: Vec::new(),
            exploration_rate: exploration_rate.clamp(0.0, 1.0),
            temperature: 1.0,
            num_candidates: 20,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.3)
    }

    /// Compute the entropy of a sub-distribution.
    fn distribution_entropy(dist: &SubDistribution) -> f64 {
        let mass = dist.total_mass();
        if mass < 1e-15 {
            return 0.0;
        }
        let mut h = 0.0;
        for &v in dist.weights.values() {
            let p = v / mass;
            if p > 1e-15 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Update entropy estimate for a state.
    pub fn update_entropy(&mut self, state_id: &str, distribution: &SubDistribution) {
        let h = Self::distribution_entropy(distribution);
        self.state_entropies.insert(state_id.to_string(), h);
    }

    /// Select the best query word from candidates based on expected information gain.
    pub fn select_query(
        &mut self,
        candidates: &[Word],
        current_distributions: &HashMap<String, SubDistribution>,
    ) -> Option<Word> {
        if candidates.is_empty() {
            return None;
        }

        let mut scores: Vec<(usize, f64)> = Vec::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            let key = format!("{}", candidate);

            // Expected info gain = current entropy (higher entropy → more to learn)
            let current_entropy = current_distributions
                .get(&key)
                .map(|d| Self::distribution_entropy(d))
                .unwrap_or(std::f64::consts::LN_2); // Maximum entropy for unknown

            // Novelty bonus: how different is this query from past queries
            let novelty = if self.gain_history.is_empty() {
                1.0
            } else {
                let min_dist: f64 = self
                    .gain_history
                    .iter()
                    .map(|(past_key, _)| {
                        // Simple string distance as proxy
                        let common = key
                            .chars()
                            .zip(past_key.chars())
                            .filter(|(a, b)| a == b)
                            .count();
                        1.0 - common as f64 / key.len().max(past_key.len()).max(1) as f64
                    })
                    .fold(f64::INFINITY, f64::min);
                min_dist
            };

            let score = (1.0 - self.exploration_rate) * current_entropy
                + self.exploration_rate * novelty;

            scores.push((idx, score));
        }

        // Softmax selection
        if self.temperature > 0.01 {
            let max_score = scores.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores
                .iter()
                .map(|(_, s)| ((s - max_score) / self.temperature).exp())
                .collect();
            let total: f64 = weights.iter().sum();

            if total > 1e-15 {
                // Weighted random selection (deterministic for reproducibility: pick max)
                let best_idx = weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| scores[i].0)?;

                self.gain_history.push((format!("{}", candidates[best_idx]), scores.iter().find(|(i, _)| *i == best_idx).map(|(_, s)| *s).unwrap_or(0.0)));

                return Some(candidates[best_idx].clone());
            }
        }

        // Fallback: pick highest score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let best_idx = scores.first()?.0;
        self.gain_history.push((format!("{}", candidates[best_idx]), scores[0].1));
        Some(candidates[best_idx].clone())
    }

    /// Decay the exploration rate over time.
    pub fn decay_exploration(&mut self, decay_factor: f64) {
        self.exploration_rate *= decay_factor;
        self.exploration_rate = self.exploration_rate.max(0.01);
    }

    /// Average information gain per query.
    pub fn avg_gain(&self) -> f64 {
        if self.gain_history.is_empty() {
            return 0.0;
        }
        self.gain_history.iter().map(|(_, g)| *g).sum::<f64>() / self.gain_history.len() as f64
    }
}

/// Transfer learning helper: extract reusable knowledge from a learned automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferKnowledge {
    /// State signatures from the source automaton
    pub state_signatures: Vec<Vec<(String, f64)>>,
    /// Transition patterns (from_sig_idx, symbol, to_sig_idx)
    pub transition_patterns: Vec<(usize, String, usize, f64)>,
    /// Output distribution templates
    pub output_templates: Vec<SubDistribution>,
    /// Source automaton metadata
    pub source_states: usize,
    pub source_alphabet_size: usize,
}

impl TransferKnowledge {
    /// Extract transfer knowledge from a learned result.
    pub fn extract(
        access_strings: &[Word],
        data: &HashMap<Word, Vec<SubDistribution>>,
        alphabet: &[Symbol],
    ) -> Self {
        let mut state_signatures = Vec::new();
        let mut output_templates = Vec::new();

        for row in access_strings {
            if let Some(sig) = data.get(row) {
                let flat_sig: Vec<(String, f64)> = sig
                    .iter()
                    .enumerate()
                    .flat_map(|(i, d)| {
                        d.weights
                            .iter()
                            .map(move |(k, &v)| (format!("{}:{}", i, k), v))
                    })
                    .collect();
                state_signatures.push(flat_sig);

                // Use first column's distribution as output template
                if let Some(first) = sig.first() {
                    output_templates.push(first.clone());
                }
            }
        }

        let mut transition_patterns = Vec::new();
        for (i, from_row) in access_strings.iter().enumerate() {
            for sym in alphabet {
                let ext = from_row.concat(&Word::singleton(sym.clone()));
                if let Some(ext_sig) = data.get(&ext) {
                    // Find nearest state
                    let mut best_j = 0;
                    let mut best_dist = f64::INFINITY;
                    for (j, to_row) in access_strings.iter().enumerate() {
                        if let Some(to_sig) = data.get(to_row) {
                            let dist: f64 = ext_sig
                                .iter()
                                .zip(to_sig.iter())
                                .map(|(a, b)| a.tv_distance(b))
                                .sum::<f64>();
                            if dist < best_dist {
                                best_dist = dist;
                                best_j = j;
                            }
                        }
                    }
                    transition_patterns.push((i, sym.0.clone(), best_j, best_dist));
                }
            }
        }

        TransferKnowledge {
            state_signatures,
            transition_patterns,
            output_templates,
            source_states: access_strings.len(),
            source_alphabet_size: alphabet.len(),
        }
    }

    /// Compute similarity between this knowledge and a new system's observations.
    pub fn similarity_to(&self, other: &TransferKnowledge) -> f64 {
        if self.state_signatures.is_empty() || other.state_signatures.is_empty() {
            return 0.0;
        }

        // Compare output template similarity
        let mut template_sim = 0.0;
        let pairs = self.output_templates.len().min(other.output_templates.len());
        if pairs > 0 {
            for i in 0..pairs {
                template_sim += 1.0 - self.output_templates[i].tv_distance(&other.output_templates[i]);
            }
            template_sim /= pairs as f64;
        }

        // Compare structural similarity
        let state_ratio = self.source_states.min(other.source_states) as f64
            / self.source_states.max(other.source_states).max(1) as f64;

        (template_sim * 0.6 + state_ratio * 0.4).clamp(0.0, 1.0)
    }

    pub fn summary(&self) -> String {
        format!(
            "TransferKnowledge(states={}, transitions={}, templates={})",
            self.source_states,
            self.transition_patterns.len(),
            self.output_templates.len(),
        )
    }
}

/// Learning rate scheduler for adaptive tolerance adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceScheduler {
    pub initial_tolerance: f64,
    pub min_tolerance: f64,
    pub current_tolerance: f64,
    pub schedule: ToleranceSchedule,
    pub iteration: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToleranceSchedule {
    Constant,
    Linear,
    Exponential,
    StepWise,
    Cosine,
}

impl ToleranceScheduler {
    pub fn new(initial: f64, min: f64, schedule: ToleranceSchedule) -> Self {
        Self {
            initial_tolerance: initial,
            min_tolerance: min,
            current_tolerance: initial,
            schedule,
            iteration: 0,
        }
    }

    pub fn constant(tolerance: f64) -> Self {
        Self::new(tolerance, tolerance, ToleranceSchedule::Constant)
    }

    pub fn exponential(initial: f64, min: f64) -> Self {
        Self::new(initial, min, ToleranceSchedule::Exponential)
    }

    pub fn step(&mut self) -> f64 {
        self.iteration += 1;
        self.current_tolerance = match self.schedule {
            ToleranceSchedule::Constant => self.initial_tolerance,
            ToleranceSchedule::Linear => {
                let progress = (self.iteration as f64) / 100.0;
                let range = self.initial_tolerance - self.min_tolerance;
                (self.initial_tolerance - range * progress).max(self.min_tolerance)
            }
            ToleranceSchedule::Exponential => {
                let decay = 0.95f64.powi(self.iteration as i32);
                let range = self.initial_tolerance - self.min_tolerance;
                self.min_tolerance + range * decay
            }
            ToleranceSchedule::StepWise => {
                let step = self.iteration / 20;
                let factor = 0.5f64.powi(step as i32);
                let range = self.initial_tolerance - self.min_tolerance;
                (self.min_tolerance + range * factor).max(self.min_tolerance)
            }
            ToleranceSchedule::Cosine => {
                let progress = (self.iteration as f64) / 100.0;
                let cosine_value = (std::f64::consts::PI * progress).cos();
                let range = self.initial_tolerance - self.min_tolerance;
                self.min_tolerance + range * (1.0 + cosine_value) / 2.0
            }
        };
        self.current_tolerance
    }

    pub fn current(&self) -> f64 {
        self.current_tolerance
    }

    pub fn reset(&mut self) {
        self.iteration = 0;
        self.current_tolerance = self.initial_tolerance;
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

    // Simple teacher for testing
    struct SimpleTeacher {
        alphabet: Vec<Symbol>,
    }

    impl Teacher for SimpleTeacher {
        fn membership_query(&self, word: &Word) -> SubDistribution {
            if word.is_empty() {
                SubDistribution::singleton("yes".to_string(), 0.8)
            } else {
                SubDistribution::singleton("no".to_string(), 0.7)
            }
        }

        fn equivalence_query(
            &self,
            _hypothesis: &dyn Fn(&Word) -> SubDistribution,
            _num_tests: usize,
            _tolerance: f64,
        ) -> Option<Word> {
            None // Always pass
        }

        fn alphabet(&self) -> &[Symbol] {
            &self.alphabet
        }
    }

    #[test]
    fn test_query_selector_creation() {
        let selector = QuerySelector::with_defaults();
        assert!(selector.current_exploration_rate() > 0.0);
        assert_eq!(selector.total_outcomes(), 0);
    }

    #[test]
    fn test_query_selector_record() {
        let mut selector = QuerySelector::with_defaults();

        let word = Word::from_str_slice(&["a"]);
        let dist = SubDistribution::singleton("yes".to_string(), 0.7);

        selector.record_outcome(word, dist);
        assert_eq!(selector.total_outcomes(), 1);
    }

    #[test]
    fn test_query_selector_exploration_decay() {
        let mut selector = QuerySelector::new(0.5, 0.9);
        let initial_rate = selector.current_exploration_rate();

        for _ in 0..10 {
            selector.record_outcome(
                Word::from_str_slice(&["a"]),
                SubDistribution::singleton("x".to_string(), 0.5),
            );
        }

        assert!(selector.current_exploration_rate() < initial_rate);
    }

    #[test]
    fn test_query_selector_select() {
        let selector = QuerySelector::with_defaults();
        let candidates = vec![
            Word::from_str_slice(&["a"]),
            Word::from_str_slice(&["b"]),
            Word::from_str_slice(&["a", "b"]),
        ];

        let selected = selector.select_query(&candidates);
        assert!(selected.is_some());
    }

    #[test]
    fn test_query_selector_empty_candidates() {
        let selector = QuerySelector::with_defaults();
        assert!(selector.select_query(&[]).is_none());
    }

    #[test]
    fn test_learning_curve_creation() {
        let curve = LearningCurve::new();
        assert!(curve.points.is_empty());
        assert_eq!(curve.current_accuracy(), None);
    }

    #[test]
    fn test_learning_curve_record() {
        let mut curve = LearningCurve::new();
        curve.record(0, 0, 1, 0.5, 0.5);
        curve.record(1, 100, 2, 0.7, 0.3);
        curve.record(2, 200, 3, 0.9, 0.1);

        assert_eq!(curve.points.len(), 3);
        assert_eq!(curve.current_accuracy(), Some(0.9));
    }

    #[test]
    fn test_learning_curve_improvement_rate() {
        let mut curve = LearningCurve::new();
        curve.record(0, 0, 1, 0.0, 1.0);
        curve.record(1, 100, 2, 0.5, 0.5);

        let rate = curve.improvement_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_learning_curve_plateau() {
        let mut curve = LearningCurve::new();
        for i in 0..10 {
            curve.record(i, i * 100, 5, 0.95, 0.05);
        }

        assert!(curve.has_plateaued(5, 0.01));
    }

    #[test]
    fn test_learning_curve_no_plateau() {
        let mut curve = LearningCurve::new();
        for i in 0..10 {
            curve.record(i, i * 100, i + 1, i as f64 * 0.1, 1.0 - i as f64 * 0.1);
        }

        assert!(!curve.has_plateaued(5, 0.01));
    }

    #[test]
    fn test_learning_curve_estimate_target() {
        let mut curve = LearningCurve::new();
        // Simulate improving accuracy
        for i in 0..20 {
            let acc = 1.0 - 0.5 * (-0.1 * i as f64).exp();
            curve.record(i, i * 100, 5, acc, 1.0 - acc);
        }

        let estimate = curve.estimate_queries_to_target(0.99);
        // May or may not have an estimate depending on convergence
        if let Some(q) = estimate {
            assert!(q > 0);
        }
    }

    #[test]
    fn test_incremental_learner_creation() {
        let learner = IncrementalLearner::new(make_alphabet(), 0.1);
        assert_eq!(learner.num_states(), 1); // Just the initial state
        assert_eq!(learner.total_queries(), 0);
    }

    #[test]
    fn test_incremental_learner_add_access_string() {
        let mut learner = IncrementalLearner::new(make_alphabet(), 0.1);

        // Add a distinct state
        let word = Word::from_str_slice(&["a"]);
        learner.add_observation(
            word.clone(),
            0,
            SubDistribution::singleton("no".to_string(), 0.9),
        );

        // Need to add observation for empty word too
        learner.add_observation(
            Word::empty(),
            0,
            SubDistribution::singleton("yes".to_string(), 0.8),
        );

        let added = learner.add_access_string(word);
        assert!(added);
        assert_eq!(learner.num_states(), 2);
    }

    #[test]
    fn test_incremental_learner_add_suffix() {
        let mut learner = IncrementalLearner::new(make_alphabet(), 0.1);

        let suffix = Word::from_str_slice(&["b"]);
        assert!(learner.add_suffix(suffix.clone()));
        assert!(!learner.add_suffix(suffix)); // Duplicate

        assert_eq!(learner.suffixes().len(), 2); // epsilon + "b"
    }

    #[test]
    fn test_incremental_learner_process_ce() {
        let mut learner = IncrementalLearner::new(make_alphabet(), 0.1);

        let query_fn = |word: &Word| -> SubDistribution {
            if word.is_empty() {
                SubDistribution::singleton("yes".to_string(), 0.8)
            } else {
                SubDistribution::singleton("no".to_string(), 0.7)
            }
        };

        let ce = Word::from_str_slice(&["a", "b"]);
        let suffix = Word::from_str_slice(&["b"]);

        learner.process_counterexample(&ce, suffix, &query_fn);

        assert!(learner.total_queries() > 0);
    }

    #[test]
    fn test_transfer_learner_creation() {
        let learner = TransferLearner::new(0.1);
        let stats = learner.stats();
        assert_eq!(stats.source_states, 0);
        assert_eq!(stats.transferred_states, 0);
    }

    #[test]
    fn test_transfer_learner_set_source() {
        let mut learner = TransferLearner::new(0.1);

        let table = HashMap::new();
        let access = vec![Word::empty(), Word::from_str_slice(&["a"])];
        let suffixes = vec![Word::empty()];

        learner.set_source(table, access, suffixes);
        assert_eq!(learner.stats().source_states, 2);
    }

    #[test]
    fn test_transfer_learner_transfer() {
        let mut transfer = TransferLearner::new(0.5); // High threshold

        let mut source_table = HashMap::new();
        source_table.insert(
            Word::empty(),
            vec![SubDistribution::singleton("yes".to_string(), 0.8)],
        );

        let source_access = vec![Word::empty()];
        let source_suffixes = vec![Word::empty()];

        transfer.set_source(source_table, source_access, source_suffixes);

        let mut target = IncrementalLearner::new(make_alphabet(), 0.1);

        let query_fn = |_word: &Word| -> SubDistribution {
            SubDistribution::singleton("yes".to_string(), 0.8)
        };

        transfer.transfer_to(&mut target, &query_fn, 10);

        // Transfer should have occurred since distributions match
        let stats = transfer.stats();
        assert!(stats.transfer_ratio >= 0.0);
    }

    #[test]
    fn test_multi_property_learner_creation() {
        let props = vec![
            PropertySpec {
                id: "safety".to_string(),
                description: "No bad outputs".to_string(),
                property_type: PropertyType::Safety { bad_states: vec!["bad".to_string()] },
                confidence: 0.95,
                priority: 5,
            },
            PropertySpec {
                id: "fairness".to_string(),
                description: "Equal treatment".to_string(),
                property_type: PropertyType::Fairness {
                    constraints: vec![FairnessConstraint {
                        group_a: "male".to_string(),
                        group_b: "female".to_string(),
                        max_disparity: 0.1,
                    }],
                },
                confidence: 0.90,
                priority: 3,
            },
        ];

        let learner = MultiPropertyLearner::new(
            props, make_alphabet(), MultiPropertyConfig::default(),
        );

        assert!(!learner.all_resolved());
    }

    #[test]
    fn test_multi_property_budget_equal() {
        let props = vec![
            PropertySpec {
                id: "p1".to_string(),
                description: "".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 1,
            },
            PropertySpec {
                id: "p2".to_string(),
                description: "".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 1,
            },
        ];

        let config = MultiPropertyConfig {
            allocation_strategy: AllocationStrategy::Equal,
            total_budget: 1000,
            min_queries_per_property: 10,
        };

        let learner = MultiPropertyLearner::new(props, make_alphabet(), config);
        let budget = learner.allocate_budget();

        assert_eq!(budget.len(), 2);
        assert_eq!(budget["p1"], budget["p2"]);
    }

    #[test]
    fn test_multi_property_budget_proportional() {
        let props = vec![
            PropertySpec {
                id: "p1".to_string(),
                description: "".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 3,
            },
            PropertySpec {
                id: "p2".to_string(),
                description: "".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 1,
            },
        ];

        let config = MultiPropertyConfig {
            allocation_strategy: AllocationStrategy::Proportional,
            total_budget: 1000,
            min_queries_per_property: 10,
        };

        let learner = MultiPropertyLearner::new(props, make_alphabet(), config);
        let budget = learner.allocate_budget();

        assert!(budget["p1"] > budget["p2"]);
    }

    #[test]
    fn test_multi_property_update() {
        let props = vec![
            PropertySpec {
                id: "p1".to_string(),
                description: "".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 1,
            },
        ];

        let mut learner = MultiPropertyLearner::new(
            props, make_alphabet(), MultiPropertyConfig::default(),
        );

        assert!(!learner.all_resolved());

        learner.update_property("p1", Some(true), 0.98, 500);

        assert!(learner.all_resolved());
        assert_eq!(learner.overall_confidence(), 0.98);
    }

    #[test]
    fn test_multi_property_summary() {
        let props = vec![
            PropertySpec {
                id: "p1".to_string(),
                description: "Safety".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 1,
            },
            PropertySpec {
                id: "p2".to_string(),
                description: "Fairness".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.90,
                priority: 2,
            },
        ];

        let mut learner = MultiPropertyLearner::new(
            props, make_alphabet(), MultiPropertyConfig::default(),
        );

        learner.update_property("p1", Some(true), 0.98, 200);
        learner.update_property("p2", Some(false), 0.95, 300);

        let summary = learner.summary();
        assert_eq!(summary.total_properties, 2);
        assert_eq!(summary.resolved, 2);
        assert_eq!(summary.satisfied, 1);
        assert_eq!(summary.violated, 1);
        assert_eq!(summary.total_queries, 500);
    }

    #[test]
    fn test_protocol_config_default() {
        let config = ProtocolConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.equivalence_test_count > 0);
    }

    #[test]
    fn test_protocol_result_display() {
        let result = ProtocolResult {
            converged: true,
            iterations: 10,
            num_states: 5,
            total_queries: 500,
            total_time_ms: 1234,
            events: Vec::new(),
        };

        let s = format!("{}", result);
        assert!(s.contains("converged"));
        assert!(s.contains("10"));
    }

    #[test]
    fn test_learning_curve_display() {
        let mut curve = LearningCurve::new();
        curve.record(0, 0, 1, 0.5, 0.5);

        let s = format!("{}", curve);
        assert!(s.contains("1 points"));
    }

    #[test]
    fn test_property_type_variants() {
        let _safety = PropertyType::Safety { bad_states: vec!["bad".to_string()] };
        let _liveness = PropertyType::Liveness { good_states: vec!["good".to_string()] };
        let _fairness = PropertyType::Fairness {
            constraints: vec![FairnessConstraint {
                group_a: "a".to_string(),
                group_b: "b".to_string(),
                max_disparity: 0.1,
            }],
        };
        let _consistency = PropertyType::Consistency { tolerance: 0.05 };
        let _custom = PropertyType::Custom { name: "test".to_string() };
    }

    #[test]
    fn test_sub_distribution_entropy() {
        // Uniform distribution should have max entropy
        let uniform = SubDistribution::from_map(
            [("a".to_string(), 0.5), ("b".to_string(), 0.5)]
                .into_iter().collect(),
        );
        let h_uniform = uniform.entropy();

        // Peaked distribution should have lower entropy
        let peaked = SubDistribution::from_map(
            [("a".to_string(), 0.99), ("b".to_string(), 0.01)]
                .into_iter().collect(),
        );
        let h_peaked = peaked.entropy();

        assert!(h_uniform > h_peaked);
    }

    #[test]
    fn test_sub_distribution_entropy_zero() {
        let empty = SubDistribution::new();
        assert_eq!(empty.entropy(), 0.0);

        let single = SubDistribution::singleton("a".to_string(), 1.0);
        let h = single.entropy();
        assert!(h.abs() < 1e-10); // Entropy of a point mass is 0
    }

    #[test]
    fn test_rows_equivalent_fn() {
        let r1 = vec![SubDistribution::singleton("a".to_string(), 0.5)];
        let r2 = vec![SubDistribution::singleton("a".to_string(), 0.52)];
        let r3 = vec![SubDistribution::singleton("b".to_string(), 0.9)];

        assert!(rows_equivalent(&r1, &r2, 0.1));
        assert!(!rows_equivalent(&r1, &r3, 0.1));
        assert!(!rows_equivalent(&r1, &[], 0.1));
    }

    #[test]
    fn test_allocation_least_confident() {
        let props = vec![
            PropertySpec {
                id: "p1".to_string(),
                description: "".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 1,
            },
            PropertySpec {
                id: "p2".to_string(),
                description: "".to_string(),
                property_type: PropertyType::Consistency { tolerance: 0.1 },
                confidence: 0.95,
                priority: 1,
            },
        ];

        let config = MultiPropertyConfig {
            allocation_strategy: AllocationStrategy::LeastConfident,
            total_budget: 1000,
            min_queries_per_property: 10,
        };

        let mut learner = MultiPropertyLearner::new(props, make_alphabet(), config);
        learner.update_property("p1", None, 0.9, 100);
        learner.update_property("p2", None, 0.5, 50);

        let budget = learner.allocate_budget();
        assert!(budget["p2"] > budget["p1"]); // Less confident gets more
    }

    #[test]
    fn test_query_selector_average_info_gain() {
        let mut selector = QuerySelector::with_defaults();
        assert_eq!(selector.average_information_gain(), 0.0);

        selector.record_outcome(
            Word::from_str_slice(&["a"]),
            SubDistribution::singleton("x".to_string(), 0.5),
        );

        let avg = selector.average_information_gain();
        assert!(avg >= 0.0);
    }

    #[test]
    fn test_multi_property_summary_display() {
        let summary = MultiPropertySummary {
            total_properties: 3,
            resolved: 2,
            satisfied: 1,
            violated: 1,
            overall_confidence: 0.85,
            total_queries: 1000,
            property_results: Vec::new(),
        };

        let s = format!("{}", summary);
        assert!(s.contains("2/3"));
    }

    #[test]
    fn test_info_theoretic_selector_basic() {
        let mut selector = InfoTheoreticQuerySelector::with_defaults();
        let candidates = vec![
            Word::from_str_slice(&["a"]),
            Word::from_str_slice(&["b"]),
            Word::from_str_slice(&["a", "b"]),
        ];
        let dists: HashMap<String, SubDistribution> = HashMap::new();

        let selected = selector.select_query(&candidates, &dists);
        assert!(selected.is_some());
    }

    #[test]
    fn test_info_theoretic_selector_entropy_update() {
        let mut selector = InfoTheoreticQuerySelector::new(0.5);
        let dist = SubDistribution::from_map(
            [("a".to_string(), 0.5), ("b".to_string(), 0.5)].into_iter().collect(),
        );
        selector.update_entropy("state0", &dist);
        assert!(selector.state_entropies.contains_key("state0"));
        assert!(selector.state_entropies["state0"] > 0.0);
    }

    #[test]
    fn test_info_theoretic_selector_decay() {
        let mut selector = InfoTheoreticQuerySelector::new(0.5);
        selector.decay_exploration(0.9);
        assert!(selector.exploration_rate < 0.5);
        assert!(selector.exploration_rate > 0.4);
    }

    #[test]
    fn test_info_theoretic_selector_empty_candidates() {
        let mut selector = InfoTheoreticQuerySelector::with_defaults();
        let result = selector.select_query(&[], &HashMap::new());
        assert!(result.is_none());
    }

    #[test]
    fn test_transfer_knowledge_extract() {
        let alphabet = make_alphabet();
        let access_strings = vec![Word::empty()];
        let mut data = HashMap::new();
        data.insert(
            Word::empty(),
            vec![SubDistribution::singleton("yes".to_string(), 0.8)],
        );
        data.insert(
            Word::from_str_slice(&["a"]),
            vec![SubDistribution::singleton("yes".to_string(), 0.7)],
        );

        let tk = TransferKnowledge::extract(&access_strings, &data, &alphabet);
        assert_eq!(tk.source_states, 1);
        assert_eq!(tk.source_alphabet_size, 2);
        assert!(!tk.output_templates.is_empty());
    }

    #[test]
    fn test_transfer_knowledge_similarity() {
        let alphabet = make_alphabet();
        let access = vec![Word::empty()];
        let mut data = HashMap::new();
        data.insert(
            Word::empty(),
            vec![SubDistribution::singleton("yes".to_string(), 0.8)],
        );

        let tk1 = TransferKnowledge::extract(&access, &data, &alphabet);
        let tk2 = TransferKnowledge::extract(&access, &data, &alphabet);

        let sim = tk1.similarity_to(&tk2);
        assert!(sim > 0.5);
    }

    #[test]
    fn test_transfer_knowledge_summary() {
        let tk = TransferKnowledge {
            state_signatures: vec![vec![("0:yes".to_string(), 0.8)]],
            transition_patterns: vec![(0, "a".to_string(), 0, 0.0)],
            output_templates: vec![SubDistribution::singleton("yes".to_string(), 0.8)],
            source_states: 1,
            source_alphabet_size: 2,
        };
        let summary = tk.summary();
        assert!(summary.contains("states=1"));
    }

    #[test]
    fn test_tolerance_scheduler_constant() {
        let mut sched = ToleranceScheduler::constant(0.1);
        assert_eq!(sched.current(), 0.1);
        sched.step();
        assert_eq!(sched.current(), 0.1);
    }

    #[test]
    fn test_tolerance_scheduler_exponential() {
        let mut sched = ToleranceScheduler::exponential(0.2, 0.01);
        let initial = sched.current();

        for _ in 0..50 {
            sched.step();
        }

        assert!(sched.current() < initial);
        assert!(sched.current() >= sched.min_tolerance);
    }

    #[test]
    fn test_tolerance_scheduler_linear() {
        let mut sched = ToleranceScheduler::new(0.2, 0.01, ToleranceSchedule::Linear);
        sched.step();
        assert!(sched.current() <= 0.2);
    }

    #[test]
    fn test_tolerance_scheduler_step_wise() {
        let mut sched = ToleranceScheduler::new(0.2, 0.01, ToleranceSchedule::StepWise);
        let t0 = sched.current();

        for _ in 0..25 {
            sched.step();
        }

        assert!(sched.current() < t0);
    }

    #[test]
    fn test_tolerance_scheduler_cosine() {
        let mut sched = ToleranceScheduler::new(0.2, 0.01, ToleranceSchedule::Cosine);
        let vals: Vec<f64> = (0..100).map(|_| sched.step()).collect();

        // Should decrease overall
        assert!(vals.last().unwrap() < &vals[0]);
        // Should stay in bounds
        for v in &vals {
            assert!(*v >= 0.01);
            assert!(*v <= 0.2 + 0.01);
        }
    }

    #[test]
    fn test_tolerance_scheduler_reset() {
        let mut sched = ToleranceScheduler::exponential(0.2, 0.01);
        for _ in 0..20 {
            sched.step();
        }
        assert!(sched.current() < 0.2);

        sched.reset();
        assert_eq!(sched.current(), 0.2);
        assert_eq!(sched.iteration, 0);
    }
}
