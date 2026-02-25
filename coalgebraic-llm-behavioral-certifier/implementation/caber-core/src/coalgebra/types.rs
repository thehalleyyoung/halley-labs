//! Core type definitions shared across the coalgebra module.
//!
//! Provides state identifiers, transition labels, action/observation spaces,
//! weighted transition types, and configuration for coalgebra parameters.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use ordered_float::OrderedFloat;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use indexmap::IndexMap;

// ---------------------------------------------------------------------------
// State identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a state in a coalgebra.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StateId(pub String);

impl StateId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn fresh() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn indexed(prefix: &str, idx: usize) -> Self {
        Self(format!("{}_{}", prefix, idx))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for StateId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for StateId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<usize> for StateId {
    fn from(idx: usize) -> Self {
        Self(format!("s{}", idx))
    }
}

/// Numeric state index used for matrix-based computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StateIndex(pub usize);

impl StateIndex {
    pub fn new(idx: usize) -> Self {
        Self(idx)
    }
}

impl fmt::Display for StateIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "idx:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Transition labels and symbols
// ---------------------------------------------------------------------------

/// A symbol in the input alphabet. For LLMs, this is typically a token or prompt fragment.
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

    pub fn length(&self) -> usize {
        self.0.len()
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

/// A word (sequence of symbols) of bounded length.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Word {
    pub symbols: Vec<Symbol>,
}

impl Word {
    pub fn empty() -> Self {
        Self { symbols: Vec::new() }
    }

    pub fn singleton(sym: Symbol) -> Self {
        Self { symbols: vec![sym] }
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

    pub fn suffix(&self, n: usize) -> Word {
        let start = self.symbols.len().saturating_sub(n);
        Word {
            symbols: self.symbols[start..].to_vec(),
        }
    }

    pub fn is_prefix_of(&self, other: &Word) -> bool {
        if self.len() > other.len() {
            return false;
        }
        self.symbols == other.symbols[..self.len()]
    }

    /// Enumerate all words up to length `max_len` over the given alphabet.
    pub fn enumerate_up_to(alphabet: &[Symbol], max_len: usize) -> Vec<Word> {
        let mut result = vec![Word::empty()];
        let mut current_level = vec![Word::empty()];
        for _ in 0..max_len {
            let mut next_level = Vec::new();
            for w in &current_level {
                for sym in alphabet {
                    let extended = w.concat(&Word::singleton(sym.clone()));
                    next_level.push(extended);
                }
            }
            result.extend(next_level.iter().cloned());
            current_level = next_level;
        }
        result
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

// ---------------------------------------------------------------------------
// Output alphabet
// ---------------------------------------------------------------------------

/// An output symbol from an LLM. For token-level analysis this is a token;
/// for semantic-level analysis this might be a cluster label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OutputSymbol {
    pub value: String,
    pub cluster_id: Option<ClusterId>,
}

impl OutputSymbol {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            cluster_id: None,
        }
    }

    pub fn with_cluster(value: impl Into<String>, cluster: ClusterId) -> Self {
        Self {
            value: value.into(),
            cluster_id: Some(cluster),
        }
    }
}

impl fmt::Display for OutputSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cluster_id {
            Some(c) => write!(f, "{}[c:{}]", self.value, c),
            None => write!(f, "{}", self.value),
        }
    }
}

/// Cluster identifier for semantic grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClusterId(pub usize);

impl fmt::Display for ClusterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Action and observation spaces
// ---------------------------------------------------------------------------

/// Specification of the input action space for a coalgebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpace {
    pub alphabet: Vec<Symbol>,
    pub max_word_length: usize,
    pub total_words: usize,
}

impl ActionSpace {
    pub fn new(alphabet: Vec<Symbol>, max_word_length: usize) -> Self {
        let mut total = 0usize;
        let n = alphabet.len();
        let mut power = 1usize;
        for _ in 0..=max_word_length {
            total = total.saturating_add(power);
            power = power.saturating_mul(n);
        }
        Self {
            alphabet,
            max_word_length,
            total_words: total,
        }
    }

    pub fn alphabet_size(&self) -> usize {
        self.alphabet.len()
    }

    pub fn enumerate_words(&self) -> Vec<Word> {
        Word::enumerate_up_to(&self.alphabet, self.max_word_length)
    }

    pub fn contains_word(&self, w: &Word) -> bool {
        if w.len() > self.max_word_length {
            return false;
        }
        let alpha_set: HashSet<&Symbol> = self.alphabet.iter().collect();
        w.symbols.iter().all(|s| alpha_set.contains(s))
    }
}

/// Specification of the output observation space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpace {
    pub output_symbols: Vec<OutputSymbol>,
    pub max_output_length: usize,
    pub num_clusters: Option<usize>,
}

impl ObservationSpace {
    pub fn new(output_symbols: Vec<OutputSymbol>, max_output_length: usize) -> Self {
        Self {
            output_symbols,
            max_output_length,
            num_clusters: None,
        }
    }

    pub fn with_clusters(mut self, n_clusters: usize) -> Self {
        self.num_clusters = Some(n_clusters);
        self
    }

    pub fn output_size(&self) -> usize {
        self.output_symbols.len()
    }
}

// ---------------------------------------------------------------------------
// Weighted transitions
// ---------------------------------------------------------------------------

/// A single weighted transition from one state to another under a given label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedTransition<W: Clone> {
    pub source: StateId,
    pub target: StateId,
    pub label: TransitionLabel,
    pub weight: W,
}

impl<W: Clone + fmt::Display> fmt::Display for WeightedTransition<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} --[{}, w={}]--> {}",
            self.source, self.label, self.weight, self.target
        )
    }
}

/// Label on a transition, combining input and output.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TransitionLabel {
    pub input: Word,
    pub output: Option<OutputSymbol>,
}

impl TransitionLabel {
    pub fn new(input: Word, output: Option<OutputSymbol>) -> Self {
        Self { input, output }
    }

    pub fn input_only(input: Word) -> Self {
        Self {
            input,
            output: None,
        }
    }

    pub fn with_output(input: Word, output: OutputSymbol) -> Self {
        Self {
            input,
            output: Some(output),
        }
    }
}

impl fmt::Display for TransitionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.output {
            Some(o) => write!(f, "{}/{}", self.input, o),
            None => write!(f, "{}", self.input),
        }
    }
}

/// A probability-weighted transition (the most common kind in CABER).
pub type ProbabilisticTransition = WeightedTransition<OrderedFloat<f64>>;

/// Full transition table: for each state, maps input words to distributions over (output, next_state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionTable {
    pub entries: HashMap<StateId, HashMap<Word, Vec<TransitionEntry>>>,
}

/// Single entry in a transition table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionEntry {
    pub output: OutputSymbol,
    pub next_state: StateId,
    pub probability: f64,
}

impl TransitionTable {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn add_entry(
        &mut self,
        source: StateId,
        input: Word,
        output: OutputSymbol,
        next_state: StateId,
        probability: f64,
    ) {
        let state_map = self.entries.entry(source).or_insert_with(HashMap::new);
        let entries = state_map.entry(input).or_insert_with(Vec::new);
        entries.push(TransitionEntry {
            output,
            next_state,
            probability,
        });
    }

    pub fn get_transitions(&self, source: &StateId, input: &Word) -> Option<&[TransitionEntry]> {
        self.entries
            .get(source)
            .and_then(|m| m.get(input))
            .map(|v| v.as_slice())
    }

    pub fn states(&self) -> HashSet<StateId> {
        let mut states = HashSet::new();
        for (src, map) in &self.entries {
            states.insert(src.clone());
            for entries in map.values() {
                for e in entries {
                    states.insert(e.next_state.clone());
                }
            }
        }
        states
    }

    pub fn num_states(&self) -> usize {
        self.states().len()
    }

    pub fn num_transitions(&self) -> usize {
        self.entries
            .values()
            .flat_map(|m| m.values())
            .map(|v| v.len())
            .sum()
    }

    /// Validate that all outgoing distributions are proper sub-distributions (sum ≤ 1).
    pub fn validate_subdistributions(&self, tolerance: f64) -> Result<(), TransitionTableError> {
        for (state, map) in &self.entries {
            for (input, entries) in map {
                let total: f64 = entries.iter().map(|e| e.probability).sum();
                if total > 1.0 + tolerance {
                    return Err(TransitionTableError::InvalidDistribution {
                        state: state.clone(),
                        input: input.clone(),
                        total,
                    });
                }
                for e in entries {
                    if e.probability < 0.0 {
                        return Err(TransitionTableError::NegativeProbability {
                            state: state.clone(),
                            input: input.clone(),
                            probability: e.probability,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Normalize all outgoing distributions to sum to 1.
    pub fn normalize(&mut self) {
        for map in self.entries.values_mut() {
            for entries in map.values_mut() {
                let total: f64 = entries.iter().map(|e| e.probability).sum();
                if total > 0.0 {
                    for e in entries.iter_mut() {
                        e.probability /= total;
                    }
                }
            }
        }
    }
}

impl Default for TransitionTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from transition table validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TransitionTableError {
    #[error("Invalid distribution at state {state}, input {input}: total = {total}")]
    InvalidDistribution {
        state: StateId,
        input: Word,
        total: f64,
    },
    #[error("Negative probability at state {state}, input {input}: p = {probability}")]
    NegativeProbability {
        state: StateId,
        input: Word,
        probability: f64,
    },
}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Configuration for coalgebra construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalgebraConfig {
    pub max_input_length: usize,
    pub max_output_length: usize,
    pub num_clusters: usize,
    pub epsilon: f64,
    pub convergence_threshold: f64,
    pub max_iterations: usize,
    pub seed: Option<u64>,
}

impl Default for CoalgebraConfig {
    fn default() -> Self {
        Self {
            max_input_length: 5,
            max_output_length: 10,
            num_clusters: 50,
            epsilon: 0.01,
            convergence_threshold: 1e-6,
            max_iterations: 1000,
            seed: None,
        }
    }
}

/// Configuration for bisimulation computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisimulationConfig {
    pub method: BisimulationMethod,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub early_termination: bool,
    pub witness_generation: bool,
}

impl Default for BisimulationConfig {
    fn default() -> Self {
        Self {
            method: BisimulationMethod::Approximate,
            tolerance: 1e-6,
            max_iterations: 1000,
            early_termination: true,
            witness_generation: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BisimulationMethod {
    Exact,
    Approximate,
    OnTheFly,
    UpTo,
}

/// Configuration for abstraction refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionConfig {
    pub initial_clusters: usize,
    pub max_refinement_steps: usize,
    pub split_threshold: f64,
    pub merge_threshold: f64,
    pub ks_test_alpha: f64,
    pub min_samples_per_cluster: usize,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            initial_clusters: 10,
            max_refinement_steps: 50,
            split_threshold: 0.1,
            merge_threshold: 0.01,
            ks_test_alpha: 0.05,
            min_samples_per_cluster: 5,
        }
    }
}

/// Configuration for bandwidth estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthConfig {
    pub covering_number_method: CoveringNumberMethod,
    pub num_samples: usize,
    pub confidence: f64,
    pub dimension_reduction: bool,
    pub max_dimension: usize,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            covering_number_method: CoveringNumberMethod::RandomSampling,
            num_samples: 1000,
            confidence: 0.95,
            dimension_reduction: true,
            max_dimension: 100,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoveringNumberMethod {
    RandomSampling,
    GreedyFarthestPoint,
    KMeansApprox,
}

// ---------------------------------------------------------------------------
// Metric-related types
// ---------------------------------------------------------------------------

/// A metric on a set of elements.
#[derive(Debug, Clone)]
pub struct FiniteMetric<T: Eq + Hash + Clone> {
    distances: HashMap<(T, T), f64>,
}

impl<T: Eq + Hash + Clone + Ord> FiniteMetric<T> {
    pub fn new() -> Self {
        Self {
            distances: HashMap::new(),
        }
    }

    /// Set distance between two points. Automatically sets d(y,x) = d(x,y) and d(x,x) = 0.
    pub fn set_distance(&mut self, x: T, y: T, d: f64) {
        if x == y {
            self.distances.insert((x.clone(), y), 0.0);
            return;
        }
        let (a, b) = if x < y {
            (x, y)
        } else {
            (y, x)
        };
        self.distances.insert((a.clone(), b.clone()), d);
        self.distances.insert((b, a), d);
    }

    pub fn distance(&self, x: &T, y: &T) -> f64 {
        if x == y {
            return 0.0;
        }
        self.distances
            .get(&(x.clone(), y.clone()))
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    /// Check triangle inequality for all triples.
    pub fn validate_triangle_inequality(&self, points: &[T]) -> bool {
        for i in 0..points.len() {
            for j in 0..points.len() {
                for k in 0..points.len() {
                    let dij = self.distance(&points[i], &points[j]);
                    let djk = self.distance(&points[j], &points[k]);
                    let dik = self.distance(&points[i], &points[k]);
                    if dik > dij + djk + 1e-10 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check symmetry for all pairs.
    pub fn validate_symmetry(&self, points: &[T]) -> bool {
        for i in 0..points.len() {
            for j in 0..points.len() {
                let dij = self.distance(&points[i], &points[j]);
                let dji = self.distance(&points[j], &points[i]);
                if (dij - dji).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Check identity of indiscernibles.
    pub fn validate_identity(&self, points: &[T]) -> bool {
        for p in points {
            if self.distance(p, p).abs() > 1e-10 {
                return false;
            }
        }
        true
    }

    /// Compute the diameter (maximum distance).
    pub fn diameter(&self, points: &[T]) -> f64 {
        let mut max_d = 0.0f64;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let d = self.distance(&points[i], &points[j]);
                if d > max_d {
                    max_d = d;
                }
            }
        }
        max_d
    }
}

impl<T: Eq + Hash + Clone + Ord> Default for FiniteMetric<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A pseudometric (allows d(x,y) = 0 for x ≠ y).
#[derive(Debug, Clone)]
pub struct Pseudometric<T: Eq + Hash + Clone> {
    inner: FiniteMetric<T>,
}

impl<T: Eq + Hash + Clone + Ord> Pseudometric<T> {
    pub fn new() -> Self {
        Self {
            inner: FiniteMetric::new(),
        }
    }

    pub fn set_distance(&mut self, x: T, y: T, d: f64) {
        self.inner.set_distance(x, y, d);
    }

    pub fn distance(&self, x: &T, y: &T) -> f64 {
        self.inner.distance(x, y)
    }

    /// Convert to a metric by identifying points at distance 0.
    pub fn quotient(&self, points: &[T]) -> (Vec<Vec<T>>, FiniteMetric<usize>) {
        let mut classes: Vec<Vec<T>> = Vec::new();
        let mut assignment: HashMap<usize, usize> = HashMap::new();

        for (i, p) in points.iter().enumerate() {
            let mut found = false;
            for (ci, class) in classes.iter().enumerate() {
                if self.distance(p, &class[0]) < 1e-10 {
                    assignment.insert(i, ci);
                    found = true;
                    break;
                }
            }
            if !found {
                assignment.insert(i, classes.len());
                classes.push(vec![p.clone()]);
            }
        }

        // add to classes
        for (i, p) in points.iter().enumerate() {
            let ci = assignment[&i];
            if classes[ci].len() == 1 && classes[ci][0] == *p {
                continue;
            }
            if !classes[ci].contains(p) {
                classes[ci].push(p.clone());
            }
        }

        let mut metric = FiniteMetric::new();
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                let d = self.distance(&classes[i][0], &classes[j][0]);
                metric.set_distance(i, j, d);
            }
        }

        (classes, metric)
    }
}

impl<T: Eq + Hash + Clone + Ord> Default for Pseudometric<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Embedding vectors
// ---------------------------------------------------------------------------

/// Dense embedding vector (e.g., from a sentence transformer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub values: Vec<f64>,
}

impl Embedding {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    pub fn zeros(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    pub fn dim(&self) -> usize {
        self.values.len()
    }

    pub fn norm(&self) -> f64 {
        self.values.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            return self.clone();
        }
        Self {
            values: self.values.iter().map(|x| x / n).collect(),
        }
    }

    pub fn dot(&self, other: &Embedding) -> f64 {
        assert_eq!(self.dim(), other.dim(), "Embedding dimension mismatch");
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn cosine_similarity(&self, other: &Embedding) -> f64 {
        let dot = self.dot(other);
        let n1 = self.norm();
        let n2 = other.norm();
        if n1 < 1e-15 || n2 < 1e-15 {
            return 0.0;
        }
        dot / (n1 * n2)
    }

    pub fn cosine_distance(&self, other: &Embedding) -> f64 {
        1.0 - self.cosine_similarity(other)
    }

    pub fn euclidean_distance(&self, other: &Embedding) -> f64 {
        assert_eq!(self.dim(), other.dim());
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn add(&self, other: &Embedding) -> Self {
        assert_eq!(self.dim(), other.dim());
        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            values: self.values.iter().map(|x| x * factor).collect(),
        }
    }

    /// Compute the centroid of a set of embeddings.
    pub fn centroid(embeddings: &[Embedding]) -> Option<Self> {
        if embeddings.is_empty() {
            return None;
        }
        let dim = embeddings[0].dim();
        let mut sum = vec![0.0; dim];
        for e in embeddings {
            assert_eq!(e.dim(), dim);
            for (i, v) in e.values.iter().enumerate() {
                sum[i] += v;
            }
        }
        let n = embeddings.len() as f64;
        Some(Self {
            values: sum.into_iter().map(|x| x / n).collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// Trace and observation types
// ---------------------------------------------------------------------------

/// A single interaction trace with an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionTrace {
    pub id: String,
    pub steps: Vec<InteractionStep>,
    pub metadata: TraceMetadata,
}

/// A single step in an interaction trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionStep {
    pub input: Word,
    pub output: OutputSymbol,
    pub timestamp: Option<DateTime<Utc>>,
    pub latency_ms: Option<u64>,
    pub embedding: Option<Embedding>,
}

/// Metadata for an interaction trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub model_id: String,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<usize>,
    pub system_prompt: Option<String>,
    pub collected_at: DateTime<Utc>,
}

impl InteractionTrace {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            steps: Vec::new(),
            metadata: TraceMetadata {
                model_id: model_id.into(),
                temperature: None,
                top_p: None,
                max_tokens: None,
                system_prompt: None,
                collected_at: Utc::now(),
            },
        }
    }

    pub fn add_step(&mut self, input: Word, output: OutputSymbol) {
        self.steps.push(InteractionStep {
            input,
            output,
            timestamp: Some(Utc::now()),
            latency_ms: None,
            embedding: None,
        });
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Extract the sequence of input words.
    pub fn input_sequence(&self) -> Vec<&Word> {
        self.steps.iter().map(|s| &s.input).collect()
    }

    /// Extract the sequence of output symbols.
    pub fn output_sequence(&self) -> Vec<&OutputSymbol> {
        self.steps.iter().map(|s| &s.output).collect()
    }
}

/// A collection of traces from the same model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCorpus {
    pub model_id: String,
    pub traces: Vec<InteractionTrace>,
}

impl TraceCorpus {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            traces: Vec::new(),
        }
    }

    pub fn add_trace(&mut self, trace: InteractionTrace) {
        self.traces.push(trace);
    }

    pub fn total_steps(&self) -> usize {
        self.traces.iter().map(|t| t.len()).sum()
    }

    pub fn num_traces(&self) -> usize {
        self.traces.len()
    }
}

// ---------------------------------------------------------------------------
// Certificate types
// ---------------------------------------------------------------------------

/// A behavioral certificate asserting a property about an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralCertificate {
    pub id: String,
    pub model_id: String,
    pub property: CertifiedProperty,
    pub confidence: f64,
    pub abstraction_level: (usize, usize, f64), // (k, n, ε)
    pub num_queries: usize,
    pub issued_at: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub witness: Option<CertificateWitness>,
}

/// The property being certified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertifiedProperty {
    BisimilarityBound { other_model: String, bound: f64 },
    SafetyProperty { description: String, holds: bool },
    FairnessProperty { metric: String, bound: f64 },
    ConsistencyProperty { self_distance: f64 },
    RobustnessProperty { perturbation_type: String, bound: f64 },
}

/// Witness data supporting a certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateWitness {
    pub distinguishing_traces: Vec<InteractionTrace>,
    pub bisimulation_distance: f64,
    pub sample_complexity: usize,
    pub bandwidth_estimate: f64,
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors from the coalgebra module.
#[derive(Debug, thiserror::Error)]
pub enum CoalgebraError {
    #[error("State not found: {0}")]
    StateNotFound(StateId),

    #[error("Invalid distribution: probabilities sum to {0}, expected ≤ 1.0")]
    InvalidDistribution(f64),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Convergence failure after {iterations} iterations, residual = {residual}")]
    ConvergenceFailure { iterations: usize, residual: f64 },

    #[error("Empty state space")]
    EmptyStateSpace,

    #[error("Abstraction error: {0}")]
    AbstractionError(String),

    #[error("Transition table error: {0}")]
    TransitionTable(#[from] TransitionTableError),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_id_creation() {
        let s1 = StateId::new("start");
        let s2 = StateId::indexed("q", 3);
        let s3 = StateId::fresh();
        assert_eq!(s1.as_str(), "start");
        assert_eq!(s2.as_str(), "q_3");
        assert!(!s3.as_str().is_empty());
    }

    #[test]
    fn test_state_id_from() {
        let s1: StateId = "hello".into();
        let s2: StateId = String::from("world").into();
        let s3: StateId = 42usize.into();
        assert_eq!(s1.as_str(), "hello");
        assert_eq!(s2.as_str(), "world");
        assert_eq!(s3.as_str(), "s42");
    }

    #[test]
    fn test_symbol_epsilon() {
        let eps = Symbol::epsilon();
        assert!(eps.is_epsilon());
        assert_eq!(eps.length(), 0);
    }

    #[test]
    fn test_word_operations() {
        let w1 = Word::from_str_slice(&["a", "b", "c"]);
        let w2 = Word::from_str_slice(&["d", "e"]);
        let w3 = w1.concat(&w2);
        assert_eq!(w3.len(), 5);

        let prefix = w3.prefix(3);
        assert_eq!(prefix.len(), 3);
        assert_eq!(prefix.symbols[0], Symbol::new("a"));

        let suffix = w3.suffix(2);
        assert_eq!(suffix.len(), 2);
        assert_eq!(suffix.symbols[0], Symbol::new("d"));

        assert!(w1.is_prefix_of(&w3));
        assert!(!w2.is_prefix_of(&w3));
    }

    #[test]
    fn test_word_enumeration() {
        let alphabet = vec![Symbol::new("0"), Symbol::new("1")];
        let words = Word::enumerate_up_to(&alphabet, 2);
        // ε, 0, 1, 00, 01, 10, 11 = 7 words
        assert_eq!(words.len(), 7);
    }

    #[test]
    fn test_action_space() {
        let alpha = vec![Symbol::new("a"), Symbol::new("b")];
        let space = ActionSpace::new(alpha, 3);
        assert_eq!(space.alphabet_size(), 2);
        // 1 + 2 + 4 + 8 = 15 words
        assert_eq!(space.total_words, 15);

        let w = Word::from_str_slice(&["a", "b"]);
        assert!(space.contains_word(&w));

        let w_long = Word::from_str_slice(&["a", "b", "a", "a"]);
        assert!(!space.contains_word(&w_long));
    }

    #[test]
    fn test_transition_table() {
        let mut table = TransitionTable::new();
        let s0 = StateId::new("s0");
        let s1 = StateId::new("s1");
        let input = Word::from_str_slice(&["hello"]);
        let output = OutputSymbol::new("world");

        table.add_entry(s0.clone(), input.clone(), output, s1.clone(), 0.8);
        table.add_entry(
            s0.clone(),
            input.clone(),
            OutputSymbol::new("hi"),
            s0.clone(),
            0.2,
        );

        let trans = table.get_transitions(&s0, &input).unwrap();
        assert_eq!(trans.len(), 2);

        let total: f64 = trans.iter().map(|t| t.probability).sum();
        assert!((total - 1.0).abs() < 1e-10);

        assert!(table.validate_subdistributions(1e-6).is_ok());
        assert_eq!(table.num_transitions(), 2);
    }

    #[test]
    fn test_transition_table_invalid() {
        let mut table = TransitionTable::new();
        let s0 = StateId::new("s0");
        let input = Word::from_str_slice(&["x"]);

        table.add_entry(s0.clone(), input.clone(), OutputSymbol::new("a"), s0.clone(), 0.7);
        table.add_entry(s0.clone(), input.clone(), OutputSymbol::new("b"), s0.clone(), 0.5);

        assert!(table.validate_subdistributions(1e-6).is_err());
    }

    #[test]
    fn test_transition_table_normalize() {
        let mut table = TransitionTable::new();
        let s0 = StateId::new("s0");
        let input = Word::from_str_slice(&["x"]);

        table.add_entry(s0.clone(), input.clone(), OutputSymbol::new("a"), s0.clone(), 3.0);
        table.add_entry(s0.clone(), input.clone(), OutputSymbol::new("b"), s0.clone(), 7.0);

        table.normalize();
        let trans = table.get_transitions(&s0, &input).unwrap();
        let total: f64 = trans.iter().map(|t| t.probability).sum();
        assert!((total - 1.0).abs() < 1e-10);
        assert!((trans[0].probability - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_finite_metric() {
        let mut metric = FiniteMetric::new();
        metric.set_distance(0usize, 1, 1.0);
        metric.set_distance(1, 2, 2.0);
        metric.set_distance(0, 2, 2.5);

        assert_eq!(metric.distance(&0, &0), 0.0);
        assert_eq!(metric.distance(&0, &1), 1.0);
        assert_eq!(metric.distance(&1, &0), 1.0);

        let points = vec![0, 1, 2];
        assert!(metric.validate_symmetry(&points));
        assert!(metric.validate_identity(&points));
        assert!(metric.validate_triangle_inequality(&points));
        assert!((metric.diameter(&points) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_finite_metric_triangle_violation() {
        let mut metric = FiniteMetric::new();
        metric.set_distance(0usize, 1, 1.0);
        metric.set_distance(1, 2, 1.0);
        metric.set_distance(0, 2, 3.0); // violates triangle inequality

        let points = vec![0, 1, 2];
        assert!(!metric.validate_triangle_inequality(&points));
    }

    #[test]
    fn test_pseudometric_quotient() {
        let mut pm = Pseudometric::new();
        pm.set_distance(0usize, 1, 0.0); // 0 and 1 are identified
        pm.set_distance(0, 2, 1.0);
        pm.set_distance(1, 2, 1.0);

        let (classes, _metric) = pm.quotient(&[0, 1, 2]);
        // 0 and 1 should be in the same class
        assert!(classes.len() <= 2);
    }

    #[test]
    fn test_embedding_operations() {
        let e1 = Embedding::new(vec![1.0, 0.0, 0.0]);
        let e2 = Embedding::new(vec![0.0, 1.0, 0.0]);

        assert!((e1.norm() - 1.0).abs() < 1e-10);
        assert!((e1.dot(&e2)).abs() < 1e-10);
        assert!((e1.cosine_similarity(&e2)).abs() < 1e-10);
        assert!((e1.euclidean_distance(&e2) - std::f64::consts::SQRT_2).abs() < 1e-10);

        let e3 = e1.add(&e2);
        assert_eq!(e3.dim(), 3);
        assert!((e3.values[0] - 1.0).abs() < 1e-10);
        assert!((e3.values[1] - 1.0).abs() < 1e-10);

        let centroid = Embedding::centroid(&[e1.clone(), e2.clone()]).unwrap();
        assert!((centroid.values[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_embedding_normalize() {
        let e = Embedding::new(vec![3.0, 4.0]);
        let en = e.normalize();
        assert!((en.norm() - 1.0).abs() < 1e-10);
        assert!((en.values[0] - 0.6).abs() < 1e-10);
        assert!((en.values[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_interaction_trace() {
        let mut trace = InteractionTrace::new("gpt-4");
        trace.add_step(
            Word::from_str_slice(&["hello"]),
            OutputSymbol::new("world"),
        );
        trace.add_step(
            Word::from_str_slice(&["how", "are", "you"]),
            OutputSymbol::new("fine"),
        );

        assert_eq!(trace.len(), 2);
        assert_eq!(trace.input_sequence().len(), 2);
        assert_eq!(trace.output_sequence().len(), 2);
    }

    #[test]
    fn test_trace_corpus() {
        let mut corpus = TraceCorpus::new("gpt-4");
        let mut trace = InteractionTrace::new("gpt-4");
        trace.add_step(Word::from_str_slice(&["a"]), OutputSymbol::new("b"));
        corpus.add_trace(trace);

        assert_eq!(corpus.num_traces(), 1);
        assert_eq!(corpus.total_steps(), 1);
    }

    #[test]
    fn test_coalgebra_config_default() {
        let config = CoalgebraConfig::default();
        assert_eq!(config.max_input_length, 5);
        assert!(config.epsilon > 0.0);
    }

    #[test]
    fn test_output_symbol_with_cluster() {
        let o = OutputSymbol::with_cluster("hello", ClusterId(5));
        assert_eq!(o.value, "hello");
        assert_eq!(o.cluster_id, Some(ClusterId(5)));
    }
}
