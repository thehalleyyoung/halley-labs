//! Channel models for cache side-channel leakage quantification.
//!
//! Models the attacker's observation as a discrete memoryless channel from
//! secret inputs to observable outputs (cache timing, access patterns, etc.).

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use shared_types::{CacheLine, CacheSet, VirtualAddress};

use crate::distribution::Distribution;
use crate::entropy::{ConditionalEntropy, MaxLeakage, MutualInformation, ShannonEntropy};
use crate::{QuantifyError, QuantifyResult};

// ---------------------------------------------------------------------------
// Channel Trait
// ---------------------------------------------------------------------------

/// A discrete memoryless channel mapping inputs to observation distributions.
pub trait Channel: fmt::Debug + Send + Sync {
    /// The type of secret input.
    type Input: Ord + Clone + fmt::Debug;
    /// The type of observable output.
    type Output: Ord + Clone + fmt::Debug;

    /// Return the conditional distribution P(Y|X=x) for a given input.
    fn observe(&self, input: &Self::Input) -> QuantifyResult<Distribution<Self::Output>>;

    /// Return the number of possible inputs (|X|).
    fn input_cardinality(&self) -> usize;

    /// Return the number of possible outputs (|Y|).
    fn output_cardinality(&self) -> usize;

    /// Compute the channel matrix as a dense row-major matrix.
    /// Row `i` = P(Y|X = input_i).
    fn to_matrix(&self) -> QuantifyResult<ChannelMatrix>;
}

// ---------------------------------------------------------------------------
// Channel Matrix
// ---------------------------------------------------------------------------

/// A dense channel matrix representation.
///
/// `rows[i][j]` = P(Y = j | X = i). Each row sums to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMatrix {
    /// Number of inputs (rows).
    pub num_inputs: usize,
    /// Number of outputs (columns).
    pub num_outputs: usize,
    /// Row-major matrix data: `rows[i][j]` = P(Y=j | X=i).
    pub rows: Vec<Vec<f64>>,
}

impl ChannelMatrix {
    /// Create from raw row-major data.
    pub fn new(rows: Vec<Vec<f64>>) -> QuantifyResult<Self> {
        if rows.is_empty() {
            return Err(QuantifyError::EmptySupport);
        }
        let num_outputs = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != num_outputs {
                return Err(QuantifyError::DimensionMismatch {
                    expected: num_outputs,
                    got: row.len(),
                });
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(QuantifyError::InvalidDistribution(format!(
                    "row {i} sums to {sum}, expected 1.0"
                )));
            }
        }
        Ok(Self {
            num_inputs: rows.len(),
            num_outputs,
            rows,
        })
    }

    /// Create a deterministic (identity) channel of size `n`.
    pub fn identity(n: usize) -> QuantifyResult<Self> {
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();
        Self::new(rows)
    }

    /// Compute Shannon mutual information I(X;Y) for a given prior.
    pub fn mutual_information(&self, prior: &[f64]) -> QuantifyResult<MutualInformation> {
        MutualInformation::from_channel(prior, &self.rows)
    }

    /// Compute conditional entropy H(X|Y) for a given prior.
    pub fn conditional_entropy(&self, prior: &[f64]) -> QuantifyResult<ConditionalEntropy> {
        ConditionalEntropy::from_channel(prior, &self.rows)
    }

    /// Compute max-leakage (prior-independent).
    pub fn max_leakage(&self) -> QuantifyResult<MaxLeakage> {
        MaxLeakage::compute(&self.rows)
    }

    /// Compose two channels sequentially: self ∘ other.
    ///
    /// self: X→Y, other: Y→Z ⟹ result: X→Z with P(Z|X) = Σ_y P(Y=y|X) P(Z|Y=y).
    pub fn compose(&self, other: &ChannelMatrix) -> QuantifyResult<ChannelMatrix> {
        if self.num_outputs != other.num_inputs {
            return Err(QuantifyError::DimensionMismatch {
                expected: self.num_outputs,
                got: other.num_inputs,
            });
        }
        let mut rows = Vec::with_capacity(self.num_inputs);
        for i in 0..self.num_inputs {
            let mut new_row = vec![0.0; other.num_outputs];
            for (y, &p_y_given_x) in self.rows[i].iter().enumerate() {
                for (z, &p_z_given_y) in other.rows[y].iter().enumerate() {
                    new_row[z] += p_y_given_x * p_z_given_y;
                }
            }
            rows.push(new_row);
        }
        ChannelMatrix::new(rows)
    }

    /// Transpose the channel matrix (swap inputs and outputs).
    pub fn transpose(&self) -> Self {
        let mut rows = vec![vec![0.0; self.num_inputs]; self.num_outputs];
        for (i, row) in self.rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                rows[j][i] = v;
            }
        }
        Self {
            num_inputs: self.num_outputs,
            num_outputs: self.num_inputs,
            rows,
        }
    }
}

impl fmt::Display for ChannelMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChannelMatrix({}×{}, ML={:.4} bits)",
            self.num_inputs,
            self.num_outputs,
            self.max_leakage().map(|m| m.value()).unwrap_or(f64::NAN)
        )
    }
}

// ---------------------------------------------------------------------------
// Cache Observation
// ---------------------------------------------------------------------------

/// A single cache side-channel observation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CacheObservation {
    /// Cache set index that was accessed.
    pub cache_set: u32,
    /// Whether the access was a cache hit.
    pub hit: bool,
    /// Access latency in (simulated) cycles, if available.
    pub latency_cycles: Option<u64>,
}

impl CacheObservation {
    /// Create a new observation.
    pub fn new(cache_set: u32, hit: bool) -> Self {
        Self {
            cache_set,
            hit,
            latency_cycles: None,
        }
    }

    /// Create with explicit latency.
    pub fn with_latency(cache_set: u32, hit: bool, latency: u64) -> Self {
        Self {
            cache_set,
            hit,
            latency_cycles: Some(latency),
        }
    }
}

impl fmt::Display for CacheObservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.hit { "hit" } else { "miss" };
        write!(f, "set[{}]:{}", self.cache_set, status)
    }
}

// ---------------------------------------------------------------------------
// Observation Set
// ---------------------------------------------------------------------------

/// A set of observations forming a single attacker measurement.
///
/// Typically one observation per memory access or cache-line granularity access
/// in a single execution trace.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ObservationSet {
    /// Ordered sequence of individual observations.
    pub observations: Vec<CacheObservation>,
    /// Optional trace identifier for correlation.
    pub trace_id: Option<u64>,
}

impl ObservationSet {
    /// Create an empty observation set.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            trace_id: None,
        }
    }

    /// Create from a vector of observations.
    pub fn from_observations(obs: Vec<CacheObservation>) -> Self {
        Self {
            observations: obs,
            trace_id: None,
        }
    }

    /// Add an observation.
    pub fn push(&mut self, obs: CacheObservation) {
        self.observations.push(obs);
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Unique cache sets touched.
    pub fn touched_sets(&self) -> Vec<u32> {
        let mut sets: Vec<u32> = self.observations.iter().map(|o| o.cache_set).collect();
        sets.sort_unstable();
        sets.dedup();
        sets
    }

    /// Count of cache hits.
    pub fn hit_count(&self) -> usize {
        self.observations.iter().filter(|o| o.hit).count()
    }

    /// Count of cache misses.
    pub fn miss_count(&self) -> usize {
        self.observations.iter().filter(|o| !o.hit).count()
    }
}

impl Default for ObservationSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ObservationSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ObsSet[{} obs, {} hits, {} misses]",
            self.len(),
            self.hit_count(),
            self.miss_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Cache Channel
// ---------------------------------------------------------------------------

/// A cache side-channel modeled as a discrete memoryless channel.
///
/// Maps secret-dependent memory addresses to attacker-observable cache states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheChannel {
    /// Number of cache sets in the modeled cache.
    pub num_sets: usize,
    /// Number of ways (associativity).
    pub associativity: usize,
    /// Cache line size in bytes.
    pub line_size: usize,
    /// Mapping from input index to observation distributions.
    /// Key: input index, Value: probability distribution over observation indices.
    observation_map: IndexMap<usize, Vec<(usize, f64)>>,
    /// Total number of distinct inputs tracked.
    num_inputs: usize,
    /// Total number of distinct observations.
    num_observations: usize,
}

impl CacheChannel {
    /// Create a new cache channel from geometry.
    pub fn new(num_sets: usize, associativity: usize, line_size: usize) -> Self {
        Self {
            num_sets,
            associativity,
            line_size,
            observation_map: IndexMap::new(),
            num_inputs: 0,
            num_observations: 0,
        }
    }

    /// Create from a shared-types `CacheSet` count and associativity.
    pub fn from_cache_params(num_sets: usize, ways: usize, line_size: usize) -> Self {
        Self::new(num_sets, ways, line_size)
    }

    /// Register an input → observation probability mapping.
    pub fn add_input(
        &mut self,
        input_idx: usize,
        observations: Vec<(usize, f64)>,
    ) {
        for &(obs_idx, _) in &observations {
            if obs_idx >= self.num_observations {
                self.num_observations = obs_idx + 1;
            }
        }
        self.observation_map.insert(input_idx, observations);
        if input_idx >= self.num_inputs {
            self.num_inputs = input_idx + 1;
        }
    }

    /// Build the dense channel matrix.
    pub fn to_channel_matrix(&self) -> QuantifyResult<ChannelMatrix> {
        if self.num_inputs == 0 || self.num_observations == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        let mut rows = vec![vec![0.0; self.num_observations]; self.num_inputs];
        for (&input, obs) in &self.observation_map {
            for &(obs_idx, prob) in obs {
                rows[input][obs_idx] = prob;
            }
            // Normalize row if needed
            let sum: f64 = rows[input].iter().sum();
            if sum > 0.0 && (sum - 1.0).abs() > 1e-9 {
                for v in &mut rows[input] {
                    *v /= sum;
                }
            }
        }
        ChannelMatrix::new(rows)
    }

    /// Compute channel capacity (maximum mutual information over all priors).
    pub fn capacity(&self) -> QuantifyResult<ChannelCapacity> {
        let matrix = self.to_channel_matrix()?;
        ChannelCapacity::compute(&matrix)
    }

    /// Compute max-leakage for this channel.
    pub fn max_leakage(&self) -> QuantifyResult<MaxLeakage> {
        let matrix = self.to_channel_matrix()?;
        matrix.max_leakage()
    }

    /// Compute the number of distinguishable inputs through the cache.
    pub fn distinguishable_input_count(&self) -> usize {
        self.observation_map.len()
    }

    /// Map a virtual address to its cache set index.
    pub fn address_to_set(&self, addr: u64) -> usize {
        if self.line_size == 0 {
            return 0;
        }
        let line_offset_bits = (self.line_size as f64).log2() as usize;
        let set_index = (addr >> line_offset_bits) as usize % self.num_sets;
        set_index
    }
}

impl Channel for CacheChannel {
    type Input = usize;
    type Output = usize;

    fn observe(&self, input: &usize) -> QuantifyResult<Distribution<usize>> {
        let obs = self.observation_map.get(input).ok_or_else(|| {
            QuantifyError::ModelError(format!("no observations for input {input}"))
        })?;
        Distribution::from_pairs(obs.iter().cloned())
    }

    fn input_cardinality(&self) -> usize {
        self.num_inputs
    }

    fn output_cardinality(&self) -> usize {
        self.num_observations
    }

    fn to_matrix(&self) -> QuantifyResult<ChannelMatrix> {
        self.to_channel_matrix()
    }
}

impl fmt::Display for CacheChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheChannel(sets={}, ways={}, line={}B, inputs={}, obs={})",
            self.num_sets, self.associativity, self.line_size, self.num_inputs, self.num_observations
        )
    }
}

// ---------------------------------------------------------------------------
// Channel Capacity
// ---------------------------------------------------------------------------

/// Channel capacity C = max_{P(X)} I(X;Y) computed via the Blahut–Arimoto
/// algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelCapacity {
    /// Capacity in bits.
    pub bits: f64,
    /// The capacity-achieving input distribution.
    pub achieving_distribution: Vec<f64>,
    /// Number of Blahut–Arimoto iterations used.
    pub iterations: usize,
    /// Final convergence residual.
    pub residual: f64,
}

/// Maximum Blahut–Arimoto iterations.
const BA_MAX_ITER: usize = 1000;
/// Convergence tolerance for Blahut–Arimoto.
const BA_TOLERANCE: f64 = 1e-10;

impl ChannelCapacity {
    /// Compute channel capacity via the Blahut–Arimoto algorithm.
    pub fn compute(matrix: &ChannelMatrix) -> QuantifyResult<Self> {
        Self::compute_with_params(matrix, BA_MAX_ITER, BA_TOLERANCE)
    }

    /// Compute with custom iteration parameters.
    pub fn compute_with_params(
        matrix: &ChannelMatrix,
        max_iter: usize,
        tolerance: f64,
    ) -> QuantifyResult<Self> {
        let n = matrix.num_inputs;
        let m = matrix.num_outputs;
        if n == 0 || m == 0 {
            return Err(QuantifyError::EmptySupport);
        }

        // Initialize with uniform distribution
        let mut q = vec![1.0 / n as f64; n];
        let mut residual = f64::INFINITY;

        for iter in 0..max_iter {
            // Compute output distribution r[j] = Σ_i q[i] * W[i][j]
            let mut r = vec![0.0; m];
            for (i, row) in matrix.rows.iter().enumerate() {
                for (j, &w) in row.iter().enumerate() {
                    r[j] += q[i] * w;
                }
            }

            // Compute c[i] = exp(Σ_j W[i][j] * log(W[i][j]/r[j]))
            let mut c = vec![0.0_f64; n];
            for (i, row) in matrix.rows.iter().enumerate() {
                let mut sum = 0.0;
                for (j, &w) in row.iter().enumerate() {
                    if w > 0.0 && r[j] > 0.0 {
                        sum += w * (w / r[j]).ln();
                    }
                }
                c[i] = sum.exp();
            }

            // Update q
            let c_sum: f64 = q.iter().zip(c.iter()).map(|(&qi, &ci)| qi * ci).sum();
            let mut new_q = vec![0.0; n];
            for i in 0..n {
                new_q[i] = q[i] * c[i] / c_sum;
            }

            residual = new_q
                .iter()
                .zip(q.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            q = new_q;

            if residual < tolerance {
                let bits = self::capacity_from_distribution(&q, matrix);
                return Ok(Self {
                    bits,
                    achieving_distribution: q,
                    iterations: iter + 1,
                    residual,
                });
            }
        }

        Err(QuantifyError::ConvergenceFailure {
            iterations: max_iter,
            residual,
        })
    }

    /// The capacity value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }
}

/// Helper: compute mutual information for a given input distribution and channel.
fn capacity_from_distribution(prior: &[f64], matrix: &ChannelMatrix) -> f64 {
    MutualInformation::from_channel(prior, &matrix.rows)
        .map(|mi| mi.value())
        .unwrap_or(0.0)
}

impl fmt::Display for ChannelCapacity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "C = {:.4} bits ({} BA iterations)",
            self.bits, self.iterations
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_matrix_identity() {
        let m = ChannelMatrix::identity(4).unwrap();
        assert_eq!(m.num_inputs, 4);
        assert_eq!(m.num_outputs, 4);
        let ml = m.max_leakage().unwrap();
        assert!((ml.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_channel_matrix_compose() {
        let a = ChannelMatrix::identity(2).unwrap();
        let b = ChannelMatrix::identity(2).unwrap();
        let c = a.compose(&b).unwrap();
        assert_eq!(c.num_inputs, 2);
        assert_eq!(c.num_outputs, 2);
    }

    #[test]
    fn test_cache_observation_display() {
        let obs = CacheObservation::new(3, true);
        assert_eq!(format!("{obs}"), "set[3]:hit");
    }

    #[test]
    fn test_observation_set_counts() {
        let mut os = ObservationSet::new();
        os.push(CacheObservation::new(0, true));
        os.push(CacheObservation::new(1, false));
        os.push(CacheObservation::new(0, true));
        assert_eq!(os.len(), 3);
        assert_eq!(os.hit_count(), 2);
        assert_eq!(os.miss_count(), 1);
        assert_eq!(os.touched_sets(), vec![0, 1]);
    }

    #[test]
    fn test_cache_channel_basic() {
        let mut ch = CacheChannel::new(64, 8, 64);
        ch.add_input(0, vec![(0, 0.5), (1, 0.5)]);
        ch.add_input(1, vec![(0, 0.3), (1, 0.7)]);
        let m = ch.to_channel_matrix().unwrap();
        assert_eq!(m.num_inputs, 2);
        assert_eq!(m.num_outputs, 2);
    }

    #[test]
    fn test_channel_capacity_binary_symmetric() {
        // Binary symmetric channel with crossover probability 0
        // → capacity = 1 bit
        let m = ChannelMatrix::identity(2).unwrap();
        let cap = ChannelCapacity::compute(&m).unwrap();
        assert!((cap.value() - 1.0).abs() < 1e-4);
    }
}
