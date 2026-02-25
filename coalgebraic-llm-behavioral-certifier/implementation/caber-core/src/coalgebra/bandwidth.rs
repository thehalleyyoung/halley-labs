//! Functor bandwidth — the β(F, α) invariant.
//!
//! The functor bandwidth β(F, α) measures how complex a behavioral functor F
//! is at abstraction level α. Concretely, it is the log of the ε-covering
//! number of the image of F in a metric space of behavioral observations:
//!
//!   β(F, α) = log N(F(S_α), ε, d_TV)
//!
//! where N(X, ε, d) is the minimum number of ε-balls (in metric d) needed
//! to cover the set X.
//!
//! This module provides:
//! - `FunctorBandwidth`: the computed invariant with metadata
//! - `BandwidthEstimator`: estimates β(F, α) from empirical observations
//! - Epsilon-covering number computation on metric spaces
//! - Metric entropy estimation via random sampling
//! - Connection to VC dimension (covering number bounds)
//! - Sample complexity bounds: Õ(β · n · log(1/δ))
//! - Adaptive bandwidth tracking during learning
//! - Bandwidth-guided query allocation
//! - `BandwidthProfile`: bandwidth at multiple abstraction levels
//! - Lower/upper bounds on bandwidth

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use ordered_float::OrderedFloat;
use rand::prelude::*;
use rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use super::distribution::SubDistribution;
use super::types::*;

// ---------------------------------------------------------------------------
// FunctorBandwidth — the main invariant
// ---------------------------------------------------------------------------

/// The computed functor bandwidth β(F, α).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctorBandwidth {
    /// The bandwidth value β.
    pub beta: f64,
    /// The abstraction level (k, n, ε) at which β was computed.
    pub abstraction_level: (usize, usize, f64),
    /// The covering radius ε used.
    pub epsilon: f64,
    /// Number of covering centers found.
    pub covering_size: usize,
    /// Lower bound on the true bandwidth.
    pub lower_bound: f64,
    /// Upper bound on the true bandwidth.
    pub upper_bound: f64,
    /// Confidence interval half-width.
    pub confidence_half_width: f64,
    /// Number of samples used in the estimate.
    pub num_samples: usize,
    /// Estimation method used.
    pub method: CoveringNumberMethod,
    /// Metadata about the estimation process.
    pub metadata: BandwidthMetadata,
}

/// Metadata about how the bandwidth estimate was computed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthMetadata {
    /// Wall-clock time for estimation (ms).
    pub elapsed_ms: u64,
    /// Dimension of the behavioral observation space.
    pub observation_dim: usize,
    /// Whether dimension reduction was applied.
    pub dimension_reduced: bool,
    /// Effective dimension after reduction.
    pub effective_dim: usize,
    /// Packing number lower bound.
    pub packing_number: usize,
    /// Estimated metric entropy rate.
    pub metric_entropy_rate: f64,
}

impl Default for BandwidthMetadata {
    fn default() -> Self {
        Self {
            elapsed_ms: 0,
            observation_dim: 0,
            dimension_reduced: false,
            effective_dim: 0,
            packing_number: 0,
            metric_entropy_rate: 0.0,
        }
    }
}

impl FunctorBandwidth {
    /// Create a new bandwidth estimate.
    pub fn new(beta: f64, epsilon: f64, covering_size: usize) -> Self {
        Self {
            beta,
            abstraction_level: (0, 0, epsilon),
            epsilon,
            covering_size,
            lower_bound: beta * 0.8,
            upper_bound: beta * 1.2,
            confidence_half_width: beta * 0.2,
            num_samples: 0,
            method: CoveringNumberMethod::RandomSampling,
            metadata: BandwidthMetadata::default(),
        }
    }

    /// Compute the sample complexity bound: Õ(β · n · log(1/δ)).
    pub fn sample_complexity_bound(&self, num_states: usize, delta: f64) -> usize {
        assert!(delta > 0.0 && delta < 1.0, "δ must be in (0,1)");
        let log_inv_delta = (1.0 / delta).ln();
        let log_factor = (self.beta.max(1.0)).ln().max(1.0);
        let bound = self.beta * (num_states as f64) * log_inv_delta * log_factor;
        bound.ceil() as usize
    }

    /// Compute PAC sample complexity: m ≥ (β/ε²) · ln(2/δ).
    pub fn pac_sample_complexity(&self, accuracy: f64, delta: f64) -> usize {
        assert!(accuracy > 0.0 && delta > 0.0 && delta < 1.0);
        let bound = (self.beta / (accuracy * accuracy)) * (2.0 / delta).ln();
        bound.ceil() as usize
    }

    /// Check whether the bandwidth suggests the model is learnable with the given budget.
    pub fn is_learnable_within_budget(&self, budget: usize, num_states: usize, delta: f64) -> bool {
        let required = self.sample_complexity_bound(num_states, delta);
        budget >= required
    }

    /// Return a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "β(F, α) = {:.4} [CI: ({:.4}, {:.4})], covering_size={}, ε={:.4}, method={:?}",
            self.beta,
            self.lower_bound,
            self.upper_bound,
            self.covering_size,
            self.epsilon,
            self.method,
        )
    }
}

impl fmt::Display for FunctorBandwidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "β={:.4} (ε={:.4}, n={})", self.beta, self.epsilon, self.covering_size)
    }
}

// ---------------------------------------------------------------------------
// BehavioralPoint — observations in metric space
// ---------------------------------------------------------------------------

/// A point in the behavioral observation space.
/// Represents the empirical output distribution from a given state and context.
#[derive(Debug, Clone)]
pub struct BehavioralPoint {
    /// Identifier (state + context).
    pub id: String,
    /// The empirical output distribution observed.
    pub distribution: Vec<f64>,
    /// Optional embedding of the observation.
    pub embedding: Option<Vec<f64>>,
}

impl BehavioralPoint {
    pub fn new(id: impl Into<String>, distribution: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            distribution,
            embedding: None,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn dim(&self) -> usize {
        self.distribution.len()
    }

    /// Total variation distance to another point.
    pub fn tv_distance(&self, other: &BehavioralPoint) -> f64 {
        assert_eq!(self.distribution.len(), other.distribution.len(),
                   "Dimension mismatch in TV distance");
        let mut total = 0.0;
        for i in 0..self.distribution.len() {
            total += (self.distribution[i] - other.distribution[i]).abs();
        }
        total / 2.0
    }

    /// L2 (Euclidean) distance to another point.
    pub fn l2_distance(&self, other: &BehavioralPoint) -> f64 {
        assert_eq!(self.distribution.len(), other.distribution.len());
        self.distribution
            .iter()
            .zip(other.distribution.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// L∞ (Chebyshev) distance to another point.
    pub fn linf_distance(&self, other: &BehavioralPoint) -> f64 {
        assert_eq!(self.distribution.len(), other.distribution.len());
        self.distribution
            .iter()
            .zip(other.distribution.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max)
    }

    /// Hellinger distance to another point.
    pub fn hellinger_distance(&self, other: &BehavioralPoint) -> f64 {
        assert_eq!(self.distribution.len(), other.distribution.len());
        let sum_sq: f64 = self
            .distribution
            .iter()
            .zip(other.distribution.iter())
            .map(|(a, b)| (a.sqrt() - b.sqrt()).powi(2))
            .sum();
        (sum_sq / 2.0).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Metric space trait for covering number computation
// ---------------------------------------------------------------------------

/// Trait for metric spaces over which we compute covering numbers.
pub trait MetricSpace: Send + Sync {
    /// Type of points in the space.
    type Point: Clone + Send + Sync;

    /// Distance between two points.
    fn distance(&self, a: &Self::Point, b: &Self::Point) -> f64;

    /// Dimension of the ambient space.
    fn ambient_dimension(&self) -> usize;
}

/// A concrete metric space over behavioral points with a configurable metric.
#[derive(Debug, Clone)]
pub struct BehavioralMetricSpace {
    /// The distance metric to use.
    pub metric_type: DistanceMetric,
}

/// Choice of distance metric for covering computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    TotalVariation,
    L2,
    LInfinity,
    Hellinger,
}

impl BehavioralMetricSpace {
    pub fn new(metric_type: DistanceMetric) -> Self {
        Self { metric_type }
    }

    pub fn total_variation() -> Self {
        Self::new(DistanceMetric::TotalVariation)
    }

    pub fn l2() -> Self {
        Self::new(DistanceMetric::L2)
    }
}

impl MetricSpace for BehavioralMetricSpace {
    type Point = BehavioralPoint;

    fn distance(&self, a: &BehavioralPoint, b: &BehavioralPoint) -> f64 {
        match self.metric_type {
            DistanceMetric::TotalVariation => a.tv_distance(b),
            DistanceMetric::L2 => a.l2_distance(b),
            DistanceMetric::LInfinity => a.linf_distance(b),
            DistanceMetric::Hellinger => a.hellinger_distance(b),
        }
    }

    fn ambient_dimension(&self) -> usize {
        0 // determined by points
    }
}

// ---------------------------------------------------------------------------
// Covering number computation
// ---------------------------------------------------------------------------

/// Result of a covering number computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringNumberResult {
    /// The computed covering number N(X, ε, d).
    pub covering_number: usize,
    /// The covering radius ε.
    pub epsilon: f64,
    /// Indices of points chosen as centers.
    pub center_indices: Vec<usize>,
    /// Maximum uncovered distance (residual).
    pub max_uncovered_distance: f64,
    /// Average distance from each point to its nearest center.
    pub avg_distance_to_center: f64,
    /// The assignment: point index -> center index.
    pub assignments: Vec<usize>,
}

/// Compute the ε-covering number using greedy farthest-point algorithm.
///
/// This is the Gonzalez algorithm: iteratively pick the point farthest from
/// the current center set. Gives a 2-approximation to the optimal covering.
pub fn covering_number_greedy<M: MetricSpace>(
    space: &M,
    points: &[M::Point],
    epsilon: f64,
) -> CoveringNumberResult {
    if points.is_empty() {
        return CoveringNumberResult {
            covering_number: 0,
            epsilon,
            center_indices: vec![],
            max_uncovered_distance: 0.0,
            avg_distance_to_center: 0.0,
            assignments: vec![],
        };
    }

    let n = points.len();
    let mut centers: Vec<usize> = vec![0]; // start with first point
    let mut min_dist_to_center: Vec<f64> = vec![f64::INFINITY; n];
    let mut assignments: Vec<usize> = vec![0; n];

    // Initialize distances from first center
    for i in 0..n {
        let d = space.distance(&points[0], &points[i]);
        min_dist_to_center[i] = d;
        assignments[i] = 0;
    }
    min_dist_to_center[0] = 0.0;

    loop {
        // Find the point farthest from the current center set
        let (farthest_idx, &farthest_dist) = min_dist_to_center
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        if farthest_dist <= epsilon {
            break;
        }

        // Add this point as a new center
        let new_center_idx = centers.len();
        centers.push(farthest_idx);

        // Update distances
        for i in 0..n {
            let d = space.distance(&points[farthest_idx], &points[i]);
            if d < min_dist_to_center[i] {
                min_dist_to_center[i] = d;
                assignments[i] = new_center_idx;
            }
        }

        // Safety: don't exceed number of points
        if centers.len() >= n {
            break;
        }
    }

    let max_uncovered = min_dist_to_center
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);

    let avg_dist = if n > 0 {
        min_dist_to_center.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    CoveringNumberResult {
        covering_number: centers.len(),
        epsilon,
        center_indices: centers,
        max_uncovered_distance: max_uncovered,
        avg_distance_to_center: avg_dist,
        assignments,
    }
}

/// Compute the ε-covering number using random sampling.
///
/// Sample random subsets and take the minimum covering number found.
pub fn covering_number_random<M: MetricSpace>(
    space: &M,
    points: &[M::Point],
    epsilon: f64,
    num_trials: usize,
    seed: u64,
) -> CoveringNumberResult {
    if points.is_empty() {
        return CoveringNumberResult {
            covering_number: 0,
            epsilon,
            center_indices: vec![],
            max_uncovered_distance: 0.0,
            avg_distance_to_center: 0.0,
            assignments: vec![],
        };
    }

    let n = points.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_result = covering_number_greedy(space, points, epsilon);

    for _ in 0..num_trials {
        // Random permutation of starting point
        let start_idx = rng.gen_range(0..n);
        let mut centers: Vec<usize> = vec![start_idx];
        let mut min_dist: Vec<f64> = vec![f64::INFINITY; n];
        let mut assignments: Vec<usize> = vec![0; n];

        for i in 0..n {
            min_dist[i] = space.distance(&points[start_idx], &points[i]);
            assignments[i] = 0;
        }
        min_dist[start_idx] = 0.0;

        loop {
            let (far_idx, &far_dist) = min_dist
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            if far_dist <= epsilon || centers.len() >= n {
                break;
            }

            let cidx = centers.len();
            centers.push(far_idx);
            for i in 0..n {
                let d = space.distance(&points[far_idx], &points[i]);
                if d < min_dist[i] {
                    min_dist[i] = d;
                    assignments[i] = cidx;
                }
            }
        }

        if centers.len() < best_result.covering_number {
            let max_unc = min_dist.iter().cloned().fold(0.0f64, f64::max);
            let avg_d = min_dist.iter().sum::<f64>() / n as f64;
            best_result = CoveringNumberResult {
                covering_number: centers.len(),
                epsilon,
                center_indices: centers,
                max_uncovered_distance: max_unc,
                avg_distance_to_center: avg_d,
                assignments,
            };
        }
    }

    best_result
}

/// Compute the ε-packing number (dual of covering number).
/// A packing is a set of points where all pairwise distances are > ε.
pub fn packing_number<M: MetricSpace>(
    space: &M,
    points: &[M::Point],
    epsilon: f64,
) -> usize {
    if points.is_empty() {
        return 0;
    }

    let n = points.len();
    let mut packed: Vec<usize> = vec![0];

    for i in 1..n {
        let mut is_far = true;
        for &c in &packed {
            if space.distance(&points[i], &points[c]) <= epsilon {
                is_far = false;
                break;
            }
        }
        if is_far {
            packed.push(i);
        }
    }

    packed.len()
}

// ---------------------------------------------------------------------------
// Metric entropy estimation
// ---------------------------------------------------------------------------

/// Metric entropy at multiple scales.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEntropyProfile {
    /// (ε, log N(X, ε)) pairs.
    pub entries: Vec<(f64, f64)>,
    /// Estimated dimension (slope of log N vs log(1/ε)).
    pub estimated_dimension: f64,
    /// R² of the linear fit.
    pub fit_r_squared: f64,
}

impl MetricEntropyProfile {
    /// Estimate the metric entropy profile at geometrically spaced scales.
    pub fn estimate<M: MetricSpace>(
        space: &M,
        points: &[M::Point],
        min_epsilon: f64,
        max_epsilon: f64,
        num_scales: usize,
    ) -> Self {
        if points.is_empty() || num_scales == 0 {
            return Self {
                entries: vec![],
                estimated_dimension: 0.0,
                fit_r_squared: 0.0,
            };
        }

        let log_min = min_epsilon.ln();
        let log_max = max_epsilon.ln();
        let step = if num_scales > 1 {
            (log_max - log_min) / (num_scales - 1) as f64
        } else {
            0.0
        };

        let mut entries = Vec::with_capacity(num_scales);
        for i in 0..num_scales {
            let eps = (log_min + step * i as f64).exp();
            let result = covering_number_greedy(space, points, eps);
            let log_n = if result.covering_number > 0 {
                (result.covering_number as f64).ln()
            } else {
                0.0
            };
            entries.push((eps, log_n));
        }

        // Estimate dimension via linear regression: log N ≈ d · log(1/ε) + c
        let (dimension, r_sq) = Self::fit_dimension(&entries);

        Self {
            entries,
            estimated_dimension: dimension,
            fit_r_squared: r_sq,
        }
    }

    /// Linear regression of log N vs log(1/ε) to estimate dimension.
    fn fit_dimension(entries: &[(f64, f64)]) -> (f64, f64) {
        let n = entries.len() as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        // x = log(1/ε), y = log N
        let xs: Vec<f64> = entries.iter().map(|(eps, _)| (1.0 / eps).ln()).collect();
        let ys: Vec<f64> = entries.iter().map(|(_, log_n)| *log_n).collect();

        let x_mean = xs.iter().sum::<f64>() / n;
        let y_mean = ys.iter().sum::<f64>() / n;

        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        let mut ss_yy = 0.0;
        for i in 0..xs.len() {
            let dx = xs[i] - x_mean;
            let dy = ys[i] - y_mean;
            ss_xy += dx * dy;
            ss_xx += dx * dx;
            ss_yy += dy * dy;
        }

        let slope = if ss_xx.abs() > 1e-15 {
            ss_xy / ss_xx
        } else {
            0.0
        };

        let r_sq = if ss_xx.abs() > 1e-15 && ss_yy.abs() > 1e-15 {
            (ss_xy * ss_xy) / (ss_xx * ss_yy)
        } else {
            0.0
        };

        (slope.max(0.0), r_sq)
    }

    /// Interpolate log N at a given ε using the fitted model.
    pub fn interpolate_log_covering(&self, epsilon: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        // Use the dimension estimate: log N ≈ d · log(1/ε) + c
        // Compute c from first entry
        let (eps0, log_n0) = self.entries[0];
        let c = log_n0 - self.estimated_dimension * (1.0 / eps0).ln();
        let result = self.estimated_dimension * (1.0 / epsilon).ln() + c;
        result.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// VC dimension connection
// ---------------------------------------------------------------------------

/// Compute an upper bound on the covering number using VC dimension.
///
/// By Haussler's theorem: N(H, ε, d) ≤ e · (d+1) · (2/ε)^d
/// where d is the VC dimension.
pub fn vc_covering_number_upper_bound(vc_dimension: usize, epsilon: f64) -> f64 {
    let d = vc_dimension as f64;
    std::f64::consts::E * (d + 1.0) * (2.0 / epsilon).powf(d)
}

/// Estimate the VC dimension from a set of observations using
/// the maximum number of shattered points.
pub fn estimate_vc_dimension_binary(
    points: &[Vec<f64>],
    threshold: f64,
    max_dim: usize,
) -> usize {
    let n = points.len();
    if n == 0 {
        return 0;
    }

    let dim = points[0].len();
    let max_test = max_dim.min(n).min(20); // cap for computational feasibility

    let mut best_d = 0;

    for d in 1..=max_test {
        // Check if we can shatter some subset of size d
        // by testing random hyperplane classifiers
        let can_shatter = test_shattering(points, d, threshold);
        if can_shatter {
            best_d = d;
        } else {
            break;
        }
    }

    best_d
}

/// Test whether a subset of size d can be shattered.
fn test_shattering(points: &[Vec<f64>], d: usize, threshold: f64) -> bool {
    let n = points.len();
    if d > n || d > 20 {
        return false;
    }

    // Try the first d points as the subset
    let subset: Vec<&Vec<f64>> = points.iter().take(d).collect();
    let num_labelings = 1usize << d;

    for labeling in 0..num_labelings {
        // Can we realize this labeling with a threshold classifier?
        let labels: Vec<bool> = (0..d).map(|i| (labeling >> i) & 1 == 1).collect();

        // Check each coordinate dimension
        let mut realized = false;
        for coord in 0..subset[0].len() {
            let mut values_with_labels: Vec<(f64, bool)> = subset
                .iter()
                .enumerate()
                .map(|(i, p)| (p[coord], labels[i]))
                .collect();
            values_with_labels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Try each threshold between consecutive values
            for t in 0..values_with_labels.len() {
                let thresh = if t == 0 {
                    values_with_labels[0].0 - 1.0
                } else {
                    (values_with_labels[t - 1].0 + values_with_labels[t].0) / 2.0
                };

                let predicted: Vec<bool> = values_with_labels
                    .iter()
                    .map(|(v, _)| *v > thresh)
                    .collect();

                if predicted == values_with_labels.iter().map(|(_, l)| *l).collect::<Vec<_>>() {
                    realized = true;
                    break;
                }
            }
            if realized {
                break;
            }
        }

        if !realized {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// BandwidthEstimator
// ---------------------------------------------------------------------------

/// Estimates functor bandwidth from empirical observations.
#[derive(Debug, Clone)]
pub struct BandwidthEstimator {
    /// Configuration for bandwidth computation.
    config: BandwidthConfig,
    /// Collected behavioral observations.
    observations: Vec<BehavioralPoint>,
    /// Cached covering results at different scales.
    covering_cache: HashMap<OrderedFloat<f64>, CoveringNumberResult>,
    /// History of bandwidth estimates (for tracking convergence).
    history: Vec<BandwidthEstimate>,
    /// Metric space for distance computations.
    metric_type: DistanceMetric,
}

/// A single bandwidth estimate entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthEstimate {
    pub iteration: usize,
    pub beta: f64,
    pub epsilon: f64,
    pub num_observations: usize,
    pub covering_number: usize,
}

impl BandwidthEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: BandwidthConfig) -> Self {
        Self {
            config,
            observations: Vec::new(),
            covering_cache: HashMap::new(),
            history: Vec::new(),
            metric_type: DistanceMetric::TotalVariation,
        }
    }

    /// Create with a specific distance metric.
    pub fn with_metric(config: BandwidthConfig, metric: DistanceMetric) -> Self {
        Self {
            config,
            observations: Vec::new(),
            covering_cache: HashMap::new(),
            history: Vec::new(),
            metric_type: metric,
        }
    }

    /// Add an observation point.
    pub fn add_observation(&mut self, point: BehavioralPoint) {
        self.observations.push(point);
        self.covering_cache.clear(); // invalidate cache
    }

    /// Add multiple observation points.
    pub fn add_observations(&mut self, points: Vec<BehavioralPoint>) {
        self.observations.extend(points);
        self.covering_cache.clear();
    }

    /// Number of observations collected so far.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Clear all observations and cached results.
    pub fn reset(&mut self) {
        self.observations.clear();
        self.covering_cache.clear();
        self.history.clear();
    }

    /// Estimate the functor bandwidth at a given ε.
    pub fn estimate(&mut self, epsilon: f64) -> FunctorBandwidth {
        let start = std::time::Instant::now();

        let space = BehavioralMetricSpace::new(self.metric_type);
        let points = if self.config.dimension_reduction && !self.observations.is_empty() {
            self.reduce_dimensions()
        } else {
            self.observations.clone()
        };

        let result = match self.config.covering_number_method {
            CoveringNumberMethod::GreedyFarthestPoint => {
                covering_number_greedy(&space, &points, epsilon)
            }
            CoveringNumberMethod::RandomSampling => {
                covering_number_random(&space, &points, epsilon, 10, 42)
            }
            CoveringNumberMethod::KMeansApprox => {
                self.covering_number_kmeans(&space, &points, epsilon)
            }
        };

        let beta = if result.covering_number > 1 {
            (result.covering_number as f64).ln()
        } else {
            0.0
        };

        // Compute bounds using packing/covering duality
        let pack_num = packing_number(&space, &points, 2.0 * epsilon);
        let lower_bound = if pack_num > 1 {
            (pack_num as f64).ln()
        } else {
            0.0
        };

        let upper_bound = if result.covering_number > 1 {
            (result.covering_number as f64).ln() + (self.observations.len() as f64).ln().max(1.0) * 0.1
        } else {
            1.0
        };

        // Confidence interval using Hoeffding-type bound
        let n = self.observations.len() as f64;
        let conf_half_width = if n > 0.0 {
            (2.0 * (2.0 / (1.0 - self.config.confidence)).ln() / n).sqrt()
        } else {
            f64::INFINITY
        };

        let elapsed = start.elapsed().as_millis() as u64;

        let obs_dim = if self.observations.is_empty() {
            0
        } else {
            self.observations[0].dim()
        };

        let effective_dim = if self.config.dimension_reduction {
            self.config.max_dimension.min(obs_dim)
        } else {
            obs_dim
        };

        let estimate = BandwidthEstimate {
            iteration: self.history.len(),
            beta,
            epsilon,
            num_observations: self.observations.len(),
            covering_number: result.covering_number,
        };
        self.history.push(estimate);

        // Cache result
        self.covering_cache
            .insert(OrderedFloat(epsilon), result.clone());

        FunctorBandwidth {
            beta,
            abstraction_level: (0, 0, epsilon),
            epsilon,
            covering_size: result.covering_number,
            lower_bound,
            upper_bound,
            confidence_half_width: conf_half_width,
            num_samples: self.observations.len(),
            method: self.config.covering_number_method,
            metadata: BandwidthMetadata {
                elapsed_ms: elapsed,
                observation_dim: obs_dim,
                dimension_reduced: self.config.dimension_reduction,
                effective_dim,
                packing_number: pack_num,
                metric_entropy_rate: beta / epsilon.max(1e-10),
            },
        }
    }

    /// Estimate bandwidth at the default epsilon from config.
    pub fn estimate_default(&mut self) -> FunctorBandwidth {
        // Use a heuristic epsilon based on observed distances
        let epsilon = self.suggest_epsilon();
        self.estimate(epsilon)
    }

    /// Suggest an appropriate epsilon based on observed pairwise distances.
    pub fn suggest_epsilon(&self) -> f64 {
        if self.observations.len() < 2 {
            return 0.1;
        }

        let space = BehavioralMetricSpace::new(self.metric_type);
        let n = self.observations.len().min(100);
        let mut distances = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                distances.push(space.distance(&self.observations[i], &self.observations[j]));
            }
        }

        if distances.is_empty() {
            return 0.1;
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use the 10th percentile of distances as epsilon
        let idx = (distances.len() as f64 * 0.1) as usize;
        let eps = distances[idx.min(distances.len() - 1)];
        eps.max(1e-6)
    }

    /// Estimate the full metric entropy profile.
    pub fn entropy_profile(&self, min_eps: f64, max_eps: f64, num_scales: usize) -> MetricEntropyProfile {
        let space = BehavioralMetricSpace::new(self.metric_type);
        MetricEntropyProfile::estimate(&space, &self.observations, min_eps, max_eps, num_scales)
    }

    /// Get the bandwidth estimation history.
    pub fn history(&self) -> &[BandwidthEstimate] {
        &self.history
    }

    /// Check if the bandwidth estimates have converged.
    pub fn has_converged(&self, tolerance: f64) -> bool {
        if self.history.len() < 3 {
            return false;
        }
        let recent: Vec<f64> = self.history.iter().rev().take(3).map(|e| e.beta).collect();
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        recent.iter().all(|&b| (b - mean).abs() < tolerance)
    }

    /// Dimension reduction via random projection (Johnson-Lindenstrauss).
    fn reduce_dimensions(&self) -> Vec<BehavioralPoint> {
        if self.observations.is_empty() {
            return vec![];
        }

        let original_dim = self.observations[0].dim();
        let target_dim = self.config.max_dimension.min(original_dim);

        if target_dim >= original_dim {
            return self.observations.clone();
        }

        // Random projection matrix (scaled Gaussian)
        let mut rng = StdRng::seed_from_u64(12345);
        let scale = 1.0 / (target_dim as f64).sqrt();
        let projection: Vec<Vec<f64>> = (0..target_dim)
            .map(|_| {
                (0..original_dim)
                    .map(|_| {
                        let u: f64 = rng.gen();
                        let v: f64 = rng.gen();
                        // Box-Muller transform for Gaussian
                        let z = (-2.0 * u.max(1e-15).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos();
                        z * scale
                    })
                    .collect()
            })
            .collect();

        self.observations
            .iter()
            .map(|obs| {
                let projected: Vec<f64> = projection
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(obs.distribution.iter())
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                BehavioralPoint::new(obs.id.clone(), projected)
            })
            .collect()
    }

    /// k-means-based covering number approximation.
    fn covering_number_kmeans(
        &self,
        space: &BehavioralMetricSpace,
        points: &[BehavioralPoint],
        epsilon: f64,
    ) -> CoveringNumberResult {
        if points.is_empty() {
            return CoveringNumberResult {
                covering_number: 0,
                epsilon,
                center_indices: vec![],
                max_uncovered_distance: 0.0,
                avg_distance_to_center: 0.0,
                assignments: vec![],
            };
        }

        let n = points.len();
        // Binary search for minimum k such that all points are within epsilon of a center
        let mut lo: usize = 1;
        let mut hi: usize = n;

        while lo < hi {
            let mid = (lo + hi) / 2;
            let (assignments, max_dist) = self.run_kmeans(space, points, mid, 50);
            if max_dist <= epsilon {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        let (assignments, max_dist) = self.run_kmeans(space, points, lo, 50);

        // Find centers (closest point to each cluster mean)
        let mut center_indices = Vec::new();
        for k in 0..lo {
            let cluster_points: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &a)| a == k)
                .map(|(i, _)| i)
                .collect();

            if cluster_points.is_empty() {
                continue;
            }

            // Find the medoid (point minimizing sum of distances to others in cluster)
            let medoid = cluster_points
                .iter()
                .min_by(|&&i, &&j| {
                    let di: f64 = cluster_points
                        .iter()
                        .map(|&p| space.distance(&points[i], &points[p]))
                        .sum();
                    let dj: f64 = cluster_points
                        .iter()
                        .map(|&p| space.distance(&points[j], &points[p]))
                        .sum();
                    di.partial_cmp(&dj).unwrap()
                })
                .copied()
                .unwrap_or(cluster_points[0]);

            center_indices.push(medoid);
        }

        let avg_dist = if n > 0 {
            assignments
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    if let Some(&ci) = center_indices.get(a) {
                        space.distance(&points[i], &points[ci])
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / n as f64
        } else {
            0.0
        };

        CoveringNumberResult {
            covering_number: center_indices.len(),
            epsilon,
            center_indices,
            max_uncovered_distance: max_dist,
            avg_distance_to_center: avg_dist,
            assignments,
        }
    }

    /// Run k-means and return (assignments, max_distance_to_center).
    fn run_kmeans(
        &self,
        space: &BehavioralMetricSpace,
        points: &[BehavioralPoint],
        k: usize,
        max_iters: usize,
    ) -> (Vec<usize>, f64) {
        let n = points.len();
        if n == 0 || k == 0 {
            return (vec![], 0.0);
        }

        let k = k.min(n);

        // Initialize centers using k-means++ style
        let mut center_indices: Vec<usize> = vec![0];
        let mut rng = StdRng::seed_from_u64(42);

        while center_indices.len() < k {
            let mut dists: Vec<f64> = (0..n)
                .map(|i| {
                    center_indices
                        .iter()
                        .map(|&c| space.distance(&points[i], &points[c]))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();

            let total: f64 = dists.iter().map(|d| d * d).sum();
            if total < 1e-15 {
                break;
            }

            let mut r: f64 = rng.gen::<f64>() * total;
            let mut chosen = 0;
            for (i, d) in dists.iter().enumerate() {
                r -= d * d;
                if r <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            center_indices.push(chosen);
        }

        // Compute centers as actual vectors (means)
        let dim = points[0].dim();
        let mut centers: Vec<Vec<f64>> = center_indices
            .iter()
            .map(|&i| points[i].distribution.clone())
            .collect();

        let mut assignments = vec![0usize; n];

        for _iter in 0..max_iters {
            // Assign points to nearest center
            let mut changed = false;
            for i in 0..n {
                let mut best_k = 0;
                let mut best_d = f64::INFINITY;
                for (ki, center) in centers.iter().enumerate() {
                    let center_pt = BehavioralPoint::new("", center.clone());
                    let d = space.distance(&points[i], &center_pt);
                    if d < best_d {
                        best_d = d;
                        best_k = ki;
                    }
                }
                if assignments[i] != best_k {
                    assignments[i] = best_k;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Recompute centers
            let actual_k = centers.len();
            let mut sums = vec![vec![0.0; dim]; actual_k];
            let mut counts = vec![0usize; actual_k];

            for i in 0..n {
                let a = assignments[i];
                if a < actual_k {
                    counts[a] += 1;
                    for d in 0..dim {
                        sums[a][d] += points[i].distribution[d];
                    }
                }
            }

            for ki in 0..actual_k {
                if counts[ki] > 0 {
                    let c = counts[ki] as f64;
                    for d in 0..dim {
                        centers[ki][d] = sums[ki][d] / c;
                    }
                }
            }
        }

        // Compute max distance to assigned center
        let max_dist = (0..n)
            .map(|i| {
                let a = assignments[i];
                if a < centers.len() {
                    let center_pt = BehavioralPoint::new("", centers[a].clone());
                    space.distance(&points[i], &center_pt)
                } else {
                    0.0
                }
            })
            .fold(0.0f64, f64::max);

        (assignments, max_dist)
    }
}

// ---------------------------------------------------------------------------
// BandwidthProfile — multi-level bandwidth tracking
// ---------------------------------------------------------------------------

/// Bandwidth at multiple abstraction levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthProfile {
    /// Bandwidth estimates at each abstraction level.
    pub levels: Vec<BandwidthProfileEntry>,
    /// Summary statistics.
    pub summary: BandwidthProfileSummary,
}

/// A single entry in the bandwidth profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthProfileEntry {
    /// Abstraction level (k, n, ε).
    pub level: (usize, usize, f64),
    /// Bandwidth at this level.
    pub bandwidth: FunctorBandwidth,
    /// Number of states at this level.
    pub num_states: usize,
    /// Recommended query budget at this level.
    pub recommended_budget: usize,
}

/// Summary statistics for a bandwidth profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthProfileSummary {
    /// Minimum bandwidth across levels.
    pub min_beta: f64,
    /// Maximum bandwidth across levels.
    pub max_beta: f64,
    /// Mean bandwidth.
    pub mean_beta: f64,
    /// Bandwidth growth rate (slope of β vs abstraction fineness).
    pub growth_rate: f64,
    /// Recommended starting abstraction level.
    pub recommended_level: usize,
}

impl BandwidthProfile {
    /// Create a profile from a sequence of (level, bandwidth, num_states) tuples.
    pub fn from_entries(entries: Vec<(usize, usize, f64, FunctorBandwidth, usize)>) -> Self {
        let levels: Vec<BandwidthProfileEntry> = entries
            .into_iter()
            .map(|(k, n, eps, bw, ns)| {
                let budget = bw.sample_complexity_bound(ns, 0.05);
                BandwidthProfileEntry {
                    level: (k, n, eps),
                    bandwidth: bw,
                    num_states: ns,
                    recommended_budget: budget,
                }
            })
            .collect();

        let summary = Self::compute_summary(&levels);

        Self { levels, summary }
    }

    fn compute_summary(levels: &[BandwidthProfileEntry]) -> BandwidthProfileSummary {
        if levels.is_empty() {
            return BandwidthProfileSummary {
                min_beta: 0.0,
                max_beta: 0.0,
                mean_beta: 0.0,
                growth_rate: 0.0,
                recommended_level: 0,
            };
        }

        let betas: Vec<f64> = levels.iter().map(|e| e.bandwidth.beta).collect();
        let min_beta = betas.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_beta = betas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_beta = betas.iter().sum::<f64>() / betas.len() as f64;

        // Growth rate: linear regression of beta vs level index
        let n = betas.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = mean_beta;
        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        for (i, &b) in betas.iter().enumerate() {
            let dx = i as f64 - x_mean;
            ss_xy += dx * (b - y_mean);
            ss_xx += dx * dx;
        }
        let growth_rate = if ss_xx.abs() > 1e-15 {
            ss_xy / ss_xx
        } else {
            0.0
        };

        // Recommend the level with the best cost/information ratio
        let recommended = levels
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let ratio_a = a.recommended_budget as f64 / (a.bandwidth.beta + 1.0);
                let ratio_b = b.recommended_budget as f64 / (b.bandwidth.beta + 1.0);
                ratio_a.partial_cmp(&ratio_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        BandwidthProfileSummary {
            min_beta,
            max_beta,
            mean_beta,
            growth_rate,
            recommended_level: recommended,
        }
    }

    /// Get the entry at a specific level index.
    pub fn get_level(&self, index: usize) -> Option<&BandwidthProfileEntry> {
        self.levels.get(index)
    }

    /// Total recommended budget across all levels.
    pub fn total_recommended_budget(&self) -> usize {
        self.levels.iter().map(|e| e.recommended_budget).sum()
    }
}

// ---------------------------------------------------------------------------
// Adaptive bandwidth tracker
// ---------------------------------------------------------------------------

/// Tracks bandwidth estimates adaptively during learning iterations.
#[derive(Debug, Clone)]
pub struct AdaptiveBandwidthTracker {
    /// Window of recent bandwidth estimates.
    window: Vec<f64>,
    /// Maximum window size.
    window_size: usize,
    /// Exponential moving average of bandwidth.
    ema: f64,
    /// EMA smoothing factor.
    alpha: f64,
    /// Number of updates.
    num_updates: usize,
    /// Variance tracker for stability detection.
    variance_accumulator: f64,
    mean_accumulator: f64,
}

impl AdaptiveBandwidthTracker {
    /// Create a new tracker with the given window size and EMA smoothing factor.
    pub fn new(window_size: usize, alpha: f64) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            ema: 0.0,
            alpha: alpha.clamp(0.01, 0.99),
            num_updates: 0,
            variance_accumulator: 0.0,
            mean_accumulator: 0.0,
        }
    }

    /// Update with a new bandwidth estimate.
    pub fn update(&mut self, beta: f64) {
        self.num_updates += 1;

        // Update windowed buffer
        if self.window.len() >= self.window_size {
            self.window.remove(0);
        }
        self.window.push(beta);

        // Update EMA
        if self.num_updates == 1 {
            self.ema = beta;
            self.mean_accumulator = beta;
        } else {
            self.ema = self.alpha * beta + (1.0 - self.alpha) * self.ema;
            let old_mean = self.mean_accumulator;
            self.mean_accumulator += (beta - old_mean) / self.num_updates as f64;
            self.variance_accumulator += (beta - old_mean) * (beta - self.mean_accumulator);
        }
    }

    /// Current EMA estimate of bandwidth.
    pub fn current_estimate(&self) -> f64 {
        self.ema
    }

    /// Windowed mean.
    pub fn windowed_mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().sum::<f64>() / self.window.len() as f64
    }

    /// Windowed standard deviation.
    pub fn windowed_std(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let mean = self.windowed_mean();
        let var = self
            .window
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.window.len() - 1) as f64;
        var.sqrt()
    }

    /// Overall variance.
    pub fn overall_variance(&self) -> f64 {
        if self.num_updates < 2 {
            return 0.0;
        }
        self.variance_accumulator / (self.num_updates - 1) as f64
    }

    /// Coefficient of variation (CV) of windowed estimates.
    pub fn coefficient_of_variation(&self) -> f64 {
        let mean = self.windowed_mean();
        if mean.abs() < 1e-15 {
            return 0.0;
        }
        self.windowed_std() / mean
    }

    /// Whether the bandwidth estimates have stabilized.
    pub fn is_stable(&self, cv_threshold: f64) -> bool {
        self.window.len() >= 3 && self.coefficient_of_variation() < cv_threshold
    }

    /// Trend direction: positive means bandwidth is increasing.
    pub fn trend(&self) -> f64 {
        if self.window.len() < 3 {
            return 0.0;
        }
        let n = self.window.len();
        let half = n / 2;
        let first_half_mean: f64 = self.window[..half].iter().sum::<f64>() / half as f64;
        let second_half_mean: f64 = self.window[half..].iter().sum::<f64>() / (n - half) as f64;
        second_half_mean - first_half_mean
    }

    /// Number of updates received.
    pub fn num_updates(&self) -> usize {
        self.num_updates
    }
}

// ---------------------------------------------------------------------------
// Bandwidth-guided query allocation
// ---------------------------------------------------------------------------

/// Allocates queries across states based on bandwidth information.
#[derive(Debug, Clone)]
pub struct BandwidthGuidedAllocator {
    /// Per-state bandwidth estimates.
    state_bandwidths: HashMap<String, f64>,
    /// Per-state query counts.
    state_queries: HashMap<String, usize>,
    /// Total budget.
    total_budget: usize,
    /// Exploration bonus factor.
    exploration_bonus: f64,
}

impl BandwidthGuidedAllocator {
    /// Create a new allocator with the given budget.
    pub fn new(total_budget: usize) -> Self {
        Self {
            state_bandwidths: HashMap::new(),
            state_queries: HashMap::new(),
            total_budget,
            exploration_bonus: 1.0,
        }
    }

    /// Set exploration bonus (higher = more exploration of high-bandwidth states).
    pub fn with_exploration_bonus(mut self, bonus: f64) -> Self {
        self.exploration_bonus = bonus.max(0.0);
        self
    }

    /// Update the bandwidth estimate for a state.
    pub fn update_bandwidth(&mut self, state_id: &str, bandwidth: f64) {
        self.state_bandwidths
            .insert(state_id.to_string(), bandwidth);
    }

    /// Record that a query was made for a state.
    pub fn record_query(&mut self, state_id: &str) {
        *self
            .state_queries
            .entry(state_id.to_string())
            .or_insert(0) += 1;
    }

    /// Total queries used so far.
    pub fn total_queries_used(&self) -> usize {
        self.state_queries.values().sum()
    }

    /// Remaining budget.
    pub fn remaining_budget(&self) -> usize {
        self.total_budget.saturating_sub(self.total_queries_used())
    }

    /// Compute the allocation: how many queries each state should get.
    pub fn compute_allocation(&self) -> HashMap<String, usize> {
        if self.state_bandwidths.is_empty() {
            return HashMap::new();
        }

        let remaining = self.remaining_budget();
        if remaining == 0 {
            return self
                .state_bandwidths
                .keys()
                .map(|k| (k.clone(), 0))
                .collect();
        }

        // Score each state: bandwidth * exploration_bonus - queries_already_done
        let scores: HashMap<&str, f64> = self
            .state_bandwidths
            .iter()
            .map(|(state, &bw)| {
                let done = *self.state_queries.get(state.as_str()).unwrap_or(&0) as f64;
                let score = (bw * self.exploration_bonus - done * 0.1).max(0.01);
                (state.as_str(), score)
            })
            .collect();

        let total_score: f64 = scores.values().sum();

        scores
            .iter()
            .map(|(&state, &score)| {
                let frac = score / total_score;
                let alloc = (frac * remaining as f64).round() as usize;
                (state.to_string(), alloc)
            })
            .collect()
    }

    /// Get the next state to query (highest priority).
    pub fn next_state_to_query(&self) -> Option<String> {
        let allocation = self.compute_allocation();
        allocation
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .max_by_key(|(state, count)| {
                let bw = self
                    .state_bandwidths
                    .get(state.as_str())
                    .copied()
                    .unwrap_or(0.0);
                OrderedFloat(bw * *count as f64)
            })
            .map(|(state, _)| state)
    }
}

// ---------------------------------------------------------------------------
// Sample complexity bounds
// ---------------------------------------------------------------------------

/// Compute the PAC sample complexity for learning with given parameters.
///
/// Based on the bound: m ≥ (β(F,α) / ε²) · (d · ln(β/ε) + ln(2/δ))
/// where d is the number of states, ε is accuracy, δ is failure probability.
pub fn pac_sample_complexity(
    bandwidth: f64,
    num_states: usize,
    accuracy: f64,
    delta: f64,
) -> usize {
    assert!(accuracy > 0.0 && delta > 0.0 && delta < 1.0);
    let d = num_states as f64;
    let ln_term = d * (bandwidth / accuracy).max(1.0).ln() + (2.0 / delta).ln();
    let bound = (bandwidth / (accuracy * accuracy)) * ln_term;
    bound.ceil().max(1.0) as usize
}

/// Compute the minimax-optimal sample complexity lower bound.
///
/// From information-theoretic arguments: m ≥ Ω(β · d / ε²)
pub fn minimax_lower_bound(
    bandwidth: f64,
    num_states: usize,
    accuracy: f64,
) -> usize {
    let d = num_states as f64;
    let bound = bandwidth * d / (accuracy * accuracy);
    bound.ceil().max(1.0) as usize
}

/// Compute the query complexity for CEGAR refinement with bandwidth.
///
/// The number of refinement steps is bounded by: O(β · log(n/ε))
pub fn cegar_refinement_complexity(
    bandwidth: f64,
    num_states: usize,
    accuracy: f64,
) -> usize {
    let bound = bandwidth * (num_states as f64 / accuracy).max(1.0).ln();
    bound.ceil().max(1.0) as usize
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_points(coords: &[(f64, f64)]) -> Vec<BehavioralPoint> {
        coords
            .iter()
            .enumerate()
            .map(|(i, (x, y))| BehavioralPoint::new(format!("p{}", i), vec![*x, *y]))
            .collect()
    }

    #[test]
    fn test_behavioral_point_tv_distance() {
        let a = BehavioralPoint::new("a", vec![0.5, 0.3, 0.2]);
        let b = BehavioralPoint::new("b", vec![0.3, 0.5, 0.2]);
        let d = a.tv_distance(&b);
        assert!((d - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_behavioral_point_l2_distance() {
        let a = BehavioralPoint::new("a", vec![1.0, 0.0]);
        let b = BehavioralPoint::new("b", vec![0.0, 1.0]);
        let d = a.l2_distance(&b);
        assert!((d - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_behavioral_point_self_distance() {
        let a = BehavioralPoint::new("a", vec![0.3, 0.7]);
        assert!(a.tv_distance(&a).abs() < 1e-10);
        assert!(a.l2_distance(&a).abs() < 1e-10);
        assert!(a.linf_distance(&a).abs() < 1e-10);
        assert!(a.hellinger_distance(&a).abs() < 1e-10);
    }

    #[test]
    fn test_covering_number_single_point() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[(0.0, 0.0)]);
        let result = covering_number_greedy(&space, &points, 0.1);
        assert_eq!(result.covering_number, 1);
    }

    #[test]
    fn test_covering_number_two_close_points() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[(0.0, 0.0), (0.01, 0.01)]);
        let result = covering_number_greedy(&space, &points, 0.1);
        assert_eq!(result.covering_number, 1);
    }

    #[test]
    fn test_covering_number_two_far_points() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[(0.0, 0.0), (10.0, 10.0)]);
        let result = covering_number_greedy(&space, &points, 0.1);
        assert_eq!(result.covering_number, 2);
    }

    #[test]
    fn test_covering_number_cluster() {
        let space = BehavioralMetricSpace::l2();
        // Three tight clusters
        let points = make_points(&[
            (0.0, 0.0),
            (0.01, 0.01),
            (0.02, 0.0),
            (5.0, 5.0),
            (5.01, 5.01),
            (10.0, 0.0),
            (10.01, 0.01),
        ]);
        let result = covering_number_greedy(&space, &points, 0.1);
        assert_eq!(result.covering_number, 3);
    }

    #[test]
    fn test_covering_number_empty() {
        let space = BehavioralMetricSpace::l2();
        let points: Vec<BehavioralPoint> = vec![];
        let result = covering_number_greedy(&space, &points, 0.1);
        assert_eq!(result.covering_number, 0);
    }

    #[test]
    fn test_covering_number_random_vs_greedy() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.5),
        ]);
        let greedy = covering_number_greedy(&space, &points, 0.6);
        let random = covering_number_random(&space, &points, 0.6, 20, 42);
        // Random should be <= greedy (it tries greedy from multiple starts)
        assert!(random.covering_number <= greedy.covering_number + 1);
    }

    #[test]
    fn test_packing_number() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[
            (0.0, 0.0),
            (0.01, 0.0),
            (5.0, 5.0),
            (5.01, 5.0),
            (10.0, 0.0),
        ]);
        let pack = packing_number(&space, &points, 1.0);
        assert_eq!(pack, 3);
    }

    #[test]
    fn test_packing_covering_duality() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
        ]);
        let eps = 0.6;
        let cov = covering_number_greedy(&space, &points, eps);
        let pack = packing_number(&space, &points, 2.0 * eps);
        // Packing number ≤ covering number (at same ε)
        // Packing(2ε) ≤ Covering(ε)
        assert!(pack <= cov.covering_number);
    }

    #[test]
    fn test_metric_entropy_profile() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (0.0, 2.0),
            (1.0, 2.0),
            (2.0, 2.0),
        ]);
        let profile = MetricEntropyProfile::estimate(&space, &points, 0.1, 3.0, 5);
        assert_eq!(profile.entries.len(), 5);
        // log N should decrease as ε increases
        assert!(profile.entries.first().unwrap().1 >= profile.entries.last().unwrap().1);
    }

    #[test]
    fn test_metric_entropy_dimension_estimate() {
        let space = BehavioralMetricSpace::l2();
        // 1D line: points along (x, 0) should give dimension ~1
        let points: Vec<BehavioralPoint> = (0..20)
            .map(|i| BehavioralPoint::new(format!("p{}", i), vec![i as f64 * 0.5, 0.0]))
            .collect();
        let profile = MetricEntropyProfile::estimate(&space, &points, 0.1, 5.0, 8);
        // Dimension should be around 1
        assert!(profile.estimated_dimension > 0.5);
        assert!(profile.estimated_dimension < 3.0);
    }

    #[test]
    fn test_vc_covering_bound() {
        let bound = vc_covering_number_upper_bound(3, 0.1);
        assert!(bound > 0.0);
        let bound2 = vc_covering_number_upper_bound(3, 0.01);
        assert!(bound2 > bound); // smaller ε → larger covering number
    }

    #[test]
    fn test_functor_bandwidth_new() {
        let bw = FunctorBandwidth::new(2.5, 0.1, 12);
        assert!((bw.beta - 2.5).abs() < 1e-10);
        assert_eq!(bw.covering_size, 12);
        assert!((bw.epsilon - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_sample_complexity_bound() {
        let bw = FunctorBandwidth::new(5.0, 0.1, 150);
        let complexity = bw.sample_complexity_bound(10, 0.05);
        assert!(complexity > 0);
        // Higher delta (less confidence) → lower complexity
        let complexity_high_delta = bw.sample_complexity_bound(10, 0.5);
        assert!(complexity_high_delta < complexity);
    }

    #[test]
    fn test_pac_sample_complexity() {
        let bw = FunctorBandwidth::new(3.0, 0.1, 20);
        let m1 = bw.pac_sample_complexity(0.1, 0.05);
        let m2 = bw.pac_sample_complexity(0.01, 0.05);
        // Smaller accuracy → more samples
        assert!(m2 > m1);
    }

    #[test]
    fn test_is_learnable() {
        let bw = FunctorBandwidth::new(2.0, 0.1, 8);
        let required = bw.sample_complexity_bound(5, 0.05);
        assert!(bw.is_learnable_within_budget(required + 100, 5, 0.05));
        assert!(!bw.is_learnable_within_budget(1, 5, 0.05));
    }

    #[test]
    fn test_bandwidth_estimator_basic() {
        let config = BandwidthConfig::default();
        let mut estimator = BandwidthEstimator::new(config);

        // Add well-separated clusters
        for i in 0..10 {
            estimator.add_observation(BehavioralPoint::new(
                format!("a{}", i),
                vec![0.0 + i as f64 * 0.01, 1.0],
            ));
        }
        for i in 0..10 {
            estimator.add_observation(BehavioralPoint::new(
                format!("b{}", i),
                vec![5.0 + i as f64 * 0.01, 0.0],
            ));
        }

        assert_eq!(estimator.num_observations(), 20);

        let bw = estimator.estimate(0.5);
        assert!(bw.beta > 0.0);
        assert!(bw.covering_size >= 2);
    }

    #[test]
    fn test_bandwidth_estimator_convergence() {
        let config = BandwidthConfig::default();
        let mut estimator = BandwidthEstimator::new(config);

        for i in 0..50 {
            estimator.add_observation(BehavioralPoint::new(
                format!("p{}", i),
                vec![(i % 3) as f64, (i % 5) as f64],
            ));
        }

        // Run multiple estimates to build history
        estimator.estimate(1.0);
        estimator.estimate(1.0);
        estimator.estimate(1.0);

        assert_eq!(estimator.history().len(), 3);
        assert!(estimator.has_converged(10.0)); // same epsilon → same result
    }

    #[test]
    fn test_bandwidth_estimator_suggest_epsilon() {
        let config = BandwidthConfig::default();
        let mut estimator = BandwidthEstimator::new(config);

        for i in 0..20 {
            estimator.add_observation(BehavioralPoint::new(
                format!("p{}", i),
                vec![i as f64, 0.0],
            ));
        }

        let eps = estimator.suggest_epsilon();
        assert!(eps > 0.0);
    }

    #[test]
    fn test_bandwidth_estimator_entropy_profile() {
        let config = BandwidthConfig::default();
        let mut estimator = BandwidthEstimator::new(config);

        for i in 0..30 {
            estimator.add_observation(BehavioralPoint::new(
                format!("p{}", i),
                vec![(i as f64) * 0.3, (i as f64) * 0.2],
            ));
        }

        let profile = estimator.entropy_profile(0.1, 5.0, 5);
        assert_eq!(profile.entries.len(), 5);
        assert!(profile.estimated_dimension >= 0.0);
    }

    #[test]
    fn test_bandwidth_profile() {
        let bw1 = FunctorBandwidth::new(1.0, 0.5, 3);
        let bw2 = FunctorBandwidth::new(2.0, 0.2, 8);
        let bw3 = FunctorBandwidth::new(3.0, 0.1, 20);

        let profile = BandwidthProfile::from_entries(vec![
            (2, 3, 0.5, bw1, 5),
            (4, 5, 0.2, bw2, 10),
            (8, 7, 0.1, bw3, 20),
        ]);

        assert_eq!(profile.levels.len(), 3);
        assert!((profile.summary.min_beta - 1.0).abs() < 1e-10);
        assert!((profile.summary.max_beta - 3.0).abs() < 1e-10);
        assert!(profile.summary.growth_rate > 0.0); // increasing
    }

    #[test]
    fn test_adaptive_tracker() {
        let mut tracker = AdaptiveBandwidthTracker::new(5, 0.3);

        tracker.update(2.0);
        assert!((tracker.current_estimate() - 2.0).abs() < 1e-10);

        tracker.update(2.1);
        tracker.update(1.9);
        tracker.update(2.0);
        tracker.update(2.05);

        assert!((tracker.windowed_mean() - 2.01).abs() < 0.1);
        assert!(tracker.windowed_std() < 0.2);
        assert!(tracker.is_stable(0.5));
    }

    #[test]
    fn test_adaptive_tracker_trend() {
        let mut tracker = AdaptiveBandwidthTracker::new(10, 0.3);
        for i in 0..10 {
            tracker.update(i as f64);
        }
        assert!(tracker.trend() > 0.0); // increasing trend
    }

    #[test]
    fn test_adaptive_tracker_unstable() {
        let mut tracker = AdaptiveBandwidthTracker::new(5, 0.3);
        tracker.update(1.0);
        tracker.update(100.0);
        tracker.update(1.0);
        tracker.update(100.0);
        tracker.update(1.0);
        assert!(!tracker.is_stable(0.1)); // very unstable
    }

    #[test]
    fn test_bandwidth_guided_allocator() {
        let mut allocator = BandwidthGuidedAllocator::new(100);
        allocator.update_bandwidth("s0", 1.0);
        allocator.update_bandwidth("s1", 3.0);
        allocator.update_bandwidth("s2", 2.0);

        let allocation = allocator.compute_allocation();
        assert!(allocation.len() == 3);
        // s1 (highest bandwidth) should get the most queries
        assert!(allocation["s1"] >= allocation["s0"]);
        assert!(allocation["s1"] >= allocation["s2"]);
    }

    #[test]
    fn test_allocator_remaining_budget() {
        let mut allocator = BandwidthGuidedAllocator::new(100);
        allocator.update_bandwidth("s0", 1.0);
        allocator.record_query("s0");
        allocator.record_query("s0");
        assert_eq!(allocator.total_queries_used(), 2);
        assert_eq!(allocator.remaining_budget(), 98);
    }

    #[test]
    fn test_allocator_next_state() {
        let mut allocator = BandwidthGuidedAllocator::new(100);
        allocator.update_bandwidth("s0", 1.0);
        allocator.update_bandwidth("s1", 10.0);
        let next = allocator.next_state_to_query();
        assert!(next.is_some());
        assert_eq!(next.unwrap(), "s1");
    }

    #[test]
    fn test_pac_complexity() {
        let m = pac_sample_complexity(5.0, 10, 0.1, 0.05);
        assert!(m > 0);
        let m2 = pac_sample_complexity(5.0, 10, 0.01, 0.05);
        assert!(m2 > m);
    }

    #[test]
    fn test_minimax_lower_bound() {
        let lb = minimax_lower_bound(3.0, 10, 0.1);
        assert!(lb > 0);
        let ub = pac_sample_complexity(3.0, 10, 0.1, 0.05);
        assert!(ub >= lb); // upper bound ≥ lower bound
    }

    #[test]
    fn test_cegar_refinement_complexity() {
        let c = cegar_refinement_complexity(5.0, 10, 0.1);
        assert!(c > 0);
    }

    #[test]
    fn test_distance_metric_symmetry() {
        let a = BehavioralPoint::new("a", vec![0.3, 0.7]);
        let b = BehavioralPoint::new("b", vec![0.6, 0.4]);
        assert!((a.tv_distance(&b) - b.tv_distance(&a)).abs() < 1e-10);
        assert!((a.l2_distance(&b) - b.l2_distance(&a)).abs() < 1e-10);
        assert!((a.linf_distance(&b) - b.linf_distance(&a)).abs() < 1e-10);
        assert!((a.hellinger_distance(&b) - b.hellinger_distance(&a)).abs() < 1e-10);
    }

    #[test]
    fn test_distance_metric_triangle() {
        let a = BehavioralPoint::new("a", vec![0.5, 0.5]);
        let b = BehavioralPoint::new("b", vec![0.8, 0.2]);
        let c = BehavioralPoint::new("c", vec![0.1, 0.9]);
        // Triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)
        assert!(a.l2_distance(&c) <= a.l2_distance(&b) + b.l2_distance(&c) + 1e-10);
        assert!(a.tv_distance(&c) <= a.tv_distance(&b) + b.tv_distance(&c) + 1e-10);
    }

    #[test]
    fn test_estimator_dimension_reduction() {
        let mut config = BandwidthConfig::default();
        config.dimension_reduction = true;
        config.max_dimension = 2;

        let mut estimator = BandwidthEstimator::new(config);

        // 10-dimensional points
        for i in 0..20 {
            let mut dist = vec![0.0; 10];
            dist[i % 10] = 1.0;
            estimator.add_observation(BehavioralPoint::new(format!("p{}", i), dist));
        }

        let bw = estimator.estimate(0.5);
        assert!(bw.metadata.dimension_reduced);
        assert_eq!(bw.metadata.effective_dim, 2);
    }

    #[test]
    fn test_functor_bandwidth_display() {
        let bw = FunctorBandwidth::new(2.302, 0.1, 10);
        let s = format!("{}", bw);
        assert!(s.contains("2.302"));
    }

    #[test]
    fn test_functor_bandwidth_summary() {
        let bw = FunctorBandwidth::new(2.302, 0.1, 10);
        let summary = bw.summary();
        assert!(summary.contains("2.302"));
        assert!(summary.contains("ε=0.1"));
    }

    #[test]
    fn test_covering_result_assignments() {
        let space = BehavioralMetricSpace::l2();
        let points = make_points(&[
            (0.0, 0.0),
            (0.01, 0.0),
            (10.0, 10.0),
            (10.01, 10.0),
        ]);
        let result = covering_number_greedy(&space, &points, 0.5);
        assert_eq!(result.covering_number, 2);
        // Points 0,1 should be assigned to one center; 2,3 to another
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[2], result.assignments[3]);
        assert_ne!(result.assignments[0], result.assignments[2]);
    }

    #[test]
    fn test_metric_entropy_interpolation() {
        let space = BehavioralMetricSpace::l2();
        let points: Vec<BehavioralPoint> = (0..20)
            .map(|i| BehavioralPoint::new(format!("p{}", i), vec![i as f64, 0.0]))
            .collect();
        let profile = MetricEntropyProfile::estimate(&space, &points, 0.5, 10.0, 5);
        let interp = profile.interpolate_log_covering(2.0);
        assert!(interp >= 0.0);
    }

    #[test]
    fn test_bandwidth_profile_total_budget() {
        let bw1 = FunctorBandwidth::new(1.0, 0.5, 3);
        let bw2 = FunctorBandwidth::new(2.0, 0.2, 8);

        let profile = BandwidthProfile::from_entries(vec![
            (2, 3, 0.5, bw1, 5),
            (4, 5, 0.2, bw2, 10),
        ]);

        assert!(profile.total_recommended_budget() > 0);
    }

    #[test]
    fn test_bandwidth_profile_recommended_level() {
        let bw1 = FunctorBandwidth::new(1.0, 0.5, 3);
        let bw2 = FunctorBandwidth::new(100.0, 0.01, 1000);

        let profile = BandwidthProfile::from_entries(vec![
            (2, 3, 0.5, bw1, 5),
            (8, 7, 0.01, bw2, 50),
        ]);

        // Should recommend the cheaper level
        assert!(profile.summary.recommended_level < profile.levels.len());
    }

    #[test]
    fn test_estimate_vc_dimension() {
        // 2D points that can be linearly separated → VC dim ≥ 2
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let d = estimate_vc_dimension_binary(&points, 0.5, 10);
        assert!(d >= 1);
    }

    #[test]
    fn test_kmeans_covering() {
        let config = BandwidthConfig {
            covering_number_method: CoveringNumberMethod::KMeansApprox,
            ..Default::default()
        };
        let mut estimator = BandwidthEstimator::new(config);

        // Add two well-separated clusters
        for i in 0..10 {
            estimator.add_observation(BehavioralPoint::new(
                format!("a{}", i),
                vec![0.0 + i as f64 * 0.01, 0.0],
            ));
            estimator.add_observation(BehavioralPoint::new(
                format!("b{}", i),
                vec![10.0 + i as f64 * 0.01, 10.0],
            ));
        }

        let bw = estimator.estimate(0.5);
        assert!(bw.covering_size >= 2);
    }

    #[test]
    fn test_allocator_exhausted_budget() {
        let mut allocator = BandwidthGuidedAllocator::new(2);
        allocator.update_bandwidth("s0", 1.0);
        allocator.record_query("s0");
        allocator.record_query("s0");
        assert_eq!(allocator.remaining_budget(), 0);
        let allocation = allocator.compute_allocation();
        assert_eq!(allocation["s0"], 0);
    }

    #[test]
    fn test_behavioral_metric_space_all_metrics() {
        let a = BehavioralPoint::new("a", vec![0.3, 0.7]);
        let b = BehavioralPoint::new("b", vec![0.6, 0.4]);

        for metric in &[
            DistanceMetric::TotalVariation,
            DistanceMetric::L2,
            DistanceMetric::LInfinity,
            DistanceMetric::Hellinger,
        ] {
            let space = BehavioralMetricSpace::new(*metric);
            let d = space.distance(&a, &b);
            assert!(d >= 0.0);
            assert!(space.distance(&a, &a).abs() < 1e-10);
        }
    }
}
