//! Metric space utilities for the CABER project.
//!
//! Provides traits and concrete implementations for various distance functions,
//! metric wrappers (bounded, normalized, weighted, product), and helper utilities
//! for pairwise distance matrices, nearest-neighbor search, and triangle-inequality
//! validation.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// A metric space over points of type `Point`.
pub trait MetricSpace {
    type Point;

    /// Returns the distance between two points.  Must satisfy:
    /// 1. d(a, b) >= 0              (non-negativity)
    /// 2. d(a, b) == 0  iff a == b  (identity of indiscernibles)
    /// 3. d(a, b) == d(b, a)        (symmetry)
    /// 4. d(a, c) <= d(a, b) + d(b, c)  (triangle inequality)
    fn distance(&self, a: &Self::Point, b: &Self::Point) -> f64;

    /// Checks that `distance` satisfies the metric axioms over the supplied
    /// sample of points (symmetry, identity of indiscernibles via approximate
    /// equality, and triangle inequality).
    fn is_valid_metric(&self, points: &[Self::Point]) -> bool
    where
        Self::Point: PartialEq,
    {
        let eps = 1e-10;
        for i in 0..points.len() {
            let d_ii = self.distance(&points[i], &points[i]);
            if d_ii.abs() > eps {
                return false;
            }
            for j in (i + 1)..points.len() {
                let d_ij = self.distance(&points[i], &points[j]);
                let d_ji = self.distance(&points[j], &points[i]);

                // non-negativity
                if d_ij < -eps {
                    return false;
                }
                // symmetry
                if (d_ij - d_ji).abs() > eps {
                    return false;
                }
                // identity of indiscernibles
                if points[i] == points[j] && d_ij.abs() > eps {
                    return false;
                }

                // triangle inequality
                for k in 0..points.len() {
                    let d_ik = self.distance(&points[i], &points[k]);
                    let d_kj = self.distance(&points[k], &points[j]);
                    if d_ij > d_ik + d_kj + eps {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// 1. DiscreteMetric
// ---------------------------------------------------------------------------

/// The discrete metric: d(x,y) = 0 if x == y, 1 otherwise.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteMetric<T: PartialEq> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: PartialEq> DiscreteMetric<T> {
    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}

impl<T: PartialEq> MetricSpace for DiscreteMetric<T> {
    type Point = T;

    fn distance(&self, a: &T, b: &T) -> f64 {
        if a == b {
            0.0
        } else {
            1.0
        }
    }
}

// We need a concrete instantiation for the helper functions that work with
// `dyn MetricSpace<Point = Vec<f64>>`.  DiscreteMetric is generic, so we
// cannot use it directly there, but the other structs below are concrete.

// ---------------------------------------------------------------------------
// 2. HammingDistance
// ---------------------------------------------------------------------------

/// Hamming distance for byte slices of equal length.
///
/// Counts the number of positions at which the corresponding bytes differ.
/// Panics if the slices have different lengths.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HammingDistance;

impl MetricSpace for HammingDistance {
    type Point = Vec<u8>;

    fn distance(&self, a: &Vec<u8>, b: &Vec<u8>) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "HammingDistance requires equal-length sequences"
        );
        a.iter()
            .zip(b.iter())
            .filter(|(x, y)| x != y)
            .count() as f64
    }
}

// ---------------------------------------------------------------------------
// 3. EditDistance  (Levenshtein)
// ---------------------------------------------------------------------------

/// Levenshtein (edit) distance using the classic O(m·n) dynamic-programming
/// matrix.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EditDistance;

impl MetricSpace for EditDistance {
    type Point = String;

    fn distance(&self, a: &String, b: &String) -> f64 {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        // dp[i][j] = edit distance between a[..i] and b[..j]
        let mut dp = vec![vec![0usize; n + 1]; m + 1];

        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[m][n] as f64
    }
}

// ---------------------------------------------------------------------------
// 4. CosineSimilarityMetric
// ---------------------------------------------------------------------------

/// Cosine-based distances for `Vec<f64>` vectors.
///
/// Two modes:
/// * `CosineDistance`  – `1 - cosine_similarity`
/// * `AngularDistance` – `arccos(cosine_similarity) / π`
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CosineMode {
    CosineDistance,
    AngularDistance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CosineSimilarityMetric {
    pub mode: CosineMode,
}

impl CosineSimilarityMetric {
    pub fn cosine_distance() -> Self {
        Self {
            mode: CosineMode::CosineDistance,
        }
    }

    pub fn angular_distance() -> Self {
        Self {
            mode: CosineMode::AngularDistance,
        }
    }

    /// Raw cosine similarity in [-1, 1].
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have equal length");
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

impl MetricSpace for CosineSimilarityMetric {
    type Point = Vec<f64>;

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        let sim = Self::cosine_similarity(a, b);
        match self.mode {
            CosineMode::CosineDistance => (1.0 - sim).max(0.0),
            CosineMode::AngularDistance => sim.acos() / std::f64::consts::PI,
        }
    }
}

// ---------------------------------------------------------------------------
// 5. EuclideanMetric
// ---------------------------------------------------------------------------

/// Standard L2 (Euclidean) distance on `Vec<f64>`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EuclideanMetric;

impl MetricSpace for EuclideanMetric {
    type Point = Vec<f64>;

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have equal length");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// 6. ManhattanMetric
// ---------------------------------------------------------------------------

/// L1 (Manhattan / taxicab) distance on `Vec<f64>`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ManhattanMetric;

impl MetricSpace for ManhattanMetric {
    type Point = Vec<f64>;

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have equal length");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// 7. ChebyshevMetric
// ---------------------------------------------------------------------------

/// L∞ (Chebyshev) distance on `Vec<f64>` – the maximum absolute difference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChebyshevMetric;

impl MetricSpace for ChebyshevMetric {
    type Point = Vec<f64>;

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have equal length");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// 8. EarthMoverDistance1D
// ---------------------------------------------------------------------------

/// Earth-mover (Wasserstein-1) distance for one-dimensional distributions
/// represented as sorted samples.
///
/// For 1-D distributions the EMD equals the L1 distance between the
/// quantile functions, which we approximate by sorting both samples and
/// summing absolute differences of the CDFs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EarthMoverDistance1D;

impl MetricSpace for EarthMoverDistance1D {
    type Point = Vec<f64>;

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        // Merge all values, compute CDF difference.
        let mut a_sorted = a.clone();
        let mut b_sorted = b.clone();
        a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

        // Collect all unique thresholds.
        let mut thresholds: Vec<f64> = Vec::with_capacity(a_sorted.len() + b_sorted.len());
        thresholds.extend_from_slice(&a_sorted);
        thresholds.extend_from_slice(&b_sorted);
        thresholds.sort_by(|x, y| x.partial_cmp(y).unwrap());
        thresholds.dedup();

        if thresholds.len() <= 1 {
            // All values identical → distance 0 if same set, otherwise we
            // fall through to the integration below which yields the correct
            // answer.
            if a_sorted == b_sorted {
                return 0.0;
            }
        }

        // Integrate |CDF_a(t) - CDF_b(t)| over consecutive thresholds.
        let cdf_val = |sorted: &[f64], t: f64| -> f64 {
            let count = sorted.partition_point(|v| *v <= t);
            count as f64 / sorted.len() as f64
        };

        let mut total = 0.0_f64;
        for i in 0..thresholds.len() - 1 {
            let t = thresholds[i];
            let width = thresholds[i + 1] - t;
            let diff = (cdf_val(&a_sorted, t) - cdf_val(&b_sorted, t)).abs();
            total += diff * width;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// 9. LpMetric
// ---------------------------------------------------------------------------

/// Parameterised Lp distance for any p ≥ 1.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LpMetric {
    pub p: f64,
}

impl LpMetric {
    pub fn new(p: f64) -> Self {
        assert!(p >= 1.0, "LpMetric requires p >= 1");
        Self { p }
    }
}

impl MetricSpace for LpMetric {
    type Point = Vec<f64>;

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have equal length");
        if self.p.is_infinite() {
            // L∞
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max)
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs().powf(self.p))
                .sum::<f64>()
                .powf(1.0 / self.p)
        }
    }
}

// ---------------------------------------------------------------------------
// 10. WeightedMetric
// ---------------------------------------------------------------------------

/// Wraps another `Vec<f64>` metric and applies per-dimension weights before
/// computing the distance.
///
/// Specifically, the weighted distance is defined as
///   d_w(a, b) = inner.distance(w ⊙ a, w ⊙ b)
/// where ⊙ denotes element-wise multiplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedMetric<M> {
    pub inner: M,
    pub weights: Vec<f64>,
}

impl<M> WeightedMetric<M> {
    pub fn new(inner: M, weights: Vec<f64>) -> Self {
        Self { inner, weights }
    }

    fn apply_weights(&self, v: &Vec<f64>) -> Vec<f64> {
        assert_eq!(
            v.len(),
            self.weights.len(),
            "vector length must match weights length"
        );
        v.iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .collect()
    }
}

impl<M: MetricSpace<Point = Vec<f64>>> MetricSpace for WeightedMetric<M> {
    type Point = Vec<f64>;

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        let wa = self.apply_weights(a);
        let wb = self.apply_weights(b);
        self.inner.distance(&wa, &wb)
    }
}

// ---------------------------------------------------------------------------
// 11. BoundedMetric
// ---------------------------------------------------------------------------

/// Wraps another metric and caps the returned distance at `max_distance`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedMetric<M> {
    pub inner: M,
    pub max_distance: f64,
}

impl<M> BoundedMetric<M> {
    pub fn new(inner: M, max_distance: f64) -> Self {
        assert!(
            max_distance > 0.0,
            "max_distance must be positive"
        );
        Self {
            inner,
            max_distance,
        }
    }
}

impl<M: MetricSpace> MetricSpace for BoundedMetric<M> {
    type Point = M::Point;

    fn distance(&self, a: &Self::Point, b: &Self::Point) -> f64 {
        self.inner.distance(a, b).min(self.max_distance)
    }
}

// ---------------------------------------------------------------------------
// 12. NormalizedMetric
// ---------------------------------------------------------------------------

/// Wraps another metric and divides by a known `diameter` so that distances
/// fall in [0, 1] (assuming the inner metric never exceeds the diameter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedMetric<M> {
    pub inner: M,
    pub diameter: f64,
}

impl<M> NormalizedMetric<M> {
    pub fn new(inner: M, diameter: f64) -> Self {
        assert!(diameter > 0.0, "diameter must be positive");
        Self { inner, diameter }
    }
}

impl<M: MetricSpace> MetricSpace for NormalizedMetric<M> {
    type Point = M::Point;

    fn distance(&self, a: &Self::Point, b: &Self::Point) -> f64 {
        self.inner.distance(a, b) / self.diameter
    }
}

// ---------------------------------------------------------------------------
// 13. ProductMetric
// ---------------------------------------------------------------------------

/// Product metric combining two component metrics.
///
/// Given metric spaces (X, d₁) and (Y, d₂) the product distance is
///   d((a₁, a₂), (b₁, b₂)) = √(d₁(a₁, b₁)² + d₂(a₂, b₂)²)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductMetric<M1, M2> {
    pub metric1: M1,
    pub metric2: M2,
}

impl<M1, M2> ProductMetric<M1, M2> {
    pub fn new(metric1: M1, metric2: M2) -> Self {
        Self { metric1, metric2 }
    }
}

impl<M1, M2> MetricSpace for ProductMetric<M1, M2>
where
    M1: MetricSpace,
    M2: MetricSpace,
{
    type Point = (M1::Point, M2::Point);

    fn distance(&self, a: &Self::Point, b: &Self::Point) -> f64 {
        let d1 = self.metric1.distance(&a.0, &b.0);
        let d2 = self.metric2.distance(&a.1, &b.1);
        (d1 * d1 + d2 * d2).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Validates the triangle inequality for every triple of points under the
/// given metric.
pub fn validate_triangle_inequality(
    points: &[Vec<f64>],
    metric: &dyn MetricSpace<Point = Vec<f64>>,
) -> bool {
    let eps = 1e-10;
    let n = points.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let d_ij = metric.distance(&points[i], &points[j]);
            for k in 0..n {
                let d_ik = metric.distance(&points[i], &points[k]);
                let d_kj = metric.distance(&points[k], &points[j]);
                if d_ij > d_ik + d_kj + eps {
                    return false;
                }
            }
        }
    }
    true
}

/// Computes the full n×n pairwise distance matrix.
pub fn pairwise_distance_matrix(
    points: &[Vec<f64>],
    metric: &dyn MetricSpace<Point = Vec<f64>>,
) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = metric.distance(&points[i], &points[j]);
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    matrix
}

/// Returns the index and distance of the nearest neighbour to `query` among
/// `points`.  Panics if `points` is empty.
pub fn nearest_neighbor(
    query: &Vec<f64>,
    points: &[Vec<f64>],
    metric: &dyn MetricSpace<Point = Vec<f64>>,
) -> (usize, f64) {
    assert!(!points.is_empty(), "points must be non-empty");
    let mut best_idx = 0;
    let mut best_dist = metric.distance(query, &points[0]);
    for (i, p) in points.iter().enumerate().skip(1) {
        let d = metric.distance(query, p);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    (best_idx, best_dist)
}

// ---------------------------------------------------------------------------
// Display impls (selected)
// ---------------------------------------------------------------------------

impl fmt::Display for EuclideanMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EuclideanMetric(L2)")
    }
}

impl fmt::Display for ManhattanMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ManhattanMetric(L1)")
    }
}

impl fmt::Display for ChebyshevMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChebyshevMetric(L∞)")
    }
}

impl fmt::Display for LpMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LpMetric(p={})", self.p)
    }
}

impl<T: PartialEq> fmt::Display for DiscreteMetric<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiscreteMetric")
    }
}

impl fmt::Display for HammingDistance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HammingDistance")
    }
}

impl fmt::Display for EditDistance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EditDistance(Levenshtein)")
    }
}

impl fmt::Display for CosineSimilarityMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.mode {
            CosineMode::CosineDistance => write!(f, "CosineDistance"),
            CosineMode::AngularDistance => write!(f, "AngularDistance"),
        }
    }
}

impl fmt::Display for EarthMoverDistance1D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EarthMoverDistance1D")
    }
}

impl<M: fmt::Display> fmt::Display for WeightedMetric<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Weighted({})", self.inner)
    }
}

impl<M: fmt::Display> fmt::Display for BoundedMetric<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bounded({}, max={})", self.inner, self.max_distance)
    }
}

impl<M: fmt::Display> fmt::Display for NormalizedMetric<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Normalized({}, diam={})", self.inner, self.diameter)
    }
}

impl<M1: fmt::Display, M2: fmt::Display> fmt::Display for ProductMetric<M1, M2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Product({}, {})", self.metric1, self.metric2)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // 1. Discrete metric
    #[test]
    fn test_discrete_metric() {
        let m = DiscreteMetric::<i32>::new();
        assert!(approx_eq(m.distance(&42, &42), 0.0));
        assert!(approx_eq(m.distance(&1, &2), 1.0));
        assert!(approx_eq(m.distance(&0, &100), 1.0));
    }

    // 2. Hamming distance
    #[test]
    fn test_hamming_distance() {
        let m = HammingDistance;
        let a = vec![1u8, 0, 1, 1, 0];
        let b = vec![1, 1, 0, 1, 0];
        assert!(approx_eq(m.distance(&a, &b), 2.0));
        assert!(approx_eq(m.distance(&a, &a), 0.0));
    }

    // 3. Edit (Levenshtein) distance
    #[test]
    fn test_edit_distance() {
        let m = EditDistance;
        assert!(approx_eq(
            m.distance(&"kitten".to_string(), &"sitting".to_string()),
            3.0
        ));
        assert!(approx_eq(
            m.distance(&"".to_string(), &"abc".to_string()),
            3.0
        ));
        assert!(approx_eq(
            m.distance(&"abc".to_string(), &"abc".to_string()),
            0.0
        ));
        assert!(approx_eq(
            m.distance(&"flaw".to_string(), &"lawn".to_string()),
            2.0
        ));
    }

    // 4. Cosine distance
    #[test]
    fn test_cosine_distance() {
        let m = CosineSimilarityMetric::cosine_distance();
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // orthogonal → cosine_sim = 0 → distance = 1
        assert!(approx_eq(m.distance(&a, &b), 1.0));
        // identical → distance = 0
        assert!(approx_eq(m.distance(&a, &a), 0.0));
    }

    // 5. Angular distance
    #[test]
    fn test_angular_distance() {
        let m = CosineSimilarityMetric::angular_distance();
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // 90° → arccos(0)/π = 0.5
        assert!(approx_eq(m.distance(&a, &b), 0.5));
        // same vector → 0
        assert!(approx_eq(m.distance(&a, &a), 0.0));
    }

    // 6. Euclidean metric
    #[test]
    fn test_euclidean_metric() {
        let m = EuclideanMetric;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!(approx_eq(m.distance(&a, &b), 5.0));
        assert!(approx_eq(m.distance(&a, &a), 0.0));
    }

    // 7. Manhattan metric
    #[test]
    fn test_manhattan_metric() {
        let m = ManhattanMetric;
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(m.distance(&a, &b), 6.0));
    }

    // 8. Chebyshev metric
    #[test]
    fn test_chebyshev_metric() {
        let m = ChebyshevMetric;
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 5.0, 3.0];
        assert!(approx_eq(m.distance(&a, &b), 5.0));
    }

    // 9. Earth-mover distance (1-D)
    #[test]
    fn test_earth_mover_distance_1d() {
        let m = EarthMoverDistance1D;
        // Two identical distributions → 0
        let a = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(m.distance(&a, &a), 0.0));

        // Shifted distribution
        let b = vec![2.0, 3.0, 4.0];
        let d = m.distance(&a, &b);
        assert!(d > 0.0);
    }

    // 10. Lp metric reduces to known cases
    #[test]
    fn test_lp_metric() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let l1 = LpMetric::new(1.0);
        assert!(approx_eq(l1.distance(&a, &b), 7.0)); // L1

        let l2 = LpMetric::new(2.0);
        assert!(approx_eq(l2.distance(&a, &b), 5.0)); // L2

        let linf = LpMetric::new(f64::INFINITY);
        assert!(approx_eq(linf.distance(&a, &b), 4.0)); // L∞
    }

    // 11. Weighted metric
    #[test]
    fn test_weighted_metric() {
        let wm = WeightedMetric::new(EuclideanMetric, vec![2.0, 1.0]);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // weighted: a_w = (2,0), b_w = (0,1) → dist = sqrt(4+1) = sqrt(5)
        let expected = 5.0_f64.sqrt();
        assert!(approx_eq(wm.distance(&a, &b), expected));
    }

    // 12. Bounded metric
    #[test]
    fn test_bounded_metric() {
        let bm = BoundedMetric::new(EuclideanMetric, 3.0);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0]; // true dist = 5
        assert!(approx_eq(bm.distance(&a, &b), 3.0));

        let c = vec![1.0, 1.0]; // true dist = sqrt(2) ≈ 1.41
        assert!(approx_eq(bm.distance(&a, &c), 2.0_f64.sqrt()));
    }

    // 13. Normalized metric
    #[test]
    fn test_normalized_metric() {
        let nm = NormalizedMetric::new(EuclideanMetric, 10.0);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0]; // dist = 5
        assert!(approx_eq(nm.distance(&a, &b), 0.5));
    }

    // 14. Product metric
    #[test]
    fn test_product_metric() {
        let pm = ProductMetric::new(EuclideanMetric, ManhattanMetric);
        let a = (vec![0.0, 0.0], vec![0.0]);
        let b = (vec![3.0, 4.0], vec![2.0]);
        // d1 = 5, d2 = 2 → sqrt(25 + 4) = sqrt(29)
        let expected = 29.0_f64.sqrt();
        assert!(approx_eq(pm.distance(&a, &b), expected));
    }

    // 15. is_valid_metric (trait default)
    #[test]
    fn test_is_valid_metric_euclidean() {
        let m = EuclideanMetric;
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        assert!(m.is_valid_metric(&points));
    }

    // 16. validate_triangle_inequality helper
    #[test]
    fn test_validate_triangle_inequality() {
        let m = ManhattanMetric;
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 2.0],
            vec![3.0, 1.0],
        ];
        assert!(validate_triangle_inequality(&points, &m));
    }

    // 17. Pairwise distance matrix
    #[test]
    fn test_pairwise_distance_matrix() {
        let m = EuclideanMetric;
        let points = vec![vec![0.0], vec![1.0], vec![3.0]];
        let mat = pairwise_distance_matrix(&points, &m);
        assert_eq!(mat.len(), 3);
        assert!(approx_eq(mat[0][0], 0.0));
        assert!(approx_eq(mat[0][1], 1.0));
        assert!(approx_eq(mat[0][2], 3.0));
        assert!(approx_eq(mat[1][2], 2.0));
        // symmetry
        assert!(approx_eq(mat[1][0], mat[0][1]));
    }

    // 18. Nearest neighbor
    #[test]
    fn test_nearest_neighbor() {
        let m = EuclideanMetric;
        let points = vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
            vec![1.0, 1.0],
        ];
        let query = vec![0.5, 0.5];
        let (idx, dist) = nearest_neighbor(&query, &points, &m);
        assert_eq!(idx, 0);
        assert!(approx_eq(dist, (0.5_f64.powi(2) * 2.0).sqrt()));
    }

    // 19. Edit distance symmetry
    #[test]
    fn test_edit_distance_symmetry() {
        let m = EditDistance;
        let a = "algorithm".to_string();
        let b = "altruistic".to_string();
        assert!(approx_eq(m.distance(&a, &b), m.distance(&b, &a)));
    }

    // 20. Hamming is_valid_metric
    #[test]
    fn test_hamming_is_valid_metric() {
        let m = HammingDistance;
        let points = vec![
            vec![0u8, 0, 0],
            vec![1, 0, 0],
            vec![1, 1, 0],
            vec![1, 1, 1],
        ];
        assert!(m.is_valid_metric(&points));
    }

    // 21. Lp monotonicity in p
    #[test]
    fn test_lp_monotonicity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        let d1 = LpMetric::new(1.0).distance(&a, &b);
        let d2 = LpMetric::new(2.0).distance(&a, &b);
        let d3 = LpMetric::new(3.0).distance(&a, &b);
        // For p1 < p2: ||x||_p1 >= ||x||_p2 (in general)
        assert!(d1 >= d2 - EPS);
        assert!(d2 >= d3 - EPS);
    }

    // 22. BoundedMetric preserves metric axioms
    #[test]
    fn test_bounded_metric_valid() {
        let bm = BoundedMetric::new(EuclideanMetric, 5.0);
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 2.0],
        ];
        assert!(bm.is_valid_metric(&points));
    }

    // 23. EMD is zero for identical distributions
    #[test]
    fn test_emd_identical() {
        let m = EarthMoverDistance1D;
        let a = vec![3.0, 1.0, 2.0];
        let b = vec![2.0, 3.0, 1.0]; // same values, different order
        assert!(approx_eq(m.distance(&a, &b), 0.0));
    }

    // 24. Display traits
    #[test]
    fn test_display_traits() {
        assert_eq!(format!("{}", EuclideanMetric), "EuclideanMetric(L2)");
        assert_eq!(format!("{}", ManhattanMetric), "ManhattanMetric(L1)");
        assert_eq!(format!("{}", LpMetric::new(3.0)), "LpMetric(p=3)");
        assert_eq!(
            format!("{}", BoundedMetric::new(EuclideanMetric, 5.0)),
            "Bounded(EuclideanMetric(L2), max=5)"
        );
        assert_eq!(
            format!("{}", ProductMetric::new(EuclideanMetric, ManhattanMetric)),
            "Product(EuclideanMetric(L2), ManhattanMetric(L1))"
        );
    }
}
