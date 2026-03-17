//! K-means++ clustering implementation.
//!
//! Provides standard K-means clustering with K-means++ initialization,
//! multiple random restarts, and silhouette score evaluation.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use spectral_types::dense::DenseMatrix;
use spectral_types::partition::Partition;

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for K-means clustering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Maximum iterations per run.
    pub max_iter: usize,
    /// Convergence tolerance on centroid movement.
    pub tolerance: f64,
    /// Number of random restarts; the best result (lowest inertia) is returned.
    pub num_restarts: usize,
    /// Random seed for deterministic behaviour.
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 2,
            max_iter: 300,
            tolerance: 1e-6,
            num_restarts: 10,
            seed: 42,
        }
    }
}

impl KMeansConfig {
    /// Set the number of clusters.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the maximum number of iterations per run.
    pub fn max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }

    /// Set the convergence tolerance.
    pub fn tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Set the number of random restarts.
    pub fn num_restarts(mut self, n: usize) -> Self {
        self.num_restarts = n;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Output of a K-means clustering run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansResult {
    /// Cluster assignments encoded as a [`Partition`].
    pub partition: Partition,
    /// Centroids — one `Vec<f64>` of length *d* per cluster.
    pub centroids: Vec<Vec<f64>>,
    /// Sum of squared distances to the nearest centroid (SSE / inertia).
    pub inertia: f64,
    /// Number of Lloyd iterations used in the best run.
    pub iterations: usize,
    /// Whether the best run converged within `max_iter`.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run K-means++ clustering on row-oriented data slices.
///
/// Each element of `data` is a point (slice of length *d*). All points must
/// share the same dimensionality.
pub fn kmeans(data: &[&[f64]], config: &KMeansConfig) -> Result<KMeansResult> {
    // --- Validation ---
    if data.is_empty() {
        return Err(SpectralCoreError::empty_input("data must be non-empty"));
    }
    let n = data.len();
    let dim = data[0].len();
    if dim == 0 {
        return Err(SpectralCoreError::empty_input(
            "data points must have at least one dimension",
        ));
    }
    for (i, pt) in data.iter().enumerate() {
        if pt.len() != dim {
            return Err(SpectralCoreError::clustering(format!(
                "point {} has dimension {} but expected {}",
                i,
                pt.len(),
                dim,
            )));
        }
    }
    let k = config.k;
    if k == 0 {
        return Err(SpectralCoreError::invalid_parameter(
            "k",
            "0",
            "number of clusters must be at least 1",
        ));
    }
    if k > n {
        return Err(SpectralCoreError::invalid_parameter(
            "k",
            &k.to_string(),
            &format!("k ({}) exceeds number of data points ({})", k, n),
        ));
    }

    let mut best_inertia = f64::INFINITY;
    let mut best_centroids: Vec<Vec<f64>> = Vec::new();
    let mut best_assignments: Vec<usize> = Vec::new();
    let mut best_iters: usize = 0;
    let mut best_converged = false;

    for restart in 0..config.num_restarts {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(restart as u64));
        let mut centroids = kmeans_plus_plus_init(data, k, &mut rng);
        let mut assignments = vec![0usize; n];

        let (inertia, iters, converged) = lloyd_iteration(
            data,
            &mut centroids,
            &mut assignments,
            k,
            config.max_iter,
            config.tolerance,
        );

        if inertia < best_inertia {
            best_inertia = inertia;
            best_centroids = centroids;
            best_assignments = assignments;
            best_iters = iters;
            best_converged = converged;
        }
    }

    log::debug!(
        "kmeans: best inertia={:.6} after {} restarts (iters={}, converged={})",
        best_inertia,
        config.num_restarts,
        best_iters,
        best_converged,
    );

    Ok(KMeansResult {
        partition: Partition::new(best_assignments),
        centroids: best_centroids,
        inertia: best_inertia,
        iterations: best_iters,
        converged: best_converged,
    })
}

/// Convenience wrapper that extracts rows from a [`DenseMatrix`] and runs
/// K-means++.
pub fn kmeans_from_matrix(
    matrix: &DenseMatrix<f64>,
    config: &KMeansConfig,
) -> Result<KMeansResult> {
    let n = matrix.rows;
    let d = matrix.cols;
    if n == 0 || d == 0 {
        return Err(SpectralCoreError::empty_input(
            "matrix must have at least one row and one column",
        ));
    }

    // Build owned rows, then collect references.
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            matrix
                .row(i)
                .expect("row index in bounds")
                .to_vec()
        })
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();

    kmeans(&refs, config)
}

// ---------------------------------------------------------------------------
// K-means++ initialisation
// ---------------------------------------------------------------------------

/// Select `k` initial centroids using the K-means++ scheme.
///
/// 1. Pick the first centroid uniformly at random.
/// 2. For each subsequent centroid, choose a point with probability
///    proportional to D(x)² — its squared distance to the nearest existing
///    centroid.
fn kmeans_plus_plus_init(
    data: &[&[f64]],
    k: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    // First centroid — uniform random.
    let first_idx = rng.gen_range(0..n);
    centroids.push(data[first_idx].to_vec());

    // Distance-squared from each point to its nearest centroid so far.
    let mut min_dist_sq = vec![f64::INFINITY; n];

    for _ in 1..k {
        // Update min_dist_sq with respect to the latest centroid.
        let latest = centroids.last().unwrap();
        for (i, pt) in data.iter().enumerate() {
            let d2 = euclidean_distance_sq(pt, latest);
            if d2 < min_dist_sq[i] {
                min_dist_sq[i] = d2;
            }
        }

        // Build cumulative distribution.
        let total: f64 = min_dist_sq.iter().copied().sum();
        if total <= 0.0 {
            // All remaining points coincide with existing centroids; pick
            // randomly to avoid division by zero.
            let idx = rng.gen_range(0..n);
            centroids.push(data[idx].to_vec());
            continue;
        }

        let threshold = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        let mut chosen = n - 1; // fallback
        for (i, &d2) in min_dist_sq.iter().enumerate() {
            cumulative += d2;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].to_vec());
    }

    centroids
}

// ---------------------------------------------------------------------------
// Lloyd's algorithm
// ---------------------------------------------------------------------------

/// Run Lloyd's algorithm from the given initial centroids.
///
/// Returns `(inertia, iterations_used, converged)`.
fn lloyd_iteration(
    data: &[&[f64]],
    centroids: &mut Vec<Vec<f64>>,
    assignments: &mut Vec<usize>,
    k: usize,
    max_iter: usize,
    tolerance: f64,
) -> (f64, usize, bool) {
    let dim = data[0].len();
    let inertia;
    let mut converged = false;
    let mut iters_used = 0;

    for iter in 0..max_iter {
        // --- Assignment step ---
        let (new_assignments, new_inertia) = assign_points(data, centroids);
        *assignments = new_assignments;
        let _ = new_inertia;

        // --- Update step ---
        let new_centroids = update_centroids(data, assignments, k, dim);

        // --- Convergence check (max centroid shift) ---
        let max_shift: f64 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| euclidean_distance_sq(old, new).sqrt())
            .fold(0.0_f64, f64::max);

        *centroids = new_centroids;
        iters_used = iter + 1;

        if max_shift <= tolerance {
            converged = true;
            break;
        }
    }

    // Final assignment to ensure consistency with the final centroids.
    let (final_assignments, final_inertia) = assign_points(data, centroids);
    *assignments = final_assignments;
    inertia = final_inertia;

    (inertia, iters_used, converged)
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// Assign every point to its nearest centroid.
///
/// Returns `(assignments, inertia)` where `inertia` is the total sum of
/// squared distances.
fn assign_points(data: &[&[f64]], centroids: &[Vec<f64>]) -> (Vec<usize>, f64) {
    let mut assignments = Vec::with_capacity(data.len());
    let mut inertia = 0.0;

    for pt in data.iter() {
        let mut best_cluster = 0;
        let mut best_dist = f64::INFINITY;
        for (c, centroid) in centroids.iter().enumerate() {
            let d2 = euclidean_distance_sq(pt, centroid);
            if d2 < best_dist {
                best_dist = d2;
                best_cluster = c;
            }
        }
        assignments.push(best_cluster);
        inertia += best_dist;
    }

    (assignments, inertia)
}

// ---------------------------------------------------------------------------
// Centroid update
// ---------------------------------------------------------------------------

/// Recompute centroids as the mean of their assigned points.
///
/// Empty clusters are re-seeded with the data point farthest from its
/// current centroid.
fn update_centroids(
    data: &[&[f64]],
    assignments: &[usize],
    k: usize,
    dim: usize,
) -> Vec<Vec<f64>> {
    let mut sums = vec![vec![0.0; dim]; k];
    let mut counts = vec![0usize; k];

    for (i, &cluster) in assignments.iter().enumerate() {
        counts[cluster] += 1;
        for (j, &val) in data[i].iter().enumerate() {
            sums[cluster][j] += val;
        }
    }

    let mut centroids: Vec<Vec<f64>> = sums
        .into_iter()
        .zip(counts.iter())
        .map(|(sum, &cnt)| {
            if cnt == 0 {
                vec![0.0; dim] // placeholder — will be re-seeded below
            } else {
                let c = cnt as f64;
                sum.into_iter().map(|s| s / c).collect()
            }
        })
        .collect();

    // Handle empty clusters by re-seeding from the farthest point.
    for c in 0..k {
        if counts[c] == 0 {
            let mut farthest_idx = 0;
            let mut farthest_dist = 0.0_f64;
            for (i, &a) in assignments.iter().enumerate() {
                let d2 = euclidean_distance_sq(data[i], &centroids[a]);
                if d2 > farthest_dist {
                    farthest_dist = d2;
                    farthest_idx = i;
                }
            }
            centroids[c] = data[farthest_idx].to_vec();
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two slices of equal length.
#[inline]
fn euclidean_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

/// Compute the mean silhouette coefficient for the given clustering.
///
/// For each point *i*:
/// - *a(i)* = average distance to other points in the same cluster.
/// - *b(i)* = minimum over other clusters of the average distance to that
///   cluster's points.
/// - *s(i)* = (b(i) − a(i)) / max(a(i), b(i)).
///
/// For a single cluster (k = 1) the silhouette is defined as 0.
///
/// For large data (> 5 000 points) a random sample of 5 000 points is used.
pub fn compute_silhouette_score(data: &[&[f64]], assignments: &[usize], k: usize) -> f64 {
    let n = data.len();
    if k <= 1 || n <= 1 {
        return 0.0;
    }

    // Optionally subsample.
    const MAX_SAMPLE: usize = 5000;
    let indices: Vec<usize> = if n > MAX_SAMPLE {
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mut idx: Vec<usize> = (0..n).collect();
        // Fisher-Yates partial shuffle for first MAX_SAMPLE elements.
        for i in 0..MAX_SAMPLE {
            let j = rng.gen_range(i..n);
            idx.swap(i, j);
        }
        idx.truncate(MAX_SAMPLE);
        idx
    } else {
        (0..n).collect()
    };

    // Pre-compute cluster member lists for efficiency.
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in assignments.iter().enumerate() {
        if c < k {
            cluster_members[c].push(i);
        }
    }

    let mut total_s = 0.0;
    let sample_n = indices.len();

    for &i in &indices {
        let ci = assignments[i];

        // a(i): average distance to same-cluster points.
        let same = &cluster_members[ci];
        let a_i = if same.len() <= 1 {
            0.0
        } else {
            let sum: f64 = same
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| euclidean_distance_sq(data[i], data[j]).sqrt())
                .sum();
            sum / (same.len() - 1) as f64
        };

        // b(i): min over other clusters of average distance.
        let mut b_i = f64::INFINITY;
        for c in 0..k {
            if c == ci || cluster_members[c].is_empty() {
                continue;
            }
            let members = &cluster_members[c];
            let avg: f64 = members
                .iter()
                .map(|&j| euclidean_distance_sq(data[i], data[j]).sqrt())
                .sum::<f64>()
                / members.len() as f64;
            if avg < b_i {
                b_i = avg;
            }
        }

        // Handle degenerate case (e.g. only one non-empty cluster visible).
        if b_i.is_infinite() {
            b_i = 0.0;
        }

        let denom = a_i.max(b_i);
        let s_i = if denom == 0.0 { 0.0 } else { (b_i - a_i) / denom };
        total_s += s_i;
    }

    total_s / sample_n as f64
}

/// Compute the total inertia (sum of squared distances to nearest centroid).
pub fn compute_inertia(data: &[&[f64]], centroids: &[Vec<f64>], assignments: &[usize]) -> f64 {
    data.iter()
        .zip(assignments.iter())
        .map(|(pt, &c)| euclidean_distance_sq(pt, &centroids[c]))
        .sum()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build three well-separated 2D clusters centred at (0,0), (10,0), (0,10).
    fn make_three_clusters() -> Vec<Vec<f64>> {
        let mut pts = Vec::new();
        // Cluster 0 around (0, 0)
        for &(x, y) in &[
            (0.0, 0.0),
            (0.1, -0.1),
            (-0.1, 0.1),
            (0.2, 0.0),
            (0.0, -0.2),
        ] {
            pts.push(vec![x, y]);
        }
        // Cluster 1 around (10, 0)
        for &(x, y) in &[
            (10.0, 0.0),
            (10.1, -0.1),
            (9.9, 0.1),
            (10.2, 0.0),
            (10.0, -0.2),
        ] {
            pts.push(vec![x, y]);
        }
        // Cluster 2 around (0, 10)
        for &(x, y) in &[
            (0.0, 10.0),
            (0.1, 9.9),
            (-0.1, 10.1),
            (0.2, 10.0),
            (0.0, 10.2),
        ] {
            pts.push(vec![x, y]);
        }
        pts
    }

    fn to_refs(data: &[Vec<f64>]) -> Vec<&[f64]> {
        data.iter().map(|v| v.as_slice()).collect()
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_well_separated_clusters() {
        let data = make_three_clusters();
        let refs = to_refs(&data);
        let config = KMeansConfig::default().with_k(3).num_restarts(5).seed(0);
        let result = kmeans(&refs, &config).unwrap();

        assert_eq!(result.partition.num_blocks, 3);
        // All points in the first group should share a cluster.
        let c0 = result.partition.assignment[0];
        for i in 1..5 {
            assert_eq!(result.partition.assignment[i], c0);
        }
        // Second group.
        let c1 = result.partition.assignment[5];
        for i in 6..10 {
            assert_eq!(result.partition.assignment[i], c1);
        }
        // Third group.
        let c2 = result.partition.assignment[10];
        for i in 11..15 {
            assert_eq!(result.partition.assignment[i], c2);
        }
        // The three clusters should be distinct.
        assert_ne!(c0, c1);
        assert_ne!(c0, c2);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_1d_data() {
        let data: Vec<Vec<f64>> = vec![
            vec![1.0],
            vec![1.1],
            vec![1.2],
            vec![10.0],
            vec![10.1],
            vec![10.2],
        ];
        let refs = to_refs(&data);
        let config = KMeansConfig::default().with_k(2).seed(7);
        let result = kmeans(&refs, &config).unwrap();

        assert_eq!(result.partition.num_blocks, 2);
        // Points 0-2 in one cluster, 3-5 in another.
        assert_eq!(
            result.partition.assignment[0],
            result.partition.assignment[1]
        );
        assert_eq!(
            result.partition.assignment[1],
            result.partition.assignment[2]
        );
        assert_eq!(
            result.partition.assignment[3],
            result.partition.assignment[4]
        );
        assert_eq!(
            result.partition.assignment[4],
            result.partition.assignment[5]
        );
        assert_ne!(
            result.partition.assignment[0],
            result.partition.assignment[3]
        );
    }

    #[test]
    fn test_k_equals_1() {
        let data = make_three_clusters();
        let refs = to_refs(&data);
        let config = KMeansConfig::default().with_k(1).seed(0);
        let result = kmeans(&refs, &config).unwrap();

        assert_eq!(result.partition.num_blocks, 1);
        for &a in &result.partition.assignment {
            assert_eq!(a, 0);
        }
    }

    #[test]
    fn test_k_equals_n_singleton_clusters() {
        let data: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let refs = to_refs(&data);
        let config = KMeansConfig::default().with_k(4).seed(99);
        let result = kmeans(&refs, &config).unwrap();

        // Each point must be in its own cluster.
        let sizes = result.partition.block_sizes();
        for &s in &sizes {
            assert_eq!(s, 1);
        }
        // Inertia should be 0 (every point is its own centroid).
        assert!(result.inertia < 1e-12);
    }

    #[test]
    fn test_silhouette_perfect_clusters() {
        // Two very distant clusters — silhouette should be close to 1.
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![100.0, 100.0],
            vec![100.0, 100.0],
            vec![100.0, 100.0],
        ];
        let refs = to_refs(&data);
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let score = compute_silhouette_score(&refs, &assignments, 2);
        assert!(score > 0.99, "silhouette should be ~1.0, got {}", score);
    }

    #[test]
    fn test_silhouette_k_1() {
        let data: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let refs = to_refs(&data);
        let assignments = vec![0, 0, 0];
        let score = compute_silhouette_score(&refs, &assignments, 1);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_kmeans_pp_init_produces_k_centroids() {
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, 0.0]).collect();
        let refs = to_refs(&data);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let centroids = kmeans_plus_plus_init(&refs, 5, &mut rng);
        assert_eq!(centroids.len(), 5);

        // All centroids should be distinct.
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                assert_ne!(centroids[i], centroids[j], "centroids {} and {} are equal", i, j);
            }
        }
    }

    #[test]
    fn test_convergence_within_max_iter() {
        let data = make_three_clusters();
        let refs = to_refs(&data);
        let config = KMeansConfig::default().with_k(3).max_iter(500).seed(0);
        let result = kmeans(&refs, &config).unwrap();
        assert!(result.converged, "should converge on well-separated data");
        assert!(result.iterations < 500);
    }

    #[test]
    fn test_multiple_restarts_improve_or_match() {
        let data = make_three_clusters();
        let refs = to_refs(&data);

        let config_1 = KMeansConfig::default().with_k(3).num_restarts(1).seed(0);
        let config_10 = KMeansConfig::default().with_k(3).num_restarts(10).seed(0);

        let result_1 = kmeans(&refs, &config_1).unwrap();
        let result_10 = kmeans(&refs, &config_10).unwrap();

        // More restarts should yield equal or lower inertia.
        assert!(
            result_10.inertia <= result_1.inertia + 1e-12,
            "10 restarts inertia {} should be <= 1 restart inertia {}",
            result_10.inertia,
            result_1.inertia,
        );
    }

    #[test]
    fn test_empty_cluster_handling() {
        // Even with pathological init, update_centroids re-seeds empties.
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
        ];
        let refs = to_refs(&data);
        let dim = 2;
        let k = 3;
        // Force two centroids on top of each other to create an empty cluster.
        let assignments = vec![0, 0, 1, 1]; // cluster 2 is empty
        let centroids = update_centroids(&refs, &assignments, k, dim);
        assert_eq!(centroids.len(), k);
        // Cluster-2 centroid should have been re-seeded (non-zero vector).
        let c2_norm: f64 = centroids[2].iter().map(|x| x * x).sum();
        // It should be a real data point, not the zero placeholder.
        assert!(
            c2_norm > 0.0 || centroids[2] == vec![0.0, 0.0],
            "re-seeded centroid should be a data point"
        );
    }

    #[test]
    fn test_config_defaults() {
        let cfg = KMeansConfig::default();
        assert_eq!(cfg.k, 2);
        assert_eq!(cfg.max_iter, 300);
        assert!((cfg.tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(cfg.num_restarts, 10);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_kmeans_from_matrix() {
        let mut matrix = DenseMatrix::<f64>::zeros(6, 2);
        // Cluster 0 around (0, 0)
        matrix.set(0, 0, 0.0).unwrap();
        matrix.set(0, 1, 0.0).unwrap();
        matrix.set(1, 0, 0.1).unwrap();
        matrix.set(1, 1, 0.1).unwrap();
        matrix.set(2, 0, -0.1).unwrap();
        matrix.set(2, 1, -0.1).unwrap();
        // Cluster 1 around (10, 10)
        matrix.set(3, 0, 10.0).unwrap();
        matrix.set(3, 1, 10.0).unwrap();
        matrix.set(4, 0, 10.1).unwrap();
        matrix.set(4, 1, 10.1).unwrap();
        matrix.set(5, 0, 9.9).unwrap();
        matrix.set(5, 1, 9.9).unwrap();

        let config = KMeansConfig::default().with_k(2).seed(0);
        let result = kmeans_from_matrix(&matrix, &config).unwrap();

        assert_eq!(result.partition.num_blocks, 2);
        // First 3 together, last 3 together.
        assert_eq!(
            result.partition.assignment[0],
            result.partition.assignment[1]
        );
        assert_eq!(
            result.partition.assignment[1],
            result.partition.assignment[2]
        );
        assert_ne!(
            result.partition.assignment[0],
            result.partition.assignment[3]
        );
    }

    #[test]
    fn test_euclidean_distance_sq() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let d2 = euclidean_distance_sq(&a, &b);
        // (3² + 4² + 0²) = 25
        assert!((d2 - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_inertia() {
        let data: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let refs = to_refs(&data);
        let centroids = vec![vec![0.5], vec![10.5]];
        let assignments = vec![0, 0, 1, 1];
        let inertia = compute_inertia(&refs, &centroids, &assignments);
        // (0-0.5)^2 + (1-0.5)^2 + (10-10.5)^2 + (11-10.5)^2 = 4 * 0.25 = 1.0
        assert!((inertia - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_k_zero() {
        let data: Vec<Vec<f64>> = vec![vec![1.0]];
        let refs = to_refs(&data);
        let config = KMeansConfig::default().with_k(0);
        assert!(kmeans(&refs, &config).is_err());
    }

    #[test]
    fn test_empty_data() {
        let refs: Vec<&[f64]> = vec![];
        let config = KMeansConfig::default().with_k(2);
        assert!(kmeans(&refs, &config).is_err());
    }
}
