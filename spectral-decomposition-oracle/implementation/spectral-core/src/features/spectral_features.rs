//! Compute the 8 spectral features from eigenvalues, eigenvectors, and optional
//! Laplacian / partition data.
//!
//! Each public function returns `f64`, using `f64::NAN` for undefined /
//! degenerate cases.  The entry-point [`compute_all_spectral_features`] bundles
//! every feature into a [`SpectralFeatures`] struct.

use spectral_types::dense::DenseMatrix;
use spectral_types::features::SpectralFeatures;
use spectral_types::partition::Partition;
use spectral_types::sparse::CsrMatrix;

use crate::error::{Result, SpectralCoreError};

// ───────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────

/// Eigenvalues smaller than this are treated as zero.
const ZERO_TOL: f64 = 1e-10;

/// Maximum k-means iterations used in block-separability.
const KMEANS_MAX_ITER: usize = 300;

// ───────────────────────────────────────────────────────────────────
// SF1 – Spectral gap  (λ₂ of the normalized Laplacian)
// ───────────────────────────────────────────────────────────────────

/// Return the second-smallest eigenvalue (the spectral gap γ₂).
///
/// `eigenvalues` **must** be sorted in ascending order.  If fewer than two
/// eigenvalues are supplied the result is `NaN`.
pub fn compute_spectral_gap(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.len() < 2 {
        return f64::NAN;
    }
    eigenvalues[1]
}

// ───────────────────────────────────────────────────────────────────
// SF2 – Spectral-gap ratio   δ² / γ₂²
// ───────────────────────────────────────────────────────────────────

/// Compute the spectral-gap ratio `coupling_energy / spectral_gap²`.
///
/// Returns `NaN` when `spectral_gap` is below [`ZERO_TOL`].
pub fn compute_spectral_gap_ratio(spectral_gap: f64, coupling_energy: f64) -> f64 {
    if spectral_gap.abs() < ZERO_TOL {
        return f64::NAN;
    }
    coupling_energy / (spectral_gap * spectral_gap)
}

// ───────────────────────────────────────────────────────────────────
// SF3 – Eigenvalue decay rate  β  (exponential fit)
// ───────────────────────────────────────────────────────────────────

/// Fit λ_i ≈ a·exp(−β·i) by ordinary least-squares on log(λ_i) vs i for
/// every *positive* eigenvalue.  Returns β (≥ 0).
///
/// If fewer than 2 positive eigenvalues exist, returns `0.0`.
pub fn compute_eigenvalue_decay_rate(eigenvalues: &[f64]) -> f64 {
    // Collect (index, log λ) for positive eigenvalues.
    let points: Vec<(f64, f64)> = eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i as f64, v.ln()))
        .collect();

    if points.len() < 2 {
        return 0.0;
    }

    // Least-squares:  y = a + b·x   →  b = (NΣxy − ΣxΣy) / (NΣx² − (Σx)²)
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    // β is the *negative* of the slope (decay, so slope < 0 → β > 0).
    let beta = -slope;
    if beta < 0.0 { 0.0 } else { beta }
}

// ───────────────────────────────────────────────────────────────────
// SF4 – Fiedler-vector localization entropy
// ───────────────────────────────────────────────────────────────────

/// H_F = −Σ p_i·ln(p_i),  where p_i = v₂[i]² / ‖v₂‖².
///
/// The Fiedler vector is column 1 of `eigenvectors` (n × k, column-major by
/// convention in `DenseMatrix` which is row-major: row i, col j → data[i*cols+j]).
///
/// Returns `NaN` when the Fiedler vector is all-zero.
pub fn compute_fiedler_entropy(eigenvectors: &DenseMatrix<f64>) -> f64 {
    if eigenvectors.cols < 2 || eigenvectors.rows == 0 {
        return f64::NAN;
    }

    // Extract the Fiedler vector (column index 1).
    let n = eigenvectors.rows;
    let fiedler: Vec<f64> = (0..n)
        .map(|i| eigenvectors.data[i * eigenvectors.cols + 1])
        .collect();

    let norm_sq: f64 = fiedler.iter().map(|v| v * v).sum();
    if norm_sq < ZERO_TOL {
        return f64::NAN;
    }

    let mut entropy = 0.0_f64;
    for &v in &fiedler {
        let p = (v * v) / norm_sq;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

// ───────────────────────────────────────────────────────────────────
// SF5 – Algebraic connectivity ratio  γ₂ / γ_k
// ───────────────────────────────────────────────────────────────────

/// γ₂ / γ_k where γ_k is the k-th eigenvalue (1-indexed).
///
/// Returns `NaN` when `k > eigenvalues.len()` or when γ_k ≈ 0.
pub fn compute_algebraic_connectivity_ratio(eigenvalues: &[f64], k: usize) -> f64 {
    if eigenvalues.len() < 2 || k == 0 || k > eigenvalues.len() {
        return f64::NAN;
    }
    let gamma_2 = eigenvalues[1];
    let gamma_k = eigenvalues[k - 1];
    if gamma_k.abs() < ZERO_TOL {
        return f64::NAN;
    }
    gamma_2 / gamma_k
}

// ───────────────────────────────────────────────────────────────────
// SF6 – Coupling energy   δ²
// ───────────────────────────────────────────────────────────────────

/// Compute coupling energy δ² = ‖L − L_block‖_F².
///
/// If a partition is supplied the block-diagonal Laplacian is constructed from
/// the partition and the full Laplacian, and the squared Frobenius norm of the
/// difference is returned.
///
/// Otherwise a proxy is used: sum of the smallest 10% of positive eigenvalues
/// (or at least the single smallest positive eigenvalue).
pub fn compute_coupling_energy(
    eigenvalues: &[f64],
    laplacian: Option<&CsrMatrix<f64>>,
    partition: Option<&Partition>,
) -> f64 {
    if let (Some(lap), Some(part)) = (laplacian, partition) {
        coupling_energy_from_partition(lap, part)
    } else {
        coupling_energy_proxy(eigenvalues)
    }
}

/// Exact coupling energy via block-diagonal Laplacian subtraction.
fn coupling_energy_from_partition(laplacian: &CsrMatrix<f64>, partition: &Partition) -> f64 {
    let n = laplacian.rows;
    if n == 0 || partition.assignment.is_empty() {
        return 0.0;
    }

    // Build the block-diagonal Laplacian: keep only entries (i,j) where
    // partition.assignment[i] == partition.assignment[j].
    let mut frob_sq = 0.0_f64;

    for i in 0..n {
        let start = laplacian.row_ptr[i];
        let end = laplacian.row_ptr[i + 1];

        let block_i = if i < partition.assignment.len() {
            partition.assignment[i]
        } else {
            usize::MAX
        };

        for idx in start..end {
            let j = laplacian.col_ind[idx];
            let v = laplacian.values[idx];

            let block_j = if j < partition.assignment.len() {
                partition.assignment[j]
            } else {
                usize::MAX
            };

            // Off-block entries: the block-diagonal version has 0, so the
            // difference is `v`.
            if block_i != block_j {
                frob_sq += v * v;
            }
            // On-block entries: the block-diagonal version has the same value,
            // so the difference is 0.  Nothing to add.
        }
    }

    frob_sq
}

/// Proxy coupling energy: sum of the smallest 10 % of positive eigenvalues.
fn coupling_energy_proxy(eigenvalues: &[f64]) -> f64 {
    let positive: Vec<f64> = eigenvalues.iter().copied().filter(|&v| v > ZERO_TOL).collect();
    if positive.is_empty() {
        return 0.0;
    }
    let count = (positive.len() as f64 * 0.1).ceil().max(1.0) as usize;
    // `positive` inherits the ascending sort of `eigenvalues`.
    positive.iter().take(count).sum()
}

// ───────────────────────────────────────────────────────────────────
// SF7 – Block separability (silhouette on eigenvector rows)
// ───────────────────────────────────────────────────────────────────

/// Silhouette score after k-means++ clustering on the rows of the n × k
/// eigenvector matrix.
///
/// Returns `NaN` when `num_clusters < 2`, `num_clusters >= n`, or the matrix
/// is empty.
pub fn compute_block_separability(
    eigenvectors: &DenseMatrix<f64>,
    num_clusters: usize,
) -> f64 {
    let n = eigenvectors.rows;
    let dim = eigenvectors.cols;

    if n == 0 || dim == 0 || num_clusters < 2 || num_clusters >= n {
        return f64::NAN;
    }

    // Extract row vectors.
    let rows: Vec<&[f64]> = (0..n)
        .map(|i| &eigenvectors.data[i * dim..(i + 1) * dim])
        .collect();

    // k-means++ clustering.
    let labels = kmeans_pp(&rows, num_clusters, dim);

    // Silhouette score.
    silhouette_score(&rows, &labels, num_clusters, dim)
}

// ───────────────────────────────────────────────────────────────────
// SF8 – Effective spectral dimension
// ───────────────────────────────────────────────────────────────────

/// d_eff = (Σ λ_i)² / Σ λ_i².
///
/// Returns `NaN` when all eigenvalues are zero.
pub fn compute_effective_dimension(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = eigenvalues.iter().sum();
    let sum_sq: f64 = eigenvalues.iter().map(|v| v * v).sum();

    if sum_sq < ZERO_TOL {
        return f64::NAN;
    }

    (sum * sum) / sum_sq
}

// ───────────────────────────────────────────────────────────────────
// Combined entry-point
// ───────────────────────────────────────────────────────────────────

/// Compute all 8 spectral features, collecting them into a
/// [`SpectralFeatures`] struct.
///
/// # Arguments
///
/// * `eigenvalues`  – sorted ascending eigenvalue sequence.
/// * `eigenvectors` – n × k matrix whose columns are eigenvectors.
/// * `laplacian`    – optional sparse Laplacian (used for exact coupling energy).
/// * `partition`    – optional block partition (used for exact coupling energy).
/// * `k`            – number of clusters / eigenvalue index for certain features.
///
/// # Errors
///
/// Returns [`SpectralCoreError::FeatureExtraction`] only on unrecoverable
/// internal failures (e.g. dimension mismatch between eigenvectors and
/// eigenvalues).
pub fn compute_all_spectral_features(
    eigenvalues: &[f64],
    eigenvectors: &DenseMatrix<f64>,
    laplacian: Option<&CsrMatrix<f64>>,
    partition: Option<&Partition>,
    k: usize,
) -> Result<SpectralFeatures> {
    // Validate consistency.
    if !eigenvalues.is_empty()
        && eigenvectors.rows > 0
        && eigenvectors.cols != eigenvalues.len()
    {
        return Err(SpectralCoreError::feature_extraction(format!(
            "eigenvector column count ({}) != eigenvalue count ({})",
            eigenvectors.cols,
            eigenvalues.len()
        )));
    }

    let spectral_gap = compute_spectral_gap(eigenvalues);

    let coupling_energy = compute_coupling_energy(eigenvalues, laplacian, partition);

    let spectral_gap_ratio = compute_spectral_gap_ratio(spectral_gap, coupling_energy);

    let eigenvalue_decay_rate = compute_eigenvalue_decay_rate(eigenvalues);

    let fiedler_localization_entropy = compute_fiedler_entropy(eigenvectors);

    let k_eff = if k == 0 { eigenvalues.len() } else { k };
    let algebraic_connectivity_ratio =
        compute_algebraic_connectivity_ratio(eigenvalues, k_eff);

    let num_clusters = if k_eff < 2 { 2 } else { k_eff };
    let block_separability_index =
        compute_block_separability(eigenvectors, num_clusters);

    let effective_spectral_dimension = compute_effective_dimension(eigenvalues);

    Ok(SpectralFeatures {
        spectral_gap,
        spectral_gap_ratio,
        eigenvalue_decay_rate,
        fiedler_localization_entropy,
        algebraic_connectivity_ratio,
        coupling_energy,
        block_separability_index,
        effective_spectral_dimension,
    })
}

// ═══════════════════════════════════════════════════════════════════
//  Internal helpers – k-means++ on small data
// ═══════════════════════════════════════════════════════════════════

/// Squared Euclidean distance between two slices.
fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Deterministic seeded RNG (xorshift64) – avoids an external crate
/// dependency inside a hot loop.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// k-means++ initialization: choose `k` initial centres from `rows`.
fn kmeans_pp_init(rows: &[&[f64]], k: usize, dim: usize, rng: &mut SimpleRng) -> Vec<Vec<f64>> {
    let n = rows.len();
    let mut centres: Vec<Vec<f64>> = Vec::with_capacity(k);

    // First centre – pick uniformly at random.
    let first = (rng.next_u64() as usize) % n;
    centres.push(rows[first].to_vec());

    // Weighted sampling for subsequent centres.
    let mut dists = vec![f64::MAX; n];

    for _ in 1..k {
        // Update distances to nearest existing centre.
        for (i, row) in rows.iter().enumerate() {
            let d = dist_sq(row, centres.last().unwrap());
            if d < dists[i] {
                dists[i] = d;
            }
        }

        let total: f64 = dists.iter().sum();
        if total < ZERO_TOL {
            // Degenerate – all points coincide.  Duplicate last centre.
            centres.push(centres.last().unwrap().clone());
            continue;
        }

        let threshold = rng.next_f64() * total;
        let mut cumulative = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centres.push(rows[chosen].to_vec());
    }

    // Pad if we somehow ended up short (should not happen).
    while centres.len() < k {
        centres.push(vec![0.0; dim]);
    }

    centres
}

/// Assign each point to the nearest centre, returning cluster labels.
fn assign_labels(rows: &[&[f64]], centres: &[Vec<f64>]) -> Vec<usize> {
    rows.iter()
        .map(|row| {
            centres
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, dist_sq(row, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci)
                .unwrap_or(0)
        })
        .collect()
}

/// Recompute centres as the mean of assigned points.
fn recompute_centres(
    rows: &[&[f64]],
    labels: &[usize],
    k: usize,
    dim: usize,
) -> Vec<Vec<f64>> {
    let mut sums = vec![vec![0.0_f64; dim]; k];
    let mut counts = vec![0usize; k];

    for (i, &label) in labels.iter().enumerate() {
        counts[label] += 1;
        for (d, val) in rows[i].iter().enumerate() {
            sums[label][d] += val;
        }
    }

    sums.iter()
        .zip(counts.iter())
        .map(|(s, &c)| {
            if c == 0 {
                s.clone()
            } else {
                let cf = c as f64;
                s.iter().map(|v| v / cf).collect()
            }
        })
        .collect()
}

/// Run k-means++ on `rows` (each of length `dim`), returning cluster labels.
fn kmeans_pp(rows: &[&[f64]], k: usize, dim: usize) -> Vec<usize> {
    let n = rows.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let k = k.min(n);

    // Deterministic seed derived from the data.
    let seed: u64 = (n as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(k as u64);
    let mut rng = SimpleRng::new(seed);

    let mut _centres = kmeans_pp_init(rows, k, dim, &mut rng);
    let mut labels = assign_labels(rows, &_centres);

    for _ in 0..KMEANS_MAX_ITER {
        let new_centres = recompute_centres(rows, &labels, k, dim);
        let new_labels = assign_labels(rows, &new_centres);

        let converged = new_labels == labels;
        labels = new_labels;
        _centres = new_centres;
        if converged {
            break;
        }
    }

    labels
}

/// Mean silhouette score across all points.
///
/// For each point the *silhouette coefficient* is
///     s(i) = (b(i) − a(i)) / max(a(i), b(i))
/// where a(i) = mean intra-cluster distance and b(i) = minimum mean
/// inter-cluster distance.
fn silhouette_score(
    rows: &[&[f64]],
    labels: &[usize],
    num_clusters: usize,
    _dim: usize,
) -> f64 {
    let n = rows.len();
    if n < 2 || num_clusters < 2 {
        return f64::NAN;
    }

    // Pre-group indices by cluster.
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
    for (i, &l) in labels.iter().enumerate() {
        if l < num_clusters {
            clusters[l].push(i);
        }
    }

    let mut total_sil = 0.0_f64;
    let mut valid_count = 0usize;

    for i in 0..n {
        let ci = labels[i];

        // a(i): mean distance to points in the same cluster.
        let a = if clusters[ci].len() <= 1 {
            0.0
        } else {
            let sum: f64 = clusters[ci]
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| dist_sq(rows[i], rows[j]).sqrt())
                .sum();
            sum / (clusters[ci].len() - 1) as f64
        };

        // b(i): smallest mean distance to any *other* cluster.
        let mut b = f64::MAX;
        for (cj, members) in clusters.iter().enumerate() {
            if cj == ci || members.is_empty() {
                continue;
            }
            let mean_d: f64 =
                members.iter().map(|&j| dist_sq(rows[i], rows[j]).sqrt()).sum::<f64>()
                    / members.len() as f64;
            if mean_d < b {
                b = mean_d;
            }
        }

        if b == f64::MAX {
            // Only one non-empty cluster – silhouette is 0.
            continue;
        }

        let denom = a.max(b);
        let s = if denom < ZERO_TOL { 0.0 } else { (b - a) / denom };
        total_sil += s;
        valid_count += 1;
    }

    if valid_count == 0 {
        return f64::NAN;
    }

    total_sil / valid_count as f64
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use spectral_types::dense::DenseMatrix;
    use spectral_types::partition::Partition;
    use spectral_types::sparse::{CooMatrix, CsrMatrix};

    // ── helpers ─────────────────────────────────────────────────

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < tol
    }

    /// Build a tiny eigenvector matrix (row-major).
    fn make_eigvecs(rows: usize, cols: usize, data: Vec<f64>) -> DenseMatrix<f64> {
        DenseMatrix { data, rows, cols }
    }

    // ── SF1: spectral gap ──────────────────────────────────────

    #[test]
    fn test_spectral_gap_normal() {
        let evals = vec![0.0, 0.5, 1.2, 2.3];
        assert!((compute_spectral_gap(&evals) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_spectral_gap_single() {
        let evals = vec![0.0];
        assert!(compute_spectral_gap(&evals).is_nan());
    }

    #[test]
    fn test_spectral_gap_empty() {
        let evals: Vec<f64> = vec![];
        assert!(compute_spectral_gap(&evals).is_nan());
    }

    // ── SF2: spectral-gap ratio ────────────────────────────────

    #[test]
    fn test_gap_ratio_normal() {
        // δ²/γ₂² = 4.0 / (2.0*2.0) = 1.0
        let ratio = compute_spectral_gap_ratio(2.0, 4.0);
        assert!((ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gap_ratio_near_zero_gap() {
        let ratio = compute_spectral_gap_ratio(1e-15, 1.0);
        assert!(ratio.is_nan());
    }

    // ── SF3: eigenvalue decay rate ─────────────────────────────

    #[test]
    fn test_decay_rate_exponential() {
        // λ_i = exp(−2·i)  →  ln(λ_i) = −2·i  →  β = 2
        let evals: Vec<f64> = (0..10).map(|i| (-2.0 * i as f64).exp()).collect();
        let beta = compute_eigenvalue_decay_rate(&evals);
        assert!(
            (beta - 2.0).abs() < 0.01,
            "expected β ≈ 2.0, got {beta}"
        );
    }

    #[test]
    fn test_decay_rate_single_positive() {
        let evals = vec![0.0, 1.0];
        // Only one positive eigenvalue → 0.0
        let beta = compute_eigenvalue_decay_rate(&evals);
        // The first eigenvalue 0.0 is not positive; second is positive but alone
        // → fewer than 2 positive → 0.0.
        // Actually: 0.0 is not > 0.0 so only 1.0 qualifies → 0.0
        assert!((beta - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_decay_rate_no_positive() {
        let evals = vec![0.0, 0.0, 0.0];
        assert!((compute_eigenvalue_decay_rate(&evals) - 0.0).abs() < 1e-12);
    }

    // ── SF4: Fiedler entropy ───────────────────────────────────

    #[test]
    fn test_fiedler_entropy_uniform() {
        // Fiedler vector = [1, 1, 1, 1] → p_i = 0.25
        // H = −4 × 0.25 × ln(0.25) = ln(4) ≈ 1.3863
        let n = 4;
        let k = 2;
        let mut data = vec![0.0; n * k];
        for i in 0..n {
            data[i * k] = 1.0; // column 0 (constant eigenvector)
            data[i * k + 1] = 1.0; // column 1 (Fiedler)
        }
        let evecs = make_eigvecs(n, k, data);
        let h = compute_fiedler_entropy(&evecs);
        assert!(
            (h - 4.0_f64.ln()).abs() < 1e-10,
            "expected ln(4) ≈ 1.3863, got {h}"
        );
    }

    #[test]
    fn test_fiedler_entropy_localized() {
        // Fiedler vector = [1, 0, 0, 0] → p = [1,0,0,0] → H = 0
        let n = 4;
        let k = 2;
        let mut data = vec![0.0; n * k];
        data[0 * k + 1] = 1.0; // only first row has nonzero Fiedler entry
        let evecs = make_eigvecs(n, k, data);
        let h = compute_fiedler_entropy(&evecs);
        assert!(
            h.abs() < 1e-12,
            "expected H ≈ 0, got {h}"
        );
    }

    #[test]
    fn test_fiedler_entropy_zero_vector() {
        let evecs = make_eigvecs(4, 2, vec![0.0; 8]);
        assert!(compute_fiedler_entropy(&evecs).is_nan());
    }

    #[test]
    fn test_fiedler_entropy_too_few_cols() {
        let evecs = make_eigvecs(4, 1, vec![1.0; 4]);
        assert!(compute_fiedler_entropy(&evecs).is_nan());
    }

    // ── SF5: algebraic connectivity ratio ──────────────────────

    #[test]
    fn test_acr_normal() {
        let evals = vec![0.0, 1.0, 3.0, 5.0];
        // γ₂/γ₄ = 1.0/5.0 = 0.2
        let r = compute_algebraic_connectivity_ratio(&evals, 4);
        assert!((r - 0.2).abs() < 1e-12, "expected 0.2, got {r}");
    }

    #[test]
    fn test_acr_gamma_k_zero() {
        let evals = vec![0.0, 1.0, 3.0];
        // k=1 → γ_1 = 0.0 → NaN
        assert!(compute_algebraic_connectivity_ratio(&evals, 1).is_nan());
    }

    #[test]
    fn test_acr_k_too_large() {
        let evals = vec![0.0, 1.0];
        assert!(compute_algebraic_connectivity_ratio(&evals, 5).is_nan());
    }

    // ── SF6: coupling energy ───────────────────────────────────

    #[test]
    fn test_coupling_energy_proxy() {
        // 10 positive eigenvalues; smallest 10 % → 1 eigenvalue = 0.1
        let evals: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();
        let ce = compute_coupling_energy(&evals, None, None);
        assert!((ce - 0.1).abs() < 1e-12, "expected 0.1, got {ce}");
    }

    #[test]
    fn test_coupling_energy_with_partition() {
        // 3×3 Laplacian (path graph 0—1—2):
        //  [ 1 -1  0 ]
        //  [-1  2 -1 ]
        //  [ 0 -1  1 ]
        //
        // Partition: {0,1} | {2}
        // Block Laplacian keeps (0,0),(0,1),(1,0),(1,1) and (2,2).
        // Off-block entries: (1,2)=-1 and (2,1)=-1
        // ‖L − L_block‖_F² = 1 + 1 = 2
        let row_ptr = vec![0, 2, 5, 7];
        let col_ind = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
        let lap = CsrMatrix {
            rows: 3,
            cols: 3,
            row_ptr,
            col_ind,
            values,
        };
        let part = Partition::new(vec![0, 0, 1]);
        let ce = compute_coupling_energy(&[], Some(&lap), Some(&part));
        assert!((ce - 2.0).abs() < 1e-12, "expected 2.0, got {ce}");
    }

    // ── SF7: block separability ────────────────────────────────

    #[test]
    fn test_block_separability_well_separated() {
        // Two well-separated 2-D clusters.
        let n = 6;
        let dim = 2;
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0,
            0.1, 0.0,
            0.0, 0.1,
            10.0, 10.0,
            10.1, 10.0,
            10.0, 10.1,
        ];
        let evecs = make_eigvecs(n, dim, data);
        let score = compute_block_separability(&evecs, 2);
        assert!(
            score > 0.8,
            "expected silhouette > 0.8, got {score}"
        );
    }

    #[test]
    fn test_block_separability_single_cluster() {
        let evecs = make_eigvecs(5, 2, vec![0.0; 10]);
        // num_clusters < 2 → NaN
        assert!(compute_block_separability(&evecs, 1).is_nan());
    }

    #[test]
    fn test_block_separability_too_many_clusters() {
        let evecs = make_eigvecs(3, 2, vec![1.0; 6]);
        // num_clusters >= n → NaN
        assert!(compute_block_separability(&evecs, 3).is_nan());
    }

    // ── SF8: effective spectral dimension ──────────────────────

    #[test]
    fn test_effective_dim_uniform() {
        // All equal → d_eff = n
        let evals = vec![1.0; 5];
        let d = compute_effective_dimension(&evals);
        assert!((d - 5.0).abs() < 1e-12, "expected 5.0, got {d}");
    }

    #[test]
    fn test_effective_dim_single() {
        // One eigenvalue → d_eff = 1
        let evals = vec![3.14];
        let d = compute_effective_dimension(&evals);
        assert!((d - 1.0).abs() < 1e-12, "expected 1.0, got {d}");
    }

    #[test]
    fn test_effective_dim_empty() {
        assert!(compute_effective_dimension(&[]).is_nan());
    }

    #[test]
    fn test_effective_dim_all_zero() {
        assert!(compute_effective_dimension(&[0.0, 0.0]).is_nan());
    }

    // ── Combined ───────────────────────────────────────────────

    #[test]
    fn test_compute_all_features() {
        let evals = vec![0.0, 0.5, 1.5, 3.0];
        let n = 4;
        let k = 4;
        // Simple eigenvector matrix: two separated groups in columns 0,1 plus
        // noise in cols 2,3.
        #[rustfmt::skip]
        let data = vec![
            1.0,  1.0, 0.0, 0.0,
            1.0,  0.9, 0.1, 0.0,
            1.0, -1.0, 0.0, 0.1,
            1.0, -0.9, 0.1, 0.1,
        ];
        let evecs = make_eigvecs(n, k, data);

        let feats =
            compute_all_spectral_features(&evals, &evecs, None, None, k).unwrap();

        assert!((feats.spectral_gap - 0.5).abs() < 1e-12);
        assert!(!feats.eigenvalue_decay_rate.is_nan());
        assert!(!feats.effective_spectral_dimension.is_nan());
        assert!(!feats.fiedler_localization_entropy.is_nan());
    }

    #[test]
    fn test_compute_all_features_dimension_mismatch() {
        let evals = vec![0.0, 1.0, 2.0];
        let evecs = make_eigvecs(3, 2, vec![0.0; 6]); // 2 cols ≠ 3 evals
        let result = compute_all_spectral_features(&evals, &evecs, None, None, 3);
        assert!(result.is_err());
    }

    // ── k-means helper ─────────────────────────────────────────

    #[test]
    fn test_kmeans_pp_trivial() {
        // 4 points, 2 clusters: (0,0),(0.1,0) and (10,10),(10.1,10)
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ];
        let rows: Vec<&[f64]> = data.iter().map(|v| v.as_slice()).collect();
        let labels = kmeans_pp(&rows, 2, 2);

        // Points 0,1 must share a label, 2,3 must share a (different) label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    // ── NaN sentinel passthrough ───────────────────────────────

    #[test]
    fn test_nan_sentinels_in_features() {
        // With empty eigenvalues every feature should be NaN.
        let evecs = make_eigvecs(0, 0, vec![]);
        let feats =
            compute_all_spectral_features(&[], &evecs, None, None, 0).unwrap();
        assert!(feats.spectral_gap.is_nan());
        assert!(feats.spectral_gap_ratio.is_nan());
        assert!(feats.fiedler_localization_entropy.is_nan());
        assert!(feats.effective_spectral_dimension.is_nan());
    }
}
