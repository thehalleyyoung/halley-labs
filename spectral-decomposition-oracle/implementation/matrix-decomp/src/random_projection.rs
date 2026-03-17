//! Randomized linear algebra: random projections, randomized SVD, JL embeddings.

use crate::{DenseMatrix, CsrMatrix, DecompError, DecompResult, dot, norm2, normalize};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rand::distributions::Standard;

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 1. Random matrix generators
// ---------------------------------------------------------------------------

/// Generate a `rows x cols` matrix with i.i.d. N(0,1) entries using Box-Muller.
pub fn random_gaussian_matrix(rows: usize, cols: usize, seed: u64) -> DenseMatrix {
    let total = rows * cols;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(total);

    let pairs = (total + 1) / 2;
    for _ in 0..pairs {
        // Clamp u1 away from zero to avoid ln(0).
        let u1: f64 = rng.sample::<f64, _>(Standard).max(f64::MIN_POSITIVE);
        let u2: f64 = rng.sample::<f64, _>(Standard);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        data.push(r * theta.cos());
        data.push(r * theta.sin());
    }
    data.truncate(total);
    DenseMatrix::from_row_major(rows, cols, data)
}

/// Generate a `rows x cols` Rademacher matrix (entries +/-1 with equal probability).
pub fn random_rademacher_matrix(rows: usize, cols: usize, seed: u64) -> DenseMatrix {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let total = rows * cols;
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        let bit: bool = rng.gen();
        data.push(if bit { 1.0 } else { -1.0 });
    }
    DenseMatrix::from_row_major(rows, cols, data)
}

/// Generate a sparse random projection matrix (Achlioptas-style).
///
/// Each entry is `+sqrt(1/density)` with probability `density/2`,
/// `-sqrt(1/density)` with probability `density/2`, and `0` otherwise.
///
/// If `density` is <= 0 or > 1, the default `1/sqrt(cols)` is used.
pub fn sparse_random_projection(
    rows: usize,
    cols: usize,
    density: f64,
    seed: u64,
) -> DenseMatrix {
    let d = if density <= 0.0 || density > 1.0 {
        if cols == 0 { 1.0 } else { 1.0 / (cols as f64).sqrt() }
    } else {
        density
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let total = rows * cols;
    let val = (1.0 / d).sqrt();
    let half = d / 2.0;
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        let u: f64 = rng.gen();
        if u < half {
            data.push(val);
        } else if u < d {
            data.push(-val);
        } else {
            data.push(0.0);
        }
    }
    DenseMatrix::from_row_major(rows, cols, data)
}

// ---------------------------------------------------------------------------
// 2. Internal helpers: QR and symmetric eigen
// ---------------------------------------------------------------------------

/// Thin QR factorization via modified Gram-Schmidt.
///
/// Given an `m x n` matrix (m >= n), returns `(Q, R)` where `Q` is `m x k`
/// orthonormal and `R` is `k x n` upper triangular, with `k = min(m, n)`.
fn qr_thin(a: &DenseMatrix) -> (DenseMatrix, DenseMatrix) {
    let m = a.rows;
    let n = a.cols;
    let k = m.min(n);

    let mut q_cols: Vec<Vec<f64>> = (0..n).map(|j| a.col(j)).collect();
    let mut r = DenseMatrix::zeros(k, n);

    for j in 0..k {
        for i in 0..j {
            let rij = dot(&q_cols[i], &q_cols[j]);
            r.set(i, j, rij);
            for row in 0..m {
                q_cols[j][row] -= rij * q_cols[i][row];
            }
        }
        let nrm = norm2(&q_cols[j]);
        r.set(j, j, nrm);
        if nrm > 1e-300 {
            let inv = 1.0 / nrm;
            for row in 0..m {
                q_cols[j][row] *= inv;
            }
        }
    }

    for j in k..n {
        for i in 0..k {
            let rij = dot(&q_cols[i], &q_cols[j]);
            r.set(i, j, rij);
            for row in 0..m {
                q_cols[j][row] -= rij * q_cols[i][row];
            }
        }
    }

    let mut q_data = vec![0.0; m * k];
    for j in 0..k {
        for i in 0..m {
            q_data[i * k + j] = q_cols[j][i];
        }
    }
    let q = DenseMatrix::from_row_major(m, k, q_data);
    (q, r)
}

/// Eigendecomposition of a small symmetric matrix via Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are columns of the
/// returned matrix, sorted by descending absolute eigenvalue.
fn small_symmetric_eigen(a: &DenseMatrix) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = a.rows;
    if n != a.cols {
        return Err(DecompError::NotSquare {
            rows: a.rows,
            cols: a.cols,
        });
    }
    if n == 0 {
        return Ok((vec![], DenseMatrix::zeros(0, 0)));
    }
    if n == 1 {
        return Ok((vec![a.get(0, 0)], DenseMatrix::eye(1)));
    }

    let mut s = a.clone();
    let mut v = DenseMatrix::eye(n);

    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _sweep in 0..max_iter {
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = s.get(i, j).abs();
                if aij > max_val {
                    max_val = aij;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        let app = s.get(p, p);
        let aqq = s.get(q, q);
        let apq = s.get(p, q);

        let (c, sn) = if (app - aqq).abs() < 1e-300 {
            let sq = std::f64::consts::FRAC_1_SQRT_2;
            (sq, sq)
        } else {
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let sn = t * c;
            (c, sn)
        };

        for i in 0..n {
            if i != p && i != q {
                let sip = s.get(i, p);
                let siq = s.get(i, q);
                let new_ip = c * sip - sn * siq;
                let new_iq = sn * sip + c * siq;
                s.set(i, p, new_ip);
                s.set(p, i, new_ip);
                s.set(i, q, new_iq);
                s.set(q, i, new_iq);
            }
        }

        let new_pp = c * c * app - 2.0 * sn * c * apq + sn * sn * aqq;
        let new_qq = sn * sn * app + 2.0 * sn * c * apq + c * c * aqq;
        s.set(p, p, new_pp);
        s.set(q, q, new_qq);
        s.set(p, q, 0.0);
        s.set(q, p, 0.0);

        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.set(i, p, c * vip - sn * viq);
            v.set(i, q, sn * vip + c * viq);
        }
    }

    let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (s.get(i, i), i)).collect();
    eig_pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = eig_pairs.iter().map(|(val, _)| *val).collect();
    let mut eigvec_data = vec![0.0; n * n];
    for (new_j, (_, old_j)) in eig_pairs.iter().enumerate() {
        for i in 0..n {
            eigvec_data[i * n + new_j] = v.get(i, *old_j);
        }
    }
    let eigvecs = DenseMatrix::from_row_major(n, n, eigvec_data);

    Ok((eigenvalues, eigvecs))
}

// ---------------------------------------------------------------------------
// 3. SVD of small dense matrix (for randomized SVD inner loop)
// ---------------------------------------------------------------------------

/// Small dense SVD via eigendecomposition of B^T B.
///
/// Returns (U, sigma, Vt) for an m x n matrix.
fn small_svd(b: &DenseMatrix) -> DecompResult<(DenseMatrix, Vec<f64>, DenseMatrix)> {
    let m = b.rows;
    let n = b.cols;

    if m == 0 || n == 0 {
        return Ok((
            DenseMatrix::zeros(m, 0),
            vec![],
            DenseMatrix::zeros(0, n),
        ));
    }

    let bt = b.transpose();
    let btb = bt.mul_mat(b)?;

    let (eigvals, eigvecs) = small_symmetric_eigen(&btb)?;

    let rank = eigvals.iter().filter(|&&v| v.abs() > 1e-14).count();
    let mut sigma = Vec::with_capacity(rank);
    let mut v_cols: Vec<usize> = Vec::with_capacity(rank);

    for (i, &lam) in eigvals.iter().enumerate() {
        if lam.abs() > 1e-14 && sigma.len() < rank {
            sigma.push(lam.abs().sqrt());
            v_cols.push(i);
        }
    }

    let r = sigma.len();
    if r == 0 {
        return Ok((
            DenseMatrix::zeros(m, 0),
            vec![],
            DenseMatrix::zeros(0, n),
        ));
    }

    let mut v_data = vec![0.0; n * r];
    for (new_j, &old_j) in v_cols.iter().enumerate() {
        for i in 0..n {
            v_data[i * r + new_j] = eigvecs.get(i, old_j);
        }
    }
    let v_mat = DenseMatrix::from_row_major(n, r, v_data);

    let bv = b.mul_mat(&v_mat)?;
    let mut u_data = vec![0.0; m * r];
    for j in 0..r {
        let inv_s = 1.0 / sigma[j];
        for i in 0..m {
            u_data[i * r + j] = bv.get(i, j) * inv_s;
        }
    }
    let u_mat = DenseMatrix::from_row_major(m, r, u_data);

    let vt = v_mat.transpose();

    Ok((u_mat, sigma, vt))
}

// ---------------------------------------------------------------------------
// 4. Subspace iteration (randomized range finder)
// ---------------------------------------------------------------------------

/// Randomized range finder for dense matrices via subspace iteration.
///
/// Computes an approximate orthonormal basis `Q` (m x k) for the range of `A`.
/// Uses `n_iter` power iterations to improve the quality of the approximation.
pub fn subspace_iteration(
    a: &DenseMatrix,
    k: usize,
    n_iter: usize,
    seed: u64,
) -> DecompResult<DenseMatrix> {
    let m = a.rows;
    let n = a.cols;

    if k == 0 {
        return Ok(DenseMatrix::zeros(m, 0));
    }
    let k_eff = k.min(m).min(n);

    let omega = random_gaussian_matrix(n, k_eff, seed);

    let mut y = a.mul_mat(&omega)?;

    let at = a.transpose();

    for _iter in 0..n_iter {
        let (q, _) = qr_thin(&y);
        let z = at.mul_mat(&q)?;
        let (q2, _) = qr_thin(&z);
        y = a.mul_mat(&q2)?;
    }

    let (q, _) = qr_thin(&y);
    Ok(q)
}

/// Randomized range finder for sparse matrices via subspace iteration.
pub fn subspace_iteration_sparse(
    a: &CsrMatrix,
    k: usize,
    n_iter: usize,
    seed: u64,
) -> DecompResult<DenseMatrix> {
    let m = a.rows;
    let n = a.cols;

    if k == 0 {
        return Ok(DenseMatrix::zeros(m, 0));
    }
    let k_eff = k.min(m).min(n);

    let omega = random_gaussian_matrix(n, k_eff, seed);
    let at = a.transpose();

    let mut y = sparse_mul_dense(a, &omega)?;

    for _iter in 0..n_iter {
        let (q, _) = qr_thin(&y);
        let z = sparse_mul_dense(&at, &q)?;
        let (q2, _) = qr_thin(&z);
        y = sparse_mul_dense(a, &q2)?;
    }

    let (q, _) = qr_thin(&y);
    Ok(q)
}

/// Multiply a sparse matrix by a dense matrix column by column.
fn sparse_mul_dense(a: &CsrMatrix, b: &DenseMatrix) -> DecompResult<DenseMatrix> {
    let m = a.rows;
    let k = b.cols;
    let mut result = DenseMatrix::zeros(m, k);

    for j in 0..k {
        let col_j = b.col(j);
        let y = a.mul_vec(&col_j)?;
        for i in 0..m {
            result.set(i, j, y[i]);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// 5. Randomized SVD (Halko-Martinsson-Tropp)
// ---------------------------------------------------------------------------

/// Randomized SVD via the Halko-Martinsson-Tropp algorithm.
///
/// Computes an approximate rank-`k` SVD of `A`:  A ~ U * diag(sigma) * Vt.
///
/// - `oversampling`: extra dimensions for the random sketch (typically 5-10).
/// - `n_power_iter`: number of power iterations to improve accuracy.
///
/// Returns `(U, sigma, Vt)` where `U` is m x k, `sigma` has k entries, `Vt` is k x n.
pub fn randomized_svd(
    a: &DenseMatrix,
    k: usize,
    oversampling: usize,
    n_power_iter: usize,
    seed: u64,
) -> DecompResult<(DenseMatrix, Vec<f64>, DenseMatrix)> {
    let m = a.rows;
    let n = a.cols;

    if k == 0 {
        return Ok((DenseMatrix::zeros(m, 0), vec![], DenseMatrix::zeros(0, n)));
    }
    if k > m.min(n) {
        return Err(DecompError::InvalidParameter {
            name: "k".into(),
            value: k.to_string(),
            reason: format!("k must be <= min(m,n) = {}", m.min(n)),
        });
    }

    let p = (k + oversampling).min(m).min(n);

    let q = subspace_iteration(a, p, n_power_iter, seed)?;

    let qt = q.transpose();
    let b = qt.mul_mat(a)?;

    let (ub, sigma_full, vt_full) = small_svd(&b)?;

    let u_full = q.mul_mat(&ub)?;

    let r = k.min(sigma_full.len());
    let sigma: Vec<f64> = sigma_full[..r].to_vec();

    let u = u_full.submatrix(0, m, 0, r);
    let vt = vt_full.submatrix(0, r, 0, n);

    Ok((u, sigma, vt))
}

// ---------------------------------------------------------------------------
// 6. Randomized eigendecomposition (for symmetric matrices)
// ---------------------------------------------------------------------------

/// Randomized eigendecomposition for symmetric matrices.
///
/// Computes the top-`k` eigenvalues/eigenvectors of a symmetric matrix `A`.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by descending absolute eigenvalue.
pub fn randomized_eigen(
    a: &DenseMatrix,
    k: usize,
    oversampling: usize,
    n_power_iter: usize,
    seed: u64,
) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = a.rows;
    if n != a.cols {
        return Err(DecompError::NotSquare {
            rows: a.rows,
            cols: a.cols,
        });
    }
    if k == 0 {
        return Ok((vec![], DenseMatrix::zeros(n, 0)));
    }
    if k > n {
        return Err(DecompError::InvalidParameter {
            name: "k".into(),
            value: k.to_string(),
            reason: format!("k must be <= n = {}", n),
        });
    }

    let p = (k + oversampling).min(n);

    let q = subspace_iteration(a, p, n_power_iter, seed)?;

    let qt = q.transpose();
    let aq = a.mul_mat(&q)?;
    let t = qt.mul_mat(&aq)?;

    let (eigvals_full, eigvecs_small) = small_symmetric_eigen(&t)?;

    let v_full = q.mul_mat(&eigvecs_small)?;

    let r = k.min(eigvals_full.len());
    let eigenvalues = eigvals_full[..r].to_vec();
    let eigenvectors = v_full.submatrix(0, n, 0, r);

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// 7. Johnson-Lindenstrauss
// ---------------------------------------------------------------------------

/// Compute the minimum target dimension for a Johnson-Lindenstrauss projection.
///
/// Returns the smallest integer `k` satisfying:
///   k >= 4 * ln(n_samples) / (eps^2/2 - eps^3/3)
///
/// Returns `n_samples` if eps is too large for any reduction.
pub fn johnson_lindenstrauss_dim(n_samples: usize, eps: f64) -> usize {
    if n_samples <= 1 {
        return 1;
    }
    let ln_n = (n_samples as f64).ln();
    let denom = eps * eps / 2.0 - eps * eps * eps / 3.0;
    if denom <= 0.0 {
        return n_samples;
    }
    let k = 4.0 * ln_n / denom;
    (k.ceil() as usize).max(1)
}

/// Apply a Johnson-Lindenstrauss random projection.
///
/// Projects the rows of `data` (n x d) into R^{target_dim} using a scaled
/// random Gaussian matrix. The projection matrix R is `target_dim x d` with
/// entries ~ N(0,1), scaled by `1/sqrt(target_dim)`. Result is `data * R^T`.
pub fn jl_projection(
    data: &DenseMatrix,
    target_dim: usize,
    seed: u64,
) -> DecompResult<DenseMatrix> {
    let (_n, d) = data.shape();
    if target_dim == 0 {
        return Err(DecompError::InvalidParameter {
            name: "target_dim".into(),
            value: "0".into(),
            reason: "Target dimension must be positive".into(),
        });
    }

    let mut r = random_gaussian_matrix(target_dim, d, seed);
    let scale = 1.0 / (target_dim as f64).sqrt();
    r.scale(scale);

    let rt = r.transpose();
    data.mul_mat(&rt)
}

// ---------------------------------------------------------------------------
// 8. Spectral norm estimation (power iteration)
// ---------------------------------------------------------------------------

/// Estimate the spectral norm ||A||_2 via power iteration.
///
/// Uses the relation ||A||_2 = sigma_max(A) = sqrt(lambda_max(A^T A)).
/// Iterates `v <- A^T A v / ||A^T A v||` and returns `||Av||` after convergence.
pub fn estimate_spectral_norm(a: &DenseMatrix, n_iter: usize, seed: u64) -> f64 {
    let n = a.cols;
    if n == 0 || a.rows == 0 {
        return 0.0;
    }

    let mut v = random_gaussian_matrix(n, 1, seed).data;
    normalize(&mut v);

    let at = a.transpose();

    for _ in 0..n_iter {
        let w = match a.mul_vec(&v) {
            Ok(w) => w,
            Err(_) => return 0.0,
        };
        let mut u = match at.mul_vec(&w) {
            Ok(u) => u,
            Err(_) => return 0.0,
        };
        let nrm = normalize(&mut u);
        if nrm < 1e-300 {
            return 0.0;
        }
        v = u;
    }

    match a.mul_vec(&v) {
        Ok(av) => norm2(&av),
        Err(_) => 0.0,
    }
}

/// Estimate the spectral norm ||A||_2 for sparse matrices via power iteration.
///
/// Identical algorithm to [`estimate_spectral_norm`] but leverages
/// [`CsrMatrix::mul_vec`] for efficiency on large sparse matrices.
pub fn estimate_spectral_norm_sparse(a: &CsrMatrix, n_iter: usize, seed: u64) -> f64 {
    let n = a.cols;
    if n == 0 || a.rows == 0 {
        return 0.0;
    }

    let mut v = random_gaussian_matrix(n, 1, seed).data;
    normalize(&mut v);

    let at = a.transpose();

    for _ in 0..n_iter {
        let w = match a.mul_vec(&v) {
            Ok(w) => w,
            Err(_) => return 0.0,
        };
        let mut u = match at.mul_vec(&w) {
            Ok(u) => u,
            Err(_) => return 0.0,
        };
        let nrm = normalize(&mut u);
        if nrm < 1e-300 {
            return 0.0;
        }
        v = u;
    }

    match a.mul_vec(&v) {
        Ok(av) => norm2(&av),
        Err(_) => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    const TOL: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn check_orthonormal(q: &DenseMatrix, tol: f64) {
        let qt = q.transpose();
        let qtq = qt.mul_mat(q).unwrap();
        let n = qtq.rows;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(qtq.get(i, j), expected, tol),
                    "Q^T Q[{},{}] = {} != {}",
                    i, j, qtq.get(i, j), expected
                );
            }
        }
    }

    fn frob_diff(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        let mut s = 0.0;
        for i in 0..a.rows {
            for j in 0..a.cols {
                let d = a.get(i, j) - b.get(i, j);
                s += d * d;
            }
        }
        s.sqrt()
    }

    fn low_rank_matrix(m: usize, n: usize, singular_values: &[f64], seed: u64) -> DenseMatrix {
        let k = singular_values.len();
        let u = random_gaussian_matrix(m, k, seed);
        let v = random_gaussian_matrix(n, k, seed.wrapping_add(999));
        let (q_u, _) = qr_thin(&u);
        let (q_v, _) = qr_thin(&v);
        let mut a = DenseMatrix::zeros(m, n);
        for l in 0..k {
            for i in 0..m {
                for j in 0..n {
                    a.set(i, j, a.get(i, j) + singular_values[l] * q_u.get(i, l) * q_v.get(j, l));
                }
            }
        }
        a
    }

    // Test 1: Gaussian matrix dimensions and statistics
    #[test]
    fn test_gaussian_matrix_shape_and_stats() {
        let m = random_gaussian_matrix(1000, 50, 42);
        assert_eq!(m.rows, 1000);
        assert_eq!(m.cols, 50);
        let total = (m.rows * m.cols) as f64;
        let mean: f64 = m.data.iter().sum::<f64>() / total;
        let var: f64 = m.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / total;
        assert!(mean.abs() < 0.1, "mean = {}", mean);
        assert!((var - 1.0).abs() < 0.2, "var = {}", var);
    }

    // Test 2: Rademacher entries are +/-1
    #[test]
    fn test_rademacher_matrix() {
        let m = random_rademacher_matrix(100, 100, 7);
        assert_eq!(m.rows, 100);
        assert_eq!(m.cols, 100);
        for &v in &m.data {
            assert!(v == 1.0 || v == -1.0);
        }
        let n_pos = m.data.iter().filter(|&&v| v == 1.0).count();
        let frac = n_pos as f64 / m.data.len() as f64;
        assert!((frac - 0.5).abs() < 0.1, "frac_pos = {frac}");
    }

    // Test 3: Sparse projection density
    #[test]
    fn test_sparse_random_projection_density() {
        let density = 1.0 / 3.0;
        let m = sparse_random_projection(500, 500, density, 99);
        let zeros = m.data.iter().filter(|&&v| v == 0.0).count();
        let total = 500 * 500;
        let frac_nonzero = 1.0 - (zeros as f64 / total as f64);
        assert!(
            (frac_nonzero - density).abs() < 0.05,
            "nonzero fraction = {}", frac_nonzero
        );
        let expected_abs = (1.0 / density).sqrt();
        for &v in &m.data {
            if v != 0.0 {
                assert!(
                    (v.abs() - expected_abs).abs() < 1e-12,
                    "non-zero entry = {v}, expected +/-{expected_abs}"
                );
            }
        }
    }

    // Test 4: QR thin correctness
    #[test]
    fn test_qr_thin_orthonormal() {
        let a = random_gaussian_matrix(20, 5, 123);
        let (q, r) = qr_thin(&a);
        assert_eq!(q.rows, 20);
        assert_eq!(q.cols, 5);
        assert_eq!(r.rows, 5);
        assert_eq!(r.cols, 5);
        check_orthonormal(&q, 1e-10);
        let qr = q.mul_mat(&r).unwrap();
        let err = frob_diff(&a, &qr);
        assert!(err < 1e-10, "||A - QR|| = {err}");
    }

    // Test 5: Subspace iteration captures dominant subspace
    #[test]
    fn test_subspace_iteration_dense() {
        let a = low_rank_matrix(30, 20, &[100.0, 50.0, 10.0], 42);
        let q = subspace_iteration(&a, 3, 2, 33).unwrap();
        assert_eq!(q.rows, 30);
        assert_eq!(q.cols, 3);
        check_orthonormal(&q, 1e-10);

        let qt = q.transpose();
        let proj = q.mul_mat(&qt.mul_mat(&a).unwrap()).unwrap();
        let residual = frob_diff(&a, &proj) / a.frobenius_norm();
        assert!(residual < 1e-6, "relative projection residual = {residual}");
    }

    // Test 6: Randomized SVD approximation
    #[test]
    fn test_randomized_svd() {
        let u_raw = random_gaussian_matrix(20, 3, 100);
        let (u_orth, _) = qr_thin(&u_raw);
        let v_raw = random_gaussian_matrix(10, 3, 200);
        let (v_orth, _) = qr_thin(&v_raw);

        let sigma_diag = DenseMatrix::from_diag(&[5.0, 3.0, 1.0]);
        let us = u_orth.mul_mat(&sigma_diag).unwrap();
        let a = us.mul_mat(&v_orth.transpose()).unwrap();

        let (u, sigma, vt) = randomized_svd(&a, 3, 5, 2, 300).unwrap();
        assert_eq!(u.rows, 20);
        assert_eq!(u.cols, 3);
        assert_eq!(sigma.len(), 3);
        assert_eq!(vt.rows, 3);
        assert_eq!(vt.cols, 10);

        let mut sigma_sorted = sigma.clone();
        sigma_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(approx_eq(sigma_sorted[0], 5.0, 2.0), "s0 = {}", sigma_sorted[0]);
        assert!(approx_eq(sigma_sorted[1], 3.0, 2.0), "s1 = {}", sigma_sorted[1]);
        assert!(approx_eq(sigma_sorted[2], 1.0, 2.0), "s2 = {}", sigma_sorted[2]);

        let s_diag = DenseMatrix::from_diag(&sigma);
        let us2 = u.mul_mat(&s_diag).unwrap();
        let reconstructed = us2.mul_mat(&vt).unwrap();
        let err = frob_diff(&a, &reconstructed);
        assert!(err < 1.0, "reconstruction error = {}", err);
    }

    // Test 7: JL dimension formula
    #[test]
    fn test_johnson_lindenstrauss_dim() {
        let k = johnson_lindenstrauss_dim(1000, 0.1);
        assert!(k > 100, "k = {}", k);
        assert!(k < 50000, "k = {}", k);

        let k2 = johnson_lindenstrauss_dim(1000, 0.5);
        assert!(k2 < k, "k(0.5) = {} should be < k(0.1) = {}", k2, k);

        assert_eq!(johnson_lindenstrauss_dim(1, 0.5), 1);
    }

    // Test 8: JL projection approximately preserves distances
    #[test]
    fn test_jl_projection() {
        let data = random_gaussian_matrix(50, 100, 777);
        let projected = jl_projection(&data, 20, 888).unwrap();
        assert_eq!(projected.rows, 50);
        assert_eq!(projected.cols, 20);

        for i in 0..5 {
            for j in (i + 1)..5 {
                let row_i_orig = data.row(i);
                let row_j_orig = data.row(j);
                let diff_orig: Vec<f64> = row_i_orig
                    .iter().zip(row_j_orig.iter())
                    .map(|(a, b)| a - b).collect();
                let dist_orig = norm2(&diff_orig);

                let row_i_proj = projected.row(i);
                let row_j_proj = projected.row(j);
                let diff_proj: Vec<f64> = row_i_proj
                    .iter().zip(row_j_proj.iter())
                    .map(|(a, b)| a - b).collect();
                let dist_proj = norm2(&diff_proj);

                let ratio = dist_proj / dist_orig.max(1e-15);
                assert!(
                    ratio > 0.2 && ratio < 5.0,
                    "distance ratio = {} for pair ({}, {})", ratio, i, j
                );
            }
        }
    }

    // Test 9: Spectral norm estimate close to true value
    #[test]
    fn test_estimate_spectral_norm_dense() {
        let a = DenseMatrix::from_diag(&[5.0, 3.0, 1.0, 0.5]);
        let sigma = estimate_spectral_norm(&a, 50, 42);
        assert!(
            approx_eq(sigma, 5.0, 0.1),
            "spectral norm = {}, expected 5.0", sigma
        );
    }

    // Test 10: Randomized eigendecomposition for symmetric matrix
    #[test]
    fn test_randomized_eigen_symmetric() {
        let q_raw = random_gaussian_matrix(10, 10, 501);
        let (q_orth, _) = qr_thin(&q_raw);
        let eigvals = [10.0, 7.0, 5.0, 3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01];
        let d = DenseMatrix::from_diag(&eigvals);
        let qd = q_orth.mul_mat(&d).unwrap();
        let a = qd.mul_mat(&q_orth.transpose()).unwrap();

        let (vals, vecs) = randomized_eigen(&a, 3, 5, 3, 601).unwrap();
        assert_eq!(vals.len(), 3);
        assert_eq!(vecs.rows, 10);
        assert_eq!(vecs.cols, 3);

        let mut vals_sorted = vals.clone();
        vals_sorted.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        assert!(approx_eq(vals_sorted[0], 10.0, 1.0), "l0 = {}", vals_sorted[0]);
        assert!(approx_eq(vals_sorted[1], 7.0, 1.0), "l1 = {}", vals_sorted[1]);
        assert!(approx_eq(vals_sorted[2], 5.0, 1.0), "l2 = {}", vals_sorted[2]);
    }

    // Test 11: small_svd on identity
    #[test]
    fn test_small_svd_identity() {
        let eye = DenseMatrix::eye(5);
        let (u, s, vt) = small_svd(&eye).unwrap();
        assert_eq!(s.len(), 5);
        for &sv in &s {
            assert!((sv - 1.0).abs() < 1e-10, "singular value = {sv}");
        }
        // Reconstruct
        let s_diag = DenseMatrix::from_diag(&s);
        let us = u.mul_mat(&s_diag).unwrap();
        let recon = us.mul_mat(&vt).unwrap();
        let err = frob_diff(&eye, &recon);
        assert!(err < 1e-10, "||I - USVt|| = {err}");
    }

    // Test 12: Sparse spectral norm
    #[test]
    fn test_estimate_spectral_norm_sparse() {
        let dense = DenseMatrix::from_diag(&[7.0, 2.0, 1.0]);
        let sparse = dense.to_csr();
        let sigma = estimate_spectral_norm_sparse(&sparse, 50, 99);
        assert!(
            approx_eq(sigma, 7.0, 0.1),
            "spectral norm = {}, expected 7.0", sigma
        );
    }

    // Test 13: Subspace iteration sparse matches dense
    #[test]
    fn test_subspace_iteration_sparse() {
        let dense = DenseMatrix::from_row_major(
            4, 3,
            vec![
                1.0, 0.0, 2.0,
                0.0, 3.0, 0.0,
                4.0, 0.0, 5.0,
                0.0, 6.0, 0.0,
            ],
        );
        let sparse = dense.to_csr();
        let q = subspace_iteration_sparse(&sparse, 2, 2, 55).unwrap();
        assert_eq!(q.rows, 4);
        assert!(q.cols <= 2);
        check_orthonormal(&q, 1e-10);
    }

    // Test 14: Deterministic seeding
    #[test]
    fn test_deterministic_seeding() {
        let m1 = random_gaussian_matrix(50, 40, 314159);
        let m2 = random_gaussian_matrix(50, 40, 314159);
        assert_eq!(m1.data, m2.data, "same seed must produce identical matrices");

        let m3 = random_gaussian_matrix(50, 40, 271828);
        assert_ne!(m1.data, m3.data, "different seeds must produce different matrices");
    }

    // Test 15: Zero-k edge cases
    #[test]
    fn test_randomized_svd_k_zero() {
        let a = DenseMatrix::zeros(5, 5);
        let (u, sigma, vt) = randomized_svd(&a, 0, 0, 0, 0).unwrap();
        assert_eq!(u.cols, 0);
        assert!(sigma.is_empty());
        assert_eq!(vt.rows, 0);
    }

    #[test]
    fn test_subspace_iteration_zero_k() {
        let a = DenseMatrix::eye(5);
        let q = subspace_iteration(&a, 0, 0, 0).unwrap();
        assert_eq!(q.rows, 5);
        assert_eq!(q.cols, 0);
    }

    // Test 16: Small symmetric eigen
    #[test]
    fn test_small_symmetric_eigen() {
        let a = DenseMatrix::from_row_major(
            3, 3,
            vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        );
        let (vals, vecs) = small_symmetric_eigen(&a).unwrap();
        assert_eq!(vals.len(), 3);
        for j in 0..3 {
            let col_j = vecs.col(j);
            let av = a.mul_vec(&col_j).unwrap();
            for i in 0..3 {
                assert!(
                    approx_eq(av[i], vals[j] * col_j[i], 1e-8),
                    "Av[{}] = {}, lambda*v[{}] = {}", i, av[i], i, vals[j] * col_j[i]
                );
            }
        }
        check_orthonormal(&vecs, 1e-10);
    }
}
