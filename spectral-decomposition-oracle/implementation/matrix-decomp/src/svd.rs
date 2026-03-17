//! Singular Value Decomposition (SVD).
//!
//! Computes A = U · diag(S) · Vᵀ where U and V are orthogonal and S contains
//! non-negative singular values in descending order.
//!
//! Algorithms provided:
//! - **Full SVD** via Golub–Kahan bidiagonalisation + implicit QR (large) or
//!   one-sided Jacobi (small ≤ 64).
//! - **Randomised SVD** (Halko–Martinsson–Tropp) for low-rank approximations
//!   of both dense and sparse matrices.

use crate::{dot, norm2, normalize, CsrMatrix, DecompError, DecompResult, DenseMatrix};
use crate::givens::GivensRotation;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ═══════════════════════════════════════════════════════════════════════════
// Result struct
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a Singular Value Decomposition A = U · diag(S) · Vᵀ.
#[derive(Debug, Clone)]
pub struct SvdDecomposition {
    /// Left singular vectors (m × k).
    pub u: DenseMatrix,
    /// Singular values in descending order (length k).
    pub s: Vec<f64>,
    /// Right singular vectors transposed (k × n).
    pub vt: DenseMatrix,
    /// Number of rows of the original matrix.
    pub m: usize,
    /// Number of columns of the original matrix.
    pub n: usize,
    /// Number of singular values / vectors stored.
    pub k: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// 2×2 SVD (analytic)
// ═══════════════════════════════════════════════════════════════════════════

/// Analytic SVD of a 2×2 matrix `[[a, b], [c, d]]`.
///
/// Returns `(u, [σ₁, σ₂], vt)` where σ₁ ≥ σ₂ ≥ 0.
pub fn svd_2x2(a: f64, b: f64, c: f64, d: f64) -> (DenseMatrix, [f64; 2], DenseMatrix) {
    // Compute A^T A = [[e, f], [f, g]]
    let e = a * a + c * c;
    let f = a * b + c * d;
    let g = b * b + d * d;

    // Eigenvalues of A^T A via the 2×2 symmetric eigenvalue formula
    let half_sum = 0.5 * (e + g);
    let half_diff = 0.5 * (e - g);
    let disc = (half_diff * half_diff + f * f).sqrt();

    let sigma1 = (half_sum + disc).max(0.0).sqrt();
    let sigma2 = (half_sum - disc).max(0.0).sqrt();

    // Right singular vectors: eigenvectors of A^T A
    let (cv, sv) = if f.abs() < 1e-300 && (e - g).abs() < 1e-300 {
        (1.0, 0.0)
    } else {
        let theta = 0.5 * f64::atan2(2.0 * f, e - g);
        (theta.cos(), theta.sin())
    };

    // V = [[cv, -sv], [sv, cv]]
    // U columns from A * v_i / sigma_i
    let u;
    let vt;

    if sigma1 > 1e-300 {
        let u1_0 = (a * cv + b * sv) / sigma1;
        let u1_1 = (c * cv + d * sv) / sigma1;
        if sigma2 > 1e-300 {
            let u2_0 = (-a * sv + b * cv) / sigma2;
            let u2_1 = (-c * sv + d * cv) / sigma2;
            u = DenseMatrix::from_row_major(2, 2, vec![u1_0, u2_0, u1_1, u2_1]);
        } else {
            // sigma2 ≈ 0 ⇒ pick u2 orthogonal to u1
            let u2_0 = -u1_1;
            let u2_1 = u1_0;
            u = DenseMatrix::from_row_major(2, 2, vec![u1_0, u2_0, u1_1, u2_1]);
        }
        vt = DenseMatrix::from_row_major(2, 2, vec![cv, sv, -sv, cv]);
    } else {
        // both singular values are zero
        u = DenseMatrix::eye(2);
        vt = DenseMatrix::eye(2);
    }

    (u, [sigma1, sigma2], vt)
}

// ═══════════════════════════════════════════════════════════════════════════
// Householder helpers (local to this module)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute Householder vector v and scalar tau such that
/// (I - tau * v * v^T) * x = ||x|| * e_1.
/// Returns (v, tau).
fn house(x: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    if n == 0 {
        return (vec![], 0.0);
    }
    let mut v = x.to_vec();
    let sigma: f64 = x[1..].iter().map(|&xi| xi * xi).sum();
    v[0] = 1.0;

    if sigma < 1e-300 {
        return (v, 0.0);
    }

    let mu = (x[0] * x[0] + sigma).sqrt();
    if x[0] <= 0.0 {
        v[0] = x[0] - mu;
    } else {
        v[0] = -sigma / (x[0] + mu);
    }
    let tau = 2.0 * v[0] * v[0] / (sigma + v[0] * v[0]);
    let v0 = v[0];
    for vi in v.iter_mut() {
        *vi /= v0;
    }
    (v, tau)
}

// ═══════════════════════════════════════════════════════════════════════════
// Golub–Kahan bidiagonalisation
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce A (m×n, m ≥ n) to upper bidiagonal form via Householder reflections:
///   U^T · A · V = B  where B is upper bidiagonal.
///
/// Returns `(U, d, e, V)`:
///   - `U` is m×m orthogonal
///   - `d` is the diagonal (length n)
///   - `e` is the super-diagonal (length n−1)
///   - `V` is n×n orthogonal (not transposed)
pub fn bidiagonalize_golub_kahan(
    a: &DenseMatrix,
) -> DecompResult<(DenseMatrix, Vec<f64>, Vec<f64>, DenseMatrix)> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(DecompError::EmptyMatrix {
            context: "bidiagonalize: empty matrix".into(),
        });
    }
    // We require m >= n. If not, caller should transpose first.
    let mut work = a.clone();
    let mut u_acc = DenseMatrix::eye(m);
    let mut v_acc = DenseMatrix::eye(n);

    for k in 0..n {
        // --- Left Householder: zero out below work[k][k] ---
        if k < m {
            let col_slice: Vec<f64> = (k..m).map(|i| work.get(i, k)).collect();
            let (v, tau) = house(&col_slice);
            if tau.abs() > 1e-300 {
                // Apply to work: work[k:m, k:n] -= tau * v * (v^T * work[k:m, k:n])
                for j in k..n {
                    let mut dot_val = 0.0;
                    for (idx, &vi) in v.iter().enumerate() {
                        dot_val += vi * work.get(k + idx, j);
                    }
                    for (idx, &vi) in v.iter().enumerate() {
                        let old = work.get(k + idx, j);
                        work.set(k + idx, j, old - tau * vi * dot_val);
                    }
                }
                // Accumulate into U: U[:, k:m] -= tau * (U[:, k:m] * v) * v^T
                for i in 0..m {
                    let mut dot_val = 0.0;
                    for (idx, &vi) in v.iter().enumerate() {
                        dot_val += u_acc.get(i, k + idx) * vi;
                    }
                    for (idx, &vi) in v.iter().enumerate() {
                        let old = u_acc.get(i, k + idx);
                        u_acc.set(i, k + idx, old - tau * dot_val * vi);
                    }
                }
            }
        }

        // --- Right Householder: zero out right of work[k][k+1] ---
        if k + 2 <= n {
            let row_slice: Vec<f64> = (k + 1..n).map(|j| work.get(k, j)).collect();
            let (v, tau) = house(&row_slice);
            if tau.abs() > 1e-300 {
                // Apply to work: work[k:m, k+1:n] -= tau * (work[k:m, k+1:n] * v) * v^T
                for i in k..m {
                    let mut dot_val = 0.0;
                    for (idx, &vi) in v.iter().enumerate() {
                        dot_val += work.get(i, k + 1 + idx) * vi;
                    }
                    for (idx, &vi) in v.iter().enumerate() {
                        let old = work.get(i, k + 1 + idx);
                        work.set(i, k + 1 + idx, old - tau * vi * dot_val);
                    }
                }
                // Accumulate into V: V[:, k+1:n] -= tau * (V[:, k+1:n] * v) * v^T
                for i in 0..n {
                    let mut dot_val = 0.0;
                    for (idx, &vi) in v.iter().enumerate() {
                        dot_val += v_acc.get(i, k + 1 + idx) * vi;
                    }
                    for (idx, &vi) in v.iter().enumerate() {
                        let old = v_acc.get(i, k + 1 + idx);
                        v_acc.set(i, k + 1 + idx, old - tau * dot_val * vi);
                    }
                }
            }
        }
    }

    let diag_len = n;
    let mut d = vec![0.0; diag_len];
    let mut e = vec![0.0; if diag_len > 0 { diag_len - 1 } else { 0 }];
    for i in 0..diag_len {
        d[i] = work.get(i, i);
    }
    for i in 0..e.len() {
        e[i] = work.get(i, i + 1);
    }

    Ok((u_acc, d, e, v_acc))
}

// ═══════════════════════════════════════════════════════════════════════════
// Bidiagonal SVD (implicit QR with Golub–Kahan shift)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute SVD of a bidiagonal matrix with diagonal `d` and super-diagonal `e`,
/// accumulating transforms into `u` (m × n) columns and `vt` (n × n) rows.
///
/// On exit `d` contains the singular values (non-negative, descending after sorting)
/// and `e` is zeroed.
pub fn bidiagonal_svd(
    d: &mut Vec<f64>,
    e: &mut Vec<f64>,
    u: &mut DenseMatrix,
    vt: &mut DenseMatrix,
    max_iter: usize,
) -> DecompResult<()> {
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        if d[0] < 0.0 {
            d[0] = -d[0];
            // flip sign of first row of vt
            let cols = vt.cols;
            for j in 0..cols {
                let old = vt.get(0, j);
                vt.set(0, j, -old);
            }
        }
        return Ok(());
    }

    let tol = 1e-14;
    let mut iter_count = 0usize;

    // Working range [p, q] — we deflate from the bottom
    let mut q_end = n - 1; // index of last active super-diagonal

    'outer: loop {
        if iter_count > max_iter * n {
            return Err(DecompError::SvdConvergenceFailure {
                context: format!(
                    "bidiagonal SVD did not converge after {} iterations",
                    iter_count
                ),
            });
        }

        // --- Deflation: zero tiny super-diag entries from bottom ---
        while q_end > 0 {
            let thresh = tol * (d[q_end - 1].abs() + d[q_end].abs()).max(1e-300);
            if e[q_end - 1].abs() <= thresh {
                e[q_end - 1] = 0.0;
                q_end -= 1;
            } else {
                break;
            }
        }
        if q_end == 0 {
            break 'outer; // fully deflated
        }

        // Find p: the top of the unreduced block [p..=q_end]
        let mut p = q_end - 1;
        while p > 0 {
            let thresh = tol * (d[p - 1].abs() + d[p].abs()).max(1e-300);
            if e[p - 1].abs() <= thresh {
                e[p - 1] = 0.0;
                break;
            }
            p -= 1;
        }

        // Check for zero diagonal entries in [p..=q_end]. If d[k]==0 for some k,
        // chase the super-diagonal to zero using Givens from the left.
        let mut found_zero = false;
        for k in p..=q_end {
            if d[k].abs() < tol * 1e-2 {
                d[k] = 0.0;
                // Zero out e[k] (or e[k-1]) by chasing with left rotations
                if k < q_end {
                    // Chase e[k] to zero
                    let mut bulge = e[k];
                    e[k] = 0.0;
                    for j in (k + 1)..=q_end {
                        let (c, s, r) = GivensRotation::compute(d[j], bulge);
                        d[j] = r;
                        if j < q_end {
                            bulge = s * e[j];
                            e[j] = c * e[j];
                        }
                        // Apply to U columns k and j
                        let rot = GivensRotation::from_cs(k, j, c, s);
                        rot.apply_right(u, 0, u.rows);
                    }
                } else if k > p && k - 1 < e.len() {
                    // Chase e[k-1] to zero
                    let mut bulge = e[k - 1];
                    e[k - 1] = 0.0;
                    for j in (p..k).rev() {
                        let (c, s, r) = GivensRotation::compute(d[j], bulge);
                        d[j] = r;
                        if j > p {
                            bulge = s * e[j - 1];
                            e[j - 1] = c * e[j - 1];
                        }
                        let rot = GivensRotation::from_cs(j, k, c, s);
                        rot.apply_left(vt, 0, vt.cols);
                    }
                }
                found_zero = true;
                break;
            }
        }
        if found_zero {
            iter_count += 1;
            continue 'outer;
        }

        // --- Golub–Kahan SVD step with Wilkinson shift ---
        // Compute shift from trailing 2×2 of T = B^T B
        let dm1 = d[q_end - 1];
        let dm = d[q_end];
        let em1 = e[q_end - 1];
        let em2 = if q_end >= 2 { e[q_end - 2] } else { 0.0 };

        let t11 = dm1 * dm1 + em2 * em2;
        let t12 = dm1 * em1;
        let t22 = dm * dm + em1 * em1;

        // Wilkinson shift: eigenvalue of [[t11,t12],[t12,t22]] closer to t22
        let half = 0.5 * (t11 - t22);
        let shift = t22 - t12 * t12
            / (half + half.signum() * (half * half + t12 * t12).sqrt() + if half == 0.0 { 1e-300 } else { 0.0 });

        // Initial bulge creation
        let mut y = d[p] * d[p] - shift;
        let mut z = d[p] * e[p];

        // Chase the bulge
        for k in p..q_end {
            // Right rotation: zero z in column [y; z]
            let (c, s, r) = GivensRotation::compute(y, z);
            if k > p {
                e[k - 1] = r;
            }
            y = c * d[k] + s * e[k];
            e[k] = -s * d[k] + c * e[k];
            z = s * d[k + 1];
            d[k + 1] = c * d[k + 1];

            // Accumulate right rotation into Vt rows k, k+1
            let rot_r = GivensRotation::from_cs(k, k + 1, c, s);
            rot_r.apply_left(vt, 0, vt.cols);

            // Left rotation: zero z in row [y; z]
            let (c2, s2, r2) = GivensRotation::compute(y, z);
            d[k] = r2;
            y = c2 * e[k] + s2 * d[k + 1];
            d[k + 1] = -s2 * e[k] + c2 * d[k + 1];
            if k + 1 < q_end {
                z = s2 * e[k + 1];
                e[k + 1] = c2 * e[k + 1];
            }

            // Accumulate left rotation into U columns k, k+1
            let rot_l = GivensRotation::from_cs(k, k + 1, c2, s2);
            rot_l.apply_right(u, 0, u.rows);
        }
        e[q_end - 1] = y;

        iter_count += 1;
    }

    // Make all singular values non-negative
    for i in 0..n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            let cols = vt.cols;
            for j in 0..cols {
                let old = vt.get(i, j);
                vt.set(i, j, -old);
            }
        }
    }

    // Sort singular values in descending order (and permute U, Vt accordingly)
    sort_svd(d, u, vt);

    Ok(())
}

/// Sort singular values descending and permute U columns / Vt rows to match.
fn sort_svd(s: &mut Vec<f64>, u: &mut DenseMatrix, vt: &mut DenseMatrix) {
    let n = s.len();
    // Simple selection sort (n is typically small after deflation)
    for i in 0..n {
        let mut max_idx = i;
        let mut max_val = s[i];
        for j in (i + 1)..n {
            if s[j] > max_val {
                max_val = s[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            s.swap(i, max_idx);
            // Swap columns i and max_idx in U
            for r in 0..u.rows {
                let a = u.get(r, i);
                let b = u.get(r, max_idx);
                u.set(r, i, b);
                u.set(r, max_idx, a);
            }
            // Swap rows i and max_idx in Vt
            let cn = vt.cols;
            for c in 0..cn {
                let va = vt.get(i, c);
                let vb = vt.get(max_idx, c);
                vt.set(i, c, vb);
                vt.set(max_idx, c, va);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Jacobi SVD (one-sided, for small matrices)
// ═══════════════════════════════════════════════════════════════════════════

/// One-sided Jacobi SVD.
///
/// Works directly on columns of A.  Suitable for matrices with min(m,n) ≤ 64.
/// Returns `SvdDecomposition`.
pub fn jacobi_svd(
    a: &DenseMatrix,
    max_iter: usize,
    tol: f64,
) -> DecompResult<SvdDecomposition> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(DecompError::EmptyMatrix {
            context: "jacobi_svd: empty matrix".into(),
        });
    }

    let mut work = a.clone();
    let mut v = DenseMatrix::eye(n);

    for _sweep in 0..max_iter {
        // Compute off-diagonal norm of A^T A to check convergence
        let mut off = 0.0;
        let mut diag_sum = 0.0;
        for i in 0..n {
            let ci = work.col(i);
            diag_sum += dot(&ci, &ci);
            for j in (i + 1)..n {
                let cj = work.col(j);
                let val = dot(&ci, &cj);
                off += val * val;
            }
        }
        if off.sqrt() <= tol * diag_sum.sqrt().max(1e-300) {
            break;
        }

        // Sweep all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let ci = work.col(i);
                let cj = work.col(j);

                let alpha = dot(&ci, &ci);
                let beta = dot(&cj, &cj);
                let gamma = dot(&ci, &cj);

                if gamma.abs() < tol * (alpha * beta).sqrt().max(1e-300) {
                    continue;
                }

                // Compute Jacobi rotation angle for the 2×2 symmetric
                // [[alpha, gamma], [gamma, beta]]
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta.abs() > 1e15 {
                    // Avoid overflow: t ≈ 1/(2*zeta)
                    1.0 / (2.0 * zeta)
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                // Apply rotation to columns i, j of work
                for r in 0..m {
                    let wi = work.get(r, i);
                    let wj = work.get(r, j);
                    work.set(r, i, c * wi - s * wj);
                    work.set(r, j, s * wi + c * wj);
                }

                // Update V
                for r in 0..n {
                    let vi = v.get(r, i);
                    let vj = v.get(r, j);
                    v.set(r, i, c * vi - s * vj);
                    v.set(r, j, s * vi + c * vj);
                }
            }
        }
    }

    // Singular values = column norms; normalise columns of work to get U
    let k = n.min(m);
    let mut s_vals = vec![0.0; k];
    let mut u_mat = DenseMatrix::zeros(m, k);

    for j in 0..k {
        let col_j = work.col(j);
        let nrm = norm2(&col_j);
        s_vals[j] = nrm;
        if nrm > 1e-300 {
            for i in 0..m {
                u_mat.set(i, j, col_j[i] / nrm);
            }
        } else {
            // zero singular value – leave column as zero, will fill with
            // arbitrary orthonormal later if needed
            for i in 0..m {
                u_mat.set(i, j, 0.0);
            }
        }
    }

    // Sort descending
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| s_vals[b].partial_cmp(&s_vals[a]).unwrap_or(std::cmp::Ordering::Equal));

    let s_sorted: Vec<f64> = order.iter().map(|&i| s_vals[i]).collect();
    let u_sorted = DenseMatrix::from_fn(m, k, |r, c| u_mat.get(r, order[c]));
    let vt_sorted = DenseMatrix::from_fn(k, n, |r, c| v.get(c, order[r]));

    Ok(SvdDecomposition {
        u: u_sorted,
        s: s_sorted,
        vt: vt_sorted,
        m,
        n,
        k,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point: factorize
// ═══════════════════════════════════════════════════════════════════════════

impl SvdDecomposition {
    /// Compute the full SVD of a dense matrix.
    ///
    /// For small matrices (min(m,n) ≤ 64) uses one-sided Jacobi.
    /// For larger matrices uses Golub–Kahan bidiagonalisation followed by
    /// the implicit QR algorithm (with Wilkinson shift).
    pub fn factorize(a: &DenseMatrix) -> DecompResult<Self> {
        let (m, n) = a.shape();
        if m == 0 || n == 0 {
            return Err(DecompError::EmptyMatrix {
                context: "SVD factorize: empty matrix".into(),
            });
        }

        // For 1×1 matrix, trivial
        if m == 1 && n == 1 {
            let val = a.get(0, 0);
            let sign = if val >= 0.0 { 1.0 } else { -1.0 };
            return Ok(SvdDecomposition {
                u: DenseMatrix::from_row_major(1, 1, vec![sign]),
                s: vec![val.abs()],
                vt: DenseMatrix::from_row_major(1, 1, vec![1.0]),
                m: 1,
                n: 1,
                k: 1,
            });
        }

        // For 2×2, use analytic formula
        if m == 2 && n == 2 {
            let (u2, s2, vt2) = svd_2x2(a.get(0, 0), a.get(0, 1), a.get(1, 0), a.get(1, 1));
            return Ok(SvdDecomposition {
                u: u2,
                s: s2.to_vec(),
                vt: vt2,
                m: 2,
                n: 2,
                k: 2,
            });
        }

        let small_threshold = 64;
        if m.min(n) <= small_threshold {
            return jacobi_svd(a, 100, 1e-14);
        }

        // Large matrix: Golub–Kahan + implicit QR
        Self::factorize_golub_kahan(a)
    }

    /// Golub–Kahan bidiagonalisation + bidiagonal SVD.
    fn factorize_golub_kahan(a: &DenseMatrix) -> DecompResult<Self> {
        let (m, n) = a.shape();
        let transposed = m < n;
        let work = if transposed { a.transpose() } else { a.clone() };
        let (mw, nw) = work.shape();

        let (u_bidiag, mut d, mut e, v_bidiag) = bidiagonalize_golub_kahan(&work)?;

        // u_bidiag is mw×mw, v_bidiag is nw×nw
        // We need to accumulate bidiag SVD transforms into them.
        let mut u_full = u_bidiag;
        let vt_full = v_bidiag.transpose();

        // Create sub-matrices for accumulation: only the first nw columns of U
        // and full Vt matter.
        let mut u_sub = DenseMatrix::from_fn(mw, nw, |r, c| u_full.get(r, c));
        let mut vt_sub = vt_full.clone();

        bidiagonal_svd(&mut d, &mut e, &mut u_sub, &mut vt_sub, 60)?;

        // Write back accumulated U columns
        for r in 0..mw {
            for c in 0..nw {
                u_full.set(r, c, u_sub.get(r, c));
            }
        }

        let k = nw;

        if transposed {
            // A^T = U * S * Vt ⇒ A = V * S * Ut
            let u_out = vt_sub.transpose(); // nw × nw → n × k (since nw = original n is the smaller)
            let vt_out = DenseMatrix::from_fn(k, m, |r, c| u_full.get(c, r)); // transpose of U_full[:, 0:k]
            Ok(SvdDecomposition {
                u: u_out,
                s: d,
                vt: vt_out,
                m,
                n,
                k,
            })
        } else {
            let u_out = DenseMatrix::from_fn(m, k, |r, c| u_full.get(r, c));
            let vt_out = vt_sub;
            Ok(SvdDecomposition {
                u: u_out,
                s: d,
                vt: vt_out,
                m,
                n,
                k,
            })
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Derived quantities
    // ═══════════════════════════════════════════════════════════════════════

    /// Numerical rank: count of singular values above `tol`.
    pub fn rank(&self, tol: f64) -> usize {
        self.s.iter().filter(|&&si| si > tol).count()
    }

    /// Condition number σ_max / σ_min (returns `f64::INFINITY` if smallest σ ≈ 0).
    pub fn condition_number(&self) -> f64 {
        if self.s.is_empty() {
            return f64::INFINITY;
        }
        let s_max = self.s[0];
        let s_min = *self.s.last().unwrap();
        if s_min < 1e-300 {
            f64::INFINITY
        } else {
            s_max / s_min
        }
    }

    /// Moore–Penrose pseudoinverse A⁺ = V · diag(1/σᵢ) · Uᵀ.
    ///
    /// Singular values below `tol * σ_max` are treated as zero.
    pub fn pseudoinverse(&self) -> DecompResult<DenseMatrix> {
        let tol = 1e-12 * self.s.first().copied().unwrap_or(0.0);
        // A+ = V * S_inv * U^T  where V = Vt^T, dimensions: (n×k) * (k×k) * (k×m) = n×m
        let mut result = DenseMatrix::zeros(self.n, self.m);

        for idx in 0..self.k {
            let si = self.s[idx];
            if si <= tol {
                continue;
            }
            let inv_s = 1.0 / si;
            // Rank-1 update: result += inv_s * v_col * u_row
            // v_col is column idx of V = row idx of Vt transposed → Vt[idx, :]^T
            // u_row is row idx of U^T = column idx of U → U[:, idx]
            for r in 0..self.n {
                let v_ri = self.vt.get(idx, r); // Vt[idx, r] → V[r, idx]
                for c in 0..self.m {
                    let u_ci = self.u.get(c, idx);
                    let old = result.get(r, c);
                    result.set(r, c, old + inv_s * v_ri * u_ci);
                }
            }
        }

        Ok(result)
    }

    /// Best rank-`target_k` approximation: U_k · diag(S_k) · Vt_k.
    pub fn low_rank_approx(&self, target_k: usize) -> DecompResult<DenseMatrix> {
        if target_k == 0 {
            return Ok(DenseMatrix::zeros(self.m, self.n));
        }
        let kk = target_k.min(self.k);

        let mut result = DenseMatrix::zeros(self.m, self.n);
        for idx in 0..kk {
            let si = self.s[idx];
            for r in 0..self.m {
                let u_ri = self.u.get(r, idx);
                for c in 0..self.n {
                    let vt_ic = self.vt.get(idx, c);
                    let old = result.get(r, c);
                    result.set(r, c, old + si * u_ri * vt_ic);
                }
            }
        }
        Ok(result)
    }

    /// Solve the least-squares problem min ‖Ax − b‖₂ via SVD.
    pub fn solve_least_squares(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        if b.len() != self.m {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.m,
                actual: b.len(),
            });
        }
        let tol = 1e-12 * self.s.first().copied().unwrap_or(0.0);

        // x = V · S_inv · U^T · b
        // First compute c = U^T * b (length k)
        let mut c = vec![0.0; self.k];
        for i in 0..self.k {
            let mut sum = 0.0;
            for j in 0..self.m {
                sum += self.u.get(j, i) * b[j];
            }
            c[i] = sum;
        }

        // Apply S_inv
        for i in 0..self.k {
            if self.s[i] > tol {
                c[i] /= self.s[i];
            } else {
                c[i] = 0.0;
            }
        }

        // x = Vt^T * c (Vt is k × n, so Vt^T is n × k)
        let mut x = vec![0.0; self.n];
        for j in 0..self.n {
            let mut sum = 0.0;
            for i in 0..self.k {
                sum += self.vt.get(i, j) * c[i];
            }
            x[j] = sum;
        }

        Ok(x)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Thin QR (used by randomised SVD)
// ═══════════════════════════════════════════════════════════════════════════

/// Thin QR decomposition via modified Gram–Schmidt, returning (Q, R).
/// Q is m×n, R is n×n (upper triangular). Requires m ≥ n.
fn thin_qr(a: &DenseMatrix) -> (DenseMatrix, DenseMatrix) {
    let (m, n) = a.shape();
    let mut q = a.clone();
    let mut r = DenseMatrix::zeros(n, n);

    for j in 0..n {
        // Orthogonalise column j against previous columns
        for i in 0..j {
            let qi = q.col(i);
            let qj = q.col(j);
            let rij = dot(&qi, &qj);
            r.set(i, j, rij);
            for row in 0..m {
                let old = q.get(row, j);
                q.set(row, j, old - rij * qi[row]);
            }
        }
        // Re-orthogonalise (CGS2)
        for i in 0..j {
            let qi = q.col(i);
            let qj = q.col(j);
            let correction = dot(&qi, &qj);
            let old_r = r.get(i, j);
            r.set(i, j, old_r + correction);
            for row in 0..m {
                let old = q.get(row, j);
                q.set(row, j, old - correction * qi[row]);
            }
        }
        let mut cj = q.col(j);
        let nrm = normalize(&mut cj);
        r.set(j, j, nrm);
        for row in 0..m {
            q.set(row, j, cj[row]);
        }
    }

    (q, r)
}

// ═══════════════════════════════════════════════════════════════════════════
// Randomised SVD (Halko–Martinsson–Tropp)
// ═══════════════════════════════════════════════════════════════════════════

/// Randomised SVD for dense matrices.
///
/// Computes an approximate rank-`k` SVD using the Halko–Martinsson–Tropp
/// algorithm with `oversampling` extra random vectors and `n_power_iter`
/// power iterations for improved accuracy.
pub fn randomized_svd(
    a: &DenseMatrix,
    k: usize,
    oversampling: usize,
    n_power_iter: usize,
) -> DecompResult<SvdDecomposition> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(DecompError::EmptyMatrix {
            context: "randomized_svd: empty matrix".into(),
        });
    }
    let p = (k + oversampling).min(n).min(m);

    // Step 1: Generate random Gaussian matrix Ω (n × p)
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let omega_data: Vec<f64> = (0..n * p)
        .map(|_| {
            let u1: f64 = rng.gen::<f64>().max(1e-300);
            let u2: f64 = rng.gen::<f64>();
            (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
        })
        .collect();
    let omega = DenseMatrix::from_row_major(n, p, omega_data);

    // Step 2: Form Y = A * Ω (m × p)
    let mut y = a.mul_mat(&omega)?;

    // Step 3: Power iteration for improved range approximation
    let at = a.transpose();
    for _ in 0..n_power_iter {
        let z = at.mul_mat(&y)?; // n × p
        let (q_z, _) = thin_qr(&z);
        y = a.mul_mat(&q_z)?;
    }

    // Step 4: QR of Y to get orthonormal basis Q
    let (q, _) = thin_qr(&y); // m × p

    // Step 5: Form B = Q^T * A (p × n)
    let qt = q.transpose();
    let b = qt.mul_mat(a)?; // p × n

    // Step 6: SVD of the small matrix B
    let svd_b = if p <= 64 {
        jacobi_svd(&b, 100, 1e-14)?
    } else {
        SvdDecomposition::factorize(&b)?
    };

    // Step 7: Recover U = Q * U_B
    let u_approx = q.mul_mat(&svd_b.u)?; // m × p

    // Truncate to k
    let kk = k.min(svd_b.k);
    let u_out = DenseMatrix::from_fn(m, kk, |r, c| u_approx.get(r, c));
    let s_out: Vec<f64> = svd_b.s[..kk].to_vec();
    let vt_out = DenseMatrix::from_fn(kk, n, |r, c| svd_b.vt.get(r, c));

    Ok(SvdDecomposition {
        u: u_out,
        s: s_out,
        vt: vt_out,
        m,
        n,
        k: kk,
    })
}

/// Randomised SVD for sparse matrices (CSR format).
pub fn randomized_svd_sparse(
    a: &CsrMatrix,
    k: usize,
    oversampling: usize,
    n_power_iter: usize,
) -> DecompResult<SvdDecomposition> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(DecompError::EmptyMatrix {
            context: "randomized_svd_sparse: empty matrix".into(),
        });
    }
    let p = (k + oversampling).min(n).min(m);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let at = a.transpose(); // CsrMatrix

    // Step 1 & 2: Y = A * Ω, column by column
    let mut y = DenseMatrix::zeros(m, p);
    for j in 0..p {
        let omega_col: Vec<f64> = (0..n).map(|_| {
            let u1: f64 = rng.gen::<f64>().max(1e-300);
            let u2: f64 = rng.gen::<f64>();
            (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
        }).collect();
        let col = a.mul_vec(&omega_col)?;
        for i in 0..m {
            y.set(i, j, col[i]);
        }
    }

    // Step 3: Power iteration
    for _ in 0..n_power_iter {
        let (q_y, _) = thin_qr(&y);
        // Z = A^T * Q_y
        let mut z = DenseMatrix::zeros(n, p);
        for j in 0..p {
            let q_col = q_y.col(j);
            let zj = at.mul_vec(&q_col)?;
            for i in 0..n {
                z.set(i, j, zj[i]);
            }
        }
        let (q_z, _) = thin_qr(&z);
        // Y = A * Q_z
        y = DenseMatrix::zeros(m, p);
        for j in 0..p {
            let qz_col = q_z.col(j);
            let yj = a.mul_vec(&qz_col)?;
            for i in 0..m {
                y.set(i, j, yj[i]);
            }
        }
    }

    // Step 4: QR of Y
    let (q, _) = thin_qr(&y);

    // Step 5: B = Q^T * A via sparse matvec on A^T
    // B[i, j] = Q[:,i]^T * A[:, j] = (A^T * Q[:,i])[j]... no, that gives A^T rows.
    // Better: B = Q^T * A. B[i,:] = Q[:,i]^T * A → we compute each row of B
    // as A^T * q_i where q_i is column i of Q, giving a column of B^T... 
    // Actually: (Q^T * A)^T = A^T * Q, so compute A^T * Q column by column.
    let mut bt = DenseMatrix::zeros(n, p); // B^T
    for j in 0..p {
        let q_col = q.col(j);
        let bt_col = at.mul_vec(&q_col)?; // A^T * q_j → row j of B = column j of B^T
        for i in 0..n {
            bt.set(i, j, bt_col[i]);
        }
    }
    let b = bt.transpose(); // p × n

    // Step 6: SVD of small B
    let svd_b = if p <= 64 {
        jacobi_svd(&b, 100, 1e-14)?
    } else {
        SvdDecomposition::factorize(&b)?
    };

    // Step 7: U = Q * U_B
    let u_approx = q.mul_mat(&svd_b.u)?;

    let kk = k.min(svd_b.k);
    let u_out = DenseMatrix::from_fn(m, kk, |r, c| u_approx.get(r, c));
    let s_out: Vec<f64> = svd_b.s[..kk].to_vec();
    let vt_out = DenseMatrix::from_fn(kk, n, |r, c| svd_b.vt.get(r, c));

    Ok(SvdDecomposition {
        u: u_out,
        s: s_out,
        vt: vt_out,
        m,
        n,
        k: kk,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that A ≈ U * diag(S) * Vt within tolerance.
    fn check_reconstruction(a: &DenseMatrix, svd: &SvdDecomposition, tol: f64) {
        let (m, n) = a.shape();
        for r in 0..m {
            for c in 0..n {
                let mut val = 0.0;
                for idx in 0..svd.k {
                    val += svd.u.get(r, idx) * svd.s[idx] * svd.vt.get(idx, c);
                }
                let diff = (val - a.get(r, c)).abs();
                assert!(
                    diff < tol,
                    "reconstruction error at ({},{}): {} vs {} (diff={})",
                    r, c, val, a.get(r, c), diff
                );
            }
        }
    }

    /// Check that columns of U are orthonormal.
    fn check_orthonormal_cols(u: &DenseMatrix, tol: f64) {
        let k = u.cols;
        for i in 0..k {
            let ci = u.col(i);
            let nrm = norm2(&ci);
            assert!(
                (nrm - 1.0).abs() < tol,
                "column {} norm = {} (expected 1.0)",
                i, nrm
            );
            for j in (i + 1)..k {
                let cj = u.col(j);
                let d = dot(&ci, &cj).abs();
                assert!(
                    d < tol,
                    "columns {} and {} dot product = {} (expected 0)",
                    i, j, d
                );
            }
        }
    }

    /// Check that rows of Vt are orthonormal.
    fn check_orthonormal_rows(vt: &DenseMatrix, tol: f64) {
        let k = vt.rows;
        for i in 0..k {
            let ri: Vec<f64> = (0..vt.cols).map(|c| vt.get(i, c)).collect();
            let nrm = norm2(&ri);
            assert!(
                (nrm - 1.0).abs() < tol,
                "row {} norm = {} (expected 1.0)",
                i, nrm
            );
            for j in (i + 1)..k {
                let rj: Vec<f64> = (0..vt.cols).map(|c| vt.get(j, c)).collect();
                let d = dot(&ri, &rj).abs();
                assert!(
                    d < tol,
                    "rows {} and {} dot product = {} (expected 0)",
                    i, j, d
                );
            }
        }
    }

    #[test]
    fn test_svd_1x1() {
        let a = DenseMatrix::from_row_major(1, 1, vec![-3.5]);
        let svd = SvdDecomposition::factorize(&a).unwrap();
        assert_eq!(svd.k, 1);
        assert!((svd.s[0] - 3.5).abs() < 1e-12);
        check_reconstruction(&a, &svd, 1e-12);
    }

    #[test]
    fn test_svd_2x2() {
        let a = DenseMatrix::from_row_major(2, 2, vec![3.0, 2.0, 2.0, 3.0]);
        let svd = SvdDecomposition::factorize(&a).unwrap();
        assert_eq!(svd.k, 2);
        assert!((svd.s[0] - 5.0).abs() < 1e-10, "s0={}", svd.s[0]);
        assert!((svd.s[1] - 1.0).abs() < 1e-10, "s1={}", svd.s[1]);
        check_reconstruction(&a, &svd, 1e-10);
    }

    #[test]
    fn test_svd_diagonal() {
        let a = DenseMatrix::from_row_major(
            3, 3,
            vec![5.0, 0.0, 0.0,
                 0.0, 3.0, 0.0,
                 0.0, 0.0, 1.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        assert!((svd.s[0] - 5.0).abs() < 1e-10);
        assert!((svd.s[1] - 3.0).abs() < 1e-10);
        assert!((svd.s[2] - 1.0).abs() < 1e-10);
        check_reconstruction(&a, &svd, 1e-10);
    }

    #[test]
    fn test_svd_known_3x3() {
        // A known matrix with analytically computable singular values
        let a = DenseMatrix::from_row_major(
            3, 3,
            vec![1.0, 0.0, 0.0,
                 0.0, 2.0, 0.0,
                 0.0, 0.0, 3.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        assert!((svd.s[0] - 3.0).abs() < 1e-10);
        assert!((svd.s[1] - 2.0).abs() < 1e-10);
        assert!((svd.s[2] - 1.0).abs() < 1e-10);
        check_reconstruction(&a, &svd, 1e-10);
        check_orthonormal_cols(&svd.u, 1e-10);
        check_orthonormal_rows(&svd.vt, 1e-10);
    }

    #[test]
    fn test_svd_rectangular_tall() {
        // 4×2 matrix
        let a = DenseMatrix::from_row_major(
            4, 2,
            vec![1.0, 2.0,
                 3.0, 4.0,
                 5.0, 6.0,
                 7.0, 8.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        assert_eq!(svd.k, 2);
        assert_eq!(svd.u.rows, 4);
        assert_eq!(svd.u.cols, 2);
        assert_eq!(svd.vt.rows, 2);
        assert_eq!(svd.vt.cols, 2);
        check_reconstruction(&a, &svd, 1e-10);
        check_orthonormal_cols(&svd.u, 1e-10);
        check_orthonormal_rows(&svd.vt, 1e-10);
    }

    #[test]
    fn test_svd_rectangular_wide() {
        // 2×4 matrix
        let a = DenseMatrix::from_row_major(
            2, 4,
            vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        assert_eq!(svd.k, 2);
        check_reconstruction(&a, &svd, 1e-10);
        check_orthonormal_cols(&svd.u, 1e-10);
        check_orthonormal_rows(&svd.vt, 1e-10);
    }

    #[test]
    fn test_svd_singular_values_nonneg_sorted() {
        let a = DenseMatrix::from_row_major(
            3, 3,
            vec![1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0,
                 7.0, 8.0, 10.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        for &si in &svd.s {
            assert!(si >= 0.0, "singular value {} is negative", si);
        }
        for i in 1..svd.s.len() {
            assert!(
                svd.s[i - 1] >= svd.s[i] - 1e-12,
                "not sorted descending: s[{}]={} < s[{}]={}",
                i - 1, svd.s[i - 1], i, svd.s[i]
            );
        }
    }

    #[test]
    fn test_svd_rank() {
        // Rank-1 matrix
        let a = DenseMatrix::from_row_major(
            3, 3,
            vec![1.0, 2.0, 3.0,
                 2.0, 4.0, 6.0,
                 3.0, 6.0, 9.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        assert_eq!(svd.rank(1e-10), 1);
    }

    #[test]
    fn test_svd_condition_number() {
        let a = DenseMatrix::from_row_major(
            2, 2,
            vec![2.0, 0.0,
                 0.0, 1.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        let cond = svd.condition_number();
        assert!((cond - 2.0).abs() < 1e-10, "condition number = {}", cond);
    }

    #[test]
    fn test_svd_pseudoinverse() {
        let a = DenseMatrix::from_row_major(
            3, 2,
            vec![1.0, 0.0,
                 0.0, 1.0,
                 0.0, 0.0],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        let pinv = svd.pseudoinverse().unwrap();
        // A+ should be 2×3: [[1,0,0],[0,1,0]]
        assert_eq!(pinv.rows, 2);
        assert_eq!(pinv.cols, 3);
        assert!((pinv.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((pinv.get(1, 1) - 1.0).abs() < 1e-10);
        assert!(pinv.get(0, 2).abs() < 1e-10);
        assert!(pinv.get(1, 2).abs() < 1e-10);
    }

    #[test]
    fn test_svd_low_rank_approx() {
        // Rank-2 matrix padded with a tiny rank-1 component
        let a = DenseMatrix::from_row_major(
            3, 3,
            vec![3.0, 0.0, 0.0,
                 0.0, 2.0, 0.0,
                 0.0, 0.0, 0.001],
        );
        let svd = SvdDecomposition::factorize(&a).unwrap();
        let approx = svd.low_rank_approx(2).unwrap();
        // Should be close to the matrix with the 0.001 replaced by 0
        assert!((approx.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((approx.get(1, 1) - 2.0).abs() < 1e-10);
        assert!(approx.get(2, 2).abs() < 1e-10);
    }

    #[test]
    fn test_svd_least_squares() {
        // Overdetermined system: A is 3×2
        let a = DenseMatrix::from_row_major(
            3, 2,
            vec![1.0, 0.0,
                 0.0, 1.0,
                 1.0, 1.0],
        );
        let b = vec![1.0, 2.0, 2.5];
        let svd = SvdDecomposition::factorize(&a).unwrap();
        let x = svd.solve_least_squares(&b).unwrap();
        // Verify A^T A x ≈ A^T b
        let at = a.transpose();
        let atb = at.mul_vec(&b).unwrap();
        let ata = at.mul_mat(&a).unwrap();
        let atax = ata.mul_vec(&x).unwrap();
        for i in 0..2 {
            assert!(
                (atax[i] - atb[i]).abs() < 1e-10,
                "normal equation mismatch at {}: {} vs {}",
                i, atax[i], atb[i]
            );
        }
    }

    #[test]
    fn test_randomized_svd() {
        // 10×8 matrix with known rank-2 structure
        let mut a = DenseMatrix::zeros(10, 8);
        for i in 0..10 {
            for j in 0..8 {
                a.set(i, j, (i as f64 + 1.0) * (j as f64 + 1.0));
            }
        }
        // This is rank-1; add a rank-2 component
        for i in 0..10 {
            for j in 0..8 {
                let old = a.get(i, j);
                let extra = if i % 2 == 0 { (j as f64) * 0.5 } else { -(j as f64) * 0.5 };
                a.set(i, j, old + extra);
            }
        }

        let rsvd = randomized_svd(&a, 2, 5, 2).unwrap();
        assert_eq!(rsvd.k, 2);

        // Reconstruct and check that the top-2 singular values capture most energy
        let full_svd = SvdDecomposition::factorize(&a).unwrap();
        let total_energy: f64 = full_svd.s.iter().map(|si| si * si).sum();
        let captured: f64 = rsvd.s.iter().map(|si| si * si).sum();
        assert!(
            captured / total_energy > 0.99,
            "randomised SVD should capture >99% energy, got {}%",
            100.0 * captured / total_energy
        );
    }

    #[test]
    fn test_randomized_svd_sparse() {
        // Create a sparse rank-2 matrix
        let mut triplets = Vec::new();
        for i in 0..20 {
            for j in 0..15 {
                let val = (i as f64 + 1.0) * (j as f64 + 1.0)
                    + if i % 3 == 0 { (j as f64) * 2.0 } else { 0.0 };
                if val.abs() > 0.1 {
                    triplets.push((i, j, val));
                }
            }
        }
        let csr = CsrMatrix::from_triplets(20, 15, &triplets);
        let rsvd = randomized_svd_sparse(&csr, 2, 5, 2).unwrap();
        assert_eq!(rsvd.k, 2);

        // Compare against dense SVD
        let dense = csr.to_dense();
        let full_svd = SvdDecomposition::factorize(&dense).unwrap();
        // Top 2 singular values should match reasonably
        for i in 0..2 {
            let rel = (rsvd.s[i] - full_svd.s[i]).abs() / full_svd.s[i].max(1e-15);
            assert!(
                rel < 0.05,
                "singular value {} mismatch: rsvd={} full={}",
                i, rsvd.s[i], full_svd.s[i]
            );
        }
    }

    #[test]
    fn test_jacobi_vs_full_agreement() {
        // Use a 5×5 matrix small enough for Jacobi
        let a = DenseMatrix::from_row_major(
            5, 5,
            vec![
                2.0, -1.0, 0.0, 0.0, 0.0,
                -1.0, 2.0, -1.0, 0.0, 0.0,
                0.0, -1.0, 2.0, -1.0, 0.0,
                0.0, 0.0, -1.0, 2.0, -1.0,
                0.0, 0.0, 0.0, -1.0, 2.0,
            ],
        );
        let jac = jacobi_svd(&a, 200, 1e-14).unwrap();
        check_reconstruction(&a, &jac, 1e-10);
        check_orthonormal_cols(&jac.u, 1e-10);
        check_orthonormal_rows(&jac.vt, 1e-10);

        // Singular values should be non-negative and sorted
        for &si in &jac.s {
            assert!(si >= -1e-14, "negative singular value {}", si);
        }
        for i in 1..jac.s.len() {
            assert!(jac.s[i - 1] >= jac.s[i] - 1e-12);
        }
    }

    #[test]
    fn test_svd_2x2_analytic() {
        let (u, s, vt) = svd_2x2(3.0, 1.0, 1.0, 3.0);
        // Eigenvalues of A^T A: 10±6 → σ = √16, √4 = 4, 2
        assert!((s[0] - 4.0).abs() < 1e-10, "s0={}", s[0]);
        assert!((s[1] - 2.0).abs() < 1e-10, "s1={}", s[1]);

        // Reconstruct
        let a = DenseMatrix::from_row_major(2, 2, vec![3.0, 1.0, 1.0, 3.0]);
        let svd = SvdDecomposition {
            u: u.clone(),
            s: s.to_vec(),
            vt: vt.clone(),
            m: 2,
            n: 2,
            k: 2,
        };
        check_reconstruction(&a, &svd, 1e-10);
    }
}
