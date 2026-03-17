//! Eigenvalue decomposition algorithms for dense and sparse matrices.
//!
//! Provides symmetric / non-symmetric eigensolvers, direct 2×2/3×3 solvers,
//! QR algorithm with Wilkinson shifts, Jacobi rotations, Lanczos iteration for
//! sparse matrices, Arnoldi iteration, Francis double-shift QR, and shift-invert.

use crate::{axpy, copy_vec, dot, normalize, norm2, scale_vec, CsrMatrix, DecompError, DecompResult, DenseMatrix};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ═══════════════════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════════════════

/// Method selector for eigenvalue computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenMethod {
    Auto,
    QrAlgorithm,
    Lanczos,
    Jacobi,
}

/// Configuration for eigenvalue solvers.
#[derive(Debug, Clone)]
pub struct EigenConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub compute_vectors: bool,
    pub method: EigenMethod,
}

impl Default for EigenConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
            compute_vectors: true,
            method: EigenMethod::Auto,
        }
    }
}

impl EigenConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }
    pub fn with_vectors(mut self, v: bool) -> Self {
        self.compute_vectors = v;
        self
    }
    pub fn with_method(mut self, m: EigenMethod) -> Self {
        self.method = m;
        self
    }
}

/// Result of a symmetric eigenvalue decomposition.
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Option<DenseMatrix>,
    pub n: usize,
}

impl EigenDecomposition {
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> Option<&DenseMatrix> {
        self.eigenvectors.as_ref()
    }

    /// Sort eigenvalues (and eigenvectors) by descending magnitude.
    pub fn sort_by_magnitude(&mut self) {
        let n = self.eigenvalues.len();
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&a, &b| {
            self.eigenvalues[b]
                .abs()
                .partial_cmp(&self.eigenvalues[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.apply_permutation(&perm);
    }

    /// Sort eigenvalues (and eigenvectors) in ascending order.
    pub fn sort_ascending(&mut self) {
        let n = self.eigenvalues.len();
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&a, &b| {
            self.eigenvalues[a]
                .partial_cmp(&self.eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.apply_permutation(&perm);
    }

    fn apply_permutation(&mut self, perm: &[usize]) {
        let n = perm.len();
        let old_vals = self.eigenvalues.clone();
        for i in 0..n {
            self.eigenvalues[i] = old_vals[perm[i]];
        }
        if let Some(ref mut vecs) = self.eigenvectors {
            let old = vecs.clone();
            for i in 0..n {
                for r in 0..vecs.rows {
                    vecs.set(r, i, old.get(r, perm[i]));
                }
            }
        }
    }
}

/// Result of a non-symmetric eigenvalue decomposition.
#[derive(Debug, Clone)]
pub struct NonSymmetricEigen {
    pub real_eigenvalues: Vec<f64>,
    pub imag_eigenvalues: Vec<f64>,
    pub schur_form: Option<DenseMatrix>,
    pub schur_vectors: Option<DenseMatrix>,
}

/// Convergence monitor tracking residuals per iteration.
#[derive(Debug, Clone)]
pub struct EigenConvergenceMonitor {
    pub residuals: Vec<f64>,
}

impl EigenConvergenceMonitor {
    pub fn new() -> Self {
        Self {
            residuals: Vec::new(),
        }
    }
    pub fn record(&mut self, r: f64) {
        self.residuals.push(r);
    }
    pub fn converged(&self, tol: f64) -> bool {
        self.residuals.last().map_or(false, |&r| r < tol)
    }
    pub fn iterations(&self) -> usize {
        self.residuals.len()
    }
}

impl Default for EigenConvergenceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Direct solvers for tiny matrices
// ═══════════════════════════════════════════════════════════════════════════

/// Eigenvalues of a symmetric 2×2 matrix [[a, b], [b, d]].
pub fn eigen_2x2(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let trace = a + d;
    let det = a * d - b * c;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let l1 = 0.5 * (trace + disc);
    let l2 = 0.5 * (trace - disc);
    (l1, l2)
}

/// Eigenvalues of a 3×3 symmetric matrix (Cardano / analytical).
pub fn eigen_3x3_symmetric(mat: &DenseMatrix) -> DecompResult<EigenDecomposition> {
    if mat.rows != 3 || mat.cols != 3 {
        return Err(DecompError::NotSquare {
            rows: mat.rows,
            cols: mat.cols,
        });
    }
    let a = mat.get(0, 0);
    let b = mat.get(0, 1);
    let c = mat.get(0, 2);
    let d = mat.get(1, 1);
    let e = mat.get(1, 2);
    let f = mat.get(2, 2);

    let p1 = b * b + c * c + e * e;
    if p1 < 1e-30 {
        // Already diagonal
        let mut evals = vec![a, d, f];
        evals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        return Ok(EigenDecomposition {
            eigenvalues: evals,
            eigenvectors: Some(DenseMatrix::eye(3)),
            n: 3,
        });
    }

    let q = (a + d + f) / 3.0;
    let p2 = (a - q).powi(2) + (d - q).powi(2) + (f - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    // B = (1/p) * (A - q*I)
    let inv_p = 1.0 / p;
    let b00 = (a - q) * inv_p;
    let b01 = b * inv_p;
    let b02 = c * inv_p;
    let b11 = (d - q) * inv_p;
    let b12 = e * inv_p;
    let b22 = (f - q) * inv_p;

    // det(B) / 2
    let det_b = b00 * (b11 * b22 - b12 * b12)
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);
    let half_det = det_b * 0.5;

    let r = half_det.clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::FRAC_PI_3).cos();
    let eig2 = 3.0 * q - eig1 - eig3;

    let mut eigenvalues = vec![eig1, eig2, eig3];
    eigenvalues.sort_by(|x, y| x.partial_cmp(y).unwrap());

    // Compute eigenvectors by solving (A - λI)x = 0 for each eigenvalue
    let mut evecs = DenseMatrix::zeros(3, 3);
    for (col, &lam) in eigenvalues.iter().enumerate() {
        let m00 = a - lam;
        let m01 = b;
        let m02 = c;
        let m11 = d - lam;
        let m12 = e;
        let m22 = f - lam;

        // Try cross products of rows to find eigenvector
        let r0 = [m00, m01, m02];
        let r1 = [m01, m11, m12];
        let r2 = [m02, m12, m22];

        let cross01 = cross3(&r0, &r1);
        let cross02 = cross3(&r0, &r2);
        let cross12 = cross3(&r1, &r2);

        let n01 = cross01[0] * cross01[0] + cross01[1] * cross01[1] + cross01[2] * cross01[2];
        let n02 = cross02[0] * cross02[0] + cross02[1] * cross02[1] + cross02[2] * cross02[2];
        let n12 = cross12[0] * cross12[0] + cross12[1] * cross12[1] + cross12[2] * cross12[2];

        let v = if n01 >= n02 && n01 >= n12 {
            cross01
        } else if n02 >= n12 {
            cross02
        } else {
            cross12
        };

        let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if nrm > 1e-15 {
            evecs.set(0, col, v[0] / nrm);
            evecs.set(1, col, v[1] / nrm);
            evecs.set(2, col, v[2] / nrm);
        } else {
            evecs.set(col, col, 1.0);
        }
    }

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors: Some(evecs),
        n: 3,
    })
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Householder tridiagonal reduction (inline)
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce a symmetric matrix to tridiagonal form via Householder reflections.
///
/// On return, `alpha` holds the diagonal and `beta` the sub-diagonal of the
/// tridiagonal matrix.  If `q` is Some, the accumulated orthogonal
/// transformation is stored there (Q such that A = Q T Q^T).
fn symmetric_tridiag_reduce(
    a: &DenseMatrix,
    alpha: &mut [f64],
    beta: &mut [f64],
    mut q: Option<&mut DenseMatrix>,
) {
    let n = a.rows;
    let mut work = a.clone();

    for k in 0..(n.saturating_sub(2)) {
        // Extract the column below the sub-diagonal
        let mut x = vec![0.0; n - k - 1];
        for i in 0..x.len() {
            x[i] = work.get(k + 1 + i, k);
        }
        let x_norm = norm2(&x);
        if x_norm < 1e-300 {
            alpha[k] = work.get(k, k);
            beta[k] = 0.0;
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        let u0 = x[0] + sign * x_norm;
        let mut v = x.clone();
        v[0] = u0;
        let v_norm_sq = dot(&v, &v);
        if v_norm_sq < 1e-300 {
            alpha[k] = work.get(k, k);
            beta[k] = 0.0;
            continue;
        }
        let scale = 2.0 / v_norm_sq;

        // Apply the Householder from left and right: H = I - scale * v v^T
        // work <- H * work * H  (similarity transform)
        // Since the matrix is symmetric, we can be clever.
        // p = scale * work[k+1:, k+1:] * v
        let m = v.len();
        let off = k + 1;
        let mut p = vec![0.0; m];
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..m {
                s += work.get(off + i, off + j) * v[j];
            }
            p[i] = scale * s;
        }
        // K = scale/2 * (p^T v)
        let kk = 0.5 * scale * dot(&p, &v);
        // q_vec = p - K * v
        let mut q_vec = p.clone();
        for i in 0..m {
            q_vec[i] -= kk * v[i];
        }
        // work[off:, off:] -= v * q^T + q * v^T
        for i in 0..m {
            for j in 0..m {
                let val = work.get(off + i, off + j) - v[i] * q_vec[j] - q_vec[i] * v[j];
                work.set(off + i, off + j, val);
            }
        }

        alpha[k] = work.get(k, k);
        beta[k] = -sign * x_norm;

        // Accumulate Q if requested
        if let Some(ref mut q_mat) = q.as_deref_mut() {
            // Q = Q * H_k  (apply Householder on the right to Q)
            // For columns off..off+m of Q, Q[:,off:off+m] -= scale * (Q[:,off:off+m] * v) * v^T
            for r in 0..n {
                let mut s = 0.0;
                for j in 0..m {
                    s += q_mat.get(r, off + j) * v[j];
                }
                s *= scale;
                for j in 0..m {
                    let val = q_mat.get(r, off + j) - s * v[j];
                    q_mat.set(r, off + j, val);
                }
            }
        }
    }

    // Fill in last elements
    if n >= 2 {
        alpha[n - 2] = work.get(n - 2, n - 2);
        beta[n - 2] = work.get(n - 1, n - 2);
    }
    if n >= 1 {
        alpha[n - 1] = work.get(n - 1, n - 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Implicit QR on tridiagonal matrix (Wilkinson shift, Givens bulge chase)
// ═══════════════════════════════════════════════════════════════════════════

/// QR iteration on a symmetric tridiagonal matrix.
///
/// `alpha` = diagonal, `beta` = sub-diagonal (length n, beta[n-1] unused).
/// Eigenvalues are stored in `alpha` on return.
/// If `eigvecs` is Some, Givens rotations are accumulated into it.
pub fn tridiagonal_qr_implicit(
    alpha: &mut [f64],
    beta: &mut [f64],
    mut eigvecs: Option<&mut DenseMatrix>,
    max_iter: usize,
    tol: f64,
) -> DecompResult<()> {
    let n = alpha.len();
    if n <= 1 {
        return Ok(());
    }

    let mut total_iter = 0;
    let lo = 0usize;
    let mut hi = n - 1;

    while hi > lo {
        // Deflation: check if beta[hi-1] is small enough
        let off_diag = beta[hi - 1].abs();
        let diag_sum = alpha[hi - 1].abs() + alpha[hi].abs();
        if off_diag <= tol * diag_sum.max(1e-300) {
            beta[hi - 1] = 0.0;
            hi -= 1;
            // Also check for splits in the middle
            while hi > lo {
                let od = beta[hi - 1].abs();
                let ds = alpha[hi - 1].abs() + alpha[hi].abs();
                if od <= tol * ds.max(1e-300) {
                    beta[hi - 1] = 0.0;
                    hi -= 1;
                } else {
                    break;
                }
            }
            continue;
        }

        // Find the start of the unreduced block
        let mut start = hi - 1;
        while start > lo {
            let od = beta[start - 1].abs();
            let ds = alpha[start - 1].abs() + alpha[start].abs();
            if od <= tol * ds.max(1e-300) {
                beta[start - 1] = 0.0;
                break;
            }
            start -= 1;
        }

        total_iter += 1;
        if total_iter > max_iter {
            return Err(DecompError::ConvergenceFailure {
                iterations: max_iter,
                context: "tridiagonal QR iteration".to_string(),
            });
        }

        // Wilkinson shift: eigenvalue of trailing 2×2 block closer to alpha[hi]
        let d = (alpha[hi - 1] - alpha[hi]) * 0.5;
        let mu = alpha[hi] - beta[hi - 1].powi(2)
            / (d + d.signum() * (d * d + beta[hi - 1].powi(2)).sqrt()
                + if d == 0.0 { 1e-300 } else { 0.0 });

        // Implicit QR step with Givens rotations (bulge chasing)
        let mut x = alpha[start] - mu;
        let mut z = beta[start];

        for k in start..hi {
            // Compute Givens rotation to zero out z
            let (cs, sn) = givens_rotation(x, z);

            // Apply Givens rotation to tridiagonal
            if k > start {
                beta[k - 1] = cs * beta[k - 1] + sn * z;
            }

            let a_k = alpha[k];
            let a_k1 = alpha[k + 1];
            let b_k = beta[k];

            let tau1 = cs * a_k + sn * b_k;
            let tau2 = -sn * a_k + cs * b_k;
            alpha[k] = cs * tau1 + sn * (cs * b_k + sn * a_k1);
            beta[k] = -sn * tau1 + cs * (cs * b_k + sn * a_k1);
            alpha[k + 1] = -sn * tau2 + cs * a_k1;

            // Correct: use symmetric property
            // Recompute more carefully
            // G^T * T * G where G rotates rows/cols k, k+1
            let d0 = a_k;
            let d1 = a_k1;
            let e0 = b_k;

            alpha[k] = cs * cs * d0 + 2.0 * cs * sn * e0 + sn * sn * d1;
            alpha[k + 1] = sn * sn * d0 - 2.0 * cs * sn * e0 + cs * cs * d1;
            beta[k] = cs * sn * (d1 - d0) + (cs * cs - sn * sn) * e0;

            if k > start {
                beta[k - 1] = cs * x + sn * z;
            }

            // Next bulge
            if k + 1 < hi {
                let bk1 = beta[k + 1];
                z = sn * bk1;
                beta[k + 1] = cs * bk1;
                x = beta[k];
            }

            // Accumulate eigenvectors
            if let Some(ref mut vecs) = eigvecs.as_deref_mut() {
                let nn = vecs.rows;
                for r in 0..nn {
                    let v0 = vecs.get(r, k);
                    let v1 = vecs.get(r, k + 1);
                    vecs.set(r, k, cs * v0 + sn * v1);
                    vecs.set(r, k + 1, -sn * v0 + cs * v1);
                }
            }
        }
    }

    Ok(())
}

/// Compute a Givens rotation (c, s) such that [c s; -s c]^T [a; b] = [r; 0].
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-300 {
        return (1.0, 0.0);
    }
    if a.abs() < 1e-300 {
        return (0.0, b.signum());
    }
    let r = a.hypot(b);
    (a / r, b / r)
}

// ═══════════════════════════════════════════════════════════════════════════
// QR algorithm for symmetric dense matrices
// ═══════════════════════════════════════════════════════════════════════════

/// Full QR algorithm for symmetric matrices:
/// 1. Householder reduction to tridiagonal
/// 2. Implicit QR iteration with Wilkinson shifts
pub fn qr_algorithm_symmetric(
    a: &DenseMatrix,
    config: &EigenConfig,
) -> DecompResult<EigenDecomposition> {
    let n = a.rows;
    DecompError::check_square(n, a.cols)?;

    let mut alpha = vec![0.0; n];
    let mut beta = vec![0.0; n];
    let mut q = if config.compute_vectors {
        Some(DenseMatrix::eye(n))
    } else {
        None
    };

    symmetric_tridiag_reduce(a, &mut alpha, &mut beta, q.as_mut());

    tridiagonal_qr_implicit(
        &mut alpha,
        &mut beta,
        q.as_mut(),
        config.max_iter * n,
        config.tol,
    )?;

    Ok(EigenDecomposition {
        eigenvalues: alpha,
        eigenvectors: q,
        n,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry: symmetric eigen
// ═══════════════════════════════════════════════════════════════════════════

/// Main entry point for symmetric eigenvalue decomposition.
///
/// Dispatches to direct solvers for n ≤ 3, Jacobi for small n with that
/// method selected, or QR algorithm otherwise.
pub fn symmetric_eigen(
    a: &DenseMatrix,
    config: &EigenConfig,
) -> DecompResult<EigenDecomposition> {
    let n = a.rows;
    DecompError::check_square(n, a.cols)?;

    if n == 0 {
        return Ok(EigenDecomposition {
            eigenvalues: vec![],
            eigenvectors: Some(DenseMatrix::zeros(0, 0)),
            n: 0,
        });
    }

    // Verify symmetry
    if !a.is_symmetric(config.tol * 1e6) {
        // Find the worst asymmetry for the error message
        let mut max_diff = 0.0f64;
        let mut mr = 0;
        let mut mc = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = (a.get(i, j) - a.get(j, i)).abs();
                if diff > max_diff {
                    max_diff = diff;
                    mr = i;
                    mc = j;
                }
            }
        }
        return Err(DecompError::NotSymmetric {
            max_diff,
            row: mr,
            col: mc,
        });
    }

    if n == 1 {
        return Ok(EigenDecomposition {
            eigenvalues: vec![a.get(0, 0)],
            eigenvectors: if config.compute_vectors {
                Some(DenseMatrix::eye(1))
            } else {
                None
            },
            n: 1,
        });
    }

    if n == 2 {
        let (l1, l2) = eigen_2x2(a.get(0, 0), a.get(0, 1), a.get(1, 0), a.get(1, 1));
        let eigenvectors = if config.compute_vectors {
            let mut vecs = DenseMatrix::zeros(2, 2);
            let b = a.get(0, 1);
            if b.abs() < 1e-15 {
                vecs.set(0, 0, 1.0);
                vecs.set(1, 1, 1.0);
            } else {
                let v0 = [l1 - a.get(1, 1), b];
                let n0 = (v0[0] * v0[0] + v0[1] * v0[1]).sqrt();
                vecs.set(0, 0, v0[0] / n0);
                vecs.set(1, 0, v0[1] / n0);
                let v1 = [l2 - a.get(1, 1), b];
                let n1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
                vecs.set(0, 1, v1[0] / n1);
                vecs.set(1, 1, v1[1] / n1);
            }
            Some(vecs)
        } else {
            None
        };
        return Ok(EigenDecomposition {
            eigenvalues: vec![l1, l2],
            eigenvectors,
            n: 2,
        });
    }

    if n == 3 {
        return eigen_3x3_symmetric(a);
    }

    match config.method {
        EigenMethod::Jacobi => jacobi_eigen(a, config.max_iter, config.tol),
        _ => qr_algorithm_symmetric(a, config),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Jacobi eigenvalue algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Classical Jacobi eigenvalue method for symmetric matrices.
///
/// Repeatedly finds the largest off-diagonal element, applies a Jacobi
/// (Givens) rotation to zero it out, and iterates until convergence.
pub fn jacobi_eigen(
    a: &DenseMatrix,
    max_iter: usize,
    tol: f64,
) -> DecompResult<EigenDecomposition> {
    let n = a.rows;
    DecompError::check_square(n, a.cols)?;

    let mut work = a.clone();
    let mut v = DenseMatrix::eye(n);

    for iter in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = work.get(i, j).abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            let eigenvalues: Vec<f64> = (0..n).map(|i| work.get(i, i)).collect();
            return Ok(EigenDecomposition {
                eigenvalues,
                eigenvectors: Some(v),
                n,
            });
        }

        // Compute Jacobi rotation angle
        let app = work.get(p, p);
        let aqq = work.get(q, q);
        let apq = work.get(p, q);

        let theta = if (app - aqq).abs() < 1e-300 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cs = theta.cos();
        let sn = theta.sin();

        // Apply rotation to work matrix: J^T * work * J
        // Update rows/cols p and q
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let wip = work.get(i, p);
            let wiq = work.get(i, q);
            let new_ip = cs * wip + sn * wiq;
            let new_iq = -sn * wip + cs * wiq;
            work.set(i, p, new_ip);
            work.set(p, i, new_ip);
            work.set(i, q, new_iq);
            work.set(q, i, new_iq);
        }

        let new_pp = cs * cs * app + 2.0 * cs * sn * apq + sn * sn * aqq;
        let new_qq = sn * sn * app - 2.0 * cs * sn * apq + cs * cs * aqq;
        work.set(p, p, new_pp);
        work.set(q, q, new_qq);
        work.set(p, q, 0.0);
        work.set(q, p, 0.0);

        // Update eigenvectors
        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.set(i, p, cs * vip + sn * viq);
            v.set(i, q, -sn * vip + cs * viq);
        }

        let _ = iter;
    }

    Err(DecompError::ConvergenceFailure {
        iterations: max_iter,
        context: "Jacobi eigenvalue iteration".to_string(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-k and Bottom-k
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the top k eigenvalues/eigenvectors (largest by value) of a
/// symmetric dense matrix.
pub fn symmetric_eigen_top_k(
    a: &DenseMatrix,
    k: usize,
    config: &EigenConfig,
) -> DecompResult<EigenDecomposition> {
    let n = a.rows;
    if k > n {
        return Err(DecompError::TooManyEigenvalues {
            requested: k,
            size: n,
        });
    }

    let mut full = symmetric_eigen(a, config)?;
    full.sort_ascending();

    // Take top k (largest eigenvalues = last k in ascending order)
    let start = n - k;
    let eigenvalues = full.eigenvalues[start..].to_vec();
    let eigenvectors = if let Some(ref vecs) = full.eigenvectors {
        let mut ev = DenseMatrix::zeros(n, k);
        for r in 0..n {
            for c in 0..k {
                ev.set(r, c, vecs.get(r, start + c));
            }
        }
        Some(ev)
    } else {
        None
    };

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
        n,
    })
}

/// Compute the bottom k eigenvalues/eigenvectors (smallest by value) of a
/// symmetric dense matrix.
pub fn symmetric_eigen_bottom_k(
    a: &DenseMatrix,
    k: usize,
    config: &EigenConfig,
) -> DecompResult<EigenDecomposition> {
    let n = a.rows;
    if k > n {
        return Err(DecompError::TooManyEigenvalues {
            requested: k,
            size: n,
        });
    }

    let mut full = symmetric_eigen(a, config)?;
    full.sort_ascending();

    let eigenvalues = full.eigenvalues[..k].to_vec();
    let eigenvectors = if let Some(ref vecs) = full.eigenvectors {
        let mut ev = DenseMatrix::zeros(n, k);
        for r in 0..n {
            for c in 0..k {
                ev.set(r, c, vecs.get(r, c));
            }
        }
        Some(ev)
    } else {
        None
    };

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
        n,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Hessenberg reduction (for non-symmetric matrices)
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce a general square matrix to upper Hessenberg form in-place via
/// Householder similarity transforms.
///
/// Returns the Householder reflectors (v, tau) for each step.
pub fn hessenberg_reduce(a: &mut DenseMatrix) -> Vec<(Vec<f64>, f64)> {
    let n = a.rows;
    let mut reflectors = Vec::new();

    for k in 0..n.saturating_sub(2) {
        let m = n - k - 1;
        let mut x = vec![0.0; m];
        for i in 0..m {
            x[i] = a.get(k + 1 + i, k);
        }
        let x_norm = norm2(&x);
        if x_norm < 1e-300 {
            reflectors.push((vec![0.0; m], 0.0));
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * x_norm;
        let v_norm = norm2(&x);
        if v_norm < 1e-300 {
            reflectors.push((vec![0.0; m], 0.0));
            continue;
        }
        for i in 0..m {
            x[i] /= v_norm;
        }
        let tau = 2.0;
        let v = x;

        // Apply from the left: A <- (I - tau * v * v^T) * A
        // Affects rows k+1..n, all columns
        let off = k + 1;
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                s += v[i] * a.get(off + i, j);
            }
            s *= tau;
            for i in 0..m {
                let val = a.get(off + i, j) - s * v[i];
                a.set(off + i, j, val);
            }
        }

        // Apply from the right: A <- A * (I - tau * v * v^T)
        // Affects all rows, columns k+1..n
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..m {
                s += a.get(i, off + j) * v[j];
            }
            s *= tau;
            for j in 0..m {
                let val = a.get(i, off + j) - s * v[j];
                a.set(i, off + j, val);
            }
        }

        reflectors.push((v, tau));
    }

    reflectors
}

/// Reconstruct Q from Hessenberg reflectors.
fn hessenberg_q(n: usize, reflectors: &[(Vec<f64>, f64)]) -> DenseMatrix {
    let mut q = DenseMatrix::eye(n);
    for k in (0..reflectors.len()).rev() {
        let (ref v, tau) = reflectors[k];
        if tau == 0.0 {
            continue;
        }
        let off = k + 1;
        let m = v.len();
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                s += v[i] * q.get(off + i, j);
            }
            s *= tau;
            for i in 0..m {
                let val = q.get(off + i, j) - s * v[i];
                q.set(off + i, j, val);
            }
        }
    }
    q
}

// ═══════════════════════════════════════════════════════════════════════════
// Francis double-shift QR step
// ═══════════════════════════════════════════════════════════════════════════

/// Francis double-shift implicit QR step on an upper Hessenberg matrix H.
///
/// Operates on the sub-matrix H[start..=end, start..=end].  Optionally
/// accumulates transformations in Q.
pub fn francis_qr_step(
    h: &mut DenseMatrix,
    start: usize,
    end: usize,
    mut q: Option<&mut DenseMatrix>,
) {
    let n = h.rows;
    if end <= start + 1 {
        return;
    }

    // Compute the first column of the double-shift polynomial
    // (H - s1 I)(H - s2 I) e_1, where s1,s2 are eigenvalues of trailing 2x2
    let hee = h.get(end, end);
    let hem = h.get(end - 1, end - 1);
    let hme = h.get(end, end - 1);
    let hem_off = h.get(end - 1, end);

    let s = hem + hee; // trace
    let t = hem * hee - hem_off * hme; // det

    let h00 = h.get(start, start);
    let h10 = h.get(start + 1, start);
    let h01 = h.get(start, start + 1);
    let h11 = h.get(start + 1, start + 1);

    let mut x = h00 * h00 + h01 * h10 - s * h00 + t;
    let mut y = h10 * (h00 + h11 - s);
    let mut z = h10 * h.get(start + 2, start + 1);

    for k in start..end - 1 {
        // Compute 3x1 (or 2x1 at last step) Householder reflector
        let r;
        let v0;
        let v1;
        let v2;

        if k == end - 2 || k == start {
            // 3x1 Householder
            let nrm = (x * x + y * y + z * z).sqrt();
            if nrm < 1e-300 {
                break;
            }
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            v0 = x + sign * nrm;
            v1 = y;
            v2 = z;
            r = 2.0 / (v0 * v0 + v1 * v1 + v2 * v2);
        } else {
            let nrm = (x * x + y * y + z * z).sqrt();
            if nrm < 1e-300 {
                break;
            }
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            v0 = x + sign * nrm;
            v1 = y;
            v2 = z;
            r = 2.0 / (v0 * v0 + v1 * v1 + v2 * v2);
        }

        let off = if k > start { k - 1 } else { k };
        let _ = off;

        // Apply reflector from the left to H
        let i_start = k;
        let row_count = if k + 2 < end { 3 } else { (end - k + 1).min(3) };

        if row_count == 3 {
            // Left multiply: rows k, k+1, k+2
            let col_lo = if k > start { k - 1 } else { k };
            for j in col_lo..n {
                let s_val = r * (v0 * h.get(k, j) + v1 * h.get(k + 1, j) + v2 * h.get(k + 2, j));
                h.set(k, j, h.get(k, j) - s_val * v0);
                h.set(k + 1, j, h.get(k + 1, j) - s_val * v1);
                h.set(k + 2, j, h.get(k + 2, j) - s_val * v2);
            }

            // Right multiply: cols k, k+1, k+2
            let row_hi = (k + 4).min(n);
            for i in 0..row_hi {
                let s_val = r * (h.get(i, k) * v0 + h.get(i, k + 1) * v1 + h.get(i, k + 2) * v2);
                h.set(i, k, h.get(i, k) - s_val * v0);
                h.set(i, k + 1, h.get(i, k + 1) - s_val * v1);
                h.set(i, k + 2, h.get(i, k + 2) - s_val * v2);
            }

            // Q accumulation
            if let Some(ref mut qq) = q.as_deref_mut() {
                for i in 0..n {
                    let s_val = r * (qq.get(i, k) * v0 + qq.get(i, k + 1) * v1 + qq.get(i, k + 2) * v2);
                    qq.set(i, k, qq.get(i, k) - s_val * v0);
                    qq.set(i, k + 1, qq.get(i, k + 1) - s_val * v1);
                    qq.set(i, k + 2, qq.get(i, k + 2) - s_val * v2);
                }
            }
        }

        let _ = i_start;

        // Set up for next iteration
        if k + 3 <= end {
            x = h.get(k + 1, k);
            y = h.get(k + 2, k);
            z = if k + 3 <= end { h.get((k + 3).min(end), k) } else { 0.0 };
        } else {
            break;
        }
    }

    // Final 2x2 Givens to clean up
    let k = end - 1;
    if k >= start {
        let x_val = h.get(k, k - 1.max(start));
        let y_val = h.get(end, k - 1.max(start));
        let (cs, sn) = givens_rotation(x_val, y_val);
        // Apply from left
        for j in (k.saturating_sub(1))..n {
            let t1 = h.get(k, j);
            let t2 = h.get(end, j);
            h.set(k, j, cs * t1 + sn * t2);
            h.set(end, j, -sn * t1 + cs * t2);
        }
        // Apply from right
        for i in 0..(end + 1).min(n) {
            let t1 = h.get(i, k);
            let t2 = h.get(i, end);
            h.set(i, k, cs * t1 + sn * t2);
            h.set(i, end, -sn * t1 + cs * t2);
        }
        if let Some(ref mut qq) = q.as_deref_mut() {
            for i in 0..n {
                let t1 = qq.get(i, k);
                let t2 = qq.get(i, end);
                qq.set(i, k, cs * t1 + sn * t2);
                qq.set(i, end, -sn * t1 + cs * t2);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-symmetric eigenvalue decomposition
// ═══════════════════════════════════════════════════════════════════════════

/// Eigenvalue decomposition for general (non-symmetric) square matrices.
///
/// 1. Reduce to upper Hessenberg form.
/// 2. Apply Francis double-shift QR iterations.
/// 3. Extract eigenvalues from the quasi-upper-triangular Schur form.
pub fn nonsymmetric_eigen(
    a: &DenseMatrix,
    config: &EigenConfig,
) -> DecompResult<NonSymmetricEigen> {
    let n = a.rows;
    DecompError::check_square(n, a.cols)?;

    if n == 0 {
        return Ok(NonSymmetricEigen {
            real_eigenvalues: vec![],
            imag_eigenvalues: vec![],
            schur_form: None,
            schur_vectors: None,
        });
    }

    if n == 1 {
        return Ok(NonSymmetricEigen {
            real_eigenvalues: vec![a.get(0, 0)],
            imag_eigenvalues: vec![0.0],
            schur_form: Some(a.clone()),
            schur_vectors: Some(DenseMatrix::eye(1)),
        });
    }

    let mut h = a.clone();
    let reflectors = hessenberg_reduce(&mut h);
    let mut q = if config.compute_vectors {
        Some(hessenberg_q(n, &reflectors))
    } else {
        None
    };

    // QR iteration on the Hessenberg matrix
    let mut hi = n - 1;
    let mut total_iter = 0;
    let max_qr_iter = config.max_iter * n;

    while hi > 0 {
        // Check for convergence of sub-diagonal element
        let test = h.get(hi - 1, hi - 1).abs() + h.get(hi, hi).abs();
        let threshold = config.tol * test.max(1e-300);

        if h.get(hi, hi - 1).abs() <= threshold {
            h.set(hi, hi - 1, 0.0);
            hi -= 1;
            continue;
        }

        // Check for 2x2 block convergence
        if hi >= 2 {
            let test2 = h.get(hi - 2, hi - 2).abs() + h.get(hi - 1, hi - 1).abs();
            let thresh2 = config.tol * test2.max(1e-300);
            if h.get(hi - 1, hi - 2).abs() <= thresh2 {
                h.set(hi - 1, hi - 2, 0.0);
                // 2x2 block at (hi-1, hi-1)..(hi, hi)
                hi = hi.saturating_sub(2);
                continue;
            }
        }

        // Find the start of the active block
        let mut lo = hi - 1;
        while lo > 0 {
            let t_lo = h.get(lo - 1, lo - 1).abs() + h.get(lo, lo).abs();
            let thr = config.tol * t_lo.max(1e-300);
            if h.get(lo, lo - 1).abs() <= thr {
                h.set(lo, lo - 1, 0.0);
                break;
            }
            lo -= 1;
        }

        total_iter += 1;
        if total_iter > max_qr_iter {
            return Err(DecompError::ConvergenceFailure {
                iterations: max_qr_iter,
                context: "non-symmetric QR iteration".to_string(),
            });
        }

        // Apply Francis step
        if hi > lo + 1 {
            francis_qr_step(&mut h, lo, hi, q.as_mut());
        } else {
            // Single QR step for 2x2 block
            let (cs, sn) = givens_rotation(
                h.get(lo, lo) - h.get(hi, hi),
                h.get(hi, lo),
            );
            // Apply from left
            for j in lo..n {
                let t1 = h.get(lo, j);
                let t2 = h.get(hi, j);
                h.set(lo, j, cs * t1 + sn * t2);
                h.set(hi, j, -sn * t1 + cs * t2);
            }
            // Apply from right
            for i in 0..=hi {
                let t1 = h.get(i, lo);
                let t2 = h.get(i, hi);
                h.set(i, lo, cs * t1 + sn * t2);
                h.set(i, hi, -sn * t1 + cs * t2);
            }
            if let Some(ref mut qq) = q {
                for i in 0..n {
                    let t1 = qq.get(i, lo);
                    let t2 = qq.get(i, hi);
                    qq.set(i, lo, cs * t1 + sn * t2);
                    qq.set(i, hi, -sn * t1 + cs * t2);
                }
            }
        }
    }

    // Extract eigenvalues from the quasi-upper-triangular Schur form
    let mut real_evals = Vec::with_capacity(n);
    let mut imag_evals = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && h.get(i + 1, i).abs() > config.tol * (h.get(i, i).abs() + h.get(i + 1, i + 1).abs()).max(1e-300) {
            // 2×2 block → complex conjugate pair
            let a11 = h.get(i, i);
            let a12 = h.get(i, i + 1);
            let a21 = h.get(i + 1, i);
            let a22 = h.get(i + 1, i + 1);
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                let re = tr * 0.5;
                let im = (-disc).sqrt() * 0.5;
                real_evals.push(re);
                imag_evals.push(im);
                real_evals.push(re);
                imag_evals.push(-im);
            } else {
                let sq = disc.sqrt();
                real_evals.push(0.5 * (tr + sq));
                imag_evals.push(0.0);
                real_evals.push(0.5 * (tr - sq));
                imag_evals.push(0.0);
            }
            i += 2;
        } else {
            real_evals.push(h.get(i, i));
            imag_evals.push(0.0);
            i += 1;
        }
    }

    Ok(NonSymmetricEigen {
        real_eigenvalues: real_evals,
        imag_eigenvalues: imag_evals,
        schur_form: Some(h),
        schur_vectors: q,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Arnoldi iteration
// ═══════════════════════════════════════════════════════════════════════════

/// Arnoldi iteration building a Krylov basis V (n×(k+1)) and upper Hessenberg
/// H ((k+1)×k).
///
/// `matvec` is a closure that computes the matrix-vector product.
pub fn arnoldi_iteration<F>(
    matvec: F,
    n: usize,
    k: usize,
    _max_iter: usize,
    tol: f64,
) -> DecompResult<(DenseMatrix, DenseMatrix)>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if k == 0 || n == 0 {
        return Err(DecompError::InvalidParameter {
            name: "k".to_string(),
            value: k.to_string(),
            reason: "k must be positive".to_string(),
        });
    }
    let kk = k.min(n);

    let mut v_mat = DenseMatrix::zeros(n, kk + 1);
    let mut h_mat = DenseMatrix::zeros(kk + 1, kk);

    // Initial vector (random)
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut v0: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
    normalize(&mut v0);
    for i in 0..n {
        v_mat.set(i, 0, v0[i]);
    }

    for j in 0..kk {
        let vj: Vec<f64> = (0..n).map(|i| v_mat.get(i, j)).collect();
        let mut w = matvec(&vj);

        // Modified Gram-Schmidt
        for i in 0..=j {
            let vi: Vec<f64> = (0..n).map(|r| v_mat.get(r, i)).collect();
            let h_ij = dot(&w, &vi);
            h_mat.set(i, j, h_ij);
            axpy(-h_ij, &vi, &mut w);
        }

        // Re-orthogonalize
        for i in 0..=j {
            let vi: Vec<f64> = (0..n).map(|r| v_mat.get(r, i)).collect();
            let corr = dot(&w, &vi);
            let old = h_mat.get(i, j);
            h_mat.set(i, j, old + corr);
            axpy(-corr, &vi, &mut w);
        }

        let h_next = norm2(&w);
        h_mat.set(j + 1, j, h_next);

        if h_next < tol {
            // Lucky breakdown - Krylov subspace is invariant
            let v_out = v_mat.submatrix(0, n, 0, j + 1);
            let h_out = h_mat.submatrix(0, j + 1, 0, j);
            return Ok((v_out, h_out));
        }

        if j + 1 < kk + 1 {
            let inv = 1.0 / h_next;
            for i in 0..n {
                v_mat.set(i, j + 1, w[i] * inv);
            }
        }
    }

    Ok((v_mat, h_mat))
}

// ═══════════════════════════════════════════════════════════════════════════
// Inline Lanczos for sparse symmetric matrices
// ═══════════════════════════════════════════════════════════════════════════

/// Which eigenvalues to compute for sparse matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhichEigen {
    Largest,
    Smallest,
    LargestMagnitude,
    SmallestMagnitude,
}

/// Sparse symmetric eigenvalue decomposition using Lanczos iteration with
/// full reorthogonalization.
pub fn sparse_symmetric_eigen(
    a: &CsrMatrix,
    k: usize,
    which: WhichEigen,
    config: &EigenConfig,
) -> DecompResult<EigenDecomposition> {
    let n = a.rows;
    if !a.is_square() {
        return Err(DecompError::NotSquare {
            rows: a.rows,
            cols: a.cols,
        });
    }
    if k > n {
        return Err(DecompError::TooManyEigenvalues {
            requested: k,
            size: n,
        });
    }
    if n == 0 {
        return Ok(EigenDecomposition {
            eigenvalues: vec![],
            eigenvectors: Some(DenseMatrix::zeros(0, 0)),
            n: 0,
        });
    }

    // For small matrices, convert to dense and solve directly
    if n <= 64 {
        let dense = a.to_dense();
        let mut full = symmetric_eigen(&dense, config)?;
        return select_eigenvalues(&mut full, k, which);
    }

    // Lanczos iteration with full reorthogonalization
    let m = (2 * k + 10).min(n); // Krylov subspace dimension
    let mut alpha_vec = Vec::with_capacity(m);
    let mut beta_vec = Vec::with_capacity(m);

    // V stores Lanczos vectors column by column
    let mut v_store: Vec<Vec<f64>> = Vec::with_capacity(m + 1);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
    normalize(&mut v);
    v_store.push(v.clone());

    let mut v_prev = vec![0.0; n];
    let mut beta = 0.0f64;

    for j in 0..m {
        // w = A * v - beta * v_prev
        let mut w = a.mul_vec(&v)?;
        if j > 0 {
            axpy(-beta, &v_prev, &mut w);
        }

        let alpha_j = dot(&w, &v);
        alpha_vec.push(alpha_j);
        axpy(-alpha_j, &v, &mut w);

        // Full reorthogonalization
        for vi in &v_store {
            let coeff = dot(&w, vi);
            axpy(-coeff, vi, &mut w);
        }
        // Second pass
        for vi in &v_store {
            let coeff = dot(&w, vi);
            axpy(-coeff, vi, &mut w);
        }

        beta = norm2(&w);
        beta_vec.push(beta);

        if beta < config.tol {
            break;
        }

        copy_vec(&mut v_prev, &v);
        scale_vec(&mut w, 1.0 / beta);
        copy_vec(&mut v, &w);
        v_store.push(v.clone());
    }

    // Solve eigenvalue problem on the tridiagonal matrix T
    let m_actual = alpha_vec.len();
    let mut t_alpha = alpha_vec;
    let mut t_beta = beta_vec;
    t_beta.resize(m_actual, 0.0);

    let mut t_vecs = if config.compute_vectors {
        Some(DenseMatrix::eye(m_actual))
    } else {
        None
    };

    tridiagonal_qr_implicit(
        &mut t_alpha,
        &mut t_beta,
        t_vecs.as_mut(),
        config.max_iter * m_actual,
        config.tol,
    )?;

    // Build EigenDecomposition for the tridiagonal problem
    let eigenvectors = if config.compute_vectors && t_vecs.is_some() {
        let s = t_vecs.as_ref().unwrap();
        // Transform back: eigenvectors = V * S
        let num_vecs = v_store.len().min(s.rows);
        let mut ev = DenseMatrix::zeros(n, m_actual);
        for col in 0..m_actual {
            for j in 0..num_vecs {
                let coeff = s.get(j, col);
                if coeff.abs() > 1e-15 {
                    for i in 0..n {
                        let old = ev.get(i, col);
                        ev.set(i, col, old + coeff * v_store[j][i]);
                    }
                }
            }
        }
        Some(ev)
    } else {
        None
    };

    let mut decomp = EigenDecomposition {
        eigenvalues: t_alpha,
        eigenvectors,
        n,
    };

    select_eigenvalues(&mut decomp, k, which)
}

/// Select k eigenvalues from a full decomposition based on the `which` criterion.
fn select_eigenvalues(
    decomp: &mut EigenDecomposition,
    k: usize,
    which: WhichEigen,
) -> DecompResult<EigenDecomposition> {
    let n_total = decomp.eigenvalues.len();
    let k = k.min(n_total);

    let mut indices: Vec<usize> = (0..n_total).collect();
    match which {
        WhichEigen::Largest => {
            indices.sort_by(|&a, &b| {
                decomp.eigenvalues[b]
                    .partial_cmp(&decomp.eigenvalues[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigen::Smallest => {
            indices.sort_by(|&a, &b| {
                decomp.eigenvalues[a]
                    .partial_cmp(&decomp.eigenvalues[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigen::LargestMagnitude => {
            indices.sort_by(|&a, &b| {
                decomp.eigenvalues[b]
                    .abs()
                    .partial_cmp(&decomp.eigenvalues[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigen::SmallestMagnitude => {
            indices.sort_by(|&a, &b| {
                decomp.eigenvalues[a]
                    .abs()
                    .partial_cmp(&decomp.eigenvalues[b].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    let sel: Vec<usize> = indices[..k].to_vec();
    let eigenvalues: Vec<f64> = sel.iter().map(|&i| decomp.eigenvalues[i]).collect();
    let mat_n = decomp.n;

    let eigenvectors = if let Some(ref vecs) = decomp.eigenvectors {
        let mut ev = DenseMatrix::zeros(mat_n, k);
        for (new_col, &old_col) in sel.iter().enumerate() {
            if old_col < vecs.cols {
                for r in 0..mat_n.min(vecs.rows) {
                    ev.set(r, new_col, vecs.get(r, old_col));
                }
            }
        }
        Some(ev)
    } else {
        None
    };

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
        n: mat_n,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Shift-invert eigenvalue computation
// ═══════════════════════════════════════════════════════════════════════════

/// Shift-invert eigenvalue computation: finds eigenvalues of A closest to σ.
///
/// Uses LU factorization of (A − σI) and inverse iteration via Lanczos on
/// (A − σI)^{-1}.
pub fn shift_invert_eigen(
    a: &CsrMatrix,
    sigma: f64,
    k: usize,
    config: &EigenConfig,
) -> DecompResult<EigenDecomposition> {
    let n = a.rows;
    if !a.is_square() {
        return Err(DecompError::NotSquare {
            rows: a.rows,
            cols: a.cols,
        });
    }

    // Build (A - sigma * I) as dense and LU factorize
    let mut shifted = a.to_dense();
    for i in 0..n {
        shifted.set(i, i, shifted.get(i, i) - sigma);
    }

    // Simple LU factorization with partial pivoting (inline)
    let mut lu_mat = shifted.clone();
    let mut piv: Vec<usize> = (0..n).collect();

    for col in 0..n {
        // Find pivot
        let mut max_val = lu_mat.get(col, col).abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = lu_mat.get(row, col).abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(DecompError::SingularMatrix {
                context: format!("shift-invert: (A - {}I) is singular", sigma),
            });
        }
        if max_row != col {
            lu_mat.swap_rows(col, max_row);
            piv.swap(col, max_row);
        }
        let pivot = lu_mat.get(col, col);
        for row in (col + 1)..n {
            let factor = lu_mat.get(row, col) / pivot;
            lu_mat.set(row, col, factor);
            for c in (col + 1)..n {
                let val = lu_mat.get(row, c) - factor * lu_mat.get(col, c);
                lu_mat.set(row, c, val);
            }
        }
    }

    // LU solve function
    let lu_solve = |b: &[f64]| -> Vec<f64> {
        let mut x = vec![0.0; n];
        // Apply permutation
        for i in 0..n {
            x[i] = b[piv[i]];
        }
        // Forward substitution (L)
        for i in 1..n {
            for j in 0..i {
                x[i] -= lu_mat.get(i, j) * x[j];
            }
        }
        // Backward substitution (U)
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= lu_mat.get(i, j) * x[j];
            }
            x[i] /= lu_mat.get(i, i);
        }
        x
    };

    // Lanczos on the operator (A - sigma I)^{-1}
    let m = (2 * k + 10).min(n);
    let mut alpha_vec = Vec::with_capacity(m);
    let mut beta_vec = Vec::with_capacity(m);
    let mut v_store: Vec<Vec<f64>> = Vec::with_capacity(m + 1);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
    normalize(&mut v);
    v_store.push(v.clone());

    let mut v_prev = vec![0.0; n];
    let mut beta_val = 0.0f64;

    for j in 0..m {
        let mut w = lu_solve(&v);
        if j > 0 {
            axpy(-beta_val, &v_prev, &mut w);
        }

        let alpha_j = dot(&w, &v);
        alpha_vec.push(alpha_j);
        axpy(-alpha_j, &v, &mut w);

        for vi in &v_store {
            let coeff = dot(&w, vi);
            axpy(-coeff, vi, &mut w);
        }

        beta_val = norm2(&w);
        beta_vec.push(beta_val);

        if beta_val < config.tol {
            break;
        }

        copy_vec(&mut v_prev, &v);
        scale_vec(&mut w, 1.0 / beta_val);
        copy_vec(&mut v, &w);
        v_store.push(v.clone());
    }

    let m_actual = alpha_vec.len();
    let mut t_alpha = alpha_vec;
    let mut t_beta = beta_vec;
    t_beta.resize(m_actual, 0.0);

    let mut t_vecs = if config.compute_vectors {
        Some(DenseMatrix::eye(m_actual))
    } else {
        None
    };

    tridiagonal_qr_implicit(
        &mut t_alpha,
        &mut t_beta,
        t_vecs.as_mut(),
        config.max_iter * m_actual,
        config.tol,
    )?;

    // The eigenvalues of (A - sigma I)^{-1} are 1/(lambda - sigma).
    // So lambda = sigma + 1/theta.
    let eigenvalues: Vec<f64> = t_alpha
        .iter()
        .map(|&theta| {
            if theta.abs() < 1e-300 {
                f64::INFINITY
            } else {
                sigma + 1.0 / theta
            }
        })
        .collect();

    // Sort by distance from sigma and take the closest k
    let mut idx: Vec<usize> = (0..eigenvalues.len()).collect();
    idx.sort_by(|&a, &b| {
        (eigenvalues[a] - sigma)
            .abs()
            .partial_cmp(&(eigenvalues[b] - sigma).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let k_actual = k.min(idx.len());
    let sel: Vec<usize> = idx[..k_actual].to_vec();
    let sel_eigenvalues: Vec<f64> = sel.iter().map(|&i| eigenvalues[i]).collect();

    let eigenvectors = if config.compute_vectors && t_vecs.is_some() {
        let s = t_vecs.as_ref().unwrap();
        let num_vecs = v_store.len().min(s.rows);
        let mut ev = DenseMatrix::zeros(n, k_actual);
        for (new_col, &old_col) in sel.iter().enumerate() {
            if old_col < s.cols {
                for j in 0..num_vecs {
                    let coeff = s.get(j, old_col);
                    if coeff.abs() > 1e-15 {
                        for i in 0..n {
                            let old = ev.get(i, new_col);
                            ev.set(i, new_col, old + coeff * v_store[j][i]);
                        }
                    }
                }
                // Normalize
                let col_vec: Vec<f64> = (0..n).map(|i| ev.get(i, new_col)).collect();
                let nrm = norm2(&col_vec);
                if nrm > 1e-15 {
                    for i in 0..n {
                        ev.set(i, new_col, ev.get(i, new_col) / nrm);
                    }
                }
            }
        }
        Some(ev)
    } else {
        None
    };

    Ok(EigenDecomposition {
        eigenvalues: sel_eigenvalues,
        eigenvectors,
        n,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── Direct 2×2 ─────────────────────────────────────────────────────
    #[test]
    fn test_eigen_2x2() {
        let (l1, l2) = eigen_2x2(2.0, 1.0, 1.0, 3.0);
        let mut evals = vec![l1, l2];
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Eigenvalues of [[2,1],[1,3]] are (5 ± sqrt(5))/2
        let expected_small = (5.0 - 5.0f64.sqrt()) / 2.0;
        let expected_big = (5.0 + 5.0f64.sqrt()) / 2.0;
        assert!(approx_eq(evals[0], expected_small, 1e-10));
        assert!(approx_eq(evals[1], expected_big, 1e-10));
    }

    // ── Direct 3×3 symmetric ───────────────────────────────────────────
    #[test]
    fn test_eigen_3x3_symmetric() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ]);
        let decomp = eigen_3x3_symmetric(&a).unwrap();
        let mut evals = decomp.eigenvalues.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Known eigenvalues: 2 - sqrt(2), 2, 2 + sqrt(2)
        assert!(approx_eq(evals[0], 2.0 - 2.0f64.sqrt(), 1e-8));
        assert!(approx_eq(evals[1], 2.0, 1e-8));
        assert!(approx_eq(evals[2], 2.0 + 2.0f64.sqrt(), 1e-8));
    }

    // ── Diagonal matrix ────────────────────────────────────────────────
    #[test]
    fn test_diagonal_matrix() {
        let a = DenseMatrix::from_diag(&[5.0, 3.0, 1.0, 4.0]);
        let config = EigenConfig::default();
        let mut decomp = symmetric_eigen(&a, &config).unwrap();
        decomp.sort_ascending();
        assert!(approx_eq(decomp.eigenvalues[0], 1.0, 1e-10));
        assert!(approx_eq(decomp.eigenvalues[1], 3.0, 1e-10));
        assert!(approx_eq(decomp.eigenvalues[2], 4.0, 1e-10));
        assert!(approx_eq(decomp.eigenvalues[3], 5.0, 1e-10));
    }

    // ── Identity matrix ────────────────────────────────────────────────
    #[test]
    fn test_identity_eigenvalues() {
        let a = DenseMatrix::eye(5);
        let config = EigenConfig::default();
        let decomp = symmetric_eigen(&a, &config).unwrap();
        for &ev in &decomp.eigenvalues {
            assert!(approx_eq(ev, 1.0, 1e-10));
        }
    }

    // ── Known 3×3 SPD via QR ───────────────────────────────────────────
    #[test]
    fn test_known_3x3_spd_qr() {
        // SPD: A = [[4,2,0],[2,5,1],[0,1,3]]
        let a = DenseMatrix::from_row_major(3, 3, vec![
            4.0, 2.0, 0.0,
            2.0, 5.0, 1.0,
            0.0, 1.0, 3.0,
        ]);
        let config = EigenConfig::new().with_method(EigenMethod::QrAlgorithm);
        let mut decomp = symmetric_eigen(&a, &config).unwrap();
        decomp.sort_ascending();
        // All eigenvalues should be positive (SPD)
        for &ev in &decomp.eigenvalues {
            assert!(ev > 0.0, "SPD matrix should have positive eigenvalues, got {}", ev);
        }
        // Verify A*v ≈ λ*v
        if let Some(ref vecs) = decomp.eigenvectors {
            for (col, &lam) in decomp.eigenvalues.iter().enumerate() {
                let v: Vec<f64> = (0..3).map(|i| vecs.get(i, col)).collect();
                let av = a.mul_vec(&v).unwrap();
                for i in 0..3 {
                    assert!(
                        approx_eq(av[i], lam * v[i], 1e-8),
                        "Eigenvector verification failed"
                    );
                }
            }
        }
    }

    // ── QR convergence ─────────────────────────────────────────────────
    #[test]
    fn test_qr_convergence() {
        let a = DenseMatrix::from_row_major(4, 4, vec![
            4.0, 1.0, 0.0, 0.0,
            1.0, 3.0, 1.0, 0.0,
            0.0, 1.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ]);
        let config = EigenConfig::new().with_method(EigenMethod::QrAlgorithm);
        let decomp = qr_algorithm_symmetric(&a, &config);
        assert!(decomp.is_ok(), "QR algorithm should converge");
        let d = decomp.unwrap();
        assert_eq!(d.eigenvalues.len(), 4);
        // Sum of eigenvalues = trace
        let sum: f64 = d.eigenvalues.iter().sum();
        assert!(approx_eq(sum, a.trace(), 1e-8));
    }

    // ── Top-k ──────────────────────────────────────────────────────────
    #[test]
    fn test_top_k() {
        let a = DenseMatrix::from_diag(&[1.0, 5.0, 3.0, 7.0, 2.0]);
        let config = EigenConfig::default();
        let decomp = symmetric_eigen_top_k(&a, 2, &config).unwrap();
        assert_eq!(decomp.eigenvalues.len(), 2);
        let mut evals = decomp.eigenvalues.clone();
        evals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(approx_eq(evals[0], 7.0, 1e-10));
        assert!(approx_eq(evals[1], 5.0, 1e-10));
    }

    // ── Bottom-k ───────────────────────────────────────────────────────
    #[test]
    fn test_bottom_k() {
        let a = DenseMatrix::from_diag(&[1.0, 5.0, 3.0, 7.0, 2.0]);
        let config = EigenConfig::default();
        let decomp = symmetric_eigen_bottom_k(&a, 2, &config).unwrap();
        assert_eq!(decomp.eigenvalues.len(), 2);
        let mut evals = decomp.eigenvalues.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(evals[0], 1.0, 1e-10));
        assert!(approx_eq(evals[1], 2.0, 1e-10));
    }

    // ── Sparse symmetric ───────────────────────────────────────────────
    #[test]
    fn test_sparse_symmetric() {
        // Construct a 6×6 symmetric tridiagonal sparse matrix
        let mut triplets = Vec::new();
        for i in 0..6 {
            triplets.push((i, i, 2.0));
            if i + 1 < 6 {
                triplets.push((i, i + 1, -1.0));
                triplets.push((i + 1, i, -1.0));
            }
        }
        let a = CsrMatrix::from_triplets(6, 6, &triplets);
        let config = EigenConfig::default();
        let decomp = sparse_symmetric_eigen(&a, 3, WhichEigen::Largest, &config).unwrap();
        assert_eq!(decomp.eigenvalues.len(), 3);
        // All eigenvalues of this matrix are positive (it's SPD)
        for &ev in &decomp.eigenvalues {
            assert!(ev > -1e-8, "Expected positive eigenvalue, got {}", ev);
        }
    }

    // ── Non-symmetric with complex pairs ───────────────────────────────
    #[test]
    fn test_nonsymmetric_complex_pairs() {
        // [[0, -1], [1, 0]] has eigenvalues ±i
        let a = DenseMatrix::from_row_major(2, 2, vec![
            0.0, -1.0,
            1.0, 0.0,
        ]);
        let config = EigenConfig::default();
        let result = nonsymmetric_eigen(&a, &config).unwrap();
        assert_eq!(result.real_eigenvalues.len(), 2);
        // Real parts should be ~0
        for &re in &result.real_eigenvalues {
            assert!(re.abs() < 1e-8, "Real part should be ~0, got {}", re);
        }
        // Imaginary parts should be ±1
        let mut imag = result.imag_eigenvalues.clone();
        imag.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(imag[0], -1.0, 1e-8));
        assert!(approx_eq(imag[1], 1.0, 1e-8));
    }

    // ── Arnoldi iteration ──────────────────────────────────────────────
    #[test]
    fn test_arnoldi_iteration() {
        let a = DenseMatrix::from_row_major(4, 4, vec![
            2.0, 1.0, 0.0, 0.0,
            1.0, 3.0, 1.0, 0.0,
            0.0, 1.0, 4.0, 1.0,
            0.0, 0.0, 1.0, 5.0,
        ]);
        let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap() };
        let (v, h) = arnoldi_iteration(matvec, 4, 3, 100, 1e-12).unwrap();
        // V should have orthonormal columns
        for i in 0..v.cols.min(3) {
            let col_i: Vec<f64> = (0..v.rows).map(|r| v.get(r, i)).collect();
            let n = norm2(&col_i);
            assert!(approx_eq(n, 1.0, 1e-10), "Column {} norm = {}", i, n);
            for j in 0..i {
                let col_j: Vec<f64> = (0..v.rows).map(|r| v.get(r, j)).collect();
                let d = dot(&col_i, &col_j);
                assert!(d.abs() < 1e-10, "Columns {} and {} not orthogonal: {}", i, j, d);
            }
        }
        // H should be upper Hessenberg (below sub-diagonal should be 0)
        for i in 2..h.rows {
            for j in 0..i.saturating_sub(1) {
                if j < h.cols && i < h.rows {
                    assert!(
                        h.get(i, j).abs() < 1e-10,
                        "H[{},{}] = {} should be ~0",
                        i, j, h.get(i, j)
                    );
                }
            }
        }
    }

    // ── Francis QR step ────────────────────────────────────────────────
    #[test]
    fn test_francis_qr_step() {
        // Build a 4×4 upper Hessenberg matrix and apply one Francis step
        let mut h = DenseMatrix::from_row_major(4, 4, vec![
            4.0, 1.0, 0.5, 0.1,
            1.0, 3.0, 1.0, 0.2,
            0.0, 1.0, 2.0, 0.5,
            0.0, 0.0, 1.0, 1.0,
        ]);
        let trace_before = h.trace();
        let mut q = DenseMatrix::eye(4);
        francis_qr_step(&mut h, 0, 3, Some(&mut q));
        let trace_after = h.trace();
        // Trace should be preserved (similarity transform)
        assert!(
            approx_eq(trace_before, trace_after, 1e-8),
            "Trace not preserved: {} vs {}",
            trace_before, trace_after
        );
    }

    // ── Shift-invert ───────────────────────────────────────────────────
    #[test]
    fn test_shift_invert() {
        let mut triplets = Vec::new();
        for i in 0..5 {
            triplets.push((i, i, (i + 1) as f64));
        }
        let a = CsrMatrix::from_triplets(5, 5, &triplets);
        let config = EigenConfig::default();
        // Find eigenvalue closest to 3.0 (should be 3.0 itself)
        let decomp = shift_invert_eigen(&a, 2.5, 2, &config).unwrap();
        assert!(!decomp.eigenvalues.is_empty());
        // The closest eigenvalue to 2.5 should be 2.0 or 3.0
        let closest = decomp.eigenvalues[0];
        assert!(
            approx_eq(closest, 2.0, 0.5) || approx_eq(closest, 3.0, 0.5),
            "Expected eigenvalue near 2.5, got {}",
            closest
        );
    }

    // ── Jacobi method ──────────────────────────────────────────────────
    #[test]
    fn test_jacobi_eigen() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            4.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ]);
        let decomp = jacobi_eigen(&a, 1000, 1e-12).unwrap();
        let mut evals = decomp.eigenvalues.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Verify trace
        let sum: f64 = evals.iter().sum();
        assert!(approx_eq(sum, 9.0, 1e-8));
        // Verify A*v ≈ λ*v
        if let Some(ref vecs) = decomp.eigenvectors {
            for (col, &lam) in decomp.eigenvalues.iter().enumerate() {
                let v: Vec<f64> = (0..3).map(|i| vecs.get(i, col)).collect();
                let av = a.mul_vec(&v).unwrap();
                for i in 0..3 {
                    assert!(
                        approx_eq(av[i], lam * v[i], 1e-8),
                        "Jacobi eigenvector verification failed at [{},{}]",
                        i, col
                    );
                }
            }
        }
    }

    // ── Sort ascending ─────────────────────────────────────────────────
    #[test]
    fn test_sort_ascending() {
        let mut decomp = EigenDecomposition {
            eigenvalues: vec![5.0, 1.0, 3.0],
            eigenvectors: None,
            n: 3,
        };
        decomp.sort_ascending();
        assert!(approx_eq(decomp.eigenvalues[0], 1.0, 1e-15));
        assert!(approx_eq(decomp.eigenvalues[1], 3.0, 1e-15));
        assert!(approx_eq(decomp.eigenvalues[2], 5.0, 1e-15));
    }

    // ── Sort by magnitude ──────────────────────────────────────────────
    #[test]
    fn test_sort_by_magnitude() {
        let mut decomp = EigenDecomposition {
            eigenvalues: vec![1.0, -5.0, 3.0],
            eigenvectors: None,
            n: 3,
        };
        decomp.sort_by_magnitude();
        assert!(approx_eq(decomp.eigenvalues[0], -5.0, 1e-15));
        assert!(approx_eq(decomp.eigenvalues[1], 3.0, 1e-15));
        assert!(approx_eq(decomp.eigenvalues[2], 1.0, 1e-15));
    }

    // ── Convergence monitor ────────────────────────────────────────────
    #[test]
    fn test_convergence_monitor() {
        let mut mon = EigenConvergenceMonitor::new();
        assert!(!mon.converged(1e-6));
        mon.record(1.0);
        mon.record(0.1);
        mon.record(0.01);
        mon.record(1e-7);
        assert_eq!(mon.iterations(), 4);
        assert!(mon.converged(1e-6));
        assert!(!mon.converged(1e-8));
    }

    // ── Large symmetric (5×5 from Hilbert-like) ────────────────────────
    #[test]
    fn test_larger_symmetric_qr() {
        let n = 5;
        let a = DenseMatrix::from_fn(n, n, |i, j| 1.0 / ((i + j + 1) as f64));
        let config = EigenConfig::new().with_method(EigenMethod::QrAlgorithm);
        let decomp = symmetric_eigen(&a, &config).unwrap();
        let sum: f64 = decomp.eigenvalues.iter().sum();
        assert!(approx_eq(sum, a.trace(), 1e-6), "Eigenvalue sum {} != trace {}", sum, a.trace());
        // All eigenvalues of a Hilbert matrix are positive
        for &ev in &decomp.eigenvalues {
            assert!(ev > -1e-10, "Expected positive eigenvalue, got {}", ev);
        }
    }

    // ── Non-symmetric real eigenvalues ─────────────────────────────────
    #[test]
    fn test_nonsymmetric_real() {
        // Upper triangular → eigenvalues are diagonal
        let a = DenseMatrix::from_row_major(3, 3, vec![
            2.0, 1.0, 3.0,
            0.0, 5.0, 4.0,
            0.0, 0.0, 1.0,
        ]);
        let config = EigenConfig::default();
        let result = nonsymmetric_eigen(&a, &config).unwrap();
        let mut reals = result.real_eigenvalues.clone();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(reals[0], 1.0, 1e-6));
        assert!(approx_eq(reals[1], 2.0, 1e-6));
        assert!(approx_eq(reals[2], 5.0, 1e-6));
        for &im in &result.imag_eigenvalues {
            assert!(im.abs() < 1e-6, "Expected real eigenvalues, got imag={}", im);
        }
    }
}
