//! Symmetric tridiagonal matrix algorithms.
//!
//! Provides the [`SymmetricTridiagonal`] type together with O(n) solves
//! (Thomas algorithm), eigenvalue computation via QR iteration, bisection,
//! and divide-and-conquer, plus various helper routines.

use crate::{DenseMatrix, DecompError, DecompResult};

// ---------------------------------------------------------------------------
// Tiny constant used to avoid division by zero in Sturm sequences.
// ---------------------------------------------------------------------------
const TINY: f64 = 1e-300;

// =========================================================================
// 8. eigen_2x2
// =========================================================================

/// Eigenvalues of the symmetric 2×2 matrix `[[a, b], [b, d]]`.
///
/// Returns `(lam1, lam2)` with `lam1 <= lam2`.
pub fn eigen_2x2(a: f64, b: f64, d: f64) -> (f64, f64) {
    let mean = 0.5 * (a + d);
    let half_diff = 0.5 * (a - d);
    let disc = (half_diff * half_diff + b * b).sqrt();
    (mean - disc, mean + disc)
}

// =========================================================================
// 9. eigen_3x3_tridiag
// =========================================================================

/// Direct eigenvalues of a 3×3 symmetric tridiagonal matrix with diagonal
/// `a` and off-diagonal `b`.  Uses Cardano's formula for the depressed
/// cubic.  Returns eigenvalues sorted ascending.
pub fn eigen_3x3_tridiag(a: [f64; 3], b: [f64; 2]) -> [f64; 3] {
    // Characteristic polynomial:
    //   (a0-λ)((a1-λ)(a2-λ) - b1²) - b0²(a2-λ) = 0
    // Expand to λ³ - pλ² + qλ - r = 0
    let p = a[0] + a[1] + a[2]; // trace
    let q = a[0] * a[1] + a[1] * a[2] + a[0] * a[2] - b[0] * b[0] - b[1] * b[1];
    let r = a[0] * a[1] * a[2] - a[0] * b[1] * b[1] - a[2] * b[0] * b[0];

    // Depressed cubic: t³ + pt' + q' = 0 with λ = t + p/3
    let p3 = p / 3.0;
    let pp = q - p * p3; // coefficient of t   (negative of usual p')
    let qq = r - q * p3 + 2.0 * p3 * p3 * p3; // free term

    // All three roots are real for a symmetric matrix.
    let m = (-pp / 3.0).max(0.0);
    let sqrt_m = m.sqrt();
    let theta = if sqrt_m == 0.0 {
        0.0
    } else {
        let arg = (-qq / (2.0 * sqrt_m * sqrt_m * sqrt_m)).clamp(-1.0, 1.0);
        arg.acos() / 3.0
    };

    let two_sqrt_m = 2.0 * sqrt_m;
    let mut eigs = [
        two_sqrt_m * theta.cos() + p3,
        two_sqrt_m * (theta - std::f64::consts::FRAC_PI_3 * 2.0).cos() + p3,
        two_sqrt_m * (theta + std::f64::consts::FRAC_PI_3 * 2.0).cos() + p3,
    ];
    eigs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

// =========================================================================
// 7. gershgorin_bounds
// =========================================================================

/// Gershgorin disc bounds on the eigenvalues of a symmetric tridiagonal
/// matrix.  Returns `(lo, hi)` such that every eigenvalue lies in `[lo, hi]`.
pub fn gershgorin_bounds(alpha: &[f64], beta: &[f64]) -> (f64, f64) {
    let n = alpha.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let radius = if i == 0 {
            if beta.is_empty() { 0.0 } else { beta[0].abs() }
        } else if i == n - 1 {
            beta[i - 1].abs()
        } else {
            beta[i - 1].abs() + beta[i].abs()
        };
        lo = lo.min(alpha[i] - radius);
        hi = hi.max(alpha[i] + radius);
    }
    (lo, hi)
}

// =========================================================================
// 3. sturm_count
// =========================================================================

/// Count the number of eigenvalues of a symmetric tridiagonal matrix that
/// are **strictly less than** `mu`, using the Sturm sequence.
///
/// `alpha` is the diagonal (length n), `beta` is the off-diagonal (length n-1).
pub fn sturm_count(alpha: &[f64], beta: &[f64], mu: f64) -> usize {
    let n = alpha.len();
    if n == 0 {
        return 0;
    }
    let mut count = 0usize;
    let mut d = alpha[0] - mu;
    if d < 0.0 {
        count += 1;
    }
    for k in 1..n {
        let b2 = beta[k - 1] * beta[k - 1];
        let prev = if d == 0.0 { TINY } else { d };
        d = (alpha[k] - mu) - b2 / prev;
        if d < 0.0 {
            count += 1;
        }
    }
    count
}

// =========================================================================
// 2. thomas_solve
// =========================================================================

/// Solve `T x = rhs` where `T` is a symmetric tridiagonal matrix with
/// diagonal `alpha` and off-diagonal `beta` using the Thomas algorithm in
/// O(n) time.
///
/// `alpha` has length n, `beta` has length n-1, `rhs` has length n.
pub fn thomas_solve(
    alpha: &[f64],
    beta: &[f64],
    rhs: &[f64],
) -> DecompResult<Vec<f64>> {
    let n = alpha.len();
    if n == 0 {
        return Err(DecompError::EmptyMatrix {
            context: "thomas_solve: empty system".into(),
        });
    }
    if beta.len() + 1 != n {
        return Err(DecompError::VectorLengthMismatch {
            expected: n - 1,
            actual: beta.len(),
        });
    }
    if rhs.len() != n {
        return Err(DecompError::VectorLengthMismatch {
            expected: n,
            actual: rhs.len(),
        });
    }

    // gamma = super-diagonal = beta for symmetric case
    let gamma = beta;

    // Forward sweep
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    if alpha[0].abs() < TINY {
        return Err(DecompError::NumericalInstability {
            context: "thomas_solve: zero pivot at index 0".into(),
        });
    }
    c_prime[0] = if n > 1 { gamma[0] / alpha[0] } else { 0.0 };
    d_prime[0] = rhs[0] / alpha[0];

    for i in 1..n {
        let denom = alpha[i] - beta[i - 1] * c_prime[i - 1];
        if denom.abs() < TINY {
            return Err(DecompError::NumericalInstability {
                context: format!("thomas_solve: zero pivot at index {}", i),
            });
        }
        c_prime[i] = if i < n - 1 { gamma[i] / denom } else { 0.0 };
        d_prime[i] = (rhs[i] - beta[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

// =========================================================================
// 4. bisection_eigenvalues
// =========================================================================

/// Find all eigenvalues of a [`SymmetricTridiagonal`] matrix in the interval
/// `[lo, hi]` by bisection combined with the Sturm sequence count.
pub fn bisection_eigenvalues(
    t: &SymmetricTridiagonal,
    lo: f64,
    hi: f64,
    tol: f64,
) -> Vec<f64> {
    let n = t.n;
    if n == 0 {
        return vec![];
    }
    let total = sturm_count(&t.alpha, &t.beta, hi) as isize
        - sturm_count(&t.alpha, &t.beta, lo) as isize;
    if total <= 0 {
        return vec![];
    }
    let k = total as usize;
    let mut eigenvalues = Vec::with_capacity(k);
    bisect_recursive(&t.alpha, &t.beta, lo, hi, k, tol, &mut eigenvalues);
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Recursively bisect `[lo, hi]` to isolate `k` eigenvalues.
fn bisect_recursive(
    alpha: &[f64],
    beta: &[f64],
    lo: f64,
    hi: f64,
    k: usize,
    tol: f64,
    out: &mut Vec<f64>,
) {
    if k == 0 {
        return;
    }
    if hi - lo < tol {
        for _ in 0..k {
            out.push(0.5 * (lo + hi));
        }
        return;
    }
    let mid = 0.5 * (lo + hi);
    let count_lo = sturm_count(alpha, beta, lo);
    let count_mid = sturm_count(alpha, beta, mid);
    let k_left = (count_mid - count_lo) as usize;
    let k_right = k - k_left;
    bisect_recursive(alpha, beta, lo, mid, k_left, tol, out);
    bisect_recursive(alpha, beta, mid, hi, k_right, tol, out);
}

// =========================================================================
// 5. tridiagonal_qr_eigen
// =========================================================================

/// Implicit QR iteration with Wilkinson shift for computing the eigenvalues
/// (and optionally eigenvectors) of a symmetric tridiagonal matrix.
///
/// On entry `alpha` and `beta` hold the diagonal and off-diagonal.  On exit
/// `alpha` holds the eigenvalues (unsorted).  If `eigvecs` is `Some`, the
/// Givens rotations are accumulated into that matrix (must be initialised to
/// identity of appropriate size).
///
/// Returns the total number of QR iterations performed.
pub fn tridiagonal_qr_eigen(
    alpha: &mut Vec<f64>,
    beta: &mut Vec<f64>,
    mut eigvecs: Option<&mut DenseMatrix>,
    max_iter: usize,
) -> DecompResult<usize> {
    let n = alpha.len();
    if n == 0 {
        return Ok(0);
    }
    if n == 1 {
        return Ok(0);
    }

    let eps = f64::EPSILON;
    let mut total_iter = 0usize;
    let mut end = n - 1;

    while end > 0 {
        // Deflation: check if beta[end-1] is negligible
        let off_norm = alpha[end].abs() + alpha[end - 1].abs();
        if beta[end - 1].abs() <= eps * off_norm.max(TINY) {
            end -= 1;
            continue;
        }

        // Find the start of the unreduced block
        let mut start = end - 1;
        while start > 0 {
            let norm = alpha[start].abs() + alpha[start - 1].abs();
            if beta[start - 1].abs() <= eps * norm.max(TINY) {
                break;
            }
            start -= 1;
        }

        if total_iter >= max_iter {
            return Err(DecompError::ConvergenceFailure {
                iterations: max_iter,
                context: format!(
                    "tridiagonal_qr_eigen: did not converge, {} unreduced elements remain",
                    end - start + 1
                ),
            });
        }

        // Wilkinson shift: eigenvalue of trailing 2×2 closer to alpha[end]
        let d = (alpha[end - 1] - alpha[end]) * 0.5;
        let b2 = beta[end - 1] * beta[end - 1];
        let sign_d = if d >= 0.0 { 1.0 } else { -1.0 };
        let shift = alpha[end] - b2 / (d + sign_d * (d * d + b2).sqrt());

        // Implicit QR step (bulge chase) with Givens rotations
        let mut bulge;
        let mut x = alpha[start] - shift;
        let mut z = beta[start];

        for k in start..end {
            // Compute Givens rotation to zero z
            let (c, s, _r) = givens_cs(x, z);

            // Apply rotation to tridiagonal: rows/cols k and k+1
            if k > start {
                beta[k - 1] = c * x + s * z;
            }

            let ak = alpha[k];
            let bk = beta[k];
            let ak1 = alpha[k + 1];

            let tau1 = c * ak + s * bk;
            let tau2 = -s * ak + c * bk;
            let tau3 = c * bk + s * ak1;
            let tau4 = -s * bk + c * ak1;

            alpha[k] = c * tau1 + s * tau3;
            beta[k] = c * tau2 + s * tau4;
            alpha[k + 1] = -s * tau2 + c * tau4;

            // Accumulate eigenvectors
            if let Some(ref mut q) = eigvecs {
                let nn = q.rows;
                for i in 0..nn {
                    let qi_k = q.get(i, k);
                    let qi_k1 = q.get(i, k + 1);
                    q.set(i, k, c * qi_k + s * qi_k1);
                    q.set(i, k + 1, -s * qi_k + c * qi_k1);
                }
            }

            // Prepare next bulge
            if k < end - 1 {
                x = beta[k];
                bulge = s * beta[k + 1];
                beta[k + 1] *= c;
                z = bulge;
            }
        }

        total_iter += 1;
    }

    Ok(total_iter)
}

/// Compute (c, s, r) for a Givens rotation that zeros the second component.
fn givens_cs(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        (1.0, 0.0, a)
    } else if a == 0.0 {
        (0.0, b.signum(), b.abs())
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = (1.0 + tau * tau).sqrt().recip() * b.signum();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c = (1.0 + tau * tau).sqrt().recip() * a.signum();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

// =========================================================================
// 6. tridiagonal_divide_conquer
// =========================================================================

/// Divide-and-conquer eigenvalue solver for a symmetric tridiagonal matrix.
///
/// Returns `(eigenvalues, eigenvectors)`.
pub fn tridiagonal_divide_conquer(
    t: &SymmetricTridiagonal,
) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = t.n;
    if n == 0 {
        return Err(DecompError::EmptyMatrix {
            context: "tridiagonal_divide_conquer: empty matrix".into(),
        });
    }
    if n == 1 {
        let q = DenseMatrix::eye(1);
        return Ok((vec![t.alpha[0]], q));
    }
    if n == 2 {
        let (l1, l2) = eigen_2x2(t.alpha[0], t.beta[0], t.alpha[1]);
        let mut q = DenseMatrix::zeros(2, 2);
        // Eigenvectors of [[a,b],[b,d]]
        let a = t.alpha[0];
        let b = t.beta[0];
        if b.abs() < f64::EPSILON * (a.abs() + t.alpha[1].abs()).max(1.0) {
            q = DenseMatrix::eye(2);
        } else {
            // v = (l - d, b) for each eigenvalue
            let _d = t.alpha[1];
            for (col, &lam) in [l1, l2].iter().enumerate() {
                let v0 = b;
                let v1 = lam - a;
                let norm = (v0 * v0 + v1 * v1).sqrt();
                if norm > 0.0 {
                    q.set(0, col, v0 / norm);
                    q.set(1, col, v1 / norm);
                } else {
                    q.set(col, col, 1.0);
                }
            }
        }
        return Ok((vec![l1, l2], q));
    }
    if n == 3 {
        let eigs = eigen_3x3_tridiag(
            [t.alpha[0], t.alpha[1], t.alpha[2]],
            [t.beta[0], t.beta[1]],
        );
        // Compute eigenvectors via inverse iteration
        let q = eigenvectors_inverse_iteration(&t.alpha, &t.beta, &eigs)?;
        return Ok((eigs.to_vec(), q));
    }

    // Divide at the middle
    let m = n / 2; // split: [0..m] and [m+1..n-1]
    let rho = t.beta[m - 1]; // coupling element (use m-1 for 0-indexed beta)

    // Form sub-problems by removing the coupling
    let mut alpha1: Vec<f64> = t.alpha[..m].to_vec();
    let beta1: Vec<f64> = t.beta[..m - 1].to_vec();
    alpha1[m - 1] -= rho; // rank-1 modification

    let mut alpha2: Vec<f64> = t.alpha[m..].to_vec();
    let beta2: Vec<f64> = t.beta[m..n - 1].to_vec();
    alpha2[0] -= rho; // rank-1 modification

    let n1 = alpha1.len();
    let n2 = alpha2.len();

    let t1 = SymmetricTridiagonal {
        alpha: alpha1,
        beta: beta1,
        n: n1,
    };
    let t2 = SymmetricTridiagonal {
        alpha: alpha2,
        beta: beta2,
        n: n2,
    };

    let (d1, q1) = tridiagonal_divide_conquer(&t1)?;
    let (d2, q2) = tridiagonal_divide_conquer(&t2)?;

    // Merge: solve the secular equation
    //   1 + rho * sum_i z_i^2 / (d_i - lambda) = 0
    // where z = Q^T * e  (the last row of Q1 and first row of Q2).

    let mut d_all = Vec::with_capacity(n);
    d_all.extend_from_slice(&d1);
    d_all.extend_from_slice(&d2);

    let mut z = Vec::with_capacity(n);
    for j in 0..n1 {
        z.push(q1.get(n1 - 1, j));
    }
    for j in 0..n2 {
        z.push(q2.get(0, j));
    }

    // Sort d_all and z together by d_all values
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| d_all[a].partial_cmp(&d_all[b]).unwrap_or(std::cmp::Ordering::Equal));
    let d_sorted: Vec<f64> = perm.iter().map(|&i| d_all[i]).collect();
    let z_sorted: Vec<f64> = perm.iter().map(|&i| z[i]).collect();

    // Solve secular equation for each eigenvalue
    let mut new_eigs = Vec::with_capacity(n);
    for k in 0..n {
        let lam = solve_secular_equation(&d_sorted, &z_sorted, rho, k);
        new_eigs.push(lam);
    }

    // Compute eigenvectors of the merged problem
    let mut q_merged = DenseMatrix::zeros(n, n);
    for k in 0..n {
        let lam = new_eigs[k];
        let mut v = vec![0.0; n];
        let mut norm_sq = 0.0;
        for i in 0..n {
            let denom = d_sorted[i] - lam;
            let val = if denom.abs() < TINY {
                1.0 / TINY
            } else {
                z_sorted[i] / denom
            };
            v[i] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for i in 0..n {
                v[i] /= norm;
            }
        }
        // Un-permute
        let mut v_unperm = vec![0.0; n];
        for i in 0..n {
            v_unperm[perm[i]] = v[i];
        }
        for i in 0..n {
            q_merged.set(i, k, v_unperm[i]);
        }
    }

    // Final eigenvectors: Q = blkdiag(Q1, Q2) * Q_merged
    let mut q_final = DenseMatrix::zeros(n, n);
    for j in 0..n {
        // Top block: Q1 * q_merged[0..n1, j]
        for i in 0..n1 {
            let mut s = 0.0;
            for l in 0..n1 {
                s += q1.get(i, l) * q_merged.get(l, j);
            }
            q_final.set(i, j, s);
        }
        // Bottom block: Q2 * q_merged[n1..n, j]
        for i in 0..n2 {
            let mut s = 0.0;
            for l in 0..n2 {
                s += q2.get(i, l) * q_merged.get(n1 + l, j);
            }
            q_final.set(n1 + i, j, s);
        }
    }

    Ok((new_eigs, q_final))
}

/// Solve the secular equation `1 + rho * sum z_i^2/(d_i - lambda) = 0` for
/// the k-th root, which lies in the interval `(d[k], d[k+1])`.
fn solve_secular_equation(d: &[f64], z: &[f64], rho: f64, k: usize) -> f64 {
    let n = d.len();
    // Bracket the eigenvalue
    let (mut lo, mut hi) = if rho > 0.0 {
        if k < n - 1 {
            (d[k], d[k + 1])
        } else {
            let spread = gershgorin_spread(d, z, rho);
            (d[n - 1], d[n - 1] + spread)
        }
    } else {
        if k > 0 {
            (d[k - 1], d[k])
        } else {
            let spread = gershgorin_spread(d, z, rho);
            (d[0] - spread, d[0])
        }
    };

    // Bisection with 100 iterations
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if (hi - lo).abs() < f64::EPSILON * mid.abs().max(1.0) * 10.0 {
            return mid;
        }
        let f = secular_function(d, z, rho, mid);
        // f changes sign at each root; secular function is decreasing between poles for rho > 0
        if rho > 0.0 {
            if f > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        } else {
            if f < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    0.5 * (lo + hi)
}

fn secular_function(d: &[f64], z: &[f64], rho: f64, lam: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..d.len() {
        let denom = d[i] - lam;
        if denom.abs() > TINY {
            sum += z[i] * z[i] / denom;
        } else {
            sum += z[i] * z[i] / TINY.copysign(denom);
        }
    }
    1.0 + rho * sum
}

fn gershgorin_spread(_d: &[f64], z: &[f64], rho: f64) -> f64 {
    let z_norm_sq: f64 = z.iter().map(|&zi| zi * zi).sum();
    rho.abs() * z_norm_sq + 1.0
}

/// Compute eigenvectors via inverse iteration for a symmetric tridiagonal matrix.
fn eigenvectors_inverse_iteration(
    alpha: &[f64],
    beta: &[f64],
    eigenvalues: &[f64],
) -> DecompResult<DenseMatrix> {
    let n = alpha.len();
    let mut q = DenseMatrix::zeros(n, n);

    for (col, &lam) in eigenvalues.iter().enumerate() {
        // Shift the diagonal
        let shifted_alpha: Vec<f64> = alpha.iter().map(|&a| a - lam).collect();

        // Solve (T - lambda*I) v = random_rhs via a modified Thomas algorithm
        // Use a random starting vector and iterate
        let mut v = vec![1.0; n];
        // Vary initial vector slightly to break degeneracy
        for i in 0..n {
            v[i] = 1.0 + 0.1 * (i as f64);
        }

        for _iter in 0..3 {
            // Forward elimination for shifted system
            let mut c_prime = vec![0.0; n];
            let mut d_prime = vec![0.0; n];

            let mut diag = shifted_alpha.clone();
            // Add small perturbation to avoid singularity
            for d_val in diag.iter_mut() {
                if d_val.abs() < 1e-14 {
                    *d_val = 1e-14;
                }
            }

            c_prime[0] = if n > 1 { beta.get(0).copied().unwrap_or(0.0) / diag[0] } else { 0.0 };
            d_prime[0] = v[0] / diag[0];

            for i in 1..n {
                let b = beta[i - 1];
                let denom = diag[i] - b * c_prime[i - 1];
                let denom = if denom.abs() < 1e-14 { 1e-14_f64.copysign(denom) } else { denom };
                c_prime[i] = if i < n - 1 { beta[i] / denom } else { 0.0 };
                d_prime[i] = (v[i] - b * d_prime[i - 1]) / denom;
            }

            v[n - 1] = d_prime[n - 1];
            for i in (0..n - 1).rev() {
                v[i] = d_prime[i] - c_prime[i] * v[i + 1];
            }

            // Normalize
            let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
        }

        for i in 0..n {
            q.set(i, col, v[i]);
        }
    }

    Ok(q)
}

// =========================================================================
// 10. detect_bandwidth
// =========================================================================

/// Detect the bandwidth of a dense matrix.  Returns `(lower, upper)` where
/// `lower` is the lower bandwidth and `upper` is the upper bandwidth.
///
/// An element `A[i][j]` is considered non-zero if `|A[i][j]| > tol`.
pub fn detect_bandwidth(mat: &DenseMatrix, tol: f64) -> (usize, usize) {
    let (rows, cols) = mat.shape();
    let mut lower = 0usize;
    let mut upper = 0usize;
    for i in 0..rows {
        for j in 0..cols {
            if mat.get(i, j).abs() > tol {
                if i > j {
                    lower = lower.max(i - j);
                } else if j > i {
                    upper = upper.max(j - i);
                }
            }
        }
    }
    (lower, upper)
}

// =========================================================================
// 1. SymmetricTridiagonal
// =========================================================================

/// A symmetric tridiagonal matrix stored efficiently as two vectors:
/// - `alpha`: the main diagonal, length `n`.
/// - `beta`:  the sub/super-diagonal, length `n-1`.
#[derive(Debug, Clone)]
pub struct SymmetricTridiagonal {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub n: usize,
}

impl SymmetricTridiagonal {
    /// Create a new symmetric tridiagonal from diagonal `alpha` and
    /// off-diagonal `beta`.  Validates that `beta.len() == alpha.len() - 1`
    /// (or both are empty).
    pub fn new(alpha: Vec<f64>, beta: Vec<f64>) -> DecompResult<Self> {
        let n = alpha.len();
        if n == 0 && beta.is_empty() {
            return Ok(Self { alpha, beta, n: 0 });
        }
        if n == 0 {
            return Err(DecompError::EmptyMatrix {
                context: "SymmetricTridiagonal::new: empty diagonal with non-empty off-diagonal".into(),
            });
        }
        if beta.len() != n - 1 {
            return Err(DecompError::VectorLengthMismatch {
                expected: n - 1,
                actual: beta.len(),
            });
        }
        Ok(Self { alpha, beta, n })
    }

    /// Extract a `SymmetricTridiagonal` from a dense matrix.  The matrix must
    /// be square and (approximately) symmetric tridiagonal – off-tridiagonal
    /// entries are ignored.
    pub fn from_dense(mat: &DenseMatrix) -> DecompResult<Self> {
        if !mat.is_square() {
            return Err(DecompError::NotSquare {
                rows: mat.rows,
                cols: mat.cols,
            });
        }
        let n = mat.rows;
        if n == 0 {
            return Err(DecompError::EmptyMatrix {
                context: "SymmetricTridiagonal::from_dense: empty matrix".into(),
            });
        }
        let mut alpha = vec![0.0; n];
        let mut beta = vec![0.0; if n > 1 { n - 1 } else { 0 }];
        for i in 0..n {
            alpha[i] = mat.get(i, i);
        }
        for i in 0..n.saturating_sub(1) {
            // Average the sub- and super-diagonal for symmetry
            beta[i] = 0.5 * (mat.get(i, i + 1) + mat.get(i + 1, i));
        }
        Ok(Self { alpha, beta, n })
    }

    /// Size of the matrix (number of rows/columns).
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Convert to a dense matrix.
    pub fn to_dense(&self) -> DenseMatrix {
        let n = self.n;
        let mut mat = DenseMatrix::zeros(n, n);
        for i in 0..n {
            mat.set(i, i, self.alpha[i]);
        }
        for i in 0..n.saturating_sub(1) {
            mat.set(i, i + 1, self.beta[i]);
            mat.set(i + 1, i, self.beta[i]);
        }
        mat
    }

    /// Compute all eigenvalues using an appropriate method for the matrix size.
    pub fn eigenvalues(&self) -> DecompResult<Vec<f64>> {
        match self.n {
            0 => Err(DecompError::EmptyMatrix {
                context: "eigenvalues: empty tridiagonal".into(),
            }),
            1 => Ok(vec![self.alpha[0]]),
            2 => {
                let (l1, l2) = eigen_2x2(self.alpha[0], self.beta[0], self.alpha[1]);
                Ok(vec![l1, l2])
            }
            3 => {
                let eigs = eigen_3x3_tridiag(
                    [self.alpha[0], self.alpha[1], self.alpha[2]],
                    [self.beta[0], self.beta[1]],
                );
                Ok(eigs.to_vec())
            }
            _ => {
                let mut a = self.alpha.clone();
                let mut b = self.beta.clone();
                tridiagonal_qr_eigen(&mut a, &mut b, None, 30 * self.n)?;
                a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                Ok(a)
            }
        }
    }

    /// Compute eigenvalues and eigenvectors.
    pub fn eigen_decomposition(&self) -> DecompResult<(Vec<f64>, DenseMatrix)> {
        let n = self.n;
        if n == 0 {
            return Err(DecompError::EmptyMatrix {
                context: "eigen_decomposition: empty tridiagonal".into(),
            });
        }
        if n <= 3 {
            return tridiagonal_divide_conquer(self);
        }
        let mut a = self.alpha.clone();
        let mut b = self.beta.clone();
        let mut q = DenseMatrix::eye(n);
        tridiagonal_qr_eigen(&mut a, &mut b, Some(&mut q), 30 * n)?;

        // Sort eigenvalues and rearrange eigenvectors
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap_or(std::cmp::Ordering::Equal));

        let eigs: Vec<f64> = perm.iter().map(|&i| a[i]).collect();
        let mut q_sorted = DenseMatrix::zeros(n, n);
        for (new_col, &old_col) in perm.iter().enumerate() {
            for row in 0..n {
                q_sorted.set(row, new_col, q.get(row, old_col));
            }
        }

        Ok((eigs, q_sorted))
    }

    /// Solve `T x = rhs` using the Thomas algorithm.
    pub fn solve(&self, rhs: &[f64]) -> DecompResult<Vec<f64>> {
        if rhs.len() != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: rhs.len(),
            });
        }
        thomas_solve(&self.alpha, &self.beta, rhs)
    }

    /// Compute the determinant using the recurrence for tridiagonal matrices:
    ///   `f[0] = alpha[0]`, `f[k] = alpha[k]*f[k-1] - beta[k-1]^2 * f[k-2]`.
    pub fn determinant(&self) -> f64 {
        let n = self.n;
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return self.alpha[0];
        }
        let mut f_prev2 = 1.0; // f[-1]
        let mut f_prev1 = self.alpha[0]; // f[0]
        for k in 1..n {
            let f_k = self.alpha[k] * f_prev1 - self.beta[k - 1] * self.beta[k - 1] * f_prev2;
            f_prev2 = f_prev1;
            f_prev1 = f_k;
        }
        f_prev1
    }

    /// Count the number of eigenvalues strictly below `mu` using the Sturm
    /// sequence.
    pub fn eigenvalue_count_below(&self, mu: f64) -> usize {
        sturm_count(&self.alpha, &self.beta, mu)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ------- Thomas solve -------

    #[test]
    fn test_thomas_solve_simple() {
        // T = [[2,1,0],[1,3,1],[0,1,2]], rhs = [1,2,3]
        let alpha = vec![2.0, 3.0, 2.0];
        let beta = vec![1.0, 1.0];
        let rhs = vec![1.0, 2.0, 3.0];
        let x = thomas_solve(&alpha, &beta, &rhs).unwrap();

        // Verify T*x = rhs
        let t = SymmetricTridiagonal::new(alpha, beta).unwrap();
        let dense = t.to_dense();
        let ax = dense.mul_vec(&x).unwrap();
        for i in 0..3 {
            assert!(approx_eq(ax[i], rhs[i], TOL), "Mismatch at {}: {} vs {}", i, ax[i], rhs[i]);
        }
    }

    #[test]
    fn test_thomas_solve_size_one() {
        let x = thomas_solve(&[5.0], &[], &[10.0]).unwrap();
        assert!(approx_eq(x[0], 2.0, TOL));
    }

    #[test]
    fn test_thomas_solve_error_mismatch() {
        let result = thomas_solve(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    // ------- Eigenvalues of known matrices -------

    #[test]
    fn test_eigenvalues_identity() {
        let t = SymmetricTridiagonal::new(vec![1.0; 4], vec![0.0; 3]).unwrap();
        let eigs = t.eigenvalues().unwrap();
        for &e in &eigs {
            assert!(approx_eq(e, 1.0, TOL), "Expected 1.0, got {}", e);
        }
    }

    #[test]
    fn test_eigenvalues_2x2() {
        // [[3,1],[1,3]] => eigenvalues 2, 4
        let t = SymmetricTridiagonal::new(vec![3.0, 3.0], vec![1.0]).unwrap();
        let mut eigs = t.eigenvalues().unwrap();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 2.0, TOL));
        assert!(approx_eq(eigs[1], 4.0, TOL));
    }

    #[test]
    fn test_eigenvalues_3x3() {
        // [[2,1,0],[1,2,1],[0,1,2]] => eigenvalues 2-sqrt2, 2, 2+sqrt2
        let t = SymmetricTridiagonal::new(vec![2.0, 2.0, 2.0], vec![1.0, 1.0]).unwrap();
        let mut eigs = t.eigenvalues().unwrap();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let s2 = std::f64::consts::SQRT_2;
        assert!(approx_eq(eigs[0], 2.0 - s2, 1e-8));
        assert!(approx_eq(eigs[1], 2.0, 1e-8));
        assert!(approx_eq(eigs[2], 2.0 + s2, 1e-8));
    }

    #[test]
    fn test_eigenvalues_larger_qr() {
        // 5x5 tridiagonal: alpha = [4,4,4,4,4], beta = [1,1,1,1]
        // Known eigenvalues for this Toeplitz tridiag: 4 + 2*cos(k*pi/6) for k=1..5
        let n = 5;
        let t = SymmetricTridiagonal::new(vec![4.0; n], vec![1.0; n - 1]).unwrap();
        let mut eigs = t.eigenvalues().unwrap();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected: Vec<f64> = (1..=n)
            .map(|k| 4.0 + 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos())
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..n {
            assert!(
                approx_eq(eigs[i], expected[i], 1e-8),
                "Eigenvalue {}: got {}, expected {}",
                i,
                eigs[i],
                expected[i]
            );
        }
    }

    // ------- Sturm count -------

    #[test]
    fn test_sturm_count_basic() {
        // [[2,1],[1,2]] => eigenvalues 1 and 3
        let alpha = vec![2.0, 2.0];
        let beta = vec![1.0];
        assert_eq!(sturm_count(&alpha, &beta, 0.0), 0);
        assert_eq!(sturm_count(&alpha, &beta, 1.5), 1);
        assert_eq!(sturm_count(&alpha, &beta, 4.0), 2);
    }

    // ------- Bisection eigenvalues -------

    #[test]
    fn test_bisection_eigenvalues() {
        let t = SymmetricTridiagonal::new(vec![2.0, 2.0, 2.0], vec![1.0, 1.0]).unwrap();
        let (lo, hi) = gershgorin_bounds(&t.alpha, &t.beta);
        let eigs = bisection_eigenvalues(&t, lo, hi, 1e-12);
        assert_eq!(eigs.len(), 3);
        let s2 = std::f64::consts::SQRT_2;
        assert!(approx_eq(eigs[0], 2.0 - s2, 1e-8));
        assert!(approx_eq(eigs[1], 2.0, 1e-8));
        assert!(approx_eq(eigs[2], 2.0 + s2, 1e-8));
    }

    // ------- QR iteration -------

    #[test]
    fn test_qr_iteration_with_eigenvectors() {
        let t = SymmetricTridiagonal::new(vec![2.0, 3.0, 1.0], vec![1.0, 0.5]).unwrap();
        let (eigs, q) = t.eigen_decomposition().unwrap();
        // Verify Q * diag(eigs) * Q^T ≈ T
        let dense = t.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                let mut val = 0.0;
                for k in 0..3 {
                    val += q.get(i, k) * eigs[k] * q.get(j, k);
                }
                assert!(
                    approx_eq(val, dense.get(i, j), 1e-6),
                    "Reconstruction mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    val,
                    dense.get(i, j)
                );
            }
        }
    }

    // ------- Divide-and-conquer -------

    #[test]
    fn test_divide_conquer() {
        let t = SymmetricTridiagonal::new(vec![4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 1.0]).unwrap();
        let (mut eigs_dc, _q) = tridiagonal_divide_conquer(&t).unwrap();
        eigs_dc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut eigs_qr = t.eigenvalues().unwrap();
        eigs_qr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..4 {
            assert!(
                approx_eq(eigs_dc[i], eigs_qr[i], 1e-6),
                "D&C eigenvalue {}: {} vs QR: {}",
                i,
                eigs_dc[i],
                eigs_qr[i]
            );
        }
    }

    // ------- Gershgorin bounds -------

    #[test]
    fn test_gershgorin_bounds() {
        let alpha = vec![2.0, 3.0, 5.0];
        let beta = vec![1.0, 0.5];
        let (lo, hi) = gershgorin_bounds(&alpha, &beta);
        // Row 0: [2-1, 2+1] = [1, 3]
        // Row 1: [3-1.5, 3+1.5] = [1.5, 4.5]
        // Row 2: [5-0.5, 5+0.5] = [4.5, 5.5]
        assert!(lo <= 1.0 + TOL);
        assert!(hi >= 5.5 - TOL);
    }

    // ------- 2x2 and 3x3 direct -------

    #[test]
    fn test_eigen_2x2_direct() {
        let (l1, l2) = eigen_2x2(5.0, 2.0, 1.0);
        // [[5,2],[2,1]] => mean=3, disc=sqrt(4+4)=2sqrt2
        let expected_lo = 3.0 - (4.0 + 4.0_f64).sqrt();
        let expected_hi = 3.0 + (4.0 + 4.0_f64).sqrt();
        assert!(approx_eq(l1, expected_lo, TOL));
        assert!(approx_eq(l2, expected_hi, TOL));
    }

    #[test]
    fn test_eigen_3x3_direct() {
        let eigs = eigen_3x3_tridiag([1.0, 2.0, 3.0], [0.5, 0.5]);
        // Verify they are sorted
        assert!(eigs[0] <= eigs[1]);
        assert!(eigs[1] <= eigs[2]);
        // Verify trace: sum of eigenvalues = sum of diagonal
        let trace_eigs = eigs[0] + eigs[1] + eigs[2];
        assert!(approx_eq(trace_eigs, 6.0, TOL));
    }

    // ------- Determinant -------

    #[test]
    fn test_determinant() {
        // [[2,1,0],[1,3,1],[0,1,2]]
        // det = 2*(3*2-1) - 1*(1*2-0) = 2*5 - 2 = 8
        let t = SymmetricTridiagonal::new(vec![2.0, 3.0, 2.0], vec![1.0, 1.0]).unwrap();
        assert!(approx_eq(t.determinant(), 8.0, TOL));
    }

    #[test]
    fn test_determinant_1x1() {
        let t = SymmetricTridiagonal::new(vec![7.0], vec![]).unwrap();
        assert!(approx_eq(t.determinant(), 7.0, TOL));
    }

    // ------- Edge cases -------

    #[test]
    fn test_from_dense_roundtrip() {
        let t = SymmetricTridiagonal::new(vec![1.0, 2.0, 3.0], vec![0.5, 0.7]).unwrap();
        let dense = t.to_dense();
        let t2 = SymmetricTridiagonal::from_dense(&dense).unwrap();
        for i in 0..3 {
            assert!(approx_eq(t.alpha[i], t2.alpha[i], TOL));
        }
        for i in 0..2 {
            assert!(approx_eq(t.beta[i], t2.beta[i], TOL));
        }
    }

    #[test]
    fn test_detect_bandwidth() {
        let t = SymmetricTridiagonal::new(vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 1.0, 1.0]).unwrap();
        let dense = t.to_dense();
        let (lower, upper) = detect_bandwidth(&dense, 1e-15);
        assert_eq!(lower, 1);
        assert_eq!(upper, 1);
    }

    #[test]
    fn test_detect_bandwidth_diagonal() {
        let mat = DenseMatrix::from_diag(&[1.0, 2.0, 3.0]);
        let (lower, upper) = detect_bandwidth(&mat, 1e-15);
        assert_eq!(lower, 0);
        assert_eq!(upper, 0);
    }

    #[test]
    fn test_eigenvalue_count_below() {
        // [[2,1],[1,2]] eigenvalues 1 and 3
        let t = SymmetricTridiagonal::new(vec![2.0, 2.0], vec![1.0]).unwrap();
        assert_eq!(t.eigenvalue_count_below(0.5), 0);
        assert_eq!(t.eigenvalue_count_below(1.5), 1);
        assert_eq!(t.eigenvalue_count_below(3.5), 2);
    }

    #[test]
    fn test_solve_identity() {
        let t = SymmetricTridiagonal::new(vec![1.0; 5], vec![0.0; 4]).unwrap();
        let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = t.solve(&rhs).unwrap();
        for i in 0..5 {
            assert!(approx_eq(x[i], rhs[i], TOL));
        }
    }
}
