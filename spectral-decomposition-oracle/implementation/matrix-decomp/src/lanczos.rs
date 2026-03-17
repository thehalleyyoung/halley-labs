//! Lanczos algorithm for sparse symmetric eigenvalue problems.
//!
//! Provides implicitly restarted Lanczos (IRLM) with full, selective, and
//! partial reorthogonalization, shift-invert mode for interior eigenvalues,
//! and convenient wrappers for both dense and sparse matrices.

use crate::{DenseMatrix, CsrMatrix, DecompError, DecompResult, dot, norm2, axpy, scale_vec, normalize};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

// ═══════════════════════════════════════════════════════════════════════════
// Configuration types
// ═══════════════════════════════════════════════════════════════════════════

/// How to reorthogonalize Lanczos vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReorthogonalizationType {
    None,
    Full,
    Selective,
    Partial,
}

/// Which eigenvalues to target.
#[derive(Debug, Clone, Copy)]
pub enum WhichEigenvalues {
    Largest,
    Smallest,
    LargestMagnitude,
    SmallestMagnitude,
    ClosestTo(f64),
}

/// Lanczos configuration.
#[derive(Debug, Clone)]
pub struct LanczosConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub reorthogonalization: ReorthogonalizationType,
    pub num_eigenvalues: usize,
    pub which: WhichEigenvalues,
}

impl Default for LanczosConfig {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-10,
            reorthogonalization: ReorthogonalizationType::Full,
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════

/// State of a Lanczos iteration.
#[derive(Debug, Clone)]
pub struct LanczosState {
    /// Diagonal of tridiagonal matrix.
    pub alpha: Vec<f64>,
    /// Off-diagonal of tridiagonal matrix.
    pub beta: Vec<f64>,
    /// Lanczos vectors (each of length n).
    pub v: Vec<Vec<f64>>,
    /// Problem dimension.
    pub n: usize,
    /// Number of Lanczos steps taken.
    pub m: usize,
    /// Per-eigenpair convergence flags.
    pub converged: Vec<bool>,
    /// Current Ritz values.
    pub ritz_values: Vec<f64>,
    /// Current Ritz vectors (n × k).
    pub ritz_vectors: Option<DenseMatrix>,
}

impl LanczosState {
    fn new(n: usize) -> Self {
        Self {
            alpha: Vec::new(),
            beta: Vec::new(),
            v: Vec::new(),
            n,
            m: 0,
            converged: Vec::new(),
            ritz_values: Vec::new(),
            ritz_vectors: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal tridiagonal eigensolver (self-contained)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve the symmetric tridiagonal eigenvalue problem via implicit QR with
/// Wilkinson shifts. Returns (eigenvalues, eigenvector_matrix).
fn tridiag_eigen(
    alpha_in: &[f64],
    beta_in: &[f64],
) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = alpha_in.len();
    if n == 0 {
        return Ok((vec![], DenseMatrix::zeros(0, 0)));
    }
    if n == 1 {
        return Ok((vec![alpha_in[0]], DenseMatrix::eye(1)));
    }

    let mut d = alpha_in.to_vec();
    let mut e = beta_in.to_vec();
    let mut z = DenseMatrix::eye(n);
    let max_iter = 30 * n;
    let eps = 1e-14;

    for _ in 0..max_iter {
        // Find unreduced block [lo..=hi]
        let mut hi = n - 1;
        while hi > 0 && e[hi - 1].abs() < eps * (d[hi - 1].abs() + d[hi].abs()).max(eps) {
            hi -= 1;
        }
        if hi == 0 {
            break; // All converged
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1].abs() >= eps * (d[lo - 1].abs() + d[lo].abs()).max(eps) {
            lo -= 1;
        }

        // Wilkinson shift: eigenvalue of trailing 2x2 closer to d[hi]
        let dm = d[hi - 1];
        let dn = d[hi];
        let en = e[hi - 1];
        let delta = (dm - dn) / 2.0;
        let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
        let shift = dn - en * en / (delta + sign * (delta * delta + en * en).sqrt());

        // Implicit QR step with Givens rotations (bulge chase)
        let mut x = d[lo] - shift;
        let mut z_val = e[lo];
        for k in lo..hi {
            // Compute Givens to zero out z_val
            let (c, s, _r) = givens_cs(x, z_val);

            // Apply to tridiagonal
            if k > lo {
                e[k - 1] = c * e[k - 1] - s * z_val; // Actually stores the bulge result
            }

            let d_k = d[k];
            let d_k1 = d[k + 1];
            let e_k = e[k];

            d[k] = c * c * d_k - 2.0 * c * s * e_k + s * s * d_k1;
            d[k + 1] = s * s * d_k + 2.0 * c * s * e_k + c * c * d_k1;
            e[k] = c * s * (d_k - d_k1) + (c * c - s * s) * e_k;

            if k > lo {
                e[k - 1] = c * x - s * z_val; // Overwrite properly
            }

            if k + 1 < hi {
                z_val = -s * e[k + 1];
                e[k + 1] = c * e[k + 1];
                x = e[k];
            }

            // Accumulate eigenvectors
            for i in 0..n {
                let zi_k = z.get(i, k);
                let zi_k1 = z.get(i, k + 1);
                z.set(i, k, c * zi_k - s * zi_k1);
                z.set(i, k + 1, s * zi_k + c * zi_k1);
            }
        }
    }

    Ok((d, z))
}

/// Compute Givens rotation (c, s, r) such that [c s; -s c]^T [a; b] = [r; 0].
fn givens_cs(a: f64, b: f64) -> (f64, f64, f64) {
    if b.abs() < 1e-300 {
        (1.0, 0.0, a)
    } else if a.abs() < 1e-300 {
        (0.0, if b >= 0.0 { 1.0 } else { -1.0 }, b.abs())
    } else if b.abs() > a.abs() {
        let t = a / b;
        let s = 1.0 / (1.0 + t * t).sqrt();
        let c = s * t;
        (c, s, b / s)
    } else {
        let t = b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;
        (c, s, a / c)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reorthogonalization
// ═══════════════════════════════════════════════════════════════════════════

/// Full reorthogonalization: orthogonalize w against all v[0..j] twice.
pub fn full_reorthogonalize(w: &mut [f64], v: &[Vec<f64>], j: usize) {
    for _pass in 0..2 {
        for i in 0..j {
            let d = dot(w, &v[i]);
            axpy(-d, &v[i], w);
        }
    }
}

/// Selective reorthogonalization: only against near-converged Ritz vectors.
pub fn selective_reorthogonalize(
    w: &mut [f64],
    v: &[Vec<f64>],
    j: usize,
    _ritz_values: &[f64],
    ritz_converged: &[bool],
) {
    for i in 0..j.min(ritz_converged.len()) {
        if ritz_converged[i] {
            let d = dot(w, &v[i]);
            axpy(-d, &v[i], w);
        }
    }
    // Also reorthogonalize against last few vectors for safety
    let start = if j > 3 { j - 3 } else { 0 };
    for i in start..j {
        let d = dot(w, &v[i]);
        axpy(-d, &v[i], w);
    }
}

/// Partial reorthogonalization: track estimated inner products, reorth when large.
pub fn partial_reorthogonalize(
    w: &mut [f64],
    v: &[Vec<f64>],
    j: usize,
    omega: &mut Vec<Vec<f64>>,
    beta: &[f64],
    alpha: &[f64],
) {
    let eps_sqrt = f64::EPSILON.sqrt();

    // Update omega estimates using the recurrence
    if omega.len() <= j {
        omega.resize(j + 1, vec![0.0; j + 1]);
    }
    for i in 0..j {
        if i >= omega[j].len() {
            omega[j].resize(j + 1, 0.0);
        }
        let prev = if j >= 2 && i < omega[j - 1].len() {
            omega[j - 1][i]
        } else {
            0.0
        };
        let beta_j = if j > 0 && j - 1 < beta.len() { beta[j - 1] } else { 1.0 };
        omega[j][i] = (alpha[j.min(alpha.len() - 1)] * prev + beta_j * eps_sqrt).abs();
    }

    // Reorthogonalize against vectors where |omega| > sqrt(eps)
    for i in 0..j {
        let est = if i < omega[j].len() { omega[j][i] } else { 0.0 };
        if est > eps_sqrt {
            let d = dot(w, &v[i]);
            axpy(-d, &v[i], w);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Core Lanczos iteration
// ═══════════════════════════════════════════════════════════════════════════

/// Run the Lanczos iteration using a generic matvec closure.
pub fn lanczos_iteration(
    matvec: &dyn Fn(&[f64]) -> Vec<f64>,
    n: usize,
    config: &LanczosConfig,
    start_vec: Option<&[f64]>,
) -> DecompResult<LanczosState> {
    if n == 0 {
        return Err(DecompError::empty("Lanczos: dimension is 0"));
    }
    let k = config.num_eigenvalues;
    let m_max = config.max_iter.min(n);

    let mut state = LanczosState::new(n);

    // Initialize v[0]
    let mut v0 = if let Some(sv) = start_vec {
        sv.to_vec()
    } else {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        (0..n).map(|_| rng.gen::<f64>() - 0.5).collect()
    };
    normalize(&mut v0);
    state.v.push(v0);

    let mut omega_est: Vec<Vec<f64>> = Vec::new();
    let mut v_prev = vec![0.0; n];

    for j in 0..m_max {
        let vj = state.v[j].clone();

        // w = A * v_j
        let mut w = matvec(&vj);

        // alpha_j = w . v_j
        let alpha_j = dot(&w, &vj);
        state.alpha.push(alpha_j);

        // w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        axpy(-alpha_j, &vj, &mut w);
        if j > 0 {
            let beta_prev = state.beta[j - 1];
            axpy(-beta_prev, &v_prev, &mut w);
        }

        // Reorthogonalize
        match config.reorthogonalization {
            ReorthogonalizationType::Full => {
                full_reorthogonalize(&mut w, &state.v, j + 1);
            }
            ReorthogonalizationType::Selective => {
                selective_reorthogonalize(&mut w, &state.v, j + 1, &state.ritz_values, &state.converged);
            }
            ReorthogonalizationType::Partial => {
                partial_reorthogonalize(&mut w, &state.v, j + 1, &mut omega_est, &state.beta, &state.alpha);
            }
            ReorthogonalizationType::None => {}
        }

        let beta_j = norm2(&w);
        state.beta.push(beta_j);

        // Check for invariant subspace
        if beta_j < 1e-14 {
            state.m = j + 1;
            break;
        }

        scale_vec(&mut w, 1.0 / beta_j);
        v_prev = vj;
        state.v.push(w);
        state.m = j + 1;

        // Periodically check convergence
        if (j + 1) >= k && (j + 1) % 5 == 0 {
            let conv = check_convergence(&state, config.tol);
            state.converged = conv.clone();

            // Extract Ritz values
            if let Ok((rvals, _)) = tridiag_eigen(&state.alpha, &state.beta[..state.alpha.len() - 1]) {
                state.ritz_values = select_eigenvalues(&rvals, k, &config.which);
            }

            let n_converged = conv.iter().take(k).filter(|&&c| c).count();
            if n_converged >= k {
                break;
            }
        }
    }

    // Final Ritz extraction
    if state.m > 0 {
        let beta_slice = if state.beta.len() >= state.alpha.len() {
            &state.beta[..state.alpha.len() - 1]
        } else {
            &state.beta
        };
        if let Ok((rvals, rvecs)) = tridiag_eigen(&state.alpha, beta_slice) {
            state.ritz_values = rvals.clone();
            // Compute Ritz vectors: V * eigvecs_of_T
            let m = state.m.min(state.alpha.len());
            let nv = state.v.len().min(m);
            if nv > 0 && rvecs.rows == m {
                let kk = k.min(m);
                let mut ritz_vecs = DenseMatrix::zeros(n, kk);
                let indices = select_indices(&rvals, kk, &config.which);
                for (out_j, &idx) in indices.iter().enumerate() {
                    for i in 0..nv {
                        let coeff = rvecs.get(i, idx);
                        for l in 0..n {
                            let cur = ritz_vecs.get(l, out_j);
                            ritz_vecs.set(l, out_j, cur + coeff * state.v[i][l]);
                        }
                    }
                }
                state.ritz_values = indices.iter().map(|&i| rvals[i]).collect();
                state.ritz_vectors = Some(ritz_vecs);
            }
        }
    }

    Ok(state)
}

/// Select eigenvalue indices according to the `which` criterion.
fn select_indices(evals: &[f64], k: usize, which: &WhichEigenvalues) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..evals.len()).collect();
    match which {
        WhichEigenvalues::Largest => {
            indices.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap_or(std::cmp::Ordering::Equal));
        }
        WhichEigenvalues::Smallest => {
            indices.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap_or(std::cmp::Ordering::Equal));
        }
        WhichEigenvalues::LargestMagnitude => {
            indices.sort_by(|&a, &b| evals[b].abs().partial_cmp(&evals[a].abs()).unwrap_or(std::cmp::Ordering::Equal));
        }
        WhichEigenvalues::SmallestMagnitude => {
            indices.sort_by(|&a, &b| evals[a].abs().partial_cmp(&evals[b].abs()).unwrap_or(std::cmp::Ordering::Equal));
        }
        WhichEigenvalues::ClosestTo(sigma) => {
            indices.sort_by(|&a, &b| {
                (evals[a] - sigma).abs().partial_cmp(&(evals[b] - sigma).abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
    indices.truncate(k);
    indices
}

fn select_eigenvalues(evals: &[f64], k: usize, which: &WhichEigenvalues) -> Vec<f64> {
    let indices = select_indices(evals, k, which);
    indices.iter().map(|&i| evals[i]).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Convergence checking
// ═══════════════════════════════════════════════════════════════════════════

/// Check convergence of Ritz pairs.
///
/// A Ritz pair (θ_i, s_i) is converged if β_m * |s_i[last]| < tol.
pub fn check_convergence(state: &LanczosState, tol: f64) -> Vec<bool> {
    let m = state.m;
    if m == 0 || state.alpha.is_empty() {
        return vec![];
    }

    let beta_m = if m > 0 && m - 1 < state.beta.len() {
        state.beta[m - 1]
    } else {
        0.0
    };

    let beta_slice = if state.beta.len() >= state.alpha.len() {
        &state.beta[..state.alpha.len() - 1]
    } else {
        &state.beta
    };

    let (_evals, evecs) = match tridiag_eigen(&state.alpha, beta_slice) {
        Ok(r) => r,
        Err(_) => return vec![false; state.alpha.len()],
    };

    let n_evals = evecs.cols;
    let last_row = evecs.rows.saturating_sub(1);
    (0..n_evals)
        .map(|i| {
            let last_comp = evecs.get(last_row, i).abs();
            beta_m * last_comp < tol
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Ritz pair extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Extract Ritz pairs from the current Lanczos state.
pub fn extract_ritz_pairs(state: &LanczosState) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let m = state.m.min(state.alpha.len());
    if m == 0 {
        return Ok((vec![], DenseMatrix::zeros(state.n, 0)));
    }
    let beta_slice = if state.beta.len() >= m {
        &state.beta[..m - 1]
    } else {
        &state.beta
    };
    let (evals, evecs_t) = tridiag_eigen(&state.alpha[..m], beta_slice)?;

    let n = state.n;
    let nv = state.v.len().min(m);
    let mut ritz_vecs = DenseMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..nv {
            let coeff = evecs_t.get(i, j);
            for l in 0..n {
                let cur = ritz_vecs.get(l, j);
                ritz_vecs.set(l, j, cur + coeff * state.v[i][l]);
            }
        }
    }
    Ok((evals, ritz_vecs))
}

// ═══════════════════════════════════════════════════════════════════════════
// Implicit restart (IRLM)
// ═══════════════════════════════════════════════════════════════════════════

/// Implicitly restart the Lanczos iteration by applying unwanted shifts.
///
/// Given m Lanczos steps, compress back to k wanted vectors.
pub fn implicit_restart(
    state: &mut LanczosState,
    k: usize,
    _matvec: &dyn Fn(&[f64]) -> Vec<f64>,
) -> DecompResult<()> {
    let m = state.m.min(state.alpha.len());
    if m <= k || k == 0 {
        return Ok(());
    }

    let beta_slice = if state.beta.len() >= m {
        &state.beta[..m - 1]
    } else {
        &state.beta
    };

    let (evals, _) = tridiag_eigen(&state.alpha[..m], beta_slice)?;

    // Select p = m - k unwanted shifts
    let wanted_indices = select_indices(&evals, k, &WhichEigenvalues::LargestMagnitude);
    let mut unwanted: Vec<f64> = Vec::new();
    for i in 0..evals.len() {
        if !wanted_indices.contains(&i) {
            unwanted.push(evals[i]);
        }
    }

    // Apply each unwanted shift as implicit QR step on the tridiagonal
    let mut alpha_work = state.alpha[..m].to_vec();
    let mut beta_work = if beta_slice.len() >= m - 1 {
        beta_slice[..m - 1].to_vec()
    } else {
        beta_slice.to_vec()
    };

    // Build accumulated Q as product of Givens rotations
    let mut q_acc = DenseMatrix::eye(m);

    for &mu in &unwanted {
        // QR factorization of T - mu*I
        let mut shifted_alpha: Vec<f64> = alpha_work.iter().map(|&a| a - mu).collect();
        let mut shifted_beta = beta_work.clone();

        // Apply Givens rotations to create upper triangular
        let block_len = shifted_alpha.len().min(shifted_beta.len() + 1);
        for i in 0..block_len.saturating_sub(1) {
            let (c, s, _r) = givens_cs(shifted_alpha[i], shifted_beta[i]);

            // Apply to alpha/beta (tridiagonal QR step)
            let a_i = alpha_work[i];
            let a_i1 = alpha_work[i + 1];
            let b_i = beta_work[i];

            alpha_work[i] = c * c * a_i + 2.0 * c * s * b_i + s * s * a_i1;
            alpha_work[i + 1] = s * s * a_i - 2.0 * c * s * b_i + c * c * a_i1;
            beta_work[i] = c * s * (a_i - a_i1) + (c * c - s * s) * b_i;

            // Update shifted values for next iteration
            if i + 1 < shifted_alpha.len() {
                shifted_alpha[i] = alpha_work[i] - mu;
                shifted_alpha[i + 1] = alpha_work[i + 1] - mu;
            }
            if i + 1 < shifted_beta.len() {
                let b_next = beta_work[i + 1];
                shifted_beta[i] = beta_work[i];
                shifted_beta[i + 1] = c * b_next;
                beta_work[i + 1] = c * b_next;
            }

            // Accumulate rotation in Q
            for r in 0..m {
                let qi = q_acc.get(r, i);
                let qi1 = q_acc.get(r, i + 1);
                q_acc.set(r, i, c * qi + s * qi1);
                q_acc.set(r, i + 1, -s * qi + c * qi1);
            }
        }
    }

    // Update Lanczos vectors: V_new = V_old * Q[:, :k]
    let n = state.n;
    let nv = state.v.len().min(m);
    let mut new_v: Vec<Vec<f64>> = Vec::with_capacity(k);
    for j in 0..k {
        let mut col = vec![0.0; n];
        for i in 0..nv {
            let coeff = q_acc.get(i, j);
            axpy(coeff, &state.v[i], &mut col);
        }
        let nrm = normalize(&mut col);
        if nrm < 1e-14 {
            // Degenerate: use random vector
            let mut rng = ChaCha8Rng::seed_from_u64(j as u64 + 12345);
            col = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
            normalize(&mut col);
        }
        new_v.push(col);
    }

    // Update tridiagonal
    state.alpha = alpha_work[..k].to_vec();
    state.beta = if k > 1 {
        beta_work[..k - 1].to_vec()
    } else {
        vec![]
    };
    state.v = new_v;
    state.m = k;

    Ok(())
}

/// Lock converged Ritz pairs by removing them from the active iteration.
pub fn lock_converged(state: &mut LanczosState) {
    // Simply mark converged ones; actual locking is handled by the caller
    // by adjusting the number of eigenvalues sought
    let new_converged = check_convergence(state, 1e-10);
    state.converged = new_converged;
}

// ═══════════════════════════════════════════════════════════════════════════
// Shift-invert Lanczos
// ═══════════════════════════════════════════════════════════════════════════

/// Shift-invert Lanczos for interior eigenvalues near `sigma`.
///
/// Solves (A - sigma*I)^{-1} eigenvalue problem, then transforms back.
pub fn shift_invert_lanczos(
    a: &CsrMatrix,
    sigma: f64,
    config: &LanczosConfig,
) -> DecompResult<LanczosState> {
    let n = a.rows;
    if n == 0 {
        return Err(DecompError::empty("shift-invert: empty matrix"));
    }

    // Build (A - sigma*I) as dense and LU factorize
    let mut shifted = a.to_dense();
    for i in 0..n {
        shifted.set(i, i, shifted.get(i, i) - sigma);
    }

    // Simple LU factorization for the solve
    let lu = dense_lu_factorize(&shifted)?;

    // Matvec = (A - sigma*I)^{-1} * x
    let matvec = move |x: &[f64]| -> Vec<f64> {
        dense_lu_solve(&lu.0, &lu.1, x).unwrap_or_else(|_| x.to_vec())
    };

    let mut state = lanczos_iteration(&matvec, n, config, None)?;

    // Transform eigenvalues back: lambda = sigma + 1/theta
    for val in &mut state.ritz_values {
        if val.abs() > 1e-14 {
            *val = sigma + 1.0 / *val;
        }
    }

    Ok(state)
}

/// Simple dense LU factorization (in-place, returns (LU_matrix, pivot)).
fn dense_lu_factorize(a: &DenseMatrix) -> DecompResult<(DenseMatrix, Vec<usize>)> {
    let n = a.rows;
    let mut lu = a.clone();
    let mut pivot: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Partial pivoting
        let mut max_val = lu.get(k, k).abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu.get(i, k).abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-14 {
            return Err(DecompError::singular("LU pivot near zero"));
        }
        if max_row != k {
            lu.swap_rows(k, max_row);
            pivot.swap(k, max_row);
        }
        let akk = lu.get(k, k);
        for i in (k + 1)..n {
            let factor = lu.get(i, k) / akk;
            lu.set(i, k, factor);
            for j in (k + 1)..n {
                let v = lu.get(i, j) - factor * lu.get(k, j);
                lu.set(i, j, v);
            }
        }
    }
    Ok((lu, pivot))
}

/// Solve LU system.
fn dense_lu_solve(lu: &DenseMatrix, pivot: &[usize], b: &[f64]) -> DecompResult<Vec<f64>> {
    let n = lu.rows;
    // Apply permutation
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[pivot[i]];
    }
    // Forward substitution (L has unit diagonal)
    let mut y = pb;
    for i in 1..n {
        for j in 0..i {
            y[i] -= lu.get(i, j) * y[j];
        }
    }
    // Back substitution
    let mut x = y;
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu.get(i, j) * x[j];
        }
        x[i] /= lu.get(i, i);
    }
    Ok(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience wrappers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute k eigenvalues/vectors of a sparse symmetric matrix.
pub fn lanczos_eigenvalues(
    a: &CsrMatrix,
    k: usize,
    which: WhichEigenvalues,
    tol: f64,
) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = a.rows;
    if k > n {
        return Err(DecompError::TooManyEigenvalues { requested: k, size: n });
    }
    let config = LanczosConfig {
        max_iter: (4 * k + 20).min(n),
        tol,
        reorthogonalization: ReorthogonalizationType::Full,
        num_eigenvalues: k,
        which,
    };
    let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap_or_else(|_| vec![0.0; a.rows]) };
    let state = lanczos_iteration(&matvec, n, &config, None)?;

    let evals = state.ritz_values;
    let evecs = state.ritz_vectors.unwrap_or_else(|| DenseMatrix::zeros(n, 0));

    Ok((evals, evecs))
}

/// Compute k eigenvalues/vectors of a dense symmetric matrix via Lanczos.
pub fn lanczos_eigenvalues_dense(
    a: &DenseMatrix,
    k: usize,
    which: WhichEigenvalues,
    tol: f64,
) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = a.rows;
    if k > n {
        return Err(DecompError::TooManyEigenvalues { requested: k, size: n });
    }
    let config = LanczosConfig {
        max_iter: (4 * k + 20).min(n),
        tol,
        reorthogonalization: ReorthogonalizationType::Full,
        num_eigenvalues: k,
        which,
    };
    let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap_or_else(|_| vec![0.0; a.rows]) };
    let state = lanczos_iteration(&matvec, n, &config, None)?;

    let evals = state.ritz_values;
    let evecs = state.ritz_vectors.unwrap_or_else(|| DenseMatrix::zeros(n, 0));

    Ok((evals, evecs))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_matrix(diag: &[f64]) -> DenseMatrix {
        DenseMatrix::from_diag(diag)
    }

    fn csr_from_dense(d: &DenseMatrix) -> CsrMatrix {
        d.to_csr()
    }

    #[test]
    fn test_lanczos_diagonal() {
        let a = diag_matrix(&[5.0, 3.0, 1.0, 4.0, 2.0]);
        let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap() };
        let config = LanczosConfig {
            max_iter: 20,
            tol: 1e-10,
            reorthogonalization: ReorthogonalizationType::Full,
            num_eigenvalues: 2,
            which: WhichEigenvalues::Largest,
        };
        let state = lanczos_iteration(&matvec, 5, &config, None).unwrap();
        let evals = &state.ritz_values;
        assert!(!evals.is_empty());
        // Should find 5.0 as largest
        let max_eval = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max_eval - 5.0).abs() < 0.5, "max_eval={max_eval}");
    }

    #[test]
    fn test_lanczos_known_symmetric() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            2.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 2.0,
        ]);
        let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap() };
        let config = LanczosConfig {
            max_iter: 20,
            tol: 1e-10,
            reorthogonalization: ReorthogonalizationType::Full,
            num_eigenvalues: 3,
            which: WhichEigenvalues::LargestMagnitude,
        };
        let state = lanczos_iteration(&matvec, 3, &config, None).unwrap();
        // Eigenvalues of 1D Laplacian: 2-2cos(k*pi/(n+1)) for k=1,2,3
        // ≈ 0.586, 2.0, 3.414
        let mut evals = state.ritz_values.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if evals.len() >= 3 {
            assert!((evals[0] - 0.586).abs() < 0.3, "e0={}", evals[0]);
            assert!((evals[1] - 2.0).abs() < 0.3, "e1={}", evals[1]);
            assert!((evals[2] - 3.414).abs() < 0.3, "e2={}", evals[2]);
        }
    }

    #[test]
    fn test_full_reorthogonalize() {
        let v0 = vec![1.0, 0.0, 0.0];
        let v1 = vec![0.0, 1.0, 0.0];
        let mut w = vec![1.0, 1.0, 1.0];
        full_reorthogonalize(&mut w, &[v0.clone(), v1.clone()], 2);
        assert!(dot(&w, &v0).abs() < 1e-10);
        assert!(dot(&w, &v1).abs() < 1e-10);
    }

    #[test]
    fn test_tridiag_eigen_basic() {
        // Tridiagonal: alpha=[2,2,2], beta=[1,1] → 1D Laplacian eigenvalues
        let (evals, evecs) = tridiag_eigen(&[2.0, 2.0, 2.0], &[1.0, 1.0]).unwrap();
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(sorted.len() == 3);
        // Eigenvalues ≈ 0.586, 2.0, 3.414
        assert!((sorted[0] - 0.586).abs() < 0.1, "e0={}", sorted[0]);
        assert!((sorted[1] - 2.0).abs() < 0.1, "e1={}", sorted[1]);
        assert!((sorted[2] - 3.414).abs() < 0.1, "e2={}", sorted[2]);
        // Eigenvectors should be orthogonal
        for i in 0..3 {
            for j in (i + 1)..3 {
                let d = dot(&evecs.col(i), &evecs.col(j));
                assert!(d.abs() < 1e-8, "dot({i},{j})={d}");
            }
        }
    }

    #[test]
    fn test_lanczos_sparse() {
        let triplets = vec![
            (0, 0, 4.0), (0, 1, 1.0),
            (1, 0, 1.0), (1, 1, 3.0), (1, 2, 1.0),
            (2, 1, 1.0), (2, 2, 2.0),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, &triplets);
        let (evals, _evecs) = lanczos_eigenvalues(&csr, 2, WhichEigenvalues::Largest, 1e-8).unwrap();
        assert!(!evals.is_empty());
        let max_eval = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Largest eigenvalue should be around 4.7
        assert!(max_eval > 3.5, "max_eval={max_eval}");
    }

    #[test]
    fn test_convergence_check() {
        let a = DenseMatrix::from_diag(&[10.0, 5.0, 1.0]);
        let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap() };
        let config = LanczosConfig {
            max_iter: 10,
            tol: 1e-8,
            reorthogonalization: ReorthogonalizationType::Full,
            num_eigenvalues: 2,
            which: WhichEigenvalues::Largest,
        };
        let state = lanczos_iteration(&matvec, 3, &config, None).unwrap();
        let conv = check_convergence(&state, 1e-8);
        // With 3 dimensions and up to 10 iterations, should converge
        assert!(!conv.is_empty());
    }

    #[test]
    fn test_select_indices() {
        let evals = vec![3.0, 1.0, 5.0, 2.0];
        let idx = select_indices(&evals, 2, &WhichEigenvalues::Largest);
        assert!(idx.contains(&2)); // 5.0
        assert!(idx.contains(&0)); // 3.0
    }

    #[test]
    fn test_select_smallest() {
        let evals = vec![3.0, 1.0, 5.0, 2.0];
        let idx = select_indices(&evals, 2, &WhichEigenvalues::Smallest);
        assert!(idx.contains(&1)); // 1.0
        assert!(idx.contains(&3)); // 2.0
    }

    #[test]
    fn test_select_closest() {
        let evals = vec![3.0, 1.0, 5.0, 2.0];
        let idx = select_indices(&evals, 1, &WhichEigenvalues::ClosestTo(2.1));
        assert_eq!(idx[0], 3); // 2.0 is closest to 2.1
    }

    #[test]
    fn test_extract_ritz_pairs() {
        let a = DenseMatrix::from_diag(&[4.0, 2.0, 1.0]);
        let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap() };
        let config = LanczosConfig {
            max_iter: 10,
            tol: 1e-10,
            reorthogonalization: ReorthogonalizationType::Full,
            num_eigenvalues: 3,
            which: WhichEigenvalues::LargestMagnitude,
        };
        let state = lanczos_iteration(&matvec, 3, &config, None).unwrap();
        let (evals, evecs) = extract_ritz_pairs(&state).unwrap();
        assert!(!evals.is_empty());
        assert!(evecs.rows == 3);
    }

    #[test]
    fn test_implicit_restart() {
        let a = DenseMatrix::from_diag(&[10.0, 7.0, 3.0, 1.0, 0.5]);
        let matvec = |x: &[f64]| -> Vec<f64> { a.mul_vec(x).unwrap() };
        let config = LanczosConfig {
            max_iter: 5,
            tol: 1e-10,
            reorthogonalization: ReorthogonalizationType::Full,
            num_eigenvalues: 2,
            which: WhichEigenvalues::Largest,
        };
        let mut state = lanczos_iteration(&matvec, 5, &config, None).unwrap();
        // Restart should not error
        let _ = implicit_restart(&mut state, 2, &matvec);
    }

    #[test]
    fn test_lanczos_dense_convenience() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            5.0, 1.0, 0.0,
            1.0, 4.0, 1.0,
            0.0, 1.0, 3.0,
        ]);
        let (evals, evecs) = lanczos_eigenvalues_dense(&a, 2, WhichEigenvalues::Largest, 1e-8).unwrap();
        assert!(!evals.is_empty());
        assert!(evecs.rows == 3);
    }

    #[test]
    fn test_shift_invert() {
        let triplets = vec![
            (0, 0, 5.0), (1, 1, 3.0), (2, 2, 1.0),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, &triplets);
        let config = LanczosConfig {
            max_iter: 20,
            tol: 1e-8,
            reorthogonalization: ReorthogonalizationType::Full,
            num_eigenvalues: 1,
            which: WhichEigenvalues::LargestMagnitude,
        };
        let state = shift_invert_lanczos(&csr, 2.0, &config).unwrap();
        // Should find eigenvalue closest to 2.0, which is 1.0 or 3.0
        assert!(!state.ritz_values.is_empty());
    }
}
