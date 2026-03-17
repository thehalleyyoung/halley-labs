//! Householder transformations for matrix decompositions.
//!
//! Provides the [`HouseholderReflector`] type for computing and applying
//! Householder reflections, a [`BlockHouseholder`] accumulator (WY
//! representation) for efficient block application, and higher-level routines
//! for bidiagonalization, Hessenberg reduction, and explicit Q formation.

use crate::{dot, norm2, DenseMatrix};

// ═══════════════════════════════════════════════════════════════════════════
// Householder reflector
// ═══════════════════════════════════════════════════════════════════════════

/// A Householder reflector H = I - τ v vᵀ.
///
/// Applying H to a vector x maps it to a multiple of e₁ (the first
/// standard basis vector):  H x = ±‖x‖ e₁.
#[derive(Clone, Debug)]
pub struct HouseholderReflector {
    /// The Householder vector.
    pub v: Vec<f64>,
    /// Scalar τ = 2 / (vᵀ v).
    pub tau: f64,
}

impl HouseholderReflector {
    /// Generate a Householder reflector that maps `x` to ‖x‖ e₁.
    ///
    /// Uses the numerically stable convention:
    ///   v = x.clone();  v[0] += sign(x[0]) * ‖x‖;  τ = 2 / (vᵀ v)
    ///
    /// When x is zero the reflector is the identity (τ = 0).
    pub fn generate(x: &[f64]) -> Self {
        let n = x.len();
        if n == 0 {
            return Self {
                v: Vec::new(),
                tau: 0.0,
            };
        }

        let mut v = x.to_vec();
        let x_norm = norm2(x);

        if x_norm < 1e-300 {
            // x is essentially zero – identity reflector
            return Self { v, tau: 0.0 };
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * x_norm;

        let vv = dot(&v, &v);
        let tau = if vv.abs() < 1e-300 { 0.0 } else { 2.0 / vv };

        Self { v, tau }
    }

    /// Apply H = I − τ v vᵀ to a sub-block of `mat` **from the left**.
    ///
    /// Operates on `mat[row_start .. row_start+nrows, col_start .. col_start+ncols]`.
    /// `v` must have length `nrows`.
    ///
    /// For each column j in the block:
    ///   w_j = Σᵢ v[i] · A[row_start+i, col_start+j]
    ///   A[row_start+i, col_start+j] −= τ · v[i] · w_j
    pub fn apply_left(
        v: &[f64],
        tau: f64,
        mat: &mut DenseMatrix,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) {
        if tau.abs() < 1e-300 || nrows == 0 || ncols == 0 {
            return;
        }
        debug_assert!(
            v.len() >= nrows,
            "apply_left: v.len()={} < nrows={}",
            v.len(),
            nrows
        );

        for j in 0..ncols {
            // w_j = vᵀ · A[:, j]
            let mut w = 0.0;
            for i in 0..nrows {
                w += v[i] * mat.get(row_start + i, col_start + j);
            }
            // A[:, j] -= τ * w * v
            let tw = tau * w;
            for i in 0..nrows {
                let cur = mat.get(row_start + i, col_start + j);
                mat.set(row_start + i, col_start + j, cur - tw * v[i]);
            }
        }
    }

    /// Apply H = I − τ v vᵀ to a sub-block of `mat` **from the right**.
    ///
    /// Operates on `mat[row_start .. row_start+nrows, col_start .. col_start+ncols]`.
    /// `v` must have length `ncols`.
    ///
    /// For each row i in the block:
    ///   w_i = Σⱼ A[row_start+i, col_start+j] · v[j]
    ///   A[row_start+i, col_start+j] −= τ · w_i · v[j]
    pub fn apply_right(
        v: &[f64],
        tau: f64,
        mat: &mut DenseMatrix,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) {
        if tau.abs() < 1e-300 || nrows == 0 || ncols == 0 {
            return;
        }
        debug_assert!(
            v.len() >= ncols,
            "apply_right: v.len()={} < ncols={}",
            v.len(),
            ncols
        );

        for i in 0..nrows {
            // w_i = A[i, :] · v
            let mut w = 0.0;
            for j in 0..ncols {
                w += mat.get(row_start + i, col_start + j) * v[j];
            }
            // A[i, :] -= τ * w * v
            let tw = tau * w;
            for j in 0..ncols {
                let cur = mat.get(row_start + i, col_start + j);
                mat.set(row_start + i, col_start + j, cur - tw * v[j]);
            }
        }
    }

    /// Convenience: apply this reflector from the left.
    pub fn apply_left_self(
        &self,
        mat: &mut DenseMatrix,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) {
        Self::apply_left(&self.v, self.tau, mat, row_start, col_start, nrows, ncols);
    }

    /// Convenience: apply this reflector from the right.
    pub fn apply_right_self(
        &self,
        mat: &mut DenseMatrix,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) {
        Self::apply_right(&self.v, self.tau, mat, row_start, col_start, nrows, ncols);
    }

    /// Apply H to a standalone vector: y = (I − τ v vᵀ) x.
    pub fn apply_to_vec(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut y = x.to_vec();
        if self.tau.abs() < 1e-300 || n == 0 {
            return y;
        }
        let d = dot(&self.v[..n], x);
        let td = self.tau * d;
        for i in 0..n {
            y[i] -= td * self.v[i];
        }
        y
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Block Householder (WY representation)
// ═══════════════════════════════════════════════════════════════════════════

/// Block Householder accumulator using the compact WY representation.
///
/// Stores the product  Q = I − W Yᵀ  where W and Y are built incrementally
/// so that adding a new reflector Hₖ = I − τₖ vₖ vₖᵀ extends the product:
///   Q_new = Q_old · Hₖ
///
/// The WY representation allows applying the accumulated product to a matrix
/// in one BLAS-3-like operation instead of one reflector at a time.
#[derive(Clone, Debug)]
pub struct BlockHouseholder {
    /// Dimension of the reflectors (length of each v).
    pub dim: usize,
    /// Number of accumulated reflectors.
    pub count: usize,
    /// The W factor  (dim × count), stored column-major in a `DenseMatrix`.
    pub w: DenseMatrix,
    /// The Y factor  (dim × count), stored as a `DenseMatrix` (Y columns = v_k modified).
    pub y: DenseMatrix,
    /// Capacity (max reflectors before re-allocation).
    capacity: usize,
}

impl BlockHouseholder {
    /// Create a new block accumulator for reflectors of length `dim`,
    /// with room for up to `capacity` reflectors.
    pub fn new(dim: usize, capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            dim,
            count: 0,
            w: DenseMatrix::zeros(dim, cap),
            y: DenseMatrix::zeros(dim, cap),
            capacity: cap,
        }
    }

    /// Accumulate a new reflector Hₖ = I − τ v vᵀ.
    ///
    /// Updates  Q ← Q · Hₖ  by extending the WY factors:
    ///   z  = τ (v − W (Yᵀ v))
    ///   W  = [W | z]
    ///   Y  = [Y | v]
    pub fn accumulate(&mut self, v: &[f64], tau: f64) {
        assert_eq!(v.len(), self.dim, "reflector dimension mismatch");

        // Grow storage if needed
        if self.count >= self.capacity {
            let new_cap = self.capacity * 2;
            let mut new_w = DenseMatrix::zeros(self.dim, new_cap);
            let mut new_y = DenseMatrix::zeros(self.dim, new_cap);
            for c in 0..self.count {
                for r in 0..self.dim {
                    new_w.set(r, c, self.w.get(r, c));
                    new_y.set(r, c, self.y.get(r, c));
                }
            }
            self.w = new_w;
            self.y = new_y;
            self.capacity = new_cap;
        }

        let k = self.count;

        if k == 0 {
            // First reflector: W[:,0] = τ v,  Y[:,0] = v
            for i in 0..self.dim {
                self.w.set(i, 0, tau * v[i]);
                self.y.set(i, 0, v[i]);
            }
        } else {
            // Compute p = Yᵀ v  (k-vector)
            let mut p = vec![0.0; k];
            for j in 0..k {
                let mut s = 0.0;
                for i in 0..self.dim {
                    s += self.y.get(i, j) * v[i];
                }
                p[j] = s;
            }

            // z = τ * (v − W p)
            let mut z = vec![0.0; self.dim];
            for i in 0..self.dim {
                let mut wp_i = 0.0;
                for j in 0..k {
                    wp_i += self.w.get(i, j) * p[j];
                }
                z[i] = tau * (v[i] - wp_i);
            }

            // Store
            for i in 0..self.dim {
                self.w.set(i, k, z[i]);
                self.y.set(i, k, v[i]);
            }
        }

        self.count += 1;
    }

    /// Apply the accumulated block reflector from the left:
    ///   A ← (I − W Yᵀ) A
    ///
    /// Operates on `mat[row_start .. row_start+nrows, col_start .. col_start+ncols]`
    /// where `nrows == self.dim`.
    pub fn apply_left_block(
        &self,
        mat: &mut DenseMatrix,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) {
        if self.count == 0 || nrows == 0 || ncols == 0 {
            return;
        }
        assert!(
            nrows <= self.dim,
            "apply_left_block: nrows={} > dim={}",
            nrows,
            self.dim
        );

        // Compute Tmp = Wᵀ A   (count × ncols)
        let k = self.count;
        let mut tmp = vec![0.0; k * ncols];
        for c in 0..k {
            for j in 0..ncols {
                let mut s = 0.0;
                for i in 0..nrows {
                    s += self.w.get(i, c) * mat.get(row_start + i, col_start + j);
                }
                tmp[c * ncols + j] = s;
            }
        }

        // A -= Y · Tmp
        for i in 0..nrows {
            for j in 0..ncols {
                let mut s = 0.0;
                for c in 0..k {
                    s += self.y.get(i, c) * tmp[c * ncols + j];
                }
                let cur = mat.get(row_start + i, col_start + j);
                mat.set(row_start + i, col_start + j, cur - s);
            }
        }
    }

    /// Apply the accumulated block reflector from the right:
    ///   A ← A (I − W Yᵀ)ᵀ = A − A Y Wᵀ
    ///
    /// Note: (I − W Yᵀ)ᵀ = I − Y Wᵀ, so A ← A − (A Y) Wᵀ.
    ///
    /// Operates on `mat[row_start .. row_start+nrows, col_start .. col_start+ncols]`
    /// where `ncols == self.dim`.
    pub fn apply_right_block(
        &self,
        mat: &mut DenseMatrix,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) {
        if self.count == 0 || nrows == 0 || ncols == 0 {
            return;
        }
        assert!(
            ncols <= self.dim,
            "apply_right_block: ncols={} > dim={}",
            ncols,
            self.dim
        );

        // Compute Tmp = A Y   (nrows × count)
        let k = self.count;
        let mut tmp = vec![0.0; nrows * k];
        for i in 0..nrows {
            for c in 0..k {
                let mut s = 0.0;
                for j in 0..ncols {
                    s += mat.get(row_start + i, col_start + j) * self.y.get(j, c);
                }
                tmp[i * k + c] = s;
            }
        }

        // A -= Tmp Wᵀ   i.e.  A[i,j] -= Σ_c  tmp[i,c] * W[j,c]
        for i in 0..nrows {
            for j in 0..ncols {
                let mut s = 0.0;
                for c in 0..k {
                    s += tmp[i * k + c] * self.w.get(j, c);
                }
                let cur = mat.get(row_start + i, col_start + j);
                mat.set(row_start + i, col_start + j, cur - s);
            }
        }
    }

    /// Reset the accumulator to empty (reuse allocated storage).
    pub fn reset(&mut self) {
        self.count = 0;
        // Zero out is not strictly necessary because `count` guards access,
        // but do it for safety.
        for v in self.w.data.iter_mut() {
            *v = 0.0;
        }
        for v in self.y.data.iter_mut() {
            *v = 0.0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bidiagonalization
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Golub–Kahan bidiagonalization A = U B Vᵀ.
#[derive(Clone, Debug)]
pub struct BidiagResult {
    /// Left orthogonal factor (m × m).
    pub u: DenseMatrix,
    /// Right orthogonal factor (n × n).
    pub vt: DenseMatrix,
    /// Main diagonal of the bidiagonal matrix.
    pub diag: Vec<f64>,
    /// Super-diagonal of the bidiagonal matrix (length min(m,n)−1).
    pub superdiag: Vec<f64>,
    /// Left Householder reflectors (v, τ) used during reduction.
    pub left_reflectors: Vec<(Vec<f64>, f64)>,
    /// Right Householder reflectors (v, τ) used during reduction.
    pub right_reflectors: Vec<(Vec<f64>, f64)>,
}

/// Reduce an m×n matrix A to upper bidiagonal form using Householder
/// reflections from the left and right.
///
/// Returns `(U, V, diag, superdiag, left_reflectors, right_reflectors)`.
///
/// The bidiagonal matrix B satisfies  B = Uᵀ A V  and has the main
/// diagonal `diag` and super-diagonal `superdiag`.
pub fn bidiagonalize(
    a: &DenseMatrix,
) -> (
    DenseMatrix,
    DenseMatrix,
    Vec<f64>,
    Vec<f64>,
    Vec<(Vec<f64>, f64)>,
    Vec<(Vec<f64>, f64)>,
) {
    let m = a.rows;
    let n = a.cols;
    let mut work = a.clone();

    let mut left_refs: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut right_refs: Vec<(Vec<f64>, f64)> = Vec::new();

    let k = m.min(n);

    for j in 0..k {
        // --- Left Householder: zero out below diagonal in column j ---
        if j < m {
            let col_len = m - j;
            let mut x = vec![0.0; col_len];
            for i in 0..col_len {
                x[i] = work.get(j + i, j);
            }
            let h = HouseholderReflector::generate(&x);
            // Apply H from the left to work[j:m, j:n]
            HouseholderReflector::apply_left(
                &h.v,
                h.tau,
                &mut work,
                j,
                j,
                col_len,
                n - j,
            );
            left_refs.push((h.v, h.tau));
        }

        // --- Right Householder: zero out beyond super-diagonal in row j ---
        if j + 1 < n {
            let row_len = n - (j + 1);
            if row_len > 0 {
                let mut x = vec![0.0; row_len];
                for i in 0..row_len {
                    x[i] = work.get(j, j + 1 + i);
                }
                let h = HouseholderReflector::generate(&x);
                // Apply H from the right to work[j:m, j+1:n]
                HouseholderReflector::apply_right(
                    &h.v,
                    h.tau,
                    &mut work,
                    j,
                    j + 1,
                    m - j,
                    row_len,
                );
                right_refs.push((h.v, h.tau));
            }
        }
    }

    // Extract diagonal and super-diagonal
    let diag_len = k;
    let mut diag = vec![0.0; diag_len];
    for i in 0..diag_len {
        diag[i] = work.get(i, i);
    }
    let sup_len = if k > 1 { k - 1 } else { 0 }.min(n.saturating_sub(1));
    let mut superdiag = vec![0.0; sup_len];
    for i in 0..sup_len {
        superdiag[i] = work.get(i, i + 1);
    }

    // Form U by accumulating left reflectors onto I_m
    let u = form_q_left(m, m, &left_refs);

    // Form V by accumulating right reflectors onto I_n
    let vt_mat = form_q_right(n, &right_refs);

    (u, vt_mat, diag, superdiag, left_refs, right_refs)
}

/// Form Q from left bidiagonalization reflectors.
///
/// Each reflector k has length (m − k) and was applied starting at row k.
fn form_q_left(
    m: usize,
    _ncols: usize,
    reflectors: &[(Vec<f64>, f64)],
) -> DenseMatrix {
    let mut q = DenseMatrix::eye(m);
    // Apply in reverse order so that Q = H_0 H_1 ... H_{k-1}
    for (k, (v, tau)) in reflectors.iter().enumerate().rev() {
        let nrows = v.len();
        HouseholderReflector::apply_left(v, *tau, &mut q, k, 0, nrows, m);
    }
    q
}

/// Form V from right bidiagonalization reflectors.
///
/// Right reflector k has length (n − k − 1) and was applied starting at column k+1.
fn form_q_right(n: usize, reflectors: &[(Vec<f64>, f64)]) -> DenseMatrix {
    let mut v = DenseMatrix::eye(n);
    for (k, (ref vec, tau)) in reflectors.iter().enumerate() {
        let ncols = vec.len();
        let col_start = k + 1;
        HouseholderReflector::apply_right(vec, *tau, &mut v, 0, col_start, n, ncols);
    }
    v
}

// ═══════════════════════════════════════════════════════════════════════════
// Hessenberg reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce a square matrix to upper Hessenberg form in-place using
/// Householder similarity transforms.
///
/// For column k = 0, 1, …, n−3 the routine computes a Householder
/// reflector Hₖ that zeros `A[k+2:n, k]`, then applies the similarity
/// transform  A ← Hₖ A Hₖ  (left then right).
///
/// Returns the sequence of reflectors `(v, τ)`.  The length of `v` in
/// step k is `n − k − 1`.
pub fn reduce_to_hessenberg(a: &mut DenseMatrix) -> Vec<(Vec<f64>, f64)> {
    let n = a.rows;
    assert!(a.is_square(), "Hessenberg reduction requires a square matrix");
    if n <= 2 {
        return Vec::new();
    }

    let mut reflectors: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n - 2);

    for k in 0..n - 2 {
        let len = n - k - 1;
        let mut x = vec![0.0; len];
        for i in 0..len {
            x[i] = a.get(k + 1 + i, k);
        }

        let h = HouseholderReflector::generate(&x);

        // Apply from left:  A[k+1:n, k:n] = H * A[k+1:n, k:n]
        HouseholderReflector::apply_left(&h.v, h.tau, a, k + 1, k, len, n - k);

        // Apply from right: A[0:n, k+1:n] = A[0:n, k+1:n] * H
        HouseholderReflector::apply_right(&h.v, h.tau, a, 0, k + 1, n, len);

        reflectors.push((h.v, h.tau));
    }

    reflectors
}

// ═══════════════════════════════════════════════════════════════════════════
// Explicit Q formation
// ═══════════════════════════════════════════════════════════════════════════

/// Explicitly form the orthogonal matrix Q from a sequence of Householder
/// reflectors.
///
/// The resulting Q has shape `m × n`.  It is built by accumulating the
/// reflectors in **reverse order** onto the identity matrix.
///
/// The k-th reflector `(v, τ)` is applied from the left starting at row k,
/// column 0 with the reflector spanning `v.len()` rows:
///   Q ← Hₖ Q        where  Hₖ = I − τ v vᵀ.
///
/// This matches the convention used by QR and bidiagonalization left
/// reflectors.
pub fn form_q_from_reflectors(
    m: usize,
    n: usize,
    reflectors: &[(Vec<f64>, f64)],
) -> DenseMatrix {
    let mut q = DenseMatrix::zeros(m, n);
    // Start from I (only the overlapping part)
    let diag_len = m.min(n);
    for i in 0..diag_len {
        q.set(i, i, 1.0);
    }

    // Apply in reverse so that Q = H_0 H_1 ... H_{k-1}
    for (k, (v, tau)) in reflectors.iter().enumerate().rev() {
        let nrows = v.len();
        HouseholderReflector::apply_left(v, *tau, &mut q, k, 0, nrows, n);
    }

    q
}

/// Form Q from Hessenberg reflectors.
///
/// Hessenberg reflector k has length `n − k − 1` and acts on rows
/// `k+1 .. n`.
pub fn form_q_from_hessenberg_reflectors(
    n: usize,
    reflectors: &[(Vec<f64>, f64)],
) -> DenseMatrix {
    let mut q = DenseMatrix::eye(n);

    for (k, (v, tau)) in reflectors.iter().enumerate().rev() {
        let nrows = v.len();
        HouseholderReflector::apply_left(v, *tau, &mut q, k + 1, 0, nrows, n);
    }

    q
}

// ═══════════════════════════════════════════════════════════════════════════
// QR factorisation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the thin QR factorization A = Q R using Householder reflectors.
///
/// Returns `(reflectors, R)` where `reflectors` is the sequence of (v, τ)
/// pairs and R is the upper triangular part stored in-place in the
/// returned matrix.
pub fn qr_householder(a: &DenseMatrix) -> (Vec<(Vec<f64>, f64)>, DenseMatrix) {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let k = m.min(n);
    let mut reflectors: Vec<(Vec<f64>, f64)> = Vec::with_capacity(k);

    for j in 0..k {
        let col_len = m - j;
        let mut x = vec![0.0; col_len];
        for i in 0..col_len {
            x[i] = r.get(j + i, j);
        }

        let h = HouseholderReflector::generate(&x);

        // Apply from left to R[j:m, j:n]
        HouseholderReflector::apply_left(&h.v, h.tau, &mut r, j, j, col_len, n - j);

        reflectors.push((h.v, h.tau));
    }

    (reflectors, r)
}

/// Extract the upper triangular matrix R from a matrix that has been
/// transformed in-place by `qr_householder`.
pub fn extract_r(qr: &DenseMatrix) -> DenseMatrix {
    let m = qr.rows;
    let n = qr.cols;
    let k = m.min(n);
    let mut r = DenseMatrix::zeros(k, n);
    for i in 0..k {
        for j in i..n {
            r.set(i, j, qr.get(i, j));
        }
    }
    r
}

// ═══════════════════════════════════════════════════════════════════════════
// Blocked QR (uses BlockHouseholder)
// ═══════════════════════════════════════════════════════════════════════════

/// Blocked QR factorization using the WY block-Householder representation.
///
/// Processes `block_size` columns at a time and flushes via
/// `BlockHouseholder::apply_left_block` for better cache performance.
///
/// Returns `(reflectors, R)`.
pub fn qr_blocked(
    a: &DenseMatrix,
    block_size: usize,
) -> (Vec<(Vec<f64>, f64)>, DenseMatrix) {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let k = m.min(n);
    let bs = block_size.max(1).min(k);
    let mut reflectors: Vec<(Vec<f64>, f64)> = Vec::with_capacity(k);

    let mut jb = 0;
    while jb < k {
        let nb = bs.min(k - jb); // columns in this block

        // Panel factorization
        let mut blk = BlockHouseholder::new(m - jb, nb);

        for jj in 0..nb {
            let j = jb + jj;
            let col_len = m - j;
            let mut x = vec![0.0; col_len];
            for i in 0..col_len {
                x[i] = r.get(j + i, j);
            }

            let h = HouseholderReflector::generate(&x);

            // Apply within the panel: R[j:m, j:jb+nb]
            HouseholderReflector::apply_left(
                &h.v,
                h.tau,
                &mut r,
                j,
                j,
                col_len,
                jb + nb - j,
            );

            // Embed the reflector into block accumulator with zero-padding
            let mut v_full = vec![0.0; m - jb];
            for i in 0..col_len {
                v_full[jj + i] = h.v[i];
            }
            blk.accumulate(&v_full, h.tau);

            reflectors.push((h.v, h.tau));
        }

        // Trailing update: apply block reflector to R[jb:m, jb+nb:n]
        let trailing_cols = n.saturating_sub(jb + nb);
        if trailing_cols > 0 {
            blk.apply_left_block(&mut r, jb, jb + nb, m - jb, trailing_cols);
        }

        jb += nb;
    }

    (reflectors, r)
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: apply a sequence of reflectors to a vector
// ═══════════════════════════════════════════════════════════════════════════

/// Apply a sequence of Householder reflectors to a vector, in order.
///
/// Each reflector k with `(v, tau)` acts on `x[k .. k + v.len()]`.
pub fn apply_reflectors_to_vec(x: &mut [f64], reflectors: &[(Vec<f64>, f64)]) {
    for (k, (v, tau)) in reflectors.iter().enumerate() {
        if *tau < 1e-300 {
            continue;
        }
        let len = v.len();
        let end = k + len;
        if end > x.len() {
            break;
        }
        let d: f64 = v.iter().zip(x[k..end].iter()).map(|(&vi, &xi)| vi * xi).sum();
        let td = tau * d;
        for i in 0..len {
            x[k + i] -= td * v[i];
        }
    }
}

/// Apply a sequence of Householder reflectors to a vector, in reverse order.
pub fn apply_reflectors_to_vec_reverse(x: &mut [f64], reflectors: &[(Vec<f64>, f64)]) {
    for (k, (v, tau)) in reflectors.iter().enumerate().rev() {
        if *tau < 1e-300 {
            continue;
        }
        let len = v.len();
        let end = k + len;
        if end > x.len() {
            break;
        }
        let d: f64 = v.iter().zip(x[k..end].iter()).map(|(&vi, &xi)| vi * xi).sum();
        let td = tau * d;
        for i in 0..len {
            x[k + i] -= td * v[i];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience: full QR with explicit Q
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the full QR factorization and return `(Q, R)` explicitly.
pub fn qr_full(a: &DenseMatrix) -> (DenseMatrix, DenseMatrix) {
    let (refs, r_raw) = qr_householder(a);
    let m = a.rows;
    let _n = a.cols;
    let q = form_q_from_reflectors(m, m, &refs);
    let r = extract_r(&r_raw);
    (q, r)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dot, norm2, DenseMatrix};

    const TOL: f64 = 1e-10;

    /// Check that a matrix is approximately orthogonal: QᵀQ ≈ I.
    fn assert_orthogonal(q: &DenseMatrix, tol: f64) {
        let qt = q.transpose();
        let qtq = qt.mul_mat(q).unwrap();
        let n = qtq.rows;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq.get(i, j) - expected).abs() < tol,
                    "QᵀQ[{},{}] = {}, expected {}",
                    i,
                    j,
                    qtq.get(i, j),
                    expected
                );
            }
        }
    }

    /// Check that a matrix is upper triangular below the k-th super-diagonal.
    fn assert_upper_bidiagonal(b: &DenseMatrix, tol: f64) {
        let m = b.rows;
        let n = b.cols;
        for i in 0..m {
            for j in 0..n {
                if j < i || j > i + 1 {
                    assert!(
                        b.get(i, j).abs() < tol,
                        "B[{},{}] = {}, expected 0",
                        i,
                        j,
                        b.get(i, j)
                    );
                }
            }
        }
    }

    fn assert_upper_hessenberg(h: &DenseMatrix, tol: f64) {
        let n = h.rows;
        for i in 2..n {
            for j in 0..i - 1 {
                assert!(
                    h.get(i, j).abs() < tol,
                    "H[{},{}] = {}, expected 0",
                    i,
                    j,
                    h.get(i, j)
                );
            }
        }
    }

    // ── Test 1: generate reflector zeros out a vector ────────────────────
    #[test]
    fn test_generate_reflector_zeros_out() {
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let h = HouseholderReflector::generate(&x);
        let y = h.apply_to_vec(&x);

        // y should be [±‖x‖, 0, 0, 0, 0]
        let x_norm = norm2(&x);
        assert!((y[0].abs() - x_norm).abs() < TOL, "y[0]={}, ‖x‖={}", y[0], x_norm);
        for i in 1..y.len() {
            assert!(y[i].abs() < TOL, "y[{}] = {}, expected 0", i, y[i]);
        }
    }

    // ── Test 2: apply_left preserves orthogonality ──────────────────────
    #[test]
    fn test_apply_left_preserves_orthogonality() {
        let q = DenseMatrix::eye(4);
        let mut m = q.clone();

        let v = vec![1.0, 1.0, 1.0, 1.0];
        let vv = dot(&v, &v);
        let tau = 2.0 / vv;

        HouseholderReflector::apply_left(&v, tau, &mut m, 0, 0, 4, 4);
        assert_orthogonal(&m, TOL);
    }

    // ── Test 3: apply_right works correctly ──────────────────────────────
    #[test]
    fn test_apply_right_correctness() {
        let mut a = DenseMatrix::from_row_major(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let a_orig = a.clone();

        let v = vec![1.0, 0.0, 0.0];
        let tau = 2.0 / dot(&v, &v);

        HouseholderReflector::apply_right(&v, tau, &mut a, 0, 0, 3, 3);

        // H = I - 2 e1 e1^T  flips sign of first column
        for i in 0..3 {
            assert!(
                (a.get(i, 0) - (-a_orig.get(i, 0))).abs() < TOL,
                "row {} col 0 mismatch",
                i
            );
            // Other columns unchanged
            for j in 1..3 {
                assert!(
                    (a.get(i, j) - a_orig.get(i, j)).abs() < TOL,
                    "row {} col {} changed",
                    i,
                    j
                );
            }
        }
    }

    // ── Test 4: bidiagonalize produces bidiagonal matrix ────────────────
    #[test]
    fn test_bidiagonalize_produces_bidiagonal() {
        let a = DenseMatrix::from_row_major(
            4,
            3,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let (u, v, diag, superdiag, _, _) = bidiagonalize(&a);

        // Reconstruct B = Uᵀ A V
        let ut = u.transpose();
        let ut_a = ut.mul_mat(&a).unwrap();
        let b = ut_a.mul_mat(&v).unwrap();

        assert_upper_bidiagonal(&b, 1e-8);
        assert_orthogonal(&u, 1e-8);
        assert_orthogonal(&v, 1e-8);

        // Check diagonal values match
        for i in 0..diag.len() {
            assert!(
                (b.get(i, i).abs() - diag[i].abs()).abs() < 1e-8,
                "diag[{}]: B={} vs d={}",
                i,
                b.get(i, i),
                diag[i]
            );
        }
    }

    // ── Test 5: Hessenberg reduction ────────────────────────────────────
    #[test]
    fn test_hessenberg_reduction() {
        let mut a = DenseMatrix::from_row_major(
            4,
            4,
            vec![
                4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0,
                -2.0, -1.0,
            ],
        );
        let a_orig = a.clone();

        let refs = reduce_to_hessenberg(&mut a);
        assert_upper_hessenberg(&a, 1e-10);

        // The similarity transform preserves eigenvalues; check trace (sum of eigenvalues).
        let trace_orig: f64 = (0..4).map(|i| a_orig.get(i, i)).sum();
        let trace_hess: f64 = (0..4).map(|i| a.get(i, i)).sum();
        assert!(
            (trace_orig - trace_hess).abs() < 1e-10,
            "Trace changed: {} vs {}",
            trace_orig,
            trace_hess
        );
    }

    // ── Test 6: Q formation from reflectors gives orthogonal matrix ─────
    #[test]
    fn test_form_q_orthogonal() {
        let a = DenseMatrix::from_row_major(
            5,
            3,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
        );
        let (refs, _r) = qr_householder(&a);
        let q = form_q_from_reflectors(5, 5, &refs);
        assert_orthogonal(&q, TOL);
    }

    // ── Test 7: block Householder matches sequential application ────────
    #[test]
    fn test_block_householder_matches_sequential() {
        let dim = 6;
        let mut a_seq = DenseMatrix::from_fn(dim, dim, |i, j| (i * dim + j) as f64 + 1.0);
        let mut a_blk = a_seq.clone();

        // Generate several reflectors
        let reflectors: Vec<(Vec<f64>, f64)> = (0..3)
            .map(|k| {
                let len = dim - k;
                let x: Vec<f64> = (0..len).map(|i| ((k + 1) * (i + 1)) as f64).collect();
                let h = HouseholderReflector::generate(&x);
                (h.v, h.tau)
            })
            .collect();

        // Sequential application from the left (each reflector embedded at its offset)
        for (k, (v, tau)) in reflectors.iter().enumerate() {
            let len = v.len();
            HouseholderReflector::apply_left(v, *tau, &mut a_seq, k, 0, len, dim);
        }

        // Block application
        let mut blk = BlockHouseholder::new(dim, 4);
        for (k, (v, tau)) in reflectors.iter().enumerate() {
            let mut v_full = vec![0.0; dim];
            for i in 0..v.len() {
                v_full[k + i] = v[i];
            }
            blk.accumulate(&v_full, *tau);
        }
        blk.apply_left_block(&mut a_blk, 0, 0, dim, dim);

        // Compare
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (a_seq.get(i, j) - a_blk.get(i, j)).abs() < 1e-8,
                    "Mismatch at ({},{}): seq={}, blk={}",
                    i,
                    j,
                    a_seq.get(i, j),
                    a_blk.get(i, j)
                );
            }
        }
    }

    // ── Test 8: zero vector edge case ───────────────────────────────────
    #[test]
    fn test_zero_vector() {
        let x = vec![0.0, 0.0, 0.0];
        let h = HouseholderReflector::generate(&x);
        assert!(h.tau.abs() < TOL, "Zero vector should give tau=0");
        let y = h.apply_to_vec(&x);
        for v in &y {
            assert!(v.abs() < TOL);
        }
    }

    // ── Test 9: single element ──────────────────────────────────────────
    #[test]
    fn test_single_element() {
        let x = vec![5.0];
        let h = HouseholderReflector::generate(&x);
        let y = h.apply_to_vec(&x);
        // Should produce [±5]
        assert!((y[0].abs() - 5.0).abs() < TOL);

        let x2 = vec![-7.0];
        let h2 = HouseholderReflector::generate(&x2);
        let y2 = h2.apply_to_vec(&x2);
        assert!((y2[0].abs() - 7.0).abs() < TOL);
    }

    // ── Test 10: QR factorization A = QR ────────────────────────────────
    #[test]
    fn test_qr_factorization_reconstruction() {
        let a = DenseMatrix::from_row_major(
            4,
            3,
            vec![
                12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0, -1.0, 1.0, 0.0,
            ],
        );
        let (q, r) = qr_full(&a);

        // Q should be orthogonal
        assert_orthogonal(&q, 1e-10);

        // R should be upper triangular
        let k = a.rows.min(a.cols);
        for i in 0..k {
            for j in 0..i {
                assert!(
                    r.get(i, j).abs() < 1e-10,
                    "R[{},{}] = {} not zero",
                    i,
                    j,
                    r.get(i, j)
                );
            }
        }

        // Reconstruct: Q[:, :k] * R ≈ A
        let q_thin = q.submatrix(0, q.rows, 0, k);
        let reconstructed = q_thin.mul_mat(&r).unwrap();
        for i in 0..a.rows {
            for j in 0..a.cols {
                assert!(
                    (reconstructed.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "Reconstruction mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    reconstructed.get(i, j),
                    a.get(i, j)
                );
            }
        }
    }

    // ── Test 11: Hessenberg Q formation ─────────────────────────────────
    #[test]
    fn test_hessenberg_q_formation() {
        let mut a = DenseMatrix::from_row_major(
            4,
            4,
            vec![
                4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0,
                -2.0, -1.0,
            ],
        );
        let a_orig = a.clone();
        let refs = reduce_to_hessenberg(&mut a);
        let q = form_q_from_hessenberg_reflectors(4, &refs);

        assert_orthogonal(&q, 1e-10);

        // Verify similarity: Q^T A_orig Q = H
        let qt = q.transpose();
        let qt_a = qt.mul_mat(&a_orig).unwrap();
        let h = qt_a.mul_mat(&q).unwrap();

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (h.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "QᵀAQ vs H mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    h.get(i, j),
                    a.get(i, j)
                );
            }
        }
    }

    // ── Test 12: blocked QR matches unblocked ───────────────────────────
    #[test]
    fn test_blocked_qr_matches_unblocked() {
        let a = DenseMatrix::from_row_major(
            5,
            4,
            vec![
                2.0, -1.0, 0.0, 3.0, 1.0, 3.0, -1.0, 2.0, -1.0, 0.0, 4.0, 1.0, 0.0, 2.0,
                -1.0, 3.0, 3.0, 1.0, 2.0, -2.0,
            ],
        );
        let (refs_ub, r_ub) = qr_householder(&a);
        let (refs_bl, r_bl) = qr_blocked(&a, 2);

        // R matrices should agree (up to sign flips on rows)
        let k = a.rows.min(a.cols);
        for i in 0..k {
            // Determine sign of R[i,i]
            let sign_ub = r_ub.get(i, i).signum();
            let sign_bl = r_bl.get(i, i).signum();
            let flip = sign_ub * sign_bl;
            for j in i..a.cols {
                assert!(
                    (r_ub.get(i, j) - flip * r_bl.get(i, j)).abs() < 1e-8,
                    "R mismatch at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    // ── Test 13: reflector is involutory (H² = I) ───────────────────────
    #[test]
    fn test_reflector_involutory() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let h = HouseholderReflector::generate(&x);

        // Apply twice: should get back x
        let y = h.apply_to_vec(&x);
        let z = h.apply_to_vec(&y);
        for i in 0..x.len() {
            assert!(
                (z[i] - x[i]).abs() < TOL,
                "H² x ≠ x at index {}: {} vs {}",
                i,
                z[i],
                x[i]
            );
        }
    }

    // ── Test 14: empty vector edge case ─────────────────────────────────
    #[test]
    fn test_empty_vector() {
        let x: Vec<f64> = vec![];
        let h = HouseholderReflector::generate(&x);
        assert_eq!(h.v.len(), 0);
        assert!(h.tau.abs() < TOL);
    }

    // ── Test 15: bidiagonalize square matrix ────────────────────────────
    #[test]
    fn test_bidiagonalize_square() {
        let a = DenseMatrix::from_row_major(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        );
        let (u, v, _diag, _superdiag, _, _) = bidiagonalize(&a);
        assert_orthogonal(&u, 1e-8);
        assert_orthogonal(&v, 1e-8);

        let ut = u.transpose();
        let ut_a = ut.mul_mat(&a).unwrap();
        let b = ut_a.mul_mat(&v).unwrap();
        assert_upper_bidiagonal(&b, 1e-8);
    }

    // ── Test 16: apply_reflectors_to_vec roundtrip ──────────────────────
    #[test]
    fn test_apply_reflectors_roundtrip() {
        let a = DenseMatrix::from_row_major(
            4,
            3,
            vec![
                1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
            ],
        );
        let (refs, _r) = qr_householder(&a);

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let x_orig = x.clone();

        // Q x
        apply_reflectors_to_vec_reverse(&mut x, &refs);
        // Qᵀ (Q x) should give back x
        apply_reflectors_to_vec(&mut x, &refs);

        for i in 0..x.len() {
            assert!(
                (x[i] - x_orig[i]).abs() < 1e-10,
                "Roundtrip failed at {}: {} vs {}",
                i,
                x[i],
                x_orig[i]
            );
        }
    }

    // ── Test 17: block Householder right application ────────────────────
    #[test]
    fn test_block_householder_right() {
        let dim = 5;
        let mut a_seq = DenseMatrix::from_fn(dim, dim, |i, j| ((i + 1) * (j + 2)) as f64);
        let mut a_blk = a_seq.clone();

        let reflectors: Vec<(Vec<f64>, f64)> = (0..2)
            .map(|k| {
                let len = dim - k;
                let x: Vec<f64> = (0..len).map(|i| ((k + 2) * (i + 1)) as f64).collect();
                let h = HouseholderReflector::generate(&x);
                (h.v, h.tau)
            })
            .collect();

        // Sequential right application
        for (k, (v, tau)) in reflectors.iter().enumerate() {
            let len = v.len();
            HouseholderReflector::apply_right(v, *tau, &mut a_seq, 0, k, dim, len);
        }

        // Block right application
        let mut blk = BlockHouseholder::new(dim, 4);
        for (k, (v, tau)) in reflectors.iter().enumerate() {
            let mut v_full = vec![0.0; dim];
            for i in 0..v.len() {
                v_full[k + i] = v[i];
            }
            blk.accumulate(&v_full, *tau);
        }
        blk.apply_right_block(&mut a_blk, 0, 0, dim, dim);

        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (a_seq.get(i, j) - a_blk.get(i, j)).abs() < 1e-8,
                    "Right-block mismatch at ({},{}): seq={}, blk={}",
                    i,
                    j,
                    a_seq.get(i, j),
                    a_blk.get(i, j)
                );
            }
        }
    }
}
