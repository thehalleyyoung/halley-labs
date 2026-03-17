//! Givens rotations for matrix transformations.
//!
//! Provides numerically stable Givens rotations for zeroing out individual
//! matrix elements, along with composite rotation sequences and specialised
//! routines for Hessenberg QR and implicit tridiagonal QR steps.

use crate::{DenseMatrix, DecompResult};

// ═══════════════════════════════════════════════════════════════════════════
// Single Givens rotation
// ═══════════════════════════════════════════════════════════════════════════

/// A single Givens rotation acting on rows `i` and `j`.
///
/// The rotation matrix G has the form:
///   G(i,j) with entries  G[i,i] = c,  G[i,j] = s,
///                         G[j,i] = -s, G[j,j] = c,
/// and all other diagonal entries = 1.
///
/// Convention: `G^T * [a; b] = [r; 0]`.
#[derive(Debug, Clone, Copy)]
pub struct GivensRotation {
    /// Cosine component.
    pub c: f64,
    /// Sine component.
    pub s: f64,
    /// First (pivot) row index.
    pub i: usize,
    /// Second row index (the one being zeroed out).
    pub j: usize,
}

impl GivensRotation {
    // ── Core computation ────────────────────────────────────────────────

    /// Compute `(c, s, r)` such that
    ///
    ///   [ c  s ]^T  [ a ]   [ r ]
    ///   [ -s c ]    [ b ] = [ 0 ]
    ///
    /// Uses the numerically stable Golub–Van Loan formulation (Algorithm 5.1.3)
    /// that avoids overflow and unnecessary loss of precision.
    pub fn compute(a: f64, b: f64) -> (f64, f64, f64) {
        if b == 0.0 {
            // Already zero – no rotation needed.
            return (1.0, 0.0, a);
        }
        if a == 0.0 {
            // Pure sine rotation.
            return (0.0, 1.0, b);
        }

        let abs_a = a.abs();
        let abs_b = b.abs();

        if abs_b > abs_a {
            // |b| dominates: use tangent t = a/b to avoid large intermediates.
            let t = a / b;
            let s = 1.0 / (1.0 + t * t).sqrt();
            let c = s * t;
            let r = b / s; // = sign(b) * hypot(a, b)
            (c, s, r)
        } else {
            // |a| dominates: use tangent t = b/a.
            let t = b / a;
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = c * t;
            let r = a / c; // = sign(a) * hypot(a, b)
            (c, s, r)
        }
    }

    /// Create a Givens rotation that zeros out the element in row `j` using
    /// row `i` as the pivot.
    ///
    /// # Arguments
    /// * `i` – pivot row
    /// * `j` – row to be zeroed (must differ from `i`)
    /// * `a` – value at position `(i, col)` in the column of interest
    /// * `b` – value at position `(j, col)` to be zeroed
    pub fn new(i: usize, j: usize, a: f64, b: f64) -> Self {
        debug_assert_ne!(i, j, "Givens rotation rows must be distinct");
        let (c, s, _r) = Self::compute(a, b);
        Self { c, s, i, j }
    }

    /// Create a row-index-free rotation from values `a` and `b`.
    ///
    /// Row indices default to `(0, 1)`.  Used by callers that manage their own
    /// index bookkeeping (e.g. the QR module).
    #[inline]
    pub fn from_values(a: f64, b: f64) -> Self {
        let (c, s, _r) = Self::compute(a, b);
        Self { c, s, i: 0, j: 1 }
    }

    /// Apply the rotation to a pair of values, returning the transformed pair
    /// `(c*a + s*b, -s*a + c*b)`.
    #[inline]
    pub fn apply(&self, a: f64, b: f64) -> (f64, f64) {
        (self.c * a + self.s * b, -self.s * a + self.c * b)
    }

    /// Create directly from precomputed `c` and `s` values.
    pub fn from_cs(i: usize, j: usize, c: f64, s: f64) -> Self {
        Self { c, s, i, j }
    }

    // ── Left application (row operations) ───────────────────────────────

    /// Apply `G^T` from the left to rows `i` and `j` of `mat`, modifying
    /// columns in the range `[col_start, col_end)`.
    ///
    /// For each column `k` in that range:
    ///   temp_i =  c * A[i,k] + s * A[j,k]
    ///   temp_j = -s * A[i,k] + c * A[j,k]
    pub fn apply_left(
        &self,
        mat: &mut DenseMatrix,
        col_start: usize,
        col_end: usize,
    ) {
        let c = self.c;
        let s = self.s;
        let ri = self.i;
        let rj = self.j;
        let cols = mat.cols;

        for k in col_start..col_end {
            let a_ik = mat.data[ri * cols + k];
            let a_jk = mat.data[rj * cols + k];
            mat.data[ri * cols + k] = c * a_ik + s * a_jk;
            mat.data[rj * cols + k] = -s * a_ik + c * a_jk;
        }
    }

    /// Convenience: apply `G^T` from the left to **all** columns of `mat`.
    pub fn apply_left_full(&self, mat: &mut DenseMatrix) {
        let cols = mat.cols;
        self.apply_left(mat, 0, cols);
    }

    // ── Right application (column operations) ───────────────────────────

    /// Apply `G` from the right to columns `i` and `j` of `mat`, modifying
    /// rows in the range `[row_start, row_end)`.
    ///
    /// For each row `k`:
    ///   temp_i =  c * A[k,i] + s * A[k,j]
    ///   temp_j = -s * A[k,i] + c * A[k,j]
    pub fn apply_right(
        &self,
        mat: &mut DenseMatrix,
        row_start: usize,
        row_end: usize,
    ) {
        let c = self.c;
        let s = self.s;
        let ci = self.i;
        let cj = self.j;
        let cols = mat.cols;

        for k in row_start..row_end {
            let a_ki = mat.data[k * cols + ci];
            let a_kj = mat.data[k * cols + cj];
            mat.data[k * cols + ci] = c * a_ki + s * a_kj;
            mat.data[k * cols + cj] = -s * a_ki + c * a_kj;
        }
    }

    /// Convenience: apply `G` from the right to **all** rows of `mat`.
    pub fn apply_right_full(&self, mat: &mut DenseMatrix) {
        let rows = mat.rows;
        self.apply_right(mat, 0, rows);
    }

    // ── Vector application ──────────────────────────────────────────────

    /// Apply `G^T` to a vector `v`, modifying entries at positions `i` and `j`.
    pub fn apply_left_vec(&self, v: &mut [f64]) {
        let c = self.c;
        let s = self.s;
        let vi = v[self.i];
        let vj = v[self.j];
        v[self.i] = c * vi + s * vj;
        v[self.j] = -s * vi + c * vj;
    }

    // ── Query helpers ───────────────────────────────────────────────────

    /// Returns the value `r` that the pivot element takes after the rotation.
    pub fn resultant(a: f64, b: f64) -> f64 {
        let (_c, _s, r) = Self::compute(a, b);
        r
    }

    /// Check whether this rotation is (approximately) the identity.
    pub fn is_identity(&self, tol: f64) -> bool {
        (self.c - 1.0).abs() < tol && self.s.abs() < tol
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sequence of Givens rotations  (Q = G_1 G_2 … G_k)
// ═══════════════════════════════════════════════════════════════════════════

/// An ordered sequence of Givens rotations.
///
/// When applied from the left the product Q^T = G_k^T … G_1^T is formed;
/// when applied from the right Q = G_1 G_2 … G_k.
#[derive(Debug, Clone)]
pub struct GivensSequence {
    rotations: Vec<GivensRotation>,
}

impl GivensSequence {
    /// Create an empty sequence.
    pub fn new() -> Self {
        Self {
            rotations: Vec::new(),
        }
    }

    /// Create with a pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            rotations: Vec::with_capacity(cap),
        }
    }

    /// Append a rotation to the sequence.
    pub fn push(&mut self, rot: GivensRotation) {
        self.rotations.push(rot);
    }

    /// Number of rotations stored.
    pub fn len(&self) -> usize {
        self.rotations.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.rotations.is_empty()
    }

    /// Iterate over the rotations in order.
    pub fn iter(&self) -> impl Iterator<Item = &GivensRotation> {
        self.rotations.iter()
    }

    // ── Bulk application ────────────────────────────────────────────────

    /// Apply **all** rotations from the left (i.e. form Q^T * mat).
    ///
    /// Rotations are applied in forward order: G_1, G_2, …, G_k.
    pub fn apply_left_all(&self, mat: &mut DenseMatrix) {
        let cols = mat.cols;
        for rot in &self.rotations {
            rot.apply_left(mat, 0, cols);
        }
    }

    /// Apply **all** rotations from the right (i.e. form mat * Q).
    ///
    /// Rotations are applied in forward order: G_1, G_2, …, G_k.
    pub fn apply_right_all(&self, mat: &mut DenseMatrix) {
        let rows = mat.rows;
        for rot in &self.rotations {
            rot.apply_right(mat, 0, rows);
        }
    }

    /// Apply the *transpose* sequence from the right, i.e. form mat * Q^T.
    ///
    /// Rotations are applied in **reverse** order with negated `s`.
    pub fn apply_right_transpose(&self, mat: &mut DenseMatrix) {
        let rows = mat.rows;
        for rot in self.rotations.iter().rev() {
            let rt = GivensRotation::from_cs(rot.i, rot.j, rot.c, -rot.s);
            rt.apply_right(mat, 0, rows);
        }
    }

    /// Apply the *transpose* sequence from the left, i.e. form Q * mat.
    ///
    /// Rotations are applied in **reverse** order with negated `s`.
    pub fn apply_left_transpose(&self, mat: &mut DenseMatrix) {
        let cols = mat.cols;
        for rot in self.rotations.iter().rev() {
            let rt = GivensRotation::from_cs(rot.i, rot.j, rot.c, -rot.s);
            rt.apply_left(mat, 0, cols);
        }
    }

    // ── Explicit matrix formation ───────────────────────────────────────

    /// Form the explicit orthogonal matrix Q = G_1 G_2 … G_k of size n×n.
    ///
    /// Starts from the identity and applies each rotation from the right.
    pub fn to_matrix(&self, n: usize) -> DenseMatrix {
        let mut q = DenseMatrix::eye(n);
        for rot in &self.rotations {
            rot.apply_right_full(&mut q);
        }
        q
    }

    /// Form Q^T = G_k^T … G_1^T explicitly.
    pub fn to_matrix_transpose(&self, n: usize) -> DenseMatrix {
        let mut qt = DenseMatrix::eye(n);
        for rot in &self.rotations {
            rot.apply_left_full(&mut qt);
        }
        qt
    }
}

impl Default for GivensSequence {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QR factorisation of an upper Hessenberg matrix
// ═══════════════════════════════════════════════════════════════════════════

/// QR factorisation of an upper Hessenberg matrix using Givens rotations.
///
/// An upper Hessenberg matrix `H` has zeros below the first subdiagonal.  Each
/// column therefore requires only **one** Givens rotation to annihilate the
/// single subdiagonal entry, giving an O(n²) algorithm.
///
/// On return `H` is overwritten with the upper-triangular factor R, and the
/// returned [`GivensSequence`] encodes Q^T = G_{n-1} … G_1  (so H = Q R).
///
/// # Panics
/// Panics if the matrix is not square.
pub fn qr_hessenberg(h: &mut DenseMatrix) -> GivensSequence {
    let n = h.rows;
    assert_eq!(h.rows, h.cols, "qr_hessenberg requires a square matrix");

    let mut seq = GivensSequence::with_capacity(n.saturating_sub(1));

    for k in 0..n.saturating_sub(1) {
        let a = h.get(k, k);
        let b = h.get(k + 1, k);

        if b.abs() < 1e-300 {
            // Already zero – push identity rotation to keep indexing consistent.
            seq.push(GivensRotation::from_cs(k, k + 1, 1.0, 0.0));
            continue;
        }

        let rot = GivensRotation::new(k, k + 1, a, b);
        // Apply G^T to rows k, k+1 for columns k..n  (everything left of k is
        // already zero in both rows for a Hessenberg matrix).
        rot.apply_left(h, k, n);
        // The subdiagonal element should now be zero (up to rounding).
        h.set(k + 1, k, 0.0); // enforce exact zero
        seq.push(rot);
    }

    seq
}

/// Perform a *full* QR step on a Hessenberg matrix:  H ← R Q + σ I.
///
/// This is the standard Francis single-shift QR iteration step used as a
/// building-block in eigenvalue algorithms.
///
/// 1. Form H - σ I.
/// 2. QR-factorise (in-place Givens on Hessenberg form).
/// 3. Form R Q (apply rotations from the right).
/// 4. Add back σ I.
///
/// Returns the Givens sequence from step 2 (useful for accumulating
/// eigenvectors).
pub fn qr_hessenberg_step(h: &mut DenseMatrix, shift: f64) -> GivensSequence {
    let n = h.rows;

    // 1. H ← H - σ I
    for i in 0..n {
        let v = h.get(i, i) - shift;
        h.set(i, i, v);
    }

    // 2. QR factorise → H holds R, rotations in seq
    let seq = qr_hessenberg(h);

    // 3. H ← R Q  (apply each G from the right)
    seq.apply_right_all(h);

    // 4. H ← H + σ I
    for i in 0..n {
        let v = h.get(i, i) + shift;
        h.set(i, i, v);
    }

    seq
}

// ═══════════════════════════════════════════════════════════════════════════
// Implicit QR step on a symmetric tridiagonal matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Wilkinson shift for a 2×2 trailing submatrix of a symmetric
/// tridiagonal matrix with diagonal `alpha` and subdiagonal `beta`.
///
/// The shift is the eigenvalue of the trailing 2×2 block closest to
/// `alpha[end]`.
fn wilkinson_shift(alpha: &[f64], beta: &[f64], end: usize) -> f64 {
    let d = (alpha[end - 1] - alpha[end]) * 0.5;
    let b_sq = beta[end - 1] * beta[end - 1];

    if d == 0.0 {
        // Avoid division by zero – just use alpha[end] - |beta|.
        alpha[end] - beta[end - 1].abs()
    } else {
        let sign_d = if d >= 0.0 { 1.0 } else { -1.0 };
        alpha[end] - b_sq / (d + sign_d * (d * d + b_sq).sqrt())
    }
}

/// One implicit QR step on a symmetric tridiagonal matrix using bulge chasing.
///
/// The tridiagonal matrix T is stored compactly:
///   - `alpha[i]` = T[i,i]         (diagonal, length n)
///   - `beta[i]`  = T[i+1,i]       (subdiagonal, length n-1)
///
/// Only the portion `[start..=end]` is processed (the rest is assumed
/// already deflated).  A Wilkinson shift is computed from the trailing 2×2
/// block.
///
/// Returns the Givens rotations applied so that eigenvector accumulation can
/// be done externally.
pub fn qr_tridiagonal_step(
    alpha: &mut [f64],
    beta: &mut [f64],
    start: usize,
    end: usize,
    shift: f64,
) -> Vec<GivensRotation> {
    debug_assert!(end > start, "qr_tridiagonal_step: end must exceed start");

    let mut rots = Vec::with_capacity(end - start);

    // Initial bulge: zero out the first subdiagonal element of (T - σ I) in
    // the active window.
    let mut x = alpha[start] - shift;
    let mut z = beta[start];

    for k in start..end {
        // Compute rotation to zero z.
        let (c, s, r) = GivensRotation::compute(x, z);

        // Save rotation (row indices in the global matrix).
        rots.push(GivensRotation { c, s, i: k, j: k + 1 });

        // Apply the similarity transformation  T ← G^T T G.
        // Because T is tridiagonal only a small band is affected.

        if k > start {
            // Fill in the super-/sub-diagonal from the previous column.
            beta[k - 1] = r;
        }

        // Diagonal block update (2×2 principal submatrix at rows k, k+1).
        let ak = alpha[k];
        let ak1 = alpha[k + 1];
        let bk = beta[k];

        // The similarity transform G^T diag(ak, ak1) G in the 2×2 block:
        //   [ c  s ]^T [ ak  bk  ] [ c  s ]
        //   [-s  c ]   [ bk  ak1 ] [-s  c ]
        let c2 = c * c;
        let s2 = s * s;
        let cs = c * s;

        alpha[k] = c2 * ak + 2.0 * cs * bk + s2 * ak1;
        alpha[k + 1] = s2 * ak - 2.0 * cs * bk + c2 * ak1;
        beta[k] = cs * (ak1 - ak) + (c2 - s2) * bk;

        // Chase the bulge: the rotation introduces a non-zero at position
        // (k+2, k) which becomes the new `z`.
        if k + 1 < end {
            x = beta[k];
            z = -s * beta[k + 1];
            beta[k + 1] *= c;
        }
    }

    rots
}

/// Run the full implicit symmetric tridiagonal QR algorithm to convergence.
///
/// On return `alpha` holds the eigenvalues.  If `q` is `Some`, the
/// eigenvector matrix is accumulated there (initialise to I before calling).
///
/// Returns the number of iterations performed.
pub fn tridiagonal_qr_eigen(
    alpha: &mut [f64],
    beta: &mut [f64],
    mut q: Option<&mut DenseMatrix>,
    max_iter: usize,
    tol: f64,
) -> DecompResult<usize> {
    let n = alpha.len();
    if n == 0 {
        return Ok(0);
    }
    if n == 1 {
        return Ok(0);
    }

    let mut total_iter = 0usize;
    let mut end = n - 1;

    while end > 0 {
        // Deflation: find the largest `end` where beta[end-1] is negligible.
        let off_diag_norm = beta[end - 1].abs();
        let diag_norm = alpha[end - 1].abs() + alpha[end].abs();
        if off_diag_norm <= tol * diag_norm.max(1e-300) {
            beta[end - 1] = 0.0;
            end -= 1;
            continue;
        }

        // Find the start of the unreduced block.
        let mut start = end - 1;
        while start > 0 {
            let off = beta[start - 1].abs();
            let diag = alpha[start - 1].abs() + alpha[start].abs();
            if off <= tol * diag.max(1e-300) {
                beta[start - 1] = 0.0;
                break;
            }
            start -= 1;
        }

        if total_iter >= max_iter {
            return Err(crate::DecompError::ConvergenceFailure {
                iterations: max_iter,
                context: "tridiagonal QR eigenvalue".into(),
            });
        }

        let shift = wilkinson_shift(alpha, beta, end);
        let rots = qr_tridiagonal_step(alpha, beta, start, end, shift);

        // Accumulate eigenvectors.
        if let Some(ref mut qm) = q.as_deref_mut() {
            let rows = qm.rows;
            for rot in &rots {
                rot.apply_right(qm, 0, rows);
            }
        }

        total_iter += 1;
    }

    Ok(total_iter)
}

// ═══════════════════════════════════════════════════════════════════════════
// Fast Givens rotation (square-root-free)
// ═══════════════════════════════════════════════════════════════════════════

/// A "fast" Givens rotation that avoids computing square roots by maintaining
/// diagonal scaling factors.
///
/// Instead of the standard form `[c s; -s c]`, a fast Givens rotation uses
///
///   type 1:  [ 1     alpha ]   with scaling  diag(d1, d2)
///            [ -beta   1   ]
///
///   type 2:  [ alpha   1   ]   with scaling  diag(d1, d2)
///            [  -1    beta ]
///
/// The invariant D^{-1/2} M^T D^{1/2} is orthogonal, allowing the product
/// to be formed without ever computing a square root.
#[derive(Debug, Clone, Copy)]
pub struct FastGivens {
    /// Parameter alpha (role depends on rotation type).
    pub alpha: f64,
    /// Parameter beta (role depends on rotation type).
    pub beta: f64,
    /// Updated first scaling factor.
    pub d1: f64,
    /// Updated second scaling factor.
    pub d2: f64,
    /// The resultant `r` value in the pivot position.
    pub r: f64,
    /// `true` iff type-2 rotation was used (|b/a| > |a/b|).
    pub swapped: bool,
}

impl FastGivens {
    /// Compute a fast Givens rotation that zeros out element `b` against
    /// pivot `a`, given current scaling factors `d1` and `d2`.
    ///
    /// Uses the BLAS DROTMG formulation.  The matrix `H` applied to `[a; b]`
    /// produces `[r; 0]` where `r` is returned.  Two forms exist:
    ///
    /// **Type 1** (`!swapped`): `H = [1, h12; h21, 1]` with `h21 = -b/a`,
    /// `h12 = d2*b / (d1*a)`.
    ///
    /// **Type 2** (`swapped`): `H = [h11, 1; -1, h22]` with `h22 = a/b`,
    /// `h11 = d1*a / (d2*b)`.
    pub fn compute(d1: f64, d2: f64, a: f64, b: f64) -> Self {
        if b == 0.0 {
            return Self {
                alpha: 0.0,
                beta: 0.0,
                d1,
                d2,
                r: a,
                swapped: false,
            };
        }
        if a == 0.0 {
            return Self {
                alpha: 0.0,
                beta: 0.0,
                d1: d2,
                d2: d1,
                r: b,
                swapped: true,
            };
        }

        let p = d1 * a;
        let q = d2 * b;

        if p.abs() >= q.abs() {
            // Type 1: H = [1, h12; h21, 1]
            let h21 = -b / a;
            let h12 = q / p; // = (d2*b)/(d1*a)
            let u = 1.0 - h12 * h21; // = 1 + d2*b^2/(d1*a^2)
            let new_d1 = d1 / u;
            let new_d2 = d2 / u;
            let r = a * u;

            Self {
                alpha: h12,
                beta: -h21, // store positive: b/a
                d1: new_d1,
                d2: new_d2,
                r,
                swapped: false,
            }
        } else {
            // Type 2: H = [h11, 1; -1, h22]
            let h11 = p / q; // = (d1*a)/(d2*b)
            let h22 = a / b;
            let u = 1.0 + h11 * h22; // = 1 + d1*a^2/(d2*b^2)
            let new_d1 = d2 / u;
            let new_d2 = d1 / u;
            let r = b * u;

            Self {
                alpha: h11,
                beta: h22,
                d1: new_d1,
                d2: new_d2,
                r,
                swapped: true,
            }
        }
    }

    /// Apply the fast rotation from the left to rows `ri`, `rj` of `mat`
    /// for columns in `[col_start, col_end)`.
    pub fn apply_left(
        &self,
        mat: &mut DenseMatrix,
        ri: usize,
        rj: usize,
        col_start: usize,
        col_end: usize,
    ) {
        let cols = mat.cols;
        if !self.swapped {
            // Type 1:  [ 1     alpha ] applied to [row_i; row_j]
            //          [ -beta   1   ]
            for k in col_start..col_end {
                let a_ik = mat.data[ri * cols + k];
                let a_jk = mat.data[rj * cols + k];
                mat.data[ri * cols + k] = a_ik + self.alpha * a_jk;
                mat.data[rj * cols + k] = -self.beta * a_ik + a_jk;
            }
        } else {
            // Type 2:  [ alpha  1 ] applied to [row_i; row_j]
            //          [ -1   beta ]
            for k in col_start..col_end {
                let a_ik = mat.data[ri * cols + k];
                let a_jk = mat.data[rj * cols + k];
                mat.data[ri * cols + k] = self.alpha * a_ik + a_jk;
                mat.data[rj * cols + k] = -a_ik + self.beta * a_jk;
            }
        }
    }

    /// Convert this fast Givens rotation to an equivalent standard Givens
    /// rotation (with explicit `c` and `s`).
    pub fn to_standard(&self, _d1_orig: f64, _d2_orig: f64) -> (f64, f64) {
        if !self.swapped {
            // H = [1, alpha; -beta, 1]
            // The standard equivalent has c, s such that
            // [c s; -s c]^T [a; b] = [r; 0], where beta = b/a, alpha = h12.
            let denom = (1.0 + self.alpha * self.beta).sqrt();
            let c = 1.0 / denom;
            let s = self.beta / denom;
            (c, s)
        } else {
            // H = [alpha, 1; -1, beta]
            let denom = (1.0 + self.alpha * self.beta).sqrt();
            let c = self.alpha / denom;
            let s = 1.0 / denom;
            (c, s)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: build a full Givens rotation matrix (for debugging / testing)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the full n×n Givens rotation matrix G for the given rotation.
///
/// G has the form:
///   G[i,i] = c,  G[i,j] = -s,
///   G[j,i] = s,  G[j,j] = c,
/// with all other diagonal entries = 1.
pub fn givens_matrix(rot: &GivensRotation, n: usize) -> DenseMatrix {
    let mut g = DenseMatrix::eye(n);
    g.set(rot.i, rot.i, rot.c);
    g.set(rot.i, rot.j, -rot.s);
    g.set(rot.j, rot.i, rot.s);
    g.set(rot.j, rot.j, rot.c);
    g
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Check that Q^T Q ≈ I (orthogonality).
    fn check_orthogonal(q: &DenseMatrix, tol: f64) {
        let qt = q.transpose();
        let prod = qt.mul_mat(q).unwrap();
        let n = prod.rows;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(prod.get(i, j), expected, tol),
                    "Q^T Q [{},{}] = {}, expected {}",
                    i,
                    j,
                    prod.get(i, j),
                    expected,
                );
            }
        }
    }

    // ── Test 1: compute gives correct c, s, r ──────────────────────────

    #[test]
    fn test_compute_basic() {
        let (c, s, r) = GivensRotation::compute(3.0, 4.0);
        // c^2 + s^2 = 1
        assert!(approx_eq(c * c + s * s, 1.0, EPS), "c²+s² = {}", c * c + s * s);
        // r should equal hypot(3,4) = 5 (up to sign)
        assert!(approx_eq(r.abs(), 5.0, EPS), "r = {}", r);
        // G^T [a; b] = [r; 0]
        let new_a = c * 3.0 + s * 4.0;
        let new_b = -s * 3.0 + c * 4.0;
        assert!(approx_eq(new_a, r, EPS), "new_a = {}, r = {}", new_a, r);
        assert!(approx_eq(new_b, 0.0, EPS), "new_b = {}", new_b);
    }

    #[test]
    fn test_compute_negative_values() {
        let (c, s, r) = GivensRotation::compute(-5.0, 12.0);
        assert!(approx_eq(c * c + s * s, 1.0, EPS));
        let new_b = -s * (-5.0) + c * 12.0;
        assert!(approx_eq(new_b, 0.0, EPS), "new_b = {}", new_b);
        assert!(approx_eq(r.abs(), 13.0, EPS), "r = {}", r);
    }

    // ── Test 2: edge case b = 0 ────────────────────────────────────────

    #[test]
    fn test_compute_b_zero() {
        let (c, s, r) = GivensRotation::compute(7.0, 0.0);
        assert!(approx_eq(c, 1.0, EPS));
        assert!(approx_eq(s, 0.0, EPS));
        assert!(approx_eq(r, 7.0, EPS));
    }

    // ── Test 3: edge case a = 0 ────────────────────────────────────────

    #[test]
    fn test_compute_a_zero() {
        let (c, s, r) = GivensRotation::compute(0.0, 5.0);
        assert!(approx_eq(c, 0.0, EPS));
        assert!(approx_eq(s, 1.0, EPS));
        // G^T [0; 5] = [r; 0]
        let new_a = c * 0.0 + s * 5.0;
        let new_b = -s * 0.0 + c * 5.0;
        assert!(approx_eq(new_a, r, EPS));
        assert!(approx_eq(new_b, 0.0, EPS));
        assert!(approx_eq(r, 5.0, EPS));
    }

    // ── Test 4: edge case a = b ────────────────────────────────────────

    #[test]
    fn test_compute_a_equals_b() {
        let (c, s, r) = GivensRotation::compute(1.0, 1.0);
        assert!(approx_eq(c * c + s * s, 1.0, EPS));
        let new_b = -s * 1.0 + c * 1.0;
        assert!(approx_eq(new_b, 0.0, EPS));
        assert!(approx_eq(r.abs(), 2.0_f64.sqrt(), EPS));
    }

    // ── Test 5: apply_left zeros out the target element ────────────────

    #[test]
    fn test_apply_left_zeros_element() {
        let mut mat = DenseMatrix::from_row_major(
            4,
            4,
            vec![
                5.0, 1.0, 2.0, 3.0,
                3.0, 4.0, 1.0, 2.0,
                0.0, 0.0, 7.0, 6.0,
                0.0, 0.0, 0.0, 8.0,
            ],
        );
        // Zero out mat[1,0] using row 0 as pivot.
        let rot = GivensRotation::new(0, 1, mat.get(0, 0), mat.get(1, 0));
        rot.apply_left(&mut mat, 0, 4);

        assert!(
            approx_eq(mat.get(1, 0), 0.0, EPS),
            "mat[1,0] = {}",
            mat.get(1, 0)
        );
    }

    // ── Test 6: apply_right works correctly ─────────────────────────────

    #[test]
    fn test_apply_right_correctness() {
        let n = 4;
        let mut mat = DenseMatrix::from_row_major(
            n,
            n,
            vec![
                2.0, 1.0, 0.0, 0.0,
                1.0, 3.0, 1.0, 0.0,
                0.0, 1.0, 4.0, 1.0,
                0.0, 0.0, 1.0, 5.0,
            ],
        );
        let mat_orig = mat.clone();
        let rot = GivensRotation::new(1, 2, 3.0, 1.0);

        // Apply from the right to all rows.
        rot.apply_right(&mut mat, 0, n);

        // Verify against explicit G matrix multiplication.
        let g = givens_matrix(&rot, n);
        let expected = mat_orig.mul_mat(&g).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(mat.get(i, j), expected.get(i, j), EPS),
                    "[{},{}]: got {} expected {}",
                    i,
                    j,
                    mat.get(i, j),
                    expected.get(i, j),
                );
            }
        }
    }

    // ── Test 7: apply_left_vec ──────────────────────────────────────────

    #[test]
    fn test_apply_left_vec() {
        let mut v = vec![3.0, 4.0, 0.0];
        let rot = GivensRotation::new(0, 1, 3.0, 4.0);
        rot.apply_left_vec(&mut v);

        assert!(approx_eq(v[1], 0.0, EPS), "v[1] = {}", v[1]);
        assert!(approx_eq(v[0].abs(), 5.0, EPS));
    }

    // ── Test 8: QR of Hessenberg matrix ─────────────────────────────────

    #[test]
    fn test_qr_hessenberg() {
        // Upper Hessenberg matrix.
        let mut h = DenseMatrix::from_row_major(
            4,
            4,
            vec![
                4.0, 1.0, 2.0, 3.0,
                3.0, 4.0, 1.0, 2.0,
                0.0, 2.0, 3.0, 1.0,
                0.0, 0.0, 1.0, 5.0,
            ],
        );
        let h_orig = h.clone();
        let seq = qr_hessenberg(&mut h);

        // h now holds R (upper triangular).
        for i in 1..4 {
            for j in 0..i {
                assert!(
                    approx_eq(h.get(i, j), 0.0, 1e-10),
                    "R[{},{}] = {} (should be 0)",
                    i,
                    j,
                    h.get(i, j),
                );
            }
        }

        // Q should be orthogonal.
        let q = seq.to_matrix(4);
        check_orthogonal(&q, 1e-10);

        // Q * R should equal H_orig.
        let qr = q.mul_mat(&h).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    approx_eq(qr.get(i, j), h_orig.get(i, j), 1e-10),
                    "QR[{},{}] = {}, H[{},{}] = {}",
                    i,
                    j,
                    qr.get(i, j),
                    i,
                    j,
                    h_orig.get(i, j),
                );
            }
        }
    }

    // ── Test 9: GivensSequence to_matrix is orthogonal ──────────────────

    #[test]
    fn test_givens_sequence_orthogonal() {
        let n = 5;
        let mut seq = GivensSequence::new();
        // Build several rotations.
        seq.push(GivensRotation::new(0, 1, 3.0, 4.0));
        seq.push(GivensRotation::new(1, 2, 1.0, 2.0));
        seq.push(GivensRotation::new(2, 3, 5.0, -1.0));
        seq.push(GivensRotation::new(3, 4, 2.0, 3.0));

        let q = seq.to_matrix(n);
        check_orthogonal(&q, 1e-12);
    }

    // ── Test 10: tridiagonal QR step preserves eigenvalues ──────────────

    #[test]
    fn test_tridiagonal_step_preserves_eigenvalues() {
        // Tridiagonal matrix with known eigenvalues.
        let mut alpha = vec![2.0, 3.0, 1.0, 4.0];
        let mut beta = vec![1.0, 0.5, 0.7];

        // Compute initial trace and squared-Frobenius (both are eigenvalue
        // invariants for symmetric matrices).
        let trace_before: f64 = alpha.iter().sum();
        let frob_sq_before: f64 = alpha.iter().map(|x| x * x).sum::<f64>()
            + 2.0 * beta.iter().map(|x| x * x).sum::<f64>();

        let shift = wilkinson_shift(&alpha, &beta, 3);
        let _rots = qr_tridiagonal_step(&mut alpha, &mut beta, 0, 3, shift);

        let trace_after: f64 = alpha.iter().sum();
        let frob_sq_after: f64 = alpha.iter().map(|x| x * x).sum::<f64>()
            + 2.0 * beta.iter().map(|x| x * x).sum::<f64>();

        assert!(
            approx_eq(trace_before, trace_after, 1e-10),
            "trace: {} vs {}",
            trace_before,
            trace_after,
        );
        assert!(
            approx_eq(frob_sq_before, frob_sq_after, 1e-10),
            "Frobenius²: {} vs {}",
            frob_sq_before,
            frob_sq_after,
        );
    }

    // ── Test 11: full tridiagonal QR eigenvalue solver ──────────────────

    #[test]
    fn test_tridiagonal_qr_eigen_full() {
        // T = diag(2, 3, 1) with off-diag (1, 0.5).
        // Eigenvalues can be verified by characteristic polynomial.
        let mut alpha = vec![2.0, 3.0, 1.0];
        let mut beta = vec![1.0, 0.5];
        let mut q = DenseMatrix::eye(3);

        let iters = tridiagonal_qr_eigen(
            &mut alpha,
            &mut beta,
            Some(&mut q),
            200,
            1e-14,
        )
        .expect("should converge");

        assert!(iters < 200, "iterations = {}", iters);

        // Off-diagonals should be zero.
        for b in &beta {
            assert!(b.abs() < 1e-12, "beta not zero: {}", b);
        }

        // Q should be orthogonal.
        check_orthogonal(&q, 1e-10);

        // Eigenvalues should sum to trace = 6.
        let eig_sum: f64 = alpha.iter().sum();
        assert!(approx_eq(eig_sum, 6.0, 1e-10), "eigenvalue sum = {}", eig_sum);

        // Eigenvalues product of squares should match.
        let eig_sq: f64 = alpha.iter().map(|x| x * x).sum();
        let expected_sq: f64 = 4.0 + 9.0 + 1.0 + 2.0 * (1.0 + 0.25); // diag² + 2*offdiag²
        assert!(
            approx_eq(eig_sq, expected_sq, 1e-10),
            "eig sq sum {} vs {}",
            eig_sq,
            expected_sq,
        );
    }

    // ── Test 12: Hessenberg QR step preserves eigenvalues ───────────────

    #[test]
    fn test_hessenberg_qr_step_eigenvalues() {
        let mut h = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                4.0, 1.0, 2.0,
                3.0, 5.0, 1.0,
                0.0, 2.0, 6.0,
            ],
        );
        let trace_before: f64 = (0..3).map(|i| h.get(i, i)).sum();

        let shift = h.get(2, 2); // Rayleigh shift
        let _seq = qr_hessenberg_step(&mut h, shift);

        let trace_after: f64 = (0..3).map(|i| h.get(i, i)).sum();
        assert!(
            approx_eq(trace_before, trace_after, 1e-10),
            "trace: {} vs {}",
            trace_before,
            trace_after,
        );

        // Result should still be upper Hessenberg.
        assert!(
            h.get(2, 0).abs() < 1e-10,
            "H[2,0] = {} (should be ~0)",
            h.get(2, 0)
        );
    }

    // ── Test 13: fast Givens basic ──────────────────────────────────────

    #[test]
    fn test_fast_givens_basic() {
        let d1 = 1.0;
        let d2 = 1.0;
        let a = 3.0;
        let b = 4.0;

        let fg = FastGivens::compute(d1, d2, a, b);

        // The resultant should be consistent: |r| ≈ hypot(a, b) up to
        // scaling.
        // For d1 = d2 = 1 the fast Givens degenerates to a scaled standard
        // rotation.  Verify that the rotation zeroes b:
        //   [1  alpha] [a]   [a + alpha*b]   [r]
        //   [-beta  1] [b] = [-beta*a + b] = [0]  (or swapped form)
        if !fg.swapped {
            let zeroed = -fg.beta * a + b;
            assert!(
                approx_eq(zeroed, 0.0, 1e-10),
                "zeroed = {}",
                zeroed,
            );
        } else {
            let zeroed = -a + fg.beta * b;
            assert!(
                approx_eq(zeroed, 0.0, 1e-10),
                "zeroed = {}",
                zeroed,
            );
        }

        // d1, d2 should remain positive.
        assert!(fg.d1 > 0.0, "d1 = {}", fg.d1);
        assert!(fg.d2 > 0.0, "d2 = {}", fg.d2);
    }

    // ── Test 14: fast Givens with different scales ──────────────────────

    #[test]
    fn test_fast_givens_scaled() {
        let d1 = 2.0;
        let d2 = 0.5;
        let a = 1.0;
        let b = 3.0;

        let fg = FastGivens::compute(d1, d2, a, b);

        // Verify the zero-out property:
        // For type 1: M [a; b] = [a + alpha*b; -beta*a + b] and the second
        // component should be zero.  For type 2: M [a; b] =
        // [alpha*a + b; -a + beta*b] and the first or second should be zero.
        if !fg.swapped {
            let zeroed = -fg.beta * a + b;
            assert!(
                approx_eq(zeroed, 0.0, 1e-10),
                "type 1 zeroed = {}",
                zeroed,
            );
        } else {
            let zeroed = -a + fg.beta * b;
            assert!(
                approx_eq(zeroed, 0.0, 1e-10),
                "type 2 zeroed = {}",
                zeroed,
            );
        }

        // d1, d2 should remain positive.
        assert!(fg.d1 > 0.0, "d1 = {}", fg.d1);
        assert!(fg.d2 > 0.0, "d2 = {}", fg.d2);
    }

    // ── Test 15: givens_matrix helper ───────────────────────────────────

    #[test]
    fn test_givens_matrix_orthogonal() {
        let rot = GivensRotation::new(1, 3, 2.0, -7.0);
        let g = givens_matrix(&rot, 5);
        check_orthogonal(&g, EPS);
    }

    // ── Test 16: round-trip left then right transpose = identity ────────

    #[test]
    fn test_sequence_roundtrip() {
        let n = 4;
        let orig = DenseMatrix::from_row_major(
            n,
            n,
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
        );
        let mut mat = orig.clone();

        let mut seq = GivensSequence::new();
        seq.push(GivensRotation::new(0, 1, 1.0, 2.0));
        seq.push(GivensRotation::new(2, 3, 3.0, 1.0));

        // Apply Q^T from the left, then Q from the left (undo).
        seq.apply_left_all(&mut mat);
        seq.apply_left_transpose(&mut mat);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(mat.get(i, j), orig.get(i, j), 1e-11),
                    "roundtrip [{},{}]: {} vs {}",
                    i,
                    j,
                    mat.get(i, j),
                    orig.get(i, j),
                );
            }
        }
    }
}
