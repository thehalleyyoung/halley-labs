//! Matrix equilibration and scaling for numerical stability.
//!
//! Provides Ruiz iterative scaling, geometric-mean scaling, and a SCIP-native
//! power-of-2 emulation.  Each method returns a [`ScalingResult`] containing
//! the scaled matrix together with the accumulated row/column scaling vectors.

use serde::{Deserialize, Serialize};
use spectral_types::sparse::CsrMatrix;



// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which equilibration strategy to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// Ruiz iterative row/column balancing (default).
    Ruiz,
    /// Geometric-mean scaling of rows and columns.
    GeometricMean,
    /// SCIP-style power-of-2 scaling.
    ScipNative,
    /// No scaling at all — pass the matrix through unchanged.
    None,
}

impl Default for ScalingMethod {
    fn default() -> Self {
        Self::Ruiz
    }
}

/// Outcome of a scaling operation.
#[derive(Debug, Clone)]
pub struct ScalingResult {
    pub scaled_matrix: CsrMatrix<f64>,
    pub row_scaling: Vec<f64>,
    pub col_scaling: Vec<f64>,
    pub iterations: usize,
    pub condition_before: f64,
    pub condition_after: f64,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RUIZ_MAX_ITER: usize = 20;
const RUIZ_TOL: f64 = 1e-6;
const ZERO_GUARD: f64 = 1e-300;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Scale `matrix` using the chosen [`ScalingMethod`].
pub fn scale_matrix(matrix: &CsrMatrix<f64>, method: ScalingMethod) -> ScalingResult {
    match method {
        ScalingMethod::Ruiz => ruiz_scaling(matrix),
        ScalingMethod::GeometricMean => geometric_mean_scaling(matrix),
        ScalingMethod::ScipNative => scip_native_scaling(matrix),
        ScalingMethod::None => no_scaling(matrix),
    }
}

// ---------------------------------------------------------------------------
// Ruiz iterative scaling
// ---------------------------------------------------------------------------

fn ruiz_scaling(matrix: &CsrMatrix<f64>) -> ScalingResult {
    let (m, n) = matrix.shape();
    let condition_before = estimate_condition_number(matrix);

    // Accumulate total scaling: D_r and D_c.
    let mut dr = vec![1.0f64; m];
    let mut dc = vec![1.0f64; n];

    // Work on a mutable copy of the values.
    let row_ptr = matrix.row_ptr.clone();
    let col_ind = matrix.col_ind.clone();
    let mut values = matrix.values.clone();

    let mut iters = 0usize;
    for _ in 0..RUIZ_MAX_ITER {
        iters += 1;

        // --- row scaling ---
        let mut converged = true;
        for i in 0..m {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            let mut max_abs = 0.0f64;
            for idx in start..end {
                let a = values[idx].abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            if max_abs < ZERO_GUARD {
                continue;
            }
            let r = 1.0 / max_abs.sqrt();
            if (r - 1.0).abs() > RUIZ_TOL {
                converged = false;
            }
            for idx in start..end {
                values[idx] *= r;
            }
            dr[i] *= r;
        }

        // --- column scaling ---
        let mut col_max = vec![0.0f64; n];
        for i in 0..m {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            for idx in start..end {
                let j = col_ind[idx];
                let a = values[idx].abs();
                if a > col_max[j] {
                    col_max[j] = a;
                }
            }
        }
        for j in 0..n {
            if col_max[j] < ZERO_GUARD {
                continue;
            }
            let c = 1.0 / col_max[j].sqrt();
            if (c - 1.0).abs() > RUIZ_TOL {
                converged = false;
            }
            dc[j] *= c;
            // Apply column scaling to all entries in column j.
            for i in 0..m {
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                for idx in start..end {
                    if col_ind[idx] == j {
                        values[idx] *= c;
                    }
                }
            }
        }

        if converged {
            log::debug!("Ruiz scaling converged in {iters} iterations");
            break;
        }
    }

    let scaled = CsrMatrix::new(m, n, row_ptr, col_ind, values)
        .expect("Ruiz: scaled matrix construction must succeed");
    let condition_after = estimate_condition_number(&scaled);

    log::info!(
        "Ruiz scaling: cond {condition_before:.2e} -> {condition_after:.2e} in {iters} iters"
    );

    ScalingResult {
        scaled_matrix: scaled,
        row_scaling: dr,
        col_scaling: dc,
        iterations: iters,
        condition_before,
        condition_after,
    }
}

// ---------------------------------------------------------------------------
// Geometric mean scaling
// ---------------------------------------------------------------------------

fn geometric_mean_scaling(matrix: &CsrMatrix<f64>) -> ScalingResult {
    let (m, n) = matrix.shape();
    let condition_before = estimate_condition_number(matrix);

    // Row geometric means of |a_ij| (nonzeros only).
    let mut row_scale = vec![1.0f64; m];
    for i in 0..m {
        let vals = matrix.row_values(i);
        let nnz = vals.len();
        if nnz == 0 {
            continue;
        }
        // log-space geometric mean to avoid overflow.
        let log_sum: f64 = vals.iter().map(|v| v.abs().max(ZERO_GUARD).ln()).sum();
        let gm = (log_sum / nnz as f64).exp();
        if gm > ZERO_GUARD {
            row_scale[i] = 1.0 / gm;
        }
    }

    // Apply row scaling, then compute column geometric means.
    let mut values = matrix.values.clone();
    for i in 0..m {
        let start = matrix.row_ptr[i];
        let end = matrix.row_ptr[i + 1];
        for idx in start..end {
            values[idx] *= row_scale[i];
        }
    }

    let mut col_scale = vec![1.0f64; n];
    let mut col_log_sum = vec![0.0f64; n];
    let mut col_nnz = vec![0usize; n];
    for i in 0..m {
        let start = matrix.row_ptr[i];
        let end = matrix.row_ptr[i + 1];
        for idx in start..end {
            let j = matrix.col_ind[idx];
            col_log_sum[j] += values[idx].abs().max(ZERO_GUARD).ln();
            col_nnz[j] += 1;
        }
    }
    for j in 0..n {
        if col_nnz[j] == 0 {
            continue;
        }
        let gm = (col_log_sum[j] / col_nnz[j] as f64).exp();
        if gm > ZERO_GUARD {
            col_scale[j] = 1.0 / gm;
        }
    }

    // Apply column scaling.
    for i in 0..m {
        let start = matrix.row_ptr[i];
        let end = matrix.row_ptr[i + 1];
        for idx in start..end {
            let j = matrix.col_ind[idx];
            values[idx] *= col_scale[j];
        }
    }

    let scaled = CsrMatrix::new(
        m,
        n,
        matrix.row_ptr.clone(),
        matrix.col_ind.clone(),
        values,
    )
    .expect("GeometricMean: scaled matrix construction must succeed");
    let condition_after = estimate_condition_number(&scaled);

    log::info!(
        "GeometricMean scaling: cond {condition_before:.2e} -> {condition_after:.2e}"
    );

    ScalingResult {
        scaled_matrix: scaled,
        row_scaling: row_scale,
        col_scaling: col_scale,
        iterations: 1,
        condition_before,
        condition_after,
    }
}

// ---------------------------------------------------------------------------
// SCIP-native (power-of-2) scaling
// ---------------------------------------------------------------------------

fn scip_native_scaling(matrix: &CsrMatrix<f64>) -> ScalingResult {
    let (m, n) = matrix.shape();
    let condition_before = estimate_condition_number(matrix);

    // Compute row inf-norms, round scaling factors to nearest power of 2.
    let mut row_scale = vec![1.0f64; m];
    for i in 0..m {
        let vals = matrix.row_values(i);
        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if max_abs > ZERO_GUARD {
            let raw = 1.0 / max_abs;
            row_scale[i] = round_to_power_of_2(raw);
        }
    }

    // Apply row scaling.
    let mut values = matrix.values.clone();
    for i in 0..m {
        let start = matrix.row_ptr[i];
        let end = matrix.row_ptr[i + 1];
        for idx in start..end {
            values[idx] *= row_scale[i];
        }
    }

    // Column inf-norms on the row-scaled matrix.
    let mut col_max = vec![0.0f64; n];
    for i in 0..m {
        let start = matrix.row_ptr[i];
        let end = matrix.row_ptr[i + 1];
        for idx in start..end {
            let j = matrix.col_ind[idx];
            let a = values[idx].abs();
            if a > col_max[j] {
                col_max[j] = a;
            }
        }
    }
    let mut col_scale = vec![1.0f64; n];
    for j in 0..n {
        if col_max[j] > ZERO_GUARD {
            let raw = 1.0 / col_max[j];
            col_scale[j] = round_to_power_of_2(raw);
        }
    }

    // Apply column scaling.
    for i in 0..m {
        let start = matrix.row_ptr[i];
        let end = matrix.row_ptr[i + 1];
        for idx in start..end {
            let j = matrix.col_ind[idx];
            values[idx] *= col_scale[j];
        }
    }

    let scaled = CsrMatrix::new(
        m,
        n,
        matrix.row_ptr.clone(),
        matrix.col_ind.clone(),
        values,
    )
    .expect("ScipNative: scaled matrix construction must succeed");
    let condition_after = estimate_condition_number(&scaled);

    log::info!(
        "SCIP-native scaling: cond {condition_before:.2e} -> {condition_after:.2e}"
    );

    ScalingResult {
        scaled_matrix: scaled,
        row_scaling: row_scale,
        col_scaling: col_scale,
        iterations: 1,
        condition_before,
        condition_after,
    }
}

// ---------------------------------------------------------------------------
// No-op passthrough
// ---------------------------------------------------------------------------

fn no_scaling(matrix: &CsrMatrix<f64>) -> ScalingResult {
    let (m, n) = matrix.shape();
    let cond = estimate_condition_number(matrix);
    ScalingResult {
        scaled_matrix: matrix.clone(),
        row_scaling: vec![1.0; m],
        col_scaling: vec![1.0; n],
        iterations: 0,
        condition_before: cond,
        condition_after: cond,
    }
}

// ---------------------------------------------------------------------------
// Public utilities
// ---------------------------------------------------------------------------

/// Estimate the condition number ‖A‖∞ · ‖A⁻¹‖∞ using row/column inf-norms
/// as a cheap proxy (ratio of max to min nonzero row norms).
pub fn estimate_condition_number(matrix: &CsrMatrix<f64>) -> f64 {
    let rnorms = row_norms(matrix);
    let mut max_norm = 0.0f64;
    let mut min_norm = f64::MAX;
    for &r in &rnorms {
        if r > ZERO_GUARD {
            if r > max_norm {
                max_norm = r;
            }
            if r < min_norm {
                min_norm = r;
            }
        }
    }
    if min_norm >= f64::MAX || max_norm < ZERO_GUARD {
        return 1.0;
    }
    max_norm / min_norm
}

/// Reverse-apply scaling: `out[i] = v[i] / scaling[i]`.
pub fn unscale_vector(v: &[f64], scaling: &[f64]) -> Vec<f64> {
    v.iter()
        .zip(scaling.iter())
        .map(|(&vi, &si)| {
            if si.abs() < ZERO_GUARD {
                vi
            } else {
                vi / si
            }
        })
        .collect()
}

/// L∞ norm (max absolute value) of each row.
pub fn row_norms(matrix: &CsrMatrix<f64>) -> Vec<f64> {
    let (m, _) = matrix.shape();
    let mut norms = Vec::with_capacity(m);
    for i in 0..m {
        let vals = matrix.row_values(i);
        let mx = vals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        norms.push(mx);
    }
    norms
}

/// L∞ norm (max absolute value) of each column.
pub fn col_norms(matrix: &CsrMatrix<f64>) -> Vec<f64> {
    let (m, n) = matrix.shape();
    let mut norms = vec![0.0f64; n];
    for i in 0..m {
        let indices = matrix.row_indices(i);
        let vals = matrix.row_values(i);
        for (k, &j) in indices.iter().enumerate() {
            let a = vals[k].abs();
            if a > norms[j] {
                norms[j] = a;
            }
        }
    }
    norms
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Round a positive value to the nearest power of 2.
fn round_to_power_of_2(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return 1.0;
    }
    let log2 = x.log2();
    let exp = log2.round() as i32;
    2.0f64.powi(exp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small CSR matrix from dense row-major data.
    fn csr_from_dense(rows: usize, cols: usize, data: &[f64]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let v = data[i * cols + j];
                if v.abs() > 1e-15 {
                    coo.push(i, j, v);
                }
            }
        }
        coo.to_csr()
    }

    #[test]
    fn test_ruiz_identity_is_noop() {
        let eye = CsrMatrix::<f64>::identity(4);
        let res = scale_matrix(&eye, ScalingMethod::Ruiz);
        // Identity should be unchanged (already balanced).
        for i in 0..4 {
            assert!((res.row_scaling[i] - 1.0).abs() < 1e-6);
            assert!((res.col_scaling[i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ruiz_reduces_condition() {
        // Ill-conditioned: row norms differ by 1e6.
        let mat = csr_from_dense(2, 2, &[1e6, 0.0, 0.0, 1.0]);
        let res = scale_matrix(&mat, ScalingMethod::Ruiz);
        assert!(res.condition_after < res.condition_before);
    }

    #[test]
    fn test_geometric_mean_basic() {
        let mat = csr_from_dense(2, 2, &[4.0, 0.0, 0.0, 9.0]);
        let res = scale_matrix(&mat, ScalingMethod::GeometricMean);
        assert_eq!(res.iterations, 1);
        assert!(res.condition_after >= 1.0);
    }

    #[test]
    fn test_scip_native_power_of_two() {
        let mat = csr_from_dense(2, 2, &[3.0, 0.0, 0.0, 5.0]);
        let res = scale_matrix(&mat, ScalingMethod::ScipNative);
        // Scaling factors should be powers of 2.
        for &r in &res.row_scaling {
            let log2 = r.log2();
            assert!(
                (log2 - log2.round()).abs() < 1e-12,
                "row scaling {r} is not a power of 2"
            );
        }
        for &c in &res.col_scaling {
            let log2 = c.log2();
            assert!(
                (log2 - log2.round()).abs() < 1e-12,
                "col scaling {c} is not a power of 2"
            );
        }
    }

    #[test]
    fn test_no_scaling_passthrough() {
        let mat = csr_from_dense(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let res = scale_matrix(&mat, ScalingMethod::None);
        assert_eq!(res.iterations, 0);
        assert_eq!(res.row_scaling, vec![1.0, 1.0]);
        assert_eq!(res.col_scaling, vec![1.0, 1.0]);
        // Values should be identical.
        assert_eq!(res.scaled_matrix.values, mat.values);
    }

    #[test]
    fn test_estimate_condition_number_identity() {
        let eye = CsrMatrix::<f64>::identity(5);
        let cond = estimate_condition_number(&eye);
        assert!((cond - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_condition_number_ill_conditioned() {
        let mat = csr_from_dense(2, 2, &[1e8, 0.0, 0.0, 1e-2]);
        let cond = estimate_condition_number(&mat);
        assert!(cond > 1e9);
    }

    #[test]
    fn test_unscale_vector_basic() {
        let v = vec![2.0, 6.0, 9.0];
        let s = vec![2.0, 3.0, 3.0];
        let out = unscale_vector(&v, &s);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_unscale_vector_zero_guard() {
        let v = vec![5.0];
        let s = vec![0.0];
        let out = unscale_vector(&v, &s);
        assert_eq!(out[0], 5.0);
    }

    #[test]
    fn test_row_norms() {
        let mat = csr_from_dense(2, 3, &[1.0, -3.0, 2.0, 0.0, 4.0, 0.0]);
        let rn = row_norms(&mat);
        assert!((rn[0] - 3.0).abs() < 1e-12);
        assert!((rn[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_col_norms() {
        let mat = csr_from_dense(2, 3, &[1.0, -3.0, 2.0, 0.0, 4.0, -5.0]);
        let cn = col_norms(&mat);
        assert!((cn[0] - 1.0).abs() < 1e-12);
        assert!((cn[1] - 4.0).abs() < 1e-12);
        assert!((cn[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_round_to_power_of_2() {
        assert!((round_to_power_of_2(1.0) - 1.0).abs() < 1e-15);
        assert!((round_to_power_of_2(3.0) - 4.0).abs() < 1e-15);
        assert!((round_to_power_of_2(0.3) - 0.25).abs() < 1e-15);
        assert!((round_to_power_of_2(7.0) - 8.0).abs() < 1e-15);
    }

    #[test]
    fn test_ruiz_empty_matrix() {
        let mat = CsrMatrix::<f64>::zeros(0, 0);
        let res = scale_matrix(&mat, ScalingMethod::Ruiz);
        assert!(res.row_scaling.is_empty());
        assert!(res.col_scaling.is_empty());
    }
}
