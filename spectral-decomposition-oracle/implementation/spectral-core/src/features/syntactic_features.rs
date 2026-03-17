//! Syntactic feature extraction from MIP instances and constraint matrices.
//!
//! Computes the 25 syntactic features defined by [`SyntacticFeatures`] from
//! either a full [`MipInstance`] or a bare constraint matrix.

use log::debug;
use spectral_types::features::SyntacticFeatures;
use spectral_types::mip::{ConstraintSense, MipInstance, VariableType};
use spectral_types::sparse::CsrMatrix;

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Local helper structs
// ---------------------------------------------------------------------------

/// Row-degree statistics for a CSR matrix.
#[derive(Debug, Clone, Copy)]
struct RowDegreeStats {
    min: f64,
    max: f64,
    avg: f64,
    std: f64,
}

/// Column-degree statistics for a CSR matrix (computed via CSC conversion).
#[derive(Debug, Clone, Copy)]
struct ColDegreeStats {
    min: f64,
    max: f64,
    avg: f64,
    std: f64,
}

/// Coefficient statistics derived from nonzero magnitudes.
#[derive(Debug, Clone, Copy)]
struct CoefficientStats {
    range: f64,
    mean: f64,
    std: f64,
}

// ---------------------------------------------------------------------------
// Main entry points
// ---------------------------------------------------------------------------

/// Compute all 25 syntactic features from a [`MipInstance`].
///
/// Returns an error if the instance contains an empty constraint matrix
/// (zero rows **and** zero columns).
pub fn compute_syntactic_features(mip: &MipInstance) -> Result<SyntacticFeatures> {
    let matrix = &mip.constraint_matrix;
    let (rows, cols) = matrix.shape();

    if rows == 0 && cols == 0 {
        return Err(SpectralCoreError::empty_input(
            "constraint matrix has zero rows and zero columns",
        ));
    }

    debug!(
        "Computing syntactic features for {}×{} matrix ({} nnz)",
        rows,
        cols,
        matrix.nnz()
    );

    let num_variables = cols as f64;
    let num_constraints = rows as f64;
    let num_nonzeros = matrix.nnz() as f64;
    let density = compute_density(rows, cols, matrix.nnz());

    let row_stats = compute_row_degree_stats(matrix);
    let col_stats = compute_col_degree_stats(matrix);
    let coeff_stats = compute_coefficient_stats(matrix);

    let (_, _, rhs_range) = compute_rhs_stats(&mip.rhs);
    let (_, _, obj_range) = compute_obj_stats(&mip.objective);

    let (frac_binary, frac_integer, frac_continuous) =
        compute_variable_type_fractions(&mip.var_types);
    let (frac_equality, frac_inequality) = compute_constraint_type_fractions(&mip.senses);

    let variable_bound_tightness =
        compute_bound_tightness(&mip.lower_bounds, &mip.upper_bounds);

    let constraint_matrix_rank_estimate = estimate_rank(matrix);
    let constraint_redundancy_estimate = estimate_row_similarity(matrix, 200);

    Ok(SyntacticFeatures {
        num_variables,
        num_constraints,
        num_nonzeros,
        density,
        constraint_matrix_rank_estimate,
        avg_row_nnz: row_stats.avg,
        max_row_nnz: row_stats.max,
        min_row_nnz: row_stats.min,
        std_row_nnz: row_stats.std,
        avg_col_nnz: col_stats.avg,
        max_col_nnz: col_stats.max,
        min_col_nnz: col_stats.min,
        std_col_nnz: col_stats.std,
        coeff_range: coeff_stats.range,
        coeff_mean: coeff_stats.mean,
        coeff_std: coeff_stats.std,
        rhs_range,
        obj_range,
        frac_binary,
        frac_integer,
        frac_continuous,
        frac_equality,
        frac_inequality,
        variable_bound_tightness,
        constraint_redundancy_estimate,
    })
}

/// Compute syntactic features from a bare constraint matrix, without
/// MIP-specific information.
///
/// Variable-type fractions default to 100% continuous.  Constraint-type
/// fractions default to 100% inequality (≤).  Bound tightness defaults to
/// 0.0.  RHS / objective ranges default to 0.0.
pub fn compute_syntactic_features_from_matrix(matrix: &CsrMatrix<f64>) -> SyntacticFeatures {
    let (rows, cols) = matrix.shape();
    let num_variables = cols as f64;
    let num_constraints = rows as f64;
    let num_nonzeros = matrix.nnz() as f64;
    let density = compute_density(rows, cols, matrix.nnz());

    let row_stats = compute_row_degree_stats(matrix);
    let col_stats = compute_col_degree_stats(matrix);
    let coeff_stats = compute_coefficient_stats(matrix);

    let constraint_matrix_rank_estimate = estimate_rank(matrix);
    let constraint_redundancy_estimate = estimate_row_similarity(matrix, 200);

    SyntacticFeatures {
        num_variables,
        num_constraints,
        num_nonzeros,
        density,
        constraint_matrix_rank_estimate,
        avg_row_nnz: row_stats.avg,
        max_row_nnz: row_stats.max,
        min_row_nnz: row_stats.min,
        std_row_nnz: row_stats.std,
        avg_col_nnz: col_stats.avg,
        max_col_nnz: col_stats.max,
        min_col_nnz: col_stats.min,
        std_col_nnz: col_stats.std,
        coeff_range: coeff_stats.range,
        coeff_mean: coeff_stats.mean,
        coeff_std: coeff_stats.std,
        rhs_range: 0.0,
        obj_range: 0.0,
        frac_binary: 0.0,
        frac_integer: 0.0,
        frac_continuous: 1.0,
        frac_equality: 0.0,
        frac_inequality: 1.0,
        variable_bound_tightness: 0.0,
        constraint_redundancy_estimate,
    }
}

// ---------------------------------------------------------------------------
// Density
// ---------------------------------------------------------------------------

/// Compute matrix density, guarding against zero-size matrices.
fn compute_density(rows: usize, cols: usize, nnz: usize) -> f64 {
    let total = rows as f64 * cols as f64;
    if total == 0.0 {
        0.0
    } else {
        nnz as f64 / total
    }
}

// ---------------------------------------------------------------------------
// Row degree statistics
// ---------------------------------------------------------------------------

/// Compute min, max, mean, and standard deviation of row degrees.
fn compute_row_degree_stats(matrix: &CsrMatrix<f64>) -> RowDegreeStats {
    let rows = matrix.rows;
    if rows == 0 {
        return RowDegreeStats {
            min: 0.0,
            max: 0.0,
            avg: 0.0,
            std: 0.0,
        };
    }

    let mut min_deg = usize::MAX;
    let mut max_deg = 0usize;
    let mut sum = 0usize;
    let mut sum_sq = 0.0f64;

    for r in 0..rows {
        let deg = matrix.row_nnz(r);
        if deg < min_deg {
            min_deg = deg;
        }
        if deg > max_deg {
            max_deg = deg;
        }
        sum += deg;
        sum_sq += (deg as f64) * (deg as f64);
    }

    let n = rows as f64;
    let avg = sum as f64 / n;
    let variance = (sum_sq / n) - (avg * avg);
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    RowDegreeStats {
        min: min_deg as f64,
        max: max_deg as f64,
        avg,
        std,
    }
}

// ---------------------------------------------------------------------------
// Column degree statistics
// ---------------------------------------------------------------------------

/// Compute min, max, mean, and standard deviation of column degrees.
///
/// Internally converts the matrix to CSC format to obtain per-column counts
/// efficiently.
fn compute_col_degree_stats(matrix: &CsrMatrix<f64>) -> ColDegreeStats {
    let cols = matrix.cols;
    if cols == 0 {
        return ColDegreeStats {
            min: 0.0,
            max: 0.0,
            avg: 0.0,
            std: 0.0,
        };
    }

    let csc = matrix.to_csc();

    let mut min_deg = usize::MAX;
    let mut max_deg = 0usize;
    let mut sum = 0usize;
    let mut sum_sq = 0.0f64;

    for c in 0..cols {
        let deg = csc.col_nnz(c);
        if deg < min_deg {
            min_deg = deg;
        }
        if deg > max_deg {
            max_deg = deg;
        }
        sum += deg;
        sum_sq += (deg as f64) * (deg as f64);
    }

    let n = cols as f64;
    let avg = sum as f64 / n;
    let variance = (sum_sq / n) - (avg * avg);
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    ColDegreeStats {
        min: min_deg as f64,
        max: max_deg as f64,
        avg,
        std,
    }
}

// ---------------------------------------------------------------------------
// Coefficient statistics
// ---------------------------------------------------------------------------

/// Compute range, mean, and standard deviation of absolute coefficient values.
///
/// Only nonzero entries are considered.  If the matrix has no nonzeros, all
/// statistics are zero.
fn compute_coefficient_stats(matrix: &CsrMatrix<f64>) -> CoefficientStats {
    let nnz = matrix.nnz();
    if nnz == 0 {
        return CoefficientStats {
            range: 0.0,
            mean: 0.0,
            std: 0.0,
        };
    }

    let mut abs_min = f64::MAX;
    let mut abs_max = 0.0f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for r in 0..matrix.rows {
        for &v in matrix.row_values(r) {
            let a = v.abs();
            if a < abs_min {
                abs_min = a;
            }
            if a > abs_max {
                abs_max = a;
            }
            sum += a;
            sum_sq += a * a;
        }
    }

    let n = nnz as f64;
    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    CoefficientStats {
        range: abs_max - abs_min,
        mean,
        std,
    }
}

// ---------------------------------------------------------------------------
// RHS / objective statistics
// ---------------------------------------------------------------------------

/// Compute min, max, and range of absolute RHS values.
///
/// Returns `(0.0, 0.0, 0.0)` for an empty slice.
fn compute_rhs_stats(rhs: &[f64]) -> (f64, f64, f64) {
    abs_range_stats(rhs)
}

/// Compute min, max, and range of absolute objective coefficients.
///
/// Returns `(0.0, 0.0, 0.0)` for an empty slice.
fn compute_obj_stats(obj: &[f64]) -> (f64, f64, f64) {
    abs_range_stats(obj)
}

/// Shared helper: min, max, and range over |values|.
fn abs_range_stats(vals: &[f64]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut lo = f64::MAX;
    let mut hi = 0.0f64;
    for &v in vals {
        let a = v.abs();
        if a < lo {
            lo = a;
        }
        if a > hi {
            hi = a;
        }
    }
    (lo, hi, hi - lo)
}

// ---------------------------------------------------------------------------
// Variable type fractions
// ---------------------------------------------------------------------------

/// Compute the fraction of binary, integer, and continuous variables.
///
/// Returns `(0.0, 0.0, 0.0)` when the slice is empty.
fn compute_variable_type_fractions(var_types: &[VariableType]) -> (f64, f64, f64) {
    let n = var_types.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut n_bin = 0usize;
    let mut n_int = 0usize;
    let mut n_cont = 0usize;
    for vt in var_types {
        match vt {
            VariableType::Binary => n_bin += 1,
            VariableType::Integer => n_int += 1,
            VariableType::Continuous => n_cont += 1,
        }
    }
    let total = n as f64;
    (n_bin as f64 / total, n_int as f64 / total, n_cont as f64 / total)
}

// ---------------------------------------------------------------------------
// Constraint type fractions
// ---------------------------------------------------------------------------

/// Compute the fraction of equality and inequality constraints.
///
/// Returns `(0.0, 0.0)` when the slice is empty.
fn compute_constraint_type_fractions(senses: &[ConstraintSense]) -> (f64, f64) {
    let n = senses.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut n_eq = 0usize;
    let mut n_ineq = 0usize;
    for s in senses {
        match s {
            ConstraintSense::Eq => n_eq += 1,
            ConstraintSense::Le | ConstraintSense::Ge => n_ineq += 1,
        }
    }
    let total = n as f64;
    (n_eq as f64 / total, n_ineq as f64 / total)
}

// ---------------------------------------------------------------------------
// Bound tightness
// ---------------------------------------------------------------------------

/// Fraction of variables whose upper bound is finite.
///
/// A bound is considered finite if it is strictly less than [`f64::MAX`] and
/// not infinite/NaN.
fn compute_bound_tightness(lower: &[f64], upper: &[f64]) -> f64 {
    let n = upper.len().max(lower.len());
    if n == 0 {
        return 0.0;
    }
    let finite_count = upper
        .iter()
        .filter(|&&u| u.is_finite() && u < f64::MAX)
        .count();
    finite_count as f64 / n as f64
}

// ---------------------------------------------------------------------------
// Rank estimate
// ---------------------------------------------------------------------------

/// Heuristic rank estimate: ratio of rows with at least one nonzero to
/// `min(rows, cols)`.
fn estimate_rank(matrix: &CsrMatrix<f64>) -> f64 {
    let (rows, cols) = matrix.shape();
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let min_dim = rows.min(cols) as f64;
    let nonzero_rows = (0..rows)
        .filter(|&r| matrix.row_nnz(r) > 0)
        .count() as f64;
    (nonzero_rows / min_dim).min(1.0)
}

// ---------------------------------------------------------------------------
// Row similarity (redundancy estimate)
// ---------------------------------------------------------------------------

/// Estimate constraint redundancy by sampling pairs of rows and computing
/// their cosine similarity.
///
/// Returns the fraction of sampled pairs whose cosine similarity exceeds a
/// threshold (0.99).  A higher value indicates more near-duplicate rows.
///
/// When `sample_size` is 0 or the matrix has fewer than 2 rows, returns 0.
fn estimate_row_similarity(matrix: &CsrMatrix<f64>, sample_size: usize) -> f64 {
    let rows = matrix.rows;
    if rows < 2 || sample_size == 0 {
        return 0.0;
    }

    const SIMILARITY_THRESHOLD: f64 = 0.99;

    // Deterministic sampling: pick pairs using a simple stride scheme.
    let max_pairs = rows * (rows - 1) / 2;
    let effective_samples = sample_size.min(max_pairs);

    let mut similar_count = 0usize;
    let mut pair_index = 0usize;
    let stride = if max_pairs > effective_samples {
        max_pairs / effective_samples
    } else {
        1
    };

    let mut sampled = 0usize;
    // Enumerate pairs (i, j) with i < j in triangular order.
    'outer: for i in 0..rows {
        for j in (i + 1)..rows {
            if pair_index % stride == 0 {
                let sim = cosine_similarity_rows(matrix, i, j);
                if sim >= SIMILARITY_THRESHOLD {
                    similar_count += 1;
                }
                sampled += 1;
                if sampled >= effective_samples {
                    break 'outer;
                }
            }
            pair_index += 1;
        }
    }

    if sampled == 0 {
        0.0
    } else {
        similar_count as f64 / sampled as f64
    }
}

/// Cosine similarity between two rows of a CSR matrix.
///
/// Returns 0.0 if either row is all-zero.
fn cosine_similarity_rows(matrix: &CsrMatrix<f64>, row_a: usize, row_b: usize) -> f64 {
    let indices_a = matrix.row_indices(row_a);
    let values_a = matrix.row_values(row_a);
    let indices_b = matrix.row_indices(row_b);
    let values_b = matrix.row_values(row_b);

    if indices_a.is_empty() || indices_b.is_empty() {
        return 0.0;
    }

    // Merge-join on sorted column indices.
    let mut dot = 0.0f64;
    let mut norm_a_sq = 0.0f64;
    let mut norm_b_sq = 0.0f64;

    let mut ia = 0usize;
    let mut ib = 0usize;

    while ia < indices_a.len() && ib < indices_b.len() {
        let ca = indices_a[ia];
        let cb = indices_b[ib];
        if ca == cb {
            dot += values_a[ia] * values_b[ib];
            norm_a_sq += values_a[ia] * values_a[ia];
            norm_b_sq += values_b[ib] * values_b[ib];
            ia += 1;
            ib += 1;
        } else if ca < cb {
            norm_a_sq += values_a[ia] * values_a[ia];
            ia += 1;
        } else {
            norm_b_sq += values_b[ib] * values_b[ib];
            ib += 1;
        }
    }
    // Remaining entries.
    while ia < indices_a.len() {
        norm_a_sq += values_a[ia] * values_a[ia];
        ia += 1;
    }
    while ib < indices_b.len() {
        norm_b_sq += values_b[ib] * values_b[ib];
        ib += 1;
    }

    let denom = (norm_a_sq * norm_b_sq).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use spectral_types::sparse::CsrMatrix;

    // -- helpers ----------------------------------------------------------

    /// Build a small CSR matrix from dense row-major data.
    fn csr_from_dense(rows: usize, cols: usize, data: &[f64]) -> CsrMatrix<f64> {
        assert_eq!(data.len(), rows * cols);
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = data[r * cols + c];
                if v != 0.0 {
                    col_ind.push(c);
                    values.push(v);
                }
            }
            row_ptr[r + 1] = col_ind.len();
        }
        CsrMatrix::new(rows, cols, row_ptr, col_ind, values).unwrap()
    }

    /// Construct a small MipInstance for testing.
    fn sample_mip() -> MipInstance {
        // 3×4 matrix:
        //  [ 1  0  2  0 ]
        //  [ 0  3  0  4 ]
        //  [ 5  0  0  6 ]
        let matrix = csr_from_dense(
            3,
            4,
            &[
                1.0, 0.0, 2.0, 0.0,
                0.0, 3.0, 0.0, 4.0,
                5.0, 0.0, 0.0, 6.0,
            ],
        );
        MipInstance {
            name: "test".to_string(),
            num_variables: 4,
            num_constraints: 3,
            constraint_matrix: matrix,
            objective: vec![1.0, -2.0, 3.0, -4.0],
            rhs: vec![10.0, 20.0, 30.0],
            senses: vec![ConstraintSense::Le, ConstraintSense::Eq, ConstraintSense::Ge],
            var_types: vec![
                VariableType::Binary,
                VariableType::Integer,
                VariableType::Continuous,
                VariableType::Binary,
            ],
            lower_bounds: vec![0.0, 0.0, 0.0, 0.0],
            upper_bounds: vec![1.0, 100.0, f64::INFINITY, 1.0],
            var_names: (0..4).map(|i| format!("x{i}")).collect(),
            con_names: (0..3).map(|i| format!("c{i}")).collect(),
            is_minimization: true,
        }
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // -- test cases -------------------------------------------------------

    #[test]
    fn test_full_mip_instance() {
        let mip = sample_mip();
        let feats = compute_syntactic_features(&mip).unwrap();

        assert_eq!(feats.num_variables, 4.0);
        assert_eq!(feats.num_constraints, 3.0);
        assert_eq!(feats.num_nonzeros, 6.0);
        assert!(approx_eq(feats.density, 6.0 / 12.0, 1e-9));
        assert_eq!(feats.to_vec().len(), SyntacticFeatures::count());
    }

    #[test]
    fn test_matrix_only() {
        let matrix = csr_from_dense(
            2,
            3,
            &[1.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        );
        let feats = compute_syntactic_features_from_matrix(&matrix);

        assert_eq!(feats.num_variables, 3.0);
        assert_eq!(feats.num_constraints, 2.0);
        assert_eq!(feats.num_nonzeros, 3.0);
        // Defaults for matrix-only mode.
        assert_eq!(feats.frac_continuous, 1.0);
        assert_eq!(feats.frac_binary, 0.0);
        assert_eq!(feats.frac_integer, 0.0);
        assert_eq!(feats.frac_inequality, 1.0);
        assert_eq!(feats.frac_equality, 0.0);
        assert_eq!(feats.rhs_range, 0.0);
        assert_eq!(feats.obj_range, 0.0);
        assert_eq!(feats.variable_bound_tightness, 0.0);
    }

    #[test]
    fn test_empty_matrix() {
        let matrix = CsrMatrix::<f64>::zeros(0, 0);
        let mip = MipInstance {
            name: "empty".to_string(),
            num_variables: 0,
            num_constraints: 0,
            constraint_matrix: matrix,
            objective: vec![],
            rhs: vec![],
            senses: vec![],
            var_types: vec![],
            lower_bounds: vec![],
            upper_bounds: vec![],
            var_names: vec![],
            con_names: vec![],
            is_minimization: true,
        };
        let result = compute_syntactic_features(&mip);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_row() {
        let matrix = csr_from_dense(1, 3, &[1.0, 0.0, 2.0]);
        let feats = compute_syntactic_features_from_matrix(&matrix);
        assert_eq!(feats.num_constraints, 1.0);
        assert_eq!(feats.min_row_nnz, 2.0);
        assert_eq!(feats.max_row_nnz, 2.0);
        assert_eq!(feats.avg_row_nnz, 2.0);
        assert_eq!(feats.std_row_nnz, 0.0);
    }

    #[test]
    fn test_single_col() {
        let matrix = csr_from_dense(3, 1, &[1.0, 0.0, 2.0]);
        let feats = compute_syntactic_features_from_matrix(&matrix);
        assert_eq!(feats.num_variables, 1.0);
        assert_eq!(feats.min_col_nnz, 2.0);
        assert_eq!(feats.max_col_nnz, 2.0);
        assert_eq!(feats.avg_col_nnz, 2.0);
        assert_eq!(feats.std_col_nnz, 0.0);
    }

    #[test]
    fn test_coefficient_statistics() {
        // Values: |1|=1, |2|=2, |3|=3, |4|=4, |5|=5, |6|=6
        let mip = sample_mip();
        let feats = compute_syntactic_features(&mip).unwrap();
        assert!(approx_eq(feats.coeff_range, 6.0 - 1.0, 1e-9));
        let expected_mean = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6.0;
        assert!(approx_eq(feats.coeff_mean, expected_mean, 1e-9));
        // Variance = E[X^2] - (E[X])^2
        let e_x2 = (1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0) / 6.0;
        let expected_std = (e_x2 - expected_mean * expected_mean).sqrt();
        assert!(approx_eq(feats.coeff_std, expected_std, 1e-9));
    }

    #[test]
    fn test_row_degree_stats() {
        // Row 0: 2 nnz, Row 1: 2 nnz, Row 2: 2 nnz
        let mip = sample_mip();
        let feats = compute_syntactic_features(&mip).unwrap();
        assert_eq!(feats.min_row_nnz, 2.0);
        assert_eq!(feats.max_row_nnz, 2.0);
        assert_eq!(feats.avg_row_nnz, 2.0);
        assert_eq!(feats.std_row_nnz, 0.0);
    }

    #[test]
    fn test_col_degree_stats() {
        // Col 0: rows 0,2 → 2 nnz
        // Col 1: row 1    → 1 nnz
        // Col 2: row 0    → 1 nnz
        // Col 3: rows 1,2 → 2 nnz
        let mip = sample_mip();
        let feats = compute_syntactic_features(&mip).unwrap();
        assert_eq!(feats.min_col_nnz, 1.0);
        assert_eq!(feats.max_col_nnz, 2.0);
        let expected_avg = (2.0 + 1.0 + 1.0 + 2.0) / 4.0;
        assert!(approx_eq(feats.avg_col_nnz, expected_avg, 1e-9));
    }

    #[test]
    fn test_variable_type_fractions() {
        let types = vec![
            VariableType::Binary,
            VariableType::Binary,
            VariableType::Integer,
            VariableType::Continuous,
            VariableType::Continuous,
        ];
        let (b, i, c) = compute_variable_type_fractions(&types);
        assert!(approx_eq(b, 2.0 / 5.0, 1e-9));
        assert!(approx_eq(i, 1.0 / 5.0, 1e-9));
        assert!(approx_eq(c, 2.0 / 5.0, 1e-9));
    }

    #[test]
    fn test_bound_tightness() {
        let lower = vec![0.0, 0.0, 0.0];
        let upper = vec![1.0, f64::INFINITY, 100.0];
        let tightness = compute_bound_tightness(&lower, &upper);
        // 2 out of 3 upper bounds are finite and < f64::MAX.
        assert!(approx_eq(tightness, 2.0 / 3.0, 1e-9));
    }

    #[test]
    fn test_row_similarity_identical_rows() {
        // All rows identical → high similarity.
        let matrix = csr_from_dense(
            4,
            3,
            &[
                1.0, 2.0, 3.0,
                1.0, 2.0, 3.0,
                1.0, 2.0, 3.0,
                1.0, 2.0, 3.0,
            ],
        );
        let sim = estimate_row_similarity(&matrix, 100);
        assert!(sim >= 0.99, "Expected high similarity, got {sim}");
    }

    #[test]
    fn test_row_similarity_distinct_rows() {
        // Orthogonal rows → zero similarity.
        let matrix = csr_from_dense(
            3,
            3,
            &[
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        );
        let sim = estimate_row_similarity(&matrix, 100);
        assert!(
            approx_eq(sim, 0.0, 1e-9),
            "Expected zero similarity, got {sim}"
        );
    }
}
