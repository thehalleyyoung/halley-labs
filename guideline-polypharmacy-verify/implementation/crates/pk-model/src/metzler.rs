//! Metzler matrix operations for pharmacokinetic systems.
//!
//! A Metzler matrix has non-negative off-diagonal elements and generates
//! monotone dynamical systems, critical for compartmental PK models.

use serde::{Deserialize, Serialize};
use nalgebra::{DMatrix, DVector};
use guardpharma_types::error::PkModelError;

// ---------------------------------------------------------------------------
// MetzlerMatrix
// ---------------------------------------------------------------------------

/// A Metzler matrix (off-diagonal elements >= 0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetzlerMatrix {
    data: Vec<Vec<f64>>,
    dim: usize,
}

impl MetzlerMatrix {
    /// Create a new MetzlerMatrix, validating the Metzler property.
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self, PkModelError> {
        let dim = data.len();
        if dim == 0 {
            return Err(PkModelError::InvalidCompartmentModel(
                "Matrix dimension must be > 0".into(),
            ));
        }
        for row in &data {
            if row.len() != dim {
                return Err(PkModelError::InvalidCompartmentModel(
                    "Matrix must be square".into(),
                ));
            }
        }
        let m = Self { data, dim };
        if !m.is_metzler() {
            return Err(PkModelError::InvalidCompartmentModel(
                "Off-diagonal elements must be >= 0 for Metzler matrix".into(),
            ));
        }
        Ok(m)
    }

    /// Create from a nalgebra DMatrix.
    pub fn from_nalgebra(m: &DMatrix<f64>) -> Result<Self, PkModelError> {
        let dim = m.nrows();
        if m.ncols() != dim {
            return Err(PkModelError::InvalidCompartmentModel(
                "Matrix must be square".into(),
            ));
        }
        let data: Vec<Vec<f64>> = (0..dim)
            .map(|i| (0..dim).map(|j| m[(i, j)]).collect())
            .collect();
        Self::new(data)
    }

    /// Create an identity-scaled Metzler matrix: -alpha * I.
    pub fn diagonal(diag: &[f64]) -> Result<Self, PkModelError> {
        let dim = diag.len();
        let mut data = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            data[i][i] = diag[i];
        }
        Self::new(data)
    }

    /// Create a zero matrix.
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![vec![0.0; dim]; dim],
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, val: f64) -> Result<(), PkModelError> {
        if i != j && val < 0.0 {
            return Err(PkModelError::InvalidCompartmentModel(
                format!("Off-diagonal element ({},{}) must be >= 0, got {}", i, j, val),
            ));
        }
        self.data[i][j] = val;
        Ok(())
    }

    pub fn to_nalgebra(&self) -> DMatrix<f64> {
        DMatrix::from_fn(self.dim, self.dim, |i, j| self.data[i][j])
    }

    /// Check Metzler property: all off-diagonal >= 0.
    pub fn is_metzler(&self) -> bool {
        for i in 0..self.dim {
            for j in 0..self.dim {
                if i != j && self.data[i][j] < -1e-15 {
                    return false;
                }
            }
        }
        true
    }

    /// Check stability: all eigenvalues have negative real part.
    pub fn is_stable(&self) -> bool {
        let eigs = self.eigenvalues();
        eigs.iter().all(|(re, _im)| *re < 1e-10)
    }

    /// Compute eigenvalues as (real, imaginary) pairs.
    pub fn eigenvalues(&self) -> Vec<(f64, f64)> {
        let m = self.to_nalgebra();
        // Use Schur decomposition for general eigenvalues
        if let Some(schur) = m.clone().schur().eigenvalues() {
            schur.iter().map(|&v| (v, 0.0)).collect()
        } else {
            // Fallback: symmetric eigendecomposition (real eigenvalues)
            let eig = m.symmetric_eigen();
            eig.eigenvalues.iter().map(|&v| (v, 0.0)).collect()
        }
    }

    /// Compute spectral radius: max|eigenvalue|.
    pub fn spectral_radius(&self) -> f64 {
        self.eigenvalues()
            .iter()
            .map(|(re, im)| (re * re + im * im).sqrt())
            .fold(0.0_f64, f64::max)
    }

    /// Compute matrix exponential exp(M*t) using scaling-and-squaring with Padé(6).
    pub fn matrix_exponential(&self, t: f64) -> DMatrix<f64> {
        let n = self.dim;
        let a = self.to_nalgebra() * t;
        let norm = matrix_norm_inf(&a);

        // Determine number of squarings
        let s = if norm > 0.5 {
            (norm.log2().ceil() as u32).max(1)
        } else {
            0
        };
        let scale = 2.0_f64.powi(-(s as i32));
        let a_scaled = &a * scale;

        // Padé(6) approximation: exp(A) ≈ D^{-1} * N
        let i_n = DMatrix::<f64>::identity(n, n);
        let a2 = &a_scaled * &a_scaled;
        let a3 = &a2 * &a_scaled;
        let a4 = &a2 * &a2;
        let a5 = &a4 * &a_scaled;
        let a6 = &a3 * &a3;

        let b = [1.0, 1.0 / 2.0, 1.0 / 9.0, 1.0 / 72.0, 1.0 / 1008.0, 1.0 / 30240.0, 1.0 / 1209600.0];

        let u = &i_n * b[0] + &a2 * b[2] + &a4 * b[4] + &a6 * b[6];
        let v = &a_scaled * b[1] + &a3 * b[3] + &a5 * b[5];

        let numer = &u + &v;
        let denom = &u - &v;

        let result = if let Some(denom_inv) = denom.try_inverse() {
            denom_inv * numer
        } else {
            // Fallback: Taylor series
            let mut exp_a = i_n.clone();
            let mut term = i_n.clone();
            for k in 1..20 {
                term = &term * &a_scaled / k as f64;
                exp_a += &term;
            }
            exp_a
        };

        // Squaring phase
        scale_and_square(&result, s)
    }

    /// Compute steady-state solution: x = -M^{-1} * b.
    pub fn steady_state_solution(&self, b: &DVector<f64>) -> Option<DVector<f64>> {
        let m = self.to_nalgebra();
        m.try_inverse().map(|mi| -mi * b)
    }

    /// Check componentwise ordering (Metzler property guarantees monotonicity).
    pub fn componentwise_ordering(&self) -> bool {
        self.is_metzler()
    }

    /// Check column sums (<= 0 for compartmental conservation).
    pub fn column_sum_check(&self) -> bool {
        for j in 0..self.dim {
            let sum: f64 = (0..self.dim).map(|i| self.data[i][j]).sum();
            if sum > 1e-10 {
                return false;
            }
        }
        true
    }

    /// Row sums (often <= 0 for compartmental models).
    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.dim)
            .map(|i| self.data[i].iter().sum())
            .collect()
    }
}

/// Construct a Metzler matrix from PK parameters.
pub fn from_pk_parameters(
    clearances: &[f64],
    volumes: &[f64],
    distribution_clearances: &[(usize, usize, f64)],
    interaction_factors: &[(usize, usize, f64)],
) -> Result<MetzlerMatrix, PkModelError> {
    let n = clearances.len();
    if volumes.len() != n {
        return Err(PkModelError::InvalidCompartmentModel(
            "Clearances and volumes must have same length".into(),
        ));
    }
    let mut data = vec![vec![0.0; n]; n];

    // Elimination rates on diagonal
    for i in 0..n {
        data[i][i] -= clearances[i] / volumes[i];
    }

    // Distribution clearances
    for &(from, to, cld) in distribution_clearances {
        if from >= n || to >= n {
            return Err(PkModelError::InvalidCompartmentModel(
                "Distribution clearance index out of bounds".into(),
            ));
        }
        data[from][from] -= cld / volumes[from];
        data[to][from] += cld / volumes[from]; // off-diagonal positive (Metzler)
        data[to][to] -= cld / volumes[to];
        data[from][to] += cld / volumes[to];
    }

    // Apply interaction factors (modify diagonal elimination)
    for &(i, j, factor) in interaction_factors {
        if i >= n || j >= n {
            continue;
        }
        // Drug j's effect on drug i's clearance
        data[i][i] *= factor;
    }

    MetzlerMatrix::new(data)
}

/// Infinity norm of a matrix.
fn matrix_norm_inf(m: &DMatrix<f64>) -> f64 {
    let mut max_sum = 0.0_f64;
    for i in 0..m.nrows() {
        let row_sum: f64 = (0..m.ncols()).map(|j| m[(i, j)].abs()).sum();
        max_sum = max_sum.max(row_sum);
    }
    max_sum
}

/// Repeated squaring: M^{2^s}.
fn scale_and_square(m: &DMatrix<f64>, s: u32) -> DMatrix<f64> {
    let mut result = m.clone();
    for _ in 0..s {
        result = &result * &result;
    }
    result
}

// ---------------------------------------------------------------------------
// MetzlerInterval
// ---------------------------------------------------------------------------

/// Interval Metzler matrix for abstract interpretation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetzlerInterval {
    pub lo_matrix: MetzlerMatrix,
    pub hi_matrix: MetzlerMatrix,
    dim: usize,
}

impl MetzlerInterval {
    pub fn new(lo: MetzlerMatrix, hi: MetzlerMatrix) -> Result<Self, PkModelError> {
        if lo.dim() != hi.dim() {
            return Err(PkModelError::InvalidCompartmentModel(
                "Interval matrices must have same dimension".into(),
            ));
        }
        let dim = lo.dim();
        for i in 0..dim {
            for j in 0..dim {
                if lo.get(i, j) > hi.get(i, j) + 1e-12 {
                    return Err(PkModelError::InvalidCompartmentModel(format!(
                        "lo[{},{}]={} > hi[{},{}]={}",
                        i, j, lo.get(i, j), i, j, hi.get(i, j)
                    )));
                }
            }
        }
        Ok(Self { lo_matrix: lo, hi_matrix: hi, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Interval matrix exponential exploiting Metzler monotonicity.
    /// For Metzler M: if lo <= M <= hi, then exp(lo*t) <= exp(M*t) <= exp(hi*t).
    pub fn interval_matrix_exponential(&self, t: f64) -> (DMatrix<f64>, DMatrix<f64>) {
        let lo_exp = self.lo_matrix.matrix_exponential(t);
        let hi_exp = self.hi_matrix.matrix_exponential(t);
        (lo_exp, hi_exp)
    }

    /// Interval steady-state exploiting monotonicity.
    pub fn interval_steady_state(
        &self,
        b_lo: &DVector<f64>,
        b_hi: &DVector<f64>,
    ) -> (DVector<f64>, DVector<f64>) {
        // For Metzler M < 0 (stable): -M^{-1} is non-negative
        // Monotonicity: more negative M => larger -M^{-1} entries
        // So: lo_ss uses hi_matrix with lo input, hi_ss uses lo_matrix with hi input
        let ss_lo = self.hi_matrix.steady_state_solution(b_lo)
            .unwrap_or_else(|| DVector::zeros(self.dim));
        let ss_hi = self.lo_matrix.steady_state_solution(b_hi)
            .unwrap_or_else(|| DVector::zeros(self.dim));

        // Ensure proper ordering
        let n = self.dim;
        let lo_vec = DVector::from_fn(n, |i, _| ss_lo[i].min(ss_hi[i]).max(0.0));
        let hi_vec = DVector::from_fn(n, |i, _| ss_lo[i].max(ss_hi[i]));
        (lo_vec, hi_vec)
    }

    /// Check if a matrix is contained in this interval.
    pub fn contains_matrix(&self, m: &MetzlerMatrix) -> bool {
        if m.dim() != self.dim {
            return false;
        }
        for i in 0..self.dim {
            for j in 0..self.dim {
                if m.get(i, j) < self.lo_matrix.get(i, j) - 1e-12
                    || m.get(i, j) > self.hi_matrix.get(i, j) + 1e-12
                {
                    return false;
                }
            }
        }
        true
    }

    /// Width: hi - lo elementwise.
    pub fn width(&self) -> DMatrix<f64> {
        let n = self.dim;
        DMatrix::from_fn(n, n, |i, j| {
            self.hi_matrix.get(i, j) - self.lo_matrix.get(i, j)
        })
    }
}

/// Verify componentwise monotonicity: if c0 <= c0' then exp(Mt)*c0 <= exp(Mt)*c0'.
pub fn verify_componentwise_monotonicity(
    m: &MetzlerMatrix,
    c0: &DVector<f64>,
    c0_prime: &DVector<f64>,
    t: f64,
) -> bool {
    let n = m.dim();
    // Check c0 <= c0'
    for i in 0..n {
        if c0[i] > c0_prime[i] + 1e-12 {
            return false;
        }
    }
    let exp_mt = m.matrix_exponential(t);
    let ct = &exp_mt * c0;
    let ct_prime = &exp_mt * c0_prime;
    for i in 0..n {
        if ct[i] > ct_prime[i] + 1e-10 {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2x2_stable() -> MetzlerMatrix {
        // M = [[-2, 1], [1, -3]] — Metzler and stable
        MetzlerMatrix::new(vec![vec![-2.0, 1.0], vec![1.0, -3.0]]).unwrap()
    }

    #[test]
    fn test_metzler_validation_valid() {
        let m = make_2x2_stable();
        assert!(m.is_metzler());
    }

    #[test]
    fn test_metzler_validation_invalid() {
        let result = MetzlerMatrix::new(vec![vec![-2.0, -1.0], vec![1.0, -3.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_2x2_stability() {
        let m = make_2x2_stable();
        assert!(m.is_stable());
    }

    #[test]
    fn test_unstable_matrix() {
        // M = [[1, 2], [2, 1]] — has positive eigenvalue
        let m = MetzlerMatrix::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]]).unwrap();
        assert!(!m.is_stable());
    }

    #[test]
    fn test_matrix_exp_identity() {
        let m = MetzlerMatrix::new(vec![vec![-1.0, 0.0], vec![0.0, -1.0]]).unwrap();
        let exp0 = m.matrix_exponential(0.0);
        let id = DMatrix::<f64>::identity(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((exp0[(i, j)] - id[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_exp_diagonal() {
        let m = MetzlerMatrix::new(vec![vec![-1.0, 0.0], vec![0.0, -2.0]]).unwrap();
        let exp1 = m.matrix_exponential(1.0);
        assert!((exp1[(0, 0)] - (-1.0_f64).exp()).abs() < 1e-6);
        assert!((exp1[(1, 1)] - (-2.0_f64).exp()).abs() < 1e-6);
        assert!(exp1[(0, 1)].abs() < 1e-10);
    }

    #[test]
    fn test_steady_state_simple() {
        // M = [[-2]], b = [4] => x = -(-2)^{-1}*4 = 2
        let m = MetzlerMatrix::new(vec![vec![-2.0]]).unwrap();
        let b = DVector::from_vec(vec![4.0]);
        let ss = m.steady_state_solution(&b).unwrap();
        assert!((ss[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_radius() {
        let m = MetzlerMatrix::new(vec![vec![-3.0, 1.0], vec![1.0, -3.0]]).unwrap();
        let sr = m.spectral_radius();
        // Eigenvalues: -2, -4. Spectral radius = 4
        assert!((sr - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_from_pk_parameters() {
        let m = from_pk_parameters(
            &[10.0, 5.0],
            &[50.0, 100.0],
            &[(0, 1, 3.0)],
            &[],
        )
        .unwrap();
        assert!(m.is_metzler());
        assert_eq!(m.dim(), 2);
    }

    #[test]
    fn test_interval_monotonicity() {
        let lo = MetzlerMatrix::new(vec![vec![-3.0, 0.5], vec![0.5, -4.0]]).unwrap();
        let hi = MetzlerMatrix::new(vec![vec![-2.0, 1.5], vec![1.5, -3.0]]).unwrap();
        let interval = MetzlerInterval::new(lo.clone(), hi.clone()).unwrap();

        let mid = MetzlerMatrix::new(vec![vec![-2.5, 1.0], vec![1.0, -3.5]]).unwrap();
        assert!(interval.contains_matrix(&mid));
        assert!(!interval.contains_matrix(
            &MetzlerMatrix::new(vec![vec![-1.0, 2.0], vec![2.0, -2.0]]).unwrap()
        ));
    }

    #[test]
    fn test_componentwise_monotonicity() {
        let m = make_2x2_stable();
        let c0 = DVector::from_vec(vec![1.0, 1.0]);
        let c0p = DVector::from_vec(vec![2.0, 2.0]);
        assert!(verify_componentwise_monotonicity(&m, &c0, &c0p, 1.0));
    }

    #[test]
    fn test_column_sums() {
        // A proper compartmental model: columns sum to <= 0
        let m = MetzlerMatrix::new(vec![
            vec![-2.0, 1.0],
            vec![1.0, -3.0],
        ])
        .unwrap();
        // Column 0: -2+1 = -1, Column 1: 1-3 = -2 => both <= 0
        assert!(m.column_sum_check());
    }

    #[test]
    fn test_zeros() {
        let m = MetzlerMatrix::zeros(3);
        assert_eq!(m.dim(), 3);
        assert!(m.is_metzler());
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_interval_steady_state() {
        let lo = MetzlerMatrix::new(vec![vec![-3.0]]).unwrap();
        let hi = MetzlerMatrix::new(vec![vec![-2.0]]).unwrap();
        let interval = MetzlerInterval::new(lo, hi).unwrap();
        let b_lo = DVector::from_vec(vec![4.0]);
        let b_hi = DVector::from_vec(vec![6.0]);
        let (ss_lo, ss_hi) = interval.interval_steady_state(&b_lo, &b_hi);
        // Exact range: [4/3, 6/2] = [1.333, 3.0]
        assert!(ss_lo[0] > 0.0);
        assert!(ss_hi[0] > ss_lo[0]);
    }
}
